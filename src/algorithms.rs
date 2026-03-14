// algorithms.rs — All matrix multiplication algorithms and comparison.
//
// Order of implementation follows verification chain:
// 1. multiply_naive  — Ground Truth (i-k-j order for cache locality)
// 2. multiply_tiled  — Cache-blocked version (6 nested loops)
// 3. multiply_tiled_parallel — rayon parallelization over rows
// 4. multiply_strassen — Recursive Strassen with rayon::join
// 5. compare_matrices — Parallel element-wise comparison

use crate::common::{
    ComparisonResult, ProgressHandle, QuadrantStats, ScientificComparisonResult,
    SimdLevel, VvAssessment, DEFAULT_TILE_SIZE,
};
use crate::matrix::Matrix;
use rayon::prelude::*;

// ─── multiply_naive (Ground Truth) ──────────────────────────────────────────
//
// LOOP ORDER: i-k-j (LAW 5)
//
// Mathematically: C[i][j] = Σ_k A[i][k] * B[k][j]
//
// The naive i-j-k order accesses B column-wise: for fixed j, varying k
// means B[k][j] jumps by B.cols elements between accesses. On a 1024×1024
// matrix that's 8192 bytes between accesses → guaranteed cache misses.
//
// The i-k-j order fixes k in the middle loop:
//   - a[i][k] is a constant (hoisted out of inner loop)
//   - b[k][j] with growing j is sequential row access → cache hits
//   - result[i][j] accumulates in the same cache line
//
// This single reordering typically gives 3-8× speedup on large matrices.

pub fn multiply_naive(a: &Matrix, b: &Matrix) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    // Single bounds check before the unsafe zone — no branching inside
    assert_eq!(a.cols(), b.rows(), "A.cols ({}) != B.rows ({})", a.cols(), b.rows());

    let mut result = Matrix::zeros(m, p).expect("Failed to allocate result matrix");

    // SAFETY: We verified dimensions above. All indices are within bounds:
    // i < m, k < n, j < p. Inside: pure arithmetic only, no branches.
    unsafe {
        for i in 0..m {
            for k in 0..n {
                let a_ik = a.get_unchecked(i, k);
                for j in 0..p {
                    let prev = result.get_unchecked(i, j);
                    result.set_unchecked(i, j, prev + a_ik * b.get_unchecked(k, j));
                }
            }
        }
    }

    result
}

// ─── multiply_naive_with_progress ────────────────────────────────────────────
//
// Same as multiply_naive but reports row-level progress via ProgressHandle.
// Reports every 8 rows to avoid atomic write overhead dominating small inner loops.

pub fn multiply_naive_with_progress(
    a: &Matrix,
    b: &Matrix,
    progress: &ProgressHandle,
    offset: u32,
    scale: u32,
) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    assert_eq!(a.cols(), b.rows(), "A.cols ({}) != B.rows ({})", a.cols(), b.rows());

    let mut result = Matrix::zeros(m, p).expect("Failed to allocate result matrix");

    unsafe {
        for i in 0..m {
            for k in 0..n {
                let a_ik = a.get_unchecked(i, k);
                for j in 0..p {
                    let prev = result.get_unchecked(i, j);
                    result.set_unchecked(i, j, prev + a_ik * b.get_unchecked(k, j));
                }
            }
            if i % 8 == 0 || i == m - 1 {
                let pct = offset + ((i as u64 + 1) * scale as u64 / m as u64) as u32;
                progress.set(pct.min(offset + scale), offset + scale);
            }
        }
    }

    result
}

// ─── multiply_tiled (Cache-Blocked) ─────────────────────────────────────────
//
// PROBLEM: For 1024×1024 matrices, B doesn't fit in L1 cache (32KB).
// When traversing B[k][j] for fixed k, data gets evicted before next row.
//
// SOLUTION — Tiling:
// Divide matrices into tile_size × tile_size blocks.
// Three tiles (A_tile, B_tile, C_tile) coexist in cache simultaneously:
//   3 × T × T × 8 bytes ≤ cache_size
//   T=64: 3 × 64 × 64 × 8 = 96KB (fits L2 256KB)
//   T=32: 3 × 32 × 32 × 8 = 24KB (fits L1 32KB)
//
// 6 nested loops: outer 3 iterate over blocks, inner 3 over elements.
// Outer order: (ib, kb, jb) — k-blocks first for accumulation.
// Inner order: i-k-j (cache-friendly, same as naive).

pub fn multiply_tiled(a: &Matrix, b: &Matrix, tile_size: usize) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();
    let ts = if tile_size == 0 { DEFAULT_TILE_SIZE } else { tile_size };

    assert_eq!(a.cols(), b.rows(), "A.cols ({}) != B.rows ({})", a.cols(), b.rows());

    let mut result = Matrix::zeros(m, p).expect("Failed to allocate result matrix");

    // SAFETY: All indices bounded by m, n, p which match matrix dimensions.
    unsafe {
        for ib in (0..m).step_by(ts) {
            for kb in (0..n).step_by(ts) {
                for jb in (0..p).step_by(ts) {
                    let i_end = (ib + ts).min(m);
                    let k_end = (kb + ts).min(n);
                    let j_end = (jb + ts).min(p);

                    for i in ib..i_end {
                        for k in kb..k_end {
                            let a_ik = a.get_unchecked(i, k);
                            for j in jb..j_end {
                                let prev = result.get_unchecked(i, j);
                                result.set_unchecked(i, j, prev + a_ik * b.get_unchecked(k, j));
                            }
                        }
                    }
                }
            }
        }
    }

    result
}

// ─── multiply_tiled_parallel ────────────────────────────────────────────────
//
// Parallelization via rayon at the row level of the result matrix.
// Each thread computes its own rows of C independently — no data races.
//
// Pattern: par_chunks_mut splits result.data into non-overlapping &mut [f64]
// slices of size p (one row each). Each thread does tiled multiplication
// for its assigned rows against the full A and B matrices.

pub fn multiply_tiled_parallel(a: &Matrix, b: &Matrix, tile_size: usize) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();
    let ts = if tile_size == 0 { DEFAULT_TILE_SIZE } else { tile_size };

    assert_eq!(a.cols(), b.rows(), "A.cols ({}) != B.rows ({})", a.cols(), b.rows());

    let mut result = Matrix::zeros(m, p).expect("Failed to allocate result matrix");

    // Each chunk is exactly one row (p elements). rayon distributes rows across threads.
    // SAFETY: Each thread writes only to its own row slice — no overlap.
    result.data.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
        unsafe {
            for kb in (0..n).step_by(ts) {
                for jb in (0..p).step_by(ts) {
                    let k_end = (kb + ts).min(n);
                    let j_end = (jb + ts).min(p);

                    for k in kb..k_end {
                        let a_ik = a.get_unchecked(i, k);
                        for j in jb..j_end {
                            *row.get_unchecked_mut(j) += a_ik * b.get_unchecked(k, j);
                        }
                    }
                }
            }
        }
    });

    result
}

// ─── multiply_strassen (Parallel Strassen with rayon::join) ─────────────────
//
// STRASSEN'S ALGORITHM reduces matrix multiplication from O(n³) to O(n^2.807).
//
// Standard 2×2 block multiplication: 8 matrix multiplications, 4 additions.
// Strassen: 7 multiplications, 18 additions/subtractions.
//
// For n=4096: theoretical speedup ≈ n^0.193 ≈ 5.4×.
// In practice: 3-8× with overhead from temporary matrices.
//
// The 7 products (from quadrants A11,A12,A21,A22, B11,B12,B21,B22):
//   M1 = (A11 + A22) × (B11 + B22)
//   M2 = (A21 + A22) × B11
//   M3 = A11 × (B12 - B22)
//   M4 = A22 × (B21 - B11)
//   M5 = (A11 + A12) × B22
//   M6 = (A21 - A11) × (B11 + B12)
//   M7 = (A12 - A22) × (B21 + B22)
//
// Result quadrants:
//   C11 = M1 + M4 - M5 + M7
//   C12 = M3 + M5
//   C21 = M2 + M4
//   C22 = M1 - M2 + M3 + M6
//
// PARALLELIZATION: rayon::join splits the 7 recursive calls into a binary
// tree of parallel tasks. Work-stealing ensures optimal load balancing.
//
// CLONE ANALYSIS (minimize allocations):
//   a11: used in M1, M3, M5, M6 → 3 clones needed (last use is owned)
//   a12: used in M5, M7 → 1 clone
//   a21: used in M2, M6 → 1 clone
//   a22: used in M1, M2, M4, M7 → 3 clones
//   b11: used in M1, M2, M6 → 2 clones
//   b12: used in M3, M6 → 1 clone
//   b21: used in M4, M7 → 1 clone
//   b22: used in M1, M3, M5, M7 → 3 clones

pub fn multiply_strassen(
    a: Matrix,
    b: Matrix,
    threshold: usize,
    _simd: SimdLevel, // reserved for future SIMD kernel dispatch
) -> Matrix {
    let n = a.rows();

    // BASE CASE: switch to tiled multiplication (single-threaded).
    // Strassen overhead (18 matrix add/sub + 7 temp matrices) exceeds gain for small n.
    if n <= threshold {
        return multiply_tiled(&a, &b, DEFAULT_TILE_SIZE);
    }

    // Split both matrices into quadrants
    let (a11, a12, a21, a22) = a.split_4();
    let (b11, b12, b21, b22) = b.split_4();

    // PRE-COMPUTE all sum/diff operands BEFORE the parallel section.
    // This resolves borrow-checker conflicts: rayon::join closures each get
    // owned Matrices with no shared borrows across closure boundaries.
    let s1  = &a11 + &a22; // for M1 left
    let s2  = &b11 + &b22; // for M1 right
    let s3  = &a21 + &a22; // for M2 left
    let s4  = &b12 - &b22; // for M3 right
    let s5  = &b21 - &b11; // for M4 right
    let s6  = &a11 + &a12; // for M5 left
    let s7  = &a21 - &a11; // for M6 left
    let s8  = &b11 + &b12; // for M6 right
    let s9  = &a12 - &a22; // for M7 left
    let s10 = &b21 + &b22; // for M7 right

    // Now all borrows are resolved. a11, a22, b11, b22 are still owned
    // and can be moved into closures that need standalone quadrants.

    // Compute 7 Strassen products in parallel via nested rayon::join.
    let ((m1, m2), (m3, (m4, (m5, (m6, m7))))) = rayon::join(
        || {
            rayon::join(
                // M1 = (A11 + A22) × (B11 + B22)
                || multiply_strassen(s1, s2, threshold, _simd),
                // M2 = (A21 + A22) × B11
                || multiply_strassen(s3, b11, threshold, _simd),
            )
        },
        || {
            rayon::join(
                // M3 = A11 × (B12 - B22)
                || multiply_strassen(a11, s4, threshold, _simd),
                || {
                    rayon::join(
                        // M4 = A22 × (B21 - B11)
                        || multiply_strassen(a22, s5, threshold, _simd),
                        || {
                            rayon::join(
                                // M5 = (A11 + A12) × B22
                                || multiply_strassen(s6, b22, threshold, _simd),
                                || {
                                    rayon::join(
                                        // M6 = (A21 - A11) × (B11 + B12)
                                        || multiply_strassen(s7, s8, threshold, _simd),
                                        // M7 = (A12 - A22) × (B21 + B22)
                                        || multiply_strassen(s9, s10, threshold, _simd),
                                    )
                                },
                            )
                        },
                    )
                },
            )
        },
    );

    // Combine result quadrants:
    //   C11 = M1 + M4 - M5 + M7
    //   C12 = M3 + M5
    //   C21 = M2 + M4
    //   C22 = M1 - M2 + M3 + M6
    let c11 = &(&m1 + &m4) - &(&m5 - &m7);
    let c12 = &m3 + &m5;
    let c21 = &m2 + &m4;
    let c22 = &(&m1 - &m2) + &(&m3 + &m6);

    Matrix::combine_4(c11, c12, c21, c22)
}

/// Public wrapper: handles padding to power-of-2 and unpadding.
/// Returns (result, padding_ms, unpadding_ms).
pub fn multiply_strassen_padded(
    a: &Matrix,
    b: &Matrix,
    threshold: usize,
    simd: SimdLevel,
) -> (Matrix, f64, f64) {
    let orig_rows = a.rows();
    let orig_cols = b.cols();

    // Pad both matrices to power-of-2 square dimensions for recursive splitting
    let (pad_a, padding_ms) = {
        let start = std::time::Instant::now();
        let (pa, _, _) = a.pad_to_power_of_2();
        let (pb, _, _) = b.pad_to_power_of_2();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        ((pa, pb), ms)
    };

    let result_padded = multiply_strassen(pad_a.0, pad_a.1, threshold, simd);

    let start = std::time::Instant::now();
    let result = result_padded.unpad(orig_rows, orig_cols);
    let unpadding_ms = start.elapsed().as_secs_f64() * 1000.0;

    (result, padding_ms, unpadding_ms)
}

// ─── multiply_winograd (Winograd's variant of Strassen) ─────────────────────
//
// WINOGRAD'S VARIANT reduces matrix additions from 18 (standard Strassen) to 15.
// Same 7 recursive multiplications, same O(n^2.807) complexity.
// The saving comes from reusing intermediate sums more efficiently.
//
// Pre-computations (8 matrix add/sub instead of 10):
//   S1 = A21 + A22       T1 = B12 - B11
//   S2 = S1 - A11        T2 = B22 - T1
//   S3 = A11 - A21       T3 = B22 - B12
//   S4 = A12 - S2        T4 = T2 - B21
//
// 7 products:
//   P1 = A11 × B11       P2 = A12 × B21
//   P3 = S4 × B22        P4 = A22 × T4
//   P5 = S1 × T1         P6 = S2 × T2
//   P7 = S3 × T3
//
// Assembly (7 additions instead of 8):
//   U2 = P1 + P6
//   U3 = U2 + P7
//   U4 = U2 + P5
//   C11 = P1 + P2
//   C12 = U4 + P3
//   C21 = U3 - P4
//   C22 = U3 + P5

pub fn multiply_winograd(
    a: Matrix,
    b: Matrix,
    threshold: usize,
    _simd: SimdLevel,
) -> Matrix {
    let n = a.rows();

    // BASE CASE: switch to tiled multiplication.
    if n <= threshold {
        return multiply_tiled(&a, &b, DEFAULT_TILE_SIZE);
    }

    // Split both matrices into quadrants
    let (a11, a12, a21, a22) = a.split_4();
    let (b11, b12, b21, b22) = b.split_4();

    // PRE-COMPUTE S and T operands (8 add/sub vs Strassen's 10)
    let s1 = &a21 + &a22;        // S1 = A21 + A22
    let s2 = &s1 - &a11;         // S2 = S1 - A11
    let s3 = &a11 - &a21;        // S3 = A11 - A21
    let s4 = &a12 - &s2;         // S4 = A12 - S2

    let t1 = &b12 - &b11;        // T1 = B12 - B11
    let t2 = &b22 - &t1;         // T2 = B22 - T1
    let t3 = &b22 - &b12;        // T3 = B22 - B12
    let t4 = &t2 - &b21;         // T4 = T2 - B21

    // Compute 7 Winograd products in parallel via nested rayon::join.
    let ((p1, p2), (p3, (p4, (p5, (p6, p7))))) = rayon::join(
        || {
            rayon::join(
                // P1 = A11 × B11
                || multiply_winograd(a11, b11, threshold, _simd),
                // P2 = A12 × B21
                || multiply_winograd(a12, b21, threshold, _simd),
            )
        },
        || {
            rayon::join(
                // P3 = S4 × B22
                || multiply_winograd(s4, b22, threshold, _simd),
                || {
                    rayon::join(
                        // P4 = A22 × T4
                        || multiply_winograd(a22, t4, threshold, _simd),
                        || {
                            rayon::join(
                                // P5 = S1 × T1
                                || multiply_winograd(s1, t1, threshold, _simd),
                                || {
                                    rayon::join(
                                        // P6 = S2 × T2
                                        || multiply_winograd(s2, t2, threshold, _simd),
                                        // P7 = S3 × T3
                                        || multiply_winograd(s3, t3, threshold, _simd),
                                    )
                                },
                            )
                        },
                    )
                },
            )
        },
    );

    // Assembly (7 add/sub instead of Strassen's 8):
    let u2 = &p1 + &p6;          // U2 = P1 + P6
    let u3 = &u2 + &p7;          // U3 = U2 + P7
    let u4 = &u2 + &p5;          // U4 = U2 + P5

    let c11 = &p1 + &p2;         // C11 = P1 + P2
    let c12 = &u4 + &p3;         // C12 = U4 + P3
    let c21 = &u3 - &p4;         // C21 = U3 - P4
    let c22 = &u3 + &p5;         // C22 = U3 + P5

    Matrix::combine_4(c11, c12, c21, c22)
}

/// Public wrapper: handles padding to power-of-2 and unpadding.
/// Returns (result, padding_ms, unpadding_ms).
pub fn multiply_winograd_padded(
    a: &Matrix,
    b: &Matrix,
    threshold: usize,
    simd: SimdLevel,
) -> (Matrix, f64, f64) {
    let orig_rows = a.rows();
    let orig_cols = b.cols();

    let (pad_a, padding_ms) = {
        let start = std::time::Instant::now();
        let (pa, _, _) = a.pad_to_power_of_2();
        let (pb, _, _) = b.pad_to_power_of_2();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        ((pa, pb), ms)
    };

    let result_padded = multiply_winograd(pad_a.0, pad_a.1, threshold, simd);

    let start = std::time::Instant::now();
    let result = result_padded.unpad(orig_rows, orig_cols);
    let unpadding_ms = start.elapsed().as_secs_f64() * 1000.0;

    (result, padding_ms, unpadding_ms)
}

// ─── compare_matrices (Parallel) ────────────────────────────────────────────
//
// Element-wise comparison using rayon's parallel fold+reduce.
// Computes max_abs_diff, avg_abs_diff, rms_diff, and match count.
//
// NOTE: f64 is not Ord, so we use f64::max() instead of std::cmp::max.

pub fn compare_matrices(a: &Matrix, b: &Matrix, epsilon: f64) -> ComparisonResult {
    assert_eq!(
        (a.rows(), a.cols()),
        (b.rows(), b.cols()),
        "Cannot compare {}×{} with {}×{}",
        a.rows(), a.cols(), b.rows(), b.cols()
    );

    let start = std::time::Instant::now();
    let total = a.data().len();

    // Parallel fold: each chunk produces (max_diff, sum_diff, sum_sq_diff, match_count)
    let (max_diff, sum_diff, sum_sq, matches) = a
        .data()
        .par_iter()
        .zip(b.data().par_iter())
        .fold(
            || (0.0_f64, 0.0_f64, 0.0_f64, 0_usize),
            |(mx, sm, sq, mc), (va, vb)| {
                let diff = (va - vb).abs();
                (
                    f64::max(mx, diff),
                    sm + diff,
                    sq + diff * diff,
                    mc + if diff < epsilon { 1 } else { 0 },
                )
            },
        )
        .reduce(
            || (0.0_f64, 0.0_f64, 0.0_f64, 0_usize),
            |(mx1, sm1, sq1, mc1), (mx2, sm2, sq2, mc2)| {
                (f64::max(mx1, mx2), sm1 + sm2, sq1 + sq2, mc1 + mc2)
            },
        );

    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let n = total as f64;

    ComparisonResult {
        max_abs_diff: max_diff,
        avg_abs_diff: if total > 0 { sum_diff / n } else { 0.0 },
        rms_diff: if total > 0 { (sum_sq / n).sqrt() } else { 0.0 },
        match_count: matches,
        total_count: total,
        is_equal: matches == total,
        time_ms,
    }
}

// ─── compare_matrices_scientific (V&V-grade) ────────────────────────────────
//
// Full scientific comparison: Frobenius norms, relative error, quadrant analysis,
// sparsity, sign changes, V&V assessment. Single parallel pass over all elements
// with per-quadrant accumulators for spatial error analysis.

pub fn compare_matrices_scientific(
    a: &Matrix,
    b: &Matrix,
    epsilon: f64,
) -> ScientificComparisonResult {
    assert_eq!(
        (a.rows(), a.cols()),
        (b.rows(), b.cols()),
        "Cannot compare {}×{} with {}×{}",
        a.rows(), a.cols(), b.rows(), b.cols()
    );

    let start = std::time::Instant::now();
    let rows = a.rows();
    let cols = a.cols();
    let total = rows * cols;
    let mid_r = rows / 2;
    let mid_c = cols / 2;

    // Accumulator for the parallel fold: global stats + 4 quadrant stats
    // Quadrant layout: 0=TL, 1=TR, 2=BL, 3=BR
    #[derive(Clone)]
    struct Acc {
        // Global
        max_diff: f64,
        sum_abs_diff: f64,
        sum_sq_diff: f64,
        sum_sq_a: f64,
        sum_sq_b: f64,
        exact_matches: usize,
        eps_matches: usize,
        sign_changes: usize,
        zeros_a: usize,
        zeros_b: usize,
        struct_zeros: usize,
        // Per-quadrant: [max_diff, sum_abs_diff, sum_sq_diff, count, eps_matches]
        q_max: [f64; 4],
        q_sum: [f64; 4],
        q_sq: [f64; 4],
        q_count: [usize; 4],
        q_eps: [usize; 4],
    }

    impl Acc {
        fn new() -> Self {
            Acc {
                max_diff: 0.0, sum_abs_diff: 0.0, sum_sq_diff: 0.0,
                sum_sq_a: 0.0, sum_sq_b: 0.0,
                exact_matches: 0, eps_matches: 0, sign_changes: 0,
                zeros_a: 0, zeros_b: 0, struct_zeros: 0,
                q_max: [0.0; 4], q_sum: [0.0; 4], q_sq: [0.0; 4],
                q_count: [0; 4], q_eps: [0; 4],
            }
        }

        fn merge(mut self, other: Acc) -> Acc {
            self.max_diff = f64::max(self.max_diff, other.max_diff);
            self.sum_abs_diff += other.sum_abs_diff;
            self.sum_sq_diff += other.sum_sq_diff;
            self.sum_sq_a += other.sum_sq_a;
            self.sum_sq_b += other.sum_sq_b;
            self.exact_matches += other.exact_matches;
            self.eps_matches += other.eps_matches;
            self.sign_changes += other.sign_changes;
            self.zeros_a += other.zeros_a;
            self.zeros_b += other.zeros_b;
            self.struct_zeros += other.struct_zeros;
            for i in 0..4 {
                self.q_max[i] = f64::max(self.q_max[i], other.q_max[i]);
                self.q_sum[i] += other.q_sum[i];
                self.q_sq[i] += other.q_sq[i];
                self.q_count[i] += other.q_count[i];
                self.q_eps[i] += other.q_eps[i];
            }
            self
        }
    }

    let a_data = a.data();
    let b_data = b.data();

    // Single parallel pass: enumerate flat indices, compute all metrics
    let acc = (0..total).into_par_iter().fold(Acc::new, |mut acc, idx| {
        let av = a_data[idx];
        let bv = b_data[idx];
        let diff = (av - bv).abs();

        // Global stats
        acc.max_diff = f64::max(acc.max_diff, diff);
        acc.sum_abs_diff += diff;
        acc.sum_sq_diff += diff * diff;
        acc.sum_sq_a += av * av;
        acc.sum_sq_b += bv * bv;
        if diff < f64::EPSILON { acc.exact_matches += 1; }
        if diff < epsilon { acc.eps_matches += 1; }
        if av * bv < 0.0 { acc.sign_changes += 1; }
        if av.abs() < 1e-10 { acc.zeros_a += 1; }
        if bv.abs() < 1e-10 { acc.zeros_b += 1; }
        if av.abs() < 1e-10 && bv.abs() < 1e-10 { acc.struct_zeros += 1; }

        // Quadrant index: TL=0, TR=1, BL=2, BR=3
        let row = idx / cols;
        let col = idx % cols;
        let qi = (if row >= mid_r { 2 } else { 0 }) + (if col >= mid_c { 1 } else { 0 });
        acc.q_max[qi] = f64::max(acc.q_max[qi], diff);
        acc.q_sum[qi] += diff;
        acc.q_sq[qi] += diff * diff;
        acc.q_count[qi] += 1;
        if diff < epsilon { acc.q_eps[qi] += 1; }

        acc
    }).reduce(Acc::new, |a, b| a.merge(b));

    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let n = total as f64;

    let frobenius_norm_diff = acc.sum_sq_diff.sqrt();
    let frobenius_norm_a = acc.sum_sq_a.sqrt();
    let frobenius_norm_b = acc.sum_sq_b.sqrt();
    let relative_error = if frobenius_norm_a > 1e-300 {
        frobenius_norm_diff / frobenius_norm_a
    } else {
        frobenius_norm_diff
    };

    let assessment = VvAssessment::from_relative_error(relative_error, acc.max_diff);

    // Total zeros in A for structural zeros match percentage
    let total_zeros = acc.zeros_a.max(acc.zeros_b).max(1);

    let labels: [&str; 4] = ["Top-Left", "Top-Right", "Bot-Left", "Bot-Right"];
    let quadrants: [QuadrantStats; 4] = std::array::from_fn(|i| {
        let cnt = acc.q_count[i] as f64;
        QuadrantStats {
            label: labels[i],
            max_diff: acc.q_max[i],
            mean_diff: if cnt > 0.0 { acc.q_sum[i] / cnt } else { 0.0 },
            rms_diff: if cnt > 0.0 { (acc.q_sq[i] / cnt).sqrt() } else { 0.0 },
            match_pct: if acc.q_count[i] > 0 {
                acc.q_eps[i] as f64 / cnt * 100.0
            } else {
                100.0
            },
        }
    });

    ScientificComparisonResult {
        rows,
        cols,
        frobenius_norm_diff,
        max_abs_diff: acc.max_diff,
        mean_abs_diff: if total > 0 { acc.sum_abs_diff / n } else { 0.0 },
        rms_diff: if total > 0 { (acc.sum_sq_diff / n).sqrt() } else { 0.0 },
        relative_error,
        frobenius_norm_a,
        frobenius_norm_b,
        exact_matches: acc.exact_matches,
        epsilon_matches: acc.eps_matches,
        total_count: total,
        match_pct: if total > 0 { acc.eps_matches as f64 / n * 100.0 } else { 100.0 },
        sign_changes: acc.sign_changes,
        sparsity_a_pct: acc.zeros_a as f64 / n * 100.0,
        sparsity_b_pct: acc.zeros_b as f64 / n * 100.0,
        structural_zeros_match_pct: acc.struct_zeros as f64 / total_zeros as f64 * 100.0,
        quadrants,
        assessment,
        time_ms,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{EPSILON, SimdLevel, STRASSEN_THRESHOLD};

    /// Helper: check all elements match within epsilon
    fn matrices_match(a: &Matrix, b: &Matrix, eps: f64) -> bool {
        if a.rows() != b.rows() || a.cols() != b.cols() {
            return false;
        }
        a.data()
            .iter()
            .zip(b.data().iter())
            .all(|(x, y)| (x - y).abs() < eps)
    }

    // ── Naive tests ──

    #[test]
    fn test_naive_identity() {
        for &n in &[4, 8, 16] {
            let a = Matrix::random(n, n, Some(42)).unwrap();
            let id = Matrix::identity(n).unwrap();
            let result = multiply_naive(&a, &id);
            assert!(
                matrices_match(&a, &result, 1e-12),
                "A * I != A for n={n}"
            );
        }
    }

    #[test]
    fn test_naive_known() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_flat(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = multiply_naive(&a, &b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    // ── Tiled tests ──

    #[test]
    fn test_tiled_matches_naive_64x64() {
        let a = Matrix::random(64, 64, Some(100)).unwrap();
        let b = Matrix::random(64, 64, Some(200)).unwrap();
        let naive = multiply_naive(&a, &b);
        let tiled = multiply_tiled(&a, &b, 32);
        assert!(
            matrices_match(&naive, &tiled, EPSILON),
            "Tiled 64×64 doesn't match naive"
        );
    }

    #[test]
    fn test_tiled_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(300)).unwrap();
        let b = Matrix::random(128, 128, Some(400)).unwrap();
        let naive = multiply_naive(&a, &b);
        let tiled = multiply_tiled(&a, &b, 64);
        assert!(
            matrices_match(&naive, &tiled, EPSILON),
            "Tiled 128×128 doesn't match naive"
        );
    }

    // ── Tiled parallel tests ──

    #[test]
    fn test_tiled_parallel_matches_naive() {
        let a = Matrix::random(128, 128, Some(500)).unwrap();
        let b = Matrix::random(128, 128, Some(600)).unwrap();
        let naive = multiply_naive(&a, &b);
        let par = multiply_tiled_parallel(&a, &b, 64);
        assert!(
            matrices_match(&naive, &par, EPSILON),
            "Tiled-parallel 128×128 doesn't match naive"
        );
    }

    // ── Strassen tests ──

    #[test]
    fn test_strassen_matches_naive_64x64() {
        let a = Matrix::random(64, 64, Some(700)).unwrap();
        let b = Matrix::random(64, 64, Some(800)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (strassen, _, _) =
            multiply_strassen_padded(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &strassen, EPSILON),
            "Strassen 64×64 doesn't match naive"
        );
    }

    #[test]
    fn test_strassen_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(900)).unwrap();
        let b = Matrix::random(128, 128, Some(1000)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (strassen, _, _) =
            multiply_strassen_padded(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &strassen, EPSILON),
            "Strassen 128×128 doesn't match naive"
        );
    }

    #[test]
    fn test_strassen_matches_naive_255x255() {
        // Asymmetric size: tests padding to 256 (power of 2) and unpadding back
        let a = Matrix::random(255, 255, Some(1100)).unwrap();
        let b = Matrix::random(255, 255, Some(1200)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (strassen, _, _) =
            multiply_strassen_padded(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &strassen, EPSILON),
            "Strassen 255×255 doesn't match naive"
        );
    }

    // ── Winograd tests ──

    #[test]
    fn test_winograd_known_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        // Critical: verifies Winograd assembly formulas are correct
        let a = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_flat(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let (c, _, _) = multiply_winograd_padded(&a, &b, 1, SimdLevel::Scalar);
        assert!((c.get(0, 0) - 19.0).abs() < 1e-9, "C[0,0] = {} != 19", c.get(0, 0));
        assert!((c.get(0, 1) - 22.0).abs() < 1e-9, "C[0,1] = {} != 22", c.get(0, 1));
        assert!((c.get(1, 0) - 43.0).abs() < 1e-9, "C[1,0] = {} != 43", c.get(1, 0));
        assert!((c.get(1, 1) - 50.0).abs() < 1e-9, "C[1,1] = {} != 50", c.get(1, 1));
    }

    #[test]
    fn test_winograd_matches_naive_64x64() {
        let a = Matrix::random(64, 64, Some(1300)).unwrap();
        let b = Matrix::random(64, 64, Some(1400)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (winograd, _, _) =
            multiply_winograd_padded(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &winograd, EPSILON),
            "Winograd 64×64 doesn't match naive"
        );
    }

    #[test]
    fn test_winograd_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(1500)).unwrap();
        let b = Matrix::random(128, 128, Some(1600)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (winograd, _, _) =
            multiply_winograd_padded(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &winograd, EPSILON),
            "Winograd 128×128 doesn't match naive"
        );
    }

    #[test]
    fn test_winograd_matches_naive_255x255() {
        // Non-power-of-2: tests padding to 256 and unpadding back
        let a = Matrix::random(255, 255, Some(1700)).unwrap();
        let b = Matrix::random(255, 255, Some(1800)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (winograd, _, _) =
            multiply_winograd_padded(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &winograd, EPSILON),
            "Winograd 255×255 doesn't match naive"
        );
    }

    // ── Comparison tests ──

    #[test]
    fn test_compare_equal_matrices() {
        let a = Matrix::random(64, 64, Some(42)).unwrap();
        let b = Matrix::random(64, 64, Some(42)).unwrap(); // same seed = identical
        let result = compare_matrices(&a, &b, EPSILON);
        assert!(result.is_equal, "Identical matrices should be equal");
        assert!(
            result.max_abs_diff < 1e-15,
            "Max diff should be ~0 for identical matrices"
        );
        assert_eq!(result.match_count, result.total_count);
    }

    #[test]
    fn test_compare_different_matrices() {
        let a = Matrix::random(64, 64, Some(42)).unwrap();
        let b = Matrix::random(64, 64, Some(99)).unwrap(); // different seed
        let result = compare_matrices(&a, &b, EPSILON);
        assert!(!result.is_equal, "Different matrices should not be equal");
    }

    // ── Scientific comparison tests ──

    #[test]
    fn test_compare_scientific_identical() {
        let a = Matrix::random(64, 64, Some(42)).unwrap();
        let b = Matrix::random(64, 64, Some(42)).unwrap();
        let result = compare_matrices_scientific(&a, &b, EPSILON);
        assert_eq!(
            result.assessment,
            crate::common::VvAssessment::Identical,
            "Same-seed matrices should be Identical"
        );
        assert_eq!(result.epsilon_matches, result.total_count);
        assert!((result.match_pct - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_compare_scientific_different() {
        let a = Matrix::random(64, 64, Some(42)).unwrap();
        let b = Matrix::random(64, 64, Some(99)).unwrap();
        let result = compare_matrices_scientific(&a, &b, EPSILON);
        assert!(
            result.assessment == crate::common::VvAssessment::Significant
                || result.assessment == crate::common::VvAssessment::Suspicious,
            "Different matrices should have high relative error, got {:?}",
            result.assessment
        );
        assert!(result.max_abs_diff > 0.1, "Max diff should be significant");
    }
}
