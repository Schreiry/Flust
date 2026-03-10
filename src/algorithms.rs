// algorithms.rs — All matrix multiplication algorithms and comparison.
//
// Order of implementation follows verification chain:
// 1. multiply_naive  — Ground Truth (i-k-j order for cache locality)
// 2. multiply_tiled  — Cache-blocked version (6 nested loops)
// 3. multiply_tiled_parallel — rayon parallelization over rows
// 4. multiply_strassen — Recursive Strassen with rayon::join
// 5. compare_matrices — Parallel element-wise comparison

use crate::common::{ComparisonResult, SimdLevel, DEFAULT_TILE_SIZE};
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
}
