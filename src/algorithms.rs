// algorithms.rs — All matrix multiplication algorithms and comparison.

// Order of implementation follows verification chain:
// 1. multiply_naive
// 2. multiply_tiled  — Cache-blocked version (6 nested loops)
// 3. multiply_tiled_parallel
// 4. multiply_strassen
// 5. compare_matrices

use crate::common::{
    ComparisonResult, ProgressHandle, QuadrantStats, ScientificComparisonResult,
    SimdLevel, VvAssessment, DEFAULT_TILE_SIZE,
};
use crate::matrix::{Matrix, align_up, SIMD_ALIGN};
use rayon::prelude::*;


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

    let c_stride = result.stride();
    let a_stride = a.stride();
    let b_stride = b.stride();
    // Thread-count-based chunking: ~num_threads tasks instead of m tasks.
    // Reduces rayon scheduling overhead dramatically for large matrices.
    let num_threads = rayon::current_num_threads().max(1);
    let rows_per_chunk = ((m + num_threads - 1) / num_threads).max(1);
    // SAFETY: Each thread writes only to its own row-band slice — no overlap.
    result.data.par_chunks_mut(rows_per_chunk * c_stride).enumerate().for_each(|(chunk_idx, chunk)| {
        let i_start = chunk_idx * rows_per_chunk;
        let i_end = (i_start + rows_per_chunk).min(m);
        unsafe {
            for i in i_start..i_end {
                let local_i = i - i_start;
                for kb in (0..n).step_by(ts) {
                    for jb in (0..p).step_by(ts) {
                        let k_end = (kb + ts).min(n);
                        let j_end = (jb + ts).min(p);

                        for k in kb..k_end {
                            let a_ik = *a.data().get_unchecked(i * a_stride + k);
                            for j in jb..j_end {
                                *chunk.get_unchecked_mut(local_i * c_stride + j) +=
                                    a_ik * *b.data().get_unchecked(k * b_stride + j);
                            }
                        }
                    }
                }
            }
        }
    });

    result
}

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

// ─── multiply_strassen_hpc (Outer Strip Decomposition) ──────────────────────
//
// PROBLEM: Standard Strassen creates 7 tasks via rayon::join at level 1.
// With 16 threads, 9 threads sit idle → 22% core utilization.
//
// SOLUTION: "Work-first" outer horizontal strip decomposition.
// Split A into num_threads horizontal strips, multiply each strip × B in parallel.
// Mathematically: C[i..i+h, :] = A[i..i+h, :] × B — identical result.
// Guarantee: ALL threads busy from the first millisecond.

pub fn multiply_strassen_hpc(
    a: &Matrix,
    b: &Matrix,
    threshold: usize,
    simd: SimdLevel,
) -> Matrix {
    let n = a.rows().max(b.cols());
    let num_threads = rayon::current_num_threads();

    // Small matrices or few threads — fall back to regular Strassen
    if n < threshold * 8 || num_threads < 4 {
        return multiply_strassen_padded(a, b, threshold, simd).0;
    }

    // Decompose A into horizontal strips (one per thread)
    let strip_height = (a.rows() + num_threads - 1) / num_threads;

    let strips: Vec<Matrix> = (0..num_threads)
        .filter_map(|t| {
            let row_start = t * strip_height;
            if row_start >= a.rows() { return None; }
            let row_end = (row_start + strip_height).min(a.rows());
            let stride = a.stride();
            // Copy the rows for this strip into a new contiguous matrix
            let mut data = vec![0.0f64; (row_end - row_start) * stride];
            for i in 0..(row_end - row_start) {
                let src_offset = (row_start + i) * stride;
                let dst_offset = i * stride;
                data[dst_offset..dst_offset + a.cols()]
                    .copy_from_slice(&a.data()[src_offset..src_offset + a.cols()]);
            }
            Some(Matrix::from_raw_parts(row_end - row_start, a.cols(), stride, data))
        })
        .collect();

    // Parallel Strassen on each strip × B
    let result_strips: Vec<Matrix> = strips
        .par_iter()
        .map(|strip| multiply_strassen_padded(strip, b, threshold, simd).0)
        .collect();

    // Assemble result from strips
    let total_rows: usize = result_strips.iter().map(|s| s.rows()).sum();
    let p = b.cols();
    let out_stride = result_strips.first().map(|s| s.stride()).unwrap_or(align_up(p, SIMD_ALIGN));
    let mut result_data = vec![0.0f64; total_rows * out_stride];
    let mut row_offset = 0;
    for strip in &result_strips {
        for i in 0..strip.rows() {
            let src_offset = i * strip.stride();
            let dst_offset = (row_offset + i) * out_stride;
            result_data[dst_offset..dst_offset + p]
                .copy_from_slice(&strip.data()[src_offset..src_offset + p]);
        }
        row_offset += strip.rows();
    }

    Matrix::from_raw_parts(total_rows, p, out_stride, result_data)
}

// ─── multiply_tiled_parallel_v2 (Two-Level L2+L1 Tiling) ───────────────────
//
// PROBLEM: Current tiled_parallel uses single-level tiling with par_chunks_mut(stride)
// = one row per chunk. For 4096×4096 with 16 threads: 4096 tiny rayon tasks.
//
// SOLUTION: Two-level cache-oblivious tiling with thread-optimal chunking.
//   Outer: L2-sized tiles (hold in 2MB P-core L2 cache)
//   Inner: L1-sized tiles (hold in 48KB P-core L1 cache)
//   Chunking: rows/num_threads rows per chunk (one chunk per thread)
//
// The inner j-loop uses slice iteration which LLVM auto-vectorizes into AVX2 FMA.

pub fn multiply_tiled_parallel_v2(
    a: &Matrix,
    b: &Matrix,
    tile_l2: usize,
    tile_l1: usize,
) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    assert_eq!(a.cols(), b.rows(), "A.cols ({}) != B.rows ({})", n, b.rows());

    let mut result = Matrix::zeros(m, p).expect("Failed to allocate result matrix");
    let c_stride = result.stride();
    let b_stride = b.stride();
    let a_stride = a.stride();
    let num_threads = rayon::current_num_threads().max(1);

    // Rows per thread, rounded up to L1 tile boundary for alignment
    let rows_per_thread = ((m + num_threads - 1) / num_threads)
        .max(tile_l1)
        .min(m);

    result.data_mut()
        .par_chunks_mut(rows_per_thread * c_stride)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let i_global_start = chunk_idx * rows_per_thread;
            let i_global_end = (i_global_start + rows_per_thread).min(m);

            // Two-level tiling: L2 outer, L1 inner
            for kb_l2 in (0..n).step_by(tile_l2) {
                let k_l2_end = (kb_l2 + tile_l2).min(n);
                for jb_l2 in (0..p).step_by(tile_l2) {
                    let j_l2_end = (jb_l2 + tile_l2).min(p);

                    // L1 tiling within each L2 tile
                    for ib_l1 in (i_global_start..i_global_end).step_by(tile_l1) {
                        let i_l1_end = (ib_l1 + tile_l1).min(i_global_end);
                        for kb_l1 in (kb_l2..k_l2_end).step_by(tile_l1) {
                            let k_l1_end = (kb_l1 + tile_l1).min(k_l2_end);
                            for jb_l1 in (jb_l2..j_l2_end).step_by(tile_l1) {
                                let j_l1_end = (jb_l1 + tile_l1).min(j_l2_end);
                                let j_width = j_l1_end - jb_l1;

                                // Micro-kernel: L1-resident computation
                                for i in ib_l1..i_l1_end {
                                    let local_i = i - i_global_start;
                                    for k in kb_l1..k_l1_end {
                                        // Broadcast a[i][k] — stays in register
                                        let a_ik = unsafe { *a.data().get_unchecked(i * a_stride + k) };

                                        // Inner j-loop: LLVM vectorizes into AVX2 FMA
                                        let c_start = local_i * c_stride + jb_l1;
                                        let b_start = k * b_stride + jb_l1;

                                        let c_slice = &mut c_chunk[c_start..c_start + j_width];
                                        let b_slice = unsafe {
                                            std::slice::from_raw_parts(
                                                b.data().as_ptr().add(b_start),
                                                j_width,
                                            )
                                        };
                                        // This loop → LLVM → _mm256_fmadd_pd automatically
                                        c_slice.iter_mut().zip(b_slice.iter())
                                            .for_each(|(c, &bv)| *c += a_ik * bv);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    result
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

// ─── Intel MKL (CBLAS DGEMM) ────────────────────────────────────────────────
//
// Feature-gated: only compiled when `cargo build --features mkl` is used.
// Calls MKL's cblas_dgemm for dense matrix multiplication. Single FFI call
// replaces the entire multiply loop — MKL handles tiling, SIMD, threading.
//
// All FFI declarations are centralized in sparse.rs (mkl_ffi module).
// This avoids the double-linking crash caused by multiple #[link(name="mkl_rt")]
// blocks conflicting with the intel-mkl-src crate.

/// Multiply two dense matrices using Intel MKL's cblas_dgemm (runtime-loaded).
/// C = alpha * A * B + beta * C  (with alpha=1.0, beta=0.0).
/// Returns `Err` if MKL is unavailable or on dimension mismatch.
pub fn multiply_mkl(a: &Matrix, b: &Matrix) -> anyhow::Result<Matrix> {
    use crate::sparse::{mkl_ffi, mkl_runtime};

    anyhow::ensure!(
        crate::sparse::is_mkl_runtime_available(),
        "MKL runtime not available (DLLs not found)"
    );

    anyhow::ensure!(
        a.cols() == b.rows(),
        "MKL DGEMM: dimension mismatch A.cols={} != B.rows={}",
        a.cols(), b.rows(),
    );

    let rt = mkl_runtime();
    let m = a.rows() as i32;
    let n = b.cols() as i32;
    let k = a.cols() as i32;

    unsafe { (rt.mkl_set_num_threads)(1); }

    let mut c_data = vec![0.0f64; (m as usize) * (n as usize)];

    unsafe {
        (rt.cblas_dgemm)(
            mkl_ffi::CBLAS_ROW_MAJOR,
            mkl_ffi::CBLAS_NO_TRANS,
            mkl_ffi::CBLAS_NO_TRANS,
            m, n, k,
            1.0,
            a.data().as_ptr(), k,
            b.data().as_ptr(), n,
            0.0,
            c_data.as_mut_ptr(), n,
        );
    }

    Ok(Matrix::from_flat(m as usize, n as usize, c_data)?)
}

/// MKL multiply with progress handle.
pub fn multiply_mkl_with_progress(
    a: &Matrix,
    b: &Matrix,
    progress: &crate::common::ProgressHandle,
) -> anyhow::Result<Matrix> {
    progress.set(1, 2);
    let result = multiply_mkl(a, b)?;
    progress.set(2, 2);
    Ok(result)
}

/// Subprocess entry point for `--mkl-check`.
/// Uses runtime loading — no static linking needed.
pub fn run_mkl_probe() -> ! {
    use crate::sparse::{mkl_ffi, is_mkl_runtime_available, mkl_runtime};

    crate::compute_worker::ensure_mkl_runtime_paths();

    unsafe {
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("MKL_DYNAMIC", "FALSE");
    }

    if !is_mkl_runtime_available() {
        std::process::exit(1); // DLLs not found
    }

    let rt = mkl_runtime();
    unsafe { (rt.mkl_set_num_threads)(1); }

    let a = [1.0f64, 0.0, 0.0, 1.0];
    let b = [2.0f64, 3.0, 4.0, 5.0];
    let mut c = [0.0f64; 4];

    unsafe {
        (rt.cblas_dgemm)(
            mkl_ffi::CBLAS_ROW_MAJOR,
            mkl_ffi::CBLAS_NO_TRANS,
            mkl_ffi::CBLAS_NO_TRANS,
            2, 2, 2,
            1.0,
            a.as_ptr(), 2,
            b.as_ptr(), 2,
            0.0,
            c.as_mut_ptr(), 2,
        );
    }

    if (c[0] - 2.0).abs() > 1e-10 || (c[3] - 5.0).abs() > 1e-10 {
        std::process::exit(2);
    }

    std::process::exit(0);
}

/// Check whether MKL runtime is available via runtime loading.
/// No subprocess needed — just checks if libloading can find the DLL.
pub fn is_mkl_available() -> bool {
    crate::sparse::is_mkl_runtime_available()
}

// ─── Chapter 20: Strip-Parallel Hybrid Algorithms ────────────────────────────
//
// VTune diagnosis: rayon::join binary tree gives only 2 concurrent tasks at top
// level → 22% core utilization on 24-core i9-13900K.
// Fix: divide A into horizontal strips (one per thread), multiply each strip × B
// independently. All cores busy from microsecond one.

/// Strip-parallel Strassen: splits A into horizontal strips, each strip × B
/// computed via Strassen in parallel. Guarantees full core utilization.
/// Falls back to regular Strassen for small matrices.
pub fn multiply_strassen_hybrid(
    a: &Matrix,
    b: &Matrix,
    threshold: usize,
    simd: SimdLevel,
) -> (Matrix, f64, f64) {
    let n_threads = rayon::current_num_threads();
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    assert_eq!(n, b.rows(), "A.cols ({}) != B.rows ({})", n, b.rows());

    // For small matrices or few threads, regular Strassen is fine
    if m < threshold * 4 || n_threads < 2 {
        return multiply_strassen_padded(a, b, threshold, simd);
    }

    let pad_start = std::time::Instant::now();

    // Divide A into strips of roughly equal height
    let rows_per_strip = (m + n_threads - 1) / n_threads;
    let a_stride = a.stride();
    let c_stride = align_up(p, SIMD_ALIGN);
    let mut result_data = vec![0.0f64; m * c_stride];

    let padding_ms = pad_start.elapsed().as_secs_f64() * 1000.0;

    // Parallel strip multiplication via par_chunks_mut — each thread owns
    // a non-overlapping slice of the output. Zero contention.
    result_data
        .par_chunks_mut(rows_per_strip * c_stride)
        .enumerate()
        .for_each(|(strip_idx, output_strip)| {
            let row_start = strip_idx * rows_per_strip;
            let row_end = (row_start + rows_per_strip).min(m);
            if row_start >= row_end {
                return;
            }
            let strip_rows = row_end - row_start;

            // Extract A[row_start..row_end, :] as a contiguous sub-matrix
            // a.data() is strided, so extract row-by-row using a_stride
            let mut a_strip_data = Vec::with_capacity(strip_rows * n);
            for r in row_start..row_end {
                a_strip_data.extend_from_slice(&a.data()[r * a_stride..r * a_stride + n]);
            }

            let a_strip = Matrix::from_flat(strip_rows, n, a_strip_data)
                .expect("strip dimensions valid");

            // Each strip uses Strassen recursion independently
            let (c_strip, _, _) = multiply_strassen_padded(&a_strip, b, threshold, simd);

            // Copy result into our output slice using strides
            let c_strip_stride = c_strip.stride();
            for r in 0..strip_rows {
                let src = &c_strip.data()[r * c_strip_stride..r * c_strip_stride + p];
                let dst_start = r * c_stride;
                output_strip[dst_start..dst_start + p].copy_from_slice(src);
            }
        });

    let unpad_start = std::time::Instant::now();
    let result = Matrix { rows: m, cols: p, stride: c_stride, data: result_data };
    let unpadding_ms = unpad_start.elapsed().as_secs_f64() * 1000.0;

    (result, padding_ms, unpadding_ms)
}

/// Strip-parallel Winograd: identical strategy to hybrid Strassen
/// but uses Winograd's variant (15 additions vs 18) for each strip.
pub fn multiply_winograd_hybrid(
    a: &Matrix,
    b: &Matrix,
    threshold: usize,
    simd: SimdLevel,
) -> (Matrix, f64, f64) {
    let n_threads = rayon::current_num_threads();
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    assert_eq!(n, b.rows(), "A.cols ({}) != B.rows ({})", n, b.rows());

    if m < threshold * 4 || n_threads < 2 {
        return multiply_winograd_padded(a, b, threshold, simd);
    }

    let pad_start = std::time::Instant::now();
    let rows_per_strip = (m + n_threads - 1) / n_threads;
    let a_stride = a.stride();
    let c_stride = align_up(p, SIMD_ALIGN);
    let mut result_data = vec![0.0f64; m * c_stride];
    let padding_ms = pad_start.elapsed().as_secs_f64() * 1000.0;

    result_data
        .par_chunks_mut(rows_per_strip * c_stride)
        .enumerate()
        .for_each(|(strip_idx, output_strip)| {
            let row_start = strip_idx * rows_per_strip;
            let row_end = (row_start + rows_per_strip).min(m);
            if row_start >= row_end {
                return;
            }
            let strip_rows = row_end - row_start;

            // Extract A[row_start..row_end, :] row-by-row using a_stride
            let mut a_strip_data = Vec::with_capacity(strip_rows * n);
            for r in row_start..row_end {
                a_strip_data.extend_from_slice(&a.data()[r * a_stride..r * a_stride + n]);
            }

            let a_strip = Matrix::from_flat(strip_rows, n, a_strip_data)
                .expect("strip dimensions valid");

            let (c_strip, _, _) = multiply_winograd_padded(&a_strip, b, threshold, simd);

            // Copy result into our output slice using strides
            let c_strip_stride = c_strip.stride();
            for r in 0..strip_rows {
                let src = &c_strip.data()[r * c_strip_stride..r * c_strip_stride + p];
                let dst_start = r * c_stride;
                output_strip[dst_start..dst_start + p].copy_from_slice(src);
            }
        });

    let unpad_start = std::time::Instant::now();
    let result = Matrix { rows: m, cols: p, stride: c_stride, data: result_data };
    let unpadding_ms = unpad_start.elapsed().as_secs_f64() * 1000.0;

    (result, padding_ms, unpadding_ms)
}

// ─── Chapter 20: Two-Level Cache-Aware Tiling ────────────────────────────────
//
// VTune: L3 Bound 19%, Memory Bound 39.6%.
// Root cause: DEFAULT_TILE_SIZE=64 → 3×64²×8 = 96KB, doesn't fit L1 (48KB).
// Fix: two-level tiling — outer L2-resident (256), inner L1-resident (32).
// Data stays in the fastest cache level possible.

/// Two-level cache-aware tiled multiplication (sequential).
/// Outer loop tiles for L2 residency, inner tiles for L1 residency.
/// L1 tile: 3×32²×8 = 24KB < 48KB (P-core L1).
/// L2 tile: 3×256²×8 = 1.5MB < 2MB (P-core L2).
pub fn multiply_tiled_l2l1(
    a: &Matrix,
    b: &Matrix,
    tile_l2: usize,
    tile_l1: usize,
) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();
    assert_eq!(n, b.rows(), "A.cols ({}) != B.rows ({})", n, b.rows());

    let b_stride = b.stride();
    let c_stride = align_up(p, SIMD_ALIGN);
    let mut result = vec![0.0f64; m * c_stride];

    // Outer tiling: L2-resident blocks
    for ib in (0..m).step_by(tile_l2) {
        for kb in (0..n).step_by(tile_l2) {
            for jb in (0..p).step_by(tile_l2) {
                let i2_end = (ib + tile_l2).min(m);
                let k2_end = (kb + tile_l2).min(n);
                let j2_end = (jb + tile_l2).min(p);

                // Inner tiling: L1-resident micro-blocks
                for ii in (ib..i2_end).step_by(tile_l1) {
                    for kk in (kb..k2_end).step_by(tile_l1) {
                        for jj in (jb..j2_end).step_by(tile_l1) {
                            let i1_end = (ii + tile_l1).min(i2_end);
                            let k1_end = (kk + tile_l1).min(k2_end);
                            let j1_end = (jj + tile_l1).min(j2_end);

                            // L1-resident kernel: dispatch to AVX2+FMA
                            // tiled micro-kernel when available, else
                            // scalar i-k-j (LLVM auto-vectorizes).
                            #[cfg(target_arch = "x86_64")]
                            {
                                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                                    unsafe {
                                        crate::simd_core::tile_kernel_avx2_fma(
                                            a.data(), a.stride(),
                                            b.data(), b_stride,
                                            &mut result, c_stride,
                                            ii, i1_end,
                                            kk, k1_end,
                                            jj, j1_end,
                                        );
                                    }
                                } else {
                                    for i in ii..i1_end {
                                        for k in kk..k1_end {
                                            let a_ik = unsafe { a.get_unchecked(i, k) };
                                            let c_base = i * c_stride;
                                            let b_base = k * b_stride;
                                            for j in jj..j1_end {
                                                unsafe {
                                                    *result.get_unchecked_mut(c_base + j) +=
                                                        a_ik * *b.data().get_unchecked(b_base + j);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                for i in ii..i1_end {
                                    for k in kk..k1_end {
                                        let a_ik = unsafe { a.get_unchecked(i, k) };
                                        let c_base = i * c_stride;
                                        let b_base = k * b_stride;
                                        for j in jj..j1_end {
                                            unsafe {
                                                *result.get_unchecked_mut(c_base + j) +=
                                                    a_ik * *b.data().get_unchecked(b_base + j);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Matrix { rows: m, cols: p, stride: c_stride, data: result }
}

/// Parallel two-level tiled multiplication.
/// Distributes L2-sized row-bands across rayon threads.
/// Each thread performs L1-tiled multiplication within its band.
pub fn multiply_tiled_l2l1_parallel(
    a: &Matrix,
    b: &Matrix,
    tile_l2: usize,
    tile_l1: usize,
) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();
    assert_eq!(n, b.rows(), "A.cols ({}) != B.rows ({})", n, b.rows());

    let b_stride = b.stride();
    let c_stride = align_up(p, SIMD_ALIGN);
    let mut result = vec![0.0f64; m * c_stride];

    // Thread-count-based chunking: distribute rows evenly across rayon threads,
    // then apply L2/L1 tiling within each thread's band.
    let num_threads = rayon::current_num_threads().max(1);
    let rows_per_band = ((m + num_threads - 1) / num_threads).max(tile_l1);
    result
        .par_chunks_mut(rows_per_band * c_stride)
        .enumerate()
        .for_each(|(block_i, c_block)| {
            let ib = block_i * rows_per_band;
            let i2_end = (ib + rows_per_band).min(m);

            for kb in (0..n).step_by(tile_l2) {
                for jb in (0..p).step_by(tile_l2) {
                    let k2_end = (kb + tile_l2).min(n);
                    let j2_end = (jb + tile_l2).min(p);

                    for ii in (ib..i2_end).step_by(tile_l1) {
                        for kk in (kb..k2_end).step_by(tile_l1) {
                            for jj in (jb..j2_end).step_by(tile_l1) {
                                let i1_end = (ii + tile_l1).min(i2_end);
                                let k1_end = (kk + tile_l1).min(k2_end);
                                let j1_end = (jj + tile_l1).min(j2_end);

                                #[cfg(target_arch = "x86_64")]
                                {
                                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                                        // c_block is offset by ib rows from the full C.
                                        // tile_kernel indexes a_data[i * a_stride + k] — needs global i.
                                        // But c_data[i * c_stride + j] — needs local i.
                                        // Solution: pass A with global indices, C with local indices.
                                        // We offset A's i_start/i_end by ib to get global, but c_block
                                        // expects local. So we use the scalar fallback for the parallel
                                        // variant to keep correctness with the local c_block offset.
                                        for i in ii..i1_end {
                                            let local_i = i - ib;
                                            let a_stride_local = a.stride();
                                            for k in kk..k1_end {
                                                unsafe {
                                                    let a_ik = *a.data().get_unchecked(i * a_stride_local + k);
                                                    let b_base = k * b_stride;
                                                    let c_base = local_i * c_stride;
                                                    let j_len = j1_end - jj;
                                                    let j_simd8 = j_len / 8 * 8;
                                                    let j_simd4 = j_len / 4 * 4;
                                                    use std::arch::x86_64::*;
                                                    let a_vec = _mm256_set1_pd(a_ik);

                                                    // 2x unrolled: 8 f64/iter, saturates both FMA ports
                                                    let mut j = 0usize;
                                                    while j < j_simd8 {
                                                        let jj_off = jj + j;
                                                        let c0 = _mm256_loadu_pd(
                                                            c_block.as_ptr().add(c_base + jj_off));
                                                        let c1 = _mm256_loadu_pd(
                                                            c_block.as_ptr().add(c_base + jj_off + 4));
                                                        let b0 = _mm256_loadu_pd(
                                                            b.data().as_ptr().add(b_base + jj_off));
                                                        let b1 = _mm256_loadu_pd(
                                                            b.data().as_ptr().add(b_base + jj_off + 4));
                                                        _mm256_storeu_pd(
                                                            c_block.as_mut_ptr().add(c_base + jj_off),
                                                            _mm256_fmadd_pd(a_vec, b0, c0));
                                                        _mm256_storeu_pd(
                                                            c_block.as_mut_ptr().add(c_base + jj_off + 4),
                                                            _mm256_fmadd_pd(a_vec, b1, c1));
                                                        j += 8;
                                                    }
                                                    // Single-vector remainder (4-7 elements)
                                                    while j < j_simd4 {
                                                        let jj_off = jj + j;
                                                        let c_vec = _mm256_loadu_pd(
                                                            c_block.as_ptr().add(c_base + jj_off));
                                                        let b_vec = _mm256_loadu_pd(
                                                            b.data().as_ptr().add(b_base + jj_off));
                                                        _mm256_storeu_pd(
                                                            c_block.as_mut_ptr().add(c_base + jj_off),
                                                            _mm256_fmadd_pd(a_vec, b_vec, c_vec));
                                                        j += 4;
                                                    }
                                                    // Scalar remainder (0-3 elements)
                                                    for j in j_simd4..j_len {
                                                        let jj_off = jj + j;
                                                        *c_block.get_unchecked_mut(c_base + jj_off) +=
                                                            a_ik * *b.data().get_unchecked(b_base + jj_off);
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        for i in ii..i1_end {
                                            let local_i = i - ib;
                                            for k in kk..k1_end {
                                                let a_ik = unsafe { a.get_unchecked(i, k) };
                                                let b_base = k * b_stride;
                                                for j in jj..j1_end {
                                                    unsafe {
                                                        *c_block.get_unchecked_mut(local_i * c_stride + j) +=
                                                            a_ik * *b.data().get_unchecked(b_base + j);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    for i in ii..i1_end {
                                        let local_i = i - ib;
                                        for k in kk..k1_end {
                                            let a_ik = unsafe { a.get_unchecked(i, k) };
                                            let b_base = k * b_stride;
                                            for j in jj..j1_end {
                                                unsafe {
                                                    *c_block.get_unchecked_mut(local_i * c_stride + j) +=
                                                        a_ik * *b.data().get_unchecked(b_base + j);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    Matrix { rows: m, cols: p, stride: c_stride, data: result }
}

// ─── Chapter 20: Benchmark Function ──────────────────────────────────────────

/// Benchmark all algorithms at multiple sizes. Returns formatted results.
/// Used by the Benchmark Suite menu item and for VTune before/after comparison.
pub fn run_benchmark_suite(simd: SimdLevel) -> Vec<BenchmarkEntry> {
    let sizes = [256usize, 512, 1024, 2048];
    let threshold = crate::common::STRASSEN_THRESHOLD;
    let tile = crate::common::DEFAULT_TILE_SIZE;
    let tile_l2 = TILE_L2;
    let tile_l1 = TILE_L1;
    let mut results = Vec::new();

    for &n in &sizes {
        let a = Matrix::random(n, n, Some(42)).unwrap();
        let b = Matrix::random(n, n, Some(43)).unwrap();

        // Warmup
        let _ = multiply_tiled_parallel(&a, &b, tile);

        // Tiled Parallel (old)
        let t = std::time::Instant::now();
        let _ = multiply_tiled_parallel(&a, &b, tile);
        let tiled_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Tiled L2/L1 Parallel (new)
        let t = std::time::Instant::now();
        let _ = multiply_tiled_l2l1_parallel(&a, &b, tile_l2, tile_l1);
        let tiled_l2l1_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Strassen (old)
        let t = std::time::Instant::now();
        let _ = multiply_strassen_padded(&a, &b, threshold, simd);
        let strassen_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Strassen Hybrid (new)
        let t = std::time::Instant::now();
        let _ = multiply_strassen_hybrid(&a, &b, threshold, simd);
        let strassen_hybrid_ms = t.elapsed().as_secs_f64() * 1000.0;

        let ops = 2.0 * n as f64 * n as f64 * n as f64;
        let best_ms = tiled_ms.min(tiled_l2l1_ms).min(strassen_ms).min(strassen_hybrid_ms);
        let gflops = ops / (best_ms / 1000.0) / 1e9;

        results.push(BenchmarkEntry {
            size: n,
            tiled_par_ms: tiled_ms,
            tiled_l2l1_ms,
            strassen_ms,
            strassen_hybrid_ms,
            best_gflops: gflops,
        });
    }
    results
}

/// Single benchmark entry for one matrix size.
pub struct BenchmarkEntry {
    pub size: usize,
    pub tiled_par_ms: f64,
    pub tiled_l2l1_ms: f64,
    pub strassen_ms: f64,
    pub strassen_hybrid_ms: f64,
    pub best_gflops: f64,
}

/// Default tile sizes for two-level tiling (i9-13900K optimized).
/// L1: 3×32²×8 = 24KB < 48KB P-core L1.
/// L2: 3×256²×8 = 1.5MB < 2MB P-core L2.
pub const TILE_L1: usize = 32;
pub const TILE_L2: usize = 256;

// ─── HPC Fused: Strip-Parallel + L2/L1 Tiling ──────────────────────────────
//
// VTune fix: combines strip parallelism (all cores busy from µs one) with
// two-level cache tiling (L2=256, L1=32). Zero per-thread allocation —
// each thread works directly on its &mut [f64] slice.
//
// Key difference from multiply_tiled_l2l1_parallel: that function chunks by
// tile_l2 (256) rows → ceil(m/256) chunks. For m=2048 on 24 cores → 8 chunks,
// 16 cores idle. HPC fused chunks by ceil(m/n_threads) → n_threads chunks.

/// High-Performance Fused Multiply: strip-parallel + L2/L1 tiling + AVX2/FMA.
///
/// Guarantees full core utilization by creating exactly n_threads parallel
/// strips, each performing L2→L1 two-level tiled multiplication with
/// AVX2+FMA micro-kernels. No per-thread Matrix allocation.
pub fn multiply_hpc_fused(
    a: &Matrix,
    b: &Matrix,
    tile_l2: usize,
    tile_l1: usize,
) -> Matrix {
    let n_threads = rayon::current_num_threads().max(1);
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();
    assert_eq!(n, b.rows(), "A.cols ({}) != B.rows ({})", n, b.rows());

    let a_stride = a.stride();
    let b_stride = b.stride();
    let c_stride = align_up(p, SIMD_ALIGN);

    // Single allocation for entire output — no per-thread allocs
    let mut result_data = vec![0.0f64; m * c_stride];

    // Divide output rows into n_threads strips. Each thread owns a
    // non-overlapping horizontal band — zero contention, zero false sharing.
    let rows_per_strip = (m + n_threads - 1) / n_threads;

    result_data
        .par_chunks_mut(rows_per_strip * c_stride)
        .enumerate()
        .for_each(|(strip_idx, c_block)| {
            let ib_global = strip_idx * rows_per_strip;
            let i_end_global = (ib_global + rows_per_strip).min(m);
            let strip_rows = i_end_global - ib_global;
            if strip_rows == 0 {
                return;
            }

            // Two-level tiling within this strip's row band
            // Outer: L2 tiles over k and j dimensions
            for kb in (0..n).step_by(tile_l2) {
                let k2_end = (kb + tile_l2).min(n);
                for jb in (0..p).step_by(tile_l2) {
                    let j2_end = (jb + tile_l2).min(p);

                    // Inner: L1 tiles (3×32²×8 = 24KB fits in 48KB L1)
                    for ii in (0..strip_rows).step_by(tile_l1) {
                        let i1_end = (ii + tile_l1).min(strip_rows);
                        for kk in (kb..k2_end).step_by(tile_l1) {
                            let k1_end = (kk + tile_l1).min(k2_end);
                            for jj in (jb..j2_end).step_by(tile_l1) {
                                let j1_end = (jj + tile_l1).min(j2_end);

                                // Micro-kernel: i-k-j with AVX2+FMA
                                #[cfg(target_arch = "x86_64")]
                                {
                                    if is_x86_feature_detected!("avx2")
                                        && is_x86_feature_detected!("fma")
                                    {
                                        for li in ii..i1_end {
                                            let gi = ib_global + li; // global row in A
                                            for k in kk..k1_end {
                                                unsafe {
                                                    let a_ik = *a.data()
                                                        .get_unchecked(gi * a_stride + k);
                                                    let b_base = k * b_stride;
                                                    let c_base = li * c_stride;
                                                    let j_len = j1_end - jj;
                                                    let j_simd = j_len / 4 * 4;

                                                    use std::arch::x86_64::*;
                                                    let a_vec = _mm256_set1_pd(a_ik);

                                                    let mut j = 0usize;
                                                    while j < j_simd {
                                                        let jj_off = jj + j;
                                                        let c_vec = _mm256_loadu_pd(
                                                            c_block.as_ptr()
                                                                .add(c_base + jj_off),
                                                        );
                                                        let b_vec = _mm256_loadu_pd(
                                                            b.data().as_ptr()
                                                                .add(b_base + jj_off),
                                                        );
                                                        let res =
                                                            _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                                                        _mm256_storeu_pd(
                                                            c_block.as_mut_ptr()
                                                                .add(c_base + jj_off),
                                                            res,
                                                        );
                                                        j += 4;
                                                    }
                                                    // Scalar remainder
                                                    for j in j_simd..j_len {
                                                        let jj_off = jj + j;
                                                        *c_block
                                                            .get_unchecked_mut(c_base + jj_off) +=
                                                            a_ik
                                                                * *b.data()
                                                                    .get_unchecked(b_base + jj_off);
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        // Scalar fallback
                                        for li in ii..i1_end {
                                            let gi = ib_global + li;
                                            for k in kk..k1_end {
                                                let a_ik =
                                                    unsafe { *a.data().get_unchecked(gi * a_stride + k) };
                                                let b_base = k * b_stride;
                                                for j in jj..j1_end {
                                                    unsafe {
                                                        *c_block
                                                            .get_unchecked_mut(li * c_stride + j) +=
                                                            a_ik
                                                                * *b.data()
                                                                    .get_unchecked(b_base + j);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    for li in ii..i1_end {
                                        let gi = ib_global + li;
                                        for k in kk..k1_end {
                                            let a_ik =
                                                unsafe { *a.data().get_unchecked(gi * a_stride + k) };
                                            let b_base = k * b_stride;
                                            for j in jj..j1_end {
                                                unsafe {
                                                    *c_block
                                                        .get_unchecked_mut(li * c_stride + j) +=
                                                        a_ik
                                                            * *b.data().get_unchecked(b_base + j);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    Matrix { rows: m, cols: p, stride: c_stride, data: result_data }
}

// ─── Matrix Power with Convergence Detection ─────────────────────────────────
//
// Binary exponentiation: P → P² → P⁴ → P⁸ → ... via repeated squaring.
// Used by MDP module for finding steady-state distribution of stochastic matrices.

/// Snapshot of convergence state at each squaring step.
#[derive(Clone)]
pub struct ConvergenceSnapshot {
    pub iteration: usize,
    pub exponent: u64,
    pub frobenius_diff: f64,
    pub max_entry_change: f64,
}

/// Repeatedly square matrix P until convergence or max_iter.
/// Returns (converged_matrix, snapshots, did_converge).
///
/// Convergence criterion: Frobenius norm ||P^(2k) - P^(2(k-1))|| < epsilon.
pub fn matrix_power_converge(
    p: &Matrix,
    epsilon: f64,
    max_iter: usize,
    progress: &std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> (Matrix, Vec<ConvergenceSnapshot>, bool) {
    use std::sync::atomic::Ordering;

    let n = p.rows();
    assert_eq!(n, p.cols(), "Matrix power requires square matrix");

    // Pack total into high 32 bits
    let total = max_iter as u64;
    progress.store(total << 32, Ordering::Relaxed);

    let mut current = p.clone();
    let mut snapshots = Vec::with_capacity(max_iter);
    let mut exponent: u64 = 1;

    for iter in 0..max_iter {
        let next = multiply_hpc_fused(&current, &current, TILE_L2, TILE_L1);
        exponent *= 2;

        // Compute Frobenius norm of difference and max entry change
        let mut frob_sum = 0.0f64;
        let mut max_change = 0.0f64;
        let c_stride = current.stride();
        let n_stride = next.stride();
        for i in 0..n {
            for j in 0..n {
                let diff = (next.data()[i * n_stride + j] - current.data()[i * c_stride + j]).abs();
                frob_sum += diff * diff;
                if diff > max_change {
                    max_change = diff;
                }
            }
        }
        let frob_diff = frob_sum.sqrt();

        snapshots.push(ConvergenceSnapshot {
            iteration: iter + 1,
            exponent,
            frobenius_diff: frob_diff,
            max_entry_change: max_change,
        });

        // Update progress: low 32 bits = done count
        let done = (iter + 1) as u64;
        progress.store((total << 32) | done, Ordering::Relaxed);

        current = next;

        if frob_diff < epsilon {
            return (current, snapshots, true);
        }
    }

    (current, snapshots, false)
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

    // ── Chapter 20: Hybrid algorithm tests ──

    #[test]
    fn test_strassen_hybrid_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(2000)).unwrap();
        let b = Matrix::random(128, 128, Some(2100)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (hybrid, _, _) =
            multiply_strassen_hybrid(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &hybrid, EPSILON),
            "Strassen hybrid 128x128 doesn't match naive"
        );
    }

    #[test]
    fn test_strassen_hybrid_matches_naive_255x255() {
        let a = Matrix::random(255, 255, Some(2200)).unwrap();
        let b = Matrix::random(255, 255, Some(2300)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (hybrid, _, _) =
            multiply_strassen_hybrid(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &hybrid, EPSILON),
            "Strassen hybrid 255x255 doesn't match naive"
        );
    }

    #[test]
    fn test_winograd_hybrid_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(2400)).unwrap();
        let b = Matrix::random(128, 128, Some(2500)).unwrap();
        let naive = multiply_naive(&a, &b);
        let (hybrid, _, _) =
            multiply_winograd_hybrid(&a, &b, STRASSEN_THRESHOLD, SimdLevel::Scalar);
        assert!(
            matrices_match(&naive, &hybrid, EPSILON),
            "Winograd hybrid 128x128 doesn't match naive"
        );
    }

    // ── Chapter 20: Two-level tiling tests ──

    #[test]
    fn test_tiled_l2l1_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(2600)).unwrap();
        let b = Matrix::random(128, 128, Some(2700)).unwrap();
        let naive = multiply_naive(&a, &b);
        let result = multiply_tiled_l2l1(&a, &b, 64, 16);
        assert!(
            matrices_match(&naive, &result, EPSILON),
            "Tiled L2/L1 128x128 doesn't match naive"
        );
    }

    #[test]
    fn test_tiled_l2l1_parallel_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(2800)).unwrap();
        let b = Matrix::random(128, 128, Some(2900)).unwrap();
        let naive = multiply_naive(&a, &b);
        let result = multiply_tiled_l2l1_parallel(&a, &b, 64, 16);
        assert!(
            matrices_match(&naive, &result, EPSILON),
            "Tiled L2/L1 parallel 128x128 doesn't match naive"
        );
    }

    #[test]
    fn test_tiled_l2l1_non_power_of_2() {
        let a = Matrix::random(100, 100, Some(3000)).unwrap();
        let b = Matrix::random(100, 100, Some(3100)).unwrap();
        let naive = multiply_naive(&a, &b);
        let result = multiply_tiled_l2l1(&a, &b, 64, 32);
        assert!(
            matrices_match(&naive, &result, EPSILON),
            "Tiled L2/L1 100x100 doesn't match naive"
        );
    }

    #[test]
    fn test_tiled_l2l1_parallel_non_power_of_2() {
        let a = Matrix::random(100, 100, Some(3200)).unwrap();
        let b = Matrix::random(100, 100, Some(3300)).unwrap();
        let naive = multiply_naive(&a, &b);
        let result = multiply_tiled_l2l1_parallel(&a, &b, 256, 32);
        assert!(
            matrices_match(&naive, &result, EPSILON),
            "Tiled L2/L1 parallel 100x100 doesn't match naive"
        );
    }

    // ── HPC Fused tests ──

    #[test]
    fn test_hpc_fused_matches_naive_128x128() {
        let a = Matrix::random(128, 128, Some(4000)).unwrap();
        let b = Matrix::random(128, 128, Some(4100)).unwrap();
        let naive = multiply_naive(&a, &b);
        let fused = multiply_hpc_fused(&a, &b, TILE_L2, TILE_L1);
        assert!(
            matrices_match(&naive, &fused, EPSILON),
            "HPC fused 128x128 doesn't match naive"
        );
    }

    #[test]
    fn test_hpc_fused_matches_naive_255x255() {
        let a = Matrix::random(255, 255, Some(4200)).unwrap();
        let b = Matrix::random(255, 255, Some(4300)).unwrap();
        let naive = multiply_naive(&a, &b);
        let fused = multiply_hpc_fused(&a, &b, TILE_L2, TILE_L1);
        assert!(
            matrices_match(&naive, &fused, EPSILON),
            "HPC fused 255x255 doesn't match naive"
        );
    }

    #[test]
    fn test_hpc_fused_non_power_of_2() {
        let a = Matrix::random(100, 100, Some(4400)).unwrap();
        let b = Matrix::random(100, 100, Some(4500)).unwrap();
        let naive = multiply_naive(&a, &b);
        let fused = multiply_hpc_fused(&a, &b, 64, 16);
        assert!(
            matrices_match(&naive, &fused, EPSILON),
            "HPC fused 100x100 doesn't match naive"
        );
    }

    // ── Matrix power convergence tests ──

    #[test]
    fn test_matrix_power_identity() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicU64;
        let id = Matrix::identity(8).unwrap();
        let progress = Arc::new(AtomicU64::new(0));
        let (result, _snaps, converged) = matrix_power_converge(&id, 1e-12, 10, &progress);
        // I^n = I for all n
        assert!(converged, "Identity should converge immediately");
        assert!(
            matrices_match(&id, &result, 1e-12),
            "I^n should equal I"
        );
    }

    #[test]
    fn test_matrix_power_stochastic_row_sums() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicU64;
        // Build a small stochastic matrix (rows sum to 1)
        let n = 8;
        let mut data = vec![0.0; n * n];
        let mut rng_state: u64 = 12345;
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((rng_state >> 33) as f64) / (u32::MAX as f64) + 0.01;
                data[i * n + j] = val;
                row_sum += val;
            }
            // Normalize row
            for j in 0..n {
                data[i * n + j] /= row_sum;
            }
        }
        let p = Matrix::from_flat(n, n, data).unwrap();
        let progress = Arc::new(AtomicU64::new(0));
        let (result, _snaps, _converged) = matrix_power_converge(&p, 1e-10, 30, &progress);
        // Row sums of P^n must still be 1.0
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| result.get(i, j)).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} sum = {}, expected 1.0",
                i,
                row_sum
            );
        }
    }
}
