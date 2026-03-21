// simd_core.rs — unsafe AVX2/AVX-512/SSE4.2 kernels (isolated)


use crate::common::SimdLevel;
use crate::matrix::Matrix;

// ─── Scalar fallback kernel ─────────────────────────────────────────────────

/// Scalar i-k-j multiplication. Safe Rust, no unsafe.
/// Used as baseline and fallback on non-x86 platforms.
pub fn multiply_scalar(a: &Matrix, b: &Matrix) -> Matrix {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();
    assert_eq!(n, b.rows(), "Inner dimensions must match: A is {}×{}, B is {}×{}", m, n, b.rows(), p);

    let mut c = Matrix::zeros(m, p).expect("Failed to allocate result matrix");
    let a_data = a.data();
    let b_data = b.data();
    let c_data = &mut c.data;

    // i-k-j loop order: iterate over rows of A, then columns of A (=rows of B),
    // then columns of B. This is cache-friendly because:
    //   - B[k][j..] is accessed sequentially (contiguous memory)
    //   - C[i][j..] is accessed sequentially (contiguous memory)
    for i in 0..m {
        for k in 0..n {
            let a_ik = a_data[i * n + k];
            for j in 0..p {
                c_data[i * p + j] += a_ik * b_data[k * p + j];
            }
        }
    }

    c
}

// ─── x86_64 SIMD kernels ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_kernels {
    use crate::matrix::Matrix;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // ─── SSE4.2 kernel ──────────────────────────────────────────────────

    /// SSE4.2 matrix multiplication kernel.
    /// Processes 2 f64 values at a time using 128-bit registers.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports SSE4.2 instructions.
    #[target_feature(enable = "sse4.2")]
    pub unsafe fn multiply_sse42(a: &Matrix, b: &Matrix) -> Matrix {
        let m = a.rows();
        let n = a.cols();
        let p = b.cols();
        assert_eq!(n, b.rows(), "Inner dimensions must match: A is {}×{}, B is {}×{}", m, n, b.rows(), p);

        let mut c = Matrix::zeros(m, p).expect("Failed to allocate result matrix");
        let a_data = a.data();
        let b_data = b.data();
        let c_data = &mut c.data;

        let simd_width: usize = 2; // 128-bit / 64-bit = 2 f64
        let p_simd = p - (p % simd_width); // largest multiple of 2 <= p

        // Bounds assertions BEFORE the unsafe math section.
        // These guarantee all pointer arithmetic below is in-bounds.
        assert!(a_data.len() >= m * n, "A data too short");
        assert!(b_data.len() >= n * p, "B data too short");
        assert!(c_data.len() >= m * p, "C data too short");

        // i-k-j loop order with SSE4.2 vectorization
        for i in 0..m {
            for k in 0..n {
                let a_ik = a_data[i * n + k];
                let c_row = i * p;
                let b_row = k * p;

                // SIMD portion: process 2 f64 at a time
                let mut j = 0usize;
                while j < p_simd {
                    unsafe {
                        let a_vec = _mm_set1_pd(a_ik);
                        let b_vec = _mm_loadu_pd(b_data.as_ptr().add(b_row + j));
                        let c_vec = _mm_loadu_pd(c_data.as_ptr().add(c_row + j));
                        let result = _mm_add_pd(c_vec, _mm_mul_pd(a_vec, b_vec));
                        _mm_storeu_pd(c_data.as_mut_ptr().add(c_row + j), result);
                    }
                    j += simd_width;
                }

                // Scalar remainder
                while j < p {
                    c_data[c_row + j] += a_ik * b_data[b_row + j];
                    j += 1;
                }
            }
        }

        c
    }

    // ─── AVX2+FMA kernel ────────────────────────────────────────────────

    /// AVX2+FMA matrix multiplication kernel.
    /// Processes 4 f64 values at a time using 256-bit registers.
    /// Uses fused multiply-add (FMA) for better precision and throughput.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX2 and FMA instructions.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn multiply_avx2(a: &Matrix, b: &Matrix) -> Matrix {
        let m = a.rows();
        let n = a.cols();
        let p = b.cols();
        assert_eq!(n, b.rows(), "Inner dimensions must match: A is {}×{}, B is {}×{}", m, n, b.rows(), p);

        let mut c = Matrix::zeros(m, p).expect("Failed to allocate result matrix");
        let a_data = a.data();
        let b_data = b.data();
        let c_data = &mut c.data;

        let simd_width: usize = 4; // 256-bit / 64-bit = 4 f64
        let p_simd = p - (p % simd_width); // largest multiple of 4 <= p

        // Bounds assertions BEFORE the unsafe math section.
        assert!(a_data.len() >= m * n, "A data too short");
        assert!(b_data.len() >= n * p, "B data too short");
        assert!(c_data.len() >= m * p, "C data too short");

        // i-k-j loop order with AVX2+FMA vectorization
        for i in 0..m {
            for k in 0..n {
                let a_ik = a_data[i * n + k];
                let c_row = i * p;
                let b_row = k * p;

                // SIMD portion: process 4 f64 at a time with FMA
                let mut j = 0usize;
                while j < p_simd {
                    unsafe {
                        let a_vec = _mm256_set1_pd(a_ik);
                        let b_vec = _mm256_loadu_pd(b_data.as_ptr().add(b_row + j));
                        let c_vec = _mm256_loadu_pd(c_data.as_ptr().add(c_row + j));
                        // c[i][j..j+4] += a[i][k] * b[k][j..j+4]
                        let result = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                        _mm256_storeu_pd(c_data.as_mut_ptr().add(c_row + j), result);
                    }
                    j += simd_width;
                }

                // Scalar remainder
                while j < p {
                    c_data[c_row + j] += a_ik * b_data[b_row + j];
                    j += 1;
                }
            }
        }

        c
    }

    // ─── AVX-512 kernel ─────────────────────────────────────────────────

    /// AVX-512 matrix multiplication kernel.
    /// Processes 8 f64 values at a time using 512-bit registers.
    /// Uses fused multiply-add for best throughput.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports AVX-512F instructions.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn multiply_avx512(a: &Matrix, b: &Matrix) -> Matrix {
        let m = a.rows();
        let n = a.cols();
        let p = b.cols();
        assert_eq!(n, b.rows(), "Inner dimensions must match: A is {}×{}, B is {}×{}", m, n, b.rows(), p);

        let mut c = Matrix::zeros(m, p).expect("Failed to allocate result matrix");
        let a_data = a.data();
        let b_data = b.data();
        let c_data = &mut c.data;

        let simd_width: usize = 8; // 512-bit / 64-bit = 8 f64
        let p_simd = p - (p % simd_width); // largest multiple of 8 <= p

        // Bounds assertions BEFORE the unsafe math section.
        assert!(a_data.len() >= m * n, "A data too short");
        assert!(b_data.len() >= n * p, "B data too short");
        assert!(c_data.len() >= m * p, "C data too short");

        // i-k-j loop order with AVX-512 vectorization
        for i in 0..m {
            for k in 0..n {
                let a_ik = a_data[i * n + k];
                let c_row = i * p;
                let b_row = k * p;

                // SIMD portion: process 8 f64 at a time with FMA
                let mut j = 0usize;
                while j < p_simd {
                    unsafe {
                        let a_vec = _mm512_set1_pd(a_ik);
                        let b_vec = _mm512_loadu_pd(b_data.as_ptr().add(b_row + j));
                        let c_vec = _mm512_loadu_pd(c_data.as_ptr().add(c_row + j));
                        // c[i][j..j+8] += a[i][k] * b[k][j..j+8]
                        let result = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
                        _mm512_storeu_pd(c_data.as_mut_ptr().add(c_row + j), result);
                    }
                    j += simd_width;
                }

                // Scalar remainder
                while j < p {
                    c_data[c_row + j] += a_ik * b_data[b_row + j];
                    j += 1;
                }
            }
        }

        c
    }
}

// ─── Dispatcher ─────────────────────────────────────────────────────────────

/// Dispatch matrix multiplication to the appropriate SIMD kernel.
///
/// Selects the kernel based on the provided `SimdLevel`:
///   - `Scalar`  → safe Rust i-k-j loop
///   - `Sse42`   → SSE4.2 128-bit intrinsics (2 f64/op)
///   - `Avx2`    → AVX2+FMA 256-bit intrinsics (4 f64/op)
///   - `Avx512`  → AVX-512 512-bit intrinsics (8 f64/op)
///
/// On non-x86_64 targets, all SIMD levels fall back to the scalar kernel.
pub fn multiply_dispatch(a: &Matrix, b: &Matrix, simd: SimdLevel) -> Matrix {
    match simd {
        SimdLevel::Scalar => multiply_scalar(a, b),

        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse42 => {
            // Safety: caller guarantees CPU supports SSE4.2 via SimdLevel detection
            unsafe { x86_kernels::multiply_sse42(a, b) }
        }

        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // Safety: caller guarantees CPU supports AVX2+FMA via SimdLevel detection
            unsafe { x86_kernels::multiply_avx2(a, b) }
        }

        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // Safety: caller guarantees CPU supports AVX-512F via SimdLevel detection
            unsafe { x86_kernels::multiply_avx512(a, b) }
        }

        // Non-x86_64: all SIMD levels fall back to scalar
        #[cfg(not(target_arch = "x86_64"))]
        _ => multiply_scalar(a, b),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::multiply_naive;
    use crate::common::EPSILON;
    use crate::matrix::Matrix;

    /// Helper: check two matrices are element-wise equal within EPSILON.
    fn assert_matrices_approx_equal(expected: &Matrix, actual: &Matrix, label: &str) {
        assert_eq!(expected.rows(), actual.rows(), "{label}: row count mismatch");
        assert_eq!(expected.cols(), actual.cols(), "{label}: col count mismatch");

        for i in 0..expected.rows() {
            for j in 0..expected.cols() {
                let e = expected.get(i, j);
                let a = actual.get(i, j);
                let diff = (e - a).abs();
                assert!(
                    diff < EPSILON,
                    "{label}: mismatch at [{i},{j}]: expected {e}, got {a} (diff={diff})"
                );
            }
        }
    }

    #[test]
    fn test_scalar_kernel_matches_naive() {
        // Test with a non-square matrix to exercise remainder handling
        let a = Matrix::random(17, 13, Some(42)).unwrap();
        let b = Matrix::random(13, 19, Some(99)).unwrap();

        let naive = multiply_naive(&a, &b);
        let scalar = multiply_scalar(&a, &b);

        assert_matrices_approx_equal(&naive, &scalar, "scalar vs naive");
    }

    #[test]
    fn test_scalar_kernel_small() {
        // 2x3 * 3x2 with known values
        let a = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_flat(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let c = multiply_scalar(&a, &b);

        // C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    fn test_dispatch_matches_naive() {
        // Detect the best SIMD level available on this machine
        let simd_level = detect_best_simd();

        // Use a size that is NOT a multiple of 8 (AVX-512 width) to test remainders
        let a = Matrix::random(23, 17, Some(123)).unwrap();
        let b = Matrix::random(17, 29, Some(456)).unwrap();

        let naive = multiply_naive(&a, &b);
        let dispatched = multiply_dispatch(&a, &b, simd_level);

        assert_matrices_approx_equal(
            &naive,
            &dispatched,
            &format!("dispatch({:?}) vs naive", simd_level),
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse42_kernel_matches_naive() {
        if !is_x86_feature_detected!("sse4.2") {
            return; // skip on CPUs without SSE4.2
        }

        let a = Matrix::random(15, 11, Some(10)).unwrap();
        let b = Matrix::random(11, 21, Some(20)).unwrap();

        let naive = multiply_naive(&a, &b);
        let sse = unsafe { x86_kernels::multiply_sse42(&a, &b) };

        assert_matrices_approx_equal(&naive, &sse, "SSE4.2 vs naive");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_kernel_matches_naive() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return; // skip on CPUs without AVX2+FMA
        }

        let a = Matrix::random(19, 14, Some(30)).unwrap();
        let b = Matrix::random(14, 23, Some(40)).unwrap();

        let naive = multiply_naive(&a, &b);
        let avx2 = unsafe { x86_kernels::multiply_avx2(&a, &b) };

        assert_matrices_approx_equal(&naive, &avx2, "AVX2 vs naive");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_kernel_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return; // skip on CPUs without AVX-512
        }

        let a = Matrix::random(21, 17, Some(50)).unwrap();
        let b = Matrix::random(17, 25, Some(60)).unwrap();

        let naive = multiply_naive(&a, &b);
        let avx512 = unsafe { x86_kernels::multiply_avx512(&a, &b) };

        assert_matrices_approx_equal(&naive, &avx512, "AVX-512 vs naive");
    }

    /// Detect the best SIMD level for the current CPU (test helper).
    fn detect_best_simd() -> SimdLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("sse4.2") {
                return SimdLevel::Sse42;
            }
        }
        SimdLevel::Scalar
    }
}
