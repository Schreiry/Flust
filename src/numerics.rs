// numerics.rs — Scientific numerical analysis toolkit.


use crate::algorithms::{multiply_naive, multiply_strassen_padded};
use crate::common::SimdLevel;
use crate::matrix::Matrix;
use rayon::prelude::*;

// ─── Matrix Norms ────────────────────────────────────────────────────────────

/// Frobenius norm: ||A||_F = sqrt(Sigma_ij a_ij^2)
/// The most intuitive norm — "length" of the matrix treated as a flat vector.
/// Application: relative error measurement, V&V workflows.
/// Parallelized via rayon for large matrices.
pub fn frobenius_norm(a: &Matrix) -> f64 {
    let rows = a.rows();
    (0..rows)
        .into_par_iter()
        .map(|i| a.row_slice(i).iter().map(|&x| x * x).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// 1-norm (column norm): ||A||_1 = max_j (Sigma_i |a_ij|)
/// Maximum absolute column sum.
/// Application: condition number estimation paired with infinity norm.
pub fn norm_1(a: &Matrix) -> f64 {
    let rows = a.rows();
    let cols = a.cols();

    // Accumulate absolute values into per-column sums
    let mut col_sums = vec![0.0_f64; cols];
    for i in 0..rows {
        let row = a.row_slice(i);
        for j in 0..cols {
            col_sums[j] += row[j].abs();
        }
    }
    col_sums.into_iter().fold(0.0_f64, f64::max)
}

/// Infinity norm (row norm): ||A||_inf = max_i (Sigma_j |a_ij|)
/// Maximum absolute row sum.
/// Application: FEA row-dominance analysis, Gershgorin disc estimates.
/// Parallelized: each row is contiguous in memory, naturally parallel.
pub fn norm_infinity(a: &Matrix) -> f64 {
    let rows = a.rows();

    (0..rows)
        .into_par_iter()
        .map(|i| a.row_slice(i).iter().map(|x| x.abs()).sum::<f64>())
        .reduce(|| 0.0_f64, f64::max)
}

// ─── Symmetry Check ──────────────────────────────────────────────────────────

/// Check if a square matrix is symmetric within tolerance.
/// A is symmetric iff |a_ij - a_ji| < tol for all i < j.
pub fn is_symmetric(a: &Matrix, tol: f64) -> bool {
    assert_eq!(
        a.rows(),
        a.cols(),
        "is_symmetric requires square matrix, got {}x{}",
        a.rows(),
        a.cols()
    );
    let n = a.rows();
    for i in 0..n {
        for j in (i + 1)..n {
            if (a.get(i, j) - a.get(j, i)).abs() >= tol {
                return false;
            }
        }
    }
    true
}

// ─── Spectral Radius (Power Method) ──────────────────────────────────────────
//
// Estimates the dominant eigenvalue |lambda_max| via power iteration.
// Algorithm:
//   1. Start with normalized vector v = [1/sqrt(n), ..., 1/sqrt(n)]
//   2. w = A * v  (matrix-vector product, O(n^2))
//   3. lambda = v^T * w  (Rayleigh quotient)
//   4. v = w / ||w||_2
//   5. Repeat until |lambda_new - lambda_old| < tol

/// Returns (|lambda_max|, iterations_used).
pub fn spectral_radius_power_method(
    a: &Matrix,
    max_iter: usize,
    tol: f64,
) -> (f64, usize) {
    assert_eq!(
        a.rows(),
        a.cols(),
        "spectral_radius requires square matrix, got {}x{}",
        a.rows(),
        a.cols()
    );
    let n = a.rows();

    // Normalized initial vector
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v: Vec<f64> = vec![inv_sqrt_n; n];
    let mut lambda = 0.0_f64;
    let mut actual_iters = 0;

    for iter in 0..max_iter {
        // w = A * v (inline matrix-vector product, O(n^2))
        let w: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = a.row_slice(i);
                (0..n).map(|j| row[j] * v[j]).sum::<f64>()
            })
            .collect();

        // ||w||_2
        let w_norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if w_norm < 1e-300 {
            break; // zero matrix or converged to null space
        }

        // Rayleigh quotient: lambda = v^T * w
        let lambda_new: f64 = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum();

        // Normalize: v = w / ||w||
        v = w.iter().map(|&x| x / w_norm).collect();

        actual_iters = iter + 1;
        if (lambda_new - lambda).abs() < tol {
            lambda = lambda_new;
            break;
        }
        lambda = lambda_new;
    }

    (lambda.abs(), actual_iters)
}

// ─── Condition Number Estimate ───────────────────────────────────────────────
//
// Approximate kappa(A) — the ratio of largest to smallest singular value.
// Well-conditioned matrices have kappa near 1; ill-conditioned near infinity.
//
// Two strategies depending on matrix size:
// - n > 2048: Gershgorin circle theorem (O(n^2), approximate)
// - n <= 2048: Power method on A^T*A for lambda_max, Gershgorin for lambda_min

/// Returns (kappa, is_approximate, iterations_used).
pub fn condition_number_estimate(a: &Matrix) -> (f64, bool, usize) {
    assert_eq!(
        a.rows(),
        a.cols(),
        "condition_number requires square matrix, got {}x{}",
        a.rows(),
        a.cols()
    );
    let n = a.rows();

    // Large matrices: Gershgorin estimate (fast, O(n^2))
    if n > 2048 {
        return condition_via_gershgorin(a);
    }

    // Small-to-medium matrices: power method on A^T * A
    // Singular values of A = sqrt(eigenvalues of A^T * A)
    let at = a.transpose();
    let ata = multiply_naive(&at, a);

    // Largest eigenvalue of A^T*A via power method
    let (lambda_max_ata, iters) = spectral_radius_power_method(&ata, 50, 1e-10);

    // Smallest eigenvalue estimate via Gershgorin on A^T*A
    let lambda_min_ata = gershgorin_min_eigenvalue(&ata).max(1e-300);

    let sigma_max = lambda_max_ata.sqrt();
    let sigma_min = lambda_min_ata.sqrt();

    let kappa = if sigma_min > 1e-300 {
        sigma_max / sigma_min
    } else {
        f64::INFINITY
    };

    (kappa, false, iters)
}

/// Gershgorin-only condition estimate for large matrices.
/// Uses ||A||_inf / min Gershgorin lower bound.
fn condition_via_gershgorin(a: &Matrix) -> (f64, bool, usize) {
    let n = a.rows();
    let inf_norm = norm_infinity(a);

    let min_gershgorin = gershgorin_min_eigenvalue(a);

    if min_gershgorin > 1e-15 {
        let kappa = inf_norm / min_gershgorin;
        return (kappa, true, 0);
    }

    // Fallback: kappa ~ ||A||_F^2 / trace(A)^2 * n
    let f_norm = frobenius_norm(a);
    let trace_sq = a.trace().powi(2);
    let kappa = if trace_sq > 1e-15 {
        f_norm * f_norm / trace_sq * n as f64
    } else {
        f64::INFINITY
    };

    (kappa, true, 0)
}

/// Minimum Gershgorin lower bound: min_i(|a_ii| - Sigma_{j!=i} |a_ij|).
/// If positive, all eigenvalues are bounded away from zero.
fn gershgorin_min_eigenvalue(a: &Matrix) -> f64 {
    let n = a.rows();
    (0..n)
        .map(|i| {
            let diag = a.get(i, i).abs();
            let off_diag: f64 = (0..n).filter(|&j| j != i).map(|j| a.get(i, j).abs()).sum();
            (diag - off_diag).max(0.0)
        })
        .fold(f64::INFINITY, f64::min)
}

/// Human-readable assessment of condition number.
pub fn condition_assessment(kappa: f64) -> &'static str {
    if kappa < 10.0 {
        "Well-conditioned"
    } else if kappa < 1e3 {
        "Moderate"
    } else if kappa < 1e6 {
        "Ill-conditioned"
    } else if kappa < 1e12 {
        "Severely ill-conditioned"
    } else {
        "SINGULAR or near-singular"
    }
}

// ─── Matrix Power (Binary Exponentiation) ────────────────────────────────────
//
// Computes A^k using repeated squaring: O(log k) matrix multiplications.
// Example: A^13 = A^8 * A^4 * A^1 (13 = 1101 in binary, 3 multiplications).
// Uses multiply_strassen_padded for each multiplication step.

pub fn matrix_power(
    a: &Matrix,
    k: u64,
    simd: SimdLevel,
    threshold: usize,
) -> Matrix {
    assert_eq!(
        a.rows(),
        a.cols(),
        "matrix_power requires square matrix, got {}x{}",
        a.rows(),
        a.cols()
    );

    if k == 0 {
        return Matrix::identity(a.rows()).expect("identity allocation");
    }
    if k == 1 {
        return a.clone();
    }

    let mut result = Matrix::identity(a.rows()).expect("identity allocation");
    let mut base = a.clone();
    let mut exp = k;

    while exp > 0 {
        if exp & 1 == 1 {
            result = multiply_strassen_padded(&result, &base, threshold, simd).0;
        }
        exp >>= 1;
        if exp > 0 {
            let base_clone = base.clone();
            base = multiply_strassen_padded(&base, &base_clone, threshold, simd).0;
        }
    }

    result
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{EPSILON, STRASSEN_THRESHOLD, SimdLevel};

    #[test]
    fn test_frobenius_known() {
        // [[1,2],[3,4]] -> sqrt(1+4+9+16) = sqrt(30)
        let m = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let f = frobenius_norm(&m);
        assert!(
            (f - 30.0_f64.sqrt()).abs() < 1e-10,
            "frobenius of [[1,2],[3,4]] = {f}, expected {}",
            30.0_f64.sqrt()
        );
    }

    #[test]
    fn test_frobenius_identity() {
        // ||I_n||_F = sqrt(n)
        let n = 10;
        let id = Matrix::identity(n).unwrap();
        let f = frobenius_norm(&id);
        assert!(
            (f - (n as f64).sqrt()).abs() < 1e-10,
            "frobenius of I_{n} = {f}, expected {}",
            (n as f64).sqrt()
        );
    }

    #[test]
    fn test_norm_1_known() {
        // [[1, -2, 3],
        //  [4,  5, -6]]
        // col sums: |1|+|4|=5, |-2|+|5|=7, |3|+|-6|=9
        // norm_1 = 9
        let m = Matrix::from_flat(2, 3, vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]).unwrap();
        let n1 = norm_1(&m);
        assert!(
            (n1 - 9.0).abs() < 1e-10,
            "norm_1 = {n1}, expected 9.0"
        );
    }

    #[test]
    fn test_norm_infinity_known() {
        // [[1, -2, 3],
        //  [4,  5, -6]]
        // row sums: 1+2+3=6, 4+5+6=15
        // norm_inf = 15
        let m = Matrix::from_flat(2, 3, vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]).unwrap();
        let ni = norm_infinity(&m);
        assert!(
            (ni - 15.0).abs() < 1e-10,
            "norm_infinity = {ni}, expected 15.0"
        );
    }

    #[test]
    fn test_spectral_radius_diagonal() {
        // Diagonal matrix with eigenvalues 1, 3, 7
        // Spectral radius = 7
        let m = Matrix::from_flat(
            3,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 7.0],
        )
        .unwrap();
        let (rho, _iters) = spectral_radius_power_method(&m, 100, 1e-12);
        assert!(
            (rho - 7.0).abs() < 1e-6,
            "spectral radius = {rho}, expected 7.0"
        );
    }

    #[test]
    fn test_condition_number_identity() {
        // kappa(I) = 1 (all singular values = 1)
        let id = Matrix::identity(10).unwrap();
        let (kappa, _, _) = condition_number_estimate(&id);
        assert!(
            kappa < 1.1,
            "condition number of I_10 = {kappa}, expected ~1.0"
        );
    }

    #[test]
    fn test_is_symmetric_identity() {
        let id = Matrix::identity(5).unwrap();
        assert!(is_symmetric(&id, 1e-10), "Identity should be symmetric");
    }

    #[test]
    fn test_is_symmetric_nonsymmetric() {
        let m = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(
            !is_symmetric(&m, 1e-10),
            "[[1,2],[3,4]] should not be symmetric"
        );
    }

    #[test]
    fn test_matrix_power_identity() {
        // A^0 = I for any square A
        let a = Matrix::random(5, 5, Some(42)).unwrap();
        let a0 = matrix_power(&a, 0, SimdLevel::Scalar, STRASSEN_THRESHOLD);
        let eye = Matrix::identity(5).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (a0.get(i, j) - eye.get(i, j)).abs() < 1e-10,
                    "A^0[{i},{j}] = {}, expected {}",
                    a0.get(i, j),
                    eye.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_matrix_power_known_2x2() {
        // [[1,2],[3,4]]^2 = [[7,10],[15,22]]
        let a = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a2 = matrix_power(&a, 2, SimdLevel::Scalar, STRASSEN_THRESHOLD);
        assert!((a2.get(0, 0) - 7.0).abs() < 1e-9, "A^2[0,0] = {}", a2.get(0, 0));
        assert!((a2.get(0, 1) - 10.0).abs() < 1e-9, "A^2[0,1] = {}", a2.get(0, 1));
        assert!((a2.get(1, 0) - 15.0).abs() < 1e-9, "A^2[1,0] = {}", a2.get(1, 0));
        assert!((a2.get(1, 1) - 22.0).abs() < 1e-9, "A^2[1,1] = {}", a2.get(1, 1));
    }

    #[test]
    fn test_matrix_power_matches_naive() {
        // A^3 via binary exp should match A * A * A via naive
        let a = Matrix::random(4, 4, Some(77)).unwrap();
        let a2_naive = multiply_naive(&a, &a);
        let a3_naive = multiply_naive(&a2_naive, &a);
        let a3_power = matrix_power(&a, 3, SimdLevel::Scalar, STRASSEN_THRESHOLD);

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a3_power.get(i, j) - a3_naive.get(i, j)).abs() < EPSILON,
                    "A^3 mismatch at [{i},{j}]: power={}, naive={}",
                    a3_power.get(i, j),
                    a3_naive.get(i, j)
                );
            }
        }
    }
}
