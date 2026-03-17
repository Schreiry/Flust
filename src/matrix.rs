// matrix.rs — Core matrix data structure with flat Vec<f64> storage.
//
// ARCHITECTURE: Matrix stores elements in a SINGLE flat Vec<f64>.
// Element [i][j] = data[i * cols + j].
//
// Why flat storage matters:
// - Row traversal: data[i*cols .. i*cols+cols] is contiguous in memory.
//   The CPU prefetcher loads these into cache lines ahead of time.
// - AVX2 loads 4 f64 at once from contiguous memory (_mm256_loadu_pd).
// - Vec<Vec<f64>> places each row at an arbitrary heap location = cache miss per row.

use std::fmt;
use std::ops::{Add, Sub};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ─── MatrixError ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum MatrixError {
    DimensionMismatch { expected: String, got: String },
    InvalidDimensions(String),
    Overflow(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            MatrixError::InvalidDimensions(msg) => {
                write!(f, "Invalid dimensions: {msg}")
            }
            MatrixError::Overflow(msg) => {
                write!(f, "Overflow: {msg}")
            }
        }
    }
}

impl std::error::Error for MatrixError {}

// ─── Matrix ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f64>,
}

// ─── Constructors ───────────────────────────────────────────────────────────

impl Matrix {
    /// Create a zero-filled matrix. Uses checked_mul for overflow protection.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        let len = rows.checked_mul(cols).ok_or_else(|| {
            MatrixError::Overflow(format!("{rows} × {cols} overflows usize"))
        })?;
        Ok(Matrix {
            rows,
            cols,
            data: vec![0.0; len],
        })
    }

    /// Create a matrix from a pre-built flat vector. Validates length.
    pub fn from_flat(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            MatrixError::Overflow(format!("{rows} × {cols} overflows usize"))
        })?;
        if data.len() != expected {
            return Err(MatrixError::DimensionMismatch {
                expected: format!("{rows}×{cols} = {expected} elements"),
                got: format!("{} elements", data.len()),
            });
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Create an n×n identity matrix (1.0 on diagonal, 0.0 elsewhere).
    pub fn identity(n: usize) -> Result<Self, MatrixError> {
        let mut m = Self::zeros(n, n)?;
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        Ok(m)
    }

    /// Create a matrix filled with random values in [-10.0, 10.0].
    /// With seed: reproducible (SmallRng::seed_from_u64). Without: entropy-based.
    pub fn random(rows: usize, cols: usize, seed: Option<u64>) -> Result<Self, MatrixError> {
        let len = rows.checked_mul(cols).ok_or_else(|| {
            MatrixError::Overflow(format!("{rows} × {cols} overflows usize"))
        })?;
        let mut rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };
        let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-10.0..=10.0)).collect();
        Ok(Matrix { rows, cols, data })
    }
}

// ─── Accessors ──────────────────────────────────────────────────────────────

impl Matrix {
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get element at (r, c). Panics on out-of-bounds.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        assert!(r < self.rows && c < self.cols, "Index ({r}, {c}) out of bounds for {}×{}", self.rows, self.cols);
        self.data[r * self.cols + c]
    }

    /// Set element at (r, c). Panics on out-of-bounds.
    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        assert!(r < self.rows && c < self.cols, "Index ({r}, {c}) out of bounds for {}×{}", self.rows, self.cols);
        self.data[r * self.cols + c] = val;
    }

    /// Get element without bounds checking. Caller must guarantee r < rows, c < cols.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, r: usize, c: usize) -> f64 {
        unsafe { *self.data.get_unchecked(r * self.cols + c) }
    }

    /// Set element without bounds checking. Caller must guarantee r < rows, c < cols.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, r: usize, c: usize, val: f64) {
        unsafe { *self.data.get_unchecked_mut(r * self.cols + c) = val; }
    }
}

// ─── Operators (borrowed) ───────────────────────────────────────────────────
//
// iter().zip().map().collect() — the compiler auto-vectorizes this into SIMD
// instructions at opt-level=3, so we get hardware acceleration for free.

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "Cannot add {}×{} and {}×{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data: Vec<f64> = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a + b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "Cannot subtract {}×{} and {}×{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data: Vec<f64> = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// ─── Operators (owned) ──────────────────────────────────────────────────────
//
// Owned versions for Strassen: consume both operands, avoiding clones
// when the source matrices are temporaries that won't be used again.

impl Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "Cannot add {}×{} and {}×{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data: Vec<f64> = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a + b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "Cannot subtract {}×{} and {}×{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data: Vec<f64> = self.data.iter().zip(rhs.data.iter()).map(|(a, b)| a - b).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// ─── Strassen helpers ───────────────────────────────────────────────────────

impl Matrix {
    /// Split an n×n matrix into 4 quadrants of (n/2)×(n/2).
    /// Requires: square matrix with even dimensions.
    /// Returns (A11, A12, A21, A22) — top-left, top-right, bottom-left, bottom-right.
    pub fn split_4(&self) -> (Matrix, Matrix, Matrix, Matrix) {
        assert_eq!(self.rows, self.cols, "split_4 requires a square matrix, got {}×{}", self.rows, self.cols);
        assert!(self.rows % 2 == 0, "split_4 requires even dimensions, got {}", self.rows);

        let half = self.rows / 2;
        let mut a11 = vec![0.0; half * half];
        let mut a12 = vec![0.0; half * half];
        let mut a21 = vec![0.0; half * half];
        let mut a22 = vec![0.0; half * half];

        for i in 0..half {
            for j in 0..half {
                a11[i * half + j] = self.data[i * self.cols + j];
                a12[i * half + j] = self.data[i * self.cols + (j + half)];
                a21[i * half + j] = self.data[(i + half) * self.cols + j];
                a22[i * half + j] = self.data[(i + half) * self.cols + (j + half)];
            }
        }

        (
            Matrix { rows: half, cols: half, data: a11 },
            Matrix { rows: half, cols: half, data: a12 },
            Matrix { rows: half, cols: half, data: a21 },
            Matrix { rows: half, cols: half, data: a22 },
        )
    }

    /// Reassemble 4 quadrants into a single matrix.
    /// Takes owned matrices — zero-copy move semantics.
    pub fn combine_4(a11: Matrix, a12: Matrix, a21: Matrix, a22: Matrix) -> Matrix {
        let half = a11.rows;
        let n = half * 2;
        let mut data = vec![0.0; n * n];

        for i in 0..half {
            for j in 0..half {
                data[i * n + j] = a11.data[i * half + j];
                data[i * n + (j + half)] = a12.data[i * half + j];
                data[(i + half) * n + j] = a21.data[i * half + j];
                data[(i + half) * n + (j + half)] = a22.data[i * half + j];
            }
        }

        Matrix { rows: n, cols: n, data }
    }

    /// Pad matrix with zeros to the nearest power of 2 in both dimensions.
    /// Returns (padded_matrix, original_rows, original_cols).
    pub fn pad_to_power_of_2(&self) -> (Matrix, usize, usize) {
        let new_rows = next_power_of_2(self.rows);
        let new_cols = next_power_of_2(self.cols);
        let target = new_rows.max(new_cols); // square for Strassen

        if target == self.rows && target == self.cols {
            // Already correct size — clone to return owned copy
            return (
                Matrix {
                    rows: target,
                    cols: target,
                    data: self.data.clone(),
                },
                self.rows,
                self.cols,
            );
        }

        let mut data = vec![0.0; target * target];
        for i in 0..self.rows {
            let src_start = i * self.cols;
            let dst_start = i * target;
            data[dst_start..dst_start + self.cols]
                .copy_from_slice(&self.data[src_start..src_start + self.cols]);
        }

        (Matrix { rows: target, cols: target, data }, self.rows, self.cols)
    }

    /// Remove padding, extracting the top-left orig_rows × orig_cols submatrix.
    /// Consumes self — the padded matrix is not needed after unpadding.
    pub fn unpad(self, orig_rows: usize, orig_cols: usize) -> Matrix {
        if self.rows == orig_rows && self.cols == orig_cols {
            return self;
        }

        let mut data = vec![0.0; orig_rows * orig_cols];
        for i in 0..orig_rows {
            let src_start = i * self.cols;
            let dst_start = i * orig_cols;
            data[dst_start..dst_start + orig_cols]
                .copy_from_slice(&self.data[src_start..src_start + orig_cols]);
        }

        Matrix {
            rows: orig_rows,
            cols: orig_cols,
            data,
        }
    }
}

// ─── Numerical helpers ─────────────────────────────────────────────────────

impl Matrix {
    /// Transpose: swap rows and columns.
    /// For m×n matrix A, returns n×m matrix B where B[j][i] = A[i][j].
    /// O(m×n) — single pass, cache-friendly write pattern.
    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Trace: sum of diagonal elements. Only defined for square matrices.
    /// tr(A) = Σ_i a_ii. Used in eigenvalue estimates and condition number bounds.
    pub fn trace(&self) -> f64 {
        assert_eq!(
            self.rows, self.cols,
            "trace requires square matrix, got {}×{}",
            self.rows, self.cols
        );
        (0..self.rows).map(|i| self.data[i * self.cols + i]).sum()
    }
}

/// Round up to the nearest power of 2. Returns 1 for input 0.
pub(crate) fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

// ─── Display ────────────────────────────────────────────────────────────────

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_display = 8;
        let show_rows = self.rows.min(max_display);
        let show_cols = self.cols.min(max_display);

        for i in 0..show_rows {
            write!(f, "│")?;
            for j in 0..show_cols {
                write!(f, "{:>10.4}", self.data[i * self.cols + j])?;
            }
            if self.cols > max_display {
                write!(f, "  ...")?;
            }
            writeln!(f, " │")?;
        }

        if self.rows > max_display {
            writeln!(f, "│ (... {}×{} matrix ...) │", self.rows, self.cols)?;
        }

        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_matrix() {
        let m = Matrix::zeros(3, 3).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);
        for val in m.data() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_identity() {
        let m = Matrix::identity(4).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(m.get(i, j), 1.0, "Diagonal [{i},{j}] should be 1.0");
                } else {
                    assert_eq!(m.get(i, j), 0.0, "Off-diagonal [{i},{j}] should be 0.0");
                }
            }
        }
    }

    #[test]
    fn test_add() {
        // [1 2]   [5 6]   [ 6  8]
        // [3 4] + [7 8] = [10 12]
        let a = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_flat(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = &a + &b;
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_sub() {
        // [5 6]   [1 2]   [4 4]
        // [7 8] - [3 4] = [4 4]
        let a = Matrix::from_flat(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let b = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = &a - &b;
        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 4.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn test_pad_unpad_roundtrip() {
        // 5×5 with known values → pad to 8×8 → unpad back to 5×5 → values preserved
        let mut original = Matrix::zeros(5, 5).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                original.set(i, j, (i * 5 + j) as f64);
            }
        }

        let (padded, orig_r, orig_c) = original.pad_to_power_of_2();
        assert_eq!(padded.rows(), 8);
        assert_eq!(padded.cols(), 8);
        assert_eq!(orig_r, 5);
        assert_eq!(orig_c, 5);

        // Verify padding is zeros
        for i in 0..8 {
            for j in 0..8 {
                if i < 5 && j < 5 {
                    assert_eq!(padded.get(i, j), (i * 5 + j) as f64);
                } else {
                    assert_eq!(padded.get(i, j), 0.0, "Padding at [{i},{j}] should be 0.0");
                }
            }
        }

        let restored = padded.unpad(orig_r, orig_c);
        assert_eq!(restored.rows(), 5);
        assert_eq!(restored.cols(), 5);
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    restored.get(i, j),
                    (i * 5 + j) as f64,
                    "Restored[{i},{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn test_split_combine_roundtrip() {
        // 4×4 → split into four 2×2 → combine back → identical to original
        let original = Matrix::from_flat(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap();

        let (a11, a12, a21, a22) = original.split_4();

        // Verify quadrants
        assert_eq!(a11.get(0, 0), 1.0);
        assert_eq!(a11.get(0, 1), 2.0);
        assert_eq!(a11.get(1, 0), 5.0);
        assert_eq!(a11.get(1, 1), 6.0);

        assert_eq!(a12.get(0, 0), 3.0);
        assert_eq!(a12.get(0, 1), 4.0);

        assert_eq!(a21.get(0, 0), 9.0);
        assert_eq!(a21.get(0, 1), 10.0);

        assert_eq!(a22.get(0, 0), 11.0);
        assert_eq!(a22.get(1, 1), 16.0);

        let restored = Matrix::combine_4(a11, a12, a21, a22);
        assert_eq!(restored.rows(), 4);
        assert_eq!(restored.cols(), 4);

        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    restored.get(i, j),
                    original.get(i, j),
                    "Mismatch at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_overflow_protection() {
        let result = Matrix::zeros(usize::MAX, 2);
        assert!(result.is_err(), "Should fail on overflow");
        if let Err(MatrixError::Overflow(_)) = result {
            // expected
        } else {
            panic!("Expected MatrixError::Overflow");
        }
    }

    #[test]
    fn test_transpose_known() {
        // 2×3 → 3×2
        let a = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let at = a.transpose();
        assert_eq!(at.rows(), 3);
        assert_eq!(at.cols(), 2);
        assert_eq!(at.get(0, 0), 1.0);
        assert_eq!(at.get(0, 1), 4.0);
        assert_eq!(at.get(1, 0), 2.0);
        assert_eq!(at.get(1, 1), 5.0);
        assert_eq!(at.get(2, 0), 3.0);
        assert_eq!(at.get(2, 1), 6.0);
    }

    #[test]
    fn test_transpose_roundtrip() {
        let a = Matrix::random(7, 11, Some(42)).unwrap();
        let att = a.transpose().transpose();
        assert_eq!(att.rows(), a.rows());
        assert_eq!(att.cols(), a.cols());
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                assert_eq!(att.get(i, j), a.get(i, j), "Mismatch at [{i},{j}]");
            }
        }
    }

    #[test]
    fn test_trace_identity() {
        let id = Matrix::identity(5).unwrap();
        assert!((id.trace() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_trace_known() {
        // [[1,2],[3,4]] → trace = 1 + 4 = 5
        let m = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((m.trace() - 5.0).abs() < 1e-15);
    }
}
