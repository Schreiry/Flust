
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
//
// Row-major flat storage with explicit stride for SIMD alignment.
// stride >= cols — each row starts at a SIMD-width-aligned offset.
// Padding elements (cols..stride) in each row are always zero.

/// Round up to nearest multiple of `align` (must be power of 2).
#[inline]
pub(crate) fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// AVX-512 width in f64s — also works for AVX2 (width 4) since 8 is a multiple.
pub(crate) const SIMD_ALIGN: usize = 8;

#[derive(Clone)]
pub struct Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) stride: usize, // row pitch in f64 elements (>= cols)
    pub(crate) data: Vec<f64>,
}

// ─── Constructors ───────────────────────────────────────────────────────────

impl Matrix {
    /// Create a zero-filled matrix. Uses checked_mul for overflow protection.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        let stride = align_up(cols, SIMD_ALIGN);
        let len = rows.checked_mul(stride).ok_or_else(|| {
            MatrixError::Overflow(format!("{rows} × {cols} (stride {stride}) overflows usize"))
        })?;
        Ok(Matrix {
            rows,
            cols,
            stride,
            data: vec![0.0; len],
        })
    }

    /// Create a matrix from a pre-built flat vector. Validates length.
    /// Input data is rows*cols elements (dense); copied into strided layout.
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
        let stride = align_up(cols, SIMD_ALIGN);
        if stride == cols {
            // No padding needed — zero-copy
            Ok(Matrix { rows, cols, stride, data })
        } else {
            // Copy into strided layout with zero-padded row tails
            let mut strided = vec![0.0; rows * stride];
            for i in 0..rows {
                strided[i * stride..i * stride + cols]
                    .copy_from_slice(&data[i * cols..i * cols + cols]);
            }
            Ok(Matrix { rows, cols, stride, data: strided })
        }
    }

    /// Create an n×n identity matrix (1.0 on diagonal, 0.0 elsewhere).
    pub fn identity(n: usize) -> Result<Self, MatrixError> {
        let mut m = Self::zeros(n, n)?;
        for i in 0..n {
            m.data[i * m.stride + i] = 1.0;
        }
        Ok(m)
    }

    /// Create a matrix from pre-built raw parts (already strided layout).
    /// Caller must ensure: data.len() == rows * stride, stride >= cols.
    pub fn from_raw_parts(rows: usize, cols: usize, stride: usize, data: Vec<f64>) -> Self {
        debug_assert!(stride >= cols);
        debug_assert!(data.len() == rows * stride);
        Matrix { rows, cols, stride, data }
    }

    /// Create a matrix filled with random values in [-10.0, 10.0].
    /// With seed: reproducible (SmallRng::seed_from_u64). Without: entropy-based.
    pub fn random(rows: usize, cols: usize, seed: Option<u64>) -> Result<Self, MatrixError> {
        let stride = align_up(cols, SIMD_ALIGN);
        let len = rows.checked_mul(stride).ok_or_else(|| {
            MatrixError::Overflow(format!("{rows} × {cols} overflows usize"))
        })?;
        let mut rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };
        // Fill only the valid cols per row; padding stays zero
        let mut data = vec![0.0; len];
        for i in 0..rows {
            for j in 0..cols {
                data[i * stride + j] = rng.gen_range(-10.0..=10.0);
            }
        }
        Ok(Matrix { rows, cols, stride, data })
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
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline(always)]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Get element at (r, c). Panics on out-of-bounds.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        assert!(r < self.rows && c < self.cols, "Index ({r}, {c}) out of bounds for {}×{}", self.rows, self.cols);
        self.data[r * self.stride + c]
    }

    /// Set element at (r, c). Panics on out-of-bounds.
    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        assert!(r < self.rows && c < self.cols, "Index ({r}, {c}) out of bounds for {}×{}", self.rows, self.cols);
        self.data[r * self.stride + c] = val;
    }

    /// Get element without bounds checking. Caller must guarantee r < rows, c < cols.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, r: usize, c: usize) -> f64 {
        unsafe { *self.data.get_unchecked(r * self.stride + c) }
    }

    /// Set element without bounds checking. Caller must guarantee r < rows, c < cols.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, r: usize, c: usize, val: f64) {
        unsafe { *self.data.get_unchecked_mut(r * self.stride + c) = val; }
    }

    /// Return a slice for row `r` covering only valid columns [0..cols).
    #[inline(always)]
    pub fn row_slice(&self, r: usize) -> &[f64] {
        let start = r * self.stride;
        &self.data[start..start + self.cols]
    }
}

// ─── Operators (borrowed) ───────────────────────────────────────────────────
//
// Row-by-row iteration to correctly handle stride > cols.

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "Cannot add {}×{} and {}×{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let stride = align_up(self.cols, SIMD_ALIGN);
        let mut data = vec![0.0; self.rows * stride];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * stride + j] =
                    self.data[i * self.stride + j] + rhs.data[i * rhs.stride + j];
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            stride,
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
        let stride = align_up(self.cols, SIMD_ALIGN);
        let mut data = vec![0.0; self.rows * stride];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * stride + j] =
                    self.data[i * self.stride + j] - rhs.data[i * rhs.stride + j];
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            stride,
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
        let stride = align_up(self.cols, SIMD_ALIGN);
        let mut data = vec![0.0; self.rows * stride];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * stride + j] =
                    self.data[i * self.stride + j] + rhs.data[i * rhs.stride + j];
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            stride,
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
        let stride = align_up(self.cols, SIMD_ALIGN);
        let mut data = vec![0.0; self.rows * stride];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i * stride + j] =
                    self.data[i * self.stride + j] - rhs.data[i * rhs.stride + j];
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            stride,
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
        let hs = align_up(half, SIMD_ALIGN);
        let mut a11 = vec![0.0; half * hs];
        let mut a12 = vec![0.0; half * hs];
        let mut a21 = vec![0.0; half * hs];
        let mut a22 = vec![0.0; half * hs];

        for i in 0..half {
            for j in 0..half {
                a11[i * hs + j] = self.data[i * self.stride + j];
                a12[i * hs + j] = self.data[i * self.stride + (j + half)];
                a21[i * hs + j] = self.data[(i + half) * self.stride + j];
                a22[i * hs + j] = self.data[(i + half) * self.stride + (j + half)];
            }
        }

        (
            Matrix { rows: half, cols: half, stride: hs, data: a11 },
            Matrix { rows: half, cols: half, stride: hs, data: a12 },
            Matrix { rows: half, cols: half, stride: hs, data: a21 },
            Matrix { rows: half, cols: half, stride: hs, data: a22 },
        )
    }

    /// Reassemble 4 quadrants into a single matrix.
    /// Takes owned matrices — zero-copy move semantics.
    pub fn combine_4(a11: Matrix, a12: Matrix, a21: Matrix, a22: Matrix) -> Matrix {
        let half = a11.rows;
        let n = half * 2;
        let ns = align_up(n, SIMD_ALIGN);
        let mut data = vec![0.0; n * ns];

        for i in 0..half {
            for j in 0..half {
                data[i * ns + j] = a11.data[i * a11.stride + j];
                data[i * ns + (j + half)] = a12.data[i * a12.stride + j];
                data[(i + half) * ns + j] = a21.data[i * a21.stride + j];
                data[(i + half) * ns + (j + half)] = a22.data[i * a22.stride + j];
            }
        }

        Matrix { rows: n, cols: n, stride: ns, data }
    }

    /// Pad matrix with zeros to the nearest power of 2 in both dimensions.
    /// Returns (padded_matrix, original_rows, original_cols).
    pub fn pad_to_power_of_2(&self) -> (Matrix, usize, usize) {
        let new_rows = next_power_of_2(self.rows);
        let new_cols = next_power_of_2(self.cols);
        let target = new_rows.max(new_cols); // square for Strassen
        let ts = align_up(target, SIMD_ALIGN);

        if target == self.rows && target == self.cols && ts == self.stride {
            return (
                Matrix {
                    rows: target,
                    cols: target,
                    stride: ts,
                    data: self.data.clone(),
                },
                self.rows,
                self.cols,
            );
        }

        let mut data = vec![0.0; target * ts];
        for i in 0..self.rows {
            let src_start = i * self.stride;
            let dst_start = i * ts;
            data[dst_start..dst_start + self.cols]
                .copy_from_slice(&self.data[src_start..src_start + self.cols]);
        }

        (Matrix { rows: target, cols: target, stride: ts, data }, self.rows, self.cols)
    }

    /// Remove padding, extracting the top-left orig_rows × orig_cols submatrix.
    /// Consumes self — the padded matrix is not needed after unpadding.
    pub fn unpad(self, orig_rows: usize, orig_cols: usize) -> Matrix {
        if self.rows == orig_rows && self.cols == orig_cols {
            return self;
        }

        let os = align_up(orig_cols, SIMD_ALIGN);
        let mut data = vec![0.0; orig_rows * os];
        for i in 0..orig_rows {
            let src_start = i * self.stride;
            let dst_start = i * os;
            data[dst_start..dst_start + orig_cols]
                .copy_from_slice(&self.data[src_start..src_start + orig_cols]);
        }

        Matrix {
            rows: orig_rows,
            cols: orig_cols,
            stride: os,
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
        let new_stride = align_up(self.rows, SIMD_ALIGN);
        let mut data = vec![0.0; self.cols * new_stride];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * new_stride + i] = self.data[i * self.stride + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            stride: new_stride,
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
        (0..self.rows).map(|i| self.data[i * self.stride + i]).sum()
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
                write!(f, "{:>10.4}", self.data[i * self.stride + j])?;
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
