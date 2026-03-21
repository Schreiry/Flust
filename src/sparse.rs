// ─── Sparse Matrix Module ──────────────────────────────────────────────────
//
// COO (Coordinate) and CSR (Compressed Sparse Row) formats.
// Used by the thermal solver for the finite-difference transition matrix.
//
// The 3D heat equation stencil produces a matrix with at most 7 nonzeros
// per row (1 center + 6 neighbors). For a 32×32×32 grid (32768 nodes),
// the full dense matrix would be 32768² = 1 billion entries.
// In CSR: ~230K nonzeros — a 4500× compression.

/// COO (Coordinate) sparse matrix — easy to build, then convert to CSR.
pub struct CooMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>, // (row, col, value)
}

impl CooMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.rows, "row {row} >= rows {}", self.rows);
        debug_assert!(col < self.cols, "col {col} >= cols {}", self.cols);
        self.entries.push((row, col, value));
    }

    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Convert COO → CSR. Sorts entries by (row, col) and merges duplicates
    /// by summing their values.
    pub fn to_csr(&self) -> CsrMatrix {
        let n = self.rows;
        let m = self.cols;

        if self.entries.is_empty() {
            return CsrMatrix {
                rows: n,
                cols: m,
                row_ptr: vec![0; n + 1],
                col_idx: Vec::new(),
                values: Vec::new(),
            };
        }

        // Sort by (row, col)
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Merge duplicates and build CSR arrays
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());

        let mut prev_row = usize::MAX;
        let mut prev_col = usize::MAX;

        for &(r, c, v) in &sorted {
            if r == prev_row && c == prev_col {
                // Duplicate: sum into last entry
                *values.last_mut().unwrap() += v;
            } else {
                col_idx.push(c);
                values.push(v);
                // Count entries per row
                row_ptr[r + 1] += 1;
                prev_row = r;
                prev_col = c;
            }
        }

        // Prefix sum to get row pointers
        for i in 1..=n {
            row_ptr[i] += row_ptr[i - 1];
        }

        CsrMatrix {
            rows: n,
            cols: m,
            row_ptr,
            col_idx,
            values,
        }
    }
}

/// CSR (Compressed Sparse Row) sparse matrix — efficient for SpMV.
///
/// Storage: row_ptr[rows+1], col_idx[nnz], values[nnz].
/// Row i has entries in col_idx[row_ptr[i]..row_ptr[i+1]].
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Number of stored nonzero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector multiply: y = A * x.
    /// Allocates and returns the result vector.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols, "spmv: x.len()={} != cols={}", x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        self.spmv_into(x, &mut y);
        y
    }

    /// Sparse matrix-vector multiply into pre-allocated buffer: y = A * x.
    /// Avoids allocation — use in hot loops.
    pub fn spmv_into(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[i] = sum;
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_spmv() {
        // I * x == x
        let n = 5;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 1.0);
        }
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), n);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = csr.spmv(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_known_3x3() {
        // A = [[2, -1, 0],
        //      [-1, 2, -1],
        //      [0, -1, 2]]
        // x = [1, 2, 3]
        // y = [0, 0, 4]
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 0, -1.0);
        coo.push(1, 1, 2.0);
        coo.push(1, 2, -1.0);
        coo.push(2, 1, -1.0);
        coo.push(2, 2, 2.0);

        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 7);
        assert_eq!(csr.rows, 3);
        assert_eq!(csr.cols, 3);

        let x = vec![1.0, 2.0, 3.0];
        let y = csr.spmv(&x);
        assert!((y[0] - 0.0).abs() < 1e-12);
        assert!((y[1] - 0.0).abs() < 1e-12);
        assert!((y[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_coo_duplicate_merge() {
        // Push duplicate entries — they should be summed
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(0, 0, 2.0); // duplicate: should sum to 3.0
        coo.push(1, 1, 5.0);

        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 2); // merged to 2 entries

        let x = vec![1.0, 1.0];
        let y = csr.spmv(&x);
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_empty_matrix() {
        let coo = CooMatrix::new(3, 3);
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.row_ptr, vec![0, 0, 0, 0]);

        let x = vec![1.0, 2.0, 3.0];
        let y = csr.spmv(&x);
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_spmv_into_reuses_buffer() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 3.0);
        coo.push(1, 1, 7.0);
        let csr = coo.to_csr();

        let x = vec![2.0, 4.0];
        let mut y = vec![0.0; 2];
        csr.spmv_into(&x, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-12);
        assert!((y[1] - 28.0).abs() < 1e-12);

        // Re-use buffer — should overwrite, not accumulate
        let x2 = vec![1.0, 1.0];
        csr.spmv_into(&x2, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_large_diagonal() {
        // 1000×1000 diagonal with values 1..=1000
        let n = 1000;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, (i + 1) as f64);
        }
        let csr = coo.to_csr();
        assert_eq!(csr.nnz(), n);

        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y = csr.spmv(&x);
        for i in 0..n {
            let expected = (i + 1) as f64 * i as f64;
            assert!(
                (y[i] - expected).abs() < 1e-10,
                "y[{i}]={} expected {expected}",
                y[i]
            );
        }
    }

    #[test]
    fn test_row_ptr_structure() {
        // Verify CSR row_ptr is correct for a specific pattern
        let mut coo = CooMatrix::new(3, 4);
        // Row 0: 2 entries
        coo.push(0, 0, 1.0);
        coo.push(0, 3, 2.0);
        // Row 1: 0 entries (empty row)
        // Row 2: 1 entry
        coo.push(2, 1, 3.0);

        let csr = coo.to_csr();
        assert_eq!(csr.row_ptr, vec![0, 2, 2, 3]);
        assert_eq!(csr.col_idx, vec![0, 3, 1]);
        assert!((csr.values[0] - 1.0).abs() < 1e-12);
        assert!((csr.values[1] - 2.0).abs() < 1e-12);
        assert!((csr.values[2] - 3.0).abs() < 1e-12);
    }
}
