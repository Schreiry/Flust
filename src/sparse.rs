// ─── Sparse Matrix Module ──────────────────────────────────────────────────
//
// Pure-Rust sparse matrix types (COO, CSR) + optional Intel MKL acceleration.
//
// All MKL FFI declarations live here — single source of truth.
// Dense MKL routines (cblas_dgemm) are re-exported for algorithms.rs.
// Sparse MKL routines (mkl_sparse_d_*) power the thermal simulation.
//
// Linking is handled by build.rs (searches for mkl_rt on the system).
// The `intel-mkl-src` crate is NOT used — it conflicts with manual #[link]
// by pulling in component libraries alongside the runtime dispatcher.

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

// ─── Intel MKL FFI Bindings ─────────────────────────────────────────────────
//
// ALL MKL extern declarations live here. Other modules import from sparse.rs.
// build.rs handles linking (emits `cargo:rustc-link-lib=dylib=mkl_rt`),
// so NO #[link] attributes are needed on the extern blocks below.
//
// Interface: ILP32 (MKL_INT = i32 = c_int). This is the default on all
// platforms and matches the LP64 interface layer where MKL_INT = int.

#[cfg(feature = "mkl")]
pub mod mkl_ffi {
    use std::os::raw::c_int;

    // ── Opaque Types ────────────────────────────────────────────────────

    /// Opaque MKL sparse matrix handle (pointer-sized, created by MKL).
    #[repr(C)]
    pub struct MklSparseMatrixOpaque {
        _opaque: [u8; 0],
    }
    pub type SparseMatrixT = *mut MklSparseMatrixOpaque;

    /// Matrix descriptor for sparse operations.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MatrixDescr {
        pub type_: c_int,
        pub mode: c_int,
        pub diag: c_int,
    }

    // ── Constants ────────────────────────────────────────────────────────

    pub const SPARSE_INDEX_BASE_ZERO: c_int = 0;
    pub const SPARSE_OPERATION_NON_TRANSPOSE: c_int = 10;
    pub const SPARSE_MATRIX_TYPE_GENERAL: c_int = 20;
    pub const SPARSE_STATUS_SUCCESS: c_int = 0;

    // Hint constants for mkl_sparse_set_mv_hint
    pub const SPARSE_OPERATION_NON_TRANSPOSE_HINT: c_int = 10;

    // CBLAS constants for dense DGEMM
    pub const CBLAS_ROW_MAJOR: c_int = 101;
    pub const CBLAS_NO_TRANS: c_int = 111;

    // ── Extern Declarations ─────────────────────────────────────────────
    //
    // No #[link] attribute — build.rs emits the link directive.
    // This avoids the double-init crash caused by mixing #[link(name="mkl_rt")]
    // with the intel-mkl-src crate's component-library linking.

    unsafe extern "C" {
        // ── Sparse Inspector-Executor API ───────────────────────────────

        /// Create CSR sparse matrix handle.
        ///
        /// MKL does NOT copy the data — it stores pointers to our buffers.
        /// The caller MUST keep rows_start, rows_end, col_indx, values alive
        /// for the entire lifetime of the handle.
        ///
        /// rows_start[i] = row_ptr[i]   (start of row i)
        /// rows_end[i]   = row_ptr[i+1] (end of row i)
        /// This is the split-pointer format, NOT a single row_ptr array.
        pub fn mkl_sparse_d_create_csr(
            a: *mut SparseMatrixT,
            indexing: c_int,
            rows: c_int,
            cols: c_int,
            rows_start: *const c_int,
            rows_end: *const c_int,
            col_indx: *const c_int,
            values: *mut f64,
        ) -> c_int;

        /// Provide a hint about planned operations so MKL can optimize.
        /// Must be called BEFORE mkl_sparse_optimize().
        ///
        /// expected_calls: how many times d_mv will be called with this handle.
        /// For thermal simulation this is total_steps (hundreds to thousands).
        pub fn mkl_sparse_set_mv_hint(
            a: SparseMatrixT,
            operation: c_int,
            descr: MatrixDescr,
            expected_calls: c_int,
        ) -> c_int;

        /// Run the Inspector phase: analyze sparsity pattern and build
        /// optimized internal data structures for subsequent Executor calls.
        ///
        /// This is where MKL examines the CSR structure and plans memory
        /// access patterns. WITHOUT this call, the first d_mv triggers
        /// a lazy analysis that can crash on certain MKL versions.
        pub fn mkl_sparse_optimize(a: SparseMatrixT) -> c_int;

        /// Sparse matrix–vector multiply (Executor phase):
        ///   y = alpha * op(A) * x + beta * y
        pub fn mkl_sparse_d_mv(
            operation: c_int,
            alpha: f64,
            a: SparseMatrixT,
            descr: MatrixDescr,
            x: *const f64,
            beta: f64,
            y: *mut f64,
        ) -> c_int;

        /// Release all internal data associated with the sparse handle.
        pub fn mkl_sparse_destroy(a: SparseMatrixT) -> c_int;

        // ── Dense BLAS (CBLAS interface) ────────────────────────────────

        /// Dense matrix multiply: C = alpha*A*B + beta*C (row-major).
        pub fn cblas_dgemm(
            layout: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f64,
            a: *const f64,
            lda: c_int,
            b: *const f64,
            ldb: c_int,
            beta: f64,
            c: *mut f64,
            ldc: c_int,
        );

        // ── Threading Control ───────────────────────────────────────────

        /// Set the number of threads MKL uses for internal parallelism.
        /// Call this to prevent oversubscription with rayon.
        pub fn mkl_set_num_threads(n: c_int);
    }
}

// ─── MKL Sparse Handle (RAII Wrapper) ───────────────────────────────────────
//
// Owns the MKL opaque handle AND the i32 buffers that MKL borrows.
// MKL does NOT copy data in mkl_sparse_d_create_csr — it stores raw pointers
// to our Vec<i32>/Vec<f64> buffers. If those Vecs drop before the handle,
// MKL dereferences freed memory → segfault.
//
// This struct ensures correct lifetime: buffers live as long as the handle.
// Drop calls mkl_sparse_destroy() to free MKL's internal structures.

#[cfg(feature = "mkl")]
pub struct MklSparseHandle {
    handle: mkl_ffi::SparseMatrixT,
    descr: mkl_ffi::MatrixDescr,
    // Buffers MKL borrows — must outlive handle.
    _row_start: Vec<i32>,
    _row_end: Vec<i32>,
    _col_idx: Vec<i32>,
    _values: Vec<f64>,
}

#[cfg(feature = "mkl")]
impl MklSparseHandle {
    /// Convert a CsrMatrix into an optimized MKL sparse handle.
    ///
    /// Full Inspector-Executor lifecycle:
    ///   1. mkl_sparse_d_create_csr()  — create handle from CSR arrays
    ///   2. mkl_sparse_set_mv_hint()   — declare planned operation pattern
    ///   3. mkl_sparse_optimize()      — Inspector: analyze sparsity, build plan
    ///
    /// After this, spmv_into() calls the fast Executor path.
    ///
    /// `expected_calls`: how many times spmv_into will be called.
    /// Pass total_steps from the thermal simulation config.
    pub fn from_csr(csr: &CsrMatrix, expected_calls: i32) -> anyhow::Result<Self> {
        // ── Step 0: Limit MKL internal threads ──────────────────────────
        // Thermal time-stepping is sequential (each step depends on previous),
        // but MKL can parallelize WITHIN a single SpMV. Cap at 8 to avoid
        // oversubscription when rayon is also active.
        let num_threads = rayon::current_num_threads().min(8) as i32;
        unsafe { mkl_ffi::mkl_set_num_threads(num_threads); }

        let rows = csr.rows as i32;
        let cols = csr.cols as i32;

        // ── Step 1: Convert usize → i32 (ILP32 interface) ──────────────
        //
        // CsrMatrix uses Vec<usize> (u64 on 64-bit). MKL_INT = i32.
        // Passing usize pointers to MKL would make it read 8-byte values
        // where it expects 4-byte → corrupted indices → crash.
        //
        // Split row_ptr[n+1] into two arrays of length n:
        //   rows_start[i] = row_ptr[i]      (first nnz of row i)
        //   rows_end[i]   = row_ptr[i+1]    (one past last nnz of row i)
        // This is MKL's "4-array" CSR variant.
        let row_start: Vec<i32> = csr.row_ptr[..csr.rows].iter().map(|&v| v as i32).collect();
        let row_end: Vec<i32> = csr.row_ptr[1..].iter().map(|&v| v as i32).collect();
        let col_idx: Vec<i32> = csr.col_idx.iter().map(|&v| v as i32).collect();
        let mut values: Vec<f64> = csr.values.clone();

        let descr = mkl_ffi::MatrixDescr {
            type_: mkl_ffi::SPARSE_MATRIX_TYPE_GENERAL,
            mode: 0,
            diag: 0,
        };

        // ── Step 2: Create handle ───────────────────────────────────────
        let mut handle: mkl_ffi::SparseMatrixT = std::ptr::null_mut();
        let status = unsafe {
            mkl_ffi::mkl_sparse_d_create_csr(
                &mut handle,
                mkl_ffi::SPARSE_INDEX_BASE_ZERO,
                rows,
                cols,
                row_start.as_ptr(),
                row_end.as_ptr(),
                col_idx.as_ptr(),
                values.as_mut_ptr(),
            )
        };
        anyhow::ensure!(
            status == mkl_ffi::SPARSE_STATUS_SUCCESS,
            "mkl_sparse_d_create_csr failed (status={status})"
        );
        anyhow::ensure!(
            !handle.is_null(),
            "mkl_sparse_d_create_csr returned null handle"
        );

        // ── Step 3: Set MV hint ─────────────────────────────────────────
        // Tell MKL we plan to call d_mv many times with NON_TRANSPOSE.
        // This lets the Inspector choose optimal data structures.
        let status = unsafe {
            mkl_ffi::mkl_sparse_set_mv_hint(
                handle,
                mkl_ffi::SPARSE_OPERATION_NON_TRANSPOSE,
                descr,
                expected_calls.max(1),
            )
        };
        if status != mkl_ffi::SPARSE_STATUS_SUCCESS {
            // Non-fatal: hint failure just means suboptimal performance.
            // Log but continue.
            eprintln!("[MKL] mkl_sparse_set_mv_hint returned status={status} (non-fatal)");
        }

        // ── Step 4: Optimize (Inspector phase) ──────────────────────────
        // This is the critical call that was MISSING in the original code.
        // Without it, MKL defers analysis to the first d_mv call, which
        // can crash on certain MKL versions when internal pointers are
        // not yet initialized.
        let status = unsafe {
            mkl_ffi::mkl_sparse_optimize(handle)
        };
        anyhow::ensure!(
            status == mkl_ffi::SPARSE_STATUS_SUCCESS,
            "mkl_sparse_optimize failed (status={status})"
        );

        Ok(MklSparseHandle {
            handle,
            descr,
            _row_start: row_start,
            _row_end: row_end,
            _col_idx: col_idx,
            _values: values,
        })
    }

    /// y = A * x  (alpha=1, beta=0 — pure multiply, no accumulate).
    ///
    /// This is the Executor phase — uses the optimized plan built by
    /// mkl_sparse_optimize() in from_csr(). Very fast for repeated calls.
    pub fn spmv_into(&self, x: &[f64], y: &mut [f64]) -> anyhow::Result<()> {
        let status = unsafe {
            mkl_ffi::mkl_sparse_d_mv(
                mkl_ffi::SPARSE_OPERATION_NON_TRANSPOSE,
                1.0,
                self.handle,
                self.descr,
                x.as_ptr(),
                0.0,
                y.as_mut_ptr(),
            )
        };
        anyhow::ensure!(
            status == mkl_ffi::SPARSE_STATUS_SUCCESS,
            "mkl_sparse_d_mv failed (status={status})"
        );
        Ok(())
    }
}

#[cfg(feature = "mkl")]
impl Drop for MklSparseHandle {
    fn drop(&mut self) {
        unsafe {
            mkl_ffi::mkl_sparse_destroy(self.handle);
        }
    }
}

// MklSparseHandle stores raw pointers internally (via MKL's opaque handle).
// MKL's Inspector-Executor API is thread-safe for read-only Executor calls
// (mkl_sparse_d_mv) after the Inspector phase completes.
#[cfg(feature = "mkl")]
unsafe impl Send for MklSparseHandle {}

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

    /// MKL SpMV test: 3×3 diagonal matrix.
    ///
    /// A = diag(2, 3, 5), x = [1, 2, 3]
    /// Expected: y = [2, 6, 15]
    ///
    /// This test runs only when compiled with `--features mkl` AND MKL is
    /// available at runtime. If MKL is not installed, the test is skipped.
    #[test]
    #[cfg(feature = "mkl")]
    fn test_mkl_spmv_3x3_diagonal() {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 3.0);
        coo.push(2, 2, 5.0);
        let csr = coo.to_csr();

        let handle = MklSparseHandle::from_csr(&csr, 1)
            .expect("MKL sparse handle creation failed");

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        handle.spmv_into(&x, &mut y)
            .expect("MKL SpMV failed");

        assert!((y[0] - 2.0).abs() < 1e-12, "y[0]={}, expected 2.0", y[0]);
        assert!((y[1] - 6.0).abs() < 1e-12, "y[1]={}, expected 6.0", y[1]);
        assert!((y[2] - 15.0).abs() < 1e-12, "y[2]={}, expected 15.0", y[2]);
    }

    /// MKL SpMV test: tridiagonal matrix (Laplacian-like).
    ///
    /// Verifies MKL result matches the pure-Rust spmv() for a non-trivial
    /// sparsity pattern similar to what the thermal simulation produces.
    #[test]
    #[cfg(feature = "mkl")]
    fn test_mkl_spmv_tridiagonal() {
        let n = 10;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0 { coo.push(i, i - 1, -1.0); }
            if i < n - 1 { coo.push(i, i + 1, -1.0); }
        }
        let csr = coo.to_csr();

        // Ground truth: pure-Rust SpMV
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let y_rust = csr.spmv(&x);

        // MKL SpMV
        let handle = MklSparseHandle::from_csr(&csr, 1)
            .expect("MKL handle creation failed");
        let mut y_mkl = vec![0.0; n];
        handle.spmv_into(&x, &mut y_mkl)
            .expect("MKL SpMV failed");

        for i in 0..n {
            assert!(
                (y_mkl[i] - y_rust[i]).abs() < 1e-10,
                "Mismatch at [{i}]: MKL={}, Rust={}",
                y_mkl[i], y_rust[i]
            );
        }
    }
}
