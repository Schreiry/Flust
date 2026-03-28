// ─── Startup Self-Tests ──────────────────────────────────────────────────────
//
// Quick diagnostics run at program launch to verify core modules work.
// Each test is wrapped in catch_unwind — a failure is reported but never
// crashes the startup sequence.

use crate::algorithms;
use crate::common::SimdLevel;
use crate::matrix::Matrix;

// ─── Types ──────────────────────────────────────────────────────────────────

pub enum TestOutcome {
    Pass,
    Warn(String),
    Skip(String),
    Fail(String),
}

impl TestOutcome {
    pub fn label(&self) -> &str {
        match self {
            TestOutcome::Pass => "PASS",
            TestOutcome::Warn(_) => "WARN",
            TestOutcome::Skip(_) => "SKIP",
            TestOutcome::Fail(_) => "FAIL",
        }
    }
}

pub struct SelfTestReport {
    pub matrix_ops: TestOutcome,
    pub simd_kernels: TestOutcome,
    pub mkl_available: TestOutcome,
}

// ─── Runner ─────────────────────────────────────────────────────────────────

pub fn run_self_tests(simd_level: SimdLevel) -> SelfTestReport {
    SelfTestReport {
        matrix_ops: test_matrix_ops(),
        simd_kernels: test_simd_kernels(simd_level),
        mkl_available: test_mkl(),
    }
}

// ─── Test 1: Matrix Operations ──────────────────────────────────────────────
//
// Multiply two small known matrices with naive algorithm, compare result.

fn test_matrix_ops() -> TestOutcome {
    let result = std::panic::catch_unwind(|| {
        // 2×3 * 3×2 → 2×2
        let a = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test matrix A");
        let b = Matrix::from_flat(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("test matrix B");

        let c = algorithms::multiply_naive(&a, &b);

        // Expected: C[0,0]=58, C[0,1]=64, C[1,0]=139, C[1,1]=154
        let eps = 1e-12;
        assert!((c.get(0, 0) - 58.0).abs() < eps, "C[0,0] mismatch");
        assert!((c.get(0, 1) - 64.0).abs() < eps, "C[0,1] mismatch");
        assert!((c.get(1, 0) - 139.0).abs() < eps, "C[1,0] mismatch");
        assert!((c.get(1, 1) - 154.0).abs() < eps, "C[1,1] mismatch");
    });

    match result {
        Ok(()) => TestOutcome::Pass,
        Err(e) => {
            let msg = crate::interactive::extract_panic_message(e);
            TestOutcome::Fail(format!("Matrix ops: {msg}"))
        }
    }
}

// ─── Test 2: SIMD Kernels ───────────────────────────────────────────────────
//
// Run tiled multiply with the detected SIMD level on a small matrix,
// compare against naive.

fn test_simd_kernels(simd: SimdLevel) -> TestOutcome {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let n = 16; // Small enough to be fast, large enough for tiling
        let a = Matrix::random(n, n, Some(7777)).expect("test matrix A");
        let b = Matrix::random(n, n, Some(8888)).expect("test matrix B");

        let naive = algorithms::multiply_naive(&a, &b);
        let (strassen, _, _) = algorithms::multiply_strassen_padded(
            &a, &b, crate::common::STRASSEN_THRESHOLD, simd,
        );

        // Compare element-by-element
        let eps = 1e-9;
        for i in 0..n {
            for j in 0..n {
                let diff = (naive.get(i, j) - strassen.get(i, j)).abs();
                assert!(
                    diff < eps,
                    "SIMD mismatch at [{i},{j}]: naive={} strassen={} diff={diff}",
                    naive.get(i, j), strassen.get(i, j)
                );
            }
        }
    }));

    match result {
        Ok(()) => TestOutcome::Pass,
        Err(e) => {
            let msg = crate::interactive::extract_panic_message(e);
            TestOutcome::Warn(format!("SIMD ({simd:?}): {msg}"))
        }
    }
}

// ─── Test 3: MKL Availability ───────────────────────────────────────────────

fn test_mkl() -> TestOutcome {
    if algorithms::is_mkl_available() {
        TestOutcome::Pass
    } else {
        TestOutcome::Skip("MKL runtime library not found on PATH".into())
    }
}
