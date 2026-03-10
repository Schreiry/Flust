// common.rs — The project's vocabulary and constitution.
// Everything used by two or more modules lives here.

// ─── Constants ──────────────────────────────────────────────────────────────

/// Float comparison threshold. Smaller → false negatives from rounding error.
/// Larger → false positives for big matrices (accumulated error).
pub const EPSILON: f64 = 1e-9;

/// Safe default tile size. Three tiles 64×64×f64 = 3 × 64 × 64 × 8 = 98304 bytes ≈ 96KB.
/// Fits in the L2 cache of most modern CPUs (typically 256KB).
pub const DEFAULT_TILE_SIZE: usize = 64;

/// Below this, Strassen switches to tiled/scalar multiplication.
/// Strassen creates ~18 temporary matrices; for small n this overhead exceeds the gain.
pub const STRASSEN_THRESHOLD: usize = 64;

/// Don't spawn rayon tasks for matrices smaller than this — thread overhead > gain.
pub const MIN_PARALLEL_SIZE: usize = 128;

// ─── SimdLevel ──────────────────────────────────────────────────────────────

/// Detected SIMD capability of the current CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    Sse42,
    Avx2,
    Avx512,
}

impl SimdLevel {
    /// Human-readable name for display/logging.
    pub fn display_name(&self) -> &'static str {
        match self {
            SimdLevel::Scalar => "Scalar",
            SimdLevel::Sse42 => "SSE4.2",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx512 => "AVX-512",
        }
    }

    /// How many f64 values fit in one SIMD register at this level.
    pub fn vector_width_f64(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse42 => 2,   // 128-bit / 64-bit = 2
            SimdLevel::Avx2 => 4,    // 256-bit / 64-bit = 4
            SimdLevel::Avx512 => 8,  // 512-bit / 64-bit = 8
        }
    }
}

// ─── MultiplicationResult ───────────────────────────────────────────────────

/// Complete record of a single matrix multiplication benchmark run.
pub struct MultiplicationResult {
    pub algorithm: String,
    pub size_m: usize,          // rows of A (original, before padding)
    pub size_n: usize,          // cols of A = rows of B
    pub size_p: usize,          // cols of B
    pub time_ms: f64,           // total time including padding/unpadding
    pub compute_time_ms: f64,   // computation only
    pub padding_time_ms: f64,
    pub unpadding_time_ms: f64,
    pub gflops: f64,            // calculated from compute_time
    pub simd_level: SimdLevel,
    pub threads_used: usize,
    pub tile_size: Option<usize>,
    pub strassen_threshold: Option<usize>,
    pub peak_ram_mb: u64,
}

impl MultiplicationResult {
    /// Calculate GFLOPS from matrix dimensions and compute time.
    ///
    /// Formula: (2 × M × N × P) / (compute_ms × 1e6)
    ///
    /// Why 2×M×N×P: each element C[i][j] requires N multiplications + N additions = 2N ops.
    /// M×P elements total, each needing 2N operations = 2×M×N×P total FLOPS.
    pub fn calculate_gflops(m: usize, n: usize, p: usize, compute_ms: f64) -> f64 {
        if compute_ms <= 0.0 {
            return 0.0;
        }
        (2.0 * m as f64 * n as f64 * p as f64) / (compute_ms * 1e6)
    }
}

// ─── ComparisonResult ───────────────────────────────────────────────────────

/// Result of comparing two matrices element-by-element.
pub struct ComparisonResult {
    pub max_abs_diff: f64,
    pub avg_abs_diff: f64,
    pub rms_diff: f64,      // Root Mean Square difference — more informative than avg
    pub match_count: usize,
    pub total_count: usize,
    pub is_equal: bool,     // all element diffs < EPSILON
    pub time_ms: f64,
}

// ─── flust_time! macro ─────────────────────────────────────────────────────

/// Times the execution of a block, returning (result, elapsed_ms).
///
/// Usage:
///   let (result, ms) = flust_time!({ expensive_computation() });
#[macro_export]
macro_rules! flust_time {
    ($block:expr) => {{
        let __start = std::time::Instant::now();
        let __result = $block;
        let __elapsed_ms = __start.elapsed().as_secs_f64() * 1000.0;
        (__result, __elapsed_ms)
    }};
}
