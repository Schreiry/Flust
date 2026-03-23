
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ratatui::style::{Color, Modifier, Style};

// ─── Theme ─────────────────────────────────────────────────────────────────
pub struct Theme;

impl Theme {
    // Backgrounds (dark → light)
    pub const BG:           Color = Color::Rgb(18, 18, 18);    // #121212 main background
    pub const SURFACE:      Color = Color::Rgb(28, 28, 28);    // #1c1c1c panels
    pub const BORDER:       Color = Color::Rgb(55, 55, 55);    // #373737 borders
    pub const BORDER_FOCUS: Color = Color::Rgb(245, 197, 24);  // #f5c518 active border

    // Text hierarchy
    pub const TEXT_DIM:     Color = Color::Rgb(90, 90, 90);    // #5a5a5a labels, captions
    pub const TEXT_MUTED:   Color = Color::Rgb(140, 140, 140); // #8c8c8c secondary text
    pub const TEXT:         Color = Color::Rgb(204, 204, 204); // #cccccc primary text
    pub const TEXT_BRIGHT:  Color = Color::Rgb(240, 240, 240); // #f0f0f0 headings

    // Accents
    pub const ACCENT:       Color = Color::Rgb(245, 197, 24);  // #f5c518 yellow (brand)
    pub const OK:           Color = Color::Rgb(74, 222, 128);  // #4ade80 green (ok)
    pub const WARN:         Color = Color::Rgb(251, 191, 36);  // #fbbf24 orange (warning)
    pub const CRIT:         Color = Color::Rgb(248, 113, 113); // #f87171 red (critical)

    /// Dynamic color by load percentage: green < 60%, orange 60-85%, red > 85%.
    pub fn load_color(pct: f32) -> Color {
        if pct >= 85.0 { Self::CRIT }
        else if pct >= 60.0 { Self::WARN }
        else { Self::OK }
    }

    pub fn block_style() -> Style {
        Style::default().fg(Self::BORDER)
    }

    pub fn block_style_focused() -> Style {
        Style::default().fg(Self::BORDER_FOCUS)
    }

    // ── Predefined composite styles ──

    pub fn style_default() -> Style {
        Style::default().fg(Self::TEXT).bg(Self::BG)
    }

    pub fn style_title() -> Style {
        Style::default().fg(Self::ACCENT).bg(Self::BG).add_modifier(Modifier::BOLD)
    }

    pub fn style_selected() -> Style {
        Style::default().fg(Self::BG).bg(Self::ACCENT).add_modifier(Modifier::BOLD)
    }

    pub fn style_muted() -> Style {
        Style::default().fg(Self::TEXT_MUTED).bg(Self::BG)
    }

    pub fn style_dim() -> Style {
        Style::default().fg(Self::TEXT_DIM).bg(Self::BG)
    }

    pub fn style_bright() -> Style {
        Style::default().fg(Self::TEXT_BRIGHT).bg(Self::BG)
    }

    pub fn style_accent() -> Style {
        Style::default().fg(Self::ACCENT).bg(Self::BG)
    }

    pub fn style_ok() -> Style {
        Style::default().fg(Self::OK).bg(Self::BG)
    }

    pub fn style_crit() -> Style {
        Style::default().fg(Self::CRIT).bg(Self::BG)
    }

    pub fn style_key_hint() -> Style {
        Style::default().fg(Self::ACCENT).bg(Self::BG).add_modifier(Modifier::BOLD)
    }

    pub fn style_surface() -> Style {
        Style::default().fg(Self::TEXT).bg(Self::SURFACE)
    }
}

// ─── ThemeKind + ThemeColors (runtime-switchable theme) ─────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThemeKind {
    Amber,  // yellow accent — brand Flust (default)
    Cyan,   // blue accent — btop/htop style
    Steel,  // monochrome gray — maximum readability
}

pub struct ThemeColors {
    pub bg:           Color,
    pub surface:      Color,
    pub border:       Color,
    pub border_focus: Color,
    pub text_dim:     Color,
    pub text_muted:   Color,
    pub text:         Color,
    pub text_bright:  Color,
    pub accent:       Color,
    pub ok:           Color,
    pub warn:         Color,
    pub crit:         Color,
}

impl ThemeColors {
    /// Dynamic color by load percentage: green < 60%, orange 60-85%, red > 85%.
    pub fn load_color(&self, pct: f32) -> Color {
        if pct >= 85.0 { self.crit }
        else if pct >= 60.0 { self.warn }
        else { self.ok }
    }
}

impl ThemeKind {
    pub fn colors(&self) -> ThemeColors {
        match self {
            ThemeKind::Amber => ThemeColors {
                bg:           Color::Rgb(18, 18, 18),
                surface:      Color::Rgb(28, 28, 28),
                border:       Color::Rgb(55, 55, 55),
                border_focus: Color::Rgb(245, 197, 24),
                text_dim:     Color::Rgb(90, 90, 90),
                text_muted:   Color::Rgb(140, 140, 140),
                text:         Color::Rgb(204, 204, 204),
                text_bright:  Color::Rgb(240, 240, 240),
                accent:       Color::Rgb(245, 197, 24),
                ok:           Color::Rgb(74, 222, 128),
                warn:         Color::Rgb(251, 191, 36),
                crit:         Color::Rgb(248, 113, 113),
            },
            ThemeKind::Cyan => ThemeColors {
                bg:           Color::Rgb(15, 20, 25),
                surface:      Color::Rgb(22, 30, 38),
                border:       Color::Rgb(45, 65, 80),
                border_focus: Color::Rgb(34, 211, 238),
                text_dim:     Color::Rgb(70, 100, 120),
                text_muted:   Color::Rgb(120, 160, 185),
                text:         Color::Rgb(190, 220, 235),
                text_bright:  Color::Rgb(225, 240, 250),
                accent:       Color::Rgb(34, 211, 238),
                ok:           Color::Rgb(74, 222, 128),
                warn:         Color::Rgb(251, 191, 36),
                crit:         Color::Rgb(248, 113, 113),
            },
            ThemeKind::Steel => ThemeColors {
                bg:           Color::Rgb(12, 12, 12),
                surface:      Color::Rgb(22, 22, 22),
                border:       Color::Rgb(50, 50, 50),
                border_focus: Color::Rgb(160, 160, 160),
                text_dim:     Color::Rgb(70, 70, 70),
                text_muted:   Color::Rgb(120, 120, 120),
                text:         Color::Rgb(185, 185, 185),
                text_bright:  Color::Rgb(230, 230, 230),
                accent:       Color::Rgb(200, 200, 200),
                ok:           Color::Rgb(150, 200, 150),
                warn:         Color::Rgb(200, 170, 100),
                crit:         Color::Rgb(200, 100, 100),
            },
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ThemeKind::Amber => "Amber",
            ThemeKind::Cyan  => "Cyan",
            ThemeKind::Steel => "Steel",
        }
    }

    pub fn next(&self) -> ThemeKind {
        match self {
            ThemeKind::Amber => ThemeKind::Cyan,
            ThemeKind::Cyan  => ThemeKind::Steel,
            ThemeKind::Steel => ThemeKind::Amber,
        }
    }
}

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

    /// Parse SIMD level from its display name. Falls back to Scalar.
    pub fn from_name(name: &str) -> Self {
        match name {
            "SSE4.2" => SimdLevel::Sse42,
            "AVX2" => SimdLevel::Avx2,
            "AVX-512" => SimdLevel::Avx512,
            _ => SimdLevel::Scalar,
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

    /// FP64 operations per cycle per core, assuming a single FMA port.
    /// FMA counts as 2 ops (multiply + add), so width × 2.
    /// Caller multiplies by fma_ports for CPUs with multiple FMA units.
    pub fn fp64_ops_per_cycle_per_fma(&self) -> f64 {
        match self {
            SimdLevel::Scalar => 2.0,   // 1 FMA = 1 mul + 1 add
            SimdLevel::Sse42 => 4.0,    // 2 f64/reg × (mul+add)
            SimdLevel::Avx2 => 8.0,     // 4 f64/reg × (mul+add)
            SimdLevel::Avx512 => 16.0,  // 8 f64/reg × (mul+add)
        }
    }
}

// ─── PeakEstimate (theoretical FP64 throughput) ─────────────────────────────
//
// Transparent breakdown of theoretical peak GFLOPS for V&V display.
// Shown on the Results screen so users can verify the formula themselves.

#[derive(Debug, Clone)]
pub struct PeakEstimate {
    pub cores: usize,
    pub freq_ghz: f64,
    pub fma_ports: u8,                    // 1 or 2 (heuristic from microarch)
    pub fp64_per_cycle_per_fma: f64,      // from SimdLevel::fp64_ops_per_cycle_per_fma()
    pub peak_gflops: f64,                 // cores × freq × fma_ports × fp64/cyc
    pub freq_source: &'static str,        // "registry", "sysinfo", "fallback"
    pub fma_source: &'static str,         // "heuristic", "default"
}

impl PeakEstimate {
    /// Human-readable formula for the Results screen.
    pub fn formula_string(&self) -> String {
        format!(
            "{} cores \u{00d7} {:.1} GHz \u{00d7} {} FMA \u{00d7} {:.0} fp64/cyc = {:.1} GFLOPS",
            self.cores, self.freq_ghz, self.fma_ports,
            self.fp64_per_cycle_per_fma, self.peak_gflops
        )
    }

    /// V&V assessment comment for a measured efficiency percentage.
    pub fn assessment(efficiency_pct: f64) -> &'static str {
        if efficiency_pct >= 50.0 {
            "Outstanding. Benchmark-class performance."
        } else if efficiency_pct >= 20.0 {
            "Excellent. Near-optimal cache utilization."
        } else if efficiency_pct >= 5.0 {
            "Good. Typical for dense matmul with SIMD."
        } else if efficiency_pct >= 1.0 {
            "Memory-bandwidth limited. Normal for matmul."
        } else {
            "Low. Likely overhead-dominated for this size."
        }
    }
}

// ─── MultiplicationResult ───────────────────────────────────────────────────

/// Complete record of a single matrix multiplication benchmark run.
#[derive(Clone)]
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
    pub theoretical_peak_gflops: f64,
    pub efficiency_pct: f64,
    pub total_flops: u64,
    pub computation_id: String,  // "FLUST-YYYYMMDD-HHMMSS-NNN"
    pub machine_name: String,    // hostname for provenance tracking
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

// ─── VvAssessment (Verification & Validation) ───────────────────────────────
//
// Scientific assessment of matrix comparison results.
// Standards reference: ASME V&V40, NASA-STD-7009.
// Thresholds based on relative Frobenius error ||A-B||_F / ||A||_F.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VvAssessment {
    Identical,          // bit-for-bit identical (max_diff < f64::EPSILON)
    NumericallyEqual,   // relative_error < 1e-12
    HighAccuracy,       // relative_error < 1e-6
    Acceptable,         // relative_error < 1e-3
    Suspicious,         // relative_error < 1e-1
    Significant,        // relative_error >= 1e-1
    Incompatible,       // dimensions do not match
}

impl VvAssessment {
    pub fn from_relative_error(rel_err: f64, max_abs_diff: f64) -> Self {
        if max_abs_diff < f64::EPSILON {
            VvAssessment::Identical
        } else if rel_err < 1e-12 {
            VvAssessment::NumericallyEqual
        } else if rel_err < 1e-6 {
            VvAssessment::HighAccuracy
        } else if rel_err < 1e-3 {
            VvAssessment::Acceptable
        } else if rel_err < 1e-1 {
            VvAssessment::Suspicious
        } else {
            VvAssessment::Significant
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Identical        => "IDENTICAL",
            Self::NumericallyEqual => "NUMERICALLY EQUAL",
            Self::HighAccuracy     => "HIGH ACCURACY",
            Self::Acceptable       => "ACCEPTABLE",
            Self::Suspicious       => "CHECK REQUIRED",
            Self::Significant      => "SIGNIFICANT DIFF",
            Self::Incompatible     => "INCOMPATIBLE",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Identical        => "Bit-for-bit identical results",
            Self::NumericallyEqual => "Numerically equivalent (rel. err < 1e-12)",
            Self::HighAccuracy     => "Relative error < 1e-6. Acceptable for FP64",
            Self::Acceptable       => "Relative error < 1e-3. Check algorithm differences",
            Self::Suspicious       => "Relative error 0.1-10%. Investigate source",
            Self::Significant      => "Relative error > 10%. Results materially different",
            Self::Incompatible     => "Matrix dimensions do not match",
        }
    }

    /// 0 = green (ok), 1 = yellow (warn), 2 = red (crit)
    pub fn severity(&self) -> u8 {
        match self {
            Self::Identical | Self::NumericallyEqual | Self::HighAccuracy => 0,
            Self::Acceptable => 1,
            Self::Suspicious | Self::Significant | Self::Incompatible => 2,
        }
    }
}

// ─── QuadrantStats ──────────────────────────────────────────────────────────

/// Per-quadrant statistics for spatial error analysis.
#[derive(Debug, Clone)]
pub struct QuadrantStats {
    pub label: &'static str,  // "Top-Left", "Top-Right", "Bot-Left", "Bot-Right"
    pub max_diff: f64,
    pub mean_diff: f64,
    pub rms_diff: f64,
    pub match_pct: f64,
}

// ─── ScientificComparisonResult ─────────────────────────────────────────────

/// Full scientific comparison of two matrices for V&V workflows.
#[derive(Debug, Clone)]
pub struct ScientificComparisonResult {
    pub rows: usize,
    pub cols: usize,

    // Norms of difference matrix ΔA = A - B
    pub frobenius_norm_diff: f64,     // ||A - B||_F
    pub max_abs_diff: f64,            // max|a_ij - b_ij|
    pub mean_abs_diff: f64,           // mean|a_ij - b_ij|
    pub rms_diff: f64,               // sqrt(mean((a_ij - b_ij)²))
    pub relative_error: f64,          // ||A-B||_F / max(||A||_F, 1e-300)

    // Norms of individual matrices
    pub frobenius_norm_a: f64,
    pub frobenius_norm_b: f64,

    // Element-wise matching
    pub exact_matches: usize,         // |diff| < f64::EPSILON
    pub epsilon_matches: usize,       // |diff| < user_epsilon
    pub total_count: usize,
    pub match_pct: f64,

    // Structural analysis
    pub sign_changes: usize,          // elements where a_ij and b_ij have different signs
    pub sparsity_a_pct: f64,
    pub sparsity_b_pct: f64,
    pub structural_zeros_match_pct: f64,

    // Quadrant analysis
    pub quadrants: [QuadrantStats; 4],

    // Assessment
    pub assessment: VvAssessment,
    pub time_ms: f64,
}

// ─── ProgressHandle (lock-free background→UI progress) ───────────────────────
//
// Packs (done, total) into a single AtomicU64 for zero-overhead cross-thread
// progress reporting. High 32 bits = total, low 32 bits = done.
// Background thread calls set(), UI thread calls fraction() every 100ms.

#[derive(Clone)]
pub struct ProgressHandle {
    inner: Arc<AtomicU64>,
}

impl ProgressHandle {
    pub fn new(total: u32) -> Self {
        let packed = (total as u64) << 32;
        Self {
            inner: Arc::new(AtomicU64::new(packed)),
        }
    }

    /// Called by background thread to report progress.
    pub fn set(&self, done: u32, total: u32) {
        let packed = ((total as u64) << 32) | (done as u64);
        self.inner.store(packed, Ordering::Relaxed);
    }

    /// Called by UI thread. Returns (done, total).
    pub fn get(&self) -> (u32, u32) {
        let packed = self.inner.load(Ordering::Relaxed);
        let done = (packed & 0xFFFF_FFFF) as u32;
        let total = (packed >> 32) as u32;
        (done, total)
    }

    /// Returns fraction 0.0..=1.0.
    pub fn fraction(&self) -> f64 {
        let (done, total) = self.get();
        if total == 0 {
            0.0
        } else {
            (done as f64 / total as f64).min(1.0)
        }
    }
}

// ─── Memory Profile ────────────────────────────────────────────────────────

/// Adaptive memory profile based on available system RAM.
/// Controls caching limits, history depth, and pre-allocation budgets
/// so Flust stays lightweight on constrained machines while leveraging
/// extra memory on beefy workstations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryProfile {
    Light,       // < 8 GB available
    Normal,      // 8–32 GB available
    Performance, // > 32 GB available
}

impl MemoryProfile {
    /// Detect profile from available RAM (in MB).
    pub fn detect(available_ram_mb: u64) -> Self {
        if available_ram_mb < 8 * 1024 {
            MemoryProfile::Light
        } else if available_ram_mb < 32 * 1024 {
            MemoryProfile::Normal
        } else {
            MemoryProfile::Performance
        }
    }

    /// Maximum session history entries kept in memory.
    pub fn max_history_entries(&self) -> usize {
        match self {
            MemoryProfile::Light => 10,
            MemoryProfile::Normal => 20,
            MemoryProfile::Performance => 50,
        }
    }

    /// Maximum matrix dimension (rows or cols) allowed in the viewer.
    pub fn max_viewer_matrix_dim(&self) -> usize {
        match self {
            MemoryProfile::Light => 512,
            MemoryProfile::Normal => 2048,
            MemoryProfile::Performance => 8192,
        }
    }

    /// Pre-allocation budget in MB for scratch buffers.
    pub fn prealloc_budget_mb(&self) -> usize {
        match self {
            MemoryProfile::Light => 64,
            MemoryProfile::Normal => 256,
            MemoryProfile::Performance => 1024,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            MemoryProfile::Light => "Light",
            MemoryProfile::Normal => "Normal",
            MemoryProfile::Performance => "Performance",
        }
    }
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
