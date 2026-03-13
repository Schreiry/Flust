// common.rs — The project's vocabulary and constitution.
// Everything used by two or more modules lives here.

use ratatui::style::{Color, Modifier, Style};

// ─── Theme ─────────────────────────────────────────────────────────────────
//
// Single source of truth for all colors in the application.
// Design: "Minimal Precision" — dark backgrounds, clear hierarchy, sparse accents.
// Inspired by btop++, Apple Terminal aesthetics, and Valve HUD clarity.

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
