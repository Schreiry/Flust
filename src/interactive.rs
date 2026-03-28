
use std::io;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use ratatui::Terminal;

use crate::algorithms;
use crate::common::{MultiplicationResult, ProgressHandle, ThemeColors, ThemeKind, STRASSEN_THRESHOLD};
use crate::matrix::Matrix;
use crate::system::SystemInfo;
use sysinfo::System as SysinfoSystem;

// ─── Terminal Scale ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum TerminalScale {
    Compact,
    Normal,
    Large,
}

impl TerminalScale {
    fn next(self) -> Self {
        match self {
            Self::Compact => Self::Normal,
            Self::Normal  => Self::Large,
            Self::Large   => Self::Compact,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Compact => "Compact",
            Self::Normal  => "Normal",
            Self::Large   => "Large",
        }
    }

    fn show_logo(self) -> bool {
        !matches!(self, Self::Compact)
    }

    fn stats_panel_width(self) -> u16 {
        match self {
            Self::Compact => 30,
            Self::Normal  => 36,
            Self::Large   => 42,
        }
    }

    fn from_name(s: &str) -> Option<Self> {
        match s {
            "Compact" => Some(Self::Compact),
            "Normal"  => Some(Self::Normal),
            "Large"   => Some(Self::Large),
            _ => None,
        }
    }

    fn display_name(self) -> &'static str {
        self.label()
    }
}

// ─── Constants ───────────────────────────────────────────────────────────────

const LOGO_LINES: [&str; 6] = [
    "███████╗██╗     ██╗   ██╗███████╗████████╗",
    "██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝",
    "█████╗  ██║     ██║   ██║███████╗   ██║   ",
    "██╔══╝  ██║     ██║   ██║╚════██║   ██║   ",
    "██║     ███████╗╚██████╔╝███████║   ██║   ",
    "╚═╝     ╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ",
];

// ─── Gear Animation Frames (Mechanical Style) ──────────────────────────────
//
// 4 rotation frames × 7 lines each. Teeth rotate around a central hub (●).
// Uses Unicode box-drawing for clear gear silhouettes.
// Three gears interlock via ═══ drive shaft connectors.
// Sequential activation: progress < 10% → 1 gear, 10–50% → 2, ≥ 50% → 3.

pub(crate) const GEAR_FRAMES: [[&str; 7]; 4] = [
    // Frame 0: teeth at cardinal positions (N/E/S/W)
    [
        "      \u{2502}              \u{2502}              \u{2502}       ",
        "   \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        " \u{2500}\u{2500}\u{2524} \u{25CF}\u{2699}\u{25CF} \u{251C}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2524} \u{25CF}\u{2699}\u{25CF} \u{251C}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2524} \u{25CF}\u{2699}\u{25CF} \u{251C}\u{2500}\u{2500}",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}  ",
        "      \u{2502}              \u{2502}              \u{2502}       ",
    ],
    // Frame 1: teeth rotated 45° (diagonals)
    [
        "  \u{2572}     \u{2571}       \u{2572}     \u{2571}       \u{2572}     \u{2571}  ",
        "   \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2502} \u{25CF}\u{2699}\u{25CF} \u{2502}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2502} \u{25CF}\u{2699}\u{25CF} \u{2502}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2502} \u{25CF}\u{2699}\u{25CF} \u{2502}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}  ",
        "  \u{2571}     \u{2572}       \u{2571}     \u{2572}       \u{2571}     \u{2572}  ",
    ],
    // Frame 2: teeth at cardinal positions (rotated 90° from frame 0)
    [
        "      \u{2502}              \u{2502}              \u{2502}       ",
        "   \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        " \u{2500}\u{2500}\u{2524} \u{2699}\u{25CF}\u{2699} \u{251C}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2524} \u{2699}\u{25CF}\u{2699} \u{251C}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2524} \u{2699}\u{25CF}\u{2699} \u{251C}\u{2500}\u{2500}",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}  ",
        "      \u{2502}              \u{2502}              \u{2502}       ",
    ],
    // Frame 3: teeth rotated 135° (diagonals, opposite of frame 1)
    [
        "  \u{2571}     \u{2572}       \u{2571}     \u{2572}       \u{2571}     \u{2572}  ",
        "   \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2502} \u{2699}\u{25CF}\u{2699} \u{2502}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2502} \u{2699}\u{25CF}\u{2699} \u{2502}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2502} \u{2699}\u{25CF}\u{2699} \u{2502}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}  ",
        "  \u{2572}     \u{2571}       \u{2572}     \u{2571}       \u{2572}     \u{2571}  ",
    ],
];

/// Single gear for when only 1 gear is active (progress < 10%).
pub(crate) const GEAR_SINGLE: [[&str; 7]; 4] = [
    [
        "      \u{2502}       ",
        "   \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}  \u{2502}  \u{2502}  ",
        " \u{2500}\u{2500}\u{2524} \u{25CF}\u{2699}\u{25CF} \u{251C}\u{2500}\u{2500}",
        "   \u{2502}  \u{2502}  \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}  ",
        "      \u{2502}       ",
    ],
    [
        "  \u{2572}     \u{2571}  ",
        "   \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}     \u{2502}  ",
        "   \u{2502} \u{25CF}\u{2699}\u{25CF} \u{2502}  ",
        "   \u{2502}     \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}  ",
        "  \u{2571}     \u{2572}  ",
    ],
    [
        "      \u{2502}       ",
        "   \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}  \u{2502}  \u{2502}  ",
        " \u{2500}\u{2500}\u{2524} \u{2699}\u{25CF}\u{2699} \u{251C}\u{2500}\u{2500}",
        "   \u{2502}  \u{2502}  \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}  ",
        "      \u{2502}       ",
    ],
    [
        "  \u{2571}     \u{2572}  ",
        "   \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}     \u{2502}  ",
        "   \u{2502} \u{2699}\u{25CF}\u{2699} \u{2502}  ",
        "   \u{2502}     \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}  ",
        "  \u{2572}     \u{2571}  ",
    ],
];

/// Double gear for when 2 gears are active (progress 10-50%).
pub(crate) const GEAR_DOUBLE: [[&str; 7]; 4] = [
    [
        "      \u{2502}              \u{2502}       ",
        "   \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        " \u{2500}\u{2500}\u{2524} \u{25CF}\u{2699}\u{25CF} \u{251C}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2524} \u{25CF}\u{2699}\u{25CF} \u{251C}\u{2500}\u{2500}",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}  ",
        "      \u{2502}              \u{2502}       ",
    ],
    [
        "  \u{2572}     \u{2571}       \u{2572}     \u{2571}  ",
        "   \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2502} \u{25CF}\u{2699}\u{25CF} \u{2502}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2502} \u{25CF}\u{2699}\u{25CF} \u{2502}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}  ",
        "  \u{2571}     \u{2572}       \u{2571}     \u{2572}  ",
    ],
    [
        "      \u{2502}              \u{2502}       ",
        "   \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        " \u{2500}\u{2500}\u{2524} \u{2699}\u{25CF}\u{2699} \u{251C}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2524} \u{2699}\u{25CF}\u{2699} \u{251C}\u{2500}\u{2500}",
        "   \u{2502}  \u{2502}  \u{2502}       \u{2502}  \u{2502}  \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2518}  ",
        "      \u{2502}              \u{2502}       ",
    ],
    [
        "  \u{2571}     \u{2572}       \u{2571}     \u{2572}  ",
        "   \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}       \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2502} \u{2699}\u{25CF}\u{2699} \u{2502}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2502} \u{2699}\u{25CF}\u{2699} \u{2502}  ",
        "   \u{2502}     \u{2502}       \u{2502}     \u{2502}  ",
        "   \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}       \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}  ",
        "  \u{2572}     \u{2571}       \u{2572}     \u{2571}  ",
    ],
];

// ─── Menu Items ──────────────────────────────────────────────────────────────

struct MenuItem {
    label: &'static str,
    shortcut: Option<char>,
    description: &'static str,
    category: Option<MenuCategory>,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum MenuCategory {
    Engineering,
    Economics,
}

struct CategorySubItem {
    label: &'static str,
    shortcut: Option<char>,
    description: &'static str,
}

const ENGINEERING_SUBITEMS: &[CategorySubItem] = &[
    CategorySubItem {
        label: "Thermal Simulation",
        shortcut: Some('t'),
        description: "FDM heat transfer simulation. Model fluid cooling in a reservoir.\n\
                      Computes TEG voltage, power output, and thermal field over time.",
    },
    CategorySubItem {
        label: "Kinematics & Pathfinding (MDP)",
        shortcut: Some('k'),
        description: "Markov Decision Process steady-state solver.\n\
                      Repeated matrix squaring P\u{00b2}\u{2192}P\u{2074}\u{2192}P\u{2078}\u{2026} via HPC multiply.",
    },
    CategorySubItem {
        label: "Computer Vision (IR Processing)",
        shortcut: Some('v'),
        description: "IR sensor upscale + convolution via im2col \u{2192} GEMM.\n\
                      Bicubic interpolation, denoising kernels.",
    },
    CategorySubItem {
        label: "Heat Propagation (MHPS)",
        shortcut: Some('p'),
        description: "2D Fourier heat equation FDM solver. Material library,\n\
                      variable geometry, multiple heat sources, heatmap visualization.",
    },
    CategorySubItem {
        label: "System Profile (HPC)",
        shortcut: Some('s'),
        description: "50-axis system analysis: CPU topology, cache hierarchy,\n\
                      SIMD capabilities, and optimal Flust configuration.",
    },
    CategorySubItem {
        label: "Help",
        shortcut: Some('h'),
        description: "Theory reference: heat equation, FDM, MDP, im2col,\n\
                      boundary conditions, and convergence criteria.",
    },
    CategorySubItem {
        label: "Info",
        shortcut: Some('i'),
        description: "Overview of available engineering simulations,\n\
                      input parameters, and solver options.",
    },
];

const ECONOMICS_SUBITEMS: &[CategorySubItem] = &[
    CategorySubItem {
        label: "Leontief Shock Simulator",
        shortcut: Some('l'),
        description: "Input\u{2013}Output model: cascade economic shocks through sectors.\n\
                      Neumann series x\u{2096} = A\u{00b7}x\u{2096}\u{208b}\u{2081} + d with HPC matrix\u{2013}vector multiply.",
    },
    CategorySubItem {
        label: "Help",
        shortcut: Some('h'),
        description: "Theory reference: Leontief input\u{2013}output model, technology matrix,\n\
                      Neumann series convergence, spectral radius conditions.",
    },
    CategorySubItem {
        label: "Info",
        shortcut: Some('i'),
        description: "Overview of macroeconomic simulation parameters,\n\
                      sector configuration, and shock propagation mechanics.",
    },
];

impl MenuCategory {
    fn label(self) -> &'static str {
        match self {
            MenuCategory::Engineering => "Engineering",
            MenuCategory::Economics => "Economics",
        }
    }

    fn subitems(self) -> &'static [CategorySubItem] {
        match self {
            MenuCategory::Engineering => ENGINEERING_SUBITEMS,
            MenuCategory::Economics => ECONOMICS_SUBITEMS,
        }
    }

    fn description(self) -> &'static str {
        match self {
            MenuCategory::Engineering => "Physical simulations: thermal FDM, MDP pathfinding, IR vision.\n\
                                          HPC matrix solvers with full visualization.",
            MenuCategory::Economics => "Macroeconomic modeling: Leontief input\u{2013}output analysis.\n\
                                        Cascade shock propagation via HPC matrix iteration.",
        }
    }
}

const MENU_ITEMS: &[MenuItem] = &[
    MenuItem {
        label: "Matrix Multiplication",
        shortcut: Some('m'),
        description: "Multiply two matrices using Strassen, Tiled, or scalar algorithms.\n\
                      Supports random generation, file input, or manual entry.",
        category: None,
    },
    MenuItem {
        label: "Algorithm Comparison",
        shortcut: Some('c'),
        description: "Compare two algorithms on identical random matrices.\n\
                      Reports speedup, GFLOPS, and numerical agreement.",
        category: None,
    },
    MenuItem {
        label: "Matrix File Comparison",
        shortcut: Some('f'),
        description: "Load two matrices from CSV files and compare scientifically.\n\
                      Frobenius norm, RMSE, quadrant analysis, V&V assessment.",
        category: None,
    },
    MenuItem {
        label: "Performance Monitor",
        shortcut: Some('p'),
        description: "Open a real-time CPU/RAM monitor in a separate console window.\n\
                      Per-core bars, sparkline history, memory gauge.",
        category: None,
    },
    MenuItem {
        label: "Benchmark Suite",
        shortcut: Some('b'),
        description: "Run all algorithms across multiple matrix sizes.\n\
                      Generates a performance comparison table and CSV report.",
        category: None,
    },
    MenuItem {
        label: "Matrix Viewer",
        shortcut: Some('v'),
        description: "Load and browse a matrix from a CSV file.\n\
                      Navigate rows/cols, highlight min/max, view statistics.",
        category: None,
    },
    MenuItem {
        label: "Engineering  \u{25b8}",
        shortcut: Some('e'),
        description: "Physical simulations: thermal FDM, TEG power generation.\n\
                      GPU-ready matrix solvers with full 3D visualization.",
        category: Some(MenuCategory::Engineering),
    },
    MenuItem {
        label: "Economics  \u{25b8}",
        shortcut: Some('o'),
        description: "Macroeconomic modeling: Leontief input\u{2013}output analysis.\n\
                      Cascade shock propagation via HPC matrix iteration.",
        category: Some(MenuCategory::Economics),
    },
    MenuItem {
        label: "Computation History",
        shortcut: Some('y'),
        description: "View results from this session.\n\
                      Compare algorithms, re-run any previous configuration.",
        category: None,
    },
];

// ─── Benchmark Data ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct BenchmarkData {
    algorithm: String,
    size: usize,
    gen_time_ms: Option<f64>,
    padding_time_ms: f64,
    compute_time_ms: f64,
    unpadding_time_ms: f64,
    gflops: f64,
    simd_level: String,
    threads: usize,
    peak_ram_mb: u64,
    result_matrix: Option<Matrix>,
    theoretical_peak_gflops: f64,
    efficiency_pct: f64,
    total_flops: u64,
    computation_id: String,  // "FLUST-YYYYMMDD-HHMMSS-NNN"
    machine_name: String,    // hostname for provenance
    // Extended profiling metrics
    allocation_ms: f64,
    rows_a: usize,
    cols_a: usize,
    cols_b: usize,
    matrix_memory_mb: f64,
}

// ─── Matrix Stats ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct MatrixStats {
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64,
    nonzero: usize,
    total: usize,
    sparsity_pct: f64,
    // Norms (Chapter 15)
    frobenius_norm: f64,
    norm_1: f64,
    norm_infinity: f64,
    // Square-matrix properties
    trace: Option<f64>,
    is_symmetric: Option<bool>,
    condition_estimate: Option<(f64, bool)>, // (kappa, is_approximate)
}

impl MatrixStats {
    fn compute(matrix: &Matrix) -> Self {
        let data = matrix.data();
        let n = data.len();
        if n == 0 {
            return Self {
                min: 0.0, max: 0.0, mean: 0.0, std_dev: 0.0,
                nonzero: 0, total: 0, sparsity_pct: 100.0,
                frobenius_norm: 0.0, norm_1: 0.0, norm_infinity: 0.0,
                trace: None, is_symmetric: None, condition_estimate: None,
            };
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut nonzero = 0usize;
        for &x in data {
            if x < min { min = x; }
            if x > max { max = x; }
            sum += x;
            if x.abs() > 1e-10 { nonzero += 1; }
        }
        let mean = sum / n as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let sparsity_pct = (1.0 - nonzero as f64 / n as f64) * 100.0;

        // Norms (always computed — O(n) or O(n^2))
        let frobenius_norm = crate::numerics::frobenius_norm(matrix);
        let norm_1 = crate::numerics::norm_1(matrix);
        let norm_infinity = crate::numerics::norm_infinity(matrix);

        // Square-matrix properties
        let rows = matrix.rows();
        let cols = matrix.cols();
        let is_square = rows == cols;

        let trace = if is_square {
            Some(matrix.trace())
        } else {
            None
        };

        let is_symmetric = if is_square && rows <= 512 {
            Some(crate::numerics::is_symmetric(matrix, 1e-10))
        } else {
            None
        };

        let condition_estimate = if is_square && rows <= 4096 && rows >= 2 {
            let (kappa, approx, _) = crate::numerics::condition_number_estimate(matrix);
            Some((kappa, approx))
        } else {
            None
        };

        Self {
            min, max, mean, std_dev, nonzero, total: n, sparsity_pct,
            frobenius_norm, norm_1, norm_infinity,
            trace, is_symmetric, condition_estimate,
        }
    }
}

// ─── Diff Result ─────────────────────────────────────────────────────────────

#[derive(Clone)]
struct DiffResultData {
    alg1_name: String,
    alg2_name: String,
    size: usize,
    time1_ms: f64,
    time2_ms: f64,
    gflops1: f64,
    gflops2: f64,
    speedup: f64,
    winner: String,
    max_diff: f64,
    is_match: bool,
}

// ─── Algorithm Choice ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum AlgorithmChoice {
    Naive,
    Strassen,
    Winograd,
    StrassenHybrid,
    WinogradHybrid,
    IntelMKL,
}

impl AlgorithmChoice {
    fn display_name(&self) -> &'static str {
        match self {
            AlgorithmChoice::Naive => "Naive (i-k-j)",
            AlgorithmChoice::Strassen => "Parallel Strassen + Tiled",
            AlgorithmChoice::Winograd => "Parallel Winograd + Tiled",
            AlgorithmChoice::StrassenHybrid => "Strassen Hybrid (Full-Core)",
            AlgorithmChoice::WinogradHybrid => "Winograd Hybrid (Full-Core)",
            AlgorithmChoice::IntelMKL => "Intel MKL (DGEMM)",
        }
    }
}

// ─── ETA Tracker (Exponential Moving Average) ───────────────────────────────

pub(crate) struct EtaTracker {
    start: Instant,
    last_fraction: f64,
    last_time: Instant,
    ema_rate: f64,
    alpha: f64,
    initialized: bool,
}

impl EtaTracker {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_fraction: 0.0,
            last_time: now,
            ema_rate: 0.0,
            alpha: 0.3,
            initialized: false,
        }
    }

    fn update(&mut self, fraction: f64) -> Option<f64> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_time).as_secs_f64();

        if dt < 0.05 || fraction <= self.last_fraction {
            return self.estimate_remaining(fraction);
        }

        let df = fraction - self.last_fraction;
        let current_rate = df / dt;

        if !self.initialized {
            self.ema_rate = current_rate;
            self.initialized = true;
        } else {
            self.ema_rate = self.alpha * current_rate + (1.0 - self.alpha) * self.ema_rate;
        }

        self.last_fraction = fraction;
        self.last_time = now;

        self.estimate_remaining(fraction)
    }

    pub(crate) fn estimate_remaining(&self, fraction: f64) -> Option<f64> {
        if !self.initialized || self.ema_rate <= 1e-12 || fraction >= 1.0 {
            return None;
        }
        Some((1.0 - fraction) / self.ema_rate)
    }

    fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub(crate) fn format_eta(&self, fraction: f64) -> String {
        match self.estimate_remaining(fraction) {
            None if fraction >= 1.0 => "Done".into(),
            None => "ETA: calculating...".into(),
            Some(secs) if secs < 1.0 => format!("ETA: <1s"),
            Some(secs) if secs < 60.0 => format!("ETA: {:.0}s", secs),
            Some(secs) => format!("ETA: {:.0}m {:.0}s", secs / 60.0, secs % 60.0),
        }
    }
}

// ─── Background Computation Types ───────────────────────────────────────────

pub(crate) enum ComputeResult {
    Multiply {
        result: Matrix,
        padding_ms: f64,
        unpadding_ms: f64,
        compute_ms: f64,
        peak_ram_bytes: u64,
        avg_freq_ghz: f64,
    },
    Diff {
        result1: Matrix,
        result2: Matrix,
        time1_ms: f64,
        time2_ms: f64,
    },
    Thermal {
        result: crate::thermal::ThermalSimResult,
    },
    Economics {
        result: crate::economics::LeontiefResult,
    },
    Mdp {
        result: crate::kinematics::MdpResult,
    },
    Vision {
        result: crate::vision::VisionResult,
    },
    Mhps {
        result: crate::mhps::MhpsResult,
    },
    /// Computation thread panicked or returned an error.
    Error {
        message: String,
    },
}

/// Extract a human-readable message from a `catch_unwind` panic payload.
pub(crate) fn extract_panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "Unknown panic in compute thread".to_string()
    }
}

#[derive(Clone)]
pub(crate) struct ComputeContext {
    pub(crate) algorithm_choice: AlgorithmChoice,
    pub(crate) algorithm_name: String,
    pub(crate) size: usize,
    pub(crate) gen_time_ms: Option<f64>,
    pub(crate) simd_level: crate::common::SimdLevel,
    // Diff-specific
    pub(crate) is_diff: bool,
    pub(crate) diff_alg1: Option<AlgorithmChoice>,
    pub(crate) diff_alg2: Option<AlgorithmChoice>,
}

pub(crate) struct ComputeTask {
    pub(crate) progress: ProgressHandle,
    pub(crate) eta: EtaTracker,
    pub(crate) receiver: mpsc::Receiver<ComputeResult>,
    pub(crate) context: ComputeContext,
    pub(crate) _join_handle: std::thread::JoinHandle<()>,
    /// Process-isolated computation (MKL). Parent polls child process.
    pub(crate) child_process: Option<std::process::Child>,
    pub(crate) temp_dir: Option<std::path::PathBuf>,
    pub(crate) compute_request: Option<crate::compute_worker::ComputeRequest>,
}

// ─── Session History ────────────────────────────────────────────────────────

/// Configuration needed to re-run a computation with the same parameters.
#[derive(Clone)]
struct RunConfig {
    algorithm: AlgorithmChoice,
    size: usize,
}

/// A single entry in the session history log.
#[derive(Clone)]
struct HistoryEntry {
    data: BenchmarkData,
    config: RunConfig,
    timestamp: std::time::SystemTime,
    label: String,
}

/// In-memory computation history — lives only for the current session.
struct SessionHistory {
    entries: Vec<HistoryEntry>,
    max_entries: usize,
}

impl SessionHistory {
    fn new(profile: crate::common::MemoryProfile) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: profile.max_history_entries(),
        }
    }

    fn push(&mut self, data: BenchmarkData, config: RunConfig, io_tx: &std::sync::mpsc::Sender<crate::io::IoTask>) {
        let label = format!("{} {}×{}", data.algorithm, data.size, data.size);

        // Persist to CSV via background I/O thread (non-blocking)
        let record = crate::io::HistoryRecord {
            unique_id:  crate::io::make_history_id(&data.algorithm, data.size),
            timestamp:  crate::io::timestamp_now(),
            algorithm:  data.algorithm.clone(),
            size:       data.size,
            compute_ms: data.compute_time_ms,
            gflops:     data.gflops,
            simd:       data.simd_level.clone(),
            threads:    data.threads,
            peak_ram_mb: data.peak_ram_mb,
        };
        let _ = io_tx.send(crate::io::IoTask::AppendHistory { record });

        let entry = HistoryEntry {
            timestamp: std::time::SystemTime::now(),
            label,
            data,
            config,
        };
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ─── App State Machine ──────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) enum Screen {
    MainMenu,
    MultiplyMenu,
    SizeInput,
    InputMethodMenu,
    NameInput,
    ManualInputA { name: String },
    ManualInputB { name_a: String, matrix_a: Matrix, name: String },
    Computing { algorithm: String },
    Results { data: BenchmarkData },
    History,
    ViewerFileInput,
    MatrixViewer,
    DiffSizeInput,
    DiffAlgSelect,
    DiffResults { data: DiffResultData },
    FileCompareInputA,
    FileCompareInputB,
    FileCompareResults { data: FileCompareData },
    ComingSoon(String),
    FileBrowser { return_to: FileBrowserReturn },
    // Category submenu
    CategoryMenu { category: MenuCategory },
    // Thermal simulation wizard
    ThermalFluidSelect,
    ThermalGeometry,
    ThermalTeg,
    ThermalHeatSources,
    ThermalSolverSelect,
    ThermalConfirm,
    ThermalComputing,
    ThermalResults { result: Box<crate::thermal::ThermalSimResult> },
    ThermalViewer,
    // Economics simulation wizard
    EconConfig,
    EconShockSelect,
    EconConfirm,
    EconComputing,
    EconResults { result: Box<crate::economics::LeontiefResult> },
    // MDP Kinematics wizard
    MdpConfig,
    MdpConfirm,
    MdpComputing,
    MdpResults { result: Box<crate::kinematics::MdpResult> },
    // Computer Vision (IR Processing) wizard
    VisionConfig,
    VisionKernelSelect,
    VisionConfirm,
    VisionComputing,
    VisionResults { result: Box<crate::vision::VisionResult> },
    // Material Heat Propagation Simulator (MHPS)
    MhpsMaterial,
    MhpsGeometry,
    MhpsHeatSources,
    MhpsSimParams,
    MhpsConfirm,
    MhpsComputing,
    MhpsResults { result: Box<crate::mhps::MhpsResult> },
    // System Profiler (HPC)
    SystemProfile { profile: Box<crate::system_profiler::SystemProfile> },
    /// Computation failed — shows error message with [Esc] to return.
    ComputeError(String),
}

#[derive(Clone)]
pub(crate) enum FileBrowserReturn {
    MatrixViewer,
    FileCompareA,
    FileCompareB,
}

// ─── File Browser Entry ────────────────────────────────────────────────────

#[derive(Clone)]
struct FbEntry {
    name: String,
    is_dir: bool,
    size_bytes: u64,
}

// ─── File Compare Data ─────────────────────────────────────────────────────

#[derive(Clone)]
struct FileCompareData {
    path_a: String,
    path_b: String,
    dims_a: (usize, usize),
    dims_b: (usize, usize),
    result: crate::common::ScientificComparisonResult,
}

#[derive(PartialEq)]
pub(crate) enum Overlay {
    None,
    Help,
    About,
    ThermalHelp,
    ThermalGraph,
    ThermalCrossSection2D,
    ThermalIsometric,
    EngineeringInfo,
    EconHelp,
    EconInfo,
    EconConvergenceGraph,
    EconSectorBars,
    EconDashboard,
    // MDP overlays
    MdpConvergenceGraph,
    MdpStateBars,
    MdpDashboard,
    // Vision overlays
    VisionHeatmap,
    VisionPipeline,
}

pub(crate) struct App {
    pub(crate) running: bool,
    pub(crate) screen: Screen,
    pub(crate) overlay: Overlay,
    pub(crate) sys_info: SystemInfo,

    // Menu state
    pub(crate) main_menu_idx: usize,
    pub(crate) category_menu_idx: usize,
    mult_menu_idx: usize,
    input_method_idx: usize,

    // Session naming (set before run_generation / run_multiplication)
    pending_session_name: String,
    pending_matrix_a: Option<Matrix>,
    pending_matrix_b: Option<Matrix>,
    pending_is_random: bool,

    // User input buffers
    size_input: String,
    chosen_size: usize,
    algorithm_choice: AlgorithmChoice,

    // Manual matrix input
    manual_buffer: String,
    manual_row: usize,
    manual_col: usize,
    manual_data: Vec<f64>,
    manual_name: String,

    // Results save state
    csv_saved: bool,
    matrix_saved: bool,

    // Session history
    session_history: SessionHistory,
    history_selected: usize,

    // Diff Mode
    diff_size_input: String,
    diff_alg1: AlgorithmChoice,
    diff_alg2: AlgorithmChoice,
    diff_select_idx: usize,
    diff_selecting_which: u8, // 1 or 2

    // Theme & scale
    current_theme: ThemeKind,
    terminal_scale: TerminalScale,

    // Matrix Viewer
    viewer_matrix: Option<Matrix>,
    pub(crate) viewer_filename: String,
    viewer_scroll_row: usize,
    viewer_scroll_col: usize,
    viewer_cursor_row: usize,
    viewer_cursor_col: usize,
    // Cached visible size from last render frame (used by input handler for scroll-follow logic)
    viewer_visible_rows: std::cell::Cell<usize>,
    viewer_visible_cols: std::cell::Cell<usize>,
    viewer_precision: usize,
    viewer_highlight: bool,
    viewer_stats: Option<MatrixStats>,
    viewer_path_input: String,
    // Edit mode
    viewer_edit_mode: bool,
    viewer_edit_buffer: String,
    viewer_unsaved_changes: bool,
    viewer_exit_warning: bool,
    viewer_loaded_metadata: Option<crate::io::MatrixMetadata>,

    // File comparison
    file_compare_path_a: String,
    file_compare_path_b: String,
    file_compare_matrix_a: Option<Matrix>,
    file_compare_error: Option<String>,

    // Background computation
    pub(crate) compute_task: Option<ComputeTask>,

    // Thermal simulation wizard state
    pub(crate) thermal_fluid_idx: usize,
    pub(crate) thermal_geometry_fields: [String; 4], // lx, ly, lz, n
    pub(crate) thermal_geometry_active: usize,
    pub(crate) thermal_teg_fields: [String; 6], // seebeck, r_int, r_load, area, thickness, k_teg
    pub(crate) thermal_teg_active: usize,
    pub(crate) thermal_solver: crate::thermal::ThermalSolver,
    pub(crate) thermal_heat_sources: Vec<crate::thermal::HeatSource>,
    pub(crate) thermal_hs_cursor: usize,
    pub(crate) thermal_hs_editing: Option<usize>,
    pub(crate) thermal_hs_fields: [String; 6], // x%, y%, z%, temp, radius, omega
    pub(crate) thermal_hs_active_field: usize,
    pub(crate) thermal_use_defaults: bool,
    pub(crate) thermal_csv_saved: bool,
    pub(crate) thermal_field_saved: bool,
    pub(crate) thermal_phase: Option<Arc<Mutex<String>>>,

    // Thermal overlay state
    pub(crate) thermal_overlay_scroll: usize,
    pub(crate) thermal_cross_slice_y: usize,
    // Viewport2D state for cross-section free-camera
    pub(crate) thermal_vp_cam_x: f64,   // pan offset X (0.0 = centered)
    pub(crate) thermal_vp_cam_y: f64,   // pan offset Y (0.0 = centered)
    pub(crate) thermal_vp_zoom: f64,    // zoom level (1.0 = fit-to-screen)
    pub(crate) thermal_vp_cursor_x: usize, // cursor grid X
    pub(crate) thermal_vp_cursor_y: usize, // cursor grid Z (vertical)

    // Thermal CSV viewer
    pub(crate) thermal_view_data: Option<crate::thermal_export::ThermalViewData>,
    pub(crate) thermal_viewer_scroll: usize,

    // Economics simulation wizard state
    pub(crate) econ_sectors_input: String,
    pub(crate) econ_sparsity_input: String,
    pub(crate) econ_tol_input: String,
    pub(crate) econ_max_iter_input: String,
    pub(crate) econ_shock_sector_input: String,
    pub(crate) econ_shock_mag_input: String,
    pub(crate) econ_active_field: usize,
    pub(crate) econ_phase: Option<Arc<Mutex<String>>>,
    pub(crate) econ_csv_saved: bool,
    pub(crate) econ_overlay_scroll: usize,
    pub(crate) econ_dashboard_tab: usize,

    // MDP Kinematics wizard state
    pub(crate) mdp_states_input: String,
    pub(crate) mdp_sparsity_input: String,
    pub(crate) mdp_eps_input: String,
    pub(crate) mdp_max_iter_input: String,
    pub(crate) mdp_seed_input: String,
    pub(crate) mdp_active_field: usize,
    pub(crate) mdp_phase: Option<Arc<Mutex<String>>>,
    pub(crate) mdp_csv_saved: bool,
    pub(crate) mdp_overlay_scroll: usize,
    pub(crate) mdp_dashboard_tab: usize,

    // Computer Vision (IR Processing) wizard state
    pub(crate) vision_rows_input: String,
    pub(crate) vision_cols_input: String,
    pub(crate) vision_upscale_input: String,
    pub(crate) vision_ksize_input: String,
    pub(crate) vision_noise_input: String,
    pub(crate) vision_seed_input: String,
    pub(crate) vision_active_field: usize,
    pub(crate) vision_kernel_idx: usize,
    pub(crate) vision_phase: Option<Arc<Mutex<String>>>,
    pub(crate) vision_csv_saved: bool,

    // MHPS (Material Heat Propagation) wizard state
    pub(crate) mhps_material_idx: usize,
    pub(crate) mhps_geo_fields: [String; 5], // lx_mm, ly_mm, thick_mm, grid_n, nz
    pub(crate) mhps_active_field: usize,
    pub(crate) mhps_shape_idx: usize, // 0=Rect, 1=Polygon, 2=RoundedRect, 3=LShape
    pub(crate) mhps_shape_params: [String; 2], // shape-specific params (e.g. radii, cutout dims)
    pub(crate) mhps_polygon_verts: Vec<(String, String)>, // polygon vertex inputs (x_m, y_m)
    pub(crate) mhps_heat_sources: Vec<crate::mhps::HeatSource>,
    pub(crate) mhps_hs_name: String,
    pub(crate) mhps_hs_x: String,
    pub(crate) mhps_hs_y: String,
    pub(crate) mhps_hs_temp: String,
    pub(crate) mhps_sim_fields: [String; 5], // t_init, t_amb, conv_h, total_time, epsilon
    pub(crate) mhps_phase: Option<Arc<Mutex<String>>>,
    pub(crate) mhps_csv_saved: bool,
    pub(crate) mhps_field_saved: bool,
    pub(crate) mhps_show_gradient: bool,
    pub(crate) mhps_view_mode: usize,    // 0..5 for result view tabs
    pub(crate) mhps_snapshot_idx: usize,  // current snapshot for time scrubbing
    pub(crate) mhps_cross_axis: u8,       // 0=XY, 1=XZ, 2=YZ slice
    pub(crate) mhps_cross_pos: usize,     // slice position along perpendicular axis

    // Gear animation state (for Computing/ThermalComputing/EconComputing/MdpComputing/VisionComputing screens)
    pub(crate) gear_frame: usize,

    // File browser state
    fb_entries: Vec<FbEntry>,
    fb_selected: usize,
    fb_scroll: usize,
    fb_current_dir: std::path::PathBuf,
    fb_error: Option<String>,

    // Background I/O worker
    io_tx: std::sync::mpsc::Sender<crate::io::IoTask>,
    io_rx: std::sync::mpsc::Receiver<crate::io::IoResult>,
    io_status: Option<String>, // "Saving..." / "Loading..." shown as overlay
}

impl App {
    fn new(sys_info: SystemInfo) -> Self {
        let mem_profile = sys_info.memory_profile;
        let (io_tx_init, io_rx_init) = crate::io::spawn_io_worker();

        // Load persisted config (theme, scale)
        let (saved_theme, saved_scale) = crate::io::load_config()
            .unwrap_or_else(|| ("Amber".into(), "Normal".into()));
        let loaded_theme = ThemeKind::from_name(&saved_theme).unwrap_or(ThemeKind::Amber);
        let loaded_scale = TerminalScale::from_name(&saved_scale).unwrap_or(TerminalScale::Normal);

        App {
            running: true,
            screen: Screen::MainMenu,
            overlay: Overlay::None,
            sys_info,
            main_menu_idx: 0,
            category_menu_idx: 0,
            mult_menu_idx: 0,
            input_method_idx: 0,
            pending_session_name: String::new(),
            pending_matrix_a: None,
            pending_matrix_b: None,
            pending_is_random: true,
            size_input: String::new(),
            chosen_size: 0,
            algorithm_choice: AlgorithmChoice::StrassenHybrid,
            manual_buffer: String::new(),
            manual_row: 0,
            manual_col: 0,
            manual_data: Vec::new(),
            manual_name: String::new(),
            csv_saved: false,
            matrix_saved: false,
            session_history: SessionHistory::new(mem_profile),
            history_selected: 0,
            diff_size_input: String::new(),
            diff_alg1: AlgorithmChoice::StrassenHybrid,
            diff_alg2: AlgorithmChoice::WinogradHybrid,
            diff_select_idx: 0,
            diff_selecting_which: 1,
            current_theme: loaded_theme,
            terminal_scale: loaded_scale,
            viewer_matrix: None,
            viewer_filename: String::new(),
            viewer_scroll_row: 0,
            viewer_scroll_col: 0,
            viewer_cursor_row: 0,
            viewer_cursor_col: 0,
            viewer_visible_rows: std::cell::Cell::new(20),
            viewer_visible_cols: std::cell::Cell::new(5),
            viewer_precision: 4,
            viewer_highlight: false,
            viewer_stats: None,
            viewer_path_input: String::new(),
            viewer_edit_mode: false,
            viewer_edit_buffer: String::new(),
            viewer_unsaved_changes: false,
            viewer_exit_warning: false,
            viewer_loaded_metadata: None,
            file_compare_path_a: String::new(),
            file_compare_path_b: String::new(),
            file_compare_matrix_a: None,
            file_compare_error: None,
            compute_task: None,
            thermal_fluid_idx: 0,
            thermal_geometry_fields: [
                "0.15".into(), "0.08".into(), "0.06".into(), "16".into(),
            ],
            thermal_geometry_active: 0,
            thermal_teg_fields: [
                "0.05".into(), "2.0".into(), "2.0".into(),
                "0.004".into(), "0.004".into(), "1.5".into(),
            ],
            thermal_teg_active: 0,
            thermal_solver: crate::thermal::ThermalSolver::NativeSparse,
            thermal_heat_sources: Vec::new(),
            thermal_hs_cursor: 0,
            thermal_hs_editing: None,
            thermal_hs_fields: [
                "50".into(), "50".into(), "50".into(),
                "200.0".into(), "0.02".into(), "0.0".into(),
            ],
            thermal_hs_active_field: 0,
            thermal_use_defaults: true,
            thermal_csv_saved: false,
            thermal_field_saved: false,
            thermal_phase: None,
            thermal_overlay_scroll: 0,
            thermal_cross_slice_y: 0,
            thermal_vp_cam_x: 0.0,
            thermal_vp_cam_y: 0.0,
            thermal_vp_zoom: 1.0,
            thermal_vp_cursor_x: 0,
            thermal_vp_cursor_y: 0,
            thermal_view_data: None,
            thermal_viewer_scroll: 0,
            econ_sectors_input: "500".into(),
            econ_sparsity_input: "0.7".into(),
            econ_tol_input: "1e-10".into(),
            econ_max_iter_input: "5000".into(),
            econ_shock_sector_input: "0".into(),
            econ_shock_mag_input: "100.0".into(),
            econ_active_field: 0,
            econ_phase: None,
            econ_csv_saved: false,
            econ_overlay_scroll: 0,
            econ_dashboard_tab: 0,
            // MDP
            mdp_states_input: "256".into(),
            mdp_sparsity_input: "0.3".into(),
            mdp_eps_input: "1e-10".into(),
            mdp_max_iter_input: "200".into(),
            mdp_seed_input: String::new(),
            mdp_active_field: 0,
            mdp_phase: None,
            mdp_csv_saved: false,
            mdp_overlay_scroll: 0,
            mdp_dashboard_tab: 0,
            // Vision
            vision_rows_input: "8".into(),
            vision_cols_input: "8".into(),
            vision_upscale_input: "4".into(),
            vision_ksize_input: "3".into(),
            vision_noise_input: "0.1".into(),
            vision_seed_input: String::new(),
            vision_active_field: 0,
            vision_kernel_idx: 0,
            vision_phase: None,
            vision_csv_saved: false,
            // MHPS
            mhps_material_idx: 1, // Aluminum
            mhps_geo_fields: [
                "200".into(), "150".into(), "5".into(), "40".into(), "1".into(),
            ],
            mhps_active_field: 0,
            mhps_shape_idx: 0,
            mhps_shape_params: ["0.02".into(), "0.02".into()],
            mhps_polygon_verts: Vec::new(),
            mhps_heat_sources: Vec::new(),
            mhps_hs_name: "H1".into(),
            mhps_hs_x: "50".into(),
            mhps_hs_y: "50".into(),
            mhps_hs_temp: "120".into(),
            mhps_sim_fields: [
                "20".into(), "20".into(), "0".into(), "60".into(), "0.01".into(),
            ],
            mhps_phase: None,
            mhps_csv_saved: false,
            mhps_field_saved: false,
            mhps_show_gradient: false,
            mhps_view_mode: 0,
            mhps_snapshot_idx: 0,
            mhps_cross_axis: 0,
            mhps_cross_pos: 0,
            gear_frame: 0,
            fb_entries: Vec::new(),
            fb_selected: 0,
            fb_scroll: 0,
            fb_current_dir: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            fb_error: None,
            io_tx: io_tx_init,
            io_rx: io_rx_init,
            io_status: None,
        }
    }

    pub(crate) fn theme(&self) -> ThemeColors {
        self.current_theme.colors()
    }
}

// ─── Main entry point ───────────────────────────────────────────────────────

pub fn run_interactive_mode() {
    let sys_info = SystemInfo::detect();
    if let Err(e) = run_tui(sys_info) {
        eprintln!("TUI error: {e}");
    }
}

/// Launch TUI with a pre-detected SystemInfo (skips re-detection).
pub fn run_interactive_mode_with_sysinfo(sys_info: SystemInfo) {
    if let Err(e) = run_tui(sys_info) {
        eprintln!("TUI error: {e}");
    }
}

fn run_tui(sys_info: SystemInfo) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(sys_info);

    // Pre-populate session history from persistent CSV (last 20 entries, newest last)
    if let Ok(records) = crate::io::load_history() {
        let start = records.len().saturating_sub(app.session_history.max_entries);
        for rec in &records[start..] {
            let bench = BenchmarkData {
                algorithm:              rec.algorithm.clone(),
                size:                   rec.size,
                gen_time_ms:            None,
                padding_time_ms:        0.0,
                compute_time_ms:        rec.compute_ms,
                unpadding_time_ms:      0.0,
                gflops:                 rec.gflops,
                simd_level:             rec.simd.clone(),
                threads:                rec.threads,
                peak_ram_mb:            rec.peak_ram_mb,
                result_matrix:          None,
                theoretical_peak_gflops: 0.0,
                efficiency_pct:         0.0,
                total_flops:            0,
                computation_id:         rec.unique_id.clone(),
                machine_name:           String::new(),
                allocation_ms:          0.0,
                rows_a:                 rec.size,
                cols_a:                 rec.size,
                cols_b:                 rec.size,
                matrix_memory_mb:       0.0,
            };
            let alg_choice = AlgorithmChoice::StrassenHybrid; // best-effort default for re-run
            // Push directly (bypass persistent write since already on disk)
            let label = format!("{} {}×{}", bench.algorithm, bench.size, bench.size);
            app.session_history.entries.push(HistoryEntry {
                timestamp: std::time::SystemTime::UNIX_EPOCH,
                label,
                data: bench,
                config: RunConfig { algorithm: alg_choice, size: rec.size },
            });
        }
    }

    while app.running {
        // Check if a background computation has finished
        check_compute_completion(&mut app);

        // Poll background I/O results
        check_io_completion(&mut app);

        // Advance gear animation on computing screens (every 50ms poll tick)
        if matches!(app.screen, Screen::Computing { .. } | Screen::ThermalComputing | Screen::EconComputing | Screen::MdpComputing | Screen::VisionComputing | Screen::MhpsComputing) {
            app.gear_frame = (app.gear_frame + 1) % 4;
        }

        terminal.draw(|f| render(&app, f))?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    handle_input(&mut app, key.code, &mut terminal);
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

// ─── Input Handling ─────────────────────────────────────────────────────────

fn handle_input(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    // Overlays intercept all input
    if app.overlay != Overlay::None {
        let is_cross = matches!(app.overlay, Overlay::ThermalCrossSection2D);
        let is_cross_or_iso = matches!(app.overlay, Overlay::ThermalCrossSection2D | Overlay::ThermalIsometric);
        let is_econ_dashboard = matches!(app.overlay, Overlay::EconDashboard);
        let is_mdp_dashboard = matches!(app.overlay, Overlay::MdpDashboard);
        match key {
            KeyCode::Esc | KeyCode::Char('q') => {
                // Reset viewport on close
                if is_cross {
                    app.thermal_vp_cam_x = 0.0;
                    app.thermal_vp_cam_y = 0.0;
                    app.thermal_vp_zoom = 1.0;
                }
                app.overlay = Overlay::None;
            }
            KeyCode::Enter => {
                if matches!(app.overlay,
                    Overlay::Help | Overlay::About |
                    Overlay::EngineeringInfo | Overlay::EconHelp | Overlay::EconInfo
                ) {
                    app.overlay = Overlay::None;
                }
            }
            // ─── Cross-section viewport: WASD pan, +/- zoom, arrows cursor ──
            KeyCode::Char('w') | KeyCode::Char('W') if is_cross => {
                app.thermal_vp_cam_y += 2.0 / app.thermal_vp_zoom;
            }
            KeyCode::Char('s') | KeyCode::Char('S') if is_cross => {
                app.thermal_vp_cam_y -= 2.0 / app.thermal_vp_zoom;
            }
            KeyCode::Char('a') | KeyCode::Char('A') if is_cross => {
                app.thermal_vp_cam_x -= 2.0 / app.thermal_vp_zoom;
            }
            KeyCode::Char('d') | KeyCode::Char('D') if is_cross => {
                app.thermal_vp_cam_x += 2.0 / app.thermal_vp_zoom;
            }
            KeyCode::Char('+') | KeyCode::Char('=') if is_cross => {
                app.thermal_vp_zoom = (app.thermal_vp_zoom * 1.25).min(8.0);
            }
            KeyCode::Char('-') | KeyCode::Char('_') if is_cross => {
                app.thermal_vp_zoom = (app.thermal_vp_zoom / 1.25).max(0.25);
            }
            KeyCode::Up if is_cross => {
                app.thermal_vp_cursor_y = app.thermal_vp_cursor_y.saturating_sub(1);
            }
            KeyCode::Down if is_cross => {
                app.thermal_vp_cursor_y = app.thermal_vp_cursor_y.saturating_add(1);
            }
            KeyCode::Left if is_cross => {
                app.thermal_vp_cursor_x = app.thermal_vp_cursor_x.saturating_sub(1);
            }
            KeyCode::Right if is_cross => {
                app.thermal_vp_cursor_x = app.thermal_vp_cursor_x.saturating_add(1);
            }
            KeyCode::Tab if is_cross_or_iso => {
                // Cycle Y-slice forward
                app.thermal_cross_slice_y = app.thermal_cross_slice_y.saturating_add(1);
            }
            KeyCode::BackTab if is_cross_or_iso => {
                app.thermal_cross_slice_y = app.thermal_cross_slice_y.saturating_sub(1);
            }
            // ─── Isometric: Left/Right change Y-slice ──
            KeyCode::Left if is_cross_or_iso => {
                app.thermal_cross_slice_y = app.thermal_cross_slice_y.saturating_sub(1);
            }
            KeyCode::Right if is_cross_or_iso => {
                app.thermal_cross_slice_y = app.thermal_cross_slice_y.saturating_add(1);
            }
            // ─── Econ dashboard: Left/Right switch tab ──
            KeyCode::Left if is_econ_dashboard => {
                app.econ_dashboard_tab = app.econ_dashboard_tab.saturating_sub(1);
            }
            KeyCode::Right if is_econ_dashboard => {
                app.econ_dashboard_tab = (app.econ_dashboard_tab + 1).min(2);
            }
            // ─── MDP dashboard: Left/Right switch tab ──
            KeyCode::Left if is_mdp_dashboard => {
                app.mdp_dashboard_tab = app.mdp_dashboard_tab.saturating_sub(1);
            }
            KeyCode::Right if is_mdp_dashboard => {
                app.mdp_dashboard_tab = (app.mdp_dashboard_tab + 1).min(2);
            }
            // ─── Generic scroll for other overlays ──
            KeyCode::Up | KeyCode::Char('k') => {
                app.thermal_overlay_scroll = app.thermal_overlay_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.thermal_overlay_scroll = app.thermal_overlay_scroll.saturating_add(1);
            }
            _ => {}
        }
        return;
    }

    match &app.screen {
        Screen::MainMenu => handle_main_menu(app, key),
        Screen::CategoryMenu { .. } => handle_category_menu(app, key),
        Screen::MultiplyMenu => handle_multiply_menu(app, key),
        Screen::SizeInput => handle_size_input(app, key),
        Screen::InputMethodMenu => handle_input_method(app, key, terminal),
        Screen::NameInput => handle_name_input(app, key, terminal),
        Screen::ManualInputA { .. } => handle_manual_input_a(app, key),
        Screen::ManualInputB { .. } => handle_manual_input_b(app, key, terminal),
        Screen::Results { .. } => handle_results(app, key),
        Screen::History => handle_history(app, key, terminal),
        Screen::ViewerFileInput => handle_viewer_file_input(app, key),
        Screen::MatrixViewer => handle_matrix_viewer(app, key),
        Screen::DiffSizeInput => handle_diff_size_input(app, key),
        Screen::DiffAlgSelect => handle_diff_alg_select(app, key, terminal),
        Screen::DiffResults { .. } => handle_diff_results(app, key),
        Screen::FileCompareInputA => handle_file_compare_input_a(app, key),
        Screen::FileCompareInputB => handle_file_compare_input_b(app, key),
        Screen::FileCompareResults { .. } => handle_file_compare_results(app, key),
        Screen::ComingSoon(_) => {
            if matches!(key, KeyCode::Esc | KeyCode::Enter) {
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::FileBrowser { .. } => handle_file_browser(app, key),
        Screen::Computing { .. } | Screen::ThermalComputing => {
            if matches!(key, KeyCode::Esc) {
                // Cancel: drop the task (thread continues but result is ignored)
                app.compute_task = None;
                app.thermal_phase = None;
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        // Thermal simulation wizard screens
        Screen::ThermalFluidSelect => crate::thermal_ui::handle_thermal_fluid_select(app, key),
        Screen::ThermalGeometry => crate::thermal_ui::handle_thermal_geometry(app, key),
        Screen::ThermalTeg => crate::thermal_ui::handle_thermal_teg(app, key),
        Screen::ThermalHeatSources => crate::thermal_ui::handle_thermal_heat_sources(app, key),
        Screen::ThermalSolverSelect => crate::thermal_ui::handle_thermal_solver_select(app, key),
        Screen::ThermalConfirm => crate::thermal_ui::handle_thermal_confirm(app, key),
        Screen::ThermalResults { .. } => crate::thermal_ui::handle_thermal_results(app, key),
        Screen::ThermalViewer => crate::thermal_ui::handle_thermal_viewer(app, key),
        // Economics simulation wizard screens
        Screen::EconConfig => crate::economics_ui::handle_econ_config(app, key),
        Screen::EconShockSelect => crate::economics_ui::handle_econ_shock(app, key),
        Screen::EconConfirm => crate::economics_ui::handle_econ_confirm(app, key),
        Screen::EconComputing => {
            if matches!(key, KeyCode::Esc) {
                app.compute_task = None;
                app.econ_phase = None;
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::EconResults { .. } => crate::economics_ui::handle_econ_results(app, key),
        // MDP Kinematics wizard screens
        Screen::MdpConfig => crate::kinematics_ui::handle_mdp_config(app, key),
        Screen::MdpConfirm => crate::kinematics_ui::handle_mdp_confirm(app, key),
        Screen::MdpComputing => {
            if matches!(key, KeyCode::Esc) {
                app.compute_task = None;
                app.mdp_phase = None;
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::MdpResults { .. } => crate::kinematics_ui::handle_mdp_results(app, key),
        // Computer Vision wizard screens
        Screen::VisionConfig => crate::vision_ui::handle_vision_config(app, key),
        Screen::VisionKernelSelect => crate::vision_ui::handle_vision_kernel(app, key),
        Screen::VisionConfirm => crate::vision_ui::handle_vision_confirm(app, key),
        Screen::VisionComputing => {
            if matches!(key, KeyCode::Esc) {
                app.compute_task = None;
                app.vision_phase = None;
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::VisionResults { .. } => crate::vision_ui::handle_vision_results(app, key),
        // MHPS wizard screens
        Screen::MhpsMaterial => crate::mhps_ui::handle_mhps_material(app, key),
        Screen::MhpsGeometry => crate::mhps_ui::handle_mhps_geometry(app, key),
        Screen::MhpsHeatSources => crate::mhps_ui::handle_mhps_heat_sources(app, key),
        Screen::MhpsSimParams => crate::mhps_ui::handle_mhps_sim_params(app, key),
        Screen::MhpsConfirm => crate::mhps_ui::handle_mhps_confirm(app, key),
        Screen::MhpsComputing => {
            if matches!(key, KeyCode::Esc) {
                app.compute_task = None;
                app.mhps_phase = None;
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::MhpsResults { .. } => crate::mhps_ui::handle_mhps_results(app, key),
        Screen::SystemProfile { .. } => {
            if matches!(key, KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q')) {
                app.screen = Screen::CategoryMenu { category: MenuCategory::Engineering };
                app.category_menu_idx = 4;
            }
        }
        Screen::ComputeError(_) => {
            if matches!(key, KeyCode::Esc | KeyCode::Enter) {
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
    }
}

fn handle_main_menu(app: &mut App, key: KeyCode) {
    let items = MENU_ITEMS.len();
    match key {
        KeyCode::Up => {
            if app.main_menu_idx > 0 {
                app.main_menu_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.main_menu_idx < items - 1 {
                app.main_menu_idx += 1;
            }
        }
        KeyCode::Enter => select_menu_item(app),
        KeyCode::Char('h') | KeyCode::Char('H') => {
            app.overlay = Overlay::Help;
        }
        KeyCode::Char('a') | KeyCode::Char('A') => {
            app.overlay = Overlay::About;
        }
        KeyCode::Char('q') | KeyCode::Esc => app.running = false,
        KeyCode::Char(c) => {
            // Shortcut keys from MENU_ITEMS (checked first)
            for (i, item) in MENU_ITEMS.iter().enumerate() {
                if item.shortcut == Some(c) || item.shortcut == Some(c.to_ascii_uppercase()) {
                    app.main_menu_idx = i;
                    select_menu_item(app);
                    return;
                }
            }
            // Non-menu shortcuts
            match c {
                't' | 'T' => {
                    app.current_theme = app.current_theme.next();
                    let _ = app.io_tx.send(crate::io::IoTask::SaveConfig {
                        theme: app.current_theme.display_name().to_string(),
                        scale: app.terminal_scale.label().to_string(),
                    });
                }
                's' | 'S' => {
                    app.terminal_scale = app.terminal_scale.next();
                    let _ = app.io_tx.send(crate::io::IoTask::SaveConfig {
                        theme: app.current_theme.display_name().to_string(),
                        scale: app.terminal_scale.label().to_string(),
                    });
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn select_menu_item(app: &mut App) {
    // Check if selected item is a category — open submenu
    if let Some(cat) = MENU_ITEMS[app.main_menu_idx].category {
        app.category_menu_idx = 0;
        app.screen = Screen::CategoryMenu { category: cat };
        return;
    }

    match app.main_menu_idx {
        0 => {
            // Matrix Multiplication
            app.screen = Screen::MultiplyMenu;
            app.mult_menu_idx = 0;
        }
        1 => {
            // Algorithm Comparison (Diff Mode)
            app.diff_size_input.clear();
            app.screen = Screen::DiffSizeInput;
        }
        2 => {
            // Matrix File Comparison
            app.file_compare_path_a.clear();
            app.file_compare_path_b.clear();
            app.file_compare_matrix_a = None;
            app.file_compare_error = None;
            app.screen = Screen::FileCompareInputA;
        }
        3 => {
            // Performance Monitor — launch in separate window
            crate::monitor::spawn_monitor_window();
        }
        4 => {
            // Benchmark Suite
            app.screen = Screen::ComingSoon("Benchmark Suite".into());
        }
        5 => {
            // Matrix Viewer
            app.viewer_path_input.clear();
            app.screen = Screen::ViewerFileInput;
        }
        // 6 = Engineering (category, handled above)
        // 7 = Economics (category, handled above)
        8 => {
            // Computation History — only if non-empty
            if !app.session_history.is_empty() {
                app.history_selected = 0;
                app.screen = Screen::History;
            }
        }
        _ => {}
    }
}

fn init_thermal_wizard(app: &mut App) {
    app.thermal_fluid_idx = 0;
    let def = crate::thermal::config_tank_project_default();
    app.thermal_geometry_fields = [
        format!("{}", def.length_x),
        format!("{}", def.length_y),
        format!("{}", def.length_z),
        format!("{}", def.nx),
    ];
    app.thermal_geometry_active = 0;
    app.thermal_teg_fields = [
        format!("{}", def.teg.seebeck_coefficient),
        format!("{}", def.teg.internal_resistance),
        format!("{}", def.teg.load_resistance),
        format!("{}", def.teg.teg_area),
        format!("{}", def.teg.teg_thickness),
        format!("{}", def.teg.teg_thermal_conductivity),
    ];
    app.thermal_teg_active = 0;
    app.thermal_use_defaults = true;
    app.screen = Screen::ThermalFluidSelect;
}

fn handle_category_menu(app: &mut App, key: KeyCode) {
    let category = if let Screen::CategoryMenu { category } = app.screen {
        category
    } else {
        return;
    };
    let subitems = category.subitems();
    let count = subitems.len();

    match key {
        KeyCode::Up => {
            if app.category_menu_idx > 0 {
                app.category_menu_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.category_menu_idx < count - 1 {
                app.category_menu_idx += 1;
            }
        }
        KeyCode::Enter => select_category_item(app, category),
        KeyCode::Esc => {
            app.screen = Screen::MainMenu;
        }
        KeyCode::Char(c) => {
            // Shortcut keys within category
            for (i, sub) in subitems.iter().enumerate() {
                if sub.shortcut == Some(c) || sub.shortcut == Some(c.to_ascii_uppercase()) {
                    app.category_menu_idx = i;
                    select_category_item(app, category);
                    return;
                }
            }
        }
        _ => {}
    }
}

fn select_category_item(app: &mut App, category: MenuCategory) {
    match category {
        MenuCategory::Engineering => match app.category_menu_idx {
            0 => init_thermal_wizard(app),
            1 => {
                app.screen = Screen::MdpConfig;
                app.mdp_active_field = 0;
            }
            2 => {
                app.screen = Screen::VisionConfig;
                app.vision_active_field = 0;
            }
            3 => {
                app.screen = Screen::MhpsMaterial;
                app.mhps_material_idx = 1;
                app.mhps_active_field = 0;
            }
            4 => {
                let profile = crate::system_profiler::collect_system_profile(&app.sys_info);
                app.screen = Screen::SystemProfile { profile: Box::new(profile) };
            }
            5 => { app.overlay = Overlay::ThermalHelp; app.thermal_overlay_scroll = 0; }
            6 => { app.overlay = Overlay::EngineeringInfo; }
            _ => {}
        },
        MenuCategory::Economics => match app.category_menu_idx {
            0 => {
                app.screen = Screen::EconConfig;
                app.econ_active_field = 0;
            }
            1 => { app.overlay = Overlay::EconHelp; }
            2 => { app.overlay = Overlay::EconInfo; }
            _ => {}
        },
    }
}

fn handle_multiply_menu(app: &mut App, key: KeyCode) {
    let items = 6; // Naive, Strassen, Winograd, StrassenHybrid, WinogradHybrid, MKL

    match key {
        KeyCode::Up => {
            if app.mult_menu_idx > 0 {
                app.mult_menu_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.mult_menu_idx < items - 1 {
                app.mult_menu_idx += 1;
            }
        }
        KeyCode::Enter => {
            let choice = match app.mult_menu_idx {
                0 => AlgorithmChoice::Naive,
                1 => AlgorithmChoice::Strassen,
                2 => AlgorithmChoice::Winograd,
                3 => AlgorithmChoice::StrassenHybrid,
                4 => AlgorithmChoice::WinogradHybrid,
                5 => {
                    if !crate::algorithms::is_mkl_available() {
                        // MKL not available at runtime — show error, don't proceed
                        app.screen = Screen::ComputeError(
                            "Intel MKL is not available. Install Intel oneAPI MKL and ensure mkl_rt.2.dll is on PATH.".into()
                        );
                        return;
                    }
                    AlgorithmChoice::IntelMKL
                }
                _ => return,
            };
            app.algorithm_choice = choice;
            app.size_input.clear();
            app.screen = Screen::SizeInput;
        },
        KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

fn handle_size_input(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char(c) if c.is_ascii_digit() => {
            if app.size_input.len() < 6 {
                app.size_input.push(c);
            }
        }
        KeyCode::Backspace => {
            app.size_input.pop();
        }
        KeyCode::Enter => {
            if let Ok(n) = app.size_input.parse::<usize>() {
                if n > 0 && n <= 10000 {
                    app.chosen_size = n;
                    app.input_method_idx = 0;
                    app.screen = Screen::InputMethodMenu;
                }
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::MultiplyMenu;
            app.mult_menu_idx = 0;
        }
        _ => {}
    }
}

// ─── Name Input Handler ──────────────────────────────────────────────────────

fn handle_name_input(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    match key {
        // Accept alphanumeric chars, spaces, dashes, underscores for the name
        KeyCode::Char(c) if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' => {
            if app.pending_session_name.len() < 40 {
                app.pending_session_name.push(c);
            }
        }
        KeyCode::Backspace => {
            app.pending_session_name.pop();
        }
        KeyCode::Enter => {
            // Trim the name; empty = auto-generated ID will be used in BenchmarkData
            app.pending_session_name = app.pending_session_name.trim().to_string();
            if app.pending_is_random {
                dispatch_generation(app, terminal);
            } else {
                // Use stored matrices from manual input
                let a = app.pending_matrix_a.take()
                    .unwrap_or_else(|| Matrix::random(app.chosen_size, app.chosen_size, None).unwrap());
                let b = app.pending_matrix_b.take()
                    .unwrap_or_else(|| Matrix::random(app.chosen_size, app.chosen_size, None).unwrap());
                run_multiplication(app, a, b, None, terminal);
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::InputMethodMenu;
            app.input_method_idx = 0;
        }
        _ => {}
    }
}

fn handle_input_method(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let items = 3;
    match key {
        KeyCode::Up => {
            if app.input_method_idx > 0 {
                app.input_method_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.input_method_idx < items - 1 {
                app.input_method_idx += 1;
            }
        }
        KeyCode::Enter => match app.input_method_idx {
            0 => {
                // Route through NameInput so the user can label this computation
                app.pending_session_name.clear();
                app.pending_is_random = true;
                app.screen = Screen::NameInput;
            }
            1 => {
                app.manual_buffer.clear();
                app.manual_row = 0;
                app.manual_col = 0;
                app.manual_data = Vec::new();
                app.manual_name = "A".to_string();
                app.screen = Screen::ManualInputA {
                    name: "A".to_string(),
                };
            }
            2 => {
                app.screen = Screen::SizeInput;
                app.size_input.clear();
            }
            _ => {}
        },
        KeyCode::Esc => {
            app.screen = Screen::SizeInput;
            app.size_input.clear();
        }
        _ => {}
    }
}

fn handle_manual_input_a(app: &mut App, key: KeyCode) {
    let n = app.chosen_size;
    let total = n * n;

    match key {
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' || c == ' ' => {
            app.manual_buffer.push(c);
        }
        KeyCode::Backspace => {
            app.manual_buffer.pop();
        }
        KeyCode::Enter => {
            let nums: Vec<f64> = app
                .manual_buffer
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            for val in &nums {
                if app.manual_data.len() < total {
                    app.manual_data.push(*val);
                    app.manual_col += 1;
                    if app.manual_col >= n {
                        app.manual_col = 0;
                        app.manual_row += 1;
                    }
                }
            }
            app.manual_buffer.clear();

            if app.manual_data.len() >= total {
                let name_a = app.manual_name.clone();
                let data = app.manual_data.clone();
                let matrix_a = Matrix::from_flat(n, n, data).unwrap();

                app.manual_buffer.clear();
                app.manual_row = 0;
                app.manual_col = 0;
                app.manual_data = Vec::new();
                app.manual_name = "B".to_string();
                app.screen = Screen::ManualInputB {
                    name_a,
                    matrix_a,
                    name: "B".to_string(),
                };
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::InputMethodMenu;
            app.input_method_idx = 0;
        }
        _ => {}
    }
}

fn handle_manual_input_b(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let n = app.chosen_size;
    let total = n * n;

    match key {
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' || c == ' ' => {
            app.manual_buffer.push(c);
        }
        KeyCode::Backspace => {
            app.manual_buffer.pop();
        }
        KeyCode::Enter => {
            let nums: Vec<f64> = app
                .manual_buffer
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            for val in &nums {
                if app.manual_data.len() < total {
                    app.manual_data.push(*val);
                    app.manual_col += 1;
                    if app.manual_col >= n {
                        app.manual_col = 0;
                        app.manual_row += 1;
                    }
                }
            }
            app.manual_buffer.clear();

            if app.manual_data.len() >= total {
                if let Screen::ManualInputB { matrix_a, .. } = &app.screen {
                    let a = matrix_a.clone();
                    let b = Matrix::from_flat(n, n, app.manual_data.clone()).unwrap();
                    app.pending_session_name.clear();
                    app.pending_is_random = false;
                    app.pending_matrix_a = Some(a);
                    app.pending_matrix_b = Some(b);
                    app.screen = Screen::NameInput;
                }
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::InputMethodMenu;
            app.input_method_idx = 0;
        }
        _ => {}
    }
}

fn handle_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if !app.csv_saved {
                let record = if let Screen::Results { ref data } = app.screen {
                    Some(crate::io::CsvRecord {
                        timestamp: crate::io::timestamp_now(),
                        algorithm: data.algorithm.clone(),
                        size_m: data.size,
                        size_n: data.size,
                        size_p: data.size,
                        compute_time_ms: data.compute_time_ms,
                        total_time_ms: data.gen_time_ms.unwrap_or(0.0)
                            + data.padding_time_ms
                            + data.compute_time_ms
                            + data.unpadding_time_ms,
                        gflops: data.gflops,
                        simd_level: data.simd_level.clone(),
                        threads: data.threads,
                        tile_size: Some(crate::common::DEFAULT_TILE_SIZE),
                        peak_ram_mb: data.peak_ram_mb,
                    })
                } else {
                    None
                };
                if let Some(record) = record {
                    if crate::io::append_csv("flust_results.csv", &record).is_ok() {
                        app.csv_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('m') | KeyCode::Char('M') => {
            if !app.matrix_saved {
                let save_info = if let Screen::Results { ref data } = app.screen {
                    data.result_matrix.as_ref().map(|mat| {
                        let filename = format!("flust_result_{}x{}.csv", data.size, data.size);
                        let meta = crate::io::MatrixMetadata {
                            algorithm: Some(data.algorithm.clone()),
                            timestamp: Some(crate::io::timestamp_now()),
                            cpu: Some(app.sys_info.cpu_brand.clone()),
                            simd: Some(data.simd_level.clone()),
                            threads: Some(data.threads),
                            compute_ms: Some(data.compute_time_ms),
                            size_rows: Some(data.size),
                            size_cols: Some(data.size),
                            gflops: Some(data.gflops),
                            peak_ram_mb: Some(data.peak_ram_mb),
                            computation_id: Some(data.computation_id.clone()),
                            machine: Some(data.machine_name.clone()),
                        };
                        (filename, mat.clone(), meta)
                    })
                } else {
                    None
                };
                if let Some((filename, mat, meta)) = save_info {
                    app.io_status = Some("Saving matrix...".into());
                    let _ = app.io_tx.send(crate::io::IoTask::SaveMatrix {
                        path: filename,
                        matrix: mat,
                        meta: Some(meta),
                    });
                }
            }
        }
        KeyCode::Char('v') | KeyCode::Char('V') => {
            if let Screen::Results { ref data } = app.screen {
                if let Some(ref mat) = data.result_matrix {
                    let meta = crate::io::MatrixMetadata {
                        algorithm: Some(data.algorithm.clone()),
                        timestamp: Some(crate::io::timestamp_now()),
                        cpu: Some(app.sys_info.cpu_brand.clone()),
                        simd: Some(data.simd_level.clone()),
                        threads: Some(data.threads),
                        compute_ms: Some(data.compute_time_ms),
                        size_rows: Some(data.size),
                        size_cols: Some(data.size),
                        gflops: Some(data.gflops),
                        peak_ram_mb: Some(data.peak_ram_mb),
                        computation_id: Some(data.computation_id.clone()),
                        machine: Some(data.machine_name.clone()),
                    };
                    app.viewer_matrix = Some(mat.clone());
                    app.viewer_filename = format!("Result_{}x{}", data.size, data.size);
                    app.viewer_loaded_metadata = Some(meta);
                    app.viewer_scroll_row = 0;
                    app.viewer_scroll_col = 0;
                    app.viewer_cursor_row = 0;
                    app.viewer_cursor_col = 0;
                    app.viewer_unsaved_changes = false;
                    app.viewer_stats = Some(MatrixStats::compute(mat));
                    app.screen = Screen::MatrixViewer;
                }
            }
        }
        KeyCode::Esc | KeyCode::Enter => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

fn handle_viewer_file_input(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char(c) => {
            app.viewer_path_input.push(c);
        }
        KeyCode::Backspace => {
            app.viewer_path_input.pop();
        }
        KeyCode::Enter => {
            if !app.viewer_path_input.is_empty() {
                // Try thermal CSV first
                if crate::thermal_export::detect_thermal_csv(&app.viewer_path_input) {
                    match crate::thermal_export::load_thermal_csv(&app.viewer_path_input) {
                        Ok(data) => {
                            app.thermal_view_data = Some(data);
                            app.viewer_filename = app.viewer_path_input.clone();
                            app.screen = Screen::ThermalViewer;
                            return;
                        }
                        Err(_) => {}
                    }
                }
                // Otherwise try regular matrix CSV
                match crate::io::load_matrix_csv_with_metadata(&app.viewer_path_input) {
                    Ok((mat, meta)) => {
                        app.viewer_stats = Some(MatrixStats::compute(&mat));
                        app.viewer_filename = app.viewer_path_input.clone();
                        app.viewer_loaded_metadata = meta;
                        app.viewer_matrix = Some(mat);
                        app.viewer_scroll_row = 0;
                        app.viewer_scroll_col = 0;
                        app.viewer_cursor_row = 0;
                        app.viewer_cursor_col = 0;
                        app.viewer_unsaved_changes = false;
                        app.screen = Screen::MatrixViewer;
                    }
                    Err(_) => {
                        // Stay on file input — user can try again
                    }
                }
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        KeyCode::Tab => {
            refresh_fb_entries(app);
            app.screen = Screen::FileBrowser { return_to: FileBrowserReturn::MatrixViewer };
        }
        _ => {}
    }
}

fn handle_matrix_viewer(app: &mut App, key: KeyCode) {
    // Handle exit-warning popup first
    if app.viewer_exit_warning {
        match key {
            KeyCode::Char('s') | KeyCode::Char('S') => {
                app.viewer_exit_warning = false;
                save_viewer_matrix(app);
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                app.viewer_exit_warning = false;
                close_viewer(app);
            }
            KeyCode::Esc => {
                app.viewer_exit_warning = false;
            }
            _ => {}
        }
        return;
    }

    // Edit mode: character input goes to buffer
    if app.viewer_edit_mode {
        match key {
            KeyCode::Char(c) if "0123456789.-eE+".contains(c) => {
                app.viewer_edit_buffer.push(c);
            }
            KeyCode::Backspace => {
                app.viewer_edit_buffer.pop();
            }
            KeyCode::Enter => {
                if let Ok(val) = app.viewer_edit_buffer.trim().parse::<f64>() {
                    if let Some(ref mut mat) = app.viewer_matrix {
                        mat.set(app.viewer_cursor_row, app.viewer_cursor_col, val);
                        app.viewer_unsaved_changes = true;
                        // Recompute stats after edit
                        app.viewer_stats = Some(MatrixStats::compute(mat));
                    }
                }
                app.viewer_edit_mode = false;
                app.viewer_edit_buffer.clear();
            }
            KeyCode::Esc => {
                app.viewer_edit_mode = false;
                app.viewer_edit_buffer.clear();
            }
            _ => {}
        }
        return;
    }

    let mat = match &app.viewer_matrix {
        Some(m) => m,
        None => {
            app.screen = Screen::MainMenu;
            return;
        }
    };
    let rows = mat.rows();
    let cols = mat.cols();
    let vis_r = app.viewer_visible_rows.get().max(1);
    let vis_c = app.viewer_visible_cols.get().max(1);

    match key {
        KeyCode::Up => {
            app.viewer_cursor_row = app.viewer_cursor_row.saturating_sub(1);
            if app.viewer_cursor_row < app.viewer_scroll_row {
                app.viewer_scroll_row = app.viewer_cursor_row;
            }
        }
        KeyCode::Down => {
            if app.viewer_cursor_row + 1 < rows {
                app.viewer_cursor_row += 1;
                if app.viewer_cursor_row >= app.viewer_scroll_row + vis_r {
                    app.viewer_scroll_row = app.viewer_cursor_row.saturating_sub(vis_r - 1);
                }
            }
        }
        KeyCode::Left => {
            app.viewer_cursor_col = app.viewer_cursor_col.saturating_sub(1);
            if app.viewer_cursor_col < app.viewer_scroll_col {
                app.viewer_scroll_col = app.viewer_cursor_col;
            }
        }
        KeyCode::Right => {
            if app.viewer_cursor_col + 1 < cols {
                app.viewer_cursor_col += 1;
                if app.viewer_cursor_col >= app.viewer_scroll_col + vis_c {
                    app.viewer_scroll_col = app.viewer_cursor_col.saturating_sub(vis_c - 1);
                }
            }
        }
        KeyCode::PageUp => {
            app.viewer_cursor_row = app.viewer_cursor_row.saturating_sub(vis_r);
            app.viewer_scroll_row = app.viewer_scroll_row.saturating_sub(vis_r);
        }
        KeyCode::PageDown => {
            app.viewer_cursor_row = (app.viewer_cursor_row + vis_r).min(rows.saturating_sub(1));
            app.viewer_scroll_row = (app.viewer_scroll_row + vis_r).min(rows.saturating_sub(1));
        }
        KeyCode::Home => {
            app.viewer_cursor_row = 0;
            app.viewer_cursor_col = 0;
            app.viewer_scroll_row = 0;
            app.viewer_scroll_col = 0;
        }
        KeyCode::End => {
            app.viewer_cursor_row = rows.saturating_sub(1);
            app.viewer_cursor_col = cols.saturating_sub(1);
            app.viewer_scroll_row = rows.saturating_sub(1);
            app.viewer_scroll_col = cols.saturating_sub(1);
        }
        KeyCode::Char('e') | KeyCode::Char('E') => {
            // Enter edit mode for current cell
            app.viewer_edit_mode = true;
            if let Some(ref mat) = app.viewer_matrix {
                let val = mat.get(app.viewer_cursor_row, app.viewer_cursor_col);
                app.viewer_edit_buffer =
                    format!("{:.prec$}", val, prec = app.viewer_precision);
            }
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            save_viewer_matrix(app);
        }
        KeyCode::Char('h') | KeyCode::Char('H') => {
            app.viewer_highlight = !app.viewer_highlight;
        }
        KeyCode::Char('+') | KeyCode::Char('=') => {
            app.viewer_precision = (app.viewer_precision + 1).min(8);
        }
        KeyCode::Char('-') => {
            app.viewer_precision = app.viewer_precision.saturating_sub(1).max(1);
        }
        KeyCode::Esc | KeyCode::Char('q') => {
            if app.viewer_unsaved_changes {
                app.viewer_exit_warning = true;
            } else {
                close_viewer(app);
            }
        }
        _ => {}
    }
}

fn save_viewer_matrix(app: &mut App) {
    if let Some(ref mat) = app.viewer_matrix {
        let path = if app.viewer_filename.contains('.') {
            app.viewer_filename.clone()
        } else {
            format!("{}.csv", app.viewer_filename)
        };
        app.io_status = Some("Saving matrix...".into());
        let _ = app.io_tx.send(crate::io::IoTask::SaveMatrix {
            path,
            matrix: mat.clone(),
            meta: app.viewer_loaded_metadata.clone(),
        });
        app.viewer_unsaved_changes = false;
    }
}

fn close_viewer(app: &mut App) {
    app.viewer_matrix = None;
    app.viewer_stats = None;
    app.viewer_loaded_metadata = None;
    app.viewer_unsaved_changes = false;
    app.viewer_edit_mode = false;
    app.viewer_edit_buffer.clear();
    app.viewer_cursor_row = 0;
    app.viewer_cursor_col = 0;
    app.viewer_scroll_row = 0;
    app.viewer_scroll_col = 0;
    app.screen = Screen::MainMenu;
    app.main_menu_idx = 0;
}

fn handle_diff_size_input(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char(c) if c.is_ascii_digit() => {
            if app.diff_size_input.len() < 6 {
                app.diff_size_input.push(c);
            }
        }
        KeyCode::Backspace => {
            app.diff_size_input.pop();
        }
        KeyCode::Enter => {
            if let Ok(n) = app.diff_size_input.parse::<usize>() {
                if n > 0 && n <= 10000 {
                    app.chosen_size = n;
                    app.diff_selecting_which = 1;
                    app.diff_select_idx = 0;
                    app.screen = Screen::DiffAlgSelect;
                }
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

fn handle_diff_alg_select(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let items = 3; // Naive, Strassen, Winograd
    match key {
        KeyCode::Up => {
            if app.diff_select_idx > 0 {
                app.diff_select_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.diff_select_idx < items - 1 {
                app.diff_select_idx += 1;
            }
        }
        KeyCode::Enter => {
            let choice = match app.diff_select_idx {
                0 => AlgorithmChoice::Naive,
                1 => AlgorithmChoice::Strassen,
                2 => AlgorithmChoice::Winograd,
                3 => AlgorithmChoice::StrassenHybrid,
                _ => AlgorithmChoice::WinogradHybrid,
            };

            if app.diff_selecting_which == 1 {
                app.diff_alg1 = choice;
                app.diff_selecting_which = 2;
                app.diff_select_idx = 0;
            } else {
                // Prevent same algorithm
                if choice == app.diff_alg1 {
                    return; // ignore — can't pick the same
                }
                app.diff_alg2 = choice;
                run_diff(app, terminal);
            }
        }
        KeyCode::Esc => {
            if app.diff_selecting_which == 2 {
                app.diff_selecting_which = 1;
                app.diff_select_idx = 0;
            } else {
                app.screen = Screen::DiffSizeInput;
                app.diff_size_input.clear();
            }
        }
        _ => {}
    }
}

fn handle_diff_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Esc | KeyCode::Enter => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// ─── File Comparison Handlers ───────────────────────────────────────────────

fn handle_file_compare_input_a(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char(c) => {
            app.file_compare_path_a.push(c);
            app.file_compare_error = None;
        }
        KeyCode::Backspace => {
            app.file_compare_path_a.pop();
            app.file_compare_error = None;
        }
        KeyCode::Enter => {
            if !app.file_compare_path_a.is_empty() {
                match crate::io::load_matrix_csv(&app.file_compare_path_a) {
                    Ok(mat) => {
                        app.file_compare_matrix_a = Some(mat);
                        app.file_compare_error = None;
                        app.file_compare_path_b.clear();
                        app.screen = Screen::FileCompareInputB;
                    }
                    Err(e) => {
                        app.file_compare_error = Some(format!("Error: {e}"));
                    }
                }
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        KeyCode::Tab => {
            refresh_fb_entries(app);
            app.screen = Screen::FileBrowser { return_to: FileBrowserReturn::FileCompareA };
        }
        _ => {}
    }
}

fn handle_file_compare_input_b(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char(c) => {
            app.file_compare_path_b.push(c);
            app.file_compare_error = None;
        }
        KeyCode::Backspace => {
            app.file_compare_path_b.pop();
            app.file_compare_error = None;
        }
        KeyCode::Enter => {
            if !app.file_compare_path_b.is_empty() {
                match crate::io::load_matrix_csv(&app.file_compare_path_b) {
                    Ok(mat_b) => {
                        if let Some(ref mat_a) = app.file_compare_matrix_a {
                            if mat_a.rows() != mat_b.rows() || mat_a.cols() != mat_b.cols() {
                                app.file_compare_error = Some(format!(
                                    "Dimension mismatch: A is {}x{}, B is {}x{}",
                                    mat_a.rows(), mat_a.cols(), mat_b.rows(), mat_b.cols()
                                ));
                                return;
                            }
                            let result = algorithms::compare_matrices_scientific(
                                mat_a, &mat_b, crate::common::EPSILON,
                            );
                            let data = FileCompareData {
                                path_a: app.file_compare_path_a.clone(),
                                path_b: app.file_compare_path_b.clone(),
                                dims_a: (mat_a.rows(), mat_a.cols()),
                                dims_b: (mat_b.rows(), mat_b.cols()),
                                result,
                            };
                            app.file_compare_matrix_a = None; // free memory
                            app.screen = Screen::FileCompareResults { data };
                        }
                    }
                    Err(e) => {
                        app.file_compare_error = Some(format!("Error: {e}"));
                    }
                }
            }
        }
        KeyCode::Esc => {
            app.file_compare_path_a.clear();
            app.file_compare_matrix_a = None;
            app.screen = Screen::FileCompareInputA;
        }
        KeyCode::Tab => {
            refresh_fb_entries(app);
            app.screen = Screen::FileBrowser { return_to: FileBrowserReturn::FileCompareB };
        }
        _ => {}
    }
}

fn handle_file_compare_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Esc | KeyCode::Enter => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

fn run_diff(app: &mut App, _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    let n = app.chosen_size;
    let alg1 = app.diff_alg1;
    let alg2 = app.diff_alg2;
    let simd = app.sys_info.simd_level;
    let alg1_name = alg1.display_name().to_string();
    let alg2_name = alg2.display_name().to_string();

    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            progress_clone.set(2, 100);
            // Generate matrices with fixed seed — both algorithms get identical input
            let a = Matrix::random(n, n, Some(42)).unwrap();
            let b = Matrix::random(n, n, Some(43)).unwrap();
            progress_clone.set(5, 100);

            // Run algorithm 1 (progress 5..50)
            let start1 = Instant::now();
            let (result1, _, _) = run_algorithm_impl(alg1, &a, &b, simd, &progress_clone, 5, 45);
            let time1 = start1.elapsed().as_secs_f64() * 1000.0;

            // Run algorithm 2 (progress 50..100)
            let start2 = Instant::now();
            let (result2, _, _) = run_algorithm_impl(alg2, &a, &b, simd, &progress_clone, 50, 50);
            let time2 = start2.elapsed().as_secs_f64() * 1000.0;

            progress_clone.set(100, 100);

            tx.send(ComputeResult::Diff {
                result1,
                result2,
                time1_ms: time1,
                time2_ms: time2,
            }).ok();
        }));
        if let Err(panic_info) = outcome {
            let msg = extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    let context = ComputeContext {
        algorithm_choice: alg1,
        algorithm_name: format!("{} vs {}", alg1_name, alg2_name),
        size: n,
        gen_time_ms: None,
        simd_level: simd,
        is_diff: true,
        diff_alg1: Some(alg1),
        diff_alg2: Some(alg2),
    };

    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context,
        _join_handle: handle,
        child_process: None,
        temp_dir: None,
        compute_request: None,
    });

    app.screen = Screen::Computing {
        algorithm: format!("{alg1_name} vs {alg2_name}  {n}\u{00d7}{n}"),
    };
}

/// Run a single algorithm with progress reporting.
/// `offset` and `scale` define the progress window within [0..total].
/// Returns (result, padding_ms, unpadding_ms).
fn run_algorithm_impl(
    alg: AlgorithmChoice,
    a: &Matrix,
    b: &Matrix,
    simd: crate::common::SimdLevel,
    progress: &ProgressHandle,
    offset: u32,
    scale: u32,
) -> (Matrix, f64, f64) {
    match alg {
        AlgorithmChoice::Naive => {
            let r = algorithms::multiply_naive_with_progress(a, b, progress, offset, scale);
            (r, 0.0, 0.0)
        }
        AlgorithmChoice::Strassen => {
            progress.set(offset + scale / 10, 100); // 10% — starting
            let result = algorithms::multiply_strassen_padded(a, b, STRASSEN_THRESHOLD, simd);
            progress.set(offset + scale, 100);
            result
        }
        AlgorithmChoice::Winograd => {
            progress.set(offset + scale / 10, 100); // 10% — starting
            let result = algorithms::multiply_winograd_padded(a, b, STRASSEN_THRESHOLD, simd);
            progress.set(offset + scale, 100);
            result
        }
        AlgorithmChoice::StrassenHybrid => {
            progress.set(offset + scale / 10, 100);
            let result = algorithms::multiply_strassen_hybrid(a, b, STRASSEN_THRESHOLD, simd);
            progress.set(offset + scale, 100);
            result
        }
        AlgorithmChoice::WinogradHybrid => {
            progress.set(offset + scale / 10, 100);
            let result = algorithms::multiply_winograd_hybrid(a, b, STRASSEN_THRESHOLD, simd);
            progress.set(offset + scale, 100);
            result
        }
        AlgorithmChoice::IntelMKL => {
            // Unwrap via panic — caught by catch_unwind in the caller thread.
            let r = algorithms::multiply_mkl_with_progress(a, b, progress)
                .unwrap_or_else(|e| panic!("MKL DGEMM failed: {e}"));
            (r, 0.0, 0.0)
        }
    }
}

fn handle_history(
    app: &mut App,
    key: KeyCode,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let len = app.session_history.len();
    if len == 0 {
        app.screen = Screen::MainMenu;
        return;
    }
    match key {
        KeyCode::Up => {
            if app.history_selected > 0 {
                app.history_selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.history_selected < len - 1 {
                app.history_selected += 1;
            }
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            // Re-run the selected computation with random matrices
            let config = app.session_history.entries[app.history_selected].config.clone();
            app.algorithm_choice = config.algorithm;
            app.chosen_size = config.size;
            dispatch_generation(app, terminal);
        }
        KeyCode::Char('d') | KeyCode::Char('D') => {
            app.session_history.entries.remove(app.history_selected);
            if app.session_history.is_empty() {
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            } else {
                app.history_selected = app.history_selected.min(app.session_history.len() - 1);
            }
        }
        KeyCode::Esc | KeyCode::Char('q') => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// ─── Computation (non-blocking, background thread) ──────────────────────────

/// Sample current-process RSS in bytes (fast — single process refresh).
pub(crate) fn sample_rss_bytes() -> u64 {
    let pid = match sysinfo::get_current_pid() {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let mut sys = SysinfoSystem::new();
    sys.refresh_process(pid);
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// Sample average CPU frequency across all logical cores in MHz.
pub(crate) fn sample_avg_freq_mhz() -> f64 {
    let mut sys = SysinfoSystem::new();
    sys.refresh_cpu();
    let cpus = sys.cpus();
    if cpus.is_empty() {
        return 0.0;
    }
    cpus.iter().map(|c| c.frequency() as f64).sum::<f64>() / cpus.len() as f64
}

/// Route computation to either process-isolated (MKL) or in-process (all others).
fn dispatch_generation(app: &mut App, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    if matches!(app.algorithm_choice, AlgorithmChoice::IntelMKL) {
        run_generation_isolated(app, terminal);
        return;
    }
    run_generation(app, terminal);
}

fn run_generation(app: &mut App, _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    let n = app.chosen_size;
    let alg = app.algorithm_choice;
    let alg_name = alg.display_name().to_string();
    let simd = app.sys_info.simd_level;

    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            progress_clone.set(2, 100);
            let gen_start = Instant::now();
            let a = Matrix::random(n, n, None).unwrap();
            let b = Matrix::random(n, n, None).unwrap();
            let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
            progress_clone.set(10, 100);

            let ram_before = sample_rss_bytes();
            let freq_pre = sample_avg_freq_mhz();

            let start = Instant::now();
            let (result, padding_ms, unpadding_ms) = run_algorithm_impl(
                alg, &a, &b, simd, &progress_clone, 10, 90,
            );
            let compute_ms = start.elapsed().as_secs_f64() * 1000.0;

            let freq_post = sample_avg_freq_mhz();
            let ram_after = sample_rss_bytes();
            let peak_ram_bytes = ram_after.saturating_sub(ram_before);
            let avg_freq_ghz = if freq_pre > 0.0 && freq_post > 0.0 {
                (freq_pre + freq_post) / 2.0 / 1000.0
            } else {
                0.0
            };
            progress_clone.set(100, 100);

            tx.send(ComputeResult::Multiply {
                result,
                padding_ms: padding_ms + gen_ms, // include gen time in padding
                unpadding_ms,
                compute_ms,
                peak_ram_bytes,
                avg_freq_ghz,
            }).ok();
        }));
        if let Err(panic_info) = outcome {
            let msg = extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    let context = ComputeContext {
        algorithm_choice: alg,
        algorithm_name: alg_name.clone(),
        size: n,
        gen_time_ms: None, // gen time will be baked into padding_ms
        simd_level: simd,
        is_diff: false,
        diff_alg1: None,
        diff_alg2: None,
    };

    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context,
        _join_handle: handle,
        child_process: None,
        temp_dir: None,
        compute_request: None,
    });

    app.screen = Screen::Computing {
        algorithm: format!("{alg_name}  {n}\u{00d7}{n}"),
    };
}

fn run_multiplication(
    app: &mut App,
    a: Matrix,
    b: Matrix,
    gen_time_ms: Option<f64>,
    _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let n = app.chosen_size;
    let alg = app.algorithm_choice;
    let alg_name = alg.display_name().to_string();
    let simd = app.sys_info.simd_level;

    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            progress_clone.set(5, 100);
            let ram_before = sample_rss_bytes();
            let freq_pre = sample_avg_freq_mhz();

            let start = Instant::now();
            let (result, padding_ms, unpadding_ms) = run_algorithm_impl(
                alg, &a, &b, simd, &progress_clone, 5, 95,
            );
            let compute_ms = start.elapsed().as_secs_f64() * 1000.0;

            let freq_post = sample_avg_freq_mhz();
            let ram_after = sample_rss_bytes();
            let peak_ram_bytes = ram_after.saturating_sub(ram_before);
            let avg_freq_ghz = if freq_pre > 0.0 && freq_post > 0.0 {
                (freq_pre + freq_post) / 2.0 / 1000.0
            } else {
                0.0
            };
            progress_clone.set(100, 100);

            tx.send(ComputeResult::Multiply {
                result,
                padding_ms,
                unpadding_ms,
                compute_ms,
                peak_ram_bytes,
                avg_freq_ghz,
            }).ok();
        }));
        if let Err(panic_info) = outcome {
            let msg = extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    let context = ComputeContext {
        algorithm_choice: alg,
        algorithm_name: alg_name.clone(),
        size: n,
        gen_time_ms,
        simd_level: simd,
        is_diff: false,
        diff_alg1: None,
        diff_alg2: None,
    };

    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context,
        _join_handle: handle,
        child_process: None,
        temp_dir: None,
        compute_request: None,
    });

    app.screen = Screen::Computing {
        algorithm: format!("{alg_name}  {n}\u{00d7}{n}"),
    };
}

// ─── Process-Isolated Computation (MKL) ─────────────────────────────────────
//
// Spawns the same binary with --compute flag. Matrix data is transferred via
// binary temp files. If the child process segfaults (e.g. MKL FFI crash),
// the parent TUI survives and shows an error.

fn run_generation_isolated(app: &mut App, _terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    use crate::compute_worker;

    let n = app.chosen_size;
    let alg_name = "Intel MKL (DGEMM)".to_string();
    let simd = app.sys_info.simd_level;

    // Set up temp directory
    let temp_dir = match compute_worker::make_compute_temp_dir() {
        Ok(d) => d,
        Err(e) => {
            app.screen = Screen::ComputeError(format!("Failed to create temp dir: {e}"));
            return;
        }
    };

    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let (tx, rx) = mpsc::channel();

    let a_path = temp_dir.join("matrix_a.bin");
    let b_path = temp_dir.join("matrix_b.bin");
    let result_path = temp_dir.join("result.bin");
    let metrics_path = temp_dir.join("metrics.json");
    let progress_path = temp_dir.join("progress.txt");
    let request_path = temp_dir.join("request.json");

    // Generate matrices in parent and save to temp files
    let gen_start = Instant::now();
    let a = match Matrix::random(n, n, None) {
        Ok(m) => m,
        Err(e) => {
            app.screen = Screen::ComputeError(format!("Matrix generation failed: {e}"));
            compute_worker::cleanup_temp_dir(&temp_dir);
            return;
        }
    };
    let b = match Matrix::random(n, n, None) {
        Ok(m) => m,
        Err(e) => {
            app.screen = Screen::ComputeError(format!("Matrix generation failed: {e}"));
            compute_worker::cleanup_temp_dir(&temp_dir);
            return;
        }
    };
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

    if let Err(e) = compute_worker::save_matrix_binary(&a_path, &a) {
        app.screen = Screen::ComputeError(format!("Failed to write matrix A: {e}"));
        compute_worker::cleanup_temp_dir(&temp_dir);
        return;
    }
    if let Err(e) = compute_worker::save_matrix_binary(&b_path, &b) {
        app.screen = Screen::ComputeError(format!("Failed to write matrix B: {e}"));
        compute_worker::cleanup_temp_dir(&temp_dir);
        return;
    }

    let request = compute_worker::ComputeRequest {
        algorithm: "mkl".into(),
        simd_level: simd.display_name().to_string(),
        matrix_a_path: a_path.to_string_lossy().into_owned(),
        matrix_b_path: b_path.to_string_lossy().into_owned(),
        result_path: result_path.to_string_lossy().into_owned(),
        metrics_path: metrics_path.to_string_lossy().into_owned(),
        progress_path: progress_path.to_string_lossy().into_owned(),
    };

    let request_json = serde_json::to_string(&request).unwrap_or_default();
    if let Err(e) = std::fs::write(&request_path, &request_json) {
        app.screen = Screen::ComputeError(format!("Failed to write request: {e}"));
        compute_worker::cleanup_temp_dir(&temp_dir);
        return;
    }

    progress.set(5, 100);

    // Spawn child process (headless — no new console).
    // Redirect stderr to a log file for MKL diagnostics.
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("flust"));
    let stderr_file = std::fs::File::create(temp_dir.join("stderr.log"))
        .map(std::process::Stdio::from)
        .unwrap_or_else(|_| std::process::Stdio::null());
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--compute")
        .arg("--request-file")
        .arg(request_path.to_string_lossy().as_ref())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(stderr_file);

    // Ensure child process can find MKL kernel DLLs (mkl_avx2.2.dll etc.)
    if let Some(augmented_path) = crate::compute_worker::get_mkl_augmented_path() {
        cmd.env("PATH", augmented_path);
    }

    let child = cmd.spawn();

    let child = match child {
        Ok(c) => c,
        Err(e) => {
            app.screen = Screen::ComputeError(format!("Failed to spawn compute worker: {e}"));
            compute_worker::cleanup_temp_dir(&temp_dir);
            return;
        }
    };

    // Dummy thread — needed to satisfy ComputeTask's _join_handle field.
    // The real work happens in the child process.
    let handle = std::thread::spawn(move || {
        // This thread just sends nothing — the parent polls the child process directly.
        drop(tx);
        drop(progress_clone);
    });

    let context = ComputeContext {
        algorithm_choice: AlgorithmChoice::IntelMKL,
        algorithm_name: alg_name.clone(),
        size: n,
        gen_time_ms: Some(gen_ms),
        simd_level: simd,
        is_diff: false,
        diff_alg1: None,
        diff_alg2: None,
    };

    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context,
        _join_handle: handle,
        child_process: Some(child),
        temp_dir: Some(temp_dir),
        compute_request: Some(request),
    });

    app.screen = Screen::Computing {
        algorithm: format!("{alg_name}  {n}\u{00d7}{n}"),
    };
}

// ─── Background I/O Completion ───────────────────────────────────────────────

fn check_io_completion(app: &mut App) {
    use crate::io::IoResult;
    while let Ok(result) = app.io_rx.try_recv() {
        match result {
            IoResult::SaveDone { path, error } => {
                app.io_status = None;
                if let Some(err) = error {
                    app.fb_error = Some(format!("Save failed: {err}"));
                } else {
                    // Mark matrix as saved on the current results screen
                    app.matrix_saved = true;
                }
            }
            IoResult::LoadDone { path: _, result: load_result } => {
                app.io_status = None;
                match load_result {
                    Ok((matrix, meta)) => {
                        app.viewer_loaded_metadata = meta;
                        app.viewer_stats = Some(MatrixStats::compute(&matrix));
                        app.viewer_scroll_row = 0;
                        app.viewer_scroll_col = 0;
                        app.viewer_cursor_row = 0;
                        app.viewer_cursor_col = 0;
                        app.viewer_matrix = Some(matrix);
                        app.screen = Screen::MatrixViewer;
                    }
                    Err(err) => {
                        app.fb_error = Some(format!("Load failed: {err}"));
                    }
                }
            }
            IoResult::HistoryAppended | IoResult::CsvSaved { .. } | IoResult::ConfigSaved => {
                // Silent fire-and-forget
            }
        }
    }
}

// ─── Background Task Completion ──────────────────────────────────────────────

fn check_compute_completion(app: &mut App) {
    // ── Process-isolated path: poll child process ──
    if let Some(ref mut task) = app.compute_task {
        if task.child_process.is_some() {
            check_child_process_completion(app);
            return;
        }
    }

    // ── Thread-based path: poll channel ──
    let completed = if let Some(ref task) = app.compute_task {
        match task.receiver.try_recv() {
            Ok(result) => Some(result),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Thread died without sending a result (shouldn't happen with catch_unwind,
                // but serves as defense-in-depth against FFI segfaults or aborts).
                Some(ComputeResult::Error {
                    message: "Compute thread terminated unexpectedly".into(),
                })
            }
        }
    } else {
        return;
    };

    if let Some(result) = completed {
        let task = app.compute_task.take().unwrap();
        let ctx = task.context;

        match result {
            ComputeResult::Multiply { result, padding_ms, unpadding_ms, compute_ms, peak_ram_bytes, avg_freq_ghz } => {
                let n = ctx.size;
                let gflops = MultiplicationResult::calculate_gflops(n, n, n, compute_ms.max(0.001));

                let total_flops = 2u64
                    .saturating_mul(n as u64)
                    .saturating_mul(n as u64)
                    .saturating_mul(n as u64);

                // Use real-time measured frequency for efficiency% if available
                let theoretical_peak = if avg_freq_ghz > 0.5 {
                    let pe = &app.sys_info.peak_estimate;
                    pe.cores as f64
                        * avg_freq_ghz
                        * pe.fma_ports as f64
                        * pe.fp64_per_cycle_per_fma
                } else {
                    app.sys_info.peak_estimate.peak_gflops
                };
                let efficiency_pct = if theoretical_peak > 0.0 {
                    (gflops / theoretical_peak * 100.0).min(999.0)
                } else {
                    0.0
                };

                let data = BenchmarkData {
                    algorithm: ctx.algorithm_name.clone(),
                    size: n,
                    gen_time_ms: ctx.gen_time_ms,
                    padding_time_ms: padding_ms,
                    compute_time_ms: compute_ms,
                    unpadding_time_ms: unpadding_ms,
                    gflops,
                    simd_level: ctx.simd_level.display_name().to_string(),
                    threads: rayon::current_num_threads(),
                    peak_ram_mb: peak_ram_bytes / (1024 * 1024),
                    result_matrix: Some(result),
                    theoretical_peak_gflops: theoretical_peak,
                    efficiency_pct,
                    total_flops,
                    computation_id: {
                        let name = app.pending_session_name.trim().to_string();
                        if name.is_empty() {
                            crate::io::generate_computation_id()
                        } else {
                            app.pending_session_name.clone()
                        }
                    },
                    machine_name: app.sys_info.hostname.clone(),
                    allocation_ms: ctx.gen_time_ms.unwrap_or(0.0),
                    rows_a: n,
                    cols_a: n,
                    cols_b: n,
                    matrix_memory_mb: (3.0 * n as f64 * n as f64 * 8.0) / (1024.0 * 1024.0),
                };

                // Add to session history
                app.session_history.push(
                    data.clone(),
                    RunConfig {
                        algorithm: ctx.algorithm_choice,
                        size: n,
                    },
                    &app.io_tx,
                );

                app.csv_saved = false;
                app.matrix_saved = false;
                app.screen = Screen::Results { data };
            }
            ComputeResult::Diff { result1, result2, time1_ms, time2_ms } => {
                let n = ctx.size;
                let alg1 = ctx.diff_alg1.unwrap_or(AlgorithmChoice::Naive);
                let alg2 = ctx.diff_alg2.unwrap_or(AlgorithmChoice::StrassenHybrid);

                let comparison = algorithms::compare_matrices(&result1, &result2, crate::common::EPSILON);
                let gflops1 = MultiplicationResult::calculate_gflops(n, n, n, time1_ms.max(0.001));
                let gflops2 = MultiplicationResult::calculate_gflops(n, n, n, time2_ms.max(0.001));

                let (winner, speedup) = if time1_ms <= time2_ms {
                    (alg1.display_name().to_string(), time2_ms / time1_ms.max(0.001))
                } else {
                    (alg2.display_name().to_string(), time1_ms / time2_ms.max(0.001))
                };

                let data = DiffResultData {
                    alg1_name: alg1.display_name().to_string(),
                    alg2_name: alg2.display_name().to_string(),
                    size: n,
                    time1_ms,
                    time2_ms,
                    gflops1,
                    gflops2,
                    speedup,
                    winner: winner.clone(),
                    max_diff: comparison.max_abs_diff,
                    is_match: comparison.is_equal,
                };

                // Push comparison result to session history so it appears in [Y] History
                let (winning_time, winning_gflops) = if time1_ms <= time2_ms {
                    (time1_ms, gflops1)
                } else {
                    (time2_ms, gflops2)
                };
                let total_flops = 2u64
                    .saturating_mul(n as u64)
                    .saturating_mul(n as u64)
                    .saturating_mul(n as u64);
                let theoretical_peak = app.sys_info.peak_estimate.peak_gflops;
                let history_data = BenchmarkData {
                    algorithm: format!(
                        "{} vs {} [{}]",
                        alg1.display_name(),
                        alg2.display_name(),
                        winner
                    ),
                    size: n,
                    gen_time_ms: None,
                    padding_time_ms: 0.0,
                    compute_time_ms: winning_time,
                    unpadding_time_ms: 0.0,
                    gflops: winning_gflops,
                    simd_level: ctx.simd_level.display_name().to_string(),
                    threads: rayon::current_num_threads(),
                    peak_ram_mb: 0,
                    result_matrix: None,
                    theoretical_peak_gflops: theoretical_peak,
                    efficiency_pct: if theoretical_peak > 0.0 {
                        (winning_gflops / theoretical_peak * 100.0).min(999.0)
                    } else {
                        0.0
                    },
                    total_flops,
                    computation_id: crate::io::generate_computation_id(),
                    machine_name: app.sys_info.hostname.clone(),
                    allocation_ms: 0.0,
                    rows_a: n,
                    cols_a: n,
                    cols_b: n,
                    matrix_memory_mb: 0.0,
                };
                app.session_history.push(
                    history_data,
                    RunConfig { algorithm: if time1_ms <= time2_ms { alg1 } else { alg2 }, size: n },
                    &app.io_tx,
                );

                app.screen = Screen::DiffResults { data };
            }
            ComputeResult::Error { message } => {
                app.thermal_phase = None;
                app.screen = Screen::ComputeError(message);
            }
            ComputeResult::Thermal { result } => {
                // Push to session history so thermal runs appear in [Y] History
                let cfg = &result.config;
                let grid_label = format!(
                    "Thermal FDM {}\u{00b3} {}",
                    cfg.nx, cfg.fluid.name
                );
                let est_ram = (cfg.total_nodes() * 7 * 12 / 1024 / 1024) as u64;
                let history_data = BenchmarkData {
                    algorithm: grid_label,
                    size: cfg.nx,
                    gen_time_ms: None,
                    padding_time_ms: 0.0,
                    compute_time_ms: result.computation_ms,
                    unpadding_time_ms: 0.0,
                    gflops: 0.0,
                    simd_level: "SpMV".to_string(),
                    threads: rayon::current_num_threads(),
                    peak_ram_mb: est_ram,
                    result_matrix: None,
                    theoretical_peak_gflops: 0.0,
                    efficiency_pct: 0.0,
                    total_flops: 0,
                    computation_id: crate::io::generate_computation_id(),
                    machine_name: app.sys_info.hostname.clone(),
                    allocation_ms: 0.0,
                    rows_a: cfg.nx,
                    cols_a: cfg.nx,
                    cols_b: cfg.nx,
                    matrix_memory_mb: 0.0,
                };
                app.session_history.push(
                    history_data,
                    RunConfig { algorithm: AlgorithmChoice::Naive, size: cfg.nx },
                    &app.io_tx,
                );

                app.thermal_csv_saved = false;
                app.thermal_field_saved = false;
                app.thermal_phase = None;
                app.screen = Screen::ThermalResults { result: Box::new(result) };
            }
            ComputeResult::Economics { result } => {
                app.econ_csv_saved = false;
                app.econ_phase = None;
                app.screen = Screen::EconResults { result: Box::new(result) };
            }
            ComputeResult::Mdp { result } => {
                app.mdp_csv_saved = false;
                app.mdp_phase = None;
                app.screen = Screen::MdpResults { result: Box::new(result) };
            }
            ComputeResult::Vision { result } => {
                app.vision_csv_saved = false;
                app.vision_phase = None;
                app.screen = Screen::VisionResults { result: Box::new(result) };
            }
            ComputeResult::Mhps { result } => {
                app.mhps_csv_saved = false;
                app.mhps_field_saved = false;
                app.mhps_phase = None;
                app.mhps_view_mode = 0;
                app.mhps_snapshot_idx = 0;
                app.screen = Screen::MhpsResults { result: Box::new(result) };
            }
        }
        // Sound notification on completion
        crate::io::play_completion_sound();
    } else if let Some(ref mut task) = app.compute_task {
        // Update ETA tracker with current progress
        let frac = task.progress.fraction();
        task.eta.update(frac);
    }
}

/// Poll a child compute process (process-isolated MKL computation).
fn check_child_process_completion(app: &mut App) {
    let task = app.compute_task.as_mut().unwrap();
    let child = task.child_process.as_mut().unwrap();

    // Update progress from the child's progress file
    if let Some(ref req) = task.compute_request {
        if let Some((done, total)) = crate::compute_worker::read_child_progress(
            std::path::Path::new(&req.progress_path),
        ) {
            task.progress.set(done, total);
        }
    }
    task.eta.update(task.progress.fraction());

    // Check if child has exited
    match child.try_wait() {
        Ok(Some(status)) => {
            // Child finished — read result
            let task = app.compute_task.take().unwrap();
            let ctx = task.context;
            let temp_dir = task.temp_dir.as_ref().cloned();
            let req = task.compute_request.as_ref().unwrap();

            if status.success() {
                // Read metrics
                let metrics_json = std::fs::read_to_string(&req.metrics_path).unwrap_or_default();
                let metrics: crate::compute_worker::ComputeMetrics =
                    serde_json::from_str(&metrics_json).unwrap_or_else(|_| {
                        crate::compute_worker::ComputeMetrics {
                            compute_ms: 0.0,
                            padding_ms: 0.0,
                            unpadding_ms: 0.0,
                            peak_ram_bytes: 0,
                            avg_freq_ghz: 0.0,
                            error: Some("Failed to parse metrics".into()),
                        }
                    });

                if let Some(ref err_msg) = metrics.error {
                    app.screen = Screen::ComputeError(err_msg.clone());
                } else {
                    // Read result matrix
                    match crate::compute_worker::load_matrix_binary(
                        std::path::Path::new(&req.result_path),
                    ) {
                        Ok(result) => {
                            let n = ctx.size;
                            let compute_ms = metrics.compute_ms;
                            let gflops = MultiplicationResult::calculate_gflops(
                                n, n, n, compute_ms.max(0.001),
                            );
                            let total_flops = 2u64
                                .saturating_mul(n as u64)
                                .saturating_mul(n as u64)
                                .saturating_mul(n as u64);
                            let gen_ms = ctx.gen_time_ms.unwrap_or(0.0);

                            let theoretical_peak = app.sys_info.peak_estimate.peak_gflops;
                            let efficiency_pct = if theoretical_peak > 0.0 {
                                (gflops / theoretical_peak * 100.0).min(999.0)
                            } else {
                                0.0
                            };

                            let data = BenchmarkData {
                                algorithm: ctx.algorithm_name.clone(),
                                size: n,
                                gen_time_ms: ctx.gen_time_ms,
                                padding_time_ms: metrics.padding_ms + gen_ms,
                                compute_time_ms: compute_ms,
                                unpadding_time_ms: metrics.unpadding_ms,
                                gflops,
                                simd_level: ctx.simd_level.display_name().to_string(),
                                threads: rayon::current_num_threads(),
                                peak_ram_mb: metrics.peak_ram_bytes / (1024 * 1024),
                                result_matrix: Some(result),
                                theoretical_peak_gflops: theoretical_peak,
                                efficiency_pct,
                                total_flops,
                                computation_id: crate::io::generate_computation_id(),
                                machine_name: app.sys_info.hostname.clone(),
                                allocation_ms: ctx.gen_time_ms.unwrap_or(0.0),
                                rows_a: n,
                                cols_a: n,
                                cols_b: n,
                                matrix_memory_mb: (3.0 * n as f64 * n as f64 * 8.0) / (1024.0 * 1024.0),
                            };

                            app.session_history.push(
                                data.clone(),
                                RunConfig {
                                    algorithm: ctx.algorithm_choice,
                                    size: n,
                                },
                                &app.io_tx,
                            );
                            app.csv_saved = false;
                            app.matrix_saved = false;
                            app.screen = Screen::Results { data };
                            crate::io::play_completion_sound();
                        }
                        Err(e) => {
                            app.screen = Screen::ComputeError(
                                format!("Failed to read result matrix: {e}"),
                            );
                        }
                    }
                }
            } else {
                // Child exited with error — collect diagnostics
                let metrics_msg = std::fs::read_to_string(&req.metrics_path)
                    .ok()
                    .and_then(|j| serde_json::from_str::<crate::compute_worker::ComputeMetrics>(&j).ok())
                    .and_then(|m| m.error);

                let stderr_log = temp_dir.as_ref()
                    .and_then(|d| std::fs::read_to_string(d.join("stderr.log")).ok())
                    .unwrap_or_default();
                // Truncate stderr to avoid flooding the TUI
                let stderr_trimmed: String = stderr_log.lines().take(20).collect::<Vec<_>>().join("\n");

                let msg = if let Some(m) = metrics_msg {
                    if stderr_trimmed.is_empty() { m }
                    else { format!("{m}\n\nstderr:\n{stderr_trimmed}") }
                } else {
                    let base = format!("Compute worker crashed (exit code: {:?})", status.code());
                    if stderr_trimmed.is_empty() {
                        format!("{base}\n\nMKL runtime DLLs may be missing.\n\
                                 Run Intel oneAPI setvars.bat before launching Flust,\n\
                                 or add the MKL bin directory to your system PATH.")
                    } else {
                        format!("{base}\n\nstderr:\n{stderr_trimmed}")
                    }
                };
                app.screen = Screen::ComputeError(msg);
            }

            // Clean up temp files
            if let Some(dir) = temp_dir {
                crate::compute_worker::cleanup_temp_dir(&dir);
            }
        }
        Ok(None) => {
            // Still running — nothing to do
        }
        Err(e) => {
            // Failed to query child status
            let task = app.compute_task.take().unwrap();
            if let Some(dir) = task.temp_dir {
                crate::compute_worker::cleanup_temp_dir(&dir);
            }
            app.screen = Screen::ComputeError(format!("Failed to check worker status: {e}"));
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────


/// Center a rectangle within `r` using percentages.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Format milliseconds into a human-readable string.
pub(crate) fn format_duration(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.2} µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.2} ms", ms)
    } else {
        format!("{:.3} s", ms / 1000.0)
    }
}

/// Format megabytes into human-readable MB/GB.
pub(crate) fn format_memory(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{mb} MB")
    }
}

// ─── Rendering ──────────────────────────────────────────────────────────────

fn render(app: &App, frame: &mut ratatui::Frame) {
    let area = frame.size();
    let t = app.theme();
    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default().style(Style::default().bg(t.bg)),
        area,
    );

    match &app.screen {
        Screen::MainMenu => render_main_menu(app, frame, area, &t),
        Screen::CategoryMenu { category } => render_category_menu(app, *category, frame, area, &t),
        Screen::MultiplyMenu => render_multiply_menu(app, frame, area, &t),
        Screen::SizeInput => render_size_input(app, frame, area, &t),
        Screen::InputMethodMenu => render_input_method(app, frame, area, &t),
        Screen::NameInput => render_name_input(app, frame, area, &t),
        Screen::ManualInputA { name } => render_manual_input(app, frame, area, name, &t),
        Screen::ManualInputB { name, .. } => render_manual_input(app, frame, area, name, &t),
        Screen::Computing { algorithm } => render_computing(app, algorithm, frame, area, &t),
        Screen::Results { data } => render_results(app, data, frame, area, &t),
        Screen::History => render_history_screen(app, frame, area, &t),
        Screen::ViewerFileInput => render_viewer_file_input(app, frame, area, &t),
        Screen::MatrixViewer => render_matrix_viewer(app, frame, area, &t),
        Screen::DiffSizeInput => render_diff_size_input(app, frame, area, &t),
        Screen::DiffAlgSelect => render_diff_alg_select(app, frame, area, &t),
        Screen::DiffResults { data } => render_diff_results(data, frame, area, &t),
        Screen::FileCompareInputA => render_file_compare_input(app, frame, area, &t, "A"),
        Screen::FileCompareInputB => render_file_compare_input(app, frame, area, &t, "B"),
        Screen::FileCompareResults { data } => render_file_compare_results(data, frame, area, &t),
        Screen::ComingSoon(label) => render_coming_soon(label, frame, area, &t),
        Screen::FileBrowser { .. } => render_file_browser(app, frame, area, &t),
        // Thermal simulation screens
        Screen::ThermalFluidSelect => crate::thermal_ui::render_thermal_fluid_select(app, frame, area, &t),
        Screen::ThermalGeometry => crate::thermal_ui::render_thermal_geometry(app, frame, area, &t),
        Screen::ThermalTeg => crate::thermal_ui::render_thermal_teg(app, frame, area, &t),
        Screen::ThermalHeatSources => crate::thermal_ui::render_thermal_heat_sources(app, frame, area, &t),
        Screen::ThermalSolverSelect => crate::thermal_ui::render_thermal_solver_select(app, frame, area, &t),
        Screen::ThermalConfirm => crate::thermal_ui::render_thermal_confirm(app, frame, area, &t),
        Screen::ThermalComputing => crate::thermal_ui::render_thermal_computing(app, frame, area, &t),
        Screen::ThermalResults { .. } => crate::thermal_ui::render_thermal_results(app, frame, area, &t),
        Screen::ThermalViewer => crate::thermal_ui::render_thermal_viewer(app, frame, area, &t),
        // Economics simulation screens
        Screen::EconConfig => crate::economics_ui::render_econ_config(app, frame, area, &t),
        Screen::EconShockSelect => crate::economics_ui::render_econ_shock(app, frame, area, &t),
        Screen::EconConfirm => crate::economics_ui::render_econ_confirm(app, frame, area, &t),
        Screen::EconComputing => crate::economics_ui::render_econ_computing(app, frame, area, &t),
        Screen::EconResults { result } => crate::economics_ui::render_econ_results(app, result, frame, area, &t),
        // MDP Kinematics screens
        Screen::MdpConfig => crate::kinematics_ui::render_mdp_config(app, frame, area, &t),
        Screen::MdpConfirm => crate::kinematics_ui::render_mdp_confirm(app, frame, area, &t),
        Screen::MdpComputing => crate::kinematics_ui::render_mdp_computing(app, frame, area, &t),
        Screen::MdpResults { result } => crate::kinematics_ui::render_mdp_results(app, result, frame, area, &t),
        // Computer Vision screens
        Screen::VisionConfig => crate::vision_ui::render_vision_config(app, frame, area, &t),
        Screen::VisionKernelSelect => crate::vision_ui::render_vision_kernel(app, frame, area, &t),
        Screen::VisionConfirm => crate::vision_ui::render_vision_confirm(app, frame, area, &t),
        Screen::VisionComputing => crate::vision_ui::render_vision_computing(app, frame, area, &t),
        Screen::VisionResults { result } => crate::vision_ui::render_vision_results(app, result, frame, area, &t),
        // MHPS screens
        Screen::MhpsMaterial => crate::mhps_ui::render_mhps_material(app, frame, area, &t),
        Screen::MhpsGeometry => crate::mhps_ui::render_mhps_geometry(app, frame, area, &t),
        Screen::MhpsHeatSources => crate::mhps_ui::render_mhps_heat_sources(app, frame, area, &t),
        Screen::MhpsSimParams => crate::mhps_ui::render_mhps_sim_params(app, frame, area, &t),
        Screen::MhpsConfirm => crate::mhps_ui::render_mhps_confirm(app, frame, area, &t),
        Screen::MhpsComputing => crate::mhps_ui::render_mhps_computing(app, frame, area, &t),
        Screen::MhpsResults { result } => crate::mhps_ui::render_mhps_results(app, result, frame, area, &t),
        Screen::SystemProfile { profile } => crate::system_profiler::render_system_profile(frame, area, profile, &t),
        Screen::ComputeError(msg) => render_compute_error(msg, frame, area, &t),
    }

    // Render overlay popups on top
    match app.overlay {
        Overlay::Help => render_help_popup(frame, area, &t),
        Overlay::About => render_about_popup(frame, area, &t),
        Overlay::ThermalHelp => {
            crate::thermal_ui::render_thermal_help_overlay(frame, area, &t, app.thermal_overlay_scroll);
        }
        Overlay::ThermalGraph => {
            if let Screen::ThermalResults { ref result } = app.screen {
                crate::thermal_ui::render_thermal_graph_overlay(result, frame, area, &t, app.thermal_overlay_scroll);
            }
        }
        Overlay::ThermalCrossSection2D => {
            if let Screen::ThermalResults { ref result } = app.screen {
                crate::thermal_ui::render_thermal_crosssection_viewport(
                    result, frame, area, &t,
                    app.thermal_cross_slice_y,
                    app.thermal_vp_cam_x,
                    app.thermal_vp_cam_y,
                    app.thermal_vp_zoom,
                    app.thermal_vp_cursor_x,
                    app.thermal_vp_cursor_y,
                );
            }
        }
        Overlay::ThermalIsometric => {
            if let Screen::ThermalResults { ref result } = app.screen {
                crate::thermal_ui::render_thermal_isometric_overlay(result, frame, area, &t, app.thermal_cross_slice_y);
            }
        }
        Overlay::EngineeringInfo => render_info_popup(
            "Engineering",
            "Flust Engineering Suite provides physics-based simulations\n\
             powered by the core HPC matrix engine.\n\n\
             \u{2022} Thermal Simulation \u{2014} 3D FDM heat transfer with TEG\n\
               power generation modeling. Sparse matrix solver.\n\n\
             \u{2022} Kinematics & Pathfinding \u{2014} Markov Decision Process\n\
               steady-state solver via repeated matrix squaring.\n\
               Models optimal pathfinding through stochastic environments.\n\n\
             \u{2022} Computer Vision (IR Processing) \u{2014} Bicubic upscale,\n\
               im2col transform, convolution via GEMM. Processes\n\
               raw IR sensor data for denoising and enhancement.\n\n\
             All solvers leverage SIMD-accelerated kernels and\n\
             rayon work-stealing parallelism for maximum throughput.",
            frame, area, &t,
        ),
        Overlay::EconHelp => render_info_popup(
            "Economics \u{2014} Theory",
            "Leontief Input\u{2013}Output Model (1936, Nobel Prize 1973)\n\n\
             The economy is modeled as N interdependent sectors.\n\
             Technology matrix A: element a\u{1d62}\u{2c7c} = fraction of sector j\u{2019}s\n\
             output consumed by sector i as intermediate input.\n\n\
             Total output x satisfies:  x = Ax + d\n\
             Solution:  x = (I \u{2212} A)\u{207b}\u{00b9} d\n\n\
             Neumann series approximation (dynamic shock waves):\n\
               x\u{2080} = d,  x\u{2096} = A\u{00b7}x\u{2096}\u{208b}\u{2081} + d\n\n\
             Converges when spectral radius \u{03c1}(A) < 1.\n\
             Each iteration k represents the k-th cascade wave\n\
             of economic shock propagating through supply chains.",
            frame, area, &t,
        ),
        Overlay::EconInfo => render_info_popup(
            "Economics \u{2014} Parameters",
            "Leontief Shock Simulator configuration:\n\n\
             \u{2022} Sectors (N) \u{2014} number of economic sectors (matrix size)\n\
             \u{2022} Sparsity \u{2014} fraction of zero entries in A\n\
             \u{2022} Spectral radius target \u{2014} ensures \u{03c1}(A) < 1\n\
             \u{2022} Convergence tolerance \u{2014} stop when \u{0394} < tol\n\
             \u{2022} Max iterations \u{2014} safety bound for divergent cases\n\
             \u{2022} Shock sector \u{2014} which sector receives demand shock\n\
             \u{2022} Shock magnitude \u{2014} multiplier for shocked sector\n\n\
             The simulator generates a random technology matrix\n\
             with controlled spectral radius and runs the Neumann\n\
             iteration using rayon-parallel matrix\u{2013}vector multiply.",
            frame, area, &t,
        ),
        Overlay::EconConvergenceGraph | Overlay::EconSectorBars | Overlay::EconDashboard => {
            if let Screen::EconResults { ref result } = app.screen {
                crate::economics_ui::render_econ_overlay(app, result, frame, area, &t);
            }
        }
        Overlay::MdpConvergenceGraph | Overlay::MdpStateBars | Overlay::MdpDashboard => {
            if let Screen::MdpResults { ref result } = app.screen {
                crate::kinematics_ui::render_mdp_overlay(app, result, frame, area, &t);
            }
        }
        Overlay::VisionHeatmap | Overlay::VisionPipeline => {
            if let Screen::VisionResults { ref result } = app.screen {
                crate::vision_ui::render_vision_overlay(app, result, frame, area, &t);
            }
        }
        Overlay::None => {}
    }

    // I/O progress overlay (saving/loading indicator)
    if let Some(ref status) = app.io_status {
        let spinner_chars = ["\u{2847}", "\u{28b7}", "\u{28bd}", "\u{28fc}", "\u{28f9}", "\u{28e3}"];
        let spinner = spinner_chars[(app.gear_frame / 2) % spinner_chars.len()];
        let text = format!(" {} {} ", spinner, status);
        let w = text.len() as u16 + 2;
        let h = 3u16;
        let x = area.width.saturating_sub(w) / 2;
        let y = area.height.saturating_sub(h) / 2;
        let popup_area = Rect::new(x, y, w, h);
        frame.render_widget(Clear, popup_area);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.accent))
            .style(Style::default().bg(t.surface));
        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(t.text_bright).bg(t.surface))
            .block(block);
        frame.render_widget(paragraph, popup_area);
    }
}

// ─── Main Menu ──────────────────────────────────────────────────────────────

fn render_main_menu(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40), // banner + logo
            Constraint::Min(8),         // menu items
            Constraint::Length(5),      // hint
            Constraint::Length(2),      // footer (keys + status)
        ])
        .split(area);

    render_banner(&app.sys_info, app.terminal_scale, frame, chunks[0], t);
    render_menu(app, frame, chunks[1], t);
    render_hint(app, frame, chunks[2], t);
    render_main_footer(app, frame, chunks[3], t);
}

fn render_banner(sys: &SystemInfo, scale: TerminalScale, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let show_logo = scale.show_logo();
    let logo_lines = if show_logo { LOGO_LINES.len() } else { 0 };
    let total_content = logo_lines + 4;
    let top_pad = (area.height as usize).saturating_sub(total_content) / 2;

    let mut lines: Vec<Line> = Vec::new();
    for _ in 0..top_pad {
        lines.push(Line::from(""));
    }
    if show_logo {
        for logo_text in &LOGO_LINES {
            lines.push(Line::from(Span::styled(
                *logo_text,
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "High-Performance Matrix Engine  ·  Rust Edition",
        Style::default().fg(t.text_muted),
    )));
    lines.push(Line::from(Span::styled(
        format!("v{}  ·  {}", env!("CARGO_PKG_VERSION"), sys.simd_level.display_name()),
        Style::default().fg(t.text_dim),
    )));

    let para = Paragraph::new(lines).alignment(Alignment::Center);
    frame.render_widget(para, area);
}

fn render_menu(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let items: Vec<ListItem> = MENU_ITEMS
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let is_selected = i == app.main_menu_idx;
            let is_inactive = i == 8 && app.session_history.is_empty();

            let content = if is_inactive {
                Line::from(vec![
                    Span::styled("     ", Style::default()),
                    Span::styled(item.label, Style::default().fg(t.text_dim)),
                    Span::styled(
                        if is_selected { "  (no computations yet)" } else { "" },
                        Style::default().fg(t.text_dim),
                    ),
                ])
            } else if is_selected {
                Line::from(vec![
                    Span::styled("  \u{25b6}  ", Style::default().fg(t.accent)),
                    Span::styled(
                        item.label,
                        Style::default().fg(t.text_bright).add_modifier(Modifier::BOLD),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled("     ", Style::default()),
                    Span::styled(item.label, Style::default().fg(t.text_muted)),
                ])
            };
            ListItem::new(content).style(if is_selected && !is_inactive {
                Style::default().bg(t.surface)
            } else {
                Style::default().bg(t.bg)
            })
        })
        .collect();

    let list = List::new(items).block(Block::default().borders(Borders::NONE));
    let menu_height = MENU_ITEMS.len() as u16;
    let top_pad = area.height.saturating_sub(menu_height) / 2;
    let menu_area = Rect {
        x: area.x,
        y: area.y + top_pad,
        width: area.width,
        height: menu_height.min(area.height.saturating_sub(top_pad)),
    };
    frame.render_widget(list, menu_area);
}

fn render_hint(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let idx = app.main_menu_idx.min(MENU_ITEMS.len() - 1);
    let item = &MENU_ITEMS[idx];

    let lines: Vec<Line> = item
        .description
        .lines()
        .map(|l| Line::from(Span::styled(format!("    {l}"), Style::default().fg(t.text_muted))))
        .collect();

    let para = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(t.border))
            .title(Span::styled(" Info ", Style::default().fg(t.text_dim))),
    );
    frame.render_widget(para, area);
}

fn render_main_footer(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let key_style = Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(t.text_muted).bg(t.surface);
    let dim_sep = Span::styled(" \u{2502} ", Style::default().fg(t.text_dim).bg(t.surface));
    let status_style = Style::default().fg(t.text_dim).bg(t.surface);

    // All keys on a single centered line
    let keys: Vec<Span> = vec![
        Span::styled(" \u{2191}\u{2193} ", key_style),
        Span::styled(" Navigate ", desc_style),
        dim_sep.clone(),
        Span::styled(" Enter ", key_style),
        Span::styled(" Select ", desc_style),
        dim_sep.clone(),
        Span::styled(" T ", key_style),
        Span::styled(format!(" {} ", app.current_theme.display_name()), desc_style),
        dim_sep.clone(),
        Span::styled(" S ", key_style),
        Span::styled(format!(" {} ", app.terminal_scale.label()), desc_style),
        dim_sep.clone(),
        Span::styled(" H ", key_style),
        Span::styled(" Help ", desc_style),
        dim_sep.clone(),
        Span::styled(" A ", key_style),
        Span::styled(" About ", desc_style),
        dim_sep.clone(),
        Span::styled(" Esc ", key_style),
        Span::styled(" Back ", desc_style),
        dim_sep.clone(),
        Span::styled(" Q ", key_style),
        Span::styled(" Quit ", desc_style),
    ];

    // Calculate total visible width to center
    let total_w: u16 = keys.iter().map(|s| s.width() as u16).sum();
    let pad_left = if area.width > total_w { (area.width - total_w) / 2 } else { 0 };

    let mut spans = vec![Span::styled(" ".repeat(pad_left as usize), Style::default().bg(t.surface))];
    spans.extend(keys);
    let row1 = Line::from(spans);

    // Status line (centered)
    let status_text = format!(
        "{} \u{2502} {} threads \u{2502} Theme: {}",
        app.sys_info.simd_level.display_name(),
        rayon::current_num_threads(),
        app.current_theme.display_name(),
    );
    let status_w = status_text.len() as u16;
    let status_pad = if area.width > status_w { (area.width - status_w) / 2 } else { 0 };
    let row2 = Line::from(vec![
        Span::styled(" ".repeat(status_pad as usize), Style::default().bg(t.surface)),
        Span::styled(status_text, status_style),
    ]);

    let footer = Paragraph::new(vec![row1, row2])
        .style(Style::default().bg(t.surface));
    frame.render_widget(footer, area);
}

// ─── Help Popup ─────────────────────────────────────────────────────────────

fn render_help_popup(frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let popup_area = centered_rect(80, 80, area);
    frame.render_widget(Clear, popup_area);

    let heading = Style::default().fg(t.text_bright);
    let body = Style::default().fg(t.text_muted);

    let content = vec![
        Line::from(""),
        Line::from(Span::styled("  FLUST — Quick Reference", Style::default().fg(t.accent).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(Span::styled("  ALGORITHMS", heading)),
        Line::from(Span::styled("  Strassen   O(n^2.807) — fastest for large matrices (>256\u{00d7}256)", body)),
        Line::from(Span::styled("  Tiled      Cache-optimized. Best when Strassen overhead is too large", body)),
        Line::from(Span::styled("  Scalar     Baseline. Used for correctness verification", body)),
        Line::from(""),
        Line::from(Span::styled("  SIMD LEVELS", heading)),
        Line::from(Span::styled("  AVX-512    8\u{00d7} f64 per instruction — Intel Xeon, newer Core", body)),
        Line::from(Span::styled("  AVX2       4\u{00d7} f64 per instruction — most modern CPUs", body)),
        Line::from(Span::styled("  SSE4.2     2\u{00d7} f64 per instruction — fallback", body)),
        Line::from(""),
        Line::from(Span::styled("  FILE FORMATS", heading)),
        Line::from(Span::styled("  Matrix files: first line 'rows,cols', then data rows as CSV", body)),
        Line::from(Span::styled("  Results log: CSV with algorithm, size, time_ms, GFLOPS", body)),
        Line::from(""),
        Line::from(Span::styled("  PERFORMANCE TIPS", heading)),
        Line::from(Span::styled("  Tile size is auto-tuned at startup for your L1/L2 cache", body)),
        Line::from(Span::styled("  Strassen threshold: 64 = balanced, 128 = less recursion overhead", body)),
        Line::from(""),
        Line::from(Span::styled("  [Esc] Close", Style::default().fg(t.accent))),
    ];

    let popup = Paragraph::new(content).block(
        Block::default()
            .title(Span::styled(" Help ", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.accent))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(popup, popup_area);
}

// ─── About Popup ────────────────────────────────────────────────────────────

fn render_about_popup(frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let popup_area = centered_rect(70, 70, area);
    frame.render_widget(Clear, popup_area);
    let body = Style::default().fg(t.text_muted);
    let bold_accent = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);

    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  FLUST  ", bold_accent),
            Span::styled("— High-Performance Matrix Engine", Style::default().fg(t.text)),
        ]),
        Line::from(""),
        Line::from(Span::styled("  The name tells the story:", body)),
        Line::from(""),
        Line::from(vec![
            Span::styled("  FL", bold_accent),
            Span::styled("uminum  +  ", body),
            Span::styled("RUST", Style::default().fg(t.crit).add_modifier(Modifier::BOLD)),
            Span::styled("  =  ", body),
            Span::styled("FL", bold_accent),
            Span::styled("UST", Style::default().fg(t.crit).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Fluminum was a C++ project — a high-performance matrix multiplication", body)),
        Line::from(Span::styled("  engine built for a student conference at Georgian Technical University.", body)),
        Line::from(""),
        Line::from(Span::styled("  Flust is its rightful successor: same core ideas, same algorithms,", body)),
        Line::from(Span::styled("  but rebuilt from the ground up in Rust — safer, faster, and portable.", body)),
        Line::from(""),
        Line::from(Span::styled("  What we inherited from Fluminum:", Style::default().fg(t.text))),
        Line::from(Span::styled("  Strassen's algorithm · AVX2 SIMD · Cache-blocked Tiling · Auto-tuner", body)),
        Line::from(""),
        Line::from(Span::styled("  What Rust gave us:", Style::default().fg(t.text))),
        Line::from(Span::styled("  Memory safety · rayon Work-Stealing · No undefined behavior", body)),
        Line::from(""),
        Line::from(Span::styled(
            format!("  v{}  ·  Built with Rust + ratatui + rayon", env!("CARGO_PKG_VERSION")),
            Style::default().fg(t.text_dim),
        )),
        Line::from(""),
        Line::from(Span::styled("  [Esc] Close", Style::default().fg(t.accent))),
    ];

    let popup = Paragraph::new(content).block(
        Block::default()
            .title(Span::styled(" About Flust ", bold_accent))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.accent))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(popup, popup_area);
}

// ─── Generic Info Popup ─────────────────────────────────────────────────────

fn render_info_popup(
    title: &str,
    body: &str,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup_area = centered_rect(70, 70, area);
    frame.render_widget(Clear, popup_area);
    let mut lines: Vec<Line> = vec![Line::from("")];
    for line in body.lines() {
        lines.push(Line::from(Span::styled(
            format!("  {line}"),
            Style::default().fg(t.text_muted),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [Esc] Close",
        Style::default().fg(t.accent),
    )));
    let popup = Paragraph::new(lines).block(
        Block::default()
            .title(Span::styled(
                format!(" {title} "),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.accent))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(popup, popup_area);
}

// ─── Category Submenu ───────────────────────────────────────────────────────

fn render_category_menu(
    app: &App,
    category: MenuCategory,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // header
            Constraint::Min(8),    // items
            Constraint::Length(3), // description
            Constraint::Length(1), // footer
        ])
        .split(area);

    // Header
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {} ▸", category.label()),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("  {}", category.description().lines().next().unwrap_or("")),
                Style::default().fg(t.text_dim),
            )),
        ]),
        chunks[0],
    );

    // Subitem list
    let subitems = category.subitems();
    let items: Vec<ListItem> = subitems
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let selected = i == app.category_menu_idx;
            let marker = if selected { "▸ " } else { "  " };
            let shortcut = item
                .shortcut
                .map(|c| format!("[{c}] "))
                .unwrap_or_default();
            let style = if selected {
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(t.text)
            };
            ListItem::new(Line::from(Span::styled(
                format!("  {marker}{shortcut}{}", item.label),
                style,
            )))
        })
        .collect();

    frame.render_widget(
        List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // Selected item description
    if let Some(item) = subitems.get(app.category_menu_idx) {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("  {}", item.description.lines().next().unwrap_or("")),
                Style::default().fg(t.text_dim),
            ))),
            chunks[2],
        );
    }

    // Footer
    let footer_spans = vec![
        Span::styled("  [↑↓] Navigate", Style::default().fg(t.text_muted)),
        Span::styled("  [Enter] Select", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc] Back", Style::default().fg(t.text_muted)),
    ];
    frame.render_widget(Paragraph::new(Line::from(footer_spans)), chunks[3]);
}

// ─── Multiply Menu ──────────────────────────────────────────────────────────

fn render_multiply_menu(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let title = Paragraph::new(Line::from(Span::styled("  MATRIX MULTIPLICATION", title_style)));
    frame.render_widget(title, chunks[0]);

    let mut items = vec![
        ("1", "Naive (no optimization)", "Simple i-k-j loop, single thread. Ground truth baseline."),
        ("2", "Strassen (classic)", "Strassen + Tiling + Rayon. Recursive parallelism."),
        ("3", "Winograd (classic)", "Strassen variant with fewer additions (15 vs 18). Same O(n^2.807)."),
        ("4", "Strassen Hybrid (Full-Core)", "Strip-parallel Strassen. All cores busy from microsecond one."),
        ("5", "Winograd Hybrid (Full-Core)", "Strip-parallel Winograd. Full core utilization + fewer ops."),
    ];
    if crate::algorithms::is_mkl_available() {
        items.push(("6", "Intel MKL (DGEMM)", "Hardware-optimized BLAS from Intel oneAPI. cblas_dgemm."));
    } else {
        items.push(("6", "Intel MKL (DGEMM) [Unavailable]", "Install Intel oneAPI MKL and ensure mkl_rt.2.dll is on PATH."));
    }
    // No "Back" item — Escape handles return to main menu

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let default_style = Style::default().fg(t.text).bg(t.bg);
    let selected_style = Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD);
    let muted_style = Style::default().fg(t.text_muted).bg(t.bg);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, (key, label, desc)) in items.iter().enumerate() {
        let selected = i == app.mult_menu_idx;
        let arrow = if selected { " \u{25b8} " } else { "   " };
        let style = if selected { selected_style } else { default_style };
        let dim = if selected { selected_style } else { muted_style };

        lines.push(Line::from(vec![
            Span::styled(arrow, style),
            Span::styled(format!("[{key}] "), key_hint),
            Span::styled(format!("{label:<30}"), style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      ", default_style),
            Span::styled(*desc, dim),
        ]));
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(Span::styled(" SELECT ALGORITHM ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    render_nav_footer(frame, chunks[2], t);
}

// ─── Size Input ─────────────────────────────────────────────────────────────

fn render_size_input(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let title = Paragraph::new(Line::from(Span::styled(
        "  MATRIX SIZE",
        title_style,
    )));
    frame.render_widget(title, chunks[0]);

    let est_ram = if let Ok(n) = app.size_input.parse::<usize>() {
        if n > 0 {
            let mb = SystemInfo::estimate_peak_ram_mb(n);
            format!("Estimated RAM: ~{}", format_memory(mb))
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Both matrices will be square (N \u{00d7} N).",
            Style::default().fg(t.text_muted),
        )),
        Line::from(Span::styled(
            "  Enter the dimension N (1-10000):",
            Style::default().fg(t.text),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  N = ", Style::default().fg(t.accent)),
            Span::styled(
                format!("{}_", &app.size_input),
                Style::default().fg(t.ok),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {est_ram}"),
            Style::default().fg(t.text_muted),
        )),
    ];

    let block = Block::default()
        .title(Span::styled(" DIMENSION ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter]", key_hint),
        Span::styled(" Confirm  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc]", key_hint),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[2],
    );
}

// ─── Name Input ─────────────────────────────────────────────────────────────

fn render_name_input(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    // Background fill
    frame.render_widget(
        Block::default().style(Style::default().bg(t.bg)),
        area,
    );

    // Center dialog: 62 wide × 11 tall
    let dw = 62u16.min(area.width.saturating_sub(4));
    let dh = 11u16;
    let dx = area.x + area.width.saturating_sub(dw) / 2;
    let dy = area.y + area.height.saturating_sub(dh) / 2;
    let dialog = Rect { x: dx, y: dy, width: dw, height: dh };

    frame.render_widget(Clear, dialog);

    let block = Block::default()
        .title(Span::styled(
            " NAME THIS COMPUTATION ",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.accent))
        .style(Style::default().bg(t.bg));

    let inner = Rect {
        x: dialog.x + 1,
        y: dialog.y + 1,
        width: dialog.width.saturating_sub(2),
        height: dialog.height.saturating_sub(2),
    };
    frame.render_widget(block, dialog);

    // Cursor blink: always show │ at end of input
    let display_name = if app.pending_session_name.is_empty() {
        "│".to_string()
    } else {
        format!("{}│", app.pending_session_name)
    };

    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Name : ", Style::default().fg(t.text_dim)),
            Span::styled(
                display_name,
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Auto-ID format: ",
                Style::default().fg(t.text_dim),
            ),
            Span::styled(
                "FLUST-YYYYMMDD-HHMMSS-NNN",
                Style::default().fg(t.text_muted),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled("[Enter]", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled(" Confirm  ", Style::default().fg(t.text_muted)),
            Span::styled("[Esc]", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled(" Back", Style::default().fg(t.text_muted)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Leave empty to use auto-generated ID",
                Style::default().fg(t.text_dim),
            ),
        ]),
    ];

    frame.render_widget(Paragraph::new(content), inner);
}

// ─── Input Method Menu ──────────────────────────────────────────────────────

fn render_input_method(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let title = Paragraph::new(Line::from(Span::styled(
        format!(
            "  DATA INPUT FOR {}\u{00d7}{} MATRICES",
            app.chosen_size, app.chosen_size
        ),
        title_style,
    )));
    frame.render_widget(title, chunks[0]);

    let items = vec![
        (
            "1",
            "Generate Random",
            "Fast random fill [-10, 10]. Best for benchmarking.",
        ),
        (
            "2",
            "Manual Input",
            "Enter elements row by row. Best for small matrices.",
        ),
        ("3", "Back", "Change matrix size"),
    ];

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let default_style = Style::default().fg(t.text).bg(t.bg);
    let selected_style = Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD);
    let muted_style = Style::default().fg(t.text_muted).bg(t.bg);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, (key, label, desc)) in items.iter().enumerate() {
        let selected = i == app.input_method_idx;
        let arrow = if selected { " \u{25b8} " } else { "   " };
        let style = if selected { selected_style } else { default_style };
        let dim = if selected { selected_style } else { muted_style };

        lines.push(Line::from(vec![
            Span::styled(arrow, style),
            Span::styled(format!("[{key}] "), key_hint),
            Span::styled(format!("{label:<25}"), style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      ", default_style),
            Span::styled(*desc, dim),
        ]));
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(Span::styled(" INPUT METHOD ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    render_nav_footer(frame, chunks[2], t);
}

// ─── Manual Input (with scrolling viewport) ─────────────────────────────────

fn render_manual_input(app: &App, frame: &mut ratatui::Frame, area: Rect, name: &str, t: &ThemeColors) {
    let n = app.chosen_size;
    let filled = app.manual_data.len();
    let total = n * n;
    let current_row = if n > 0 { filled / n } else { 0 };
    let current_col = if n > 0 { filled % n } else { 0 };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);

    // Title with progress
    let pct = if total > 0 { (filled * 100) / total } else { 0 };
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            format!("  MATRIX \"{name}\" \u{2014} {n}\u{00d7}{n}  "),
            title_style,
        ),
        Span::styled(
            format!("({filled}/{total} elements, {pct}%)"),
            Style::default().fg(t.ok),
        ),
    ]));
    frame.render_widget(title, chunks[0]);

    // Progress bar
    let bar_width = (chunks[1].width as usize).saturating_sub(6);
    let bar_filled = if total > 0 {
        (filled * bar_width) / total
    } else {
        0
    };
    let bar_empty = bar_width.saturating_sub(bar_filled);
    let progress_lines = vec![
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(
                "\u{2588}".repeat(bar_filled),
                Style::default().fg(t.ok),
            ),
            Span::styled(
                "\u{2591}".repeat(bar_empty),
                Style::default().fg(t.border),
            ),
        ]),
        Line::from(Span::styled(
            format!(
                "  Row {}/{n}, Col {}/{n}. Enter space-separated values, press Enter to submit.",
                current_row + 1,
                current_col + 1
            ),
            Style::default().fg(t.text_muted),
        )),
    ];
    frame.render_widget(Paragraph::new(progress_lines), chunks[1]);

    // Matrix viewport
    let mat_area = chunks[2];
    let available_rows = (mat_area.height as usize).saturating_sub(2);
    let cell_width = 10usize;
    let row_label_width = 6usize;
    let available_cols =
        (mat_area.width as usize).saturating_sub(row_label_width + 4) / cell_width;
    let display_rows = available_rows.min(n);
    let display_cols = available_cols.min(n).max(1);

    let scroll_row = if current_row >= display_rows / 2 {
        (current_row - display_rows / 2).min(n.saturating_sub(display_rows))
    } else {
        0
    };
    let scroll_col = if current_col >= display_cols / 2 {
        (current_col - display_cols / 2).min(n.saturating_sub(display_cols))
    } else {
        0
    };

    let mut matrix_lines: Vec<Line> = Vec::new();

    // Column header
    let mut col_header_spans = vec![Span::styled(
        format!("{:>width$}", "", width = row_label_width),
        Style::default().fg(t.text_dim),
    )];
    for c in scroll_col..(scroll_col + display_cols).min(n) {
        col_header_spans.push(Span::styled(
            format!("{:>width$}", format!("C{}", c + 1), width = cell_width),
            Style::default().fg(t.text_dim),
        ));
    }
    if scroll_col + display_cols < n {
        col_header_spans.push(Span::styled(
            " ...",
            Style::default().fg(t.text_dim),
        ));
    }
    matrix_lines.push(Line::from(col_header_spans));

    // Matrix rows
    for r in scroll_row..(scroll_row + display_rows).min(n) {
        let mut spans = vec![Span::styled(
            format!("R{:<4}\u{2502}", r + 1),
            Style::default().fg(t.text_dim),
        )];

        for c in scroll_col..(scroll_col + display_cols).min(n) {
            let idx = r * n + c;
            if idx < filled {
                spans.push(Span::styled(
                    format!("{:>9.2} ", app.manual_data[idx]),
                    Style::default().fg(t.text),
                ));
            } else if idx == filled {
                spans.push(Span::styled(
                    format!("{:>9} ", "\u{2588}_"),
                    Style::default().fg(t.ok),
                ));
            } else {
                spans.push(Span::styled(
                    format!("{:>9} ", "\u{00b7}"),
                    Style::default().fg(t.border),
                ));
            }
        }

        if scroll_col + display_cols < n {
            spans.push(Span::styled(
                " ...",
                Style::default().fg(t.text_dim),
            ));
        }
        spans.push(Span::styled(
            "\u{2502}",
            Style::default().fg(t.text_dim),
        ));
        matrix_lines.push(Line::from(spans));
    }

    if scroll_row + display_rows < n {
        matrix_lines.push(Line::from(Span::styled(
            format!(
                "{:>width$}\u{2502} ... ({} more rows) ... \u{2502}",
                "",
                n - scroll_row - display_rows,
                width = row_label_width
            ),
            Style::default().fg(t.text_dim),
        )));
    }

    let block = Block::default()
        .title(Span::styled(
            format!(
                " MATRIX {name} [{}-{}] \u{00d7} [{}-{}] ",
                scroll_row + 1,
                (scroll_row + display_rows).min(n),
                scroll_col + 1,
                (scroll_col + display_cols).min(n),
            ),
            title_style,
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(matrix_lines).block(block), mat_area);

    // Input line
    let input_lines = vec![Line::from(vec![
        Span::styled("  > ", Style::default().fg(t.accent)),
        Span::styled(
            format!("{}_", &app.manual_buffer),
            Style::default().fg(t.ok),
        ),
    ])];
    frame.render_widget(Paragraph::new(input_lines), chunks[3]);

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter]", key_hint),
        Span::styled(" Submit row  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc]", key_hint),
        Span::styled(" Cancel", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[4],
    );
}

// ─── Computing Screen ───────────────────────────────────────────────────────

fn render_computing(app: &App, algorithm: &str, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let block = Block::default()
        .title(Span::styled(" COMPUTING ", Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));

    let (fraction, elapsed_str, eta_str) = if let Some(ref task) = app.compute_task {
        let frac = task.progress.fraction();
        let elapsed = task.eta.elapsed_secs();
        let elapsed_s = if elapsed < 60.0 {
            format!("{:.1}s", elapsed)
        } else {
            format!("{:.0}m {:.0}s", elapsed / 60.0, elapsed % 60.0)
        };
        let eta_s = task.eta.format_eta(frac);
        (frac, elapsed_s, eta_s)
    } else {
        (0.0, "0.0s".into(), "ETA: calculating...".into())
    };

    let pct = (fraction * 100.0).min(100.0);
    let bar_width: usize = 30;
    let filled = ((fraction * bar_width as f64).round() as usize).min(bar_width);
    let empty = bar_width - filled;
    let bar = format!(
        "[{}{}] {:.0}%",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(empty),
        pct,
    );

    // Select gear frames based on progress (sequential activation)
    let gear_idx = app.gear_frame % 4;
    let gear_lines: &[&str; 7] = if fraction < 0.10 {
        &GEAR_SINGLE[gear_idx]
    } else if fraction < 0.50 {
        &GEAR_DOUBLE[gear_idx]
    } else {
        &GEAR_FRAMES[gear_idx]
    };

    let mut lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            algorithm.to_string(),
            Style::default().fg(t.ok).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            bar,
            Style::default().fg(t.accent),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("Elapsed: {elapsed_str}"), Style::default().fg(t.text_dim)),
            Span::styled("  \u{2502}  ", Style::default().fg(t.border)),
            Span::styled(eta_str, Style::default().fg(t.text_dim)),
        ]),
        Line::from(""),
    ];

    // Gear animation
    for gear_line in gear_lines {
        lines.push(Line::from(Span::styled(
            *gear_line,
            Style::default().fg(t.accent),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "[Esc] Cancel",
        Style::default().fg(t.text_muted),
    )));

    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .alignment(Alignment::Center),
        area,
    );
}

// ─── Compute Error Screen ───────────────────────────────────────────────────

fn render_compute_error(message: &str, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let block = Block::default()
        .title(Span::styled(
            " COMPUTATION FAILED ",
            Style::default().fg(Color::Red).bg(t.bg).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .style(Style::default().bg(t.bg));

    let lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "\u{26A0}  Error",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            message.to_string(),
            Style::default().fg(t.text),
        )),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "The computation thread crashed or returned an error.",
            Style::default().fg(t.text_dim),
        )),
        Line::from(Span::styled(
            "The application remains stable \u{2014} no data was lost.",
            Style::default().fg(t.text_dim),
        )),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "[Esc] Return to Main Menu",
            Style::default().fg(t.text_muted),
        )),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .alignment(Alignment::Center),
        area,
    );
}

// ─── Results Screen (redesigned with timing graph) ──────────────────────────

fn render_results(app: &App, data: &BenchmarkData, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let has_matrix = data.result_matrix.is_some();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header "COMPLETE"
            Constraint::Length(10), // results + timing (horizontal split)
            Constraint::Length(6),  // performance section
            Constraint::Min(4),    // matrix preview
            Constraint::Length(1), // footer
        ])
        .split(area);

    // ── Header ──
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  MULTIPLICATION COMPLETE",
            Style::default()
                .fg(t.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  \u{2713}", Style::default().fg(t.ok)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(header, chunks[0]);

    // ── Results + Timing (horizontal split) ──
    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Left: parameters with extended profiling
    let dim_str = if data.rows_a == data.cols_a && data.cols_a == data.cols_b {
        format!("{0} \u{00d7} {0}", data.size)
    } else {
        format!(
            "{}\u{00d7}{} \u{00d7} {}\u{00d7}{} = {}\u{00d7}{}",
            data.rows_a, data.cols_a, data.cols_a, data.cols_b, data.rows_a, data.cols_b,
        )
    };
    let mem_str = if data.matrix_memory_mb > 0.0 {
        format!("~{} ({:.1} MB matrices)", format_memory(data.peak_ram_mb), data.matrix_memory_mb)
    } else {
        format!("~{}", format_memory(data.peak_ram_mb))
    };
    let params = vec![
        Line::from(""),
        kv_line("  Algorithm ", &data.algorithm, t.text_bright, t.text_dim),
        kv_line("  SIMD      ", &data.simd_level, t.accent, t.text_dim),
        kv_line("  Dimensions", &dim_str, t.text, t.text_dim),
        kv_line("  Threads   ", &format!("{}", data.threads), t.text, t.text_dim),
        kv_line("  Memory    ", &mem_str, t.text, t.text_dim),
        kv_line("  Alloc     ", &format!("{:.2} ms", data.allocation_ms), t.text_muted, t.text_dim),
    ];
    let params_para = Paragraph::new(params).block(
        Block::default()
            .title(Span::styled(
                " Results ",
                Style::default().fg(t.text_dim),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(params_para, mid_chunks[0]);

    // Right: timing breakdown as mini bar chart
    let total_ms = data.gen_time_ms.unwrap_or(0.0)
        + data.padding_time_ms
        + data.compute_time_ms
        + data.unpadding_time_ms;

    let timings = [
        (
            "Gen    ",
            data.gen_time_ms.unwrap_or(0.0),
        ),
        ("Pad    ", data.padding_time_ms),
        ("Compute", data.compute_time_ms),
        ("Unpad  ", data.unpadding_time_ms),
    ];
    let max_t = timings
        .iter()
        .map(|(_, tv)| *tv)
        .fold(0.0_f64, f64::max)
        .max(0.001);
    const BAR_W: usize = 14;

    let mut timing_lines = vec![Line::from("")];
    for (label, tv) in &timings {
        let filled = ((tv / max_t) * BAR_W as f64) as usize;
        let color = if label.contains("Compute") {
            t.accent
        } else {
            t.text_dim
        };
        timing_lines.push(Line::from(vec![
            Span::styled(format!("  {label}  "), Style::default().fg(t.text_dim)),
            Span::styled("\u{2593}".repeat(filled), Style::default().fg(color)),
            Span::styled(
                "\u{2591}".repeat(BAR_W.saturating_sub(filled)),
                Style::default().fg(t.border),
            ),
            Span::styled(
                format!("  {:>7.1} ms", tv),
                Style::default().fg(t.text_muted),
            ),
        ]));
    }
    timing_lines.push(Line::from(""));
    timing_lines.push(Line::from(vec![
        Span::styled("  Total    ", Style::default().fg(t.text_dim)),
        Span::styled(
            format!("{}", format_duration(total_ms)),
            Style::default()
                .fg(t.text_bright)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    let timing_para = Paragraph::new(timing_lines).block(
        Block::default()
            .title(Span::styled(
                " Time Breakdown ",
                Style::default().fg(t.text_dim),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(timing_para, mid_chunks[1]);

    // ── Performance section — enhanced GFLOPS with formula breakdown ──
    let peak = &app.sys_info.peak_estimate;
    let peak_gflops = peak.peak_gflops;
    let efficiency = if peak_gflops > 0.0 {
        (data.gflops / peak_gflops * 100.0).min(100.0)
    } else {
        0.0
    };

    let bar_width = 30usize;
    let bar_filled = ((efficiency / 100.0) * bar_width as f64).round() as usize;
    let bar_empty = bar_width.saturating_sub(bar_filled);

    let perf_lines = vec![
        Line::from(""),
        // Line 1: GFLOPS value + bar + efficiency %
        Line::from(vec![
            Span::styled("  GFLOPS    ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:.2}  ", data.gflops),
                Style::default()
                    .fg(t.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "\u{2588}".repeat(bar_filled),
                Style::default().fg(t.ok),
            ),
            Span::styled(
                "\u{2591}".repeat(bar_empty),
                Style::default().fg(t.border),
            ),
            if peak_gflops > 0.0 {
                Span::styled(
                    format!("  {:.1}% of theoretical peak", efficiency),
                    Style::default().fg(t.text_muted),
                )
            } else {
                Span::styled("", Style::default())
            },
        ]),
        // Line 2: Peak formula breakdown
        Line::from(vec![
            Span::styled("  Peak      ", Style::default().fg(t.text_dim)),
            Span::styled(
                peak.formula_string(),
                Style::default().fg(t.text_muted),
            ),
        ]),
        // Line 3: Total FLOPs
        Line::from(vec![
            Span::styled("  FLOPs     ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:.2e}  (2 \u{00d7} {} \u{00d7} {} \u{00d7} {})",
                    data.total_flops as f64,
                    data.size, data.size, data.size),
                Style::default().fg(t.text_muted),
            ),
        ]),
        // Line 4: Assessment comment
        Line::from(vec![
            Span::styled("  Note      ", Style::default().fg(t.text_dim)),
            Span::styled(
                crate::common::PeakEstimate::assessment(efficiency),
                Style::default().fg(t.text),
            ),
        ]),
        // Line 5: Source info (dim)
        Line::from(vec![
            Span::styled("  Source     ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("freq={}, FMA={}", peak.freq_source, peak.fma_source),
                Style::default().fg(t.text_dim),
            ),
        ]),
        Line::from(""),
    ];

    let perf_block = Block::default()
        .title(Span::styled(
            " Performance ",
            Style::default().fg(t.text_dim),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(
        Paragraph::new(perf_lines).block(perf_block),
        chunks[2],
    );

    // ── Matrix Preview ──
    if let Some(ref mat) = data.result_matrix {
        let mut mat_lines: Vec<Line> = vec![Line::from(Span::styled(
            "  Result matrix C = A \u{00d7} B:",
            Style::default().fg(t.text_muted),
        ))];
        let show_r = mat.rows().min(8);
        let show_c = mat.cols().min(8);
        for i in 0..show_r {
            let mut spans = vec![Span::styled(
                "  \u{2502}",
                Style::default().fg(t.accent),
            )];
            for j in 0..show_c {
                spans.push(Span::styled(
                    format!("{:>10.4}", mat.get(i, j)),
                    Style::default().fg(t.text),
                ));
            }
            if mat.cols() > show_c {
                spans.push(Span::styled(
                    "  ...",
                    Style::default().fg(t.text_dim),
                ));
            }
            spans.push(Span::styled(
                " \u{2502}",
                Style::default().fg(t.accent),
            ));
            mat_lines.push(Line::from(spans));
        }
        if mat.rows() > show_r {
            mat_lines.push(Line::from(Span::styled(
                format!(
                    "  \u{2502} ... ({} more rows) ... \u{2502}",
                    mat.rows() - show_r
                ),
                Style::default().fg(t.text_dim),
            )));
        }

        let mat_block = Block::default()
            .title(Span::styled(
                " Output Matrix ",
                Style::default().fg(t.text_dim),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg));
        frame.render_widget(
            Paragraph::new(mat_lines).block(mat_block),
            chunks[3],
        );
    } else {
        let info = Paragraph::new(Line::from(Span::styled(
            "  Matrix too large to display (size > 16). Result computed successfully.",
            Style::default().fg(t.text_muted),
        )))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border))
                .style(Style::default().bg(t.bg)),
        );
        frame.render_widget(info, chunks[3]);
    }

    // Footer with save options
    let csv_status = if app.csv_saved { " [Saved!]" } else { "" };
    let mat_hint = if has_matrix {
        let mat_status = if app.matrix_saved { " [Saved!]" } else { "" };
        format!("  [M] Save matrix{mat_status}")
    } else {
        String::new()
    };

    let view_hint = if has_matrix {
        "   [V] View"
    } else {
        ""
    };

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [S]", key_hint),
        Span::styled(
            format!(" Save CSV{csv_status}"),
            Style::default().fg(t.text_muted),
        ),
        Span::styled(&mat_hint, Style::default().fg(t.text_muted)),
        Span::styled(view_hint, Style::default().fg(t.text_muted)),
        Span::styled("   [Enter/Esc]", key_hint),
        Span::styled(" Back to menu", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[4],
    );
}

/// Helper: format a floating-point value in scientific notation when very small or large.
pub(crate) fn fmt_sci(val: f64) -> String {
    if val.abs() < 1e-3 || val.abs() > 1e6 {
        format!("{:.4e}", val)
    } else {
        format!("{:.4}", val)
    }
}

/// Helper: key-value line for results panel.
pub(crate) fn kv_line(key: &str, value: &str, value_color: ratatui::style::Color, key_color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{key}   "),
            Style::default().fg(key_color),
        ),
        Span::styled(value.to_string(), Style::default().fg(value_color)),
    ])
}

// ─── History Screen ──────────────────────────────────────────────────────────

fn render_history_screen(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let history = &app.session_history;
    let selected = app.history_selected;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),    // list
            Constraint::Length(9), // detail panel
            Constraint::Length(1), // footer
        ])
        .split(area);

    // ── Header ──
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  SESSION HISTORY",
            Style::default()
                .fg(t.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("   {} entries", history.len()),
            Style::default().fg(t.text_muted),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(header, chunks[0]);

    // ── Entry list: ID/name first, then algo, size, timing ──
    let items: Vec<ListItem> = history
        .entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let is_sel = i == selected;
            let prefix = if is_sel { "\u{25b6}" } else { " " };

            // Truncate computation_id to 22 chars so line stays readable
            let id = &entry.data.computation_id;
            let id_display = if id.len() > 22 {
                format!("{}…", &id[..21])
            } else {
                format!("{:<22}", id)
            };

            let algo_short = {
                let a = &entry.data.algorithm;
                if a.len() > 22 { format!("{}…", &a[..21]) } else { format!("{:<22}", a) }
            };

            let line = Line::from(vec![
                Span::styled(
                    format!("{prefix} "),
                    Style::default().fg(t.accent),
                ),
                Span::styled(
                    id_display,
                    Style::default().fg(if is_sel { t.accent } else { t.text_bright })
                        .add_modifier(if is_sel { Modifier::BOLD } else { Modifier::empty() }),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(algo_short, Style::default().fg(t.text_muted)),
                Span::styled(
                    format!("  {}×{}  ", entry.data.size, entry.data.size),
                    Style::default().fg(t.text_dim),
                ),
                Span::styled(
                    format!("{:>7.0}ms  {:>5.1}G",
                        entry.data.compute_time_ms, entry.data.gflops),
                    Style::default().fg(if is_sel { t.text_bright } else { t.text }),
                ),
            ]);

            if is_sel {
                ListItem::new(line).style(Style::default().bg(t.surface))
            } else {
                ListItem::new(line)
            }
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(list, chunks[1]);

    // ── Detail panel for selected entry ──
    if let Some(entry) = history.entries.get(selected) {
        let d = &entry.data;
        let _total_ms = d.gen_time_ms.unwrap_or(0.0)
            + d.padding_time_ms
            + d.compute_time_ms
            + d.unpadding_time_ms;

        let details = vec![
            Line::from(""),
            kv_line("  ID       ", &d.computation_id, t.accent, t.text_dim),
            kv_line("  Algorithm", &d.algorithm, t.text_bright, t.text_dim),
            kv_line(
                "  Size     ",
                &format!("{} \u{00d7} {}", d.size, d.size),
                t.text, t.text_dim,
            ),
            kv_line("  SIMD     ", &d.simd_level, t.accent, t.text_dim),
            kv_line("  Threads  ", &d.threads.to_string(), t.text, t.text_dim),
            kv_line(
                "  Compute  ",
                &format!("{:.2} ms", d.compute_time_ms),
                t.text, t.text_dim,
            ),
            kv_line(
                "  GFLOPS   ",
                &format!("{:.2}", d.gflops),
                t.accent, t.text_dim,
            ),
            {
                let peak_g = app.sys_info.peak_estimate.peak_gflops;
                let eff = if peak_g > 0.0 { (d.gflops / peak_g * 100.0).min(100.0) } else { 0.0 };
                kv_line(
                    "  Effic.   ",
                    &format!("{:.1}% of {:.0} GFLOPS peak", eff, peak_g),
                    t.text_muted, t.text_dim,
                )
            },
            kv_line(
                "  RAM      ",
                &format!("~{}", format_memory(d.peak_ram_mb)),
                t.text, t.text_dim,
            ),
        ];

        let detail_block = Block::default()
            .title(Span::styled(
                " SELECTED ",
                Style::default().fg(t.text_dim),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg));
        frame.render_widget(Paragraph::new(details).block(detail_block), chunks[2]);
    }

    // ── Footer ──
    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled(
            "  [\u{2191}\u{2193}]",
            key_hint,
        ),
        Span::styled(" Navigate", Style::default().fg(t.text_muted)),
        Span::styled("   [R]", key_hint),
        Span::styled(" Re-run", Style::default().fg(t.text_muted)),
        Span::styled("   [D]", key_hint),
        Span::styled(" Delete", Style::default().fg(t.text_muted)),
        Span::styled("   [Q]", key_hint),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[3],
    );
}

// ─── Coming Soon ────────────────────────────────────────────────────────────

// ─── Viewer File Input ──────────────────────────────────────────────────────

fn render_viewer_file_input(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let title = Paragraph::new(Line::from(Span::styled("  MATRIX VIEWER", title_style)));
    frame.render_widget(title, chunks[0]);

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Enter path to a CSV matrix file:",
            Style::default().fg(t.text),
        )),
        Line::from(Span::styled(
            "  (comma-separated f64 values, one row per line)",
            Style::default().fg(t.text_muted),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  > ", Style::default().fg(t.accent)),
            Span::styled(
                format!("{}_", &app.viewer_path_input),
                Style::default().fg(t.ok),
            ),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(" FILE PATH ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter]", key_hint),
        Span::styled(" Load  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Tab]", key_hint),
        Span::styled(" Browse  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc]", key_hint),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[2],
    );
}

// ─── Matrix Viewer ──────────────────────────────────────────────────────────

fn render_matrix_viewer(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let mat = match &app.viewer_matrix {
        Some(m) => m,
        None => return,
    };
    let stats = match &app.viewer_stats {
        Some(s) => s,
        None => return,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),    // body (matrix + stats)
            Constraint::Length(1), // footer
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);

    // ── Header ──
    let mem_mb = (mat.rows() * mat.cols() * 8) / (1024 * 1024);
    let header = Paragraph::new(Line::from(vec![
        Span::styled("  MATRIX VIEWER", title_style),
        Span::styled("  \u{00b7}  ", Style::default().fg(t.text_dim)),
        Span::styled(&app.viewer_filename, Style::default().fg(t.text)),
        Span::styled("  \u{00b7}  ", Style::default().fg(t.text_dim)),
        Span::styled(
            format!("{}\u{00d7}{}  \u{00b7}  {} MB", mat.rows(), mat.cols(), mem_mb),
            Style::default().fg(t.text_muted),
        ),
        Span::styled(
            format!("  \u{00b7}  [{}, {}]", app.viewer_cursor_row, app.viewer_cursor_col),
            Style::default().fg(t.text_dim),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(header, chunks[0]);

    // ── Body: matrix grid (left) + stats (right) ──
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(30), Constraint::Length(app.terminal_scale.stats_panel_width())])
        .split(chunks[1]);

    // ── Matrix grid ──
    let grid_area = body_chunks[0];
    let cell_w = app.viewer_precision + 7; // sign + digits + dot + precision + space
    let row_label_w = 8usize;
    let inner_w = (grid_area.width as usize).saturating_sub(4); // borders + margin
    let inner_h = (grid_area.height as usize).saturating_sub(3); // borders + col header

    let visible_cols = ((inner_w.saturating_sub(row_label_w)) / cell_w).max(1);
    let visible_rows = inner_h.max(1);

    let mut grid_lines: Vec<Line> = Vec::new();

    // Column header row
    let mut col_spans = vec![Span::styled(
        format!("{:>width$}", "", width = row_label_w),
        Style::default().fg(t.text_dim),
    )];
    for c in app.viewer_scroll_col..(app.viewer_scroll_col + visible_cols).min(mat.cols()) {
        col_spans.push(Span::styled(
            format!("{:>width$}", c, width = cell_w),
            Style::default().fg(t.text_dim),
        ));
    }
    grid_lines.push(Line::from(col_spans));

    // Cache visible sizes for input handler (interior mutation via Cell)
    app.viewer_visible_rows.set(visible_rows);
    app.viewer_visible_cols.set(visible_cols);

    // Data rows
    for r in app.viewer_scroll_row..(app.viewer_scroll_row + visible_rows).min(mat.rows()) {
        let mut row_spans = vec![Span::styled(
            format!("{:>6}  ", r),
            Style::default().fg(t.text_dim),
        )];

        for c in app.viewer_scroll_col..(app.viewer_scroll_col + visible_cols).min(mat.cols()) {
            let val = mat.get(r, c);
            let is_cursor = r == app.viewer_cursor_row && c == app.viewer_cursor_col;
            let style = if is_cursor {
                // Inverted accent: background = accent, foreground = bg
                Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD)
            } else if app.viewer_highlight {
                if (val - stats.min).abs() < 1e-10 {
                    Style::default().fg(t.crit) // min → red
                } else if (val - stats.max).abs() < 1e-10 {
                    Style::default().fg(t.ok) // max → green
                } else {
                    Style::default().fg(t.text)
                }
            } else {
                Style::default().fg(t.text)
            };
            row_spans.push(Span::styled(
                format!("{:>width$.prec$}", val, width = cell_w, prec = app.viewer_precision),
                style,
            ));
        }
        grid_lines.push(Line::from(row_spans));
    }

    let grid_block = Block::default()
        .title(Span::styled(" DATA ", Style::default().fg(t.text_dim)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(grid_lines).block(grid_block), grid_area);

    // ── Stats panel ──
    let hl_status = if app.viewer_highlight { "ON" } else { "OFF" };
    let mut stats_lines = vec![
        Line::from(""),
        kv_line("  Min    ", &format!("{:.6}", stats.min), t.crit, t.text_dim),
        kv_line("  Max    ", &format!("{:.6}", stats.max), t.ok, t.text_dim),
        kv_line("  Mean   ", &format!("{:.6}", stats.mean), t.text, t.text_dim),
        kv_line("  StdDev ", &format!("{:.6}", stats.std_dev), t.text, t.text_dim),
        Line::from(""),
        kv_line("  NNZ    ", &format!("{}", stats.nonzero), t.text, t.text_dim),
        kv_line("  Zeros  ", &format!("{:.1}%", stats.sparsity_pct), t.text_muted, t.text_dim),
        Line::from(""),
        kv_line("  Frob.  ", &fmt_sci(stats.frobenius_norm), t.text, t.text_dim),
        kv_line("  ||A||1 ", &fmt_sci(stats.norm_1), t.text, t.text_dim),
        kv_line("  ||A||inf", &fmt_sci(stats.norm_infinity), t.text, t.text_dim),
    ];

    // Square-matrix properties
    if let Some(trace) = stats.trace {
        stats_lines.push(Line::from(""));
        stats_lines.push(kv_line("  Trace  ", &fmt_sci(trace), t.text, t.text_dim));

        if let Some(sym) = stats.is_symmetric {
            let (sym_str, sym_color) = if sym { ("YES", t.ok) } else { ("NO", t.warn) };
            stats_lines.push(kv_line("  Symm.  ", sym_str, sym_color, t.text_dim));
        }

        if let Some((kappa, approx)) = stats.condition_estimate {
            let prefix = if approx { "~" } else { "" };
            let assess = crate::numerics::condition_assessment(kappa);
            let kappa_color = if kappa < 10.0 {
                t.ok
            } else if kappa < 1e6 {
                t.accent
            } else {
                t.crit
            };
            stats_lines.push(kv_line(
                "  Cond.  ",
                &format!("{prefix}{:.1e}", kappa),
                kappa_color,
                t.text_dim,
            ));
            stats_lines.push(kv_line("         ", assess, kappa_color, t.text_dim));
        }
    }

    stats_lines.push(Line::from(""));
    stats_lines.push(kv_line("  Prec.  ", &format!("{}", app.viewer_precision), t.accent, t.text_dim));
    stats_lines.push(kv_line("  Hi-lite", hl_status, t.accent, t.text_dim));

    // File metadata section (shown only when matrix was saved by Flust)
    if let Some(ref meta) = app.viewer_loaded_metadata {
        stats_lines.push(Line::from(""));
        stats_lines.push(Line::from(Span::styled(
            "  \u{2500}\u{2500} File Info \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
            Style::default().fg(t.text_dim),
        )));
        if let Some(ref v) = meta.algorithm {
            let short = if v.len() > 18 { &v[..18] } else { v.as_str() };
            stats_lines.push(kv_line("  Algo   ", short, t.accent, t.text_dim));
        }
        if let Some(ref v) = meta.cpu {
            let short = if v.len() > 18 { &v[..18] } else { v.as_str() };
            stats_lines.push(kv_line("  CPU    ", short, t.text_muted, t.text_dim));
        }
        if let Some(ref v) = meta.simd {
            stats_lines.push(kv_line("  SIMD   ", v, t.text_muted, t.text_dim));
        }
        if let Some(v) = meta.threads {
            stats_lines.push(kv_line("  Threads", &format!("{v}"), t.text_muted, t.text_dim));
        }
        if let Some(v) = meta.compute_ms {
            stats_lines.push(kv_line("  Compute", &format!("{v:.1} ms"), t.text, t.text_dim));
        }
        if let Some(v) = meta.gflops {
            stats_lines.push(kv_line("  GFlops ", &format!("{v:.2}"), t.accent, t.text_dim));
        }
        if let Some(v) = meta.peak_ram_mb {
            stats_lines.push(kv_line("  RAM    ", &format!("{v} MB"), t.text_muted, t.text_dim));
        }
        if let Some(ref v) = meta.timestamp {
            let short = if v.len() > 19 { &v[..19] } else { v.as_str() };
            stats_lines.push(kv_line("  Saved  ", short, t.text_dim, t.text_dim));
        }
    }

    let stats_lines = stats_lines;

    let stats_block = Block::default()
        .title(Span::styled(" STATISTICS ", Style::default().fg(t.text_dim)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(stats_lines).block(stats_block), body_chunks[1]);

    // ── Edit mode input line or exit-warning overlay ──
    if app.viewer_exit_warning {
        let warn_area = chunks[2];
        let warning = Line::from(vec![
            Span::styled("  Unsaved changes. ", Style::default().fg(t.warn)),
            Span::styled("[S]", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled(" Save  ", Style::default().fg(t.text_muted)),
            Span::styled("[Q]", Style::default().fg(t.crit).add_modifier(Modifier::BOLD)),
            Span::styled(" Discard  ", Style::default().fg(t.text_muted)),
            Span::styled("[Esc]", Style::default().fg(t.text_dim).add_modifier(Modifier::BOLD)),
            Span::styled(" Cancel", Style::default().fg(t.text_muted)),
        ]);
        frame.render_widget(Paragraph::new(warning).style(Style::default().bg(t.surface)), warn_area);
    } else if app.viewer_edit_mode {
        let edit_area = chunks[2];
        let edit_line = Line::from(vec![
            Span::styled("  Edit [", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{}, {}",  app.viewer_cursor_row, app.viewer_cursor_col),
                Style::default().fg(t.accent),
            ),
            Span::styled("]: ", Style::default().fg(t.text_dim)),
            Span::styled(&app.viewer_edit_buffer, Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
            Span::styled("\u{2588}", Style::default().fg(t.accent)),  // cursor block
            Span::styled("   [Enter] Confirm  [Esc] Cancel", Style::default().fg(t.text_dim)),
        ]);
        frame.render_widget(Paragraph::new(edit_line).style(Style::default().bg(t.surface)), edit_area);
    } else {
        let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
        let edit_indicator = if app.viewer_unsaved_changes {
            Span::styled("  * unsaved", Style::default().fg(t.warn))
        } else {
            Span::styled("", Style::default())
        };
        let footer = Line::from(vec![
            edit_indicator,
            Span::styled("  [\u{2190}\u{2191}\u{2192}\u{2193}]", key_hint),
            Span::styled(" Navigate", Style::default().fg(t.text_muted)),
            Span::styled("  [E]", key_hint),
            Span::styled(" Edit", Style::default().fg(t.text_muted)),
            Span::styled("  [S]", key_hint),
            Span::styled(" Save", Style::default().fg(t.text_muted)),
            Span::styled("  [H]", key_hint),
            Span::styled(" Highlight", Style::default().fg(t.text_muted)),
            Span::styled("  [+/-]", key_hint),
            Span::styled(" Precision", Style::default().fg(t.text_muted)),
            Span::styled("  [Q]", key_hint),
            Span::styled(" Back", Style::default().fg(t.text_muted)),
        ]);
        frame.render_widget(
            Paragraph::new(footer).style(Style::default().bg(t.surface)),
            chunks[2],
        );
    }
}

// ─── Diff Size Input ────────────────────────────────────────────────────────

fn render_diff_size_input(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let title = Paragraph::new(Line::from(Span::styled("  ALGORITHM COMPARISON", title_style)));
    frame.render_widget(title, chunks[0]);

    let est_ram = if let Ok(n) = app.diff_size_input.parse::<usize>() {
        if n > 0 {
            let mb = SystemInfo::estimate_peak_ram_mb(n);
            format!("Estimated RAM: ~{}", format_memory(mb))
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Compare two algorithms on identical input matrices.",
            Style::default().fg(t.text_muted),
        )),
        Line::from(Span::styled(
            "  Enter the dimension N (1-10000):",
            Style::default().fg(t.text),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  N = ", Style::default().fg(t.accent)),
            Span::styled(
                format!("{}_", &app.diff_size_input),
                Style::default().fg(t.ok),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {est_ram}"),
            Style::default().fg(t.text_muted),
        )),
    ];

    let block = Block::default()
        .title(Span::styled(" DIMENSION ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter]", key_hint),
        Span::styled(" Confirm  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc]", key_hint),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[2],
    );
}

// ─── Diff Algorithm Select ──────────────────────────────────────────────────

fn render_diff_alg_select(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let which = app.diff_selecting_which;
    let subtitle = if which == 1 {
        format!("  SELECT ALGORITHM 1  ({}x{})", app.chosen_size, app.chosen_size)
    } else {
        format!(
            "  SELECT ALGORITHM 2  (vs {})",
            app.diff_alg1.display_name()
        )
    };
    let title = Paragraph::new(Line::from(Span::styled(&subtitle, title_style)));
    frame.render_widget(title, chunks[0]);

    let algorithms = [
        ("1", "Naive (i-k-j)", "Simple baseline. Single thread, no optimization.", AlgorithmChoice::Naive),
        ("2", "Strassen (classic)", "Strassen + Tiling + Rayon. O(n^2.807).", AlgorithmChoice::Strassen),
        ("3", "Winograd (classic)", "Strassen variant with fewer additions (15 vs 18).", AlgorithmChoice::Winograd),
        ("4", "Strassen Hybrid", "Strip-parallel Strassen. Full core utilization.", AlgorithmChoice::StrassenHybrid),
        ("5", "Winograd Hybrid", "Strip-parallel Winograd. Full core + fewer ops.", AlgorithmChoice::WinogradHybrid),
    ];

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let default_style = Style::default().fg(t.text).bg(t.bg);
    let selected_style = Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD);
    let disabled_style = Style::default().fg(t.text_dim).bg(t.bg);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, (key, label, desc, choice)) in algorithms.iter().enumerate() {
        let is_selected = i == app.diff_select_idx;
        let is_disabled = which == 2 && *choice == app.diff_alg1;

        let arrow = if is_selected { " \u{25b8} " } else { "   " };
        let style = if is_disabled {
            disabled_style
        } else if is_selected {
            selected_style
        } else {
            default_style
        };
        let dim = if is_disabled {
            disabled_style
        } else if is_selected {
            selected_style
        } else {
            Style::default().fg(t.text_muted).bg(t.bg)
        };

        let mut label_text = format!("{label:<30}");
        if is_disabled {
            label_text = format!("{label:<30} (already selected)");
        }

        lines.push(Line::from(vec![
            Span::styled(arrow, style),
            Span::styled(format!("[{key}] "), if is_disabled { disabled_style } else { key_hint }),
            Span::styled(label_text, style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      ", default_style),
            Span::styled(*desc, dim),
        ]));
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(Span::styled(
            if which == 1 { " ALGORITHM 1 " } else { " ALGORITHM 2 " },
            title_style,
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    render_nav_footer(frame, chunks[2], t);
}

// ─── Diff Results ───────────────────────────────────────────────────────────

fn render_diff_results(data: &DiffResultData, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(10),   // side-by-side
            Constraint::Length(5), // winner + verification
            Constraint::Length(1), // footer
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled("  ALGORITHM COMPARISON", title_style),
        Span::styled(
            format!("  \u{00b7}  {}\u{00d7}{}", data.size, data.size),
            Style::default().fg(t.text_muted),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border))
            .style(Style::default().bg(t.bg)),
    );
    frame.render_widget(header, chunks[0]);

    // Side-by-side panels
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    let max_time = data.time1_ms.max(data.time2_ms).max(0.001);
    let bar_w = 20usize;

    // Left panel (algorithm 1)
    let filled1 = ((data.time1_ms / max_time) * bar_w as f64).round() as usize;
    let is_winner1 = data.time1_ms <= data.time2_ms;
    let color1 = if is_winner1 { t.ok } else { t.text_muted };

    let left_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", data.alg1_name),
            Style::default().fg(if is_winner1 { t.text_bright } else { t.text }).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  Time   ", &format_duration(data.time1_ms), color1, t.text_dim),
        kv_line("  GFLOPS ", &format!("{:.2}", data.gflops1), t.accent, t.text_dim),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled("\u{2593}".repeat(filled1), Style::default().fg(color1)),
            Span::styled(
                "\u{2591}".repeat(bar_w.saturating_sub(filled1)),
                Style::default().fg(t.border),
            ),
        ]),
    ];

    let left_block = Block::default()
        .title(Span::styled(
            if is_winner1 { " \u{2605} WINNER " } else { " " },
            Style::default().fg(if is_winner1 { t.ok } else { t.text_dim }),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if is_winner1 { t.ok } else { t.border }))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(left_lines).block(left_block), mid[0]);

    // Right panel (algorithm 2)
    let filled2 = ((data.time2_ms / max_time) * bar_w as f64).round() as usize;
    let is_winner2 = !is_winner1;
    let color2 = if is_winner2 { t.ok } else { t.text_muted };

    let right_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", data.alg2_name),
            Style::default().fg(if is_winner2 { t.text_bright } else { t.text }).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  Time   ", &format_duration(data.time2_ms), color2, t.text_dim),
        kv_line("  GFLOPS ", &format!("{:.2}", data.gflops2), t.accent, t.text_dim),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled("\u{2593}".repeat(filled2), Style::default().fg(color2)),
            Span::styled(
                "\u{2591}".repeat(bar_w.saturating_sub(filled2)),
                Style::default().fg(t.border),
            ),
        ]),
    ];

    let right_block = Block::default()
        .title(Span::styled(
            if is_winner2 { " \u{2605} WINNER " } else { " " },
            Style::default().fg(if is_winner2 { t.ok } else { t.text_dim }),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if is_winner2 { t.ok } else { t.border }))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(right_lines).block(right_block), mid[1]);

    // Winner + verification
    let verify_icon = if data.is_match { "\u{2713}" } else { "\u{2717}" };
    let verify_color = if data.is_match { t.ok } else { t.crit };
    let verify_text = if data.is_match {
        format!("Verified: {} Mathematically identical (max diff: {:.2e})", verify_icon, data.max_diff)
    } else {
        format!("WARNING: {} Results differ! Max diff: {:.2e}", verify_icon, data.max_diff)
    };

    let bottom_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  WINNER:  \u{25b6}  ", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("{}  is  {:.1}\u{00d7} faster", data.winner, data.speedup),
                Style::default().fg(t.ok).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            format!("  {verify_text}"),
            Style::default().fg(verify_color),
        )),
    ];

    let bottom_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(bottom_lines).block(bottom_block), chunks[2]);

    // Footer
    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter/Esc]", key_hint),
        Span::styled(" Back to menu", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[3],
    );
}

// ─── File Compare Input ─────────────────────────────────────────────────────

fn render_file_compare_input(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    which: &str,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let title = Paragraph::new(Line::from(Span::styled(
        format!("  MATRIX FILE COMPARISON \u{2014} Load Matrix {which}"),
        title_style,
    )));
    frame.render_widget(title, chunks[0]);

    let path = if which == "A" {
        &app.file_compare_path_a
    } else {
        &app.file_compare_path_b
    };

    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  Enter path to CSV file for matrix {which}:"),
            Style::default().fg(t.text),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Path: ", Style::default().fg(t.accent)),
            Span::styled(
                format!("{path}_"),
                Style::default().fg(t.ok),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Format: plain CSV (comma-separated f64 rows, no header)",
            Style::default().fg(t.text_muted),
        )),
    ];

    if let Some(ref err) = app.file_compare_error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {err}"),
            Style::default().fg(t.crit),
        )));
    }

    if which == "B" {
        if let Some(ref mat_a) = app.file_compare_matrix_a {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("  Matrix A loaded: {}x{}", mat_a.rows(), mat_a.cols()),
                Style::default().fg(t.ok),
            )));
        }
    }

    let block = Block::default()
        .title(Span::styled(
            format!(" MATRIX {which} "),
            title_style,
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter]", key_hint),
        Span::styled(" Load  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc]", key_hint),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[2],
    );
}

// ─── File Compare Results ───────────────────────────────────────────────────

fn render_file_compare_results(
    data: &FileCompareData,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(9),  // file info + global metrics
            Constraint::Min(10),   // quadrant + V&V
            Constraint::Length(1), // footer
        ])
        .split(area);

    let title_style = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let r = &data.result;

    // Header
    let header = Paragraph::new(Line::from(Span::styled(
        format!("  MATRIX FILE COMPARISON  {}x{}", r.rows, r.cols),
        title_style,
    )));
    frame.render_widget(header, chunks[0]);

    // Middle: file info (left) + global metrics (right)
    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(chunks[1]);

    // Left panel: file info
    let assessment_color = match r.assessment.severity() {
        0 => t.ok,
        1 => t.accent,
        _ => t.crit,
    };

    let left_lines = vec![
        Line::from(Span::styled(
            format!("  A: {}", data.path_a),
            Style::default().fg(t.text),
        )),
        Line::from(Span::styled(
            format!("     {}x{}  ||A||_F = {:.4e}", data.dims_a.0, data.dims_a.1, r.frobenius_norm_a),
            Style::default().fg(t.text_muted),
        )),
        Line::from(Span::styled(
            format!("     Sparsity: {:.1}%", r.sparsity_a_pct),
            Style::default().fg(t.text_dim),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  B: {}", data.path_b),
            Style::default().fg(t.text),
        )),
        Line::from(Span::styled(
            format!("     {}x{}  ||B||_F = {:.4e}", data.dims_b.0, data.dims_b.1, r.frobenius_norm_b),
            Style::default().fg(t.text_muted),
        )),
        Line::from(Span::styled(
            format!("     Sparsity: {:.1}%", r.sparsity_b_pct),
            Style::default().fg(t.text_dim),
        )),
    ];
    let left_block = Block::default()
        .title(Span::styled(" FILES ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(left_lines).block(left_block), mid_chunks[0]);

    // Right panel: global metrics
    let right_lines = vec![
        Line::from(vec![
            Span::styled("  Max |diff|:   ", Style::default().fg(t.text_muted)),
            Span::styled(format!("{:.4e}", r.max_abs_diff), Style::default().fg(t.text_bright)),
        ]),
        Line::from(vec![
            Span::styled("  Mean |diff|:  ", Style::default().fg(t.text_muted)),
            Span::styled(format!("{:.4e}", r.mean_abs_diff), Style::default().fg(t.text)),
        ]),
        Line::from(vec![
            Span::styled("  RMSE:         ", Style::default().fg(t.text_muted)),
            Span::styled(format!("{:.4e}", r.rms_diff), Style::default().fg(t.text)),
        ]),
        Line::from(vec![
            Span::styled("  ||A-B||_F:    ", Style::default().fg(t.text_muted)),
            Span::styled(format!("{:.4e}", r.frobenius_norm_diff), Style::default().fg(t.text)),
        ]),
        Line::from(vec![
            Span::styled("  Rel. error:   ", Style::default().fg(t.text_muted)),
            Span::styled(format!("{:.4e}", r.relative_error), Style::default().fg(t.text_bright)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("  Match (\u{03b5}=1e-9): ", ),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(format!("{:.2}%", r.match_pct), Style::default().fg(assessment_color)),
        ]),
        Line::from(vec![
            Span::styled("  Sign changes: ", Style::default().fg(t.text_muted)),
            Span::styled(format!("{}", r.sign_changes), Style::default().fg(t.text)),
        ]),
    ];
    let right_block = Block::default()
        .title(Span::styled(" DIFFERENCE ANALYSIS ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(right_lines).block(right_block), mid_chunks[1]);

    // Bottom: quadrant analysis (left) + V&V assessment (right)
    let bot_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[2]);

    // Quadrant table
    let mut q_lines = Vec::new();
    q_lines.push(Line::from(vec![
        Span::styled("  Quadrant      ", Style::default().fg(t.text_dim)),
        Span::styled("RMS         ", Style::default().fg(t.text_dim)),
        Span::styled("Max         ", Style::default().fg(t.text_dim)),
        Span::styled("Match", Style::default().fg(t.text_dim)),
    ]));
    for q in &r.quadrants {
        q_lines.push(Line::from(vec![
            Span::styled(format!("  {:<14}", q.label), Style::default().fg(t.text)),
            Span::styled(format!("{:<12.3e}", q.rms_diff), Style::default().fg(t.text_muted)),
            Span::styled(format!("{:<12.3e}", q.max_diff), Style::default().fg(t.text_muted)),
            Span::styled(format!("{:.1}%", q.match_pct), Style::default().fg(t.text)),
        ]));
    }
    let q_block = Block::default()
        .title(Span::styled(" QUADRANT ANALYSIS ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(q_lines).block(q_block), bot_chunks[0]);

    // V&V Assessment
    let vv_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", r.assessment.label()),
            Style::default().fg(assessment_color).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", r.assessment.description()),
            Style::default().fg(t.text_muted),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  Comparison time: {:.2} ms", r.time_ms),
            Style::default().fg(t.text_dim),
        )),
        Line::from(Span::styled(
            format!("  Elements: {} total", r.total_count),
            Style::default().fg(t.text_dim),
        )),
        Line::from(Span::styled(
            format!("  Exact matches: {}", r.exact_matches),
            Style::default().fg(t.text_dim),
        )),
    ];
    let vv_block = Block::default()
        .title(Span::styled(" V&V ASSESSMENT ", Style::default().fg(assessment_color).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(assessment_color))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(vv_lines).block(vv_block), bot_chunks[1]);

    // Footer
    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [Enter/Esc]", key_hint),
        Span::styled(" Back to menu", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[3],
    );
}

// ─── Coming Soon ────────────────────────────────────────────────────────────

fn render_coming_soon(label: &str, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let block = Block::default()
        .title(Span::styled(format!(" {label} "), Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));

    let lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "  Coming Soon",
            Style::default().fg(t.accent),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  This feature is under development.",
            Style::default().fg(t.text_muted),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Press Enter or Esc to return.",
            Style::default().fg(t.text_muted),
        )),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .alignment(Alignment::Center),
        area,
    );
}

// ─── File Browser ──────────────────────────────────────────────────────────

fn refresh_fb_entries(app: &mut App) {
    app.fb_entries.clear();
    app.fb_selected = 0;
    app.fb_scroll = 0;
    app.fb_error = None;

    // Parent directory entry
    if app.fb_current_dir.parent().is_some() {
        app.fb_entries.push(FbEntry {
            name: "..".into(),
            is_dir: true,
            size_bytes: 0,
        });
    }

    match std::fs::read_dir(&app.fb_current_dir) {
        Ok(entries) => {
            let mut dirs = Vec::new();
            let mut files = Vec::new();

            for entry in entries.flatten() {
                let meta = entry.metadata().ok();
                let is_dir = meta.as_ref().map_or(false, |m| m.is_dir());
                let size = meta.as_ref().map_or(0, |m| m.len());
                let name = entry.file_name().to_string_lossy().to_string();

                if is_dir {
                    dirs.push(FbEntry { name, is_dir: true, size_bytes: 0 });
                } else if name.ends_with(".csv") || name.ends_with(".CSV") {
                    files.push(FbEntry { name, is_dir: false, size_bytes: size });
                }
            }

            dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
            files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

            app.fb_entries.extend(dirs);
            app.fb_entries.extend(files);
        }
        Err(e) => {
            app.fb_error = Some(format!("Cannot read directory: {e}"));
        }
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn handle_file_browser(app: &mut App, key: KeyCode) {
    let return_to = match &app.screen {
        Screen::FileBrowser { return_to } => return_to.clone(),
        _ => return,
    };

    match key {
        KeyCode::Up => {
            if app.fb_selected > 0 {
                app.fb_selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.fb_selected + 1 < app.fb_entries.len() {
                app.fb_selected += 1;
            }
        }
        KeyCode::Home => {
            app.fb_selected = 0;
        }
        KeyCode::End => {
            if !app.fb_entries.is_empty() {
                app.fb_selected = app.fb_entries.len() - 1;
            }
        }
        KeyCode::PageUp => {
            app.fb_selected = app.fb_selected.saturating_sub(15);
        }
        KeyCode::PageDown => {
            app.fb_selected = (app.fb_selected + 15).min(app.fb_entries.len().saturating_sub(1));
        }
        KeyCode::Enter => {
            if let Some(entry) = app.fb_entries.get(app.fb_selected).cloned() {
                if entry.name == ".." {
                    if let Some(parent) = app.fb_current_dir.parent().map(|p| p.to_path_buf()) {
                        app.fb_current_dir = parent;
                        refresh_fb_entries(app);
                    }
                } else if entry.is_dir {
                    app.fb_current_dir.push(&entry.name);
                    refresh_fb_entries(app);
                } else {
                    // File selected — build full path and route to appropriate screen
                    let full_path = app.fb_current_dir.join(&entry.name);
                    let path_str = full_path.to_string_lossy().to_string();

                    match return_to {
                        FileBrowserReturn::MatrixViewer => {
                            app.viewer_path_input = path_str.clone();
                            // Try thermal CSV first
                            if crate::thermal_export::detect_thermal_csv(&path_str) {
                                if let Ok(data) = crate::thermal_export::load_thermal_csv(&path_str) {
                                    app.thermal_view_data = Some(data);
                                    app.viewer_filename = path_str;
                                    app.screen = Screen::ThermalViewer;
                                    return;
                                }
                            }
                            // Otherwise regular matrix CSV
                            match crate::io::load_matrix_csv_with_metadata(&path_str) {
                                Ok((mat, meta)) => {
                                    app.viewer_stats = Some(MatrixStats::compute(&mat));
                                    app.viewer_filename = path_str;
                                    app.viewer_loaded_metadata = meta;
                                    app.viewer_matrix = Some(mat);
                                    app.viewer_scroll_row = 0;
                                    app.viewer_scroll_col = 0;
                                    app.viewer_cursor_row = 0;
                                    app.viewer_cursor_col = 0;
                                    app.viewer_unsaved_changes = false;
                                    app.screen = Screen::MatrixViewer;
                                }
                                Err(e) => {
                                    app.fb_error = Some(format!("Load error: {e}"));
                                }
                            }
                        }
                        FileBrowserReturn::FileCompareA => {
                            match crate::io::load_matrix_csv(&path_str) {
                                Ok(mat) => {
                                    app.file_compare_path_a = path_str;
                                    app.file_compare_matrix_a = Some(mat);
                                    app.file_compare_error = None;
                                    app.file_compare_path_b.clear();
                                    app.screen = Screen::FileCompareInputB;
                                }
                                Err(e) => {
                                    app.fb_error = Some(format!("Load error: {e}"));
                                }
                            }
                        }
                        FileBrowserReturn::FileCompareB => {
                            match crate::io::load_matrix_csv(&path_str) {
                                Ok(mat_b) => {
                                    if let Some(ref mat_a) = app.file_compare_matrix_a {
                                        if mat_a.rows() != mat_b.rows() || mat_a.cols() != mat_b.cols() {
                                            app.fb_error = Some(format!(
                                                "Dimension mismatch: A={}x{}, B={}x{}",
                                                mat_a.rows(), mat_a.cols(), mat_b.rows(), mat_b.cols()
                                            ));
                                            return;
                                        }
                                        let result = crate::algorithms::compare_matrices_scientific(
                                            mat_a, &mat_b, crate::common::EPSILON,
                                        );
                                        let data = FileCompareData {
                                            path_a: app.file_compare_path_a.clone(),
                                            path_b: path_str,
                                            dims_a: (mat_a.rows(), mat_a.cols()),
                                            dims_b: (mat_b.rows(), mat_b.cols()),
                                            result,
                                        };
                                        app.file_compare_matrix_a = None;
                                        app.screen = Screen::FileCompareResults { data };
                                    }
                                }
                                Err(e) => {
                                    app.fb_error = Some(format!("Load error: {e}"));
                                }
                            }
                        }
                    }
                }
            }
        }
        KeyCode::Backspace => {
            if let Some(parent) = app.fb_current_dir.parent().map(|p| p.to_path_buf()) {
                app.fb_current_dir = parent;
                refresh_fb_entries(app);
            }
        }
        KeyCode::Esc => {
            match return_to {
                FileBrowserReturn::MatrixViewer => {
                    app.screen = Screen::ViewerFileInput;
                }
                FileBrowserReturn::FileCompareA => {
                    app.screen = Screen::FileCompareInputA;
                }
                FileBrowserReturn::FileCompareB => {
                    app.screen = Screen::FileCompareInputB;
                }
            }
        }
        _ => {}
    }
}

fn render_file_browser(app: &App, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // header: current path
            Constraint::Min(5),      // file list
            Constraint::Length(3),   // footer
        ])
        .split(area);

    // Header: current directory
    let dir_str = app.fb_current_dir.to_string_lossy();
    let header = Paragraph::new(Line::from(vec![
        Span::styled("  ", Style::default()),
        Span::styled(dir_str.as_ref(), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
    ]))
    .block(
        Block::default()
            .title(Span::styled(" FILE BROWSER ", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border_focus)),
    );
    frame.render_widget(header, chunks[0]);

    // File list
    let list_height = chunks[1].height.saturating_sub(2) as usize; // minus borders
    // Adjust scroll to keep selected visible
    let scroll = if app.fb_selected < app.fb_scroll {
        app.fb_selected
    } else if app.fb_selected >= app.fb_scroll + list_height {
        app.fb_selected.saturating_sub(list_height - 1)
    } else {
        app.fb_scroll
    };

    let mut lines: Vec<Line> = Vec::new();

    if let Some(ref err) = app.fb_error {
        lines.push(Line::from(Span::styled(
            format!("  {err}"),
            Style::default().fg(t.crit),
        )));
        lines.push(Line::from(""));
    }

    if app.fb_entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no CSV files in this directory)",
            Style::default().fg(t.text_dim),
        )));
    } else {
        for (i, entry) in app.fb_entries.iter().enumerate().skip(scroll).take(list_height) {
            let selected = i == app.fb_selected;
            let icon = if entry.name == ".." {
                "\u{2191} "
            } else if entry.is_dir {
                "\u{1f4c1} "
            } else {
                "\u{1f4c4} "
            };

            let size_str = if entry.is_dir {
                String::new()
            } else {
                format!("  {}", format_size(entry.size_bytes))
            };

            let prefix = if selected { "\u{25b8} " } else { "  " };
            let text = format!("{prefix}{icon}{}{size_str}", entry.name);

            let style = if selected {
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
            } else if entry.is_dir {
                Style::default().fg(t.text)
            } else {
                Style::default().fg(t.text_muted)
            };

            lines.push(Line::from(Span::styled(text, style)));
        }
    }

    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));
    frame.render_widget(Paragraph::new(lines).block(list_block), chunks[1]);

    // Footer
    let footer_text = Line::from(vec![
        Span::styled("[Enter]", Style::default().fg(t.accent)),
        Span::styled(" Open  ", Style::default().fg(t.text_muted)),
        Span::styled("[Bksp]", Style::default().fg(t.accent)),
        Span::styled(" Up  ", Style::default().fg(t.text_muted)),
        Span::styled("[Esc]", Style::default().fg(t.accent)),
        Span::styled(" Cancel", Style::default().fg(t.text_muted)),
    ]);
    let footer = Paragraph::new(footer_text)
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        );
    frame.render_widget(footer, chunks[2]);
}

// ─── Shared Footer ──────────────────────────────────────────────────────────

fn render_nav_footer(frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let key_style = Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD);
    let desc = Style::default().fg(t.text_muted).bg(t.surface);
    let sep = Span::styled("  ", desc);
    let footer = Line::from(vec![
        Span::styled(" \u{2191}\u{2193} ", key_style),
        Span::styled(" Navigate", desc),
        sep.clone(),
        Span::styled(" Enter ", key_style),
        Span::styled(" Select", desc),
        sep.clone(),
        Span::styled(" Esc ", key_style),
        Span::styled(" Back", desc),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        area,
    );
}
