// interactive.rs — Full ratatui TUI: menus, states, user interaction.
//
// Architecture: single event loop with state machine + tick-based rendering.
// App struct holds current screen state.
// Each state has its own render + input handler.
//
// Redesigned in Chapter 9: "Minimal Precision" aesthetic.
// Static centered FLUST logo, Apple-clean menu with contextual hints,
// Help/About popups, redesigned results screen with timing graph.

use std::io;
use std::sync::mpsc;
use std::time::Instant;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
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

// ─── Menu Items ──────────────────────────────────────────────────────────────

struct MenuItem {
    label: &'static str,
    shortcut: Option<char>,
    description: &'static str,
}

const MENU_ITEMS: &[MenuItem] = &[
    MenuItem {
        label: "Matrix Multiplication",
        shortcut: Some('m'),
        description: "Multiply two matrices using Strassen, Tiled, or scalar algorithms.\n\
                      Supports random generation, file input, or manual entry.",
    },
    MenuItem {
        label: "Algorithm Comparison",
        shortcut: Some('c'),
        description: "Compare two algorithms on identical random matrices.\n\
                      Reports speedup, GFLOPS, and numerical agreement.",
    },
    MenuItem {
        label: "Matrix File Comparison",
        shortcut: Some('f'),
        description: "Load two matrices from CSV files and compare scientifically.\n\
                      Frobenius norm, RMSE, quadrant analysis, V&V assessment.",
    },
    MenuItem {
        label: "Performance Monitor",
        shortcut: Some('p'),
        description: "Open a real-time CPU/RAM monitor in a separate console window.\n\
                      Per-core bars, sparkline history, memory gauge.",
    },
    MenuItem {
        label: "Benchmark Suite",
        shortcut: Some('b'),
        description: "Run all algorithms across multiple matrix sizes.\n\
                      Generates a performance comparison table and CSV report.",
    },
    MenuItem {
        label: "Matrix Viewer",
        shortcut: Some('v'),
        description: "Load and browse a matrix from a CSV file.\n\
                      Navigate rows/cols, highlight min/max, view statistics.",
    },
    MenuItem {
        label: "Computation History",
        shortcut: Some('y'),
        description: "View results from this session.\n\
                      Compare algorithms, re-run any previous configuration.",
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
enum AlgorithmChoice {
    Naive,
    Strassen,
    Winograd,
}

impl AlgorithmChoice {
    fn display_name(&self) -> &'static str {
        match self {
            AlgorithmChoice::Naive => "Naive (i-k-j)",
            AlgorithmChoice::Strassen => "Parallel Strassen + Tiled",
            AlgorithmChoice::Winograd => "Parallel Winograd + Tiled",
        }
    }
}

// ─── ETA Tracker (Exponential Moving Average) ───────────────────────────────

struct EtaTracker {
    start: Instant,
    last_fraction: f64,
    last_time: Instant,
    ema_rate: f64,
    alpha: f64,
    initialized: bool,
}

impl EtaTracker {
    fn new() -> Self {
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

    fn estimate_remaining(&self, fraction: f64) -> Option<f64> {
        if !self.initialized || self.ema_rate <= 1e-12 || fraction >= 1.0 {
            return None;
        }
        Some((1.0 - fraction) / self.ema_rate)
    }

    fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn format_eta(&self, fraction: f64) -> String {
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

enum ComputeResult {
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
}

#[derive(Clone)]
struct ComputeContext {
    algorithm_choice: AlgorithmChoice,
    algorithm_name: String,
    size: usize,
    gen_time_ms: Option<f64>,
    simd_level: crate::common::SimdLevel,
    // Diff-specific
    is_diff: bool,
    diff_alg1: Option<AlgorithmChoice>,
    diff_alg2: Option<AlgorithmChoice>,
}

struct ComputeTask {
    progress: ProgressHandle,
    eta: EtaTracker,
    receiver: mpsc::Receiver<ComputeResult>,
    context: ComputeContext,
    _join_handle: std::thread::JoinHandle<()>,
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
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: 20,
        }
    }

    fn push(&mut self, data: BenchmarkData, config: RunConfig) {
        let label = format!("{} {}×{}", data.algorithm, data.size, data.size);

        // Persist to CSV for cross-session history and monitor overlay
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
        let _ = crate::io::append_history(&record); // fire-and-forget

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
enum Screen {
    MainMenu,
    MultiplyMenu,
    SizeInput,
    InputMethodMenu,
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
enum Overlay {
    None,
    Help,
    About,
}

struct App {
    running: bool,
    screen: Screen,
    overlay: Overlay,
    sys_info: SystemInfo,

    // Menu state
    main_menu_idx: usize,
    mult_menu_idx: usize,
    input_method_idx: usize,

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
    viewer_filename: String,
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
    compute_task: Option<ComputeTask>,
}

impl App {
    fn new(sys_info: SystemInfo) -> Self {
        App {
            running: true,
            screen: Screen::MainMenu,
            overlay: Overlay::None,
            sys_info,
            main_menu_idx: 0,
            mult_menu_idx: 0,
            input_method_idx: 0,
            size_input: String::new(),
            chosen_size: 0,
            algorithm_choice: AlgorithmChoice::Strassen,
            manual_buffer: String::new(),
            manual_row: 0,
            manual_col: 0,
            manual_data: Vec::new(),
            manual_name: String::new(),
            csv_saved: false,
            matrix_saved: false,
            session_history: SessionHistory::new(),
            history_selected: 0,
            diff_size_input: String::new(),
            diff_alg1: AlgorithmChoice::Strassen,
            diff_alg2: AlgorithmChoice::Winograd,
            diff_select_idx: 0,
            diff_selecting_which: 1,
            current_theme: ThemeKind::Amber,
            terminal_scale: TerminalScale::Normal,
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
        }
    }

    fn theme(&self) -> ThemeColors {
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
            };
            let alg_choice = AlgorithmChoice::Strassen; // best-effort default for re-run
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
    // Overlays (Help/About) intercept all input
    if app.overlay != Overlay::None {
        match key {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => {
                app.overlay = Overlay::None;
            }
            _ => {}
        }
        return;
    }

    match &app.screen {
        Screen::MainMenu => handle_main_menu(app, key),
        Screen::MultiplyMenu => handle_multiply_menu(app, key),
        Screen::SizeInput => handle_size_input(app, key),
        Screen::InputMethodMenu => handle_input_method(app, key, terminal),
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
        Screen::Computing { .. } => {
            if matches!(key, KeyCode::Esc) {
                // Cancel: drop the task (thread continues but result is ignored)
                app.compute_task = None;
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
        KeyCode::Char('t') | KeyCode::Char('T') => {
            app.current_theme = app.current_theme.next();
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            app.terminal_scale = app.terminal_scale.next();
        }
        KeyCode::Char('q') | KeyCode::Esc => app.running = false,
        KeyCode::Char(c) => {
            // Shortcut keys from MENU_ITEMS
            for (i, item) in MENU_ITEMS.iter().enumerate() {
                if item.shortcut == Some(c) || item.shortcut == Some(c.to_ascii_uppercase()) {
                    app.main_menu_idx = i;
                    select_menu_item(app);
                    return;
                }
            }
        }
        _ => {}
    }
}

fn select_menu_item(app: &mut App) {
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
        6 => {
            // Computation History — only if non-empty
            if !app.session_history.is_empty() {
                app.history_selected = 0;
                app.screen = Screen::History;
            }
        }
        _ => {}
    }
}

fn handle_multiply_menu(app: &mut App, key: KeyCode) {
    let items = 4;
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
        KeyCode::Enter => match app.mult_menu_idx {
            0 => {
                app.algorithm_choice = AlgorithmChoice::Naive;
                app.size_input.clear();
                app.screen = Screen::SizeInput;
            }
            1 => {
                app.algorithm_choice = AlgorithmChoice::Strassen;
                app.size_input.clear();
                app.screen = Screen::SizeInput;
            }
            2 => {
                app.algorithm_choice = AlgorithmChoice::Winograd;
                app.size_input.clear();
                app.screen = Screen::SizeInput;
            }
            3 => {
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
            _ => {}
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
            0 => run_generation(app, terminal),
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
                    run_multiplication(app, a, b, None, terminal);
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
                    if crate::io::save_matrix_csv_with_metadata(&filename, &mat, Some(&meta)).is_ok() {
                        app.matrix_saved = true;
                    }
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
        let meta = app.viewer_loaded_metadata.as_ref();
        let _ = crate::io::save_matrix_csv_with_metadata(&path, mat, meta);
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
                _ => AlgorithmChoice::Winograd,
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
            run_generation(app, terminal);
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
fn sample_rss_bytes() -> u64 {
    let pid = match sysinfo::get_current_pid() {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let mut sys = SysinfoSystem::new();
    sys.refresh_process(pid);
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// Sample average CPU frequency across all logical cores in MHz.
fn sample_avg_freq_mhz() -> f64 {
    let mut sys = SysinfoSystem::new();
    sys.refresh_cpu();
    let cpus = sys.cpus();
    if cpus.is_empty() {
        return 0.0;
    }
    cpus.iter().map(|c| c.frequency() as f64).sum::<f64>() / cpus.len() as f64
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
    });

    app.screen = Screen::Computing {
        algorithm: format!("{alg_name}  {n}\u{00d7}{n}"),
    };
}

// ─── Background Task Completion ──────────────────────────────────────────────

fn check_compute_completion(app: &mut App) {
    let completed = if let Some(ref task) = app.compute_task {
        task.receiver.try_recv().ok()
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
                    computation_id: crate::io::generate_computation_id(),
                    machine_name: app.sys_info.hostname.clone(),
                };

                // Add to session history
                app.session_history.push(
                    data.clone(),
                    RunConfig {
                        algorithm: ctx.algorithm_choice,
                        size: n,
                    },
                );

                app.csv_saved = false;
                app.matrix_saved = false;
                app.screen = Screen::Results { data };
            }
            ComputeResult::Diff { result1, result2, time1_ms, time2_ms } => {
                let n = ctx.size;
                let alg1 = ctx.diff_alg1.unwrap_or(AlgorithmChoice::Naive);
                let alg2 = ctx.diff_alg2.unwrap_or(AlgorithmChoice::Strassen);

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
                };
                app.session_history.push(
                    history_data,
                    RunConfig { algorithm: if time1_ms <= time2_ms { alg1 } else { alg2 }, size: n },
                );

                app.screen = Screen::DiffResults { data };
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
fn format_duration(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.2} µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.2} ms", ms)
    } else {
        format!("{:.3} s", ms / 1000.0)
    }
}

/// Format megabytes into human-readable MB/GB.
fn format_memory(mb: u64) -> String {
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
        Screen::MultiplyMenu => render_multiply_menu(app, frame, area, &t),
        Screen::SizeInput => render_size_input(app, frame, area, &t),
        Screen::InputMethodMenu => render_input_method(app, frame, area, &t),
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
    }

    // Render overlay popups on top
    match app.overlay {
        Overlay::Help => render_help_popup(frame, area, &t),
        Overlay::About => render_about_popup(frame, area, &t),
        Overlay::None => {}
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
            Constraint::Length(1),      // footer
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
            let is_inactive = i == 6 && app.session_history.is_empty();

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
    let key_hint = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);
    let muted = Style::default().fg(t.text_muted);
    let line = Line::from(vec![
        Span::styled("  [\u{2191}\u{2193}]", key_hint),
        Span::styled(" Navigate", muted),
        Span::styled("   [Enter]", key_hint),
        Span::styled(" Select", muted),
        Span::styled("   [H]", key_hint),
        Span::styled(" Help", muted),
        Span::styled("   [A]", key_hint),
        Span::styled(" About", muted),
        Span::styled("   [T]", key_hint),
        Span::styled(format!(" {}", app.current_theme.display_name()), muted),
        Span::styled("   [S]", key_hint),
        Span::styled(format!(" {}", app.terminal_scale.label()), muted),
        Span::styled("   [Q]", key_hint),
        Span::styled(" Quit", muted),
    ]);

    let footer = Paragraph::new(line).style(Style::default().bg(t.surface));
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

    let items = vec![
        ("1", "Naive (no optimization)", "Simple i-k-j loop, single thread. Ground truth baseline."),
        ("2", "Full Parallel (Strassen)", "Strassen + Tiling + Rayon. Maximum performance."),
        ("3", "Winograd variant", "Strassen variant with fewer additions (15 vs 18). Same O(n^2.807)."),
        ("4", "Back", "Return to main menu"),
    ];

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

    let lines = vec![
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
        Line::from(Span::styled(
            "[Esc] Cancel",
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
            Constraint::Length(9),  // results + timing (horizontal split)
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

    // Left: parameters
    let params = vec![
        Line::from(""),
        kv_line("  Algorithm", &data.algorithm, t.text_bright, t.text_dim),
        kv_line("  SIMD     ", &data.simd_level, t.accent, t.text_dim),
        kv_line(
            "  Size     ",
            &format!("{} \u{00d7} {}", data.size, data.size),
            t.text, t.text_dim,
        ),
        kv_line("  Threads  ", &data.threads.to_string(), t.text, t.text_dim),
        kv_line(
            "  RAM      ",
            &format!("~{}", format_memory(data.peak_ram_mb)),
            t.text, t.text_dim,
        ),
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
fn fmt_sci(val: f64) -> String {
    if val.abs() < 1e-3 || val.abs() > 1e6 {
        format!("{:.4e}", val)
    } else {
        format!("{:.4}", val)
    }
}

/// Helper: key-value line for results panel.
fn kv_line(key: &str, value: &str, value_color: ratatui::style::Color, key_color: ratatui::style::Color) -> Line<'static> {
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

    // ── Entry list ──
    let items: Vec<ListItem> = history
        .entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let is_sel = i == selected;
            let prefix = if is_sel { "\u{25b6}  " } else { "   " };
            let line_text = format!(
                "{prefix}{:<30} {:>7.0} ms  {:>6.1} GFLOPS  {:<7}  {}T",
                entry.label,
                entry.data.compute_time_ms,
                entry.data.gflops,
                entry.data.simd_level,
                entry.data.threads,
            );
            let style = if is_sel {
                Style::default()
                    .fg(t.text_bright)
                    .bg(t.surface)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(t.text_muted).bg(t.bg)
            };
            ListItem::new(Line::from(Span::styled(line_text, style)))
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
        ("2", "Parallel Strassen", "Strassen + Tiling + Rayon. O(n^2.807).", AlgorithmChoice::Strassen),
        ("3", "Winograd variant", "Strassen variant with fewer additions (15 vs 18).", AlgorithmChoice::Winograd),
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

// ─── Shared Footer ──────────────────────────────────────────────────────────

fn render_nav_footer(frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled(
            "  [\u{2191}\u{2193}]",
            key_hint,
        ),
        Span::styled(" Navigate  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Enter]", key_hint),
        Span::styled(" Select  ", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc]", key_hint),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        area,
    );
}
