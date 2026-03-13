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
use crate::common::{MultiplicationResult, Theme, ThemeColors, ThemeKind, STRASSEN_THRESHOLD};
use crate::matrix::Matrix;
use crate::system::SystemInfo;

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
        label: "Matrix Comparison",
        shortcut: Some('c'),
        description: "Compare two matrices element-wise with configurable epsilon.\n\
                      Reports max difference, RMS error, and match percentage.",
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
}

// ─── Session History ────────────────────────────────────────────────────────

/// Configuration needed to re-run a computation with the same parameters.
#[derive(Clone)]
struct RunConfig {
    naive_mode: bool,
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
    ComingSoon(String),
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
    naive_mode: bool,

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

    // Theme
    current_theme: ThemeKind,
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
            naive_mode: false,
            manual_buffer: String::new(),
            manual_row: 0,
            manual_col: 0,
            manual_data: Vec::new(),
            manual_name: String::new(),
            csv_saved: false,
            matrix_saved: false,
            session_history: SessionHistory::new(),
            history_selected: 0,
            current_theme: ThemeKind::Amber,
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

fn run_tui(sys_info: SystemInfo) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(sys_info);

    while app.running {
        terminal.draw(|f| render(&app, f))?;

        if event::poll(std::time::Duration::from_millis(100))? {
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
        Screen::ComingSoon(_) => {
            if matches!(key, KeyCode::Esc | KeyCode::Enter) {
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::Computing { .. } => {}
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
            // Matrix Comparison
            app.screen = Screen::ComingSoon("Matrix Comparison".into());
        }
        2 => {
            // Performance Monitor — launch in separate window
            crate::monitor::spawn_monitor_window();
        }
        3 => {
            // Benchmark Suite
            app.screen = Screen::ComingSoon("Benchmark Suite".into());
        }
        4 => {
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
    let items = 3;
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
                app.naive_mode = true;
                app.size_input.clear();
                app.screen = Screen::SizeInput;
            }
            1 => {
                app.naive_mode = false;
                app.size_input.clear();
                app.screen = Screen::SizeInput;
            }
            2 => {
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
                        (
                            format!("flust_result_{}x{}.csv", data.size, data.size),
                            mat.clone(),
                        )
                    })
                } else {
                    None
                };
                if let Some((filename, mat)) = save_info {
                    if crate::io::save_matrix_csv(&filename, &mat).is_ok() {
                        app.matrix_saved = true;
                    }
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
            app.naive_mode = config.naive_mode;
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

// ─── Computation ────────────────────────────────────────────────────────────

fn run_generation(app: &mut App, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    let n = app.chosen_size;

    app.screen = Screen::Computing {
        algorithm: "Generating matrices...".into(),
    };
    terminal.draw(|f| render(app, f)).ok();

    let gen_start = Instant::now();
    let a = Matrix::random(n, n, None).unwrap();
    let b = Matrix::random(n, n, None).unwrap();
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

    run_multiplication(app, a, b, Some(gen_ms), terminal);
}

fn run_multiplication(
    app: &mut App,
    a: Matrix,
    b: Matrix,
    gen_time_ms: Option<f64>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let n = app.chosen_size;
    let alg_name = if app.naive_mode {
        "Naive (i-k-j)"
    } else {
        "Parallel Strassen + Tiled"
    };

    app.screen = Screen::Computing {
        algorithm: format!("{alg_name}  {n}x{n}"),
    };
    terminal.draw(|f| render(app, f)).ok();

    let start = Instant::now();
    let (result, padding_ms, unpadding_ms) = if app.naive_mode {
        let r = algorithms::multiply_naive(&a, &b);
        (r, 0.0, 0.0)
    } else {
        algorithms::multiply_strassen_padded(&a, &b, STRASSEN_THRESHOLD, app.sys_info.simd_level)
    };
    let total_compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pure_compute_ms = (total_compute_ms - padding_ms - unpadding_ms).max(0.001);

    let threads = rayon::current_num_threads();
    let peak_ram = SystemInfo::estimate_peak_ram_mb(n);
    let gflops = MultiplicationResult::calculate_gflops(n, n, n, pure_compute_ms);

    let show_result = if n <= 16 {
        Some(result)
    } else {
        drop(result);
        None
    };

    let data = BenchmarkData {
        algorithm: alg_name.into(),
        size: n,
        gen_time_ms,
        padding_time_ms: padding_ms,
        compute_time_ms: pure_compute_ms,
        unpadding_time_ms: unpadding_ms,
        gflops,
        simd_level: app.sys_info.simd_level.display_name().to_string(),
        threads,
        peak_ram_mb: peak_ram,
        result_matrix: show_result,
    };

    app.csv_saved = false;
    app.matrix_saved = false;

    // Push to session history
    let config = RunConfig {
        naive_mode: app.naive_mode,
        size: n,
    };
    app.session_history.push(data.clone(), config);

    app.screen = Screen::Results { data };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Theoretical peak GFLOPS (rough estimate for the efficiency bar).
fn theoretical_peak_gflops(sys: &SystemInfo) -> f64 {
    let freq_ghz = sys.base_frequency_mhz as f64 / 1000.0;
    if freq_ghz <= 0.0 {
        return 0.0;
    }
    let flops_per_cycle: f64 = match sys.simd_level {
        crate::common::SimdLevel::Avx512 => 32.0,
        crate::common::SimdLevel::Avx2 => 16.0,
        crate::common::SimdLevel::Sse42 => 4.0,
        crate::common::SimdLevel::Scalar => 2.0,
    };
    sys.physical_cores as f64 * freq_ghz * flops_per_cycle
}

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
        Screen::Computing { algorithm } => render_computing(algorithm, frame, area, &t),
        Screen::Results { data } => render_results(app, data, frame, area, &t),
        Screen::History => render_history_screen(app, frame, area, &t),
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

    render_banner(&app.sys_info, frame, chunks[0], t);
    render_menu(app, frame, chunks[1], t);
    render_hint(app, frame, chunks[2], t);
    render_main_footer(app, frame, chunks[3], t);
}

fn render_banner(sys: &SystemInfo, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let logo_lines = LOGO_LINES.len();
    let total_content = logo_lines + 4;
    let top_pad = (area.height as usize).saturating_sub(total_content) / 2;

    let mut lines: Vec<Line> = Vec::new();
    for _ in 0..top_pad {
        lines.push(Line::from(""));
    }
    for logo_text in &LOGO_LINES {
        lines.push(Line::from(Span::styled(
            *logo_text,
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )));
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
            let is_inactive = i == 4 && app.session_history.is_empty();

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
        ("2", "Full Parallel", "Strassen + Tiling + Rayon + best SIMD. Maximum performance."),
        ("3", "Back", "Return to main menu"),
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

fn render_computing(algorithm: &str, frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let block = Block::default()
        .title(Span::styled(" COMPUTING ", Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.bg));

    let lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "  \u{2593}\u{2593}\u{2593}\u{2593}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}",
            Style::default().fg(t.accent),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {algorithm}"),
            Style::default().fg(t.ok),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Please wait...",
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

    // ── Performance section ──
    let peak_gflops = theoretical_peak_gflops(&app.sys_info);
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
                    format!("  {:.0}% theor.", efficiency),
                    Style::default().fg(t.text_muted),
                )
            } else {
                Span::styled("", Style::default())
            },
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

    let key_hint = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::BOLD);
    let footer = Line::from(vec![
        Span::styled("  [S]", key_hint),
        Span::styled(
            format!(" Save CSV{csv_status}"),
            Style::default().fg(t.text_muted),
        ),
        Span::styled(&mat_hint, Style::default().fg(t.text_muted)),
        Span::styled("   [Enter/Esc]", key_hint),
        Span::styled(" Back to menu", Style::default().fg(t.text_muted)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[4],
    );
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
