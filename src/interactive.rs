// interactive.rs — Full ratatui TUI: menus, states, user interaction.
//
// Architecture: single event loop with state machine.
// App struct holds current screen state. Each state has its own render + input handler.
// crossterm alternate screen ensures the TUI doesn't pollute the compilation terminal.
//
// Design: Valve/Half-Life industrial amber HUD, Apple clean layout, Android material flow.

use std::io;
use std::time::Instant;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Terminal;

use crate::algorithms;
use crate::common::STRASSEN_THRESHOLD;
use crate::io as theme;
use crate::matrix::Matrix;
use crate::system::SystemInfo;

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
    Results {
        algorithm: String,
        size: usize,
        gen_time_ms: Option<f64>,
        compute_time_ms: f64,
        result_matrix: Option<Matrix>,
    },
    ComingSoon(String), // placeholder for unimplemented features
}

struct App {
    running: bool,
    screen: Screen,
    sys_info: SystemInfo,

    // Menu state
    main_menu_idx: usize,
    mult_menu_idx: usize,
    input_method_idx: usize,

    // User input buffers
    size_input: String,
    chosen_size: usize,
    naive_mode: bool, // true = naive, false = parallel+strassen

    // Manual matrix input
    manual_buffer: String,
    manual_row: usize,
    manual_col: usize,
    manual_data: Vec<f64>,
    manual_name: String,
}

impl App {
    fn new(sys_info: SystemInfo) -> Self {
        App {
            running: true,
            screen: Screen::MainMenu,
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
        }
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
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(sys_info);

    // Event loop
    while app.running {
        terminal.draw(|f| render(&app, f))?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    handle_input(&mut app, key.code, &mut terminal);
                }
            }
        }
    }

    // Restore terminal
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
    match &app.screen {
        Screen::MainMenu => handle_main_menu(app, key),
        Screen::MultiplyMenu => handle_multiply_menu(app, key),
        Screen::SizeInput => handle_size_input(app, key),
        Screen::InputMethodMenu => handle_input_method(app, key, terminal),
        Screen::ManualInputA { .. } => handle_manual_input_a(app, key),
        Screen::ManualInputB { .. } => handle_manual_input_b(app, key, terminal),
        Screen::Results { .. } => handle_results(app, key),
        Screen::ComingSoon(_) => {
            if matches!(key, KeyCode::Esc | KeyCode::Enter) {
                app.screen = Screen::MainMenu;
                app.main_menu_idx = 0;
            }
        }
        Screen::Computing { .. } => {} // non-interactive, handled in run step
    }
}

fn handle_main_menu(app: &mut App, key: KeyCode) {
    let items = 4; // Multiply, Compare, Scientific, Exit
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
        KeyCode::Enter => match app.main_menu_idx {
            0 => {
                app.screen = Screen::MultiplyMenu;
                app.mult_menu_idx = 0;
            }
            1 => app.screen = Screen::ComingSoon("Matrix Comparison".into()),
            2 => app.screen = Screen::ComingSoon("Scientific Computing".into()),
            3 => app.running = false,
            _ => {}
        },
        KeyCode::Esc => app.running = false,
        _ => {}
    }
}

fn handle_multiply_menu(app: &mut App, key: KeyCode) {
    let items = 3; // Naive, Parallel, Back
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
        KeyCode::Backspace => { app.size_input.pop(); }
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
    let items = 3; // Generate, Manual, Back
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
            0 => run_generation(app, terminal), // Generate random
            1 => {
                // Manual input — start with matrix A
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
        KeyCode::Backspace => { app.manual_buffer.pop(); }
        KeyCode::Enter => {
            // Parse numbers from the buffer (space-separated on one line = one row)
            let nums: Vec<f64> = app.manual_buffer
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

            // If matrix A is complete, move to matrix B
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
        KeyCode::Backspace => { app.manual_buffer.pop(); }
        KeyCode::Enter => {
            let nums: Vec<f64> = app.manual_buffer
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

            // If matrix B is complete, run multiplication
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
        KeyCode::Esc | KeyCode::Enter => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// ─── Computation ────────────────────────────────────────────────────────────

fn run_generation(
    app: &mut App,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) {
    let n = app.chosen_size;

    // Show computing screen
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
    let alg_name = if app.naive_mode {
        "Naive (i-k-j)"
    } else {
        "Parallel Strassen + Tiled"
    };

    app.screen = Screen::Computing {
        algorithm: format!("{}  {}x{}", alg_name, app.chosen_size, app.chosen_size),
    };
    terminal.draw(|f| render(app, f)).ok();

    let start = Instant::now();
    let result = if app.naive_mode {
        algorithms::multiply_naive(&a, &b)
    } else {
        let (r, _, _) = algorithms::multiply_strassen_padded(
            &a,
            &b,
            STRASSEN_THRESHOLD,
            app.sys_info.simd_level,
        );
        r
    };
    let compute_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Keep result for small matrices (display), drop for large ones
    let show_result = if app.chosen_size <= 8 {
        Some(result)
    } else {
        drop(result);
        None
    };

    app.screen = Screen::Results {
        algorithm: alg_name.into(),
        size: app.chosen_size,
        gen_time_ms,
        compute_time_ms: compute_ms,
        result_matrix: show_result,
    };
}

// ─── Rendering ──────────────────────────────────────────────────────────────

fn render(app: &App, frame: &mut ratatui::Frame) {
    let area = frame.size();

    // Clear background
    frame.render_widget(Clear, area);
    let bg_block = Block::default().style(theme::style_default());
    frame.render_widget(bg_block, area);

    match &app.screen {
        Screen::MainMenu => render_main_menu(app, frame, area),
        Screen::MultiplyMenu => render_multiply_menu(app, frame, area),
        Screen::SizeInput => render_size_input(app, frame, area),
        Screen::InputMethodMenu => render_input_method(app, frame, area),
        Screen::ManualInputA { name } => render_manual_input(app, frame, area, name, false),
        Screen::ManualInputB { name, .. } => render_manual_input(app, frame, area, name, true),
        Screen::Computing { algorithm } => render_computing(algorithm, frame, area),
        Screen::Results { algorithm, size, gen_time_ms, compute_time_ms, result_matrix } => {
            render_results(algorithm, *size, *gen_time_ms, compute_time_ms, result_matrix, frame, area);
        }
        Screen::ComingSoon(label) => render_coming_soon(label, frame, area),
    }
}

fn render_main_menu(app: &App, frame: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),   // logo
            Constraint::Length(12),  // system info
            Constraint::Min(8),     // menu
            Constraint::Length(2),  // footer
        ])
        .split(area);

    // ── Logo ──
    let logo = vec![
        Line::from(Span::styled(
            "  ███████╗██╗     ██╗   ██╗███████╗████████╗",
            theme::style_accent(),
        )),
        Line::from(Span::styled(
            "  ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝",
            theme::style_accent(),
        )),
        Line::from(Span::styled(
            "  █████╗  ██║     ██║   ██║███████╗   ██║   ",
            theme::style_title(),
        )),
        Line::from(Span::styled(
            "  ██╔══╝  ██║     ██║   ██║╚════██║   ██║   ",
            theme::style_accent(),
        )),
        Line::from(Span::styled(
            "  ██║     ███████╗╚██████╔╝███████║   ██║   ",
            theme::style_accent(),
        )),
        Line::from(Span::styled(
            "  ╚═╝     ╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ",
            theme::style_muted(),
        )),
        Line::from(Span::styled(
            "  High-Performance Matrix Engine  v0.1.0",
            theme::style_muted(),
        )),
    ];
    let logo_widget = Paragraph::new(logo).alignment(Alignment::Left);
    frame.render_widget(logo_widget, chunks[0]);

    // ── System Info Panel ──
    render_system_panel(&app.sys_info, frame, chunks[1]);

    // ── Menu ──
    let menu_items = vec![
        ("1", "Matrix Multiplication", "High-speed parallel matrix multiplication"),
        ("2", "Matrix Comparison", "Coming soon"),
        ("3", "Scientific Computing", "Coming soon"),
        ("4", "Exit", "Quit the application"),
    ];

    let mut menu_lines: Vec<Line> = Vec::new();
    menu_lines.push(Line::from(""));
    for (i, (key, label, desc)) in menu_items.iter().enumerate() {
        let is_selected = i == app.main_menu_idx;
        let arrow = if is_selected { " ▸ " } else { "   " };
        let style = if is_selected {
            theme::style_selected()
        } else {
            theme::style_default()
        };
        let dim_style = if is_selected {
            theme::style_selected()
        } else {
            theme::style_muted()
        };

        menu_lines.push(Line::from(vec![
            Span::styled(arrow, style),
            Span::styled(format!("[{key}] "), theme::style_key_hint()),
            Span::styled(format!("{label:<28}"), style),
            Span::styled(format!("  {desc}"), dim_style),
        ]));
        menu_lines.push(Line::from(""));
    }

    let menu_block = Block::default()
        .title(Span::styled(" MAIN MENU ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());

    let menu_widget = Paragraph::new(menu_lines)
        .block(menu_block)
        .wrap(Wrap { trim: false });
    frame.render_widget(menu_widget, chunks[2]);

    // ── Footer ──
    let footer = Line::from(vec![
        Span::styled(" ↑↓ ", theme::style_key_hint()),
        Span::styled("Navigate  ", theme::style_muted()),
        Span::styled(" Enter ", theme::style_key_hint()),
        Span::styled("Select  ", theme::style_muted()),
        Span::styled(" Esc ", theme::style_key_hint()),
        Span::styled("Exit", theme::style_muted()),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[3]);
}

fn render_system_panel(sys: &SystemInfo, frame: &mut ratatui::Frame, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left: CPU + System
    let cpu_freq = if sys.base_frequency_mhz > 0 {
        format!("{} MHz", sys.base_frequency_mhz)
    } else {
        "N/A".into()
    };

    let cache_info = format!(
        "L1: {}KB  L2: {}KB  L3: {}KB",
        sys.l1_cache_kb.map_or("?".into(), |v| v.to_string()),
        sys.l2_cache_kb.map_or("?".into(), |v| v.to_string()),
        sys.l3_cache_kb.map_or("?".into(), |v| v.to_string()),
    );

    let left_lines = vec![
        Line::from(vec![
            Span::styled(" PC: ", theme::style_muted()),
            Span::styled(&sys.hostname, theme::style_info()),
        ]),
        Line::from(vec![
            Span::styled(" CPU: ", theme::style_muted()),
            Span::styled(&sys.cpu_brand, theme::style_default()),
        ]),
        Line::from(vec![
            Span::styled(" Arch: ", theme::style_muted()),
            Span::styled(&sys.cpu_arch, theme::style_info()),
        ]),
        Line::from(vec![
            Span::styled(" Freq: ", theme::style_muted()),
            Span::styled(&cpu_freq, theme::style_default()),
            Span::styled("  Cores: ", theme::style_muted()),
            Span::styled(
                format!("{}C/{}T", sys.physical_cores, sys.logical_cores),
                theme::style_default(),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Cache: ", theme::style_muted()),
            Span::styled(&cache_info, theme::style_default()),
        ]),
        Line::from(vec![
            Span::styled(" RAM: ", theme::style_muted()),
            Span::styled(
                format!(
                    "{} / {}",
                    theme::format_memory_mb(sys.available_ram_mb),
                    theme::format_memory_mb(sys.total_ram_mb)
                ),
                theme::style_default(),
            ),
            Span::styled(" available", theme::style_muted()),
        ]),
    ];

    let left_block = Block::default()
        .title(Span::styled(" SYSTEM ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(left_lines).block(left_block), cols[0]);

    // Right: SIMD capabilities
    let simd_items = vec![
        ("SSE4.2", sys.supports_sse42),
        ("AVX2", sys.supports_avx2),
        ("AVX-512", sys.supports_avx512),
        ("Rayon (multi-thread)", true),
        ("Tiling (cache-blocked)", true),
        ("Strassen O(n^2.807)", true),
    ];

    let mut right_lines: Vec<Line> = Vec::new();
    right_lines.push(Line::from(vec![
        Span::styled(" Best SIMD: ", theme::style_muted()),
        Span::styled(sys.simd_level.display_name(), theme::style_accent()),
    ]));
    right_lines.push(Line::from(""));

    for (name, supported) in &simd_items {
        let (icon, style) = if *supported {
            ("  ● ", theme::style_success())
        } else {
            ("  ○ ", theme::style_danger())
        };
        right_lines.push(Line::from(vec![
            Span::styled(icon, style),
            Span::styled(*name, style),
        ]));
    }

    let right_block = Block::default()
        .title(Span::styled(" TECHNOLOGIES ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(right_lines).block(right_block), cols[1]);
}

fn render_multiply_menu(app: &App, frame: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    // Title
    let title = Paragraph::new(Line::from(Span::styled(
        "  MATRIX MULTIPLICATION",
        theme::style_title(),
    )));
    frame.render_widget(title, chunks[0]);

    // Menu
    let items = vec![
        ("1", "Naive (no optimization)", "Simple i-k-j loop, single thread. Ground truth baseline."),
        ("2", "Full Parallel", "Strassen + Tiling + Rayon + best SIMD. Maximum performance."),
        ("3", "Back", "Return to main menu"),
    ];

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, (key, label, desc)) in items.iter().enumerate() {
        let selected = i == app.mult_menu_idx;
        let arrow = if selected { " ▸ " } else { "   " };
        let style = if selected { theme::style_selected() } else { theme::style_default() };
        let dim = if selected { theme::style_selected() } else { theme::style_muted() };

        lines.push(Line::from(vec![
            Span::styled(arrow, style),
            Span::styled(format!("[{key}] "), theme::style_key_hint()),
            Span::styled(format!("{label:<30}"), style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      ", theme::style_default()),
            Span::styled(*desc, dim),
        ]));
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(Span::styled(" SELECT ALGORITHM ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    render_nav_footer(frame, chunks[2]);
}

fn render_size_input(app: &App, frame: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(2),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(Span::styled(
        "  MATRIX SIZE",
        theme::style_title(),
    )));
    frame.render_widget(title, chunks[0]);

    let est_ram = if let Ok(n) = app.size_input.parse::<usize>() {
        if n > 0 {
            let mb = SystemInfo::estimate_peak_ram_mb(n);
            format!("Estimated RAM: ~{}", theme::format_memory_mb(mb))
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Both matrices will be square (N × N).",
            theme::style_muted(),
        )),
        Line::from(Span::styled(
            "  Enter the dimension N (1-10000):",
            theme::style_default(),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  N = ", theme::style_accent()),
            Span::styled(
                format!("{}_", &app.size_input),
                theme::style_info(),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {est_ram}"),
            theme::style_muted(),
        )),
    ];

    let block = Block::default()
        .title(Span::styled(" DIMENSION ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    let footer = Line::from(vec![
        Span::styled(" Enter ", theme::style_key_hint()),
        Span::styled("Confirm  ", theme::style_muted()),
        Span::styled(" Esc ", theme::style_key_hint()),
        Span::styled("Back", theme::style_muted()),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[2]);
}

fn render_input_method(app: &App, frame: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(Span::styled(
        format!("  DATA INPUT FOR {}×{} MATRICES", app.chosen_size, app.chosen_size),
        theme::style_title(),
    )));
    frame.render_widget(title, chunks[0]);

    let items = vec![
        ("1", "Generate Random", "Fast random fill [-10, 10]. Best for benchmarking."),
        ("2", "Manual Input", "Enter elements row by row. Best for small matrices."),
        ("3", "Back", "Change matrix size"),
    ];

    let mut lines: Vec<Line> = vec![Line::from("")];
    for (i, (key, label, desc)) in items.iter().enumerate() {
        let selected = i == app.input_method_idx;
        let arrow = if selected { " ▸ " } else { "   " };
        let style = if selected { theme::style_selected() } else { theme::style_default() };
        let dim = if selected { theme::style_selected() } else { theme::style_muted() };

        lines.push(Line::from(vec![
            Span::styled(arrow, style),
            Span::styled(format!("[{key}] "), theme::style_key_hint()),
            Span::styled(format!("{label:<25}"), style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      ", theme::style_default()),
            Span::styled(*desc, dim),
        ]));
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(Span::styled(" INPUT METHOD ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(lines).block(block), chunks[1]);

    render_nav_footer(frame, chunks[2]);
}

fn render_manual_input(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    name: &str,
    _is_b: bool,
) {
    let n = app.chosen_size;
    let filled = app.manual_data.len();
    let total = n * n;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(Span::styled(
        format!("  MATRIX \"{}\" — {n}×{n}  ({filled}/{total} elements)", name),
        theme::style_title(),
    )));
    frame.render_widget(title, chunks[0]);

    // Show current matrix state with brackets
    let display_rows = n.min(12); // limit display for large matrices
    let display_cols = n.min(12);
    let mut matrix_lines: Vec<Line> = Vec::new();

    matrix_lines.push(Line::from(Span::styled(
        "  Enter numbers separated by spaces, press Enter for each row:",
        theme::style_muted(),
    )));
    matrix_lines.push(Line::from(""));

    for r in 0..display_rows {
        let mut spans = vec![Span::styled("  │ ", theme::style_accent())];
        for c in 0..display_cols {
            let idx = r * n + c;
            if idx < filled {
                spans.push(Span::styled(
                    format!("{:>8.2} ", app.manual_data[idx]),
                    theme::style_default(),
                ));
            } else if idx == filled {
                spans.push(Span::styled("    _    ", theme::style_info()));
            } else {
                spans.push(Span::styled("    ·    ", theme::style_muted()));
            }
        }
        if n > display_cols {
            spans.push(Span::styled(" ...", theme::style_muted()));
        }
        spans.push(Span::styled(" │", theme::style_accent()));
        matrix_lines.push(Line::from(spans));
    }
    if n > display_rows {
        matrix_lines.push(Line::from(Span::styled(
            format!("  │ ... ({} more rows) ... │", n - display_rows),
            theme::style_muted(),
        )));
    }

    let block = Block::default()
        .title(Span::styled(format!(" MATRIX {name} "), theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(matrix_lines).block(block), chunks[1]);

    // Input line
    let input_line = Line::from(vec![
        Span::styled("  > ", theme::style_accent()),
        Span::styled(
            format!("{}_", &app.manual_buffer),
            theme::style_info(),
        ),
    ]);
    frame.render_widget(Paragraph::new(input_line), chunks[2]);

    let footer = Line::from(vec![
        Span::styled(" Enter ", theme::style_key_hint()),
        Span::styled("Submit row  ", theme::style_muted()),
        Span::styled(" Esc ", theme::style_key_hint()),
        Span::styled("Cancel", theme::style_muted()),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[3]);
}

fn render_computing(algorithm: &str, frame: &mut ratatui::Frame, area: Rect) {
    let block = Block::default()
        .title(Span::styled(" COMPUTING ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());

    let lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "  ▓▓▓▓░░░░░░░░░░",
            theme::style_accent(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {algorithm}"),
            theme::style_info(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Please wait...",
            theme::style_muted(),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block).alignment(Alignment::Center), area);
}

fn render_results(
    algorithm: &str,
    size: usize,
    gen_time_ms: Option<f64>,
    compute_time_ms: &f64,
    result_matrix: &Option<Matrix>,
    frame: &mut ratatui::Frame,
    area: Rect,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Length(10),
            Constraint::Length(2),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(Span::styled(
        "  MULTIPLICATION COMPLETE",
        theme::style_title(),
    )));
    frame.render_widget(title, chunks[0]);

    // Timing results
    let gflops = crate::common::MultiplicationResult::calculate_gflops(
        size, size, size, *compute_time_ms,
    );

    let mut result_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Algorithm:  ", theme::style_muted()),
            Span::styled(algorithm, theme::style_accent()),
        ]),
        Line::from(vec![
            Span::styled("  Size:       ", theme::style_muted()),
            Span::styled(format!("{size}×{size}"), theme::style_default()),
        ]),
    ];

    if let Some(gen_ms) = gen_time_ms {
        result_lines.push(Line::from(vec![
            Span::styled("  Generation: ", theme::style_muted()),
            Span::styled(theme::format_duration_ms(gen_ms), theme::style_info()),
        ]));
    }

    result_lines.push(Line::from(vec![
        Span::styled("  Compute:    ", theme::style_muted()),
        Span::styled(
            theme::format_duration_ms(*compute_time_ms),
            theme::style_success(),
        ),
    ]));
    result_lines.push(Line::from(vec![
        Span::styled("  GFLOPS:     ", theme::style_muted()),
        Span::styled(format!("{gflops:.2}"), theme::style_accent()),
    ]));

    let block = Block::default()
        .title(Span::styled(" RESULTS ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(result_lines).block(block), chunks[1]);

    // Result matrix preview (for small matrices)
    if let Some(mat) = result_matrix {
        let mut mat_lines: Vec<Line> = vec![
            Line::from(Span::styled(
                "  Result matrix C = A × B:",
                theme::style_muted(),
            )),
        ];
        let show = mat.rows().min(8);
        for i in 0..show {
            let mut spans = vec![Span::styled("  │", theme::style_accent())];
            for j in 0..mat.cols().min(8) {
                spans.push(Span::styled(
                    format!("{:>10.4}", mat.get(i, j)),
                    theme::style_default(),
                ));
            }
            spans.push(Span::styled(" │", theme::style_accent()));
            mat_lines.push(Line::from(spans));
        }

        let mat_block = Block::default()
            .title(Span::styled(" OUTPUT MATRIX ", theme::style_title()))
            .borders(Borders::ALL)
            .border_style(theme::style_muted())
            .style(theme::style_default());
        frame.render_widget(Paragraph::new(mat_lines).block(mat_block), chunks[2]);
    } else {
        let info = Paragraph::new(Line::from(Span::styled(
            "  Matrix too large to display (size > 8). Result computed successfully.",
            theme::style_muted(),
        )))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::style_muted())
                .style(theme::style_default()),
        );
        frame.render_widget(info, chunks[2]);
    }

    let footer = Line::from(vec![
        Span::styled(" Enter/Esc ", theme::style_key_hint()),
        Span::styled("Return to main menu", theme::style_muted()),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[3]);
}

fn render_coming_soon(label: &str, frame: &mut ratatui::Frame, area: Rect) {
    let block = Block::default()
        .title(Span::styled(format!(" {label} "), theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());

    let lines = vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "  ⚠  Coming Soon",
            theme::style_accent(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  This feature is under development.",
            theme::style_muted(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Press Enter or Esc to return.",
            theme::style_muted(),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block).alignment(Alignment::Center), area);
}

fn render_nav_footer(frame: &mut ratatui::Frame, area: Rect) {
    let footer = Line::from(vec![
        Span::styled(" ↑↓ ", theme::style_key_hint()),
        Span::styled("Navigate  ", theme::style_muted()),
        Span::styled(" Enter ", theme::style_key_hint()),
        Span::styled("Select  ", theme::style_muted()),
        Span::styled(" Esc ", theme::style_key_hint()),
        Span::styled("Back", theme::style_muted()),
    ]);
    frame.render_widget(Paragraph::new(footer), area);
}
