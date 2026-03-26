// ═══════════════════════════════════════════════════════════════════════════════
//  ECONOMICS MODULE — TUI WIZARD & RESULTS
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::{mpsc, Arc, Mutex};

use crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::common::{ProgressHandle, ThemeColors};
use crate::economics::{self, LeontiefConfig, LeontiefResult};
use crate::interactive::{
    App, ComputeContext, ComputeResult, ComputeTask, EtaTracker, Overlay, Screen,
    format_duration, kv_line, GEAR_FRAMES, GEAR_SINGLE,
};

// ─── Field Labels ───────────────────────────────────────────────────────────

const CONFIG_LABELS: &[&str] = &[
    "Sectors (N)",
    "Sparsity (0–1)",
    "Convergence tol",
    "Max iterations",
];

const SHOCK_LABELS: &[&str] = &[
    "Shock sector",
    "Shock magnitude",
];

// ─── Input Handlers ─────────────────────────────────────────────────────────

pub fn handle_econ_config(app: &mut App, key: KeyCode) {
    let n_fields = CONFIG_LABELS.len();
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.econ_active_field = (app.econ_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.econ_active_field = if app.econ_active_field == 0 {
                n_fields - 1
            } else {
                app.econ_active_field - 1
            };
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' => {
            active_config_field_mut(app).push(c);
        }
        KeyCode::Backspace => {
            active_config_field_mut(app).pop();
        }
        KeyCode::Enter => {
            app.screen = Screen::EconShockSelect;
            app.econ_active_field = 0;
        }
        KeyCode::Esc => {
            app.screen = Screen::CategoryMenu {
                category: crate::interactive::MenuCategory::Economics,
            };
            app.category_menu_idx = 0;
        }
        _ => {}
    }
}

fn active_config_field_mut(app: &mut App) -> &mut String {
    match app.econ_active_field {
        0 => &mut app.econ_sectors_input,
        1 => &mut app.econ_sparsity_input,
        2 => &mut app.econ_tol_input,
        3 => &mut app.econ_max_iter_input,
        _ => &mut app.econ_sectors_input,
    }
}

pub fn handle_econ_shock(app: &mut App, key: KeyCode) {
    let n_fields = SHOCK_LABELS.len();
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.econ_active_field = (app.econ_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.econ_active_field = if app.econ_active_field == 0 {
                n_fields - 1
            } else {
                app.econ_active_field - 1
            };
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' => {
            active_shock_field_mut(app).push(c);
        }
        KeyCode::Backspace => {
            active_shock_field_mut(app).pop();
        }
        KeyCode::Enter => {
            app.screen = Screen::EconConfirm;
        }
        KeyCode::Esc => {
            app.screen = Screen::EconConfig;
            app.econ_active_field = 0;
        }
        _ => {}
    }
}

fn active_shock_field_mut(app: &mut App) -> &mut String {
    match app.econ_active_field {
        0 => &mut app.econ_shock_sector_input,
        1 => &mut app.econ_shock_mag_input,
        _ => &mut app.econ_shock_sector_input,
    }
}

pub fn handle_econ_confirm(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => {
            if let Some(config) = build_econ_config(app) {
                launch_econ_simulation(app, config);
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::EconShockSelect;
            app.econ_active_field = 0;
        }
        _ => {}
    }
}

pub fn handle_econ_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if !app.econ_csv_saved {
                if let Screen::EconResults { ref result } = app.screen {
                    let path = format!(
                        "flust_leontief_{}.csv",
                        crate::io::timestamp_now().replace(':', "-")
                    );
                    if crate::economics_export::export_leontief_csv(result, &path).is_ok() {
                        app.econ_csv_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('g') | KeyCode::Char('G') => {
            app.overlay = Overlay::EconConvergenceGraph;
            app.econ_overlay_scroll = 0;
        }
        KeyCode::Char('b') | KeyCode::Char('B') => {
            app.overlay = Overlay::EconSectorBars;
            app.econ_overlay_scroll = 0;
        }
        KeyCode::Char('d') | KeyCode::Char('D') => {
            app.overlay = Overlay::EconDashboard;
            app.econ_dashboard_tab = 0;
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.screen = Screen::EconConfirm;
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// ─── Config Builder ─────────────────────────────────────────────────────────

fn build_econ_config(app: &App) -> Option<LeontiefConfig> {
    let sectors: usize = app.econ_sectors_input.parse::<f64>().ok()? as usize;
    let sparsity: f64 = app.econ_sparsity_input.parse().ok()?;
    let tolerance: f64 = app.econ_tol_input.parse().ok()?;
    let max_iterations: usize = app.econ_max_iter_input.parse::<f64>().ok()? as usize;
    let shock_sector: usize = app.econ_shock_sector_input.parse::<f64>().ok()? as usize;
    let shock_magnitude: f64 = app.econ_shock_mag_input.parse().ok()?;

    if sectors < 2 || sectors > 10000 {
        return None;
    }
    if !(0.0..1.0).contains(&sparsity) {
        return None;
    }
    if tolerance <= 0.0 {
        return None;
    }
    if max_iterations < 1 {
        return None;
    }

    Some(LeontiefConfig {
        sectors,
        sparsity,
        spectral_target: 0.85, // hard-coded safe value
        tolerance,
        max_iterations,
        shock_sector: shock_sector.min(sectors.saturating_sub(1)),
        shock_magnitude,
    })
}

// ─── Simulation Launch ──────────────────────────────────────────────────────

fn launch_econ_simulation(app: &mut App, config: LeontiefConfig) {
    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let phase = Arc::new(Mutex::new("Initializing...".to_string()));
    let phase_clone = phase.clone();
    let (tx, rx) = mpsc::channel();

    let config_owned = config;
    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match economics::run_leontief_simulation(
                &config_owned,
                &progress_clone.inner,
                &phase_clone,
            ) {
                Ok(result) => {
                    let _ = tx.send(ComputeResult::Economics { result });
                }
                Err(e) => {
                    let _ = tx.send(ComputeResult::Error {
                        message: format!("Leontief simulation error: {e}"),
                    });
                }
            }
        }));
        if let Err(panic_info) = outcome {
            let msg = crate::interactive::extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    app.econ_phase = Some(phase);
    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context: ComputeContext {
            algorithm_choice: crate::interactive::AlgorithmChoice::Naive,
            algorithm_name: "Leontief I/O".into(),
            size: 0,
            gen_time_ms: None,
            simd_level: app.sys_info.simd_level,
            is_diff: false,
            diff_alg1: None,
            diff_alg2: None,
        },
        _join_handle: handle,
        child_process: None,
        temp_dir: None,
        compute_request: None,
    });

    app.screen = Screen::EconComputing;
}

// ════════════════════════════════════════════════════════════════════════════
//  RENDER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

// ─── Screen 1: Config (sectors, sparsity, tol, max_iter) ────────────────────

pub fn render_econ_config(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // title
            Constraint::Min(12),   // fields
            Constraint::Length(5), // hints
            Constraint::Length(1), // footer
        ])
        .split(area);

    // Title
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  LEONTIEF SHOCK SIMULATOR — Configuration",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    // Fields
    let values = [
        &app.econ_sectors_input,
        &app.econ_sparsity_input,
        &app.econ_tol_input,
        &app.econ_max_iter_input,
    ];
    let hints = [
        "Number of economic sectors (N×N matrix). Recommended: 100–2000",
        "Fraction of zero entries in A. Higher = sparser economy",
        "Stop when ‖Δx‖∞ < tolerance. Typical: 1e-8 to 1e-12",
        "Safety bound. Neumann series converges in O(log(1/tol)) iters",
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    for (i, label) in CONFIG_LABELS.iter().enumerate() {
        let active = i == app.econ_active_field;
        let marker = if active { "▸ " } else { "  " };
        let field_style = if active {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text)
        };
        let val_display = if active {
            format!("{}_", values[i])
        } else {
            values[i].clone()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {marker}{label}: "), Style::default().fg(t.text_muted)),
            Span::styled(val_display, field_style),
        ]));
        if active {
            lines.push(Line::from(Span::styled(
                format!("    {}", hints[i]),
                Style::default().fg(t.text_dim),
            )));
        }
        lines.push(Line::from(""));
    }

    // RAM estimate
    if let Ok(n) = app.econ_sectors_input.parse::<f64>() {
        let n = n as usize;
        if n >= 2 {
            let cfg = LeontiefConfig {
                sectors: n,
                sparsity: 0.0,
                spectral_target: 0.85,
                tolerance: 1e-10,
                max_iterations: 1,
                shock_sector: 0,
                shock_magnitude: 1.0,
            };
            lines.push(Line::from(Span::styled(
                format!("    Estimated RAM: {:.1} MB ({}×{} matrix)", cfg.estimate_memory_mb(), n, n),
                Style::default().fg(t.text_dim),
            )));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Economy Parameters ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // Hints
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "  The technology matrix A represents inter-sector consumption.",
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                "  Column sums normalized to ρ(A) < 1 for guaranteed convergence.",
                Style::default().fg(t.text_dim),
            )),
        ]),
        chunks[2],
    );

    // Footer
    render_footer(
        &["[Tab] Next field", "[Enter] Continue", "[Esc] Back"],
        frame,
        chunks[3],
        t,
    );
}

// ─── Screen 2: Shock Selection ──────────────────────────────────────────────

pub fn render_econ_shock(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  LEONTIEF SHOCK SIMULATOR — Demand Shock",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let values = [&app.econ_shock_sector_input, &app.econ_shock_mag_input];
    let hints = [
        "Index of the sector hit by the demand shock (0-based)",
        "Monetary units of demand injected into the shocked sector",
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    for (i, label) in SHOCK_LABELS.iter().enumerate() {
        let active = i == app.econ_active_field;
        let marker = if active { "▸ " } else { "  " };
        let field_style = if active {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text)
        };
        let val_display = if active {
            format!("{}_", values[i])
        } else {
            values[i].clone()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {marker}{label}: "), Style::default().fg(t.text_muted)),
            Span::styled(val_display, field_style),
        ]));
        if active {
            lines.push(Line::from(Span::styled(
                format!("    {}", hints[i]),
                Style::default().fg(t.text_dim),
            )));
        }
        lines.push(Line::from(""));
    }

    // Sector name preview
    if let Ok(idx) = app.econ_shock_sector_input.parse::<usize>() {
        if idx < economics::SECTOR_NAMES.len() {
            lines.push(Line::from(Span::styled(
                format!("    Sector name: {}", economics::SECTOR_NAMES[idx]),
                Style::default().fg(t.accent),
            )));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Shock Parameters ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "  A demand shock d is injected into one sector.",
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                "  Neumann iteration propagates cascading effects: xₖ = A·xₖ₋₁ + d",
                Style::default().fg(t.text_dim),
            )),
        ]),
        chunks[2],
    );

    render_footer(
        &["[Tab] Next field", "[Enter] Continue", "[Esc] Back"],
        frame,
        chunks[3],
        t,
    );
}

// ─── Screen 3: Confirm ──────────────────────────────────────────────────────

pub fn render_econ_confirm(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(16),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  LEONTIEF SHOCK SIMULATOR — Confirm",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let sectors = app.econ_sectors_input.as_str();
    let sparsity = app.econ_sparsity_input.as_str();
    let tol = app.econ_tol_input.as_str();
    let max_iter = app.econ_max_iter_input.as_str();
    let shock_sec = app.econ_shock_sector_input.as_str();
    let shock_mag = app.econ_shock_mag_input.as_str();

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(kv_line("  Sectors", sectors, t.text, t.text_muted));
    lines.push(kv_line("  Sparsity", sparsity, t.text, t.text_muted));
    lines.push(kv_line("  Spectral target", "0.85 (auto)", t.text, t.text_muted));
    lines.push(kv_line("  Tolerance", tol, t.text, t.text_muted));
    lines.push(kv_line("  Max iterations", max_iter, t.text, t.text_muted));
    lines.push(Line::from(""));
    lines.push(kv_line("  Shock sector", shock_sec, t.accent, t.text_muted));
    lines.push(kv_line("  Shock magnitude", shock_mag, t.accent, t.text_muted));
    lines.push(Line::from(""));

    // RAM estimate
    if let Ok(n) = sectors.parse::<f64>() {
        let n = n as usize;
        if n >= 2 {
            let ram_mb = (n * n + 3 * n) as f64 * 8.0 / (1024.0 * 1024.0);
            lines.push(kv_line(
                "  Estimated RAM",
                &format!("{:.1} MB", ram_mb),
                t.text,
                t.text_muted,
            ));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Computation: generate A (N×N), then iterate xₖ = A·xₖ₋₁ + d",
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(Span::styled(
        "  using rayon-parallel matrix–vector multiply until convergence.",
        Style::default().fg(t.text_dim),
    )));

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Summary ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(
        &["[Enter] Run simulation", "[Esc] Back"],
        frame,
        chunks[2],
        t,
    );
}

// ─── Screen 4: Computing ────────────────────────────────────────────────────

pub fn render_econ_computing(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  LEONTIEF SHOCK SIMULATOR — Computing",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let (frac, phase_text, eta_text): (f64, String, String) = if let Some(ref task) = app.compute_task {
        let f = task.progress.fraction();
        let ph = app
            .econ_phase
            .as_ref()
            .and_then(|p| p.lock().ok())
            .map(|s| s.clone())
            .unwrap_or_default();
        let eta = task.eta.estimate_remaining(f);
        let eta_str = eta.map(|s| format_duration(s * 1000.0)).unwrap_or_default();
        (f, ph, eta_str)
    } else {
        (0.0, String::new(), String::new())
    };

    let pct = (frac * 100.0).min(100.0);
    let bar_w = (area.width as usize).saturating_sub(20);
    let filled = ((pct / 100.0) * bar_w as f64) as usize;
    let bar: String = format!(
        "  [{}>{}] {:.1}%",
        "█".repeat(filled),
        "░".repeat(bar_w.saturating_sub(filled)),
        pct,
    );

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    // Gear animation
    let gear_idx = app.gear_frame % 4;
    let gears = if pct < 10.0 {
        &GEAR_SINGLE[gear_idx]
    } else {
        &GEAR_FRAMES[gear_idx]
    };
    for line in gears {
        lines.push(Line::from(Span::styled(
            format!("  {line}"),
            Style::default().fg(t.accent),
        )));
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        bar,
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));
    if !phase_text.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  {phase_text}"),
            Style::default().fg(t.text_muted),
        )));
    }
    if !eta_text.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  ETA: {eta_text}"),
            Style::default().fg(t.text_dim),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Neumann Iteration ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(&["[Esc] Cancel"], frame, chunks[2], t);
}

// ─── Screen 5: Results ──────────────────────────────────────────────────────

pub fn render_econ_results(
    app: &App,
    result: &LeontiefResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // header
            Constraint::Length(12),  // summary
            Constraint::Min(10),    // top sectors
            Constraint::Length(1),  // footer
        ])
        .split(area);

    // Header
    let status = if result.converged {
        Span::styled(
            "  LEONTIEF SIMULATION COMPLETE ✓",
            Style::default().fg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            "  LEONTIEF SIMULATION — DID NOT CONVERGE ✗",
            Style::default().fg(Color::Rgb(248, 113, 113)).add_modifier(Modifier::BOLD),
        )
    };
    frame.render_widget(Paragraph::new(Line::from(status)), chunks[0]);

    // Summary panel
    let cfg = &result.config;
    let mut summary: Vec<Line> = Vec::new();
    summary.push(Line::from(""));
    summary.push(kv_line("  Sectors", &format!("{}", cfg.sectors), t.text, t.text_muted));
    summary.push(kv_line("  Iterations", &format!("{}", result.iterations), t.text, t.text_muted));
    summary.push(kv_line("  Final Δ", &format!("{:.2e}", result.final_delta), t.text, t.text_muted));
    summary.push(kv_line("  Converged", &format!("{}", result.converged), t.text, t.text_muted));
    summary.push(kv_line(
        "  Multiplier",
        &format!("{:.4}×", result.multiplier),
        t.accent,
        t.text_muted,
    ));
    summary.push(kv_line(
        "  Computation",
        &format_duration(result.computation_ms),
        t.text,
        t.text_muted,
    ));
    summary.push(kv_line(
        "  MatVec ops",
        &format!("{}", result.total_matvec_ops),
        t.text,
        t.text_muted,
    ));
    summary.push(kv_line(
        "  ρ(A) estimate",
        &format!("{:.4}", result.spectral_radius_est),
        t.text,
        t.text_muted,
    ));
    let save_hint = if app.econ_csv_saved { " (saved)" } else { "" };
    summary.push(kv_line("  CSV", save_hint, t.text_dim, t.text_muted));

    frame.render_widget(
        Paragraph::new(summary).block(
            Block::default()
                .title(" Summary ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // Top 15 sectors by cascade loss
    let mut indexed: Vec<(usize, f64)> = result
        .sector_losses
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let display_count = 15.min(indexed.len());
    let max_loss = indexed.first().map(|x| x.1.abs()).unwrap_or(1.0).max(1e-15);
    let bar_max_w = (area.width as usize).saturating_sub(40);

    let mut sector_lines: Vec<Line> = Vec::new();
    sector_lines.push(Line::from(""));
    for &(idx, loss) in indexed.iter().take(display_count) {
        let name = if idx < result.sector_names.len() {
            &result.sector_names[idx]
        } else {
            "?"
        };
        let bar_len = ((loss.abs() / max_loss) * bar_max_w as f64) as usize;
        let bar_char = if loss >= 0.0 { "▓" } else { "░" };
        let color = if loss >= 0.0 {
            Color::Rgb(74, 222, 128)
        } else {
            Color::Rgb(248, 113, 113)
        };
        sector_lines.push(Line::from(vec![
            Span::styled(
                format!("  {:>3}. {:<18} ", idx, truncate_str(name, 18)),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(bar_char.repeat(bar_len.max(1)), Style::default().fg(color)),
            Span::styled(format!(" {:.2}", loss), Style::default().fg(t.text)),
        ]));
    }

    frame.render_widget(
        Paragraph::new(sector_lines).block(
            Block::default()
                .title(" Top Sectors — Cascade Impact ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[2],
    );

    // Footer
    render_footer(
        &["[S] Save CSV", "[G] Convergence", "[B] Sectors", "[D] Dashboard", "[R] Re-run", "[Q] Back"],
        frame,
        chunks[3],
        t,
    );
}

// ─── Overlays ───────────────────────────────────────────────────────────────

pub fn render_econ_overlay(
    app: &App,
    result: &LeontiefResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    match app.overlay {
        Overlay::EconConvergenceGraph => render_convergence_graph(result, frame, area, t),
        Overlay::EconSectorBars => render_sector_bars(result, frame, area, t),
        Overlay::EconDashboard => render_dashboard(app, result, frame, area, t),
        _ => {}
    }
}

/// [G] Convergence graph — Δ over iterations (log scale via text)
fn render_convergence_graph(
    result: &LeontiefResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.92, 0.88);
    frame.render_widget(Clear, popup);

    let snaps = &result.snapshots;
    if snaps.is_empty() {
        frame.render_widget(
            Paragraph::new("No convergence data").block(
                Block::default()
                    .title(" Convergence ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(t.accent)),
            ),
            popup,
        );
        return;
    }

    let inner_h = popup.height.saturating_sub(6) as usize;
    let inner_w = popup.width.saturating_sub(14) as usize;

    // Log-scale range
    let deltas: Vec<f64> = snaps.iter().map(|s| s.delta.max(1e-20)).collect();
    let log_min = deltas.iter().cloned().fold(f64::MAX, |a, b| a.min(b.ln()));
    let log_max = deltas.iter().cloned().fold(f64::MIN, |a, b| a.max(b.ln()));
    let log_range = (log_max - log_min).max(1e-9);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " Convergence: {} iterations, final Δ = {:.2e}",
            result.iterations, result.final_delta
        ),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    // Text-based scatter plot
    if inner_h > 0 && inner_w > 0 {
        // Build a character grid
        let mut grid = vec![vec![' '; inner_w]; inner_h];

        for snap in snaps.iter() {
            let x_frac = if result.iterations > 0 {
                snap.iteration as f64 / result.iterations as f64
            } else {
                0.0
            };
            let y_frac = 1.0 - (snap.delta.max(1e-20).ln() - log_min) / log_range;
            let col = ((x_frac * (inner_w - 1) as f64).round() as usize).min(inner_w - 1);
            let row = ((y_frac * (inner_h - 1) as f64).round() as usize).min(inner_h - 1);
            grid[row][col] = '●';
        }

        // Draw tolerance line
        let tol_y = 1.0 - (result.config.tolerance.max(1e-20).ln() - log_min) / log_range;
        let tol_row = ((tol_y * (inner_h - 1) as f64).round() as usize).min(inner_h - 1);
        for c in 0..inner_w {
            if grid[tol_row][c] == ' ' {
                grid[tol_row][c] = '─';
            }
        }

        for (r, row) in grid.iter().enumerate() {
            // Y-axis label (log scale)
            let log_val = log_max - (r as f64 / (inner_h - 1).max(1) as f64) * log_range;
            let label = format!("{:>8.1e} │", log_val.exp());
            let mut spans = vec![Span::styled(label, Style::default().fg(t.text_dim))];
            for &ch in row {
                let color = if ch == '●' {
                    t.accent
                } else if ch == '─' {
                    Color::Rgb(248, 113, 113)
                } else {
                    t.bg
                };
                spans.push(Span::styled(String::from(ch), Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }
    }

    // X-axis
    lines.push(Line::from(Span::styled(
        format!("           └{}→ Iteration", "─".repeat(inner_w.min(60))),
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(Span::styled(
        "  [Esc] Close",
        Style::default().fg(t.text_muted),
    )));

    let visible_h = popup.height.saturating_sub(2) as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .title(" Convergence Graph [G] ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent))
                .style(Style::default().bg(t.bg)),
        ),
        popup,
    );
}

/// [B] Sector bar chart — top 15 cascade losses
fn render_sector_bars(
    result: &LeontiefResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.92, 0.88);
    frame.render_widget(Clear, popup);

    let mut indexed: Vec<(usize, f64)> = result
        .sector_losses
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let display_count = 20.min(indexed.len());
    let max_val = indexed.first().map(|x| x.1.abs()).unwrap_or(1.0).max(1e-15);
    let bar_max_w = (popup.width as usize).saturating_sub(36);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " Top {} sectors by cascade amplification (multiplier: {:.3}×)",
            display_count, result.multiplier,
        ),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for &(idx, loss) in indexed.iter().take(display_count) {
        let name = if idx < result.sector_names.len() {
            &result.sector_names[idx]
        } else {
            "?"
        };
        let bar_len = ((loss.abs() / max_val) * bar_max_w as f64) as usize;
        let (bar_char, color) = if loss >= 0.0 {
            ("▓", Color::Rgb(74, 222, 128))
        } else {
            ("░", Color::Rgb(248, 113, 113))
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:>3}. {:<16} ", idx, truncate_str(name, 16)),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(bar_char.repeat(bar_len.max(1)), Style::default().fg(color)),
            Span::styled(format!(" {:.2}", loss), Style::default().fg(t.text)),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [Esc] Close",
        Style::default().fg(t.text_muted),
    )));

    let visible_h = popup.height.saturating_sub(2) as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .title(" Sector Impact [B] ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent))
                .style(Style::default().bg(t.bg)),
        ),
        popup,
    );
}

/// [D] Dashboard with tabs
fn render_dashboard(
    app: &App,
    result: &LeontiefResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.95, 0.92);
    frame.render_widget(Clear, popup);

    let tab = app.econ_dashboard_tab;
    let tab_labels = ["Convergence", "Sectors", "Output Distribution"];

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // tabs
            Constraint::Min(10),  // content
            Constraint::Length(1), // footer
        ])
        .split(popup);

    // Tab bar
    let mut tab_spans: Vec<Span> = Vec::new();
    tab_spans.push(Span::raw("  "));
    for (i, label) in tab_labels.iter().enumerate() {
        if i == tab {
            tab_spans.push(Span::styled(
                format!(" [{label}] "),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ));
        } else {
            tab_spans.push(Span::styled(
                format!("  {label}  "),
                Style::default().fg(t.text_dim),
            ));
        }
    }
    frame.render_widget(
        Paragraph::new(Line::from(tab_spans)).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(t.border))
                .style(Style::default().bg(t.bg)),
        ),
        chunks[0],
    );

    // Tab content
    match tab {
        0 => render_convergence_graph(result, frame, chunks[1], t),
        1 => render_sector_bars(result, frame, chunks[1], t),
        2 => render_output_distribution(result, frame, chunks[1], t),
        _ => {}
    }

    render_footer(
        &["[←/→] Switch tab", "[Esc] Close"],
        frame,
        chunks[2],
        t,
    );
}

/// Output distribution — histogram of final sector outputs
fn render_output_distribution(
    result: &LeontiefResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let outputs = &result.output;
    if outputs.is_empty() {
        return;
    }

    let min_val = outputs.iter().cloned().fold(f64::MAX, f64::min);
    let max_val = outputs.iter().cloned().fold(f64::MIN, f64::max);
    let range = (max_val - min_val).max(1e-15);

    // Build histogram with 20 bins
    let n_bins = 20;
    let mut bins = vec![0usize; n_bins];
    for &v in outputs {
        let idx = (((v - min_val) / range) * (n_bins - 1) as f64).round() as usize;
        bins[idx.min(n_bins - 1)] += 1;
    }
    let max_count = *bins.iter().max().unwrap_or(&1);
    let bar_h = (area.height as usize).saturating_sub(6);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(" Output Distribution — range [{:.2}, {:.2}]", min_val, max_val),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    // Vertical bars rendered as horizontal text rows
    for row in 0..bar_h {
        let threshold = ((bar_h - row) as f64 / bar_h as f64) * max_count as f64;
        let mut spans: Vec<Span> = vec![Span::styled("  ", Style::default())];
        for &count in &bins {
            let ch = if count as f64 >= threshold { "█" } else { " " };
            spans.push(Span::styled(
                format!(" {ch} "),
                Style::default().fg(t.accent),
            ));
        }
        lines.push(Line::from(spans));
    }

    // X-axis
    lines.push(Line::from(Span::styled(
        format!("  {:.1}{:>width$}{:.1}", min_val, "", max_val, width = n_bins * 3 - 8),
        Style::default().fg(t.text_dim),
    )));

    let visible_h = area.height.saturating_sub(2) as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border))
                .style(Style::default().bg(t.bg)),
        ),
        area,
    );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

fn centered_popup(area: Rect, w_frac: f32, h_frac: f32) -> Rect {
    let popup_w = (area.width as f32 * w_frac) as u16;
    let popup_h = (area.height as f32 * h_frac) as u16;
    Rect {
        x: area.x + (area.width.saturating_sub(popup_w)) / 2,
        y: area.y + (area.height.saturating_sub(popup_h)) / 2,
        width: popup_w.min(area.width),
        height: popup_h.min(area.height),
    }
}

fn render_footer(items: &[&str], frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors) {
    let spans: Vec<Span> = items
        .iter()
        .enumerate()
        .flat_map(|(i, item)| {
            let sep = if i > 0 { "  " } else { "  " };
            vec![
                Span::styled(sep, Style::default()),
                Span::styled(*item, Style::default().fg(t.text_muted)),
            ]
        })
        .collect();
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
