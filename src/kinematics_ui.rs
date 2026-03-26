// =============================================================================
//  KINEMATICS & PATHFINDING (MDP) — TUI WIZARD & RESULTS
// =============================================================================

use std::sync::{mpsc, Arc, Mutex};

use crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::common::{ProgressHandle, ThemeColors};
use crate::kinematics::{self, MdpConfig, MdpResult};
use crate::interactive::{
    App, ComputeContext, ComputeResult, ComputeTask, EtaTracker, Overlay, Screen,
    format_duration, kv_line, GEAR_FRAMES, GEAR_SINGLE,
};

// --- Field Labels -----------------------------------------------------------

const CONFIG_LABELS: &[&str] = &[
    "States (N)",
    "Sparsity (0-1)",
    "Convergence eps",
    "Max iterations",
    "Seed (optional)",
];

// --- Input Handlers ---------------------------------------------------------

pub fn handle_mdp_config(app: &mut App, key: KeyCode) {
    let n_fields = CONFIG_LABELS.len();
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.mdp_active_field = (app.mdp_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.mdp_active_field = if app.mdp_active_field == 0 {
                n_fields - 1
            } else {
                app.mdp_active_field - 1
            };
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' => {
            active_config_field_mut(app).push(c);
        }
        KeyCode::Backspace => {
            active_config_field_mut(app).pop();
        }
        KeyCode::Enter => {
            app.screen = Screen::MdpConfirm;
        }
        KeyCode::Esc => {
            app.screen = Screen::CategoryMenu {
                category: crate::interactive::MenuCategory::Engineering,
            };
            app.category_menu_idx = 0;
        }
        _ => {}
    }
}

fn active_config_field_mut(app: &mut App) -> &mut String {
    match app.mdp_active_field {
        0 => &mut app.mdp_states_input,
        1 => &mut app.mdp_sparsity_input,
        2 => &mut app.mdp_eps_input,
        3 => &mut app.mdp_max_iter_input,
        4 => &mut app.mdp_seed_input,
        _ => &mut app.mdp_states_input,
    }
}

pub fn handle_mdp_confirm(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => {
            if let Some(config) = build_mdp_config(app) {
                launch_mdp_simulation(app, config);
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::MdpConfig;
        }
        _ => {}
    }
}

pub fn handle_mdp_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if !app.mdp_csv_saved {
                if let Screen::MdpResults { ref result } = app.screen {
                    let path = format!(
                        "flust_mdp_{}.csv",
                        crate::io::timestamp_now().replace(':', "-")
                    );
                    if crate::kinematics_export::export_mdp_csv(result, &path).is_ok() {
                        app.mdp_csv_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('g') | KeyCode::Char('G') => {
            app.overlay = Overlay::MdpConvergenceGraph;
            app.mdp_overlay_scroll = 0;
        }
        KeyCode::Char('b') | KeyCode::Char('B') => {
            app.overlay = Overlay::MdpStateBars;
            app.mdp_overlay_scroll = 0;
        }
        KeyCode::Char('d') | KeyCode::Char('D') => {
            app.overlay = Overlay::MdpDashboard;
            app.mdp_dashboard_tab = 0;
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.screen = Screen::MdpConfirm;
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// --- Config Builder ---------------------------------------------------------

fn build_mdp_config(app: &App) -> Option<MdpConfig> {
    let matrix_size: usize = app.mdp_states_input.parse::<f64>().ok()? as usize;
    let sparsity: f64 = app.mdp_sparsity_input.parse().ok()?;
    let epsilon: f64 = app.mdp_eps_input.parse().ok()?;
    let max_iterations: usize = app.mdp_max_iter_input.parse::<f64>().ok()? as usize;
    let seed: Option<u64> = if app.mdp_seed_input.is_empty() {
        None
    } else {
        Some(app.mdp_seed_input.parse::<u64>().ok()?)
    };

    if matrix_size < 2 || matrix_size > 4096 {
        return None;
    }
    if !(0.0..1.0).contains(&sparsity) {
        return None;
    }
    if epsilon <= 0.0 {
        return None;
    }
    if max_iterations < 1 {
        return None;
    }

    Some(MdpConfig {
        matrix_size,
        sparsity,
        epsilon,
        max_iterations,
        seed,
    })
}

// --- Launch Simulation ------------------------------------------------------

fn launch_mdp_simulation(app: &mut App, config: MdpConfig) {
    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let phase = Arc::new(Mutex::new("Initializing...".to_string()));
    let phase_clone = phase.clone();
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match kinematics::run_mdp_simulation(&config, &progress_clone.inner, &phase_clone) {
                Ok(result) => {
                    let _ = tx.send(ComputeResult::Mdp { result });
                }
                Err(e) => {
                    let _ = tx.send(ComputeResult::Error {
                        message: format!("MDP simulation error: {e}"),
                    });
                }
            }
        }));
        if let Err(panic_info) = outcome {
            let msg = crate::interactive::extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    app.mdp_phase = Some(phase);
    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context: ComputeContext {
            algorithm_choice: crate::interactive::AlgorithmChoice::Naive,
            algorithm_name: "MDP Power Iteration".into(),
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

    app.screen = Screen::MdpComputing;
}

// ============================================================================
//  RENDER FUNCTIONS
// ============================================================================

// --- Screen 1: Config -------------------------------------------------------

pub fn render_mdp_config(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(14),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  KINEMATICS & PATHFINDING — MDP Configuration",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let values = [
        &app.mdp_states_input,
        &app.mdp_sparsity_input,
        &app.mdp_eps_input,
        &app.mdp_max_iter_input,
        &app.mdp_seed_input,
    ];
    let hints = [
        "State space dimension (N\u{00d7}N transition matrix). Recommended: 64\u{2013}2048",
        "Fraction of zero transitions. Higher = sparser connectivity graph",
        "Convergence: \u{2016}P\u{207f} \u{2212} P\u{207f}\u{2044}\u{00b2}\u{2016}_F < \u{03b5}. Typical: 1e-10",
        "Max repeated-squaring steps. Safety bound: 100\u{2013}500",
        "Leave empty for random, or fix for reproducibility",
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    for (i, label) in CONFIG_LABELS.iter().enumerate() {
        let active = i == app.mdp_active_field;
        let marker = if active { "\u{25b8} " } else { "  " };
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
    if let Ok(n) = app.mdp_states_input.parse::<f64>() {
        let n = n as usize;
        if n >= 2 {
            let cfg = MdpConfig {
                matrix_size: n,
                sparsity: 0.0,
                epsilon: 1e-10,
                max_iterations: 1,
                seed: None,
            };
            lines.push(Line::from(Span::styled(
                format!("    Estimated RAM: {:.1} MB ({}\u{00d7}{} matrix)", cfg.estimate_memory_mb(), n, n),
                Style::default().fg(t.text_dim),
            )));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" MDP Parameters ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "  A row-stochastic matrix P models state transitions.",
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                "  Repeated squaring P\u{00b2}\u{2192}P\u{2074}\u{2192}P\u{2078}\u{2026} converges to steady-state \u{03c0}.",
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

// --- Screen 2: Confirm ------------------------------------------------------

pub fn render_mdp_confirm(
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
            "  KINEMATICS & PATHFINDING \u{2014} Confirm",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(kv_line("  States (N)", &app.mdp_states_input, t.text, t.text_muted));
    lines.push(kv_line("  Sparsity", &app.mdp_sparsity_input, t.text, t.text_muted));
    lines.push(kv_line("  Epsilon", &app.mdp_eps_input, t.text, t.text_muted));
    lines.push(kv_line("  Max iterations", &app.mdp_max_iter_input, t.text, t.text_muted));
    let seed_display = if app.mdp_seed_input.is_empty() {
        "random".to_string()
    } else {
        app.mdp_seed_input.clone()
    };
    lines.push(kv_line("  Seed", &seed_display, t.text, t.text_muted));
    lines.push(Line::from(""));

    // RAM estimate
    if let Ok(n) = app.mdp_states_input.parse::<f64>() {
        let n = n as usize;
        if n >= 2 {
            let cfg = MdpConfig {
                matrix_size: n,
                sparsity: 0.0,
                epsilon: 1e-10,
                max_iterations: 1,
                seed: None,
            };
            lines.push(kv_line(
                "  Estimated RAM",
                &format!("{:.1} MB", cfg.estimate_memory_mb()),
                t.text,
                t.text_muted,
            ));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Computation: generate stochastic P, then square P\u{00b2}\u{2192}P\u{2074}\u{2192}P\u{2078}\u{2026}",
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(Span::styled(
        "  using HPC multiply_hpc_fused until \u{2016}\u{0394}\u{2016}_F < \u{03b5}.",
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

// --- Screen 3: Computing ----------------------------------------------------

pub fn render_mdp_computing(
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
            "  KINEMATICS & PATHFINDING \u{2014} Computing",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let (frac, phase_text, eta_text): (f64, String, String) = if let Some(ref task) = app.compute_task {
        let f = task.progress.fraction();
        let ph = app
            .mdp_phase
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
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(bar_w.saturating_sub(filled)),
        pct,
    );

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

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
                .title(" Matrix Power Iteration ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(&["[Esc] Cancel"], frame, chunks[2], t);
}

// --- Screen 4: Results ------------------------------------------------------

pub fn render_mdp_results(
    app: &App,
    result: &MdpResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(13),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    // Header
    let status = if result.converged {
        Span::styled(
            "  MDP STEADY-STATE CONVERGED \u{2713}",
            Style::default().fg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            "  MDP ITERATION \u{2014} DID NOT CONVERGE \u{2717}",
            Style::default().fg(Color::Rgb(248, 113, 113)).add_modifier(Modifier::BOLD),
        )
    };
    frame.render_widget(Paragraph::new(Line::from(status)), chunks[0]);

    // Summary panel
    let cfg = &result.config;
    let mut summary: Vec<Line> = Vec::new();
    summary.push(Line::from(""));
    summary.push(kv_line("  States (N)", &format!("{}", cfg.matrix_size), t.text, t.text_muted));
    summary.push(kv_line("  Iterations", &format!("{}", result.iterations), t.text, t.text_muted));
    summary.push(kv_line("  Final \u{0394}", &format!("{:.2e}", result.final_norm_diff), t.text, t.text_muted));
    summary.push(kv_line("  Spectral gap", &format!("{:.4}", result.spectral_gap), t.accent, t.text_muted));
    summary.push(kv_line("  Peak state", &format!("S{} ({:.4e})", result.peak_state, result.peak_probability), t.accent, t.text_muted));
    summary.push(kv_line(
        "  Computation",
        &format_duration(result.computation_ms),
        t.text,
        t.text_muted,
    ));
    summary.push(kv_line(
        "  Multiplications",
        &format!("{}", result.total_multiplications),
        t.text,
        t.text_muted,
    ));
    let save_hint = if app.mdp_csv_saved { " (saved)" } else { "" };
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

    // Top states by probability
    let mut indexed: Vec<(usize, f64)> = result
        .steady_state
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let display_count = 20.min(indexed.len());
    let max_prob = indexed.first().map(|x| x.1).unwrap_or(1.0).max(1e-15);
    let bar_max_w = (area.width as usize).saturating_sub(30);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(" Top {} states by steady-state probability \u{03c0}", display_count),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for &(idx, prob) in indexed.iter().take(display_count) {
        let bar_len = ((prob / max_prob) * bar_max_w as f64) as usize;
        lines.push(Line::from(vec![
            Span::styled(
                format!("  S{:<5} ", idx),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(
                "\u{2593}".repeat(bar_len.max(1)),
                Style::default().fg(Color::Rgb(74, 222, 128)),
            ),
            Span::styled(format!(" {:.4e}", prob), Style::default().fg(t.text)),
        ]));
    }

    let visible_h = chunks[2].height.saturating_sub(2) as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .title(" Steady-State Distribution ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[2],
    );

    render_footer(
        &["[S] Save CSV", "[G] Convergence", "[B] State Bars", "[D] Dashboard", "[R] Re-run", "[Esc] Menu"],
        frame,
        chunks[3],
        t,
    );
}

// ============================================================================
//  OVERLAY RENDERERS
// ============================================================================

pub fn render_mdp_overlay(
    app: &App,
    result: &MdpResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    match app.overlay {
        Overlay::MdpConvergenceGraph => render_convergence_graph(result, frame, area, t),
        Overlay::MdpStateBars => render_state_bars(result, frame, area, t),
        Overlay::MdpDashboard => render_dashboard(app, result, frame, area, t),
        _ => {}
    }
}

/// [G] Convergence graph: Frobenius norm vs squaring steps (log scale)
fn render_convergence_graph(
    result: &MdpResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.92, 0.88);
    frame.render_widget(Clear, popup);

    let snaps = &result.snapshots;
    if snaps.is_empty() {
        frame.render_widget(
            Paragraph::new(" No convergence data.").block(
                Block::default().borders(Borders::ALL).border_style(Style::default().fg(t.border)),
            ),
            popup,
        );
        return;
    }

    let inner_h = popup.height.saturating_sub(6) as usize;
    let inner_w = popup.width.saturating_sub(14) as usize;

    let norms: Vec<f64> = snaps.iter().map(|s| s.norm_diff.max(1e-20)).collect();
    let log_min = norms.iter().cloned().fold(f64::MAX, |a, b| a.min(b.ln()));
    let log_max = norms.iter().cloned().fold(f64::MIN, |a, b| a.max(b.ln()));
    let log_range = (log_max - log_min).max(1e-9);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " Convergence: {} squaring steps, final \u{0394} = {:.2e}",
            result.iterations, result.final_norm_diff,
        ),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    if inner_h > 0 && inner_w > 0 {
        let mut grid = vec![vec![' '; inner_w]; inner_h];

        for (si, snap) in snaps.iter().enumerate() {
            let x_frac = if snaps.len() > 1 {
                si as f64 / (snaps.len() - 1) as f64
            } else {
                0.5
            };
            let y_frac = 1.0 - (snap.norm_diff.max(1e-20).ln() - log_min) / log_range;
            let col = ((x_frac * (inner_w - 1) as f64).round() as usize).min(inner_w - 1);
            let row = ((y_frac * (inner_h - 1) as f64).round() as usize).min(inner_h - 1);
            grid[row][col] = '\u{25cf}';
        }

        // Epsilon line
        let eps_y = 1.0 - (result.config.epsilon.max(1e-20).ln() - log_min) / log_range;
        let eps_row = ((eps_y * (inner_h - 1) as f64).round() as usize).min(inner_h - 1);
        for c in 0..inner_w {
            if grid[eps_row][c] == ' ' {
                grid[eps_row][c] = '\u{2500}';
            }
        }

        for (r, row) in grid.iter().enumerate() {
            let log_val = log_max - (r as f64 / (inner_h - 1).max(1) as f64) * log_range;
            let label = format!("{:>8.1e} \u{2502}", log_val.exp());
            let mut spans = vec![Span::styled(label, Style::default().fg(t.text_dim))];
            for &ch in row {
                let color = if ch == '\u{25cf}' {
                    t.accent
                } else if ch == '\u{2500}' {
                    Color::Rgb(248, 113, 113)
                } else {
                    t.bg
                };
                spans.push(Span::styled(String::from(ch), Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }
    }

    lines.push(Line::from(Span::styled(
        format!("           \u{2514}{}\u{2192} Squaring step", "\u{2500}".repeat(inner_w.min(60))),
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

/// [B] State probability bar chart
fn render_state_bars(
    result: &MdpResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.92, 0.88);
    frame.render_widget(Clear, popup);

    let mut indexed: Vec<(usize, f64)> = result
        .steady_state
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let display_count = 25.min(indexed.len());
    let max_val = indexed.first().map(|x| x.1).unwrap_or(1.0).max(1e-15);
    let bar_max_w = (popup.width as usize).saturating_sub(30);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " Top {} states by \u{03c0} (spectral gap: {:.4})",
            display_count, result.spectral_gap,
        ),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for &(idx, prob) in indexed.iter().take(display_count) {
        let bar_len = ((prob / max_val) * bar_max_w as f64) as usize;
        lines.push(Line::from(vec![
            Span::styled(
                format!("  S{:<5} ", idx),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(
                "\u{2593}".repeat(bar_len.max(1)),
                Style::default().fg(Color::Rgb(74, 222, 128)),
            ),
            Span::styled(format!(" {:.4e}", prob), Style::default().fg(t.text)),
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
                .title(" State Distribution [B] ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent))
                .style(Style::default().bg(t.bg)),
        ),
        popup,
    );
}

/// [D] Dashboard with tabs: Convergence, States, Entropy
fn render_dashboard(
    app: &App,
    result: &MdpResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.95, 0.92);
    frame.render_widget(Clear, popup);

    let tab = app.mdp_dashboard_tab;
    let tab_labels = ["Convergence", "State Distribution", "Probability Histogram"];

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
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

    match tab {
        0 => render_convergence_graph(result, frame, chunks[1], t),
        1 => render_state_bars(result, frame, chunks[1], t),
        2 => render_probability_histogram(result, frame, chunks[1], t),
        _ => {}
    }

    render_footer(
        &["[\u{2190}/\u{2192}] Switch tab", "[Esc] Close"],
        frame,
        chunks[2],
        t,
    );
}

/// Probability histogram: bin distribution of steady-state probabilities
fn render_probability_histogram(
    result: &MdpResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let probs = &result.steady_state;
    if probs.is_empty() {
        return;
    }

    let min_val = probs.iter().cloned().fold(f64::MAX, f64::min);
    let max_val = probs.iter().cloned().fold(f64::MIN, f64::max);
    let range = (max_val - min_val).max(1e-15);

    let n_bins = 20;
    let mut bins = vec![0usize; n_bins];
    for &v in probs {
        let idx = (((v - min_val) / range) * (n_bins - 1) as f64).round() as usize;
        bins[idx.min(n_bins - 1)] += 1;
    }
    let max_count = *bins.iter().max().unwrap_or(&1);
    let bar_h = (area.height as usize).saturating_sub(6);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(" Probability Distribution \u{2014} range [{:.2e}, {:.2e}]", min_val, max_val),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for row in 0..bar_h {
        let threshold = ((bar_h - row) as f64 / bar_h as f64) * max_count as f64;
        let mut spans: Vec<Span> = vec![Span::styled("  ", Style::default())];
        for &count in &bins {
            let ch = if count as f64 >= threshold { "\u{2588}" } else { " " };
            spans.push(Span::styled(
                format!(" {ch} "),
                Style::default().fg(t.accent),
            ));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(Span::styled(
        format!("  {:.1e}{:>width$}{:.1e}", min_val, "", max_val, width = n_bins * 3 - 8),
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

// --- Helpers ----------------------------------------------------------------

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
