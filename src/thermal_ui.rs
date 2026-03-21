// ─── Thermal Simulation TUI ────────────────────────────────────────────────


use std::sync::{mpsc, Arc, Mutex};

use crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::common::{ProgressHandle, ThemeColors};
use crate::interactive::{
    App, ComputeContext, ComputeResult, ComputeTask, EtaTracker, Screen,
    format_duration, kv_line,
};
use crate::thermal::{
    self, FluidProperties, TegProperties, TegWall, ThermalSimConfig, ThermalSimResult,
};

// ─── Fluid Names ───────────────────────────────────────────────────────────

const FLUID_NAMES: &[&str] = &["Water", "Engine Oil", "Ethylene Glycol (Antifreeze)"];

fn fluid_for_index(idx: usize) -> FluidProperties {
    match idx {
        0 => FluidProperties::water(),
        1 => FluidProperties::oil(),
        2 => FluidProperties::ethylene_glycol(),
        _ => FluidProperties::water(),
    }
}

// ─── Geometry Field Labels ─────────────────────────────────────────────────

const GEOM_LABELS: &[&str] = &[
    "Length X [m]",
    "Length Y [m]",
    "Length Z [m]",
    "Grid N (N\u{00d7}N\u{00d7}N)",
];

const TEG_LABELS: &[&str] = &[
    "Seebeck coeff [V/K]",
    "R_internal [\u{03a9}]",
    "R_load [\u{03a9}]",
    "TEG area [m\u{00b2}]",
    "TEG thickness [m]",
    "TEG \u{03bb} [W/(m\u{00b7}K)]",
];

// ════════════════════════════════════════════════════════════════════════════
//  INPUT HANDLERS
// ════════════════════════════════════════════════════════════════════════════

pub fn handle_thermal_fluid_select(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Up => {
            if app.thermal_fluid_idx > 0 {
                app.thermal_fluid_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.thermal_fluid_idx < FLUID_NAMES.len() - 1 {
                app.thermal_fluid_idx += 1;
            }
        }
        KeyCode::Enter => {
            app.screen = Screen::ThermalGeometry;
        }
        KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

pub fn handle_thermal_geometry(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Tab => {
            app.thermal_geometry_active = (app.thermal_geometry_active + 1) % 4;
        }
        KeyCode::BackTab => {
            app.thermal_geometry_active = (app.thermal_geometry_active + 3) % 4;
        }
        KeyCode::Up => {
            app.thermal_geometry_active = (app.thermal_geometry_active + 3) % 4;
        }
        KeyCode::Down => {
            app.thermal_geometry_active = (app.thermal_geometry_active + 1) % 4;
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => {
            let field = &mut app.thermal_geometry_fields[app.thermal_geometry_active];
            if field.len() < 10 {
                field.push(c);
            }
        }
        KeyCode::Backspace => {
            app.thermal_geometry_fields[app.thermal_geometry_active].pop();
        }
        KeyCode::Enter => {
            // Validate all fields parse
            let valid = app.thermal_geometry_fields.iter().all(|f| f.parse::<f64>().is_ok());
            if valid {
                app.screen = Screen::ThermalTeg;
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::ThermalFluidSelect;
        }
        _ => {}
    }
}

pub fn handle_thermal_teg(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('d') | KeyCode::Char('D') => {
            // Reset to defaults
            let def = TegProperties::standard_bi2te3();
            app.thermal_teg_fields = [
                format!("{}", def.seebeck_coefficient),
                format!("{}", def.internal_resistance),
                format!("{}", def.load_resistance),
                format!("{}", def.teg_area),
                format!("{}", def.teg_thickness),
                format!("{}", def.teg_thermal_conductivity),
            ];
            app.thermal_use_defaults = true;
        }
        KeyCode::Tab => {
            app.thermal_teg_active = (app.thermal_teg_active + 1) % 6;
        }
        KeyCode::BackTab => {
            app.thermal_teg_active = (app.thermal_teg_active + 5) % 6;
        }
        KeyCode::Up => {
            app.thermal_teg_active = (app.thermal_teg_active + 5) % 6;
        }
        KeyCode::Down => {
            app.thermal_teg_active = (app.thermal_teg_active + 1) % 6;
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => {
            let field = &mut app.thermal_teg_fields[app.thermal_teg_active];
            if field.len() < 10 {
                field.push(c);
            }
            app.thermal_use_defaults = false;
        }
        KeyCode::Backspace => {
            app.thermal_teg_fields[app.thermal_teg_active].pop();
            app.thermal_use_defaults = false;
        }
        KeyCode::Enter => {
            let valid = app.thermal_teg_fields.iter().all(|f| f.parse::<f64>().is_ok());
            if valid {
                app.screen = Screen::ThermalConfirm;
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::ThermalGeometry;
        }
        _ => {}
    }
}

pub fn handle_thermal_confirm(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => {
            // Build config and launch simulation
            if let Some(config) = build_config_from_app(app) {
                launch_thermal_simulation(app, config);
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::ThermalTeg;
        }
        _ => {}
    }
}

pub fn handle_thermal_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if !app.thermal_csv_saved {
                if let Screen::ThermalResults { ref result } = app.screen {
                    let path = format!(
                        "flust_thermal_{}.csv",
                        crate::io::timestamp_now().replace(':', "-")
                    );
                    if crate::thermal_export::export_snapshots_csv(result, &path).is_ok() {
                        app.thermal_csv_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('x') | KeyCode::Char('X') => {
            if !app.thermal_field_saved {
                if let Screen::ThermalResults { ref result } = app.screen {
                    let path = format!(
                        "flust_thermal_field_{}.csv",
                        crate::io::timestamp_now().replace(':', "-")
                    );
                    if crate::thermal_export::export_final_field_csv(result, &path).is_ok() {
                        app.thermal_field_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            // Re-run with same parameters
            app.screen = Screen::ThermalConfirm;
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// ─── Config Builder ────────────────────────────────────────────────────────

fn build_config_from_app(app: &App) -> Option<ThermalSimConfig> {
    let lx: f64 = app.thermal_geometry_fields[0].parse().ok()?;
    let ly: f64 = app.thermal_geometry_fields[1].parse().ok()?;
    let lz: f64 = app.thermal_geometry_fields[2].parse().ok()?;
    let n: usize = app.thermal_geometry_fields[3].parse::<f64>().ok()? as usize;
    let n = n.max(4).min(48);

    let seebeck: f64 = app.thermal_teg_fields[0].parse().ok()?;
    let r_int: f64 = app.thermal_teg_fields[1].parse().ok()?;
    let r_load: f64 = app.thermal_teg_fields[2].parse().ok()?;
    let area: f64 = app.thermal_teg_fields[3].parse().ok()?;
    let thickness: f64 = app.thermal_teg_fields[4].parse().ok()?;
    let k_teg: f64 = app.thermal_teg_fields[5].parse().ok()?;

    Some(ThermalSimConfig {
        length_x: lx,
        length_y: ly,
        length_z: lz,
        nx: n,
        ny: n,
        nz: n,
        t_initial: 85.0,
        t_boundary: 20.0,
        fluid: fluid_for_index(app.thermal_fluid_idx),
        teg: TegProperties {
            seebeck_coefficient: seebeck,
            internal_resistance: r_int,
            load_resistance: r_load,
            teg_area: area,
            teg_thickness: thickness,
            teg_thermal_conductivity: k_teg,
        },
        time_step_dt: 0.0, // auto
        total_steps: 1000,
        save_every_n: 10,
        teg_wall: TegWall::XMin,
    })
}

// ─── Launch Simulation ─────────────────────────────────────────────────────

fn launch_thermal_simulation(app: &mut App, config: ThermalSimConfig) {
    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let phase = Arc::new(Mutex::new("Initializing...".to_string()));
    let phase_clone = phase.clone();
    let (tx, rx) = mpsc::channel();

    let mut config_owned = config;
    let handle = std::thread::spawn(move || {
        match thermal::run_thermal_simulation(&mut config_owned, &progress_clone, &phase_clone) {
            Ok(result) => {
                let _ = tx.send(ComputeResult::Thermal { result });
            }
            Err(e) => {
                let mut p = phase_clone.lock().unwrap();
                *p = format!("ERROR: {e}");
            }
        }
    });

    app.thermal_phase = Some(phase);
    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context: ComputeContext {
            algorithm_choice: crate::interactive::AlgorithmChoice::Naive, // dummy
            algorithm_name: "Thermal FDM".into(),
            size: 0,
            gen_time_ms: None,
            simd_level: app.sys_info.simd_level,
            is_diff: false,
            diff_alg1: None,
            diff_alg2: None,
        },
        _join_handle: handle,
    });

    app.screen = Screen::ThermalComputing;
}

// ════════════════════════════════════════════════════════════════════════════
//  RENDER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

// ─── Screen 1: Fluid Selection ─────────────────────────────────────────────

pub fn render_thermal_fluid_select(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(12),   // fluid list
            Constraint::Length(3), // footer
        ])
        .split(area);

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL SIMULATION",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Step 1/4: Select Fluid", Style::default().fg(t.text_muted)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    // Fluid list with properties
    let mut lines = vec![Line::from("")];
    for (i, name) in FLUID_NAMES.iter().enumerate() {
        let props = fluid_for_index(i);
        let selected = i == app.thermal_fluid_idx;
        let marker = if selected { "\u{25b6} " } else { "  " };
        let style = if selected {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text)
        };

        lines.push(Line::from(Span::styled(
            format!("  {marker}{name}"),
            style,
        )));
        lines.push(Line::from(Span::styled(
            format!(
                "      \u{03bb}={:.3} W/(m\u{00b7}K)   \u{03c1}={:.0} kg/m\u{00b3}   c={:.0} J/(kg\u{00b7}K)   \u{03b1}={:.2e} m\u{00b2}/s",
                props.thermal_conductivity,
                props.density,
                props.specific_heat,
                props.thermal_diffusivity(),
            ),
            Style::default().fg(if selected { t.text_muted } else { t.text_dim }),
        )));
        lines.push(Line::from(""));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Select Fluid ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // Footer
    render_footer(
        frame,
        chunks[2],
        t,
        &["[\u{2191}\u{2193}] Navigate", "[Enter] Select", "[Esc] Back"],
    );
}

// ─── Screen 2: Geometry Input ──────────────────────────────────────────────

pub fn render_thermal_geometry(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(12),   // fields
            Constraint::Length(5), // estimate
            Constraint::Length(3), // footer
        ])
        .split(area);

    // Header
    let step_label = format!("  Step 2/4: Reservoir Geometry  |  Fluid: {}", FLUID_NAMES[app.thermal_fluid_idx]);
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL SIMULATION",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            step_label,
            Style::default().fg(t.text_muted),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    // Fields
    let mut lines = vec![Line::from("")];
    for (i, label) in GEOM_LABELS.iter().enumerate() {
        let active = i == app.thermal_geometry_active;
        let cursor = if active { "\u{25b6} " } else { "  " };
        let val = &app.thermal_geometry_fields[i];
        let display_val = if active {
            format!("{val}_")
        } else {
            val.clone()
        };
        let style = if active {
            Style::default().fg(t.accent)
        } else {
            Style::default().fg(t.text)
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {cursor}{label:<24}"), Style::default().fg(t.text_muted)),
            Span::styled(display_val, style),
        ]));
        lines.push(Line::from(""));
    }
    lines.push(Line::from(Span::styled(
        "  Recommended: 16 (fast), 24 (balanced), 32 (accurate)",
        Style::default().fg(t.text_dim),
    )));

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Reservoir Dimensions ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // Memory/grid estimate
    let est_lines = if let Ok(n) = app.thermal_geometry_fields[3].parse::<f64>() {
        let n = (n as usize).max(4).min(48);
        let total = n * n * n;
        let nnz = total * 7;
        let ram_mb = (nnz as f64 * 16.0 / 1024.0 / 1024.0) as usize;
        vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Grid: ", Style::default().fg(t.text_dim)),
                Span::styled(
                    format!("{n}\u{00d7}{n}\u{00d7}{n} = {total} nodes"),
                    Style::default().fg(t.text),
                ),
                Span::styled(
                    format!("    Matrix NNZ \u{2248} {nnz}    Est. RAM: ~{ram_mb} MB"),
                    Style::default().fg(t.text_muted),
                ),
            ]),
        ]
    } else {
        vec![Line::from("")]
    };
    frame.render_widget(
        Paragraph::new(est_lines).block(
            Block::default()
                .title(" Estimate ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[2],
    );

    render_footer(
        frame,
        chunks[3],
        t,
        &["[Tab/\u{2191}\u{2193}] Navigate", "[Enter] Next", "[Esc] Back"],
    );
}

// ─── Screen 3: TEG Parameters ──────────────────────────────────────────────

pub fn render_thermal_teg(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(16),   // fields
            Constraint::Length(3), // footer
        ])
        .split(area);

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL SIMULATION",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Step 3/4: TEG Parameters", Style::default().fg(t.text_muted)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    let mut lines = vec![Line::from("")];
    if app.thermal_use_defaults {
        lines.push(Line::from(Span::styled(
            "  Using standard Bi\u{2082}Te\u{2083} module defaults",
            Style::default().fg(t.ok),
        )));
        lines.push(Line::from(""));
    }
    for (i, label) in TEG_LABELS.iter().enumerate() {
        let active = i == app.thermal_teg_active;
        let cursor = if active { "\u{25b6} " } else { "  " };
        let val = &app.thermal_teg_fields[i];
        let display_val = if active {
            format!("{val}_")
        } else {
            val.clone()
        };
        let style = if active {
            Style::default().fg(t.accent)
        } else {
            Style::default().fg(t.text)
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {cursor}{label:<24}"), Style::default().fg(t.text_muted)),
            Span::styled(display_val, style),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Press [D] to reset to Bi\u{2082}Te\u{2083} defaults",
        Style::default().fg(t.text_dim),
    )));

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" TEG Module (Thermoelectric Generator) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(
        frame,
        chunks[2],
        t,
        &["[Tab/\u{2191}\u{2193}] Navigate", "[D] Defaults", "[Enter] Next", "[Esc] Back"],
    );
}

// ─── Screen 4: Confirmation ────────────────────────────────────────────────

pub fn render_thermal_confirm(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(18),   // summary
            Constraint::Length(3), // footer
        ])
        .split(area);

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL SIMULATION",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Step 4/4: Confirm & Run", Style::default().fg(t.text_muted)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    // Build config to show summary
    let summary = if let Some(cfg) = build_config_from_app(app) {
        let alpha = cfg.fluid.thermal_diffusivity();
        let max_dt = cfg.max_stable_dt(alpha);
        let dt = 0.8 * max_dt;
        let total_nodes = cfg.total_nodes();
        let mem_mb = cfg.estimate_memory_mb();

        vec![
            Line::from(""),
            kv_line("  Fluid       ", &cfg.fluid.name, t.text, t.text_dim),
            kv_line(
                "  Reservoir   ",
                &format!(
                    "{:.1}\u{00d7}{:.1}\u{00d7}{:.1} cm",
                    cfg.length_x * 100.0, cfg.length_y * 100.0, cfg.length_z * 100.0
                ),
                t.text, t.text_dim,
            ),
            kv_line(
                "  Grid        ",
                &format!("{}\u{00d7}{}\u{00d7}{} = {} nodes", cfg.nx, cfg.ny, cfg.nz, total_nodes),
                t.text, t.text_dim,
            ),
            kv_line("  T_initial   ", &format!("{:.1}\u{00b0}C", cfg.t_initial), t.accent, t.text_dim),
            kv_line("  T_boundary  ", &format!("{:.1}\u{00b0}C", cfg.t_boundary), t.text_muted, t.text_dim),
            kv_line(
                "  TEG Seebeck ",
                &format!("{:.0} mV/K", cfg.teg.seebeck_coefficient * 1000.0),
                t.text, t.text_dim,
            ),
            kv_line(
                "  R_int/R_load",
                &format!("{:.1}\u{03a9} / {:.1}\u{03a9}", cfg.teg.internal_resistance, cfg.teg.load_resistance),
                t.text, t.text_dim,
            ),
            Line::from(""),
            kv_line("  \u{03b1} (diffus.) ", &format!("{:.3e} m\u{00b2}/s", alpha), t.text, t.text_dim),
            kv_line("  dt (auto)   ", &format!("{:.4} s", dt), t.text, t.text_dim),
            kv_line("  Max dt      ", &format!("{:.4} s", max_dt), t.text_dim, t.text_dim),
            kv_line("  Steps       ", &format!("{}", cfg.total_steps), t.text, t.text_dim),
            kv_line(
                "  Sim time    ",
                &format!("{:.0} s ({:.1} min)", cfg.total_steps as f64 * dt, cfg.total_steps as f64 * dt / 60.0),
                t.text, t.text_dim,
            ),
            kv_line("  Est. RAM    ", &format!("{:.1} MB", mem_mb), t.text, t.text_dim),
            Line::from(""),
            Line::from(Span::styled(
                "  Press [Enter] to start simulation",
                Style::default().fg(t.ok).add_modifier(Modifier::BOLD),
            )),
        ]
    } else {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Invalid configuration. Go back and fix inputs.",
                Style::default().fg(t.crit),
            )),
        ]
    };

    frame.render_widget(
        Paragraph::new(summary).block(
            Block::default()
                .title(" Configuration Summary ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(frame, chunks[2], t, &["[Enter] Run", "[Esc] Back"]);
}

// ─── Screen 5: Computing ───────────────────────────────────────────────────

pub fn render_thermal_computing(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(area);

    let header = Paragraph::new(Line::from(Span::styled(
        "  THERMAL SIMULATION  \u{2014}  Running...",
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    // Progress info
    let (frac, phase_text, eta_text) = if let Some(ref task) = app.compute_task {
        let frac = task.progress.fraction();
        let eta = task.eta.format_eta(frac);
        let phase = app
            .thermal_phase
            .as_ref()
            .map(|p| p.lock().unwrap().clone())
            .unwrap_or_default();
        (frac, phase, eta)
    } else {
        (0.0, "Waiting...".to_string(), String::new())
    };

    let pct = (frac * 100.0) as usize;
    let bar_width = (area.width as usize).saturating_sub(16);
    let filled = (frac * bar_width as f64) as usize;
    let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(bar_width.saturating_sub(filled));

    let progress_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {phase_text}"),
            Style::default().fg(t.text),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(&bar, Style::default().fg(t.accent)),
            Span::styled(format!("  {pct}%"), Style::default().fg(t.text_muted)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {eta_text}"),
            Style::default().fg(t.text_dim),
        )),
    ];

    frame.render_widget(
        Paragraph::new(progress_lines).block(
            Block::default()
                .title(" Progress ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(frame, chunks[2], t, &["[Esc] Cancel"]);
}

// ─── Screen 6: Results ─────────────────────────────────────────────────────

pub fn render_thermal_results(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let result = match &app.screen {
        Screen::ThermalResults { result } => result,
        _ => return,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(10), // config + performance
            Constraint::Length(7),  // thermal dynamics
            Constraint::Length(9),  // power timeline
            Constraint::Length(10), // engineering analysis
            Constraint::Min(6),    // thermal cross-section
            Constraint::Length(3),  // footer
        ])
        .split(area);

    // ─── HEADER ───
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL SIMULATION COMPLETE",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  \u{2713}", Style::default().fg(t.ok)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    // ─── CONFIG + PERFORMANCE (split horizontal) ───
    render_config_performance(result, frame, chunks[1], t);

    // ─── THERMAL DYNAMICS ───
    render_dynamics(result, frame, chunks[2], t);

    // ─── POWER TIMELINE ───
    render_power_timeline(result, frame, chunks[3], t);

    // ─── ENGINEERING ANALYSIS ───
    render_engineering_analysis(result, frame, chunks[4], t);

    // ─── THERMAL CROSS-SECTION ───
    render_thermal_crosssection(result, frame, chunks[5], t);

    // ─── FOOTER ───
    let footer_items = vec![
        if app.thermal_csv_saved {
            Span::styled("  [S] \u{2713} Saved", Style::default().fg(t.ok))
        } else {
            Span::styled("  [S]", Style::default().fg(t.accent))
        },
        if app.thermal_csv_saved {
            Span::styled("", Style::default())
        } else {
            Span::styled(" Save CSV", Style::default().fg(t.text_muted))
        },
        if app.thermal_field_saved {
            Span::styled("   [X] \u{2713} Exported", Style::default().fg(t.ok))
        } else {
            Span::styled("   [X]", Style::default().fg(t.accent))
        },
        if app.thermal_field_saved {
            Span::styled("", Style::default())
        } else {
            Span::styled(" Export field", Style::default().fg(t.text_muted))
        },
        Span::styled("   [R]", Style::default().fg(t.accent)),
        Span::styled(" Re-run", Style::default().fg(t.text_muted)),
        Span::styled("   [Q]", Style::default().fg(t.accent)),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(footer_items))
            .style(Style::default().bg(t.surface)),
        chunks[6],
    );
}

// ─── Results Sub-panels ────────────────────────────────────────────────────

fn render_config_performance(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let cfg = &result.config;
    let total_nodes = cfg.total_nodes();
    let nnz_estimate = total_nodes * 7;
    let spmv_per_sec = if result.computation_ms > 0.0 {
        (result.total_matrix_multiplications as f64 / (result.computation_ms / 1000.0)) as usize
    } else {
        0
    };

    let config_lines = vec![
        Line::from(""),
        kv_line("  Fluid     ", &cfg.fluid.name, t.text, t.text_dim),
        kv_line(
            "  Reservoir ",
            &format!(
                "{:.0}\u{00d7}{:.0}\u{00d7}{:.0} cm",
                cfg.length_x * 100.0, cfg.length_y * 100.0, cfg.length_z * 100.0
            ),
            t.text, t.text_dim,
        ),
        kv_line("  T_initial ", &format!("{:.1}\u{00b0}C", cfg.t_initial), t.accent, t.text_dim),
        kv_line("  T_ambient ", &format!("{:.1}\u{00b0}C", cfg.t_boundary), t.text_muted, t.text_dim),
        kv_line(
            "  TEG S     ",
            &format!("{:.0}mV/K", cfg.teg.seebeck_coefficient * 1000.0),
            t.text, t.text_dim,
        ),
        kv_line("  R_int     ", &format!("{:.1}\u{03a9}", cfg.teg.internal_resistance), t.text, t.text_dim),
        kv_line("  R_load    ", &format!("{:.1}\u{03a9}", cfg.teg.load_resistance), t.text, t.text_dim),
    ];

    let perf_lines = vec![
        Line::from(""),
        kv_line(
            "  Grid      ",
            &format!("{}\u{00b3} = {} nodes", cfg.nx, total_nodes),
            t.text, t.text_dim,
        ),
        kv_line("  Matrix NNZ", &format!("~{}", nnz_estimate), t.text_muted, t.text_dim),
        kv_line(
            "  SpMV iter ",
            &format!("{}", result.total_matrix_multiplications),
            t.text, t.text_dim,
        ),
        kv_line("  Sim time  ", &format_duration(result.computation_ms), t.accent, t.text_dim),
        kv_line("  SpMV/sec  ", &format!("{}", spmv_per_sec), t.text, t.text_dim),
        kv_line(
            "  Simulated ",
            &format!(
                "{:.0}s ({:.1}min)",
                cfg.total_steps as f64 * cfg.time_step_dt,
                cfg.total_steps as f64 * cfg.time_step_dt / 60.0
            ),
            t.text, t.text_dim,
        ),
        kv_line("  dt        ", &format!("{:.4}s", cfg.time_step_dt), t.text_muted, t.text_dim),
    ];

    frame.render_widget(
        Paragraph::new(config_lines).block(
            Block::default()
                .title(" Configuration ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        mid[0],
    );
    frame.render_widget(
        Paragraph::new(perf_lines).block(
            Block::default()
                .title(" Performance ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        mid[1],
    );
}

fn render_dynamics(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let mut lines = vec![Line::from("")];
    let snaps = &result.snapshots;
    let key_indices = [
        0,
        snaps.len() / 4,
        snaps.len() / 2,
        snaps.len() * 3 / 4,
        snaps.len().saturating_sub(1),
    ];

    for &idx in &key_indices {
        if let Some(snap) = snaps.get(idx) {
            let dt_color = if snap.delta_t > 20.0 {
                t.ok
            } else if snap.delta_t > 5.0 {
                t.warn
            } else {
                t.crit
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  t={:>6.0}s  ", snap.time_s),
                    Style::default().fg(t.text_dim),
                ),
                Span::styled(
                    format!("T_center={:>5.1}\u{00b0}C  ", snap.t_center),
                    Style::default().fg(t.text),
                ),
                Span::styled(
                    format!("\u{0394}T={:>5.1}\u{00b0}C  ", snap.delta_t),
                    Style::default().fg(dt_color),
                ),
                Span::styled(format!("V={:.3}V  ", snap.voltage), Style::default().fg(t.accent)),
                Span::styled(
                    format!("P={:.1}mW", snap.power_mw),
                    Style::default().fg(t.text_muted),
                ),
            ]));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Thermal Dynamics ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

fn render_power_timeline(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let inner = Rect {
        x: area.x + 2,
        y: area.y + 1,
        width: area.width.saturating_sub(4),
        height: area.height.saturating_sub(2),
    };
    let chart_height = inner.height.saturating_sub(2) as usize;
    let chart_width = inner.width.saturating_sub(8) as usize;

    let max_v = result
        .snapshots
        .iter()
        .map(|s| s.voltage)
        .fold(0.0_f64, f64::max)
        .max(0.001);
    let threshold_v = result.threshold_voltage;

    // Build ASCII chart
    let mut grid = vec![vec![' '; chart_width]; chart_height];
    let n_snaps = result.snapshots.len();

    for col in 0..chart_width {
        let snap_idx = (col * n_snaps / chart_width.max(1)).min(n_snaps.saturating_sub(1));
        if let Some(snap) = result.snapshots.get(snap_idx) {
            let bar_height = ((snap.voltage / max_v) * chart_height as f64) as usize;
            for row in 0..bar_height.min(chart_height) {
                let grid_row = chart_height - 1 - row;
                grid[grid_row][col] = if snap.voltage >= threshold_v { '\u{2593}' } else { '\u{2591}' };
            }
        }
    }

    // Threshold line
    let threshold_row =
        chart_height.saturating_sub(((threshold_v / max_v) * chart_height as f64) as usize + 1);
    if threshold_row < chart_height {
        for col in 0..chart_width {
            if grid[threshold_row][col] == ' ' {
                grid[threshold_row][col] = '\u{2500}';
            }
        }
    }

    let block = Block::default()
        .title(format!(
            " Power Timeline  V(t) = S\u{00b7}\u{0394}T(t)  [threshold: {:.1}V] ",
            threshold_v
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border));
    frame.render_widget(block, area);

    for (row_idx, row_data) in grid.iter().enumerate() {
        let v_label = max_v * (chart_height - row_idx) as f64 / chart_height as f64;
        let label = format!("{:>4.1}V ", v_label);
        let bar: String = row_data.iter().collect();

        let line = Line::from(vec![
            Span::styled(label, Style::default().fg(t.text_dim)),
            Span::styled(bar, Style::default().fg(t.accent)),
        ]);

        let row_area = Rect {
            x: inner.x,
            y: inner.y + row_idx as u16 + 1,
            width: inner.width,
            height: 1,
        };
        if row_area.y < area.y + area.height - 1 {
            frame.render_widget(Paragraph::new(line), row_area);
        }
    }
}

fn render_engineering_analysis(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let threshold_v = result.threshold_voltage;

    let recommendation = if result.runtime_minutes < 5.0 {
        format!(
            "\u{26a0} CRITICAL: Motor runtime only {:.1}min. Consider doubling reservoir volume or adding insulation.",
            result.runtime_minutes
        )
    } else if result.runtime_minutes < 15.0 {
        format!(
            "\u{26a0} SHORT RUNTIME: {:.1}min. Optimize: larger TEG contact area or thermal insulation.",
            result.runtime_minutes
        )
    } else {
        format!(
            "\u{2713} GOOD: {:.1}min runtime. Fine-tune: check TEG load matching (R_load=R_int).",
            result.runtime_minutes
        )
    };

    let rt_color = if result.runtime_minutes > 10.0 {
        t.ok
    } else if result.runtime_minutes > 5.0 {
        t.warn
    } else {
        t.crit
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Peak voltage:  ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!(
                    "{:>6.2} V",
                    result.snapshots.first().map(|s| s.voltage).unwrap_or(0.0)
                ),
                Style::default().fg(t.ok).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("     at t=0s   (motors need \u{2265} {:.1}V)", threshold_v),
                Style::default().fg(t.text_dim),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Peak power:    ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:>6.0} mW", result.max_power_mw),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("     at t={:.0}s", result.time_to_max_power_s),
                Style::default().fg(t.text_dim),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Motor runtime: ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:>5.1} min", result.runtime_minutes),
                Style::default().fg(rt_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  (until V < {:.1}V)", threshold_v),
                Style::default().fg(t.text_dim),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Average power: ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:>6.0} mW", result.average_power_mw),
                Style::default().fg(t.text),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Total energy:  ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!(
                    "{:>8.0} mJ  = {:.2} J",
                    result.total_energy_mj,
                    result.total_energy_mj / 1000.0
                ),
                Style::default().fg(t.text),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {recommendation}"),
            Style::default().fg(if result.runtime_minutes < 5.0 {
                t.crit
            } else if result.runtime_minutes < 15.0 {
                t.warn
            } else {
                t.ok
            }),
        )),
    ];

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Engineering Analysis ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

fn render_thermal_crosssection(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let cfg = &result.config;
    let field = &result.final_field;
    let j = cfg.ny / 2; // Y-slice through center

    let t_min = cfg.t_boundary;
    let t_max = result
        .snapshots
        .first()
        .map(|s| s.t_center)
        .unwrap_or(cfg.t_initial);
    let t_range = (t_max - t_min).max(0.001);

    let mut lines = vec![Line::from(Span::styled(
        "  Z\u{2191}  (X-Z cross-section through center, final state)",
        Style::default().fg(t.text_dim),
    ))];

    let display_rows = (area.height as usize).saturating_sub(4).min(cfg.nz);
    let display_cols = (area.width as usize).saturating_sub(8).min(cfg.nx);
    let step_z = (cfg.nz / display_rows.max(1)).max(1);
    let step_x = (cfg.nx / display_cols.max(1)).max(1);

    for k_disp in (0..cfg.nz).rev().step_by(step_z).take(display_rows) {
        let mut spans = vec![Span::styled(
            format!("{:>3}  ", k_disp),
            Style::default().fg(t.text_dim),
        )];

        for i_disp in (0..cfg.nx).step_by(step_x).take(display_cols) {
            let temp = field[cfg.linear_index(i_disp, j, k_disp)];
            let normalized = ((temp - t_min) / t_range).clamp(0.0, 1.0);

            let (ch, color) = match (normalized * 5.0) as usize {
                0 => ('\u{00b7}', t.text_dim),                    // cold
                1 => ('\u{2591}', Color::Rgb(100, 150, 200)),     // cool (blue)
                2 => ('\u{2592}', t.ok),                          // moderate (green)
                3 => ('\u{2593}', t.warn),                        // warm (orange)
                _ => ('\u{2588}', t.crit),                        // hot (red)
            };
            spans.push(Span::styled(
                format!("{ch}"),
                Style::default().fg(color),
            ));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(Span::styled(
        "     \u{2192}X   [\u{00b7}=cold  \u{2591}\u{2592}\u{2593}\u{2588}=hot]",
        Style::default().fg(t.text_dim),
    )));

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Thermal Cross-Section (final) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

// ─── Footer Helper ─────────────────────────────────────────────────────────

fn render_footer(frame: &mut ratatui::Frame, area: Rect, t: &ThemeColors, items: &[&str]) {
    let mut spans = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("   ", Style::default()));
        }
        spans.push(Span::styled(
            format!(" {item} "),
            Style::default().fg(t.text_muted),
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(t.surface)),
        area,
    );
}
