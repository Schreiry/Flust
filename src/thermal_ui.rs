// ─── Thermal Simulation TUI ────────────────────────────────────────────────


use std::sync::{mpsc, Arc, Mutex};

use crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::common::{ProgressHandle, ThemeColors};
use crate::interactive::{
    App, ComputeContext, ComputeResult, ComputeTask, EtaTracker, Overlay, Screen,
    format_duration, kv_line,
};
use crate::thermal::{
    self, FluidProperties, TegProperties, TegWall, ThermalSimConfig, ThermalSimResult,
};

// ─── Fluid Catalog ────────────────────────────────────────────────────────

use crate::fluids::{self, FluidCategory, FluidEntry};

fn fluid_catalog() -> Vec<FluidEntry> {
    fluids::all_fluids()
}

fn fluid_for_index(idx: usize) -> FluidProperties {
    fluids::fluid_by_index(idx)
}

fn fluid_count() -> usize {
    fluids::fluid_count()
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

// ─── Wizard Header Helper ─────────────────────────────────────────────────

fn render_wizard_header(
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    subtitle: &str,
) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL SIMULATION",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  {subtitle}"), Style::default().fg(t.text_muted)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, area);
}

// ════════════════════════════════════════════════════════════════════════════
//  INPUT HANDLERS
// ════════════════════════════════════════════════════════════════════════════

pub fn handle_thermal_fluid_select(app: &mut App, key: KeyCode) {
    let count = fluid_count();
    match key {
        KeyCode::Up => {
            if app.thermal_fluid_idx > 0 {
                app.thermal_fluid_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.thermal_fluid_idx < count - 1 {
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
                app.screen = Screen::ThermalHeatSources;
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::ThermalGeometry;
        }
        _ => {}
    }
}

pub fn handle_thermal_solver_select(app: &mut App, key: KeyCode) {
    use crate::thermal::ThermalSolver;
    match key {
        KeyCode::Up | KeyCode::Down => {
            // Toggle between available solvers
            app.thermal_solver = match app.thermal_solver {
                ThermalSolver::NativeSparse => {
                    #[cfg(feature = "mkl")]
                    { ThermalSolver::IntelMKL }
                    #[cfg(not(feature = "mkl"))]
                    { ThermalSolver::NativeSparse }
                }
                #[cfg(feature = "mkl")]
                ThermalSolver::IntelMKL => ThermalSolver::NativeSparse,
            };
        }
        KeyCode::Enter => {
            app.screen = Screen::ThermalConfirm;
        }
        KeyCode::Esc => {
            app.screen = Screen::ThermalHeatSources;
        }
        _ => {}
    }
}

// ─── Heat Sources Wizard Step ─────────────────────────────────────────────

pub fn handle_thermal_heat_sources(app: &mut App, key: KeyCode) {
    use crate::thermal::HeatSource;

    if let Some(edit_idx) = app.thermal_hs_editing {
        // Editing a specific source — field input mode
        match key {
            KeyCode::Tab => {
                app.thermal_hs_active_field = (app.thermal_hs_active_field + 1) % 6;
            }
            KeyCode::BackTab => {
                app.thermal_hs_active_field =
                    (app.thermal_hs_active_field + 5) % 6;
            }
            KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' => {
                app.thermal_hs_fields[app.thermal_hs_active_field].push(c);
            }
            KeyCode::Backspace => {
                app.thermal_hs_fields[app.thermal_hs_active_field].pop();
            }
            KeyCode::Enter => {
                // Validate and save source
                let ok = app.thermal_hs_fields.iter().all(|f| f.parse::<f64>().is_ok());
                if ok {
                    // x/y/z fields are percentages (0-100) of geometry
                    let pct_x: f64 = app.thermal_hs_fields[0].parse().unwrap_or(50.0);
                    let pct_y: f64 = app.thermal_hs_fields[1].parse().unwrap_or(50.0);
                    let pct_z: f64 = app.thermal_hs_fields[2].parse().unwrap_or(50.0);

                    // Convert to meters using geometry fields
                    let lx: f64 = app.thermal_geometry_fields[0].parse().unwrap_or(0.15);
                    let ly: f64 = app.thermal_geometry_fields[1].parse().unwrap_or(0.08);
                    let lz: f64 = app.thermal_geometry_fields[2].parse().unwrap_or(0.06);

                    let src = HeatSource {
                        x: lx * pct_x / 100.0,
                        y: ly * pct_y / 100.0,
                        z: lz * pct_z / 100.0,
                        temperature: app.thermal_hs_fields[3].parse().unwrap_or(200.0),
                        radius: app.thermal_hs_fields[4].parse().unwrap_or(0.02),
                        omega: app.thermal_hs_fields[5].parse().unwrap_or(0.0),
                    };

                    if edit_idx < app.thermal_heat_sources.len() {
                        app.thermal_heat_sources[edit_idx] = src;
                    } else {
                        app.thermal_heat_sources.push(src);
                    }
                    app.thermal_hs_editing = None;
                }
            }
            KeyCode::Esc => {
                app.thermal_hs_editing = None; // cancel edit, return to list
            }
            _ => {}
        }
    } else {
        // List mode — browse sources
        match key {
            KeyCode::Up => {
                if app.thermal_hs_cursor > 0 {
                    app.thermal_hs_cursor -= 1;
                }
            }
            KeyCode::Down => {
                if !app.thermal_heat_sources.is_empty()
                    && app.thermal_hs_cursor < app.thermal_heat_sources.len() - 1
                {
                    app.thermal_hs_cursor += 1;
                }
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                // Add new source — enter edit mode
                app.thermal_hs_editing = Some(app.thermal_heat_sources.len());
                app.thermal_hs_fields = [
                    "50".into(), "50".into(), "50".into(),
                    "200.0".into(), "0.02".into(), "0.0".into(),
                ];
                app.thermal_hs_active_field = 0;
            }
            KeyCode::Char('e') | KeyCode::Char('E') => {
                // Edit existing source
                if !app.thermal_heat_sources.is_empty() {
                    let idx = app.thermal_hs_cursor;
                    let s = &app.thermal_heat_sources[idx];
                    let lx: f64 = app.thermal_geometry_fields[0].parse().unwrap_or(0.15);
                    let ly: f64 = app.thermal_geometry_fields[1].parse().unwrap_or(0.08);
                    let lz: f64 = app.thermal_geometry_fields[2].parse().unwrap_or(0.06);
                    app.thermal_hs_fields = [
                        format!("{:.1}", s.x / lx * 100.0),
                        format!("{:.1}", s.y / ly * 100.0),
                        format!("{:.1}", s.z / lz * 100.0),
                        format!("{:.1}", s.temperature),
                        format!("{:.4}", s.radius),
                        format!("{:.2}", s.omega),
                    ];
                    app.thermal_hs_editing = Some(idx);
                    app.thermal_hs_active_field = 0;
                }
            }
            KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Delete => {
                if !app.thermal_heat_sources.is_empty() {
                    app.thermal_heat_sources.remove(app.thermal_hs_cursor);
                    if app.thermal_hs_cursor > 0
                        && app.thermal_hs_cursor >= app.thermal_heat_sources.len()
                    {
                        app.thermal_hs_cursor -= 1;
                    }
                }
            }
            KeyCode::Enter => {
                // Proceed to solver selection
                app.screen = Screen::ThermalSolverSelect;
            }
            KeyCode::Esc => {
                app.screen = Screen::ThermalTeg;
            }
            _ => {}
        }
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
            app.screen = Screen::ThermalSolverSelect;
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
        KeyCode::Char('h') | KeyCode::Char('H') => {
            app.overlay = Overlay::ThermalHelp;
            app.thermal_overlay_scroll = 0;
        }
        KeyCode::Char('g') | KeyCode::Char('G') => {
            app.overlay = Overlay::ThermalGraph;
            app.thermal_overlay_scroll = 0;
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            app.overlay = Overlay::ThermalCrossSection2D;
            if let Screen::ThermalResults { ref result } = app.screen {
                app.thermal_cross_slice_y = result.config.ny / 2;
            }
        }
        KeyCode::Char('i') | KeyCode::Char('I') => {
            app.overlay = Overlay::ThermalIsometric;
            if let Screen::ThermalResults { ref result } = app.screen {
                app.thermal_cross_slice_y = result.config.ny / 2;
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
        boundary_type: thermal::BoundaryType::Dirichlet,
        solver: app.thermal_solver,
        heat_sources: app.thermal_heat_sources.clone(),
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
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match thermal::run_thermal_simulation(&mut config_owned, &progress_clone, &phase_clone) {
                Ok(result) => {
                    let _ = tx.send(ComputeResult::Thermal { result });
                }
                Err(e) => {
                    // BUG FIX: previously this only wrote to phase mutex without
                    // sending to channel, causing UI to hang forever.
                    let _ = tx.send(ComputeResult::Error {
                        message: format!("Thermal simulation error: {e}"),
                    });
                }
            }
        }));
        if let Err(panic_info) = outcome {
            let msg = crate::interactive::extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
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
        child_process: None,
        temp_dir: None,
        compute_request: None,
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
            Constraint::Length(3), // header
            Constraint::Min(12),  // fluid list
            Constraint::Length(3), // footer
        ])
        .split(area);

    // Header
    render_wizard_header(frame, chunks[0], t, "Step 1/5: Select Fluid");

    // Fluid list with categories, properties, and descriptions
    let catalog = fluid_catalog();
    let mut lines: Vec<Line> = Vec::new();
    let mut prev_category: Option<FluidCategory> = None;

    for (i, entry) in catalog.iter().enumerate() {
        // Category separator
        if prev_category != Some(entry.category) {
            if prev_category.is_some() {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(Span::styled(
                format!("  \u{2500}\u{2500} {} \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", entry.category.label()),
                Style::default().fg(t.text_dim),
            )));
            prev_category = Some(entry.category);
        }

        let selected = i == app.thermal_fluid_idx;
        let marker = if selected { "\u{25b6} " } else { "  " };
        let name_style = if selected {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text)
        };

        lines.push(Line::from(Span::styled(
            format!("  {marker}{}", entry.props.name),
            name_style,
        )));
        lines.push(Line::from(vec![
            Span::styled(
                format!(
                    "      \u{03bb}={:.3} W/(m\u{00b7}K)  \u{03c1}={:.0} kg/m\u{00b3}  c={:.0} J/(kg\u{00b7}K)  \u{03b1}={:.2e} m\u{00b2}/s",
                    entry.props.thermal_conductivity,
                    entry.props.density,
                    entry.props.specific_heat,
                    entry.props.thermal_diffusivity(),
                ),
                Style::default().fg(if selected { t.text_muted } else { t.text_dim }),
            ),
        ]));
        // Description line
        lines.push(Line::from(Span::styled(
            format!("      {} \u{2502} {}", entry.description, entry.source),
            Style::default().fg(t.text_dim),
        )));
    }

    // Scroll: each fluid = 3 lines (name + props + desc). Categories add 1-2 lines.
    // Approximate: selected item starts at roughly line (idx * 3 + category_offset).
    let visible_height = chunks[1].height.saturating_sub(2) as usize;
    // Count lines before selected entry by scanning
    let mut selected_line = 0_usize;
    let mut count = 0_usize;
    let mut prev_cat: Option<FluidCategory> = None;
    for entry in catalog.iter() {
        if prev_cat != Some(entry.category) {
            if prev_cat.is_some() { selected_line += 1; } // blank line
            selected_line += 1; // category header
            prev_cat = Some(entry.category);
        }
        if count == app.thermal_fluid_idx { break; }
        selected_line += 3; // name + props + desc
        count += 1;
    }

    let scroll_offset = if selected_line + 4 > visible_height {
        (selected_line + 4).saturating_sub(visible_height)
    } else {
        0
    };

    frame.render_widget(
        Paragraph::new(lines)
            .scroll((scroll_offset as u16, 0))
            .block(
                Block::default()
                    .title(format!(
                        " Select Fluid ({}/{}) ",
                        app.thermal_fluid_idx + 1,
                        fluid_count()
                    ))
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
    let fluid_name = fluid_for_index(app.thermal_fluid_idx).name;
    render_wizard_header(
        frame, chunks[0], t,
        &format!("Step 2/5: Reservoir Geometry  |  Fluid: {}", fluid_name),
    );

    // Contextual hints for each field
    const GEOM_HINTS: &[&str] = &[
        "Reservoir length along X axis [m]. Typical: 0.05\u{2013}0.5m. Larger = slower cooling",
        "Reservoir width along Y axis [m]. Typical: 0.05\u{2013}0.3m",
        "Reservoir height along Z axis [m]. Typical: 0.05\u{2013}0.3m",
        "Grid nodes per axis. 16=fast, 24=balanced, 32=precise, 48=research. RAM \u{221d} N\u{00b3}",
    ];

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
        // Show hint for the active field
        if active {
            lines.push(Line::from(Span::styled(
                format!("      \u{2514}\u{2500} {}", GEOM_HINTS[i]),
                Style::default().fg(t.text_dim),
            )));
        }
        lines.push(Line::from(""));
    }

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

    render_wizard_header(frame, chunks[0], t, "Step 3/5: TEG Parameters");

    const TEG_HINTS: &[&str] = &[
        "Voltage per degree [V/K]. Bi\u{2082}Te\u{2083}: 0.04\u{2013}0.06. Higher = more volts per \u{0394}T",
        "Module internal resistance [\u{03a9}]. Lower = more current, but more heat loss",
        "External load resistance [\u{03a9}]. R_load = R_int \u{2192} maximum power transfer",
        "TEG contact area [m\u{00b2}]. Larger area = more heat flux through module",
        "Module thickness [m]. Thinner = more heat flow, but smaller \u{0394}T across TEG",
        "TEG thermal conductivity [W/(m\u{00b7}K)]. Bi\u{2082}Te\u{2083}: 1.0\u{2013}2.0",
    ];

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
        // Show hint for active field
        if active {
            lines.push(Line::from(Span::styled(
                format!("      \u{2514}\u{2500} {}", TEG_HINTS[i]),
                Style::default().fg(t.text_dim),
            )));
        }
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

// ─── Screen: Heat Sources ──────────────────────────────────────────────────

pub fn render_thermal_heat_sources(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    use ratatui::layout::{Constraint, Direction, Layout, Margin};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph};
    use ratatui::style::{Modifier, Style};

    let title_style = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);
    let default_style = Style::default().fg(t.text).bg(t.bg);
    let selected_style = Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD);
    let muted = Style::default().fg(t.text_muted).bg(t.bg);
    let field_active = Style::default().fg(t.accent).bg(t.bg).add_modifier(Modifier::UNDERLINED);

    let outer = Block::default()
        .title(Span::styled(" HEAT SOURCES ", title_style))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.accent).bg(t.bg))
        .style(default_style);
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let content_area = Rect {
        x: inner.x + 1,
        y: inner.y,
        width: inner.width.saturating_sub(2),
        height: inner.height,
    };

    if let Some(_edit_idx) = app.thermal_hs_editing {
        // ── Edit mode: 6 fields for a single source ──
        let field_labels = [
            "X position [%]", "Y position [%]", "Z position [%]",
            "Temperature [\u{00b0}C]", "Radius [m]", "\u{03c9} [rad/s]",
        ];

        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled("  Configure heat source parameters:", muted)),
            Line::from(""),
        ];

        for (i, (label, val)) in field_labels.iter().zip(app.thermal_hs_fields.iter()).enumerate() {
            let is_active = i == app.thermal_hs_active_field;
            let cursor = if is_active { "\u{25b8} " } else { "  " };
            let style = if is_active { field_active } else { default_style };
            let val_display = if is_active {
                format!("{}_", val)
            } else {
                val.clone()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {cursor}{label:<22} "), muted),
                Span::styled(format!("[{val_display:<12}]"), style),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  [Tab] Next field  [Enter] Save  [Esc] Cancel",
            muted,
        )));

        let p = Paragraph::new(lines).style(default_style);
        frame.render_widget(p, content_area);
    } else {
        // ── List mode: show existing sources ──
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                "  Localized Gaussian heat sources (optional):",
                muted,
            )),
            Line::from(""),
        ];

        if app.thermal_heat_sources.is_empty() {
            lines.push(Line::from(Span::styled(
                "    (no sources defined \u{2014} simulation uses uniform initial temperature)",
                muted,
            )));
        } else {
            for (i, src) in app.thermal_heat_sources.iter().enumerate() {
                let is_sel = i == app.thermal_hs_cursor;
                let arrow = if is_sel { " \u{25b8} " } else { "   " };
                let style = if is_sel { selected_style } else { default_style };
                lines.push(Line::from(vec![
                    Span::styled(arrow, style),
                    Span::styled(
                        format!(
                            "Source {} \u{2014} pos=({:.3},{:.3},{:.3})m  T={:.0}\u{00b0}C  r={:.3}m  \u{03c9}={:.1}",
                            i + 1, src.x, src.y, src.z, src.temperature, src.radius, src.omega
                        ),
                        style,
                    ),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  [A]dd  [E]dit  [D]elete  [Enter] Proceed  [Esc] Back",
            muted,
        )));

        let p = Paragraph::new(lines).style(default_style);
        frame.render_widget(p, content_area);
    }
}

// ─── Screen 4: Confirmation ────────────────────────────────────────────────

// ─── Screen 4: Solver Selection ─────────────────────────────────────────────

pub fn render_thermal_solver_select(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    use crate::thermal::ThermalSolver;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(12),   // options
            Constraint::Length(3), // footer
        ])
        .split(area);

    render_wizard_header(frame, chunks[0], t, "Step 4/5: Solver");

    // Build solver options list
    let mut lines: Vec<Line> = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Select computation method for SpMV (sparse matrix \u{00d7} vector):",
            Style::default().fg(t.text),
        )),
        Line::from(""),
    ];

    // Option 1: Native Sparse
    let native_selected = matches!(app.thermal_solver, ThermalSolver::NativeSparse);
    let marker = if native_selected { "\u{25b6}" } else { " " };
    let style = if native_selected {
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(t.text_dim)
    };
    lines.push(Line::from(Span::styled(
        format!("  {marker} Native Sparse (hand-rolled CSR SpMV)"),
        style,
    )));
    lines.push(Line::from(Span::styled(
        "      Rust scalar inner loop, zero dependencies, portable",
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(""));

    // Option 2: Intel MKL (only when feature enabled)
    #[cfg(feature = "mkl")]
    {
        let mkl_selected = matches!(app.thermal_solver, ThermalSolver::IntelMKL);
        let marker = if mkl_selected { "\u{25b6}" } else { " " };
        let style = if mkl_selected {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text_dim)
        };
        lines.push(Line::from(Span::styled(
            format!("  {marker} Intel MKL Sparse BLAS (mkl_sparse_d_mv)"),
            style,
        )));
        lines.push(Line::from(Span::styled(
            "      Inspector-Executor API, auto-vectorised, optimised for Intel CPUs",
            Style::default().fg(t.text_dim),
        )));
        lines.push(Line::from(""));
    }

    #[cfg(not(feature = "mkl"))]
    {
        lines.push(Line::from(Span::styled(
            "    Intel MKL Sparse BLAS  [not available]",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::styled(
            "      Build with --features mkl to enable",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Solver ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(frame, chunks[2], t, &["[\u{2191}/\u{2193}] Select", "[Enter] Next", "[Esc] Back"]);
}

// ─── Screen 5: Confirm & Run ───────────────────────────────────────────────

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

    render_wizard_header(frame, chunks[0], t, "Step 5/5: Confirm & Run");

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
            kv_line("  Solver      ", cfg.solver.display_name(), t.accent, t.text_dim),
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
                "  \u{2500}\u{2500} What will be computed \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                "  3D heat equation via FDM (7-point stencil Laplacian)",
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                format!("  Sparse A (CSR, ~{} NNZ) \u{00d7} {} SpMV [{solver}]", total_nodes * 7, cfg.total_steps, solver = cfg.solver.display_name()),
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                "  TEG output: V(t)=S\u{00b7}\u{0394}T, P(t)=V\u{00b2}\u{00b7}R_load/(R_int+R_load)\u{00b2}",
                Style::default().fg(t.text_dim),
            )),
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
            .map(|p| p.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_default();
        (frac, phase, eta)
    } else {
        (0.0, "Waiting...".to_string(), String::new())
    };

    let pct = (frac * 100.0) as usize;
    let bar_width = (area.width as usize).saturating_sub(16);
    let filled = (frac * bar_width as f64) as usize;
    let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(bar_width.saturating_sub(filled));

    // Select gear animation frames based on progress
    let gear_idx = app.gear_frame % 4;
    let gear_lines: &[&str; 7] = if frac < 0.10 {
        &crate::interactive::GEAR_SINGLE[gear_idx]
    } else if frac < 0.50 {
        &crate::interactive::GEAR_DOUBLE[gear_idx]
    } else {
        &crate::interactive::GEAR_FRAMES[gear_idx]
    };

    let mut progress_lines = vec![
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
        Line::from(""),
    ];

    for gl in gear_lines {
        progress_lines.push(Line::from(Span::styled(
            *gl,
            Style::default().fg(t.accent),
        )));
    }

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
            Constraint::Length(12), // engineering analysis (expanded)
            Constraint::Length(9),  // narrative "What This Means"
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

    // ─── WHAT THIS MEANS FOR YOUR TANK ───
    render_thermal_narrative(result, frame, chunks[5], t);

    // ─── THERMAL CROSS-SECTION ───
    render_thermal_crosssection(result, frame, chunks[6], t);

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
        Span::styled("   [H]", Style::default().fg(t.accent)),
        Span::styled(" Help", Style::default().fg(t.text_muted)),
        Span::styled("  [G]", Style::default().fg(t.accent)),
        Span::styled(" Graph", Style::default().fg(t.text_muted)),
        Span::styled("  [C]", Style::default().fg(t.accent)),
        Span::styled(" Cross", Style::default().fg(t.text_muted)),
        Span::styled("  [I]", Style::default().fg(t.accent)),
        Span::styled(" Iso3D", Style::default().fg(t.text_muted)),
        Span::styled("  [R]", Style::default().fg(t.accent)),
        Span::styled(" Re-run", Style::default().fg(t.text_muted)),
        Span::styled("  [Q]", Style::default().fg(t.accent)),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(footer_items))
            .style(Style::default().bg(t.surface)),
        chunks[7],
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

    if chart_height == 0 || chart_width == 0 { return; }

    let max_v = result
        .snapshots
        .iter()
        .map(|s| s.voltage)
        .fold(0.0_f64, f64::max)
        .max(0.001);
    let threshold_v = result.threshold_voltage;

    // Grid stores: 'A' = above threshold bar, 'B' = below threshold bar,
    //              'T' = threshold line, ' ' = empty
    let mut grid = vec![vec![' '; chart_width]; chart_height];
    let n_snaps = result.snapshots.len();

    // Build voltage bars
    for col in 0..chart_width {
        let snap_idx = (col * n_snaps / chart_width.max(1)).min(n_snaps.saturating_sub(1));
        if let Some(snap) = result.snapshots.get(snap_idx) {
            let bar_height = ((snap.voltage / max_v) * chart_height as f64) as usize;
            let thresh_row_from_bottom =
                ((threshold_v / max_v) * chart_height as f64) as usize;
            for row in 0..bar_height.min(chart_height) {
                let grid_row = chart_height - 1 - row;
                grid[grid_row][col] = if row >= thresh_row_from_bottom { 'A' } else { 'B' };
            }
        }
    }

    // Threshold line — drawn ON TOP of bars (never hidden)
    let threshold_row =
        chart_height.saturating_sub(((threshold_v / max_v) * chart_height as f64) as usize + 1);
    if threshold_row < chart_height {
        for col in 0..chart_width {
            grid[threshold_row][col] = 'T';
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

        // Threshold row: show threshold label instead of voltage
        let label = if row_idx == threshold_row {
            format!("{:>4.1}V\u{2500}", threshold_v)
        } else {
            format!("{:>4.1}V ", v_label)
        };

        // Build per-character styled spans for the bar
        let mut spans = vec![Span::styled(
            label,
            Style::default().fg(if row_idx == threshold_row { t.crit } else { t.text_dim }),
        )];

        for &ch in row_data.iter() {
            let (display_ch, color) = match ch {
                'A' => ('\u{2593}', t.ok),    // above threshold = green (good output)
                'B' => ('\u{2591}', t.warn),   // below threshold = yellow (low output)
                'T' => ('\u{2550}', t.crit),   // threshold line = red ═
                _ => (' ', t.text_dim),         // empty
            };
            spans.push(Span::styled(
                String::from(display_ch),
                Style::default().fg(color),
            ));
        }

        let line = Line::from(spans);
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

    // Legend line at bottom of chart
    let legend_y = inner.y + chart_height as u16 + 1;
    if legend_y < area.y + area.height - 1 {
        let peak_v = result.snapshots.first().map(|s| s.voltage).unwrap_or(0.0);
        let legend = Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled("\u{2593}", Style::default().fg(t.ok)),
            Span::styled("=above threshold  ", Style::default().fg(t.text_dim)),
            Span::styled("\u{2591}", Style::default().fg(t.warn)),
            Span::styled("=below threshold  ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("Motor start: {:.1}V  |  At startup: {:.2}V  |  Runtime: {:.0}min",
                    threshold_v, peak_v, result.runtime_minutes),
                Style::default().fg(t.text_muted),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(legend),
            Rect { x: inner.x, y: legend_y, width: inner.width, height: 1 },
        );
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

    let dim = Style::default().fg(t.text_dim);

    let peak_v = result.snapshots.first().map(|s| s.voltage).unwrap_or(0.0);

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Peak voltage:  ", dim),
            Span::styled(
                format!("{:>6.2} V", peak_v),
                Style::default().fg(t.ok).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  at start  ", dim),
            Span::styled(
                format!("-> {}", voltage_motor_note(peak_v)),
                Style::default().fg(t.text_muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Peak power:    ", dim),
            Span::styled(
                format!("{:>6.0} mW", result.max_power_mw),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  at t={:.0}s  ", result.time_to_max_power_s),
                dim,
            ),
            Span::styled(
                format!("-> {}", power_consumption_note(result.max_power_mw)),
                Style::default().fg(t.text_muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Motor runtime: ", dim),
            Span::styled(
                format!("{:>5.1} min", result.runtime_minutes),
                Style::default().fg(rt_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  (until V < {:.1}V \u{2014} motor stalls)", threshold_v),
                dim,
            ),
        ]),
        Line::from(vec![
            Span::styled("  Average power: ", dim),
            Span::styled(
                format!("{:>6.0} mW", result.average_power_mw),
                Style::default().fg(t.text),
            ),
            Span::styled(
                "  (mean over full simulation, including low-output tail)",
                dim,
            ),
        ]),
        Line::from(vec![
            Span::styled("  Total energy:  ", dim),
            Span::styled(
                format!(
                    "{:>8.0} mJ  = {:.2} J",
                    result.total_energy_mj,
                    result.total_energy_mj / 1000.0
                ),
                Style::default().fg(t.text),
            ),
            Span::styled(
                "  (\u{222b}P(t)dt, trapezoidal integration)",
                dim,
            ),
        ]),
        // ── Deep Analytics Dashboard ──
        Line::from(""),
        Line::from(Span::styled(
            "  \u{2500}\u{2500}\u{2500} Deep Analytics \u{2500}\u{2500}\u{2500}",
            Style::default().fg(t.accent),
        )),
        Line::from(vec![
            Span::styled("  Peak gradient: ", dim),
            Span::styled(
                format!(
                    "{:>8.1} \u{00b0}C/m",
                    result.snapshots.last().map(|s| s.max_gradient).unwrap_or(0.0)
                ),
                Style::default().fg(t.warn).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  (max |dT/dx| via central difference, interior nodes)",
                dim,
            ),
        ]),
        Line::from(vec![
            Span::styled("  Entropy rate:  ", dim),
            Span::styled(
                format!(
                    "{:>8.4}",
                    result.snapshots.last().map(|s| s.entropy_rate).unwrap_or(0.0)
                ),
                Style::default().fg({
                    let e = result.snapshots.last().map(|s| s.entropy_rate).unwrap_or(0.0);
                    if e > 0.95 { t.ok } else if e > 0.8 { t.warn } else { t.crit }
                }).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  (normalized Shannon entropy, 1.0 = thermal equilibrium)",
                dim,
            ),
        ]),
        Line::from(vec![
            Span::styled("  Heat loss:     ", dim),
            {
                let loss = result.snapshots.last().map(|s| s.heat_loss_pct).unwrap_or(0.0);
                let bar_width = 20;
                let filled = ((loss / 100.0) * bar_width as f64).round() as usize;
                let bar: String = "\u{2588}".repeat(filled.min(bar_width))
                    + &"\u{2591}".repeat(bar_width - filled.min(bar_width));
                Span::styled(
                    format!("[{bar}] {loss:>5.1}%"),
                    Style::default().fg(if loss < 30.0 { t.ok } else if loss < 60.0 { t.warn } else { t.crit }),
                )
            },
            Span::styled(
                "  (thermal energy dissipated through boundaries)",
                dim,
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

// ─── Contextual Notes for Engineering Values ────────────────────────────────

fn voltage_motor_note(v: f64) -> &'static str {
    match (v * 10.0) as u32 {
        30..=u32::MAX => "Motors run at high speed \u{2014} optimal for movement",
        20..=29       => "Motors run at medium speed \u{2014} tank moves normally",
        10..=19       => "Motors run at low speed \u{2014} tank moves slowly",
        _             => "Below motor start threshold \u{2014} tank stalls",
    }
}

fn power_consumption_note(p_mw: f64) -> &'static str {
    match p_mw as u32 {
        800..=u32::MAX => "Enough for 2 motors + laser + controller",
        400..=799      => "Enough for 2 motors OR laser + controller",
        100..=399      => "Enough for controller and laser only",
        _              => "Insufficient for any drive component",
    }
}

// ─── "What This Means For Your Tank" Narrative ──────────────────────────────

fn generate_thermal_narrative(result: &ThermalSimResult) -> Vec<String> {
    let first = result.snapshots.first();
    let v0 = first.map(|s| s.voltage).unwrap_or(0.0);
    let dt0 = first.map(|s| s.delta_t).unwrap_or(0.0);
    let runtime = result.runtime_minutes;

    vec![
        format!(
            "At startup, the fluid is at {:.0}\u{00b0}C and ambient is {:.0}\u{00b0}C \u{2014} \
             a temperature difference of {:.1}\u{00b0}C. \
             Your TEG converts this into {:.2}V, which {}.",
            result.config.t_initial,
            result.config.t_boundary,
            dt0, v0,
            voltage_motor_note(v0)
        ),
        format!(
            "As the fluid cools, voltage drops. Based on the simulation, your tank \
             will have enough voltage (>{:.1}V) to drive motors for approximately \
             {:.0} minutes before the system stalls.",
            result.threshold_voltage, runtime
        ),
        if runtime < 5.0 {
            "Engineering suggestion: the runtime is short. Consider \
             insulating the reservoir walls (except the TEG contact face), increasing \
             fluid volume, or using a fluid with higher heat capacity such as oil.".into()
        } else if runtime < 15.0 {
            format!(
                "Engineering suggestion: {:.0} minutes is reasonable for a demonstration. \
                 For extended operation, consider a second TEG module in parallel to \
                 increase current output without changing voltage.",
                runtime
            )
        } else {
            format!(
                "Engineering assessment: {:.0} minutes is excellent for a course project. \
                 Your design is thermally well-matched to the load.",
                runtime
            )
        },
    ]
}

fn render_thermal_narrative(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let narrative = generate_thermal_narrative(result);
    let width = area.width.saturating_sub(6) as usize;
    let mut lines = vec![Line::from("")];
    for sentence in &narrative {
        let wrapped = wrap_text_thermal(sentence, width);
        for line in wrapped {
            lines.push(Line::from(Span::styled(
                format!("  {}", line),
                Style::default().fg(t.text),
            )));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(Span::styled(
                    " What This Means For Your Tank ",
                    Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent)),
        ),
        area,
    );
}

fn wrap_text_thermal(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.len() + word.len() + 1 > width && !current.is_empty() {
            lines.push(current.clone());
            current.clear();
        }
        if !current.is_empty() { current.push(' '); }
        current.push_str(word);
    }
    if !current.is_empty() { lines.push(current); }
    lines
}

/// Map normalized temperature [0,1] to (character, color) using 16-level gradient.
/// 16 levels provide significantly better visual resolution for thermal fields.
fn temp_to_style(normalized: f64) -> (char, Color) {
    const CHARS: [char; 16] = [
        ' ',        // 0: coldest
        '\u{00b7}', // 1: ·
        '\u{00b7}', // 2: ·
        '\u{2591}', // 3: ░
        '\u{2591}', // 4: ░
        '\u{2592}', // 5: ▒
        '\u{2592}', // 6: ▒
        '\u{2593}', // 7: ▓
        '\u{2593}', // 8: ▓
        '\u{2588}', // 9: █
        '\u{2588}', // 10: █
        '\u{2588}', // 11: █
        '\u{2588}', // 12: █
        '\u{2588}', // 13: █
        '\u{2588}', // 14: █
        '\u{2588}', // 15: █ (hottest)
    ];
    const COLORS: [Color; 16] = [
        Color::Rgb(20, 30, 70),    // 0: deep navy
        Color::Rgb(30, 50, 120),   // 1: dark blue
        Color::Rgb(40, 80, 160),   // 2: blue
        Color::Rgb(50, 120, 190),  // 3: med blue
        Color::Rgb(60, 160, 200),  // 4: sky blue
        Color::Rgb(70, 190, 190),  // 5: cyan
        Color::Rgb(60, 200, 140),  // 6: teal
        Color::Rgb(80, 200, 80),   // 7: green
        Color::Rgb(140, 210, 60),  // 8: lime
        Color::Rgb(200, 210, 50),  // 9: yellow-green
        Color::Rgb(230, 190, 40),  // 10: yellow
        Color::Rgb(240, 160, 30),  // 11: amber
        Color::Rgb(240, 120, 30),  // 12: orange
        Color::Rgb(230, 80, 30),   // 13: dark orange
        Color::Rgb(210, 50, 35),   // 14: red
        Color::Rgb(255, 60, 50),   // 15: bright red (hottest)
    ];
    let level = (normalized * 15.0).floor() as usize;
    let idx = level.min(15);
    (CHARS[idx], COLORS[idx])
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

    // Temperature range from actual field data for accurate mapping
    let mut t_min_actual = f64::MAX;
    let mut t_max_actual = f64::MIN;
    for k in 0..cfg.nz {
        for i in 0..cfg.nx {
            let temp = field[cfg.linear_index(i, j, k)];
            if temp < t_min_actual { t_min_actual = temp; }
            if temp > t_max_actual { t_max_actual = temp; }
        }
    }
    let t_range = (t_max_actual - t_min_actual).max(0.001);

    let mut lines = vec![Line::from(Span::styled(
        format!(
            "  Z\u{2191}  (X-Z cross-section through center, Y={}/{}, final state)",
            j, cfg.ny
        ),
        Style::default().fg(t.text_dim),
    ))];

    // Display dimensions — fill available area by upscaling small grids
    let display_rows = (area.height as usize).saturating_sub(5); // header + legend + border
    let display_cols = (area.width as usize).saturating_sub(8);   // label + border

    if display_rows == 0 || display_cols == 0 {
        frame.render_widget(
            Paragraph::new("(area too small)").block(
                Block::default()
                    .title(" Thermal Cross-Section (final) ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(t.border)),
            ),
            area,
        );
        return;
    }

    // Nearest-neighbor upscaling: map display coords → grid coords
    for disp_row in 0..display_rows {
        // Z axis: top row = highest k, bottom row = lowest k
        let k = if display_rows > 1 {
            ((display_rows - 1 - disp_row) * (cfg.nz - 1)) / (display_rows - 1)
        } else {
            cfg.nz / 2
        };

        // Z-axis label (physical coordinate in mm)
        let z_mm = (k as f64 / cfg.nz.max(1) as f64) * cfg.length_z * 1000.0;
        let mut spans = vec![Span::styled(
            format!("{:>4.0}  ", z_mm),
            Style::default().fg(t.text_dim),
        )];

        for disp_col in 0..display_cols {
            let i = if display_cols > 1 {
                (disp_col * (cfg.nx - 1)) / (display_cols - 1)
            } else {
                cfg.nx / 2
            };

            let temp = field[cfg.linear_index(i.min(cfg.nx - 1), j, k.min(cfg.nz - 1))];
            let normalized = ((temp - t_min_actual) / t_range).clamp(0.0, 1.0);
            let (ch, color) = temp_to_style(normalized);
            spans.push(Span::styled(
                String::from(ch),
                Style::default().fg(color),
            ));
        }
        lines.push(Line::from(spans));
    }

    // X-axis label + color legend with actual temperatures
    lines.push(Line::from(Span::styled(
        "      \u{2192}X [mm]",
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {:.1}\u{00b0}C ", t_min_actual),
            Style::default().fg(Color::Rgb(60, 100, 180)),
        ),
        Span::styled("\u{00b7}", Style::default().fg(Color::Rgb(60, 100, 180))),
        Span::styled("\u{2591}", Style::default().fg(Color::Rgb(80, 180, 180))),
        Span::styled("\u{2592}", Style::default().fg(Color::Rgb(80, 180, 80))),
        Span::styled("\u{2593}", Style::default().fg(Color::Rgb(200, 200, 60))),
        Span::styled("\u{2588}", Style::default().fg(Color::Rgb(220, 150, 40))),
        Span::styled("\u{2588}", Style::default().fg(Color::Rgb(200, 60, 40))),
        Span::styled("\u{2588}", Style::default().fg(Color::Rgb(255, 80, 60))),
        Span::styled(
            format!(" {:.1}\u{00b0}C", t_max_actual),
            Style::default().fg(Color::Rgb(255, 80, 60)),
        ),
    ]));

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

// ════════════════════════════════════════════════════════════════════════════
//  THERMAL OVERLAYS (H, G, C)
// ════════════════════════════════════════════════════════════════════════════

/// [H] Full reference guide for the thermal simulation.
pub fn render_thermal_help_overlay(
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    scroll: usize,
) {
    use ratatui::widgets::Clear;

    let popup_w = (area.width as f32 * 0.90) as u16;
    let popup_h = (area.height as f32 * 0.90) as u16;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(popup_w)) / 2,
        y: area.y + (area.height.saturating_sub(popup_h)) / 2,
        width: popup_w.min(area.width),
        height: popup_h.min(area.height),
    };
    frame.render_widget(Clear, popup);

    let accent = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);
    let text = Style::default().fg(t.text);
    let dim = Style::default().fg(t.text_dim);
    let muted = Style::default().fg(t.text_muted);

    let content = vec![
        Line::from(Span::styled(" THERMAL SIMULATION \u{2014} COMPLETE REFERENCE", accent)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} WORKFLOW \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" Step 1: Select fluid (Water, Oil, Glycerin, Mercury, etc.)", text)),
        Line::from(Span::styled(" Step 2: Set reservoir geometry (X\u{00d7}Y\u{00d7}Z dimensions + grid N)", text)),
        Line::from(Span::styled(" Step 3: Configure TEG module (Seebeck, resistances, contact)", text)),
        Line::from(Span::styled(" Step 4: Review & run simulation", text)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} PARAMETERS \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" \u{03bb} (lambda)    Thermal conductivity [W/(m\u{00b7}K)] \u{2014} how fast heat conducts", text)),
        Line::from(Span::styled(" \u{03c1} (rho)       Density [kg/m\u{00b3}] \u{2014} mass per unit volume", text)),
        Line::from(Span::styled(" c            Specific heat [J/(kg\u{00b7}K)] \u{2014} energy to raise 1kg by 1\u{00b0}C", text)),
        Line::from(Span::styled(" \u{03b1} (alpha)     Thermal diffusivity = \u{03bb}/(\u{03c1}\u{00b7}c) [m\u{00b2}/s]", text)),
        Line::from(Span::styled("              Higher \u{03b1} = heat spreads faster, reaches equilibrium sooner", dim)),
        Line::from(""),
        Line::from(Span::styled(" S (Seebeck)  Voltage per degree of \u{0394}T [V/K]", text)),
        Line::from(Span::styled(" R_int        Internal resistance of TEG module [\u{03a9}]", text)),
        Line::from(Span::styled(" R_load       External load resistance [\u{03a9}]. Best: R_load = R_int", text)),
        Line::from(Span::styled(" TEG area     Contact surface between TEG and reservoir wall [m\u{00b2}]", text)),
        Line::from(Span::styled(" TEG thick.   Module thickness [m]. Thinner = more heat flux", text)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} RESULTS EXPLAINED \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" T_center     Temperature at the geometric center of reservoir", text)),
        Line::from(Span::styled(" \u{0394}T           Temperature difference: T_hot \u{2212} T_ambient", text)),
        Line::from(Span::styled(" Voltage      V = S \u{00b7} \u{0394}T \u{2014} open-circuit TEG voltage", text)),
        Line::from(Span::styled(" Power        P = V\u{00b2} \u{00b7} R_load / (R_int + R_load)\u{00b2}", text)),
        Line::from(Span::styled(" Runtime      Time until voltage drops below motor threshold", text)),
        Line::from(Span::styled(" Avg Power    Mean power over entire simulation window", text)),
        Line::from(Span::styled(" Total Energy \u{222b}P(t)dt via trapezoidal integration", text)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} GRAPHS \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" Power Timeline:  V(t) = S\u{00b7}\u{0394}T(t) over simulation time", text)),
        Line::from(Span::styled("   Green bars (\u{2593}): voltage above motor threshold \u{2014} motor can run", dim)),
        Line::from(Span::styled("   Yellow bars (\u{2591}): voltage below threshold \u{2014} motor stalls", dim)),
        Line::from(Span::styled("   Red line (\u{2550}): motor threshold voltage (default 1.0V)", dim)),
        Line::from(""),
        Line::from(Span::styled(" Cross-Section:  X-Z slice through reservoir center (Y=N/2)", text)),
        Line::from(Span::styled("   8-level heat map: \u{00b7}\u{2591}\u{2592}\u{2593}\u{2588}\u{2588}\u{2588}\u{2588} from cold to hot", dim)),
        Line::from(Span::styled("   Colors: blue (cold) \u{2192} cyan \u{2192} green \u{2192} yellow \u{2192} red (hot)", dim)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} PHYSICS \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" Method: 3D Finite Difference Method (FDM)", text)),
        Line::from(Span::styled(" Heat equation: \u{2202}T/\u{2202}t = \u{03b1} \u{00b7} \u{2207}\u{00b2}T", text)),
        Line::from(Span::styled(" 7-point stencil Laplacian: 6 neighbors + center node", text)),
        Line::from(Span::styled(" Explicit Euler: T_new = A \u{00b7} T_old (sparse CSR matrix)", text)),
        Line::from(Span::styled(" Stability: dt < h\u{00b2}/(6\u{03b1}) \u{2014} auto-calculated at 80% limit", dim)),
        Line::from(Span::styled(" Boundary: Dirichlet (fixed T on walls = T_ambient)", dim)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} CSV EXPORT \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" [S] Snapshots CSV: time series (T, V, P, efficiency)", text)),
        Line::from(Span::styled("     Open in Excel/Python for plotting T(t), V(t), P(t)", dim)),
        Line::from(Span::styled(" [X] Field CSV: full 3D temperature field (x, y, z, T)", text)),
        Line::from(Span::styled("     Open in ParaView/Python for 3D visualization", dim)),
        Line::from(""),
        Line::from(Span::styled(" \u{2550}\u{2550}\u{2550} SHORTCUTS \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}", accent)),
        Line::from(Span::styled(" [S] Save snapshots CSV       [X] Export 3D field CSV", text)),
        Line::from(Span::styled(" [H] This help window         [G] Detailed graph", text)),
        Line::from(Span::styled(" [C] Full cross-section view   [R] Re-run simulation", text)),
        Line::from(Span::styled(" [Q] Return to main menu", text)),
        Line::from(""),
        Line::from(Span::styled(" Scroll: [\u{2191}\u{2193}]   Close: [Esc] or [Q]", muted)),
    ];

    let visible_h = popup.height.saturating_sub(2) as usize;
    let clamped_scroll = scroll.min(content.len().saturating_sub(visible_h));
    let visible: Vec<Line> = content
        .into_iter()
        .skip(clamped_scroll)
        .take(visible_h)
        .collect();

    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .title(" Thermal Simulation \u{2014} Help [H] ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent)),
        ),
        popup,
    );
}

/// [G] Detailed temperature and voltage graphs with Braille rendering.
pub fn render_thermal_graph_overlay(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    scroll: usize,
) {
    use ratatui::widgets::Clear;

    let popup_w = (area.width as f32 * 0.95) as u16;
    let popup_h = (area.height as f32 * 0.92) as u16;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(popup_w)) / 2,
        y: area.y + (area.height.saturating_sub(popup_h)) / 2,
        width: popup_w.min(area.width),
        height: popup_h.min(area.height),
    };
    frame.render_widget(Clear, popup);

    let snaps = &result.snapshots;
    if snaps.is_empty() {
        frame.render_widget(
            Paragraph::new("No snapshot data").block(
                Block::default()
                    .title(" Thermal Graph [G] ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(t.accent)),
            ),
            popup,
        );
        return;
    }

    // Split into two charts: Temperature (top) and Voltage+Power (bottom)
    let inner = Rect {
        x: popup.x + 1,
        y: popup.y + 1,
        width: popup.width.saturating_sub(2),
        height: popup.height.saturating_sub(2),
    };
    let chart_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),          // header
            Constraint::Percentage(48),    // temperature chart
            Constraint::Length(1),          // separator
            Constraint::Percentage(48),    // voltage/power chart
            Constraint::Length(1),          // footer
        ])
        .split(inner);

    // Block border
    frame.render_widget(
        Block::default()
            .title(" Thermal Graph [G] \u{2014} T(t) & V(t) ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.accent)),
        popup,
    );

    // Header
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" Temperature & Voltage over time", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled("    Close: [Esc]", Style::default().fg(t.text_dim)),
        ])),
        chart_chunks[0],
    );

    // ── Temperature Chart ──
    let t_max_val = snaps.iter().map(|s| s.t_center).fold(f64::MIN, f64::max);
    let t_min_val = snaps.iter().map(|s| s.t_center).fold(f64::MAX, f64::min);
    let t_range = (t_max_val - t_min_val).max(0.1);

    render_braille_chart(
        frame,
        chart_chunks[1],
        t,
        snaps,
        |s| s.t_center,
        t_min_val,
        t_max_val,
        "T_center [\u{00b0}C]",
        "\u{00b0}C",
        Color::Rgb(255, 120, 60), // warm orange for temperature
    );

    // Separator
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
            Style::default().fg(t.border),
        ))),
        chart_chunks[2],
    );

    // ── Voltage Chart ──
    let v_max = snaps.iter().map(|s| s.voltage).fold(0.0_f64, f64::max).max(0.001);

    render_braille_chart(
        frame,
        chart_chunks[3],
        t,
        snaps,
        |s| s.voltage,
        0.0,
        v_max,
        "Voltage [V]",
        "V",
        Color::Rgb(80, 200, 120), // green for voltage
    );

    // Footer
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " Close: [Esc] or [Q]",
            Style::default().fg(t.text_dim),
        ))),
        chart_chunks[4],
    );
}

/// Render a single Braille-dot chart in the given area.
fn render_braille_chart(
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    snaps: &[crate::thermal::ThermalSnapshot],
    value_fn: fn(&crate::thermal::ThermalSnapshot) -> f64,
    y_min: f64,
    y_max: f64,
    label: &str,
    unit: &str,
    color: Color,
) {
    let chart_w = area.width.saturating_sub(10) as usize; // left labels
    let chart_h = area.height.saturating_sub(2) as usize;  // top label + bottom axis

    if chart_w < 4 || chart_h < 2 || snaps.is_empty() { return; }

    let y_range = (y_max - y_min).max(0.001);
    let n = snaps.len();

    // Braille: each character cell is 2 dots wide × 4 dots tall
    // So we have chart_w*2 horizontal dots, chart_h*4 vertical dots
    let dot_w = chart_w * 2;
    let dot_h = chart_h * 4;

    // Braille grid (row-major, row 0 = top)
    let mut braille = vec![vec![false; dot_w]; dot_h];

    // Plot data points
    for dx in 0..dot_w {
        let snap_idx = (dx * n / dot_w.max(1)).min(n - 1);
        let val = value_fn(&snaps[snap_idx]);
        let dy = ((val - y_min) / y_range * (dot_h - 1) as f64).round() as usize;
        let dy = dy.min(dot_h - 1);
        let row = dot_h - 1 - dy; // invert Y
        braille[row][dx] = true;

        // Also fill one dot above and below for visibility if possible
        if row > 0 { braille[row - 1][dx] = true; }
    }

    // Convert braille grid to Unicode Braille characters
    // Braille pattern: dots numbered 1-8 in a 2×4 grid
    //   1 4
    //   2 5
    //   3 6
    //   7 8
    let mut lines: Vec<Line> = Vec::new();

    // Y-axis label
    lines.push(Line::from(vec![
        Span::styled(format!(" {label}"), Style::default().fg(t.text_muted)),
    ]));

    for char_row in 0..chart_h {
        let y_val = y_max - (char_row as f64 / chart_h.max(1) as f64) * y_range;
        let y_label = if char_row == 0 || char_row == chart_h - 1 || char_row == chart_h / 2 {
            format!("{:>7.1}{} ", y_val, unit)
        } else {
            "          ".to_string()
        };

        let mut spans = vec![Span::styled(y_label, Style::default().fg(t.text_dim))];

        let mut braille_str = String::new();
        for char_col in 0..chart_w {
            let base_row = char_row * 4;
            let base_col = char_col * 2;

            let mut code: u32 = 0x2800; // braille base
            // Map dots: (row_offset, col_offset) → bit
            if base_row < dot_h && base_col < dot_w {
                if braille[base_row][base_col] { code |= 0x01; }     // dot 1
                if base_row + 1 < dot_h && braille[base_row + 1][base_col] { code |= 0x02; }  // dot 2
                if base_row + 2 < dot_h && braille[base_row + 2][base_col] { code |= 0x04; }  // dot 3
                if base_col + 1 < dot_w && braille[base_row][base_col + 1] { code |= 0x08; }  // dot 4
                if base_row + 1 < dot_h && base_col + 1 < dot_w && braille[base_row + 1][base_col + 1] { code |= 0x10; } // dot 5
                if base_row + 2 < dot_h && base_col + 1 < dot_w && braille[base_row + 2][base_col + 1] { code |= 0x20; } // dot 6
                if base_row + 3 < dot_h && braille[base_row + 3][base_col] { code |= 0x40; }  // dot 7
                if base_row + 3 < dot_h && base_col + 1 < dot_w && braille[base_row + 3][base_col + 1] { code |= 0x80; } // dot 8
            }
            if let Some(ch) = char::from_u32(code) {
                braille_str.push(ch);
            }
        }
        spans.push(Span::styled(braille_str, Style::default().fg(color)));
        lines.push(Line::from(spans));
    }

    // X-axis: time labels
    let t_start = snaps.first().map(|s| s.time_s).unwrap_or(0.0);
    let t_end = snaps.last().map(|s| s.time_s).unwrap_or(1.0);
    let fmt_time = |s: f64| -> String {
        if s < 60.0 { format!("{:.0}s", s) }
        else if s < 3600.0 { format!("{:.1}m", s / 60.0) }
        else { format!("{:.1}h", s / 3600.0) }
    };
    lines.push(Line::from(vec![
        Span::styled("          ", Style::default()),
        Span::styled(fmt_time(t_start), Style::default().fg(t.text_dim)),
        Span::styled(
            format!("{:>width$}", fmt_time(t_end), width = chart_w.saturating_sub(fmt_time(t_start).len())),
            Style::default().fg(t.text_dim),
        ),
    ]));

    let visible_h = area.height as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(Paragraph::new(visible), area);
}

/// [C] Full-screen 2D thermal cross-section with free-camera viewport.
///
/// Viewport2D: WASD pan, +/- zoom, arrow keys move cursor, Tab/BackTab cycle Y-slice.
/// Cursor position shows exact temperature in a tooltip overlay.
/// 16-level gradient for high-resolution thermal visualization.
pub fn render_thermal_crosssection_overlay(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    slice_y: usize,
) {
    render_thermal_crosssection_viewport(result, frame, area, t, slice_y, 0.0, 0.0, 1.0, 0, 0);
}

/// Internal renderer with full viewport parameters, called from the overlay system.
pub fn render_thermal_crosssection_viewport(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    slice_y: usize,
    cam_x: f64,
    cam_y: f64,
    zoom: f64,
    cursor_x: usize,
    cursor_z: usize,
) {
    use ratatui::widgets::Clear;

    let popup_w = (area.width as f32 * 0.95) as u16;
    let popup_h = (area.height as f32 * 0.92) as u16;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(popup_w)) / 2,
        y: area.y + (area.height.saturating_sub(popup_h)) / 2,
        width: popup_w.min(area.width),
        height: popup_h.min(area.height),
    };
    frame.render_widget(Clear, popup);

    let cfg = &result.config;
    let field = &result.final_field;
    let j = slice_y.min(cfg.ny.saturating_sub(1));
    let zoom = zoom.clamp(0.25, 8.0);

    // Temperature range for this slice
    let mut t_min = f64::MAX;
    let mut t_max = f64::MIN;
    for k in 0..cfg.nz {
        for i in 0..cfg.nx {
            let temp = field[cfg.linear_index(i, j, k)];
            if temp < t_min { t_min = temp; }
            if temp > t_max { t_max = temp; }
        }
    }
    let t_range = (t_max - t_min).max(0.001);

    let inner_h = popup.height.saturating_sub(6) as usize;
    let inner_w = popup.width.saturating_sub(10) as usize;

    let mut lines: Vec<Line> = Vec::new();

    // Header with viewport info
    lines.push(Line::from(vec![
        Span::styled(
            format!(
                " X-Z Cross  Y={}/{} ({:.1}mm)  zoom:{:.1}×  pan:({:.0},{:.0})",
                j, cfg.ny,
                (j as f64 / cfg.ny.max(1) as f64) * cfg.length_y * 1000.0,
                zoom, cam_x, cam_y,
            ),
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
    ]));

    // Cursor tooltip — show exact temperature
    let cursor_x_clamped = cursor_x.min(cfg.nx.saturating_sub(1));
    let cursor_z_clamped = cursor_z.min(cfg.nz.saturating_sub(1));
    let cursor_temp = field[cfg.linear_index(cursor_x_clamped, j, cursor_z_clamped)];
    let cursor_x_mm = (cursor_x_clamped as f64 / cfg.nx.max(1) as f64) * cfg.length_x * 1000.0;
    let cursor_z_mm = (cursor_z_clamped as f64 / cfg.nz.max(1) as f64) * cfg.length_z * 1000.0;
    lines.push(Line::from(vec![
        Span::styled(
            format!(
                " T({:.1}mm,{:.1}mm) = ",
                cursor_x_mm, cursor_z_mm,
            ),
            Style::default().fg(t.text_muted),
        ),
        Span::styled(
            format!("{:.2}°C", cursor_temp),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  |  Range: {:.1}°C — {:.1}°C  Grid: {}×{}", t_min, t_max, cfg.nx, cfg.nz),
            Style::default().fg(t.text_dim),
        ),
    ]));

    // Heat map with viewport transformation
    if inner_h > 0 && inner_w > 0 {
        // Viewport: cam_x/cam_y shift the center, zoom scales the visible range
        let center_x = (cfg.nx as f64 / 2.0) + cam_x;
        let center_z = (cfg.nz as f64 / 2.0) + cam_y;
        let half_w = (cfg.nx as f64 / 2.0) / zoom;
        let half_h = (cfg.nz as f64 / 2.0) / zoom;
        let view_x_min = center_x - half_w;
        let view_x_max = center_x + half_w;
        let view_z_min = center_z - half_h;
        let view_z_max = center_z + half_h;

        for disp_row in 0..inner_h {
            // Z axis: top row = highest z, bottom row = lowest z
            let z_frac = if inner_h > 1 {
                1.0 - (disp_row as f64 / (inner_h - 1) as f64)
            } else {
                0.5
            };
            let z_world = view_z_min + z_frac * (view_z_max - view_z_min);
            let k = (z_world.round() as isize).clamp(0, cfg.nz.saturating_sub(1) as isize) as usize;

            let z_mm = (k as f64 / cfg.nz.max(1) as f64) * cfg.length_z * 1000.0;
            let mut spans = vec![Span::styled(
                format!("{:>5.1}mm ", z_mm),
                Style::default().fg(t.text_dim),
            )];

            for disp_col in 0..inner_w {
                let x_frac = if inner_w > 1 {
                    disp_col as f64 / (inner_w - 1) as f64
                } else {
                    0.5
                };
                let x_world = view_x_min + x_frac * (view_x_max - view_x_min);
                let i = (x_world.round() as isize).clamp(0, cfg.nx.saturating_sub(1) as isize) as usize;

                // Is this the cursor position?
                let is_cursor = i == cursor_x_clamped && k == cursor_z_clamped;

                let temp = field[cfg.linear_index(i, j, k)];
                let normalized = ((temp - t_min) / t_range).clamp(0.0, 1.0);
                let (ch, color) = temp_to_style(normalized);

                if is_cursor {
                    // Cursor: inverted highlight with crosshair
                    spans.push(Span::styled(
                        "╋",
                        Style::default().fg(Color::White).bg(color).add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::styled(
                        String::from(ch),
                        Style::default().fg(color),
                    ));
                }
            }
            lines.push(Line::from(spans));
        }
    }

    // 16-level color legend
    lines.push(Line::from(vec![
        Span::styled("       →X ", Style::default().fg(t.text_dim)),
        Span::styled(format!("{:.1}°C ", t_min), Style::default().fg(Color::Rgb(20, 30, 70))),
        Span::styled("·", Style::default().fg(Color::Rgb(40, 80, 160))),
        Span::styled("░", Style::default().fg(Color::Rgb(60, 160, 200))),
        Span::styled("▒", Style::default().fg(Color::Rgb(80, 200, 80))),
        Span::styled("▓", Style::default().fg(Color::Rgb(200, 210, 50))),
        Span::styled("█", Style::default().fg(Color::Rgb(240, 120, 30))),
        Span::styled("█", Style::default().fg(Color::Rgb(210, 50, 35))),
        Span::styled("█", Style::default().fg(Color::Rgb(255, 60, 50))),
        Span::styled(format!(" {:.1}°C", t_max), Style::default().fg(Color::Rgb(255, 60, 50))),
    ]));

    // Footer
    lines.push(Line::from(vec![
        Span::styled(" [WASD] Pan", Style::default().fg(t.text_muted)),
        Span::styled("  [+/-] Zoom", Style::default().fg(t.text_muted)),
        Span::styled("  [↑↓←→] Cursor", Style::default().fg(t.text_muted)),
        Span::styled("  [Tab] Y-slice", Style::default().fg(t.text_muted)),
        Span::styled("  [Esc] Close", Style::default().fg(t.text_muted)),
    ]));

    let visible_h = popup.height.saturating_sub(2) as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .title(" Thermal Cross-Section [C] — Free Camera ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent)),
        ),
        popup,
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  THERMAL CSV VIEWER
// ════════════════════════════════════════════════════════════════════════════

pub fn handle_thermal_viewer(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Up | KeyCode::Char('k') => {
            app.thermal_viewer_scroll = app.thermal_viewer_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.thermal_viewer_scroll = app.thermal_viewer_scroll.saturating_add(1);
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            app.thermal_view_data = None;
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

pub fn render_thermal_viewer(
    app: &App,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let data = match &app.thermal_view_data {
        Some(d) => d,
        None => {
            frame.render_widget(
                Paragraph::new("No thermal data loaded"),
                area,
            );
            return;
        }
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(8),    // config summary
            Constraint::Min(10),   // data table
            Constraint::Length(3), // footer
        ])
        .split(area);

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "  THERMAL CSV VIEWER",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {}  |  {} snapshots", app.viewer_filename, data.snapshots.len()),
            Style::default().fg(t.text_muted),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.border)),
    );
    frame.render_widget(header, chunks[0]);

    // Config summary from CSV header
    let mut config_lines: Vec<Line> = vec![Line::from("")];
    for (key, val) in &data.config_lines {
        config_lines.push(kv_line(
            &format!("  {:<16}", key),
            val,
            t.text,
            t.text_dim,
        ));
    }

    frame.render_widget(
        Paragraph::new(config_lines)
            .scroll((0, 0))
            .block(
                Block::default()
                    .title(" Configuration (from CSV header) ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(t.border)),
            ),
        chunks[1],
    );

    // Data table
    let visible_rows = chunks[2].height.saturating_sub(4) as usize;
    let scroll = app.thermal_viewer_scroll.min(
        data.snapshots.len().saturating_sub(visible_rows),
    );

    let mut table_lines: Vec<Line> = Vec::new();
    // Column headers
    table_lines.push(Line::from(Span::styled(
        "  Time [s]    Step   T_center  \u{0394}T [K]   V [V]    P [mW]    Eff [%]   T_mean    T_max     T_min",
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    table_lines.push(Line::from(Span::styled(
        "  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
        Style::default().fg(t.border),
    )));

    for snap in data.snapshots.iter().skip(scroll).take(visible_rows) {
        table_lines.push(Line::from(Span::styled(
            format!(
                "  {:>9.1}  {:>5}  {:>8.2}\u{00b0}C  {:>6.2}   {:>6.3}  {:>8.2}   {:>6.2}%  {:>8.2}\u{00b0}  {:>8.2}\u{00b0}  {:>8.2}\u{00b0}",
                snap.time_s,
                snap.step,
                snap.t_center,
                snap.delta_t,
                snap.voltage,
                snap.power_mw,
                snap.efficiency_pct,
                snap.mean_temp,
                snap.max_temp,
                snap.min_temp,
            ),
            Style::default().fg(t.text),
        )));
    }

    frame.render_widget(
        Paragraph::new(table_lines).block(
            Block::default()
                .title(format!(
                    " Snapshot Data ({}\u{2013}{} of {}) ",
                    scroll + 1,
                    (scroll + visible_rows).min(data.snapshots.len()),
                    data.snapshots.len(),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[2],
    );

    // Footer
    render_footer(
        frame,
        chunks[3],
        t,
        &["[\u{2191}\u{2193}] Scroll", "[Q] Back"],
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

// ─── Pseudo-3D Isometric Projection ──────────────────────────────────────

/// Temperature → RGB color via 5-stop heat gradient:
///   0.0 → blue(70,130,180)
///   0.25 → cyan(0,200,200)
///   0.50 → green(50,205,50)
///   0.75 → yellow(255,200,0)
///   1.00 → red(255,50,30)
fn heat_color(norm: f64) -> Color {
    let n = norm.clamp(0.0, 1.0);
    let (r, g, b) = if n < 0.25 {
        let t = n / 0.25;
        (lerp8(70, 0, t), lerp8(130, 200, t), lerp8(180, 200, t))
    } else if n < 0.5 {
        let t = (n - 0.25) / 0.25;
        (lerp8(0, 50, t), lerp8(200, 205, t), lerp8(200, 50, t))
    } else if n < 0.75 {
        let t = (n - 0.5) / 0.25;
        (lerp8(50, 255, t), lerp8(205, 200, t), lerp8(50, 0, t))
    } else {
        let t = (n - 0.75) / 0.25;
        (lerp8(255, 255, t), lerp8(200, 50, t), lerp8(0, 30, t))
    };
    Color::Rgb(r, g, b)
}

/// Darker shade for side faces — reduces brightness by ~40%.
fn heat_color_dark(norm: f64) -> Color {
    let n = norm.clamp(0.0, 1.0);
    let (r, g, b) = if n < 0.25 {
        let t = n / 0.25;
        (lerp8(70, 0, t), lerp8(130, 200, t), lerp8(180, 200, t))
    } else if n < 0.5 {
        let t = (n - 0.25) / 0.25;
        (lerp8(0, 50, t), lerp8(200, 205, t), lerp8(200, 50, t))
    } else if n < 0.75 {
        let t = (n - 0.5) / 0.25;
        (lerp8(50, 255, t), lerp8(205, 200, t), lerp8(50, 0, t))
    } else {
        let t = (n - 0.75) / 0.25;
        (lerp8(255, 255, t), lerp8(200, 50, t), lerp8(0, 30, t))
    };
    Color::Rgb(
        (r as f64 * 0.6) as u8,
        (g as f64 * 0.6) as u8,
        (b as f64 * 0.6) as u8,
    )
}

#[inline]
fn lerp8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u8
}

/// Render a pseudo-3D isometric projection of the thermal field.
///
/// Takes a 2D slice T(x,z) at y=slice_y from the 3D field.
/// Temperature acts as the Z (height) axis.
/// Uses painter's algorithm (back-to-front) with block characters,
/// rendering into a virtual char/color buffer then emitting as Paragraph lines.
pub fn render_thermal_isometric_overlay(
    result: &ThermalSimResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    slice_y: usize,
) {
    use ratatui::widgets::Clear;

    // Popup dimensions
    let popup_w = (area.width as f32 * 0.95) as u16;
    let popup_h = (area.height as f32 * 0.92) as u16;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(popup_w)) / 2,
        y: area.y + (area.height.saturating_sub(popup_h)) / 2,
        width: popup_w.min(area.width),
        height: popup_h.min(area.height),
    };
    frame.render_widget(Clear, popup);

    let outer = Block::default()
        .title(Span::styled(
            format!(
                " ISOMETRIC 3D  [y={}/{}]  [\u{2190}\u{2192}] Slice ",
                slice_y,
                result.config.ny.saturating_sub(1)
            ),
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.accent).bg(t.bg))
        .style(Style::default().bg(t.bg));
    let inner = outer.inner(popup);
    frame.render_widget(outer, popup);

    if inner.width < 10 || inner.height < 6 {
        return;
    }

    let cfg = &result.config;
    let field = &result.final_field;
    if field.is_empty() {
        return;
    }

    let j = slice_y.min(cfg.ny.saturating_sub(1));
    let nx = cfg.nx;
    let nz = cfg.nz;

    // Extract 2D slice T(x,z) and find temperature range
    let mut slice_data = vec![0.0_f64; nx * nz];
    let mut t_min = f64::MAX;
    let mut t_max = f64::MIN;
    for ix in 0..nx {
        for iz in 0..nz {
            let temp = field[cfg.linear_index(ix, j, iz)];
            slice_data[ix * nz + iz] = temp;
            if temp < t_min { t_min = temp; }
            if temp > t_max { t_max = temp; }
        }
    }
    let t_range = (t_max - t_min).max(1e-9);

    // Virtual canvas (char + fg color), filled with spaces
    let cw = inner.width as usize;
    let ch = (inner.height as usize).saturating_sub(1); // reserve 1 row for footer
    let mut canvas_ch: Vec<char> = vec![' '; cw * ch];
    let mut canvas_fg: Vec<Color> = vec![t.bg; cw * ch];

    // Isometric projection parameters
    let max_height = (ch as f64 * 0.4).max(2.0);

    // Cell spacing — fit the diamond footprint into the canvas
    // Isometric diamond: width = (nx + nz) * cell_w, height = (nx + nz) * cell_h / 2
    let cell_w = (cw as f64 / (nx + nz).max(1) as f64).max(1.0).min(3.0);
    let cell_h_iso = (ch as f64 / ((nx + nz) as f64 * 0.5 + max_height)).max(0.3).min(1.5);

    // Origin — center the diamond in the canvas
    let origin_col = cw as f64 / 2.0;
    let origin_row = max_height * cell_h_iso + 1.0;

    // Painter's algorithm: iterate back-to-front
    for ix in 0..nx {
        for iz in 0..nz {
            let norm = (slice_data[ix * nz + iz] - t_min) / t_range;
            let height = norm * max_height * cell_h_iso;

            // Isometric: x axis goes down-right, z axis goes down-left
            let sc = origin_col + (ix as f64 - iz as f64) * cell_w;
            let sr = origin_row + (ix as f64 + iz as f64) * cell_h_iso * 0.5 - height;

            let col = sc.round() as i32;
            let top_row = sr.round() as i32;
            let base_row =
                (origin_row + (ix as f64 + iz as f64) * cell_h_iso * 0.5).round() as i32;

            let top_color = heat_color(norm);
            let side_color = heat_color_dark(norm);

            // Draw top face
            let cw_i = (cell_w.ceil() as i32).max(1);
            for dc in 0..cw_i {
                let c = col + dc;
                if c >= 0 && (c as usize) < cw && top_row >= 0 && (top_row as usize) < ch {
                    let idx = top_row as usize * cw + c as usize;
                    canvas_ch[idx] = if dc == 0 { '\u{2588}' } else { '\u{2593}' }; // █ ▓
                    canvas_fg[idx] = top_color;
                }
            }

            // Draw front face (column from top+1 down to base)
            for row in (top_row + 1)..=base_row {
                if row >= 0 && (row as usize) < ch {
                    for dc in 0..cw_i {
                        let c = col + dc;
                        if c >= 0 && (c as usize) < cw {
                            let idx = row as usize * cw + c as usize;
                            canvas_ch[idx] = if dc == 0 { '\u{2588}' } else { '\u{2591}' }; // █ ░
                            canvas_fg[idx] = side_color;
                        }
                    }
                }
            }
        }
    }

    // Convert virtual canvas to Paragraph lines
    let mut lines: Vec<Line> = Vec::with_capacity(ch);
    for row in 0..ch {
        let mut spans: Vec<Span> = Vec::new();
        let mut run_start = 0;
        while run_start < cw {
            // Find run of same color
            let color = canvas_fg[row * cw + run_start];
            let mut run_end = run_start + 1;
            while run_end < cw && canvas_fg[row * cw + run_end] == color {
                run_end += 1;
            }
            let s: String = canvas_ch[row * cw + run_start..row * cw + run_end]
                .iter()
                .collect();
            spans.push(Span::styled(s, Style::default().fg(color).bg(t.bg)));
            run_start = run_end;
        }
        lines.push(Line::from(spans));
    }

    // Footer line
    lines.push(Line::from(vec![
        Span::styled(
            " [\u{2190}\u{2192}] Y-slice  ",
            Style::default().fg(t.text_muted).bg(t.bg),
        ),
        Span::styled("[Esc] Close ", Style::default().fg(t.text_muted).bg(t.bg)),
        Span::styled(
            format!(" T: {:.1}\u{00b0}C \u{2014} {:.1}\u{00b0}C ", t_min, t_max),
            Style::default().fg(t.accent).bg(t.bg),
        ),
    ]));

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(t.bg)),
        inner,
    );
}
