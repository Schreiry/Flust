// =============================================================================
//  MHPS — Material Heat Propagation Simulator — TUI Wizard & Results
// =============================================================================

use std::sync::{mpsc, Arc, Mutex};

use crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::common::{ProgressHandle, ThemeColors};
use crate::interactive::{
    App, ComputeContext, ComputeResult, ComputeTask, EtaTracker, Screen,
    format_duration, kv_line, GEAR_FRAMES, GEAR_SINGLE,
};
use crate::mhps::{self, HeatSource, HeatSourceType, ShapeDefinition, MATERIALS, MhpsConfig, MhpsGeometry, MhpsResult, MhpsSnapshot};

// ─── Wizard Field Labels ─────────────────────────────────────────────────────

const GEO_LABELS: &[&str] = &[
    "Length X (mm)",
    "Length Y (mm)",
    "Thickness (mm)",
    "Grid N (nodes per axis)",
    "Grid NZ (1=2D)",
];

const SIM_LABELS: &[&str] = &[
    "Initial temp (C)",
    "Ambient temp (C)",
    "Convection h (W/m2K, 0=off)",
    "Simulation time (s)",
    "Convergence eps (0=off)",
];

// ─── Screen 1: Material Select ───────────────────────────────────────────────

pub fn handle_mhps_material(app: &mut App, key: KeyCode) {
    let n = MATERIALS.len();
    match key {
        KeyCode::Up => {
            app.mhps_material_idx = app.mhps_material_idx.saturating_sub(1);
        }
        KeyCode::Down => {
            if app.mhps_material_idx < n - 1 {
                app.mhps_material_idx += 1;
            }
        }
        KeyCode::Enter => {
            app.screen = Screen::MhpsGeometry;
            app.mhps_active_field = 0;
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

pub fn render_mhps_material(
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
            Constraint::Length(6),
            Constraint::Length(1),
        ])
        .split(area);

    // Header
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  MATERIAL HEAT PROPAGATION — Select Material",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    // Material list
    let sel = app.mhps_material_idx;
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    // Scrolling window
    let visible = (chunks[1].height as usize).saturating_sub(2);
    let scroll = if sel >= visible { sel - visible + 1 } else { 0 };

    for (i, mat) in MATERIALS.iter().enumerate().skip(scroll).take(visible) {
        let prefix = if i == sel { " > " } else { "   " };
        let color = if i == sel { t.accent } else { t.text };
        let alpha = mat.thermal_diffusivity();
        lines.push(Line::from(vec![
            Span::styled(prefix, Style::default().fg(color)),
            Span::styled(
                format!("{:<20}", mat.name),
                Style::default()
                    .fg(color)
                    .add_modifier(if i == sel { Modifier::BOLD } else { Modifier::empty() }),
            ),
            Span::styled(
                format!("  lambda={:.1}  alpha={:.2e}", mat.thermal_conductivity, alpha),
                Style::default().fg(t.text_dim),
            ),
        ]));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Materials ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // Detail panel for selected material
    let mat = &MATERIALS[sel];
    let detail = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {} — {}", mat.name, mat.description),
            Style::default().fg(t.text),
        )),
        Line::from(Span::styled(
            format!(
                "  lambda={:.1} W/(m*K)  rho={:.0} kg/m3  c={:.0} J/(kg*K)  {}",
                mat.thermal_conductivity,
                mat.density,
                mat.specific_heat,
                mat.melting_point_c
                    .map(|mp| format!("Tm={:.0}C", mp))
                    .unwrap_or_else(|| "no melting point".into()),
            ),
            Style::default().fg(t.text_muted),
        )),
        Line::from(Span::styled(
            format!("  {}", mat.characteristic()),
            Style::default().fg(t.text_dim),
        )),
    ];
    frame.render_widget(
        Paragraph::new(detail).block(
            Block::default()
                .title(" Properties ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[2],
    );

    render_footer(&["[Enter] Select", "[Esc] Back"], frame, chunks[3], t);
}

// ─── Screen 2: Geometry ──────────────────────────────────────────────────────

const SHAPE_NAMES: &[&str] = &["Rectangle", "Polygon", "Rounded Rect", "L-Shape"];

pub fn handle_mhps_geometry(app: &mut App, key: KeyCode) {
    let n_fields = GEO_LABELS.len();
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.mhps_active_field = (app.mhps_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.mhps_active_field = if app.mhps_active_field == 0 {
                n_fields - 1
            } else {
                app.mhps_active_field - 1
            };
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => {
            active_geo_field_mut(app).push(c);
        }
        KeyCode::Backspace => {
            active_geo_field_mut(app).pop();
        }
        KeyCode::F(1) => app.mhps_shape_idx = 0, // Rectangle
        KeyCode::F(2) => app.mhps_shape_idx = 1, // Polygon
        KeyCode::F(3) => app.mhps_shape_idx = 2, // Rounded Rect
        KeyCode::F(4) => app.mhps_shape_idx = 3, // L-Shape
        KeyCode::Enter => {
            app.screen = Screen::MhpsHeatSources;
            app.mhps_active_field = 0;
        }
        KeyCode::Esc => {
            app.screen = Screen::MhpsMaterial;
        }
        _ => {}
    }
}

fn active_geo_field_mut(app: &mut App) -> &mut String {
    match app.mhps_active_field {
        0 => &mut app.mhps_geo_fields[0],
        1 => &mut app.mhps_geo_fields[1],
        2 => &mut app.mhps_geo_fields[2],
        3 => &mut app.mhps_geo_fields[3],
        4 => &mut app.mhps_geo_fields[4],
        _ => &mut app.mhps_geo_fields[0],
    }
}

pub fn render_mhps_geometry(
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
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  MATERIAL HEAT PROPAGATION — Geometry",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    let mat = &MATERIALS[app.mhps_material_idx];
    lines.push(Line::from(Span::styled(
        format!("  Material: {}", mat.name),
        Style::default().fg(t.text_muted),
    )));
    lines.push(Line::from(""));

    // Shape type selector
    lines.push(Line::from(Span::styled(
        "  Shape type:",
        Style::default().fg(t.text_muted),
    )));
    for (i, name) in SHAPE_NAMES.iter().enumerate() {
        let prefix = if i == app.mhps_shape_idx { " > " } else { "   " };
        let color = if i == app.mhps_shape_idx { t.accent } else { t.text };
        let fkey_label = format!("F{}", i + 1);
        lines.push(Line::from(vec![
            Span::styled(prefix, Style::default().fg(color)),
            Span::styled(
                format!("[{}] {}", fkey_label, name),
                Style::default()
                    .fg(color)
                    .add_modifier(if i == app.mhps_shape_idx { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }
    lines.push(Line::from(""));

    // Dimension fields
    for (i, label) in GEO_LABELS.iter().enumerate() {
        let is_active = i == app.mhps_active_field;
        let val = &app.mhps_geo_fields[i];
        let cursor = if is_active { "_" } else { "" };
        let color = if is_active { t.accent } else { t.text };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<28}", label),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(
                format!("{}{}", val, cursor),
                Style::default()
                    .fg(color)
                    .add_modifier(if is_active { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }

    // Preview
    lines.push(Line::from(""));
    if let (Ok(lx), Ok(ly), Ok(thick), Ok(n)) = (
        app.mhps_geo_fields[0].parse::<f64>(),
        app.mhps_geo_fields[1].parse::<f64>(),
        app.mhps_geo_fields[2].parse::<f64>(),
        app.mhps_geo_fields[3].parse::<usize>(),
    ) {
        let (nx, ny, _nz_preview) = crate::mhps::proportional_grid(
            lx / 1000.0, ly / 1000.0, thick / 1000.0, n, false,
        );
        let nodes = nx * ny;
        lines.push(Line::from(Span::styled(
            format!(
                "  Grid: {}x{} = {} nodes   hx={:.3}mm  hy={:.3}mm",
                nx,
                ny,
                nodes,
                lx / nx.max(1) as f64,
                ly / ny.max(1) as f64,
            ),
            Style::default().fg(t.text_dim),
        )));
        lines.push(Line::from(Span::styled(
            format!("  Plate: {:.0}x{:.0}mm, t={:.1}mm  Shape: {}", lx, ly, thick, SHAPE_NAMES[app.mhps_shape_idx]),
            Style::default().fg(t.text_dim),
        )));

        // ASCII shape preview (small)
        if nx >= 4 && ny >= 4 {
            let preview_w = 40usize.min(nx);
            let preview_h = 8usize.min(ny);
            let lx_m = lx / 1000.0;
            let ly_m = ly / 1000.0;
            let mut geo_preview = MhpsGeometry::uniform(lx_m, ly_m, thick / 1000.0, nx, ny, 0);
            geo_preview.shape_definition = build_shape_definition(app);
            geo_preview.rebuild_mask();

            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("  Preview:", Style::default().fg(t.text_dim))));
            let step_x = (nx / preview_w).max(1);
            let step_y = (ny / preview_h).max(1);
            for sy in 0..preview_h {
                let nj = sy * step_y;
                if nj >= ny { break; }
                let mut row = String::from("  ");
                for sx in 0..preview_w {
                    let ni = sx * step_x;
                    if ni >= nx { break; }
                    row.push(if geo_preview.is_void(ni, nj) { ' ' } else { '\u{2591}' });
                }
                lines.push(Line::from(Span::styled(row, Style::default().fg(t.accent))));
            }
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Geometry (mm) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(
        &["[Tab] Next field", "[F1-F4] Shape", "[Enter] Continue", "[Esc] Back"],
        frame,
        chunks[2],
        t,
    );
}

fn build_shape_definition(app: &App) -> ShapeDefinition {
    match app.mhps_shape_idx {
        1 => {
            let verts: Vec<(f64, f64)> = app
                .mhps_polygon_verts
                .iter()
                .filter_map(|(xs, ys)| {
                    let x = xs.parse::<f64>().ok()?;
                    let y = ys.parse::<f64>().ok()?;
                    Some((x, y))
                })
                .collect();
            if verts.len() >= 3 {
                ShapeDefinition::Polygon(verts)
            } else {
                ShapeDefinition::Rectangle
            }
        }
        2 => {
            let rx = app.mhps_shape_params[0].parse::<f64>().unwrap_or(0.02) / 1000.0;
            let ry = app.mhps_shape_params[1].parse::<f64>().unwrap_or(0.02) / 1000.0;
            ShapeDefinition::RoundedRect {
                radius_x: rx,
                radius_y: ry,
            }
        }
        3 => {
            let cx = app.mhps_shape_params[0].parse::<f64>().unwrap_or(50.0) / 1000.0;
            let cy = app.mhps_shape_params[1].parse::<f64>().unwrap_or(50.0) / 1000.0;
            ShapeDefinition::LShape {
                cutout_x: cx,
                cutout_y: cy,
            }
        }
        _ => ShapeDefinition::Rectangle,
    }
}

// ─── Screen 3: Heat Sources ──────────────────────────────────────────────────

pub fn handle_mhps_heat_sources(app: &mut App, key: KeyCode) {
    let n_fields = 4; // name, position_x%, position_y%, temperature
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.mhps_active_field = (app.mhps_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.mhps_active_field = if app.mhps_active_field == 0 {
                n_fields - 1
            } else {
                app.mhps_active_field - 1
            };
        }
        KeyCode::Char(c) => {
            let is_name = app.mhps_active_field == 0;
            let field = active_hs_field_mut(app);
            if is_name {
                // Name — allow alphanumeric
                if c.is_ascii_alphanumeric() && field.len() < 8 {
                    field.push(c);
                }
            } else if (c.is_ascii_digit() || c == '.' || c == '-') && field.len() < 10 {
                field.push(c);
            }
        }
        KeyCode::Backspace => {
            active_hs_field_mut(app).pop();
        }
        KeyCode::Enter => {
            // Try to add the source
            if try_add_mhps_source(app) {
                // Reset fields for next source
                app.mhps_hs_name.clear();
                let count = app.mhps_heat_sources.len();
                app.mhps_hs_name = format!("H{}", count + 1);
                app.mhps_hs_x.clear();
                app.mhps_hs_x.push_str("50");
                app.mhps_hs_y.clear();
                app.mhps_hs_y.push_str("50");
                app.mhps_hs_temp.clear();
                app.mhps_hs_temp.push_str("100");
                app.mhps_active_field = 0;
            }
        }
        KeyCode::Char('n') | KeyCode::Char('N') if app.mhps_active_field == 0 && app.mhps_hs_name.is_empty() => {
            // N with empty name — skip adding, proceed to confirm
        }
        KeyCode::F(2) => {
            // F2 = done adding sources, go to sim params
            app.screen = Screen::MhpsSimParams;
            app.mhps_active_field = 0;
        }
        KeyCode::Delete => {
            // Delete last source
            app.mhps_heat_sources.pop();
        }
        KeyCode::Esc => {
            app.screen = Screen::MhpsGeometry;
        }
        _ => {}
    }
}

fn active_hs_field_mut(app: &mut App) -> &mut String {
    match app.mhps_active_field {
        0 => &mut app.mhps_hs_name,
        1 => &mut app.mhps_hs_x,
        2 => &mut app.mhps_hs_y,
        3 => &mut app.mhps_hs_temp,
        _ => &mut app.mhps_hs_name,
    }
}

fn try_add_mhps_source(app: &mut App) -> bool {
    let name = app.mhps_hs_name.trim().to_string();
    if name.is_empty() {
        return false;
    }
    let x_pct: f64 = match app.mhps_hs_x.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let y_pct: f64 = match app.mhps_hs_y.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let temp: f64 = match app.mhps_hs_temp.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };

    // We'll compute actual grid coords at config-build time
    let src = HeatSource {
        name,
        source_type: HeatSourceType::FixedTemperature,
        position_i: x_pct as usize,  // store as percentage for now
        position_j: y_pct as usize,
        position_k: 0,               // resolved at config-build time
        radius_nodes: 1,
        temperature_c: temp,
        is_active: true,
    };
    app.mhps_heat_sources.push(src);
    true
}

pub fn render_mhps_heat_sources(
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
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  MATERIAL HEAT PROPAGATION — Heat Sources",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    // Current sources list
    let mut list_lines: Vec<Line> = Vec::new();
    list_lines.push(Line::from(""));
    if app.mhps_heat_sources.is_empty() {
        list_lines.push(Line::from(Span::styled(
            "  No sources yet. Add at least one.",
            Style::default().fg(t.text_dim),
        )));
    } else {
        for (i, src) in app.mhps_heat_sources.iter().enumerate() {
            let label = if src.temperature_c > app.mhps_sim_fields[0].parse::<f64>().unwrap_or(20.0) {
                "HOT"
            } else {
                "COLD"
            };
            let color = if label == "HOT" {
                Color::Rgb(248, 113, 113)
            } else {
                Color::Rgb(100, 180, 255)
            };
            list_lines.push(Line::from(vec![
                Span::styled(format!("  {}. ", i + 1), Style::default().fg(t.text_dim)),
                Span::styled(
                    format!("{:<8}", src.name),
                    Style::default().fg(t.text).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        "  pos=({:.0}%, {:.0}%)  T={:.0}C  [{}]",
                        src.position_i, src.position_j, src.temperature_c, label
                    ),
                    Style::default().fg(color),
                ),
            ]));
        }
    }

    frame.render_widget(
        Paragraph::new(list_lines).block(
            Block::default()
                .title(format!(" Sources ({}) ", app.mhps_heat_sources.len()))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // New source form
    let hs_labels = ["Name", "Position X %", "Position Y %", "Temperature (C)"];
    let hs_vals = [
        &app.mhps_hs_name,
        &app.mhps_hs_x,
        &app.mhps_hs_y,
        &app.mhps_hs_temp,
    ];
    let mut form_lines: Vec<Line> = Vec::new();
    form_lines.push(Line::from(""));
    for (i, (label, val)) in hs_labels.iter().zip(hs_vals.iter()).enumerate() {
        let is_active = i == app.mhps_active_field;
        let cursor = if is_active { "_" } else { "" };
        let color = if is_active { t.accent } else { t.text };
        form_lines.push(Line::from(vec![
            Span::styled(format!("  {:<20}", label), Style::default().fg(t.text_muted)),
            Span::styled(
                format!("{}{}", val, cursor),
                Style::default()
                    .fg(color)
                    .add_modifier(if is_active { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }

    frame.render_widget(
        Paragraph::new(form_lines).block(
            Block::default()
                .title(" Add Source ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border_focus)),
        ),
        chunks[2],
    );

    render_footer(
        &["[Enter] Add source", "[F2] Done", "[Del] Remove last", "[Esc] Continue"],
        frame,
        chunks[3],
        t,
    );
}

// ─── Screen 4: Simulation Parameters ─────────────────────────────────────────

pub fn handle_mhps_sim_params(app: &mut App, key: KeyCode) {
    let n_fields = SIM_LABELS.len();
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.mhps_active_field = (app.mhps_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.mhps_active_field = if app.mhps_active_field == 0 {
                n_fields - 1
            } else {
                app.mhps_active_field - 1
            };
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' => {
            active_sim_field_mut(app).push(c);
        }
        KeyCode::Backspace => {
            active_sim_field_mut(app).pop();
        }
        KeyCode::Enter => {
            app.screen = Screen::MhpsConfirm;
        }
        KeyCode::Esc => {
            app.screen = Screen::MhpsHeatSources;
            app.mhps_active_field = 0;
        }
        _ => {}
    }
}

fn active_sim_field_mut(app: &mut App) -> &mut String {
    match app.mhps_active_field {
        0 => &mut app.mhps_sim_fields[0],
        1 => &mut app.mhps_sim_fields[1],
        2 => &mut app.mhps_sim_fields[2],
        3 => &mut app.mhps_sim_fields[3],
        4 => &mut app.mhps_sim_fields[4],
        _ => &mut app.mhps_sim_fields[0],
    }
}

pub fn render_mhps_sim_params(
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
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  MATERIAL HEAT PROPAGATION — Simulation Parameters",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    for (i, label) in SIM_LABELS.iter().enumerate() {
        let is_active = i == app.mhps_active_field;
        let val = &app.mhps_sim_fields[i];
        let cursor = if is_active { "_" } else { "" };
        let color = if is_active { t.accent } else { t.text };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<32}", label),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(
                format!("{}{}", val, cursor),
                Style::default()
                    .fg(color)
                    .add_modifier(if is_active { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ]));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Parameters ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(
        &["[Tab] Next field", "[Enter] Review", "[Esc] Back"],
        frame,
        chunks[2],
        t,
    );
}

// ─── Screen 5: Confirm ──────────────────────────────────────────────────────

pub fn handle_mhps_confirm(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => {
            if let Some(config) = build_mhps_config(app) {
                launch_mhps_simulation(app, config);
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::MhpsSimParams;
            app.mhps_active_field = 0;
        }
        _ => {}
    }
}

pub fn render_mhps_confirm(
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
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  MATERIAL HEAT PROPAGATION — Confirm & Run",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let mat = &MATERIALS[app.mhps_material_idx];
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(kv_line("  Material", mat.name, t.text, t.text_muted));
    lines.push(kv_line(
        "  Size",
        &format!(
            "{}x{}mm, t={}mm",
            app.mhps_geo_fields[0], app.mhps_geo_fields[1], app.mhps_geo_fields[2]
        ),
        t.text,
        t.text_muted,
    ));
    lines.push(kv_line(
        "  Grid N",
        &app.mhps_geo_fields[3],
        t.text,
        t.text_muted,
    ));
    lines.push(kv_line(
        "  Sources",
        &format!("{}", app.mhps_heat_sources.len()),
        t.text,
        t.text_muted,
    ));
    for (i, src) in app.mhps_heat_sources.iter().enumerate() {
        lines.push(kv_line(
            &format!("    #{}", i + 1),
            &format!(
                "{} ({:.0}%,{:.0}%) T={:.0}C",
                src.name, src.position_i, src.position_j, src.temperature_c
            ),
            t.text_dim,
            t.text_muted,
        ));
    }
    lines.push(Line::from(""));
    lines.push(kv_line("  T_initial", &format!("{}C", app.mhps_sim_fields[0]), t.text, t.text_muted));
    lines.push(kv_line("  T_ambient", &format!("{}C", app.mhps_sim_fields[1]), t.text, t.text_muted));
    lines.push(kv_line("  Convection h", &format!("{} W/m2K", app.mhps_sim_fields[2]), t.text, t.text_muted));
    lines.push(kv_line("  Sim time", &format!("{}s", app.mhps_sim_fields[3]), t.text, t.text_muted));
    lines.push(kv_line("  Epsilon", &app.mhps_sim_fields[4], t.text, t.text_muted));

    // Memory estimate
    if let Ok(n) = app.mhps_geo_fields[3].parse::<usize>() {
        let lx: f64 = app.mhps_geo_fields[0].parse().unwrap_or(100.0) / 1000.0;
        let ly: f64 = app.mhps_geo_fields[1].parse().unwrap_or(100.0) / 1000.0;
        let thick: f64 = app.mhps_geo_fields[2].parse().unwrap_or(10.0) / 1000.0;
        let (nx, ny, nz) = crate::mhps::proportional_grid(lx, ly, thick, n, false);
        let nodes = nx * ny * nz;
        let mem_mb = (nodes * 8 * 7) as f64 / (1024.0 * 1024.0); // rough: field + matrix entries
        lines.push(Line::from(""));
        lines.push(kv_line(
            "  Est. memory",
            &format!("{:.1} MB ({} nodes)", mem_mb, nodes),
            t.text_dim,
            t.text_muted,
        ));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Configuration Preview ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(
        &["[Enter] RUN", "[Esc] Back"],
        frame,
        chunks[2],
        t,
    );
}

// ─── Config Builder ──────────────────────────────────────────────────────────

fn build_mhps_config(app: &App) -> Option<MhpsConfig> {
    let lx_mm: f64 = app.mhps_geo_fields[0].parse().ok()?;
    let ly_mm: f64 = app.mhps_geo_fields[1].parse().ok()?;
    let thick_mm: f64 = app.mhps_geo_fields[2].parse().ok()?;
    let n: usize = app.mhps_geo_fields[3].parse::<f64>().ok()? as usize;
    let nz_input: usize = app.mhps_geo_fields[4].parse::<f64>().ok().map(|v| (v as usize).max(1)).unwrap_or(1);

    if n < 4 || n > 500 || lx_mm <= 0.0 || ly_mm <= 0.0 || thick_mm <= 0.0 {
        return None;
    }

    let lx = lx_mm / 1000.0; // mm -> m
    let ly = ly_mm / 1000.0;
    let thick = thick_mm / 1000.0;
    let enable_3d = nz_input > 1;
    let (nx, ny, nz) = crate::mhps::proportional_grid(lx, ly, thick, n, enable_3d);
    // If user explicitly set nz, override proportional nz
    let nz = if enable_3d { nz } else { 1 };

    let mut geometry = MhpsGeometry::uniform_3d(lx, ly, thick, nx, ny, nz, app.mhps_material_idx);
    geometry.shape_definition = build_shape_definition(app);
    geometry.rebuild_mask();

    // Convert heat sources from percentage to grid coords
    let heat_sources: Vec<HeatSource> = app
        .mhps_heat_sources
        .iter()
        .map(|src| {
            let i = ((src.position_i as f64 / 100.0) * (nx - 1) as f64).round() as usize;
            let j = ((src.position_j as f64 / 100.0) * (ny - 1) as f64).round() as usize;
            HeatSource {
                name: src.name.clone(),
                source_type: src.source_type,
                position_i: i.min(nx - 1),
                position_j: j.min(ny - 1),
                position_k: nz / 2, // center of slab
                radius_nodes: 1,
                temperature_c: src.temperature_c,
                is_active: true,
            }
        })
        .collect();

    let t_initial: f64 = app.mhps_sim_fields[0].parse().unwrap_or(20.0);
    let t_ambient: f64 = app.mhps_sim_fields[1].parse().unwrap_or(20.0);
    let convection_h: f64 = app.mhps_sim_fields[2].parse().unwrap_or(0.0);
    let total_time: f64 = app.mhps_sim_fields[3].parse().unwrap_or(60.0);
    let epsilon: f64 = app.mhps_sim_fields[4].parse().unwrap_or(0.01);

    if total_time <= 0.0 {
        return None;
    }

    Some(MhpsConfig {
        geometry,
        heat_sources,
        t_initial_c: t_initial,
        t_ambient_c: t_ambient,
        convection_h: convection_h.max(0.0),
        time_step_dt: 0.0, // auto
        total_time_s: total_time,
        save_every_n_steps: 50,
        convergence_epsilon: epsilon.max(0.0),
    })
}

// ─── Simulation Launch ───────────────────────────────────────────────────────

fn launch_mhps_simulation(app: &mut App, config: MhpsConfig) {
    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let phase = Arc::new(Mutex::new("Initializing...".to_string()));
    let phase_clone = phase.clone();
    let (tx, rx) = mpsc::channel();

    let mut config_owned = config;
    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match mhps::run_mhps(&mut config_owned, &progress_clone, &phase_clone) {
                Ok(result) => {
                    let _ = tx.send(ComputeResult::Mhps { result });
                }
                Err(e) => {
                    let _ = tx.send(ComputeResult::Error {
                        message: format!("MHPS simulation error: {e}"),
                    });
                }
            }
        }));
        if let Err(panic_info) = outcome {
            let msg = crate::interactive::extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    app.mhps_phase = Some(phase);
    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context: ComputeContext {
            algorithm_choice: crate::interactive::AlgorithmChoice::Naive,
            algorithm_name: "MHPS Heat Propagation".into(),
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

    app.screen = Screen::MhpsComputing;
}

// ─── Screen 6: Computing ─────────────────────────────────────────────────────

pub fn render_mhps_computing(
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
            "  MATERIAL HEAT PROPAGATION — Computing",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let (frac, phase_text, eta_text): (f64, String, String) = if let Some(ref task) = app.compute_task
    {
        let f = task.progress.fraction();
        let ph = app
            .mhps_phase
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
    let bar = format!(
        "  [{}>{}] {:.1}%",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(bar_w.saturating_sub(filled)),
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
                .title(" FDM Solver ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(&["[Esc] Cancel"], frame, chunks[2], t);
}

// ─── Screen 7: Results ───────────────────────────────────────────────────────

pub fn handle_mhps_results(app: &mut App, key: KeyCode) {
    let n_snaps = if let Screen::MhpsResults { ref result } = app.screen {
        result.snapshots.len()
    } else {
        1
    };

    match key {
        // View mode tabs [1]-[6]
        KeyCode::Char('1') => app.mhps_view_mode = 0, // 2D Heatmap
        KeyCode::Char('2') => app.mhps_view_mode = 1, // Gradient
        KeyCode::Char('3') => app.mhps_view_mode = 2, // ASCII 3D
        KeyCode::Char('4') => app.mhps_view_mode = 3, // Timeline
        KeyCode::Char('5') => app.mhps_view_mode = 4, // Statistics
        KeyCode::Char('6') => app.mhps_view_mode = 5, // Cross-section
        // Cross-section axis switching (when in cross-section view)
        KeyCode::Char('x') | KeyCode::Char('X') if app.mhps_view_mode == 5 => {
            app.mhps_cross_axis = 0; // XY
            app.mhps_cross_pos = 0;
        }
        KeyCode::Char('y') | KeyCode::Char('Y') if app.mhps_view_mode == 5 => {
            app.mhps_cross_axis = 1; // XZ
            app.mhps_cross_pos = 0;
        }
        KeyCode::Char('z') | KeyCode::Char('Z') if app.mhps_view_mode == 5 => {
            app.mhps_cross_axis = 2; // YZ
            app.mhps_cross_pos = 0;
        }
        KeyCode::Char('+') | KeyCode::Char('=') if app.mhps_view_mode == 5 => {
            app.mhps_cross_pos = app.mhps_cross_pos.saturating_add(1);
        }
        KeyCode::Char('-') if app.mhps_view_mode == 5 => {
            app.mhps_cross_pos = app.mhps_cross_pos.saturating_sub(1);
        }
        // Time scrubbing
        KeyCode::Left => {
            app.mhps_snapshot_idx = app.mhps_snapshot_idx.saturating_sub(1);
        }
        KeyCode::Right => {
            if app.mhps_snapshot_idx + 1 < n_snaps {
                app.mhps_snapshot_idx += 1;
            }
        }
        // Export
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if !app.mhps_csv_saved {
                if let Screen::MhpsResults { ref result } = app.screen {
                    let path = format!(
                        "flust_mhps_history_{}.csv",
                        crate::io::timestamp_now().replace(':', "-")
                    );
                    if mhps::export_mhps_history_csv(result, &path).is_ok() {
                        app.mhps_csv_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('e') | KeyCode::Char('E') => {
            if let Screen::MhpsResults { ref result } = app.screen {
                let bundle = crate::mhps_export::MhpsExportBundle::new();
                let _ = bundle.export_all(result, ".");
                app.mhps_csv_saved = true;
                app.mhps_field_saved = true;
            }
        }
        KeyCode::Char('p') | KeyCode::Char('P') => {
            if let Screen::MhpsResults { ref result } = app.screen {
                let bundle = crate::mhps_export::MhpsExportBundle::new();
                let _ = bundle.export_python_script(result, ".");
            }
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.screen = Screen::MhpsConfirm;
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

const VIEW_TAB_NAMES: &[&str] = &[
    "[1] 2D Heat", "[2] Gradient", "[3] 3D View", "[4] Timeline", "[5] Stats", "[6] Cross-Sec",
];

pub fn render_mhps_results(
    app: &App,
    result: &MhpsResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // header
            Constraint::Length(3),   // tab bar
            Constraint::Min(10),    // main view (changes by mode)
            Constraint::Length(5),  // quick physics
            Constraint::Length(1),  // footer
        ])
        .split(area);

    // Header
    let mat = &MATERIALS[result.config.geometry.material_map[0].min(MATERIALS.len() - 1)];
    let geo = &result.config.geometry;
    let snap_info = result
        .snapshots
        .get(app.mhps_snapshot_idx)
        .map(|s| format!("  t={:.1}s", s.time_s))
        .unwrap_or_default();
    let conv_mark = if result.converged { "  \u{2713}" } else { "" };
    let status = Span::styled(
        format!(
            "  MHPS — {} {:.0}x{:.0}x{:.1}mm  {}x{}  [{}/{}]{}{}",
            mat.name,
            geo.length_x * 1000.0,
            geo.length_y * 1000.0,
            geo.base_thickness * 1000.0,
            geo.nx,
            geo.ny,
            app.mhps_snapshot_idx + 1,
            result.snapshots.len(),
            snap_info,
            conv_mark,
        ),
        Style::default()
            .fg(if result.converged { Color::Rgb(74, 222, 128) } else { t.accent })
            .add_modifier(Modifier::BOLD),
    );
    frame.render_widget(Paragraph::new(Line::from(status)), chunks[0]);

    // Tab bar
    render_view_tab_bar(frame, chunks[1], app.mhps_view_mode, t);

    // Main view
    match app.mhps_view_mode {
        0 => {
            // 2D Heatmap with current snapshot
            let field = get_display_field(result, app.mhps_snapshot_idx);
            render_mhps_heatmap(frame, chunks[2], field, &result.config, false, t);
        }
        1 => {
            // Gradient map
            let field = get_display_field(result, app.mhps_snapshot_idx);
            render_mhps_heatmap(frame, chunks[2], field, &result.config, true, t);
        }
        2 => {
            // ASCII 3D view
            render_ascii_3d_view(frame, chunks[2], result, app, t);
        }
        3 => {
            // Timeline chart
            render_mhps_timeline(frame, chunks[2], &result.snapshots, t);
        }
        4 => {
            // Statistics
            render_full_statistics(frame, chunks[2], result, t);
        }
        5 => {
            // Cross-section (horizontal at middle)
            render_cross_section(frame, chunks[2], result, app, t);
        }
        _ => {
            let field = get_display_field(result, app.mhps_snapshot_idx);
            render_mhps_heatmap(frame, chunks[2], field, &result.config, false, t);
        }
    }

    // Quick physics analysis
    render_mhps_analysis(frame, chunks[3], result, t);

    // Footer
    let save_label = if app.mhps_csv_saved { "[S] Saved" } else { "[S] CSV" };
    let export_label = if app.mhps_field_saved { "[E] Exported" } else { "[E] Export all" };
    render_footer(
        &["[1-6] View", "[<>] Scrub", save_label, export_label, "[R] Re-run", "[Q] Back"],
        frame,
        chunks[4],
        t,
    );
}

fn render_view_tab_bar(frame: &mut ratatui::Frame, area: Rect, active: usize, t: &ThemeColors) {
    let mut spans: Vec<Span> = Vec::new();
    for (i, label) in VIEW_TAB_NAMES.iter().enumerate() {
        if i == active {
            spans.push(Span::styled(
                *label,
                Style::default()
                    .fg(t.bg)
                    .bg(t.accent)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(*label, Style::default().fg(t.text_muted)));
        }
        spans.push(Span::styled("  ", Style::default()));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

fn get_display_field<'a>(result: &'a MhpsResult, snapshot_idx: usize) -> &'a [f64] {
    result
        .snapshots
        .get(snapshot_idx)
        .and_then(|s| s.field.as_deref())
        .unwrap_or(&result.final_field)
}

fn render_ascii_3d_view(
    frame: &mut ratatui::Frame,
    area: Rect,
    result: &MhpsResult,
    app: &App,
    t: &ThemeColors,
) {
    let geo = &result.config.geometry;
    if geo.nz <= 1 {
        // 2D object — show heatmap with note
        let field = get_display_field(result, app.mhps_snapshot_idx);
        render_mhps_heatmap(frame, area, field, &result.config, false, t);
        if area.height > 2 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    " 2D geometry \u{2014} 3D view not applicable (nz=1). Showing 2D map. ",
                    Style::default().fg(t.text_dim),
                )),
                Rect {
                    x: area.x + 1,
                    y: area.y + area.height - 2,
                    width: area.width.saturating_sub(2),
                    height: 1,
                },
            );
        }
        return;
    }

    // For nz > 1: show multiple layers with isometric offset
    let field = get_display_field(result, app.mhps_snapshot_idx);
    let t_min = result.snapshots.iter().map(|s| s.min_temp_c).fold(f64::INFINITY, f64::min);
    let t_max = result.snapshots.iter().map(|s| s.max_temp_c).fold(f64::NEG_INFINITY, f64::max);
    let temp_range = (t_max - t_min).max(0.001);
    let temp_range = if temp_range.is_finite() { temp_range } else { 1.0 };
    let max_layers = 8usize.min(geo.nz);
    let layer_step = (geo.nz / max_layers).max(1);
    let iso_dx = 2u16;
    let iso_dy = 1u16;

    // Render border FIRST so layer content doesn't get overwritten
    let block = Block::default()
        .title(format!(" ASCII 3D View \u{2014} {} layers shown ", max_layers))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let map_w = inner.width.saturating_sub(iso_dx * max_layers as u16) as usize;
    let map_h = inner.height.saturating_sub(iso_dy * max_layers as u16) as usize;
    let step_x = (geo.nx / map_w.max(1)).max(1);
    let step_y = (geo.ny / map_h.max(1)).max(1);

    for layer_display in 0..max_layers {
        let k = layer_display * layer_step;
        let offset_x = (max_layers - 1 - layer_display) as u16 * iso_dx;
        let offset_y = (max_layers - 1 - layer_display) as u16 * iso_dy;

        for (sy, nj) in (0..geo.ny).step_by(step_y).enumerate() {
            let row_y = inner.y + offset_y + sy as u16;
            if row_y >= inner.y + inner.height { break; }

            let mut spans = Vec::new();
            for (_sx, ni) in (0..geo.nx).step_by(step_x).enumerate() {
                // Use 3D void check
                if geo.is_void3(ni, nj, k) {
                    spans.push(Span::raw(" "));
                    continue;
                }
                let idx = geo.idx3(ni, nj, k);
                let temp = if idx < field.len() { field[idx] } else { t_min };
                let norm = ((temp - t_min) / temp_range).clamp(0.0, 1.0);
                let level = (norm * 7.0) as usize;
                let alpha = 0.5 + 0.5 * (k as f64 / geo.nz.max(1) as f64);
                let base = HEAT_COLORS[level.min(7)];
                let c = (
                    (base.0 as f64 * alpha) as u8,
                    (base.1 as f64 * alpha) as u8,
                    (base.2 as f64 * alpha) as u8,
                );
                spans.push(Span::styled(
                    HEAT_CHARS[level.min(7)].to_string(),
                    Style::default().fg(Color::Rgb(c.0, c.1, c.2)),
                ));
            }

            let render_x = inner.x + offset_x;
            if render_x < inner.x + inner.width {
                frame.render_widget(
                    Paragraph::new(Line::from(spans)),
                    Rect { x: render_x, y: row_y, width: map_w as u16, height: 1 },
                );
            }
        }
    }
}

/// Extract a 2D slice from a 3D (or 2D) field.
/// Returns (slice_data, slice_rows, slice_cols) where data[row * cols + col] = temperature.
fn extract_slice(
    field: &[f64],
    geo: &MhpsGeometry,
    axis: u8,
    pos: usize,
) -> (Vec<f64>, usize, usize) {
    if geo.nz <= 1 {
        // 2D mode: always return XY plane
        let mut slice = Vec::with_capacity(geo.nx * geo.ny);
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                let idx = geo.idx(i, j);
                slice.push(if idx < field.len() { field[idx] } else { f64::NAN });
            }
        }
        return (slice, geo.nx, geo.ny);
    }
    match axis {
        0 => {
            // XY slice at z=pos
            let k = pos.min(geo.nz.saturating_sub(1));
            let mut slice = Vec::with_capacity(geo.nx * geo.ny);
            for i in 0..geo.nx {
                for j in 0..geo.ny {
                    let idx = geo.idx3(i, j, k);
                    let val = if idx < field.len() && !geo.is_void3(i, j, k) {
                        field[idx]
                    } else { f64::NAN };
                    slice.push(val);
                }
            }
            (slice, geo.nx, geo.ny)
        }
        1 => {
            // XZ slice at y=pos
            let j = pos.min(geo.ny.saturating_sub(1));
            let mut slice = Vec::with_capacity(geo.nx * geo.nz);
            for i in 0..geo.nx {
                for k in 0..geo.nz {
                    let idx = geo.idx3(i, j, k);
                    let val = if idx < field.len() && !geo.is_void3(i, j, k) {
                        field[idx]
                    } else { f64::NAN };
                    slice.push(val);
                }
            }
            (slice, geo.nx, geo.nz)
        }
        _ => {
            // YZ slice at x=pos
            let i = pos.min(geo.nx.saturating_sub(1));
            let mut slice = Vec::with_capacity(geo.ny * geo.nz);
            for j in 0..geo.ny {
                for k in 0..geo.nz {
                    let idx = geo.idx3(i, j, k);
                    let val = if idx < field.len() && !geo.is_void3(i, j, k) {
                        field[idx]
                    } else { f64::NAN };
                    slice.push(val);
                }
            }
            (slice, geo.ny, geo.nz)
        }
    }
}

fn render_cross_section(
    frame: &mut ratatui::Frame,
    area: Rect,
    result: &MhpsResult,
    app: &App,
    t: &ThemeColors,
) {
    let field = get_display_field(result, app.mhps_snapshot_idx);
    let geo = &result.config.geometry;

    let axis = app.mhps_cross_axis;
    let pos = app.mhps_cross_pos;

    let axis_label = match axis {
        0 => "XY",
        1 => "XZ",
        _ => "YZ",
    };
    let pos_mm = match axis {
        0 => pos as f64 * geo.hz() * 1000.0,
        1 => pos as f64 * geo.hy() * 1000.0,
        _ => pos as f64 * geo.hx() * 1000.0,
    };
    let max_pos = match axis {
        0 => geo.nz.saturating_sub(1),
        1 => geo.ny.saturating_sub(1),
        _ => geo.nx.saturating_sub(1),
    };

    let title = format!(
        " Cross-Section {} at {:.1}mm ({}/{}) [X/Y/Z] axis [+/-] pos ",
        axis_label, pos_mm, pos.min(max_pos), max_pos
    );
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 4 || inner.height < 2 {
        return;
    }

    let (slice, slice_rows, slice_cols) = extract_slice(field, geo, axis, pos);

    // Normalization
    let valid: Vec<f64> = slice.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        frame.render_widget(Paragraph::new("  No active nodes in slice"), inner);
        return;
    }
    let t_min_s = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max_s = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let temp_range = (t_max_s - t_min_s).max(0.001);
    let temp_range = if temp_range.is_finite() { temp_range } else { 1.0 };

    let map_h = inner.height as usize;
    let map_w = inner.width as usize;
    let step_r = (slice_rows as f64 / map_h as f64).max(1.0);
    let step_c = (slice_cols as f64 / map_w as f64).max(1.0);

    for row in 0..map_h {
        let sr = ((row as f64) * step_r) as usize;
        if sr >= slice_rows { break; }

        let mut spans = Vec::with_capacity(map_w);
        for col in 0..map_w {
            let sc = ((col as f64) * step_c) as usize;
            if sc >= slice_cols {
                spans.push(Span::raw(" "));
                continue;
            }
            let val = slice[sr * slice_cols + sc];
            if !val.is_finite() {
                spans.push(Span::styled("\u{25AA}", Style::default().fg(Color::Rgb(40, 40, 40))));
            } else {
                let norm = ((val - t_min_s) / temp_range).clamp(0.0, 1.0);
                let level = (norm * 7.0) as usize;
                let (r, g, b) = HEAT_COLORS[level.min(7)];
                spans.push(Span::styled(
                    HEAT_CHARS[level.min(7)].to_string(),
                    Style::default().fg(Color::Rgb(r, g, b)),
                ));
            }
        }

        let y = inner.y + row as u16;
        if y < inner.y + inner.height {
            frame.render_widget(
                Paragraph::new(Line::from(spans)),
                Rect { x: inner.x, y, width: inner.width, height: 1 },
            );
        }
    }
}

// ─── Heatmap Renderer ────────────────────────────────────────────────────────

const HEAT_CHARS: &[char] = &[' ', '\u{00B7}', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}', '\u{2593}', '\u{2593}'];
const HEAT_COLORS: [(u8, u8, u8); 8] = [
    (30, 30, 60),    // cold blue-ish
    (50, 80, 150),   // cool blue
    (80, 150, 200),  // moderate cyan
    (100, 200, 100), // warm green
    (220, 200, 80),  // hot yellow
    (240, 140, 40),  // very hot orange
    (220, 60, 20),   // critical red
    (255, 20, 20),   // danger bright red
];

fn render_mhps_heatmap(
    frame: &mut ratatui::Frame,
    area: Rect,
    field: &[f64],
    config: &MhpsConfig,
    show_gradient: bool,
    t: &ThemeColors,
) {
    let geo = &config.geometry;
    let inner = area.inner(&ratatui::layout::Margin::new(1, 1));

    // If showing gradient, compute gradient field
    let grad_field: Vec<f64>;
    let display_field: &[f64];
    let (t_min, t_max);

    if show_gradient {
        let mut gf = vec![0.0f64; geo.nx * geo.ny];
        let mut max_g = 0.0f64;
        if geo.nx > 2 && geo.ny > 2 {
            for i in 1..geo.nx - 1 {
                for j in 1..geo.ny - 1 {
                    if !geo.is_void(i, j) {
                        let gx = (field[geo.idx(i + 1, j)] - field[geo.idx(i - 1, j)]) / (2.0 * geo.hx());
                        let gy = (field[geo.idx(i, j + 1)] - field[geo.idx(i, j - 1)]) / (2.0 * geo.hy());
                        let g = (gx * gx + gy * gy).sqrt();
                        gf[geo.idx(i, j)] = g;
                        max_g = max_g.max(g);
                    }
                }
            }
        }
        t_min = 0.0;
        t_max = max_g.max(1.0);
        grad_field = gf;
        display_field = &grad_field;
    } else {
        // Find min/max from active nodes
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                if !geo.is_void(i, j) {
                    let v = field[geo.idx(i, j)];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
            }
        }
        t_min = mn;
        t_max = mx;
        display_field = field;
    };

    let temp_range = (t_max - t_min).max(0.001);
    let temp_range = if temp_range.is_finite() { temp_range } else { 1.0 };

    let step_x = (geo.nx as f64 / inner.width as f64).max(1.0) as usize;
    let step_y = (geo.ny as f64 / inner.height as f64).max(1.0) as usize;

    let title = if show_gradient {
        " Gradient Map [K/m] "
    } else {
        " 2D Heat Map "
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border));
    frame.render_widget(block, area);

    let mut screen_y = 0u16;
    let mut node_j = 0usize;
    while node_j < geo.ny && screen_y < inner.height {
        let row_y = inner.y + screen_y;
        let mut spans = Vec::new();

        let mut node_i = 0usize;
        while node_i < geo.nx && (spans.len() as u16) < inner.width {
            let j = node_j.min(geo.ny - 1);

            if geo.is_void(node_i, j) {
                spans.push(Span::styled(
                    "\u{25AA}",
                    Style::default().fg(Color::Rgb(40, 40, 40)),
                ));
            } else {
                // Check for source markers
                let is_source = if !show_gradient {
                    config.heat_sources.iter().any(|s| {
                        (s.position_i as isize - node_i as isize).unsigned_abs() <= step_x
                            && (s.position_j as isize - j as isize).unsigned_abs() <= step_y
                    })
                } else {
                    false
                };

                if is_source {
                    let src = config.heat_sources.iter().find(|s| {
                        (s.position_i as isize - node_i as isize).unsigned_abs() <= step_x
                            && (s.position_j as isize - j as isize).unsigned_abs() <= step_y
                    }).unwrap();
                    let color = if src.temperature_c > config.t_initial_c {
                        Color::Rgb(255, 80, 80)
                    } else {
                        Color::Rgb(80, 180, 255)
                    };
                    spans.push(Span::styled("\u{25CF}", Style::default().fg(color)));
                } else {
                    let val = display_field[geo.idx(node_i, j)];
                    let norm = ((val - t_min) / temp_range).clamp(0.0, 1.0);
                    let level = (norm * 7.0) as usize;
                    let (r, g, b) = HEAT_COLORS[level];
                    spans.push(Span::styled(
                        HEAT_CHARS[level].to_string(),
                        Style::default().fg(Color::Rgb(r, g, b)),
                    ));
                }
            }
            node_i += step_x.max(1);
        }

        frame.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect {
                x: inner.x,
                y: row_y,
                width: inner.width,
                height: 1,
            },
        );
        node_j += step_y.max(1);
        screen_y += 1;
    }
}

// ─── Full Statistics (view mode 5) ───────────────────────────────────────────

fn render_full_statistics(
    frame: &mut ratatui::Frame,
    area: Rect,
    result: &MhpsResult,
    t: &ThemeColors,
) {
    let config = &result.config;
    let geo = &config.geometry;
    let mat_idx = geo.material_map.get(geo.nx * geo.ny / 2).copied().unwrap_or(0);
    let mat = &MATERIALS[mat_idx.min(MATERIALS.len() - 1)];

    let last = result.snapshots.last();
    let max_t = last.map(|s| s.max_temp_c).unwrap_or(0.0);
    let min_t = last.map(|s| s.min_temp_c).unwrap_or(0.0);
    let max_grad = last.map(|s| s.max_gradient).unwrap_or(0.0);

    // Two-column layout
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // === LEFT: Physical results ===
    let melting_ok = mat.melting_point_c.map_or(true, |mp| max_t < mp * 0.9);
    let melting_str = if let Some(mp) = mat.melting_point_c {
        if melting_ok {
            format!("SAFE (max {:.0}C / melt {:.0}C)", max_t, mp)
        } else {
            format!("RISK ({:.0}C / melt {:.0}C)", max_t, mp)
        }
    } else {
        "N/A (no melting point)".to_string()
    };
    let melting_color = if melting_ok {
        Color::Rgb(74, 222, 128)
    } else {
        Color::Rgb(248, 113, 113)
    };

    let thermal_resistance = if config.convection_h > 0.0 {
        (max_t - config.t_ambient_c) / config.convection_h
    } else {
        0.0
    };

    let grad_color = if max_grad > 2000.0 {
        Color::Rgb(248, 113, 113)
    } else {
        Color::Rgb(74, 222, 128)
    };

    let phys_lines = vec![
        Line::from(Span::styled(
            "  PHYSICAL RESULTS",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  Material     ", mat.name, t.text, t.text_muted),
        kv_line("  T initial    ", &format!("{:.1}C", config.t_initial_c), t.text, t.text_muted),
        kv_line("  T ambient    ", &format!("{:.1}C", config.t_ambient_c), t.text_dim, t.text_muted),
        kv_line(
            "  T max final  ",
            &format!("{:.2}C", max_t),
            Color::Rgb(220, 60, 20),
            t.text_muted,
        ),
        kv_line(
            "  T min final  ",
            &format!("{:.2}C", min_t),
            Color::Rgb(100, 150, 255),
            t.text_muted,
        ),
        kv_line("  dT overall   ", &format!("{:.2}C", max_t - min_t), t.accent, t.text_muted),
        kv_line(
            "  Max gradient ",
            &format!("{:.0} K/m", max_grad),
            grad_color,
            t.text_muted,
        ),
        kv_line("  Melting check", &melting_str, melting_color, t.text_muted),
        Line::from(""),
        kv_line(
            "  Convection h ",
            &format!("{:.1} W/m2K", config.convection_h),
            t.text_dim,
            t.text_muted,
        ),
        kv_line(
            "  Thermal R    ",
            &format!("{:.4} m2K/W", thermal_resistance),
            t.text_dim,
            t.text_muted,
        ),
        kv_line(
            "  a (diffusiv.)",
            &format!("{:.3e} m2/s", mat.thermal_diffusivity()),
            t.text_dim,
            t.text_muted,
        ),
        kv_line(
            "  l (conduct.) ",
            &format!("{:.1} W/mK", mat.thermal_conductivity),
            t.text_dim,
            t.text_muted,
        ),
    ];

    // === RIGHT: Computation results ===
    let spmv_per_sec = if result.computation_ms > 0.0 {
        result.total_steps as f64 / (result.computation_ms / 1000.0)
    } else {
        0.0
    };
    let mem_mb = (geo.total_nodes() * 8 * 7) as f64 / 1024.0 / 1024.0;
    let active_nodes = (0..geo.nx)
        .flat_map(|i| (0..geo.ny).map(move |j| (i, j)))
        .filter(|&(i, j)| !geo.is_void(i, j))
        .count();

    let conv_at_str = result
        .convergence_step
        .map(|s| {
            format!(
                "step {} ({:.1}s)",
                s,
                s as f64 * config.time_step_dt
            )
        })
        .unwrap_or_else(|| "\u{2014}".into());
    let conv_color = if result.converged {
        Color::Rgb(74, 222, 128)
    } else {
        Color::Rgb(248, 113, 113)
    };

    let comp_lines = vec![
        Line::from(Span::styled(
            "  COMPUTATION RESULTS",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line(
            "  Grid         ",
            &format!("{}x{}x{}", geo.nx, geo.ny, geo.nz),
            t.text,
            t.text_muted,
        ),
        kv_line(
            "  Total nodes  ",
            &format!("{}", geo.total_nodes()),
            t.text,
            t.text_muted,
        ),
        kv_line(
            "  Active nodes ",
            &format!("{}", active_nodes),
            t.text_dim,
            t.text_muted,
        ),
        kv_line(
            "  Matrix NNZ   ",
            &format!("~{}", geo.total_nodes() * 5),
            t.text_dim,
            t.text_muted,
        ),
        kv_line(
            "  dt           ",
            &format!("{:.5}s", config.time_step_dt),
            t.text,
            t.text_muted,
        ),
        kv_line(
            "  Sim time     ",
            &format!("{:.1}s ({:.1}min)", config.total_time_s, config.total_time_s / 60.0),
            t.text,
            t.text_muted,
        ),
        kv_line(
            "  Steps        ",
            &format!("{}", result.total_steps),
            t.text,
            t.text_muted,
        ),
        kv_line(
            "  Compute ms   ",
            &format!("{:.1}ms", result.computation_ms),
            t.accent,
            t.text_muted,
        ),
        kv_line(
            "  SpMV/sec     ",
            &format!("{:.0}", spmv_per_sec),
            t.accent,
            t.text_muted,
        ),
        kv_line(
            "  RAM (matrix) ",
            &format!("{:.1}MB", mem_mb),
            t.text_dim,
            t.text_muted,
        ),
        kv_line(
            "  Converged    ",
            if result.converged { "YES" } else { "NO (time limit)" },
            conv_color,
            t.text_muted,
        ),
        kv_line("  Conv. at     ", &conv_at_str, t.text_dim, t.text_muted),
        kv_line(
            "  Snapshots    ",
            &format!("{}", result.snapshots.len()),
            t.text_dim,
            t.text_muted,
        ),
    ];

    frame.render_widget(
        Paragraph::new(phys_lines).block(
            Block::default()
                .title(" Physical ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        cols[0],
    );
    frame.render_widget(
        Paragraph::new(comp_lines).block(
            Block::default()
                .title(" Computation ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        cols[1],
    );
}

// ─── Stats Panel ─────────────────────────────────────────────────────────────

fn render_mhps_stats(
    frame: &mut ratatui::Frame,
    area: Rect,
    result: &MhpsResult,
    app: &App,
    t: &ThemeColors,
) {
    let geo = &result.config.geometry;
    let mat = &MATERIALS[geo.material_map[0].min(MATERIALS.len() - 1)];
    let last = result.snapshots.last();

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(kv_line(
        "  Material",
        &format!("{} (lambda={:.0} W/mK)", mat.name, mat.thermal_conductivity),
        t.text,
        t.text_muted,
    ));
    lines.push(kv_line(
        "  Size",
        &format!(
            "{:.0}x{:.0}mm, t={:.1}mm",
            geo.length_x * 1000.0,
            geo.length_y * 1000.0,
            geo.base_thickness * 1000.0
        ),
        t.text,
        t.text_muted,
    ));
    lines.push(kv_line(
        "  Grid",
        &format!("{}x{} = {} nodes", geo.nx, geo.ny, geo.nx * geo.ny),
        t.text,
        t.text_muted,
    ));
    lines.push(kv_line(
        "  Sources",
        &format!("{}", result.config.heat_sources.len()),
        t.text,
        t.text_muted,
    ));
    for src in &result.config.heat_sources {
        let label = if src.temperature_c > result.config.t_initial_c {
            "HOT"
        } else {
            "COLD"
        };
        lines.push(kv_line(
            &format!("    {}", src.name),
            &format!("{:.0}C ({},{})[{}]", src.temperature_c, src.position_i, src.position_j, label),
            t.text_dim,
            t.text_muted,
        ));
    }
    lines.push(Line::from(""));
    lines.push(kv_line(
        "  Sim time",
        &format!("{:.1}s (dt={:.4}s)", result.config.total_time_s, result.config.time_step_dt),
        t.text,
        t.text_muted,
    ));
    lines.push(kv_line(
        "  Steps",
        &format!("{}", result.total_steps),
        t.text,
        t.text_muted,
    ));
    if result.converged {
        lines.push(kv_line(
            "  Converged",
            &format!(
                "step {} (t={:.1}s)",
                result.convergence_step.unwrap_or(0),
                result.convergence_step.unwrap_or(0) as f64 * result.config.time_step_dt,
            ),
            Color::Rgb(74, 222, 128),
            t.text_muted,
        ));
    } else {
        lines.push(kv_line("  Converged", "no", Color::Rgb(248, 113, 113), t.text_muted));
    }
    lines.push(kv_line(
        "  Compute",
        &format_duration(result.computation_ms),
        t.accent,
        t.text_muted,
    ));

    let save_h = if app.mhps_csv_saved { " (saved)" } else { "" };
    let save_f = if app.mhps_field_saved { " (saved)" } else { "" };
    lines.push(kv_line("  CSV", save_h, t.text_dim, t.text_muted));
    lines.push(kv_line("  Field", save_f, t.text_dim, t.text_muted));

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Configuration ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

// ─── Timeline ────────────────────────────────────────────────────────────────

fn render_mhps_timeline(
    frame: &mut ratatui::Frame,
    area: Rect,
    snapshots: &[MhpsSnapshot],
    t: &ThemeColors,
) {
    let block = Block::default()
        .title(" Temperature Timeline ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(t.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if snapshots.is_empty() || inner.height == 0 || inner.width < 8 {
        frame.render_widget(
            Paragraph::new("  No data"),
            inner,
        );
        return;
    }

    let t_max_all = snapshots.iter().map(|s| s.max_temp_c).fold(f64::NEG_INFINITY, f64::max);
    let t_min_all = snapshots.iter().map(|s| s.min_temp_c).fold(f64::INFINITY, f64::min);
    let range = (t_max_all - t_min_all).max(0.001);
    let range = if range.is_finite() { range } else { 1.0 };
    let n_snaps = snapshots.len();

    // Uniform temperature guard
    if range < 0.01 {
        frame.render_widget(
            Paragraph::new(format!("  Uniform temperature: {:.1}°C", t_max_all)),
            inner,
        );
        return;
    }

    let label_w = 6usize; // "9999° "
    let chart_w = (inner.width as usize).saturating_sub(label_w);
    let chart_h = inner.height as usize;

    for row in 0..chart_h {
        let threshold = t_max_all - (row as f64 / chart_h as f64) * range;
        let mut row_spans = vec![Span::styled(
            format!("{:>4.0}\u{00B0} ", threshold),
            Style::default().fg(t.text_dim),
        )];

        for col in 0..chart_w {
            let snap_idx = (col * n_snaps / chart_w.max(1)).min(n_snaps.saturating_sub(1));
            if let Some(snap) = snapshots.get(snap_idx) {
                let ch = if snap.max_temp_c >= threshold {
                    Span::styled("\u{2588}", Style::default().fg(Color::Rgb(220, 60, 20)))
                } else if snap.mean_temp_c >= threshold {
                    Span::styled("\u{2593}", Style::default().fg(Color::Rgb(220, 200, 80)))
                } else if snap.min_temp_c >= threshold {
                    Span::styled("\u{2591}", Style::default().fg(Color::Rgb(50, 80, 150)))
                } else {
                    Span::styled("\u{00B7}", Style::default().fg(t.text_dim))
                };
                row_spans.push(ch);
            }
        }

        let y = inner.y + row as u16;
        if y < inner.y + inner.height {
            frame.render_widget(
                Paragraph::new(Line::from(row_spans)),
                Rect { x: inner.x, y, width: inner.width, height: 1 },
            );
        }
    }
}

// ─── Physics Analysis ────────────────────────────────────────────────────────

fn render_mhps_analysis(
    frame: &mut ratatui::Frame,
    area: Rect,
    result: &MhpsResult,
    t: &ThemeColors,
) {
    let geo = &result.config.geometry;
    let last = result.snapshots.last();

    // Find max temp node
    let (max_i, max_j) = {
        let mut max_t = f64::NEG_INFINITY;
        let mut pos = (0, 0);
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                let temp = result.final_field[geo.idx(i, j)];
                if temp > max_t && !geo.is_void(i, j) {
                    max_t = temp;
                    pos = (i, j);
                }
            }
        }
        pos
    };

    let mat_idx = geo.material_map[geo.idx(max_i, max_j)].min(MATERIALS.len() - 1);
    let mat = &MATERIALS[mat_idx];
    let max_temp = last.map(|s| s.max_temp_c).unwrap_or(0.0);
    let min_temp = last.map(|s| s.min_temp_c).unwrap_or(0.0);
    let max_grad = last.map(|s| s.max_gradient).unwrap_or(0.0);

    let melting_status = match mat.melting_point_c {
        Some(mp) if max_temp > mp * 0.9 => ("\u{26A0} DANGER: near melting point!", Color::Rgb(248, 113, 113)),
        Some(mp) if max_temp > mp * 0.7 => ("\u{26A0} WARNING: high temp vs melting", Color::Rgb(220, 200, 80)),
        Some(_) => ("\u{2713} SAFE: well below melting", Color::Rgb(74, 222, 128)),
        None => ("no melting point (non-metal)", t.text_muted),
    };

    let gradient_info = if max_grad > 5000.0 {
        format!("Grad: {:.0} K/m — SEVERE stress", max_grad)
    } else if max_grad > 1000.0 {
        format!("Grad: {:.0} K/m — moderate stress", max_grad)
    } else {
        format!("Grad: {:.0} K/m — low stress", max_grad)
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  T_max: ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:.1}\u{00B0}C", max_temp),
                Style::default()
                    .fg(Color::Rgb(220, 60, 20))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" at ({},{})", max_i, max_j),
                Style::default().fg(t.text_muted),
            ),
        ]),
        Line::from(vec![
            Span::styled("  T_min: ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!("{:.1}\u{00B0}C", min_temp),
                Style::default().fg(Color::Rgb(100, 150, 255)),
            ),
        ]),
        Line::from(Span::styled(
            format!("  {}", gradient_info),
            Style::default().fg(t.text_muted),
        )),
        Line::from(Span::styled(
            format!("  {} — {}", mat.name, melting_status.0),
            Style::default().fg(melting_status.1),
        )),
        Line::from(vec![
            Span::styled("  Compute: ", Style::default().fg(t.text_dim)),
            Span::styled(
                format!(
                    "{} for {} SpMV iterations",
                    format_duration(result.computation_ms),
                    result.total_steps
                ),
                Style::default().fg(t.accent),
            ),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Physics Analysis ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

// ─── Footer ──────────────────────────────────────────────────────────────────

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
