// ═══════════════════════════════════════════════════════════════════════════════
//  UNIVERSAL GRAPHICS & ANALYTICS ENGINE
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Reusable chart renderers for any domain (Thermal, Economics, etc.).
//  All renderers accept raw data slices and a target Rect, producing
//  ratatui primitives. No domain-specific logic here.
//
//  Renderers:
//    1. Line chart / Scatter plot — Canvas with Braille markers
//    2. Vertical bar chart (histogram)
//    3. Pie chart — geometric circle via Canvas Braille dots
//    4. Isometric 3D surface plot — Canvas with painter's algorithm
//    5. Horizontal gauge stack (fallback pie alternative)
// ═══════════════════════════════════════════════════════════════════════════════

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    canvas::{Canvas, Circle, Line as CanvasLine, Points},
    Block, Borders, Paragraph,
};

use crate::common::ThemeColors;

// ─── Color Palette ──────────────────────────────────────────────────────────

/// 12-color palette for chart series, chosen for maximum contrast on dark bg.
pub const SERIES_COLORS: [Color; 12] = [
    Color::Rgb(245, 197, 24),  // gold (accent)
    Color::Rgb(74, 222, 128),  // green
    Color::Rgb(96, 165, 250),  // blue
    Color::Rgb(248, 113, 113), // red
    Color::Rgb(192, 132, 252), // purple
    Color::Rgb(45, 212, 191),  // teal
    Color::Rgb(251, 191, 36),  // amber
    Color::Rgb(244, 114, 182), // pink
    Color::Rgb(34, 211, 238),  // cyan
    Color::Rgb(163, 230, 53),  // lime
    Color::Rgb(232, 121, 249), // fuchsia
    Color::Rgb(251, 146, 60),  // orange
];

pub fn series_color(idx: usize) -> Color {
    SERIES_COLORS[idx % SERIES_COLORS.len()]
}

// ─── Data Types ─────────────────────────────────────────────────────────────

/// A single data series for line/scatter charts.
pub struct ChartSeries<'a> {
    pub label: &'a str,
    pub data: &'a [(f64, f64)], // (x, y) pairs
    pub color: Color,
}

/// A bar for the bar chart.
pub struct BarEntry<'a> {
    pub label: &'a str,
    pub value: f64,
    pub color: Color,
}

/// A slice for the pie chart.
pub struct PieSlice<'a> {
    pub label: &'a str,
    pub value: f64,
    pub color: Color,
}

/// A 2D grid for isometric 3D surface rendering.
pub struct SurfaceGrid<'a> {
    /// Flat row-major data[ix * nz + iz], dimensions nx × nz.
    pub data: &'a [f64],
    pub nx: usize,
    pub nz: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  1. LINE CHART & SCATTER PLOT
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Uses ratatui Canvas with Braille marker resolution (each cell = 2×4 dots).
//  Multiple series overlaid with distinct colors.
//  Auto-scales axes to fit data range.

pub fn render_line_chart<'a>(
    title: &str,
    series: &[ChartSeries<'a>],
    x_label: &str,
    y_label: &str,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    scatter_only: bool,
) {
    if area.width < 10 || area.height < 6 {
        return;
    }

    // Compute global bounds
    let (mut x_min, mut x_max) = (f64::MAX, f64::MIN);
    let (mut y_min, mut y_max) = (f64::MAX, f64::MIN);
    for s in series {
        for &(x, y) in s.data {
            if x < x_min { x_min = x; }
            if x > x_max { x_max = x; }
            if y < y_min { y_min = y; }
            if y > y_max { y_max = y; }
        }
    }
    if (x_max - x_min).abs() < 1e-15 { x_max = x_min + 1.0; }
    if (y_max - y_min).abs() < 1e-15 { y_max = y_min + 1.0; }

    // Add 5% padding
    let x_pad = (x_max - x_min) * 0.05;
    let y_pad = (y_max - y_min) * 0.05;
    x_min -= x_pad;
    x_max += x_pad;
    y_min -= y_pad;
    y_max += y_pad;

    let canvas = Canvas::default()
        .block(
            Block::default()
                .title(Span::styled(
                    format!(" {title} "),
                    Style::default().fg(t.accent),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        )
        .x_bounds([x_min, x_max])
        .y_bounds([y_min, y_max])
        .paint(move |ctx| {
            for s in series {
                if scatter_only {
                    // Scatter: draw points only
                    let points: Vec<(f64, f64)> = s.data.to_vec();
                    ctx.draw(&Points {
                        coords: &points,
                        color: s.color,
                    });
                } else {
                    // Line: connect consecutive points
                    for w in s.data.windows(2) {
                        ctx.draw(&CanvasLine {
                            x1: w[0].0,
                            y1: w[0].1,
                            x2: w[1].0,
                            y2: w[1].1,
                            color: s.color,
                        });
                    }
                }
            }
        });

    frame.render_widget(canvas, area);

    // Axis labels as text overlay at edges
    // X-label at bottom-right
    if area.height > 2 && area.width > x_label.len() as u16 + 4 {
        let x_area = Rect {
            x: area.x + area.width - x_label.len() as u16 - 2,
            y: area.y + area.height - 1,
            width: x_label.len() as u16 + 2,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Span::styled(x_label, Style::default().fg(t.text_dim))),
            x_area,
        );
    }
    // Y-label at top-left (first 2 chars)
    if area.height > 2 && !y_label.is_empty() {
        let y_area = Rect {
            x: area.x + 1,
            y: area.y,
            width: y_label.len().min(12) as u16,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Span::styled(y_label, Style::default().fg(t.text_dim))),
            y_area,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  2. VERTICAL BAR CHART
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Text-based vertical bars using block characters (█▓▒░).
//  Each bar gets a label, value, and color.

pub fn render_bar_chart(
    title: &str,
    bars: &[BarEntry],
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    if bars.is_empty() || area.height < 5 || area.width < 10 {
        return;
    }

    let max_val = bars.iter().map(|b| b.value.abs()).fold(0.0_f64, f64::max).max(1e-15);
    let bar_max_w = (area.width as usize).saturating_sub(28);

    let mut lines: Vec<Line> = Vec::new();
    for bar in bars {
        let bar_len = ((bar.value.abs() / max_val) * bar_max_w as f64) as usize;
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<16} ", truncate_str(bar.label, 16)),
                Style::default().fg(t.text_muted),
            ),
            Span::styled("▓".repeat(bar_len.max(1)), Style::default().fg(bar.color)),
            Span::styled(format!(" {:.2}", bar.value), Style::default().fg(t.text)),
        ]));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(Span::styled(
                    format!(" {title} "),
                    Style::default().fg(t.accent),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  3. PIE CHART (Braille Circle)
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Geometric approach: draw a filled circle using Canvas Braille dots.
//  Each slice is a sector [θ_start, θ_end] filled with its color.
//  Uses polar coordinate scan: for each dot, check which sector it falls in.
//
//  Engineering rationale:
//    Canvas in Braille mode gives 2×4 sub-pixel resolution per character cell,
//    yielding ~160×88 effective dots in a 80×22 area. Sufficient for visual
//    distinction of 6–10 slices.

pub fn render_pie_chart(
    title: &str,
    slices: &[PieSlice],
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    if slices.is_empty() || area.height < 8 || area.width < 20 {
        return;
    }

    let total: f64 = slices.iter().map(|s| s.value.abs()).sum();
    if total < 1e-15 {
        return;
    }

    // Compute sector angles [0, 2π)
    let mut angles: Vec<(f64, f64, Color)> = Vec::new();
    let mut theta = 0.0_f64;
    for slice in slices {
        let sweep = (slice.value.abs() / total) * std::f64::consts::TAU;
        angles.push((theta, theta + sweep, slice.color));
        theta += sweep;
    }

    // Reserve right side for legend
    let chart_w = (area.width as f64 * 0.6) as u16;
    let legend_w = area.width.saturating_sub(chart_w);

    let chart_area = Rect {
        x: area.x,
        y: area.y,
        width: chart_w,
        height: area.height,
    };
    let legend_area = Rect {
        x: area.x + chart_w,
        y: area.y + 2,
        width: legend_w,
        height: area.height.saturating_sub(2),
    };

    // Circle parameters — fit inside chart area
    let radius = (chart_area.height as f64 * 0.8).min(chart_area.width as f64 * 0.35);
    let cx = 0.0;
    let cy = 0.0;

    let angles_clone = angles.clone();

    let canvas = Canvas::default()
        .block(
            Block::default()
                .title(Span::styled(
                    format!(" {title} "),
                    Style::default().fg(t.accent),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        )
        .x_bounds([-radius * 1.2, radius * 1.2])
        .y_bounds([-radius * 1.2, radius * 1.2])
        .paint(move |ctx| {
            // Draw filled sectors using dense point sampling
            // Sample in a grid and assign each point to its sector
            let steps = 120; // radial resolution
            let angular_steps = 360; // angular resolution
            for &(theta_start, theta_end, color) in &angles_clone {
                // Sample points within this sector
                let mut points: Vec<(f64, f64)> = Vec::new();
                let n_angular = ((theta_end - theta_start) / std::f64::consts::TAU
                    * angular_steps as f64)
                    .ceil() as usize;
                for ai in 0..=n_angular {
                    let a = theta_start
                        + (ai as f64 / n_angular.max(1) as f64) * (theta_end - theta_start);
                    let cos_a = a.cos();
                    let sin_a = a.sin();
                    for ri in 1..=steps {
                        let r = (ri as f64 / steps as f64) * radius;
                        points.push((cx + r * cos_a, cy + r * sin_a));
                    }
                }
                ctx.draw(&Points {
                    coords: &points,
                    color,
                });
            }

            // Draw circle outline
            ctx.draw(&Circle {
                x: cx,
                y: cy,
                radius,
                color: Color::White,
            });
        });

    frame.render_widget(canvas, chart_area);

    // Legend
    let mut legend_lines: Vec<Line> = Vec::new();
    for (i, slice) in slices.iter().enumerate() {
        let pct = slice.value.abs() / total * 100.0;
        legend_lines.push(Line::from(vec![
            Span::styled("██ ", Style::default().fg(slice.color)),
            Span::styled(
                format!("{} ({:.1}%)", truncate_str(slice.label, 14), pct),
                Style::default().fg(t.text_muted),
            ),
        ]));
        if i >= 10 {
            legend_lines.push(Line::from(Span::styled(
                "  ...",
                Style::default().fg(t.text_dim),
            )));
            break;
        }
    }
    frame.render_widget(Paragraph::new(legend_lines), legend_area);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  4. ISOMETRIC 3D SURFACE PLOT
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Projects a 2D heightfield into pseudo-3D using isometric projection:
//    u = (x − z) · cos(θ)
//    v = (x + z) · sin(θ) − height · scale_z
//
//  Uses Canvas Braille dots for sub-character resolution.
//  Painter's algorithm: iterate back-to-front (high x, high z first).
//
//  Engineering rationale:
//    Canvas with Braille markers gives us ~2×4 sub-pixel density per cell.
//    For a 80×40 area, that's ~160×160 effective dots — sufficient for
//    visualizing terrain-like surfaces (temperature fields, sector outputs).

pub fn render_isometric_surface(
    title: &str,
    grid: &SurfaceGrid,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
    color_fn: fn(f64) -> Color,
) {
    if grid.nx == 0 || grid.nz == 0 || area.height < 8 || area.width < 20 {
        return;
    }

    let nx = grid.nx;
    let nz = grid.nz;

    // Find data range
    let mut d_min = f64::MAX;
    let mut d_max = f64::MIN;
    for &v in grid.data.iter().take(nx * nz) {
        if v < d_min { d_min = v; }
        if v > d_max { d_max = v; }
    }
    let d_range = (d_max - d_min).max(1e-15);

    let theta = std::f64::consts::FRAC_PI_6; // 30°
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Projection bounds to determine canvas coordinate system
    let scale_z = 0.5;
    let u_min = -(nz as f64) * cos_t - 1.0;
    let u_max = (nx as f64) * cos_t + 1.0;
    let v_min = -scale_z * 1.2;
    let v_max = ((nx + nz) as f64) * sin_t + 1.0;

    // Owned copy for the closure
    let data: Vec<f64> = grid.data[..nx * nz].to_vec();

    let canvas = Canvas::default()
        .block(
            Block::default()
                .title(Span::styled(
                    format!(" {title} "),
                    Style::default().fg(t.accent),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        )
        .x_bounds([u_min, u_max])
        .y_bounds([v_min, v_max])
        .paint(move |ctx| {
            // Painter's algorithm: back-to-front
            // We draw column-lines along the Z axis for each X row
            for ix in 0..nx {
                for iz in 0..nz {
                    let val = data[ix * nz + iz];
                    let norm = (val - d_min) / d_range;
                    let height = norm * scale_z;

                    let u = (ix as f64 - iz as f64) * cos_t;
                    let v = (ix as f64 + iz as f64) * sin_t;
                    let v_top = v + height;

                    let color = color_fn(norm);

                    // Draw a short vertical line from base to top
                    ctx.draw(&CanvasLine {
                        x1: u,
                        y1: v_max - v,
                        x2: u,
                        y2: v_max - v_top,
                        color,
                    });

                    // Connect to next Z neighbor for wireframe effect
                    if iz + 1 < nz {
                        let val2 = data[ix * nz + iz + 1];
                        let norm2 = (val2 - d_min) / d_range;
                        let h2 = norm2 * scale_z;
                        let u2 = (ix as f64 - (iz + 1) as f64) * cos_t;
                        let v2 = (ix as f64 + (iz + 1) as f64) * sin_t + h2;
                        ctx.draw(&CanvasLine {
                            x1: u,
                            y1: v_max - v_top,
                            x2: u2,
                            y2: v_max - v2,
                            color,
                        });
                    }

                    // Connect to next X neighbor
                    if ix + 1 < nx {
                        let val2 = data[(ix + 1) * nz + iz];
                        let norm2 = (val2 - d_min) / d_range;
                        let h2 = norm2 * scale_z;
                        let u2 = ((ix + 1) as f64 - iz as f64) * cos_t;
                        let v2 = ((ix + 1) as f64 + iz as f64) * sin_t + h2;
                        ctx.draw(&CanvasLine {
                            x1: u,
                            y1: v_max - v_top,
                            x2: u2,
                            y2: v_max - v2,
                            color,
                        });
                    }
                }
            }
        });

    frame.render_widget(canvas, area);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  5. HORIZONTAL GAUGE STACK (alternative pie visualization)
// ═══════════════════════════════════════════════════════════════════════════════
//
//  A stacked horizontal bar showing proportional segments.
//  Useful as a fallback for pie charts in narrow terminals.

pub fn render_gauge_stack(
    title: &str,
    slices: &[PieSlice],
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    if slices.is_empty() || area.width < 10 {
        return;
    }

    let total: f64 = slices.iter().map(|s| s.value.abs()).sum();
    if total < 1e-15 {
        return;
    }

    let bar_w = (area.width as usize).saturating_sub(4);
    let mut spans: Vec<Span> = vec![Span::raw("  ")];

    for slice in slices {
        let frac = slice.value.abs() / total;
        let chars = ((frac * bar_w as f64).round() as usize).max(if frac > 0.01 { 1 } else { 0 });
        if chars > 0 {
            spans.push(Span::styled("█".repeat(chars), Style::default().fg(slice.color)));
        }
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(spans));
    lines.push(Line::from(""));

    // Legend
    for slice in slices.iter().take(12) {
        let pct = slice.value.abs() / total * 100.0;
        lines.push(Line::from(vec![
            Span::styled("  ██ ", Style::default().fg(slice.color)),
            Span::styled(
                format!("{}: {:.1}%", truncate_str(slice.label, 20), pct),
                Style::default().fg(t.text_muted),
            ),
        ]));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(Span::styled(
                    format!(" {title} "),
                    Style::default().fg(t.accent),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

// ─── Shared Helpers ─────────────────────────────────────────────────────────

/// Standard heat color gradient: blue → cyan → green → yellow → red.
pub fn heat_gradient(norm: f64) -> Color {
    let n = norm.clamp(0.0, 1.0);
    let (r, g, b) = if n < 0.25 {
        let t = n / 0.25;
        (lerp8(30, 0, t), lerp8(40, 200, t), lerp8(80, 200, t))
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

/// Economic color gradient: green (gain) → yellow (neutral) → red (loss).
pub fn econ_gradient(norm: f64) -> Color {
    let n = norm.clamp(0.0, 1.0);
    let (r, g, b) = if n < 0.5 {
        let t = n / 0.5;
        (lerp8(40, 200, t), lerp8(180, 200, t), lerp8(40, 60, t))
    } else {
        let t = (n - 0.5) / 0.5;
        (lerp8(200, 255, t), lerp8(200, 60, t), lerp8(60, 30, t))
    };
    Color::Rgb(r, g, b)
}

#[inline]
fn lerp8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u8
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}
