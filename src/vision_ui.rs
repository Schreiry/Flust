// =============================================================================
//  COMPUTER VISION (IR PROCESSING) — TUI WIZARD & RESULTS
// =============================================================================

use std::sync::{mpsc, Arc, Mutex};

use crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::common::{ProgressHandle, ThemeColors};
use crate::vision::{self, KernelType, VisionConfig, VisionResult};
use crate::interactive::{
    App, ComputeContext, ComputeResult, ComputeTask, EtaTracker, Overlay, Screen,
    format_duration, kv_line, GEAR_FRAMES, GEAR_SINGLE,
};

// --- Field Labels -----------------------------------------------------------

const CONFIG_LABELS: &[&str] = &[
    "Input rows",
    "Input cols",
    "Upscale factor",
    "Kernel size (odd)",
    "Noise level (0-1)",
    "Seed (optional)",
];

// --- Input Handlers ---------------------------------------------------------

pub fn handle_vision_config(app: &mut App, key: KeyCode) {
    let n_fields = CONFIG_LABELS.len();
    match key {
        KeyCode::Tab | KeyCode::Down => {
            app.vision_active_field = (app.vision_active_field + 1) % n_fields;
        }
        KeyCode::BackTab | KeyCode::Up => {
            app.vision_active_field = if app.vision_active_field == 0 {
                n_fields - 1
            } else {
                app.vision_active_field - 1
            };
        }
        KeyCode::Char(c) if c.is_ascii_digit() || c == '.' => {
            active_config_field_mut(app).push(c);
        }
        KeyCode::Backspace => {
            active_config_field_mut(app).pop();
        }
        KeyCode::Enter => {
            app.screen = Screen::VisionKernelSelect;
            app.vision_kernel_idx = 0;
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
    match app.vision_active_field {
        0 => &mut app.vision_rows_input,
        1 => &mut app.vision_cols_input,
        2 => &mut app.vision_upscale_input,
        3 => &mut app.vision_ksize_input,
        4 => &mut app.vision_noise_input,
        5 => &mut app.vision_seed_input,
        _ => &mut app.vision_rows_input,
    }
}

pub fn handle_vision_kernel(app: &mut App, key: KeyCode) {
    let kernels = KernelType::all();
    match key {
        KeyCode::Up => {
            if app.vision_kernel_idx > 0 {
                app.vision_kernel_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.vision_kernel_idx < kernels.len() - 1 {
                app.vision_kernel_idx += 1;
            }
        }
        KeyCode::Enter => {
            app.screen = Screen::VisionConfirm;
        }
        KeyCode::Esc => {
            app.screen = Screen::VisionConfig;
        }
        _ => {}
    }
}

pub fn handle_vision_confirm(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Enter => {
            if let Some(config) = build_vision_config(app) {
                launch_vision_pipeline(app, config);
            }
        }
        KeyCode::Esc => {
            app.screen = Screen::VisionKernelSelect;
        }
        _ => {}
    }
}

pub fn handle_vision_results(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if !app.vision_csv_saved {
                if let Screen::VisionResults { ref result } = app.screen {
                    let path = format!(
                        "flust_vision_{}.csv",
                        crate::io::timestamp_now().replace(':', "-")
                    );
                    if crate::vision_export::export_vision_csv(result, &path).is_ok() {
                        app.vision_csv_saved = true;
                    }
                }
            }
        }
        KeyCode::Char('h') | KeyCode::Char('H') => {
            app.overlay = Overlay::VisionHeatmap;
        }
        KeyCode::Char('p') | KeyCode::Char('P') => {
            app.overlay = Overlay::VisionPipeline;
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.screen = Screen::VisionConfirm;
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            app.screen = Screen::MainMenu;
            app.main_menu_idx = 0;
        }
        _ => {}
    }
}

// --- Config Builder ---------------------------------------------------------

fn build_vision_config(app: &App) -> Option<VisionConfig> {
    let input_rows: usize = app.vision_rows_input.parse::<f64>().ok()? as usize;
    let input_cols: usize = app.vision_cols_input.parse::<f64>().ok()? as usize;
    let upscale_factor: usize = app.vision_upscale_input.parse::<f64>().ok()? as usize;
    let kernel_size: usize = app.vision_ksize_input.parse::<f64>().ok()? as usize;
    let noise_level: f64 = app.vision_noise_input.parse().ok()?;
    let seed: Option<u64> = if app.vision_seed_input.is_empty() {
        None
    } else {
        Some(app.vision_seed_input.parse::<u64>().ok()?)
    };

    if input_rows < 2 || input_rows > 256 { return None; }
    if input_cols < 2 || input_cols > 256 { return None; }
    if upscale_factor < 1 || upscale_factor > 32 { return None; }
    if kernel_size < 3 || kernel_size > 15 || kernel_size % 2 == 0 { return None; }
    if !(0.0..1.0).contains(&noise_level) { return None; }

    let kernels = KernelType::all();
    let kernel_type = kernels.get(app.vision_kernel_idx).copied().unwrap_or(KernelType::Gaussian);

    Some(VisionConfig {
        input_rows,
        input_cols,
        upscale_factor,
        kernel_type,
        kernel_size,
        seed,
        noise_level,
    })
}

// --- Launch Pipeline --------------------------------------------------------

fn launch_vision_pipeline(app: &mut App, config: VisionConfig) {
    let progress = ProgressHandle::new(100);
    let progress_clone = progress.clone();
    let phase = Arc::new(Mutex::new("Initializing...".to_string()));
    let phase_clone = phase.clone();
    let (tx, rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match vision::run_vision_pipeline(&config, &progress_clone.inner, &phase_clone) {
                Ok(result) => {
                    let _ = tx.send(ComputeResult::Vision { result });
                }
                Err(e) => {
                    let _ = tx.send(ComputeResult::Error {
                        message: format!("Vision pipeline error: {e}"),
                    });
                }
            }
        }));
        if let Err(panic_info) = outcome {
            let msg = crate::interactive::extract_panic_message(panic_info);
            let _ = tx.send(ComputeResult::Error { message: msg });
        }
    });

    app.vision_phase = Some(phase);
    app.compute_task = Some(ComputeTask {
        progress,
        eta: EtaTracker::new(),
        receiver: rx,
        context: ComputeContext {
            algorithm_choice: crate::interactive::AlgorithmChoice::Naive,
            algorithm_name: "IR Vision Pipeline".into(),
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

    app.screen = Screen::VisionComputing;
}

// ============================================================================
//  RENDER FUNCTIONS
// ============================================================================

// --- Screen 1: Config -------------------------------------------------------

pub fn render_vision_config(
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
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  COMPUTER VISION \u{2014} IR Processing Configuration",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let values = [
        &app.vision_rows_input,
        &app.vision_cols_input,
        &app.vision_upscale_input,
        &app.vision_ksize_input,
        &app.vision_noise_input,
        &app.vision_seed_input,
    ];
    let hints = [
        "IR sensor rows. Typical: 8, 16, 32, 64",
        "IR sensor columns. Typical: 8, 16, 32, 64",
        "Bicubic upscale factor. 4 = 8\u{00d7}8 \u{2192} 32\u{00d7}32",
        "Convolution kernel size (must be odd: 3, 5, 7)",
        "Synthetic noise amplitude [0, 1). 0.1 = 10% noise",
        "Leave empty for random, or fix for reproducibility",
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    for (i, label) in CONFIG_LABELS.iter().enumerate() {
        let active = i == app.vision_active_field;
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
    if let (Ok(r), Ok(c), Ok(u), Ok(k)) = (
        app.vision_rows_input.parse::<f64>(),
        app.vision_cols_input.parse::<f64>(),
        app.vision_upscale_input.parse::<f64>(),
        app.vision_ksize_input.parse::<f64>(),
    ) {
        let cfg = VisionConfig {
            input_rows: r as usize,
            input_cols: c as usize,
            upscale_factor: u as usize,
            kernel_type: KernelType::Gaussian,
            kernel_size: k as usize,
            seed: None,
            noise_level: 0.0,
        };
        if cfg.input_rows >= 2 && cfg.input_cols >= 2 && cfg.kernel_size >= 3 {
            lines.push(Line::from(Span::styled(
                format!("    Estimated RAM: {:.1} MB", cfg.estimate_memory_mb()),
                Style::default().fg(t.text_dim),
            )));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" IR Sensor Parameters ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "  Pipeline: Generate IR data \u{2192} Bicubic upscale \u{2192} im2col \u{2192} GEMM convolution",
                Style::default().fg(t.text_dim),
            )),
            Line::from(Span::styled(
                "  Convolution is computed as a matrix multiply: Y = W \u{00d7} im2col(X)",
                Style::default().fg(t.text_dim),
            )),
        ]),
        chunks[2],
    );

    render_footer(
        &["[Tab] Next field", "[Enter] Select kernel", "[Esc] Back"],
        frame,
        chunks[3],
        t,
    );
}

// --- Screen 2: Kernel Selection ---------------------------------------------

pub fn render_vision_kernel(
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
            "  COMPUTER VISION \u{2014} Kernel Selection",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let kernels = KernelType::all();
    let descriptions = [
        "Smoothing filter: weighted average by distance. Reduces noise while\n      preserving edges better than box blur. \u{03c3} = K/4.",
        "Enhances edges and fine detail. Center weight = K\u{00b2},\n      all neighbors = -1. Amplifies high-frequency components.",
        "Laplacian edge detection: highlights intensity transitions.\n      Center = 1, neighbors = -1/(K\u{00b2}-1). Zero-sum kernel.",
        "Simple mean filter: all weights equal 1/K\u{00b2}.\n      Maximum smoothing, blurs edges uniformly.",
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    for (i, kernel) in kernels.iter().enumerate() {
        let active = i == app.vision_kernel_idx;
        let marker = if active { "\u{25b8} " } else { "  " };
        let style = if active {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.text)
        };
        lines.push(Line::from(Span::styled(
            format!("  {marker}{}", kernel.label()),
            style,
        )));
        if active {
            for desc_line in descriptions[i].split('\n') {
                lines.push(Line::from(Span::styled(
                    format!("    {desc_line}"),
                    Style::default().fg(t.text_dim),
                )));
            }
        }
        lines.push(Line::from(""));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" Convolution Kernel ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(
        &["[\u{2191}/\u{2193}] Select", "[Enter] Continue", "[Esc] Back"],
        frame,
        chunks[2],
        t,
    );
}

// --- Screen 3: Confirm ------------------------------------------------------

pub fn render_vision_confirm(
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
            "  COMPUTER VISION \u{2014} Confirm",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let kernels = KernelType::all();
    let kernel_name = kernels
        .get(app.vision_kernel_idx)
        .map(|k| k.label())
        .unwrap_or("Gaussian");

    let up_r = app.vision_rows_input.parse::<usize>().unwrap_or(8)
        * app.vision_upscale_input.parse::<usize>().unwrap_or(4);
    let up_c = app.vision_cols_input.parse::<usize>().unwrap_or(8)
        * app.vision_upscale_input.parse::<usize>().unwrap_or(4);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    lines.push(kv_line("  Input", &format!("{}x{}", app.vision_rows_input, app.vision_cols_input), t.text, t.text_muted));
    lines.push(kv_line("  Upscale factor", &app.vision_upscale_input, t.text, t.text_muted));
    lines.push(kv_line("  Upscaled size", &format!("{}x{}", up_r, up_c), t.accent, t.text_muted));
    lines.push(kv_line("  Kernel", kernel_name, t.accent, t.text_muted));
    lines.push(kv_line("  Kernel size", &app.vision_ksize_input, t.text, t.text_muted));
    lines.push(kv_line("  Noise level", &app.vision_noise_input, t.text, t.text_muted));
    let seed_display = if app.vision_seed_input.is_empty() {
        "random".to_string()
    } else {
        app.vision_seed_input.clone()
    };
    lines.push(kv_line("  Seed", &seed_display, t.text, t.text_muted));
    lines.push(Line::from(""));

    // RAM estimate
    if let Some(cfg) = build_vision_config(app) {
        lines.push(kv_line(
            "  Estimated RAM",
            &format!("{:.1} MB", cfg.estimate_memory_mb()),
            t.text,
            t.text_muted,
        ));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Pipeline: IR generation \u{2192} bicubic upscale \u{2192} im2col \u{2192} GEMM convolution",
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(Span::styled(
        "  The convolution step uses HPC multiply_hpc_fused for W \u{00d7} patches.",
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
        &["[Enter] Run pipeline", "[Esc] Back"],
        frame,
        chunks[2],
        t,
    );
}

// --- Screen 4: Computing ----------------------------------------------------

pub fn render_vision_computing(
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
            "  COMPUTER VISION \u{2014} Processing",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    let (frac, phase_text, eta_text): (f64, String, String) = if let Some(ref task) = app.compute_task {
        let f = task.progress.fraction();
        let ph = app
            .vision_phase
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
                .title(" IR Pipeline ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    render_footer(&["[Esc] Cancel"], frame, chunks[2], t);
}

// --- Screen 5: Results ------------------------------------------------------

pub fn render_vision_results(
    app: &App,
    result: &VisionResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(14),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    // Header
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "  IR VISION PROCESSING COMPLETE \u{2713}",
            Style::default().fg(Color::Rgb(74, 222, 128)).add_modifier(Modifier::BOLD),
        ))),
        chunks[0],
    );

    // Summary
    let cfg = &result.config;
    let mut summary: Vec<Line> = Vec::new();
    summary.push(Line::from(""));
    summary.push(kv_line("  Input", &format!("{}x{}", cfg.input_rows, cfg.input_cols), t.text, t.text_muted));
    summary.push(kv_line("  Upscaled", &format!("{}x{}", result.upscaled_rows, result.upscaled_cols), t.text, t.text_muted));
    summary.push(kv_line("  Output", &format!("{}x{}", result.output_rows, result.output_cols), t.text, t.text_muted));
    summary.push(kv_line("  Kernel", cfg.kernel_type.label(), t.accent, t.text_muted));
    summary.push(kv_line("  im2col shape", &format!("{}x{}", result.im2col_rows, result.im2col_cols), t.text, t.text_muted));
    summary.push(kv_line("  Upscale", &format_duration(result.upscale_ms), t.text, t.text_muted));
    summary.push(kv_line("  im2col", &format_duration(result.im2col_ms), t.text, t.text_muted));
    summary.push(kv_line("  GEMM", &format_duration(result.gemm_ms), t.accent, t.text_muted));
    summary.push(kv_line("  Total", &format_duration(result.computation_ms), t.text, t.text_muted));
    summary.push(kv_line("  SNR", &format!("{:.1} dB", result.snr_estimate), t.text, t.text_muted));
    let save_hint = if app.vision_csv_saved { " (saved)" } else { "" };
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

    // ASCII heatmap preview of output
    render_ascii_heatmap(&result.output_matrix, result.output_rows, result.output_cols, frame, chunks[2], t);

    render_footer(
        &["[S] Save CSV", "[H] Heatmap", "[P] Pipeline", "[R] Re-run", "[Esc] Menu"],
        frame,
        chunks[3],
        t,
    );
}

/// Render a compact ASCII heatmap using block characters with heat colors.
fn render_ascii_heatmap(
    data: &[f64],
    rows: usize,
    cols: usize,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let avail_h = area.height.saturating_sub(3) as usize;
    let avail_w = area.width.saturating_sub(4) as usize;
    if avail_h == 0 || avail_w == 0 || data.is_empty() {
        return;
    }

    let min_val = data.iter().cloned().fold(f64::MAX, f64::min);
    let max_val = data.iter().cloned().fold(f64::MIN, f64::max);
    let range = (max_val - min_val).max(1e-15);

    // Sample rows and cols to fit display
    let step_r = (rows as f64 / avail_h as f64).max(1.0);
    let step_c = (cols as f64 / avail_w as f64).max(1.0);

    let blocks = [' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(" Output heatmap ({}x{}, range [{:.3}, {:.3}])", rows, cols, min_val, max_val),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));

    let display_rows = avail_h.min(rows);
    let display_cols = avail_w.min(cols);

    for ri in 0..display_rows {
        let r = ((ri as f64 * step_r) as usize).min(rows - 1);
        let mut spans: Vec<Span> = vec![Span::raw(" ")];
        for ci in 0..display_cols {
            let c = ((ci as f64 * step_c) as usize).min(cols - 1);
            let val = data[r * cols + c];
            let norm = ((val - min_val) / range).clamp(0.0, 1.0);
            let idx = (norm * 4.0).round() as usize;
            let ch = blocks[idx.min(4)];
            let color = heat_color(norm);
            spans.push(Span::styled(String::from(ch), Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    let visible_h = area.height.saturating_sub(2) as usize;
    let visible: Vec<Line> = lines.into_iter().take(visible_h).collect();
    frame.render_widget(
        Paragraph::new(visible).block(
            Block::default()
                .title(" Thermal Map ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        area,
    );
}

/// Map [0,1] to a blue-green-yellow-red heat gradient.
fn heat_color(norm: f64) -> Color {
    let n = norm.clamp(0.0, 1.0);
    if n < 0.25 {
        let t = n / 0.25;
        Color::Rgb(0, (t * 128.0) as u8, (255.0 - t * 127.0) as u8)
    } else if n < 0.5 {
        let t = (n - 0.25) / 0.25;
        Color::Rgb(0, (128.0 + t * 127.0) as u8, (128.0 - t * 128.0) as u8)
    } else if n < 0.75 {
        let t = (n - 0.5) / 0.25;
        Color::Rgb((t * 255.0) as u8, 255, 0)
    } else {
        let t = (n - 0.75) / 0.25;
        Color::Rgb(255, (255.0 - t * 255.0) as u8, 0)
    }
}

// ============================================================================
//  OVERLAY RENDERERS
// ============================================================================

pub fn render_vision_overlay(
    app: &App,
    result: &VisionResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    match app.overlay {
        Overlay::VisionHeatmap => render_heatmap_overlay(result, frame, area, t),
        Overlay::VisionPipeline => render_pipeline_overlay(result, frame, area, t),
        _ => {}
    }
}

/// [H] Full-screen heatmap overlay
fn render_heatmap_overlay(
    result: &VisionResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.95, 0.92);
    frame.render_widget(Clear, popup);

    let inner = Rect {
        x: popup.x + 1,
        y: popup.y + 1,
        width: popup.width.saturating_sub(2),
        height: popup.height.saturating_sub(4),
    };

    render_ascii_heatmap(
        &result.output_matrix,
        result.output_rows,
        result.output_cols,
        frame,
        inner,
        t,
    );

    // Footer inside popup
    let footer_area = Rect {
        x: popup.x,
        y: popup.y + popup.height.saturating_sub(2),
        width: popup.width,
        height: 1,
    };
    render_footer(&["[Esc] Close"], frame, footer_area, t);

    frame.render_widget(
        Block::default()
            .title(" Heatmap [H] ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(t.accent))
            .style(Style::default().bg(t.bg)),
        popup,
    );
}

/// [P] Pipeline breakdown overlay
fn render_pipeline_overlay(
    result: &VisionResult,
    frame: &mut ratatui::Frame,
    area: Rect,
    t: &ThemeColors,
) {
    let popup = centered_popup(area, 0.85, 0.7);
    frame.render_widget(Clear, popup);

    let total = result.computation_ms.max(0.001);
    let stages = [
        ("IR Data Generation", result.computation_ms - result.upscale_ms - result.im2col_ms - result.gemm_ms),
        ("Bicubic Upscale", result.upscale_ms),
        ("im2col Transform", result.im2col_ms),
        ("GEMM Convolution", result.gemm_ms),
    ];
    let max_ms = stages.iter().map(|s| s.1).fold(0.0f64, f64::max).max(0.001);
    let bar_max_w = (popup.width as usize).saturating_sub(40);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(" Pipeline Breakdown \u{2014} Total: {}", format_duration(total)),
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    let colors = [
        Color::Rgb(96, 165, 250),   // blue
        Color::Rgb(74, 222, 128),    // green
        Color::Rgb(250, 204, 21),    // yellow
        Color::Rgb(248, 113, 113),   // red
    ];

    for (i, (name, ms)) in stages.iter().enumerate() {
        let pct = ms / total * 100.0;
        let bar_len = ((ms / max_ms) * bar_max_w as f64) as usize;
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<22} ", name),
                Style::default().fg(t.text_muted),
            ),
            Span::styled(
                "\u{2593}".repeat(bar_len.max(1)),
                Style::default().fg(colors[i]),
            ),
            Span::styled(
                format!(" {} ({:.1}%)", format_duration(*ms), pct),
                Style::default().fg(t.text),
            ),
        ]));
        lines.push(Line::from(""));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  Input: {}x{}  \u{2192}  Upscaled: {}x{}  \u{2192}  Output: {}x{}",
            result.config.input_rows, result.config.input_cols,
            result.upscaled_rows, result.upscaled_cols,
            result.output_rows, result.output_cols,
        ),
        Style::default().fg(t.text_dim),
    )));
    lines.push(Line::from(Span::styled(
        format!("  im2col: {}x{} patch matrix  |  SNR: {:.1} dB",
            result.im2col_rows, result.im2col_cols, result.snr_estimate,
        ),
        Style::default().fg(t.text_dim),
    )));

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
                .title(" Pipeline [P] ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.accent))
                .style(Style::default().bg(t.bg)),
        ),
        popup,
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
