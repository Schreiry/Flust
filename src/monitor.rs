// monitor.rs — Performance Monitor: ratatui TUI in a separate console window.
//
// Architecture: Launched as a separate process via `--monitor` CLI flag.
// On Windows, the main process spawns us with CREATE_NEW_CONSOLE for a dedicated window.
// This mirrors Fluminum's PerformanceMonitor but replaces Windows-only PDH
// with cross-platform sysinfo crate, and replaces manual char-buffer rendering
// with ratatui widgets styled through the unified Theme system.
//
// Redesigned in Chapter 8: "Minimal Precision" aesthetic — btop++ level clarity.
// Large objects, clear hierarchy, readable at a glance.

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem, Paragraph};
use ratatui::Terminal;
use sysinfo::System;

use crate::common::{Theme, ThemeKind};
use crate::system::SystemInfo;

// ─── Monitor State ─────────────────────────────────────────────

struct MonitorState {
    sys: System,
    sys_info: SystemInfo,

    // CPU tracking
    total_cpu_pct: f32,
    per_core_cpu: Vec<f32>,
    cpu_history: VecDeque<u64>,  // raw values 0..100, one per sample
    cpu_history_ema: VecDeque<f64>, // EMA-smoothed history for second line
    max_history: usize,

    // RAM tracking
    used_ram_mb: u64,
    total_ram_mb: u64,
    available_ram_mb: u64,

    // CPU temperature (current + session stats)
    cpu_temp: Option<f64>,
    temp_min: Option<f64>,
    temp_max: Option<f64>,
    temp_sum: f64,
    temp_count: u64,
    last_temp_check: Instant,

    // Timing & config
    last_collect: Instant,
    refresh_ms: u64,
    uptime: Instant,

    // Theme (runtime-switchable with [T])
    theme_kind: ThemeKind,

    // Computation history overlay
    show_history: bool,
    history_records: Vec<crate::io::HistoryRecord>,
    history_scroll: usize,
}

impl MonitorState {
    fn new(sys_info: SystemInfo) -> Self {
        // First refresh establishes a baseline for CPU usage delta.
        // The sleep is done in run_monitor_tui AFTER showing a loading frame,
        // so we only do a minimal setup here.
        let mut sys = System::new_all();
        sys.refresh_all();

        let total_ram = sys.total_memory() / (1024 * 1024);
        let avail_ram = sys.available_memory() / (1024 * 1024);

        Self {
            sys,
            sys_info,
            total_cpu_pct: 0.0,
            per_core_cpu: Vec::new(),
            cpu_history: VecDeque::with_capacity(120),
            cpu_history_ema: VecDeque::with_capacity(120),
            max_history: 120, // 120 samples × 500ms = 60 seconds of history
            used_ram_mb: total_ram.saturating_sub(avail_ram),
            total_ram_mb: total_ram,
            available_ram_mb: avail_ram,
            cpu_temp: None,
            temp_min: None,
            temp_max: None,
            temp_sum: 0.0,
            temp_count: 0,
            last_temp_check: Instant::now() - Duration::from_secs(10), // force immediate check
            last_collect: Instant::now(),
            refresh_ms: 500,
            uptime: Instant::now(),
            theme_kind: ThemeKind::Amber,
            show_history: false,
            history_records: Vec::new(),
            history_scroll: 0,
        }
    }

    fn tick(&mut self) {
        if self.last_collect.elapsed() >= Duration::from_millis(self.refresh_ms) {
            self.collect();
            self.last_collect = Instant::now();
        }
    }

    fn collect(&mut self) {
        self.sys.refresh_all();

        let cpus = self.sys.cpus();
        self.total_cpu_pct = if cpus.is_empty() {
            0.0
        } else {
            cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
        };
        self.per_core_cpu = cpus.iter().map(|c| c.cpu_usage()).collect();

        self.total_ram_mb = self.sys.total_memory() / (1024 * 1024);
        self.available_ram_mb = self.sys.available_memory() / (1024 * 1024);
        self.used_ram_mb = self.total_ram_mb.saturating_sub(self.available_ram_mb);

        let cpu_val = (self.total_cpu_pct as u64).min(100);
        if self.cpu_history.len() >= self.max_history {
            self.cpu_history.pop_front();
            self.cpu_history_ema.pop_front();
        }
        self.cpu_history.push_back(cpu_val);

        // EMA smoothing (alpha=0.35): blends current and previous smoothed value
        let alpha = 0.35_f64;
        let last_ema = self.cpu_history_ema.back().copied().unwrap_or(cpu_val as f64);
        let new_ema = alpha * cpu_val as f64 + (1.0 - alpha) * last_ema;
        self.cpu_history_ema.push_back(new_ema);

        // Refresh CPU temperature every 2 seconds (PowerShell is slow)
        if self.last_temp_check.elapsed() >= Duration::from_secs(2) {
            self.cpu_temp = crate::system::get_cpu_temperature();
            self.last_temp_check = Instant::now();
        }

        // Track temperature session stats
        if let Some(t) = self.cpu_temp {
            self.temp_min = Some(self.temp_min.map_or(t, |m: f64| m.min(t)));
            self.temp_max = Some(self.temp_max.map_or(t, |m: f64| m.max(t)));
            self.temp_sum += t;
            self.temp_count += 1;
        }
    }

    fn uptime_secs(&self) -> u64 {
        self.uptime.elapsed().as_secs()
    }
}

// ─── Public Entry Point ────────────────────────────────────────

/// Called from main.rs when launched with `--monitor` flag.
pub fn run_monitor() {
    let sys_info = SystemInfo::detect();
    if let Err(e) = run_monitor_tui(sys_info) {
        eprintln!("Monitor error: {e}");
    }
}

fn run_monitor_tui(sys_info: SystemInfo) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Show loading frame immediately — prevents blank black screen
    terminal.draw(|f| render_loading(f))?;

    let mut state = MonitorState::new(sys_info);

    // Wait for sysinfo CPU delta baseline (required for accurate first reading)
    std::thread::sleep(Duration::from_millis(250));
    state.collect();

    loop {
        state.tick();
        terminal.draw(|f| render(&state, f))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if state.show_history {
                        // History overlay controls
                        match key.code {
                            KeyCode::Char('c') | KeyCode::Char('C') | KeyCode::Esc => {
                                state.show_history = false;
                            }
                            KeyCode::Up => {
                                state.history_scroll = state.history_scroll.saturating_sub(1);
                            }
                            KeyCode::Down => {
                                if !state.history_records.is_empty() {
                                    state.history_scroll = (state.history_scroll + 1)
                                        .min(state.history_records.len().saturating_sub(1));
                                }
                            }
                            _ => {}
                        }
                    } else {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => break,
                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                if state.refresh_ms > 100 {
                                    state.refresh_ms -= 100;
                                }
                            }
                            KeyCode::Char('-') => {
                                if state.refresh_ms < 2000 {
                                    state.refresh_ms += 100;
                                }
                            }
                            KeyCode::Char('r') | KeyCode::Char('R') => {
                                state.refresh_ms = 500;
                            }
                            KeyCode::Char('t') | KeyCode::Char('T') => {
                                state.theme_kind = match state.theme_kind {
                                    ThemeKind::Amber => ThemeKind::Cyan,
                                    ThemeKind::Cyan  => ThemeKind::Steel,
                                    ThemeKind::Steel => ThemeKind::Amber,
                                };
                            }
                            KeyCode::Char('c') | KeyCode::Char('C') => {
                                state.show_history = true;
                                state.history_scroll = 0;
                                match crate::io::load_history() {
                                    Ok(records) => {
                                        // Scroll to end (newest entries)
                                        let n = records.len();
                                        state.history_records = records;
                                        state.history_scroll = n.saturating_sub(1);
                                    }
                                    Err(_) => {
                                        state.history_records.clear();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn render_loading(frame: &mut ratatui::Frame) {
    let area = frame.size();
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "  FLUST  ",
                Style::default().fg(Theme::ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Initializing performance monitor...",
                Style::default().fg(Theme::TEXT_MUTED),
            ),
        ]))
        .style(Style::default().bg(Theme::BG)),
        area,
    );
}

// ─── Rendering ─────────────────────────────────────────────────

fn render(state: &MonitorState, frame: &mut ratatui::Frame) {
    let area = frame.size();
    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::default().style(Style::default().bg(Theme::BG)),
        area,
    );

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(10), // CPU sparkline + RAM (horizontal split)
            Constraint::Min(6),    // per-core CPU bars
            Constraint::Length(1), // footer
        ])
        .split(area);

    render_header(state, frame, chunks[0]);
    render_metrics(state, frame, chunks[1]);
    if state.show_history {
        render_history_overlay(state, frame, chunks[2]);
    } else {
        render_cores(state, frame, chunks[2]);
    }
    render_footer(state, frame, chunks[3]);
}

// ─── Header ────────────────────────────────────────────────────

fn render_header(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let secs = state.uptime_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;

    let mut spans = vec![
        Span::styled(
            "  FLUST  ",
            Style::default()
                .fg(Theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(&state.sys_info.cpu_brand, Style::default().fg(Theme::TEXT)),
        Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(
            format!(
                "{}C/{}T",
                state.sys_info.physical_cores, state.sys_info.logical_cores
            ),
            Style::default().fg(Theme::TEXT_BRIGHT),
        ),
        Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(
            state.sys_info.simd_level.display_name(),
            Style::default().fg(Theme::ACCENT),
        ),
        Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(
            format!("{h:02}:{m:02}:{s:02}"),
            Style::default().fg(Theme::TEXT_MUTED),
        ),
    ];

    // CPU base frequency
    {
        let freq_str = if state.sys_info.peak_estimate.freq_source == "fallback" {
            format!("~{:.1}GHz", state.sys_info.base_freq_ghz)
        } else {
            format!("{:.1}GHz", state.sys_info.base_freq_ghz)
        };
        spans.push(Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)));
        spans.push(Span::styled(freq_str, Style::default().fg(Theme::TEXT_MUTED)));
    }

    // L3 cache size (detected at startup, static)
    if let Some(l3_kb) = state.sys_info.l3_cache_kb {
        spans.push(Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)));
        let l3_str = if l3_kb >= 1024 {
            format!("L3 {:.0}MB", l3_kb as f64 / 1024.0)
        } else {
            format!("L3 {l3_kb}KB")
        };
        spans.push(Span::styled(l3_str, Style::default().fg(Theme::TEXT_MUTED)));
    }

    // CPU temperature — always shown; "–°C" when sensor data is unavailable
    spans.push(Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)));
    match state.cpu_temp {
        Some(temp) => {
            let temp_color: Color = if temp > 90.0 {
                Color::Red
            } else if temp > 80.0 {
                Color::Rgb(255, 140, 0)
            } else if temp > 60.0 {
                Color::Yellow
            } else {
                Color::Green
            };
            let label = if temp > 90.0 {
                format!("{temp:.0}°C ⚠")
            } else {
                format!("{temp:.0}°C")
            };
            spans.push(Span::styled(
                label,
                Style::default().fg(temp_color).add_modifier(Modifier::BOLD),
            ));
            // Show session stats when we have at least 3 samples
            if state.temp_count >= 3 {
                let avg = state.temp_sum / state.temp_count as f64;
                let min_t = state.temp_min.unwrap_or(temp);
                let max_t = state.temp_max.unwrap_or(temp);
                spans.push(Span::styled(
                    format!(" ↑{max_t:.0} ↓{min_t:.0} ø{avg:.0}"),
                    Style::default().fg(Theme::TEXT_DIM),
                ));
            }
        }
        None => {
            spans.push(Span::styled(
                "–°C",
                Style::default().fg(Theme::TEXT_DIM),
            ));
        }
    }

    spans.push(Span::styled("  ", Style::default()));
    let title_line = Line::from(spans);

    let header = Paragraph::new(title_line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Theme::block_style())
            .style(Style::default().bg(Theme::BG)),
    );
    frame.render_widget(header, area);
}

// ─── CPU Sparkline + RAM ───────────────────────────────────────

fn render_metrics(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    render_cpu_history(state, frame, cols[0]);
    render_memory(state, frame, cols[1]);
}

fn render_cpu_history(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let data_len = state.cpu_history.len();
    let current_pct = state.cpu_history.back().copied().unwrap_or(0);
    let accent = state.theme_kind.colors().accent;
    let color = Theme::load_color(current_pct as f32);

    // Build (x, y) point arrays where x=0 is newest, x=-(n-1) is oldest
    // x range: [-(max_history-1), 0]
    let raw_data: Vec<(f64, f64)> = state.cpu_history
        .iter()
        .enumerate()
        .map(|(i, &v)| ((i as f64) - (data_len as f64 - 1.0), v as f64))
        .collect();

    let ema_data: Vec<(f64, f64)> = state.cpu_history_ema
        .iter()
        .enumerate()
        .map(|(i, &v)| ((i as f64) - (data_len as f64 - 1.0), v))
        .collect();

    let x_min = -((state.max_history as f64) - 1.0);
    let secs = state.max_history as u64 * state.refresh_ms / 1000;

    // Guard: ratatui Chart panics if labels vec is empty
    let x_labels = if data_len >= 2 {
        vec![
            Span::styled(format!("-{secs}s"), Style::default().fg(Theme::TEXT_DIM)),
            Span::styled("now", Style::default().fg(Theme::TEXT_DIM).add_modifier(Modifier::BOLD)),
        ]
    } else {
        vec![
            Span::styled("–", Style::default().fg(Theme::TEXT_DIM)),
            Span::styled("now", Style::default().fg(Theme::TEXT_DIM)),
        ]
    };
    let y_labels = vec![
        Span::styled("  0%", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(" 50%", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled("100%", Style::default().fg(Theme::TEXT_DIM).add_modifier(Modifier::BOLD)),
    ];

    // Peak / avg for title
    let peak = state.cpu_history.iter().max().copied().unwrap_or(0);
    let avg = if data_len > 0 {
        state.cpu_history.iter().sum::<u64>() as f64 / data_len as f64
    } else {
        0.0
    };

    let mut datasets = vec![
        Dataset::default()
            .data(&raw_data)
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(color)),
    ];
    // Add smoothed line in dimmer accent color only if we have EMA data
    if !ema_data.is_empty() {
        datasets.push(
            Dataset::default()
                .data(&ema_data)
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(accent).add_modifier(Modifier::DIM)),
        );
    }

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" CPU LOAD ", Style::default().fg(Theme::TEXT_DIM)),
                    Span::styled(
                        format!("{:.1}%", state.total_cpu_pct),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  peak:{peak}%  avg:{avg:.1}%"),
                        Style::default().fg(Theme::TEXT_DIM),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_style(Theme::block_style())
                .style(Style::default().bg(Theme::BG)),
        )
        .x_axis(
            Axis::default()
                .bounds([x_min, 0.0])
                .labels(x_labels)
                .style(Style::default().fg(Theme::TEXT_DIM)),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(y_labels)
                .style(Style::default().fg(Theme::TEXT_DIM)),
        );

    frame.render_widget(chart, area);
}

fn render_memory(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let pct = if state.total_ram_mb > 0 {
        (state.used_ram_mb * 100 / state.total_ram_mb) as u16
    } else {
        0
    };
    let used_gb = state.used_ram_mb as f64 / 1024.0;
    let total_gb = state.total_ram_mb as f64 / 1024.0;
    let free_gb = state.available_ram_mb as f64 / 1024.0;
    let color = Theme::load_color(pct as f32);

    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(
                format!("{used_gb:.1} GB"),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  /  ", Style::default().fg(Theme::TEXT_DIM)),
            Span::styled(
                format!("{total_gb:.1} GB"),
                Style::default().fg(Theme::TEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Free  ", Style::default().fg(Theme::TEXT_DIM)),
            Span::styled(
                format!("{free_gb:.1} GB"),
                Style::default().fg(Theme::TEXT_MUTED),
            ),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(
            " MEMORY ",
            Style::default().fg(Theme::TEXT_DIM),
        ))
        .borders(Borders::ALL)
        .border_style(Theme::block_style())
        .style(Style::default().bg(Theme::BG));

    let para = Paragraph::new(content).block(block);
    frame.render_widget(para, area);

    // Gauge in the lower portion of the memory block
    if area.height > 4 {
        let gauge_area = Rect {
            x: area.x + 1,
            y: area.y + area.height.saturating_sub(3),
            width: area.width.saturating_sub(2),
            height: 1,
        };
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(color).bg(Theme::SURFACE))
            .percent(pct)
            .label(Span::styled(
                format!("{pct}%"),
                Style::default().fg(if pct > 70 {
                    Theme::BG
                } else {
                    Theme::TEXT
                }),
            ));
        frame.render_widget(gauge, gauge_area);
    }
}

// ─── Per-Core CPU Bars ─────────────────────────────────────────

fn render_cores(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let cores = &state.per_core_cpu;

    let block = Block::default()
        .title(Span::styled(
            " PROCESSOR CORES ",
            Style::default().fg(Theme::TEXT_DIM),
        ))
        .borders(Borders::ALL)
        .border_style(Theme::block_style())
        .style(Style::default().bg(Theme::BG));

    if cores.is_empty() {
        frame.render_widget(
            Paragraph::new("  No CPU data available").block(block),
            area,
        );
        return;
    }

    frame.render_widget(block, area);

    let inner = area.inner(&Margin::new(2, 1));
    let col_count = 2usize;
    let cores_per_col = (cores.len() + col_count - 1) / col_count;

    // Decide whether to add gaps between rows
    let available_rows = inner.height as usize;
    let with_gaps = cores_per_col * 2 <= available_rows;
    let row_height: u16 = if with_gaps { 2 } else { 1 };

    // Bar width: fill available space minus label (4 chars) and percent (8 chars)
    let col_width = inner.width / col_count as u16;
    let bar_width = (col_width as usize).saturating_sub(14).max(10).min(30);

    for col in 0..col_count {
        let col_x = inner.x + (col as u16) * col_width;

        for row in 0..cores_per_col {
            let core_idx = col * cores_per_col + row;
            if core_idx >= cores.len() {
                break;
            }

            let pct = cores[core_idx];
            let row_y = inner.y + (row as u16) * row_height;
            if row_y >= inner.y + inner.height {
                break;
            }

            let color = Theme::load_color(pct);
            let filled = ((pct / 100.0) * bar_width as f32) as usize;
            let empty = bar_width.saturating_sub(filled);

            let core_line = Line::from(vec![
                Span::styled(
                    format!("{:>2}  ", core_idx),
                    Style::default().fg(Theme::TEXT_DIM),
                ),
                Span::styled("▓".repeat(filled), Style::default().fg(color)),
                Span::styled("░".repeat(empty), Style::default().fg(Theme::BORDER)),
                Span::styled(
                    format!("  {:>5.1}%", pct),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]);

            let core_area = Rect {
                x: col_x,
                y: row_y,
                width: col_width,
                height: 1,
            };
            frame.render_widget(Paragraph::new(core_line), core_area);
        }
    }
}

// ─── Footer ────────────────────────────────────────────────────

fn render_footer(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let accent = state.theme_kind.colors().accent;
    let key = Style::default().fg(accent).add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Theme::TEXT_DIM);
    let muted = Style::default().fg(Theme::TEXT_MUTED);
    let dot = Span::styled("  ·  ", dim);

    let theme_label = match state.theme_kind {
        ThemeKind::Amber => "Amber",
        ThemeKind::Cyan  => "Cyan",
        ThemeKind::Steel => "Steel",
    };

    let line = Line::from(vec![
        Span::styled("  [Q]", key), Span::styled(" Quit", muted), dot.clone(),
        Span::styled("[R]", key), Span::styled(" Reset", muted), dot.clone(),
        Span::styled("[+/-]", key), Span::styled(" Speed", muted), dot.clone(),
        Span::styled(format!("{}ms", state.refresh_ms), Style::default().fg(Theme::TEXT)), dot.clone(),
        Span::styled("[T]", key),
        Span::styled(format!(" Theme:{theme_label}"), muted), dot.clone(),
        Span::styled("[C]", key), Span::styled(" History", muted), dot.clone(),
        Span::styled(
            format!("Samples: {}", state.cpu_history.len()),
            dim,
        ),
    ]);

    let footer = Paragraph::new(line).style(Style::default().bg(Theme::SURFACE));
    frame.render_widget(footer, area);
}

// ─── History Overlay ───────────────────────────────────────────

fn render_history_overlay(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let accent = state.theme_kind.colors().accent;

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" COMPUTATION HISTORY ", Style::default().fg(Theme::TEXT_DIM)),
            Span::styled("[C/Esc] Close  [↑↓] Scroll", Style::default().fg(Theme::TEXT_DIM)),
        ]))
        .borders(Borders::ALL)
        .border_style(Theme::block_style())
        .style(Style::default().bg(Theme::BG));

    if state.history_records.is_empty() {
        let msg = Paragraph::new(Line::from(Span::styled(
            "  No history found. Run a multiplication first.",
            Style::default().fg(Theme::TEXT_MUTED),
        )))
        .block(block);
        frame.render_widget(msg, area);
        return;
    }

    let inner = area.inner(&Margin::new(1, 1));
    frame.render_widget(block, area);

    let visible_rows = inner.height as usize;
    let total = state.history_records.len();
    let scroll = state.history_scroll.min(total.saturating_sub(1));
    let start = if scroll + 1 > visible_rows {
        scroll + 1 - visible_rows
    } else {
        0
    };
    let end = (start + visible_rows).min(total);

    let items: Vec<ListItem> = state.history_records[start..end]
        .iter()
        .enumerate()
        .map(|(i, rec)| {
            let abs_idx = start + i;
            let is_selected = abs_idx == scroll;
            let algo_short = if rec.algorithm.len() > 28 {
                format!("{}…", &rec.algorithm[..27])
            } else {
                rec.algorithm.clone()
            };
            let line = Line::from(vec![
                Span::styled(
                    format!("{:>28}  ", algo_short),
                    Style::default().fg(if is_selected { accent } else { Theme::TEXT }),
                ),
                Span::styled(
                    format!("{:>6}×{:<6}  ", rec.size, rec.size),
                    Style::default().fg(Theme::TEXT_MUTED),
                ),
                Span::styled(
                    format!("{:>8.1}ms  ", rec.compute_ms),
                    Style::default().fg(Theme::TEXT),
                ),
                Span::styled(
                    format!("{:>6.2} GFlops  ", rec.gflops),
                    Style::default().fg(if is_selected { accent } else { Theme::TEXT_BRIGHT }),
                ),
                Span::styled(
                    format!("{:>4}T  {:>6}", rec.threads, rec.simd),
                    Style::default().fg(Theme::TEXT_MUTED),
                ),
            ]);
            if is_selected {
                ListItem::new(line).style(Style::default().bg(Theme::SURFACE))
            } else {
                ListItem::new(line)
            }
        })
        .collect();

    let list = List::new(items).style(Style::default().bg(Theme::BG));
    frame.render_widget(list, inner);

    // Scroll indicator (right edge)
    if total > visible_rows {
        let pct = scroll as f64 / total.saturating_sub(1) as f64;
        let thumb_y = (pct * visible_rows.saturating_sub(1) as f64) as u16;
        let scroll_area = Rect {
            x: area.x + area.width.saturating_sub(1),
            y: inner.y + thumb_y,
            width: 1,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new("▐").style(Style::default().fg(accent)),
            scroll_area,
        );
    }
}

// ─── Spawn in New Window ───────────────────────────────────────

/// Spawn the performance monitor in a separate console window.
pub fn spawn_monitor_window() {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_CONSOLE: u32 = 0x00000010;
        match std::env::current_exe() {
            Ok(exe) => {
                match std::process::Command::new(exe)
                    .arg("--monitor")
                    .creation_flags(CREATE_NEW_CONSOLE)
                    .spawn()
                {
                    Ok(_) => {}
                    Err(e) => eprintln!("Failed to spawn monitor: {e}"),
                }
            }
            Err(e) => eprintln!("Failed to get executable path: {e}"),
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(exe) = std::env::current_exe() {
            let exe_str = exe.to_string_lossy().to_string();
            let terminals = [
                ("xterm", vec!["-e", &exe_str, "--monitor"]),
                ("gnome-terminal", vec!["--", &exe_str, "--monitor"]),
                ("konsole", vec!["-e", &exe_str, "--monitor"]),
            ];
            for (term, args) in &terminals {
                if std::process::Command::new(term)
                    .args(args.clone())
                    .spawn()
                    .is_ok()
                {
                    return;
                }
            }
            eprintln!("No supported terminal emulator found");
        }
    }
}
