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
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, Paragraph, Sparkline};
use ratatui::Terminal;
use sysinfo::System;

use crate::common::Theme;
use crate::system::SystemInfo;

// ─── Monitor State ─────────────────────────────────────────────

struct MonitorState {
    sys: System,
    sys_info: SystemInfo,

    // CPU tracking
    total_cpu_pct: f32,
    per_core_cpu: Vec<f32>,
    cpu_history: VecDeque<u64>, // sparkline data, values in 0..100
    max_history: usize,

    // RAM tracking
    used_ram_mb: u64,
    total_ram_mb: u64,
    available_ram_mb: u64,

    // Timing & config
    last_collect: Instant,
    refresh_ms: u64,
    uptime: Instant,
}

impl MonitorState {
    fn new(sys_info: SystemInfo) -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        // sysinfo requires a baseline refresh for CPU usage delta calculation.
        // Without this pause the first cpu_usage() call returns 0%.
        std::thread::sleep(Duration::from_millis(250));
        sys.refresh_all();

        let total_ram = sys.total_memory() / (1024 * 1024);
        let avail_ram = sys.available_memory() / (1024 * 1024);

        Self {
            sys,
            sys_info,
            total_cpu_pct: 0.0,
            per_core_cpu: Vec::new(),
            cpu_history: VecDeque::with_capacity(120),
            max_history: 120, // 120 samples × 500ms = 60 seconds of history
            used_ram_mb: total_ram.saturating_sub(avail_ram),
            total_ram_mb: total_ram,
            available_ram_mb: avail_ram,
            last_collect: Instant::now(),
            refresh_ms: 500,
            uptime: Instant::now(),
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
        }
        self.cpu_history.push_back(cpu_val);
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

    let mut state = MonitorState::new(sys_info);

    loop {
        state.tick();
        terminal.draw(|f| render(&state, f))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
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
                        _ => {}
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
    render_cores(state, frame, chunks[2]);
    render_footer(state, frame, chunks[3]);
}

// ─── Header ────────────────────────────────────────────────────

fn render_header(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let secs = state.uptime_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;

    let title_line = Line::from(vec![
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
        Span::styled("  ", Style::default()),
    ]);

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
    let cpu_data: Vec<u64> = state.cpu_history.iter().copied().collect();
    let current_pct = cpu_data.last().copied().unwrap_or(0);
    let data_len = cpu_data.len();

    // Normalize: if value > 0 but very small, bump to minimum visible height
    let normalized: Vec<u64> = cpu_data
        .iter()
        .map(|&v| if v == 0 { 0 } else { v.max(3) })
        .collect();

    let color = Theme::load_color(current_pct as f32);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" CPU LOAD ", Style::default().fg(Theme::TEXT_DIM)),
            Span::styled(
                format!("{:.1}%", state.total_cpu_pct),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ", Style::default()),
        ]))
        .borders(Borders::ALL)
        .border_style(Theme::block_style())
        .style(Style::default().bg(Theme::BG));

    let sparkline = Sparkline::default()
        .block(block)
        .data(&normalized)
        .max(100)
        .style(Style::default().fg(color));

    frame.render_widget(sparkline, area);

    // History info overlay at bottom of sparkline area
    if area.height > 3 {
        let info_area = Rect {
            x: area.x + 2,
            y: area.y + area.height.saturating_sub(2),
            width: area.width.saturating_sub(4),
            height: 1,
        };
        let peak = cpu_data.iter().max().copied().unwrap_or(0);
        let avg = if data_len > 0 {
            cpu_data.iter().sum::<u64>() as f64 / data_len as f64
        } else {
            0.0
        };
        let info = Paragraph::new(Line::from(Span::styled(
            format!(" {data_len}s history  ·  peak: {peak}%  ·  avg: {avg:.1}% "),
            Style::default().fg(Theme::TEXT_DIM).bg(Theme::BG),
        )));
        frame.render_widget(info, info_area);
    }
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
    let line = Line::from(vec![
        Span::styled(
            "  [Q]",
            Style::default()
                .fg(Theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Quit", Style::default().fg(Theme::TEXT_MUTED)),
        Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(
            "[R]",
            Style::default()
                .fg(Theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Reset", Style::default().fg(Theme::TEXT_MUTED)),
        Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(
            "[+/-]",
            Style::default()
                .fg(Theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Speed", Style::default().fg(Theme::TEXT_MUTED)),
        Span::styled("  ·  ", Style::default().fg(Theme::TEXT_DIM)),
        Span::styled(
            format!("{}ms", state.refresh_ms),
            Style::default().fg(Theme::TEXT),
        ),
        Span::styled(
            format!("  ·  Samples: {}", state.cpu_history.len()),
            Style::default().fg(Theme::TEXT_DIM),
        ),
    ]);

    let footer = Paragraph::new(line).style(Style::default().bg(Theme::SURFACE));
    frame.render_widget(footer, area);
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
