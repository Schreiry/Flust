// monitor.rs — Performance Monitor: ratatui TUI in a separate console window.
//
// Architecture: Launched as a separate process via `--monitor` CLI flag.
// On Windows, the main process spawns us with CREATE_NEW_CONSOLE for a dedicated window.
// This mirrors Fluminum's PerformanceMonitor (CREATE_NEW_CONSOLE + PDH API)
// but replaces Windows-only PDH with cross-platform sysinfo crate,
// and replaces manual char-buffer rendering with ratatui widgets.
//
// The monitor is fully standalone — it collects system metrics independently.
// No IPC needed for basic system monitoring.

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Sparkline};
use ratatui::Terminal;
use sysinfo::System;

use crate::io as theme;
use crate::system::SystemInfo;

// ─── Monitor State ─────────────────────────────────────────────
//
// Single struct holds sysinfo::System, derived metrics, and sparkline history.
// No Arc/Mutex needed — everything runs on one thread in the monitor process.

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

    // Timing
    last_collect: Instant,
    collect_interval: Duration,
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
            collect_interval: Duration::from_millis(500),
            uptime: Instant::now(),
        }
    }

    /// Check if enough time has passed and collect new metrics if so.
    fn tick(&mut self) {
        if self.last_collect.elapsed() >= self.collect_interval {
            self.collect();
            self.last_collect = Instant::now();
        }
    }

    /// Refresh sysinfo and update all derived metrics.
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

        // Push to sparkline history (circular buffer)
        let cpu_val = (self.total_cpu_pct as u64).min(100);
        if self.cpu_history.len() >= self.max_history {
            self.cpu_history.pop_front();
        }
        self.cpu_history.push_back(cpu_val);
    }

    fn uptime_str(&self) -> String {
        let secs = self.uptime.elapsed().as_secs();
        let m = secs / 60;
        let s = secs % 60;
        format!("{m:02}:{s:02}")
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
    frame.render_widget(Block::default().style(theme::style_default()), area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // header + system info
            Constraint::Length(8),  // CPU sparkline + RAM
            Constraint::Min(6),    // per-core CPU bars
            Constraint::Length(2), // footer
        ])
        .split(area);

    render_header(state, frame, chunks[0]);
    render_metrics(state, frame, chunks[1]);
    render_per_core(state, frame, chunks[2]);
    render_footer(state, frame, chunks[3]);
}

fn render_header(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(
            "  ╔═══ FLUST PERFORMANCE MONITOR ═══╗",
            theme::style_title(),
        )),
        Line::from(vec![
            Span::styled("  CPU: ", theme::style_muted()),
            Span::styled(&state.sys_info.cpu_brand, theme::style_default()),
        ]),
        Line::from(vec![
            Span::styled("  Cores: ", theme::style_muted()),
            Span::styled(
                format!(
                    "{}C/{}T",
                    state.sys_info.physical_cores, state.sys_info.logical_cores
                ),
                theme::style_info(),
            ),
            Span::styled("  SIMD: ", theme::style_muted()),
            Span::styled(
                state.sys_info.simd_level.display_name(),
                theme::style_accent(),
            ),
            Span::styled("  Uptime: ", theme::style_muted()),
            Span::styled(state.uptime_str(), theme::style_default()),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_metrics(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // ── Left: CPU Sparkline ──
    // VecDeque → Vec for the Sparkline widget (needs contiguous &[u64])
    let cpu_data: Vec<u64> = state.cpu_history.iter().copied().collect();
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(Span::styled(
                    format!(" CPU LOAD ▸ {:.1}% ", state.total_cpu_pct),
                    theme::style_title(),
                ))
                .borders(Borders::ALL)
                .border_style(theme::style_accent())
                .style(theme::style_default()),
        )
        .data(&cpu_data)
        .max(100)
        .style(theme::style_accent());
    frame.render_widget(sparkline, cols[0]);

    // ── Right: RAM ──
    let ram_pct = if state.total_ram_mb > 0 {
        ((state.used_ram_mb as f64 / state.total_ram_mb as f64) * 100.0) as u16
    } else {
        0
    };

    let ram_color = if ram_pct >= 80 {
        theme::DANGER
    } else if ram_pct >= 50 {
        theme::ACCENT
    } else {
        theme::SUCCESS
    };

    let ram_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Used: ", theme::style_muted()),
            Span::styled(
                theme::format_memory_mb(state.used_ram_mb),
                theme::style_default(),
            ),
            Span::styled(" / ", theme::style_muted()),
            Span::styled(
                theme::format_memory_mb(state.total_ram_mb),
                theme::style_default(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Free: ", theme::style_muted()),
            Span::styled(
                theme::format_memory_mb(state.available_ram_mb),
                theme::style_success(),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ", theme::style_default()),
            Span::styled(
                format!("{} {ram_pct}%", make_bar(ram_pct as f32, 20)),
                Style::default().fg(ram_color).bg(theme::BG),
            ),
        ]),
    ];

    let ram_block = Block::default()
        .title(Span::styled(" MEMORY ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(ram_lines).block(ram_block), cols[1]);
}

fn render_per_core(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let cores = &state.per_core_cpu;

    if cores.is_empty() {
        let block = Block::default()
            .title(Span::styled(" PER-CORE CPU ", theme::style_title()))
            .borders(Borders::ALL)
            .border_style(theme::style_accent())
            .style(theme::style_default());
        frame.render_widget(
            Paragraph::new("  No CPU data available").block(block),
            area,
        );
        return;
    }

    // Adaptive column count: 2 for ≤16 cores, 4 for more
    let col_count = if cores.len() <= 16 { 2 } else { 4 };

    let mut lines: Vec<Line> = Vec::new();

    // Row-major layout: C00 C01 | C02 C03 ...
    for row_start in (0..cores.len()).step_by(col_count) {
        let mut spans: Vec<Span> = vec![Span::styled("  ", theme::style_default())];

        for col in 0..col_count {
            let idx = row_start + col;
            if idx >= cores.len() {
                break;
            }
            let pct = cores[idx];
            let color = if pct >= 80.0 {
                theme::DANGER
            } else if pct >= 50.0 {
                theme::ACCENT
            } else {
                theme::SUCCESS
            };

            let bar = make_bar(pct, 10);
            spans.push(Span::styled(
                format!("C{:02} ", idx),
                theme::style_muted(),
            ));
            spans.push(Span::styled(
                bar,
                Style::default().fg(color).bg(theme::BG),
            ));
            spans.push(Span::styled(
                format!(" {:5.1}%  ", pct),
                Style::default().fg(color).bg(theme::BG),
            ));
        }

        lines.push(Line::from(spans));
    }

    let block = Block::default()
        .title(Span::styled(" PER-CORE CPU ", theme::style_title()))
        .borders(Borders::ALL)
        .border_style(theme::style_accent())
        .style(theme::style_default());
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_footer(state: &MonitorState, frame: &mut ratatui::Frame, area: Rect) {
    let footer = Line::from(vec![
        Span::styled(" q/Esc ", theme::style_key_hint()),
        Span::styled("Quit  ", theme::style_muted()),
        Span::styled("  Refresh: 500ms  ", theme::style_muted()),
        Span::styled(
            format!("  Samples: {} ", state.cpu_history.len()),
            theme::style_muted(),
        ),
    ]);
    frame.render_widget(Paragraph::new(footer), area);
}

// ─── Utilities ─────────────────────────────────────────────────

/// Create a text-based progress bar: █████░░░░░
fn make_bar(pct: f32, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f32).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

// ─── Spawn in New Window ───────────────────────────────────────
//
// On Windows: CREATE_NEW_CONSOLE gives the monitor its own console window,
// identical to how Fluminum's PerformanceMonitor used CreateProcess().
// The monitor process is fully independent — closing it doesn't affect
// the main Flust process, and vice versa.

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
                    Ok(_) => {} // monitor launched successfully
                    Err(e) => eprintln!("Failed to spawn monitor: {e}"),
                }
            }
            Err(e) => eprintln!("Failed to get executable path: {e}"),
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Best-effort: try common terminal emulators on Linux/macOS
        if let Ok(exe) = std::env::current_exe() {
            let exe_str = exe.to_string_lossy().to_string();
            // Try in order: common terminal emulators
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
