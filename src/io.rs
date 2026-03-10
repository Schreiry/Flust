// io.rs — TUI theme, color palette, rendering helpers, CSV logging.
//
// Design philosophy: Valve/Half-Life industrial aesthetic meets Apple clarity.
// Dark background, amber/orange accents, clean typography with box-drawing.
// All color definitions centralized here so the entire look can be reskinned.

use ratatui::style::{Color, Modifier, Style};

// ─── Color Palette ──────────────────────────────────────────────────────────
// Inspired by Valve's Half-Life amber HUD + Apple's clean contrast ratios.

pub const BG: Color = Color::Rgb(15, 15, 20);           // near-black
pub const FG: Color = Color::Rgb(200, 200, 210);        // soft white
pub const ACCENT: Color = Color::Rgb(255, 176, 0);      // Valve amber/orange
pub const ACCENT_DIM: Color = Color::Rgb(180, 120, 0);  // muted amber
pub const HIGHLIGHT: Color = Color::Rgb(255, 200, 60);  // bright gold
pub const SUCCESS: Color = Color::Rgb(80, 220, 100);    // green — supported
pub const DANGER: Color = Color::Rgb(220, 60, 60);      // red — unsupported
pub const INFO: Color = Color::Rgb(80, 160, 255);       // blue — info
pub const MUTED: Color = Color::Rgb(100, 100, 120);     // grey — secondary text
pub const SURFACE: Color = Color::Rgb(25, 25, 35);      // slightly lighter bg for panels

// ─── Predefined Styles ─────────────────────────────────────────────────────

pub fn style_default() -> Style {
    Style::default().fg(FG).bg(BG)
}

pub fn style_accent() -> Style {
    Style::default().fg(ACCENT).bg(BG)
}

pub fn style_title() -> Style {
    Style::default()
        .fg(ACCENT)
        .bg(BG)
        .add_modifier(Modifier::BOLD)
}

pub fn style_selected() -> Style {
    Style::default()
        .fg(BG)
        .bg(ACCENT)
        .add_modifier(Modifier::BOLD)
}

pub fn style_success() -> Style {
    Style::default().fg(SUCCESS).bg(BG)
}

pub fn style_danger() -> Style {
    Style::default().fg(DANGER).bg(BG)
}

pub fn style_info() -> Style {
    Style::default().fg(INFO).bg(BG)
}

pub fn style_muted() -> Style {
    Style::default().fg(MUTED).bg(BG)
}

pub fn style_surface() -> Style {
    Style::default().fg(FG).bg(SURFACE)
}

pub fn style_key_hint() -> Style {
    Style::default()
        .fg(HIGHLIGHT)
        .bg(BG)
        .add_modifier(Modifier::BOLD)
}

// ─── Utility ────────────────────────────────────────────────────────────────

/// Format milliseconds into a human-readable string.
pub fn format_duration_ms(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.2} us", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.2} ms", ms)
    } else {
        format!("{:.3} s", ms / 1000.0)
    }
}

/// Format bytes into human-readable MB/GB.
pub fn format_memory_mb(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{} MB", mb)
    }
}
