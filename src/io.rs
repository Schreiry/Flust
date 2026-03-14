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

// ─── CSV Logging ──────────────────────────────────────────────────────────────
//
// Appends benchmark results to a CSV file. Creates the file with headers if missing.
// Format matches Fluminum's logMultiplicationResultToCSV for continuity.

use std::io::Write;
use std::path::Path;

/// A single benchmark record for CSV export.
pub struct CsvRecord {
    pub timestamp: String,
    pub algorithm: String,
    pub size_m: usize,
    pub size_n: usize,
    pub size_p: usize,
    pub compute_time_ms: f64,
    pub total_time_ms: f64,
    pub gflops: f64,
    pub simd_level: String,
    pub threads: usize,
    pub tile_size: Option<usize>,
    pub peak_ram_mb: u64,
}

const CSV_HEADER: &str = "timestamp,algorithm,size_m,size_n,size_p,compute_ms,total_ms,gflops,simd,threads,tile_size,peak_ram_mb";

/// Append a benchmark result to CSV file. Creates file + header if it doesn't exist.
pub fn append_csv(path: &str, record: &CsvRecord) -> std::io::Result<()> {
    let file_exists = Path::new(path).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    if !file_exists {
        writeln!(file, "{}", CSV_HEADER)?;
    }

    writeln!(
        file,
        "{},{},{},{},{},{:.4},{:.4},{:.4},{},{},{},{}",
        record.timestamp,
        record.algorithm,
        record.size_m,
        record.size_n,
        record.size_p,
        record.compute_time_ms,
        record.total_time_ms,
        record.gflops,
        record.simd_level,
        record.threads,
        record.tile_size.map_or("N/A".to_string(), |v| v.to_string()),
        record.peak_ram_mb,
    )?;
    Ok(())
}

/// Save a matrix to CSV file (for small matrices).
pub fn save_matrix_csv(path: &str, matrix: &crate::matrix::Matrix) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            if j > 0 {
                write!(file, ",")?;
            }
            write!(file, "{:.6}", matrix.get(i, j))?;
        }
        writeln!(file)?;
    }
    Ok(())
}

/// Load a matrix from CSV file (plain comma-separated f64 rows, no header).
/// Infers dimensions from data: cols from first row, rows from line count.
pub fn load_matrix_csv(path: &str) -> std::io::Result<crate::matrix::Matrix> {
    let content = std::fs::read_to_string(path)?;
    let mut rows = 0usize;
    let mut cols = 0usize;
    let mut data = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let values: Vec<f64> = trimmed
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        if rows == 0 {
            cols = values.len();
        } else if values.len() != cols {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Row {} has {} columns, expected {}", rows, values.len(), cols),
            ));
        }
        data.extend(values);
        rows += 1;
    }

    if rows == 0 || cols == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Empty matrix file",
        ));
    }

    crate::matrix::Matrix::from_flat(rows, cols, data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

/// Get current timestamp as ISO-like string for CSV records.
pub fn timestamp_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        1970 + secs / 31536000, // approximate year
        (secs % 31536000) / 2592000 + 1, // approximate month
        (secs % 2592000) / 86400 + 1, // approximate day
        hours, mins, s
    )
}

/// Create a text-based progress bar: █████░░░░░
pub fn make_bar(pct: f32, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f32).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}
