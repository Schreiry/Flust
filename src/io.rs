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

// ─── Matrix Metadata ─────────────────────────────────────────────────────────
//
// Stored as `# key=value` comment lines at the top of matrix CSV files.
// This preserves compatibility: old plain-CSV files load fine (# lines skipped).
// Flust-generated matrices carry full provenance info for the Matrix Viewer.

pub struct MatrixMetadata {
    pub algorithm: Option<String>,
    pub timestamp: Option<String>,
    pub cpu: Option<String>,
    pub simd: Option<String>,
    pub threads: Option<usize>,
    pub compute_ms: Option<f64>,
    pub size_rows: Option<usize>,
    pub size_cols: Option<usize>,
    pub gflops: Option<f64>,
    pub peak_ram_mb: Option<u64>,
}

impl MatrixMetadata {
    pub fn empty() -> Self {
        MatrixMetadata {
            algorithm: None, timestamp: None, cpu: None, simd: None,
            threads: None, compute_ms: None, size_rows: None, size_cols: None,
            gflops: None, peak_ram_mb: None,
        }
    }
}

/// Write matrix data to CSV with optional `# key=value` metadata header.
pub fn save_matrix_csv_with_metadata(
    path: &str,
    matrix: &crate::matrix::Matrix,
    meta: Option<&MatrixMetadata>,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    if let Some(m) = meta {
        writeln!(file, "# FLUST_MATRIX_V1")?;
        if let Some(ref v) = m.algorithm   { writeln!(file, "# algorithm={v}")?; }
        if let Some(ref v) = m.timestamp   { writeln!(file, "# timestamp={v}")?; }
        if let Some(ref v) = m.cpu         { writeln!(file, "# cpu={v}")?; }
        if let Some(ref v) = m.simd        { writeln!(file, "# simd={v}")?; }
        if let Some(v)     = m.threads     { writeln!(file, "# threads={v}")?; }
        if let Some(v)     = m.compute_ms  { writeln!(file, "# compute_ms={v:.4}")?; }
        if let Some(v)     = m.size_rows   { writeln!(file, "# size_rows={v}")?; }
        if let Some(v)     = m.size_cols   { writeln!(file, "# size_cols={v}")?; }
        if let Some(v)     = m.gflops      { writeln!(file, "# gflops={v:.4}")?; }
        if let Some(v)     = m.peak_ram_mb { writeln!(file, "# peak_ram_mb={v}")?; }
    }
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

/// Save a matrix to CSV file (plain, no metadata). Kept for backwards compatibility.
pub fn save_matrix_csv(path: &str, matrix: &crate::matrix::Matrix) -> std::io::Result<()> {
    save_matrix_csv_with_metadata(path, matrix, None)
}

/// Load a matrix from CSV, also parsing any `# key=value` metadata header.
/// Returns (matrix, metadata_if_present).
pub fn load_matrix_csv_with_metadata(
    path: &str,
) -> std::io::Result<(crate::matrix::Matrix, Option<MatrixMetadata>)> {
    let content = std::fs::read_to_string(path)?;
    let mut meta = MatrixMetadata::empty();
    let mut has_meta = false;
    let mut rows = 0usize;
    let mut cols = 0usize;
    let mut data = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            has_meta = true;
            // Parse "# key=value" lines
            if let Some(kv) = trimmed.strip_prefix("# ") {
                if let Some((k, v)) = kv.split_once('=') {
                    match k.trim() {
                        "algorithm"   => meta.algorithm   = Some(v.trim().to_string()),
                        "timestamp"   => meta.timestamp   = Some(v.trim().to_string()),
                        "cpu"         => meta.cpu         = Some(v.trim().to_string()),
                        "simd"        => meta.simd        = Some(v.trim().to_string()),
                        "threads"     => meta.threads     = v.trim().parse().ok(),
                        "compute_ms"  => meta.compute_ms  = v.trim().parse().ok(),
                        "size_rows"   => meta.size_rows   = v.trim().parse().ok(),
                        "size_cols"   => meta.size_cols   = v.trim().parse().ok(),
                        "gflops"      => meta.gflops      = v.trim().parse().ok(),
                        "peak_ram_mb" => meta.peak_ram_mb = v.trim().parse().ok(),
                        _ => {}
                    }
                }
            }
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

    let matrix = crate::matrix::Matrix::from_flat(rows, cols, data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    Ok((matrix, if has_meta { Some(meta) } else { None }))
}

/// Load a matrix from CSV file (plain or with metadata header — metadata discarded).
pub fn load_matrix_csv(path: &str) -> std::io::Result<crate::matrix::Matrix> {
    load_matrix_csv_with_metadata(path).map(|(m, _)| m)
}

// ─── Persistent Computation History ──────────────────────────────────────────
//
// Appends to `flust_history.csv` after every computation (multiply or compare).
// Loaded at app startup to pre-populate the session history.
// Also read by the Performance Monitor (separate process) for [C] overlay.

const HISTORY_CSV: &str = "flust_history.csv";
const HISTORY_HEADER: &str =
    "unique_id,timestamp,algorithm,size,compute_ms,gflops,simd,threads,peak_ram_mb";

/// A slim history record (distinct from CsvRecord which has more benchmark fields).
#[derive(Clone)]
pub struct HistoryRecord {
    pub unique_id: String,
    pub timestamp: String,
    pub algorithm: String,
    pub size: usize,
    pub compute_ms: f64,
    pub gflops: f64,
    pub simd: String,
    pub threads: usize,
    pub peak_ram_mb: u64,
}

/// Append one history record to `flust_history.csv`. Creates file + header if missing.
/// Called fire-and-forget from session history push — errors are silently ignored.
pub fn append_history(record: &HistoryRecord) -> std::io::Result<()> {
    let file_exists = Path::new(HISTORY_CSV).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(HISTORY_CSV)?;
    if !file_exists {
        writeln!(file, "{}", HISTORY_HEADER)?;
    }
    // Escape algorithm name in case it contains commas (wrap in quotes)
    let algo_escaped = format!("\"{}\"", record.algorithm.replace('"', "\"\""));
    writeln!(
        file,
        "{},{},{},{},{:.4},{:.4},{},{},{}",
        record.unique_id,
        record.timestamp,
        algo_escaped,
        record.size,
        record.compute_ms,
        record.gflops,
        record.simd,
        record.threads,
        record.peak_ram_mb,
    )?;
    Ok(())
}

/// Load all history records from `flust_history.csv` (most recent last).
pub fn load_history() -> std::io::Result<Vec<HistoryRecord>> {
    if !Path::new(HISTORY_CSV).exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(HISTORY_CSV)?;
    let mut records = Vec::new();

    for line in content.lines().skip(1) {
        // Simple CSV parse: handle quoted algorithm field
        let fields = parse_history_csv_line(line);
        if fields.len() < 9 {
            continue;
        }
        let record = HistoryRecord {
            unique_id:  fields[0].trim().to_string(),
            timestamp:  fields[1].trim().to_string(),
            algorithm:  fields[2].trim().to_string(),
            size:       fields[3].trim().parse().unwrap_or(0),
            compute_ms: fields[4].trim().parse().unwrap_or(0.0),
            gflops:     fields[5].trim().parse().unwrap_or(0.0),
            simd:       fields[6].trim().to_string(),
            threads:    fields[7].trim().parse().unwrap_or(0),
            peak_ram_mb: fields[8].trim().parse().unwrap_or(0),
        };
        if record.size > 0 {
            records.push(record);
        }
    }
    Ok(records)
}

/// Minimal CSV line parser that handles one quoted field (algorithm name).
fn parse_history_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_quotes => { in_quotes = true; }
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            _ => { current.push(ch); }
        }
    }
    fields.push(current);
    fields
}

/// Build a unique ID for a history record from timestamp + algo + size.
pub fn make_history_id(algo: &str, size: usize) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let algo_slug: String = algo.chars()
        .filter(|c| c.is_alphanumeric())
        .take(8)
        .collect();
    format!("{ts}-{algo_slug}-{size}")
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

// ─── Startup Progress ───────────────────────────────────────────────────────

/// Print a startup progress step to stderr (not captured by TUI).
/// Example: `[██████░░░░]  2/3  Detecting SIMD capabilities...`
pub fn print_startup_step(step: usize, total: usize, label: &str) {
    let pct = (step as f32 / total as f32) * 100.0;
    let bar = make_bar(pct, 20);
    eprint!("\r  [{bar}]  {step}/{total}  {label}");
    let _ = std::io::stderr().flush();
}

/// Clear the startup progress line.
pub fn finish_startup() {
    eprint!("\r{}\r", " ".repeat(80));
    let _ = std::io::stderr().flush();
}

// ─── Sound ─────────────────────────────────────────────────────────────────

/// Play a terminal bell (BEL character) to signal computation completion.
pub fn play_completion_sound() {
    eprint!("\x07");
    let _ = std::io::stderr().flush();
}

// ─── Bar ────────────────────────────────────────────────────────────────────

/// Create a text-based progress bar: █████░░░░░
pub fn make_bar(pct: f32, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f32).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}
