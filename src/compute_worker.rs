// ─── Compute Worker: Subprocess Isolation ───────────────────────────────────
//
// MKL FFI calls can segfault (signal, not Rust panic). catch_unwind doesn't
// catch signals. The only way to survive an MKL crash is process isolation:
// run the computation in a child process, and if it dies, the parent TUI
// shows an error instead of crashing.
//
// Protocol:
//   Parent writes request JSON + binary matrices → spawns child with --compute
//   Child reads request, runs computation, writes result + metrics
//   Parent polls child with try_wait(), reads result on success

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::matrix::Matrix;

// ─── Binary Matrix I/O ─────────────────────────────────────────────────────
//
// Format: [rows: u64][cols: u64][data: rows*cols × f64]
// Much faster than CSV for large matrices.

pub fn save_matrix_binary(path: &Path, matrix: &Matrix) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let rows = matrix.rows() as u64;
    let cols = matrix.cols() as u64;
    f.write_all(&rows.to_le_bytes())?;
    f.write_all(&cols.to_le_bytes())?;

    // Safety: reinterpret &[f64] as &[u8] for bulk write.
    let data = matrix.data();
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8)
    };
    f.write_all(byte_slice)?;
    Ok(())
}

pub fn load_matrix_binary(path: &Path) -> std::io::Result<Matrix> {
    let mut f = std::fs::File::open(path)?;
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf8)?;
    let rows = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8)?;
    let cols = u64::from_le_bytes(buf8) as usize;

    let count = rows.checked_mul(cols)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "overflow"))?;
    let mut data = vec![0.0f64; count];

    // Safety: reinterpret &mut [f64] as &mut [u8] for bulk read.
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, count * 8)
    };
    f.read_exact(byte_slice)?;

    Matrix::from_flat(rows, cols, data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

// ─── Request / Response ─────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ComputeRequest {
    pub algorithm: String,      // "naive", "strassen", "winograd", "mkl"
    pub simd_level: String,     // "Scalar", "SSE4.2", "AVX2", "AVX-512"
    pub matrix_a_path: String,
    pub matrix_b_path: String,
    pub result_path: String,
    pub metrics_path: String,
    pub progress_path: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ComputeMetrics {
    pub compute_ms: f64,
    pub padding_ms: f64,
    pub unpadding_ms: f64,
    pub peak_ram_bytes: u64,
    pub avg_freq_ghz: f64,
    pub error: Option<String>,
}

// ─── Child Process Entry Point ──────────────────────────────────────────────

pub fn run_compute_worker(args: &[String]) {
    // Parse --request-file <path>
    let request_path = args.iter()
        .position(|a| a == "--request-file")
        .and_then(|i| args.get(i + 1))
        .expect("--compute requires --request-file <path>");

    let request_json = std::fs::read_to_string(request_path)
        .expect("Failed to read request file");
    let request: ComputeRequest = serde_json::from_str(&request_json)
        .expect("Failed to parse request JSON");

    let metrics = execute_request(&request);

    // Write metrics (always, even on error)
    let metrics_json = serde_json::to_string(&metrics).unwrap_or_default();
    let _ = std::fs::write(&request.metrics_path, metrics_json);

    if metrics.error.is_some() {
        std::process::exit(1);
    }
}

// ─── MKL Runtime DLL Discovery ───────────────────────────────────────────────
//
// Intel oneAPI MKL on Windows consists of multiple DLLs:
//   mkl_rt.2.dll   — runtime dispatcher (loaded by the linker)
//   mkl_avx2.2.dll — architecture-specific compute kernel (loaded LAZILY by mkl_rt)
//   mkl_core.2.dll, mkl_def.2.dll — dependencies
//   libiomp5md.dll — OpenMP runtime (in compiler dir, NOT mkl dir)
//
// build.rs finds mkl_rt.lib at compile time, but at RUNTIME the kernel DLLs
// must be on PATH. Without running setvars.bat, they often aren't.
// These functions mirror build.rs discovery and add the bin directories to PATH.

/// Collect MKL runtime directories that should be on PATH.
/// Returns directories that exist on disk but are not already in PATH.
fn find_mkl_bin_dirs() -> Vec<String> {
    let current_path = std::env::var("PATH").unwrap_or_default();
    let current_lower = current_path.to_lowercase();
    let mut dirs = Vec::new();

    let mut candidates: Vec<String> = Vec::new();

    // Strategy 1: MKLROOT is set (setvars.bat was run, or user set it manually)
    if let Ok(root) = std::env::var("MKLROOT") {
        candidates.push(format!(r"{root}\bin"));
        candidates.push(format!(r"{root}\redist\intel64"));
        // OpenMP runtime lives in the compiler component, sibling to mkl
        candidates.push(format!(r"{root}\..\..\compiler\latest\bin"));
    }

    for c in &candidates {
        if std::path::Path::new(c).is_dir() && !current_lower.contains(&c.to_lowercase()) {
            dirs.push(c.clone());
        }
    }

    // Strategy 2: Probe common oneAPI install locations
    if dirs.is_empty() {
        let bases = [
            std::env::var("ProgramFiles(x86)")
                .unwrap_or_else(|_| r"C:\Program Files (x86)".into()),
            std::env::var("ProgramFiles")
                .unwrap_or_else(|_| r"C:\Program Files".into()),
        ];
        for base in &bases {
            let mkl_root = format!(r"{base}\Intel\oneAPI\mkl\latest");
            let probes = [
                format!(r"{mkl_root}\bin"),
                format!(r"{mkl_root}\redist\intel64"),
                format!(r"{base}\Intel\oneAPI\compiler\latest\bin"),
            ];
            for p in &probes {
                if std::path::Path::new(p).is_dir() && !current_lower.contains(&p.to_lowercase()) {
                    dirs.push(p.clone());
                }
            }
            if !dirs.is_empty() {
                break;
            }
        }
    }

    dirs
}

/// Add MKL runtime directories to the current process's PATH.
/// Call this inside child processes before any MKL FFI call.
pub fn ensure_mkl_runtime_paths() {
    let dirs = find_mkl_bin_dirs();
    if dirs.is_empty() {
        return;
    }
    let current = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", dirs.join(";"), current);
    eprintln!("[MKL] Augmented PATH with: {:?}", dirs);
    unsafe { std::env::set_var("PATH", &new_path); }
}

/// Return an augmented PATH string without modifying the current process.
/// Use this on the parent side when spawning child processes via Command::env().
pub fn get_mkl_augmented_path() -> Option<String> {
    let dirs = find_mkl_bin_dirs();
    if dirs.is_empty() {
        return None;
    }
    let current = std::env::var("PATH").unwrap_or_default();
    Some(format!("{};{}", dirs.join(";"), current))
}

fn execute_request(req: &ComputeRequest) -> ComputeMetrics {
    // MKL runtime DLL discovery: add bin directories to PATH so that
    // mkl_avx2.2.dll, mkl_core.2.dll, libiomp5md.dll etc. can be found.
    // Without this, cblas_dgemm crashes with ACCESS_VIOLATION (0xC0000005)
    // because the compute kernel DLLs are loaded lazily and not on PATH.
    if req.algorithm == "mkl" {
        ensure_mkl_runtime_paths();
    }

    // MKL threading guard: force single-threaded MKL to avoid OpenMP + rayon
    // conflict that causes ACCESS_VIOLATION (0xC0000005) on Windows.
    // Must be set BEFORE any MKL function is called (including mkl_set_num_threads).
    if req.algorithm == "mkl" {
        unsafe {
            std::env::set_var("MKL_NUM_THREADS", "1");
            std::env::set_var("OMP_NUM_THREADS", "1");
            std::env::set_var("MKL_DYNAMIC", "FALSE");
            std::env::set_var("OMP_STACKSIZE", "64M");
        }
    }

    // Load matrices
    let a = match load_matrix_binary(Path::new(&req.matrix_a_path)) {
        Ok(m) => m,
        Err(e) => return ComputeMetrics::error(format!("Failed to load matrix A: {e}")),
    };
    let b = match load_matrix_binary(Path::new(&req.matrix_b_path)) {
        Ok(m) => m,
        Err(e) => return ComputeMetrics::error(format!("Failed to load matrix B: {e}")),
    };

    // Write initial progress
    let _ = std::fs::write(&req.progress_path, "5 100");

    let simd = crate::common::SimdLevel::from_name(&req.simd_level);

    let ram_before = crate::interactive::sample_rss_bytes();
    let freq_pre = crate::interactive::sample_avg_freq_mhz();

    let start = std::time::Instant::now();
    let result = match req.algorithm.as_str() {
        "naive" => {
            let r = crate::algorithms::multiply_naive(&a, &b);
            (r, 0.0, 0.0)
        }
        "strassen" => {
            crate::algorithms::multiply_strassen_padded(
                &a, &b, crate::common::STRASSEN_THRESHOLD, simd,
            )
        }
        "winograd" => {
            crate::algorithms::multiply_winograd_padded(
                &a, &b, crate::common::STRASSEN_THRESHOLD, simd,
            )
        }
        "mkl" => {
            match crate::algorithms::multiply_mkl(&a, &b) {
                Ok(r) => (r, 0.0, 0.0),
                Err(e) => return ComputeMetrics::error(format!("MKL DGEMM: {e}")),
            }
        }
        other => return ComputeMetrics::error(format!("Unknown algorithm: {other}")),
    };
    let compute_ms = start.elapsed().as_secs_f64() * 1000.0;

    let freq_post = crate::interactive::sample_avg_freq_mhz();
    let ram_after = crate::interactive::sample_rss_bytes();

    // Write progress 100%
    let _ = std::fs::write(&req.progress_path, "100 100");

    // Save result matrix
    if let Err(e) = save_matrix_binary(Path::new(&req.result_path), &result.0) {
        return ComputeMetrics::error(format!("Failed to save result: {e}"));
    }

    ComputeMetrics {
        compute_ms,
        padding_ms: result.1,
        unpadding_ms: result.2,
        peak_ram_bytes: ram_after.saturating_sub(ram_before),
        avg_freq_ghz: if freq_pre > 0.0 && freq_post > 0.0 {
            (freq_pre + freq_post) / 2.0 / 1000.0
        } else {
            0.0
        },
        error: None,
    }
}

impl ComputeMetrics {
    fn error(msg: String) -> Self {
        ComputeMetrics {
            compute_ms: 0.0,
            padding_ms: 0.0,
            unpadding_ms: 0.0,
            peak_ram_bytes: 0,
            avg_freq_ghz: 0.0,
            error: Some(msg),
        }
    }
}

// ─── Parent-Side Helpers ────────────────────────────────────────────────────

/// Generate a unique temp directory for this computation.
pub fn make_compute_temp_dir() -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir().join("flust_compute");
    std::fs::create_dir_all(&base)?;
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let dir = base.join(format!("{id}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Clean up a computation temp directory.
pub fn cleanup_temp_dir(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Clean up stale temp dirs older than 1 hour.
pub fn cleanup_stale_temps() {
    let base = std::env::temp_dir().join("flust_compute");
    if let Ok(entries) = std::fs::read_dir(&base) {
        let cutoff = std::time::SystemTime::now()
            - std::time::Duration::from_secs(3600);
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if let Ok(modified) = meta.modified() {
                    if modified < cutoff {
                        let _ = std::fs::remove_dir_all(entry.path());
                    }
                }
            }
        }
    }
}

/// Read progress from the child's progress file.
/// Returns (done, total) or None if file doesn't exist yet.
pub fn read_child_progress(progress_path: &Path) -> Option<(u32, u32)> {
    let content = std::fs::read_to_string(progress_path).ok()?;
    let mut parts = content.trim().split_whitespace();
    let done: u32 = parts.next()?.parse().ok()?;
    let total: u32 = parts.next()?.parse().ok()?;
    Some((done, total))
}
