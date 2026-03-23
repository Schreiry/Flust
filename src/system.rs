// system.rs — CPU detection, SIMD capability probing, system information.


use crate::common::{MemoryProfile, SimdLevel};
use sysinfo::System;

// ─── SystemInfo ─────────────────────────────────────────────────────────────

/// Complete snapshot of the host system's capabilities, taken once at startup.
pub struct SystemInfo {
    pub hostname: String,
    pub cpu_brand: String,
    pub cpu_arch: String,           // "Raptor Lake", "Zen 4", etc. (best-effort)
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub base_frequency_mhz: u64,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub simd_level: SimdLevel,      // best available
    pub supports_sse42: bool,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub l1_cache_kb: Option<u32>,
    pub l2_cache_kb: Option<u32>,
    pub l3_cache_kb: Option<u32>,
    pub base_freq_ghz: f64,
    pub peak_estimate: crate::common::PeakEstimate,
    pub memory_profile: MemoryProfile,
}

impl SystemInfo {
    /// Probe the system and return a complete info snapshot.
    /// Call once at startup — sysinfo refresh is not free.
    pub fn detect() -> Self {
        // Targeted refresh: only CPU info + memory. Avoids loading the full
        // process table (~500 MB on a typical Windows machine with 300+ processes).
        let mut sys = System::new();
        sys.refresh_cpu();       // populates cpus() for brand/frequency
        sys.refresh_memory();    // populates total/available memory

        let hostname = System::host_name().unwrap_or_else(|| "Unknown".into());

        // CPU info from first CPU entry
        let cpus = sys.cpus();
        let (cpu_brand, base_freq) = if let Some(cpu) = cpus.first() {
            (cpu.brand().to_string(), cpu.frequency())
        } else {
            ("Unknown CPU".to_string(), 0)
        };

        let physical_cores = sys.physical_core_count().unwrap_or(0);
        let logical_cores = cpus.len();
        let total_ram_mb = sys.total_memory() / (1024 * 1024);
        let available_ram_mb = sys.available_memory() / (1024 * 1024);

        // SIMD detection via Rust's built-in macro (uses CPUID on x86)
        let supports_sse42 = detect_sse42();
        let supports_avx2 = detect_avx2();
        let supports_avx512 = detect_avx512();

        let simd_level = if supports_avx512 {
            SimdLevel::Avx512
        } else if supports_avx2 {
            SimdLevel::Avx2
        } else if supports_sse42 {
            SimdLevel::Sse42
        } else {
            SimdLevel::Scalar
        };

        // CPU microarchitecture — best-effort heuristic from brand string
        let cpu_arch = guess_microarchitecture(&cpu_brand);

        // Cache sizes — try to extract from OS/CPUID
        let (l1, l2, l3) = detect_cache_sizes();

        // Accurate frequency detection (registry > sysinfo > fallback)
        let (base_freq_ghz, freq_source) = detect_frequency_ghz(base_freq);

        // FMA port count heuristic from microarchitecture
        let (fma_ports, fma_source) = guess_fma_ports(&cpu_arch);

        // Theoretical peak FP64 GFLOPS
        let fp64_per_cycle = simd_level.fp64_ops_per_cycle_per_fma();
        let peak_gflops = physical_cores as f64
            * base_freq_ghz
            * fma_ports as f64
            * fp64_per_cycle;

        let peak_estimate = crate::common::PeakEstimate {
            cores: physical_cores,
            freq_ghz: base_freq_ghz,
            fma_ports,
            fp64_per_cycle_per_fma: fp64_per_cycle,
            peak_gflops,
            freq_source,
            fma_source,
        };

        SystemInfo {
            hostname,
            cpu_brand,
            cpu_arch,
            physical_cores,
            logical_cores,
            base_frequency_mhz: base_freq,
            total_ram_mb,
            available_ram_mb,
            simd_level,
            supports_sse42,
            supports_avx2,
            supports_avx512,
            l1_cache_kb: l1,
            l2_cache_kb: l2,
            l3_cache_kb: l3,
            base_freq_ghz,
            peak_estimate,
            memory_profile: MemoryProfile::detect(available_ram_mb),
        }
    }

    /// Estimated peak RAM for multiplying two NxN matrices (3 matrices in memory).
    pub fn estimate_peak_ram_mb(n: usize) -> u64 {
        // 3 matrices (A, B, C) × N² elements × 8 bytes per f64
        (3 * n * n * 8 / (1024 * 1024)) as u64
    }
}

// ─── SIMD Detection ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn detect_sse42() -> bool {
    is_x86_feature_detected!("sse4.2")
}

#[cfg(target_arch = "x86_64")]
fn detect_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
fn detect_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_sse42() -> bool { false }

#[cfg(not(target_arch = "x86_64"))]
fn detect_avx2() -> bool { false }

#[cfg(not(target_arch = "x86_64"))]
fn detect_avx512() -> bool { false }

// ─── Microarchitecture guess ────────────────────────────────────────────────

/// Best-effort guess of CPU microarchitecture from brand string.
/// This is cosmetic — used for display only, not for code path selection.
fn guess_microarchitecture(brand: &str) -> String {
    let brand_lower = brand.to_lowercase();

    // Intel — extract generationeration from Core i-series model number
    if brand_lower.contains("intel") {
        if brand_lower.contains("14") && brand_lower.contains("generation") {
            return "Meteor Lake".into();
        }
        // Check for Core i model numbers: i7-13700K → 13th generation = Raptor Lake
        if let Some(generation) = extract_intel_generation(&brand_lower) {
            return match generation {
                14.. => "Meteor Lake",
                13 => "Raptor Lake",
                12 => "Alder Lake",
                11 => "Rocket Lake / Tiger Lake",
                10 => "Comet Lake / Ice Lake",
                8 | 9 => "Coffee Lake",
                7 => "Kaby Lake",
                6 => "Skylake",
                _ => "Intel (older)",
            }
            .into();
        }
        return "Intel (unknown generation)".into();
    }

    // AMD
    if brand_lower.contains("amd") {
        if brand_lower.contains("9x") || brand_lower.contains("9 9") {
            return "Zen 5 (Granite Ridge)".into();
        }
        if brand_lower.contains("7x") || brand_lower.contains("7 7") {
            return "Zen 4 (Raphael)".into();
        }
        if brand_lower.contains("5x") || brand_lower.contains("5 5") {
            return "Zen 3 (Vermeer)".into();
        }
        if brand_lower.contains("3x") || brand_lower.contains("3 3") {
            return "Zen 2 (Matisse)".into();
        }
        return "AMD Ryzen".into();
    }

    "Unknown architecture".into()
}

/// Try to extract Intel generationeration number from brand string.
/// e.g. "i7-13700K" → 13, "i5-12400" → 12
fn extract_intel_generation(brand: &str) -> Option<u32> {
    // Look for patterns like "i7-NNNNN" or "i5-NNNN"
    for marker in ["i9-", "i7-", "i5-", "i3-"] {
        if let Some(pos) = brand.find(marker) {
            let after = &brand[pos + marker.len()..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 4 {
                // First 2 digits = generationeration for 5-digit models (13700 → 13)
                // First 1 digit for 4-digit models (9900 → 9)
                if digits.len() >= 5 {
                    return digits[..2].parse().ok();
                } else {
                    return digits[..1].parse().ok();
                }
            }
        }
    }
    None
}

// ─── Cache detection ────────────────────────────────────────────────────────

/// Attempt to detect L1/L2/L3 cache sizes.
/// Uses Windows WMI via command line as a best-effort approach.
/// Returns (L1_KB, L2_KB, L3_KB) — None if detection fails.
fn detect_cache_sizes() -> (Option<u32>, Option<u32>, Option<u32>) {
    #[cfg(target_os = "windows")]
    {
        detect_cache_windows()
    }
    #[cfg(not(target_os = "windows"))]
    {
        (None, None, None)
    }
}

#[cfg(target_os = "windows")]
fn detect_cache_windows() -> (Option<u32>, Option<u32>, Option<u32>) {
    // Query WMI for cache info: Level 3=L1, 4=L2, 5=L3 in Win32_CacheMemory
    // Simpler approach: use GetLogicalProcessorInformation via std::process::Command
    use std::process::Command;

    let mut l1 = None;
    let mut l2 = None;
    let mut l3 = None;

    // L2 cache
    if let Ok(output) = Command::new("wmic")
        .args(["cpu", "get", "L2CacheSize", "/value"])
        .output()
    {
        if let Ok(text) = String::from_utf8(output.stdout) {
            for line in text.lines() {
                if let Some(val) = line.strip_prefix("L2CacheSize=") {
                    l2 = val.trim().parse().ok();
                }
            }
        }
    }

    // L3 cache
    if let Ok(output) = Command::new("wmic")
        .args(["cpu", "get", "L3CacheSize", "/value"])
        .output()
    {
        if let Ok(text) = String::from_utf8(output.stdout) {
            for line in text.lines() {
                if let Some(val) = line.strip_prefix("L3CacheSize=") {
                    l3 = val.trim().parse().ok();
                }
            }
        }
    }

    // L1 — not directly in wmic cpu, estimate from core count
    // Typical: 32KB per core for data cache
    // We'll leave it None if we can't get it
    if l1.is_none() {
        // Common default: 32KB L1d + 32KB L1i = 64KB per core, but report just L1d
        // This is just a display hint, not used for computation
        l1 = Some(32); // per-core L1d estimate
    }

    (l1, l2, l3)
}

// ─── Frequency detection ─────────────────────────────────────────────────────
//
// sysinfo often reports incorrect or zero frequency values.
// On Windows we try the registry first (~MHz key), which is authoritative.
// Fallback chain: registry → sysinfo → conservative 3.0 GHz default.

fn detect_frequency_ghz(sysinfo_mhz: u64) -> (f64, &'static str) {
    #[cfg(target_os = "windows")]
    {
        if let Some(mhz) = detect_freq_registry() {
            if mhz > 100 && mhz < 10000 {
                return (mhz as f64 / 1000.0, "registry");
            }
        }
    }

    if sysinfo_mhz > 100 {
        (sysinfo_mhz as f64 / 1000.0, "sysinfo")
    } else {
        (3.0, "fallback")
    }
}

#[cfg(target_os = "windows")]
fn detect_freq_registry() -> Option<u64> {
    use std::process::Command;
    let output = Command::new("powershell")
        .args([
            "-NoProfile", "-Command",
            "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0').'~MHz'",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.trim().parse::<u64>().ok()
}

// ─── FMA port heuristic ─────────────────────────────────────────────────────
//
// Modern Intel (Skylake+) and AMD (Zen 3+) CPUs have 2 FMA ports per core,
// doubling theoretical FP throughput. Older CPUs have 1 FMA port.
// This is a cosmetic heuristic — affects efficiency % display, not computation.

fn guess_fma_ports(arch: &str) -> (u8, &'static str) {
    let arch_lower = arch.to_lowercase();

    // Intel Skylake and later: 2 × 256-bit FMA units per core
    if arch_lower.contains("skylake")
        || arch_lower.contains("coffee")
        || arch_lower.contains("kaby")
        || arch_lower.contains("alder")
        || arch_lower.contains("raptor")
        || arch_lower.contains("meteor")
        || arch_lower.contains("rocket")
        || arch_lower.contains("tiger")
        || arch_lower.contains("comet")
        || arch_lower.contains("ice")
    {
        return (2, "heuristic");
    }

    // AMD Zen 3+ has 2 × 256-bit FMA units per core
    if arch_lower.contains("zen 3")
        || arch_lower.contains("zen 4")
        || arch_lower.contains("zen 5")
        || arch_lower.contains("vermeer")
        || arch_lower.contains("raphael")
        || arch_lower.contains("granite")
    {
        return (2, "heuristic");
    }

    // Conservative default: 1 FMA port
    (1, "default")
}

// ─── CPU Temperature ────────────────────────────────────────────────────────

/// Query LibreHardwareMonitor (LHM) or OpenHardwareMonitor (OHM) WMI provider
/// for real CPU die temperature. These tools install a WMI namespace that exposes
/// actual hardware sensor values (Tdie/Tctl), unlike ACPI which reads thermal zones.
/// Returns None if neither LHM nor OHM is running.
#[cfg(target_os = "windows")]
fn get_temp_from_hardware_monitor_wmi() -> Option<f64> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    // Single PowerShell invocation: tries LHM namespace first, then OHM.
    // Get-CimInstance returns nothing (not an error) if the namespace doesn't exist.
    let script = "\
        $t = $null; \
        foreach ($ns in 'root/LibreHardwareMonitor','root/OpenHardwareMonitor') { \
            $s = Get-CimInstance -Namespace $ns -ClassName Sensor -ErrorAction SilentlyContinue \
                | Where-Object {$_.SensorType -eq 'Temperature'}; \
            if ($s) { $t = ($s | Measure-Object Value -Maximum).Maximum; break } \
        } \
        $t";

    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;

    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let temp: f64 = trimmed.parse().ok()?;
    if temp > 0.0 && temp < 125.0 { Some(temp) } else { None }
}

/// Attempt to read CPU temperature in °C.
/// Priority: LHM/OHM WMI (real die sensor) → MSAcpi WMI (unreliable, needs admin).
/// Returns None if all sources fail or the platform is unsupported.
#[cfg(target_os = "windows")]
pub fn get_cpu_temperature() -> Option<f64> {
    // Priority 1: LHM/OHM — real Tdie/Tctl, no admin needed, requires LHM/OHM running
    if let Some(t) = get_temp_from_hardware_monitor_wmi() {
        return Some(t);
    }

    // Priority 2: MSAcpi thermal zone — often needs admin, often wrong sensor
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi -ErrorAction SilentlyContinue | Select-Object -First 1).CurrentTemperature",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;

    let text = String::from_utf8(output.stdout).ok()?;
    let tenths_kelvin: f64 = text.trim().parse().ok()?;
    if tenths_kelvin < 2000.0 || tenths_kelvin > 5000.0 {
        return None;
    }
    let celsius = tenths_kelvin / 10.0 - 273.15;
    // MSAcpi values below 35°C on a running system are almost certainly
    // the ambient ACPI thermal zone, not CPU die temperature. Returning None
    // lets the monitor show the honest "–°C [no sensor]" hint instead of
    // a misleading static value like 28°C.
    if celsius < 35.0 {
        return None;
    }
    Some(celsius)
}

#[cfg(not(target_os = "windows"))]
pub fn get_cpu_temperature() -> Option<f64> {
    None
}

/// Query current CPU clock speed in MHz via WMI Win32_Processor.CurrentClockSpeed.
/// Unlike registry-based detection, this reflects actual boost frequency.
/// No admin rights required. Returns None if the query fails.
#[cfg(target_os = "windows")]
pub fn get_current_freq_mhz() -> Option<u64> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty CurrentClockSpeed",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;

    let text = String::from_utf8(output.stdout).ok()?;
    // Multi-socket systems return multiple lines — take the maximum
    text.lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .max()
}

#[cfg(not(target_os = "windows"))]
pub fn get_current_freq_mhz() -> Option<u64> {
    None
}

// ─── Auto-tuning ─────────────────────────────────────────────────────────────

/// Benchmark tile sizes [16, 24, 32, 48, 64, 96, 128] on 256×256 matrices.
/// Each size: 3 runs, take minimum. Returns optimal tile size.
/// Uses single-threaded multiply_tiled to avoid thread overhead skewing results.
pub fn auto_tune_tile_size() -> usize {
    let test_dim = 256usize;
    let sizes = [16usize, 24, 32, 48, 64, 96, 128];
    let a = crate::matrix::Matrix::random(test_dim, test_dim, Some(42)).unwrap();
    let b = crate::matrix::Matrix::random(test_dim, test_dim, Some(43)).unwrap();

    let mut best_time = f64::MAX;
    let mut best_size = 64usize;

    for &size in &sizes {
        let mut min_time = f64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let _ = crate::algorithms::multiply_tiled(&a, &b, size);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            min_time = min_time.min(elapsed);
        }
        if min_time < best_time {
            best_time = min_time;
            best_size = size;
        }
    }
    best_size
}

// ─── Chapter 20: Enhanced auto-tuner with L1/L2 tile config ─────────────────

/// Configuration for two-level cache-aware tiling.
pub struct TileConfig {
    pub l1_tile: usize,   // for L1-resident micro-kernel
    pub l2_tile: usize,   // for L2-resident outer tiling
    pub optimal: usize,   // backward compat = best single-level tile
}

/// Benchmark two-level tile sizes. Returns optimal L1 and L2 tile dimensions.
/// L1 candidates sized to fit 3 tiles in P-core L1 (48KB).
/// L2 candidates sized to fit 3 tiles in P-core L2 (2MB).
pub fn auto_tune_tile_sizes() -> TileConfig {
    let test_dim = 512usize;
    let a = crate::matrix::Matrix::random(test_dim, test_dim, Some(42)).unwrap();
    let b = crate::matrix::Matrix::random(test_dim, test_dim, Some(43)).unwrap();

    // Warmup
    let _ = crate::algorithms::multiply_tiled(&a, &b, 64);

    // L1 candidates: 3 × T² × 8 bytes must fit in L1 (48KB P-core, 32KB E-core)
    let l1_candidates = [16usize, 24, 32, 40, 48];
    let mut best_l1 = 32usize;
    let mut best_l1_time = f64::MAX;

    for &t in &l1_candidates {
        let mut min_time = f64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let _ = crate::algorithms::multiply_tiled(&a, &b, t);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            min_time = min_time.min(elapsed);
        }
        if min_time < best_l1_time {
            best_l1_time = min_time;
            best_l1 = t;
        }
    }

    // L2 candidates: test two-level tiling with best L1
    let l2_candidates = [128usize, 192, 256, 320];
    let mut best_l2 = 256usize;
    let mut best_l2_time = f64::MAX;

    for &t in &l2_candidates {
        let mut min_time = f64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let _ = crate::algorithms::multiply_tiled_l2l1(&a, &b, t, best_l1);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            min_time = min_time.min(elapsed);
        }
        if min_time < best_l2_time {
            best_l2_time = min_time;
            best_l2 = t;
        }
    }

    TileConfig {
        l1_tile: best_l1,
        l2_tile: best_l2,
        optimal: best_l2,
    }
}

// ─── Chapter 20: Rayon pool configuration ───────────────────────────────────

/// Initialize rayon global thread pool with the given number of threads.
/// Call ONCE in main() before any rayon operations.
/// If num_threads == 0, uses physical core count (no HyperThreading).
/// HT hurts pure FP64 compute — competing for the same FMA ports.
pub fn init_rayon_pool(num_threads: usize) {
    let threads = if num_threads > 0 {
        num_threads
    } else {
        // Default: physical cores. For pure compute, HT adds contention.
        let sys = System::new_all();
        let physical = sys.physical_core_count().unwrap_or(4);
        physical
    };

    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        eprintln!("Rayon pool init ({} threads): {}", threads, e);
    }
}

/// Get the rayon pool thread count from environment or default to 0 (auto).
pub fn rayon_threads_from_env() -> usize {
    std::env::var("FLUST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

// ─── Memory estimation ───────────────────────────────────────────────────────

/// Estimate peak RAM for Strassen on padded matrix size.
/// Strassen holds up to ~18 temp matrices at peak recursion.
pub fn estimate_strassen_memory_mb(n_padded: usize) -> u64 {
    (18 * n_padded * n_padded * 8 / (1024 * 1024)) as u64
}

/// Estimate RAM for naive/tiled multiplication (3 matrices: A, B, C).
pub fn estimate_naive_memory_mb(m: usize, n: usize, p: usize) -> u64 {
    ((m * n + n * p + m * p) * 8 / (1024 * 1024)) as u64
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_info_detects() {
        let info = SystemInfo::detect();
        assert!(info.logical_cores > 0);
        assert!(info.total_ram_mb > 0);
        assert!(!info.cpu_brand.is_empty());
    }

    #[test]
    fn test_auto_tune_returns_valid_size() {
        let size = auto_tune_tile_size();
        assert!(size >= 16 && size <= 128, "Tuned size {} out of range", size);
    }

    #[test]
    fn test_memory_estimation() {
        let mb = estimate_strassen_memory_mb(1024);
        assert!(mb > 0, "Strassen memory estimate should be > 0");
        let mb2 = estimate_naive_memory_mb(1024, 1024, 1024);
        assert!(mb2 > 0, "Naive memory estimate should be > 0");
    }
}
