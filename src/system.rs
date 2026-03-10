// system.rs — CPU detection, SIMD capability probing, system information.
//
// Provides a SystemInfo snapshot taken at startup: CPU brand, hostname,
// core/thread counts, SIMD support levels (green/red in UI), and cache sizes.
// Uses sysinfo crate for cross-platform basics and is_x86_feature_detected!
// for precise SIMD capability detection at runtime.

use crate::common::SimdLevel;
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
}

impl SystemInfo {
    /// Probe the system and return a complete info snapshot.
    /// Call once at startup — sysinfo refresh is not free.
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

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
