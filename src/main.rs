// ─── Global Allocator ────────────────────────────────────────────────────────
//
// MiMalloc replaces the default system allocator (HeapAlloc on Windows).
// Benefits for Flust:
//   - No global lock for large allocations (>512KB) — critical for 16+ threads
//     simultaneously allocating Strassen intermediate matrices.
//   - 20-40% faster alloc/dealloc for large f64 buffers.
//   - Better thread-local caching reduces contention on hybrid P+E cores.
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// import Sapce :
mod common;
mod matrix;
mod algorithms;
mod simd_core;
mod system;
mod monitor;
mod io;
mod interactive;
mod numerics;
mod sparse;
mod thermal;
mod thermal_ui;
mod thermal_export;
mod fluids;
mod selftest;
mod compute_worker;
mod economics;
mod economics_ui;
mod economics_export;
mod charts;
mod kinematics;
mod kinematics_export;
mod kinematics_ui;
mod vision;
mod vision_export;
mod vision_ui;
mod system_profiler;
mod mhps;
mod mhps_export;
mod mhps_ui;

// ─── Main Entry Point ─────────────────────────────────────────────────────────

fn main() {
    // CLI routing: --monitor flag launches the performance monitor TUI
    // in a dedicated console window (spawned by the main interactive process).
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--monitor") {
        monitor::run_monitor();
    } else if args.iter().any(|a| a == "--compute") {
        // Subprocess mode: run an isolated computation and exit.
        compute_worker::run_compute_worker(&args);
    } else if args.iter().any(|a| a == "--mkl-check") {
        // Subprocess probe: try MKL init, exit 0 if OK.
        // If MKL segfaults, this child dies — parent survives.
        algorithms::run_mkl_probe();
    } else {
        // Set console window title
        crossterm::execute!(std::io::stdout(), crossterm::terminal::SetTitle("FLUST \u{2014} Matrix Engine")).ok();

        // Startup with progress feedback
        eprintln!("\n  ╔═══════════════════════════════╗");
        eprintln!("  ║   FLUST — Matrix Engine       ║");
        eprintln!("  ╚═══════════════════════════════╝\n");

        // Initialize rayon thread pool BEFORE any parallel work.
        // Uses FLUST_THREADS env var, or defaults to physical core count.
        let rayon_threads = system::rayon_threads_from_env();
        system::init_rayon_pool(rayon_threads);

        io::print_startup_step(1, 5, "Detecting CPU & SIMD capabilities...");
        let sys_info = system::SystemInfo::detect();

        io::print_startup_step(2, 5, &format!(
            "Found {} ({}, {})",
            sys_info.cpu_brand, sys_info.simd_level.display_name(),
            format!("{}C/{}T", sys_info.physical_cores, sys_info.logical_cores)
        ));

        io::print_startup_step(3, 5, "Self-test: verifying core modules...");
        let report = selftest::run_self_tests(sys_info.simd_level);

        io::print_startup_step(4, 5, &format!(
            "Matrix: {}  |  SIMD: {}  |  MKL: {}",
            report.matrix_ops.label(),
            report.simd_kernels.label(),
            report.mkl_available.label(),
        ));

        io::print_startup_step(5, 5, "Ready. Launching TUI...");
        std::thread::sleep(std::time::Duration::from_millis(300));
        io::finish_startup();

        interactive::run_interactive_mode_with_sysinfo(sys_info);
    }
}
