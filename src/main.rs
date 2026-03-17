mod common;
mod matrix;
mod algorithms;
mod simd_core;
mod system;
mod monitor;
mod io;
mod interactive;
mod numerics;

fn main() {
    // CLI routing: --monitor flag launches the performance monitor TUI
    // in a dedicated console window (spawned by the main interactive process).
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--monitor") {
        monitor::run_monitor();
    } else {
        // Startup with progress feedback
        eprintln!("\n  ╔═══════════════════════════════╗");
        eprintln!("  ║   FLUST — Matrix Engine       ║");
        eprintln!("  ╚═══════════════════════════════╝\n");

        io::print_startup_step(1, 3, "Detecting CPU & SIMD capabilities...");
        let sys_info = system::SystemInfo::detect();

        io::print_startup_step(2, 3, &format!(
            "Found {} ({}, {})",
            sys_info.cpu_brand, sys_info.simd_level.display_name(),
            format!("{}C/{}T", sys_info.physical_cores, sys_info.logical_cores)
        ));

        io::print_startup_step(3, 3, "Ready. Launching TUI...");
        std::thread::sleep(std::time::Duration::from_millis(300));
        io::finish_startup();

        interactive::run_interactive_mode_with_sysinfo(sys_info);
    }
}
