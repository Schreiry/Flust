mod common;
mod matrix;
mod algorithms;
mod simd_core;
mod system;
mod monitor;
mod io;
mod interactive;

fn main() {
    // CLI routing: --monitor flag launches the performance monitor TUI
    // in a dedicated console window (spawned by the main interactive process).
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--monitor") {
        monitor::run_monitor();
    } else {
        interactive::run_interactive_mode();
    }
}
