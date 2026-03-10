mod common;
mod matrix;
mod algorithms;
mod simd_core;
mod system;
mod monitor;
mod io;
mod interactive;

fn main() {
    interactive::run_interactive_mode();
}
