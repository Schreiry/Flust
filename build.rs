// ─── Build Script: Intel MKL Discovery ─────────────────────────────────────
//
// When compiled with `--features mkl`, locates the system MKL installation
// and emits `cargo:rustc-link-lib` / `cargo:rustc-link-search` directives
// so that all `extern "C"` blocks resolve against mkl_rt (runtime dispatcher).
//
// mkl_rt.dll is a single dynamic library that replaces the trio
// (mkl_intel_lp64 + mkl_sequential + mkl_core). Using mkl_rt avoids
// the "component library" linking that caused double-init crashes
// when mixed with the intel-mkl-src crate.
//
// Search order:
//   1. MKLROOT environment variable (Intel's standard convention)
//   2. Common oneAPI install paths on Windows
//   3. Fallback: let the system linker try PATH

fn main() {
    // MKL is now loaded at runtime via libloading — no compile-time linking needed.
    // This build script is kept for future use (e.g., custom native dependencies).
    println!("cargo:rerun-if-env-changed=MKLROOT");
}
