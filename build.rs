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
    // Only relevant when the `mkl` feature is enabled.
    if std::env::var("CARGO_FEATURE_MKL").is_err() {
        return;
    }

    // Intel MKL library search paths (64-bit).
    let candidate_dirs: Vec<String> = {
        let mut dirs = Vec::new();

        // 1. MKLROOT (set by Intel oneAPI setvars.bat / setvars.sh)
        if let Ok(root) = std::env::var("MKLROOT") {
            dirs.push(format!("{root}/lib"));           // Linux / unified
            dirs.push(format!("{root}/lib/intel64"));    // Windows oneAPI layout
        }

        // 2. Common Windows oneAPI install locations
        if cfg!(target_os = "windows") {
            let program_x86 = std::env::var("ProgramFiles(x86)")
                .unwrap_or_else(|_| r"C:\Program Files (x86)".into());
            let program = std::env::var("ProgramFiles")
                .unwrap_or_else(|_| r"C:\Program Files".into());

            // oneAPI default: "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib"
            dirs.push(format!(r"{program_x86}\Intel\oneAPI\mkl\latest\lib"));
            dirs.push(format!(r"{program_x86}\Intel\oneAPI\mkl\latest\lib\intel64"));
            dirs.push(format!(r"{program}\Intel\oneAPI\mkl\latest\lib"));
            dirs.push(format!(r"{program}\Intel\oneAPI\mkl\latest\lib\intel64"));
        }

        dirs
    };

    // Emit link-search for every directory that actually exists.
    let mut found = false;
    for dir in &candidate_dirs {
        let path = std::path::Path::new(dir);
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={dir}");
            found = true;
        }
    }

    if !found {
        println!(
            "cargo:warning=MKL library directory not found. \
             Set MKLROOT or ensure Intel oneAPI MKL is installed. \
             Falling back to system linker PATH."
        );
    }

    // Link against mkl_rt — the single runtime dispatcher DLL.
    // On Windows this resolves to mkl_rt.lib (import library) → mkl_rt.2.dll at runtime.
    println!("cargo:rustc-link-lib=dylib=mkl_rt");

    // Re-run this script if MKLROOT changes.
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MKL");
}
