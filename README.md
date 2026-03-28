<div align="center">

```
  ███████╗██╗     ██╗   ██╗███████╗████████╗
  ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝
  █████╗  ██║     ██║   ██║███████╗   ██║   
  ██╔══╝  ██║     ██║   ██║╚════██║   ██║   
  ██║     ███████╗╚██████╔╝███████║   ██║   
  ╚═╝     ╚══════╝ ╚═════╝ ╚══════╝   ╚═╝  
```

**High-Performance Matrix Engine · Rust Edition**

*A spiritual successor to Fluminum — reborn in Rust with zero compromises.*

[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange?logo=rust)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)]()
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX--512%20%7C%20SSE4.2-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

</div>

---

## What is Flust?

Flust is a terminal-based high-performance matrix computation engine, written in Rust. It began as a direct port of **Fluminum** — a C++ HPC engine built around the Strassen algorithm, AVX2 SIMD intrinsics, and a hand-rolled thread pool. But Flust is not just a port. It is a complete rethinking of what that engine can become when idiomatic Rust replaces manual memory management, `rayon` replaces a custom thread pool, and `ratatui` replaces a Windows-only PDH console monitor.

Where Fluminum was a Windows-first performance experiment, Flust aims to be a universal engineering computation platform — capable of tackling matrix multiplication, thermal field simulation, pathfinding via Markov Decision Processes, and eventually sparse solvers, eigensystems, and beyond. All inside a terminal. All with observable, real-time performance metrics.

---

## Feature Overview

| Category | Features |
|---|---|
| **Algorithms** | Naïve O(n³), Tiled (cache-blocked), Tiled-Parallel, Strassen O(n^2.807), Winograd, SIMD dispatch |
| **SIMD** | Runtime detection — AVX-512 / AVX2+FMA / SSE4.2 / Scalar fallback |
| **Parallelism** | `rayon` work-stealing, outer-parallel Strassen, adaptive chunking, 24-core support |
| **Engineering Modules** | 3D Thermal Simulation (FDM), TEG power output via Seebeck effect |
| **TUI** | Full `ratatui` interface — per-core CPU bars, sparkline history, memory gauge, benchmark tables |
| **Analytics** | GFLOPS measurement, timing breakdown, VTune-informed optimization |
| **I/O** | CSV export, matrix file I/O (Fluminum-compatible format), clipboard copy |
| **CLI** | `clap`-based — batch mode (`multiply`, `compare`, `bench`, `info`) or interactive TUI |

---

## Architecture

Flust follows a **flat monolithic module structure** — a deliberate mirror of the Fluminum C++ layout. No microservices, no nested crate hierarchies. Each file is a complete domain.

```
Cargo.toml            ← dependencies + release optimization profiles
.cargo/config.toml    ← rustflags: target-cpu=native, AVX2, FMA, BMI2
src/
├── main.rs           ← entry point, CLI routing, rayon thread-pool init
├── common.rs         ← types, enums, constants, macros, result structs
├── matrix.rs         ← Matrix struct, operators, split/combine/pad
├── algorithms.rs     ← Strassen, Tiled, Winograd, compare
├── simd_core.rs      ← unsafe AVX2/AVX-512/SSE4.2 kernels (isolated)
├── system.rs         ← CPU detection, SystemInfo, auto-tuner
├── monitor.rs        ← PerformanceMonitor, ratatui TUI, per-core metrics
├── io.rs             ← ANSI UI, CSV logging, file I/O, progress bars
├── interactive.rs    ← menus, AppState, benchmark suite, system info
├── thermal.rs        ← FDM heat solver, Laplacian matrix, TEG physics
├── thermal_ui.rs     ← TUI screens for thermal simulation
├── thermal_export.rs ← CSV export of temperature fields and snapshots
└── sparse.rs         ← CSR sparse matrix (used by thermal Laplacian)
```

### The Ten Laws of Flust Code

These are architectural invariants. Breaking them is regression, not progress.

1. **Flat Vector** — `Matrix` is always a `Vec<f64>` with index `i * cols + j`. `Vec<Vec<f64>>` is banned — it is a cache catastrophe.
2. **Unsafe only in SIMD** — `unsafe` blocks live exclusively in `simd_core.rs`, preceded by a single `assert!` boundary check outside the block.
3. **Rayon, not threads** — All compute parallelism goes through `rayon`. `std::thread` is only for the background performance monitor.
4. **No gratuitous `clone()`** — In hot paths, only slices `&[f64]` and `&mut [f64]`. Clones in Strassen recursion are allowed only where a quadrant is consumed more than once.
5. **i-k-j loop order** — The innermost loop always iterates over the column index of C. The i-j-k order causes guaranteed cache misses on the B matrix.
6. **One cause, one patch** — Root cause analysis before any fix. No "let's try this."
7. **Analysis before code** — Every non-trivial block is preceded by 2–3 lines of architectural reasoning: *why this decision is optimal for the hardware.*
8. **`cargo check` before continuing** — A chapter is not done until zero compiler errors.
9. **Tests are ground truth** — Every optimized algorithm is verified against naïve multiplication with `epsilon = 1e-9`.
10. **No jumping ahead** — Chapters are strictly sequential. Foundation before algorithms, algorithms before SIMD, SIMD before UI.

---

## Algorithms & Mathematics

### Matrix Representation

Every `Matrix` in Flust is a contiguous `Vec<f64>` in **row-major order**:

```
element(i, j) = data[i * cols + j]
```

This layout guarantees that the CPU hardware prefetcher loads entire rows into cache lines before they are needed. AVX2 then consumes 4 doubles per cycle with `_mm256_loadu_pd`. A `Vec<Vec<f64>>` would scatter each row to an arbitrary heap location, producing a cache miss on every row traversal.

---

### Naïve Multiplication

The ground-truth reference implementation. Strictly i-k-j loop order:

```
C[i][j] += A[i][k] × B[k][j]
```

**FLOPS count:** `2 × M × N × P` (one multiply + one add per inner step).  
**GFLOPS formula:** `(2·M·N·P) / (compute_ms × 10⁶)`  
**Complexity:** O(n³).

---

### Cache-Blocked Tiled Multiplication

Divides the matrices into tiles of size `T × T` and processes them in i-k-j block order, ensuring three tiles (`A_tile`, `B_tile`, `C_tile`) fit simultaneously in the L1/L2 cache.

**Tile size selection:** `auto_tune_tile_size()` benchmarks `T ∈ {16, 24, 32, 48, 64, 96, 128}` on 256×256 matrices (3 runs, take minimum) and selects the fastest. This is done once at startup and cached in `AppState`.

**Cache constraint (example — i9-13900K P-core L1 = 48 KB):**
```
3 × T² × 8 bytes ≤ 48 × 1024  →  T ≤ 45  →  optimal T_L1 = 32 or 40
```

The parallel variant (`multiply_tiled_parallel`) decomposes the outer i-block loop via `rayon::par_chunks_mut`, giving each worker thread an independent horizontal slice of C — no memory contention, no locks.

---

### Strassen's Algorithm

Based on **Volker Strassen's 1969 paper** *"Gaussian Elimination is not Optimal"* (Numerische Mathematik, 13, 354–356). Instead of 8 recursive submatrix multiplications, Strassen uses only 7, reducing asymptotic complexity from O(n³) to **O(n^2.807)**.

The 7 intermediate products (for submatrices A₁₁, A₁₂, A₂₁, A₂₂ of A and B₁₁, B₁₂, B₂₁, B₂₂ of B):

```
M₁ = (A₁₁ + A₂₂) × (B₁₁ + B₂₂)
M₂ = (A₂₁ + A₂₂) × B₁₁
M₃ = A₁₁ × (B₁₂ − B₂₂)
M₄ = A₂₂ × (B₂₁ − B₁₁)
M₅ = (A₁₁ + A₁₂) × B₂₂
M₆ = (A₂₁ − A₁₁) × (B₁₁ + B₁₂)
M₇ = (A₁₂ − A₂₂) × (B₂₁ + B₂₂)
```

The result quadrants are then assembled:
```
C₁₁ = M₁ + M₄ − M₅ + M₇
C₁₂ = M₃ + M₅
C₂₁ = M₂ + M₄
C₂₂ = M₁ − M₂ + M₃ + M₆
```

**Padding:** Non-power-of-2 matrices are zero-padded before recursion and unpaddedafter. The original dimensions are preserved.  
**Threshold:** Recursion bottoms out at matrices smaller than `STRASSEN_THRESHOLD` (tuned to 32–64 depending on parallelism analysis — see Chapter 20 discussion below).  
**Parallelism:** `rayon::join` decomposes recursive branches onto worker threads. To saturate 24 cores from the very first recursive level, Flust also implements **outer-parallel Strassen** — pre-decomposing the matrix into K horizontal strips before entering recursion, where K ≈ number of physical cores.

---

### Winograd Variant

A variant of Strassen with fewer additions in the assembly phase. Implemented in Chapter 12. Shares the same 7-product structure but reduces some intermediate additions by precomputing row/column factor sums.

---

### AVX2/FMA SIMD Kernel

The hot inner loop in `simd_core.rs`. For each pair (i, k), broadcasts `A[i][k]` into a 256-bit AVX2 register and processes 4 doubles of row k in B simultaneously via fused multiply-add:

```rust
let a_ik = _mm256_set1_pd(*a.get_unchecked(i * n + k));
// Inner j-loop, step 4:
let b_chunk = _mm256_loadu_pd(b.as_ptr().add(k * p + j));
let c_chunk = _mm256_loadu_pd(c.as_ptr().add(i * p + j) as *const f64);
let result  = _mm256_fmadd_pd(a_ik, b_chunk, c_chunk);   // FMA: a*b + c in one instruction
_mm256_storeu_pd(c.as_mut_ptr().add(i * p + j), result);
```

Scalar remainder handles columns when `P mod 4 ≠ 0`. The dispatch function (`multiply_dispatch`) selects AVX-512, AVX2, SSE4.2, or scalar at **runtime** via `is_x86_feature_detected!` — the binary remains portable across machines.

---

## Thermal Simulation Module

The `Thermal Simulation` module implements a **3D Finite Difference Method (FDM)** solver for the Fourier heat conduction equation, with real-time visualization in the terminal.

### Physics

**Governing Equation** (3D Fourier heat conduction):
```
∂T/∂t = α · (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
```
where `α = λ / (ρ · c)` is thermal diffusivity [m²/s].

**Discretization** (explicit Euler scheme on an Nₓ×N_y×N_z grid):
```
T_new[n] = T_old[n] + dt · α · L · T_old[n]
```
where `L` is the discrete Laplacian (stored as a CSR sparse matrix). In matrix form:
```
T_new = A · T_old,   A = I + dt·α·L
```

**Stability criterion (Courant condition) — enforced at startup:**
```
dt ≤ 1 / (2·α · (1/hₓ² + 1/h_y² + 1/h_z²))
```
Violating this makes temperature diverge to ±∞. Flust detects and rejects unstable configurations before running.

**Boundary conditions:** Dirichlet — grid boundary nodes are pinned to `T_boundary` (ambient temperature). In matrix `A`, the corresponding row is the identity row.

### TEG Power Output (Seebeck Effect)

For thermoelectric generators (TEG, e.g. Bi₂Te₃ modules):

```
V = S · ΔT                             (Seebeck voltage)
I = V / (R_internal + R_load)          (current)
P = I² · R_load                        (power output)
η = P / Q_hot                          (efficiency)
Q_hot = λ_teg · A_teg · ΔT / d_teg   (heat flux through TEG)
```

where `S ≈ 0.05 V/K` for Bi₂Te₃. Maximum power transfer occurs at `R_load = R_internal` (impedance matching).

### Application: Alexander's Tank Project

The thermal module was directly designed to support a real engineering project — a robot tank that tracks heat sources via an infrared sensor. The preset configuration `config_tank_project_default()` models a 150×80×60 mm fluid reservoir at 85°C cooled to 20°C ambient, with a TEG on one wall generating power for drive motors. The simulation predicts motor runtime before thermal equilibrium drops voltage below 1V.

---

## Performance: VTune Diagnosis

Intel VTune Amplifier profiling on an **i9-13900K** (Raptor Lake-DT, 24 cores / 24 threads, HT disabled, AVX2):

| Metric | Current | Target |
|---|---|---|
| Logical Core Utilization | **63.7%** (2.3 of 24 cores) | 80%+ |
| Back-End Bound | **58.4%** | < 20% |
| Memory Bound | **38.0%** | < 15% |
| L3 Bound | **14.8%** | < 5% |
| DRAM Bound | **11.2%** | < 5% |
| IPC | **0.696** | 2.5–4.0 |
| DP GFLOPS | **1.498** | 20+ |
| AVX2 (256-bit packed) | 71.9% | — |

**Root cause:** Strassen at the top recursion level creates only 7 tasks — 17 of 24 cores idle until level 2. Tiles (64×64) don't fully fit in P-core L1 (48 KB), causing L3 thrash. The system allocator (HeapAlloc on Windows) serializes large allocations under lock.

**Optimizations implemented (Ch. 20/21):**
- Strassen threshold lowered: 64 → 32 (more recursive levels = more parallel tasks from the start)
- **Outer-parallel Strassen**: pre-decompose matrix into K strips before recursion, where K = `rayon::current_num_threads()`, guaranteeing all cores are loaded from step zero
- **Two-level cache-aware tiling**: outer tile T_L2 = 256 (fits P-core L2 @ 2 MB), inner tile T_L1 = 32 (fits P-core L1 @ 48 KB)
- **`mimalloc` global allocator**: replaces HeapAlloc, eliminates the global lock on large allocations, 20–40% faster for Strassen's intermediate matrix lifecycle
- **32-byte aligned allocation**: enables `_mm256_load_pd` (aligned) vs `_mm256_loadu_pd` (unaligned), removes potential scalar fallback in the AVX2 kernel
- **Rayon ThreadPoolBuilder**: configured with `num_threads = physical_cores` and `stack_size = 4 MB` — avoids hyperthreading overhead on AVX2 workloads where both logical cores share the AVX execution unit

---

## Building

**Requirements:** Rust 1.75+ (stable), x86_64 CPU with AVX2 (most Intel/AMD CPUs since 2013).

```bash
# Clone and build in release mode
git clone https://github.com/yourname/flust
cd flust
cargo build --release

# Run interactive TUI
cargo run --release

# Or launch the binary directly (recommended — sets proper terminal size)
./target/release/flust
```

For Windows, use the provided `run_flust.bat` launcher which sets the terminal to 120×40 before starting.

**Verify SIMD level:**
```bash
cargo run --release -- info
```

---

## CLI Usage

Flust supports both an interactive TUI mode and a batch command-line mode:

```bash
# Multiply two matrices (output to file)
flust multiply --size 2048 --algorithm strassen --output result.csv

# Compare two matrix files
flust compare matrix_a.csv matrix_b.csv --epsilon 1e-9

# Run benchmark suite
flust bench --size 1024

# Print system information (CPU, SIMD, RAM, optimal tile)
flust info
```

In interactive mode, navigate with `↑↓ Enter` and press `[H]` for help, `[A]` for About, `[Q]` or `Esc` to go back.

---

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `rayon` | 1.10 | Work-stealing parallel computation |
| `rand` | 0.8 | Matrix random generation (SmallRng) |
| `sysinfo` | 0.30 | Cross-platform CPU/RAM monitoring |
| `ratatui` | 0.26 | Terminal UI framework |
| `crossterm` | 0.27 | Terminal backend, alternate screen |
| `clap` | 4 | CLI argument parsing (derive) |
| `serde` + `csv` | 1 + 1.3 | Structured data export |
| `anyhow` | 1 | Unified error handling |
| `mimalloc` | 0.1 | High-performance global allocator |

---

## Mathematical Background

| Concept | Reference |
|---|---|
| Strassen's algorithm | V. Strassen, *"Gaussian Elimination is not Optimal"*, Numerische Mathematik 13 (1969), 354–356 |
| Winograd variant | S. Winograd, *"On Multiplication of 2×2 Matrices"*, Linear Algebra and its Applications 4 (1971) |
| FMA SIMD (AVX2) | Intel Intrinsics Guide — `_mm256_fmadd_pd` |
| Fourier heat equation | J.B.J. Fourier, *"Théorie analytique de la chaleur"* (1822) |
| FDM explicit Euler | R.D. Richtmyer & K.W. Morton, *"Difference Methods for Initial-Value Problems"* (1967) |
| Seebeck effect | T.J. Seebeck, *"Magnetische Polarisation der Metalle"* (1821) |
| Markov Decision Processes | R.A. Howard, *"Dynamic Programming and Markov Processes"* (1960) |
| Cache-oblivious tiling | E. Frigo et al., *"Cache-Oblivious Algorithms"*, FOCS (1999) |
| Memory layout & SIMD | A. Fog, *"Optimizing software in C++"*, Technical Univ. of Denmark |

---

## License

MIT License. See `LICENSE` for details.

---

<div align="center">

*Flust is dedicated to the pursuit of clean, honest, hardware-aware code.*  
*Every microsecond saved is a proof that understanding the machine matters.*

</div>
