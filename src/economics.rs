// ═══════════════════════════════════════════════════════════════════════════════
//  LEONTIEF INPUT–OUTPUT SHOCK SIMULATOR
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Mathematical basis:
//    Total output x satisfies:  x = (I − A)⁻¹ d
//    where A = technology matrix, d = final demand vector.
//
//  We approximate via Neumann series (dynamic shock waves):
//    x₀ = d
//    xₖ = A · xₖ₋₁ + d
//    Converges when spectral radius ρ(A) < 1.
//
//  Each iteration k represents the k-th cascade wave of economic shock
//  propagating through supply chains.
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

/// Configuration for a Leontief shock simulation.
#[derive(Clone)]
pub struct LeontiefConfig {
    /// Number of economic sectors (N×N technology matrix).
    pub sectors: usize,
    /// Fraction of zero entries in the technology matrix [0, 1).
    pub sparsity: f64,
    /// Target spectral radius for generated matrix (must be < 1).
    pub spectral_target: f64,
    /// Convergence tolerance: stop when Δ = ‖xₖ − xₖ₋₁‖∞ < tol.
    pub tolerance: f64,
    /// Maximum iterations (safety bound).
    pub max_iterations: usize,
    /// Index of the sector receiving the demand shock.
    pub shock_sector: usize,
    /// Magnitude of the demand shock (monetary units).
    pub shock_magnitude: f64,
}

impl LeontiefConfig {
    /// Estimate RAM usage in MB for the technology matrix + working vectors.
    pub fn estimate_memory_mb(&self) -> f64 {
        let n = self.sectors;
        // A matrix (n×n) + 3 vectors (x_old, x_new, demand) of length n
        let bytes = (n * n + 3 * n) * std::mem::size_of::<f64>();
        bytes as f64 / (1024.0 * 1024.0)
    }
}

/// A single iteration snapshot for convergence tracking.
#[derive(Clone)]
pub struct IterationSnapshot {
    /// Iteration index k.
    pub iteration: usize,
    /// L∞ convergence delta ‖xₖ − xₖ₋₁‖∞.
    pub delta: f64,
    /// L2 norm of the current output vector.
    pub l2_norm: f64,
    /// Sum of all sector outputs (total economic activity).
    pub total_output: f64,
    /// Top affected sector index at this iteration.
    pub top_sector: usize,
    /// Output value of the top affected sector.
    pub top_sector_value: f64,
}

/// Final result of a Leontief shock simulation.
#[derive(Clone)]
pub struct LeontiefResult {
    pub config: LeontiefConfig,
    /// Final output vector x (length = sectors).
    pub output: Vec<f64>,
    /// Demand vector d (length = sectors).
    pub demand: Vec<f64>,
    /// Per-sector loss (output − demand), showing cascade amplification.
    pub sector_losses: Vec<f64>,
    /// Sector names (generated).
    pub sector_names: Vec<String>,
    /// Convergence history — one snapshot per iteration.
    pub snapshots: Vec<IterationSnapshot>,
    /// Number of iterations to convergence.
    pub iterations: usize,
    /// Final convergence delta.
    pub final_delta: f64,
    /// Whether the series converged within tolerance.
    pub converged: bool,
    /// Computation wall-clock time in milliseconds.
    pub computation_ms: f64,
    /// Total matrix–vector multiplications performed.
    pub total_matvec_ops: u64,
    /// Estimated spectral radius of the generated matrix.
    pub spectral_radius_est: f64,
    /// Total economic multiplier: sum(x) / sum(d).
    pub multiplier: f64,
}

// ─── Technology Matrix Generation ───────────────────────────────────────────
//
// Generate a random N×N technology matrix A with:
//   - Controlled sparsity (fraction of zeros)
//   - Column sums < 1 (productive economy — ensures ρ(A) < 1)
//   - Row values representing inter-sector consumption coefficients

/// Generate a technology matrix with controlled spectral properties.
///
/// Returns flat Vec<f64> of size n×n in row-major order.
/// Column sums are normalized to `spectral_target` to guarantee convergence.
pub fn generate_technology_matrix(
    n: usize,
    sparsity: f64,
    spectral_target: f64,
    seed: u64,
) -> Vec<f64> {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let mut rng = SmallRng::seed_from_u64(seed);
    let sparsity = sparsity.clamp(0.0, 0.99);
    let target = spectral_target.clamp(0.01, 0.99);

    // Phase 1: Generate raw entries with sparsity mask
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal = 0 (sector doesn't consume its own output as input)
                continue;
            }
            if rng.r#gen::<f64>() < sparsity {
                continue; // sparse zero
            }
            a[i * n + j] = rng.r#gen::<f64>();
        }
    }

    // Phase 2: Normalize column sums to `target`
    // This ensures the column-sum norm ‖A‖₁ ≈ target, hence ρ(A) < 1.
    for j in 0..n {
        let col_sum: f64 = (0..n).map(|i| a[i * n + j]).sum();
        if col_sum > 1e-12 {
            let scale = target / col_sum;
            for i in 0..n {
                a[i * n + j] *= scale;
            }
        }
    }

    a
}

/// Generate a demand shock vector: all zeros except `shock_sector`.
pub fn generate_demand_vector(
    n: usize,
    shock_sector: usize,
    shock_magnitude: f64,
) -> Vec<f64> {
    let mut d = vec![0.0_f64; n];
    if shock_sector < n {
        d[shock_sector] = shock_magnitude;
    }
    d
}

// ─── HPC Matrix–Vector Multiply ─────────────────────────────────────────────
//
// y = A · x + d
// where A is n×n dense (row-major), x and d are length-n vectors.
// Fully parallel via rayon — each row is an independent dot product.

/// Parallel matrix–vector multiply: y = A · x + d
///
/// A: flat n×n row-major, x: length n, d: length n.
/// Returns y of length n.
#[inline]
pub fn matvec_add_parallel(a: &[f64], x: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    // Parallel over rows — each row computes one dot product
    (0..n)
        .into_par_iter()
        .map(|i| {
            let row_start = i * n;
            let row = &a[row_start..row_start + n];
            // Dot product: sum(A[i,j] * x[j]) + d[i]
            let dot: f64 = row.iter().zip(x.iter()).map(|(&a_ij, &x_j)| a_ij * x_j).sum();
            dot + d[i]
        })
        .collect()
}

/// Parallel matrix–vector multiply without addition: y = A · x
#[inline]
pub fn matvec_parallel(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let row_start = i * n;
            let row = &a[row_start..row_start + n];
            row.iter().zip(x.iter()).map(|(&a_ij, &x_j)| a_ij * x_j).sum()
        })
        .collect()
}

/// L∞ norm of (a − b): max absolute difference.
fn linf_delta(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .reduce(|| 0.0_f64, f64::max)
}

/// L2 norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.par_iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─── Sector Name Generator ──────────────────────────────────────────────────

pub const SECTOR_NAMES: &[&str] = &[
    "Energy", "Agriculture", "Mining", "Manufacturing", "Construction",
    "Transport", "Telecom", "Finance", "Healthcare", "Education",
    "Retail", "Wholesale", "Real Estate", "Technology", "Defense",
    "Chemicals", "Textiles", "Automotive", "Aerospace", "Pharma",
    "Food Processing", "Metals", "Electronics", "Tourism", "Media",
    "Logistics", "Insurance", "Banking", "Utilities", "Government",
    "Water Supply", "Forestry", "Fishing", "Paper", "Plastics",
    "Rubber", "Glass", "Cement", "Steel", "Aluminum",
    "Copper", "Software", "Hardware", "Biotech", "Nanotech",
    "Robotics", "AI Services", "Cloud Infra", "Semiconductors", "Optics",
];

fn generate_sector_names(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            if i < SECTOR_NAMES.len() {
                SECTOR_NAMES[i].to_string()
            } else {
                format!("Sector-{}", i)
            }
        })
        .collect()
}

// ─── Main Simulation Loop ───────────────────────────────────────────────────

/// Run the Leontief shock simulation.
///
/// `progress` is an atomic counter for UI feedback: high 32 bits = total, low 32 = done.
/// `phase` is a shared string describing the current computation phase.
pub fn run_leontief_simulation(
    config: &LeontiefConfig,
    progress: &Arc<AtomicU64>,
    phase: &Arc<std::sync::Mutex<String>>,
) -> anyhow::Result<LeontiefResult> {
    let n = config.sectors;
    if n < 2 {
        anyhow::bail!("Need at least 2 sectors");
    }
    if config.shock_sector >= n {
        anyhow::bail!("Shock sector {} out of range (0..{})", config.shock_sector, n);
    }

    let start = std::time::Instant::now();

    // ─── Phase 1: Generate technology matrix ────────────────────────────
    {
        let mut ph = phase.lock().unwrap();
        *ph = format!("Generating {}×{} technology matrix...", n, n);
    }
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    let a = generate_technology_matrix(n, config.sparsity, config.spectral_target, seed);

    // Estimate spectral radius (column-sum norm as upper bound)
    let spectral_est: f64 = (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    // ─── Phase 2: Generate demand vector ────────────────────────────────
    let d = generate_demand_vector(n, config.shock_sector, config.shock_magnitude);

    // ─── Phase 3: Neumann iteration ─────────────────────────────────────
    {
        let mut ph = phase.lock().unwrap();
        *ph = "Running Neumann series iteration...".to_string();
    }
    // Set progress total
    let total = config.max_iterations as u32;
    progress.store((total as u64) << 32, Ordering::Relaxed);

    let mut x = d.clone(); // x₀ = d
    let mut snapshots = Vec::with_capacity(config.max_iterations.min(10000));
    let mut converged = false;
    let mut final_delta = f64::MAX;
    let mut iterations = 0;
    let mut total_ops: u64 = 0;

    // Initial snapshot
    snapshots.push(IterationSnapshot {
        iteration: 0,
        delta: config.shock_magnitude,
        l2_norm: l2_norm(&x),
        total_output: x.iter().sum(),
        top_sector: config.shock_sector,
        top_sector_value: x[config.shock_sector],
    });

    for k in 1..=config.max_iterations {
        // xₖ = A · xₖ₋₁ + d
        let x_new = matvec_add_parallel(&a, &x, &d, n);
        total_ops += 1;

        let delta = linf_delta(&x_new, &x);
        let norm = l2_norm(&x_new);
        let total_out: f64 = x_new.iter().sum();

        // Find top sector
        let (top_idx, top_val) = x_new
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0));

        // Save snapshot (every iteration for small runs, sampled for large)
        let save = k <= 100
            || k % 10 == 0
            || k % (config.max_iterations / 200).max(1) == 0
            || delta < config.tolerance;
        if save {
            snapshots.push(IterationSnapshot {
                iteration: k,
                delta,
                l2_norm: norm,
                total_output: total_out,
                top_sector: top_idx,
                top_sector_value: top_val,
            });
        }

        // Update progress
        let done = k as u32;
        progress.store(((total as u64) << 32) | done as u64, Ordering::Relaxed);

        // Update phase text periodically
        if k % 50 == 0 || delta < config.tolerance {
            let mut ph = phase.lock().unwrap();
            *ph = format!(
                "Iteration {}/{} — Δ = {:.2e}  (tol = {:.2e})",
                k, config.max_iterations, delta, config.tolerance
            );
        }

        x = x_new;
        final_delta = delta;
        iterations = k;

        if delta < config.tolerance {
            converged = true;
            break;
        }
    }

    // ─── Phase 4: Compute derived metrics ───────────────────────────────
    {
        let mut ph = phase.lock().unwrap();
        *ph = "Computing analytics...".to_string();
    }

    // Sector losses: output amplification beyond direct demand
    let sector_losses: Vec<f64> = x.iter().zip(d.iter()).map(|(&xi, &di)| xi - di).collect();
    let sector_names = generate_sector_names(n);
    let demand_sum: f64 = d.iter().sum();
    let output_sum: f64 = x.iter().sum();
    let multiplier = if demand_sum.abs() > 1e-15 {
        output_sum / demand_sum
    } else {
        1.0
    };

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Mark 100% complete
    progress.store(((total as u64) << 32) | total as u64, Ordering::Relaxed);

    Ok(LeontiefResult {
        config: config.clone(),
        output: x,
        demand: d,
        sector_losses,
        sector_names,
        snapshots,
        iterations,
        final_delta,
        converged,
        computation_ms: elapsed,
        total_matvec_ops: total_ops,
        spectral_radius_est: spectral_est,
        multiplier,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_parallel_identity() {
        // I · x = x
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = matvec_parallel(&a, &x, n);
        for i in 0..n {
            assert!((y[i] - x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_matvec_add() {
        let n = 3;
        // A = [[0.1, 0.2, 0], [0, 0.1, 0.3], [0.2, 0, 0.1]]
        let a = vec![0.1, 0.2, 0.0, 0.0, 0.1, 0.3, 0.2, 0.0, 0.1];
        let x = vec![10.0, 20.0, 30.0];
        let d = vec![1.0, 2.0, 3.0];
        let y = matvec_add_parallel(&a, &x, &d, n);
        // y[0] = 0.1*10 + 0.2*20 + 0*30 + 1 = 1+4+1 = 6.0
        assert!((y[0] - 6.0).abs() < 1e-10);
        // y[1] = 0*10 + 0.1*20 + 0.3*30 + 2 = 0+2+9+2 = 13.0
        assert!((y[1] - 13.0).abs() < 1e-10);
        // y[2] = 0.2*10 + 0*20 + 0.1*30 + 3 = 2+0+3+3 = 8.0
        assert!((y[2] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_technology_matrix_column_sums() {
        let n = 50;
        let target = 0.8;
        let a = generate_technology_matrix(n, 0.3, target, 12345);
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| a[i * n + j]).sum();
            // Column sums should be approximately `target`
            // (may be 0 if entire column was sparse)
            assert!(col_sum <= target + 1e-10 || col_sum < 1e-12,
                "Column {} sum = {} exceeds target {}", j, col_sum, target);
        }
    }

    #[test]
    fn test_leontief_convergence() {
        let config = LeontiefConfig {
            sectors: 20,
            sparsity: 0.5,
            spectral_target: 0.7,
            tolerance: 1e-8,
            max_iterations: 5000,
            shock_sector: 0,
            shock_magnitude: 100.0,
        };
        let progress = Arc::new(AtomicU64::new(0));
        let phase = Arc::new(std::sync::Mutex::new(String::new()));
        let result = run_leontief_simulation(&config, &progress, &phase).unwrap();
        assert!(result.converged, "Should converge with ρ(A) < 1");
        assert!(result.final_delta < config.tolerance);
        assert!(result.multiplier >= 1.0, "Multiplier should be >= 1");
        assert!(!result.snapshots.is_empty());
    }

    #[test]
    fn test_demand_vector() {
        let d = generate_demand_vector(10, 3, 500.0);
        assert_eq!(d.len(), 10);
        assert!((d[3] - 500.0).abs() < 1e-15);
        assert!((d[0]).abs() < 1e-15);
    }

    #[test]
    fn test_linf_delta() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 1.5, 3.7, 4.0];
        let d = linf_delta(&a, &b);
        assert!((d - 0.7).abs() < 1e-10);
    }
}
