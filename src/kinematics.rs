// ═══════════════════════════════════════════════════════════════════════════════
//  KINEMATICS & PATHFINDING — MARKOV DECISION PROCESSES
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Mathematical basis:
//    Given a stochastic transition matrix P (rows sum to 1),
//    the steady-state distribution π satisfies:  π = π · P
//
//  We find π by repeated squaring of P until convergence:
//    P¹ → P² → P⁴ → P⁸ → ...
//    Convergence: ‖P^(2k) − P^(2(k−1))‖_F < ε
//
//  At convergence, every row of P^∞ equals the steady-state distribution π.
//  This models a tank navigating through states toward an IR heat source,
//  where P encodes transition probabilities between discrete map cells.
//
//  Uses Flust HPC multiply_hpc_fused for each squaring step.
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use crate::algorithms::{self, ConvergenceSnapshot};
use crate::matrix::Matrix;

/// Configuration for an MDP simulation.
#[derive(Clone)]
pub struct MdpConfig {
    /// State space dimension (N×N transition matrix).
    pub matrix_size: usize,
    /// Fraction of zero entries in P [0, 1).
    pub sparsity: f64,
    /// Convergence threshold ε for Frobenius norm.
    pub epsilon: f64,
    /// Maximum squaring iterations.
    pub max_iterations: usize,
    /// Random seed for reproducibility (None = entropy).
    pub seed: Option<u64>,
}

impl MdpConfig {
    /// Estimate RAM usage in MB for the transition matrix + working copies.
    pub fn estimate_memory_mb(&self) -> f64 {
        let n = self.matrix_size;
        // P matrix + P_prev copy + result = 3 × N×N
        let bytes = 3 * n * n * std::mem::size_of::<f64>();
        bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Snapshot of MDP convergence at each squaring step.
#[derive(Clone)]
pub struct MdpSnapshot {
    /// Squaring iteration index (1, 2, 3, ...).
    pub iteration: usize,
    /// Current exponent: P raised to this power.
    pub exponent: u64,
    /// Frobenius norm of difference ‖P^n − P^(n/2)‖_F.
    pub norm_diff: f64,
    /// Maximum entry-wise change.
    pub max_entry_change: f64,
    /// Shannon entropy of steady-state estimate (first row of P^n).
    pub steady_state_entropy: f64,
}

/// Final result of an MDP simulation.
#[derive(Clone)]
pub struct MdpResult {
    pub config: MdpConfig,
    /// Converged steady-state distribution π (length = matrix_size).
    pub steady_state: Vec<f64>,
    /// Convergence history — one snapshot per squaring step.
    pub snapshots: Vec<MdpSnapshot>,
    /// Whether the iteration converged within ε.
    pub converged: bool,
    /// Number of squaring iterations performed.
    pub iterations: usize,
    /// Final Frobenius norm difference.
    pub final_norm_diff: f64,
    /// Computation wall-clock time in milliseconds.
    pub computation_ms: f64,
    /// Total matrix multiplications performed (one per squaring).
    pub total_multiplications: u64,
    /// Estimated spectral gap: 1 − |λ₂| (larger = faster convergence).
    pub spectral_gap: f64,
    /// Peak state index in steady-state distribution.
    pub peak_state: usize,
    /// Peak state probability.
    pub peak_probability: f64,
}

// ─── Stochastic Matrix Generation ────────────────────────────────────────────

/// Generate a random N×N row-stochastic transition matrix.
/// Each row sums to 1.0. `sparsity` fraction of entries are zero.
pub fn generate_stochastic_matrix(n: usize, sparsity: f64, seed: Option<u64>) -> Matrix {
    let mut rng_state: u64 = seed.unwrap_or(42);
    let mut data = vec![0.0f64; n * n];

    let sparsity_thresh = (sparsity * u32::MAX as f64) as u32;

    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            // Simple LCG PRNG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rand_bits = (rng_state >> 33) as u32;

            if rand_bits < sparsity_thresh && i != j {
                data[i * n + j] = 0.0;
            } else {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((rng_state >> 33) as f64) / (u32::MAX as f64) + 0.001;
                data[i * n + j] = val;
                row_sum += val;
            }
        }

        // Normalize row to sum to 1.0 (stochastic constraint)
        if row_sum > 0.0 {
            for j in 0..n {
                data[i * n + j] /= row_sum;
            }
        } else {
            // Degenerate: self-loop
            data[i * n + i] = 1.0;
        }
    }

    Matrix::from_flat(n, n, data).expect("MDP matrix dimensions valid")
}

// ─── Shannon Entropy ─────────────────────────────────────────────────────────

/// Compute Shannon entropy of a probability distribution (bits).
fn shannon_entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in probs {
        if p > 1e-15 {
            h -= p * p.ln();
        }
    }
    h
}

// ─── Main Simulation ─────────────────────────────────────────────────────────

/// Run MDP steady-state computation via repeated matrix squaring.
pub fn run_mdp_simulation(
    config: &MdpConfig,
    progress: &Arc<AtomicU64>,
    phase: &Arc<Mutex<String>>,
) -> anyhow::Result<MdpResult> {
    let n = config.matrix_size;

    // Phase 1: Generate stochastic matrix
    {
        let mut ph = phase.lock().unwrap();
        *ph = format!("Generating {}×{} stochastic matrix...", n, n);
    }

    let start = std::time::Instant::now();
    let p = generate_stochastic_matrix(n, config.sparsity, config.seed);

    // Phase 2: Repeated squaring via HPC multiply
    {
        let mut ph = phase.lock().unwrap();
        *ph = "Matrix power iteration (repeated squaring)...".to_string();
    }

    let (converged_mat, algo_snaps, did_converge) =
        algorithms::matrix_power_converge(&p, config.epsilon, config.max_iterations, progress);

    let computation_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Extract steady-state distribution from first row of converged P^n
    let c_stride = converged_mat.stride();
    let steady_state: Vec<f64> = (0..n)
        .map(|j| converged_mat.data()[j]) // first row
        .collect();

    // Build MDP snapshots from algorithm convergence snapshots
    let snapshots: Vec<MdpSnapshot> = algo_snaps
        .iter()
        .map(|s| {
            // Estimate steady-state entropy from first row at each step
            // (we only have the final matrix, so use final entropy for all)
            MdpSnapshot {
                iteration: s.iteration,
                exponent: s.exponent,
                norm_diff: s.frobenius_diff,
                max_entry_change: s.max_entry_change,
                steady_state_entropy: 0.0, // filled below for final
            }
        })
        .collect();

    // Compute final entropy
    let final_entropy = shannon_entropy(&steady_state);
    let mut snapshots = snapshots;
    if let Some(last) = snapshots.last_mut() {
        last.steady_state_entropy = final_entropy;
    }

    // Find peak state
    let (peak_state, peak_probability) = steady_state
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    // Estimate spectral gap from convergence rate
    let spectral_gap = if snapshots.len() >= 2 {
        let s0 = &snapshots[0];
        let s1 = &snapshots[snapshots.len() - 1];
        if s0.norm_diff > 1e-15 && s1.norm_diff > 1e-15 {
            let ratio = (s1.norm_diff / s0.norm_diff).ln() / (snapshots.len() as f64);
            1.0 - ratio.exp().abs()
        } else {
            1.0
        }
    } else {
        0.0
    };

    let final_norm_diff = snapshots.last().map(|s| s.norm_diff).unwrap_or(0.0);

    Ok(MdpResult {
        config: config.clone(),
        steady_state,
        snapshots,
        converged: did_converge,
        iterations: algo_snaps.len(),
        final_norm_diff,
        computation_ms,
        total_multiplications: algo_snaps.len() as u64,
        spectral_gap: spectral_gap.max(0.0).min(1.0),
        peak_state,
        peak_probability,
    })
}

// ─── State Names ─────────────────────────────────────────────────────────────

/// Generate state names for display: S0, S1, ..., SN-1.
pub fn state_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("S{}", i)).collect()
}
