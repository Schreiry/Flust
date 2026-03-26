// ─── Thermal Simulation Module ─────────────────────────────────────────────
//
// 3D Finite-Difference Method (FDM) for the heat equation:
//   ∂T/∂t = α · ∇²T
//
// Explicit Euler time-stepping: T_new = A · T_old
// where A = I + dt·α·L (L = discrete Laplacian, 7-point stencil).
//
// Application: cooling dynamics of fluid in a reservoir,
// with thermoelectric generator (TEG) power output via Seebeck effect.
// я надеюсь что все работает верно и вычисления будут полезны для александра.
// Все в надежде того, что я не напрастно потратил время на это....

use std::sync::{Arc, Mutex};
use crate::common::ProgressHandle;
use crate::sparse::CooMatrix;
pub use crate::sparse::CsrMatrix;

// MKL sparse handle is now in sparse.rs (MklSparseHandle).
// All MKL FFI declarations are centralized there to avoid
// double-linking crashes from multiple #[link(name="mkl_rt")] blocks.
#[cfg(feature = "mkl")]
use crate::sparse::MklSparseHandle;

// ─── Fluid Properties ──────────────────────────────────────────────────────

/// Physical properties of common fluids.
///   λ — thermal conductivity [W/(m·K)]
///   ρ — density [kg/m³]
///   c — specific heat capacity [J/(kg·K)]
#[derive(Debug, Clone)]
pub struct FluidProperties {
    pub name: String,
    pub thermal_conductivity: f64,
    pub density: f64,
    pub specific_heat: f64,
}

impl FluidProperties {
    /// Thermal diffusivity α = λ / (ρ · c) [m²/s]
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    pub fn water() -> Self {
        Self {
            name: "Water".into(),
            thermal_conductivity: 0.598,
            density: 998.0,
            specific_heat: 4182.0,
        }
    }

    pub fn oil() -> Self {
        Self {
            name: "Engine Oil".into(),
            thermal_conductivity: 0.145,
            density: 880.0,
            specific_heat: 1900.0,
        }
    }

    pub fn ethylene_glycol() -> Self {
        Self {
            name: "Ethylene Glycol (Antifreeze)".into(),
            thermal_conductivity: 0.400,
            density: 1070.0,
            specific_heat: 3400.0,
        }
    }
}

// ─── TEG (Thermoelectric Generator) Properties ────────────────────────────

/// Thermoelectric generator module (Seebeck effect).
///   V = S · ΔT
///   P = I² · R_load  where  I = V / (R_int + R_load)
#[derive(Debug, Clone)]
pub struct TegProperties {
    pub seebeck_coefficient: f64,       // S [V/K]
    pub internal_resistance: f64,       // R_int [Ω]
    pub load_resistance: f64,           // R_load [Ω]
    pub teg_area: f64,                  // contact area [m²]
    pub teg_thickness: f64,             // thickness [m]
    pub teg_thermal_conductivity: f64,  // λ_teg [W/(m·K)]
}

impl TegProperties {
    /// Standard Bi₂Te₃ module (commonly available).
    pub fn standard_bi2te3() -> Self {
        Self {
            seebeck_coefficient: 0.05,
            internal_resistance: 2.0,
            load_resistance: 2.0, // matched for max power
            teg_area: 0.004,      // 40×100 mm
            teg_thickness: 0.004, // 4 mm
            teg_thermal_conductivity: 1.5,
        }
    }

    pub fn voltage(&self, delta_t: f64) -> f64 {
        self.seebeck_coefficient * delta_t
    }

    pub fn current(&self, delta_t: f64) -> f64 {
        self.voltage(delta_t) / (self.internal_resistance + self.load_resistance)
    }

    pub fn power_output(&self, delta_t: f64) -> f64 {
        let i = self.current(delta_t);
        i * i * self.load_resistance
    }

    /// Heat flux through the TEG [W].
    pub fn heat_flux(&self, delta_t: f64) -> f64 {
        self.teg_thermal_conductivity * self.teg_area * delta_t / self.teg_thickness
    }

    /// TEG conversion efficiency.
    pub fn efficiency(&self, delta_t: f64) -> f64 {
        let q = self.heat_flux(delta_t);
        if q < 1e-10 {
            return 0.0;
        }
        self.power_output(delta_t) / q
    }
}

// ─── Simulation Configuration ──────────────────────────────────────────────

/// Boundary condition type for the simulation domain walls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryType {
    /// Fixed temperature on walls: T = T_boundary (default, most stable).
    Dirichlet,
    /// Insulated walls: dT/dn = 0 (perfect thermos, no heat loss).
    Neumann,
    /// Convective cooling: -λ·dT/dn = h·(T - T_ambient).
    /// h_conv is the convective heat transfer coefficient [W/(m²·K)].
    /// Typical values: 5-25 (natural air), 50-200 (forced air), 500-10000 (water).
    Mixed { h_conv: f64 },
}

impl Default for BoundaryType {
    fn default() -> Self {
        Self::Dirichlet
    }
}

impl BoundaryType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dirichlet => "Dirichlet (fixed T)",
            Self::Neumann => "Neumann (insulated)",
            Self::Mixed { .. } => "Mixed (convective)",
        }
    }
}

/// Which wall the TEG module is attached to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TegWall {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

/// Computation method for the thermal simulation SpMV step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalSolver {
    /// Hand-rolled CSR sparse matrix-vector multiply.
    NativeSparse,
    /// Intel MKL sparse BLAS (mkl_sparse_d_mv). Feature-gated.
    #[cfg(feature = "mkl")]
    IntelMKL,
}

impl ThermalSolver {
    pub fn display_name(&self) -> &'static str {
        match self {
            ThermalSolver::NativeSparse => "Native Sparse",
            #[cfg(feature = "mkl")]
            ThermalSolver::IntelMKL => "Intel MKL Sparse",
        }
    }
}

// ─── Heat Sources ─────────────────────────────────────────────────────────

/// A localized heat source with Gaussian spatial falloff and optional
/// temporal pulsation.  Multiple sources create interference patterns
/// that the diffusion solver resolves naturally — no special coupling needed.
#[derive(Debug, Clone)]
pub struct HeatSource {
    pub x: f64,           // position in meters (absolute, within geometry)
    pub y: f64,
    pub z: f64,
    pub temperature: f64, // peak source temperature [°C]
    pub radius: f64,      // Gaussian falloff radius σ [m]
    pub omega: f64,       // angular frequency [rad/s], 0.0 = static source
}

/// Complete configuration for a thermal simulation run.
#[derive(Debug, Clone)]
pub struct ThermalSimConfig {
    // Reservoir geometry [m]
    pub length_x: f64,
    pub length_y: f64,
    pub length_z: f64,

    // Grid resolution
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    // Initial / boundary conditions [°C]
    pub t_initial: f64,
    pub t_boundary: f64,

    // Materials
    pub fluid: FluidProperties,
    pub teg: TegProperties,

    // Time-stepping
    pub time_step_dt: f64,  // [s], 0.0 = auto-calculate
    pub total_steps: usize,
    pub save_every_n: usize,

    // TEG placement
    pub teg_wall: TegWall,

    // Boundary condition type
    pub boundary_type: BoundaryType,

    // Computation method
    pub solver: ThermalSolver,

    // Localized heat sources (Gaussian falloff, optional pulsation)
    pub heat_sources: Vec<HeatSource>,
}

impl ThermalSimConfig {
    pub fn total_nodes(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn hx(&self) -> f64 {
        self.length_x / (self.nx - 1).max(1) as f64
    }
    pub fn hy(&self) -> f64 {
        self.length_y / (self.ny - 1).max(1) as f64
    }
    pub fn hz(&self) -> f64 {
        self.length_z / (self.nz - 1).max(1) as f64
    }

    /// Maximum stable time step (Courant criterion).
    /// dt ≤ 1 / (2·α · (1/hx² + 1/hy² + 1/hz²))
    pub fn max_stable_dt(&self, alpha: f64) -> f64 {
        let hx = self.hx();
        let hy = self.hy();
        let hz = self.hz();
        1.0 / (2.0 * alpha * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)))
    }

    #[inline]
    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    pub fn is_boundary(&self, i: usize, j: usize, k: usize) -> bool {
        i == 0
            || i == self.nx - 1
            || j == 0
            || j == self.ny - 1
            || k == 0
            || k == self.nz - 1
    }

    /// Check if node is on the TEG wall.
    pub fn is_teg_wall(&self, i: usize, j: usize, k: usize) -> bool {
        match self.teg_wall {
            TegWall::XMin => i == 0,
            TegWall::XMax => i == self.nx - 1,
            TegWall::YMin => j == 0,
            TegWall::YMax => j == self.ny - 1,
            TegWall::ZMin => k == 0,
            TegWall::ZMax => k == self.nz - 1,
        }
    }

    /// Estimated memory for the CSR matrix + temperature vectors [MB].
    pub fn estimate_memory_mb(&self) -> f64 {
        let n = self.total_nodes();
        let nnz_est = n * 7;
        // CSR: row_ptr (n+1)*8 + col_idx nnz*8 + values nnz*8
        let csr_bytes = (n + 1) * 8 + nnz_est * 8 + nnz_est * 8;
        // Two temperature vectors: 2 * n * 8
        let vec_bytes = 2 * n * 8;
        (csr_bytes + vec_bytes) as f64 / (1024.0 * 1024.0)
    }
}

// ─── Snapshot & Result ─────────────────────────────────────────────────────

/// Snapshot of simulation state at a given time step.
#[derive(Debug, Clone)]
pub struct ThermalSnapshot {
    pub time_s: f64,
    pub step: usize,
    pub t_center: f64,
    pub t_teg_hot: f64,
    pub t_teg_cold: f64,
    pub delta_t: f64,
    pub voltage: f64,
    pub current: f64,
    pub power_w: f64,
    pub power_mw: f64,
    pub efficiency_pct: f64,
    pub mean_temp: f64,
    pub max_temp: f64,
    pub min_temp: f64,
    // Deep analytics — computed per snapshot
    pub max_gradient: f64,   // peak |∇T| [°C/m]
    pub entropy_rate: f64,   // normalized Shannon entropy [0..1], 1 = equilibrium
    pub heat_loss_pct: f64,  // (E_initial − E_current) / E_initial × 100
}

impl Default for ThermalSnapshot {
    fn default() -> Self {
        Self {
            time_s: 0.0,
            step: 0,
            t_center: 0.0,
            t_teg_hot: 0.0,
            t_teg_cold: 0.0,
            delta_t: 0.0,
            voltage: 0.0,
            current: 0.0,
            power_w: 0.0,
            power_mw: 0.0,
            efficiency_pct: 0.0,
            mean_temp: 0.0,
            max_temp: 0.0,
            min_temp: 0.0,
            max_gradient: 0.0,
            entropy_rate: 0.0,
            heat_loss_pct: 0.0,
        }
    }
}

/// Complete result of a thermal simulation run.
#[derive(Clone)]
pub struct ThermalSimResult {
    pub config: ThermalSimConfig,
    pub snapshots: Vec<ThermalSnapshot>,
    pub final_field: Vec<f64>,
    pub computation_ms: f64,
    pub total_matrix_multiplications: usize,

    // Derived engineering metrics
    pub runtime_minutes: f64,
    pub threshold_voltage: f64,
    pub max_power_mw: f64,
    pub time_to_max_power_s: f64,
    pub average_power_mw: f64,
    pub total_energy_mj: f64,
}

// ─── Matrix Assembly ───────────────────────────────────────────────────────

/// Build the transition matrix A = I + dt·α·L for explicit Euler.
///
/// For interior node n = idx(i,j,k):
///   A[n,n] = 1 - 2·dt·α·(1/hx² + 1/hy² + 1/hz²)
///   A[n, neighbors] = dt·α/h²  (6 neighbors)
///
/// For boundary node (Dirichlet): A[n,n] = 1, rest = 0.
pub fn build_transition_matrix(config: &ThermalSimConfig) -> CsrMatrix {
    let alpha = config.fluid.thermal_diffusivity();
    let dt = config.time_step_dt;
    let hx2 = config.hx().powi(2);
    let hy2 = config.hy().powi(2);
    let hz2 = config.hz().powi(2);

    let cx = dt * alpha / hx2;
    let cy = dt * alpha / hy2;
    let cz = dt * alpha / hz2;
    let c_center = 1.0 - 2.0 * (cx + cy + cz);

    assert!(
        c_center > 0.0,
        "STABILITY VIOLATION: c_center={:.6}. Reduce dt or increase grid spacing.\n\
         Max stable dt = {:.6}s, current dt = {:.6}s",
        c_center,
        config.max_stable_dt(alpha),
        dt
    );

    let n = config.total_nodes();
    let mut coo = CooMatrix::with_capacity(n, n, n * 7);

    for i in 0..config.nx {
        for j in 0..config.ny {
            for k in 0..config.nz {
                let n_idx = config.linear_index(i, j, k);

                if config.is_boundary(i, j, k) {
                    match config.boundary_type {
                        BoundaryType::Dirichlet => {
                            // Fixed temperature: identity row → T_new = T_boundary
                            coo.push(n_idx, n_idx, 1.0);
                        }
                        BoundaryType::Neumann => {
                            // Insulated: dT/dn = 0 (ghost node = interior neighbor).
                            // Count how many directions are NOT boundaries (active neighbors).
                            let mut diag = 1.0;
                            let has_xm = i > 0;
                            let has_xp = i < config.nx - 1;
                            let has_ym = j > 0;
                            let has_yp = j < config.ny - 1;
                            let has_zm = k > 0;
                            let has_zp = k < config.nz - 1;

                            // For each direction: if neighbor exists, add off-diagonal;
                            // if not (boundary face), the ghost node mirrors the boundary
                            // node itself, so heat doesn't leave.
                            if has_xm { coo.push(n_idx, config.linear_index(i - 1, j, k), cx); diag -= cx; }
                            if has_xp { coo.push(n_idx, config.linear_index(i + 1, j, k), cx); diag -= cx; }
                            if has_ym { coo.push(n_idx, config.linear_index(i, j - 1, k), cy); diag -= cy; }
                            if has_yp { coo.push(n_idx, config.linear_index(i, j + 1, k), cy); diag -= cy; }
                            if has_zm { coo.push(n_idx, config.linear_index(i, j, k - 1), cz); diag -= cz; }
                            if has_zp { coo.push(n_idx, config.linear_index(i, j, k + 1), cz); diag -= cz; }
                            coo.push(n_idx, n_idx, diag);
                        }
                        BoundaryType::Mixed { h_conv } => {
                            // Convective: -λ·dT/dn = h·(T - T_amb).
                            // Discretized: boundary node gets additional loss term.
                            let lambda = config.fluid.thermal_conductivity;
                            // Count missing neighbor directions
                            let has_xm = i > 0;
                            let has_xp = i < config.nx - 1;
                            let has_ym = j > 0;
                            let has_yp = j < config.ny - 1;
                            let has_zm = k > 0;
                            let has_zp = k < config.nz - 1;

                            let mut diag = 1.0;
                            let mut n_faces_exposed = 0_usize;

                            if has_xm { coo.push(n_idx, config.linear_index(i - 1, j, k), cx); diag -= cx; }
                            else { n_faces_exposed += 1; }
                            if has_xp { coo.push(n_idx, config.linear_index(i + 1, j, k), cx); diag -= cx; }
                            else { n_faces_exposed += 1; }
                            if has_ym { coo.push(n_idx, config.linear_index(i, j - 1, k), cy); diag -= cy; }
                            else { n_faces_exposed += 1; }
                            if has_yp { coo.push(n_idx, config.linear_index(i, j + 1, k), cy); diag -= cy; }
                            else { n_faces_exposed += 1; }
                            if has_zm { coo.push(n_idx, config.linear_index(i, j, k - 1), cz); diag -= cz; }
                            else { n_faces_exposed += 1; }
                            if has_zp { coo.push(n_idx, config.linear_index(i, j, k + 1), cz); diag -= cz; }
                            else { n_faces_exposed += 1; }

                            // Biot-like convective loss: each exposed face loses
                            // dt · h · (surface/volume) per exposed face.
                            // Simplified: loss_per_face ≈ dt·α·h/(λ·h_grid)
                            let h_grid = config.hx().min(config.hy()).min(config.hz());
                            let conv_loss = dt * alpha * h_conv / (lambda * h_grid);
                            diag -= conv_loss * n_faces_exposed as f64;

                            coo.push(n_idx, n_idx, diag);
                            // Note: the convective term also contributes T_ambient to RHS.
                            // This is handled in apply_boundary_conditions_mixed().
                        }
                    }
                } else {
                    // Interior: 7-point stencil (unchanged)
                    coo.push(n_idx, n_idx, c_center);
                    coo.push(n_idx, config.linear_index(i - 1, j, k), cx);
                    coo.push(n_idx, config.linear_index(i + 1, j, k), cx);
                    coo.push(n_idx, config.linear_index(i, j - 1, k), cy);
                    coo.push(n_idx, config.linear_index(i, j + 1, k), cy);
                    coo.push(n_idx, config.linear_index(i, j, k - 1), cz);
                    coo.push(n_idx, config.linear_index(i, j, k + 1), cz);
                }
            }
        }
    }

    coo.to_csr()
}

// ─── Temperature Vector ────────────────────────────────────────────────────

/// Initialize temperature: interior = t_initial, boundary = t_boundary.
pub fn init_temperature_vector(config: &ThermalSimConfig) -> Vec<f64> {
    let n = config.total_nodes();
    let mut t = vec![config.t_initial; n];

    for i in 0..config.nx {
        for j in 0..config.ny {
            for k in 0..config.nz {
                if config.is_boundary(i, j, k) {
                    t[config.linear_index(i, j, k)] = config.t_boundary;
                }
            }
        }
    }
    t
}

/// Re-apply boundary conditions after each SpMV step.
/// - Dirichlet: force T = T_boundary on all walls.
/// - Neumann: no action needed (matrix handles insulation).
/// - Mixed: add convective source term T_ambient contribution.
pub fn apply_boundary_conditions(t: &mut [f64], config: &ThermalSimConfig) {
    match config.boundary_type {
        BoundaryType::Dirichlet => {
            for i in 0..config.nx {
                for j in 0..config.ny {
                    for k in 0..config.nz {
                        if config.is_boundary(i, j, k) {
                            t[config.linear_index(i, j, k)] = config.t_boundary;
                        }
                    }
                }
            }
        }
        BoundaryType::Neumann => {
            // No action — insulated walls, no heat leaves.
        }
        BoundaryType::Mixed { h_conv } => {
            // Add convective source: each exposed boundary node gets pulled
            // toward T_ambient proportional to h_conv.
            let alpha = config.fluid.thermal_diffusivity();
            let lambda = config.fluid.thermal_conductivity;
            let dt = config.time_step_dt;
            let h_grid = config.hx().min(config.hy()).min(config.hz());
            let conv_coeff = dt * alpha * h_conv / (lambda * h_grid);

            for i in 0..config.nx {
                for j in 0..config.ny {
                    for k in 0..config.nz {
                        if config.is_boundary(i, j, k) {
                            let idx = config.linear_index(i, j, k);
                            // Count exposed faces
                            let mut n_faces = 0_usize;
                            if i == 0 { n_faces += 1; }
                            if i == config.nx - 1 { n_faces += 1; }
                            if j == 0 { n_faces += 1; }
                            if j == config.ny - 1 { n_faces += 1; }
                            if k == 0 { n_faces += 1; }
                            if k == config.nz - 1 { n_faces += 1; }
                            // RHS contribution: add conv_coeff * n_faces * T_ambient
                            t[idx] += conv_coeff * n_faces as f64 * config.t_boundary;
                        }
                    }
                }
            }
        }
    }
}

// ─── Heat Source Application ──────────────────────────────────────────────

/// Impose localized heat sources onto the temperature field.
///
/// Each source has a Gaussian spatial profile (3σ cutoff for performance)
/// and optional sinusoidal temporal modulation (ω > 0).
/// We use `max(T_current, T_source)` rather than addition — this preserves
/// energy conservation by treating sources as Dirichlet-like constraints.
pub fn apply_heat_sources(t: &mut [f64], config: &ThermalSimConfig, step: usize) {
    if config.heat_sources.is_empty() {
        return;
    }
    let time = step as f64 * config.time_step_dt;
    let hx = config.hx();
    let hy = config.hy();
    let hz = config.hz();

    for src in &config.heat_sources {
        // Temporal modulation: static (ω=0) or sinusoidal envelope [0, 1]
        let modulation = if src.omega > 0.0 {
            0.5 * (1.0 + (src.omega * time).sin())
        } else {
            1.0
        };

        // 3σ cutoff — beyond this the Gaussian contribution is < 0.01%
        let cutoff_sq = (3.0 * src.radius).powi(2);
        let inv_2sigma_sq = 1.0 / (2.0 * src.radius.powi(2));

        for i in 0..config.nx {
            let px = i as f64 * hx;
            let dx = px - src.x;
            if dx * dx > cutoff_sq { continue; }

            for j in 0..config.ny {
                let py = j as f64 * hy;
                let dy = py - src.y;
                if dx * dx + dy * dy > cutoff_sq { continue; }

                for k in 0..config.nz {
                    let pz = k as f64 * hz;
                    let dz = pz - src.z;
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq < cutoff_sq {
                        let weight = (-dist_sq * inv_2sigma_sq).exp();
                        let target = config.t_boundary
                            + (src.temperature - config.t_boundary) * modulation * weight;
                        let idx = config.linear_index(i, j, k);
                        t[idx] = t[idx].max(target);
                    }
                }
            }
        }
    }
}

// ─── Thermal Analytics ────────────────────────────────────────────────────

/// Compute deep analytics for the current temperature field:
/// - Peak thermal gradient |∇T| via central differences [°C/m]
/// - Normalized Shannon entropy of the field (equilibrium → max entropy)
/// - Heat loss percentage relative to initial total energy
pub fn compute_analytics(
    t: &[f64],
    config: &ThermalSimConfig,
    e_initial: f64,
) -> (f64, f64, f64) {
    let hx = config.hx();
    let hy = config.hy();
    let hz = config.hz();
    let n = t.len() as f64;

    // 1. Peak gradient — central difference on interior nodes
    let mut max_grad = 0.0_f64;
    for i in 1..config.nx - 1 {
        for j in 1..config.ny - 1 {
            for k in 1..config.nz - 1 {
                let dtdx = (t[config.linear_index(i + 1, j, k)]
                    - t[config.linear_index(i - 1, j, k)])
                    / (2.0 * hx);
                let dtdy = (t[config.linear_index(i, j + 1, k)]
                    - t[config.linear_index(i, j - 1, k)])
                    / (2.0 * hy);
                let dtdz = (t[config.linear_index(i, j, k + 1)]
                    - t[config.linear_index(i, j, k - 1)])
                    / (2.0 * hz);
                let grad_mag = (dtdx * dtdx + dtdy * dtdy + dtdz * dtdz).sqrt();
                max_grad = max_grad.max(grad_mag);
            }
        }
    }

    // 2. Normalized Shannon entropy — measures thermal equilibrium approach
    //    H = -Σ(p_i · ln(p_i)) / ln(N), where p_i = T_i / ΣT
    let t_sum: f64 = t.iter().filter(|&&v| v > 0.0).sum();
    let entropy_rate = if t_sum > 0.0 {
        let raw: f64 = t
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / t_sum;
                -p * p.ln()
            })
            .sum();
        raw / n.ln() // normalize to [0, 1]
    } else {
        0.0
    };

    // 3. Heat loss — total thermal energy remaining vs initial
    let e_current: f64 = t.iter().sum();
    let heat_loss_pct = if e_initial > 0.0 {
        ((e_initial - e_current) / e_initial * 100.0).max(0.0)
    } else {
        0.0
    };

    (max_grad, entropy_rate, heat_loss_pct)
}

// ─── Snapshot Computation ──────────────────────────────────────────────────

/// Compute physical metrics from the current temperature field.
pub fn compute_snapshot(
    t: &[f64],
    step: usize,
    config: &ThermalSimConfig,
    e_initial: f64,
) -> ThermalSnapshot {
    let n = t.len();

    // Center temperature
    let ci = config.nx / 2;
    let cj = config.ny / 2;
    let ck = config.nz / 2;
    let t_center = t[config.linear_index(ci, cj, ck)];

    // TEG hot side: average of nodes adjacent to TEG wall (one layer inward)
    let t_teg_hot = average_teg_wall_temp(t, config);
    let t_teg_cold = config.t_boundary;
    let delta_t = (t_teg_hot - t_teg_cold).max(0.0);

    // Field statistics
    let mean_temp = t.iter().sum::<f64>() / n as f64;
    let max_temp = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_temp = t.iter().cloned().fold(f64::INFINITY, f64::min);

    let power_w = config.teg.power_output(delta_t);

    // Deep analytics: gradient, entropy, heat loss
    let (max_gradient, entropy_rate, heat_loss_pct) =
        compute_analytics(t, config, e_initial);

    ThermalSnapshot {
        time_s: step as f64 * config.time_step_dt,
        step,
        t_center,
        t_teg_hot,
        t_teg_cold,
        delta_t,
        voltage: config.teg.voltage(delta_t),
        current: config.teg.current(delta_t),
        power_w,
        power_mw: power_w * 1000.0,
        efficiency_pct: config.teg.efficiency(delta_t) * 100.0,
        mean_temp,
        max_temp,
        min_temp,
        max_gradient,
        entropy_rate,
        heat_loss_pct,
    }
}

/// Average temperature of nodes one layer inward from the TEG wall.
fn average_teg_wall_temp(t: &[f64], config: &ThermalSimConfig) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;

    for i in 0..config.nx {
        for j in 0..config.ny {
            for k in 0..config.nz {
                let is_adjacent = match config.teg_wall {
                    TegWall::XMin => i == 1,
                    TegWall::XMax => i == config.nx - 2,
                    TegWall::YMin => j == 1,
                    TegWall::YMax => j == config.ny - 2,
                    TegWall::ZMin => k == 1,
                    TegWall::ZMax => k == config.nz - 2,
                };
                if is_adjacent {
                    sum += t[config.linear_index(i, j, k)];
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        config.t_boundary
    }
}

// ─── Main Simulation Loop ──────────────────────────────────────────────────

/// Run the full thermal simulation.
///
/// Uses SpMV from CsrMatrix as the core engine:
///   T_new = A · T_old  (repeated total_steps times)
///
/// Progress is reported via `progress` (numeric 0..100) and
/// `phase` (descriptive string for the UI).
pub fn run_thermal_simulation(
    config: &mut ThermalSimConfig,
    progress: &ProgressHandle,
    phase: &Arc<Mutex<String>>,
) -> anyhow::Result<ThermalSimResult> {
    let alpha = config.fluid.thermal_diffusivity();

    // Auto-calculate dt if not set (80% of stability limit for safety margin)
    if config.time_step_dt <= 0.0 {
        config.time_step_dt = 0.8 * config.max_stable_dt(alpha);
    }

    let max_dt = config.max_stable_dt(alpha);
    anyhow::ensure!(
        config.time_step_dt <= max_dt,
        "dt={:.6}s exceeds stability limit={:.6}s. Reduce dt.",
        config.time_step_dt,
        max_dt
    );

    // 1. Build transition matrix A
    {
        let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
        *p = "Building transition matrix A...".into();
    }
    progress.set(5, 100);
    let a_matrix = build_transition_matrix(config);

    // 1b. If MKL solver selected, create MKL sparse handle from CSR.
    //     Handle is reused across all timesteps — Inspector phase runs once,
    //     subsequent calls hit the fast Executor path.
    #[cfg(feature = "mkl")]
    let mkl_handle = if config.solver == ThermalSolver::IntelMKL {
        let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
        *p = "Creating MKL sparse handle (Inspector phase)...".into();
        drop(p);
        Some(MklSparseHandle::from_csr(&a_matrix, config.total_steps as i32)?)
    } else {
        None
    };

    // 2. Initialize temperature vector
    let mut t_current = init_temperature_vector(config);
    let mut t_next = vec![0.0; config.total_nodes()];

    // Initial total thermal energy — baseline for heat loss analytics
    let e_initial: f64 = t_current.iter().sum();

    // 3. First snapshot (t=0)
    let mut snapshots = Vec::new();
    snapshots.push(compute_snapshot(&t_current, 0, config, e_initial));

    let mut total_muls = 0usize;
    let sim_start = std::time::Instant::now();

    // 4. Time-stepping loop — dispatch based on solver selection
    for step in 1..=config.total_steps {
        // T_new = A · T_old
        match config.solver {
            ThermalSolver::NativeSparse => {
                a_matrix.spmv_into(&t_current, &mut t_next);
            }
            #[cfg(feature = "mkl")]
            ThermalSolver::IntelMKL => {
                mkl_handle.as_ref().unwrap().spmv_into(&t_current, &mut t_next)?;
            }
        }
        total_muls += 1;

        // Apply boundary conditions, then impose localized heat sources
        apply_boundary_conditions(&mut t_next, config);
        apply_heat_sources(&mut t_next, config, step);

        // Swap buffers (avoid allocation)
        std::mem::swap(&mut t_current, &mut t_next);

        // Save snapshot periodically
        if step % config.save_every_n == 0 || step == config.total_steps {
            snapshots.push(compute_snapshot(&t_current, step, config, e_initial));
        }

        // Update progress every 50 steps
        if step % 50 == 0 || step == config.total_steps {
            let pct = ((step as f64 / config.total_steps as f64) * 90.0) as u32 + 5;
            progress.set(pct, 100);
            if let Some(snap) = snapshots.last() {
                let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
                *p = format!(
                    "Step {}/{} | t={:.1}s  \u{0394}T={:.1}\u{00b0}C  V={:.3}V  P={:.2}mW",
                    step, config.total_steps, snap.time_s, snap.delta_t,
                    snap.voltage, snap.power_mw
                );
            }
        }
    }

    let computation_ms = sim_start.elapsed().as_secs_f64() * 1000.0;

    // 5. Derive engineering metrics
    let threshold_voltage = 1.0_f64;
    let runtime_minutes = snapshots
        .iter()
        .find(|s| s.voltage < threshold_voltage)
        .map(|s| s.time_s / 60.0)
        .unwrap_or(config.total_steps as f64 * config.time_step_dt / 60.0);

    let max_power_mw = snapshots
        .iter()
        .map(|s| s.power_mw)
        .fold(0.0_f64, f64::max);

    let time_to_max_power_s = snapshots
        .iter()
        .max_by(|a, b| a.power_mw.partial_cmp(&b.power_mw).unwrap_or(std::cmp::Ordering::Equal))
        .map(|s| s.time_s)
        .unwrap_or(0.0);

    // Total energy via trapezoidal integration [mJ]
    let total_energy_mj: f64 = snapshots
        .windows(2)
        .map(|w| {
            let dt = w[1].time_s - w[0].time_s;
            (w[0].power_mw + w[1].power_mw) / 2.0 * dt
        })
        .sum();

    let average_power_mw = if !snapshots.is_empty() {
        snapshots.iter().map(|s| s.power_mw).sum::<f64>() / snapshots.len() as f64
    } else {
        0.0
    };

    progress.set(100, 100);
    {
        let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
        *p = "Simulation complete".into();
    }

    Ok(ThermalSimResult {
        config: config.clone(),
        snapshots,
        final_field: t_current,
        computation_ms,
        total_matrix_multiplications: total_muls,
        runtime_minutes,
        threshold_voltage,
        max_power_mw,
        time_to_max_power_s,
        average_power_mw,
        total_energy_mj,
    })
}

// ─── Preset Configuration ──────────────────────────────────────────────────

/// Default config matching a typical student tank project.
/// Reservoir ~150×80×60mm, water at 85°C, ambient 20°C.
pub fn config_tank_project_default() -> ThermalSimConfig {
    ThermalSimConfig {
        length_x: 0.150,
        length_y: 0.080,
        length_z: 0.060,
        nx: 16,
        ny: 16,
        nz: 16,
        t_initial: 85.0,
        t_boundary: 20.0,
        fluid: FluidProperties::water(),
        teg: TegProperties::standard_bi2te3(),
        time_step_dt: 0.0, // auto
        total_steps: 1000,
        save_every_n: 10,
        teg_wall: TegWall::XMin,
        boundary_type: BoundaryType::Dirichlet,
        solver: ThermalSolver::NativeSparse,
        heat_sources: Vec::new(),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seebeck_voltage_formula() {
        let teg = TegProperties::standard_bi2te3();
        let delta_t = 60.0;
        let v = teg.voltage(delta_t);
        // V = S · ΔT = 0.05 · 60 = 3.0 V
        assert!((v - 3.0).abs() < 1e-10, "Expected 3.0V, got {}V", v);
    }

    #[test]
    fn test_seebeck_power() {
        let teg = TegProperties::standard_bi2te3();
        let delta_t = 60.0;
        // V = 3.0V, I = 3.0/(2+2) = 0.75A, P = 0.75² × 2 = 1.125W
        let p = teg.power_output(delta_t);
        assert!((p - 1.125).abs() < 1e-10, "Expected 1.125W, got {}W", p);
    }

    #[test]
    fn test_fluid_diffusivity() {
        let water = FluidProperties::water();
        let alpha = water.thermal_diffusivity();
        // α = 0.598 / (998 * 4182) ≈ 1.432e-7
        assert!(alpha > 1.4e-7 && alpha < 1.5e-7, "alpha={}", alpha);
    }

    #[test]
    fn test_stability_criterion_catches_bad_dt() {
        let mut config = config_tank_project_default();
        config.time_step_dt = 1000.0; // way too large
        let result = std::panic::catch_unwind(|| build_transition_matrix(&config));
        assert!(result.is_err(), "Should panic on unstable dt");
    }

    #[test]
    fn test_temperature_decreases_over_time() {
        let mut config = config_tank_project_default();
        config.nx = 8;
        config.ny = 8;
        config.nz = 8;
        config.total_steps = 100;
        config.save_every_n = 50;

        let progress = ProgressHandle::new(100);
        let phase = Arc::new(Mutex::new(String::new()));
        let result = run_thermal_simulation(&mut config, &progress, &phase).unwrap();

        let first = result.snapshots.first().unwrap();
        let last = result.snapshots.last().unwrap();
        assert!(
            last.t_center < first.t_center,
            "Temperature should decrease: {} -> {}",
            first.t_center,
            last.t_center
        );
    }

    #[test]
    fn test_boundary_conditions_stable() {
        let mut config = config_tank_project_default();
        config.nx = 8;
        config.ny = 8;
        config.nz = 8;
        config.total_steps = 50;
        config.save_every_n = 50;

        let progress = ProgressHandle::new(100);
        let phase = Arc::new(Mutex::new(String::new()));
        let result = run_thermal_simulation(&mut config, &progress, &phase).unwrap();

        let t = &result.final_field;
        let eps = 1e-10;
        for i in 0..config.nx {
            for j in 0..config.ny {
                for k in 0..config.nz {
                    if config.is_boundary(i, j, k) {
                        let val = t[config.linear_index(i, j, k)];
                        assert!(
                            (val - config.t_boundary).abs() < eps,
                            "Boundary node ({},{},{}) = {} != {}",
                            i, j, k, val, config.t_boundary
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_auto_dt_within_stability() {
        let mut config = config_tank_project_default();
        config.time_step_dt = 0.0; // auto

        // Set grid BEFORE computing max_dt so both use the same geometry
        config.nx = 6;
        config.ny = 6;
        config.nz = 6;
        config.total_steps = 1;
        config.save_every_n = 1;

        let alpha = config.fluid.thermal_diffusivity();
        let max_dt = config.max_stable_dt(alpha);

        let progress = ProgressHandle::new(100);
        let phase = Arc::new(Mutex::new(String::new()));
        let result = run_thermal_simulation(&mut config, &progress, &phase).unwrap();
        assert!(
            result.config.time_step_dt <= max_dt,
            "auto dt {} > max {}",
            result.config.time_step_dt,
            max_dt
        );
        assert!(result.config.time_step_dt > 0.0);
    }

    #[test]
    fn test_matrix_structure() {
        let mut config = config_tank_project_default();
        config.nx = 4;
        config.ny = 4;
        config.nz = 4;
        config.time_step_dt = 0.0;
        let alpha = config.fluid.thermal_diffusivity();
        config.time_step_dt = 0.5 * config.max_stable_dt(alpha);

        let a = build_transition_matrix(&config);
        let n = config.total_nodes();
        assert_eq!(a.rows, n);
        assert_eq!(a.cols, n);
        // Interior nodes: 2×2×2 = 8, each has 7 entries = 56
        // Boundary nodes: 64 - 8 = 56, each has 1 entry = 56
        // Total NNZ = 56 + 56 = 112
        assert_eq!(a.nnz(), 112);
    }

    #[test]
    fn test_energy_decreasing() {
        // Total internal energy (sum of temperatures) should decrease
        // as heat flows out through boundaries.
        let mut config = config_tank_project_default();
        config.nx = 8;
        config.ny = 8;
        config.nz = 8;
        config.total_steps = 200;
        config.save_every_n = 100;

        let progress = ProgressHandle::new(100);
        let phase = Arc::new(Mutex::new(String::new()));
        let result = run_thermal_simulation(&mut config, &progress, &phase).unwrap();

        let first = &result.snapshots[0];
        let last = result.snapshots.last().unwrap();
        assert!(
            last.mean_temp < first.mean_temp,
            "Mean temp should decrease: {} -> {}",
            first.mean_temp,
            last.mean_temp
        );
    }
}
