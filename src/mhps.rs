// =============================================================================
//  MHPS — Material Heat Propagation Simulator
//  Chapter 22: 2D Fourier heat equation FDM solver for solid materials.
//  Variable geometry, multiple heat sources, material library.
// =============================================================================

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::common::ProgressHandle;
use crate::sparse::{CooMatrix, CsrMatrix};

// ─── Material Library ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Material {
    pub name: &'static str,
    pub thermal_conductivity: f64, // lambda [W/(m*K)]
    pub density: f64,              // rho    [kg/m^3]
    pub specific_heat: f64,        // c      [J/(kg*K)]
    pub melting_point_c: Option<f64>,
    pub description: &'static str,
}

impl Material {
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    pub fn characteristic(&self) -> &'static str {
        match (self.thermal_conductivity * 10.0) as u32 {
            3000.. => "Excellent conductor — heat spreads almost instantly",
            500..  => "Good conductor — heat spreads rapidly (seconds)",
            50..   => "Moderate conductor — heat spreads in minutes",
            5..    => "Poor conductor — heat spreads slowly (hours)",
            _      => "Thermal insulator — minimal heat transfer",
        }
    }
}

pub const MATERIALS: &[Material] = &[
    Material {
        name: "Copper",
        thermal_conductivity: 385.0,
        density: 8960.0,
        specific_heat: 385.0,
        melting_point_c: Some(1085.0),
        description: "PCB traces, heat spreaders. Extremely fast propagation.",
    },
    Material {
        name: "Aluminum",
        thermal_conductivity: 205.0,
        density: 2700.0,
        specific_heat: 900.0,
        melting_point_c: Some(660.0),
        description: "Heatsinks, enclosures. Very good conductor, light weight.",
    },
    Material {
        name: "Steel (carbon)",
        thermal_conductivity: 50.0,
        density: 7850.0,
        specific_heat: 490.0,
        melting_point_c: Some(1370.0),
        description: "Structural parts, frames. Moderate conductivity.",
    },
    Material {
        name: "Stainless Steel",
        thermal_conductivity: 16.0,
        density: 8000.0,
        specific_heat: 500.0,
        melting_point_c: Some(1400.0),
        description: "Corrosion-resistant parts. Poor thermal conductor vs carbon steel.",
    },
    Material {
        name: "Iron (cast)",
        thermal_conductivity: 55.0,
        density: 7200.0,
        specific_heat: 460.0,
        melting_point_c: Some(1200.0),
        description: "Engine blocks, cookware. Good structural thermal mass.",
    },
    Material {
        name: "Titanium",
        thermal_conductivity: 22.0,
        density: 4510.0,
        specific_heat: 520.0,
        melting_point_c: Some(1668.0),
        description: "Aerospace parts. Low conductivity despite metallic nature.",
    },
    Material {
        name: "Concrete",
        thermal_conductivity: 1.7,
        density: 2300.0,
        specific_heat: 880.0,
        melting_point_c: None,
        description: "Building structures. Slow thermal response, high mass.",
    },
    Material {
        name: "Wood (oak)",
        thermal_conductivity: 0.17,
        density: 750.0,
        specific_heat: 1700.0,
        melting_point_c: None,
        description: "Natural insulator. Heat spreads very slowly across the grain.",
    },
    Material {
        name: "Glass",
        thermal_conductivity: 1.0,
        density: 2500.0,
        specific_heat: 840.0,
        melting_point_c: Some(700.0),
        description: "Poor conductor. Windows, substrates.",
    },
    Material {
        name: "Ceramic (Al2O3)",
        thermal_conductivity: 25.0,
        density: 3900.0,
        specific_heat: 775.0,
        melting_point_c: Some(2054.0),
        description: "Electronic substrates, furnace components.",
    },
    Material {
        name: "Silicon",
        thermal_conductivity: 150.0,
        density: 2330.0,
        specific_heat: 700.0,
        melting_point_c: Some(1414.0),
        description: "Semiconductors. Good conductor — thermal design critical for chips.",
    },
    Material {
        name: "Graphite",
        thermal_conductivity: 120.0,
        density: 2200.0,
        specific_heat: 710.0,
        melting_point_c: None,
        description: "Thermal interface pads, electrodes. Anisotropic in reality.",
    },
    Material {
        name: "PTFE (Teflon)",
        thermal_conductivity: 0.25,
        density: 2200.0,
        specific_heat: 1000.0,
        melting_point_c: Some(327.0),
        description: "Electrical insulation. Excellent thermal insulator.",
    },
    Material {
        name: "Epoxy resin",
        thermal_conductivity: 0.35,
        density: 1200.0,
        specific_heat: 1400.0,
        melting_point_c: None,
        description: "PCB substrate, adhesives. Thermal bottleneck in electronics.",
    },
    Material {
        name: "Custom",
        thermal_conductivity: 1.0,
        density: 1000.0,
        specific_heat: 1000.0,
        melting_point_c: None,
        description: "User-defined material. Enter your own values.",
    },
];

// ─── Grid Helpers ────────────────────────────────────────────────────────────

/// Compute proportional nx, ny, nz from physical dimensions and a node budget.
/// The largest axis gets `grid_n` nodes; others are scaled proportionally.
/// Minimum 3 nodes on any active axis.
pub fn proportional_grid(lx: f64, ly: f64, lz: f64, grid_n: usize, enable_3d: bool) -> (usize, usize, usize) {
    let max_dim = lx.max(ly).max(if enable_3d { lz } else { 0.0 });
    if max_dim < 1e-12 {
        return (grid_n, grid_n, if enable_3d { grid_n } else { 1 });
    }
    let nx = ((lx / max_dim) * grid_n as f64).round().max(3.0) as usize;
    let ny = ((ly / max_dim) * grid_n as f64).round().max(3.0) as usize;
    let nz = if enable_3d {
        ((lz / max_dim) * grid_n as f64).round().max(3.0) as usize
    } else {
        1
    };
    (nx, ny, nz)
}

// ─── Geometry ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ShapeDefinition {
    Rectangle,
    Polygon(Vec<(f64, f64)>),
    RoundedRect { radius_x: f64, radius_y: f64 },
    LShape { cutout_x: f64, cutout_y: f64 },
    Custom { description: String },
}

#[derive(Debug, Clone)]
pub struct MhpsGeometry {
    pub length_x: f64,        // [m]
    pub length_y: f64,        // [m]
    pub length_z: f64,        // [m] thickness/depth axis
    pub base_thickness: f64,  // [m] (legacy, = length_z for uniform)
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,            // nodes along Z (1 = 2D mode)
    /// thickness_map[i * ny + j] — thickness of node (i,j) in [m].
    /// 0.0 = void (hole).
    pub thickness_map: Vec<f64>,
    /// material_map[i * ny + j] — index into MATERIALS.
    pub material_map: Vec<usize>,
    /// shape_mask[i * ny + j] — true = node present, false = void
    pub shape_mask: Vec<bool>,
    pub shape_definition: ShapeDefinition,
}

impl MhpsGeometry {
    pub fn uniform(lx: f64, ly: f64, thickness: f64, nx: usize, ny: usize, mat_idx: usize) -> Self {
        let n = nx * ny;
        Self {
            length_x: lx,
            length_y: ly,
            length_z: thickness,
            base_thickness: thickness,
            nx,
            ny,
            nz: 1,
            thickness_map: vec![thickness; n],
            material_map: vec![mat_idx; n],
            shape_mask: vec![true; n],
            shape_definition: ShapeDefinition::Rectangle,
        }
    }

    /// 3D-aware constructor. For nz=1, behaves identically to `uniform`.
    pub fn uniform_3d(lx: f64, ly: f64, lz: f64, nx: usize, ny: usize, nz: usize, mat_idx: usize) -> Self {
        let total = nx * ny * nz.max(1);
        Self {
            length_x: lx,
            length_y: ly,
            length_z: lz,
            base_thickness: lz,
            nx,
            ny,
            nz: nz.max(1),
            thickness_map: vec![lz; total],
            material_map: vec![mat_idx; total],
            shape_mask: vec![true; total],
            shape_definition: ShapeDefinition::Rectangle,
        }
    }

    pub fn hx(&self) -> f64 {
        self.length_x / (self.nx - 1).max(1) as f64
    }

    pub fn hy(&self) -> f64 {
        self.length_y / (self.ny - 1).max(1) as f64
    }

    pub fn hz(&self) -> f64 {
        if self.nz <= 1 {
            self.length_z
        } else {
            self.length_z / (self.nz - 1) as f64
        }
    }

    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    /// 3D linear index: i along X, j along Y, k along Z.
    pub fn idx3(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    pub fn total_nodes(&self) -> usize {
        self.nx * self.ny * self.nz.max(1)
    }

    pub fn is_void(&self, i: usize, j: usize) -> bool {
        let idx = if self.nz <= 1 {
            self.idx(i, j)
        } else {
            self.idx3(i, j, 0)
        };
        if idx < self.shape_mask.len() && !self.shape_mask[idx] {
            return true;
        }
        if idx < self.thickness_map.len() {
            self.thickness_map[idx] < 1e-10
        } else {
            true
        }
    }

    /// Check void status for a 3D node.
    pub fn is_void3(&self, i: usize, j: usize, k: usize) -> bool {
        let idx = self.idx3(i, j, k);
        if idx >= self.shape_mask.len() {
            return true;
        }
        !self.shape_mask[idx]
    }

    /// Rebuild shape_mask from shape_definition.
    /// For 3D (nz > 1), extrudes the 2D mask across all Z layers.
    pub fn rebuild_mask(&mut self) {
        let n = self.nx * self.ny;
        self.shape_mask = vec![true; n];

        let def = self.shape_definition.clone();
        match def {
            ShapeDefinition::Rectangle => {
                // All nodes active — already true
            }
            ShapeDefinition::Polygon(ref vertices) => {
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        let x = i as f64 * self.hx();
                        let y = j as f64 * self.hy();
                        self.shape_mask[i * self.ny + j] = point_in_polygon(x, y, vertices);
                    }
                }
            }
            ShapeDefinition::RoundedRect { radius_x, radius_y } => {
                let rx = radius_x;
                let ry = radius_y;
                let cx_max = self.length_x - rx;
                let cy_max = self.length_y - ry;
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        let x = i as f64 * self.hx();
                        let y = j as f64 * self.hy();
                        let in_corner_zone =
                            (x < rx || x > cx_max) && (y < ry || y > cy_max);
                        let in_shape = if in_corner_zone {
                            let ex = if x < rx { rx } else { cx_max };
                            let ey = if y < ry { ry } else { cy_max };
                            let dx = (x - ex) / rx;
                            let dy = (y - ey) / ry;
                            dx * dx + dy * dy <= 1.0
                        } else {
                            true
                        };
                        self.shape_mask[i * self.ny + j] = in_shape;
                    }
                }
            }
            ShapeDefinition::LShape { cutout_x, cutout_y } => {
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        let x = i as f64 * self.hx();
                        let y = j as f64 * self.hy();
                        let in_cutout =
                            x > self.length_x - cutout_x && y > self.length_y - cutout_y;
                        self.shape_mask[i * self.ny + j] = !in_cutout;
                    }
                }
            }
            ShapeDefinition::Custom { .. } => {
                // Mask already set manually — don't rebuild
            }
        }

        // Extrude 2D mask across Z layers for 3D
        if self.nz > 1 {
            let mask_2d = self.shape_mask.clone(); // nx*ny 2D mask
            let total = self.nx * self.ny * self.nz;
            self.shape_mask = vec![true; total];
            self.thickness_map.resize(total, self.base_thickness);
            self.material_map.resize(total, self.material_map.first().copied().unwrap_or(0));
            for i in 0..self.nx {
                for j in 0..self.ny {
                    let active = mask_2d[i * self.ny + j];
                    for k in 0..self.nz {
                        let idx3 = i * self.ny * self.nz + j * self.nz + k;
                        self.shape_mask[idx3] = active;
                        if !active {
                            self.thickness_map[idx3] = 0.0;
                        }
                    }
                }
            }
        } else {
            // Sync thickness_map: void nodes get thickness 0
            for idx in 0..n {
                if !self.shape_mask[idx] {
                    self.thickness_map[idx] = 0.0;
                }
            }
        }
    }

    pub fn set_region_thickness(
        &mut self,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        thickness: f64,
    ) {
        let ny = self.ny;
        for i in i_start..i_end.min(self.nx) {
            for j in j_start..j_end.min(ny) {
                self.thickness_map[i * ny + j] = thickness;
            }
        }
    }

    pub fn set_region_material(
        &mut self,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        mat_idx: usize,
    ) {
        let ny = self.ny;
        for i in i_start..i_end.min(self.nx) {
            for j in j_start..j_end.min(ny) {
                self.material_map[i * ny + j] = mat_idx;
            }
        }
    }

    pub fn drill_hole(&mut self, center_i: usize, center_j: usize, radius_nodes: usize) {
        let ci = center_i as isize;
        let cj = center_j as isize;
        let r = radius_nodes as isize;
        let nx = self.nx as isize;
        let nyy = self.ny as isize;
        let ny = self.ny;
        for di in -r..=r {
            for dj in -r..=r {
                if di * di + dj * dj <= r * r {
                    let i = ci + di;
                    let j = cj + dj;
                    if i >= 0 && i < nx && j >= 0 && j < nyy {
                        let idx = i as usize * ny + j as usize;
                        self.thickness_map[idx] = 0.0;
                        self.shape_mask[idx] = false;
                    }
                }
            }
        }
    }
}

/// Ray casting algorithm for point-in-polygon test.
pub fn point_in_polygon(x: f64, y: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Validate and add a heat source to config (no artificial limit).
pub fn add_heat_source(config: &mut MhpsConfig, source: HeatSource) -> anyhow::Result<()> {
    let geo = &config.geometry;
    anyhow::ensure!(
        source.position_i < geo.nx && source.position_j < geo.ny,
        "Source position ({}, {}) out of grid bounds ({}, {})",
        source.position_i,
        source.position_j,
        geo.nx,
        geo.ny
    );
    config.heat_sources.push(source);
    Ok(())
}

// ─── Heat Sources ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HeatSource {
    pub name: String,
    pub source_type: HeatSourceType,
    pub position_i: usize,
    pub position_j: usize,
    pub position_k: usize, // Z-axis position (0 for 2D mode)
    pub radius_nodes: usize, // 1 = point source
    pub temperature_c: f64,
    pub is_active: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum HeatSourceType {
    FixedTemperature,
    HeatFlux(f64), // [W/m^2]
}

impl HeatSource {
    pub fn new_heater(name: &str, i: usize, j: usize, radius: usize, temp: f64) -> Self {
        Self {
            name: name.into(),
            source_type: HeatSourceType::FixedTemperature,
            position_i: i,
            position_j: j,
            position_k: 0,
            radius_nodes: radius,
            temperature_c: temp,
            is_active: true,
        }
    }

    pub fn new_cooler(name: &str, i: usize, j: usize, radius: usize, temp: f64) -> Self {
        Self::new_heater(name, i, j, radius, temp)
    }

    pub fn contains(&self, i: usize, j: usize) -> bool {
        let di = (i as isize - self.position_i as isize).unsigned_abs();
        let dj = (j as isize - self.position_j as isize).unsigned_abs();
        di <= self.radius_nodes && dj <= self.radius_nodes
    }

    pub fn contains_3d(&self, i: usize, j: usize, k: usize) -> bool {
        let di = (i as isize - self.position_i as isize).unsigned_abs();
        let dj = (j as isize - self.position_j as isize).unsigned_abs();
        let dk = (k as isize - self.position_k as isize).unsigned_abs();
        di <= self.radius_nodes && dj <= self.radius_nodes && dk <= self.radius_nodes
    }
}

// ─── Simulation Config ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MhpsConfig {
    pub geometry: MhpsGeometry,
    pub heat_sources: Vec<HeatSource>,
    pub t_initial_c: f64,
    pub t_ambient_c: f64,
    pub convection_h: f64, // [W/(m^2*K)] — 0 = adiabatic
    pub time_step_dt: f64, // 0 = auto
    pub total_time_s: f64,
    pub save_every_n_steps: usize,
    pub convergence_epsilon: f64, // 0 = don't check
}

impl MhpsConfig {
    pub fn total_nodes(&self) -> usize {
        self.geometry.total_nodes()
    }

    /// Stability limit for explicit FDM (Fourier number criterion).
    /// 2D: Fo <= 1/4, 3D: Fo <= 1/6. Safety factor 0.8 applied.
    pub fn max_stable_dt(&self) -> f64 {
        let mut max_alpha = 0.0_f64;
        let used: HashSet<usize> = self.geometry.material_map.iter().copied().collect();
        for &idx in &used {
            let mat = &MATERIALS[idx.min(MATERIALS.len() - 1)];
            max_alpha = max_alpha.max(mat.thermal_diffusivity());
        }
        if max_alpha < 1e-20 {
            return 1.0;
        }
        let hx2 = self.geometry.hx().powi(2);
        let hy2 = self.geometry.hy().powi(2);
        let inv_h_sum = 1.0 / hx2 + 1.0 / hy2;

        if self.geometry.nz > 1 {
            let hz2 = self.geometry.hz().powi(2);
            // 3D: Fo <= 1/6, with 0.8 safety => Fo <= 0.133
            0.8 / (6.0 * max_alpha * (inv_h_sum + 1.0 / hz2))
        } else {
            // 2D: Fo <= 1/4, with 0.8 safety => Fo <= 0.20
            0.8 / (4.0 * max_alpha * inv_h_sum)
        }
    }
}

// ─── Sparse Matrix Assembly ──────────────────────────────────────────────────

fn build_mhps_matrix(config: &MhpsConfig, dt: f64) -> CsrMatrix {
    let geo = &config.geometry;
    let n = config.total_nodes();
    let hx2 = geo.hx().powi(2);
    let hy2 = geo.hy().powi(2);

    let mut coo = CooMatrix::with_capacity(n, n, n * 5);

    for i in 0..geo.nx {
        for j in 0..geo.ny {
            let n_idx = geo.idx(i, j);

            if geo.is_void(i, j) {
                coo.push(n_idx, n_idx, 1.0);
                continue;
            }

            let mat_idx = geo.material_map[n_idx];
            let mat = &MATERIALS[mat_idx.min(MATERIALS.len() - 1)];
            let alpha = mat.thermal_diffusivity();

            let is_boundary = i == 0 || i == geo.nx - 1 || j == 0 || j == geo.ny - 1;

            if is_boundary && config.convection_h > 0.0 {
                let thickness = geo.thickness_map[n_idx];
                let h = config.convection_h;
                let conv_coeff = h * dt / (mat.density * mat.specific_heat * thickness);
                coo.push(n_idx, n_idx, 1.0 - conv_coeff);
                continue;
            }

            let left = i > 0 && !geo.is_void(i - 1, j);
            let right = i + 1 < geo.nx && !geo.is_void(i + 1, j);
            let down = j > 0 && !geo.is_void(i, j - 1);
            let up = j + 1 < geo.ny && !geo.is_void(i, j + 1);

            let cx = dt * alpha / hx2;
            let cy = dt * alpha / hy2;

            let neighbors_x = (left as u8 + right as u8) as f64;
            let neighbors_y = (down as u8 + up as u8) as f64;
            let c_center = 1.0 - cx * neighbors_x - cy * neighbors_y;

            coo.push(n_idx, n_idx, c_center);

            if left {
                coo.push(n_idx, geo.idx(i - 1, j), cx);
            }
            if right {
                coo.push(n_idx, geo.idx(i + 1, j), cx);
            }
            if down {
                coo.push(n_idx, geo.idx(i, j - 1), cy);
            }
            if up {
                coo.push(n_idx, geo.idx(i, j + 1), cy);
            }
        }
    }

    coo.to_csr()
}

// ─── Boundary Condition Application ──────────────────────────────────────────

fn apply_mhps_bc(t: &mut [f64], config: &MhpsConfig) {
    let geo = &config.geometry;

    // 1. Heat sources (fixed temperature / flux)
    for source in &config.heat_sources {
        if !source.is_active {
            continue;
        }
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                if source.contains(i, j) && !geo.is_void(i, j) {
                    match source.source_type {
                        HeatSourceType::FixedTemperature => {
                            t[geo.idx(i, j)] = source.temperature_c;
                        }
                        HeatSourceType::HeatFlux(flux) => {
                            let mat = &MATERIALS[geo.material_map[geo.idx(i, j)]];
                            let thick = geo.thickness_map[geo.idx(i, j)];
                            let dv = geo.hx() * geo.hy() * thick;
                            let delta_t = flux * geo.hx() * geo.hy() * config.time_step_dt
                                / (mat.density * mat.specific_heat * dv);
                            t[geo.idx(i, j)] += delta_t;
                        }
                    }
                }
            }
        }
    }

    // 2. Boundary convection (Robin BC)
    if config.convection_h > 0.0 {
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                let is_boundary = i == 0 || i == geo.nx - 1 || j == 0 || j == geo.ny - 1;
                if is_boundary && !geo.is_void(i, j) {
                    let mat = &MATERIALS[geo.material_map[geo.idx(i, j)]];
                    let thick = geo.thickness_map[geo.idx(i, j)];
                    let h = config.convection_h;
                    let conv = h * config.time_step_dt
                        / (mat.density * mat.specific_heat * thick);
                    let idx = geo.idx(i, j);
                    t[idx] += conv * (config.t_ambient_c - t[idx]);
                }
            }
        }
    }

    // 3. Voids stay at ambient
    for i in 0..geo.nx {
        for j in 0..geo.ny {
            if geo.is_void(i, j) {
                t[geo.idx(i, j)] = config.t_ambient_c;
            }
        }
    }
}

// ─── 3D Boundary Conditions ──────────────────────────────────────────────────

fn apply_mhps_bc_3d(t: &mut [f64], config: &MhpsConfig) {
    let geo = &config.geometry;
    let nx = geo.nx;
    let ny = geo.ny;
    let nz = geo.nz;

    // 1. Heat sources (3D)
    for source in &config.heat_sources {
        if !source.is_active { continue; }
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if source.contains_3d(i, j, k) && !geo.is_void3(i, j, k) {
                        let idx = geo.idx3(i, j, k);
                        match source.source_type {
                            HeatSourceType::FixedTemperature => {
                                t[idx] = source.temperature_c;
                            }
                            HeatSourceType::HeatFlux(flux) => {
                                let mat = &MATERIALS[geo.material_map[idx].min(MATERIALS.len() - 1)];
                                let vol = geo.hx() * geo.hy() * geo.hz();
                                let delta_t = flux * geo.hx() * geo.hy() * config.time_step_dt
                                    / (mat.density * mat.specific_heat * vol);
                                t[idx] += delta_t;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Boundary convection (Robin BC) — all 6 faces
    if config.convection_h > 0.0 {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let on_boundary = i == 0 || i == nx - 1
                        || j == 0 || j == ny - 1
                        || k == 0 || k == nz - 1;
                    if on_boundary && !geo.is_void3(i, j, k) {
                        let idx = geo.idx3(i, j, k);
                        let mat = &MATERIALS[geo.material_map[idx].min(MATERIALS.len() - 1)];
                        let char_len = geo.hz(); // characteristic length for convection
                        let conv = config.convection_h * config.time_step_dt
                            / (mat.density * mat.specific_heat * char_len);
                        t[idx] += conv * (config.t_ambient_c - t[idx]);
                    }
                }
            }
        }
    }

    // 3. Void nodes stay at ambient
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if geo.is_void3(i, j, k) {
                    t[geo.idx3(i, j, k)] = config.t_ambient_c;
                }
            }
        }
    }
}

// ─── 3D Stencil Solver ──────────────────────────────────────────────────────

fn run_mhps_3d(
    config: &mut MhpsConfig,
    progress: &ProgressHandle,
    phase: &Arc<Mutex<String>>,
) -> anyhow::Result<MhpsResult> {
    use rayon::prelude::*;

    if config.time_step_dt <= 0.0 {
        config.time_step_dt = config.max_stable_dt();
    }
    let dt = config.time_step_dt;
    let total_steps = ((config.total_time_s / dt).ceil() as usize).max(1);

    let geo = &config.geometry;
    let nx = geo.nx;
    let ny = geo.ny;
    let nz = geo.nz;
    let hx2 = geo.hx().powi(2);
    let hy2 = geo.hy().powi(2);
    let hz2 = geo.hz().powi(2);
    let n = geo.total_nodes();
    let ny_nz = ny * nz;

    {
        let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
        *p = format!("3D stencil: {}x{}x{} = {} nodes", nx, ny, nz, n);
    }
    progress.set(3, 100);

    let mut t_cur = vec![config.t_initial_c; n];
    let mut t_new = vec![0.0f64; n];
    apply_mhps_bc_3d(&mut t_cur, config);

    let mut snapshots = Vec::new();
    let mut converged = false;
    let mut convergence_step = None;

    let save_field_at: HashSet<usize> = {
        let kf = [0, total_steps / 4, total_steps / 2, 3 * total_steps / 4, total_steps];
        kf.into_iter().collect()
    };

    snapshots.push(compute_mhps_snapshot_3d(&t_cur, config, 0, Some(t_cur.clone())));

    let sim_start = Instant::now();
    let use_parallel = n >= 10_000;

    // Pre-compute per-node material diffusivity for stencil
    let alpha_map: Vec<f64> = (0..n)
        .map(|idx| {
            let mat = &MATERIALS[geo.material_map[idx].min(MATERIALS.len() - 1)];
            mat.thermal_diffusivity()
        })
        .collect();

    for step in 1..=total_steps {
        // Explicit 3D stencil: T_new = T + dt*α*(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
        if use_parallel {
            // Parallelized over X-slices
            t_new.par_chunks_mut(ny_nz).enumerate().for_each(|(i, slice)| {
                for j in 0..ny {
                    for k in 0..nz {
                        let local = j * nz + k;
                        let global = i * ny_nz + local;

                        if geo.shape_mask.get(global).copied() != Some(true) {
                            slice[local] = t_cur[global];
                            continue;
                        }

                        let alpha = alpha_map[global];

                        let lap_x = if i > 0 && i < nx - 1 {
                            (t_cur[global - ny_nz] - 2.0 * t_cur[global] + t_cur[global + ny_nz]) / hx2
                        } else { 0.0 };

                        let lap_y = if j > 0 && j < ny - 1 {
                            (t_cur[global - nz] - 2.0 * t_cur[global] + t_cur[global + nz]) / hy2
                        } else { 0.0 };

                        let lap_z = if k > 0 && k < nz - 1 {
                            (t_cur[global - 1] - 2.0 * t_cur[global] + t_cur[global + 1]) / hz2
                        } else { 0.0 };

                        slice[local] = t_cur[global] + dt * alpha * (lap_x + lap_y + lap_z);
                    }
                }
            });
        } else {
            // Sequential fallback for small grids
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let global = i * ny_nz + j * nz + k;

                        if geo.shape_mask.get(global).copied() != Some(true) {
                            t_new[global] = t_cur[global];
                            continue;
                        }

                        let alpha = alpha_map[global];

                        let lap_x = if i > 0 && i < nx - 1 {
                            (t_cur[global - ny_nz] - 2.0 * t_cur[global] + t_cur[global + ny_nz]) / hx2
                        } else { 0.0 };

                        let lap_y = if j > 0 && j < ny - 1 {
                            (t_cur[global - nz] - 2.0 * t_cur[global] + t_cur[global + nz]) / hy2
                        } else { 0.0 };

                        let lap_z = if k > 0 && k < nz - 1 {
                            (t_cur[global - 1] - 2.0 * t_cur[global] + t_cur[global + 1]) / hz2
                        } else { 0.0 };

                        t_new[global] = t_cur[global] + dt * alpha * (lap_x + lap_y + lap_z);
                    }
                }
            }
        }

        std::mem::swap(&mut t_cur, &mut t_new);
        apply_mhps_bc_3d(&mut t_cur, config);

        // Convergence check
        if config.convergence_epsilon > 0.0 {
            let max_delta = t_cur.iter().zip(t_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_delta < config.convergence_epsilon {
                converged = true;
                convergence_step = Some(step);
            }
        }

        // Snapshot
        if step % config.save_every_n_steps == 0 || converged || step == total_steps {
            let field = if save_field_at.contains(&step) || converged || step == total_steps {
                Some(t_cur.clone())
            } else {
                None
            };
            snapshots.push(compute_mhps_snapshot_3d(&t_cur, config, step, field));
        }

        // Progress
        if step % 50 == 0 || converged || step == total_steps {
            let pct = ((step as f64 / total_steps as f64) * 90.0) as u32 + 5;
            progress.set(pct.min(95), 100);
            let snap = snapshots.last();
            let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
            *p = format!(
                "t={:.1}s  T_max={:.1}C  T_min={:.1}C{}",
                step as f64 * dt,
                snap.map(|s| s.max_temp_c).unwrap_or(0.0),
                snap.map(|s| s.min_temp_c).unwrap_or(0.0),
                if converged { "  [CONVERGED]" } else { "" }
            );
        }

        if converged { break; }
    }

    progress.set(100, 100);
    { let mut p = phase.lock().unwrap_or_else(|p| p.into_inner()); *p = "Done.".into(); }

    Ok(MhpsResult {
        config: config.clone(),
        snapshots,
        final_field: t_cur,
        computation_ms: sim_start.elapsed().as_secs_f64() * 1000.0,
        total_steps,
        converged,
        convergence_step,
    })
}

// ─── 3D Snapshot Computation ─────────────────────────────────────────────────

fn compute_mhps_snapshot_3d(
    t: &[f64],
    config: &MhpsConfig,
    step: usize,
    field: Option<Vec<f64>>,
) -> MhpsSnapshot {
    let geo = &config.geometry;
    let total = geo.total_nodes();

    // Active (non-void) temperatures
    let active_t: Vec<f64> = (0..total)
        .filter(|&idx| geo.shape_mask.get(idx).copied() == Some(true))
        .map(|idx| t[idx])
        .collect();

    let max_temp = active_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_temp = active_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_temp = active_t.iter().sum::<f64>() / active_t.len().max(1) as f64;

    // 3D gradient (central differences)
    let mut max_grad = 0.0f64;
    let ny_nz = geo.ny * geo.nz;
    if geo.nx > 2 && geo.ny > 2 {
        for i in 1..geo.nx - 1 {
            for j in 1..geo.ny - 1 {
                for k in 0..geo.nz {
                    let global = i * ny_nz + j * geo.nz + k;
                    if geo.shape_mask.get(global).copied() != Some(true) { continue; }

                    let gx = (t[global + ny_nz] - t[global - ny_nz]) / (2.0 * geo.hx());
                    let gy = (t[global + geo.nz] - t[global - geo.nz]) / (2.0 * geo.hy());
                    let gz = if geo.nz > 2 && k > 0 && k < geo.nz - 1 {
                        (t[global + 1] - t[global - 1]) / (2.0 * geo.hz())
                    } else {
                        0.0
                    };
                    max_grad = max_grad.max((gx * gx + gy * gy + gz * gz).sqrt());
                }
            }
        }
    }

    let source_temps: Vec<(String, f64)> = config.heat_sources.iter().map(|src| {
        let idx = geo.idx3(
            src.position_i.min(geo.nx.saturating_sub(1)),
            src.position_j.min(geo.ny.saturating_sub(1)),
            src.position_k.min(geo.nz.saturating_sub(1)),
        );
        let temp = if idx < t.len() && geo.shape_mask.get(idx).copied() == Some(true) {
            t[idx]
        } else {
            config.t_ambient_c
        };
        (src.name.clone(), temp)
    }).collect();

    MhpsSnapshot {
        time_s: step as f64 * config.time_step_dt,
        step,
        max_temp_c: max_temp,
        min_temp_c: min_temp,
        mean_temp_c: mean_temp,
        max_gradient: max_grad,
        source_temps,
        field,
    }
}

// ─── Snapshot ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MhpsSnapshot {
    pub time_s: f64,
    pub step: usize,
    pub max_temp_c: f64,
    pub min_temp_c: f64,
    pub mean_temp_c: f64,
    pub max_gradient: f64,
    pub source_temps: Vec<(String, f64)>,
    pub field: Option<Vec<f64>>,
}

#[derive(Clone)]
pub struct MhpsResult {
    pub config: MhpsConfig,
    pub snapshots: Vec<MhpsSnapshot>,
    pub final_field: Vec<f64>,
    pub computation_ms: f64,
    pub total_steps: usize,
    pub converged: bool,
    pub convergence_step: Option<usize>,
}

// ─── Main Solver ─────────────────────────────────────────────────────────────

pub fn run_mhps(
    config: &mut MhpsConfig,
    progress: &ProgressHandle,
    phase: &Arc<Mutex<String>>,
) -> anyhow::Result<MhpsResult> {
    // Dispatch: 3D stencil for nz > 1, original SpMV for 2D
    if config.geometry.nz > 1 {
        return run_mhps_3d(config, progress, phase);
    }

    // Auto-calculate dt
    if config.time_step_dt <= 0.0 {
        config.time_step_dt = config.max_stable_dt();
    }

    let total_steps = ((config.total_time_s / config.time_step_dt).ceil() as usize).max(1);

    // Build transition matrix
    {
        let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
        *p = format!(
            "Building {}x{} transition matrix...",
            config.total_nodes(),
            config.total_nodes()
        );
    }
    progress.set(3, 100);

    let a = build_mhps_matrix(config, config.time_step_dt);

    // Initialize temperature field
    let mut t = vec![config.t_initial_c; config.total_nodes()];
    apply_mhps_bc(&mut t, config);

    let mut snapshots = Vec::new();
    let mut converged = false;
    let mut convergence_step = None;

    // Keyframes for storing full field snapshots
    let save_field_at: HashSet<usize> = {
        let kf = [0, total_steps / 4, total_steps / 2, 3 * total_steps / 4, total_steps];
        kf.into_iter().collect()
    };

    // Save initial snapshot
    snapshots.push(compute_mhps_snapshot(&t, config, 0, Some(t.clone())));

    let sim_start = Instant::now();

    for step in 1..=total_steps {
        let t_prev = t.clone();

        // T_new = A * T_old
        t = a.spmv(&t);

        // Apply boundary conditions
        apply_mhps_bc(&mut t, config);

        // Check convergence
        if config.convergence_epsilon > 0.0 {
            let max_delta = t
                .iter()
                .zip(t_prev.iter())
                .map(|(tn, to)| (tn - to).abs())
                .fold(0.0_f64, f64::max);
            if max_delta < config.convergence_epsilon {
                converged = true;
                convergence_step = Some(step);
            }
        }

        // Snapshot
        if step % config.save_every_n_steps == 0 || converged || step == total_steps {
            let field = if save_field_at.contains(&step) || converged || step == total_steps {
                Some(t.clone())
            } else {
                None
            };
            snapshots.push(compute_mhps_snapshot(&t, config, step, field));
        }

        // Progress reporting
        if step % 50 == 0 || converged || step == total_steps {
            let pct = ((step as f64 / total_steps as f64) * 90.0) as u32 + 5;
            progress.set(pct.min(95), 100);
            let snap = snapshots.last();
            let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
            *p = format!(
                "t={:.1}s  T_max={:.1}C  T_min={:.1}C{}",
                step as f64 * config.time_step_dt,
                snap.map(|s| s.max_temp_c).unwrap_or(0.0),
                snap.map(|s| s.min_temp_c).unwrap_or(0.0),
                if converged { "  [CONVERGED]" } else { "" }
            );
        }

        if converged {
            break;
        }
    }

    progress.set(100, 100);
    {
        let mut p = phase.lock().unwrap_or_else(|p| p.into_inner());
        *p = "Done.".into();
    }

    Ok(MhpsResult {
        config: config.clone(),
        snapshots,
        final_field: t,
        computation_ms: sim_start.elapsed().as_secs_f64() * 1000.0,
        total_steps,
        converged,
        convergence_step,
    })
}

fn compute_mhps_snapshot(
    t: &[f64],
    config: &MhpsConfig,
    step: usize,
    field: Option<Vec<f64>>,
) -> MhpsSnapshot {
    let geo = &config.geometry;
    let active_t: Vec<f64> = (0..geo.nx * geo.ny)
        .filter(|&idx| !geo.is_void(idx / geo.ny, idx % geo.ny))
        .map(|idx| t[idx])
        .collect();

    let max_temp = active_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_temp = active_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_temp = active_t.iter().sum::<f64>() / active_t.len().max(1) as f64;

    // Max gradient (central differences)
    let mut max_grad = 0.0f64;
    if geo.nx > 2 && geo.ny > 2 {
        for i in 1..geo.nx - 1 {
            for j in 1..geo.ny - 1 {
                if !geo.is_void(i, j) {
                    let gx = (t[geo.idx(i + 1, j)] - t[geo.idx(i - 1, j)]) / (2.0 * geo.hx());
                    let gy = (t[geo.idx(i, j + 1)] - t[geo.idx(i, j - 1)]) / (2.0 * geo.hy());
                    max_grad = max_grad.max((gx * gx + gy * gy).sqrt());
                }
            }
        }
    }

    // Temperature at each source
    let source_temps: Vec<(String, f64)> = config
        .heat_sources
        .iter()
        .map(|src| {
            let temp = if src.position_i < geo.nx
                && src.position_j < geo.ny
                && !geo.is_void(src.position_i, src.position_j)
            {
                t[geo.idx(src.position_i, src.position_j)]
            } else {
                config.t_ambient_c
            };
            (src.name.clone(), temp)
        })
        .collect();

    MhpsSnapshot {
        time_s: step as f64 * config.time_step_dt,
        step,
        max_temp_c: max_temp,
        min_temp_c: min_temp,
        mean_temp_c: mean_temp,
        max_gradient: max_grad,
        source_temps,
        field,
    }
}

// ─── CSV Export ──────────────────────────────────────────────────────────────

pub fn export_mhps_history_csv(result: &MhpsResult, path: &str) -> anyhow::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;

    // Header
    let mut headers = vec![
        "time_s".to_string(),
        "max_temp_C".to_string(),
        "min_temp_C".to_string(),
        "mean_temp_C".to_string(),
        "max_gradient_K_m".to_string(),
    ];
    for src in &result.config.heat_sources {
        headers.push(format!("T_{}", src.name));
    }
    wtr.write_record(&headers)?;

    for snap in &result.snapshots {
        let mut row = vec![
            format!("{:.4}", snap.time_s),
            format!("{:.2}", snap.max_temp_c),
            format!("{:.2}", snap.min_temp_c),
            format!("{:.2}", snap.mean_temp_c),
            format!("{:.1}", snap.max_gradient),
        ];
        for (_, temp) in &snap.source_temps {
            row.push(format!("{:.2}", temp));
        }
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn export_mhps_field_csv(result: &MhpsResult, path: &str) -> anyhow::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    let geo = &result.config.geometry;

    if geo.nz > 1 {
        // 3D export with z coordinate
        wtr.write_record(["x_m", "y_m", "z_m", "temperature_C", "material"])?;
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                for k in 0..geo.nz {
                    let x = i as f64 * geo.hx();
                    let y = j as f64 * geo.hy();
                    let z = k as f64 * geo.hz();
                    let idx = geo.idx3(i, j, k);
                    if idx >= result.final_field.len() { continue; }
                    let mat_idx = geo.material_map.get(idx).copied().unwrap_or(0);
                    let mat = &MATERIALS[mat_idx.min(MATERIALS.len() - 1)];
                    wtr.write_record(&[
                        format!("{:.6}", x),
                        format!("{:.6}", y),
                        format!("{:.6}", z),
                        format!("{:.4}", result.final_field[idx]),
                        mat.name.to_string(),
                    ])?;
                }
            }
        }
    } else {
        // 2D export (backward compatible)
        wtr.write_record(["x_m", "y_m", "temperature_C", "material", "thickness_m"])?;
        for i in 0..geo.nx {
            for j in 0..geo.ny {
                let x = i as f64 * geo.hx();
                let y = j as f64 * geo.hy();
                let idx = geo.idx(i, j);
                if idx >= result.final_field.len() { continue; }
                let mat_idx = geo.material_map.get(idx).copied().unwrap_or(0);
                let mat = &MATERIALS[mat_idx.min(MATERIALS.len() - 1)];
                wtr.write_record(&[
                    format!("{:.6}", x),
                    format!("{:.6}", y),
                    format!("{:.4}", result.final_field[idx]),
                    mat.name.to_string(),
                    format!("{:.6}", geo.thickness_map.get(idx).copied().unwrap_or(0.0)),
                ])?;
            }
        }
    }
    wtr.flush()?;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_material_reaches_steady_state() {
        // Aluminum plate: heater on the left, cooler on the right.
        // Steady state should produce a roughly linear gradient.
        let mut config = MhpsConfig {
            geometry: MhpsGeometry::uniform(0.1, 0.05, 0.003, 20, 10, 1), // Aluminum
            heat_sources: vec![
                HeatSource::new_heater("H1", 0, 5, 1, 100.0),
                HeatSource::new_cooler("C1", 19, 5, 1, 20.0),
            ],
            t_initial_c: 20.0,
            t_ambient_c: 20.0,
            convection_h: 0.0,
            time_step_dt: 0.0,
            total_time_s: 120.0,
            save_every_n_steps: 50,
            convergence_epsilon: 0.01,
        };
        let progress = ProgressHandle::new(100);
        let phase = Arc::new(Mutex::new(String::new()));
        let result = run_mhps(&mut config, &progress, &phase).unwrap();
        assert!(result.converged, "Should converge to steady state");
        // Center node (10,5) should be roughly 60C (linear gradient 100->20)
        let t_center = result.final_field[config.geometry.idx(10, 5)];
        assert!(
            (t_center - 60.0).abs() < 10.0,
            "Expected ~60C at center, got {:.1}",
            t_center
        );
    }

    #[test]
    fn test_void_nodes_stay_at_ambient() {
        let mut geo = MhpsGeometry::uniform(0.1, 0.1, 0.005, 20, 20, 0); // Copper
        geo.drill_hole(10, 10, 3);
        let mut config = MhpsConfig {
            geometry: geo,
            heat_sources: vec![HeatSource::new_heater("H1", 2, 10, 2, 200.0)],
            t_initial_c: 20.0,
            t_ambient_c: 20.0,
            convection_h: 0.0,
            time_step_dt: 0.0,
            total_time_s: 5.0,
            save_every_n_steps: 10,
            convergence_epsilon: 0.0,
        };
        let progress = ProgressHandle::new(100);
        let phase = Arc::new(Mutex::new(String::new()));
        let result = run_mhps(&mut config, &progress, &phase).unwrap();
        // Void nodes must remain at ambient
        for i in 8..13 {
            for j in 8..13 {
                if config.geometry.is_void(i, j) {
                    let t = result.final_field[config.geometry.idx(i, j)];
                    assert!(
                        (t - 20.0).abs() < 0.01,
                        "Void node ({},{}) = {} != 20",
                        i,
                        j,
                        t
                    );
                }
            }
        }
    }

    #[test]
    fn test_material_library_diffusivity() {
        // Copper should have much higher diffusivity than wood
        let copper = &MATERIALS[0];
        let wood = &MATERIALS[7];
        assert!(
            copper.thermal_diffusivity() > wood.thermal_diffusivity() * 100.0,
            "Copper alpha={:.2e} should be >> Wood alpha={:.2e}",
            copper.thermal_diffusivity(),
            wood.thermal_diffusivity()
        );
    }

    #[test]
    fn test_geometry_drill_hole() {
        let mut geo = MhpsGeometry::uniform(0.1, 0.1, 0.005, 20, 20, 0);
        geo.drill_hole(10, 10, 2);
        // Center should be void
        assert!(geo.is_void(10, 10));
        // Corner at (0,0) should not be void
        assert!(!geo.is_void(0, 0));
    }

    #[test]
    fn test_no_heat_source_limit() {
        let geo = MhpsGeometry::uniform(0.1, 0.1, 0.005, 100, 100, 0);
        let mut config = MhpsConfig {
            geometry: geo,
            heat_sources: Vec::new(),
            t_initial_c: 20.0,
            t_ambient_c: 20.0,
            convection_h: 0.0,
            time_step_dt: 0.0,
            total_time_s: 1.0,
            save_every_n_steps: 10,
            convergence_epsilon: 0.0,
        };
        let nx = config.geometry.nx;
        for i in 0..50 {
            add_heat_source(
                &mut config,
                HeatSource::new_heater(&format!("H{}", i), i % nx, 5, 0, 100.0),
            )
            .unwrap();
        }
        assert_eq!(config.heat_sources.len(), 50);
    }

    #[test]
    fn test_polygon_point_in_polygon() {
        // L-shaped polygon
        let verts = vec![
            (0.0, 0.0),
            (0.2, 0.0),
            (0.2, 0.1),
            (0.1, 0.1),
            (0.1, 0.2),
            (0.0, 0.2),
        ];
        assert!(point_in_polygon(0.05, 0.05, &verts));  // bottom-left — inside
        assert!(point_in_polygon(0.05, 0.15, &verts));  // top-left — inside
        assert!(!point_in_polygon(0.15, 0.15, &verts)); // top-right — outside (cutout)
        assert!(point_in_polygon(0.15, 0.05, &verts));  // bottom-right — inside
    }

    #[test]
    fn test_geometry_mask_l_shape() {
        let mut geo = MhpsGeometry {
            length_x: 0.2,
            length_y: 0.15,
            length_z: 0.005,
            base_thickness: 0.005,
            nx: 20,
            ny: 15,
            nz: 1,
            thickness_map: vec![0.005; 300],
            material_map: vec![1; 300],
            shape_mask: vec![true; 300],
            shape_definition: ShapeDefinition::LShape {
                cutout_x: 0.1,
                cutout_y: 0.075,
            },
        };
        geo.rebuild_mask();

        // Top-right corner (19, 14) should be in the cutout (void)
        assert!(
            geo.is_void(19, 14),
            "Top-right corner should be void in L-shape"
        );
        // Bottom-left corner (0, 0) should be active
        assert!(!geo.is_void(0, 0), "Bottom-left corner should be active");
    }
}
