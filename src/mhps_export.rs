// =============================================================================
//  MHPS Export Bundle — history CSV, field CSV, simulation log, Python script
// =============================================================================

use crate::mhps::{MhpsResult, MATERIALS};

pub struct MhpsExportBundle {
    pub base_name: String,
}

impl MhpsExportBundle {
    pub fn new() -> Self {
        let ts = crate::io::timestamp_now().replace(':', "-");
        Self {
            base_name: format!("mhps_{}", ts),
        }
    }

    /// Export snapshot history as CSV.
    pub fn export_history_csv(&self, result: &MhpsResult, dir: &str) -> anyhow::Result<String> {
        let path = format!("{}/{}_history.csv", dir, self.base_name);
        crate::mhps::export_mhps_history_csv(result, &path)?;
        Ok(path)
    }

    /// Export final 2D temperature field as CSV.
    pub fn export_final_field_csv(&self, result: &MhpsResult, dir: &str) -> anyhow::Result<String> {
        let path = format!("{}/{}_final_field.csv", dir, self.base_name);
        crate::mhps::export_mhps_field_csv(result, &path)?;
        Ok(path)
    }

    /// Export a plain-text simulation log.
    pub fn export_simulation_log(&self, result: &MhpsResult, dir: &str) -> anyhow::Result<String> {
        let path = format!("{}/{}_simulation.log", dir, self.base_name);

        let config = &result.config;
        let geo = &config.geometry;
        let mat_idx = geo.material_map.get(0).copied().unwrap_or(0);
        let mat = &MATERIALS[mat_idx.min(MATERIALS.len() - 1)];

        let last = result.snapshots.last();
        let max_t = last.map(|s| s.max_temp_c).unwrap_or(0.0);
        let min_t = last.map(|s| s.min_temp_c).unwrap_or(0.0);
        let max_grad = last.map(|s| s.max_gradient).unwrap_or(0.0);

        let mut log = String::new();
        log.push_str("FLUST MHPS Simulation Log\n");
        log.push_str(&format!("Generated: {}\n\n", self.base_name));

        log.push_str("=== Configuration ===\n");
        log.push_str(&format!("Material:       {} (lambda={:.1} W/mK)\n", mat.name, mat.thermal_conductivity));
        log.push_str(&format!("Geometry:       {:.1}x{:.1}x{:.1} mm\n",
            geo.length_x * 1000.0, geo.length_y * 1000.0, geo.length_z * 1000.0));
        log.push_str(&format!("Grid:           {}x{}x{} = {} nodes\n",
            geo.nx, geo.ny, geo.nz, geo.total_nodes()));
        log.push_str(&format!("T_initial:      {:.1} C\n", config.t_initial_c));
        log.push_str(&format!("T_ambient:      {:.1} C\n", config.t_ambient_c));
        log.push_str(&format!("Convection h:   {:.1} W/m2K\n", config.convection_h));
        log.push_str(&format!("dt:             {:.6} s\n", config.time_step_dt));
        log.push_str(&format!("Total time:     {:.1} s\n", config.total_time_s));
        log.push_str(&format!("Epsilon:        {:.2e}\n\n", config.convergence_epsilon));

        log.push_str(&format!("Heat sources:   {}\n", config.heat_sources.len()));
        for (i, src) in config.heat_sources.iter().enumerate() {
            log.push_str(&format!("  #{}: {} at ({},{}) T={:.1}C r={}\n",
                i + 1, src.name, src.position_i, src.position_j,
                src.temperature_c, src.radius_nodes));
        }
        log.push_str("\n");

        log.push_str("=== Results ===\n");
        log.push_str(&format!("Converged:      {}\n", if result.converged { "YES" } else { "NO" }));
        if let Some(step) = result.convergence_step {
            log.push_str(&format!("Conv. step:     {} (t={:.1}s)\n",
                step, step as f64 * config.time_step_dt));
        }
        log.push_str(&format!("Total steps:    {}\n", result.total_steps));
        log.push_str(&format!("Compute time:   {:.1} ms\n", result.computation_ms));
        log.push_str(&format!("T_max final:    {:.2} C\n", max_t));
        log.push_str(&format!("T_min final:    {:.2} C\n", min_t));
        log.push_str(&format!("dT:             {:.2} C\n", max_t - min_t));
        log.push_str(&format!("Max gradient:   {:.1} K/m\n\n", max_grad));

        log.push_str("=== Snapshot History ===\n");
        log.push_str(&format!("{:<10} {:>10} {:>10} {:>10} {:>12}\n",
            "time_s", "T_max", "T_min", "T_mean", "max_grad"));
        for snap in &result.snapshots {
            log.push_str(&format!("{:<10.3} {:>10.2} {:>10.2} {:>10.2} {:>12.1}\n",
                snap.time_s, snap.max_temp_c, snap.min_temp_c,
                snap.mean_temp_c, snap.max_gradient));
        }

        std::fs::write(&path, &log)?;
        Ok(path)
    }

    /// Export a Python/matplotlib visualization script.
    pub fn export_python_script(&self, result: &MhpsResult, dir: &str) -> anyhow::Result<String> {
        let path = format!("{}/{}_visualize.py", dir, self.base_name);
        let field_csv = format!("{}_final_field.csv", self.base_name);
        let history_csv = format!("{}_history.csv", self.base_name);
        let geo = &result.config.geometry;

        let is_3d = geo.nz > 1;
        let script = format!(r#"""
FLUST MHPS Visualization Script
Generated by Flust — Material Heat Propagation Simulator
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load final field ---
field_path = os.path.join(script_dir, "{field_csv}")
x, y, z, temp = [], [], [], []
is_3d = {is_3d}
with open(field_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        x.append(float(row['x_m']))
        y.append(float(row['y_m']))
        if 'z_m' in row:
            z.append(float(row['z_m']))
        temp.append(float(row['temperature_C']))

nx, ny, nz = {nx}, {ny}, {nz}

if is_3d:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Mid-Z slice heatmap
    ax1 = axes[0]
    mid_k = nz // 2
    start = mid_k
    indices = list(range(start, len(temp), nz))[:nx * ny]
    T_slice = np.array([temp[i] for i in indices]).reshape(nx, ny) if len(indices) == nx * ny else np.zeros((nx, ny))
    X = np.array([x[i] for i in indices]).reshape(nx, ny) * 1000
    Y = np.array([y[i] for i in indices]).reshape(nx, ny) * 1000
    im = ax1.pcolormesh(X, Y, T_slice, cmap='hot', shading='auto')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title(f'Temperature at Z-slice k={{mid_k}} [C]')
    ax1.set_aspect('equal')
    plt.colorbar(im, ax=ax1, label='T [C]')
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    X = np.array(x).reshape(nx, ny) * 1000
    Y = np.array(y).reshape(nx, ny) * 1000
    T = np.array(temp).reshape(nx, ny)
    ax1 = axes[0]
    im = ax1.pcolormesh(X, Y, T, cmap='hot', shading='auto')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title('Temperature Field [C]')
    ax1.set_aspect('equal')
    plt.colorbar(im, ax=ax1, label='T [C]')

# Source markers
sources = {sources_repr}
for name, xi, yj, tc in sources:
    color = 'red' if tc > {t_init} else 'blue'
    ax1.plot(xi * 1000, yj * 1000, 'o', color=color, markersize=8)
    ax1.annotate(name, (xi * 1000, yj * 1000), fontsize=7, ha='center', va='bottom')

# Timeline from history CSV
hist_path = os.path.join(script_dir, "{history_csv}")
t_time, t_max, t_min, t_mean = [], [], [], []
with open(hist_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_time.append(float(row['time_s']))
        t_max.append(float(row['max_temp_C']))
        t_min.append(float(row['min_temp_C']))
        t_mean.append(float(row['mean_temp_C']))

ax2 = axes[1]
ax2.plot(t_time, t_max, 'r-', label='T_max')
ax2.plot(t_time, t_mean, 'y--', label='T_mean')
ax2.plot(t_time, t_min, 'b-', label='T_min')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Temperature [C]')
ax2.set_title('Temperature Timeline')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '{base}_plot.png'), dpi=150)
plt.show()
"#,
            field_csv = field_csv,
            history_csv = history_csv,
            is_3d = if is_3d { "True" } else { "False" },
            nx = geo.nx,
            ny = geo.ny,
            nz = geo.nz,
            t_init = result.config.t_initial_c,
            sources_repr = format_sources_python(&result.config.heat_sources, geo),
            base = self.base_name,
        );

        std::fs::write(&path, script)?;
        Ok(path)
    }

    /// Export all files. Returns list of (description, result).
    pub fn export_all(&self, result: &MhpsResult, dir: &str) -> Vec<(&'static str, anyhow::Result<String>)> {
        vec![
            ("History CSV", self.export_history_csv(result, dir)),
            ("Field CSV", self.export_final_field_csv(result, dir)),
            ("Simulation log", self.export_simulation_log(result, dir)),
            ("Python script", self.export_python_script(result, dir)),
        ]
    }
}

fn format_sources_python(
    sources: &[crate::mhps::HeatSource],
    geo: &crate::mhps::MhpsGeometry,
) -> String {
    let mut parts = Vec::new();
    for src in sources {
        let x = src.position_i as f64 * geo.hx();
        let y = src.position_j as f64 * geo.hy();
        parts.push(format!(
            "('{}', {:.6}, {:.6}, {:.1})",
            src.name, x, y, src.temperature_c
        ));
    }
    format!("[{}]", parts.join(", "))
}
