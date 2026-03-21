// ─── Thermal Simulation CSV Export ─────────────────────────────────────────
//
// Exports simulation results in CSV format compatible with
// Excel, Python/pandas, and MATLAB.
// Header block is a human-readable engineering report (comment lines).

use crate::thermal::ThermalSimResult;

/// Save the time-series of snapshots to CSV.
/// One row per snapshot with all physical metrics.
/// Includes a rich engineering report header for human readability.
pub fn export_snapshots_csv(
    result: &ThermalSimResult,
    path: &str,
) -> anyhow::Result<()> {
    let cfg = &result.config;
    let alpha = cfg.fluid.thermal_diffusivity();
    let total_nodes = cfg.total_nodes();
    let spmv_per_sec = if result.computation_ms > 0.0 {
        (result.total_matrix_multiplications as f64 / (result.computation_ms / 1000.0)) as usize
    } else {
        0
    };

    let mut c = String::with_capacity(4096);

    // ── Rich engineering report header ──
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");
    c.push_str("#  FLUST THERMAL SIMULATION REPORT\n");
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");
    c.push_str("#\n");

    // Configuration
    c.push_str("#  CONFIGURATION\n");
    c.push_str(&format!(
        "#    Fluid:        {} (\u{03bb}={:.3} W/m\u{00b7}K, \u{03c1}={:.0} kg/m\u{00b3}, c={:.0} J/kg\u{00b7}K)\n",
        cfg.fluid.name,
        cfg.fluid.thermal_conductivity,
        cfg.fluid.density,
        cfg.fluid.specific_heat,
    ));
    c.push_str(&format!(
        "#    Diffusivity:   \u{03b1} = {:.3e} m\u{00b2}/s\n",
        alpha,
    ));
    c.push_str(&format!(
        "#    Reservoir:     {:.1} x {:.1} x {:.1} cm\n",
        cfg.length_x * 100.0,
        cfg.length_y * 100.0,
        cfg.length_z * 100.0,
    ));
    c.push_str(&format!(
        "#    Grid:          {}x{}x{} = {} nodes\n",
        cfg.nx, cfg.ny, cfg.nz, total_nodes,
    ));
    c.push_str(&format!(
        "#    T_initial:     {:.1}\u{00b0}C\n",
        cfg.t_initial,
    ));
    c.push_str(&format!(
        "#    T_boundary:    {:.1}\u{00b0}C\n",
        cfg.t_boundary,
    ));
    c.push_str(&format!(
        "#    dt:            {:.6}s  (stability limit: {:.6}s)\n",
        cfg.time_step_dt,
        cfg.max_stable_dt(alpha),
    ));
    c.push_str(&format!(
        "#    Steps:         {}  (simulated {:.0}s = {:.1}min)\n",
        cfg.total_steps,
        cfg.total_steps as f64 * cfg.time_step_dt,
        cfg.total_steps as f64 * cfg.time_step_dt / 60.0,
    ));
    c.push_str("#\n");

    // TEG
    c.push_str("#  TEG (Thermoelectric Generator)\n");
    c.push_str(&format!(
        "#    Seebeck:       S = {:.0} mV/K\n",
        cfg.teg.seebeck_coefficient * 1000.0,
    ));
    c.push_str(&format!(
        "#    R_internal:    {:.1} \u{03a9},  R_load: {:.1} \u{03a9}\n",
        cfg.teg.internal_resistance,
        cfg.teg.load_resistance,
    ));
    c.push_str(&format!(
        "#    Contact area:  {:.0}x{:.0} mm,  thickness: {:.0} mm\n",
        (cfg.teg.teg_area * 1000.0).sqrt() * 1000.0 * 0.4,  // approximate dims
        (cfg.teg.teg_area * 1000.0).sqrt() * 1000.0 / 0.4,
        cfg.teg.teg_thickness * 1000.0,
    ));
    c.push_str("#\n");

    // Results summary
    c.push_str("#  RESULTS SUMMARY\n");
    c.push_str(&format!(
        "#    Computation:   {:.1} ms  ({} SpMV/s)\n",
        result.computation_ms, spmv_per_sec,
    ));

    if let Some(first) = result.snapshots.first() {
        c.push_str(&format!(
            "#    Peak voltage:  {:.2} V  at t=0s\n",
            first.voltage,
        ));
    }
    c.push_str(&format!(
        "#    Peak power:    {:.0} mW\n",
        result.max_power_mw,
    ));
    c.push_str(&format!(
        "#    Motor runtime: {:.1} min  (threshold: {:.1} V)\n",
        result.runtime_minutes,
        result.threshold_voltage,
    ));
    c.push_str(&format!(
        "#    Avg power:     {:.0} mW\n",
        result.average_power_mw,
    ));
    c.push_str(&format!(
        "#    Total energy:  {:.0} mJ = {:.2} J\n",
        result.total_energy_mj,
        result.total_energy_mj / 1000.0,
    ));

    if let Some(first) = result.snapshots.first() {
        c.push_str(&format!(
            "#    TEG eff.:      ~{:.1}%\n",
            first.efficiency_pct,
        ));
    }
    c.push_str("#\n");
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");
    c.push_str("#  TIME-SERIES DATA\n");
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");

    // CSV header
    c.push_str(
        "time_s,step,t_center_C,t_teg_hot_C,delta_t_K,\
         voltage_V,current_A,power_W,power_mW,\
         efficiency_pct,mean_temp_C,max_temp_C,min_temp_C\n",
    );

    // Data rows
    for snap in &result.snapshots {
        c.push_str(&format!(
            "{:.3},{},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            snap.time_s,
            snap.step,
            snap.t_center,
            snap.t_teg_hot,
            snap.delta_t,
            snap.voltage,
            snap.current,
            snap.power_w,
            snap.power_mw,
            snap.efficiency_pct,
            snap.mean_temp,
            snap.max_temp,
            snap.min_temp,
        ));
    }

    std::fs::write(path, &c)?;
    Ok(())
}

/// Export the final 3D temperature field to CSV.
/// Each row: x_m, y_m, z_m, temperature_C.
/// Suitable for visualization in Python Matplotlib or ParaView.
pub fn export_final_field_csv(
    result: &ThermalSimResult,
    path: &str,
) -> anyhow::Result<()> {
    let cfg = &result.config;
    let t = &result.final_field;
    let total_nodes = cfg.total_nodes();
    let sim_time = cfg.total_steps as f64 * cfg.time_step_dt;

    let mut c = String::with_capacity(total_nodes * 40);

    // Rich header
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");
    c.push_str("#  FLUST FINAL TEMPERATURE FIELD\n");
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");
    c.push_str("#\n");
    c.push_str(&format!(
        "#  Fluid:       {}\n",
        cfg.fluid.name,
    ));
    c.push_str(&format!(
        "#  Reservoir:   {:.1} x {:.1} x {:.1} cm\n",
        cfg.length_x * 100.0,
        cfg.length_y * 100.0,
        cfg.length_z * 100.0,
    ));
    c.push_str(&format!(
        "#  Grid:        {}x{}x{} = {} nodes\n",
        cfg.nx, cfg.ny, cfg.nz, total_nodes,
    ));
    c.push_str(&format!(
        "#  T_initial:   {:.1}\u{00b0}C,  T_boundary: {:.1}\u{00b0}C\n",
        cfg.t_initial, cfg.t_boundary,
    ));
    c.push_str(&format!(
        "#  Snapshot at: t = {:.1}s  (after {} steps)\n",
        sim_time, cfg.total_steps,
    ));

    if let Some(last) = result.snapshots.last() {
        c.push_str(&format!(
            "#  T_center:    {:.1}\u{00b0}C,  T_mean: {:.1}\u{00b0}C\n",
            last.t_center, last.mean_temp,
        ));
    }
    c.push_str("#\n");
    c.push_str("# ═══════════════════════════════════════════════════════════════\n");

    c.push_str("x_m,y_m,z_m,temperature_C\n");

    for i in 0..cfg.nx {
        for j in 0..cfg.ny {
            for k in 0..cfg.nz {
                let x = i as f64 * cfg.hx();
                let y = j as f64 * cfg.hy();
                let z = k as f64 * cfg.hz();
                let temp = t[cfg.linear_index(i, j, k)];
                c.push_str(&format!("{:.4},{:.4},{:.4},{:.6}\n", x, y, z, temp));
            }
        }
    }

    std::fs::write(path, &c)?;
    Ok(())
}
