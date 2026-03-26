// =============================================================================
//  COMPUTER VISION — IR PROCESSING (im2col + Bicubic + Convolution via GEMM)
// =============================================================================
//
//  Mathematical basis:
//    1. Bicubic interpolation: upscale low-res IR sensor data (e.g., 8x8)
//       to high-res using cubic Hermite splines. Each output pixel is a
//       weighted sum of 4x4 neighborhood in the source.
//
//    2. Convolution via im2col + GEMM:
//       Standard 2D convolution y = conv(x, k) is reshaped into a matrix
//       multiply by extracting image patches into columns (im2col transform),
//       then multiplying the filter weight matrix by the patch matrix:
//         Y = W * im2col(X)
//
//    3. Denoising pipeline: upscale → convolve with smoothing kernel → output.
//
//  Uses Flust HPC multiply_hpc_fused for the GEMM step.
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use crate::algorithms;
use crate::matrix::Matrix;

/// Configuration for an IR vision processing pipeline.
#[derive(Clone)]
pub struct VisionConfig {
    /// Input IR sensor resolution (rows).
    pub input_rows: usize,
    /// Input IR sensor resolution (cols).
    pub input_cols: usize,
    /// Upscale factor (e.g., 4 → 8x8 becomes 32x32).
    pub upscale_factor: usize,
    /// Convolution kernel type.
    pub kernel_type: KernelType,
    /// Convolution kernel size (must be odd: 3, 5, 7).
    pub kernel_size: usize,
    /// Random seed for synthetic IR data (None = entropy).
    pub seed: Option<u64>,
    /// Noise level added to synthetic data [0, 1).
    pub noise_level: f64,
}

impl VisionConfig {
    /// Estimate RAM usage in MB.
    pub fn estimate_memory_mb(&self) -> f64 {
        let hr = self.input_rows * self.upscale_factor;
        let hc = self.input_cols * self.upscale_factor;
        let ks = self.kernel_size;
        // upscaled + im2col patches + kernel matrix + output
        let patch_cols = (hr - ks + 1) * (hc - ks + 1);
        let patch_rows = ks * ks;
        let bytes = (hr * hc + patch_rows * patch_cols + patch_cols + hr * hc) * 8;
        bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Available convolution kernels.
#[derive(Clone, Copy, PartialEq)]
pub enum KernelType {
    Gaussian,
    Sharpen,
    EdgeDetect,
    BoxBlur,
}

impl KernelType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Gaussian => "Gaussian Blur",
            Self::Sharpen => "Sharpen",
            Self::EdgeDetect => "Edge Detect (Laplacian)",
            Self::BoxBlur => "Box Blur (Mean)",
        }
    }

    pub fn all() -> &'static [KernelType] {
        &[Self::Gaussian, Self::Sharpen, Self::EdgeDetect, Self::BoxBlur]
    }
}

/// Snapshot of pipeline progress.
#[derive(Clone)]
pub struct VisionSnapshot {
    pub stage: String,
    pub progress_frac: f64,
}

/// Final result of vision processing pipeline.
#[derive(Clone)]
pub struct VisionResult {
    pub config: VisionConfig,
    /// Original low-res input (input_rows x input_cols).
    pub input_matrix: Vec<f64>,
    /// Upscaled via bicubic interpolation.
    pub upscaled_matrix: Vec<f64>,
    pub upscaled_rows: usize,
    pub upscaled_cols: usize,
    /// Convolved output after kernel application.
    pub output_matrix: Vec<f64>,
    pub output_rows: usize,
    pub output_cols: usize,
    /// im2col patch matrix dimensions (for diagnostics).
    pub im2col_rows: usize,
    pub im2col_cols: usize,
    /// Computation wall-clock time in milliseconds.
    pub computation_ms: f64,
    /// Time breakdown.
    pub upscale_ms: f64,
    pub im2col_ms: f64,
    pub gemm_ms: f64,
    /// Statistics.
    pub input_min: f64,
    pub input_max: f64,
    pub output_min: f64,
    pub output_max: f64,
    pub snr_estimate: f64,
}

// --- Synthetic IR Data Generation -------------------------------------------

/// Generate synthetic IR sensor data with hotspots and noise.
pub fn generate_ir_data(rows: usize, cols: usize, noise: f64, seed: Option<u64>) -> Vec<f64> {
    let mut rng_state: u64 = seed.unwrap_or(12345);
    let mut data = vec![0.0f64; rows * cols];

    let cx = cols as f64 / 2.0;
    let cy = rows as f64 / 2.0;
    let sigma = (rows.min(cols) as f64) / 3.0;

    for i in 0..rows {
        for j in 0..cols {
            let dx = j as f64 - cx;
            let dy = i as f64 - cy;
            // Gaussian hotspot centered in the frame
            let base = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();

            // Add secondary hotspot at offset
            let dx2 = j as f64 - cx * 0.3;
            let dy2 = i as f64 - cy * 0.7;
            let spot2 = 0.6 * (-((dx2 * dx2 + dy2 * dy2) / (1.5 * sigma * sigma))).exp();

            // Noise
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise_val = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise;

            data[i * cols + j] = (base + spot2 + noise_val).clamp(0.0, 1.0);
        }
    }
    data
}

// --- Bicubic Interpolation --------------------------------------------------

/// Cubic Hermite basis function for bicubic interpolation.
fn cubic_hermite(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2,
    ]
}

/// Bicubic interpolation: upscale by integer factor.
pub fn bicubic_upscale(
    data: &[f64],
    rows: usize,
    cols: usize,
    factor: usize,
) -> (Vec<f64>, usize, usize) {
    let out_rows = rows * factor;
    let out_cols = cols * factor;
    let mut out = vec![0.0f64; out_rows * out_cols];

    let get = |r: isize, c: isize| -> f64 {
        let r = r.clamp(0, rows as isize - 1) as usize;
        let c = c.clamp(0, cols as isize - 1) as usize;
        data[r * cols + c]
    };

    for oy in 0..out_rows {
        let src_y = oy as f64 / factor as f64;
        let iy = src_y.floor() as isize;
        let ty = src_y - src_y.floor();
        let wy = cubic_hermite(ty);

        for ox in 0..out_cols {
            let src_x = ox as f64 / factor as f64;
            let ix = src_x.floor() as isize;
            let tx = src_x - src_x.floor();
            let wx = cubic_hermite(tx);

            let mut val = 0.0;
            for m in 0..4isize {
                for n in 0..4isize {
                    val += wy[m as usize] * wx[n as usize] * get(iy + m - 1, ix + n - 1);
                }
            }
            out[oy * out_cols + ox] = val;
        }
    }

    (out, out_rows, out_cols)
}

// --- im2col Transform -------------------------------------------------------

/// im2col: extract image patches as columns for convolution via GEMM.
///
/// Given input of size (H, W) and kernel (K, K), produces a matrix:
///   rows = K*K, cols = (H-K+1)*(W-K+1)
/// Each column is one flattened patch.
pub fn im2col(
    data: &[f64],
    h: usize,
    w: usize,
    k: usize,
) -> (Vec<f64>, usize, usize) {
    let out_h = h - k + 1;
    let out_w = w - k + 1;
    let patch_rows = k * k;
    let patch_cols = out_h * out_w;
    let mut patches = vec![0.0f64; patch_rows * patch_cols];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let col_idx = oy * out_w + ox;
            for ky in 0..k {
                for kx in 0..k {
                    let row_idx = ky * k + kx;
                    patches[row_idx * patch_cols + col_idx] =
                        data[(oy + ky) * w + (ox + kx)];
                }
            }
        }
    }

    (patches, patch_rows, patch_cols)
}

// --- Kernel Generation ------------------------------------------------------

/// Generate a convolution kernel of given type and size.
pub fn generate_kernel(ktype: KernelType, size: usize) -> Vec<f64> {
    let n = size;
    let center = n as f64 / 2.0;
    match ktype {
        KernelType::Gaussian => {
            let sigma = n as f64 / 4.0;
            let mut k = vec![0.0; n * n];
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let dx = i as f64 - center + 0.5;
                    let dy = j as f64 - center + 0.5;
                    let v = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
                    k[i * n + j] = v;
                    sum += v;
                }
            }
            // Normalize
            for v in &mut k {
                *v /= sum;
            }
            k
        }
        KernelType::Sharpen => {
            let mut k = vec![0.0; n * n];
            let c = n / 2;
            // Center = n*n, neighbors = -1
            for i in 0..n {
                for j in 0..n {
                    k[i * n + j] = -1.0;
                }
            }
            k[c * n + c] = (n * n) as f64;
            k
        }
        KernelType::EdgeDetect => {
            let mut k = vec![0.0; n * n];
            let c = n / 2;
            let neighbor_count = (n * n - 1) as f64;
            for i in 0..n {
                for j in 0..n {
                    k[i * n + j] = -1.0 / neighbor_count;
                }
            }
            k[c * n + c] = 1.0;
            k
        }
        KernelType::BoxBlur => {
            let val = 1.0 / (n * n) as f64;
            vec![val; n * n]
        }
    }
}

// --- Main Pipeline ----------------------------------------------------------

/// Run the complete IR vision processing pipeline.
pub fn run_vision_pipeline(
    config: &VisionConfig,
    progress: &Arc<AtomicU64>,
    phase: &Arc<Mutex<String>>,
) -> anyhow::Result<VisionResult> {
    let start = std::time::Instant::now();

    // Phase 1: Generate synthetic IR data
    {
        let mut ph = phase.lock().unwrap();
        *ph = format!("Generating {}x{} IR sensor data...", config.input_rows, config.input_cols);
    }
    let input = generate_ir_data(
        config.input_rows,
        config.input_cols,
        config.noise_level,
        config.seed,
    );
    let input_min = input.iter().cloned().fold(f64::MAX, f64::min);
    let input_max = input.iter().cloned().fold(f64::MIN, f64::max);
    progress.store(10, Ordering::Relaxed);

    // Phase 2: Bicubic upscale
    {
        let mut ph = phase.lock().unwrap();
        *ph = format!(
            "Bicubic interpolation {}x{} -> {}x{}...",
            config.input_rows,
            config.input_cols,
            config.input_rows * config.upscale_factor,
            config.input_cols * config.upscale_factor,
        );
    }
    let upscale_start = std::time::Instant::now();
    let (upscaled, up_rows, up_cols) = bicubic_upscale(
        &input,
        config.input_rows,
        config.input_cols,
        config.upscale_factor,
    );
    let upscale_ms = upscale_start.elapsed().as_secs_f64() * 1000.0;
    progress.store(30, Ordering::Relaxed);

    // Phase 3: im2col transform
    let ks = config.kernel_size;
    {
        let mut ph = phase.lock().unwrap();
        *ph = format!("im2col transform ({}x{} patches)...", ks, ks);
    }
    let im2col_start = std::time::Instant::now();
    let (patches, patch_rows, patch_cols) = im2col(&upscaled, up_rows, up_cols, ks);
    let im2col_ms = im2col_start.elapsed().as_secs_f64() * 1000.0;
    progress.store(50, Ordering::Relaxed);

    // Phase 4: Build kernel weight matrix (1 x K*K for single-filter convolution)
    // For GEMM: output = W * patches where W is (1 x K*K) and patches is (K*K x patch_cols)
    // Result is (1 x patch_cols) which reshapes to output image.
    {
        let mut ph = phase.lock().unwrap();
        *ph = format!("GEMM convolution via HPC multiply ({} kernel)...", config.kernel_type.label());
    }
    let kernel = generate_kernel(config.kernel_type, ks);

    // Build Matrix objects for GEMM
    let w_mat = Matrix::from_flat(1, patch_rows, kernel)
        .map_err(|e| anyhow::anyhow!("Kernel matrix error: {e}"))?;
    let p_mat = Matrix::from_flat(patch_rows, patch_cols, patches)
        .map_err(|e| anyhow::anyhow!("Patch matrix error: {e}"))?;

    let gemm_start = std::time::Instant::now();
    // Use HPC multiply for the GEMM step
    let result_mat = if patch_rows >= 64 && patch_cols >= 64 {
        algorithms::multiply_hpc_fused(&w_mat, &p_mat, 256, 32)
    } else {
        // Small matrix: use basic multiply
        algorithms::multiply_naive(&w_mat, &p_mat)
    };
    let gemm_ms = gemm_start.elapsed().as_secs_f64() * 1000.0;
    progress.store(90, Ordering::Relaxed);

    // Reshape output: (1 x patch_cols) -> (out_h x out_w)
    let out_h = up_rows - ks + 1;
    let out_w = up_cols - ks + 1;
    let output = result_mat.data()[..out_h * out_w].to_vec();

    let output_min = output.iter().cloned().fold(f64::MAX, f64::min);
    let output_max = output.iter().cloned().fold(f64::MIN, f64::max);

    // Estimate SNR (signal-to-noise ratio)
    let signal_power: f64 = output.iter().map(|&v| v * v).sum::<f64>() / output.len() as f64;
    let noise_power = config.noise_level * config.noise_level;
    let snr = if noise_power > 1e-15 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };

    let computation_ms = start.elapsed().as_secs_f64() * 1000.0;
    progress.store(100, Ordering::Relaxed);

    Ok(VisionResult {
        config: config.clone(),
        input_matrix: input,
        upscaled_matrix: upscaled,
        upscaled_rows: up_rows,
        upscaled_cols: up_cols,
        output_matrix: output,
        output_rows: out_h,
        output_cols: out_w,
        im2col_rows: patch_rows,
        im2col_cols: patch_cols,
        computation_ms,
        upscale_ms,
        im2col_ms,
        gemm_ms,
        input_min,
        input_max,
        output_min,
        output_max,
        snr_estimate: snr,
    })
}
