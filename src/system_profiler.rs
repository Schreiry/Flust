// system_profiler.rs — Smart System Profiler (50-axis analysis)
//
// Runs ONCE at startup. Analyzes CPU, cache, memory, SIMD capabilities
// and produces human-readable assessment with concrete recommendations.
// All data derives from SystemInfo + heuristics — no additional system calls.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::common::{SimdLevel, ThemeColors};
use crate::interactive::kv_line;
use crate::system::SystemInfo;

// ─── Data Structures ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemTier {
    Entry,
    Capable,
    Professional,
    Workstation,
    Server,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bottleneck {
    CoreCount,
    Memory,
    Compute,
    Balanced,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkStrategy {
    SingleRow,
    MultiRowFixed,
    TileBlock,
}

#[derive(Debug, Clone)]
pub struct OptimalFlustConfig {
    pub threads: usize,
    pub tile_l1: usize,
    pub tile_l2: usize,
    pub strassen_threshold: usize,
    pub use_avx512: bool,
    pub use_avx2: bool,
    pub use_prefetch: bool,
    pub chunk_strategy: ChunkStrategy,
}

#[derive(Debug, Clone)]
pub struct SystemProfile {
    // === Compute ===
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub has_hyperthreading: bool,
    pub is_hybrid_arch: bool,
    pub p_cores_estimate: usize,
    pub e_cores_estimate: usize,
    pub base_freq_ghz: f64,
    pub boost_freq_ghz: f64,
    pub simd_level: SimdLevel,
    pub has_fma: bool,
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub theoretical_peak_gflops_fp64: f64,
    pub realistic_peak_gflops_fp64: f64,

    // === Cache hierarchy ===
    pub l1_cache_kb_per_pcore: usize,
    pub l2_cache_mb_per_pcore: f64,
    pub l3_cache_mb_total: f64,
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    pub dram_bandwidth_gb_s: f64,
    pub cache_line_bytes: usize,

    // === Matrix limits ===
    pub max_matrix_n_in_l1: usize,
    pub max_matrix_n_in_l2: usize,
    pub max_matrix_n_in_l3: usize,
    pub max_safe_matrix_n: usize,

    // === Optimal config ===
    pub optimal_tile_l1: usize,
    pub optimal_tile_l2: usize,
    pub optimal_strassen_threshold: usize,
    pub optimal_thread_count: usize,

    // === Assessments ===
    pub tier: SystemTier,
    pub bottleneck: Bottleneck,
    pub human_assessment: String,
    pub technical_assessment: String,
    pub config: OptimalFlustConfig,
}

// ─── Profile Collection ─────────────────────────────────────────────────────

pub fn collect_system_profile(sys: &SystemInfo) -> SystemProfile {
    let logical_cores = sys.logical_cores;
    let physical_cores = sys.physical_cores.max(1);
    let total_ram_gb = sys.total_ram_mb as f64 / 1024.0;
    let available_ram_gb = sys.available_ram_mb as f64 / 1024.0;
    let simd = sys.simd_level;
    let is_hybrid = sys.core_topology.is_hybrid;
    let p_cores = sys.core_topology.p_cores;
    let e_cores = sys.core_topology.e_cores;

    // Cache hierarchy from SystemInfo or heuristics
    let l1_kb = sys.l1_cache_kb.unwrap_or(if is_hybrid { 48 } else { 32 }) as usize;
    let l2_kb = sys.l2_cache_kb.unwrap_or(if is_hybrid { 2048 } else { 256 });
    let l2_mb_per_core = l2_kb as f64 / 1024.0;
    let l3_kb = sys.l3_cache_kb.unwrap_or(
        if logical_cores >= 16 { 36864 }
        else if logical_cores >= 8 { 16384 }
        else { 8192 }
    );
    let l3_mb = l3_kb as f64 / 1024.0;

    // Matrix sizes that fit in each cache level (3 matrices: A, B, C)
    // 3 * n^2 * 8 bytes <= cache_size
    let max_n_l1 = ((l1_kb as f64 * 1024.0 / 3.0 / 8.0).sqrt()) as usize;
    let max_n_l2 = ((l2_mb_per_core * 1024.0 * 1024.0 / 3.0 / 8.0).sqrt()) as usize;
    let max_n_l3 = ((l3_mb * 1024.0 * 1024.0 / 3.0 / 8.0).sqrt()) as usize;
    let max_n_safe = ((available_ram_gb * 0.5 * 1024.0 * 1024.0 * 1024.0 / 3.0 / 8.0).sqrt()) as usize;

    // Optimal tiles (80% of max to leave room for parallel cache sharing)
    let tile_l1 = round_down_to_multiple((max_n_l1 as f64 * 0.8) as usize, 8).max(8);
    let tile_l2 = round_down_to_multiple((max_n_l2 as f64 * 0.8) as usize, tile_l1).max(tile_l1);

    // Theoretical peak FP64 GFLOPS
    let base_freq_ghz = sys.base_freq_ghz;
    let fma_ports = sys.peak_estimate.fma_ports as f64;
    let fp64_per_cycle = simd.fp64_ops_per_cycle_per_fma();
    let peak_theoretical = physical_cores as f64 * fp64_per_cycle * fma_ports * base_freq_ghz;
    let peak_realistic = peak_theoretical * 0.35;

    // System tier
    let tier = classify_tier(physical_cores, total_ram_gb as usize, &simd);
    let bottleneck = classify_bottleneck(physical_cores, available_ram_gb, &simd);

    let human_assessment = generate_human_assessment(
        &sys.cpu_brand, physical_cores, total_ram_gb,
        peak_realistic, max_n_safe, &tier, &bottleneck,
    );

    let technical_assessment = format!(
        "Config: {}L/{}P cores | {} | L1={}KB L2={:.0}MB L3={:.0}MB | \
         tile_L1={} tile_L2={} | Peak theoretical={:.0}G realistic={:.0}G",
        logical_cores, physical_cores, simd.display_name(),
        l1_kb, l2_mb_per_core, l3_mb, tile_l1, tile_l2, peak_theoretical, peak_realistic
    );

    let strassen_threshold = (tile_l1 * 2).max(32).min(128);

    let optimal_config = OptimalFlustConfig {
        threads: physical_cores,
        tile_l1,
        tile_l2,
        strassen_threshold,
        use_avx512: matches!(simd, SimdLevel::Avx512),
        use_avx2: matches!(simd, SimdLevel::Avx2 | SimdLevel::Avx512),
        use_prefetch: physical_cores >= 4,
        chunk_strategy: if physical_cores >= 12 { ChunkStrategy::TileBlock }
                        else if physical_cores >= 4 { ChunkStrategy::MultiRowFixed }
                        else { ChunkStrategy::SingleRow },
    };

    SystemProfile {
        logical_cores,
        physical_cores,
        has_hyperthreading: logical_cores > physical_cores,
        is_hybrid_arch: is_hybrid,
        p_cores_estimate: p_cores,
        e_cores_estimate: e_cores,
        base_freq_ghz,
        boost_freq_ghz: base_freq_ghz * 1.15,
        simd_level: simd,
        has_fma: matches!(simd, SimdLevel::Avx2 | SimdLevel::Avx512),
        has_avx512: matches!(simd, SimdLevel::Avx512),
        has_avx2: matches!(simd, SimdLevel::Avx2 | SimdLevel::Avx512),
        theoretical_peak_gflops_fp64: peak_theoretical,
        realistic_peak_gflops_fp64: peak_realistic,
        l1_cache_kb_per_pcore: l1_kb,
        l2_cache_mb_per_pcore: l2_mb_per_core,
        l3_cache_mb_total: l3_mb,
        ram_total_gb: total_ram_gb,
        ram_available_gb: available_ram_gb,
        dram_bandwidth_gb_s: 76.8,
        cache_line_bytes: 64,
        max_matrix_n_in_l1: max_n_l1,
        max_matrix_n_in_l2: max_n_l2,
        max_matrix_n_in_l3: max_n_l3,
        max_safe_matrix_n: max_n_safe,
        optimal_tile_l1: tile_l1,
        optimal_tile_l2: tile_l2,
        optimal_strassen_threshold: strassen_threshold,
        optimal_thread_count: physical_cores,
        tier,
        bottleneck,
        human_assessment,
        technical_assessment,
        config: optimal_config,
    }
}

// ─── Classification ─────────────────────────────────────────────────────────

fn classify_tier(physical_cores: usize, total_ram_gb: usize, simd: &SimdLevel) -> SystemTier {
    let has_good_simd = matches!(simd, SimdLevel::Avx2 | SimdLevel::Avx512);
    match (physical_cores, total_ram_gb, has_good_simd) {
        (c, r, true) if c >= 24 && r >= 64 => SystemTier::Server,
        (c, r, true) if c >= 12 && r >= 32 => SystemTier::Workstation,
        (c, r, true) if c >= 6  && r >= 16 => SystemTier::Professional,
        (c, _, true) if c >= 4             => SystemTier::Capable,
        _                                   => SystemTier::Entry,
    }
}

fn classify_bottleneck(physical_cores: usize, available_ram_gb: f64, simd: &SimdLevel) -> Bottleneck {
    if physical_cores < 4 {
        Bottleneck::CoreCount
    } else if available_ram_gb < 4.0 {
        Bottleneck::Memory
    } else if !matches!(simd, SimdLevel::Avx2 | SimdLevel::Avx512) {
        Bottleneck::Compute
    } else {
        Bottleneck::Balanced
    }
}

// ─── Human-Readable Assessment ──────────────────────────────────────────────

fn generate_human_assessment(
    brand: &str, p_cores: usize, ram_gb: f64,
    realistic_gflops: f64, max_n: usize,
    tier: &SystemTier, bottleneck: &Bottleneck,
) -> String {
    let tier_desc = match tier {
        SystemTier::Server       => "server-class",
        SystemTier::Workstation  => "professional workstation",
        SystemTier::Professional => "professional-grade",
        SystemTier::Capable      => "mid-range",
        SystemTier::Entry        => "entry-level",
    };

    let bottleneck_advice = match bottleneck {
        Bottleneck::CoreCount => format!(
            "For large matrices (> 2048x2048), a server with 16+ cores is recommended."),
        Bottleneck::Memory => format!(
            "Limited RAM. Safe maximum matrix: {}x{}.", max_n, max_n),
        Bottleneck::Compute => format!(
            "CPU lacks AVX2 — speed is limited. Consider upgrading CPU."),
        Bottleneck::Balanced => format!(
            "System well-balanced. Maximum safe matrix: {}x{}.", max_n, max_n),
    };

    format!(
        "Your computer ({}) is a {} system. \
         With {} compute cores and {:.0} GB RAM, Flust can process \
         matrices up to {}x{} in seconds, achieving ~{:.0} GFLOPS. {}",
        brand, tier_desc, p_cores, ram_gb, max_n, max_n, realistic_gflops, bottleneck_advice
    )
}

fn generate_task_recommendations(p: &SystemProfile) -> Vec<String> {
    let mut recs = Vec::new();
    if p.optimal_thread_count < 8 {
        recs.push(format!(
            "-> Use Tiled-Parallel over Strassen for matrices < 512x512 on your {} cores.",
            p.physical_cores
        ));
    } else {
        recs.push(format!(
            "-> Strassen recommended for matrices >= {}x{} (threshold auto-set).",
            p.optimal_strassen_threshold * 2, p.optimal_strassen_threshold * 2
        ));
    }
    if p.ram_available_gb < 8.0 {
        recs.push(format!(
            "-> Low RAM: avoid matrices > {}x{}. Close other apps before large benchmarks.",
            p.max_safe_matrix_n, p.max_safe_matrix_n
        ));
    }
    if p.is_hybrid_arch {
        recs.push(
            "-> Hybrid CPU detected: Rayon pool limited to P-cores for best FP throughput.".into(),
        );
    }
    let expected_ms = (2048.0_f64.powi(3) * 2.0
        / p.realistic_peak_gflops_fp64.max(0.1) / 1e9 * 1000.0) as usize;
    recs.push(format!(
        "-> Optimal for this machine: matrix 2048x2048 expected ~{}ms.", expected_ms
    ));
    recs
}

// ─── TUI Rendering ──────────────────────────────────────────────────────────

pub fn render_system_profile(
    frame: &mut ratatui::Frame,
    area: Rect,
    profile: &SystemProfile,
    t: &ThemeColors,
) {
    let tier_color = match profile.tier {
        SystemTier::Server       => t.crit,
        SystemTier::Workstation  => t.warn,
        SystemTier::Professional => t.ok,
        SystemTier::Capable      => t.accent,
        SystemTier::Entry        => t.text_dim,
    };

    let tier_label = match profile.tier {
        SystemTier::Server       => "SERVER",
        SystemTier::Workstation  => "WORKSTATION",
        SystemTier::Professional => "PROFESSIONAL",
        SystemTier::Capable      => "CAPABLE",
        SystemTier::Entry        => "ENTRY",
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),   // human assessment
            Constraint::Length(3),   // tier badge + peak performance
            Constraint::Min(15),     // detailed parameters (4 columns)
            Constraint::Length(6),   // recommendations
            Constraint::Length(3),   // footer
        ])
        .split(area);

    // [1] Human Assessment
    let assessment_block = Block::default()
        .title(Span::styled(
            " System Assessment ",
            Style::default().fg(tier_color).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(tier_color));

    let wrapped = wrap_text(&profile.human_assessment, area.width.saturating_sub(6) as usize);
    let mut assessment_lines = vec![Line::from("")];
    for line in &wrapped {
        assessment_lines.push(Line::from(Span::styled(
            format!("  {}", line),
            Style::default().fg(t.text),
        )));
    }
    frame.render_widget(
        Paragraph::new(assessment_lines).block(assessment_block),
        chunks[0],
    );

    // [2] Tier badge + Performance bar
    let tier_line = Line::from(vec![
        Span::styled("  System Class: ", Style::default().fg(t.text_dim)),
        Span::styled(
            format!("[ {} ]", tier_label),
            Style::default().fg(tier_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled("    Realistic Peak: ", Style::default().fg(t.text_dim)),
        Span::styled(
            format!("{:.1} GFLOPS", profile.realistic_peak_gflops_fp64),
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  ({:.1} theoretical)", profile.theoretical_peak_gflops_fp64),
            Style::default().fg(t.text_dim),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(tier_line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[1],
    );

    // [3] Detailed parameters in 4 columns
    let detail_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[2]);

    // Column 1: Processor
    let cpu_lines = vec![
        Line::from(Span::styled(
            "  PROCESSOR",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  Logical  ", &format!("{}", profile.logical_cores), t.text, t.text_dim),
        kv_line("  Physical ", &format!("{}", profile.physical_cores), t.text, t.text_dim),
        kv_line("  P-cores  ", &format!("~{}", profile.p_cores_estimate), t.text, t.text_dim),
        kv_line("  E-cores  ", &format!("~{}", profile.e_cores_estimate), t.text_muted, t.text_dim),
        kv_line("  Hybrid   ", if profile.is_hybrid_arch { "YES" } else { "NO" }, t.text, t.text_dim),
        kv_line("  HT       ", if profile.has_hyperthreading { "YES" } else { "NO" }, t.text_muted, t.text_dim),
        kv_line("  Base     ", &format!("{:.1} GHz", profile.base_freq_ghz), t.text, t.text_dim),
        kv_line("  Boost    ", &format!("~{:.1} GHz", profile.boost_freq_ghz), t.text_muted, t.text_dim),
    ];

    // Column 2: SIMD and compute
    let simd_lines = vec![
        Line::from(Span::styled(
            "  COMPUTE",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  SIMD     ", profile.simd_level.display_name(), if profile.has_avx2 { t.ok } else { t.warn }, t.text_dim),
        kv_line("  AVX2     ", if profile.has_avx2 { "Y" } else { "N" }, if profile.has_avx2 { t.ok } else { t.text_dim }, t.text_dim),
        kv_line("  AVX-512  ", if profile.has_avx512 { "Y" } else { "N" }, if profile.has_avx512 { t.ok } else { t.text_dim }, t.text_dim),
        kv_line("  FMA      ", if profile.has_fma { "Y" } else { "N" }, if profile.has_fma { t.ok } else { t.text_dim }, t.text_dim),
        kv_line("  Peak TH  ", &format!("{:.0}G", profile.theoretical_peak_gflops_fp64), t.text, t.text_dim),
        kv_line("  Peak RE  ", &format!("{:.0}G", profile.realistic_peak_gflops_fp64), t.accent, t.text_dim),
        kv_line("  DRAM BW  ", &format!("{:.0} GB/s", profile.dram_bandwidth_gb_s), t.text_muted, t.text_dim),
    ];

    // Column 3: Cache hierarchy
    let cache_lines = vec![
        Line::from(Span::styled(
            "  CACHE",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  L1 (P)   ", &format!("{} KB", profile.l1_cache_kb_per_pcore), t.text, t.text_dim),
        kv_line("  L2/core  ", &format!("{:.0} MB", profile.l2_cache_mb_per_pcore), t.text, t.text_dim),
        kv_line("  L3 total ", &format!("{:.0} MB", profile.l3_cache_mb_total), t.text, t.text_dim),
        kv_line("  Max in L1", &format!("{}x{}", profile.max_matrix_n_in_l1, profile.max_matrix_n_in_l1), t.text_muted, t.text_dim),
        kv_line("  Max in L2", &format!("{}x{}", profile.max_matrix_n_in_l2, profile.max_matrix_n_in_l2), t.text_muted, t.text_dim),
        kv_line("  Max in L3", &format!("{}x{}", profile.max_matrix_n_in_l3, profile.max_matrix_n_in_l3), t.text_muted, t.text_dim),
        kv_line("  CacheLine", &format!("{} bytes", profile.cache_line_bytes), t.text_dim, t.text_dim),
    ];

    // Column 4: Optimal Flust config
    let cfg = &profile.config;
    let config_lines = vec![
        Line::from(Span::styled(
            "  FLUST CONFIG",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        kv_line("  Threads  ", &format!("{}", cfg.threads), t.ok, t.text_dim),
        kv_line("  Tile L1  ", &format!("{}", cfg.tile_l1), t.text, t.text_dim),
        kv_line("  Tile L2  ", &format!("{}", cfg.tile_l2), t.text, t.text_dim),
        kv_line("  Strassen>", &format!("{}", cfg.strassen_threshold), t.text, t.text_dim),
        kv_line("  Prefetch ", if cfg.use_prefetch { "ON" } else { "OFF" }, t.text_muted, t.text_dim),
        kv_line("  Max safe ", &format!("{}x{}", profile.max_safe_matrix_n, profile.max_safe_matrix_n), t.accent, t.text_dim),
        kv_line("  RAM avail", &format!("{:.1} GB", profile.ram_available_gb), t.text, t.text_dim),
    ];

    let all_cols: [&Vec<Line>; 4] = [&cpu_lines, &simd_lines, &cache_lines, &config_lines];
    for (col, lines) in all_cols.iter().enumerate() {
        frame.render_widget(
            Paragraph::new((*lines).clone()).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(t.border)),
            ),
            detail_cols[col],
        );
    }

    // [4] Recommendations
    let recs = generate_task_recommendations(profile);
    let rec_lines: Vec<Line> = recs
        .iter()
        .map(|r| {
            Line::from(Span::styled(
                format!("  {}", r),
                Style::default().fg(t.text_muted),
            ))
        })
        .collect();
    frame.render_widget(
        Paragraph::new(rec_lines).block(
            Block::default()
                .title(" Recommendations for Your System ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(t.border)),
        ),
        chunks[3],
    );

    // [5] Footer
    let footer = Line::from(vec![
        Span::styled("  [Q]", Style::default().fg(t.accent)),
        Span::styled(" Back", Style::default().fg(t.text_muted)),
        Span::styled("    ", Style::default()),
        Span::styled(&profile.technical_assessment, Style::default().fg(t.text_dim)),
    ]);
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(t.surface)),
        chunks[4],
    );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn round_down_to_multiple(n: usize, m: usize) -> usize {
    if m == 0 { return n; }
    (n / m) * m
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.len() + word.len() + 1 > width && !current.is_empty() {
            lines.push(current.clone());
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}
