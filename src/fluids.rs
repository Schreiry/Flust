// ─── Fluid Catalog ──────────────────────────────────────────────────────────
//
// Extended catalog of fluids and semi-fluids with scientifically-referenced
// thermophysical properties. All values at ~20-25°C unless noted.
//
// Sources:
//   [1] Incropera, DeWitt — "Fundamentals of Heat and Mass Transfer", 7th ed.
//   [2] CRC Handbook of Chemistry and Physics, 97th ed.
//   [3] TPRC — Thermophysical Properties of Matter (Touloukian)
//   [4] Janz et al. — "Physical Properties Data for Molten Salts", 1988
//   [5] ASHRAE Handbook — Fundamentals
//   [6] Engineering Toolbox (engineeringtoolbox.com)
//   [7] Zalba et al. — "Review on PCM thermal storage", Applied Thermal Eng. 2003
//   [8] Sopade et al. — "Rheological characterisation of food materials", 2003

use crate::thermal::FluidProperties;

// ─── Category ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluidCategory {
    Liquid,
    SemiFluid,
}

impl FluidCategory {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Liquid => "Liquid",
            Self::SemiFluid => "Semi-fluid",
        }
    }
}

// ─── Catalog Entry ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FluidEntry {
    pub props: FluidProperties,
    pub category: FluidCategory,
    pub description: &'static str,
    pub source: &'static str,
}

// ─── Full Catalog ────────────────────────────────────────────────────────────

pub fn all_fluids() -> Vec<FluidEntry> {
    vec![
        // ── Liquids ──────────────────────────────────────────────────────
        FluidEntry {
            props: FluidProperties {
                name: "Water".into(),
                thermal_conductivity: 0.598,
                density: 998.0,
                specific_heat: 4182.0,
            },
            category: FluidCategory::Liquid,
            description: "Baseline coolant. High heat capacity, slow cooling",
            source: "Incropera [1], T=20\u{00b0}C",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Engine Oil".into(),
                thermal_conductivity: 0.145,
                density: 880.0,
                specific_heat: 1900.0,
            },
            category: FluidCategory::Liquid,
            description: "Lubricant/coolant. Low conductivity, retains heat",
            source: "Incropera [1]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Ethylene Glycol".into(),
                thermal_conductivity: 0.400,
                density: 1070.0,
                specific_heat: 3400.0,
            },
            category: FluidCategory::Liquid,
            description: "Antifreeze. Moderate properties, wide temp range",
            source: "Incropera [1]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Glycerin".into(),
                thermal_conductivity: 0.286,
                density: 1260.0,
                specific_heat: 2427.0,
            },
            category: FluidCategory::Liquid,
            description: "Viscous liquid. Slow convection, even heat distribution",
            source: "CRC Handbook [2]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Mercury".into(),
                thermal_conductivity: 8.54,
                density: 13534.0,
                specific_heat: 139.0,
            },
            category: FluidCategory::Liquid,
            description: "Ultra-high conductivity. Nuclear/industrial heat exchangers",
            source: "CRC Handbook [2]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Liquid Sodium (400K)".into(),
                thermal_conductivity: 86.2,
                density: 927.0,
                specific_heat: 1385.0,
            },
            category: FluidCategory::Liquid,
            description: "Fast breeder reactor coolant. Extreme conductivity",
            source: "TPRC [3], T=127\u{00b0}C",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Molten Salt (Solar)".into(),
                thermal_conductivity: 0.55,
                density: 1794.0,
                specific_heat: 1500.0,
            },
            category: FluidCategory::Liquid,
            description: "NaNO\u{2083}/KNO\u{2083} eutectic. Solar thermal storage",
            source: "Janz [4]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Honey".into(),
                thermal_conductivity: 0.50,
                density: 1420.0,
                specific_heat: 2500.0,
            },
            category: FluidCategory::Liquid,
            description: "High-viscosity natural fluid. Food processing modeling",
            source: "Sopade [8]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Motor Oil 5W-30".into(),
                thermal_conductivity: 0.145,
                density: 860.0,
                specific_heat: 2000.0,
            },
            category: FluidCategory::Liquid,
            description: "Thin motor oil. Automotive cooling/lubrication",
            source: "Engineering Toolbox [6]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Motor Oil 10W-40".into(),
                thermal_conductivity: 0.150,
                density: 875.0,
                specific_heat: 1950.0,
            },
            category: FluidCategory::Liquid,
            description: "Thick motor oil. Heavy-duty engine applications",
            source: "Engineering Toolbox [6]",
        },

        // ── Semi-Fluids ──────────────────────────────────────────────────
        FluidEntry {
            props: FluidProperties {
                name: "Wet Sand".into(),
                thermal_conductivity: 2.0,
                density: 1922.0,
                specific_heat: 830.0,
            },
            category: FluidCategory::SemiFluid,
            description: "Porous medium approx. Geothermal/construction modeling",
            source: "ASHRAE [5]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Concrete (wet)".into(),
                thermal_conductivity: 1.4,
                density: 2300.0,
                specific_heat: 880.0,
            },
            category: FluidCategory::SemiFluid,
            description: "Hydrating concrete. Thermal mass/curing simulation",
            source: "ASHRAE [5]",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Hydrogel".into(),
                thermal_conductivity: 0.55,
                density: 1020.0,
                specific_heat: 3800.0,
            },
            category: FluidCategory::SemiFluid,
            description: "Water-based gel. Biomedical/tissue-phantom modeling",
            source: "Various biomedical refs",
        },
        FluidEntry {
            props: FluidProperties {
                name: "Paraffin Wax (liquid)".into(),
                thermal_conductivity: 0.21,
                density: 780.0,
                specific_heat: 2500.0,
            },
            category: FluidCategory::SemiFluid,
            description: "Phase-change material (PCM). Thermal energy storage",
            source: "Zalba [7]",
        },
    ]
}

/// Total number of fluids in the catalog.
pub fn fluid_count() -> usize {
    14
}

/// Get FluidProperties by catalog index. Falls back to Water.
pub fn fluid_by_index(idx: usize) -> FluidProperties {
    let catalog = all_fluids();
    catalog
        .into_iter()
        .nth(idx)
        .map(|e| e.props)
        .unwrap_or_else(FluidProperties::water)
}
