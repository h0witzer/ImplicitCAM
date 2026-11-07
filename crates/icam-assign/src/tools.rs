//! One physical cut model per end mill on the tool sheet, in 6061-T6 on the router, plus
//! the vendor wide-cut envelope and a feed table.
//!
//! The 1/2 in tool is the roughing planner's own model (CutModel::al6061). The 1/4, 1/8
//! and 1/16 in tools are built here field by field; every constant carries its source,
//! and "assumed" marks the ones nothing on disk pins. Material constants (Kienzle kc,
//! exponent, radial ratio) and machine constants (spindle, efficiency, feed cap,
//! deflection limit) are copied from the 1/2 in model so all four tools see the same
//! material and machine.

use std::f64::consts::PI;

use icam_motion::CutModel;
use icam_post::ToolDef;
use rayon::prelude::*;

use crate::vendor::{envelope, VendorRow};

const IN: f64 = 25.4;
/// mm the holder must stay above the stock top (assumed).
pub const HOLDER_MARGIN: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub number: i32,
    pub label: String,
    /// At no particular depth.
    pub model: CutModel,
    /// Nominal size, inches (the vendor class).
    pub d_in: f64,
    /// Deepest floor below the stock top the holder allows, mm.
    pub max_depth: f64,
    pub sources: String,
    /// Wide-cut envelope (ae/D, ap/D); empty: none.
    pub env: Vec<(f64, f64)>,
    /// Feed table: ceiling feed (mm/s) at pass depth ap_grid[a] and engagement deg (0..180).
    pub ap_grid: Vec<f64>,
    pub vtab: Vec<Vec<f32>>,
}

impl ToolSpec {
    pub fn d(&self) -> f64 {
        self.model.d
    }

    pub fn r(&self) -> f64 {
        self.model.d / 2.0
    }

    /// Deepest pass the envelope allows at an equivalent straight engagement ae/D.
    pub fn env_ap(&self, ae_d: f64) -> f64 {
        let loc = self.model.loc;
        let env = &self.env;
        if env.is_empty() {
            return loc;
        }
        let ap_d = if ae_d <= env[0].0 {
            env[0].1
        } else if ae_d >= env[env.len() - 1].0 {
            env[env.len() - 1].1
        } else {
            let mut k = 1;
            while env[k].0 < ae_d {
                k += 1;
            }
            let ((a0, p0), (a1, p1)) = (env[k - 1], env[k]);
            p0 + (p1 - p0) * (ae_d - a0) / (a1 - a0)
        };
        loc.min(ap_d * self.d())
    }

    /// Envelope at an engagement angle (its equivalent straight ae = r (1 - cos phi)).
    pub fn env_ap_phi(&self, phi: f64) -> f64 {
        self.env_ap((1.0 - phi.min(PI).cos()) / 2.0)
    }

    /// Ceiling feed at engagement phi (rad) and pass depth ap, read at the next deeper
    /// tabulated depth (feed falls with depth, so this never overstates it).
    pub fn v(&self, phi: f64, ap: f64) -> f64 {
        let n = self.ap_grid.len() as i64;
        let step = self.ap_grid[0];
        let a = (((ap / step - 1e-9).ceil() as i64) - 1).clamp(0, n - 1) as usize;
        let deg = (phi * 180.0 / PI).clamp(0.0, 180.0);
        let i = 179.min(deg as usize);
        let row = &self.vtab[a];
        row[i] as f64 + (row[i + 1] - row[i]) as f64 * (deg - i as f64)
    }

    /// One CutModel::at_depth (its helical force integral) per depth.
    pub fn build_table(&mut self, n_ap: usize) {
        let m = &self.model;
        self.ap_grid = (0..n_ap).map(|a| m.loc * (a + 1) as f64 / n_ap as f64).collect();
        let grid = &self.ap_grid;
        self.vtab = (0..n_ap)
            .into_par_iter()
            .map(|a| {
                let md = m.at_depth(grid[a]);
                (0..=180).map(|deg| md.ceiling((deg as f64).max(0.25) * PI / 180.0).0 as f32).collect()
            })
            .collect();
    }
}

#[derive(Clone, Debug)]
pub struct ModelOptions {
    /// The vendor wide-cut envelope on (default) or off (the planner's limit set).
    pub envelope: bool,
    /// Deflection limit override, inches (all tools).
    pub defl_in: Option<f64>,
    /// Scales the small tools' hangouts (a stub-length what-if).
    pub hangout_scale_small: f64,
    /// The 1/2 in tool's spindle speed override.
    pub rpm4: Option<f64>,
}

impl Default for ModelOptions {
    fn default() -> ModelOptions {
        ModelOptions { envelope: true, defl_in: None, hangout_scale_small: 1.0, rpm4: None }
    }
}

/// The models for the end mills on a tool sheet, biggest first (the 1/2 in tool must be
/// on it; end mills of other sizes have no model and are left out).
pub fn build(sheet: &[ToolDef], o: &ModelOptions, vendor: &[VendorRow]) -> Result<Vec<ToolSpec>, String> {
    let t4 = sheet
        .iter()
        .find(|t| t.kind == "ENDMILL" && (t.diameter - 12.7).abs() < 0.01)
        .ok_or("the tool sheet has no 1/2 in end mill (the cost model is built around it)")?;
    let base4 = CutModel::al6061(t4, o.rpm4)?;
    let like = |name: &str| {
        let mut m = CutModel::blank(name);
        // Material and machine: identical to the 1/2 in model (see CutModel::al6061 for their sources).
        m.kc_ref = base4.kc_ref;
        m.h_ref = base4.h_ref;
        m.mc = base4.mc;
        m.kr = base4.kr;
        m.spindle_w = base4.spindle_w;
        m.eff = base4.eff;
        m.v_max = base4.v_max;
        m.defl_max = base4.defl_max;
        m.e_modulus = 600_000.0; // N/mm2, cemented carbide
        m
    };
    let spec = |t: &ToolDef, label: String, model: CutModel, d_in: f64, sources: &str| ToolSpec {
        number: t.number,
        label,
        max_depth: model.hangout - HOLDER_MARGIN,
        model,
        d_in,
        sources: sources.into(),
        env: Vec::new(),
        ap_grid: Vec::new(),
        vtab: Vec::new(),
    };
    let mut list = vec![spec(
        t4,
        format!("T{:02} 1/2 in carbide 2FL", t4.number),
        base4.clone(),
        0.5,
        &format!("{}; rpm 18,000 (Amana aluminum chart)", base4.sources),
    )];
    let mut others: Vec<&ToolDef> = sheet.iter().filter(|t| t.kind == "ENDMILL" && (t.diameter - 12.7).abs() > 0.01).collect();
    others.sort_by(|a, b| b.diameter.partial_cmp(&a.diameter).unwrap_or(std::cmp::Ordering::Equal));
    for t in others {
        if (t.diameter - 6.35).abs() < 0.01 {
            let mut m = like("6061-T6, 1/4 in carbide 3-flute, MultiCAM");
            m.d = t.diameter;
            m.flutes = 3.0;
            m.loc = 0.75 * IN; // assumed: the sheet gives none; a stock 1/4 x 3/4 in 3FL
            m.hangout = 1.5 * IN; // tool sheet: hangout 1.500 in
            m.helix_deg = 35.0; // assumed: common 3FL aluminum helix
            m.core_ratio = 0.8; // assumed, as the 1/2 in tool
            // Chip ceiling: Amana SC "Spiral O aluminum" 1/4 in 0.006 in, Vortex 1/4 in 3FL
            // aluminum 0.007 in, Freud 1/4 in 0.007 in; the lower value taken.
            m.h_ceil = 0.006 * IN;
            m.h_floor = 0.0003 * IN; // Datron 6 mm aluminum: thinnest published max chip 0.00027 in
            m.rpm = 24000.0; // machine maximum; carbide in aluminum is not speed-limited here
            list.push(spec(t, format!("T{:02} 1/4 in carbide 3FL", t.number), m, 0.25,
                "chip 0.006 in (Amana 1/4 aluminum; Vortex 3FL 0.007); floor 0.0003 in (Datron 6 mm); hangout 1.5 in (tool sheet); assumed: LOC 0.75 in, helix 35 deg, core 0.8 D, 24,000 rpm"));
        } else if (t.diameter - 3.175).abs() < 0.01 {
            let mut m = like("6061-T6, 1/8 in carbide 1-flute, MultiCAM");
            m.d = t.diameter;
            m.flutes = 1.0;
            m.loc = 0.5 * IN; // assumed: the sheet gives none; Amana SC 1FL aluminum 1/8 D x 1/2 in
            m.hangout = 1.2 * IN; // tool sheet: hangout 1.200 in
            m.helix_deg = 21.0; // assumed: single-flute "O" aluminum geometry
            m.core_ratio = 0.7; // assumed: a single flute leaves a smaller core than 2FL (0.8)
            // Chip ceiling: Amana SC "Spiral O aluminum" 1/8 in single flute 0.004 in and Vortex
            // 1/8 in 1FL aluminum 0.004 in. Datron 3 mm 1FL goes to 0.0063 in in shallow slots.
            m.h_ceil = 0.004 * IN;
            m.h_floor = 0.0004 * IN; // Datron 3 mm aluminum: thinnest published max chip 0.00040 in
            m.rpm = 24000.0; // machine maximum; Datron runs this size at 34,000-48,000
            list.push(spec(t, format!("T{:02} 1/8 in carbide 1FL", t.number), m, 0.125,
                "chip 0.004 in (Amana 1FL O aluminum, Vortex 1FL); floor 0.0004 in (Datron 3 mm); hangout 1.2 in (tool sheet); assumed: LOC 0.5 in, helix 21 deg, core 0.7 D, 24,000 rpm"));
        } else if (t.diameter - 1.5875).abs() < 0.01 {
            let mut m = like("6061-T6, 1/16 in HSS 2-flute, MultiCAM");
            m.d = t.diameter;
            m.flutes = 2.0;
            m.loc = 0.125 * IN; // tool sheet: "1/16 EM HSS 2FL 1/8 LOC"
            m.hangout = 0.625 * IN; // tool sheet: hangout 0.625 in
            m.helix_deg = 30.0; // assumed
            m.core_ratio = 0.8; // assumed, as the 1/2 in tool
            m.e_modulus = 210_000.0; // N/mm2, high-speed steel (typical 200-220 GPa)
            // Chip ceiling: LMT/Belin 1.5 mm carbide in aluminum 0.00118 in; Datron 1.5 mm 1FL
            // carbide 0.00118-0.00276 in. Taken at the low end for HSS 2FL (an assumed derating).
            m.h_ceil = 0.0012 * IN;
            m.h_floor = 0.0003 * IN; // Datron 1.5 mm aluminum: thinnest published max chip 0.00035 in
            m.rpm = 24000.0; // 120 m/min (393 sfm), assumed: a conservative HSS speed in aluminum
            list.push(spec(t, format!("T{:02} 1/16 in HSS 2FL", t.number), m, 0.0625,
                "chip 0.0012 in (LMT/Belin 1.5 mm 0.00118; Datron 1.5 mm low end; HSS derating assumed); E 210 GPa (HSS); floor 0.0003 in (Datron 1.5 mm); LOC 1/8 in, hangout 0.625 in (tool sheet); assumed: 393 sfm (HSS), helix 30 deg, core 0.8 D"));
        }
    }
    for s in &mut list {
        if let Some(dl) = o.defl_in {
            s.model.defl_max = dl * IN;
        }
        if s.number != t4.number && o.hangout_scale_small != 1.0 {
            s.model.hangout = (s.model.loc + 1.0).max(s.model.hangout * o.hangout_scale_small);
            s.max_depth = s.model.hangout - HOLDER_MARGIN;
        }
        s.env = if o.envelope { envelope(vendor, s.d_in) } else { Vec::new() };
    }
    list.par_iter_mut().for_each(|s| s.build_table(64));
    Ok(list)
}
