//! The finishing cutters' physical models, built on icam-motion's CutModel. CutModel's
//! aluminum factory carries only the 1/2 in 2-flute, so the 1/4 in 3-flute is built here
//! field by field. Every constant names its source; "assumed" marks the ones no source on
//! disk pins.

use icam_motion::CutModel;
use icam_post::ToolDef;

use crate::Settings;

const IN: f64 = 25.4;
const HP: f64 = 745.7;

/// 6061-T6 with a 1/4 in 3-flute on the router.
pub fn quarter_3fl(t: &ToolDef, s: &Settings) -> Result<CutModel, String> {
    if (t.diameter - 6.35).abs() > 0.01 {
        return Err(format!("T{}: expected the 1/4 in 3-flute, got {:.3} mm", t.number, t.diameter));
    }
    let mut m = CutModel::blank("6061-T6, 1/4 in carbide 3-flute (T06), MultiCAM");
    // Tool sheet: T06 ".25 Endmill 3 Flute", hangout 1.500 in.
    m.d = t.diameter;
    m.flutes = 3.0;
    m.loc = s.loc06_in * IN;
    m.hangout = s.hangout06_in * IN;
    m.helix_deg = s.helix06;
    m.core_ratio = 0.8; // assumed: the fluted section bends like a solid rod of 0.8 D
    m.e_modulus = 600_000.0; // N/mm2, cemented carbide (typical 550-650 GPa)
    // Chip ceiling: Amana #46590 SC flat bottom 3-flute 1/4 in, aluminum 0.0025-0.0035 in
    // (Amana's published chip loads): the band's middle.
    m.h_ceil = 0.0030 * IN;
    // Floor: the thinnest chip Datron publishes for a 6 mm cutter in aluminum (ae 0.1 mm,
    // ap 29 mm, fz 0.00098 in at 30,000 rpm -> h_max 0.00025 in).
    m.h_floor = 0.00025 * IN;
    // Specific cutting energy: Kp 0.25 hp/(in3/min), the unit power commonly published for
    // aluminum alloys, at a 0.010 in chip (assumed), Kienzle exponent 0.25 (Sandvik, aluminum
    // groups).
    m.kc_ref = 0.25 * HP / (IN.powf(3.0) / 60.0) * 1000.0;
    m.h_ref = 0.010 * IN;
    m.mc = 0.25;
    m.kr = 0.3; // assumed: radial / tangential force ratio for aluminum
    // Machine: MultiCAM 3000, 24,000 rpm, 13.6 hp, 1,000 ipm cutting (its specification).
    m.rpm = s.rpm06;
    m.spindle_w = 13.6 * HP;
    m.eff = 0.85; // assumed
    m.v_max = 1000.0 * IN / 60.0;
    m.defl_max = s.tol_in * IN;
    m.sources = format!(
        "chip 0.0025-0.0035 in (Amana #46590 3FL 1/4 in, aluminum); floor 0.00025 in (Datron 6 mm, thinnest published); Kp 0.25 hp/(in3/min) (published unit power, aluminum alloys); hangout {:.3} in (tool sheet 1.500); machine 13.6 hp / 1000 ipm / 24,000 rpm (its specification); assumed: LOC {:.2} in, helix {:.0} deg, core 0.8 D, Kr 0.3, Eff 0.85, rpm {:.0}, kc at a 0.010 in chip, tolerance {:.4} in",
        s.hangout06_in, s.loc06_in, s.helix06, s.rpm06, s.tol_in
    );
    Ok(m)
}

/// The finisher's model for its tool, at the wall's axial depth, with the finishing
/// tolerance as its deflection limit.
pub fn finisher(t: &ToolDef, ap: f64, s: &Settings) -> Result<CutModel, String> {
    let m = if (t.diameter - 6.35).abs() < 0.01 {
        quarter_3fl(t, s)?
    } else {
        // The 1/2 in tool finishing its own shallow step: the roughing model with the
        // finishing tolerance instead of its 0.001 in roughing limit.
        let mut m = CutModel::al6061(t, Some(s.rpm04))?;
        m.defl_max = s.tol_in * IN;
        m.sources = m.sources.replace("deflection 0.001 in", &format!("tolerance {:.4} in", s.tol_in));
        m
    };
    Ok(m.at_depth(ap))
}

/// The rougher as the roughing planner models it, for the roughing-time estimate.
pub fn rougher(t: &ToolDef, ap: f64, s: &Settings) -> Result<CutModel, String> {
    Ok(CutModel::al6061(t, Some(s.rpm04))?.at_depth(ap))
}
