//! The chatter limit of a cut whose depth varies along the engaged half turn: weighting the
//! directional coefficients by the depth must give the full-slot limit at full depth
//! everywhere, and twice it at half depth everywhere (the limit is on the deepest point,
//! and the coefficients are linear in the weight).

use std::f64::consts::PI;

use icam_motion::CutModel;
use icam_post::ToolDef;

#[test]
fn the_weighted_limit_holds_the_full_slot_limit_and_scales_with_depth() {
    let t = ToolDef { number: 1, kind: "ENDMILL".into(), diameter: 12.7, corner_radius: 0.0, description: String::new(), tip_angle_deg: 118.0 };
    let c = CutModel::for_material("al6061", &t, None).unwrap().at_depth(10.0);
    let slot = c.stable_depth(PI);
    let full = c.stable_depth_profile(|_| 1.0);
    let half = c.stable_depth_profile(|_| 0.5);
    println!("slot {slot:.5} mm, weighted at full depth {full:.5}, at half depth {half:.5}");
    assert!((full / slot - 1.0).abs() < 1e-4, "{full} against {slot}");
    assert!((half / (2.0 * slot) - 1.0).abs() < 1e-4, "{half} against {}", 2.0 * slot);
    // A depth that falls off across the arc can only raise the limit.
    let ramp = c.stable_depth_profile(|phi| phi / PI);
    assert!(ramp > slot, "{ramp} against {slot}");
}
