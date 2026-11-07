//! The entry loop's search: the loop it finds must come within three blends of the least
//! weighed descent time a fine brute-force search finds under the same rules (no face-only
//! core where the room allows, else its weight), and must move continuously as the inputs
//! move. The bore a loop cuts is held to a count on a grid against the loop sampled
//! densely, and the depth the helix leaves to a search of when the tool last covered each
//! point.

use icam_motion::CutModel;
use icam_post::ToolDef;
use icam_strategy::entry::{bore, helix_depth, least, loop_ap, Descent, LoopAp, BLEND, FACE_ONLY_WEIGHT};

fn descent(v: f64, ap_stable: f64) -> Descent {
    Descent { dz: 19.05, tan_ramp: 15f64.to_radians().tan(), ap: LoopAp::uniform(ap_stable), v, accel: 2500.0, r: 6.35 }
}

/// Room shrinking along a ridge 60 mm long: 6 mm at its middle, 1 mm at its ends.
fn cap(run: f64) -> f64 {
    (0.9 * 6.35f64).min(6.0 - 5.0 * (run / 60.0).powi(2) - 0.25)
}

#[test]
fn the_search_comes_near_the_least_time() {
    for (v, ap) in [(34.0, f64::INFINITY), (200.0, 3.0), (400.0, 2.0), (600.0, f64::INFINITY), (600.0, 1.0)] {
        let d = descent(v, ap);
        let weighed = |rho: f64, run: f64| d.time(rho, run) * (bore(rho, run, d.r).face_only / FACE_ONLY_WEIGHT).exp();
        let (run, rho, t) = least(&d, 60.0, cap, 0.635);
        let n = 400;
        let mut brute = f64::INFINITY;
        for i in 0..=n {
            let run = 60.0 * i as f64 / n as f64;
            let c = cap(run);
            let lo = 0.635f64.min(c).max(d.r - 0.5 * run).min(c);
            for j in 0..=n {
                brute = brute.min(weighed(lo + (c - lo) * j as f64 / n as f64, run));
            }
        }
        let got = weighed(rho, run);
        println!("v {v} ap {ap}: run {run:.3} rho {rho:.3} time {t:.4} s, face only {:.4}, weighed {got:.4} s, grid {brute:.4} s", bore(rho, run, d.r).face_only);
        assert!(got <= brute * (1.0 + 3.0 * BLEND), "v {v} ap {ap}: {got} s against {brute} s on the grid");
    }
}

#[test]
fn the_loop_moves_continuously_with_the_feed() {
    for ap in [f64::INFINITY, 2.0] {
        let mut last: Option<(f64, f64, f64)> = None;
        let mut worst = 0.0f64;
        for k in 0..=2000 {
            let v = 50.0 + 550.0 * k as f64 / 2000.0;
            let (run, rho, _) = least(&descent(v, ap), 60.0, cap, 0.635);
            if let Some((v0, run0, rho0)) = last {
                worst = worst.max((run - run0).abs().max((rho - rho0).abs()) / (v - v0));
            }
            last = Some((v, run, rho));
        }
        println!("ap {ap}: the loop moves at most {worst:.4} mm per mm/s of feed");
        assert!(worst < 1.0, "ap {ap}: the loop jumps {worst:.3} mm per mm/s of feed");
    }
}

/// The face-only core and the chips' gap of a loop, counted on a 0.05 mm grid: a point is
/// in the bore when some loop point is within r of it, in the core when every loop point
/// is (the loop sampled every 0.025 mm).
#[test]
fn the_bore_matches_a_count_on_a_grid() {
    let r = 6.35;
    for (rho, run) in [(0.635, 0.0), (3.175, 0.0), (5.715, 0.0), (1.0, 4.0), (2.0, 9.0), (0.0, 5.0), (0.3, 0.0)] {
        // The loop: the run grown by rho, around the run along the x axis from 0 to run.
        let mut lp: Vec<(f64, f64)> = Vec::new();
        let n_side = 1 + (run / 0.025) as usize;
        let n_cap = 1 + (std::f64::consts::PI * rho / 0.025) as usize;
        for k in 0..=n_side {
            let x = run * k as f64 / n_side as f64;
            lp.push((x, rho));
            lp.push((x, -rho));
        }
        for k in 0..=n_cap {
            let a = std::f64::consts::PI * (k as f64 / n_cap as f64 - 0.5);
            lp.push((run + rho * a.cos(), rho * a.sin()));
            lp.push((-rho * a.cos(), rho * a.sin()));
        }
        let h = 0.05;
        let reach = rho + r + h;
        let (mut in_bore, mut in_core) = (0usize, 0usize);
        let mut y = -reach;
        while y <= reach {
            let mut x = -reach;
            while x <= run + reach {
                let (mut near, mut far) = (f64::INFINITY, 0.0f64);
                for p in &lp {
                    let d = (x - p.0).hypot(y - p.1);
                    near = near.min(d);
                    far = far.max(d);
                }
                in_bore += (near <= r) as usize;
                in_core += (far <= r) as usize;
                x += h;
            }
            y += h;
        }
        let (area, core) = (in_bore as f64 * h * h, in_core as f64 * h * h);
        let gap = area - std::f64::consts::PI * r * r;
        let b = bore(rho, run, r);
        println!("rho {rho} run {run}: face only {:.4} (grid {:.4}), bore / gap {:.3} (grid {:.3})", b.face_only, core / area, b.bore_to_gap, area / gap);
        assert!((b.face_only - core / area).abs() < 0.01, "rho {rho} run {run}: face only {} against {}", b.face_only, core / area);
        assert!((b.bore_to_gap / (area / gap) - 1.0).abs() < 0.03, "rho {rho} run {run}: bore / gap {} against {}", b.bore_to_gap, area / gap);
    }
    let b = bore(0.635, 0.0, r);
    assert!((b.face_only - ((r - 0.635) / (r + 0.635)).powi(2)).abs() < 1e-12);
    assert!(bore(0.0, 0.0, r).bore_to_gap.is_infinite() && (bore(0.0, 0.0, r).face_only - 1.0).abs() < 1e-12);
}

/// The depth the helix leaves just outside its edge against a search back in time for when
/// the tool last covered the point (the share of a lap since then).
#[test]
fn the_helix_depth_matches_when_the_tool_last_passed() {
    let r = 6.35;
    for rho in [0.01, 0.635, 2.0, 3.175, 5.715] {
        let mut worst = 0.0f64;
        for k in 0..=60 {
            let b = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 60.0;
            // The tool's center at angle 0 on the circle, going counterclockwise; the point
            // just outside its edge at angle b from that direction, toward the center for b > 0.
            let (px, py) = (rho - (r + 1e-9) * b.sin(), (r + 1e-9) * b.cos());
            let mut back = 1.0;
            for j in 1..=200_000 {
                let s = std::f64::consts::TAU * j as f64 / 200_000.0;
                let (qx, qy) = (rho * (-s).cos(), rho * (-s).sin());
                if (px - qx).hypot(py - qy) <= r {
                    back = s / std::f64::consts::TAU;
                    break;
                }
            }
            worst = worst.max((helix_depth(rho, r, b) - back).abs());
        }
        println!("rho {rho}: closed form within {worst:.5} of the search");
        assert!(worst < 2e-3, "rho {rho}: {worst}");
    }
}

/// The loops' stable drop per lap: on a circle (no run) the depth profile is the same all
/// the way round, so the table must give the closed form's limit; a long run's sides cut
/// like a slot near their far ends, so the drop falls toward the full-slot limit (within
/// a tenth of it at 16 r), never below it, and never above the circle's.
#[test]
fn the_loop_table_holds_the_circle_and_the_slot() {
    let t = ToolDef { number: 1, kind: "ENDMILL".into(), diameter: 12.7, corner_radius: 0.0, description: String::new(), tip_angle_deg: 118.0 };
    let c = CutModel::for_material("al6061", &t, None).unwrap().at_depth(10.0);
    let (tab, r) = (loop_ap(&c), 6.35);
    let slot = c.stable_depth(std::f64::consts::PI);
    for k in 0..=10 {
        let rho = r * k as f64 / 10.0;
        let circle = c.stable_depth_profile(|phi| helix_depth(rho, r, 0.5 * std::f64::consts::PI - phi));
        let at = tab.at(rho, 0.0, r);
        assert!((at / circle - 1.0).abs() < 0.005, "rho {rho}: {at} against the circle's {circle}");
        let long = tab.at(rho, 16.0 * r, r);
        assert!(long >= slot * (1.0 - 1e-9) && long <= circle.min(1.1 * slot), "rho {rho}: a 16 r run gives {long}, the slot {slot}, the circle {circle}");
    }
}
