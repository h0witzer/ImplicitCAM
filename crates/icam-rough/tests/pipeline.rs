//! The whole job on a part built here: a 200 x 120 sheet, 19.05 thick, with
//!   A  a pocket (10,10)-(90,70) down to z 6 around an island (40,30)-(55,45) at full height
//!   B  a stepped pocket (110,10)-(190,70) down to z 12, with a step (130,25)-(170,55) to z 4
//!   C  a hole (20,85)-(60,110) through the sheet
//! Its floors are 12, 6, 4 and the sheet's bottom, so the ops are found at four slabs, each
//! pocket at the slab of its own floor only (one that goes on below is cut by the slab
//! under it, in one go): B at 12.254, A at 6.254, B's step at 4.254, C at 0.254. They run
//! B, B's step (down to its floor before the tool moves on), A, C.

use icam_core::frep::{compile_prismatic, Frep};
use icam_core::geom::{Curve, V2};
use icam_core::part::Part;
use icam_motion::CutModel;
use icam_post::post::Dialect;
use icam_post::{Box2, PathPt, Tag, ToolDef};
use icam_rough::run::{MAX_UNCUT, RAMP_DEG};
use icam_rough::{run, Job, OpOut, PlanKind};
use icam_strategy::entry;

fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Curve> {
    let p = [V2::new(x0, y0), V2::new(x1, y0), V2::new(x1, y1), V2::new(x0, y1)];
    (0..4).map(|i| Curve::Line { a: p[i], b: p[(i + 1) % 4] }).collect()
}

fn reversed(l: Vec<Curve>) -> Vec<Curve> {
    l.into_iter()
        .rev()
        .map(|c| match c {
            Curve::Line { a, b } => Curve::Line { a: b, b: a },
            other => other,
        })
        .collect()
}

/// An upward face at `z` from loops, each loop's walls reaching down to its `zlo`.
fn up_face(p: &mut Part, id: u32, z: f64, loops: Vec<(Vec<Curve>, f64)>) {
    let n0 = p.edges.len();
    let zlo: Vec<f64> = loops.iter().flat_map(|(l, z)| std::iter::repeat(*z).take(l.len())).collect();
    p.add_face(id, z, loops.into_iter().map(|(l, _)| l).collect(), None, 1e-9).unwrap();
    for (e, z) in p.edges[n0..].iter_mut().zip(zlo) {
        e.wall_zlo = z;
    }
    p.faces.last_mut().unwrap().up = Some(true);
}

fn part() -> Frep {
    part_at(0.0, 0.0, 0.0)
}

/// The part moved by (dx, dy, dz).
fn part_at(dx: f64, dy: f64, dz: f64) -> Frep {
    part_with(dx, dy, dz, 0.0)
}

/// The part moved by (dx, dy, dz), pocket A's right wall moved out by `wall`.
fn part_with(dx: f64, dy: f64, dz: f64, wall: f64) -> Frep {
    let sq = |x0: f64, y0: f64, x1: f64, y1: f64| square(x0 + dx, y0 + dy, x1 + dx, y1 + dy);
    let top = 19.05 + dz;
    let mut p = Part::default();
    up_face(
        &mut p,
        1,
        top,
        vec![
            (sq(0.0, 0.0, 200.0, 120.0), dz),
            (reversed(sq(10.0, 10.0, 90.0 + wall, 70.0)), 6.0 + dz),
            (reversed(sq(110.0, 10.0, 190.0, 70.0)), 12.0 + dz),
            (reversed(sq(20.0, 85.0, 60.0, 110.0)), dz),
        ],
    );
    up_face(&mut p, 2, top, vec![(sq(40.0, 30.0, 55.0, 45.0), 6.0 + dz)]);
    up_face(&mut p, 3, 6.0 + dz, vec![(sq(10.0, 10.0, 90.0 + wall, 70.0), 6.0 + dz), (reversed(sq(40.0, 30.0, 55.0, 45.0)), 6.0 + dz)]);
    up_face(&mut p, 4, 12.0 + dz, vec![(sq(110.0, 10.0, 190.0, 70.0), 12.0 + dz), (reversed(sq(130.0, 25.0, 170.0, 55.0)), 4.0 + dz)]);
    up_face(&mut p, 5, 4.0 + dz, vec![(sq(130.0, 25.0, 170.0, 55.0), 4.0 + dz)]);
    p.add_face(6, dz, vec![reversed(sq(0.0, 0.0, 200.0, 120.0)), sq(20.0, 85.0, 60.0, 110.0)], None, 1e-9).unwrap();
    p.faces.last_mut().unwrap().up = Some(false);
    compile_prismatic(&p, 1e-9).unwrap()
}

fn tool() -> ToolDef {
    ToolDef { number: 1, kind: "ENDMILL".into(), diameter: 12.7, corner_radius: 0.0, description: "1/2 FLAT".into(), tip_angle_deg: 118.0 }
}

#[test]
fn every_op_is_found_cut_and_clean() {
    let frep = part();
    let job = Job::new(tool(), PlanKind::MaxCut);
    let out = run(&frep, Box2::new(0.0, 0.0, 200.0, 120.0), &job).unwrap();
    for o in &out.ops {
        let c = o.check.as_ref();
        println!(
            "{:<22} skipped {:?} gouges {:?} worst {:?} uncut {:?} wall {:?} posted {:?} s",
            o.spec.name,
            o.skipped,
            c.map(|c| c.gouges),
            c.map(|c| c.max_gouge),
            c.map(|c| c.residue_area),
            c.map(|c| c.wall_median),
            o.posted.as_ref().map(|m| m.time())
        );
    }
    let slabs: Vec<String> = out.ops.iter().map(|o| format!("{:.3}", o.spec.z_cut)).collect();
    assert_eq!(slabs, ["12.254", "4.254", "6.254", "0.254"]);
    for o in &out.ops {
        assert!(o.skipped.is_none(), "{}: skipped: {:?}", o.spec.name, o.skipped);
        let c = o.check.as_ref().unwrap();
        assert!(c.part_points > 0, "{}: no design points", o.spec.name);
        assert_eq!(c.gouges, 0, "{}: {} gouges, worst {:.3} mm", o.spec.name, c.gouges, c.max_gouge);
        assert!(c.residue_area <= MAX_UNCUT, "{}: {:.1} mm2 uncut", o.spec.name, c.residue_area);
        assert!(c.prior_residue_area > c.residue_area, "{}: the op removed nothing", o.spec.name);
    }
    let moves = icam_post::post::Writer::read_back(&out.program, out.origin, &mut Vec::new());
    assert!(moves.len() > 100);
    assert!(moves.iter().all(|m| m.to.z >= 0.254 - 1e-6), "a move goes below the lowest slab");
}

#[test]
fn multicam_program_reads_back_the_same() {
    let frep = part();
    let mut job = Job::new(tool(), PlanKind::MaxCut);
    job.dialect = Dialect::MultiCam;
    job.stepover = Some(9.0);
    let out = run(&frep, Box2::new(0.0, 0.0, 200.0, 120.0), &job).unwrap();
    assert!(out.ops.iter().all(|o| o.skipped.is_none() && !o.gouged()));
    let mut w = Vec::new();
    let moves = icam_post::post::Writer::read_back(&out.program, out.origin, &mut w);
    assert!(w.iter().any(|w| w.contains("counts Z downward")), "{w:?}");
    // Inches to four decimals: every height within 0.0001 in / 2 = 0.00127 mm of its value.
    assert!(moves.iter().all(|m| m.to.z >= 0.254 - 0.002 && m.to.z <= 19.05 + 5.0 + 0.002));
}

/// A part far from the origin and below Z zero, as CAD models often are: the program is written
/// in the work frame (Z zero on the table), so it reads back where it was planned.
#[test]
fn a_part_below_zero_is_cut_where_it_is() {
    let (dx, dy, dz) = (-9000.0, -11000.0, -361.0);
    let frep = part_at(dx, dy, dz);
    let job = Job::new(tool(), PlanKind::MaxCut);
    let out = run(&frep, Box2::new(dx, dy, 200.0 + dx, 120.0 + dy), &job).unwrap();
    assert_eq!(out.ops.len(), 4);
    for o in &out.ops {
        let c = o.check.as_ref().unwrap();
        assert_eq!(c.gouges, 0, "{}", o.spec.name);
        assert!(c.residue_area <= MAX_UNCUT, "{}: {:.1} mm2 uncut", o.spec.name, c.residue_area);
    }
    let first_rapid_z = out.program.lines().find_map(|l| l.strip_prefix("G00 Z")).and_then(|z| z.parse::<f64>().ok()).unwrap();
    assert!(first_rapid_z > 0.0, "the program's heights are measured from the table");
}

#[test]
fn finishing_and_assignment_run_on_the_field_pockets() {
    use icam_assign::{ModelOptions, Penalties};
    use icam_finish::Settings;
    use icam_rough::assign::{assign_pockets, AssignJob};
    use icam_rough::finish::study_pockets;

    let frep = part();
    let job = Job::new(tool(), PlanKind::MaxCut);
    let out = run(&frep, Box2::new(0.0, 0.0, 200.0, 120.0), &job).unwrap();

    // Studied where each pocket ends: A at 6, B at 12, B's step at 4, C through the
    // sheet; the slabs a pocket only passes through are skipped.
    let mut s = Settings::default();
    s.fit_levels();
    let ft = ToolDef { number: 2, kind: "ENDMILL".into(), diameter: 6.35, corner_radius: 0.0, description: "1/4 FLAT".into(), tip_angle_deg: 118.0 };
    let fin = study_pockets(&frep, &job, &out, &ft, &s).unwrap();
    let studied: Vec<(f64, usize, usize)> = fin
        .iter()
        .filter_map(|f| f.pocket.as_ref().map(|pk| (f.floor, pk.loops.len(), pk.loops.iter().filter(|l| l.island).count())))
        .collect();
    let mut floors: Vec<f64> = studied.iter().map(|s| s.0).collect();
    floors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(floors, vec![0.0, 4.0, 6.0, 12.0], "studied floors: {studied:?}");
    // A's island is a wall loop of its own.
    assert!(studied.iter().any(|&(f, loops, isl)| f == 6.0 && loops == 2 && isl == 1), "{studied:?}");
    for f in fin.iter().filter(|f| f.result.is_some()) {
        let r = f.result.as_ref().unwrap();
        // No field beats the thinnest uniform allowance, and the fastest field is at least as fast as (a).
        assert!(r.fast.sched().time <= r.a_sched.time + 1e-9, "{}: (b) {} s vs (a) {} s", f.op, r.fast.sched().time, r.a_sched.time);
        assert!(r.thin_sched.time <= r.fast.sched().time + 1e-6, "{}", f.op);
    }

    // Every op is a layer; nothing goes uncut that some tool reaches, and paying for
    // tool changes never costs more than the cheapest plan.
    let a = AssignJob {
        sheet: vec![tool(), ToolDef { number: 2, kind: "ENDMILL".into(), diameter: 6.35, corner_radius: 0.0, description: "1/4".into(), tip_angle_deg: 118.0 }],
        options: ModelOptions { envelope: false, ..ModelOptions::default() },
        vendor: None,
        cell: 0.5,
        penalties: Penalties::default(),
    };
    let asg = assign_pockets(&frep, &job, &out, &a).unwrap();

    // Both pages carry their sections and their data; ICAM_REPORT_DIR keeps them for a look.
    let fpage = icam_rough::finish_report::html("synthetic part", &fin, &s, 12.7, 1.0);
    let apage = icam_rough::assign_report::html("synthetic part", &asg, &a, &job, 1.0);
    for key in ["Finish allowance field", "\"pockets\":[{", "Z6.254 POCKET", "skipped"] {
        assert!(fpage.contains(key), "finish page lacks {key}");
    }
    for key in ["Tool assignment", "How far from flipping", "class=\"map\"", "\"layers\":{\"base\":["] {
        assert!(apage.contains(key), "assign page lacks {key}");
    }
    assert!(!fpage.contains("__DATA__") && !apage.contains("__BODY__") && !apage.contains("__DATA__"));
    if let Ok(d) = std::env::var("ICAM_REPORT_DIR") {
        std::fs::write(std::path::Path::new(&d).join("finish.html"), &fpage).unwrap();
        std::fs::write(std::path::Path::new(&d).join("assign.html"), &apage).unwrap();
    }
    assert_eq!(asg.layers.len() + asg.empty.len(), out.ops.len());
    assert!(!asg.layers.is_empty());
    for l in &asg.layers {
        assert!(l.rest.uncut_area <= 25.0, "{}: {} mm2 uncut", l.op, l.rest.uncut_area);
        assert!(l.layer.mat_area() > 0.0);
    }
    assert!(asg.nest.2 <= asg.nest.1 + 1e-9, "{:?}", asg.nest);
}

#[test]
fn the_bench_scores_a_baseline_on_the_same_part() {
    use icam_rough::bench::{bench, score, Baseline};
    use std::collections::BTreeMap;

    // A baseline we know: our own program, its ops renamed as a CAM program names them.
    let frep = part();
    let job = Job::new(tool(), PlanKind::MaxCut);
    let first = run(&frep, Box2::new(0.0, 0.0, 200.0, 120.0), &job).unwrap();
    let mut moves = first.moves.clone();
    for (k, o) in first.ops.iter().enumerate() {
        for m in &mut moves[o.range.0..o.range.1] {
            m.op = format!("ROUGH MILL{}", k + 1);
        }
    }
    let base = Baseline { moves, tools: BTreeMap::from([(1, tool())]), warnings: Vec::new() };
    let b = bench(&frep, Box2::new(0.0, 0.0, 200.0, 120.0), &job, &base).unwrap();
    assert!(!b.rough_ops.is_empty() && b.levels.len() >= b.rough_ops.len());
    assert!(b.inside < 0.02, "the baseline's cuts read as inside the part: {}", b.inside);
    assert!(b.phi_ref > 0.0 && b.phi_ref < std::f64::consts::PI && b.v_prog > 0.0);
    // The baseline passed its own checks, and so does the field roughing held to its load.
    for g in &b.gates {
        assert_eq!(g.base.gouges, 0, "{}", g.op);
        assert!(g.base.residue_area <= 5.0, "{}: {} mm2", g.op, g.base.residue_area);
    }
    let rows = score(&b, false);
    assert!(rows[0].pass && rows[1].pass, "{}: {} / {}: {}", rows[0].objective, rows[0].ours, rows[1].objective, rows[1].ours);
    let page = icam_rough::bench_report::html("synthetic part", "itself", &b, None, 12.7, 1.0);
    assert!(page.contains("Roughing scoreboard") && page.contains("objectives met"));
    if let Ok(d) = std::env::var("ICAM_REPORT_DIR") {
        std::fs::write(std::path::Path::new(&d).join("bench.html"), &page).unwrap();
        for r in &rows {
            eprintln!("{} | {} | {} | {} | {}", r.objective, r.bar, r.base, r.ours, r.pass);
        }
    }
}

/// Each entry's descents: runs of consecutive entry points.
fn entries(o: &OpOut) -> Vec<Vec<PathPt>> {
    let mut runs: Vec<Vec<PathPt>> = Vec::new();
    let mut open = false;
    for p in o.path.as_ref().map_or(&[][..], |p| &p.points[..]) {
        if p.tag != Tag::Entry {
            open = false;
            continue;
        }
        if !open {
            runs.push(Vec::new());
            open = true;
        }
        runs.last_mut().unwrap().push(*p);
    }
    runs
}

/// An entry loop's shape read back from its points: (radius, run). The radius is half the
/// least width of the points (rotating calipers over their convex hull), the run what is
/// left of one lap's length after the turns. Read from points every 0.5 mm on the loop,
/// both are within about half a percent of the loop's.
fn loop_shape(e: &[PathPt]) -> (f64, f64) {
    let rho = loop_radius(e);
    let mut along = 0.0;
    for j in 1..e.len() {
        along += (e[j].x - e[j - 1].x).hypot(e[j].y - e[j - 1].y);
        if along > 1e-6 && (e[j].x - e[0].x).hypot(e[j].y - e[0].y) < 1e-6 {
            return (rho, (0.5 * (along - 2.0 * std::f64::consts::PI * rho)).max(0.0));
        }
    }
    (rho, 0.0)
}

/// Half the least width of an entry's points.
fn loop_radius(e: &[PathPt]) -> f64 {
    let mut pts: Vec<(f64, f64)> = e.iter().map(|p| (p.x, p.y)).collect();
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup_by(|a, b| (a.0 - b.0).hypot(a.1 - b.1) < 1e-9);
    if pts.len() < 3 {
        return 0.0;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let it: Vec<(f64, f64)> = if pass == 0 { pts.clone() } else { pts.iter().rev().copied().collect() };
        for p in it {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    let width = (0..hull.len())
        .map(|i| {
            let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
            let len = (b.0 - a.0).hypot(b.1 - a.1).max(1e-12);
            hull.iter().map(|&p| cross(a, b, p).abs() / len).fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    0.5 * width
}

/// Every entry ramps no steeper than the tool's limit and goes no deeper per lap than the
/// cut its loop takes stays stable at, measured on the path: the slope between consecutive
/// points, and the drop between a point and the next one at the same place after a lap,
/// against the stable drop for the loop's shape read back from it (within half a percent).
#[test]
fn entries_stay_inside_the_ramp_and_the_stable_depth() {
    let frep = part();
    let tan_max = RAMP_DEG.to_radians().tan();
    for kind in [PlanKind::MaxCut, PlanKind::Conductive] {
        let mut job = Job::new(tool(), kind);
        job.material = Some("al6061".into());
        let out = run(&frep, Box2::new(0.0, 0.0, 200.0, 120.0), &job).unwrap();
        let mut laps = 0;
        for o in &out.ops {
            let c = CutModel::for_material("al6061", &tool(), job.rpm).unwrap().at_depth(o.ap);
            let r = tool().diameter / 2.0;
            let table = entry::loop_ap(&c);
            for e in entries(o) {
                let (rho, run) = loop_shape(&e);
                let ap_stable = table.at(rho, run, r);
                assert!((e.last().unwrap().z - o.spec.z_cut).abs() < 1e-9, "{kind:?} {}: the entry ends above the cut", o.spec.name);
                for w in e.windows(2) {
                    let run = (w[1].x - w[0].x).hypot(w[1].y - w[0].y);
                    if run > 1e-9 {
                        assert!(w[0].z - w[1].z <= tan_max * run + 1e-9, "{kind:?} {}: ramp steeper than {RAMP_DEG} deg", o.spec.name);
                    }
                }
                for i in 0..e.len() {
                    let mut along = 0.0;
                    for j in i + 1..e.len() {
                        along += (e[j].x - e[j - 1].x).hypot(e[j].y - e[j - 1].y);
                        if along > 1e-6 && (e[j].x - e[i].x).hypot(e[j].y - e[i].y) < 1e-6 {
                            assert!(e[i].z - e[j].z <= 1.005 * ap_stable + 1e-9, "{kind:?} {}: {:.2} mm in one lap, stable {ap_stable:.2}", o.spec.name, e[i].z - e[j].z);
                            laps += 1;
                            break;
                        }
                    }
                }
            }
        }
        assert!(laps > 100, "{kind:?}: only {laps} laps checked");
    }
}

/// The continuity gate for the entries alone: a 5 um change to a wall moves no entry by
/// more than 1 mm (every point of each op's entry within 1 mm of the other run's entry,
/// both ways).
#[test]
fn entries_move_little_when_a_wall_moves() {
    let bx = Box2::new(0.0, 0.0, 200.0, 120.0);
    let (a, b) = (part(), part_with(0.0, 0.0, 0.0, 0.005));
    for kind in [PlanKind::MaxCut, PlanKind::Conductive] {
        let job = Job::new(tool(), kind);
        let (oa, ob) = (run(&a, bx, &job).unwrap(), run(&b, bx, &job).unwrap());
        assert_eq!(oa.ops.len(), ob.ops.len());
        let mut worst = 0.0f64;
        for (p, q) in oa.ops.iter().zip(&ob.ops) {
            assert_eq!(p.spec.name, q.spec.name);
            let (ep, eq): (Vec<PathPt>, Vec<PathPt>) = (entries(p).concat(), entries(q).concat());
            assert!(!ep.is_empty() && !eq.is_empty(), "{kind:?} {}: no entry", p.spec.name);
            let far = |u: &[PathPt], v: &[PathPt]| u.iter().map(|s| v.iter().map(|t| (s.x - t.x).hypot(s.y - t.y).hypot(s.z - t.z)).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max);
            worst = worst.max(far(&ep, &eq)).max(far(&eq, &ep));
        }
        println!("{kind:?}: entries moved at most {worst:.4} mm");
        assert!(worst <= 1.0, "{kind:?}: an entry moved {worst:.3} mm");
    }
}
