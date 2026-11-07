//! One STEP file through the importer: each solid's report (faces and edges by kind,
//! vertex and edge offsets), then the exact distance timed at points over the padded
//! box, the slowest few listed with the feature they ended on.
//!
//!   cargo run --release -p icam-check --bin step_probe -- <file> [points]

use std::time::Instant;

use icam_core::curve3::V3;
use icam_core::solid3::{Feature, Solid3};
use icam_core::step;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(args.get(1).expect("usage: step_probe <file> [points]"));
    let n: usize = args.get(2).map_or(20, |s| s.parse().expect("points"));
    let brep = step::read(&path).expect("read");
    println!("{} solids, unit {} ({} mm), tolerance {} mm", brep.solids.len(), brep.unit_name, brep.unit_mm, brep.tolerance_mm);
    let tol = brep.tolerance_mm.max(1e-6);
    let mut solids = Vec::new();
    for s in &brep.solids {
        match Solid3::from_step(s, tol) {
            Ok((sol, rep)) => {
                println!("#{} '{}': faces {:?}, edges {:?}, vertex off {:.2e}, edge off {:.2e}, box {:?} .. {:?}", s.id, s.name, rep.faces, rep.edges, rep.max_vertex_off, rep.max_edge_off, sol.lo, sol.hi);
                solids.push(sol);
            }
            Err(e) => println!("#{} '{}': refused: {e}", s.id, s.name),
        }
    }
    // Trim self-test: from the middle of each edge of each B-spline face, a step into
    // the face (outward normal x travel tangent) should classify inside, a step out of
    // it outside. Counts per face: (in: yes, no, unsure), (out: yes, no, unsure).
    if args.get(3).map(String::as_str) == Some("trim") {
        for s in &solids {
            let diag = (s.hi - s.lo).len();
            for (fi, f) in s.faces.iter().enumerate() {
                let (icam_core::surface3::Surf::Nurbs(ns), Some(tr)) = (&f.surf, s.old_trim(fi)) else { continue };
                let mut c = [[0usize; 3]; 2];
                for u in f.loops.iter().flatten().filter(|u| !f.seams.contains(&u.edge)) {
                    let e = &s.edges[u.edge];
                    let x = e.curve.at(0.5);
                    let cl = e.curve.closest(x);
                    let t = if u.rev { -cl.tangent } else { cl.tangent };
                    let m = f.normal(x).cross(t).unit();
                    let step = (1e-3 * diag).max(20.0 * tr.thr);
                    for (k, sg) in [(0, 1.0), (1, -1.0)] {
                        let Some(h) = ns.closest(x + m * (sg * step), f64::INFINITY) else { continue };
                        c[k][match s.classify_nurbs(fi, h.u, h.v) {
                            Some(true) => 0,
                            Some(false) => 1,
                            None => 2,
                        }] += 1;
                    }
                }
                println!("face #{:6} {:?} spans, {:3} pieces, thr {:.1e}, closed {}/{}: in {:?} out {:?}", f.id, ns.spans_uv(), tr.pieces.len(), tr.thr, tr.closed_u, tr.closed_v, c[0], c[1]);
            }
        }
        return;
    }
    // A point given: each face's share of its distance, slowest first.
    if args.len() >= 6 {
        let p = V3::new(args[3].parse().unwrap(), args[4].parse().unwrap(), args[5].parse().unwrap());
        let s = &solids[0];
        let mut rows = Vec::new();
        for fi in 0..s.faces.len() {
            let c0 = icam_core::solid3::TRIM_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let t = Instant::now();
            let d = s.face_distance(fi, p, f64::INFINITY);
            let calls = icam_core::solid3::TRIM_CALLS.load(std::sync::atomic::Ordering::Relaxed) - c0;
            rows.push((t.elapsed().as_secs_f64(), fi, d, calls));
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (t, fi, d, calls) in rows.iter().take(8) {
            let f = &s.faces[*fi];
            let extra = match (&f.surf, s.old_trim(*fi)) {
                (icam_core::surface3::Surf::Nurbs(ns), Some(tr)) => format!(", {:?} spans, {} trim pieces, thr {:.1e}", ns.spans_uv(), tr.pieces.len(), tr.thr),
                _ => String::new(),
            };
            println!("  {:8.1} ms  face #{} ({}): {d:.5}, {calls} trim tests{extra}", t * 1e3, f.id, f.surf.name());
        }
        return;
    }
    let mut times = Vec::new();
    let mut state = 0x5EED_u64;
    let mut rnd = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for (si, s) in solids.iter().enumerate() {
        let pad = 0.05 * (s.hi - s.lo).len();
        for k in 0..n {
            // Half over the box, half near edges (within 5 % of the box, closer more often).
            let p = if k % 2 == 0 || s.edges.is_empty() {
                V3::new(s.lo.x - pad + rnd() * (s.hi.x - s.lo.x + 2.0 * pad), s.lo.y - pad + rnd() * (s.hi.y - s.lo.y + 2.0 * pad), s.lo.z - pad + rnd() * (s.hi.z - s.lo.z + 2.0 * pad))
            } else {
                let e = &s.edges[(rnd() * s.edges.len() as f64) as usize % s.edges.len()];
                let dir = V3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).unit();
                e.curve.at(rnd()) + dir * (rnd().powi(2) * pad)
            };
            let t = Instant::now();
            let d = s.sdf(p);
            times.push((t.elapsed().as_secs_f64(), si, p, d));
        }
    }
    times.sort_by(|a, b| b.0.total_cmp(&a.0));
    let total: f64 = times.iter().map(|t| t.0).sum();
    println!("{} distances, mean {:.3} ms, slowest:", times.len(), 1e3 * total / times.len().max(1) as f64);
    for (t, si, p, d) in times.iter().take(5) {
        let s = &solids[*si];
        let what = match d.feature {
            Feature::Face(f) => format!("face #{} ({})", s.faces[f].id, s.faces[f].surf.name()),
            Feature::Edge(e) => format!("edge #{}", s.edges[e].id),
            Feature::Vertex(v) => format!("vertex #{}", s.verts[v].id),
        };
        println!("  {:8.1} ms  solid {si} at ({:.3}, {:.3}, {:.3}): d {:.4} on {what}{}", t * 1e3, p.x, p.y, p.z, d.d, if d.unsure { " (unsure)" } else { "" });
    }
}
