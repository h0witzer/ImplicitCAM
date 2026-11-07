//! icam-check exact: the exact evaluator (`solid3::Exact`) checked without the old
//! B-rep distance as a reference.
//!
//! Per solid, queries of three kinds: uniform in the solid's box grown by a fifth
//! each way, near the faces (dense face points moved by random vectors of
//! log-uniform length), and near the edges (edge points moved alike). Each answer
//! is checked against:
//! - a dense cloud of boundary points (each face's parameter grid inside its
//!   certified loops, edge points, vertices): no cloud point may be nearer than the
//!   answer by more than the file's tolerance (the loops reach past their edges by
//!   up to it). A nearer point is a missed feature;
//! - the half-segment rule: the point halfway from p to its nearest point has half
//!   the distance with the same sign and the same nearest point, exactly;
//! - the 1-Lipschitz bound on random nearby pairs, and a positive value outside
//!   the solid's box;
//! - surfcheck's samples (points well clear of the edges): zero on the surface,
//!   plus and minus delta at delta along the normal.
//!
//! The band: the file's tolerance plus the loops' largest certified deviation from
//! their edges; failures inside it are counted apart, and failures beyond it among
//! answers not marked unsure are the ones nothing flags.
//!
//! `--old` also compares 200 queries with the old B-rep distance; `--no-test`
//! turns the critical-point test off in the B-spline searches; `--raw` reads the
//! file without analytic recognition, so every face as written; `--profile` times
//! the parts of a query on one thread.
//!
//! `--at=x,y,z` explains one point instead: the answer, the faces near it with
//! their feet and both trim tests, grid points of a face nearer than the answer,
//! and for a face or edge answer its edges and points further out. Environment:
//! ICAM_EXPLAIN (the loop test's reasoning at each foot), ICAM_OLD (the old
//! distance there), ICAM_EDGES (every edge within the answer, searched and
//! sampled), ICAM_FACE=index (that face's loops, and with ICAM_UV=u,v its loop test
//! at those parameters), ICAM_PROBE=index (a B-spline face's search replayed by
//! depth), ICAM_COVER=i,j (which faces' loops and old tests claim a grid on face
//! i's surface).

use std::path::Path;

use icam_core::curve3::V3;
use icam_core::solid3::{Exact, Feature, Solid3};
use icam_core::step;
use icam_core::surface3::Surf;
use rayon::prelude::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn dir(&mut self) -> V3 {
        loop {
            let v = V3::new(2.0 * self.next() - 1.0, 2.0 * self.next() - 1.0, 2.0 * self.next() - 1.0);
            let l = v.len();
            if l > 1e-3 && l <= 1.0 {
                return v * (1.0 / l);
            }
        }
    }
}

/// Boundary points of a solid: each face's parameter grid inside its certified
/// loops, points along each edge, the vertices. Faces without loops add none.
fn cloud(sol: &Solid3, grid: usize) -> Vec<V3> {
    let mut pts: Vec<V3> = (0..sol.faces.len())
        .into_par_iter()
        .flat_map_iter(|fi| {
            let f = &sol.faces[fi];
            let mut out = Vec::new();
            let Some(t) = &f.trim2 else { return out };
            let ns = match (&f.surf, &f.analytic2) {
                (Surf::Nurbs(ns), _) => ns.as_ref(),
                (_, Some(a)) => &a.ns,
                _ => return out,
            };
            let n = if matches!(f.surf, Surf::Nurbs(_)) { grid } else { grid / 2 };
            for i in 0..=n {
                for j in 0..=n {
                    let u = ns.u_range.0 + (ns.u_range.1 - ns.u_range.0) * i as f64 / n as f64;
                    let v = ns.v_range.0 + (ns.v_range.1 - ns.v_range.0) * j as f64 / n as f64;
                    if t.inside(u, v) {
                        out.push(ns.eval(u, v).p);
                    }
                }
            }
            out
        })
        .collect();
    for e in &sol.edges {
        for k in 0..=64 {
            pts.push(e.curve.at(k as f64 / 64.0));
        }
    }
    pts.extend(sol.verts.iter().map(|v| v.p));
    pts
}

/// The trimmed search's steps replayed for one face, counted by depth: patches
/// popped, wholly in, straddling, cleared by the critical-point test, split.
fn probe(ns: &icam_core::surface3::NurbsSurf, t: &icam_core::trim2::Trim2, p: V3, c: &icam_core::surface3::Creases) {
    use icam_core::surface3::{box_dist, slab_bound};
    let mut best = f64::INFINITY;
    // Seed from every span patch (no node tree), best first by box distance.
    let (nu, nv) = (ns.spans_u(), ns.spans_v());
    let mut heap: Vec<(f64, icam_core::surface3::Patch, usize)> = Vec::new();
    for a in 0..nu {
        for b in 0..nv {
            let pa = ns.patch(a, b);
            heap.push((box_dist(p, &pa.bbox()), pa, 0));
        }
    }
    let mut rows = vec![[0usize; 6]; 41];
    let mut pops = 0;
    let mut kept: Vec<String> = Vec::new();
    while !heap.is_empty() && pops < 20000 {
        let k = (0..heap.len()).min_by(|&i, &j| heap[i].0.total_cmp(&heap[j].0)).unwrap();
        let (lb, pa, depth) = heap.swap_remove(k);
        if lb >= best - 1e-9 {
            break;
        }
        pops += 1;
        let r = &mut rows[depth];
        r[0] += 1;
        let state = t.box_class([pa.u0, pa.v0], [pa.u1, pa.v1]);
        if state == Some(false) {
            r[1] += 1;
            continue;
        }
        if state.is_none() {
            r[2] += 1;
        }
        let (s, tt) = pa.newton_capped(p, 0.5, 0.5, 12);
        let e = pa.eval(s, tt);
        let d = (e.p - p).len();
        let (u, v) = (pa.u0 + s * (pa.u1 - pa.u0), pa.v0 + tt * (pa.v1 - pa.v0));
        if d < best && t.inside(u, v) {
            best = d;
        }
        let n = e.ps.cross(e.pt).unit();
        let slab = slab_bound(&pa, p, if d > 1e-12 { (p - e.p) * (1.0 / d) } else { n });
        if slab >= best - 1e-9 || depth >= 40 {
            r[3] += 1;
            continue;
        }
        if !c.touches(&pa) && pa.no_critical_point(p) {
            r[4] += 1;
            continue;
        }
        r[5] += 1;
        if depth >= 12 && kept.len() < 6 {
            kept.push(format!("depth {depth} u {:.9}..{:.9} v {:.9}..{:.9} lb {lb:.3e} slab {slab:.3e} best {best:.3e} state {state:?} touches crease {}", pa.u0, pa.u1, pa.v0, pa.v1, c.touches(&pa)));
        }
        for q in pa.quarters() {
            let lbq = box_dist(p, &q.bbox()).max(slab);
            if lbq < best - 1e-9 {
                heap.push((lbq, q, depth + 1));
            }
        }
    }
    println!("      probe: {pops} steps, best {best:.6e}; by depth (popped, outside, straddling, bounded, cleared by the test, split):");
    for (dpt, r) in rows.iter().enumerate().filter(|(_, r)| r[0] > 0) {
        println!("        {dpt:>2}: {:?}", r);
    }
    for k in kept {
        println!("        {k}");
    }
}

/// Distance from p to face fi's untrimmed surface (a lower bound on the face's).
pub(crate) fn face_lb(sol: &Solid3, fi: usize, p: V3) -> f64 {
    (sol.faces[fi].surf.foot(p) - p).len()
}

fn nearest(cloud: &[V3], p: V3) -> f64 {
    cloud.iter().fold(f64::INFINITY, |a, &c| { let d = c - p; a.min(d.dot(d)) }).sqrt()
}

#[derive(Default)]
struct Kind {
    n: usize,
    /// A cloud point nearer than the answer by over the file's tolerance.
    missed: usize,
    worst_miss: f64,
    /// The half-segment rule broken: distance, sign; and beyond the band (the
    /// file's tolerance plus the loops' certified deviation from their edges, where
    /// a face's answer next to its edge may differ from the edge's).
    half_off: usize,
    half_sign: usize,
    half_off_band: usize,
    half_sign_band: usize,
    secs: f64,
    pops: usize,
    max_pops: usize,
    capped: usize,
    uncertified: usize,
    /// Failures beyond the band (missed, half-segment) among answers not marked
    /// unsure: the wrong answers nothing flags.
    unflagged: usize,
    features: usize,
    by_feature: [usize; 3],
}

/// `icam-check exact <file.step | dir> [queries per kind]`.
pub fn run(path: &Path, per_kind: usize, args: &[String]) {
    let old = args.iter().any(|a| a == "--old");
    let test = !args.iter().any(|a| a == "--no-test");
    let raw = args.iter().any(|a| a == "--raw");
    let files: Vec<std::path::PathBuf> = if path.is_dir() {
        let mut f: Vec<_> = std::fs::read_dir(path)
            .expect("read dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| matches!(x.to_string_lossy().to_lowercase().as_str(), "step" | "stp")))
            .collect();
        f.sort();
        f
    } else {
        vec![path.to_path_buf()]
    };
    println!("exact evaluator: {per_kind} queries per kind per solid; critical-point test {}; {} reading", if test { "on" } else { "off" }, if raw { "raw" } else { "importer's" });
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let brep = match if raw { step::read_raw(f) } else { step::read(f) } {
            Ok(b) => b,
            Err(e) => {
                println!("{name}: read failed: {e}");
                continue;
            }
        };
        let tol = brep.tolerance_mm.max(1e-6);
        for sr in &brep.solids {
            let t0 = std::time::Instant::now();
            let (sol, rep) = match Solid3::from_step(sr, tol) {
                Ok(x) => x,
                Err(e) => {
                    println!("{name}: solid #{} refused: {e}", sr.id);
                    continue;
                }
            };
            let build = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let ex = Exact::with_test(&sol, test);
            let ex_secs = t1.elapsed().as_secs_f64();
            let uncert_faces = sol.faces.iter().filter(|f| f.trim2.is_none()).count();
            // The band beside the edges: the file's tolerance plus the loops' largest
            // certified deviation from their edges (which includes the file's own
            // edge offsets and vertex gaps).
            let band = tol + rep.trim2_bound;
            let cl = cloud(&sol, 48);
            let diag = (sol.hi - sol.lo).len();
            // Queries.
            let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ sr.id as u64);
            let (lo, hi) = (sol.lo, sol.hi);
            let ext = hi - lo;
            let uniform: Vec<V3> = (0..per_kind).map(|_| V3::new(lo.x + ext.x * (1.4 * rng.next() - 0.2), lo.y + ext.y * (1.4 * rng.next() - 0.2), lo.z + ext.z * (1.4 * rng.next() - 0.2))).collect();
            let nedge = sol.edges.len() * 65;
            let face_pts = cl.len() - nedge - sol.verts.len();
            let moved = |base: V3, rng: &mut Rng| base + rng.dir() * (diag * 10f64.powf(-4.0 + 3.0 * rng.next()));
            let near_face: Vec<V3> = (0..per_kind).filter(|_| face_pts > 0).map(|_| { let b = cl[(rng.next() * face_pts as f64) as usize % face_pts.max(1)]; moved(b, &mut rng) }).collect();
            let near_edge: Vec<V3> = (0..per_kind).filter(|_| nedge > 0).map(|_| { let b = cl[face_pts + (rng.next() * nedge as f64) as usize % nedge]; moved(b, &mut rng) }).collect();
            println!(
                "{name} solid #{}: {} faces ({} without certified loops), {} edges, {} vertices; import {:.1} s (loops refused {}), evaluator setup {:.2} s; cloud {} points; band {:.1e} mm (tolerance {:.1e}); lemon faces {} (turned over {}, undecided {})",
                sr.id,
                sol.faces.len(),
                uncert_faces,
                sol.edges.len(),
                sol.verts.len(),
                build,
                rep.trim2_refused,
                ex_secs,
                cl.len(),
                band,
                tol,
                rep.lemons,
                rep.lemons_turned,
                rep.lemons_undecided
            );
            if let Some(at) = args.iter().find_map(|a| a.strip_prefix("--at=")) {
                // One point, explained: the answer, then every face whose untrimmed
                // nearest point is within twice the answer.
                let c: Vec<f64> = at.split(',').map(|x| x.parse().unwrap()).collect();
                let p = V3::new(c[0], c[1], c[2]);
                let (sd, st) = ex.sdf(p);
                println!("  at {p:?}: {:.9} at {:?} ({:?}) {st:?}", sd.d, sd.q, sd.feature);
                if std::env::var("ICAM_OLD").is_ok() {
                    let o = sol.sdf(p);
                    println!("    old B-rep distance: {:.9} at {:?} ({:?}) unsure {}", o.d, o.q, o.feature, o.unsure);
                }
                if let Feature::Edge(ei) = sd.feature {
                    let e = &sol.edges[ei];
                    let ns: Vec<String> = e.faces.iter().map(|&g| format!("face {g} #{} {} sense {} normal {:?}", sol.faces[g].id, sol.faces[g].surf.name(), sol.faces[g].sense, sol.faces[g].normal(sd.q))).collect();
                    println!("    edge {ei} #{} {}: {}", e.id, icam_core::solid3::edge_kind_name(&e.curve), ns.join("; "));
                    let h = sd.q + (p - sd.q) * 0.5;
                    let (sh, _) = ex.sdf(h);
                    println!("    halfway {:.9} at {:?} ({:?})", sh.d, sh.q, sh.feature);
                }
                if let Feature::Face(fi) = sd.feature {
                    let dir = (p - sd.q).unit();
                    for far in [10.0, 100.0, 1000.0] {
                        let x = p + dir * far;
                        let (s2, _) = ex.sdf(x);
                        println!("    {far} further out: {:.6} ({:?})", s2.d, s2.feature);
                    }
                    for lp in &sol.faces[fi].loops {
                        for u in lp {
                            let e = &sol.edges[u.edge];
                            let xm = e.curve.at(0.5);
                            let others: Vec<String> = e.faces.iter().filter(|&&g| g != fi).map(|&g| format!("face {g} #{} {} rev {:?} normals dot {:.6}", sol.faces[g].id, sol.faces[g].surf.name(), sol.use_of(g, u.edge).map(|x| x.rev), sol.faces[g].normal(xm).dot(sol.faces[fi].normal(xm)))).collect();
                            println!("    edge {} #{}: face {fi} rev {}; {}; from {:?} to {:?}, mid {:?}", u.edge, e.id, u.rev, others.join(", "), e.curve.start(), e.curve.end(), e.curve.at(0.5));
                        }
                    }
                }
                if std::env::var("ICAM_EDGES").is_ok() {
                    for (ei, e) in sol.edges.iter().enumerate() {
                        let lb = icam_core::surface3::box_dist(p, &(e.lo, e.hi));
                        if lb < sd.d.abs() {
                            let c = e.curve.closest(p);
                            let fine = (0..=4000).map(|k| (e.curve.at(k as f64 / 4000.0) - p).len()).fold(f64::INFINITY, f64::min);
                            println!("    edge {ei} #{} {}: box {:.4}, nearest {:.6} ({:?}), sampled {:.6}, faces {:?}", e.id, icam_core::solid3::edge_kind_name(&e.curve), lb, c.dist, c.near, fine, e.faces);
                        }
                    }
                }
                let cn = cl.iter().fold((f64::INFINITY, V3::default()), |a, &c| if (c - p).len() < a.0 { ((c - p).len(), c) } else { a });
                println!("    nearest cloud point {:.6} at {:?}; refused faces {:?}", cn.0, cn.1, sol.faces.iter().enumerate().filter(|(_, f)| f.trim2.is_none()).map(|(i, f)| (i, f.id, f.surf.name(), crate::exact_check::face_lb(&sol, i, p))).collect::<Vec<_>>());
                if let Feature::Vertex(vi) = sd.feature {
                    let v = &sol.verts[vi];
                    println!("    vertex {vi} #{} pseudo-normal {:?}", v.id, v.pn);
                    for (ei, e) in sol.edges.iter().enumerate().filter(|(_, e)| e.v.contains(&vi)) {
                        let ns: Vec<String> = e.faces.iter().map(|&fi| format!("face {fi} #{} {} normal {:?}", sol.faces[fi].id, sol.faces[fi].surf.name(), sol.faces[fi].normal(v.p))).collect();
                        println!("      edge {ei} #{} {} from {:?} to {:?}: {}", e.id, icam_core::solid3::edge_kind_name(&e.curve), e.curve.start(), e.curve.end(), ns.join("; "));
                        for s_ in [1e-3, 1e-1, 1.0] {
                            let t = if e.v[0] == vi { e.curve.closest(v.p + (e.curve.at(0.01) - v.p).unit() * s_).t } else { e.curve.closest(v.p + (e.curve.at(0.99) - v.p).unit() * s_).t };
                            let _ = t;
                            let x = if e.v[0] == vi { e.curve.at(1e-5 * s_) } else { e.curve.at(1.0 - 1e-5 * s_) };
                            let ns: Vec<String> = e.faces.iter().map(|&fi| format!("{:?}", sol.faces[fi].normal(x))).collect();
                            println!("        near the vertex along it ({:.1e} of the edge): {}", 1e-5 * s_, ns.join("; "));
                        }
                    }
                }
                // Faces with a grid point nearer than the answer, with that point.
                for (fi, f) in sol.faces.iter().enumerate() {
                    let (Some(t), Some(ns)) = (&f.trim2, match (&f.surf, &f.analytic2) {
                        (Surf::Nurbs(ns), _) => Some(ns.as_ref()),
                        (_, Some(a)) => Some(&a.ns),
                        _ => None,
                    }) else { continue };
                    let mut best = (f64::INFINITY, V3::default());
                    for i in 0..=48 {
                        for j in 0..=48 {
                            let u = ns.u_range.0 + (ns.u_range.1 - ns.u_range.0) * i as f64 / 48.0;
                            let v = ns.v_range.0 + (ns.v_range.1 - ns.v_range.0) * j as f64 / 48.0;
                            if t.inside(u, v) {
                                let x = ns.eval(u, v).p;
                                if (x - p).len() < best.0 {
                                    best = ((x - p).len(), x);
                                }
                            }
                        }
                    }
                    if best.0 < sd.d.abs() - 1e-9 {
                        let fb = {
                            let mut b = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
                            for u in f.loops.iter().flatten() {
                                let e = &sol.edges[u.edge];
                                b = (V3::new(b.0.x.min(e.lo.x), b.0.y.min(e.lo.y), b.0.z.min(e.lo.z)), V3::new(b.1.x.max(e.hi.x), b.1.y.max(e.hi.y), b.1.z.max(e.hi.z)));
                            }
                            b
                        };
                        println!("    NEARER grid point of face {fi} #{} {}: {:.6} at {:?}; the face's edge box {:?}", f.id, f.surf.name(), best.0, best.1, fb);
                    }
                }
                if let Some(fi) = std::env::var("ICAM_FACE").ok().and_then(|x| x.parse::<usize>().ok()).filter(|&i| i < sol.faces.len()) {
                    let f = &sol.faces[fi];
                    for (li, lp) in f.loops.iter().enumerate() {
                        for u in lp {
                            let e = &sol.edges[u.edge];
                            let (a, b) = if u.rev { (e.curve.end(), e.curve.start()) } else { (e.curve.start(), e.curve.end()) };
                            let uses: Vec<String> = e.faces.iter().map(|&g| format!("{g}:{:?}", sol.use_of(g, u.edge).map(|x| x.rev))).collect();
                            println!("    loop {li} edge {} #{} {} uses {:?}: {:?} -> {:?} (mid {:?}), off the surface at the middle {:.2e}", u.edge, e.id, icam_core::solid3::edge_kind_name(&e.curve), uses, a, b, e.curve.at(0.5), f.surf.dist(e.curve.at(0.5)));
                        }
                    }
                    if let (Some(t), Ok(uv)) = (&f.trim2, std::env::var("ICAM_UV")) {
                        let c: Vec<f64> = uv.split(',').map(|x| x.parse().unwrap()).collect();
                        println!("    face {fi} at uv ({}, {}): inside {}; {}", c[0], c[1], t.inside(c[0], c[1]), t.explain(c[0], c[1]));
                    }
                    if let (Surf::Nurbs(ns), Some(t)) = (&f.surf, &f.trim2) {
                        if let Some(h) = ns.closest(p, f64::INFINITY) {
                            println!("    face {fi} #{}: nearest surface point {:.3e} away at uv ({}, {}), domain u {:?} v {:?}, closed u {} v {}; loops: {}", f.id, h.dist, h.u, h.v, ns.u_range, ns.v_range, ns.closed(true, sol.tol), ns.closed(false, sol.tol), t.explain(h.u, h.v));
                            let es: Vec<String> = f.loops.iter().map(|lp| lp.iter().map(|u| format!("{}{}", u.edge, if sol.edges[u.edge].faces.iter().filter(|&&g| g == fi).count() > 1 { "s" } else { "" })).collect::<Vec<_>>().join(",")).collect();
                            println!("    loops by edge (s: seam): {:?}; senses {}", es, f.sense);
                        }
                    }
                }
                if let Ok(list) = std::env::var("ICAM_COVER") {
                    // Which of these faces claim each point of a grid on the first
                    // one's surface, by the loops and by the old test.
                    let ids: Vec<usize> = list.split(',').map(|x| x.parse().unwrap()).collect();
                    if ids.iter().any(|&i| i >= sol.faces.len()) { continue; }
                    let f0 = &sol.faces[ids[0]];
                    if let Some(a) = &f0.analytic2 {
                        let ns = &a.ns;
                        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
                        let mut shown = 0;
                        for i in 0..64 {
                            for j in 0..64 {
                                let u = ns.u_range.0 + (ns.u_range.1 - ns.u_range.0) * (i as f64 + 0.5) / 64.0;
                                let v = ns.v_range.0 + (ns.v_range.1 - ns.v_range.0) * (j as f64 + 0.5) / 64.0;
                                let x = ns.eval(u, v).p;
                                let new: Vec<bool> = ids.iter().map(|&fi| sol.classify2(fi, x) == Some(true)).collect();
                                let old: Vec<Option<bool>> = ids.iter().map(|&fi| sol.classify(fi, x)).collect();
                                *counts.entry(format!("loops {new:?} old {old:?}")).or_default() += 1;
                                if new.iter().all(|b| !b) && old.iter().any(|b| *b == Some(true)) && shown < 3 {
                                    shown += 1;
                                    println!("    claimed by no loops, by the old test: {x:?}");
                                }
                            }
                        }
                        for (k, n) in counts {
                            println!("    cover {k}: {n}");
                        }
                    }
                }
                for (fi, f) in sol.faces.iter().enumerate() {
                    if let (Surf::Nurbs(ns), Some(t)) = (&f.surf, &f.trim2) {
                        let mut c = ns.creases();
                        let (on, steps_on, done_on) = {
                            let (h, done, pops) = ns.closest_trimmed(p, f64::INFINITY, &|lo, hi| t.box_class(lo, hi), &|u, v| t.inside(u, v), &c);
                            (h.map(|h| h.dist), pops, done)
                        };
                        c.test = false;
                        let (off, steps_off, done_off) = {
                            let (h, done, pops) = ns.closest_trimmed(p, f64::INFINITY, &|lo, hi| t.box_class(lo, hi), &|u, v| t.inside(u, v), &c);
                            (h.map(|h| h.dist), pops, done)
                        };
                        if std::env::var("ICAM_PROBE").ok().and_then(|x| x.parse::<usize>().ok()) == Some(fi) {
                            probe(ns, t, p, &c);
                        }
                        println!("    B-spline face {fi} #{}: degree {}x{}, spans {}x{}, C0 lines {}, pole sides u {:?} v {:?}; search with the test {on:?} in {steps_on} steps (done {done_on}), without {off:?} in {steps_off} (done {done_off})", f.id, ns.du, ns.dv, ns.spans_u(), ns.spans_v(), ns.c0_lines().len(), c.u.len(), c.v.len());
                    }
                }
                for (fi, f) in sol.faces.iter().enumerate() {
                    let q = f.surf.foot(p);
                    let d = (p - q).len();
                    if d > 2.0 * sd.d.abs() + 1e-6 {
                        continue;
                    }
                    let (uv, ins) = match (&f.trim2, &f.analytic2) {
                        (Some(t), Some(a)) => {
                            let (u, v) = a.uv(q);
                            (Some((u, v)), Some(t.inside(u, v)))
                        }
                        _ => (None, None),
                    };
                    if let (Some(a), Some((u, v))) = (&f.analytic2, uv) {
                        let e = a.ns.eval(u, v);
                        println!("      form at uv: point off the foot by {:.2e}, (Su x Sv).normal {:.3}; {:?}; surface {:?}", (e.p - q).len(), e.ps.cross(e.pt).unit().dot(f.surf.normal(q)), a, f.surf);
                        // Round trip over the form: point -> uv -> point.
                        let mut worst = (0.0, (0.0, 0.0));
                        for i in 0..=16 {
                            for j in 0..=16 {
                                let (u2, v2) = (a.ns.u_range.0 + (a.ns.u_range.1 - a.ns.u_range.0) * i as f64 / 16.0, a.ns.v_range.0 + (a.ns.v_range.1 - a.ns.v_range.0) * j as f64 / 16.0);
                                let x = a.ns.eval(u2, v2).p;
                                let (u3, v3) = a.uv(x);
                                let y = a.ns.eval(u3, v3).p;
                                let off_surf = f.surf.dist(x);
                                if (y - x).len().max(off_surf) > worst.0 {
                                    worst = ((y - x).len().max(off_surf), (u2, v2));
                                }
                            }
                        }
                        println!("      round trip worst {:.2e} at {:?}; foot q off the surface (Surf::dist) {:.2e}", worst.0, worst.1, f.surf.dist(q));
                    }
                    let edges: Vec<(usize, f64)> = f.loops.iter().flatten().map(|u| (u.edge, sol.edges[u.edge].curve.closest(q).dist)).collect();
                    let near_e = edges.iter().fold((0, f64::INFINITY), |a, &b| if b.1 < a.1 { b } else { a });
                    println!("    face {fi} #{} {}: foot {:.9} at {q:?}, uv {uv:?}, loops say {ins:?}, old test {:?}, nearest edge {} at {:.2e}", f.id, f.surf.name(), d, sol.classify(fi, q), near_e.0, near_e.1);
                    if std::env::var("ICAM_EXPLAIN").is_ok() {
                        if let (Some(t), Some((u, v))) = (&f.trim2, uv) {
                            println!("      {}", t.explain(u, v));
                        }
                    }
                }
                continue;
            }
            if args.iter().any(|a| a == "--profile") {
                // Single thread: each edge's nearest point, each analytic face's foot
                // and loop test, and whole queries.
                let qs = &uniform[..uniform.len().min(100)];
                let t = std::time::Instant::now();
                let mut kinds: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
                for &p in qs {
                    for e in &sol.edges {
                        let t = std::time::Instant::now();
                        std::hint::black_box(e.curve.closest(p));
                        let k = kinds.entry(icam_core::solid3::edge_kind_name(&e.curve)).or_default();
                        k.0 += 1;
                        k.1 += t.elapsed().as_secs_f64();
                    }
                }
                let edges_secs = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                for &p in qs {
                    for f in &sol.faces {
                        if let (Some(tr), Some(a)) = (&f.trim2, &f.analytic2) {
                            let q = f.surf.foot(p);
                            let (u, v) = a.uv(q);
                            std::hint::black_box(tr.inside(u, v));
                        }
                    }
                }
                let faces_secs = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                for &p in qs {
                    std::hint::black_box(ex.sdf(p));
                }
                let q_secs = t.elapsed().as_secs_f64();
                let n = qs.len() as f64;
                let ks: Vec<String> = kinds.iter().map(|(k, (c, s))| format!("{k} {:.2} us ({c})", 1e6 * s / *c as f64)).collect();
                println!("  profile, one thread: every edge's nearest point {:.1} us per query ({}); every analytic face's foot and loop test {:.1} us per query; whole query {:.1} us", 1e6 * edges_secs / n, ks.join(", "), 1e6 * faces_secs / n, 1e6 * q_secs / n);
            }
            let mut examples: Vec<String> = Vec::new();
            for (label, qs) in [("uniform", &uniform), ("near faces", &near_face), ("near edges", &near_edge)] {
                // Throughput first: the queries alone, on all threads.
                let tw = std::time::Instant::now();
                let _: Vec<f64> = qs.par_iter().map(|&p| ex.sdf(p).0.d).collect();
                let rate = qs.len() as f64 / tw.elapsed().as_secs_f64().max(1e-9);
                let rows: Vec<(V3, icam_core::solid3::Sd, icam_core::solid3::ExactStats, f64, f64, Option<icam_core::solid3::Sd>)> = qs
                    .par_iter()
                    .map(|&p| {
                        let t = std::time::Instant::now();
                        let (sd, st) = ex.sdf(p);
                        let secs = t.elapsed().as_secs_f64();
                        let m = nearest(&cl, p);
                        let half = if sd.d.is_finite() && sd.d.abs() > 1e-9 { Some(ex.sdf(sd.q + (p - sd.q) * 0.5).0) } else { None };
                        (p, sd, st, secs, m, half)
                    })
                    .collect();
                let mut k = Kind { n: rows.len(), ..Default::default() };
                for (p, sd, st, secs, m, half) in &rows {
                    k.secs += secs;
                    k.pops += st.pops;
                    k.max_pops = k.max_pops.max(st.pops);
                    k.capped += st.capped as usize;
                    k.uncertified += st.uncertified as usize;
                    k.features += st.features;
                    k.by_feature[match sd.feature {
                        Feature::Face(_) => 0,
                        Feature::Edge(_) => 1,
                        Feature::Vertex(_) => 2,
                    }] += 1;
                    if st.capped && examples.len() < 12 {
                        examples.push(format!("{label} capped {p:?}: answer {:.9} ({:?}), {} steps, cloud {m:.6}", sd.d, sd.feature, st.pops));
                    }
                    let miss = sd.d.abs() - m;
                    let half_bad = half.map_or(false, |h| (h.d - 0.5 * sd.d).abs() > band + 1e-9 || ((h.d < 0.0) != (sd.d < 0.0) && h.d.abs().min(0.5 * sd.d.abs()) > band));
                    if (!(miss <= band + 1e-9) || half_bad) && !sd.unsure && !half.map_or(false, |h| h.unsure) {
                        k.unflagged += 1;
                    }
                    if !(miss <= band + 1e-9) {
                        k.missed += 1;
                        k.worst_miss = k.worst_miss.max(miss);
                        if examples.len() < 12 {
                            examples.push(format!("{label} {p:?}: answer {:.6} at {:?} ({:?}), cloud {m:.6}", sd.d, sd.q, sd.feature));
                        }
                    }
                    if let Some(h) = half {
                        let off = (h.d - 0.5 * sd.d).abs();
                        if off > 1e-9 + 1e-12 * sd.d.abs() {
                            k.half_off += 1;
                        }
                        if off > band + 1e-9 {
                            k.half_off_band += 1;
                            if examples.len() < 12 {
                                examples.push(format!("{label} half {p:?}: answer {:.9} ({:?}), halfway {:.9} ({:?})", sd.d, sd.feature, h.d, h.feature));
                            }
                        }
                        if (h.d < 0.0) != (sd.d < 0.0) {
                            k.half_sign += 1;
                            if h.d.abs().min(0.5 * sd.d.abs()) > band {
                                k.half_sign_band += 1;
                            }
                        }
                    }
                }
                println!(
                    "  {label:<10} {:>6} queries ({rate:.0} per second on all threads): missed nearer boundary beyond the band {} (worst {:.2e} mm); half-segment off {} (sign {}), beyond the band {} (sign {}); nearest face {} edge {} vertex {}; {:.1} us per query, {:.1} features, B-spline search steps mean {:.1} max {}; capped {}, uncertified faces met {}; failures beyond the band not marked unsure {}",
                    k.n,
                    k.missed,
                    k.worst_miss,
                    k.half_off,
                    k.half_sign,
                    k.half_off_band,
                    k.half_sign_band,
                    k.by_feature[0],
                    k.by_feature[1],
                    k.by_feature[2],
                    1e6 * k.secs / k.n.max(1) as f64,
                    k.features as f64 / k.n.max(1) as f64,
                    k.pops as f64 / k.n.max(1) as f64,
                    k.max_pops,
                    k.capped,
                    k.uncertified,
                    k.unflagged
                );
            }
            // No reference: Lipschitz pairs and the outside of the box.
            let pairs: Vec<(V3, V3)> = (0..per_kind)
                .map(|_| {
                    let x = V3::new(lo.x + ext.x * (1.4 * rng.next() - 0.2), lo.y + ext.y * (1.4 * rng.next() - 0.2), lo.z + ext.z * (1.4 * rng.next() - 0.2));
                    (x, x + rng.dir() * (0.02 * diag * rng.next()))
                })
                .collect();
            let excess: Vec<f64> = pairs.par_iter().map(|(x, y)| (ex.sdf(*x).0.d - ex.sdf(*y).0.d).abs() - (*x - *y).len() * (1.0 + 1e-9) - 1e-12).filter(|e| *e > 0.0).collect();
            let lip = excess.len();
            let lip_band = excess.iter().filter(|e| **e > 2.0 * band).count();
            let out_bad: Vec<(V3, icam_core::solid3::Sd)> = pairs.par_iter().filter(|(x, _)| x.x < lo.x || x.y < lo.y || x.z < lo.z || x.x > hi.x || x.y > hi.y || x.z > hi.z).map(|(x, _)| (*x, ex.sdf(*x).0)).filter(|(_, sd)| !(sd.d > 0.0)).collect();
            let outside = out_bad.len();
            let gap = |x: &V3| V3::new((lo.x - x.x).max(x.x - hi.x).max(0.0), (lo.y - x.y).max(x.y - hi.y).max(0.0), (lo.z - x.z).max(x.z - hi.z).max(0.0)).len();
            let outside_band = out_bad.iter().filter(|(x, sd)| gap(x) > band || sd.d < -band).count();
            for (x, sd) in out_bad.iter().filter(|(x, sd)| gap(x) > band || sd.d < -band).take(3) {
                examples.push(format!("outside the box by {:.3e} at {x:?}: {:.9} at {:?} ({:?})", gap(x), sd.d, sd.q, sd.feature));
            }
            // surfcheck's samples: on the surface and +-delta along the normal.
            let samples: Vec<crate::surfcheck::Sample> = (0..sol.faces.len())
                .into_par_iter()
                .flat_map_iter(|fi| {
                    let mut r = crate::surfcheck::Rng(0x9E37_79B9_7F4A_7C15 ^ ((fi as u64 + 1) * 0x2545_F491_4F6C_DD1D));
                    crate::surfcheck::face_samples(&sol, fi, 4, &mut r)
                })
                .collect();
            // A moved sample whose answer is a nearer point of another feature (a wall
            // thinner than delta, a face behind it) is not held to plus or minus
            // delta: the rule assumes the sample's own face is nearest.
            // The sampler takes a face's points by the old trim test; where the
            // certified loops say a point is not in its face (the old test's errors
            // on narrow B-spline domains and small spherical triangles, by the
            // beside-the-edge judge) the sample is disputed, not a test of the field.
            let samples: Vec<crate::surfcheck::Sample> = samples.into_iter().filter(|s| s.delta > 1e-9).collect();
            let disputed = samples.iter().filter(|s| sol.classify2(s.face, s.p) == Some(false)).count();
            let samples: Vec<crate::surfcheck::Sample> = samples.into_iter().filter(|s| sol.classify2(s.face, s.p) != Some(false)).collect();
            let bad: Vec<(f64, f64, bool)> = samples
                .par_iter()
                .map(|s| {
                    let z = ex.sdf(s.p).0.d;
                    let (a, b) = (ex.sdf(s.p + s.n * s.delta).0, ex.sdf(s.p - s.n * s.delta).0);
                    let other = |x: &icam_core::solid3::Sd| x.feature != Feature::Face(s.face) && x.d.abs() < s.delta - 1e-9;
                    let off_a = if other(&a) { 0.0 } else { (a.d - s.delta).abs() };
                    let off_b = if other(&b) { 0.0 } else { (b.d + s.delta).abs() };
                    (z.abs(), off_a.max(off_b), !((a.d > 0.0 || other(&a)) && (b.d < 0.0 || other(&b))))
                })
                .collect();
            for (smp, b) in samples.iter().zip(&bad).filter(|(_, b)| b.1 > 1e-9 || b.2).take(3) {
                let f = &sol.faces[smp.face];
                let (z, st) = ex.sdf(smp.p);
                let (a, _) = ex.sdf(smp.p + smp.n * smp.delta);
                let (c, _) = ex.sdf(smp.p - smp.n * smp.delta);
                let loops = sol.classify2(smp.face, smp.p);
                examples.push(format!("sample on face {} #{} {} at {:?} (delta {:.2e}, old test inside, loops say {loops:?}): on it {:.3e} ({:?}), out {:.6e} ({:?}), in {:.6e} ({:?}) {st:?}; off by {:.2e}", smp.face, f.id, f.surf.name(), smp.p, smp.delta, z.d, z.feature, a.d, a.feature, c.d, c.feature, b.1));
            }
            let zero = bad.iter().map(|x| x.0).fold(0.0, f64::max);
            let off = bad.iter().filter(|x| x.1 > 1e-9).count();
            let sign = bad.iter().filter(|x| x.2).count();
            println!("  Lipschitz broken at {lip} of {per_kind} pairs (by over twice the band {lip_band}); not positive outside the box at {outside} (beyond the band {outside_band}); surface samples {} ({disputed} disputed): |d| on the surface at most {zero:.1e} mm, off +-delta by over 1e-9 mm at {off}, sign wrong at {sign}", samples.len());
            if old {
                let qs: Vec<V3> = uniform.iter().chain(&near_face).chain(&near_edge).step_by(((3 * per_kind) / 200).max(1)).copied().collect();
                let t = std::time::Instant::now();
                let olds: Vec<icam_core::solid3::Sd> = qs.par_iter().map(|&p| sol.sdf(p)).collect();
                let old_secs = t.elapsed().as_secs_f64();
                let mut differ = 0;
                let mut new_nearer = 0;
                let mut old_nearer = 0;
                let mut sign = 0;
                for (p, o) in qs.iter().zip(&olds) {
                    let n = ex.sdf(*p).0;
                    if (n.d - o.d).abs() > 1e-9 {
                        differ += 1;
                        if n.d.abs() < o.d.abs() - 1e-9 {
                            new_nearer += 1;
                        } else if o.d.abs() < n.d.abs() - band - 1e-9 {
                            old_nearer += 1;
                            if examples.len() < 24 {
                                examples.push(format!("old nearer at {p:?}: new {:.9} ({:?}), old {:.9} ({:?}) unsure {}", n.d, n.feature, o.d, o.feature, o.unsure));
                            }
                        }
                        if (n.d < 0.0) != (o.d < 0.0) {
                            sign += 1;
                        }
                    }
                }
                println!("  against the old B-rep distance, {} queries: differ {differ} (new nearer {new_nearer}, old nearer beyond the band {old_nearer}, sign {sign}); old {:.0} queries/s on all threads", qs.len(), qs.len() as f64 / old_secs.max(1e-9));
            }
            for e in &examples {
                println!("    {e}");
            }
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
}
