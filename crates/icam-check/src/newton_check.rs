//! icam-check newton: how well a fixed number of Newton steps, seeded once per
//! brick, finds the nearest point on a B-spline face, against the certified search.
//!
//! A GPU kernel can run the same instructions for every point of a brick if the
//! nearest point on each candidate surface comes from a fixed number of Newton
//! steps started at the brick center's own nearest point (found once, by the full
//! search). This measures that on every B-spline face of the raw reading: bricks of
//! three sizes centered near random points of the face (inside its certified trim),
//! random points in each brick, and for each point the distance after k fixed steps
//! against the certified search's distance (`NurbsSurf::closest`, exact to 1e-9 mm).
//!
//! The Newton step is projected and straight-line: the full Hessian of the squared
//! distance where it is positive definite, else Gauss-Newton; parameters clamped to
//! the domain (wrapped on a closed direction). After k steps a point counts as
//! settled when the next step would move it less than 1e-9 mm. The number that
//! matters is the settled points whose distance is still wrong: those are errors a
//! kernel would not catch.

use std::path::Path;

use icam_core::curve3::V3;
use icam_core::solid3::Solid3;
use icam_core::step;
use icam_core::surface3::{box_dist, NurbsSurf, Patch, Surf};
use rayon::prelude::*;

use crate::surfcheck::nurbs_radius;

const SIZES: [f64; 3] = [0.5, 2.0, 8.0];
const KS: [usize; 5] = [2, 3, 4, 6, 8];
const BRICKS_PER_FACE: usize = 12;
const POINTS_PER_BRICK: usize = 32;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One projected Newton step from (u, v) toward the nearest point to x; returns the
/// new parameters and how far the surface point moved (mm).
/// The same projected step on one span's Bezier patch, in its own parameters [0, 1]^2.
fn step_patch(pa: &Patch, x: V3, s: f64, t: f64) -> (f64, f64, f64) {
    let e = pa.eval(s, t);
    let r = e.p - x;
    let g = [e.ps.dot(r), e.pt.dot(r)];
    let (a, b, c) = (e.ps.dot(e.ps), e.ps.dot(e.pt), e.pt.dot(e.pt));
    let mut h = [a + e.pss.dot(r), b + e.pst.dot(r), c + e.ptt.dot(r)];
    if !(h[0] > 0.0 && h[0] * h[2] - h[1] * h[1] > 0.0) {
        h = [a, b, c];
    }
    let det = h[0] * h[2] - h[1] * h[1];
    if !(det.abs() > 1e-300) {
        return (s, t, 0.0);
    }
    let mut ds = -(h[2] * g[0] - h[1] * g[1]) / det;
    let mut dt = -(h[0] * g[1] - h[1] * g[0]) / det;
    let out = |x: f64, d: f64| (x <= 0.0 && d < 0.0) || (x >= 1.0 && d > 0.0);
    let one_d = |gi: f64, hii: f64, plain: f64| if hii > 0.0 { -gi / hii } else if plain > 0.0 { -gi / plain } else { 0.0 };
    match (out(s, ds), out(t, dt)) {
        (true, true) => {
            ds = 0.0;
            dt = one_d(g[1], h[2], c);
            if out(t, dt) {
                dt = 0.0;
                ds = one_d(g[0], h[0], a);
                if out(s, ds) {
                    ds = 0.0;
                }
            }
        }
        (true, false) => {
            ds = 0.0;
            dt = one_d(g[1], h[2], c);
        }
        (false, true) => {
            dt = 0.0;
            ds = one_d(g[0], h[0], a);
        }
        (false, false) => {}
    }
    let (ns, nt) = ((s + ds).clamp(0.0, 1.0), (t + dt).clamp(0.0, 1.0));
    let moved = (pa.eval(ns, nt).p - e.p).len();
    (ns, nt, moved)
}

fn step_newton(ns: &NurbsSurf, closed: [bool; 2], x: V3, u: f64, v: f64) -> (f64, f64, f64) {
    let e = ns.eval(u, v);
    let r = e.p - x;
    let g = [e.ps.dot(r), e.pt.dot(r)];
    let (a, b, c) = (e.ps.dot(e.ps), e.ps.dot(e.pt), e.pt.dot(e.pt));
    let mut h = [a + e.pss.dot(r), b + e.pst.dot(r), c + e.ptt.dot(r)];
    if !(h[0] > 0.0 && h[0] * h[2] - h[1] * h[1] > 0.0) {
        h = [a, b, c];
    }
    let det = h[0] * h[2] - h[1] * h[1];
    if !(det.abs() > 1e-300) {
        return (u, v, 0.0);
    }
    let mut du = -(h[2] * g[0] - h[1] * g[1]) / det;
    let mut dv = -(h[0] * g[1] - h[1] * g[0]) / det;
    // A parameter on its bound whose step points out of the domain is held, and the
    // step is taken in the other one alone (the minimum along that side): a step
    // computed with both free and then clamped leaves a false resting point.
    let out_u = !closed[0] && ((u <= ns.u_range.0 && du < 0.0) || (u >= ns.u_range.1 && du > 0.0));
    let out_v = !closed[1] && ((v <= ns.v_range.0 && dv < 0.0) || (v >= ns.v_range.1 && dv > 0.0));
    let one_d = |gi: f64, hii: f64, plain: f64| if hii > 0.0 { -gi / hii } else if plain > 0.0 { -gi / plain } else { 0.0 };
    match (out_u, out_v) {
        (true, true) => {
            // Both held: test each alone.
            du = 0.0;
            dv = one_d(g[1], h[2], c);
            if (v <= ns.v_range.0 && dv < 0.0) || (v >= ns.v_range.1 && dv > 0.0) {
                dv = 0.0;
                du = one_d(g[0], h[0], a);
                if (u <= ns.u_range.0 && du < 0.0) || (u >= ns.u_range.1 && du > 0.0) {
                    du = 0.0;
                }
            }
        }
        (true, false) => {
            du = 0.0;
            dv = one_d(g[1], h[2], c);
        }
        (false, true) => {
            dv = 0.0;
            du = one_d(g[0], h[0], a);
        }
        (false, false) => {}
    }
    let fit = |x: f64, range: (f64, f64), closed: bool| {
        if closed {
            range.0 + (x - range.0).rem_euclid(range.1 - range.0)
        } else {
            x.clamp(range.0, range.1)
        }
    };
    let (nu, nv) = (fit(u + du, ns.u_range, closed[0]), fit(v + dv, ns.v_range, closed[1]));
    let moved = (ns.eval(nu, nv).p - e.p).len();
    (nu, nv, moved)
}

#[derive(Default, Clone, Copy)]
struct Tally {
    points: usize,
    /// Within 1e-9 mm of the certified distance.
    exact: usize,
    /// Within 1e-6 mm.
    near: usize,
    /// Not settled after k steps (the kernel would flag them).
    flagged: usize,
    /// Settled but more than 1e-9 mm off: errors a kernel would not catch.
    silent: usize,
    /// The worst silent error (mm).
    silent_worst: f64,
    /// Silent errors where Newton's point is nearer than the search's: the search,
    /// not Newton, missed the nearest point.
    search_missed: usize,
    search_missed_worst: f64,
    /// Silent errors where Newton ended on the domain's border.
    on_border: usize,
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.points += o.points;
        self.exact += o.exact;
        self.near += o.near;
        self.flagged += o.flagged;
        self.silent += o.silent;
        self.silent_worst = self.silent_worst.max(o.silent_worst);
        self.search_missed += o.search_missed;
        self.search_missed_worst = self.search_missed_worst.max(o.search_missed_worst);
        self.on_border += o.on_border;
    }
}

/// Per brick size: per k, and per (brick size / radius of curvature) band at the
/// largest k; plus the time of the certified search and of the fixed steps.
#[derive(Default, Clone)]
struct Out {
    /// Seeded once per span that can hold a brick point's nearest point, Newton in
    /// each span's own patch, the least distance kept.
    spans_k: [Tally; KS.len()],
    /// Candidate spans per brick, summed, and the most in one brick.
    cand_sum: usize,
    cand_max: usize,
    span_secs: f64,
    by_k: [Tally; KS.len()],
    by_ratio: [Tally; 4],
    search_secs: f64,
    newton_secs: f64,
    bricks: usize,
}

fn ratio_band(r: f64) -> usize {
    match r {
        x if x < 0.1 => 0,
        x if x < 0.5 => 1,
        x if x < 1.0 => 2,
        _ => 3,
    }
}

/// `icam-check newton <file.step | dir>`.
pub fn run(path: &Path) {
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
    println!("fixed Newton steps from a per-brick seed against the certified nearest-point search; {BRICKS_PER_FACE} bricks per face and size, {POINTS_PER_BRICK} points per brick");
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let Ok(brep) = step::read_raw(f) else {
            println!("{name}: read failed");
            continue;
        };
        let tol = brep.tolerance_mm.max(1e-6);
        let mut outs: Vec<Out> = vec![Out::default(); SIZES.len()];
        let t0 = std::time::Instant::now();
        for s in &brep.solids {
            let Ok((sol, _)) = Solid3::from_step(s, tol) else { continue };
            let faces: Vec<usize> = (0..sol.faces.len()).filter(|&fi| matches!(sol.faces[fi].surf, Surf::Nurbs(_))).collect();
            let per: Vec<Vec<Out>> = faces
                .par_iter()
                .map(|&fi| {
                    let Surf::Nurbs(ns) = &sol.faces[fi].surf else { unreachable!() };
                    let closed = [ns.closed(true, tol), ns.closed(false, tol)];
                    let (nsu, nsv) = ns.spans_uv();
                    let patches: Vec<(Patch, (V3, V3))> = (0..nsu).flat_map(|a| (0..nsv).map(move |b| (a, b))).map(|(a, b)| {
                        let pa = ns.patch(a, b);
                        let bx = pa.bbox();
                        (pa, bx)
                    }).collect();
                    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((fi as u64 + 1) * 0x2545_F491_4F6C_DD1D));
                    let mut res = vec![Out::default(); SIZES.len()];
                    for (si, &h) in SIZES.iter().enumerate() {
                        let mut made = 0;
                        let mut tries = 0;
                        while made < BRICKS_PER_FACE && tries < 20 * BRICKS_PER_FACE {
                            tries += 1;
                            let (u, v) = (ns.u_range.0 + (ns.u_range.1 - ns.u_range.0) * rng.next(), ns.v_range.0 + (ns.v_range.1 - ns.v_range.0) * rng.next());
                            if sol.classify_nurbs2(fi, u, v) == Some(false) {
                                continue;
                            }
                            let e = ns.eval(u, v);
                            let nn = e.ps.cross(e.pt);
                            if !(nn.len() > 0.0) {
                                continue;
                            }
                            let rad = nurbs_radius(&e);
                            let c = e.p + nn.unit() * (h * (2.0 * rng.next() - 1.0));
                            let ts = std::time::Instant::now();
                            let Some(seed) = ns.closest(c, f64::INFINITY) else { continue };
                            res[si].search_secs += ts.elapsed().as_secs_f64();
                            made += 1;
                            res[si].bricks += 1;
                            // Spans that can hold the nearest point of a point in the brick:
                            // within the center's distance plus the brick's diagonal.
                            let reach = seed.dist + h * 3f64.sqrt();
                            let cands: Vec<(&Patch, f64, f64)> = patches
                                .iter()
                                .filter(|(_, bx)| box_dist(c, bx) <= reach)
                                .map(|(pa, _)| {
                                    let (mut s0, mut t0) = (0.5, 0.5);
                                    for _ in 0..30 {
                                        let (a, b, _) = step_patch(pa, c, s0, t0);
                                        s0 = a;
                                        t0 = b;
                                    }
                                    (pa, s0, t0)
                                })
                                .collect();
                            res[si].cand_sum += cands.len();
                            res[si].cand_max = res[si].cand_max.max(cands.len());
                            for _ in 0..POINTS_PER_BRICK {
                                let x = c + V3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5) * h;
                                let ts = std::time::Instant::now();
                                let Some(truth) = ns.closest(x, f64::INFINITY) else { continue };
                                res[si].search_secs += ts.elapsed().as_secs_f64();
                                // Per span: k fixed steps in its patch from its seed.
                                let tsp = std::time::Instant::now();
                                for (ki, &k) in KS.iter().enumerate() {
                                    let mut best = f64::INFINITY;
                                    let mut all_settled = true;
                                    for &(pa, s0, t0) in &cands {
                                        let (mut s1, mut t1) = (s0, t0);
                                        for _ in 0..k {
                                            let (a, b, _) = step_patch(pa, x, s1, t1);
                                            s1 = a;
                                            t1 = b;
                                        }
                                        let p1 = pa.eval(s1, t1).p;
                                        let d = (p1 - x).len();
                                        let (_, _, moved) = step_patch(pa, x, s1, t1);
                                        if moved > 1e-9 {
                                            all_settled = false;
                                        }
                                        best = best.min(d);
                                    }
                                    let err = (best - truth.dist).abs();
                                    let t = &mut res[si].spans_k[ki];
                                    t.points += 1;
                                    if err <= 1e-9 {
                                        t.exact += 1;
                                    }
                                    if err <= 1e-6 {
                                        t.near += 1;
                                    }
                                    if !all_settled {
                                        t.flagged += 1;
                                    } else if err > 1e-9 {
                                        t.silent += 1;
                                        t.silent_worst = t.silent_worst.max(err);
                                        if best < truth.dist {
                                            t.search_missed += 1;
                                            t.search_missed_worst = t.search_missed_worst.max(err);
                                        }
                                    }
                                }
                                res[si].span_secs += tsp.elapsed().as_secs_f64();
                                let tn = std::time::Instant::now();
                                let (mut uu, mut vv) = (seed.u, seed.v);
                                let mut done = 0;
                                let mut after: Vec<(f64, f64, bool)> = Vec::new();
                                for &k in &KS {
                                    while done < k {
                                        let (a, b, _) = step_newton(ns, closed, x, uu, vv);
                                        uu = a;
                                        vv = b;
                                        done += 1;
                                    }
                                    let d = (ns.eval(uu, vv).p - x).len();
                                    // How far the next step would move: the settle check.
                                    let (_, _, moved) = step_newton(ns, closed, x, uu, vv);
                                    let border = (!closed[0] && (uu <= ns.u_range.0 || uu >= ns.u_range.1)) || (!closed[1] && (vv <= ns.v_range.0 || vv >= ns.v_range.1));
                                    after.push((d, moved, border));
                                }
                                res[si].newton_secs += tn.elapsed().as_secs_f64();
                                for (ki, &(d, moved, border)) in after.iter().enumerate() {
                                    let err = (d - truth.dist).abs();
                                    let t = &mut res[si].by_k[ki];
                                    t.points += 1;
                                    if err <= 1e-9 {
                                        t.exact += 1;
                                    }
                                    if err <= 1e-6 {
                                        t.near += 1;
                                    }
                                    let settled = moved <= 1e-9;
                                    if !settled {
                                        t.flagged += 1;
                                    } else if err > 1e-9 {
                                        t.silent += 1;
                                        t.silent_worst = t.silent_worst.max(err);
                                        if d < truth.dist {
                                            t.search_missed += 1;
                                            t.search_missed_worst = t.search_missed_worst.max(err);
                                        }
                                        if border {
                                            t.on_border += 1;
                                        }
                                    }
                                    if ki + 1 == KS.len() {
                                        let band = ratio_band(if rad > 0.0 { h / rad } else { f64::INFINITY });
                                        let tb = &mut res[si].by_ratio[band];
                                        tb.points += 1;
                                        if err <= 1e-9 {
                                            tb.exact += 1;
                                        }
                                        if err <= 1e-6 {
                                            tb.near += 1;
                                        }
                                        if !settled {
                                            tb.flagged += 1;
                                        } else if err > 1e-9 {
                                            tb.silent += 1;
                                            tb.silent_worst = tb.silent_worst.max(err);
                                            if d < truth.dist {
                                                tb.search_missed += 1;
                                                tb.search_missed_worst = tb.search_missed_worst.max(err);
                                            }
                                            if border {
                                                tb.on_border += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    res
                })
                .collect();
            for r in per {
                for (o, x) in outs.iter_mut().zip(r) {
                    for k in 0..KS.len() {
                        o.by_k[k].add(&x.by_k[k]);
                        o.spans_k[k].add(&x.spans_k[k]);
                    }
                    o.cand_sum += x.cand_sum;
                    o.cand_max = o.cand_max.max(x.cand_max);
                    o.span_secs += x.span_secs;
                    for b in 0..4 {
                        o.by_ratio[b].add(&x.by_ratio[b]);
                    }
                    o.search_secs += x.search_secs;
                    o.newton_secs += x.newton_secs;
                    o.bricks += x.bricks;
                }
            }
        }
        let pct = |a: usize, b: usize| if b == 0 { "-".to_string() } else { format!("{:.3}%", 100.0 * a as f64 / b as f64) };
        println!("{name}: {:.1} s", t0.elapsed().as_secs_f64());
        for (si, o) in outs.iter().enumerate() {
            let n = o.by_k[0].points.max(1);
            println!(
                "  brick {:.1} mm: {} bricks, {} points; certified search {:.1} us a point, {} fixed steps {:.1} us a point (CPU, one thread, NurbsSurf::eval per step)",
                SIZES[si],
                o.bricks,
                o.by_k[0].points,
                1e6 * o.search_secs / (n + o.bricks) as f64,
                KS[KS.len() - 1],
                1e6 * o.newton_secs / n as f64
            );
            for (ki, t) in o.by_k.iter().enumerate() {
                println!(
                    "    k = {}: exact {}, within 1e-6 mm {}, not settled (flagged) {}, settled but differs {} (worst {:.2e} mm); of those, Newton nearer than the search {} (worst {:.2e}), Newton on the domain border {}",
                    KS[ki],
                    pct(t.exact, t.points),
                    pct(t.near, t.points),
                    pct(t.flagged, t.points),
                    pct(t.silent, t.points),
                    t.silent_worst,
                    t.search_missed,
                    t.search_missed_worst,
                    t.on_border
                );
            }
            println!(
                "    seeded per span: {:.1} candidate spans a brick on average, {} at most; all k in sequence {:.1} us a point",
                o.cand_sum as f64 / o.bricks.max(1) as f64,
                o.cand_max,
                1e6 * o.span_secs / n as f64
            );
            for (ki, t) in o.spans_k.iter().enumerate() {
                println!(
                    "      k = {}: exact {}, within 1e-6 mm {}, some span not settled (flagged) {}, settled but differs {} (worst {:.2e} mm); nearer than the search {}",
                    KS[ki],
                    pct(t.exact, t.points),
                    pct(t.near, t.points),
                    pct(t.flagged, t.points),
                    pct(t.silent, t.points),
                    t.silent_worst,
                    t.search_missed
                );
            }
            let names = ["brick under 0.1 R", "0.1 to 0.5 R", "0.5 to 1 R", "over 1 R"];
            for (b, t) in o.by_ratio.iter().enumerate() {
                if t.points > 0 {
                    println!(
                        "    at k = {}, {} ({} points): exact {}, flagged {}, settled but differs {} (worst {:.2e} mm); Newton nearer {}, on border {}",
                        KS[KS.len() - 1],
                        names[b],
                        t.points,
                        pct(t.exact, t.points),
                        pct(t.flagged, t.points),
                        pct(t.silent, t.points),
                        t.silent_worst,
                        t.search_missed,
                        t.on_border
                    );
                }
            }
        }
    }
}
