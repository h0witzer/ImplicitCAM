//! icam-check spans: the implicit matrix representation (icam_core::mrep) of
//! every Bezier span of every B-spline face in a STEP file, measured per span:
//! the build (size, base points, residual, time), the evaluation rate, whether
//! the extended algebraic surface passes through the span itself a second time
//! (a corank-2 point on the patch: no fence around the span can exclude that
//! sheet), and whether it has zeros inside the span's own box away from the
//! patch (a foreign sheet a box fence would not exclude, which subdivision
//! shrinks away from). The reference distances here are a sampled Newton on the
//! patch, which is allowed in this checker and nowhere else.

use std::path::Path;

use icam_core::curve3::V3;
use icam_core::mrep::MRep;
use icam_core::solid3::Solid3;
use icam_core::step;
use icam_core::surface3::{Patch, Surf};
use rayon::prelude::*;

struct SpanStat {
    face: u32,
    span: (usize, usize),
    deg: (usize, usize),
    m: usize,
    r: usize,
    build_s: f64,
    /// System singular values (relative) just inside and just past the r taken.
    s_last: f64,
    s_next: f64,
    residual: f64,
    /// Largest smallest-to-largest singular value ratio of M(p) over the patch
    /// samples: how exactly the matrix drops rank on the patch.
    on_ratio: f64,
    /// Smallest second-smallest-to-largest singular value ratio over the patch
    /// samples: near zero means a second sheet passes through the patch there.
    cross_ratio: f64,
    cross_hits: usize,
    cross_confirmed: usize,
    /// Box samples whose first-order algebraic distance is far below their
    /// distance to the patch, and those that then descend to a zero of the
    /// M-rep still far from the patch.
    box_pts: usize,
    foreign_flagged: usize,
    /// Zeros reached that are the span's own sheet continuing past its edges
    /// (a real preimage within half a span of the unit square), and zeros that
    /// are not (another real sheet, or one only complex parameters reach).
    continuation: usize,
    foreign_confirmed: usize,
    foreign_e: f64,
    /// Sign changes of the square M-rep's determinant along lines through the
    /// padded box: real sheets of the patch's own algebraic surface, sorted into
    /// the patch, its continuation past its edges, and other sheets.
    lines: usize,
    sheet_patch: usize,
    sheet_cont: usize,
    /// The span's own sheet reached only from parameters 0.5 to 3 spans out.
    sheet_far: usize,
    sheet_other: usize,
    sheet_e: f64,
    /// Nearest other sheet to the patch, in span sizes, and where it was found.
    sheet_near: f64,
    sheet_near_pt: Option<V3>,
    /// At the other-sheet points, the floor-based M-rep's smallest-to-largest
    /// singular value ratio (median over the span): tiny means the idealized
    /// surface passes there too, large means only the exact degree-2mn one does.
    wide_at_other: f64,
    err: Option<String>,
}

/// Whether x is on the span's own sheet continued past its edges: a real
/// preimage within `reach` spans of the unit square.
pub(crate) fn own_sheet(patch: &Patch, x: V3, scale: f64, reach: f64) -> bool {
    let n = if reach > 1.0 { 9 } else { 5 };
    let (lo, hi) = (-reach, 1.0 + reach);
    for a in 0..n {
        for b in 0..n {
            let (s1, t1) = patch.newton_in(x, lo + (hi - lo) * (a as f64 + 0.5) / n as f64, lo + (hi - lo) * (b as f64 + 0.5) / n as f64, (lo, hi), (lo, hi));
            if (patch.eval(s1, t1).p - x).len() < 1e-7 * scale {
                return true;
            }
        }
    }
    false
}

/// Distance from q to the patch: the least over a 24 x 24 sample of the patch
/// and Newton from a 4 x 4 grid of starts (an upper bound either way, so a point
/// on the patch is never reported far from it).
pub(crate) fn patch_dist(p: &Patch, q: V3) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..=24 {
        for j in 0..=24 {
            best = best.min((p.eval(i as f64 / 24.0, j as f64 / 24.0).p - q).len());
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            let (s, t) = p.newton_capped(q, (i as f64 + 0.5) / 4.0, (j as f64 + 0.5) / 4.0, 25);
            best = best.min((p.eval(s, t).p - q).len());
        }
    }
    best
}

/// sigma_min of the M-rep at q and its first-order distance sigma_min / |grad sigma_min|.
fn alg_dist(rep: &MRep, q: V3) -> (f64, f64, V3) {
    let h = 1e-4 * rep.scale;
    let f = |p: V3| rep.sigma_min(p);
    let s0 = f(q);
    let g = V3::new(
        (f(q + V3::new(h, 0.0, 0.0)) - f(q - V3::new(h, 0.0, 0.0))) / (2.0 * h),
        (f(q + V3::new(0.0, h, 0.0)) - f(q - V3::new(0.0, h, 0.0))) / (2.0 * h),
        (f(q + V3::new(0.0, 0.0, h)) - f(q - V3::new(0.0, 0.0, h))) / (2.0 * h),
    );
    let gl = g.len();
    (s0, if gl > 0.0 { s0 / gl } else { f64::INFINITY }, if gl > 0.0 { g * (1.0 / gl) } else { V3::new(0.0, 0.0, 0.0) })
}

fn span_stat(face: u32, span: (usize, usize), patch: &Patch) -> SpanStat {
    let deg = (patch.pw.len() - 1, patch.pw[0].len() - 1);
    let t0 = std::time::Instant::now();
    let tol: f64 = std::env::var("ICAM_MREP_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(MRep::MEASURE_FLOOR);
    let rep = match MRep::with_tol(patch, tol) {
        Ok(r) => r,
        Err(e) => {
            return SpanStat { face, span, deg, m: 0, r: 0, build_s: t0.elapsed().as_secs_f64(), s_last: 0.0, s_next: 0.0, residual: 0.0, on_ratio: 0.0, cross_ratio: 0.0, cross_hits: 0, cross_confirmed: 0, box_pts: 0, foreign_flagged: 0, continuation: 0, foreign_confirmed: 0, foreign_e: 0.0, lines: 0, sheet_patch: 0, sheet_cont: 0, sheet_far: 0, sheet_other: 0, sheet_e: 0.0, sheet_near: f64::INFINITY, sheet_near_pt: None, wide_at_other: f64::NAN, err: Some(e) };
        }
    };
    let build_s = t0.elapsed().as_secs_f64();
    let scale = rep.scale;
    // A second sheet through the patch: corank 2 at a patch point.
    let n = 15;
    let (mut cross_ratio, mut cross_hits, mut cross_confirmed) = (f64::INFINITY, 0, 0);
    let mut on_ratio = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let (s, t) = ((i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64);
            let p = patch.eval(s, t).p;
            let sig = rep.svd(p).0;
            on_ratio = on_ratio.max(sig[sig.len() - 1] / sig[0]);
            if sig.len() < 2 {
                continue;
            }
            let ratio = sig[sig.len() - 2] / sig[0];
            cross_ratio = cross_ratio.min(ratio);
            if ratio < 1e-7 {
                cross_hits += 1;
                // Another real preimage in the polynomial's extended domain confirms it.
                let mut found = false;
                for a in 0..10 {
                    for b in 0..10 {
                        let (s1, t1) = patch.newton_in(p, -3.0 + 7.0 * (a as f64 + 0.5) / 10.0, -3.0 + 7.0 * (b as f64 + 0.5) / 10.0, (-3.0, 4.0), (-3.0, 4.0));
                        if (patch.eval(s1, t1).p - p).len() < 1e-9 * scale && (s1 - s).abs() + (t1 - t).abs() > 1e-3 {
                            found = true;
                        }
                    }
                }
                if found {
                    cross_confirmed += 1;
                }
            }
        }
    }
    // Zeros of the M-rep inside the span's box away from the patch.
    let (lo, hi) = patch.bbox();
    let pad = (hi - lo) * 0.1;
    let (lo, hi) = (lo - pad, hi + pad);
    let k = 7;
    let (mut box_pts, mut foreign_flagged, mut continuation, mut foreign_confirmed, mut foreign_e) = (0, 0, 0, 0, 0.0f64);
    for i in 0..k {
        for j in 0..k {
            for l in 0..k {
                let q = V3::new(lo.x + (hi.x - lo.x) * (i as f64 + 0.5) / k as f64, lo.y + (hi.y - lo.y) * (j as f64 + 0.5) / k as f64, lo.z + (hi.z - lo.z) * (l as f64 + 0.5) / k as f64);
                box_pts += 1;
                let (_, a, _) = alg_dist(&rep, q);
                let e = patch_dist(patch, q);
                if !(a < 0.25 * e && e > 0.02 * scale) {
                    continue;
                }
                foreign_flagged += 1;
                // Descend to the zero the first-order distance points at.
                let mut x = q;
                let mut hit = false;
                for _ in 0..12 {
                    let (s0, a, g) = alg_dist(&rep, x);
                    let sig = rep.svd(x).0;
                    if sig[sig.len() - 1] / sig[0] < 1e-8 {
                        hit = true;
                        break;
                    }
                    if !(a.is_finite() && s0 > 0.0) {
                        break;
                    }
                    x = x - g * a;
                }
                if hit {
                    let e2 = patch_dist(patch, x);
                    if e2 > 0.02 * scale {
                        if own_sheet(patch, x, scale, 0.5) {
                            continuation += 1;
                        } else {
                            foreign_confirmed += 1;
                            foreign_e = foreign_e.max(e2 / scale);
                        }
                    }
                }
            }
        }
    }
    // Real sheets of the exact algebraic surface: sign changes of the square
    // M-rep's determinant along lines through the padded box.
    let (mut lines, mut sheet_patch, mut sheet_cont, mut sheet_far, mut sheet_other, mut sheet_e, mut sheet_near) = (0, 0, 0, 0, 0, 0.0f64, f64::INFINITY);
    let mut sheet_near_pt = None;
    let mut wide_ratios: Vec<f64> = Vec::new();
    if let Ok(sq) = MRep::square(patch) {
        let g = 5;
        let n = 160;
        for axis in 0..3 {
            for i in 0..g {
                for j in 0..g {
                    lines += 1;
                    let (fi, fj) = ((i as f64 + 0.5) / g as f64, (j as f64 + 0.5) / g as f64);
                    let at = |u: f64| -> V3 {
                        match axis {
                            0 => V3::new(lo.x + (hi.x - lo.x) * u, lo.y + (hi.y - lo.y) * fi, lo.z + (hi.z - lo.z) * fj),
                            1 => V3::new(lo.x + (hi.x - lo.x) * fi, lo.y + (hi.y - lo.y) * u, lo.z + (hi.z - lo.z) * fj),
                            _ => V3::new(lo.x + (hi.x - lo.x) * fi, lo.y + (hi.y - lo.y) * fj, lo.z + (hi.z - lo.z) * u),
                        }
                    };
                    let mut prev = sq.det(at(0.0)).unwrap();
                    for k in 1..=n {
                        let u = k as f64 / n as f64;
                        let cur = sq.det(at(u)).unwrap();
                        if prev != 0.0 && cur != 0.0 && (prev < 0.0) != (cur < 0.0) {
                            // Narrow the change to a point, then sort it.
                            let (mut a, mut b, mut fa) = ((k - 1) as f64 / n as f64, u, prev);
                            for _ in 0..30 {
                                let c = 0.5 * (a + b);
                                let fc = sq.det(at(c)).unwrap();
                                if fc == 0.0 {
                                    a = c;
                                    b = c;
                                    break;
                                }
                                if (fc < 0.0) == (fa < 0.0) {
                                    a = c;
                                    fa = fc;
                                } else {
                                    b = c;
                                }
                            }
                            let x = at(0.5 * (a + b));
                            let e = patch_dist(patch, x);
                            if e < 0.02 * scale {
                                sheet_patch += 1;
                            } else if own_sheet(patch, x, scale, 0.5) {
                                sheet_cont += 1;
                            } else if own_sheet(patch, x, scale, 3.0) {
                                sheet_far += 1;
                            } else {
                                sheet_other += 1;
                                sheet_e = sheet_e.max(e / scale);
                                let sg = rep.svd(x).0;
                                wide_ratios.push(sg[sg.len() - 1] / sg[0]);
                                if e / scale < sheet_near {
                                    sheet_near = e / scale;
                                    sheet_near_pt = Some(x);
                                }
                            }
                        }
                        prev = cur;
                    }
                }
            }
        }
    }
    SpanStat { face, span, deg, m: rep.m, r: rep.r, build_s, s_last: rep.spectrum[rep.r - 1], s_next: rep.spectrum[rep.r], residual: rep.residual, on_ratio, cross_ratio, cross_hits, cross_confirmed, box_pts, foreign_flagged, continuation, foreign_confirmed, foreign_e, lines, sheet_patch, sheet_cont, sheet_far, sheet_other, sheet_e, sheet_near, sheet_near_pt, wide_at_other: pct(&mut wide_ratios, 0.5), err: None }
}

pub(crate) fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

pub fn spans(path: &Path, max_spans: usize) {
    let brep = step::read(path).unwrap();
    let tol = brep.tolerance_mm.max(1e-6);
    for s in &brep.solids {
        let (sol, _) = match Solid3::from_step(s, tol) {
            Ok(x) => x,
            Err(e) => {
                println!("solid #{}: {e}", s.id);
                continue;
            }
        };
        let mut jobs: Vec<(u32, (usize, usize), Patch)> = Vec::new();
        let mut nfaces = 0;
        for f in &sol.faces {
            let Surf::Nurbs(ns) = &f.surf else { continue };
            nfaces += 1;
            let (nu, nv) = ns.spans_uv();
            for a in 0..nu {
                for b in 0..nv {
                    jobs.push((f.id, (a, b), ns.patch(a, b)));
                }
            }
        }
        let total = jobs.len();
        if let Ok(k) = std::env::var("ICAM_SPAN_DUMP") {
            // The system spectrum of the first k spans of each degree, to see where the null space ends.
            let k: usize = k.parse().unwrap_or(3);
            let mut seen: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
            for (f, sp, p) in &jobs {
                let deg = (p.pw.len() - 1, p.pw[0].len() - 1);
                let c = seen.entry(deg).or_insert(0);
                if *c >= k {
                    continue;
                }
                *c += 1;
                let Ok(rep) = MRep::with_tol(p, 1.0) else { continue };
                let wmin = p.pw.iter().flatten().map(|h| h[3]).fold(f64::INFINITY, f64::min);
                let wmax = p.pw.iter().flatten().map(|h| h[3]).fold(0.0, f64::max);
                let top: Vec<String> = rep.spectrum.iter().take(rep.m + 14).map(|x| format!("{x:.0e}")).collect();
                println!("  face #{f} span ({},{}) degree ({},{}) m {} weights {wmin:.3}..{wmax:.3}: spectrum {}", sp.0, sp.1, deg.0, deg.1, rep.m, top.join(" "));
            }
        }
        if max_spans > 0 && jobs.len() > max_spans {
            // An even subsample keeps every face represented.
            let step = jobs.len() as f64 / max_spans as f64;
            jobs = (0..max_spans).map(|i| jobs[(i as f64 * step) as usize].clone()).collect();
        }
        println!("solid #{} {}: {} B-spline faces, {} spans ({} measured)", s.id, s.name, nfaces, total, jobs.len());
        let t0 = std::time::Instant::now();
        let stats: Vec<SpanStat> = jobs.par_iter().map(|(f, sp, p)| span_stat(*f, *sp, p)).collect();
        let wall = t0.elapsed().as_secs_f64();
        // Evaluation rate on one thread: SVD of M(p) at random points in the first spans' boxes.
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut rate_by_m: std::collections::BTreeMap<usize, (usize, f64)> = Default::default();
        for (_, _, p) in jobs.iter().take(40) {
            let Ok(rep) = MRep::with_tol(p, MRep::MEASURE_FLOOR) else { continue };
            let (lo, hi) = p.bbox();
            let pts: Vec<V3> = (0..500).map(|_| V3::new(lo.x + (hi.x - lo.x) * next(), lo.y + (hi.y - lo.y) * next(), lo.z + (hi.z - lo.z) * next())).collect();
            let t = std::time::Instant::now();
            let mut acc = 0.0;
            for q in &pts {
                acc += rep.sigma_min(*q);
            }
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(acc);
            let e = rate_by_m.entry(rep.m).or_insert((0, 0.0));
            e.0 += pts.len();
            e.1 += dt;
        }
        // Per degree: spans, with a null space of exactly m, with more, with no
        // clear gap at the floor (the next singular value within 100x of the last
        // taken), refused.
        let mut by_deg: std::collections::BTreeMap<(usize, usize), (usize, usize, usize, usize, usize)> = Default::default();
        let (mut build, mut errs) = (0.0, 0);
        let (mut cross_spans, mut cross_conf_spans, mut foreign_spans, mut foreign_conf_spans) = (0, 0, 0, 0);
        let (mut box_pts, mut flagged, mut cont, mut confirmed) = (0, 0, 0, 0);
        let (mut lines, mut sh_patch, mut sh_cont, mut sh_far, mut sh_other, mut sh_spans) = (0, 0, 0, 0, 0, 0);
        let mut nears = Vec::new();
        let mut wides = Vec::new();
        let mut ratios = Vec::new();
        let mut resid = Vec::new();
        let mut lasts = Vec::new();
        let mut nexts = Vec::new();
        let mut ons = Vec::new();
        let mut rs: std::collections::BTreeMap<(usize, usize, usize), usize> = Default::default();
        for st in &stats {
            let e = by_deg.entry(st.deg).or_insert((0, 0, 0, 0, 0));
            e.0 += 1;
            build += st.build_s;
            if st.err.is_some() {
                errs += 1;
                e.4 += 1;
                continue;
            }
            if st.r == st.m {
                e.1 += 1;
            } else {
                e.2 += 1;
            }
            if st.s_next < 100.0 * st.s_last.max(1e-300) {
                e.3 += 1;
            }
            *rs.entry((st.deg.0, st.deg.1, st.r)).or_insert(0) += 1;
            ratios.push(st.cross_ratio);
            resid.push(st.residual);
            lasts.push(st.s_last);
            nexts.push(st.s_next);
            ons.push(st.on_ratio);
            cont += st.continuation;
            lines += st.lines;
            sh_patch += st.sheet_patch;
            sh_cont += st.sheet_cont;
            sh_far += st.sheet_far;
            sh_other += st.sheet_other;
            sh_spans += (st.sheet_other > 0) as usize;
            if st.sheet_other > 0 {
                nears.push(st.sheet_near);
                wides.push(st.wide_at_other);
            }
            cross_spans += (st.cross_hits > 0) as usize;
            cross_conf_spans += (st.cross_confirmed > 0) as usize;
            foreign_spans += (st.foreign_flagged > 0) as usize;
            foreign_conf_spans += (st.foreign_confirmed > 0) as usize;
            box_pts += st.box_pts;
            flagged += st.foreign_flagged;
            confirmed += st.foreign_confirmed;
        }
        println!("  measured in {wall:.1} s on {} threads; M-rep builds {:.1} ms each on average; {errs} refused", rayon::current_num_threads(), 1e3 * build / stats.len().max(1) as f64);
        println!("  {:<10} {:>6} {:>8} {:>8} {:>8} {:>8}   (null space at the floor: exactly m, more than m, no clear gap, refused)", "degree", "spans", "exact m", "more", "no gap", "refused");
        for (d, (n, sq, wide, unclear, bad)) in &by_deg {
            println!("  ({:>2},{:>2})    {:>6} {:>8} {:>8} {:>8} {:>8}", d.0, d.1, n, sq, wide, unclear, bad);
        }
        let widths: Vec<String> = rs.iter().map(|((a, b, r), n)| format!("({a},{b}) r={r}: {n}")).collect();
        println!("  columns taken: {}", widths.join(", "));
        for (m, (n, dt)) in &rate_by_m {
            println!("  sigma_min rate, {m} rows: {:.0} points/s on one thread", *n as f64 / dt);
        }
        println!("  moving-plane residual p50 {:.1e} p99 {:.1e} max {:.1e}", pct(&mut resid, 0.5), pct(&mut resid, 0.99), pct(&mut resid, 1.0));
        println!("  system singular values (relative): last taken p50 {:.1e} p99 {:.1e} max {:.1e}; first left out p1 {:.1e} p10 {:.1e} p50 {:.1e}", pct(&mut lasts, 0.5), pct(&mut lasts, 0.99), pct(&mut lasts, 1.0), pct(&mut nexts, 0.01), pct(&mut nexts, 0.1), pct(&mut nexts, 0.5));
        println!("  rank drop on the patch (smallest/largest singular value of M(p), worst sample per span): p50 {:.1e} p99 {:.1e} max {:.1e}", pct(&mut ons, 0.5), pct(&mut ons, 0.99), pct(&mut ons, 1.0));
        println!("  second sheet through the patch (second singular value ratio over 225 patch samples): min {:.1e} p1 {:.1e} p10 {:.1e} p50 {:.1e}", pct(&mut ratios, 0.0), pct(&mut ratios, 0.01), pct(&mut ratios, 0.1), pct(&mut ratios, 0.5));
        println!("    spans with a ratio under 1e-7: {cross_spans}; with a second real preimage confirmed by Newton: {cross_conf_spans}");
        println!("  zeros in the span box away from the patch (7^3 samples, box padded 10%): {flagged} of {box_pts} samples flagged; descended to a zero over 2% of the span size from the patch: {cont} the span's own sheet past its edges, {confirmed} foreign");
        println!("    spans with a flagged sample: {foreign_spans}; with a confirmed foreign zero: {foreign_conf_spans}");
        println!("  real sheets of the exact degree-2mn surface (sign changes of the square M-rep's determinant along {lines} lines of 160 steps through the padded boxes): {sh_patch} through the patch, {sh_cont} its continuation within half a span past its edges, {sh_far} its own sheet from 0.5 to 3 spans out, {sh_other} other sheets; spans with another sheet in their box: {sh_spans}");
        println!("    nearest other sheet to the patch, in span sizes, over those spans: min {:.3} p10 {:.3} p50 {:.3} p90 {:.3}", pct(&mut nears, 0.0), pct(&mut nears, 0.1), pct(&mut nears, 0.5), pct(&mut nears, 0.9));
        println!("    floor-based M-rep rank-drop ratio at the other-sheet points (median per span): p10 {:.1e} p50 {:.1e} p90 {:.1e}", pct(&mut wides, 0.1), pct(&mut wides, 0.5), pct(&mut wides, 0.9));
        let mut worst: Vec<&SpanStat> = stats.iter().filter(|s| s.err.is_none()).collect();
        worst.sort_by(|a, b| (b.sheet_other, b.foreign_confirmed, b.cross_confirmed).cmp(&(a.sheet_other, a.foreign_confirmed, a.cross_confirmed)));
        println!("  worst spans: face  span  degree  m r  spectrum last/next  cross ratio  cross hits/confirmed  flagged/continuation/foreign (farthest)  sheets patch/cont/far/other  other sheets nearest..farthest (span sizes)");
        for s in worst.iter().take(12) {
            println!("    #{:<6} ({:>2},{:>2}) ({},{})  {:>2} {:>2}  {:.0e}/{:.0e}  {:.1e}  {:>3}/{:<3}  {:>3}/{:>3}/{:<3} ({:.2})  {:>3}/{:>3}/{:>3}/{:<3}  {:.3}..{:.2}", s.face, s.span.0, s.span.1, s.deg.0, s.deg.1, s.m, s.r, s.s_last, s.s_next, s.cross_ratio, s.cross_hits, s.cross_confirmed, s.foreign_flagged, s.continuation, s.foreign_confirmed, s.foreign_e, s.sheet_patch, s.sheet_cont, s.sheet_far, s.sheet_other, s.sheet_near, s.sheet_e);
        }
        for s in stats.iter().filter(|s| s.err.is_some()).take(8) {
            println!("    refused #{} ({},{}) degree ({},{}): {}", s.face, s.span.0, s.span.1, s.deg.0, s.deg.1, s.err.as_ref().unwrap());
        }
    }
}


/// Least-squares bicubic Bezier fit of a 12 x 12 sample of f over the unit
/// square, by the normal equations (16 unknowns per coordinate).
fn fit_bicubic(f: &dyn Fn(f64, f64) -> V3) -> Patch {
    let n = 12;
    let bern = |x: f64| -> [f64; 4] {
        let y = 1.0 - x;
        [y * y * y, 3.0 * x * y * y, 3.0 * x * x * y, x * x * x]
    };
    let mut ata = vec![0.0; 16 * 16];
    let mut atb = vec![[0.0; 3]; 16];
    for i in 0..n {
        for j in 0..n {
            let (s, t) = (i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64);
            let (bs, bt) = (bern(s), bern(t));
            let p = f(s, t);
            let row: Vec<f64> = (0..16).map(|k| bs[k / 4] * bt[k % 4]).collect();
            for a in 0..16 {
                for b in 0..16 {
                    ata[a * 16 + b] += row[a] * row[b];
                }
                atb[a][0] += row[a] * p.x;
                atb[a][1] += row[a] * p.y;
                atb[a][2] += row[a] * p.z;
            }
        }
    }
    // Gaussian elimination with row pivoting, three right-hand sides.
    let mut m = ata;
    let mut rhs = atb;
    for k in 0..16 {
        let piv = (k..16).max_by(|&a, &b| m[a * 16 + k].abs().partial_cmp(&m[b * 16 + k].abs()).unwrap()).unwrap();
        for j in 0..16 {
            m.swap(k * 16 + j, piv * 16 + j);
        }
        rhs.swap(k, piv);
        let d = m[k * 16 + k];
        for i in 0..16 {
            if i == k {
                continue;
            }
            let fct = m[i * 16 + k] / d;
            for j in 0..16 {
                m[i * 16 + j] -= fct * m[k * 16 + j];
            }
            for c in 0..3 {
                rhs[i][c] -= fct * rhs[k][c];
            }
        }
    }
    let pw: Vec<Vec<[f64; 4]>> = (0..4)
        .map(|i| {
            (0..4)
                .map(|j| {
                    let k = i * 4 + j;
                    let d = m[k * 16 + k];
                    [rhs[k][0] / d, rhs[k][1] / d, rhs[k][2] / d, 1.0]
                })
                .collect()
        })
        .collect();
    Patch { pw, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
}

/// Synthetic spans with a known truth. A bicubic least-squares fit of a 60
/// degree cylinder patch is an exact extrusion of a cubic curve (x and y depend
/// on s alone): a degree-drop surface, whose square M-rep carries extraneous
/// factors. Adding control-point noise of a known size makes it exactly generic
/// (implicit degree 18, the square M-rep exact) while leaving it near-degenerate
/// at that size, which is what a fitted CAD span is. The same measurements as
/// `spans`, plus the distance from the nearest "other sheet" point to the
/// cylinder, read against the noise.
pub fn lab() {
    let r = 10.0;
    let mut cases: Vec<(String, Patch, Box<dyn Fn(V3) -> f64 + Sync>)> = Vec::new();
    let ang = 60f64.to_radians();
    let h = r * ang;
    let cyl = move |s: f64, t: f64| V3::new(r * (ang * s).cos(), r * (ang * s).sin(), h * t);
    let base = fit_bicubic(&cyl);
    let (lo, hi) = base.bbox();
    let size = (hi - lo).len() * 0.5;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut unit = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for &eta in &[0.0f64, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let mut p = base.clone();
        for row in p.pw.iter_mut() {
            for c in row.iter_mut() {
                for k in 0..3 {
                    c[k] += eta * size * unit();
                }
            }
        }
        cases.push((format!("cyl 60 + noise {eta:.0e}"), p, Box::new(move |x: V3| ((x.x * x.x + x.y * x.y).sqrt() - r).abs())));
    }
    let big = 20.0;
    let tor = move |s: f64, t: f64| {
        let (a, b) = (ang * s, std::f64::consts::FRAC_PI_2 * t);
        V3::new((big + r * b.cos()) * a.cos(), (big + r * b.cos()) * a.sin(), r * b.sin())
    };
    let tbase = fit_bicubic(&tor);
    let (tlo, thi) = tbase.bbox();
    let tsize = (thi - tlo).len() * 0.5;
    for &eta in &[0.0f64, 1e-8, 1e-6, 1e-4] {
        let mut p = tbase.clone();
        for row in p.pw.iter_mut() {
            for c in row.iter_mut() {
                for k in 0..3 {
                    c[k] += eta * tsize * unit();
                }
            }
        }
        cases.push((format!("torus 60 + noise {eta:.0e}"), p, Box::new(move |x: V3| (((x.x * x.x + x.y * x.y).sqrt() - big).hypot(x.z) - r).abs())));
    }
    let sph = move |s: f64, t: f64| {
        let (a, b) = (ang * s, ang * (t - 0.5));
        V3::new(r * b.cos() * a.cos(), r * b.cos() * a.sin(), r * b.sin())
    };
    let sbase = fit_bicubic(&sph);
    for &eta in &[0.0f64, 1e-8, 1e-6, 1e-4] {
        let mut p = sbase.clone();
        for row in p.pw.iter_mut() {
            for c in row.iter_mut() {
                for k in 0..3 {
                    c[k] += eta * r * unit();
                }
            }
        }
        cases.push((format!("sphere 60 + noise {eta:.0e}"), p, Box::new(move |x: V3| (x.len() - r).abs())));
    }
    {
        // A generic bicubic: random control net.
        let pw: Vec<Vec<[f64; 4]>> = (0..4).map(|i| (0..4).map(|j| [i as f64 * 10.0 + 3.0 * unit(), j as f64 * 10.0 + 3.0 * unit(), 4.0 * unit(), 1.0]).collect()).collect();
        cases.push(("random net".into(), Patch { pw, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }, Box::new(|_x: V3| f64::NAN)));
    }
    println!("{:<22} {:>9} {:>3} {:>12} {:>12} {:>9} {:>9} {:>16} {:>14} {:>10} {:>10} {:>12}", "case", "fit err", "r", "spectrum 18", "spectrum 19", "rank drop", "cross", "sheets p/c/f/o", "nearest other", "to truth", "wide@other", "foreign(1e-8)");
    println!("  fit err and distances in span sizes (half the box diagonal); spectrum 18 and 19 are the last of the 18 directions of an exact bicubic and the first past them");
    // The kernel's rate: the determinant of the square 18 x 18 M-rep (LU) and the
    // smallest singular value of the floor-based one (Jacobi), one thread, at
    // random points in the first case's padded box.
    {
        let patch = &cases[0].1;
        let sq = MRep::square(patch).unwrap();
        let wide = MRep::with_tol(patch, MRep::MEASURE_FLOOR).unwrap();
        let (lo, hi) = patch.bbox();
        let pts: Vec<V3> = (0..200_000).map(|_| V3::new(lo.x + (hi.x - lo.x) * (0.5 + 0.6 * unit()), lo.y + (hi.y - lo.y) * (0.5 + 0.6 * unit()), lo.z + (hi.z - lo.z) * (0.5 + 0.6 * unit()))).collect();
        let t = std::time::Instant::now();
        let mut acc = 0.0;
        for q in &pts {
            acc += sq.det(*q).unwrap().signum();
        }
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(acc);
        let t2 = std::time::Instant::now();
        let mut acc2 = 0.0;
        for q in pts.iter().take(20_000) {
            acc2 += wide.sigma_min(*q);
        }
        let dt2 = t2.elapsed().as_secs_f64();
        std::hint::black_box(acc2);
        println!("kernel rate, one thread, f64: det of the 18 x 18 square M-rep {:.2e} points/s ({:.2} us); sigma_min of the {} x {} floor-based M-rep {:.2e} points/s ({:.1} us)", pts.len() as f64 / dt, 1e6 * dt / pts.len() as f64, wide.m, wide.r, 20_000.0 / dt2, 1e6 * dt2 / 20_000.0);
    }
    for (name, patch, truth) in &cases {
        let st = span_stat(0, (0, 0), patch);
        let rep = MRep::square(patch).unwrap();
        let (lo, hi) = patch.bbox();
        let scale = (hi - lo).len() * 0.5;
        let mut fit_err = 0.0f64;
        for i in 0..=20 {
            for j in 0..=20 {
                let e = patch.eval(i as f64 / 20.0, j as f64 / 20.0).p;
                let d = truth(e);
                if d.is_finite() {
                    fit_err = fit_err.max(d / scale);
                }
            }
        }
        let to_truth = st.sheet_near_pt.map(|x| truth(x) / scale).unwrap_or(f64::NAN);
        println!("{:<22} {:>9.1e} {:>3} {:>12.1e} {:>12.1e} {:>9.1e} {:>9.1e} {:>4}/{:>3}/{:>3}/{:<4} {:>14.3} {:>10.3} {:>10.1e} {:>12}", name, fit_err, st.r, rep.spectrum[17], rep.spectrum[18], st.on_ratio, st.cross_ratio, st.sheet_patch, st.sheet_cont, st.sheet_far, st.sheet_other, st.sheet_near, to_truth, st.wide_at_other, st.foreign_confirmed);
    }
}


/// One file of a folder run: its census and the measured sample.
struct FileRow {
    name: String,
    note: String,
    solids: (usize, usize),
    faces: usize,
    spans: usize,
    degrees: std::collections::BTreeMap<(usize, usize), usize>,
    rational: usize,
    stats: Vec<SpanStat>,
    secs: f64,
}

fn file_row(path: &Path, per_file: usize, census_only: bool) -> FileRow {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let t0 = std::time::Instant::now();
    let mut row = FileRow { name, note: String::new(), solids: (0, 0), faces: 0, spans: 0, degrees: Default::default(), rational: 0, stats: Vec::new(), secs: 0.0 };
    let brep = match step::read(path) {
        Ok(b) => b,
        Err(e) => {
            row.note = format!("read: {e}");
            return row;
        }
    };
    let tol = brep.tolerance_mm.max(1e-6);
    row.solids.1 = brep.solids.len();
    let mut jobs: Vec<(u32, (usize, usize), Patch)> = Vec::new();
    // An assembly places one solid many times; a face's geometry is measured once
    // per distinct control net (the placement moves it, the M-rep's statistics are
    // invariant under it), keyed by the face's STEP id.
    let mut seen = std::collections::HashSet::new();
    for s in &brep.solids {
        let sol = match Solid3::from_step(s, tol) {
            Ok((sol, _)) => sol,
            Err(e) => {
                if row.note.is_empty() {
                    row.note = format!("solid #{}: {e}", s.id);
                }
                continue;
            }
        };
        row.solids.0 += 1;
        for f in &sol.faces {
            let Surf::Nurbs(ns) = &f.surf else { continue };
            if !seen.insert(f.id) {
                continue;
            }
            row.faces += 1;
            let (nu, nv) = ns.spans_uv();
            for a in 0..nu {
                for b in 0..nv {
                    let p = ns.patch(a, b);
                    let deg = (p.pw.len() - 1, p.pw[0].len() - 1);
                    *row.degrees.entry(deg).or_insert(0) += 1;
                    let w0 = p.pw[0][0][3];
                    if p.pw.iter().flatten().any(|h| (h[3] - w0).abs() > 1e-12 * w0.abs()) {
                        row.rational += 1;
                    }
                    jobs.push((f.id, (a, b), p));
                }
            }
        }
    }
    row.spans = jobs.len();
    if !census_only && !jobs.is_empty() {
        if per_file > 0 && jobs.len() > per_file {
            let step = jobs.len() as f64 / per_file as f64;
            jobs = (0..per_file).map(|i| jobs[(i as f64 * step) as usize].clone()).collect();
        }
        row.stats = jobs.par_iter().map(|(f, sp, p)| span_stat(*f, *sp, p)).collect();
    }
    row.secs = t0.elapsed().as_secs_f64();
    row
}

/// `icam-check spans <dir> [n]`: every STEP file in a folder, up to n spans each
/// (an even subsample), one line per file and the tally. ICAM_CENSUS_ONLY=1 counts
/// spans and degrees without measuring.
pub fn spans_dir(dir: &Path, per_file: usize) {
    let census_only = std::env::var("ICAM_CENSUS_ONLY").is_ok();
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| matches!(x.to_string_lossy().to_lowercase().as_str(), "step" | "stp")))
        .collect();
    files.sort();
    println!("{} STEP files in {}, up to {} spans measured per file{}", files.len(), dir.display(), per_file, if census_only { " (census only)" } else { "" });
    println!(
        "{:<26} {:>7} {:>5} {:>6} {:>5} {:>4} {:>8} {:>7} {:>7} {:>9} {:>6} {:>7} {:>8} {:>7} {:>9} {:>6}  {}",
        "file", "solids", "faces", "spans", "meas", "ref", "ratnl", "no gap", "r max", "on max", "cross", "flr zero", "sq other", "nearest", "wide@oth", "secs", "degrees"
    );
    let mut all: Vec<FileRow> = Vec::new();
    for f in &files {
        let row = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| file_row(f, per_file, census_only))) {
            Ok(r) => r,
            Err(e) => {
                let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                FileRow { name: f.file_name().unwrap().to_string_lossy().to_string(), note: format!("PANIC: {msg}"), solids: (0, 0), faces: 0, spans: 0, degrees: Default::default(), rational: 0, stats: Vec::new(), secs: 0.0 }
            }
        };
        let ok: Vec<&SpanStat> = row.stats.iter().filter(|s| s.err.is_none()).collect();
        let refused = row.stats.len() - ok.len();
        let nogap = ok.iter().filter(|s| s.s_next < 100.0 * s.s_last.max(1e-300)).count();
        let rmax = ok.iter().map(|s| s.r).max().unwrap_or(0);
        let on_max = ok.iter().map(|s| s.on_ratio).fold(0.0f64, f64::max);
        let cross = ok.iter().filter(|s| s.cross_confirmed > 0).count();
        let flr = ok.iter().filter(|s| s.foreign_confirmed > 0).count();
        let other = ok.iter().filter(|s| s.sheet_other > 0).count();
        let nearest = ok.iter().map(|s| s.sheet_near).fold(f64::INFINITY, f64::min);
        let mut wides: Vec<f64> = ok.iter().filter(|s| s.sheet_other > 0).map(|s| s.wide_at_other).collect();
        let degs: Vec<String> = row.degrees.iter().map(|((a, b), n)| format!("({a},{b})x{n}")).collect();
        println!(
            "{:<26} {:>3}/{:<3} {:>5} {:>6} {:>5} {:>4} {:>8} {:>7} {:>7} {:>9.1e} {:>6} {:>7} {:>8} {:>7.3} {:>9.1e} {:>6.1}  {}{}",
            row.name.chars().take(26).collect::<String>(),
            row.solids.0,
            row.solids.1,
            row.faces,
            row.spans,
            row.stats.len(),
            refused,
            row.rational,
            nogap,
            rmax,
            on_max,
            cross,
            flr,
            other,
            nearest,
            pct(&mut wides, 0.5),
            row.secs,
            degs.join(" "),
            if row.note.is_empty() { String::new() } else { format!("  [{}]", row.note) }
        );
        for s in row.stats.iter().filter(|s| s.err.is_some()).take(3) {
            println!("    refused face #{} span ({},{}) degree ({},{}): {}", s.face, s.span.0, s.span.1, s.deg.0, s.deg.1, s.err.as_ref().unwrap());
        }
        use std::io::Write;
        std::io::stdout().flush().ok();
        all.push(row);
    }
    // The tally.
    let stats: Vec<&SpanStat> = all.iter().flat_map(|r| r.stats.iter()).collect();
    let ok: Vec<&&SpanStat> = stats.iter().filter(|s| s.err.is_none()).collect();
    let mut degrees: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
    for r in &all {
        for (d, n) in &r.degrees {
            *degrees.entry(*d).or_insert(0) += n;
        }
    }
    let total_spans: usize = all.iter().map(|r| r.spans).sum();
    let rational: usize = all.iter().map(|r| r.rational).sum();
    println!();
    println!("files {}; with B-spline spans {}; spans {} ({} rational); measured {}; refused {}", all.len(), all.iter().filter(|r| r.spans > 0).count(), total_spans, rational, stats.len(), stats.len() - ok.len());
    let degs: Vec<String> = degrees.iter().map(|((a, b), n)| format!("({a},{b}) {n}")).collect();
    println!("degrees: {}", degs.join(", "));
    if ok.is_empty() {
        return;
    }
    let mut ons: Vec<f64> = ok.iter().map(|s| s.on_ratio).collect();
    let mut nexts: Vec<f64> = ok.iter().map(|s| s.s_next).collect();
    let mut resid: Vec<f64> = ok.iter().map(|s| s.residual).collect();
    println!("rank drop on the patch (worst sample per span): p50 {:.1e} p99 {:.1e} max {:.1e}; moving-plane residual max {:.1e}", pct(&mut ons, 0.5), pct(&mut ons, 0.99), pct(&mut ons, 1.0), pct(&mut resid, 1.0));
    println!("first system direction left out (relative): p1 {:.1e} p10 {:.1e} p50 {:.1e} p90 {:.1e}", pct(&mut nexts, 0.01), pct(&mut nexts, 0.1), pct(&mut nexts, 0.5), pct(&mut nexts, 0.9));
    let by = |pred: &dyn Fn(&SpanStat) -> bool| ok.iter().filter(|s| pred(s)).count();
    println!("by degree: {:<8} {:>6} {:>7} {:>8} {:>8} {:>8} {:>9}", "degree", "spans", "no gap", "cross", "flr zero", "sq other", "wide@oth");
    for d in degrees.keys() {
        let sel: Vec<&&&SpanStat> = ok.iter().filter(|s| s.deg == *d).collect();
        if sel.is_empty() {
            continue;
        }
        let mut w: Vec<f64> = sel.iter().filter(|s| s.sheet_other > 0).map(|s| s.wide_at_other).collect();
        println!(
            "           ({},{})    {:>6} {:>7} {:>8} {:>8} {:>8} {:>9.1e}",
            d.0,
            d.1,
            sel.len(),
            sel.iter().filter(|s| s.s_next < 100.0 * s.s_last.max(1e-300)).count(),
            sel.iter().filter(|s| s.cross_confirmed > 0).count(),
            sel.iter().filter(|s| s.foreign_confirmed > 0).count(),
            sel.iter().filter(|s| s.sheet_other > 0).count(),
            pct(&mut w, 0.5)
        );
    }
    let mut nears: Vec<f64> = ok.iter().filter(|s| s.sheet_other > 0).map(|s| s.sheet_near).collect();
    let mut wides: Vec<f64> = ok.iter().filter(|s| s.sheet_other > 0).map(|s| s.wide_at_other).collect();
    println!(
        "spans crossed by their own extension: {}; floor M-rep zeros over 2% of the span from the patch: {} spans; square det other sheets in the box: {} spans, nearest min {:.3} p10 {:.3} p50 {:.3}; floor M-rep ratio at those points p10 {:.1e} p50 {:.1e} p90 {:.1e}",
        by(&|s| s.cross_confirmed > 0),
        by(&|s| s.foreign_confirmed > 0),
        by(&|s| s.sheet_other > 0),
        pct(&mut nears, 0.0),
        pct(&mut nears, 0.1),
        pct(&mut nears, 0.5),
        pct(&mut wides, 0.1),
        pct(&mut wides, 0.5),
        pct(&mut wides, 0.9)
    );
}
