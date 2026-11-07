//! icam-check: reads exact B-rep solids from STEP, builds the exact part, and
//! checks the GPU contact query against the exact CPU reference.
//!
//!   icam-check survey <dir>              parse every STEP file in a folder; entities and prismatic verdicts
//!   icam-check import <file.step>        import one file's prismatic solids, report
//!   icam-check gpu <file.step> [n] [--ref m]   GPU vs CPU on n random points per tool and mode (m on the CPU)
//!   icam-check frep <file.step>          compile the F-rep; check sign, bound, Lipschitz and per-cell pruning
//!   icam-check solid <file.step> [n]     exact 3D signed distance of general solids; Lipschitz and sign checks
//!   icam-check spans <file.step> [n]     M-rep of every B-spline span (n at most): size, residual, second sheets, foreign zeros
//!   icam-check spans <dir> [n]           the same over every STEP file in a folder, n spans per file (60 by default)
//!   icam-check spanpoly <file|dir> [n]   signed implicit polynomial of each span (n per file): degree, exactness, extraneous zeros, f32
//!   icam-check spanpolygpu <file> [n]   the same polynomials on the GPU (f32) against the CPU (f64): rates and sign agreement
//!   icam-check spanshape <file|dir>      each span's nearest plane, sphere, cylinder or cone and its distance from it
//!   icam-check analytic <file|dir> [n] [s]  B-spline faces read as the planes, spheres, cylinders, cones they are: counts, distance agreement, speed
//!   icam-check surfcheck <file|dir> [n]  the importer's analytic readings against samples of the exact face surfaces (no B-rep distance)
//!   icam-check newton <file|dir>         fixed Newton steps from a per-brick seed against the certified nearest-point search on B-spline faces
//!   icam-check trim2 <file|dir> [n]       certified parameter-domain trim loops against the path-walking trim test, n random points per B-spline face
//!   icam-check gaps <file|dir>           where local rules have no formula yet: mixed vertices, tangent and changing edges, features under the refinement floor
//!   icam-check spanlab                   the same on synthetic bicubic fits of cylinder patches with known fit error

use std::path::{Path, PathBuf};

use icam_core::{prismatic, step};

mod exact_check;
mod gaps;
mod newton_check;
mod trim2_check;
mod span_poly_check;
mod spans;
mod surfcheck;

fn load(path: &Path) -> Result<(icam_core::part::Part, step::Brep, prismatic::PrismReport, f64), String> {
    let t0 = std::time::Instant::now();
    let brep = step::read(path)?;
    let (part, rep) = prismatic::from_brep(&brep)?;
    Ok((part, brep, rep, t0.elapsed().as_secs_f64()))
}

fn survey(dir: &Path) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| matches!(x.to_string_lossy().to_lowercase().as_str(), "step" | "stp")))
        .collect();
    files.sort();
    let (mut ok, mut prism, mut failed) = (0, 0, 0);
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    let t0 = std::time::Instant::now();
    for f in &files {
        match step::read(f) {
            Err(e) => {
                failed += 1;
                println!("PARSE FAIL  {}: {e}", f.file_name().unwrap().to_string_lossy());
            }
            Ok(b) => {
                ok += 1;
                if b.solids.is_empty() {
                    *reasons.entry("no solid (surfaces or assembly references only)".into()).or_default() += 1;
                    continue;
                }
                match prismatic::from_brep(&b) {
                    Ok((part, _)) => {
                        prism += 1;
                        println!(
                            "prismatic   {}: {} solid(s), {} horizontal faces, {} curves ({})",
                            f.file_name().unwrap().to_string_lossy(),
                            b.solids.len(),
                            part.faces.len(),
                            part.edges.len(),
                            b.unit_name
                        );
                    }
                    Err(e) => {
                        // Bucket by the kind of refusal, not the face number.
                        let key = e.split(": ").skip(1).collect::<Vec<_>>().join(": ");
                        let key = key.split(" (").next().unwrap_or(&key).to_string();
                        let key = if key.starts_with("plane tilted") { "tilted plane".to_string() } else { key };
                        *reasons.entry(key).or_default() += 1;
                    }
                }
            }
        }
    }
    println!(
        "
{} STEP files in {:.1} s: {ok} parsed, {failed} failed; {prism} prismatic in their own z",
        files.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut r: Vec<_> = reasons.into_iter().collect();
    r.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in r {
        println!("  not prismatic: {n:4}  {k}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    if cmd == "spanlab" {
        spans::lab();
        return;
    }
    let path = PathBuf::from(args.get(2).expect("usage: icam-check survey <dir> | import <file.step> | gpu <file.step> [n]"));
    if cmd == "survey" {
        survey(&path);
        return;
    }
    if cmd == "newton" {
        newton_check::run(&path);
        return;
    }
    if cmd == "trim2" {
        let per = std::env::args().nth(3).and_then(|x| x.parse().ok()).unwrap_or(200);
        trim2_check::run(&path, per);
        return;
    }
    if cmd == "exact" {
        let n: usize = args.iter().skip(3).find_map(|a| a.parse().ok()).unwrap_or(1000);
        exact_check::run(&path, n, &args);
        return;
    }
    if cmd == "gaps" {
        gaps::run(&path);
        return;
    }
    if cmd == "surfcheck" {
        let n: usize = args.iter().skip(3).find_map(|a| a.parse().ok()).unwrap_or(48);
        surfcheck::run(&path, n);
        return;
    }
    if cmd == "faceprobe" {
        // One raw B-spline face: its recognition, and how far its edges' ends are from the recognized surface.
        let id: u32 = args.get(3).and_then(|a| a.parse().ok()).expect("face id (0: every unrecognized B-spline face, briefly)");
        let brep = step::read_raw(&path).unwrap();
        if id == 0 {
            for s in &brep.solids {
                for f in &s.faces {
                    let step::Surface::BSpline(b) = &f.surface else { continue };
                    let Ok(ns) = icam_core::surface3::NurbsSurf::new(b.du, b.dv, &b.ctrl, b.weights.as_ref(), &b.ku, &b.kv) else { continue };
                    if icam_core::span_poly::recognize(&ns).is_some() {
                        continue;
                    }
                    let (nu, nv) = ns.spans_uv();
                    let p0 = ns.patch(0, 0);
                    let (lo, hi) = p0.bbox();
                    let (c, sc) = ((lo + hi) * 0.5, ((hi - lo).len() * 0.5).max(1e-300));
                    let mut best = (f64::INFINITY, String::new());
                    for k in 1..=2usize {
                        let sp = icam_core::span_poly::SpanPoly::quadric_of(&p0);
                        let _ = (k, c, sc);
                        for cand in sp.analytic_candidates() {
                            let d = icam_core::span_poly::certified_distance(&cand, &p0);
                            if d < best.0 {
                                best = (d, cand.name().to_string());
                            }
                        }
                    }
                    if let Some(t) = icam_core::span_poly::torus_candidate(&p0) {
                        let d = icam_core::span_poly::certified_distance(&t, &p0);
                        if let icam_core::surface3::Surf::Torus { major, minor, .. } = t {
                            println!("face #{} degree ({},{}) {}x{} spans: torus candidate R {major:.6} r {minor:.6} certified {d:.2e}; best other {} {:.2e}", f.id, b.du, b.dv, nu, nv, best.1, best.0);
                            continue;
                        }
                    }
                    println!("face #{} degree ({},{}) {}x{} spans: no torus candidate; best other {} {:.2e}", f.id, b.du, b.dv, nu, nv, best.1, best.0);
                }
            }
            return;
        }
        for s in &brep.solids {
            for f in &s.faces {
                if f.id != id {
                    continue;
                }
                let step::Surface::BSpline(b) = &f.surface else { println!("face #{id} is not a B-spline"); return };
                let ns = icam_core::surface3::NurbsSurf::new(b.du, b.dv, &b.ctrl, b.weights.as_ref(), &b.ku, &b.kv).unwrap();
                let (nu, nv) = ns.spans_uv();
                println!("face #{id}: degree ({},{}), {} x {} spans, rational {}", b.du, b.dv, nu, nv, b.weights.is_some());
                let p0 = ns.patch(0, 0);
                for d in icam_core::span_poly::SpanPoly::profile(&p0, 4) {
                    println!("  degree {}: sigma {:.1e} next {:.1e} on span {:.1e} far {:.1e}", d.k, d.sigma, d.sigma_next, d.on_span, d.ext);
                }
                let pts: Vec<String> = p0.pw.iter().flatten().map(|h| format!("({:.6},{:.6},{:.6} w{:.4})", h[0] / h[3], h[1] / h[3], h[2] / h[3], h[3])).collect();
                println!("  control net {}", pts.join(" "));
                match icam_core::span_poly::SpanPoly::search(&p0, 2) {
                    Ok(sp) => {
                        println!("  span (0,0) polynomial degree {} sigma {:.1e} next {:.1e}", sp.degree, sp.sigma, sp.sigma_next);
                        for c in sp.analytic_candidates() {
                            let mut dev = 0.0f64;
                            for i in 0..=24 {
                                for j in 0..=24 {
                                    dev = dev.max(c.dist(p0.eval(i as f64 / 24.0, j as f64 / 24.0).p));
                                }
                            }
                            let ends: f64 = f.bounds.iter().flat_map(|bd| bd.edges.iter()).map(|e| c.dist(e.start)).fold(0.0, f64::max);
                            println!("    candidate {:?}: span dev {:.3e}, edge ends {:.3e}, certified {:.3e}", c, dev, ends, icam_core::span_poly::certified_distance(&c, &p0));
                        }
                    }
                    Err(e) => println!("  span (0,0): {}", e.reason),
                }
                match icam_core::span_poly::recognize(&ns) {
                    Some((surf, flip)) => {
                        println!("  recognized {:?} flip {flip}", surf);
                        for bd in &f.bounds {
                            for e in &bd.edges {
                                for q in [e.start, e.end] {
                                    println!("  edge #{} end {:?}: distance to the recognized surface {:.3e}, to the B-spline {:.3e}", e.edge, q, surf.dist(q), ns.closest(q, f64::INFINITY).map_or(f64::NAN, |h| (h.q - q).len()));
                                }
                            }
                        }
                    }
                    None => println!("  not recognized"),
                }
            }
        }
        return;
    }
    if cmd == "analytic" {
        let n: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(2000);
        let secs: f64 = args.get(4).and_then(|a| a.parse().ok()).unwrap_or(60.0);
        span_poly_check::analytic(&path, n, secs);
        return;
    }
    if cmd == "spanshape" {
        span_poly_check::shape(&path);
        return;
    }
    if cmd == "spanpolygpu" {
        let n: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(1 << 20);
        span_poly_check::gpu_bench(&path, n);
        return;
    }
    if cmd == "spanpoly" {
        let n: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(0);
        span_poly_check::run(&path, n);
        return;
    }
    if cmd == "spans" {
        let n: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(0);
        if path.is_dir() {
            spans::spans_dir(&path, if n == 0 { 60 } else { n });
        } else {
            spans::spans(&path, n);
        }
        return;
    }
    if cmd == "census" {
        census(&path);
        return;
    }
    if cmd == "seps" {
        // The tangent-edge separators of each solid (numbered after the faces, as the retired per-cell compiler numbered its primitives): edge, curve, ends, faces.
        let brep = step::read(&path).unwrap();
        let tol = brep.tolerance_mm.max(1e-6);
        for s in &brep.solids {
            let Ok((sol, _)) = icam_core::solid3::Solid3::from_step(s, tol) else { continue };
            let mut k = sol.faces.len();
            for (ei, e) in sol.edges.iter().enumerate() {
                if e.faces[0] == e.faces[1] {
                    continue;
                }
                let (_, n1, n2, _) = sol.edge_frame(ei, e.curve.at(0.5));
                if n1.cross(n2).len() < 1e-6 && n1.dot(n2) > 0.0 {
                    println!("prim {k}: edge {ei} #{} {} from {:?} to {:?}, faces {} #{} ({}) and {} #{} ({})", e.id, icam_core::solid3::edge_kind_name(&e.curve), e.curve.start(), e.curve.end(), e.faces[0], sol.faces[e.faces[0]].id, sol.faces[e.faces[0]].surf.name(), e.faces[1], sol.faces[e.faces[1]].id, sol.faces[e.faces[1]].surf.name());
                    k += 1;
                }
            }
        }
        return;
    }
    if cmd == "probe" {
        use icam_core::curve3::V3;
        let v: Vec<f64> = args[3..6].iter().map(|a| a.parse().unwrap()).collect();
        let p = V3::new(v[0], v[1], v[2]);
        let brep = step::read(&path).unwrap();
        for s in &brep.solids {
            let (sol, _) = icam_core::solid3::Solid3::from_step(s, brep.tolerance_mm.max(1e-6)).unwrap();
            let t0 = std::time::Instant::now();
            let sd = sol.sdf(p);
            println!(
                "sdf {:?} in {:.1} ms, trim tests {}, curve pair visits {}",
                sd,
                t0.elapsed().as_secs_f64() * 1e3,
                icam_core::solid3::TRIM_CALLS.load(std::sync::atomic::Ordering::Relaxed),
                icam_core::curve3::NEAR_CALLS.load(std::sync::atomic::Ordering::Relaxed)
            );
            for (fi, f) in sol.faces.iter().enumerate() {
                let q = f.surf.foot(p);
                let c = sol.classify(fi, q);
                println!("face {fi} #{} {} sense {} foot {:?} d {:.6} classify {:?}", f.id, f.surf.name(), f.sense, q, (p - q).len(), c);
                if let Ok(ei) = std::env::var("ICAM_EDGE").map(|v| v.parse::<usize>().unwrap()) {
                    if fi == 0 {
                        println!("  edge {ei}: {:?}", sol.edges[ei].curve);
                    }
                }
                if std::env::var("ICAM_LOOPS").map_or(false, |v| v.parse::<usize>() == Ok(fi)) {
                    println!("  frame {:?} seams {:?}", f.surf.frame(), f.seams);
                    for lp in &f.loops {
                        for u in lp {
                            let e = &sol.edges[u.edge];
                            println!("  edge {} #{} rev {} {:?} -> {:?}", u.edge, e.id, u.rev, e.curve.start(), e.curve.end());
                        }
                    }
                }
            }
        }
        return;
    }
    if cmd == "solid" {
        let n: usize = args.iter().skip(3).find_map(|a| a.parse().ok()).unwrap_or(20_000);
        solid_check(&path, n);
        return;
    }
    let (part, brep, rep, secs) = match load(&path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("import refused: {e}");
            std::process::exit(2);
        }
    };
    println!(
        "{}: {} solid(s), unit {} -> {} horizontal faces ({} vertical planes, {} vertical cylinders, {} vertical freeform walls), {} lines + {} arcs + {} native B-splines ({} flat Bezier pieces) in {} loops, in {:.0} ms",
        path.display(),
        brep.solids.len(),
        brep.unit_name,
        rep.horizontal,
        rep.vertical_planes,
        rep.vertical_cylinders,
        rep.vertical_freeform,
        rep.lines,
        rep.arcs,
        rep.splines,
        rep.spline_pieces,
        part.faces.iter().map(|f| f.loops.len()).sum::<usize>(),
        secs * 1e3
    );
    println!(
        "vertices sit on their exact planes and curves to {:.1e} mm; line/arc ends moved at most {:.1e} mm onto their junctions, arc-to-spline gaps at most {:.1e} mm (file tolerance {:.1e} mm)",
        rep.max_vertex_off, rep.max_snap, rep.max_gap, rep.tolerance_mm
    );
    match cmd {
        "import" => {}
        "frep" => frep_check(&part, rep.tolerance_mm.max(1e-6)),
        "gpu" => {
            let n: usize = args.iter().skip(3).find_map(|a| a.parse().ok()).unwrap_or(2_000_000);
            let m = args.iter().position(|a| a == "--ref").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(n.min(100_000));
            gpu_check(&part, n, m);
        }
        other => eprintln!("unknown command {other}"),
    }
}

/// xorshift64*: deterministic, no dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

/// Query points: half uniform over the part's box, half within 12 mm of a
/// boundary curve (where signs, vertices and arcs are decided). Slice heights
/// stay 1 um away from every face height so f32 and f64 agree on "above".
fn queries(part: &icam_core::part::Part, n: usize, seed: u64) -> Vec<[f64; 3]> {
    use icam_core::geom::{Curve, V2};
    let mut rng = Rng(seed);
    let (lo, hi) = part.bbox();
    let zs: Vec<f64> = part.faces.iter().map(|f| f.z).collect();
    let (zmin, zmax) = zs.iter().fold((f64::MAX, f64::MIN), |(a, b), &z| (a.min(z), b.max(z)));
    (0..n)
        .map(|k| {
            let p = if k % 2 == 0 {
                V2::new(rng.range(lo.x - 10.0, hi.x + 10.0), rng.range(lo.y - 10.0, hi.y + 10.0))
            } else {
                let e = &part.edges[(rng.next() * part.edges.len() as f64) as usize % part.edges.len()];
                let t = rng.next();
                let on = match &e.curve {
                    Curve::Line { a, b } => *a + (*b - *a) * t,
                    Curve::Arc { c, a, sweep, .. } => *c + (*a - *c).rotate(sweep * t),
                    Curve::Spline(s) => {
                        let k = ((t * s.pieces.len() as f64) as usize).min(s.pieces.len() - 1);
                        s.pieces[k].eval(rng.next()).p
                    }
                };
                let ang = rng.range(0.0, 2.0 * std::f64::consts::PI);
                on + V2::new(ang.cos(), ang.sin()) * rng.range(0.0, 12.0)
            };
            let mut z = rng.range(zmin - 1.0, zmax + 1.0);
            while zs.iter().any(|&f| (f - z).abs() < 1e-3) {
                z += 2.1e-3;
            }
            [p.x, p.y, z]
        })
        .collect()
}

fn gpu_check(part: &icam_core::part::Part, n: usize, m: usize) {
    use icam_core::query::{drop_ref, slice_ref, NONE};
    use icam_core::tool::Tool;
    use icam_gpu::{Gpu, GpuPart, Mode};
    use rayon::prelude::*;

    let gpu = Gpu::new().expect("GPU");
    println!("GPU: {}", gpu.adapter);
    let tools = vec![
        Tool::flat(12.7, 40.0),
        Tool::ball(6.35, 25.0),
        Tool::bull(12.7, 1.5875, 40.0),
        Tool::vee(12.7, 90.0, 30.0),
        Tool::flat(12.7, 40.0).with_holder(25.0, 32.0),
    ];
    let band = tools.iter().map(|t| t.r_max()).fold(0.0, f64::max) + 4.0;
    let t0 = std::time::Instant::now();
    let gp = match GpuPart::new(&gpu, part, band, 2.0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU pack refused: {e}");
            std::process::exit(2);
        }
    };
    let pk = &gp.packed;
    println!(
        "packed: {} curves, grid {} x {} cells of {} mm, band {} mm, {} curve refs + {} cover refs ({:.1} MB) in {:.2} s",
        pk.edges.len(),
        pk.nx,
        pk.ny,
        pk.cell,
        band,
        pk.item_count,
        pk.cover_count,
        (pk.idx.len() * 4 + pk.edges.len() * 64) as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let pts = queries(part, n, 0x9E37_79B9_7F4A_7C15);
    let m = m.min(n);
    println!("{n} queries per tool and mode on the GPU; the first {m} also on the CPU reference (f64, all faces, ray-crossing inside test)");
    println!(
        "{:<34} {:<5} {:>9} {:>10} {:>10} {:>9} {:>9} {:>9} {:>8} {:>11} {:>9}",
        "tool", "mode", "GPU Mq/s", "err max", "err p99.9", "face ok", "curve ok", "grad err", "clamped", "CPU q/s", "speedup"
    );
    for tool in &tools {
        for mode in [Mode::Drop, Mode::Slice] {
            let _ = gp.query(&gpu, tool, mode, &pts[..m.min(1000)]); // warm-up
            let (res, timing) = match gp.query(&gpu, tool, mode, &pts) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("{} {:?}: {e}", tool.name, mode);
                    std::process::exit(3);
                }
            };
            let tc = std::time::Instant::now();
            let reference: Vec<_> = pts[..m]
                .par_iter()
                .map(|p| {
                    let q = icam_core::geom::V2::new(p[0], p[1]);
                    if mode == Mode::Drop {
                        drop_ref(part, tool, q)
                    } else {
                        slice_ref(part, tool, q, p[2])
                    }
                })
                .collect();
            let cpu_s = tc.elapsed().as_secs_f64();
            let (mut errs, mut unamb, mut face_ok, mut curve_n, mut curve_ok, mut clamped, mut none_bad) = (Vec::new(), 0, 0, 0, 0, 0, 0);
            let (mut steep, mut steep_err) = (0, 0.0f64);
            let mut grad_err: f64 = 0.0;
            for ((g, c), p) in res[..m].iter().zip(&reference).zip(&pts[..m]) {
                if g.flags & (icam_gpu::CLAMP_COVER | icam_gpu::CLAMP_BAND) != 0 {
                    clamped += 1;
                    continue;
                }
                if g.flags & icam_gpu::STEEP != 0 {
                    steep += 1;
                    if c.face != NONE {
                        steep_err = steep_err.max((g.value - c.value).abs());
                    }
                    continue;
                }
                if c.face == NONE || g.face == NONE {
                    if (c.face == NONE) != (g.face == NONE) {
                        none_bad += 1;
                    }
                    continue;
                }
                errs.push((g.value - c.value).abs());
                if (c.value - c.second).abs() > 1e-3 {
                    unamb += 1;
                    if g.face == c.face {
                        face_ok += 1;
                        if c.edge != NONE {
                            curve_n += 1;
                            if g.edge == c.edge {
                                curve_ok += 1;
                                // A gradient is a direction from the contact point; closer than
                                // 10 um to it, f32 cannot resolve the direction.
                                if (icam_core::geom::V2::new(p[0], p[1]) - c.q).len() > 0.01 {
                                    grad_err = grad_err.max((g.grad - c.grad).len());
                                }
                            }
                        }
                    }
                }
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |q: f64| errs.get(((errs.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0.0);
            println!(
                "{:<34} {:<5} {:>9.1} {:>10.2e} {:>10.2e} {:>8.3}% {:>8.3}% {:>9.1e} {:>7.2}% {:>11.0} {:>8.0}x{}",
                tool.name,
                if mode == Mode::Drop { "drop" } else { "slice" },
                n as f64 / timing.kernel_s / 1e6,
                pct(1.0),
                pct(0.999),
                100.0 * face_ok as f64 / unamb.max(1) as f64,
                100.0 * curve_ok as f64 / curve_n.max(1) as f64,
                grad_err,
                100.0 * clamped as f64 / m as f64,
                m as f64 / cpu_s,
                (n as f64 / timing.kernel_s) / (m as f64 / cpu_s),
                if none_bad > 0 { format!("  ({none_bad} reach disagreements)") } else { String::new() }
            );
            if steep > 0 {
                println!("{:<34} {:<5} {steep} contacts on the tool's near-vertical side flagged for CPU certification (f32 error there up to {steep_err:.1e} mm)", "", "");
            }
        }
    }
}

/// Compiles the part's F-rep and checks it against the B-rep: sign against
/// point-in-solid, |f| never above the exact distance to the boundary, 1-Lipschitz,
/// and per-cell pruned trees equal to the global function.
fn frep_check(part: &icam_core::part::Part, tol: f64) {
    use icam_core::frep::compile_prismatic;
    use icam_core::prismatic_reference::{exact_boundary_distance, inside_solid};
    let t0 = std::time::Instant::now();
    let f = match compile_prismatic(part, tol) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("F-rep compile refused: {e}");
            std::process::exit(2);
        }
    };
    println!(
        "F-rep: {} levels (z {:?}), {} section loops, {} leaves, compiled in {:.1} ms",
        f.levels.len(),
        f.levels.iter().map(|z| (z * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        f.sections.iter().map(|s| s.faces[0].loops.len()).sum::<usize>(),
        f.leaf_count(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    let (lo, hi) = part.bbox();
    let (z0, z1) = (f.z_bot, *f.levels.last().unwrap());
    let pad = 3.0;
    let mut rng = Rng(0xC0FF_EE12_3456_789A);
    let n = if part.edges.iter().any(|e| matches!(e.curve, icam_core::geom::Curve::Spline(_))) { 3_000 } else { 20_000 };
    let (mut sign_bad, mut bound_bad, mut lip_bad, mut exact_hits) = (0, 0, 0, 0);
    let mut worst_excess: f64 = 0.0;
    let mut prev: Option<([f64; 3], f64)> = None;
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        let p = [rng.range(lo.x - pad, hi.x + pad), rng.range(lo.y - pad, hi.y + pad), rng.range(z0 - pad, z1 + pad)];
        let v = f.eval(p).value;
        let ex = exact_boundary_distance(part, p);
        if ex > 1e-6 && (v < 0.0) != inside_solid(part, p) {
            sign_bad += 1;
        }
        if v.abs() > ex + 1e-9 {
            bound_bad += 1;
            worst_excess = worst_excess.max(v.abs() - ex);
        }
        if (v.abs() - ex).abs() < 1e-9 {
            exact_hits += 1;
        }
        if let Some((q, w)) = prev {
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            if (v - w).abs() > d + 1e-9 {
                lip_bad += 1;
            }
        }
        prev = Some((p, v));
    }
    println!(
        "{n} random points in {:.1} s: sign disagrees with the B-rep at {sign_bad}; |f| above the exact distance at {bound_bad} (worst {worst_excess:.1e} mm); Lipschitz violations {lip_bad}; |f| equals the exact distance at {:.1}%",
        t1.elapsed().as_secs_f64(),
        100.0 * exact_hits as f64 / n as f64
    );
    // Per-cell pruning on a grid of boxes spanning the part's height.
    let cell = ((hi.x - lo.x).max(hi.y - lo.y) / 40.0).max(1.0);
    let (mut boxes, mut leaves, mut mism) = (0usize, 0usize, 0usize);
    let t2 = std::time::Instant::now();
    let mut x = lo.x - pad;
    while x < hi.x + pad {
        let mut y = lo.y - pad;
        while y < hi.y + pad {
            let (blo, bhi) = ([x, y, z0 - pad], [x + cell, y + cell, z1 + pad]);
            let pf = f.prune(blo, bhi);
            boxes += 1;
            leaves += pf.leaf_count();
            for _ in 0..8 {
                let p = [rng.range(blo[0], bhi[0]), rng.range(blo[1], bhi[1]), rng.range(blo[2], bhi[2])];
                if (f.eval(p).value - pf.eval(p).value).abs() > 1e-12 {
                    mism += 1;
                }
            }
            y += cell;
        }
        x += cell;
    }
    println!(
        "pruning over {boxes} columns of {cell:.2} mm: {:.2} leaves per cell on average (global tree {}), pruned value differs from global at {mism} of {} samples, in {:.1} s",
        leaves as f64 / boxes as f64,
        f.leaf_count(),
        boxes * 8,
        t2.elapsed().as_secs_f64()
    );
}

/// Exact signed distance of general B-rep solids (every face on its exact
/// surface). Checks: the signed distance is 1-Lipschitz on random pairs (a wrong
/// sign or a foot accepted outside its face breaks it), points outside the box
/// are positive, and on prismatic parts it equals the prismatic exact distance
/// with the prismatic point-in-solid sign.
fn solid_check(path: &Path, n: usize) {
    use icam_core::curve3::V3;
    use icam_core::solid3::{Feature, Solid3};
    use rayon::prelude::*;
    let brep = match step::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read failed: {e}");
            std::process::exit(2);
        }
    };
    let tol = brep.tolerance_mm.max(1e-6);
    let t0 = std::time::Instant::now();
    let mut solids = Vec::new();
    for s in &brep.solids {
        match Solid3::from_step(s, tol) {
            Ok((sol, rep)) => {
                println!(
                    "solid #{} {:?}: faces {:?}, edges {:?}; vertices within {:.1e} mm of their curves, edges within {:.1e} mm of their surfaces",
                    s.id, s.name, rep.faces, rep.edges, rep.max_vertex_off, rep.max_edge_off
                );
                solids.push(sol);
            }
            Err(e) => println!("solid #{} {:?}: refused: {e}", s.id, s.name),
        }
    }
    if solids.is_empty() {
        eprintln!("no solid imported");
        std::process::exit(2);
    }
    println!("{} of {} solids imported in {:.0} ms (file tolerance {:.1e} mm)", solids.len(), brep.solids.len(), t0.elapsed().as_secs_f64() * 1e3, brep.tolerance_mm);
    let (mut lo, mut hi) = (solids[0].lo, solids[0].hi);
    for s in &solids {
        lo = V3::new(lo.x.min(s.lo.x), lo.y.min(s.lo.y), lo.z.min(s.lo.z));
        hi = V3::new(hi.x.max(s.hi.x), hi.y.max(s.hi.y), hi.z.max(s.hi.z));
    }
    let diag = (hi - lo).len();
    let sdf = |p: V3| {
        let mut best = solids[0].sdf(p);
        for s in &solids[1..] {
            let d = s.sdf(p);
            if d.d < best.d {
                best = d;
            }
        }
        best
    };
    // Points: half uniform over the padded box, half near a random edge.
    let mut rng = Rng(0x5EED_0F_3D_50_11D5);
    let pad = 0.05 * diag;
    let mut pts = Vec::with_capacity(n);
    for k in 0..n {
        let p = if k % 2 == 0 {
            V3::new(rng.range(lo.x - pad, hi.x + pad), rng.range(lo.y - pad, hi.y + pad), rng.range(lo.z - pad, hi.z + pad))
        } else {
            let s = &solids[(rng.next() * solids.len() as f64) as usize % solids.len()];
            let e = &s.edges[(rng.next() * s.edges.len() as f64) as usize % s.edges.len()];
            let on = e.curve.at(rng.next());
            let dir = V3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).unit();
            on + dir * (rng.next().powi(2) * 0.05 * diag)
        };
        // A partner point within a little more than the distance scale.
        let dir = V3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).unit();
        pts.push((p, dir, rng.next()));
    }
    let t1 = std::time::Instant::now();
    let res: Vec<_> = pts
        .par_iter()
        .map(|&(p, dir, u)| {
            let tp = std::time::Instant::now();
            let a = sdf(p);
            let ta = tp.elapsed().as_secs_f64();
            if ta > 2.0 {
                eprintln!("slow distance {:.1} s at {p:?}", ta);
            }
            let q = p + dir * (u * (2.0 * a.d.abs() + 1e-3 * diag));
            let b = sdf(q);
            (p, a, q, b, ta)
        })
        .collect();
    let secs = t1.elapsed().as_secs_f64();
    if let Some((p, a, _, _, ta)) = res.iter().max_by(|x, y| x.4.total_cmp(&y.4)) {
        println!("slowest distance {:.1} ms at {p:?} (d {:.4}, {:?})", ta * 1e3, a.d, a.feature);
    }
    let res: Vec<_> = res.into_iter().map(|(p, a, q, b, _)| (p, a, q, b)).collect();
    let (mut lip_bad, mut unsure, mut inside, mut out_neg) = (0, 0, 0, 0);
    let (mut on_face, mut on_edge, mut on_vert) = (0, 0, 0);
    let mut worst: f64 = 0.0;
    let mut example = None;
    for (p, a, q, b) in &res {
        let d = (*p - *q).len();
        let excess = (a.d - b.d).abs() - d;
        if excess > 1e-6 {
            lip_bad += 1;
            if excess > worst {
                worst = excess;
                example = Some((*p, *a, *q, *b));
            }
        }
        unsure += (a.unsure || b.unsure) as usize;
        inside += (a.d < 0.0) as usize;
        let outside_box = p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z;
        out_neg += (outside_box && a.d < 0.0) as usize;
        match a.feature {
            Feature::Face(_) => on_face += 1,
            Feature::Edge(_) => on_edge += 1,
            Feature::Vertex(_) => on_vert += 1,
        }
    }
    println!(
        "{n} point pairs in {secs:.1} s ({:.0} distances/s on {} threads): nearest feature face {on_face} / edge {on_edge} / vertex {on_vert}; {inside} inside",
        2.0 * n as f64 / secs,
        rayon::current_num_threads()
    );
    println!("Lipschitz violations {lip_bad} (worst excess {worst:.2e} mm); negative outside the box {out_neg}; trim tests with no usable path {unsure}");
    if let Some((p, a, q, b)) = example {
        println!("  worst pair: {p:?} d {:.6} ({:?}) and {q:?} d {:.6} ({:?}), {:.6} apart", a.d, a.feature, b.d, b.feature, (p - q).len());
    }
    // Prismatic cross-check.
    // Trims: from points along every edge, step a little into and out of each of
    // its faces along the in-face direction, drop onto the surface, and classify.
    // A step shorter than a tenth of every edge of the face stays on the face.
    let t_trim = std::time::Instant::now();
    let (mut tin, mut tin_bad, mut tout, mut tout_bad, mut tnone) = (0, 0, 0, 0, 0);
    let mut trim_examples = Vec::new();
    for sol in &solids {
        for (ei, e) in sol.edges.iter().enumerate() {
            for (k, &fi) in e.faces.iter().enumerate() {
                let f = &sol.faces[fi];
                if f.seams.contains(&ei) {
                    continue;
                }
                let rev = f.loops.iter().flatten().filter(|u| u.edge == ei).nth(if e.faces[0] == e.faces[1] { k } else { 0 }).map_or(false, |u| u.rev);
                let shortest = f
                    .loops
                    .iter()
                    .flatten()
                    .map(|u| (sol.edges[u.edge].curve.start() - sol.edges[u.edge].curve.end()).len().max((sol.edges[u.edge].curve.at(0.5) - sol.edges[u.edge].curve.start()).len()))
                    .fold(f64::INFINITY, f64::min);
                // A face narrower than a few model tolerances cannot be stepped into reliably.
                let step = (0.1 * shortest).min(0.01 * diag);
                if step < 5.0 * tol {
                    continue;
                }
                for s in [0.23, 0.5, 0.77] {
                    let b = e.curve.at(s);
                    let t = e.curve.tangent_at(s) * if rev { -1.0 } else { 1.0 };
                    let m = f.normal(b).cross(t).unit();
                    for (dir, want) in [(1.0, true), (-1.0, false)] {
                        let q = f.surf.foot(b + m * (step * dir));
                        // Stepping out of an edge on the surface's own boundary lands back on it.
                        if !want && ((q - b).len() < 0.5 * step || e.curve.closest(q).dist < 10.0 * tol) {
                            continue;
                        }
                        match sol.classify(fi, q) {
                            None => tnone += 1,
                            Some(got) => {
                                if want {
                                    tin += 1;
                                    tin_bad += (got != want) as usize;
                                } else {
                                    tout += 1;
                                    tout_bad += (got != want) as usize;
                                }
                                if got != want && trim_examples.len() < 3 {
                                    trim_examples.push((f.id, f.surf.name(), e.id, q, want));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "trim test ({:.1} s): stepped into faces {tin} times, classified outside {tin_bad}; stepped out {tout} times, classified inside {tout_bad}; no usable path {tnone}",
        t_trim.elapsed().as_secs_f64()
    );
    for (fid, name, eid, q, want) in trim_examples {
        println!("  face #{fid} ({name}) next to edge #{eid}: {q:?} should be {}", if want { "inside" } else { "outside" });
    }
    // Only where the F-rep compiler accepts the part: its brute-force distance
    // assumes full walls under every curve, which overhangs break.
    let prism = prismatic::from_brep(&brep).ok().filter(|(part, _)| solids.len() == brep.solids.len() && icam_core::frep::compile_prismatic(part, tol).is_ok());
    if let Some((part, _)) = prism {
        use icam_core::prismatic_reference::{exact_boundary_distance, inside_solid};
        let m = n.min(5000);
        let (mut dist_bad, mut sign_bad) = (0, 0);
        let mut worst_d: f64 = 0.0;
        for (p, a, _, _) in &res[..m] {
            let pp = [p.x, p.y, p.z];
            let ex = exact_boundary_distance(&part, pp);
            let e = (a.d.abs() - ex).abs();
            if e > 1e-6 {
                dist_bad += 1;
                if dist_bad <= 5 && std::env::var("ICAM_DIAG").is_ok() {
                    println!("  differs at {pp:?}: solid {:.6} ({:?}, foot {:?}) vs prismatic {:.6} inside {}", a.d, a.feature, a.q, ex, inside_solid(&part, pp));
                }
            }
            worst_d = worst_d.max(e);
            if ex > 1e-6 && (a.d < 0.0) != inside_solid(&part, pp) {
                sign_bad += 1;
            }
        }
        println!("prismatic cross-check on {m} points: |distance| differs at {dist_bad} (worst {worst_d:.2e} mm), sign differs at {sign_bad}");
    }
}

/// Faces by surface type and B-spline span counts, for every STEP file in a directory.
fn census(dir: &Path) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
    files.retain(|f| f.extension().map_or(false, |e| e.eq_ignore_ascii_case("step") || e.eq_ignore_ascii_case("stp")));
    files.sort();
    for f in files {
        let Ok(brep) = step::read(&f) else { continue };
        let tol = brep.tolerance_mm.max(1e-6);
        let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
        let (mut solids, mut spans) = (0, 0usize);
        for s in &brep.solids {
            if let Ok((sol, _)) = icam_core::solid3::Solid3::from_step(s, tol) {
                solids += 1;
                for face in &sol.faces {
                    *kinds.entry(face.surf.name()).or_default() += 1;
                    if let icam_core::surface3::Surf::Nurbs(n) = &face.surf {
                        spans += n.span_count();
                    }
                }
            }
        }
        println!("{} | {} solids | {:?} | {} B-spline spans", f.file_name().unwrap().to_string_lossy(), solids, kinds, spans);
    }
}
