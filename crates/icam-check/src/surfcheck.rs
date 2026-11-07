//! icam-check surfcheck: fields checked against their solid's exact surfaces,
//! without the B-rep distance (`Solid3::sdf`).
//!
//! Derek, 2026-10-26, on checks that compare against the slow B-rep reference: "We
//! gotta find a way to evaluate these things that doesn't rely on the legacy brep
//! path", and "I've always hated the over reliance on regression tests when
//! sometimes the code we had before was bad". The B-rep reference walks trims and
//! searches B-spline patches per query, 4 to 12 points a second on B-spline-heavy
//! files, and it is not ground truth (it read two points outside moon_buggy's own
//! box as inside).
//!
//! Reference data is made once per face from the face's exact surface: points on
//! it (a span grid for a B-spline face, the feet of random points for an analytic
//! one), kept when the face's own trim test calls them inside and they lie well
//! clear of the face's edges (where trim tests fail), each with its outward normal
//! and an offset delta under a quarter of its distance to the edges, a quarter of
//! its least radius of curvature and 1e-4 of the solid's box diagonal. A field is
//! then checked by evaluating it alone: zero at the points; plus delta and minus
//! delta at the points moved delta out and in along the normal (the exact distance
//! there, the point's own foot being nearest); and, with no reference at all, the
//! 1-Lipschitz bound on random pairs and a positive value outside the solid's box.
//!
//! The use here: the importer's analytic readings (each face read as a plane,
//! sphere, cylinder, cone, torus or quadric, checked against the B-spline face it
//! was read from), on the CPU over all threads.

use std::path::Path;

use icam_core::curve3::V3;
use icam_core::solid3::{Face3, Solid3};
use icam_core::step;
use icam_core::surface3::Surf;
use rayon::prelude::*;

/// A point of a face with its outward unit normal and the offset it is checked at.
#[derive(Clone, Copy)]
pub(crate) struct Sample {
    pub(crate) p: V3,
    pub(crate) n: V3,
    pub(crate) delta: f64,
    pub(crate) face: usize,
}

pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The least radius of curvature of a B-spline surface at (u, v), from its
/// fundamental forms; infinite where it is flat.
/// Trim-test outcomes while sampling: inside, outside, unsure.
pub(crate) static TRIM: [std::sync::atomic::AtomicUsize; 3] = [std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0)];
fn tally(r: Option<bool>) -> Option<bool> {
    let i = match r { Some(true) => 0, Some(false) => 1, None => 2 };
    TRIM[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    r
}

pub(crate) fn nurbs_radius(e: &icam_core::surface3::SEval) -> f64 {
    let nn = e.ps.cross(e.pt);
    let l = nn.len();
    if !(l > 0.0) {
        return 0.0;
    }
    let n = nn * (1.0 / l);
    let (ee, ff, gg) = (e.ps.dot(e.ps), e.ps.dot(e.pt), e.pt.dot(e.pt));
    let (ll, mm, nn2) = (e.pss.dot(n), e.pst.dot(n), e.ptt.dot(n));
    let a = ee * gg - ff * ff;
    let b = ee * nn2 - 2.0 * ff * mm + gg * ll;
    let c = ll * nn2 - mm * mm;
    // Roots of a k^2 - b k + c = 0.
    let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
    let kmax = ((b.abs() + disc) / (2.0 * a)).abs();
    if kmax > 0.0 {
        1.0 / kmax
    } else {
        f64::INFINITY
    }
}

pub(crate) fn analytic_radius(s: &Surf, x: V3) -> f64 {
    match s {
        Surf::Plane { .. } => f64::INFINITY,
        Surf::Sphere { r, .. } | Surf::Cylinder { r, .. } => *r,
        Surf::Torus { major, minor, .. } => minor.min((major.abs() - minor).abs().max(1e-300)),
        Surf::Cone { f, a, .. } => {
            let d = x - f.o;
            let h = d.dot(f.z);
            ((d - f.z * h).len() / a.cos()).max(1e-300)
        }
        Surf::Nurbs(_) => f64::INFINITY,
    }
}

/// Up to `want` samples of face fi of a solid.
pub(crate) fn face_samples(sol: &Solid3, fi: usize, want: usize, rng: &mut Rng) -> Vec<Sample> {
    let f = &sol.faces[fi];
    let edges: Vec<usize> = f.loops.iter().flatten().map(|u| u.edge).collect();
    let diag = (sol.hi - sol.lo).len();
    let edge_dist = |p: V3| edges.iter().map(|&e| sol.edges[e].curve.closest(p).dist).fold(f64::INFINITY, f64::min);
    // The face's box, from its edges.
    let (mut lo, mut hi) = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
    for &e in &edges {
        let r = &sol.edges[e];
        lo = V3::new(lo.x.min(r.lo.x), lo.y.min(r.lo.y), lo.z.min(r.lo.z));
        hi = V3::new(hi.x.max(r.hi.x), hi.y.max(r.hi.y), hi.z.max(r.hi.z));
    }
    let size = (hi - lo).len().max(1e-9);
    let mut out = Vec::new();
    let mut tries = 0;
    while out.len() < want && tries < 12 * want {
        tries += 1;
        let (p, n, rad) = match &f.surf {
            Surf::Nurbs(ns) => {
                let (u, v) = (ns.u_range.0 + (ns.u_range.1 - ns.u_range.0) * rng.next(), ns.v_range.0 + (ns.v_range.1 - ns.v_range.0) * rng.next());
                if tally(sol.classify_nurbs(fi, u, v)) != Some(true) {
                    continue;
                }
                let e = ns.eval(u, v);
                let nn = e.ps.cross(e.pt);
                if !(nn.len() > 0.0) {
                    continue;
                }
                (e.p, nn.unit() * f.sense, nurbs_radius(&e))
            }
            s => {
                let pad = (hi - lo) * 0.05;
                let x = V3::new(lo.x - pad.x + (hi.x - lo.x + 2.0 * pad.x) * rng.next(), lo.y - pad.y + (hi.y - lo.y + 2.0 * pad.y) * rng.next(), lo.z - pad.z + (hi.z - lo.z + 2.0 * pad.z) * rng.next());
                let q = s.foot(x);
                if tally(sol.classify(fi, q)) != Some(true) {
                    continue;
                }
                (q, s.normal(q) * f.sense, analytic_radius(s, q))
            }
        };
        let de = edge_dist(p);
        // Well clear of the face's edges: a tenth of the face's size.
        if !(de > 0.1 * size) {
            continue;
        }
        let delta = (0.25 * de).min(0.25 * rad).min(1e-4 * diag);
        if !(delta > 0.0) {
            continue;
        }
        out.push(Sample { p, n, delta, face: fi });
    }
    out
}

/// What a field did at a set of samples.
#[derive(Default)]
struct Report {
    n: usize,
    zero_worst: f64,
    off_worst: f64,
    off_exact: usize,
    sign: usize,
    evals: usize,
    secs: f64,
}

fn check_samples(samples: &[Sample], field: &(dyn Fn(usize, V3) -> f64 + Sync)) -> Report {
    let t = std::time::Instant::now();
    let rows: Vec<(f64, f64, f64, bool)> = samples
        .par_iter()
        .map(|s| {
            let z = field(s.face, s.p);
            let (a, b) = (field(s.face, s.p + s.n * s.delta), field(s.face, s.p - s.n * s.delta));
            let off = (a - s.delta).abs().max((b + s.delta).abs());
            (z.abs(), off, s.delta, !(a > 0.0 && b < 0.0))
        })
        .collect();
    let mut r = Report { n: rows.len(), evals: 3 * rows.len(), secs: t.elapsed().as_secs_f64(), ..Default::default() };
    for (z, off, _d, bad) in rows {
        r.zero_worst = r.zero_worst.max(z);
        r.off_worst = r.off_worst.max(off);
        if off <= 1e-9 {
            r.off_exact += 1;
        }
        if bad {
            r.sign += 1;
        }
    }
    r
}

/// The signed distance to an analytic face's whole surface (its sense applied), in
/// closed form; None for a B-spline face or a cone whose half-angle is not in
/// (0, 90 degrees).
fn analytic(fc: &Face3, x: V3) -> Option<f64> {
    let s = fc.sense;
    // Height along the axis and distance from it.
    let radial = |o: V3, z: V3| {
        let d = x - o;
        let h = d.dot(z);
        (h, (d - z * h).len())
    };
    Some(match &fc.surf {
        Surf::Plane { f } => (x - f.o).dot(f.z * s),
        Surf::Cylinder { f, r } => s * (radial(f.o, f.z).1 - r),
        Surf::Sphere { f, r } => s * ((x - f.o).len() - r),
        Surf::Torus { f, major, minor } => {
            let (h, rho) = radial(f.o, f.z);
            s * ((rho - major).hypot(h) - minor)
        }
        Surf::Cone { f, r, a } => {
            if !(*a > 0.0 && *a < std::f64::consts::FRAC_PI_2) {
                return None;
            }
            // In the half-plane (rho, h): the generatrix ray from the apex along
            // (sin a, cos a); behind the apex the apex is nearest (outside).
            let (h, rho) = radial(f.o, f.z);
            let (sa, ca) = a.sin_cos();
            let ha = -r / a.tan();
            if rho * sa + (h - ha) * ca <= 0.0 {
                s * (x - (f.o + f.z * ha)).len()
            } else {
                s * (rho * ca - (h - ha) * sa)
            }
        }
        Surf::Nurbs(_) => return None,
    })
}

/// `icam-check surfcheck <file.step | dir> [samples per face]`.
pub fn run(path: &Path, per_face: usize) {
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
    println!("fields against samples of the exact face surfaces, up to {per_face} per face; no B-rep distance is evaluated");
    let trim_line = |name: &str| {
        let t: Vec<usize> = TRIM.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed)).collect();
        println!("  trim tests while sampling {name}: inside {}, outside {}, unsure {}", t[0], t[1], t[2]);
    };
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let (raw, now) = match (step::read_raw(f), step::read(f)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => {
                println!("{name}: read failed");
                continue;
            }
        };
        let tol = now.tolerance_mm.max(1e-6);
        let mut by_kind: std::collections::BTreeMap<String, Report> = Default::default();
        let (mut sample_secs, mut nsamples) = (0.0, 0usize);
        for (sr, sn) in raw.solids.iter().zip(&now.solids) {
            let (Ok((a, _)), Ok((b, _))) = (Solid3::from_step(sr, tol), Solid3::from_step(sn, tol)) else { continue };
            // Samples of every face, from the raw reading (the faces as written).
            let t = std::time::Instant::now();
            let samples: Vec<Sample> = (0..a.faces.len())
                .into_par_iter()
                .flat_map_iter(|fi| {
                    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((fi as u64 + 1) * 0x2545_F491_4F6C_DD1D));
                    face_samples(&a, fi, per_face, &mut rng)
                })
                .collect();
            sample_secs += t.elapsed().as_secs_f64();
            nsamples += samples.len();
            // The importer's readings: each face read as an analytic surface or a
            // quadric, against the B-spline face it was read from.
            let mut groups: std::collections::BTreeMap<String, Vec<Sample>> = Default::default();
            for s in &samples {
                let (fa, fb) = (&a.faces[s.face], &b.faces[s.face]);
                if !matches!(fa.surf, Surf::Nurbs(_)) {
                    continue;
                }
                let kind = if !matches!(fb.surf, Surf::Nurbs(_)) {
                    fb.surf.name().to_string()
                } else if let Some(q) = &fb.quadric {
                    q.name().to_string()
                } else {
                    continue;
                };
                groups.entry(kind).or_default().push(*s);
            }
            for (kind, ss) in groups {
                let leaf = |fi: usize, x: V3| -> f64 {
                    let fc = &b.faces[fi];
                    match (analytic(fc, x), &fc.quadric) {
                        (Some(d), _) => d,
                        (None, Some(q)) => q.signed_distance(x, fc.sense),
                        _ => f64::NAN,
                    }
                };
                let r = check_samples(&ss, &leaf);
                let e = by_kind.entry(kind).or_default();
                e.n += r.n;
                e.zero_worst = e.zero_worst.max(r.zero_worst);
                e.off_worst = e.off_worst.max(r.off_worst);
                e.off_exact += r.off_exact;
                e.sign += r.sign;
                e.evals += r.evals;
                e.secs += r.secs;
            }
        }
        println!("{name}: {nsamples} samples in {sample_secs:.2} s");
        trim_line(&name);
        for (kind, r) in &by_kind {
            println!(
                "  read as {kind:<26} {:>6} samples: |F| on the surface at most {:.1e} mm; at +-delta off it within {:.1e} mm of +-delta ({:.1}% within 1e-9 mm); sign wrong at {}; {:.2e} evals/s",
                r.n,
                r.zero_worst,
                r.off_worst,
                100.0 * r.off_exact as f64 / r.n.max(1) as f64,
                r.sign,
                r.evals as f64 / r.secs.max(1e-12)
            );
        }
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}
