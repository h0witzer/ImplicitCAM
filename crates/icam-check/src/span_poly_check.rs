//! icam-check spanpoly: the signed implicit polynomial (icam_core::span_poly) of
//! every span of a STEP file or folder, measured against independent references.
//!
//! Per span: the degree profile (least singular value at each degree), the degree
//! read at a range of floors, and for spans with a polynomial at the default
//! floor: how exactly F vanishes on the patch (F / |grad F| over a grid of patch
//! points), whether F's orientation holds over the patch, whether every zero of F
//! met along lines through the padded box is a point where the exact matrix
//! representation (icam_core::mrep, an independent construction) also drops rank
//! (if not, F has an extraneous zero), and how often f32 evaluation in the part's
//! frame disagrees with f64 in sign, and how far from the zero set that happens.

use std::path::Path;

use icam_core::curve3::V3;
use icam_core::mrep::MRep;
use icam_core::solid3::Solid3;
use icam_core::span_poly::{monomials, DegreeStep, SpanPoly};
use icam_core::step;
use icam_core::surface3::{Patch, Surf};
use rayon::prelude::*;

use crate::spans::{own_sheet, patch_dist, pct};

const FLOORS: [f64; 7] = [1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8];

/// Eigenvalues and eigenvectors (columns) of a symmetric 3 x 3 matrix by Jacobi.
fn sym3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let mut off = 0.0;
        for p in 0..3 {
            for q in p + 1..3 {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-40 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..3 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let th = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for r in 0..3 {
                    let (arp, arq) = (a[r][p], a[r][q]);
                    a[r][p] = c * arp - sn * arq;
                    a[r][q] = sn * arp + c * arq;
                }
                for r in 0..3 {
                    let (apr, aqr) = (a[p][r], a[q][r]);
                    a[p][r] = c * apr - sn * aqr;
                    a[q][r] = sn * apr + c * aqr;
                }
                for r in 0..3 {
                    let (vrp, vrq) = (v[r][p], v[r][q]);
                    v[r][p] = c * vrp - sn * vrq;
                    v[r][q] = sn * vrp + c * vrq;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// The kind of a degree-2 span polynomial, for the report only (it reads ranks
/// at 1e-8 of the largest eigenvalue; nothing in the field depends on it).
fn quadric_kind(sp: &SpanPoly) -> &'static str {
    let c = |e: [u8; 3]| sp.mons.iter().position(|m| *m == e).map_or(0.0, |i| sp.coef[i]);
    let a = [[c([2, 0, 0]), 0.5 * c([1, 1, 0]), 0.5 * c([1, 0, 1])], [0.5 * c([1, 1, 0]), c([0, 2, 0]), 0.5 * c([0, 1, 1])], [0.5 * c([1, 0, 1]), 0.5 * c([0, 1, 1]), c([0, 0, 2])]];
    let b = [c([1, 0, 0]), c([0, 1, 0]), c([0, 0, 1])];
    let k0 = c([0, 0, 0]);
    let (lam, v) = sym3(a);
    let lmax = lam.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let tol = 1e-8 * lmax.max(b.iter().map(|x| x.abs()).fold(0.0, f64::max));
    // In the eigenframe: sum lam_i y_i^2 + beta_i y_i + k0.
    let beta: Vec<f64> = (0..3).map(|i| (0..3).map(|r| v[r][i] * b[r]).sum()).collect();
    let nz: Vec<usize> = (0..3).filter(|&i| lam[i].abs() > tol).collect();
    let zero: Vec<usize> = (0..3).filter(|&i| lam[i].abs() <= tol).collect();
    // Complete the squares in the nonzero directions.
    let k1 = k0 - nz.iter().map(|&i| beta[i] * beta[i] / (4.0 * lam[i])).sum::<f64>();
    let linear_left = zero.iter().any(|&i| beta[i].abs() > tol);
    let pos = nz.iter().filter(|&&i| lam[i] > 0.0).count();
    let neg = nz.len() - pos;
    let k1z = k1.abs() <= tol;
    match (nz.len(), linear_left) {
        (3, _) => {
            let same = pos == 3 || neg == 3;
            if k1z {
                if same { "point" } else { "cone" }
            } else if same {
                let (lo, hi) = (nz.iter().map(|&i| lam[i].abs()).fold(f64::INFINITY, f64::min), nz.iter().map(|&i| lam[i].abs()).fold(0.0, f64::max));
                if (hi - lo) <= 1e-8 * hi { "sphere" } else { "ellipsoid" }
            } else {
                // One sheet when the constant has the sign of the majority.
                let majority_pos = pos == 2;
                if (k1 < 0.0) == majority_pos { "hyperboloid of one sheet" } else { "hyperboloid of two sheets" }
            }
        }
        (2, true) => {
            if pos == 2 || neg == 2 { "elliptic paraboloid" } else { "hyperbolic paraboloid" }
        }
        (2, false) => {
            if k1z {
                "two planes"
            } else if pos == 2 || neg == 2 {
                let (l0, l1) = (lam[nz[0]].abs(), lam[nz[1]].abs());
                if (l0 - l1).abs() <= 1e-8 * l0.max(l1) { "circular cylinder" } else { "elliptic cylinder" }
            } else {
                "hyperbolic cylinder"
            }
        }
        (1, true) => "parabolic cylinder",
        (1, false) => "parallel planes",
        _ => "plane",
    }
}

struct Stat {
    deg: (usize, usize),
    rational: bool,
    profile: Vec<DegreeStep>,
    /// The polynomial at the default floor, if any.
    k: Option<usize>,
    sigma: f64,
    below: f64,
    /// max |F / grad F| over patch points, in span sizes.
    on_patch: f64,
    /// min over patch points of cos(grad F, Su x Sv).
    orient: f64,
    /// Zeros of F met along lines: on the patch, on the span's own extension,
    /// elsewhere on the exact surface (the M-rep drops rank there), and not on it.
    z_patch: usize,
    z_own: usize,
    z_surface: usize,
    z_extraneous: usize,
    /// Largest M-rep ratio at a zero of F away from the patch.
    worst_ratio: f64,
    /// min over patch points of cos(Su x Sv there, Su x Sv at the center): negative
    /// where the span's own normal turns over (a fold or a pole).
    nflip: f64,
    /// Why no polynomial, when there is none.
    reason: String,
    /// The quadric's kind when the degree is 2.
    kind: &'static str,
    /// f32 sign disagreements of the 3,000 box points, and the largest
    /// |F / grad F| (span sizes) where one happened.
    f32_bad: usize,
    f32_band: f64,
    mrep_err: Option<String>,
}

fn eval_f32(sp: &SpanPoly, mons: &[[u8; 3]], origin: V3, p: V3) -> f32 {
    // As the GPU sees it: the point and the span's center in f32 relative to the
    // part's origin, the scale and coefficients in f32.
    let (px, py, pz) = ((p.x - origin.x) as f32, (p.y - origin.y) as f32, (p.z - origin.z) as f32);
    let (cx, cy, cz) = ((sp.center.x - origin.x) as f32, (sp.center.y - origin.y) as f32, (sp.center.z - origin.z) as f32);
    let inv = (1.0 / sp.scale) as f32;
    let q = [(px - cx) * inv, (py - cy) * inv, (pz - cz) * inv];
    let k = sp.degree;
    let mut pw = [[1.0f32; 9]; 3];
    for a in 0..3 {
        for i in 1..=k {
            pw[a][i] = pw[a][i - 1] * q[a];
        }
    }
    let mut s = 0.0f32;
    for (m, c) in mons.iter().zip(&sp.coef) {
        s += (*c as f32) * pw[0][m[0] as usize] * pw[1][m[1] as usize] * pw[2][m[2] as usize];
    }
    s
}

fn dump(patch: &Patch, st: &Stat) {
    let pw: Vec<String> = patch.pw.iter().flatten().map(|h| format!("({:.6},{:.6},{:.6} w{:.6})", h[0] / h[3], h[1] / h[3], h[2] / h[3], h[3])).collect();
    let prof: Vec<String> = st.profile.iter().map(|p| format!("k{}:{:.1e}/{:.1e}", p.k, p.sigma, p.sigma_next)).collect();
    eprintln!("span ({},{}) k {:?} orient {:.3} nflip {:.3} extraneous {} worst {:.1e} reason '{}'
  net {}
  profile {}", st.deg.0, st.deg.1, st.k, st.orient, st.nflip, st.z_extraneous, st.worst_ratio, st.reason, pw.join(" "), prof.join(" "));
}

fn stat(patch: &Patch, origin: V3) -> Stat {
    let st = stat_inner(patch, origin);
    if std::env::var("ICAM_SPANPOLY_DUMP").is_ok() && (st.orient < 0.0 || st.z_extraneous > 0 || st.k.is_none()) {
        dump(patch, &st);
    }
    st
}

fn stat_inner(patch: &Patch, origin: V3) -> Stat {
    let deg = (patch.pw.len() - 1, patch.pw[0].len() - 1);
    let profile = SpanPoly::profile(patch, SpanPoly::KMAX);
    let w0 = patch.pw[0][0][3];
    let rational = patch.pw.iter().flatten().any(|h| (h[3] - w0).abs() > 1e-12 * w0.abs());
    let mut st = Stat { deg, rational, profile, k: None, sigma: f64::NAN, below: f64::NAN, on_patch: f64::NAN, orient: f64::NAN, z_patch: 0, z_own: 0, z_surface: 0, z_extraneous: 0, worst_ratio: 0.0, nflip: f64::NAN, reason: String::new(), kind: "", f32_bad: 0, f32_band: 0.0, mrep_err: None };
    let sp = match SpanPoly::of_patch(patch) {
        Ok(sp) => sp,
        Err(e) => {
            let _ = e;
            st.reason = "no unique polynomial of degree <= 8 vanishes on the span and its extension".into();
            return st;
        }
    };
    st.k = Some(sp.degree);
    if sp.degree == 2 {
        st.kind = quadric_kind(&sp);
    } else if sp.degree == 1 {
        st.kind = "plane";
    }
    st.sigma = sp.sigma;
    st.below = sp.sigma_below;
    let scale = sp.scale;
    // Orientation over the patch where both the span's normal and F's gradient
    // are well defined: a pole's normal and a singular point's gradient vanish and
    // point anywhere.
    let mut grid = Vec::new();
    let (mut nmax, mut gmax) = (0.0f64, 0.0f64);
    for i in 0..=24 {
        for j in 0..=24 {
            let e = patch.eval(i as f64 / 24.0, j as f64 / 24.0);
            let (n, g) = (e.ps.cross(e.pt), sp.grad(e.p));
            nmax = nmax.max(n.len());
            gmax = gmax.max(g.len());
            grid.push((e.p, n, g));
        }
    }
    let ec = patch.eval(0.5, 0.5);
    let nc = ec.ps.cross(ec.pt);
    let (mut on, mut orient, mut nflip) = (0.0f64, f64::INFINITY, f64::INFINITY);
    for &(p, n, g) in &grid {
        if n.len() > 1e-6 * nmax && g.len() > 1e-6 * gmax {
            orient = orient.min(g.dot(n) / (g.len() * n.len()));
            on = on.max(sp.first_order(p).abs() / scale);
            if nc.len() > 0.0 {
                nflip = nflip.min(n.dot(nc) / (n.len() * nc.len()));
            }
        }
    }
    st.on_patch = on;
    st.orient = orient;
    st.nflip = nflip;
    // Zeros of F along lines through the padded box, each checked against the
    // exact matrix representation.
    let rep = match MRep::with_tol(patch, 1e-10) {
        Ok(r) => Some(r),
        Err(e) => {
            st.mrep_err = Some(e);
            None
        }
    };
    // Padded by a tenth of the diagonal on every side: a planar span's box is flat.
    let (lo, hi) = patch.bbox();
    let pad = (hi - lo).len() * 0.1;
    let pad = V3::new(pad, pad, pad);
    let (lo, hi) = (lo - pad, hi + pad);
    let g = 5;
    let n = 160;
    for axis in 0..3 {
        for i in 0..g {
            for j in 0..g {
                let (fi, fj) = ((i as f64 + 0.5) / g as f64, (j as f64 + 0.5) / g as f64);
                let at = |u: f64| -> V3 {
                    match axis {
                        0 => V3::new(lo.x + (hi.x - lo.x) * u, lo.y + (hi.y - lo.y) * fi, lo.z + (hi.z - lo.z) * fj),
                        1 => V3::new(lo.x + (hi.x - lo.x) * fi, lo.y + (hi.y - lo.y) * u, lo.z + (hi.z - lo.z) * fj),
                        _ => V3::new(lo.x + (hi.x - lo.x) * fi, lo.y + (hi.y - lo.y) * fj, lo.z + (hi.z - lo.z) * u),
                    }
                };
                let mut prev = sp.value(at(0.0));
                for kk in 1..=n {
                    let u = kk as f64 / n as f64;
                    let cur = sp.value(at(u));
                    if prev != 0.0 && cur != 0.0 && (prev < 0.0) != (cur < 0.0) {
                        let (mut a, mut b, mut fa) = ((kk - 1) as f64 / n as f64, u, prev);
                        for _ in 0..40 {
                            let c = 0.5 * (a + b);
                            let fc = sp.value(at(c));
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
                        if patch_dist(patch, x) < 1e-6 * scale {
                            st.z_patch += 1;
                        } else if own_sheet(patch, x, scale, 3.0) {
                            st.z_own += 1;
                        } else if let Some(rep) = &rep {
                            let s = rep.svd(x).0;
                            let ratio = s[s.len() - 1] / s[0];
                            st.worst_ratio = st.worst_ratio.max(ratio);
                            if ratio < 1e-7 {
                                st.z_surface += 1;
                            } else {
                                st.z_extraneous += 1;
                            }
                        }
                    }
                    prev = cur;
                }
            }
        }
    }
    // f32 against f64 at random points of the padded box.
    let mons = monomials(sp.degree);
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut u = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..3000 {
        let x = V3::new(lo.x + (hi.x - lo.x) * u(), lo.y + (hi.y - lo.y) * u(), lo.z + (hi.z - lo.z) * u());
        let f64v = sp.value(x);
        let f32v = eval_f32(&sp, &mons, origin, x);
        if (f64v > 0.0) != (f32v > 0.0) {
            st.f32_bad += 1;
            st.f32_band = st.f32_band.max(sp.first_order(x).abs() / scale);
        }
    }
    st
}

struct Row {
    name: String,
    note: String,
    spans: usize,
    stats: Vec<Stat>,
    secs: f64,
}

fn file_stats(path: &Path, per_file: usize) -> Row {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let t0 = std::time::Instant::now();
    let mut row = Row { name, note: String::new(), spans: 0, stats: Vec::new(), secs: 0.0 };
    let brep = match step::read(path) {
        Ok(b) => b,
        Err(e) => {
            row.note = format!("read: {e}");
            return row;
        }
    };
    let tol = brep.tolerance_mm.max(1e-6);
    let mut jobs: Vec<Patch> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let (mut lo, mut hi) = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
    for s in &brep.solids {
        let Ok((sol, _)) = Solid3::from_step(s, tol) else { continue };
        lo = V3::new(lo.x.min(sol.lo.x), lo.y.min(sol.lo.y), lo.z.min(sol.lo.z));
        hi = V3::new(hi.x.max(sol.hi.x), hi.y.max(sol.hi.y), hi.z.max(sol.hi.z));
        for f in &sol.faces {
            let Surf::Nurbs(ns) = &f.surf else { continue };
            if !seen.insert(f.id) {
                continue;
            }
            let (nu, nv) = ns.spans_uv();
            for a in 0..nu {
                for b in 0..nv {
                    jobs.push(ns.patch(a, b));
                }
            }
        }
    }
    row.spans = jobs.len();
    if jobs.is_empty() {
        return row;
    }
    if per_file > 0 && jobs.len() > per_file {
        let step = jobs.len() as f64 / per_file as f64;
        jobs = (0..per_file).map(|i| jobs[(i as f64 * step) as usize].clone()).collect();
    }
    let origin = (lo + hi) * 0.5;
    row.stats = jobs.par_iter().map(|p| stat(p, origin)).collect();
    if let Ok(csv) = std::env::var("ICAM_SPANPOLY_CSV") {
        // One line per span: the file, the span, what the floor rule chose and
        // checked, and the whole degree profile.
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&csv).expect("csv");
        for st in &row.stats {
            let prof: Vec<String> = st.profile.iter().map(|d| format!("{}:{:.3e}:{:.3e}:{:.3e}:{:.3e}", d.k, d.sigma, d.sigma_next, d.on_span, d.ext)).collect();
            writeln!(f, "{},{},{},{},{},{},{},{},{:.3e},{:.3e},{:.3e},{},{}", row.name, st.deg.0, st.deg.1, st.rational as u8, st.k.map_or(0, |k| k), st.z_patch, st.z_surface, st.z_extraneous, st.worst_ratio, st.orient, st.nflip, st.kind, prof.join(" ")).unwrap();
        }
    }
    row.secs = t0.elapsed().as_secs_f64();
    row
}

/// `icam-check spanpoly <file.step | dir> [n]`: n spans per file (all when 0).
pub fn run(path: &Path, per_file: usize) {
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
    println!("signed span polynomials, degree <= {}, vanishing {} spans out to {:.0e}, unique by {:.0e}; {} file(s), up to {} spans each", SpanPoly::KMAX, SpanPoly::REACH, SpanPoly::EXT_TOL, SpanPoly::UNIQUE, files.len(), per_file);
    println!("{:<24} {:>6} {:>5} {:>6} {:>18} {:>9} {:>9} {:>8} {:>13} {:>8} {:>9} {:>6}", "file", "spans", "meas", "exact", "degrees found", "on patch", "orient", "jump min", "zeros p/o/s/x", "worst r", "f32 bad", "secs");
    let mut all: Vec<Stat> = Vec::new();
    for f in &files {
        let row = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| file_stats(f, per_file))) {
            Ok(r) => r,
            Err(_) => Row { name: f.file_name().unwrap().to_string_lossy().to_string(), note: "PANIC".into(), spans: 0, stats: Vec::new(), secs: 0.0 },
        };
        if row.spans == 0 && row.note.is_empty() {
            all.extend(row.stats);
            continue;
        }
        let ex: Vec<&Stat> = row.stats.iter().filter(|s| s.k.is_some()).collect();
        let mut by_k: std::collections::BTreeMap<usize, usize> = Default::default();
        for s in &ex {
            *by_k.entry(s.k.unwrap()).or_insert(0) += 1;
        }
        let ks: Vec<String> = by_k.iter().map(|(k, n)| format!("{k}:{n}")).collect();
        let on = ex.iter().map(|s| s.on_patch).fold(0.0f64, f64::max);
        let orient = ex.iter().map(|s| s.orient).fold(f64::INFINITY, f64::min);
        let jump = ex.iter().map(|s| s.below / s.sigma.max(1e-300)).fold(f64::INFINITY, f64::min);
        let z: [usize; 4] = [ex.iter().map(|s| s.z_patch).sum(), ex.iter().map(|s| s.z_own).sum(), ex.iter().map(|s| s.z_surface).sum(), ex.iter().map(|s| s.z_extraneous).sum()];
        let worst = ex.iter().map(|s| s.worst_ratio).fold(0.0f64, f64::max);
        let bad: usize = ex.iter().map(|s| s.f32_bad).sum();
        println!(
            "{:<24} {:>6} {:>5} {:>6} {:>18} {:>9.1e} {:>9.3} {:>8.1e} {:>13} {:>8.1e} {:>4}/{:<4} {:>6.1}{}",
            row.name.chars().take(24).collect::<String>(),
            row.spans,
            row.stats.len(),
            ex.len(),
            ks.join(" "),
            on,
            orient,
            jump,
            format!("{}/{}/{}/{}", z[0], z[1], z[2], z[3]),
            worst,
            bad,
            ex.len() * 3000,
            row.secs,
            if row.note.is_empty() { String::new() } else { format!("  [{}]", row.note) }
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
        all.extend(row.stats);
    }
    if all.is_empty() {
        return;
    }
    println!();
    println!("{} spans measured", all.len());
    // How many spans are exact at each floor, by bidegree.
    let mut degs: Vec<(usize, usize)> = all.iter().map(|s| s.deg).collect();
    degs.sort();
    degs.dedup();
    print!("{:<10} {:>6}", "bidegree", "spans");
    for f in FLOORS {
        print!(" {:>7.0e}", f);
    }
    println!("   (spans with a polynomial of degree <= {} at each floor)", SpanPoly::KMAX);
    for d in &degs {
        let sel: Vec<&Stat> = all.iter().filter(|s| s.deg == *d).collect();
        print!("({:>2},{:>2})    {:>6}", d.0, d.1, sel.len());
        for f in FLOORS {
            print!(" {:>7}", sel.iter().filter(|s| s.profile.iter().any(|p| p.sigma <= f)).count());
        }
        println!();
    }
    // The profile around the floor: for the spans exact at the default floor, the
    // least singular value at their degree and one below; for the others, the
    // least singular value at the highest degree tried.
    let ex: Vec<&Stat> = all.iter().filter(|s| s.k.is_some()).collect();
    let mut at_k: Vec<f64> = ex.iter().map(|s| s.sigma).collect();
    let mut below: Vec<f64> = ex.iter().map(|s| s.below).collect();
    let mut rest: Vec<f64> = all.iter().filter(|s| s.k.is_none()).map(|s| s.profile.iter().map(|p| p.sigma).fold(f64::INFINITY, f64::min)).collect();
    println!("exact: {} spans; least singular value at their degree p50 {:.1e} p99 {:.1e} max {:.1e}; one degree below min {:.1e} p1 {:.1e} p50 {:.1e}", ex.len(), pct(&mut at_k, 0.5), pct(&mut at_k, 0.99), pct(&mut at_k, 1.0), pct(&mut below, 0.0), pct(&mut below, 0.01), pct(&mut below, 0.5));
    println!("not exact: {} spans; least singular value over every degree tried min {:.1e} p1 {:.1e} p10 {:.1e} p50 {:.1e}", rest.len(), pct(&mut rest, 0.0), pct(&mut rest, 0.01), pct(&mut rest, 0.1), pct(&mut rest, 0.5));
    let mut reasons: std::collections::BTreeMap<(String, usize, usize), usize> = Default::default();
    for s in all.iter().filter(|s| s.k.is_none()) {
        *reasons.entry((s.reason.clone(), s.deg.0, s.deg.1)).or_insert(0) += 1;
    }
    for ((r, a, b), n) in &reasons {
        println!("  not exact, ({a},{b}): {n} spans, {r}");
    }
    let mut on: Vec<f64> = ex.iter().map(|s| s.on_patch).collect();
    let mut orient: Vec<f64> = ex.iter().map(|s| s.orient).collect();
    println!("F / |grad F| on the patch, span sizes: p50 {:.1e} max {:.1e}; orientation cos min {:.3}, spans where it turns negative {} (of them, spans whose own normal turns over: {})", pct(&mut on, 0.5), pct(&mut on, 1.0), pct(&mut orient, 0.0), ex.iter().filter(|s| s.orient < 0.0).count(), ex.iter().filter(|s| s.orient < 0.0 && s.nflip < 0.0).count());
    for s in ex.iter().filter(|s| s.orient < 0.0 && !(s.nflip < 0.0)).take(6) {
        println!("  orientation turns with the normal steady: bidegree ({},{}) degree {} orient {:.3} normal {:.3}", s.deg.0, s.deg.1, s.k.unwrap(), s.orient, s.nflip);
    }
    let z: [usize; 4] = [ex.iter().map(|s| s.z_patch).sum(), ex.iter().map(|s| s.z_own).sum(), ex.iter().map(|s| s.z_surface).sum(), ex.iter().map(|s| s.z_extraneous).sum()];
    println!("zeros of F along lines: {} on the patch, {} on the span's own extension, {} elsewhere on the exact surface, {} not on it (extraneous); largest M-rep ratio at a zero off the patch {:.1e}; M-rep refused on {} spans", z[0], z[1], z[2], z[3], ex.iter().map(|s| s.worst_ratio).fold(0.0f64, f64::max), ex.iter().filter(|s| s.mrep_err.is_some()).count());
    let mut kinds: std::collections::BTreeMap<(usize, &str), usize> = Default::default();
    for s in &ex {
        *kinds.entry((s.k.unwrap(), s.kind)).or_insert(0) += 1;
    }
    let ks: Vec<String> = kinds.iter().map(|((k, kind), n)| if kind.is_empty() { format!("degree {k}: {n}") } else { format!("{kind}: {n}") }).collect();
    println!("exact spans by kind: {}", ks.join(", "));
    let bad: usize = ex.iter().map(|s| s.f32_bad).sum();
    println!("f32 in the part's frame: {} of {} box points disagree with f64 in sign, all within {:.1e} span sizes of the zero set", bad, ex.len() * 3000, ex.iter().map(|s| s.f32_band).fold(0.0f64, f64::max));
}


/// `icam-check spanpolygpu <file.step> [points]`: the exact spans of a file,
/// grouped by degree, evaluated at random points of the part's box on the GPU
/// (f32, straight-line kernel) and on the CPU (f64, all threads); rates, and the
/// sign agreement of the two with how far from the zero set any disagreement is.
pub fn gpu_bench(path: &Path, npts: usize) {
    let brep = step::read(path).unwrap();
    let tol = brep.tolerance_mm.max(1e-6);
    let (mut lo, mut hi) = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
    let mut patches = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for s in &brep.solids {
        let Ok((sol, _)) = Solid3::from_step(s, tol) else { continue };
        lo = V3::new(lo.x.min(sol.lo.x), lo.y.min(sol.lo.y), lo.z.min(sol.lo.z));
        hi = V3::new(hi.x.max(sol.hi.x), hi.y.max(sol.hi.y), hi.z.max(sol.hi.z));
        for f in &sol.faces {
            let Surf::Nurbs(ns) = &f.surf else { continue };
            if !seen.insert(f.id) {
                continue;
            }
            let (nu, nv) = ns.spans_uv();
            for a in 0..nu {
                for b in 0..nv {
                    patches.push(ns.patch(a, b));
                }
            }
        }
    }
    let polys: Vec<SpanPoly> = patches.par_iter().filter_map(|p| SpanPoly::of_patch(p).ok()).collect();
    let origin = (lo + hi) * 0.5;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut u = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let pad = (hi - lo) * 0.05;
    let (blo, bhi) = (lo - pad, hi + pad);
    let pts: Vec<V3> = (0..npts).map(|_| V3::new(blo.x + (bhi.x - blo.x) * u(), blo.y + (bhi.y - blo.y) * u(), blo.z + (bhi.z - blo.z) * u())).collect();
    let gpu = match icam_gpu::Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            println!("no GPU: {e}");
            return;
        }
    };
    println!("{}: {} spans, {} with a polynomial; {} points in the part's box; GPU {}", path.file_name().unwrap().to_string_lossy(), patches.len(), polys.len(), npts, gpu.adapter);
    let mut kern = icam_gpu::span_poly::SpanPolyGpu::new();
    let mut by_k: std::collections::BTreeMap<usize, Vec<&SpanPoly>> = Default::default();
    for p in &polys {
        by_k.entry(p.degree).or_default().push(p);
    }
    // Warm the card: it idles at low clocks and times about twice slow cold.
    if let Some((_, g)) = by_k.iter().next() {
        let t = std::time::Instant::now();
        while t.elapsed().as_secs_f64() < 2.0 {
            let _ = kern.eval(&gpu, &g[..g.len().min(8)], origin, &pts, 1);
        }
    }
    println!("{:>6} {:>6} {:>12} {:>14} {:>14} {:>12} {:>16}", "degree", "spans", "terms", "GPU f32 eval/s", "CPU f64 eval/s", "sign differ", "within (sizes)");
    for (k, group) in &by_k {
        let chunk = ((64usize << 20) / (4 * npts)).max(1);
        let (mut evals, mut gpu_s, mut cpu_s, mut differ, mut band) = (0usize, 0.0, 0.0, 0usize, 0.0f64);
        for g in group.chunks(chunk) {
            let (vals, t) = match kern.eval(&gpu, g, origin, &pts, 5) {
                Ok(x) => x,
                Err(e) => {
                    println!("degree {k}: {e}");
                    break;
                }
            };
            gpu_s += t.kernel_s;
            evals += g.len() * npts;
            let tc = std::time::Instant::now();
            let reference: Vec<Vec<f64>> = g.iter().map(|sp| pts.par_iter().map(|&x| sp.value(x)).collect()).collect();
            cpu_s += tc.elapsed().as_secs_f64();
            for (j, sp) in g.iter().enumerate() {
                for (i, &x) in pts.iter().enumerate() {
                    let (a, b) = (reference[j][i], vals[j * npts + i]);
                    if (a > 0.0) != (b > 0.0) {
                        differ += 1;
                        band = band.max(sp.first_order(x).abs() / sp.scale);
                    }
                }
            }
        }
        println!("{:>6} {:>6} {:>12} {:>14.3e} {:>14.3e} {:>12} {:>16.1e}", k, group.len(), monomials(*k).len(), evals as f64 / gpu_s, evals as f64 / cpu_s, differ, band);
    }
}


/// `icam-check spanshape <file.step | dir>`: for every B-spline span, whether it
/// is exactly a plane or quadric (a polynomial of degree <= 2 that vanishes on it
/// and along its extension, and is unique), and the plane, sphere, circular
/// cylinder or circular cone nearest it with the span's largest distance from it,
/// relative to the span's size and in the file's units against the tolerance the
/// file declares.
pub fn shape(path: &Path) {
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
    // Buckets of the span's distance from the nearest analytic surface, relative to its size.
    const EDGES: [f64; 6] = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4];
    println!("nearest analytic surface of each B-spline span; distance buckets relative to the span size: <=1e-14 <=1e-12 <=1e-10 <=1e-8 <=1e-6 <=1e-4 above");
    let mut total: std::collections::BTreeMap<&'static str, [usize; 7]> = Default::default();
    let mut exact_total: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut other_total: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for f in &files {
        let Ok(brep) = step::read_raw(f) else { continue };
        let tol = brep.tolerance_mm.max(1e-6);
        let mut patches = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for s in &brep.solids {
            let Ok((sol, _)) = Solid3::from_step(s, tol) else { continue };
            for fc in &sol.faces {
                let Surf::Nurbs(ns) = &fc.surf else { continue };
                if !seen.insert(fc.id) {
                    continue;
                }
                let (nu, nv) = ns.spans_uv();
                for a in 0..nu {
                    for b in 0..nv {
                        patches.push(ns.patch(a, b));
                    }
                }
            }
        }
        if patches.is_empty() {
            continue;
        }
        let res: Vec<(Option<usize>, Option<(&'static str, f64, f64)>, &'static str)> = patches
            .par_iter()
            .map(|p| {
                let exact = SpanPoly::search(p, 2).ok();
                let k = exact.as_ref().map(|s| s.degree);
                let sp = exact.unwrap_or_else(|| SpanPoly::quadric_of(p));
                let a = sp.analytic(p).map(|a| (a.surf.name(), a.dev_rel, a.dev));
                let kind = if k == Some(2) { quadric_kind(&sp) } else if k == Some(1) { "plane" } else { "" };
                (k, a, kind)
            })
            .collect();
        let mut by: std::collections::BTreeMap<&'static str, [usize; 7]> = Default::default();
        let mut exact: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut within_tol = 0;
        let mut worst_exact = 0.0f64;
        let mut other_kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for (k, a, kind) in &res {
            if k.is_some() && a.map_or(true, |(_, rel, _)| rel > 1e-10) {
                *other_kinds.entry(kind).or_insert(0) += 1;
            }
            let Some((name, rel, abs)) = a else {
                by.entry("none").or_insert([0; 7])[6] += 1;
                continue;
            };
            let bucket = EDGES.iter().position(|&e| *rel <= e).unwrap_or(6);
            by.entry(name).or_insert([0; 7])[bucket] += 1;
            if *abs <= tol {
                within_tol += 1;
            }
            if k.is_some() {
                *exact.entry(name).or_insert(0) += 1;
                worst_exact = worst_exact.max(*rel);
            }
        }
        if let Ok(path) = std::env::var("ICAM_SHAPE_CSV") {
            use std::io::Write;
            let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("csv");
            for (k, a, kind) in &res {
                if let Some((name, rel, abs)) = a {
                    writeln!(out, "{},{},{},{:.3e},{:.3e},{}", f.file_name().unwrap().to_string_lossy(), name, k.map_or(0, |k| k), rel, abs, kind).unwrap();
                }
            }
        }
        let rows: Vec<String> = by.iter().map(|(n, b)| format!("{n} {:?}", b)).collect();
        let ex: Vec<String> = exact.iter().map(|(n, c)| format!("{n} {c}")).collect();
        println!("{:<24} spans {:>6}  file tolerance {:.0e} mm, spans within it {:>6}  exact plane/quadric: {} (worst distance {:.1e})", f.file_name().unwrap().to_string_lossy(), res.len(), tol, within_tol, if ex.is_empty() { "none".into() } else { ex.join(", ") }, worst_exact);
        println!("    nearest: {}", rows.join("; "));
        if !other_kinds.is_empty() {
            let ok: Vec<String> = other_kinds.iter().map(|(n, c)| format!("{n} {c}")).collect();
            println!("    exact quadrics that are not a plane, sphere, circular cylinder or cone: {}", ok.join(", "));
            for (n, c) in &other_kinds {
                *other_total.entry(n).or_insert(0) += c;
            }
        }
        for (n, b) in by {
            let t = total.entry(n).or_insert([0; 7]);
            for i in 0..7 {
                t[i] += b[i];
            }
        }
        for (n, c) in exact {
            *exact_total.entry(n).or_insert(0) += c;
        }
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    for (n, b) in &total {
        println!("all files, nearest {n:<10} {:?}", b);
    }
    println!("all files, exact plane/quadric spans by nearest simple surface: {:?}", exact_total);
    println!("all files, exact quadrics of other kinds: {:?}", other_total);
}


/// `icam-check analytic <file.step | dir> [points] [seconds]`: each file read as
/// B-spline surfaces (`step::read_raw`) and with the B-spline surfaces that are
/// planes, spheres, cylinders or cones read as those (`step::read`): the faces that
/// changed, by kind, and the exact signed distance of both readings at the same
/// random points of each solid's padded box (largest difference, sign
/// disagreements) with the time each took. A time budget bounds the slow reading.
pub fn analytic(path: &Path, npts: usize, budget: f64) {
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
    println!("{:<26} {:>6} {:>30} {:>8} {:>12} {:>7} {:>10} {:>10}  {}", "file", "faces", "B-spline faces now analytic", "points", "max |dd|", "sign", "raw pt/s", "now pt/s", "notes");
    let (mut all_conv, mut all_bs, mut all_sign, mut all_max) = (0usize, 0usize, 0usize, 0.0f64);
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let (raw, now) = match (step::read_raw(f), step::read(f)) {
            (Ok(a), Ok(b)) => (a, b),
            (a, b) => {
                println!("{name:<26} read failed: raw {:?} now {:?}", a.err(), b.err());
                continue;
            }
        };
        // Faces that changed, by kind.
        let mut kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let (mut nfaces, mut nbs) = (0usize, 0usize);
        for (sr, sn) in raw.solids.iter().zip(&now.solids) {
            for (fr, fnow) in sr.faces.iter().zip(&sn.faces) {
                nfaces += 1;
                if matches!(fr.surface, step::Surface::BSpline(_)) {
                    nbs += 1;
                    let k = match fnow.surface {
                        step::Surface::Plane { .. } => "plane",
                        step::Surface::Sphere { .. } => "sphere",
                        step::Surface::Cylinder { .. } => "cylinder",
                        step::Surface::Cone { .. } => "cone",
                        step::Surface::Torus { .. } => "torus",
                        _ => continue,
                    };
                    *kinds.entry(k).or_insert(0) += 1;
                }
            }
        }
        // Faces read as the other quadrics (they stay B-spline faces for trims), and
        // how their exact distance compares with the B-spline's own just off them.
        let tol0 = now.tolerance_mm.max(1e-6);
        let mut quad_kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut quad_worst = 0.0f64;
        for sn in &now.solids {
            let Ok((b, _)) = Solid3::from_step(sn, tol0) else { continue };
            for fc in &b.faces {
                let (Some(qd), Surf::Nurbs(ns)) = (&fc.quadric, &fc.surf) else { continue };
                *quad_kinds.entry(qd.name()).or_insert(0) += 1;
                let (nu, nv) = ns.spans_uv();
                for i in 0..nu.min(4) {
                    for j in 0..nv.min(4) {
                        let p = ns.patch(i, j);
                        let (lo, hi) = p.bbox();
                        let h = (hi - lo).len() * 1e-3;
                        for (s, t) in [(0.3, 0.4), (0.7, 0.6)] {
                            let e = p.eval(s, t);
                            let n = e.ps.cross(e.pt).unit();
                            for side in [-1.0, 1.0] {
                                let x = e.p + n * (side * h);
                                let dq = qd.signed_distance(x, 1.0).abs();
                                let dn = ns.closest(x, f64::INFINITY).map_or(f64::NAN, |hit| (hit.q - x).len());
                                if std::env::var("ICAM_QUAD_DEBUG").is_ok() && (dq - dn).abs() > 1e-6 {
                                    let crit: Vec<String> = qd.crit(x).iter().map(|c| format!("{:.4}", (*c - x).len())).collect();
                                    eprintln!("  face #{} {} span ({i},{j}) h {h:.2e}: quadric {dq:.6} B-spline {dn:.6}; G on the surface point {:.2e}, |grad| {:.2e}; quadric {:?}; critical distances {}", fc.id, qd.name(), qd.value(e.p), qd.grad(e.p).len(), qd, crit.join(" "));
                                }
                                quad_worst = quad_worst.max((dq - dn).abs());
                            }
                        }
                    }
                }
            }
        }
        if std::env::var("ICAM_SPAN_QUADRICS").is_ok() {
            // Span by span, on the B-spline faces no reading took whole.
            let mut per: std::collections::BTreeMap<&'static str, usize> = Default::default();
            let (mut faces_with, mut total) = (0usize, 0usize);
            let mut span_worst = 0.0f64;
            let (mut faces_all, mut spans_all) = (0usize, 0usize);
            let mut sizes: Vec<f64> = Vec::new();
            for sn in &now.solids {
                let Ok((b, _)) = Solid3::from_step(sn, tol0) else { continue };
                for fc in &b.faces {
                    let Surf::Nurbs(ns) = &fc.surf else { continue };
                    if fc.quadric.is_some() {
                        continue;
                    }
                    let (nu, nv) = ns.spans_uv();
                    let mut any = false;
                    let mut every = true;
                    for i in 0..nu {
                        for j in 0..nv {
                            total += 1;
                            let pt = ns.patch(i, j);
                            if let Some(q) = icam_core::span_poly::quadric_of_span(&pt) {
                                *per.entry(q.name()).or_insert(0) += 1;
                                any = true;
                                let (lo, hi) = pt.bbox();
                                let size = (hi - lo).len();
                                sizes.push(size);
                                for (s, t) in [(0.3, 0.4), (0.7, 0.6), (0.5, 0.5)] {
                                    let e = pt.eval(s, t);
                                    let n = e.ps.cross(e.pt).unit();
                                    for side in [-1.0, 1.0] {
                                        let x = e.p + n * (side * size * 1e-3);
                                        let dq = q.signed_distance(x, 1.0).abs();
                                        let dn = ns.closest(x, f64::INFINITY).map_or(f64::NAN, |hit| (hit.q - x).len());
                                        span_worst = span_worst.max((dq - dn).abs());
                                    }
                                }
                            } else {
                                every = false;
                            }
                        }
                    }
                    faces_with += any as usize;
                    if every {
                        faces_all += 1;
                        spans_all += nu * nv;
                    }
                }
            }
            println!("{name:<26} spans of the remaining B-spline faces that are a quadric on their own: {:?} of {total} spans, in {faces_with} faces; their size (box diagonal) p10 {:.2e} p50 {:.2e} p90 {:.2e} mm; quadric against B-spline distance 1e-3 of a span off: largest difference {span_worst:.1e} mm; faces quadric in every span: {faces_all} ({spans_all} spans)", per, pct(&mut sizes.clone(), 0.1), pct(&mut sizes.clone(), 0.5), pct(&mut sizes, 0.9));
        }
        if !quad_kinds.is_empty() {
            let qk: Vec<String> = quad_kinds.iter().map(|(k, n)| format!("{k} {n}")).collect();
            println!("{name:<26} faces read as other quadrics: {}; their exact distance against the B-spline's, 1e-3 of a span off it: largest difference {quad_worst:.1e} mm", qk.join(", "));
        }
        let conv: usize = kinds.values().sum();
        all_conv += conv;
        all_bs += nbs;
        if conv == 0 {
            println!("{name:<26} {nfaces:>6} {:>30}", format!("0 of {nbs}"));
            continue;
        }
        let tol = now.tolerance_mm.max(1e-6);
        let mut seed = 0x51_7CC1_B727_220Au64;
        let mut u = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut n_pts, mut max_dd, mut sign, mut t_raw, mut t_now) = (0usize, 0.0f64, 0usize, 0.0f64, 0.0f64);
        let mut notes = Vec::new();
        for (sr, sn) in raw.solids.iter().zip(&now.solids) {
            let (a, b) = match (Solid3::from_step(sr, tol), Solid3::from_step(sn, tol)) {
                (Ok((a, _)), Ok((b, _))) => (a, b),
                (Err(e), _) => {
                    notes.push(format!("raw solid #{}: {e}", sr.id));
                    continue;
                }
                (_, Err(e)) => {
                    notes.push(format!("analytic solid #{}: {e}", sn.id));
                    continue;
                }
            };
            // Every converted face keeps its outward side: the raw face's outward
            // normal (Su x Sv times its sense) and the analytic face's at a point of it.
            for (fa, fb) in a.faces.iter().zip(&b.faces) {
                let Surf::Nurbs(ns) = &fa.surf else { continue };
                if matches!(fb.surf, Surf::Nurbs(_)) {
                    continue;
                }
                let (nu, nv) = ns.spans_uv();
                let e = ns.patch(nu / 2, nv / 2).eval(0.5, 0.5);
                let (na, nb) = (fa.surf.normal(e.p) * fa.sense, fb.surf.normal(e.p) * fb.sense);
                if na.dot(nb) < 0.0 {
                    notes.push(format!("face #{} ({}) turned inside out", fb.id, fb.surf.name()));
                }
                let raw_n = e.ps.cross(e.pt).unit() * fa.sense;
                if raw_n.dot(na) < 0.9 {
                    notes.push(format!("face #{}: raw normal {:?} vs Su x Sv {:?}", fa.id, na, raw_n));
                }
            }
            let pad = (a.hi - a.lo) * 0.05;
            let (lo, hi) = (a.lo - pad, a.hi + pad);
            let per = (npts / raw.solids.len().max(1)).max(50);
            let pts: Vec<V3> = (0..per).map(|_| V3::new(lo.x + (hi.x - lo.x) * u(), lo.y + (hi.y - lo.y) * u(), lo.z + (hi.z - lo.z) * u())).collect();
            // Interleaved in chunks so both readings see the same load.
            let size = (hi - lo).len();
            for chunk in pts.chunks(64) {
                if t_raw + t_now > budget {
                    notes.push("time budget".into());
                    break;
                }
                let t0 = std::time::Instant::now();
                let da: Vec<f64> = chunk.par_iter().map(|&p| a.sdf(p).d).collect();
                t_raw += t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let db: Vec<f64> = chunk.par_iter().map(|&p| b.sdf(p).d).collect();
                t_now += t1.elapsed().as_secs_f64();
                for (i, (x, y)) in da.iter().zip(&db).enumerate() {
                    n_pts += 1;
                    if std::env::var("ICAM_ANALYTIC_DEBUG").is_ok() && (x - y).abs() > 1e-6 * size {
                        let (sa, sb) = (a.sdf(chunk[i]), b.sdf(chunk[i]));
                        let feat = |s: &Solid3, f: &icam_core::solid3::Feature| match f {
                            icam_core::solid3::Feature::Face(k) => format!("face #{} {}", s.faces[*k].id, s.faces[*k].surf.name()),
                            icam_core::solid3::Feature::Edge(k) => format!("edge #{}", s.edges[*k].id),
                            icam_core::solid3::Feature::Vertex(k) => format!("vertex {k}"),
                        };
                        let p = chunk[i];
                        let outside_box = p.x < a.lo.x || p.y < a.lo.y || p.z < a.lo.z || p.x > a.hi.x || p.y > a.hi.y || p.z > a.hi.z;
                        eprintln!("  {:?}: raw {:.6} at {} ({:?}), analytic {:.6} at {} ({:?}); solid box {:?}..{:?}, point outside it: {outside_box}", p, sa.d, feat(&a, &sa.feature), sa.q, sb.d, feat(&b, &sb.feature), sb.q, a.lo, a.hi);
                    }
                    max_dd = max_dd.max((x - y).abs() / size);
                    if (*x < 0.0) != (*y < 0.0) && x.abs().min(y.abs()) > tol {
                        sign += 1;
                    }
                }
            }
        }
        all_sign += sign;
        all_max = all_max.max(max_dd);
        let ks: Vec<String> = kinds.iter().map(|(k, n)| format!("{k} {n}")).collect();
        println!(
            "{:<26} {:>6} {:>30} {:>8} {:>12.1e} {:>7} {:>10.0} {:>10.0}  {}",
            name,
            nfaces,
            format!("{conv} of {nbs}: {}", ks.join(", ")),
            n_pts,
            max_dd,
            sign,
            n_pts as f64 / t_raw.max(1e-9),
            n_pts as f64 / t_now.max(1e-9),
            notes.join("; ")
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    println!("{all_conv} of {all_bs} B-spline faces read as analytic surfaces; largest distance difference {all_max:.1e} of the part size; {all_sign} sign disagreements beyond the file tolerance");
}
