//! icam-check trim2: the certified parameter-domain trim loops (icam-core trim2)
//! against the importer's path-walking trim test, on every B-spline face of the raw
//! reading (no analytic recognition, so every B-spline face is tested).
//!
//! Per file: faces built and refused (with the reasons), the largest certified
//! deviation, the build time; then random surface parameters on each face, with the
//! old test's answer (inside, outside, unsure) against the new one, and the time
//! per test of each.

use std::path::Path;

use icam_core::curve3::V3;
use icam_core::solid3::Solid3;
use icam_core::step;
use icam_core::surface3::Surf;
use rayon::prelude::*;

/// Beside-the-edge points left unjudged because their foot is on the surface's
/// own border (the face's boundary there).
static ON_BORDER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Default)]
struct Tally {
    faces: usize,
    built: usize,
    refused: usize,
    bound: f64,
    build_secs: f64,
    vertex_gap: f64,
    faces_over_tol: usize,
    cells: usize,
    pieces: usize,
    agree: usize,
    disagree: usize,
    unsure: usize,
    old_secs: f64,
    new_secs: f64,
    /// Disagreements judged by the solid's exact distance at the surface point:
    /// the new test right, the old test right.
    new_right: usize,
    old_right: usize,
    /// Disagreements within two tolerances of the face's own edges: undecidable.
    band: usize,
    /// Test points where the tree's answer differs from the plain polygon count on
    /// the same loop: a fault in the machinery.
    machinery: usize,
    /// Points just inside and just outside each edge of a face, by the file's own
    /// loop direction (the face lies to the left of its loops seen from outside):
    /// tested, the new test wrong, the old test wrong.
    edge_tested: usize,
    edge_new_wrong: usize,
    edge_old_wrong: usize,
    /// Beside-the-edge points on the surface's own border, not judged.
    edge_on_border: usize,
    examples: Vec<String>,
}

fn add(a: &mut Tally, b: &Tally) {
    a.faces += b.faces;
    a.built += b.built;
    a.refused += b.refused;
    a.bound = a.bound.max(b.bound);
    a.build_secs += b.build_secs;
    a.vertex_gap = a.vertex_gap.max(b.vertex_gap);
    a.faces_over_tol += b.faces_over_tol;
    a.cells += b.cells;
    a.pieces += b.pieces;
    a.agree += b.agree;
    a.disagree += b.disagree;
    a.unsure += b.unsure;
    a.old_secs += b.old_secs;
    a.new_secs += b.new_secs;
    a.new_right += b.new_right;
    a.old_right += b.old_right;
    a.band += b.band;
    a.machinery += b.machinery;
    a.edge_tested += b.edge_tested;
    a.edge_new_wrong += b.edge_new_wrong;
    a.edge_old_wrong += b.edge_old_wrong;
    a.edge_on_border += b.edge_on_border;
}

fn line(name: &str, t: &Tally) {
    let n = (t.agree + t.disagree).max(1) as f64;
    let tests = (t.agree + t.disagree + t.unsure).max(1) as f64;
    println!(
        "{name:<26} B-spline faces {:>5}: loops built {:>5}, refused {:>3}, largest certified deviation {:.2e} mm, largest vertex gap {:.1e} mm ({} faces over tol), build {:.2} s, {} pieces, {} cells | tests {:>7}: agree {}, DISAGREE {} ({:.3}%; within two tolerances of an edge {}; beyond, judged by the solid's distance: new right {}, old right {}), old unsure {}, tree vs polygon count on the same loop: {} differ | beside the edges, by the file's loop direction: {} points, new wrong {}, old wrong {} ({} more on the surface's own border, not judged) | per test: old {:.1} us, new {:.2} us",
        t.faces,
        t.built,
        t.refused,
        t.bound,
        t.vertex_gap,
        t.faces_over_tol,
        t.build_secs,
        t.pieces,
        t.cells,
        t.agree + t.disagree + t.unsure,
        t.agree,
        t.disagree,
        100.0 * t.disagree as f64 / n,
        t.band,
        t.new_right,
        t.old_right,
        t.unsure,
        t.machinery,
        t.edge_tested,
        t.edge_new_wrong,
        t.edge_old_wrong,
        t.edge_on_border,
        1e6 * t.old_secs / tests,
        1e6 * t.new_secs / tests
    );
}

/// `icam-check trim2 <file.step | dir> [tests per face]`.
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
    let mut total = Tally::default();
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let Ok(brep) = step::read_raw(f) else {
            println!("{name}: read failed");
            continue;
        };
        let tol = brep.tolerance_mm.max(1e-6);
        let mut t = Tally::default();
        for s in &brep.solids {
            let Ok((sol, rep)) = Solid3::from_step(s, tol) else { continue };
            t.refused += rep.trim2_refused;
            t.bound = t.bound.max(rep.trim2_bound);
            t.build_secs += rep.trim2_secs;
            t.vertex_gap = t.vertex_gap.max(rep.vertex_gap);
            t.faces_over_tol += rep.faces_over_tol;
            for n in rep.trim2_notes.iter().take(40) {
                t.examples.push(n.clone());
            }
            for n in rep.gap_notes.iter().take(3) {
                t.examples.push(format!("      {n}"));
            }
            if rep.off_tol_faces > 0 {
                t.examples.push(format!("      edges off their surfaces beyond the file's tolerance on {} faces ({} points, up to {:.3e} mm)", rep.off_tol_faces, rep.off_tol_points, rep.off_tol_max));
            }
            let analytic_only = std::env::var("ICAM_TRIM2_ANALYTIC").is_ok();
            let faces: Vec<usize> = (0..sol.faces.len()).filter(|&fi| matches!(sol.faces[fi].surf, Surf::Nurbs(_)) != analytic_only).collect();
            t.faces += faces.len();
            let mut none_by_kind: std::collections::BTreeMap<String, usize> = Default::default();
            for &fi in &faces {
                if let Some(t2) = &sol.faces[fi].trim2 {
                    t.built += 1;
                    t.cells += t2.cell_count();
                    t.pieces += t2.pieces.len();
                } else {
                    let f = &sol.faces[fi];
                    let what = match &f.surf {
                        Surf::Torus { major, minor, .. } => format!("torus with major {major:.3} and minor {minor:.3}"),
                        other => other.name().to_string(),
                    };
                    *none_by_kind.entry(what).or_insert(0) += 1;
                }
            }
            if !none_by_kind.is_empty() {
                let mut v: Vec<_> = none_by_kind.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1));
                t.examples.push(format!("      faces without loops (refused ones included): {}", v.iter().take(8).map(|(k, n)| format!("{k}: {n}")).collect::<Vec<_>>().join(", ")));
            }
            let per: Vec<(usize, usize, usize, f64, f64, Vec<String>, (usize, usize, usize), usize, (usize, usize, usize))> = faces
                .par_iter()
                .map(|&fi| {
                    let ns: &icam_core::surface3::NurbsSurf = match (&sol.faces[fi].surf, &sol.faces[fi].analytic2) {
                        (Surf::Nurbs(ns), _) => ns,
                        (_, Some(a)) => &a.ns,
                        _ => return (0, 0, 0, 0.0, 0.0, Vec::new(), (0, 0, 0), 0, (0, 0, 0)),
                    };
                    if sol.faces[fi].trim2.is_none() {
                        return (0, 0, 0, 0.0, 0.0, Vec::new(), (0, 0, 0), 0, (0, 0, 0));
                    }
                    // Beside each edge: a step off its middle into and out of the face,
                    // along the outward normal crossed with the edge's direction in the
                    // loop, put back on the surface.
                    let face = &sol.faces[fi];
                    let mut edge_judge = (0usize, 0usize, 0usize);
                    let mut on_border = 0usize;
                    let diag = {
                        let (lo, hi) = face.loops.iter().flatten().fold((V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN)), |(lo, hi), u| {
                            let e = &sol.edges[u.edge];
                            (V3::new(lo.x.min(e.lo.x), lo.y.min(e.lo.y), lo.z.min(e.lo.z)), V3::new(hi.x.max(e.hi.x), hi.y.max(e.hi.y), hi.z.max(e.hi.z)))
                        });
                        (hi - lo).len()
                    };
                    for u in face.loops.iter().flatten() {
                        if face.seams.contains(&u.edge) {
                            continue;
                        }
                        let e = &sol.edges[u.edge];
                        let q = e.curve.at(0.5);
                        let mut tan = e.curve.tangent_at(0.5);
                        if u.rev {
                            tan = -tan;
                        }
                        let n = face.normal(q);
                        let d = n.cross(tan);
                        if !(d.len() > 1e-9) {
                            continue;
                        }
                        let d = d.unit();
                        let len: f64 = (0..8).map(|k| (e.curve.at((k + 1) as f64 / 8.0) - e.curve.at(k as f64 / 8.0)).len()).sum();
                        let step = (1e-3 * diag).min(0.05 * len).max(20.0 * sol.tol);
                        for (sign, want) in [(1.0, true), (-1.0, false)] {
                            let x = face.surf.foot(q + d * (sign * step));
                            // A step out past an edge along the surface's own border puts
                            // the foot on that border, which is the face's boundary: either
                            // answer is right there, and it is not judged.
                            if let Surf::Nurbs(ns) = &face.surf {
                                if let Some(h) = ns.closest(x, f64::INFINITY) {
                                    let near = |w: f64, r: (f64, f64)| (w - r.0).abs().min((w - r.1).abs()) <= 1e-9 * (r.1 - r.0);
                                    if near(h.u, ns.u_range) || near(h.v, ns.v_range) {
                                        on_border += 1;
                                        continue;
                                    }
                                }
                            }
                            let (Some(new), old) = (sol.classify2(fi, x), sol.classify(fi, x)) else { continue };
                            edge_judge.0 += 1;
                            if new != want {
                                edge_judge.1 += 1;
                                if std::env::var("ICAM_TRIM2_JUDGE").ok().and_then(|x| x.parse::<u32>().ok()) == Some(face.id) {
                                    if let Surf::Nurbs(ns) = &face.surf {
                                        if let Some(h) = ns.closest(x, f64::INFINITY) {
                                            println!("      JUDGE face #{} edge #{} want {want}: point {:?} uv ({}, {}) domain u {:?} v {:?}; {}", face.id, e.id, x, h.u, h.v, ns.u_range, ns.v_range, face.trim2.as_ref().unwrap().explain(h.u, h.v).split("; pieces").next().unwrap_or(""));
                                        }
                                    }
                                }
                            }
                            if old != Some(want) {
                                edge_judge.2 += 1;
                            }
                        }
                    }
                    ON_BORDER.fetch_add(on_border, std::sync::atomic::Ordering::Relaxed);
                    let mut machinery = 0;
                    // The third opinion on a disagreement beyond the file's own band
                    // about the edges: the solid's exact distance at the surface point
                    // is about zero when the point is on the face.
                    let mut judged = (0, 0, 0);
                    let face_edges: Vec<usize> = sol.faces[fi].loops.iter().flatten().map(|u| u.edge).collect();
                    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((fi as u64 + 1) * 0x2545_F491_4F6C_DD1D));
                    let (mut agree, mut disagree, mut unsure) = (0, 0, 0);
                    let (mut old_s, mut new_s) = (0.0, 0.0);
                    let mut ex = Vec::new();
                    for _ in 0..per_face {
                        let (u, v) = (ns.u_range.0 + (ns.u_range.1 - ns.u_range.0) * rng.next(), ns.v_range.0 + (ns.v_range.1 - ns.v_range.0) * rng.next());
                        let t0 = std::time::Instant::now();
                        let old = if analytic_only { sol.classify(fi, ns.eval(u, v).p) } else { sol.classify_nurbs(fi, u, v) };
                        if analytic_only {
                            // The closed-form inverse returns the parameters.
                            let (uu, vv) = sol.faces[fi].analytic2.as_ref().unwrap().uv(ns.eval(u, v).p);
                            if !((uu - u).abs() <= 1e-9 * (1.0 + u.abs()) && (vv - v).abs() <= 1e-9 * (1.0 + v.abs())) && ex.len() < 4 {
                                ex.push(format!("      INVERSE: face #{} ({u}, {v}) came back as ({uu}, {vv})", sol.faces[fi].id));
                            }
                        }
                        old_s += t0.elapsed().as_secs_f64();
                        let t1 = std::time::Instant::now();
                        let new = sol.classify_nurbs2(fi, u, v).unwrap();
                        new_s += t1.elapsed().as_secs_f64();
                        let t2 = sol.faces[fi].trim2.as_ref().unwrap();
                        if t2.inside_slow(u, v) != new {
                            machinery += 1;
                            if ex.len() < 4 {
                                ex.push(format!("      TREE DIFFERS FROM THE POLYGON COUNT: face #{} (u, v) = ({u:.6}, {v:.6}): {}", sol.faces[fi].id, t2.explain(u, v)));
                            }
                        }
                        match old {
                            None => unsure += 1,
                            Some(b) if b == new => agree += 1,
                            Some(b) => {
                                disagree += 1;
                                let p = ns.eval(u, v).p;
                                let to_edge = face_edges.iter().map(|&e| sol.edges[e].curve.closest(p).dist).fold(f64::INFINITY, f64::min);
                                if to_edge <= 2.0 * sol.tol {
                                    judged.2 += 1;
                                } else if judged.0 + judged.1 < 8 {
                                    let d = sol.sdf(p).d;
                                    let truth = d.abs() <= 2.0 * sol.tol;
                                    if truth == new {
                                        judged.0 += 1;
                                    } else {
                                        judged.1 += 1;
                                        if ex.len() < 4 {
                                            ex.push(format!("      beyond the band ({to_edge:.2e} mm from an edge), the solid's distance {d:.2e} says the old test is right; {}", sol.faces[fi].trim2.as_ref().unwrap().explain(u, v)));
                                        }
                                    }
                                }
                                if ex.len() < 2 {
                                    ex.push(format!("face #{} sense {}: (u, v) = ({u:.6}, {v:.6}): old {b}, new {new}", sol.faces[fi].id, sol.faces[fi].sense));
                                }
                                if std::env::var("ICAM_TRIM2_FACE").ok().and_then(|x| x.parse::<u32>().ok()) == Some(sol.faces[fi].id) && ex.len() < 8 {
                                    ex.push(format!("      {}", sol.faces[fi].trim2.as_ref().unwrap().explain(u, v)));
                                }
                            }
                        }
                    }
                    if std::env::var("ICAM_TRIM2_PERFACE").is_ok() && (disagree > 0 || edge_judge.1 > 0 || edge_judge.2 > 0) {
                        ex.push(format!("      PERFACE face #{} ({}, sense {}): disagree {disagree} of {}; beside edges {} points, new wrong {}, old wrong {}", sol.faces[fi].id, sol.faces[fi].surf.name(), sol.faces[fi].sense, agree + disagree + unsure, edge_judge.0, edge_judge.1, edge_judge.2));
                    }
                    (agree, disagree, unsure, old_s, new_s, ex, judged, machinery, edge_judge)
                })
                .collect();
            for (a, d, u, os, nsec, ex, judged, machinery, ej) in per {
                t.machinery += machinery;
                t.edge_tested += ej.0;
                t.edge_new_wrong += ej.1;
                t.edge_old_wrong += ej.2;
                t.new_right += judged.0;
                t.old_right += judged.1;
                t.band += judged.2;
                t.agree += a;
                t.disagree += d;
                t.unsure += u;
                t.old_secs += os;
                t.new_secs += nsec;
                if t.examples.len() < 6 || std::env::var("ICAM_TRIM2_FACE").is_ok() || std::env::var("ICAM_TRIM2_PERFACE").is_ok() {
                    t.examples.extend(ex);
                }
            }
        }
        t.edge_on_border = ON_BORDER.swap(0, std::sync::atomic::Ordering::Relaxed);
        line(&name, &t);
        for x in &t.examples {
            println!("    {x}");
        }
        add(&mut total, &t);
    }
    if files.len() > 1 {
        line("all files", &total);
    }
}
