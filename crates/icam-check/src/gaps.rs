//! icam-check gaps: where a field compiled from local rules has no rule yet, counted
//! from each solid's topology and the face normals along its edges.
//!
//! A small enough ball about a point of a valid solid holds one face (its surface
//! is the formula), one edge (max where convex, min where concave, a separator
//! where tangent) or one vertex (a cone of faces). The compiler gives a formula to
//! a vertex whose edges are all convex or all concave; a vertex mixing convex and
//! concave edges, a vertex at a tangent edge and an edge whose kind changes along
//! it go to probes. Features closer together than a cell can be refined cannot be
//! put in cells of their own. Each is counted here with the compiler's own test for
//! an edge's kind (`edge_frame`: tangent when the normals agree to 1e-6, else convex
//! when the second face turns away from the first's side), on the importer's
//! reading (B-spline faces read as the analytic surfaces they are). No distance is
//! evaluated and nothing is compiled.
//!
//! Near features are counted at two scales: the file's declared tolerance (closer
//! than that, the file itself says two features are one) and the reach of a cell at
//! the compiler's refinement floor, ten times that tolerance and at least 1e-4 mm.
//! Lengths and gaps are measured at nine points along each edge.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use icam_core::curve3::V3;
use icam_core::solid3::Solid3;
use icam_core::step;
use rayon::prelude::*;

const TANGENT: f64 = 1e-6;
/// Where along an edge its kind is read (fractions of the edge).
const AT: [f64; 5] = [0.02, 0.25, 0.5, 0.75, 0.98];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Joint {
    Convex,
    Concave,
    Tangent,
    /// The normals are opposite: a knife edge, whose kind the compiler's test
    /// reads from roundoff.
    Knife,
}

/// The compiler's reading of an edge at a point, and the sine of the angle
/// between the faces' normals there.
fn joint(sol: &Solid3, ei: usize, p: V3) -> (Joint, f64) {
    let (_, n1, n2, m1) = sol.edge_frame(ei, p);
    let s = n1.cross(n2).len();
    let j = if s < TANGENT {
        if n1.dot(n2) > 0.0 { Joint::Tangent } else { Joint::Knife }
    } else if m1.dot(n2) < 0.0 {
        Joint::Convex
    } else {
        Joint::Concave
    };
    (j, s)
}

/// Decade of a sine: 0 for [0.1, 1], 1 for [0.01, 0.1), ... 16 for anything below 1e-16.
fn decade(s: f64) -> usize {
    if s <= 0.0 { 16 } else { ((-s.log10()).floor().max(0.0) as usize).min(16) }
}

#[derive(Default, Clone)]
struct Count {
    solids: usize,
    faces: usize,
    edges: usize,
    kind: BTreeMap<Joint, usize>,
    /// Edges whose kind changes along them: between convex and concave without a
    /// tangent reading, and through a tangent reading.
    flip: usize,
    via_tangent: usize,
    /// Edges read tangent anywhere, by the decade of the largest sine along them.
    tangent_max: [usize; 17],
    /// Every edge, by the decade of the sine at its middle.
    mid: [usize; 17],
    verts: usize,
    v_convex: usize,
    v_concave: usize,
    v_mixed: BTreeMap<usize, usize>,
    /// Vertices at a tangent or knife edge, by (convex, concave, tangent, knife) edges.
    v_tangent: BTreeMap<(usize, usize, usize, usize), usize>,
    v_changing: usize,
    /// Vertices whose faces do not close into one fan (two cones touching).
    v_nonmanifold: usize,
    tol: f64,
    floor: f64,
    /// Near features at the declared tolerance and at the floor's reach.
    short: [usize; 2],
    pairs: [usize; 2],
    thin: [usize; 2],
}

fn add(a: &mut Count, b: &Count) {
    a.solids += b.solids;
    a.faces += b.faces;
    a.edges += b.edges;
    for (k, v) in &b.kind {
        *a.kind.entry(*k).or_insert(0) += v;
    }
    a.flip += b.flip;
    a.via_tangent += b.via_tangent;
    for i in 0..17 {
        a.tangent_max[i] += b.tangent_max[i];
        a.mid[i] += b.mid[i];
    }
    a.verts += b.verts;
    a.v_convex += b.v_convex;
    a.v_concave += b.v_concave;
    for (k, v) in &b.v_mixed {
        *a.v_mixed.entry(*k).or_insert(0) += v;
    }
    for (k, v) in &b.v_tangent {
        *a.v_tangent.entry(*k).or_insert(0) += v;
    }
    a.v_changing += b.v_changing;
    a.v_nonmanifold += b.v_nonmanifold;
    a.tol = a.tol.max(b.tol);
    a.floor = a.floor.max(b.floor);
    for i in 0..2 {
        a.short[i] += b.short[i];
        a.pairs[i] += b.pairs[i];
        a.thin[i] += b.thin[i];
    }
}

fn find(p: &mut HashMap<usize, usize>, x: usize) -> usize {
    let mut r = x;
    while let Some(&q) = p.get(&r) {
        if q == r {
            break;
        }
        r = q;
    }
    p.insert(x, r);
    r
}

fn count_solid(sol: &Solid3) -> Count {
    let floor = (10.0 * sol.tol).max(1e-4);
    let scale = [sol.tol, 2.0 * 3f64.sqrt() * floor];
    let mut c = Count { solids: 1, faces: sol.faces.len(), tol: sol.tol, floor, ..Default::default() };
    let real = |ei: usize| sol.edges[ei].faces.len() == 2 && sol.edges[ei].faces[0] != sol.edges[ei].faces[1];
    // Edges: the kind and the angle at five points along each.
    let reads: Vec<Option<[(Joint, f64); 5]>> = (0..sol.edges.len())
        .into_par_iter()
        .map(|ei| real(ei).then(|| AT.map(|s| joint(sol, ei, sol.edges[ei].curve.at(s)))))
        .collect();
    let mut changing = vec![false; sol.edges.len()];
    for (ei, r) in reads.iter().enumerate() {
        let Some(r) = r else { continue };
        c.edges += 1;
        *c.kind.entry(r[2].0).or_insert(0) += 1;
        c.mid[decade(r[2].1)] += 1;
        if r.iter().any(|x| x.0 != r[0].0) {
            changing[ei] = true;
            if r.iter().any(|x| matches!(x.0, Joint::Tangent | Joint::Knife)) {
                c.via_tangent += 1
            } else {
                c.flip += 1
            }
        }
        if r.iter().any(|x| x.0 == Joint::Tangent) {
            c.tangent_max[decade(r.iter().map(|x| x.1).fold(0.0, f64::max))] += 1;
        }
    }
    // Vertices: their edges' kinds just off the vertex.
    let mut v_edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); sol.verts.len()];
    for (ei, e) in sol.edges.iter().enumerate() {
        if real(ei) {
            v_edges[e.v[0]].push((ei, true));
            if e.v[1] != e.v[0] {
                v_edges[e.v[1]].push((ei, false));
            }
        }
    }
    // Fans: edges at a vertex are linked when one face's loop passes from one to the
    // other there (seams included, so a cylinder's seam does not split its fan).
    let mut parent: Vec<HashMap<usize, usize>> = vec![HashMap::new(); sol.verts.len()];
    for f in &sol.faces {
        for lp in &f.loops {
            for i in 0..lp.len() {
                let (a, b) = (lp[i], lp[(i + 1) % lp.len()]);
                let ea = &sol.edges[a.edge];
                let v = if a.rev { ea.v[0] } else { ea.v[1] };
                let p = &mut parent[v];
                let (ra, rb) = (find(p, a.edge), find(p, b.edge));
                p.insert(ra, rb);
            }
        }
    }
    for (vi, es) in v_edges.iter().enumerate() {
        if es.is_empty() {
            continue;
        }
        c.verts += 1;
        let roots: HashSet<usize> = es.iter().map(|&(ei, _)| find(&mut parent[vi], ei)).collect();
        if roots.len() > 1 {
            c.v_nonmanifold += 1;
        }
        let ks: Vec<Joint> = es.iter().map(|&(ei, start)| joint(sol, ei, sol.edges[ei].curve.at(if start { 1e-3 } else { 1.0 - 1e-3 })).0).collect();
        let n = |j: Joint| ks.iter().filter(|k| **k == j).count();
        let (cv, cc, tg, kn) = (n(Joint::Convex), n(Joint::Concave), n(Joint::Tangent), n(Joint::Knife));
        if tg + kn > 0 {
            *c.v_tangent.entry((cv, cc, tg, kn)).or_insert(0) += 1;
        } else if cv > 0 && cc > 0 {
            *c.v_mixed.entry(ks.len()).or_insert(0) += 1;
        } else if cv > 0 {
            c.v_convex += 1;
        } else {
            c.v_concave += 1;
        }
        if es.iter().any(|&(ei, _)| changing[ei]) {
            c.v_changing += 1;
        }
    }
    // Near features.
    let pts = |ei: usize| -> Vec<V3> { (0..9).map(|k| sol.edges[ei].curve.at(k as f64 / 8.0)).collect() };
    for ei in 0..sol.edges.len() {
        if real(ei) {
            let p = pts(ei);
            let len: f64 = p.windows(2).map(|w| (w[1] - w[0]).len()).sum();
            for i in 0..2 {
                c.short[i] += (len < scale[i]) as usize;
            }
        }
    }
    let joined: HashSet<(usize, usize)> = sol.edges.iter().map(|e| (e.v[0].min(e.v[1]), e.v[0].max(e.v[1]))).collect();
    let vp: Vec<V3> = sol.verts.iter().map(|v| v.p).collect();
    let pairs: Vec<[usize; 2]> = (0..vp.len())
        .into_par_iter()
        .map(|i| {
            let mut n = [0; 2];
            for j in i + 1..vp.len() {
                let d = (vp[i] - vp[j]).len();
                if d < scale[1] && !joined.contains(&(i, j)) {
                    for k in 0..2 {
                        n[k] += (d < scale[k]) as usize;
                    }
                }
            }
            n
        })
        .collect();
    for n in pairs {
        for k in 0..2 {
            c.pairs[k] += n[k];
        }
    }
    // Thin faces: two edges of one face that share no vertex, closer than the scale.
    let gaps: Vec<f64> = (0..sol.faces.len())
        .into_par_iter()
        .map(|fi| {
            let es: Vec<usize> = sol.faces[fi].loops.iter().flatten().map(|u| u.edge).filter(|&e| real(e)).collect();
            let mut best = f64::INFINITY;
            for (a, &ea) in es.iter().enumerate() {
                for &eb in &es[a + 1..] {
                    let (ra, rb) = (&sol.edges[ea], &sol.edges[eb]);
                    if ea == eb || ra.v.iter().any(|x| rb.v.contains(x)) {
                        continue;
                    }
                    if icam_core::surface3::box_dist(ra.curve.at(0.5), &(rb.lo, rb.hi)) > (ra.hi - ra.lo).len() + scale[1] {
                        continue;
                    }
                    for p in pts(ea) {
                        best = best.min(rb.curve.closest(p).dist);
                    }
                }
            }
            best
        })
        .collect();
    for g in gaps {
        for k in 0..2 {
            c.thin[k] += (g < scale[k]) as usize;
        }
    }
    c
}

fn line(name: &str, c: &Count) {
    let k = |j: Joint| c.kind.get(&j).copied().unwrap_or(0);
    let mixed: usize = c.v_mixed.values().sum();
    let tangent: usize = c.v_tangent.values().sum();
    println!(
        "{name:<26} tol {:.0e} floor {:.0e} | edges {:>6}: cvx {} ccv {} tan {} knife {} changing {}+{} | verts {:>5}: cvx {} ccv {} MIXED {} TANGENT {} nonmanifold {} | under tol: short {} pairs {} thin {} | under floor reach: short {} pairs {} thin {}",
        c.tol,
        c.floor,
        c.edges,
        k(Joint::Convex),
        k(Joint::Concave),
        k(Joint::Tangent),
        k(Joint::Knife),
        c.flip,
        c.via_tangent,
        c.verts,
        c.v_convex,
        c.v_concave,
        mixed,
        tangent,
        c.v_nonmanifold,
        c.short[0],
        c.pairs[0],
        c.thin[0],
        c.short[1],
        c.pairs[1],
        c.thin[1]
    );
}

fn detail(c: &Count) {
    let hist = |h: &[usize; 17]| h.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, n)| format!("1e-{i}:{n}")).collect::<Vec<_>>().join(" ");
    println!("  edges by the sine of the angle between normals at their middle (decade: count): {}", hist(&c.mid));
    println!("  edges read tangent somewhere, by the largest sine along them: {}", hist(&c.tangent_max));
    println!("  edges changing kind: convex<->concave without a tangent reading {}, through a tangent or knife reading {}", c.flip, c.via_tangent);
    let md: Vec<String> = c.v_mixed.iter().map(|(k, v)| format!("{k} edges:{v}")).collect();
    println!("  mixed vertices by edge count: {}", md.join(", "));
    let mut tv: Vec<_> = c.v_tangent.iter().collect();
    tv.sort_by(|a, b| b.1.cmp(a.1));
    let tv: Vec<String> = tv
        .iter()
        .take(8)
        .map(|((cv, cc, tg, kn), n)| format!("{cv}cvx+{cc}ccv+{tg}tan{}:{n}", if *kn > 0 { format!("+{kn}knife") } else { String::new() }))
        .collect();
    println!("  tangent vertices by edge kinds (most common 8): {}", tv.join(", "));
    println!("  vertices at an edge that changes kind: {}", c.v_changing);
    // Vertices by the rule that would give them a formula.
    let mixed3 = c.v_mixed.get(&3).copied().unwrap_or(0);
    let mixed_more: usize = c.v_mixed.iter().filter(|(k, _)| **k != 3).map(|(_, v)| v).sum();
    let one_tan3: usize = c.v_tangent.iter().filter(|((cv, cc, tg, kn), _)| *tg == 1 && *kn == 0 && cv + cc == 2).map(|(_, v)| v).sum();
    let all_tan: usize = c.v_tangent.iter().filter(|((cv, cc, _, kn), _)| cv + cc == 0 && *kn == 0).map(|(_, v)| v).sum();
    let tangent: usize = c.v_tangent.values().sum();
    let pct = |n: usize| 100.0 * n as f64 / c.verts.max(1) as f64;
    let now = c.v_convex + c.v_concave;
    println!(
        "  vertices by rule: all one kind (covered now) {} ({:.0}%); three edges, mixed {} ({:.0}%); three edges, one tangent {} ({:.0}%); all tangent {} ({:.0}%); mixed, four or more edges {} ({:.0}%); other tangent {} ({:.0}%)",
        now,
        pct(now),
        mixed3,
        pct(mixed3),
        one_tan3,
        pct(one_tan3),
        all_tan,
        pct(all_tan),
        mixed_more,
        pct(mixed_more),
        tangent - one_tan3 - all_tan,
        pct(tangent - one_tan3 - all_tan)
    );
}

/// `icam-check gaps <file.step | dir>`.
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
    let mut total = Count::default();
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let Ok(brep) = step::read(f) else {
            println!("{name}: read failed");
            continue;
        };
        let tol = brep.tolerance_mm.max(1e-6);
        let mut c = Count::default();
        for s in &brep.solids {
            if let Ok((sol, _)) = Solid3::from_step(s, tol) {
                add(&mut c, &count_solid(&sol));
            }
        }
        if c.solids > 0 {
            line(&name, &c);
            add(&mut total, &c);
        }
    }
    if files.len() > 1 {
        line("all files", &total);
    }
    detail(&total);
}
