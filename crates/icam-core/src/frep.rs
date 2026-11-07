//! F-rep compiler: an exact B-rep solid becomes one global F-rep, a CSG tree
//! (min = union, max = intersection) over exact leaves, and any box of space
//! gets a pruned tree that is an exact restriction of that one function.
//!
//! Version 1 covers prismatic solids that sit on their bottom (every downward
//! face at the lowest z): the solid is the union over levels z_k (the distinct
//! heights of its upward faces) of the column
//!     { (x, y, z) : (x, y) in S_k,  z_bot <= z <= z_k }
//! where S_k is the cross-section just below z_k. The columns are nested
//! (S_k shrinks as k grows), so they overlap instead of touching and the union
//! has no internal seams:
//!     f(x, y, z) = min_k max( sd_Sk(x, y),  z - z_k,  z_bot - z )
//! Each section leaf sd_Sk is the exact signed distance to S_k, whose boundary
//! is exactly the curves of the upward faces at or above z_k whose walls reach
//! below z_k (lines, arcs and native B-splines, straight from the B-rep).
//!
//! Every leaf is 1-Lipschitz and min / max keep that, and the zero set is
//! exactly the solid's boundary, so |f| never exceeds the true distance to the
//! boundary: f is the bound field of the two-field discipline, and it is exact
//! wherever a single leaf is active. Contact identity survives: evaluation
//! reports the active leaf, and a section leaf reports the source face and
//! curve of its nearest boundary point.
//!
//! Version 2 (next) replaces each section leaf by a per-curve CSG subtree over
//! untrimmed primitives, which needs B-rep to CSG conversion with separating
//! primitives (Shapiro and Vossler); nothing above the leaves changes.

use std::sync::Arc;

use crate::geom::{Curve, V2};
use crate::part::Part;
use crate::query::{face_sd, NONE};

#[derive(Clone, Debug)]
pub enum Node {
    /// Exact signed distance (negative inside) to cross-section `sec`, in x and y.
    Section { sec: u32 },
    /// z - z0: negative below the plane (material under a top at z0).
    Below { z: f64 },
    /// z0 - z: negative above the plane (material over a bottom at z0).
    Above { z: f64 },
    Min(Vec<u32>),
    Max(Vec<u32>),
}

#[derive(Clone, Debug)]
pub struct Frep {
    pub nodes: Vec<Node>,
    pub root: u32,
    /// Cross-sections: single-face parts whose loops bound S_k. This and the other
    /// per-section data are shared, so a pruned copy of the tree costs only its nodes.
    pub sections: Arc<Vec<Part>>,
    /// For each section and each of its curves, the source part's edge index.
    pub section_src: Arc<Vec<Vec<u32>>>,
    pub levels: Arc<Vec<f64>>,
    pub z_bot: f64,
    /// A curve grid per section with many curves (None: few enough to read them all).
    pub grids: Arc<Vec<Option<crate::curvegrid::CurveGrid>>>,
}

/// Sections with at least this many curves get a curve grid.
const GRID_FROM: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct FrepEval {
    pub value: f64,
    pub grad: [f64; 3],
    /// Node id of the active leaf.
    pub leaf: u32,
    /// For a section leaf: the source part's edge nearest the point, else NONE.
    pub src_edge: u32,
}

impl Frep {
    /// Signed distance from (x, y) to section `sec`'s region.
    pub fn section_sd(&self, sec: u32, p: V2) -> crate::query::FaceDist {
        let s = &self.sections[sec as usize];
        match self.grids.get(sec as usize).and_then(Option::as_ref) {
            Some(g) => g.sd(s, p),
            None => face_sd(s, 0, p),
        }
    }

    fn leaf_eval(&self, id: u32, p: [f64; 3]) -> FrepEval {
        match &self.nodes[id as usize] {
            Node::Section { sec } => {
                let fd = self.section_sd(*sec, V2::new(p[0], p[1]));
                FrepEval { value: fd.sd, grad: [fd.grad.x, fd.grad.y, 0.0], leaf: id, src_edge: self.section_src[*sec as usize][fd.edge as usize] }
            }
            Node::Below { z } => FrepEval { value: p[2] - z, grad: [0.0, 0.0, 1.0], leaf: id, src_edge: NONE },
            Node::Above { z } => FrepEval { value: z - p[2], grad: [0.0, 0.0, -1.0], leaf: id, src_edge: NONE },
            _ => unreachable!(),
        }
    }

    /// Value, gradient and active leaf of node `id` at p.
    pub fn eval_node(&self, id: u32, p: [f64; 3]) -> FrepEval {
        match &self.nodes[id as usize] {
            Node::Min(c) => c.iter().map(|&k| self.eval_node(k, p)).min_by(|a, b| a.value.total_cmp(&b.value)).unwrap(),
            Node::Max(c) => c.iter().map(|&k| self.eval_node(k, p)).max_by(|a, b| a.value.total_cmp(&b.value)).unwrap(),
            _ => self.leaf_eval(id, p),
        }
    }

    pub fn eval(&self, p: [f64; 3]) -> FrepEval {
        self.eval_node(self.root, p)
    }

    /// Certified bounds of node `id` over the box [lo, hi]: exact for the z
    /// planes; for a section, its value at the box's center in x, y plus and
    /// minus the half diagonal (the leaf is 1-Lipschitz).
    pub fn interval(&self, id: u32, lo: [f64; 3], hi: [f64; 3]) -> (f64, f64) {
        match &self.nodes[id as usize] {
            Node::Section { sec } => {
                let c = V2::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
                let r = 0.5 * ((hi[0] - lo[0]).hypot(hi[1] - lo[1]));
                let v = self.section_sd(*sec, c).sd;
                (v - r, v + r)
            }
            Node::Below { z } => (lo[2] - z, hi[2] - z),
            Node::Above { z } => (z - hi[2], z - lo[2]),
            Node::Min(c) => c.iter().map(|&k| self.interval(k, lo, hi)).fold((f64::INFINITY, f64::INFINITY), |a, b| (a.0.min(b.0), a.1.min(b.1))),
            Node::Max(c) => c.iter().map(|&k| self.interval(k, lo, hi)).fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |a, b| (a.0.max(b.0), a.1.max(b.1))),
        }
    }

    /// The tree restricted to the box [lo, hi]: every Min / Max child that can
    /// never be the active one anywhere in the box is dropped (it is dominated by
    /// the child with the best bound there), and single-child nodes collapse.
    /// The result equals the global function at every point of the box.
    pub fn prune(&self, lo: [f64; 3], hi: [f64; 3]) -> Frep {
        let mut out = Frep {
            nodes: Vec::new(),
            root: 0,
            sections: self.sections.clone(),
            section_src: self.section_src.clone(),
            levels: self.levels.clone(),
            z_bot: self.z_bot,
            grids: self.grids.clone(),
        };
        let root = self.prune_node(self.root, lo, hi, &mut out.nodes);
        out.root = root;
        out
    }

    fn prune_node(&self, id: u32, lo: [f64; 3], hi: [f64; 3], out: &mut Vec<Node>) -> u32 {
        let (is_min, children) = match &self.nodes[id as usize] {
            Node::Min(c) => (true, c),
            Node::Max(c) => (false, c),
            leaf => {
                out.push(leaf.clone());
                return (out.len() - 1) as u32;
            }
        };
        let iv: Vec<(f64, f64)> = children.iter().map(|&k| self.interval(k, lo, hi)).collect();
        // The child whose bound decides: for Max the highest lower bound, for Min the lowest upper bound.
        let best = if is_min {
            (0..iv.len()).min_by(|&a, &b| iv[a].1.total_cmp(&iv[b].1)).unwrap()
        } else {
            (0..iv.len()).max_by(|&a, &b| iv[a].0.total_cmp(&iv[b].0)).unwrap()
        };
        let keep: Vec<usize> = (0..iv.len())
            .filter(|&i| i == best || if is_min { iv[i].0 < iv[best].1 } else { iv[i].1 > iv[best].0 })
            .collect();
        let kids: Vec<u32> = keep.iter().map(|&i| self.prune_node(children[i], lo, hi, out)).collect();
        if kids.len() == 1 {
            return kids[0];
        }
        out.push(if is_min { Node::Min(kids) } else { Node::Max(kids) });
        (out.len() - 1) as u32
    }

    /// Leaves in the tree reachable from the root (a measure of pruning).
    pub fn leaf_count(&self) -> usize {
        fn count(f: &Frep, id: u32) -> usize {
            match &f.nodes[id as usize] {
                Node::Min(c) | Node::Max(c) => c.iter().map(|&k| count(f, k)).sum(),
                _ => 1,
            }
        }
        count(self, self.root)
    }
}

/// Compiles an imported prismatic part (from `prismatic::from_brep`, which
/// records each face's orientation and solid, and each curve's wall depth) into
/// its F-rep: each solid on its own bottom, the solids joined by min.
pub fn compile_prismatic(part: &Part, tol: f64) -> Result<Frep, String> {
    if part.faces.is_empty() {
        return Err("no faces".into());
    }
    let mut solids: Vec<u32> = part.faces.iter().map(|f| f.solid).collect();
    solids.sort();
    solids.dedup();
    let mut out = Frep { nodes: Vec::new(), root: 0, sections: Default::default(), section_src: Default::default(), levels: Default::default(), z_bot: f64::INFINITY, grids: Default::default() };
    let mut roots = Vec::new();
    for s in solids {
        roots.push(compile_solid(part, s, tol, &mut out)?);
    }
    let levels = Arc::make_mut(&mut out.levels);
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|a, b| (*a - *b).abs() <= tol);
    out.root = if roots.len() == 1 {
        roots[0]
    } else {
        out.nodes.push(Node::Min(roots));
        (out.nodes.len() - 1) as u32
    };
    out.grids = Arc::new(out.sections.iter().map(|s| (s.edges.len() >= GRID_FROM).then(|| crate::curvegrid::CurveGrid::new(s, 0))).collect());
    Ok(out)
}

fn compile_solid(part: &Part, solid: u32, tol: f64, out: &mut Frep) -> Result<u32, String> {
    let faces: Vec<usize> = (0..part.faces.len()).filter(|&f| part.faces[f].solid == solid).collect();
    let z_bot = faces.iter().map(|&f| part.faces[f].z).fold(f64::INFINITY, f64::min);
    // A face looking down above the bottom roofs an overhang (a back-side pocket, an
    // undercut). A vertical tool from above never reaches under it: each column stands
    // from the bottom, so the space below is solid here.
    for &fi in &faces {
        let f = &part.faces[fi];
        if f.up.is_none() {
            return Err(format!("face {}: orientation unknown", f.id));
        }
    }
    let mut levels: Vec<f64> = faces.iter().filter(|&&f| part.faces[f].up == Some(true)).map(|&f| part.faces[f].z).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|a, b| (*a - *b).abs() <= tol);
    if levels.is_empty() || levels.iter().all(|&z| z <= z_bot + tol) {
        return Err(format!("solid {solid}: no upward face above its bottom (zero thickness, or not a closed solid)"));
    }
    let mut columns = Vec::new();
    for &zk in &levels {
        // Boundary curves of S_k: upward faces at or above z_k, walls reaching below it.
        let mut cur: Vec<(Curve, u32)> = Vec::new();
        for &fi in &faces {
            let f = &part.faces[fi];
            if f.up != Some(true) || f.z < zk - tol {
                continue;
            }
            for (ei, e) in part.face_edges(fi) {
                if e.wall_zlo.is_nan() {
                    return Err(format!("face {}: a boundary curve has no wall under it (an open shell or a zero-thickness body)", f.id));
                }
                // A curve that projects to a point (an edge running up a wall) bounds nothing.
                let (lo, hi) = e.curve.bbox();
                if e.wall_zlo < zk - tol && (hi - lo).len() > tol {
                    cur.push((e.curve.clone(), ei as u32));
                }
            }
        }
        let (loops, srcs) = chain_loops(&cur, tol, zk)?;
        let mut sp = Part::default();
        sp.add_face(out.sections.len() as u32, zk, loops, None, tol)?;
        let sec = out.sections.len() as u32;
        Arc::make_mut(&mut out.sections).push(sp);
        Arc::make_mut(&mut out.section_src).push(srcs);
        let nodes = &mut out.nodes;
        nodes.push(Node::Section { sec });
        nodes.push(Node::Below { z: zk });
        nodes.push(Node::Above { z: z_bot });
        let n = nodes.len() as u32;
        nodes.push(Node::Max(vec![n - 3, n - 2, n - 1]));
        columns.push(n);
    }
    // Floors: the heights of real upward faces (a fillet's top bounds a column only).
    Arc::make_mut(&mut out.levels).extend(levels.iter().filter(|&&z| faces.iter().any(|&f| part.faces[f].up == Some(true) && !part.faces[f].envelope && (part.faces[f].z - z).abs() <= tol)));
    out.z_bot = out.z_bot.min(z_bot);
    Ok(if columns.len() == 1 {
        columns[0]
    } else {
        out.nodes.push(Node::Min(columns));
        (out.nodes.len() - 1) as u32
    })
}

/// Chains curves (each with a source tag) into closed loops by matching each curve's
/// end to the nearest unused start within `tol` (ties: the earlier curve); the tags come
/// back in loop order. Starts are hashed in cells of `tol`, so a match reads only the
/// 3 x 3 cells around the end.
pub(crate) fn chain_loops(cur: &[(Curve, u32)], tol: f64, zk: f64) -> Result<(Vec<Vec<Curve>>, Vec<u32>), String> {
    let h = tol.max(1e-12);
    let key = |p: V2| ((p.x / h).floor() as i64, (p.y / h).floor() as i64);
    let mut starts: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
    for (i, c) in cur.iter().enumerate() {
        starts.entry(key(c.0.start())).or_default().push(i);
    }
    let mut used = vec![false; cur.len()];
    let mut loops = Vec::new();
    let mut srcs = Vec::new();
    for s0 in 0..cur.len() {
        if used[s0] {
            continue;
        }
        used[s0] = true;
        let mut lp = vec![cur[s0].0.clone()];
        let mut src = vec![cur[s0].1];
        let start = cur[s0].0.start();
        let mut end = cur[s0].0.end();
        while (end - start).len() > tol {
            let (kx, ky) = key(end);
            let mut best: Option<(f64, usize)> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for &i in starts.get(&(kx + dx, ky + dy)).map_or(&[][..], |v| v.as_slice()) {
                        if used[i] {
                            continue;
                        }
                        let d = (cur[i].0.start() - end).len();
                        if d <= tol && best.is_none_or(|(bd, bi)| d < bd || (d == bd && i < bi)) {
                            best = Some((d, i));
                        }
                    }
                }
            }
            let next = best.map(|b| b.1).ok_or(format!("section at z {zk:.4}: boundary does not close near ({:.4}, {:.4})", end.x, end.y))?;
            used[next] = true;
            end = cur[next].0.end();
            lp.push(cur[next].0.clone());
            src.push(cur[next].1);
        }
        loops.push(lp);
        srcs.extend(src);
    }
    Ok((loops, srcs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prismatic_reference::{exact_boundary_distance, inside_solid};

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Curve> {
        let (a, b, c, d) = (V2::new(x0, y0), V2::new(x1, y0), V2::new(x1, y1), V2::new(x0, y1));
        vec![Curve::Line { a, b }, Curve::Line { a: b, b: c }, Curve::Line { a: c, b: d }, Curve::Line { a: d, b: a }]
    }

    /// A 10 x 10 x 10 block with a round pocket (r 2) down to z 4, as the importer
    /// would produce it: faces with orientation, curves with wall depths.
    fn pocketed_block() -> Part {
        let mut p = Part::default();
        let c = V2::new(5.0, 5.0);
        let two_pi = 2.0 * std::f64::consts::PI;
        let hole = vec![Curve::Arc { c, r: 2.0, a: V2::new(7.0, 5.0), sweep: -two_pi }];
        p.add_face(1, 10.0, vec![square(0.0, 0.0, 10.0, 10.0), hole], None, 1e-9).unwrap();
        for (i, e) in p.edges.iter_mut().enumerate() {
            e.wall_zlo = if i < 4 { 0.0 } else { 4.0 };
        }
        p.faces[0].up = Some(true);
        let floor = vec![Curve::Arc { c, r: 2.0, a: V2::new(7.0, 5.0), sweep: two_pi }];
        p.add_face(2, 4.0, vec![floor], None, 1e-9).unwrap();
        p.edges.last_mut().unwrap().wall_zlo = 4.0;
        p.faces[1].up = Some(true);
        // The bottom (faces down).
        let mut bottom = square(0.0, 0.0, 10.0, 10.0);
        bottom.reverse();
        let bottom = bottom.into_iter().map(|c| match c {
            Curve::Line { a, b } => Curve::Line { a: b, b: a },
            other => other,
        });
        p.add_face(3, 0.0, vec![bottom.collect()], None, 1e-9).unwrap();
        p.faces[2].up = Some(false);
        p
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn prismatic_frep_is_a_bound_field_with_the_right_sign() {
        let part = pocketed_block();
        let f = compile_prismatic(&part, 1e-9).unwrap();
        assert_eq!(*f.levels, vec![4.0, 10.0]);
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut pts = Vec::new();
        for _ in 0..4000 {
            let p = [rng.next() * 14.0 - 2.0, rng.next() * 14.0 - 2.0, rng.next() * 14.0 - 2.0];
            let v = f.eval(p).value;
            let inside = inside_solid(&part, p);
            let exact = exact_boundary_distance(&part, p);
            if exact > 1e-9 {
                assert_eq!(v < 0.0, inside, "sign at {p:?}: f {v}, inside {inside}");
            }
            assert!(v.abs() <= exact + 1e-9, "bound exceeds exact at {p:?}: |f| {} > {exact}", v.abs());
            pts.push((p, v));
        }
        // 1-Lipschitz.
        for w in pts.windows(2) {
            let (a, b) = (w[0].0, w[1].0);
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            assert!((w[0].1 - w[1].1).abs() <= d + 1e-9);
        }
        // Exact on a face, away from edges: the pocket floor's center is at distance 0.
        assert!(f.eval([5.0, 5.0, 4.0]).value.abs() < 1e-12);
        // Deep inside the pocket (air), directly above the floor: distance to the floor.
        assert!((f.eval([5.0, 5.0, 5.0]).value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pruned_trees_restrict_exactly() {
        let part = pocketed_block();
        let f = compile_prismatic(&part, 1e-9).unwrap();
        let mut rng = Rng(0x0fed_cba9_8765_4321);
        let mut total_leaves = 0;
        for _ in 0..300 {
            let c = [rng.next() * 14.0 - 2.0, rng.next() * 14.0 - 2.0, rng.next() * 14.0 - 2.0];
            let h = 0.25 + rng.next();
            let (lo, hi) = ([c[0] - h, c[1] - h, c[2] - h], [c[0] + h, c[1] + h, c[2] + h]);
            let pf = f.prune(lo, hi);
            total_leaves += pf.leaf_count();
            for _ in 0..20 {
                let p = [lo[0] + 2.0 * h * rng.next(), lo[1] + 2.0 * h * rng.next(), lo[2] + 2.0 * h * rng.next()];
                let (a, b) = (f.eval(p).value, pf.eval(p).value);
                assert!((a - b).abs() < 1e-12, "pruned {b} vs global {a} at {p:?}");
            }
        }
        assert!(total_leaves < 300 * f.leaf_count(), "pruning removed nothing");
    }
}
