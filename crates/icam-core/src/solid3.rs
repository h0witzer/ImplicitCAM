//! Exact signed distance to a general B-rep solid: trimmed faces on exact
//! surfaces, exact edges, vertices, and the sign from topology.
//!
//! Distance. The nearest boundary point of p is a face foot that lies in its
//! face, an interior point of an edge, or a vertex. Each analytic surface has a
//! single local minimum of distance (its foot), so a face contributes its foot
//! when the foot is in the face; otherwise its nearest point is on its edges,
//! which are candidates of their own.
//!
//! Trims. Whether a surface point q lies in its face is decided on the surface
//! itself, with no parameter-space curves (the STEP files carry none): walk from
//! q along a path on the surface made of exact planar sections (a ray on a
//! plane, a generator of a cylinder or cone, a meridian or latitude circle, a
//! great circle), find where the face's edges cross the path's cutting plane
//! (closed form, or Bernstein root isolation for B-splines), and look at the
//! first crossing: q is inside when the path leaves the face there, which the
//! edge's in-face direction (outward normal cross travel tangent) tells. A path
//! is abandoned when its first crossing is at a vertex or tangential, and the
//! next one is tried; seam edges (used twice by one face) are not boundaries.
//!
//! Sign. From the nearest feature: the outward normal at a face foot, the sum
//! of the two faces' normals at an edge point, the angle-weighted normal at a
//! vertex (Baerentzen and Aanaes).

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::curve3::{bez_box, box_gap, flatten, near_hits, swept, Bez3, Cross, Edge3, NearHit, V3};
use crate::geom::Near;
use crate::step;
use crate::surface3::{axial, box_dist, slab_bound, Creases, Entry, NurbsSurf, PatchItem, Surf, DIST_EPS};

mod gpu;

/// One use of an edge by a face loop; `rev` when the loop runs against the
/// stored curve's direction.
#[derive(Clone, Copy, Debug)]
pub struct Use {
    pub edge: usize,
    pub rev: bool,
}

#[derive(Clone, Debug)]
pub struct Face3 {
    pub id: u32,
    pub surf: Surf,
    /// +1 when the outward normal is the surface normal, else -1.
    pub sense: f64,
    pub loops: Vec<Vec<Use>>,
    /// Its vertex loops (a cone's apex): points of the face that bound its extent
    /// as its edges do.
    pub points: Vec<V3>,
    /// Edges this face uses twice (seams of a periodic surface); not boundaries.
    pub seams: Vec<usize>,
    /// Trim data of a B-spline face for the old path-walking trim test (the reference
    /// the certified loops are checked against, and the evaluator's fallback for a face
    /// whose loops were refused): built on first use (`Solid3::old_trim`).
    pub trim: std::sync::OnceLock<Option<NurbsTrim>>,
    /// The quadric a B-spline face is, when it is one that is not a STEP kind
    /// (`span_poly::recognize_quadric`): oriented so its G grows along the
    /// B-spline's Su x Sv, so `sense` applies to it as to the B-spline. The face
    /// stays a B-spline face for trims; the quadric is its exact surface.
    pub quadric: Option<crate::quadric::Quadric>,
    /// The face's trim loops in its parameter domain (`trim2`), certified within the
    /// file's tolerance; None when the face is not a B-spline face or was refused.
    pub trim2: Option<crate::trim2::Trim2>,
    /// An analytic face's surface as an exact rational B-spline, whose parameters
    /// its `trim2` loops are in, with the closed-form map from points to them.
    pub analytic2: Option<std::sync::Arc<crate::trim2::Analytic>>,
}

/// A B-spline face's boundary as exact rational Bezier pieces, for trim tests
/// along iso-curves.
#[derive(Clone, Debug)]
pub struct NurbsTrim {
    /// Two curves closer than this cross (edges sit on the surface only to within
    /// the file's tolerance, so an iso-curve passes an edge at up to that distance).
    pub thr: f64,
    /// Every trim piece with the edge use it belongs to; the tree indexes them.
    pub pieces: Vec<(Use, Bez3)>,
    pub tree: BoxTree,
    /// Edges lying along a side of the surface's parameter domain: the face reaches
    /// that side there, so a walk arriving at it leaves the face. With no trim
    /// pieces left, the face is the whole surface.
    pub sides: Vec<Use>,
    /// The surface closes on itself in u (or v): that side of the domain is a seam,
    /// and a walk along that direction wraps around it.
    pub closed_u: bool,
    pub closed_v: bool,
}

/// A bounding-volume tree over boxes: answers "is any box within d of this one".
#[derive(Clone, Debug, Default)]
pub struct BoxTree {
    boxes: Vec<(V3, V3)>,
    /// Caller's index of each (sorted) box.
    ids: Vec<usize>,
    /// Where each caller's index sits among the sorted boxes.
    at: Vec<usize>,
    /// (box, first, last) over the sorted boxes, children at 2k + 1 and 2k + 2.
    nodes: Vec<((V3, V3), usize, usize)>,
}

impl BoxTree {
    pub fn new(boxes: Vec<(V3, V3)>) -> BoxTree {
        let mut ids: Vec<usize> = (0..boxes.len()).collect();
        ids.sort_by(|&a, &b| (boxes[a].0.x + boxes[a].1.x).total_cmp(&(boxes[b].0.x + boxes[b].1.x)));
        let boxes = ids.iter().map(|&i| boxes[i]).collect();
        let mut at = vec![0; ids.len()];
        for (k, &i) in ids.iter().enumerate() {
            at[i] = k;
        }
        let mut t = BoxTree { boxes, ids, at, nodes: Vec::new() };
        if !t.boxes.is_empty() {
            t.build(0, 0, t.boxes.len());
        }
        t
    }
    fn build(&mut self, k: usize, a: usize, b: usize) -> (V3, V3) {
        if self.nodes.len() <= k {
            self.nodes.resize(k + 1, ((V3::default(), V3::default()), 0, 0));
        }
        let bx = if b - a <= 4 {
            self.boxes[a..b].iter().fold((V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN)), |acc, x| {
                (V3::new(acc.0.x.min(x.0.x), acc.0.y.min(x.0.y), acc.0.z.min(x.0.z)), V3::new(acc.1.x.max(x.1.x), acc.1.y.max(x.1.y), acc.1.z.max(x.1.z)))
            })
        } else {
            let m = (a + b) / 2;
            let (l, r) = (self.build(2 * k + 1, a, m), self.build(2 * k + 2, m, b));
            (V3::new(l.0.x.min(r.0.x), l.0.y.min(r.0.y), l.0.z.min(r.0.z)), V3::new(l.1.x.max(r.1.x), l.1.y.max(r.1.y), l.1.z.max(r.1.z)))
        };
        self.nodes[k] = (bx, a, b);
        bx
    }
    pub fn any_within(&self, q: &(V3, V3), d: f64) -> bool {
        !self.nodes.is_empty() && self.any_at(0, q, d)
    }
    /// Calls f with the index of every box within d of q.
    pub fn visit(&self, q: &(V3, V3), d: f64, f: &mut dyn FnMut(usize)) {
        if !self.nodes.is_empty() {
            self.visit_at(0, q, d, f);
        }
    }
    fn visit_at(&self, k: usize, q: &(V3, V3), d: f64, f: &mut dyn FnMut(usize)) {
        let (bx, a, b) = self.nodes[k];
        if box_gap(&bx, q) > d {
            return;
        }
        if b - a <= 4 {
            for i in a..b {
                if box_gap(&self.boxes[i], q) <= d {
                    f(self.ids[i]);
                }
            }
            return;
        }
        self.visit_at(2 * k + 1, q, d, f);
        self.visit_at(2 * k + 2, q, d, f);
    }
    /// Calls f with each box's index in order of the box's distance from p, while
    /// that distance is under the bound, which starts at `bound` and is what f
    /// last returned.
    pub fn nearest_first(&self, p: V3, mut bound: f64, f: &mut dyn FnMut(usize) -> f64) {
        if self.nodes.is_empty() {
            return;
        }
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(Nearer { lb: box_dist(p, &self.nodes[0].0), k: 0, leaf: false });
        while let Some(Nearer { lb, k, leaf }) = heap.pop() {
            if lb >= bound {
                break;
            }
            if leaf {
                bound = f(self.ids[k]);
                continue;
            }
            let (_, a, b) = self.nodes[k];
            if b - a <= 4 {
                for i in a..b {
                    let d = box_dist(p, &self.boxes[i]);
                    if d < bound {
                        heap.push(Nearer { lb: d, k: i, leaf: true });
                    }
                }
            } else {
                for c in [2 * k + 1, 2 * k + 2] {
                    let d = box_dist(p, &self.nodes[c].0);
                    if d < bound {
                        heap.push(Nearer { lb: d, k: c, leaf: false });
                    }
                }
            }
        }
    }
    /// The box of the caller's box `id`.
    pub fn boxes_of(&self, id: usize) -> (V3, V3) {
        self.boxes[self.at[id]]
    }
    fn any_at(&self, k: usize, q: &(V3, V3), d: f64) -> bool {
        let (bx, a, b) = self.nodes[k];
        if box_gap(&bx, q) > d {
            return false;
        }
        if b - a <= 4 {
            return self.boxes[a..b].iter().any(|x| box_gap(x, q) <= d);
        }
        self.any_at(2 * k + 1, q, d) || self.any_at(2 * k + 2, q, d)
    }
}

/// Min-heap entry of `BoxTree::nearest_first`: a node, or a box when `leaf`.
struct Nearer {
    lb: f64,
    k: usize,
    leaf: bool,
}
impl PartialEq for Nearer {
    fn eq(&self, o: &Self) -> bool {
        self.lb == o.lb
    }
}
impl Eq for Nearer {}
impl PartialOrd for Nearer {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Nearer {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.lb.total_cmp(&self.lb)
    }
}

impl Face3 {
    /// Outward unit normal at the surface point nearest x.
    pub fn normal(&self, x: V3) -> V3 {
        self.surf.normal(x) * self.sense
    }
}

#[derive(Clone, Debug)]
pub struct EdgeRec {
    pub id: u32,
    pub curve: Edge3,
    /// Faces using the edge (two in a closed shell; the same face twice on a seam).
    pub faces: Vec<usize>,
    /// Start and end vertex, in the stored direction.
    pub v: [usize; 2],
    pub lo: V3,
    pub hi: V3,
    /// The first use's STEP sense, to orient later uses.
    forward: bool,
}

#[derive(Clone, Debug)]
pub struct Vert {
    pub id: u32,
    pub p: V3,
    /// Angle-weighted pseudo-normal (not unit).
    pub pn: V3,
}

#[derive(Clone, Debug)]
pub struct Solid3 {
    pub id: u32,
    pub name: String,
    pub faces: Vec<Face3>,
    pub edges: Vec<EdgeRec>,
    pub verts: Vec<Vert>,
    /// Model tolerance (mm): paths passing closer than a few of these to a vertex are not trusted.
    pub tol: f64,
    pub lo: V3,
    pub hi: V3,
}

#[derive(Clone, Debug, Default)]
pub struct Report3 {
    pub faces: std::collections::BTreeMap<&'static str, usize>,
    pub edges: std::collections::BTreeMap<&'static str, usize>,
    /// Largest distance of a vertex from its edge's curve.
    pub max_vertex_off: f64,
    /// Largest distance of an edge point (ends and middle) from a face's surface.
    pub max_edge_off: f64,
    /// B-spline faces whose parameter-domain trim loops could not be certified.
    pub trim2_refused: usize,
    pub trim2_notes: Vec<String>,
    /// Largest certified deviation of a fitted trim piece from its edge (mm).
    pub trim2_bound: f64,
    pub trim2_secs: f64,
    /// Largest gap between consecutive edges' ends at a vertex of a B-spline face
    /// (the file's own; a loop closes across it).
    pub vertex_gap: f64,
    /// B-spline faces with such a gap over the file's tolerance.
    pub faces_over_tol: usize,
    pub gap_notes: Vec<String>,
    /// Faces whose edges lie farther from their surface than the file's declared
    /// tolerance (the edges' points sampled in building the loops), those points, the
    /// farthest, and the worst faces (at most 20).
    pub off_tol_faces: usize,
    pub off_tol_points: u64,
    pub off_tol_max: f64,
    pub off_tol_notes: Vec<String>,
    /// Lemon faces; those turned over to match their tangent neighbors; those
    /// with no tangent neighbor to tell (left with the standard's sense).
    pub lemons: usize,
    pub lemons_turned: usize,
    pub lemons_undecided: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Feature {
    Face(usize),
    Edge(usize),
    Vertex(usize),
}

/// Signed distance with its nearest point and feature.
#[derive(Clone, Copy, Debug)]
pub struct Sd {
    pub d: f64,
    pub q: V3,
    pub feature: Feature,
    /// A face foot whose trim test found no usable path (treated as outside the face).
    pub unsure: bool,
}

pub fn edge_kind_name(e: &Edge3) -> &'static str {
    edge_kind(e)
}

fn edge_kind(e: &Edge3) -> &'static str {
    match e {
        Edge3::Line { .. } => "line",
        Edge3::Arc { .. } => "circle",
        Edge3::Ellipse { .. } => "ellipse",
        Edge3::Spline(_) => "B-spline",
    }
}

/// The parameter-domain trim loops of a B-spline face (`trim2::build`), from its
/// loops of edge uses; Ok(None) for other faces.
/// The old path-walking trim test's data for a B-spline face (`Face3::trim`).
fn old_trim(f: &Face3, edges: &[EdgeRec], tol: f64) -> Option<NurbsTrim> {
    let Surf::Nurbs(ns) = &f.surf else { return None };
    // Edge samples with their surface parameters; their distance from the surface sets
    // the crossing threshold.
    let uses: Vec<Use> = f.loops.iter().flatten().filter(|u| !f.seams.contains(&u.edge)).copied().collect();
    let mut dev: f64 = 0.0;
    let samples: Vec<Vec<(V3, f64, f64)>> = uses
        .iter()
        .map(|u| {
            (0..=8)
                .map(|k| {
                    let x = edges[u.edge].curve.at(k as f64 / 8.0);
                    let h = ns.closest(x, f64::INFINITY).unwrap();
                    dev = dev.max(h.dist);
                    (x, h.u, h.v)
                })
                .collect()
        })
        .collect();
    let thr = (4.0 * dev).max(1e-7);
    let mut pieces = Vec::new();
    let mut boxes = Vec::new();
    let mut sides = Vec::new();
    for (u, smp) in uses.iter().zip(&samples) {
        // An edge along a side of the parameter domain is that side, not a trim.
        let side = [(true, ns.u_range.0), (true, ns.u_range.1), (false, ns.v_range.0), (false, ns.v_range.1)].into_iter().find(|&(in_u, x)| {
            smp.iter().all(|&(q, pu, pv)| {
                let e = if in_u { ns.eval(x, pv) } else { ns.eval(pu, x) };
                (e.p - q).len() <= thr
            })
        });
        if side.is_some() {
            sides.push(*u);
            continue;
        }
        // Boundary pieces no bigger than about half a span, so the patches that count
        // as touching the boundary stay close to it.
        let size = 0.5 * ns.span_size();
        let mut stack: Vec<Bez3> = flatten(&edges[u.edge].curve.bez_pieces()).into_iter().rev().collect();
        while let Some(z) = stack.pop() {
            let bx = bez_box(&z);
            if (bx.1 - bx.0).len() > size && (z.t1 - z.t0).abs() > 1e-12 {
                let (l, r) = z.split(0.5);
                stack.push(r);
                stack.push(l);
            } else {
                boxes.push(bx);
                pieces.push((*u, z));
            }
        }
    }
    let (closed_u, closed_v) = (ns.closed(true, tol), ns.closed(false, tol));
    Some(NurbsTrim { thr, pieces, tree: BoxTree::new(boxes), sides, closed_u, closed_v })
}

pub fn build_trim2(f: &Face3, edges: &[EdgeRec], tol: f64) -> Result<Option<(crate::trim2::Trim2, f64, Option<crate::trim2::Analytic>)>, String> {
    // An analytic face is first written as an exact rational B-spline over its
    // extent (from its edges and vertex loops, padded), in whose parameters its
    // loops are built.
    let analytic = match &f.surf {
        Surf::Nurbs(_) => None,
        surf => {
            let pts: Vec<V3> = f.loops.iter().flatten().flat_map(|u| (0..=8).map(move |k| (u.edge, k))).map(|(e, k)| edges[e].curve.at(k as f64 / 8.0)).chain(f.points.iter().copied()).collect();
            let fr = surf.frame().ok_or("an analytic surface without a frame")?;
            let local: Vec<V3> = pts.iter().map(|p| fr.local(*p)).collect();
            let span = |g: &dyn Fn(&V3) -> f64| {
                let (lo, hi) = local.iter().map(|l| g(l)).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
                let pad = 0.05 * (hi - lo) + 1e-3 * tol.max(1e-6) + 1e-9 * (hi.abs() + lo.abs());
                (lo - pad, hi + pad)
            };
            let (x, y, z) = (span(&|l: &V3| l.x), span(&|l: &V3| l.y), span(&|l: &V3| l.z));
            let range = match surf {
                Surf::Plane { .. } => [x.0, x.1, y.0, y.1],
                _ => [z.0, z.1, 0.0, 0.0],
            };
            match crate::trim2::Analytic::of_with(surf, range, &pts) {
                Some(a) => Some(a),
                None => return Ok(None),
            }
        }
    };
    let ns: &crate::surface3::NurbsSurf = match (&f.surf, &analytic) {
        (Surf::Nurbs(ns), _) => ns,
        (_, Some(a)) => &a.ns,
        _ => return Ok(None),
    };
    let loops: Vec<Vec<crate::trim2::EdgeIn>> = f
        .loops
        .iter()
        .map(|lp| {
            lp.iter()
                .map(|u| {
                    let e = &edges[u.edge];
                    let mut pieces = e.curve.bez_pieces();
                    if u.rev {
                        pieces = pieces
                            .iter()
                            .rev()
                            .map(|z| {
                                let mut r = z.reversed();
                                r.t0 = 1.0 - z.t1;
                                r.t1 = 1.0 - z.t0;
                                r
                            })
                            .collect();
                    }
                    let (start, end) = if u.rev { (e.curve.end(), e.curve.start()) } else { (e.curve.start(), e.curve.end()) };
                    let (v0, v1) = if u.rev { (e.v[1], e.v[0]) } else { (e.v[0], e.v[1]) };
                    crate::trim2::EdgeIn { edge: u.edge, pieces, start, end, v0, v1, seam: f.seams.contains(&u.edge) }
                })
                .collect()
        })
        .collect();
    let debug = std::env::var("ICAM_TRIM2_FACE").ok().and_then(|x| x.parse::<u32>().ok()) == Some(f.id);
    if debug {
        println!("face #{} sense {}: loops {:?}, seams {:?}", f.id, f.sense, f.loops.iter().map(|l| l.iter().map(|u| (u.edge, u.rev)).collect::<Vec<_>>()).collect::<Vec<_>>(), f.seams);
    }
    // An analytic surface closes where its kind does; a sampled test would also
    // call a direction closed when the face's extent along it is thinner than the
    // tolerance.
    let (closed_u, closed_v) = match &analytic {
        Some(a) => a.closed(),
        None => (ns.closed(true, tol), ns.closed(false, tol)),
    };
    // An analytic face's preimages in closed form: the foot, by its parameters in the
    // exact rational form (`Analytic::uv`, the map the evaluator reads loops through).
    let foot = |x: V3| {
        let q = f.surf.foot(x);
        let (u, v) = analytic.as_ref().map_or((0.0, 0.0), |a| a.uv(q));
        ([u, v], (x - q).len())
    };
    let exact: Option<&(dyn Fn(V3) -> ([f64; 2], f64) + Sync)> = analytic.is_some().then_some(&foot);
    crate::trim2::build(ns, exact, f.sense, &loops, tol, closed_u, closed_v, debug).map(|(t, gap)| Some((t, gap, analytic)))
}

pub fn surf_from_step(s: &step::Surface) -> Result<Surf, String> {
    Ok(match s {
        step::Surface::Plane { frame } => Surf::Plane { f: *frame },
        step::Surface::Cylinder { frame, r } => Surf::Cylinder { f: *frame, r: *r },
        step::Surface::Cone { frame, r, semi_angle } => {
            if !(*semi_angle > 0.0 && *semi_angle < PI / 2.0) {
                return Err(format!("cone with semi-angle {semi_angle} (radians expected)"));
            }
            Surf::Cone { f: *frame, r: *r, a: *semi_angle }
        }
        step::Surface::Sphere { frame, r } => Surf::Sphere { f: *frame, r: *r },
        step::Surface::Torus { frame, major, minor } => {
            if !(*minor > 0.0 && major.abs() < f64::MAX) {
                return Err(format!("torus with minor radius {minor}"));
            }
            Surf::Torus { f: *frame, major: *major, minor: *minor }
        }
        step::Surface::BSpline(b) => {
            let ns = NurbsSurf::new(b.du, b.dv, &b.ctrl, b.weights.as_ref(), &b.ku, &b.kv)?;
            Surf::Nurbs(std::sync::Arc::new(ns))
        }
        other => return Err(format!("{} surface", crate::prismatic::surface_name(other))),
    })
}

// ---------------------------------------------------------------- paths

#[derive(Clone, Copy, Debug)]
enum Side {
    Any,
    /// The half-space (x - o).er > 0: one side of a plane through an axis.
    Half { o: V3, er: V3 },
    /// Half and on the apex's positive side along the axis z (one generator of a cone).
    Nappe { o: V3, er: V3, apex: V3, z: V3 },
    /// Distance from the axis (o, z) above or below `major` (one circle of a torus latitude).
    Ring { o: V3, z: V3, major: f64, outer: bool },
}

impl Side {
    fn ok(&self, x: V3) -> bool {
        match *self {
            Side::Any => true,
            Side::Half { o, er } => (x - o).dot(er) > 0.0,
            Side::Nappe { o, er, apex, z } => (x - o).dot(er) > 0.0 && (x - apex).dot(z) > 0.0,
            Side::Ring { o, z, major, outer } => {
                let d = x - o;
                let rho = (d - z * d.dot(z)).len();
                (rho > major) == outer
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Ray { o: V3, d: V3 },
    /// Circle about c in the (e1, e2) plane, walked from angle a0 in direction dir (+1 / -1).
    Circle { c: V3, e1: V3, e2: V3, a0: f64, dir: f64 },
}

/// One leg of a path: the part of a planar section of the surface from a start
/// point, up to `s_end` (length for rays, angle for circles).
#[derive(Clone, Copy, Debug)]
struct Leg {
    n: V3,
    c: f64,
    kind: Kind,
    side: Side,
    s_end: f64,
    /// The leg runs off the (bounded) face: no crossing means outside.
    exits: bool,
    /// Where the leg starts.
    start: V3,
}

impl Leg {
    fn param(&self, x: V3) -> f64 {
        match self.kind {
            Kind::Ray { o, d } => (x - o).dot(d),
            Kind::Circle { c, e1, e2, a0, dir } => {
                let r = x - c;
                swept(a0, r.dot(e2).atan2(r.dot(e1)), dir)
            }
        }
    }
    fn dir_at(&self, x: V3) -> V3 {
        match self.kind {
            Kind::Ray { d, .. } => d,
            Kind::Circle { c, e1, e2, dir, .. } => {
                let r = x - c;
                let a = r.dot(e2).atan2(r.dot(e1));
                (e2 * a.cos() - e1 * a.sin()) * dir
            }
        }
    }
}

enum Outcome {
    Inside(bool),
    NoHit,
    Degenerate,
}

fn ray(q: V3, d: V3, plane_n: V3, side: Side) -> Leg {
    Leg { n: plane_n, c: plane_n.dot(q), kind: Kind::Ray { o: q, d }, side, s_end: f64::INFINITY, exits: true, start: q }
}

/// Circle through `start` about center c with in-plane basis (e1, e2), normal n = e1 x e2.
fn circle(start: V3, c: V3, e1: V3, e2: V3, dir: f64, side: Side, s_end: f64) -> Leg {
    let n = e1.cross(e2).unit();
    let r = start - c;
    Leg { n, c: n.dot(c), kind: Kind::Circle { c, e1, e2, a0: r.dot(e2).atan2(r.dot(e1)), dir }, side, s_end, exits: false, start }
}

impl Solid3 {
    pub fn from_step(solid: &step::Solid, tol: f64) -> Result<(Solid3, Report3), String> {
        let mut rep = Report3::default();
        let mut faces = Vec::new();
        let mut edges: Vec<EdgeRec> = Vec::new();
        let mut verts: Vec<Vert> = Vec::new();
        let mut edge_ix: HashMap<u32, usize> = HashMap::new();
        let mut vert_ix: HashMap<u32, usize> = HashMap::new();
        let mut point_loops: Vec<(usize, V3)> = Vec::new();
        let mut vert = |id: u32, p: V3, verts: &mut Vec<Vert>| -> usize {
            *vert_ix.entry(id).or_insert_with(|| {
                verts.push(Vert { id, p, pn: V3::default() });
                verts.len() - 1
            })
        };
        let trace = std::env::var("ICAM_TRACE_BUILD").is_ok();
        let t0 = std::time::Instant::now();
        let lap = |what: &str| {
            if trace {
                eprintln!("  build {:>8.1} ms  {what}", t0.elapsed().as_secs_f64() * 1e3);
            }
        };
        for bf in &solid.faces {
            let surf = surf_from_step(&bf.surface).map_err(|e| format!("face #{}: {e}", bf.id))?;
            lap(&format!("face #{} surface ({})", bf.id, surf.name()));
            let fi = faces.len();
            *rep.faces.entry(surf.name()).or_default() += 1;
            let mut loops = Vec::new();
            let mut seams = Vec::new();
            let mut points = Vec::new();
            for b in &bf.bounds {
                if let Some(v) = b.vertex {
                    point_loops.push((fi, v));
                    points.push(v);
                    continue;
                }
                let mut lp = Vec::new();
                for oe in &b.edges {
                    let ix = match edge_ix.get(&oe.edge) {
                        Some(&ix) => {
                            let er = &mut edges[ix];
                            if er.faces.contains(&fi) {
                                seams.push(ix);
                            }
                            er.faces.push(fi);
                            lp.push(Use { edge: ix, rev: oe.forward != er.forward });
                            continue;
                        }
                        None => {
                            let (curve, off) = Edge3::from_step(oe).map_err(|e| format!("face #{}: {e}", bf.id))?;
                            rep.max_vertex_off = rep.max_vertex_off.max(off);
                            *rep.edges.entry(edge_kind(&curve)).or_default() += 1;
                            let vs = vert(oe.start_vertex, oe.start, &mut verts);
                            let ve = vert(oe.end_vertex, oe.end, &mut verts);
                            let (lo, hi) = curve.bbox();
                            edges.push(EdgeRec { id: oe.edge, curve, faces: vec![fi], v: [vs, ve], lo, hi, forward: oe.forward });
                            edge_ix.insert(oe.edge, edges.len() - 1);
                            edges.len() - 1
                        }
                    };
                    lp.push(Use { edge: ix, rev: false });
                }
                loops.push(lp);
            }
            faces.push(Face3 { id: bf.id, surf, sense: if bf.same_sense { 1.0 } else { -1.0 }, loops, points, seams, trim: std::sync::OnceLock::new(), quadric: None, trim2: None, analytic2: None });
        }
        lap("faces and edges");
        // B-spline faces that are quadrics of no STEP kind, face by face in parallel.
        {
            use rayon::prelude::*;
            faces.par_iter_mut().for_each(|f| {
                if let Surf::Nurbs(ns) = &f.surf {
                    f.quadric = crate::span_poly::recognize_quadric(ns);
                }
            });
        }
        lap("quadrics");
        // Lemon faces (the inner part of a self-intersecting torus): ISO 10303-42
        // (2021, degenerate_toroidal_surface, note 1) puts their normal out of the
        // lemon, opposite to Su x Sv, and some writers took Su x Sv (THE PART,
        // HOOPS Exchange 2025.8: with the standard's normal, 97 of 1,000 points
        // outside its box came out inside). Which one a file used shows at its
        // tangent edges: there the two faces' outward normals agree in a valid
        // solid, and are never nearly opposite (that would be a fold of no
        // thickness). A lemon whose nearly opposite neighbors outnumber its nearly
        // equal ones is turned over; one with neither keeps the standard's sense.
        for fi in 0..faces.len() {
            if !matches!(faces[fi].surf, Surf::Torus { major, .. } if major < 0.0) {
                continue;
            }
            rep.lemons += 1;
            let (mut opposite, mut equal) = (0, 0);
            for u in faces[fi].loops.iter().flatten() {
                let e = &edges[u.edge];
                let x = e.curve.at(0.5);
                for &g in &e.faces {
                    if g == fi || matches!(faces[g].surf, Surf::Torus { major, .. } if major < 0.0) {
                        continue;
                    }
                    let dot = faces[fi].normal(x).dot(faces[g].normal(x));
                    if dot < -0.999 {
                        opposite += 1;
                    } else if dot > 0.999 {
                        equal += 1;
                    }
                }
            }
            if opposite > equal {
                faces[fi].sense = -faces[fi].sense;
                rep.lemons_turned += 1;
            } else if opposite == 0 && equal == 0 {
                rep.lemons_undecided += 1;
            }
        }
        if let Some(e) = edges.iter().find(|e| e.faces.len() != 2) {
            return Err(format!("edge #{} is used by {} faces (not a closed manifold shell)", e.id, e.faces.len()));
        }
        {
            use rayon::prelude::*;
            let off = edges
                .par_iter()
                .map(|e| e.faces.iter().flat_map(|&fi| [0.0, 0.5, 1.0].map(|s| faces[fi].surf.dist(e.curve.at(s)))).fold(0.0, f64::max))
                .reduce(|| 0.0, f64::max);
            rep.max_edge_off = rep.max_edge_off.max(off);
        }
        lap("edge offsets");
        for f in &faces {
            if let Surf::Cone { f: fr, .. } = &f.surf {
                let apex = f.surf.apex().unwrap();
                for lp in &f.loops {
                    for u in lp {
                        if (edges[u.edge].curve.at(0.5) - apex).dot(fr.z) < 0.0 {
                            let e = &edges[u.edge];
                            return Err(format!("face #{}: on the far nappe of its cone (edge #{} midpoint {:.3e} along the axis from the apex, ends {:?} {:?}, apex {:?})", f.id, e.id, (e.curve.at(0.5) - apex).dot(fr.z), e.curve.start(), e.curve.end(), apex));
                        }
                    }
                }
            }
        }
        lap("trims");
        {
            use rayon::prelude::*;
            let t = std::time::Instant::now();
            let built: Vec<(Result<Option<(crate::trim2::Trim2, f64, Option<crate::trim2::Analytic>)>, String>, (u64, f64))> = faces
                .par_iter()
                .map(|f| {
                    let tf = std::time::Instant::now();
                    crate::trim2::take_beyond();
                    let r = build_trim2(f, &edges, tol);
                    if trace && tf.elapsed().as_secs_f64() > 0.5 {
                        eprintln!("  trim2 {:>8.1} ms  face #{} ({}, {} loops, {} uses)", tf.elapsed().as_secs_f64() * 1e3, f.id, f.surf.name(), f.loops.len(), f.loops.iter().map(|l| l.len()).sum::<usize>());
                    }
                    (r, crate::trim2::take_beyond())
                })
                .collect();
            // Faces whose edges lie off their surface beyond the declared tolerance,
            // the worst first in the notes.
            let mut off: Vec<(f64, u64, usize)> = Vec::new();
            for (fi, (_, (n, far))) in built.iter().enumerate() {
                if *n > 0 {
                    off.push((*far, *n, fi));
                    rep.off_tol_faces += 1;
                    rep.off_tol_points += n;
                    rep.off_tol_max = rep.off_tol_max.max(*far);
                }
            }
            off.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(far, n, fi) in off.iter().take(20) {
                rep.off_tol_notes.push(format!("face #{} ({}): {n} edge points up to {far:.3e} mm off its surface, beyond the file's {tol:.1e} mm", faces[fi].id, faces[fi].surf.name()));
            }
            for (f, (r, _)) in faces.iter_mut().zip(built) {
                match r {
                    Ok(t2) => {
                        if let Some((t2, gap, an)) = t2 {
                            rep.trim2_bound = rep.trim2_bound.max(t2.bound);
                            rep.vertex_gap = rep.vertex_gap.max(gap);
                            if gap > tol {
                                rep.faces_over_tol += 1;
                                if rep.gap_notes.len() < 20 {
                                    rep.gap_notes.push(format!("face #{} ({}): consecutive edges miss each other by {gap:.3e} mm", f.id, f.surf.name()));
                                }
                            }
                            f.trim2 = Some(t2);
                            f.analytic2 = an.map(std::sync::Arc::new);
                        }
                    }
                    Err(e) => {
                        rep.trim2_refused += 1;
                        rep.trim2_notes.push(format!("face #{}: {e}", f.id));
                    }
                }
            }
            rep.trim2_secs = t.elapsed().as_secs_f64();
        }
        if trace {
            let c = crate::trim2::CLOCK.each_ref().map(|x| x.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9);
            eprintln!("  trim2 totals over all faces (thread seconds): builds {:.1}, preimages {:.1}, certifying {:.1}, placing {:.1}", c[3], c[0], c[1], c[2]);
            let n = crate::trim2::PREIMAGES.each_ref().map(|x| x.load(std::sync::atomic::Ordering::Relaxed));
            eprintln!("  trim2 preimages: closed form {}, Newton from the neighbor {}, whole-surface search {} ({:.1} s; no neighbor {}, Newton unsettled {}, beyond the tolerance {})", n[0], n[1], n[2], n[3] as f64 * 1e-9, n[4], n[5], n[6]);
        }
        lap("trim loops in the parameter domain");
        // Angle-weighted pseudo-normals. A face whose surface collapses to a point
        // at the vertex (a pole, a cone's apex) has no normal there: its fan of
        // limiting normals around the point is weighted by angle instead.
        let use_tan = |u: &Use, s: f64| if u.rev { -edges[u.edge].curve.tangent_at(1.0 - s) } else { edges[u.edge].curve.tangent_at(s) };
        let use_end = |u: &Use| if u.rev { edges[u.edge].v[0] } else { edges[u.edge].v[1] };
        let mut pn = vec![V3::default(); verts.len()];
        let mut fanned: Vec<(usize, usize)> = Vec::new();
        for (fi, f) in faces.iter().enumerate() {
            let (Some(ns), Some(t)) = (face_nurbs(f), &f.trim2) else { continue };
            for (k, value, x) in ns.collapsed_sides() {
                for (vi, v) in verts.iter().enumerate() {
                    if (v.p - x).len() <= tol && f.loops.iter().flatten().any(|u| edges[u.edge].v.contains(&vi)) {
                        pn[vi] = pn[vi] + fan_normal(ns, &|u, w| t.inside(u, w), k, value) * f.sense;
                        fanned.push((fi, vi));
                    }
                }
            }
        }
        for (fi, f) in faces.iter().enumerate() {
            for lp in &f.loops {
                for i in 0..lp.len() {
                    let (a, b) = (&lp[i], &lp[(i + 1) % lp.len()]);
                    let v = use_end(a);
                    if fanned.contains(&(fi, v)) {
                        continue;
                    }
                    let n = f.normal(verts[v].p);
                    let proj = |t: V3| (t - n * n.dot(t)).unit();
                    let (ti, to) = (proj(use_tan(a, 1.0)), proj(use_tan(b, 0.0)));
                    let turn = n.dot(ti.cross(to)).atan2(ti.dot(to));
                    pn[v] = pn[v] + n * (PI - turn);
                }
            }
        }
        for (v, p) in verts.iter_mut().zip(pn) {
            v.pn = p;
        }
        lap("vertex normals");
        for (fi, p) in point_loops {
            let f: &Face3 = &faces[fi];
            let n = match &f.surf {
                Surf::Cone { f: fr, a, .. } => fr.z * (-a.sin() * f.sense),
                _ => f.normal(p),
            };
            verts.push(Vert { id: 0, p, pn: n * (2.0 * PI) });
        }
        let (mut lo, mut hi) = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
        for f in &faces {
            let (a, b) = face_box(f, &edges);
            lo = V3::new(lo.x.min(a.x), lo.y.min(a.y), lo.z.min(a.z));
            hi = V3::new(hi.x.max(b.x), hi.y.max(b.y), hi.z.max(b.z));
        }
        lap("boxes");
        Ok((Solid3 { id: solid.id, name: solid.name.clone(), faces, edges, verts, tol, lo, hi }, rep))
    }

    fn eps(&self) -> f64 {
        4.0 * self.tol + 1e-9
    }

    fn use_tangent(&self, u: &Use, x: &Cross) -> V3 {
        if u.rev {
            -x.tangent
        } else {
            x.tangent
        }
    }

    /// Walks one leg: the first crossing of the face boundary along it.
    fn walk(&self, fi: usize, leg: &Leg, first_leg: bool) -> Outcome {
        let f = &self.faces[fi];
        let eps = self.eps();
        let mut hits: Vec<(f64, Cross, Use)> = Vec::new();
        let mut buf = Vec::new();
        for lp in &f.loops {
            for u in lp {
                if f.seams.contains(&u.edge) {
                    continue;
                }
                // The cutting plane misses the edge's box: no crossing.
                let er = &self.edges[u.edge];
                let (cb, eb) = ((er.lo + er.hi) * 0.5, (er.hi - er.lo) * 0.5);
                if (leg.n.dot(cb) - leg.c).abs() > leg.n.x.abs() * eb.x + leg.n.y.abs() * eb.y + leg.n.z.abs() * eb.z + eps {
                    continue;
                }
                buf.clear();
                self.edges[u.edge].curve.plane_crossings(leg.n, leg.c, eps, &mut buf);
                for x in &buf {
                    if !leg.side.ok(x.q) {
                        continue;
                    }
                    // Only crossings strictly ahead of the start: one just behind it says
                    // nothing about the start's side.
                    let s = leg.param(x.q);
                    if s <= leg.s_end && s > 0.0 {
                        hits.push((s, *x, *u));
                    }
                }
            }
        }
        if hits.is_empty() {
            return if leg.exits { Outcome::Inside(false) } else { Outcome::NoHit };
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        if std::env::var("ICAM_TRACE_TRIM").is_ok() {
            for (s, x, u) in &hits {
                let m = f.normal(x.q).cross(self.use_tangent(u, x));
                eprintln!("    leg {:?}: hit s {s:.6} at {:?} edge {} rev {} vertex {} tangential {} dir.m {:.4}", leg.kind, x.q, u.edge, u.rev, x.at_vertex, x.tangential, leg.dir_at(x.q).dot(m.unit()));
            }
        }
        let (s, x, u) = hits[0];
        // Circle legs measure s in radians: compare arc length with eps.
        let arc = |s: f64| match leg.kind {
            Kind::Ray { .. } => s,
            Kind::Circle { c, .. } => s * (x.q - c).len(),
        };
        // The first crossing decides however close it is to the start, except that a
        // later leg starting on the boundary cannot tell which side its junction is on.
        if !first_leg && arc(s) < eps {
            return Outcome::Degenerate;
        }
        if x.at_vertex || x.tangential {
            return Outcome::Degenerate;
        }
        let m = f.normal(x.q).cross(self.use_tangent(&u, &x));
        let dm = leg.dir_at(x.q).dot(m.unit());
        if dm.abs() < 1e-7 {
            return Outcome::Degenerate;
        }
        Outcome::Inside(dm < 0.0)
    }

    /// Whether x lies within tol of an edge of face fi (seams left out).
    fn near_face_edge(&self, fi: usize, x: V3, tol: f64) -> bool {
        let f = &self.faces[fi];
        f.loops.iter().flatten().filter(|u| !f.seams.contains(&u.edge)).any(|u| {
            let e = &self.edges[u.edge];
            crate::surface3::box_dist(x, &(e.lo, e.hi)) <= tol && e.curve.closest(x).dist <= tol
        })
    }

    fn run(&self, fi: usize, legs: &[Leg]) -> Option<bool> {
        for (k, leg) in legs.iter().enumerate() {
            // A later leg that starts on an edge of the face cannot tell which side
            // its start is on: the crossing at the start is not ahead of it, so the
            // walk would report the next edge instead.
            if k > 0 && self.near_face_edge(fi, leg.start, 10.0 * self.eps()) {
                return None;
            }
            match self.walk(fi, leg, k == 0) {
                Outcome::Inside(b) => return Some(b),
                Outcome::Degenerate => return None,
                Outcome::NoHit => {}
            }
        }
        None
    }

    /// Boundary points of a face to aim closed-surface paths at.
    fn targets(&self, fi: usize) -> Vec<V3> {
        let f = &self.faces[fi];
        let mut t = Vec::new();
        for lp in &f.loops {
            for u in lp {
                if !f.seams.contains(&u.edge) {
                    for s in [0.5, 0.3, 0.7] {
                        t.push(self.edges[u.edge].curve.at(s));
                    }
                }
            }
        }
        t
    }

    /// Is the surface point q in face fi? None when every path was degenerate.
    pub fn classify(&self, fi: usize, q: V3) -> Option<bool> {
        let f = &self.faces[fi];
        match &f.surf {
            Surf::Plane { f: fr } => {
                for k in 0..12 {
                    let a = 0.3 + k as f64 * 0.52;
                    let d = fr.x * a.cos() + fr.y() * a.sin();
                    if let Some(b) = self.run(fi, &[ray(q, d, d.cross(fr.z).unit(), Side::Any)]) {
                        return Some(b);
                    }
                }
                None
            }
            Surf::Cylinder { f: fr, r } | Surf::Cone { f: fr, r, .. } => {
                let is_cone = matches!(f.surf, Surf::Cone { .. });
                let ax = axial(fr, q);
                let apex = f.surf.apex();
                // The generator through the point at angle th and height h, walked away from
                // the apex (cone) or both ways (cylinder).
                let generator = |start: V3, th: f64, dir: f64| {
                    let (s, c) = th.sin_cos();
                    let er = fr.dir(V3::new(c, s, 0.0));
                    let et = fr.dir(V3::new(-s, c, 0.0));
                    match (&f.surf, apex) {
                        (Surf::Cone { a, .. }, Some(ap)) => {
                            let g = (er * a.sin() + fr.z * a.cos()) * dir;
                            ray(start, g, et, Side::Nappe { o: fr.o, er, apex: ap, z: fr.z })
                        }
                        _ => ray(start, fr.z * dir, et, Side::Half { o: fr.o, er }),
                    }
                };
                let dirs: &[f64] = if is_cone { &[1.0] } else { &[1.0, -1.0] };
                for &dir in dirs {
                    if let Some(b) = self.run(fi, &[generator(q, ax.theta, dir)]) {
                        return Some(b);
                    }
                }
                // Around the latitude circle first, then along a generator.
                let c = fr.o + fr.z * ax.h;
                let rho = if is_cone { ax.rho.max(1e-300) } else { *r };
                for k in 1..12 {
                    let dth = k as f64 * 0.37 * if k % 2 == 0 { -1.0 } else { 1.0 };
                    let th = ax.theta + dth;
                    let start = c + fr.dir(V3::new(th.cos(), th.sin(), 0.0)) * rho;
                    let lat = circle(q, c, fr.x, fr.y(), dth.signum(), Side::Any, dth.abs());
                    for &dir in dirs {
                        if let Some(b) = self.run(fi, &[lat, generator(start, th, dir)]) {
                            return Some(b);
                        }
                    }
                }
                None
            }
            Surf::Sphere { f: fr, .. } => {
                let o = fr.o;
                let e1 = (q - o).unit();
                let ax = axial(fr, q);
                let mut legs: Vec<Leg> = Vec::new();
                for dir in [1.0, -1.0] {
                    legs.push(circle(q, o, ax.er, fr.z, dir, Side::Any, f64::INFINITY));
                    legs.push(circle(q, o + fr.z * ax.h, fr.x, fr.y(), dir, Side::Any, f64::INFINITY));
                }
                for t in self.targets(fi) {
                    let n = e1.cross(t - o);
                    if n.len() < 1e-9 * (t - o).len() {
                        continue;
                    }
                    let e2 = n.unit().cross(e1);
                    for dir in [1.0, -1.0] {
                        legs.push(circle(q, o, e1, e2, dir, Side::Any, f64::INFINITY));
                    }
                }
                legs.iter().find_map(|l| self.run(fi, std::slice::from_ref(l)))
            }
            Surf::Torus { f: fr, major, minor } => {
                let ax = axial(fr, q);
                let ct = fr.o + ax.er * *major;
                let outer = ax.rho > *major;
                for dir in [1.0, -1.0] {
                    let mer = circle(q, ct, ax.er, fr.z, dir, Side::Half { o: fr.o, er: ax.er }, f64::INFINITY);
                    if let Some(b) = self.run(fi, &[mer]) {
                        return Some(b);
                    }
                    let lat = circle(q, fr.o + fr.z * ax.h, fr.x, fr.y(), dir, Side::Ring { o: fr.o, z: fr.z, major: *major, outer }, f64::INFINITY);
                    if let Some(b) = self.run(fi, &[lat]) {
                        return Some(b);
                    }
                }
                // Along q's meridian to a target's tube angle, then around that latitude.
                let phi_q = (q - ct).dot(fr.z).atan2((q - ct).dot(ax.er));
                for t in self.targets(fi) {
                    let at = axial(fr, t);
                    let phi_t = at.h.atan2(at.rho - major);
                    if phi_t.cos().abs() < 0.05 {
                        continue;
                    }
                    let start = ct + (ax.er * phi_t.cos() + fr.z * phi_t.sin()) * *minor;
                    let ring = Side::Ring { o: fr.o, z: fr.z, major: *major, outer: phi_t.cos() > 0.0 };
                    for d1 in [1.0, -1.0] {
                        let mer = circle(q, ct, ax.er, fr.z, d1, Side::Half { o: fr.o, er: ax.er }, swept(phi_q, phi_t, d1));
                        for d2 in [1.0, -1.0] {
                            let lat = circle(start, fr.o + fr.z * (start - fr.o).dot(fr.z), fr.x, fr.y(), d2, ring, f64::INFINITY);
                            if let Some(b) = self.run(fi, &[mer, lat]) {
                                return Some(b);
                            }
                        }
                    }
                }
                None
            }
            Surf::Nurbs(ns) => {
                let h = ns.closest(q, f64::INFINITY)?;
                self.classify_nurbs(fi, h.u, h.v)
            }
        }
    }

    /// Exact distance from p to trimmed face fi, if below `cutoff`: its foot when
    /// the foot is in the face, else its nearest edge point. For the compiler: which
    /// faces come near a cell.
    pub fn face_distance(&self, fi: usize, p: V3, cutoff: f64) -> f64 {
        let f = &self.faces[fi];
        let mut best = cutoff;
        for u in f.loops.iter().flatten() {
            let e = &self.edges[u.edge];
            if crate::surface3::box_dist(p, &(e.lo, e.hi)) < best {
                best = best.min(e.curve.closest(p).dist);
            }
        }
        if let Surf::Nurbs(_) = f.surf {
            if let Some((d, _)) = self.nurbs_face_closest(fi, p, best) {
                best = best.min(d);
            }
        } else {
            let q = f.surf.foot(p);
            let d = (p - q).len();
            if d < best && is_foot(p, q, f.surf.normal(q)) && self.classify(fi, q) == Some(true) {
                best = d;
            }
        }
        best
    }

    /// The use of edge ei by face fi (the first, on a seam).
    pub fn use_of(&self, fi: usize, ei: usize) -> Option<Use> {
        self.faces[fi].loops.iter().flatten().find(|u| u.edge == ei).copied()
    }

    /// Geometry of edge ei at its point nearest p: the point b, the outward normals
    /// n1, n2 of its two faces there, and m1, the direction into face 1 across the
    /// edge (perpendicular to it, tangent to face 1).
    pub fn edge_frame(&self, ei: usize, p: V3) -> (V3, V3, V3, V3) {
        let e = &self.edges[ei];
        let c = e.curve.closest(p);
        let (f1, f2) = (e.faces[0], e.faces[1]);
        let rev = self.use_of(f1, ei).map_or(false, |u| u.rev);
        let t = if rev { -c.tangent } else { c.tangent };
        let n1 = self.faces[f1].normal(c.q);
        let n2 = self.faces[f2].normal(c.q);
        (c.q, n1, n2, n1.cross(t).unit())
    }

    /// Exact signed distance (negative inside) with the nearest point and feature.
    pub fn sdf(&self, p: V3) -> Sd {
        let (f, e, v): (Vec<usize>, Vec<usize>, Vec<usize>) = ((0..self.faces.len()).collect(), (0..self.edges.len()).collect(), (0..self.verts.len()).collect());
        self.sdf_among(p, &f, &e, &v)
    }

    /// The signed distance to the given faces, edges and vertices only (the
    /// solid's, when the nearest feature is among them).
    pub fn sdf_among(&self, p: V3, faces: &[usize], edges: &[usize], verts: &[usize]) -> Sd {
        let mut best = Sd { d: f64::INFINITY, q: p, feature: Feature::Vertex(0), unsure: false };
        let mut best_abs = f64::INFINITY;
        let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
        for &vi in verts {
            let v = &self.verts[vi];
            let d = (p - v.p).len();
            if d < best_abs {
                best_abs = d;
                best = Sd { d: d * sign((p - v.p).dot(v.pn)), q: v.p, feature: Feature::Vertex(vi), unsure: false };
            }
        }
        for &ei in edges {
            let e = &self.edges[ei];
            let bd = V3::new((e.lo.x - p.x).max(p.x - e.hi.x).max(0.0), (e.lo.y - p.y).max(p.y - e.hi.y).max(0.0), (e.lo.z - p.z).max(p.z - e.hi.z).max(0.0)).len();
            if bd >= best_abs {
                continue;
            }
            let c = e.curve.closest(p);
            if c.near != Near::Interior || c.dist >= best_abs {
                continue;
            }
            let pn = e.faces.iter().fold(V3::default(), |a, &fi| a + self.faces[fi].normal(c.q));
            best_abs = c.dist;
            best = Sd { d: c.dist * sign((p - c.q).dot(pn)), q: c.q, feature: Feature::Edge(ei), unsure: false };
        }
        for &fi in faces {
            if matches!(self.faces[fi].surf, Surf::Nurbs(_)) {
                if let Some((d, q)) = self.nurbs_face_closest(fi, p, best_abs) {
                    best_abs = d;
                    let n = self.faces[fi].normal(q);
                    best = Sd { d: d * sign((p - q).dot(n)), q, feature: Feature::Face(fi), unsure: false };
                }
            }
        }
        let mut feet: Vec<(f64, usize, V3)> = faces
            .iter()
            .filter(|&&fi| !matches!(self.faces[fi].surf, Surf::Nurbs(_)))
            .map(|&fi| {
                let q = self.faces[fi].surf.foot(p);
                ((p - q).len(), fi, q)
            })
            .filter(|x| x.0 < best_abs)
            .collect();
        feet.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut unsure = false;
        for (d, fi, q) in feet {
            if !is_foot(p, q, self.faces[fi].surf.normal(q)) {
                continue;
            }
            match self.classify(fi, q) {
                Some(true) => {
                    best = Sd { d: d * sign((p - q).dot(self.faces[fi].normal(q))), q, feature: Feature::Face(fi), unsure };
                    return best;
                }
                Some(false) => {}
                None => unsure = true,
            }
        }
        best.unsure = unsure;
        best
    }
}


/// A box holding a face (not tight): a planar face lies in the hull of its
/// edges and vertex loops; a cylinder or cone face within its full circles over
/// the axial range of its edges and vertex loops (a cone's apex); a sphere or torus face within the whole surface; a B-spline face
/// within its control points.
fn face_box(f: &Face3, edges: &[EdgeRec]) -> (V3, V3) {
    let mut lo = V3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = V3::new(f64::MIN, f64::MIN, f64::MIN);
    let mut add = |q: V3| {
        lo = V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z));
        hi = V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z));
    };
    let axes = [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)];
    // Box of a circle of radius r about c with axis z.
    let circle_box = |c: V3, z: V3, r: f64, add: &mut dyn FnMut(V3)| {
        for ax in axes {
            let e = (1.0 - z.dot(ax).powi(2)).max(0.0).sqrt() * r;
            add(c + ax * e);
            add(c - ax * e);
        }
    };
    let edge_boxes = |add: &mut dyn FnMut(V3)| {
        for u in f.loops.iter().flatten() {
            add(edges[u.edge].lo);
            add(edges[u.edge].hi);
        }
        for &p in &f.points {
            add(p);
        }
    };
    match &f.surf {
        Surf::Plane { .. } => edge_boxes(&mut add),
        Surf::Cylinder { f: fr, r } | Surf::Cone { f: fr, r, .. } => {
            let (mut h0, mut h1) = (f64::MAX, f64::MIN);
            for &p in &f.points {
                let h = (p - fr.o).dot(fr.z);
                h0 = h0.min(h);
                h1 = h1.max(h);
            }
            for u in f.loops.iter().flatten() {
                let (a, b) = (edges[u.edge].lo, edges[u.edge].hi);
                for k in 0..8 {
                    let q = V3::new(if k & 1 == 0 { a.x } else { b.x }, if k & 2 == 0 { a.y } else { b.y }, if k & 4 == 0 { a.z } else { b.z });
                    let h = (q - fr.o).dot(fr.z);
                    h0 = h0.min(h);
                    h1 = h1.max(h);
                }
            }
            for h in [h0, h1] {
                let rad = match &f.surf {
                    Surf::Cone { a, .. } => (r + h * a.tan()).abs(),
                    _ => *r,
                };
                circle_box(fr.o + fr.z * h, fr.z, rad, &mut add);
            }
        }
        Surf::Sphere { f: fr, r } => {
            add(fr.o - V3::new(*r, *r, *r));
            add(fr.o + V3::new(*r, *r, *r));
        }
        Surf::Torus { f: fr, major, minor } => {
            for h in [-minor, *minor] {
                circle_box(fr.o + fr.z * h, fr.z, major + minor, &mut add);
            }
        }
        Surf::Nurbs(ns) => {
            let (a, b) = ns.bbox();
            add(a);
            add(b);
        }
    }
    (lo, hi)
}

/// A face foot counts only as an interior critical point of the distance, with
/// p - q along the surface normal: then its normal gives the sign. A closest point
/// on the border of a surface's domain (or a cone's apex) is not one; the edge or
/// vertex there is the nearest feature and carries the sign.
pub(crate) fn is_foot(p: V3, q: V3, n: V3) -> bool {
    let d = p - q;
    let l = d.len();
    l < 1e-9 || d.cross(n).len() <= 1e-5 * l
}

// ---------------------------------------------------------------- B-spline faces

/// Count of B-spline trim tests (for profiling).
pub static TRIM_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Solid3 {
    /// Is the point at parameters (u, v) of B-spline face fi in the face? Walks
    /// the iso-curves from (u, v) to the surface's boundary (exact rational Bezier
    /// pieces), finds where the face's edges pass within the trim threshold of the
    /// walk (certified box branch and bound, Newton), and reads the first crossing:
    /// inside when the walk leaves the face there. No crossing: outside (the walk
    /// ran off the surface, and the face lies on the surface).
    /// Whether the surface point at (u, v) of B-spline face fi lies in the face, by
    /// its certified parameter-domain loops; None when the face has none.
    pub fn classify_nurbs2(&self, fi: usize, u: f64, v: f64) -> Option<bool> {
        self.faces[fi].trim2.as_ref().map(|t| t.inside(u, v))
    }

    /// Whether the surface point nearest q lies in face fi, by its certified loops:
    /// the parameters in closed form for an analytic face, by the certified search
    /// for a B-spline face. None when the face has no loops.
    pub fn classify2(&self, fi: usize, q: V3) -> Option<bool> {
        let f = &self.faces[fi];
        let t = f.trim2.as_ref()?;
        let (u, v) = match (&f.surf, &f.analytic2) {
            (_, Some(a)) => a.uv(q),
            (Surf::Nurbs(ns), None) => {
                let h = ns.closest(q, f64::INFINITY)?;
                (h.u, h.v)
            }
            _ => return None,
        };
        Some(t.inside(u, v))
    }

    pub fn classify_nurbs(&self, fi: usize, u: f64, v: f64) -> Option<bool> {
        TRIM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let f = &self.faces[fi];
        let (Surf::Nurbs(ns), Some(tr)) = (&f.surf, self.old_trim(fi)) else { return None };
        let eps = self.eps().max(10.0 * tr.thr);
        // Shortest walks first: toward the nearest side of the parameter domain. A
        // walk along a closed direction wraps around the seam, so it costs a full
        // turn at most; it goes after the open walks unless they are long.
        let (un, vn) = ((u - ns.u_range.0) / (ns.u_range.1 - ns.u_range.0), (v - ns.v_range.0) / (ns.v_range.1 - ns.v_range.0));
        let cost = |closed: bool, x: f64| if closed { 0.75 } else { x };
        let mut dirs = [(true, 1.0, cost(tr.closed_u, 1.0 - un)), (false, 1.0, cost(tr.closed_v, 1.0 - vn)), (true, -1.0, cost(tr.closed_u, un)), (false, -1.0, cost(tr.closed_v, vn))];
        dirs.sort_by(|a, b| a.2.total_cmp(&b.2));
        'paths: for (along_u, sg, _) in dirs {
            let (from, fixed, range) = if along_u { (u, v, ns.u_range) } else { (v, u, ns.v_range) };
            let closed = if along_u { tr.closed_u } else { tr.closed_v };
            let period = range.1 - range.0;
            let to = if sg > 0.0 { range.1 } else { range.0 };
            if (to - from).abs() < 1e-12 && !closed {
                continue;
            }
            // How far along the walk a hit at parameter ta lies (wrapping on a closed
            // direction).
            let ahead = |ta: f64| {
                let d = (ta - from) * sg;
                if closed {
                    d.rem_euclid(period)
                } else {
                    d
                }
            };
            // Walk span by span; the first crossing lies in the first span with a hit
            // ahead of the start (one more span is read for reports on its border).
            // On a closed direction the walk continues from the seam back to the start.
            let mut hits: Vec<(NearHit, Use)> = Vec::new();
            let (mut grazed, mut last_span) = (false, false);
            let legs: Vec<(f64, f64)> = if closed { vec![(from, to), (if sg > 0.0 { range.0 } else { range.1 }, from)] } else { vec![(from, to)] };
            for (leg_from, leg_to) in legs {
                if (leg_to - leg_from).abs() < 1e-12 || last_span {
                    continue;
                }
                ns.iso_walk(along_u, fixed, leg_from, leg_to, &mut |z| {
                    for a in flatten(std::slice::from_ref(&z)) {
                        let mut near: Vec<usize> = Vec::new();
                        tr.tree.visit(&bez_box(&a), tr.thr, &mut |i| near.push(i));
                        for i in near {
                            let (us, b) = &tr.pieces[i];
                            let mut out = Vec::new();
                            let grazing = near_hits(&a, b, tr.thr, 0, &mut out);
                            // A graze ahead of the start spoils this walk; behind it, it is irrelevant.
                            if grazing && out.iter().any(|h| ahead(h.ta) > 0.0) {
                                grazed = true;
                                return false;
                            }
                            hits.extend(out.into_iter().map(|h| (h, *us)));
                        }
                    }
                    if last_span {
                        return false;
                    }
                    last_span = hits.iter().any(|(h, _)| ahead(h.ta) > 1e-12 * period);
                    true
                });
                if grazed {
                    continue 'paths;
                }
            }
            hits.retain(|(h, _)| ahead(h.ta) > 1e-12 * period);
            if hits.is_empty() {
                if closed {
                    // A full turn with no crossing: the iso-loop is wholly in or out of
                    // the face, which this walk cannot tell.
                    continue 'paths;
                }
                // The walk reached the domain's side: it left the face there when an
                // edge of the face runs along that side at the arrival point.
                let end = if along_u { ns.eval(to, v).p } else { ns.eval(u, to).p };
                return Some(tr.sides.iter().any(|us| self.edges[us.edge].curve.closest(end).dist <= 4.0 * tr.thr));
            }
            hits.sort_by(|a, b| ahead(a.0.ta).total_cmp(&ahead(b.0.ta)));
            // One crossing can be reported by neighboring pieces: take the closest
            // report among those within a small radius of the first.
            let cr = (100.0 * tr.thr).max(1e-6);
            let first = hits[0].0.q;
            let cluster: Vec<&(NearHit, Use)> = hits.iter().filter(|(h, _)| (h.q - first).len() < cr).collect();
            if cluster.iter().any(|(_, us)| us.edge != cluster[0].1.edge) {
                continue 'paths;
            }
            let (h, us) = cluster.iter().min_by(|a, b| a.0.dist.total_cmp(&b.0.dist)).unwrap();
            let e = &self.edges[us.edge].curve;
            if (h.q - e.start()).len() < eps || (h.q - e.end()).len() < eps || h.tan_a.cross(h.tan_b).len() < 1e-6 {
                continue 'paths;
            }
            let (pu, pv) = if along_u { (h.ta, v) } else { (u, h.ta) };
            let se = ns.eval(pu, pv);
            let n = se.ps.cross(se.pt).unit() * f.sense;
            let t = if us.rev { -h.tan_b } else { h.tan_b };
            let dm = (h.tan_a * sg).dot(n.cross(t).unit());
            if dm.abs() < 1e-7 {
                continue 'paths;
            }
            return Some(dm < 0.0);
        }
        None
    }

    /// The old trim test's data for face fi (B-spline faces), built on first use.
    pub fn old_trim(&self, fi: usize) -> Option<&NurbsTrim> {
        self.faces[fi].trim.get_or_init(|| old_trim(&self.faces[fi], &self.edges, self.tol)).as_ref()
    }

    /// Closest point of trimmed B-spline face fi to p, if closer than `cutoff`.
    /// Branch and bound over the surface's patches: a patch that no edge of the
    /// face comes near lies wholly inside or wholly outside the face, so one trim
    /// test settles it; others are split, and their Newton points tested one by one.
    pub fn nurbs_face_closest(&self, fi: usize, p: V3, cutoff: f64) -> Option<(f64, V3)> {
        let f = &self.faces[fi];
        let (Surf::Nurbs(ns), Some(tr)) = (&f.surf, self.old_trim(fi)) else { return None };
        if tr.pieces.is_empty() {
            // The face is the whole surface.
            return ns.closest(p, cutoff).filter(|h| is_foot(p, h.q, h.n)).map(|h| (h.dist, h.q));
        }
        let mut best: Option<(f64, V3)> = None;
        let mut best_d = cutoff;
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(PatchItem { lb: box_dist(p, &ns.bbox()), entry: Entry::Node(0), depth: 0, inside: false });
        let mut pops = 0;
        while let Some(PatchItem { lb, entry, depth, mut inside }) = heap.pop() {
            if lb >= best_d - DIST_EPS || best_d <= DIST_EPS {
                break;
            }
            pops += 1;
            if pops > 50_000 {
                break;
            }
            // A region that no edge of the face comes near is wholly in or out of it.
            let region = match &entry {
                Entry::Node(id) => {
                    let n = &ns.nodes[*id as usize];
                    let (u0, u1, v0, v1) = ns.node_params(n);
                    ((n.lo, n.hi), 0.5 * (u0 + u1), 0.5 * (v0 + v1))
                }
                Entry::Patch(pa) => (pa.bbox(), 0.5 * (pa.u0 + pa.u1), 0.5 * (pa.v0 + pa.v1)),
            };
            if !inside && !tr.tree.any_within(&region.0, tr.thr) {
                match self.classify_nurbs(fi, region.1, region.2) {
                    Some(false) => continue,
                    Some(true) => inside = true,
                    None => {}
                }
            }
            let patch = match entry {
                Entry::Node(id) => {
                    let n = &ns.nodes[id as usize];
                    if n.kids.is_empty() {
                        let pa = ns.patch(n.i0 as usize, n.j0 as usize);
                        let lbp = box_dist(p, &pa.bbox()).max(lb);
                        if lbp < best_d - DIST_EPS {
                            heap.push(PatchItem { lb: lbp, entry: Entry::Patch(pa), depth: 0, inside });
                        }
                    } else {
                        for &k in &n.kids {
                            let c = &ns.nodes[k as usize];
                            let lbk = box_dist(p, &(c.lo, c.hi));
                            if lbk < best_d - DIST_EPS {
                                heap.push(PatchItem { lb: lbk, entry: Entry::Node(k), depth: 0, inside });
                            }
                        }
                    }
                    continue;
                }
                Entry::Patch(pa) => pa,
            };
            let (s, t) = patch.newton(p, 0.5, 0.5);
            let e = patch.eval(s, t);
            let d = (e.p - p).len();
            if d < best_d {
                let (pu, pv) = (patch.u0 + s * (patch.u1 - patch.u0), patch.v0 + t * (patch.v1 - patch.v0));
                if inside || self.classify_nurbs(fi, pu, pv) == Some(true) {
                    // Any point of the face bounds the search; only a foot is reported.
                    best_d = d;
                    if is_foot(p, e.p, e.ps.cross(e.pt).unit()) {
                        best = Some((d, e.p));
                    }
                }
            }
            let slab = slab_bound(&patch, p, if d > 1e-12 { (p - e.p) * (1.0 / d) } else { e.ps.cross(e.pt).unit() });
            if slab >= best_d - DIST_EPS || depth >= 30 {
                continue;
            }
            for q in patch.quarters() {
                let lbq = box_dist(p, &q.bbox()).max(slab);
                if lbq < best_d - DIST_EPS {
                    heap.push(PatchItem { lb: lbq, entry: Entry::Patch(q), depth: depth + 1, inside });
                }
            }
        }
        best
    }

}

// ---------------------------------------------------------------- exact evaluator

/// The solid's exact signed distance from its certified face loops (`trim2`).
///
/// The nearest boundary point of p is a critical point of the distance inside a
/// face, an interior point of an edge, or a vertex; the features are visited best
/// first through one box tree, each against the nearest found so far. An analytic
/// surface's distance has one local minimum on the whole surface, its closed-form
/// foot, so a face contributes its foot when the foot lies in the face by its
/// loops, and nothing otherwise (its nearest point is then on its boundary). A
/// B-spline face contributes the critical points its trimmed search finds inside
/// it. A face point that is not a critical point only bounds that search: it
/// stands for the boundary, where the edges answer.
///
/// Signs come from the nearest feature: the outward normal at a face's critical
/// point, the two faces' normals summed at an edge point, the angle-weighted
/// pseudo-normal at a vertex. A face without certified loops falls back to the old
/// trim test and marks the answer unsure.
pub struct Exact<'a> {
    pub s: &'a Solid3,
    /// Faces, then edges, then vertices, then C0 lines, then B-spline poles.
    tree: BoxTree,
    creases: Vec<Option<Creases>>,
    /// The interior C0 knot lines of B-spline faces, searched like edges.
    lines: Vec<C0Line>,
    /// Poles of B-spline faces (sides collapsed on to one point) in the face away
    /// from every vertex: face, point, outward pseudo-normal.
    poles: Vec<(usize, V3, V3)>,
    /// Points an analytic face's surface is singular at (a cone's apex, a lemon or
    /// apple torus's tips) that lie in the face away from every vertex, with the
    /// outward pseudo-normal there.
    tips: Vec<Vec<(V3, V3)>>,
}

/// An interior C0 knot line of a B-spline face: the direction held fixed (0: u),
/// its value, and its pieces with the patches before and after it.
struct C0Line {
    face: usize,
    k: usize,
    value: f64,
    pieces: Vec<(crate::curve3::Bez3, crate::surface3::Patch, crate::surface3::Patch)>,
}

/// What one query of `Exact` did.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactStats {
    /// Steps of the trimmed B-spline searches.
    pub pops: usize,
    /// A trimmed search stopped at its step cap.
    pub capped: bool,
    /// A face without certified loops was looked at.
    pub uncertified: bool,
    /// The nearest point is on an edge where its two faces touch tangentially with
    /// opposite normals (a cusp): their normals cancel, and the edge gives no sign.
    pub cusp: bool,
    /// Faces, edges and vertices looked at.
    pub features: usize,
}

impl<'a> Exact<'a> {
    pub fn new(s: &'a Solid3) -> Exact<'a> {
        Self::with_test(s, true)
    }

    /// `test`: use the critical-point test in the B-spline searches (off only to
    /// measure what it saves).
    pub fn with_test(s: &'a Solid3, test: bool) -> Exact<'a> {
        let mut boxes: Vec<(V3, V3)> = s.faces.iter().map(|f| face_box(f, &s.edges)).collect();
        boxes.extend(s.edges.iter().map(|e| (e.lo, e.hi)));
        boxes.extend(s.verts.iter().map(|v| (v.p, v.p)));
        let creases = s
            .faces
            .iter()
            .map(|f| match &f.surf {
                Surf::Nurbs(ns) => {
                    let mut c = ns.creases();
                    c.test = test;
                    Some(c)
                }
                _ => None,
            })
            .collect();
        let mut lines = Vec::new();
        let mut poles = Vec::new();
        for (fi, f) in s.faces.iter().enumerate() {
            let (Surf::Nurbs(ns), Some(t)) = (&f.surf, &f.trim2) else { continue };
            for (k, value, pieces) in ns.c0_lines() {
                lines.push(C0Line { face: fi, k, value, pieces });
            }
            for (k, value, x) in ns.collapsed_sides() {
                if s.verts.iter().any(|v| (v.p - x).len() <= s.tol) {
                    continue;
                }
                let other = if k == 0 { ns.v_range } else { ns.u_range };
                let at = |w: f64| if k == 0 { (value, w) } else { (w, value) };
                if !t.inside(at(0.5 * (other.0 + other.1)).0, at(0.5 * (other.0 + other.1)).1) {
                    continue;
                }
                poles.push((fi, x, fan_normal(ns, &|u, w| t.inside(u, w), k, value) * f.sense));
            }
        }
        let tips = s
            .faces
            .iter()
            .map(|f| {
                let (Some(t), Some(a)) = (&f.trim2, &f.analytic2) else { return Vec::new() };
                singular_points(&f.surf, f.sense)
                    .into_iter()
                    .filter(|(x, _)| s.verts.iter().all(|v| (v.p - *x).len() > s.tol))
                    .filter(|(x, _)| {
                        let (u, v) = a.uv(*x);
                        t.inside(u, v)
                    })
                    .collect()
            })
            .collect();
        for l in &lines {
            let mut b = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
            for (z, _, _) in &l.pieces {
                let (lo, hi) = crate::curve3::bez_box(z);
                b = (V3::new(b.0.x.min(lo.x), b.0.y.min(lo.y), b.0.z.min(lo.z)), V3::new(b.1.x.max(hi.x), b.1.y.max(hi.y), b.1.z.max(hi.z)));
            }
            boxes.push(b);
        }
        boxes.extend(poles.iter().map(|(_, x, _)| (*x, *x)));
        Exact { s, tree: BoxTree::new(boxes), creases, lines, poles, tips }
    }

    /// The features that can be nearest to some point x within `r` of `c`, given that the
    /// boundary comes within `d` of c, by tight lower bounds from c: a feature F is
    /// nearest at x only when its distance from x is at most |d(x)| <= d + r, and it is at
    /// least F's lower bound from c less r, so F is kept when that bound is at most d + 2 r.
    /// The bounds: an analytic face's untrimmed surface (closed form), a line or arc
    /// edge's and a vertex's exact distance, a B-spline face's or other edge's box. Ids are
    /// the tree's: faces, then edges, vertices, knot lines, poles.
    pub fn candidates(&self, c: V3, r: f64, d: f64) -> Vec<usize> {
        let s = self.s;
        let (nf, ne, nv, nl) = (s.faces.len(), s.edges.len(), s.verts.len(), self.lines.len());
        let reach = d + 2.0 * r;
        let mut out = Vec::new();
        self.tree.visit(&(c, c), reach, &mut |id| {
            let lb = if id < nf {
                let b = box_dist(c, &self.tree.boxes_of(id));
                match &s.faces[id].surf {
                    Surf::Nurbs(_) => b,
                    surf => (c - surf.foot(c)).len().max(b),
                }
            } else if id < nf + ne {
                match &s.edges[id - nf].curve {
                    e @ (Edge3::Line { .. } | Edge3::Arc { .. }) => e.closest(c).dist,
                    _ => 0.0,
                }
            } else if id < nf + ne + nv {
                (c - s.verts[id - nf - ne].p).len()
            } else if id < nf + ne + nv + nl {
                0.0
            } else {
                (c - self.poles[id - nf - ne - nv - nl].1).len()
            };
            if lb <= reach {
                out.push(id);
            }
        });
        out
    }

    /// The GPU kernel that reads this evaluator's packing (`gpu_pack`).
    pub fn gpu_wgsl() -> &'static str {
        gpu::WGSL
    }

    /// The tree's ids: faces from 0, then edges, vertices, knot lines and poles, from
    /// these starts.
    pub fn id_starts(&self) -> [usize; 5] {
        let s = self.s;
        let (nf, ne, nv, nl) = (s.faces.len(), s.edges.len(), s.verts.len(), self.lines.len());
        [0, nf, nf + ne, nf + ne + nv, nf + ne + nv + nl]
    }

    /// The kind of a tree id (`near_kinds`'s order).
    pub fn kind_of(&self, id: usize) -> usize {
        let s = self.s;
        let (nf, ne, nv) = (s.faces.len(), s.edges.len(), s.verts.len());
        if id < nf {
            if matches!(s.faces[id].surf, Surf::Nurbs(_)) { 1 } else { 0 }
        } else if id < nf + ne {
            if matches!(s.edges[id - nf].curve, Edge3::Line { .. } | Edge3::Arc { .. }) { 2 } else { 3 }
        } else if id < nf + ne + nv {
            4
        } else if id < nf + ne + nv + self.lines.len() {
            3
        } else {
            4
        }
    }

    /// The features that can be nearest to some point within `r` of `c`, given that the
    /// boundary comes within `d` of c: those whose boxes come within d + 2 r of c (a
    /// point within r of c has its nearest boundary point within d + r of itself). By
    /// kind: analytic faces, B-spline faces, edges on lines or arcs, other edges, vertices
    /// (and the knot lines and poles of B-spline faces among the other edges and vertices).
    pub fn near_kinds(&self, c: V3, r: f64, d: f64) -> [usize; 5] {
        let s = self.s;
        let (nf, ne, nv) = (s.faces.len(), s.edges.len(), s.verts.len());
        let reach = d + 2.0 * r;
        let mut k = [0usize; 5];
        self.tree.visit(&(c, c), reach, &mut |id| {
            let i = if id < nf {
                if matches!(s.faces[id].surf, Surf::Nurbs(_)) { 1 } else { 0 }
            } else if id < nf + ne {
                if matches!(s.edges[id - nf].curve, Edge3::Line { .. } | Edge3::Arc { .. }) { 2 } else { 3 }
            } else if id < nf + ne + nv {
                4
            } else if id < nf + ne + nv + self.lines.len() {
                3
            } else {
                4
            };
            k[i] += 1;
        });
        k
    }

    /// Signed distance (negative inside) with the nearest point and feature.
    pub fn sdf(&self, p: V3) -> (Sd, ExactStats) {
        self.sdf_within(p, f64::INFINITY)
    }

    /// The same, given that the boundary comes within `bound` of p (a neighbor's
    /// distance plus how far it is, by the 1-Lipschitz bound): features farther than
    /// that are never looked at. If none is nearer after all, the search is redone
    /// without the bound.
    pub fn sdf_within(&self, p: V3, bound: f64) -> (Sd, ExactStats) {
        let s = self.s;
        let (nf, ne, nv, nl) = (s.faces.len(), s.edges.len(), s.verts.len(), self.lines.len());
        let mut st = ExactStats::default();
        let mut best = Sd { d: bound, q: p, feature: Feature::Vertex(0), unsure: false };
        let mut found = false;
        let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
        // Whether the current answer is a cusp edge's (reset by any nearer answer).
        let mut cusp = false;
        self.tree.nearest_first(p, bound, &mut |id| {
            st.features += 1;
            let cur = best.d.abs();
            let before = best.d;
            if id < nf {
                if let Some((d, q, n)) = self.face(id, p, cur, &mut st) {
                    best = Sd { d: d * sign((p - q).dot(n)), q, feature: Feature::Face(id), unsure: false };
                }
            } else if id < nf + ne {
                let ei = id - nf;
                let e = &s.edges[ei];
                if let Some(c) = e.curve.closest_below(p, cur).filter(|c| c.near == Near::Interior) {
                    let pn = e.faces.iter().fold(V3::default(), |a, &fi| a + self.normal_at(fi, c.q));
                    best = Sd { d: c.dist * sign((p - c.q).dot(pn)), q: c.q, feature: Feature::Edge(ei), unsure: false };
                    cusp = pn.len() < 1e-9;
                }
            } else if id < nf + ne + nv {
                let vi = id - nf - ne;
                let v = &s.verts[vi];
                let d = (p - v.p).len();
                if d < cur {
                    best = Sd { d: d * sign((p - v.p).dot(v.pn)), q: v.p, feature: Feature::Vertex(vi), unsure: false };
                }
            } else if id < nf + ne + nv + nl {
                let l = &self.lines[id - nf - ne - nv];
                if let Some((d, q, n)) = self.c0_line(l, p, cur, &mut st) {
                    best = Sd { d: d * sign((p - q).dot(n)), q, feature: Feature::Face(l.face), unsure: false };
                }
            } else {
                let (fi, x, n) = self.poles[id - nf - ne - nv - nl];
                let d = (p - x).len();
                if d < cur {
                    best = Sd { d: d * sign((p - x).dot(n)), q: x, feature: Feature::Face(fi), unsure: false };
                }
            }
            if best.d != before {
                found = true;
                if !matches!(best.feature, Feature::Edge(_)) {
                    cusp = false;
                }
            }
            best.d.abs()
        });
        if !found && bound.is_finite() {
            return self.sdf_within(p, f64::INFINITY);
        }
        st.cusp = cusp;
        best.unsure = st.capped || st.uncertified || st.cusp;
        (best, st)
    }

    /// The nearest point of a C0 line inside its face below `cutoff` where the
    /// distance along the line is critical (where it meets the face's boundary the
    /// edges answer): distance, point, the two sides' outward normals summed.
    fn c0_line(&self, l: &C0Line, p: V3, cutoff: f64, st: &mut ExactStats) -> Option<(f64, V3, V3)> {
        let f = &self.s.faces[l.face];
        let t = f.trim2.as_ref()?;
        let at = |w: f64| if l.k == 0 { [l.value, w] } else { [w, l.value] };
        let mut heap: Vec<(f64, usize, crate::curve3::Bez3, u32)> = Vec::new();
        for (i, (z, _, _)) in l.pieces.iter().enumerate() {
            let lb = box_dist(p, &crate::curve3::bez_box(z));
            if lb < cutoff {
                heap.push((lb, i, z.clone(), 0));
            }
        }
        let (mut bound, mut best): (f64, Option<(f64, V3, usize, f64)>) = (cutoff, None);
        let mut pops = 0;
        while !heap.is_empty() {
            let k = (0..heap.len()).min_by(|&a, &b| heap[a].0.total_cmp(&heap[b].0)).unwrap();
            let (lb, i, z, depth) = heap.swap_remove(k);
            if lb >= bound - DIST_EPS {
                break;
            }
            pops += 1;
            if pops > 4096 {
                st.capped = true;
                break;
            }
            let (a, b) = (at(z.t0), at(z.t1));
            let (lo, hi) = ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])]);
            let state = t.box_class(lo, hi);
            if state == Some(false) {
                continue;
            }
            let s = z.newton(p, 0.5);
            let e = z.eval(s);
            let d = (e.p - p).len();
            let w = z.t0 + s * (z.t1 - z.t0);
            let uv = at(w);
            if d < bound && (state == Some(true) || t.inside(uv[0], uv[1])) {
                bound = d;
                let r = e.p - p;
                let big = |x: V3| x.x.abs().max(x.y.abs()).max(x.z.abs());
                let round = 1024.0 * f64::EPSILON * (big(e.p) + big(p) + d);
                if r.dot(e.d1).abs() <= (round + 1e-12 * e.d1.len()) * e.d1.len() {
                    best = Some((d, e.p, i, w));
                }
            }
            if depth >= 40 || crate::surface3::curve_no_critical_point(&z.pw, p) {
                continue;
            }
            for c in [z.split(0.5).0, z.split(0.5).1] {
                let lbc = box_dist(p, &crate::curve3::bez_box(&c));
                if lbc < bound - DIST_EPS {
                    heap.push((lbc, i, c, depth + 1));
                }
            }
        }
        st.pops += pops;
        let (d, q, i, w) = best?;
        // The two sides' normals at the point: the patch before the line at its far
        // side, the patch after at its near side.
        let (_, before, after) = &l.pieces[i];
        let n = |pa: &crate::surface3::Patch, side: f64| {
            let (s, t) = if l.k == 0 { (side, (w - pa.v0) / (pa.v1 - pa.v0)) } else { ((w - pa.u0) / (pa.u1 - pa.u0), side) };
            let e = pa.eval(s, t);
            e.ps.cross(e.pt).unit()
        };
        Some((d, q, (n(before, 1.0) + n(after, 0.0)) * f.sense))
    }

    /// Face fi's outward normal at a point x on (or within the file's tolerance of)
    /// its surface: a B-spline face's search looks only within a few tolerances of
    /// x first.
    fn normal_at(&self, fi: usize, x: V3) -> V3 {
        let f = &self.s.faces[fi];
        match &f.surf {
            Surf::Nurbs(ns) => ns.closest(x, 4.0 * self.s.tol + 1e-9).or_else(|| ns.closest(x, f64::INFINITY)).map_or(V3::new(0.0, 0.0, 1.0), |h| h.n) * f.sense,
            _ => f.normal(x),
        }
    }

    /// Face fi's nearest critical point inside it below `cutoff`: distance, point,
    /// outward normal.
    fn face(&self, fi: usize, p: V3, cutoff: f64, st: &mut ExactStats) -> Option<(f64, V3, V3)> {
        let f = &self.s.faces[fi];
        match (&f.surf, &f.trim2, &f.analytic2) {
            (Surf::Nurbs(ns), Some(t), _) => {
                let (hit, complete, pops) = ns.closest_trimmed(p, cutoff, &|lo, hi| t.box_class(lo, hi), &|u, v| t.inside(u, v), self.creases[fi].as_ref().unwrap());
                st.pops += pops;
                st.capped |= !complete;
                let h = hit?;
                (h.dist < cutoff && h.n.len() > 0.5).then(|| (h.dist, h.q, h.n * f.sense))
            }
            (surf, Some(t), Some(a)) => {
                let q = surf.foot(p);
                let d = (p - q).len();
                if !(d < cutoff) {
                    return None;
                }
                if let Some(&(x, n)) = self.tips[fi].iter().find(|(x, _)| (*x - q).len() <= 1e-12 * (1.0 + x.len())) {
                    return Some(((p - x).len(), x, n));
                }
                if is_singular(surf, q) {
                    return None;
                }
                let (u, v) = a.uv(q);
                t.inside(u, v).then(|| (d, q, f.normal(q)))
            }
            _ => {
                // No certified loops: the old trim test, and the answer is unsure.
                st.uncertified = true;
                match &f.surf {
                    Surf::Nurbs(_) => self.s.nurbs_face_closest(fi, p, cutoff).map(|(d, q)| (d, q, f.normal(q))),
                    surf => {
                        let q = surf.foot(p);
                        let d = (p - q).len();
                        (d < cutoff && is_foot(p, q, surf.normal(q)) && self.s.classify(fi, q) == Some(true)).then(|| (d, q, f.normal(q)))
                    }
                }
            }
        }
    }
}

/// The surface a face's loops are in: a B-spline face's own, an analytic face's
/// exact rational form.
fn face_nurbs(f: &Face3) -> Option<&NurbsSurf> {
    match (&f.surf, &f.analytic2) {
        (Surf::Nurbs(ns), _) => Some(ns.as_ref()),
        (_, Some(a)) => Some(&a.ns),
        _ => None,
    }
}

/// The angle-weighted normal of a surface's fan around a side it collapses on to
/// one point (direction k held at `value`), over the part of the side the face
/// covers (`inside`, tested a step 1e-9 of the domain into it): near the point the
/// surface is the cone of its limiting tangents S_k(w), with limiting normals
/// S_k x S_kw (u held) or S_kw x S_v (v held), oriented as S_u x S_v just inside.
/// Each normal is weighted by the angle its tangent turns through, the curved
/// version of a corner's normal times its angle. In the surface's own
/// orientation.
fn fan_normal(ns: &NurbsSurf, inside: &dyn Fn(f64, f64) -> bool, k: usize, value: f64) -> V3 {
    let (range, other) = if k == 0 { (ns.u_range, ns.v_range) } else { (ns.v_range, ns.u_range) };
    let h = if value == range.0 { 1.0 } else { -1.0 };
    let step = 1e-9 * (range.1 - range.0) * h;
    let n = 512;
    let mut out = V3::default();
    let mut prev: Option<(V3, V3, bool)> = None;
    for i in 0..=n {
        let w = other.0 + (other.1 - other.0) * i as f64 / n as f64;
        let (u, v) = if k == 0 { (value, w) } else { (w, value) };
        let e = ns.eval(u, v);
        let (tan, nrm) = if k == 0 { (e.ps, e.ps.cross(e.pst)) } else { (e.pt, e.pst.cross(e.pt)) };
        let (tan, nrm) = (tan * h, nrm * h);
        let covered = if k == 0 { inside(value + step, w) } else { inside(w, value + step) };
        if let Some((pt, pnrm, pc)) = prev {
            if pc && covered && tan.len() > 0.0 && pt.len() > 0.0 {
                let ang = pt.unit().cross(tan.unit()).len().atan2(pt.unit().dot(tan.unit()));
                let mid = pnrm.unit() + nrm.unit();
                if mid.len() > 0.0 {
                    out = out + mid.unit() * ang;
                }
            }
        }
        prev = Some((tan, nrm, covered));
    }
    out
}

/// The points an analytic surface is singular at, with the outward pseudo-normal
/// there (the mean of the surface's normals around the point, times the face's
/// sense): a cone's apex, and the tips of a lemon or apple torus on its axis.
fn singular_points(surf: &Surf, sense: f64) -> Vec<(V3, V3)> {
    match surf {
        Surf::Cone { f, r, a } => vec![(f.o + f.z * (-r / a.tan()), f.z * (-a.sin() * sense))],
        Surf::Torus { f, major, minor } if major.abs() <= *minor => {
            let h = (minor * minor - major * major).max(0.0).sqrt();
            vec![(f.o + f.z * h, f.z * sense), (f.o - f.z * h, f.z * -sense)]
        }
        _ => Vec::new(),
    }
}

/// Whether `Surf::foot` returned a singular point (where the surface has no normal).
fn is_singular(surf: &Surf, q: V3) -> bool {
    singular_points(surf, 1.0).iter().any(|(x, _)| (*x - q).len() <= 1e-12 * (1.0 + x.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve3::Frame;
    use crate::step::{self, BFace, Bound, Curve3, OEdge, Solid, Surface};

    /// A solid cylinder r 3 from z 0 to z 5 (two planes, one cylinder, two circles,
    /// one seam) as STEP would write it.
    fn can() -> Solid {
        let z = V3::new(0.0, 0.0, 1.0);
        let x = V3::new(1.0, 0.0, 0.0);
        let f0 = Frame { o: V3::new(0.0, 0.0, 0.0), z, x };
        let f1 = Frame { o: V3::new(0.0, 0.0, 5.0), z, x };
        let (a, b) = (V3::new(3.0, 0.0, 0.0), V3::new(3.0, 0.0, 5.0));
        let circ = |f: Frame, id: u32, v: u32, p: V3, forward: bool| OEdge { edge: id, start: p, end: p, start_vertex: v, end_vertex: v, curve: Curve3::Circle { frame: f, r: 3.0 }, forward };
        let seam = |up: bool| OEdge {
            edge: 3,
            start: if up { a } else { b },
            end: if up { b } else { a },
            start_vertex: if up { 1 } else { 2 },
            end_vertex: if up { 2 } else { 1 },
            curve: Curve3::Line { p: a, dir: z },
            forward: up,
        };
        let face = |id: u32, surface: Surface, same_sense: bool, edges: Vec<OEdge>| BFace { id, surface, same_sense, bounds: vec![Bound { outer: true, edges, vertex: None }] };
        Solid {
            id: 1,
            name: "can".into(),
            faces: vec![
                // Bottom: outward -z, so the loop runs clockwise seen from +z.
                face(10, Surface::Plane { frame: f0 }, false, vec![circ(f0, 1, 1, a, false)]),
                face(11, Surface::Plane { frame: f1 }, true, vec![circ(f1, 2, 2, b, true)]),
                face(12, Surface::Cylinder { frame: f0, r: 3.0 }, true, vec![circ(f0, 1, 1, a, true), seam(true), circ(f1, 2, 2, b, false), seam(false)]),
            ],
        }
    }

    #[test]
    fn exact_evaluator_on_the_can() {
        let (s, rep) = Solid3::from_step(&can(), 1e-9).unwrap();
        assert_eq!(rep.trim2_refused, 0, "{:?}", rep.trim2_notes);
        let ex = Exact::new(&s);
        let exact = |p: V3| {
            let rho = p.x.hypot(p.y);
            let (dr, dz) = (rho - 3.0, (p.z - 2.5).abs() - 2.5);
            if dr <= 0.0 && dz <= 0.0 {
                dr.max(dz)
            } else {
                (dr.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt()
            }
        };
        // Random points, and points on the axis (where every point of a circle of
        // the cylinder is nearest).
        let axis = (0..40).map(|k| V3::new(0.0, 0.0, -2.0 + 9.0 * k as f64 / 39.0));
        for p in (0..2000).map(|k| {
            let t = k as f64;
            V3::new(6.0 * (t * 0.37).sin(), 6.0 * (t * 0.71).cos(), 2.5 + 5.0 * (t * 1.3).sin())
        }).chain(axis) {
            let (sd, _) = ex.sdf(p);
            assert!(!sd.unsure, "unsure at {p:?}");
            assert!((sd.d - exact(p)).abs() < 1e-9, "at {p:?}: {} vs {} ({:?})", sd.d, exact(p), sd.feature);
        }
    }

    #[test]
    fn cylinder_can_signed_distance_is_exact() {
        let (s, _) = Solid3::from_step(&can(), 1e-9).unwrap();
        let exact = |p: V3| {
            let rho = p.x.hypot(p.y);
            let (dr, dz) = (rho - 3.0, (p.z - 2.5).abs() - 2.5);
            if dr <= 0.0 && dz <= 0.0 {
                dr.max(dz)
            } else {
                (dr.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt()
            }
        };
        for k in 0..2000 {
            let t = k as f64;
            let p = V3::new(6.0 * (t * 0.37).sin(), 6.0 * (t * 0.71).cos(), 2.5 + 5.0 * (t * 1.3).sin());
            let sd = s.sdf(p);
            assert!(!sd.unsure, "unsure at {p:?}");
            assert!((sd.d - exact(p)).abs() < 1e-9, "at {p:?}: {} vs {} ({:?})", sd.d, exact(p), sd.feature);
        }
    }

    /// A 10 x 10 box whose top is a biquadratic B-spline surface z = S(x, y) over
    /// the square, bounded by two native B-spline edges (the raised y = 0 and
    /// x = 10 boundaries) and two straight ones, on four side planes and a bottom.
    fn bump_box() -> (Solid, impl Fn(f64, f64) -> f64) {
        let zc = [[5.0, 5.0, 5.0], [7.0, 9.0, 5.0], [5.0, 6.0, 5.0]]; // zc[i][j]: i along x, j along y
        let top = move |x: f64, y: f64| {
            let b = |t: f64| [(1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t];
            let (bu, bv) = (b(x / 10.0), b(y / 10.0));
            (0..3).map(|i| (0..3).map(|j| bu[i] * bv[j] * zc[i][j]).sum::<f64>()).sum()
        };
        let ctrl: Vec<Vec<V3>> = (0..3).map(|i| (0..3).map(|j| V3::new(5.0 * i as f64, 5.0 * j as f64, zc[i][j])).collect()).collect();
        let k = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let surf = Surface::BSpline(step::BSplineSurface { du: 2, dv: 2, ctrl, weights: None, ku: k.clone(), kv: k.clone() });
        let v = |x: f64, y: f64, z: f64| V3::new(x, y, z);
        let (b00, b10, b11, b01) = (v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0), v(10.0, 10.0, 0.0), v(0.0, 10.0, 0.0));
        let (t00, t10, t11, t01) = (v(0.0, 0.0, 5.0), v(10.0, 0.0, 5.0), v(10.0, 10.0, 5.0), v(0.0, 10.0, 5.0));
        let line = |a: V3, b: V3| Curve3::Line { p: a, dir: (b - a).unit() };
        let bs = |p: [V3; 3]| Curve3::BSpline(step::BSplineCurve { degree: 2, ctrl: p.to_vec(), weights: None, knots: k.clone() });
        // (id, curve, start vertex id and point, end vertex id and point) in the curve's own direction.
        let edges: Vec<(u32, Curve3, (u32, V3), (u32, V3))> = vec![
            (1, line(b00, b10), (1, b00), (2, b10)),
            (2, line(b10, b11), (2, b10), (3, b11)),
            (3, line(b11, b01), (3, b11), (4, b01)),
            (4, line(b01, b00), (4, b01), (1, b00)),
            (5, line(b00, t00), (1, b00), (5, t00)),
            (6, line(b10, t10), (2, b10), (6, t10)),
            (7, line(b11, t11), (3, b11), (7, t11)),
            (8, line(b01, t01), (4, b01), (8, t01)),
            (9, bs([t00, v(5.0, 0.0, 7.0), t10]), (5, t00), (6, t10)),
            (10, bs([t10, v(10.0, 5.0, 6.0), t11]), (6, t10), (7, t11)),
            (11, line(t11, t01), (7, t11), (8, t01)),
            (12, line(t01, t00), (8, t01), (5, t00)),
        ];
        let use_ = |id: u32, fwd: bool| {
            let (_, c, s, e) = edges.iter().find(|x| x.0 == id).unwrap().clone();
            let (a, b) = if fwd { (s, e) } else { (e, s) };
            OEdge { edge: id, start: a.1, end: b.1, start_vertex: a.0, end_vertex: b.0, curve: c, forward: fwd }
        };
        let plane = |o: V3, z: V3, x: V3| Surface::Plane { frame: Frame { o, z, x } };
        let face = |id: u32, surface: Surface, uses: &[(u32, bool)]| BFace {
            id,
            surface,
            same_sense: true,
            bounds: vec![Bound { outer: true, edges: uses.iter().map(|&(e, f)| use_(e, f)).collect(), vertex: None }],
        };
        let (x, y, z) = (v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0));
        let faces = vec![
            face(20, surf, &[(9, true), (10, true), (11, true), (12, true)]),
            face(21, plane(b00, -z, x), &[(4, false), (3, false), (2, false), (1, false)]),
            face(22, plane(b00, -y, x), &[(1, true), (6, true), (9, false), (5, false)]),
            face(23, plane(b10, x, y), &[(2, true), (7, true), (10, false), (6, false)]),
            face(24, plane(b11, y, x), &[(3, true), (8, true), (11, false), (7, false)]),
            face(25, plane(b01, -x, y), &[(4, true), (5, true), (12, false), (8, false)]),
        ];
        (Solid { id: 1, name: "bump box".into(), faces }, top)
    }

    #[test]
    fn bspline_top_box_signed_distance() {
        let (solid, top) = bump_box();
        let (s, rep) = Solid3::from_step(&solid, 1e-9).unwrap();
        assert!(rep.max_edge_off < 1e-9, "edges off their surfaces by {}", rep.max_edge_off);
        // Dense samples of every face (the reference distance is their minimum).
        let n = 300;
        let mut samples = Vec::new();
        for a in 0..=n {
            for b in 0..=n {
                let (fx, fy) = (10.0 * a as f64 / n as f64, 10.0 * b as f64 / n as f64);
                samples.push(V3::new(fx, fy, top(fx, fy)));
                samples.push(V3::new(fx, fy, 0.0));
                for (px, py) in [(fx, 0.0), (fx, 10.0), (0.0, fx), (10.0, fx)] {
                    samples.push(V3::new(px, py, top(px, py) * b as f64 / n as f64));
                }
            }
        }
        let h = 10.0 / n as f64;
        for k in 0..60 {
            let t = k as f64;
            let p = V3::new(5.0 + 8.0 * (t * 0.37).sin(), 5.0 + 8.0 * (t * 0.71).cos(), 4.0 + 6.0 * (t * 1.3).sin());
            let sd = s.sdf(p);
            let inside = p.x > 0.0 && p.x < 10.0 && p.y > 0.0 && p.y < 10.0 && p.z > 0.0 && p.z < top(p.x, p.y);
            let dense = samples.iter().map(|&q| (q - p).len()).fold(f64::INFINITY, f64::min);
            assert_eq!(sd.d < 0.0, inside, "sign at {p:?}: {sd:?}");
            assert!(sd.d.abs() <= dense + 1e-9, "at {p:?}: |sd| {} above sampled {dense} ({:?})", sd.d.abs(), sd.feature);
            assert!(sd.d.abs() >= dense - h, "at {p:?}: |sd| {} far below sampled {dense} ({:?})", sd.d.abs(), sd.feature);
        }
    }

    #[test]
    fn exact_evaluator_on_the_bump_box() {
        let (solid, top) = bump_box();
        let (s, rep) = Solid3::from_step(&solid, 1e-9).unwrap();
        assert_eq!(rep.trim2_refused, 0, "{:?}", rep.trim2_notes);
        let ex = Exact::new(&s);
        let n = 300;
        let mut samples = Vec::new();
        for a in 0..=n {
            for b in 0..=n {
                let (fx, fy) = (10.0 * a as f64 / n as f64, 10.0 * b as f64 / n as f64);
                samples.push(V3::new(fx, fy, top(fx, fy)));
                samples.push(V3::new(fx, fy, 0.0));
                for (px, py) in [(fx, 0.0), (fx, 10.0), (0.0, fx), (10.0, fx)] {
                    samples.push(V3::new(px, py, top(px, py) * b as f64 / n as f64));
                }
            }
        }
        let h = 10.0 / n as f64;
        for k in 0..200 {
            let t = k as f64;
            let p = V3::new(5.0 + 8.0 * (t * 0.37).sin(), 5.0 + 8.0 * (t * 0.71).cos(), 4.0 + 6.0 * (t * 1.3).sin());
            let (sd, _) = ex.sdf(p);
            let inside = p.x > 0.0 && p.x < 10.0 && p.y > 0.0 && p.y < 10.0 && p.z > 0.0 && p.z < top(p.x, p.y);
            let dense = samples.iter().map(|&q| (q - p).len()).fold(f64::INFINITY, f64::min);
            assert!(!sd.unsure, "unsure at {p:?}");
            assert_eq!(sd.d < 0.0, inside, "sign at {p:?}: {sd:?}");
            assert!(sd.d.abs() <= dense + 1e-9, "at {p:?}: |sd| {} above sampled {dense} ({:?})", sd.d.abs(), sd.feature);
            assert!(sd.d.abs() >= dense - h, "at {p:?}: |sd| {} far below sampled {dense} ({:?})", sd.d.abs(), sd.feature);
        }
    }

}
