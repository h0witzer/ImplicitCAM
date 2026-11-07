//! Exact surfaces of B-rep faces: planes, circular cylinders and cones, spheres,
//! tori and rational B-spline surfaces, with the surface's own normal and the
//! closest point of an untrimmed surface.
//!
//! Analytic surfaces use the STEP (ISO 10303-42) parameterizations:
//!   cylinder  o + r e_r(u) + v z
//!   cone      o + (r + v tan a) e_r(u) + v z       (a = semi-angle)
//!   sphere    o + r cos v e_r(u) + r sin v z
//!   torus     o + (R + r cos v) e_r(u) + r sin v z
//! with e_r(u) = cos u x + sin u y in the surface frame. The surface normal is
//! the STEP one (outward from the axis, center or tube center; Su x Sv for
//! B-spline surfaces).
//!
//! Each analytic surface has exactly one local minimum of the distance to a
//! point (the foot), so the closest point of a trimmed face is its foot when
//! the foot lies in the face and on the face's boundary otherwise. B-spline
//! surfaces can have several; their closest point is a certified branch and
//! bound over Bezier patches.

use std::sync::Arc;

use crate::curve3::{clamp, insert_knot, Bez3, Frame, V3};

#[derive(Clone, Debug)]
pub enum Surf {
    Plane { f: Frame },
    Cylinder { f: Frame, r: f64 },
    Cone { f: Frame, r: f64, a: f64 },
    Sphere { f: Frame, r: f64 },
    /// See `step::Surface::Torus`: `major` < `minor` is the apple of a self-intersecting
    /// torus, `major` < 0 its lemon; either way the meridian circle on the axis's own side.
    Torus { f: Frame, major: f64, minor: f64 },
    Nurbs(Arc<NurbsSurf>),
}

/// Axial coordinates of p in a frame: angle about z, distance from the axis,
/// height along z, and the radial and tangential unit vectors at that angle.
#[derive(Clone, Copy, Debug)]
pub struct Axial {
    pub theta: f64,
    pub rho: f64,
    pub h: f64,
    pub er: V3,
    pub et: V3,
}

pub fn axial(f: &Frame, p: V3) -> Axial {
    let l = f.local(p);
    let rho = (l.x * l.x + l.y * l.y).sqrt();
    let theta = if rho > 0.0 { l.y.atan2(l.x) } else { 0.0 };
    let (s, c) = theta.sin_cos();
    Axial { theta, rho, h: l.z, er: f.dir(V3::new(c, s, 0.0)), et: f.dir(V3::new(-s, c, 0.0)) }
}

impl Surf {
    pub fn name(&self) -> &'static str {
        match self {
            Surf::Plane { .. } => "plane",
            Surf::Cylinder { .. } => "cylinder",
            Surf::Cone { .. } => "cone",
            Surf::Sphere { .. } => "sphere",
            Surf::Torus { .. } => "torus",
            Surf::Nurbs(_) => "B-spline surface",
        }
    }

    pub fn frame(&self) -> Option<&Frame> {
        match self {
            Surf::Plane { f } | Surf::Cylinder { f, .. } | Surf::Cone { f, .. } | Surf::Sphere { f, .. } | Surf::Torus { f, .. } => Some(f),
            Surf::Nurbs(_) => None,
        }
    }

    /// Apex of a cone (height -r / tan a on its axis).
    pub fn apex(&self) -> Option<V3> {
        match self {
            Surf::Cone { f, r, a } => Some(f.world(V3::new(0.0, 0.0, -r / a.tan()))),
            _ => None,
        }
    }

    /// The foot of p on an analytic surface: its unique local minimum of distance.
    /// On a degenerate position (on the axis, at the center) any closest point.
    /// For a cone, the nappe r + v tan a >= 0 (the apex when p is behind it).
    pub fn foot(&self, p: V3) -> V3 {
        match self {
            Surf::Plane { f } => p - f.z * (p - f.o).dot(f.z),
            Surf::Cylinder { f, r } => {
                let ax = axial(f, p);
                f.o + ax.er * *r + f.z * ax.h
            }
            Surf::Cone { f, r, a } => {
                let ax = axial(f, p);
                let (sa, ca) = a.sin_cos();
                let ha = -r / a.tan();
                let s = ax.rho * sa + (ax.h - ha) * ca;
                if s <= 0.0 {
                    f.o + f.z * ha
                } else {
                    f.o + ax.er * (s * sa) + f.z * (ha + s * ca)
                }
            }
            Surf::Sphere { f, r } => {
                let d = p - f.o;
                let l = d.len();
                if l > 0.0 {
                    f.o + d * (r / l)
                } else {
                    f.o + f.x * *r
                }
            }
            Surf::Torus { f, major, minor } => {
                let ax = axial(f, p);
                let c = f.o + ax.er * *major;
                let d = p - c;
                let l = d.len();
                let q = if l > 0.0 { c + d * (minor / l) } else { c + ax.er * *minor };
                if (q - f.o).dot(ax.er) >= 0.0 || major.abs() > *minor {
                    q
                } else {
                    // The nearest point of the circle is across the axis: the meridian arc
                    // ends at the axis, at the nearer of its two points there.
                    let h = (minor * minor - major * major).max(0.0).sqrt();
                    if ax.h >= 0.0 { f.o + f.z * h } else { f.o - f.z * h }
                }
            }
            Surf::Nurbs(n) => n.closest(p, f64::INFINITY).map(|h| h.q).unwrap_or(p),
        }
    }

    /// The surface's own unit normal at the surface point nearest x.
    pub fn normal(&self, x: V3) -> V3 {
        match self {
            Surf::Plane { f } => f.z,
            Surf::Cylinder { f, .. } => axial(f, x).er,
            Surf::Cone { f, a, .. } => {
                let ax = axial(f, x);
                let (sa, ca) = a.sin_cos();
                ax.er * ca - f.z * sa
            }
            Surf::Sphere { f, .. } => (x - f.o).unit(),
            Surf::Torus { f, major, .. } => {
                let ax = axial(f, x);
                (x - (f.o + ax.er * *major)).unit()
            }
            Surf::Nurbs(n) => n.closest(x, f64::INFINITY).map(|h| h.n).unwrap_or(V3::new(0.0, 0.0, 1.0)),
        }
    }

    /// The foot of p and the surface's own unit normal there, from one search.
    pub fn foot_normal(&self, p: V3) -> (V3, V3) {
        match self {
            Surf::Nurbs(n) => n.closest(p, f64::INFINITY).map(|h| (h.q, h.n)).unwrap_or((p, V3::new(0.0, 0.0, 1.0))),
            _ => {
                let q = self.foot(p);
                (q, self.normal(q))
            }
        }
    }

    /// Distance from x to the untrimmed surface (for tolerance checks).
    pub fn dist(&self, x: V3) -> f64 {
        (x - self.foot(x)).len()
    }
}

// ---------------------------------------------------------------- B-spline surfaces

/// Bernstein basis of degree n at t with first and second derivatives.
fn bernstein(n: usize, t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut levels: Vec<Vec<f64>> = vec![vec![1.0]];
    for k in 1..=n {
        let prev = &levels[k - 1];
        let mut b = vec![0.0; k + 1];
        for i in 0..=k {
            let left = if i > 0 { prev[i - 1] } else { 0.0 };
            let right = if i < k { prev[i] } else { 0.0 };
            b[i] = (1.0 - t) * right + t * left;
        }
        levels.push(b);
    }
    let get = |lv: &Vec<f64>, i: isize| if i < 0 || i as usize >= lv.len() { 0.0 } else { lv[i as usize] };
    let mut d1 = vec![0.0; n + 1];
    let mut d2 = vec![0.0; n + 1];
    for i in 0..=n {
        let ii = i as isize;
        if n >= 1 {
            let l = &levels[n - 1];
            d1[i] = n as f64 * (get(l, ii - 1) - get(l, ii));
        }
        if n >= 2 {
            let l = &levels[n - 2];
            d2[i] = (n * (n - 1)) as f64 * (get(l, ii - 2) - 2.0 * get(l, ii - 1) + get(l, ii));
        }
    }
    (levels.pop().unwrap(), d1, d2)
}

/// Newton iterations taken on patches (for profiling).
pub static NEWTON_ITERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Degrees below this evaluate on the stack.
const BMAX: usize = 16;

/// Bernstein basis of degree n < BMAX at t with first and second derivatives,
/// without allocating.
fn bernstein_small(n: usize, t: f64) -> ([f64; BMAX], [f64; BMAX], [f64; BMAX]) {
    // Levels n - 2 and n - 1 of the triangle are kept for the derivatives.
    let (mut cur, mut l1, mut l2) = ([0.0f64; BMAX], [0.0f64; BMAX], [0.0f64; BMAX]);
    cur[0] = 1.0;
    if n >= 1 && n - 1 == 0 {
        l1 = cur;
    }
    if n >= 2 && n - 2 == 0 {
        l2 = cur;
    }
    for k in 1..=n {
        for i in (0..=k).rev() {
            let left = if i > 0 { cur[i - 1] } else { 0.0 };
            let right = if i < k { cur[i] } else { 0.0 };
            cur[i] = (1.0 - t) * right + t * left;
        }
        if n >= 1 && k == n - 1 {
            l1 = cur;
        }
        if n >= 2 && k == n - 2 {
            l2 = cur;
        }
    }
    let get = |lv: &[f64; BMAX], len: usize, i: isize| if i < 0 || i as usize >= len { 0.0 } else { lv[i as usize] };
    let (mut d1, mut d2) = ([0.0f64; BMAX], [0.0f64; BMAX]);
    for i in 0..=n {
        let ii = i as isize;
        if n >= 1 {
            d1[i] = n as f64 * (get(&l1, n, ii - 1) - get(&l1, n, ii));
        }
        if n >= 2 {
            d2[i] = (n * (n - 1)) as f64 * (get(&l2, n - 1, ii - 2) - 2.0 * get(&l2, n - 1, ii - 1) + get(&l2, n - 1, ii));
        }
    }
    (cur, d1, d2)
}

/// Point, first and second derivatives of a surface at local (s, t).
#[derive(Clone, Copy, Debug)]
pub struct SEval {
    pub p: V3,
    pub ps: V3,
    pub pt: V3,
    pub pss: V3,
    pub pst: V3,
    pub ptt: V3,
}

/// A rational Bezier patch in homogeneous form: pw[i][j], i along u.
#[derive(Clone, Debug)]
pub struct Patch {
    pub pw: Vec<Vec<[f64; 4]>>,
    pub u0: f64,
    pub u1: f64,
    pub v0: f64,
    pub v1: f64,
}

fn v3h(h: [f64; 4]) -> V3 {
    V3::new(h[0], h[1], h[2])
}

fn casteljau_split(pts: &[[f64; 4]], s: f64) -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
    let n = pts.len() - 1;
    let mut b = pts.to_vec();
    let (mut left, mut right) = (vec![b[0]], vec![b[n]]);
    for level in 1..=n {
        for i in 0..=(n - level) {
            for c in 0..4 {
                b[i][c] = (1.0 - s) * b[i][c] + s * b[i + 1][c];
            }
        }
        left.push(b[0]);
        right.push(b[n - level]);
    }
    right.reverse();
    (left, right)
}

impl Patch {
    pub fn ctrl(&self) -> impl Iterator<Item = V3> + '_ {
        self.pw.iter().flatten().map(|h| V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
    }
    pub fn bbox(&self) -> (V3, V3) {
        let mut lo = V3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = V3::new(f64::MIN, f64::MIN, f64::MIN);
        for q in self.ctrl() {
            lo = V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z));
            hi = V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z));
        }
        (lo, hi)
    }
    pub fn eval(&self, s: f64, t: f64) -> SEval {
        let (du, dv) = (self.pw.len() - 1, self.pw[0].len() - 1);
        if du < BMAX && dv < BMAX {
            return self.eval_small(s, t);
        }
        let (bu, bu1, bu2) = bernstein(du, s);
        let (bv, bv1, bv2) = bernstein(dv, t);
        let mut a = [[0.0f64; 4]; 6]; // A, As, At, Ass, Ast, Att
        for i in 0..=du {
            for j in 0..=dv {
                let h = self.pw[i][j];
                let w = [bu[i] * bv[j], bu1[i] * bv[j], bu[i] * bv1[j], bu2[i] * bv[j], bu1[i] * bv1[j], bu[i] * bv2[j]];
                for k in 0..6 {
                    for c in 0..4 {
                        a[k][c] += w[k] * h[c];
                    }
                }
            }
        }
        let w = a[0][3];
        let p = v3h(a[0]) * (1.0 / w);
        let ps = (v3h(a[1]) - p * a[1][3]) * (1.0 / w);
        let pt = (v3h(a[2]) - p * a[2][3]) * (1.0 / w);
        let pss = (v3h(a[3]) - ps * (2.0 * a[1][3]) - p * a[3][3]) * (1.0 / w);
        let pst = (v3h(a[4]) - ps * a[2][3] - pt * a[1][3] - p * a[4][3]) * (1.0 / w);
        let ptt = (v3h(a[5]) - pt * (2.0 * a[2][3]) - p * a[5][3]) * (1.0 / w);
        SEval { p, ps, pt, pss, pst, ptt }
    }
    /// eval for degrees below BMAX, with the basis on the stack.
    fn eval_small(&self, s: f64, t: f64) -> SEval {
        let (du, dv) = (self.pw.len() - 1, self.pw[0].len() - 1);
        let (bu, bu1, bu2) = bernstein_small(du, s);
        let (bv, bv1, bv2) = bernstein_small(dv, t);
        let mut a = [[0.0f64; 4]; 6]; // A, As, At, Ass, Ast, Att
        for i in 0..=du {
            for j in 0..=dv {
                let h = self.pw[i][j];
                let w = [bu[i] * bv[j], bu1[i] * bv[j], bu[i] * bv1[j], bu2[i] * bv[j], bu1[i] * bv1[j], bu[i] * bv2[j]];
                for k in 0..6 {
                    for c in 0..4 {
                        a[k][c] += w[k] * h[c];
                    }
                }
            }
        }
        let w = a[0][3];
        let p = v3h(a[0]) * (1.0 / w);
        let ps = (v3h(a[1]) - p * a[1][3]) * (1.0 / w);
        let pt = (v3h(a[2]) - p * a[2][3]) * (1.0 / w);
        let pss = (v3h(a[3]) - ps * (2.0 * a[1][3]) - p * a[3][3]) * (1.0 / w);
        let pst = (v3h(a[4]) - ps * a[2][3] - pt * a[1][3] - p * a[4][3]) * (1.0 / w);
        let ptt = (v3h(a[5]) - pt * (2.0 * a[2][3]) - p * a[5][3]) * (1.0 / w);
        SEval { p, ps, pt, pss, pst, ptt }
    }

    pub fn split_u(&self, s: f64) -> (Patch, Patch) {
        let (du, dv) = (self.pw.len() - 1, self.pw[0].len() - 1);
        let mut l = vec![vec![[0.0; 4]; dv + 1]; du + 1];
        let mut r = l.clone();
        for j in 0..=dv {
            let col: Vec<[f64; 4]> = (0..=du).map(|i| self.pw[i][j]).collect();
            let (a, b) = casteljau_split(&col, s);
            for i in 0..=du {
                l[i][j] = a[i];
                r[i][j] = b[i];
            }
        }
        let um = self.u0 + s * (self.u1 - self.u0);
        // The ranges field by field: `..self.clone()` would copy the whole control net
        // twice per split only to drop it.
        let (u0, u1, v0, v1) = (self.u0, self.u1, self.v0, self.v1);
        (Patch { pw: l, u0, u1: um, v0, v1 }, Patch { pw: r, u0: um, u1, v0, v1 })
    }
    pub fn split_v(&self, t: f64) -> (Patch, Patch) {
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for row in &self.pw {
            let (a, b) = casteljau_split(row, t);
            l.push(a);
            r.push(b);
        }
        let vm = self.v0 + t * (self.v1 - self.v0);
        let (u0, u1, v0, v1) = (self.u0, self.u1, self.v0, self.v1);
        (Patch { pw: l, u0, u1, v0, v1: vm }, Patch { pw: r, u0, u1, v0: vm, v1 })
    }
    /// Newton on |S - p|^2 over [0, 1]^2 from (s, t); Gauss-Newton where the
    /// Hessian is not positive definite.
    pub fn newton(&self, p: V3, s: f64, t: f64) -> (f64, f64) {
        self.newton_in(p, s, t, (0.0, 1.0), (0.0, 1.0))
    }

    /// Newton as above, with s and t kept in the given ranges; ranges past [0, 1]
    /// continue the patch's polynomial exactly.
    pub fn newton_in(&self, p: V3, s: f64, t: f64, sr: (f64, f64), tr: (f64, f64)) -> (f64, f64) {
        self.newton_iters(p, s, t, sr, tr, 50)
    }

    /// Newton over [0, 1]^2 from (s, t) with at most `iters` steps: enough for a
    /// branch and bound, which only needs a good point (its bounds certify).
    pub fn newton_capped(&self, p: V3, s: f64, t: f64, iters: usize) -> (f64, f64) {
        self.newton_iters(p, s, t, (0.0, 1.0), (0.0, 1.0), iters)
    }

    fn newton_iters(&self, p: V3, mut s: f64, mut t: f64, sr: (f64, f64), tr: (f64, f64), iters: usize) -> (f64, f64) {
        // A parameter sitting on its bound with its step pointing out is held
        // there, and Newton continues in the other one alone (the minimum along
        // that side of the patch); clamping both each step would bounce off the
        // bound until the iteration cap.
        let (mut hold_s, mut hold_t) = (false, false);
        for _ in 0..iters {
            NEWTON_ITERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let e = self.eval(s, t);
            let r = e.p - p;
            let g = [e.ps.dot(r), e.pt.dot(r)];
            let (jss, jst, jtt) = (e.ps.dot(e.ps), e.ps.dot(e.pt), e.pt.dot(e.pt));
            let mut h = [jss + e.pss.dot(r), jst + e.pst.dot(r), jtt + e.ptt.dot(r)];
            let (ds, dt) = match (hold_s, hold_t) {
                (true, true) => break,
                (true, false) => (0.0, -g[1] / if h[2] > 0.0 { h[2] } else { jtt.max(1e-300) }),
                (false, true) => (-g[0] / if h[0] > 0.0 { h[0] } else { jss.max(1e-300) }, 0.0),
                (false, false) => {
                    if !(h[0] > 0.0 && h[0] * h[2] - h[1] * h[1] > 0.0) {
                        h = [jss, jst, jtt];
                    }
                    let det = h[0] * h[2] - h[1] * h[1];
                    if det.abs() < 1e-300 {
                        break;
                    }
                    (-(h[2] * g[0] - h[1] * g[1]) / det, -(h[0] * g[1] - h[1] * g[0]) / det)
                }
            };
            let at_bound = |x: f64, d: f64, r: (f64, f64)| (x <= r.0 && d < 0.0) || (x >= r.1 && d > 0.0);
            if !hold_s && at_bound(s, ds, sr) {
                hold_s = true;
                continue;
            }
            if !hold_t && at_bound(t, dt, tr) {
                hold_t = true;
                continue;
            }
            let (ns, nt) = ((s + ds).clamp(sr.0, sr.1), (t + dt).clamp(tr.0, tr.1));
            let moved = (ns - s).abs() + (nt - t).abs();
            s = ns;
            t = nt;
            if moved < 1e-13 {
                break;
            }
        }
        (s, t)
    }
}

/// Bezier points of the single knot span knots[p] .. knots[p + 1] of a local
/// B-spline piece (2p + 2 knots, p + 1 points): raise both span ends to
/// multiplicity p by knot insertion; the p + 1 points ending at the last copy of
/// the span's start are then its Bezier points.
pub fn span_bezier<const N: usize>(p: usize, knots: &[f64], pts: &[[f64; N]]) -> Vec<[f64; N]> {
    let (a, b) = (knots[p], knots[p + 1]);
    let (mut kn, mut pt) = (knots.to_vec(), pts.to_vec());
    while kn.iter().filter(|&&x| x == a).count() < p {
        insert_knot(p, &mut kn, &mut pt, a);
    }
    while kn.iter().filter(|&&x| x == b).count() < p {
        insert_knot(p, &mut kn, &mut pt, b);
    }
    let last_a = kn.iter().rposition(|&x| x == a).unwrap();
    pt[last_a - p..=last_a].to_vec()
}

/// Node of the quadtree over a B-spline surface's grid of knot spans.
#[derive(Clone, Debug)]
pub struct QNode {
    pub lo: V3,
    pub hi: V3,
    /// Span index ranges [i0, i1) along u and [j0, j1) along v.
    pub i0: u32,
    pub i1: u32,
    pub j0: u32,
    pub j1: u32,
    pub kids: Vec<u32>,
}

/// A clamped rational B-spline surface, kept as its control net. Bezier patches
/// are extracted one knot span at a time when a search reaches them, and kept once
/// extracted when the surface is small enough to keep them all (`CACHE_POINTS`; a
/// surface like the brain coral's, 363 thousand control points at degree 5, would need
/// hundreds of megabytes as patches, and extracts each time). Each span is bounded by
/// the box of the control points that act on it (convex hull property, positive
/// weights), and a quadtree over the span grid carries the union boxes.
#[derive(Clone, Debug)]
pub struct NurbsSurf {
    pub du: usize,
    pub dv: usize,
    nv: usize,
    /// Homogeneous control net, point (i, j) at i * nv + j.
    pw: Vec<[f64; 4]>,
    ku: Vec<f64>,
    kv: Vec<f64>,
    /// Knot index s of each nonempty span ku[s] .. ku[s + 1] (and alike in v).
    su: Vec<usize>,
    sv: Vec<usize>,
    pub nodes: Vec<QNode>,
    pub u_range: (f64, f64),
    pub v_range: (f64, f64),
    /// The spans' patches as extracted (span (a, b) at a * spans along v + b), when kept.
    kept: Option<Arc<[std::sync::OnceLock<Patch>]>>,
}

/// The most patch points a surface keeps extracted (4 million: 128 MB).
const CACHE_POINTS: usize = 1 << 22;

/// A point of a B-spline surface nearest a query.
#[derive(Clone, Copy, Debug)]
pub struct SurfHit {
    pub q: V3,
    pub dist: f64,
    pub u: f64,
    pub v: f64,
    /// Unit normal Su x Sv at the hit.
    pub n: V3,
}

/// What a branch and bound entry holds: a quadtree node, or an extracted patch.
pub enum Entry {
    Node(u32),
    Patch(Patch),
}

/// Min-heap entry for branch and bound over a B-spline surface.
pub struct PatchItem {
    pub lb: f64,
    pub entry: Entry,
    pub depth: u32,
    /// Known to lie wholly inside its trimmed face.
    pub inside: bool,
}
impl PartialEq for PatchItem {
    fn eq(&self, o: &Self) -> bool {
        self.lb == o.lb
    }
}
impl Eq for PatchItem {}
impl PartialOrd for PatchItem {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for PatchItem {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.lb.total_cmp(&self.lb)
    }
}

/// Newton steps per patch inside a branch and bound search.
pub const SEARCH_NEWTON: usize = 12;

/// Distances below this (mm) are treated as settled by the branch and bound searches.
pub const DIST_EPS: f64 = 1e-9;

pub fn box_dist(p: V3, b: &(V3, V3)) -> f64 {
    V3::new((b.0.x - p.x).max(p.x - b.1.x).max(0.0), (b.0.y - p.y).max(p.y - b.1.y).max(0.0), (b.0.z - p.z).max(p.z - b.1.z).max(0.0)).len()
}

/// Whether a surface point is a critical point of the squared distance from p: the
/// gradient (S - p).S_s, (S - p).S_t is zero to within what rounding leaves of it
/// (a thousand units in the last place of the coordinates involved) and what
/// Newton's stop leaves (a step under 1e-12 of the patch).
pub fn critical(p: V3, e: &SEval) -> bool {
    let r = e.p - p;
    let big = |x: V3| x.x.abs().max(x.y.abs()).max(x.z.abs());
    let round = 1024.0 * f64::EPSILON * (big(e.p) + big(p) + r.len());
    let ok = |d: V3| r.dot(d).abs() <= (round + 1e-12 * d.len()) * d.len();
    ok(e.ps) && ok(e.pt)
}

/// Parameter lines of a surface where the distance to it can have a minimum that
/// no search covers otherwise and that is not a critical point of a patch's
/// parameters: the sides it collapses on to a point. A patch with a side on one is
/// never cleared by the critical-point test. (Seams are covered by their seam
/// edges, and interior C0 knot lines by searches along them, `c0_lines`.)
#[derive(Clone, Debug, Default)]
pub struct Creases {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    /// Use the critical-point test at all (off only to measure what it saves).
    pub test: bool,
}

impl Creases {
    pub fn touches(&self, pa: &Patch) -> bool {
        let on = |x: f64, lines: &[f64], a: f64, b: f64| lines.iter().any(|&c| (x - c).abs() <= 1e-12 * (b - a).abs().max(c.abs()));
        on(pa.u0, &self.u, pa.u0, pa.u1) || on(pa.u1, &self.u, pa.u0, pa.u1) || on(pa.v0, &self.v, pa.v0, pa.v1) || on(pa.v1, &self.v, pa.v0, pa.v1)
    }
}

/// A polynomial over [0, 1]^2 in tensor Bernstein form: degree (m, n), coefficient
/// (i, j) at i * (n + 1) + j.
#[derive(Clone)]
struct Bern2 {
    m: usize,
    n: usize,
    c: Vec<f64>,
}

fn binomials(n: usize) -> Vec<f64> {
    let mut b = vec![1.0; n + 1];
    for k in 1..=n {
        b[k] = b[k - 1] * (n + 1 - k) as f64 / k as f64;
    }
    b
}

impl Bern2 {
    fn mul(&self, o: &Bern2) -> Bern2 {
        let (m, n) = (self.m + o.m, self.n + o.n);
        let (ba, bb, bo, bp) = (binomials(self.m), binomials(self.n), binomials(o.m), binomials(o.n));
        let (bm, bn) = (binomials(m), binomials(n));
        let mut c = vec![0.0; (m + 1) * (n + 1)];
        for i in 0..=self.m {
            for j in 0..=self.n {
                let a = self.c[i * (self.n + 1) + j] * ba[i] * bb[j];
                if a == 0.0 {
                    continue;
                }
                for k in 0..=o.m {
                    for l in 0..=o.n {
                        c[(i + k) * (n + 1) + j + l] += a * o.c[k * (o.n + 1) + l] * bo[k] * bp[l];
                    }
                }
            }
        }
        for k in 0..=m {
            for l in 0..=n {
                c[k * (n + 1) + l] /= bm[k] * bn[l];
            }
        }
        Bern2 { m, n, c }
    }
    fn d_u(&self) -> Bern2 {
        if self.m == 0 {
            return Bern2 { m: 0, n: self.n, c: vec![0.0; self.n + 1] };
        }
        let n1 = self.n + 1;
        let c = (0..self.m).flat_map(|i| (0..n1).map(move |j| (i, j))).map(|(i, j)| self.m as f64 * (self.c[(i + 1) * n1 + j] - self.c[i * n1 + j])).collect();
        Bern2 { m: self.m - 1, n: self.n, c }
    }
    fn d_v(&self) -> Bern2 {
        if self.n == 0 {
            return Bern2 { m: self.m, n: 0, c: vec![0.0; self.m + 1] };
        }
        let n1 = self.n + 1;
        let c = (0..=self.m).flat_map(|i| (0..self.n).map(move |j| (i, j))).map(|(i, j)| self.n as f64 * (self.c[i * n1 + j + 1] - self.c[i * n1 + j])).collect();
        Bern2 { m: self.m, n: self.n - 1, c }
    }
    fn elevate_u(&self) -> Bern2 {
        let (m, n1) = (self.m + 1, self.n + 1);
        let mut c = vec![0.0; (m + 1) * n1];
        for i in 0..=m {
            let a = i as f64 / m as f64;
            for j in 0..n1 {
                let lo = if i > 0 { self.c[(i - 1) * n1 + j] } else { 0.0 };
                let hi = if i < m { self.c[i * n1 + j] } else { 0.0 };
                c[i * n1 + j] = a * lo + (1.0 - a) * hi;
            }
        }
        Bern2 { m, n: self.n, c }
    }
    fn elevate_v(&self) -> Bern2 {
        let (n, n0) = (self.n + 1, self.n + 1);
        let mut c = vec![0.0; (self.m + 1) * (n + 1)];
        for i in 0..=self.m {
            for j in 0..=n {
                let a = j as f64 / n as f64;
                let lo = if j > 0 { self.c[i * n0 + j - 1] } else { 0.0 };
                let hi = if j < n { self.c[i * n0 + j] } else { 0.0 };
                c[i * (n + 1) + j] = a * lo + (1.0 - a) * hi;
            }
        }
        Bern2 { m: self.m, n, c }
    }
    fn axpy(&self, a: f64, o: &Bern2, b: f64) -> Bern2 {
        Bern2 { m: self.m, n: self.n, c: self.c.iter().zip(&o.c).map(|(x, y)| a * x + b * y).collect() }
    }
    fn max_abs(&self) -> f64 {
        self.c.iter().fold(0.0, |a: f64, x| a.max(x.abs()))
    }
}

impl Patch {
    /// Whether the squared distance g = |S - p|^2 has no critical point on the
    /// closed patch. With Q = P - p w (the homogeneous point less p), the
    /// derivatives are g_s w^3 / 2 = w Q.Q_s - w_s Q.Q and alike in t: polynomials,
    /// so the combination a g_s + b g_t along the gradient at the patch's middle is
    /// one in Bernstein form, and when all its coefficients are positive (clear of
    /// what rounding can make of them) it has no zero on the patch, so neither has
    /// the gradient. Weights must be positive.
    pub fn no_critical_point(&self, p: V3) -> bool {
        let (m, n) = (self.pw.len() - 1, self.pw[0].len() - 1);
        let mk = |f: &dyn Fn(&[f64; 4]) -> f64| Bern2 { m, n, c: self.pw.iter().flatten().map(f).collect() };
        let w = mk(&|h| h[3]);
        if w.c.iter().any(|&x| !(x > 0.0)) {
            return false;
        }
        let e = self.eval(0.5, 0.5);
        let r = e.p - p;
        let (a, b) = (e.ps.dot(r), e.pt.dot(r));
        let l = a.hypot(b);
        if !(l > 0.0) {
            return false;
        }
        let (a, b) = (a / l, b / l);
        let q = [mk(&|h| h[0] - p.x * h[3]), mk(&|h| h[1] - p.y * h[3]), mk(&|h| h[2] - p.z * h[3])];
        let (qu, qv) = (q.clone().map(|x| x.d_u()), q.clone().map(|x| x.d_v()));
        let dot = |x: &[Bern2; 3], y: &[Bern2; 3]| {
            let mut s = x[0].mul(&y[0]);
            for k in 1..3 {
                s = s.axpy(1.0, &x[k].mul(&y[k]), 1.0);
            }
            s
        };
        let qq = dot(&q, &q);
        let gs = w.mul(&dot(&q, &qu)).axpy(1.0, &w.d_u().mul(&qq), -1.0);
        let gt = w.mul(&dot(&q, &qv)).axpy(1.0, &w.d_v().mul(&qq), -1.0);
        let f = gs.elevate_u().axpy(a, &gt.elevate_v(), b);
        // Each coefficient is a weighted mean of products of the factors'
        // coefficients: its rounding error is a small multiple of these products.
        let qmax = |x: &[Bern2; 3]| x.iter().map(|y| y.max_abs()).fold(0.0, f64::max);
        let (q0, q1, q2) = (qmax(&q), qmax(&qu), qmax(&qv));
        let size = w.max_abs() * 3.0 * q0 * (a.abs() * q1 + b.abs() * q2) + (a.abs() * w.d_u().max_abs() + b.abs() * w.d_v().max_abs()) * 3.0 * q0 * q0;
        f.c.iter().all(|&x| x > 1e-11 * size)
    }
}

/// Whether |C - p|^2 has no critical point on a rational Bezier curve (homogeneous
/// points): (C - p).C' w^3 = w Q.Q' - w' Q.Q with Q = P - p w has Bernstein
/// coefficients of one strict sign. Weights must be positive.
pub fn curve_no_critical_point(pw: &[[f64; 4]], p: V3) -> bool {
    let m = pw.len() - 1;
    let mk = |f: &dyn Fn(&[f64; 4]) -> f64| Bern2 { m, n: 0, c: pw.iter().map(f).collect() };
    let w = mk(&|h| h[3]);
    if w.c.iter().any(|&x| !(x > 0.0)) {
        return false;
    }
    let q = [mk(&|h| h[0] - p.x * h[3]), mk(&|h| h[1] - p.y * h[3]), mk(&|h| h[2] - p.z * h[3])];
    let qu = q.clone().map(|x| x.d_u());
    let dot = |x: &[Bern2; 3], y: &[Bern2; 3]| {
        let mut s = x[0].mul(&y[0]);
        for k in 1..3 {
            s = s.axpy(1.0, &x[k].mul(&y[k]), 1.0);
        }
        s
    };
    let qq = dot(&q, &q);
    let f = w.mul(&dot(&q, &qu)).axpy(1.0, &w.d_u().mul(&qq), -1.0);
    let qmax = |x: &[Bern2; 3]| x.iter().map(|y| y.max_abs()).fold(0.0, f64::max);
    let (q0, q1) = (qmax(&q), qmax(&qu));
    let size = w.max_abs() * 3.0 * q0 * q1 + w.d_u().max_abs() * 3.0 * q0 * q0;
    let (lo, hi) = f.c.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    lo > 1e-11 * size || hi < -1e-11 * size
}

// ---------------------------------------------------------------- radii of convexity

/// An interval.
#[derive(Clone, Copy, Debug)]
struct Iv(f64, f64);

impl Iv {
    fn of(xs: impl Iterator<Item = f64>) -> Iv {
        xs.fold(Iv(f64::INFINITY, f64::NEG_INFINITY), |a, x| Iv(a.0.min(x), a.1.max(x)))
    }
    fn add(self, o: Iv) -> Iv {
        Iv(self.0 + o.0, self.1 + o.1)
    }
    fn sub(self, o: Iv) -> Iv {
        Iv(self.0 - o.1, self.1 - o.0)
    }
    fn mul(self, o: Iv) -> Iv {
        let p = [self.0 * o.0, self.0 * o.1, self.1 * o.0, self.1 * o.1];
        Iv(p.iter().cloned().fold(f64::INFINITY, f64::min), p.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    fn scale(self, k: f64) -> Iv {
        if k >= 0.0 { Iv(self.0 * k, self.1 * k) } else { Iv(self.1 * k, self.0 * k) }
    }
    /// Divided by a positive interval.
    fn div(self, w: Iv) -> Iv {
        self.mul(Iv(1.0 / w.1, 1.0 / w.0))
    }
    fn mig(self) -> f64 {
        if self.0 <= 0.0 && self.1 >= 0.0 { 0.0 } else { self.0.abs().min(self.1.abs()) }
    }
    fn mag(self) -> f64 {
        self.0.abs().max(self.1.abs())
    }
}

type Iv3 = [Iv; 3];

fn dot3(a: &Iv3, b: &Iv3) -> Iv {
    a[0].mul(b[0]).add(a[1].mul(b[1])).add(a[2].mul(b[2]))
}

/// Least |x|^2 over a vector interval.
fn mig2(a: &Iv3) -> f64 {
    a.iter().map(|x| x.mig() * x.mig()).sum()
}

/// Margin on a radius of convexity, for the rounding of whoever compares against it (the
/// GPU kernel in single precision).
const RHO_MARGIN: f64 = 0.99;

/// Bounds over a patch (or a piece) of its derivatives, componentwise [lo, hi], and a
/// lower bound on lambda_min(G) (for a piece, on |C'|^2): a patch's S_s, S_t, S_ss, S_st,
/// S_tt, in its parameters scaled so both tangents are of unit length at its middle (any
/// such scaling keeps every test made with them); a piece's C', C'' (the rest zero).
#[derive(Clone, Copy, Debug)]
pub struct DerivBounds {
    pub lambda: f64,
    pub d: [[[f64; 2]; 3]; 5],
}

impl DerivBounds {
    /// The radius of convexity they give a patch (`Patch::convex_radius`).
    pub fn patch_radius(&self) -> f64 {
        let mag2 = |k: usize| self.d[k].iter().map(|i| i[0].abs().max(i[1].abs()).powi(2)).sum::<f64>();
        if !(self.lambda > 0.0) {
            return 0.0;
        }
        let h = (mag2(2) + 2.0 * mag2(3) + mag2(4)).sqrt();
        if h > 0.0 { RHO_MARGIN * self.lambda / h } else { f64::INFINITY }
    }

    /// The radius they give a piece (`bez_convex_radius`).
    pub fn piece_radius(&self) -> f64 {
        if !(self.lambda > 0.0) {
            return 0.0;
        }
        let h = self.d[1].iter().map(|i| i[0].abs().max(i[1].abs()).powi(2)).sum::<f64>().sqrt();
        if h > 0.0 { RHO_MARGIN * self.lambda / h } else { f64::INFINITY }
    }
}

fn bounds3(iv: &Iv3) -> [[f64; 2]; 3] {
    iv.map(|i| [i.0, i.1])
}

impl Patch {
    /// Radius of convexity: within it of every point of the patch, |S(s, t) - p|^2 is
    /// strictly convex in the parameters (one minimum on the patch), since its Hessian
    /// S_i.S_j + (S - p).S_ij has its least eigenvalue at least lambda_min(G) -
    /// |S - p| |S_ij|. The radius is lambda_min(G) / sqrt(|S_ss|^2 + 2 |S_st|^2 + |S_tt|^2)
    /// over the patch, both bounded by interval arithmetic on the Bernstein hulls of the
    /// derivatives (relative to the points' middle, so the bounds do not carry the
    /// patch's distance from the origin). Zero when G can be singular.
    pub fn convex_radius(&self) -> f64 {
        self.derivative_bounds().map_or(0.0, |b| b.patch_radius())
    }

    /// The bounds the radius is made from; None when a weight is not positive.
    pub fn derivative_bounds(&self) -> Option<DerivBounds> {
        patch_bounds(&self.pw)
    }
}

fn patch_bounds(pw: &[Vec<[f64; 4]>]) -> Option<DerivBounds> {
    let (du, dv) = (pw.len() - 1, pw[0].len() - 1);
    let pts: Vec<V3> = pw.iter().flatten().map(|h| V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])).collect();
    let c = pts.iter().fold(V3::default(), |a, &q| a + q) * (1.0 / pts.len() as f64);
    let q = |i: usize, j: usize| {
        let h = pw[i][j];
        [h[0] - c.x * h[3], h[1] - c.y * h[3], h[2] - c.z * h[3], h[3]]
    };
    // Hull of each derivative's Bernstein coefficients, component by component.
    let hull = |f: &dyn Fn(usize, usize, usize) -> Option<f64>| -> [Iv; 4] {
        std::array::from_fn(|k| {
            let mut v = Vec::new();
            for i in 0..=du {
                for j in 0..=dv {
                    if let Some(x) = f(i, j, k) {
                        v.push(x);
                    }
                }
            }
            if v.is_empty() { Iv(0.0, 0.0) } else { Iv::of(v.into_iter()) }
        })
    };
    let (fu, fv) = (du as f64, dv as f64);
    let a = hull(&|i, j, k| Some(q(i, j)[k]));
    let a_s = hull(&|i, j, k| (i < du).then(|| fu * (q(i + 1, j)[k] - q(i, j)[k])));
    let a_t = hull(&|i, j, k| (j < dv).then(|| fv * (q(i, j + 1)[k] - q(i, j)[k])));
    let a_ss = hull(&|i, j, k| (i + 1 < du).then(|| fu * (fu - 1.0) * (q(i + 2, j)[k] - 2.0 * q(i + 1, j)[k] + q(i, j)[k])));
    let a_st = hull(&|i, j, k| (i < du && j < dv).then(|| fu * fv * (q(i + 1, j + 1)[k] - q(i + 1, j)[k] - q(i, j + 1)[k] + q(i, j)[k])));
    let a_tt = hull(&|i, j, k| (j + 1 < dv).then(|| fv * (fv - 1.0) * (q(i, j + 2)[k] - 2.0 * q(i, j + 1)[k] + q(i, j)[k])));
    let w = a[3];
    if !(w.0 > 0.0) {
        return None;
    }
    let s: Iv3 = std::array::from_fn(|k| Iv::of(pts.iter().map(|p| [p.x - c.x, p.y - c.y, p.z - c.z][k])));
    let ss: Iv3 = std::array::from_fn(|k| a_s[k].sub(s[k].mul(a_s[3])).div(w));
    let st: Iv3 = std::array::from_fn(|k| a_t[k].sub(s[k].mul(a_t[3])).div(w));
    let sss: Iv3 = std::array::from_fn(|k| a_ss[k].sub(ss[k].mul(a_s[3]).scale(2.0)).sub(s[k].mul(a_ss[3])).div(w));
    let sst: Iv3 = std::array::from_fn(|k| a_st[k].sub(ss[k].mul(a_t[3])).sub(st[k].mul(a_s[3])).sub(s[k].mul(a_st[3])).div(w));
    let stt: Iv3 = std::array::from_fn(|k| a_tt[k].sub(st[k].mul(a_t[3]).scale(2.0)).sub(s[k].mul(a_tt[3])).div(w));
    // The parameters scaled so both tangents are of unit length at the middle (s' = a s,
    // t' = b t): the Hessian becomes D H D with D = diag(1 / a, 1 / b), positive definite
    // exactly when H is, and the bounds below no longer pit a long direction's curvature
    // against a short direction's metric.
    let e0 = Patch { pw: pw.to_vec(), u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }.eval(0.5, 0.5);
    let (sa, sb) = (e0.ps.len(), e0.pt.len());
    let (sa, sb) = (if sa > 0.0 { sa } else { 1.0 }, if sb > 0.0 { sb } else { 1.0 });
    let sc = |iv: &Iv3, k: f64| iv.map(|i| i.scale(1.0 / k));
    let (ss, st, sss, sst, stt) = (sc(&ss, sa), sc(&st, sb), sc(&sss, sa * sa), sc(&sst, sa * sb), sc(&stt, sb * sb));
    let (ps0, pt0) = (e0.ps * (1.0 / sa), e0.pt * (1.0 / sb));
    // lambda_min(G) two ways, the larger kept: from the entries' intervals, and by Weyl's
    // inequality from the middle, sigma_min(J) >= sigma_min(J0) - |J - J0| (J the 3 x 2
    // matrix of first derivatives; the second holds however the tangents are turned,
    // where the first loses S_s.S_t to its components' intervals).
    let (ga, gc, gb) = (mig2(&ss), mig2(&st), dot3(&ss, &st).mag());
    let by_entries = 0.5 * ((ga + gc) - ((ga - gc) * (ga - gc) + 4.0 * gb * gb).sqrt());
    let e0 = SEval { ps: ps0, pt: pt0, ..e0 };
    let (a0, b0, c0) = (e0.ps.dot(e0.ps), e0.ps.dot(e0.pt), e0.pt.dot(e0.pt));
    let sigma0 = (0.5 * ((a0 + c0) - ((a0 - c0) * (a0 - c0) + 4.0 * b0 * b0).sqrt())).max(0.0).sqrt();
    let dev = |iv: &Iv3, x: V3| [x.x, x.y, x.z].iter().zip(iv).map(|(&x, i)| (i.0 - x).abs().max((i.1 - x).abs()).powi(2)).sum::<f64>();
    let spread = (dev(&ss, e0.ps) + dev(&st, e0.pt)).sqrt();
    let by_middle = (sigma0 - spread).max(0.0).powi(2);
    let lambda = by_entries.max(by_middle);
    Some(DerivBounds { lambda, d: [bounds3(&ss), bounds3(&st), bounds3(&sss), bounds3(&sst), bounds3(&stt)] })
}

/// The same for a rational Bezier piece (homogeneous points, positive weights): |C'|^2 /
/// |C''| over it.
pub fn bez_convex_radius(pw: &[[f64; 4]]) -> f64 {
    bez_bounds(pw).map_or(0.0, |b| b.piece_radius())
}

/// The bounds a piece's radius is made from; None when a weight is not positive.
pub fn bez_bounds(pw: &[[f64; 4]]) -> Option<DerivBounds> {
    let n = pw.len() - 1;
    let pts: Vec<V3> = pw.iter().map(|h| V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])).collect();
    let c = pts.iter().fold(V3::default(), |a, &q| a + q) * (1.0 / pts.len() as f64);
    let q = |i: usize| {
        let h = pw[i];
        [h[0] - c.x * h[3], h[1] - c.y * h[3], h[2] - c.z * h[3], h[3]]
    };
    let fnn = n as f64;
    let hull = |f: &dyn Fn(usize, usize) -> Option<f64>| -> [Iv; 4] {
        std::array::from_fn(|k| {
            let v: Vec<f64> = (0..=n).filter_map(|i| f(i, k)).collect();
            if v.is_empty() { Iv(0.0, 0.0) } else { Iv::of(v.into_iter()) }
        })
    };
    let a = hull(&|i, k| Some(q(i)[k]));
    let a1 = hull(&|i, k| (i < n).then(|| fnn * (q(i + 1)[k] - q(i)[k])));
    let a2 = hull(&|i, k| (i + 1 < n).then(|| fnn * (fnn - 1.0) * (q(i + 2)[k] - 2.0 * q(i + 1)[k] + q(i)[k])));
    let w = a[3];
    if !(w.0 > 0.0) {
        return None;
    }
    let s: Iv3 = std::array::from_fn(|k| Iv::of(pts.iter().map(|p| [p.x - c.x, p.y - c.y, p.z - c.z][k])));
    let d1: Iv3 = std::array::from_fn(|k| a1[k].sub(s[k].mul(a1[3])).div(w));
    let d2: Iv3 = std::array::from_fn(|k| a2[k].sub(d1[k].mul(a1[3]).scale(2.0)).sub(s[k].mul(a2[3])).div(w));
    // |C'|^2 two ways, the larger kept: from the components' intervals, and from the
    // middle's |C'| less how far C' strays from it.
    let e0 = Bez3 { pw: pw.to_vec(), t0: 0.0, t1: 1.0 }.eval(0.5);
    let spread = [e0.d1.x, e0.d1.y, e0.d1.z].iter().zip(&d1).map(|(&x, i)| (i.0 - x).abs().max((i.1 - x).abs()).powi(2)).sum::<f64>().sqrt();
    let l = mig2(&d1).max((e0.d1.len() - spread).max(0.0).powi(2));
    Some(DerivBounds { lambda: l, d: [bounds3(&d1), bounds3(&d2), [[0.0; 2]; 3], [[0.0; 2]; 3], [[0.0; 2]; 3]] })
}

/// Lower bound on the distance from p to a patch along unit direction n: the
/// patch lies in the hull of its control points, and every hull point y has
/// (p - y).n at least the least (p - c).n over the control points. Taken along
/// the direction to the patch's Newton point, it tightens quadratically as the
/// patch flattens, where a box bound only tightens linearly. It also holds for
/// every sub-patch (their control hulls lie in the parent's).
pub fn slab_bound(pa: &Patch, p: V3, n: V3) -> f64 {
    pa.ctrl().map(|c| (p - c).dot(n)).fold(f64::INFINITY, f64::min)
}

impl Patch {
    /// The four quarters of the patch (split at the middle in u and in v).
    pub fn quarters(&self) -> [Patch; 4] {
        let (a, b) = self.split_u(0.5);
        let (a0, a1) = a.split_v(0.5);
        let (b0, b1) = b.split_v(0.5);
        [a0, a1, b0, b1]
    }
}

fn casteljau_point(pts: &[[f64; 4]], t: f64) -> [f64; 4] {
    let mut b = pts.to_vec();
    let n = b.len() - 1;
    for level in 1..=n {
        for i in 0..=(n - level) {
            for c in 0..4 {
                b[i][c] = (1.0 - t) * b[i][c] + t * b[i + 1][c];
            }
        }
    }
    b[0]
}

impl NurbsSurf {
    /// Rows ctrl[i][j] with i along u, weights alike, clamped knot vectors.
    pub fn new(du: usize, dv: usize, ctrl: &[Vec<V3>], weights: Option<&Vec<Vec<f64>>>, ku: &[f64], kv: &[f64]) -> Result<NurbsSurf, String> {
        let nu = ctrl.len();
        let nv = ctrl.first().map_or(0, |r| r.len());
        if nu + du + 1 != ku.len() || nv + dv + 1 != kv.len() || ctrl.iter().any(|r| r.len() != nv) {
            return Err(format!("B-spline surface: {nu} x {nv} control points do not match degrees {du}, {dv} and {} + {} knots", ku.len(), kv.len()));
        }
        let w = |i: usize, j: usize| weights.map_or(1.0, |w| w[i][j]);
        if (0..nu).any(|i| (0..nv).any(|j| !(w(i, j) > 0.0))) {
            return Err("B-spline surface with a weight that is not positive".into());
        }
        // Homogeneous net, clamped exactly in each direction (knot insertion leaves
        // the surface unchanged; every row takes the same insertions).
        let mut net: Vec<Vec<[f64; 4]>> = (0..nu)
            .map(|i| {
                (0..nv)
                    .map(|j| {
                        let (c, wt) = (ctrl[i][j], w(i, j));
                        [c.x * wt, c.y * wt, c.z * wt, wt]
                    })
                    .collect()
            })
            .collect();
        let mut kv2 = kv.to_vec();
        for row in net.iter_mut() {
            let (k, r) = clamp(dv, kv, row);
            kv2 = k;
            *row = r;
        }
        let nv = kv2.len() - dv - 1;
        let mut ku2 = ku.to_vec();
        let mut cols: Vec<Vec<[f64; 4]>> = (0..nv).map(|j| net.iter().map(|r| r[j]).collect()).collect();
        for col in cols.iter_mut() {
            let (k, c) = clamp(du, ku, col);
            ku2 = k;
            *col = c;
        }
        let nu = ku2.len() - du - 1;
        let (ku, kv) = (&ku2[..], &kv2[..]);
        let mut pw = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            for j in 0..nv {
                pw.push(cols[j][i]);
            }
        }
        let spans = |k: &[f64], d: usize| -> Vec<usize> { (d..k.len() - d - 1).filter(|&s| k[s] < k[s + 1]).collect() };
        let (su, sv) = (spans(ku, du), spans(kv, dv));
        let mut ns = NurbsSurf {
            du,
            dv,
            nv,
            pw,
            ku: ku.to_vec(),
            kv: kv.to_vec(),
            su,
            sv,
            nodes: Vec::new(),
            u_range: (ku[du], ku[ku.len() - du - 1]),
            v_range: (kv[dv], kv[kv.len() - dv - 1]),
            kept: None,
        };
        let spans = ns.su.len() * ns.sv.len();
        if spans * (du + 1) * (dv + 1) <= CACHE_POINTS {
            ns.kept = Some((0..spans).map(|_| std::sync::OnceLock::new()).collect());
        }
        ns.build(0, ns.su.len() as u32, 0, ns.sv.len() as u32);
        Ok(ns)
    }

    pub fn span_count(&self) -> usize {
        self.su.len() * self.sv.len()
    }

    /// Spans along u and along v.
    pub fn spans_uv(&self) -> (usize, usize) {
        (self.su.len(), self.sv.len())
    }

    /// The span indices holding (u, v).
    pub fn span_at(&self, u: f64, v: f64) -> (usize, usize) {
        (Self::span_of(&self.su, &self.ku, u), Self::span_of(&self.sv, &self.kv, v))
    }

    /// Box of the control points acting on span (a, b).
    fn span_box(&self, a: usize, b: usize) -> (V3, V3) {
        let (s, t) = (self.su[a], self.sv[b]);
        let mut lo = V3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = V3::new(f64::MIN, f64::MIN, f64::MIN);
        for i in s - self.du..=s {
            for j in t - self.dv..=t {
                let h = self.pw[i * self.nv + j];
                let q = V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
                lo = V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z));
                hi = V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z));
            }
        }
        (lo, hi)
    }

    fn build(&mut self, i0: u32, i1: u32, j0: u32, j1: u32) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(QNode { lo: V3::default(), hi: V3::default(), i0, i1, j0, j1, kids: Vec::new() });
        let (lo, hi, kids) = if i1 - i0 == 1 && j1 - j0 == 1 {
            let (lo, hi) = self.span_box(i0 as usize, j0 as usize);
            (lo, hi, Vec::new())
        } else {
            let (im, jm) = ((i0 + i1) / 2, (j0 + j1) / 2);
            let mut kids = Vec::new();
            let iranges: Vec<(u32, u32)> = if i1 - i0 > 1 { vec![(i0, im), (im, i1)] } else { vec![(i0, i1)] };
            let jranges: Vec<(u32, u32)> = if j1 - j0 > 1 { vec![(j0, jm), (jm, j1)] } else { vec![(j0, j1)] };
            for &(a, b) in &iranges {
                for &(c, d) in &jranges {
                    kids.push(self.build(a, b, c, d));
                }
            }
            let mut lo = V3::new(f64::MAX, f64::MAX, f64::MAX);
            let mut hi = V3::new(f64::MIN, f64::MIN, f64::MIN);
            for &k in &kids {
                let n = &self.nodes[k as usize];
                lo = V3::new(lo.x.min(n.lo.x), lo.y.min(n.lo.y), lo.z.min(n.lo.z));
                hi = V3::new(hi.x.max(n.hi.x), hi.y.max(n.hi.y), hi.z.max(n.hi.z));
            }
            (lo, hi, kids)
        };
        let n = &mut self.nodes[id as usize];
        n.lo = lo;
        n.hi = hi;
        n.kids = kids;
        id
    }

    /// Parameter box of a node: (u0, u1, v0, v1).
    pub fn node_params(&self, n: &QNode) -> (f64, f64, f64, f64) {
        (self.ku[self.su[n.i0 as usize]], self.ku[self.su[n.i1 as usize - 1] + 1], self.kv[self.sv[n.j0 as usize]], self.kv[self.sv[n.j1 as usize - 1] + 1])
    }

    /// The Bezier patch of span (a, b), by local knot insertion in u then v.
    pub fn patch(&self, a: usize, b: usize) -> Patch {
        self.span_patch(a, b).into_owned()
    }

    /// The patch of span (a, b), borrowed from the kept ones when the surface keeps them.
    fn span_patch(&self, a: usize, b: usize) -> std::borrow::Cow<'_, Patch> {
        match &self.kept {
            Some(k) => std::borrow::Cow::Borrowed(k[a * self.sv.len() + b].get_or_init(|| self.extract(a, b))),
            None => std::borrow::Cow::Owned(self.extract(a, b)),
        }
    }

    fn extract(&self, a: usize, b: usize) -> Patch {
        let (s, t) = (self.su[a], self.sv[b]);
        let (du, dv) = (self.du, self.dv);
        let lu = &self.ku[s - du..=s + du + 1];
        let lv = &self.kv[t - dv..=t + dv + 1];
        // Along u for each local column j, then along v for each Bezier row i.
        let cols: Vec<Vec<[f64; 4]>> = (t - dv..=t).map(|j| span_bezier(du, lu, &(s - du..=s).map(|i| self.pw[i * self.nv + j]).collect::<Vec<_>>())).collect();
        let pw: Vec<Vec<[f64; 4]>> = (0..=du).map(|i| span_bezier(dv, lv, &cols.iter().map(|c| c[i]).collect::<Vec<_>>())).collect();
        Patch { pw, u0: self.ku[s], u1: self.ku[s + 1], v0: self.kv[t], v1: self.kv[t + 1] }
    }

    /// The Bezier patch of the span holding (u, v).
    pub fn patch_at(&self, u: f64, v: f64) -> Patch {
        self.patch(Self::span_of(&self.su, &self.ku, u), Self::span_of(&self.sv, &self.kv, v))
    }

    /// Index of the span holding parameter x (the last span for the far end).
    fn span_of(spans: &[usize], k: &[f64], x: f64) -> usize {
        match spans.iter().rposition(|&s| k[s] <= x) {
            Some(i) => i,
            None => 0,
        }
    }

    /// Point and derivatives at global parameters (u, v); derivatives per unit u and v.
    pub fn eval(&self, u: f64, v: f64) -> SEval {
        let (a, b) = (Self::span_of(&self.su, &self.ku, u), Self::span_of(&self.sv, &self.kv, v));
        let pa = self.span_patch(a, b);
        let (s, t) = ((u - pa.u0) / (pa.u1 - pa.u0), (v - pa.v0) / (pa.v1 - pa.v0));
        let e = pa.eval(s.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
        let (x, y) = (1.0 / (pa.u1 - pa.u0), 1.0 / (pa.v1 - pa.v0));
        SEval { p: e.p, ps: e.ps * x, pt: e.pt * y, pss: e.pss * (x * x), pst: e.pst * (x * y), ptt: e.ptt * (y * y) }
    }

    /// Box holding the whole surface.
    pub fn bbox(&self) -> (V3, V3) {
        (self.nodes[0].lo, self.nodes[0].hi)
    }

    /// The nearest critical point of the squared distance inside a trimmed region,
    /// below `cutoff`: where the trimmed surface's nearest point lies when it is not
    /// on the region's boundary (which the caller covers by its edges). `class`
    /// tells whether a parameter box lies wholly inside the region (Some(true)),
    /// wholly outside (Some(false)) or straddles it (None); `inside` tells it of
    /// one point. Boxes wholly outside are skipped and straddling boxes are split
    /// until they are not, so a second critical point inside a box whose first lies
    /// outside is still found.
    ///
    /// Every point found inside the region bounds the search. Only a critical point
    /// is returned: a point near the boundary that is not one stands for the
    /// boundary, and its normal says nothing of which side p is on. A point that is
    /// not critical is first carried on by Newton across spans (`polish`), which
    /// ends on the critical point it was heading for, inside the region or not.
    ///
    /// A patch the critical-point test clears is not split: the face's nearest
    /// point is then not in it, or lies on the face's boundary, which the caller's
    /// edges cover, or on a C0 knot line or seam, which the caller searches along.
    /// Without the test, a nearest point on the boundary with the untrimmed surface
    /// running on past it toward p keeps every straddling patch along the boundary
    /// above the bound, and the search splits them down to the settling distance.
    /// Patches touching a collapsed side are never cleared.
    ///
    /// Returns the hit, whether the search ended at its bound (not its step cap),
    /// and the steps taken.
    pub fn closest_trimmed(&self, p: V3, cutoff: f64, class: &dyn Fn([f64; 2], [f64; 2]) -> Option<bool>, inside: &dyn Fn(f64, f64) -> bool, creases: &Creases) -> (Option<SurfHit>, bool, usize) {
        let mut best: Option<SurfHit> = None;
        let mut best_d = cutoff;
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(PatchItem { lb: box_dist(p, &self.bbox()), entry: Entry::Node(0), depth: 0, inside: false });
        let mut pops = 0;
        let answer = |h: SurfHit, best: &mut Option<SurfHit>| {
            if best.map_or(true, |b| h.dist < b.dist) {
                *best = Some(h);
            }
        };
        while let Some(PatchItem { lb, entry, depth, inside: known_in }) = heap.pop() {
            if lb >= best_d - DIST_EPS || best_d <= DIST_EPS {
                break;
            }
            pops += 1;
            if pops > 20000 {
                return (best, false, pops);
            }
            let pa = match entry {
                Entry::Node(id) => {
                    let n = &self.nodes[id as usize];
                    let (u0, u1, v0, v1) = self.node_params(n);
                    let state = if known_in { Some(true) } else { class([u0, v0], [u1, v1]) };
                    if state == Some(false) {
                        continue;
                    }
                    let now_in = state == Some(true);
                    if n.kids.is_empty() {
                        let pa = self.patch(n.i0 as usize, n.j0 as usize);
                        let lbp = box_dist(p, &pa.bbox()).max(lb);
                        if lbp < best_d - DIST_EPS {
                            heap.push(PatchItem { lb: lbp, entry: Entry::Patch(pa), depth: 0, inside: now_in });
                        }
                    } else {
                        for &k in &n.kids {
                            let c = &self.nodes[k as usize];
                            let lbk = box_dist(p, &(c.lo, c.hi));
                            if lbk < best_d - DIST_EPS {
                                heap.push(PatchItem { lb: lbk, entry: Entry::Node(k), depth: 0, inside: now_in });
                            }
                        }
                    }
                    continue;
                }
                Entry::Patch(pa) => pa,
            };
            let state = if known_in { Some(true) } else { class([pa.u0, pa.v0], [pa.u1, pa.v1]) };
            if state == Some(false) {
                continue;
            }
            let now_in = state == Some(true);
            let (s, t) = pa.newton_capped(p, 0.5, 0.5, SEARCH_NEWTON);
            let e = pa.eval(s, t);
            let d = (e.p - p).len();
            let n = e.ps.cross(e.pt).unit();
            let (u, v) = (pa.u0 + s * (pa.u1 - pa.u0), pa.v0 + t * (pa.v1 - pa.v0));
            if d < best_d && (now_in || inside(u, v)) {
                best_d = d;
                if critical(p, &e) {
                    answer(SurfHit { q: e.p, dist: d, u, v, n }, &mut best);
                } else if let Some(h) = self.polish(p, u, v) {
                    if h.dist <= d && inside(h.u, h.v) {
                        best_d = best_d.min(h.dist);
                        answer(h, &mut best);
                    }
                }
            }
            let slab = slab_bound(&pa, p, if d > 1e-12 { (p - e.p) * (1.0 / d) } else { n });
            if slab >= best_d - DIST_EPS || depth >= 40 {
                continue;
            }
            if creases.test && !creases.touches(&pa) && pa.no_critical_point(p) {
                continue;
            }
            for q in pa.quarters() {
                let lbq = box_dist(p, &q.bbox()).max(slab);
                if lbq < best_d - DIST_EPS {
                    heap.push(PatchItem { lb: lbq, entry: Entry::Patch(q), depth: depth + 1, inside: now_in });
                }
            }
        }
        (best, true, pops)
    }

    /// Newton on the squared distance from (u, v), moving to the next span when it
    /// leaves one across a side that is not the domain's, until it rests at a
    /// critical point; None if it does not within the step cap or rests on the
    /// domain's border.
    pub fn polish(&self, p: V3, u: f64, v: f64) -> Option<SurfHit> {
        self.descend(p, u, v, false)
    }

    /// The local nearest point reached from (u, v) by the same descent: a critical point,
    /// or where the descent is held at the domain's border pointing out of it (an edge
    /// along a side, or across a closed surface's seam). None if it does not settle.
    pub fn nearest_from(&self, p: V3, u: f64, v: f64) -> Option<SurfHit> {
        self.descend(p, u, v, true)
    }

    fn descend(&self, p: V3, u: f64, v: f64, border: bool) -> Option<SurfHit> {
        let (mut a, mut b) = (Self::span_of(&self.su, &self.ku, u), Self::span_of(&self.sv, &self.kv, v));
        let mut pa = self.span_patch(a, b);
        let (mut s, mut t) = ((u - pa.u0) / (pa.u1 - pa.u0), (v - pa.v0) / (pa.v1 - pa.v0));
        for _ in 0..16 {
            let (ns, nt) = pa.newton_capped(p, s.clamp(0.0, 1.0), t.clamp(0.0, 1.0), 40);
            let e = pa.eval(ns, nt);
            if critical(p, &e) {
                let n = e.ps.cross(e.pt).unit();
                return Some(SurfHit { q: e.p, dist: (e.p - p).len(), u: pa.u0 + ns * (pa.u1 - pa.u0), v: pa.v0 + nt * (pa.v1 - pa.v0), n });
            }
            // Held at a side with the descent pointing out: carry on in the span
            // beyond it.
            let r = e.p - p;
            let (gs, gt) = (r.dot(e.ps), r.dot(e.pt));
            let (mut moved, mut na, mut nb) = (false, a, b);
            if ns >= 1.0 && gs < 0.0 && a + 1 < self.su.len() {
                na = a + 1;
                s = 0.0;
                moved = true;
            } else if ns <= 0.0 && gs > 0.0 && a > 0 {
                na = a - 1;
                s = 1.0;
                moved = true;
            } else {
                s = ns;
            }
            if nt >= 1.0 && gt < 0.0 && b + 1 < self.sv.len() {
                nb = b + 1;
                t = 0.0;
                moved = true;
            } else if nt <= 0.0 && gt > 0.0 && b > 0 {
                nb = b - 1;
                t = 1.0;
                moved = true;
            } else {
                t = nt;
            }
            if !moved {
                if border {
                    let n = e.ps.cross(e.pt).unit();
                    return Some(SurfHit { q: e.p, dist: r.len(), u: pa.u0 + ns * (pa.u1 - pa.u0), v: pa.v0 + nt * (pa.v1 - pa.v0), n });
                }
                return None;
            }
            a = na;
            b = nb;
            pa = self.span_patch(a, b);
        }
        None
    }

    /// Number of nonempty knot spans along u.
    pub fn spans_u(&self) -> usize {
        self.su.len()
    }
    /// Number of nonempty knot spans along v.
    pub fn spans_v(&self) -> usize {
        self.sv.len()
    }

    /// The sides of the domain the surface collapses on to one point (a pole), as
    /// `Creases` lines: the distance's minimum at a pole need not be a critical
    /// point of a patch's parameters, so no patch with a side there is cleared.
    pub fn creases(&self) -> Creases {
        let mut out = Creases { u: Vec::new(), v: Vec::new(), test: true };
        for (k, side, _) in self.collapsed_sides() {
            if k == 0 {
                out.u.push(side);
            } else {
                out.v.push(side);
            }
        }
        out
    }

    /// Sides of the domain the surface collapses on to one point: the direction
    /// held fixed (0: u), its value, the point.
    pub fn collapsed_sides(&self) -> Vec<(usize, f64, V3)> {
        let (blo, bhi) = self.bbox();
        let scale = (bhi - blo).len().max(1e-9);
        let mut out = Vec::new();
        for k in 0..2 {
            let range = if k == 0 { self.u_range } else { self.v_range };
            let other = if k == 0 { self.v_range } else { self.u_range };
            for side in [range.0, range.1] {
                let at = |t: f64| {
                    let w = other.0 + (other.1 - other.0) * t;
                    if k == 0 { self.eval(side, w).p } else { self.eval(w, side).p }
                };
                let p0 = at(0.0);
                if (1..=16).all(|i| (at(i as f64 / 16.0) - p0).len() <= 1e-12 * scale) {
                    out.push((k, side, p0));
                }
            }
        }
        out
    }

    /// The surface's interior knot lines of multiplicity at least the degree, where
    /// it is only C0 in its parameters and may fold: per line, the direction held
    /// fixed (0: u), its value, and its rational Bezier pieces (one per span of the
    /// other direction, parameters along that direction) with the patches on either
    /// side of the piece (before and after the line).
    pub fn c0_lines(&self) -> Vec<(usize, f64, Vec<(Bez3, Patch, Patch)>)> {
        let mut out = Vec::new();
        for k in 0..2 {
            let (spans, knots, deg, range) = if k == 0 { (&self.su, &self.ku, self.du, self.u_range) } else { (&self.sv, &self.kv, self.dv, self.v_range) };
            for (a, &si) in spans.iter().enumerate() {
                let x = knots[si];
                if a == 0 || !(x > range.0 && x < range.1) || knots.iter().filter(|&&y| y == x).count() < deg {
                    continue;
                }
                let others = if k == 0 { self.sv.len() } else { self.su.len() };
                let pieces = (0..others)
                    .map(|b| {
                        let (before, after) = if k == 0 { (self.patch(a - 1, b), self.patch(a, b)) } else { (self.patch(b, a - 1), self.patch(b, a)) };
                        let (pw, t0, t1) = if k == 0 { (after.pw[0].clone(), after.v0, after.v1) } else { (after.pw.iter().map(|row| row[0]).collect(), after.u0, after.u1) };
                        (Bez3 { pw, t0, t1 }, before, after)
                    })
                    .collect();
                out.push((k, x, pieces));
            }
        }
        out
    }

    /// Certified closest point of the untrimmed surface: branch and bound down the
    /// span quadtree (box bounds) into extracted patches (box and support-plane
    /// bounds, Newton at each popped patch).
    pub fn closest(&self, p: V3, cutoff: f64) -> Option<SurfHit> {
        let mut best: Option<SurfHit> = None;
        let mut best_d = cutoff;
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(PatchItem { lb: box_dist(p, &self.bbox()), entry: Entry::Node(0), depth: 0, inside: false });
        let mut pops = 0;
        while let Some(PatchItem { lb, entry, depth, .. }) = heap.pop() {
            if lb >= best_d - DIST_EPS || best_d <= DIST_EPS {
                break;
            }
            pops += 1;
            if pops > 20000 {
                break;
            }
            let pa = match entry {
                Entry::Node(id) => {
                    let n = &self.nodes[id as usize];
                    if n.kids.is_empty() {
                        let pa = self.patch(n.i0 as usize, n.j0 as usize);
                        let lbp = box_dist(p, &pa.bbox()).max(lb);
                        if lbp < best_d - DIST_EPS {
                            heap.push(PatchItem { lb: lbp, entry: Entry::Patch(pa), depth: 0, inside: false });
                        }
                    } else {
                        for &k in &n.kids {
                            let c = &self.nodes[k as usize];
                            let lbk = box_dist(p, &(c.lo, c.hi));
                            if lbk < best_d - DIST_EPS {
                                heap.push(PatchItem { lb: lbk, entry: Entry::Node(k), depth: 0, inside: false });
                            }
                        }
                    }
                    continue;
                }
                Entry::Patch(pa) => pa,
            };
            let (s, t) = pa.newton_capped(p, 0.5, 0.5, SEARCH_NEWTON);
            let e = pa.eval(s, t);
            let d = (e.p - p).len();
            let n = e.ps.cross(e.pt).unit();
            if d < best_d {
                best_d = d;
                best = Some(SurfHit { q: e.p, dist: d, u: pa.u0 + s * (pa.u1 - pa.u0), v: pa.v0 + t * (pa.v1 - pa.v0), n });
            }
            let slab = slab_bound(&pa, p, if d > 1e-12 { (p - e.p) * (1.0 / d) } else { n });
            if slab >= best_d - DIST_EPS || depth >= 40 {
                continue;
            }
            for q in pa.quarters() {
                let lbq = box_dist(p, &q.bbox()).max(slab);
                if lbq < best_d - DIST_EPS {
                    heap.push(PatchItem { lb: lbq, entry: Entry::Patch(q), depth: depth + 1, inside: false });
                }
            }
        }
        best
    }

    /// The iso-curve at fixed v (along u, when `along_u`) or fixed u (along v),
    /// walked from parameter `from` toward `to`: exact rational Bezier pieces (each
    /// span's patch contracted at the fixed parameter; a patch row is a Bezier
    /// curve of the patch's degree), handed to `f` one span at a time in walking
    /// order, each in increasing parameter. Patches are extracted only as the walk
    /// reaches them; `f` returns false to stop.
    pub fn iso_walk(&self, along_u: bool, fixed: f64, from: f64, to: f64, f: &mut dyn FnMut(crate::curve3::Bez3) -> bool) {
        use crate::curve3::Bez3;
        let (lo, hi) = (from.min(to), from.max(to));
        let (fs, fk, rs, rk) = if along_u { (&self.sv, &self.kv, &self.su, &self.ku) } else { (&self.su, &self.ku, &self.sv, &self.kv) };
        let fixed_span = Self::span_of(fs, fk, fixed);
        let order: Vec<usize> = if to >= from { (0..rs.len()).collect() } else { (0..rs.len()).rev().collect() };
        for ri in order {
            let r = rs[ri];
            let (r0, r1) = (rk[r], rk[r + 1]);
            if r1 <= lo || r0 >= hi {
                continue;
            }
            let pa = if along_u { self.patch(ri, fixed_span) } else { self.patch(fixed_span, ri) };
            let (f0, f1) = if along_u { (pa.v0, pa.v1) } else { (pa.u0, pa.u1) };
            let t = ((fixed - f0) / (f1 - f0)).clamp(0.0, 1.0);
            let pw: Vec<[f64; 4]> = if along_u {
                pa.pw.iter().map(|row| casteljau_point(row, t)).collect()
            } else {
                let du = pa.pw.len();
                (0..pa.pw[0].len()).map(|j| casteljau_point(&(0..du).map(|i| pa.pw[i][j]).collect::<Vec<_>>(), t)).collect()
            };
            let mut z = Bez3 { pw, t0: r0, t1: r1 };
            if z.t0 < lo {
                z = z.split((lo - z.t0) / (z.t1 - z.t0)).1;
            }
            if z.t1 > hi {
                z = z.split((hi - z.t0) / (z.t1 - z.t0)).0;
            }
            if !f(z) {
                return;
            }
        }
    }

    /// The whole iso-curve between two parameters, in increasing parameter.
    pub fn iso(&self, along_u: bool, fixed: f64, from: f64, to: f64) -> Vec<crate::curve3::Bez3> {
        let mut out = Vec::new();
        self.iso_walk(along_u, fixed, from.min(to), from.max(to), &mut |z| {
            out.push(z);
            true
        });
        out
    }

    /// Average span size (for sizing boundary boxes against patches).
    pub fn span_size(&self) -> f64 {
        let (lo, hi) = self.bbox();
        (hi - lo).len() / (self.span_count() as f64).sqrt().max(1.0)
    }

    /// True when the surface closes on itself in u (or v): its first and last
    /// iso-curves coincide.
    pub fn closed(&self, in_u: bool, tol: f64) -> bool {
        let (a, b) = if in_u { self.u_range } else { self.v_range };
        (0..=8).all(|k| {
            let s = k as f64 / 8.0;
            let (p, q) = if in_u {
                let v = self.v_range.0 + s * (self.v_range.1 - self.v_range.0);
                (self.eval(a, v).p, self.eval(b, v).p)
            } else {
                let u = self.u_range.0 + s * (self.u_range.1 - self.u_range.0);
                (self.eval(u, a).p, self.eval(u, b).p)
            };
            (p - q).len() < tol
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve3::decompose_h;

    fn frame() -> Frame {
        Frame { o: V3::new(1.0, 2.0, 3.0), z: V3::new(0.0, 0.6, 0.8), x: V3::new(1.0, 0.0, 0.0) }
    }

    /// Every analytic foot is a critical point of the distance (p - q parallel to the
    /// normal) and lies on its surface.
    #[test]
    fn analytic_feet_are_critical_points() {
        let f = frame();
        let surfs = [
            Surf::Plane { f },
            Surf::Cylinder { f, r: 2.0 },
            Surf::Cone { f, r: 2.0, a: 0.5 },
            Surf::Sphere { f, r: 2.0 },
            Surf::Torus { f, major: 5.0, minor: 1.5 },
        ];
        for s in &surfs {
            for k in 0..200 {
                let x = k as f64;
                let p = V3::new(1.0 + 7.0 * (x * 0.37).sin(), 2.0 + 6.0 * (x * 0.73).cos(), 3.0 + 5.0 * (x * 1.31).sin());
                let q = s.foot(p);
                let n = s.normal(q);
                let d = p - q;
                if let Surf::Cone { .. } = s {
                    if (q - s.apex().unwrap()).len() < 1e-9 {
                        continue;
                    }
                }
                assert!(d.cross(n).len() < 1e-9 * (1.0 + d.len()), "{}: p - q not normal at {p:?}", s.name());
                assert!(s.dist(q) < 1e-12, "{}: foot off the surface", s.name());
            }
        }
    }

    /// The two parts of a self-intersecting torus (and a horn torus): the foot is no
    /// farther than any point of a dense sweep of the surface (the meridian circle's arc
    /// on the axis's own side, revolved), and no nearer than the sweep allows.
    #[test]
    fn spindle_torus_feet_beat_dense_sampling() {
        let f = frame();
        for (major, minor) in [(1.0, 2.5), (-1.0, 2.5), (2.0, 2.0), (0.3, 1.0), (-0.3, 1.0)] {
            let s = Surf::Torus { f, major, minor };
            // The surface: angle t around the meridian circle about `major`, kept where it is
            // on the axis's own side, revolved by u.
            let mut pts = Vec::new();
            for i in 0..720 {
                let t = std::f64::consts::TAU * i as f64 / 720.0;
                let (rho, h) = (major + minor * t.cos(), minor * t.sin());
                if rho < 0.0 {
                    continue;
                }
                for j in 0..360 {
                    let u = std::f64::consts::TAU * j as f64 / 360.0;
                    pts.push(f.o + (f.x * u.cos() + f.y() * u.sin()) * rho + f.z * h);
                }
            }
            let step = std::f64::consts::TAU / 360.0 * (major.abs() + minor);
            for k in 0..60 {
                let x = k as f64;
                let p = f.o + V3::new(4.0 * (x * 0.37).sin(), 4.0 * (x * 0.73).cos(), 4.0 * (x * 1.31).sin());
                let d = (p - s.foot(p)).len();
                let brute = pts.iter().map(|q| (p - *q).len()).fold(f64::INFINITY, f64::min);
                assert!(d <= brute + 1e-9, "major {major}, minor {minor}: foot {d} beyond the sweep's {brute} at {p:?}");
                assert!(d >= brute - step, "major {major}, minor {minor}: foot {d} well inside the sweep's {brute} at {p:?}");
            }
        }
    }

    /// A bicubic patch surface: the certified closest point is no farther than any
    /// point of a dense parameter sweep.
    #[test]
    fn nurbs_closest_beats_dense_sampling() {
        let mut ctrl = Vec::new();
        for i in 0..5 {
            let mut row = Vec::new();
            for j in 0..4 {
                let (x, y) = (i as f64 * 2.0, j as f64 * 2.0);
                row.push(V3::new(x, y, (x * 0.7).sin() * 2.0 + (y * 0.9).cos()));
            }
            ctrl.push(row);
        }
        let ku = [0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0];
        let kv = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let s = NurbsSurf::new(3, 3, &ctrl, None, &ku, &kv).unwrap();
        assert_eq!(s.span_count(), 2);
        for k in 0..40 {
            let x = k as f64;
            let p = V3::new(4.0 + 5.0 * (x * 0.37).sin(), 3.0 + 4.0 * (x * 0.71).cos(), 1.0 + 3.0 * (x * 1.3).sin());
            let h = s.closest(p, f64::INFINITY).unwrap();
            let mut dense = f64::INFINITY;
            for a in 0..=200 {
                for b in 0..=200 {
                    dense = dense.min((s.eval(a as f64 / 200.0, b as f64 / 200.0).p - p).len());
                }
            }
            assert!(h.dist <= dense + 1e-9, "closest {} vs dense {}", h.dist, dense);
            assert!(h.dist > dense - 0.05, "closest {} far below dense {}", h.dist, dense);
        }
    }

    /// Local extraction of every span equals the full Bezier decomposition.
    #[test]
    fn span_extraction_matches_decomposition() {
        let p = 3;
        let knots = [0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0];
        let n = knots.len() - p - 1;
        let pts: Vec<[f64; 4]> = (0..n).map(|i| {
            let w = 1.0 + 0.3 * (i as f64 * 1.7).sin();
            let x = i as f64;
            [x * w, (x * 0.9).cos() * w, (x * 0.4).sin() * w, w]
        }).collect();
        let full = decompose_h(p, &pts, &knots);
        let spans: Vec<usize> = (p..knots.len() - p - 1).filter(|&s| knots[s] < knots[s + 1]).collect();
        assert_eq!(full.len(), spans.len());
        for (k, &s) in spans.iter().enumerate() {
            let local = span_bezier(p, &knots[s - p..=s + p + 1], &pts[s - p..=s]);
            for (a, b) in local.iter().zip(&full[k].0) {
                for c in 0..4 {
                    assert!((a[c] - b[c]).abs() < 1e-12, "span {k}: {a:?} vs {b:?}");
                }
            }
        }
    }

}

