//! Exact 3D geometry for general B-rep solids: vector operators, frames, and
//! the edge curves of trimmed faces (lines, circular arcs, elliptical arcs and
//! native rational B-splines), each with an exact closest point.
//!
//! An edge is stored trimmed to its two vertices and oriented in the direction
//! its loop traverses it.

use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use crate::geom::Near;
pub use crate::step::{Frame, V3};

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f64> for V3 {
    type Output = V3;
    fn mul(self, s: f64) -> V3 {
        V3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        V3::new(-self.x, -self.y, -self.z)
    }
}

impl Frame {
    /// Coordinates of p in this frame.
    pub fn local(&self, p: V3) -> V3 {
        let d = p - self.o;
        V3::new(d.dot(self.x), d.dot(self.y()), d.dot(self.z))
    }
    /// World point from frame coordinates.
    pub fn world(&self, l: V3) -> V3 {
        self.o + self.x * l.x + self.y() * l.y + self.z * l.z
    }
    /// World direction from frame coordinates.
    pub fn dir(&self, l: V3) -> V3 {
        self.x * l.x + self.y() * l.y + self.z * l.z
    }
}

/// Angle of t0 -> t measured in the direction of `sweep`, in [0, 2 pi).
pub fn swept(t0: f64, t: f64, sweep: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let d = if sweep >= 0.0 { t - t0 } else { t0 - t };
    d.rem_euclid(two_pi)
}

// ---------------------------------------------------------------- 3D B-splines

/// A rational Bezier piece in 3D, homogeneous (x w, y w, z w, w), local s in [0, 1].
#[derive(Clone, Debug)]
pub struct Bez3 {
    pub pw: Vec<[f64; 4]>,
    pub t0: f64,
    pub t1: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Eval3 {
    pub p: V3,
    pub d1: V3,
    pub d2: V3,
}

fn hp(h: [f64; 4]) -> V3 {
    V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
}

/// Bezier pieces with at most this many control points run de Casteljau on the stack.
const STACK: usize = 16;

/// A working copy of a Bezier piece's homogeneous control points for de Casteljau,
/// which overwrites them level by level: on the stack up to STACK points, on the heap
/// past that. A closest-point search evaluates a piece many times per query, and a
/// heap copy each time was most of what a query allocated.
pub(crate) struct Scratch<const N: usize> {
    small: [[f64; N]; STACK],
    big: Vec<[f64; N]>,
    len: usize,
}

impl<const N: usize> Scratch<N> {
    pub(crate) fn of(pw: &[[f64; N]]) -> Scratch<N> {
        let mut s = Scratch { small: [[0.0; N]; STACK], big: Vec::new(), len: pw.len() };
        if pw.len() <= STACK {
            s.small[..pw.len()].copy_from_slice(pw);
        } else {
            s.big.extend_from_slice(pw);
        }
        s
    }

    pub(crate) fn points(&mut self) -> &mut [[f64; N]] {
        if self.len <= STACK {
            &mut self.small[..self.len]
        } else {
            &mut self.big
        }
    }
}

impl Bez3 {
    pub fn degree(&self) -> usize {
        self.pw.len() - 1
    }
    pub fn start(&self) -> V3 {
        hp(self.pw[0])
    }
    pub fn end(&self) -> V3 {
        hp(self.pw[self.pw.len() - 1])
    }
    pub fn ctrl(&self) -> impl Iterator<Item = V3> + '_ {
        self.pw.iter().map(|&h| hp(h))
    }
    pub fn eval(&self, s: f64) -> Eval3 {
        let n = self.degree();
        let mut scratch = Scratch::of(&self.pw);
        let b = scratch.points();
        let (mut d1h, mut d2h) = ([0.0; 4], [0.0; 4]);
        for level in 1..=n {
            if n >= 2 && level == n - 1 {
                for c in 0..4 {
                    d2h[c] = (n * (n - 1)) as f64 * (b[2][c] - 2.0 * b[1][c] + b[0][c]);
                }
            }
            if level == n {
                for c in 0..4 {
                    d1h[c] = n as f64 * (b[1][c] - b[0][c]);
                }
            }
            for i in 0..=(n - level) {
                for c in 0..4 {
                    b[i][c] = (1.0 - s) * b[i][c] + s * b[i + 1][c];
                }
            }
        }
        let w = b[0][3];
        let p = hp(b[0]);
        let d1 = (V3::new(d1h[0], d1h[1], d1h[2]) - p * d1h[3]) * (1.0 / w);
        let d2 = (V3::new(d2h[0], d2h[1], d2h[2]) - d1 * (2.0 * d1h[3]) - p * d2h[3]) * (1.0 / w);
        Eval3 { p, d1, d2 }
    }
    pub fn split(&self, s: f64) -> (Bez3, Bez3) {
        let n = self.degree();
        let mut scratch = Scratch::of(&self.pw);
        let b = scratch.points();
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
        let tm = self.t0 + s * (self.t1 - self.t0);
        (Bez3 { pw: left, t0: self.t0, t1: tm }, Bez3 { pw: right, t0: tm, t1: self.t1 })
    }
    pub fn reversed(&self) -> Bez3 {
        let mut pw = self.pw.clone();
        pw.reverse();
        Bez3 { pw, t0: self.t1, t1: self.t0 }
    }
    /// Total turning of the control polygon (radians).
    pub fn turning(&self) -> f64 {
        let c: Vec<V3> = self.ctrl().collect();
        let legs: Vec<V3> = c.windows(2).map(|w| w[1] - w[0]).filter(|v| v.len() > 1e-14).collect();
        legs.windows(2).map(|w| w[0].cross(w[1]).len().atan2(w[0].dot(w[1]))).sum()
    }
    /// Largest distance of a control point from the chord: the piece stays within it.
    pub fn thickness(&self) -> f64 {
        let (a, b) = (self.start(), self.end());
        self.ctrl().map(|q| seg_dist3(q, a, b)).fold(0.0, f64::max)
    }
    pub fn newton(&self, p: V3, mut s: f64) -> f64 {
        for _ in 0..40 {
            let e = self.eval(s);
            let r = e.p - p;
            let f = e.d1.dot(r);
            // Full Newton where the distance is convex along the curve, else Gauss-Newton
            // (a negative second derivative would step toward a maximum).
            let mut fp = e.d2.dot(r) + e.d1.dot(e.d1);
            if fp <= 0.0 {
                fp = e.d1.dot(e.d1);
            }
            if fp.abs() < 1e-300 {
                break;
            }
            let ns = (s - f / fp).clamp(0.0, 1.0);
            if (ns - s).abs() < 1e-15 {
                return ns;
            }
            s = ns;
        }
        s
    }
}

pub fn seg_dist3(p: V3, a: V3, b: V3) -> f64 {
    let ab = b - a;
    let l2 = ab.dot(ab);
    let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p - (a + ab * t)).len()
}

/// Bezier extraction of a clamped B-spline in homogeneous form by The NURBS Book
/// algorithm 5.6 (DecomposeCurve): the control points of each nonempty knot span
/// with its parameter interval. Works on any homogeneous dimension, so surfaces
/// decompose row by row and then column by column.
pub fn decompose_h<const N: usize>(degree: usize, pw: &[[f64; N]], knots: &[f64]) -> Vec<(Vec<[f64; N]>, f64, f64)> {
    let p = degree;
    let u = knots;
    let m = u.len() - 1;
    let mut out = Vec::new();
    let mut qw: Vec<[f64; N]> = pw[..=p].to_vec();
    let mut next: Vec<[f64; N]> = vec![[0.0; N]; p + 1];
    let (mut a, mut b) = (p, p + 1);
    let mut alphas = vec![0.0; p + 1];
    while b < m {
        let i = b;
        while b < m && u[b + 1] == u[b] {
            b += 1;
        }
        let mult = b - i + 1;
        if mult < p {
            let numer = u[b] - u[a];
            for j in (mult + 1..=p).rev() {
                alphas[j - mult - 1] = numer / (u[a + j] - u[a]);
            }
            let r = p - mult;
            for j in 1..=r {
                let save = r - j;
                let s = mult + j;
                for k in (s..=p).rev() {
                    let al = alphas[k - s];
                    for c in 0..N {
                        qw[k][c] = al * qw[k][c] + (1.0 - al) * qw[k - 1][c];
                    }
                }
                if b < m {
                    next[save] = qw[p];
                }
            }
        }
        out.push((qw.clone(), u[a], u[b]));
        if b < m {
            for i in (p - mult.min(p))..=p {
                next[i] = pw[b - p + i];
            }
            qw.clone_from(&next);
            a = b;
            b += 1;
        }
    }
    out.retain(|z| z.2 > z.1);
    out
}

/// Bezier pieces of a clamped rational 3D B-spline.
pub fn decompose3(degree: usize, ctrl: &[V3], weights: &[f64], knots: &[f64]) -> Vec<Bez3> {
    let pw: Vec<[f64; 4]> = ctrl.iter().zip(weights).map(|(c, &w)| [c.x * w, c.y * w, c.z * w, w]).collect();
    decompose_h(degree, &pw, knots).into_iter().map(|(pw, t0, t1)| Bez3 { pw, t0, t1 }).collect()
}

/// Inserts knot u once into a (local piece of a) B-spline: The NURBS Book
/// algorithm 5.1 for one insertion, in homogeneous coordinates. Point i has
/// support knots[i] .. knots[i + p + 1].
pub fn insert_knot<const N: usize>(p: usize, knots: &mut Vec<f64>, pts: &mut Vec<[f64; N]>, u: f64) {
    let k = (0..knots.len() - 1).rev().find(|&i| knots[i] <= u && knots[i + 1] > u).expect("knot inside the vector");
    let s = knots[..=k].iter().rev().take_while(|&&x| x == u).count();
    let mut q: Vec<[f64; N]> = Vec::with_capacity(pts.len() + 1);
    q.extend_from_slice(&pts[..=k - p]);
    for i in (k - p + 1)..=(k - s) {
        let a = (u - knots[i]) / (knots[i + p] - knots[i]);
        let mut r = [0.0; N];
        for c in 0..N {
            r[c] = a * pts[i][c] + (1.0 - a) * pts[i - 1][c];
        }
        q.push(r);
    }
    q.extend_from_slice(&pts[(k - s)..]);
    knots.insert(k + 1, u);
    *pts = q;
}


/// The same B-spline with clamped ends (first and last knots repeated degree + 1
/// times), exactly: insert each end of the domain knots[p] .. knots[m - p] until
/// it has multiplicity p, then drop the knots and points outside the domain.
/// A clamped input comes back unchanged.
pub fn clamp<const N: usize>(p: usize, knots: &[f64], pts: &[[f64; N]]) -> (Vec<f64>, Vec<[f64; N]>) {
    let (mut kn, mut pt) = (knots.to_vec(), pts.to_vec());
    let clamped = |k: &[f64]| k[..=p].iter().all(|&x| x == k[0]) && k[k.len() - p - 1..].iter().all(|&x| x == k[k.len() - 1]);
    if clamped(&kn) {
        return (kn, pt);
    }
    // Start: raise knots[p] to multiplicity p, keep the points from last_a - p on.
    let a = kn[p];
    while kn.iter().filter(|&&x| x == a).count() < p {
        insert_knot(p, &mut kn, &mut pt, a);
    }
    let last_a = kn.iter().rposition(|&x| x == a).unwrap();
    let mut k2 = vec![a; p + 1];
    k2.extend_from_slice(&kn[last_a + 1..]);
    let (mut kn, mut pt) = (k2, pt[last_a - p..].to_vec());
    // End: the mirror image.
    let m = kn.len() - 1;
    let b = kn[m - p];
    while kn.iter().filter(|&&x| x == b).count() < p {
        insert_knot(p, &mut kn, &mut pt, b);
    }
    let first_b = kn.iter().position(|&x| x == b).unwrap();
    let mut k3 = kn[..first_b].to_vec();
    k3.extend(std::iter::repeat(b).take(p + 1));
    let n = k3.len() - p - 1;
    pt.truncate(n);
    (k3, pt)
}

/// A B-spline edge: flat rational Bezier pieces in traversal order.
#[derive(Clone, Debug)]
pub struct Spline3 {
    pub pieces: Vec<Bez3>,
    pub thickness: Vec<f64>,
    pub boxes: Vec<(V3, V3)>,
}

impl Spline3 {
    /// The part of the curve between parameters ta and tb, traversed from ta to tb.
    pub fn new(degree: usize, ctrl: &[V3], weights: &[f64], knots: &[f64], ta: f64, tb: f64) -> Spline3 {
        let (lo, hi) = (ta.min(tb), ta.max(tb));
        let mut pieces = Vec::new();
        for mut z in decompose3(degree, ctrl, weights, knots) {
            if z.t1 <= lo || z.t0 >= hi {
                continue;
            }
            if z.t0 < lo {
                z = z.split((lo - z.t0) / (z.t1 - z.t0)).1;
            }
            if z.t1 > hi {
                z = z.split((hi - z.t0) / (z.t1 - z.t0)).0;
            }
            pieces.push(z);
        }
        if tb < ta {
            pieces = pieces.iter().rev().map(Bez3::reversed).collect();
        }
        let mut flat = Vec::new();
        let mut stack: Vec<Bez3> = pieces.into_iter().rev().collect();
        while let Some(z) = stack.pop() {
            if z.turning() <= 0.35 || (z.t1 - z.t0).abs() < 1e-12 {
                flat.push(z);
            } else {
                let (l, r) = z.split(0.5);
                stack.push(r);
                stack.push(l);
            }
        }
        let thickness = flat.iter().map(Bez3::thickness).collect();
        let boxes = flat
            .iter()
            .map(|z| {
                let (mut lo, mut hi) = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
                for q in z.ctrl() {
                    lo = V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z));
                    hi = V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z));
                }
                (lo, hi)
            })
            .collect();
        Spline3 { pieces: flat, thickness, boxes }
    }

    /// This curve followed by another that starts where it ends (pieces that are
    /// empty, at the seam of a closed curve, are dropped).
    pub fn join(mut self, other: Spline3) -> Spline3 {
        for ((z, t), b) in other.pieces.into_iter().zip(other.thickness).zip(other.boxes) {
            self.pieces.push(z);
            self.thickness.push(t);
            self.boxes.push(b);
        }
        self
    }

    /// Parameter (on the source curve) of the point of the full curve nearest p;
    /// used to find the parameters of an edge's vertices.
    pub fn param_of(&self, p: V3) -> f64 {
        self.closest(p, f64::INFINITY).map(|f| f.t).unwrap_or(0.0)
    }

    /// Certified closest point (branch and bound over the flat pieces, Newton polish).
    pub fn closest(&self, p: V3, cutoff: f64) -> Option<Foot3> {
        let mut best: Option<Foot3> = None;
        let mut best_d = cutoff;
        let mut stack: Vec<(f64, usize, Bez3, u32)> = Vec::new();
        for (i, z) in self.pieces.iter().enumerate() {
            let (lo, hi) = self.boxes[i];
            let d = V3::new((lo.x - p.x).max(p.x - hi.x).max(0.0), (lo.y - p.y).max(p.y - hi.y).max(0.0), (lo.z - p.z).max(p.z - hi.z).max(0.0)).len();
            let lb = d.max(seg_dist3(p, z.start(), z.end()) - self.thickness[i]);
            if lb < best_d {
                stack.push((lb, i, z.clone(), 0));
            }
        }
        stack.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut popped = 0;
        while let Some((lb, i, z, depth)) = stack.pop() {
            if lb >= best_d - 1e-12 {
                continue;
            }
            popped += 1;
            if popped > 4096 {
                break;
            }
            let (a, b) = (z.start(), z.end());
            let ab = b - a;
            let l2 = ab.dot(ab);
            let f = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.5 };
            for s0 in [f, 0.0, 1.0, -1.0, 2.0] {
                // -1 and 2 stand for the raw ends, without Newton.
                let s = if s0 < 0.0 {
                    0.0
                } else if s0 > 1.0 {
                    1.0
                } else {
                    z.newton(p, s0)
                };
                let e = z.eval(s);
                let d = (p - e.p).len();
                if d < best_d {
                    best_d = d;
                    let last = i == self.pieces.len() - 1 && s >= 1.0 && z.t1 == self.pieces[i].t1;
                    let first = i == 0 && s <= 0.0 && z.t0 == self.pieces[i].t0;
                    let near = if first {
                        Near::Start
                    } else if last {
                        Near::End
                    } else {
                        Near::Interior
                    };
                    best = Some(Foot3 { q: e.p, dist: d, t: z.t0 + s * (z.t1 - z.t0), near, tangent: e.d1.unit() });
                }
            }
            if depth < 40 && z.thickness() > 1e-12 {
                let (l, r) = z.split(0.5);
                for c in [l, r] {
                    let lbc = seg_dist3(p, c.start(), c.end()) - c.thickness();
                    if lbc < best_d - 1e-12 {
                        stack.push((lbc, i, c, depth + 1));
                    }
                }
                stack.sort_by(|a, b| b.0.total_cmp(&a.0));
            }
        }
        best
    }
}

// ---------------------------------------------------------------- edge curves

/// Where a point's foot landed on an edge.
#[derive(Clone, Copy, Debug)]
pub struct Foot3 {
    pub q: V3,
    pub dist: f64,
    /// Curve parameter of the foot (line: fraction; arc and ellipse: angle; spline: source parameter).
    pub t: f64,
    pub near: Near,
    /// Unit tangent in the direction of travel.
    pub tangent: V3,
}

#[derive(Clone, Debug)]
pub enum Edge3 {
    Line { a: V3, b: V3 },
    /// P(t) = o + r (cos t x + sin t y), t from t0 through the signed sweep.
    Arc { f: Frame, r: f64, t0: f64, sweep: f64 },
    /// P(t) = o + a cos t x + b sin t y, t from t0 through the signed sweep.
    Ellipse { f: Frame, a: f64, b: f64, t0: f64, sweep: f64 },
    Spline(Arc<Spline3>),
}

impl Edge3 {
    /// Point at fraction s in [0, 1] of the traversal (for sampling pcurves).
    pub fn at(&self, s: f64) -> V3 {
        match self {
            Edge3::Line { a, b } => *a + (*b - *a) * s,
            Edge3::Arc { f, r, t0, sweep } => {
                let t = t0 + s * sweep;
                f.world(V3::new(r * t.cos(), r * t.sin(), 0.0))
            }
            Edge3::Ellipse { f, a, b, t0, sweep } => {
                let t = t0 + s * sweep;
                f.world(V3::new(a * t.cos(), b * t.sin(), 0.0))
            }
            Edge3::Spline(sp) => {
                let n = sp.pieces.len() as f64;
                let k = ((s * n) as usize).min(sp.pieces.len() - 1);
                sp.pieces[k].eval(s * n - k as f64).p
            }
        }
    }
    pub fn start(&self) -> V3 {
        self.at(0.0)
    }
    pub fn end(&self) -> V3 {
        self.at(1.0)
    }

    pub fn closest(&self, p: V3) -> Foot3 {
        match self {
            Edge3::Line { a, b } => {
                let ab = *b - *a;
                let l2 = ab.dot(ab);
                let t = if l2 > 0.0 { ((p - *a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
                let q = *a + ab * t;
                let near = if t <= 0.0 {
                    Near::Start
                } else if t >= 1.0 {
                    Near::End
                } else {
                    Near::Interior
                };
                Foot3 { q, dist: (p - q).len(), t, near, tangent: ab.unit() }
            }
            Edge3::Arc { f, r, t0, sweep } => {
                let l = f.local(p);
                let full = (sweep.abs() - 2.0 * std::f64::consts::PI).abs() < 1e-12;
                let t = l.y.atan2(l.x);
                let tan = |t: f64| f.dir(V3::new(-t.sin(), t.cos(), 0.0)) * sweep.signum();
                if (l.x != 0.0 || l.y != 0.0) && (full || swept(*t0, t, *sweep) <= sweep.abs()) {
                    let q = f.world(V3::new(r * t.cos(), r * t.sin(), 0.0));
                    return Foot3 { q, dist: (p - q).len(), t, near: Near::Interior, tangent: tan(t) };
                }
                let (qa, qb) = (self.start(), self.end());
                if (p - qa).len() <= (p - qb).len() {
                    Foot3 { q: qa, dist: (p - qa).len(), t: *t0, near: Near::Start, tangent: tan(*t0) }
                } else {
                    Foot3 { q: qb, dist: (p - qb).len(), t: t0 + sweep, near: Near::End, tangent: tan(t0 + sweep) }
                }
            }
            Edge3::Ellipse { f, a, b, t0, sweep } => {
                // In-plane problem: critical points of |E(t) - p|^2 (at most four), found by
                // Newton from 64 seeds over the arc, plus the two ends.
                let l = f.local(p);
                let e = |t: f64| V3::new(a * t.cos(), b * t.sin(), 0.0);
                let g = |t: f64| {
                    let (s, c) = t.sin_cos();
                    ((b * b - a * a) * s * c + a * l.x * s - b * l.y * c, (b * b - a * a) * (c * c - s * s) + a * l.x * c + b * l.y * s)
                };
                let d = |t: f64| (e(t) - l).len();
                let mut best = (d(*t0), *t0, Near::Start);
                let te = t0 + sweep;
                if d(te) < best.0 {
                    best = (d(te), te, Near::End);
                }
                for k in 0..64 {
                    let mut t = t0 + sweep * (k as f64 + 0.5) / 64.0;
                    for _ in 0..30 {
                        let (gv, gd) = g(t);
                        if gd.abs() < 1e-300 {
                            break;
                        }
                        let nt = t - gv / gd;
                        if (nt - t).abs() < 1e-15 {
                            t = nt;
                            break;
                        }
                        t = nt;
                    }
                    if swept(*t0, t, *sweep) <= sweep.abs() && d(t) < best.0 {
                        best = (d(t), t, Near::Interior);
                    }
                }
                let t = best.1;
                let tan = f.dir(V3::new(-a * t.sin(), b * t.cos(), 0.0)).unit() * sweep.signum();
                Foot3 { q: f.world(e(t)), dist: best.0, t, near: best.2, tangent: tan }
            }
            Edge3::Spline(sp) => sp.closest(p, f64::INFINITY).expect("spline has a closest point"),
        }
    }

    /// The point nearest p, when nearer than `cutoff` (a B-spline's search stops at
    /// the cutoff instead of finding a farther point).
    pub fn closest_below(&self, p: V3, cutoff: f64) -> Option<Foot3> {
        match self {
            Edge3::Spline(sp) => sp.closest(p, cutoff),
            _ => Some(self.closest(p)).filter(|f| f.dist < cutoff),
        }
    }
}

// ---------------------------------------------------------------- plane crossings

/// A point where an edge crosses a plane n.x = c.
#[derive(Clone, Copy, Debug)]
pub struct Cross {
    pub q: V3,
    /// Unit tangent in the direction of travel.
    pub tangent: V3,
    /// Within eps of one of the edge's vertices.
    pub at_vertex: bool,
    /// The plane touches or contains the edge here instead of crossing it.
    pub tangential: bool,
}

/// Roots of A + B cos t + C sin t on the arc t0 .. t0 + sweep.
fn trig_roots(a: f64, b: f64, c: f64, t0: f64, sweep: f64, eps_rel: f64) -> Vec<(f64, bool)> {
    let r = b.hypot(c);
    if r < 1e-300 {
        return Vec::new();
    }
    let k = -a / r;
    if k.abs() > 1.0 + eps_rel {
        return Vec::new();
    }
    let phi = c.atan2(b);
    let w = k.clamp(-1.0, 1.0).acos();
    let tangential = w < 1e-7;
    let roots: &[f64] = if tangential { &[0.0] } else { &[1.0, -1.0] };
    roots
        .iter()
        .map(|sg| phi + sg * w)
        .filter(|&t| {
            let full = (sweep.abs() - 2.0 * std::f64::consts::PI).abs() < 1e-12;
            full || swept(t0, t, sweep) <= sweep.abs() + 1e-12
        })
        .map(|t| (t, tangential))
        .collect()
}

impl Edge3 {
    /// Frame, semi-axes, start angle and signed sweep of an arc or ellipse.
    pub fn conic(&self) -> Option<(Frame, f64, f64, f64, f64)> {
        match self {
            Edge3::Arc { f, r, t0, sweep } => Some((*f, *r, *r, *t0, *sweep)),
            Edge3::Ellipse { f, a, b, t0, sweep } => Some((*f, *a, *b, *t0, *sweep)),
            _ => None,
        }
    }

    /// Unit tangent (direction of travel) at fraction s of the traversal.
    pub fn tangent_at(&self, s: f64) -> V3 {
        match self {
            Edge3::Line { a, b } => (*b - *a).unit(),
            Edge3::Arc { f, t0, sweep, .. } => {
                let t = t0 + s * sweep;
                f.dir(V3::new(-t.sin(), t.cos(), 0.0)) * sweep.signum()
            }
            Edge3::Ellipse { f, a, b, t0, sweep } => {
                let t = t0 + s * sweep;
                f.dir(V3::new(-a * t.sin(), b * t.cos(), 0.0)).unit() * sweep.signum()
            }
            Edge3::Spline(sp) => {
                let n = sp.pieces.len() as f64;
                let k = ((s * n) as usize).min(sp.pieces.len() - 1);
                sp.pieces[k].eval(s * n - k as f64).d1.unit()
            }
        }
    }

    /// Axis-aligned box holding the edge.
    pub fn bbox(&self) -> (V3, V3) {
        let mut lo = V3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = V3::new(f64::MIN, f64::MIN, f64::MIN);
        let mut add = |q: V3| {
            lo = V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z));
            hi = V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z));
        };
        match self {
            Edge3::Line { a, b } => {
                add(*a);
                add(*b);
            }
            Edge3::Arc { f, r, .. } => {
                // The whole circle's box: loose for short arcs, never wrong.
                for ax in [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)] {
                    let e = (1.0 - f.z.dot(ax).powi(2)).max(0.0).sqrt() * r;
                    add(f.o + ax * e);
                    add(f.o - ax * e);
                }
            }
            Edge3::Ellipse { f, a, b, .. } => {
                for ax in [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)] {
                    let e = (a * f.x.dot(ax)).hypot(b * f.y().dot(ax));
                    add(f.o + ax * e);
                    add(f.o - ax * e);
                }
            }
            Edge3::Spline(sp) => {
                for (l, h) in &sp.boxes {
                    add(*l);
                    add(*h);
                }
            }
        }
        (lo, hi)
    }

    /// Every crossing of the edge with the plane n.x = c (n unit). Exact: closed
    /// form for lines, arcs and ellipses; for B-splines, root isolation on the
    /// Bernstein coefficients of n.P(s) - c (their sign variations bound the
    /// number of roots), then bisection.
    pub fn plane_crossings(&self, n: V3, c: f64, eps: f64, out: &mut Vec<Cross>) {
        let (s0, s1) = (self.start(), self.end());
        let mk = |q: V3, tangent: V3, tangential: bool| Cross {
            q,
            tangent,
            at_vertex: (q - s0).len() < eps || (q - s1).len() < eps,
            tangential: tangential || n.dot(tangent).abs() < 1e-9,
        };
        match self {
            Edge3::Line { a, b } => {
                let (g0, g1) = (n.dot(*a) - c, n.dot(*b) - c);
                if g0.abs() < eps && g1.abs() < eps {
                    out.push(mk((*a + *b) * 0.5, (*b - *a).unit(), true));
                } else if g0 * g1 <= 0.0 {
                    let s = g0 / (g0 - g1);
                    out.push(mk(*a + (*b - *a) * s, (*b - *a).unit(), false));
                }
            }
            Edge3::Arc { .. } | Edge3::Ellipse { .. } => {
                let (f, ra, rb, t0, sweep) = self.conic().unwrap();
                let a0 = n.dot(f.o) - c;
                let (b0, c0) = (ra * n.dot(f.x), rb * n.dot(f.y()));
                if b0.hypot(c0) < eps && a0.abs() < eps {
                    out.push(mk(self.at(0.5), self.tangent_at(0.5), true));
                    return;
                }
                for (t, tg) in trig_roots(a0, b0, c0, t0, sweep, eps / ra.max(rb)) {
                    let q = f.world(V3::new(ra * t.cos(), rb * t.sin(), 0.0));
                    let tan = f.dir(V3::new(-ra * t.sin(), rb * t.cos(), 0.0)).unit() * sweep.signum();
                    out.push(mk(q, tan, tg));
                }
            }
            Edge3::Spline(sp) => {
                for z in &sp.pieces {
                    bez_plane_roots(z, n, c, 0, &mut |q, d1, tangential| out.push(mk(q, d1.unit(), tangential)));
                }
            }
        }
    }
}

/// Roots of n.P(s) - c on one rational Bezier piece. The numerator's Bernstein
/// coefficients are w_i (n.P_i - c); with positive weights the number of their
/// sign changes bounds the number of roots, and one sign change means exactly one.
fn bez_plane_roots(z: &Bez3, n: V3, c: f64, depth: u32, found: &mut dyn FnMut(V3, V3, bool)) {
    let cf: Vec<f64> = z.pw.iter().map(|h| n.x * h[0] + n.y * h[1] + n.z * h[2] - c * h[3]).collect();
    let pos = |x: f64| x >= 0.0;
    let changes = cf.windows(2).filter(|w| pos(w[0]) != pos(w[1])).count();
    if changes == 0 {
        return;
    }
    if changes == 1 && pos(cf[0]) != pos(cf[cf.len() - 1]) {
        let g = |s: f64| {
            let e = z.eval(s).p;
            n.dot(e) - c
        };
        let (mut lo, mut hi) = (0.0, 1.0);
        let glo = pos(g(0.0));
        for _ in 0..80 {
            let m = 0.5 * (lo + hi);
            if pos(g(m)) == glo {
                lo = m;
            } else {
                hi = m;
            }
        }
        let e = z.eval(0.5 * (lo + hi));
        found(e.p, e.d1, false);
        return;
    }
    if depth > 50 {
        // A double root (the plane touches the curve): report it as tangential.
        let e = z.eval(0.5);
        found(e.p, e.d1, true);
        return;
    }
    let (l, r) = z.split(0.5);
    bez_plane_roots(&l, n, c, depth + 1, found);
    bez_plane_roots(&r, n, c, depth + 1, found);
}

// ---------------------------------------------------------------- from STEP

impl Edge3 {
    /// The edge of a STEP loop, trimmed to its vertices and oriented along the
    /// loop. Also returns how far its vertices sit from the curve.
    pub fn from_step(e: &crate::step::OEdge) -> Result<(Edge3, f64), String> {
        use crate::step::Curve3;
        let two_pi = 2.0 * std::f64::consts::PI;
        let closed = e.start_vertex == e.end_vertex;
        let sweep_of = |ta: f64, tb: f64| match (e.forward, closed) {
            (true, true) => two_pi,
            (false, true) => -two_pi,
            (true, false) => (tb - ta).rem_euclid(two_pi),
            (false, false) => -(ta - tb).rem_euclid(two_pi),
        };
        match &e.curve {
            Curve3::Line { p, dir } => {
                let off = [e.start, e.end].iter().map(|&q| (q - *p).cross(dir.unit()).len()).fold(0.0, f64::max);
                Ok((Edge3::Line { a: e.start, b: e.end }, off))
            }
            Curve3::Circle { frame, r } => {
                let ang = |q: V3| {
                    let l = frame.local(q);
                    l.y.atan2(l.x)
                };
                let (ta, tb) = (ang(e.start), ang(e.end));
                let arc = Edge3::Arc { f: *frame, r: *r, t0: ta, sweep: sweep_of(ta, tb) };
                let off = [e.start, e.end].iter().map(|&q| arc.closest(q).dist).fold(0.0, f64::max);
                Ok((arc, off))
            }
            Curve3::Ellipse { frame, a, b } => {
                let ang = |q: V3| {
                    let l = frame.local(q);
                    (l.y / b).atan2(l.x / a)
                };
                let (ta, tb) = (ang(e.start), ang(e.end));
                let el = Edge3::Ellipse { f: *frame, a: *a, b: *b, t0: ta, sweep: sweep_of(ta, tb) };
                let off = [e.start, e.end].iter().map(|&q| el.closest(q).dist).fold(0.0, f64::max);
                Ok((el, off))
            }
            Curve3::BSpline(bc) => {
                let p = bc.degree;
                let nw = vec![1.0; bc.ctrl.len()];
                let w0 = bc.weights.as_ref().unwrap_or(&nw);
                if bc.ctrl.len() + p + 1 != bc.knots.len() {
                    return Err(format!("edge #{}: B-spline with {} control points and {} knots at degree {p}", e.edge, bc.ctrl.len(), bc.knots.len()));
                }
                if w0.iter().any(|&x| !(x > 0.0)) {
                    return Err(format!("edge #{}: B-spline weight that is not positive", e.edge));
                }
                // Clamp exactly (knot insertion), in homogeneous form.
                let hw: Vec<[f64; 4]> = bc.ctrl.iter().zip(w0).map(|(c, &x)| [c.x * x, c.y * x, c.z * x, x]).collect();
                let (kc, hc) = clamp(p, &bc.knots, &hw);
                let ctrl: Vec<V3> = hc.iter().map(|h| V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])).collect();
                let wts: Vec<f64> = hc.iter().map(|h| h[3]).collect();
                let (k, w) = (&kc, &wts);
                let bc = crate::step::BSplineCurve { degree: p, ctrl, weights: None, knots: kc.clone() };
                let (d0, d1) = (k[p], k[k.len() - p - 1]);
                let (ta, tb, off) = if closed {
                    let full = Spline3::new(p, &bc.ctrl, w, k, d0, d1);
                    let off = (full.pieces[0].start() - e.start).len();
                    if e.forward {
                        (d0, d1, off)
                    } else {
                        (d1, d0, off)
                    }
                } else {
                    let full = Spline3::new(p, &bc.ctrl, w, k, d0, d1);
                    let mut hs = full.closest(e.start, f64::INFINITY).unwrap();
                    let mut ht = full.closest(e.end, f64::INFINITY).unwrap();
                    // A curve whose ends meet to within a millionth of its size is closed:
                    // a vertex where they meet may project to either end, so it takes the
                    // end the edge's sense calls for.
                    let (c0, c1) = (full.pieces[0].start(), full.pieces[full.pieces.len() - 1].end());
                    let (lo, hi) = bc.ctrl.iter().fold((c0, c0), |(a, b), q| (V3::new(a.x.min(q.x), a.y.min(q.y), a.z.min(q.z)), V3::new(b.x.max(q.x), b.y.max(q.y), b.z.max(q.z))));
                    let gap = (c0 - c1).len();
                    let ends_meet = gap < 1e-9 * (1.0 + c0.len()) + 1e-6 * (hi - lo).len();
                    if ends_meet {
                        let at_join = |q: V3, off: f64| (q - c0).len().min((q - c1).len()) <= gap + 2.0 * off + 1e-9 * (1.0 + q.len());
                        if at_join(e.start, hs.dist) {
                            hs.t = if e.forward { d0 } else { d1 };
                        }
                        if at_join(e.end, ht.dist) {
                            ht.t = if e.forward { d1 } else { d0 };
                        }
                    }
                    if (hs.t < ht.t) != e.forward {
                        // On a closed curve the edge may run through the curve's own
                        // start: then it is the two stretches on either side, joined there.
                        if !ends_meet {
                            return Err(format!("edge #{}: B-spline edge runs against its sense on an open curve", e.edge));
                        }
                        let (a, b) = if e.forward {
                            (Spline3::new(p, &bc.ctrl, w, k, hs.t, d1), Spline3::new(p, &bc.ctrl, w, k, d0, ht.t))
                        } else {
                            (Spline3::new(p, &bc.ctrl, w, k, hs.t, d0), Spline3::new(p, &bc.ctrl, w, k, d1, ht.t))
                        };
                        return Ok((Edge3::Spline(Arc::new(a.join(b))), hs.dist.max(ht.dist)));
                    }
                    (hs.t, ht.t, hs.dist.max(ht.dist))
                };
                Ok((Edge3::Spline(Arc::new(Spline3::new(p, &bc.ctrl, w, k, ta, tb))), off))
            }
            other => Err(format!("edge #{} is a {}", e.edge, crate::prismatic::curve_name(other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        Frame { o: V3::new(1.0, -2.0, 0.5), z: V3::new(0.0, 0.6, 0.8), x: V3::new(1.0, 0.0, 0.0) }
    }

    fn sample_min(e: &Edge3, p: V3) -> f64 {
        (0..=20000).map(|i| (e.at(i as f64 / 20000.0) - p).len()).fold(f64::INFINITY, f64::min)
    }

    fn edges() -> Vec<Edge3> {
        let f = frame();
        let ctrl = [V3::new(0.0, 0.0, 0.0), V3::new(2.0, 3.0, 1.0), V3::new(5.0, -1.0, 2.0), V3::new(7.0, 2.0, -1.0), V3::new(9.0, 0.0, 0.0)];
        let knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let w = [1.0, 0.7, 1.3, 0.9, 1.0];
        vec![
            Edge3::Line { a: V3::new(0.0, 0.0, 0.0), b: V3::new(3.0, 4.0, 5.0) },
            Edge3::Arc { f, r: 3.0, t0: 0.4, sweep: 4.0 },
            Edge3::Arc { f, r: 3.0, t0: 0.4, sweep: -2.5 },
            Edge3::Ellipse { f, a: 4.0, b: 1.5, t0: -1.0, sweep: 3.5 },
            Edge3::Spline(Arc::new(Spline3::new(3, &ctrl, &w, &knots, 0.1, 0.9))),
            Edge3::Spline(Arc::new(Spline3::new(3, &ctrl, &w, &knots, 0.9, 0.2))),
        ]
    }

    #[test]
    fn closest_points_match_dense_sampling() {
        for e in edges() {
            for k in 0..60 {
                let x = k as f64;
                let p = V3::new(3.0 + 6.0 * (x * 0.37).sin(), 1.0 + 5.0 * (x * 0.71).cos(), 1.0 + 4.0 * (x * 1.3).sin());
                let c = e.closest(p);
                let s = sample_min(&e, p);
                assert!(c.dist <= s + 1e-9 && c.dist > s - 1e-3, "{e:?}: closest {} vs sampled {s}", c.dist);
                assert!((c.q - p).len() - c.dist < 1e-12);
            }
        }
    }

    #[test]
    fn plane_crossings_match_sign_changes() {
        for e in edges() {
            for k in 0..40 {
                let x = k as f64;
                let n = V3::new((x * 0.9).sin(), (x * 0.4).cos(), 0.5).unit();
                let c = n.dot(e.at(0.5)) + 0.7 * (x * 1.7).sin();
                let mut hits = Vec::new();
                e.plane_crossings(n, c, 1e-9, &mut hits);
                let m = 20000;
                let changes = (0..m).filter(|&i| (n.dot(e.at(i as f64 / m as f64)) - c >= 0.0) != (n.dot(e.at((i + 1) as f64 / m as f64)) - c >= 0.0)).count();
                assert_eq!(hits.iter().filter(|h| !h.tangential).count(), changes, "{} plane {k}", edge_kind_dbg(&e));
                for h in &hits {
                    assert!((n.dot(h.q) - c).abs() < 1e-9, "crossing off the plane by {} (tangential {}) on {}", n.dot(h.q) - c, h.tangential, edge_kind_dbg(&e));
                }
            }
        }
    }
}
#[cfg(test)]
fn edge_kind_dbg(e: &Edge3) -> String {
    match e {
        Edge3::Line { .. } => "line".into(),
        Edge3::Arc { sweep, .. } => format!("arc sweep {sweep}"),
        Edge3::Ellipse { .. } => "ellipse".into(),
        Edge3::Spline(s) => format!("spline t {} -> {}", s.pieces[0].t0, s.pieces[s.pieces.len() - 1].t1),
    }
}

// ---------------------------------------------------------------- rational Bezier form

impl Edge3 {
    /// The edge as rational Bezier pieces in traversal order, exactly: a line is one
    /// linear piece, an arc or ellipse is rational quadratics of at most a quarter
    /// turn each (the ellipse as the affine image of the circle construction),
    /// a B-spline keeps its pieces. Parameters t0, t1 are traversal fractions,
    /// except for B-splines, which keep their own.
    pub fn bez_pieces(&self) -> Vec<Bez3> {
        let h = |p: V3, w: f64| [p.x * w, p.y * w, p.z * w, w];
        match self {
            Edge3::Line { a, b } => vec![Bez3 { pw: vec![h(*a, 1.0), h(*b, 1.0)], t0: 0.0, t1: 1.0 }],
            Edge3::Arc { .. } | Edge3::Ellipse { .. } => {
                let (f, ra, rb, t0, sweep) = self.conic().unwrap();
                let k = (sweep.abs() / (0.5 * std::f64::consts::PI)).ceil().max(1.0) as usize;
                let dt = sweep / k as f64;
                let phi = 0.5 * dt;
                let pt = |t: f64, s: f64| f.world(V3::new(ra * t.cos() * s, rb * t.sin() * s, 0.0));
                (0..k)
                    .map(|i| {
                        let ta = t0 + dt * i as f64;
                        let w = phi.cos();
                        Bez3 {
                            pw: vec![h(pt(ta, 1.0), 1.0), h(pt(ta + phi, 1.0 / w), w), h(pt(ta + dt, 1.0), 1.0)],
                            t0: i as f64 / k as f64,
                            t1: (i + 1) as f64 / k as f64,
                        }
                    })
                    .collect()
            }
            Edge3::Spline(sp) => sp.pieces.clone(),
        }
    }
}

/// Largest control-polygon turning (radians) of a piece treated as flat.
pub const FLAT: f64 = 0.35;

/// Pieces split in halves until each turns at most FLAT.
pub fn flatten(pieces: &[Bez3]) -> Vec<Bez3> {
    let mut out = Vec::new();
    let mut stack: Vec<Bez3> = pieces.iter().rev().cloned().collect();
    while let Some(z) = stack.pop() {
        if z.turning() <= FLAT || (z.t1 - z.t0).abs() < 1e-12 {
            out.push(z);
        } else {
            let (l, r) = z.split(0.5);
            stack.push(r);
            stack.push(l);
        }
    }
    out
}

pub fn bez_box(z: &Bez3) -> (V3, V3) {
    let mut lo = V3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = V3::new(f64::MIN, f64::MIN, f64::MIN);
    for q in z.ctrl() {
        lo = V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z));
        hi = V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z));
    }
    (lo, hi)
}

pub fn box_gap(a: &(V3, V3), b: &(V3, V3)) -> f64 {
    let g = |l1: f64, h1: f64, l2: f64, h2: f64| (l1 - h2).max(l2 - h1).max(0.0);
    V3::new(g(a.0.x, a.1.x, b.0.x, b.1.x), g(a.0.y, a.1.y, b.0.y, b.1.y), g(a.0.z, a.1.z, b.0.z, b.1.z)).len()
}

/// A place where two curves come within a threshold of each other.
#[derive(Clone, Copy, Debug)]
pub struct NearHit {
    /// Parameter on the first curve (its pieces' t0 .. t1 scale).
    pub ta: f64,
    /// The point on the second curve.
    pub q: V3,
    pub tan_a: V3,
    pub tan_b: V3,
    pub dist: f64,
}

/// Parameters of the closest points of segments p0 + s d1 and q0 + t d2, from
/// aa = d1.d1, bb = d1.d2, cc = d2.d2, dd = d1.r, ee = d2.r with r = p0 - q0.
fn seg_seg(aa: f64, bb: f64, cc: f64, dd: f64, ee: f64) -> (f64, f64) {
    let den = aa * cc - bb * bb;
    let mut s = if den > 1e-14 * aa * cc { ((bb * ee - cc * dd) / den).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = if cc > 0.0 { (bb * s + ee) / cc } else { 0.0 };
    if t < 0.0 {
        t = 0.0;
        s = if aa > 0.0 { (-dd / aa).clamp(0.0, 1.0) } else { 0.0 };
    } else if t > 1.0 {
        t = 1.0;
        s = if aa > 0.0 { ((bb - dd) / aa).clamp(0.0, 1.0) } else { 0.0 };
    }
    (s, t)
}

/// Every place where Bezier pieces a and b come within `thr` of each other:
/// branch and bound on the control-point boxes (each piece lies in its box),
/// splitting the larger piece, with Newton on |A(s) - B(t)|^2 once both are flat.
pub static NEAR_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Returns true, and stops, on a tangential hit (the curves meet at under about
/// 3 degrees or run along each other): a walk that grazes an edge decides nothing.
pub fn near_hits(a: &Bez3, b: &Bez3, thr: f64, depth: u32, out: &mut Vec<NearHit>) -> bool {
    NEAR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (ba, bb) = (bez_box(a), bez_box(b));
    if box_gap(&ba, &bb) > thr {
        return false;
    }
    if a.turning() <= FLAT && b.turning() <= FLAT {
        // Closest points of the two chords: each piece lies within its thickness of
        // its chord, so chord distance minus both thicknesses bounds the curves'
        // distance from below (and tightens quadratically, unlike boxes).
        let (p0, p1, q0, q1) = (a.start(), a.end(), b.start(), b.end());
        let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
        let (aa, bb2, cc, dd, ee) = (d1.dot(d1), d1.dot(d2), d2.dot(d2), d1.dot(r), d2.dot(r));
        let (sc, tc) = seg_seg(aa, bb2, cc, dd, ee);
        let dseg = (p0 + d1 * sc - (q0 + d2 * tc)).len();
        let (tha, thb) = (a.thickness(), b.thickness());
        if dseg - tha - thb > thr {
            return false;
        }
        // Nearly parallel and within reach: the curves run along each other, which a
        // crossing test cannot use; report it as tangential and stop.
        if d1.cross(d2).len() < 1e-3 * (aa * cc).sqrt() && dseg - tha - thb < 2.0 * thr {
            let e = b.eval(tc);
            out.push(NearHit { ta: a.t0 + sc * (a.t1 - a.t0), q: e.p, tan_a: d1.unit(), tan_b: d1.unit(), dist: dseg });
            return true;
        }
        let (mut s, mut t) = (sc, tc);
        for _ in 0..40 {
            let (ea, eb) = (a.eval(s), b.eval(t));
            let r = ea.p - eb.p;
            let g = [ea.d1.dot(r), -eb.d1.dot(r)];
            let h = [ea.d1.dot(ea.d1) + ea.d2.dot(r), -ea.d1.dot(eb.d1), eb.d1.dot(eb.d1) - eb.d2.dot(r)];
            let (h0, h1, h2) = if h[0] > 0.0 && h[0] * h[2] - h[1] * h[1] > 0.0 { (h[0], h[1], h[2]) } else { (ea.d1.dot(ea.d1), -ea.d1.dot(eb.d1), eb.d1.dot(eb.d1)) };
            let det = h0 * h2 - h1 * h1;
            if det.abs() < 1e-300 {
                break;
            }
            let ns = (s - (h2 * g[0] - h1 * g[1]) / det).clamp(0.0, 1.0);
            let nt = (t - (h0 * g[1] - h1 * g[0]) / det).clamp(0.0, 1.0);
            let moved = (ns - s).abs() + (nt - t).abs();
            s = ns;
            t = nt;
            if moved < 1e-15 {
                break;
            }
        }
        let (ea, eb) = (a.eval(s), b.eval(t));
        let dist = (ea.p - eb.p).len();
        if dist <= thr {
            let (ta, tb) = (ea.d1.unit(), eb.d1.unit());
            out.push(NearHit { ta: a.t0 + s * (a.t1 - a.t0), q: eb.p, tan_a: ta, tan_b: tb, dist });
            return ta.cross(tb).len() < 0.05;
        }
    }
    if depth >= 48 {
        return false;
    }
    let da = (ba.1 - ba.0).len();
    let db = (bb.1 - bb.0).len();
    if da >= db {
        let (l, r) = a.split(0.5);
        near_hits(&l, b, thr, depth + 1, out) || near_hits(&r, b, thr, depth + 1, out)
    } else {
        let (l, r) = b.split(0.5);
        near_hits(a, &l, thr, depth + 1, out) || near_hits(a, &r, thr, depth + 1, out)
    }
}
