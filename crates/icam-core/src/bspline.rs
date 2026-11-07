//! Native rational B-spline curves in the plane.
//!
//! A B-spline edge stays a B-spline all the way through: exact evaluation, exact
//! first and second derivatives, and a certified closest point. It is never
//! replaced by lines or arcs; the post processor is the only place allowed to
//! lower it.
//!
//! Internally a curve is split into rational Bezier pieces (Piegl and Tiller,
//! The NURBS Book, 2nd ed., Algorithm A5.6 DecomposeCurve, transcribed from the
//! page image in the nurbs-refinement skill) and the pieces are subdivided until
//! each is "flat": its control polygon turns less than FLAT_TURN in total. A
//! piece with positive weights lies inside the convex hull of its projected
//! control points, so the largest distance of a control point from the piece's
//! chord (its `thickness`) bounds the curve's distance from the chord; that
//! gives a certified lower bound on the distance from any point to the piece.
//! Everything is homogeneous (x w, y w, w) until the last step (project last).

use crate::curve3::Scratch;
use crate::geom::V2;

/// Total turning of a flat piece's control polygon (radians).
const FLAT_TURN: f64 = 0.35;

/// A clamped rational B-spline curve in the plane.
#[derive(Clone, Debug)]
pub struct Nurbs2 {
    pub degree: usize,
    pub ctrl: Vec<V2>,
    pub weights: Vec<f64>,
    pub knots: Vec<f64>,
}

/// A rational Bezier piece: homogeneous control points (x w, y w, w), local
/// parameter s in [0, 1], mapped to the source curve's parameter t0..t1 (t1 < t0
/// when the piece runs against the curve's direction).
#[derive(Clone, Debug)]
pub struct Bez2 {
    pub pw: Vec<[f64; 3]>,
    pub t0: f64,
    pub t1: f64,
}

/// Point and derivatives in the local parameter s.
#[derive(Clone, Copy, Debug)]
pub struct Eval {
    pub p: V2,
    pub d1: V2,
    pub d2: V2,
}

impl Nurbs2 {
    pub fn new(degree: usize, ctrl: Vec<V2>, weights: Option<Vec<f64>>, knots: Vec<f64>) -> Result<Nurbs2, String> {
        let n = ctrl.len();
        let weights = weights.unwrap_or_else(|| vec![1.0; n]);
        if degree == 0 || n < degree + 1 {
            return Err(format!("B-spline of degree {degree} with {n} control points"));
        }
        if knots.len() != n + degree + 1 {
            return Err(format!("B-spline knot count {} != control points {n} + degree {degree} + 1", knots.len()));
        }
        if weights.len() != n {
            return Err(format!("B-spline has {} weights for {n} control points", weights.len()));
        }
        if weights.iter().any(|&w| !(w > 0.0)) {
            return Err("B-spline with a zero or negative weight (infinite control point) is not supported".into());
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            return Err("B-spline knots decrease".into());
        }
        // Unclamped ends are clamped exactly by knot insertion (same curve).
        let hw: Vec<[f64; 3]> = ctrl.iter().zip(&weights).map(|(c, &w)| [c.x * w, c.y * w, w]).collect();
        let (knots, hc) = crate::curve3::clamp(degree, &knots, &hw);
        let ctrl = hc.iter().map(|h| V2::new(h[0] / h[2], h[1] / h[2])).collect();
        let weights = hc.iter().map(|h| h[2]).collect();
        Ok(Nurbs2 { degree, ctrl, weights, knots })
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.knots.len() - 1 - self.degree])
    }

    /// Bezier pieces by Algorithm A5.6 (DecomposeCurve), in homogeneous form.
    pub fn decompose(&self) -> Vec<Bez2> {
        let p = self.degree;
        let u = &self.knots;
        let m = u.len() - 1;
        let pw: Vec<[f64; 3]> = self.ctrl.iter().zip(&self.weights).map(|(c, &w)| [c.x * w, c.y * w, w]).collect();
        let mut out: Vec<Bez2> = Vec::new();
        let mut qw: Vec<[f64; 3]> = pw[..=p].to_vec();
        let mut next: Vec<[f64; 3]> = vec![[0.0; 3]; p + 1];
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
                        for c in 0..3 {
                            qw[k][c] = al * qw[k][c] + (1.0 - al) * qw[k - 1][c];
                        }
                    }
                    if b < m {
                        next[save] = qw[p];
                    }
                }
            }
            out.push(Bez2 { pw: qw.clone(), t0: u[a], t1: u[b] });
            if b < m {
                for i in (p - mult.min(p))..=p {
                    next[i] = pw[b - p + i];
                }
                qw.clone_from(&next);
                a = b;
                b += 1;
            }
        }
        out.retain(|z| z.t1 > z.t0);
        out
    }
}

impl Bez2 {
    pub fn degree(&self) -> usize {
        self.pw.len() - 1
    }

    /// Projected control points.
    pub fn ctrl(&self) -> Vec<V2> {
        self.ctrl_points().collect()
    }

    /// The same, without collecting them (the bounds read them once per subdivision).
    fn ctrl_points(&self) -> impl Iterator<Item = V2> + '_ {
        self.pw.iter().map(|h| V2::new(h[0] / h[2], h[1] / h[2]))
    }

    pub fn start(&self) -> V2 {
        let h = self.pw[0];
        V2::new(h[0] / h[2], h[1] / h[2])
    }

    pub fn end(&self) -> V2 {
        let h = self.pw[self.pw.len() - 1];
        V2::new(h[0] / h[2], h[1] / h[2])
    }

    /// Point, first and second derivative at local s (de Casteljau in homogeneous
    /// space, then the quotient rule).
    pub fn eval(&self, s: f64) -> Eval {
        let n = self.degree();
        let mut scratch = Scratch::of(&self.pw);
        let b = scratch.points();
        let (mut d1h, mut d2h) = ([0.0; 3], [0.0; 3]);
        // Before step `level`, b holds the n + 2 - level points of level - 1: the three
        // points two steps from the end give the second derivative, the two points one
        // step from the end give the first.
        for level in 1..=n {
            if n >= 2 && level == n - 1 {
                // b[0..=2] are the three points of level n-2 -> second derivative.
                for c in 0..3 {
                    d2h[c] = (n * (n - 1)) as f64 * (b[2][c] - 2.0 * b[1][c] + b[0][c]);
                }
            }
            if level == n {
                // b[0..=1] are the two points of level n-1 -> first derivative.
                for c in 0..3 {
                    d1h[c] = n as f64 * (b[1][c] - b[0][c]);
                }
            }
            for i in 0..=(n - level) {
                for c in 0..3 {
                    b[i][c] = (1.0 - s) * b[i][c] + s * b[i + 1][c];
                }
            }
        }
        let a = b[0];
        let w = a[2];
        let p = V2::new(a[0] / w, a[1] / w);
        let d1 = V2::new((d1h[0] - d1h[2] * p.x) / w, (d1h[1] - d1h[2] * p.y) / w);
        let d2 = V2::new(
            (d2h[0] - 2.0 * d1h[2] * d1.x - d2h[2] * p.x) / w,
            (d2h[1] - 2.0 * d1h[2] * d1.y - d2h[2] * p.y) / w,
        );
        Eval { p, d1, d2 }
    }

    /// Splits at local s into two pieces (de Casteljau).
    pub fn split(&self, s: f64) -> (Bez2, Bez2) {
        let n = self.degree();
        let mut scratch = Scratch::of(&self.pw);
        let b = scratch.points();
        let mut left = Vec::with_capacity(n + 1);
        let mut right = Vec::with_capacity(n + 1);
        left.push(b[0]);
        right.push(b[n]);
        for level in 1..=n {
            for i in 0..=(n - level) {
                for c in 0..3 {
                    b[i][c] = (1.0 - s) * b[i][c] + s * b[i + 1][c];
                }
            }
            left.push(b[0]);
            right.push(b[n - level]);
        }
        right.reverse();
        let tm = self.t0 + s * (self.t1 - self.t0);
        (Bez2 { pw: left, t0: self.t0, t1: tm }, Bez2 { pw: right, t0: tm, t1: self.t1 })
    }

    pub fn reversed(&self) -> Bez2 {
        let mut pw = self.pw.clone();
        pw.reverse();
        Bez2 { pw, t0: self.t1, t1: self.t0 }
    }

    /// Total turning of the control polygon (radians).
    pub fn turning(&self) -> f64 {
        let c = self.ctrl();
        let legs: Vec<V2> = c.windows(2).map(|w| w[1] - w[0]).filter(|v| v.len() > 1e-14).collect();
        legs.windows(2).map(|w| w[0].cross(w[1]).atan2(w[0].dot(w[1])).abs()).sum()
    }

    /// Largest distance of a projected control point from the chord segment:
    /// the whole piece lies within this distance of the chord.
    pub fn thickness(&self) -> f64 {
        let (a, b) = (self.start(), self.end());
        self.ctrl_points().map(|q| seg_dist(q, a, b)).fold(0.0, f64::max)
    }

    /// Certified lower bound on the distance from p to any point of the piece.
    pub fn lower_bound(&self, p: V2) -> f64 {
        let c0 = self.start();
        let (mut lo, mut hi) = (c0, c0);
        for q in self.ctrl_points() {
            lo = V2::new(lo.x.min(q.x), lo.y.min(q.y));
            hi = V2::new(hi.x.max(q.x), hi.y.max(q.y));
        }
        let dx = (lo.x - p.x).max(p.x - hi.x).max(0.0);
        let dy = (lo.y - p.y).max(p.y - hi.y).max(0.0);
        let bbox = (dx * dx + dy * dy).sqrt();
        bbox.max(seg_dist(p, self.start(), self.end()) - self.thickness())
    }

    /// Newton on f(s) = C'(s) . (C(s) - p) from `s`, kept in [0, 1].
    pub fn newton(&self, p: V2, mut s: f64) -> f64 {
        for _ in 0..40 {
            let e = self.eval(s);
            let r = e.p - p;
            let f = e.d1.dot(r);
            // Gauss-Newton where the second derivative is negative (full Newton would
            // step toward a maximum there).
            let mut fp = e.d2.dot(r) + e.d1.dot(e.d1);
            if fp <= 0.0 {
                fp = e.d1.dot(e.d1);
            }
            if fp.abs() < 1e-300 {
                break;
            }
            let ns = (s - f / fp).clamp(0.0, 1.0);
            if (ns - s).abs() < 1e-15 {
                s = ns;
                break;
            }
            s = ns;
        }
        s
    }
}

pub fn seg_dist(p: V2, a: V2, b: V2) -> f64 {
    let ab = b - a;
    let l2 = ab.dot(ab);
    let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p - (a + ab * t)).len()
}

/// A B-spline edge ready for queries: flat rational Bezier pieces in traversal
/// order, each with local s running in the direction of travel.
#[derive(Clone, Debug)]
pub struct Spline2 {
    pub source: Nurbs2,
    pub pieces: Vec<Bez2>,
    pub thickness: Vec<f64>,
    /// Control-point bounding box per piece (contains the piece).
    pub boxes: Vec<(V2, V2)>,
}

fn ctrl_box(z: &Bez2) -> (V2, V2) {
    let mut lo = V2::new(f64::MAX, f64::MAX);
    let mut hi = V2::new(f64::MIN, f64::MIN);
    for h in &z.pw {
        let q = V2::new(h[0] / h[2], h[1] / h[2]);
        lo = V2::new(lo.x.min(q.x), lo.y.min(q.y));
        hi = V2::new(hi.x.max(q.x), hi.y.max(q.y));
    }
    (lo, hi)
}

fn box_dist(p: V2, (lo, hi): (V2, V2)) -> f64 {
    let dx = (lo.x - p.x).max(p.x - hi.x).max(0.0);
    let dy = (lo.y - p.y).max(p.y - hi.y).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

/// Does the ray p + t dir (t > 0) meet the box? (slab test)
fn ray_hits_box(p: V2, dir: V2, (lo, hi): (V2, V2)) -> bool {
    let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
    for (o, d, a, b) in [(p.x, dir.x, lo.x, hi.x), (p.y, dir.y, lo.y, hi.y)] {
        if d.abs() < 1e-300 {
            if o < a || o > b {
                return false;
            }
        } else {
            let (ta, tb) = ((a - o) / d, (b - o) / d);
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
        }
    }
    t0 <= t1
}

/// Where the closest point landed on a spline.
#[derive(Clone, Copy, Debug)]
pub struct SplineHit {
    pub q: V2,
    pub dist: f64,
    pub piece: usize,
    pub s: f64,
    /// Parameter on the source NURBS curve.
    pub t: f64,
    /// Unit tangent in the direction of travel, and signed curvature (positive = turning left).
    pub tangent: V2,
    pub curvature: f64,
}

impl Spline2 {
    /// The part of `source` between parameters `ta` and `tb`, traversed from ta to tb
    /// (tb < ta runs against the curve's direction).
    pub fn new(source: Nurbs2, ta: f64, tb: f64) -> Spline2 {
        let (lo, hi) = (ta.min(tb), ta.max(tb));
        let mut pieces = Vec::new();
        for z in source.decompose() {
            if z.t1 <= lo || z.t0 >= hi {
                continue;
            }
            let mut z = z;
            if z.t0 < lo {
                let s = (lo - z.t0) / (z.t1 - z.t0);
                z = z.split(s).1;
            }
            if z.t1 > hi {
                let s = (hi - z.t0) / (z.t1 - z.t0);
                z = z.split(s).0;
            }
            pieces.push(z);
        }
        if tb < ta {
            pieces = pieces.iter().rev().map(Bez2::reversed).collect();
        }
        // Subdivide until every piece is flat.
        let mut flat = Vec::new();
        let mut stack: Vec<Bez2> = pieces.into_iter().rev().collect();
        while let Some(z) = stack.pop() {
            if z.turning() <= FLAT_TURN || (z.t1 - z.t0).abs() < 1e-12 {
                flat.push(z);
            } else {
                let (l, r) = z.split(0.5);
                stack.push(r);
                stack.push(l);
            }
        }
        let thickness = flat.iter().map(Bez2::thickness).collect();
        let boxes = flat.iter().map(ctrl_box).collect();
        Spline2 { source, pieces: flat, thickness, boxes }
    }

    pub fn start(&self) -> V2 {
        self.pieces[0].start()
    }

    pub fn end(&self) -> V2 {
        self.pieces[self.pieces.len() - 1].end()
    }

    /// The same points traversed the other way.
    pub fn reversed(&self) -> Spline2 {
        let pieces: Vec<Bez2> = self.pieces.iter().rev().map(Bez2::reversed).collect();
        let thickness = self.thickness.iter().rev().copied().collect();
        let boxes = self.boxes.iter().rev().copied().collect();
        Spline2 { source: self.source.clone(), pieces, thickness, boxes }
    }

    /// Splits where the tangent turns by more than `angle` radians between
    /// consecutive pieces (a full-multiplicity interior knot makes a corner), so
    /// every corner becomes a vertex with its own pseudo-normal.
    pub fn split_at_kinks(self, angle: f64) -> Vec<Spline2> {
        let mut out = Vec::new();
        let mut cur: Vec<usize> = vec![0];
        for i in 1..self.pieces.len() {
            let a = self.pieces[i - 1].eval(1.0).d1;
            let b = self.pieces[i].eval(0.0).d1;
            if a.cross(b).atan2(a.dot(b)).abs() > angle {
                out.push(std::mem::take(&mut cur));
            }
            cur.push(i);
        }
        out.push(cur);
        if out.len() == 1 {
            return vec![self];
        }
        out.into_iter()
            .map(|idx| Spline2 {
                source: self.source.clone(),
                pieces: idx.iter().map(|&i| self.pieces[i].clone()).collect(),
                thickness: idx.iter().map(|&i| self.thickness[i]).collect(),
                boxes: idx.iter().map(|&i| self.boxes[i]).collect(),
            })
            .collect()
    }

    pub fn tangent_start(&self) -> V2 {
        self.pieces[0].eval(0.0).d1.unit()
    }

    pub fn tangent_end(&self) -> V2 {
        let z = &self.pieces[self.pieces.len() - 1];
        z.eval(1.0).d1.unit()
    }

    fn hit(&self, piece: usize, s: f64, p: V2) -> SplineHit {
        let z = &self.pieces[piece];
        let e = z.eval(s);
        let sp = e.d1.len();
        let tangent = e.d1 * (1.0 / sp.max(1e-300));
        let curvature = e.d1.cross(e.d2) / (sp * sp * sp).max(1e-300);
        SplineHit { q: e.p, dist: (p - e.p).len(), piece, s, t: z.t0 + s * (z.t1 - z.t0), tangent, curvature }
    }

    /// Certified closest point: branch and bound over the flat pieces with their
    /// hull lower bounds, subdividing until no piece can beat the best point by
    /// more than 1e-12 mm, then Newton to the exact foot. Returns None when no
    /// point is closer than `cutoff`.
    pub fn closest(&self, p: V2, cutoff: f64) -> Option<SplineHit> {
        use std::cmp::Ordering;
        #[derive(PartialEq)]
        struct Item(f64, usize, f64, f64, u32); // lower bound, piece, s0, s1, depth
        impl Eq for Item {}
        impl PartialOrd for Item {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Item {
            fn cmp(&self, o: &Self) -> Ordering {
                o.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
            }
        }
        let mut best: Option<SplineHit> = None;
        let mut best_d = cutoff;
        let mut heap = std::collections::BinaryHeap::new();
        for (i, z) in self.pieces.iter().enumerate() {
            // Allocation-free bound from the stored box and thickness.
            let bd = box_dist(p, self.boxes[i]);
            if bd >= best_d {
                continue;
            }
            let lb = bd.max(seg_dist(p, z.start(), z.end()) - self.thickness[i]);
            if lb < best_d {
                heap.push(Item(lb, i, 0.0, 1.0, 0));
            }
        }
        // The proof stops after MAX_ITEMS sub-pieces. The answer is still an actual
        // point on the curve found by Newton; only the proof of optimality is cut
        // short, which matters only when a long stretch of curve is almost equally
        // far (a point near the center of a circular arc).
        const MAX_ITEMS: usize = 4096;
        let mut popped = 0;
        while let Some(Item(lb, i, s0, s1, depth)) = heap.pop() {
            if lb >= best_d - 1e-12 || popped >= MAX_ITEMS {
                break;
            }
            popped += 1;
            // Candidate: Newton from the chord projection of this sub-range.
            let z = &self.pieces[i];
            let sub = sub_piece(z, s0, s1);
            let (a, b) = (sub.start(), sub.end());
            let ab = b - a;
            let l2 = ab.dot(ab);
            let f = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.5 };
            for s in [s0 + f * (s1 - s0), s0, s1] {
                let sn = z.newton(p, s);
                let h = self.hit(i, sn, p);
                if h.dist < best_d {
                    best_d = h.dist;
                    best = Some(h);
                }
            }
            if depth < 48 && (s1 - s0) > 1e-14 {
                let sm = 0.5 * (s0 + s1);
                for (a0, a1) in [(s0, sm), (sm, s1)] {
                    let lbc = sub_piece(z, a0, a1).lower_bound(p);
                    if lbc < best_d - 1e-12 {
                        heap.push(Item(lbc, i, a0, a1, depth + 1));
                    }
                }
            }
        }
        best
    }

    /// Crossings of the ray p + t dir (t > 0) with the spline, by subdivision
    /// until pieces are straight to 1e-11 mm (then a segment test, end excluded).
    pub fn ray_crossings(&self, p: V2, dir: V2) -> u32 {
        fn rec(z: &Bez2, p: V2, dir: V2, depth: u32) -> u32 {
            let c = z.ctrl();
            let nrm = dir.perp();
            let side: Vec<f64> = c.iter().map(|q| (*q - p).dot(nrm)).collect();
            if side.iter().all(|&s| s > 0.0) || side.iter().all(|&s| s < 0.0) {
                return 0;
            }
            if c.iter().all(|q| (*q - p).dot(dir) <= 0.0) {
                return 0;
            }
            if z.thickness() < 1e-11 || depth > 60 {
                let (a, b) = (z.start(), z.end());
                let e = b - a;
                let den = dir.cross(e);
                if den == 0.0 {
                    return 0;
                }
                let w = a - p;
                let t = w.cross(e) / den;
                let s = w.cross(dir) / den;
                return u32::from(t > 0.0 && (0.0..1.0).contains(&s));
            }
            let (l, r) = z.split(0.5);
            rec(&l, p, dir, depth + 1) + rec(&r, p, dir, depth + 1)
        }
        self.pieces.iter().zip(&self.boxes).filter(|(_, b)| ray_hits_box(p, dir, **b)).map(|(z, _)| rec(z, p, dir, 0)).sum()
    }

    /// Integral of x dy along the spline (Green's theorem area term), by 16-point
    /// Gauss-Legendre per flat piece: exact for polynomial pieces up to degree 16,
    /// and to rounding for the smooth rational ones after flattening.
    pub fn area_term(&self) -> f64 {
        const X: [f64; 8] = [
            0.0950125098376374,
            0.2816035507792589,
            0.4580167776572274,
            0.6178762444026438,
            0.7554044083550030,
            0.8656312023878318,
            0.9445750230732326,
            0.9894009349916499,
        ];
        const W: [f64; 8] = [
            0.1894506104550685,
            0.1826034150449236,
            0.1691565193950025,
            0.1495959888165767,
            0.1246289712555339,
            0.0951585116824928,
            0.0622535239386479,
            0.0271524594117541,
        ];
        let mut sum = 0.0;
        for z in &self.pieces {
            for k in 0..8 {
                for sgn in [-1.0, 1.0] {
                    let s = 0.5 + 0.5 * sgn * X[k];
                    let e = z.eval(s);
                    sum += 0.5 * W[k] * e.p.x * e.d1.y;
                }
            }
        }
        sum
    }
}

fn sub_piece(z: &Bez2, s0: f64, s1: f64) -> Bez2 {
    if s0 <= 0.0 && s1 >= 1.0 {
        return z.clone();
    }
    let right = if s0 > 0.0 { z.split(s0).1 } else { z.clone() };
    if s1 >= 1.0 {
        return right;
    }
    right.split((s1 - s0) / (1.0 - s0)).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit circle as a rational quadratic NURBS (nine points, weights 1 and 1/sqrt 2).
    fn circle(r: f64, c: V2) -> Nurbs2 {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let pts = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (1.0, 0.0)];
        let ctrl = pts.iter().map(|&(x, y)| c + V2::new(x, y) * r).collect();
        let weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
        let knots = vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0];
        Nurbs2::new(2, ctrl, Some(weights), knots).unwrap()
    }

    #[test]
    fn rational_circle_is_exact() {
        let c = V2::new(3.0, -2.0);
        let sp = Spline2::new(circle(5.0, c), 0.0, 1.0);
        for k in 0..50 {
            let t = k as f64 / 50.0;
            let s = sp.pieces.iter().find(|z| t >= z.t0.min(z.t1) - 1e-12 && t <= z.t0.max(z.t1) + 1e-12).unwrap();
            let e = s.eval((t - s.t0) / (s.t1 - s.t0));
            assert!(((e.p - c).len() - 5.0).abs() < 1e-12, "point off circle by {}", ((e.p - c).len() - 5.0).abs());
        }
        // Closest point from outside and inside, and curvature 1/r.
        for p in [V2::new(12.0, 1.0), V2::new(4.0, -1.0), V2::new(-9.0, -7.5)] {
            let h = sp.closest(p, f64::INFINITY).unwrap();
            assert!((h.dist - ((p - c).len() - 5.0).abs()).abs() < 1e-11, "dist {} vs {}", h.dist, ((p - c).len() - 5.0).abs());
            assert!((h.curvature - 0.2).abs() < 1e-9, "curvature {}", h.curvature);
        }
        // Area by Green: pi r^2.
        let a = sp.area_term();
        assert!((a - std::f64::consts::PI * 25.0).abs() < 1e-9, "area {a}");
    }

    #[test]
    fn cubic_closest_matches_dense_sampling() {
        let ctrl = vec![V2::new(0.0, 0.0), V2::new(1.0, 3.0), V2::new(4.0, 3.0), V2::new(5.0, -1.0), V2::new(7.0, 2.0), V2::new(9.0, 0.0)];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0];
        let nb = Nurbs2::new(3, ctrl, None, knots).unwrap();
        let sp = Spline2::new(nb.clone(), 0.0, 1.0);
        // Dense sample of the curve.
        let mut samples = Vec::new();
        for z in &sp.pieces {
            for k in 0..=4000 {
                samples.push(z.eval(k as f64 / 4000.0).p);
            }
        }
        for p in [V2::new(2.0, 1.0), V2::new(6.0, 4.0), V2::new(-1.0, -1.0), V2::new(4.5, 0.5), V2::new(8.0, -3.0)] {
            let h = sp.closest(p, f64::INFINITY).unwrap();
            let dense = samples.iter().map(|&q| (p - q).len()).fold(f64::MAX, f64::min);
            assert!(h.dist <= dense + 1e-12, "{p:?}: {} > sampled {}", h.dist, dense);
            assert!(dense - h.dist < 1e-5, "{p:?}: {} much below sampled {}", h.dist, dense);
        }
        // Reversed traversal gives the same set of points.
        let rev = Spline2::new(nb, 1.0, 0.0);
        assert!((rev.start() - sp.end()).len() < 1e-14 && (rev.end() - sp.start()).len() < 1e-14);
        let p = V2::new(3.0, 2.0);
        assert!((rev.closest(p, f64::INFINITY).unwrap().dist - sp.closest(p, f64::INFINITY).unwrap().dist).abs() < 1e-12);
    }

    /// An unclamped uniform cubic equals its exact clamping at every parameter.
    #[test]
    fn clamping_keeps_the_curve() {
        let ctrl: Vec<V2> = (0..7).map(|i| V2::new(i as f64, ((i * i) % 5) as f64)).collect();
        let knots: Vec<f64> = (0..11).map(|i| i as f64).collect();
        // Reference: de Boor by Cox-de Boor basis on the original knots.
        let basis = |i: usize, p: usize, t: f64| -> f64 {
            fn n(k: &[f64], i: usize, p: usize, t: f64) -> f64 {
                if p == 0 {
                    return if k[i] <= t && t < k[i + 1] { 1.0 } else { 0.0 };
                }
                let a = if k[i + p] > k[i] { (t - k[i]) / (k[i + p] - k[i]) * n(k, i, p - 1, t) } else { 0.0 };
                let b = if k[i + p + 1] > k[i + 1] { (k[i + p + 1] - t) / (k[i + p + 1] - k[i + 1]) * n(k, i + 1, p - 1, t) } else { 0.0 };
                a + b
            }
            n(&knots, i, p, t)
        };
        let nb = Nurbs2::new(3, ctrl.clone(), None, knots.clone()).unwrap();
        let sp = Spline2::new(nb, 3.0, 7.0);
        for k in 0..=40 {
            let t = 3.0 + 4.0 * k as f64 / 40.0 * 0.999999;
            let mut p = V2::new(0.0, 0.0);
            for (i, c) in ctrl.iter().enumerate() {
                p = p + *c * basis(i, 3, t);
            }
            let h = sp.closest(p, f64::INFINITY).unwrap();
            assert!(h.dist < 1e-9, "t {t}: clamped curve misses the original by {}", h.dist);
        }
    }
}
