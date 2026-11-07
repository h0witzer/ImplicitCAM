//! A B-spline face's trim loops as curves in its surface's parameter domain, built
//! once at import and certified against the file's own edges, and the inside test
//! on them.
//!
//! The files carry no parameter-domain curves, so each loop is constructed: every
//! exact piece of each edge is sampled, the samples' surface parameters are found by
//! the certified closest-point search, and a cubic curve is fitted through them in
//! the domain, with its ends shared exactly with its neighbors (consecutive edges
//! meet at one 2D point). A fitted piece is kept only when a bound proves it: with
//! the surface's span patch P/W and the piece c(t), the composition S(c(t)) is a
//! polynomial in t whose Bernstein coefficients bound its distance from the edge
//! e(t) over the whole piece (no sampling in the proof). The edge itself sits off
//! the surface by up to the file's tolerance, which no curve on the surface can
//! beat, so the bound must sit under the tolerance plus the edge's own offset (the
//! largest sampled distance of the edge from the surface, which never exceeds the
//! true one). A piece that cannot be certified is split in halves and refitted; one
//! that crosses a knot line is split where it crosses, so each piece lies in one
//! span. A face that still fails is refused with the edge and the bound.
//!
//! The inside test is crossing parity: pieces are split until each is monotone in
//! both parameters, and a point's answer is a known reference's answer flipped once
//! per crossing of the path between them (one vertical leg, one horizontal leg),
//! with the usual half-open rule at piece ends, so a corner on the path counts once.
//! No tangent, corner or nearest-point decision enters: a cusp, where two curves meet
//! at a 180 degree turn, decides nothing. A tree over the domain, built at import,
//! holds in each cell a reference point with its answer and the pieces entering the
//! cell; references are settled cell by cell from the root's, whose answer comes
//! from outside the domain (or, when both directions are periodic, from a point
//! offset from one piece whose nearest loop point is proven to be that piece).
//! Periodic directions are handled by shifted copies of the pieces, so seams need no
//! special case. Each query is one descent and a few bounded root solves.

use crate::curve3::{Bez3, Frame, V3};
use crate::surface3::{axial, NurbsSurf, Patch, Surf};

/// One edge use of a loop, as the importer hands it over.
pub struct EdgeIn {
    pub edge: usize,
    /// The edge's exact pieces in traversal order, t0 < t1 fractions of the traversal.
    pub pieces: Vec<Bez3>,
    pub start: V3,
    pub end: V3,
    /// The vertices the use starts and ends at.
    pub v0: usize,
    pub v1: usize,
    /// An edge the face uses twice (a seam of a periodic surface, or a slit from an
    /// outer boundary into a hole): not a boundary of the face.
    pub seam: bool,
}

/// A fitted piece of a loop: a cubic in the domain, monotone in both parameters.
#[derive(Clone, Debug)]
pub struct Piece {
    pub c: [[f64; 2]; 4],
    pub edge: usize,
    pub t0: f64,
    pub t1: f64,
}

#[derive(Clone, Debug)]
enum Node {
    Leaf,
    /// Pieces (instances) entering the cell.
    Crossed(Vec<u32>),
    Split([u32; 4]),
}

#[derive(Clone, Debug)]
struct Cell {
    lo: [f64; 2],
    hi: [f64; 2],
    /// A reference point in the cell and the answer there.
    r: [f64; 2],
    inside: bool,
    node: Node,
}

#[derive(Clone, Debug)]
pub struct Trim2 {
    pub pieces: Vec<Piece>,
    /// Shifted copies of pieces: (piece, shift), for periodic directions.
    instances: Vec<(u32, [f64; 2])>,
    cells: Vec<Cell>,
    lo: [f64; 2],
    hi: [f64; 2],
    period: [Option<f64>; 2],
    /// The face lies to the left of its loops' direction in the parameter plane when
    /// its outward normal is Su x Sv (sense +1), to the right otherwise: the loops run
    /// counterclockwise seen from outside the solid.
    inside_left: bool,
    /// The largest certified deviation of a fitted piece from its edge (mm).
    pub bound: f64,
    debug: bool,
}

const FIT_SAMPLES: usize = 12;
const MAX_DEPTH: usize = 12;
const TREE_DEPTH: usize = 10;
/// Where a cell's reference sits: off its center and off every dyadic fraction, so
/// it does not land on an edge that runs along a knot line.
const REF: [f64; 2] = [0.5137, 0.4871];

// ---------------------------------------------------------------- Bernstein helpers

fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

fn b_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (m, n) = (a.len() - 1, b.len() - 1);
    // The binomials once each (the same values as computed in the loop, so the product
    // is the same to the last bit).
    let (bm, bn, bmn): (Vec<f64>, Vec<f64>, Vec<f64>) = ((0..=m).map(|i| binom(m, i)).collect(), (0..=n).map(|j| binom(n, j)).collect(), (0..=m + n).map(|k| binom(m + n, k)).collect());
    let mut c = vec![0.0; m + n + 1];
    for i in 0..=m {
        for j in 0..=n {
            c[i + j] += a[i] * b[j] * bm[i] * bn[j] / bmn[i + j];
        }
    }
    c
}

fn b_pow(a: &[f64], k: usize) -> Vec<f64> {
    let mut r = vec![1.0];
    for _ in 0..k {
        r = b_mul(&r, a);
    }
    r
}

fn b_eval(c: &[f64], t: f64) -> f64 {
    // A constant is itself (the weighted means below need not round back to it).
    if c.iter().all(|&x| x == c[0]) {
        return c[0];
    }
    let mut b = c.to_vec();
    let n = b.len() - 1;
    for level in 1..=n {
        for i in 0..=(n - level) {
            b[i] = (1.0 - t) * b[i] + t * b[i + 1];
        }
    }
    b[0]
}

/// Point of a planar cubic.
fn p_at(c: &[[f64; 2]; 4], t: f64) -> [f64; 2] {
    let s = 1.0 - t;
    let (b0, b1, b2, b3) = (s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t);
    [b0 * c[0][0] + b1 * c[1][0] + b2 * c[2][0] + b3 * c[3][0], b0 * c[0][1] + b1 * c[1][1] + b2 * c[2][1] + b3 * c[3][1]]
}

fn p_split(c: &[[f64; 2]; 4], t: f64) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let mut b = *c;
    let (mut left, mut right) = ([b[0]; 4], [b[3]; 4]);
    for level in 1..=3 {
        for i in 0..=(3 - level) {
            for k in 0..2 {
                b[i][k] = (1.0 - t) * b[i][k] + t * b[i + 1][k];
            }
        }
        left[level] = b[0];
        right[3 - level] = b[3 - level];
    }
    (left, right)
}

/// Roots in (0, 1) of the derivative of one coordinate of a cubic: where the
/// coordinate stops being monotone.
fn turning_points(a: [f64; 4]) -> Vec<f64> {
    // Derivative in Bernstein form (degree 2), then power form.
    let (d0, d1, d2) = (3.0 * (a[1] - a[0]), 3.0 * (a[2] - a[1]), 3.0 * (a[3] - a[2]));
    let (qa, qb, qc) = (d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0);
    // Every turning point counts, however small the wobble: the crossing count
    // takes each piece's ends as its range, and a straight edge fitted with a
    // wobble of 1e-12 about its line, with a query exactly on that line, is
    // crossed where the ends say it is not.
    let mut out = Vec::new();
    if qa == 0.0 {
        if qb != 0.0 {
            out.push(-qc / qb);
        }
    } else {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let q = -0.5 * (qb + qb.signum() * sq);
            if q != 0.0 {
                out.push(q / qa);
                out.push(qc / q);
            } else {
                out.push(0.0);
            }
        }
    }
    out.retain(|t| *t > 0.0 && *t < 1.0);
    out
}

// ---------------------------------------------------------------- fitting and certification

/// A cubic through fixed ends, least squares through the samples (t, point).
fn fit_cubic(samples: &[(f64, [f64; 2])], p0: [f64; 2], p3: [f64; 2]) -> [[f64; 2]; 4] {
    let (mut a11, mut a12, mut a22) = (0.0, 0.0, 0.0);
    let (mut r1, mut r2) = ([0.0; 2], [0.0; 2]);
    for &(t, q) in samples {
        let s = 1.0 - t;
        let (b0, b1, b2, b3) = (s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t);
        a11 += b1 * b1;
        a12 += b1 * b2;
        a22 += b2 * b2;
        for k in 0..2 {
            let rest = q[k] - b0 * p0[k] - b3 * p3[k];
            r1[k] += b1 * rest;
            r2[k] += b2 * rest;
        }
    }
    let det = a11 * a22 - a12 * a12;
    let (mut p1, mut p2) = ([0.0; 2], [0.0; 2]);
    for k in 0..2 {
        if det.abs() > 1e-300 {
            p1[k] = (a22 * r1[k] - a12 * r2[k]) / det;
            p2[k] = (a11 * r2[k] - a12 * r1[k]) / det;
        } else {
            p1[k] = p0[k] + (p3[k] - p0[k]) / 3.0;
            p2[k] = p0[k] + 2.0 * (p3[k] - p0[k]) / 3.0;
        }
    }
    [p0, p1, p2, p3]
}

/// A bound on |S(c(t)) - e(t)| over t in [0, 1], c in the patch's span: the
/// composition's homogeneous coordinates against the edge piece's, as Bernstein
/// polynomials in t, with the weights bounded below by their least coefficients.
/// Where the coefficient bound is loose, both curves are split exactly in halves
/// and the halves bounded, to six levels: the bound tightens with the square of
/// the interval.
fn certify_to(pa: &Patch, c: &[[f64; 2]; 4], e: &Bez3, depth: usize, want: f64) -> f64 {
    let b = certify_once(pa, c, e);
    if b <= want || depth >= 6 {
        return b;
    }
    let (cl, cr) = p_split(c, 0.5);
    let (el, er) = e.split(0.5);
    let bl = certify_to(pa, &cl, &el, depth + 1, want);
    if bl > want {
        return bl.max(b.min(f64::INFINITY)).min(b);
    }
    bl.max(certify_to(pa, &cr, &er, depth + 1, want)).min(b)
}

fn certify_once(pa: &Patch, c: &[[f64; 2]; 4], e: &Bez3) -> f64 {
    let (du, dv) = (pa.pw.len() - 1, pa.pw[0].len() - 1);
    let s: Vec<f64> = c.iter().map(|p| (p[0] - pa.u0) / (pa.u1 - pa.u0)).collect();
    let t: Vec<f64> = c.iter().map(|p| (p[1] - pa.v0) / (pa.v1 - pa.v0)).collect();
    let one_minus = |x: &[f64]| x.iter().map(|v| 1.0 - v).collect::<Vec<_>>();
    let (s1, t1) = (one_minus(&s), one_minus(&t));
    let pow_s: Vec<Vec<f64>> = (0..=du).map(|i| b_pow(&s, i)).collect();
    let pow_s1: Vec<Vec<f64>> = (0..=du).map(|i| b_pow(&s1, i)).collect();
    let pow_t: Vec<Vec<f64>> = (0..=dv).map(|j| b_pow(&t, j)).collect();
    let pow_t1: Vec<Vec<f64>> = (0..=dv).map(|j| b_pow(&t1, j)).collect();
    let d = c.len() - 1;
    let deg = (du + dv) * d;
    let mut h = vec![[0.0; 4]; deg + 1];
    let bvs: Vec<Vec<f64>> = (0..=dv).map(|j| b_mul(&pow_t[j], &pow_t1[dv - j])).collect();
    for i in 0..=du {
        let bu = b_mul(&pow_s[i], &pow_s1[du - i]);
        for (j, bv) in bvs.iter().enumerate() {
            let term = b_mul(&bu, bv);
            let w = binom(du, i) * binom(dv, j);
            for (k, tk) in term.iter().enumerate() {
                for m in 0..4 {
                    h[k][m] += pa.pw[i][j][m] * w * tk;
                }
            }
        }
    }
    let hw: Vec<f64> = h.iter().map(|x| x[3]).collect();
    let ew: Vec<f64> = e.pw.iter().map(|x| x[3]).collect();
    let min_hw = hw.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_ew = ew.iter().cloned().fold(f64::INFINITY, f64::min);
    if !(min_hw > 0.0 && min_ew > 0.0) {
        return f64::INFINITY;
    }
    let mut n = vec![[0.0; 3]; deg + e.pw.len()];
    for m in 0..3 {
        let hm: Vec<f64> = h.iter().map(|x| x[m]).collect();
        let em: Vec<f64> = e.pw.iter().map(|x| x[m]).collect();
        let a = b_mul(&hm, &ew);
        let b = b_mul(&em, &hw);
        for k in 0..n.len() {
            n[k][m] = a[k] - b[k];
        }
    }
    let max_n = n.iter().map(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()).fold(0.0, f64::max);
    max_n / (min_hw * min_ew)
}

/// The first root of a Bernstein polynomial in (lo, hi), by sign changes over a
/// fine sampling and halving.
fn first_root(c: &[f64], lo: f64, hi: f64) -> Option<f64> {
    let n = 64;
    let mut prev = (lo, b_eval(c, lo));
    for k in 1..=n {
        let x = lo + (hi - lo) * k as f64 / n as f64;
        let y = b_eval(c, x);
        if prev.1 == 0.0 {
            return Some(prev.0);
        }
        if prev.1 * y < 0.0 {
            let (mut a, mut b) = (prev.0, x);
            let mut fa = prev.1;
            for _ in 0..60 {
                let m = 0.5 * (a + b);
                let fm = b_eval(c, m);
                if fm == 0.0 {
                    return Some(m);
                }
                if fa * fm < 0.0 {
                    b = m;
                } else {
                    a = m;
                    fa = fm;
                }
            }
            return Some(0.5 * (a + b));
        }
        prev = (x, y);
    }
    None
}

/// A fitted, certified piece with the 3D point its end stands for and the edge's
/// own offset from the surface there.
struct Fitted {
    c: [[f64; 2]; 4],
    t0: f64,
    t1: f64,
    bound: f64,
    end3: V3,
    /// The edge's largest offset from the surface over the piece (what its
    /// certificate allowed beyond the tolerance).
    off: f64,
}

/// A side of the parameter domain the surface collapses on to one point (a cone's
/// apex, a sphere's pole, a lemon or apple torus's tip): the coordinate held fixed
/// along it, its value, and the point.
#[derive(Clone, Copy)]
struct Collapsed {
    k: usize,
    value: f64,
    point: V3,
}

/// Time spent building loops, over all faces, in nanoseconds: preimages, certifying,
/// placing the cell trees, and the whole builds (for an import trace).
pub static CLOCK: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Preimages by how they were found: closed form, Newton from the neighbor, the whole
/// surface's search; and the time in the last.
pub static PREIMAGES: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

thread_local! {
    /// On this thread, since the last `take_beyond`: preimages farther than the file's
    /// tolerance from the surface (edge points off it), and the farthest.
    static BEYOND: std::cell::Cell<(u64, f64)> = const { std::cell::Cell::new((0, 0.0)) };
}

fn beyond(d: f64) {
    BEYOND.with(|b| {
        let (n, m) = b.get();
        b.set((n + 1, m.max(d)));
    });
}

/// This thread's count of preimages beyond the tolerance and the farthest, reset.
pub fn take_beyond() -> (u64, f64) {
    BEYOND.with(|b| b.replace((0, 0.0)))
}

fn clocked(k: usize, t: std::time::Instant) {
    CLOCK[k].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

struct Builder<'a> {
    ns: &'a NurbsSurf,
    /// For an analytic face, its closed-form foot and the foot's parameters in its exact
    /// rational form, with the distance (the searches' answer, without searching).
    exact: Option<&'a (dyn Fn(V3) -> ([f64; 2], f64) + 'a)>,
    tol: f64,
    period: [Option<f64>; 2],
    scale: f64,
    debug: bool,
    collapsed_sides: Vec<Collapsed>,
    /// For a debugged face: time in and calls of the preimage search, time certifying.
    clock: std::cell::Cell<(f64, u64, f64)>,
}

impl<'a> Builder<'a> {
    /// The surface parameters of x, unwrapped next to `near` on periodic directions,
    /// and x's distance from the surface. An analytic face's from its closed form. A
    /// B-spline face's by Newton from `near` (the neighboring point's parameters, along
    /// the edge) when it settles within the file's tolerance of x (x is then on the
    /// surface by the file's own measure, reached continuously along the edge), else by
    /// the whole surface's search. Every piece fitted through these is certified
    /// against the edge itself, whichever way its points were found.
    fn preimage(&self, x: V3, near: Option<[f64; 2]>) -> Result<([f64; 2], f64), String> {
        let t0 = std::time::Instant::now();
        let t = self.debug.then_some(t0);
        let (mut p, dist) = match self.exact {
            Some(foot) => {
                PREIMAGES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (mut q, d) = foot(x);
                if d > self.tol {
                    beyond(d);
                }
                for k in 0..2 {
                    if self.period[k].is_none() {
                        let r = if k == 0 { self.ns.u_range } else { self.ns.v_range };
                        q[k] = q[k].clamp(r.0, r.1);
                    }
                }
                (q, d)
            }
            None => {
                let local = near.and_then(|n| {
                    let r = [self.ns.u_range, self.ns.v_range];
                    let w: [f64; 2] = std::array::from_fn(|k| match self.period[k] {
                        Some(per) => r[k].0 + (n[k] - r[k].0).rem_euclid(per),
                        None => n[k].clamp(r[k].0, r[k].1),
                    });
                    let h = self.ns.nearest_from(x, w[0], w[1]);
                    let k = match &h {
                        None => 5,
                        Some(h) if h.dist > self.tol => 6,
                        _ => 0,
                    };
                    if k > 0 {
                        PREIMAGES[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    h.filter(|h| h.dist <= self.tol)
                });
                if near.is_none() {
                    PREIMAGES[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let h = match local {
                    Some(h) => {
                        PREIMAGES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        h
                    }
                    None => {
                        let tg = std::time::Instant::now();
                        let h = self.ns.closest(x, f64::INFINITY).ok_or_else(|| "no closest point on the surface".to_string())?;
                        PREIMAGES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        PREIMAGES[3].fetch_add(tg.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        if h.dist > self.tol {
                            beyond(h.dist);
                        }
                        h
                    }
                };
                ([h.u, h.v], h.dist)
            }
        };
        clocked(0, t0);
        if let Some(t) = t {
            let c = self.clock.get();
            self.clock.set((c.0 + t.elapsed().as_secs_f64(), c.1 + 1, c.2));
        }
        if let Some(n) = near {
            for k in 0..2 {
                if let Some(per) = self.period[k] {
                    p[k] += ((n[k] - p[k]) / per).round() * per;
                }
            }
        }
        // Within the file's tolerance of a point the surface collapses to, the file
        // does not say which way round it the edge goes: the edge goes through it.
        for c in &self.collapsed_sides {
            if (x - c.point).len() <= self.tol {
                p[c.k] = c.value;
            }
        }
        Ok((p, dist))
    }

    /// The parameter direction along which the surface collapses to a point at q
    /// (a pole), if any.
    fn collapsed(&self, q: [f64; 2]) -> Option<usize> {
        for c in &self.collapsed_sides {
            let span = if c.k == 0 { self.ns.u_range.1 - self.ns.u_range.0 } else { self.ns.v_range.1 - self.ns.v_range.0 };
            if (q[c.k] - c.value).abs() <= 1e-12 * span {
                return Some(1 - c.k);
            }
        }
        self.collapsed_interior(q)
    }

    /// A row through q along which the surface is one point, away from the sides.
    fn collapsed_interior(&self, q: [f64; 2]) -> Option<usize> {
        let u = match self.period[0] {
            Some(per) => self.ns.u_range.0 + (q[0] - self.ns.u_range.0).rem_euclid(per),
            None => q[0].clamp(self.ns.u_range.0, self.ns.u_range.1),
        };
        let v = match self.period[1] {
            Some(per) => self.ns.v_range.0 + (q[1] - self.ns.v_range.0).rem_euclid(per),
            None => q[1].clamp(self.ns.v_range.0, self.ns.v_range.1),
        };
        let e = self.ns.eval(u, v);
        let (lu, lv) = (e.ps.len(), e.pt.len());
        let tiny = 1e-12 * self.scale;
        if lu <= tiny && lv > tiny {
            Some(0)
        } else if lv <= tiny && lu > tiny {
            Some(1)
        } else {
            None
        }
    }

    /// The parameter path of an edge piece from `s2`: points (t, uv) refined until
    /// consecutive points differ by under an eighth of a period along each periodic
    /// direction, so each point's period copy follows the one before it. Near a
    /// pole the parameter around it turns fast, and with a dozen samples a piece
    /// could change copy between two of them; every fit, split and closure takes its
    /// copy from this path. A point the surface collapses at is exempt (its
    /// parameter along the collapsed side is arbitrary).
    fn path(&self, p3: &Bez3, s2: [f64; 2]) -> Result<Vec<(f64, [f64; 2])>, String> {
        let mut pts = vec![(0.0, s2)];
        for i in 1..=32 {
            self.extend_path(p3, &mut pts, i as f64 / 32.0, 0)?;
        }
        Ok(pts)
    }

    fn extend_path(&self, p3: &Bez3, pts: &mut Vec<(f64, [f64; 2])>, t: f64, depth: usize) -> Result<(), String> {
        let (t0, q0) = *pts.last().unwrap();
        let q = self.preimage(p3.eval(t).p, Some(q0))?.0;
        let jump = (0..2).any(|k| self.period[k].map_or(false, |per| (q[k] - q0[k]).abs() > per / 8.0));
        if jump && depth < 40 && self.collapsed(q).is_none() && self.collapsed(q0).is_none() {
            self.extend_path(p3, pts, 0.5 * (t0 + t), depth + 1)?;
            return self.extend_path(p3, pts, t, depth + 1);
        }
        pts.push((t, q));
        Ok(())
    }

    /// The span patch holding the point m, its parameter box lifted next to m on
    /// periodic directions.
    fn patch_at(&self, m: [f64; 2]) -> Patch {
        let r = |x: f64, k: usize| match self.period[k] {
            Some(per) => {
                let lo = if k == 0 { self.ns.u_range.0 } else { self.ns.v_range.0 };
                lo + (x - lo).rem_euclid(per)
            }
            None => x,
        };
        let (a, b) = self.ns.span_at(r(m[0], 0).clamp(self.ns.u_range.0, self.ns.u_range.1), r(m[1], 1).clamp(self.ns.v_range.0, self.ns.v_range.1));
        let mut pa = self.ns.patch(a, b);
        // Lift the patch's parameter box next to the piece on periodic directions.
        for k in 0..2 {
            if let Some(per) = self.period[k] {
                let (lo, hi) = if k == 0 { (&mut pa.u0, &mut pa.u1) } else { (&mut pa.v0, &mut pa.v1) };
                let shift = ((m[k] - 0.5 * (*lo + *hi)) / per).round() * per;
                *lo += shift;
                *hi += shift;
            }
        }
        pa
    }

    /// Pieces bridging a gap between a piece's end `a` and the next piece's start
    /// `b` (both stand for the 3D point `p`, with the edge's offset `off` there): the
    /// straight segment, split where it crosses knot lines, each part certified
    /// against the point. Clamping a shared point into one piece's span and not the
    /// other's opens such gaps; parity needs the loop closed exactly.
    fn bridge(&self, a: [f64; 2], b: [f64; 2], p: V3, off: f64) -> Result<Vec<Fitted>, String> {
        let mut cuts: Vec<f64> = vec![0.0, 1.0];
        let (nu, nv) = self.ns.spans_uv();
        let knots_u: Vec<f64> = (0..nu).map(|i| self.ns.patch(i, 0).u0).chain(std::iter::once(self.ns.u_range.1)).collect();
        let knots_v: Vec<f64> = (0..nv).map(|j| self.ns.patch(0, j).v0).chain(std::iter::once(self.ns.v_range.1)).collect();
        for (k, knots) in [(0usize, &knots_u), (1, &knots_v)] {
            if a[k] == b[k] {
                continue;
            }
            for &x in knots {
                // Lifted copies on a periodic direction.
                let shifts: Vec<f64> = match self.period[k] {
                    Some(per) => (-2..=2).map(|m| m as f64 * per).collect(),
                    None => vec![0.0],
                };
                for sh in shifts {
                    let t = (x + sh - a[k]) / (b[k] - a[k]);
                    if t > 0.0 && t < 1.0 {
                        cuts.push(t);
                    }
                }
            }
        }
        cuts.sort_by(|x, y| x.total_cmp(y));
        cuts.dedup();
        let point = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let h = [p.x, p.y, p.z, 1.0];
        let e = Bez3 { pw: vec![h, h], t0: 0.0, t1: 0.0 };
        let mut out = Vec::new();
        for w in cuts.windows(2) {
            let (s0, s1) = (point(w[0]), point(w[1]));
            let c = [s0, [s0[0] + (s1[0] - s0[0]) / 3.0, s0[1] + (s1[1] - s0[1]) / 3.0], [s0[0] + 2.0 * (s1[0] - s0[0]) / 3.0, s0[1] + 2.0 * (s1[1] - s0[1]) / 3.0], s1];
            let pa = self.patch_at(p_at(&c, 0.5));
            let allowed = self.tol + off;
            let bound = certify_to(&pa, &c, &e, 0, allowed);
            if bound > allowed {
                return Err(format!("a bridge over a gap in the loop from {:?} to {:?} deviates {bound:.3e} mm from the point both stand for, over the tolerance {:.1e} plus the edge's offset {off:.1e}", a, b, self.tol));
            }
            out.push(Fitted { c, t0: 0.0, t1: 0.0, bound, end3: p, off });
        }
        Ok(out)
    }

    /// Fitted, certified pieces for one exact edge piece from the 2D point `s2`, and
    /// the 2D point it ends at. `out` gets (curve, t0, t1, bound).
    /// `s2` and `e2` come with their points' own distances from the surface (the
    /// edge's offsets there), which count toward the edge's offset like the
    /// interior samples' do.
    /// `close_to`, for a loop's last piece with no settled end: its end is the
    /// loop's start, the copy (a whole number of periods away) nearest the piece's
    /// own last sample, so a piece that winds is followed and not cut short.
    /// `path` is the whole edge piece's parameter path and [a, b] the part of it
    /// this call covers: each point takes its period copy from it.
    #[allow(clippy::too_many_arguments)]
    fn fit(&self, p3: &Bez3, s2: ([f64; 2], f64), e2: Option<([f64; 2], f64)>, close_to: Option<([f64; 2], f64)>, slack: [f64; 2], depth: usize, out: &mut Vec<Fitted>, path: &[(f64, [f64; 2])], a: f64, b: f64) -> Result<([f64; 2], f64), String> {
        let (s2, s_off) = s2;
        let at_t = |t: f64| path_at(path, a + (b - a) * t);
        let m = FIT_SAMPLES;
        // A piece with no length: a straight segment between its two shared points
        // (one point, or the same surface point under two parameters at a pole),
        // certified like any other.
        let span3 = {
            let pts: Vec<V3> = p3.ctrl().collect();
            pts.iter().map(|q| (*q - pts[0]).len()).fold(0.0, f64::max)
        };
        if span3 <= 1e-10 * self.scale {
            let (mut end, e_off) = e2.or(close_to).unwrap_or((s2, s_off));
            // Both ends stand for one point: the end's copy nearest the start.
            for k in 0..2 {
                if let Some(per) = self.period[k] {
                    end[k] = s2[k] + ((end[k] - s2[k]) - per * ((end[k] - s2[k]) / per).round());
                }
            }
            if end == s2 {
                return Ok((end, e_off));
            }
            let c = [s2, [s2[0] + (end[0] - s2[0]) / 3.0, s2[1] + (end[1] - s2[1]) / 3.0], [s2[0] + 2.0 * (end[0] - s2[0]) / 3.0, s2[1] + 2.0 * (end[1] - s2[1]) / 3.0], end];
            let pa = self.patch_at(p_at(&c, 0.5));
            let allowed = self.tol + s_off.max(e_off) + slack[0].max(slack[1]);
            let bound = certify_to(&pa, &c, p3, 0, allowed);
            if bound <= allowed {
                out.push(Fitted { c, t0: p3.t0, t1: p3.t1, bound, end3: p3.end(), off: s_off.max(e_off) });
                return Ok((end, e_off));
            }
            return Err(format!("a piece with no length whose two parameter points {:?} and {:?} are {bound:.3e} mm apart on the surface, over the tolerance {:.1e}", s2, end, self.tol));
        }
        let mut samples: Vec<(f64, [f64; 2])> = vec![(0.0, s2)];
        let mut off: f64 = s_off;
        for k in 1..m {
            let tk = k as f64 / m as f64;
            let (q, d) = self.preimage(p3.eval(tk).p, Some(at_t(tk)))?;
            off = off.max(d);
            samples.push((tk, q));
        }
        let near = at_t(1.0);
        let (mut end, e_off) = match (e2, close_to) {
            (Some(e), _) => e,
            (None, Some((start, start_off))) => {
                let (q, _) = self.preimage(p3.end(), Some(near))?;
                let mut c = start;
                for k in 0..2 {
                    if let Some(per) = self.period[k] {
                        c[k] = start[k] + ((q[k] - start[k]) / per).round() * per;
                    }
                }
                (c, start_off)
            }
            (None, None) => self.preimage(p3.end(), Some(near))?,
        };
        off = off.max(e_off);
        // At a pole (a side of the domain the surface collapses to one point) the
        // point's parameter along the collapsed side is arbitrary: the piece meets
        // the pole at its own, the one its neighboring sample has, and the loop runs
        // along the collapsed side between neighbors (a bridge, exact there).
        let mut s2 = s2;
        if let Some(k) = self.collapsed(s2) {
            s2[k] = samples[1].1[k];
            samples[0].1 = s2;
        }
        if let Some(k) = self.collapsed(end) {
            end[k] = samples[m - 1].1[k];
        }
        samples.push((1.0, end));
        let fitted = fit_cubic(&samples, s2, end);
        // The span of the middle sample; the curve is held inside its box (an edge
        // along a knot line strays by the fit's noise), and the certificate decides.
        let pa = self.patch_at(samples[m / 2].1);
        let bx = [pa.u0, pa.u1, pa.v0, pa.v1];
        let mut c = fitted;
        for p in c.iter_mut() {
            p[0] = p[0].clamp(bx[0], bx[1]);
            p[1] = p[1].clamp(bx[2], bx[3]);
        }
        // The piece's ends are the shared vertex points; where the file's consecutive
        // edges miss each other at a vertex, that gap is the file's own and is allowed.
        let allowed = self.tol + off + slack[0].max(slack[1]);
        let tc = std::time::Instant::now();
        let bound = certify_to(&pa, &c, p3, 0, allowed);
        clocked(1, tc);
        if self.debug {
            let k = self.clock.get();
            self.clock.set((k.0, k.1, k.2 + tc.elapsed().as_secs_f64()));
        }
        if self.debug {
            let pat: String = samples.iter().map(|(_, q)| if q[0] >= bx[0] && q[0] <= bx[1] && q[1] >= bx[2] && q[1] <= bx[3] { '.' } else { 'x' }).collect();
            println!("      fit depth {depth} t {:.6}..{:.6}: box u {:.6}..{:.6} v {:.6}..{:.6}, samples {pat}, bound {bound:.2e} allowed {allowed:.2e}", p3.t0, p3.t1, bx[0], bx[1], bx[2], bx[3]);
            let resid = samples.iter().map(|&(t, q)| { let f = p_at(&fitted, t); ((f[0] - q[0]).powi(2) + (f[1] - q[1]).powi(2)).sqrt() }).fold(0.0, f64::max);
            let clamp = fitted.iter().zip(c.iter()).map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()).fold(0.0, f64::max);
            let dev3 = (0..=64).map(|k| { let t = k as f64 / 64.0; let q = p_at(&c, t); (self.ns.eval(q[0], q[1]).p - p3.eval(t).p).len() }).fold(0.0, f64::max);
            let dev3f = (0..=64).map(|k| { let t = k as f64 / 64.0; let q = p_at(&fitted, t); (self.ns.eval(q[0], q[1]).p - p3.eval(t).p).len() }).fold(0.0, f64::max);
            println!("        2D fit residual {resid:.2e}, clamp moved control points by {clamp:.2e}; sampled 3D deviation: clamped curve {dev3:.2e}, unclamped {dev3f:.2e}; control u {:?}", c.iter().map(|q| format!("{:.7}", q[0])).collect::<Vec<_>>());
            // The surface's own continuity across the box's sides at the first sample.
            let q0 = samples[0].1;
            for (k, line) in [(0usize, bx[0]), (0, bx[1]), (1, bx[2]), (1, bx[3])] {
                let mut a = q0;
                let mut b = q0;
                a[k] = line - 1e-9;
                b[k] = line + 1e-9;
                let (pa_, pb_) = (self.ns.eval(a[0], a[1]).p, self.ns.eval(b[0], b[1]).p);
                if (pa_ - pb_).len() > 1e-7 {
                    println!("        the surface jumps {:.3e} mm across the line {} = {line:.6} at the first sample", (pa_ - pb_).len(), if k == 0 { "u" } else { "v" });
                }
            }
        }
        if bound <= allowed {
            out.push(Fitted { c, t0: p3.t0, t1: p3.t1, bound, end3: p3.end(), off });
            return Ok((end, e_off));
        }
        if depth >= MAX_DEPTH {
            let far = samples.iter().map(|&(t, q)| {
                let chord = [s2[0] + (end[0] - s2[0]) * t, s2[1] + (end[1] - s2[1]) * t];
                (((q[0] - chord[0]).powi(2) + (q[1] - chord[1]).powi(2)).sqrt(), t, q)
            }).fold((0.0, 0.0, [0.0; 2]), |a, b| if b.0 > a.0 { b } else { a });
            let s3 = |q: [f64; 2]| self.ns.eval(q[0], q[1]).p;
            let (xs, xe) = (p3.start(), p3.end());
            let dump: Vec<String> = samples.iter().map(|(t, q)| format!("{t:.3}:({:.6},{:.6})", q[0], q[1])).collect();
            let detail = format!("; samples {}; in 3D: piece start", dump.join(" "))
                + &format!(" {:?} end {:?}; surface at the 2D start {:?} (off the piece start {:.2e}), at the 2D end {:?} (off the piece end {:.2e}), at the last free sample {:?} (off the piece end {:.2e})", xs, xe, s3(s2), (s3(s2) - xs).len(), s3(end), (s3(end) - xe).len(), s3(samples[m - 1].1), (s3(samples[m - 1].1) - xe).len());
            return Err(detail + &format!("; piece over t {:.4}..{:.4}: certified deviation {bound:.3e} mm exceeds the tolerance {:.1e} plus the edge's own offset {off:.1e} and the vertex gap {gap:.1e} at depth {depth}; in the domain from {:?} to {:?} (end {}), the sample farthest off the chord at t {:.3} is {:?}, {:.3e} off", p3.t0, p3.t1, self.tol, s2, end, if e2.is_some() { "forced" } else { "free" }, far.1, far.2, far.0, gap = slack[0].max(slack[1])));
        }
        // A sample outside the middle sample's span: the curve crosses a knot line
        // between it and its neighbor. Split there (at the root of the fitted curve
        // against that line when the fit shows it, else between the two samples).
        // A sample on the line by rounding counts too; its split costs one level.
        let in_box = |q: [f64; 2]| q[0] >= bx[0] && q[0] <= bx[1] && q[1] >= bx[2] && q[1] <= bx[3];
        // The side a point lies farthest beyond, relative to the box, and by how much.
        let excursion = |q: [f64; 2]| {
            let sides = [(0usize, bx[0], (bx[0] - q[0]) / (bx[1] - bx[0])), (0, bx[1], (q[0] - bx[1]) / (bx[1] - bx[0])), (1, bx[2], (bx[2] - q[1]) / (bx[3] - bx[2])), (1, bx[3], (q[1] - bx[3]) / (bx[3] - bx[2]))];
            sides.into_iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap()
        };
        // Among the places the samples enter or leave the box, the one whose outside
        // sample lies farthest out: a sample a rounding error past one line, next to a
        // run of samples well past another, is not the crossing to split at.
        let crossing = (1..samples.len())
            .filter(|&k| in_box(samples[k].1) != in_box(samples[k - 1].1))
            .map(|k| {
                let outside = if in_box(samples[k].1) { samples[k - 1].1 } else { samples[k].1 };
                (k, excursion(outside))
            })
            .max_by(|a, b| a.1 .2.total_cmp(&b.1 .2));
        if let Some((k, (coord, line, _))) = crossing {
            let (ta, tb) = (samples[k - 1].0, samples[k].0);
            // Where the edge itself crosses the line: bisection along the edge, each
            // point's parameters from the certified search in the path's copy (the
            // fitted curve's own crossing can sit on the wrong side of the line).
            // At a collapsed point the crossing is meaningless: split between the
            // samples instead.
            // The fitted curve's root is only a fallback: a poor fit's root can sit well off
            // the edge's own crossing, and a split there peeled a sliver a level until a long
            // piece ran out of levels before reaching its crossing.
            let coef: Vec<f64> = fitted.iter().map(|p| p[coord] - line).collect();
            let r0 = match first_root(&coef, ta, tb) {
                Some(r) if r > 1e-6 && r < 1.0 - 1e-6 => r,
                _ => 0.5 * (ta + tb),
            };
            // The crossing between the two samples: a bracketed root of the point's parameter
            // less the line along the edge, each point's parameters from the certified search
            // in the path's copy, by regula falsi with the Illinois step (Dowell and Jarratt
            // 1971; the side that stays has its value halved, so it does not stall). The
            // probes are not the file's samples: their distances stay out of the report of
            // edges off their surfaces.
            let r = if self.collapsed(samples[k - 1].1).is_some() || self.collapsed(samples[k].1).is_some() {
                r0
            } else {
                let saved = BEYOND.with(|b| b.get());
                // (a, fa) the end kept, (b, fb) the newest; fa and fb of opposite signs.
                let (mut a, mut b) = (ta, tb);
                let (mut fa, mut fb) = (samples[k - 1].1[coord] - line, samples[k].1[coord] - line);
                let scale = (bx[2 * coord + 1] - bx[2 * coord]).abs();
                let mut found = None;
                let bracketed = fa * fb < 0.0;
                for _ in 0..80 {
                    let (lo, hi) = (a.min(b), a.max(b));
                    if !bracketed || hi - lo <= 1e-13 {
                        break;
                    }
                    let t = (a * fb - b * fa) / (fb - fa);
                    let t = if t > lo && t < hi { t } else { 0.5 * (lo + hi) };
                    let Ok((q, _)) = self.preimage(p3.eval(t).p, Some(at_t(t))) else { break };
                    let ft = q[coord] - line;
                    if ft.abs() <= 1e-12 * scale {
                        found = Some(t);
                        break;
                    }
                    if ft * fb < 0.0 {
                        (a, fa) = (b, fb);
                    } else {
                        fa *= 0.5;
                    }
                    (b, fb) = (t, ft);
                }
                BEYOND.with(|c| c.set(saved));
                let m = found.unwrap_or(0.5 * (a + b));
                if bracketed && m > 1e-6 && m < 1.0 - 1e-6 { m } else { r0 }
            };
            let (l, rr) = p3.split(r);
            let mid = self.preimage(l.end(), Some(at_t(r)))?;
            let am = a + (b - a) * r;
            self.fit(&l, (s2, s_off), Some(mid), None, [slack[0], 0.0], depth + 1, out, path, a, am)?;
            return self.fit(&rr, mid, Some((end, e_off)), None, [0.0, slack[1]], depth + 1, out, path, am, b);
        }
        let (l, rr) = p3.split(0.5);
        let mid = self.preimage(l.end(), Some(at_t(0.5)))?;
        let am = 0.5 * (a + b);
        self.fit(&l, (s2, s_off), Some(mid), None, [slack[0], 0.0], depth + 1, out, path, a, am)?;
        self.fit(&rr, mid, Some((end, e_off)), None, [0.0, slack[1]], depth + 1, out, path, am, b)
    }
}

/// Builds the trim loops of a face on `ns`, with `sense` +1 when the face's outward
/// normal is Su x Sv, from its loops of edge uses, within the file's tolerance. `exact`:
/// for an analytic face, a point's foot as parameters of `ns` and its distance.
pub fn build(ns: &NurbsSurf, exact: Option<&(dyn Fn(V3) -> ([f64; 2], f64) + Sync)>, sense: f64, loops: &[Vec<EdgeIn>], tol: f64, closed_u: bool, closed_v: bool, debug: bool) -> Result<(Trim2, f64), String> {
    let (lo, hi) = ([ns.u_range.0, ns.v_range.0], [ns.u_range.1, ns.v_range.1]);
    let period = [closed_u.then(|| hi[0] - lo[0]), closed_v.then(|| hi[1] - lo[1])];
    if debug {
        println!("trim2: domain u {:?} v {:?}, periods {:?}, sense {sense}, {} loops of {:?} uses", ns.u_range, ns.v_range, period, loops.len(), loops.iter().map(|l| l.len()).collect::<Vec<_>>());
    }
    let (blo, bhi) = ns.bbox();
    let scale = (bhi - blo).len().max(1e-9);
    let mut collapsed_sides = Vec::new();
    for k in 0..2 {
        if period[k].is_some() {
            continue;
        }
        for value in [lo[k], hi[k]] {
            let at = |t: f64| {
                let free = 1 - k;
                let mut q = [0.0; 2];
                q[k] = value;
                q[free] = lo[free] + (hi[free] - lo[free]) * t;
                ns.eval(q[0], q[1]).p
            };
            let p0 = at(0.0);
            if (1..=16).all(|i| (at(i as f64 / 16.0) - p0).len() <= 1e-12 * scale) {
                collapsed_sides.push(Collapsed { k, value, point: p0 });
            }
        }
    }
    if debug && !collapsed_sides.is_empty() {
        println!("  collapsed sides: {:?}", collapsed_sides.iter().map(|c| (c.k, c.value)).collect::<Vec<_>>());
    }
    let inside_left = sense > 0.0;
    let exact = exact.map(|f| f as &dyn Fn(V3) -> ([f64; 2], f64));
    let b = Builder { ns, exact, tol, period, scale, debug, collapsed_sides: collapsed_sides.clone(), clock: std::cell::Cell::new((0.0, 0, 0.0)) };
    let t_build = std::time::Instant::now();
    let mut pieces: Vec<Piece> = Vec::new();
    let mut bound: f64 = 0.0;
    // The largest gap between consecutive edges' ends at a vertex: the file's own.
    let mut gap_max: f64 = 0.0;
    for lp in loops {
        // Cycles of boundary uses: an edge the face uses twice is dropped, and each
        // use is followed by the next unvisited use, in loop order, that starts at
        // the vertex it ends at. A seam's or a slit's two uses enclose a run that
        // starts and ends at one vertex, so dropping them leaves closed cycles; the
        // pairing is by vertex, never by distance.
        let n = lp.len();
        if n == 0 {
            continue;
        }
        let mut visited = vec![false; n];
        let mut chains: Vec<Vec<usize>> = Vec::new();
        for i0 in 0..n {
            if lp[i0].seam || visited[i0] {
                continue;
            }
            let mut cyc = vec![i0];
            visited[i0] = true;
            let mut cur = i0;
            loop {
                if lp[cur].v1 == lp[i0].v0 {
                    break;
                }
                let next = (1..n).map(|d| (cur + d) % n).find(|&j| !lp[j].seam && !visited[j] && lp[j].v0 == lp[cur].v1);
                match next {
                    Some(j) => {
                        visited[j] = true;
                        cyc.push(j);
                        cur = j;
                    }
                    None => return Err(format!("edge #{}: the boundary does not close (no use starts where it ends)", lp[cur].edge)),
                }
            }
            chains.push(cyc);
        }
        for chain in chains {
            let (start, start_off) = b.preimage(lp[chain[0]].start, None)?;
            let mut at = (start, start_off);
            let mut chain_pieces: Vec<(Fitted, usize)> = Vec::new();
            if debug {
                println!("  chain of {} edges {:?}, start {:?}", chain.len(), chain.iter().map(|&i| lp[i].edge).collect::<Vec<_>>(), start);
            }
            for (ci, &i) in chain.iter().enumerate() {
                let e = &lp[i];
                let last_edge = ci + 1 == chain.len();
                // The file's gaps at this edge's two vertices (its start is snapped to
                // the previous edge's end; the chain's end is snapped to its start).
                let gap_start = if ci == 0 { 0.0 } else { (e.start - lp[chain[ci - 1]].end).len() };
                let gap_end = if last_edge { (e.end - lp[chain[0]].start).len() } else { 0.0 };
                gap_max = gap_max.max(gap_start).max(gap_end);
                for (pi, p3) in e.pieces.iter().enumerate() {
                    let last = last_edge && pi + 1 == e.pieces.len();
                    let slack = [if pi == 0 { gap_start } else { 0.0 }, if last { gap_end } else { 0.0 }];
                    // The chain closes on its start, shifted by whole periods.
                    let close_to = if last { Some((start, start_off)) } else { None };
                    let mut out = Vec::new();
                    let from = at;
                    let path = b.path(p3, at.0).map_err(|m| format!("edge #{}: {m}", e.edge))?;
                    at = b.fit(p3, at, None, close_to, slack, 0, &mut out, &path, 0.0, 1.0).map_err(|m| format!("edge #{}: {m}", e.edge))?;
                    if debug {
                        println!("    edge {} piece t {:.3}..{:.3}: {:?} -> {:?}, {} fitted, bounds {:?}", e.edge, p3.t0, p3.t1, from.0, at.0, out.len(), out.iter().map(|o| format!("{:.1e}", o.bound)).collect::<Vec<_>>());
                    }
                    for f in out {
                        chain_pieces.push((f, e.edge));
                    }
                }
            }
            // Close every gap between consecutive pieces (cyclically) with a bridge.
            let n = chain_pieces.len();
            let mut bridged: Vec<(Fitted, usize)> = Vec::new();
            for i in 0..n {
                let (ref f, edge) = chain_pieces[i];
                // The next piece's start, lifted next to this piece's end on periodic
                // directions: a chain going once around closes on its start a period away.
                let mut next_start = chain_pieces[(i + 1) % n].0.c[0];
                for k in 0..2 {
                    if let Some(per) = period[k] {
                        next_start[k] += ((f.c[3][k] - next_start[k]) / per).round() * per;
                    }
                }
                // Along a collapsed side the loop goes the way that keeps the face on
                // its side of the loop: the side is one point, so either way is exact
                // in space, but only one is right for which side the face is on.
                if let (Some(fa), Some(fb)) = (b.collapsed(f.c[3]), b.collapsed(next_start)) {
                    if fa == fb {
                        let k = 1 - fa;
                        if let Some(c) = collapsed_sides.iter().find(|c| c.k == k && (f.c[3][k] - c.value).abs() <= 1e-9 * (hi[k] - lo[k]).abs().max(1e-300)) {
                            let s_in = if (c.value - lo[k]).abs() <= (c.value - hi[k]).abs() { 1.0 } else { -1.0 };
                            let sigma = if fa == 0 { if inside_left { s_in } else { -s_in } } else if inside_left { -s_in } else { s_in };
                            if let Some(per) = period[fa] {
                                let d = next_start[fa] - f.c[3][fa];
                                if sigma > 0.0 && d < 0.0 {
                                    next_start[fa] += per;
                                } else if sigma < 0.0 && d > 0.0 {
                                    next_start[fa] -= per;
                                }
                            }
                        }
                    }
                }
                let (c3, end3, off, bd) = (f.c[3], f.end3, f.off.max(chain_pieces[(i + 1) % n].0.off), f.bound);
                bridged.push((Fitted { c: f.c, t0: f.t0, t1: f.t1, bound: bd, end3, off: f.off }, edge));
                if c3 != next_start {
                    if debug {
                        println!("    bridge from {:?} to {:?} ({:.2e} apart)", c3, next_start, ((c3[0] - next_start[0]).powi(2) + (c3[1] - next_start[1]).powi(2)).sqrt());
                    }
                    for g in b.bridge(c3, next_start, end3, off)? {
                        bridged.push((g, edge));
                    }
                }
            }
            for (f, edge) in bridged {
                let (c, t0, t1) = (f.c, f.t0, f.t1);
                bound = bound.max(f.bound);
                // Monotone in both parameters: split at the turning points.
                let mut ts: Vec<f64> = turning_points([c[0][0], c[1][0], c[2][0], c[3][0]]);
                ts.extend(turning_points([c[0][1], c[1][1], c[2][1], c[3][1]]));
                ts.sort_by(|a, b| a.total_cmp(b));
                ts.dedup();
                let (mut rest, mut ta) = (c, 0.0);
                for &t in &ts {
                    let local = (t - ta) / (1.0 - ta);
                    let (l, r) = p_split(&rest, local);
                    pieces.push(Piece { c: l, edge, t0: t0 + (t1 - t0) * ta, t1: t0 + (t1 - t0) * t });
                    rest = r;
                    ta = t;
                }
                pieces.push(Piece { c: rest, edge, t0: t0 + (t1 - t0) * ta, t1 });
            }
        }
    }
    // Consecutive pieces of a chain share their point exactly; anything else is a
    // gap in the loop, which parity cannot tolerate.
    let mut gap2d: f64 = 0.0;
    for w in pieces.windows(2) {
        let (a, b) = (w[0].c[3], w[1].c[0]);
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        if d > 0.0 && d < 1e-3 * ((hi[0] - lo[0]).max(hi[1] - lo[1])) {
            gap2d = gap2d.max(d);
        }
    }
    if debug {
        println!("  largest gap between consecutive pieces (below a thousandth of the domain, so not a chain break): {gap2d:.3e}");
    }
    // Control values snapped to a grid of 2^-40 of the domain's extent along each
    // direction. A piece fitted to a straight edge wobbles about it in its last
    // bits, and the crossing count reads a piece's range from its ends but its
    // value in between by evaluation, which then disagree for a query on that line
    // (rear: a torus face read as its own complement at a grid point on its edge's
    // line). Snapped, such a piece is exactly constant and evaluates to its
    // constant. Rounding to a grid keeps order, so monotone pieces stay monotone and
    // shared ends stay shared; it moves the loops by 2^-41 of the domain, far inside
    // the certificate's bound. A periodic direction is snapped only when its period
    // is a whole number of grid steps (an analytic form's 4): a loop closes on its
    // start one period over, and snapping its two ends apart would open a gap.
    for k in 0..2 {
        // The step from the domain's extent alone: a domain far from zero (knots near
        // 1,000 spanning 0.5) would otherwise get a step that moves its loops by a
        // tenth of a 1e-6 mm tolerance. Where a value is too large for the step
        // (beyond 2^52 steps), rounding leaves it as it is.
        let size = (hi[k] - lo[k]).abs().max(1e-300);
        let h = 2f64.powi(size.log2().floor() as i32 - 40);
        if period[k].is_some_and(|per| (per / h).fract() != 0.0) {
            continue;
        }
        // Steps counted from the domain's start, and the domain's ends kept: a loop
        // that runs along a side of the domain stays exactly on it, where a point
        // clamped onto that side is taken.
        let (a, b) = (lo[k], hi[k]);
        let snap = |x: f64| {
            if (x - a).abs() <= 0.5 * h {
                a
            } else if (x - b).abs() <= 0.5 * h {
                b
            } else {
                a + ((x - a) / h).round() * h
            }
        };
        for p in pieces.iter_mut() {
            for q in p.c.iter_mut() {
                q[k] = snap(q[k]);
            }
        }
    }
    let mut t = Trim2 { pieces, instances: Vec::new(), cells: Vec::new(), lo, hi, period, inside_left: sense > 0.0, bound, debug };
    let t_place = std::time::Instant::now();
    let fitting = t_build.elapsed().as_secs_f64();
    t.place()?;
    clocked(2, t_place);
    clocked(3, t_build);
    if debug {
        let c = b.clock.get();
        println!("  time: fitting {fitting:.3} s ({} preimage searches in {:.3} s, certifying {:.3} s), placing {:.3} s", c.1, c.0, c.2, t_place.elapsed().as_secs_f64());
    }
    if debug {
        println!("  {} pieces, {} instances, {} cells; largest vertex gap {gap_max:.2e} mm", t.pieces.len(), t.instances.len(), t.cells.len());
    }
    Ok((t, gap_max))
}

/// The path's parameters at t, linear between its points.
fn path_at(path: &[(f64, [f64; 2])], t: f64) -> [f64; 2] {
    let i = path.partition_point(|p| p.0 < t);
    if i == 0 {
        return path[0].1;
    }
    if i >= path.len() {
        return path[path.len() - 1].1;
    }
    let ((ta, a), (tb, b)) = (path[i - 1], path[i]);
    let w = if tb > ta { (t - ta) / (tb - ta) } else { 0.0 };
    [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w]
}

fn bbox(c: &[[f64; 2]; 4]) -> ([f64; 2], [f64; 2]) {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in c {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

/// Whether a piece monotone in both parameters passes through the closed box
/// [lo, hi]: over the parameters where its u lies in the box's u range, its v
/// runs between its values at the two ends of that range.
fn meets_box(c: &[[f64; 2]; 4], lo: [f64; 2], hi: [f64; 2]) -> bool {
    let (b0, b1) = bbox(c);
    if !overlaps(b0, b1, lo, hi) {
        return false;
    }
    let (cu, cv) = ([c[0][0], c[1][0], c[2][0], c[3][0]], [c[0][1], c[1][1], c[2][1], c[3][1]]);
    let (u0, u3) = (cu[0], cu[3]);
    let (ta, tb) = if u0 == u3 {
        (0.0, 1.0)
    } else {
        let (a, b) = (lo[0].max(u0.min(u3)), hi[0].min(u0.max(u3)));
        (param_at(cu, a), param_at(cu, b))
    };
    let (va, vb) = (b_eval(&cv, ta), b_eval(&cv, tb));
    va.min(vb) <= hi[1] && va.max(vb) >= lo[1]
}

/// The parameter where a monotone cubic coordinate equals x, for x between its
/// end values (inclusive).
fn param_at(a: [f64; 4], x: f64) -> f64 {
    if x == a[0] {
        return 0.0;
    }
    if x == a[3] {
        return 1.0;
    }
    crossing(a, x).unwrap_or(if (x - a[0]).abs() < (x - a[3]).abs() { 0.0 } else { 1.0 })
}

fn overlaps(alo: [f64; 2], ahi: [f64; 2], blo: [f64; 2], bhi: [f64; 2]) -> bool {
    (0..2).all(|k| alo[k] <= bhi[k] && blo[k] <= ahi[k])
}

/// The parameter where a monotone cubic coordinate equals x, when x lies in its
/// half-open range [min, max) (so a crossing through a shared end counts once).
fn crossing(a: [f64; 4], x: f64) -> Option<f64> {
    let (p0, p1) = (a[0], a[3]);
    if !(p0.min(p1) <= x && x < p0.max(p1)) {
        return None;
    }
    // Safeguarded Newton: the bracket always shrinks.
    let (mut lo, mut hi) = (0.0, 1.0);
    let f = |t: f64| b_eval(&a, t) - x;
    let mut flo = f(0.0);
    if flo == 0.0 {
        return Some(0.0);
    }
    let mut t = ((x - p0) / (p1 - p0)).clamp(0.0, 1.0);
    for _ in 0..60 {
        let ft = f(t);
        if ft == 0.0 {
            return Some(t);
        }
        if (ft < 0.0) == (flo < 0.0) {
            lo = t;
            flo = ft;
        } else {
            hi = t;
        }
        let d = 3.0 * b_eval(&[a[1] - a[0], a[2] - a[1], a[3] - a[2]], t);
        let mut nt = if d != 0.0 { t - ft / d } else { 0.5 * (lo + hi) };
        if !(nt > lo && nt < hi) {
            nt = 0.5 * (lo + hi);
        }
        if (nt - t).abs() <= 1e-16 {
            return Some(nt);
        }
        t = nt;
    }
    Some(t)
}

impl Trim2 {
    fn shifted(&self, inst: u32) -> [[f64; 2]; 4] {
        let (pi, s) = self.instances[inst as usize];
        let mut c = self.pieces[pi as usize].c;
        for p in c.iter_mut() {
            p[0] += s[0];
            p[1] += s[1];
        }
        c
    }

    /// Whether the path from a to b (vertical leg to (a.u, b.v), then horizontal)
    /// crosses the given pieces an odd number of times.
    ///
    /// The whole path is taken moved up by an infinitesimal e and right by e^2 (a
    /// fixed symbolic perturbation; a is a reference point at least an offset away
    /// from every piece, so its answer is the moved point's too). Round-number
    /// parts put queries, and references, exactly on their loops' straight pieces,
    /// and then a leg would run along a piece or end on one; moved, the vertical leg
    /// runs just right of the line u = a.u from a.v to b.v, the horizontal one just
    /// above v = b.v from a.u to b.u, and each piece is crossed or not on one rule.
    fn parity(&self, a: [f64; 2], b: [f64; 2], among: &[u32]) -> bool {
        let mut n = 0u32;
        let up = a[1] < b[1];
        let right = a[0] < b[0];
        for &i in among {
            let c = self.shifted(i);
            let (cu, cv) = ([c[0][0], c[1][0], c[2][0], c[3][0]], [c[0][1], c[1][1], c[2][1], c[3][1]]);
            if a[1] != b[1] {
                // Pieces whose u range holds a.u + e^2: [min, max), as `crossing` takes.
                if let Some(t) = crossing(cu, a[0]) {
                    let y = b_eval(&cv, t);
                    if if up { y > a[1] && y <= b[1] } else { y > b[1] && y <= a[1] } {
                        n += 1;
                    }
                }
            }
            if a[0] != b[0] {
                if let Some(t) = crossing(cv, b[1]) {
                    let x = b_eval(&cu, t);
                    // Where the piece passes through an end of the leg, it meets the
                    // raised leg at x + (du/dv) e, against the leg's ends a.u + e^2
                    // and b.u + e^2.
                    let slope_pos = || {
                        let du = b_eval(&[cu[1] - cu[0], cu[2] - cu[1], cu[3] - cu[2]], t);
                        let mut dv = b_eval(&[cv[1] - cv[0], cv[2] - cv[1], cv[3] - cv[2]], t);
                        if dv == 0.0 {
                            dv = cv[3] - cv[0];
                        }
                        du * dv > 0.0
                    };
                    let counted = if x == b[0] {
                        !slope_pos() == right
                    } else if x == a[0] {
                        slope_pos() == right
                    } else if right {
                        x > a[0] && x < b[0]
                    } else {
                        x > b[0] && x < a[0]
                    };
                    if counted {
                        n += 1;
                    }
                }
            }
        }
        n % 2 == 1
    }

    /// Shifted copies, the root's answer and the cell tree.
    fn place(&mut self) -> Result<(), String> {
        // Copies of each piece, along each periodic direction, that reach within one
        // period of the domain (the directions' units differ: an arc parameter beside
        // a height in mm, so each direction keeps its own margin).
        let (mut rlo, mut rhi) = (self.lo, self.hi);
        for k in 0..2 {
            if let Some(per) = self.period[k] {
                rlo[k] -= per;
                rhi[k] += per;
            }
        }
        let mut inst = Vec::new();
        for (i, p) in self.pieces.iter().enumerate() {
            let (plo, phi) = bbox(&p.c);
            let range = |k: usize| -> (i64, i64) {
                match self.period[k] {
                    Some(per) => (((rlo[k] - phi[k]) / per).ceil() as i64, ((rhi[k] - plo[k]) / per).floor() as i64),
                    None => (0, 0),
                }
            };
            let (ku, kv) = (range(0), range(1));
            for a in ku.0..=ku.1 {
                for c in kv.0..=kv.1 {
                    inst.push((i as u32, [a as f64 * self.period[0].unwrap_or(0.0), c as f64 * self.period[1].unwrap_or(0.0)]));
                }
            }
        }
        self.instances = inst;
        let boxes: Vec<([f64; 2], [f64; 2])> = (0..self.instances.len() as u32).map(|i| bbox(&self.shifted(i))).collect();
        let all: Vec<u32> = (0..self.instances.len() as u32).collect();
        let root_r = [self.lo[0] + (self.hi[0] - self.lo[0]) * REF[0], self.lo[1] + (self.hi[1] - self.lo[1]) * REF[1]];
        // The root's answer, from a point offset to the left of one piece whose
        // nearest loop point is proven to be that piece: inside exactly when the face
        // lies to the left of its loops. A point outside the domain is not known to be
        // outside the face: a face can reach through a side the surface collapses on
        // (a sphere's pole, a lemon torus's tip) with no edge there.
        let root_inside = if self.pieces.is_empty() {
            true
        } else {
            let (r, left) = self.offset_reference(&all)?;
            let ans = (left == self.inside_left) ^ self.parity(r, root_r, &all);
            if self.debug {
                let out = [root_r[0], self.lo[1] - (self.hi[1] - self.lo[1])];
                let signed: f64 = self.pieces.iter().map(|p| {
                    let pts: Vec<[f64; 2]> = (0..=16).map(|k| p_at(&p.c, k as f64 / 16.0)).collect();
                    pts.windows(2).map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1]).sum::<f64>()
                }).sum::<f64>() * 0.5;
                println!("  anchor: reference {:?} (left of a piece), inside_left {}, root {:?} answer {ans}; from below the domain: {}; signed area of the loops {signed:.6e}", r, self.inside_left, root_r, self.parity(out, root_r, &all));
            }
            ans
        };
        self.cells.clear();
        self.cells.push(Cell { lo: self.lo, hi: self.hi, r: root_r, inside: root_inside, node: Node::Leaf });
        let node = self.build_cell(0, &all, &boxes, 0);
        self.cells[0].node = node;
        Ok(())
    }

    /// The node of cell `id` (already placed with its reference answered): its
    /// children's references are answered through paths inside this cell.
    fn build_cell(&mut self, id: usize, cand: &[u32], boxes: &[([f64; 2], [f64; 2])], depth: usize) -> Node {
        let (lo, hi, r, inside) = (self.cells[id].lo, self.cells[id].hi, self.cells[id].r, self.cells[id].inside);
        let touching: Vec<u32> = cand.iter().copied().filter(|&i| overlaps(lo, hi, boxes[i as usize].0, boxes[i as usize].1)).collect();
        if touching.is_empty() {
            return Node::Leaf;
        }
        if depth >= TREE_DEPTH || touching.len() <= 4 {
            return Node::Crossed(touching);
        }
        let m = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
        let quads = [(lo, m), ([m[0], lo[1]], [hi[0], m[1]]), ([lo[0], m[1]], [m[0], hi[1]]), (m, hi)];
        let mut kids = [0u32; 4];
        for (k, (qlo, qhi)) in quads.into_iter().enumerate() {
            let qr = [qlo[0] + (qhi[0] - qlo[0]) * REF[0], qlo[1] + (qhi[1] - qlo[1]) * REF[1]];
            let qin = inside ^ self.parity(r, qr, &touching);
            let kid = self.cells.len();
            self.cells.push(Cell { lo: qlo, hi: qhi, r: qr, inside: qin, node: Node::Leaf });
            let node = self.build_cell(kid, &touching, boxes, depth + 1);
            self.cells[kid].node = node;
            kids[k] = kid as u32;
        }
        Node::Split(kids)
    }

    /// A point offset from the first piece's middle, on its left, whose nearest loop
    /// point is that middle: the offset halves until no other point of the loop is
    /// as close. Returns the point and `true` (it is on the left).
    fn offset_reference(&self, all: &[u32]) -> Result<([f64; 2], bool), String> {
        // Pieces longest first; the first whose left-offset point settles inside the
        // domain (a piece along one of its sides may have that side's outside on its
        // left, where the point would not be in the domain at all).
        let mut order: Vec<&Piece> = self.pieces.iter().collect();
        let chord = |p: &Piece| ((p.c[3][0] - p.c[0][0]).powi(2) + (p.c[3][1] - p.c[0][1]).powi(2)).sqrt();
        order.sort_by(|a, b| chord(b).total_cmp(&chord(a)));
        for p in order.into_iter().take(64) {
            for left in [true, false] {
                if let Some(r) = self.offset_from(p, all, left) {
                    return Ok((r, left));
                }
            }
        }
        Err("no loop piece gives a settled reference point inside the domain".into())
    }

    /// The point offset to one side of piece p's middle (its left, or its right)
    /// whose nearest loop point is that middle, when one lies inside the domain by
    /// at least half the offset. The offset never shrinks below a millionth of the
    /// domain's smaller extent, so the side is never decided at rounding scale.
    fn offset_from(&self, p: &Piece, all: &[u32], left: bool) -> Option<[f64; 2]> {
        let small = (self.hi[0] - self.lo[0]).min(self.hi[1] - self.lo[1]);
        let in_domain = |r: [f64; 2], margin: f64| (0..2).all(|k| self.period[k].is_some() || (r[k] > self.lo[k] + margin && r[k] < self.hi[k] - margin));
        // The piece's copy whose middle lies in the domain: the periodic copies are
        // only guaranteed within a period of it, and queries are reduced into it.
        let mut m = p_at(&p.c, 0.5);
        for k in 0..2 {
            if let Some(per) = self.period[k] {
                m[k] = self.lo[k] + (m[k] - self.lo[k]).rem_euclid(per);
            }
        }
        let d = {
            let (a, b) = (p_at(&p.c, 0.49), p_at(&p.c, 0.51));
            [b[0] - a[0], b[1] - a[1]]
        };
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if !(l > 0.0) {
            return None;
        }
        let side = if left { 1.0 } else { -1.0 };
        let n = [-d[1] / l * side, d[0] / l * side];
        let mut delta = 1e-3 * small;
        while delta >= 1e-6 * small {
            let r = [m[0] + n[0] * delta, m[1] + n[1] * delta];
            if in_domain(r, 0.5 * delta) {
                let nearest = all.iter().map(|&i| self.dist2_to(i, r)).fold(f64::INFINITY, f64::min);
                if nearest >= delta * delta * (1.0 - 1e-9) {
                    return Some(r);
                }
            }
            delta *= 0.5;
        }
        None
    }

    /// Squared distance from q to a shifted piece, by sampling and refinement.
    fn dist2_to(&self, inst: u32, q: [f64; 2]) -> f64 {
        let c = self.shifted(inst);
        let d2 = |t: f64| {
            let p = p_at(&c, t);
            (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)
        };
        let mut best = (f64::INFINITY, 0.0);
        for k in 0..=16 {
            let t = k as f64 / 16.0;
            let d = d2(t);
            if d < best.0 {
                best = (d, t);
            }
        }
        let (mut a, mut b) = ((best.1 - 1.0 / 16.0).max(0.0), (best.1 + 1.0 / 16.0).min(1.0));
        for _ in 0..40 {
            let (x, y) = (a + 0.381966 * (b - a), b - 0.381966 * (b - a));
            if d2(x) < d2(y) {
                b = y;
            } else {
                a = x;
            }
        }
        best.0.min(d2(0.5 * (a + b)))
    }

    /// Whether the surface point at (u, v) lies in the face.
    pub fn inside(&self, u: f64, v: f64) -> bool {
        let mut q = [u, v];
        for k in 0..2 {
            q[k] = match self.period[k] {
                Some(per) => self.lo[k] + (q[k] - self.lo[k]).rem_euclid(per),
                None => q[k].clamp(self.lo[k], self.hi[k]),
            };
        }
        if self.pieces.is_empty() {
            return true;
        }
        let mut id = 0usize;
        loop {
            let cell = &self.cells[id];
            match &cell.node {
                Node::Leaf => return cell.inside,
                Node::Crossed(among) => return cell.inside ^ self.parity(cell.r, q, among),
                Node::Split(kids) => {
                    let m = [0.5 * (cell.lo[0] + cell.hi[0]), 0.5 * (cell.lo[1] + cell.hi[1])];
                    let k = (q[0] >= m[0]) as usize + 2 * (q[1] >= m[1]) as usize;
                    id = kids[k] as usize;
                }
            }
        }
    }

    /// What decides the answer at (u, v), for a report.
    pub fn explain(&self, u: f64, v: f64) -> String {
        let mut q = [u, v];
        for k in 0..2 {
            q[k] = match self.period[k] {
                Some(per) => self.lo[k] + (q[k] - self.lo[k]).rem_euclid(per),
                None => q[k].clamp(self.lo[k], self.hi[k]),
            };
        }
        if self.pieces.is_empty() {
            return "no loop pieces: the whole surface".into();
        }
        let all: Vec<u32> = (0..self.instances.len() as u32).collect();
        let root = &self.cells[0];
        let direct = root.inside ^ self.parity(root.r, q, &all);
        let anchored = if self.period[1].is_none() {
            Some(self.parity([q[0], self.lo[1] - (self.hi[1] - self.lo[1])], q, &all))
        } else if self.period[0].is_none() {
            Some(self.parity([self.lo[0] - (self.hi[0] - self.lo[0]), q[1]], q, &all))
        } else {
            None
        };
        let mut id = 0usize;
        let mut path = String::new();
        loop {
            let cell = &self.cells[id];
            match &cell.node {
                Node::Leaf => {
                    path += &format!(" leaf {:?}..{:?} reference {:?} inside {}", cell.lo, cell.hi, cell.r, cell.inside);
                    break;
                }
                Node::Crossed(among) => {
                    let each: Vec<String> = among.iter().map(|&i| format!("{}:{}", self.instances[i as usize].0, self.parity(cell.r, q, &[i]) as u8)).collect();
                    path += &format!(" crossed {:?}..{:?} reference {:?} inside {} with {} pieces, parity {} (piece:count {})", cell.lo, cell.hi, cell.r, cell.inside, among.len(), self.parity(cell.r, q, among), each.join(" "));
                    break;
                }
                Node::Split(kids) => {
                    let m = [0.5 * (cell.lo[0] + cell.hi[0]), 0.5 * (cell.lo[1] + cell.hi[1])];
                    let k = (q[0] >= m[0]) as usize + 2 * (q[1] >= m[1]) as usize;
                    id = kids[k] as usize;
                }
            }
        }
        // Which pieces the anchored vertical ray counts.
        let mut hits = String::new();
        if self.period[1].is_none() {
            let a = [q[0], self.lo[1] - (self.hi[1] - self.lo[1])];
            for i in 0..self.instances.len() as u32 {
                let c = self.shifted(i);
                let cu = [c[0][0], c[1][0], c[2][0], c[3][0]];
                let cv = [c[0][1], c[1][1], c[2][1], c[3][1]];
                if let Some(t) = crossing(cu, a[0]) {
                    let y = b_eval(&cv, t);
                    hits += &format!(" [piece {} edge {} u {:.6}..{:.6} v {:.6}..{:.6}: t {t:.4} y {y:.6} counted {}]", self.instances[i as usize].0, self.pieces[self.instances[i as usize].0 as usize].edge, cu[0], cu[3], cv[0], cv[3], y > a[1].min(q[1]) && y < a[1].max(q[1]));
                }
            }
        }
        let listing: Vec<String> = self.pieces.iter().enumerate().map(|(i, p)| format!("{i}:e{} u{:?} v{:?}", p.edge, [p.c[0][0], p.c[1][0], p.c[2][0], p.c[3][0]], [p.c[0][1], p.c[1][1], p.c[2][1], p.c[3][1]])).collect();
        format!("q {:?}: from the root reference {:?} (inside {}) over all pieces: {direct}; straight from outside the domain: {anchored:?}; polygon count {}; anchored ray hits:{hits}; tree:{path}; pieces: {}", q, root.r, root.inside, self.inside_slow(u, v), listing.join(" "))
    }

    /// The answer by a plain crossing count against every piece sampled as a
    /// polygon (256 segments each), independent of the tree, the monotone pieces and
    /// the root solves: a check on the machinery, sharing only the fitted loop.
    pub fn inside_slow(&self, u: f64, v: f64) -> bool {
        let mut q = [u, v];
        for k in 0..2 {
            q[k] = match self.period[k] {
                Some(per) => self.lo[k] + (q[k] - self.lo[k]).rem_euclid(per),
                None => q[k].clamp(self.lo[k], self.hi[k]),
            };
        }
        if self.pieces.is_empty() {
            return true;
        }
        // From the root's reference, count crossings of the vertical then the
        // horizontal leg.
        let root = &self.cells[0];
        let (a, start_inside) = (root.r, root.inside);
        let mut n = 0u32;
        let between = |x: f64, p: f64, r: f64| (x > p.min(r)) && (x < p.max(r));
        for i in 0..self.instances.len() as u32 {
            let c = self.shifted(i);
            let pts: Vec<[f64; 2]> = (0..=256).map(|k| p_at(&c, k as f64 / 256.0)).collect();
            for w in pts.windows(2) {
                let (p0, p1) = (w[0], w[1]);
                // Vertical leg at u = a[0] from a[1] to q[1].
                if a[1] != q[1] && p0[0].min(p1[0]) <= a[0] && a[0] < p0[0].max(p1[0]) {
                    let t = (a[0] - p0[0]) / (p1[0] - p0[0]);
                    let y = p0[1] + t * (p1[1] - p0[1]);
                    if between(y, a[1], q[1]) {
                        n += 1;
                    }
                }
                // Horizontal leg at v = q[1] from a[0] to q[0].
                if a[0] != q[0] && p0[1].min(p1[1]) <= q[1] && q[1] < p0[1].max(p1[1]) {
                    let t = (q[1] - p0[1]) / (p1[1] - p0[1]);
                    let x = p0[0] + t * (p1[0] - p0[0]);
                    if between(x, a[0], q[0]) {
                        n += 1;
                    }
                }
            }
        }
        start_inside ^ (n % 2 == 1)
    }

    /// Whether the box [lo, hi] of the domain lies wholly in the face (Some(true)),
    /// wholly out of it (Some(false)), or straddles its loops (None): every cell the
    /// box meets is a leaf with that answer, or a crossed cell none of whose pieces
    /// comes into the box.
    pub fn box_class(&self, lo: [f64; 2], hi: [f64; 2]) -> Option<bool> {
        if self.pieces.is_empty() {
            return Some(true);
        }
        self.class_in(0, lo, hi)
    }

    fn class_in(&self, id: usize, lo: [f64; 2], hi: [f64; 2]) -> Option<bool> {
        let cell = &self.cells[id];
        let ilo = [lo[0].max(cell.lo[0]), lo[1].max(cell.lo[1])];
        let ihi = [hi[0].min(cell.hi[0]), hi[1].min(cell.hi[1])];
        match &cell.node {
            Node::Leaf => Some(cell.inside),
            Node::Crossed(among) => {
                // Grown by rounding's reach, so a curve grazing the box counts.
                let g = 1e-12 * ((self.hi[0] - self.lo[0]).abs() + (self.hi[1] - self.lo[1]).abs());
                let (glo, ghi) = ([ilo[0] - g, ilo[1] - g], [ihi[0] + g, ihi[1] + g]);
                let hit = among.iter().any(|&i| meets_box(&self.shifted(i), glo, ghi));
                if hit {
                    None
                } else {
                    let c = [0.5 * (ilo[0] + ihi[0]), 0.5 * (ilo[1] + ihi[1])];
                    Some(cell.inside ^ self.parity(cell.r, c, among))
                }
            }
            Node::Split(kids) => {
                let mut answer: Option<bool> = None;
                for &k in kids {
                    let kc = &self.cells[k as usize];
                    if !overlaps(kc.lo, kc.hi, lo, hi) {
                        continue;
                    }
                    let a = self.class_in(k as usize, lo, hi)?;
                    match answer {
                        None => answer = Some(a),
                        Some(b) if b != a => return None,
                        _ => {}
                    }
                }
                answer
            }
        }
    }

    /// Appends these loops to `p` for a GPU kernel (icam-sim's partgpu.rs), every index
    /// global, and returns the face's header: the first cell, 1 when there are no pieces
    /// (the whole surface), the domain's low and high corners, and each direction's period
    /// (0 for none). A cell is 12 words: its low and high corners, its reference point (as
    /// f32 bits), whether the reference is inside, its kind (0 leaf, 1 crossed, 2 split),
    /// then for a crossed cell where its copies' list starts and how long it is, for a
    /// split one its four children (child k = (u >= mid u) + 2 (v >= mid v)). A copy is its
    /// piece and its shift; a piece its four control points.
    pub fn pack_into(&self, p: &mut Packed) -> [u32; 10] {
        let (c0, i0, p0) = (p.cells.len() as u32, p.inst.len() as u32, p.pieces.len() as u32);
        for pc in &self.pieces {
            let c = pc.c;
            p.pieces.push([c[0][0] as f32, c[0][1] as f32, c[1][0] as f32, c[1][1] as f32, c[2][0] as f32, c[2][1] as f32, c[3][0] as f32, c[3][1] as f32]);
        }
        for &(pi, sh) in &self.instances {
            p.inst.push([f32::from_bits(p0 + pi), sh[0] as f32, sh[1] as f32, 0.0]);
        }
        for cell in &self.cells {
            let f = |x: f64| (x as f32).to_bits();
            let mut w = [f(cell.lo[0]), f(cell.lo[1]), f(cell.hi[0]), f(cell.hi[1]), f(cell.r[0]), f(cell.r[1]), cell.inside as u32, 0, 0, 0, 0, 0];
            match &cell.node {
                Node::Leaf => {}
                Node::Crossed(among) => {
                    w[7] = 1;
                    w[8] = p.among.len() as u32;
                    w[9] = among.len() as u32;
                    p.among.extend(among.iter().map(|&i| i0 + i));
                }
                Node::Split(kids) => {
                    w[7] = 2;
                    for k in 0..4 {
                        w[8 + k] = c0 + kids[k];
                    }
                }
            }
            p.cells.push(w);
        }
        let f = |x: f64| (x as f32).to_bits();
        [c0, self.pieces.is_empty() as u32, f(self.lo[0]), f(self.lo[1]), f(self.hi[0]), f(self.hi[1]), f(self.period[0].unwrap_or(0.0)), f(self.period[1].unwrap_or(0.0)), 0, 0]
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

// ---------------------------------------------------------------- analytic faces

/// An analytic face's surface written exactly as a rational B-spline, so its trim
/// loops are built and certified by the same construction as a B-spline face's, and
/// the closed-form map from a point to its parameters.
///
/// Circles are rational quadratic quarter arcs (weights 1, cos 45 degrees, 1), the
/// first starting on the frame's x axis. On such an arc the tangent of the half
/// angle from the arc's middle is linear in the arc's parameter,
/// tan(alpha / 2) = tan(pi / 8) (2 t - 1), so the parameter of a point at a known
/// angle is closed form. Parameters, with Su x Sv along the surface's own normal:
/// - plane: (x, y) in the plane's frame, over a rectangle about the face;
/// - cylinder and cone: (arc parameter in [0, 4], height along the axis), the
///   height range about the face's;
/// - sphere: (longitude arc parameter in [0, 4], latitude arc parameter in [0, 2]);
/// - torus: (arc parameter around the axis, arc parameter around the tube). A ring
///   or horn torus (major at least minor) turns its whole tube circle from the outer
///   equator, in [0, 4]; an apple or lemon keeps the arc on the axis's own side,
///   between the two angles where it meets the axis, in equal arcs of at most 90
///   degrees.
pub struct Analytic {
    pub ns: NurbsSurf,
    kind: Kind,
}

/// Loops packed for a GPU kernel (`Trim2::pack_into`).
#[derive(Default)]
pub struct Packed {
    pub cells: Vec<[u32; 12]>,
    pub among: Vec<u32>,
    pub inst: Vec<[f32; 4]>,
    pub pieces: Vec<[f32; 8]>,
}

impl Analytic {
    /// The point-to-parameters map for a GPU kernel: its kind (0 plane, 1 cylinder, 2
    /// cone, 3 sphere, 4 torus), its frame's origin, x and z, and its constants (cone: r,
    /// a; torus: major, psi0, span, arcs).
    pub fn pack(&self) -> (u32, Frame, [f64; 4]) {
        match &self.kind {
            Kind::Plane(f) => (0, *f, [0.0; 4]),
            Kind::Cylinder(f) => (1, *f, [0.0; 4]),
            Kind::Cone(f, r, a) => (2, *f, [*r, *a, 0.0, 0.0]),
            Kind::Sphere(f) => (3, *f, [0.0; 4]),
            Kind::Torus(f, major, psi0, span, n) => (4, *f, [*major, *psi0, *span, *n as f64]),
        }
    }
}

impl std::fmt::Debug for Analytic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Analytic {{ u {:?}, v {:?} }}", self.ns.u_range, self.ns.v_range)
    }
}

enum Kind {
    Plane(Frame),
    Cylinder(Frame),
    Cone(Frame, f64, f64),
    Sphere(Frame),
    /// Frame, major radius, and the tube angles the meridian covers: from psi0
    /// through span (a full turn for a ring torus).
    Torus(Frame, f64, f64, f64, usize),
}

const QW: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// The control points (unit radius, in the (cos, sin) plane) and weights of `n`
/// quarter arcs from angle `a0`.
fn arcs(a0: f64, n: usize) -> (Vec<(f64, f64)>, Vec<f64>) {
    arcs_over(a0, std::f64::consts::FRAC_PI_2 * n as f64, n)
}

/// The same for `n` equal arcs covering `span` (each at most 90 degrees): an arc of
/// half angle phi has its middle control point at 1 / cos(phi) and weight cos(phi).
fn arcs_over(a0: f64, span: f64, n: usize) -> (Vec<(f64, f64)>, Vec<f64>) {
    let q = span / n as f64;
    let w = (0.5 * q).cos();
    let mut pts = Vec::new();
    let mut ws = Vec::new();
    for k in 0..n {
        let a = a0 + q * k as f64;
        let m = a + 0.5 * q;
        pts.push((a.cos(), a.sin()));
        ws.push(1.0);
        pts.push((m.cos() / w, m.sin() / w));
        ws.push(w);
    }
    let a = a0 + span;
    pts.push((a.cos(), a.sin()));
    ws.push(1.0);
    let _ = QW;
    (pts, ws)
}

/// Knots of `n` quarter arcs: parameter k at the start of arc k.
fn arc_knots(n: usize) -> Vec<f64> {
    let mut k = vec![0.0, 0.0, 0.0];
    for i in 1..n {
        k.push(i as f64);
        k.push(i as f64);
    }
    k.extend([n as f64; 3]);
    k
}

/// The arc parameter of angle `a` on `n` quarter arcs from `a0` (a reduced into the
/// arcs' range first when they make a full turn).
fn arc_param(a: f64, a0: f64, n: usize) -> f64 {
    arc_param_over(a, a0, std::f64::consts::FRAC_PI_2 * n as f64, n)
}

/// The same on `n` equal arcs covering `span`: on an arc of half angle phi, the
/// tangent of half the angle from its middle is tan(phi / 2) (2 t - 1).
fn arc_param_over(a: f64, a0: f64, span: f64, n: usize) -> f64 {
    let q = span / n as f64;
    let mut x = a - a0;
    if (span - std::f64::consts::TAU).abs() < 1e-12 {
        x = x.rem_euclid(std::f64::consts::TAU);
    } else {
        // The nearer end for an angle outside the arcs.
        let wrap = |y: f64| y - std::f64::consts::TAU * ((y - 0.5 * span) / std::f64::consts::TAU).round();
        x = wrap(x).clamp(0.0, span);
    }
    let k = ((x / q).floor() as i64).clamp(0, n as i64 - 1) as usize;
    let alpha = x - q * k as f64 - 0.5 * q;
    let t = 0.5 * ((0.5 * alpha).tan() / (0.25 * q).tan() + 1.0);
    k as f64 + t.clamp(0.0, 1.0)
}

impl Analytic {
    /// The exact rational B-spline form of an analytic surface. `range` is the
    /// face's extent: for a plane (x0, x1, y0, y1) in its frame; for a cylinder or
    /// cone (h0, h1, _, _) along its axis; unused otherwise. None for a B-spline
    /// surface.
    pub fn of(surf: &Surf, range: [f64; 4]) -> Option<Analytic> {
        Analytic::of_with(surf, range, &[])
    }

    /// The same, for a sphere with its poles placed as far as possible from the
    /// given points (a face's boundary): a sphere's parametrization is singular only
    /// at its poles, and where they lie is free. Of 49 candidate axes, the one whose
    /// poles make the largest least angle to the points.
    pub fn of_with(surf: &Surf, range: [f64; 4], boundary: &[V3]) -> Option<Analytic> {
        if let (Surf::Sphere { f, r }, false) = (surf, boundary.is_empty()) {
            let mut axes: Vec<V3> = Vec::new();
            for i in -1i32..=1 {
                for j in -1i32..=1 {
                    for k in 0i32..=1 {
                        if (i, j, k) > (0, 0, 0) || k == 1 {
                            axes.push(V3::new(i as f64, j as f64, k as f64).unit());
                        }
                    }
                }
            }
            let mut extra: Vec<V3> = Vec::new();
            for a in &axes {
                for b in &axes {
                    let c = *a + *b;
                    if c.len() > 1e-9 {
                        extra.push(c.unit());
                    }
                }
            }
            axes.extend(extra.into_iter().take(36));
            let dirs: Vec<V3> = boundary.iter().map(|p| *p - f.o).filter(|d| d.len() > 0.0).map(|d| d.unit()).collect();
            let worst = |a: &V3| dirs.iter().map(|d| d.dot(*a).abs()).fold(0.0, f64::max);
            let best = axes.iter().copied().min_by(|a, b| worst(a).total_cmp(&worst(b)))?;
            let z = f.dir(best);
            let x0 = if z.cross(f.x).len() > 0.3 { f.x } else { f.y() };
            let x = (x0 - z * x0.dot(z)).unit();
            let frame = Frame { o: f.o, x, z };
            return Analytic::of_with(&Surf::Sphere { f: frame, r: *r }, range, &[]);
        }
        let z = |f: &Frame| f.z;
        let row = |f: &Frame, pts: &[(f64, f64)], radial: f64, height: f64| -> Vec<V3> {
            pts.iter().map(|&(c, s)| f.o + f.x * (c * radial) + f.y() * (s * radial) + z(f) * height).collect::<Vec<_>>()
        };
        let (ns, kind) = match surf {
            Surf::Plane { f } => {
                let [x0, x1, y0, y1] = range;
                let ctrl: Vec<Vec<V3>> = [x0, x1].iter().map(|&x| [y0, y1].iter().map(|&y| f.o + f.x * x + f.y() * y).collect()).collect();
                (NurbsSurf::new(1, 1, &ctrl, None, &[x0, x0, x1, x1], &[y0, y0, y1, y1]).ok()?, Kind::Plane(f.clone()))
            }
            Surf::Cylinder { f, r } => {
                let (pts, w) = arcs(0.0, 4);
                let [h0, h1, _, _] = range;
                let ctrl: Vec<Vec<V3>> = (0..pts.len()).map(|i| [h0, h1].iter().map(|&h| row(f, &pts[i..i + 1], *r, h)[0]).collect()).collect();
                let wts: Vec<Vec<f64>> = w.iter().map(|&wi| vec![wi, wi]).collect();
                (NurbsSurf::new(2, 1, &ctrl, Some(&wts), &arc_knots(4), &[h0, h0, h1, h1]).ok()?, Kind::Cylinder(f.clone()))
            }
            Surf::Cone { f, r, a } => {
                let (pts, w) = arcs(0.0, 4);
                let ha = -r / a.tan();
                let [h0, h1, _, _] = range;
                let (h0, h1) = (h0.max(ha), h1.max(ha));
                let rho = |h: f64| r + h * a.tan();
                let ctrl: Vec<Vec<V3>> = (0..pts.len()).map(|i| [h0, h1].iter().map(|&h| row(f, &pts[i..i + 1], rho(h), h)[0]).collect()).collect();
                let wts: Vec<Vec<f64>> = w.iter().map(|&wi| vec![wi, wi]).collect();
                (NurbsSurf::new(2, 1, &ctrl, Some(&wts), &arc_knots(4), &[h0, h0, h1, h1]).ok()?, Kind::Cone(f.clone(), *r, *a))
            }
            Surf::Sphere { f, r } => {
                let (pts, w) = arcs(0.0, 4);
                let (mer, mw) = arcs(-std::f64::consts::FRAC_PI_2, 2);
                let ctrl: Vec<Vec<V3>> = (0..pts.len()).map(|i| mer.iter().map(|&(mr, mz)| row(f, &pts[i..i + 1], r * mr, r * mz)[0]).collect()).collect();
                let wts: Vec<Vec<f64>> = w.iter().map(|&wi| mw.iter().map(|&wj| wi * wj).collect()).collect();
                (NurbsSurf::new(2, 2, &ctrl, Some(&wts), &arc_knots(4), &arc_knots(2)).ok()?, Kind::Sphere(f.clone()))
            }
            Surf::Torus { f, major, minor } => {
                // A ring (or horn) torus turns its whole tube circle; an apple or lemon
                // keeps the arc on the axis's own side, where major + minor cos(psi) >= 0.
                let (psi0, span) = if *major >= *minor {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let c = (-major / minor).clamp(-1.0, 1.0);
                    let p = c.acos();
                    (-p, 2.0 * p)
                };
                if !(span > 1e-9) {
                    return None;
                }
                let n = ((span / std::f64::consts::FRAC_PI_2) - 1e-9).ceil().max(1.0) as usize;
                let (pts, w) = arcs(0.0, 4);
                let (mer, mw) = arcs_over(psi0, span, n);
                let ctrl: Vec<Vec<V3>> = (0..pts.len()).map(|i| mer.iter().map(|&(mr, mz)| row(f, &pts[i..i + 1], major + minor * mr, minor * mz)[0]).collect()).collect();
                let wts: Vec<Vec<f64>> = w.iter().map(|&wi| mw.iter().map(|&wj| wi * wj).collect()).collect();
                (NurbsSurf::new(2, 2, &ctrl, Some(&wts), &arc_knots(4), &arc_knots(n)).ok()?, Kind::Torus(f.clone(), *major, psi0, span, n))
            }
            Surf::Nurbs(_) => return None,
        };
        Some(Analytic { ns, kind })
    }

    /// The parameters of the point of the surface nearest q, in closed form.
    /// Whether the form closes on itself along u and along v: every kind but the
    /// plane turns a full circle in u; a ring or horn torus also in v.
    pub fn closed(&self) -> (bool, bool) {
        match &self.kind {
            Kind::Plane(_) => (false, false),
            Kind::Cylinder(_) | Kind::Cone(..) | Kind::Sphere(_) => (true, false),
            Kind::Torus(_, _, _, span, _) => (true, *span >= std::f64::consts::TAU),
        }
    }

    pub fn uv(&self, q: V3) -> (f64, f64) {
        match &self.kind {
            Kind::Plane(f) => {
                let l = f.local(q);
                (l.x, l.y)
            }
            Kind::Cylinder(f) => {
                let ax = axial(f, q);
                (arc_param(ax.theta, 0.0, 4), ax.h)
            }
            Kind::Cone(f, r, a) => {
                // The foot's height: along the generatrix through q's angle.
                let ax = axial(f, q);
                let (sa, ca) = a.sin_cos();
                let ha = -r / a.tan();
                let s = (ax.rho * sa + (ax.h - ha) * ca).max(0.0);
                (arc_param(ax.theta, 0.0, 4), ha + s * ca)
            }
            Kind::Sphere(f) => {
                let ax = axial(f, q);
                let lat = ax.h.atan2(ax.rho);
                (arc_param(ax.theta, 0.0, 4), arc_param(lat, -std::f64::consts::FRAC_PI_2, 2))
            }
            Kind::Torus(f, major, psi0, span, n) => {
                let ax = axial(f, q);
                let psi = ax.h.atan2(ax.rho - major);
                (arc_param(ax.theta, 0.0, 4), arc_param_over(psi, *psi0, *span, *n))
            }
        }
    }
}

#[cfg(test)]
mod analytic_tests {
    use super::*;

    /// A lemon's closed-form feet (THE PART's radii) land on its form at their
    /// parameters.
    #[test]
    fn lemon_feet_invert() {
        let f = frame();
        let surf = Surf::Torus { f, major: -66.600473569, minor: 99.5 };
        let an = Analytic::of(&surf, [0.0; 4]).unwrap();
        for p in [V3::new(300.0, 10.0, -200.0), V3::new(5.0, 3.0, 40.0), V3::new(-80.0, 20.0, -10.0)] {
            let q = surf.foot(p);
            let (u, v) = an.uv(q);
            assert!((an.ns.eval(u, v).p - q).len() < 1e-9, "foot of {p:?}");
        }
    }

    fn frame() -> Frame {
        let z = V3::new(0.3, -0.2, 0.93).unit();
        let x = V3::new(1.0, 0.0, 0.0);
        let x = (x - z * x.dot(z)).unit();
        Frame { o: V3::new(1.5, -2.0, 0.7), x, z }
    }

    /// Each form is the surface (its points are on it) and its inverse returns the
    /// parameters a point came from, and Su x Sv is the surface's own normal.
    #[test]
    fn forms_are_exact_and_invert() {
        let f = frame();
        let cases = [
            Surf::Plane { f: f.clone() },
            Surf::Cylinder { f: f.clone(), r: 2.5 },
            Surf::Cone { f: f.clone(), r: 2.0, a: 0.4 },
            Surf::Sphere { f: f.clone(), r: 3.0 },
            Surf::Torus { f: f.clone(), major: 5.0, minor: 1.2 },
            Surf::Torus { f: f.clone(), major: 3.0, minor: 3.0 },
            Surf::Torus { f: f.clone(), major: 2.0, minor: 3.0 },
            Surf::Torus { f: f.clone(), major: -1.5, minor: 4.0 },
        ];
        for surf in &cases {
            let an = Analytic::of(surf, [-3.0, 4.0, -2.0, 5.0]).unwrap();
            let (u0, u1, v0, v1) = (an.ns.u_range.0, an.ns.u_range.1, an.ns.v_range.0, an.ns.v_range.1);
            for i in 1..40 {
                for j in 1..40 {
                    let (u, v) = (u0 + (u1 - u0) * i as f64 / 40.0, v0 + (v1 - v0) * j as f64 / 40.0);
                    let e = an.ns.eval(u, v);
                    let on = (surf.foot(e.p) - e.p).len();
                    assert!(on < 1e-12, "{}: point off the surface by {on:e}", surf.name());
                    let (uu, vv) = an.uv(e.p);
                    // Where a parameter direction collapses (a pole, a horn torus's
                    // center) every value of it is the same point.
                    let u_free = e.ps.len() < 1e-9;
                    let v_free = e.pt.len() < 1e-9;
                    assert!((u_free || (uu - u).abs() < 1e-10) && (v_free || (vv - v).abs() < 1e-10), "{}: ({u}, {v}) came back as ({uu}, {vv})", surf.name());
                    let n = e.ps.cross(e.pt);
                    if n.len() > 1e-9 {
                        assert!(n.unit().dot(surf.normal(e.p)) > 1.0 - 1e-9, "{}: Su x Sv against the surface's normal", surf.name());
                    }
                }
            }
        }
    }
}
