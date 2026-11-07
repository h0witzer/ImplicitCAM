//! Exact plane geometry for horizontal faces: points, line, arc and native
//! B-spline boundary curves, closest points, and ray crossings for inside tests.
//!
//! Faces are regions in a horizontal plane bounded by closed loops of curves.
//! Loops are oriented with the region on their left (outer loops
//! counterclockwise, holes clockwise), so a curve's outward normal is the right
//! perpendicular of its direction of travel.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use crate::bspline::Spline2;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct V2 {
    pub x: f64,
    pub y: f64,
}

impl V2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn dot(self, o: V2) -> f64 {
        self.x * o.x + self.y * o.y
    }
    pub fn cross(self, o: V2) -> f64 {
        self.x * o.y - self.y * o.x
    }
    pub fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn unit(self) -> V2 {
        let l = self.len();
        if l > 0.0 {
            self * (1.0 / l)
        } else {
            V2::default()
        }
    }
    /// Left perpendicular.
    pub fn perp(self) -> V2 {
        V2::new(-self.y, self.x)
    }
    /// Right perpendicular: the outward normal of a curve travelling along `self`.
    pub fn rperp(self) -> V2 {
        V2::new(self.y, -self.x)
    }
    pub fn rotate(self, ang: f64) -> V2 {
        let (s, c) = ang.sin_cos();
        V2::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }
}

impl Add for V2 {
    type Output = V2;
    fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for V2 {
    type Output = V2;
    fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
}
impl Mul<f64> for V2 {
    type Output = V2;
    fn mul(self, s: f64) -> V2 {
        V2::new(self.x * s, self.y * s)
    }
}
impl Neg for V2 {
    type Output = V2;
    fn neg(self) -> V2 {
        V2::new(-self.x, -self.y)
    }
}

/// A boundary curve of a horizontal face.
#[derive(Clone, Debug)]
pub enum Curve {
    Line { a: V2, b: V2 },
    /// Arc about `c` of radius `r`, starting at `a` and turning through the
    /// signed angle `sweep` (radians, positive = counterclockwise).
    /// |sweep| = 2 pi is a full circle that starts and ends at `a`.
    Arc { c: V2, r: f64, a: V2, sweep: f64 },
    /// A native rational B-spline, traversed in the direction of its pieces.
    Spline(Arc<Spline2>),
}

/// Where on a curve the closest point landed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Near {
    Interior,
    Start,
    End,
}

/// The foot of a point on a curve: where, how far, whether at an end, the unit
/// direction of travel and signed curvature there (positive = turning left),
/// and the curve parameter (line: fraction, arc: fraction of the sweep,
/// spline: the source NURBS parameter).
#[derive(Clone, Copy, Debug)]
pub struct Closest {
    pub q: V2,
    pub dist: f64,
    pub near: Near,
    pub tangent: V2,
    pub curvature: f64,
    pub t: f64,
}

impl Closest {
    /// Outward normal (away from the face region) at the foot.
    pub fn outward(&self) -> V2 {
        self.tangent.rperp()
    }

    /// Hessian (xx, xy, yy) of the unsigned distance to the curve at `p`.
    /// At an end point the distance is to a point: (I - n n^T) / d. At an
    /// interior foot with unit tangent t it is k / (1 + k d) t t^T, where k is
    /// the curvature counted positive when the curve bends away from p.
    pub fn dist_hessian(&self, p: V2) -> [f64; 3] {
        if self.dist <= 0.0 {
            return [0.0; 3];
        }
        let n = (p - self.q) * (1.0 / self.dist);
        if self.near != Near::Interior {
            let k = 1.0 / self.dist;
            return [k * (1.0 - n.x * n.x), -k * n.x * n.y, k * (1.0 - n.y * n.y)];
        }
        let t = self.tangent;
        let k = -self.curvature * n.dot(t.perp());
        let den = 1.0 + k * self.dist;
        if den.abs() < 1e-300 {
            return [0.0; 3];
        }
        let f = k / den;
        [f * t.x * t.x, f * t.x * t.y, f * t.y * t.y]
    }
}

/// Angle of `u` measured from `from` in the direction of `sweep`, in [0, 2 pi).
fn swept_angle(from: V2, u: V2, sweep: f64) -> f64 {
    let t = from.cross(u).atan2(from.dot(u));
    let t = if sweep >= 0.0 { t } else { -t };
    if t < 0.0 {
        t + 2.0 * PI
    } else {
        t
    }
}

impl Curve {
    /// The same curve traversed the other way.
    pub fn reversed(&self) -> Curve {
        match self {
            Curve::Line { a, b } => Curve::Line { a: *b, b: *a },
            Curve::Arc { c, r, sweep, .. } => Curve::Arc { c: *c, r: *r, a: self.end(), sweep: -sweep },
            Curve::Spline(s) => Curve::Spline(Arc::new(s.reversed())),
        }
    }

    pub fn start(&self) -> V2 {
        match self {
            Curve::Line { a, .. } => *a,
            Curve::Arc { a, .. } => *a,
            Curve::Spline(s) => s.start(),
        }
    }

    pub fn end(&self) -> V2 {
        match self {
            Curve::Line { b, .. } => *b,
            Curve::Arc { c, a, sweep, .. } => *c + (*a - *c).rotate(*sweep),
            Curve::Spline(s) => s.end(),
        }
    }

    pub fn is_full_circle(&self) -> bool {
        matches!(*self, Curve::Arc { sweep, .. } if (sweep.abs() - 2.0 * PI).abs() < 1e-12)
    }

    fn arc_tangent(c: V2, sweep: f64, q: V2) -> V2 {
        let t = (q - c).perp().unit();
        if sweep >= 0.0 {
            t
        } else {
            -t
        }
    }

    /// Unit direction of travel at the start and at the end.
    pub fn tangent_start(&self) -> V2 {
        match self {
            Curve::Line { a, b } => (*b - *a).unit(),
            Curve::Arc { c, a, sweep, .. } => Curve::arc_tangent(*c, *sweep, *a),
            Curve::Spline(s) => s.tangent_start(),
        }
    }

    pub fn tangent_end(&self) -> V2 {
        match self {
            Curve::Line { a, b } => (*b - *a).unit(),
            Curve::Arc { c, sweep, .. } => Curve::arc_tangent(*c, *sweep, self.end()),
            Curve::Spline(s) => s.tangent_end(),
        }
    }

    /// Outward normals at the ends.
    pub fn outward_start(&self) -> V2 {
        self.tangent_start().rperp()
    }

    pub fn outward_end(&self) -> V2 {
        self.tangent_end().rperp()
    }

    pub fn closest(&self, p: V2) -> Closest {
        self.closest_within(p, f64::INFINITY).expect("a curve always has a closest point")
    }

    /// Closest point, or None if every point is at least `cutoff` away (splines
    /// use the cutoff to prune; lines and arcs always answer).
    pub fn closest_within(&self, p: V2, cutoff: f64) -> Option<Closest> {
        match self {
            Curve::Line { a, b } => {
                let (a, b) = (*a, *b);
                let ab = b - a;
                let l2 = ab.dot(ab);
                let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
                let q = a + ab * t;
                let near = if t <= 0.0 {
                    Near::Start
                } else if t >= 1.0 {
                    Near::End
                } else {
                    Near::Interior
                };
                Some(Closest { q, dist: (p - q).len(), near, tangent: ab.unit(), curvature: 0.0, t })
            }
            Curve::Arc { c, r, a, sweep } => {
                let (c, r, a, sweep) = (*c, *r, *a, *sweep);
                let u = p - c;
                let s = u.len();
                let full = self.is_full_circle();
                let ang = swept_angle(a - c, u, sweep);
                let curvature = sweep.signum() / r;
                if s > 0.0 && (full || ang <= sweep.abs()) {
                    let q = c + u * (r / s);
                    return Some(Closest {
                        q,
                        dist: (s - r).abs(),
                        near: Near::Interior,
                        tangent: Curve::arc_tangent(c, sweep, q),
                        curvature,
                        t: ang / sweep.abs(),
                    });
                }
                let e = self.end();
                let (da, de) = ((p - a).len(), (p - e).len());
                Some(if da <= de {
                    Closest { q: a, dist: da, near: Near::Start, tangent: self.tangent_start(), curvature, t: 0.0 }
                } else {
                    Closest { q: e, dist: de, near: Near::End, tangent: self.tangent_end(), curvature, t: 1.0 }
                })
            }
            Curve::Spline(sp) => {
                let h = sp.closest(p, cutoff)?;
                let last = sp.pieces.len() - 1;
                let near = if h.piece == 0 && h.s <= 0.0 {
                    Near::Start
                } else if h.piece == last && h.s >= 1.0 {
                    Near::End
                } else {
                    Near::Interior
                };
                Some(Closest { q: h.q, dist: h.dist, near, tangent: h.tangent, curvature: h.curvature, t: h.t })
            }
        }
    }

    /// Bounding box: tight for lines and arcs, the control-point box (which
    /// contains the curve) for splines.
    pub fn bbox(&self) -> (V2, V2) {
        let grow = |lo: &mut V2, hi: &mut V2, p: V2| {
            *lo = V2::new(lo.x.min(p.x), lo.y.min(p.y));
            *hi = V2::new(hi.x.max(p.x), hi.y.max(p.y));
        };
        let (mut lo, mut hi) = (self.start(), self.start());
        match self {
            Curve::Line { b, .. } => grow(&mut lo, &mut hi, *b),
            Curve::Arc { c, r, a, sweep } => {
                grow(&mut lo, &mut hi, self.end());
                for d in [V2::new(1.0, 0.0), V2::new(0.0, 1.0), V2::new(-1.0, 0.0), V2::new(0.0, -1.0)] {
                    if self.is_full_circle() || swept_angle(*a - *c, d, *sweep) <= sweep.abs() {
                        grow(&mut lo, &mut hi, *c + d * *r);
                    }
                }
            }
            Curve::Spline(s) => {
                for z in &s.pieces {
                    for q in z.ctrl() {
                        grow(&mut lo, &mut hi, q);
                    }
                }
            }
        }
        (lo, hi)
    }

    /// Contribution to the region's area by Green's theorem (integral of x dy).
    pub fn area_term(&self) -> f64 {
        match self {
            Curve::Line { a, b } => 0.5 * (a.x + b.x) * (b.y - a.y),
            Curve::Arc { c, r, a, sweep } => {
                // x = cx + r cos t, y = cy + r sin t, t from t0 to t0 + sweep.
                let t0 = (a.y - c.y).atan2(a.x - c.x);
                let t1 = t0 + sweep;
                let f = |t: f64| c.x * r * t.sin() + 0.5 * r * r * (t + t.sin() * t.cos());
                f(t1) - f(t0)
            }
            Curve::Spline(s) => s.area_term(),
        }
    }

    /// Number of times the ray p + t * dir (t > 0) crosses the curve. Tangent
    /// touches count zero; the curve's end point is excluded so shared vertices
    /// are not counted twice.
    pub fn ray_crossings(&self, p: V2, dir: V2) -> u32 {
        match self {
            Curve::Line { a, b } => {
                let e = *b - *a;
                let den = dir.cross(e);
                if den == 0.0 {
                    return 0;
                }
                let w = *a - p;
                let t = w.cross(e) / den;
                let s = w.cross(dir) / den;
                u32::from(t > 0.0 && (0.0..1.0).contains(&s))
            }
            Curve::Arc { c, r, a, sweep } => {
                let w = p - *c;
                let bq = w.dot(dir);
                let cq = w.dot(w) - r * r;
                let disc = bq * bq - cq;
                if disc <= 0.0 {
                    return 0;
                }
                let sq = disc.sqrt();
                let full = self.is_full_circle();
                let mut n = 0;
                for t in [-bq - sq, -bq + sq] {
                    if t <= 0.0 {
                        continue;
                    }
                    let q = p + dir * t;
                    if full || swept_angle(*a - *c, q - *c, *sweep) < sweep.abs() {
                        n += 1;
                    }
                }
                n
            }
            Curve::Spline(s) => s.ray_crossings(p, dir),
        }
    }
}
