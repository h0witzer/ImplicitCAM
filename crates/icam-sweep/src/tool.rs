//! The cutter as a solid: its profile of revolution (icam_core::tool) about the tool
//! axis, from the tip up to its stick-out, where the holder begins. The holder collides
//! but does not cut, so it is not part of what a move removes.
//!
//! The solid is { (rho, h) : 0 <= h <= length, rho <= R(h) } in the profile's half-plane,
//! R the profile's radius at height h. A solid of revolution's exact distance is the
//! distance in that half-plane, to the profile's elements (cut at the length) and the cap
//! across the top; the axis itself is not a boundary.

use std::f64::consts::FRAC_PI_2;

use icam_core::tool::{ProfEl, Tool};
use icam_post::ToolDef;

use crate::geom::{Aabb, Pose, V3};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Piece {
    Seg { a: (f64, f64), b: (f64, f64) },
    /// Part of a circle about `c` from angle `a0` to `a1` (radians from the +rho axis).
    Arc { c: (f64, f64), rad: f64, a0: f64, a1: f64 },
}

fn seg_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}

impl Piece {
    /// The piece's farthest point in direction `d` of the half-plane.
    fn support(&self, d: (f64, f64)) -> (f64, f64) {
        let better = |x: (f64, f64), y: (f64, f64)| if x.0 * d.0 + x.1 * d.1 >= y.0 * d.0 + y.1 * d.1 { x } else { y };
        match *self {
            Piece::Seg { a, b } => better(a, b),
            Piece::Arc { c, rad, a0, a1 } => {
                let end = |a: f64| (c.0 + rad * a.cos(), c.1 + rad * a.sin());
                let ang = d.1.atan2(d.0);
                let ends = better(end(a0), end(a1));
                if ang >= a0 && ang <= a1 {
                    better(end(ang), ends)
                } else {
                    ends
                }
            }
        }
    }

    fn dist(&self, p: (f64, f64)) -> f64 {
        match *self {
            Piece::Seg { a, b } => seg_dist(p, a, b),
            Piece::Arc { c, rad, a0, a1 } => {
                let (dx, dy) = (p.0 - c.0, p.1 - c.1);
                let ang = dy.atan2(dx);
                if ang >= a0 && ang <= a1 {
                    (dx.hypot(dy) - rad).abs()
                } else {
                    let end = |a: f64| (p.0 - c.0 - rad * a.cos()).hypot(p.1 - c.1 - rad * a.sin());
                    end(a0).min(end(a1))
                }
            }
        }
    }
}

/// One stretch of the profile's boundary, walked from the tip outward and up and back
/// in across the cap (counterclockwise in the (rho, h) half-plane, so the outward
/// normal is to the right of the walk). A corner carries every normal between the two
/// stretches it joins.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Elem {
    Seg { a: (f64, f64), b: (f64, f64), n: (f64, f64) },
    Arc { c: (f64, f64), rad: f64, a0: f64, a1: f64 },
    Corner { p: (f64, f64), t0: f64, t1: f64 },
}

impl Elem {
    /// Position and outward normal at u in [0, 1].
    pub(crate) fn at(&self, u: f64) -> ((f64, f64), (f64, f64)) {
        match *self {
            Elem::Seg { a, b, n } => ((a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u), n),
            Elem::Arc { c, rad, a0, a1 } => {
                let (s, k) = (a0 + (a1 - a0) * u).sin_cos();
                ((c.0 + rad * k, c.1 + rad * s), (k, s))
            }
            Elem::Corner { p, t0, t1 } => {
                let (s, k) = (t0 + (t1 - t0) * u).sin_cos();
                (p, (k, s))
            }
        }
    }

    /// The normal's angle at the start and the end.
    fn turn(&self) -> (f64, f64) {
        match *self {
            Elem::Seg { n, .. } => (n.1.atan2(n.0), n.1.atan2(n.0)),
            Elem::Arc { a0, a1, .. } => (a0, a1),
            Elem::Corner { t0, t1, .. } => (t0, t1),
        }
    }

    /// Steps along u for chords within `tol` and pieces no longer than `longest` (a
    /// corner by its turn).
    pub(crate) fn steps(&self, tol: f64, longest: f64) -> usize {
        match *self {
            Elem::Seg { a, b, .. } => ((b.0 - a.0).hypot(b.1 - a.1) / longest).ceil().clamp(1.0, 4096.0) as usize,
            Elem::Arc { rad, a0, a1, .. } => ((a1 - a0).abs() / (8.0 * tol / rad).sqrt().min(0.2).min(longest / rad)).ceil().clamp(1.0, 1024.0) as usize,
            Elem::Corner { t0, t1, .. } => ((t1 - t0).abs() / 0.1).ceil().clamp(1.0, 32.0) as usize,
        }
    }
}

/// The profile's slope dr/dh along the boundary, from the tip up, never increases: the
/// radius is a concave function of height, so the solid is convex. A flat bottom at the
/// tip is the infinite slope the chain starts from; the cap closes the top.
fn convex(pieces: &[Piece]) -> bool {
    let mut last = f64::INFINITY;
    for p in &pieces[..pieces.len().saturating_sub(1)] {
        let (start, end) = match *p {
            Piece::Seg { a, b } => {
                let s = if b.1 - a.1 > 1e-12 { (b.0 - a.0) / (b.1 - a.1) } else { f64::INFINITY };
                (s, s)
            }
            // The lower-right quarter: horizontal at the bottom, vertical at the side.
            // Going up the arc at angle a the boundary runs along (-sin a, cos a).
            Piece::Arc { a0, a1, .. } => (-a0.tan(), -a1.tan()),
        };
        if start > last + 1e-9 && last.is_finite() {
            return false;
        }
        last = end;
    }
    true
}

#[derive(Clone, Debug)]
pub struct ToolSolid {
    pub profile: Tool,
    /// Stick-out: the solid runs from the tip (h = 0) to here.
    pub length: f64,
    pieces: Vec<Piece>,
    radius: f64,
    convex: bool,
}

impl ToolSolid {
    pub fn new(profile: Tool, length: f64) -> ToolSolid {
        let mut pieces = Vec::new();
        let mut top = (0.0, 0.0);
        for e in &profile.els {
            match *e {
                ProfEl::Seg { r0, h0, r1, h1 } => {
                    if h0 >= length {
                        break;
                    }
                    let b = if h1 > length { (r0 + (r1 - r0) * (length - h0) / (h1 - h0), length) } else { (r1, h1) };
                    if (b.0 - r0).hypot(b.1 - h0) > 0.0 {
                        pieces.push(Piece::Seg { a: (r0, h0), b });
                    }
                    top = b;
                    if h1 > length {
                        break;
                    }
                }
                ProfEl::Arc { rc, hc, rad } => {
                    if hc - rad >= length {
                        break;
                    }
                    let a1 = if hc > length { ((length - hc) / rad).asin() } else { 0.0 };
                    pieces.push(Piece::Arc { c: (rc, hc), rad, a0: -FRAC_PI_2, a1 });
                    top = (rc + rad * a1.cos(), hc + rad * a1.sin());
                    if hc > length {
                        break;
                    }
                }
            }
        }
        if top.1 < length {
            pieces.push(Piece::Seg { a: top, b: (top.0, length) });
            top = (top.0, length);
        }
        pieces.push(Piece::Seg { a: (top.0, length), b: (0.0, length) });
        let radius = pieces
            .iter()
            .map(|p| match *p {
                Piece::Seg { a, b } => a.0.max(b.0),
                Piece::Arc { c, rad, a1, .. } => c.0 + rad * a1.cos().max(0.0),
            })
            .fold(0.0, f64::max);
        let convex = convex(&pieces);
        ToolSolid { profile, length, pieces, radius, convex }
    }

    /// True when the solid is convex: its radius never grows faster with height than it
    /// did lower down (no step out, as a shank wider than the flutes makes).
    pub fn convex(&self) -> bool {
        self.convex
    }

    /// The cutter a tool sheet describes, `length` long.
    pub fn from_def(def: &ToolDef, length: f64) -> ToolSolid {
        let (d, rc) = (def.diameter, def.corner_radius.max(0.0));
        let profile = if def.kind == "DRILL" {
            Tool::vee(d, def.tip_angle_deg, length)
        } else if def.kind == "BALL" || rc >= 0.5 * d - 1e-9 {
            Tool::ball(d, length)
        } else if rc > 1e-9 {
            Tool::bull(d, rc, length)
        } else {
            Tool::flat(d, length)
        };
        ToolSolid::new(profile, length)
    }

    /// Exact signed distance (negative inside) at q in the tool frame (tip at the
    /// origin, +z up the axis).
    pub fn sd(&self, q: V3) -> f64 {
        let p = (q.x.hypot(q.y), q.z);
        let d = self.pieces.iter().map(|e| e.dist(p)).fold(f64::INFINITY, f64::min);
        let inside = p.1 >= 0.0 && p.1 <= self.length && p.0 <= self.profile.radius_at(p.1).0;
        if inside {
            -d
        } else {
            d
        }
    }

    /// The solid's farthest point in direction `d` (work frame) when placed at `pose`.
    pub fn support(&self, pose: Pose, d: V3) -> V3 {
        let l = pose.rot.inverse().apply(d);
        let rd = l.x.hypot(l.y);
        let dir = (rd, l.z);
        let (r, h) = self.pieces.iter().map(|p| p.support(dir)).fold((0.0, 0.0), |m, x| if x.0 * dir.0 + x.1 * dir.1 > m.0 * dir.0 + m.1 * dir.1 { x } else { m });
        let q = if rd > 1e-300 { V3::new(r * l.x / rd, r * l.y / rd, h) } else { V3::new(r, 0.0, h) };
        pose.apply(q)
    }

    /// The profile's boundary from the tip up to the cap across the top (the last piece).
    pub(crate) fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// The widest radius within the length.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The boundary of the profile, stretch by stretch with the corners between.
    pub(crate) fn elements(&self) -> Vec<Elem> {
        let mut out: Vec<Elem> = Vec::new();
        for p in &self.pieces {
            let e = match *p {
                Piece::Seg { a, b } => {
                    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                    let l = dx.hypot(dy);
                    Elem::Seg { a, b, n: (dy / l, -dx / l) }
                }
                Piece::Arc { c, rad, a0, a1 } => Elem::Arc { c, rad, a0, a1 },
            };
            if let Some(last) = out.last() {
                let (t0, t1) = (last.turn().1, e.turn().0);
                if t1 - t0 > 1e-9 {
                    let p = match *last {
                        Elem::Seg { b, .. } => b,
                        _ => last.at(1.0).0,
                    };
                    out.push(Elem::Corner { p, t0, t1 });
                }
            }
            out.push(e);
        }
        out
    }

    /// The farthest the solid reaches from the tip.
    pub fn reach(&self) -> f64 {
        self.radius.hypot(self.length)
    }

    /// A box holding the tool at `pose`: the cylinder of its widest radius from the tip
    /// to its length.
    pub fn bounds(&self, pose: Pose) -> Aabb {
        let (a, b) = (pose.at, pose.apply(V3::new(0.0, 0.0, self.length)));
        let ax = pose.axis();
        let e = |c: f64| self.radius * (1.0 - c * c).max(0.0).sqrt();
        let r = V3::new(e(ax.x), e(ax.y), e(ax.z));
        Aabb::new(V3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)) - r, V3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)) + r)
    }
}
