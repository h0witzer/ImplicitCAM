//! One move's field: the signed distance to everything the tool passes through,
//! s(p) = min over t in [0, 1] of g(t), g(t) the tool's distance at pose t.
//!
//! Branch and bound over stretches of t, each with a floor under g:
//!
//! A convex tool over a stretch [a, b] stays inside its copy held at the middle's turn
//! and slid along the straight chord from its origin at a to its origin at b, grown by
//! how far the real motion strays from that model: the origin's stray from the chord
//! (zero for a straight move, radius x angle^2 / 8 for an arc) plus the turn's swing of
//! the tool's farthest point. A grown convex body's distance is its distance less the
//! growth, and the slid copy's least distance is a convex problem in the slide (solved
//! by golden section), so the floor is that least less the growth. The tool is a solid
//! of revolution, so only its axis direction's turn counts, not a spin about it. A straight move of a
//! convex tool is exact in one step; elsewhere the growth shrinks with the stretch.
//!
//! That floor is first order in the stretch when the tool tilts. The second floor is
//! second order everywhere: over [a, b] each point of the tool (carried without spin
//! about the axis) stays within (b - a)^2 / 8 times its acceleration of the straight
//! line between where it is at a and at b, and that line lies in the hull of the tool at
//! a and at b. So the tool stays inside that hull grown by the bend, and g is at least
//! the hull's distance less the bend; the hull's distance comes from support points
//! (hull.rs), with a certificate at every step. Inside the hull the same containment
//! bounds the depth: along any unit direction u, no tool pose on the stretch reaches
//! past the farther of the two end tools' support planes plus the bend, so g is at least
//! u . p less that reach; u is the outward normal at the boundary nearest p of the
//! deepest pose found.
//!
//! A tool that is not convex falls back to the rate bound alone: g changes at most as
//! fast as p moves through the tool frame, v + w |p - origin(t)| per unit t.
//!
//! The search is best first: the stretch with the lowest floor is split next (each
//! stretch's ends and middle evaluated, and the model's least), and it ends when the
//! lowest floor left cannot beat the best value of g found by more than the tolerance.
//! The answer is an actual value of g within the tolerance of the least.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::geom::{dot, scale, Aabb, V3};
use crate::hull;
use crate::motion::Motion;
use crate::tool::ToolSolid;

#[derive(Clone, Debug)]
pub struct Sweep {
    pub tool: Arc<ToolSolid>,
    pub motion: Arc<dyn Motion>,
    bounds: Aabb,
}

/// A stretch of t waiting to be split, with a floor under g on it and g at its ends.
struct Stretch {
    floor: f64,
    a: f64,
    b: f64,
    ga: f64,
    gb: f64,
}

impl PartialEq for Stretch {
    fn eq(&self, o: &Self) -> bool {
        self.floor.total_cmp(&o.floor) == Ordering::Equal
    }
}

impl Eq for Stretch {}

impl PartialOrd for Stretch {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Lowest floor first out of the (max-) heap.
impl Ord for Stretch {
    fn cmp(&self, o: &Self) -> Ordering {
        o.floor.total_cmp(&self.floor)
    }
}

/// The least of a convex function on [0, 1] by golden section over `steps` steps, the
/// ends included: (least value, where).
fn least(h: impl Fn(f64) -> f64, steps: usize) -> (f64, f64) {
    const R: f64 = 0.618_033_988_749_894_8;
    let (mut a, mut b) = (0.0, 1.0);
    let (mut c, mut d) = (b - R * (b - a), a + R * (b - a));
    let (mut fc, mut fd) = (h(c), h(d));
    for _ in 0..steps {
        if fc <= fd {
            (b, d, fd) = (d, c, fc);
            c = b - R * (b - a);
            fc = h(c);
        } else {
            (a, c, fc) = (c, d, fd);
            d = a + R * (b - a);
            fd = h(d);
        }
    }
    [(fc, c), (fd, d), (h(0.0), 0.0), (h(1.0), 1.0)].into_iter().fold((f64::INFINITY, 0.0), |m, x| if x.0 < m.0 { x } else { m })
}

impl Sweep {
    pub fn new(tool: Arc<ToolSolid>, motion: Arc<dyn Motion>) -> Sweep {
        let bounds = swept_bounds(&tool, motion.as_ref());
        Sweep { tool, motion, bounds }
    }

    /// A box holding everything the move sweeps.
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    /// The tool's distance at p at time t.
    pub fn at(&self, p: V3, t: f64) -> f64 {
        self.tool.sd(self.motion.pose(t).to_local(p))
    }

    /// The signed distance at p (negative inside), within `tol` above the least.
    pub fn sd(&self, p: V3, tol: f64) -> f64 {
        self.sd_below(p, f64::INFINITY, tol).unwrap_or(f64::INFINITY)
    }

    /// The signed distance at p if it is below `cut` (within `tol` above the least),
    /// else None.
    pub fn sd_below(&self, p: V3, cut: f64, tol: f64) -> Option<f64> {
        self.least_below(p, cut, tol).map(|x| x.0)
    }

    /// The field at p (within `tol` above the least) and its gradient there: the tool's
    /// own gradient at the time the move comes nearest p (where the least over t is
    /// taken, the time's own change does not count).
    pub fn value_grad(&self, p: V3, tol: f64) -> (f64, V3) {
        let t = self.least_below(p, f64::INFINITY, tol).map_or(0.0, |x| x.1);
        let pose = self.motion.pose(t);
        let q = pose.to_local(p);
        let h = 1e-6 * (1.0 + self.tool.reach());
        let d = |e: V3| self.tool.sd(q + e) - self.tool.sd(q - e);
        let g = V3::new(d(V3::new(h, 0.0, 0.0)), d(V3::new(0.0, h, 0.0)), d(V3::new(0.0, 0.0, h)));
        (self.tool.sd(q), pose.rot.apply(scale(g, 0.5 / h)))
    }

    /// As `sd_below`, with the time the least was found at.
    fn least_below(&self, p: V3, cut: f64, tol: f64) -> Option<(f64, f64)> {
        if !may_hold(self.bounds.dist(p), cut) {
            return None;
        }
        // Seed the best with a coarse pass, then split the lowest floor first.
        const SEED: usize = 8;
        let seed: Vec<f64> = (0..=SEED).map(|i| self.at(p, i as f64 / SEED as f64)).collect();
        let (mut best, mut when) = (f64::INFINITY, 0.0);
        for (i, &g) in seed.iter().enumerate() {
            if g < best {
                (best, when) = (g, i as f64 / SEED as f64);
            }
        }
        let mut heap = BinaryHeap::new();
        for i in 0..SEED {
            heap.push(Stretch { floor: f64::NEG_INFINITY, a: i as f64 / SEED as f64, b: (i + 1) as f64 / SEED as f64, ga: seed[i], gb: seed[i + 1] });
        }
        while let Some(s) = heap.pop() {
            if s.floor >= best.min(cut) - tol {
                break;
            }
            let (a, b) = (s.a, s.b);
            let m = 0.5 * (a + b);
            let gm = self.at(p, m);
            if gm < best {
                (best, when) = (gm, m);
            }
            let target = best.min(cut) - tol;
            let deep = if gm <= s.ga.min(s.gb) { m } else if s.ga <= s.gb { a } else { b };
            let hull = self.hull_floor(p, a, b, deep, target, tol);
            if hull.is_some_and(|f| f >= target) {
                continue;
            }
            let (slide, t) = self.floor(p, a, b, tol);
            let gt = self.at(p, t);
            if gt < best {
                (best, when) = (gt, t);
            }
            let floor = slide.max(hull.unwrap_or(f64::NEG_INFINITY));
            if floor < best.min(cut) - tol && b - a > 1e-9 {
                heap.push(Stretch { floor, a, b: m, ga: s.ga, gb: gm });
                heap.push(Stretch { floor, a: m, b, ga: gm, gb: s.gb });
            }
        }
        (best < cut).then_some((best, when))
    }

    /// Over [a, b]: the second floor (see the module notes), if the tool is convex and
    /// the motion bounds its second derivatives. Outside the hull it stops working as
    /// soon as it knows whether the floor reaches `target`; inside, it reads the support
    /// planes across the outward normal at the tool's boundary nearest p at time `deep`.
    fn hull_floor(&self, p: V3, a: f64, b: f64, deep: f64, target: f64, tol: f64) -> Option<f64> {
        if !self.tool.convex() {
            return None;
        }
        let (c2, a2) = self.motion.accel(a, b);
        if !(c2.is_finite() && a2.is_finite()) {
            return None;
        }
        let (_, w) = self.motion.rate(a, b);
        let dt = b - a;
        let bend = dt * dt / 8.0 * (c2 + (a2 + w * w) * self.tool.reach());
        let (pa, pb) = (self.motion.pose(a), self.motion.pose(b));
        let support = |d: V3| {
            let (x, y) = (self.tool.support(pa, d), self.tool.support(pb, d));
            if dot(x, d) >= dot(y, d) {
                x
            } else {
                y
            }
        };
        let floor = hull::distance_floor(&support, p, pa.apply(V3::new(0.0, 0.0, 0.5 * self.tool.length)), target + bend, 0.1 * tol);
        if floor > 0.0 {
            return Some(floor - bend);
        }
        // Inside: the outward normal by central differences of the tool's distance.
        let pose = self.motion.pose(deep);
        let h = 1e-6 * (1.0 + self.tool.reach());
        let d = |e: V3| self.tool.sd(pose.to_local(p + e)) - self.tool.sd(pose.to_local(p - e));
        let n = V3::new(d(V3::new(h, 0.0, 0.0)), d(V3::new(0.0, h, 0.0)), d(V3::new(0.0, 0.0, h)));
        if n.len() < 1e-12 {
            return None;
        }
        let u = scale(n, 1.0 / n.len());
        let reach = dot(self.tool.support(pa, u), u).max(dot(self.tool.support(pb, u), u));
        Some(dot(u, p) - reach - bend)
    }

    /// Over [a, b]: a floor under g, and a time where g is likely least.
    fn floor(&self, p: V3, a: f64, b: f64, tol: f64) -> (f64, f64) {
        let half = 0.5 * (b - a);
        let (v, w) = self.motion.rate(a, b);
        let mid = self.motion.pose(a + half);
        if !self.tool.convex() {
            let lip = v + w * ((p - mid.at).len() + v * half);
            return (self.tool.sd(mid.to_local(p)) - lip * half, a + half);
        }
        let (pa, pb) = (self.motion.pose(a).at, self.motion.pose(b).at);
        let grow = self.motion.stray(a, b) + w * half * self.tool.reach();
        let inv = mid.rot.inverse();
        let (q0, dq) = (inv.apply(p - pa), inv.apply(pb - pa));
        let span = dq.len();
        let steps = if span > 0.01 * tol { ((span / (0.01 * tol)).ln() / 0.481_211_825).ceil() as usize + 1 } else { 0 };
        let (h, s) = least(|s| self.tool.sd(q0 - scale(dq, s)), steps);
        // Golden section brackets the least to 0.618^steps; the slide is span-Lipschitz.
        let slack = span * 0.618_034f64.powi(steps as i32);
        (h - grow - slack, a + s * (b - a))
    }
}

/// Whether a move whose box lies `d` from p can read below `cut` there: outside its box
/// its field is at least `d`, but a box holding p says nothing of how deep p lies.
pub(crate) fn may_hold(d: f64, cut: f64) -> bool {
    d < cut || d == 0.0
}

/// A box holding the tool over the whole move: the hull of the two ends for a straight
/// move, else the tool's box at the middle of each stretch of t, grown by how far any of
/// its points can move within the stretch, the stretches split until that is small.
fn swept_bounds(tool: &ToolSolid, motion: &dyn Motion) -> Aabb {
    if motion.straight() {
        return tool.bounds(motion.pose(0.0)).union(tool.bounds(motion.pose(1.0)));
    }
    let fine = 0.25 * tool.radius() + 0.05;
    let mut out = Aabb::EMPTY;
    let mut stack = vec![(0.0, 1.0)];
    while let Some((a, b)) = stack.pop() {
        let half = 0.5 * (b - a);
        let (v, w) = motion.rate(a, b);
        let margin = (v + w * tool.reach()) * half;
        if margin <= fine || half < 1e-9 {
            out = out.union(tool.bounds(motion.pose(a + half)).grow(margin));
        } else {
            stack.push((a, a + half));
            stack.push((a + half, b));
        }
    }
    out
}
