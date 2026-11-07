//! The stock's top after a program, at any point: the lowest the bottom of an earlier
//! cutting move's tool reaches there, else the stock's top. For tools along Z (3-axis
//! programs) and every tool shape the tool sheets carry (flat, bull and ball end mills,
//! pointed drills), on lines, ramps, plunges, drill cycles, arcs and helices:
//!
//!   flat tools   closed form: the tip's height is linear along a move, so its least over
//!                the stretch whose axis reaches the point is where the point enters or
//!                leaves the tool's reach, or at an end of the move
//!   round tools  the tip's height plus the profile's lift at the point's distance from
//!                the axis. On a line that sum is convex in the move's parameter (the
//!                distance is convex, the lift convex and rising), so a golden-section
//!                search finds its least; on an arc it need not be, so a branch and bound
//!                does, each stretch bounded below by its lowest tip plus the lift at its
//!                nearest pass, to `StockTop::TOL`
//!
//! A bucket grid only finds candidates, and a move is skipped whole when its own lower
//! bound cannot beat the lowest reach found so far. Rapids cut nothing, as in SweptStock.
//! Separate code from `ImplicitStock`, the 3-axis checker, as the generator's own stock was
//! required to be; both retire with the prismatic code (roadmap milestones 0 and 1), and
//! `icam-sweep` is then the one stock, for planning and for checking.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::f64::consts::PI;

use icam_post::{Box2, Move, MoveKind, ToolDef, V3};

use crate::swept::{md, Buckets};

/// The bottom of a tool turning about Z: flat out to `r - rc`, a corner of radius `rc`
/// beyond that, or a point cone (`cot` = 1 / tan of half the tip angle) on a drill.
#[derive(Clone, Copy, Debug)]
struct Profile {
    r: f64,
    rc: f64,
    cot: f64,
}

impl Profile {
    fn of(t: &ToolDef) -> Profile {
        let r = t.diameter / 2.0;
        if t.kind == "DRILL" {
            return Profile { r, rc: 0.0, cot: 1.0 / (t.tip_angle_deg / 2.0 * PI / 180.0).tan() };
        }
        Profile { r, rc: t.corner_radius.clamp(0.0, r), cot: 0.0 }
    }

    fn flat(&self) -> bool {
        self.rc <= 0.0 && self.cot <= 0.0
    }

    /// Height of the bottom above the tip at `rho` from the axis (`rho` <= r).
    fn lift(&self, rho: f64) -> f64 {
        let u = rho - (self.r - self.rc);
        let corner = if u > 0.0 { self.rc - (self.rc * self.rc - u * u).max(0.0).sqrt() } else { 0.0 };
        corner + self.cot * rho
    }
}

/// A cutting move's tip: a line (a point for a drill cycle), or an arc of radius `ra`
/// about (`cx`, `cy`) from angle `th0` through `sweep` (signed), its height linear in the
/// angle.
#[derive(Clone, Copy, Debug)]
enum Track {
    Line { a: V3, b: V3 },
    Arc { cx: f64, cy: f64, ra: f64, th0: f64, sweep: f64, z0: f64, z1: f64 },
}

struct Cut {
    track: Track,
    tool: Profile,
}

pub struct StockTop {
    cuts: Vec<Cut>,
    grid: Buckets,
    top: f64,
}

impl StockTop {
    /// How far above the true least an arc's branch and bound may stop, mm.
    pub const TOL: f64 = 1e-7;

    /// The top that moves `..end` leave on stock whose top is `top`, read inside `region`
    /// (its edges included: the grid is filed over the region grown by 1 mm, since a
    /// point on a grid's far edge falls past its last cell).
    pub fn new(moves: &[Move], end: usize, tools: &BTreeMap<i32, ToolDef>, top: f64, region: Box2) -> StockTop {
        let mut st = StockTop { cuts: Vec::new(), grid: Buckets::new(region.grow(1.0), 3.0), top };
        for m in &moves[..end.min(moves.len())] {
            if m.kind == MoveKind::Rapid || m.from.z.min(m.to.z) >= top {
                continue;
            }
            let Some(t) = tools.get(&m.tool) else { continue };
            let tool = Profile::of(t);
            let (track, (x0, y0, x1, y1)) = match m.kind {
                MoveKind::Drill => (Track::Line { a: m.to, b: m.to }, (m.to.x, m.to.y, m.to.x, m.to.y)),
                MoveKind::Feed => (Track::Line { a: m.from, b: m.to }, (m.from.x.min(m.to.x), m.from.y.min(m.to.y), m.from.x.max(m.to.x), m.from.y.max(m.to.y))),
                _ => {
                    let c = m.center;
                    let ra = (m.from.x - c.x).hypot(m.from.y - c.y);
                    let (t0, t1) = ((m.from.y - c.y).atan2(m.from.x - c.x), (m.to.y - c.y).atan2(m.to.x - c.x));
                    let full = (m.to - m.from).len() < 1e-6;
                    let sweep = if m.kind == MoveKind::ArcCcw {
                        if full { 2.0 * PI } else { md(t1 - t0) }
                    } else {
                        -(if full { 2.0 * PI } else { md(t0 - t1) })
                    };
                    (Track::Arc { cx: c.x, cy: c.y, ra, th0: t0, sweep, z0: m.from.z, z1: m.to.z }, (c.x - ra, c.y - ra, c.x + ra, c.y + ra))
                }
            };
            st.grid.add(st.cuts.len(), x0 - tool.r, y0 - tool.r, x1 + tool.r, y1 + tool.r);
            st.cuts.push(Cut { track, tool });
        }
        st
    }

    /// The stock's top at (`x`, `y`).
    pub fn at(&self, x: f64, y: f64) -> f64 {
        self.grid.at(x, y).iter().fold(self.top, |best, &k| {
            let c = &self.cuts[k];
            match c.track {
                Track::Line { a, b } => line(x, y, a, b, &c.tool, best),
                Track::Arc { cx, cy, ra, th0, sweep, z0, z1 } => ArcAt { cx, cy, ra, th0, sweep, z0, z1, t: c.tool, x, y }.lowest(best),
            }
        })
    }
}

/// The lowest the tool reaches over (`x`, `y`) moving from `a` to `b`, if below `best`.
fn line(x: f64, y: f64, a: V3, b: V3, t: &Profile, best: f64) -> f64 {
    let (dx, dy, dz) = (b.x - a.x, b.y - a.y, b.z - a.z);
    let (wx, wy) = (x - a.x, y - a.y);
    let l2 = dx * dx + dy * dy;
    // The stretch [lo, hi] of the move whose axis is within r of the point.
    let (lo, hi, nearest) = if l2 < 1e-18 {
        if wx * wx + wy * wy > t.r * t.r {
            return best;
        }
        (0.0, 1.0, 0.0)
    } else {
        let wd = wx * dx + wy * dy;
        let disc = wd * wd - l2 * (wx * wx + wy * wy - t.r * t.r);
        if disc < 0.0 {
            return best;
        }
        let s = disc.sqrt();
        let (lo, hi) = (((wd - s) / l2).max(0.0), ((wd + s) / l2).min(1.0));
        if lo > hi {
            return best;
        }
        (lo, hi, (wd / l2).clamp(lo, hi))
    };
    let z = |u: f64| a.z + u * dz;
    if t.flat() {
        return best.min(z(lo)).min(z(hi));
    }
    let rho = |u: f64| ((wx - u * dx) * (wx - u * dx) + (wy - u * dy) * (wy - u * dy)).sqrt().min(t.r);
    if z(lo).min(z(hi)) + t.lift(rho(nearest)) >= best {
        return best;
    }
    let f = |u: f64| z(u) + t.lift(rho(u));
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut p, mut q) = (lo, hi);
    let (mut c, mut d) = (q - g * (q - p), p + g * (q - p));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..90 {
        if q - p <= 1e-15 {
            break;
        }
        if fc <= fd {
            (q, d, fd) = (d, c, fc);
            c = q - g * (q - p);
            fc = f(c);
        } else {
            (p, c, fc) = (c, d, fd);
            d = p + g * (q - p);
            fd = f(d);
        }
    }
    best.min(f(lo)).min(f(hi)).min(fc).min(fd)
}

/// An arc move read at one point (`x`, `y`); `s` in [0, 1] runs along the arc.
struct ArcAt {
    cx: f64,
    cy: f64,
    ra: f64,
    th0: f64,
    sweep: f64,
    z0: f64,
    z1: f64,
    t: Profile,
    x: f64,
    y: f64,
}

/// A stretch [sa, sb] of an arc and its lower bound; the heap pops the lowest bound first.
struct Stretch {
    lb: f64,
    sa: f64,
    sb: f64,
}

impl PartialEq for Stretch {
    fn eq(&self, o: &Self) -> bool {
        self.lb.total_cmp(&o.lb) == Ordering::Equal
    }
}
impl Eq for Stretch {}
impl PartialOrd for Stretch {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Stretch {
    fn cmp(&self, o: &Self) -> Ordering {
        o.lb.total_cmp(&self.lb)
    }
}

impl ArcAt {
    fn z(&self, s: f64) -> f64 {
        self.z0 + s * (self.z1 - self.z0)
    }

    fn rho(&self, s: f64) -> f64 {
        let a = self.th0 + s * self.sweep;
        (self.x - self.cx - self.ra * a.cos()).hypot(self.y - self.cy - self.ra * a.sin())
    }

    /// Angle from the arc's point at `s` to the angle `b`, measured along the arc.
    fn ahead(&self, s: f64, b: f64) -> f64 {
        let a = self.th0 + s * self.sweep;
        if self.sweep >= 0.0 { md(b - a) } else { md(a - b) }
    }

    /// The least distance from the point to the stretch [sa, sb]: the distance to the
    /// circle where the stretch passes the point's own angle, else the nearer end (the
    /// distance only grows away from that angle).
    fn nearest(&self, sa: f64, sb: f64) -> f64 {
        let (dx, dy) = (self.x - self.cx, self.y - self.cy);
        if self.ahead(sa, dy.atan2(dx)) <= (sb - sa) * self.sweep.abs() {
            return (dx.hypot(dy) - self.ra).abs();
        }
        self.rho(sa).min(self.rho(sb))
    }

    fn bound(&self, sa: f64, sb: f64) -> f64 {
        let n = self.nearest(sa, sb);
        if n > self.t.r { f64::INFINITY } else { self.z(sa).min(self.z(sb)) + self.t.lift(n) }
    }

    fn at(&self, s: f64) -> f64 {
        let r = self.rho(s);
        if r > self.t.r { f64::INFINITY } else { self.z(s) + self.t.lift(r) }
    }

    /// The lowest the tool reaches over the point along the arc, if below `best`.
    fn lowest(&self, best: f64) -> f64 {
        if self.bound(0.0, 1.0) >= best {
            return best;
        }
        if self.t.flat() {
            return best.min(self.flat_least());
        }
        let mut ub = best.min(self.at(0.0)).min(self.at(1.0));
        let mut heap = BinaryHeap::new();
        heap.push(Stretch { lb: self.bound(0.0, 1.0), sa: 0.0, sb: 1.0 });
        while let Some(Stretch { lb, sa, sb }) = heap.pop() {
            if lb >= ub - StockTop::TOL {
                break;
            }
            if sb - sa < 1e-15 {
                continue;
            }
            let m = 0.5 * (sa + sb);
            ub = ub.min(self.at(m));
            for (a, b) in [(sa, m), (m, sb)] {
                let lb = self.bound(a, b);
                if lb < ub - StockTop::TOL {
                    heap.push(Stretch { lb, sa: a, sb: b });
                }
            }
        }
        ub
    }

    /// Flat tool: the least tip height over the stretches within reach, at their ends.
    fn flat_least(&self) -> f64 {
        let (dx, dy) = (self.x - self.cx, self.y - self.cy);
        let q = dx.hypot(dy);
        let r = self.t.r;
        let mut low = f64::INFINITY;
        let mut take = |s: f64| low = low.min(self.z(s));
        if q * self.ra < 1e-12 {
            // The point on the axis of the arc: every point of the arc is equally far.
            if q.max(self.ra) <= r {
                take(0.0);
                take(1.0);
            }
            return low;
        }
        let k = (q * q + self.ra * self.ra - r * r) / (2.0 * q * self.ra);
        if k > 1.0 {
            return low;
        }
        // The reach is the arc's angles within acos(k) of the point's own angle.
        let alpha = k.max(-1.0).acos();
        for s in [0.0, 1.0] {
            if self.rho(s) <= r {
                take(s);
            }
        }
        let th = dy.atan2(dx);
        for b in [th - alpha, th + alpha] {
            let s = self.ahead(0.0, b) / self.sweep.abs();
            if s <= 1.0 {
                take(s);
            }
        }
        low
    }
}
