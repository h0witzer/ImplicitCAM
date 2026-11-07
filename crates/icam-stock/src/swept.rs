//! The in-process stock at one slab: material unless some swept tool disk (a 2D capsule,
//! or an arc band) covers the point. Exact and incremental, and deliberately separate
//! code from the checker's ImplicitStock. A uniform bucket grid only finds candidates;
//! every answer is computed exactly from the sweeps.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use icam_post::{Box2, Move, MoveKind, ToolDef};

/// A swept tool disk at the slab: a segment capsule, or an arc band when `arc_r` > 0.
#[derive(Clone, Copy, Debug)]
struct Sweep {
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    r: f64,
    cx: f64,
    cy: f64,
    arc_r: f64,
    th0: f64,
    dth: f64,
}

/// Uniform bucket grid of item indices over a box.
pub(crate) struct Buckets {
    bx: Box2,
    cell: f64,
    nx: i64,
    ny: i64,
    cells: Vec<Vec<usize>>,
}

impl Buckets {
    pub(crate) fn new(bx: Box2, cell: f64) -> Buckets {
        let nx = 1.max(((bx.x1 - bx.x0) / cell).ceil() as i64);
        let ny = 1.max(((bx.y1 - bx.y0) / cell).ceil() as i64);
        Buckets { bx, cell, nx, ny, cells: vec![Vec::new(); (nx * ny) as usize] }
    }

    pub(crate) fn add(&mut self, item: usize, x0: f64, y0: f64, x1: f64, y1: f64) {
        let i0 = 0.max(((x0 - self.bx.x0) / self.cell) as i64);
        let i1 = (self.nx - 1).min(((x1 - self.bx.x0) / self.cell) as i64);
        let j0 = 0.max(((y0 - self.bx.y0) / self.cell) as i64);
        let j1 = (self.ny - 1).min(((y1 - self.bx.y0) / self.cell) as i64);
        for j in j0..=j1 {
            for i in i0..=i1 {
                self.cells[(j * self.nx + i) as usize].push(item);
            }
        }
    }

    pub(crate) fn at(&self, x: f64, y: f64) -> &[usize] {
        let i = ((x - self.bx.x0) / self.cell) as i64;
        let j = ((y - self.bx.y0) / self.cell) as i64;
        if i < 0 || j < 0 || i >= self.nx || j >= self.ny {
            return &[];
        }
        &self.cells[(j * self.nx + i) as usize]
    }
}

pub struct SweptStock {
    s: Vec<Sweep>,
    grid: Buckets,
    sheet: Box2,
}

pub(crate) fn md(a: f64) -> f64 {
    let a = a % (2.0 * PI);
    if a < 0.0 { a + 2.0 * PI } else { a }
}

fn seg(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (ux, uy) = (bx - ax, by - ay);
    let l2 = ux * ux + uy * uy;
    let t = if l2 < 1e-18 { 0.0 } else { (((x - ax) * ux + (y - ay) * uy) / l2).clamp(0.0, 1.0) };
    let (ex, ey) = (x - (ax + t * ux), y - (ay + t * uy));
    (ex * ex + ey * ey).sqrt()
}

impl SweptStock {
    /// Stock over the sheet; sweeps are filed on a 3 mm grid over `region`.
    pub fn new(region: Box2, sheet: Box2) -> SweptStock {
        SweptStock { s: Vec::new(), grid: Buckets::new(region, 3.0), sheet }
    }

    pub fn count(&self) -> usize {
        self.s.len()
    }

    pub fn add_segment(&mut self, ax: f64, ay: f64, bx: f64, by: f64, r: f64) {
        let id = self.s.len();
        self.s.push(Sweep { ax, ay, bx, by, r, cx: 0.0, cy: 0.0, arc_r: 0.0, th0: 0.0, dth: 0.0 });
        self.grid.add(id, ax.min(bx) - r, ay.min(by) - r, ax.max(bx) + r, ay.max(by) + r);
    }

    /// Arc of radius `arc_r` about (cx, cy) from angle `th0` through `dth` (signed).
    pub fn add_arc(&mut self, cx: f64, cy: f64, arc_r: f64, th0: f64, dth: f64, r: f64) {
        let id = self.s.len();
        self.s.push(Sweep { ax: 0.0, ay: 0.0, bx: 0.0, by: 0.0, r, cx, cy, arc_r, th0, dth });
        let e = arc_r + r;
        self.grid.add(id, cx - e, cy - e, cx + e, cy + e);
    }

    /// Material at the slab: inside the sheet and under no sweep. Newest sweeps are
    /// checked first; they are the likeliest to cover a point near the tool.
    pub fn material(&self, x: f64, y: f64) -> bool {
        if !self.sheet.contains(x, y) {
            return false;
        }
        !self.grid.at(x, y).iter().rev().any(|&k| covers(&self.s[k], x, y))
    }

    /// Material as the stock stood when it held only its first `n` sweeps. The grid files
    /// sweeps in the order they came, so those are a prefix of every cell's list: any point
    /// can be read at any `n`, in any order, without replaying the sweeps.
    pub fn material_at(&self, x: f64, y: f64, n: usize) -> bool {
        if !self.sheet.contains(x, y) {
            return false;
        }
        let c = self.grid.at(x, y);
        !c[..c.partition_point(|&k| k < n)].iter().rev().any(|&k| covers(&self.s[k], x, y))
    }

    /// Every cutting move of the program before `end` whose tool reaches the slab at `z`.
    pub fn add_program(&mut self, moves: &[Move], end: usize, tools: &BTreeMap<i32, ToolDef>, z: f64) {
        const EPS: f64 = 0.05;
        for m in &moves[..end.min(moves.len())] {
            if m.kind == MoveKind::Rapid {
                continue;
            }
            let Some(tool) = tools.get(&m.tool) else { continue };
            let r = tool.diameter / 2.0;
            match m.kind {
                MoveKind::Drill => {
                    if m.to.z <= z + EPS {
                        self.add_segment(m.to.x, m.to.y, m.to.x, m.to.y, r);
                    }
                }
                MoveKind::Feed => {
                    let (mut a, mut b) = (m.from, m.to);
                    if a.z.min(b.z) > z + EPS {
                        continue;
                    }
                    if a.z.max(b.z) > z + EPS {
                        // A ramp through the slab: keep the part of it below the slab top.
                        let s = (z + EPS - a.z) / (b.z - a.z);
                        let p = icam_post::V3::new(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s, a.z + (b.z - a.z) * s);
                        if a.z > b.z {
                            a = p;
                        } else {
                            b = p;
                        }
                    }
                    self.add_segment(a.x, a.y, b.x, b.y, r);
                }
                _ => {
                    if m.from.z.min(m.to.z) > z + EPS {
                        continue;
                    }
                    let (cx, cy) = (m.center.x, m.center.y);
                    let arc_r = ((m.from.x - cx) * (m.from.x - cx) + (m.from.y - cy) * (m.from.y - cy)).sqrt();
                    let (t0, t1) = ((m.from.y - cy).atan2(m.from.x - cx), (m.to.y - cy).atan2(m.to.x - cx));
                    let full = (m.to - m.from).len() < 1e-6;
                    let d = if m.kind == MoveKind::ArcCcw {
                        if full { 2.0 * PI } else { md(t1 - t0) }
                    } else if full {
                        -2.0 * PI
                    } else {
                        -md(t0 - t1)
                    };
                    self.add_arc(cx, cy, arc_r, t0, d, r);
                }
            }
        }
    }
}

fn covers(s: &Sweep, x: f64, y: f64) -> bool {
    if s.arc_r <= 0.0 {
        return seg(x, y, s.ax, s.ay, s.bx, s.by) <= s.r;
    }
    let (dx, dy) = (x - s.cx, y - s.cy);
    let rho = (dx * dx + dy * dy).sqrt();
    let th = dy.atan2(dx) - s.th0;
    // Angle of the point measured along the arc's direction, in [0, 2pi).
    let a = if s.dth >= 0.0 { md(th) } else { md(-th) };
    if a <= s.dth.abs() {
        return (rho - s.arc_r).abs() <= s.r;
    }
    // Beyond the arc's span: distance to its end points.
    let (e0x, e0y) = (s.cx + s.arc_r * s.th0.cos(), s.cy + s.arc_r * s.th0.sin());
    let (e1x, e1y) = (s.cx + s.arc_r * (s.th0 + s.dth).cos(), s.cy + s.arc_r * (s.th0 + s.dth).sin());
    (x - e0x) * (x - e0x) + (y - e0y) * (y - e0y) <= s.r * s.r || (x - e1x) * (x - e1x) + (y - e1y) * (y - e1y) <= s.r * s.r
}
