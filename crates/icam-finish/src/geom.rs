//! The small 2D tools the finishing study needs beside the clearance: a nodal grid with
//! bilinear lookup, marching squares with chaining, Douglas-Peucker, a segment index for
//! "is any capsule within r" queries, and the stock top a program leaves at a point.
//! Units: mm.

use std::collections::HashMap;
use std::f64::consts::PI;

use icam_post::{Box2, Move, MoveKind, ToolDef};
use std::collections::BTreeMap;

/// A field sampled on the nodes of a uniform grid; node (i, j) is (x0 + i h, y0 + j h).
#[derive(Clone)]
pub struct Grid2 {
    pub x0: f64,
    pub y0: f64,
    pub h: f64,
    pub nx: usize,
    pub ny: usize,
    pub v: Vec<f64>,
}

impl Grid2 {
    pub fn new(x0: f64, y0: f64, h: f64, nx: usize, ny: usize) -> Grid2 {
        Grid2 { x0, y0, h, nx, ny, v: vec![0.0; nx * ny] }
    }

    pub fn over(b: Box2, h: f64) -> Grid2 {
        Grid2::new(b.x0, b.y0, h, ((b.x1 - b.x0) / h).ceil() as usize + 1, ((b.y1 - b.y0) / h).ceil() as usize + 1)
    }

    /// The same nodes, all zero.
    pub fn like(&self) -> Grid2 {
        Grid2::new(self.x0, self.y0, self.h, self.nx, self.ny)
    }

    pub fn x(&self, i: usize) -> f64 {
        self.x0 + i as f64 * self.h
    }

    pub fn y(&self, j: usize) -> f64 {
        self.y0 + j as f64 * self.h
    }

    pub fn nearest(&self, x: f64, y: f64) -> Option<usize> {
        let i = ((x - self.x0) / self.h).round() as i64;
        let j = ((y - self.y0) / self.h).round() as i64;
        if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.ny as i64 { None } else { Some(j as usize * self.nx + i as usize) }
    }

    /// Bilinear value, extrapolated from the edge cells outside the grid.
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let (fx, fy) = ((x - self.x0) / self.h, (y - self.y0) / self.h);
        let i = (fx.floor() as i64).clamp(0, self.nx as i64 - 2);
        let j = (fy.floor() as i64).clamp(0, self.ny as i64 - 2);
        let u = (fx - i as f64).clamp(0.0, 1.0);
        let v = (fy - j as f64).clamp(0.0, 1.0);
        let s = j as usize * self.nx + i as usize;
        let n = self.nx;
        (1.0 - u) * (1.0 - v) * self.v[s] + u * (1.0 - v) * self.v[s + 1] + (1.0 - u) * v * self.v[s + n] + u * v * self.v[s + n + 1]
    }

    /// Unit gradient by central differences over one cell.
    pub fn grad(&self, x: f64, y: f64) -> (f64, f64) {
        let e = 0.5 * self.h;
        let gx = self.at(x + e, y) - self.at(x - e, y);
        let gy = self.at(x, y + e) - self.at(x, y - e);
        let n = (gx * gx + gy * gy).sqrt();
        if n < 1e-12 { (0.0, 0.0) } else { (gx / n, gy / n) }
    }
}

fn key(p: (f64, f64)) -> (i64, i64) {
    ((p.0 * 1e6).round() as i64, (p.1 * 1e6).round() as i64)
}

/// Level set of a nodal field by marching squares, chained into point chains: each with
/// whether it closes (a closed chain does not repeat its first point).
pub fn contour(g: &Grid2, level: f64) -> Vec<(Vec<(f64, f64)>, bool)> {
    let f = |i: usize, j: usize| g.v[j * g.nx + i];
    let cut = |i0: usize, j0: usize, i1: usize, j1: usize| {
        let (f0, f1) = (f(i0, j0), f(i1, j1));
        let s = (level - f0) / (f1 - f0);
        (g.x(i0) + (g.x(i1) - g.x(i0)) * s, g.y(j0) + (g.y(j1) - g.y(j0)) * s)
    };
    let mut segs: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for j in 0..g.ny.saturating_sub(1) {
        for i in 0..g.nx.saturating_sub(1) {
            let code = (f(i, j) > level) as u8 | ((f(i + 1, j) > level) as u8) << 1 | ((f(i + 1, j + 1) > level) as u8) << 2 | ((f(i, j + 1) > level) as u8) << 3;
            if code == 0 || code == 15 {
                continue;
            }
            let (e0, e1, e2, e3) = (cut(i, j, i + 1, j), cut(i + 1, j, i + 1, j + 1), cut(i, j + 1, i + 1, j + 1), cut(i, j, i, j + 1));
            let center = 0.25 * (f(i, j) + f(i + 1, j) + f(i + 1, j + 1) + f(i, j + 1));
            match code {
                1 | 14 => segs.push((e3, e0)),
                2 | 13 => segs.push((e0, e1)),
                3 | 12 => segs.push((e3, e1)),
                4 | 11 => segs.push((e1, e2)),
                6 | 9 => segs.push((e0, e2)),
                7 | 8 => segs.push((e3, e2)),
                5 => {
                    if center > level {
                        segs.push((e3, e2));
                        segs.push((e0, e1));
                    } else {
                        segs.push((e3, e0));
                        segs.push((e1, e2));
                    }
                }
                10 => {
                    if center > level {
                        segs.push((e3, e0));
                        segs.push((e1, e2));
                    } else {
                        segs.push((e3, e2));
                        segs.push((e0, e1));
                    }
                }
                _ => {}
            }
        }
    }
    let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (k, s) in segs.iter().enumerate() {
        for p in [s.0, s.1] {
            adj.entry(key(p)).or_default().push(k);
        }
    }
    let deg = |p: (f64, f64)| adj[&key(p)].len();
    let mut used = vec![false; segs.len()];
    let mut out = Vec::new();
    // Open ends first (degree-1 points), so open chains are walked whole.
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by_key(|&k| if deg(segs[k].0) == 1 || deg(segs[k].1) == 1 { 0 } else { 1 });
    for k0 in order {
        if used[k0] {
            continue;
        }
        used[k0] = true;
        let (mut a, mut b) = segs[k0];
        if deg(b) == 1 && deg(a) != 1 {
            std::mem::swap(&mut a, &mut b);
        }
        let mut chain = vec![a, b];
        let mut end = b;
        loop {
            let Some(&next) = adj[&key(end)].iter().find(|&&c| !used[c]) else { break };
            used[next] = true;
            end = if key(segs[next].0) == key(end) { segs[next].1 } else { segs[next].0 };
            chain.push(end);
        }
        let closed = chain.len() > 2 && key(chain[0]) == key(chain[chain.len() - 1]);
        if closed {
            chain.pop();
        }
        out.push((chain, closed));
    }
    out
}

/// Douglas-Peucker on an open chain (keeps both ends).
pub fn simplify(p: &[(f64, f64)], tol: f64) -> Vec<(f64, f64)> {
    if p.len() < 3 {
        return p.to_vec();
    }
    let mut keep = vec![false; p.len()];
    keep[0] = true;
    keep[p.len() - 1] = true;
    let mut stack = vec![(0usize, p.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        let (mut best, mut idx) = (-1.0, None);
        for k in a + 1..b {
            let d = icam_query::seg(p[k].0, p[k].1, p[a].0, p[a].1, p[b].0, p[b].1);
            if d > best {
                best = d;
                idx = Some(k);
            }
        }
        if let Some(i) = idx {
            if best > tol {
                keep[i] = true;
                stack.push((a, i));
                stack.push((i, b));
            }
        }
    }
    p.iter().zip(&keep).filter(|(_, &k)| k).map(|(&q, _)| q).collect()
}

/// Segments (capsule axes) filed in a bucket grid: answers "is any capsule of radius r
/// within reach of (x, y)", optionally only among capsules with id below a limit.
pub struct SegIndex {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: i64,
    ny: i64,
    cells: Vec<Vec<u32>>,
    ax: Vec<f64>,
    ay: Vec<f64>,
    bx: Vec<f64>,
    by: Vec<f64>,
    id: Vec<i32>,
}

fn d2(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (ux, uy) = (bx - ax, by - ay);
    let l2 = ux * ux + uy * uy;
    let t = if l2 < 1e-18 { 0.0 } else { (((x - ax) * ux + (y - ay) * uy) / l2).clamp(0.0, 1.0) };
    let (ex, ey) = (x - (ax + t * ux), y - (ay + t * uy));
    ex * ex + ey * ey
}

impl SegIndex {
    /// Segments (ax, ay, bx, by, id); queries reach at most `reach` from a segment.
    pub fn new(segs: &[(f64, f64, f64, f64, i32)], reach: f64, cell: f64) -> SegIndex {
        let n = segs.len();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for s in segs {
            x0 = x0.min(s.0.min(s.2));
            x1 = x1.max(s.0.max(s.2));
            y0 = y0.min(s.1.min(s.3));
            y1 = y1.max(s.1.max(s.3));
        }
        if n == 0 {
            (x0, y0, x1, y1) = (0.0, 0.0, 1.0, 1.0);
        }
        let (gx0, gy0) = (x0 - reach - cell, y0 - reach - cell);
        let nx = ((x1 + reach + cell - gx0) / cell).ceil() as i64 + 1;
        let ny = ((y1 + reach + cell - gy0) / cell).ceil() as i64 + 1;
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); (nx * ny) as usize];
        for (k, s) in segs.iter().enumerate() {
            let i0 = ((s.0.min(s.2) - reach - gx0) / cell) as i64;
            let i1 = ((s.0.max(s.2) + reach - gx0) / cell) as i64;
            let j0 = ((s.1.min(s.3) - reach - gy0) / cell) as i64;
            let j1 = ((s.1.max(s.3) + reach - gy0) / cell) as i64;
            for j in j0.max(0)..=j1.min(ny - 1) {
                for i in i0.max(0)..=i1.min(nx - 1) {
                    cells[(j * nx + i) as usize].push(k as u32);
                }
            }
        }
        SegIndex {
            x0: gx0,
            y0: gy0,
            cell,
            nx,
            ny,
            cells,
            ax: segs.iter().map(|s| s.0).collect(),
            ay: segs.iter().map(|s| s.1).collect(),
            bx: segs.iter().map(|s| s.2).collect(),
            by: segs.iter().map(|s| s.3).collect(),
            id: segs.iter().map(|s| s.4).collect(),
        }
    }

    pub fn count(&self) -> usize {
        self.ax.len()
    }

    fn cell_of(&self, x: f64, y: f64) -> &[u32] {
        let i = ((x - self.x0) / self.cell) as i64;
        let j = ((y - self.y0) / self.cell) as i64;
        if i < 0 || j < 0 || i >= self.nx || j >= self.ny {
            return &[];
        }
        &self.cells[(j * self.nx + i) as usize]
    }

    fn d2(&self, k: usize, x: f64, y: f64) -> f64 {
        d2(x, y, self.ax[k], self.ay[k], self.bx[k], self.by[k])
    }

    pub fn within(&self, x: f64, y: f64, r: f64) -> bool {
        let r2 = r * r;
        self.cell_of(x, y).iter().any(|&k| self.d2(k as usize, x, y) <= r2)
    }

    pub fn for_each_within(&self, x: f64, y: f64, r: f64, mut f: impl FnMut(i32)) {
        let r2 = r * r;
        for &k in self.cell_of(x, y) {
            if self.d2(k as usize, x, y) <= r2 {
                f(self.id[k as usize]);
            }
        }
    }

    /// Id of the nearest segment within r (ties: the later one), if any.
    pub fn nearest_id(&self, x: f64, y: f64, r: f64) -> Option<i32> {
        let (mut best, mut id) = (r * r, None);
        for &k in self.cell_of(x, y) {
            let d = self.d2(k as usize, x, y);
            if d <= best {
                best = d;
                id = Some(self.id[k as usize]);
            }
        }
        id
    }

    /// Among segments with id below `limit`, newest first (the capsules just behind the
    /// tool are the likeliest hits).
    pub fn within_before(&self, x: f64, y: f64, r: f64, limit: i32) -> bool {
        let r2 = r * r;
        self.cell_of(x, y).iter().rev().any(|&k| self.id[k as usize] < limit && self.d2(k as usize, x, y) <= r2)
    }
}

fn md(a: f64) -> f64 {
    let a = a % (2.0 * PI);
    if a < 0.0 { a + 2.0 * PI } else { a }
}

/// The stock top at (x, y) left by the cutting moves before `end`: the lowest height any
/// of their tool disks reaches over the point (`top` where none does).
pub fn top_before(moves: &[Move], tools: &BTreeMap<i32, ToolDef>, end: usize, x: f64, y: f64, mut top: f64) -> f64 {
    for m in &moves[..end.min(moves.len())] {
        if m.kind == MoveKind::Rapid {
            continue;
        }
        let Some(t) = tools.get(&m.tool) else { continue };
        let r = t.diameter / 2.0;
        let z = match m.kind {
            MoveKind::Drill => {
                if (x - m.to.x) * (x - m.to.x) + (y - m.to.y) * (y - m.to.y) > r * r {
                    continue;
                }
                m.to.z
            }
            MoveKind::Feed => {
                let (dx, dy) = (m.to.x - m.from.x, m.to.y - m.from.y);
                let l2 = dx * dx + dy * dy;
                let u = if l2 < 1e-12 { 0.0 } else { (((x - m.from.x) * dx + (y - m.from.y) * dy) / l2).clamp(0.0, 1.0) };
                let (qx, qy) = (m.from.x + u * dx - x, m.from.y + u * dy - y);
                if qx * qx + qy * qy > r * r {
                    continue;
                }
                m.from.z + u * (m.to.z - m.from.z)
            }
            _ => {
                let c = m.center;
                let ra = ((m.from.x - c.x) * (m.from.x - c.x) + (m.from.y - c.y) * (m.from.y - c.y)).sqrt();
                let rp = ((x - c.x) * (x - c.x) + (y - c.y) * (y - c.y)).sqrt();
                if (rp - ra).abs() > r {
                    continue;
                }
                let t0 = (m.from.y - c.y).atan2(m.from.x - c.x);
                let t1 = (m.to.y - c.y).atan2(m.to.x - c.x);
                let tp = (y - c.y).atan2(x - c.x);
                let full = full_turn(m);
                let within = full || if m.kind == MoveKind::ArcCcw { md(tp - t0) <= md(t1 - t0) } else { md(t0 - tp) <= md(t0 - t1) };
                let near_end = (x - m.from.x) * (x - m.from.x) + (y - m.from.y) * (y - m.from.y) <= r * r
                    || (x - m.to.x) * (x - m.to.x) + (y - m.to.y) * (y - m.to.y) <= r * r;
                if !within && !near_end {
                    continue;
                }
                m.from.z.min(m.to.z)
            }
        };
        top = top.min(z);
    }
    top
}

/// An arc whose end is its start is a full turn.
pub(crate) fn full_turn(m: &Move) -> bool {
    let (dx, dy, dz) = (m.to.x - m.from.x, m.to.y - m.from.y, m.to.z - m.from.z);
    (dx * dx + dy * dy + dz * dz).sqrt() < 1e-6
}
