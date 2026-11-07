//! One rough-then-finish pocket, set up once and reused by every stock and finisher
//! simulation:
//!
//!   cg       the exact XY clearance to the part's shadow above the floor, sampled on a
//!            0.1 mm grid for bilinear lookup.
//!   loops    the finisher's own path: the clearance = r_finish level set, one closed
//!            loop per wall it finishes (outer wall and islands), projected onto the
//!            exact level set, resampled every ds mm, oriented for climb milling (wall
//!            on the right).
//!   samples  per point: tangent, wall normal, wall foot, curvature radius.
//!   comp     the grid nodes of the rougher's reachable center region, seeded by the
//!            rough op's tool centers (other pockets and the sheet outside the part are
//!            not roughed by this op).
//!   owner    for nodes where the rougher's wall pass can lie, the finisher sample whose
//!            wall foot is nearest: the allowance field delta(x), defined on the samples,
//!            is extended off the wall through it.
//!   prior    the program's cuts before the rough op that reach the floor.

use std::f64::consts::PI;

use icam_post::{Box2, ToolDef};
use icam_query::Clearance;
use icam_stock::SweptStock;
use rayon::prelude::*;

use crate::geom::{contour, Grid2, SegIndex};

/// One closed wall loop of the finisher: its samples are start..start + count.
#[derive(Clone, Copy, Debug)]
pub struct Loop {
    pub start: usize,
    pub count: usize,
    pub length: f64,
    /// The wall is inside the loop (it runs clockwise).
    pub island: bool,
}

/// Which clearance = r_finish loops the finisher runs.
pub enum Walls<'a> {
    /// A program's finishing path, sampled: a loop is run when a quarter of its length
    /// has path points within 1.5 mm (the baseline program's contour op).
    Path(&'a [(f64, f64)]),
    /// Every loop around the region the finisher's center reaches from these points
    /// (the pocket found from the field).
    Reach(&'a [(f64, f64)]),
}

pub struct Pocket {
    pub name: String,
    pub finish_name: String,
    pub rough_tool: ToolDef,
    pub finish_tool: ToolDef,
    pub rr: f64,
    pub rf: f64,
    pub z_cut: f64,
    pub ap: f64,
    pub band: f64,
    pub ds: f64,
    /// Allowance bounds the stock model must cover, mm.
    pub d_min: f64,
    pub d_max: f64,
    pub region: Box2,
    pub cg: Grid2,
    pub prior: SweptStock,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub tx: Vec<f64>,
    pub ty: Vec<f64>,
    pub nx: Vec<f64>,
    pub ny: Vec<f64>,
    pub wx: Vec<f64>,
    pub wy: Vec<f64>,
    pub s: Vec<f64>,
    pub rk: Vec<f64>,
    pub dw: Vec<f64>,
    pub loop_of: Vec<usize>,
    pub loops: Vec<Loop>,
    pub comp: Vec<bool>,
    pub owner: Vec<i32>,
    /// The finisher's own sweep: capsule j is the move from sample j - 1 to sample j.
    pub fin_sweep: SegIndex,
    /// The part's wall at the floor (clearance 0), for pictures.
    pub outline: Vec<Vec<(f64, f64)>>,
    pub notes: Vec<String>,
}

/// The inputs of a pocket besides the clearance and the walls to run.
pub struct PocketInput<'a> {
    pub rough: &'a str,
    pub finish: &'a str,
    pub rough_tool: &'a ToolDef,
    pub finish_tool: &'a ToolDef,
    /// The floor.
    pub z: f64,
    pub region: Box2,
    pub prior: SweptStock,
    /// The rough op's tool centers (they seed its component).
    pub rough_centers: &'a [(f64, f64)],
    /// The stock top the cuts before the rough op leave at a point.
    pub top_before: &'a (dyn Fn(f64, f64) -> f64 + Sync),
    /// Wall sample spacing, mm.
    pub ds: f64,
    /// Clearance grid pitch, mm (see `grid_pitch`).
    pub grid: f64,
    pub d_min: f64,
    pub d_max: f64,
}

/// The clearance grid pitch for a region: 0.1 mm, coarser where that would pass
/// `max_nodes` (a large pocket keeps its whole wall instead of being left out).
pub fn grid_pitch(region: Box2, max_nodes: f64) -> f64 {
    let area = (region.x1 - region.x0) * (region.y1 - region.y0);
    0.1f64.max((area / max_nodes).sqrt())
}

/// The clearance band the pocket needs: the finisher's diameter, or the rougher's center
/// at the thickest allowance, plus 2 mm.
pub fn band_for(rr: f64, rf: f64, d_max: f64) -> f64 {
    (2.0 * rf).max(rr + d_max) + 2.0
}

fn sq(v: f64) -> f64 {
    v * v
}

fn wrap(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Move a point onto the exact level set clear = level (Newton on a distance field).
fn project(clear: &dyn Clearance, mut x: f64, mut y: f64, level: f64) -> (f64, f64) {
    for _ in 0..6 {
        let g = clear.at(x, y) - level;
        if g.abs() < 1e-5 {
            break;
        }
        let (gx, gy) = clear.grad(x, y);
        if gx == 0.0 && gy == 0.0 {
            break;
        }
        x -= g * gx;
        y -= g * gy;
    }
    (x, y)
}

fn closed_len(c: &[(f64, f64)]) -> f64 {
    let n = c.len();
    (0..n).map(|k| (sq(c[(k + 1) % n].0 - c[k].0) + sq(c[(k + 1) % n].1 - c[k].1)).sqrt()).sum()
}

impl Pocket {
    pub fn make(inp: PocketInput, clear: &dyn Clearance, walls: Walls) -> Result<Pocket, String> {
        let (rr, rf) = (inp.rough_tool.diameter / 2.0, inp.finish_tool.diameter / 2.0);
        let band = band_for(rr, rf, inp.d_max);
        let mut cg = Grid2::over(inp.region, inp.grid);
        let (x0, y0, h, nx) = (cg.x0, cg.y0, cg.h, cg.nx);
        cg.v.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = clear.at(x0 + i as f64 * h, y0 + j as f64 * h);
            }
        });
        let outline = contour(&cg, 1e-3).into_iter().filter(|(p, c)| *c && p.len() > 20).map(|(p, _)| p).collect();
        let mut pk = Pocket {
            name: inp.rough.into(),
            finish_name: inp.finish.into(),
            rough_tool: inp.rough_tool.clone(),
            finish_tool: inp.finish_tool.clone(),
            rr,
            rf,
            z_cut: inp.z,
            ap: 0.0,
            band,
            ds: inp.ds,
            d_min: inp.d_min,
            d_max: inp.d_max,
            region: inp.region,
            cg,
            prior: inp.prior,
            x: Vec::new(),
            y: Vec::new(),
            tx: Vec::new(),
            ty: Vec::new(),
            nx: Vec::new(),
            ny: Vec::new(),
            wx: Vec::new(),
            wy: Vec::new(),
            s: Vec::new(),
            rk: Vec::new(),
            dw: Vec::new(),
            loop_of: Vec::new(),
            loops: Vec::new(),
            comp: Vec::new(),
            owner: Vec::new(),
            fin_sweep: SegIndex::new(&[], 1.0, 1.0),
            outline,
            notes: Vec::new(),
        };
        pk.build_loops(clear, walls)?;
        pk.build_comp(inp.rough_centers);
        pk.build_owner();
        pk.build_sweep();
        // Wall stock height the finisher engages: the stock top just off the wall (0.1 mm
        // into the allowance) left by everything before the rough op, above the floor.
        let mut tops = Vec::new();
        let step = (pk.n() / 80).max(1);
        for i in (0..pk.n()).step_by(step) {
            tops.push((inp.top_before)(pk.wx[i] + 0.1 * pk.nx[i], pk.wy[i] + 0.1 * pk.ny[i]) - inp.z);
        }
        tops.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        pk.ap = tops[tops.len() / 2];
        if tops[tops.len() - 1] - tops[0] > 0.5 {
            pk.notes.push(format!("wall stock height varies {:.2}-{:.2} mm along the wall; the median {:.2} mm is used", tops[0], tops[tops.len() - 1], pk.ap));
        }
        Ok(pk)
    }

    pub fn n(&self) -> usize {
        self.x.len()
    }

    fn build_loops(&mut self, clear: &dyn Clearance, walls: Walls) -> Result<(), String> {
        let rf = self.rf;
        let cands: Vec<Vec<(f64, f64)>> = contour(&self.cg, rf).into_iter().filter(|(p, c)| *c && p.len() >= 8).map(|(p, _)| p).collect();
        let mut picked = Vec::new();
        match &walls {
            Walls::Path(path) => {
                for c in cands {
                    let n = c.len();
                    let segs: Vec<_> = (0..n).map(|k| (c[k].0, c[k].1, c[(k + 1) % n].0, c[(k + 1) % n].1, k as i32)).collect();
                    let idx = SegIndex::new(&segs, 1.5, 2.0);
                    let votes = path.iter().filter(|p| idx.within(p.0, p.1, 1.5)).count();
                    let len = closed_len(&c);
                    if votes as f64 * 0.5 >= 0.25 * len {
                        picked.push(c);
                    } else if votes > 0 {
                        self.notes.push(format!("loop of {len:.0} mm near the contour op but only {:.0} mm of its path on it: skipped", votes as f64 * 0.5));
                    }
                }
            }
            Walls::Reach(seeds) => {
                // The finisher's center region connected to the seeds; a loop bounds it
                // when most of its points sit next to it.
                let reach = flood(&self.cg, seeds, rf, rf);
                for c in cands {
                    let inside = c.iter().filter(|p| self.cg.nearest(p.0, p.1).is_some_and(|s| near(&self.cg, &reach, s))).count();
                    if 2 * inside >= c.len() {
                        picked.push(c);
                    }
                }
            }
        }
        if picked.is_empty() {
            return Err(match walls {
                Walls::Path(_) => format!("{}: no finisher wall loop matches the contour op's path", self.finish_name),
                Walls::Reach(_) => format!("{}: no closed wall loop around the pocket at the finisher's radius", self.finish_name),
            });
        }
        let (mut xs, mut ys, mut lo) = (Vec::new(), Vec::new(), Vec::new());
        for raw in &picked {
            // Exact level set, uniform arc length, exact again.
            let p: Vec<(f64, f64)> = raw.iter().map(|q| project(clear, q.0, q.1, rf)).collect();
            let m = p.len();
            let mut cum = vec![0.0f64; m + 1];
            for k in 0..m {
                cum[k + 1] = cum[k] + (sq(p[(k + 1) % m].0 - p[k].0) + sq(p[(k + 1) % m].1 - p[k].1)).sqrt();
            }
            let len = cum[m];
            let n = 16.max((len / self.ds).round() as usize);
            let mut q2 = Vec::with_capacity(n);
            let mut seg = 0;
            for k in 0..n {
                let s = len * k as f64 / n as f64;
                while seg < m - 1 && cum[seg + 1] < s {
                    seg += 1;
                }
                let f = (s - cum[seg]) / 1e-12f64.max(cum[seg + 1] - cum[seg]);
                let (a, b) = (p[seg], p[(seg + 1) % m]);
                q2.push(project(clear, a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, rf));
            }
            // Climb: the wall on the right of travel, so the normal away from the wall is on the left.
            let mut vote = 0.0;
            for k in 0..n {
                let (a, b) = (q2[(k + n - 1) % n], q2[(k + 1) % n]);
                let (gx, gy) = self.cg.grad(q2[k].0, q2[k].1);
                vote += (b.0 - a.0) * gy - (b.1 - a.1) * gx;
            }
            if vote < 0.0 {
                q2.reverse();
            }
            // Start on the straightest stretch (least turning over +/- 2 mm), away from corners.
            let turn: Vec<f64> = (0..n)
                .map(|k| {
                    let (a, b, c) = (q2[(k + n - 1) % n], q2[k], q2[(k + 1) % n]);
                    let (t1, t2) = ((b.1 - a.1).atan2(b.0 - a.0), (c.1 - b.1).atan2(c.0 - b.0));
                    wrap(t2 - t1).abs()
                })
                .collect();
            let w = 2.max((2.0 / self.ds) as i64);
            let (mut best, mut best_t) = (0, f64::MAX);
            for k in 0..n {
                let mut t = 0.0;
                for d in -w..=w {
                    t += turn[(k as i64 + d).rem_euclid(n as i64) as usize];
                }
                if t < best_t {
                    best_t = t;
                    best = k;
                }
            }
            let (lp, start) = (self.loops.len(), xs.len());
            for k in 0..n {
                let r = q2[(best + k) % n];
                xs.push(r.0);
                ys.push(r.1);
                lo.push(lp);
            }
            // Island: the wall is inside the loop (the loop runs clockwise).
            let mut area = 0.0;
            for k in 0..n {
                let (a, b) = (q2[k], q2[(k + 1) % n]);
                area += a.0 * b.1 - b.0 * a.1;
            }
            self.loops.push(Loop { start, count: n, length: len, island: area < 0.0 });
        }
        let n0 = xs.len();
        self.x = xs;
        self.y = ys;
        self.loop_of = lo;
        for v in [&mut self.tx, &mut self.ty, &mut self.nx, &mut self.ny, &mut self.wx, &mut self.wy, &mut self.s, &mut self.rk, &mut self.dw] {
            *v = vec![0.0; n0];
        }
        for lp in self.loops.clone() {
            let (start, n) = (lp.start, lp.count as i64);
            let at = |k: i64| start + k.rem_euclid(n) as usize;
            for k in 0..n {
                let (i, a, b) = (at(k), at(k - 1), at(k + 1));
                let (tx, ty) = (self.x[b] - self.x[a], self.y[b] - self.y[a]);
                let tl = (tx * tx + ty * ty).sqrt();
                self.tx[i] = tx / tl;
                self.ty[i] = ty / tl;
                let (mut gx, mut gy) = clear.grad(self.x[i], self.y[i]);
                if gx == 0.0 && gy == 0.0 {
                    (gx, gy) = (-self.ty[i], self.tx[i]);
                }
                self.nx[i] = gx;
                self.ny[i] = gy;
                self.wx[i] = self.x[i] - rf * gx;
                self.wy[i] = self.y[i] - rf * gy;
                self.s[i] = if k == 0 { 0.0 } else { self.s[at(k - 1)] + (sq(self.x[i] - self.x[at(k - 1)]) + sq(self.y[i] - self.y[at(k - 1)])).sqrt() };
            }
            for k in 0..n {
                let i = at(k);
                self.dw[i] = 0.5 * (sq(self.wx[at(k + 1)] - self.wx[at(k - 1)]) + sq(self.wy[at(k + 1)] - self.wy[at(k - 1)])).sqrt();
                // Curvature radius of the tool-center path from the turn over +/- 1 mm.
                let w = 1.max((1.0 / self.ds).round() as i64);
                let (a, b) = (at(k - w), at(k + w));
                let (ux, uy, vx, vy) = (self.x[i] - self.x[a], self.y[i] - self.y[a], self.x[b] - self.x[i], self.y[b] - self.y[i]);
                let (nu, nv) = ((ux * ux + uy * uy).sqrt(), (vx * vx + vy * vy).sqrt());
                let c = ((ux * vx + uy * vy) / 1e-12f64.max(nu * nv)).clamp(-1.0, 1.0);
                let turn = c.acos();
                self.rk[i] = if turn < 1e-6 { f64::INFINITY } else { 0.5 * (nu + nv) / turn };
            }
        }
        Ok(())
    }

    /// The rougher's center region at the thinnest allowance, widened by 0.3 mm so the
    /// bilinear F never mixes in the "other component" value near a real boundary;
    /// seeded by the rough op's own tool centers.
    fn build_comp(&mut self, centers: &[(f64, f64)]) {
        let (seed_min, th) = (self.rr + self.d_min, self.rr + self.d_min - 0.3);
        self.comp = flood(&self.cg, centers, seed_min, th);
        if !self.comp.iter().any(|&c| c) {
            self.notes.push("the rough op's tool centers never fit at the thinnest allowance: the rougher clears nothing here".into());
        }
    }

    /// Nodes where the rougher's wall pass can lie (clearance between r_R + d_min and
    /// r_R + d_max, with margin) get the sample whose wall foot is nearest to theirs.
    fn build_owner(&mut self) {
        let feet: Vec<_> = (0..self.n()).map(|i| (self.wx[i], self.wy[i], self.wx[i], self.wy[i], i as i32)).collect();
        let idx = SegIndex::new(&feet, 1.0, 1.0);
        let (lo, hi) = (self.rr + self.d_min - 0.5, self.rr + self.d_max + 0.5);
        let cg = &self.cg;
        let comp = &self.comp;
        let mut owner = vec![-1i32; cg.nx * cg.ny];
        owner.par_chunks_mut(cg.nx).enumerate().for_each(|(j, row)| {
            for (i, o) in row.iter_mut().enumerate() {
                let s = j * cg.nx + i;
                let c = cg.v[s];
                if !comp[s] || c < lo || c > hi {
                    continue;
                }
                let (x, y) = (cg.x(i), cg.y(j));
                let (gx, gy) = cg.grad(x, y);
                let (fx, fy) = (x - c * gx, y - c * gy);
                // Nearest wall foot within 1 mm of this node's foot, else the nearest overall (rare: corners).
                *o = idx.nearest_id(fx, fy, 1.0).unwrap_or_else(|| self.nearest_slow(fx, fy));
            }
        });
        self.owner = owner;
    }

    fn nearest_slow(&self, x: f64, y: f64) -> i32 {
        let (mut best, mut bd) = (-1, f64::MAX);
        for i in 0..self.n() {
            let d = sq(self.wx[i] - x) + sq(self.wy[i] - y);
            if d < bd {
                bd = d;
                best = i as i32;
            }
        }
        best
    }

    fn build_sweep(&mut self) {
        let mut segs = Vec::new();
        for lp in &self.loops {
            for k in 1..lp.count {
                let (a, b) = (lp.start + k - 1, lp.start + k);
                segs.push((self.x[a], self.y[a], self.x[b], self.y[b], b as i32));
            }
        }
        self.fin_sweep = SegIndex::new(&segs, self.rf, 1.0);
    }
}

/// 4-connected flood fill of the nodes at or above `grow`, from the seed points whose
/// nearest node is at or above `seed`.
fn flood(g: &Grid2, seeds: &[(f64, f64)], seed: f64, grow: f64) -> Vec<bool> {
    let mut mark = vec![false; g.nx * g.ny];
    let mut stack = Vec::new();
    for &(x, y) in seeds {
        if let Some(s) = g.nearest(x, y) {
            if !mark[s] && g.v[s] >= seed {
                mark[s] = true;
                stack.push(s);
            }
        }
    }
    while let Some(c) = stack.pop() {
        let (ci, cj) = (c % g.nx, c / g.nx);
        let mut push = |m: usize| {
            if !mark[m] && g.v[m] >= grow {
                mark[m] = true;
                stack.push(m);
            }
        };
        if ci > 0 {
            push(c - 1);
        }
        if ci < g.nx - 1 {
            push(c + 1);
        }
        if cj > 0 {
            push(c - g.nx);
        }
        if cj < g.ny - 1 {
            push(c + g.nx);
        }
    }
    mark
}

/// The node or one of its 4 neighbors is marked.
fn near(g: &Grid2, mark: &[bool], s: usize) -> bool {
    let (i, j) = (s % g.nx, s / g.nx);
    mark[s] || (i > 0 && mark[s - 1]) || (i + 1 < g.nx && mark[s + 1]) || (j > 0 && mark[s - g.nx]) || (j + 1 < g.ny && mark[s + g.nx])
}
