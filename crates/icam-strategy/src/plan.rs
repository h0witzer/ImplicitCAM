//! The clearing-schedule field T for one pocket at one slab.
//!
//!   Dw   the exact distance from a tool center to the wall (clear = r + delta): the
//!        clearance field minus (r + delta), which is itself a distance field.
//!   seed the ridge of Dw (its top level set, the pocket's medial core), reduced to a
//!        segment A-B; the entry (`entry`) ramps down a loop of radius rho around a run
//!        of it and clears the stadium inside
//!   T    0 on that stadium, 1 on the wall, in between either
//!          ratio       T = ds / (ds + Dw), ds = geodesic distance from the stadium,
//!                      then a few Laplace sweeps to round medial kinks
//!          harmonic    Laplace(T) = 0
//!          conductive  div(k grad T) = 0 with k = ds + Dw
//!        or, for max-cut, the geodesic distance itself (fast marching).
//!
//! Seeding on the ridge of the distance field puts every front at about the same
//! distance from its wall, so the fronts come out as implicit offsets of the wall (no
//! curve is offset) and their spacing is nearly uniform. A point seed in an elongated
//! pocket cannot do that: the far end sets the number of revolutions and the near walls
//! get crowded fronts. The PDE is solved on a grid, the only discretized object here;
//! the wall and the stadium enter with sub-cell accuracy (Shortley-Weller arms) from the
//! exact fields.
//!
//! The clearance is exact everywhere (icam_query::SliceClearance has no band), so Dw is
//! simply G on the pocket.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use icam_query::{seg, Box2, Clearance};
use rayon::prelude::*;

use crate::path::P;

const UNREACHED: f64 = 1e18;

/// A node's tentative arrival in fast marching: the heap pops the earliest first, and of
/// equal arrivals the lower node.
struct Arrival(f64, usize);

impl PartialEq for Arrival {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Arrival {}
impl PartialOrd for Arrival {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Arrival {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// One entry per significant maximum of Dw (a "room" of the pocket): the ridge segment
/// A-B around it and a radius: the room `find_seeds` trims the ridge to, then the entry
/// loop's radius (`entry::tracks`). Ridge minima between rooms (saddles of Dw) are
/// where neighboring fronts merge.
#[derive(Clone, Debug, PartialEq)]
pub struct Seed {
    pub ax: f64,
    pub ay: f64,
    pub bx: f64,
    pub by: f64,
    pub rho: f64,
    pub height: f64,
    pub persistence: f64,
}

impl Seed {
    pub fn mid_x(&self) -> f64 {
        0.5 * (self.ax + self.bx)
    }
    pub fn mid_y(&self) -> f64 {
        0.5 * (self.ay + self.by)
    }
}

struct Stencil {
    unknown: Vec<usize>,
    w4: Vec<f64>,
    b: Vec<f64>,
    w: Vec<f64>,
}

pub struct Plan<'a> {
    /// Grid pitch, mm.
    pub h: f64,
    pub x0: f64,
    pub y0: f64,
    pub nx: usize,
    pub ny: usize,
    /// clear - (r + delta).
    pub g: Vec<f64>,
    /// Exact wall distance on the pocket's nodes.
    pub dw: Vec<f64>,
    /// 1 = the pocket's free nodes, 0 = elsewhere.
    pub label: Vec<u8>,
    /// Plan value; 1 outside the pocket, 0 inside the entry stadium.
    pub t: Vec<f64>,
    pub d_max: f64,
    pub iterations: usize,
    pub regions: usize,
    pub residual: f64,
    pub seeds: Vec<Seed>,
    /// The conductivity of the last conductive solve (the objective plan iterates it).
    pub k: Option<Vec<f64>>,
    /// Max-cut: the largest arrival time on the pocket.
    pub max_t: f64,
    clear: &'a dyn Clearance,
    limit: f64,
}

impl<'a> Plan<'a> {
    /// The plan grid over `bx` at pitch `h` for a tool whose center keeps `limit` (r +
    /// delta) from the part. The pocket is the 4-connected free component holding most
    /// of the `hints`.
    pub fn new(clear: &'a dyn Clearance, limit: f64, bx: Box2, h: f64, hints: &[P]) -> Plan<'a> {
        let (x0, y0) = (bx.x0, bx.y0);
        let nx = ((bx.x1 - bx.x0) / h).ceil() as usize + 1;
        let ny = ((bx.y1 - bx.y0) / h).ceil() as usize + 1;
        let n = nx * ny;
        let mut g = vec![0.0; n];
        g.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = clear.at(x0 + i as f64 * h, y0 + j as f64 * h) - limit;
            }
        });
        let mut p = Plan {
            h,
            x0,
            y0,
            nx,
            ny,
            g,
            dw: vec![0.0; n],
            label: vec![0; n],
            t: vec![0.0; n],
            d_max: 0.0,
            iterations: 0,
            regions: 0,
            residual: 0.0,
            seeds: Vec::new(),
            k: None,
            max_t: 0.0,
            clear,
            limit,
        };

        let neighbors4 = |c: usize| -> [Option<usize>; 4] {
            let (ci, cj) = (c % nx, c / nx);
            [
                (ci > 0).then(|| c - 1),
                (ci < nx - 1).then(|| c + 1),
                (cj > 0).then(|| c - nx),
                (cj < ny - 1).then(|| c + nx),
            ]
        };
        let mut comp = vec![0usize; n];
        let mut ncomp = 0;
        let mut stack = Vec::new();
        for s in 0..n {
            if p.g[s] <= 0.0 || comp[s] != 0 {
                continue;
            }
            ncomp += 1;
            comp[s] = ncomp;
            stack.push(s);
            while let Some(c) = stack.pop() {
                for m in neighbors4(c).into_iter().flatten() {
                    if p.g[m] > 0.0 && comp[m] == 0 {
                        comp[m] = ncomp;
                        stack.push(m);
                    }
                }
            }
        }
        let mut votes = vec![0usize; ncomp + 1];
        for &(x, y) in hints {
            let i = ((x - x0) / h).round() as i64;
            let j = ((y - y0) / h).round() as i64;
            if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny {
                votes[comp[j as usize * nx + i as usize]] += 1;
            }
        }
        let mut best = 0;
        for k in 1..=ncomp {
            if votes[k] > votes[best] {
                best = k;
            }
        }
        if best == 0 {
            p.t.fill(1.0);
            return p;
        }
        // The pocket's open area (clear > 0) can hold several tool-center regions joined
        // only by necks narrower than the tool. Take every region inside the open area
        // around the main one; each gets its own seeds and wall.
        let mut open = vec![false; n];
        let start = comp.iter().position(|&c| c == best).unwrap_or(0);
        open[start] = true;
        stack.push(start);
        while let Some(c) = stack.pop() {
            for m in neighbors4(c).into_iter().flatten() {
                if !open[m] && p.g[m] + limit > 0.0 {
                    open[m] = true;
                    stack.push(m);
                }
            }
        }
        let mut in_open: HashSet<usize> = (0..n).filter(|&s| comp[s] != 0 && open[s]).map(|s| comp[s]).collect();
        // A region reaching the grid edge is the open sheet outside the pocket, not part of it.
        for s in 0..n {
            let (i, j) = (s % nx, s / nx);
            if comp[s] != 0 && comp[s] != best && (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
                in_open.remove(&comp[s]);
            }
        }
        for s in 0..n {
            p.label[s] = u8::from(comp[s] != 0 && in_open.contains(&comp[s]));
        }
        p.regions = in_open.len();
        for s in 0..n {
            if p.label[s] == 1 {
                p.dw[s] = p.g[s];
                p.d_max = p.d_max.max(p.dw[s]);
            }
        }
        p
    }

    pub fn ok(&self) -> bool {
        self.d_max > 0.0
    }

    pub fn g_max(&self) -> f64 {
        self.d_max
    }

    /// The largest entry loop radius among the seeds.
    pub fn rho(&self) -> f64 {
        self.seeds.iter().map(|s| s.rho).fold(None, |m: Option<f64>, r| Some(m.map_or(r, |m| m.max(r)))).unwrap_or(0.0)
    }

    pub fn x(&self, i: usize) -> f64 {
        self.x0 + i as f64 * self.h
    }

    pub fn y(&self, j: usize) -> f64 {
        self.y0 + j as f64 * self.h
    }

    fn bilinear(&self, f: &[f64], x: f64, y: f64) -> f64 {
        let (fx, fy) = ((x - self.x0) / self.h, (y - self.y0) / self.h);
        let i = (fx.floor() as i64).clamp(0, self.nx as i64 - 2) as usize;
        let j = (fy.floor() as i64).clamp(0, self.ny as i64 - 2) as usize;
        let (u, v) = (fx - i as f64, fy - j as f64);
        let s = j * self.nx + i;
        (1.0 - u) * (1.0 - v) * f[s] + u * (1.0 - v) * f[s + 1] + (1.0 - u) * v * f[s + self.nx] + u * v * f[s + self.nx + 1]
    }

    pub fn wall_dist(&self, x: f64, y: f64) -> f64 {
        self.bilinear(&self.dw, x, y)
    }

    /// Significant maxima of Dw by persistence: sweep the free nodes from high to low
    /// Dw, union-find over 8-neighbors; when two components meet at a node (a saddle, a
    /// ridge minimum), the one with the lower peak dies there, its persistence = peak -
    /// saddle. Peaks with persistence >= `min_persistence` (and room for an entry loop) become
    /// seeds, highest first, at most `max_seeds`.
    pub fn peaks(&self, min_persistence: f64, min_height: f64, max_seeds: usize) -> Vec<(usize, f64)> {
        let (nx, ny) = (self.nx, self.ny);
        let mut order: Vec<usize> = (0..self.dw.len()).filter(|&s| self.label[s] == 1).collect();
        order.sort_by(|&a, &b| self.dw[b].partial_cmp(&self.dw[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut parent = vec![usize::MAX; self.dw.len()];
        let mut peak = vec![0usize; self.dw.len()];
        // Insertion-ordered: the ties of the final sort keep this order.
        let mut pers: Vec<(usize, f64)> = Vec::new();
        let mut pers_at: HashMap<usize, usize> = HashMap::new();
        let mut set = |pers: &mut Vec<(usize, f64)>, k: usize, v: f64| match pers_at.get(&k) {
            Some(&i) => pers[i].1 = v,
            None => {
                pers_at.insert(k, pers.len());
                pers.push((k, v));
            }
        };
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for &s in &order {
            parent[s] = s;
            peak[s] = s;
            let (i, j) = ((s % nx) as i64, (s / nx) as i64);
            for dj in -1..=1i64 {
                for di in -1..=1i64 {
                    let (ii, jj) = (i + di, j + dj);
                    if (di == 0 && dj == 0) || ii < 0 || jj < 0 || ii >= nx as i64 || jj >= ny as i64 {
                        continue;
                    }
                    let m = jj as usize * nx + ii as usize;
                    if parent[m] == usize::MAX {
                        continue;
                    }
                    let (ra, rb) = (find(&mut parent, s), find(&mut parent, m));
                    if ra == rb {
                        continue;
                    }
                    let lo = if self.dw[peak[ra]] < self.dw[peak[rb]] { ra } else { rb };
                    let hi = if lo == ra { rb } else { ra };
                    if peak[lo] != s {
                        set(&mut pers, peak[lo], self.dw[peak[lo]] - self.dw[s]);
                    }
                    parent[lo] = hi;
                }
            }
        }
        // Each region's highest peak never dies.
        for &s in &order {
            if parent[s] == s {
                set(&mut pers, peak[s], f64::INFINITY);
            }
        }
        let mut out: Vec<(usize, f64)> = pers.into_iter().filter(|&(k, v)| v >= min_persistence && self.dw[k] >= min_height).collect();
        out.sort_by(|a, b| self.dw[b.0].partial_cmp(&self.dw[a.0]).unwrap_or(std::cmp::Ordering::Equal));
        // Every region keeps its highest peak (a region without one is never cut); the
        // other rooms fill up to max_seeds.
        let mut rooms = max_seeds.saturating_sub(out.iter().filter(|p| p.1.is_infinite()).count());
        out.retain(|p| {
            if p.1.is_infinite() {
                return true;
            }
            let keep = rooms > 0;
            rooms = rooms.saturating_sub(1);
            keep
        });
        out
    }

    /// The free components of the pocket (4-connected, as the pocket is found): per node,
    /// 0 off the pocket, else the component's number from 1.
    pub fn components(&self) -> Vec<u32> {
        let (nx, ny) = (self.nx, self.ny);
        let mut comp = vec![0u32; nx * ny];
        let mut n = 0;
        let mut stack = Vec::new();
        for s in 0..nx * ny {
            if self.label[s] != 1 || comp[s] != 0 {
                continue;
            }
            n += 1;
            comp[s] = n;
            stack.push(s);
            while let Some(c) = stack.pop() {
                let (i, j) = (c % nx, c / nx);
                for m in [(i > 0).then(|| c - 1), (i + 1 < nx).then(|| c + 1), (j > 0).then(|| c - nx), (j + 1 < ny).then(|| c + nx)].into_iter().flatten() {
                    if self.label[m] == 1 && comp[m] == 0 {
                        comp[m] = n;
                        stack.push(m);
                    }
                }
            }
        }
        comp
    }

    /// The component (of [`Plan::components`]) of the pocket node nearest (x, y) within
    /// three cells, 0 when there is none.
    pub fn component_at(&self, comp: &[u32], x: f64, y: f64) -> u32 {
        let (fi, fj) = ((x - self.x0) / self.h, (y - self.y0) / self.h);
        let (ci, cj) = (fi.round() as i64, fj.round() as i64);
        let mut best = (f64::INFINITY, 0u32);
        for j in cj - 3..=cj + 3 {
            for i in ci - 3..=ci + 3 {
                if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.ny as i64 {
                    continue;
                }
                let c = comp[j as usize * self.nx + i as usize];
                let d = (i as f64 - fi).hypot(j as f64 - fj);
                if c != 0 && d < best.0 {
                    best = (d, c);
                }
            }
        }
        best.1
    }

    /// Keeps only the pocket's nodes where `keep` is true; the others leave it (the walls
    /// stay where they are). Call before the seeds are found.
    pub fn restrict(&mut self, keep: &[bool]) {
        for s in 0..self.label.len() {
            if self.label[s] == 1 && !keep[s] {
                self.label[s] = 0;
                self.dw[s] = 0.0;
            }
        }
        self.regions = self.components().iter().copied().max().unwrap_or(0) as usize;
        self.d_max = (0..self.label.len()).filter(|&s| self.label[s] == 1).map(|s| self.dw[s]).fold(0.0, f64::max);
    }

    /// A seed for every free region left without one because it is too small for a room
    /// (a sliver where a narrow slot meets a wider one): rho 0, and as its ridge
    /// the longest straight run between its nodes that stays in it, along its principal
    /// axis, which the entry ramps down along (straight down when the run is a point).
    /// Call after `find_seeds`.
    pub fn add_sliver_seeds(&mut self) {
        let comp = self.components();
        let count = comp.iter().copied().max().unwrap_or(0) as usize;
        let mut seeded = vec![false; count + 1];
        for e in &self.seeds {
            seeded[self.component_at(&comp, e.mid_x(), e.mid_y()) as usize] = true;
        }
        let mut nodes: Vec<Vec<usize>> = vec![Vec::new(); count + 1];
        for (s, &c) in comp.iter().enumerate() {
            if c != 0 && !seeded[c as usize] {
                nodes[c as usize].push(s);
            }
        }
        for v in nodes.iter().filter(|v| !v.is_empty()) {
            let pts: Vec<P> = v.iter().map(|&s| (self.x(s % self.nx), self.y(s / self.nx))).collect();
            let top = *v.iter().max_by(|&&a, &&b| self.dw[a].total_cmp(&self.dw[b])).unwrap();
            let m = pts.len() as f64;
            let (cx, cy) = (pts.iter().map(|p| p.0).sum::<f64>() / m, pts.iter().map(|p| p.1).sum::<f64>() / m);
            let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
            for p in &pts {
                sxx += (p.0 - cx) * (p.0 - cx);
                syy += (p.1 - cy) * (p.1 - cy);
                sxy += (p.0 - cx) * (p.1 - cy);
            }
            let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
            let along = |p: &P| (p.0 - cx) * ang.cos() + (p.1 - cy) * ang.sin();
            let a = *pts.iter().min_by(|p, q| along(p).total_cmp(&along(q))).unwrap();
            let b = *pts.iter().max_by(|p, q| along(p).total_cmp(&along(q))).unwrap();
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            let steps = 1.max((len / (0.25 * self.h)).ceil() as usize);
            let inside = (0..=steps).all(|k| {
                let u = k as f64 / steps as f64;
                self.in_pocket(a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)
            });
            let (a, b) = if inside { (a, b) } else { let t = (self.x(top % self.nx), self.y(top / self.nx)); (t, t) };
            self.seeds.push(Seed { ax: a.0, ay: a.1, bx: b.0, by: b.1, rho: 0.0, height: self.dw[top], persistence: f64::INFINITY });
        }
    }

    /// Seeds at the pocket's significant ridge maxima; each gets its local ridge as a
    /// segment and a radius that fits there.
    pub fn find_seeds(&mut self, rho_max: f64, rho_min: f64, min_persistence: f64, max_seeds: usize) {
        self.seeds.clear();
        for (node, p) in self.peaks(min_persistence, rho_min / 0.7, max_seeds) {
            let rho = rho_max.clamp(rho_min, 0.7 * self.dw[node]);
            let kappa = if p.is_infinite() { (0.1 * self.dw[node]).max(1.0) } else { (0.1 * self.dw[node]).max(1.0).min(0.5 * p) };
            let (ax, ay, bx, by) = self.ridge(node, rho, kappa);
            self.seeds.push(Seed { ax, ay, bx, by, rho, height: self.dw[node], persistence: p });
        }
    }

    /// The ridge around a peak: nodes connected to it with Dw within kappa of its
    /// height, reduced to their principal segment, trimmed so a circle of radius rho
    /// around every point of it stays in the pocket. A short ridge becomes a point.
    fn ridge(&self, node: usize, rho: f64, kappa: f64) -> (f64, f64, f64, f64) {
        let (nx, ny) = (self.nx, self.ny);
        let (mx, my, floor) = (self.x(node % nx), self.y(node / nx), self.dw[node] - kappa);
        let mut core: Vec<P> = Vec::new();
        let mut seen = HashSet::from([node]);
        let mut q = VecDeque::from([node]);
        while let Some(s) = q.pop_front() {
            core.push((self.x(s % nx), self.y(s / nx)));
            let (i, j) = ((s % nx) as i64, (s / nx) as i64);
            for dj in -1..=1i64 {
                for di in -1..=1i64 {
                    let (ii, jj) = (i + di, j + dj);
                    if ii < 0 || jj < 0 || ii >= nx as i64 || jj >= ny as i64 {
                        continue;
                    }
                    let m = jj as usize * nx + ii as usize;
                    if self.label[m] == 1 && self.dw[m] >= floor && seen.insert(m) {
                        q.push_back(m);
                    }
                }
            }
        }
        let count = core.len() as f64;
        let cx = core.iter().map(|p| p.0).sum::<f64>() / count;
        let cy = core.iter().map(|p| p.1).sum::<f64>() / count;
        let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
        for p in &core {
            sxx += (p.0 - cx) * (p.0 - cx);
        }
        for p in &core {
            syy += (p.1 - cy) * (p.1 - cy);
        }
        for p in &core {
            sxy += (p.0 - cx) * (p.1 - cy);
        }
        let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let (ux, uy) = (ang.cos(), ang.sin());
        let along = |p: &P| (p.0 - cx) * ux + (p.1 - cy) * uy;
        let (mut t0, mut t1) = (along(&core[0]), along(&core[0]));
        for p in &core[1..] {
            let a = along(p);
            if a < t0 {
                t0 = a;
            }
            if a > t1 {
                t1 = a;
            }
        }
        // Keep the longest run of the segment whose points have room for the circle.
        let (step, mut best_len, mut ra, mut rb, mut run_start) = (self.h, -1.0, 0.0, 0.0, f64::NAN);
        let mut t = t0;
        while t <= t1 + 1e-9 {
            let ok = self.wall_dist(cx + ux * t, cy + uy * t) >= rho + 0.25;
            if ok && run_start.is_nan() {
                run_start = t;
            }
            if (!ok || t + step > t1 + 1e-9) && !run_start.is_nan() {
                let end = if ok { t } else { t - step };
                if end - run_start > best_len {
                    best_len = end - run_start;
                    ra = run_start;
                    rb = end;
                }
                run_start = f64::NAN;
            }
            t += step;
        }
        if best_len < 2.0 * rho {
            return (mx, my, mx, my);
        }
        (cx + ux * ra, cy + uy * ra, cx + ux * rb, cy + uy * rb)
    }

    /// Harmonic plan (Laplace, SOR to convergence).
    pub fn solve(&mut self, tol: f64, max_iter: usize) {
        let st = self.stencil(None);
        for &s in &st.unknown {
            self.t[s] = 0.5;
        }
        self.relax(&st, 1.94, tol, max_iter);
    }

    /// Distance-ratio plan: fronts reach every wall together; with the ridge seed, ds +
    /// Dw is nearly constant along the flow, so spacing is nearly uniform.
    pub fn solve_ratio(&mut self, smooth: usize) {
        let st = self.stencil(None);
        let ds = self.geodesic();
        for &s in &st.unknown {
            self.t[s] = ds[s] / (ds[s] + self.dw[s].max(1e-9));
        }
        self.relax(&st, 1.0, 0.0, smooth);
    }

    /// Designed-conductivity plan: div(k grad T) = 0 with k = ds + Dw, the
    /// entry-to-wall length through each point. Along a straight flow line (k T')' = 0
    /// gives T' ~ 1/k = 1/L, the ratio field's even spacing; unlike the ratio field this
    /// is elliptic, so the maximum principle holds and no material can be stranded (the
    /// ratio field strands rings between islands).
    pub fn solve_conductive(&mut self, tol: f64, max_iter: usize) {
        let ds = self.geodesic();
        let k: Vec<f64> =
            (0..ds.len()).map(|s| if self.label[s] == 1 && ds[s] < f64::MAX { self.h.max(ds[s] + self.dw[s]) } else { 0.0 }).collect();
        let st = self.stencil(Some(&k));
        for &s in &st.unknown {
            self.t[s] = ds[s] / (ds[s] + self.dw[s].max(1e-9));
        }
        self.relax(&st, 1.9, tol, max_iter);
        self.k = Some(k);
    }

    /// The conductive plan again with a given conductivity, warm-started from T.
    pub fn solve_conductive_with(&mut self, k: Vec<f64>, tol: f64, max_iter: usize) {
        let st = self.stencil(Some(&k));
        self.relax(&st, 1.9, tol, max_iter);
        self.k = Some(k);
    }

    /// Max-cut plan: T = geodesic distance (mm) from the entry stadiums through the free
    /// area, by fast marching (first-order upwind eikonal |grad T| = 1). Its level sets
    /// are exactly one stepover apart everywhere, and they simply end where they meet a
    /// wall: the profile shapes nothing but the last pass. With a speed field F
    /// (relative, > 0): |grad T| = 1 / F, so T = k sRef are fronts F(x) sRef apart.
    pub fn solve_eikonal(&mut self, speed: Option<&[f64]>) {
        let n = self.nx * self.ny;
        self.t = self.eikonal(speed);
        let mut far = 0.0f64;
        for s in 0..n {
            if self.label[s] == 1 && self.t[s] < UNREACHED {
                far = far.max(self.t[s]);
            }
        }
        self.max_t = far;
        // Off the free area: one layer extrapolated (for gradients at the wall), then far.
        let mut ext = self.t.clone();
        for (s, e) in ext.iter_mut().enumerate() {
            if self.label[s] == 1 && self.t[s] < UNREACHED {
                continue;
            }
            let (i, j) = ((s % self.nx) as i64, (s / self.nx) as i64);
            let best = self.tv(&self.t, i - 1, j).min(self.tv(&self.t, i + 1, j)).min(self.tv(&self.t, i, j - 1).min(self.tv(&self.t, i, j + 1)));
            *e = if best < UNREACHED { best + self.h } else { far + 50.0 };
        }
        self.t = ext;
    }

    fn tv(&self, t: &[f64], i: i64, j: i64) -> f64 {
        if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.ny as i64 || self.label[j as usize * self.nx + i as usize] != 1 {
            UNREACHED
        } else {
            t[j as usize * self.nx + i as usize]
        }
    }

    /// Arrival time from the entry stadiums through the free nodes by fast marching
    /// (first-order upwind |grad T| = 1 / F, 4-neighbor stencil), UNREACHED where no
    /// front gets. Nodes within 1.5 cells of a stadium start from its exact distance.
    fn eikonal(&self, speed: Option<&[f64]>) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let mut t = vec![UNREACHED; n];
        let mut known = vec![false; n];
        let mut pq = BinaryHeap::new();
        for s in 0..n {
            if self.label[s] != 1 {
                continue;
            }
            let e = self.seed_dist(s);
            if e <= self.h * 1.5 {
                t[s] = e.max(0.0) / speed.map_or(1.0, |f| f[s]);
                pq.push(Arrival(t[s], s));
            }
        }
        while let Some(Arrival(ts, s)) = pq.pop() {
            if known[s] || ts > t[s] {
                continue;
            }
            known[s] = true;
            let (si, sj) = ((s % nx) as i64, (s / nx) as i64);
            for (i, j) in [(si - 1, sj), (si + 1, sj), (si, sj - 1), (si, sj + 1)] {
                if i < 0 || j < 0 || i >= nx as i64 || j >= ny as i64 {
                    continue;
                }
                let m = j as usize * nx + i as usize;
                if self.label[m] != 1 || known[m] {
                    continue;
                }
                let a = self.tv(&t, i - 1, j).min(self.tv(&t, i + 1, j));
                let b = self.tv(&t, i, j - 1).min(self.tv(&t, i, j + 1));
                let hm = self.h / speed.map_or(1.0, |f| f[m]).max(1e-3);
                let tm = if (a - b).abs() >= hm { a.min(b) + hm } else { 0.5 * (a + b + (2.0 * hm * hm - (a - b) * (a - b)).sqrt()) };
                if tm < t[m] {
                    t[m] = tm;
                    pq.push(Arrival(tm, m));
                }
            }
        }
        t
    }

    /// Pieces of the front T = level inside the free area: closed loops, and open arcs
    /// whose ends are carried out to the exact wall.
    pub fn front_pieces(&self, level: f64) -> Vec<(Vec<P>, bool)> {
        let nx = self.nx;
        let free = |i: usize, j: usize| {
            self.label[j * nx + i] == 1 && self.label[j * nx + i + 1] == 1 && self.label[(j + 1) * nx + i] == 1 && self.label[(j + 1) * nx + i + 1] == 1
        };
        let segs = self.march(&self.t, level, Some(&free));
        let key = |p: P| ((p.0 * 1e6).round() as i64, (p.1 * 1e6).round() as i64);
        let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (k, &(a, b)) in segs.iter().enumerate() {
            adj.entry(key(a)).or_default().push(k);
            adj.entry(key(b)).or_default().push(k);
        }
        let degree = |p: P| adj.get(&key(p)).map_or(0, Vec::len);
        let mut used = vec![false; segs.len()];
        let mut pieces = Vec::new();
        // Chain from every open end first (degree-1 points), then what is left is closed loops.
        let mut order: Vec<usize> = (0..segs.len()).collect();
        order.sort_by_key(|&k| if degree(segs[k].0) == 1 || degree(segs[k].1) == 1 { 0 } else { 1 });
        for k0 in order {
            if used[k0] {
                continue;
            }
            used[k0] = true;
            let (mut a, mut b) = segs[k0];
            if degree(b) == 1 && degree(a) != 1 {
                std::mem::swap(&mut a, &mut b); // start at the open end
            }
            let mut chain = vec![a, b];
            let mut end = b;
            loop {
                let next = adj[&key(end)].iter().copied().find(|&c| !used[c]);
                let Some(next) = next else { break };
                used[next] = true;
                end = if key(segs[next].0) == key(end) { segs[next].1 } else { segs[next].0 };
                chain.push(end);
            }
            let closed = key(chain[0]) == key(chain[chain.len() - 1]);
            if closed {
                chain.pop();
            }
            let mut len = 0.0;
            for q in 1..chain.len() {
                len += ((chain[q].0 - chain[q - 1].0) * (chain[q].0 - chain[q - 1].0) + (chain[q].1 - chain[q - 1].1) * (chain[q].1 - chain[q - 1].1)).sqrt();
            }
            if len < 1.0 {
                continue;
            }
            if !closed {
                chain[0] = self.to_wall(chain[1], chain[0]);
                let last = chain.len() - 1;
                chain[last] = self.to_wall(chain[last - 1], chain[last]);
            }
            pieces.push((chain, closed));
        }
        pieces
    }

    /// Marching squares over the nodal field `f` at `level`: one or two segments per
    /// cell (the saddle cases decided by the cell's center value). `keep` limits the
    /// cells.
    fn march(&self, f: &[f64], level: f64, keep: Option<&dyn Fn(usize, usize) -> bool>) -> Vec<(P, P)> {
        let nx = self.nx;
        let at = |i: usize, j: usize| f[j * nx + i];
        let cut = |i0: usize, j0: usize, i1: usize, j1: usize| -> P {
            let (f0, f1) = (at(i0, j0), at(i1, j1));
            let s = (level - f0) / (f1 - f0);
            (self.x(i0) + (self.x(i1) - self.x(i0)) * s, self.y(j0) + (self.y(j1) - self.y(j0)) * s)
        };
        let mut segs = Vec::new();
        for j in 0..self.ny - 1 {
            for i in 0..nx - 1 {
                if keep.is_some_and(|k| !k(i, j)) {
                    continue;
                }
                // Corner order: 0 (i,j), 1 (i+1,j), 2 (i+1,j+1), 3 (i,j+1).
                let code = u8::from(at(i, j) > level) | (u8::from(at(i + 1, j) > level) << 1) | (u8::from(at(i + 1, j + 1) > level) << 2) | (u8::from(at(i, j + 1) > level) << 3);
                if code == 0 || code == 15 {
                    continue;
                }
                let e0 = || cut(i, j, i + 1, j);
                let e1 = || cut(i + 1, j, i + 1, j + 1);
                let e2 = || cut(i, j + 1, i + 1, j + 1);
                let e3 = || cut(i, j, i, j + 1);
                let center = 0.25 * (at(i, j) + at(i + 1, j) + at(i + 1, j + 1) + at(i, j + 1));
                match code {
                    1 | 14 => segs.push((e3(), e0())),
                    2 | 13 => segs.push((e0(), e1())),
                    3 | 12 => segs.push((e3(), e1())),
                    4 | 11 => segs.push((e1(), e2())),
                    6 | 9 => segs.push((e0(), e2())),
                    7 | 8 => segs.push((e3(), e2())),
                    5 => {
                        if center > level {
                            segs.push((e3(), e2()));
                            segs.push((e0(), e1()));
                        } else {
                            segs.push((e3(), e0()));
                            segs.push((e1(), e2()));
                        }
                    }
                    _ => {
                        if center > level {
                            segs.push((e3(), e0()));
                            segs.push((e1(), e2()));
                        } else {
                            segs.push((e3(), e2()));
                            segs.push((e0(), e1()));
                        }
                    }
                }
            }
        }
        segs
    }

    /// Carry an open front's end point out along its end direction to the exact wall.
    fn to_wall(&self, prev: P, end: P) -> P {
        let (mut ux, mut uy) = (end.0 - prev.0, end.1 - prev.1);
        let l = (ux * ux + uy * uy).sqrt();
        if l < 1e-9 {
            return end;
        }
        ux /= l;
        uy /= l;
        let mut p = end;
        let mut k = 0;
        while k < 20 && self.clearance(p.0 + ux * 0.1, p.1 + uy * 0.1) > 0.0 {
            p = (p.0 + ux * 0.1, p.1 + uy * 0.1);
            k += 1;
        }
        let w = self.project_to_wall(p.0, p.1);
        if self.clearance(w.0, w.1).abs() < 1e-3 && (w.0 - p.0) * (w.0 - p.0) + (w.1 - p.1) * (w.1 - p.1) < 1.0 {
            w
        } else {
            p
        }
    }

    /// Grid nodes that are interior local maxima of T (stranded material in the plan).
    pub fn interior_maxima(&self) -> usize {
        let nx = self.nx;
        let mut n = 0;
        for j in 1..self.ny - 1 {
            for i in 1..nx - 1 {
                let s = j * nx + i;
                if self.label[s] != 1 || self.seed_dist(s) <= 0.0 {
                    continue;
                }
                let mut max = true;
                'outer: for dj in -1..=1i64 {
                    for di in -1..=1i64 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let m = (s as i64 + dj * nx as i64 + di) as usize;
                        if self.label[m] != 1 || self.t[m] >= self.t[s] {
                            max = false;
                            break 'outer;
                        }
                    }
                }
                if max {
                    n += 1;
                }
            }
        }
        n
    }

    /// Geodesic distance from the entry stadium through the pocket's free nodes
    /// (f64::MAX where unreached). Fast marching, not graph Dijkstra: an 8-neighbor
    /// chamfer overestimates by up to 8.2% at 22.5 degrees off the grid axes, which bent
    /// the ratio and conductive plans' k = ds + Dw.
    fn geodesic(&self) -> Vec<f64> {
        let mut d = self.eikonal(None);
        for v in &mut d {
            if *v >= UNREACHED {
                *v = f64::MAX;
            }
        }
        d
    }

    /// Unknown nodes and their Shortley-Weller weights; sets T = 1 outside, 0 in the stadium.
    fn stencil(&mut self, cond: Option<&[f64]>) -> Stencil {
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let (mut w4, mut b, mut w) = (vec![0.0; n * 4], vec![0.0; n], vec![0.0; n]);
        let mut unknown = Vec::new();
        for s in 0..n {
            if self.label[s] != 1 {
                self.t[s] = 1.0;
                continue;
            }
            if self.seed_dist(s) <= 0.0 {
                self.t[s] = 0.0;
                continue;
            }
            unknown.push(s);
        }
        let mut is_unknown = vec![false; n];
        for &s in &unknown {
            is_unknown[s] = true;
        }
        for &s in &unknown {
            let (i, j) = (s % nx, s / nx);
            let nb = [(i > 0).then(|| s - 1), (i < nx - 1).then(|| s + 1), (j > 0).then(|| s - nx), (j < ny - 1).then(|| s + nx)];
            let (mut arm, mut val, mut interior) = ([0.0; 4], [0.0; 4], [false; 4]);
            for k in 0..4 {
                let m = nb[k];
                if let Some(m) = m {
                    if is_unknown[m] {
                        arm[k] = 1.0;
                        interior[k] = true;
                        continue;
                    }
                }
                // Boundary between s and m: the wall (G crosses 0) or the stadium (seed distance crosses 0).
                let (gs, ds) = (self.g[s], self.seed_dist(s));
                let gm = m.map_or(-gs, |m| self.g[m]);
                let dm = m.map_or(ds, |m| self.seed_dist(m));
                let tw = if gm <= 0.0 { gs / (gs - gm) } else { 2.0 };
                let th = if dm <= 0.0 { ds / (ds - dm) } else { 2.0 };
                if tw <= th {
                    arm[k] = tw.max(0.02);
                    val[k] = 1.0;
                } else {
                    arm[k] = th.max(0.02);
                    val[k] = 0.0;
                }
            }
            // Shortley-Weller: u'' ~ 2/(a+c) * ((u+ - u0)/c + (u- - u0)/a) with arms a (minus side) and c (plus side).
            let ax = 2.0 / (arm[0] + arm[1]);
            let ay = 2.0 / (arm[2] + arm[3]);
            let mut coef = [ax / arm[0], ax / arm[1], ay / arm[2], ay / arm[3]];
            for k in 0..4 {
                // Conductivity on the face: the mean of the two nodes (the node's own at a boundary arm).
                if let Some(c) = cond {
                    coef[k] *= if interior[k] { 0.5 * (c[s] + c[nb[k].unwrap_or(s)]) } else { c[s] };
                }
                w[s] += coef[k];
                if interior[k] {
                    w4[s * 4 + k] = coef[k];
                } else {
                    b[s] += coef[k] * val[k];
                }
            }
        }
        Stencil { unknown, w4, b, w }
    }

    /// Gauss-Seidel / SOR sweeps; stops at `max_iter` or when the largest update is below `tol`.
    fn relax(&mut self, st: &Stencil, omega: f64, tol: f64, max_iter: usize) {
        let nx = self.nx;
        self.iterations = 0;
        while self.iterations < max_iter {
            let mut maxd = 0.0f64;
            for &s in &st.unknown {
                let mut sum = st.b[s];
                if st.w4[s * 4] != 0.0 {
                    sum += st.w4[s * 4] * self.t[s - 1];
                }
                if st.w4[s * 4 + 1] != 0.0 {
                    sum += st.w4[s * 4 + 1] * self.t[s + 1];
                }
                if st.w4[s * 4 + 2] != 0.0 {
                    sum += st.w4[s * 4 + 2] * self.t[s - nx];
                }
                if st.w4[s * 4 + 3] != 0.0 {
                    sum += st.w4[s * 4 + 3] * self.t[s + nx];
                }
                let u = sum / st.w[s];
                let d = u - self.t[s];
                self.t[s] += omega * d;
                if d.abs() > maxd {
                    maxd = d.abs();
                }
            }
            self.residual = maxd;
            if maxd < tol {
                break;
            }
            self.iterations += 1;
        }
    }

    /// Signed distance from a node to the nearest entry stadium (a seed's segment
    /// widened by its rho).
    fn seed_dist(&self, s: usize) -> f64 {
        let (x, y) = (self.x(s % self.nx), self.y(s / self.nx));
        let mut d = f64::MAX;
        for e in &self.seeds {
            d = d.min(seg(x, y, e.ax, e.ay, e.bx, e.by) - e.rho);
        }
        d
    }

    /// Bilinear T.
    pub fn value(&self, x: f64, y: f64) -> f64 {
        self.bilinear(&self.t, x, y)
    }

    /// Gradient of the bilinear T by central differences over half a cell.
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        let e = self.h * 0.5;
        ((self.value(x + e, y) - self.value(x - e, y)) / (2.0 * e), (self.value(x, y + e) - self.value(x, y - e)) / (2.0 * e))
    }

    /// Follow the gradient flow (the fronts' orthogonal trajectories) from p, which sits
    /// at level t0, to level t1. dx/dT = grad T / |grad T|^2 (RK2). Each substep is
    /// capped at `max_step` mm: where the gradient nearly vanishes (a plateau, or off the
    /// pocket) dx = grad/|grad|^2 would jump away.
    pub fn flow(&self, mut x: f64, mut y: f64, t0: f64, t1: f64, steps: usize, max_step: f64) -> P {
        let dt = (t1 - t0) / steps as f64;
        let step = |px: f64, py: f64, h: f64| -> P {
            let (gx, gy) = self.gradient(px, py);
            let g2 = (gx * gx + gy * gy).max(1e-12);
            let (mut sx, mut sy) = (h * gx / g2, h * gy / g2);
            let l = (sx * sx + sy * sy).sqrt();
            if l > max_step {
                sx *= max_step / l;
                sy *= max_step / l;
            }
            (sx, sy)
        };
        for _ in 0..steps {
            let (hx, hy) = step(x, y, 0.5 * dt);
            let (sx, sy) = step(x + hx, y + hy, dt);
            x += sx;
            y += sy;
        }
        (x, y)
    }

    /// Inside the pocket's free area (the tool center may be here).
    pub fn in_pocket(&self, x: f64, y: f64) -> bool {
        let i = ((x - self.x0) / self.h).round() as i64;
        let j = ((y - self.y0) / self.h).round() as i64;
        i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.ny && self.label[j as usize * self.nx + i as usize] == 1 && self.clearance(x, y) >= -1e-6
    }

    /// Closed contours at `level` of T (`use_wall` false) or of G (the wall, level 0),
    /// by marching squares, each as its run of points (at least 8).
    pub fn contours(&self, level: f64, use_wall: bool) -> Vec<Vec<P>> {
        self.contours_min(level, use_wall, 8)
    }

    /// [`Plan::contours`], keeping loops of at least `min_points` points.
    pub fn contours_min(&self, level: f64, use_wall: bool, min_points: usize) -> Vec<Vec<P>> {
        let f = if use_wall { &self.g } else { &self.t };
        let segs = self.march(f, level, None);
        // Chain segments into loops by their shared end points.
        let key = |p: P| ((p.0 * 1e6).round() as i64, (p.1 * 1e6).round() as i64);
        let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (k, &(a, b)) in segs.iter().enumerate() {
            adj.entry(key(a)).or_default().push(k);
            adj.entry(key(b)).or_default().push(k);
        }
        let mut used = vec![false; segs.len()];
        let mut loops = Vec::new();
        for k in 0..segs.len() {
            if used[k] {
                continue;
            }
            used[k] = true;
            let mut lp = vec![segs[k].0, segs[k].1];
            let mut end = segs[k].1;
            loop {
                let Some(next) = adj[&key(end)].iter().copied().find(|&c| !used[c]) else { break };
                used[next] = true;
                end = if key(segs[next].0) == key(end) { segs[next].1 } else { segs[next].0 };
                lp.push(end);
            }
            if key(lp[0]) == key(lp[lp.len() - 1]) {
                lp.pop();
            }
            if lp.len() >= min_points {
                loops.push(lp);
            }
        }
        loops
    }

    /// Move a point onto the exact wall (clear = r + delta) by Newton steps on the
    /// clearance, which is a true distance: one step lands on it away from the medial axis.
    pub fn project_to_wall(&self, mut x: f64, mut y: f64) -> P {
        for _ in 0..4 {
            let g = self.clear.at(x, y) - self.limit;
            if g.abs() < 1e-5 {
                break;
            }
            let (gx, gy) = self.clear.grad(x, y);
            if gx == 0.0 && gy == 0.0 {
                break;
            }
            x -= g * gx;
            y -= g * gy;
        }
        (x, y)
    }

    pub fn clearance(&self, x: f64, y: f64) -> f64 {
        self.clear.at(x, y) - self.limit
    }

    /// True iff every tool center on the move a-b has clearance >= -tol (exact).
    pub fn clear_along(&self, ax: f64, ay: f64, bx: f64, by: f64, tol: f64) -> bool {
        self.clear.clear_along(ax, ay, bx, by, self.limit - tol)
    }
}
