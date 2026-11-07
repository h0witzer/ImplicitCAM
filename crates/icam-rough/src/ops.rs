//! The operations, found from the field: no other program says where the pockets are.
//!
//!   slabs    every section level of the compiled part below the stock top is a floor,
//!            cut at floor + allowance, top down (the sheet's bottom too: a pocket cut
//!            through the sheet leaves a skin of the allowance). With a stepdown, a gap
//!            between floors that is deeper than it gets equal intermediate slabs.
//!   pockets  at each slab, the clearance field (icam_query::SliceClearance) on a grid
//!            over the part, with every face above the cut's floor standing as a wall
//!            (a face less than the allowance under the tool is one the tool must keep
//!            off, not a floor it may rest on): every 4-connected region where the tool center fits
//!            (clearance > r + allowance) and that does not reach the grid's edge (that
//!            is the open sheet around the part) is a pocket. Regions joined only by
//!            necks narrower than the tool (clearance > 0 all the way) are one op, as the
//!            plan treats them.
//!   design   the points the checker measures: the part's walls (the clearance's zero
//!            set, Newton-projected, every 1 mm) at heights from just below the cut to the
//!            top, and the floor under the pocket, each kept only where the field says
//!            it is on the part's boundary.
//!
//! A stepped pocket is found twice, as it should be: at the upper floor's slab the deeper
//! step is still under stock and is cleared with the rest, and at the deeper floor's slab
//! only the step is free. Without a stepdown, a pocket with none of its slab's floor in
//! the tool's reach (a hole going on down past another pocket's floor) is left to the
//! slab under it, which cuts it in one go.

use icam_core::frep::Frep;
use icam_post::V3;
use icam_query::{Box2, Clearance, SliceClearance};
use icam_strategy::path::resample;
use icam_strategy::Plan;
use rayon::prelude::*;

/// One operation: a pocket at a slab.
#[derive(Clone, Debug)]
pub struct OpSpec {
    pub name: String,
    pub z_cut: f64,
    /// The height the walls are taken at: z_cut less the allowance. Every face above it
    /// is a wall, so the tool keeps the allowance off a floor just under the cut too.
    pub z_wall: f64,
    /// The floor this slab finishes on (None for an intermediate slab).
    pub floor: Option<f64>,
    /// The plan's grid box: the tool centers' extent grown by the tool radius and 8 mm.
    pub region: Box2,
    /// A point where the tool center fits, which picks the pocket out of the plan grid.
    pub hint: (f64, f64),
    /// Where its tool centers go, on the grid the pockets are found on.
    pub centers: std::sync::Arc<Centers>,
}

/// Cells of the grid the pockets are found on (pitch `pitch` from the part's corner, the
/// same at every slab) where an op's tool centers go: by row, over their box.
#[derive(Clone, Debug)]
pub struct Centers {
    pub pitch: f64,
    pub i0: usize,
    pub j0: usize,
    pub w: usize,
    pub h: usize,
    pub on: Vec<bool>,
}

impl Centers {
    /// The cells within `reach` of a center, as a square (which only ever adds cells), over
    /// the box grown by it: (i0, j0, w, h, cells).
    fn grown(&self, reach: f64) -> (usize, usize, usize, usize, Vec<bool>) {
        let k = (reach / self.pitch).ceil() as usize;
        let (i0, j0) = (self.i0.saturating_sub(k), self.j0.saturating_sub(k));
        let (w, h) = (self.i0 + self.w + k - i0, self.j0 + self.h + k - j0);
        let mut src = vec![false; w * h];
        for j in 0..self.h {
            for i in 0..self.w {
                src[(self.j0 + j - j0) * w + (self.i0 + i - i0)] = self.on[j * self.w + i];
            }
        }
        let grow = |src: &[bool], n: usize, m: usize, at: &dyn Fn(usize, usize) -> usize| -> Vec<bool> {
            let mut out = vec![false; src.len()];
            let mut count = vec![0usize; n + 1];
            for b in 0..m {
                for a in 0..n {
                    count[a + 1] = count[a] + usize::from(src[at(a, b)]);
                }
                for a in 0..n {
                    out[at(a, b)] = count[(a + k + 1).min(n)] > count[a.saturating_sub(k)];
                }
            }
            out
        };
        let rows = grow(&src, w, h, &|a, b| b * w + a);
        (i0, j0, w, h, grow(&rows, h, w, &|a, b| a * w + b))
    }
}

/// For each op, the earlier ops (at a higher slab) it waits for: those whose tool reaches
/// where its tool does (centers within 2r). A pocket at a lower slab is cut after the
/// stock above it, and only after that.
pub fn waits(ops: &[OpSpec], r: f64) -> Vec<Vec<usize>> {
    let grown: Vec<(usize, usize, usize, usize, Vec<bool>)> = ops.par_iter().map(|o| o.centers.grown(2.0 * r)).collect();
    (0..ops.len())
        .into_par_iter()
        .map(|b| {
            let cb = &ops[b].centers;
            (0..ops.len())
                .filter(|&a| ops[a].z_cut > ops[b].z_cut + 1e-9)
                .filter(|&a| {
                    let (i0, j0, w, h, on) = &grown[a];
                    if cb.i0 + cb.w <= *i0 || i0 + w <= cb.i0 || cb.j0 + cb.h <= *j0 || j0 + h <= cb.j0 {
                        return false;
                    }
                    // Over the boxes' overlap only.
                    let (gi0, gi1) = (cb.i0.max(*i0), (cb.i0 + cb.w).min(i0 + w));
                    let (gj0, gj1) = (cb.j0.max(*j0), (cb.j0 + cb.h).min(j0 + h));
                    (gj0..gj1).any(|gj| (gi0..gi1).any(|gi| cb.on[(gj - cb.j0) * cb.w + (gi - cb.i0)] && on[(gj - j0) * w + (gi - i0)]))
                })
                .collect()
        })
        .collect()
}

/// The slabs, top down: (cut height, the floor it finishes on).
pub fn slabs(levels: &[f64], z_bot: f64, stock_top: f64, allowance: f64, stepdown: Option<f64>) -> Vec<(f64, Option<f64>)> {
    let mut floors: Vec<f64> = levels.iter().copied().chain([z_bot]).filter(|&z| z < stock_top - 1e-6).collect();
    floors.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    floors.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let mut out = Vec::new();
    let mut prev = stock_top;
    for f in floors {
        let c = f + allowance;
        if c >= prev - 1e-6 {
            continue;
        }
        if let Some(sd) = stepdown.filter(|&s| s > 0.0) {
            let n = ((prev - c) / sd).ceil() as usize;
            for k in 1..n {
                out.push((prev - k as f64 * (prev - c) / n as f64, None));
            }
        }
        out.push((c, Some(f)));
        prev = c;
    }
    out
}

/// The pockets at one slab whose walls are taken at `z_wall` (see the module notes): their
/// grid boxes, a point in each, and their tool centers, ordered by that point (y, then x).
pub fn pockets(frep: &Frep, part: Box2, z_wall: f64, r: f64, limit: f64, pitch: f64) -> Vec<(Box2, (f64, f64), Centers)> {
    let clear = SliceClearance::new(frep, z_wall, 1e-3);
    if clear.column_count() == 0 {
        return Vec::new();
    }
    let bx = part.grow(limit + 2.0 * pitch);
    let nx = ((bx.x1 - bx.x0) / pitch).ceil() as usize + 1;
    let ny = ((bx.y1 - bx.y0) / pitch).ceil() as usize + 1;
    let mut g = vec![0.0; nx * ny];
    g.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        for (i, v) in row.iter_mut().enumerate() {
            *v = clear.at(bx.x0 + i as f64 * pitch, bx.y0 + j as f64 * pitch);
        }
    });
    let label = |inside: &dyn Fn(usize) -> bool| -> (Vec<usize>, usize) {
        let mut comp = vec![0usize; nx * ny];
        let mut n = 0;
        let mut stack = Vec::new();
        for s in 0..nx * ny {
            if !inside(s) || comp[s] != 0 {
                continue;
            }
            n += 1;
            comp[s] = n;
            stack.push(s);
            while let Some(c) = stack.pop() {
                let (i, j) = (c % nx, c / nx);
                for m in [(i > 0).then(|| c - 1), (i + 1 < nx).then(|| c + 1), (j > 0).then(|| c - nx), (j + 1 < ny).then(|| c + nx)].into_iter().flatten() {
                    if inside(m) && comp[m] == 0 {
                        comp[m] = n;
                        stack.push(m);
                    }
                }
            }
        }
        (comp, n)
    };
    let (free, nfree) = label(&|s| g[s] > limit);
    let (open, _) = label(&|s| g[s] > 0.0);
    let mut edge = vec![false; nfree + 1];
    for s in 0..nx * ny {
        let (i, j) = (s % nx, s / nx);
        if i == 0 || j == 0 || i == nx - 1 || j == ny - 1 {
            edge[free[s]] = true;
        }
    }
    // Group the free regions by the open area that holds them.
    let mut groups: Vec<(usize, Box2, f64, usize, Vec<usize>)> = Vec::new(); // open id, tool-center box, best clearance, its node, the nodes
    let mut group_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for s in 0..nx * ny {
        let c = free[s];
        if c == 0 || edge[c] {
            continue;
        }
        let (x, y) = (bx.x0 + (s % nx) as f64 * pitch, bx.y0 + (s / nx) as f64 * pitch);
        match group_of.get(&open[s]).map(|&k| &mut groups[k]) {
            Some(e) => {
                e.1 = Box2::new(e.1.x0.min(x), e.1.y0.min(y), e.1.x1.max(x), e.1.y1.max(y));
                if g[s] > e.2 {
                    e.2 = g[s];
                    e.3 = s;
                }
                e.4.push(s);
            }
            None => {
                group_of.insert(open[s], groups.len());
                groups.push((open[s], Box2::new(x, y, x, y), g[s], s, vec![s]));
            }
        }
    }
    let mut out: Vec<(Box2, (f64, f64), Centers)> = groups
        .into_iter()
        .map(|(_, b, _, s, nodes)| {
            let (i0, i1) = nodes.iter().fold((usize::MAX, 0), |(lo, hi), &t| (lo.min(t % nx), hi.max(t % nx)));
            let (j0, j1) = (nodes[0] / nx, nodes[nodes.len() - 1] / nx);
            let (w, h) = (i1 - i0 + 1, j1 - j0 + 1);
            let mut on = vec![false; w * h];
            for &t in &nodes {
                on[(t / nx - j0) * w + (t % nx - i0)] = true;
            }
            (b.grow(r + 8.0), (bx.x0 + (s % nx) as f64 * pitch, bx.y0 + (s / nx) as f64 * pitch), Centers { pitch, i0, j0, w, h, on })
        })
        .collect();
    out.sort_by(|a, b| (a.1 .1, a.1 .0).partial_cmp(&(b.1 .1, b.1 .0)).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Every operation of the part, top down (see the module notes).
#[allow(clippy::too_many_arguments)]
pub fn find(frep: &Frep, part: Box2, stock_top: f64, r: f64, allowance: f64, stepdown: Option<f64>, pitch: f64) -> Vec<OpSpec> {
    let mut ops = Vec::new();
    for (z_cut, floor) in slabs(&frep.levels, frep.z_bot, stock_top, allowance, stepdown) {
        let z_wall = z_cut - allowance;
        for (k, (region, hint, centers)) in pockets(frep, part, z_wall, r, r + allowance, pitch).into_iter().enumerate() {
            ops.push(OpSpec { name: format!("Z{z_cut:.3} POCKET {}", k + 1), z_cut, z_wall, floor, region, hint, centers: std::sync::Arc::new(centers) });
        }
    }
    if stepdown.is_none() {
        let keep: Vec<bool> = ops.par_iter().map(|s| ends_on_floor(frep, s, r, allowance)).collect();
        let mut keep = keep.into_iter();
        ops.retain(|_| keep.next().unwrap_or(true));
    }
    ops
}

/// Whether the pocket of an op ends on its slab's floor: the floor is the sheet's bottom
/// (a cut through), or the tool reaches part just under the floor from somewhere its
/// center fits in the pocket, past the allowance off the walls (a shelf narrower than the
/// tool counts). A pocket that goes on below (it is cleared deeper by a later slab) does
/// not; its walls end at its own floor.
pub fn ends_on_floor(frep: &Frep, spec: &OpSpec, r: f64, allowance: f64) -> bool {
    let Some(f) = spec.floor else { return false };
    if (f - frep.z_bot).abs() < 1e-6 {
        return true;
    }
    let clear = SliceClearance::new(frep, spec.z_wall, 1e-3);
    let plan = Plan::new(&clear, r + allowance, spec.region, 1.0, &[spec.hint]);
    let centers: Vec<usize> = (0..plan.nx * plan.ny).filter(|&s| plan.label[s] == 1).collect();
    floor_in_reach(frep, &plan, &centers, f, r)
}

/// Whether a tool of radius `r` at the plan nodes `centers` reaches part just under the
/// floor `f` past the allowance off the walls (nodes where the plan's clearance leaves
/// `r` to spare: g >= -r). The reach is the centers grown by r, as a square (which only
/// ever adds nodes), over their own box.
pub fn floor_in_reach(frep: &Frep, plan: &Plan, centers: &[usize], f: f64, r: f64) -> bool {
    if centers.is_empty() {
        return false;
    }
    let (nx, h) = (plan.nx, plan.h);
    let k = (r / h).ceil() as usize;
    let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
    for &s in centers {
        let (i, j) = (s % nx, s / nx);
        (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
    }
    let (i0, j0) = (i0.saturating_sub(k), j0.saturating_sub(k));
    let (i1, j1) = ((i1 + k).min(nx - 1), (j1 + k).min(plan.ny - 1));
    let (w, t) = (i1 - i0 + 1, j1 - j0 + 1);
    let mut mark = vec![false; w * t];
    for &s in centers {
        mark[(s / nx - j0) * w + (s % nx - i0)] = true;
    }
    // Grown along rows, then along columns, by prefix counts.
    let grow = |src: &[bool], n: usize, m: usize, at: &dyn Fn(usize, usize) -> usize| -> Vec<bool> {
        let mut out = vec![false; src.len()];
        let mut count = vec![0usize; n + 1];
        for b in 0..m {
            for a in 0..n {
                count[a + 1] = count[a] + usize::from(src[at(a, b)]);
            }
            for a in 0..n {
                out[at(a, b)] = count[(a + k + 1).min(n)] > count[a.saturating_sub(k)];
            }
        }
        out
    };
    let rows = grow(&mark, w, t, &|a, b| b * w + a);
    let reach = grow(&rows, t, w, &|a, b| a * w + b);
    (0..w * t).into_par_iter().any(|q| {
        let (i, j) = (i0 + q % w, j0 + q / w);
        reach[q] && plan.g[j * nx + i] >= -r && frep.eval([plan.x(i), plan.y(j), f - 0.05]).value < 0.0
    })
}

/// The plan grid of an op's pocket for a tool of radius `r` leaving `allowance`. Without
/// a stepdown, only the tool centers that reach stock this slab has to take stay in it:
/// stock over the slab's floor, or stock the slab under it cannot reach (its walls stand
/// higher). The rest of the pocket goes on below, and the slab under it cuts it in one go.
/// Stock is taken past the allowance off the walls (plan nodes with g >= -r), within r of
/// a center (by the exact distance transform of the plan grid).
pub fn plan<'a>(frep: &Frep, clear: &'a SliceClearance, spec: &OpSpec, r: f64, allowance: f64, pitch: f64, stepdown: Option<f64>) -> Plan<'a> {
    let mut plan = Plan::new(clear, r + allowance, spec.region, pitch, &[spec.hint]);
    let Some(f) = spec.floor.filter(|&f| stepdown.is_none() && f > frep.z_bot + 1e-6) else { return plan };
    let (nx, n) = (plan.nx, plan.nx * plan.ny);
    let grid = icam_assign::grid::Grid { x0: plan.x0, y0: plan.y0, h: plan.h, nx: plan.nx, ny: plan.ny };
    // The slab under this one has its walls at the next level down.
    let below = frep.levels.iter().copied().chain([frep.z_bot]).filter(|&z| z < f - 1e-6).fold(f64::NEG_INFINITY, f64::max);
    let clear_below = SliceClearance::new(frep, below, 1e-3);
    let centers_below: Vec<bool> = (0..n).into_par_iter().map(|s| plan.g[s] >= 0.0 && clear_below.at(plan.x(s % nx), plan.y(s / nx)) >= r + allowance).collect();
    let covered = grid.edt(&centers_below);
    let need: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|s| plan.g[s] >= -r && (f64::from(covered[s]) > r || frep.eval([plan.x(s % nx), plan.y(s / nx), f - 0.05]).value < 0.0))
        .collect();
    let reach: Vec<bool> = grid.edt(&need).into_iter().map(|d| f64::from(d) <= r).collect();
    plan.restrict(&reach);
    plan
}

/// Design points for the checker on one op's pocket (see the module notes).
pub fn design_points(frep: &Frep, plan: &Plan, clear: &dyn Clearance, limit: f64, z_cut: f64, floor: Option<f64>, stock_top: f64) -> Vec<V3> {
    let on_part = |x: f64, y: f64, z: f64| frep.eval([x, y, z]).value.abs() <= 1e-3;
    let mut heights = vec![z_cut - 0.25];
    let mut z = z_cut + 0.5;
    while z <= stock_top - 0.5 {
        heights.push(z);
        z += 2.0;
    }
    let mut pts = Vec::new();
    for lp in plan.contours(-limit, true) {
        // Onto the wall itself: the clearance is a true distance, so Newton lands on it.
        let wall: Vec<(f64, f64)> = lp
            .iter()
            .map(|&(mut x, mut y)| {
                for _ in 0..4 {
                    let c = clear.at(x, y);
                    if c.abs() < 1e-6 {
                        break;
                    }
                    let (gx, gy) = clear.grad(x, y);
                    if gx == 0.0 && gy == 0.0 {
                        break;
                    }
                    x -= c * gx;
                    y -= c * gy;
                }
                (x, y)
            })
            .collect();
        for (x, y) in resample(&wall, 1.0) {
            for &z in &heights {
                if on_part(x, y, z) {
                    pts.push(V3::new(x, y, z));
                }
            }
        }
    }
    if let Some(f) = floor {
        for j in (0..plan.ny).step_by(4) {
            for i in (0..plan.nx).step_by(4) {
                let (x, y) = (plan.x(i), plan.y(j));
                if plan.label[j * plan.nx + i] == 1 && on_part(x, y, f) {
                    pts.push(V3::new(x, y, f));
                }
            }
        }
    }
    pts
}
