//! One pocket at one floor level: the material a rough op removes, and the geometric
//! fields the assignment reads.
//!
//!   floor       the op's floor
//!   pocket      the connected free region at the floor (exact part clearance >= the
//!               allowance) that holds the op's own cut positions, inside the op's box,
//!               less the free floor outside the part's outline (the sheet skeleton)
//!   material    pocket cells where the stock left by every EARLIER op (swept tool disks)
//!               stands above the floor; its height H(x) is the local axial depth
//!   reach[k]    tool k can remove x: some tool-center position within r_k of x has
//!               clearance >= r_k + allowance (lattice distance transform to the center
//!               set, then an exact test along the clearance gradient for the cells the
//!               lattice misses), and the floor is within the tool's hangout below the
//!               stock top
//!   width[k]    the channel width tool k works in at x: the widest free disk (diameter
//!               w) whose center lies within max(w/2, 2 D_k) of x. A neck reads as its
//!               width; a room corner reads as the room.
//!   thick       local thickness: the widest free disk that contains x

use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;

use icam_post::{Move, MoveKind, ToolDef};
use icam_query::Clearance;
use rayon::prelude::*;

use crate::grid::Grid;
use crate::tools::{ToolSpec, HOLDER_MARGIN};

/// Rest thinner than this beyond the allowance counts as reached (left to the finish
/// pass, like a CAM system's minimum rest thickness); also absorbs lattice noise. Assumed.
pub const REST_TOL: f64 = 0.1;

pub struct Layer {
    pub name: String,
    pub op_tool: i32,
    pub op_d: f64,
    pub z_floor: f64,
    pub delta: f64,
    pub g: Grid,
    pub clear: Vec<f32>,
    pub top: Vec<f32>,
    pub hgt: Vec<f32>,
    pub thick: Vec<f32>,
    pub pocket: Vec<bool>,
    pub mat: Vec<bool>,
    pub w: Vec<Vec<f32>>,
    pub reach: Vec<Vec<bool>>,
    pub clipped: bool,
    pub notes: Vec<String>,
}

/// The inputs of a layer besides its clearances.
pub struct LayerInput<'a> {
    pub name: &'a str,
    pub op_tool: i32,
    pub op_d: f64,
    pub z_floor: f64,
    pub g: Grid,
    /// The op's cut positions at the floor (they pick the pocket).
    pub cuts: &'a [(f64, f64)],
    /// The stock top every earlier op leaves, per cell.
    pub top: Vec<f32>,
    pub stock_top: f64,
    pub tools: &'a [ToolSpec],
    pub delta: f64,
}

/// The clearance band the layer needs: the biggest tool's center at the allowance, plus 1.5 mm.
pub fn band_for(tools: &[ToolSpec], delta: f64) -> f64 {
    tools.iter().map(|t| t.r()).fold(f64::NEG_INFINITY, f64::max) + delta + 1.5
}

fn par_cells<T: Send>(g: &Grid, f: impl Fn(f64, f64) -> T + Sync) -> Vec<T> {
    (0..g.n()).into_par_iter().map(|k| f(g.x(k % g.nx), g.y(k / g.nx))).collect()
}

impl Layer {
    pub fn cell_area(&self) -> f64 {
        self.g.h * self.g.h
    }

    pub fn mat_area(&self) -> f64 {
        self.mat.iter().filter(|&&m| m).count() as f64 * self.cell_area()
    }

    pub fn volume(&self) -> f64 {
        let mut v = 0.0;
        for k in 0..self.mat.len() {
            if self.mat[k] {
                v += self.hgt[k] as f64;
            }
        }
        v * self.cell_area()
    }

    /// `clear`: the exact clearance at the floor, read to `band_for`; `foot`: the clearance
    /// just above the sheet bottom (the part's outline).
    pub fn make(inp: LayerInput, clear: &dyn Clearance, foot: &dyn Clearance) -> Layer {
        let g = inp.g;
        let (n, nx, delta, zf, tools) = (g.n(), g.nx, inp.delta, inp.z_floor, inp.tools);
        let mut notes = Vec::new();

        // Exact clearance within a band; beyond it, band + lattice distance to the band's edge.
        let band = band_for(tools, delta);
        let mut c: Vec<f32> = par_cells(&g, |x, y| clear.at(x, y) as f32);
        let near: Vec<bool> = c.iter().map(|&v| (v as f64) < band - 1e-4).collect();
        let d_near = g.edt(&near);
        for k in 0..n {
            if !near[k] {
                c[k] = (band + d_near[k] as f64) as f32;
            }
        }
        let top = inp.top;
        let hgt: Vec<f32> = top.iter().map(|&t| 0.0f64.max(t as f64 - zf) as f32).collect();

        // Pocket: free component(s) holding the op's own cut positions at the floor.
        let free: Vec<bool> = c.iter().map(|&v| v as f64 >= delta).collect();
        let (lab, _) = g.components(&free);
        let mut keep = HashSet::new();
        let (mut seeds, mut bad) = (0, 0);
        for &(x, y) in inp.cuts {
            let (ci, cj) = (g.i(x), g.j(y));
            if !g.contains(ci, cj) {
                continue;
            }
            let s = cj as usize * nx + ci as usize;
            seeds += 1;
            if lab[s] >= 0 && hgt[s] > 0.05 {
                keep.insert(lab[s]);
            } else {
                bad += 1;
            }
        }
        // Footprint: inside the part's outline (its shadow just above the sheet bottom,
        // holes filled). Free floor outside it is the sheet skeleton, freed by the profile
        // cuts, not pocketed.
        let outside: Vec<bool> = par_cells(&g, |x, y| foot.at(x, y) >= 0.0);
        let (olab, ocount) = g.components(&outside);
        let mut skeleton = vec![false; ocount];
        let mut mark = |k: usize| {
            if olab[k] >= 0 {
                skeleton[olab[k] as usize] = true;
            }
        };
        for i in 0..nx {
            mark(i);
            mark((g.ny - 1) * nx + i);
        }
        for j in 0..g.ny {
            mark(j * nx);
            mark(j * nx + nx - 1);
        }
        let (mut pocket, mut mat) = (vec![false; n], vec![false; n]);
        let mut skel = 0.0;
        for k in 0..n {
            let in_comp = lab[k] >= 0 && keep.contains(&lab[k]);
            let on_skel = olab[k] >= 0 && skeleton[olab[k] as usize];
            if in_comp && on_skel {
                skel += g.h * g.h;
            }
            pocket[k] = in_comp && !on_skel;
            mat[k] = pocket[k] && hgt[k] > 0.05;
        }
        if skel > 1.0 {
            notes.push(format!("the free floor runs out past the part's outline: {skel:.0} mm2 of nest skeleton (stock outside the part, freed by the profile cuts) left out"));
        }
        if bad as f64 > 0.5 * seeds as f64 {
            notes.push(format!("{bad} of {seeds} of the op's cut positions lie inside the design part's shadow (+ allowance) or on stock already cut: the op cuts where the part is solid"));
        }
        let clipped = (0..nx).any(|i| pocket[i] || pocket[(g.ny - 1) * nx + i]) || (0..g.ny).any(|j| pocket[j * nx] || pocket[j * nx + nx - 1]);
        if clipped {
            notes.push("the free region runs out of the op's box (grown by r + 10 mm); only the part inside is counted".into());
        }

        // Width fields from a ladder of disk sizes (one lattice distance transform each).
        let nt = tools.len();
        let (mut w, mut reach) = (vec![vec![0f32; n]; nt], vec![vec![false; n]; nt]);
        let mut thick = vec![0f32; n];
        let mut c_max = 0.0f64;
        for k in 0..n {
            if pocket[k] {
                c_max = c_max.max(c[k] as f64);
            }
        }
        let mut ladder = Vec::new();
        let mut lw = tools.iter().map(|t| t.d()).fold(f64::INFINITY, f64::min) * 0.9;
        while lw <= 2.0 * (c_max - delta) + 1e-9 {
            ladder.push(lw);
            lw *= 1.05;
        }
        for t in tools {
            if !ladder.iter().any(|&l| (l - t.d()).abs() < 1e-9) {
                ladder.push(t.d());
            }
        }
        ladder.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for &lw in &ladder {
            let seed: Vec<bool> = (0..n).map(|k| pocket[k] && c[k] as f64 - delta >= lw / 2.0 - 1e-9).collect();
            let d = g.edt(&seed);
            let wf = lw as f32;
            for k in 0..n {
                if !pocket[k] || d[k].is_infinite() {
                    continue;
                }
                let dk = d[k] as f64;
                if dk <= lw / 2.0 + 1e-6 {
                    thick[k] = wf;
                }
                for t in 0..nt {
                    if dk <= (lw / 2.0).max(2.0 * tools[t].d()) + 1e-6 {
                        w[t][k] = wf;
                    }
                }
            }
            for t in 0..nt {
                if (lw - tools[t].d()).abs() >= 1e-9 {
                    continue;
                }
                let deep_ok = inp.stock_top - zf <= tools[t].max_depth;
                let r = tools[t].r();
                let mat = &mat;
                let d = &d;
                let g = &g;
                reach[t] = (0..n)
                    .into_par_iter()
                    .map(|k| {
                        if !mat[k] || !deep_ok {
                            return false;
                        }
                        let dk = d[k] as f64;
                        if dk <= r + REST_TOL + 1e-6 {
                            return true;
                        }
                        if dk > r + 1.0 {
                            return false;
                        }
                        // Exact test: the center one radius up the clearance gradient.
                        let (x, y) = (g.x(k % g.nx), g.y(k / g.nx));
                        let (gx, gy) = clear.grad(x, y);
                        if gx == 0.0 && gy == 0.0 {
                            return false;
                        }
                        clear.at(x + r * gx, y + r * gy) >= r + delta - REST_TOL - 1e-6
                    })
                    .collect();
                if !deep_ok {
                    notes.push(format!(
                        "{}: the floor is {:.2} mm below the stock top, deeper than its hangout allows ({:.2} mm with {HOLDER_MARGIN:.1} mm holder clearance)",
                        tools[t].label,
                        inp.stock_top - zf,
                        tools[t].max_depth
                    ));
                }
            }
        }
        Layer { name: inp.name.into(), op_tool: inp.op_tool, op_d: inp.op_d, z_floor: zf, delta, g, clear: c, top, hgt, thick, pocket, mat, w, reach, clipped, notes }
    }
}

fn md(a: f64) -> f64 {
    let a = a % (2.0 * PI);
    if a < 0.0 { a + 2.0 * PI } else { a }
}

/// Calls hit(cell, z) for every cell a cutting move's tool disk covers.
pub fn sweep(m: &Move, tools: &BTreeMap<i32, ToolDef>, g: &Grid, mut hit: impl FnMut(usize, f64)) {
    if m.kind == MoveKind::Rapid {
        return;
    }
    let Some(t) = tools.get(&m.tool) else { return };
    let r = t.diameter / 2.0;
    let arc = matches!(m.kind, MoveKind::ArcCw | MoveKind::ArcCcw);
    let (bx0, by0, bx1, by1) = if arc {
        let c = m.center;
        let ra = ((m.from.x - c.x) * (m.from.x - c.x) + (m.from.y - c.y) * (m.from.y - c.y)).sqrt() + r;
        (c.x - ra, c.y - ra, c.x + ra, c.y + ra)
    } else if m.kind == MoveKind::Drill {
        (m.to.x - r, m.to.y - r, m.to.x + r, m.to.y + r)
    } else {
        (m.from.x.min(m.to.x) - r, m.from.y.min(m.to.y) - r, m.from.x.max(m.to.x) + r, m.from.y.max(m.to.y) + r)
    };
    let i0 = 0.max(((bx0 - g.x0) / g.h).floor() as i64);
    let i1 = (g.nx as i64 - 1).min(((bx1 - g.x0) / g.h).ceil() as i64);
    let j0 = 0.max(((by0 - g.y0) / g.h).floor() as i64);
    let j1 = (g.ny as i64 - 1).min(((by1 - g.y0) / g.h).ceil() as i64);
    if i0 > i1 || j0 > j1 {
        return;
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let (x, y) = (g.x(i as usize), g.y(j as usize));
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
                    let (ddx, ddy, ddz) = (m.to.x - m.from.x, m.to.y - m.from.y, m.to.z - m.from.z);
                    let full = (ddx * ddx + ddy * ddy + ddz * ddz).sqrt() < 1e-6;
                    let within = full || if m.kind == MoveKind::ArcCcw { md(tp - t0) <= md(t1 - t0) } else { md(t0 - tp) <= md(t0 - t1) };
                    let near_end = (x - m.from.x) * (x - m.from.x) + (y - m.from.y) * (y - m.from.y) <= r * r
                        || (x - m.to.x) * (x - m.to.x) + (y - m.to.y) * (y - m.to.y) <= r * r;
                    if !within && !near_end {
                        continue;
                    }
                    m.from.z.min(m.to.z)
                }
            };
            hit(j as usize * g.nx + i as usize, z);
        }
    }
}

/// The lowest Z any cutting move before `end` sweeps each cell to (the stock top where
/// none does).
pub fn top_before(moves: &[Move], tools: &BTreeMap<i32, ToolDef>, end: usize, g: &Grid, stock_top: f64) -> Vec<f32> {
    let mut top = vec![stock_top as f32; g.n()];
    for m in &moves[..end.min(moves.len())] {
        sweep(m, tools, g, |k, z| {
            if z < top[k] as f64 {
                top[k] = z as f32;
            }
        });
    }
    top
}
