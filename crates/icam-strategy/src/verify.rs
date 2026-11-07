//! Checks a program with the exact swept-tool stock (icam_stock::ImplicitStock), which
//! shares no code with the generator's own stock. Run it on the program as posted, so
//! the post is checked too.
//!
//! After the op:
//!   gouge    design-part points near the pocket cut past by more than the shop
//!            tolerance (0.001 in)
//!   residue  pocket area the tool could reach, outside part + allowance, that is still
//!            material just above the cut level
//!   wall     median stock left on the pocket's walls (should be ~ allowance)
//!
//! The design points are an input: points on the part's boundary near the pocket, from
//! the field (or, for a test, from the polygons the clearance was given as), never the
//! vertices of a tessellated part, which the F-rep rules out.

use icam_post::V3;
use icam_stock::ImplicitStock;
use rayon::prelude::*;

use crate::path::P;
use crate::plan::Plan;

/// The shop tolerance, mm (0.001 in).
pub const TOL: f64 = 0.0254;

#[derive(Clone, Debug, Default)]
pub struct Check {
    pub part_points: usize,
    pub gouges: usize,
    pub max_gouge: f64,
    pub wall_median: f64,
    pub residue_area: f64,
    pub prior_residue_area: f64,
    /// The deepest gouges, deepest first (at most 6).
    pub worst: Vec<(V3, f64)>,
    /// Uncut sample points, row by row (at most 4000).
    pub spots: Vec<P>,
}

/// Gouge, wall stock and residue of the program behind `stock` on the pocket of `plan`
/// at `z_cut`, for a tool of radius `r` leaving `delta`; `pts` are the design points.
pub fn run(stock: &ImplicitStock, pts: &[V3], plan: &Plan, z_cut: f64, r: f64, delta: f64, stock_top: f64) -> Check {
    let mut c = Check::default();
    let dev: Vec<f64> = pts.par_iter().map(|&p| stock.deviation(p)).collect();
    c.part_points = pts.len();
    c.gouges = dev.iter().filter(|&&d| d < -TOL).count();
    c.max_gouge = if dev.is_empty() { 0.0 } else { (-dev.iter().copied().fold(f64::INFINITY, f64::min)).max(0.0) };
    let mut bad: Vec<usize> = (0..pts.len()).filter(|&i| dev[i] < -TOL).collect();
    bad.sort_by(|&a, &b| dev[a].partial_cmp(&dev[b]).unwrap_or(std::cmp::Ordering::Equal));
    c.worst = bad.into_iter().take(6).map(|i| (pts[i], dev[i])).collect();
    // Wall stock: part points just beside the pocket's free area, between floor and top.
    let mut wall = Vec::new();
    for (i, p) in pts.iter().enumerate() {
        if p.z < z_cut + 1.0 || p.z > stock_top - 1.0 {
            continue;
        }
        let g = plan.clearance(p.x, p.y); // clear - (r + delta) at the wall point: about -(r + delta)
        if g < -(r + delta) - 0.5 {
            continue;
        }
        if near_pocket(plan, p.x, p.y, r + delta + 1.5) {
            wall.push(dev[i]);
        }
    }
    wall.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    c.wall_median = if wall.is_empty() { f64::NAN } else { wall[wall.len() / 2] };
    c.residue_area = residue(stock, plan, z_cut, r, delta, Some(&mut c.spots));
    c
}

/// Reachable pocket area (mm2) still material just above the cut level. With the program
/// before the op it is the pocket's full area, a sanity check that the probe sees material.
pub fn residue(stock: &ImplicitStock, plan: &Plan, z_cut: f64, r: f64, delta: f64, mut spots: Option<&mut Vec<P>>) -> f64 {
    let reach = reachable(plan, r);
    let mut res = 0usize;
    for j in 0..plan.ny {
        for i in 0..plan.nx {
            let s = j * plan.nx + i;
            if !reach[s] {
                continue;
            }
            let (x, y) = (plan.x(i), plan.y(j));
            let clear = plan.clearance(x, y) + r + delta;
            if clear <= delta + 0.1 {
                continue;
            }
            if stock.is_material(V3::new(x, y, z_cut + 0.1)) {
                res += 1;
                if let Some(sp) = spots.as_deref_mut() {
                    if sp.len() < 4000 {
                        sp.push((x, y));
                    }
                }
            }
        }
    }
    res as f64 * plan.h * plan.h
}

fn near_pocket(plan: &Plan, x: f64, y: f64, d: f64) -> bool {
    let ri = (d / plan.h).ceil() as i64;
    let ci = ((x - plan.x0) / plan.h).round() as i64;
    let cj = ((y - plan.y0) / plan.h).round() as i64;
    for j in 0.max(cj - ri)..=(plan.ny as i64 - 1).min(cj + ri) {
        for i in 0.max(ci - ri)..=(plan.nx as i64 - 1).min(ci + ri) {
            if plan.label[j as usize * plan.nx + i as usize] == 1 {
                return true;
            }
        }
    }
    false
}

/// Grid nodes that belong to this pocket: within r of one of its allowed tool centers,
/// and not also within r of free space outside it (an open pocket's mouth belongs to
/// whatever cuts the outside).
pub fn reachable(plan: &Plan, r: f64) -> Vec<bool> {
    let mut own = dilate(plan, r, |s| plan.label[s] == 1);
    let other = dilate(plan, r, |s| plan.label[s] != 1 && plan.g[s] > 0.0);
    for (o, x) in own.iter_mut().zip(other) {
        *o &= !x;
    }
    own
}

fn dilate(plan: &Plan, r: f64, free: impl Fn(usize) -> bool) -> Vec<bool> {
    let (nx, ny) = (plan.nx, plan.ny);
    let mut reach = vec![false; nx * ny];
    let ri = (r / plan.h).floor() as i64;
    let mut offs = Vec::new();
    for dj in -ri..=ri {
        for di in -ri..=ri {
            if (di * di + dj * dj) as f64 * plan.h * plan.h <= (r - 0.05) * (r - 0.05) {
                offs.push((di, dj));
            }
        }
    }
    for s in 0..reach.len() {
        if !free(s) {
            continue;
        }
        let (i, j) = (s % nx, s / nx);
        // Interior nodes of the free set reach nothing their neighbors do not.
        let edge = i == 0 || j == 0 || i == nx - 1 || j == ny - 1 || !free(s - 1) || !free(s + 1) || !free(s - nx) || !free(s + nx);
        if !edge {
            reach[s] = true;
            continue;
        }
        for &(di, dj) in &offs {
            let (ii, jj) = (i as i64 + di, j as i64 + dj);
            if ii >= 0 && jj >= 0 && ii < nx as i64 && jj < ny as i64 {
                reach[jj as usize * nx + ii as usize] = true;
            }
        }
    }
    reach
}
