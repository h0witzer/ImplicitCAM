//! A bucket grid over one face's boundary curves, so the signed distance to a section
//! with thousands of curves (a sheet of patterned features) reads only the curves
//! near the point instead of all of them.
//!
//! Distance: the cells around the point are visited ring by ring, each curve met is
//! asked for its closest point within the best distance so far, and the search stops
//! when the next ring's cells are all farther than the best. Sign: from the nearest
//! feature, as the edges' pseudo-normals are built for (a point is inside exactly when
//! it lies behind the outward normal at its closest boundary point, or behind the
//! vertex pseudo-normal when that point is a vertex). The result is the same record
//! `query::face_sd` returns.

use crate::geom::{Closest, Near, V2};
use crate::part::Part;
use crate::query::FaceDist;

#[derive(Clone, Debug)]
pub struct CurveGrid {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    /// Indices into `Part::edges`, per cell.
    cells: Vec<Vec<u32>>,
}

impl CurveGrid {
    /// The grid over face `f`'s curves: about two curves per cell, at most 4 M cells.
    pub fn new(part: &Part, f: usize) -> CurveGrid {
        let boxes: Vec<(u32, V2, V2)> = part.face_edges(f).map(|(i, e)| (i as u32, e.curve.bbox().0, e.curve.bbox().1)).collect();
        let (mut lo, mut hi) = (V2::new(f64::MAX, f64::MAX), V2::new(f64::MIN, f64::MIN));
        for &(_, a, b) in &boxes {
            lo = V2::new(lo.x.min(a.x), lo.y.min(a.y));
            hi = V2::new(hi.x.max(b.x), hi.y.max(b.y));
        }
        let (w, h) = ((hi.x - lo.x).max(1e-6), (hi.y - lo.y).max(1e-6));
        let n = boxes.len().max(1) as f64;
        let cell = (w * h / (0.5 * n)).sqrt().max((w * h / 4e6).sqrt()).max(1e-3);
        let (nx, ny) = ((w / cell).ceil() as usize + 1, (h / cell).ceil() as usize + 1);
        let mut cells = vec![Vec::new(); nx * ny];
        for &(i, a, b) in &boxes {
            let i0 = (((a.x - lo.x) / cell).floor() as usize).min(nx - 1);
            let i1 = (((b.x - lo.x) / cell).floor() as usize).min(nx - 1);
            let j0 = (((a.y - lo.y) / cell).floor() as usize).min(ny - 1);
            let j1 = (((b.y - lo.y) / cell).floor() as usize).min(ny - 1);
            for j in j0..=j1 {
                for k in i0..=i1 {
                    cells[j * nx + k].push(i);
                }
            }
        }
        CurveGrid { x0: lo.x, y0: lo.y, cell, nx, ny, cells }
    }

    /// Distance from p to cell (i, j)'s rectangle.
    fn cell_dist(&self, p: V2, i: usize, j: usize) -> f64 {
        let (x0, y0) = (self.x0 + i as f64 * self.cell, self.y0 + j as f64 * self.cell);
        let dx = (x0 - p.x).max(0.0).max(p.x - (x0 + self.cell));
        let dy = (y0 - p.y).max(0.0).max(p.y - (y0 + self.cell));
        dx.hypot(dy)
    }

    /// Signed distance from p to face `f` of `part` (the face the grid was built on).
    pub fn sd(&self, part: &Part, p: V2) -> FaceDist {
        let ci = (((p.x - self.x0) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let cj = (((p.y - self.y0) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        let mut best: Option<(usize, Closest)> = None;
        let mut seen: Vec<u32> = Vec::new();
        let rings = self.nx.max(self.ny);
        for r in 0..=rings {
            let cut = best.as_ref().map_or(f64::INFINITY, |(_, b)| b.dist);
            let (i0, i1) = (ci.saturating_sub(r), (ci + r).min(self.nx - 1));
            let (j0, j1) = (cj.saturating_sub(r), (cj + r).min(self.ny - 1));
            let mut nearest = f64::INFINITY;
            for j in j0..=j1 {
                for i in i0..=i1 {
                    // Only the ring itself: cells r away in x or in y.
                    if i.abs_diff(ci) != r && j.abs_diff(cj) != r {
                        continue;
                    }
                    let d = self.cell_dist(p, i, j);
                    nearest = nearest.min(d);
                    if d > cut {
                        continue;
                    }
                    for &e in &self.cells[j * self.nx + i] {
                        if seen.contains(&e) {
                            continue;
                        }
                        seen.push(e);
                        let cut = best.as_ref().map_or(f64::INFINITY, |(_, b)| b.dist);
                        if let Some(cl) = part.edges[e as usize].curve.closest_within(p, cut) {
                            if best.as_ref().is_none_or(|(_, b)| cl.dist < b.dist) {
                                best = Some((e as usize, cl));
                            }
                        }
                    }
                }
            }
            let cut = best.as_ref().map_or(f64::INFINITY, |(_, b)| b.dist);
            if nearest > cut || (i0 == 0 && j0 == 0 && i1 == self.nx - 1 && j1 == self.ny - 1) {
                break;
            }
        }
        let (i, cl) = best.expect("face without edges");
        let e = &part.edges[i];
        let n = match cl.near {
            Near::Interior => cl.outward(),
            Near::Start => e.pn_start,
            Near::End => e.pn_end,
        };
        if cl.dist > 0.0 {
            let sign = if (p - cl.q).dot(n) < 0.0 { -1.0 } else { 1.0 };
            let g = (p - cl.q) * (1.0 / cl.dist);
            let h = cl.dist_hessian(p);
            FaceDist { sd: sign * cl.dist, q: cl.q, edge: i as u32, grad: g * sign, hess: [h[0] * sign, h[1] * sign, h[2] * sign], t: cl.t }
        } else {
            FaceDist { sd: 0.0, q: cl.q, edge: i as u32, grad: n, hess: [0.0; 3], t: cl.t }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Curve;
    use crate::query::face_sd;

    /// A 300 x 200 plate with a 12 x 8 pattern of round holes and square holes.
    fn plate() -> Part {
        let sq = |x0: f64, y0: f64, x1: f64, y1: f64, ccw: bool| {
            let p = [V2::new(x0, y0), V2::new(x1, y0), V2::new(x1, y1), V2::new(x0, y1)];
            let mut l: Vec<Curve> = (0..4).map(|i| Curve::Line { a: p[i], b: p[(i + 1) % 4] }).collect();
            if !ccw {
                l = l.into_iter().rev().map(|c| if let Curve::Line { a, b } = c { Curve::Line { a: b, b: a } } else { c }).collect();
            }
            l
        };
        let mut loops = vec![sq(0.0, 0.0, 300.0, 200.0, true)];
        for i in 0..12 {
            for j in 0..8 {
                let (x, y) = (12.0 + 24.0 * i as f64, 12.0 + 24.0 * j as f64);
                if (i + j) % 2 == 0 {
                    let c = V2::new(x, y);
                    loops.push(vec![Curve::Arc { c, r: 6.0, a: c + V2::new(6.0, 0.0), sweep: -2.0 * std::f64::consts::PI }]);
                } else {
                    loops.push(sq(x - 5.0, y - 5.0, x + 5.0, y + 5.0, false));
                }
            }
        }
        let mut p = Part::default();
        p.add_face(0, 10.0, loops, None, 1e-9).unwrap();
        p
    }

    #[test]
    fn the_grid_reads_the_same_signed_distance_as_every_curve() {
        let p = plate();
        let g = CurveGrid::new(&p, 0);
        let mut worst: f64 = 0.0;
        let mut flips = 0;
        // Points over the plate and past its edges, on an irrational lattice.
        for k in 0..20_000 {
            let (u, v) = ((k as f64 * 0.618_033_988_75).fract(), (k as f64 * 0.754_877_666_25).fract());
            let q = V2::new(-40.0 + 380.0 * u, -40.0 + 280.0 * v);
            let (a, b) = (face_sd(&p, 0, q), g.sd(&p, q));
            worst = worst.max((a.sd.abs() - b.sd.abs()).abs());
            if a.sd.signum() != b.sd.signum() && a.sd.abs() > 1e-9 {
                flips += 1;
            }
        }
        assert!(worst < 1e-12, "distance differs by {worst:e}");
        assert_eq!(flips, 0, "sign differs at {flips} points");
    }
}
