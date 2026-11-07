//! What an ideal rougher leaves on the wall for a given allowance field delta (one value
//! per finisher sample, extended off the wall through the nearest wall foot).
//!
//!   rougher centers  C = { clear(x) - delta(x) >= r_R } inside the rough op's own
//!                    component: every center the rougher may take while keeping delta
//!                    from the part. Its interior is cleared, its boundary (the
//!                    rougher's last wall pass, the zero level of F = clear - delta - r_R)
//!                    is swept at radius r_R.
//!   stock            a point is stock iff it is outside the part (clear > 0), not in C,
//!                    farther than r_R from the boundary of C, and not cut by the
//!                    program before the rough op.
//!
//! With delta constant this is the constant-allowance rougher: the wall stock is delta on
//! straight and convex walls, and inside corners keep the rougher's own radius (what it
//! cannot reach) on top of it.

use rayon::prelude::*;

use crate::geom::{contour, simplify, Grid2, SegIndex};
use crate::pocket::Pocket;

pub struct RoughStock {
    f: Grid2,
    edge: SegIndex,
    pub delta: Vec<f64>,
}

impl RoughStock {
    pub fn new(pk: &Pocket, delta: Vec<f64>) -> RoughStock {
        let cg = &pk.cg;
        let mut f = cg.like();
        let hi = pk.rr + pk.d_max + 0.5;
        f.v.par_iter_mut().enumerate().for_each(|(s, v)| {
            let c = cg.v[s];
            *v = if !pk.comp[s] {
                -1.0
            } else if c > hi || pk.owner[s] < 0 {
                c - pk.rr - pk.d_max
            } else {
                c - delta[pk.owner[s] as usize] - pk.rr
            };
        });
        // The rougher's wall pass: F = 0, simplified (its segments are capsule axes, exact union).
        let mut segs = Vec::new();
        for (pts, closed) in contour(&f, 0.0) {
            let mut p = pts;
            if closed {
                p.push(p[0]);
            }
            let q = simplify(&p, 0.002);
            for k in 1..q.len() {
                segs.push((q[k - 1].0, q[k - 1].1, q[k].0, q[k].1, segs.len() as i32));
            }
        }
        let edge = SegIndex::new(&segs, pk.rr, 0.5);
        RoughStock { f, edge, delta }
    }

    pub fn uniform(pk: &Pocket, d: f64) -> RoughStock {
        RoughStock::new(pk, vec![d; pk.n()])
    }

    pub fn edge_segments(&self) -> usize {
        self.edge.count()
    }

    /// Stock the finisher meets (the part itself is not stock).
    pub fn stock(&self, pk: &Pocket, x: f64, y: f64) -> bool {
        if pk.cg.at(x, y) <= 5e-4 {
            return false;
        }
        if self.f.at(x, y) >= 0.0 {
            return false;
        }
        if self.edge.within(x, y, pk.rr) {
            return false;
        }
        pk.prior.material(x, y)
    }
}
