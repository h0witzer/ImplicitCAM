//! Continuity - the premise of field control, as a measure. A plan that is a solved field
//! should move the path by about as much as the input moved; one that reorganizes under
//! a tiny change is chaotic, however good its time.
//!
//! [`compare`] holds a base path against a perturbed one: both resampled every 1 mm in
//! cutting order, a point of one path counts as moved when no point of the other path
//! within +/- `window` of its order position (as a share of path length) is within
//! `limit`. Symmetric. The window lets a pass appear or vanish (a local change) but
//! catches a reorganized order. (Resampling to a fixed point count instead would put
//! samples 16-40 mm apart on long paths and report nearly identical paths as moved.)
//! The gate: at most 5% of the path moves more than 1 mm, on every op and perturbation.

use rayon::prelude::*;

use crate::path::P;

/// "Moved" along the alignment, mm.
pub const LIMIT: f64 = 1.0;
/// Share of the path allowed to move more than `LIMIT`.
pub const SHARE_LIMIT: f64 = 0.05;
/// Order window, as a share of path length.
pub const WINDOW: f64 = 0.10;

/// (share moved, 95th percentile gap, largest gap) between two paths.
pub fn compare(a: &[P], b: &[P], limit: f64, window: f64) -> (f64, f64, f64) {
    let (ra, rb) = (resample(a, 1.0), resample(b, 1.0));
    let mut d = gaps(&ra, &rb, window);
    d.extend(gaps(&rb, &ra, window));
    d.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    if d.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let moved = d.iter().filter(|&&x| x > limit).count();
    (moved as f64 / d.len() as f64, d[(0.95 * (d.len() - 1) as f64) as usize], d[d.len() - 1])
}

fn gaps(a: &[P], b: &[P], window: f64) -> Vec<f64> {
    let (na, nb) = (a.len(), b.len());
    (0..na)
        .into_par_iter()
        .map(|i| {
            let f = if na > 1 { i as f64 / (na - 1) as f64 } else { 0.0 };
            let j0 = 0.max(((f - window) * (nb as f64 - 1.0)).floor() as i64) as usize;
            let j1 = (nb as i64 - 1).min(((f + window) * (nb as f64 - 1.0)).ceil() as i64);
            let mut best = f64::INFINITY;
            let mut j = j0 as i64;
            while j <= j1 {
                let (dx, dy) = (a[i].0 - b[j as usize].0, a[i].1 - b[j as usize].1);
                let dd = dx * dx + dy * dy;
                if dd < best {
                    best = dd;
                }
                j += 1;
            }
            best.sqrt()
        })
        .collect()
}

fn resample(p: &[P], step: f64) -> Vec<P> {
    if p.len() < 2 {
        return p.to_vec();
    }
    let mut c = vec![0.0; p.len()];
    for i in 1..p.len() {
        c[i] = c[i - 1] + ((p[i].0 - p[i - 1].0) * (p[i].0 - p[i - 1].0) + (p[i].1 - p[i - 1].1) * (p[i].1 - p[i - 1].1)).sqrt();
    }
    let total = c[c.len() - 1];
    let n = 2.max((total / step) as usize + 1);
    let mut o = Vec::with_capacity(n);
    let mut j = 0;
    for k in 0..n {
        let s = total * k as f64 / (n - 1) as f64;
        while j < p.len() - 2 && c[j + 1] < s {
            j += 1;
        }
        let seg = c[j + 1] - c[j];
        let u = if seg < 1e-12 { 0.0 } else { (s - c[j]) / seg };
        o.push((p[j].0 + u * (p[j + 1].0 - p[j].0), p[j].1 + u * (p[j + 1].1 - p[j].1)));
    }
    o
}
