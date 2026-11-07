//! A still of the fields: the stock's relief lit from the upper left, colored by what
//! the program left against the part. The viewer colors the same way.
//!
//!   green   on size (within the shop tolerance)
//!   blue    material left over the part, light for a thin skin to deep for full stock
//!   red     gouged, orange for a skim to deep red for 2.5 mm and more
//!   gray    no part under it: light where the stock stands, dark where it is cut away
//!
//! A pixel covers `per` x `per` nodes. A gouge anywhere in it shows (a thin gouge
//! would vanish in a mean); otherwise the commonest class wins.

use crate::part::{Class, TOL};
use crate::Grid;

/// RGB rows, top row first (+y up), `per` nodes to a pixel. `left` is what the program
/// left at every node (`part::left_field`), or None to show only the stock.
pub fn picture(grid: &Grid, h: &[f32], left: Option<&[f32]>, z_top: f32, per: usize) -> (u32, u32, Vec<u8>) {
    let per = per.max(1);
    let (w, ht) = (grid.nx / per, grid.ny / per);
    // Mean stock height per pixel, for the relief.
    let mut mean = vec![0f32; w * ht];
    for (p, m) in mean.iter_mut().enumerate() {
        let (pi, pj) = (p % w, p / w);
        let mut s = 0.0;
        for j in pj * per..(pj + 1) * per {
            for i in pi * per..(pi + 1) * per {
                s += h[j * grid.nx + i];
            }
        }
        *m = s / (per * per) as f32;
    }
    let size = (per as f64 * grid.step) as f32;
    let light = {
        let (x, y, z) = (-1.0f32, 1.0f32, 1.4f32);
        let n = (x * x + y * y + z * z).sqrt();
        [x / n, y / n, z / n]
    };
    let mut rgb = vec![0u8; w * ht * 3];
    for pj in 0..ht {
        for pi in 0..w {
            let at = |i: usize, j: usize| mean[j.min(ht - 1) * w + i.min(w - 1)];
            let gx = (at(pi + 1, pj) - at(pi.saturating_sub(1), pj)) / (2.0 * size);
            let gy = (at(pi, pj + 1) - at(pi, pj.saturating_sub(1))) / (2.0 * size);
            let n = (gx * gx + gy * gy + 1.0).sqrt();
            let lit = ((-gx * light[0] - gy * light[1] + light[2]) / n).max(0.0);
            let shade = 0.45 + 0.55 * lit;
            let mut count = [0usize; 4];
            let (mut deep, mut over) = (0f32, 0f32);
            if let Some(l) = left {
                for j in pj * per..(pj + 1) * per {
                    for i in pi * per..(pi + 1) * per {
                        let v = l[j * grid.nx + i];
                        let c = Class::of(v as f64);
                        count[c as usize] += 1;
                        match c {
                            Class::Gouge => deep = deep.max(-v),
                            Class::Left => over += v,
                            _ => {}
                        }
                    }
                }
            } else {
                count[Class::Waste as usize] = 1;
            }
            let class = if count[Class::Gouge as usize] > 0 {
                Class::Gouge
            } else {
                [Class::Waste, Class::Left, Class::OnSize].into_iter().max_by_key(|&c| count[c as usize]).unwrap_or(Class::Waste)
            };
            let tol = TOL as f32;
            let mix = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
            let base = match class {
                Class::OnSize => [0.30, 0.68, 0.33],
                Class::Left => {
                    let amount = over / count[Class::Left as usize].max(1) as f32;
                    mix([0.66, 0.82, 0.98], [0.07, 0.22, 0.58], ((amount / tol).log10() / 3.0).clamp(0.0, 1.0))
                }
                Class::Gouge => mix([0.98, 0.60, 0.18], [0.75, 0.03, 0.05], ((deep / tol).log10() / 2.0).clamp(0.0, 1.0)),
                Class::Waste => {
                    let g = 0.22 + 0.55 * (mean[pj * w + pi] / z_top).clamp(0.0, 1.0);
                    [g, g, g]
                }
            };
            let o = ((ht - 1 - pj) * w + pi) * 3;
            for (k, c) in base.iter().enumerate() {
                rgb[o + k] = ((c * shade).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    (w as u32, ht as u32, rgb)
}
