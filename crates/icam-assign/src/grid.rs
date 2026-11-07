//! A uniform XY raster over one layer's domain, with the exact Euclidean distance
//! transform (Felzenszwalb and Huttenlocher 2012, two separable passes of the lower
//! envelope of parabolas) and 8-connected component labeling. Cell (i, j) is the point
//! (x0 + i h, y0 + j h).

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Grid {
    pub x0: f64,
    pub y0: f64,
    pub h: f64,
    pub nx: usize,
    pub ny: usize,
}

const INF: f64 = 1e20;

impl Grid {
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64, h: f64) -> Grid {
        Grid { x0, y0, h, nx: ((x1 - x0) / h).ceil() as usize + 1, ny: ((y1 - y0) / h).ceil() as usize + 1 }
    }

    pub fn n(&self) -> usize {
        self.nx * self.ny
    }

    pub fn x(&self, i: usize) -> f64 {
        self.x0 + i as f64 * self.h
    }

    pub fn y(&self, j: usize) -> f64 {
        self.y0 + j as f64 * self.h
    }

    pub fn i(&self, x: f64) -> i64 {
        ((x - self.x0) / self.h).round() as i64
    }

    pub fn j(&self, y: f64) -> i64 {
        ((y - self.y0) / self.h).round() as i64
    }

    pub fn contains(&self, i: i64, j: i64) -> bool {
        i >= 0 && j >= 0 && i < self.nx as i64 && j < self.ny as i64
    }

    /// Distance (mm) from every cell to the nearest cell where `seed` is true (infinite
    /// when there is none). Exact on the lattice.
    pub fn edt(&self, seed: &[bool]) -> Vec<f32> {
        let (nx, ny) = (self.nx, self.ny);
        if !seed.iter().any(|&s| s) {
            return vec![f32::INFINITY; self.n()];
        }
        let mut f: Vec<f64> = seed.iter().map(|&s| if s { 0.0 } else { INF }).collect();
        // Columns (along j) then rows (along i), squared distances in cells.
        let cols: Vec<Vec<f64>> = (0..nx)
            .into_par_iter()
            .map(|i| {
                let src: Vec<f64> = (0..ny).map(|j| f[j * nx + i]).collect();
                dt1(&src)
            })
            .collect();
        for (i, col) in cols.iter().enumerate() {
            for j in 0..ny {
                f[j * nx + i] = col[j];
            }
        }
        let h = self.h;
        let mut o = vec![0f32; self.n()];
        o.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
            let d = dt1(&f[j * nx..(j + 1) * nx]);
            for i in 0..nx {
                row[i] = if d[i] >= 1e19 { f32::INFINITY } else { (d[i].sqrt() * h) as f32 };
            }
        });
        o
    }

    /// 8-connected components of the cells where `mask` is true: the label per cell (-1
    /// outside the mask) and the number of components.
    pub fn components(&self, mask: &[bool]) -> (Vec<i32>, usize) {
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        let mut lab = vec![-1i32; self.n()];
        let mut count = 0;
        let mut stack = Vec::new();
        for s in 0..self.n() {
            if !mask[s] || lab[s] >= 0 {
                continue;
            }
            lab[s] = count as i32;
            stack.push(s);
            while let Some(c) = stack.pop() {
                let (ci, cj) = ((c % self.nx) as i64, (c / self.nx) as i64);
                for dj in -1..=1 {
                    for di in -1..=1 {
                        let (ni, nj) = (ci + di, cj + dj);
                        if (di == 0 && dj == 0) || ni < 0 || nj < 0 || ni >= nx || nj >= ny {
                            continue;
                        }
                        let n = (nj * nx + ni) as usize;
                        if mask[n] && lab[n] < 0 {
                            lab[n] = count as i32;
                            stack.push(n);
                        }
                    }
                }
            }
            count += 1;
        }
        (lab, count)
    }
}

/// One pass: the lower envelope of the parabolas of the finite samples.
fn dt1(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let (mut v, mut z) = (vec![0usize; n], vec![0.0f64; n + 1]);
    let mut k: i64 = -1;
    for q in 0..n {
        if f[q] >= INF {
            continue;
        }
        if k < 0 {
            k = 0;
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            continue;
        }
        let s = loop {
            let p = v[k as usize];
            let s = ((f[q] + q as f64 * q as f64) - (f[p] + p as f64 * p as f64)) / (2.0 * q as f64 - 2.0 * p as f64);
            if s <= z[k as usize] {
                k -= 1; // z[0] = -inf: never pops the first
                continue;
            }
            break s;
        };
        k += 1;
        v[k as usize] = q;
        z[k as usize] = s;
        z[k as usize + 1] = f64::INFINITY;
    }
    if k < 0 {
        return vec![INF; n];
    }
    let mut d = vec![0.0; n];
    let mut k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let dq = q as f64 - v[k] as f64;
        d[q] = dq * dq + f[v[k]];
    }
    d
}
