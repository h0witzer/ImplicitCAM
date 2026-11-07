//! Implicit matrix representations (M-reps) of rational Bezier patches, after
//! Busé, "Implicit matrix representations of rational Bézier curves and
//! surfaces", Computer-Aided Design 46 (2014).
//!
//! A patch of bidegree (d1, d2) gets a matrix M(p) = M0 + x M1 + y M2 + z M3 whose
//! rank drops exactly where p lies on the algebraic surface the patch is a piece
//! of. That surface is the whole extended one: any analytic field that vanishes
//! on the patch vanishes on all of it, so the extension is not a choice. The
//! matrix is evaluated at a point by a fixed arithmetic kernel over the four
//! stored matrices; no edge, trim or intersection data enters it. Its product of
//! singular values is zero exactly on the surface (Busé, Theorem 1) and grows
//! with the distance to it in the Lojasiewicz sense (Theorem 2); its left null
//! vector at a surface point is the Bernstein basis there, which reads the
//! parameters back (the inversion property).
//!
//! Construction (Busé §2.3): rows are the Bernstein basis of bidegree
//! nu = (2 d1 - 1, d2 - 1) on the patch's own parameter square, columns a basis of
//! the moving planes g0 + g1 x + g2 y + g3 z of that bidegree that vanish along
//! the patch, g0 w + g1 w x + g2 w y + g3 w z == 0 as polynomials of bidegree
//! nu + d, found as the null space of one linear system by one-sided Jacobi.
//! Coordinates are first mapped into the unit ball about the patch's box center,
//! which keeps that system well scaled. Without base points or a degree drop the
//! null space has exactly 2 d1 d2 directions (18 for a bicubic) and the matrix is
//! square; base points (collapsed edges, poles) and degree drops (a plane,
//! cylinder or sweep written as a bicubic) give it more columns, and the
//! rank-drop property holds for the whole null space (Busé, Fact 2).
//!
//! `square` takes exactly 2 d1 d2 directions and decides nothing by a threshold;
//! for a span without base points or degree drop it is the exact representation,
//! to the conditioning of the system (a fitted span's near-null directions mix in
//! at about 1e-16 over its 19th singular value). `with_tol` reads the null space
//! at a floor, treating near-syzygies as syzygies: a measurement for icam-check,
//! not a field. Read as a field it is the surface a fitted span was fitted to,
//! not the span's own, and that reading was withdrawn (Derek, 2026-10-25;
//! docs/SPAN_IMPLICIT.md).

use crate::curve3::V3;
use crate::surface3::Patch;

/// A rational Bezier patch's implicit matrix representation.
#[derive(Clone, Debug)]
pub struct MRep {
    /// Rows: the Bernstein basis of bidegree `nu`.
    pub m: usize,
    /// Columns: the moving planes found.
    pub r: usize,
    pub nu: (usize, usize),
    /// Row-major m x r matrices M0, Mx, My, Mz over the normalized coordinates.
    pub mats: [Vec<f64>; 4],
    /// Normalized coordinates: q = (p - center) / scale.
    pub center: V3,
    pub scale: f64,
    /// Singular values of the moving-plane system, ascending, relative to the
    /// largest. The first `r` belong to the planes taken; entry `r` is the first
    /// left out.
    pub spectrum: Vec<f64>,
    /// The precision floor the null space was read at.
    pub tol: f64,
    /// Largest Bernstein coefficient of g0 w + g1 w x + g2 w y + g3 w z over the
    /// found planes, relative to the patch's homogeneous size: the planes'
    /// residual as polynomials.
    pub residual: f64,
}

/// Binomial coefficient as f64.
fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

/// Bernstein basis of degree n at s.
fn bernstein(n: usize, s: f64) -> Vec<f64> {
    let mut b = vec![0.0; n + 1];
    b[0] = 1.0;
    for j in 1..=n {
        for i in (1..=j).rev() {
            b[i] = s * b[i - 1] + (1.0 - s) * b[i];
        }
        b[0] *= 1.0 - s;
    }
    b
}

/// One-sided (Hestenes) Jacobi on the columns of a column-major m x n matrix:
/// returns the singular values (one per column, unsorted) and V (n x n,
/// column-major) with A V = U diag(sigma). Columns whose singular value is zero
/// are left as zero; their V columns span the null space of A.
pub(crate) fn hestenes(a: &mut [f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let eps = 1e-15;
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (cp, cq) = (&a[p * m..(p + 1) * m], &a[q * m..(q + 1) * m]);
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    alpha += cp[i] * cp[i];
                    beta += cq[i] * cq[i];
                    gamma += cp[i] * cq[i];
                }
                if gamma.abs() <= eps * (alpha * beta).sqrt() || alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (a[p * m + i], a[q * m + i]);
                    a[p * m + i] = c * x - s * y;
                    a[q * m + i] = s * x + c * y;
                }
                for i in 0..n {
                    let (x, y) = (v[p * n + i], v[q * n + i]);
                    v[p * n + i] = c * x - s * y;
                    v[q * n + i] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = (0..n).map(|j| a[j * m..(j + 1) * m].iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
    (sigma, v)
}

impl MRep {
    /// The floor icam-check's measurements read the null space at. Not a field
    /// reading (see the module notes).
    pub const MEASURE_FLOOR: f64 = 1e-8;

    /// The square M-rep: exactly m moving planes, the directions the system acts
    /// on least, whatever the spectrum looks like. With no base points and no
    /// degree drop this is the exact M-rep and its determinant the implicit
    /// equation of the patch's own algebraic surface.
    pub fn square(patch: &Patch) -> Result<MRep, String> {
        Self::with_tol(patch, f64::NEG_INFINITY)
    }

    /// The M-rep at Busé's critical bidegree (2 d1 - 1, d2 - 1), reading the null
    /// space at the given floor (a floor below every singular value takes exactly
    /// m directions, as `square` does). A measurement, not a field.
    pub fn with_tol(patch: &Patch, tol: f64) -> Result<MRep, String> {
        let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
        if d1 == 0 || d2 == 0 {
            return Err("M-rep: a patch of degree 0 in a direction".into());
        }
        let nu = (2 * d1 - 1, d2 - 1);
        Self::at_degree(patch, nu, tol)
    }

    /// The M-rep of a patch with moving planes of bidegree `nu`.
    pub fn at_degree(patch: &Patch, nu: (usize, usize), tol: f64) -> Result<MRep, String> {
        let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
        let (lo, hi) = patch.bbox();
        let center = (lo + hi) * 0.5;
        let scale = ((hi - lo).len() * 0.5).max(1e-300);
        // Homogeneous components in normalized coordinates, weights scaled to
        // a unit largest weight: f0 = w, f1 = w x, f2 = w y, f3 = w z.
        let wmax = patch.pw.iter().flatten().map(|h| h[3].abs()).fold(0.0, f64::max);
        if !(wmax > 0.0) {
            return Err("M-rep: patch with zero weights".into());
        }
        let f: Vec<Vec<[f64; 4]>> = patch
            .pw
            .iter()
            .map(|row| {
                row.iter()
                    .map(|h| {
                        let w = h[3] / wmax;
                        let q = (V3::new(h[0], h[1], h[2]) * (1.0 / h[3]) - center) * (1.0 / scale);
                        [w, w * q.x, w * q.y, w * q.z]
                    })
                    .collect()
            })
            .collect();
        let fsize = f.iter().flatten().flat_map(|c| c.iter()).map(|x| x.abs()).fold(0.0, f64::max);
        let (n1, n2) = nu;
        let m = (n1 + 1) * (n2 + 1);
        let (e1, e2) = (n1 + d1, n2 + d2);
        let rows = (e1 + 1) * (e2 + 1);
        let cols = 4 * m;
        // The linear system: column (k, i, j) is the Bernstein expansion of
        // B_i^{n1}(s) B_j^{n2}(t) f_k(s, t) in bidegree (e1, e2).
        let mut a = vec![0.0; rows * cols];
        for k in 0..4 {
            for i in 0..=n1 {
                for j in 0..=n2 {
                    let col = k * m + i * (n2 + 1) + j;
                    for a1 in 0..=d1 {
                        for a2 in 0..=d2 {
                            let coef = binom(n1, i) * binom(d1, a1) / binom(e1, i + a1) * binom(n2, j) * binom(d2, a2) / binom(e2, j + a2);
                            let row = (i + a1) * (e2 + 1) + (j + a2);
                            a[col * rows + row] += coef * f[a1][a2][k];
                        }
                    }
                }
            }
        }
        let (sigma, v) = hestenes(&mut a, rows, cols);
        let smax = sigma.iter().cloned().fold(0.0, f64::max);
        if !(smax > 0.0) {
            return Err("M-rep: the moving-plane system vanished".into());
        }
        // The null space at the floor: every direction the system acts on by
        // less than tol of its norm. With no base points and no degree drop that
        // is exactly m directions (Busé: r = m at the critical degree).
        let mut order: Vec<usize> = (0..cols).collect();
        order.sort_by(|&x, &y| sigma[x].partial_cmp(&sigma[y]).unwrap());
        let spectrum: Vec<f64> = order.iter().map(|&j| sigma[j] / smax).collect();
        let r = spectrum.iter().filter(|&&x| x <= tol).count().max(m);
        let null = &order[..r];
        // Column j of M_k is the g_k coefficients of null vector j.
        let mut mats: [Vec<f64>; 4] = [vec![0.0; m * r], vec![0.0; m * r], vec![0.0; m * r], vec![0.0; m * r]];
        for (j, &nj) in null.iter().enumerate() {
            for k in 0..4 {
                for i in 0..m {
                    mats[k][i * r + j] = v[nj * cols + k * m + i];
                }
            }
        }
        // Residual of the found planes as polynomials, from the untouched system
        // (the columns of `a` were rotated, so rebuild the product directly).
        let mut residual = 0.0f64;
        for j in 0..r {
            let mut prod = vec![0.0; rows];
            for k in 0..4 {
                for i in 0..=n1 {
                    for jj in 0..=n2 {
                        let g = mats[k][(i * (n2 + 1) + jj) * r + j];
                        if g == 0.0 {
                            continue;
                        }
                        for a1 in 0..=d1 {
                            for a2 in 0..=d2 {
                                let coef = binom(n1, i) * binom(d1, a1) / binom(e1, i + a1) * binom(n2, jj) * binom(d2, a2) / binom(e2, jj + a2);
                                prod[(i + a1) * (e2 + 1) + (jj + a2)] += coef * g * f[a1][a2][k];
                            }
                        }
                    }
                }
            }
            residual = residual.max(prod.iter().map(|x| x.abs()).fold(0.0, f64::max) / fsize);
        }
        Ok(MRep { m, r, nu, mats, center, scale, spectrum, tol, residual })
    }

    /// M(p), row-major m x r.
    pub fn matrix(&self, p: V3) -> Vec<f64> {
        let q = (p - self.center) * (1.0 / self.scale);
        let n = self.m * self.r;
        let mut out = vec![0.0; n];
        let [m0, mx, my, mz] = &self.mats;
        for i in 0..n {
            out[i] = m0[i] + q.x * mx[i] + q.y * my[i] + q.z * mz[i];
        }
        out
    }

    /// Singular values of M(p), descending, and the left singular vectors as
    /// columns (m x m, column-major) in the same order.
    pub fn svd(&self, p: V3) -> (Vec<f64>, Vec<f64>) {
        let mat = self.matrix(p);
        let (m, r) = (self.m, self.r);
        // Hestenes on M^T (r x m, column-major: column i is row i of M) gives
        // the right vectors of M^T, which are the left vectors of M, including
        // the ones with zero singular value.
        let mut at = vec![0.0; r * m];
        for i in 0..m {
            for j in 0..r {
                at[i * r + j] = mat[i * r + j];
            }
        }
        let (sigma, v) = hestenes(&mut at, r, m);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&x, &y| sigma[y].partial_cmp(&sigma[x]).unwrap());
        let s: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
        let mut u = vec![0.0; m * m];
        for (k, &i) in order.iter().enumerate() {
            u[k * m..(k + 1) * m].copy_from_slice(&v[i * m..(i + 1) * m]);
        }
        (s, u)
    }

    /// The determinant of M(p) for a square M-rep (LU with row pivoting over the
    /// normalized coordinates), the implicit equation's value: its sign flips
    /// across every real sheet of the algebraic surface. The kernel the field
    /// would evaluate; a non-square M-rep has none.
    pub fn det(&self, p: V3) -> Option<f64> {
        if self.r != self.m {
            return None;
        }
        let n = self.m;
        let mut a = self.matrix(p);
        let mut det = 1.0;
        for k in 0..n {
            let mut piv = k;
            for i in k + 1..n {
                if a[i * n + k].abs() > a[piv * n + k].abs() {
                    piv = i;
                }
            }
            if a[piv * n + k] == 0.0 {
                return Some(0.0);
            }
            if piv != k {
                for j in 0..n {
                    a.swap(k * n + j, piv * n + j);
                }
                det = -det;
            }
            let d = a[k * n + k];
            det *= d;
            for i in k + 1..n {
                let f = a[i * n + k] / d;
                if f != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= f * a[k * n + j];
                    }
                }
            }
        }
        Some(det)
    }

    /// Busé's evaluation function: the product of the singular values of M(p),
    /// zero exactly on the extended surface.
    pub fn delta(&self, p: V3) -> f64 {
        self.svd(p).0.iter().product()
    }

    /// The smallest singular value of M(p): zero on the surface, linear in the
    /// distance near a regular sheet.
    pub fn sigma_min(&self, p: V3) -> f64 {
        *self.svd(p).0.last().unwrap()
    }

    /// Parameters (s, t) in the patch's unit square read from the left null
    /// vector of M(p) (the Bernstein basis of bidegree nu there). Meaningful at a
    /// point on the surface with one preimage; elsewhere it is the reading of the
    /// smallest left singular vector. A direction whose moving-plane degree is
    /// zero (t at the critical bidegree of a patch linear in v) reads as 0.5.
    pub fn invert(&self, p: V3) -> (f64, f64) {
        let (_, u) = self.svd(p);
        let (n1, n2) = self.nu;
        let m = self.m;
        let psi = &u[(m - 1) * m..m * m];
        let (mut bs, mut bt) = (vec![0.0; n1 + 1], vec![0.0; n2 + 1]);
        for i in 0..=n1 {
            for j in 0..=n2 {
                let x = psi[i * (n2 + 1) + j];
                bs[i] += x;
                bt[j] += x;
            }
        }
        let read = |b: &[f64], n: usize| -> f64 {
            let total: f64 = b.iter().sum();
            if n == 0 || total == 0.0 {
                return 0.5;
            }
            b.iter().enumerate().map(|(i, &x)| i as f64 / n as f64 * x).sum::<f64>() / total
        };
        (read(&bs, n1), read(&bt, n2))
    }

    /// Bernstein basis of bidegree nu at (s, t), row order of the M-rep.
    pub fn basis(&self, s: f64, t: f64) -> Vec<f64> {
        let (bs, bt) = (bernstein(self.nu.0, s), bernstein(self.nu.1, t));
        let mut out = Vec::with_capacity(self.m);
        for i in 0..=self.nu.0 {
            for j in 0..=self.nu.1 {
                out.push(bs[i] * bt[j]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch(pts: Vec<Vec<[f64; 4]>>) -> Patch {
        Patch { pw: pts, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
    }

    /// z = x y over [0, 1]^2 as a bilinear patch: a quadric.
    fn saddle() -> Patch {
        patch(vec![vec![[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], vec![[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    }

    /// A bicubic with no base points, points on a wavy sheet.
    fn wavy() -> Patch {
        let rows = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let (fi, fj) = (i as f64, j as f64);
                        let x = fi * 10.0 + 2.0 * (fi + fj).sin();
                        let y = fj * 7.0 + 1.5 * (2.0 * fi + fj).cos();
                        let z = ((i * 3 + j * 5) % 7) as f64 - 3.0 + 0.25 * x.sin();
                        [x, y, z, 1.0]
                    })
                    .collect()
            })
            .collect();
        patch(rows)
    }

    /// A sphere octant of radius 2 about (1, 1, 1) as a rational biquadratic: the
    /// quarter circle in the x z plane revolved a quarter turn; the pole row
    /// collapses, so the parameterization has base points.
    fn octant() -> Patch {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let (r, c) = (2.0, V3::new(1.0, 1.0, 1.0));
        // Meridian (u): equator (1, 0, 0) -> corner (1, 0, 1) w -> pole (0, 0, 1).
        let mer = [([1.0, 0.0, 0.0], 1.0), ([1.0, 0.0, 1.0], w), ([0.0, 0.0, 1.0], 1.0)];
        // Turn (v) about z: (x, y) -> (x, x) w -> (0, x) ... applied to the x of each meridian point.
        let turn = |x: f64| [([x, 0.0], 1.0), ([x, x], w), ([0.0, x], 1.0)];
        let rows = mer
            .iter()
            .map(|&(mp, mw)| {
                turn(mp[0])
                    .iter()
                    .map(|&(xy, tw)| {
                        let wt = mw * tw;
                        let p = V3::new(c.x + r * xy[0], c.y + r * xy[1], c.z + r * mp[2]);
                        [p.x * wt, p.y * wt, p.z * wt, wt]
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        patch(rows)
    }

    fn on_and_off(p: &Patch, name: &str) -> MRep {
        let rep = MRep::with_tol(p, MRep::MEASURE_FLOOR).unwrap();
        assert!(rep.residual < 1e-10, "{name}: residual {}", rep.residual);
        let mut worst_on = 0.0f64;
        let mut worst_inv = 0.0f64;
        let mut least_off = f64::INFINITY;
        let scale = rep.scale;
        for i in 0..=6 {
            for j in 0..=6 {
                let (s, t) = (i as f64 / 6.0, j as f64 / 6.0);
                let e = p.eval(s, t);
                let sig = rep.svd(e.p).0;
                worst_on = worst_on.max(sig[sig.len() - 1] / sig[0]);
                if sig.len() > 1 && sig[sig.len() - 2] / sig[0] > 1e-6 {
                    let (si, ti) = rep.invert(e.p);
                    if rep.nu.0 > 0 {
                        worst_inv = worst_inv.max((si - s).abs());
                    }
                    if rep.nu.1 > 0 {
                        worst_inv = worst_inv.max((ti - t).abs());
                    }
                }
                let n = e.ps.cross(e.pt);
                if n.len() > 1e-9 {
                    let q = e.p + n.unit() * (0.05 * scale);
                    let sq = rep.svd(q).0;
                    least_off = least_off.min(sq[sq.len() - 1] / sq[0]);
                }
            }
        }
        assert!(worst_on < 1e-9, "{name}: on-surface singular value ratio {worst_on:.2e}");
        assert!(worst_inv < 1e-6, "{name}: inversion error {worst_inv:.2e}");
        assert!(least_off > 1e-4, "{name}: off-surface singular value ratio {least_off:.2e}");
        rep
    }

    #[test]
    fn saddle_is_a_quadric() {
        let rep = on_and_off(&saddle(), "saddle");
        assert_eq!((rep.m, rep.r), (2, 2));
        // Its determinant is the implicit equation: zero on z = x y, nonzero off it.
        let det = |p: V3| {
            let a = rep.matrix(p);
            a[0] * a[3] - a[1] * a[2]
        };
        assert!(det(V3::new(0.3, 0.7, 0.21)).abs() < 1e-12);
        assert!(det(V3::new(3.0, -2.0, -6.0)).abs() < 1e-10, "the extended surface is the whole quadric");
        assert!(det(V3::new(0.3, 0.7, 0.5)).abs() > 1e-3);
    }

    #[test]
    fn bicubic_is_square_18() {
        let rep = on_and_off(&wavy(), "wavy");
        assert_eq!((rep.m, rep.r), (18, 18));
        assert!(rep.spectrum[17] < 1e-12 && rep.spectrum[18] > 1e-6, "spectrum about m: {:.2e} / {:.2e}", rep.spectrum[17], rep.spectrum[18]);
    }

    #[test]
    fn sphere_octant_with_base_points() {
        let rep = on_and_off(&octant(), "octant");
        // Base points: more moving planes than rows.
        assert!(rep.r > rep.m, "{} planes for {} rows", rep.r, rep.m);
        // The extended surface is the whole sphere: a point on its far side is on it.
        let far = V3::new(1.0 - 2.0 / 3f64.sqrt(), 1.0 - 2.0 / 3f64.sqrt(), 1.0 - 2.0 / 3f64.sqrt());
        let sig = rep.svd(far).0;
        assert!(sig[sig.len() - 1] / sig[0] < 1e-9, "far side ratio {:.2e}", sig[sig.len() - 1] / sig[0]);
        let inside = rep.svd(V3::new(1.0, 1.0, 1.0)).0;
        assert!(inside[inside.len() - 1] / inside[0] > 1e-3);
    }

    #[test]
    fn determinant_is_the_implicit_equation() {
        let p = wavy();
        let rep = MRep::square(&p).unwrap();
        assert_eq!(rep.r, 18);
        // Zero on the patch, opposite signs on the two sides of it, and the
        // same sign as det along the normal on each side across the patch.
        let mut signs = Vec::new();
        for i in 0..=4 {
            for j in 0..=4 {
                let e = p.eval(i as f64 / 4.0, j as f64 / 4.0);
                let n = e.ps.cross(e.pt).unit();
                let on = rep.det(e.p).unwrap();
                let (a, b) = (rep.det(e.p + n * (0.02 * rep.scale)).unwrap(), rep.det(e.p - n * (0.02 * rep.scale)).unwrap());
                assert!(on.abs() < 1e-9 * a.abs().max(b.abs()), "det on the patch {on:.2e} against {a:.2e} / {b:.2e}");
                assert!((a < 0.0) != (b < 0.0), "no sign change across the patch: {a:.2e} / {b:.2e}");
                signs.push(a < 0.0);
            }
        }
        assert!(signs.iter().all(|&x| x == signs[0]), "the sign along the normal flips over the patch");
        // The same patch at a precision floor takes the same 18 directions.
        assert_eq!(MRep::with_tol(&p, MRep::MEASURE_FLOOR).unwrap().r, 18);
    }

    #[test]
    fn delta_grows_with_distance() {
        let p = wavy();
        let rep = MRep::with_tol(&p, MRep::MEASURE_FLOOR).unwrap();
        let e = p.eval(0.4, 0.6);
        let n = e.ps.cross(e.pt).unit();
        let d: Vec<f64> = [0.01, 0.02, 0.04, 0.08].iter().map(|&h| rep.sigma_min(e.p + n * (h * rep.scale))).collect();
        for k in 1..d.len() {
            assert!(d[k] > d[k - 1], "sigma_min not increasing: {d:?}");
        }
        // Linear near the sheet: doubling the distance about doubles sigma_min.
        let ratio = d[1] / d[0];
        assert!((ratio - 2.0).abs() < 0.3, "ratio {ratio}");
    }
}
