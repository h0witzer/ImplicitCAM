//! The signed implicit polynomial of a rational Bezier span: the polynomial F of
//! least degree that vanishes on the span, oriented so F > 0 on the side its
//! normal Su x Sv points to.
//!
//! Why this and not the matrix representation's determinant (`mrep`): a span
//! whose implicit degree is below 2 d1 d2 (an exact conic, sphere, torus or cone
//! written as a rational patch) or that has base points (a collapsed edge, a pole)
//! has more moving planes than rows. Its exact matrix representation is then not
//! square, has no determinant, and its evaluation function is a sum of squares
//! with no sign; any square choice of its columns picks up extraneous factors
//! (measured on the STEP Tools suite, docs/SPAN_IMPLICIT.md section 7). The least
//! degree polynomial has neither problem. The span is the image of an irreducible
//! parameterization, so its algebraic surface is irreducible and its ideal is
//! principal: the polynomials of least degree vanishing on it are the multiples
//! of one generator by constants. F is that generator, so F's zero set is the
//! span's extended surface and nothing else, and F changes sign across every
//! smooth real point of it.
//!
//! Construction (exact implicitization by linear algebra, after Dokken; see
//! Barrowclough and Dokken, "Approximate implicitization using linear algebra",
//! J. Appl. Math. 2012, section 6.4): for degree k, W^k F(P / W) is a polynomial of
//! bidegree (k d1, k d2) in the span's parameters, so it is zero identically when
//! it is zero on a tensor grid of (k d1 + 2) x (k d2 + 2) distinct parameters, and
//! because W > 0 on the span that is the same as F vanishing at the span's points
//! there. The rows of the system are the monomials of degree <= k at those points
//! (coordinates normalized to the unit ball about the span's box center), and F's
//! candidate coefficients are its least singular direction, found by one-sided
//! Jacobi.
//!
//! Which degree is the span's own: the least k whose candidate (1) vanishes on
//! the span, (2) still vanishes `REACH` spans out along the span's polynomial
//! extension, both to `EXT_TOL` in the scale-free measure of `ext_residual`, and
//! (3) is unique, its system's second singular value at least `UNIQUE` times the
//! least. Vanishing on the span alone decides nothing: a small span is
//! approximated at low degree to any floor, while only the span's algebraic
//! surface vanishes along its whole extension. Uniqueness keeps out products such
//! as a sphere times a plane, which vanish wherever the sphere does. No floor on
//! the singular values is used: a file that writes its weights to 8 digits holds
//! a sphere to 2.6e-11, and that is its surface.
//!
//! The evaluation is a fixed polynomial in three variables: a straight-line kernel
//! of C(k + 3, 3) terms per point (35 at degree 4, 165 at degree 8), with no edge,
//! trim or neighbor data. F is a sign and a zero set, not a distance; the distance is
//! the exact evaluator's (`solid3::Exact`, docs/EXACT_FIELD.md).

use crate::curve3::V3;
use crate::mrep::hestenes;
use crate::surface3::Patch;

/// The exponents (a, b, c) of the monomials x^a y^b z^c of degree <= k, by total
/// degree and then by falling powers of x and y.
pub fn monomials(k: usize) -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    for t in 0..=k {
        for a in (0..=t).rev() {
            for b in (0..=t - a).rev() {
                out.push([a as u8, b as u8, (t - a - b) as u8]);
            }
        }
    }
    out
}

/// What the degree search saw at one degree: the smallest and second smallest
/// singular values of the system, relative to its largest, and how well the
/// least direction vanishes on the span's extension.
#[derive(Clone, Copy, Debug)]
pub struct DegreeStep {
    pub k: usize,
    pub sigma: f64,
    pub sigma_next: f64,
    /// The least direction's relative residual (`ext_residual`) on the span,
    /// and on its extension to parameters [-REACH, 1 + REACH]^2.
    pub on_span: f64,
    pub ext: f64,
}

/// The signed implicit polynomial of one span.
#[derive(Clone, Debug)]
pub struct SpanPoly {
    /// Implicit degree.
    pub degree: usize,
    /// Coefficients over `mons` in the normalized coordinates
    /// q = (p - center) / scale, unit 2-norm, signed so F > 0 along Su x Sv.
    pub coef: Vec<f64>,
    /// `monomials(degree)`.
    pub mons: Vec<[u8; 3]>,
    pub center: V3,
    pub scale: f64,
    /// The system at `degree`: its smallest singular value and the next,
    /// relative to its largest (their ratio is the uniqueness gap).
    pub sigma: f64,
    pub sigma_next: f64,
    /// The smallest singular value one degree lower: how far the span is from
    /// lying on a surface of lower degree. 1 at degree 1.
    pub sigma_below: f64,
    /// Every degree tried, in order.
    pub steps: Vec<DegreeStep>,
}

/// Why a span has no polynomial within the degree limit, with what was seen.
#[derive(Clone, Debug)]
pub struct NoPoly {
    pub reason: String,
    pub steps: Vec<DegreeStep>,
}

fn chebyshev(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * n) as f64).cos()).collect()
}

impl SpanPoly {
    /// How far apart the least and second singular values must be for the least
    /// direction to be the polynomial: measured, spans with a known surface 2e4 and
    /// up, products and approximations 7 and down (`print_lab`).
    pub const UNIQUE: f64 = 1e3;
    /// The highest degree searched by default: 165 coefficients.
    pub const KMAX: usize = 8;
    /// How far past the span, in span sizes, F must still vanish. Far enough
    /// that a small piece's extension reaches the scale of its curvature: an
    /// approximation's misfit grows like reach^(k (d - 1)) times its residual on
    /// the span, and at 8 spans a generic piece cut to a hundredth was still at
    /// 1e-8, beside the exact pieces.
    pub const REACH: f64 = 128.0;
    /// The largest relative residual allowed there. Measured (test
    /// `exactness_lab`, `print_lab` for the table): spans with a known surface,
    /// whole or cut to a hundredth, with misrounded weights and elevated degrees,
    /// stay at or below 1.5e-9; spans only approximated at degree <= 8, whole or
    /// cut to a hundredth, are at 9.9e-3 or above. A torus cut to a tenth or a
    /// hundredth lands between (1.4e-7 to 3.7e-4): its data fix the quartic only
    /// up to a second near-null direction of 1e-11 to 1e-15.
    pub const EXT_TOL: f64 = 1e-6;

    pub fn of_patch(patch: &Patch) -> Result<SpanPoly, NoPoly> {
        Self::search(patch, Self::KMAX)
    }

    /// The least degree k <= kmax (and <= 2 d1 d2, the degree of a span without
    /// base points) whose candidate vanishes on the span and `REACH` spans out
    /// to `EXT_TOL` and is unique (see the module notes).
    ///
    /// Withdrawn as an acceptance rule (2026-10-25): evaluated far outside the
    /// span, the residual's mixed terms amplify the arithmetic's rounding by
    /// 1e12, so exact spans of real files fail it (tork's chamfer plane scored
    /// 2.4e-4 while lying on its plane to 1.4e-10). It is not used by the
    /// importer; analytic surfaces are read by `recognize`, on the span itself.
    /// Kept for icam-check's measurements until the polynomial kernel is rebuilt
    /// on the same certified standard.
    pub fn search(patch: &Patch, kmax: usize) -> Result<SpanPoly, NoPoly> {
        let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
        let (lo, hi) = patch.bbox();
        let center = (lo + hi) * 0.5;
        let scale = ((hi - lo).len() * 0.5).max(1e-300);
        let kcap = kmax.min((2 * d1 * d2).max(1));
        let mut steps: Vec<DegreeStep> = Vec::new();
        for k in 1..=kcap {
            let (step, coef) = Self::system(patch, k, center, scale);
            steps.push(step);
            let (s1, s2) = (step.sigma, step.sigma_next);
            if step.on_span > Self::EXT_TOL || step.ext > Self::EXT_TOL || !(s2 >= Self::UNIQUE * s1) {
                // Not the surface at this degree: it misses the span, or vanishes
                // on the span only (an approximation, or a surface the data do not
                // pin down), or is one of a family (a product with a lower degree).
                continue;
            }
            let sigma_below = if steps.len() > 1 { steps[steps.len() - 2].sigma } else { 1.0 };
            let mut sp = SpanPoly { degree: k, coef, mons: monomials(k), center, scale, sigma: s1, sigma_next: s2, sigma_below, steps };
            // Orientation from the span's own normal at its center.
            let e = patch.eval(0.5, 0.5);
            if sp.grad(e.p).dot(e.ps.cross(e.pt)) < 0.0 {
                for c in &mut sp.coef {
                    *c = -*c;
                }
            }
            return Ok(sp);
        }
        Err(NoPoly { reason: format!("no polynomial of degree <= {kcap} vanishes on the span and {} spans out to {:.0e} and is unique", Self::REACH, Self::EXT_TOL), steps })
    }

    /// Every degree from 1 to min(kmax, 2 d1 d2), whatever it finds: the profile
    /// a floor is read against.
    pub fn profile(patch: &Patch, kmax: usize) -> Vec<DegreeStep> {
        let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
        let (lo, hi) = patch.bbox();
        let center = (lo + hi) * 0.5;
        let scale = ((hi - lo).len() * 0.5).max(1e-300);
        (1..=kmax.min((2 * d1 * d2).max(1))).map(|k| Self::system(patch, k, center, scale).0).collect()
    }

    /// The degree-k system's two smallest singular values and its least
    /// direction as coefficients (unit norm, unoriented).
    fn system(patch: &Patch, k: usize, center: V3, scale: f64) -> (DegreeStep, Vec<f64>) {
        let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
        {
            let mons = monomials(k);
            let cols = mons.len();
            let (n1, n2) = (k * d1 + 2, k * d2 + 2);
            let (gs, gt) = (chebyshev(n1), chebyshev(n2));
            let rows = n1 * n2;
            // Column-major system over the unit ball. The columns are not rescaled
            // to unit norm: a monomial that vanishes on the span (x on the plane
            // x = 0) is a column of roundoff, and scaling it up turns the noise into
            // a direction (bull.stp's axis-aligned planar spans then failed degree 1
            // and came back as the plane times a stray second plane at degree 2).
            let mut a = vec![0.0; rows * cols];
            let mut r = 0;
            for &s in &gs {
                for &t in &gt {
                    let q = (patch.eval(s, t).p - center) * (1.0 / scale);
                    let (px, py, pz) = (powers(q.x, k), powers(q.y, k), powers(q.z, k));
                    for (j, m) in mons.iter().enumerate() {
                        a[j * rows + r] = px[m[0] as usize] * py[m[1] as usize] * pz[m[2] as usize];
                    }
                    r += 1;
                }
            }
            let (sig, v) = hestenes(&mut a, rows, cols);
            let smax = sig.iter().cloned().fold(0.0, f64::max);
            let mut order: Vec<usize> = (0..cols).collect();
            order.sort_by(|&x, &y| sig[x].partial_cmp(&sig[y]).unwrap());
            let s1 = sig[order[0]] / smax;
            let s2 = if cols > 1 { sig[order[1]] / smax } else { 1.0 };
            let j0 = order[0];
            let mut coef: Vec<f64> = (0..cols).map(|j| v[j0 * cols + j]).collect();
            let cn = coef.iter().map(|x| x * x).sum::<f64>().sqrt();
            for c in &mut coef {
                *c /= cn;
            }
            let (on_span, ext) = (ext_residual(patch, k, &mons, &coef, center, scale, 0.0), ext_residual(patch, k, &mons, &coef, center, scale, Self::REACH));
            (DegreeStep { k, sigma: s1, sigma_next: s2, on_span, ext }, coef)
        }
    }

    /// F at p.
    pub fn value(&self, p: V3) -> f64 {
        let q = (p - self.center) * (1.0 / self.scale);
        let (px, py, pz) = (pow_table(q.x, self.degree), pow_table(q.y, self.degree), pow_table(q.z, self.degree));
        self.mons.iter().zip(&self.coef).map(|(m, c)| c * px[m[0] as usize] * py[m[1] as usize] * pz[m[2] as usize]).sum()
    }

    /// The gradient of F at p, per unit length.
    pub fn grad(&self, p: V3) -> V3 {
        let q = (p - self.center) * (1.0 / self.scale);
        let (px, py, pz) = (pow_table(q.x, self.degree), pow_table(q.y, self.degree), pow_table(q.z, self.degree));
        let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
        for (m, c) in self.mons.iter().zip(&self.coef) {
            let (a, b, cc) = (m[0] as usize, m[1] as usize, m[2] as usize);
            if a > 0 {
                gx += c * a as f64 * px[a - 1] * py[b] * pz[cc];
            }
            if b > 0 {
                gy += c * b as f64 * px[a] * py[b - 1] * pz[cc];
            }
            if cc > 0 {
                gz += c * cc as f64 * px[a] * py[b] * pz[cc - 1];
            }
        }
        V3::new(gx, gy, gz) * (1.0 / self.scale)
    }

    /// F / |grad F| at p: the first-order distance to the zero set, exact on it.
    pub fn first_order(&self, p: V3) -> f64 {
        let g = self.grad(p).len();
        if g > 0.0 {
            self.value(p) / g
        } else {
            f64::INFINITY
        }
    }
}

/// The relative residual of a degree-k polynomial on a span, in the span's
/// homogeneous form: over a 9 x 9 grid of parameters on [-e, 1 + e]^2, the largest
/// |sum c m| / (|c| |m|) with m the monomials W^(k - |a|) X^a of the span's
/// homogeneous coordinates in the normalized frame (the cosine between the
/// coefficients and the monomials, which Cauchy-Schwarz bounds by 1). Not
/// |sum c m| / sum |c m|: on a plane every true term is exactly zero, and that
/// ratio is then noise over noise. It is about the arithmetic's rounding for
/// the polynomial of the span's algebraic surface, which vanishes on the whole
/// extension, and it grows fast off the span for a polynomial that only
/// approximates it there (a polynomial residual of bidegree (k d1, k d2) small on
/// the unit square grows like a Chebyshev polynomial outside it). Homogeneous, so
/// no division by a weight that may cross zero off the span.
pub fn ext_residual(patch: &Patch, k: usize, mons: &[[u8; 3]], coef: &[f64], center: V3, scale: f64, e: f64) -> f64 {
    let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
    // The span in powers of (s - 1/2) and (t - 1/2): evaluated by Horner far from
    // the span, the Bernstein form sums huge alternating terms and loses digits
    // that are then mistaken for the polynomial not vanishing.
    let pw = power_form(patch);
    let mut worst = 0.0f64;
    for i in 0..9 {
        for j in 0..9 {
            let (s, t) = (-e + (1.0 + 2.0 * e) * i as f64 / 8.0, -e + (1.0 + 2.0 * e) * j as f64 / 8.0);
            let (ds, dt) = (s - 0.5, t - 0.5);
            let mut h = [0.0f64; 4];
            for c in 0..4 {
                let mut acc = 0.0;
                for a in (0..=d1).rev() {
                    let mut row = 0.0;
                    for b in (0..=d2).rev() {
                        row = row * dt + pw[a][b][c];
                    }
                    acc = acc * ds + row;
                }
                h[c] = acc;
            }
            let w = h[3];
            let x = [(h[0] - center.x * w) / scale, (h[1] - center.y * w) / scale, (h[2] - center.z * w) / scale];
            // The measure is the same for any multiple of the homogeneous point, so
            // take the unit one: far out a (5,11) span's coordinates reach 128^16,
            // and their eighth powers would overflow.
            let len = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + w * w).sqrt();
            if !(len > 0.0) {
                continue;
            }
            let (x, w) = ([x[0] / len, x[1] / len, x[2] / len], w / len);
            let (px, py, pz, pw) = (pow_table(x[0], k), pow_table(x[1], k), pow_table(x[2], k), pow_table(w, k));
            let (mut num, mut mm, mut cc) = (0.0f64, 0.0f64, 0.0f64);
            for (m, c) in mons.iter().zip(coef) {
                let deg = (m[0] + m[1] + m[2]) as usize;
                let v = px[m[0] as usize] * py[m[1] as usize] * pz[m[2] as usize] * pw[k - deg];
                num += c * v;
                mm += v * v;
                cc += c * c;
            }
            if mm > 0.0 && cc > 0.0 {
                worst = worst.max(num.abs() / (mm.sqrt() * cc.sqrt()));
            }
        }
    }
    worst
}

/// Taylor coefficients about 1/2 of a univariate Bernstein polynomial: the
/// j-th is n! / (n - j)! / j! times the j-th forward difference of the control
/// values evaluated at 1/2 by de Casteljau.
fn taylor_half(b: &[f64]) -> Vec<f64> {
    let n = b.len() - 1;
    let mut diff = b.to_vec();
    let mut out = Vec::with_capacity(n + 1);
    let mut fall = 1.0; // n! / (n - j)! / j!
    for j in 0..=n {
        let mut c = diff.clone();
        for level in 1..c.len() {
            for i in 0..c.len() - level {
                c[i] = 0.5 * (c[i] + c[i + 1]);
            }
        }
        out.push(fall * c[0]);
        if j < n {
            diff = diff.windows(2).map(|w| w[1] - w[0]).collect();
            fall = fall * (n - j) as f64 / (j + 1) as f64;
        }
    }
    out
}

/// A homogeneous Bezier patch in powers of (s - 1/2) and (t - 1/2):
/// out[a][b][c] multiplies (s - 1/2)^a (t - 1/2)^b in component c.
fn power_form(patch: &Patch) -> Vec<Vec<[f64; 4]>> {
    let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
    let mut mid = vec![vec![[0.0f64; 4]; d2 + 1]; d1 + 1];
    for b in 0..=d2 {
        for c in 0..4 {
            let col: Vec<f64> = (0..=d1).map(|a| patch.pw[a][b][c]).collect();
            for (a, v) in taylor_half(&col).into_iter().enumerate() {
                mid[a][b][c] = v;
            }
        }
    }
    let mut out = vec![vec![[0.0f64; 4]; d2 + 1]; d1 + 1];
    for a in 0..=d1 {
        for c in 0..4 {
            let row: Vec<f64> = (0..=d2).map(|b| mid[a][b][c]).collect();
            for (b, v) in taylor_half(&row).into_iter().enumerate() {
                out[a][b][c] = v;
            }
        }
    }
    out
}

/// x^0 ..= x^k on the stack (k <= 24, far above any degree searched).
fn pow_table(x: f64, k: usize) -> [f64; 25] {
    let mut p = [0.0; 25];
    p[0] = 1.0;
    for i in 0..k.min(24) {
        p[i + 1] = p[i] * x;
    }
    p
}

fn powers(x: f64, k: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(k + 1);
    p.push(1.0);
    for i in 0..k {
        p.push(p[i] * x);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch(pw: Vec<Vec<[f64; 4]>>) -> Patch {
        Patch { pw, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
    }

    const W: f64 = std::f64::consts::FRAC_1_SQRT_2;

    /// The rational quadratic quarter circle (1,0) -> (0,1) as homogeneous points,
    /// scaled by r: control (r,0) w 1, (r,r) w 1/sqrt2, (0,r) w 1.
    fn circ(r: f64) -> [([f64; 2], f64); 3] {
        [([r, 0.0], 1.0), ([r, r], W), ([0.0, r], 1.0)]
    }

    /// A surface of revolution about z from a meridian of homogeneous (rho, z, w)
    /// points: rows along the meridian, columns the quarter turn.
    fn revolve(mer: &[([f64; 2], f64)]) -> Patch {
        let rows = mer
            .iter()
            .map(|&([rho, z], wm)| {
                circ(rho)
                    .iter()
                    .map(|&([x, y], wt)| {
                        let w = wm * wt;
                        [x * w, y * w, z * w, w]
                    })
                    .collect()
            })
            .collect();
        patch(rows)
    }

    /// Sign agreement of F with a true signed function over random points of the
    /// padded box, skipping points within `band` of the surface; the truth is
    /// oriented by the span's normal at its center.
    fn agrees(sp: &SpanPoly, p: &Patch, truth: &dyn Fn(V3) -> f64, band: f64) -> (usize, usize) {
        let e = p.eval(0.5, 0.5);
        let n = e.ps.cross(e.pt).unit();
        let h = 1e-5 * sp.scale;
        let side = (truth(e.p + n * h) - truth(e.p - n * h)).signum();
        let (lo, hi) = p.bbox();
        let pad = (hi - lo) * 0.5;
        let (lo, hi) = (lo - pad, hi + pad);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut u = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut n_ok, mut n_all) = (0, 0);
        for _ in 0..4000 {
            let x = V3::new(lo.x + (hi.x - lo.x) * u(), lo.y + (hi.y - lo.y) * u(), lo.z + (hi.z - lo.z) * u());
            let t = truth(x);
            if t.abs() < band {
                continue;
            }
            n_all += 1;
            if (sp.value(x) > 0.0) == (t * side > 0.0) {
                n_ok += 1;
            }
        }
        (n_ok, n_all)
    }

    #[test]
    fn plane_is_degree_one() {
        let p = patch(vec![vec![[0.0, 0.0, 1.0, 1.0], [0.0, 3.0, 1.0, 1.0]], vec![[2.0, 0.0, 1.0, 1.0], [2.0, 3.0, 1.0, 1.0]]]);
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 1);
        // Su x Sv is +z here, so F > 0 above the plane z = 1, and F / |grad F| is the distance.
        assert!((sp.first_order(V3::new(5.0, -4.0, 3.5)) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn axis_aligned_rectangle_is_degree_one() {
        // bull.stp: the plane x = 31.75, where the x column is pure roundoff.
        let p = patch(vec![vec![[31.75, 31.75, 21.3614, 1.0], [31.75, 31.75, 77.34173, 1.0]], vec![[31.75, 0.0, 21.3614, 1.0], [31.75, 0.0, 77.34173, 1.0]]]);
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 1, "{:?}", sp.steps);
        let n = { let e = p.eval(0.5, 0.5); e.ps.cross(e.pt).unit() };
        let x = V3::new(40.0, 10.0, 50.0);
        assert!((sp.first_order(x) - (x.x - 31.75) * n.x).abs() < 1e-9);
    }

    #[test]
    fn saddle_is_degree_two() {
        let p = patch(vec![vec![[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], vec![[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]);
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 2);
        let (ok, all) = agrees(&sp, &p, &|x: V3| x.z - x.x * x.y, 1e-9);
        assert_eq!(ok, all, "{ok} of {all}");
    }

    #[test]
    fn rational_cylinder_is_a_quadric() {
        let (r, h) = (10.0, 7.0);
        let rows = circ(r).iter().map(|&([x, y], w)| vec![[x * w, y * w, 0.0, w], [x * w, y * w, h * w, w]]).collect();
        let p = patch(rows);
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 2);
        assert!(sp.sigma < 1e-13 && sp.sigma_below > 1e-6, "degree gap {:.1e} / {:.1e}", sp.sigma, sp.sigma_below);
        // The far side of the cylinder is on the zero set: the extension is the whole quadric.
        assert!(sp.first_order(V3::new(-r, 0.0, 30.0)).abs() < 1e-9 * r);
        let (ok, all) = agrees(&sp, &p, &|x: V3| (x.x * x.x + x.y * x.y).sqrt() - r, 1e-9);
        assert_eq!(ok, all, "{ok} of {all}");
    }

    #[test]
    fn rational_cone_is_a_quadric() {
        let (r0, r1, h) = (4.0, 9.0, 10.0);
        let rows = circ(1.0).iter().map(|&([x, y], w)| vec![[x * r0 * w, y * r0 * w, 0.0, w], [x * r1 * w, y * r1 * w, h * w, w]]).collect();
        let p = patch(rows);
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 2);
        let slope = (r1 - r0) / h;
        let truth = |x: V3| (x.x * x.x + x.y * x.y).sqrt() - (r0 + slope * x.z);
        // The cone's quadric is both nappes; compare where the radius is positive.
        let (ok, all) = agrees(&sp, &p, &|x: V3| if r0 + slope * x.z > 0.0 { truth(x) } else { 0.0 }, 1e-9);
        assert_eq!(ok, all, "{ok} of {all}");
    }

    #[test]
    fn sphere_octant_with_a_pole_is_a_quadric() {
        let (r, c) = (2.0, V3::new(1.0, 1.0, 1.0));
        // Meridian from the equator to the pole; the pole row collapses (base points).
        let mer = [([r, 0.0], 1.0), ([r, r], W), ([0.0, r], 1.0)];
        let mut p = revolve(&mer);
        for row in p.pw.iter_mut() {
            for h in row.iter_mut() {
                h[0] += c.x * h[3];
                h[1] += c.y * h[3];
                h[2] += c.z * h[3];
            }
        }
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 2);
        assert!(sp.first_order(c - V3::new(r, 0.0, 0.0)).abs() < 1e-9 * r, "far side of the sphere");
        let (ok, all) = agrees(&sp, &p, &|x: V3| (x - c).len() - r, 1e-9);
        assert_eq!(ok, all, "{ok} of {all}");
    }

    #[test]
    fn rational_torus_is_a_quartic() {
        let (big, r) = (20.0, 5.0);
        // Meridian: the tube's quarter circle about (big, 0) in the (rho, z) plane.
        let mer = [([big + r, 0.0], 1.0), ([big + r, r], W), ([big, r], 1.0)];
        let p = revolve(&mer);
        let sp = SpanPoly::of_patch(&p).unwrap();
        assert_eq!(sp.degree, 4);
        // Exact at degree 4 to double precision, a cubic misses by far more.
        assert!(sp.sigma < 1e-13 && sp.sigma_below > 1e-6, "degree gap {:.1e} / {:.1e}", sp.sigma, sp.sigma_below);
        let truth = |x: V3| ((x.x * x.x + x.y * x.y).sqrt() - big).hypot(x.z) - r;
        assert!(sp.first_order(V3::new(-big + r, 0.0, 0.0)).abs() < 1e-8 * big, "inner ring across the axis");
        let (ok, all) = agrees(&sp, &p, &truth, 1e-9);
        assert_eq!(ok, all, "{ok} of {all}");
    }

    #[test]
    fn generic_bicubic_is_above_the_limit() {
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
        let e = SpanPoly::of_patch(&patch(rows)).unwrap_err();
        assert_eq!(e.steps.len(), 8);
        // Approximation, not exactness: the least singular value falls by one to
        // two orders a degree (0.56 at degree 1, under 1e-8 at degree 8), so a
        // floor of 1e-8 would have taken degree 8; off the span it does not vanish.
        let s: Vec<f64> = e.steps.iter().map(|s| s.sigma).collect();
        assert!(s.windows(2).all(|w| w[1] < w[0] && w[1] > 1e-3 * w[0]), "{s:?}");
        assert!(s[7] < 1e-8, "{s:?}");
        assert!(e.steps[7].ext > 1e-2, "{:?}", e.steps[7]);
    }

    #[test]
    fn power_form_matches_the_bernstein_form() {
        let p = patch(vec![vec![[1.0, 2.0, 0.5, 1.0], [2.0, -1.0, 3.0, 0.8], [0.0, 4.0, 1.0, 1.2]], vec![[3.0, 0.0, 2.0, 0.9], [1.0, 1.0, 1.0, 1.1], [2.5, 2.0, -1.0, 1.0]]]);
        let pw = power_form(&p);
        for &(s, t) in &[(0.0, 0.0), (0.3, 0.8), (1.0, 1.0), (0.5, 0.5)] {
            let e = p.eval(s, t).p;
            let mut h = [0.0f64; 4];
            for a in 0..pw.len() {
                for b in 0..pw[0].len() {
                    for c in 0..4 {
                        h[c] += pw[a][b][c] * (s - 0.5f64).powi(a as i32) * (t - 0.5f64).powi(b as i32);
                    }
                }
            }
            let q = V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
            assert!((q - e).len() < 1e-13, "{s} {t}: {q:?} {e:?}");
        }
    }

    #[test]
    fn monomial_count() {
        for k in 0..9 {
            assert_eq!(monomials(k).len(), (k + 1) * (k + 2) * (k + 3) / 6);
        }
    }
}

#[cfg(test)]
mod exactness_lab {
    //! Ground truth for the exactness test: spans whose algebraic surface is known
    //! (cylinder, cone, sphere, torus, with the misrounded weights real files
    //! write, degree-elevated as writers do, and cut down to small pieces) against
    //! spans that are only approximated at degree <= 8 (generic bicubic and rational
    //! (2,3) pieces of the same sizes).
    use super::*;

    const WB: f64 = 0.707106781186707; // bull.stp's square root of a half

    fn circ(r: f64, w: f64) -> [([f64; 2], f64); 3] {
        [([r, 0.0], 1.0), ([r, r], w), ([0.0, r], 1.0)]
    }

    fn revolve(mer: &[([f64; 2], f64)], w: f64) -> Patch {
        let rows = mer
            .iter()
            .map(|&([rho, z], wm)| circ(rho, w).iter().map(|&([x, y], wt)| { let ww = wm * wt; [x * ww, y * ww, z * ww, ww] }).collect())
            .collect();
        Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
    }

    /// Degree elevation in u of a homogeneous Bezier patch (rows along u).
    fn elevate_u(p: &Patch) -> Patch {
        let n = p.pw.len() - 1;
        let cols = p.pw[0].len();
        let mut rows = vec![vec![[0.0; 4]; cols]; n + 2];
        for i in 0..=n + 1 {
            let a = i as f64 / (n + 1) as f64;
            for j in 0..cols {
                for c in 0..4 {
                    let lo = if i > 0 { p.pw[i - 1][j][c] } else { 0.0 };
                    let hi = if i <= n { p.pw[i][j][c] } else { 0.0 };
                    rows[i][j][c] = a * lo + (1.0 - a) * hi;
                }
            }
        }
        Patch { pw: rows, ..p.clone() }
    }

    fn transpose(p: &Patch) -> Patch {
        let (n, m) = (p.pw.len(), p.pw[0].len());
        Patch { pw: (0..m).map(|j| (0..n).map(|i| p.pw[i][j]).collect()).collect(), u0: p.v0, u1: p.v1, v0: p.u0, v1: p.u1 }
    }

    /// The piece [a, a + f] x [b, b + f] of a patch.
    fn piece(p: &Patch, a: f64, b: f64, f: f64) -> Patch {
        let (_, r) = p.split_u(a);
        let (l, _) = r.split_u(f / (1.0 - a));
        let (_, r) = l.split_v(b);
        let (l, _) = r.split_v(f / (1.0 - b));
        l
    }

    fn generic(d1: usize, d2: usize, rational: bool) -> Patch {
        let rows = (0..=d1)
            .map(|i| {
                (0..=d2)
                    .map(|j| {
                        let (fi, fj) = (i as f64, j as f64);
                        let w = if rational { 1.0 + 0.3 * ((fi * 1.7 + fj * 0.9).sin()) } else { 1.0 };
                        let (x, y, z) = (fi * 10.0 + 2.0 * (fi + fj).sin(), fj * 7.0 + 1.5 * (2.0 * fi + fj).cos(), 3.0 * ((fi * 3.0 + fj * 5.0) * 0.7).sin());
                        [x * w, y * w, z * w, w]
                    })
                    .collect()
            })
            .collect();
        Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
    }

    fn ext_at(p: &Patch, k: usize, e: f64) -> f64 {
        let (lo, hi) = p.bbox();
        let (center, scale) = ((lo + hi) * 0.5, (hi - lo).len() * 0.5);
        let sp_coef = {
            // The least direction at degree k, as the search computes it.
            let (_, coef) = SpanPoly::system(p, k, center, scale);
            coef
        };
        ext_residual(p, k, &monomials(k), &sp_coef, center, scale, e)
    }

    #[test]
    fn exact_spans_pass_and_approximations_do_not() {
        let (r, big) = (5.0, 20.0);
        let sphere = revolve(&[([r, 0.0], 1.0), ([r, r], WB), ([0.0, r], 1.0)], WB);
        let torus = revolve(&[([big + r, 0.0], 1.0), ([big + r, r], WB), ([big, r], 1.0)], WB);
        let cyl = {
            let rows = circ(r, WB).iter().map(|&([x, y], w)| vec![[x * w, y * w, 0.0, w], [x * w, y * w, 9.0 * w, w]]).collect();
            Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
        };
        let elevated = transpose(&elevate_u(&elevate_u(&transpose(&elevate_u(&cyl)))));
        for (name, p, k) in [("sphere", &sphere, 2), ("torus", &torus, 4), ("cylinder", &cyl, 2), ("cylinder elevated to (3,3)", &elevated, 2)] {
            for f in [1.0, 0.1, 0.01] {
                if name == "torus" && f < 1.0 {
                    continue;
                }
                let q = if f < 1.0 { piece(p, 0.37, 0.41, f) } else { p.clone() };
                let sp = SpanPoly::of_patch(&q).unwrap_or_else(|e| panic!("{name} piece {f}: {}", e.reason));
                assert_eq!(sp.degree, k, "{name} piece {f}");
            }
        }
        // Weights written to 8 digits: the data hold a sphere to 2.6e-11, and the
        // sphere is what is read, not the cubic product of it with a plane.
        let sphere8 = revolve(&[([r, 0.0], 1.0), ([r, r], 0.70710678), ([0.0, r], 1.0)], 0.70710678);
        assert_eq!(SpanPoly::of_patch(&sphere8).unwrap().degree, 2);
        // A torus cut to a hundredth: its data leave a second near-null quartic.
        assert!(SpanPoly::of_patch(&piece(&torus, 0.37, 0.41, 0.01)).is_err());
        for (name, p) in [("generic (3,3)", generic(3, 3, false)), ("generic rational (2,3)", generic(2, 3, true))] {
            for f in [1.0, 0.1, 0.01] {
                let q = if f < 1.0 { piece(&p, 0.37, 0.41, f) } else { p.clone() };
                assert!(SpanPoly::of_patch(&q).is_err(), "{name} piece {f} read as exact");
            }
        }
    }

    #[test]
    #[ignore]
    fn print_lab() {
        let (r, big) = (5.0, 20.0);
        let cyl = {
            let rows = circ(r, WB).iter().map(|&([x, y], w)| vec![[x * w, y * w, 0.0, w], [x * w, y * w, 9.0 * w, w]]).collect();
            Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
        };
        let cone = {
            let rows = circ(1.0, WB).iter().map(|&([x, y], w)| vec![[x * 4.0 * w, y * 4.0 * w, 0.0, w], [x * 9.0 * w, y * 9.0 * w, 10.0 * w, w]]).collect();
            Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 }
        };
        let sphere = revolve(&[([r, 0.0], 1.0), ([r, r], WB), ([0.0, r], 1.0)], WB);
        let torus = revolve(&[([big + r, 0.0], 1.0), ([big + r, r], WB), ([big, r], 1.0)], WB);
        let sphere8 = revolve(&[([r, 0.0], 1.0), ([r, r], 0.70710678), ([0.0, r], 1.0)], 0.70710678);
        for d in SpanPoly::profile(&sphere8, 8) {
            println!("  8-digit sphere degree {}: sigma {:.1e} next {:.1e} ext {:.1e}", d.k, d.sigma, d.sigma_next, d.ext);
        }
        match SpanPoly::of_patch(&sphere8) {
            Ok(sp) => println!("sphere with 8-digit weights: read at degree {} (sigma {:.1e}, below {:.1e})", sp.degree, sp.sigma, sp.sigma_below),
            Err(e) => println!("sphere with 8-digit weights: refused, {}", e.reason),
        }
        let cases: Vec<(&str, Patch, usize)> = vec![
            ("cylinder (2,1)", cyl.clone(), 2),
            ("cylinder elevated (3,3)", transpose(&elevate_u(&elevate_u(&transpose(&elevate_u(&cyl))))), 2),
            ("cone (2,1)", cone, 2),
            ("sphere (2,2)", sphere.clone(), 2),
            ("torus (2,2)", torus.clone(), 4),
            ("torus elevated (3,3)", transpose(&elevate_u(&transpose(&elevate_u(&torus)))), 4),
            ("generic (3,3)", generic(3, 3, false), 0),
            ("generic rational (2,3)", generic(2, 3, true), 0),
            ("generic (1,3)", generic(1, 3, false), 0),
        ];
        println!("{:<26} {:>7} {:>3} {:>10} {:>10} {:>10} {:>10} {:>10}", "case", "piece", "k", "sigma", "next", "ext 8", "ext 32", "ext 128");
        for (name, p, truth) in &cases {
            for f in [1.0, 0.1, 0.01] {
                let q = if f < 1.0 { piece(p, 0.37, 0.41, f) } else { p.clone() };
                let prof = SpanPoly::profile(&q, 8);
                // The floor rule's pick, or the true degree for the exact cases.
                let k = if *truth > 0 { *truth } else { prof.iter().find(|d| d.sigma <= 1e-12).map_or(prof.len(), |d| d.k) };
                let d = prof[k - 1];
                println!("{:<26} {:>7} {:>3} {:>10.1e} {:>10.1e} {:>10.1e} {:>10.1e} {:>10.1e}", name, f, k, d.sigma, d.sigma_next, ext_at(&q, k, 8.0), ext_at(&q, k, 32.0), ext_at(&q, k, 128.0));
            }
        }
    }
}

#[cfg(test)]
mod strict_probe {
    use super::*;

    /// Does the exact surface of a misrounded sphere span keep a sign across it?
    #[test]
    #[ignore]
    fn misrounded_sphere_sign() {
        const W: f64 = std::f64::consts::FRAC_1_SQRT_2;
        for (name, w) in [("f64 sqrt(1/2)", W), ("bull 0.707106781186707", 0.707106781186707), ("8 digits", 0.70710678)] {
            let r = 5.0;
            let circ = |rr: f64| [([rr, 0.0], 1.0), ([rr, rr], w), ([0.0, rr], 1.0)];
            let mer = [([r, 0.0], 1.0), ([r, r], w), ([0.0, r], 1.0)];
            let rows = mer.iter().map(|&([rho, z], wm)| circ(rho).iter().map(|&([x, y], wt)| { let ww = wm * wt; [x * ww, y * ww, z * ww, ww] }).collect()).collect();
            let p = Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 };
            let (lo, hi) = p.bbox();
            let (center, scale) = ((lo + hi) * 0.5, (hi - lo).len() * 0.5);
            for k in [2usize, 4] {
                let (step, coef) = SpanPoly::system(&p, k, center, scale);
                let sp = SpanPoly { degree: k, coef, mons: monomials(k), center, scale, sigma: step.sigma, sigma_next: step.sigma_next, sigma_below: 0.0, steps: vec![] };
                // F along the sphere's normal through the patch center, at offsets of
                // the sphere's radius times h, both sides.
                let e = p.eval(0.5, 0.5);
                let n = (e.p).unit();
                let vals: Vec<String> = [-1e-2, -1e-4, -1e-6, -1e-8, 0.0, 1e-8, 1e-6, 1e-4, 1e-2].iter().map(|&h| format!("{:+.1e}", sp.value(e.p + n * (h * r)))).collect();
                println!("{name:<24} degree {k}: sigma {:.1e} next {:.1e}; F at -1e-2..+1e-2 radii: {}", step.sigma, step.sigma_next, vals.join(" "));
            }
        }
    }
}

// ---------------------------------------------------------------- analytic surfaces

/// Eigenvalues and unit eigenvectors (columns) of a symmetric 3 x 3 matrix.
pub fn sym3_eigen(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let off: f64 = (0..3).flat_map(|p| (p + 1..3).map(move |q| (p, q))).map(|(p, q)| a[p][q] * a[p][q]).sum();
        let diag: f64 = (0..3).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-34 * diag || off == 0.0 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..3 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let th = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = if th == 0.0 { 1.0 } else { th.signum() / (th.abs() + (th * th + 1.0).sqrt()) };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..3 {
                    let (x, y) = (a[r][p], a[r][q]);
                    a[r][p] = c * x - s * y;
                    a[r][q] = s * x + c * y;
                }
                for r in 0..3 {
                    let (x, y) = (a[p][r], a[q][r]);
                    a[p][r] = c * x - s * y;
                    a[q][r] = s * x + c * y;
                }
                for r in 0..3 {
                    let (x, y) = (v[r][p], v[r][q]);
                    v[r][p] = c * x - s * y;
                    v[r][q] = s * x + c * y;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

fn frame_on(o: V3, z: V3) -> crate::curve3::Frame {
    let z = z.unit();
    let t = if z.x.abs() < 0.9 { V3::new(1.0, 0.0, 0.0) } else { V3::new(0.0, 1.0, 0.0) };
    let x = (t - z * t.dot(z)).unit();
    crate::curve3::Frame { o, z, x }
}

/// An analytic surface read off a span's polynomial, and how far the span's own
/// points are from it.
#[derive(Clone, Debug)]
pub struct Analytic {
    pub surf: crate::surface3::Surf,
    /// The span is this surface: every point of the span is within `ROUNDING`
    /// of it, relative to the span's size (`certified_distance`). Derek,
    /// 2026-10-25: a span is its analytic surface when the two differ only by the
    /// rounding of the span's own numbers.
    pub exact: bool,
    /// The largest distance from a 25 x 25 grid of the span's points to the
    /// surface, in the span's units and relative to the span's size.
    pub dev: f64,
    pub dev_rel: f64,
}

impl SpanPoly {
    /// The degree-2 least direction of a span whatever its profile says (the
    /// quadric nearest the span in the system's sense), oriented by Su x Sv.
    pub fn quadric_of(patch: &Patch) -> SpanPoly {
        let (lo, hi) = patch.bbox();
        let center = (lo + hi) * 0.5;
        let scale = ((hi - lo).len() * 0.5).max(1e-300);
        let (step, coef) = Self::system(patch, 2, center, scale);
        let mut sp = SpanPoly { degree: 2, coef, mons: monomials(2), center, scale, sigma: step.sigma, sigma_next: step.sigma_next, sigma_below: f64::NAN, steps: vec![step] };
        let e = patch.eval(0.5, 0.5);
        if sp.grad(e.p).dot(e.ps.cross(e.pt)) < 0.0 {
            for c in &mut sp.coef {
                *c = -*c;
            }
        }
        sp
    }

    /// The plane, sphere, circular cylinder or circular cone this polynomial is
    /// nearest to, built from its coefficients (degree 1 or 2), with its
    /// deviation from the span's points. Every kind that can be built is built
    /// and measured, and the one nearest the span is returned: no threshold
    /// decides the kind. Whether the deviation makes it the span's surface is the
    /// caller's to judge against a stated standard.
    pub fn analytic(&self, patch: &Patch) -> Option<Analytic> {
        let mut best: Option<Analytic> = None;
        for s in self.analytic_candidates() {
            let mut dev = 0.0f64;
            for i in 0..=24 {
                for j in 0..=24 {
                    dev = dev.max(s.dist(patch.eval(i as f64 / 24.0, j as f64 / 24.0).p));
                }
            }
            if best.as_ref().map_or(true, |b| dev < b.dev) {
                let exact = within_rounding(&s, patch);
                best = Some(Analytic { surf: s, exact, dev, dev_rel: dev / self.scale });
            }
        }
        best
    }

    /// Every plane, sphere, circular cylinder or circular cone that can be built
    /// from this polynomial's coefficients.
    pub fn analytic_candidates(&self) -> Vec<crate::surface3::Surf> {
        use crate::surface3::Surf;
        let c = |e: [u8; 3]| self.mons.iter().position(|m| *m == e).map_or(0.0, |i| self.coef[i]);
        let b = V3::new(c([1, 0, 0]), c([0, 1, 0]), c([0, 0, 1]));
        let k0 = c([0, 0, 0]);
        let to_world = |q: V3| self.center + q * self.scale;
        let mut cands: Vec<Surf> = Vec::new();
        if self.degree == 1 {
            if b.len() > 0.0 {
                let n = b * (1.0 / b.len());
                cands.push(Surf::Plane { f: frame_on(to_world(n * (-k0 / b.len())), n) });
            }
        } else if self.degree == 2 {
            let a = [[c([2, 0, 0]), 0.5 * c([1, 1, 0]), 0.5 * c([1, 0, 1])], [0.5 * c([1, 1, 0]), c([0, 2, 0]), 0.5 * c([0, 1, 1])], [0.5 * c([1, 0, 1]), 0.5 * c([0, 1, 1]), c([0, 0, 2])]];
            let (lam, v) = sym3_eigen(a);
            let ev = |i: usize| V3::new(v[0][i], v[1][i], v[2][i]);
            // Sphere: the mean eigenvalue.
            let l = (lam[0] + lam[1] + lam[2]) / 3.0;
            if l != 0.0 {
                let q0 = b * (-0.5 / l);
                let r2 = q0.dot(q0) - k0 / l;
                if r2 > 0.0 {
                    cands.push(Surf::Sphere { f: frame_on(to_world(q0), V3::new(0.0, 0.0, 1.0)), r: r2.sqrt() * self.scale });
                }
            }
            // Cylinder: the axis is the eigenvector of the eigenvalue nearest zero.
            let ia = (0..3).min_by(|&i, &j| lam[i].abs().partial_cmp(&lam[j].abs()).unwrap()).unwrap();
            let others: Vec<usize> = (0..3).filter(|&i| i != ia).collect();
            let lc = 0.5 * (lam[others[0]] + lam[others[1]]);
            if lc != 0.0 {
                let ax = ev(ia);
                let bp = b - ax * b.dot(ax);
                let q0 = bp * (-0.5 / lc);
                let r2 = q0.dot(q0) - k0 / lc;
                if r2 > 0.0 {
                    cands.push(Surf::Cylinder { f: frame_on(to_world(q0), ax), r: r2.sqrt() * self.scale });
                }
            }
            // Cone: the axis is the eigenvector whose eigenvalue's sign differs from
            // the other two; the apex is the quadric's center.
            for ia in 0..3 {
                let o: Vec<usize> = (0..3).filter(|&i| i != ia).collect();
                let lo_ = 0.5 * (lam[o[0]] + lam[o[1]]);
                if !(lo_ * lam[ia] < 0.0) {
                    continue;
                }
                let mut q0 = V3::new(0.0, 0.0, 0.0);
                let mut ok = true;
                for i in 0..3 {
                    if lam[i] == 0.0 {
                        ok = false;
                        break;
                    }
                    q0 = q0 + ev(i) * (-0.5 * b.dot(ev(i)) / lam[i]);
                }
                if !ok {
                    continue;
                }
                let ax = ev(ia);
                let half = (-lam[ia] / lo_).sqrt().atan();
                // The frame at the span's center height on the axis, z away from the apex.
                let apex = to_world(q0);
                let h = (self.center - apex).dot(ax);
                let z = if h >= 0.0 { ax } else { ax * -1.0 };
                let o_ = apex + z * h.abs();
                cands.push(Surf::Cone { f: frame_on(o_, z), r: h.abs() * half.tan(), a: half });
            }
        }
        cands
    }
}

#[cfg(test)]
mod analytic_tests {
    use super::*;
    use crate::surface3::Surf;

    const W: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn circ(r: f64) -> [([f64; 2], f64); 3] {
        [([r, 0.0], 1.0), ([r, r], W), ([0.0, r], 1.0)]
    }

    #[test]
    fn rational_sphere_reads_as_the_sphere() {
        let (r, c) = (5.0, V3::new(3.0, -2.0, 7.0));
        let mer = [([r, 0.0], 1.0), ([r, r], W), ([0.0, r], 1.0)];
        let rows = mer
            .iter()
            .map(|&([rho, z], wm)| {
                circ(rho)
                    .iter()
                    .map(|&([x, y], wt)| {
                        let w = wm * wt;
                        [(x + c.x) * w, (y + c.y) * w, (z + c.z) * w, w]
                    })
                    .collect()
            })
            .collect();
        let p = Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 };
        let a = SpanPoly::of_patch(&p).unwrap().analytic(&p).unwrap();
        match a.surf {
            Surf::Sphere { f, r: rr } => {
                assert!((f.o - c).len() < 1e-12 && (rr - r).abs() < 1e-12, "{:?} {rr}", f.o);
            }
            s => panic!("read as {}", s.name()),
        }
        assert!(a.dev < 1e-13, "{:.1e}", a.dev);
    }

    #[test]
    fn rational_cylinder_and_cone_read_as_themselves() {
        let rows = circ(4.0).iter().map(|&([x, y], w)| vec![[x * w, y * w, 1.0 * w, w], [x * w, y * w, 9.0 * w, w]]).collect();
        let p = Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 };
        let a = SpanPoly::of_patch(&p).unwrap().analytic(&p).unwrap();
        assert!(matches!(a.surf, Surf::Cylinder { r, .. } if (r - 4.0).abs() < 1e-12), "{:?}", a.surf);
        assert!(a.dev < 1e-13);
        let rows = circ(1.0).iter().map(|&([x, y], w)| vec![[x * 4.0 * w, y * 4.0 * w, 0.0, w], [x * 9.0 * w, y * 9.0 * w, 10.0 * w, w]]).collect();
        let p = Patch { pw: rows, u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 };
        let a = SpanPoly::of_patch(&p).unwrap().analytic(&p).unwrap();
        assert!(matches!(a.surf, Surf::Cone { a: half, .. } if (half - (5.0f64 / 10.0).atan()).abs() < 1e-12), "{:?}", a.surf);
        assert!(a.dev < 1e-12, "{:.1e}", a.dev);
    }
}


/// A polynomial in three variables as coefficients over `monomials(deg)`.
#[derive(Clone, Debug)]
struct Poly3 {
    deg: usize,
    c: Vec<f64>,
}

impl Poly3 {
    fn zero(deg: usize) -> Poly3 {
        Poly3 { deg, c: vec![0.0; monomials(deg).len()] }
    }
    fn index(deg: usize, e: [u8; 3]) -> usize {
        monomials(deg).iter().position(|m| *m == e).expect("monomial within the degree")
    }
    /// a + b . (x, y, z)
    fn linear(a: f64, b: V3) -> Poly3 {
        let mut p = Poly3::zero(1);
        p.c[Self::index(1, [0, 0, 0])] = a;
        p.c[Self::index(1, [1, 0, 0])] = b.x;
        p.c[Self::index(1, [0, 1, 0])] = b.y;
        p.c[Self::index(1, [0, 0, 1])] = b.z;
        p
    }
    fn raise(&self, deg: usize) -> Poly3 {
        let mut p = Poly3::zero(deg);
        for (m, v) in monomials(self.deg).iter().zip(&self.c) {
            p.c[Self::index(deg, *m)] += v;
        }
        p
    }
    fn add(&self, o: &Poly3, k: f64) -> Poly3 {
        let deg = self.deg.max(o.deg);
        let mut p = self.raise(deg);
        for (m, v) in monomials(o.deg).iter().zip(&o.c) {
            p.c[Self::index(deg, *m)] += k * v;
        }
        p
    }
    fn mul(&self, o: &Poly3) -> Poly3 {
        let deg = self.deg + o.deg;
        let mut p = Poly3::zero(deg);
        let (ma, mb) = (monomials(self.deg), monomials(o.deg));
        for (x, a) in ma.iter().zip(&self.c) {
            if *a == 0.0 {
                continue;
            }
            for (y, b) in mb.iter().zip(&o.c) {
                if *b != 0.0 {
                    p.c[Self::index(deg, [x[0] + y[0], x[1] + y[1], x[2] + y[2]])] += a * b;
                }
            }
        }
        p
    }
}

/// The implicit equation of a plane, sphere, circular cylinder, circular cone or
/// torus in a span's normalized frame q = (p - center) / scale: its degree,
/// monomials and coefficients (the quadrics' distance-squared form
/// (q - p)' M (q - p) + c; the torus's (|d|^2 + R^2 - r^2)^2 - 4 R^2 (|d|^2 - (d.z)^2)).
pub fn implicit_of(surf: &crate::surface3::Surf, center: V3, scale: f64) -> Option<(usize, Vec<[u8; 3]>, Vec<f64>)> {
    use crate::surface3::Surf;
    let at = |o: V3| (o - center) * (1.0 / scale);
    if let Surf::Torus { f, major, minor } = surf {
        // d = q - o: its components as linear polynomials.
        let o = at(f.o);
        let (big, small) = (major / scale, minor / scale);
        let dx = Poly3::linear(-o.x, V3::new(1.0, 0.0, 0.0));
        let dy = Poly3::linear(-o.y, V3::new(0.0, 1.0, 0.0));
        let dz = Poly3::linear(-o.z, V3::new(0.0, 0.0, 1.0));
        let d2 = dx.mul(&dx).add(&dy.mul(&dy), 1.0).add(&dz.mul(&dz), 1.0);
        let h = dx.add(&dx, f.z.x - 1.0).add(&dy, f.z.y).add(&dz, f.z.z);
        let h2 = h.mul(&h);
        let k = Poly3::linear(big * big - small * small, V3::new(0.0, 0.0, 0.0));
        let a = d2.add(&k, 1.0);
        let g = a.mul(&a).add(&d2.add(&h2, -1.0), -4.0 * big * big);
        return Some((4, monomials(4), g.c));
    }
    let (m, p, c0): ([[f64; 3]; 3], V3, f64) = match surf {
        Surf::Plane { f } => {
            let (n, p) = (f.z, at(f.o));
            let mons = monomials(1);
            let coef = mons
                .iter()
                .map(|e| match e {
                    [0, 0, 0] => -n.dot(p),
                    [1, 0, 0] => n.x,
                    [0, 1, 0] => n.y,
                    _ => n.z,
                })
                .collect();
            return Some((1, mons, coef));
        }
        Surf::Sphere { f, r } => ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], at(f.o), -(r / scale) * (r / scale)),
        Surf::Cylinder { f, r } => {
            let a = [f.z.x, f.z.y, f.z.z];
            let mut m = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] = if i == j { 1.0 } else { 0.0 } - a[i] * a[j];
                }
            }
            (m, at(f.o), -(r / scale) * (r / scale))
        }
        Surf::Cone { f, r, a } => {
            let z = [f.z.x, f.z.y, f.z.z];
            let sec2 = 1.0 / (a.cos() * a.cos());
            let mut m = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] = if i == j { 1.0 } else { 0.0 } - sec2 * z[i] * z[j];
                }
            }
            (m, at(f.o - f.z * (r / a.tan())), 0.0)
        }
        _ => return None,
    };
    let pv = [p.x, p.y, p.z];
    let mp: Vec<f64> = (0..3).map(|i| (0..3).map(|j| m[i][j] * pv[j]).sum()).collect();
    let pmp: f64 = (0..3).map(|i| pv[i] * mp[i]).sum();
    let mons = monomials(2);
    let coef = mons
        .iter()
        .map(|e| match *e {
            [0, 0, 0] => pmp + c0,
            [1, 0, 0] => -2.0 * mp[0],
            [0, 1, 0] => -2.0 * mp[1],
            [0, 0, 1] => -2.0 * mp[2],
            [2, 0, 0] => m[0][0],
            [0, 2, 0] => m[1][1],
            [0, 0, 2] => m[2][2],
            [1, 1, 0] => 2.0 * m[0][1],
            [1, 0, 1] => 2.0 * m[0][2],
            _ => 2.0 * m[1][2],
        })
        .collect();
    Some((2, mons, coef))
}

/// The line between a span that differs from an analytic surface only by the
/// rounding of its numbers and one that was fitted to it, relative to the span's
/// size (half its box diagonal). Measured on the STEP Tools suite and THE PART,
/// 49,509 B-spline spans, each against its nearest plane, sphere, circular
/// cylinder or circular cone: 1,158 within 1e-9 (writers' rounding: tork writes
/// 3.78 as 3.779999999999759), 48,351 at 1e-8 or more (fits), none between; the
/// line sits in that empty decade. Derek, 2026-10-25: "That's approximately where
/// every CAD kernel tries to build their surfaces to". Checked: every span it
/// accepts is within 1e-8 mm of its surface, inside every published kernel
/// resolution (`KERNEL_RESOLUTION_MM`).
pub const ROUNDING: f64 = 3e-9;

/// The tightest published CAD kernel resolution, the distance below which a
/// kernel treats two points as one, as a second, absolute ceiling on the same
/// test: Open CASCADE Precision::Confusion, 1e-7 mm (ACIS SPAresabs 1e-6, model
/// units normally mm; Parasolid linear resolution 1e-8 m = 1e-5 mm; CATIA V5
/// modeler resolution 1e-3 mm). A span within `ROUNDING` of a surface but farther
/// than this from it in absolute terms would be a different surface to that
/// kernel, and is not read as the surface.
pub const KERNEL_RESOLUTION_MM: f64 = 1e-7;

fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

/// The product of two bivariate Bernstein polynomials, coefficient arrays
/// [i][j] of bidegrees (len - 1).
fn bern_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (m1, m2, n1, n2) = (a.len() - 1, a[0].len() - 1, b.len() - 1, b[0].len() - 1);
    let mut c = vec![vec![0.0; m2 + n2 + 1]; m1 + n1 + 1];
    for i in 0..=m1 {
        for j in 0..=m2 {
            for k in 0..=n1 {
                for l in 0..=n2 {
                    c[i + k][j + l] += a[i][j] * b[k][l] * binom(m1, i) * binom(n1, k) / binom(m1 + n1, i + k) * binom(m2, j) * binom(n2, l) / binom(m2 + n2, j + l);
                }
            }
        }
    }
    c
}

/// An upper bound, over the whole span, on its distance from a plane, sphere,
/// circular cylinder or circular cone, relative to the span's size (half its box
/// diagonal), with no sampling: the surface's implicit equation G composed with
/// the span is W^k G(P / W), a Bernstein polynomial in the span's parameters
/// whose largest coefficient bounds it on the span; with the least weight
/// coefficient that bounds G, and G bounds the distance (|G| for a plane,
/// |G| / r for a sphere or cylinder, |G| cos^2 a / (h sin a) for a cone, h the
/// least height above the apex, so a span reaching the apex is not bounded).
/// Infinite for any other surface.
/// A bound on |G| over the span, G a polynomial over `monomials(k)` in the span's
/// normalized frame: the largest Bernstein coefficient of W^k G(P / W) over the
/// least weight coefficient to the k-th power. With the homogeneous coordinate
/// arrays, for callers that bound more.
struct Composed {
    g: f64,
    xs: Vec<Vec<f64>>,
    ys: Vec<Vec<f64>>,
    zs: Vec<Vec<f64>>,
    ws: Vec<Vec<f64>>,
    wmax: f64,
}

fn composed_bound(patch: &Patch, k: usize, mons: &[[u8; 3]], coef: &[f64], center: V3, scale: f64) -> Option<Composed> {
    let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
    let comp = |f: &dyn Fn(&[f64; 4]) -> f64| -> Vec<Vec<f64>> { (0..=d1).map(|i| (0..=d2).map(|j| f(&patch.pw[i][j])).collect()).collect() };
    let xs = comp(&|h| (h[0] - center.x * h[3]) / scale);
    let ys = comp(&|h| (h[1] - center.y * h[3]) / scale);
    let zs = comp(&|h| (h[2] - center.z * h[3]) / scale);
    let ws = comp(&|h| h[3]);
    let mut r = vec![vec![0.0; k * d2 + 1]; k * d1 + 1];
    for (m, c) in mons.iter().zip(coef) {
        if *c == 0.0 {
            continue;
        }
        let exps = [m[0] as usize, m[1] as usize, m[2] as usize, k - (m[0] + m[1] + m[2]) as usize];
        let mut prod: Vec<Vec<f64>> = vec![vec![1.0]];
        for (e, arr) in exps.iter().zip([&xs, &ys, &zs, &ws].iter()) {
            for _ in 0..*e {
                prod = bern_mul(&prod, arr);
            }
        }
        for i in 0..=k * d1 {
            for j in 0..=k * d2 {
                r[i][j] += c * prod[i][j];
            }
        }
    }
    let rmax = r.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);
    let wmin = ws.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
    let wmax = ws.iter().flatten().cloned().fold(0.0, f64::max);
    if !(wmin > 0.0) {
        return None;
    }
    Some(Composed { g: rmax / wmin.powi(k as i32), xs, ys, zs, ws, wmax })
}

pub fn certified_distance(surf: &crate::surface3::Surf, patch: &Patch) -> f64 {
    use crate::surface3::Surf;
    let (lo, hi) = patch.bbox();
    let center = (lo + hi) * 0.5;
    let scale = ((hi - lo).len() * 0.5).max(1e-300);
    let Some((k, mons, coef)) = implicit_of(surf, center, scale) else { return f64::INFINITY };
    let Some(cb) = composed_bound(patch, k, &mons, &coef, center, scale) else { return f64::INFINITY };
    let (d1, d2) = (patch.pw.len() - 1, patch.pw[0].len() - 1);
    let (g, xs, ys, zs, ws, wmax) = (cb.g, cb.xs, cb.ys, cb.zs, cb.ws, cb.wmax);
    let off = offset_rounding(surf_origin(surf), center, scale);
    off + match surf {
        Surf::Plane { .. } => g,
        Surf::Sphere { r, .. } | Surf::Cylinder { r, .. } => g / (r / scale),
        Surf::Cone { f, r, a } => {
            // Height above the apex along the axis, in the normalized frame.
            let apex = (f.o - f.z * (r / a.tan()) - center) * (1.0 / scale);
            let mut hmin = f64::INFINITY;
            for i in 0..=d1 {
                for j in 0..=d2 {
                    let hw = (xs[i][j] - apex.x * ws[i][j]) * f.z.x + (ys[i][j] - apex.y * ws[i][j]) * f.z.y + (zs[i][j] - apex.z * ws[i][j]) * f.z.z;
                    hmin = hmin.min(hw);
                }
            }
            if !(hmin > 0.0) {
                return f64::INFINITY;
            }
            g * a.cos() * a.cos() / ((hmin / wmax) * a.sin())
        }
        // G = (s - r)(s + r)((rho + R)^2 + h^2 - r^2), s the distance to the tube's
        // center circle, which for a ring torus (R > r) is the distance to the
        // torus |s - r|: so it is at most |G| / (r (R^2 - r^2)).
        Surf::Torus { major, minor, .. } if *major > *minor && *minor > 0.0 => {
            let (big, small) = (major / scale, minor / scale);
            g / (small * (big * big - small * small))
        }
        _ => f64::INFINITY,
    }
}

/// The distance error, relative to the span's size, of working in a frame whose
/// origin is |o| span sizes from the span: coordinates relative to it carry about
/// |o| epsilon, and the expanded coefficients the same order once divided by the
/// gradient; 16 covers the operations between them.
fn offset_rounding(origin: V3, center: V3, scale: f64) -> f64 {
    16.0 * f64::EPSILON * (1.0 + ((origin - center) * (1.0 / scale)).len())
}

fn surf_origin(surf: &crate::surface3::Surf) -> V3 {
    use crate::surface3::Surf;
    match surf {
        Surf::Plane { f } | Surf::Sphere { f, .. } | Surf::Cylinder { f, .. } | Surf::Cone { f, .. } | Surf::Torus { f, .. } => f.o,
        Surf::Nurbs(ns) => {
            let (lo, hi) = ns.bbox();
            (lo + hi) * 0.5
        }
    }
}

/// Whether a span is an analytic surface: its certified distance from it is
/// within `ROUNDING` of the span's size and within `KERNEL_RESOLUTION_MM`.
pub fn within_rounding(surf: &crate::surface3::Surf, patch: &Patch) -> bool {
    // A point of the span farther from the surface than the limit settles it at once
    // (the certified bound is at least that point's distance): sampled first, with four
    // times the limit and the rounding of the distance as margin, so a span the
    // certificate would accept is never turned away here.
    let (lo, hi) = patch.bbox();
    let limit = (ROUNDING * ((hi - lo).len() * 0.5).max(1e-300)).min(KERNEL_RESOLUTION_MM);
    for i in 0..=4 {
        for j in 0..=4 {
            let x = patch.eval(i as f64 / 4.0, j as f64 / 4.0).p;
            if surf.dist(x) > 4.0 * limit + 1e-13 * (x.len() + 1.0) {
                return false;
            }
        }
    }
    certify(patch, &|p: &Patch| certified_distance(surf, p))
}

/// Whether a bound (relative to the size of the piece it is given) holds within
/// `ROUNDING` of the span's size and `KERNEL_RESOLUTION_MM` on the whole span:
/// on the span, or where it does not (a box holding a point where the bound's
/// gradient vanishes, such as a quadric's center), on each of its quarters, to
/// four levels, each against the whole span's lines.
fn certify(patch: &Patch, bound: &dyn Fn(&Patch) -> f64) -> bool {
    let (lo, hi) = patch.bbox();
    let scale = ((hi - lo).len() * 0.5).max(1e-300);
    let limit = (ROUNDING * scale).min(KERNEL_RESOLUTION_MM);
    fn go(p: &Patch, bound: &dyn Fn(&Patch) -> f64, limit: f64, depth: usize) -> bool {
        let (lo, hi) = p.bbox();
        let sc = ((hi - lo).len() * 0.5).max(1e-300);
        if bound(p) * sc <= limit {
            return true;
        }
        depth > 0 && p.quarters().iter().all(|q| go(q, bound, limit, depth - 1))
    }
    go(patch, bound, limit, 4)
}

/// The torus a span is nearest to, if it is a surface of revolution: the axis
/// is the line every normal line of a surface of revolution meets, a linear
/// condition on the axis's Plucker coordinates (direction a, moment m = a x o:
/// a . (n x p) + n . m = 0 for a point p with unit normal n), read as the least
/// singular vector over a grid of the span's points; the tube is the circle the
/// points make in the meridian half-plane (rho, h), read the same way from
/// rho^2 + h^2 + D rho + E h + F = 0. A candidate only: `recognize` certifies it.
pub fn torus_candidate(patch: &Patch) -> Option<crate::surface3::Surf> {
    let (lo, hi) = patch.bbox();
    let c0 = (lo + hi) * 0.5;
    let n = 7;
    let mut pts = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let e = patch.eval((i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64);
            let nn = e.ps.cross(e.pt);
            if nn.len() > 0.0 {
                pts.push((e.p - c0, nn.unit()));
            }
        }
    }
    if pts.len() < 12 {
        return None;
    }
    // Axis.
    let rows = pts.len();
    let mut a = vec![0.0; rows * 6];
    for (r, (p, nn)) in pts.iter().enumerate() {
        let np = nn.cross(*p);
        for (k, v) in [np.x, np.y, np.z, nn.x, nn.y, nn.z].iter().enumerate() {
            a[k * rows + r] = *v;
        }
    }
    let (sig, v) = hestenes(&mut a, rows, 6);
    let j0 = (0..6).min_by(|&x, &y| sig[x].partial_cmp(&sig[y]).unwrap())?;
    let dir = V3::new(v[j0 * 6], v[j0 * 6 + 1], v[j0 * 6 + 2]);
    let mom = V3::new(v[j0 * 6 + 3], v[j0 * 6 + 4], v[j0 * 6 + 5]);
    let l2 = dir.dot(dir);
    if !(l2 > 0.0) {
        return None;
    }
    let o = mom.cross(dir) * (1.0 / l2);
    let z = dir * (1.0 / l2.sqrt());
    // Meridian circle.
    let mut b = vec![0.0; rows * 4];
    for (r, (p, _)) in pts.iter().enumerate() {
        let d = *p - o;
        let h = d.dot(z);
        let rho = (d - z * h).len();
        for (k, val) in [rho * rho + h * h, rho, h, 1.0].iter().enumerate() {
            b[k * rows + r] = *val;
        }
    }
    let (sig, v) = hestenes(&mut b, rows, 4);
    let j0 = (0..4).min_by(|&x, &y| sig[x].partial_cmp(&sig[y]).unwrap())?;
    let (q, dd, ee, ff) = (v[j0 * 4], v[j0 * 4 + 1], v[j0 * 4 + 2], v[j0 * 4 + 3]);
    if q == 0.0 {
        return None;
    }
    let (big, h0) = (-0.5 * dd / q, -0.5 * ee / q);
    let r2 = big * big + h0 * h0 - ff / q;
    if !(r2 > 0.0) {
        return None;
    }
    Some(crate::surface3::Surf::Torus { f: frame_on(c0 + o + z * h0, z), major: big, minor: r2.sqrt() })
}

/// An upper bound on a span's distance from a quadric, relative to the span's
/// size: with |G| <= g on the span and |grad G| >= gamma over the span's box, the
/// line from a span point along grad G meets the quadric within 2 g / gamma when
/// 4 |A| g <= gamma^2 (A the quadratic part: G along the line is a quadratic in
/// the step), all in the normalized frame. Infinite when that does not hold.
pub fn certified_quadric_distance(qd: &crate::quadric::Quadric, patch: &Patch) -> f64 {
    let (lo, hi) = patch.bbox();
    let center = (lo + hi) * 0.5;
    let scale = ((hi - lo).len() * 0.5).max(1e-300);
    let coef = qd.implicit(center, scale);
    let mons = monomials(2);
    let Some(cb) = composed_bound(patch, 2, &mons, &coef, center, scale) else { return f64::INFINITY };
    let c = |e: [u8; 3]| mons.iter().position(|m| *m == e).map_or(0.0, |i| coef[i]);
    let a = [[c([2, 0, 0]), 0.5 * c([1, 1, 0]), 0.5 * c([1, 0, 1])], [0.5 * c([1, 1, 0]), c([0, 2, 0]), 0.5 * c([0, 1, 1])], [0.5 * c([1, 0, 1]), 0.5 * c([0, 1, 1]), c([0, 0, 2])]];
    let b = [c([1, 0, 0]), c([0, 1, 0]), c([0, 0, 1])];
    let anorm = sym3_eigen(a).0.iter().map(|x| x.abs()).fold(0.0, f64::max);
    // The span's box in the normalized frame: center and half widths.
    let (bl, bh) = ((lo - center) * (1.0 / scale), (hi - center) * (1.0 / scale));
    let (mid, half) = ([0.5 * (bl.x + bh.x), 0.5 * (bl.y + bh.y), 0.5 * (bl.z + bh.z)], [0.5 * (bh.x - bl.x), 0.5 * (bh.y - bl.y), 0.5 * (bh.z - bl.z)]);
    let mut g2 = 0.0;
    for k in 0..3 {
        let centre: f64 = b[k] + (0..3).map(|j| 2.0 * a[k][j] * mid[j]).sum::<f64>();
        let spread: f64 = (0..3).map(|j| (2.0 * a[k][j]).abs() * half[j]).sum();
        let least = if centre.abs() > spread { centre.abs() - spread } else { 0.0 };
        g2 += least * least;
    }
    let gamma = g2.sqrt();
    if !(gamma > 0.0) || 4.0 * anorm * cb.g > gamma * gamma {
        return f64::INFINITY;
    }
    2.0 * cb.g / gamma + offset_rounding(qd.f.o, center, scale)
}

/// The quadric that is not a STEP kind a single span is, if it is one: the
/// candidates of `recognize_quadric` read off this span and certified on it.
pub fn quadric_of_span(patch: &Patch) -> Option<crate::quadric::Quadric> {
    let ns = crate::surface3::NurbsSurf::new(
        patch.pw.len() - 1,
        patch.pw[0].len() - 1,
        &patch.pw.iter().map(|r| r.iter().map(|h| V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])).collect()).collect::<Vec<_>>(),
        Some(&patch.pw.iter().map(|r| r.iter().map(|h| h[3]).collect()).collect::<Vec<_>>()),
        &[vec![0.0; patch.pw.len()], vec![1.0; patch.pw.len()]].concat(),
        &[vec![0.0; patch.pw[0].len()], vec![1.0; patch.pw[0].len()]].concat(),
    )
    .ok()?;
    recognize_quadric(&ns)
}

/// The quadric that is not a plane, sphere, circular cylinder or circular cone
/// (`crate::quadric`) a whole B-spline surface is, if every span is within
/// rounding of it, oriented so G grows along the B-spline's Su x Sv. Candidates
/// from the first span's degree-2 polynomial in its eigenframe, with its two
/// least eigenvalues, its least, or none set to zero (the cylinders and
/// paraboloids before the central kinds), each certified on every span
/// (`certified_quadric_distance`, within `ROUNDING` and `KERNEL_RESOLUTION_MM`).
pub fn recognize_quadric(ns: &crate::surface3::NurbsSurf) -> Option<crate::quadric::Quadric> {
    use crate::quadric::Quadric;
    let (nu, nv) = ns.spans_uv();
    if nu == 0 || nv == 0 {
        return None;
    }
    let p0 = ns.patch(0, 0);
    let sp = SpanPoly::quadric_of(&p0);
    let (center, scale) = (sp.center, sp.scale);
    let c = |e: [u8; 3]| sp.mons.iter().position(|m| *m == e).map_or(0.0, |i| sp.coef[i]);
    let a = [[c([2, 0, 0]), 0.5 * c([1, 1, 0]), 0.5 * c([1, 0, 1])], [0.5 * c([1, 1, 0]), c([0, 2, 0]), 0.5 * c([0, 1, 1])], [0.5 * c([1, 0, 1]), 0.5 * c([0, 1, 1]), c([0, 0, 2])]];
    let b = V3::new(c([1, 0, 0]), c([0, 1, 0]), c([0, 0, 1]));
    let c0 = c([0, 0, 0]);
    let (lam, v) = sym3_eigen(a);
    let ev = |i: usize| V3::new(v[0][i], v[1][i], v[2][i]);
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&x, &y| lam[x].abs().partial_cmp(&lam[y].abs()).unwrap());
    let lmax = lam.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if !(lmax > 0.0) {
        return None;
    }
    let mut cands: Vec<Quadric> = Vec::new();
    // A candidate whose origin is farther than this from the span (in span sizes)
    // cannot hold the rounding line in its own coordinates: their rounding alone,
    // the offset times the machine epsilon, would exceed it.
    let far = ROUNDING / f64::EPSILON;
    for zeros in [2usize, 1, 0] {
        let zero: Vec<usize> = order[..zeros].to_vec();
        let keep: Vec<usize> = order[zeros..].to_vec();
        // In the eigenframe y = V' q: sum lam y^2 + beta . y + c0; complete the squares.
        let beta: Vec<f64> = (0..3).map(|i| ev(i).dot(b)).collect();
        let mut base = V3::new(0.0, 0.0, 0.0);
        let mut c1 = c0;
        for &i in &keep {
            base = base - ev(i) * (beta[i] / (2.0 * lam[i]));
            c1 -= beta[i] * beta[i] / (4.0 * lam[i]);
        }
        // What is left of the linear part, in the zero eigenvalues' span. Each
        // zero set is read without it (the cylinders) and with it (the paraboloids).
        let lin = zero.iter().fold(V3::new(0.0, 0.0, 0.0), |acc, &i| acc + ev(i) * beta[i]);
        for use_lin in [false, true] {
            if use_lin && (zero.is_empty() || !(lin.len() > 0.0)) {
                continue;
            }
            let mut origin = base;
            let (mut l, mut cc) = (0.0, c1);
            let mut axes: Vec<V3> = keep.iter().map(|&i| ev(i)).collect();
            let mut qs: Vec<f64> = keep.iter().map(|&i| lam[i]).collect();
            if use_lin {
                let zdir = lin.unit();
                l = lin.len();
                origin = origin - zdir * (c1 / l);
                cc = 0.0;
                if zero.len() == 2 {
                    // The kept eigenvector is normal to the zero plane, so this lies in it.
                    axes.push(zdir.cross(axes[0]).unit());
                    qs.push(0.0);
                }
                axes.push(zdir);
                qs.push(0.0);
            } else {
                for &i in &zero {
                    axes.push(ev(i));
                    qs.push(0.0);
                }
            }
            if axes.len() != 3 || !(origin.len() <= far) {
                continue;
            }
            // A right-handed frame with axes[2] as z and axes[0] as x; axes[1] is
            // +-(z x x), and its coefficient is quadratic, so the sign does not matter.
            let z = axes[2].unit();
            let x = (axes[0] - z * axes[0].dot(z)).unit();
            // To world units: q / scale^2, l / scale, c unchanged, then scaled to a
            // unit largest quadratic coefficient.
            let mut qd = Quadric { f: crate::curve3::Frame { o: center + origin * scale, z, x }, q: [qs[0] / (scale * scale), qs[1] / (scale * scale), qs[2] / (scale * scale)], l: l / scale, c: cc };
            let m = qd.q.iter().map(|x| x.abs()).fold(0.0, f64::max);
            if !(m > 0.0) {
                continue;
            }
            qd.q = [qd.q[0] / m, qd.q[1] / m, qd.q[2] / m];
            qd.l /= m;
            qd.c /= m;
            if matches!(qd.name(), "point" | "empty" | "two planes" | "parallel planes" | "plane") {
                continue;
            }
            cands.push(qd);
        }
    }
    'cand: for mut qd in cands {
        for i in 0..nu {
            for j in 0..nv {
                if !certify(&ns.patch(i, j), &|p: &Patch| certified_quadric_distance(&qd, p)) {
                    continue 'cand;
                }
            }
        }
        let e = p0.eval(0.5, 0.5);
        if qd.grad(e.p).dot(e.ps.cross(e.pt)) < 0.0 {
            qd.q = [-qd.q[0], -qd.q[1], -qd.q[2]];
            qd.l = -qd.l;
            qd.c = -qd.c;
        }
        return Some(qd);
    }
    None
}

/// The plane, sphere, circular cylinder or circular cone a whole B-spline surface
/// is, if every span is within rounding of it (candidates read off the first
/// span's degree-1 and degree-2 polynomials, then certified on each span), and
/// whether the face's sense flips because the analytic surface's outward normal
/// runs against the B-spline's Su x Sv. None when no candidate holds on every span.
pub fn recognize(ns: &crate::surface3::NurbsSurf) -> Option<(crate::surface3::Surf, bool)> {
    let (nu, nv) = ns.spans_uv();
    if nu == 0 || nv == 0 {
        return None;
    }
    let p0 = ns.patch(0, 0);
    let (lo, hi) = p0.bbox();
    let (center, scale) = ((lo + hi) * 0.5, ((hi - lo).len() * 0.5).max(1e-300));
    let mut cands: Vec<crate::surface3::Surf> = Vec::new();
    for k in 1..=2usize {
        let (step, coef) = SpanPoly::system(&p0, k, center, scale);
        let sp = SpanPoly { degree: k, coef, mons: monomials(k), center, scale, sigma: step.sigma, sigma_next: step.sigma_next, sigma_below: f64::NAN, steps: vec![step] };
        cands.extend(sp.analytic_candidates());
    }
    if let Some(t) = torus_candidate(&p0) {
        cands.push(t);
    }
    'cand: for surf in cands {
        for i in 0..nu {
            for j in 0..nv {
                if !within_rounding(&surf, &ns.patch(i, j)) {
                    continue 'cand;
                }
            }
        }
        let e = p0.eval(0.5, 0.5);
        let flip = surf.normal(e.p).dot(e.ps.cross(e.pt)) < 0.0;
        return Some((surf, flip));
    }
    None
}

#[cfg(test)]
mod recognize_tests {
    use super::*;
    use crate::surface3::{NurbsSurf, Surf};

    const W: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const WB: f64 = 0.707106781186707; // bull.stp's square root of a half

    /// A quarter sphere (two octants across u) as one rational B-spline surface of
    /// two spans, written with weight w.
    fn sphere(w: f64, r: f64, c: V3) -> NurbsSurf {
        // Meridian (u): equator to pole; turn (v): two quarter turns.
        let mer = [(r, 0.0, 1.0), (r, r, w), (0.0, r, 1.0)];
        let turn = [(1.0, 0.0, 1.0), (1.0, 1.0, w), (0.0, 1.0, 1.0), (-1.0, 1.0, w), (-1.0, 0.0, 1.0)];
        let ctrl: Vec<Vec<V3>> = mer.iter().map(|&(rho, z, _)| turn.iter().map(|&(x, y, _)| V3::new(c.x + rho * x, c.y + rho * y, c.z + z)).collect()).collect();
        let wts: Vec<Vec<f64>> = mer.iter().map(|&(_, _, wm)| turn.iter().map(|&(_, _, wt)| wm * wt).collect()).collect();
        NurbsSurf::new(2, 2, &ctrl, Some(&wts), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]).unwrap()
    }

    #[test]
    fn a_rational_sphere_surface_is_the_sphere() {
        for w in [W, WB] {
            let (r, c) = (4.0, V3::new(1.0, 2.0, -3.0));
            let ns = sphere(w, r, c);
            let (s, _) = recognize(&ns).unwrap_or_else(|| panic!("weight {w}: not recognized"));
            match s {
                Surf::Sphere { f, r: rr } => assert!((f.o - c).len() < 1e-11 && (rr - r).abs() < 1e-11, "{:?} {rr}", f.o),
                other => panic!("read as {}", other.name()),
            }
        }
    }

    /// A quarter of a torus (major R, minor r) about z through c as one rational
    /// B-spline surface of two spans along the turn.
    fn torus(w: f64, big: f64, r: f64, c: V3) -> NurbsSurf {
        let mer = [(big + r, 0.0, 1.0), (big + r, r, w), (big, r, 1.0)];
        let turn = [(1.0, 0.0, 1.0), (1.0, 1.0, w), (0.0, 1.0, 1.0), (-1.0, 1.0, w), (-1.0, 0.0, 1.0)];
        let ctrl: Vec<Vec<V3>> = mer.iter().map(|&(rho, z, _)| turn.iter().map(|&(x, y, _)| V3::new(c.x + rho * x, c.y + rho * y, c.z + z)).collect()).collect();
        let wts: Vec<Vec<f64>> = mer.iter().map(|&(_, _, wm)| turn.iter().map(|&(_, _, wt)| wm * wt).collect()).collect();
        NurbsSurf::new(2, 2, &ctrl, Some(&wts), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]).unwrap()
    }

    #[test]
    fn a_rational_torus_surface_is_the_torus() {
        for w in [W, WB] {
            let (big, r, c) = (20.0, 5.0, V3::new(3.0, -1.0, 2.0));
            let (s, flip) = recognize(&torus(w, big, r, c)).unwrap_or_else(|| panic!("weight {w}: not recognized"));
            match s {
                Surf::Torus { f, major, minor } => {
                    assert!((f.o - c).len() < 1e-9 && (major - big).abs() < 1e-9 && (minor - r).abs() < 1e-9 && f.z.cross(V3::new(0.0, 0.0, 1.0)).len() < 1e-9, "{:?} {major} {minor}", f);
                }
                other => panic!("read as {}", other.name()),
            }
            // Su x Sv of this net points into the tube; the torus's normal points out.
            let e = torus(w, big, r, c).patch(0, 0).eval(0.5, 0.5);
            let out = e.p - (c + V3::new(e.p.x - c.x, e.p.y - c.y, 0.0).unit() * big);
            assert_eq!(flip, out.dot(e.ps.cross(e.pt)) < 0.0);
        }
    }

    #[test]
    fn a_torus_fit_is_not_the_torus() {
        let ns = torus(W, 20.0, 5.0, V3::new(0.0, 0.0, 0.0));
        let (d1, d2) = (2, 2);
        let mut ctrl: Vec<Vec<V3>> = Vec::new();
        let mut wts: Vec<Vec<f64>> = Vec::new();
        let p = ns.patch(0, 0);
        for i in 0..=d1 {
            ctrl.push((0..=d2).map(|j| { let h = p.pw[i][j]; V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]) }).collect());
            wts.push((0..=d2).map(|j| p.pw[i][j][3]).collect());
        }
        ctrl[1][1] = ctrl[1][1] + V3::new(0.0, 0.0, 5e-6);
        let fit = NurbsSurf::new(2, 2, &ctrl, Some(&wts), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(recognize(&fit).is_none());
    }

    #[test]
    fn a_twisted_bilinear_surface_is_a_hyperbolic_paraboloid() {
        // Two bilinear spans of z = x y over a tilted frame.
        let f = |x: f64, y: f64| V3::new(x + 0.3 * y, y - 0.2 * x, x * y + 0.1 * x);
        let ctrl: Vec<Vec<V3>> = (0..2).map(|i| (0..3).map(|j| f(i as f64 * 2.0 - 1.0, j as f64 - 1.0)).collect()).collect();
        let ns = NurbsSurf::new(1, 1, &ctrl, None, &[0.0, 0.0, 1.0, 1.0], &[0.0, 0.0, 0.5, 1.0, 1.0]).unwrap();
        assert!(recognize(&ns).is_none());
        let qd = recognize_quadric(&ns).expect("a hyperbolic paraboloid");
        assert_eq!(qd.name(), "hyperbolic paraboloid");
        // Every point of the surface is on it, and the distance off it is exact.
        for (s, t) in [(0.1, 0.2), (0.7, 0.9), (0.5, 0.5)] {
            let p = ns.eval(s, t);
            assert!((qd.foot(p.p) - p.p).len() < 1e-9);
            let n = p.ps.cross(p.pt).unit();
            assert!((qd.signed_distance(p.p + n * 0.01, 1.0) - 0.01).abs() < 1e-6, "{}", qd.signed_distance(p.p + n * 0.01, 1.0));
        }
    }

    #[test]
    fn a_ruled_hyperboloid_patch_is_the_hyperboloid() {
        // x^2 + y^2 - z^2 = 1 holds the lines through (cos a, sin a, 0) along
        // (-sin a, cos a, 1): a bilinear patch between two of its rulings... is not
        // bilinear in general, so build a rational (2,1) patch: the circle at z = 0
        // (radius 1) and at z = 1 (radius sqrt 2), joined along rulings.
        let w = W;
        let (r0, r1) = (1.0f64, 2.0f64.sqrt());
        // The ruling from angle a at z = 0 reaches angle a + 45 degrees at z = 1.
        let arc = |r: f64, a0: f64| -> Vec<(V3, f64)> {
            let (c0, s0) = (a0.cos(), a0.sin());
            let (c1, s1) = ((a0 + std::f64::consts::FRAC_PI_2).cos(), (a0 + std::f64::consts::FRAC_PI_2).sin());
            let mid = V3::new(c0 + c1, s0 + s1, 0.0) * r;
            vec![(V3::new(c0, s0, 0.0) * r, 1.0), (mid, w), (V3::new(c1, s1, 0.0) * r, 1.0)]
        };
        let bottom = arc(r0, 0.0);
        let top: Vec<(V3, f64)> = arc(r1, std::f64::consts::FRAC_PI_4).into_iter().map(|(p, wt)| (p + V3::new(0.0, 0.0, 1.0), wt)).collect();
        let ctrl: Vec<Vec<V3>> = (0..3).map(|i| vec![bottom[i].0, top[i].0]).collect();
        let wts: Vec<Vec<f64>> = (0..3).map(|i| vec![bottom[i].1, top[i].1]).collect();
        let ns = NurbsSurf::new(2, 1, &ctrl, Some(&wts), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 1.0, 1.0]).unwrap();
        assert!(recognize(&ns).is_none());
        let qd = recognize_quadric(&ns).expect("the hyperboloid");
        assert_eq!(qd.name(), "hyperboloid of one sheet");
        for (s, t) in [(0.2, 0.3), (0.8, 0.6)] {
            let p = ns.eval(s, t).p;
            assert!((qd.foot(p) - p).len() < 1e-9, "{}", qd.name());
            assert!((p.x * p.x + p.y * p.y - p.z * p.z - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn a_fit_is_not_the_sphere() {
        // The same sphere with one control point moved by a millionth of the radius:
        // close, but not the sphere.
        let (r, c) = (4.0, V3::new(1.0, 2.0, -3.0));
        let mer = [(r, 0.0, 1.0), (r, r, W), (0.0, r, 1.0)];
        let turn = [(1.0, 0.0, 1.0), (1.0, 1.0, W), (0.0, 1.0, 1.0)];
        let mut ctrl: Vec<Vec<V3>> = mer.iter().map(|&(rho, z, _)| turn.iter().map(|&(x, y, _)| V3::new(c.x + rho * x, c.y + rho * y, c.z + z)).collect()).collect();
        ctrl[1][1] = ctrl[1][1] + V3::new(0.0, 0.0, 1e-6 * r);
        let wts: Vec<Vec<f64>> = mer.iter().map(|&(_, _, wm)| turn.iter().map(|&(_, _, wt)| wm * wt).collect()).collect();
        let ns = NurbsSurf::new(2, 2, &ctrl, Some(&wts), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(recognize(&ns).is_none());
    }

    #[test]
    fn the_orientation_follows_the_b_spline_normal() {
        let ns = sphere(W, 4.0, V3::new(0.0, 0.0, 0.0));
        let (s, flip) = recognize(&ns).unwrap();
        // The face's outward side is Su x Sv; the sphere's is away from its center.
        let p = ns.patch(0, 0);
        let e = p.eval(0.5, 0.5);
        let n = e.ps.cross(e.pt);
        assert_eq!(flip, s.normal(e.p).dot(n) < 0.0);
        assert_eq!(flip, e.p.dot(n) < 0.0);
    }
}
