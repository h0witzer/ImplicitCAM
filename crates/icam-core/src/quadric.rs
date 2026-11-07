//! Quadrics beyond the STEP kinds: ellipsoids, hyperboloids of one and two
//! sheets, elliptic cones, elliptic and hyperbolic paraboloids, and elliptic,
//! hyperbolic and parabolic cylinders, with their exact signed distance.
//!
//! A quadric is kept in canonical form in a local frame (origin o, axes x, y, z):
//! G(u) = q0 u0^2 + q1 u1^2 + q2 u2^2 + l u2 + c, u the point's local coordinates,
//! with l nonzero only when q2 is zero (the paraboloids' and the parabolic
//! cylinder's axis is local z). B-spline faces whose every span differs from such
//! a quadric only by the rounding of its numbers are read as it
//! (`span_poly::recognize_quadric`, the standard of `span_poly::ROUNDING`; Derek,
//! 2026-10-25: such a span is the analytic surface, and these "have an exact form
//! we can use").
//!
//! The distance is exact: the nearest point solves u - p = -(mu / 2) grad G(u),
//! so u_i = p_i / (1 + mu q_i) (and u2 = p2 - mu l / 2 on a paraboloid's axis),
//! and G(u(mu)) = 0 is, cleared of its denominators, a polynomial in mu of degree
//! at most 6. Its real roots are all found (`real_roots`: between consecutive
//! roots of its derivative a polynomial is monotone, so each holds at most one
//! root, found by bracketing), each gives a critical point of the distance, and
//! the points where a denominator vanishes (a point on a plane of symmetry, whose
//! foot can leave it) are added; the nearest of them is the foot. No search over
//! the surface, no starting guess.

use crate::curve3::{Frame, V3};

/// A quadric in canonical form; see the module notes.
#[derive(Clone, Debug)]
pub struct Quadric {
    pub f: Frame,
    pub q: [f64; 3],
    pub l: f64,
    pub c: f64,
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            c[i + j] += x * y;
        }
    }
    c
}

fn poly_add(a: &mut Vec<f64>, b: &[f64], k: f64) {
    if a.len() < b.len() {
        a.resize(b.len(), 0.0);
    }
    for (i, y) in b.iter().enumerate() {
        a[i] += k * y;
    }
}

fn poly_eval(a: &[f64], x: f64) -> f64 {
    a.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

/// Every real root of a polynomial (coefficients from the constant term up),
/// ascending. The roots of its derivative split the line into intervals on which
/// it is monotone; an interval whose ends differ in sign holds exactly one root,
/// found by the Illinois variant of regula falsi; the outer intervals end at the
/// Cauchy bound 1 + max |a_i / a_n|.
pub fn real_roots(a: &[f64]) -> Vec<f64> {
    let mut a = a.to_vec();
    let scale = a.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if !(scale > 0.0) {
        return Vec::new();
    }
    // Drop leading coefficients that are zero to the arithmetic of the others.
    while a.len() > 1 && a[a.len() - 1].abs() <= 1e-15 * scale {
        a.pop();
    }
    let n = a.len() - 1;
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![-a[0] / a[1]];
    }
    let bound = 1.0 + a[..n].iter().map(|x| (x / a[n]).abs()).fold(0.0, f64::max);
    let d: Vec<f64> = (1..=n).map(|i| a[i] * i as f64).collect();
    let mut cuts = vec![-bound];
    cuts.extend(real_roots(&d).into_iter().filter(|x| x.abs() < bound));
    cuts.push(bound);
    let mut roots = Vec::new();
    for w in cuts.windows(2) {
        let (mut lo, mut hi) = (w[0], w[1]);
        let (mut flo, mut fhi) = (poly_eval(&a, lo), poly_eval(&a, hi));
        if flo == 0.0 {
            roots.push(lo);
            continue;
        }
        if (flo < 0.0) == (fhi < 0.0) {
            // No sign change; a root of even multiplicity at a cut is caught there.
            continue;
        }
        let mut side = 0i32;
        let mut x = lo;
        for _ in 0..200 {
            x = (lo * fhi - hi * flo) / (fhi - flo);
            let fx = poly_eval(&a, x);
            if fx == 0.0 || (hi - lo).abs() <= 4.0 * f64::EPSILON * x.abs().max(1e-300) {
                break;
            }
            if (fx < 0.0) == (flo < 0.0) {
                lo = x;
                flo = fx;
                if side == -1 {
                    fhi *= 0.5;
                }
                side = -1;
            } else {
                hi = x;
                fhi = fx;
                if side == 1 {
                    flo *= 0.5;
                }
                side = 1;
            }
        }
        roots.push(x);
    }
    // A double root sits at a root of the derivative, where the sign does not change.
    for &x in &cuts[1..cuts.len() - 1] {
        let fx = poly_eval(&a, x);
        let mag: f64 = a.iter().enumerate().map(|(i, c)| (c * x.abs().powi(i as i32)).abs()).sum();
        if fx.abs() <= 1e-12 * mag && !roots.iter().any(|r| (r - x).abs() <= 1e-9 * x.abs().max(1.0)) {
            roots.push(x);
        }
    }
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    roots
}

impl Quadric {
    fn local(&self, x: V3) -> [f64; 3] {
        let d = x - self.f.o;
        [d.dot(self.f.x), d.dot(self.f.y()), d.dot(self.f.z)]
    }
    fn world(&self, u: [f64; 3]) -> V3 {
        self.f.o + self.f.x * u[0] + self.f.y() * u[1] + self.f.z * u[2]
    }

    /// G at x.
    pub fn value(&self, x: V3) -> f64 {
        let u = self.local(x);
        self.q[0] * u[0] * u[0] + self.q[1] * u[1] * u[1] + self.q[2] * u[2] * u[2] + self.l * u[2] + self.c
    }

    /// The gradient of G at x.
    pub fn grad(&self, x: V3) -> V3 {
        let u = self.local(x);
        self.f.x * (2.0 * self.q[0] * u[0]) + self.f.y() * (2.0 * self.q[1] * u[1]) + self.f.z * (2.0 * self.q[2] * u[2] + self.l)
    }

    /// The kind, by the signs of the coefficients.
    pub fn name(&self) -> &'static str {
        let nz: Vec<f64> = self.q.iter().cloned().filter(|v| *v != 0.0).collect();
        let pos = nz.iter().filter(|v| **v > 0.0).count();
        let mixed = pos != 0 && pos != nz.len();
        match (nz.len(), self.l != 0.0) {
            (3, _) => {
                if self.c == 0.0 {
                    if mixed { "elliptic cone" } else { "point" }
                } else if !mixed {
                    if (self.c < 0.0) == (pos == 3) { "ellipsoid" } else { "empty" }
                } else {
                    // The constant's sign against the majority's: one sheet when it opposes it.
                    let majority_pos = pos == 2;
                    if (self.c < 0.0) == majority_pos { "hyperboloid of one sheet" } else { "hyperboloid of two sheets" }
                }
            }
            (2, true) => if mixed { "hyperbolic paraboloid" } else { "elliptic paraboloid" },
            (2, false) => {
                if self.c == 0.0 {
                    "two planes"
                } else if mixed {
                    "hyperbolic cylinder"
                } else if (self.c < 0.0) == (pos == 2) {
                    "elliptic cylinder"
                } else {
                    "empty"
                }
            }
            (1, true) => "parabolic cylinder",
            (1, false) => "parallel planes",
            _ => "plane",
        }
    }

    /// Every critical point of the distance from x on the quadric: the roots of
    /// the Lagrange equation and the points where a denominator vanishes.
    pub fn crit(&self, x: V3) -> Vec<V3> {
        let p = self.local(x);
        let (q, l, c) = (self.q, self.l, self.c);
        // P(mu) = sum_i q_i p_i^2 prod_{j != i} D_j^2 + (l p2 - mu l^2 / 2 + c) prod_j D_j^2,
        // D_i = 1 + mu q_i over the axes with q_i nonzero (with l, axis 2 has q2 = 0).
        let axes: Vec<usize> = (0..3).filter(|&i| q[i] != 0.0).collect();
        let d2 = |i: usize| -> Vec<f64> { poly_mul(&[1.0, q[i]], &[1.0, q[i]]) };
        let prod_except = |skip: Option<usize>| -> Vec<f64> {
            let mut r = vec![1.0];
            for &j in &axes {
                if Some(j) != skip {
                    r = poly_mul(&r, &d2(j));
                }
            }
            r
        };
        let mut poly = vec![0.0];
        for &i in &axes {
            poly_add(&mut poly, &prod_except(Some(i)), q[i] * p[i] * p[i]);
        }
        let all = prod_except(None);
        let rest: Vec<f64> = if l != 0.0 { vec![l * p[2] + c, -0.5 * l * l] } else { vec![c] };
        poly_add(&mut poly, &poly_mul(&rest, &all), 1.0);
        let mut out = Vec::new();
        for mu in real_roots(&poly) {
            let mut u = [0.0; 3];
            let mut ok = true;
            for i in 0..3 {
                if q[i] != 0.0 {
                    let den = 1.0 + mu * q[i];
                    if den == 0.0 {
                        ok = false;
                        break;
                    }
                    u[i] = p[i] / den;
                } else if i == 2 && l != 0.0 {
                    u[i] = p[i] - 0.5 * mu * l;
                } else {
                    u[i] = p[i];
                }
            }
            if ok {
                out.push(self.world(u));
            }
        }
        // mu = -1 / q_i: u_i is free, the other coordinates follow, u_i from G = 0.
        for &i in &axes {
            let mu = -1.0 / q[i];
            let mut u = [0.0; 3];
            let mut ok = true;
            let mut g_rest = c;
            for j in 0..3 {
                if j == i {
                    continue;
                }
                if q[j] != 0.0 {
                    let den = 1.0 + mu * q[j];
                    if den == 0.0 {
                        ok = false;
                        break;
                    }
                    u[j] = p[j] / den;
                    g_rest += q[j] * u[j] * u[j];
                } else if j == 2 && l != 0.0 {
                    u[j] = p[j] - 0.5 * mu * l;
                    g_rest += l * u[j];
                } else {
                    u[j] = p[j];
                }
            }
            let ui2 = -g_rest / q[i];
            if ok && ui2 >= 0.0 {
                for sgn in [1.0, -1.0] {
                    let mut v = u;
                    v[i] = sgn * ui2.sqrt();
                    out.push(self.world(v));
                }
            }
        }
        out
    }

    /// The nearest point of the quadric to x.
    pub fn foot(&self, x: V3) -> V3 {
        self.crit(x).into_iter().min_by(|a, b| (*a - x).len().partial_cmp(&(*b - x).len()).unwrap()).unwrap_or(x)
    }

    /// The distance from x to the quadric, signed like G times s.
    pub fn signed_distance(&self, x: V3, s: f64) -> f64 {
        let d = (self.foot(x) - x).len();
        let g = self.value(x) * s;
        if g < 0.0 { -d } else { d }
    }

    /// The unit normal at a point of the quadric, the direction G grows in.
    pub fn normal(&self, x: V3) -> V3 {
        self.grad(x).unit()
    }

    /// G's coefficients over `span_poly::monomials(2)` in the normalized frame
    /// q = (p - center) / scale, for certification.
    pub fn implicit(&self, center: V3, scale: f64) -> Vec<f64> {
        // G(p) = sum_i q_i (e_i . (p - o))^2 + l (z . (p - o)) + c with p = center + scale q.
        let axes = [self.f.x, self.f.y(), self.f.z];
        let o = (self.f.o - center) * (1.0 / scale);
        let mut m = [[0.0; 3]; 3];
        for (k, e) in axes.iter().enumerate() {
            let ev = [e.x, e.y, e.z];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += self.q[k] * ev[i] * ev[j] * scale * scale;
                }
            }
        }
        let z = [self.f.z.x, self.f.z.y, self.f.z.z];
        let ov = [o.x, o.y, o.z];
        let mo: Vec<f64> = (0..3).map(|i| (0..3).map(|j| m[i][j] * ov[j]).sum()).collect();
        let omo: f64 = (0..3).map(|i| ov[i] * mo[i]).sum();
        let lz: Vec<f64> = (0..3).map(|i| self.l * scale * z[i]).collect();
        let lzo: f64 = (0..3).map(|i| lz[i] * ov[i]).sum();
        crate::span_poly::monomials(2)
            .iter()
            .map(|e| match *e {
                [0, 0, 0] => omo - lzo + self.c,
                [1, 0, 0] => -2.0 * mo[0] + lz[0],
                [0, 1, 0] => -2.0 * mo[1] + lz[1],
                [0, 0, 1] => -2.0 * mo[2] + lz[2],
                [2, 0, 0] => m[0][0],
                [0, 2, 0] => m[1][1],
                [0, 0, 2] => m[2][2],
                [1, 1, 0] => 2.0 * m[0][1],
                [1, 0, 1] => 2.0 * m[0][2],
                _ => 2.0 * m[1][2],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        // A tilted frame, so nothing lines up with the world axes.
        let z = V3::new(0.3, -0.4, 0.866).unit();
        let x = (V3::new(1.0, 0.2, 0.0) - z * V3::new(1.0, 0.2, 0.0).dot(z)).unit();
        Frame { o: V3::new(2.0, -1.0, 3.0), z, x }
    }

    /// The least distance from x to a dense sample of a parameterized surface,
    /// refined around the best sample by a local grid.
    fn brute(x: V3, s: &dyn Fn(f64, f64) -> V3, (u0, u1): (f64, f64), (v0, v1): (f64, f64)) -> f64 {
        let n = 400;
        let mut best = (f64::INFINITY, 0.0, 0.0);
        for i in 0..=n {
            for j in 0..=n {
                let (u, v) = (u0 + (u1 - u0) * i as f64 / n as f64, v0 + (v1 - v0) * j as f64 / n as f64);
                let d = (s(u, v) - x).len();
                if d < best.0 {
                    best = (d, u, v);
                }
            }
        }
        let (mut du, mut dv) = ((u1 - u0) / n as f64, (v1 - v0) / n as f64);
        for _ in 0..40 {
            let (_, bu, bv) = best;
            for i in -4..=4 {
                for j in -4..=4 {
                    let (u, v) = (bu + du * i as f64 / 4.0, bv + dv * j as f64 / 4.0);
                    let d = (s(u, v) - x).len();
                    if d < best.0 {
                        best = (d, u, v);
                    }
                }
            }
            du *= 0.5;
            dv *= 0.5;
        }
        best.0
    }

    fn check(q: &Quadric, s: &dyn Fn(f64, f64) -> V3, ur: (f64, f64), vr: (f64, f64)) {
        // The parameterization lies on the quadric.
        for i in 0..5 {
            for j in 0..5 {
                let p = s(ur.0 + (ur.1 - ur.0) * i as f64 / 4.0, vr.0 + (vr.1 - vr.0) * j as f64 / 4.0);
                assert!(q.value(p).abs() < 1e-9 * (1.0 + (p - q.f.o).len().powi(2)), "{} off its surface", q.name());
            }
        }
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut u = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for _ in 0..40 {
            let x = q.f.o + V3::new(u(), u(), u()) * 2.5;
            let exact = (q.foot(x) - x).len();
            let b = brute(x, s, ur, vr);
            assert!(exact <= b + 1e-9, "{}: foot {exact} beyond the sampled {b} at {x:?}", q.name());
            assert!(b - exact < 1e-6, "{}: foot {exact} short of the sampled {b} at {x:?}", q.name());
            assert!(q.value(q.foot(x)).abs() < 1e-9, "{}: foot off the surface", q.name());
        }
    }

    #[test]
    fn roots() {
        // (x - 1)(x + 2)(x - 3)(x^2 + 1) and a double root.
        let a = poly_mul(&poly_mul(&poly_mul(&[-1.0, 1.0], &[2.0, 1.0]), &[-3.0, 1.0]), &[1.0, 0.0, 1.0]);
        let r = real_roots(&a);
        assert_eq!(r.len(), 3, "{r:?}");
        for (x, y) in r.iter().zip([-2.0, 1.0, 3.0]) {
            assert!((x - y).abs() < 1e-12, "{r:?}");
        }
        let r = real_roots(&poly_mul(&[-1.0, 1.0], &[-1.0, 1.0]));
        assert!(r.iter().any(|x| (x - 1.0).abs() < 1e-7), "{r:?}");
    }

    #[test]
    fn exact_distance_to_each_kind() {
        let f = frame();
        let at = |u: [f64; 3]| f.o + f.x * u[0] + f.y() * u[1] + f.z * u[2];
        let (a, b, c) = (1.3, 0.8, 1.1);
        // Ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1.
        let q = Quadric { f, q: [1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)], l: 0.0, c: -1.0 };
        assert_eq!(q.name(), "ellipsoid");
        check(&q, &|u, v| at([a * v.cos() * u.cos(), b * v.cos() * u.sin(), c * v.sin()]), (-3.2, 3.2), (-1.6, 1.6));
        // Hyperboloid of one sheet x^2/a^2 + y^2/b^2 - z^2/c^2 = 1.
        let q = Quadric { f, q: [1.0 / (a * a), 1.0 / (b * b), -1.0 / (c * c)], l: 0.0, c: -1.0 };
        assert_eq!(q.name(), "hyperboloid of one sheet");
        check(&q, &|u, v| at([a * v.cosh() * u.cos(), b * v.cosh() * u.sin(), c * v.sinh()]), (-3.2, 3.2), (-2.5, 2.5));
        // Hyperbolic paraboloid z = x^2/a^2 - y^2/b^2: x^2/a^2 - y^2/b^2 - z = 0.
        let q = Quadric { f, q: [1.0 / (a * a), -1.0 / (b * b), 0.0], l: -1.0, c: 0.0 };
        assert_eq!(q.name(), "hyperbolic paraboloid");
        check(&q, &|u, v| at([u, v, u * u / (a * a) - v * v / (b * b)]), (-6.0, 6.0), (-6.0, 6.0));
        // Elliptic paraboloid.
        let q = Quadric { f, q: [1.0 / (a * a), 1.0 / (b * b), 0.0], l: -1.0, c: 0.0 };
        assert_eq!(q.name(), "elliptic paraboloid");
        check(&q, &|u, v| at([u, v, u * u / (a * a) + v * v / (b * b)]), (-6.0, 6.0), (-6.0, 6.0));
        // Elliptic cylinder.
        let q = Quadric { f, q: [1.0 / (a * a), 1.0 / (b * b), 0.0], l: 0.0, c: -1.0 };
        assert_eq!(q.name(), "elliptic cylinder");
        check(&q, &|u, v| at([a * u.cos(), b * u.sin(), v]), (-3.2, 3.2), (-6.0, 6.0));
        // Hyperbolic cylinder x^2/a^2 - y^2/b^2 = 1, both branches.
        let q = Quadric { f, q: [1.0 / (a * a), -1.0 / (b * b), 0.0], l: 0.0, c: -1.0 };
        assert_eq!(q.name(), "hyperbolic cylinder");
        let both = |u: f64, v: f64| if u >= 0.0 { at([a * (u - 3.0).cosh(), b * (u - 3.0).sinh(), v]) } else { at([-a * (u + 3.0).cosh(), b * (u + 3.0).sinh(), v]) };
        check(&q, &both, (-6.0, 6.0), (-6.0, 6.0));
        // Parabolic cylinder z = x^2 / a^2.
        let q = Quadric { f, q: [1.0 / (a * a), 0.0, 0.0], l: -1.0, c: 0.0 };
        assert_eq!(q.name(), "parabolic cylinder");
        check(&q, &|u, v| at([u, v, u * u / (a * a)]), (-6.0, 6.0), (-6.0, 6.0));
    }

    #[test]
    #[ignore]
    fn rate() {
        let f = frame();
        let kinds = [
            Quadric { f, q: [0.6, 1.5, 0.8], l: 0.0, c: -1.0 },
            Quadric { f, q: [0.6, 1.5, -0.8], l: 0.0, c: -1.0 },
            Quadric { f, q: [0.6, -1.5, 0.0], l: -1.0, c: 0.0 },
            Quadric { f, q: [0.6, 1.5, 0.0], l: 0.0, c: -1.0 },
        ];
        let mut seed = 0x77u64;
        let mut u = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let pts: Vec<V3> = (0..200_000).map(|_| f.o + V3::new(u(), u(), u()) * 3.0).collect();
        for q in &kinds {
            let t = std::time::Instant::now();
            let mut acc = 0.0;
            for x in &pts {
                acc += q.signed_distance(*x, 1.0);
            }
            let dt = t.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();
            let mut acc2 = 0.0;
            for x in &pts {
                acc2 += q.value(*x);
            }
            let dt2 = t2.elapsed().as_secs_f64();
            println!("{:<26} exact distance {:.2e} points/s ({:.2} us); G {:.2e} points/s; one thread, f64 ({acc:.1} {acc2:.1})", q.name(), pts.len() as f64 / dt, 1e6 * dt / pts.len() as f64, pts.len() as f64 / dt2);
        }
    }

    #[test]
    fn implicit_matches_value() {
        let q = Quadric { f: frame(), q: [0.7, -0.4, 0.0], l: -1.3, c: 0.25 };
        let (center, scale) = (V3::new(1.0, 2.0, -1.0), 3.5);
        let coef = q.implicit(center, scale);
        for p in [V3::new(0.3, -0.2, 1.4), V3::new(-2.0, 4.0, 0.5)] {
            let n = (p - center) * (1.0 / scale);
            let mons = crate::span_poly::monomials(2);
            let v: f64 = mons.iter().zip(&coef).map(|(m, c)| c * n.x.powi(m[0] as i32) * n.y.powi(m[1] as i32) * n.z.powi(m[2] as i32)).sum();
            assert!((v - q.value(p)).abs() < 1e-12, "{v} {}", q.value(p));
        }
    }
}
