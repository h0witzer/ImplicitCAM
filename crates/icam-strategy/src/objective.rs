//! The plan field designed by the objective (feasibility doc 4.5-4.6): minimum achieved
//! time under the physical load limits (CutModel) and the machine's acceleration,
//! instead of a fixed stepover.
//!
//!   kappa(x)  curvature of the plan's front through x (tool-center space, so it is the
//!             tool's own turn radius): div(grad T / |grad T|), smoothed.
//!   s*(x)     load-optimal stepover: argmax over ae of ae * v(ae, kappa(x)),
//!             v = min(CutModel ceiling at ae, sqrt(A / kappa), v_max). Where the front
//!             turns tight and the machine slows anyway, a wider cut at a lower feed
//!             wins; on open sweeps the ceiling's own best wins. With a local depth field
//!             (ap), only stable stepovers count: the engagement on a front of this
//!             curvature (PhysPlan::engagement) must stay within CutModel::stable_phi(ap(x)).
//!   sigma(x)  the spacing the spiral will realize: it steps each level by the tightest
//!             point of that level, dT = min over the level of s* |grad T|, so
//!             sigma = dT / |grad T| <= s*.
//!   J[T]      estimated time = sum over the pocket of dA / (sigma v(sigma, kappa))
//!             (coarea: front length per unit area = 1 / sigma).
//!   Plan      div(k grad T) = 0, elliptic, so no stranded material and closed fronts
//!             that reach the walls together. k is iterated so the ratio
//!             rho = s* |grad T| is even along each level (then every point of a level
//!             hits its s* at once): k <- k * (rho / level median)^alpha, and the iterate
//!             with the lowest J is kept.

use std::f64::consts::PI;

use icam_motion::CutModel;

use crate::physplan::PhysPlan;
use crate::plan::Plan;

pub struct Objective {
    accel: f64,
    ae_min: f64,
    ae_max: f64,
    ae_tab: Vec<f64>, // ceiling feed by stepover (independent of x)
    v_tab: Vec<f64>,
    pub s_star: Vec<f64>,
    pub kappa: Vec<f64>,
    pub sigma: Vec<f64>,
    /// Local axial depth on the plan grid (chatter limit), optional.
    pub ap: Option<Vec<f64>>,
    pub j: f64,
    pub trace: Vec<String>,
    nx: usize,
    ny: usize,
    h: f64,
    x0: f64,
    y0: f64,
}

impl Objective {
    pub fn new(p: &Plan, m: &CutModel, accel: f64, ae_max: f64) -> Objective {
        let ae_min = 0.03 * m.d;
        let n = 60;
        let (mut ae_tab, mut v_tab) = (vec![0.0; n + 1], vec![0.0; n + 1]);
        for i in 0..=n {
            let ae = ae_min + (ae_max - ae_min) * i as f64 / n as f64;
            ae_tab[i] = ae;
            v_tab[i] = m.ceiling((1.0 - ae / m.r()).clamp(-1.0, 1.0).acos()).0;
        }
        Objective {
            accel,
            ae_min,
            ae_max,
            ae_tab,
            v_tab,
            s_star: Vec::new(),
            kappa: Vec::new(),
            sigma: Vec::new(),
            ap: None,
            j: 0.0,
            trace: Vec::new(),
            nx: p.nx,
            ny: p.ny,
            h: p.h,
            x0: p.x0,
            y0: p.y0,
        }
    }

    fn v_ceil(&self, ae: f64) -> f64 {
        let f = ((ae - self.ae_min) / (self.ae_max - self.ae_min)).clamp(0.0, 1.0) * (self.ae_tab.len() - 1) as f64;
        let i = (self.ae_tab.len() - 2).min(f as usize);
        self.v_tab[i] + (self.v_tab[i + 1] - self.v_tab[i]) * (f - i as f64)
    }

    fn v_kin(&self, kappa: f64) -> f64 {
        if kappa < 1e-9 { f64::INFINITY } else { (self.accel / kappa).sqrt() }
    }

    pub fn feed(&self, m: &CutModel, ae: f64, kappa: f64) -> f64 {
        m.v_max.min(self.v_ceil(ae).min(self.v_kin(kappa)))
    }

    /// Iterate the conductivity toward the objective; keeps the best plan in `p.t`.
    pub fn optimize(&mut self, p: &mut Plan, m: &CutModel, iterations: usize, alpha: f64) {
        let n = self.nx * self.ny;
        let mut k = p.k.clone().unwrap_or_default();
        let mut best_t = p.t.clone();
        let mut best_k = k.clone();
        self.evaluate(p, m);
        let mut best_j = self.j;
        let (mut best_s, mut best_kap, mut best_sig) = (self.s_star.clone(), self.kappa.clone(), self.sigma.clone());
        self.trace.push(format!("iter 0 (conductive): J {:.1} s", self.j));
        for it in 1..=iterations {
            // rho = s* |grad T|, normalized by the median of its level.
            let (gx, gy) = self.grad(p);
            let mut rho = vec![0.0; n];
            let (bins, nb) = self.bins(p);
            let mut med = vec![0.0; nb];
            let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); nb];
            for s in 0..n {
                if bins[s] < 0 {
                    continue;
                }
                rho[s] = self.s_star[s] * (gx[s] * gx[s] + gy[s] * gy[s]).sqrt();
                per_bin[bins[s] as usize].push(rho[s]);
            }
            for b in 0..nb {
                let l = &mut per_bin[b];
                if l.is_empty() {
                    continue;
                }
                l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                med[b] = l[l.len() / 2];
            }
            let mut mult: Vec<f64> = (0..n)
                .map(|s| if bins[s] >= 0 && med[bins[s] as usize] > 0.0 { (rho[s] / med[bins[s] as usize]).powf(alpha).clamp(0.5, 2.0) } else { 1.0 })
                .collect();
            self.smooth(p, &mut mult, 3);
            // Where the level is too tight (rho high) the gradient must drop: raise k there.
            for s in 0..n {
                if k[s] > 0.0 {
                    k[s] *= mult[s];
                }
            }
            p.solve_conductive_with(k.clone(), 1e-6, 20000);
            self.evaluate(p, m);
            self.trace.push(format!("iter {it}: J {:.1} s{}", self.j, if self.j < best_j { "  (best)" } else { "" }));
            if self.j < best_j {
                best_j = self.j;
                best_t = p.t.clone();
                best_k = k.clone();
                best_s = self.s_star.clone();
                best_kap = self.kappa.clone();
                best_sig = self.sigma.clone();
            }
        }
        p.t = best_t;
        p.k = Some(best_k);
        self.s_star = best_s;
        self.kappa = best_kap;
        self.sigma = best_sig;
        self.j = best_j;
    }

    /// kappa, s*, sigma and J for the plan as it stands.
    pub fn evaluate(&mut self, p: &Plan, m: &CutModel) {
        let (n, nx) = (self.nx * self.ny, self.nx);
        let (gx, gy) = self.grad(p);
        // Front curvature (signed: + convex toward the uncut stock): divergence of the unit
        // normal, then smoothed; capped at half the tool radius.
        let (mut ux, mut uy) = (vec![0.0; n], vec![0.0; n]);
        for s in 0..n {
            let g = (gx[s] * gx[s] + gy[s] * gy[s]).sqrt();
            if p.label[s] == 1 && g > 1e-9 {
                ux[s] = gx[s] / g;
                uy[s] = gy[s] / g;
            }
        }
        let mut kappa = vec![0.0; n];
        for j in 1..self.ny - 1 {
            for i in 1..nx - 1 {
                let s = j * nx + i;
                if p.label[s] != 1 {
                    continue;
                }
                let cap = 1.0 / (0.5 * m.r());
                kappa[s] = ((ux[s + 1] - ux[s - 1] + uy[s + nx] - uy[s - nx]) / (2.0 * self.h)).clamp(-cap, cap);
            }
        }
        self.smooth(p, &mut kappa, 4);
        self.kappa = kappa;
        // s*: the stepover with the best ae * v at this curvature.
        self.s_star = vec![0.0; n];
        let mut valid = Vec::new();
        for s in 0..n {
            if p.label[s] != 1 {
                continue;
            }
            let (mut best, mut arg) = (-1.0, self.ae_tab[0]);
            let phi_max = match &self.ap {
                None => PI,
                Some(ap) => m.stable_phi(ap[s].clamp(0.1, m.loc)),
            };
            for i in 0..self.ae_tab.len() {
                if i > 0 && PhysPlan::engagement(self.ae_tab[i], self.kappa[s], m.r()) > phi_max {
                    break;
                }
                let q = self.ae_tab[i] * m.v_max.min(self.v_tab[i].min(self.v_kin(self.kappa[s].abs())));
                if q > best * 1.0001 {
                    best = q;
                    arg = self.ae_tab[i];
                }
            }
            self.s_star[s] = arg;
            valid.push(arg);
        }
        valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let fill = if valid.is_empty() { self.ae_max } else { valid[valid.len() / 2] };
        for s in 0..n {
            if p.label[s] != 1 {
                self.s_star[s] = fill;
            }
        }
        // sigma: each level steps by its tightest point (3rd percentile of rho).
        let (bins, nb) = self.bins(p);
        let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); nb];
        let mut rho = vec![0.0; n];
        for s in 0..n {
            if bins[s] < 0 {
                continue;
            }
            rho[s] = self.s_star[s] * (gx[s] * gx[s] + gy[s] * gy[s]).sqrt();
            per_bin[bins[s] as usize].push(rho[s]);
        }
        let mut low = vec![0.0; nb];
        for b in 0..nb {
            let l = &mut per_bin[b];
            if l.is_empty() {
                continue;
            }
            l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            low[b] = l[(0.03 * (l.len() - 1) as f64) as usize];
        }
        self.sigma = vec![0.0; n];
        self.j = 0.0;
        let a = self.h * self.h;
        for s in 0..n {
            if bins[s] < 0 {
                continue;
            }
            let g = (gx[s] * gx[s] + gy[s] * gy[s]).sqrt();
            let sig = if g > 1e-9 { (low[bins[s] as usize] / g).clamp(0.25 * self.ae_min, self.s_star[s]) } else { self.s_star[s] };
            self.sigma[s] = sig;
            self.j += a / (sig * self.feed(m, sig, self.kappa[s].abs()));
        }
    }

    /// s* at a point (bilinear), for the spiral's level spacing.
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let (fx, fy) = ((x - self.x0) / self.h, (y - self.y0) / self.h);
        let i = (fx.floor() as i64).clamp(0, self.nx as i64 - 2) as usize;
        let j = (fy.floor() as i64).clamp(0, self.ny as i64 - 2) as usize;
        let (u, v) = ((fx - i as f64).clamp(0.0, 1.0), (fy - j as f64).clamp(0.0, 1.0));
        let s = j * self.nx + i;
        let f = &self.s_star;
        (1.0 - u) * (1.0 - v) * f[s] + u * (1.0 - v) * f[s + 1] + (1.0 - u) * v * f[s + self.nx] + u * v * f[s + self.nx + 1]
    }

    /// (min, median, max) of s* on the pocket.
    pub fn range(&self, p: &Plan) -> (f64, f64, f64) {
        let mut v: Vec<f64> = (0..self.s_star.len()).filter(|&s| p.label[s] == 1).map(|s| self.s_star[s]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() { (0.0, 0.0, 0.0) } else { (v[0], v[v.len() / 2], v[v.len() - 1]) }
    }

    /// Level bins of T over the pocket's free nodes (outside the stadiums).
    fn bins(&self, p: &Plan) -> (Vec<i64>, usize) {
        let nb = 200;
        let b = (0..p.t.len())
            .map(|s| if p.label[s] == 1 && p.t[s] > 1e-6 && p.t[s] < 1.0 - 1e-6 { (nb as i64 - 1).min((p.t[s] * nb as f64) as i64) } else { -1 })
            .collect();
        (b, nb)
    }

    fn grad(&self, p: &Plan) -> (Vec<f64>, Vec<f64>) {
        let (n, nx) = (self.nx * self.ny, self.nx);
        let (mut gx, mut gy) = (vec![0.0; n], vec![0.0; n]);
        for j in 1..self.ny - 1 {
            for i in 1..nx - 1 {
                let s = j * nx + i;
                if p.label[s] != 1 {
                    continue;
                }
                gx[s] = (p.t[s + 1] - p.t[s - 1]) / (2.0 * self.h);
                gy[s] = (p.t[s + nx] - p.t[s - nx]) / (2.0 * self.h);
            }
        }
        (gx, gy)
    }

    /// Jacobi passes over the pocket's free nodes (the others keep their values).
    fn smooth(&self, p: &Plan, f: &mut [f64], passes: usize) {
        let nx = self.nx;
        for _ in 0..passes {
            let mut g = f.to_vec();
            for j in 1..self.ny - 1 {
                for i in 1..nx - 1 {
                    let s = j * nx + i;
                    if p.label[s] != 1 {
                        continue;
                    }
                    let (mut sum, mut c) = (4.0 * f[s], 4);
                    for m in [s - 1, s + 1, s - nx, s + nx] {
                        if p.label[m] == 1 {
                            sum += f[m];
                            c += 1;
                        }
                    }
                    g[s] = sum / c as f64;
                }
            }
            f.copy_from_slice(&g);
        }
    }
}
