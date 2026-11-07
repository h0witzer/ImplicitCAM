//! The allowance field delta(s) along each wall loop, solved as a 1D problem on the
//! finisher's samples.
//!
//! The engagement at a wall sample depends on the allowance there and, near inside
//! corners, on the allowance around it (where the rougher's corner lands). Model: the
//! engagement tables phi_i(d) measured with the full 2D instrument at uniform allowances d
//! in `levels` (the constant-allowance rougher at each level), interpolated in d, plus a
//! per-sample correction c_i = phi_true - phi_i(delta_i) from a true 2D simulation of the
//! current field (defect correction, a few rounds). Every schedule and every number
//! reported is recomputed on the TRUE engagement of the final field, so the tolerance
//! holds by construction; the model only decides where stock goes.
//!
//! Two ways to set the field:
//!   target   each sample takes the LARGEST allowance in [d_min, d_max] whose engagement
//!            keeps every CutModel limit (deflection = the finishing tolerance) at or
//!            above a target feed there; where even d_min cannot, it takes d_min and the
//!            feed drops.
//!   budget   minimum finishing time for a given stock volume (Lagrangian): each sample
//!            maximizes lambda x delta x wall length - its time ds / min(ceiling(phi(delta)),
//!            kinematic limit), lambda bisected to the volume. Equal marginal time per
//!            unit of stock everywhere.
//! Both are then slope-limited (|d delta / ds| <= slope) by their lower envelope, which
//! never raises a sample above what it chose.

use std::f64::consts::PI;

use icam_motion::CutModel;
use rayon::prelude::*;

use crate::finisher::{engagement, removed_area, Run};
use crate::geom::SegIndex;
use crate::pocket::Pocket;
use crate::stock::RoughStock;

pub struct FieldResult {
    pub label: String,
    pub delta: Vec<f64>,
    pub phi: Vec<f64>,
    pub corr: Vec<f64>,
    pub stock: RoughStock,
    pub area: f64,
    pub lambda: f64,
    pub sched: Option<Run>,
    pub trace: Vec<String>,
}

impl FieldResult {
    pub fn sched(&self) -> &Run {
        self.sched.as_ref().expect("a solved field is scheduled")
    }
}

const STEP: f64 = 0.005;

pub struct Field<'a> {
    pk: &'a Pocket,
    m: &'a CutModel,
    pub levels: Vec<f64>,
    pub slope: f64,
    /// tab[k][i]: engagement at uniform levels[k].
    pub tab: Vec<Vec<f64>>,
    /// Stock the finisher removes at uniform levels[k], mm2.
    pub areas: Vec<f64>,
    pub sims: usize,
}

impl<'a> Field<'a> {
    pub fn new(pk: &'a Pocket, m: &'a CutModel, levels: &[f64], slope: f64) -> Field<'a> {
        let mut f = Field { pk, m, levels: levels.to_vec(), slope, tab: Vec::new(), areas: Vec::new(), sims: 0 };
        for &l in levels {
            let st = RoughStock::uniform(pk, l);
            f.tab.push(engagement(pk, &st));
            f.areas.push(removed_area(pk, &st));
            f.sims += 1;
        }
        f
    }

    pub fn level_index(&self, d: f64) -> Option<usize> {
        self.levels.iter().position(|&l| (l - d).abs() < 1e-9)
    }

    fn phi_model(&self, i: usize, d: f64) -> f64 {
        let l = &self.levels;
        if d <= l[0] {
            return self.tab[0][i];
        }
        for k in 1..l.len() {
            if d <= l[k] {
                let f = (d - l[k - 1]) / (l[k] - l[k - 1]);
                return self.tab[k - 1][i] + f * (self.tab[k][i] - self.tab[k - 1][i]);
            }
        }
        self.tab[l.len() - 1][i]
    }

    fn v_at(&self, phi: f64) -> f64 {
        if phi < 0.2 * PI / 180.0 { self.m.v_max } else { self.m.v_max.min(self.m.ceiling(phi.min(PI)).0) }
    }

    /// Largest allowance per sample (at most ub) that keeps the target feed.
    pub fn target_solve(&self, target: &[f64], corr: &[f64], ub: Option<&[f64]>) -> Vec<f64> {
        let (lo, d_max) = (self.pk.d_min, self.pk.d_max);
        let mut d: Vec<f64> = (0..self.pk.n())
            .into_par_iter()
            .map(|i| {
                let hi = ub.map_or(d_max, |u| u[i]);
                let mut x = hi;
                while x >= lo - 1e-9 {
                    if self.v_at(0.0f64.max(self.phi_model(i, x) + corr[i])) >= target[i] * (1.0 - 1e-6) {
                        return lo.max(x);
                    }
                    x -= STEP;
                }
                lo
            })
            .collect();
        self.slope_limit(&mut d);
        d
    }

    /// Per sample and allowance step: the time over the sample, ds / min(ceiling, kinematic limit).
    fn time_table(&self, v_geo: &[f64], corr: &[f64]) -> Vec<Vec<f64>> {
        let pk = self.pk;
        let nx = ((pk.d_max - pk.d_min) / STEP).round() as usize + 1;
        (0..pk.n())
            .into_par_iter()
            .map(|i| {
                (0..nx)
                    .map(|k| {
                        let x = pk.d_min + k as f64 * STEP;
                        let v = self.v_at(0.0f64.max(self.phi_model(i, x) + corr[i])).min(v_geo[i]);
                        pk.ds / v.max(1e-3)
                    })
                    .collect()
            })
            .collect()
    }

    /// Per sample, the allowance step that maximizes lambda d dw - t(d).
    fn budget_args(&self, lambda: f64, tt: &[Vec<f64>]) -> Vec<usize> {
        let pk = self.pk;
        (0..pk.n())
            .map(|i| {
                let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
                for (k, &t) in tt[i].iter().enumerate() {
                    let j = lambda * (pk.d_min + k as f64 * STEP) * pk.dw[i] - t;
                    if j > best {
                        best = j;
                        arg = k;
                    }
                }
                arg
            })
            .collect()
    }

    fn limited(&self, args: &[usize]) -> Vec<f64> {
        let mut d: Vec<f64> = args.iter().map(|&k| self.pk.d_min + k as f64 * STEP).collect();
        self.slope_limit(&mut d);
        d
    }

    /// Minimum time for the stock: per sample, argmax of lambda d dw - t(d).
    fn budget_solve(&self, lambda: f64, tt: &[Vec<f64>]) -> Vec<f64> {
        self.limited(&self.budget_args(lambda, tt))
    }

    /// Where the stock jumps across the target as lambda crosses a value (many samples
    /// change their choice at once, as on a wall whose time is flat in the allowance), no
    /// lambda meets the budget. Between the two sides, switch the samples that add stock
    /// for the least time per mm2 first (equal costs in wall order, so switched stretches
    /// stay whole under the slope limit), as few as meet the target.
    fn budget_blend(&self, lo: &[usize], hi: &[usize], tt: &[Vec<f64>], target: f64) -> Vec<f64> {
        let pk = self.pk;
        let mut cand: Vec<(i64, usize)> = (0..pk.n())
            .filter(|&i| hi[i] > lo[i])
            .map(|i| {
                let dt = tt[i][hi[i]] - tt[i][lo[i]];
                let da = (hi[i] - lo[i]) as f64 * STEP * pk.dw[i];
                let ratio = dt / da.max(1e-12);
                (if ratio <= 1e-12 { i64::MIN } else { (ratio.ln() / 1.01f64.ln()).floor() as i64 }, i)
            })
            .collect();
        cand.sort();
        let build = |m: usize| {
            let mut args = lo.to_vec();
            for &(_, i) in &cand[..m] {
                args[i] = hi[i];
            }
            self.limited(&args)
        };
        let (mut a, mut b) = (0, cand.len());
        while a < b {
            let mid = (a + b) / 2;
            if self.area_model(&build(mid)) >= target {
                b = mid;
            } else {
                a = mid + 1;
            }
        }
        build(a)
    }

    fn slope_limit(&self, d: &mut [f64]) {
        let pk = self.pk;
        for lp in &pk.loops {
            let (start, n) = (lp.start, lp.count);
            for _ in 0..2 {
                for k in 0..n {
                    let (i, p) = (start + k, start + (k + n - 1) % n);
                    d[i] = d[i].min(d[p] + self.slope * self.dist(p, i));
                }
                for k in (0..n).rev() {
                    let (i, q) = (start + k, start + (k + 1) % n);
                    d[i] = d[i].min(d[q] + self.slope * self.dist(q, i));
                }
            }
        }
    }

    fn dist(&self, a: usize, b: usize) -> f64 {
        let pk = self.pk;
        ((pk.x[a] - pk.x[b]) * (pk.x[a] - pk.x[b]) + (pk.y[a] - pk.y[b]) * (pk.y[a] - pk.y[b])).sqrt()
    }

    /// Allowance x wall length (mm2): the stock estimate the budget bisects on, calibrated
    /// against the true removed area after each simulation.
    fn area_model(&self, d: &[f64]) -> f64 {
        let mut a = 0.0;
        for i in 0..self.pk.n() {
            a += d[i] * self.pk.dw[i];
        }
        a
    }

    fn simulate(&mut self, label: &str, delta: Vec<f64>) -> FieldResult {
        let st = RoughStock::new(self.pk, delta.clone());
        let phi = engagement(self.pk, &st);
        self.sims += 1;
        let area = removed_area(self.pk, &st);
        FieldResult { label: label.into(), delta, phi, corr: Vec::new(), stock: st, area, lambda: f64::NAN, sched: None, trace: Vec::new() }
    }

    fn correct(&self, r: &FieldResult, corr: &mut [f64]) {
        for i in 0..self.pk.n() {
            corr[i] = r.phi[i] - self.phi_model(i, r.delta[i]);
        }
    }

    /// Target field, defect-corrected; keeps the round with the shortest time under
    /// `schedule` (ties: more stock).
    pub fn solve_target(&mut self, label: &str, target: &[f64], rounds: usize, schedule: &dyn Fn(&[f64]) -> Run, corr0: Option<&[f64]>) -> FieldResult {
        let pk = self.pk;
        let n = pk.n();
        let mut corr = corr0.map_or_else(|| vec![0.0; n], |c| c.to_vec());
        let mut best: Option<FieldResult> = None;
        let mut trace = Vec::new();
        // Monotone safeguard: a sample whose true ceiling falls under its target caps its
        // own allowance and, within the rougher's reach (its corner lands there), its
        // neighbors' too; the caps only ever come down.
        let mut ub = vec![pk.d_max; n];
        let reach = pk.rr + pk.rf;
        let owners: Vec<_> = (0..n).map(|i| (pk.x[i], pk.y[i], pk.x[i], pk.y[i], i as i32)).collect();
        let near = SegIndex::new(&owners, reach, 2.0);
        for it in 0..rounds {
            let d = self.target_solve(target, &corr, Some(&ub));
            let mut r = self.simulate(label, d);
            let sched = schedule(&r.phi);
            self.correct(&r, &mut corr);
            let mut below = 0;
            for i in 0..n {
                if !(sched.v_ceil[i] < 0.99 * target[i].min(self.m.v_max)) {
                    continue;
                }
                below += 1;
                ub[i] = ub[i].min(pk.d_min + 0.5 * (r.delta[i] - pk.d_min));
                if r.delta[i] > pk.d_min + 1e-6 {
                    continue;
                }
                // Already at the floor: the excess comes from around it.
                near.for_each_within(pk.x[i], pk.y[i], reach, |j| {
                    let j = j as usize;
                    ub[j] = ub[j].min(pk.d_min + 0.6 * (r.delta[j] - pk.d_min));
                });
            }
            r.sched = Some(sched);
            let keep = match &best {
                None => true,
                Some(b) => r.sched().time < b.sched().time - 1e-3 || ((r.sched().time - b.sched().time).abs() <= 1e-3 && r.area > b.area),
            };
            trace.push(format!(
                "round {}: time {:.2} s, stock {:.1} mm2, {below} samples whose ceiling is under the target{}",
                it + 1,
                r.sched().time,
                r.area,
                if keep { "" } else { " (kept the previous round)" }
            ));
            if keep {
                best = Some(r);
            }
        }
        let mut best = best.expect("at least one round");
        best.corr = corr;
        best.trace.extend(trace);
        best
    }

    /// Minimum-time field for a stock area (mm2), defect-corrected on the area.
    pub fn solve_budget(&mut self, label: &str, area: f64, v_geo: &[f64], corr0: &[f64], rounds: usize, schedule: &dyn Fn(&[f64]) -> Run) -> FieldResult {
        let mut corr = corr0.to_vec();
        let mut calib = 0.0;
        let mut r: Option<FieldResult> = None;
        let mut trace = Vec::new();
        for it in 0..rounds {
            let tt = self.time_table(v_geo, &corr);
            let (mut lo, mut hi) = (-14.0f64, 7.0f64); // log10 lambda, s/mm2
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let a = self.area_model(&self.budget_solve(10f64.powf(mid), &tt)) + calib;
                if a < area {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let lam = 10f64.powf(0.5 * (lo + hi));
            // Meet the budget across a jump in the Lagrangian solution.
            let (a_lo, a_hi) = (self.budget_args(10f64.powf(lo), &tt), self.budget_args(10f64.powf(hi), &tt));
            let d = self.budget_blend(&a_lo, &a_hi, &tt, area - calib);
            let am = self.area_model(&d);
            let mut rr = self.simulate(label, d);
            rr.lambda = lam;
            calib = rr.area - am;
            self.correct(&rr, &mut corr);
            trace.push(format!("round {}: lambda {lam:.2e} s/mm2, stock {:.1} mm2 (target {area:.1})", it + 1, rr.area));
            r = Some(rr);
        }
        let mut r = r.expect("at least one round");
        r.sched = Some(schedule(&r.phi));
        r.trace.extend(trace);
        r.corr = corr;
        r
    }
}
