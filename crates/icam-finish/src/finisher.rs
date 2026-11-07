//! One instrument for every allowance: the finisher's wall pass over a given rough stock.
//!
//! Engagement: at every sample (ds apart) the angle of the cutter's leading half-circle
//! that touches stock, measured against the stock as it stands just before that step
//! (the rough stock minus the finisher's own sweep so far), 1 deg samples with each
//! stock/air transition bisected to ~0.02 deg. Only the leading half counts: the trailing
//! half is cut by the previous step, and counting it would add a discretization sliver.
//!
//! Feed: at each sample the fastest feed inside every CutModel limit at the measured
//! engagement (chip, power, deflection = the finishing tolerance, machine feed), then the
//! tool-center path's curvature limit sqrt(A R) and forward/backward acceleration passes
//! (each loop starts and ends at rest). Deflection is then evaluated at the feed actually
//! run.

use std::f64::consts::PI;

use icam_motion::{CutModel, Limit};
use rayon::prelude::*;

use crate::pocket::Pocket;
use crate::stock::RoughStock;

/// What set the feed actually run at a sample: a CutModel ceiling, the path's
/// curvature/acceleration, or the program's single feed (held below its ceiling).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Chip,
    Power,
    Deflection,
    Machine,
    Kinematics,
    OneFeed,
}

impl Bind {
    pub const ALL: [Bind; 6] = [Bind::Chip, Bind::Power, Bind::Deflection, Bind::Machine, Bind::Kinematics, Bind::OneFeed];

    fn of(l: Limit) -> Bind {
        match l {
            Limit::Chip => Bind::Chip,
            Limit::Power => Bind::Power,
            Limit::Deflection => Bind::Deflection,
            Limit::Kinematics => Bind::Kinematics,
            Limit::Machine => Bind::Machine,
        }
    }
}

/// Feed programs: a feed per sample (F word per block), one feed for the whole pass (a
/// one-feed post, set by the worst spot), or one feed on the wall with the corner zones
/// slowed to their own safe feed (a corner-slowdown program).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedProgram {
    PerSample,
    OneFeed,
    WallFeed,
}

#[derive(Clone, Debug)]
pub struct Run {
    pub label: String,
    pub mode: FeedProgram,
    pub phi: Vec<f64>,
    pub v_ceil: Vec<f64>,
    pub v: Vec<f64>,
    pub defl: Vec<f64>,
    pub binds: Vec<Bind>,
    pub time: f64,
    pub corner_time: f64,
    pub cut_len: f64,
    pub rub_len: f64,
    pub defl_max: f64,
    pub defl_p95: f64,
    pub defl_mean: f64,
    pub phi_max: f64,
    pub phi_med: f64,
    pub v_med: f64,
    pub v_const: f64,
    /// The sample that sets the one feed.
    pub worst: i32,
    /// Wall length bound by each limit, indexed as `Bind::ALL`.
    pub bind_len: [f64; 6],
}

impl Run {
    pub fn bind(&self, b: Bind) -> f64 {
        self.bind_len[b as usize]
    }
}

const NA: usize = 180; // 1 deg over the leading half
/// Below this a sample is an air cut.
pub const PHI_AIR: f64 = 0.2 * PI / 180.0;

pub fn engagement(pk: &Pocket, st: &RoughStock) -> Vec<f64> {
    let rs = pk.rf - 0.0015;
    let cs: Vec<f64> = (0..=NA).map(|k| (-PI / 2.0 + PI * k as f64 / NA as f64).cos()).collect();
    let sn: Vec<f64> = (0..=NA).map(|k| (-PI / 2.0 + PI * k as f64 / NA as f64).sin()).collect();
    (0..pk.n())
        .into_par_iter()
        .map(|i| {
            let (px, py, tx, ty) = (pk.x[i], pk.y[i], pk.tx[i], pk.ty[i]);
            // a > 0 turns from the travel direction toward the wall side (right of travel).
            let m = |c: f64, s: f64| {
                let (dx, dy) = (tx * c + ty * s, ty * c - tx * s);
                let (qx, qy) = (px + rs * dx, py + rs * dy);
                st.stock(pk, qx, qy) && !pk.fin_sweep.within_before(qx, qy, pk.rf, i as i32)
            };
            let hit: Vec<bool> = (0..=NA).map(|k| m(cs[k], sn[k])).collect();
            if !hit.iter().any(|&h| h) {
                return 0.0;
            }
            let (mut sum, step) = (0.0, PI / NA as f64);
            for k in 0..NA {
                if hit[k] == hit[k + 1] {
                    if hit[k] {
                        sum += step;
                    }
                    continue;
                }
                let mut a0 = -PI / 2.0 + step * k as f64;
                let mut a1 = a0 + step;
                let m0 = hit[k];
                for _ in 0..6 {
                    let am = 0.5 * (a0 + a1);
                    if m(am.cos(), am.sin()) == m0 {
                        a0 = am;
                    } else {
                        a1 = am;
                    }
                }
                let split = 0.5 * (a0 + a1) - (-PI / 2.0 + step * k as f64);
                sum += if m0 { split } else { step - split };
            }
            sum
        })
        .collect()
}

/// Stock thickness along the wall normal at each sample's wall foot (what the finisher
/// must remove there).
pub fn thickness(pk: &Pocket, st: &RoughStock) -> Vec<f64> {
    let reach = 2.0 * pk.rf + 2.0;
    (0..pk.n())
        .into_par_iter()
        .map(|i| {
            let (wx, wy, nx, ny) = (pk.wx[i], pk.wy[i], pk.nx[i], pk.ny[i]);
            let s = |d: f64| st.stock(pk, wx + d * nx, wy + d * ny);
            let d0 = 0.002;
            if !s(d0) {
                return 0.0;
            }
            let mut d1 = d0;
            while d1 < reach && s(d1 + 0.02) {
                d1 += 0.02;
            }
            let (mut lo, mut hi) = (d1, reach.min(d1 + 0.02));
            for _ in 0..8 {
                let m = 0.5 * (lo + hi);
                if s(m) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            0.5 * (lo + hi)
        })
        .collect()
}

/// Area of stock the finisher's sweep removes (mm2 at the floor; times ap = volume).
pub fn removed_area(pk: &Pocket, st: &RoughStock) -> f64 {
    let cg = &pk.cg;
    let lim = 2.0 * pk.rf + 0.01;
    let count: i64 = (0..cg.ny)
        .into_par_iter()
        .map(|j| {
            let mut acc = 0i64;
            for i in 0..cg.nx {
                let c = cg.v[j * cg.nx + i];
                if c <= 0.0 || c > lim {
                    continue;
                }
                let (x, y) = (cg.x(i), cg.y(j));
                if st.stock(pk, x, y) && pk.fin_sweep.within(x, y, pk.rf) {
                    acc += 1;
                }
            }
            acc
        })
        .sum();
    count as f64 * cg.h * cg.h
}

pub fn ceilings(m: &CutModel, phi: &[f64]) -> Vec<f64> {
    phi.iter().map(|&p| if p < PHI_AIR { m.v_max } else { m.v_max.min(m.ceiling(p.min(PI)).0) }).collect()
}

/// Kinematic feed along the loops for a ceiling per sample.
pub fn kinematic(pk: &Pocket, ceil: &[f64], accel: f64) -> Vec<f64> {
    let mut v: Vec<f64> = (0..pk.n()).map(|i| ceil[i].min(if pk.rk[i].is_infinite() { f64::INFINITY } else { (accel * pk.rk[i]).sqrt() })).collect();
    for lp in &pk.loops {
        let (start, n) = (lp.start, lp.count);
        v[start] = 0.0;
        v[start + n - 1] = 0.0;
        for k in 1..n {
            let i = start + k;
            v[i] = v[i].min((v[i - 1] * v[i - 1] + 2.0 * accel * (pk.s[i] - pk.s[i - 1])).sqrt());
        }
        for k in (0..n - 1).rev() {
            let i = start + k;
            v[i] = v[i].min((v[i + 1] * v[i + 1] + 2.0 * accel * (pk.s[i + 1] - pk.s[i])).sqrt());
        }
    }
    v
}

pub fn time(pk: &Pocket, v: &[f64]) -> f64 {
    let mut t = 0.0;
    for lp in &pk.loops {
        for k in 1..lp.count {
            let i = lp.start + k;
            t += (pk.s[i] - pk.s[i - 1]) / (0.5 * (v[i] + v[i - 1])).max(1e-3);
        }
    }
    t
}

/// Schedule on measured engagement under one of the feed programs. `corner` marks the
/// corner zones (samples whose leading edge meets rougher leftovers): WallFeed runs one
/// feed outside them and each zone sample at its own ceiling; it also splits out the time
/// spent in them for every program.
pub fn schedule(pk: &Pocket, m: &CutModel, phi: &[f64], mode: FeedProgram, label: &str, corner: Option<&[bool]>, accel: f64) -> Run {
    let n = pk.n();
    let v_ceil = ceilings(m, phi);
    let mut r = Run {
        label: label.into(),
        mode,
        phi: phi.to_vec(),
        v_ceil,
        v: Vec::new(),
        defl: Vec::new(),
        binds: Vec::new(),
        time: 0.0,
        corner_time: 0.0,
        cut_len: 0.0,
        rub_len: 0.0,
        defl_max: 0.0,
        defl_p95: 0.0,
        defl_mean: 0.0,
        phi_max: 0.0,
        phi_med: 0.0,
        v_med: 0.0,
        v_const: f64::NAN,
        worst: -1,
        bind_len: [0.0; 6],
    };
    let mut ceil = r.v_ceil.clone();
    if mode != FeedProgram::PerSample {
        // The programmed F: the largest one whose run feed (after the controller's own
        // curvature and acceleration slowdowns) stays under every sample's ceiling. At
        // least the smallest ceiling (always safe), at most the machine feed.
        let wall = mode == FeedProgram::WallFeed && corner.is_some();
        let vc = &r.v_ceil;
        let ceil_at = |f: f64| -> Vec<f64> { (0..n).map(|i| if wall && corner.is_some_and(|c| c[i]) { f.min(vc[i]) } else { f }).collect() };
        let ok = |f: f64| -> (bool, i32) {
            let v = kinematic(pk, &ceil_at(f), accel);
            let (mut bad, mut worst) = (-1, 1.0);
            for i in 0..n {
                if phi[i] >= PHI_AIR && v[i] > vc[i] * (1.0 + 1e-9) && v[i] / vc[i] > worst {
                    worst = v[i] / vc[i];
                    bad = i as i32;
                }
            }
            (bad < 0, bad)
        };
        let mut lo = m.v_max;
        for i in 0..n {
            if phi[i] >= PHI_AIR {
                lo = lo.min(vc[i]);
            }
        }
        let mut hi = m.v_max;
        let (fits, mut worst_i) = ok(hi);
        if !fits {
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                let (fits, b) = ok(mid);
                if fits {
                    lo = mid;
                } else {
                    hi = mid;
                    worst_i = b;
                }
            }
            hi = lo;
        }
        ceil = ceil_at(hi);
        r.v_const = hi;
        r.worst = worst_i;
    }
    r.v = kinematic(pk, &ceil, accel);
    r.time = time(pk, &r.v);
    r.defl = vec![0.0; n];
    r.binds = vec![Bind::Chip; n];
    let (mut defl, mut phis, mut vs) = (Vec::new(), Vec::new(), Vec::new());
    let mut dsum = 0.0;
    for i in 0..n {
        let linked = i > 0 && pk.loop_of[i] == pk.loop_of[i - 1];
        let ds = if linked { pk.s[i] - pk.s[i - 1] } else { 0.0 };
        if linked && corner.is_some_and(|c| c[i]) {
            r.corner_time += ds / (0.5 * (r.v[i] + r.v[i - 1])).max(1e-3);
        }
        if phi[i] < PHI_AIR {
            r.binds[i] = Bind::Machine;
            continue;
        }
        let (p, vi) = (phi[i].min(PI), r.v[i].max(1e-3));
        r.defl[i] = m.deflection(p, vi);
        let (vc, lim) = m.ceiling(p);
        let cap = vc.min(m.v_max);
        r.binds[i] = if r.v[i] >= 0.97 * cap {
            if vc < m.v_max { Bind::of(lim) } else { Bind::Machine }
        } else if mode != FeedProgram::PerSample && r.v[i] >= 0.97 * ceil[i] && ceil[i] < 0.97 * cap {
            Bind::OneFeed
        } else {
            Bind::Kinematics
        };
        r.bind_len[r.binds[i] as usize] += ds;
        r.cut_len += ds;
        if m.rubs(p, vi) {
            r.rub_len += ds;
        }
        defl.push(r.defl[i]);
        phis.push(phi[i]);
        vs.push(r.v[i]);
        dsum += r.defl[i] * ds;
    }
    if !defl.is_empty() {
        r.defl_mean = dsum / 1e-9f64.max(r.cut_len);
        let by = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        defl.sort_by(by);
        phis.sort_by(by);
        vs.sort_by(by);
        r.defl_max = defl[defl.len() - 1];
        r.defl_p95 = defl[(0.95 * (defl.len() - 1) as f64) as usize];
        r.phi_max = phis[phis.len() - 1];
        r.phi_med = phis[phis.len() / 2];
        r.v_med = vs[vs.len() / 2];
    }
    r
}
