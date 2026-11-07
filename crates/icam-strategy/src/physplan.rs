//! Physics-shaped stepover: fast marching with a speed field.
//!
//! The plan is T with |grad T| = s_ref / s*(x): its level sets T = k s_ref are s*(x)
//! apart, so every point is cut at its own stepover, and max-cut reads the path off it
//! unchanged (closed fronts, then open arcs lobe by lobe, walls last).
//!
//!   s*(x)  the widest stepover whose engagement stays stable (chatter,
//!          CutModel::stable_phi) at the local axial depth ap(x) on a front of the local
//!          curvature, within [s_min, cap]. Every other limit lets removal rate rise
//!          with stepover, so the optimum sits on the stability boundary; where even
//!          s_min is unstable the level is too deep for this tool and the node is
//!          counted, not hidden.
//!   ap(x)  stock height the program leaves before this op, minus the cut level: the
//!          lowest any earlier op's tool reaches over each plan node, exact for every
//!          tool shape and motion (`height_before`).
//!   kappa  signed curvature of the front through x. A front convex toward the uncut
//!          stock cuts like an internal circle: for tool-center radius R and stepover s
//!          the engaged angle is acos(((R - s + r)^2 - R^2 - r^2) / (2 R r)), more than
//!          on a straight line. Concave fronts are taken as straight.
//!
//! Continuity (default): s* comes from fixed geometry only, in one solve. kappa is READ
//! from the field: the curvature of the fronts of the constant-speed distance T0 from
//! the entry, after T0 is smoothed over half the tool radius (so the derivative is
//! regularized and continuous in the data). Unlike the closed form 1 / (T0 + rho) (kept
//! as `analytic_kappa`), it sees how fronts wrap islands and bend at walls. The target
//! is smoothed over half a tool diameter (normalized Gaussian on the pocket) and capped
//! at the stable value again (min is continuous). No feedback, no thresholds, so a small
//! change in the input moves s* and T by a small amount. The earlier coupled version
//! (kappa from the plan being solved, iterated) is kept as `feedback` for comparison: it
//! failed the continuity test (57-101 mm path moves for a 5 um wall change).

use std::collections::BTreeMap;

use icam_motion::{CutModel, Sample};
use icam_post::{Box2, Move, ToolDef};
use icam_stock::StockTop;
use rayon::prelude::*;

use crate::plan::Plan;

#[derive(Clone, Copy, Debug)]
pub struct PhysOptions {
    /// kappa = 1 / (T0 + rho) instead of reading the field.
    pub analytic_kappa: bool,
    /// 1: kappa from smoothed T0; more re-read it from the solved plan.
    pub read_passes: usize,
    /// Plan the engagement at (1 - margin) of the stable angle. Measured in the interior
    /// of the five pockets, the path's engagement runs at 0.84-0.98 of the plan's
    /// prediction at the median and up to about 1.1 at the 90th percentile.
    pub margin: f64,
}

impl Default for PhysOptions {
    fn default() -> Self {
        PhysOptions { analytic_kappa: false, read_passes: 1, margin: 0.12 }
    }
}

/// The earlier program a stepover plan stands on: its moves, tools and stock top.
pub struct Prior<'m> {
    pub moves: &'m [Move],
    pub tools: &'m BTreeMap<i32, ToolDef>,
    pub stock_top: f64,
}

pub struct PhysPlan {
    pub ap: Vec<f64>,
    pub s_star: Vec<f64>,
    pub kappa: Vec<f64>,
    pub s_ref: f64,
    pub s_min: f64,
    pub s_cap: f64,
    /// Pocket nodes, and those where even s_min is unstable.
    pub nodes: usize,
    pub unstable: usize,
    pub trace: Vec<String>,
    margin: f64,
    nx: usize,
    ny: usize,
    h: f64,
    x0: f64,
    y0: f64,
    label: Vec<u8>,
}

impl PhysPlan {
    fn on(plan: &Plan, s_cap: f64, cut: &CutModel, margin: f64) -> PhysPlan {
        PhysPlan {
            ap: Vec::new(),
            s_star: Vec::new(),
            kappa: Vec::new(),
            s_ref: 0.0,
            s_min: 0.03 * cut.d,
            s_cap,
            nodes: 0,
            unstable: 0,
            trace: Vec::new(),
            margin,
            nx: plan.nx,
            ny: plan.ny,
            h: plan.h,
            x0: plan.x0,
            y0: plan.y0,
            label: plan.label.clone(),
        }
    }

    /// The physics plan: s*, then T re-solved with speed s* / s_ref. `constant`: start
    /// from one stepover everywhere instead; `feedback`: the old coupled version.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        prior: &Prior,
        i0: usize,
        z_cut: f64,
        plan: &mut Plan,
        cut: &CutModel,
        s_cap: f64,
        passes: usize,
        ap_scale: f64,
        constant: Option<f64>,
        feedback: bool,
        opts: PhysOptions,
    ) -> PhysPlan {
        let mut pp = PhysPlan::on(plan, s_cap, cut, opts.margin);
        pp.ap = height_before(prior, i0, plan, z_cut).into_iter().map(|a| a * ap_scale).collect();
        plan.solve_eikonal(None);
        let n = plan.nx * plan.ny;
        if let Some(c0) = constant {
            pp.s_star = vec![c0; n];
            pp.s_ref = c0;
            pp.kappa = vec![0.0; n];
            pp.nodes = plan.label.iter().filter(|&&l| l == 1).count();
            pp.trace.push(format!("start: constant s* {c0:.2} mm (the stable knee at the median depth)"));
            return pp;
        }
        if !feedback {
            let rho = plan.rho().max(0.5 * plan.h);
            for _pass in 1..=opts.read_passes.max(1) {
                if opts.analytic_kappa {
                    pp.kappa = (0..n).map(|s| if plan.label[s] == 1 { 1.0 / (plan.t[s].max(0.0) + rho) } else { 0.0 }).collect();
                } else {
                    pp.kappa = pp.front_curvature(&plan.t, 0.5 * cut.r(), cut.r());
                }
                pp.shape(cut);
                pp.s_ref = 0.1 * cut.d; // a fixed unit: T levels are k * s_ref apart in T, s*(x) apart on the part
                let speed: Vec<f64> = pp.s_star.iter().map(|s| s / pp.s_ref).collect();
                plan.solve_eikonal(Some(&speed));
                if opts.analytic_kappa {
                    break;
                }
            }
            pp.trace.push(pp.summary("geometry-only"));
            return pp;
        }
        for it in 0..passes {
            pp.curvature(plan, cut.r());
            pp.stepover(cut);
            let speed: Vec<f64> = pp.s_star.iter().map(|s| s / pp.s_ref).collect();
            plan.solve_eikonal(Some(&speed));
            pp.trace.push(pp.summary(&format!("pass {}:", it + 1)));
        }
        pp
    }

    /// Walls-together plan: the stepover field for a plan that is already solved (the
    /// conductive plan; nothing here changes it). kappa is read from that plan's own
    /// smoothed fronts, the target is the widest stable stepover with the margin, then
    /// Gaussian-smoothed and capped: all continuous.
    pub fn for_plan(prior: &Prior, i0: usize, z_cut: f64, plan: &Plan, cut: &CutModel, s_cap: f64, ap_scale: f64, opts: PhysOptions) -> PhysPlan {
        let mut pp = PhysPlan::on(plan, s_cap, cut, opts.margin);
        pp.ap = height_before(prior, i0, plan, z_cut).into_iter().map(|a| a * ap_scale).collect();
        pp.kappa = pp.front_curvature(&plan.t, 0.5 * cut.r(), cut.r());
        pp.shape(cut);
        pp.trace.push(pp.summary("walls plan"));
        pp
    }

    /// Measure, then correct: where the generated path's measured engagement runs past
    /// the stable engagement at the local depth, shrink s* on the nodes the tool covered
    /// there (by the ratio of the engaged widths), then re-solve T. Returns the cut length
    /// (mm) that was past the limit before the correction.
    pub fn correct(&mut self, plan: &mut Plan, path: &[Sample], cut: &CutModel) -> f64 {
        let n = self.nx * self.ny;
        let mut f = vec![1.0f64; n];
        let (mut over, r) = (0.0f64, cut.r());
        let mut prev: Option<&Sample> = None;
        for q in path {
            if let (false, Some(pv)) = (q.phi.is_nan(), prev) {
                let phi_max = cut.stable_phi(self.ap_at(q.x, q.y).clamp(0.1, cut.loc));
                if q.phi > phi_max {
                    over += ((q.x - pv.x) * (q.x - pv.x) + (q.y - pv.y) * (q.y - pv.y)).sqrt();
                    let ratio = ((1.0 - phi_max.cos()) / (1.0 - q.phi.cos()).max(1e-9)).clamp(0.2, 1.0);
                    let i0 = 0.max(((q.x - r - self.x0) / self.h) as i64);
                    let i1 = (self.nx as i64 - 1).min(((q.x + r - self.x0) / self.h) as i64 + 1);
                    let j0 = 0.max(((q.y - r - self.y0) / self.h) as i64);
                    let j1 = (self.ny as i64 - 1).min(((q.y + r - self.y0) / self.h) as i64 + 1);
                    for j in j0..=j1 {
                        for i in i0..=i1 {
                            let (x, y) = (self.x0 + i as f64 * self.h - q.x, self.y0 + j as f64 * self.h - q.y);
                            if x * x + y * y <= r * r {
                                let s = j as usize * self.nx + i as usize;
                                f[s] = f[s].min(ratio);
                            }
                        }
                    }
                }
            }
            prev = Some(q);
        }
        self.smooth(&mut f, 2);
        let mut speed = vec![0.0; n];
        for s in 0..n {
            if self.label[s] == 1 {
                self.s_star[s] = self.s_min.max(self.s_star[s] * f[s]);
            }
            speed[s] = self.s_star[s] / self.s_ref;
        }
        plan.solve_eikonal(Some(&speed));
        over
    }

    fn summary(&self, what: &str) -> String {
        format!(
            "{what} s* {:.2} / {:.2} / {:.2} mm (p5 / median / p95), {} of {} nodes unstable even at {:.2} mm",
            self.quant(0.05),
            self.s_ref,
            self.quant(0.95),
            self.unstable,
            self.nodes,
            self.s_min
        )
    }

    /// s* from kappa: the stable target, smoothed over half a tool diameter, capped,
    /// smoothed over a millimetre, capped again; then s_ref and the off-pocket fill.
    fn shape(&mut self, cut: &CutModel) {
        let n = self.nx * self.ny;
        let target = self.target(cut);
        let mut s1 = self.blur(&target, 0.5 * cut.d);
        for s in 0..n {
            s1[s] = s1[s].min(target[s]);
        }
        let s2 = self.blur(&s1, 1.0);
        self.s_star = (0..n).map(|s| s2[s].min(target[s])).collect();
        self.finish();
    }

    fn quant(&self, q: f64) -> f64 {
        let mut v: Vec<f64> = (0..self.s_star.len()).filter(|&s| self.label[s] == 1).map(|s| self.s_star[s]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() { 0.0 } else { v[(q * (v.len() - 1) as f64) as usize] }
    }

    /// Local axial depth at a point (bilinear on the plan grid).
    pub fn ap_at(&self, x: f64, y: f64) -> f64 {
        self.bilinear(&self.ap, x, y)
    }

    pub fn s_at(&self, x: f64, y: f64) -> f64 {
        self.bilinear(&self.s_star, x, y)
    }

    pub fn kappa_at(&self, x: f64, y: f64) -> f64 {
        self.bilinear(&self.kappa, x, y)
    }

    fn bilinear(&self, f: &[f64], x: f64, y: f64) -> f64 {
        let (fx, fy) = ((x - self.x0) / self.h, (y - self.y0) / self.h);
        let i = (fx.floor() as i64).clamp(0, self.nx as i64 - 2) as usize;
        let j = (fy.floor() as i64).clamp(0, self.ny as i64 - 2) as usize;
        let (u, v) = ((fx - i as f64).clamp(0.0, 1.0), (fy - j as f64).clamp(0.0, 1.0));
        let s = j * self.nx + i;
        (1.0 - u) * (1.0 - v) * f[s] + u * (1.0 - v) * f[s + 1] + (1.0 - u) * v * f[s + self.nx] + u * v * f[s + self.nx + 1]
    }

    /// Engaged angle for stepover s on a front of signed curvature kappa (tool radius r).
    pub fn engagement(s: f64, kappa: f64, r: f64) -> f64 {
        if kappa <= 1e-6 {
            return (1.0 - s / r).clamp(-1.0, 1.0).acos();
        }
        let big_r = 1.0 / kappa;
        let c = ((big_r - s + r) * (big_r - s + r) - big_r * big_r - r * r) / (2.0 * big_r * r);
        c.clamp(-1.0, 1.0).acos()
    }

    /// The widest stable stepover at each node (continuous in ap and kappa).
    fn target(&mut self, cut: &CutModel) -> Vec<f64> {
        let n = self.nx * self.ny;
        let mut t = vec![0.0; n];
        self.nodes = 0;
        self.unstable = 0;
        for s in 0..n {
            if self.label[s] != 1 {
                continue;
            }
            self.nodes += 1;
            let phi_max = (1.0 - self.margin) * cut.stable_phi(self.ap[s].clamp(0.1, cut.loc));
            if PhysPlan::engagement(self.s_min, self.kappa[s], cut.r()) > phi_max {
                t[s] = self.s_min;
                self.unstable += 1;
                continue;
            }
            if PhysPlan::engagement(self.s_cap, self.kappa[s], cut.r()) <= phi_max {
                t[s] = self.s_cap;
                continue;
            }
            let (mut lo, mut hi) = (self.s_min, self.s_cap);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if PhysPlan::engagement(mid, self.kappa[s], cut.r()) <= phi_max {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            t[s] = lo;
        }
        t
    }

    /// Signed curvature of the fronts of T, read after smoothing T over sigma on the
    /// pocket (normalized Gaussian): div(grad Ts / |grad Ts|), + where the front bulges
    /// toward the uncut stock, capped at half the tool radius. T in any monotone units:
    /// only the direction of its gradient matters.
    fn front_curvature(&self, t: &[f64], sigma: f64, r: f64) -> Vec<f64> {
        let (n, nx0, h) = (self.nx * self.ny, self.nx, self.h);
        let ts = self.blur(t, sigma);
        let (mut ux, mut uy) = (vec![0.0; n], vec![0.0; n]);
        let is_in = |m: usize| self.label[m] == 1;
        for j in 1..self.ny - 1 {
            for i in 1..nx0 - 1 {
                let s = j * nx0 + i;
                if !is_in(s) {
                    continue;
                }
                // One-sided where a neighbor is off the pocket.
                let gx = if is_in(s + 1) && is_in(s - 1) {
                    (ts[s + 1] - ts[s - 1]) / (2.0 * h)
                } else if is_in(s + 1) {
                    (ts[s + 1] - ts[s]) / h
                } else if is_in(s - 1) {
                    (ts[s] - ts[s - 1]) / h
                } else {
                    0.0
                };
                let gy = if is_in(s + nx0) && is_in(s - nx0) {
                    (ts[s + nx0] - ts[s - nx0]) / (2.0 * h)
                } else if is_in(s + nx0) {
                    (ts[s + nx0] - ts[s]) / h
                } else if is_in(s - nx0) {
                    (ts[s] - ts[s - nx0]) / h
                } else {
                    0.0
                };
                let g = (gx * gx + gy * gy).sqrt();
                if g > 1e-12 {
                    ux[s] = gx / g;
                    uy[s] = gy / g;
                }
            }
        }
        let mut k = vec![0.0; n];
        let cap = 1.0 / (0.5 * r);
        for j in 1..self.ny - 1 {
            for i in 1..nx0 - 1 {
                let s = j * nx0 + i;
                if !is_in(s) {
                    continue;
                }
                let dx = if is_in(s + 1) && is_in(s - 1) { (ux[s + 1] - ux[s - 1]) / (2.0 * h) } else { 0.0 };
                let dy = if is_in(s + nx0) && is_in(s - nx0) { (uy[s + nx0] - uy[s - nx0]) / (2.0 * h) } else { 0.0 };
                k[s] = (dx + dy).clamp(-cap, cap);
            }
        }
        self.blur(&k, 0.25 * r)
    }

    /// s_ref = median s* on the pocket; off the pocket s* = s_ref.
    fn finish(&mut self) {
        let mut v: Vec<f64> = (0..self.s_star.len()).filter(|&s| self.label[s] == 1).map(|s| self.s_star[s]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.s_ref = if v.is_empty() { self.s_cap } else { v[v.len() / 2] };
        for s in 0..self.s_star.len() {
            if self.label[s] != 1 {
                self.s_star[s] = self.s_ref;
            }
        }
    }

    /// Normalized Gaussian blur over the pocket's nodes (separable; nodes off the pocket
    /// neither contribute nor receive). Linear, so continuous in its input.
    fn blur(&self, f: &[f64], sigma: f64) -> Vec<f64> {
        let (nx0, ny0, h) = (self.nx, self.ny, self.h);
        let n = nx0 * ny0;
        let rr = 1.max((3.0 * sigma / h).ceil() as i64);
        let w: Vec<f64> = (-rr..=rr).map(|k| (-0.5 * (k as f64 * h / sigma) * (k as f64 * h / sigma)).exp()).collect();
        let (mut num, mut den) = (vec![0.0; n], vec![0.0; n]);
        for j in 0..ny0 {
            for i in 0..nx0 {
                let (mut a, mut b) = (0.0, 0.0);
                for k in -rr..=rr {
                    let ii = i as i64 + k;
                    if ii < 0 || ii >= nx0 as i64 {
                        continue;
                    }
                    let m = j * nx0 + ii as usize;
                    if self.label[m] != 1 {
                        continue;
                    }
                    a += w[(k + rr) as usize] * f[m];
                    b += w[(k + rr) as usize];
                }
                num[j * nx0 + i] = a;
                den[j * nx0 + i] = b;
            }
        }
        let mut o = vec![0.0; n];
        for j in 0..ny0 {
            for i in 0..nx0 {
                let s = j * nx0 + i;
                if self.label[s] != 1 {
                    o[s] = f[s];
                    continue;
                }
                let (mut a, mut b) = (0.0, 0.0);
                for k in -rr..=rr {
                    let jj = j as i64 + k;
                    if jj < 0 || jj >= ny0 as i64 {
                        continue;
                    }
                    let m = jj as usize * nx0 + i;
                    a += w[(k + rr) as usize] * num[m];
                    b += w[(k + rr) as usize] * den[m];
                }
                o[s] = if b > 0.0 { a / b } else { f[s] };
            }
        }
        o
    }

    fn stepover(&mut self, cut: &CutModel) {
        let n = self.nx * self.ny;
        self.s_star = vec![0.0; n];
        self.nodes = 0;
        self.unstable = 0;
        for s in 0..n {
            if self.label[s] != 1 {
                continue;
            }
            self.nodes += 1;
            let phi_max = cut.stable_phi(self.ap[s].clamp(0.1, cut.loc));
            if PhysPlan::engagement(self.s_min, self.kappa[s], cut.r()) > phi_max {
                self.s_star[s] = self.s_min;
                self.unstable += 1;
                continue;
            }
            if PhysPlan::engagement(self.s_cap, self.kappa[s], cut.r()) <= phi_max {
                self.s_star[s] = self.s_cap;
                continue;
            }
            let (mut lo, mut hi) = (self.s_min, self.s_cap);
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                if PhysPlan::engagement(mid, self.kappa[s], cut.r()) <= phi_max {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            self.s_star[s] = lo;
        }
        let mut s_star = std::mem::take(&mut self.s_star);
        self.smooth(&mut s_star, 4);
        self.s_star = s_star;
        let mut vals: Vec<f64> = (0..n).filter(|&s| self.label[s] == 1).map(|s| self.s_star[s]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.s_ref = if vals.is_empty() { self.s_cap } else { vals[vals.len() / 2] };
        for s in 0..n {
            if self.label[s] != 1 {
                self.s_star[s] = self.s_ref;
            }
        }
    }

    /// Signed curvature of the fronts of T: div(grad T / |grad T|), smoothed and capped
    /// at the tool radius (a tighter front is a shock or a seed).
    fn curvature(&mut self, plan: &Plan, r: f64) {
        let (n, nx0, h) = (self.nx * self.ny, self.nx, self.h);
        let (mut ux, mut uy) = (vec![0.0; n], vec![0.0; n]);
        for j in 1..self.ny - 1 {
            for i in 1..nx0 - 1 {
                let s = j * nx0 + i;
                if self.label[s] != 1 {
                    continue;
                }
                let gx = (plan.t[s + 1] - plan.t[s - 1]) / (2.0 * h);
                let gy = (plan.t[s + nx0] - plan.t[s - nx0]) / (2.0 * h);
                let g = (gx * gx + gy * gy).sqrt();
                if g > 1e-9 {
                    ux[s] = gx / g;
                    uy[s] = gy / g;
                }
            }
        }
        let mut kappa = vec![0.0; n];
        let cap = 1.0 / (0.5 * r);
        for j in 1..self.ny - 1 {
            for i in 1..nx0 - 1 {
                let s = j * nx0 + i;
                if self.label[s] != 1 {
                    continue;
                }
                let all = self.label[s - 1] == 1 && self.label[s + 1] == 1 && self.label[s - nx0] == 1 && self.label[s + nx0] == 1;
                kappa[s] = if all { ((ux[s + 1] - ux[s - 1] + uy[s + nx0] - uy[s - nx0]) / (2.0 * h)).clamp(-cap, cap) } else { 0.0 };
            }
        }
        self.smooth(&mut kappa, 4);
        self.kappa = kappa;
    }

    fn smooth(&self, f: &mut [f64], passes: usize) {
        let nx0 = self.nx;
        for _ in 0..passes {
            let mut g = f.to_vec();
            for j in 1..self.ny - 1 {
                for i in 1..nx0 - 1 {
                    let s = j * nx0 + i;
                    if self.label[s] != 1 {
                        continue;
                    }
                    let (mut sum, mut c) = (4.0 * f[s], 4);
                    for m in [s - 1, s + 1, s - nx0, s + nx0] {
                        if self.label[m] == 1 {
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

/// Stock height above the cut level before move `i1` of the prior program, on the plan's
/// nodes: the lowest any earlier cutting move's tool reaches over each node, exact for
/// every tool shape and motion (`icam_stock::StockTop`), the nodes read in parallel.
pub fn height_before(prior: &Prior, i1: usize, p: &Plan, z_cut: f64) -> Vec<f64> {
    let (x1, y1) = (p.x0 + (p.nx - 1) as f64 * p.h, p.y0 + (p.ny - 1) as f64 * p.h);
    let top = StockTop::new(prior.moves, i1, prior.tools, prior.stock_top, Box2::new(p.x0, p.y0, x1, y1));
    (0..p.nx * p.ny).into_par_iter().map(|s| (top.at(p.x0 + (s % p.nx) as f64 * p.h, p.y0 + (s / p.nx) as f64 * p.h) - z_cut).max(0.0)).collect()
}
