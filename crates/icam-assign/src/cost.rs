//! Seconds per mm2 of floor for one tool to clear a column of height H in a channel of
//! width W, inside that tool's physical limits.
//!
//! Channel model ("open a lane, then widen it"), per unit channel length:
//!
//!   T(W, H) = min over lane width W1 in [D, W] of  lane(W1, H) + side(W - W1, H)
//!
//!   lane(D, H)    a full slot (phi = 180 deg), in n axial passes
//!   lane(W1 > D)  circular loops of radius rho = (W1 - D)/2 stepping ae per loop; the
//!                 peak engagement is the internal-circle formula (a tool orbiting inside
//!                 a bore of radius W1/2 taking a radial step ae). Half of each loop cuts
//!                 at min(ceiling(phi), sqrt(a rho)), half returns in air at min(feed cap,
//!                 sqrt(a rho)): the centripetal limit is what makes tight loops slow.
//!   side(Wr, H)   m straight side passes of ae = Wr / m at the straight engagement, each
//!                 in n axial passes.
//!
//! Every cut obeys the tool's feed ceilings at its engagement and pass depth and, when
//! enabled, the vendor wide-cut envelope. Cost per area = T / W: as W grows the lane's
//! share vanishes and the cost tends to the open-area side-milling rate; at W = D it is a
//! slot. Every channel is treated as long; in a short room this overstates the lane's
//! share, by at most D / W.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::tools::ToolSpec;

const G: f64 = 1.06;

pub struct CostTable {
    pub tool: ToolSpec,
    pub accel: f64,
    /// Channel widths, geometric from D.
    pub wg: Vec<f64>,
    /// Column heights 0, h_step, ...
    pub h_step: f64,
    pub nh: usize,
    /// s/mm2, [w][h] as w * nh + h.
    pub per_area: Vec<f32>,
    /// 0 slot lane, 1 loop lane.
    pub lane_kind: Vec<u8>,
    /// Best open-area steady removal, mm3/s, at the best depth (for bounds).
    pub open_rate: f64,
}

impl CostTable {
    pub fn new(t: &ToolSpec, accel: f64, w_max: f64, h_max: f64) -> CostTable {
        let h_step = 0.25;
        let mut wg = Vec::new();
        let mut w = t.d();
        while w < w_max * G {
            wg.push(w);
            w *= G;
        }
        let nh = (h_max / h_step).ceil() as usize + 1;
        let mut c = CostTable { tool: t.clone(), accel, per_area: vec![0.0; wg.len() * nh], lane_kind: vec![0; wg.len() * nh], wg, h_step, nh, open_rate: 0.0 };
        c.build();
        c
    }

    /// Time per unit length of one straight cut at engagement phi through height H (best
    /// axial split), and the number of axial passes.
    pub fn pass(&self, phi: f64, h: f64) -> (f64, i32) {
        if h <= 1e-6 {
            return (0.0, 0);
        }
        let t = &self.tool;
        let env_ap = t.env_ap_phi(phi);
        if env_ap < 0.05 {
            return (f64::INFINITY, 0);
        }
        let n0 = 1.max((h / env_ap - 1e-9).ceil() as i32);
        let (mut best, mut n_best) = (f64::INFINITY, 0);
        for n in n0..=n0 + 4 {
            let ap = h / n as f64;
            let v = t.v(phi, ap);
            if v <= 1e-6 {
                continue;
            }
            let tt = n as f64 / v;
            if tt < best {
                best = tt;
                n_best = n;
            }
        }
        (best, n_best)
    }

    /// Loop lane of width w1 > D through height h: time per unit length.
    fn loops(&self, w1: f64, h: f64) -> f64 {
        let t = &self.tool;
        let (r, rho, rw) = (t.r(), (w1 - t.d()) / 2.0, w1 / 2.0);
        if rho < 0.02 {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        for k in 1..=24 {
            let ae = t.d() * k as f64 / 24.0;
            // Internal-circle engagement: tool center at rho from the bore axis, prior radius rw - ae.
            let c = (rho * rho + r * r - (rw - ae) * (rw - ae)) / (2.0 * rho * r);
            let phi = PI - c.clamp(-1.0, 1.0).acos();
            let env_ap = t.env_ap_phi(phi);
            if env_ap < 0.05 {
                continue;
            }
            // One loop of the trochoid: its length is about sqrt((2 pi rho)^2 + ae^2); the
            // lateral swing needs (2 pi v / L)^2 rho <= a; the share in air goes from 0 (a
            // wavy slot) to one half (wide loops).
            let len = (4.0 * PI * PI * rho * rho + ae * ae).sqrt();
            let v_kin = len * (self.accel / rho).sqrt() / (2.0 * PI);
            let air = 0.5 * ((2.0 * PI * rho - ae) / (2.0 * PI * rho + ae)).clamp(0.0, 1.0);
            let v_air = t.model.v_max.min(v_kin);
            let n0 = 1.max((h / env_ap - 1e-9).ceil() as i32);
            for n in n0..=n0 + 2 {
                let vc = t.v(phi, h / n as f64).min(v_kin);
                if vc <= 1e-6 {
                    continue;
                }
                let per_loop = len * (1.0 - air) / vc + len * air / v_air;
                // No path clears floor faster than the tool's own width times its feed.
                let tt = n as f64 * (per_loop / ae).max(w1 / (t.d() * vc));
                if tt < best {
                    best = tt;
                }
            }
        }
        best
    }

    /// m straight side passes clearing a width wr: time per unit length.
    fn side(&self, wr: f64, h: f64) -> f64 {
        if wr <= 1e-6 {
            return 0.0;
        }
        let t = &self.tool;
        let mut best = f64::INFINITY;
        for k in 1..=40 {
            let ae_try = t.d() * k as f64 / 40.0;
            let m = (wr / ae_try - 1e-9).ceil() as i32;
            let ae = wr / m as f64;
            let phi = (1.0 - ae / t.r()).clamp(-1.0, 1.0).acos();
            let tt = m as f64 * self.pass(phi, h).0;
            if tt < best {
                best = tt;
            }
        }
        best
    }

    fn build(&mut self) {
        let (nw, nh, h_step) = (self.wg.len(), self.nh, self.h_step);
        let d = self.tool.d();
        // Side-pass time on a fine grid of remaining widths (geometric).
        let last = self.wg[nw - 1];
        let mut wr = vec![0.0];
        let mut w = 0.02 * d;
        while w < last {
            wr.push(w);
            w *= 1.04;
        }
        wr.push(last);
        // Per height: the side times, then the lanes (width D is the slot; a wider lane is
        // cut in loops; a slot plus side passes is the i1 = 0 choice below).
        let cols: Vec<(Vec<f64>, Vec<f64>)> = (0..nh)
            .into_par_iter()
            .map(|h| {
                let hh = h as f64 * h_step;
                let side: Vec<f64> = wr.iter().map(|&x| self.side(x, hh)).collect();
                let mut lane = vec![self.pass(PI, hh).0];
                lane.extend((1..nw).map(|i| self.loops(self.wg[i], hh)));
                (side, lane)
            })
            .collect();
        let side_at = |w: f64, h: usize| -> f64 {
            let side = &cols[h].0;
            if w <= 0.0 {
                return 0.0;
            }
            match wr.binary_search_by(|p| p.partial_cmp(&w).unwrap_or(std::cmp::Ordering::Less)) {
                Ok(k) => side[k],
                Err(k) if k >= wr.len() => side[wr.len() - 1] * w / wr[wr.len() - 1],
                Err(k) => {
                    // The side time grows with width; interpolate between the neighbors.
                    let (w0, w1) = (wr[k - 1], wr[k]);
                    let a = (w - w0) / (w1 - w0);
                    side[k - 1] + (side[k] - side[k - 1]) * a
                }
            }
        };
        let wg = &self.wg;
        let cells: Vec<Vec<(f32, u8)>> = (0..nh)
            .into_par_iter()
            .map(|h| {
                (0..nw)
                    .map(|i| {
                        if h == 0 {
                            return (0.0f32, 0u8);
                        }
                        let (mut best, mut kind) = (f64::INFINITY, 0u8);
                        for i1 in 0..=i {
                            let tt = cols[h].1[i1] + side_at(wg[i] - wg[i1], h);
                            if tt < best {
                                best = tt;
                                kind = if i1 == 0 { 0 } else { 1 };
                            }
                        }
                        ((best / wg[i]) as f32, kind)
                    })
                    .collect()
            })
            .collect();
        for (h, row) in cells.iter().enumerate() {
            for (i, &(v, k)) in row.iter().enumerate() {
                self.per_area[i * nh + h] = v;
                self.lane_kind[i * nh + h] = k;
            }
        }
        // Best open-area removal anywhere (for the absolute bound): the widest channel at each height.
        self.open_rate = 0.0;
        for h in 1..nh {
            self.open_rate = self.open_rate.max(h as f64 * h_step / self.per_area[(nw - 1) * nh + h] as f64);
        }
    }

    fn pa(&self, i: usize, j: usize) -> f64 {
        self.per_area[i * self.nh + j] as f64
    }

    /// s/mm2 at channel width W and column height H (bilinear in log W and H).
    pub fn at(&self, w: f64, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        let d = self.tool.d();
        let fw = (w.max(d) / d).ln() / G.ln();
        let i = (fw as usize).min(self.wg.len() - 2);
        let a = (fw - i as f64).clamp(0.0, 1.0);
        let fh = h / self.h_step;
        let mut j = (fh as usize).min(self.nh - 2);
        let mut b = (fh - j as f64).clamp(0.0, 1.0);
        if fh > (self.nh - 1) as f64 {
            j = self.nh - 2;
            b = 1.0;
        }
        let (v00, v10, v01, v11) = (self.pa(i, j), self.pa(i + 1, j), self.pa(i, j + 1), self.pa(i + 1, j + 1));
        if (v00 + v10 + v01 + v11).is_infinite() {
            return f64::INFINITY;
        }
        (v00 * (1.0 - a) + v10 * a) * (1.0 - b) + (v01 * (1.0 - a) + v11 * a) * b
    }

    pub fn kind(&self, w: f64, h: f64) -> u8 {
        let d = self.tool.d();
        let fw = (w.max(d) / d).ln() / G.ln();
        let i = (fw.round() as i64).clamp(0, self.wg.len() as i64 - 1) as usize;
        let j = ((h / self.h_step).round() as i64).clamp(0, self.nh as i64 - 1) as usize;
        self.lane_kind[i * self.nh + j]
    }
}
