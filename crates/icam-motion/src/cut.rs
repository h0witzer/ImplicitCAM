//! Physical cut limits for one tool in one material on one machine. Each limit is a
//! ceiling on the feed v at a sample's measured engagement angle phi and the op's axial
//! depth ap:
//!
//!   chip        h_max = fz * s(phi) <= h_ceil                 (radial chip thinning)
//!   power       kc(h_m) * ae * ap * v <= eff * spindle_w      Kienzle: kc = kc_ref (h_m / h_ref)^-mc,
//!                                                             h_m = fz (1 - cos phi) / phi
//!   deflection  peak resultant force x tool compliance <= defl_max
//!               (force integrated along the helical flutes, both climb and conventional)
//!   machine     v <= v_max
//!
//! and one floor: h_max >= h_floor (below it the edge rubs instead of cutting).
//!
//! Chatter (`stable_depth`): the lobe-independent minimum stable axial depth at an
//! engagement, Altintas-Budak zeroth-order solution for a symmetric tool with one mode
//! per direction (static tip stiffness from the same cantilever, damping ratio assumed).
//! It is the depth that is stable at EVERY spindle speed, so it is conservative; holder
//! and spindle compliance are not included (optimistic). Unlike the feed ceilings, feed
//! cannot cure it: only a narrower or shallower cut can. The scheduler runs the fastest
//! feed inside every ceiling; the floor is checked on the feed the machine actually runs
//! (after curvature and acceleration).
//!
//! Aluminum 6061-T6 is the proving material: it has more published cut data than
//! anything else, so the path logic can be proven before the shop's own materials. Each
//! constant names its source; "assumed" marks the ones no source on disk pins.

use std::f64::consts::PI;
use std::sync::OnceLock;

use icam_post::ToolDef;

use crate::complex::C;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    Chip,
    Power,
    Deflection,
    Machine,
    Kinematics,
}

const IN: f64 = 25.4;
const HP: f64 = 745.7;

#[derive(Clone, Debug)]
pub struct CutModel {
    pub name: String,
    // Tool.
    pub d: f64,
    pub flutes: f64,
    pub helix_deg: f64,
    pub loc: f64,
    pub hangout: f64,
    pub core_ratio: f64,
    pub e_modulus: f64,
    // Material.
    pub h_ceil: f64,
    pub h_floor: f64,
    pub kc_ref: f64,
    pub h_ref: f64,
    pub mc: f64,
    pub kr: f64,
    // Machine.
    pub rpm: f64,
    pub spindle_w: f64,
    pub eff: f64,
    pub v_max: f64,
    pub defl_max: f64,
    // Per op.
    pub ap: f64,
    /// Modal damping ratio of the tool mode (assumed).
    pub damping: f64,
    pub sources: String,
    tip_compliance: f64, // mm/N at the tool tip
    g: Vec<f64>,         // peak resultant force, N per (fz in mm)^(1 - mc), per whole degree of phi 0..180
    compliance: f64,     // tool deflection per unit force at the engaged band's centroid, mm/N
    stab: OnceLock<Vec<f64>>, // stable depth by half degree of phi 0..180 (independent of ap), built on first use
}

impl CutModel {
    /// 6061-T6 with a 1/2 in carbide 2-flute upcut on the router.
    pub fn al6061(t: &ToolDef, rpm: Option<f64>) -> Result<CutModel, String> {
        if (t.diameter - 12.7).abs() > 0.01 {
            return Err(format!("al6061: the model carries the 1/2 in 2-flute (T04) only, not a {:.3} mm tool", t.diameter));
        }
        Ok(CutModel {
            name: "6061-T6, 1/2 in carbide 2-flute upcut, MultiCAM".into(),
            // Tool: the program's tool sheet (T04 "1/2 EM CRB 2FL 1 LOC", hangout 2.000 in).
            d: t.diameter,
            flutes: 2.0,
            loc: 1.0 * IN,
            hangout: 2.0 * IN,
            helix_deg: 30.0,   // assumed: common 2-flute aluminum helix; the tool sheet does not say
            core_ratio: 0.8,   // assumed: fluted section bends like a solid rod of 0.8 D
            e_modulus: 600_000.0, // N/mm2, cemented carbide (typical 550-650 GPa)
            // Chip ceiling: Amana #46462 SC upcut 2FL 1/2 in, aluminum 0.006-0.008 in (Amana's
            // published chip loads); Vortex and Freud 1/2 in bands 0.008-0.010.
            h_ceil: 0.007 * IN,
            // Floor: the thinnest chip Datron publishes for a 10 mm cutter in aluminum (1 flute,
            // ae 0.01 D, fz 0.00197 in -> h_max 0.0004 in; Datron's HSC aluminum tables).
            h_floor: 0.0004 * IN,
            // Specific cutting energy: Kp 0.25 hp/(in3/min) = 683 N/mm2, the unit power commonly
            // published for aluminum alloys (W / (mm3/s) x 1000 = N/mm2), taken at a 0.010 in
            // chip (assumed); Kienzle exponent 0.25 (Sandvik, aluminum groups).
            kc_ref: 0.25 * HP / (IN.powf(3.0) / 60.0) * 1000.0,
            h_ref: 0.010 * IN,
            mc: 0.25,
            kr: 0.3, // assumed: radial / tangential force ratio for aluminum
            // Machine: MultiCAM 3000, 24,000 rpm, 13.6 hp, 1,000 ipm cutting (its specification).
            rpm: rpm.unwrap_or(18000.0), // Amana aluminum chart operating rpm 18,000
            spindle_w: 13.6 * HP,
            eff: 0.85, // assumed
            v_max: 1000.0 * IN / 60.0,
            defl_max: 0.001 * IN, // assumed: roughing rule of thumb; excludes the gantry's own compliance
            ap: 0.0,
            damping: 0.03,
            sources: "chip 0.006-0.008 in (Amana 2FL 1/2 in); floor 0.0004 in (Datron); Kp 0.25 hp/(in3/min) (published unit power, aluminum alloys); machine 13.6 hp / 1000 ipm; assumed: helix 30 deg, Kr 0.3, Eff 0.85, deflection 0.001 in, kc at a 0.010 in chip".into(),
            tip_compliance: 0.0,
            g: Vec::new(),
            compliance: 0.0,
            stab: OnceLock::new(),
        })
    }

    /// A model to fill in field by field: every constant zero,
    /// the damping ratio at its assumed 0.03.
    pub fn blank(name: &str) -> CutModel {
        CutModel {
            name: name.into(),
            d: 0.0,
            flutes: 0.0,
            helix_deg: 0.0,
            loc: 0.0,
            hangout: 0.0,
            core_ratio: 0.0,
            e_modulus: 0.0,
            h_ceil: 0.0,
            h_floor: 0.0,
            kc_ref: 0.0,
            h_ref: 0.0,
            mc: 0.0,
            kr: 0.0,
            rpm: 0.0,
            spindle_w: 0.0,
            eff: 0.0,
            v_max: 0.0,
            defl_max: 0.0,
            ap: 0.0,
            damping: 0.03,
            sources: String::new(),
            tip_compliance: 0.0,
            g: Vec::new(),
            compliance: 0.0,
            stab: OnceLock::new(),
        }
    }

    pub fn for_material(material: &str, t: &ToolDef, rpm: Option<f64>) -> Result<CutModel, String> {
        match material.to_lowercase().as_str() {
            "al6061" | "6061" | "aluminum" => CutModel::al6061(t, rpm),
            _ => Err(format!("unknown material '{material}' (known: al6061)")),
        }
    }

    pub fn r(&self) -> f64 {
        self.d / 2.0
    }

    fn n(&self) -> f64 {
        self.rpm / 60.0 // rev/s
    }

    /// Feed (mm/s) at a feed per tooth.
    pub fn v(&self, fz: f64) -> f64 {
        fz * self.flutes * self.n()
    }

    /// Feed per tooth (mm) at a feed.
    pub fn fz(&self, v: f64) -> f64 {
        v / (self.flutes * self.n())
    }

    /// Radial engagement at an engagement angle.
    pub fn ae(&self, phi: f64) -> f64 {
        self.r() * (1.0 - phi.min(PI).cos())
    }

    pub fn compliance(&self) -> f64 {
        self.compliance
    }

    /// The same model at one axial depth (clamped to the flute length).
    pub fn at_depth(&self, ap: f64) -> CutModel {
        let mut m = self.clone();
        m.ap = ap.clamp(0.1, self.loc);
        m.g = peak_force(self.r(), self.flutes as i32, self.helix_deg, m.ap, self.kr, self.mc, self.kc_ref * self.h_ref.powf(self.mc));
        // Load at the centroid of the engaged band, a from the collet; the shank is full
        // diameter up to the flutes, the fluted section core_ratio D.
        let a = self.hangout - m.ap / 2.0;
        let i_s = PI * self.d.powf(4.0) / 64.0;
        let i_f = PI * (self.core_ratio * self.d).powf(4.0) / 64.0;
        const N: usize = 400;
        let mut c = 0.0;
        for k in 0..N {
            let x = (k as f64 + 0.5) * a / N as f64;
            c += (a - x) * (a - x) / (self.e_modulus * if x < self.hangout - self.loc { i_s } else { i_f }) * a / N as f64;
        }
        m.compliance = c;
        let mut ct = 0.0;
        for k in 0..N {
            let x = (k as f64 + 0.5) * self.hangout / N as f64;
            ct += (self.hangout - x) * (self.hangout - x) / (self.e_modulus * if x < self.hangout - self.loc { i_s } else { i_f }) * self.hangout / N as f64;
        }
        m.tip_compliance = ct;
        m
    }

    /// Radial chip thinning: the share of fz that is the thickest chip.
    pub fn s(phi: f64) -> f64 {
        if phi >= PI / 2.0 { 1.0 } else { phi.sin() }
    }

    /// The feed at which each ceiling binds at engagement phi (mm/s): chip, power, deflection.
    pub fn ceilings(&self, phi: f64) -> (f64, f64, f64) {
        let phi = phi.min(PI);
        if phi < 1e-4 {
            return (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        }
        let e = 1.0 - self.mc;
        let v_chip = self.v(self.h_ceil / CutModel::s(phi));
        // Power(fz) = A fz^(1 - mc), A = kc_ref (c / h_ref)^-mc ae ap z n, W = N mm/s / 1000.
        let c = (1.0 - phi.cos()) / phi;
        let a = self.kc_ref * (c / self.h_ref).powf(-self.mc) * self.ae(phi) * self.ap * self.flutes * self.n() / 1000.0;
        let v_pow = self.v((self.eff * self.spindle_w / a).powf(1.0 / e));
        let g = self.g_at(phi);
        let v_def = if g <= 0.0 { f64::INFINITY } else { self.v((self.defl_max / (self.compliance * g)).powf(1.0 / e)) };
        (v_chip, v_pow, v_def)
    }

    /// Fastest feed inside every ceiling, and the limit that sets it.
    pub fn ceiling(&self, phi: f64) -> (f64, Limit) {
        let (c, p, d) = self.ceilings(phi);
        let (mut v, mut l) = (self.v_max, Limit::Machine);
        if c < v {
            v = c;
            l = Limit::Chip;
        }
        if p < v {
            v = p;
            l = Limit::Power;
        }
        if d < v {
            v = d;
            l = Limit::Deflection;
        }
        (v, l)
    }

    /// Worst ratio to its limit at feed v: chip thickness, power, deflection.
    pub fn load(&self, phi: f64, v: f64) -> f64 {
        let (c, p, d) = self.ceilings(phi);
        let e = 1.0 - self.mc;
        (v / c).max((v / p).powf(e).max((v / d).powf(e)))
    }

    pub fn rubs(&self, phi: f64, v: f64) -> bool {
        self.fz(v) * CutModel::s(phi.min(PI)) < self.h_floor
    }

    pub fn deflection(&self, phi: f64, v: f64) -> f64 {
        self.compliance * self.g_at(phi) * self.fz(v).powf(1.0 - self.mc)
    }

    pub fn power_w(&self, phi: f64, v: f64) -> f64 {
        let phi = phi.min(PI);
        // No engagement or no feed cuts nothing (the chip-thickness term alone would
        // read 0 x inf at zero feed).
        if phi < 1e-4 || v <= 0.0 {
            return 0.0;
        }
        let fz = self.fz(v);
        let hm = fz * (1.0 - phi.cos()) / phi;
        self.kc_ref * (hm / self.h_ref).powf(-self.mc) * self.ae(phi) * self.ap * v / 1000.0
    }

    /// Steady removal rate at radial engagement ae (mm3/s), at the fastest allowed feed.
    pub fn steady_mrr(&self, ae: f64) -> f64 {
        let phi = (1.0 - ae / self.r()).clamp(-1.0, 1.0).acos();
        ae * self.ap * self.ceiling(phi).0
    }

    /// The narrowest radial engagement that reaches (1 - slack) of the best steady removal
    /// rate at or below `ae_max`: past it a wider cut buys no time, only load. Returns
    /// (ae, its removal rate, the best).
    pub fn knee(&self, ae_max: f64, slack: f64) -> (f64, f64, f64) {
        let (mut best, step) = (0.0f64, 0.005 * self.d);
        let mut ae = step;
        while ae <= ae_max + 1e-9 {
            best = best.max(self.steady_mrr(ae));
            ae += step;
        }
        let mut ae = step;
        while ae <= ae_max + 1e-9 {
            let q = self.steady_mrr(ae);
            if q >= (1.0 - slack) * best {
                return (ae, q, best);
            }
            ae += step;
        }
        (ae_max, self.steady_mrr(ae_max), best)
    }

    /// Static tool-tip stiffness, N/mm.
    pub fn tip_stiffness(&self) -> f64 {
        1.0 / self.tip_compliance
    }

    fn stab(&self) -> &[f64] {
        self.stab.get_or_init(|| (0..361).map(|i| self.stable_depth((i as f64 * 0.5).max(0.25) * PI / 180.0)).collect())
    }

    /// Stable depth at an engagement, from the table by half degree.
    pub fn stable_depth_at(&self, phi: f64) -> f64 {
        let d = (phi * 180.0 / PI).clamp(0.0, 180.0) * 2.0;
        let t = self.stab();
        let i = 359.min(d as usize);
        t[i] + (t[i + 1] - t[i]) * (d - i as f64)
    }

    /// Widest engagement (rad) that is stable at axial depth `ap`.
    pub fn stable_phi(&self, ap: f64) -> f64 {
        let t = self.stab();
        if t[360] >= ap {
            return PI;
        }
        for i in 1..=360 {
            if t[i] < ap {
                return (i as f64 - 1.0 + (t[i - 1] - ap) / (t[i - 1] - t[i])) * 0.5 * PI / 180.0;
            }
        }
        PI
    }

    pub fn stable_ae(&self, ap: f64) -> f64 {
        let phi = self.stable_phi(ap);
        self.ae(phi)
    }

    /// Lobe-independent minimum stable axial depth (mm) at engagement phi, the lower of
    /// climb and conventional. kt is the Kienzle kc at a mid-window chip.
    pub fn stable_depth(&self, phi: f64) -> f64 {
        let phi = phi.clamp(1e-3, PI);
        let kr = self.kr;
        let mut best = f64::INFINITY;
        for climb in 0..2 {
            let (st, ex) = if climb == 1 { (PI - phi, PI) } else { (0.0, phi) };
            let axx_f = |f: f64| 0.5 * ((2.0 * f).cos() - 2.0 * kr * f + kr * (2.0 * f).sin());
            let axy_f = |f: f64| 0.5 * (-(2.0 * f).sin() - 2.0 * f + kr * (2.0 * f).cos());
            let ayx_f = |f: f64| 0.5 * (-(2.0 * f).sin() + 2.0 * f + kr * (2.0 * f).cos());
            let ayy_f = |f: f64| 0.5 * (-(2.0 * f).cos() - 2.0 * kr * f - kr * (2.0 * f).sin());
            best = best.min(self.limit(axx_f(ex) - axx_f(st), axy_f(ex) - axy_f(st), ayx_f(ex) - ayx_f(st), ayy_f(ex) - ayy_f(st)));
        }
        best
    }

    /// Lobe-independent minimum stable depth (mm) at the deepest point of a full-width cut
    /// whose axial depth varies along the engaged half turn: `w(phi)` in [0, 1] is the
    /// depth at immersion phi (0 to pi) as a share of the deepest. The directional
    /// coefficients (the derivatives of `stable_depth`'s closed forms) are integrated with
    /// that weight; with w = 1 this is the full-slot limit. The lower of w and its mirror
    /// image (either way round).
    pub fn stable_depth_profile(&self, w: impl Fn(f64) -> f64) -> f64 {
        const N: usize = 360;
        let (mut a, mut s, mut c) = (0.0, 0.0, 0.0);
        for i in 0..N {
            let f = PI * (i as f64 + 0.5) / N as f64;
            let wt = w(f).clamp(0.0, 1.0) * PI / N as f64;
            a += wt;
            s += wt * (2.0 * f).sin();
            c += wt * (2.0 * f).cos();
        }
        self.stable_depth_moments(a, s, c).min(self.stable_depth_moments(a, -s, c))
    }

    /// The limit of [`CutModel::stable_depth_profile`] from the weight's moments over phi in
    /// (0, pi): a = int w, s = int w sin 2 phi, c = int w cos 2 phi (the directional
    /// coefficients are linear in them). Its mirror image is (a, -s, c).
    pub fn stable_depth_moments(&self, a: f64, s: f64, c: f64) -> f64 {
        let kr = self.kr;
        self.limit(-s - kr * a + kr * c, -c - a - kr * s, -c + a - kr * s, s - kr * a - kr * c)
    }

    /// The least positive critical depth over chatter frequencies for the directional
    /// coefficients [alpha] (Altintas and Budak's zeroth-order solution).
    fn limit(&self, axx: f64, axy: f64, ayx: f64, ayy: f64) -> f64 {
        let kt = self.kc_ref * (0.5 * self.h_ceil / self.h_ref).powf(-self.mc);
        let (k, z) = (self.tip_stiffness(), self.damping);
        let mut best = f64::INFINITY;
        // Eigenvalues mu of [alpha]; Lambda = -1 / (mu G), G = 1 / (k (1 - r^2 + 2 i z r)).
        let tr = axx + ayy;
        let disc = C::sqrt_real((tr * tr - 0.0 * 0.0) - 4.0 * (axx * ayy - axy * ayx));
        for mu in [C::new((tr + disc.re) / 2.0, (0.0 + disc.im) / 2.0), C::new((tr - disc.re) / 2.0, (0.0 - disc.im) / 2.0)] {
            if mu.magnitude() < 1e-12 {
                continue;
            }
            let mut r = 0.3;
            while r <= 3.0 {
                let lam = C::new(1.0 - r * r, 2.0 * z * r).scale(-k).div(mu);
                if lam.re < 0.0 {
                    let kap = lam.im / lam.re;
                    let a = -2.0 * PI * lam.re * (1.0 + kap * kap) / (self.flutes * kt);
                    if a > 0.0 && a < best {
                        best = a;
                    }
                }
                r += 0.0005;
            }
        }
        best
    }

    fn g_at(&self, phi: f64) -> f64 {
        let deg = (phi * 180.0 / PI).clamp(0.0, 180.0);
        let i = 179.min(deg as usize);
        self.g[i] + (self.g[i + 1] - self.g[i]) * (deg - i as f64)
    }
}

/// Peak over one revolution of the XY resultant (tangential + kr radial) on z helical
/// flutes over ap, for a unit feed per tooth; chip h = fz sin(theta). The engaged arc is
/// taken both at the start (conventional) and at the end (climb) of the half turn, and
/// the larger peak kept.
fn peak_force(r: f64, z: i32, helix_deg: f64, ap: f64, kr: f64, mc: f64, kc_coef: f64) -> Vec<f64> {
    const N_DISC: i32 = 24;
    const N_PSI: i32 = 240;
    let lag = (helix_deg * PI / 180.0).tan() / r;
    let dz = ap / N_DISC as f64;
    let e = 1.0 - mc;
    let mut g = vec![0.0; 181];
    for deg in 1..=180 {
        let phi = deg as f64 * PI / 180.0;
        let mut best = 0.0f64;
        for climb in 0..2 {
            for k in 0..N_PSI {
                let psi = 2.0 * PI * k as f64 / N_PSI as f64;
                let (mut fx, mut fy) = (0.0, 0.0);
                for j in 0..z {
                    for dd in 0..N_DISC {
                        let mut th = psi + 2.0 * PI * j as f64 / z as f64 - (dd as f64 + 0.5) * dz * lag;
                        th %= 2.0 * PI;
                        if th < 0.0 {
                            th += 2.0 * PI;
                        }
                        let on = if climb == 1 { th >= PI - phi && th <= PI } else { th <= phi };
                        if !on {
                            continue;
                        }
                        let (sn, cs) = (th.sin(), th.cos());
                        if sn <= 0.0 {
                            continue;
                        }
                        let ft = kc_coef * sn.powf(e) * dz;
                        let fr = kr * ft;
                        fx += -ft * cs - fr * sn;
                        fy += ft * sn - fr * cs;
                    }
                }
                best = best.max((fx * fx + fy * fy).sqrt());
            }
        }
        g[deg as usize] = best;
    }
    g
}
