//! One instrument for both toolpaths (the baseline's and ours).
//!
//! Engagement: at every sample on the slab, the angle phi of the cutter's circle that
//! still touches material, measured against the stock as it stands just before that step
//! (prior program + the path so far). phi and the radial engagement ae = r (1 - cos phi)
//! are measured, never inferred from stepover.
//!
//! Load model (explicit assumptions, printed in the report):
//!   chip thickness  h   = fz * s(phi),  s = sin(phi) below 90 deg, else 1
//!   removal rate    MRR ~ ae * feed          (per unit axial depth)
//! Reference = the baseline's own nominal cut: its median engagement at its programmed
//! feed. A schedule is "within load" when h <= h_ref and MRR <= MRR_ref. Feed for our
//! path: the fastest feed inside both limits, capped at feed_cap x programmed. Time: that
//! feed, a curvature limit v <= sqrt(a R), and forward/backward acceleration passes.
//! With a CutModel the reference is replaced by physical limits (chip, power, deflection,
//! machine feed) and load is the worst ratio to them.

use std::f64::consts::PI;

use icam_post::{Move, MoveKind, PathPt, Tag};
use icam_stock::SweptStock;
use rayon::prelude::*;

use crate::cut::{CutModel, Limit};

/// Rapid traverse, mm/s (500 ipm).
pub const RAPID_MMPS: f64 = 500.0 * 25.4 / 60.0;

#[derive(Clone, Debug)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Engagement, rad (NaN off the slab).
    pub phi: f64,
    /// Scheduled feed, mm/s.
    pub feed: f64,
    /// The program's own feed, mm/s.
    pub programmed: f64,
    /// Enclosure: share of the circle at r + 0.5 mm that is material.
    pub enc: f64,
    pub tag: Tag,
}

impl Sample {
    fn new(x: f64, y: f64, z: f64, programmed: f64, tag: Tag) -> Sample {
        Sample { x, y, z, phi: f64::NAN, feed: 0.0, programmed, enc: f64::NAN, tag }
    }
}

/// The load model's reference and the machine.
pub struct LoadParams<'a> {
    pub r: f64,
    pub z_cut: f64,
    pub v_prog: f64,
    pub phi_ref: f64,
    pub feed_cap: f64,
    pub accel: f64,
    pub v_entry: f64,
    pub v_rapid: f64,
    /// Physical limits instead of the reference cut.
    pub cut: Option<&'a CutModel>,
    /// Local axial depth (else the cut model's).
    pub ap_at: Option<&'a dyn Fn(f64, f64) -> f64>,
}

impl LoadParams<'_> {
    pub fn s_ref(&self) -> f64 {
        LoadParams::s(self.phi_ref)
    }
    pub fn ae_ref(&self) -> f64 {
        self.r * (1.0 - self.phi_ref.cos())
    }
    pub fn s(phi: f64) -> f64 {
        if phi >= PI / 2.0 { 1.0 } else { phi.sin() }
    }
    pub fn ae(&self, phi: f64) -> f64 {
        self.r * (1.0 - phi.min(PI).cos())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub cut_length: f64,
    pub entry_length: f64,
    pub time_cut: f64,
    pub time_entry: f64,
    pub time_rapid: f64,
    pub rapid_length: f64,
    pub phi_max_deg: f64,
    pub phi_p95_deg: f64,
    pub phi_med_deg: f64,
    pub over_phi_cap_len: f64,
    /// max(h/h_ref, MRR/MRR_ref) at the feed actually used.
    pub load_max: f64,
    pub load_p95: f64,
    pub over_load_len: f64,
    /// With a CutModel: cut length by the limit that set the feed actually run
    /// (Kinematics = slowed below every ceiling by curvature or acceleration), in the
    /// order the limits first bind.
    pub bind_len: Vec<(Limit, f64)>,
    /// Length run below the chip floor, volume removed, peak power and deflection.
    pub rub_len: f64,
    pub volume: f64,
    pub power_max_w: f64,
    pub defl_max_mm: f64,
    /// Cut length past 0.75 enclosure (1 = a full slot), the median enclosure, and cut
    /// length whose local depth exceeds the stable depth at its engagement.
    pub enclosed_len: f64,
    pub enc_med: f64,
    pub unstable_len: f64,
    /// The chip where the tool touches stock (engagement over 1 deg), as a share of the
    /// target chip (the reference cut's at its feed and engagement, or the cut model's
    /// ceiling): its 5th percentile and median, the cut length it covers, and the part
    /// of that where it is 85-115 % of the target (thinner rubs, thicker overloads).
    pub chip_p5: f64,
    pub chip_med: f64,
    pub chip_len: f64,
    pub chip_band_len: f64,
}

impl Metrics {
    pub fn time(&self) -> f64 {
        self.time_cut + self.time_entry + self.time_rapid
    }
}

fn md(a: f64) -> f64 {
    let a = a % (2.0 * PI);
    if a < 0.0 { a + 2.0 * PI } else { a }
}

fn add(s: &mut Vec<Sample>, x: f64, y: f64, z: f64, f: f64, z_cut: f64) {
    s.push(Sample::new(x, y, z, f, if z > z_cut + 0.01 { Tag::Entry } else { Tag::Cut }));
}

/// Samples every <= `step` mm along moves `i0..i1` of a program (lines, arcs, plunges;
/// rapids at `v_rapid`, so a link is never integrated as a feed move). `feed_of_line`:
/// the programmed feed (mm/s) of each source line. `entry_after_ramp`: path length on the
/// slab, right after a ramped arrival, that still belongs to the entry (the entry loop's
/// flat lap at the cut).
#[allow(clippy::too_many_arguments)]
pub fn from_moves(moves: &[Move], i0: usize, i1: usize, feed_of_line: &[f64], z_cut: f64, step: f64, entry_after_ramp: f64, v_rapid: f64) -> Vec<Sample> {
    let mut s = Vec::new();
    for m in &moves[i0..i1] {
        if m.kind == MoveKind::Drill {
            continue;
        }
        let (d0, d1, d2) = (m.to.x - m.from.x, m.to.y - m.from.y, m.to.z - m.from.z);
        if m.kind == MoveKind::Rapid {
            let lr = (m.to - m.from).len();
            let nr = 1.max((lr / 2.0).ceil() as i64);
            for k in (if s.is_empty() { 0 } else { 1 })..=nr {
                let f = k as f64 / nr as f64;
                s.push(Sample::new(m.from.x + d0 * f, m.from.y + d1 * f, m.from.z + d2 * f, v_rapid, Tag::Rapid));
            }
            continue;
        }
        let f = feed_of_line[m.line.min(feed_of_line.len() - 1)];
        if m.kind == MoveKind::Feed {
            let l = (m.to - m.from).len();
            let n = 1.max((l / step).ceil() as i64);
            for k in (if s.is_empty() { 0 } else { 1 })..=n {
                let u = k as f64 / n as f64;
                add(&mut s, m.from.x + d0 * u, m.from.y + d1 * u, m.from.z + d2 * u, f, z_cut);
            }
        } else {
            let c = m.center;
            let r = ((m.from.x - c.x) * (m.from.x - c.x) + (m.from.y - c.y) * (m.from.y - c.y)).sqrt();
            let (t0, t1) = ((m.from.y - c.y).atan2(m.from.x - c.x), (m.to.y - c.y).atan2(m.to.x - c.x));
            let full = (m.to - m.from).len() < 1e-6;
            let d = if m.kind == MoveKind::ArcCcw {
                if full { 2.0 * PI } else { md(t1 - t0) }
            } else if full {
                -2.0 * PI
            } else {
                -md(t0 - t1)
            };
            let n = 2.max((d.abs() * r / step).ceil() as i64);
            for k in 1..=n {
                let a = t0 + d * k as f64 / n as f64;
                add(&mut s, c.x + r * a.cos(), c.y + r * a.sin(), m.from.z + (m.to.z - m.from.z) * k as f64 / n as f64, f, z_cut);
            }
        }
    }
    if entry_after_ramp <= 0.0 {
        return s;
    }
    let mut run = -1.0;
    for i in 1..s.len() {
        let (p, q) = (&s[i - 1], &s[i]);
        // Arriving on the slab along a ramp (not a vertical plunge).
        if p.z > z_cut + 0.01 && q.z <= z_cut + 0.01 && ((p.x - q.x).abs() + (p.y - q.y).abs()) > 1e-6 {
            run = 0.0;
        }
        if run < 0.0 || q.tag != Tag::Cut {
            continue;
        }
        run += ((q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y)).sqrt();
        if run <= entry_after_ramp {
            s[i].tag = Tag::Entry;
        } else {
            run = -1.0;
        }
    }
    s
}

/// Samples every <= `step` mm along a strategy's path, each carrying the tag of the
/// point it runs toward.
pub fn from_path(path: &[PathPt], step: f64) -> Vec<Sample> {
    let mut s = Vec::new();
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            let q = &path[i - 1];
            let l = ((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z)).sqrt();
            let n = (l / step).ceil() as i64;
            for k in 1..n {
                let f = k as f64 / n as f64;
                s.push(Sample::new(q.x + (p.x - q.x) * f, q.y + (p.y - q.y) * f, q.z + (p.z - q.z) * f, 0.0, p.tag));
            }
        }
        s.push(Sample::new(p.x, p.y, p.z, 0.0, p.tag));
    }
    s
}

/// Engagement along the samples; mutates `stock` (the path's sweeps are added to it).
///
/// Each sample is measured against the stock as it stood just before its own step. That
/// stock is the first n sweeps whatever order the samples are read in, so the sweeps are
/// filed first, in path order, noting n for every sample measured, and then all samples
/// are measured at once (`SweptStock::material_at`).
pub fn engagement(s: &mut [Sample], stock: &mut SweptStock, r: f64, z_cut: f64) {
    const N: usize = 180;
    let rs = r - 0.01;
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..N).map(|k| ((2.0 * PI * k as f64 / N as f64).cos(), (2.0 * PI * k as f64 / N as f64).sin())).unzip();
    let step = 2.0 * PI / N as f64;
    let mut before: Vec<Option<usize>> = vec![None; s.len()];
    for i in 0..s.len() {
        let on_slab = s[i].z <= z_cut + 0.01;
        let prev = (i > 0).then(|| (s[i - 1].x, s[i - 1].y, s[i - 1].z));
        let p = &mut s[i];
        match prev {
            Some((px, py, _)) if on_slab && p.tag == Tag::Entry => {
                // The entry loop's flat lap at the cut: it only removes the last ramp step (axial
                // depth <= one pitch), so the slab view would call it a full slot. Clear
                // it, but count it as entry, like a plunge.
                p.phi = f64::NAN;
                stock.add_segment(px, py, p.x, p.y, r);
            }
            Some((px, py, pz)) if on_slab && pz <= z_cut + 0.01 => {
                before[i] = Some(stock.count());
                stock.add_segment(px, py, p.x, p.y, r);
            }
            _ if on_slab => {
                p.phi = f64::NAN; // arriving on the slab: plunge/entry end
                stock.add_segment(p.x, p.y, p.x, p.y, r);
            }
            _ => {}
        }
    }
    let stock = &*stock;
    s.par_iter_mut().zip(before.par_iter()).for_each(|(p, &n)| {
        let Some(n) = n else { return };
        let mut hit = [false; N];
        let mut count = 0;
        for k in 0..N {
            hit[k] = stock.material_at(p.x + rs * cos[k], p.y + rs * sin[k], n);
            if hit[k] {
                count += 1;
            }
        }
        let mut phi = (count * 2) as f64 * PI / N as f64;
        // Refine each material/air transition by bisection (sub-degree phi).
        if count > 0 && count < N {
            phi = 0.0;
            for k in 0..N {
                let k1 = (k + 1) % N;
                if hit[k] == hit[k1] {
                    if hit[k] {
                        phi += step;
                    }
                    continue;
                }
                let mut a0 = 2.0 * PI * k as f64 / N as f64;
                let mut a1 = a0 + step;
                let m0 = hit[k];
                for _ in 0..5 {
                    let am = 0.5 * (a0 + a1);
                    if stock.material_at(p.x + rs * am.cos(), p.y + rs * am.sin(), n) == m0 {
                        a0 = am;
                    } else {
                        a1 = am;
                    }
                }
                let split = 0.5 * (a0 + a1) - 2.0 * PI * k as f64 / N as f64;
                phi += if m0 { split } else { step - split };
            }
        }
        p.phi = phi.min(PI);
        let (mut enc_n, re) = (0, r + 0.5);
        for k in (0..N).step_by(2) {
            if stock.material_at(p.x + re * cos[k], p.y + re * sin[k], n) {
                enc_n += 1;
            }
        }
        p.enc = enc_n as f64 / (N as f64 / 2.0);
    });
}

/// Feed schedule inside the load limits (`programmed` stays as it is). An entry turning
/// on radius rho cuts its bore's wall with the edge on radius rho + r, which runs
/// (rho + r) / rho times as fast as the center, so the entry feed drops by that much on
/// turns (the internal-circle feed of Heuwinkel, Cutting Tool Engineering 58(8), 2006).
pub fn schedule(s: &mut [Sample], lp: &LoadParams) {
    let cap = lp.v_prog * lp.feed_cap;
    let turns = entry_turns(s, lp.r);
    for (p, turn) in s.iter_mut().zip(turns) {
        if p.phi.is_nan() {
            p.feed = if p.tag == Tag::Rapid {
                lp.v_rapid
            } else if p.tag == Tag::Entry {
                lp.v_entry * turn
            } else if p.z > lp.z_cut + 0.01 || lp.cut.is_some() {
                lp.v_entry
            } else {
                lp.v_prog
            };
            continue;
        }
        if let Some(c) = lp.cut {
            p.feed = c.ceiling(p.phi).0;
            continue;
        }
        let (sphi, ae) = (LoadParams::s(p.phi), lp.ae(p.phi));
        let vh = if sphi < 1e-6 { cap } else { lp.v_prog * lp.s_ref() / sphi };
        let vm = if ae < 1e-6 { cap } else { lp.v_prog * lp.ae_ref() / ae };
        p.feed = cap.min(vh.min(vm));
    }
}

/// For every sample, rho / (rho + r) where it is an entry turning on radius rho, else 1.
/// rho is read over about a millimeter of the entry either way (samples lie along the
/// path's chords, so the turn shows only across several of them).
fn entry_turns(s: &[Sample], r: f64) -> Vec<f64> {
    const SPAN: f64 = 1.0;
    let xy = |i: usize, j: usize| (s[j].x - s[i].x).hypot(s[j].y - s[i].y);
    (0..s.len())
        .map(|i| {
            if s[i].tag != Tag::Entry {
                return 1.0;
            }
            let (mut a, mut c, mut back, mut ahead) = (i, i, 0.0, 0.0);
            while a > 0 && s[a - 1].tag == Tag::Entry && back < SPAN {
                back += xy(a - 1, a);
                a -= 1;
            }
            while c + 1 < s.len() && s[c + 1].tag == Tag::Entry && ahead < SPAN {
                ahead += xy(c, c + 1);
                c += 1;
            }
            entry_turn(&s[a], &s[i], &s[c], r)
        })
        .collect()
}

/// rho / (rho + r) for the turn through three samples in XY, rho its radius (the circle
/// through them; half the step at a reversal). 1 where they are straight or stand still.
fn entry_turn(a: &Sample, b: &Sample, c: &Sample, r: f64) -> f64 {
    let (ab, bc, ca) = ((b.x - a.x).hypot(b.y - a.y), (c.x - b.x).hypot(c.y - b.y), (a.x - c.x).hypot(a.y - c.y));
    if ab < 1e-9 || bc < 1e-9 {
        return 1.0;
    }
    let cross = ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
    if cross < 1e-12 * ab * bc && ca > ab.max(bc) {
        return 1.0;
    }
    let rho = (ab * bc * ca / (2.0 * cross).max(1e-300)).max(0.5 * ab.max(bc));
    rho / (rho + r)
}

/// Load ratio at a given feed: the worse of chip thickness and removal rate vs the reference.
pub fn load(p: &Sample, v: f64, lp: &LoadParams) -> f64 {
    if p.phi.is_nan() {
        return 0.0;
    }
    match lp.cut {
        Some(c) => c.load(p.phi, v),
        None => (v / lp.v_prog * LoadParams::s(p.phi) / lp.s_ref()).max(v / lp.v_prog * lp.ae(p.phi) / lp.ae_ref()),
    }
}

/// Kinematic time with the scheduled feeds (`use_programmed`: the program as posted).
pub fn measure(s: &[Sample], lp: &LoadParams, use_programmed: bool, rapid_length: f64, phi_cap: f64) -> Metrics {
    let mut m = Metrics { rapid_length, time_rapid: rapid_length / lp.v_rapid, ..Metrics::default() };
    let n = s.len();
    if n < 2 {
        return m;
    }
    let (mut v, mut ds) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        v[i] = if use_programmed { s[i].programmed } else { s[i].feed };
        if i > 0 {
            ds[i] = ((s[i].x - s[i - 1].x) * (s[i].x - s[i - 1].x) + (s[i].y - s[i - 1].y) * (s[i].y - s[i - 1].y) + (s[i].z - s[i - 1].z) * (s[i].z - s[i - 1].z)).sqrt();
        }
    }
    // Curvature limit from the turn over about +/-1 mm.
    const WIN: f64 = 1.0;
    for i in 1..n - 1 {
        let (mut a, mut b, mut la, mut lb) = (i, i, 0.0, 0.0);
        while a > 0 && la < WIN {
            la += ds[a];
            a -= 1;
        }
        while b < n - 1 && lb < WIN {
            b += 1;
            lb += ds[b];
        }
        let (ux, uy, wx, wy) = (s[i].x - s[a].x, s[i].y - s[a].y, s[b].x - s[i].x, s[b].y - s[i].y);
        let (nu, nw) = ((ux * ux + uy * uy).sqrt(), (wx * wx + wy * wy).sqrt());
        if nu < 1e-9 || nw < 1e-9 {
            continue;
        }
        let c = ((ux * wx + uy * wy) / (nu * nw)).clamp(-1.0, 1.0);
        let turn = c.acos();
        if turn < 1e-6 {
            continue;
        }
        let big_r = 0.5 * (nu + nw) / turn; // arc length / angle
        v[i] = v[i].min((lp.accel * big_r).sqrt());
    }
    v[0] = 0.0;
    v[n - 1] = 0.0;
    for i in 1..n {
        v[i] = v[i].min((v[i - 1] * v[i - 1] + 2.0 * lp.accel * ds[i]).sqrt());
    }
    for i in (0..n - 1).rev() {
        v[i] = v[i].min((v[i + 1] * v[i + 1] + 2.0 * lp.accel * ds[i + 1]).sqrt());
    }
    let (mut phis, mut loads, mut encs, mut chips) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 1..n {
        let vm = (0.5 * (v[i] + v[i - 1])).max(1e-3);
        let dt = ds[i] / vm;
        let cut = !s[i].phi.is_nan();
        if s[i].tag == Tag::Rapid {
            m.rapid_length += ds[i];
            m.time_rapid += dt;
            continue;
        }
        if cut {
            m.cut_length += ds[i];
            m.time_cut += dt;
        } else {
            m.entry_length += ds[i];
            m.time_entry += dt;
        }
        if !cut {
            continue;
        }
        let deg = s[i].phi * 180.0 / PI;
        phis.push(deg);
        // The chip, where the tool touches stock (a feed move through air makes none).
        if s[i].phi > PI / 180.0 {
            let chip = match lp.cut {
                Some(c) => c.fz(v[i]) * LoadParams::s(s[i].phi) / c.h_ceil,
                None => v[i] / lp.v_prog * LoadParams::s(s[i].phi) / lp.s_ref(),
            };
            chips.push(chip);
            m.chip_len += ds[i];
            if (0.85..=1.15).contains(&chip) {
                m.chip_band_len += ds[i];
            }
        }
        if deg > phi_cap {
            m.over_phi_cap_len += ds[i];
        }
        // Load at the feed actually run (after the kinematic limits).
        let ld = load(&s[i], v[i], lp);
        loads.push(ld);
        if ld > 1.10 {
            m.over_load_len += ds[i];
        }
        if let Some(cm) = lp.cut {
            let (vc, lim) = cm.ceiling(s[i].phi);
            let b = if v[i] < 0.97 * vc { Limit::Kinematics } else { lim };
            match m.bind_len.iter_mut().find(|e| e.0 == b) {
                Some(e) => e.1 += ds[i],
                None => m.bind_len.push((b, 0.0 + ds[i])),
            }
            if cm.rubs(s[i].phi, v[i]) {
                m.rub_len += ds[i];
            }
            m.volume += cm.ae(s[i].phi) * cm.ap * ds[i];
            m.power_max_w = m.power_max_w.max(cm.power_w(s[i].phi, v[i]));
            m.defl_max_mm = m.defl_max_mm.max(cm.deflection(s[i].phi, v[i]));
            let ap_here = lp.ap_at.map_or(cm.ap, |f| f(s[i].x, s[i].y));
            if ap_here > cm.stable_depth_at(s[i].phi) {
                m.unstable_len += ds[i];
            }
        }
        if !s[i].enc.is_nan() {
            encs.push(s[i].enc);
            if s[i].enc >= 0.75 {
                m.enclosed_len += ds[i];
            }
        }
    }
    if !phis.is_empty() {
        let by = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        phis.sort_by(by);
        loads.sort_by(by);
        m.phi_max_deg = phis[phis.len() - 1];
        m.phi_p95_deg = phis[(0.95 * (phis.len() - 1) as f64) as usize];
        m.phi_med_deg = phis[phis.len() / 2];
        m.load_max = loads[loads.len() - 1];
        m.load_p95 = loads[(0.95 * (loads.len() - 1) as f64) as usize];
        if !chips.is_empty() {
            chips.sort_by(by);
            m.chip_p5 = chips[(0.05 * (chips.len() - 1) as f64) as usize];
            m.chip_med = chips[chips.len() / 2];
        }
        if !encs.is_empty() {
            encs.sort_by(by);
            m.enc_med = encs[encs.len() / 2];
        }
    }
    m
}
