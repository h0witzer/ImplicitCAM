//! The entry: how the tool gets down to the cut, and the plan's source.
//!
//! E(x) is the distance from x to a run of the seed's ridge: the segment of length `run`
//! centered on the ridge's middle (the ridge is the top of the wall distance, where the
//! pocket has the most room). The tool ramps down E's level set {E = rho}, a closed loop
//! around the run (a stadium; a circle when the run is a point; the run itself, out and
//! back, when rho is 0), then goes once around it at the cut. While rho <= 0.9 r the tool
//! sweeps the loop's whole inside, so in tool-center space it clears {E <= rho}: the
//! plan's source, where T = 0. No wall comes within rho + `ROOM` of the run, so E is the
//! straight-line distance, a closed form: nothing is solved.
//!
//! rho and run come from what the entry is for: down to the cut inside the limits (the
//! tool's ramp angle, the depth a full-width cut stays stable at per lap, the feed the
//! machine holds along the loop under its acceleration) in the least time
//! ([`Descent::time`]). The time is continuous in rho, run and every input, and its least
//! is found by golden-section searches over continuous rho and run, so a small change in
//! the pocket moves the loop a little: there is no lattice, and no switch between loop
//! shapes (a sliver's back-and-forth ramp is the same loop at rho = 0).
//!
//! What keeps the loop from shrinking to a plunge is the chip at the bore wall. On a turn
//! of radius rho the edge cutting the wall runs on radius rho + r, so it moves (rho + r) /
//! rho times as fast as the tool's center, and the center's feed must drop by that much to
//! hold the edge's chip (Heuwinkel, "As easy as X, Y, Z", Cutting Tool Engineering 58(8),
//! 2006: the center's feed on an internal circle is (1 - Dc / Dw) n fz Z). A tight loop
//! therefore descends slowly, and the least time moves out toward the largest loop the
//! room allows (a bore up to twice the tool's diameter from solid, as the same article
//! gives). What a loop does to the cut is also measured ([`Bore`]): the share of every
//! layer only the end face cuts, and how crowded the chips' gap is. `RHO_MIN_D` x D stays
//! as the least radius where the room allows. (The tool's center plows a layer of its
//! dead zone x tan(ramp) on any loop: that limits the ramp angle, not the loop.)
//!
//! No part of a layer is left to the end face alone. Its axial chip is the drop per lap
//! over the tooth passes per lap, a few thousandths of a millimeter, where the edge
//! (rubbing) forces outweigh the shearing ones many times over (Engin and Altintas,
//! generalized milling mechanics, part I: Al-7075 averages Kte 19.65 N/mm against Ktc
//! 1319.41 N/mm2, so the edge force leads below a 0.015 mm chip). So the loop is a stadium
//! long enough that no point stays under the tool all lap (run >= 2 (r - rho)) where the
//! room allows; where it does not, the core is kept as small as the room allows (a steep
//! weight on it in the choice of run, so the choice stays continuous).
//!
//! The drop per lap is the stable depth of the cut the loop actually takes: full width
//! over the front half of the tool, but only as deep as the material the tool last left
//! there, the whole drop where it was last cut a lap ago and none where the tool has just
//! passed. On a circle that profile is the same all the way round ([`helix_depth`]); on a
//! stadium it changes along the lap (near the far end of a side the material ahead was
//! last cut almost a lap ago, a full slot), so the loop's drop is the worst of its
//! positions ([`loop_ap`]): the cut model's chatter limit weighted by each position's
//! profile (`CutModel::stable_depth_moments`).

use std::f64::consts::PI;
use std::sync::OnceLock;

use icam_motion::CutModel;
use rayon::prelude::*;

use crate::path::{dist, PathPt, PathResult, Tag, P};
use crate::plan::{Plan, Seed};

/// Clearance kept between the loop and the walls, mm.
pub const ROOM: f64 = 0.25;

/// The loop's least radius as a share of the tool's diameter, where the room allows.
pub const RHO_MIN_D: f64 = 0.05;

/// A loop leaving this share of each layer to the end face alone counts e times its time
/// when runs are weighed.
pub const FACE_ONLY_WEIGHT: f64 = 0.01;

/// Share of the drop per lap still standing just outside the edge of a tool of radius `r`
/// going round a circle of radius `rho`, at angle `b` from its direction of travel (-pi/2
/// to pi/2, positive toward the circle's center). The material there was last swept when
/// the tool last covered it: centers within r of the point lie within alpha of its own
/// angle about the circle's center, cos alpha = (rho - r sin b) / |p| for the point p, so
/// it was a lap less 2 alpha ago.
pub fn helix_depth(rho: f64, r: f64, b: f64) -> f64 {
    let (px, py) = (rho - r * b.sin(), r * b.cos());
    let len = px.hypot(py);
    if len < 1e-12 {
        return 0.0;
    }
    1.0 - (px / len).clamp(-1.0, 1.0).acos() / PI
}

/// Loop radii (x r) and runs (x r) the stable drop is tabulated at.
const RHOS: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const RUNS: [f64; 8] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

/// For a loop of radius `rho` around a run of length `run` (both x r, the tool's radius 1),
/// at positions along half the lap (the other half is the same turned round): the moments
/// (a, s, c) of the depth left just outside the leading half of the edge (see
/// [`CutModel::stable_depth_moments`]). The depth at a point is the share of a lap since
/// the tool last covered it, found by walking back along the loop.
fn loop_moments(rho: f64, run: f64) -> Vec<(f64, f64, f64)> {
    const LOOP: usize = 720;
    const AT: usize = 24;
    const PHI: usize = 90;
    let lap = 2.0 * run + 2.0 * PI * rho;
    if lap < 1e-9 {
        // Turning in place: the limit of a circle as it shrinks.
        let m = (0..PHI).fold((0.0, 0.0, 0.0), |m, i| {
            let f = PI * (i as f64 + 0.5) / PHI as f64;
            let w = helix_depth(0.0, 1.0, 0.5 * PI - f) * PI / PHI as f64;
            (m.0 + w, m.1 + w * (2.0 * f).sin(), m.2 + w * (2.0 * f).cos())
        });
        return vec![m];
    }
    // The loop counterclockwise, from the start of the side y = -rho: (point, heading).
    let at = |u: f64| -> ((f64, f64), f64) {
        let u = u.rem_euclid(lap);
        let half = 0.5 * run;
        if u < run {
            return ((-half + u, -rho), 0.0);
        }
        let u = u - run;
        if u < PI * rho {
            let a = -0.5 * PI + u / rho.max(1e-12);
            return ((half + rho * a.cos(), rho * a.sin()), a + 0.5 * PI);
        }
        let u = u - PI * rho;
        if u < run {
            return ((half - u, rho), PI);
        }
        let a = 0.5 * PI + (u - run) / rho.max(1e-12);
        ((-half + rho * a.cos(), rho * a.sin()), a + 0.5 * PI)
    };
    let pts: Vec<(f64, f64)> = (0..LOOP).map(|k| at(lap * k as f64 / LOOP as f64).0).collect();
    (0..AT)
        .map(|i| {
            let u = 0.5 * lap * (i as f64 + 0.5) / AT as f64;
            let (c, h) = at(u);
            let k0 = ((u / lap * LOOP as f64).floor() as usize) % LOOP;
            (0..PHI).fold((0.0, 0.0, 0.0), |m, j| {
                let f = PI * (j as f64 + 0.5) / PHI as f64;
                let b = 0.5 * PI - f; // from the heading, toward the loop's inside for b > 0
                let p = (c.0 + (1.0 + 1e-9) * (h + b).cos(), c.1 + (1.0 + 1e-9) * (h + b).sin());
                let mut back = 1.0;
                for step in 1..LOOP {
                    let q = pts[(k0 + LOOP - step) % LOOP];
                    if (p.0 - q.0).hypot(p.1 - q.1) <= 1.0 {
                        back = (u - lap * ((k0 + LOOP - step) % LOOP) as f64 / LOOP as f64).rem_euclid(lap) / lap;
                        break;
                    }
                }
                let w = back * PI / PHI as f64;
                (m.0 + w, m.1 + w * (2.0 * f).sin(), m.2 + w * (2.0 * f).cos())
            })
        })
        .collect()
}

/// [`loop_moments`] for every tabulated shape, once.
fn moments() -> &'static Vec<Vec<(f64, f64, f64)>> {
    static M: OnceLock<Vec<Vec<(f64, f64, f64)>>> = OnceLock::new();
    M.get_or_init(|| (0..RHOS.len() * RUNS.len()).into_par_iter().map(|k| loop_moments(RHOS[k / RUNS.len()], RUNS[k % RUNS.len()])).collect())
}

/// The stable drop per lap (mm) by loop shape: radius `RHOS` x r, run `RUNS` x r.
#[derive(Clone, Copy, Debug)]
pub struct LoopAp {
    t: [[f64; 8]; 11],
}

impl LoopAp {
    /// The same drop for every loop (no chatter model: infinite).
    pub fn uniform(ap: f64) -> LoopAp {
        LoopAp { t: [[ap; 8]; 11] }
    }

    /// The drop on a loop of radius `rho` around a run `run` for a tool of radius `r`,
    /// bilinear between the table's shapes (past the longest run, the longest's).
    pub fn at(&self, rho: f64, run: f64, r: f64) -> f64 {
        let x = (10.0 * rho / r.max(1e-12)).clamp(0.0, 10.0);
        let i = (x as usize).min(9);
        let fx = x - i as f64;
        let y = (run / r.max(1e-12)).clamp(0.0, RUNS[RUNS.len() - 1]);
        let j = (0..RUNS.len() - 1).find(|&j| y <= RUNS[j + 1]).unwrap_or(RUNS.len() - 2);
        let fy = (y - RUNS[j]) / (RUNS[j + 1] - RUNS[j]);
        let v = [self.t[i][j], self.t[i + 1][j], self.t[i][j + 1], self.t[i + 1][j + 1]];
        if v.iter().any(|a| a.is_infinite()) {
            return v.iter().copied().fold(f64::INFINITY, f64::min);
        }
        (v[0] * (1.0 - fx) + v[1] * fx) * (1.0 - fy) + (v[2] * (1.0 - fx) + v[3] * fx) * fy
    }
}

/// The stable drop per lap for every tabulated loop shape: the cut model's chatter limit
/// at the worst of the loop's positions (either way round).
pub fn loop_ap(c: &CutModel) -> LoopAp {
    let m = moments();
    let mut t = [[0.0; 8]; 11];
    for (k, at) in m.iter().enumerate() {
        t[k / RUNS.len()][k % RUNS.len()] = at.iter().map(|&(a, s, cc)| c.stable_depth_moments(a, s, cc).min(c.stable_depth_moments(a, -s, cc))).fold(f64::INFINITY, f64::min);
    }
    LoopAp { t }
}

/// What the descent works with: the drop, the tool's ramp limit (rise over run), the
/// stable drop per lap by loop shape ([`loop_ap`]), the entry feed (at the edge, where the
/// chip forms), the machine's acceleration and the tool's radius.
#[derive(Clone, Copy, Debug)]
pub struct Descent {
    pub dz: f64,
    pub tan_ramp: f64,
    pub ap: LoopAp,
    pub v: f64,
    pub accel: f64,
    pub r: f64,
}

impl Descent {

    /// Time for one lap of the loop of radius `rho` around a run of length `run`: each
    /// half circle at the slower of the center's feed that holds the edge's chip (v x rho
    /// / (rho + r)) and the feed the machine holds there (sqrt(accel x rho)), each side
    /// accelerating from that feed toward v and back (a triangle when the side is too
    /// short to reach v).
    pub fn lap_time(&self, rho: f64, run: f64) -> f64 {
        let a = self.accel;
        let vc = (self.v * rho / (rho + self.r).max(1e-12)).min((a * rho).sqrt());
        let vp = self.v.min((vc * vc + a * run).sqrt());
        let side = if vp > 0.0 { 2.0 * (vp - vc) / a + (run - (vp * vp - vc * vc) / a).max(0.0) / vp } else { 0.0 };
        let turns = if rho > 0.0 { 2.0 * PI * rho / vc } else { 0.0 };
        2.0 * side + turns
    }

    /// Ramp slope (rise over run) on a loop of radius `rho` around a run `run`: the tool's
    /// limit, or shallower so that one lap goes no deeper than the stable drop.
    pub fn tan(&self, rho: f64, run: f64) -> f64 {
        let lap = 2.0 * run + 2.0 * PI * rho;
        self.tan_ramp.min(self.ap.at(rho, run, self.r) / lap.max(1e-9))
    }

    /// The ramp slope on a seed's loop.
    pub fn tan_on(&self, e: &Seed) -> f64 {
        self.tan(e.rho, (e.bx - e.ax).hypot(e.by - e.ay))
    }

    /// Time to ramp down the loop, then once around it at the cut.
    pub fn time(&self, rho: f64, run: f64) -> f64 {
        let lap = 2.0 * run + 2.0 * PI * rho;
        if lap <= 1e-9 {
            return f64::INFINITY;
        }
        (self.dz / (lap * self.tan(rho, run)) + 1.0) * self.lap_time(rho, run)
    }
}

/// What a loop does to the cut: the bore it clears (every point within r of the loop) is
/// a stadium of radius rho + r around the run.
#[derive(Clone, Copy, Debug)]
pub struct Bore {
    /// Share of every layer that stays under the tool all lap, so only the end face cuts
    /// it: the points within r of the whole loop. The loop is the run grown by rho, so the
    /// farthest it reaches from x is rho past the farther end of the run, and the core is
    /// the lens of two disks of radius r - rho on the run's ends.
    pub face_only: f64,
    /// The bore's area over the gap around the tool that the chips leave by (infinite for
    /// a plunge).
    pub bore_to_gap: f64,
}

/// The [`Bore`] of a loop of radius `rho` around a run of length `run`, for a tool of
/// radius `r`.
pub fn bore(rho: f64, run: f64, r: f64) -> Bore {
    let big = rho + r;
    let area = 2.0 * big * run + PI * big * big;
    let s = (r - rho).max(0.0);
    let core = if run >= 2.0 * s { 0.0 } else { 2.0 * s * s * (run / (2.0 * s)).acos() - 0.5 * run * (4.0 * s * s - run * run).max(0.0).sqrt() };
    let gap = area - PI * r * r;
    Bore { face_only: core / area, bore_to_gap: if gap > 1e-12 { area / gap } else { f64::INFINITY } }
}

/// Length of a seed's loop.
pub fn lap(e: &Seed) -> f64 {
    2.0 * (e.bx - e.ax).hypot(e.by - e.ay) + 2.0 * PI * e.rho
}

/// The least of `f` over [lo, hi] by golden-section search: the argument and the value.
fn golden(lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> (f64, f64) {
    if hi - lo <= 1e-9 {
        return (lo, f(lo));
    }
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut p, mut q) = (lo, hi);
    let (mut c, mut d) = (q - g * (q - p), p + g * (q - p));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..80 {
        if q - p <= 1e-9 {
            break;
        }
        if fc <= fd {
            (q, d, fd) = (d, c, fc);
            c = q - g * (q - p);
            fc = f(c);
        } else {
            (p, c, fc) = (c, d, fd);
            d = p + g * (q - p);
            fd = f(d);
        }
    }
    [(lo, f(lo)), (hi, f(hi)), (c, fc), (d, fd)].into_iter().fold((c, f64::INFINITY), |b, x| if x.1 < b.1 { x } else { b })
}

/// Runs whose descent times are within about this share of the least are taken as
/// equally good: the run is their blend (see [`least`]).
pub const BLEND: f64 = 0.02;

/// The loop for a ridge of length `ridge`: (run, rho, time). `cap(run)` is the largest rho
/// the room allows around a run (it may only shrink as the run grows), `floor` the least
/// rho wanted where the room allows it. For a given run the time is least at one rho,
/// found by golden section above both the floor and r - run / 2 (no face-only core); a
/// run whose room cannot empty the core is weighed by its time x exp(core share /
/// `FACE_ONLY_WEIGHT`). Over runs it can have two near-equal minima (a circle at the
/// room's limit, and a stadium whose straight sides dilute its slow turns), so the least
/// run alone would jump between them as the inputs move. The run is instead the blend of
/// runs across the ridge, each weighted exp(-(T - T_least) / (BLEND x T_least)):
/// continuous in every input, and near the least wherever one minimum stands out.
pub fn least(d: &Descent, ridge: f64, cap: impl Fn(f64) -> f64, floor: f64) -> (f64, f64, f64) {
    // No point of a layer stays under the tool all lap: rho >= r - run / 2, as far as the
    // room allows.
    let best_rho = |run: f64| {
        let c = cap(run).max(0.0);
        golden(floor.min(c).max(d.r - 0.5 * run).min(c), c, |rho| d.time(rho, run))
    };
    const N: usize = 128;
    let times: Vec<(f64, f64)> = (0..=N)
        .map(|k| ridge * k as f64 / N as f64)
        .map(|run| {
            let (rho, t) = best_rho(run);
            (run, t * (bore(rho, run, d.r).face_only / FACE_ONLY_WEIGHT).exp())
        })
        .collect();
    let t_least = times.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
    let run = if t_least.is_finite() {
        let (w, wr) = times.iter().fold((0.0, 0.0), |(w, wr), &(run, t)| {
            let k = if t.is_finite() { (-(t - t_least) / (BLEND * t_least)).exp() } else { 0.0 };
            (w + k, wr + k * run)
        });
        wr / w
    } else {
        0.0
    };
    let (rho, t) = best_rho(run);
    (run, rho, t)
}

/// The entry loop for each of the plan's seeds, as seeds (the run as A-B, rho), for a
/// tool of radius `r`. The notes say what each came to.
pub fn tracks(plan: &Plan, r: f64, d: &Descent, notes: &mut Vec<String>) -> Vec<Seed> {
    plan.seeds.iter().map(|e| track(plan, e, r, d, notes)).collect()
}

fn track(plan: &Plan, e: &Seed, r: f64, d: &Descent, notes: &mut Vec<String>) -> Seed {
    let (mx, my) = (e.mid_x(), e.mid_y());
    let ridge = (e.bx - e.ax).hypot(e.by - e.ay);
    let (ux, uy) = if ridge > 1e-9 { ((e.bx - e.ax) / ridge, (e.by - e.ay) / ridge) } else { (1.0, 0.0) };
    // The least wall distance along the run: samples every half cell out from the middle
    // and the run's own ends, so it moves continuously with the run.
    let room = |run: f64| {
        let half = 0.5 * run;
        let at = |t: f64| plan.wall_dist(mx + ux * t, my + uy * t).min(plan.wall_dist(mx - ux * t, my - uy * t));
        let mut m = at(half);
        let mut t = 0.0;
        while t < half {
            m = m.min(at(t));
            t += 0.5 * plan.h;
        }
        m
    };
    let (run, rho, t) = least(d, ridge, |run| (0.9 * r).min(room(run) - ROOM), RHO_MIN_D * 2.0 * r);
    let lap = 2.0 * run + 2.0 * PI * rho;
    let tan = d.tan(rho, run);
    let b = bore(rho, run, r);
    notes.push(if lap > 1e-9 {
        format!(
            "entry: loop radius {rho:.2} mm around a {run:.1} mm run, ramp {:.1} deg, {:.1} laps down {:.1} mm, {t:.1} s; {:.0} % of each layer cut by the end face alone, bore {:.1} x the chips' gap",
            tan.atan().to_degrees(),
            d.dz / (lap * tan),
            d.dz,
            100.0 * b.face_only,
            b.bore_to_gap
        )
    } else {
        format!("entry: no room to ramp, straight down {:.1} mm (a plunge: the end face cuts it all, no gap for the chips)", d.dz)
    });
    let half = 0.5 * run;
    Seed { ax: mx - ux * half, ay: my - uy * half, bx: mx + ux * half, by: my + uy * half, rho, height: e.height, persistence: e.persistence }
}

/// Where the descent on a seed's loop starts: the tip of the loop behind A (on the
/// circle's +x side when the run is a point).
pub fn track_start(e: &Seed) -> P {
    let len = (e.bx - e.ax).hypot(e.by - e.ay);
    let (ux, uy) = if len > 1e-9 { ((e.bx - e.ax) / len, (e.by - e.ay) / len) } else { (-1.0, 0.0) };
    (e.ax - e.rho * ux, e.ay - e.rho * uy)
}

/// The entry as a descent on the seed's loop, counterclockwise from `track_start`,
/// ramped down at `tan_ramp` (rise over run) from `z_entry` to the cut, then once around
/// at the cut; straight down when the loop has no length.
pub fn track_entry(res: &mut PathResult, e: &Seed, z_cut: f64, z_entry: f64, tan_ramp: f64, ds: f64) {
    let len = (e.bx - e.ax).hypot(e.by - e.ay);
    let (ux, uy) = if len > 1e-9 { ((e.bx - e.ax) / len, (e.by - e.ay) / len) } else { (-1.0, 0.0) };
    // The loop, counterclockwise from the tip behind A: around A's cap to its right side,
    // along to B, around B's cap, back along the left side, around A's cap home.
    let base = uy.atan2(ux);
    let cap = |c: (f64, f64), a0: f64, a1: f64, out: &mut Vec<P>| {
        let n = 2.max(((a1 - a0) * e.rho / ds).ceil() as usize);
        for k in 0..n {
            let a = base + a0 + (a1 - a0) * k as f64 / n as f64;
            out.push((c.0 + e.rho * a.cos(), c.1 + e.rho * a.sin()));
        }
    };
    let side = |p: P, q: P, out: &mut Vec<P>| {
        let n = ((dist(p, q) / ds).ceil() as usize).max(1);
        for k in 0..n {
            let u = k as f64 / n as f64;
            out.push((p.0 + (q.0 - p.0) * u, p.1 + (q.1 - p.1) * u));
        }
    };
    let (a, b) = ((e.ax, e.ay), (e.bx, e.by));
    let mut track: Vec<P> = Vec::new();
    if len > 1e-9 {
        let (nx, ny) = (-uy, ux);
        cap(a, PI, 1.5 * PI, &mut track);
        side((a.0 - e.rho * nx, a.1 - e.rho * ny), (b.0 - e.rho * nx, b.1 - e.rho * ny), &mut track);
        cap(b, -0.5 * PI, 0.5 * PI, &mut track);
        side((b.0 + e.rho * nx, b.1 + e.rho * ny), (a.0 + e.rho * nx, a.1 + e.rho * ny), &mut track);
        cap(a, 0.5 * PI, PI, &mut track);
    } else {
        cap(a, PI, 3.0 * PI, &mut track);
    }
    let start = track[0];
    if let Some(&last) = res.points.last() {
        res.points.push(PathPt::new(last.x, last.y, z_entry, Tag::Rapid));
        res.points.push(PathPt::new(start.0, start.1, z_entry, Tag::Rapid));
        res.links += 1;
    }
    res.points.push(PathPt::new(start.0, start.1, z_entry, Tag::Entry));
    let perimeter: f64 = (0..track.len()).map(|k| dist(track[k], track[(k + 1) % track.len()])).sum();
    if perimeter <= 1e-9 {
        res.points.push(PathPt::new(start.0, start.1, z_cut, Tag::Entry));
        return;
    }
    // Down at the ramp, then one lap at the cut.
    let run = (z_entry - z_cut) / tan_ramp.max(1e-9);
    let (mut s, mut at, mut flat) = (0.0, start, 0.0);
    let mut k = 1;
    while flat < perimeter {
        let p = track[k % track.len()];
        let step = dist(at, p);
        s += step;
        if s >= run {
            flat += step;
        }
        res.points.push(PathPt::new(p.0, p.1, (z_entry - s * tan_ramp).max(z_cut), Tag::Entry));
        at = p;
        k += 1;
    }
}
