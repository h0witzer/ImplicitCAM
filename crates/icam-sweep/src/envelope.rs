//! A move's own surface: what the tool leaves behind, for any motion in 3 space.
//!
//! The boundary of everything a tool passes through is made of: at the start, the part
//! of the tool facing away from the motion; at the end, the part facing into it; and in
//! between, the curves where the tool's surface grazes its own motion (its normal
//! perpendicular to its velocity), swept over the move. The tool is a solid of
//! revolution: the point of a profile element at u, turned phi about the axis, sits at
//! q = (rho cos phi, rho sin phi, h) with normal (n_rho cos phi, n_rho sin phi, n_h) in
//! the tool frame. With the frame's velocity V and angular velocity W (both in the tool
//! frame), the point moves at V + W x q, and the grazing condition n . (V + W x q) = 0
//! reads
//!     A cos phi + B sin phi + C = 0,
//!     A = n_rho V_x - k W_y,  B = n_rho V_y + k W_x,  C = n_h V_z,  k = rho n_h - h n_rho,
//! a spin about the axis dropping out. Its two solutions, psi +- beta with
//! psi = atan2(B, A) and beta = acos(-C / |(A, B)|), are the two sheets of the
//! envelope; the arc between them facing the motion leads, the other trails. A ring that
//! never grazes (|C| > |(A, B)|) leads or trails whole; the sheets end where grazing
//! ends, found by bisection on |(A, B)|^2 - C^2 across each cell of a sheet, and meet
//! there (beta = 0 or pi). A flat face sliding in its own plane grazes everywhere; its
//! sheet is the diameter across the motion, which with the two end faces covers what it
//! sweeps.
//!
//! A tool sliding straight along its own axis (a plunge, a drill, a retract) grazes
//! everywhere on its side, so the rule above has nothing to choose; what it sweeps is
//! the tool stretched by the move, drawn as that solid of revolution.
//!
//! The pieces come out with chords within the tolerance asked; where the envelope folds
//! through itself (a tight turn), the caller trims it to the move's own boundary with the
//! move's field.

use std::f64::consts::{FRAC_PI_2, PI};

use crate::geom::{cross, dot, scale, Pose, Rot, V3};
use crate::motion::Motion;
use crate::skin::Patch;
use crate::sweep::Sweep;
use crate::tool::{Elem, ToolSolid};

/// The tool frame at one time: its pose, and its velocity and angular velocity in the
/// tool frame (by central differences).
struct Frame {
    pose: Pose,
    v: V3,
    w: V3,
}

fn frame(m: &dyn Motion, t: f64) -> Frame {
    let h = 1e-6;
    let (t0, t1) = ((t - h).max(0.0), (t + h).min(1.0));
    let (p0, p1) = (m.pose(t0), m.pose(t1));
    let dt = t1 - t0;
    let pose = m.pose(t);
    let inv = pose.rot.inverse();
    let mut d = p1.rot * p0.rot.inverse();
    if d.w < 0.0 {
        d = Rot { w: -d.w, x: -d.x, y: -d.y, z: -d.z };
    }
    let spin = V3::new(d.x, d.y, d.z);
    let w = if spin.len() < 1e-300 { V3::default() } else { scale(spin, 2.0 * spin.len().atan2(d.w) / (spin.len() * dt)) };
    Frame { pose, v: inv.apply(scale(p1.at - p0.at, 1.0 / dt)), w: inv.apply(w) }
}

/// One ring of the tool at one time: its profile point and normal, and how its points
/// face the motion: n . v = a cos phi + b sin phi + c (for a face sliding in its own
/// plane, toward its velocity instead: cos(phi - psi)).
#[derive(Clone, Copy)]
struct Ring {
    q: (f64, f64),
    n: (f64, f64),
    a: f64,
    b: f64,
    c: f64,
    flat: bool,
    psi: f64,
    size: f64,
}

impl Ring {
    /// Positive where the point at `phi` leads, negative where it trails.
    fn lead(&self, phi: f64) -> f64 {
        if self.flat {
            (phi - self.psi).cos()
        } else {
            self.a * phi.cos() + self.b * phi.sin() + self.c
        }
    }

    /// The grazing angles psi -+ beta, and a value non-negative where they exist.
    fn grazing(&self) -> (f64, f64, f64) {
        if self.flat {
            return (self.psi, FRAC_PI_2, 1.0);
        }
        let r = self.a.hypot(self.b);
        (self.b.atan2(self.a), (-self.c / r.max(1e-300)).clamp(-1.0, 1.0).acos(), (r * r - self.c * self.c) / (self.size * self.size))
    }
}

fn ring(f: &Frame, e: &Elem, u: f64, reach: f64) -> Ring {
    let (q, n) = e.at(u);
    let k = q.0 * n.1 - q.1 * n.0;
    let a = n.0 * f.v.x - k * f.w.y;
    let b = n.0 * f.v.y + k * f.w.x;
    let c = n.1 * f.v.z;
    let size = f.v.len() + f.w.len() * reach + 1e-300;
    let flat = a.hypot(b) < 1e-9 * size && c.abs() < 1e-9 * size;
    Ring { q, n, a, b, c, flat, psi: f.v.y.atan2(f.v.x), size }
}

/// The ring's point and normal at angle `phi`, placed by the frame.
fn place(f: &Frame, r: &Ring, phi: f64) -> (V3, V3) {
    let (s, c) = phi.sin_cos();
    (f.pose.apply(V3::new(r.q.0 * c, r.q.0 * s, r.q.1)), f.pose.rot.apply(V3::new(r.n.0 * c, r.n.0 * s, r.n.1)))
}

/// The angle between two directions, radians in [0, pi].
fn gap(a: f64, b: f64) -> f64 {
    (a - b).sin().atan2((a - b).cos()).abs()
}

/// A ring's grazing angle on the sheet `sign` labels in a cell whose direction toward the
/// motion is `psi_ref`: psi + sign beta, the label swapped where psi has turned half way
/// round (the ring's velocity pattern reversed, as where the motion's instantaneous axis
/// crosses the tool; the two grazing points trade places there). And whether it grazes.
fn on_sheet(r: &Ring, sign: f64, psi_ref: f64) -> (f64, f64) {
    let (psi, beta, grazes) = r.grazing();
    let s = if gap(psi, psi_ref) > FRAC_PI_2 { -sign } else { sign };
    (psi + s * beta, grazes)
}

/// A cell's corner: its parameters, its point and normal, and its value.
type Corner = ((f64, f64), (V3, V3), f64);

/// Cuts a cell to where its value is non-negative (the crossings on its sides by
/// bisection with `at`, which gives the point and value at any parameters) and pushes the
/// pieces, wound to face their normals.
fn clip(c: [Corner; 4], at: &dyn Fn(f64, f64) -> ((V3, V3), f64), out: &mut Vec<Patch>) {
    let mut poly: Vec<(V3, V3)> = Vec::with_capacity(8);
    for k in 0..4 {
        let (a, b) = (c[k], c[(k + 1) % 4]);
        if a.2 >= 0.0 {
            poly.push(a.1);
        }
        if (a.2 >= 0.0) != (b.2 >= 0.0) {
            let along = |x: f64| (a.0 .0 + (b.0 .0 - a.0 .0) * x, a.0 .1 + (b.0 .1 - a.0 .1) * x);
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                let (x, y) = along(mid);
                if (at(x, y).1 >= 0.0) == (a.2 >= 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let (x, y) = along(if a.2 >= 0.0 { lo } else { hi });
            poly.push(at(x, y).0);
        }
    }
    for k in 1..poly.len().saturating_sub(1) {
        if let Some(piece) = Patch::facing(poly[0], poly[k], poly[k + 1]) {
            out.push(piece);
        }
    }
}

/// One cell of a sheet: the motion, the profile element, the sheet's label and the
/// direction toward the motion its labels are read against.
#[derive(Clone, Copy)]
struct Cell<'a> {
    m: &'a dyn Motion,
    e: &'a Elem,
    reach: f64,
    tol: f64,
    sign: f64,
    psi_ref: f64,
}

impl Cell<'_> {
    /// The sheet's grazing point at (t, u), whether the ring grazes there, and its angle.
    fn at(&self, t: f64, u: f64) -> ((V3, V3), f64, f64) {
        let f = frame(self.m, t);
        let r = ring(&f, self.e, u, self.reach);
        let (phi, grazes) = on_sheet(&r, self.sign, self.psi_ref);
        (place(&f, &r, phi), grazes, phi)
    }

    /// The sheet over [t0, t1] x [u0, u1], split while the grazing point swings more than
    /// a chord of its ring allows.
    fn sheet(&self, (t0, t1): (f64, f64), (u0, u1): (f64, f64), depth: usize, out: &mut Vec<Patch>) {
        let (c00, c01, c11, c10) = (self.at(t0, u0), self.at(t0, u1), self.at(t1, u1), self.at(t1, u0));
        let rho = |u: f64| self.e.at(u).0 .0;
        let step = (8.0 * self.tol / rho(u0).max(rho(u1)).max(1e-9)).sqrt().min(0.3);
        let along_u = gap(c00.2, c01.2).max(gap(c10.2, c11.2));
        let along_t = gap(c00.2, c10.2).max(gap(c01.2, c11.2));
        if depth < 10 && along_u.max(along_t) > step {
            if along_u >= along_t {
                let um = 0.5 * (u0 + u1);
                self.sheet((t0, t1), (u0, um), depth + 1, out);
                self.sheet((t0, t1), (um, u1), depth + 1, out);
            } else {
                let tm = 0.5 * (t0 + t1);
                self.sheet((t0, tm), (u0, u1), depth + 1, out);
                self.sheet((tm, t1), (u0, u1), depth + 1, out);
            }
            return;
        }
        let corners = [((t0, u0), c00.0, c00.1), ((t0, u1), c01.0, c01.1), ((t1, u1), c11.0, c11.1), ((t1, u0), c10.0, c10.1)];
        let at = |t: f64, u: f64| {
            let (p, g, _) = self.at(t, u);
            (p, g)
        };
        clip(corners, &at, out);
    }
}

/// Steps in t so that no point of the tool strays more than `tol` from straight chords
/// between steps (by the motion's second-derivative bounds, or its rates without them),
/// with half of `tol` to spare: a grazing point slides over the tool as it moves, so its
/// path bends a little more than any one tool point's (on an arc it rides the outer wall,
/// a tool radius farther out).
fn t_steps(m: &dyn Motion, reach: f64, tol: f64) -> usize {
    let mut n = 1usize;
    loop {
        let worst = (0..n)
            .map(|i| {
                let (a, b) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
                let dt = b - a;
                let (c2, a2) = m.accel(a, b);
                let (v, w) = m.rate(a, b);
                if c2.is_finite() && a2.is_finite() {
                    dt * dt / 8.0 * (c2 + (a2 + w * w) * reach)
                } else {
                    0.5 * (v + w * reach) * dt
                }
            })
            .fold(0.0, f64::max);
        if worst <= 0.5 * tol || n >= 4096 {
            return n;
        }
        n *= 2;
    }
}

/// Pieces of a grid of points (rows by columns), each wound to face its normals.
fn grid(pts: &[Vec<(V3, V3)>], out: &mut Vec<Patch>) {
    for i in 0..pts.len().saturating_sub(1) {
        for j in 0..pts[i].len().saturating_sub(1) {
            let (a, b, c, d) = (pts[i][j], pts[i][j + 1], pts[i + 1][j], pts[i + 1][j + 1]);
            for (x, y, z) in [(a, b, d), (a, d, c)] {
                if let Some(p) = Patch::facing(x, y, z) {
                    out.push(p);
                }
            }
        }
    }
}

/// The whole surface of a solid of revolution placed at `pose`, chords within `tol` and
/// pieces no longer than about `longest`.
fn revolved(tool: &ToolSolid, pose: Pose, tol: f64, longest: f64) -> Vec<Patch> {
    let mut out = Vec::new();
    let f = Frame { pose, v: V3::default(), w: V3::default() };
    for e in tool.elements() {
        if matches!(e, Elem::Corner { .. }) {
            continue;
        }
        let nu = e.steps(tol, longest);
        let widest = (0..=nu).map(|j| e.at(j as f64 / nu as f64).0 .0).fold(0.0, f64::max);
        let nphi = phi_steps(widest, tol, longest);
        let pts: Vec<Vec<(V3, V3)>> = (0..=nu)
            .map(|j| {
                let r = ring(&f, &e, j as f64 / nu as f64, tool.reach());
                (0..=nphi).map(|k| place(&f, &r, 2.0 * PI * k as f64 / nphi as f64)).collect()
            })
            .collect();
        grid(&pts, &mut out);
    }
    out
}

/// Steps round a ring of radius `rho` for chords within `tol` and pieces no longer than
/// `longest`.
fn phi_steps(rho: f64, tol: f64, longest: f64) -> usize {
    if rho <= 1e-9 {
        return 8;
    }
    let by_chord = 2.0 * PI / (8.0 * tol / rho).sqrt().min(0.3);
    let by_length = 2.0 * PI * rho / longest;
    by_chord.max(by_length).ceil().clamp(8.0, 1024.0) as usize
}

/// The move's surface (untrimmed): chords within `tol`, pieces no longer than about
/// `longest` (so a piece that lies on another surface is seen whole, its corners exact).
pub fn envelope(s: &Sweep, tol: f64, longest: f64) -> Vec<Patch> {
    let m = s.motion.as_ref();
    let tool = &s.tool;
    if m.straight() {
        let (p0, p1) = (m.pose(0.0), m.pose(1.0));
        let (d, a) = (p1.at - p0.at, p0.axis());
        if cross(d, a).len() <= 1e-9 * (1.0 + d.len()) {
            // Sliding along its own axis: the tool stretched by the move, from its lower end.
            let lower = if dot(d, a) >= 0.0 { p0 } else { p1 };
            return revolved(&ToolSolid::new(tool.profile.clone(), tool.length + d.len()), lower, tol, longest);
        }
    }
    let reach = tool.reach();
    let nt = t_steps(m, reach, tol).max((m.rate(0.0, 1.0).0 / longest).ceil() as usize).min(1 << 16);
    let frames: Vec<Frame> = (0..=nt).map(|i| frame(m, i as f64 / nt as f64)).collect();
    let mut out = Vec::new();
    for e in tool.elements() {
        let nu = e.steps(tol, longest);
        let rings: Vec<Vec<Ring>> = frames.iter().map(|f| (0..=nu).map(|j| ring(f, &e, j as f64 / nu as f64, reach)).collect()).collect();
        // The two sheets: the grazing points swept over the move, cell by cell, each cut
        // to where the ring grazes, the labels kept straight by `on_sheet`. A cell splits
        // while its grazing point swings too far round its ring for a chord (fast near
        // where grazing begins).
        for sign in [1.0, -1.0] {
            for i in 0..nt {
                for j in 0..nu {
                    // The direction toward the motion from the corner where it is clearest
                    // (at a ring of radius 0 on a flat face it is not defined at all).
                    let clear = [&rings[i][j], &rings[i][j + 1], &rings[i + 1][j], &rings[i + 1][j + 1]].into_iter().fold(&rings[i][j], |a, b| if b.a.hypot(b.b) > a.a.hypot(a.b) { b } else { a });
                    let cell = Cell { m, e: &e, reach, tol, sign, psi_ref: clear.grazing().0 };
                    cell.sheet((i as f64 / nt as f64, (i + 1) as f64 / nt as f64), (j as f64 / nu as f64, (j + 1) as f64 / nu as f64), 0, &mut out);
                }
            }
        }
        // The caps: at the start the trailing part of the tool, at the end the leading
        // part, each cell of the (u, phi) grid cut where the point turns from one to the
        // other (the grazing curve). A corner's rings are all one circle, so its share of
        // a cap is an arc with no area.
        if matches!(e, Elem::Corner { .. }) {
            continue;
        }
        let widest = (0..=nu).map(|j| e.at(j as f64 / nu as f64).0 .0).fold(0.0, f64::max);
        let nphi = phi_steps(widest, tol, longest);
        for (f, rs, side) in [(&frames[0], &rings[0], -1.0), (&frames[nt], &rings[nt], 1.0)] {
            for j in 0..nu {
                for k in 0..nphi {
                    let (u0, u1) = (j as f64 / nu as f64, (j + 1) as f64 / nu as f64);
                    let (p0, p1) = (2.0 * PI * k as f64 / nphi as f64, 2.0 * PI * (k + 1) as f64 / nphi as f64);
                    let corner = |a: usize, b: usize| {
                        let (u, phi) = (if a == 0 { u0 } else { u1 }, if b == 0 { p0 } else { p1 });
                        let r = &rs[j + a];
                        ((u, phi), place(f, r, phi), side * r.lead(phi))
                    };
                    let at = |u: f64, phi: f64| {
                        let r = ring(f, &e, u, reach);
                        (place(f, &r, phi), side * r.lead(phi))
                    };
                    clip([corner(0, 0), corner(0, 1), corner(1, 1), corner(1, 0)], &at, &mut out);
                }
            }
        }
    }
    out
}

/// The volume a closed, outward-facing set of pieces encloses (the divergence theorem).
pub fn enclosed(patches: &[Patch]) -> f64 {
    patches.iter().map(|p| dot(p.p[0], cross(p.p[1], p.p[2]))).sum::<f64>() / 6.0
}
