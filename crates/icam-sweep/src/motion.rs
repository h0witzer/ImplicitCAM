//! Where the tool is during a move, whatever machine puts it there.
//!
//! A [`Motion`] is the tool frame's pose in the work frame for t in [0, 1], with bounds,
//! over any stretch of t, on how fast that frame's origin moves (v, mm per unit t) and
//! how fast the tool's axis direction turns (w, radians per unit t). The tool is a solid
//! of revolution, so spinning about its own axis moves none of it; a point fixed in the
//! work frame moves relative to the tool at most v + w d per unit t, d its distance from
//! the tool origin: that is all a field needs to find the least distance over a move
//! with a certificate. Bounds on the second derivatives (the origin's acceleration and
//! the axis direction's) let it bound the bend of every tool point's path as well, which
//! makes the certificate tighten with the square of the stretch instead of linearly.
//!
//! [`Linear`] is a move straight in pose space (the origin along a line, the frame
//! turning about one fixed axis at a steady rate), as a controller runs with tool center
//! point control. [`Helix`] is an arc or helix in any plane with the frame held. [`Joint`]
//! runs a [`Machine`]'s axes linearly from one set of values to another, as a controller
//! runs without tool center point control; [`Chain`] is any serial machine.

use std::sync::Arc;

use crate::geom::{cross, dot, scale, unit, Pose, Rot, V3};

/// `Any` lets a backend that cannot call back into Rust (a GPU kernel) recognize the
/// motions it knows how to place and leave the rest to the CPU.
pub trait Motion: std::any::Any + Send + Sync + std::fmt::Debug {
    /// The tool frame's pose at t in [0, 1].
    fn pose(&self, t: f64) -> Pose;
    /// Over [t0, t1]: (v, w), bounds on the tool origin's speed and the turning rate of
    /// the tool axis's direction, per unit t.
    fn rate(&self, t0: f64, t1: f64) -> (f64, f64);
    /// Over [t0, t1]: (a, b), bounds on the tool origin's acceleration |c''| and on the
    /// axis direction's second derivative |a''|, per unit t squared. Infinite (the
    /// default) when the motion cannot say.
    fn accel(&self, _t0: f64, _t1: f64) -> (f64, f64) {
        (f64::INFINITY, f64::INFINITY)
    }
    /// Over [t0, t1]: a bound on how far the tool origin strays from the straight chord
    /// between its positions at t0 and t1 (at the same fraction of the way). Any path
    /// whose speed stays under v strays at most v (t1 - t0) / 2.
    fn stray(&self, t0: f64, t1: f64) -> f64 {
        0.5 * self.rate(t0, t1).0 * (t1 - t0)
    }
    /// True when the frame only translates, along a straight line at a steady rate (the
    /// swept tool is then the hull of the tool at the two ends).
    fn straight(&self) -> bool {
        false
    }
    /// The motion from a to b (0 <= a < b <= 1) as a motion of its own over [0, 1]: its
    /// pose at t is this one's at a + (b - a) t. None (the default) when the motion cannot
    /// say. A line's piece is a line and a helix's a helix, so the GPU's kernels place them.
    fn piece(&self, _a: f64, _b: f64) -> Option<Arc<dyn Motion>> {
        None
    }
}

/// Straight in pose space: the origin along the line from `from` to `to`, the frame
/// turning about one fixed axis at a steady rate.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub from: Pose,
    pub to: Pose,
}

impl Motion for Linear {
    fn pose(&self, t: f64) -> Pose {
        Pose { rot: self.from.rot.slerp(self.to.rot, t), at: self.from.at + scale(self.to.at - self.from.at, t) }
    }

    /// Slerp and lerp both run partway as themselves.
    fn piece(&self, a: f64, b: f64) -> Option<Arc<dyn Motion>> {
        Some(Arc::new(Linear { from: self.pose(a), to: self.pose(b) }))
    }

    /// The frame turns at a steady rate about one axis fixed in the work frame, so the
    /// tool axis turns at that rate times the sine of its (fixed) angle to that axis.
    fn rate(&self, _: f64, _: f64) -> (f64, f64) {
        let v = (self.to.at - self.from.at).len();
        let mut d = self.from.rot.inverse() * self.to.rot;
        if d.w < 0.0 {
            d = Rot { w: -d.w, x: -d.x, y: -d.y, z: -d.z };
        }
        let spin = V3::new(d.x, d.y, d.z);
        if spin.len() < 1e-15 {
            return (v, 0.0);
        }
        let turn = 2.0 * spin.len().atan2(d.w);
        let around = self.from.rot.apply(unit(spin));
        (v, turn * cross(around, self.from.axis()).len())
    }

    /// The origin runs at a steady rate (no acceleration); the axis turns at a steady rate
    /// about a fixed axis, so a'' = turn^2 u x (u x a), of length turn^2 sin(angle to u).
    fn accel(&self, _: f64, _: f64) -> (f64, f64) {
        let mut d = self.from.rot.inverse() * self.to.rot;
        if d.w < 0.0 {
            d = Rot { w: -d.w, x: -d.x, y: -d.y, z: -d.z };
        }
        let spin = V3::new(d.x, d.y, d.z);
        if spin.len() < 1e-15 {
            return (0.0, 0.0);
        }
        let turn = 2.0 * spin.len().atan2(d.w);
        let around = self.from.rot.apply(unit(spin));
        (0.0, turn * turn * cross(around, self.from.axis()).len())
    }

    fn stray(&self, _: f64, _: f64) -> f64 {
        0.0
    }

    fn straight(&self) -> bool {
        self.from.rot.angle_to(self.to.rot) < 1e-12
    }
}

/// An arc or helix: the tool origin turns `sweep` radians (right-handed about the unit
/// `normal`) about the line through `center` along `normal`, starting at `start`, and
/// rises `rise` along `normal`; the frame keeps `rot`. A full turn is a sweep of 2 pi.
#[derive(Clone, Copy, Debug)]
pub struct Helix {
    pub center: V3,
    pub normal: V3,
    pub start: V3,
    pub sweep: f64,
    pub rise: f64,
    pub rot: Rot,
}

impl Motion for Helix {
    fn pose(&self, t: f64) -> Pose {
        let turn = Rot::about(self.normal, self.sweep * t);
        Pose { rot: self.rot, at: self.center + turn.apply(self.start - self.center) + scale(unit(self.normal), self.rise * t) }
    }

    /// The axis's point rises with the start, so the start stays level with it.
    fn piece(&self, a: f64, b: f64) -> Option<Arc<dyn Motion>> {
        let center = self.center + scale(unit(self.normal), self.rise * a);
        Some(Arc::new(Helix { center, start: self.pose(a).at, sweep: self.sweep * (b - a), rise: self.rise * (b - a), ..*self }))
    }

    fn rate(&self, _: f64, _: f64) -> (f64, f64) {
        let n = unit(self.normal);
        let r = self.start - self.center;
        let radius = (r - scale(n, dot(r, n))).len();
        ((radius * self.sweep).hypot(self.rise), 0.0)
    }

    /// The origin's acceleration is centripetal (the rise is steady); the axis is held.
    fn accel(&self, _: f64, _: f64) -> (f64, f64) {
        let n = unit(self.normal);
        let r = self.start - self.center;
        let radius = (r - scale(n, dot(r, n))).len();
        (radius * self.sweep * self.sweep, 0.0)
    }

    /// The rise is linear in t; the turn strays from its chord by at most an eighth of
    /// its second derivative: radius x (angle turned)^2 / 8.
    fn stray(&self, t0: f64, t1: f64) -> f64 {
        let n = unit(self.normal);
        let r = self.start - self.center;
        let radius = (r - scale(n, dot(r, n))).len();
        let a = self.sweep * (t1 - t0);
        radius * a * a / 8.0
    }
}

/// A machine: where its tool sits in the work frame for given axis values.
pub trait Machine: std::any::Any + Send + Sync + std::fmt::Debug {
    fn axes(&self) -> usize;
    fn pose(&self, q: &[f64]) -> Pose;
    /// Bounds (v, w) as in [`Motion::rate`] while the axes run linearly from `q0` to
    /// `q1`, axis i moving |dq_i| per unit t.
    fn rate(&self, q0: &[f64], q1: &[f64], dq: &[f64]) -> (f64, f64);
    /// Bounds as in [`Motion::accel`] over the same stretch; infinite when the machine
    /// cannot say.
    fn accel(&self, _q0: &[f64], _q1: &[f64], _dq: &[f64]) -> (f64, f64) {
        (f64::INFINITY, f64::INFINITY)
    }
}

/// One link of a serial machine, read from the work frame toward the tool.
#[derive(Clone, Copy, Debug)]
pub enum Link {
    /// A fixed offset.
    Fixed(Pose),
    /// Rotary axis `q`: turns q radians about `axis` through the current origin.
    Turn { axis: V3, q: usize },
    /// Linear axis `q`: moves q mm along `axis`.
    Slide { axis: V3, q: usize },
}

/// Any serial machine: the chain of links from the work frame to the tool frame. A
/// table-side axis is read in the same chain, turned the other way (the work turns under
/// the tool). An R-theta mill is a turn and a slide; a head-head five-axis is three
/// slides, two turns and the offsets between them.
#[derive(Clone, Debug)]
pub struct Chain {
    pub links: Vec<Link>,
    pub axes: usize,
}

impl Machine for Chain {
    fn axes(&self) -> usize {
        self.axes
    }

    fn pose(&self, q: &[f64]) -> Pose {
        let mut p = Pose::IDENTITY;
        for l in &self.links {
            p = p * match *l {
                Link::Fixed(f) => f,
                Link::Turn { axis, q: i } => Pose::new(Rot::about(axis, q[i]), V3::default()),
                Link::Slide { axis, q: i } => Pose::translation(scale(unit(axis), q[i])),
            };
        }
        p
    }

    fn rate(&self, q0: &[f64], q1: &[f64], dq: &[f64]) -> (f64, f64) {
        // A turn moves the tool origin at its rate times the origin's distance from its
        // axis, which is at most the length of the chain after it (turns keep lengths).
        let reach = |from: usize| -> f64 {
            self.links[from + 1..]
                .iter()
                .map(|l| match *l {
                    Link::Fixed(f) => f.at.len(),
                    Link::Turn { .. } => 0.0,
                    Link::Slide { q, .. } => q0[q].abs().max(q1[q].abs()),
                })
                .sum()
        };
        // A turn swings the tool axis at its rate times the sine of the angle between its
        // own axis and the tool's: read at the middle of the stretch, that sine can change
        // by no more than the two axes turn in half the stretch.
        let mid: Vec<f64> = q0.iter().zip(q1).map(|(a, b)| 0.5 * (a + b)).collect();
        let mut at = Pose::IDENTITY;
        let mut turns = Vec::new();
        let mut half_turned = 0.0;
        for (k, l) in self.links.iter().enumerate() {
            if let Link::Turn { axis, q } = *l {
                turns.push((k, q, at.rot.apply(unit(axis)), half_turned));
                half_turned += 0.5 * (q1[q] - q0[q]).abs();
            }
            at = at * match *l {
                Link::Fixed(f) => f,
                Link::Turn { axis, q } => Pose::new(Rot::about(axis, mid[q]), V3::default()),
                Link::Slide { axis, q } => Pose::translation(scale(unit(axis), mid[q])),
            };
        }
        let tool_axis = at.axis();
        let (mut v, mut w) = (0.0, 0.0);
        for l in &self.links {
            if let Link::Slide { q, .. } = *l {
                v += dq[q].abs();
            }
        }
        for (k, q, axis, before) in turns {
            v += dq[q].abs() * reach(k);
            w += dq[q].abs() * (cross(axis, tool_axis).len() + before + half_turned).min(1.0);
        }
        (v, w)
    }

    /// With every axis running at a steady rate, the second derivatives come from turns
    /// acting on turns. With W the sum of the turn rates, R the longest reach after a
    /// turn and v the origin's speed bound: the origin's velocity is a sum of k_i x r_i
    /// rates and slide rates along e_j, each k_i and e_j turning at most W and each r_i
    /// changing at most 2 v, so |c''| <= W (W R + 2 v) + W v; the axis's velocity is
    /// Omega x a with |Omega'| <= W^2, so |a''| <= 2 W^2.
    fn accel(&self, q0: &[f64], q1: &[f64], dq: &[f64]) -> (f64, f64) {
        let (v, _) = self.rate(q0, q1, dq);
        let mut turns = 0.0;
        let mut reach = 0.0f64;
        for (k, l) in self.links.iter().enumerate() {
            if let Link::Turn { q, .. } = *l {
                turns += dq[q].abs();
                let after: f64 = self.links[k + 1..]
                    .iter()
                    .map(|l| match *l {
                        Link::Fixed(f) => f.at.len(),
                        Link::Turn { .. } => 0.0,
                        Link::Slide { q, .. } => q0[q].abs().max(q1[q].abs()),
                    })
                    .sum();
                reach = reach.max(after);
            }
        }
        (turns * (turns * reach + 3.0 * v), 2.0 * turns * turns)
    }
}

/// A machine's axes running linearly from `from` to `to`.
#[derive(Clone, Debug)]
pub struct Joint {
    pub machine: Arc<dyn Machine>,
    pub from: Vec<f64>,
    pub to: Vec<f64>,
}

impl Joint {
    fn at(&self, t: f64) -> Vec<f64> {
        self.from.iter().zip(&self.to).map(|(a, b)| a + (b - a) * t).collect()
    }
}

impl Motion for Joint {
    fn pose(&self, t: f64) -> Pose {
        self.machine.pose(&self.at(t))
    }

    fn rate(&self, t0: f64, t1: f64) -> (f64, f64) {
        let dq: Vec<f64> = self.from.iter().zip(&self.to).map(|(x, y)| y - x).collect();
        self.machine.rate(&self.at(t0), &self.at(t1), &dq)
    }

    fn accel(&self, t0: f64, t1: f64) -> (f64, f64) {
        let dq: Vec<f64> = self.from.iter().zip(&self.to).map(|(x, y)| y - x).collect();
        self.machine.accel(&self.at(t0), &self.at(t1), &dq)
    }
}
