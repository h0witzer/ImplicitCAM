//! Points, rotations, placements and boxes in 3 space.

pub use icam_post::V3;

pub fn dot(a: V3, b: V3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn cross(a: V3, b: V3) -> V3 {
    V3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

pub fn scale(a: V3, s: f64) -> V3 {
    V3::new(a.x * s, a.y * s, a.z * s)
}

pub fn unit(a: V3) -> V3 {
    scale(a, 1.0 / a.len().max(1e-300))
}

/// A rotation, as a unit quaternion (w, x, y, z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rot {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Rot {
    pub const IDENTITY: Rot = Rot { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// `angle` radians about `axis` (right-handed; the axis need not be unit length).
    pub fn about(axis: V3, angle: f64) -> Rot {
        let a = unit(axis);
        let (s, c) = (0.5 * angle).sin_cos();
        Rot { w: c, x: a.x * s, y: a.y * s, z: a.z * s }
    }

    /// The shortest turn that takes +z onto `dir`.
    pub fn z_to(dir: V3) -> Rot {
        let d = unit(dir);
        let c = d.z.clamp(-1.0, 1.0);
        let axis = V3::new(-d.y, d.x, 0.0);
        if axis.len() < 1e-12 {
            return if c > 0.0 { Rot::IDENTITY } else { Rot::about(V3::new(1.0, 0.0, 0.0), std::f64::consts::PI) };
        }
        Rot::about(axis, c.acos())
    }

    fn v(self) -> V3 {
        V3::new(self.x, self.y, self.z)
    }

    pub fn apply(self, p: V3) -> V3 {
        let u = self.v();
        let t = scale(cross(u, p), 2.0);
        p + scale(t, self.w) + cross(u, t)
    }

    pub fn inverse(self) -> Rot {
        Rot { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// `self` after `o`: applies `o` first.
    fn compose(self, o: Rot) -> Rot {
        let (u, v) = (self.v(), o.v());
        let c = cross(u, v);
        Rot {
            w: self.w * o.w - dot(u, v),
            x: self.w * v.x + o.w * u.x + c.x,
            y: self.w * v.y + o.w * u.y + c.y,
            z: self.w * v.z + o.w * u.z + c.z,
        }
        .normalized()
    }

    fn normalized(self) -> Rot {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Rot { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    fn qdot(self, o: Rot) -> f64 {
        self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// The angle of the turn from `self` to `o`, radians in [0, pi].
    pub fn angle_to(self, o: Rot) -> f64 {
        let d = self.inverse() * o;
        2.0 * d.v().len().atan2(d.w.abs())
    }

    /// The turn from `self` to `o` a fraction `t` of the way, about one fixed axis at a
    /// steady rate (the shorter way round).
    pub fn slerp(self, o: Rot, t: f64) -> Rot {
        let mut d = self.qdot(o);
        let o = if d < 0.0 {
            d = -d;
            Rot { w: -o.w, x: -o.x, y: -o.y, z: -o.z }
        } else {
            o
        };
        let (a, b) = if d > 1.0 - 1e-12 {
            (1.0 - t, t)
        } else {
            let th = d.acos();
            let s = th.sin();
            (((1.0 - t) * th).sin() / s, (t * th).sin() / s)
        };
        Rot { w: a * self.w + b * o.w, x: a * self.x + b * o.x, y: a * self.y + b * o.y, z: a * self.z + b * o.z }.normalized()
    }
}

/// Where a frame sits in another: its point p is at `rot` p + `at`. For the tool, the
/// frame's origin is the tool tip and its +z runs up the axis toward the spindle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub rot: Rot,
    pub at: V3,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { rot: Rot::IDENTITY, at: V3::new(0.0, 0.0, 0.0) };

    pub fn new(rot: Rot, at: V3) -> Pose {
        Pose { rot, at }
    }

    /// The tool tip at `at`, its axis (tip toward spindle) along `axis`.
    pub fn tip_axis(at: V3, axis: V3) -> Pose {
        Pose { rot: Rot::z_to(axis), at }
    }

    pub fn translation(at: V3) -> Pose {
        Pose { rot: Rot::IDENTITY, at }
    }

    /// A point of this frame, in the outer frame.
    pub fn apply(self, p: V3) -> V3 {
        self.rot.apply(p) + self.at
    }

    /// A point of the outer frame, in this frame.
    pub fn to_local(self, p: V3) -> V3 {
        self.rot.inverse().apply(p - self.at)
    }

    /// `self` after `o`: a point of `o`'s frame placed by `o`, then by `self`.
    fn compose(self, o: Pose) -> Pose {
        Pose { rot: self.rot * o.rot, at: self.rot.apply(o.at) + self.at }
    }

    pub fn inverse(self) -> Pose {
        let r = self.rot.inverse();
        Pose { rot: r, at: scale(r.apply(self.at), -1.0) }
    }

    /// The frame's +z in the outer frame (the tool axis).
    pub fn axis(self) -> V3 {
        self.rot.apply(V3::new(0.0, 0.0, 1.0))
    }
}

/// `a * b` is `a` after `b`: `b` applied first.
impl std::ops::Mul for Rot {
    type Output = Rot;
    fn mul(self, o: Rot) -> Rot {
        self.compose(o)
    }
}

/// `a * b` is `a` after `b`: a point of `b`'s frame placed by `b`, then by `a`.
impl std::ops::Mul for Pose {
    type Output = Pose;
    fn mul(self, o: Pose) -> Pose {
        self.compose(o)
    }
}

/// An axis-aligned box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: V3,
    pub max: V3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: V3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY), max: V3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY) };

    pub fn new(min: V3, max: V3) -> Aabb {
        Aabb { min, max }
    }

    pub fn union(self, o: Aabb) -> Aabb {
        Aabb {
            min: V3::new(self.min.x.min(o.min.x), self.min.y.min(o.min.y), self.min.z.min(o.min.z)),
            max: V3::new(self.max.x.max(o.max.x), self.max.y.max(o.max.y), self.max.z.max(o.max.z)),
        }
    }

    pub fn grow(self, r: f64) -> Aabb {
        Aabb { min: self.min - V3::new(r, r, r), max: self.max + V3::new(r, r, r) }
    }

    /// Distance from p to the box (0 inside).
    pub fn dist(self, p: V3) -> f64 {
        let d = |lo: f64, hi: f64, v: f64| (lo - v).max(0.0).max(v - hi);
        V3::new(d(self.min.x, self.max.x, p.x), d(self.min.y, self.max.y, p.y), d(self.min.z, self.max.z, p.z)).len()
    }

    /// Exact signed distance to the box's boundary (negative inside).
    pub fn sd(self, p: V3) -> f64 {
        let q = V3::new((p.x - 0.5 * (self.min.x + self.max.x)).abs() - 0.5 * (self.max.x - self.min.x), (p.y - 0.5 * (self.min.y + self.max.y)).abs() - 0.5 * (self.max.y - self.min.y), (p.z - 0.5 * (self.min.z + self.max.z)).abs() - 0.5 * (self.max.z - self.min.z));
        V3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).len() + q.x.max(q.y).max(q.z).min(0.0)
    }

    pub fn center(self) -> V3 {
        scale(self.min + self.max, 0.5)
    }
}
