//! A floor under the distance from a point to a convex set known only by its support
//! mapping (its farthest point in any direction), by the Gilbert-Johnson-Keerthi
//! iteration: the nearest point of a growing simplex of support points closes in on the
//! set's nearest point from outside.
//!
//! Every step gives a certificate. With v the simplex's point nearest p (relative to p)
//! and w the set's support point toward p (least along v), every point y of the set has
//! v . (y - p) >= v . w, so its distance from p is at least v . w / |v|; and |v| is a
//! distance actually reached, so the set's distance lies between the two. The iteration
//! stops as soon as the floor settles what the caller asked, either way.

use crate::geom::{cross, dot, scale, V3};

/// The point of segment ab nearest the origin, and the vertices that hold it.
fn segment(a: V3, b: V3) -> (V3, Vec<V3>) {
    let ab = b - a;
    let t = -dot(a, ab) / dot(ab, ab).max(1e-300);
    if t <= 0.0 {
        (a, vec![a])
    } else if t >= 1.0 {
        (b, vec![b])
    } else {
        (a + scale(ab, t), vec![a, b])
    }
}

/// The point of triangle abc nearest the origin, and the vertices that hold it (by the
/// regions of the triangle's Voronoi diagram, after Ericson).
fn triangle(a: V3, b: V3, c: V3) -> (V3, Vec<V3>) {
    let (ab, ac) = (b - a, c - a);
    let (d1, d2) = (-dot(ab, a), -dot(ac, a));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, vec![a]);
    }
    let (d3, d4) = (-dot(ab, b), -dot(ac, b));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, vec![b]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + scale(ab, d1 / (d1 - d3)), vec![a, b]);
    }
    let (d5, d6) = (-dot(ab, c), -dot(ac, c));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, vec![c]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + scale(ac, d2 / (d2 - d6)), vec![a, c]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + scale(c - b, w), vec![b, c]);
    }
    let s = 1.0 / (va + vb + vc);
    (a + scale(ab, vb * s) + scale(ac, vc * s), vec![a, b, c])
}

/// The point of tetrahedron abcd nearest the origin (the origin itself when inside), and
/// the vertices that hold it.
fn tetrahedron(a: V3, b: V3, c: V3, d: V3) -> (V3, Vec<V3>) {
    // The origin is outside face xyz when it lies across the face's plane from w.
    let outside = |x: V3, y: V3, z: V3, w: V3| {
        let n = cross(y - x, z - x);
        let (so, sw) = (-dot(x, n), dot(w - x, n));
        so * sw < 0.0 || sw.abs() < 1e-300
    };
    let mut best: Option<(V3, Vec<V3>)> = None;
    for (x, y, z, w) in [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)] {
        if outside(x, y, z, w) {
            let (q, keep) = triangle(x, y, z);
            if best.as_ref().is_none_or(|(b, _)| q.len_sq() < b.len_sq()) {
                best = Some((q, keep));
            }
        }
    }
    best.unwrap_or((V3::default(), vec![a, b, c, d]))
}

/// A floor under the distance from p to the set `support` describes. Returns as soon as
/// the floor reaches `enough`, or the distance is known to be under it (then the floor
/// so far), or the floor is within `gap` of the distance. 0 when p may be inside the set.
pub(crate) fn distance_floor(support: &impl Fn(V3) -> V3, p: V3, start: V3, enough: f64, gap: f64) -> f64 {
    let mut simplex: Vec<V3> = Vec::with_capacity(4);
    let mut v = support(start - p) - p;
    let mut floor = 0.0f64;
    for _ in 0..64 {
        let dist = v.len();
        if dist < 1e-12 {
            return 0.0;
        }
        let w = support(scale(v, -1.0)) - p;
        floor = floor.max(dot(v, w) / dist);
        if floor >= enough || dist < enough || dist - floor <= gap {
            return floor;
        }
        simplex.push(w);
        let (next, keep) = match simplex.len() {
            1 => (simplex[0], vec![simplex[0]]),
            2 => segment(simplex[0], simplex[1]),
            3 => triangle(simplex[0], simplex[1], simplex[2]),
            _ => tetrahedron(simplex[0], simplex[1], simplex[2], simplex[3]),
        };
        simplex = keep;
        if next.len() >= dist * (1.0 - 1e-12) {
            return floor;
        }
        v = next;
    }
    floor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_meets_known_distances() {
        // A ball of radius 2 at c, and a box from lo to hi.
        let c = V3::new(1.0, -2.0, 3.0);
        let ball = |d: V3| c + scale(d, 2.0 / d.len());
        let (lo, hi) = (V3::new(-1.0, -1.0, -1.0), V3::new(2.0, 1.0, 0.5));
        let block = |d: V3| V3::new(if d.x >= 0.0 { hi.x } else { lo.x }, if d.y >= 0.0 { hi.y } else { lo.y }, if d.z >= 0.0 { hi.z } else { lo.z });
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0
        };
        for _ in 0..2000 {
            let p = V3::new(r(), r(), r());
            let want = ((p - c).len() - 2.0).max(0.0);
            // Asking for exactly the distance runs the iteration until it closes in.
            let got = distance_floor(&ball, p, c, want, 1e-10);
            assert!(got <= want + 1e-9 && got >= want - 1e-6, "ball: {got} for {want} at {p:?}");
            let q = V3::new(p.x.clamp(lo.x, hi.x), p.y.clamp(lo.y, hi.y), p.z.clamp(lo.z, hi.z));
            let want = (p - q).len();
            let got = distance_floor(&block, p, V3::default(), want, 1e-10);
            assert!(got <= want + 1e-9 && got >= want - 1e-6, "box: {got} for {want} at {p:?}");
        }
    }
}

