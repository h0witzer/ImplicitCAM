//! The core contact query, exact reference implementation (f64, brute force).
//!
//! Two modes over the same data:
//!   drop   lowest gouge-free tip height at (x, y):
//!          z = max over faces F of  z_F - h_min(d_F)
//!   slice  horizontal clearance with the tip at height z:
//!          c = min over faces F above z of  sd_F - R(z_F - z)
//! where d_F is the distance to face F's region, sd_F its signed version
//! (negative inside), h_min the height of the tool's lowest point at a radius
//! and R the tool radius at a height. `slice` is 1-Lipschitz in (x, y), so it
//! is a distance field for planning; `drop` is the 3-axis gouge limit.
//!
//! Every answer is a contact record, not a number: which face and boundary
//! curve and which tool element decide it, the contact point, gradient and
//! Hessian, and the runner-up face.
//!
//! This implementation checks every face and decides inside with ray crossings.
//! It shares no code path with the GPU kernel's grid and pseudo-normal logic,
//! which is what makes it a reference.

use crate::geom::{Near, V2};
use crate::part::Part;
use crate::tool::Tool;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub value: f64,
    /// Index into `Part::faces` of the deciding face, NONE if nothing is in reach.
    pub face: u32,
    /// Index into `Part::edges` of the deciding boundary curve, NONE when the
    /// contact is in a face's interior.
    pub edge: u32,
    /// Tool profile element in contact.
    pub tool_el: u32,
    /// Contact point on the part (the face's closest point) and the face height.
    pub q: V2,
    pub z_face: f64,
    pub grad: V2,
    /// Hessian of the value (xx, xy, yy).
    pub hess: [f64; 3],
    /// The best value among the other faces, and which face.
    pub second: f64,
    pub second_face: u32,
    /// Parameter of the contact on its boundary curve (spline: the source NURBS
    /// parameter), NaN for an interior contact.
    pub t: f64,
}

impl Contact {
    fn empty(value: f64) -> Contact {
        Contact {
            value,
            face: NONE,
            edge: NONE,
            tool_el: NONE,
            q: V2::default(),
            z_face: 0.0,
            grad: V2::default(),
            hess: [0.0; 3],
            second: value,
            second_face: NONE,
            t: f64::NAN,
        }
    }
}

/// Signed distance from `p` to a face's region, with the closest boundary
/// curve, gradient and Hessian.
#[derive(Clone, Copy, Debug)]
pub struct FaceDist {
    pub sd: f64,
    pub q: V2,
    pub edge: u32,
    pub grad: V2,
    pub hess: [f64; 3],
    pub t: f64,
}

pub fn face_sd(part: &Part, f: usize, p: V2) -> FaceDist {
    let mut best: Option<(usize, crate::geom::Closest)> = None;
    for (i, e) in part.face_edges(f) {
        let cut = best.as_ref().map_or(f64::INFINITY, |(_, b)| b.dist);
        if let Some(cl) = e.curve.closest_within(p, cut) {
            if best.as_ref().map_or(true, |(_, b)| cl.dist < b.dist) {
                best = Some((i, cl));
            }
        }
    }
    let (i, cl) = best.expect("face without edges");
    let sign = if part.inside(f, p) { -1.0 } else { 1.0 };
    let (grad, hess) = if cl.dist > 0.0 {
        let n = (p - cl.q) * (1.0 / cl.dist);
        let h = cl.dist_hessian(p);
        (n * sign, [h[0] * sign, h[1] * sign, h[2] * sign])
    } else {
        // On the boundary: use the outward normal there.
        let n = match cl.near {
            Near::Interior => cl.outward(),
            Near::Start => part.edges[i].pn_start,
            Near::End => part.edges[i].pn_end,
        };
        (n, [0.0; 3])
    };
    FaceDist { sd: sign * cl.dist, q: cl.q, edge: i as u32, grad, hess, t: cl.t }
}

/// Lowest tip height at (x, y) that keeps the tool out of the part.
/// `value` is -infinity when no face is within the tool's reach.
pub fn drop_ref(part: &Part, tool: &Tool, p: V2) -> Contact {
    let mut best = Contact::empty(f64::NEG_INFINITY);
    let r_max = tool.r_max();
    for f in 0..part.faces.len() {
        let fd = face_sd(part, f, p);
        let d = fd.sd.max(0.0);
        if d > r_max {
            continue;
        }
        let Some(hm) = tool.h_min(d) else { continue };
        let z = part.faces[f].z - hm.h;
        let term = if fd.sd <= 0.0 {
            Contact {
                value: z,
                face: f as u32,
                edge: NONE,
                tool_el: hm.el as u32,
                q: p,
                z_face: part.faces[f].z,
                grad: V2::default(),
                hess: [0.0; 3],
                second: f64::NEG_INFINITY,
                second_face: NONE,
                t: f64::NAN,
            }
        } else {
            let g = fd.grad;
            Contact {
                value: z,
                face: f as u32,
                edge: fd.edge,
                tool_el: hm.el as u32,
                q: fd.q,
                z_face: part.faces[f].z,
                grad: g * -hm.dh,
                hess: [
                    -hm.d2h * g.x * g.x - hm.dh * fd.hess[0],
                    -hm.d2h * g.x * g.y - hm.dh * fd.hess[1],
                    -hm.d2h * g.y * g.y - hm.dh * fd.hess[2],
                ],
                second: f64::NEG_INFINITY,
                second_face: NONE,
                t: fd.t,
            }
        };
        if term.value > best.value {
            let (s, sf) = (best.value, best.face);
            best = Contact { second: s, second_face: sf, ..term };
        } else if term.value > best.second {
            best.second = term.value;
            best.second_face = term.face;
        }
    }
    best
}

/// Horizontal clearance with the tip at height `z`: how far the tool center is
/// from gouging, measured sideways (negative = gouging by that much).
/// `value` is +infinity when no face is above the tip.
pub fn slice_ref(part: &Part, tool: &Tool, p: V2, z: f64) -> Contact {
    let mut best = Contact::empty(f64::INFINITY);
    for f in 0..part.faces.len() {
        let zf = part.faces[f].z;
        if zf <= z + 1e-9 {
            continue;
        }
        let (r, el) = tool.radius_at(zf - z);
        let fd = face_sd(part, f, p);
        let term = Contact {
            value: fd.sd - r,
            face: f as u32,
            edge: fd.edge,
            tool_el: el as u32,
            q: fd.q,
            z_face: zf,
            grad: fd.grad,
            hess: fd.hess,
            second: f64::INFINITY,
            second_face: NONE,
            t: fd.t,
        };
        if term.value < best.value {
            let (s, sf) = (best.value, best.face);
            best = Contact { second: s, second_face: sf, ..term };
        } else if term.value < best.second {
            best.second = term.value;
            best.second_face = term.face;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Curve;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Curve> {
        let (a, b, c, d) = (V2::new(x0, y0), V2::new(x1, y0), V2::new(x1, y1), V2::new(x0, y1));
        vec![Curve::Line { a, b }, Curve::Line { a: b, b: c }, Curve::Line { a: c, b: d }, Curve::Line { a: d, b: a }]
    }

    /// A 10 x 10 block, top at z = 10, with a 4 mm-radius round hole down to a floor at z = 4.
    fn block() -> Part {
        let mut p = Part::default();
        let c = V2::new(5.0, 5.0);
        let hole = vec![Curve::Arc { c, r: 2.0, a: V2::new(7.0, 5.0), sweep: -2.0 * std::f64::consts::PI }];
        p.add_face(1, 10.0, vec![square(0.0, 0.0, 10.0, 10.0), hole], None, 1e-9).unwrap();
        let floor = vec![Curve::Arc { c, r: 2.0, a: V2::new(7.0, 5.0), sweep: 2.0 * std::f64::consts::PI }];
        p.add_face(2, 4.0, vec![floor], None, 1e-9).unwrap();
        p
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn areas_are_exact() {
        let p = block();
        let pi = std::f64::consts::PI;
        assert!(close(p.face_area(0), 100.0 - 4.0 * pi));
        assert!(close(p.face_area(1), 4.0 * pi));
    }

    #[test]
    fn drop_values_by_tool() {
        let p = block();
        let out = V2::new(12.0, 5.0); // 2 mm right of the block
        assert!(close(drop_ref(&p, &Tool::flat(6.0, 30.0), out).value, 10.0));
        assert!(close(drop_ref(&p, &Tool::ball(6.0, 30.0), out).value, 10.0 - (3.0 - (9.0f64 - 4.0).sqrt())));
        assert!(close(drop_ref(&p, &Tool::vee(6.0, 90.0, 30.0), out).value, 8.0));
        let bull = Tool::bull(6.0, 1.0, 30.0);
        assert!(close(drop_ref(&p, &bull, out).value, 10.0));
        assert!(close(drop_ref(&p, &bull, V2::new(12.5, 5.0)).value, 10.0 - (1.0 - (1.0f64 - 0.25).sqrt())));
        // Over the hole's center a 2 mm flat drops to the floor; a 6 mm one sits on the top.
        assert!(close(drop_ref(&p, &Tool::flat(2.0, 30.0), V2::new(5.0, 5.0)).value, 4.0));
        assert!(close(drop_ref(&p, &Tool::flat(6.0, 30.0), V2::new(5.0, 5.0)).value, 10.0));
        // A 90 deg V over the hole center touches the rim 2 mm out: 10 - 2.
        assert!(close(drop_ref(&p, &Tool::vee(6.0, 90.0, 30.0), V2::new(5.0, 5.0)).value, 8.0));
    }

    #[test]
    fn slice_values_and_signs() {
        let p = block();
        let t = Tool::flat(2.0, 30.0);
        // Tip at z = 5 in the hole center: the rim is 2 mm away, the tool radius 1.
        let c = slice_ref(&p, &t, V2::new(5.0, 5.0), 5.0);
        assert!(close(c.value, 1.0));
        assert_eq!(c.face, 0);
        // Tip at z = 3: the floor at z = 4 is above the tip and covers the point: gouging.
        let c = slice_ref(&p, &t, V2::new(5.0, 5.0), 3.0);
        assert!(close(c.value, -2.0 - 1.0));
        assert_eq!(c.face, 1);
    }

    /// A D-shaped face: a straight bottom and a cubic B-spline top, at z = 5.
    fn d_shape() -> Part {
        use crate::bspline::{Nurbs2, Spline2};
        let ctrl = vec![V2::new(10.0, 0.0), V2::new(11.0, 6.0), V2::new(5.0, 9.0), V2::new(-1.0, 6.0), V2::new(0.0, 0.0)];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let nb = Nurbs2::new(3, ctrl, None, knots).unwrap();
        let top = Curve::Spline(std::sync::Arc::new(Spline2::new(nb, 0.0, 1.0)));
        let bottom = Curve::Line { a: V2::new(0.0, 0.0), b: V2::new(10.0, 0.0) };
        let mut p = Part::default();
        p.add_face(7, 5.0, vec![vec![bottom, top]], None, 1e-12).unwrap();
        p
    }

    #[test]
    fn spline_face_derivatives_match_finite_differences() {
        let p = d_shape();
        let h = 1e-5;
        let tools = [Tool::ball(4.0, 20.0), Tool::vee(6.0, 90.0, 20.0)];
        for q in [V2::new(5.0, 10.5), V2::new(12.0, 5.0), V2::new(5.0, 6.0), V2::new(-1.5, 3.0)] {
            // Slice: value sd - R, gradient and Hessian of sd.
            let c = slice_ref(&p, &tools[0], q, 1.0);
            let f = |x: V2| slice_ref(&p, &tools[0], x, 1.0).value;
            let (gx, gy) = ((f(q + V2::new(h, 0.0)) - f(q - V2::new(h, 0.0))) / (2.0 * h), (f(q + V2::new(0.0, h)) - f(q - V2::new(0.0, h))) / (2.0 * h));
            assert!((c.grad.x - gx).abs() < 1e-6 && (c.grad.y - gy).abs() < 1e-6, "slice grad at {q:?}: {:?} vs ({gx}, {gy})", c.grad);
            let hxx = (f(q + V2::new(h, 0.0)) - 2.0 * f(q) + f(q - V2::new(h, 0.0))) / (h * h);
            let hyy = (f(q + V2::new(0.0, h)) - 2.0 * f(q) + f(q - V2::new(0.0, h))) / (h * h);
            assert!((c.hess[0] - hxx).abs() < 1e-3 && (c.hess[2] - hyy).abs() < 1e-3, "slice hess at {q:?}: {:?} vs ({hxx}, {hyy})", c.hess);
            // Drop, for both tools, outside the face.
            if !p.inside(0, q) {
                for t in &tools {
                    let c = drop_ref(&p, t, q);
                    if c.face == NONE {
                        continue;
                    }
                    let g = |x: V2| drop_ref(&p, t, x).value;
                    let gx = (g(q + V2::new(h, 0.0)) - g(q - V2::new(h, 0.0))) / (2.0 * h);
                    let gy = (g(q + V2::new(0.0, h)) - g(q - V2::new(0.0, h))) / (2.0 * h);
                    if !(gx.is_finite() && gy.is_finite()) {
                        continue; // a neighbor is out of the tool's reach
                    }
                    assert!((c.grad.x - gx).abs() < 1e-5 && (c.grad.y - gy).abs() < 1e-5, "{} drop grad at {q:?}: {:?} vs ({gx}, {gy})", t.name, c.grad);
                }
            }
        }
        // The contact reports the spline's own parameter.
        let c = slice_ref(&p, &tools[0], V2::new(5.0, 10.5), 1.0);
        assert!(c.t > 0.3 && c.t < 0.7, "spline parameter at the top {}", c.t);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let p = block();
        let tools = [Tool::ball(6.0, 30.0), Tool::bull(6.0, 1.0, 30.0), Tool::vee(6.0, 60.0, 30.0)];
        let h = 1e-6;
        for t in &tools {
            for q in [V2::new(11.3, 4.1), V2::new(6.1, 5.9), V2::new(-1.2, -0.7)] {
                let c = drop_ref(&p, t, q);
                let gx = (drop_ref(&p, t, q + V2::new(h, 0.0)).value - drop_ref(&p, t, q - V2::new(h, 0.0)).value) / (2.0 * h);
                let gy = (drop_ref(&p, t, q + V2::new(0.0, h)).value - drop_ref(&p, t, q - V2::new(0.0, h)).value) / (2.0 * h);
                assert!((c.grad.x - gx).abs() < 1e-5 && (c.grad.y - gy).abs() < 1e-5, "{} at {q:?}: {:?} vs ({gx}, {gy})", t.name, c.grad);
            }
        }
    }
}
