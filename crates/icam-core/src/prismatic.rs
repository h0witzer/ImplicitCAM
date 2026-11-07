//! Prismatic STEP solids -> exact horizontal faces (milestone-1 input to the
//! F-rep compile).
//!
//! A solid is prismatic (for a vertical tool) when every face is a horizontal
//! plane or a vertical wall: a vertical plane, a cylinder with a vertical axis,
//! a surface of linear extrusion along z, or a B-spline surface whose control
//! net is vertical in one direction. Its horizontal faces, with their line, arc
//! and B-spline boundaries, then describe it completely for a tool coming from
//! above. Everything comes straight from the B-rep: line endpoints are the
//! file's vertices, arcs are the file's circles, B-spline edges are the file's
//! own NURBS (control points, weights, knots) trimmed between the edge's
//! vertices, and loop orientation follows the face's outward normal. Nothing
//! is approximated by lines or arcs.
//!
//! Fillets and round-overs (a torus or a cone about a vertical axis, a cylinder
//! about a horizontal one) are carried conservatively, for roughing: each stands as
//! solid over its footprint (its boundary projected to the plan; a fillet meets its
//! neighbors tangentially, so its boundary bounds its shadow) from the bottom up to
//! its top, in `Part::envelope`. The faces beside it are kept exact, a curve next to
//! a fillet taking the depth of the wall beyond the fillet. Anything else is refused
//! by name.

use std::sync::Arc;

use crate::bspline::{Nurbs2, Spline2};
use crate::geom::{Curve, V2};
use crate::part::Part;
use crate::step::{BFace, BSplineSurface, Brep, Curve3, Surface, V3};

/// |axis z| within this of 1 is vertical, of 0 horizontal. STEP writes
/// directions to ~15 digits, so this only absorbs the last digit.
const AXIS_TOL: f64 = 1e-12;
/// Tangent turn (radians) inside a B-spline edge that counts as a corner.
const KINK: f64 = 1e-6;

#[derive(Clone, Debug, Default)]
pub struct PrismReport {
    pub solids: usize,
    pub horizontal: usize,
    pub vertical_planes: usize,
    pub vertical_cylinders: usize,
    /// Vertical walls that are B-spline surfaces or linear extrusions.
    pub vertical_freeform: usize,
    pub lines: usize,
    pub arcs: usize,
    /// B-spline edges kept native, and the flat rational Bezier pieces they became.
    pub splines: usize,
    pub spline_pieces: usize,
    /// Largest distance of a horizontal face's vertex from the face's plane or
    /// from its own curve (mm): the file's own rounding.
    pub max_vertex_off: f64,
    /// Largest move of a line or arc end onto its junction (mm). Within the
    /// file's declared tolerance, or the import fails.
    pub max_snap: f64,
    /// Largest remaining gap where an arc meets a B-spline (the spline's end is
    /// exact and never moved; the arc's end sits on its circle).
    pub max_gap: f64,
    pub tolerance_mm: f64,
    /// Fillet and round-over faces carried as envelopes, and the heights they stand to.
    pub envelopes: usize,
    pub envelope_levels: usize,
    /// Faces looking down above their solid's bottom: overhangs, filled below for a
    /// tool from above.
    pub overhangs: usize,
}

fn reverse(c: Curve) -> Curve {
    c.reversed()
}

/// A B-spline surface is a vertical wall when, along one parameter direction,
/// every control point of a row shares its x and y, and the weights along that
/// direction are proportional from row to row (so every surface point in that
/// direction keeps the same x and y).
fn vertical_bspline(s: &BSplineSurface, tol: f64) -> bool {
    let rows = s.ctrl.len();
    let cols = s.ctrl.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return false;
    }
    let same = |a: V3, b: V3| (a.x - b.x).hypot(a.y - b.y) <= tol;
    let w = |i: usize, j: usize| s.weights.as_ref().map_or(1.0, |w| w[i][j]);
    let u_vertical = (0..cols).all(|j| (0..rows).all(|i| same(s.ctrl[i][j], s.ctrl[0][j])))
        && (0..rows).all(|i| (0..cols).all(|j| (w(i, j) / w(0, j) - w(i, 0) / w(0, 0)).abs() < 1e-12));
    let v_vertical = (0..rows).all(|i| (0..cols).all(|j| same(s.ctrl[i][j], s.ctrl[i][0])))
        && (0..cols).all(|j| (0..rows).all(|i| (w(i, j) / w(i, 0) - w(0, j) / w(0, 0)).abs() < 1e-12));
    u_vertical || v_vertical
}

/// The import entry for runtime code: a prismatic STEP file read and compiled straight to
/// its F-rep, so nothing past import holds the B-rep. Returns the field, the part's XY
/// box, and the import report. The compile tolerance is the file's own (at least 1e-6 mm).
pub fn compile_file(path: &std::path::Path) -> Result<(crate::frep::Frep, (V2, V2), PrismReport), String> {
    let brep = crate::step::read(path)?;
    let (part, rep) = from_brep(&brep)?;
    let frep = crate::frep::compile_prismatic(&part, rep.tolerance_mm.max(1e-6))?;
    Ok((frep, part.bbox(), rep))
}

/// Faces carried as envelopes: a torus or cone about a vertical axis, a cylinder about a
/// horizontal one.
fn is_envelope(s: &Surface) -> bool {
    match s {
        Surface::Torus { frame, .. } | Surface::Cone { frame, .. } => frame.z.z.abs() >= 1.0 - AXIS_TOL,
        Surface::Cylinder { frame, .. } => frame.z.z.abs() < AXIS_TOL,
        _ => false,
    }
}

/// Signed area of a closed loop of curves (positive counterclockwise); a spline counts
/// as its chord, which is enough to tell a loop's sense.
fn signed_area(lp: &[(Curve, u32)]) -> f64 {
    let mut a = 0.0;
    for (c, _) in lp {
        let (s, e) = (c.start(), c.end());
        a += 0.5 * s.cross(e);
        if let Curve::Arc { c: ctr, r, sweep, .. } = c {
            a += 0.5 * (r * r * sweep - (s - *ctr).cross(e - *ctr));
        }
    }
    a
}

/// An arc in a horizontal plane, traversed from s to t as the edge runs.
fn plan_arc(cf: &crate::step::Frame, r: f64, s: V2, t: V2, forward: bool, full: bool) -> Curve {
    let ctr = V2::new(cf.o.x, cf.o.y);
    let ccw = forward == (cf.z.z > 0.0);
    let (u, v) = (s - ctr, t - ctr);
    let ang = u.cross(v).atan2(u.dot(v));
    let two_pi = 2.0 * std::f64::consts::PI;
    let sweep = match (ccw, full) {
        (true, true) => two_pi,
        (false, true) => -two_pi,
        (true, false) => {
            if ang > 0.0 {
                ang
            } else {
                ang + two_pi
            }
        }
        (false, false) => {
            if ang < 0.0 {
                ang
            } else {
                ang - two_pi
            }
        }
    };
    Curve::Arc { c: ctr, r, a: s, sweep }
}

/// A B-spline edge projected to the plan (dropping z keeps it an exact NURBS), trimmed
/// between the edge's vertices and split at its kinks.
fn plan_spline(bc: &crate::step::BSplineCurve, s: V2, t: V2, e: &crate::step::OEdge, face: u32, rep: &mut PrismReport) -> Result<Vec<Curve>, String> {
    let nb = Nurbs2::new(bc.degree, bc.ctrl.iter().map(|q| V2::new(q.x, q.y)).collect(), bc.weights.clone(), bc.knots.clone())
        .map_err(|m| format!("face #{face}: edge #{}: {m}", e.edge))?;
    let (d0, d1) = nb.domain();
    let (ta, tb) = if e.start_vertex == e.end_vertex {
        if e.forward {
            (d0, d1)
        } else {
            (d1, d0)
        }
    } else {
        let full = Spline2::new(nb.clone(), d0, d1);
        let hs = full.closest(s, f64::INFINITY).unwrap();
        let ht = full.closest(t, f64::INFINITY).unwrap();
        rep.max_vertex_off = rep.max_vertex_off.max(hs.dist).max(ht.dist);
        if (hs.t < ht.t) != e.forward {
            return Err(format!("face #{face}: B-spline edge #{} runs against its sense (seam on a closed curve?)", e.edge));
        }
        (hs.t, ht.t)
    };
    let mut out = Vec::new();
    for part_sp in Spline2::new(nb, ta, tb).split_at_kinks(KINK) {
        rep.splines += 1;
        rep.spline_pieces += part_sp.pieces.len();
        out.push(Curve::Spline(Arc::new(part_sp)));
    }
    Ok(out)
}

/// An envelope face's footprint: its boundary projected to closed loops of curves tagged
/// with their STEP edges (the outer counterclockwise, the others clockwise; a seam
/// cancels), and the face's top.
fn footprint(f: &BFace, tol: f64, rep: &mut PrismReport) -> Result<(Vec<Vec<(Curve, u32)>>, f64), String> {
    let mut loops: Vec<(Curve, u32)> = Vec::new();
    let mut top = f64::NEG_INFINITY;
    for b in &f.bounds {
        if b.vertex.is_some() {
            return Err(format!("face #{}: point bound on a fillet", f.id));
        }
        let mut lp: Vec<(Curve, u32)> = Vec::new();
        for e in &b.edges {
            top = top.max(e.start.z).max(e.end.z);
            let (s, t) = (V2::new(e.start.x, e.start.y), V2::new(e.end.x, e.end.y));
            match &e.curve {
                Curve3::Line { .. } => {
                    if (t - s).len() > tol {
                        lp.push((Curve::Line { a: s, b: t }, e.edge));
                    }
                }
                Curve3::Circle { frame: cf, r } => {
                    if cf.z.z.abs() >= 1.0 - AXIS_TOL {
                        lp.push((plan_arc(cf, *r, s, t, e.forward, e.start_vertex == e.end_vertex), e.edge));
                    } else if cf.z.z.abs() < AXIS_TOL {
                        // In a vertical plane: the arc projects onto the plane's horizontal
                        // line, and must not turn back on it (or reach above its ends).
                        let (u, v) = (cf.x, cf.y());
                        let hz = cf.z.cross(V3 { x: 0.0, y: 0.0, z: 1.0 }).unit();
                        let ang = |q: V3| {
                            let d = V3 { x: q.x - cf.o.x, y: q.y - cf.o.y, z: q.z - cf.o.z };
                            d.dot(v).atan2(d.dot(u))
                        };
                        let (a0, a1) = (ang(e.start), ang(e.end));
                        let two_pi = 2.0 * std::f64::consts::PI;
                        let span = if e.forward { (a1 - a0).rem_euclid(two_pi) } else { -(a0 - a1).rem_euclid(two_pi) };
                        let inside = |th: f64| {
                            let d = if span >= 0.0 { (th - a0).rem_euclid(two_pi) } else { (a0 - th).rem_euclid(two_pi) };
                            d > 1e-9 && d < span.abs() - 1e-9
                        };
                        let w = hz.dot(v).atan2(hz.dot(u));
                        if inside(w) || inside(w + std::f64::consts::PI) {
                            return Err(format!("face #{}: edge #{} turns back in plan (not a fillet the envelope can carry)", f.id, e.edge));
                        }
                        let zt = v.z.atan2(u.z);
                        if inside(zt) {
                            top = top.max(cf.o.z + r * u.z.hypot(v.z));
                        }
                        if (t - s).len() > tol {
                            lp.push((Curve::Line { a: s, b: t }, e.edge));
                        }
                    } else {
                        return Err(format!("face #{}: edge #{} is an arc in a tilted plane", f.id, e.edge));
                    }
                }
                Curve3::BSpline(bc) => {
                    for q in &bc.ctrl {
                        top = top.max(q.z);
                    }
                    for c in plan_spline(bc, s, t, e, f.id, rep)? {
                        lp.push((c, e.edge));
                    }
                }
                other => return Err(format!("face #{}: edge #{} is a {} (not a line, arc or B-spline)", f.id, e.edge, curve_name(other))),
            }
        }
        loops.extend(lp);
    }
    // A seam (an edge the face uses twice) cancels; what is left chains into loops.
    let mut uses: std::collections::HashMap<u32, usize> = Default::default();
    for b in &f.bounds {
        for e in &b.edges {
            *uses.entry(e.edge).or_default() += 1;
        }
    }
    let curves: Vec<(Curve, u32)> = loops.into_iter().filter(|(_, e)| uses[e] == 1).collect();
    let (chained, tags) = crate::frep::chain_loops(&curves, tol, top).map_err(|m| format!("fillet #{}: {m}", f.id))?;
    let mut tag = tags.into_iter();
    let loops: Vec<Vec<(Curve, u32)>> = chained.into_iter().map(|l| l.into_iter().map(|c| (c, tag.next().unwrap_or(u32::MAX))).collect()).collect();
    let areas: Vec<f64> = loops.iter().map(|l| signed_area(l)).collect();
    let outer = (0..loops.len()).max_by(|&a, &b| areas[a].abs().total_cmp(&areas[b].abs()));
    let mut out = Vec::new();
    for (k, lp) in loops.into_iter().enumerate() {
        if areas[k].abs() <= tol * tol {
            continue;
        }
        let want_ccw = Some(k) == outer;
        out.push(if (areas[k] > 0.0) == want_ccw { lp } else { lp.into_iter().rev().map(|(c, e)| (reverse(c), e)).collect() });
    }
    Ok((out, top))
}

pub use crate::step::Place;

/// A STEP part read once, to compile under different placements (a nest places a part
/// turned, or upside down, on its sheet).
pub struct ReadPart {
    brep: Brep,
}

impl ReadPart {
    pub fn read(path: &std::path::Path) -> Result<ReadPart, String> {
        Ok(ReadPart { brep: crate::step::read(path)? })
    }

    /// The placed part's box from its vertices: (min, max) as (x, y, z).
    pub fn bounds(&self, place: &Place) -> ([f64; 3], [f64; 3]) {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for s in &self.brep.solids {
            for f in &s.faces {
                for b in &f.bounds {
                    for e in &b.edges {
                        for p in [place.point(e.start), place.point(e.end)] {
                            for (k, v) in [p.x, p.y, p.z].into_iter().enumerate() {
                                lo[k] = lo[k].min(v);
                                hi[k] = hi[k].max(v);
                            }
                        }
                    }
                }
            }
        }
        (lo, hi)
    }

    /// The placed part compiled to its F-rep (as `compile_file`).
    pub fn compile(&self, place: &Place) -> Result<(crate::frep::Frep, (V2, V2), PrismReport), String> {
        // Unplaced, the part is read where it is: no copy of the B-rep.
        let placed;
        let brep = if *place == Place::default() {
            &self.brep
        } else {
            placed = self.brep.placed(place);
            &placed
        };
        let (part, rep) = from_brep(brep)?;
        let frep = crate::frep::compile_prismatic(&part, rep.tolerance_mm.max(1e-6))?;
        Ok((frep, part.bbox(), rep))
    }
}

pub fn from_brep(brep: &Brep) -> Result<(Part, PrismReport), String> {
    let mut part = Part::default();
    let tol = brep.tolerance_mm.max(1e-6);
    // Which faces use each STEP edge, and each face's lowest z: a horizontal
    // face's curve records how far down the wall on its other side reaches.
    let mut edge_faces: std::collections::HashMap<u32, Vec<u32>> = Default::default();
    let mut face_zmin: std::collections::HashMap<u32, f64> = Default::default();
    for solid in &brep.solids {
        for f in &solid.faces {
            let mut zmin = f64::INFINITY;
            for b in &f.bounds {
                if let Some(v) = b.vertex {
                    zmin = zmin.min(v.z);
                }
                for e in &b.edges {
                    edge_faces.entry(e.edge).or_default().push(f.id);
                    zmin = zmin.min(e.start.z).min(e.end.z);
                }
            }
            face_zmin.insert(f.id, zmin);
        }
    }
    // Face kinds: 0 a horizontal plane, 1 an envelope (fillet), 2 a wall or anything else;
    // and the planes facing down (a bottom, or an overhang's roof) with their heights.
    let mut kind: std::collections::HashMap<u32, u8> = Default::default();
    let mut down: std::collections::HashMap<u32, f64> = Default::default();
    for solid in &brep.solids {
        for f in &solid.faces {
            let k = match &f.surface {
                Surface::Plane { frame } if frame.z.z.abs() >= 1.0 - AXIS_TOL => {
                    if (frame.z.z > 0.0) != f.same_sense {
                        down.insert(f.id, frame.o.z);
                    }
                    0
                }
                s if is_envelope(s) => 1,
                _ => 2,
            };
            kind.insert(f.id, k);
        }
    }
    let mut rep = PrismReport { solids: brep.solids.len(), tolerance_mm: brep.tolerance_mm, ..Default::default() };
    // The envelopes first: each stands as a region at its top.
    let mut env: Vec<(usize, u32, Vec<Vec<(Curve, u32)>>, f64)> = Vec::new();
    let mut env_top: std::collections::HashMap<u32, f64> = Default::default();
    for (si, solid) in brep.solids.iter().enumerate() {
        for f in solid.faces.iter().filter(|f| is_envelope(&f.surface)) {
            let (loops, top) = footprint(f, tol, &mut rep)?;
            rep.envelopes += 1;
            env_top.insert(f.id, top);
            if !loops.is_empty() {
                env.push((si, f.id, loops, top));
            }
        }
    }
    // An envelope on a down-facing plane (a round-over at the bottom of a part turned
    // upside down) fills the undercut up from that plane: the height it stands from.
    let mut env_from: std::collections::HashMap<u32, f64> = Default::default();
    for solid in &brep.solids {
        for f in solid.faces.iter().filter(|f| kind.get(&f.id) == Some(&1)) {
            for e in f.bounds.iter().flat_map(|b| &b.edges) {
                if let Some(&z) = edge_faces.get(&e.edge).and_then(|l| l.iter().find(|&&g| g != f.id)).and_then(|g| down.get(g)) {
                    env_from.insert(f.id, z);
                }
            }
        }
    }
    // A curve's wall depth: how low the face across its edge reaches, a fillet counting
    // as a region at its top. Across a face standing as high, the curve is inside the
    // section and drops out. An envelope's own curve across a down-facing plane is inside
    // too (the solid goes on past it), and across a wall rising from its top it carries
    // that wall on down its side to the plane it stands on.
    let across = |face: u32, edge: u32| -> f64 {
        let Some(g) = edge_faces.get(&edge).and_then(|l| l.iter().find(|&&g| g != face)) else { return f64::NAN };
        if let Some(&top) = env_top.get(&face) {
            if down.contains_key(g) {
                return top;
            }
            if let Some(&from) = env_from.get(&face) {
                if kind.get(g) == Some(&2) && face_zmin.get(g).is_some_and(|&z| z >= top - tol) {
                    return from;
                }
            }
        }
        match kind.get(g) {
            Some(&1) => env_top.get(g).copied().unwrap_or(f64::NAN),
            _ => face_zmin.get(g).copied().unwrap_or(f64::NAN),
        }
    };
    for (si, solid) in brep.solids.iter().enumerate() {
        for f in &solid.faces {
            if is_envelope(&f.surface) {
                continue;
            }
            let nz = match &f.surface {
                Surface::Plane { frame } => frame.z.z,
                Surface::Cylinder { frame, .. } => {
                    if frame.z.z.abs() < 1.0 - AXIS_TOL {
                        return Err(format!("face #{}: cylinder axis is not vertical (not prismatic)", f.id));
                    }
                    rep.vertical_cylinders += 1;
                    continue;
                }
                Surface::Extrusion { dir, .. } => {
                    if dir.unit().z.abs() < 1.0 - AXIS_TOL {
                        return Err(format!("face #{}: extrusion is not along z (not prismatic)", f.id));
                    }
                    rep.vertical_freeform += 1;
                    continue;
                }
                Surface::BSpline(s) => {
                    if !vertical_bspline(s, tol) {
                        return Err(format!("face #{}: B-spline surface is not a vertical wall (not prismatic)", f.id));
                    }
                    rep.vertical_freeform += 1;
                    continue;
                }
                other => return Err(format!("face #{}: {} is not prismatic", f.id, surface_name(other))),
            };
            if nz.abs() < AXIS_TOL {
                rep.vertical_planes += 1;
                continue;
            }
            if nz.abs() < 1.0 - AXIS_TOL {
                return Err(format!("face #{}: plane tilted {:.4} deg from horizontal (not prismatic)", f.id, nz.abs().acos().to_degrees()));
            }
            let Surface::Plane { frame } = &f.surface else { unreachable!() };
            let z = frame.o.z;
            let out_up = (frame.z.z > 0.0) == f.same_sense;
            let mut loops = Vec::new();
            let mut metas: Vec<Vec<f64>> = Vec::new();
            for b in &f.bounds {
                if b.vertex.is_some() {
                    return Err(format!("face #{}: point bound on a plane", f.id));
                }
                let mut lp = Vec::new();
                let mut lm: Vec<f64> = Vec::new();
                for e in &b.edges {
                    let zlo = across(f.id, e.edge);
                    for p in [e.start, e.end] {
                        rep.max_vertex_off = rep.max_vertex_off.max((p.z - z).abs());
                    }
                    let (s, t) = (V2::new(e.start.x, e.start.y), V2::new(e.end.x, e.end.y));
                    match &e.curve {
                        Curve3::Line { .. } => {
                            rep.lines += 1;
                            lp.push(Curve::Line { a: s, b: t });
                            lm.push(zlo);
                        }
                        Curve3::Circle { frame: cf, r } => {
                            if cf.z.z.abs() < 1.0 - AXIS_TOL {
                                return Err(format!("face #{}: arc edge #{} not in a horizontal plane", f.id, e.edge));
                            }
                            let ctr = V2::new(cf.o.x, cf.o.y);
                            for p in [s, t] {
                                rep.max_vertex_off = rep.max_vertex_off.max(((p - ctr).len() - r).abs());
                            }
                            let ccw = e.forward == (cf.z.z > 0.0);
                            let (u, v) = (s - ctr, t - ctr);
                            let ang = u.cross(v).atan2(u.dot(v));
                            let full = e.start_vertex == e.end_vertex;
                            let two_pi = 2.0 * std::f64::consts::PI;
                            let sweep = match (ccw, full) {
                                (true, true) => two_pi,
                                (false, true) => -two_pi,
                                (true, false) => {
                                    if ang > 0.0 {
                                        ang
                                    } else {
                                        ang + two_pi
                                    }
                                }
                                (false, false) => {
                                    if ang < 0.0 {
                                        ang
                                    } else {
                                        ang - two_pi
                                    }
                                }
                            };
                            rep.arcs += 1;
                            lp.push(Curve::Arc { c: ctr, r: *r, a: s, sweep });
                            lm.push(zlo);
                        }
                        Curve3::BSpline(bc) => {
                            for q in &bc.ctrl {
                                if (q.z - z).abs() > tol {
                                    return Err(format!("face #{}: B-spline edge #{} leaves the face's plane", f.id, e.edge));
                                }
                            }
                            let nb = Nurbs2::new(bc.degree, bc.ctrl.iter().map(|q| V2::new(q.x, q.y)).collect(), bc.weights.clone(), bc.knots.clone())
                                .map_err(|m| format!("face #{}: edge #{}: {m}", f.id, e.edge))?;
                            let (d0, d1) = nb.domain();
                            let (ta, tb) = if e.start_vertex == e.end_vertex {
                                if e.forward {
                                    (d0, d1)
                                } else {
                                    (d1, d0)
                                }
                            } else {
                                // Parameters of the edge's vertices on the full curve.
                                let full = Spline2::new(nb.clone(), d0, d1);
                                let hs = full.closest(s, f64::INFINITY).unwrap();
                                let ht = full.closest(t, f64::INFINITY).unwrap();
                                rep.max_vertex_off = rep.max_vertex_off.max(hs.dist).max(ht.dist);
                                if (hs.t < ht.t) != e.forward {
                                    return Err(format!("face #{}: B-spline edge #{} runs against its sense (seam on a closed curve?)", f.id, e.edge));
                                }
                                (hs.t, ht.t)
                            };
                            let sp = Spline2::new(nb, ta, tb);
                            for part_sp in sp.split_at_kinks(KINK) {
                                rep.splines += 1;
                                rep.spline_pieces += part_sp.pieces.len();
                                lp.push(Curve::Spline(Arc::new(part_sp)));
                                lm.push(zlo);
                            }
                        }
                        other => {
                            return Err(format!("face #{}: edge #{} is a {} (not a line, arc or B-spline)", f.id, e.edge, curve_name(other)))
                        }
                    }
                }
                // STEP bounds run counterclockwise seen from the outward normal; we want
                // the region on the left seen from +z.
                if !out_up {
                    lp = lp.into_iter().rev().map(reverse).collect();
                    lm.reverse();
                }
                let (snap, gap) = watertight(&mut lp);
                if snap > tol || gap > tol {
                    return Err(format!(
                        "face #{}: a junction is {:.2e} mm off its curves, beyond the file's tolerance {:.1e} mm",
                        f.id,
                        snap.max(gap),
                        brep.tolerance_mm
                    ));
                }
                rep.max_snap = rep.max_snap.max(snap);
                rep.max_gap = rep.max_gap.max(gap);
                loops.push(lp);
                metas.push(lm);
            }
            let first_edge = part.edges.len();
            part.add_face(f.id, z, loops, None, tol).map_err(|e| format!("solid #{}: {e}", solid.id))?;
            for (e, zlo) in part.edges[first_edge..].iter_mut().zip(metas.iter().flatten()) {
                e.wall_zlo = *zlo;
            }
            let last = part.faces.last_mut().unwrap();
            last.up = Some(out_up);
            last.solid = si as u32;
            rep.horizontal += 1;
        }
    }
    for si in 0..brep.solids.len() as u32 {
        let fs: Vec<&crate::part::Face> = part.faces.iter().filter(|f| f.solid == si).collect();
        let bottom = fs.iter().map(|f| f.z).fold(f64::INFINITY, f64::min);
        rep.overhangs += fs.iter().filter(|f| f.up == Some(false) && f.z > bottom + tol).count();
    }
    // The envelopes as regions at their tops.
    let mut tops: Vec<f64> = Vec::new();
    for (si, id, loops, top) in &env {
        let zlos: Vec<f64> = loops.iter().flatten().map(|(_, e)| across(*id, *e)).collect();
        let curves: Vec<Vec<Curve>> = loops.iter().map(|l| l.iter().map(|(c, _)| c.clone()).collect()).collect();
        let first_edge = part.edges.len();
        part.add_face(*id, *top, curves, None, tol).map_err(|e| format!("fillet #{id}: {e}"))?;
        for (e, z) in part.edges[first_edge..].iter_mut().zip(zlos) {
            e.wall_zlo = z;
        }
        let last = part.faces.last_mut().unwrap();
        last.up = Some(true);
        last.solid = *si as u32;
        last.envelope = true;
        if !tops.iter().any(|&t| (t - top).abs() <= tol) {
            tops.push(*top);
        }
    }
    rep.envelope_levels = tops.len();
    Ok((part, rep))
}

/// Makes a loop watertight on its exact curves. A tolerant B-rep may place a
/// vertex up to its declared tolerance off a curve. B-spline ends are exact and
/// never move: a junction next to a spline is the spline's end. Otherwise a
/// junction is moved onto the circle(s) it touches (the mean of the two
/// projections where two arcs meet). Lines take the junctions as ends; arcs
/// start on their circle and sweep to the junction's angle. Returns the largest
/// move of a line or arc end, and the largest gap left where an arc meets a spline.
fn watertight(lp: &mut [Curve]) -> (f64, f64) {
    let n = lp.len();
    let onto = |v: V2, c: V2, r: f64| {
        let d = v - c;
        let l = d.len();
        if l > 0.0 {
            c + d * (r / l)
        } else {
            v
        }
    };
    let circle = |c: &Curve| match c {
        Curve::Arc { c, r, .. } => Some((*c, *r)),
        _ => None,
    };
    let junction: Vec<V2> = (0..n)
        .map(|i| {
            let (prev, next) = (&lp[(i + n - 1) % n], &lp[i]);
            match (prev, next) {
                (Curve::Spline(a), Curve::Spline(b)) => (a.end() + b.start()) * 0.5,
                (Curve::Spline(a), _) => a.end(),
                (_, Curve::Spline(b)) => b.start(),
                _ => {
                    let v = next.start();
                    match (circle(prev), circle(next)) {
                        (Some((c1, r1)), Some((c2, r2))) => (onto(v, c1, r1) + onto(v, c2, r2)) * 0.5,
                        (Some((c1, r1)), None) => onto(v, c1, r1),
                        (None, Some((c2, r2))) => onto(v, c2, r2),
                        (None, None) => v,
                    }
                }
            }
        })
        .collect();
    let mut moved: f64 = 0.0;
    for i in 0..n {
        let (a, b) = (junction[i], junction[(i + 1) % n]);
        let new = match &lp[i] {
            Curve::Line { a: a0, b: b0 } => {
                moved = moved.max((*a0 - a).len()).max((*b0 - b).len());
                Curve::Line { a, b }
            }
            Curve::Arc { c, r, a: a0, sweep } => {
                let (c, r, sweep) = (*c, *r, *sweep);
                let (a, b) = (onto(a, c, r), onto(b, c, r));
                moved = moved.max((*a0 - a).len());
                let full = (sweep.abs() - 2.0 * std::f64::consts::PI).abs() < 1e-9;
                let new_sweep = if full {
                    sweep
                } else {
                    let (u, v) = (a - c, b - c);
                    let d = u.cross(v).atan2(u.dot(v));
                    let turns = ((sweep - d) / (2.0 * std::f64::consts::PI)).round();
                    d + turns * 2.0 * std::f64::consts::PI
                };
                Curve::Arc { c, r, a, sweep: new_sweep }
            }
            Curve::Spline(s) => Curve::Spline(s.clone()),
        };
        lp[i] = new;
    }
    let gap = (0..n).map(|i| (lp[i].end() - lp[(i + 1) % n].start()).len()).fold(0.0, f64::max);
    (moved, gap)
}

pub fn surface_name(s: &Surface) -> String {
    match s {
        Surface::Plane { .. } => "plane".into(),
        Surface::Cylinder { .. } => "cylinder".into(),
        Surface::Cone { .. } => "cone".into(),
        Surface::Sphere { .. } => "sphere".into(),
        Surface::Torus { .. } => "torus".into(),
        Surface::BSpline(_) => "B-spline surface".into(),
        Surface::Extrusion { .. } => "extrusion surface".into(),
        Surface::Revolution { .. } => "revolution surface".into(),
        Surface::Other(n) => n.to_lowercase(),
    }
}

pub fn curve_name(c: &Curve3) -> String {
    match c {
        Curve3::Line { .. } => "line".into(),
        Curve3::Circle { .. } => "circle".into(),
        Curve3::Ellipse { .. } => "ellipse".into(),
        Curve3::Parabola { .. } => "parabola".into(),
        Curve3::Hyperbola { .. } => "hyperbola".into(),
        Curve3::BSpline(_) => "B-spline curve".into(),
        Curve3::Other(n) => n.to_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::{BFace, Bound, Frame, OEdge, Solid};

    fn v(x: f64, y: f64, z: f64) -> V3 {
        V3::new(x, y, z)
    }

    fn line(edge: u32, a: V3, b: V3, va: u32, vb: u32) -> OEdge {
        OEdge { edge, start: a, end: b, start_vertex: va, end_vertex: vb, curve: Curve3::Line { p: a, dir: v(b.x - a.x, b.y - a.y, b.z - a.z) }, forward: true }
    }

    /// A full horizontal circle about (20, 20) at height z, run counterclockwise seen
    /// from above when `ccw`.
    fn ring(edge: u32, r: f64, z: f64, vtx: u32, ccw: bool) -> OEdge {
        let p = v(20.0 + r, 20.0, z);
        let frame = Frame { o: v(20.0, 20.0, z), z: v(0.0, 0.0, 1.0), x: v(1.0, 0.0, 0.0) };
        OEdge { edge, start: p, end: p, start_vertex: vtx, end_vertex: vtx, curve: Curve3::Circle { frame, r }, forward: ccw }
    }

    fn bound(edges: Vec<OEdge>, outer: bool) -> Bound {
        Bound { outer, edges, vertex: None }
    }

    /// A 40 x 40 x 10 plate with a hole of radius 5 at (20, 20) whose top edge is rounded
    /// over with radius 1 (a torus from radius 6 on the top face down to the wall at z 9).
    fn plate() -> Brep {
        let up = Frame { o: v(0.0, 0.0, 10.0), z: v(0.0, 0.0, 1.0), x: v(1.0, 0.0, 0.0) };
        let down = Frame { o: v(0.0, 0.0, 0.0), z: v(0.0, 0.0, -1.0), x: v(1.0, 0.0, 0.0) };
        let c = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)];
        let sq = |z: f64, base: u32, ccw: bool| -> Vec<OEdge> {
            let mut e: Vec<OEdge> = (0..4).map(|k| line(base + k as u32, v(c[k].0, c[k].1, z), v(c[(k + 1) % 4].0, c[(k + 1) % 4].1, z), base + k as u32, base + ((k + 1) % 4) as u32)).collect();
            if !ccw {
                e = e.into_iter().rev().map(|mut o| {
                    std::mem::swap(&mut o.start, &mut o.end);
                    std::mem::swap(&mut o.start_vertex, &mut o.end_vertex);
                    o.forward = false;
                    o
                }).collect();
            }
            e
        };
        let mut faces = vec![
            BFace { id: 1, surface: Surface::Plane { frame: up }, same_sense: true, bounds: vec![bound(sq(10.0, 10, true), true), bound(vec![ring(1, 6.0, 10.0, 1, false)], false)] },
            BFace {
                id: 2,
                surface: Surface::Torus { frame: Frame { o: v(20.0, 20.0, 9.0), z: v(0.0, 0.0, 1.0), x: v(1.0, 0.0, 0.0) }, major: 6.0, minor: 1.0 },
                same_sense: true,
                bounds: vec![bound(vec![ring(1, 6.0, 10.0, 1, true)], true), bound(vec![ring(2, 5.0, 9.0, 2, false)], false)],
            },
            BFace {
                id: 3,
                surface: Surface::Cylinder { frame: Frame { o: v(20.0, 20.0, 0.0), z: v(0.0, 0.0, 1.0), x: v(1.0, 0.0, 0.0) }, r: 5.0 },
                same_sense: false,
                bounds: vec![bound(vec![ring(2, 5.0, 9.0, 2, true)], true), bound(vec![ring(3, 5.0, 0.0, 3, false)], false)],
            },
            BFace { id: 4, surface: Surface::Plane { frame: down }, same_sense: true, bounds: vec![bound(sq(0.0, 20, false), true), bound(vec![ring(3, 5.0, 0.0, 3, true)], false)] },
        ];
        // The four sides, each between a top and a bottom square edge.
        for k in 0..4u32 {
            let (a, b) = (c[k as usize], c[((k + 1) % 4) as usize]);
            let n = v(b.1 - a.1, a.0 - b.0, 0.0).unit();
            let frame = Frame { o: v(a.0, a.1, 0.0), z: n, x: v(b.0 - a.0, b.1 - a.1, 0.0).unit() };
            let e = vec![
                line(20 + k, v(a.0, a.1, 0.0), v(b.0, b.1, 0.0), 0, 0),
                line(40 + k, v(b.0, b.1, 0.0), v(b.0, b.1, 10.0), 0, 0),
                line(10 + k, v(b.0, b.1, 10.0), v(a.0, a.1, 10.0), 0, 0),
                line(40 + (k + 3) % 4, v(a.0, a.1, 10.0), v(a.0, a.1, 0.0), 0, 0),
            ];
            faces.push(BFace { id: 5 + k, surface: Surface::Plane { frame }, same_sense: true, bounds: vec![bound(e, true)] });
        }
        Brep { solids: vec![Solid { id: 1, name: "plate".into(), faces }], unit_mm: 1.0, unit_name: "mm".into(), tolerance_mm: 1e-6 }
    }

    #[test]
    fn a_round_over_stands_as_solid_and_its_hole_stays_open() {
        let (part, rep) = from_brep(&plate()).unwrap();
        assert_eq!((rep.horizontal, rep.envelopes, rep.envelope_levels), (2, 1, 1));
        let frep = crate::frep::compile_prismatic(&part, 1e-6).unwrap();
        assert_eq!(*frep.levels, vec![10.0], "an envelope's top is not a floor");
        let at = |r: f64, z: f64| frep.eval([20.0 + r, 20.0, z]).value;
        // The hole is open at every depth, out to its wall at radius 5.
        for z in [0.5, 5.0, 8.9, 9.5, 9.95] {
            assert!(at(4.5, z) > 0.4, "hole at z {z}: {}", at(4.5, z));
        }
        // Under the round-over (radius 5 to 6, z 9 to 10) the envelope keeps the sharp edge.
        assert!(at(5.5, 9.8) < 0.0 && at(5.5, 5.0) < 0.0 && at(15.0, 5.0) < 0.0);
        assert!(at(5.5, 10.2) > 0.0);
    }

    #[test]
    fn a_placed_part_is_the_same_part_turned() {
        let rp = ReadPart { brep: plate() };
        let place = Place::new(1, true, 100.0, 50.0, 10.0);
        let (lo, hi) = rp.bounds(&place);
        // A quarter turn after flipping: x from -y (40 wide), y from -x... all within 1e-12.
        assert!((hi[0] - lo[0] - 40.0).abs() < 1e-12 && (hi[2] - lo[2] - 10.0).abs() < 1e-12 && lo[2].abs() < 1e-12);
        let (frep, _, rep) = rp.compile(&place).unwrap();
        // Upside down, the plate's top is the old bottom; the round-over is at the bottom
        // now and the hole still goes through.
        assert_eq!(rep.envelopes, 1);
        let c = place.point(v(20.0, 20.0, 5.0));
        assert!(frep.eval([c.x, c.y, c.z]).value > 0.4);
        assert!(frep.eval([c.x + 7.0, c.y, c.z]).value < 0.0);
        // Under the round-over, now at the bottom, the hole keeps its wall (the envelope
        // fills the undercut), and every section's curves face the side its material is on:
        // the curve grid (signs from the curves' normals) reads what the exact distance
        // (signs from crossing parity) reads.
        let low = place.point(v(20.0, 20.0, 9.5));
        assert!(frep.eval([low.x + 4.5, low.y, low.z]).value > 0.4, "the hole is open under the round-over");
        assert!(frep.eval([low.x + 5.5, low.y, low.z]).value < 0.0, "the undercut is filled");
        for (k, sec) in frep.sections.iter().enumerate() {
            let grid = crate::curvegrid::CurveGrid::new(sec, 0);
            for i in 0..60 {
                for j in 0..60 {
                    let q = V2::new(lo[0] - 5.0 + i as f64, lo[1] - 5.0 + j as f64);
                    let (a, b) = (grid.sd(sec, q).sd, crate::query::face_sd(sec, 0, q).sd);
                    assert!((a - b).abs() < 1e-9, "section {k} at {q:?}: grid {a}, exact {b}");
                }
            }
        }
    }
}
