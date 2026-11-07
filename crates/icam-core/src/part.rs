//! A prismatic part as exact horizontal faces.
//!
//! For a vertical tool coming from above, a part made of horizontal faces and
//! vertical walls is fully described by its horizontal faces: every wall's top
//! edge is the boundary of some horizontal face, and a wall point is never
//! higher than that edge at the same (x, y). Each face keeps its exact loops.

use crate::geom::{Curve, V2};

/// One boundary curve with the face it bounds and the outward pseudo-normals at
/// its two ends (the normalized sum of the outward normals of the two curves
/// meeting there). A point whose closest boundary point is a vertex is inside
/// the face exactly when it lies behind that vertex's pseudo-normal.
#[derive(Clone, Debug)]
pub struct Edge {
    pub face: u32,
    pub curve: Curve,
    pub pn_start: V2,
    pub pn_end: V2,
    /// Lowest z of the vertical wall that meets the face along this curve (NaN
    /// if unknown). A cross-section at height z has this curve on its boundary
    /// when the face is at or above z and the wall reaches below z.
    pub wall_zlo: f64,
}

#[derive(Clone, Debug)]
pub struct Face {
    /// Source face id (CAD face index in the part cache, or synthetic).
    pub id: u32,
    pub z: f64,
    /// Ranges into `Part::edges`, one per closed loop.
    pub loops: Vec<std::ops::Range<usize>>,
    /// Area reported by the source B-rep, for checking the recovered loops.
    pub source_area: Option<f64>,
    /// Whether the material lies below the face (its outward normal points up),
    /// when known.
    pub up: Option<bool>,
    /// Which solid of the source the face belongs to.
    pub solid: u32,
    /// Not a real horizontal face: a fillet standing as a region at its top (its
    /// footprint), conservative for a tool from above. Its height bounds a column but
    /// is not a floor.
    pub envelope: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Part {
    pub faces: Vec<Face>,
    pub edges: Vec<Edge>,
}

impl Part {
    /// Adds a horizontal face at height `z` from closed loops (region on the left).
    /// Consecutive curves must meet within `close_tol` mm (the source's own tolerance).
    pub fn add_face(&mut self, id: u32, z: f64, loops: Vec<Vec<Curve>>, source_area: Option<f64>, close_tol: f64) -> Result<(), String> {
        let face = self.faces.len() as u32;
        let mut ranges = Vec::new();
        for lp in loops {
            if lp.is_empty() {
                continue;
            }
            for k in 0..lp.len() {
                let (e, n) = (lp[k].end(), lp[(k + 1) % lp.len()].start());
                if (e - n).len() > close_tol {
                    return Err(format!("face {id}: loop does not close at curve {k} (gap {:.3e} mm)", (e - n).len()));
                }
            }
            let start = self.edges.len();
            for k in 0..lp.len() {
                let prev = &lp[(k + lp.len() - 1) % lp.len()];
                let next = &lp[(k + 1) % lp.len()];
                let cur = &lp[k];
                let pn_start = (prev.outward_end() + cur.outward_start()).unit();
                let pn_end = (cur.outward_end() + next.outward_start()).unit();
                self.edges.push(Edge { face, curve: cur.clone(), pn_start, pn_end, wall_zlo: f64::NAN });
            }
            ranges.push(start..self.edges.len());
        }
        self.faces.push(Face { id, z, loops: ranges, source_area, up: None, solid: 0, envelope: false });
        Ok(())
    }

    /// Area of a face from its exact loops.
    pub fn face_area(&self, f: usize) -> f64 {
        self.faces[f].loops.iter().flat_map(|r| self.edges[r.clone()].iter()).map(|e| e.curve.area_term()).sum()
    }

    pub fn face_edges(&self, f: usize) -> impl Iterator<Item = (usize, &Edge)> + '_ {
        self.faces[f].loops.iter().flat_map(move |r| r.clone().map(move |i| (i, &self.edges[i])))
    }

    /// Bounding box of all faces (conservative for arcs and splines).
    pub fn bbox(&self) -> (V2, V2) {
        let mut lo = V2::new(f64::MAX, f64::MAX);
        let mut hi = V2::new(f64::MIN, f64::MIN);
        for e in &self.edges {
            let (p, q) = e.curve.bbox();
            lo = V2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = V2::new(hi.x.max(q.x), hi.y.max(q.y));
        }
        (lo, hi)
    }

    /// Exact inside test by ray crossings over every loop of the face. Used by
    /// the reference query and by grid construction, never by the GPU kernel.
    pub fn inside(&self, f: usize, p: V2) -> bool {
        // An irrational-looking direction, so the ray almost never hits a vertex.
        let dir = V2::new(0.822_635_643_121_35, 0.568_563_221_742_84).unit();
        let n: u32 = self.face_edges(f).map(|(_, e)| e.curve.ray_crossings(p, dir)).sum();
        n % 2 == 1
    }
}
