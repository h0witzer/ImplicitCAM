//! The B-rep reference for prismatic solids, by brute force over the part's faces
//! and the STEP curves under them. For tests and icam-check only (decision 3): it
//! checks the compiled prismatic field by sampling and never supplies a field value.

use crate::geom::V2;
use crate::part::Part;
use crate::query::face_sd;

/// Exact distance from p to the boundary of a prismatic solid, by brute force
/// over its faces: horizontal faces, and the vertical walls under each upward
/// face's curves (from the curve's wall depth up to the face).
pub fn exact_boundary_distance(part: &Part, p: [f64; 3]) -> f64 {
    let q = V2::new(p[0], p[1]);
    let mut best = f64::INFINITY;
    for (fi, f) in part.faces.iter().enumerate() {
        let fd = face_sd(part, fi, q);
        let dz = p[2] - f.z;
        best = best.min((fd.sd.max(0.0).powi(2) + dz * dz).sqrt());
        if f.up == Some(true) {
            for (_, e) in part.face_edges(fi) {
                let (zl, zh) = (e.wall_zlo, f.z);
                if zl.is_nan() || zl >= zh {
                    continue;
                }
                let d = e.curve.closest(q).dist;
                let out = if p[2] < zl { zl - p[2] } else if p[2] > zh { p[2] - zh } else { 0.0 };
                best = best.min((d * d + out * out).sqrt());
            }
        }
    }
    best
}

/// Point-in-solid for prismatic solids on their bottoms: for some solid, z
/// between its bottom and the highest of its upward faces containing (x, y).
pub fn inside_solid(part: &Part, p: [f64; 3]) -> bool {
    let q = V2::new(p[0], p[1]);
    let mut solids: Vec<u32> = part.faces.iter().map(|f| f.solid).collect();
    solids.sort();
    solids.dedup();
    solids.into_iter().any(|s| {
        let fs: Vec<usize> = (0..part.faces.len()).filter(|&f| part.faces[f].solid == s).collect();
        let z_bot = fs.iter().map(|&f| part.faces[f].z).fold(f64::INFINITY, f64::min);
        let top = fs
            .iter()
            .filter(|&&f| part.faces[f].up == Some(true) && part.inside(f, q))
            .map(|&f| part.faces[f].z)
            .fold(f64::NEG_INFINITY, f64::max);
        p[2] >= z_bot && p[2] <= top
    })
}
