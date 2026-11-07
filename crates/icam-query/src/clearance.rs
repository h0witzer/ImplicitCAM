//! The clearance field the planner reads: for a vertical tool with its tip at the slab
//! height z, the exact XY distance from the tool center to the part's shadow above the
//! slab, negative inside the shadow.
//!
//! On a compiled prismatic part (`icam_core::frep`, version 1) the solid is
//!     f(x, y, z) = min_k max( sd_Sk(x, y),  z - z_k,  z_bot - z )
//! a union of columns, column k standing on the cross-section S_k up to its top z_k.
//! The part above the slab is the union of the columns whose top is above z, so its
//! shadow is the union of their S_k, and the distance to a union is the min of the
//! distances: the clearance is the min of those columns' section leaves sd_Sk. That is
//! exact Euclidean distance everywhere outside the part, with no band and no grid,
//! read straight off the field's own leaves (nothing is tessellated or sliced).

use icam_core::frep::{Frep, Node};
use icam_core::geom::V2;

/// Distance from (x, y) to the segment a-b.
pub fn seg(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (ux, uy) = (bx - ax, by - ay);
    let l2 = ux * ux + uy * uy;
    let t = if l2 < 1e-18 { 0.0 } else { (((x - ax) * ux + (y - ay) * uy) / l2).clamp(0.0, 1.0) };
    let (ex, ey) = (x - (ax + t * ux), y - (ay + t * uy));
    (ex * ex + ey * ey).sqrt()
}

/// Distance between segments a-b and c-d: 0 if they cross, else the least of the four
/// endpoint-to-segment distances.
#[allow(clippy::too_many_arguments)]
pub fn seg_seg(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, dx: f64, dy: f64) -> f64 {
    let o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    let o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
    let o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    let o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
    if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
        return 0.0;
    }
    seg(ax, ay, cx, cy, dx, dy).min(seg(bx, by, cx, cy, dx, dy)).min(seg(cx, cy, ax, ay, bx, by).min(seg(dx, dy, ax, ay, bx, by)))
}

/// The clearance a vertical tool sees at one slab (see the module notes).
pub trait Clearance: Sync {
    /// XY distance from a tool center at (x, y) to the part's shadow above the slab,
    /// negative inside it, never more than [`Clearance::band`].
    fn at(&self, x: f64, y: f64) -> f64;

    /// Distances past this read as this; infinite for an exact field.
    fn band(&self) -> f64 {
        f64::INFINITY
    }

    /// True iff `at >= need` at every point of the segment a-b.
    fn clear_along(&self, ax: f64, ay: f64, bx: f64, by: f64, need: f64) -> bool;

    /// Unit gradient by central differences (the direction away from the part).
    fn grad(&self, x: f64, y: f64) -> (f64, f64) {
        let h = 0.02;
        let gx = (self.at(x + h, y) - self.at(x - h, y)) / (2.0 * h);
        let gy = (self.at(x, y + h) - self.at(x, y - h)) / (2.0 * h);
        let n = (gx * gx + gy * gy).sqrt();
        if n < 1e-12 { (0.0, 0.0) } else { (gx / n, gy / n) }
    }
}

/// The clearance at one slab of a compiled prismatic part: the min of the section
/// leaves of the columns that rise above the slab.
pub struct SliceClearance<'a> {
    frep: &'a Frep,
    sections: Vec<u32>,
}

impl<'a> SliceClearance<'a> {
    /// The clearance with the tool tip at `z`. A column counts when its top is more than
    /// `eps` above the slab, so a floor the tool rests on is not a wall.
    pub fn new(frep: &'a Frep, z: f64, eps: f64) -> SliceClearance<'a> {
        let mut sections = Vec::new();
        for node in &frep.nodes {
            let Node::Max(children) = node else { continue };
            let (mut sec, mut top) = (None, None);
            for &c in children {
                match frep.nodes[c as usize] {
                    Node::Section { sec: s } => sec = Some(s),
                    Node::Below { z: zk } => top = Some(zk),
                    _ => {}
                }
            }
            if let (Some(s), Some(zk)) = (sec, top) {
                if zk > z + eps {
                    sections.push(s);
                }
            }
        }
        SliceClearance { frep, sections }
    }

    /// How many columns stand above the slab.
    pub fn column_count(&self) -> usize {
        self.sections.len()
    }
}

impl Clearance for SliceClearance<'_> {
    fn at(&self, x: f64, y: f64) -> f64 {
        let p = V2::new(x, y);
        self.sections.iter().map(|&s| self.frep.section_sd(s, p).sd).fold(f64::INFINITY, f64::min)
    }

    /// Branch and bound on the section leaves, which are 1-Lipschitz: over a stretch of
    /// half-length h around its midpoint m the clearance is at least at(m) - h. A
    /// stretch is cleared by that bound or split; one point below `need` decides
    /// against. Stretches shorter than 0.1 um left undecided count as clear.
    fn clear_along(&self, ax: f64, ay: f64, bx: f64, by: f64, need: f64) -> bool {
        if need <= 0.0 {
            return true;
        }
        if self.at(ax, ay) < need || self.at(bx, by) < need {
            return false;
        }
        let len = ((bx - ax) * (bx - ax) + (by - ay) * (by - ay)).sqrt();
        let mut stack = vec![(0.0f64, 1.0f64)];
        while let Some((t0, t1)) = stack.pop() {
            let tm = 0.5 * (t0 + t1);
            let v = self.at(ax + (bx - ax) * tm, ay + (by - ay) * tm);
            if v < need {
                return false;
            }
            let half = 0.5 * (t1 - t0) * len;
            if v - half >= need || half < 1e-7 {
                continue;
            }
            stack.push((tm, t1));
            stack.push((t0, tm));
        }
        true
    }
}

/// The clearance of a shadow given directly as polygon regions (each a list of closed
/// loops, inside by crossing parity), as for a 2D profile or a test part. Distances
/// past `band` read as `band`; inside, minus the distance to the nearest edge.
pub struct PolygonClearance {
    regions: Vec<Vec<Vec<(f64, f64)>>>,
    band: f64,
}

impl PolygonClearance {
    pub fn new(regions: Vec<Vec<Vec<(f64, f64)>>>, band: f64) -> PolygonClearance {
        PolygonClearance { regions, band }
    }

    fn inside(&self, x: f64, y: f64) -> bool {
        self.regions.iter().any(|region| {
            let mut c = false;
            for l in region {
                let mut j = l.len() - 1;
                for i in 0..l.len() {
                    if (l[i].1 > y) != (l[j].1 > y) && x < (l[j].0 - l[i].0) * (y - l[i].1) / (l[j].1 - l[i].1) + l[i].0 {
                        c = !c;
                    }
                    j = i;
                }
            }
            c
        })
    }

    fn edges(&self) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
        self.regions.iter().flatten().flat_map(|l| (0..l.len()).map(move |i| (l[i], l[(i + 1) % l.len()])))
    }
}

impl Clearance for PolygonClearance {
    fn at(&self, x: f64, y: f64) -> f64 {
        let d = self.edges().map(|(a, b)| seg(x, y, a.0, a.1, b.0, b.1)).fold(f64::INFINITY, f64::min);
        if self.inside(x, y) { -d } else { d.min(self.band) }
    }

    fn band(&self) -> f64 {
        self.band
    }

    fn clear_along(&self, ax: f64, ay: f64, bx: f64, by: f64, need: f64) -> bool {
        if need > self.band {
            return false;
        }
        if need <= 0.0 {
            return true;
        }
        if self.inside(ax, ay) || self.inside(bx, by) {
            return false;
        }
        self.edges().all(|(c, d)| seg_seg(ax, ay, bx, by, c.0, c.1, d.0, d.1) >= need)
    }
}
