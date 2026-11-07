//! Queries against the part field for planning. Today: the clearance a vertical tool
//! sees at a slab ([`Clearance`]), read from the compiled F-rep's section leaves
//! ([`SliceClearance`]) or given directly as polygon regions ([`PolygonClearance`]).

mod clearance;

pub use clearance::{seg, seg_seg, Clearance, PolygonClearance, SliceClearance};
pub use icam_post::Box2;
