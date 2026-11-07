//! ImplicitCAM core: exact geometry, exact tools, and the reference
//! implementation of the contact query that the GPU kernels are checked against.

pub mod bspline;
pub mod curve3;
pub mod curvegrid;
pub mod frep;
pub mod geom;
pub mod mrep;
pub mod part;
pub mod prismatic;
pub mod prismatic_reference;
pub mod quadric;
pub mod query;
pub mod solid3;
pub mod span_poly;
pub mod step;
pub mod surface3;
pub mod trim2;
pub mod tool;
