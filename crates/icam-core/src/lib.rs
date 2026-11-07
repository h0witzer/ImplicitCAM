//! ImplicitCAM core: exact geometry (B-splines, curves, surfaces, solids), STEP import, the
//! exact evaluator `solid3::Exact` with its GPU kernel, and exact tool profiles. `frep`,
//! `part`, `prismatic` and `query` are the prismatic part model and its reference contact
//! query, being retired (roadmap milestone 0).

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
