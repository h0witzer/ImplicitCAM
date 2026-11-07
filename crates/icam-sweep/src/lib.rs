//! The stock as the box less every tool sweep, for any machine that places a tool in 3
//! space.
//!
//! Nothing here knows a machine. A move is the tool frame's placement in the work frame
//! as a function of t in [0, 1] ([`Motion`]), with bounds on how fast that frame moves
//! and turns over any stretch of t. Machines turn their axis values into that placement:
//! [`Chain`] is any serial machine (fixed offsets, rotary and linear axes in any order,
//! table side or head side); anything else implements [`Machine`] or [`Motion`] itself.
//!
//! The fields:
//!   tool    [`ToolSolid`]: the cutter as a solid of revolution about its axis, cut off
//!           at its stick-out; its exact signed distance is the distance in its profile's
//!           half-plane
//!   sweep   [`Sweep`]: the tool over one move, s(p) = min over t of the tool's distance
//!           at pose t, found by branch and bound on the move's rate bounds, so the answer
//!           is certified to a tolerance (nothing is sampled)
//!   stock   [`Stock`]: F = max(box, -min over sweeps), negative inside the material;
//!           sweeps are applied one at a time at runtime and found through a forest of
//!           bounding-box trees
//!
//! Min and max of exact distances bound the true distance from below (|F| never exceeds
//! it) and are exact wherever one term is active: the bound field of the two-field
//! discipline.
//!
//! [`program`] reads a 3-axis program into sweeps (a vertical tool, G17 arcs and
//! helices, canned drills); other flavors of program only need to produce motions.

pub mod bake;
pub mod cache;
pub mod envelope;
mod geom;
pub mod gpu;
mod hull;
mod motion;
pub mod program;
pub mod skin;
mod stock;
mod sweep;
mod tool;

pub use geom::{cross, dot, scale, unit, Aabb, Pose, Rot, V3};
pub use motion::{Chain, Helix, Joint, Linear, Link, Machine, Motion};
pub use skin::{Patch, Skin};
pub use stock::Stock;
pub use sweep::Sweep;
pub use tool::ToolSolid;
