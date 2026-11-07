//! icam-assign: assigning material to tools.
//!
//! For every point of each pocket: how long would each available tool take to remove it
//! (reach, removal rate inside that tool's physical limits, local channel width)? Give
//! each point to the cheapest tool, and compare with ordinary rest roughing ("big tool
//! first, the rest to the next smaller tool that reaches") under the SAME model.
//!
//!   grid    the raster, its exact distance transform and components
//!   vendor  the wide-cut envelope from a vendor cut-data table
//!   tools   one cut model per end mill, with its feed table
//!   cost    s/mm2 per tool by channel width and column height
//!   layer   one pocket at one floor: material, reach and channel width per tool
//!   assign  rest roughing, the cheapest tool per point, and tool changes paid for

pub mod assign;
pub mod cost;
pub mod grid;
pub mod layer;
pub mod tools;
pub mod vendor;

pub use assign::{Penalties, Plan};
pub use cost::CostTable;
pub use grid::Grid;
pub use layer::{Layer, LayerInput};
pub use tools::{ModelOptions, ToolSpec};
