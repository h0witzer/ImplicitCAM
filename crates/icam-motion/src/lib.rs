//! Motion and feed: what a tool may do in a material on a machine. The physical cut
//! model ([`CutModel`]) that the stepover plans read, and the instrument ([`analyze`])
//! that measures any path's engagement against the stock, schedules its feed inside the
//! load limits and times it.

pub mod analyze;
mod complex;
mod cut;

pub use analyze::{LoadParams, Metrics, Sample};
pub use cut::{CutModel, Limit};
