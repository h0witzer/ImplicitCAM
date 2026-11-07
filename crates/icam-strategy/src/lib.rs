//! Roughing strategies read off solved plan fields. A [`plan::Plan`] solves the
//! clearing-schedule field T for one pocket at one slab from the clearance field
//! (`icam_query`); [`maxcut`] and [`spiral`] read the toolpath off it; [`physplan`] and
//! [`objective`] shape the stepover from the physical cut model (`icam_motion`);
//! [`continuity`] measures how far a path moves when its input moves; [`verify`] checks
//! a program against the exact stock.
//!
//! The seven plans, each a solve plus a path builder:
//!   maxcut      solve_eikonal                      -> maxcut::build at a fixed stepover
//!   physics     PhysPlan::solve (eikonal at s*)    -> maxcut::build at s_ref
//!   harmonic    solve                              -> spiral::build
//!   ratio       solve_ratio                        -> spiral::build
//!   conductive  solve_conductive                   -> spiral::build
//!   walls       solve_conductive + PhysPlan::for_plan -> spiral::build spaced by s*
//!   objective   solve_conductive + Objective::optimize -> spiral::build spaced by s*

pub mod continuity;
pub mod entry;
pub mod maxcut;
pub mod objective;
pub mod path;
pub mod physplan;
pub mod plan;
pub mod spiral;
pub mod verify;

pub use objective::Objective;
pub use path::{PathPt, PathResult, Tag};
pub use physplan::{PhysOptions, PhysPlan, Prior};
pub use plan::{Plan, Seed};
