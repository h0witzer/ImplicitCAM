//! Field-based roughing, end to end: a prismatic part's compiled F-rep in, a posted,
//! measured and checked program out. The operations are found from the field itself
//! ([`ops`]): no other CAM program says where the pockets are. [`run`] plans, measures
//! and posts each op on the stock the ones before it leave, then reads the program back
//! and checks it against the exact stock; [`report`] writes the page.
//!
//! On the same field-found pockets: [`finish`] runs the finishing-allowance study
//! (icam-finish) and [`assign`] assigns material to tools (icam-assign).
//!
//! The command line is `icam-rough` (src/main.rs).

pub mod assign;
pub mod assign_report;
pub mod bench;
pub mod bench_report;
pub mod finish;
pub mod finish_report;
pub mod json;
pub mod ops;
pub mod report;
pub mod run;

pub use run::{run, Job, JobOut, OpOut, PlanKind};
