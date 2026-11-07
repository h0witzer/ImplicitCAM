//! icam-finish: a variable finish allowance so the finisher sees constant load.
//!
//! Roughing leaves a constant wall allowance plus whatever the rougher cannot reach, so
//! the finishing wall pass sees load spikes and is fed for them. This measures the
//! finisher's engagement, load and deflection along every wall with the cut model,
//! schedules the feed that holds a deflection tolerance, and then designs an allowance
//! field delta(x) along the wall that holds the same tolerance at a higher feed. It also
//! estimates what the field costs the rougher. (a) vs (b) is always scored by the same
//! instrument, the same model and the same tolerance.
//!
//!   pocket    the pocket at its floor: clearance grid, the finisher's wall loops, the
//!             rougher's component, the cuts before the rough op
//!   stock     the stock an ideal rougher leaves for an allowance field
//!   finisher  the instrument: engagement, thickness, removed area, feed schedules
//!   field     the allowance field, by target feed or by stock budget
//!   study     one pocket end to end
//!   tools     the finishing cutters' models

pub mod field;
pub mod finisher;
pub mod geom;
pub mod pocket;
pub mod stock;
pub mod study;
pub mod tools;

pub use field::{Field, FieldResult};
pub use finisher::{Bind, FeedProgram, Run};
pub use pocket::{Pocket, PocketInput, Walls};
pub use stock::RoughStock;
pub use study::PocketResult;

/// The study's settings, each with the value the study assumed.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Thinnest and thickest allowance the field may leave, mm.
    pub d_min: f64,
    pub d_max: f64,
    /// The uniform allowances the engagement tables are measured at, mm.
    pub levels: Vec<f64>,
    /// Field slope limit, mm of allowance per mm of wall (assumed).
    pub slope: f64,
    /// Machine acceleration, mm/s2 (assumed).
    pub accel: f64,
    /// Finishing tolerance on the tool's deflection at the cut, in (assumed: half of a
    /// +/-0.001 in finishing band is spent on the cutter's own bending).
    pub tol_in: f64,
    /// The 1/4 in 3-flute: rpm (the router's maximum), flute length in (3 x D, assumed),
    /// helix deg (assumed), hangout in (tool sheet).
    pub rpm06: f64,
    pub loc06_in: f64,
    pub helix06: f64,
    pub hangout06_in: f64,
    /// The 1/2 in tool's rpm (Amana aluminum chart).
    pub rpm04: f64,
    /// (a): the constant wall allowance, mm (0.010 in).
    pub allowance: f64,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            d_min: 0.05,
            d_max: 1.0,
            levels: vec![0.05, 0.075, 0.1, 0.15, 0.2, 0.254, 0.35, 0.5, 0.75, 1.0],
            slope: 0.05,
            accel: 2500.0,
            tol_in: 0.0005,
            rpm06: 24000.0,
            loc06_in: 0.75,
            helix06: 35.0,
            hangout06_in: 1.5,
            rpm04: 18000.0,
            allowance: 0.254,
        }
    }
}

impl Settings {
    /// The table levels start at d_min and include the (a) allowance.
    pub fn fit_levels(&mut self) {
        let sorted = |mut l: Vec<f64>| {
            l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            l
        };
        if (self.d_min - self.levels[0]).abs() > 1e-9 {
            let mut l = vec![self.d_min];
            l.extend(self.levels.iter().copied().filter(|&v| v > self.d_min + 1e-9));
            self.levels = l;
            if self.levels.iter().all(|&v| (v - self.allowance).abs() > 1e-9) {
                self.levels.push(self.allowance);
                self.levels = sorted(std::mem::take(&mut self.levels));
            }
        }
        if !self.levels.iter().any(|&v| (v - self.allowance).abs() < 1e-9) {
            self.levels.push(self.allowance);
            self.levels = sorted(std::mem::take(&mut self.levels));
        }
    }
}
