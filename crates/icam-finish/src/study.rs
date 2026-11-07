//! One pocket end to end: measure the finisher over constant allowance (a), over the
//! thinnest uniform allowance (the control that bounds every field from below), solve the
//! fields (b), trace time against stock for both families, and estimate what the
//! allowance change does to roughing.
//!
//! Three feed programs are scored, each at the same tolerance:
//!   per-sample  a feed per block
//!   one feed    one F for the pass, set by its worst spot
//!   wall feed   one F on the wall, the corner zones slowed to their own safe feed
//! Corner zones: samples whose leading edge meets rougher leftovers, i.e. whose engagement
//! at the thinnest uniform allowance is over twice its median.

use std::f64::consts::PI;
use std::time::Instant;

use icam_motion::CutModel;

use crate::field::{Field, FieldResult};
use crate::finisher::{kinematic, schedule, thickness, FeedProgram, Run};
use crate::pocket::Pocket;
use crate::stock::RoughStock;
use crate::{tools, Settings};

pub struct PocketResult {
    pub model: CutModel,
    pub rough: CutModel,
    /// The engagement tables at the uniform levels, the area each removes, simulations run.
    pub levels: Vec<f64>,
    pub tab: Vec<Vec<f64>>,
    pub areas: Vec<f64>,
    pub sims: usize,
    /// The constant allowance of (a), mm.
    pub da: f64,
    pub corner: Vec<bool>,
    pub a_sched: Run,
    pub a_one: Run,
    pub a_wall: Run,
    pub thin_sched: Run,
    pub thin_one: Run,
    pub thin_wall: Run,
    pub thick_a: Vec<f64>,
    pub thick_fast: Vec<f64>,
    pub thick_eq: Vec<f64>,
    pub thick_wall: Vec<f64>,
    pub area_a: f64,
    pub area_thin: f64,
    pub fast: FieldResult,
    pub eq: FieldResult,
    pub wall: FieldResult,
    /// (a)'s engagement at (b) fast's feed: how far over the tolerance.
    pub defl_a_at_fast: Vec<f64>,
    pub uniform: Vec<(f64, f64, f64)>,
    pub budget: Vec<(f64, f64)>,
    /// The rougher's steady removal at its knee, mm3/s.
    pub rough_mrr: f64,
    pub ms: f64,
    pub notes: Vec<String>,
}

impl PocketResult {
    /// Roughing impact of a field vs (a): extra volume the rougher removes (mm3) and its
    /// time at steady removal (s).
    pub fn rougher(&self, pk: &Pocket, f: &FieldResult) -> (f64, f64) {
        let dv = (self.area_a - f.area) * pk.ap;
        (dv, dv / self.rough_mrr)
    }
}

pub fn run(pk: &Pocket, s: &Settings) -> Result<PocketResult, String> {
    let clock = Instant::now();
    let m = tools::finisher(&pk.finish_tool, pk.ap, s)?;
    let rm = tools::rougher(&pk.rough_tool, pk.ap, s)?;
    let mut field = Field::new(pk, &m, &s.levels, s.slope);
    let mut notes = pk.notes.clone();
    if pk.ap > m.loc {
        notes.push(format!("wall stock height {:.1} mm exceeds the {:.1} mm flute length; the model is taken at the flute length", pk.ap, m.loc));
    }
    let rough_mrr = rm.knee(0.75 * rm.d, 0.03).1;

    let ka = field.level_index(s.allowance).ok_or(format!("allowance {} mm is not one of the table levels", s.allowance))?;
    let phi_a = field.tab[ka].clone();
    let phi_t = field.tab[0].clone();
    let mut cut: Vec<f64> = phi_t.iter().copied().filter(|&p| p > 0.2 * PI / 180.0).collect();
    cut.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = cut.get(cut.len() / 2).copied().unwrap_or(0.0);
    let corner: Vec<bool> = phi_t.iter().map(|&p| p > 2.0 * med).collect();
    let sch = |phi: &[f64], mode: FeedProgram, label: &str| schedule(pk, &m, phi, mode, label, Some(&corner), s.accel);

    let a_sched = sch(&phi_a, FeedProgram::PerSample, "(a) per-sample feed");
    let a_one = sch(&phi_a, FeedProgram::OneFeed, "(a) one feed");
    let a_wall = sch(&phi_a, FeedProgram::WallFeed, "(a) wall feed + corners");
    let thin_sched = sch(&phi_t, FeedProgram::PerSample, "thin per-sample feed");
    let thin_one = sch(&phi_t, FeedProgram::OneFeed, "thin one feed");
    let thin_wall = sch(&phi_t, FeedProgram::WallFeed, "thin wall feed + corners");
    let (area_a, area_thin) = (field.areas[ka], field.areas[0]);
    let uniform: Vec<(f64, f64, f64)> = (0..s.levels.len()).map(|k| (field.areas[k], sch(&field.tab[k], FeedProgram::PerSample, "").time, s.levels[k])).collect();

    // (b) fastest: the per-sample profile no field can beat (uniform d_min), carrying all the stock it can.
    // A solved field that runs slower than the uniform allowance on its true schedule
    // gives way to it (the uniform field is one the solver could have chosen).
    let keep_uniform = |f: FieldResult, uniform: &Run| -> FieldResult {
        if f.sched().time <= uniform.time {
            return f;
        }
        let field_time = f.sched().time;
        let mut trace = f.trace;
        trace.push(format!("the uniform {:.3} mm allowance runs {:.2} s against the field's {field_time:.2} s: kept the uniform allowance", s.allowance, uniform.time));
        FieldResult {
            label: f.label,
            delta: vec![s.allowance; pk.n()],
            phi: phi_a.clone(),
            corr: f.corr,
            stock: RoughStock::uniform(pk, s.allowance),
            area: area_a,
            lambda: f64::NAN,
            sched: Some(Run { label: uniform.label.clone(), ..uniform.clone() }),
            trace,
        }
    };
    let fast = field.solve_target("(b) fastest, per-sample", &thin_sched.v, 5, &|phi| sch(phi, FeedProgram::PerSample, "(b) fastest, per-sample"), None);
    let fast = keep_uniform(fast, &a_sched);
    // (b) equal stock: minimum time for (a)'s stock volume (same roughing, redistributed).
    let v_geo = kinematic(pk, &vec![m.v_max; pk.n()], s.accel);
    let eq = field.solve_budget("(b) equal stock, per-sample", area_a, &v_geo, &fast.corr, 3, &|phi| sch(phi, FeedProgram::PerSample, "(b) equal stock, per-sample"));
    let eq = keep_uniform(eq, &a_sched);
    // (b) for a wall-feed program: hold the best wall feed any field allows, as much stock
    // as it can carry. Target: the thin wall program's own run feed, the corners at their
    // thin ceilings.
    let tw: Vec<f64> = (0..pk.n()).map(|i| if corner[i] { thin_sched.v_ceil[i] } else { thin_wall.v[i] }).collect();
    let wall = field.solve_target("(b) wall feed + corners", &tw, 5, &|phi| sch(phi, FeedProgram::WallFeed, "(b) wall feed + corners"), Some(&fast.corr));
    let wall = keep_uniform(wall, &a_wall);

    // Time against stock for fields at several budgets (one corrected round each).
    let top = (field.areas[field.areas.len() - 1] * 0.8).max(area_a * 1.5);
    let mut budget = Vec::new();
    for f in [0.1, 0.25, 0.45, 0.7, 1.0] {
        let a = area_thin + f * (top - area_thin);
        let r = field.solve_budget("", a, &v_geo, &eq.corr, 1, &|phi| sch(phi, FeedProgram::PerSample, ""));
        budget.push((r.area, r.sched().time));
    }
    budget.push((eq.area, eq.sched().time));
    budget.push((fast.area, fast.sched().time));
    budget.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));

    let defl_a_at_fast: Vec<f64> =
        (0..pk.n()).map(|i| if phi_a[i] < 0.2 * PI / 180.0 { 0.0 } else { m.deflection(phi_a[i].min(PI), fast.sched().v[i].max(1e-3)) }).collect();
    let thick_a = thickness(pk, &RoughStock::uniform(pk, s.allowance));
    let thick_fast = thickness(pk, &fast.stock);
    let thick_eq = thickness(pk, &eq.stock);
    let thick_wall = thickness(pk, &wall.stock);
    let (tab, areas, sims) = (std::mem::take(&mut field.tab), std::mem::take(&mut field.areas), field.sims);
    Ok(PocketResult {
        model: m.clone(),
        rough: rm,
        levels: s.levels.clone(),
        tab,
        areas,
        sims,
        da: s.allowance,
        corner,
        a_sched,
        a_one,
        a_wall,
        thin_sched,
        thin_one,
        thin_wall,
        thick_a,
        thick_fast,
        thick_eq,
        thick_wall,
        area_a,
        area_thin,
        fast,
        eq,
        wall,
        defl_a_at_fast,
        uniform,
        budget,
        rough_mrr,
        ms: clock.elapsed().as_secs_f64() * 1000.0,
        notes,
    })
}
