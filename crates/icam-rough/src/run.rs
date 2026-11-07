//! The job, end to end. For each operation found from the field (`ops`), in nearest-ready
//! order (see `run`):
//!
//!   plan     the clearance at its slab (the compiled F-rep's section leaves), the plan
//!            field and its seeds, the chosen plan's solve and path (icam_strategy)
//!   measure  the path's engagement against the stock the program so far leaves (read
//!            back from that program, so every op stands on what the earlier ones really
//!            cut), the feed schedule inside the load limits, and its time
//!   post     the op's block, appended to the program
//!
//! then the whole program is read back once more and each op is measured again as posted
//! and checked against the exact stock (icam_stock::ImplicitStock) at design points taken
//! from the field: gouges past 0.001 in, the stock left on the walls, and the reachable
//! area left uncut. An op that gouges is named in the program's header and in the result;
//! nothing here should drive a machine.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

use icam_core::frep::Frep;
use icam_motion::analyze::{self, LoadParams, Metrics, Sample};
use icam_motion::CutModel;
use icam_post::post::{self, Dialect, PostPt, Writer};
use icam_post::{Box2, Move, MoveKind, ToolDef, V3};
use icam_query::{Clearance, SliceClearance};
use icam_stock::{ImplicitStock, SweptStock};
use icam_strategy::entry::{self, Descent, LoopAp};
use icam_strategy::maxcut::Air;
use icam_strategy::path::P;
use icam_strategy::physplan::height_before;
use icam_strategy::verify::{self, Check};
use icam_strategy::{maxcut, spiral, Objective, PathResult, PhysOptions, PhysPlan, Plan, Prior};

use crate::ops::{self, OpSpec};

/// Reachable stock an op may leave uncut, mm2.
pub const MAX_UNCUT: f64 = 5.0;
/// The steepest an entry ramps down, degrees.
pub const RAMP_DEG: f64 = 15.0;
/// Stock over a pocket thinner than this, mm, is taken as already cleared (the op is skipped).
const CLEARED_MM: f64 = 0.05;
pub(crate) const FEED_CAP: f64 = 2.0;
pub(crate) const PHI_HEAVY_DEG: f64 = 120.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanKind {
    MaxCut,
    Physics,
    Harmonic,
    Ratio,
    Conductive,
    Walls,
    Objective,
}

impl PlanKind {
    pub fn parse(s: &str) -> Option<PlanKind> {
        Some(match s {
            "maxcut" => PlanKind::MaxCut,
            "physics" => PlanKind::Physics,
            "harmonic" => PlanKind::Harmonic,
            "ratio" => PlanKind::Ratio,
            "conductive" => PlanKind::Conductive,
            "walls" => PlanKind::Walls,
            "objective" => PlanKind::Objective,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            PlanKind::MaxCut => "maxcut",
            PlanKind::Physics => "physics",
            PlanKind::Harmonic => "harmonic",
            PlanKind::Ratio => "ratio",
            PlanKind::Conductive => "conductive",
            PlanKind::Walls => "walls",
            PlanKind::Objective => "objective",
        }
    }

    /// Read off with max-cut (fronts one stepover apart) rather than the spiral.
    fn maxcut(self) -> bool {
        matches!(self, PlanKind::MaxCut | PlanKind::Physics)
    }

    /// Needs the physical cut model (the stepover comes from it).
    pub fn needs_material(self) -> bool {
        matches!(self, PlanKind::Physics | PlanKind::Walls | PlanKind::Objective)
    }
}

/// What to cut with and how.
#[derive(Clone, Debug)]
pub struct Job {
    pub tool: ToolDef,
    pub plan: PlanKind,
    /// Physical limits (al6061); required by physics, walls and objective.
    pub material: Option<String>,
    pub rpm: Option<f64>,
    /// Wall and floor stock to leave, mm.
    pub allowance: f64,
    /// Deepest slab, mm (None: one slab per floor).
    pub stepdown: Option<f64>,
    /// Stock top (None: the part's top).
    pub stock_top: Option<f64>,
    /// Plan grid pitch, mm.
    pub pitch: f64,
    /// Reference cut when there is no material: engagement (rad) and feed (mm/s).
    pub phi_ref: f64,
    pub v_prog: f64,
    /// Machine acceleration, mm/s2.
    pub accel: f64,
    /// Fastest cutting feed, as a multiple of the reference feed: where a light cut thins
    /// the chip, the feed rises to keep it, up to this.
    pub feed_cap: f64,
    /// Max-cut stepover, mm (None: from the reference or the cut model).
    pub stepover: Option<f64>,
    /// Seed significance for the spiral plans, mm (None: 1/8 of the tool radius).
    pub persist: Option<f64>,
    pub one_way: bool,
    pub greedy: bool,
    pub dialect: Dialect,
    pub title: String,
    /// Report each op's progress on standard error (long jobs).
    pub progress: bool,
    /// Plan only the pockets whose seed point lies in this box (a patch of a large part).
    pub window: Option<Box2>,
    /// The work frame's zero in part coordinates (None: the stock's corner, Z on the table).
    pub origin: Option<(f64, f64, f64)>,
    /// Bench: the reference cut (feed mm/s, engagement rad) instead of the baseline's
    /// medians (a window clips the baseline's moves, which moves its medians).
    pub reference: Option<(f64, f64)>,
}

impl Job {
    pub fn new(tool: ToolDef, plan: PlanKind) -> Job {
        Job {
            tool,
            plan,
            material: None,
            rpm: None,
            allowance: 0.254,
            stepdown: None,
            stock_top: None,
            pitch: 0.5,
            phi_ref: (1.0f64 - 0.8).acos(),
            v_prog: 34.0,
            accel: 2500.0,
            feed_cap: FEED_CAP,
            stepover: None,
            persist: None,
            one_way: false,
            greedy: false,
            dialect: Dialect::Fanuc,
            title: "IMPLICITCAM FIELD ROUGHING".into(),
            progress: false,
            window: None,
            reference: None,
            origin: None,
        }
    }
}

/// One operation's outcome.
pub struct OpOut {
    pub spec: OpSpec,
    /// Why the op was not planned (no room for the tool, no entry), if it was not.
    pub skipped: Option<String>,
    pub path: Option<PathResult>,
    /// The planned path, measured: samples, and the time at the scheduled feeds.
    pub ours: Vec<Sample>,
    pub planned: Option<Metrics>,
    /// The op as posted and read back, timed at the feeds it carries.
    pub posted: Option<Metrics>,
    pub check: Option<Check>,
    pub ap: f64,
    /// The widest entry loop radius (the posted entry turns that are still entry).
    pub rho: f64,
    pub notes: Vec<String>,
    /// The plan grid (for the report): origin, pitch, size, and the part's outline at the slab.
    pub grid: (f64, f64, f64, usize, usize),
    pub outline: Vec<Vec<(f64, f64)>>,
    /// The op's moves in the program read back ([`JobOut::moves`]), start..end.
    pub range: (usize, usize),
    pub ms: f64,
}

impl OpOut {
    pub fn gouged(&self) -> bool {
        self.check.as_ref().is_some_and(|c| c.gouges > 0)
    }
}

/// Moves filed by the box their tool sweeps, so an op reads only the moves that pass
/// near its region (in program order): exact, since nothing outside the region changes
/// the stock inside it.
struct MoveIndex {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
    boxes: Vec<Box2>,
}

impl MoveIndex {
    fn new(over: Box2, cell: f64) -> MoveIndex {
        let nx = ((over.x1 - over.x0) / cell).ceil().max(1.0) as usize + 1;
        let ny = ((over.y1 - over.y0) / cell).ceil().max(1.0) as usize + 1;
        MoveIndex { x0: over.x0, y0: over.y0, cell, nx, ny, cells: vec![Vec::new(); nx * ny], boxes: Vec::new() }
    }

    fn span(&self, b: Box2) -> (usize, usize, usize, usize) {
        let i = |x: f64| (((x - self.x0) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let j = |y: f64| (((y - self.y0) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        (i(b.x0), i(b.x1), j(b.y0), j(b.y1))
    }

    fn add(&mut self, m: &Move, r: f64) {
        let mut b = Box2::new(m.from.x.min(m.to.x), m.from.y.min(m.to.y), m.from.x.max(m.to.x), m.from.y.max(m.to.y));
        if matches!(m.kind, MoveKind::ArcCw | MoveKind::ArcCcw) {
            let ra = (m.from.x - m.center.x).hypot(m.from.y - m.center.y);
            b = Box2::new(m.center.x - ra, m.center.y - ra, m.center.x + ra, m.center.y + ra);
        }
        let b = b.grow(r);
        let k = self.boxes.len() as u32;
        let (i0, i1, j0, j1) = self.span(b);
        for j in j0..=j1 {
            for i in i0..=i1 {
                self.cells[j * self.nx + i].push(k);
            }
        }
        self.boxes.push(b);
    }

    /// Indices below `end` of the moves whose box meets `region`, ascending.
    fn near(&self, region: Box2, end: usize) -> Vec<usize> {
        let (i0, i1, j0, j1) = self.span(region);
        let mut out = Vec::new();
        for j in j0..=j1 {
            for i in i0..=i1 {
                for &k in &self.cells[j * self.nx + i] {
                    let b = self.boxes[k as usize];
                    if (k as usize) < end && b.x0 <= region.x1 && b.x1 >= region.x0 && b.y0 <= region.y1 && b.y1 >= region.y0 {
                        out.push(k as usize);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

fn subset(moves: &[Move], idx: &[usize]) -> Vec<Move> {
    idx.iter().map(|&i| moves[i].clone()).collect()
}

pub struct JobOut {
    pub ops: Vec<OpOut>,
    pub program: String,
    pub warnings: Vec<String>,
    pub stock: (V3, V3),
    /// The work frame's zero in part coordinates (the program is written relative to it).
    pub origin: (f64, f64, f64),
    /// The program read back, in part coordinates.
    pub moves: Vec<Move>,
}

fn to_post(s: &[Sample]) -> Vec<PostPt> {
    s.iter().map(|p| PostPt { x: p.x, y: p.y, z: p.z, feed: p.feed, tag: p.tag }).collect()
}

/// Programmed feed (mm/s) by source line, from the moves read back.
fn feeds_by_line(moves: &[Move]) -> Vec<f64> {
    let n = moves.iter().map(|m| m.line).max().unwrap_or(0) + 2;
    let mut f = vec![0.0; n];
    for m in moves {
        f[m.line] = m.feed / 60.0;
    }
    f
}

/// Rough the part whose compiled F-rep is `frep` and whose XY box is `part`.
pub fn run(frep: &Frep, part: Box2, job: &Job) -> Result<JobOut, String> {
    let t = &job.tool;
    let (d, r, delta) = (t.diameter, t.diameter / 2.0, job.allowance);
    let part_top = frep.levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let top = job.stock_top.unwrap_or(part_top);
    let stock = (V3::new(part.x0, part.y0, frep.z_bot), V3::new(part.x1, part.y1, top));
    let sheet = Box2::new(part.x0, part.y0, part.x1, part.y1);
    let (z_entry, clear_z) = (top + 1.0, top + 5.0);
    let tools: BTreeMap<i32, ToolDef> = BTreeMap::from([(t.number, t.clone())]);
    let base_cut = match (&job.material, job.plan.needs_material()) {
        (Some(m), _) => Some(CutModel::for_material(m, t, job.rpm)?),
        (None, true) => return Err(format!("--plan {} needs --material (the stepover comes from the physical limits)", job.plan.name())),
        (None, false) => None,
    };
    let mut warnings = Vec::new();
    if job.stock_top.is_some_and(|s| s > part_top + 1e-6) {
        warnings.push(format!("the stock top {top:.3} is above the part's top {part_top:.3}: facing is not planned"));
    }
    let rpm = job.rpm.unwrap_or(18000.0);
    let note = format!("TOOL {} D{:.3} R{:.3} {}", t.number, t.diameter, t.corner_radius, t.description);
    // The work frame: X and Y from the stock's corner, Z from the table (the stock's bottom),
    // so every height in the program is positive and a reader cannot take it for Z-down.
    let origin = job.origin.unwrap_or((stock.0.x, stock.0.y, stock.0.z));
    let mut writer = Writer::new(job.dialect, origin, &job.title, t.number, &note, rpm, clear_z);
    let ae_ref = r * (1.0 - job.phi_ref.cos());
    let ae_cap = ae_ref.max(r * (1.0 - (PHI_HEAVY_DEG * PI / 180.0).cos()));

    // The entries' stable drop per lap by loop shape: the chatter limit depends on the tool
    // and its hangout, not on the op's depth, so it is the same for every op.
    let loop_ap = base_cut.as_ref().map_or(LoopAp::uniform(f64::INFINITY), |c| entry::loop_ap(&c.at_depth(c.loc)));
    let mut specs = ops::find(frep, part, top, r, delta, job.stepdown, 1.0);
    if let Some(w) = job.window {
        specs.retain(|s| w.contains(s.hint.0, s.hint.1));
    }
    let n_ops = specs.len();
    let started = Instant::now();
    // The order: from where the tool is, the nearest op whose stock above is cut (the first
    // op to begin), so a pocket goes down to its floor before the tool moves on.
    let waits = ops::waits(&specs, r);
    let mut done = vec![false; n_ops];
    let mut at: Option<(f64, f64)> = None;
    let mut outs = Vec::new();
    // The program so far, read back op by op, and filed by where its moves sweep.
    let mut posted: Vec<Move> = Vec::new();
    let mut index = MoveIndex::new(sheet.grow(d + 10.0), 25.0);
    let mut line_ranges = Vec::new(); // each op's lines in the program, [first, end)
    for k in 0..n_ops {
        let box_dist = |b: Box2, p: (f64, f64)| (b.x0 - p.0).max(p.0 - b.x1).max(0.0).hypot((b.y0 - p.1).max(p.1 - b.y1).max(0.0));
        let ready = (0..n_ops).filter(|&i| !done[i] && waits[i].iter().all(|&a| done[a]));
        let next = match at {
            None => ready.min(),
            Some(p) => ready.min_by(|&a, &b| {
                let key = |i: usize| (box_dist(specs[i].region, p), (specs[i].hint.0 - p.0).hypot(specs[i].hint.1 - p.1));
                key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            }),
        };
        // (The waits only point up, so some op is always ready.)
        let next = next.unwrap_or_else(|| (0..n_ops).find(|&i| !done[i]).unwrap_or(0));
        done[next] = true;
        let spec = specs[next].clone();
        let clock = Instant::now();
        if job.progress && (k % 25 == 0 || k + 1 == n_ops) {
            eprintln!("  planning op {}/{n_ops} ({:.0} s)", k + 1, started.elapsed().as_secs_f64());
        }
        let prior_moves = subset(&posted, &index.near(spec.region.grow(d), posted.len()));
        let i0 = prior_moves.len();
        let clear = SliceClearance::new(frep, spec.z_wall, 1e-3);
        let mut plan = ops::plan(frep, &clear, &spec, r, delta, job.pitch, job.stepdown);
        let mut out = OpOut {
            spec: spec.clone(),
            skipped: None,
            path: None,
            ours: Vec::new(),
            planned: None,
            posted: None,
            check: None,
            ap: 0.0,
            rho: 0.0,
            notes: Vec::new(),
            grid: (plan.x0, plan.y0, plan.h, plan.nx, plan.ny),
            outline: plan.contours(-(r + delta), true),
            range: (i0, i0),
            ms: 0.0,
        };
        let prior = Prior { moves: &prior_moves, tools: &tools, stock_top: top };
        let planned: Result<PathResult, String> = 'plan: {
            if !plan.ok() {
                break 'plan Err("no room for the tool center in this pocket, or none of its regions reaches the slab's floor".into());
            }
            // (Max-cut seeds slivers of its own; the other plans need a room the size of a seed.)
            let maxcut = job.plan.maxcut();
            if !maxcut && plan.g_max() * 0.7 < 0.05 * d {
                break 'plan Err(format!("too narrow for a seed (widest free radius {:.2} mm)", plan.g_max()));
            }
            // Axial depth: the stock the program so far leaves over the pocket, median, above the cut.
            let above = height_before(&prior, i0, &plan, spec.z_cut);
            let mut over: Vec<f64> = above.iter().enumerate().filter(|&(s, _)| plan.label[s] == 1).map(|(_, &a)| a).collect();
            over.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out.ap = over.get(over.len() / 2).copied().unwrap_or(0.0);
            let highest = over.last().copied().unwrap_or(0.0);
            if highest <= CLEARED_MM {
                break 'plan Err("already cleared: an earlier op left no stock over the pocket".into());
            }
            // Moves in air go just over the stock where the way is clear, else to the entry height.
            let guard = AirGuard::new(frep, &plan, &above, spec.z_cut, z_entry.min(spec.z_cut + highest + 1.0), r, delta);
            let low_ok = |a: P, b: P| guard.clear(a, b);
            let cut = base_cut.as_ref().map(|c| c.at_depth(out.ap));
            // The entry descends a loop (`entry`), ramped at the tool's limit or shallower,
            // so one lap goes no deeper than the cut it takes stays stable at. Max-cut's come
            // down from just over the stock, the spiral's from the entry height.
            let tan_ramp = (RAMP_DEG * PI / 180.0).tan();
            let v_entry = cut.as_ref().map_or(job.v_prog, |c| c.ceiling(PI).0);
            let z_top = if maxcut { guard.z } else { z_entry };
            let descent = Descent { dz: (z_top - spec.z_cut).max(0.0), tan_ramp, ap: loop_ap, v: v_entry, accel: job.accel, r };
            let air = Air { low: guard.z, high: z_entry, low_ok: &low_ok, tour: true, descent };
            if job.stepdown.is_none() && out.ap > 2.0 * d {
                out.notes.push(format!("the cut is {:.1} mm deep in one slab, more than twice the tool diameter; set --stepdown", out.ap));
            }
            if let Some(c) = &cut {
                if out.ap > c.loc {
                    out.notes.push(format!("stock above the cut is {:.1} mm, deeper than the {:.1} mm flute length; the model is taken at the flute length", out.ap, c.loc));
                }
            }
            plan.find_seeds(0.3 * d, 0.05 * d, if maxcut { f64::INFINITY } else { job.persist.unwrap_or(0.125 * r) }, 12);
            if maxcut {
                plan.add_sliver_seeds();
            }
            if plan.seeds.is_empty() {
                break 'plan Err("no room for an entry".into());
            }
            // Every plan starts from what its entries clear.
            plan.seeds = entry::tracks(&plan, r, &descent, &mut out.notes);
            out.rho = plan.rho();
            let (mut phys, mut obj) = (None, None);
            match (job.plan, cut.as_ref()) {
                (PlanKind::MaxCut, _) => plan.solve_eikonal(None),
                (PlanKind::Physics, Some(c)) => {
                    phys = Some(PhysPlan::solve(&prior, i0, spec.z_cut, &mut plan, c, ae_cap, 3, 1.0, None, false, PhysOptions::default()));
                }
                (PlanKind::Harmonic, _) => plan.solve(1e-7, 20000),
                (PlanKind::Ratio, _) => plan.solve_ratio(40),
                _ => plan.solve_conductive(1e-6, 20000),
            }
            if let (PlanKind::Walls, Some(c)) = (job.plan, cut.as_ref()) {
                phys = Some(PhysPlan::for_plan(&prior, i0, spec.z_cut, &plan, c, ae_cap, 1.0, PhysOptions::default()));
            }
            if let (PlanKind::Objective, Some(c)) = (job.plan, cut.as_ref()) {
                let mut o = Objective::new(&plan, c, job.accel, ae_cap);
                o.ap = Some(height_before(&prior, i0, &plan, spec.z_cut));
                o.optimize(&mut plan, c, 8, 0.5);
                let (lo, med, hi) = o.range(&plan);
                out.notes.push(format!("objective: J {:.1} s estimated; s* {lo:.2} / {med:.2} / {hi:.2} mm (min / median / max)", o.j));
                obj = Some(o);
            }
            if let Some(p) = &phys {
                out.notes.extend(p.trace.iter().cloned());
            }
            // Stepover: the reference cap, or with physical limits the knee (the narrowest
            // cut that already reaches the best steady removal rate), or the physics plan's own.
            let mut ae_max = ae_cap;
            if let Some(c) = &cut {
                ae_max = c.knee((0.03 * d).max(ae_cap.min(c.stable_ae(out.ap))), 0.03).0;
            }
            if let Some(p) = &phys {
                ae_max = p.s_ref;
            }
            let s_at = |x: f64, y: f64| phys.as_ref().map_or(0.0, |p: &PhysPlan| p.s_at(x, y));
            let o_at = |x: f64, y: f64| obj.as_ref().map_or(0.0, |o: &Objective| o.at(x, y));
            let sp = if maxcut {
                maxcut::build(&plan, spec.z_cut, &air, job.stepover.unwrap_or(ae_max), job.one_way, job.greedy, 0.5)
            } else {
                let spacing: Option<&dyn Fn(f64, f64) -> f64> = if obj.is_some() { Some(&o_at) } else if phys.is_some() { Some(&s_at) } else { None };
                spiral::build(&plan, spec.z_cut, z_entry, descent, ae_ref, ae_cap, 0.85 * d, 0.5, spacing)
            };
            for n in &sp.notes {
                if !out.notes.contains(n) {
                    out.notes.push(n.clone());
                }
            }
            if sp.points.is_empty() {
                break 'plan Err("the free area is a thin sliver with no closed wall loop around its center".into());
            }

            // Measure against the stock the program so far leaves, schedule, time.
            let ap_grid = obj.as_ref().and_then(|o| o.ap.clone());
            let ap_p = |x: f64, y: f64| phys.as_ref().map_or(0.0, |p| p.ap_at(x, y));
            let ap_o = |x: f64, y: f64| ap_grid.as_ref().map_or(0.0, |a| grid_at(&plan, a, x, y));
            let ap_at: Option<&dyn Fn(f64, f64) -> f64> = if ap_grid.is_some() { Some(&ap_o) } else if phys.is_some() { Some(&ap_p) } else { None };
            let lp = load_params(job, r, spec.z_cut, cut.as_ref(), ap_at);
            let mut stock_before = SweptStock::new(spec.region.grow(d), sheet);
            stock_before.add_program(&prior_moves, i0, &tools, spec.z_cut);
            let mut ours = analyze::from_path(&sp.points, 0.25);
            analyze::engagement(&mut ours, &mut stock_before, r, spec.z_cut);
            analyze::schedule(&mut ours, &lp);
            out.planned = Some(analyze::measure(&ours, &lp, false, 0.0, PHI_HEAVY_DEG));
            out.ours = ours;
            Ok(sp)
        };
        let first_line = writer.line_count();
        match planned {
            Ok(sp) => {
                let block = post::body(&to_post(&out.ours), z_entry, 0.005, job.dialect, origin);
                writer.op(&spec.name, (out.ours[0].x, out.ours[0].y), &block);
                for m in writer.read_back_from(first_line, &mut Vec::new()) {
                    index.add(&m, r);
                    posted.push(m);
                }
                out.path = Some(sp);
            }
            Err(why) => out.skipped = Some(why),
        }
        line_ranges.push((first_line, writer.line_count()));
        if let Some(p) = out.ours.last() {
            at = Some((p.x, p.y));
        }
        out.ms = clock.elapsed().as_secs_f64() * 1000.0;
        outs.push(out);
    }

    // The program as written, read back: each op measured as posted and checked.
    let draft = writer.finish();
    let moves = Writer::read_back(&draft, origin, &mut warnings);
    let feeds = feeds_by_line(&moves);
    let mut final_index = MoveIndex::new(sheet.grow(d + 10.0), 25.0);
    for m in &moves {
        final_index.add(m, r);
    }
    let mut gouging = Vec::new();
    for (k, (out, &(a, b))) in outs.iter_mut().zip(&line_ranges).enumerate() {
        // Lines a..b (0-based) are numbers a + 1 ..= b.
        let (i0, i1) = (moves.partition_point(|m| m.line <= a), moves.partition_point(|m| m.line <= b));
        out.range = (i0, i1);
        if job.progress && (k % 25 == 0 || k + 1 == n_ops) {
            eprintln!("  checking op {}/{n_ops} ({:.0} s)", k + 1, started.elapsed().as_secs_f64());
        }
        let spec = &out.spec;
        let clear = SliceClearance::new(frep, spec.z_wall, 1e-3);
        // The whole pocket (an op may leave a region to the slab under it), checked on the
        // stock the whole program leaves; skipped ops too, since a later one must cut them.
        let plan = Plan::new(&clear, r + delta, spec.region, job.pitch, &[spec.hint]);
        if !plan.ok() {
            continue;
        }
        let region = spec.region.grow(d);
        let near_before = subset(&moves, &final_index.near(region, i0));
        if out.path.is_some() {
            let mut s = analyze::from_moves(&moves, i0, i1, &feeds, spec.z_cut, 0.25, 2.0 * PI * out.rho, analyze::RAPID_MMPS);
            let mut before = SweptStock::new(region, sheet);
            before.add_program(&near_before, near_before.len(), &tools, spec.z_cut);
            analyze::engagement(&mut s, &mut before, r, spec.z_cut);
            let cut = base_cut.as_ref().map(|c| c.at_depth(out.ap));
            out.posted = Some(analyze::measure(&s, &load_params(job, r, spec.z_cut, cut.as_ref(), None), true, 0.0, PHI_HEAVY_DEG));
        }
        let near_final = subset(&moves, &final_index.near(region, moves.len()));
        let (lo, hi) = (V3::new(region.x0.max(stock.0.x), region.y0.max(stock.0.y), stock.0.z), V3::new(region.x1.min(stock.1.x), region.y1.min(stock.1.y), stock.1.z));
        let after = ImplicitStock::new(lo, hi, &near_final, &tools, ImplicitStock::CELL_MM);
        let pts = ops::design_points(frep, &plan, &clear, r + delta, spec.z_cut, spec.floor, top);
        let mut check = verify::run(&after, &pts, &plan, spec.z_cut, r, delta, top);
        let before = ImplicitStock::new(lo, hi, &near_before, &tools, ImplicitStock::CELL_MM);
        check.prior_residue_area = verify::residue(&before, &plan, spec.z_cut, r, delta, None);
        if check.gouges > 0 {
            gouging.push(format!("{} ({} points, worst {:.3} mm)", spec.name, check.gouges, check.max_gouge));
        }
        if check.residue_area > MAX_UNCUT {
            out.notes.push(format!("the checker found {:.0} mm2 of reachable stock left uncut", check.residue_area));
        }
        out.check = Some(check);
    }
    let program = if gouging.is_empty() {
        draft
    } else {
        let msg = format!("CHECKER FOUND GOUGES IN: {} - DO NOT RUN", gouging.join("; "));
        warnings.push(msg.to_lowercase());
        format!("({msg})\n{draft}")
    };
    Ok(JobOut { ops: outs, program, warnings, stock, origin, moves })
}

/// What a rapid at height `z` meets under the tool, on one op's plan grid: the stock tops
/// the program so far leaves there, and the part's columns standing above z - 0.5 with
/// their allowance skin (by the exact clearance). A grid node above z - 0.5 that is not
/// in such a column or its skin is stock in the way.
struct AirGuard<'a> {
    x0: f64,
    y0: f64,
    h: f64,
    nx: usize,
    ny: usize,
    tops: Vec<f64>,
    z: f64,
    tall: SliceClearance<'a>,
    r: f64,
    delta: f64,
    /// Per node: 0 not yet asked, 1 in a tall column or its skin, 2 stock.
    kind: RefCell<Vec<u8>>,
}

impl<'a> AirGuard<'a> {
    fn new(frep: &'a Frep, plan: &Plan, above: &[f64], z_cut: f64, z: f64, r: f64, delta: f64) -> AirGuard<'a> {
        AirGuard {
            x0: plan.x0,
            y0: plan.y0,
            h: plan.h,
            nx: plan.nx,
            ny: plan.ny,
            tops: above.iter().map(|a| z_cut + a).collect(),
            z,
            tall: SliceClearance::new(frep, z - 0.5, 1e-3),
            r,
            delta,
            kind: RefCell::new(vec![0; plan.nx * plan.ny]),
        }
    }

    /// The tool rapids from `a` to `b` at the guard's height without meeting the part or
    /// stock. Nodes are taken one pitch beyond the tool's radius (the stock between nodes).
    fn clear(&self, a: P, b: P) -> bool {
        // 0.01 mm into the skin: a link along a wall runs at exactly r + delta, where the
        // exact test would split the segment down to 0.1 um; this way it stops at 0.02 mm.
        if !self.tall.clear_along(a.0, a.1, b.0, b.1, self.r + self.delta - 0.01) {
            return false;
        }
        let h = self.h;
        let reach = self.r + h;
        let k = (reach / h).ceil() as i64;
        let n = 1.max(((b.0 - a.0).hypot(b.1 - a.1) / h).ceil() as usize);
        let mut kind = self.kind.borrow_mut();
        for q in 0..=n {
            let t = q as f64 / n as f64;
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            let (ci, cj) = (((x - self.x0) / h).round() as i64, ((y - self.y0) / h).round() as i64);
            for j in cj - k..=cj + k {
                for i in ci - k..=ci + k {
                    let (px, py) = (self.x0 + i as f64 * h, self.y0 + j as f64 * h);
                    if (px - x).hypot(py - y) > reach {
                        continue;
                    }
                    if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.ny as i64 {
                        return false;
                    }
                    let s = j as usize * self.nx + i as usize;
                    if self.tops[s] <= self.z - 0.5 {
                        continue;
                    }
                    if kind[s] == 0 {
                        kind[s] = if self.tall.at(px, py) < self.delta { 1 } else { 2 };
                    }
                    if kind[s] == 2 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// The move that reaches deepest past a gouge point `p` (index into `moves`) and by how
/// much, sideways, mm: a flat end mill of radius `r` covers `p` where its tip is at or
/// below it. Arcs are taken in chords of 0.5 mm, every chord sampled at 0.1 mm.
pub fn gouge_culprit(moves: &[Move], p: V3, r: f64) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (k, m) in moves.iter().enumerate() {
        let mut pts = vec![m.from];
        if matches!(m.kind, MoveKind::ArcCw | MoveKind::ArcCcw) {
            let c = m.center;
            let ra = (m.from.x - c.x).hypot(m.from.y - c.y);
            let (t0, t1) = ((m.from.y - c.y).atan2(m.from.x - c.x), (m.to.y - c.y).atan2(m.to.x - c.x));
            let full = (m.to - m.from).len() < 1e-6;
            let md = |a: f64| a.rem_euclid(2.0 * PI);
            let sweep = if m.kind == MoveKind::ArcCcw {
                if full { 2.0 * PI } else { md(t1 - t0) }
            } else {
                -(if full { 2.0 * PI } else { md(t0 - t1) })
            };
            let n = 2.max((sweep.abs() * ra / 0.5).ceil() as usize);
            for q in 1..=n {
                let a = t0 + sweep * q as f64 / n as f64;
                pts.push(V3::new(c.x + ra * a.cos(), c.y + ra * a.sin(), m.from.z + (m.to.z - m.from.z) * q as f64 / n as f64));
            }
        } else {
            pts.push(m.to);
        }
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a.z.min(b.z) > p.z + 1e-6 || (a.x.min(b.x) - r > p.x) || (a.x.max(b.x) + r < p.x) || (a.y.min(b.y) - r > p.y) || (a.y.max(b.y) + r < p.y) {
                continue;
            }
            let n = 1.max(((b - a).len() / 0.1).ceil() as usize);
            for q in 0..=n {
                let t = q as f64 / n as f64;
                let z = a.z + (b.z - a.z) * t;
                if z > p.z + 1e-6 {
                    continue;
                }
                let depth = r - (a.x + (b.x - a.x) * t - p.x).hypot(a.y + (b.y - a.y) * t - p.y);
                if depth > 0.0 && best.map_or(true, |(_, d)| depth > d) {
                    best = Some((k, depth));
                }
            }
        }
    }
    best
}

pub(crate) fn load_params<'a>(job: &Job, r: f64, z_cut: f64, cut: Option<&'a CutModel>, ap_at: Option<&'a dyn Fn(f64, f64) -> f64>) -> LoadParams<'a> {
    LoadParams {
        r,
        z_cut,
        v_prog: job.v_prog,
        phi_ref: job.phi_ref,
        feed_cap: job.feed_cap,
        accel: job.accel,
        v_entry: cut.map_or(job.v_prog, |c| c.ceiling(PI).0),
        v_rapid: analyze::RAPID_MMPS,
        cut,
        ap_at,
    }
}

fn grid_at(p: &Plan, g: &[f64], x: f64, y: f64) -> f64 {
    let (fx, fy) = ((x - p.x0) / p.h, (y - p.y0) / p.h);
    let i = (fx.floor() as i64).clamp(0, p.nx as i64 - 2) as usize;
    let j = (fy.floor() as i64).clamp(0, p.ny as i64 - 2) as usize;
    let (u, v) = ((fx - i as f64).clamp(0.0, 1.0), (fy - j as f64).clamp(0.0, 1.0));
    let s = j * p.nx + i;
    (1.0 - u) * (1.0 - v) * g[s] + u * (1.0 - v) * g[s + 1] + (1.0 - u) * v * g[s + p.nx] + u * v * g[s + p.nx + 1]
}
