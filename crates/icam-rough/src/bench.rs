//! The scoreboard against a baseline program: the objectives the first roughing spike
//! (ROUGHING_FEASIBILITY.md section 12) was scored on, measured for the baseline and for the field roughing of the same part by
//! the same instruments.
//!
//!   baseline   the program's rough ops with the roughing tool's diameter, split into
//!              their cut levels (a level's run: the moves leading into its cuts and the
//!              cuts), each measured against the stock everything before it leaves: as
//!              posted (its own feeds) and re-fed to the reference load
//!   reference  the baseline's own nominal cut: the median engagement and the median
//!              programmed feed of its rough cuts (the spike's rule). Our roughing is
//!              held to the same load, so "re-fed" and "ours" face one limit
//!   gates      every pocket found from the field, checked on the stock each program's
//!              rough ops leave: gouges against the part, reachable stock left uncut
//!
//! With a material the reference is replaced by the physical limits for both.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::Path;

use icam_core::frep::Frep;
use icam_core::prismatic::{Place, PrismReport, ReadPart};
use icam_motion::analyze::{self, Metrics, Sample, RAPID_MMPS};
use icam_motion::CutModel;
use icam_post::{Box2, Move, MoveKind, Tag, ToolDef};
use icam_query::SliceClearance;
use icam_stock::{ImplicitStock, SweptStock};
use icam_strategy::verify::{self, Check};
use icam_strategy::Plan;
use rayon::prelude::*;

use crate::ops;
use crate::run::{load_params, run, Job, JobOut, PHI_HEAVY_DEG};

pub struct Baseline {
    pub moves: Vec<Move>,
    pub tools: BTreeMap<i32, ToolDef>,
    pub warnings: Vec<String>,
}

/// The program and its tool sheet (`set`, else the program's name with .set).
pub fn read_baseline(nc: &Path, set: Option<&Path>) -> Result<Baseline, String> {
    let mut warnings = Vec::new();
    let moves = icam_post::read(nc, &mut warnings).map_err(|e| format!("{}: {e}", nc.display()))?;
    let set = set.map_or_else(|| nc.with_extension("set"), Path::to_path_buf);
    let tools = icam_post::read_tool_table(&set, 25.4).map_err(|e| format!("{}: {e}", set.display()))?;
    Ok(Baseline { moves, tools, warnings })
}

/// One cut level of a baseline rough op, measured.
pub struct Level {
    pub op: String,
    pub z: f64,
    pub posted: Metrics,
    pub refed: Metrics,
}

/// One field pocket checked on both programs' stock.
pub struct Gate {
    pub op: String,
    pub ours: Option<Check>,
    pub base: Check,
}

pub struct BenchOut {
    pub rough_ops: Vec<String>,
    /// The reference cut: programmed feed (mm/s) and engagement (rad).
    pub v_prog: f64,
    pub phi_ref: f64,
    pub levels: Vec<Level>,
    pub ours: JobOut,
    pub gates: Vec<Gate>,
    /// Share of the baseline's rough cut samples more than 0.5 mm inside the part (a
    /// frame check: near 0 when the program and the part share coordinates).
    pub inside: f64,
    pub notes: Vec<String>,
}

fn cut_z(m: &Move) -> Option<f64> {
    (m.kind != MoveKind::Rapid && m.kind != MoveKind::Drill && (m.from.z - m.to.z).abs() < 1e-6).then_some(m.to.z)
}

/// Each op's moves split into runs by the cut level they lead into (a run ends where the
/// next level's approach begins).
fn level_runs(moves: &[Move], i0: usize, i1: usize) -> Vec<(f64, usize, usize)> {
    let mut lv = vec![f64::NAN; i1 - i0];
    let mut next = f64::NAN;
    for i in (i0..i1).rev() {
        if let Some(z) = cut_z(&moves[i]) {
            next = (z * 1000.0).round() / 1000.0;
        }
        lv[i - i0] = next;
    }
    // Moves after the last cut belong to it.
    let last = lv.iter().rev().find(|z| !z.is_nan()).copied().unwrap_or(f64::NAN);
    for z in lv.iter_mut().rev() {
        if !z.is_nan() {
            break;
        }
        *z = last;
    }
    let mut runs: Vec<(f64, usize, usize)> = Vec::new();
    for (k, &z) in lv.iter().enumerate() {
        if z.is_nan() {
            continue;
        }
        match runs.last_mut() {
            Some(r) if r.0 == z && r.2 == i0 + k => r.2 += 1,
            _ => runs.push((z, i0 + k, i0 + k + 1)),
        }
    }
    runs
}

fn feeds_by_line(moves: &[Move]) -> Vec<f64> {
    let n = moves.iter().map(|m| m.line).max().unwrap_or(0) + 2;
    let mut f = vec![0.0; n];
    for m in moves {
        f[m.line] = m.feed / 60.0;
    }
    f
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.get(v.len() / 2).copied()
}

/// Score the field roughing of the part (`frep`, XY box `part`) under `job` against the
/// baseline program.
pub fn bench(frep: &Frep, part: Box2, job: &Job, base: &Baseline) -> Result<BenchOut, String> {
    let d = job.tool.diameter;
    let r = d / 2.0;
    let mut notes = Vec::new();
    // In a window, the baseline's moves that sweep it (a move crossing its edge counts whole).
    let windowed;
    let base = match job.window {
        Some(w) => {
            let g = w.grow(r);
            let near = |m: &Move| {
                let (x0, x1) = (m.from.x.min(m.to.x), m.from.x.max(m.to.x));
                let (y0, y1) = (m.from.y.min(m.to.y), m.from.y.max(m.to.y));
                x0 <= g.x1 && x1 >= g.x0 && y0 <= g.y1 && y1 >= g.y0
            };
            windowed = Baseline { moves: base.moves.iter().filter(|m| near(m)).cloned().collect(), tools: base.tools.clone(), warnings: Vec::new() };
            notes.push(format!("window ({:.1}, {:.1}) .. ({:.1}, {:.1}): {} of the baseline's {} moves", w.x0, w.y0, w.x1, w.y1, windowed.moves.len(), base.moves.len()));
            &windowed
        }
        None => base,
    };
    // The baseline's ops in program order, and its rough ops with our tool's diameter.
    let mut ops_seen: Vec<(String, usize, usize)> = Vec::new();
    for (i, m) in base.moves.iter().enumerate() {
        match ops_seen.last_mut() {
            Some(o) if o.0 == m.op && o.2 == i => o.2 += 1,
            _ => ops_seen.push((m.op.clone(), i, i + 1)),
        }
    }
    let tool_of = |i0: usize, i1: usize| base.moves[i0..i1].iter().find(|m| m.kind != MoveKind::Rapid).and_then(|m| base.tools.get(&m.tool));
    let rough: Vec<(String, usize, usize)> = ops_seen
        .into_iter()
        .filter(|(name, i0, i1)| name.to_uppercase().starts_with("ROUGH") && tool_of(*i0, *i1).is_some_and(|t| (t.diameter - d).abs() < 0.01))
        .collect();
    if rough.is_empty() {
        return Err(format!("the baseline has no rough op with a {d:.3} mm tool"));
    }
    let part_top = frep.levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let top = job.stock_top.unwrap_or(part_top);
    let sheet = part;
    let feeds = feeds_by_line(&base.moves);

    // Measure every level run against the stock before it.
    let mut runs: Vec<(String, f64, usize, usize)> = Vec::new();
    for (name, i0, i1) in &rough {
        for (z, a, b) in level_runs(&base.moves, *i0, *i1) {
            runs.push((name.clone(), z, a, b));
        }
    }
    let measured: Vec<Vec<Sample>> = runs
        .par_iter()
        .map(|(_, z, a, b)| {
            let mut s = analyze::from_moves(&base.moves, *a, *b, &feeds, *z, 0.25, 0.0, RAPID_MMPS);
            let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for p in &s {
                (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
            }
            if !x0.is_finite() {
                return s;
            }
            let mut prior = SweptStock::new(Box2::new(x0, y0, x1, y1).grow(r + 2.0), sheet);
            prior.add_program(&base.moves, *a, &base.tools, *z);
            analyze::engagement(&mut s, &mut prior, r, *z);
            s
        })
        .collect();

    // The reference cut: the baseline's median engagement and programmed feed.
    let cuts: Vec<&Sample> = measured.iter().flatten().filter(|p| p.tag == Tag::Cut && p.phi.is_finite()).collect();
    let phi_ref = job.reference.map(|r| r.1).or_else(|| median(cuts.iter().filter(|p| p.phi > PI / 180.0).map(|p| p.phi).collect())).unwrap_or((1.0f64 - 0.8).acos());
    let v_prog = job.reference.map(|r| r.0).or_else(|| median(cuts.iter().map(|p| p.programmed).collect())).unwrap_or(job.v_prog);
    let inside = cuts.iter().filter(|p| frep.eval([p.x, p.y, p.z]).value < -0.5).count() as f64 / cuts.len().max(1) as f64;
    if inside > 0.02 {
        notes.push(format!("{:.0}% of the baseline's rough cuts lie more than 0.5 mm inside the part: do the program and the part share coordinates?", 100.0 * inside));
    }
    let mut job = job.clone();
    if job.material.is_none() {
        job.v_prog = v_prog;
        job.phi_ref = phi_ref;
    }
    let base_cut = job.material.as_deref().map(|m| CutModel::for_material(m, &job.tool, job.rpm)).transpose()?;

    // Each level as posted and re-fed (with a material: the depth from the level above).
    let mut levels = Vec::new();
    let mut above: BTreeMap<String, f64> = BTreeMap::new();
    for ((name, z, _, _), s) in runs.iter().zip(measured) {
        let ap = above.get(name).copied().unwrap_or(top) - z;
        if ap > 1e-6 {
            above.insert(name.clone(), *z);
        }
        let cut = base_cut.as_ref().map(|c| c.at_depth(ap.max(0.1)));
        let lp = load_params(&job, r, *z, cut.as_ref(), None);
        let posted = analyze::measure(&s, &lp, true, 0.0, PHI_HEAVY_DEG);
        let mut s2 = s;
        analyze::schedule(&mut s2, &lp);
        let refed = analyze::measure(&s2, &lp, false, 0.0, PHI_HEAVY_DEG);
        levels.push(Level { op: name.clone(), z: *z, posted, refed });
    }

    // Our roughing, held to the same reference.
    let ours = run(frep, part, &job)?;

    // Both programs' stock on every field pocket.
    let rough_moves: Vec<Move> = rough.iter().flat_map(|(_, a, b)| base.moves[*a..*b].iter().cloned()).collect();
    let (mut lo, mut hi) = ours.stock;
    if let Some(w) = job.window {
        let w = w.grow(d + 10.0);
        (lo.x, lo.y, hi.x, hi.y) = (lo.x.max(w.x0), lo.y.max(w.y0), hi.x.min(w.x1), hi.y.min(w.y1));
    }
    let base_stock = ImplicitStock::new(lo, hi, &rough_moves, &base.tools, ImplicitStock::CELL_MM);
    let gates = ours
        .ops
        .iter()
        .map(|o| {
            let spec = &o.spec;
            let clear = SliceClearance::new(frep, spec.z_wall, 1e-3);
            let plan = Plan::new(&clear, r + job.allowance, spec.region, job.pitch, &[spec.hint]);
            let pts = ops::design_points(frep, &plan, &clear, r + job.allowance, spec.z_cut, spec.floor, top);
            let check = verify::run(&base_stock, &pts, &plan, spec.z_cut, r, job.allowance, top);
            Gate { op: spec.name.clone(), ours: o.check.clone(), base: check }
        })
        .collect();
    Ok(BenchOut { rough_ops: rough.iter().map(|o| o.0.clone()).collect(), v_prog, phi_ref, levels, ours, gates, inside, notes })
}

/// One line of the scoreboard: the objective, the spike's result against its baseline
/// (the bar), the baseline here, ours here, and whether ours clears the bar.
pub struct Row {
    pub objective: String,
    pub bar: String,
    pub base: String,
    pub ours: String,
    pub pass: bool,
}

fn sum(m: &[&Metrics], f: fn(&Metrics) -> f64) -> f64 {
    m.iter().map(|x| f(x)).sum()
}

/// The scoreboard rows for a bench (reference load, or physical limits with a material).
pub fn score(b: &BenchOut, material: bool) -> Vec<Row> {
    let posted: Vec<&Metrics> = b.ours.ops.iter().filter_map(|o| o.posted.as_ref()).collect();
    let base_p: Vec<&Metrics> = b.levels.iter().map(|l| &l.posted).collect();
    let base_r: Vec<&Metrics> = b.levels.iter().map(|l| &l.refed).collect();
    let checks: Vec<&Check> = b.ours.ops.iter().filter_map(|o| o.check.as_ref()).collect();
    let gouges: usize = checks.iter().map(|c| c.gouges).sum();
    let base_gouges: usize = b.gates.iter().map(|g| g.base.gouges).sum();
    let worst = checks.iter().map(|c| c.residue_area).fold(0.0, f64::max);
    let base_worst = b.gates.iter().map(|g| g.base.residue_area).fold(0.0, f64::max);
    let t = |m: &[&Metrics]| sum(m, Metrics::time);
    let (ours_t, base_t, refed_t) = (t(&posted), t(&base_p), t(&base_r));
    let max_of = |m: &[&Metrics], f: fn(&Metrics) -> f64| m.iter().map(|x| f(x)).fold(0.0, f64::max);
    let mut rows = vec![
        Row {
            objective: "Gouges (checker points)".into(),
            bar: "0 (spike: 0)".into(),
            base: base_gouges.to_string(),
            ours: gouges.to_string(),
            pass: gouges == 0,
        },
        Row {
            objective: "Reachable stock left uncut, worst pocket".into(),
            bar: "<= 5 mm2 (spike: 0.5)".into(),
            base: format!("{base_worst:.1} mm2"),
            ours: format!("{worst:.1} mm2"),
            pass: worst <= 5.0,
        },
    ];
    if material {
        let cut = |m: &[&Metrics]| sum(m, |x| x.time_cut);
        rows.push(Row {
            objective: "Cut time vs the baseline re-fed to the same limits".into(),
            bar: "<= 0.87x (spike: 94.5 vs 109.1 s)".into(),
            base: format!("{:.1} s re-fed, {:.1} s posted", cut(&base_r), cut(&base_p)),
            ours: format!("{:.1} s ({:.2}x)", cut(&posted), cut(&posted) / cut(&base_r).max(1e-9)),
            pass: cut(&posted) <= 0.87 * cut(&base_r),
        });
        rows.push(Row {
            objective: "Total time vs the baseline re-fed to the same limits".into(),
            bar: "<= 1.08x (spike: 143 vs 132 s)".into(),
            base: format!("{refed_t:.1} s re-fed, {base_t:.1} s posted"),
            ours: format!("{ours_t:.1} s ({:.2}x)", ours_t / refed_t.max(1e-9)),
            pass: ours_t <= 1.08 * refed_t,
        });
    } else {
        rows.push(Row {
            objective: "Time vs the baseline held to the same load".into(),
            bar: "<= 1.014x (spike: 221 vs 218 s)".into(),
            base: format!("{refed_t:.1} s held, {base_t:.1} s posted"),
            ours: format!("{ours_t:.1} s ({:.3}x)", ours_t / refed_t.max(1e-9)),
            pass: ours_t <= 1.014 * refed_t,
        });
        rows.push(Row {
            objective: "Peak load / the reference cut".into(),
            bar: "<= 1.15 (spike: 1.04-1.15)".into(),
            base: format!("{:.2} posted", max_of(&base_p, |m| m.load_max)),
            ours: format!("{:.2}", max_of(&posted, |m| m.load_max)),
            pass: max_of(&posted, |m| m.load_max) <= 1.15,
        });
    }
    // Chips on target: thinner rubs (and melts a plastic), thicker overloads the edge.
    // (Engagement is measured for the chip, never planned for.)
    let band = |m: &[&Metrics]| sum(m, |x| x.chip_band_len) / sum(m, |x| x.chip_len).max(1e-9);
    let thin = |m: &[&Metrics]| m.iter().filter(|x| x.chip_len > 0.0).map(|x| x.chip_p5).fold(f64::INFINITY, f64::min);
    let (ob, bb) = (band(&posted), band(&base_p));
    rows.push(Row {
        objective: "Chip at 85-115 % of the target, share of the cut".into(),
        bar: ">= the baseline (spike: not scored)".into(),
        base: format!("{:.0} % (thinnest p5 {:.2})", 100.0 * bb, thin(&base_p)),
        ours: format!("{:.0} % (thinnest p5 {:.2})", 100.0 * ob, thin(&posted)),
        pass: ob >= bb,
    });
    rows
}

/// Where a part sits in a baseline program: its orientation (quarter turns, upside down
/// or not) and shift, found from the program's own cuts.
pub struct Aligned {
    pub place: Place,
    pub frep: Frep,
    pub bbox: Box2,
    pub report: PrismReport,
    /// Mean depth (mm) the program's tool tips reach inside the part at that placement,
    /// and at every orientation tried.
    pub depth: f64,
    pub tried: Vec<(Place, f64)>,
    /// Orientations that matched the program's extent but did not compile, and why.
    pub failed: Vec<(Place, String)>,
}

/// Points along the program's cutting moves (ends, middles, arcs at quarter sweeps).
fn cut_points(base: &Baseline) -> Vec<[f64; 3]> {
    let mut pts = Vec::new();
    for m in &base.moves {
        match m.kind {
            MoveKind::Rapid | MoveKind::Drill => {}
            MoveKind::Feed => {
                for f in [0.0, 0.5, 1.0] {
                    pts.push([m.from.x + (m.to.x - m.from.x) * f, m.from.y + (m.to.y - m.from.y) * f, m.from.z + (m.to.z - m.from.z) * f]);
                }
            }
            _ => {
                let c = m.center;
                let r = ((m.from.x - c.x).powi(2) + (m.from.y - c.y).powi(2)).sqrt();
                let (t0, t1) = ((m.from.y - c.y).atan2(m.from.x - c.x), (m.to.y - c.y).atan2(m.to.x - c.x));
                let two_pi = 2.0 * PI;
                let sweep = if m.kind == MoveKind::ArcCcw { (t1 - t0).rem_euclid(two_pi) } else { -(t0 - t1).rem_euclid(two_pi) };
                for k in 0..=4 {
                    let f = k as f64 / 4.0;
                    let a = t0 + sweep * f;
                    pts.push([c.x + r * a.cos(), c.y + r * a.sin(), m.from.z + (m.to.z - m.from.z) * f]);
                }
            }
        }
    }
    // At most 40,000, evenly through the program.
    let step = (pts.len() / 40_000).max(1);
    pts.into_iter().step_by(step).collect()
}

/// Mean depth the points reach inside the part, the part shifted by (dx, dy).
fn depth(frep: &Frep, pts: &[[f64; 3]], dx: f64, dy: f64) -> f64 {
    let s: f64 = pts.par_iter().map(|p| (-frep.eval([p[0] - dx, p[1] - dy, p[2]]).value).max(0.0)).sum();
    s / pts.len().max(1) as f64
}

/// The part's placement in the baseline program: of the eight square orientations, those
/// whose box matches the extent of the program's cuts (less the largest cutting radius
/// all round), each centered on them and set on the table (z 0 at its bottom); the one
/// the program's tool tips reach least deep into, then its shift refined on a lattice.
pub fn align(part: &ReadPart, base: &Baseline) -> Result<Aligned, String> {
    let pts = cut_points(base);
    if pts.is_empty() {
        return Err("the baseline has no cutting moves to place the part by".into());
    }
    let r_max = base.tools.values().filter(|t| t.kind == "ENDMILL").map(|t| t.diameter / 2.0).fold(0.0, f64::max);
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &pts {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    let (cx, cy) = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
    let (px, py) = (hi[0] - lo[0] - 2.0 * r_max, hi[1] - lo[1] - 2.0 * r_max);
    let (mut tried, mut failed) = (Vec::new(), Vec::new());
    let mut best: Option<(f64, Place, Frep, Box2, PrismReport)> = None;
    for flip in [false, true] {
        for q in 0..4u8 {
            let turn = Place::new(q, flip, 0.0, 0.0, 0.0);
            let (a, b) = part.bounds(&turn);
            let (sx, sy) = (b[0] - a[0], b[1] - a[1]);
            if (sx - px).abs() > 0.05 * px.max(1.0) || (sy - py).abs() > 0.05 * py.max(1.0) {
                continue;
            }
            let place = Place::new(q, flip, cx - 0.5 * (a[0] + b[0]), cy - 0.5 * (a[1] + b[1]), -a[2]);
            let (frep, (l, h), rep) = match part.compile(&place) {
                Ok(c) => c,
                Err(e) => {
                    failed.push((place, e));
                    continue;
                }
            };
            let d = depth(&frep, &pts, 0.0, 0.0);
            tried.push((place, d));
            if best.as_ref().is_none_or(|b| d < b.0) {
                best = Some((d, place, frep, Box2::new(l.x, l.y, h.x, h.y), rep));
            }
        }
    }
    let why: Vec<String> = failed.iter().map(|(p, e)| format!("{}{}: {e}", p.quarter_turns, if p.flip { "f" } else { "" })).collect();
    let (_, place, frep, bbox, report) =
        best.ok_or(format!("no orientation of the part fits the program's cuts ({px:.1} x {py:.1} mm less the tool){}", if why.is_empty() { String::new() } else { format!("; {}", why.join("; ")) }))?;
    // Refine the shift: coarse then fine lattices around the best.
    let (mut dx, mut dy) = (0.0, 0.0);
    for (span, step) in [(4.0, 0.5), (0.5, 0.05)] {
        let n = (span / step) as i32;
        let (mut bd, mut bx, mut by) = (f64::INFINITY, dx, dy);
        for i in -n..=n {
            for j in -n..=n {
                let (x, y) = (dx + i as f64 * step, dy + j as f64 * step);
                let d = depth(&frep, &pts, x, y);
                if d < bd - 1e-12 || ((d - bd).abs() <= 1e-12 && x.hypot(y) < bx.hypot(by)) {
                    (bd, bx, by) = (d, x, y);
                }
            }
        }
        (dx, dy) = (bx, by);
    }
    let place = place.shifted(dx, dy, 0.0);
    let (frep, bbox, report) = if dx == 0.0 && dy == 0.0 {
        (frep, bbox, report)
    } else {
        let (f, (l, h), r) = part.compile(&place)?;
        (f, Box2::new(l.x, l.y, h.x, h.y), r)
    };
    let d = depth(&frep, &pts, 0.0, 0.0);
    Ok(Aligned { place, frep, bbox, report, depth: d, tried, failed })
}
