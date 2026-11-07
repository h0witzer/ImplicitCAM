//! A toolpath driven by the cut, not by the profile.
//!
//! T is the geodesic distance from the entry (`Plan::solve_eikonal`), so its level sets
//! T = k * ae are exactly one stepover apart everywhere: every pass cuts the full chosen
//! width. Fronts are not bent to meet the walls; where one reaches a wall it simply
//! becomes an open arc from wall to wall. The profile only shapes the last pass: one
//! wall loop per wall, at the end.
//!
//!   closed phase: while each level is one closed front, consecutive fronts are one
//!     morphing revolution (points carried back along grad T), no transition
//!   open phase: every remaining piece of every level goes into a schedule. A piece is
//!     ready once the pieces of the level before that touch it are cut. The order comes
//!     from the field: depth first over the dependencies, a finished piece's ready
//!     successors in order along it, the first pieces by angle around the entry. Open
//!     arcs run climb on even levels and conventional on odd (zigzag), or climb only.
//!   links: kept down at feed through stock already cut, a short step-over when the
//!     next piece is within 1.25 stepovers, else retract, rapid, and a plunge at the
//!     piece's start (rapid through cut air, feed the last 0.5 mm). The retract goes
//!     only as high as the way needs ([`Air`]): just over the stock where nothing
//!     taller stands under the tool's path, else to the clearance height.
//! One entry per connected free region: fronts go around islands by themselves. A region
//! owns the front pieces and the wall loops on its own free side, so a region inside
//! another's outer wall (a cell in a frame) is not cut twice or taken for an island.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::entry::{track_entry, track_start, Descent};
use crate::path::{area, at_seam, dist, inside, nearest, orient, resample, resample_open, reverse, PathPt, PathResult, Tag, P};
use crate::plan::{Plan, Seed};

struct Piece {
    pts: Vec<P>,
    closed: bool,
    level: usize,
    preds: Vec<usize>,
    done: bool,
}

/// Heights for the moves in air. Entries start at `low` and links rapid there where
/// `low_ok` finds the way from one point to the other clear for the tool (no stock and
/// no part above `low` under it), else at `high`, which is always clear. With `tour` the
/// regions are cut nearest first from where the tool is, else in seed order.
pub struct Air<'f> {
    pub low: f64,
    pub high: f64,
    pub low_ok: &'f dyn Fn(P, P) -> bool,
    pub tour: bool,
    /// The entry as a descent on each seed's loop (`entry::track_entry`), ramped at the
    /// shallower of the tool's ramp limit and the loop's stable drop per lap.
    pub descent: Descent,
}

impl Air<'_> {
    fn z(&self, a: P, b: P) -> f64 {
        if self.low < self.high && (self.low_ok)(a, b) {
            self.low
        } else {
            self.high
        }
    }
}

struct Builder<'p, 'a, 'f> {
    plan: &'p Plan<'a>,
    res: PathResult,
    z_cut: f64,
    air: &'f Air<'f>,
    ae: f64,
    ds: f64,
    one_way: bool,
}

impl Builder<'_, '_, '_> {
    fn cur(&self) -> P {
        let p = self.res.points[self.res.points.len() - 1];
        (p.x, p.y)
    }

    fn add(&mut self, x: f64, y: f64, t: Tag) {
        self.res.points.push(PathPt::new(x, y, self.z_cut, t));
    }

    fn safe(&mut self, p: P) -> P {
        if self.plan.clearance(p.0, p.1) >= 0.0 {
            return p;
        }
        self.res.clamped += 1;
        self.plan.project_to_wall(p.0, p.1)
    }

    /// Get the tool from where it is to `to`, the start of a piece at level `cut`.
    fn link(&mut self, to: P, cut: f64) {
        let from = self.cur();
        let d = dist(from, to);
        if d < 0.05 {
            return;
        }
        let n = 1.max((d / 0.5).ceil() as i64);
        let path: Vec<P> = (1..=n).map(|i| (from.0 + (to.0 - from.0) * i as f64 / n as f64, from.1 + (to.1 - from.1) * i as f64 / n as f64)).collect();
        let inside = path.iter().all(|p| self.plan.in_pocket(p.0, p.1));
        // Kept down: every center on the way has its whole disk in stock already cut.
        if inside && path.iter().all(|p| self.plan.value(p.0, p.1) <= cut - self.ae + 0.1 * self.ae) {
            for p in path {
                self.add(p.0, p.1, Tag::Cut);
            }
            self.res.kept_down += 1;
            return;
        }
        if inside && d <= 1.25 * self.ae {
            for p in path {
                self.add(p.0, p.1, Tag::Cut);
            }
            return;
        }
        let last = self.res.points[self.res.points.len() - 1];
        let z = self.air.z(from, to);
        self.res.points.push(PathPt::new(last.x, last.y, z, Tag::Rapid));
        self.res.points.push(PathPt::new(to.0, to.1, z, Tag::Rapid));
        self.res.points.push(PathPt::new(to.0, to.1, self.z_cut + 0.5, Tag::Rapid));
        self.res.points.push(PathPt::new(to.0, to.1, self.z_cut, Tag::Entry));
        self.res.links += 1;
    }

    /// Cut one piece of the open phase, then the successors it made ready, in order
    /// along it from where the tool left it.
    fn cut_piece(&mut self, pieces: &mut [Piece], succ: &HashMap<usize, Vec<usize>>, pc: usize) {
        let at = self.cur();
        let pts: Vec<P> = if pieces[pc].closed {
            let lp = orient(pieces[pc].pts.clone());
            let s0 = nearest(&lp, at);
            let mut pts = at_seam(&lp, s0);
            pts.push(pts[0]);
            pts
        } else {
            // Direction from the field: climb on even levels, conventional on odd (zigzag
            // through a lobe), climb only with one_way. Not "nearer end", which flips
            // when the two ends are about equally far.
            let fwd = pieces[pc].pts.clone();
            let climb = climb(self.plan, &fwd);
            let want_climb = self.one_way || pieces[pc].level % 2 == 0;
            if climb == want_climb { fwd } else { reverse(fwd) }
        };
        let links0 = self.res.links;
        self.link(pts[0], pieces[pc].level as f64 * self.ae);
        for p in resample_open(&pts, self.ds) {
            self.add(p.0, p.1, Tag::Cut);
        }
        if !pieces[pc].closed {
            self.res.arcs += 1;
            if !climb(self.plan, &pts) {
                self.res.conventional_mm += len(&pts);
            }
        }
        pieces[pc].done = true;
        let mid = pts[pts.len() / 2];
        self.res.trace.push(format!(
            "open L{} preds {} {} len {:.0} mid ({:.0},{:.0}){}",
            pieces[pc].level,
            pieces[pc].preds.len(),
            if pieces[pc].closed { "closed" } else { "arc" },
            len(&pts),
            mid.0,
            mid.1,
            if self.res.links > links0 { " after retract" } else { "" }
        ));
        // Position along this piece, counted back from where the tool left it.
        let back = |q: usize| -> usize {
            let m = pieces[q].pts[pieces[q].pts.len() / 2];
            let (mut bi, mut bd) = (0, f64::MAX);
            for (i, &p) in pts.iter().enumerate() {
                let d = dist(p, m);
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            pts.len() - 1 - bi
        };
        let mut next: Vec<usize> = succ[&pc].iter().copied().filter(|&q| !pieces[q].done).collect();
        next.sort_by_key(|&q| back(q));
        for q in next {
            if !pieces[q].done && pieces[q].preds.iter().all(|&r| pieces[r].done) {
                self.cut_piece(pieces, succ, q);
            }
        }
    }
}

/// The max-cut path over `plan` (already solved by `solve_eikonal`) at stepover `ae`, the
/// moves in air at the heights of `air`, every seed entered by its descent loop. `greedy`
/// replaces the order from the field by nearest-ready (it fails the continuity test).
#[allow(clippy::too_many_arguments)]
pub fn build(plan: &Plan, z_cut: f64, air: &Air, ae: f64, one_way: bool, greedy: bool, ds: f64) -> PathResult {
    let mut b = Builder { plan, res: PathResult::default(), z_cut, air, ae, ds, one_way };
    let seeds = &plan.seeds;
    b.res.seeds = seeds.len();
    if seeds.is_empty() {
        b.res.notes.push("no seed".into());
        return b.res;
    }

    // Walls per region: of the wall loops on the region's free side, the innermost one
    // around its seed (counterclockwise) and the others (islands, clockwise).
    // (Sliver regions, new with the tour, have small loops: keep them.)
    let all_walls = plan.contours_min(0.0, true, if air.tour { 3 } else { 8 });
    let comp = plan.components();
    // The component on a loop's free side: the most common among its points' nearest nodes.
    let loop_comp: Vec<u32> = all_walls
        .iter()
        .map(|l| {
            let mut votes: HashMap<u32, usize> = HashMap::new();
            for p in l.iter().step_by(1.max(l.len() / 16)) {
                let c = plan.component_at(&comp, p.0, p.1);
                if c != 0 {
                    *votes.entry(c).or_default() += 1;
                }
            }
            votes.into_iter().max_by_key(|&(c, n)| (n, std::cmp::Reverse(c))).map_or(0, |(c, _)| c)
        })
        .collect();
    let project = |l: &[P]| -> Vec<P> { l.iter().map(|p| plan.project_to_wall(p.0, p.1)).collect() };
    let mut regions: Vec<(Seed, usize, u32, Vec<P>, Vec<Vec<P>>)> = Vec::new();
    for e in seeds {
        let rc = plan.component_at(&comp, e.mid_x(), e.mid_y());
        let mut raw: Option<usize> = None;
        for (i, l) in all_walls.iter().enumerate() {
            if loop_comp[i] == rc && inside(l, e.mid_x(), e.mid_y()) && raw.map_or(true, |r| area(l).abs() < area(&all_walls[r]).abs()) {
                raw = Some(i);
            }
        }
        let Some(raw) = raw else {
            b.res.notes.push("a seed has no wall loop around it".into());
            continue;
        };
        let outer = resample(&orient(project(&all_walls[raw])), ds * 0.5);
        let islands: Vec<Vec<P>> =
            (0..all_walls.len()).filter(|&i| i != raw && loop_comp[i] == rc).map(|i| resample(&reverse(orient(project(&all_walls[i]))), ds * 0.5)).collect();
        regions.push((e.clone(), raw, rc, outer, islands));
    }
    b.res.islands = regions.iter().map(|r| r.4.len()).sum();
    if regions.is_empty() {
        return b.res;
    }
    b.res.wall_loop = regions[0].3.clone();

    let kk = 0.max((plan.max_t / ae).ceil() as i64 - 1) as usize;
    for k in 0..=kk {
        b.res.levels.push(k as f64 * ae);
    }

    let mut left: Vec<Option<(Seed, usize, u32, Vec<P>, Vec<Vec<P>>)>> = regions.into_iter().map(Some).collect();
    loop {
        // Next: in seed order, or with a tour the region whose entry starts nearest the tool.
        let next = if air.tour && !b.res.points.is_empty() {
            let at = b.cur();
            (0..left.len()).filter(|&k| left[k].is_some()).min_by(|&p, &q| {
                let d = |k: usize| dist(track_start(&left[k].as_ref().unwrap().0), at);
                d(p).partial_cmp(&d(q)).unwrap_or(std::cmp::Ordering::Equal)
            })
        } else {
            left.iter().position(Option::is_some)
        };
        let Some(next) = next else { break };
        let (seed, raw, rc, outer_wall, islands) = left[next].take().unwrap();
        // The descent starts low where the tool can come straight down onto its start;
        // the way there from the last region is at its own height.
        let start: P = track_start(&seed);
        let z_entry = air.z(start, start);
        if air.low < air.high {
            if let Some(&last) = b.res.points.last() {
                let z = air.z((last.x, last.y), start).max(z_entry);
                b.res.points.push(PathPt::new(last.x, last.y, z, Tag::Rapid));
                b.res.points.push(PathPt::new(start.0, start.1, z, Tag::Rapid));
            } else if z_entry < air.high {
                b.res.points.push(PathPt::new(start.0, start.1, z_entry, Tag::Rapid));
            }
        }
        track_entry(&mut b.res, &seed, z_cut, z_entry, air.descent.tan_on(&seed), ds);
        let mut prev_start = start;

        let mut pieces: Vec<Piece> = Vec::new();
        let mut levels: Vec<Vec<usize>> = Vec::new();
        for k in 1..=kk {
            let mut lv = Vec::new();
            for (pts, closed) in plan.front_pieces(k as f64 * ae) {
                let m = pts[pts.len() / 2];
                let c = plan.component_at(&comp, m.0, m.1);
                if (c != 0 && c != rc) || (c == 0 && !inside(&all_walls[raw], m.0, m.1)) {
                    continue;
                }
                let pts: Vec<P> = pts.into_iter().map(|p| b.safe(p)).collect();
                lv.push(pieces.len());
                pieces.push(Piece { pts, closed, level: k, preds: Vec::new(), done: false });
            }
            levels.push(lv);
        }

        // Closed phase: one morphing revolution per level.
        let mut k_open = 1;
        while k_open <= kk {
            let lv = &levels[k_open - 1];
            if lv.len() != 1 || !pieces[lv[0]].closed {
                break;
            }
            let pc = lv[0];
            let cut = k_open as f64 * ae;
            pieces[pc].pts = orient(std::mem::take(&mut pieces[pc].pts));
            let lp = pieces[pc].pts.clone();
            let f = plan.flow(prev_start.0, prev_start.1, cut - ae, cut, 8, 2.0);
            let start = nearest(&lp, f);
            let sm = resample(&at_seam(&lp, start), ds);
            for i in 0..sm.len() {
                let u = i as f64 / sm.len() as f64;
                let mut p = plan.flow(sm[i].0, sm[i].1, cut, cut - (1.0 - u) * ae, 6, 2.0);
                if !plan.in_pocket(p.0, p.1) {
                    p = sm[i];
                    b.res.clamped += 1;
                }
                b.add(p.0, p.1, Tag::Cut);
            }
            // Close the last revolution on its own front before the open phase starts.
            if k_open == kk || levels[k_open].len() != 1 || !pieces[levels[k_open][0]].closed {
                for p in sm.iter().skip(1).chain(std::iter::once(&sm[0])) {
                    b.add(p.0, p.1, Tag::Cut);
                }
            }
            pieces[pc].done = true;
            prev_start = start;
            b.res.revolutions2 += 1;
            k_open += 1;
        }

        // Open phase: dependencies between consecutive levels (a piece needs the pieces of
        // the level before that touch it).
        let open: Vec<usize> = levels.iter().skip(k_open - 1).flatten().copied().filter(|&p| !pieces[p].done).collect();
        for &p in &open {
            if pieces[p].level == 1 {
                continue;
            }
            let before = levels[pieces[p].level - 2].clone();
            for q in before {
                if near(&pieces[p].pts, &pieces[q].pts, 1.5 * ae) {
                    pieces[p].preds.push(q);
                }
            }
        }
        if !greedy {
            // Order from the field, not a greedy choice: every key is a position, so a
            // tiny input change can only swap two pieces whose positions coincide, not
            // re-decide "nearest" at every step (the greedy order failed the continuity test).
            let mut succ: HashMap<usize, Vec<usize>> = open.iter().map(|&p| (p, Vec::new())).collect();
            for &p in &open {
                for &q in &pieces[p].preds {
                    if let Some(s) = succ.get_mut(&q) {
                        s.push(p);
                    }
                }
            }
            // Reference angle: the entry ridge's own direction (geometry), not where the tool is.
            let (cx, cy, a0) = (seed.mid_x(), seed.mid_y(), (seed.by - seed.ay).atan2(seed.bx - seed.ax));
            let ang = |p: &Piece| {
                let m = p.pts[p.pts.len() / 2];
                let a = ((m.1 - cy).atan2(m.0 - cx) - a0) % (2.0 * PI);
                if a < 0.0 { a + 2.0 * PI } else { a }
            };
            let mut roots: Vec<usize> = open.iter().copied().filter(|&p| pieces[p].preds.iter().all(|&q| pieces[q].done)).collect();
            roots.sort_by(|&x, &y| ang(&pieces[x]).partial_cmp(&ang(&pieces[y])).unwrap_or(std::cmp::Ordering::Equal));
            for root in roots {
                if !pieces[root].done {
                    b.cut_piece(&mut pieces, &succ, root);
                }
            }
            let left = open.iter().filter(|&&p| !pieces[p].done).count();
            if left > 0 {
                b.res.notes.push(format!("field order left {left} pieces; finished nearest-ready"));
            }
        }
        loop {
            let ready: Vec<usize> = open.iter().copied().filter(|&p| !pieces[p].done && pieces[p].preds.iter().all(|&q| pieces[q].done)).collect();
            if ready.is_empty() {
                if open.iter().any(|&p| !pieces[p].done) {
                    b.res.notes.push("max-cut schedule stalled".into());
                }
                break;
            }
            let at = b.cur();
            let (mut best, mut best_pts, mut best_d): (Option<usize>, Vec<P>, f64) = (None, Vec::new(), f64::MAX);
            for &pc in &ready {
                if pieces[pc].closed {
                    let lp = orient(pieces[pc].pts.clone());
                    let s = nearest(&lp, at);
                    let dd = dist(s, at) + pieces[pc].level as f64 * 1e-3;
                    if dd < best_d {
                        best_d = dd;
                        best = Some(pc);
                        best_pts = at_seam(&lp, s);
                        best_pts.push(best_pts[0]);
                    }
                    continue;
                }
                for rev in [false, true] {
                    let arc = if rev { reverse(pieces[pc].pts.clone()) } else { pieces[pc].pts.clone() };
                    if one_way && !climb(plan, &arc) {
                        continue;
                    }
                    let dd = dist(arc[0], at) + pieces[pc].level as f64 * 1e-3;
                    if dd < best_d {
                        best_d = dd;
                        best = Some(pc);
                        best_pts = arc;
                    }
                }
            }
            let Some(best) = best else { break };
            b.link(best_pts[0], pieces[best].level as f64 * ae);
            for p in resample_open(&best_pts, ds) {
                b.add(p.0, p.1, Tag::Cut);
            }
            if !pieces[best].closed {
                b.res.arcs += 1;
                if !climb(plan, &best_pts) {
                    b.res.conventional_mm += len(&best_pts);
                }
            }
            pieces[best].done = true;
        }

        // The profile: every wall of the region, once, at the end.
        for w in std::iter::once(&outer_wall).chain(islands.iter()) {
            let s = nearest(w, b.cur());
            b.link(s, (kk + 1) as f64 * ae);
            let ws = resample(&at_seam(w, s), ds);
            for j in 0..=ws.len() + (3.0 / ds) as usize {
                let p = ws[j % ws.len()];
                b.add(p.0, p.1, Tag::Wall);
            }
        }
    }
    b.res
}

fn len(l: &[P]) -> f64 {
    let mut s = 0.0;
    for i in 1..l.len() {
        s += dist(l[i - 1], l[i]);
    }
    s
}

/// Two runs of points come within d of each other (sampled).
fn near(a: &[P], b: &[P], d: f64) -> bool {
    let d2 = d * d;
    let (sa, sb) = (1.max(a.len() / 60), 1.max(b.len() / 60));
    for i in (0..a.len()).step_by(sa) {
        for j in (0..b.len()).step_by(sb) {
            if (a[i].0 - b[j].0) * (a[i].0 - b[j].0) + (a[i].1 - b[j].1) * (a[i].1 - b[j].1) <= d2 {
                return true;
            }
        }
    }
    false
}

/// Climb (right-hand tool, M03): the uncut side, up the gradient of T, is on the right of travel.
fn climb(plan: &Plan, arc: &[P]) -> bool {
    let m = arc.len() / 2;
    if m + 1 >= arc.len() {
        return true;
    }
    let (dx, dy) = (arc[m + 1].0 - arc[m].0, arc[m + 1].1 - arc[m].1);
    let (gx, gy) = plan.gradient(arc[m].0, arc[m].1);
    dy * gx - dx * gy > 0.0
}
