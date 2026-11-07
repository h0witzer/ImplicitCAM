//! The toolpath read off the plan field, with its topology.
//!
//! Levels t_0 = 0 (the entry stadiums) < t_1 < ... < t_K = 1 (the walls) keep the
//! spacing between consecutive fronts within [ae_ref, ae_cap] where the field allows:
//! spacing ~ dT / |grad T|, so
//!   dT = min(ae_ref * max|grad T|, ae_cap * min|grad T|)   over all loops at t,
//! corrected with the true spacing measured along flow lines.
//!
//! Every loop at a level is keyed by what it encloses:
//!   outer  the seeds inside it (cleared inside; counterclockwise = climb)
//!   hole   the islands inside it and no seed (material inside; clockwise)
//! A branch is one key across consecutive levels. The field's topology shows up as keys
//! appearing and disappearing: seed fronts merge at the ridge minima between rooms (an
//! outer key with more seeds), holes are born behind islands and split between them.
//! Each branch is cut as one continuous morphing spiral: revolution j is parametrized by
//! the branch's loop at t_j (arc fraction u from a seam) and every point is carried back
//! along the gradient flow to level t_j-1 + u (t_j - t_j-1). The seam is a flow line, so
//! revolutions join exactly. Order: seed branches (each entered by its descent loop,
//! `entry`), then merged outer branches once their parents are done (ending on the
//! outer wall), then holes, largest first (ending on each island's wall). Branches are
//! joined by retract, rapid, and a plunge on a front already cut, with a lead-in down
//! the flow line.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;
use std::rc::Rc;

use crate::entry::{track_entry, Descent};
use crate::path::{area, at_seam, inside, nearest, orient, resample, reverse, PathPt, PathResult, Tag, P};
use crate::plan::{Plan, Seed};

struct Loop {
    pts: Vec<P>,
    outer: bool,
    mask: u32,
}

#[derive(Clone, Copy, PartialEq)]
struct Branch {
    outer: bool,
    mask: u32,
    a: usize,
    b: usize,
}

struct Sp<'p, 'a> {
    plan: &'p Plan<'a>,
    res: PathResult,
    z_cut: f64,
    z_entry: f64,
    descent: Descent,
    ds: f64,
    outers: Vec<(u32, Vec<P>)>,
    islands: Vec<(Vec<P>, f64, f64)>,
    cache: HashMap<u64, Rc<Vec<Loop>>>,
    levels: Vec<f64>,
}

impl Sp<'_, '_> {
    fn seeds(&self) -> &[Seed] {
        &self.plan.seeds
    }

    fn seed_mask(&self, l: &[P]) -> u32 {
        let mut m = 0;
        for (i, s) in self.seeds().iter().enumerate() {
            if inside(l, s.mid_x(), s.mid_y()) {
                m |= 1 << i;
            }
        }
        m
    }

    fn island_mask(&self, l: &[P]) -> u32 {
        let mut m = 0;
        for (i, isl) in self.islands.iter().enumerate() {
            if inside(l, isl.1, isl.2) {
                m |= 1 << i;
            }
        }
        m
    }

    /// Stadium around a seed's ridge segment, counterclockwise from its B tip.
    fn stadium(&self, e: &Seed) -> Vec<P> {
        let len = ((e.bx - e.ax) * (e.bx - e.ax) + (e.by - e.ay) * (e.by - e.ay)).sqrt();
        let th = if len > 1e-9 { (e.by - e.ay).atan2(e.bx - e.ax) } else { 0.0 };
        let mut st = Vec::new();
        let nc = 12.max((PI * e.rho / self.ds) as i64);
        for i in 0..=nc {
            let a = th - PI / 2.0 + PI * i as f64 / nc as f64;
            st.push((e.bx + e.rho * a.cos(), e.by + e.rho * a.sin()));
        }
        for i in 0..=nc {
            let a = th + PI / 2.0 + PI * i as f64 / nc as f64;
            st.push((e.ax + e.rho * a.cos(), e.ay + e.rho * a.sin()));
        }
        at_seam(&orient(st), (e.bx + e.rho * th.cos(), e.by + e.rho * th.sin()))
    }

    /// Loops at a level, keyed.
    fn loops(&mut self, t: f64) -> Rc<Vec<Loop>> {
        if let Some(c) = self.cache.get(&t.to_bits()) {
            return c.clone();
        }
        let mut r = Vec::new();
        if t <= 0.0 {
            for (i, s) in self.seeds().iter().enumerate() {
                r.push(Loop { pts: self.stadium(s), outer: true, mask: 1 << i });
            }
        } else if t >= 1.0 {
            for o in &self.outers {
                r.push(Loop { pts: o.1.clone(), outer: true, mask: o.0 });
            }
            for (i, isl) in self.islands.iter().enumerate() {
                r.push(Loop { pts: isl.0.clone(), outer: false, mask: 1 << i });
            }
        } else {
            for l in self.plan.contours(t, false) {
                let sm = self.seed_mask(&l);
                if sm != 0 {
                    r.push(Loop { pts: orient(l), outer: true, mask: sm });
                    continue;
                }
                let im = self.island_mask(&l);
                if im != 0 {
                    r.push(Loop { pts: reverse(orient(l)), outer: false, mask: im });
                } else {
                    self.res.dropped += 1; // encloses nothing: stranded material
                }
            }
        }
        let r = Rc::new(r);
        self.cache.insert(t.to_bits(), r.clone());
        r
    }

    fn loop_of(&mut self, k: usize, outer: bool, mask: u32) -> Vec<P> {
        let t = self.levels[k];
        let ls = self.loops(t);
        ls.iter().find(|l| l.outer == outer && l.mask == mask).map(|l| l.pts.clone()).unwrap_or_default()
    }

    /// The exact wall is the hard boundary. Contours of the grid plan near T = 1 can sit
    /// up to a cell past it (the grid's outside nodes carry T = 1), so every point is
    /// pulled back onto the exact clearance, or dropped if the projection cannot reach
    /// it. Every feed move is checked against the exact clearance: one that would cross
    /// the wall (a run of dropped points, a flow line that left the free area) becomes a
    /// retract, rapid and plunge instead, counted in safety_links - a defect of the
    /// extraction to fix, never a gouge.
    fn add(&mut self, mut x: f64, mut y: f64, tag: Tag) {
        if self.plan.clearance(x, y) < 0.0 {
            let w = self.plan.project_to_wall(x, y);
            if self.plan.clearance(w.0, w.1) < -1e-4 {
                return;
            }
            (x, y) = w;
            self.res.clamped += 1;
        }
        if let Some(&last) = self.res.points.last() {
            // Exact over the whole move, not sampled (sampling every 0.1 mm could step over a
            // part corner poking about 0.05 mm into it). 10 um: chord sag along a curved wall
            // is a few um.
            if last.tag != Tag::Rapid && (last.z - self.z_cut).abs() < 1e-6 && !self.plan.clear_along(last.x, last.y, x, y, 0.01) {
                self.res.points.push(PathPt::new(last.x, last.y, self.z_entry, Tag::Rapid));
                self.res.points.push(PathPt::new(x, y, self.z_entry, Tag::Rapid));
                self.res.points.push(PathPt::new(x, y, self.z_cut + 0.5, Tag::Rapid));
                self.res.points.push(PathPt::new(x, y, self.z_cut, Tag::Entry));
                self.res.links += 1;
                self.res.safety_links += 1;
            }
        }
        self.res.points.push(PathPt::new(x, y, self.z_cut, tag));
    }

    fn safe(&mut self, p: P, fallback: P) -> P {
        if self.plan.in_pocket(p.0, p.1) {
            return p;
        }
        self.res.clamped += 1;
        if self.plan.clearance(p.0, p.1) < 0.0 && self.plan.in_pocket(fallback.0, fallback.1) {
            let w = self.plan.project_to_wall(p.0, p.1);
            if self.plan.in_pocket(w.0, w.1) || self.plan.clearance(w.0, w.1).abs() < 1e-3 {
                return w;
            }
        }
        fallback
    }

    /// The flow line through (x, y) at level t0, up to 0.999.
    fn seam(&self, x: f64, y: f64, t0: f64) -> Vec<(f64, f64, f64)> {
        let mut s = vec![(t0, x, y)];
        let mut t = t0;
        while t < 0.999 {
            let t1 = (t + 0.002).min(0.999);
            let last = s[s.len() - 1];
            let (nx, ny) = self.plan.flow(last.1, last.2, t, t1, 2, 2.0);
            s.push((t1, nx, ny));
            t = t1;
        }
        s
    }

    /// One revolution from t_{j-1} to t_j on `lp` (the branch's loop at t_j), seamed on `seam`.
    fn revolution(&mut self, j: usize, lp: &[P], seam: &[(f64, f64, f64)]) {
        let (t0, t1) = (self.levels[j - 1], self.levels[j]);
        let mut start = seam_point(seam, t1.min(0.999));
        if t1 >= 1.0 {
            start = self.plan.project_to_wall(start.0, start.1);
        }
        let samples = resample(&at_seam(lp, nearest(lp, start)), self.ds);
        let n = samples.len();
        for (i, &q) in samples.iter().enumerate() {
            let u = i as f64 / n as f64;
            let tau = t0 + u * (t1 - t0);
            let (x, y) = self.plan.flow(q.0, q.1, t1.min(1.0 - 1e-6), tau, 6, 2.0);
            let p = self.safe((x, y), q);
            self.add(p.0, p.1, Tag::Cut);
        }
    }

    /// Retract, rapid, plunge at `p` (on cut material), then feed along `lead`.
    fn link(&mut self, p: P, lead: &[P]) {
        if let Some(&last) = self.res.points.last() {
            self.res.points.push(PathPt::new(last.x, last.y, self.z_entry, Tag::Rapid));
            self.res.points.push(PathPt::new(p.0, p.1, self.z_entry, Tag::Rapid));
            self.res.links += 1;
        }
        self.res.points.push(PathPt::new(p.0, p.1, self.z_cut, Tag::Entry));
        for &q in lead {
            self.add(q.0, q.1, Tag::Cut);
        }
    }

    /// Run one branch: its entry or link, its revolutions, and its wall when it ends on one.
    fn run(&mut self, b: Branch) {
        let big_l = self.levels.len() - 1;
        let (seam, first);
        if b.a == 0 {
            let si = b.mask.trailing_zeros() as usize;
            let e = self.seeds()[si].clone();
            track_entry(&mut self.res, &e, self.z_cut, self.z_entry, self.descent.tan_on(&e), self.ds);
            let l0 = self.loops(0.0);
            let tip = l0.iter().find(|l| l.mask == b.mask).map_or((e.ax, e.ay), |l| l.pts[0]);
            seam = self.seam(tip.0, tip.1, 0.0);
            first = 1;
        } else {
            // Start where the tool is closest; plunge two levels back on material already cut.
            let lp = self.loop_of(b.a, b.outer, b.mask);
            let at = self.res.points.last().map_or(lp[0], |p| (p.x, p.y));
            let s = nearest(&lp, at);
            let (t_a, t_prev, t_safe) = (self.levels[b.a].min(1.0 - 1e-6), self.levels[b.a - 1], self.levels[b.a.saturating_sub(2)]);
            let mut lead = Vec::new();
            for q in 0..=12 {
                let p = self.plan.flow(s.0, s.1, t_a, t_safe + (t_prev - t_safe) * q as f64 / 12.0, 12, 2.0);
                if self.plan.in_pocket(p.0, p.1) {
                    lead.push(p);
                }
            }
            if lead.is_empty() {
                lead.push(self.plan.flow(s.0, s.1, t_a, t_prev, 12, 2.0));
            }
            self.link(lead[0], &lead[1..]);
            let mut sm = self.seam(s.0, s.1, t_a);
            let last = lead[lead.len() - 1];
            sm.insert(0, (t_prev, last.0, last.1));
            seam = sm;
            first = b.a;
        }
        for j in first..=b.b {
            if j == 0 {
                continue;
            }
            let lp = self.loop_of(j, b.outer, b.mask);
            self.revolution(j, &lp, &seam);
        }
        if b.b == big_l {
            // One full turn on the wall this branch ends on, a little past the seam.
            let w = self.loop_of(big_l, b.outer, b.mask);
            let last = self.res.points[self.res.points.len() - 1];
            let ws = resample(&at_seam(&w, nearest(&w, (last.x, last.y))), self.ds);
            for j in 0..=ws.len() + (3.0 / self.ds) as usize {
                let p = ws[j % ws.len()];
                self.add(p.0, p.1, Tag::Wall);
            }
        }
    }
}

/// Seam: the flow line through p (at level t0), at level t.
fn seam_point(seam: &[(f64, f64, f64)], t: f64) -> P {
    if t <= seam[0].0 {
        return (seam[0].1, seam[0].2);
    }
    let last = seam[seam.len() - 1];
    if t >= last.0 {
        return (last.1, last.2);
    }
    let k = seam.iter().position(|s| s.0 >= t).unwrap_or(seam.len() - 1);
    let (a, b) = (seam[k - 1], seam[k]);
    let f = (t - a.0) / (b.0 - a.0);
    (a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f)
}

/// The spiral over `plan` (already solved), each seed entered by its descent loop
/// (`entry::track_entry`), ramped at the shallower of `descent`'s ramp limit and the
/// loop's stable drop per lap. `spacing`: a target stepover
/// field s*(x, y) instead of [ae_ref, ae_cap]: each level steps by its tightest point, so
/// spacing <= s* everywhere.
#[allow(clippy::too_many_arguments)]
pub fn build(
    plan: &Plan,
    z_cut: f64,
    z_entry: f64,
    descent: Descent,
    ae_ref: f64,
    ae_cap: f64,
    max_gap: f64,
    ds: f64,
    spacing: Option<&dyn Fn(f64, f64) -> f64>,
) -> PathResult {
    let mut sp = Sp {
        plan,
        res: PathResult::default(),
        z_cut,
        z_entry,
        descent,
        ds,
        outers: Vec::new(),
        islands: Vec::new(),
        cache: HashMap::new(),
        levels: Vec::new(),
    };
    let nseeds = plan.seeds.len();
    sp.res.seeds = nseeds;
    if nseeds == 0 {
        sp.res.notes.push("no seed".into());
        return sp.res;
    }
    let all_seeds: u32 = (1u32 << nseeds) - 1;

    // Walls: each tool-center region's outer wall holds its seeds (the innermost wall loop
    // around them); islands are the seedless wall loops inside an outer wall.
    let all_walls = plan.contours(0.0, true);
    let project = |l: &[P]| -> Vec<P> { l.iter().map(|p| plan.project_to_wall(p.0, p.1)).collect() };
    let mut cand: Vec<usize> = (0..all_walls.len()).filter(|&i| sp.seed_mask(&all_walls[i]) != 0).collect();
    cand.sort_by(|&a, &b| area(&all_walls[a]).abs().partial_cmp(&area(&all_walls[b]).abs()).unwrap_or(std::cmp::Ordering::Equal));
    let mut outer_raw: Vec<usize> = Vec::new();
    for l in cand {
        if outer_raw.iter().all(|&o| sp.seed_mask(&all_walls[o]) & sp.seed_mask(&all_walls[l]) == 0) {
            outer_raw.push(l);
        }
    }
    if outer_raw.is_empty() || outer_raw.iter().fold(0, |m, &l| m | sp.seed_mask(&all_walls[l])) != all_seeds {
        sp.res.notes.push("no wall loop around every seed".into());
        return sp.res;
    }
    sp.outers = outer_raw.iter().map(|&l| (sp.seed_mask(&all_walls[l]), resample(&orient(project(&all_walls[l])), ds * 0.5))).collect();
    sp.res.wall_loop = sp.outers[0].1.clone();
    let mut islands = Vec::new();
    for (i, l) in all_walls.iter().enumerate() {
        if islands.len() == 30 {
            break;
        }
        if outer_raw.contains(&i) || !outer_raw.iter().any(|&o| inside(&all_walls[o], l[0].0, l[0].1)) || sp.seed_mask(l) != 0 {
            continue;
        }
        let wall = resample(&reverse(orient(project(l))), ds * 0.5);
        let cx = l.iter().map(|p| p.0).sum::<f64>() / l.len() as f64;
        let cy = l.iter().map(|p| p.1).sum::<f64>() / l.len() as f64;
        if inside(&wall, cx, cy) {
            islands.push((wall, cx, cy));
        }
    }
    sp.islands = islands;
    sp.res.islands = sp.islands.len();

    // Levels.
    let grad_len = |p: &P| {
        let (gx, gy) = plan.gradient(p.0, p.1);
        (gx * gx + gy * gy).sqrt()
    };
    let mut levels = vec![0.0];
    while levels[levels.len() - 1] < 1.0 && levels.len() < 400 {
        let t = levels[levels.len() - 1];
        let lt = sp.loops(t);
        // T is 0 all over a source, so where the level-0 loop turns a corner of the source's
        // nodes both half-cell samples sit on it and the gradient reads exactly 0: those
        // points are the source's edge, not a front, and do not set the pace.
        let mut gs: Vec<f64> = lt.iter().flat_map(|l| l.pts.iter()).map(grad_len).filter(|&g| g > 0.0).collect();
        if gs.is_empty() {
            break; // no front at this level
        }
        gs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // The slowest 3% (kinks, stadium seams) do not set the pace; the true-spacing check still bounds them.
        let (gmin, gmax) = (gs[(0.03 * (gs.len() - 1) as f64) as usize], gs[gs.len() - 1]);
        let mut dt = (ae_ref * gmax).min(ae_cap * gmin);
        let mut tn = t + dt;
        if let Some(f) = spacing {
            let mut rho: Vec<f64> = lt.iter().flat_map(|l| l.pts.iter()).map(|p| f(p.0, p.1) * grad_len(p)).filter(|&g| g > 0.0).collect();
            rho.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(&r) = rho.get((0.03 * rho.len().saturating_sub(1) as f64) as usize) {
                dt = r;
                tn = t + dt;
            }
        }
        for iter in 0..10 {
            if 1.0 - tn < 0.25 * (tn - t) {
                tn = 1.0; // a small last gap goes to the wall
            }
            let mut spc = Vec::new();
            for lp in sp.loops(tn).iter() {
                let step = 1.max(lp.pts.len() / 160);
                for k in (0..lp.pts.len()).step_by(step) {
                    let q = lp.pts[k];
                    let (px, py) = plan.flow(q.0, q.1, tn.min(1.0 - 1e-6), t, 6, 2.0);
                    // With a target field, spacing is checked against the target at that point (as a multiple of ae_cap).
                    let scale = spacing.map_or(1.0, |f| ae_cap / f(q.0, q.1));
                    spc.push(((px - q.0) * (px - q.0) + (py - q.1) * (py - q.1)).sqrt() * scale);
                }
            }
            if spc.is_empty() {
                tn = 1.0;
                break;
            }
            spc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (p97, max, min) = (spc[(0.97 * (spc.len() - 1) as f64) as usize], spc[spc.len() - 1], spc[0]);
            if (p97 <= ae_cap * 1.05 && max <= (ae_cap * 1.3).min(max_gap)) || iter == 9 {
                let n = sp.loops(tn).len();
                sp.res.trace.push(format!(
                    "t {t:.4} -> {tn:.4}  |grad| {gmin:.4}..{gmax:.4}  spacing {min:.2} / p97 {p97:.2} / max {max:.2}  loops {n}"
                ));
                break;
            }
            tn = t + (tn - t) * (ae_cap / p97).min((ae_cap * 1.3).min(max_gap) / max) * 0.97;
        }
        levels.push(tn);
    }
    sp.res.levels.extend(&levels);
    sp.levels = levels;
    let big_l = sp.levels.len() - 1;

    // Branches: each key's run of consecutive levels.
    let mut branches: Vec<Branch> = Vec::new();
    let mut open: BTreeMap<(bool, u32), usize> = BTreeMap::new();
    for k in 0..=big_l {
        let lk = sp.loops(sp.levels[k]);
        let mut keys: Vec<(bool, u32)> = Vec::new();
        let mut seen: HashSet<(bool, u32)> = HashSet::new();
        for l in lk.iter() {
            if seen.insert((l.outer, l.mask)) {
                keys.push((l.outer, l.mask));
            }
        }
        open.retain(|key, a| {
            if seen.contains(key) {
                return true;
            }
            branches.push(Branch { outer: key.0, mask: key.1, a: *a, b: k - 1 });
            false
        });
        for key in keys {
            open.entry(key).or_insert(k);
        }
    }
    for (key, a) in open {
        branches.push(Branch { outer: key.0, mask: key.1, a, b: big_l });
    }
    sp.res.branches = branches.len();
    let mut by_a: Vec<Branch> = branches.clone();
    by_a.sort_by_key(|b| b.a);
    for b in &by_a {
        sp.res.trace.push(format!(
            "branch {} mask {} levels {}..{} (t {:.3}..{:.3})",
            if b.outer { "outer" } else { "hole" },
            b.mask,
            b.a,
            b.b,
            sp.levels[b.a],
            sp.levels[b.b]
        ));
    }

    // Order: seed branches (nearest next), merged outer branches after their parents,
    // then holes by birth level, larger holes before the ones they split into.
    let mut done: HashSet<usize> = HashSet::new();
    let outer_b: Vec<Branch> = branches.iter().copied().filter(|b| b.outer).collect();
    let index_of = |p: &Branch| outer_b.iter().position(|q| q == p).unwrap_or(usize::MAX);
    while done.len() < outer_b.len() {
        let ready: Vec<(Branch, usize)> = outer_b
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, b)| {
                !done.contains(&i)
                    && outer_b
                        .iter()
                        .enumerate()
                        .filter(|&(pi, p)| pi != i && p.b + 1 == b.a && (p.mask & b.mask) == p.mask && p.mask != b.mask)
                        .all(|(_, p)| done.contains(&index_of(p)))
            })
            .map(|(i, b)| (b, i))
            .collect();
        if ready.is_empty() {
            sp.res.notes.push("outer branches out of order".into());
            break;
        }
        let at = sp.res.points.last().map_or((plan.seeds[0].ax, plan.seeds[0].ay), |p| (p.x, p.y));
        let key2 = |b: &Branch| {
            if b.a != 0 {
                return 0.0;
            }
            let e = &plan.seeds[b.mask.trailing_zeros() as usize];
            (e.ax - at.0) * (e.ax - at.0) + (e.ay - at.1) * (e.ay - at.1)
        };
        let mut pick = ready[0];
        for &c in &ready[1..] {
            if c.0.a < pick.0.a || (c.0.a == pick.0.a && key2(&c.0) < key2(&pick.0)) {
                pick = c;
            }
        }
        sp.run(pick.0);
        done.insert(pick.1);
    }
    let mut holes: Vec<Branch> = branches.iter().copied().filter(|b| !b.outer).collect();
    holes.sort_by(|x, y| x.a.cmp(&y.a).then(y.mask.count_ones().cmp(&x.mask.count_ones())));
    for b in holes {
        sp.run(b);
    }
    if sp.res.dropped > 0 {
        let d = sp.res.dropped;
        sp.res.notes.push(format!("{d} front loops enclose no seed or island (stranded material)"));
    }
    sp.res
}
