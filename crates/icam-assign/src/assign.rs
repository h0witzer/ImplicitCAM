//! Give every material cell of a layer to one tool, and time every way with the same
//! model.
//!
//!   rest      the usual rest roughing: the biggest tool takes everything it reaches,
//!             each smaller tool takes what no bigger tool reached
//!   cheapest  every cell goes to the reachable tool with the lowest s/mm2 at that cell,
//!             then components smaller than D_k^2 are merged into a neighboring tool
//!             that reaches them (no speckle)
//!   cheapest+ the cheapest assignment, then any tool whose share does not pay for its
//!             entries (in a pocket) or its tool change (in the nest) is folded into the
//!             other tools (consolidate_nest)
//!
//! Penalties (explicit, labeled assumptions, the same for every strategy):
//!   tool change  t_change s once per tool used anywhere in the nest
//!   entry        per connected component of a tool's cells, cut in tool order biggest
//!                first: a LINK (t_link s: stop, lift, lower in cleared space) when the
//!                tool can drop in beside it in stock already cleared to the floor, else
//!                a HELIX (t_link plus a ramp_deg helix down the component's median height
//!                at the tool's slot feed for a helix pitch of depth); plus the travel of a
//!                nearest-neighbor tour of the components at rapid

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::cost::CostTable;
use crate::layer::Layer;

#[derive(Clone, Debug)]
pub struct Penalties {
    /// s, assumed (the router's tool change time is not on disk).
    pub t_change: f64,
    /// s per link, assumed: stop, lift clear of the cleared floor, lower again.
    pub t_link: f64,
    /// Travel between a tool's components, mm/s (assumed 500 ipm).
    pub rapid_mmps: f64,
    /// Helical entry ramp, deg (assumed).
    pub ramp_deg: f64,
    /// Helix diameter / tool diameter (assumed).
    pub helix_dia_d: f64,
}

impl Default for Penalties {
    fn default() -> Penalties {
        Penalties { t_change: 15.0, t_link: 0.5, rapid_mmps: 500.0 * 25.4 / 60.0, ramp_deg: 3.0, helix_dia_d: 1.2 }
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub name: String,
    /// Per cell: tool index, -1 = not material, -2 = no tool reaches.
    pub tool: Vec<i32>,
    pub area: Vec<f64>,
    pub vol: Vec<f64>,
    pub cut: Vec<f64>,
    pub entry_t: Vec<f64>,
    pub helix: Vec<i32>,
    pub links: Vec<i32>,
    pub tools_used: usize,
    pub cut_total: f64,
    pub entry_total: f64,
    pub change_total: f64,
    pub uncut_area: f64,
    pub uncut_vol: f64,
}

impl Plan {
    /// What the assignment changes inside the pocket.
    pub fn local(&self) -> f64 {
        self.cut_total + self.entry_total
    }

    /// The pocket cut alone, with its own tool changes.
    pub fn total(&self) -> f64 {
        self.cut_total + self.entry_total + self.change_total
    }
}

/// s/mm2 per tool per cell (infinite where the tool does not reach).
pub fn costs(l: &Layer, tabs: &[CostTable]) -> Vec<Vec<f64>> {
    tabs.iter()
        .enumerate()
        .map(|(t, tab)| {
            (0..l.g.n())
                .into_par_iter()
                .map(|k| if l.mat[k] && l.reach[t][k] { tab.at((l.w[t][k] as f64).max(tab.tool.d()), l.hgt[k] as f64) } else { f64::INFINITY })
                .collect()
        })
        .collect()
}

/// Tools are ordered biggest first in every list here.
pub fn rest(l: &Layer, cost: &[Vec<f64>], tabs: &[CostTable], p: &Penalties) -> Plan {
    let tool: Vec<i32> = (0..l.g.n())
        .map(|k| if !l.mat[k] { -1 } else { (0..cost.len()).find(|&t| !cost[t][k].is_infinite()).map_or(-2, |t| t as i32) })
        .collect();
    score("rest (big tool first)", l, tool, cost, tabs, p)
}

pub fn cheapest(l: &Layer, cost: &[Vec<f64>], tabs: &[CostTable], p: &Penalties) -> Plan {
    let mut tool: Vec<i32> = (0..l.g.n())
        .map(|k| {
            if !l.mat[k] {
                return -1;
            }
            let (mut best, mut bc) = (-2, f64::INFINITY);
            for (t, c) in cost.iter().enumerate() {
                if c[k] < bc {
                    bc = c[k];
                    best = t as i32;
                }
            }
            best
        })
        .collect();
    despeckle(l, &mut tool, cost, tabs, p);
    score("cheapest per point", l, tool, cost, tabs, p)
}

/// Nest level: each tool is changed once for the whole nest (its ops grouped). From the
/// cheapest plans, repeatedly apply the single best move that lowers
///   sum over pockets (cut + entries) + t_change x (tools used anywhere):
///   (a) drop a tool from the whole nest, or (b) drop a tool from one pocket,
/// reassigning its cells to their cheapest remaining tool (only when every cell has one).
/// A cheap lower bound on the cut-time increase prunes moves that cannot pay before they
/// are scored.
pub fn consolidate_nest(ls: &[&Layer], costs: &[Vec<Vec<f64>>], start: &[Plan], tabs: &[CostTable], p: &Penalties) -> Vec<Plan> {
    let (nl, nt) = (ls.len(), tabs.len());
    let mut plans: Vec<Plan> = start.to_vec();
    enum Move {
        Nest(Vec<Option<Plan>>),
        One(usize, Plan),
    }
    for _ in 0..3 * nt {
        let used: Vec<bool> = (0..nt).map(|t| plans.iter().any(|pl| pl.area[t] > 0.0)).collect();
        // Reassign tool u's cells in pocket i; also returns the cut-time increase.
        let fold = |i: usize, u: usize, allowed: &[bool]| -> Option<(Vec<i32>, f64)> {
            let (mut tool, c, ca) = (plans[i].tool.clone(), &costs[i], ls[i].cell_area());
            let mut d = 0.0;
            for k in 0..tool.len() {
                if tool[k] != u as i32 {
                    continue;
                }
                let (mut b, mut bc) = (None, f64::INFINITY);
                for t in 0..nt {
                    if t != u && allowed[t] && c[t][k] < bc {
                        bc = c[t][k];
                        b = Some(t);
                    }
                }
                let b = b?;
                tool[k] = b as i32;
                d += (bc - c[u][k]) * ca;
            }
            Some((tool, d))
        };
        let mut best_gain = 1e-6;
        let mut apply: Option<Move> = None;
        for u in 0..nt {
            if !used[u] {
                continue;
            }
            let mut allowed = used.clone();
            allowed[u] = false;
            // (a) the whole nest
            let mut folds: Vec<Option<Vec<i32>>> = vec![None; nl];
            let (mut ok, mut dcut, mut entries) = (true, 0.0, 0.0);
            for i in 0..nl {
                if !ok {
                    break;
                }
                if plans[i].area[u] <= 0.0 {
                    continue;
                }
                match fold(i, u, &allowed) {
                    Some((tool, dc)) => {
                        folds[i] = Some(tool);
                        dcut += dc;
                    }
                    None => {
                        ok = false;
                        dcut += f64::INFINITY;
                    }
                }
                entries += plans[i].entry_total;
            }
            if ok && p.t_change + entries - dcut > best_gain {
                let mut np: Vec<Option<Plan>> = vec![None; nl];
                let mut gain = p.t_change;
                for i in 0..nl {
                    if let Some(tool) = &folds[i] {
                        let s = score(&plans[i].name, ls[i], tool.clone(), &costs[i], tabs, p);
                        gain += plans[i].local() - s.local();
                        np[i] = Some(s);
                    }
                }
                if gain > best_gain {
                    best_gain = gain;
                    apply = Some(Move::Nest(np));
                }
            }
            // (b) one pocket at a time
            for i in 0..nl {
                if plans[i].area[u] <= 0.0 {
                    continue;
                }
                let last = (0..nl).all(|j| j == i || plans[j].area[u] <= 0.0);
                let Some((tool, dc)) = fold(i, u, &allowed) else { continue };
                let change = if last { p.t_change } else { 0.0 };
                if plans[i].entry_total + change - dc <= best_gain {
                    continue;
                }
                let np = score(&plans[i].name, ls[i], tool, &costs[i], tabs, p);
                let gain = plans[i].local() - np.local() + change;
                if gain > best_gain {
                    best_gain = gain;
                    apply = Some(Move::One(i, np));
                }
            }
        }
        match apply {
            None => break,
            Some(Move::Nest(np)) => {
                for (i, pl) in np.into_iter().enumerate() {
                    if let Some(pl) = pl {
                        plans[i] = pl;
                    }
                }
            }
            Some(Move::One(i, pl)) => plans[i] = pl,
        }
    }
    plans.iter().enumerate().map(|(i, pl)| score("cheapest, tool changes paid for", ls[i], pl.tool.clone(), &costs[i], tabs, p)).collect()
}

/// Nest total: cut + entries over the pockets, plus one change per tool used anywhere.
pub fn nest_total(plans: &[&Plan], p: &Penalties) -> f64 {
    if plans.is_empty() {
        return 0.0;
    }
    let nt = plans[0].area.len();
    let mut s = 0.0;
    for x in plans {
        s += x.local();
    }
    s + p.t_change * (0..nt).filter(|&t| plans.iter().any(|x| x.area[t] > 0.0)).count() as f64
}

/// Components of one tool smaller than D^2 join the cheapest adjacent tool that reaches
/// all their cells, when that costs less than the link it saves.
fn despeckle(l: &Layer, tool: &mut [i32], cost: &[Vec<f64>], tabs: &[CostTable], p: &Penalties) {
    let g = &l.g;
    for _ in 0..2 {
        for t in 0..cost.len() {
            let mask: Vec<bool> = tool.iter().map(|&v| v == t as i32).collect();
            let (lab, count) = g.components(&mask);
            if count == 0 {
                continue;
            }
            let mut cells: Vec<Vec<usize>> = vec![Vec::new(); count];
            for k in 0..g.n() {
                if lab[k] >= 0 {
                    cells[lab[k] as usize].push(k);
                }
            }
            let min_area = tabs[t].tool.d() * tabs[t].tool.d();
            for cs in &cells {
                if cs.len() as f64 * l.cell_area() >= min_area {
                    continue;
                }
                // Neighboring tools, in the order first met.
                let mut adj: Vec<usize> = Vec::new();
                for &k in cs {
                    let (i, j) = ((k % g.nx) as i64, (k / g.nx) as i64);
                    for dj in -1..=1 {
                        for di in -1..=1 {
                            if !g.contains(i + di, j + dj) {
                                continue;
                            }
                            let q = tool[((j + dj) as usize) * g.nx + (i + di) as usize];
                            if q >= 0 && q != t as i32 && !adj.contains(&(q as usize)) {
                                adj.push(q as usize);
                            }
                        }
                    }
                }
                let (mut best, mut best_add) = (None, f64::INFINITY);
                for &u in &adj {
                    let mut add = 0.0;
                    for &k in cs {
                        add += (cost[u][k] - cost[t][k]) * l.cell_area();
                    }
                    if add < best_add {
                        best_add = add;
                        best = Some(u);
                    }
                }
                // Merge only when it pays: the speck's own entry (a link) costs more than the extra cut time.
                if let Some(b) = best {
                    if best_add <= p.t_link {
                        for &k in cs {
                            tool[k] = b as i32;
                        }
                    }
                }
            }
        }
    }
}

pub fn score(name: &str, l: &Layer, tool: Vec<i32>, cost: &[Vec<f64>], tabs: &[CostTable], p: &Penalties) -> Plan {
    let g = &l.g;
    let nt = cost.len();
    let mut pl = Plan {
        name: name.into(),
        tool,
        area: vec![0.0; nt],
        vol: vec![0.0; nt],
        cut: vec![0.0; nt],
        entry_t: vec![0.0; nt],
        helix: vec![0; nt],
        links: vec![0; nt],
        tools_used: 0,
        cut_total: 0.0,
        entry_total: 0.0,
        change_total: 0.0,
        uncut_area: 0.0,
        uncut_vol: 0.0,
    };
    let ca = l.cell_area();
    for k in 0..g.n() {
        let t = pl.tool[k];
        if t == -2 {
            pl.uncut_area += ca;
            pl.uncut_vol += l.hgt[k] as f64 * ca;
            continue;
        }
        if t < 0 {
            continue;
        }
        let t = t as usize;
        pl.area[t] += ca;
        pl.vol[t] += l.hgt[k] as f64 * ca;
        pl.cut[t] += cost[t][k] * ca;
    }
    // Entries, biggest tool first; "cleared" = pocket cells already at the floor.
    let mut cleared: Vec<bool> = (0..g.n()).map(|k| l.pocket[k] && !l.mat[k]).collect();
    let tan_ramp = (p.ramp_deg * PI / 180.0).tan();
    for t in 0..nt {
        if pl.area[t] <= 0.0 {
            continue;
        }
        let ts = &tabs[t].tool;
        let mask: Vec<bool> = pl.tool.iter().map(|&v| v == t as i32).collect();
        let (lab, count) = g.components(&mask);
        // A component can be entered from the side (a link) only if the tool can plunge
        // next to it in cleared space: a center whose whole disk is cleared, within one
        // radius of the component.
        let not_cleared: Vec<bool> = cleared.iter().map(|&c| !c).collect();
        let d_free = g.edt(&not_cleared);
        let fit: Vec<bool> = (0..g.n()).map(|k| cleared[k] && d_free[k] as f64 >= ts.r() - 0.5 * g.h).collect();
        let d_fit = g.edt(&fit);
        let mut touches = vec![false; count];
        let mut heights: Vec<Vec<f32>> = vec![Vec::new(); count];
        let (mut cx, mut cy, mut cn) = (vec![0.0; count], vec![0.0; count], vec![0usize; count]);
        for k in 0..g.n() {
            if lab[k] < 0 {
                continue;
            }
            let c = lab[k] as usize;
            heights[c].push(l.hgt[k]);
            cx[c] += g.x(k % g.nx);
            cy[c] += g.y(k / g.nx);
            cn[c] += 1;
            if !touches[c] && d_fit[k] as f64 <= ts.r() + 1.5 * g.h {
                touches[c] = true;
            }
        }
        // Travel between the tool's components: a nearest-neighbor tour of their centroids at rapid.
        if count > 1 {
            for c in 0..count {
                cx[c] /= cn[c] as f64;
                cy[c] /= cn[c] as f64;
            }
            let mut seen = vec![false; count];
            seen[0] = true;
            let (mut at, mut tour) = (0, 0.0);
            for _ in 1..count {
                let (mut nb, mut bd) = (0, f64::INFINITY);
                for c in 0..count {
                    if !seen[c] {
                        let d = ((cx[c] - cx[at]) * (cx[c] - cx[at]) + (cy[c] - cy[at]) * (cy[c] - cy[at])).sqrt();
                        if d < bd {
                            bd = d;
                            nb = c;
                        }
                    }
                }
                seen[nb] = true;
                at = nb;
                tour += bd;
            }
            pl.entry_t[t] += tour / p.rapid_mmps;
        }
        for c in 0..count {
            if touches[c] {
                pl.links[t] += 1;
                pl.entry_t[t] += p.t_link;
                continue;
            }
            // Helix: t_link plus a ramp_deg helix down the component's median height, fed
            // at the tool's slot ceiling for a pass one helix pitch deep.
            let hs = &mut heights[c];
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let h = hs[hs.len() / 2] as f64;
            let pitch = PI * p.helix_dia_d * ts.d() * tan_ramp;
            let v = ts.v(PI, pitch.min(ts.model.loc));
            pl.helix[t] += 1;
            pl.entry_t[t] += p.t_link + h / tan_ramp / v.max(1e-3);
        }
        for k in 0..g.n() {
            if pl.tool[k] == t as i32 {
                cleared[k] = true;
            }
        }
    }
    pl.tools_used = pl.area.iter().filter(|&&a| a > 0.0).count();
    pl.cut_total = pl.cut.iter().fold(0.0, |s, &v| s + v);
    pl.entry_total = pl.entry_t.iter().fold(0.0, |s, &v| s + v);
    pl.change_total = pl.tools_used as f64 * p.t_change;
    pl
}
