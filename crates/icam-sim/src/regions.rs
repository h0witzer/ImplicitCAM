//! The comparison's regions for the panel (ui.rs): the gouges the comparison finds
//! (compare.rs), and the stock it finds left on the part gathered into regions, both at any
//! tolerance from the shop's up.
//!
//! The comparison runs once, over the whole program, on a thread of its own with its own
//! stock and GPU fields. Gouges come from it whole: depth along the part's normal, area,
//! volume and the move that cut. Stock left is gathered as its points pass: into cells CELL mm
//! a side, each with its area by how thick the stock is and the most it holds there. Cells
//! whose stock is thicker than the tolerance that touch (any of 26 neighbors) are one region.
//! The thickness at a point is the stock's field there (how far the point is into the
//! stock, read to LIMIT), a lower bound on the stock along the normal; its area is good to
//! a band of thickness either way (the cells keep areas by band, not by point).

use std::collections::HashMap;
use std::time::Instant;

use icam_core::frep::Frep;
use icam_sim::compare::{analyze, Band, Bands, Gouge};
use icam_sim::part::TOL;
use icam_sweep::gpu::Fields;
use icam_sweep::Stock;

/// Cells' side, mm.
const CELL: f64 = 2.0;
/// The stock's field is read no farther than this (compare.rs): thicker stock reads as this.
pub const LIMIT: f64 = 5.0;
/// Thickness bands' lower edges, mm.
const EDGES: [f64; 8] = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gouge,
    Left,
}

/// A region at some tolerance: how deep it cuts into the part or how thick the stock on it
/// is at most (mm), its area (mm^2), the volume of a gouge, where it is (the area's middle,
/// and its box), and for a gouge the move that cut its deepest point.
#[derive(Clone, Debug)]
pub struct Region {
    pub kind: Kind,
    pub dev: f64,
    pub area: f64,
    pub volume: Option<f64>,
    pub center: [f64; 3],
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    pub by: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct Cell {
    /// Area by thickness band (EDGES).
    area: [f64; 8],
    most: f64,
    /// Points weighted by area, for the middle.
    sum: [f64; 3],
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Cell {
    fn add(&mut self, p: [f64; 3], w: f64, thick: f64) {
        if self.area.iter().all(|&a| a == 0.0) {
            (self.lo, self.hi) = (p, p);
        }
        let band = EDGES.iter().rposition(|&e| thick > e).unwrap_or(0);
        self.area[band] += w;
        self.most = self.most.max(thick);
        for (a, &x) in p.iter().enumerate() {
            self.sum[a] += w * x;
            self.lo[a] = self.lo[a].min(x);
            self.hi[a] = self.hi[a].max(x);
        }
    }

    /// Area of the bands that reach past t.
    fn over(&self, t: f64) -> f64 {
        (0..8).filter(|&b| EDGES.get(b + 1).is_none_or(|&up| up > t)).map(|b| self.area[b]).sum()
    }
}

/// What the comparison found, kept so the regions can be read at any tolerance.
pub struct Report {
    pub bands: Bands,
    pub gouges: Vec<Gouge>,
    pub points: usize,
    pub step: f64,
    pub secs: f64,
    /// Cells of stock left, and of gouged points (for each gouge's box).
    left: HashMap<[i64; 3], Cell>,
    gouged: HashMap<[i64; 3], Cell>,
}

fn key(p: [f64; 3]) -> [i64; 3] {
    std::array::from_fn(|a| (p[a] / CELL).floor() as i64)
}

/// The cells in `cells` that `keep` keeps, in groups that touch.
type Group<'a> = Vec<(&'a [i64; 3], &'a Cell)>;

fn groups(cells: &HashMap<[i64; 3], Cell>, keep: impl Fn(&Cell) -> bool) -> Vec<Group<'_>> {
    let mut seen: HashMap<[i64; 3], bool> = cells.iter().filter(|(_, c)| keep(c)).map(|(k, _)| (*k, false)).collect();
    let mut out = Vec::new();
    let starts: Vec<[i64; 3]> = seen.keys().copied().collect();
    for k0 in starts {
        if seen[&k0] {
            continue;
        }
        seen.insert(k0, true);
        let (mut stack, mut group) = (vec![k0], Vec::new());
        while let Some(k) = stack.pop() {
            let (kk, c) = cells.get_key_value(&k).expect("a kept cell");
            group.push((kk, c));
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let n = [k[0] + dx, k[1] + dy, k[2] + dz];
                        if seen.get(&n) == Some(&false) {
                            seen.insert(n, true);
                            stack.push(n);
                        }
                    }
                }
            }
        }
        out.push(group);
    }
    out
}

impl Report {
    /// The regions past tolerance t: gouges deeper than t, deepest first, then stock left
    /// thicker than t, thickest first.
    pub fn regions(&self, t: f64) -> Vec<Region> {
        let mut out = Vec::new();
        // Each gouge's box: the group of gouged cells its deepest point lies in.
        let gouge_groups = groups(&self.gouged, |_| true);
        let mut group_of: HashMap<[i64; 3], usize> = HashMap::new();
        for (i, g) in gouge_groups.iter().enumerate() {
            for (k, _) in g {
                group_of.insert(**k, i);
            }
        }
        let bounds = |g: &[(&[i64; 3], &Cell)]| {
            g.iter().fold(([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]), |(lo, hi), (_, c)| (std::array::from_fn(|a| lo[a].min(c.lo[a])), std::array::from_fn(|a| hi[a].max(c.hi[a]))))
        };
        for g in self.gouges.iter().filter(|g| g.depth > t) {
            let (lo, hi) = group_of.get(&key(g.at)).map_or((g.at, g.at), |&i| bounds(&gouge_groups[i]));
            out.push(Region { kind: Kind::Gouge, dev: g.depth, area: g.area, volume: Some(g.volume), center: g.at, lo, hi, by: g.by });
        }
        let mut left: Vec<Region> = groups(&self.left, |c| c.most > t)
            .into_iter()
            .map(|g| {
                let area: f64 = g.iter().map(|(_, c)| c.over(t)).sum();
                let total: f64 = g.iter().map(|(_, c)| c.area.iter().sum::<f64>()).sum();
                let sum = g.iter().fold([0.0; 3], |s, (_, c)| std::array::from_fn(|a| s[a] + c.sum[a]));
                let (lo, hi) = bounds(&g);
                let dev = g.iter().map(|(_, c)| c.most).fold(0.0, f64::max);
                Region { kind: Kind::Left, dev, area, volume: None, center: std::array::from_fn(|a| sum[a] / total.max(1e-12)), lo, hi, by: None }
            })
            .collect();
        left.sort_by(|a, b| b.dev.total_cmp(&a.dev).then(b.area.total_cmp(&a.area)));
        out.extend(left);
        out
    }
}

/// The comparison over the part's box [lo, hi] against the whole program's stock, points
/// `step` apart; `say` hears its stages.
pub fn run(frep: &Frep, lo: [f64; 3], hi: [f64; 3], stock: &Stock, gpu: &Fields, step: f64, say: &dyn Fn(&str)) -> Result<Report, String> {
    let clock = Instant::now();
    let (mut left, mut gouged): (HashMap<[i64; 3], Cell>, HashMap<[i64; 3], Cell>) = (HashMap::new(), HashMap::new());
    let mut visit = |samples: &[icam_sim::compare::Sample], values: &[f64], bands: &[Band]| {
        for ((s, &v), &b) in samples.iter().zip(values).zip(bands) {
            match b {
                Band::Left => left.entry(key(s.p)).or_default().add(s.p, s.w as f64, (-v).max(TOL)),
                Band::Gouged => gouged.entry(key(s.p)).or_default().add(s.p, s.w as f64, v.max(0.0)),
                Band::OnSize => {}
            }
        }
    };
    let a = analyze(frep, lo, hi, stock, gpu, stock.len(), step, false, say, &mut visit)?;
    Ok(Report { bands: a.bands, gouges: a.gouges, points: a.points, step, secs: clock.elapsed().as_secs_f64(), left, gouged })
}
