//! The part's field cached for drawing, as the stock's is (icam_sweep::cache): bricks of
//! 8 x 8 x 8 cells, on the same grid as a level of the stock's cache, with values where the
//! part's surface can be, ray-cast by the same caster in the part's view (cache.wgsl), which
//! colors each hit by the stock's field there.
//!
//! The prismatic part is f = min_k max(sd_k(x, y), z - top_k, bottom_k - z) (icam_core::frep).
//! A box is decided whole by the field's certified bounds over it (`Frep::interval`): past a
//! cell's diagonal either way it holds no surface, air or material; otherwise it splits,
//! down to bricks, whose 9 x 9 x 9 nodes are read: each column's section distance at the
//! brick's 9 x 9 nodes in x and y, its planes in z. Values are clamped to a cell's diagonal
//! and stored as the stock's are (F over the clamp, 16-bit, two to a word). The part does
//! not move, so a brick is read once and kept for every placement of its level.
//!
//! Any other solid is read by its exact signed distance (`icam_core::solid3::Exact`, the
//! field itself: decision 3). A box's bounds are the distance at its center plus and minus
//! half its diagonal, which the 1-Lipschitz field guarantees; an answer the evaluator marks
//! unsure decides nothing, so the box splits. A brick's nodes are each read with the bound
//! the brick's center gives them (its distance plus how far the node is), so only features
//! that near are looked at.
//!
//! With the GPU kernel (partgpu.rs), a brick whose every candidate feature it reads has
//! all its nodes read there at once. A node it flags as known only to lie farther than
//! the clamp takes its sign from a neighbor (a node within a cell diagonal of it, which no
//! surface separates from it), by flood fill from the nodes it read. A node it could not
//! decide is bridged from its neighbors for the first picture: their mean, held within
//! the bracket each decided neighbor gives it (the field is 1-Lipschitz), and kept as
//! provisional; `settle` later reads every provisional node exactly (the GPU, a thread a
//! node, refining the leaves it needs; the CPU for what it leaves) and the level is sent
//! again. Any other brick is read on the CPU as above. Boxes are decided on the GPU too, a thread a box center,
//! once each has a bound on its distance from its parent's (the first boxes, which have
//! none, on the CPU), and a brick's center distance from its box serves its nodes.

use std::collections::HashMap;

use icam_core::frep::Frep;
use icam_core::solid3::Exact;

use crate::partgpu::PartGpu;
use icam_core::geom::V2;
use icam_sim::part::{self, Column};
use icam_sweep::cache::{AIR, SOLID, WORDS};
use icam_sweep::V3;
use rayon::prelude::*;

/// Exact queries made (for `part-bench`).
pub static QUERIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bricks read on the GPU, and their nodes the CPU read again (for `part-bench`).
pub static GPU_BRICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static GPU_REREADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// For `part-bench`: per brick with values, the features that can be nearest in it, by
/// kind (`Exact::near_kinds`), summed, and how many bricks had any B-spline face or other
/// edge among them, and the most features in one brick.
pub static CENSUS: std::sync::Mutex<([u64; 5], u64, u64, u64)> = std::sync::Mutex::new(([0; 5], 0, 0, 0));
/// Whether the census is taken (only `part-bench` turns it on), and whether it is all a
/// level does (`part-bench --census`: bricks are counted, not read).
pub static CENSUS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static CENSUS_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// For `part-bench`: the CPU's exact reads by where they come from (box decisions with no
/// bound yet, box centers the GPU left, bricks the GPU cannot list, brick centers, settle
/// centers, settle nodes), and the time of each step of a level in microseconds (box
/// decisions, candidate lists, the GPU's bricks, bricks finished on the CPU, and within the
/// box decisions their candidate lists and their GPU reads); of the box
/// centers the GPU left, those it knew only to lie past the box (6) and those it could not
/// list (7); of the settle's nodes, those it could not list (8) and those the GPU left
/// undecided (9).
pub static CPU_AT: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];
pub static STEP_US: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// For `part-bench`: why the GPU left the settle's nodes it left (the kernel's codes).
pub static SETTLE_WHY: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];

fn tally(at: &std::sync::atomic::AtomicU64, n: u64) {
    at.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

fn since(at: &std::sync::atomic::AtomicU64, t: std::time::Instant) {
    tally(at, t.elapsed().as_micros() as u64);
}

/// `since` when dropped.
struct Since<'a>(&'a std::sync::atomic::AtomicU64, std::time::Instant);

impl Drop for Since<'_> {
    fn drop(&mut self) {
        since(self.0, self.1);
    }
}

/// Bricks kept, at most, before the store starts again.
const KEPT: usize = 1 << 19;

/// A box of bricks being decided: its first brick, a bound on the distance at its center,
/// its parent's signed distance at the parent's center (NaN when not known) and how far
/// that center is from its own.
type Boxed = ([i64; 3], f64, f64, f64);

/// A brick of the part: all air, all material, or its words.
#[derive(Clone)]
enum Brick {
    Air,
    Solid,
    Words(Box<[u32]>),
}

/// A level of the part's cache: the brick map (x fastest; AIR, SOLID or a slot) and the
/// slots' words, slots in brick order.
pub struct PartLevel {
    pub map: Vec<u32>,
    pub atlas: Vec<u32>,
}

/// A level `PartCache::build` made, not yet kept: its exponent, layout and bricks read.
pub struct Built {
    e: i32,
    layout: Layout,
    bricks: Vec<((i32, [i64; 3]), Brick, Option<Provisional>)>,
}

impl Built {
    /// Its nodes bridged until settled.
    pub fn bridged(&self) -> usize {
        self.bricks.iter().filter_map(|b| b.2.as_ref()).map(|p| p.nodes.len()).sum()
    }
}

/// A level laid out by `layout`, its bricks by name from `brick` (None when one is missing).
fn assemble<'a>(layout: &Layout, brick: impl Fn(&[i64; 3]) -> Option<&'a Brick>) -> Option<PartLevel> {
    let mut map = layout.map.clone();
    let mut atlas = Vec::new();
    for &(i, name) in &layout.read {
        match brick(&name)? {
            Brick::Air => map[i] = AIR,
            Brick::Solid => map[i] = SOLID,
            Brick::Words(w) => {
                map[i] = (atlas.len() / WORDS) as u32;
                atlas.extend_from_slice(w);
            }
        }
    }
    Some(PartLevel { map, atlas })
}

/// Where the part's field comes from.
enum Source {
    /// A prismatic part: its columns.
    Columns { frep: Frep, columns: Vec<Column> },
    /// Any solid: its exact signed distance.
    Exact(Exact<'static>),
}

pub struct PartCache {
    source: Source,
    /// The stock box's low corner: bricks are named by their first node's cells from it.
    origin: V3,
    kept: HashMap<(i32, [i64; 3]), Brick>,
    /// The exact source's kernel, when there is a GPU to run it.
    gpu: Option<PartGpu>,
    /// Bricks with nodes bridged, not read: their first node, the distance at their center
    /// (NaN when not known), their values, and those nodes.
    provisional: HashMap<(i32, [i64; 3]), Provisional>,
    /// Each level's last layout, so it can be sent again once settled.
    layouts: HashMap<i32, Layout>,
    /// Each box's bounds and center distance once decided, by level, size and first brick on
    /// the grid every placement of the level shares, so a level placed anew decides only the
    /// boxes it has not seen.
    decided: std::sync::Mutex<HashMap<(i32, i64, [i64; 3]), (f64, f64, f64)>>,
}

struct Provisional {
    p: V3,
    d0: f64,
    q: [i16; 729],
    nodes: Vec<u16>,
}

/// A level's brick map with its decided bricks filled in, and its bricks with values
/// (by name, in map order with their map index).
struct Layout {
    map: Vec<u32>,
    read: Vec<(usize, [i64; 3])>,
}

impl PartCache {
    /// None when the part is not a union of columns (only prismatic parts so far).
    pub fn new(frep: Frep, origin: V3) -> Option<PartCache> {
        let columns = part::columns(&frep);
        (!columns.is_empty()).then(|| PartCache { source: Source::Columns { frep, columns }, origin, kept: HashMap::new(), gpu: None, provisional: HashMap::new(), layouts: HashMap::new(), decided: Default::default() })
    }

    /// Any solid, by its exact signed distance.
    pub fn exact(ex: Exact<'static>, origin: V3) -> PartCache {
        PartCache { source: Source::Exact(ex), origin, kept: HashMap::new(), gpu: None, provisional: HashMap::new(), layouts: HashMap::new(), decided: Default::default() }
    }

    /// The same, reading bricks on the GPU where its kernel can (the CPU alone when there is
    /// no GPU, said once); `finest` is the finest cell side the cache will be asked for.
    pub fn exact_gpu(ex: Exact<'static>, origin: V3, finest: f64) -> PartCache {
        let t = std::time::Instant::now();
        // Leaves packed for cells no finer than a millimeter (a finer level's nodes refine
        // the leaves they need on the GPU): a few hundred megabytes at most on THE PART,
        // where packing for a quarter millimeter took two gigabytes.
        let gpu = match PartGpu::new(&ex, 4.0 * finest.max(1.0) * 3f64.sqrt()) {
            Ok(g) => {
                println!("part cache: reading bricks on {} ({} sub-patches and pieces, packed in {:.1} s)", g.adapter, g.nodes, t.elapsed().as_secs_f64());
                Some(g)
            }
            Err(e) => {
                println!("part cache: no GPU kernel ({e}); reading every brick on the CPU");
                None
            }
        };
        PartCache { source: Source::Exact(ex), origin, kept: HashMap::new(), gpu, provisional: HashMap::new(), layouts: HashMap::new(), decided: Default::default() }
    }

    /// Keeps the GPU's clocks up for `secs` (`PartGpu::warm`), when there is a GPU.
    pub fn warm(&self, secs: f64) {
        if let Some(g) = &self.gpu {
            g.warm(secs);
        }
    }

    /// The level's brick map with every brick decided whole (AIR or SOLID), and the bricks
    /// whose nodes must be read, each with the signed distance at its center when known (NaN when
    /// not).
    fn decide(&self, h: f64, lo: V3, nb: [i64; 3]) -> (Vec<u32>, Vec<([i64; 3], f64)>) {
        let side = 8.0 * h;
        let clamp = h * 3f64.sqrt();
        let first = |b: [i64; 3]| lo + V3::new(side * b[0] as f64, side * b[1] as f64, side * b[2] as f64);
        // The box of bricks [a, a + n) grown by the clamp: its corners.
        let grown = |a: [i64; 3], n: i64| (first(a) - V3::new(clamp, clamp, clamp), first([a[0] + n, a[1] + n, a[2] + n]) + V3::new(clamp, clamp, clamp));
        let center = |a: [i64; 3], n: i64| {
            let (p, q) = grown(a, n);
            V3::new(0.5 * (p.x + q.x), 0.5 * (p.y + q.y), 0.5 * (p.z + q.z))
        };
        let mut map = vec![AIR; (nb[0] * nb[1] * nb[2]) as usize];
        let at = |b: [i64; 3]| (b[0] + nb[0] * (b[1] + nb[1] * b[2])) as usize;
        let mut size = 1i64;
        while size < *nb.iter().max().unwrap_or(&1) {
            size *= 2;
        }
        // The level's first brick on the grid every placement of it shares (bricks from a
        // cell below the box's corner, as icam_sweep::cache::place puts them): boxes sit on
        // that grid, so another placement meets the same boxes and takes what they had.
        let e = h.log2().round() as i32;
        let o = self.origin - V3::new(h, h, h);
        let g0 = [((lo.x - o.x) / side).round() as i64, ((lo.y - o.y) / side).round() as i64, ((lo.z - o.z) / side).round() as i64];
        let start = |k: usize| g0[k].div_euclid(size) * size - g0[k];
        // Coarse to fine: boxes of `size` bricks, split while the bounds hold surface, each
        // with a bound on the distance at its center (from its parent's; none at first), and
        // its parent's signed distance at its center and how far that center is (NaN and
        // infinity at first).
        let mut boxes: Vec<Boxed> = Vec::new();
        for z in (start(2)..nb[2]).step_by(size as usize) {
            for y in (start(1)..nb[1]).step_by(size as usize) {
                for x in (start(0)..nb[0]).step_by(size as usize) {
                    boxes.push(([x, y, z], f64::INFINITY, f64::NAN, f64::INFINITY));
                }
            }
        }
        let mut read: Vec<([i64; 3], f64)> = Vec::new();
        while !boxes.is_empty() {
            let key = |a: [i64; 3]| (e, size, [a[0] + g0[0], a[1] + g0[1], a[2] + g0[2]]);
            // The boxes decided before, by any placement; the rest decided now and kept.
            let known: Vec<Option<(f64, f64, f64)>> = {
                let memo = self.decided.lock().unwrap_or_else(|p| p.into_inner());
                boxes.iter().map(|b| memo.get(&key(b.0)).copied()).collect()
            };
            let todo: Vec<Boxed> = boxes.iter().zip(&known).filter(|(_, k)| k.is_none()).map(|(b, _)| *b).collect();
            let fresh = self.box_bounds(&todo, size, &grown, clamp);
            {
                let mut memo = self.decided.lock().unwrap_or_else(|p| p.into_inner());
                for (b, &r) in todo.iter().zip(&fresh) {
                    memo.insert(key(b.0), r);
                }
            }
            // ICAM_DECIDE_CHECK: each box decided whole read on the CPU at its center, its side
            // compared (`verify` reads only the bricks read, never the boxes decided).
            if std::env::var_os("ICAM_DECIDE_CHECK").is_some() {
                if let Source::Exact(ex) = &self.source {
                    let bad: Vec<String> = todo
                        .par_iter()
                        .zip(&fresh)
                        .filter(|(_, r)| r.0 > clamp || r.1 < -clamp)
                        .filter_map(|(b, r)| {
                            let c = center(b.0, size);
                            let t = ex.sdf(core(c)).0.d;
                            let air = r.0 > clamp;
                            (t.abs() > 1e-6 && (t > 0.0) != air).then(|| format!("decide check: e {e} size {size} box {:?}: decided {}, true {t:.5}, bounds ({:.5}, {:.5}), d {:.5}", b.0, if air { "air" } else { "material" }, r.0, r.1, r.2))
                        })
                        .collect();
                    for s in bad.iter().take(10) {
                        eprintln!("{s}");
                    }
                    eprintln!("decide check: e {e} size {size}: {} boxes decided whole, {} on the wrong side", todo.iter().zip(&fresh).filter(|(_, r)| r.0 > clamp || r.1 < -clamp).count(), bad.len());
                }
            }
            let mut fresh = fresh.into_iter();
            let decided: Vec<(f64, f64, f64)> = known.iter().map(|k| k.unwrap_or_else(|| fresh.next().unwrap_or((f64::NEG_INFINITY, f64::INFINITY, f64::NAN)))).collect();
            let mut next = Vec::new();
            for (&(a, ..), &(lo_f, hi_f, d)) in boxes.iter().zip(&decided) {
                if lo_f > clamp || hi_f < -clamp {
                    let v = if lo_f > clamp { AIR } else { SOLID };
                    for z in a[2].max(0)..(a[2] + size).min(nb[2]) {
                        for y in a[1].max(0)..(a[1] + size).min(nb[1]) {
                            for x in a[0].max(0)..(a[0] + size).min(nb[0]) {
                                map[at([x, y, z])] = v;
                            }
                        }
                    }
                } else if size == 1 {
                    read.push((a, d));
                } else {
                    let half = size / 2;
                    let c = center(a, size);
                    for q in 0..8 {
                        let k = [a[0] + half * (q & 1), a[1] + half * ((q >> 1) & 1), a[2] + half * ((q >> 2) & 1)];
                        // Parts of the box that reach into the level.
                        if (0..3).all(|i| k[i] < nb[i] && k[i] + half > 0) {
                            let reach = (center(k, half) - c).len();
                            let up = if d.is_nan() { f64::INFINITY } else { d.abs() + reach };
                            next.push((k, up, d, reach));
                        }
                    }
                }
            }
            boxes = next;
            size /= 2;
        }
        (map, read)
    }

    /// Bounds on f over each box of `size` bricks (grown by the clamp, `grown` its
    /// corners), and f at its center when known (NaN when not), from f at the center
    /// plus and minus the half diagonal. For the exact source with the GPU kernel, once the
    /// boxes have bounds at their centers, those centers are read on the GPU. A center it
    /// knows only to lie past the box (its sign unknown) has its parent's sign: no surface
    /// lies within that bound of it, and the parent's center lies within it, so nothing
    /// separates the two (read on the CPU when the parent's sign is not known). A center it
    /// leaves undecided is read on the CPU.
    fn box_bounds(&self, boxes: &[Boxed], size: i64, grown: &(dyn Fn([i64; 3], i64) -> (V3, V3) + Sync), clamp: f64) -> Vec<(f64, f64, f64)> {
        let ex = match &self.source {
            Source::Columns { frep, .. } => {
                return boxes
                    .par_iter()
                    .map(|&(a, ..)| {
                        let (p, q) = grown(a, size);
                        let (l, h) = frep.interval(frep.root, [p.x, p.y, p.z], [q.x, q.y, q.z]);
                        (l, h, f64::NAN)
                    })
                    .collect();
            }
            Source::Exact(ex) => ex,
        };
        let at = |a: [i64; 3]| {
            let (p, q) = grown(a, size);
            (V3::new(0.5 * (p.x + q.x), 0.5 * (p.y + q.y), 0.5 * (p.z + q.z)), 0.5 * (q - p).len())
        };
        let cpu = |c: V3, r: f64, site: usize| {
            QUERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tally(&CPU_AT[site], 1);
            let (sd, _) = ex.sdf(core(c));
            if sd.unsure { (f64::NEG_INFINITY, f64::INFINITY, f64::NAN) } else { (sd.d - r, sd.d + r, sd.d) }
        };
        let all_cpu = || {
            boxes
                .par_iter()
                .map(|&(a, up, ..)| {
                    let (c, r) = at(a);
                    cpu(c, r, if up.is_finite() { 1 } else { 0 })
                })
                .collect()
        };
        let Some(gpu) = self.gpu.as_ref().filter(|_| !CENSUS_ON.load(std::sync::atomic::Ordering::Relaxed)) else { return all_cpu() };
        // A box with no bound on its center's distance yet (the first, or one whose parent's
        // reading was unsure) is read on the CPU; the rest on the GPU. A box is decided once
        // its center is known to lie past the box grown by the clamp, so a center's list
        // holds only the features within a hair past that (`cut`): past it the GPU answers
        // with that bound alone, and every feature not listed lies past it.
        let cut = |r: f64| (r + clamp) * (1.0 + 1e-4);
        let t = std::time::Instant::now();
        let lists: Vec<Option<Vec<u32>>> = boxes.par_iter().map(|&(a, up, ..)| if up.is_finite() { gpu.list(ex, core(at(a).0), 0.0, up.min(cut(at(a).1))) } else { None }).collect();
        since(&STEP_US[4], t);
        let mut slot = Vec::with_capacity(boxes.len());
        let mut pts: Vec<(icam_core::curve3::V3, f64, &[u32])> = Vec::new();
        for (&(a, ..), l) in boxes.iter().zip(&lists) {
            slot.push(l.as_ref().map(|l| {
                let (c, r) = at(a);
                pts.push((core(c), cut(r), l.as_slice()));
                pts.len() - 1
            }));
        }
        let t = std::time::Instant::now();
        let read = gpu.points_scatter_capped(&pts, true);
        since(&STEP_US[5], t);
        let vals = match read {
            Ok(v) => v,
            Err(e) => {
                println!("part cache: the GPU kernel failed ({e}); deciding these boxes on the CPU");
                return all_cpu();
            }
        };
        boxes
            .par_iter()
            .zip(&slot)
            .map(|(&(a, up, pd, reach), s)| {
                let (c, r) = at(a);
                match s.map(|i| vals[i]) {
                    Some([d, flag]) if flag == 0.0 => {
                        let d = d as f64;
                        (d - r, d + r, d)
                    }
                    // Past the box (l > r + clamp): air or material, as the parent's center.
                    Some([l, flag]) if flag == 1.0 && pd != 0.0 && !pd.is_nan() && reach < l as f64 * (1.0 - 1e-6) => {
                        let l = l as f64;
                        if pd > 0.0 { (l - r, f64::INFINITY, f64::NAN) } else { (f64::NEG_INFINITY, r - l, f64::NAN) }
                    }
                    Some([_, flag]) => cpu(c, r, if flag == 1.0 { 6 } else { 1 }),
                    None => cpu(c, r, if up.is_finite() { 7 } else { 0 }),
                }
            })
            .collect()
    }

    /// The first nodes of the level's bricks whose nodes must be read (for `part-bench`),
    /// with the signed distance at their centers when known (NaN when not).
    pub fn surface_bricks(&self, h: f64, lo: V3, nb: [i64; 3]) -> Vec<(V3, f64)> {
        let side = 8.0 * h;
        self.decide(h, lo, nb).1.iter().map(|&(b, d)| (lo + V3::new(side * b[0] as f64, side * b[1] as f64, side * b[2] as f64), d)).collect()
    }

    /// The level of cells of side h, its first node at lo, nb bricks along each axis.
    pub fn level(&mut self, h: f64, lo: V3, nb: [i64; 3]) -> PartLevel {
        self.trim();
        let (built, level) = self.build(h, lo, nb);
        self.keep(built);
        level
    }

    /// The bricks kept and the boxes decided forgotten once there are more than KEPT (to be
    /// called before `build`).
    pub fn trim(&mut self) {
        if self.kept.len() > KEPT {
            self.kept.clear();
            self.provisional.clear();
            self.layouts.clear();
        }
        if let Ok(mut memo) = self.decided.lock() {
            if memo.len() > 8 * KEPT {
                memo.clear();
            }
        }
    }

    /// `level`'s work, the cache left as it was (so levels can be built at once, one listing
    /// on the CPU while another waits on the GPU): the bricks read, for `keep` to take in,
    /// and the level as the window draws it.
    pub fn build(&self, h: f64, lo: V3, nb: [i64; 3]) -> (Built, PartLevel) {
        let e = h.log2().round() as i32;
        let side = 8.0 * h;
        let clamp = h * 3f64.sqrt();
        let origin = self.origin;
        let first = |b: [i64; 3]| lo + V3::new(side * b[0] as f64, side * b[1] as f64, side * b[2] as f64);
        let name = |b: [i64; 3]| {
            let p = first(b);
            [((p.x - origin.x) / h).round() as i64, ((p.y - origin.y) / h).round() as i64, ((p.z - origin.z) / h).round() as i64]
        };
        let t = std::time::Instant::now();
        let (map, mut read) = self.decide(h, lo, nb);
        since(&STEP_US[0], t);
        let at = |b: [i64; 3]| (b[0] + nb[0] * (b[1] + nb[1] * b[2])) as usize;
        // The bricks to read: those not kept from before.
        let fresh: Vec<([i64; 3], f64)> = read.iter().copied().filter(|&(b, _)| !self.kept.contains_key(&(e, name(b)))).collect();
        let made = self.bricks(&fresh.iter().map(|&(b, d)| (first(b), d)).collect::<Vec<_>>(), h, clamp);
        let bricks: Vec<((i32, [i64; 3]), Brick, Option<Provisional>)> = fresh
            .iter()
            .zip(made)
            .map(|(&(b, d), (brick, bridged))| ((e, name(b)), brick, bridged.map(|(q, nodes)| Provisional { p: first(b), d0: d, q, nodes })))
            .collect();
        read.sort_by_key(|&(b, _)| at(b));
        let layout = Layout { map, read: read.iter().map(|&(b, _)| (at(b), name(b))).collect() };
        let new: HashMap<[i64; 3], &Brick> = bricks.iter().map(|(k, b, _)| (k.1, b)).collect();
        let level = assemble(&layout, |n| new.get(n).copied().or_else(|| self.kept.get(&(e, *n)))).unwrap_or(PartLevel { map: Vec::new(), atlas: Vec::new() });
        drop(new);
        (Built { e, layout, bricks }, level)
    }

    /// Takes in a level `build` made.
    pub fn keep(&mut self, b: Built) {
        for (key, brick, bridged) in b.bricks {
            self.kept.insert(key, brick);
            if let Some(p) = bridged {
                self.provisional.insert(key, p);
            }
        }
        self.layouts.insert(b.e, b.layout);
    }

    /// Level e as last laid out, from the bricks as they now stand.
    pub fn relevel(&self, e: i32) -> Option<PartLevel> {
        assemble(self.layouts.get(&e)?, |n| self.kept.get(&(e, *n)))
    }

    /// Level e's stored bricks against the exact source on the CPU, every `every`th node
    /// of each (and the air and material bricks' centers): nodes compared, signs that
    /// differ (beyond the stored step of zero), the most stored steps any is off by, and
    /// how many are off by more than one. None for a source with no exact distance.
    pub fn verify(&self, e: i32, every: usize) -> Option<(usize, usize, i64, usize)> {
        let Source::Exact(ex) = &self.source else { return None };
        let h = 2f64.powi(e);
        let clamp = h * 3f64.sqrt();
        let q = |d: f64| ((d / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i64;
        let picks: Vec<(V3, i64)> = self
            .kept
            .iter()
            .filter(|(k, _)| k.0 == e)
            .flat_map(|(k, b)| {
                let p = self.origin + V3::new(h * k.1[0] as f64, h * k.1[1] as f64, h * k.1[2] as f64);
                let at = move |n: usize| p + V3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
                let v: Vec<(V3, i64)> = match b {
                    Brick::Air => vec![(at(364), 32767)],
                    Brick::Solid => vec![(at(364), -32767)],
                    Brick::Words(w) => (0..729).step_by(every.max(1)).map(|n| (at(n), (w[n / 2] >> (16 * (n % 2))) as u16 as i16 as i64)).collect(),
                };
                v
            })
            .collect();
        let found: Vec<(bool, i64)> = picks
            .par_iter()
            .map(|&(x, stored)| {
                let (sd, _) = ex.sdf(core(x));
                let d = sd.d;
                let off = (q(d) - stored).abs();
                if off > 1 && std::env::var_os("ICAM_VERIFY_WORST").is_some() {
                    println!("part-bench:     off {off} steps at ({:.5}, {:.5}, {:.5}): CPU {d:.6} ({:?}{}), stored {:.6}", x.x, x.y, x.z, sd.feature, if sd.unsure { ", unsure" } else { "" }, stored as f64 / 32767.0 * clamp);
                }
                // A sign differs when the stored value and the distance lie on opposite sides,
                // each beyond the step of zero.
                ((stored > 1 && d < 0.0 && q(d) < -1) || (stored < -1 && d > 0.0 && q(d) > 1), off)
            })
            .collect();
        Some((found.len(), found.iter().filter(|f| f.0).count(), found.iter().map(|f| f.1).max().unwrap_or(0), found.iter().filter(|f| f.1 > 1).count()))
    }

    /// Nodes of level e bridged and not yet read.
    pub fn provisional(&self, e: i32) -> usize {
        self.provisional.iter().filter(|(k, _)| k.0 == e).map(|(_, v)| v.nodes.len()).sum()
    }

    /// Nodes of level e bridged and not yet read, in the bricks `keep` takes (by name).
    pub fn provisional_where(&self, e: i32, keep: impl Fn(&[i64; 3]) -> bool) -> usize {
        self.provisional.iter().filter(|(k, _)| k.0 == e && keep(&k.1)).map(|(_, v)| v.nodes.len()).sum()
    }

    /// Reads every provisional node of level e exactly: on the GPU a thread a node (with
    /// every feature that can be nearest it, refining the leaves it needs), on the CPU what
    /// it leaves; the bricks are packed again. The number read on the CPU.
    pub fn settle(&mut self, e: i32) -> usize {
        self.settle_where(e, |_| true)
    }

    /// The same, for the bricks `keep` takes (by name) alone.
    pub fn settle_where(&mut self, e: i32, keep: impl Fn(&[i64; 3]) -> bool) -> usize {
        let Source::Exact(ex) = &self.source else { return 0 };
        let h = 2f64.powi(e);
        let clamp = h * 3f64.sqrt();
        let half = 4.0 * h;
        let keys: Vec<(i32, [i64; 3])> = self.provisional.keys().filter(|k| k.0 == e && keep(&k.1)).copied().collect();
        if keys.is_empty() {
            return 0;
        }
        // Each brick's signed distance at its center (from its box's decision, or a CPU
        // reading; NaN when the CPU is unsure of it).
        let centers: HashMap<(i32, [i64; 3]), f64> = keys
            .par_iter()
            .map(|k| {
                let v = &self.provisional[k];
                let d0 = if v.d0.is_nan() {
                    tally(&CPU_AT[4], 1);
                    let (sd, _) = ex.sdf(core(v.p + V3::new(half, half, half)));
                    if sd.unsure { f64::NAN } else { sd.d }
                } else {
                    v.d0
                };
                (*k, d0)
            })
            .collect();
        // Every provisional node, with a bound on its distance (its brick's center's, or a
        // CPU reading of that center).
        let items: Vec<((i32, [i64; 3]), u16, V3, f64)> = keys
            .par_iter()
            .flat_map_iter(|k| {
                let v = &self.provisional[k];
                let c = v.p + V3::new(half, half, half);
                let d0 = if centers[k].is_nan() { ex.sdf(core(c)).0.d.abs() } else { centers[k].abs() };
                v.nodes.iter().map(move |&n| {
                    let n = n as usize;
                    let x = v.p + V3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
                    (*k, n as u16, x, d0 + (x - c).len() + 1e-9)
                })
            })
            .collect();
        let lists: Vec<Option<Vec<u32>>> = match &self.gpu {
            Some(gpu) => items.par_iter().map(|&(_, _, x, up)| gpu.list(ex, core(x), 0.0, up)).collect(),
            None => vec![None; items.len()],
        };
        let mut pts: Vec<(icam_core::curve3::V3, f64, &[u32])> = Vec::new();
        let mut slot = Vec::with_capacity(items.len());
        for (it, l) in items.iter().zip(&lists) {
            slot.push(l.as_ref().map(|l| {
                pts.push((core(it.2), clamp, l.as_slice()));
                pts.len() - 1
            }));
        }
        // The scatter, or the refining gather it replaced when ICAM_GATHER is set (to compare).
        let gather = std::env::var_os("ICAM_GATHER").is_some();
        let vals = self.gpu.as_ref().and_then(|g| if gather { g.points_refined(&pts) } else { g.points_scatter(&pts) }.ok()).unwrap_or_default();
        // Per brick: the nodes read exactly, those known only to lie past the clamp (with the
        // bound known on their distance), and the rest, read on the CPU. A node past the clamp
        // has its brick center's sign when either lies within the other's empty ball (no
        // surface lies within a point's distance of it): within the node's bound of it, or
        // the node within the center's distance; else its neighbors' sign, by flood fill as
        // in `bricks`.
        tally(&CPU_AT[8], slot.iter().filter(|s| s.is_none()).count() as u64);
        tally(&CPU_AT[9], vals.iter().filter(|v| v[1] == 2.0).count() as u64);
        for v in vals.iter().filter(|v| v[1] == 2.0) {
            tally(&SETTLE_WHY[(v[0] as usize).min(9)], 1);
        }
        let mut per: HashMap<(i32, [i64; 3]), Vec<(u16, Option<f64>, Option<f64>)>> = HashMap::new();
        for (it, s) in items.iter().zip(&slot) {
            let (d, far) = match s.and_then(|i| vals.get(i)) {
                Some(&[d, flag]) if flag == 0.0 => (Some(d as f64), None),
                Some(&[l, flag]) if flag == 1.0 => (None, Some(l as f64)),
                _ => (None, None),
            };
            per.entry(it.0).or_default().push((it.1, d, far));
        }
        let signed: Vec<((i32, [i64; 3]), Vec<(u16, f64)>, Vec<u16>)> = per
            .par_iter()
            .map(|(k, nodes)| {
                let v = &self.provisional[k];
                let mut f = [f64::NAN; 729];
                for (n, x) in f.iter_mut().enumerate() {
                    *x = v.q[n] as f64 / 32767.0 * clamp;
                }
                for &(n, _, _) in nodes {
                    f[n as usize] = f64::NAN;
                }
                let mut far: Vec<usize> = Vec::new();
                let d0 = centers[k];
                let c = v.p + V3::new(half, half, half);
                for &(n, d, bound) in nodes {
                    let n = n as usize;
                    if let Some(d) = d {
                        f[n] = d;
                    } else if let Some(l) = bound {
                        let x = v.p + V3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
                        let r = (x - c).len();
                        if d0 != 0.0 && !d0.is_nan() && (r < l.max(clamp) * (1.0 - 1e-6) || r < d0.abs() * (1.0 - 1e-6)) {
                            f[n] = clamp.copysign(d0);
                        } else {
                            far.push(n);
                        }
                    }
                }
                loop {
                    let before = far.len();
                    far.retain(|&n| {
                        let (i, j, kk) = ((n % 9) as i64, ((n / 9) % 9) as i64, (n / 81) as i64);
                        for dk in -1..=1i64 {
                            for dj in -1..=1i64 {
                                for di in -1..=1i64 {
                                    let (a, b, c) = (i + di, j + dj, kk + dk);
                                    if !(0..9).contains(&a) || !(0..9).contains(&b) || !(0..9).contains(&c) {
                                        continue;
                                    }
                                    let m = (a + 9 * b + 81 * c) as usize;
                                    if f[m] != 0.0 && !f[m].is_nan() {
                                        f[n] = clamp.copysign(f[m]);
                                        return false;
                                    }
                                }
                            }
                        }
                        true
                    });
                    if far.len() == before {
                        break;
                    }
                }
                let known: Vec<(u16, f64)> = nodes.iter().filter(|&&(n, _, _)| !f[n as usize].is_nan()).map(|&(n, _, _)| (n, f[n as usize])).collect();
                let rest: Vec<u16> = nodes.iter().filter(|&&(n, _, _)| f[n as usize].is_nan()).map(|&(n, _, _)| n).collect();
                (*k, known, rest)
            })
            .collect();
        // ICAM_SETTLE_CHECK: each node signed here read on the CPU, its sign compared.
        if std::env::var_os("ICAM_SETTLE_CHECK").is_some() {
            let idx: HashMap<((i32, [i64; 3]), u16), usize> = items.iter().enumerate().map(|(i, it)| ((it.0, it.1), i)).collect();
            let bad: Vec<String> = signed
                .par_iter()
                .flat_map_iter(|(k, known, _)| known.iter().map(move |&(n, f)| (*k, n, f)))
                .filter_map(|(k, n, f)| {
                    let i = idx[&(k, n)];
                    let (x, up) = (items[i].2, items[i].3);
                    let t = ex.sdf(core(x)).0.d;
                    if t.abs() > 1e-6 && f.abs() > 1e-6 && t.signum() != f.signum() {
                        let g = slot[i].and_then(|s| vals.get(s)).copied();
                        let c = self.provisional[&k].p + V3::new(half, half, half);
                        Some(format!("settle check: e {e} brick {:?} node {n}: stored {f:.5}, true {t:.5}, gpu {g:?}, up {up:.5}, d0 {:.5}, r {:.5}", k.1, centers[&k], (x - c).len()))
                    } else {
                        None
                    }
                })
                .collect();
            for b in bad.iter().take(20) {
                eprintln!("{b}");
            }
            eprintln!("settle check: e {e}: {} nodes signed, {} signs differ", signed.iter().map(|s| s.1.len()).sum::<usize>(), bad.len());
        }
        let at: HashMap<((i32, [i64; 3]), u16), (V3, f64)> = items.iter().map(|&(k, n, x, up)| ((k, n), (x, up))).collect();
        let cpu: Vec<((i32, [i64; 3]), u16, f64)> = signed
            .par_iter()
            .flat_map_iter(|(k, _, rest)| rest.iter().map(move |&n| (*k, n)))
            .map(|(k, n)| {
                let (x, up) = at[&(k, n)];
                QUERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                (k, n, ex.sdf_within(core(x), up).0.d)
            })
            .collect();
        let on_cpu = cpu.len();
        tally(&CPU_AT[5], on_cpu as u64);
        let set = |q: &mut [i16; 729], n: u16, d: f64| q[n as usize] = ((d / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i16;
        for (k, known, _) in &signed {
            if let Some(v) = self.provisional.get_mut(k) {
                for &(n, d) in known {
                    set(&mut v.q, n, d);
                }
            }
        }
        for (k, n, d) in cpu {
            if let Some(v) = self.provisional.get_mut(&k) {
                set(&mut v.q, n, d);
            }
        }
        for k in keys {
            if let Some(v) = self.provisional.remove(&k) {
                self.kept.insert(k, packed(&v.q));
            }
        }
        on_cpu
    }

    /// The bricks whose first nodes are `ps` (with the signed distance at their centers when known,
    /// else NaN). With the GPU kernel: each brick's candidate features listed within the
    /// clamp of its nodes (only there must a node's value be exact; beyond, its sign comes
    /// from a neighbor), the bricks whose every candidate the kernel reads sent to it at
    /// once, the nodes it flags signed from their neighbors or read again on the CPU; the
    /// other bricks read on the CPU.
    fn bricks(&self, ps: &[(V3, f64)], h: f64, clamp: f64) -> Vec<(Brick, Option<([i16; 729], Vec<u16>)>)> {
        let cpu = || ps.par_iter().map(|&(p, _)| (self.brick(p, h, clamp), None)).collect();
        let (Source::Exact(ex), Some(gpu)) = (&self.source, &self.gpu) else { return cpu() };
        if CENSUS_ON.load(std::sync::atomic::Ordering::Relaxed) {
            return cpu();
        }
        let half = 4.0 * h;
        let t = std::time::Instant::now();
        let lists: Vec<(f64, Option<Vec<u32>>)> = ps.par_iter().map(|&(p, d)| (d, gpu.list(ex, core(p + V3::new(half, half, half)), 0.0, half * 3f64.sqrt() + clamp))).collect();
        let on: Vec<(icam_core::curve3::V3, f64, &[u32])> = ps.iter().zip(&lists).filter_map(|(&(p, _), (_, l))| l.as_ref().map(|l| (core(p), h, l.as_slice()))).collect();
        since(&STEP_US[1], t);
        let t = std::time::Instant::now();
        // The scatter, or the gather it replaced when ICAM_GATHER is set (to compare).
        let read = if std::env::var_os("ICAM_GATHER").is_some() { gpu.bricks(&on) } else { gpu.bricks_scatter(&on) };
        since(&STEP_US[2], t);
        let t = std::time::Instant::now();
        let _post = Since(&STEP_US[3], t);
        let vals = match read {
            Ok(v) => v,
            Err(e) => {
                println!("part cache: the GPU kernel failed ({e}); reading these bricks on the CPU");
                return cpu();
            }
        };
        GPU_BRICKS.fetch_add(on.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let mut start = Vec::with_capacity(ps.len());
        let mut k = 0;
        for (_, l) in &lists {
            start.push(l.as_ref().map(|_| k));
            k += if l.is_some() { 729 } else { 0 };
        }
        ps.par_iter()
            .zip(&lists)
            .zip(&start)
            .map(|((&(p, dc), _), &first)| {
                let Some(first) = first else {
                    tally(&CPU_AT[2], 1);
                    return (exact_brick(ex, p, h, clamp), None);
                };
                // Exact values (NaN where not yet known), the far nodes, the undecided ones.
                let mut f = [f64::NAN; 729];
                let mut far = Vec::new();
                let mut open = Vec::new();
                for n in 0..729 {
                    let [v, flag] = vals[first + n];
                    if flag == 0.0 {
                        f[n] = v as f64;
                    } else if flag == 1.0 {
                        far.push(n);
                    } else {
                        open.push(n);
                    }
                }
                let around = |n: usize| {
                    let (i, j, k) = ((n % 9) as i64, ((n / 9) % 9) as i64, (n / 81) as i64);
                    (-1..=1i64)
                        .flat_map(move |dk| (-1..=1i64).flat_map(move |dj| (-1..=1i64).map(move |di| (di, dj, dk))))
                        .filter(|&o| o != (0, 0, 0))
                        .filter_map(move |(di, dj, dk)| {
                            let (a, b, c) = (i + di, j + dj, k + dk);
                            ((0..9).contains(&a) && (0..9).contains(&b) && (0..9).contains(&c)).then(|| ((a + 9 * b + 81 * c) as usize, ((di * di + dj * dj + dk * dk) as f64).sqrt()))
                        })
                };
                // Signs of the far nodes: a far node lies farther than the clamp (a cell
                // diagonal) from the surface, so it has the sign of every node within a
                // cell of it, by flood fill from the nodes read.
                let fill = |f: &mut [f64; 729], far: &mut Vec<usize>| loop {
                    let before = far.len();
                    far.retain(|&n| match around(n).find(|&(m, _)| f[m] != 0.0 && !f[m].is_nan()) {
                        Some((m, _)) => {
                            f[n] = clamp.copysign(f[m]);
                            false
                        }
                        None => true,
                    });
                    if far.len() == before {
                        break;
                    }
                };
                fill(&mut f, &mut far);
                // A far node and the brick's center (its distance dc known from its box) have
                // one sign when either lies in the other's empty ball: the node within |dc| of
                // the center, or the center within the clamp of the node (a far node's own
                // distance is past it). Then the fill again, from those.
                if dc.is_finite() && dc != 0.0 && !far.is_empty() {
                    let c = p + V3::new(half, half, half);
                    let reach = 0.999 * clamp.max(dc.abs());
                    far.retain(|&n| {
                        let x = p + V3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
                        if (x - c).len() < reach {
                            f[n] = clamp.copysign(dc);
                            return false;
                        }
                        true
                    });
                    fill(&mut f, &mut far);
                }
                // A brick whose every node lies past the clamp (its box was kept, but the
                // surface does not come within a cell diagonal of any node) has one sign
                // throughout, its center's: from the box's decision, or one reading.
                if far.len() == 729 {
                    let c = p + V3::new(half, half, half);
                    let sign = if dc.is_finite() && dc != 0.0 {
                        dc
                    } else {
                        tally(&CPU_AT[3], 1);
                        ex.sdf(core(c)).0.d
                    };
                    for &n in &far {
                        f[n] = clamp.copysign(sign);
                    }
                    far.clear();
                }
                // The undecided nodes and the far nodes left unsigned are bridged: the mean
                // of the neighbors known so far, held within the bracket each node read
                // gives (|f(n) - f(m)| <= its distance), passes until none is left.
                let exact = f;
                let mut provisional: Vec<u16> = Vec::new();
                let mut todo: Vec<usize> = open.iter().chain(far.iter()).copied().collect();
                while !todo.is_empty() {
                    let before = todo.len();
                    let snapshot = f;
                    todo.retain(|&n| {
                        let (mut sum, mut cnt, mut lo, mut hi) = (0.0, 0usize, -clamp, clamp);
                        for (m, dist) in around(n) {
                            if !snapshot[m].is_nan() {
                                sum += snapshot[m];
                                cnt += 1;
                            }
                            if !exact[m].is_nan() {
                                lo = lo.max(exact[m] - dist * h);
                                hi = hi.min(exact[m] + dist * h);
                            }
                        }
                        if cnt == 0 {
                            return true;
                        }
                        let v = sum / cnt as f64;
                        f[n] = if lo <= hi { v.clamp(lo, hi) } else { 0.5 * (lo + hi) };
                        provisional.push(n as u16);
                        false
                    });
                    if todo.len() == before {
                        // Nothing around them is known: the brick's far side, taken as air.
                        for &n in &todo {
                            f[n] = clamp;
                            provisional.push(n as u16);
                        }
                        break;
                    }
                }
                GPU_REREADS.fetch_add(provisional.len() as u64, std::sync::atomic::Ordering::Relaxed);
                let mut q = [0i16; 729];
                for (n, x) in q.iter_mut().enumerate() {
                    *x = ((f[n] / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i16;
                }
                (packed(&q), (!provisional.is_empty()).then_some((q, provisional)))
            })
            .collect()
    }

    /// The brick whose first node is p: its 9 x 9 x 9 values, or air or material when they
    /// all lie on one side of zero.
    fn brick(&self, p: V3, h: f64, clamp: f64) -> Brick {
        let (frep, columns) = match &self.source {
            Source::Columns { frep, columns } => (frep, columns),
            Source::Exact(ex) => return exact_brick(ex, p, h, clamp),
        };
        let side = 8.0 * h;
        let r = 0.5 * side * 2f64.sqrt();
        // The columns that can be the least somewhere in the brick.
        let (z0, z1) = (p.z, p.z + side);
        let mut can: Vec<(f64, f64, usize)> = Vec::new();
        for (k, c) in columns.iter().enumerate() {
            let s = frep.section_sd(c.sec, V2::new(p.x + 0.5 * side, p.y + 0.5 * side)).sd;
            let lo = (s - r).max(z0 - c.top).max(c.bottom - z1);
            let hi = (s + r).max(z1 - c.top).max(c.bottom - z0);
            can.push((lo, hi, k));
        }
        let least = can.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        can.retain(|c| c.0 < least.min(clamp));
        if can.is_empty() {
            return Brick::Air;
        }
        // Each column's section distance at the 9 x 9 nodes.
        let sd: Vec<Vec<f64>> = can.iter().map(|&(_, _, k)| (0..81).map(|n| frep.section_sd(columns[k].sec, V2::new(p.x + h * (n % 9) as f64, p.y + h * (n / 9) as f64)).sd).collect()).collect();
        let mut q = [0i16; 729];
        for (n, v) in q.iter_mut().enumerate() {
            let (xy, z) = (n % 81, p.z + h * (n / 81) as f64);
            let f = can.iter().zip(&sd).map(|(&(_, _, k), s)| s[xy].max(z - columns[k].top).max(columns[k].bottom - z)).fold(f64::INFINITY, f64::min);
            *v = ((f / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i16;
        }
        packed(&q)
    }
}

/// A point in the core's vector type.
fn core(p: V3) -> icam_core::curve3::V3 {
    icam_core::curve3::V3::new(p.x, p.y, p.z)
}

/// A brick's values as stored: air or material when they all lie on one side of zero,
/// else two to a word.
fn packed(q: &[i16; 729]) -> Brick {
    if q.iter().all(|&x| x > 0) {
        return Brick::Air;
    }
    if q.iter().all(|&x| x <= 0) {
        return Brick::Solid;
    }
    Brick::Words((0..WORDS).map(|w| q[2 * w] as u16 as u32 | if 2 * w + 1 < 729 { (q[2 * w + 1] as u16 as u32) << 16 } else { 0 }).collect())
}

/// The brick whose first node is p, read from the exact signed distance. Values are clamped
/// to `clamp`, so a node only needs its exact value within that of the surface: nodes are
/// read coarse to fine (every 8th, 4th, 2nd, then every node), and a reading of |d| past
/// the clamp settles every node within |d| - clamp of it as the clamp with its sign, since
/// the 1-Lipschitz field holds no surface within |d| of where it read d. Each reading takes
/// the bound the brick's center gives it (its distance plus how far the node is), so only
/// features that near are looked at.
fn exact_brick(ex: &Exact<'static>, p: V3, h: f64, clamp: f64) -> Brick {
    let half = 4.0 * h;
    let c = p + V3::new(half, half, half);
    let d0 = ex.sdf(core(c)).0.d.abs();
    if CENSUS_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        let mut k = [0usize; 5];
        for id in ex.candidates(core(c), half * 3f64.sqrt(), d0) {
            k[ex.kind_of(id)] += 1;
        }
        if let Ok(mut t) = CENSUS.lock() {
            for i in 0..5 {
                t.0[i] += k[i] as u64;
            }
            t.1 += 1;
            if k[1] + k[3] > 0 {
                t.2 += 1;
            }
            t.3 = t.3.max(k.iter().sum::<usize>() as u64);
        }
        QUERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Brick::Air;
    }
    let at = |n: usize| p + V3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
    let mut v: [Option<f64>; 729] = [None; 729];
    let mut reads = 0u64;
    for step in [8usize, 4, 2, 1] {
        for k in (0..9).step_by(step) {
            for j in (0..9).step_by(step) {
                for i in (0..9).step_by(step) {
                    let n = i + 9 * j + 81 * k;
                    if v[n].is_some() {
                        continue;
                    }
                    let x = at(n);
                    let f = ex.sdf_within(core(x), d0 + (x - c).len() + 1e-9).0.d;
                    reads += 1;
                    v[n] = Some(f);
                    let r = f.abs() - clamp;
                    if r > 0.0 {
                        // The nodes within r of x: a box of cells around it, by distance.
                        let cells = (r / h).floor() as i64;
                        let (a, b, cc) = (i as i64, j as i64, k as i64);
                        for kk in (cc - cells).max(0)..=(cc + cells).min(8) {
                            for jj in (b - cells).max(0)..=(b + cells).min(8) {
                                for ii in (a - cells).max(0)..=(a + cells).min(8) {
                                    let m = (ii + 9 * jj + 81 * kk) as usize;
                                    if v[m].is_none() && (at(m) - x).len() < r {
                                        v[m] = Some(clamp.copysign(f));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    QUERIES.fetch_add(reads + 1, std::sync::atomic::Ordering::Relaxed);
    if CENSUS_ON.load(std::sync::atomic::Ordering::Relaxed) && v.iter().any(|x| x.is_some_and(|f| f <= 0.0)) && v.iter().any(|x| x.is_some_and(|f| f > 0.0)) {
        let mut k = [0usize; 5];
        for id in ex.candidates(core(c), half * 3f64.sqrt(), d0) {
            k[ex.kind_of(id)] += 1;
        }
        if let Ok(mut t) = CENSUS.lock() {
            for i in 0..5 {
                t.0[i] += k[i] as u64;
            }
            t.1 += 1;
            if k[1] + k[3] > 0 {
                t.2 += 1;
            }
            t.3 = t.3.max(k.iter().sum::<usize>() as u64);
        }
    }
    let mut q = [0i16; 729];
    for (n, x) in q.iter_mut().enumerate() {
        *x = ((v[n].unwrap_or(clamp) / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i16;
    }
    packed(&q)
}
