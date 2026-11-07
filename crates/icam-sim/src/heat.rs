//! The part's field beside the stock's in the cache, so the ray caster can color the stock
//! by what the program left there (docs/SIM_VISUALIZATION_ENGINE.md, section 4.9).
//!
//! The prismatic part is f = min_k max(sd_k(x, y), z - top_k, bottom_k - z) (icam_core::frep).
//! Its z terms are planes, so each brick with values gets a record of the few columns that can
//! be the nearest within REACH there: the column's top and bottom, and its section's distance
//! at the brick's 9 x 9 nodes in x and y. cache.wgsl takes the planes exactly and interpolates
//! only the section distances, bilinearly. A straight wall's distance is linear, so it is
//! interpolated exactly, an arc's to within h^2 / 8R; only where a section's own distance
//! bends (its medial axis, as at a sharp inside corner) is it off, by up to a quarter cell.
//! The colors are for looking: the numbers come from the comparison (compare.rs), never
//! from here.
//!
//! A section's distances depend only on x and y, so they are kept by brick column and
//! shared by every brick stacked over it; a brick's record depends only on where it is, so
//! it is made once and kept for every playhead.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use icam_core::frep::Frep;
use icam_core::geom::V2;
use icam_sim::part::{self, Column};
use icam_sweep::cache::{Cache, SKIP, WORDS};
use icam_sweep::V3;
use rayon::prelude::*;

/// Columns a brick's record holds, at most.
pub const COLUMNS: usize = 4;
/// A column in a record: its top and bottom (f32, the box's frame), then its section's
/// distance at the 9 x 9 nodes (x fastest), each d / REACH as a 16-bit signed fraction, two
/// to a word (low half first).
const COLUMN_WORDS: usize = 2 + 41;
/// 32-bit words to a brick's record (cache.wgsl, `HEAT_WORDS`).
pub const HEAT_WORDS: usize = COLUMNS * COLUMN_WORDS;
/// Distances are clamped to this, mm: past it the stock is away from the part.
pub const REACH: f64 = 8.0;
/// Brick columns kept, at most, before the store starts again.
const KEPT: usize = 1 << 20;
/// Bricks' records kept, at most, before the store starts again.
const MADE: usize = 1 << 18;

/// A section's distances over one brick column: at its center (unclamped), and at its nodes.
type Distances = (f64, Box<[u32; 41]>);
/// Where a brick lies: its cell exponent and its first node in cells from the origin.
type Brick = (i32, i64, i64, i64);

pub struct Heat {
    frep: Frep,
    columns: Vec<Column>,
    /// The box's low corner: records are in its frame.
    origin: V3,
    /// By (cell exponent, the brick's first node in cells from the origin along x and y,
    /// column).
    kept: Mutex<HashMap<(i32, i64, i64, usize), Distances>>,
    /// Bricks' records, by (cell exponent, the brick's first node in cells from the origin).
    made: Mutex<HashMap<Brick, Box<[u32]>>>,
    /// The most columns any brick could have needed (more than COLUMNS: some dropped).
    pub most: AtomicUsize,
}

impl Heat {
    /// None when the part is not a union of columns (only prismatic parts so far).
    pub fn new(frep: Frep, origin: V3) -> Option<Heat> {
        let columns = part::columns(&frep);
        (!columns.is_empty()).then(|| Heat { frep, columns, origin, kept: Mutex::new(HashMap::new()), made: Mutex::new(HashMap::new()), most: AtomicUsize::new(0) })
    }

    /// The records of the bricks in `slots` (the slot each brick has in `c`), one after
    /// another.
    pub fn records(&self, c: &Cache, slots: &[u32]) -> Vec<u32> {
        let (e, h) = (c.h.log2().round() as i32, c.h);
        let side = 8.0 * h;
        let (nx, ny) = (c.dims[0] as usize, c.dims[1] as usize);
        let mut brick = vec![u32::MAX; c.atlas.len() / WORDS];
        for (i, &x) in c.map.iter().enumerate() {
            if x < SKIP {
                brick[x as usize] = i as u32;
            }
        }
        let first: Vec<V3> = slots
            .iter()
            .map(|&s| {
                let i = brick[s as usize] as usize;
                let (bx, by, bz) = (i % nx, (i / nx) % ny, i / (nx * ny));
                c.lo + V3::new(side * bx as f64, side * by as f64, side * bz as f64)
            })
            .collect();
        let cell = |p: V3| (((p.x - self.origin.x) / h).round() as i64, ((p.y - self.origin.y) / h).round() as i64);
        let key = |p: V3| {
            let (gx, gy) = cell(p);
            (e, gx, gy, ((p.z - self.origin.z) / h).round() as i64)
        };
        let mut made = self.made.lock().expect("the store is not poisoned");
        if made.len() > MADE {
            made.clear();
        }
        let mut new: Vec<V3> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &p in &first {
            if !made.contains_key(&key(p)) && seen.insert(key(p)) {
                new.push(p);
            }
        }
        let mut kept = self.kept.lock().expect("the store is not poisoned");
        if kept.len() > KEPT {
            kept.clear();
        }
        let mut want: Vec<((i32, i64, i64, usize), V3)> = Vec::new();
        let mut asked = std::collections::HashSet::new();
        for &p in &new {
            let (gx, gy) = cell(p);
            for (k, col) in self.columns.iter().enumerate() {
                let key = (e, gx, gy, k);
                if self.near_z(p.z, side, col) && !kept.contains_key(&key) && asked.insert(key) {
                    want.push((key, p));
                }
            }
        }
        let found: Vec<_> = want.par_iter().map(|&(key, p)| (key, self.distances(key.3, p, h))).collect();
        kept.extend(found);
        let kept = &*kept;
        let records: Vec<_> = new
            .par_iter()
            .map(|&p| {
                let (gx, gy) = cell(p);
                let mut rec = vec![0u32; HEAT_WORDS];
                self.record(p, side, |k| kept.get(&(e, gx, gy, k)), &mut rec);
                (key(p), rec.into_boxed_slice())
            })
            .collect();
        made.extend(records);
        let made = &*made;
        let mut out = vec![0u32; first.len() * HEAT_WORDS];
        out.par_chunks_mut(HEAT_WORDS).zip(&first).for_each(|(rec, &p)| rec.copy_from_slice(&made[&key(p)]));
        out
    }

    /// Whether column `col` can come within REACH of the slab [z, z + side] by its planes.
    fn near_z(&self, z: f64, side: f64, col: &Column) -> bool {
        z - col.top < REACH && col.bottom - (z + side) < REACH
    }

    /// Column k's section distances over the brick column whose first node is p.
    fn distances(&self, k: usize, p: V3, h: f64) -> Distances {
        let sec = self.columns[k].sec;
        let half = 4.0 * h;
        let center = self.frep.section_sd(sec, V2::new(p.x + half, p.y + half)).sd;
        let mut words = Box::new([0u32; 41]);
        let far = center.abs() > half * 2f64.sqrt() + REACH;
        for n in 0..81 {
            let d = if far { center } else { self.frep.section_sd(sec, V2::new(p.x + h * (n % 9) as f64, p.y + h * (n / 9) as f64)).sd };
            let q = ((d / REACH).clamp(-1.0, 1.0) * 32767.0).round() as i16 as u16 as u32;
            words[n / 2] |= if n.is_multiple_of(2) { q } else { q << 16 };
        }
        (center, words)
    }

    /// The record of the brick whose first node is p: the columns that can be the part's
    /// nearest within REACH there, the likeliest first.
    fn record<'a>(&self, p: V3, side: f64, kept: impl Fn(usize) -> Option<&'a Distances>, rec: &mut [u32]) {
        let r = 0.5 * side * 2f64.sqrt();
        // Each column's bounds over the brick: (low, high, column).
        let mut can: Vec<(f64, f64, usize)> = Vec::new();
        for (k, col) in self.columns.iter().enumerate() {
            let Some((s, _)) = kept(k).filter(|_| self.near_z(p.z, side, col)) else { continue };
            let lo = (s - r).max(p.z - col.top).max(col.bottom - (p.z + side));
            let hi = (s + r).max(p.z + side - col.top).max(col.bottom - p.z);
            can.push((lo, hi, k));
        }
        let least = can.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        can.retain(|c| c.0 < least.min(REACH));
        can.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Bounds over a whole brick keep columns that are never the nearest in it (a lower
        // column under a higher one's section reads more everywhere there). With more than
        // a record holds, keep those nearest, or within a cell of it, at some node.
        if can.len() > COLUMNS {
            let h = side / 8.0;
            let at = |k: usize, n: usize| {
                let w = kept(k).map_or(0, |d| d.1[n / 2]);
                (if n.is_multiple_of(2) { w & 0xffff } else { w >> 16 }) as u16 as i16 as f64 / 32767.0 * REACH
            };
            let mut near = vec![0usize; can.len()];
            let mut v = vec![0.0; can.len()];
            for n in 0..81 {
                for kz in 0..9 {
                    let z = p.z + h * kz as f64;
                    for (i, c) in can.iter().enumerate() {
                        let col = self.columns[c.2];
                        v[i] = at(c.2, n).max(z - col.top).max(col.bottom - z);
                    }
                    let m = v.iter().cloned().fold(f64::INFINITY, f64::min);
                    if m < REACH {
                        for (i, x) in v.iter().enumerate() {
                            near[i] += (*x <= m + h) as usize;
                        }
                    }
                }
            }
            let mut order: Vec<usize> = (0..can.len()).filter(|&i| near[i] > 0).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(near[i]));
            can = order.into_iter().map(|i| can[i]).collect();
        }
        self.most.fetch_max(can.len(), Ordering::Relaxed);
        for (j, w) in rec.chunks_mut(COLUMN_WORDS).enumerate() {
            match can.get(j).and_then(|&(_, _, k)| kept(k).map(|d| (k, d))) {
                Some((k, (_, d))) => {
                    let col = self.columns[k];
                    w[0] = ((col.top - self.origin.z) as f32).to_bits();
                    w[1] = ((col.bottom - self.origin.z) as f32).to_bits();
                    w[2..].copy_from_slice(&d[..]);
                }
                // No column: its top below its bottom, which the caster skips.
                None => {
                    w[0] = (-1e30f32).to_bits();
                    w[1] = 1e30f32.to_bits();
                }
            }
        }
    }
}
