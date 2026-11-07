//! The stock's field cached for drawing (docs/SIM_VISUALIZATION_ENGINE.md, sections 4.3
//! to 4.5): bricks of 8 x 8 x 8 cells, values only where the field cannot rule out a
//! surface the moves made, ray-cast every frame and never meshed.
//!
//! Finding the bricks: space is split coarse to fine. The field never reads more than the
//! true distance to the surface, so a box whose center reads beyond half the box's
//! diagonal holds none of the surface and is marked whole, air or material; a box no
//! move comes near holds at most the stock box's own faces, which the drawing takes from
//! the box itself; every other box splits, down to bricks, whose nodes are read on the GPU
//! from the moves that can come near each brick.
//!
//! Values are F clamped to a cell's diagonal: every corner of a cell the surface crosses
//! is within that of it, so the clamp never bends the drawn surface; and a move changes
//! values only within that of its own box, which is what keeps later re-bakes exact.
//!
//! The drawing reads several caches at once, levels around the view, each twice the cell
//! of the one before (section 4.4): `place` puts a level on its own grid around a point,
//! `cache_at` bakes it. When the view moves on, `scroll` places a level anew from the one
//! held: the search's boxes lie on one grid for every placement of a level, so the boxes
//! wholly within the bricks both placements hold stand as decided, and only the side the
//! level gained is searched and read (with, when the playhead moved too, the boxes the moves
//! between come near, as a backward re-bake reads them).
//!
//! A cache is of the stock after its first `upto` moves (the playhead), and `rebake` moves
//! it to another playhead exactly (section 4.6). Every decision the search makes about a
//! box depends only on the moves that come near the box (its list, whose order does not
//! depend on the playhead: `Stock::near_in`), and a box's children ask about less of space
//! than it does. So a box no move between the two playheads comes near decides as it did
//! before, and so does everything below it; only the boxes those moves come near are
//! decided again, and the result is the cache a bake from nothing would make, word for
//! word (slots are numbered in brick order, so two bakes compare directly).
//!
//! Forward, the stock only loses material: F = max(box, -min s) can only grow as moves are
//! added, F_new = max(F_old, -min over the moves added). So a box that was air stays air
//! unread, and a brick whose values were stored takes, node by node, the greater of its
//! value and what the moves added alone read there: a few moves against each node instead
//! of every move that ever came near it, which is what keeps the stock up with the tool
//! while the program plays. (The 16-bit rounding is monotone, so taking the greater on the
//! stored values is taking it on the field's; the result is within the field's tolerance of
//! a bake from nothing, not word for word, and stays so however many steps are taken.)
//! A brick no added move comes within a cell's diagonal of keeps its values unread (one
//! reading of the added moves at its center decides it). The air bricks' skip distances
//! are kept as they were: air stays air, so every one still holds (a brick that turns to
//! air gets none until a bake that is not forward).
//!
//! Between two moves' ends the playhead is (moves, part): the first `upto` moves and the
//! first `part` of the next, the stock's tail (stock.rs), listed as one more move wherever
//! its box meets. The tail lies within its whole move, so the moves that differ between
//! two playheads are the whole moves from the earlier's to the later's, its own included
//! when either stands partway along it; forward, the moves added are the whole moves
//! passed and the new tail, which holds the old one when both are on the same move. Along
//! the same move only the stock's piece is added, what the tail gained since the old one;
//! on to a later move, the rest of the old one is the piece, then the whole moves past it
//! and the new tail (the whole old move, read again, can be thousands of fine bricks).
//!
//! A forward re-bake keeps every brick's slot: the bricks read again are written over in
//! place, new ones take freed slots or are added at the end, and the cache says which
//! slots changed and the brick map's entries that did (`patch`), so a window holding the
//! cache it came from writes only those.
//! Any other bake numbers the slots in brick order, so two such bakes compare directly.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::geom::{Aabb, V3};
use crate::gpu::{Fields, Lists};
use crate::stock::Stock;

/// Cells to a brick's side.
const B: i64 = 8;
/// Nodes to a brick's side.
const NB: i64 = B + 1;
/// A box with more moves near it than this is split without reading its center: one GPU
/// thread would read them all one after another, and so many moves near a box all but
/// ensure it holds surface.
const LONG: usize = 1024;

/// A brick map entry with no values: no surface inside (air or material), or only the
/// stock box's own faces (no move comes near), drawn from the box itself.
pub const AIR: u32 = u32::MAX;
pub const SOLID: u32 = u32::MAX - 1;
pub const FACES: u32 = u32::MAX - 2;
/// Air with `SKIP | k`: the nearest brick that is not air is k bricks off (Chebyshev,
/// 2 <= k <= 255), so the k - 1 bricks all round are air and a ray can cross them at once.
/// Entries below SKIP are slots.
pub const SKIP: u32 = 0xffff_0000;
/// 32-bit words to a brick's values: 9 x 9 x 9 of them, two to a word.
pub const WORDS: usize = 365;
/// 32-bit words to a brick's cells that may hold surface (`Cache::cells`): a bit a cell.
pub const CELL_WORDS: usize = 16;
/// Caches' names, for `patch`.
static NEXT: AtomicU64 = AtomicU64::new(1);

/// Bricks of 8 x 8 x 8 cells of side `h`, the first node at `lo`; `map` holds, brick by
/// brick (x fastest), the brick's slot in `atlas` or AIR, SOLID, FACES. A slot is WORDS
/// words: the brick's 9 x 9 x 9 node values (x fastest), each F / `clamp` as a 16-bit
/// signed fraction, two to a word (low half first).
#[derive(Clone)]
pub struct Cache {
    pub lo: V3,
    pub h: f64,
    pub dims: [u32; 3],
    pub map: Vec<u32>,
    pub atlas: Vec<u32>,
    /// Slot by slot, CELL_WORDS words: bit c (cells x fastest, 8 x 8 x 8) set when a corner
    /// of cell c reads zero or less, the caster's test for a cell that may hold surface, so
    /// it passes the others without reading their corners (`cells_of`).
    pub cells: Vec<u32>,
    pub clamp: f64,
    /// Bricks (map indices) whose values were read but all lie on one side of zero, so
    /// they are drawn as air or material: for the check.
    pub emptied: Vec<u32>,
    pub report: String,
    /// The playhead: the cache is of the stock after its first `upto` moves, and the first
    /// `part` of the next (its tail; 0 for none).
    pub upto: usize,
    pub part: f64,
    /// The field's tolerance, kept for re-bakes.
    pub tol: f64,
    /// The boxes the search split (`key`; at the bottom, bricks whose values were read).
    split: HashSet<u64>,
    /// This cache's name, and for a forward re-bake, what changed from the cache it was made
    /// from (see the module's notes).
    pub id: u64,
    pub patch: Option<Patch>,
    /// Bricks with values.
    with_values: usize,
    /// Slots no brick uses (forward re-bakes leave them where bricks emptied).
    free: Vec<u32>,
}

impl Cache {
    /// How two caches differ, for a report.
    pub fn diff(&self, o: &Cache) -> String {
        let map = self.map.iter().zip(&o.map).filter(|(a, b)| a != b).count() + self.map.len().abs_diff(o.map.len());
        let words = self.atlas.iter().zip(&o.atlas).filter(|(a, b)| a != b).count();
        format!(
            "playheads {} and {}; {map} brick entries differ; {} and {} bricks with values, {words} words differ; {} and {} emptied; {} and {} boxes split, {} in one only",
            self.upto,
            o.upto,
            self.bricks(),
            o.bricks(),
            self.emptied.len(),
            o.emptied.len(),
            self.split.len(),
            o.split.len(),
            self.split.symmetric_difference(&o.split).count()
        )
    }
}

impl Cache {
    /// How far two caches of the same region and cells are apart: the bricks drawn
    /// otherwise (values in one, air, material or box faces in the other, or different
    /// kinds), and the most two stored values of a brick both hold differ by, mm.
    pub fn apart(&self, o: &Cache) -> (usize, f64) {
        let kind = |x: u32| if x < SKIP { 0 } else if x == SOLID { 2 } else if x == FACES { 3 } else { 1 };
        let value = |c: &Cache, slot: u32, n: usize| {
            let w = c.atlas[slot as usize * WORDS + n / 2];
            (if n.is_multiple_of(2) { w & 0xffff } else { w >> 16 }) as u16 as i16 as f64 / 32767.0 * c.clamp
        };
        let (mut otherwise, mut most) = (0usize, 0f64);
        for (&a, &b) in self.map.iter().zip(&o.map) {
            if kind(a) != kind(b) {
                otherwise += 1;
            } else if a < SKIP {
                for n in 0..(NB * NB * NB) as usize {
                    most = most.max((value(self, a, n) - value(o, b, n)).abs());
                }
            }
        }
        (otherwise + self.map.len().abs_diff(o.map.len()), most)
    }
}

impl PartialEq for Cache {
    /// The same cache: where it lies, its playhead, every brick and value, and the search
    /// that made it.
    fn eq(&self, o: &Cache) -> bool {
        self.lo == o.lo && self.h == o.h && self.dims == o.dims && (self.upto, self.part) == (o.upto, o.part) && self.map == o.map && self.atlas == o.atlas && self.emptied == o.emptied && self.split == o.split
    }
}

impl Cache {
    /// Bricks with values.
    pub fn bricks(&self) -> usize {
        self.with_values
    }

    /// The bricks whose cells that may hold surface (`cells`) disagree with their values: 0
    /// for a sound cache.
    pub fn wrong_cells(&self) -> usize {
        let slots = self.atlas.len() / WORDS;
        if self.cells.len() != slots * CELL_WORDS {
            return slots.max(1);
        }
        (0..slots).into_par_iter().filter(|&s| cells_of(&halves(&self.atlas[s * WORDS..(s + 1) * WORDS]))[..] != self.cells[s * CELL_WORDS..(s + 1) * CELL_WORDS]).count()
    }
}

/// What a forward re-bake changed from the cache it was made from (`from`, its id): the slots
/// whose values changed, those of them a brick took that did not hold them before (whatever
/// is kept per brick, such as the part's records, changes only there), and the brick map's
/// entries that changed, each in order.
#[derive(Clone, Debug)]
pub struct Patch {
    pub from: u64,
    pub slots: Vec<u32>,
    pub placed: Vec<u32>,
    pub entries: Vec<u32>,
}

/// A brick's cells that may hold surface, from its 9 x 9 x 9 node values (x fastest): bit c
/// (c = x + 8 (y + 8 z)) set when one of the cell's eight corners is zero or less, which is
/// where the caster's interpolant can reach zero (`Cache::cells`). Each row of nodes along x
/// becomes a 9-bit mask; a cell row is the four node rows about it, each widened by one.
pub fn cells_of(q: &[i16]) -> [u32; CELL_WORDS] {
    let n = NB as usize;
    let mut rows = [0u32; 81];
    for (r, row) in rows.iter_mut().enumerate() {
        for x in 0..n {
            if q[r * n + x] <= 0 {
                *row |= 1 << x;
            }
        }
        *row = (*row | *row >> 1) & 0xff;
    }
    let mut out = [0u32; CELL_WORDS];
    for z in 0..B as usize {
        for y in 0..B as usize {
            let row = |y: usize, z: usize| rows[y + n * z];
            let m = row(y, z) | row(y + 1, z) | row(y, z + 1) | row(y + 1, z + 1);
            let c = 8 * (y + 8 * z);
            out[c / 32] |= m << (c % 32);
        }
    }
    out
}

/// A brick's node values (WORDS words, two to a word, low half first) as 16-bit fractions.
fn halves(words: &[u32]) -> Vec<i16> {
    (0..(NB * NB * NB) as usize).map(|n| (if n.is_multiple_of(2) { words[n / 2] & 0xffff } else { words[n / 2] >> 16 }) as u16 as i16).collect()
}

/// The clamp on the values of cells of side h: a cell's diagonal.
fn clamp_of(h: f64) -> f64 {
    h * 3f64.sqrt()
}

/// Moves listed in one GPU call, at most: a forward step past thousands of short moves can
/// list millions against the bricks near them, past what one call takes (gpu.rs).
const BUDGET: usize = 1 << 22;

/// A call that reads points against their own lists of moves (`Fields::listed` and the
/// like), split so that no call lists more than BUDGET moves (a point with a longer list
/// still goes alone).
fn split_listed(ask: &Lists, call: impl Fn(&Lists) -> Result<Option<Vec<[f64; 2]>>, String>) -> Result<Option<Vec<[f64; 2]>>, String> {
    if ask.idx.len() <= BUDGET {
        return call(ask);
    }
    let mut out = Vec::with_capacity(ask.pts.len());
    let mut i = 0;
    while i < ask.pts.len() {
        let mut part = Lists::default();
        while i < ask.pts.len() && (part.pts.is_empty() || part.idx.len() + ask.count[i] as usize <= BUDGET) {
            let (f, c) = (ask.first[i] as usize, ask.count[i] as usize);
            part.pts.push(ask.pts[i]);
            part.first.push(part.idx.len() as u32);
            part.count.push(c as u32);
            part.idx.extend_from_slice(&ask.idx[f..f + c]);
            i += 1;
        }
        match call(&part)? {
            Some(v) => out.extend(v),
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// Whether two boxes meet.
fn meets(a: Aabb, b: Aabb) -> bool {
    a.max.x >= b.min.x && a.min.x <= b.max.x && a.max.y >= b.min.y && a.min.y <= b.max.y && a.max.z >= b.min.z && a.min.z <= b.max.z
}

/// The box of the stock's tail when it is the first `part` of move `upto` (none for part
/// 0); an error when the stock's tail is not that.
fn tail_box(stock: &Stock, upto: usize, part: f64) -> Result<Option<Aabb>, String> {
    if part <= 0.0 || upto >= stock.len() {
        return Ok(None);
    }
    match stock.tail() {
        Some((k, p, s)) if k == upto && p == part => Ok(Some(s.bounds())),
        _ => Err(format!("the stock's tail is not the first {part} of move {upto}")),
    }
}

/// A box of the search: its level (bricks 2^level a side) and its first brick.
fn key(level: i64, c: [i64; 3]) -> u64 {
    ((level as u64) << 60) | ((c[0] as u64) << 40) | ((c[1] as u64) << 20) | c[2] as u64
}

impl Cache {
    /// Whether p lies in the cache's bricks.
    pub fn holds(&self, p: V3) -> bool {
        let side = B as f64 * self.h;
        let q = p - self.lo;
        [q.x, q.y, q.z].iter().zip(self.dims).all(|(&x, n)| x >= 0.0 && x < side * n as f64)
    }

    /// The cached field at p, mm: the trilinear value in a brick with values, the box's
    /// own distance in a brick of box faces, plus or minus the clamp in air or material.
    pub fn value(&self, p: V3, stock_box: Aabb) -> f64 {
        let side = B as f64 * self.h;
        let q = p - self.lo;
        let b = [(q.x / side).floor() as i64, (q.y / side).floor() as i64, (q.z / side).floor() as i64];
        let d = self.dims.map(|n| n as i64);
        if (0..3).any(|k| b[k] < 0 || b[k] >= d[k]) {
            return self.clamp;
        }
        match self.map[(b[0] + d[0] * (b[1] + d[1] * b[2])) as usize] {
            AIR => self.clamp,
            SOLID => -self.clamp,
            FACES => stock_box.sd(p).clamp(-self.clamp, self.clamp),
            x if x >= SKIP => self.clamp,
            slot => {
                let u = [q.x / self.h - (b[0] * B) as f64, q.y / self.h - (b[1] * B) as f64, q.z / self.h - (b[2] * B) as f64];
                let c = u.map(|x| (x.floor() as i64).clamp(0, B - 1));
                let f = [u[0] - c[0] as f64, u[1] - c[1] as f64, u[2] - c[2] as f64];
                let at = |i: i64, j: i64, k: i64| {
                    let n = (c[0] + i + NB * (c[1] + j + NB * (c[2] + k))) as usize;
                    let w = self.atlas[slot as usize * WORDS + n / 2];
                    let half = if n.is_multiple_of(2) { w & 0xffff } else { w >> 16 } as u16 as i16;
                    (half as f64 / 32767.0).max(-1.0) * self.clamp
                };
                let lerp = |a: f64, z: f64, t: f64| a + (z - a) * t;
                let bottom = lerp(lerp(at(0, 0, 0), at(1, 0, 0), f[0]), lerp(at(0, 1, 0), at(1, 1, 0), f[0]), f[1]);
                let top = lerp(lerp(at(0, 0, 1), at(1, 0, 1), f[0]), lerp(at(0, 1, 1), at(1, 1, 1), f[0]), f[1]);
                lerp(bottom, top, f[2])
            }
        }
    }
}

/// The cache against the exact field, at the center of every cell of every brick whose
/// values were read, drawn from them or emptied (bricks of air, material or box faces
/// were proven so by the field when the cache was made): the cells whose interpolated center has the other sign from the exact
/// field there (material the cache misses, or invents), and how far the two part where
/// the surface is near. All the exact values come from the GPU in a few batches.
pub fn check(stock: &Stock, gpu: &Fields, c: &Cache, tol: f64) -> Result<String, String> {
    let clock = Instant::now();
    let tail = tail_box(stock, c.upto, c.part)?;
    let d = c.dims.map(|n| n as i64);
    let brick = |i: i64| [i % d[0], (i / d[0]) % d[1], i / (d[0] * d[1])];
    let slots: Vec<[i64; 3]> = (0..c.map.len()).filter(|&i| c.map[i] < SKIP).map(|i| brick(i as i64)).collect();
    let bricks: Vec<[i64; 3]> = slots.iter().copied().chain(c.emptied.iter().map(|&i| brick(i as i64))).collect();
    let center = |b: [i64; 3], q: i64| c.lo + V3::new(c.h * ((b[0] * B + q % B) as f64 + 0.5), c.h * ((b[1] * B + (q / B) % B) as f64 + 0.5), c.h * ((b[2] * B + q / (B * B)) as f64 + 0.5));
    let o = c.lo + V3::new(0.5 * c.h * (d[0] * B) as f64, 0.5 * c.h * (d[1] * B) as f64, 0.5 * c.h * (d[2] * B) as f64);
    let (mut misses, mut invented, mut worst, mut cells) = (0usize, 0usize, 0f64, 0usize);
    // How deep each disagreement lies (the exact field's size there), in cells, and where
    // the deepest is (point, exact, cached).
    let mut deep = [0usize; 3];
    let mut deepest: Option<(f64, V3, f64, f64)> = None;
    for chunk in bricks.chunks(2048) {
        let mut ask = Lists::default();
        for b in chunk {
            let lo = c.lo + V3::new(c.h * (b[0] * B) as f64, c.h * (b[1] * B) as f64, c.h * (b[2] * B) as f64);
            let side = c.h * B as f64;
            let q = Aabb::new(lo, lo + V3::new(side, side, side)).grow(c.clamp);
            let mut list: Vec<u32> = stock.near_in(q, 0..c.upto).into_iter().map(|k| k as u32).collect();
            if tail.is_some_and(|b| meets(b, q)) {
                list.push(stock.len() as u32);
            }
            let first = ask.idx.len() as u32;
            ask.idx.extend(&list);
            for q in 0..B * B * B {
                ask.pts.push(center(*b, q));
                ask.first.push(first);
                ask.count.push(list.len() as u32);
            }
        }
        let got = gpu.listed(stock, o, &ask, c.clamp, tol)?.ok_or("the kernels cannot place a move")?;
        let found: Vec<(bool, bool, f64, f64, V3, f64, f64)> = ask
            .pts
            .par_iter()
            .zip(&got)
            .zip(&ask.count)
            .map(|((&p, x), &n)| {
                let exact = if n == 0 { stock.bounds.sd(p).clamp(-c.clamp, c.clamp) } else { 0.5 * (x[0] + x[1]) };
                let cached = c.value(p, stock.bounds);
                // (Cells of emptied bricks read the clamp: no departure to speak of.)
                let near = exact.abs() < 0.5 * c.clamp && cached.abs() < c.clamp;
                let gap = if near { (cached - exact).abs() } else { 0.0 };
                (exact < -tol && cached > 0.0, exact > tol && cached < 0.0, gap, exact.abs() / c.h, p, exact, cached)
            })
            .collect();
        cells += found.len();
        misses += found.iter().filter(|x| x.0).count();
        invented += found.iter().filter(|x| x.1).count();
        worst = found.iter().map(|x| x.2).fold(worst, f64::max);
        for x in found.iter().filter(|x| x.0 || x.1) {
            for (k, edge) in [0.1, 0.25, 0.5].iter().enumerate() {
                if x.3 > *edge {
                    deep[k] += 1;
                }
            }
            if deepest.is_none_or(|d| x.3 > d.0) {
                deepest = Some((x.3, x.4, x.5, x.6));
            }
        }
    }
    // Neighboring bricks each keep a copy of the face they share: the copies must agree.
    let decode = |slot: u32, n: usize| {
        let w = c.atlas[slot as usize * WORDS + n / 2];
        (if n.is_multiple_of(2) { w & 0xffff } else { w >> 16 }) as u16 as i16 as f64 / 32767.0 * c.clamp
    };
    let (mut faces, mut apart, mut widest) = (0usize, 0usize, 0f64);
    for b in &slots {
        let here = c.map[(b[0] + d[0] * (b[1] + d[1] * b[2])) as usize];
        for axis in 0..3 {
            let mut n = *b;
            n[axis] += 1;
            if n[axis] >= d[axis] {
                continue;
            }
            let there = c.map[(n[0] + d[0] * (n[1] + d[1] * n[2])) as usize];
            if there >= SKIP {
                continue;
            }
            faces += 1;
            for u in 0..NB {
                for v in 0..NB {
                    let (mut x, mut y) = ([0i64; 3], [0i64; 3]);
                    let (p, q) = ((axis + 1) % 3, (axis + 2) % 3);
                    x[axis] = B;
                    y[axis] = 0;
                    x[p] = u;
                    y[p] = u;
                    x[q] = v;
                    y[q] = v;
                    let idx = |k: [i64; 3]| (k[0] + NB * (k[1] + NB * k[2])) as usize;
                    let gap = (decode(here, idx(x)) - decode(there, idx(y))).abs();
                    if gap > 1e-3 * c.h {
                        apart += 1;
                    }
                    widest = widest.max(gap);
                }
            }
        }
    }
    let (wrong, slots_held) = (c.wrong_cells(), c.atlas.len() / WORDS);
    Ok(format!(
        "{wrong} of {slots_held} bricks' cells that may hold surface disagree with their values. {faces} faces shared by bricks with values: {apart} node copies differ by more than a thousandth of a cell, the widest by {widest:.4} mm. {cells} cell centers in {} bricks against the exact field in {:.2} s: {misses} where the cache shows air over material, {invented} where it shows material in air ({} of them more than a tenth of a cell from the surface, {} a quarter, {} a half{}); near the surface the interpolated field departs from the exact by at most {worst:.4} mm",
        bricks.len(),
        clock.elapsed().as_secs_f64(),
        deep[0],
        deep[1],
        deep[2],
        deepest.map_or(String::new(), |(_, p, e, v)| format!("; the deepest at {:.3}, {:.3}, {:.3}, the field {e:.4} mm, the cache {v:.4}", p.x, p.y, p.z))
    ))
}

/// Marks every air brick with how far off the nearest brick that is not air lies (`SKIP`):
/// the chessboard distance transform, two passes over the 26 neighbors (exact for that
/// distance). Bricks past the map's edge do not count: the caster leaves the level there.
fn skips(map: &mut [u32], nb: [i64; 3]) {
    let at = |x: i64, y: i64, z: i64| (x + nb[0] * (y + nb[1] * z)) as usize;
    let mut d: Vec<u8> = map.iter().map(|&x| if x == AIR { 255 } else { 0 }).collect();
    // The 13 neighbors before a brick in x-fastest order; the other 13 are their mirrors.
    let before: Vec<[i64; 3]> = (-1..=1).flat_map(|z| (-1..=1).flat_map(move |y| (-1..=1).map(move |x| [x, y, z]))).filter(|o| (o[2], o[1], o[0]) < (0, 0, 0)).collect();
    let mut pass = |order: &mut dyn Iterator<Item = (i64, i64, i64)>, sign: i64| {
        for (x, y, z) in order {
            let i = at(x, y, z);
            if d[i] == 0 {
                continue;
            }
            let mut m = d[i];
            for o in &before {
                let (u, v, w) = (x + sign * o[0], y + sign * o[1], z + sign * o[2]);
                if u >= 0 && v >= 0 && w >= 0 && u < nb[0] && v < nb[1] && w < nb[2] {
                    m = m.min(d[at(u, v, w)].saturating_add(1));
                }
            }
            d[i] = m;
        }
    };
    let (a, b, c) = (nb[0], nb[1], nb[2]);
    pass(&mut (0..c).flat_map(|z| (0..b).flat_map(move |y| (0..a).map(move |x| (x, y, z)))), 1);
    pass(&mut (0..c).rev().flat_map(|z| (0..b).rev().flat_map(move |y| (0..a).rev().map(move |x| (x, y, z)))), -1);
    for (m, &k) in map.iter_mut().zip(&d) {
        if *m == AIR && k >= 2 {
            *m = SKIP | k as u32;
        }
    }
}

/// Where a level of cells of side `h` lies (section 4.4): on the grid of bricks that starts
/// a cell below the stock box's low corner, at most `n` bricks a side, centered on `at` as
/// nearly as the box allows. Returns the first node, the bricks along each axis, and
/// whether the level spans the whole box.
pub fn place(bounds: Aabb, h: f64, at: V3, n: i64) -> (V3, [i64; 3], bool) {
    let side = B as f64 * h;
    let o = bounds.min - V3::new(h, h, h);
    let full = |a: f64, z: f64| (((z - a + 2.0 * h) / h).ceil() as i64 + B - 1) / B;
    let full = [full(bounds.min.x, bounds.max.x), full(bounds.min.y, bounds.max.y), full(bounds.min.z, bounds.max.z)];
    let (from, to) = ([o.x, o.y, o.z], [at.x, at.y, at.z]);
    let mut first = [0i64; 3];
    let mut nb = full;
    for k in 0..3 {
        if full[k] > n {
            nb[k] = n;
            first[k] = (((to[k] - from[k]) / side).round() as i64 - n / 2).clamp(0, full[k] - n);
        }
    }
    let lo = o + V3::new(side * first[0] as f64, side * first[1] as f64, side * first[2] as f64);
    (lo, nb, full.iter().all(|&f| f <= n))
}

/// The cache of the stock after its first `at.0` moves and the first `at.1` of the next (the
/// stock's tail must be that), in `nb` bricks of cells of side `h` from the node `lo`, the
/// field read to within `tol`.
pub fn cache_at(stock: &Stock, gpu: &Fields, h: f64, lo: V3, nb: [i64; 3], at: (usize, f64), tol: f64) -> Result<Cache, String> {
    bake(stock, gpu, (h, lo, nb), at, tol, None)
}

/// `prev` moved to the playhead `at` (as for `cache_at`), baking again only where a move
/// between its playhead and `at` comes near: the same cache as `cache_at` would make, or
/// forward, within the field's tolerance of it (see the module's notes).
pub fn rebake(stock: &Stock, gpu: &Fields, prev: &Cache, at: (usize, f64)) -> Result<Cache, String> {
    bake(stock, gpu, (prev.h, prev.lo, prev.dims.map(|n| n as i64)), at, prev.tol, Some(prev))
}

/// `prev` placed anew, `nb` bricks from the node `lo` (as `place` puts a level), and moved to
/// the playhead `at` (as for `cache_at`) in the same pass: the bricks both placements hold
/// keep what they had, and only the boxes of the search that reach past `prev`'s bricks, or
/// that a move between the two playheads comes near, are decided again. So a level that
/// follows the view a little is read only along the side it gained, while the program
/// plays. The search's boxes lie on one grid for every placement (`bake`), so the boxes kept
/// are the ones a bake from nothing there would make; it reads the moves in frames about its
/// own bricks, so the result is within the field's tolerance of that bake, not word for word.
pub fn scroll(stock: &Stock, gpu: &Fields, prev: &Cache, lo: V3, nb: [i64; 3], at: (usize, f64)) -> Result<Cache, String> {
    bake(stock, gpu, (prev.h, lo, nb), at, prev.tol, Some(prev))
}

/// Where a level of cells of side `h` from the node `lo` lies on the grid of bricks every
/// level of that cell shares (`place`): its first brick's place on that grid.
fn on_grid(bounds: Aabb, h: f64, lo: V3) -> Result<[i64; 3], String> {
    let side = B as f64 * h;
    let o = bounds.min - V3::new(h, h, h);
    let g = [lo.x - o.x, lo.y - o.y, lo.z - o.z].map(|d| d / side);
    let r = g.map(|x| x.round() as i64);
    if (0..3).any(|k| r[k] < 0 || (g[k] - r[k] as f64).abs() > 1e-6) {
        return Err(format!("the level's first node {:.6}, {:.6}, {:.6} is not on the grid of its bricks", lo.x, lo.y, lo.z));
    }
    Ok(r)
}

fn bake(stock: &Stock, gpu: &Fields, (h, lo, nb): (f64, V3, [i64; 3]), (upto, part): (usize, f64), tol: f64, prev: Option<&Cache>) -> Result<Cache, String> {
    let clock = Instant::now();
    let upto = upto.min(stock.len());
    let part = if upto < stock.len() { part.max(0.0) } else { 0.0 };
    let tail = tail_box(stock, upto, part)?;
    // The search's boxes lie on the grid every placement of the level shares (bricks
    // counted from `place`'s origin, a box of 2^level bricks starting at a multiple of
    // 2^level), so two placements decide the bricks they share by the same boxes. `g0` is the
    // level's first brick on that grid, `pg0` the first of `prev`'s.
    let g0 = on_grid(stock.bounds, h, lo)?;
    let moved = match prev {
        Some(p) if p.h != h => return Err(format!("a level of cells of {} mm cannot be baked from one of {} mm", h, p.h)),
        Some(p) => {
            let pg0 = on_grid(stock.bounds, h, p.lo)?;
            let pnb = p.dims.map(|n| n as i64);
            if pg0 == g0 && pnb == nb { None } else { Some((pg0, pnb)) }
        }
        None => None,
    };
    // The moves numbered within `m` whose boxes meet q, and the tail where its box does.
    let near = |q: Aabb, m: Range<usize>| -> Vec<u32> {
        let mut l: Vec<u32> = stock.near_in(q, m).into_iter().map(|k| k as u32).collect();
        if tail.is_some_and(|b| meets(b, q)) {
            l.push(stock.len() as u32);
        }
        l
    };
    // Forward, what the move the old playhead stood on gained (the stock's piece, when it
    // starts at that playhead): along the same move, the piece to the new part (else the whole
    // tail, which holds it); on to a later move, the rest of it.
    #[derive(Clone, Copy)]
    enum Gain {
        Along(Aabb),
        Rest(Aabb),
    }
    // (Placed anew, a level takes the general way: every box a move between the playheads
    // comes near is decided again from all its moves.)
    let gain = match (prev.filter(|_| moved.is_none()), stock.piece()) {
        (Some(p), Some((k, a, b, s))) if p.upto == upto && k == upto && a == p.part && b == part && part > p.part => Some(Gain::Along(s.bounds())),
        (Some(p), Some((k, a, b, s))) if p.upto < upto && k == p.upto && a == p.part && p.part > 0.0 && b == 1.0 => Some(Gain::Rest(s.bounds())),
        _ => None,
    };
    let near_added = |q: Aabb, m: Range<usize>| -> Vec<u32> {
        let piece = |l: &mut Vec<u32>, g: Aabb| {
            if meets(g, q) {
                l.push(stock.len() as u32 + 1);
            }
        };
        match gain {
            Some(Gain::Along(g)) => {
                let mut l: Vec<u32> = stock.near_in(q, m).into_iter().map(|k| k as u32).collect();
                piece(&mut l, g);
                l
            }
            Some(Gain::Rest(g)) => {
                let mut l = near(q, m);
                piece(&mut l, g);
                l
            }
            None => near(q, m),
        }
    };
    // Nodes and bricks counted from the level's first (`node`, `at`); the search's boxes are
    // counted on the shared grid (`at_g`, and `local` to turn one into the other).
    let node = |i: i64, j: i64, k: i64| V3::new(lo.x + h * i as f64, lo.y + h * j as f64, lo.z + h * k as f64);
    let at = |b: [i64; 3]| (b[0] + nb[0] * (b[1] + nb[1] * b[2])) as usize;
    let local = |c: [i64; 3]| [c[0] - g0[0], c[1] - g0[1], c[2] - g0[2]];
    let at_g = |c: [i64; 3]| at(local(c));
    // The bricks of a box (first brick on the grid, side in bricks) that lie in the level,
    // from and to along each axis; None when it has none.
    let within = |c: [i64; 3], size: i64| -> Option<([i64; 3], [i64; 3])> {
        let a = [0, 1, 2].map(|k| c[k].max(g0[k]));
        let z = [0, 1, 2].map(|k| (c[k] + size).min(g0[k] + nb[k]));
        (0..3).all(|k| a[k] < z[k]).then_some((a, z))
    };
    // The moves between the two playheads, and whether any comes near a region.
    let between = prev.map(|p| {
        if (p.upto, p.part) == (upto, part) {
            upto..upto
        } else {
            p.upto.min(upto)..(p.upto + (p.part > 0.0) as usize).max(upto + (part > 0.0) as usize).min(stock.len())
        }
    });
    // Forward: only moves are added (see the module's notes). Its slots are kept in place, so
    // a level placed anew is never baked forward.
    let forward = moved.is_none() && prev.is_some_and(|p| (upto, part) > (p.upto, p.part));
    let o = lo + V3::new(0.5 * h * (nb[0] * B) as f64, 0.5 * h * (nb[1] * B) as f64, 0.5 * h * (nb[2] * B) as f64);
    let mut level = 0;
    while (1i64 << level) < *nb.iter().max().unwrap_or(&1) && level < 6 {
        level += 1;
    }
    // A box's center, and the cube about it whose moves decide it.
    let cube = |c: [i64; 3], level: i64| {
        let side = ((1i64 << level) * B) as f64 * h;
        let r = 0.5 * side * 3f64.sqrt();
        let l = local(c);
        let p = node(l[0] * B, l[1] * B, l[2] * B) + V3::new(0.5 * side, 0.5 * side, 0.5 * side);
        (p, Aabb::new(p - V3::new(r, r, r), p + V3::new(r, r, r)))
    };
    // Whether a box must be decided again. Along the same move, only the piece the tail
    // gained changed anything; on to a later move, the rest of the old one, the moves past
    // it and the new tail. Placed anew, a box whose bricks in the level are not all `prev`'s.
    let touched = |c: [i64; 3], level: i64| {
        let q = cube(c, level).1;
        let gone = moved.is_some_and(|(pg0, pnb)| within(c, 1 << level).is_none_or(|(a, z)| (0..3).any(|k| a[k] < pg0[k] || z[k] > pg0[k] + pnb[k])));
        gone || match gain {
            Some(Gain::Along(g)) => meets(g, q),
            Some(Gain::Rest(g)) => meets(g, q) || tail.is_some_and(|b| meets(b, q)) || prev.is_some_and(|p| !stock.near_in(q, p.upto + 1..upto).is_empty()),
            None => between.as_ref().is_none_or(|m| !m.is_empty() && !stock.near_in(q, m.clone()).is_empty()),
        }
    };
    let was_split = |k: u64| prev.is_some_and(|p| p.split.contains(&k));
    // Forward, the skip distances stand (see the module's notes); otherwise they are found
    // again at the end. Placed anew, the bricks both placements hold start as `prev`'s.
    let mut map: Vec<u32> = match (prev, moved) {
        (Some(p), _) if forward => p.map.clone(),
        (Some(p), Some((pg0, pnb))) => {
            let mut m = vec![AIR; (nb[0] * nb[1] * nb[2]) as usize];
            for z in 0..nb[2] {
                for y in 0..nb[1] {
                    for x in 0..nb[0] {
                        let q = [x + g0[0] - pg0[0], y + g0[1] - pg0[1], z + g0[2] - pg0[2]];
                        if (0..3).all(|k| q[k] >= 0 && q[k] < pnb[k]) {
                            let v = p.map[(q[0] + pnb[0] * (q[1] + pnb[1] * q[2])) as usize];
                            m[at([x, y, z])] = if (SKIP..FACES).contains(&v) { AIR } else { v };
                        }
                    }
                }
            }
            m
        }
        (Some(p), None) => p.map.iter().map(|&x| if (SKIP..FACES).contains(&x) { AIR } else { x }).collect(),
        (None, _) => vec![AIR; (nb[0] * nb[1] * nb[2]) as usize],
    };
    let is_air = |x: u32| x == AIR || (SKIP..FACES).contains(&x);
    let mark = |map: &mut Vec<u32>, c: [i64; 3], size: i64, v: u32| {
        if let Some((a, z)) = within(c, size) {
            for gz in a[2]..z[2] {
                for gy in a[1]..z[1] {
                    for gx in a[0]..z[0] {
                        map[at_g([gx, gy, gz])] = v;
                    }
                }
            }
        }
    };
    // The boxes of this level of the search, each with whether it stands as decided before:
    // those of the shared grid that hold bricks of the level.
    let mut boxes: Vec<([i64; 3], bool)> = Vec::new();
    let start = |k: usize| g0[k].div_euclid(1 << level) * (1 << level);
    for x in (start(0)..g0[0] + nb[0]).step_by(1 << level) {
        for y in (start(1)..g0[1] + nb[1]).step_by(1 << level) {
            for z in (start(2)..g0[2] + nb[2]).step_by(1 << level) {
                boxes.push(([x, y, z], prev.is_some() && !touched([x, y, z], level)));
            }
        }
    }
    let mut split_set = HashSet::new();
    let (mut checked, mut kept_boxes) = (0usize, 0usize);
    // Bricks whose values are read now, and bricks that keep what they had.
    let (mut fresh, mut kept): (Vec<[i64; 3]>, Vec<[i64; 3]>) = (Vec::new(), Vec::new());
    loop {
        let size = 1i64 << level;
        let half = size / 2;
        let children = |c: [i64; 3]| (0..8).map(move |q| [c[0] + half * (q & 1), c[1] + half * ((q >> 1) & 1), c[2] + half * ((q >> 2) & 1)]).filter(move |c| within(*c, half).is_some());
        let mut next: Vec<([i64; 3], bool)> = Vec::new();
        // As decided before: split ones go on down (their children are untouched too);
        // whole ones already stand in the map.
        for &(c, _) in boxes.iter().filter(|b| b.1) {
            kept_boxes += 1;
            if was_split(key(level, c)) {
                split_set.insert(key(level, c));
                if level == 0 {
                    kept.push(local(c));
                } else {
                    next.extend(children(c).map(|c| (c, true)));
                }
            }
        }
        // Forward, a box that was air (decided whole before) stays air unread: its bricks in
        // the level all hold what it was decided, so the first of them says.
        let still_air = |c: [i64; 3]| forward && !was_split(key(level, c)) && within(c, size).is_some_and(|(a, _)| is_air(map[at_g(a)]));
        kept_boxes += boxes.iter().filter(|b| !b.1 && still_air(b.0)).count();
        let todo: Vec<[i64; 3]> = boxes.iter().filter(|b| !b.1 && !still_air(b.0)).map(|b| b.0).collect();
        let r = 0.5 * (size * B) as f64 * h * 3f64.sqrt();
        let centers: Vec<(V3, Aabb)> = todo.iter().map(|&c| cube(c, level)).collect();
        let lists: Vec<Vec<u32>> = centers.par_iter().map(|&(_, q)| near(q, 0..upto)).collect();
        let mut ask = Lists::default();
        let mut asked = Vec::new();
        for (i, l) in lists.iter().enumerate() {
            if !l.is_empty() && (l.len() <= LONG || level == 0) {
                ask.pts.push(centers[i].0);
                ask.first.push(ask.idx.len() as u32);
                ask.count.push(l.len() as u32);
                ask.idx.extend(l);
                asked.push(i);
            }
        }
        let got = split_listed(&ask, |a| gpu.listed(stock, o, a, 1.01 * r, tol))?.ok_or("the kernels cannot place a move")?;
        checked += todo.len();
        let far = r * (1.0 + 1e-6);
        let mut on = vec![None; todo.len()];
        for (&i, v) in asked.iter().zip(got) {
            on[i] = Some(v);
        }
        let mut split = Vec::new();
        for (i, &c) in todo.iter().enumerate() {
            match on[i] {
                None if !lists[i].is_empty() => split.push(c),
                None => {
                    // No move comes near: the box's own faces, or nothing.
                    let b = stock.bounds.sd(centers[i].0);
                    let v = if b.abs() < far {
                        FACES
                    } else if b > 0.0 {
                        AIR
                    } else {
                        SOLID
                    };
                    mark(&mut map, c, size, v);
                }
                Some(v) if v[0] > far => mark(&mut map, c, size, AIR),
                Some(v) if v[1] < -far => mark(&mut map, c, size, SOLID),
                Some(_) => split.push(c),
            }
        }
        for c in split {
            split_set.insert(key(level, c));
            if level == 0 {
                fresh.push(local(c));
            } else {
                // A child stands as before if the search reached it before (this box was
                // split then too) and no move between the playheads comes near it.
                let before = was_split(key(level, c));
                next.extend(children(c).map(|c| (c, before && !touched(c, level - 1))));
            }
        }
        if level == 0 {
            break;
        }
        boxes = next;
        level -= 1;
    }
    let found = clock.elapsed().as_secs_f64();
    // Every node of every brick read now, F clamped to a cell's diagonal.
    let clamp = h * 3f64.sqrt();
    let t_nodes = Instant::now();
    // Forward: a brick that had values stored reads only the moves added (and keeps the
    // greater); one whose values were all air stays so; the rest read every move.
    let old_slot = |b: [i64; 3]| -> Option<u32> { prev.map(|p| p.map[at(b)]).filter(|&x| forward && x < SKIP) };
    let stays_air = |b: &[i64; 3]| forward && prev.is_some_and(|p| (SKIP..FACES).contains(&p.map[at(*b)]) || p.map[at(*b)] == AIR);
    let (still, fresh): (Vec<[i64; 3]>, Vec<[i64; 3]>) = fresh.into_iter().partition(stays_air);
    kept.extend(still);
    // The whole moves added: past the old playhead's move when its rest is the piece.
    let added = prev.map_or(0..0, |p| if matches!(gain, Some(Gain::Rest(_))) { p.upto + 1..upto } else { p.upto..upto });
    // Forward, a brick with values stored that no added move comes within a cell's diagonal
    // of keeps them unread: the added moves read once at its center, out to that plus its
    // half diagonal (the moves' distances are 1-Lipschitz).
    let fresh: Vec<[i64; 3]> = if forward {
        let (old, new): (Vec<[i64; 3]>, Vec<[i64; 3]>) = fresh.into_iter().partition(|&b| old_slot(b).is_some());
        let reach = clamp_of(h) + 0.5 * (B as f64) * h * 3f64.sqrt();
        let near: Vec<Vec<u32>> = old
            .par_iter()
            .map(|&b| {
                let a = node(b[0] * B, b[1] * B, b[2] * B);
                let z = node(b[0] * B + B, b[1] * B + B, b[2] * B + B);
                near_added(Aabb::new(a, z).grow(clamp_of(h)), added.clone())
            })
            .collect();
        let mut ask = Lists::default();
        let mut asked = Vec::new();
        for (i, l) in near.iter().enumerate() {
            if !l.is_empty() {
                let b = old[i];
                ask.pts.push(node(b[0] * B, b[1] * B, b[2] * B) + V3::new(0.5 * (B as f64) * h, 0.5 * (B as f64) * h, 0.5 * (B as f64) * h));
                ask.first.push(ask.idx.len() as u32);
                ask.count.push(l.len() as u32);
                ask.idx.extend(l);
                asked.push(i);
            }
        }
        let got = split_listed(&ask, |a| gpu.listed_moves(stock, o, a, reach, tol))?.ok_or("the kernels cannot place a move")?;
        let mut comes = vec![false; old.len()];
        for (&i, v) in asked.iter().zip(got) {
            comes[i] = v[1] > -reach;
        }
        let mut fresh = new;
        for (i, b) in old.into_iter().enumerate() {
            if comes[i] {
                fresh.push(b);
            } else {
                kept.push(b);
            }
        }
        fresh
    } else {
        fresh
    };
    let lists: Vec<Vec<u32>> = fresh
        .par_iter()
        .map(|&b| {
            let a = node(b[0] * B, b[1] * B, b[2] * B);
            let z = node(b[0] * B + B, b[1] * B + B, b[2] * B + B);
            let q = Aabb::new(a, z).grow(clamp);
            if old_slot(b).is_some() {
                near_added(q, added.clone())
            } else {
                near(q, 0..upto)
            }
        })
        .collect();
    let per = (NB * NB * NB) as usize;
    let stored = |slot: u32, n: usize| -> f32 {
        let w = prev.map_or(0, |p| p.atlas[slot as usize * WORDS + n / 2]);
        ((if n.is_multiple_of(2) { w & 0xffff } else { w >> 16 }) as u16 as i16 as f64 / 32767.0 * clamp) as f32
    };
    let corner = |b: [i64; 3], q: i64| node(b[0] * B + q % NB, b[1] * B + (q / NB) % NB, b[2] * B + q / (NB * NB));
    let mut values = vec![0f32; fresh.len() * per];
    let mut pending: Vec<usize> = Vec::new();
    for (i, b) in fresh.iter().enumerate() {
        if lists[i].is_empty() {
            for q in 0..NB * NB * NB {
                values[i * per + q as usize] = match old_slot(*b) {
                    // No move added comes near: as stored.
                    Some(slot) => stored(slot, q as usize),
                    None => stock.bounds.sd(corner(*b, q)) as f32,
                };
            }
        } else {
            pending.push(i);
        }
    }
    // Read in the frame of the cache's first node, where every node is a whole number of
    // cells from it, so a node two bricks share is the same point in both.
    let mut gpu_nodes = 0usize;
    // At most 8192 bricks a call, and no more than BUDGET moves listed.
    let mut chunks: Vec<&[usize]> = Vec::new();
    let (mut from, mut listed) = (0, 0);
    for (j, &i) in pending.iter().enumerate() {
        if j > from && (j - from >= 8192 || listed + lists[i].len() > BUDGET) {
            chunks.push(&pending[from..j]);
            (from, listed) = (j, 0);
        }
        listed += lists[i].len();
    }
    if from < pending.len() {
        chunks.push(&pending[from..]);
    }
    for chunk in chunks {
        let mut idx = Vec::new();
        let mut asked = Vec::with_capacity(chunk.len());
        for &i in chunk {
            asked.push((node(fresh[i][0] * B, fresh[i][1] * B, fresh[i][2] * B), h, idx.len() as u32, lists[i].len() as u32));
            idx.extend(&lists[i]);
        }
        let got = gpu.bricks(stock, lo, &asked, &idx, clamp, tol)?.ok_or("the kernels cannot place a move")?;
        gpu_nodes += got.len();
        for (n, &i) in chunk.iter().enumerate() {
            match old_slot(fresh[i]) {
                Some(slot) => {
                    for q in 0..per {
                        values[i * per + q] = stored(slot, q).max(got[n * per + q]);
                    }
                }
                None => values[i * per..(i + 1) * per].copy_from_slice(&got[n * per..(n + 1) * per]),
            }
        }
    }
    let nodes = t_nodes.elapsed().as_secs_f64();
    let snorm = |v: f32| ((v as f64 / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i16;
    let t_put = Instant::now();
    // A brick read now as words, or as AIR or SOLID: a brick whose stored values all lie on
    // one side of zero holds none of the drawn surface (the interpolant keeps their sign),
    // so it is drawn as air, or as material (met on entry, as the caster meets a cell
    // inside), the same picture.
    // With its cells that may hold surface (`cells_of`). Packed on every core, then laid out
    // in order.
    let packed: Vec<Result<(Vec<u32>, [u32; CELL_WORDS]), u32>> = (0..fresh.len())
        .into_par_iter()
        .map(|i| {
            let q: Vec<i16> = values[i * per..(i + 1) * per].iter().map(|&v| snorm(v)).collect();
            if q.iter().all(|&x| x > 0) || q.iter().all(|&x| x <= 0) {
                return Err(if q[0] > 0 { AIR } else { SOLID });
            }
            Ok(((0..WORDS).map(|w| q[2 * w] as u16 as u32 | if 2 * w + 1 < per { (q[2 * w + 1] as u16 as u32) << 16 } else { 0 }).collect(), cells_of(&q)))
        })
        .collect();
    let pack = |i: usize| -> Result<(&[u32], &[u32]), u32> { packed[i].as_ref().map(|(w, c)| (w.as_slice(), c.as_slice())).map_err(|&v| v) };
    let (atlas, cells, emptied, free, patch) = match prev.filter(|_| forward) {
        // Forward, every brick keeps its slot (see the module's notes).
        Some(p) => {
            let (mut atlas, mut cells, mut free, mut emptied, mut dirty, mut placed) = (p.atlas.clone(), p.cells.clone(), p.free.clone(), p.emptied.clone(), Vec::new(), Vec::new());
            // Bricks in boxes the search now decided whole free their slots.
            for (&was, &now) in p.map.iter().zip(map.iter()) {
                if was < SKIP && now >= SKIP {
                    free.push(was);
                }
            }
            for (i, &b) in fresh.iter().enumerate() {
                let m = at(b);
                let old = Some(p.map[m]).filter(|&x| x < SKIP);
                match pack(i) {
                    Err(v) => {
                        free.extend(old);
                        map[m] = v;
                        emptied.push(m as u32);
                    }
                    Ok((words, may)) => {
                        let s = old.or_else(|| free.pop()).unwrap_or_else(|| {
                            atlas.resize(atlas.len() + WORDS, 0);
                            cells.resize(cells.len() + CELL_WORDS, 0);
                            (atlas.len() / WORDS - 1) as u32
                        });
                        atlas[s as usize * WORDS..(s as usize + 1) * WORDS].copy_from_slice(words);
                        cells[s as usize * CELL_WORDS..(s as usize + 1) * CELL_WORDS].copy_from_slice(may);
                        map[m] = s;
                        dirty.push(s);
                        if old.is_none() {
                            placed.push(s);
                        }
                    }
                }
            }
            dirty.sort_unstable();
            dirty.dedup();
            let entries = (0..map.len()).filter(|&i| map[i] != p.map[i]).map(|i| i as u32).collect();
            placed.sort_unstable();
            (atlas, cells, emptied, free, Some(Patch { from: p.id, slots: dirty, placed, entries }))
        }
        None => {
            // Bricks with values, in brick order: read now (index into `values`), or as before.
            enum From {
                Now(usize),
                Before,
            }
            let mut read: Vec<(usize, From)> = fresh.iter().enumerate().map(|(i, &b)| (at(b), From::Now(i))).chain(kept.iter().map(|&b| (at(b), From::Before))).collect();
            read.sort_by_key(|r| r.0);
            let mut atlas = Vec::with_capacity(read.len() * WORDS);
            let mut cells = Vec::with_capacity(read.len() * CELL_WORDS);
            let mut emptied = Vec::new();
            for (m, from) in read {
                match from {
                    From::Now(i) => match pack(i) {
                        Err(v) => {
                            map[m] = v;
                            emptied.push(m as u32);
                        }
                        Ok((words, may)) => {
                            map[m] = (atlas.len() / WORDS) as u32;
                            atlas.extend_from_slice(words);
                            cells.extend_from_slice(may);
                        }
                    },
                    From::Before => {
                        let x = map[m];
                        if x < SKIP {
                            map[m] = (atlas.len() / WORDS) as u32;
                            if let Some(p) = prev {
                                atlas.extend_from_slice(&p.atlas[x as usize * WORDS..(x as usize + 1) * WORDS]);
                                cells.extend_from_slice(&p.cells[x as usize * CELL_WORDS..(x as usize + 1) * CELL_WORDS]);
                            }
                        } else {
                            emptied.push(m as u32);
                        }
                    }
                }
            }
            (atlas, cells, emptied, Vec::new(), None)
        }
    };
    let put = t_put.elapsed().as_secs_f64();
    let t_skips = Instant::now();
    if !forward {
        skips(&mut map, nb);
    }
    let skipped = t_skips.elapsed().as_secs_f64();
    let count = |v: u32| map.iter().filter(|&&x| x == v || (v == AIR && (SKIP..FACES).contains(&x))).count();
    let report = format!(
        "h {h} mm, {} x {} x {} bricks: {} with values ({} read now, {} read on the GPU, {gpu_nodes} values; {} kept), {} box faces only, {} air, {} material; {checked} boxes checked ({kept_boxes} kept) in {found:.2} s, values in {nodes:.2} s, laid out in {put:.3} s, skips in {skipped:.3} s, {:.3} s in all; {:.1} MB",
        nb[0],
        nb[1],
        nb[2],
        map.iter().filter(|&&x| x < SKIP).count(),
        fresh.len(),
        pending.len(),
        kept.len(),
        count(FACES),
        count(AIR),
        count(SOLID),
        clock.elapsed().as_secs_f64(),
        (atlas.len() * 4 + cells.len() * 4 + map.len() * 4) as f64 / 1e6
    );
    let with_values = map.iter().filter(|&&x| x < SKIP).count();
    Ok(Cache { lo, h, dims: [nb[0] as u32, nb[1] as u32, nb[2] as u32], map, atlas, cells, clamp, emptied, report, upto, part, tol, split: split_set, id: NEXT.fetch_add(1, Ordering::Relaxed), patch, free, with_values })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_that_may_hold_surface_are_the_casters_test() {
        // Bricks of random values, some all one sign, some mostly so, against the caster's
        // own test: a corner of the cell reads zero or less.
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for k in 0..400 {
            let below = (k % 20) as u64;
            let q: Vec<i16> = (0..729).map(|_| if r() % 20 < below { -((r() % 32768) as i16) } else { (r() % 32767) as i16 + 1 }).collect();
            let got = cells_of(&q);
            for z in 0..8usize {
                for y in 0..8usize {
                    for x in 0..8usize {
                        let n0 = x + 9 * (y + 9 * z);
                        let want = [0, 1, 9, 10, 81, 82, 90, 91].iter().any(|&o| q[n0 + o] <= 0);
                        let c = x + 8 * (y + 8 * z);
                        assert_eq!(got[c / 32] >> (c % 32) & 1 == 1, want, "brick {k}, cell {x} {y} {z}");
                    }
                }
            }
            // And read back from the words a slot keeps.
            let words: Vec<u32> = (0..WORDS).map(|w| q[2 * w] as u16 as u32 | if 2 * w + 1 < 729 { (q[2 * w + 1] as u16 as u32) << 16 } else { 0 }).collect();
            assert_eq!(halves(&words), q);
        }
    }
}
