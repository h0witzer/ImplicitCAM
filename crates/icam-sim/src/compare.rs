//! The stock a program leaves against the part, on the part's own surface
//! (docs/SIM_VISUALIZATION_ENGINE.md, section 4.10).
//!
//! Points on the surface: grid lines along x, y and z, a step apart, through the cells a
//! coarse-to-fine search keeps (those where the part's certified interval holds zero).
//! Along each line, the part's field changes sign across a kept cell where the surface
//! crosses it, and the root is solved there. A root is kept by the axis its normal lies
//! nearest and stands for step^2 / |n_axis| of area: a patch of surface is met by the
//! lines of that one axis at a density that makes the weights sum to its area. Every
//! point lies on the exact surface; nothing is meshed.
//!
//! A prismatic part (columns over cross-sections, `part::columns`) gives the same points
//! far faster: which columns hold each vertical line through a cell's center is read once
//! for the whole grid (blocks of lines decided by one exact section distance). Along a
//! vertical line that alone places the faces; along a level line, a column's bit flips
//! between two neighboring lines where its section's boundary crosses, so a wall is one
//! root solved between them, and the bits on either side say at which heights the union
//! of the columns standing there changes across it.
//!
//! The surface is taken a tile at a time (`Surface::points`) and its points are read,
//! sorted into bands and dropped batch by batch: only the bands' sums and the gouged points
//! are kept, whatever the size of the part. One thread finds the next batch's points while
//! the GPU reads the stock at the last one's.
//!
//! What was left there, along the part's normal, d = TOL (and a hair) from the surface: a
//! point is gouged where the stock is air at q = p - d n and q lies deeper than TOL in
//! the part (beside another face of the part it does not: cutting there stays within TOL
//! of the part), has stock left where the stock is material at p + d n, and is on size
//! otherwise. On faces that point up this is exactly what a top-down check measures (the
//! deepest the removed column reaches into the part; the material standing over the
//! top). The stock's field F at p, read on the GPU (each tile of points reading the moves
//! that come near it), decides the stock at p -/+ d n alone when it is past d either way
//! (that ball is all air or all material); points within d of the stock's surface are
//! read again there, and the part is read at q only where the stock is air. Gouged points
//! within two steps of each other form one gouge, which gets its area, the volume the
//! field bounds from below, and around its deepest
//! point, read again on a finer step, the depth along the part's normal and the move that
//! cut there, both in double precision on the CPU.
//!
//! The record (`Analysis::json`) holds the inputs' hashes, the bands' areas and every
//! gouge: a few kilobytes, whatever the size of the part.

use std::collections::HashMap;
use std::time::Instant;

use icam_core::frep::Frep;
use icam_sweep::gpu::{Fields, Lists};
use icam_sweep::{Aabb, Stock, V3};
use rayon::prelude::*;

use crate::part::{Column, TOL};
use icam_core::geom::V2;

/// A point on the part's surface: where, its outward normal, and the area it stands for
/// (mm^2).
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub p: [f64; 3],
    pub n: [f32; 3],
    pub w: f32,
}

/// Cells to a side of the blocks the search starts from: one task each, the part's tree
/// pruned to each.
const BLOCK: i64 = 32;
/// Points sharing one list of moves: tiles this many mm a side.
const TILE: f64 = 8.0;
/// The stock's field is read to this far either side of zero, mm: deeper gouges read as
/// this deep and are read again with a larger limit.
const LIMIT: f64 = 5.0;

/// The part's columns when its F-rep is prismatic (the least of its columns and nothing
/// else), so `Surface` may read them; None otherwise.
pub fn prism(frep: &Frep) -> Option<Vec<Column>> {
    let columns = crate::part::columns(frep);
    (!columns.is_empty() && columns.len() <= 32 && frep.leaf_count() == 3 * columns.len()).then_some(columns)
}

/// A prismatic part's columns on the grid: which of them hold each vertical line through a
/// cell's center (bit k for column k, x fastest), and which stand at each height of the
/// cells' centers.
struct Prism {
    columns: Vec<Column>,
    held: Vec<u32>,
    standing: Vec<u32>,
}

/// The part's surface on a grid of cells a step apart from `lo`, taken a tile at a time:
/// the general search, or a prismatic part's columns.
pub struct Surface<'a> {
    frep: &'a Frep,
    lo: [f64; 3],
    step: f64,
    /// Cells along each axis.
    pub n: [usize; 3],
    prism: Option<Prism>,
}

impl<'a> Surface<'a> {
    /// Over [lo, hi] (`generic`: the general search even for a prismatic part).
    pub fn new(frep: &'a Frep, lo: [f64; 3], hi: [f64; 3], step: f64, generic: bool) -> Surface<'a> {
        let n: [usize; 3] = std::array::from_fn(|a| ((hi[a] - lo[a]) / step).ceil().max(1.0) as usize);
        let prism = prism(frep).filter(|_| !generic).map(|columns| {
            let at = |a: usize, i: usize| lo[a] + step * (i as f64 + 0.5);
            let sd = |k: usize, x: f64, y: f64| frep.section_sd(columns[k].sec, V2::new(x, y)).sd;
            let boxes: Vec<(V2, V2)> = columns.iter().map(|c| frep.sections[c.sec as usize].bbox()).collect();
            // Block by block: a section's distance at a block's center decides the block when
            // it is past the block's half diagonal.
            let mut held = vec![0u32; n[0] * n[1]];
            const ROWS: usize = 64;
            held.par_chunks_mut(n[0] * ROWS).enumerate().for_each(|(tj, rows)| {
                let j0 = tj * ROWS;
                let j1 = (j0 + ROWS).min(n[1]);
                fn decide(sd: &dyn Fn(f64, f64) -> f64, at: &dyn Fn(usize, usize) -> f64, b: [usize; 4], step: f64, set: &mut dyn FnMut(usize, usize)) {
                    let (i0, j0, i1, j1) = (b[0], b[1], b[2], b[3]);
                    let (cx, cy) = (0.5 * (at(0, i0) + at(0, i1 - 1)), 0.5 * (at(1, j0) + at(1, j1 - 1)));
                    let r = 0.5 * step * (((i1 - i0 - 1) as f64).hypot((j1 - j0 - 1) as f64));
                    let v = sd(cx, cy);
                    if v < -r || (i1 - i0 == 1 && j1 - j0 == 1 && v < 0.0) {
                        for j in j0..j1 {
                            for i in i0..i1 {
                                set(i, j);
                            }
                        }
                        return;
                    }
                    if v > r || (i1 - i0 == 1 && j1 - j0 == 1) {
                        return;
                    }
                    let (im, jm) = ((i0 + i1).div_ceil(2), (j0 + j1).div_ceil(2));
                    for q in [[i0, j0, im, jm], [im, j0, i1, jm], [i0, jm, im, j1], [im, jm, i1, j1]] {
                        if q[2] > q[0] && q[3] > q[1] {
                            decide(sd, at, q, step, set);
                        }
                    }
                }
                for (k, bb) in boxes.iter().enumerate() {
                    for i0 in (0..n[0]).step_by(ROWS) {
                        let i1 = (i0 + ROWS).min(n[0]);
                        if at(0, i1 - 1) < bb.0.x - step || at(0, i0) > bb.1.x + step || at(1, j1 - 1) < bb.0.y - step || at(1, j0) > bb.1.y + step {
                            continue;
                        }
                        let mut set = |i: usize, j: usize| rows[(j - j0) * n[0] + i] |= 1 << k;
                        decide(&|x, y| sd(k, x, y), &at, [i0, j0, i1, j1], step, &mut set);
                    }
                }
            });
            let standing = (0..n[2]).map(|l| (0..columns.len()).filter(|&k| at(2, l) > columns[k].bottom && at(2, l) < columns[k].top).fold(0u32, |m, k| m | (1 << k))).collect();
            Prism { columns, held, standing }
        });
        Surface { frep, lo, step, n, prism }
    }

    /// Tiles of at most `side` by `side` cells in x and y, every cell in z: [i0, j0, i1, j1].
    pub fn tiles(&self, side: usize) -> Vec<[usize; 4]> {
        let n = self.n;
        (0..n[1]).step_by(side).flat_map(|j| (0..n[0]).step_by(side).map(move |i| [i, j, (i + side).min(n[0]), (j + side).min(n[1])])).collect()
    }

    /// Every point, all tiles at once (for small boxes).
    pub fn all(&self) -> Vec<Sample> {
        self.tiles(256).par_iter().flat_map_iter(|&t| self.points(t)).collect()
    }

    /// The points of one tile. Together the tiles give each point once: a vertical line
    /// belongs to the tile its cell is in, and the stretch of a level line between two
    /// neighboring cell centers to the tile of the lower one. (A prismatic part's walls
    /// within half a step of the grid's sides are not met: the grid is padded past the part.)
    pub fn points(&self, t: [usize; 4]) -> Vec<Sample> {
        let (lo, step, n) = (self.lo, self.step, self.n);
        let Some(prism) = &self.prism else {
            let corner = [lo[0] + step * t[0] as f64, lo[1] + step * t[1] as f64, lo[2]];
            return surface_cells(self.frep, corner, [t[2] - t[0], t[3] - t[1], n[2]], step);
        };
        let (columns, held, standing) = (&prism.columns, &prism.held, &prism.standing);
        let at = |a: usize, i: usize| lo[a] + step * (i as f64 + 0.5);
        let sd = |k: usize, x: f64, y: f64| self.frep.section_sd(columns[k].sec, V2::new(x, y));
        let (zlo, zhi) = (lo[2], lo[2] + step * n[2] as f64);
        let mut out = Vec::new();
        // Faces: the columns holding a vertical line, merged, top and bottom of each span.
        let w = (step * step) as f32;
        for j in t[1]..t[3] {
            for i in t[0]..t[2] {
                let mask = held[j * n[0] + i];
                if mask == 0 {
                    continue;
                }
                let mut spans: Vec<(f64, f64)> = (0..columns.len()).filter(|&k| mask & (1 << k) != 0).map(|k| (columns[k].bottom, columns[k].top)).collect();
                spans.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut merged: Vec<(f64, f64)> = Vec::new();
                for (b, top) in spans {
                    match merged.last_mut() {
                        Some(m) if b <= m.1 => m.1 = m.1.max(top),
                        _ => merged.push((b, top)),
                    }
                }
                let (x, y) = (at(0, i), at(1, j));
                for (b, top) in merged {
                    if b > zlo && b < zhi {
                        out.push(Sample { p: [x, y, b], n: [0.0, 0.0, -1.0], w });
                    }
                    if top > zlo && top < zhi {
                        out.push(Sample { p: [x, y, top], n: [0.0, 0.0, 1.0], w });
                    }
                }
            }
        }
        // Walls: between two neighboring centers along x (then y), each column whose bit
        // flips has its boundary crossing there, solved on its exact distance. In order
        // along the line the crossings flip the bits from the first center's to the
        // second's; at each height, a crossing is surface where that flips whether any
        // column standing there holds the line. Each is kept by the axis its normal lies
        // nearest; the rest still flip the bits.
        for axis in 0..2 {
            let (a0, a1, b0, b1) = if axis == 0 { (t[0], t[2], t[1], t[3]) } else { (t[1], t[3], t[0], t[2]) };
            let idx = |a: usize, b: usize| if axis == 0 { b * n[0] + a } else { a * n[0] + b };
            for b in b0..b1 {
                let c = at(1 - axis, b);
                let point = |u: f64| if axis == 0 { (u, c) } else { (c, u) };
                for a in a0..a1.min(n[axis] - 1) {
                    let (ma, mb) = (held[idx(a, b)], held[idx(a + 1, b)]);
                    if ma == mb {
                        continue;
                    }
                    let (ua, ub) = (at(axis, a), at(axis, a + 1));
                    let mut found: Vec<(u32, f64, f64, f64, bool)> = Vec::new();
                    for k in 0..columns.len() {
                        if (ma ^ mb) & (1 << k) == 0 {
                            continue;
                        }
                        let v_at = |u: f64| {
                            let (x, y) = point(u);
                            sd(k, x, y).sd
                        };
                        let (va, vb) = (v_at(ua), v_at(ub));
                        // Regula falsi, the stale end halved; inside is a negative distance, as
                        // in the bits.
                        let (mut p, mut q, mut fp, mut fq) = (ua, ub, va, vb);
                        let mut last = 0i8;
                        let mut m = 0.5 * (ua + ub);
                        for _ in 0..60 {
                            if (fq - fp).abs() < 1e-300 {
                                break;
                            }
                            m = (p * fq - q * fp) / (fq - fp);
                            let fm = v_at(m);
                            if fm.abs() < 1e-12 || (q - p) < 1e-10 {
                                break;
                            }
                            if (fm < 0.0) == (fp < 0.0) {
                                (p, fp) = (m, fm);
                                if last == -1 {
                                    fq *= 0.5;
                                }
                                last = -1;
                            } else {
                                (q, fq) = (m, fm);
                                if last == 1 {
                                    fp *= 0.5;
                                }
                                last = 1;
                            }
                        }
                        // The normal a hundred-thousandth of a millimeter outside (see `cell`).
                        let (ox, oy) = point(m + if va >= 0.0 { -1e-5 } else { 1e-5 });
                        let g = sd(k, ox, oy).grad;
                        let l = g.x.hypot(g.y);
                        let (nx, ny) = if l > 1e-9 { (g.x / l, g.y / l) } else { (0.0, 0.0) };
                        let keep = l > 1e-9 && if axis == 0 { nx.abs() >= ny.abs() } else { ny.abs() > nx.abs() };
                        found.push((1 << k, m, nx, ny, keep));
                    }
                    found.sort_by(|p, q| p.1.total_cmp(&q.1));
                    let mut state = ma;
                    let mut g = 0;
                    while g < found.len() {
                        let mut e = g + 1;
                        while e < found.len() && found[e].1 - found[g].1 < 1e-7 {
                            e += 1;
                        }
                        let bits = found[g..e].iter().fold(0u32, |m, f| m | f.0);
                        let (before, after) = (state, state ^ bits);
                        state = after;
                        if found[g..e].iter().any(|f| f.4) {
                            let (x, y) = point(found[g].1);
                            for (l, &st) in standing.iter().enumerate() {
                                if (before & st != 0) == (after & st != 0) {
                                    continue;
                                }
                                // The normal of a crossing here whose column stands here.
                                if let Some(f) = found[g..e].iter().find(|f| f.4 && f.0 & st != 0) {
                                    let w = (step * step / if axis == 0 { f.2.abs() } else { f.3.abs() }) as f32;
                                    out.push(Sample { p: [x, y, at(2, l)], n: [f.2 as f32, f.3 as f32, 0.0], w });
                                }
                            }
                        }
                        g = e;
                    }
                }
            }
        }
        out
    }
}

/// Points on the part's surface inside [lo, hi], a step apart (see the module's notes),
/// the general way.
pub fn surface(frep: &Frep, lo: [f64; 3], hi: [f64; 3], step: f64) -> Vec<Sample> {
    let n: [usize; 3] = std::array::from_fn(|a| ((hi[a] - lo[a]) / step).ceil().max(1.0) as usize);
    surface_cells(frep, lo, n, step)
}

/// As `surface`, over n cells from lo.
fn surface_cells(frep: &Frep, lo: [f64; 3], n: [usize; 3], step: f64) -> Vec<Sample> {
    let n: [i64; 3] = n.map(|x| x as i64);
    let corner = move |c: [i64; 3]| -> [f64; 3] { std::array::from_fn(|a| lo[a] + step * c[a] as f64) };
    let blocks: Vec<[i64; 3]> = (0..n[2]).step_by(BLOCK as usize).flat_map(|z| (0..n[1]).step_by(BLOCK as usize).flat_map(move |y| (0..n[0]).step_by(BLOCK as usize).map(move |x| [x, y, z]))).collect();
    blocks
        .par_iter()
        .flat_map_iter(|&b| {
            let end: [i64; 3] = std::array::from_fn(|a| (b[a] + BLOCK).min(n[a]));
            let (blo, bhi) = (corner(b), corner(end));
            let mut out = Vec::new();
            let (a, z) = frep.interval(frep.root, blo, bhi);
            if a <= 0.0 && z >= 0.0 {
                let f = frep.prune(blo, bhi);
                descend(&f, b, BLOCK, end, &corner, step, &mut out);
            }
            out.into_iter()
        })
        .collect()
}

fn descend(f: &Frep, c: [i64; 3], size: i64, end: [i64; 3], corner: &dyn Fn([i64; 3]) -> [f64; 3], step: f64, out: &mut Vec<Sample>) {
    let top: [i64; 3] = std::array::from_fn(|a| (c[a] + size).min(end[a]));
    if (0..3).any(|a| top[a] <= c[a]) {
        return;
    }
    let (a, z) = f.interval(f.root, corner(c), corner(top));
    if a > 0.0 || z < 0.0 {
        return;
    }
    if size == 1 {
        cell(f, corner(c), step, out);
        return;
    }
    let half = size / 2;
    for q in 0..8 {
        descend(f, [c[0] + half * (q & 1), c[1] + half * ((q >> 1) & 1), c[2] + half * ((q >> 2) & 1)], half, end, corner, step, out);
    }
}

/// The roots on the three lines through the cell's center, each kept by the axis its
/// normal lies nearest.
fn cell(f: &Frep, lo: [f64; 3], step: f64, out: &mut Vec<Sample>) {
    let center: [f64; 3] = std::array::from_fn(|a| lo[a] + 0.5 * step);
    let value = |p: [f64; 3]| f.eval(p).value;
    for axis in 0..3 {
        let mut p0 = center;
        let mut p1 = center;
        p0[axis] -= 0.5 * step;
        p1[axis] += 0.5 * step;
        let (v0, v1) = (value(p0), value(p1));
        // Inside is v <= 0: a root at a face between two cells belongs to one of them.
        if (v0 <= 0.0) == (v1 <= 0.0) {
            continue;
        }
        // Regula falsi, halving the stale end (Illinois): the field is nearly linear along
        // a line near its surface, so a few steps do.
        let (mut ta, mut tb, mut fa, mut fb) = (0.0f64, 1.0f64, v0, v1);
        let mut last = 0i8;
        let mut t = 0.5;
        for _ in 0..60 {
            t = (ta * fb - tb * fa) / (fb - fa);
            let mut p = p0;
            p[axis] += t * step;
            let v = value(p);
            if v.abs() < 1e-12 || (tb - ta) * step < 1e-10 {
                break;
            }
            if (v <= 0.0) == (v0 <= 0.0) {
                (ta, fa) = (t, v);
                if last == -1 {
                    fb *= 0.5;
                }
                last = -1;
            } else {
                (tb, fb) = (t, v);
                if last == 1 {
                    fa *= 0.5;
                }
                last = 1;
            }
        }
        let mut q = p0;
        q[axis] += t * step;
        // The normal is read a hundred-thousandth of a millimeter outside the root, along the
        // line: on the surface itself a distance's gradient is a difference of two points
        // that agree to the last bits, any direction at all.
        let mut beside = q;
        beside[axis] += if v0 > 0.0 { -1e-5 } else { 1e-5 };
        let g = f.eval(beside).grad;
        let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if l.is_nan() || l <= 1e-9 {
            continue;
        }
        let n = [g[0] / l, g[1] / l, g[2] / l];
        let near = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()).then(b.cmp(&a))).unwrap_or(0);
        if near == axis {
            out.push(Sample { p: q, n: [n[0] as f32, n[1] as f32, n[2] as f32], w: (step * step / n[axis].abs()) as f32 });
        }
    }
}

/// The stock's field (after its first `upto` moves) at each point, clamped to
/// [-limit, limit], on the GPU: the points in tiles, each tile reading the moves that
/// come within the limit of it.
pub fn stock_at(stock: &Stock, gpu: &Fields, pts: &[[f64; 3]], upto: usize, limit: f64, tol: f64) -> Result<Vec<f64>, String> {
    let mut tiles: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        tiles.entry(std::array::from_fn(|a| (p[a] / TILE).floor() as i64)).or_default().push(i as u32);
    }
    let mut tiles: Vec<([i64; 3], Vec<u32>)> = tiles.into_iter().collect();
    tiles.sort_by_key(|t| t.0);
    let o = stock.bounds.center();
    let mut out = vec![0f64; pts.len()];
    // A few hundred thousand points a call.
    let mut start = 0;
    while start < tiles.len() {
        let mut stop = start;
        let mut count = 0;
        while stop < tiles.len() && (count == 0 || count + tiles[stop].1.len() <= 1 << 19) {
            count += tiles[stop].1.len();
            stop += 1;
        }
        let lists: Vec<Vec<u32>> = tiles[start..stop]
            .par_iter()
            .map(|(k, _)| {
                let lo = V3::new(k[0] as f64 * TILE, k[1] as f64 * TILE, k[2] as f64 * TILE);
                stock.near_in(Aabb::new(lo, lo + V3::new(TILE, TILE, TILE)).grow(limit), 0..upto).into_iter().map(|m| m as u32).collect()
            })
            .collect();
        let mut ask = Lists::default();
        for ((_, members), list) in tiles[start..stop].iter().zip(&lists) {
            if list.is_empty() {
                continue;
            }
            let first = ask.idx.len() as u32;
            ask.idx.extend(list);
            for &i in members {
                let p = pts[i as usize];
                ask.pts.push(V3::new(p[0], p[1], p[2]));
                ask.first.push(first);
                ask.count.push(list.len() as u32);
            }
        }
        let got = gpu.listed(stock, o, &ask, limit, tol)?.ok_or("the kernels cannot place a move")?;
        let mut k = 0;
        for ((_, members), list) in tiles[start..stop].iter().zip(&lists) {
            for &i in members {
                let p = pts[i as usize];
                // No move near: the box's own distance.
                out[i as usize] = if list.is_empty() {
                    stock.bounds.sd(V3::new(p[0], p[1], p[2])).clamp(-limit, limit)
                } else {
                    k += 1;
                    0.5 * (got[k - 1][0] + got[k - 1][1])
                };
            }
        }
        start = stop;
    }
    Ok(out)
}

/// The stock's field at p after its first `upto` moves, clamped to [-limit, limit], in
/// double precision on the CPU, and the move whose sweep is the field there (None where
/// the stock box's own face is).
pub fn stock_f64(stock: &Stock, p: V3, upto: usize, limit: f64, tol: f64) -> (f64, Option<usize>) {
    let b = stock.bounds.sd(p);
    let mut best: Option<(f64, usize)> = None;
    let mut cut = (-b).min(limit);
    for k in stock.near_in(Aabb::new(p, p).grow(limit), 0..upto) {
        if let Some(s) = stock.sweep(k).sd_below(p, cut, tol) {
            cut = s;
            best = Some((s, k));
        }
    }
    match best {
        Some((u, k)) if -u > b => ((-u).clamp(-limit, limit), Some(k)),
        _ => (b.clamp(-limit, limit), None),
    }
}

/// How deep the cut goes into the part from its surface point p along the inward normal
/// (-n): the first material met, stepping by the stock's field in double precision (it
/// never reads past the stock's surface), to a ten-thousandth of a millimeter, at most
/// 50 mm.
pub fn normal_depth(stock: &Stock, p: [f64; 3], n: [f32; 3], upto: usize) -> f64 {
    let at = |t: f64| V3::new(p[0] - t * n[0] as f64, p[1] - t * n[1] as f64, p[2] - t * n[2] as f64);
    let mut t = 0.0;
    while t < 50.0 {
        let v = stock_f64(stock, at(t), upto, 10.0 * LIMIT, 1e-6).0;
        if v <= 0.0 {
            return t;
        }
        t += v.max(1e-4);
    }
    t
}

/// One gouge: its deepest point (by the field), how deep the cut goes there along the
/// part's normal, the most the field reads over the gouge (a distance into the part every
/// one of its points is cut at least), its area and the volume the field bounds from
/// below, and the move that cut at the deepest point.
#[derive(Clone, Debug)]
pub struct Gouge {
    pub at: [f64; 3],
    pub depth: f64,
    pub field: f64,
    pub area: f64,
    pub volume: f64,
    pub points: usize,
    pub by: Option<usize>,
}

/// Areas over the part's surface, mm^2.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bands {
    pub part: f64,
    pub on_size: f64,
    pub left: f64,
    pub gouged: f64,
    /// Stock left by thickness: TOL to 0.1 mm, 0.1 to 0.5, over 0.5.
    pub left_by: [f64; 3],
}

/// What was left at a point of the part's surface, along its normal (see the module's
/// notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Left,
    OnSize,
    Gouged,
}

impl Bands {
    /// A point of area w in band b, the field there v (stock left at least -v thick).
    fn add(&mut self, w: f64, b: Band, v: f64) {
        self.part += w;
        match b {
            Band::Gouged => self.gouged += w,
            Band::OnSize => self.on_size += w,
            Band::Left => {
                self.left += w;
                self.left_by[if -v <= 0.1 { 0 } else if -v <= 0.5 { 1 } else { 2 }] += w;
            }
        }
    }
}

/// What the comparison found.
pub struct Analysis {
    pub step: f64,
    pub upto: usize,
    pub moves: usize,
    pub points: usize,
    pub bands: Bands,
    /// The same over the faces whose normal points up (the faces a top-down check sees),
    /// down, and to the side.
    pub up: Bands,
    pub down: Bands,
    pub side: Bands,
    pub gouges: Vec<Gouge>,
    pub report: String,
}

/// Gouges whose depth is read again on a finer step around their deepest point, deepest
/// first; the rest keep the bound from the step itself.
const REFINED: usize = 64;

/// What `analyze` shows a batch to: its points, the stock's field at them, their bands.
pub type Visit<'v> = dyn FnMut(&[Sample], &[f64], &[Band]) + 'v;

/// Cells to a tile's side, and tiles read together: a few million points a batch.
const TILE_CELLS: usize = 256;
const BATCH: usize = 12;

/// The part's surface inside [lo, hi] against the stock after its first `upto` moves
/// (`generic`: sample the surface the general way even for a prismatic part), batch by
/// batch: `visit` sees each batch's points, the stock's field at them and their bands
/// before they are dropped; `say` hears each stage.
#[allow(clippy::too_many_arguments)]
pub fn analyze(
    frep: &Frep,
    lo: [f64; 3],
    hi: [f64; 3],
    stock: &Stock,
    gpu: &Fields,
    upto: usize,
    step: f64,
    generic: bool,
    say: &dyn Fn(&str),
    visit: &mut Visit,
) -> Result<Analysis, String> {
    let clock = Instant::now();
    let upto = upto.min(stock.len());
    let surface = Surface::new(frep, lo, hi, step, generic);
    let t_grid = clock.elapsed().as_secs_f64();
    say(&format!("the part on a grid of {} x {} x {} cells in {t_grid:.1} s", surface.n[0], surface.n[1], surface.n[2]));
    let tol = 1e-4;
    let d = TOL * (1.0 + 1e-6);
    let (mut t_points, mut t_stock, mut t_along) = (0.0, 0.0, 0.0);
    let mut count = 0usize;
    let mut bands = Bands::default();
    let (mut up, mut down, mut side) = (Bands::default(), Bands::default(), Bands::default());
    // The gouged points, with the field at them.
    let mut kept: Vec<(Sample, f64)> = Vec::new();
    let tiles = surface.tiles(TILE_CELLS);
    // The next batch's points are found while this one's are read: one batch waits ahead.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<Sample>, f64)>(1);
    std::thread::scope(|scope| -> Result<(), String> {
        let (surface, tiles) = (&surface, &tiles);
        scope.spawn(move || {
            for batch in tiles.chunks(BATCH) {
                let t0 = Instant::now();
                let samples: Vec<Sample> = batch.par_iter().flat_map_iter(|&t| surface.points(t)).collect();
                if tx.send((samples, t0.elapsed().as_secs_f64())).is_err() {
                    return;
                }
            }
        });
        // Owned here, so an early return drops it and the finding thread stops.
        let rx = rx;
        for (b, (samples, found)) in rx.iter().enumerate() {
            t_points += found;
            let t1 = Instant::now();
            let pts: Vec<[f64; 3]> = samples.iter().map(|s| s.p).collect();
            let values = stock_at(stock, gpu, &pts, upto, LIMIT, tol)?;
            t_stock += t1.elapsed().as_secs_f64();
            // The stock d inside and d outside the part along each point's normal: from the
            // field at p where it is past d, else read there.
            let t2 = Instant::now();
            let at_depth = |i: usize, t: f64| -> [f64; 3] { std::array::from_fn(|a| samples[i].p[a] + t * samples[i].n[a] as f64) };
            let close: Vec<usize> = (0..samples.len()).filter(|&i| values[i].abs() <= d).collect();
            let inside = stock_at(stock, gpu, &close.iter().map(|&i| at_depth(i, -d)).collect::<Vec<_>>(), upto, LIMIT, tol)?;
            let outside = stock_at(stock, gpu, &close.iter().map(|&i| at_depth(i, d)).collect::<Vec<_>>(), upto, LIMIT, tol)?;
            let mut air_in: Vec<bool> = values.iter().map(|&v| v > d).collect();
            let mut stock_out: Vec<bool> = values.iter().map(|&v| v < -d).collect();
            for (k, &i) in close.iter().enumerate() {
                air_in[i] = inside[k] > 0.0;
                stock_out[i] = outside[k] < 0.0;
            }
            // Where the stock is air d inside, whether that point lies deeper than TOL in the part.
            let candidates: Vec<usize> = (0..samples.len()).filter(|&i| air_in[i]).collect();
            let deep: Vec<bool> = candidates.par_iter().map(|&i| frep.eval(at_depth(i, -d)).value < -TOL).collect();
            let mut bands_of: Vec<Band> = stock_out.iter().map(|&o| if o { Band::Left } else { Band::OnSize }).collect();
            for (&i, &deep) in candidates.iter().zip(&deep) {
                if deep {
                    bands_of[i] = Band::Gouged;
                }
            }
            t_along += t2.elapsed().as_secs_f64();
            for ((s, &v), &band) in samples.iter().zip(&values).zip(&bands_of) {
                bands.add(s.w as f64, band, v);
                if s.n[2] > 0.999 {
                    up.add(s.w as f64, band, v);
                } else if s.n[2] < -0.999 {
                    down.add(s.w as f64, band, v);
                } else {
                    side.add(s.w as f64, band, v);
                }
                if band == Band::Gouged {
                    kept.push((*s, v));
                }
            }
            visit(&samples, &values, &bands_of);
            count += samples.len();
            if b % 8 == 7 {
                say(&format!("{count} points, {} of {} tiles", ((b + 1) * BATCH).min(tiles.len()), tiles.len()));
            }
        }
        Ok(())
    })?;
    say(&format!("{count} points (finding them took {t_points:.1} s beside the rest), the stock's field at them in {t_stock:.1} s, read along their normals in {t_along:.1} s"));
    // Gouges: gouged points within two steps of each other.
    let mut parent: Vec<usize> = (0..kept.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let reach = 2.0 * step;
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let key = |p: [f64; 3]| -> [i64; 3] { std::array::from_fn(|a| (p[a] / reach).floor() as i64) };
    for (g, (s, _)) in kept.iter().enumerate() {
        grid.entry(key(s.p)).or_default().push(g);
    }
    for (g, (s, _)) in kept.iter().enumerate() {
        let k = key(s.p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(near) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        for &h in near {
                            let q = kept[h].0.p;
                            let d2: f64 = (0..3).map(|a| (q[a] - s.p[a]).powi(2)).sum();
                            if d2 <= reach * reach {
                                let (ra, rb) = (root(&mut parent, g), root(&mut parent, h));
                                if ra != rb {
                                    parent[ra] = rb;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for g in 0..kept.len() {
        let r = root(&mut parent, g);
        groups.entry(r).or_default().push(g);
    }
    say(&format!("{} gouged points in {} gouges", kept.len(), groups.len()));
    let mut gouges = Vec::new();
    let mut order: Vec<&Vec<usize>> = groups.values().collect();
    let deepest_of = |m: &Vec<usize>| *m.iter().max_by(|&&a, &&b| kept[a].1.total_cmp(&kept[b].1)).unwrap_or(&m[0]);
    order.sort_by(|a, b| kept[deepest_of(b)].1.total_cmp(&kept[deepest_of(a)].1));
    for (rank, members) in order.into_iter().enumerate() {
        let area: f64 = members.iter().map(|&g| kept[g].0.w as f64).sum();
        let volume: f64 = members.iter().map(|&g| kept[g].0.w as f64 * kept[g].1.max(0.0)).sum();
        let (s, v) = kept[deepest_of(members)];
        let field = v.max(0.0);
        if rank >= REFINED {
            gouges.push(Gouge { at: s.p, depth: normal_depth(stock, s.p, s.n, upto), field, area, volume, points: members.len(), by: None });
            continue;
        }
        // Around the deepest point, the surface again on an eighth of the step, read with a
        // larger limit.
        let fine = step / 8.0;
        let c = s.p;
        let around = Surface::new(frep, std::array::from_fn(|a| c[a] - step), std::array::from_fn(|a| c[a] + step), fine, generic).all();
        let around_pts: Vec<[f64; 3]> = around.iter().map(|s| s.p).collect();
        let around_v = stock_at(stock, gpu, &around_pts, upto, 10.0 * LIMIT, tol)?;
        let (k, best) = around_v.iter().enumerate().fold((usize::MAX, field), |(k, b), (i, &v)| if v > b { (i, v) } else { (k, b) });
        let (at, n) = if k == usize::MAX { (c, s.n) } else { (around_pts[k], around[k].n) };
        let (_, by) = stock_f64(stock, V3::new(at[0], at[1], at[2]), upto, 10.0 * LIMIT, 1e-6);
        gouges.push(Gouge { at, depth: normal_depth(stock, at, n, upto), field: best, area, volume, points: members.len(), by });
    }
    gouges.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    let report = format!(
        "{count} points on the part's surface at {step} mm in {:.1} s; part {:.4} m2: on size {:.4} m2, stock left {:.4} m2 (to 0.1 mm {:.4}, 0.1 to 0.5 {:.4}, deeper {:.4}), gouged {:.1} mm2 in {} gouges; faces up: on size {:.4} m2, left {:.4} m2, gouged {:.1} mm2; down: {:.4}, {:.4}, {:.1}; walls: {:.4}, {:.4}, {:.1}",
        clock.elapsed().as_secs_f64(),
        bands.part / 1e6,
        bands.on_size / 1e6,
        bands.left / 1e6,
        bands.left_by[0] / 1e6,
        bands.left_by[1] / 1e6,
        bands.left_by[2] / 1e6,
        bands.gouged,
        gouges.len(),
        up.on_size / 1e6,
        up.left / 1e6,
        up.gouged,
        down.on_size / 1e6,
        down.left / 1e6,
        down.gouged,
        side.on_size / 1e6,
        side.left / 1e6,
        side.gouged
    );
    Ok(Analysis { step, upto, moves: stock.len(), points: count, bands, up, down, side, gouges, report })
}

/// A file's size and FNV-1a 64-bit hash, to tell a stale record.
pub fn fingerprint(path: &std::path::Path) -> Result<(u64, u64), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok((bytes.len() as u64, h))
}

fn quoted(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Analysis {
    /// The analysis record: the inputs (name, size, hash; `extra` as given), the bands'
    /// areas and every gouge, as JSON.
    pub fn json(&self, inputs: &[(&str, &std::path::Path)], extra: &[(&str, String)]) -> Result<String, String> {
        let mut s = String::from("{\n  \"record\": \"icam-sim analysis\",\n  \"version\": 1,\n  \"inputs\": {\n");
        let mut items = Vec::new();
        for (name, path) in inputs {
            let (size, hash) = fingerprint(path)?;
            let file = path.file_name().map_or(String::new(), |f| f.to_string_lossy().into_owned());
            items.push(format!("    {}: {{\"file\": {}, \"bytes\": {size}, \"fnv1a64\": \"{hash:016x}\"}}", quoted(name), quoted(&file)));
        }
        for (name, value) in extra {
            items.push(format!("    {}: {value}", quoted(name)));
        }
        s.push_str(&items.join(",\n"));
        let b = &self.bands;
        let u = &self.up;
        s.push_str(&format!(
            "\n  }},\n  \"playhead\": {},\n  \"moves\": {},\n  \"step_mm\": {},\n  \"tolerance_mm\": {TOL},\n  \"points\": {},\n  \"area_mm2\": {{\"part\": {:.1}, \"on_size\": {:.1}, \"stock_left\": {:.1}, \"stock_left_to_0.1\": {:.1}, \"stock_left_0.1_to_0.5\": {:.1}, \"stock_left_over_0.5\": {:.1}, \"gouged\": {:.3}}},\n  \"faces_up_mm2\": {{\"on_size\": {:.1}, \"stock_left\": {:.1}, \"gouged\": {:.3}}},\n  \"faces_down_mm2\": {{\"on_size\": {:.1}, \"stock_left\": {:.1}, \"gouged\": {:.3}}},\n  \"walls_mm2\": {{\"on_size\": {:.1}, \"stock_left\": {:.1}, \"gouged\": {:.3}}},\n  \"gouges\": [",
            self.upto, self.moves, self.step, self.points, b.part, b.on_size, b.left, b.left_by[0], b.left_by[1], b.left_by[2], b.gouged, u.on_size, u.left, u.gouged, self.down.on_size, self.down.left, self.down.gouged, self.side.on_size, self.side.left, self.side.gouged
        ));
        let g: Vec<String> = self
            .gouges
            .iter()
            .map(|g| {
                format!(
                    "\n    {{\"at\": [{:.4}, {:.4}, {:.4}], \"depth_mm\": {:.4}, \"field_mm\": {:.4}, \"area_mm2\": {:.3}, \"volume_at_least_mm3\": {:.4}, \"points\": {}, \"move\": {}}}",
                    g.at[0],
                    g.at[1],
                    g.at[2],
                    g.depth,
                    g.field,
                    g.area,
                    g.volume,
                    g.points,
                    g.by.map_or("null".to_string(), |k| k.to_string())
                )
            })
            .collect();
        s.push_str(&g.join(","));
        s.push_str(if g.is_empty() { "]\n}\n" } else { "\n  ]\n}\n" });
        Ok(s)
    }
}
