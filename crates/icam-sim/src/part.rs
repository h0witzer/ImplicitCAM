//! The part on the stock's grid: the height of its top over every node, read from the
//! compiled F-rep, and what the program left at each node against it.
//!
//! The prismatic F-rep is f = min_k max(sd_Sk(x, y), z - z_k, z_bot - z) (see
//! `icam_core::frep`): column k stands on z_bot up to z_k over cross-section S_k. The
//! part's top over a node is the highest z_k whose section holds it. Each section's sd
//! is an exact distance, so one value decides a whole block of nodes when it is farther
//! from zero than the block's half diagonal; blocks split only near a section's
//! boundary, and only the nodes within about a step of it are read one by one.
//!
//! What was left: the stock height H against the part. Above the part's top, H - top
//! of material stands there. Below it, the removed span (H, top] reaches into the part
//! as deep as the most of -f(x, y, z) over that span, which per column is
//! min(-sd_k, z_k - max(H, z_bot), (z_k - z_bot) / 2); past the shop tolerance that
//! depth is a gouge. A node on a wall the program cut to size is not one: its depth is
//! its distance into the wall, under the tolerance.

use icam_core::frep::{Frep, Node};
use icam_core::geom::V2;
use rayon::prelude::*;

use crate::{Grid, StockField};

/// The shop tolerance, mm (0.001 in), the checker's.
pub const TOL: f64 = 0.0254;

/// One column of the F-rep: section `sec`, standing from `bottom` up to `top`.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    pub sec: u32,
    pub top: f64,
    pub bottom: f64,
}

/// The F-rep's columns, read off its tree (each is max(section, below z_k, above z_bot)).
pub fn columns(frep: &Frep) -> Vec<Column> {
    let node = |i: u32| &frep.nodes[i as usize];
    frep.nodes
        .iter()
        .filter_map(|n| match n {
            Node::Max(c) if c.len() == 3 => match (node(c[0]), node(c[1]), node(c[2])) {
                (Node::Section { sec }, Node::Below { z: top }, Node::Above { z: bottom }) => Some(Column { sec: *sec, top: *top, bottom: *bottom }),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// The part's top over every node of a grid (mm, NaN where no column stands), row by row.
pub struct PartField {
    pub grid: Grid,
    pub top: Vec<f32>,
    pub columns: Vec<Column>,
    /// Section distances read to build it.
    pub reads: usize,
}

/// Nodes per side of the blocks the build starts from (one task each).
const TILE: usize = 64;

impl PartField {
    pub fn new(frep: &Frep, grid: Grid) -> PartField {
        let columns = columns(frep);
        let boxes: Vec<(V2, V2)> = columns.iter().map(|c| frep.sections[c.sec as usize].bbox()).collect();
        let (tx, ty) = (grid.nx.div_ceil(TILE), grid.ny.div_ceil(TILE));
        let tiles: Vec<(Vec<f32>, usize)> = (0..tx * ty)
            .into_par_iter()
            .map(|t| {
                let (i0, j0) = ((t % tx) * TILE, (t / tx) * TILE);
                let (i1, j1) = ((i0 + TILE).min(grid.nx), (j0 + TILE).min(grid.ny));
                let w = i1 - i0;
                let mut top = vec![f32::NAN; w * (j1 - j0)];
                let mut reads = 0;
                for (c, b) in columns.iter().zip(&boxes) {
                    if grid.x(i1 - 1) < b.0.x || grid.x(i0) > b.1.x || grid.y(j1 - 1) < b.0.y || grid.y(j0) > b.1.y {
                        continue;
                    }
                    let z = c.top as f32;
                    let mut inside = |i: usize, j: usize| {
                        let v = &mut top[(j - j0) * w + (i - i0)];
                        *v = v.max(z);
                    };
                    decide(frep, c.sec, &grid, [i0, i1, j0, j1], &mut inside, &mut reads);
                }
                (top, reads)
            })
            .collect();
        let mut top = vec![f32::NAN; grid.nx * grid.ny];
        let mut reads = 0;
        for (t, (tile, n)) in tiles.into_iter().enumerate() {
            let (i0, j0) = ((t % tx) * TILE, (t / tx) * TILE);
            let w = (i0 + TILE).min(grid.nx) - i0;
            for (r, row) in tile.chunks(w).enumerate() {
                let k = (j0 + r) * grid.nx + i0;
                top[k..k + w].copy_from_slice(row);
            }
            reads += n;
        }
        PartField { grid, top, columns, reads }
    }

    /// The reference: every column's section read at the node itself, no blocks.
    pub fn top_at(frep: &Frep, columns: &[Column], x: f64, y: f64) -> f64 {
        columns.iter().filter(|c| frep.section_sd(c.sec, V2::new(x, y)).sd < 0.0).map(|c| c.top).fold(f64::NAN, f64::max)
    }
}

/// Calls `inside` for every node of the block [i0, i1) x [j0, j1) inside section `sec`.
fn decide(frep: &Frep, sec: u32, grid: &Grid, b: [usize; 4], inside: &mut impl FnMut(usize, usize), reads: &mut usize) {
    let [i0, i1, j0, j1] = b;
    let (w, h) = (i1 - i0, j1 - j0);
    if w * h <= 4 {
        for j in j0..j1 {
            for i in i0..i1 {
                *reads += 1;
                if frep.section_sd(sec, V2::new(grid.x(i), grid.y(j))).sd < 0.0 {
                    inside(i, j);
                }
            }
        }
        return;
    }
    let c = V2::new(0.5 * (grid.x(i0) + grid.x(i1 - 1)), 0.5 * (grid.y(j0) + grid.y(j1 - 1)));
    let half = 0.5 * grid.step * ((w - 1) as f64).hypot((h - 1) as f64);
    *reads += 1;
    let sd = frep.section_sd(sec, c).sd;
    if sd.abs() > half + 1e-9 {
        if sd < 0.0 {
            for j in j0..j1 {
                for i in i0..i1 {
                    inside(i, j);
                }
            }
        }
        return;
    }
    let (im, jm) = (i0 + w / 2, j0 + h / 2);
    for q in [[i0, im, j0, jm], [im, i1, j0, jm], [i0, im, jm, j1], [im, i1, jm, j1]] {
        if q[0] < q[1] && q[2] < q[3] {
            decide(frep, sec, grid, q, inside, reads);
        }
    }
}

/// How deep the span removed above stock height `h` at (x, y) reaches into the part:
/// the most of -f(x, y, z) over z > h, column by column (0 when it misses the part).
pub fn cut_depth(frep: &Frep, columns: &[Column], x: f64, y: f64, h: f64) -> f64 {
    let mut g = 0.0f64;
    for c in columns {
        let dz = (c.top - h.max(c.bottom)).min(0.5 * (c.top - c.bottom));
        if dz <= g {
            continue;
        }
        g = g.max((-frep.section_sd(c.sec, V2::new(x, y)).sd).min(dz));
    }
    g
}

/// What the program left at a node over the part (top `top`, stock height `h`): above
/// zero, material over the part's top; below zero, how deep the cut reached into the
/// part (a gouge past TOL); NaN where no part stands. The cut cannot reach deeper than
/// top - h, so only a node cut more than TOL under the top reads the sections.
pub fn left(frep: &Frep, columns: &[Column], x: f64, y: f64, top: f64, h: f64) -> f64 {
    if top.is_nan() {
        f64::NAN
    } else if h >= top - TOL {
        h - top
    } else {
        -cut_depth(frep, columns, x, y, h)
    }
}

/// Node classes, from what was left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// No part under the node: stock that is not the part.
    Waste,
    /// Material over the part's top, more than TOL.
    Left,
    /// Within TOL of the part.
    OnSize,
    /// Cut into the part, more than TOL.
    Gouge,
}

impl Class {
    pub fn of(left: f64) -> Class {
        if left.is_nan() {
            Class::Waste
        } else if left > TOL {
            Class::Left
        } else if left >= -TOL {
            Class::OnSize
        } else {
            Class::Gouge
        }
    }
}

/// What the program left at every node (see `left`), from the stock field and the part.
pub fn left_field(frep: &Frep, part: &PartField, stock: &StockField) -> Vec<f32> {
    let g = part.grid;
    let mut out = vec![0f32; g.nx * g.ny];
    out.par_chunks_mut(g.nx).enumerate().for_each(|(j, row)| {
        for (i, v) in row.iter_mut().enumerate() {
            let k = j * g.nx + i;
            *v = left(frep, &part.columns, g.x(i), g.y(j), part.top[k] as f64, stock.h[k] as f64) as f32;
        }
    });
    out
}

/// One connected patch of gouged nodes.
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub nodes: usize,
    /// The deepest node and its depth (mm).
    pub deepest: usize,
    pub depth: f32,
    /// Node box: (i0, j0, i1, j1), inclusive.
    pub span: [usize; 4],
}

/// The gouged nodes in connected patches (eight neighbors), largest first.
pub fn spots(grid: &Grid, left: &[f32]) -> Vec<Spot> {
    let gouged = |k: usize| Class::of(left[k] as f64) == Class::Gouge;
    let mut seen = vec![false; left.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for k0 in 0..left.len() {
        if seen[k0] || !gouged(k0) {
            continue;
        }
        seen[k0] = true;
        stack.push(k0);
        let (i, j) = (k0 % grid.nx, k0 / grid.nx);
        let mut s = Spot { nodes: 0, deepest: k0, depth: -left[k0], span: [i, j, i, j] };
        while let Some(k) = stack.pop() {
            let (i, j) = (k % grid.nx, k / grid.nx);
            s.nodes += 1;
            if -left[k] > s.depth {
                (s.depth, s.deepest) = (-left[k], k);
            }
            s.span = [s.span[0].min(i), s.span[1].min(j), s.span[2].max(i), s.span[3].max(j)];
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    let (a, b) = (i as i64 + di, j as i64 + dj);
                    if a < 0 || b < 0 || a >= grid.nx as i64 || b >= grid.ny as i64 {
                        continue;
                    }
                    let n = b as usize * grid.nx + a as usize;
                    if !seen[n] && gouged(n) {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        out.push(s);
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.nodes));
    out
}
