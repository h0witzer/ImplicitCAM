//! The stock's surface taken from its field alone (the bake): nothing is assembled from
//! the moves' own surfaces, so nothing depends on the order the moves came in and no
//! decision between two coincident surfaces is ever made. The stock at any moment is the
//! box less the moves so far, F = max(box, -min s_k), whatever their order.
//!
//! Finding the surface: space is split into cells, coarse to fine. The field never reads
//! more than the true distance to the surface, so a cell whose center reads farther from
//! zero than half the cell's diagonal holds none of the surface and is dropped whole; a
//! cell no move can reach holds at most the box's faces; every other cell splits, down to
//! bricks of 8 x 8 x 8 cells of side h, where the field is read at every node (on the GPU,
//! each brick reading the moves that can come near it; on the CPU where only the box can).
//!
//! Drawing it (surface nets): a vertex in every cell the zero crosses, at the mean of the
//! crossings on its edges; a quad across every edge the zero crosses, joining the four
//! cells around that edge. Each node has one value, read from one brick, so every crossed
//! edge's four cells all hold vertices and the surface closes on itself: no holes, by
//! construction (`Baked::open` counts the crossed edges that could not close, the region's
//! own border aside). Vertices lie within a cell of the surface; sharp edges are rounded
//! to the cell (dual contouring with the field's own normals places them exactly).

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::geom::{cross, dot, scale, unit, Aabb, V3};
use crate::gpu::{Fields, Lists};
use crate::stock::Stock;

/// Cells to a brick's side.
const B: i64 = 8;
/// Nodes to a brick's side.
const NB: i64 = B + 1;

/// A drawn piece of the surface, three-cornered: corners (point, outward normal), wound
/// counterclockwise seen from outside, and whether it lies on the box's own faces (raw
/// stock).
pub type Drawn = ([(V3, V3); 3], bool);

pub struct Baked {
    pub pieces: Vec<Drawn>,
    /// Crossed edges inside the region that could not be closed (zero when watertight).
    pub open: usize,
    /// What was baked and how long each stage took.
    pub report: String,
}

/// A brick's node values (B + 1 to a side, x fastest).
struct Brick {
    at: [i64; 3],
    v: Vec<f32>,
}

/// The surface of `stock` within `region`, from cells of side `h`, the field read to
/// within `tol`.
pub fn bake(stock: &Stock, gpu: &Fields, h: f64, region: Aabb, tol: f64) -> Result<Baked, String> {
    let clock = Instant::now();
    let pad = 2.0 * h;
    let lo = region.min - V3::new(pad, pad, pad);
    let cells = [((region.max.x - region.min.x + 2.0 * pad) / h).ceil() as i64, ((region.max.y - region.min.y + 2.0 * pad) / h).ceil() as i64, ((region.max.z - region.min.z + 2.0 * pad) / h).ceil() as i64];
    let nbricks = cells.map(|c| (c + B - 1) / B);
    let node = |i: i64, j: i64, k: i64| V3::new(lo.x + h * i as f64, lo.y + h * j as f64, lo.z + h * k as f64);
    let o = region.center();
    // Coarse to fine: cells of 2^level bricks.
    let mut level = 0;
    while (1i64 << level) < *nbricks.iter().max().unwrap_or(&1) && level < 6 {
        level += 1;
    }
    let span = 1i64 << level;
    let mut cellz: Vec<[i64; 3]> = Vec::new();
    for x in (0..nbricks[0]).step_by(span as usize) {
        for y in (0..nbricks[1]).step_by(span as usize) {
            for z in (0..nbricks[2]).step_by(span as usize) {
                cellz.push([x, y, z]);
            }
        }
    }
    let mut found_secs = 0.0;
    let mut checked = 0usize;
    let kept: Vec<[i64; 3]> = loop {
        let size = (1i64 << level) * B;
        let r = 0.5 * h * size as f64 * 3f64.sqrt();
        let centers: Vec<V3> = cellz.iter().map(|c| node(c[0] * B + size / 2, c[1] * B + size / 2, c[2] * B + size / 2)).collect();
        let lists: Vec<Vec<u32>> = centers.par_iter().map(|&p| stock.near(Aabb::new(p - V3::new(r, r, r), p + V3::new(r, r, r))).into_iter().map(|k| k as u32).collect()).collect();
        let mut ask = Lists::default();
        let mut asked = Vec::new();
        for (i, l) in lists.iter().enumerate() {
            if !l.is_empty() {
                ask.pts.push(centers[i]);
                ask.first.push(ask.idx.len() as u32);
                ask.count.push(l.len() as u32);
                ask.idx.extend(l);
                asked.push(i);
            }
        }
        let t = Instant::now();
        let got = gpu.listed(stock, o, &ask, 1.01 * r, tol)?.ok_or("the kernels cannot place a move")?;
        found_secs += t.elapsed().as_secs_f64();
        checked += cellz.len();
        let mut iv: Vec<[f64; 2]> = centers.iter().map(|&p| { let b = stock.bounds.sd(p); [b, b] }).collect();
        for (&i, v) in asked.iter().zip(got) {
            iv[i] = v;
        }
        let holds = |v: [f64; 2]| !(v[0] > r * (1.0 + 1e-6) || v[1] < -r * (1.0 + 1e-6));
        let next: Vec<[i64; 3]> = cellz.iter().zip(&iv).filter(|(_, v)| holds(**v)).map(|(c, _)| *c).collect();
        if level == 0 {
            break next;
        }
        let half = 1i64 << (level - 1);
        cellz = next
            .iter()
            .flat_map(|c| {
                let c = *c;
                (0..8).map(move |q| [c[0] + half * (q & 1), c[1] + half * ((q >> 1) & 1), c[2] + half * ((q >> 2) & 1)])
            })
            .filter(|c| c[0] < nbricks[0] && c[1] < nbricks[1] && c[2] < nbricks[2])
            .collect();
        level -= 1;
    };
    let found = clock.elapsed().as_secs_f64();
    // Every node of every brick kept: from the GPU where moves can come near, from the
    // box's own distance where none can.
    let limit = 2.0 * h;
    let t_nodes = Instant::now();
    let brick_lists: Vec<Vec<u32>> = kept
        .par_iter()
        .map(|b| {
            let a = node(b[0] * B, b[1] * B, b[2] * B);
            let z = node(b[0] * B + B, b[1] * B + B, b[2] * B + B);
            stock.near(Aabb::new(a, z).grow(limit)).into_iter().map(|k| k as u32).collect()
        })
        .collect();
    let mut bricks: Vec<Brick> = Vec::with_capacity(kept.len());
    let mut gpu_nodes = 0usize;
    let mut gpu_secs = 0.0;
    let mut pending: Vec<usize> = Vec::new();
    for (i, b) in kept.iter().enumerate() {
        if brick_lists[i].is_empty() {
            let v = (0..NB * NB * NB)
                .map(|q| {
                    let p = node(b[0] * B + q % NB, b[1] * B + (q / NB) % NB, b[2] * B + q / (NB * NB));
                    stock.bounds.sd(p).clamp(-limit, limit) as f32
                })
                .collect();
            bricks.push(Brick { at: *b, v });
        } else {
            bricks.push(Brick { at: *b, v: Vec::new() });
            pending.push(i);
        }
    }
    for chunk in pending.chunks(2048) {
        let mut ask = Lists::default();
        for &i in chunk {
            let b = kept[i];
            let first = ask.idx.len() as u32;
            ask.idx.extend(&brick_lists[i]);
            for q in 0..NB * NB * NB {
                ask.pts.push(node(b[0] * B + q % NB, b[1] * B + (q / NB) % NB, b[2] * B + q / (NB * NB)));
                ask.first.push(first);
                ask.count.push(brick_lists[i].len() as u32);
            }
        }
        let t = Instant::now();
        let got = gpu.listed(stock, o, &ask, limit, tol)?.ok_or("the kernels cannot place a move")?;
        gpu_secs += t.elapsed().as_secs_f64();
        gpu_nodes += ask.pts.len();
        for (n, &i) in chunk.iter().enumerate() {
            let per = (NB * NB * NB) as usize;
            bricks[i].v = got[n * per..(n + 1) * per].iter().map(|x| (0.5 * (x[0] + x[1])) as f32).collect();
        }
    }
    let nodes_secs = t_nodes.elapsed().as_secs_f64();
    // One value per node: from the brick that owns it (the one it is a low corner of) when
    // that brick was read, else from the first read brick holding it.
    let index: HashMap<[i64; 3], usize> = bricks.iter().enumerate().map(|(i, b)| (b.at, i)).collect();
    let value = |i: i64, j: i64, k: i64, own: usize| -> f32 {
        let fl = |a: i64| a.div_euclid(B);
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let at = [fl(i) - dx, fl(j) - dy, fl(k) - dz];
                    if let Some(&b) = index.get(&at) {
                        let (li, lj, lk) = (i - at[0] * B, j - at[1] * B, k - at[2] * B);
                        if (0..NB).contains(&li) && (0..NB).contains(&lj) && (0..NB).contains(&lk) {
                            return bricks[b].v[(li + NB * (lj + NB * lk)) as usize];
                        }
                    }
                }
            }
        }
        let b = &bricks[own];
        bricks[own].v[((i - b.at[0] * B) + NB * ((j - b.at[1] * B) + NB * (k - b.at[2] * B))) as usize]
    };
    // Vertices: one in every crossed cell.
    let t_nets = Instant::now();
    const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
    let verts: Vec<Vec<(u32, V3, V3)>> = bricks
        .par_iter()
        .enumerate()
        .map(|(bi, b)| {
            let mut out = Vec::new();
            for q in 0..B * B * B {
                let (ci, cj, ck) = (b.at[0] * B + q % B, b.at[1] * B + (q / B) % B, b.at[2] * B + q / (B * B));
                let c: [f32; 8] = std::array::from_fn(|n| value(ci + (n as i64 & 1), cj + ((n as i64 >> 1) & 1), ck + ((n as i64 >> 2) & 1), bi));
                let neg = c.iter().filter(|&&x| x < 0.0).count();
                if neg == 0 || neg == 8 {
                    continue;
                }
                let corner = |n: usize| V3::new((n & 1) as f64, ((n >> 1) & 1) as f64, ((n >> 2) & 1) as f64);
                let mut sum = V3::default();
                let mut m = 0.0;
                for &(a, z) in &EDGES {
                    if (c[a] < 0.0) != (c[z] < 0.0) {
                        let t = c[a] as f64 / (c[a] as f64 - c[z] as f64);
                        sum = sum + corner(a) + scale(corner(z) - corner(a), t);
                        m += 1.0;
                    }
                }
                let u = scale(sum, 1.0 / m);
                // The trilinear field's gradient there: the outward normal (F grows into air).
                let f = |n: usize| c[n] as f64;
                let lerp = |a: f64, z: f64, t: f64| a + (z - a) * t;
                let gx = lerp(lerp(f(1) - f(0), f(3) - f(2), u.y), lerp(f(5) - f(4), f(7) - f(6), u.y), u.z);
                let gy = lerp(lerp(f(2) - f(0), f(3) - f(1), u.x), lerp(f(6) - f(4), f(7) - f(5), u.x), u.z);
                let gz = lerp(lerp(f(4) - f(0), f(5) - f(1), u.x), lerp(f(6) - f(2), f(7) - f(3), u.x), u.y);
                out.push((q as u32, node(ci, cj, ck) + scale(u, h), unit(V3::new(gx, gy, gz))));
            }
            out
        })
        .collect();
    let mut base = vec![0u32; bricks.len() + 1];
    for i in 0..bricks.len() {
        base[i + 1] = base[i] + verts[i].len() as u32;
    }
    let cell_vertex: Vec<HashMap<u32, u32>> = verts.iter().enumerate().map(|(i, v)| v.iter().enumerate().map(|(n, x)| (x.0, base[i] + n as u32)).collect()).collect();
    let all: Vec<(V3, V3)> = verts.iter().flatten().map(|x| (x.1, x.2)).collect();
    let vertex_of = |i: i64, j: i64, k: i64| -> Option<u32> {
        let at = [i.div_euclid(B), j.div_euclid(B), k.div_euclid(B)];
        let &b = index.get(&at)?;
        let q = (i - at[0] * B) + B * ((j - at[1] * B) + B * (k - at[2] * B));
        cell_vertex[b].get(&(q as u32)).copied()
    };
    // Quads: one across every crossed edge, from the four cells around it.
    let inside = |i: i64, j: i64, k: i64| i >= 1 && j >= 1 && k >= 1 && i < cells[0] && j < cells[1] && k < cells[2];
    let quads: Vec<(Vec<[u32; 4]>, usize)> = bricks
        .par_iter()
        .enumerate()
        .map(|(bi, b)| {
            let mut out = Vec::new();
            let mut open = 0;
            for q in 0..B * B * B {
                let (i, j, k) = (b.at[0] * B + q % B, b.at[1] * B + (q / B) % B, b.at[2] * B + q / (B * B));
                let v0 = value(i, j, k, bi);
                for axis in 0..3 {
                    let (di, dj, dk) = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][axis];
                    let v1 = value(i + di, j + dj, k + dk, bi);
                    if (v0 < 0.0) == (v1 < 0.0) {
                        continue;
                    }
                    // The four cells around the edge, in turn about it.
                    let around: [(i64, i64, i64); 4] = match axis {
                        0 => [(i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)],
                        1 => [(i - 1, j, k - 1), (i - 1, j, k), (i, j, k), (i, j, k - 1)],
                        _ => [(i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)],
                    };
                    let ids: Vec<Option<u32>> = around.iter().map(|&(a, b, c)| vertex_of(a, b, c)).collect();
                    if ids.iter().all(|x| x.is_some()) {
                        out.push([ids[0].unwrap(), ids[1].unwrap(), ids[2].unwrap(), ids[3].unwrap()]);
                    } else if around.iter().all(|&(a, b, c)| inside(a, b, c)) {
                        open += 1;
                    }
                }
            }
            (out, open)
        })
        .collect();
    let open: usize = quads.iter().map(|x| x.1).sum();
    let pieces: Vec<Drawn> = quads
        .into_par_iter()
        .flat_map_iter(|(qs, _)| qs.into_iter())
        .flat_map_iter(|q| {
            let c = q.map(|i| all[i as usize]);
            let one = |a: (V3, V3), b: (V3, V3), d: (V3, V3)| -> Option<Drawn> {
                let g = cross(b.0 - a.0, d.0 - a.0);
                if g.len() < 1e-14 {
                    return None;
                }
                let corners = if dot(g, a.1 + b.1 + d.1) >= 0.0 { [a, b, d] } else { [a, d, b] };
                let mid = scale(a.0 + b.0 + d.0, 1.0 / 3.0);
                Some((corners, stock.bounds.sd(mid).abs() < 0.3 * h))
            };
            [one(c[0], c[1], c[2]), one(c[0], c[2], c[3])].into_iter().flatten()
        })
        .collect();
    let nets = t_nets.elapsed().as_secs_f64();
    let report = format!(
        "h {h} mm: {checked} cells checked, {} bricks ({} read on the GPU, {} nodes) in {:.2} s finding (GPU {:.2} s) + {:.2} s nodes (GPU {:.2} s) + {:.2} s surface nets: {} pieces, {} crossed edges left open",
        bricks.len(),
        pending.len(),
        gpu_nodes,
        found,
        found_secs,
        nodes_secs,
        gpu_secs,
        nets,
        pieces.len(),
        open
    );
    Ok(Baked { pieces, open, report })
}
