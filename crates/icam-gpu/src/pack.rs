//! Packs an exact part for the GPU: curves as flat f32 records in a local frame,
//! B-spline curves as flat rational Bezier pieces, and a uniform grid that tells
//! each query which curves and faces can matter.
//!
//! Grid cell (i, j) lists every boundary curve within `band + diag` of the cell
//! (diag = the cell's diagonal), sorted by face so the kernel finishes one face
//! before the next. That radius guarantees: if a face's nearest listed curve is
//! within `band` of the query, it is the face's true nearest curve (the true
//! nearest is within `band + diag` of the cell, so it is listed). The kernel then
//! has the exact distance and takes the sign from the topology: the outward
//! normal at an interior foot, the vertex pseudo-normal at an end.
//!
//! When a face's nearest listed curve is farther than `band`, the face's whole
//! boundary is farther than `band` from the query, so no boundary crosses the
//! cell and the query is on the same side as the cell's reference point: each
//! listed (cell, face) pair carries that side in bit 31 of its items (decided on
//! the CPU in f64 by ray crossings). Faces with no listed curve that contain the
//! cell are in the cell's cover list. The kernel is exact within `band` and
//! reports a clamp beyond it.

use icam_core::geom::{Curve, V2};
use icam_core::part::Part;
use icam_core::tool::{ProfEl, Tool};
use rayon::prelude::*;

/// Highest B-spline degree the kernel evaluates (8 control points per piece).
pub const MAX_DEGREE: usize = 7;

pub struct Packed {
    /// Subtracted from every x, y before converting to f32 (the part's center),
    /// so single precision keeps ~0.1 um across a full sheet.
    pub origin: V2,
    pub cell: f64,
    pub gx0: f64,
    pub gy0: f64,
    pub nx: u32,
    pub ny: u32,
    pub band: f64,
    /// 16 f32 per curve: (start, end), then (cx, cy, r, sweep) for arcs or
    /// (first piece, piece count) as bits for splines, (pn_start, pn_end),
    /// (kind, face, edge, 0) as bits. Kind 0 line, 1 arc, 2 spline.
    pub edges: Vec<[f32; 16]>,
    /// 36 f32 per flat Bezier piece: 8 homogeneous control points as vec4
    /// ((x - ox) w, (y - oy) w, w, 0), then (degree bits, thickness, source t0, source t1).
    pub bez: Vec<[f32; 36]>,
    pub face_z: Vec<f32>,
    /// [cell starts (ncell + 1) | cell items (curve | inside bit) | cover starts (ncell + 1) | cover items | cell reference points (2 f32 bits per cell)]
    pub idx: Vec<u32>,
    pub cell_start: u32,
    pub cell_items: u32,
    pub cover_start: u32,
    pub cover_items: u32,
    pub cell_ref: u32,
    /// Reference points nudged off a curve that passed within 1 um of the cell center.
    pub nudged: usize,
    pub item_count: usize,
    pub cover_count: usize,
}

pub fn pack(part: &Part, band: f64, cell: f64) -> Result<Packed, String> {
    let (lo, hi) = part.bbox();
    let origin = V2::new(0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y));
    let reg = band + cell * std::f64::consts::SQRT_2;
    let gx0 = lo.x - reg - origin.x;
    let gy0 = lo.y - reg - origin.y;
    let nx = (((hi.x - lo.x) + 2.0 * reg) / cell).ceil() as u32 + 1;
    let ny = (((hi.y - lo.y) + 2.0 * reg) / cell).ceil() as u32 + 1;
    let ncell = (nx * ny) as usize;
    let cell_of = |x: f64, y: f64| -> (i64, i64) { (((x - origin.x - gx0) / cell).floor() as i64, ((y - origin.y - gy0) / cell).floor() as i64) };
    let clampi = |i: i64, n: u32| i.clamp(0, n as i64 - 1) as u32;

    // Curves within band + diag of each cell (by box, so a superset).
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ncell];
    let boxes: Vec<(V2, V2)> = part.edges.iter().map(|e| e.curve.bbox()).collect();
    for (k, (a, b)) in boxes.iter().enumerate() {
        let (i0, j0) = cell_of(a.x - reg, a.y - reg);
        let (i1, j1) = cell_of(b.x + reg, b.y + reg);
        for j in clampi(j0, ny)..=clampi(j1, ny) {
            for i in clampi(i0, nx)..=clampi(i1, nx) {
                lists[(j * nx + i) as usize].push(k as u32);
            }
        }
    }
    lists.par_iter_mut().for_each(|l| l.sort_by_key(|&k| (part.edges[k as usize].face, k)));

    // Reference point per cell (in the local frame, as the kernel sees it in f32),
    // kept at least 1 um from every listed curve, and the inside bit per listed
    // (cell, face).
    let refs: Vec<(f32, f32, bool)> = (0..ncell)
        .into_par_iter()
        .map(|c| {
            let (i, j) = ((c as u32 % nx) as f64, (c as u32 / nx) as f64);
            let mut nudged = false;
            for t in 0..16 {
                let (fx, fy) = (0.5 + 0.037 * t as f64, 0.5 + 0.029 * t as f64);
                let rx = (gx0 + (i + fx) * cell) as f32;
                let ry = (gy0 + (j + fy) * cell) as f32;
                let p = V2::new(rx as f64 + origin.x, ry as f64 + origin.y);
                if lists[c].iter().all(|&k| part.edges[k as usize].curve.closest_within(p, 1e-3).is_none_or(|cl| cl.dist > 1e-3)) {
                    return (rx, ry, nudged);
                }
                nudged = true;
            }
            panic!("cell {c}: no reference point clear of its curves");
        })
        .collect();
    lists.par_iter_mut().enumerate().for_each(|(c, l)| {
        let p = V2::new(refs[c].0 as f64 + origin.x, refs[c].1 as f64 + origin.y);
        let mut k = 0;
        while k < l.len() {
            let f = part.edges[l[k] as usize].face;
            let inside = part.inside(f as usize, p);
            while k < l.len() && part.edges[(l[k] & 0x7fff_ffff) as usize].face == f {
                if inside {
                    l[k] |= 1 << 31;
                }
                k += 1;
            }
        }
    });

    // Faces covering whole cells: no own curve listed there, and the cell center inside.
    let covers: Vec<(u32, u32)> = (0..part.faces.len())
        .into_par_iter()
        .flat_map_iter(|f| {
            let (mut a, mut b) = (V2::new(f64::MAX, f64::MAX), V2::new(f64::MIN, f64::MIN));
            for (k, _) in part.face_edges(f) {
                let (p, q) = boxes[k];
                a = V2::new(a.x.min(p.x), a.y.min(p.y));
                b = V2::new(b.x.max(q.x), b.y.max(q.y));
            }
            let (i0, j0) = cell_of(a.x, a.y);
            let (i1, j1) = cell_of(b.x, b.y);
            let mut out = Vec::new();
            for j in clampi(j0, ny)..=clampi(j1, ny) {
                for i in clampi(i0, nx)..=clampi(i1, nx) {
                    let c = (j * nx + i) as usize;
                    if lists[c].iter().any(|&k| part.edges[(k & 0x7fff_ffff) as usize].face == f as u32) {
                        continue;
                    }
                    let center = V2::new(origin.x + gx0 + (i as f64 + 0.5) * cell, origin.y + gy0 + (j as f64 + 0.5) * cell);
                    if part.inside(f, center) {
                        out.push((c as u32, f as u32));
                    }
                }
            }
            out
        })
        .collect();
    let mut cover_lists: Vec<Vec<u32>> = vec![Vec::new(); ncell];
    for (c, f) in &covers {
        cover_lists[*c as usize].push(*f);
    }

    let mut idx = Vec::new();
    let flatten = |idx: &mut Vec<u32>, lists: &[Vec<u32>]| -> (u32, u32) {
        let start_off = idx.len() as u32;
        let mut acc = 0u32;
        for l in lists {
            idx.push(acc);
            acc += l.len() as u32;
        }
        idx.push(acc);
        let items_off = idx.len() as u32;
        for l in lists {
            idx.extend_from_slice(l);
        }
        (start_off, items_off)
    };
    let (cell_start, cell_items) = flatten(&mut idx, &lists);
    let (cover_start, cover_items) = flatten(&mut idx, &cover_lists);
    let cell_ref = idx.len() as u32;
    for r in &refs {
        idx.push(r.0.to_bits());
        idx.push(r.1.to_bits());
    }

    let o = origin;
    let mut bez: Vec<[f32; 36]> = Vec::new();
    let mut edges = Vec::with_capacity(part.edges.len());
    for (k, e) in part.edges.iter().enumerate() {
        let (a, b) = (e.curve.start() - o, e.curve.end() - o);
        let mut rec = [0f32; 16];
        rec[..4].copy_from_slice(&[a.x as f32, a.y as f32, b.x as f32, b.y as f32]);
        let kind = match &e.curve {
            Curve::Line { .. } => 0u32,
            Curve::Arc { c, r, sweep, .. } => {
                let c = *c - o;
                rec[4..8].copy_from_slice(&[c.x as f32, c.y as f32, *r as f32, *sweep as f32]);
                1
            }
            Curve::Spline(s) => {
                rec[4] = f32::from_bits(bez.len() as u32);
                rec[5] = f32::from_bits(s.pieces.len() as u32);
                for (z, th) in s.pieces.iter().zip(&s.thickness) {
                    let d = z.degree();
                    if d > MAX_DEGREE {
                        return Err(format!("B-spline of degree {d} (the kernel handles up to {MAX_DEGREE})"));
                    }
                    let mut r = [0f32; 36];
                    for (i, h) in z.pw.iter().enumerate() {
                        r[4 * i] = (h[0] - o.x * h[2]) as f32;
                        r[4 * i + 1] = (h[1] - o.y * h[2]) as f32;
                        r[4 * i + 2] = h[2] as f32;
                    }
                    r[32] = f32::from_bits(d as u32);
                    r[33] = *th as f32;
                    r[34] = z.t0 as f32;
                    r[35] = z.t1 as f32;
                    bez.push(r);
                }
                2
            }
        };
        rec[8..12].copy_from_slice(&[e.pn_start.x as f32, e.pn_start.y as f32, e.pn_end.x as f32, e.pn_end.y as f32]);
        rec[12] = f32::from_bits(kind);
        rec[13] = f32::from_bits(e.face);
        rec[14] = f32::from_bits(k as u32);
        edges.push(rec);
    }
    Ok(Packed {
        origin,
        cell,
        gx0,
        gy0,
        nx,
        ny,
        band,
        edges,
        bez,
        face_z: part.faces.iter().map(|f| f.z as f32).collect(),
        item_count: lists.iter().map(Vec::len).sum(),
        cover_count: covers.len(),
        nudged: refs.iter().filter(|r| r.2).count(),
        idx,
        cell_start,
        cell_items,
        cover_start,
        cover_items,
        cell_ref,
    })
}

/// 8 f32 per profile element: (r0, h0, r1, h1) or (rc, hc, rad, 0), then (kind, 0, 0, 0).
pub fn pack_tool(tool: &Tool) -> Vec<[f32; 8]> {
    tool.els
        .iter()
        .map(|e| match *e {
            ProfEl::Seg { r0, h0, r1, h1 } => [r0 as f32, h0 as f32, r1 as f32, h1 as f32, 0.0, 0.0, 0.0, 0.0],
            ProfEl::Arc { rc, hc, rad } => [rc as f32, hc as f32, rad as f32, 0.0, 1.0, 0.0, 0.0, 0.0],
        })
        .collect()
}
