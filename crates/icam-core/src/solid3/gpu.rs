//! The exact evaluator packed for a GPU kernel (gpu.wgsl, run by icam-sim's partgpu.rs):
//! the same features read by the same rules, in single precision relative to an origin
//! near the part, with every answer either exact (to that precision) or flagged.
//!
//! Analytic faces, line and arc edges and vertices are read in closed form, as on the
//! CPU. A B-spline face is split, span by span, into sub-patches, each with a radius of
//! convexity rho: within rho of every point of the sub-patch, the squared distance
//! |S(s, t) - p|^2 is strictly convex in the parameters, since its Hessian
//! S_i.S_j + (S - p).S_ij has its least eigenvalue at least lambda_min(G) - |S - p| |S_ij|
//! (G the first fundamental form), bounded over the sub-patch by interval arithmetic on
//! the Bernstein hulls of the derivatives. A point p whose distance to every point of the
//! sub-patch is below rho (its box's farthest corner, on the GPU) then has one minimum of
//! distance on it, which Newton finds: a critical point of the face when it is not held
//! at the sub-patch's border, and none in the sub-patch when it is. Spline and ellipse
//! edges and the knot lines of B-spline faces are rational Bezier pieces read the same way
//! in one parameter. Each span's or piece's sub-patches form a tree (a split's parts
//! stored together), which the kernel walks for each node nearest part first: a part no
//! nearer than the node's best answer is passed over, a part the node is close enough to
//! for its radius is read by Newton, any other is opened while it comes within the cache's
//! clamp of the node. A part left (too far for its radius, past the clamp, or a leaf)
//! leaves the node undecided when it could still hold the nearest point: the kernel
//! reports the least lower bound it has, and the caller reads the node on the CPU or, when
//! that bound is past the clamp, takes its sign from a neighbor (no surface lies within
//! the bound of it).

use super::{box_dist, Exact, Surf, V3};
use crate::curve3::{Bez3, Edge3, Frame};
use crate::surface3::{bez_bounds, DerivBounds, Patch};
use crate::trim2::Packed;
use rayon::prelude::*;

/// The kernel.
pub const WGSL: &str = include_str!("gpu.wgsl");

/// Words per record (gpu.wgsl).
const FACE: usize = 48;
const EDGE: usize = 16;
const VERT: usize = 8;
const SUB: usize = 48;
const CURVE: usize = 40;
/// Highest degree the kernel evaluates.
const MAX_DEGREE: usize = 7;
/// Deepest split of a span or piece.
const MAX_DEPTH: u32 = 8;

/// Kinds of the kernel's candidates (a code's top four bits).
const K_FACE: u32 = 0;
const K_EDGE: u32 = 1;
const K_POINT: u32 = 2;
const K_SUB: u32 = 3;
const K_CURVE: u32 = 4;

/// What the kernel reads for one of the evaluator's tree ids.
#[derive(Clone)]
enum Code {
    /// Nothing: a brick with it is read on the CPU.
    Cpu,
    One(u32),
    /// Whole spans or pieces, roots[a..b].
    Nodes(u32, u32),
}

/// A whole span or piece: its box and its kernel code.
struct Root {
    lo: V3,
    hi: V3,
    code: u32,
}

/// The evaluator packed: f32 words and u32 words, sections at `at`, and how to list a
/// brick's candidates for the kernel.
pub struct GpuPack {
    pub geo: Vec<f32>,
    pub ints: Vec<u32>,
    /// Where each section starts: in `geo` faces, edges, vertices, sub-patches, patches,
    /// pieces, piece control points, loop copies, loop pieces; in `ints` loop cells and
    /// lists.
    pub at: [u32; 11],
    /// Sub-patches and pieces packed.
    pub nodes: usize,
    code: Vec<Code>,
    roots: Vec<Root>,
}

/// Derivative bounds as f32 words, rounded outward: lambda (rounded down), then each
/// bound's components' [lo, hi], `n` bounds.
fn bound_words(b: &DerivBounds, n: usize) -> Vec<f32> {
    let down = |x: f64| (x - x.abs() * 1e-6 - 1e-30) as f32;
    let up = |x: f64| (x + x.abs() * 1e-6 + 1e-30) as f32;
    let mut w = vec![down(b.lambda)];
    for d in &b.d[..n] {
        for i in d {
            w.push(down(i[0]));
            w.push(up(i[1]));
        }
    }
    w
}

fn bbox(pts: impl Iterator<Item = V3>) -> (V3, V3) {
    pts.fold((V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN)), |(lo, hi), q| {
        (V3::new(lo.x.min(q.x), lo.y.min(q.y), lo.z.min(q.z)), V3::new(hi.x.max(q.x), hi.y.max(q.y), hi.z.max(q.z)))
    })
}

/// Whether a part of radius rho and box diagonal `diag` is split: while rho is short of
/// its size and what the nodes near it need, and it is not yet small beside that need
/// (past which splitting only tightens the bounds of a surface curved tighter than the
/// need).
fn split(rho: f64, diag: f64, need: f64, depth: u32) -> bool {
    depth < MAX_DEPTH && rho < 2.0 * diag + need && diag > 0.25 * need
}

fn rel(p: V3, o: V3) -> [f32; 3] {
    [(p.x - o.x) as f32, (p.y - o.y) as f32, (p.z - o.z) as f32]
}

fn dir(v: V3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// A frame as nine words: origin (relative to `o`), x, z.
fn frame(f: &Frame, o: V3) -> [f32; 9] {
    let (a, x, z) = (rel(f.o, o), dir(f.x), dir(f.z));
    [a[0], a[1], a[2], x[0], x[1], x[2], z[0], z[1], z[2]]
}

/// Homogeneous points relative to `o`, as f32.
fn hom(h: &[f64; 4], o: V3) -> [f32; 4] {
    [(h[0] - o.x * h[3]) as f32, (h[1] - o.y * h[3]) as f32, (h[2] - o.z * h[3]) as f32, h[3] as f32]
}

/// A Bezier curve's homogeneous points raised one degree (the same curve).
fn raise(pts: &[[f64; 4]]) -> Vec<[f64; 4]> {
    let n = pts.len();
    (0..=n)
        .map(|i| {
            let a = i as f64 / n as f64;
            let prev = if i > 0 { pts[i - 1] } else { [0.0; 4] };
            let cur = if i < n { pts[i] } else { [0.0; 4] };
            std::array::from_fn(|k| a * prev[k] + (1.0 - a) * cur[k])
        })
        .collect()
}

/// A patch record: degrees, parameter ranges, homogeneous points (i along u). A patch of
/// degree up to 3 each way is raised to bicubic, which the kernel evaluates in straight-line
/// code.
fn patch_words(pa: &Patch, o: V3) -> Vec<f32> {
    let mut pw = pa.pw.clone();
    if pw.len() <= 4 && pw[0].len() <= 4 {
        while pw[0].len() < 4 {
            pw = pw.iter().map(|r| raise(r)).collect();
        }
        while pw.len() < 4 {
            let cols: Vec<Vec<[f64; 4]>> = (0..pw[0].len()).map(|j| raise(&pw.iter().map(|r| r[j]).collect::<Vec<_>>())).collect();
            pw = (0..cols[0].len()).map(|i| cols.iter().map(|c| c[i]).collect()).collect();
        }
    }
    let (du, dv) = (pw.len() - 1, pw[0].len() - 1);
    let mut w = vec![f32::from_bits(du as u32), f32::from_bits(dv as u32), pa.u0 as f32, pa.u1 as f32, pa.v0 as f32, pa.v1 as f32];
    for row in &pw {
        for h in row {
            w.extend(hom(h, o));
        }
    }
    if du == 3 && dv == 3 {
        w.extend(hodographs(&pw));
    }
    w
}

/// A bicubic's derivative nets (hodographs), from its points in double precision, as the
/// kernel's `make_part3` reads them after the points (HODO): Su 3 x 4, Sv 4 x 3, Suu 2 x 4,
/// Suv 3 x 3, Svv 4 x 2 (i along u, three words a point), then each net's largest
/// coordinate. A part's derivatives are bounded by these nets cut to the part, so their
/// rounding stays relative to the derivatives: differencing the part's own single-precision
/// points, which lie hundreds of millimeters from the packing's origin, left mostly rounding
/// on parts a few tens of micrometers wide. A rational patch has no such nets (its
/// derivatives are not these differences): its largest coordinates are written -1, and the
/// kernel makes no part of it.
fn hodographs(pw: &[Vec<[f64; 4]>]) -> Vec<f32> {
    let p = |i: usize, j: usize| {
        let h = pw[i][j];
        V3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
    };
    let mut nets: [Vec<V3>; 5] = Default::default();
    for i in 0..3 {
        for j in 0..4 {
            nets[0].push((p(i + 1, j) - p(i, j)) * 3.0);
        }
    }
    for i in 0..4 {
        for j in 0..3 {
            nets[1].push((p(i, j + 1) - p(i, j)) * 3.0);
        }
    }
    for i in 0..2 {
        for j in 0..4 {
            nets[2].push((p(i + 2, j) - p(i + 1, j) * 2.0 + p(i, j)) * 6.0);
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            nets[3].push((p(i + 1, j + 1) - p(i + 1, j) - p(i, j + 1) + p(i, j)) * 9.0);
        }
    }
    for i in 0..4 {
        for j in 0..2 {
            nets[4].push((p(i, j + 2) - p(i, j + 1) * 2.0 + p(i, j)) * 6.0);
        }
    }
    let polynomial = pw.iter().flatten().all(|h| h[3] == pw[0][0][3]);
    let mut w: Vec<f32> = nets.iter().flatten().flat_map(|v| [v.x as f32, v.y as f32, v.z as f32]).collect();
    w.extend(nets.iter().map(|n| if polynomial { largest(n) } else { -1.0 }));
    w
}

/// The largest coordinate of a net's points.
fn largest(n: &[V3]) -> f32 {
    n.iter().map(|v| v.x.abs().max(v.y.abs()).max(v.z.abs())).fold(0.0f64, f64::max) as f32
}

/// A cubic piece's derivative nets, as `hodographs` gives a patch's: C' 3 points, C'' 2,
/// then each net's largest coordinate (-1 for a rational piece), read after its points by
/// the kernel's `make_piece`.
fn piece_hodographs(pw: &[[f64; 4]]) -> Vec<f32> {
    let p = |i: usize| V3::new(pw[i][0] / pw[i][3], pw[i][1] / pw[i][3], pw[i][2] / pw[i][3]);
    let d1: Vec<V3> = (0..3).map(|i| (p(i + 1) - p(i)) * 3.0).collect();
    let d2: Vec<V3> = (0..2).map(|i| (p(i + 2) - p(i + 1) * 2.0 + p(i)) * 6.0).collect();
    let polynomial = pw.iter().all(|h| h[3] == pw[0][3]);
    let mut w: Vec<f32> = d1.iter().chain(&d2).flat_map(|v| [v.x as f32, v.y as f32, v.z as f32]).collect();
    w.extend([&d1, &d2].map(|n| if polynomial { largest(n) } else { -1.0 }));
    w
}

fn degree_ok(pa: &Patch) -> bool {
    pa.pw.len() - 1 <= MAX_DEGREE && pa.pw[0].len() - 1 <= MAX_DEGREE
}

/// A B-spline face's sub-patches, built for one face: records and tree nodes with
/// face-local indices (sub records from 0, patch words from 0, nodes from 0).
struct FaceTree {
    subs: Vec<[f32; SUB]>,
    patches: Vec<f32>,
    /// Each sub record's patch offset within `patches`, to be made absolute.
    sub_patch: Vec<u32>,
    /// The spans: box and sub record.
    roots: Vec<(V3, V3, u32)>,
    /// Each span's sub record (span (a, b) at a * spans along v + b), None outside the face.
    span_root: Vec<Option<u32>>,
}

impl<'a> Exact<'a> {
    /// The evaluator packed for the GPU kernel, relative to `origin` (a point near the
    /// part). `need`: how far from a B-spline face nodes should be read on the GPU (a few
    /// of the finest cell diagonals); its spans are split until their sub-patches are
    /// convex that far out, or no split helps.
    pub fn gpu_pack(&self, origin: V3, need: f64) -> GpuPack {
        let s = self.s;
        let o = origin;
        let starts = self.id_starts();
        // ICAM_TRACE_PACK: each step's time.
        let trace = std::env::var_os("ICAM_TRACE_PACK").is_some();
        let clock = std::time::Instant::now();
        let lap = |what: &str| {
            if trace {
                eprintln!("  pack: {what} by {:.2} s", clock.elapsed().as_secs_f64());
            }
        };
        let mut code = vec![Code::Cpu; starts[4] + self.poles.len()];
        let mut faces = vec![0f32; FACE * s.faces.len().max(1)];
        let mut loops = Packed::default();
        // Which faces the kernel reads, and how (0..4 analytic, 5 B-spline).
        let mut kind = vec![7u32; s.faces.len()];
        let trees: Vec<Option<FaceTree>> = s
            .faces
            .par_iter()
            .enumerate()
            .map(|(fi, f)| {
                let (Surf::Nurbs(ns), Some(t)) = (&f.surf, &f.trim2) else { return None };
                let (nu, nv) = ns.spans_uv();
                let mut ft = FaceTree { subs: Vec::new(), patches: Vec::new(), sub_patch: Vec::new(), roots: Vec::new(), span_root: Vec::new() };
                for a in 0..nu {
                    for b in 0..nv {
                        let pa = ns.patch(a, b);
                        if !degree_ok(&pa) {
                            return None;
                        }
                        let pb = ft.patches.len() as u32;
                        ft.patches.extend(patch_words(&pa, o));
                        // A sub-patch's record (its parts filled in when it splits), with its
                        // radius and box diagonal; None when it lies outside the face.
                        let make = |sp: &Patch, r: [f64; 4]| {
                            let class = t.box_class([sp.u0, sp.v0], [sp.u1, sp.v1]);
                            if class == Some(false) {
                                return None;
                            }
                            let (lo, hi) = sp.bbox();
                            let bounds = sp.derivative_bounds();
                            let rho = bounds.map_or(0.0, |b| b.patch_radius());
                            let mut rec = [0f32; SUB];
                            rec[..3].copy_from_slice(&rel(lo, o));
                            rec[3..6].copy_from_slice(&rel(hi, o));
                            rec[6] = rho.min(f32::MAX as f64) as f32;
                            rec[8] = r[0] as f32;
                            rec[9] = r[1] as f32;
                            rec[10] = r[2] as f32;
                            rec[11] = r[3] as f32;
                            rec[12] = f32::from_bits(fi as u32);
                            rec[13] = f32::from_bits((class == Some(true)) as u32);
                            // Words 16 on: lambda_min(G) and the derivatives' bounds (zero,
                            // so never used, when the weights give none).
                            if let Some(b) = bounds {
                                let w = bound_words(&b, 5);
                                rec[16..16 + w.len()].copy_from_slice(&w);
                            }
                            Some((rec, rho, (lo, hi)))
                        };
                        let Some((rec, rho, b)) = make(&pa, [0.0, 1.0, 0.0, 1.0]) else {
                            ft.span_root.push(None);
                            continue;
                        };
                        let id = ft.subs.len() as u32;
                        ft.span_root.push(Some(id));
                        ft.subs.push(rec);
                        ft.sub_patch.push(pb);
                        ft.roots.push((b.0, b.1, id));
                        // Breadth first, so a split's parts lie together.
                        let mut queue = std::collections::VecDeque::from([(id, pa, [0.0, 1.0, 0.0, 1.0], 0u32, rho, (b.1 - b.0).len())]);
                        while let Some((id, sp, r, depth, rho, diag)) = queue.pop_front() {
                            if !split(rho, diag, need, depth) {
                                continue;
                            }
                            let (sm, tm) = (0.5 * (r[0] + r[1]), 0.5 * (r[2] + r[3]));
                            let [a0, a1, b0, b1] = sp.quarters();
                            let first = ft.subs.len() as u32;
                            for (q, rq) in [(a0, [r[0], sm, r[2], tm]), (a1, [r[0], sm, tm, r[3]]), (b0, [sm, r[1], r[2], tm]), (b1, [sm, r[1], tm, r[3]])] {
                                if let Some((rec, rho, b)) = make(&q, rq) {
                                    let k = ft.subs.len() as u32;
                                    ft.subs.push(rec);
                                    ft.sub_patch.push(pb);
                                    queue.push_back((k, q, rq, depth + 1, rho, (b.1 - b.0).len()));
                                }
                            }
                            ft.subs[id as usize][14] = f32::from_bits(first);
                            ft.subs[id as usize][15] = f32::from_bits(ft.subs.len() as u32 - first);
                        }
                    }
                }
                Some(ft)
            })
            .collect();
        lap("B-spline faces' trees");
        let mut subs: Vec<[f32; SUB]> = Vec::new();
        let mut patches: Vec<f32> = Vec::new();
        let mut roots: Vec<Root> = Vec::new();
        // Sub records' patch offsets, made absolute once the sections are placed.
        let mut sub_patch: Vec<u32> = Vec::new();
        // Where each B-spline face's sub records start.
        let mut face_s0 = vec![0u32; s.faces.len()];
        for (fi, f) in s.faces.iter().enumerate() {
            let w = &mut faces[FACE * fi..FACE * (fi + 1)];
            w[0] = f32::from_bits(7);
            w[1] = f.sense as f32;
            let Some(t) = &f.trim2 else { continue };
            if let Some(ft) = &trees[fi] {
                let (s0, p0, r0) = (subs.len() as u32, patches.len() as u32, roots.len() as u32);
                face_s0[fi] = s0;
                subs.extend(ft.subs.iter().map(|rec| {
                    let mut rec = *rec;
                    if rec[15].to_bits() > 0 {
                        rec[14] = f32::from_bits(s0 + rec[14].to_bits());
                    }
                    rec
                }));
                sub_patch.extend(ft.sub_patch.iter().map(|&p| p0 + p));
                patches.extend(ft.patches.iter().cloned());
                roots.extend(ft.roots.iter().map(|&(lo, hi, id)| Root { lo, hi, code: (K_SUB << 28) | (s0 + id) }));
                code[starts[0] + fi] = Code::Nodes(r0, roots.len() as u32);
                kind[fi] = 5;
                // Words 2 to 5: the spans each way, where the face's patches start in their
                // section and each one's size (one a span, u major), so the kernel can walk
                // from span to span.
                if let Surf::Nurbs(ns) = &f.surf {
                    let (nu, nv) = ns.spans_uv();
                    w[2] = f32::from_bits(nu as u32);
                    w[3] = f32::from_bits(nv as u32);
                    w[4] = f32::from_bits(p0);
                    w[5] = f32::from_bits((ft.patches.len() / (nu * nv).max(1)) as u32);
                }
            } else if let Some(a) = &f.analytic2 {
                let tips = &self.tips[fi];
                if tips.len() > 2 {
                    continue;
                }
                let (k, fr, r, r2) = match &f.surf {
                    Surf::Plane { f } => (0, f, 0.0, 0.0),
                    Surf::Cylinder { f, r } => (1, f, *r, 0.0),
                    Surf::Cone { f, r, a } => (2, f, *r, *a),
                    Surf::Sphere { f, r } => (3, f, *r, 0.0),
                    Surf::Torus { f, major, minor } => (4, f, *major, *minor),
                    Surf::Nurbs(_) => continue,
                };
                w[2..11].copy_from_slice(&frame(fr, o));
                w[11] = r as f32;
                w[12] = r2 as f32;
                let (uk, uf, up) = a.pack();
                w[13] = f32::from_bits(uk);
                w[14..23].copy_from_slice(&frame(&uf, o));
                for i in 0..4 {
                    w[23 + i] = up[i] as f32;
                }
                w[35] = f32::from_bits(tips.len() as u32);
                for (i, (x, n)) in tips.iter().enumerate() {
                    let (x, n) = (rel(*x, o), dir(*n));
                    w[36 + 6 * i..42 + 6 * i].copy_from_slice(&[x[0], x[1], x[2], n[0], n[1], n[2]]);
                }
                kind[fi] = k;
                code[starts[0] + fi] = Code::One((K_FACE << 28) | fi as u32);
            } else {
                continue;
            }
            w[0] = f32::from_bits(kind[fi]);
            let head = t.pack_into(&mut loops);
            for i in 0..8 {
                w[27 + i] = f32::from_bits(head[i]);
            }
        }
        lap("faces placed");
        // Edges: lines and arcs in closed form, other curves as pieces; either way signed
        // by their faces' normals, so every face of theirs must be read too.
        let mut edges = vec![0f32; EDGE * s.edges.len().max(1)];
        let mut curves: Vec<[f32; CURVE]> = Vec::new();
        let mut cpts: Vec<f32> = Vec::new();
        // Piece records' control point offsets and knot-line patch offsets, made absolute
        // later: (curve record, word, offset within its section).
        let mut curve_fix: Vec<(usize, usize, u32, bool)> = Vec::new();
        let piece_tree = |pw: &[[f64; 4]], rec: [f32; CURVE], roots: &mut Vec<Root>, curves: &mut Vec<[f32; CURVE]>, cpts: &mut Vec<f32>, fix: &mut Vec<(usize, usize, u32, bool)>| {
            let cp = cpts.len() as u32;
            // Raised to a cubic when below (the kernel's straight-line case).
            let mut stored = pw.to_vec();
            while stored.len() < 4 {
                stored = raise(&stored);
            }
            for h in &stored {
                cpts.extend(hom(h, o));
            }
            if stored.len() == 4 {
                cpts.extend(piece_hodographs(&stored));
            }
            let mut make = |zz: &Bez3, r: [f64; 2], curves: &mut Vec<[f32; CURVE]>| {
                let (lo, hi) = bbox(zz.ctrl());
                let bounds = bez_bounds(&zz.pw);
                let rho = bounds.map_or(0.0, |b| b.piece_radius());
                let mut c = rec;
                // Words 21 on: |C'|^2's lower bound and C', C'' bounds.
                if let Some(b) = bounds {
                    let w = bound_words(&b, 2);
                    c[21..21 + w.len()].copy_from_slice(&w);
                }
                c[..3].copy_from_slice(&rel(lo, o));
                c[3..6].copy_from_slice(&rel(hi, o));
                c[6] = rho.min(f32::MAX as f64) as f32;
                c[7] = f32::from_bits((stored.len() - 1) as u32);
                c[9] = r[0] as f32;
                c[10] = r[1] as f32;
                let id = curves.len();
                curves.push(c);
                fix.push((id, 8, cp, false));
                (id, rho, (lo, hi))
            };
            let z = Bez3 { pw: pw.to_vec(), t0: 0.0, t1: 1.0 };
            let (id, rho, b) = make(&z, [0.0, 1.0], curves);
            roots.push(Root { lo: b.0, hi: b.1, code: (K_CURVE << 28) | id as u32 });
            // Breadth first, so a split's halves lie together.
            let mut queue = std::collections::VecDeque::from([(id, z, [0.0f64, 1.0], 0u32, rho, (b.1 - b.0).len())]);
            while let Some((id, zz, r, depth, rho, diag)) = queue.pop_front() {
                if !split(rho, diag, need, depth) {
                    continue;
                }
                let m = 0.5 * (r[0] + r[1]);
                let (a, bb) = zz.split(0.5);
                let first = curves.len();
                for (q, rq) in [(a, [r[0], m]), (bb, [m, r[1]])] {
                    let (k, rho, b) = make(&q, rq, curves);
                    queue.push_back((k, q, rq, depth + 1, rho, (b.1 - b.0).len()));
                }
                curves[id][19] = f32::from_bits(first as u32);
                curves[id][20] = f32::from_bits(2);
            }
        };
        // Each edge's words and pieces on its own (in parallel: each piece record's seed is a
        // search of its faces), with record, point and root numbers its own; then laid out
        // in edge order, those numbers made the packing's.
        struct EdgeOut {
            w: Vec<f32>,
            one: bool,
            pieces: Option<(Vec<Root>, Vec<[f32; CURVE]>, Vec<f32>, Vec<(usize, usize, u32, bool)>)>,
        }
        let outs: Vec<Option<EdgeOut>> = s
            .edges
            .par_iter()
            .enumerate()
            .map(|(ei, e)| {
                let mut w = vec![0f32; EDGE];
                if !(1..=2).contains(&e.faces.len()) || !e.faces.iter().all(|&fi| kind[fi] <= 5) {
                    return None;
                }
                w[13] = f32::from_bits(e.faces[0] as u32);
                w[14] = f32::from_bits(e.faces[e.faces.len() - 1] as u32);
                w[15] = f32::from_bits(e.faces.len() as u32);
                match &e.curve {
                    Edge3::Line { a, b } => {
                        w[0] = f32::from_bits(0);
                        w[1..4].copy_from_slice(&rel(*a, o));
                        w[4..7].copy_from_slice(&rel(*b, o));
                        Some(EdgeOut { w, one: true, pieces: None })
                    }
                    Edge3::Arc { f, r, t0, sweep } => {
                        w[0] = f32::from_bits(1);
                        w[1..10].copy_from_slice(&frame(f, o));
                        w[10] = *r as f32;
                        w[11] = *t0 as f32;
                        w[12] = *sweep as f32;
                        Some(EdgeOut { w, one: true, pieces: None })
                    }
                    curve => {
                        let pieces = curve.bez_pieces();
                        if pieces.iter().any(|z| z.pw.len() - 1 > MAX_DEGREE || z.pw.iter().any(|h| !(h[3] > 0.0))) {
                            // Its words still written: the kernel never reads them.
                            return Some(EdgeOut { w, one: false, pieces: None });
                        }
                        w[0] = f32::from_bits(2);
                        let (mut roots, mut curves, mut cpts, mut fix) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                        let mut rec = [0f32; CURVE];
                        rec[11] = f32::from_bits(0);
                        rec[12] = f32::from_bits(ei as u32);
                        for z in &pieces {
                            let first = curves.len();
                            piece_tree(&z.pw, rec, &mut roots, &mut curves, &mut cpts, &mut fix);
                            // Each record's seed on each of the edge's faces (words 34 on: the
                            // span's sub record, then s and t in it; the edge's first face, then its
                            // last): the face's nearest point to the record's middle, from which
                            // the kernel finds the face's normal at a point of the record.
                            let whole = Bez3 { pw: z.pw.clone(), t0: 0.0, t1: 1.0 };
                            for (k, &fi) in [e.faces[0], e.faces[e.faces.len() - 1]].iter().enumerate() {
                                let surf = match (&s.faces[fi].surf, &trees[fi]) {
                                    (Surf::Nurbs(ns), Some(ft)) => Some((ns, ft)),
                                    _ => None,
                                };
                                // A record's parts start from its seed.
                                let mut from: Vec<Option<(f64, f64)>> = vec![None; curves.len() - first];
                                for id in first..curves.len() {
                                    let c = &mut curves[id];
                                    c[34 + 3 * k] = f32::from_bits(u32::MAX);
                                    let Some((ns, ft)) = surf else { continue };
                                    let q = whole.eval(0.5 * (c[9] as f64 + c[10] as f64)).p;
                                    let Some(h) = from[id - first].and_then(|(u, v)| ns.nearest_from(q, u, v)).or_else(|| ns.closest(q, 1.0)) else { continue };
                                    let (kf, kn) = (c[19].to_bits() as usize, c[20].to_bits() as usize);
                                    for kid in kf..kf + kn {
                                        from[kid - first] = Some((h.u, h.v));
                                    }
                                    let (a, b) = ns.span_at(h.u, h.v);
                                    let Some(rid) = ft.span_root[a * ns.spans_v() + b] else { continue };
                                    let pa = ns.patch(a, b);
                                    c[34 + 3 * k] = f32::from_bits(face_s0[fi] + rid);
                                    c[35 + 3 * k] = ((h.u - pa.u0) / (pa.u1 - pa.u0)) as f32;
                                    c[36 + 3 * k] = ((h.v - pa.v0) / (pa.v1 - pa.v0)) as f32;
                                }
                            }
                        }
                        Some(EdgeOut { w, one: false, pieces: Some((roots, curves, cpts, fix)) })
                    }
                }
            })
            .collect();
        for (ei, out) in outs.into_iter().enumerate() {
            let Some(out) = out else { continue };
            edges[EDGE * ei..EDGE * (ei + 1)].copy_from_slice(&out.w);
            if out.one {
                code[starts[1] + ei] = Code::One((K_EDGE << 28) | ei as u32);
            }
            let Some((rs, cs, ps, fx)) = out.pieces else { continue };
            let (base_c, base_p, r0) = (curves.len() as u32, cpts.len() as u32, roots.len() as u32);
            for r in rs {
                roots.push(Root { lo: r.lo, hi: r.hi, code: (K_CURVE << 28) | ((r.code & 0x0fff_ffff) + base_c) });
            }
            for mut c in cs {
                if c[20].to_bits() > 0 {
                    c[19] = f32::from_bits(c[19].to_bits() + base_c);
                }
                curves.push(c);
            }
            cpts.extend(ps);
            curve_fix.extend(fx.into_iter().map(|(id, word, cp, is_patch)| (id + base_c as usize, word, cp + base_p, is_patch)));
            code[starts[1] + ei] = Code::Nodes(r0, roots.len() as u32);
        }
        lap("edges");
        let mut verts = vec![0f32; VERT * (s.verts.len() + self.poles.len()).max(1)];
        for (vi, v) in s.verts.iter().enumerate() {
            let (p, n) = (rel(v.p, o), dir(v.pn));
            verts[VERT * vi..VERT * vi + 6].copy_from_slice(&[p[0], p[1], p[2], n[0], n[1], n[2]]);
            code[starts[2] + vi] = Code::One((K_POINT << 28) | vi as u32);
        }
        for (pi, (_, x, n)) in self.poles.iter().enumerate() {
            let k = s.verts.len() + pi;
            let (p, n) = (rel(*x, o), dir(*n));
            verts[VERT * k..VERT * k + 6].copy_from_slice(&[p[0], p[1], p[2], n[0], n[1], n[2]]);
            code[starts[4] + pi] = Code::One((K_POINT << 28) | k as u32);
        }
        // Knot lines: pieces of the face's iso-curve, with the spans on either side.
        for (li, l) in self.lines.iter().enumerate() {
            if kind[l.face] != 5 || l.pieces.iter().any(|(z, a, b)| z.pw.len() - 1 > MAX_DEGREE || !degree_ok(a) || !degree_ok(b)) {
                continue;
            }
            let r0 = roots.len() as u32;
            for (z, before, after) in &l.pieces {
                let pb = patches.len() as u32;
                patches.extend(patch_words(before, o));
                let pa = patches.len() as u32;
                patches.extend(patch_words(after, o));
                let mut rec = [0f32; CURVE];
                rec[11] = f32::from_bits(1);
                rec[12] = f32::from_bits(l.face as u32);
                rec[13] = f32::from_bits(l.k as u32);
                rec[14] = l.value as f32;
                rec[15] = z.t0 as f32;
                rec[16] = z.t1 as f32;
                let first = curves.len();
                piece_tree(&z.pw, rec, &mut roots, &mut curves, &mut cpts, &mut curve_fix);
                for id in first..curves.len() {
                    curve_fix.push((id, 17, pb, true));
                    curve_fix.push((id, 18, pa, true));
                }
            }
            code[starts[3] + li] = Code::Nodes(r0, roots.len() as u32);
        }
        lap("vertices and knot lines");
        // Lay the sections out and make offsets absolute.
        let mut geo: Vec<f32> = Vec::new();
        let mut at = [0u32; 11];
        let place = |geo: &mut Vec<f32>, words: &[f32]| {
            let a = geo.len() as u32;
            geo.extend_from_slice(words);
            if words.is_empty() {
                geo.push(0.0);
            }
            a
        };
        at[0] = place(&mut geo, &faces);
        at[1] = place(&mut geo, &edges);
        at[2] = place(&mut geo, &verts);
        at[3] = geo.len() as u32;
        let subs_at = at[3];
        for _ in 0..subs.len().max(1) {
            geo.extend([0f32; SUB]);
        }
        at[4] = place(&mut geo, &patches);
        at[5] = geo.len() as u32;
        for _ in 0..curves.len().max(1) {
            geo.extend([0f32; CURVE]);
        }
        at[6] = place(&mut geo, &cpts);
        at[7] = place(&mut geo, &loops.inst.iter().flatten().cloned().collect::<Vec<f32>>());
        at[8] = place(&mut geo, &loops.pieces.iter().flatten().cloned().collect::<Vec<f32>>());
        for (i, mut rec) in subs.into_iter().enumerate() {
            rec[7] = f32::from_bits(at[4] + sub_patch[i]);
            let b = subs_at as usize + SUB * i;
            geo[b..b + SUB].copy_from_slice(&rec);
        }
        for (i, rec) in curves.iter().enumerate() {
            let b = at[5] as usize + CURVE * i;
            geo[b..b + CURVE].copy_from_slice(rec);
        }
        for (id, word, off, is_patch) in curve_fix {
            let base = if is_patch { at[4] } else { at[6] };
            geo[at[5] as usize + CURVE * id + word] = f32::from_bits(base + off);
        }
        let mut ints: Vec<u32> = loops.cells.iter().flatten().cloned().collect();
        at[9] = 0;
        if ints.is_empty() {
            ints.push(0);
        }
        at[10] = ints.len() as u32;
        ints.extend(&loops.among);
        ints.push(0);
        let nodes = sub_patch.len() + curves.len();
        lap("laid out");
        if std::env::var_os("ICAM_PACK_HASH").is_some() {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            geo.iter().map(|x| x.to_bits()).collect::<Vec<u32>>().hash(&mut h);
            ints.hash(&mut h);
            at.hash(&mut h);
            roots.iter().map(|r| (r.code, r.lo.x.to_bits(), r.lo.y.to_bits(), r.lo.z.to_bits(), r.hi.x.to_bits(), r.hi.y.to_bits(), r.hi.z.to_bits())).collect::<Vec<_>>().hash(&mut h);
            code.iter().map(|c| match c { Code::Cpu => (0u32, 0u32, 0u32), Code::One(x) => (1, *x, 0), Code::Nodes(a, b) => (2, *a, *b) }).collect::<Vec<_>>().hash(&mut h);
            eprintln!("  pack: hash {:016x} ({} geo words, {} ints)", h.finish(), geo.len(), ints.len());
        }
        GpuPack { geo, ints, at, nodes, code, roots }
    }
}

impl GpuPack {
    /// The kernel's candidates for the points within `r` of `c`, given that the boundary
    /// comes within `d` of c (the evaluator's `candidates`, with each B-spline face and
    /// curve opened into its spans or pieces that come within d + 2 r of c, nearest
    /// first, after the closed-form features); None when any must be read on the CPU.
    pub fn list(&self, ex: &Exact, c: V3, r: f64, d: f64) -> Option<Vec<u32>> {
        let reach = d + 2.0 * r;
        let mut one = Vec::new();
        let mut near: Vec<(f64, u32)> = Vec::new();
        for id in ex.candidates(c, r, d) {
            match self.code.get(id)? {
                Code::Cpu => return None,
                Code::One(x) => one.push(*x),
                Code::Nodes(a, b) => {
                    for n in &self.roots[*a as usize..*b as usize] {
                        let lb = box_dist(c, &(n.lo, n.hi));
                        if lb <= reach {
                            near.push((lb, n.code));
                        }
                    }
                }
            }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        one.extend(near.into_iter().map(|x| x.1));
        Some(one)
    }
}
