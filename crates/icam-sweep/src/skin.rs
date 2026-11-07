//! The stock's skin at runtime: flat three-cornered pieces whose corners the fields
//! place on the stock's zero set, updated move by move.
//!
//! Applying a move: the skin the move passes through is cut away along the move's zero
//! set; the move's own surface (envelope.rs), trimmed to the move's boundary where it
//! folds through itself, is added where it lay in material before the move. Both cuts
//! read only fields.
//!
//! Keeping the part of a piece where a field is non-negative: the fields here change no
//! faster than distance, and every point of a piece lies within 0.58 of its longest side
//! of a corner, so a piece whose corner values all clear that is wholly on one side and
//! is kept or dropped whole. Otherwise it splits across its longest side to look closer,
//! down to `finest`, where it is cut along the zero (found by bisection on each crossed
//! side); halves that both come out whole one way merge back, so only pieces the zero
//! crosses stay split.
//!
//! Surfaces that coincide (two passes at one depth share a floor, a plunge starts where
//! a ramp ended) put a piece on the other field's zero, where no value clears anything.
//! A tie rule settles them: the surface already in the skin keeps a spot both reach, so
//! the move's field is read a micrometer high on the old skin and the old stock's a
//! micrometer low on the move's surface; a small piece whose corners all sit within ten
//! micrometers of zero is then decided whole.
//!
//! The tie holds only if every corner lies on its own surface. A corner left on a flat
//! piece's chord (where a cut crosses a side, or where a split halves one) sits off the
//! curved surface by up to the chord tolerance; a later pass along the same surface reads
//! it inside itself and drops it, and drops its own copy too, the fields saying the old
//! one is there: a hole. So every new corner is settled onto the surface of the move its
//! piece came from (Newton steps on that move's field; the box's faces are flat), and a
//! cut's corner onto the cut as well, half a tie inside the side kept, where a later pass
//! along the same surface reads it clearly kept.
//!
//! A move's surface lies on its own field's zero except where it folds through itself, so
//! trimming it by its own field reads each piece's corners only (`trim`).
//!
//! Pieces are flat chords of curved surfaces: their corners lie on the stock's zero set,
//! their insides within the chord tolerance (`chord`) of it. Each piece keeps its source
//! (a box face or a move), for settling its corners and for coloring by move or time.
//!
//! Fields are read a batch at a time ([`Field`]): every piece's corners at once, then
//! every split's new point a level at a time, then every crossed side's root, so a GPU
//! (gpu.rs) reads them in a few dispatches a move. Its values are intervals; a point whose
//! interval holds zero (the GPU cannot say which side it is on) is read again on the CPU
//! in double precision, so the tie rule above decides the same way on either.

use rayon::prelude::*;

use std::sync::Arc;

use crate::envelope::envelope;
use crate::geom::{cross, dot, scale, unit, Aabb, V3};
use crate::gpu::{Fields, Of};
use crate::stock::Stock;
use crate::sweep::Sweep;

/// A flat three-cornered piece of skin: corners `p` (wound counterclockwise seen from
/// outside), the surface's outward normals there `n`, and the surface it lies on: a move
/// of the stock by index, or `BOX` for the stock box's faces.
#[derive(Clone, Copy, Debug)]
pub struct Patch {
    pub p: [V3; 3],
    pub n: [V3; 3],
    pub src: u32,
}

/// The source of the stock box's faces.
pub const BOX: u32 = u32::MAX;

impl Patch {
    /// The piece on three (point, normal) corners, wound to face the normals; None when
    /// it has no area.
    pub fn facing(a: (V3, V3), b: (V3, V3), c: (V3, V3)) -> Option<Patch> {
        let g = cross(b.0 - a.0, c.0 - a.0);
        if g.len() < 1e-14 {
            return None;
        }
        Some(if dot(g, a.1 + b.1 + c.1) >= 0.0 { Patch { p: [a.0, b.0, c.0], n: [a.1, b.1, c.1], src: BOX } } else { Patch { p: [a.0, c.0, b.0], n: [a.1, c.1, b.1], src: BOX } })
    }

    /// The piece, on surface `src`.
    pub fn on(mut self, src: u32) -> Patch {
        self.src = src;
        self
    }

    /// The piece facing the other way (normals reversed, winding reversed).
    pub fn flipped(self) -> Patch {
        Patch { p: [self.p[0], self.p[2], self.p[1]], n: [scale(self.n[0], -1.0), scale(self.n[2], -1.0), scale(self.n[1], -1.0)], src: self.src }
    }

    pub fn bounds(&self) -> Aabb {
        Aabb::new(self.p[0], self.p[0]).union(Aabb::new(self.p[1], self.p[1])).union(Aabb::new(self.p[2], self.p[2]))
    }

    fn longest(&self) -> f64 {
        (self.p[1] - self.p[0]).len().max((self.p[2] - self.p[1]).len()).max((self.p[0] - self.p[2]).len())
    }
}

fn mid(a: Corner, b: Corner) -> Corner {
    (scale(a.0 + b.0, 0.5), unit(a.1 + b.1))
}

/// A corner's normal: the surface's gradient, turned to agree with `like` (the normal the
/// corner had before it was settled), or `like` itself where there is no gradient.
fn oriented(g: Option<V3>, like: V3) -> V3 {
    match g {
        Some(g) if g.len() > 1e-12 => {
            let u = unit(g);
            if dot(u, like) < 0.0 {
                scale(u, -1.0)
            } else {
                u
            }
        }
        _ => like,
    }
}

/// A corner settled from `c` to `x` (gradient `g` there), or `c` itself when settling
/// failed: a point settles no farther than a chord's sag, a fraction of a millimeter.
fn kept(c: Corner, x: V3, g: Option<V3>) -> Corner {
    let ok = x.x.is_finite() && x.y.is_finite() && x.z.is_finite() && (x - c.0).len() <= 0.5;
    if ok {
        (x, oriented(g, c.1))
    } else {
        c
    }
}

/// The move a piece's source names; None for the box's faces (flat: a chord lies on them).
pub type Sources<'a> = dyn Fn(u32) -> Option<&'a Sweep> + Sync + 'a;

/// A field the skin cuts by (keeping its non-negative side), read a batch at a time.
pub trait Field: Sync {
    /// For each point, an interval holding the field there (clamped to any limit no
    /// smaller than the pieces it decides).
    fn values(&self, pts: &[V3]) -> Vec<[f64; 2]>;
    /// For each side (a, b), the field non-negative at a and negative at b: the fraction
    /// of the way to b, on the chord, where the cut crosses (where the field reads its
    /// `at`, see the module notes; a side the cut does not cross ends at a or b).
    fn roots(&self, sides: &[(V3, V3)]) -> Vec<f64>;
    /// Each point, with the source of the piece it is a corner of, moved onto that
    /// source's surface and, where asked, onto the cut as well (along the source's
    /// surface): the point, the source surface's gradient there (None on the box's faces),
    /// and an interval holding the field there.
    #[allow(clippy::type_complexity)]
    fn settle(&self, items: &[(V3, u32, bool)]) -> Vec<(V3, Option<V3>, [f64; 2])>;
}

/// A field read one point at a time on the CPU; its values are exact intervals, its cut
/// where it reads `at`, the sources' surfaces read to within `tol`.
pub struct Pointwise<'a> {
    pub f: &'a (dyn Fn(V3) -> f64 + Sync),
    pub at: f64,
    pub sources: &'a Sources<'a>,
    pub tol: f64,
}

/// Settling steps no longer than a chord's sag can be (onto a source surface), or than a
/// fraction of a chord (along it, toward the cut): a longer step along the surface means
/// the cut runs nearly along it there, where the chord's point is already good (as
/// gpu.wgsl, `capped`).
fn capped(d: V3, reach: f64) -> V3 {
    scale(d, (reach / d.len().max(1e-300)).min(1.0))
}

impl Pointwise<'_> {
    /// Two Newton steps onto move `s`'s surface.
    fn onto(&self, s: &Sweep, mut y: V3) -> V3 {
        for _ in 0..2 {
            let (v, g) = s.value_grad(y, self.tol);
            if g.len_sq() > 1e-24 {
                y = y - capped(scale(g, v / g.len_sq()), 0.25);
            }
        }
        y
    }
}

impl Field for Pointwise<'_> {
    fn values(&self, pts: &[V3]) -> Vec<[f64; 2]> {
        pts.par_iter().map(|&p| (self.f)(p)).map(|v| [v, v]).collect()
    }

    /// By bisection on the side.
    fn roots(&self, sides: &[(V3, V3)]) -> Vec<f64> {
        sides
            .par_iter()
            .map(|&(a, b)| {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..30 {
                    let m = 0.5 * (lo + hi);
                    if (self.f)(a + scale(b - a, m)) >= self.at {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                0.5 * (lo + hi)
            })
            .collect()
    }

    /// Newton steps onto the source's surface; onto the cut, Newton steps on the field
    /// within the source's tangent plane (its gradient by differences far wider than the
    /// field's tolerance), each followed by steps back onto the source.
    fn settle(&self, items: &[(V3, u32, bool)]) -> Vec<(V3, Option<V3>, [f64; 2])> {
        items
            .par_iter()
            .map(|&(x, src, cut)| {
                let Some(s) = (self.sources)(src) else {
                    let v = (self.f)(x);
                    return (x, None, [v, v]);
                };
                let mut x = self.onto(s, x);
                if cut {
                    let h = (100.0 * self.tol).max(1e-6);
                    for _ in 0..3 {
                        let n = s.value_grad(x, self.tol).1;
                        let d = |e: V3| (self.f)(x + e) - (self.f)(x - e);
                        let g = scale(V3::new(d(V3::new(h, 0.0, 0.0)), d(V3::new(0.0, h, 0.0)), d(V3::new(0.0, 0.0, h))), 0.5 / h);
                        let t = g - scale(n, dot(g, n) / n.len_sq().max(1e-24));
                        if t.len_sq() > 1e-24 {
                            x = x - capped(scale(t, ((self.f)(x) - self.at) / t.len_sq()), 0.1);
                        }
                        x = self.onto(s, x);
                    }
                }
                let v = (self.f)(x);
                (x, Some(s.value_grad(x, self.tol).1), [v, v])
            })
            .collect()
    }
}

/// A field read on the GPU, sgn (one move's or the stock's field) + off, clamped to
/// [-limit, limit] before the shift, its cut where it reads `at`; points whose interval
/// holds zero, and anything the kernels cannot place, are read on the CPU from `cpu` (the
/// same field in double precision).
pub struct OnGpu<'a> {
    pub gpu: &'a Fields,
    pub of: Of<'a>,
    pub limit: f64,
    pub tol: f64,
    pub sgn: f64,
    pub off: f64,
    pub at: f64,
    pub cpu: &'a (dyn Fn(V3) -> f64 + Sync),
    pub sources: &'a Sources<'a>,
}

impl<'a> OnGpu<'a> {
    fn on_cpu(&self) -> Pointwise<'a> {
        Pointwise { f: self.cpu, at: self.at, sources: self.sources, tol: self.tol }
    }

    /// Intervals where the GPU's hold zero read again on the CPU.
    fn sure(&self, pts: impl Iterator<Item = V3>, out: &mut [[f64; 2]]) {
        let pts: Vec<V3> = pts.collect();
        let unsure: Vec<usize> = (0..out.len()).filter(|&i| !(out[i][0] >= 0.0 || out[i][1] < 0.0)).collect();
        let again: Vec<f64> = unsure.par_iter().map(|&i| (self.cpu)(pts[i])).collect();
        for (&i, v) in unsure.iter().zip(again) {
            out[i] = [v, v];
        }
    }
}

impl Field for OnGpu<'_> {
    fn values(&self, pts: &[V3]) -> Vec<[f64; 2]> {
        let Ok(iv) = self.gpu.values(self.of, pts, self.limit, self.tol) else { return self.on_cpu().values(pts) };
        let mut out: Vec<[f64; 2]> = iv.into_iter().map(|[lo, hi]| if self.sgn >= 0.0 { [lo + self.off, hi + self.off] } else { [self.off - hi, self.off - lo] }).collect();
        self.sure(pts.iter().copied(), &mut out);
        out
    }

    /// The cut is where sgn field + off reads `at`, so the kernels look for the zero of
    /// sgn field + off - at.
    fn roots(&self, sides: &[(V3, V3)]) -> Vec<f64> {
        match self.gpu.roots(self.of, sides, self.limit, self.tol, (self.sgn, self.off - self.at)) {
            Ok(Some(t)) => t,
            _ => self.on_cpu().roots(sides),
        }
    }

    fn settle(&self, items: &[(V3, u32, bool)]) -> Vec<(V3, Option<V3>, [f64; 2])> {
        // The box's corners stay put (its faces are flat); the rest settle on the GPU.
        let (moved, flat): (Vec<usize>, Vec<usize>) = (0..items.len()).partition(|&i| (self.sources)(items[i].1).is_some());
        let mut out: Vec<(V3, Option<V3>, [f64; 2])> = items.iter().map(|&(x, _, _)| (x, None, [0.0, 0.0])).collect();
        let flat_pts: Vec<V3> = flat.iter().map(|&i| items[i].0).collect();
        for (&i, v) in flat.iter().zip(self.values(&flat_pts)) {
            out[i].2 = v;
        }
        let asks: Vec<(V3, &Sweep, bool)> = moved.iter().map(|&i| (items[i].0, (self.sources)(items[i].1).expect("a move"), items[i].2)).collect();
        match self.gpu.settle(self.of, &asks, self.limit, self.tol, (self.sgn, self.off - self.at)) {
            Ok(Some(got)) => {
                let mut iv: Vec<[f64; 2]> = got.iter().map(|g| [g.2[0] + self.at, g.2[1] + self.at]).collect();
                self.sure(got.iter().map(|g| g.0), &mut iv);
                for ((&i, g), v) in moved.iter().zip(got).zip(iv) {
                    out[i] = (g.0, Some(g.1), v);
                }
            }
            _ => {
                let mine: Vec<(V3, u32, bool)> = moved.iter().map(|&i| items[i]).collect();
                for (&i, g) in moved.iter().zip(self.on_cpu().settle(&mine)) {
                    out[i] = g;
                }
            }
        }
        out
    }
}

/// A piece is on the field's non-negative side by its interval (the midpoint when the
/// interval holds zero, which a [`Field`] avoids).
fn nonneg(v: [f64; 2]) -> bool {
    v[0] >= 0.0 || (v[1] >= 0.0 && v[0] + v[1] >= 0.0)
}

/// A piece whose corners all read within this of zero, and no longer than this many
/// `finest`, lies on the other surface and is decided whole by the sign of its mean.
const COINCIDE: f64 = 1e-2;
const COINCIDE_SIZE: f64 = 8.0;
/// The tie between coincident surfaces (see the module notes), a tenth of that.
const TIE: f64 = 0.1 * COINCIDE;

/// Every point of a triangle lies within this many times its longest side of a corner
/// (the circumradius of an acute triangle is at most side / sqrt 3; an obtuse one's far
/// points sit on its longest side).
const COVER: f64 = 0.578;

/// What to do with a piece, from its corners' intervals.
#[derive(Clone, Copy)]
enum Step {
    Kept,
    Dropped,
    /// At the finest: cut along the zero.
    Cut,
    /// Split the side from corner k to the next.
    Split(usize),
}

/// A piece being decided: corners (point, normal), their intervals, its halves once split.
struct Node {
    c: [(V3, V3); 3],
    v: [[f64; 2]; 3],
    kids: Option<[usize; 2]>,
}

fn decide(n: &Node, finest: f64) -> Step {
    let (c, v) = (n.c, n.v);
    let Some(piece) = Patch::facing(c[0], c[1], c[2]) else { return Step::Dropped };
    let longest = piece.longest();
    if !longest.is_finite() {
        return Step::Dropped;
    }
    let lo = v[0][0].min(v[1][0]).min(v[2][0]);
    let hi = v[0][1].max(v[1][1]).max(v[2][1]);
    if lo >= COVER * longest || (lo >= 0.0 && longest <= finest) {
        return Step::Kept;
    }
    if hi < -COVER * longest || (hi < 0.0 && longest <= finest) {
        return Step::Dropped;
    }
    if lo >= -COINCIDE && hi <= COINCIDE && longest <= COINCIDE_SIZE * finest {
        let mean: f64 = v.iter().map(|x| x[0] + x[1]).sum();
        return if mean >= 0.0 { Step::Kept } else { Step::Dropped };
    }
    if longest <= finest {
        return Step::Cut;
    }
    // Split the longest side to look closer (halves stay as round as the piece).
    Step::Split((0..3).max_by(|&a, &b| (c[(a + 1) % 3].0 - c[a].0).len_sq().total_cmp(&(c[(b + 1) % 3].0 - c[b].0).len_sq())).unwrap_or(0))
}

/// A corner of a piece: its point and the surface's normal there.
type Corner = (V3, V3);
/// A piece to cut: its corners, their intervals, its source.
type Cutting = ([Corner; 3], [[f64; 2]; 3], u32);

/// The sides a cut crosses, each from its non-negative corner to its negative one.
fn crossed(c: &[Corner; 3], v: &[[f64; 2]; 3]) -> Vec<(Corner, Corner)> {
    let inside: Vec<usize> = (0..3).filter(|&i| nonneg(v[i])).collect();
    match inside.len() {
        1 => {
            let i = inside[0];
            vec![(c[i], c[(i + 1) % 3]), (c[i], c[(i + 2) % 3])]
        }
        2 => {
            let k = (0..3).find(|&k| !nonneg(v[k])).unwrap_or(0);
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            vec![(c[j], c[k]), (c[i], c[k])]
        }
        _ => Vec::new(),
    }
}

/// The piece cut along the zero, keeping the non-negative side, with the crossed sides'
/// new corners (in `crossed`'s order).
fn cut(c: &[Corner; 3], v: &[[f64; 2]; 3], r: &[Corner], src: u32, out: &mut Vec<Patch>) {
    let mut push = |a: Corner, b: Corner, d: Corner| {
        if let Some(p) = Patch::facing(a, b, d) {
            out.push(p.on(src));
        }
    };
    let inside: Vec<usize> = (0..3).filter(|&i| nonneg(v[i])).collect();
    match inside.len() {
        1 => {
            let i = inside[0];
            push(c[i], r[0], r[1]);
        }
        2 => {
            let k = (0..3).find(|&k| !nonneg(v[k])).unwrap_or(0);
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            push(c[i], c[j], r[0]);
            push(c[i], r[0], r[1]);
        }
        3 => push(c[0], c[1], c[2]),
        _ => {}
    }
}

/// Cuts the given pieces along the zero: the roots of every crossed side read in one
/// batch, then (`settle`) settled onto their pieces' surfaces and the cut in another.
fn cut_all(items: &[Cutting], field: &dyn Field, settle: bool) -> Vec<Vec<Patch>> {
    let sides: Vec<Vec<(Corner, Corner)>> = items.iter().map(|(c, v, _)| crossed(c, v)).collect();
    let flat: Vec<(V3, V3)> = sides.iter().flatten().map(|(a, b)| (a.0, b.0)).collect();
    let ts = field.roots(&flat);
    let mut corners: Vec<Corner> = sides.iter().flatten().zip(&ts).map(|((a, b), &t)| (a.0 + scale(b.0 - a.0, t), unit(a.1 + scale(b.1 - a.1, t)))).collect();
    if settle {
        let srcs = items.iter().zip(&sides).flat_map(|((_, _, src), x)| std::iter::repeat_n(*src, x.len()));
        let asks: Vec<(V3, u32, bool)> = corners.iter().zip(srcs).map(|(c, src)| (c.0, src, true)).collect();
        corners = corners.iter().zip(field.settle(&asks)).map(|(c, (x, g, _))| kept(*c, x, g)).collect();
    }
    let mut at = 0;
    items
        .iter()
        .zip(&sides)
        .map(|((c, v, src), x)| {
            let mut out = Vec::new();
            cut(c, v, &corners[at..at + x.len()], *src, &mut out);
            at += x.len();
            out
        })
        .collect()
}

/// The parts of `patches` where the field is non-negative, read at their corners only
/// (for a surface on the field's zero but where it folds; see the module notes).
pub fn trim_in(patches: Vec<Patch>, field: &dyn Field) -> Vec<Patch> {
    let corners: Vec<V3> = patches.iter().flat_map(|p| p.p).collect();
    let vals = field.values(&corners);
    let items: Vec<Cutting> = patches.iter().enumerate().map(|(i, p)| ([(p.p[0], p.n[0]), (p.p[1], p.n[1]), (p.p[2], p.n[2])], [vals[3 * i], vals[3 * i + 1], vals[3 * i + 2]], p.src)).collect();
    cut_all(&items, field, false).into_iter().flatten().collect()
}

/// As [`trim_in`], the field read on the CPU.
pub fn trim(patches: Vec<Patch>, f: &(dyn Fn(V3) -> f64 + Sync)) -> Vec<Patch> {
    trim_in(patches, &Pointwise { f, at: 0.0, sources: &|_| None, tol: 0.0 })
}

/// The parts of `patches` where the field is non-negative (see the module notes): every
/// piece decided a level at a time, the new points of a level's splits settled onto their
/// pieces' surfaces and read in one batch.
pub fn keep_in(patches: Vec<Patch>, field: &dyn Field, finest: f64) -> Vec<Patch> {
    if patches.is_empty() {
        return patches;
    }
    let corners: Vec<V3> = patches.iter().flat_map(|p| p.p).collect();
    let vals = field.values(&corners);
    let mut nodes: Vec<Node> = patches.iter().enumerate().map(|(i, p)| Node { c: [(p.p[0], p.n[0]), (p.p[1], p.n[1]), (p.p[2], p.n[2])], v: [vals[3 * i], vals[3 * i + 1], vals[3 * i + 2]], kids: None }).collect();
    let src: Vec<u32> = patches.iter().map(|p| p.src).collect();
    let mut owner: Vec<usize> = (0..patches.len()).collect();
    let mut steps: Vec<Step> = vec![Step::Kept; nodes.len()];
    let mut open: Vec<usize> = (0..nodes.len()).collect();
    let mut cuts: Vec<usize> = Vec::new();
    while !open.is_empty() {
        let decided: Vec<Step> = open.par_iter().map(|&i| decide(&nodes[i], finest)).collect();
        let mut splits: Vec<(usize, usize)> = Vec::new();
        for (&i, st) in open.iter().zip(decided) {
            steps[i] = st;
            match st {
                Step::Split(k) => splits.push((i, k)),
                Step::Cut => cuts.push(i),
                _ => {}
            }
        }
        let mids: Vec<(V3, u32, bool)> = splits.iter().map(|&(i, k)| (mid(nodes[i].c[k], nodes[i].c[(k + 1) % 3]).0, src[owner[i]], false)).collect();
        let settled = field.settle(&mids);
        open.clear();
        for (&(i, k), (x, g, fm)) in splits.iter().zip(settled) {
            let (c, v) = (nodes[i].c, nodes[i].v);
            let (a, b, o) = (k, (k + 1) % 3, (k + 2) % 3);
            let m = kept(mid(c[a], c[b]), x, g);
            let first = nodes.len();
            nodes.push(Node { c: [c[a], m, c[o]], v: [v[a], fm, v[o]], kids: None });
            nodes.push(Node { c: [m, c[b], c[o]], v: [fm, v[b], v[o]], kids: None });
            nodes[i].kids = Some([first, first + 1]);
            owner.extend([owner[i], owner[i]]);
            steps.extend([Step::Kept, Step::Kept]);
            open.extend([first, first + 1]);
        }
    }
    let items: Vec<Cutting> = cuts.iter().map(|&i| (nodes[i].c, nodes[i].v, src[owner[i]])).collect();
    let mut pieces: Vec<Option<Vec<Patch>>> = (0..nodes.len()).map(|_| None).collect();
    for (&i, out) in cuts.iter().zip(cut_all(&items, field, true)) {
        pieces[i] = Some(out);
    }
    // Settle from the finest up: halves that came out whole one way merge back, so only
    // pieces the zero crosses stay split.
    #[derive(Clone)]
    enum Res {
        Kept,
        Dropped,
        Pieces(Vec<Patch>),
    }
    let mut res: Vec<Res> = vec![Res::Dropped; nodes.len()];
    for i in (0..nodes.len()).rev() {
        res[i] = match steps[i] {
            Step::Kept => Res::Kept,
            Step::Dropped => Res::Dropped,
            Step::Cut => Res::Pieces(pieces[i].take().unwrap_or_default()),
            Step::Split(_) => {
                let [x, y] = nodes[i].kids.unwrap_or([i, i]);
                match (&res[x], &res[y]) {
                    (Res::Kept, Res::Kept) => Res::Kept,
                    (Res::Dropped, Res::Dropped) => Res::Dropped,
                    _ => {
                        let mut out = Vec::new();
                        for k in [x, y] {
                            match std::mem::replace(&mut res[k], Res::Dropped) {
                                Res::Kept => out.extend(Patch::facing(nodes[k].c[0], nodes[k].c[1], nodes[k].c[2]).map(|p| p.on(src[owner[i]]))),
                                Res::Dropped => {}
                                Res::Pieces(v) => out.extend(v),
                            }
                        }
                        Res::Pieces(out)
                    }
                }
            }
        };
    }
    let mut out = Vec::new();
    for (i, p) in patches.into_iter().enumerate() {
        match std::mem::replace(&mut res[i], Res::Dropped) {
            Res::Kept => out.push(p),
            Res::Dropped => {}
            Res::Pieces(v) => out.extend(v),
        }
    }
    out
}

/// As [`keep_in`], the field read on the CPU.
pub fn keep(patches: Vec<Patch>, f: &(dyn Fn(V3) -> f64 + Sync), finest: f64) -> Vec<Patch> {
    keep_in(patches, &Pointwise { f, at: 0.0, sources: &|_| None, tol: 0.0 }, finest)
}

/// The stock and its skin, moves applied one at a time.
pub struct Skin {
    pub stock: Stock,
    pub patches: Vec<Patch>,
    /// Field tolerance, mm.
    pub tol: f64,
    /// Chords and cut resolution, mm.
    pub finest: f64,
    /// Where the fields are read when there is a GPU (the CPU otherwise).
    pub gpu: Option<Arc<Fields>>,
}

impl Skin {
    /// The box, its six faces as skin.
    pub fn new(bounds: Aabb, tol: f64, finest: f64) -> Skin {
        let (a, b) = (bounds.min, bounds.max);
        let corner = |i: usize| V3::new(if i & 1 == 0 { a.x } else { b.x }, if i & 2 == 0 { a.y } else { b.y }, if i & 4 == 0 { a.z } else { b.z });
        let mut patches = Vec::new();
        // Each face: four corners and its outward normal.
        for (q, n) in [
            ([0, 2, 6, 4], V3::new(-1.0, 0.0, 0.0)),
            ([1, 5, 7, 3], V3::new(1.0, 0.0, 0.0)),
            ([0, 4, 5, 1], V3::new(0.0, -1.0, 0.0)),
            ([2, 3, 7, 6], V3::new(0.0, 1.0, 0.0)),
            ([0, 1, 3, 2], V3::new(0.0, 0.0, -1.0)),
            ([4, 6, 7, 5], V3::new(0.0, 0.0, 1.0)),
        ] {
            let c = q.map(|i| (corner(i), n));
            patches.extend(Patch::facing(c[0], c[1], c[2]));
            patches.extend(Patch::facing(c[0], c[2], c[3]));
        }
        Skin { stock: Stock::new(bounds), patches, tol, finest, gpu: None }
    }

    /// The chord tolerance of the move surfaces the skin is made of, mm.
    pub fn chord(&self) -> f64 {
        0.25 * self.finest
    }

    /// Takes `s` out of the stock and updates the skin.
    pub fn apply(&mut self, s: Sweep) {
        let (tol, finest) = (self.tol, self.finest);
        let bb = s.bounds().grow(finest);
        let (near, mut far): (Vec<Patch>, Vec<Patch>) = std::mem::take(&mut self.patches).into_iter().partition(|p| {
            let b = p.bounds();
            b.max.x >= bb.min.x && b.min.x <= bb.max.x && b.max.y >= bb.min.y && b.min.y <= bb.max.y && b.max.z >= bb.min.z && b.min.z <= bb.max.z
        });
        // Ties go to the old skin, and cuts sit half a tie inside the side kept (see the
        // module notes).
        let (tie, at) = (TIE, 0.5 * TIE);
        let me = self.stock.len() as u32;
        let stock = &self.stock;
        let sources = |k: u32| -> Option<&Sweep> {
            if k == BOX {
                None
            } else if k == me {
                Some(&s)
            } else {
                Some(stock.sweep(k as usize))
            }
        };
        let gpu = self.gpu.as_deref();
        let by_move = |p: V3| s.sd(p, tol) + tie;
        let move_field = |limit: f64| -> Box<dyn Field + '_> {
            match gpu {
                Some(g) => Box::new(OnGpu { gpu: g, of: Of::Sweep(&s), limit, tol, sgn: 1.0, off: tie, at, cpu: &by_move, sources: &sources }),
                None => Box::new(Pointwise { f: &by_move, at, sources: &sources, tol }),
            }
        };
        let near_limit = 2.0 * near.iter().map(Patch::longest).fold(finest, f64::max);
        far.extend(keep_in(near, move_field(near_limit).as_ref(), finest));
        // The move's own surface (its source the index the move is about to take): off its
        // own fold, and only where it was material. It faces out of the move, which is into
        // the air the stock now has there, so on the stock's skin it faces the other way.
        let surface = envelope(&s, self.chord(), 0.5 * COINCIDE_SIZE * finest);
        // The old stock's field only matters out to the size of the largest piece.
        let limit = 2.0 * surface.iter().map(Patch::longest).fold(finest, f64::max);
        let trimmed = trim_in(surface, move_field(limit).as_ref());
        let own: Vec<Patch> = trimmed.into_iter().map(|p| p.on(me).flipped()).collect();
        let by_stock = |p: V3| -stock.sd_clamped(p, limit, tol) - tie;
        far.extend(match gpu {
            Some(g) => keep_in(own, &OnGpu { gpu: g, of: Of::Stock(stock), limit, tol, sgn: -1.0, off: -tie, at, cpu: &by_stock, sources: &sources }, finest),
            None => keep_in(own, &Pointwise { f: &by_stock, at, sources: &sources, tol }, finest),
        });
        self.stock.apply(s);
        self.patches = far;
    }
}
