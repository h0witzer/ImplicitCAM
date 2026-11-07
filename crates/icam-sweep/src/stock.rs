//! The stock: the box less every sweep applied so far, F = max(box, -min_k s_k),
//! negative inside the material.
//!
//! Sweeps are applied one at a time, at runtime. They are found through a forest of
//! bounding-box trees: new sweeps wait in a short list, a full list becomes a small tree,
//! and two trees of about the same size merge into one, so applying n sweeps costs
//! O(n log^2 n) in all and a query visits O(log n) trees.
//!
//! Every tree node knows the lowest and highest sweep number below it, so a query can ask
//! for the sweeps of the stock as it stood after any number of them (`near_in`).
//!
//! Between two moves' ends the stock is also cut by the move the tool is on, up to where
//! the tool stands: its tail (`set_tail`), a sweep of its own read as number `len()`.
//! When the tail moves on, what its move gained is a piece read as number `len() + 1`: along
//! the same move, from the old part to the new; on to a later move, the rest of the old one.
//! So a step forward reads only that, never the whole move the tool was on. Queries by
//! number never find either; the cache adds them where their boxes meet (cache.rs).
//!
//! A query needs the sweeps only where they can win: F = max(b, -U) with b the box's
//! distance and U the least sweep distance, so a sweep matters only if its distance is
//! below -b, and a tree node only if its box is nearer than that or holds the point (a
//! point inside several sweeps reads the deepest).

use std::ops::Range;

use crate::geom::{Aabb, V3};
use crate::sweep::{may_hold, Sweep};

/// Sweeps waiting for a tree.
const PENDING: usize = 64;
/// Sweeps per leaf.
const LEAF: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Node {
    bb: Aabb,
    /// A leaf's first item, or an inner node's right child (its left child follows it).
    index: u32,
    /// Items in a leaf; 0 for an inner node.
    count: u32,
    /// The lowest and highest sweep number below the node.
    low: u32,
    high: u32,
}

#[derive(Clone, Debug)]
struct Tree {
    nodes: Vec<Node>,
    items: Vec<u32>,
}

impl Tree {
    fn build(mut items: Vec<u32>, boxes: &[Aabb]) -> Tree {
        let mut t = Tree { nodes: Vec::with_capacity(2 * items.len() / LEAF + 1), items: Vec::new() };
        let n = items.len();
        t.split(&mut items, 0, n, boxes);
        t.items = items;
        t
    }

    fn split(&mut self, items: &mut [u32], first: usize, end: usize, boxes: &[Aabb]) -> u32 {
        let me = self.nodes.len() as u32;
        let bb = items[first..end].iter().fold(Aabb::EMPTY, |b, &k| b.union(boxes[k as usize]));
        let low = items[first..end].iter().copied().min().unwrap_or(0);
        let high = items[first..end].iter().copied().max().unwrap_or(0);
        if end - first <= LEAF {
            self.nodes.push(Node { bb, index: first as u32, count: (end - first) as u32, low, high });
            return me;
        }
        self.nodes.push(Node { bb, index: 0, count: 0, low, high });
        // Split at the median of the box centers along the widest spread of them.
        let c = |k: u32| boxes[k as usize].center();
        let cb = items[first..end].iter().fold(Aabb::EMPTY, |b, &k| b.union(Aabb::new(c(k), c(k))));
        let e = cb.max - cb.min;
        let key = |k: u32| {
            let p = c(k);
            if e.x >= e.y && e.x >= e.z {
                p.x
            } else if e.y >= e.z {
                p.y
            } else {
                p.z
            }
        };
        let mid = (first + end) / 2;
        items[first..end].select_nth_unstable_by(mid - first, |&a, &b| key(a).total_cmp(&key(b)));
        self.split(items, first, mid, boxes);
        let right = self.split(items, mid, end, boxes);
        self.nodes[me as usize].index = right;
        me
    }
}

pub struct Stock {
    pub bounds: Aabb,
    sweeps: Vec<Sweep>,
    boxes: Vec<Aabb>,
    trees: Vec<Tree>,
    pending: Vec<u32>,
    /// The tail: sweep k's first `part` (see the module's notes).
    tail: Option<(usize, f64, Sweep)>,
    /// What the old tail's sweep gained when the tail last moved on: which sweep, from and
    /// to which part.
    piece: Option<(usize, f64, f64, Sweep)>,
}

impl Stock {
    pub fn new(bounds: Aabb) -> Stock {
        Stock { bounds, sweeps: Vec::new(), boxes: Vec::new(), trees: Vec::new(), pending: Vec::new(), tail: None, piece: None }
    }

    /// The tail becomes the first `part` of sweep k (0 < part < 1), the stock as it stands
    /// while the tool is that far along sweep k, or none (part 0). False, and no tail, when
    /// sweep k's motion cannot be cut short.
    pub fn set_tail(&mut self, k: usize, part: f64) -> bool {
        if self.tail.as_ref().is_some_and(|t| t.0 == k && t.1 == part) {
            return true;
        }
        let before = self.tail.take().map(|t| (t.0, t.1));
        self.piece = None;
        // What the old tail's sweep gained: along it, from the old part to the new; on to a
        // later sweep, the rest of it.
        let gained = match before {
            Some((k0, a)) if k0 == k && a < part => Some((k, a, part.min(1.0))),
            Some((k0, a)) if k0 < k && a > 0.0 && k0 < self.sweeps.len() => Some((k0, a, 1.0)),
            _ => None,
        };
        if let Some((k0, a, b)) = gained {
            let s = &self.sweeps[k0];
            self.piece = s.motion.piece(a, b).map(|m| (k0, a, b, Sweep::new(s.tool.clone(), m)));
        }
        if part <= 0.0 || k >= self.sweeps.len() {
            return part <= 0.0;
        }
        let s = &self.sweeps[k];
        let part = part.min(1.0);
        match s.motion.piece(0.0, part) {
            Some(m) => {
                self.tail = Some((k, part, Sweep::new(s.tool.clone(), m)));
                true
            }
            None => false,
        }
    }

    /// What the old tail's sweep gained when the tail last moved on: which sweep, from and to
    /// which part, and the sweep of that piece.
    pub fn piece(&self) -> Option<(usize, f64, f64, &Sweep)> {
        self.piece.as_ref().map(|p| (p.0, p.1, p.2, &p.3))
    }

    /// The tail: which sweep, how far along it, and the sweep of that part.
    pub fn tail(&self) -> Option<(usize, f64, &Sweep)> {
        self.tail.as_ref().map(|t| (t.0, t.1, &t.2))
    }

    pub fn len(&self) -> usize {
        self.sweeps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sweeps.is_empty()
    }

    /// Sweep k; number `len()` is the tail, `len() + 1` its piece.
    pub fn sweep(&self, k: usize) -> &Sweep {
        match (&self.tail, &self.piece) {
            (Some(t), _) if k == self.sweeps.len() => &t.2,
            (_, Some(p)) if k == self.sweeps.len() + 1 => &p.3,
            _ => &self.sweeps[k],
        }
    }

    /// Takes the material `s` sweeps through out of the stock; returns its index.
    pub fn apply(&mut self, s: Sweep) -> usize {
        let k = self.sweeps.len();
        self.boxes.push(s.bounds());
        self.sweeps.push(s);
        self.pending.push(k as u32);
        if self.pending.len() >= PENDING {
            let items = std::mem::take(&mut self.pending);
            self.trees.push(Tree::build(items, &self.boxes));
            while self.trees.len() >= 2 {
                let n = self.trees.len();
                let (a, b) = (self.trees[n - 2].items.len(), self.trees[n - 1].items.len());
                if a > 2 * b {
                    break;
                }
                let mut items = self.trees.pop().map(|t| t.items).unwrap_or_default();
                items.extend(self.trees.pop().map(|t| t.items).unwrap_or_default());
                self.trees.push(Tree::build(items, &self.boxes));
            }
        }
        k
    }

    /// The sweeps whose boxes meet `bb`, in no particular order.
    pub fn near(&self, bb: Aabb) -> Vec<usize> {
        self.near_in(bb, 0..self.len())
    }

    /// The sweeps numbered within `moves` whose boxes meet `bb`. Their order depends on the
    /// forest and `bb` only, not on `moves`: a box asked about at two playheads lists the
    /// sweeps both have in the same order (the cache's re-bakes rely on it).
    pub fn near_in(&self, bb: Aabb, moves: Range<usize>) -> Vec<usize> {
        let (lo, hi) = (moves.start.min(u32::MAX as usize) as u32, moves.end.min(u32::MAX as usize) as u32);
        let meets = |b: &Aabb| b.max.x >= bb.min.x && b.min.x <= bb.max.x && b.max.y >= bb.min.y && b.min.y <= bb.max.y && b.max.z >= bb.min.z && b.min.z <= bb.max.z;
        let wanted = |k: u32| k >= lo && k < hi && meets(&self.boxes[k as usize]);
        let mut out: Vec<usize> = self.pending.iter().filter(|&&k| wanted(k)).map(|&k| k as usize).collect();
        if lo >= hi {
            return out;
        }
        let mut stack = Vec::with_capacity(64);
        for t in &self.trees {
            stack.clear();
            stack.push(0u32);
            while let Some(i) = stack.pop() {
                let n = t.nodes[i as usize];
                if !meets(&n.bb) || n.high < lo || n.low >= hi {
                    continue;
                }
                if n.count > 0 {
                    out.extend(t.items[n.index as usize..(n.index + n.count) as usize].iter().filter(|&&k| wanted(k)).map(|&k| k as usize));
                } else {
                    stack.push(i + 1);
                    stack.push(n.index);
                }
            }
        }
        out
    }

    /// The least sweep distance at p if it is below `cut`, and which sweep has it.
    pub fn nearest_sweep(&self, p: V3, cut: f64, tol: f64) -> Option<(f64, usize)> {
        self.nearest_until(p, cut, f64::NEG_INFINITY, tol)
    }

    /// As `nearest_sweep`, but stopping as soon as a sweep at or below `enough` is found.
    fn nearest_until(&self, p: V3, cut: f64, enough: f64, tol: f64) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        let mut cut = cut;
        let mut visit = |k: u32, cut: &mut f64| {
            if let Some(s) = self.sweeps[k as usize].sd_below(p, *cut, tol) {
                *cut = s;
                best = Some((s, k as usize));
            }
        };
        for &k in &self.pending {
            visit(k, &mut cut);
            if cut <= enough {
                return best;
            }
        }
        let mut stack = Vec::with_capacity(64);
        for t in &self.trees {
            stack.clear();
            stack.push(0u32);
            while let Some(i) = stack.pop() {
                let n = t.nodes[i as usize];
                if !may_hold(n.bb.dist(p), cut) {
                    continue;
                }
                if n.count > 0 {
                    for &k in &t.items[n.index as usize..(n.index + n.count) as usize] {
                        if may_hold(self.boxes[k as usize].dist(p), cut) {
                            visit(k, &mut cut);
                            if cut <= enough {
                                return best;
                            }
                        }
                    }
                } else {
                    // Nearer child last, so it is searched first.
                    let (l, r) = (i + 1, n.index);
                    if t.nodes[l as usize].bb.dist(p) <= t.nodes[r as usize].bb.dist(p) {
                        stack.push(r);
                        stack.push(l);
                    } else {
                        stack.push(l);
                        stack.push(r);
                    }
                }
            }
        }
        best
    }

    /// The stock's field at p: negative inside the material, within `tol`.
    pub fn sd(&self, p: V3, tol: f64) -> f64 {
        let b = self.bounds.sd(p);
        match self.nearest_sweep(p, -b, tol) {
            Some((u, _)) => b.max(-u),
            None => b,
        }
    }

    /// The stock's field at p clamped to [-limit, limit], the search stopping as soon as
    /// the answer is past either end: enough to decide a piece no longer than `limit`.
    pub fn sd_clamped(&self, p: V3, limit: f64, tol: f64) -> f64 {
        let b = self.bounds.sd(p);
        if b >= limit {
            return limit;
        }
        // Sweeps at or past min(-b, limit) cannot change the clamped answer.
        match self.nearest_until(p, (-b).min(limit), -limit, tol) {
            Some((u, _)) => b.max(-u).clamp(-limit, limit),
            None => b.clamp(-limit, limit),
        }
    }

    pub fn is_material(&self, p: V3) -> bool {
        self.sd(p, 1e-9) < 0.0
    }
}
