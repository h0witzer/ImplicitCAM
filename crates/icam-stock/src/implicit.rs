//! The machined stock as a function, never a grid of material.
//!
//! Material is present at p iff p is inside the initial stock box and no tool sweep
//! contains p. With a vertical tool (3-axis) every move lowers a height function: at
//! (x,y) the tool bottom passes as low as h_i(x,y), so the stock is
//! { z < H(x,y) = min_i h_i(x,y) } inside the box. Each h_i is closed-form for flat /
//! bull end mills along lines and (helical) arcs and for drill points, so there is no
//! grid: accuracy is set by the program's own moves.
//!
//! Distances (mm) used by the comparison:
//!   to air    material point -> nearest air: exact 3D distance to the nearest sweep,
//!             the sheet top or its sides (see `distance_to`).
//!   depth     air point -> distance to the nearest remaining stock (a search over
//!             the exact height function; see `depth_to_material`).
//! Holders and shanks are not modeled (collision checking is a separate job).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use icam_post::{Move, MoveKind, ToolDef, V3};

const TAU: f64 = 2.0 * PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Line,
    Arc,
    Drill,
}

/// One tool move, pre-digested. Angles are radians; arcs run from `theta0` through
/// `sweep` (signed: + counterclockwise).
#[derive(Clone, Copy, Debug)]
struct Sweep {
    kind: Shape,
    a: V3,
    b: V3,
    c: V3,
    r: f64,   // tool radius
    rc: f64,  // corner radius
    cot: f64, // drill: cot(half point angle); 0 for a mill
    arc_r: f64,
    theta0: f64,
    sweep: f64,
    mv: usize,
}

impl Sweep {
    #[allow(clippy::too_many_arguments)]
    fn new(kind: Shape, a: V3, b: V3, c: V3, r: f64, rc: f64, cot: f64, mv: usize) -> Sweep {
        let (mut arc_r, mut theta0) = (0.0, 0.0);
        if kind == Shape::Arc {
            arc_r = ((a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y)).sqrt();
            theta0 = (a.y - c.y).atan2(a.x - c.x);
        }
        Sweep { kind, a, b, c, r, rc, cot, arc_r, theta0, sweep: 0.0, mv }
    }
}

/// [`ImplicitStock::tables`]: every sweep as 16 numbers (kind 0 line, 1 arc, 2 drill;
/// start a, end b, arc center c; tool radius, corner radius, drill cot; arc radius,
/// start angle, signed sweep), the stock top, and the candidate grid (cell size, its
/// corner, size, and CSR lists of sweeps per cell), all relative to `origin`.
pub struct StockTables {
    pub origin: V3,
    pub sweeps: Vec<f32>,
    pub top: f32,
    pub cell: f32,
    pub ox: f32,
    pub oy: f32,
    pub nx: u32,
    pub ny: u32,
    pub cell_start: Vec<u32>,
    pub cell_items: Vec<u32>,
}

/// The stock a program leaves (see the module notes).
pub struct ImplicitStock {
    min: V3,
    max: V3,
    sweeps: Vec<Sweep>,
    cell: f64,
    ox: f64,
    oy: f64,
    nx: i64,
    ny: i64,
    cell_start: Vec<usize>, // CSR: sweeps overlapping each XY cell
    cell_items: Vec<usize>,
}

impl ImplicitStock {
    /// The candidate grid's cell (mm) the checker has always used.
    pub const CELL_MM: f64 = 4.0;

    /// The stock left in the box `box_min`..`box_max` by `moves`. Moves whose tool is
    /// not in `tools`, and moves entirely above the stock, cut nothing. `cell_mm` only
    /// sizes the grid that finds candidate sweeps; every answer is exact.
    pub fn new(box_min: V3, box_max: V3, moves: &[Move], tools: &BTreeMap<i32, ToolDef>, cell_mm: f64) -> ImplicitStock {
        let mut sweeps = Vec::new();
        for (m, mv) in moves.iter().enumerate() {
            let Some(tool) = tools.get(&mv.tool) else { continue };
            let r = tool.diameter / 2.0;
            let rc = tool.corner_radius.min(r);
            let cot = if tool.kind == "DRILL" { 1.0 / (tool.tip_angle_deg / 2.0 * PI / 180.0).tan() } else { 0.0 };
            if mv.from.z.min(mv.to.z) > box_max.z {
                continue; // entirely above the stock
            }
            match mv.kind {
                MoveKind::ArcCw | MoveKind::ArcCcw => {
                    let mut s = Sweep::new(Shape::Arc, mv.from, mv.to, mv.center, r, rc, cot, m);
                    let t1 = (mv.to.y - mv.center.y).atan2(mv.to.x - mv.center.x);
                    let d = t1 - s.theta0;
                    let ccw = mv.kind == MoveKind::ArcCcw;
                    let full = (mv.to - mv.from).len() < 1e-6;
                    s.sweep = match (full, ccw) {
                        (true, true) => TAU,
                        (true, false) => -TAU,
                        (false, true) => mod_2pi(d),
                        (false, false) => -mod_2pi(-d),
                    };
                    sweeps.push(s);
                }
                MoveKind::Drill => sweeps.push(Sweep::new(Shape::Drill, mv.from, mv.to, V3::default(), r, 0.0, cot, m)),
                MoveKind::Rapid | MoveKind::Feed => sweeps.push(Sweep::new(Shape::Line, mv.from, mv.to, V3::default(), r, rc, cot, m)),
            }
        }

        // XY grid over the stock (plus a tool radius of margin).
        let cell = cell_mm;
        let (ox, oy) = (box_min.x - 20.0, box_min.y - 20.0);
        let nx = ((box_max.x - ox + 20.0) / cell).ceil() as i64 + 1;
        let ny = ((box_max.y - oy + 20.0) / cell).ceil() as i64 + 1;
        let cells = (nx * ny) as usize;
        let mut counts = vec![0usize; cells + 1];
        let mut spans = Vec::with_capacity(sweeps.len());
        for s in &sweeps {
            let (lo, hi) = footprint_box(s);
            let x0 = (((lo.x - ox) / cell) as i64).clamp(0, nx - 1);
            let x1 = (((hi.x - ox) / cell) as i64).clamp(0, nx - 1);
            let y0 = (((lo.y - oy) / cell) as i64).clamp(0, ny - 1);
            let y1 = (((hi.y - oy) / cell) as i64).clamp(0, ny - 1);
            spans.push((x0, x1, y0, y1));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    counts[(y * nx + x) as usize + 1] += 1;
                }
            }
        }
        let mut cell_start = vec![0usize; cells + 1];
        for c in 0..cells {
            cell_start[c + 1] = cell_start[c] + counts[c + 1];
        }
        let mut cell_items = vec![0usize; cell_start[cells]];
        let mut fill = cell_start.clone();
        for (i, &(x0, x1, y0, y1)) in spans.iter().enumerate() {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let c = (y * nx + x) as usize;
                    cell_items[fill[c]] = i;
                    fill[c] += 1;
                }
            }
        }
        ImplicitStock { min: box_min, max: box_max, sweeps, cell, ox, oy, nx, ny, cell_start, cell_items }
    }

    pub fn sweep_count(&self) -> usize {
        self.sweeps.len()
    }

    /// The smallest cutter radius (drills aside).
    pub fn min_tool_radius(&self) -> f64 {
        self.sweeps.iter().filter(|s| s.cot <= 0.0).map(|s| s.r).reduce(f64::min).unwrap_or(0.0)
    }

    pub fn box_min(&self) -> V3 {
        self.min
    }

    pub fn box_max(&self) -> V3 {
        self.max
    }

    /// The sweeps and the candidate grid as plain arrays, for a kernel that evaluates H
    /// elsewhere (`icam-sim` on the GPU). Positions are relative to `origin` (the box's
    /// low corner), so 32-bit floats keep about 0.0002 mm on a 2.4 m sheet.
    pub fn tables(&self) -> StockTables {
        let o = self.min;
        let mut sweeps = Vec::with_capacity(16 * self.sweeps.len());
        for s in &self.sweeps {
            let kind = match s.kind {
                Shape::Line => 0.0,
                Shape::Arc => 1.0,
                Shape::Drill => 2.0,
            };
            let (a, b, c) = (s.a - o, s.b - o, if s.kind == Shape::Arc { s.c - o } else { s.c });
            for v in [kind, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, s.r, s.rc, s.cot, s.arc_r, s.theta0, s.sweep] {
                sweeps.push(v as f32);
            }
        }
        StockTables {
            origin: o,
            sweeps,
            top: (self.max.z - o.z) as f32,
            cell: self.cell as f32,
            ox: (self.ox - o.x) as f32,
            oy: (self.oy - o.y) as f32,
            nx: self.nx as u32,
            ny: self.ny as u32,
            cell_start: self.cell_start.iter().map(|&v| v as u32).collect(),
            cell_items: self.cell_items.iter().map(|&v| v as u32).collect(),
        }
    }

    /// Profiling: candidate sweeps in the cell holding (x, y).
    pub fn candidate_count(&self, x: f64, y: f64) -> usize {
        self.candidates(x, y).len()
    }

    fn candidates(&self, x: f64, y: f64) -> &[usize] {
        if !(x.is_finite() && y.is_finite()) {
            return &[];
        }
        let cx = ((x - self.ox) / self.cell) as i64;
        let cy = ((y - self.oy) / self.cell) as i64;
        if cx < 0 || cy < 0 || cx >= self.nx || cy >= self.ny {
            return &[];
        }
        let c = (cy * self.nx + cx) as usize;
        &self.cell_items[self.cell_start[c]..self.cell_start[c + 1]]
    }

    pub fn in_box(&self, p: V3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y && p.z >= self.min.z && p.z <= self.max.z
    }

    /// H(x,y), capped at the stock top.
    pub fn cut_height(&self, x: f64, y: f64) -> f64 {
        let mut h = self.max.z;
        for &i in self.candidates(x, y) {
            h = h.min(height(&self.sweeps[i], x, y));
        }
        h
    }

    /// The first move (index into the moves given to `new`) that removes p: its tool
    /// bottom passes below p over it. Sweeps are kept in move order, so the first hit
    /// is it.
    pub fn first_cut(&self, p: V3) -> Option<usize> {
        self.candidates(p.x, p.y).iter().find(|&&i| height(&self.sweeps[i], p.x, p.y) < p.z).map(|&i| self.sweeps[i].mv)
    }

    pub fn is_material(&self, p: V3) -> bool {
        self.in_box(p) && p.z < self.cut_height(p.x, p.y)
    }

    /// Signed deviation of a point that should lie on the finished surface: + material
    /// left over it (thickness to air), - cut past it (depth).
    pub fn deviation(&self, p: V3) -> f64 {
        if !self.in_box(p) {
            // Outside the stock: air, but no cut went there.
            return 0.0;
        }
        let h = self.cut_height(p.x, p.y);
        if p.z < h {
            // The sheet's own uncut bottom face sits on the table: not machined, on size.
            if p.z - self.min.z < 1e-3 {
                return 0.0;
            }
            // Nearest air: over the sheet top, past its sides, or any sweep. The table
            // below is not air. Search outward cell by cell until the ring is farther
            // away than the best distance found.
            let mut best = (self.max.z - p.z).min((p.x - self.min.x).min(self.max.x - p.x).min((p.y - self.min.y).min(self.max.y - p.y)));
            let cx = ((p.x - self.ox) / self.cell) as i64;
            let cy = ((p.y - self.oy) / self.cell) as i64;
            let mut ring = 0i64;
            while ring <= 8 && (ring - 1) as f64 * self.cell < best {
                for y in cy - ring..=cy + ring {
                    for x in cx - ring..=cx + ring {
                        if (x - cx).abs().max((y - cy).abs()) != ring || x < 0 || y < 0 || x >= self.nx || y >= self.ny {
                            continue;
                        }
                        let c = (y * self.nx + x) as usize;
                        for &i in &self.cell_items[self.cell_start[c]..self.cell_start[c + 1]] {
                            best = best.min(distance_to(&self.sweeps[i], p));
                        }
                    }
                }
                ring += 1;
            }
            return best;
        }
        -self.depth_to_material(p, h)
    }

    /// Air point -> distance to the nearest remaining stock. Material in the column at
    /// (x,y) spans [bottom, H(x,y)), so the distance is
    ///   min over (x,y) of sqrt(|p.xy - (x,y)|^2 + max(0, p.z - H(x,y))^2).
    /// Searched on rings of growing radius (stop once a ring is farther than the best
    /// so far), then bisected along the best direction to ~1 micron.
    fn depth_to_material(&self, p: V3, h0: f64) -> f64 {
        const ANGLES: usize = 64;
        let mut best = if h0 > self.min.z { p.z - h0 } else { f64::INFINITY }; // straight down to this column's floor

        // Fast path: leave each sweep that contains p by its nearest wall or floor. To
        // reach material you must leave every containing sweep, so the deepest
        // containment is a lower bound; an exit that lands in material is an upper
        // bound. When they meet (the usual case - a design point just inside one cut)
        // the answer is exact.
        let mut lower = 0.0f64;
        for &i in self.candidates(p.x, p.y) {
            let Some((depth, exit)) = escape(&self.sweeps[i], p) else { continue };
            lower = lower.max(depth);
            if depth < best && self.is_material(exit) {
                best = depth;
            }
        }
        if best <= lower + 1e-6 {
            return best;
        }

        let (mut best_angle, mut inner) = (f64::NAN, 0.0);
        let column_gap = |x: f64, y: f64, r: f64| -> f64 {
            if x < self.min.x || x > self.max.x || y < self.min.y || y > self.max.y {
                return f64::INFINITY;
            }
            let hk = self.cut_height(x, y);
            if hk <= self.min.z {
                return f64::INFINITY; // cut through: no material here
            }
            let dz = (p.z - hk).max(0.0);
            (r * r + dz * dz).sqrt()
        };
        let mut r = 0.001;
        while r < 30.0 && r < best {
            for k in 0..ANGLES {
                let a = TAU * k as f64 / ANGLES as f64;
                let g = column_gap(p.x + r * a.cos(), p.y + r * a.sin(), r);
                if g < best {
                    best = g;
                    best_angle = a;
                    inner = r / 1.5;
                }
                if best <= r * 1.0001 {
                    break; // nothing on this ring can beat its own radius
                }
            }
            r *= 1.5;
        }
        if best_angle.is_nan() {
            return if best.is_infinite() { 0.0 } else { best };
        }
        // Bisect the radius along the best direction: the smallest r whose column still
        // reaches the point's height (or gives the best gap).
        let (mut lo, mut hi) = (inner, best);
        let mut it = 0;
        while it < 30 && hi - lo > 1e-4 {
            let mid = (lo + hi) / 2.0;
            let g = column_gap(p.x + mid * best_angle.cos(), p.y + mid * best_angle.sin(), mid);
            if g <= mid + 1e-9 {
                hi = mid;
                best = best.min(g);
            } else {
                lo = mid;
            }
            it += 1;
        }
        best
    }
}

fn mod_2pi(a: f64) -> f64 {
    let a = a % TAU;
    if a < 0.0 { a + TAU } else { a }
}

fn footprint_box(s: &Sweep) -> (V3, V3) {
    let r = s.r;
    if s.kind == Shape::Arc {
        let e = s.arc_r + r;
        return (V3::new(s.c.x - e, s.c.y - e, 0.0), V3::new(s.c.x + e, s.c.y + e, 0.0));
    }
    (V3::new(s.a.x.min(s.b.x) - r, s.a.y.min(s.b.y) - r, 0.0), V3::new(s.a.x.max(s.b.x) + r, s.a.y.max(s.b.y) + r, 0.0))
}

/// Tool bottom height above its tip at radial distance `rho` from the axis (+inf
/// outside the tool).
fn profile(s: &Sweep, rho: f64) -> f64 {
    if rho > s.r + 1e-9 {
        return f64::INFINITY;
    }
    if s.cot > 0.0 {
        return rho * s.cot; // drill point
    }
    let flat = s.r - s.rc;
    if rho <= flat {
        return 0.0;
    }
    let u = (rho - flat).min(s.rc);
    s.rc - (s.rc * s.rc - u * u).max(0.0).sqrt()
}

/// Lowest tool-bottom height this sweep reaches over (x, y).
fn height(s: &Sweep, x: f64, y: f64) -> f64 {
    match s.kind {
        Shape::Drill => {
            let rho = ((x - s.a.x) * (x - s.a.x) + (y - s.a.y) * (y - s.a.y)).sqrt();
            s.a.z.min(s.b.z) + profile(s, rho)
        }
        Shape::Line => {
            let (dx, dy, wx, wy) = (s.b.x - s.a.x, s.b.y - s.a.y, x - s.a.x, y - s.a.y);
            let a = dx * dx + dy * dy;
            if a < 1e-18 {
                return s.a.z.min(s.b.z) + profile(s, (wx * wx + wy * wy).sqrt());
            }
            if s.rc > 1e-9 || (s.b.z - s.a.z).abs() < 1e-9 {
                // Constant-height pass (or corner-radius tool, approximated at the closest point).
                let t = ((wx * dx + wy * dy) / a).clamp(0.0, 1.0);
                let (ex, ey) = (wx - t * dx, wy - t * dy);
                return s.a.z + t * (s.b.z - s.a.z) + profile(s, (ex * ex + ey * ey).sqrt());
            }
            // Flat tool on a ramp: the stretch of t where the tool covers (x,y) solves
            // |w - t d| <= R; z is linear in t, so its minimum is at an end.
            let b = -2.0 * (wx * dx + wy * dy);
            let c = wx * wx + wy * wy - s.r * s.r;
            let disc = b * b - 4.0 * a * c;
            if disc < 0.0 {
                return f64::INFINITY;
            }
            let sq = disc.sqrt();
            let t0 = ((-b - sq) / (2.0 * a)).max(0.0);
            let t1 = ((-b + sq) / (2.0 * a)).min(1.0);
            if t0 > t1 {
                return f64::INFINITY;
            }
            (s.a.z + t0 * (s.b.z - s.a.z)).min(s.a.z + t1 * (s.b.z - s.a.z))
        }
        Shape::Arc => arc_height(s, x, y),
    }
}

/// Arc (possibly helical): z is linear in the swept angle. The tool at angle theta
/// covers (x,y) when |Q - P(theta)| <= R, i.e. theta is within +/-alpha of Q's own
/// angle; the minimum z is at an end of that window clipped to the arc.
fn arc_height(s: &Sweep, x: f64, y: f64) -> f64 {
    let (qx, qy) = (x - s.c.x, y - s.c.y);
    let d = (qx * qx + qy * qy).sqrt();
    let ra = s.arc_r;
    let (span, dz, az) = (s.sweep.abs(), s.b.z - s.a.z, s.a.z);
    let z_at = |phi: f64| az + if span < 1e-12 { 0.0 } else { dz * phi / span };
    let helical = dz.abs() > 1e-9;

    if !helical || s.rc > 1e-9 {
        let (rho, phi) = closest_on_arc(s, qx, qy, d);
        return z_at(phi) + profile(s, rho);
    }
    // window of angles covering Q
    let (lo, hi);
    if d < 1e-12 {
        if ra > s.r {
            return f64::INFINITY;
        }
        lo = 0.0;
        hi = span;
    } else {
        let k = (d * d + ra * ra - s.r * s.r) / (2.0 * d * ra);
        if k > 1.0 {
            return f64::INFINITY;
        }
        if k <= -1.0 {
            lo = 0.0;
            hi = span;
        } else {
            let alpha = k.acos();
            let psi = qy.atan2(qx);
            // position of psi along the sweep direction, measured from theta0
            let mid = if s.sweep >= 0.0 { mod_2pi(psi - s.theta0) } else { mod_2pi(s.theta0 - psi) };
            lo = mid - alpha;
            hi = mid + alpha;
        }
    }
    let mut best = f64::INFINITY;
    // the window may wrap around; test it and its 2*pi shifts against [0, span]
    for w in -1..=1 {
        let a0 = (lo + w as f64 * TAU).max(0.0);
        let a1 = (hi + w as f64 * TAU).min(span);
        if a0 > a1 {
            continue;
        }
        best = best.min(z_at(a0).min(z_at(a1)));
    }
    best
}

/// XY distance from Q (relative to the center) to the arc, and the angle `phi` along
/// the sweep of the closest point.
fn closest_on_arc(s: &Sweep, qx: f64, qy: f64, d: f64) -> (f64, f64) {
    let span = s.sweep.abs();
    let psi = qy.atan2(qx);
    let along = if s.sweep >= 0.0 { mod_2pi(psi - s.theta0) } else { mod_2pi(s.theta0 - psi) };
    if along <= span {
        return ((d - s.arc_r).abs(), along);
    }
    // nearest end
    let (e0, e1) = (end_distance(s, qx, qy, 0.0), end_distance(s, qx, qy, span));
    (e0.min(e1), if e0 <= e1 { 0.0 } else { span })
}

fn end_distance(s: &Sweep, qx: f64, qy: f64, phi: f64) -> f64 {
    let th = s.theta0 + if s.sweep >= 0.0 { phi } else { -phi };
    let (ex, ey) = (qx - s.arc_r * th.cos(), qy - s.arc_r * th.sin());
    (ex * ex + ey * ey).sqrt()
}

/// If p is inside this sweep: how far to its nearest wall or floor, and a point just
/// past that exit (1e-4 mm out). Ramps and helices use the tool position closest to
/// p, so their depth is approximate (the caller's material check and fallback search
/// keep the result honest).
fn escape(s: &Sweep, p: V3) -> Option<(f64, V3)> {
    let (ax, ay, zt); // nearest tool axis position and its tip height
    match s.kind {
        Shape::Drill => {
            ax = s.a.x;
            ay = s.a.y;
            zt = s.a.z.min(s.b.z);
        }
        Shape::Line => {
            let (dx, dy, wx, wy) = (s.b.x - s.a.x, s.b.y - s.a.y, p.x - s.a.x, p.y - s.a.y);
            let a = dx * dx + dy * dy;
            let t = if a < 1e-18 { 0.0 } else { ((wx * dx + wy * dy) / a).clamp(0.0, 1.0) };
            ax = s.a.x + t * dx;
            ay = s.a.y + t * dy;
            zt = if a < 1e-18 { s.a.z.min(s.b.z) } else { s.a.z + t * (s.b.z - s.a.z) };
        }
        Shape::Arc => {
            let (qx, qy) = (p.x - s.c.x, p.y - s.c.y);
            let (_, phi) = closest_on_arc(s, qx, qy, (qx * qx + qy * qy).sqrt());
            let th = s.theta0 + if s.sweep >= 0.0 { phi } else { -phi };
            ax = s.c.x + s.arc_r * th.cos();
            ay = s.c.y + s.arc_r * th.sin();
            let span = s.sweep.abs();
            zt = s.a.z + if span < 1e-12 { 0.0 } else { (s.b.z - s.a.z) * phi / span };
        }
    }
    let (rx, ry) = (p.x - ax, p.y - ay);
    let rho = (rx * rx + ry * ry).sqrt();
    let bottom = zt + profile(s, rho.min(s.r));
    if rho > s.r || p.z < bottom {
        return None; // not inside
    }
    let (side, down) = (s.r - rho, p.z - bottom);
    if side <= down {
        let (ux, uy) = if rho > 1e-12 { (rx / rho, ry / rho) } else { (1.0, 0.0) };
        Some((side, V3::new(ax + ux * (s.r + 1e-4), ay + uy * (s.r + 1e-4), p.z)))
    } else {
        Some((down, V3::new(p.x, p.y, bottom - 1e-4)))
    }
}

/// Euclidean distance from p (outside it) to this sweep's solid. A pass at constant
/// height is a vertical prism above the tool bottom: the gap is
/// sqrt(horizontal^2 + vertical^2), exact for flat tools - this is what gets the
/// wall-meets-floor corners right. Ramps and helices are sampled every 0.25 mm along
/// the move.
fn distance_to(s: &Sweep, p: V3) -> f64 {
    match s.kind {
        Shape::Drill => {
            let rho = ((p.x - s.a.x) * (p.x - s.a.x) + (p.y - s.a.y) * (p.y - s.a.y)).sqrt();
            let zt = s.a.z.min(s.b.z);
            if rho <= s.r {
                return (zt + rho * s.cot - p.z).max(0.0);
            }
            let dz = (zt + s.r * s.cot - p.z).max(0.0);
            ((rho - s.r) * (rho - s.r) + dz * dz).sqrt()
        }
        Shape::Line if (s.b.z - s.a.z).abs() < 1e-9 => {
            let (dx, dy, wx, wy) = (s.b.x - s.a.x, s.b.y - s.a.y, p.x - s.a.x, p.y - s.a.y);
            let a = dx * dx + dy * dy;
            let t = if a < 1e-18 { 0.0 } else { ((wx * dx + wy * dy) / a).clamp(0.0, 1.0) };
            let (ex, ey) = (wx - t * dx, wy - t * dy);
            prism(s, (ex * ex + ey * ey).sqrt(), s.a.z, p.z)
        }
        Shape::Arc if (s.b.z - s.a.z).abs() < 1e-9 => {
            let (qx, qy) = (p.x - s.c.x, p.y - s.c.y);
            let (rho, _) = closest_on_arc(s, qx, qy, (qx * qx + qy * qy).sqrt());
            prism(s, rho, s.a.z, p.z)
        }
        _ => {
            // ramp / helix / plunge: sample the tool positions along the move
            let len = if s.kind == Shape::Arc {
                s.sweep.abs() * s.arc_r
            } else {
                ((s.b.x - s.a.x).powf(2.0) + (s.b.y - s.a.y).powf(2.0)).sqrt()
            };
            let n = ((len / 0.25).ceil() as i64).clamp(1, 400);
            let mut best = f64::INFINITY;
            for k in 0..=n {
                let f = k as f64 / n as f64;
                let z = s.a.z + f * (s.b.z - s.a.z);
                let (x, y) = if s.kind == Shape::Arc {
                    let th = s.theta0 + f * s.sweep;
                    (s.c.x + s.arc_r * th.cos(), s.c.y + s.arc_r * th.sin())
                } else {
                    (s.a.x + f * (s.b.x - s.a.x), s.a.y + f * (s.b.y - s.a.y))
                };
                let rho = ((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y)).sqrt();
                best = best.min(prism(s, rho, z, p.z));
            }
            best
        }
    }
}

/// Distance from a point at radial distance `rho` and height `pz` to a tool standing
/// with its tip at `zt` (the tool and everything above it is air).
fn prism(s: &Sweep, rho: f64, zt: f64, pz: f64) -> f64 {
    let bottom = zt + profile(s, rho.min(s.r));
    let dz = (bottom - pz).max(0.0);
    let dr = (rho - s.r).max(0.0);
    (dr * dr + dz * dz).sqrt()
}
