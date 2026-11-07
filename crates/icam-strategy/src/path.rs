//! What a strategy produces (tagged tool-center points at the slab) and the loop helpers
//! both path builders share: orientation, nearest point, crossing-parity inside, seams
//! and resampling. The entry is in `entry`.

pub use icam_post::{PathPt, Tag};

/// A point in the XY plane.
pub type P = (f64, f64);

/// A strategy's path and what it counted on the way.
#[derive(Clone, Debug, Default)]
pub struct PathResult {
    pub points: Vec<PathPt>,
    pub levels: Vec<f64>,
    pub notes: Vec<String>,
    pub trace: Vec<String>,
    pub wall_loop: Vec<P>,
    /// Path points the clearance pulled back onto the wall.
    pub clamped: usize,
    /// Feed moves that would have crossed the wall, replaced by a retract (extraction defects).
    pub safety_links: usize,
    pub islands: usize,
    pub branches: usize,
    pub seeds: usize,
    pub links: usize,
    pub dropped: usize,
    /// Max-cut: links kept at depth.
    pub kept_down: usize,
    /// Max-cut: open arcs.
    pub arcs: usize,
    /// Max-cut: morphing revolutions.
    pub revolutions2: usize,
    /// Max-cut: arc length cut conventional (zigzag).
    pub conventional_mm: f64,
}

impl PathResult {
    pub fn revolutions(&self) -> usize {
        self.levels.len().saturating_sub(1)
    }
}

pub fn area(l: &[P]) -> f64 {
    let mut a = 0.0;
    for i in 0..l.len() {
        let (p, q) = (l[i], l[(i + 1) % l.len()]);
        a += p.0 * q.1 - q.0 * p.1;
    }
    a / 2.0
}

/// Counterclockwise: with the material outside, climb milling with a right-hand (M03) tool.
pub fn orient(mut l: Vec<P>) -> Vec<P> {
    if area(&l) < 0.0 {
        l.reverse();
    }
    l
}

pub fn reverse(mut l: Vec<P>) -> Vec<P> {
    l.reverse();
    l
}

/// The first point of `l` nearest `p`.
pub fn nearest(l: &[P], p: P) -> P {
    let mut best = l[0];
    let mut bd = f64::INFINITY;
    for &q in l {
        let d = (q.0 - p.0) * (q.0 - p.0) + (q.1 - p.1) * (q.1 - p.1);
        if d < bd {
            bd = d;
            best = q;
        }
    }
    best
}

/// Crossing parity.
pub fn inside(l: &[P], x: f64, y: f64) -> bool {
    let mut c = false;
    let mut j = l.len().wrapping_sub(1);
    for i in 0..l.len() {
        if (l[i].1 > y) != (l[j].1 > y) && x < (l[j].0 - l[i].0) * (y - l[i].1) / (l[j].1 - l[i].1) + l[i].0 {
            c = !c;
        }
        j = i;
    }
    c
}

/// Rotate a closed loop to start at the vertex nearest p, and put p there.
pub fn at_seam(l: &[P], p: P) -> Vec<P> {
    let (mut best, mut bd) = (0, f64::MAX);
    for (i, q) in l.iter().enumerate() {
        let d = (q.0 - p.0) * (q.0 - p.0) + (q.1 - p.1) * (q.1 - p.1);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    let mut r = Vec::with_capacity(l.len());
    r.push(p);
    for i in 1..l.len() {
        r.push(l[(best + i) % l.len()]);
    }
    r
}

pub fn dist(a: P, b: P) -> f64 {
    ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)).sqrt()
}

/// Closed loop -> points every ds along it, starting at l[0], not repeating it.
pub fn resample(l: &[P], ds: f64) -> Vec<P> {
    let mut total = 0.0;
    for i in 0..l.len() {
        total += dist(l[i], l[(i + 1) % l.len()]);
    }
    let n = 8.max((total / ds).round() as i64) as usize;
    let step = total / n as f64;
    let mut r = Vec::with_capacity(n);
    let (mut seg, mut acc) = (0usize, 0.0);
    for k in 0..n {
        let target = k as f64 * step;
        while seg < l.len() - 1 && acc + dist(l[seg], l[seg + 1]) < target {
            acc += dist(l[seg], l[seg + 1]);
            seg += 1;
        }
        let (a, b) = (l[seg], l[(seg + 1) % l.len()]);
        let len = dist(a, b);
        let f = if len < 1e-12 { 0.0 } else { (target - acc) / len };
        r.push((a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f));
    }
    r
}

/// Open run of points -> points every <= ds along it, first point excluded, last included.
pub fn resample_open(l: &[P], ds: f64) -> Vec<P> {
    let mut r = Vec::new();
    for i in 1..l.len() {
        let n = 1.max((dist(l[i - 1], l[i]) / ds).ceil() as i64);
        for k in 1..=n {
            r.push((l[i - 1].0 + (l[i].0 - l[i - 1].0) * k as f64 / n as f64, l[i - 1].1 + (l[i].1 - l[i - 1].1) * k as f64 / n as f64));
        }
    }
    r
}
