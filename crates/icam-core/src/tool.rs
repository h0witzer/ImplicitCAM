//! Cutters as exact profiles of revolution.
//!
//! A tool is a chain of profile elements in the (r, h) half-plane, starting on
//! the axis at the tip (r = 0, h = 0) and running outward and upward. Radius
//! and height never decrease along the chain, which covers flat, ball, bull,
//! V, chamfer and tapered cutters plus a stepped shank and holder. The last
//! element's top radius continues upward forever (the holder column), so a
//! tool sitting above a point never leaves it uncovered.

#[derive(Clone, Copy, Debug)]
pub enum ProfEl {
    /// Straight segment from (r0, h0) to (r1, h1).
    Seg { r0: f64, h0: f64, r1: f64, h1: f64 },
    /// Lower-right quarter of a circle about (rc, hc) of radius `rad`: from the
    /// bottom point (rc, hc - rad) to the side point (rc + rad, hc). Convex, as on
    /// ball and bull-nose cutters.
    Arc { rc: f64, hc: f64, rad: f64 },
}

impl ProfEl {
    pub fn start(&self) -> (f64, f64) {
        match *self {
            ProfEl::Seg { r0, h0, .. } => (r0, h0),
            ProfEl::Arc { rc, hc, rad } => (rc, hc - rad),
        }
    }
    pub fn end(&self) -> (f64, f64) {
        match *self {
            ProfEl::Seg { r1, h1, .. } => (r1, h1),
            ProfEl::Arc { rc, hc, rad } => (rc + rad, hc),
        }
    }
}

/// Lowest point of the tool at a given radius, with its derivatives in r.
#[derive(Clone, Copy, Debug)]
pub struct HMin {
    pub h: f64,
    pub dh: f64,
    pub d2h: f64,
    pub el: usize,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub els: Vec<ProfEl>,
}

const EPS: f64 = 1e-12;

impl Tool {
    pub fn new(name: &str, els: Vec<ProfEl>) -> Result<Tool, String> {
        let t = Tool { name: name.to_string(), els };
        t.validate()?;
        Ok(t)
    }

    /// Flat end mill of diameter `d`, flute length `len`.
    pub fn flat(d: f64, len: f64) -> Tool {
        let r = 0.5 * d;
        Tool::new(
            &format!("flat {d} mm"),
            vec![ProfEl::Seg { r0: 0.0, h0: 0.0, r1: r, h1: 0.0 }, ProfEl::Seg { r0: r, h0: 0.0, r1: r, h1: len }],
        )
        .unwrap()
    }

    /// Ball end mill of diameter `d`.
    pub fn ball(d: f64, len: f64) -> Tool {
        let r = 0.5 * d;
        Tool::new(
            &format!("ball {d} mm"),
            vec![ProfEl::Arc { rc: 0.0, hc: r, rad: r }, ProfEl::Seg { r0: r, h0: r, r1: r, h1: len.max(r) }],
        )
        .unwrap()
    }

    /// Bull-nose end mill of diameter `d` with corner radius `rc`.
    pub fn bull(d: f64, rc: f64, len: f64) -> Tool {
        let r = 0.5 * d;
        Tool::new(
            &format!("bull {d} mm r{rc}"),
            vec![
                ProfEl::Seg { r0: 0.0, h0: 0.0, r1: r - rc, h1: 0.0 },
                ProfEl::Arc { rc: r - rc, hc: rc, rad: rc },
                ProfEl::Seg { r0: r, h0: rc, r1: r, h1: len.max(rc) },
            ],
        )
        .unwrap()
    }

    /// V bit of diameter `d` and included angle `deg`, sharp tip.
    pub fn vee(d: f64, deg: f64, len: f64) -> Tool {
        let r = 0.5 * d;
        let hv = r / (0.5 * deg.to_radians()).tan();
        Tool::new(
            &format!("V {d} mm {deg} deg"),
            vec![ProfEl::Seg { r0: 0.0, h0: 0.0, r1: r, h1: hv }, ProfEl::Seg { r0: r, h0: hv, r1: r, h1: len.max(hv) }],
        )
        .unwrap()
    }

    /// Adds a holder of diameter `hd` whose face sits `stickout` above the tip.
    /// The cutter is trimmed or extended to that height.
    pub fn with_holder(mut self, stickout: f64, hd: f64) -> Tool {
        let (r_top, _) = self.els.last().unwrap().end();
        if let Some(ProfEl::Seg { r0, h0, r1, h1 }) = self.els.last_mut() {
            if *r0 == *r1 && *h0 <= stickout {
                *h1 = stickout;
            }
        }
        let hr = 0.5 * hd;
        self.els.push(ProfEl::Seg { r0: r_top, h0: stickout, r1: hr, h1: stickout });
        self.els.push(ProfEl::Seg { r0: hr, h0: stickout, r1: hr, h1: stickout + 100.0 });
        self.name = format!("{} + holder {hd} mm at {stickout} mm", self.name);
        self.validate().unwrap();
        self
    }

    fn validate(&self) -> Result<(), String> {
        let mut prev = (0.0, 0.0);
        for (i, e) in self.els.iter().enumerate() {
            let (s, t) = (e.start(), e.end());
            if (s.0 - prev.0).abs() > 1e-9 || (s.1 - prev.1).abs() > 1e-9 {
                return Err(format!("{}: element {i} does not start where the last ended", self.name));
            }
            if t.0 < s.0 - EPS || t.1 < s.1 - EPS {
                return Err(format!("{}: element {i} turns inward or downward", self.name));
            }
            if let ProfEl::Arc { rad, .. } = *e {
                if rad <= 0.0 {
                    return Err(format!("{}: element {i} has no radius", self.name));
                }
            }
            prev = t;
        }
        Ok(())
    }

    /// Widest radius the tool reaches (the holder column's, if there is one).
    pub fn r_max(&self) -> f64 {
        self.els.iter().map(|e| e.end().0).fold(0.0, f64::max)
    }

    /// Height of the tool's lowest point at radius `r` above the tip, or None if
    /// the tool never reaches that radius.
    pub fn h_min(&self, r: f64) -> Option<HMin> {
        let r = r.max(0.0);
        for (el, e) in self.els.iter().enumerate() {
            match *e {
                ProfEl::Seg { r0, h0, r1, h1 } => {
                    if r1 - r0 <= EPS || r > r1 + EPS {
                        continue;
                    }
                    let m = (h1 - h0) / (r1 - r0);
                    return Some(HMin { h: h0 + (r.max(r0) - r0) * m, dh: m, d2h: 0.0, el });
                }
                ProfEl::Arc { rc, hc, rad } => {
                    if r > rc + rad + EPS {
                        continue;
                    }
                    let x = (r - rc).clamp(0.0, rad);
                    let w = (rad * rad - x * x).max(1e-24).sqrt();
                    return Some(HMin { h: hc - w, dh: x / w, d2h: rad * rad / (w * w * w), el });
                }
            }
        }
        None
    }

    /// Tool radius at height `h` above the tip (the widest element there), and
    /// which element it is. Below the tip the tool is empty.
    pub fn radius_at(&self, h: f64) -> (f64, usize) {
        if h < 0.0 {
            return (0.0, usize::MAX);
        }
        let mut best = (-1.0, usize::MAX);
        for (el, e) in self.els.iter().enumerate() {
            let r = match *e {
                ProfEl::Seg { r0, h0, r1, h1 } => {
                    if h1 - h0 <= EPS {
                        if (h - h0).abs() <= 1e-9 {
                            r1.max(r0)
                        } else {
                            continue;
                        }
                    } else if h >= h0 - EPS && h <= h1 + EPS {
                        r0 + (h.clamp(h0, h1) - h0) * (r1 - r0) / (h1 - h0)
                    } else {
                        continue;
                    }
                }
                ProfEl::Arc { rc, hc, rad } => {
                    if h < hc - rad - EPS || h > hc + EPS {
                        continue;
                    }
                    let y = (hc - h).clamp(0.0, rad);
                    rc + (rad * rad - y * y).max(0.0).sqrt()
                }
            };
            if r > best.0 {
                best = (r, el);
            }
        }
        if best.1 == usize::MAX {
            // Above the last element: the top radius continues upward.
            let last = self.els.len() - 1;
            return (self.els[last].end().0, last);
        }
        best
    }
}
