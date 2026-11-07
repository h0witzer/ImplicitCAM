//! SweptStock read at an earlier sweep count must equal a stock that only ever held
//! those sweeps, at every point, for every count.

use std::f64::consts::TAU;

use icam_post::Box2;
use icam_stock::SweptStock;

/// A fixed sequence in [0, 1), so the case is the same on every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

enum Add {
    Segment(f64, f64, f64, f64, f64),
    Arc(f64, f64, f64, f64, f64, f64),
}

fn add(stock: &mut SweptStock, a: &Add) {
    match *a {
        Add::Segment(ax, ay, bx, by, r) => stock.add_segment(ax, ay, bx, by, r),
        Add::Arc(cx, cy, arc_r, th0, dth, r) => stock.add_arc(cx, cy, arc_r, th0, dth, r),
    }
}

#[test]
fn material_at_equals_a_stock_of_only_the_first_sweeps() {
    let region = Box2::new(0.0, 0.0, 100.0, 80.0);
    let sheet = Box2::new(-5.0, -5.0, 105.0, 85.0);
    let mut rng = Lcg(7);
    // A walk of short segments (a path at its sample step) with arcs and plunges mixed in.
    let (mut x, mut y) = (50.0, 40.0);
    let mut adds = Vec::new();
    for i in 0..600 {
        let r = 3.0 + 3.0 * rng.next();
        if i % 37 == 0 {
            adds.push(Add::Segment(x, y, x, y, r));
        } else if i % 11 == 0 {
            let dth = (rng.next() - 0.5) * 8.0;
            adds.push(Add::Arc(x, y, 2.0 + 10.0 * rng.next(), rng.next() * TAU, dth, r));
        } else {
            let (nx, ny) = ((x + (rng.next() - 0.5) * 6.0).clamp(0.0, 100.0), (y + (rng.next() - 0.5) * 6.0).clamp(0.0, 80.0));
            adds.push(Add::Segment(x, y, nx, ny, r));
            (x, y) = (nx, ny);
        }
    }
    let mut all = SweptStock::new(region, sheet);
    for a in &adds {
        add(&mut all, a);
    }
    let pts: Vec<(f64, f64)> = (0..4000).map(|_| (-8.0 + 116.0 * rng.next(), -8.0 + 96.0 * rng.next())).collect();
    let mut prefix = SweptStock::new(region, sheet);
    let (mut checked, mut changed) = (0, 0);
    for n in 0..=adds.len() {
        if n > 0 {
            add(&mut prefix, &adds[n - 1]);
        }
        if n % 7 != 0 && n != adds.len() {
            continue;
        }
        for &(px, py) in &pts {
            let want = prefix.material(px, py);
            assert_eq!(all.material_at(px, py, n), want, "n = {n} at ({px}, {py})");
            if want != all.material(px, py) {
                changed += 1;
            }
            checked += 1;
        }
    }
    // The case must exercise the count: many points change as sweeps are added.
    assert!(changed > checked / 10, "{changed} of {checked} points differ from the final stock");
    assert_eq!(all.material_at(1.0, 1.0, adds.len()), all.material(1.0, 1.0));
}
