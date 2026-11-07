//! The clearance read off the compiled F-rep (SliceClearance) against the exact polygon
//! clearance (PolygonClearance) on the same part: an island pocket, built as a prismatic
//! solid (a sheet
//! with a pocket down to z 3 and an island back up to the top) and compiled to its F-rep.

use icam_core::frep::compile_prismatic;
use icam_core::geom::{Curve, V2};
use icam_core::part::Part;
use icam_query::{Clearance, PolygonClearance, SliceClearance};

fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Curve> {
    let p = [V2::new(x0, y0), V2::new(x1, y0), V2::new(x1, y1), V2::new(x0, y1)];
    (0..4).map(|i| Curve::Line { a: p[i], b: p[(i + 1) % 4] }).collect()
}

fn reversed(l: Vec<Curve>) -> Vec<Curve> {
    l.into_iter()
        .rev()
        .map(|c| match c {
            Curve::Line { a, b } => Curve::Line { a: b, b: a },
            other => other,
        })
        .collect()
}

/// Sheet (-30,-30)-(120,90), 19.05 thick; pocket (0,0)-(90,60) down to z 3; island
/// (40,22)-(55,38) back up to the top.
fn island_part() -> Part {
    let mut p = Part::default();
    p.add_face(1, 19.05, vec![square(-30.0, -30.0, 120.0, 90.0), reversed(square(0.0, 0.0, 90.0, 60.0))], None, 1e-9).unwrap();
    for (i, e) in p.edges.iter_mut().enumerate() {
        e.wall_zlo = if i < 4 { 0.0 } else { 3.0 };
    }
    p.faces[0].up = Some(true);
    let n = p.edges.len();
    p.add_face(2, 19.05, vec![square(40.0, 22.0, 55.0, 38.0)], None, 1e-9).unwrap();
    for e in &mut p.edges[n..] {
        e.wall_zlo = 3.0;
    }
    p.faces[1].up = Some(true);
    let n = p.edges.len();
    p.add_face(3, 3.0, vec![square(0.0, 0.0, 90.0, 60.0), reversed(square(40.0, 22.0, 55.0, 38.0))], None, 1e-9).unwrap();
    for e in &mut p.edges[n..] {
        e.wall_zlo = 3.0;
    }
    p.faces[2].up = Some(true);
    p.add_face(4, 0.0, vec![reversed(square(-30.0, -30.0, 120.0, 90.0))], None, 1e-9).unwrap();
    p.faces[3].up = Some(false);
    p
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

fn polygons() -> PolygonClearance {
    PolygonClearance::new(vec![vec![rect(-30.0, -30.0, 120.0, 90.0), rect(0.0, 0.0, 90.0, 60.0)], vec![rect(40.0, 22.0, 55.0, 38.0)]], f64::INFINITY)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn slice_clearance_is_the_exact_polygon_distance() {
    let part = island_part();
    let frep = compile_prismatic(&part, 1e-9).unwrap();
    let slab = SliceClearance::new(&frep, 6.0, 1e-3);
    assert_eq!(slab.column_count(), 1, "only the top column stands above the slab; the pocket floor does not");
    let poly = polygons();
    let (mut worst, mut n) = (0.0f64, 0);
    let mut y = -35.0;
    while y <= 95.0 {
        let mut x = -35.0;
        while x <= 125.0 {
            let (a, b) = (slab.at(x, y), poly.at(x, y));
            worst = worst.max((a - b).abs());
            if b.abs() > 1e-9 {
                assert_eq!(a < 0.0, b < 0.0, "sign at ({x}, {y}): field {a}, polygons {b}");
            }
            n += 1;
            x += 0.7;
        }
        y += 0.7;
    }
    println!("{n} points, worst difference {worst:e} mm");
    assert!(worst < 1e-12, "the field's clearance is off the exact distance by {worst:e} mm");

    // Below the floor the whole sheet is a wall; above the part nothing is.
    assert!(SliceClearance::new(&frep, 2.0, 1e-3).at(45.0, 10.0) < 0.0);
    assert_eq!(SliceClearance::new(&frep, 20.0, 1e-3).column_count(), 0);
}

#[test]
fn clear_along_agrees_with_the_exact_segment_test() {
    let part = island_part();
    let frep = compile_prismatic(&part, 1e-9).unwrap();
    let slab = SliceClearance::new(&frep, 6.0, 1e-3);
    let poly = polygons();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut agree, mut near) = (0, 0);
    for _ in 0..4000 {
        let (ax, ay) = (rng.next() * 100.0 - 5.0, rng.next() * 70.0 - 5.0);
        let (bx, by) = (ax + rng.next() * 30.0 - 15.0, ay + rng.next() * 30.0 - 15.0);
        let need = rng.next() * 15.0;
        let (a, b) = (slab.clear_along(ax, ay, bx, by, need), poly.clear_along(ax, ay, bx, by, need));
        if a == b {
            agree += 1;
            continue;
        }
        // A disagreement is allowed only where the segment's clearance touches `need`.
        let mut least = f64::INFINITY;
        for k in 0..=20000 {
            let t = k as f64 / 20000.0;
            least = least.min(poly.at(ax + (bx - ax) * t, ay + (by - ay) * t));
        }
        assert!((least - need).abs() < 1e-3, "segment ({ax}, {ay})-({bx}, {by}) need {need}: field {a}, polygons {b}, least {least}");
        near += 1;
    }
    println!("{agree} agree, {near} disagree at the limit");
    assert!(near <= 2);
}
