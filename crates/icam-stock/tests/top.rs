//! StockTop against brute force: every move type and tool shape, read alone at points all
//! over and near its track, must give the least of the tool bottom's heights sampled
//! densely along the move (never above any sample, and below the samples only by what
//! lies between them).

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use icam_post::{Box2, Move, MoveKind, ToolDef, V3};
use icam_stock::StockTop;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn tool(number: i32, kind: &str, diameter: f64, corner: f64) -> ToolDef {
    ToolDef { number, kind: kind.into(), diameter, corner_radius: corner, description: String::new(), tip_angle_deg: 118.0 }
}

/// The bottom's height above the tip at `rho` from the axis, written out again here.
fn lift(t: &ToolDef, rho: f64) -> Option<f64> {
    let r = t.diameter / 2.0;
    if rho > r {
        return None;
    }
    if t.kind == "DRILL" {
        return Some(rho / (t.tip_angle_deg.to_radians() / 2.0).tan());
    }
    let rc = t.corner_radius.min(r);
    let u = rho - (r - rc);
    Some(if u > 0.0 { rc - (rc * rc - u * u).sqrt() } else { 0.0 })
}

fn mv(tool: i32, kind: MoveKind, from: V3, to: V3, center: V3) -> Move {
    Move { line: 0, op: String::new(), tool, kind, from, to, center, feed: 1000.0 }
}

/// An arc about `c` of radius `ra` from angle `th0` through `sweep`, from height z0 to z1.
fn arc(tool: i32, c: (f64, f64), ra: f64, th0: f64, sweep: f64, z0: f64, z1: f64) -> (Move, Vec<(f64, f64, f64)>) {
    let at = |a: f64| (c.0 + ra * a.cos(), c.1 + ra * a.sin());
    let (f, t) = (at(th0), at(th0 + sweep));
    let kind = if sweep >= 0.0 { MoveKind::ArcCcw } else { MoveKind::ArcCw };
    let m = mv(tool, kind, V3::new(f.0, f.1, z0), V3::new(t.0, t.1, z1), V3::new(c.0, c.1, z0));
    let n = 20000;
    let pts = (0..=n).map(|k| k as f64 / n as f64).map(|s| (at(th0 + s * sweep).0, at(th0 + s * sweep).1, z0 + s * (z1 - z0))).collect();
    (m, pts)
}

fn line(tool: i32, a: V3, b: V3) -> (Move, Vec<(f64, f64, f64)>) {
    let n = 20000;
    let pts = (0..=n).map(|k| k as f64 / n as f64).map(|s| (a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), a.z + s * (b.z - a.z))).collect();
    (mv(tool, MoveKind::Feed, a, b, V3::default()), pts)
}

#[test]
fn stock_top_is_the_least_reach_of_every_tool_on_every_move() {
    let tools: BTreeMap<i32, ToolDef> = [tool(1, "ENDMILL", 12.7, 0.0), tool(2, "BALL", 10.0, 5.0), tool(3, "ENDMILL", 12.0, 2.0), tool(4, "DRILL", 8.0, 0.0)].into_iter().map(|t| (t.number, t)).collect();
    let top = 100.0;
    let region = Box2::new(-60.0, -60.0, 60.0, 60.0);
    let mut rng = Lcg(11);
    let (mut reached, mut worst_below) = (0usize, 0.0f64);
    for t in 1..=4 {
        let mut cases = vec![
            line(t, V3::new(-30.0, -10.0, 5.0), V3::new(25.0, 12.0, 5.0)),
            line(t, V3::new(-30.0, 10.0, 12.0), V3::new(20.0, -15.0, 2.0)),
            line(t, V3::new(5.0, 5.0, 20.0), V3::new(5.0, 5.0, 3.0)),
            line(t, V3::new(-20.0, 0.0, 9.0), V3::new(-20.0 + 1e-3, 0.0, 1.0)),
            arc(t, (3.0, -4.0), 18.0, 0.4, 2.2, 6.0, 6.0),
            arc(t, (3.0, -4.0), 15.0, 2.5, -4.0, 10.0, 2.0),
            arc(t, (-5.0, 2.0), 9.0, 1.0, TAU, 4.0, 4.0),
            arc(t, (0.0, 0.0), 4.0, -1.0, 1.5 * PI, 8.0, 3.0),
            arc(t, (10.0, 10.0), 2.0, 0.0, 5.5, 7.0, 1.0),
        ];
        let d = V3::new(-8.0, 7.0, 2.5);
        cases.push((mv(t, MoveKind::Drill, V3::new(d.x, d.y, 30.0), d, V3::default()), vec![(d.x, d.y, d.z)]));
        let td = &tools[&t];
        for (m, pts) in &cases {
            let st = StockTop::new(std::slice::from_ref(m), 1, &tools, top, region);
            // Points all over, and near the track where the tool's edge passes.
            let mut probes: Vec<(f64, f64)> = (0..200).map(|_| (-50.0 + 100.0 * rng.next(), -50.0 + 100.0 * rng.next())).collect();
            for _ in 0..600 {
                let p = pts[(rng.next() * pts.len() as f64) as usize % pts.len()];
                let (a, rr) = (rng.next() * TAU, rng.next() * 1.1 * td.diameter / 2.0);
                probes.push((p.0 + rr * a.cos(), p.1 + rr * a.sin()));
            }
            for (x, y) in probes {
                let brute = pts.iter().filter_map(|p| lift(td, (x - p.0).hypot(y - p.1)).map(|l| p.2 + l)).fold(top, f64::min);
                let got = st.at(x, y);
                assert!(got <= brute + 1e-9 + StockTop::TOL, "tool {t} {:?}: {got} above a sampled reach {brute} at ({x}, {y})", m.kind);
                assert!(brute - got <= 2e-3, "tool {t} {:?}: {got} too far below the samples {brute} at ({x}, {y})", m.kind);
                if brute < top {
                    reached += 1;
                    worst_below = worst_below.max(brute - got);
                }
            }
        }
    }
    assert!(reached > 12000, "only {reached} probes were reached by a tool");
    println!("{reached} reached probes; samples at most {worst_below:.2e} mm above the exact least");
}

/// A plan reads its nodes out to the region's far edges; a region 30 mm wide is ten whole
/// bucket cells, so its far edge lies exactly past the last one (found on a real nest).
#[test]
fn stock_top_reads_the_region_edges() {
    let tools: BTreeMap<i32, ToolDef> = [(1, tool(1, "ENDMILL", 12.7, 0.0))].into_iter().collect();
    let (m, _) = line(1, V3::new(-10.0, 15.0, 2.0), V3::new(40.0, 15.0, 2.0));
    let st = StockTop::new(&[m], 1, &tools, 20.0, Box2::new(0.0, 0.0, 30.0, 30.0));
    for (x, y) in [(30.0, 15.0), (0.0, 15.0), (30.0, 20.0), (15.0, 21.0)] {
        assert_eq!(st.at(x, y), 2.0, "({x}, {y})");
    }
}
