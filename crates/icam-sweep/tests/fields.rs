//! The fields against brute force, on machines of every arrangement: the tool's distance
//! against its boundary sampled densely, the rate bounds against finite differences, a
//! move's distance against its least over a dense sweep of t, the machine chains placing
//! the tool where they should, and a 3-axis program's stock against the independent
//! checker (icam_stock::ImplicitStock).

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use icam_core::tool::Tool;
use icam_post::ToolDef;
use icam_sweep::{scale, Aabb, Chain, Helix, Joint, Linear, Link, Machine, Motion, Pose, Rot, Stock, Sweep, ToolSolid, V3};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }

    fn in_box(&mut self, b: Aabb) -> V3 {
        V3::new(self.range(b.min.x, b.max.x), self.range(b.min.y, b.max.y), self.range(b.min.z, b.max.z))
    }
}

fn tools() -> Vec<ToolSolid> {
    vec![
        ToolSolid::new(Tool::flat(10.0, 40.0), 30.0),
        ToolSolid::new(Tool::ball(8.0, 40.0), 30.0),
        ToolSolid::new(Tool::bull(12.0, 2.0, 40.0), 25.0),
        ToolSolid::new(Tool::vee(6.0, 118.0, 40.0), 20.0),
    ]
}

/// Distance in the profile's half-plane to the solid's boundary, from points sampled
/// densely along it (by height and by radius, so steep and flat stretches are both fine).
fn brute_tool(t: &ToolSolid, q: V3) -> f64 {
    let p = (q.x.hypot(q.y), q.z);
    let mut pts = Vec::new();
    let n = 20000;
    for i in 0..=n {
        let h = t.length * i as f64 / n as f64;
        pts.push((t.profile.radius_at(h).0, h));
        let r = t.radius() * i as f64 / n as f64;
        if let Some(m) = t.profile.h_min(r) {
            if m.h <= t.length {
                pts.push((r, m.h));
            }
        }
        pts.push((r, t.length));
        pts.push((t.profile.radius_at(0.0).0 * i as f64 / n as f64, 0.0));
    }
    let d = pts.iter().map(|&(a, b)| (p.0 - a).hypot(p.1 - b)).fold(f64::INFINITY, f64::min);
    let inside = p.1 >= 0.0 && p.1 <= t.length && p.0 <= t.profile.radius_at(p.1).0;
    if inside {
        -d
    } else {
        d
    }
}

#[test]
fn the_tool_distance_is_exact() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for t in tools() {
        let b = Aabb::new(V3::new(-12.0, -12.0, -6.0), V3::new(12.0, 12.0, t.length + 6.0));
        let mut worst = 0.0f64;
        for _ in 0..1000 {
            let q = rng.in_box(b);
            worst = worst.max((t.sd(q) - brute_tool(&t, q)).abs());
        }
        assert!(worst < 3e-3, "{}: worst {worst}", t.profile.name);
    }
}

fn head_head() -> Arc<dyn Machine> {
    let (x, y, z) = (V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0));
    Arc::new(Chain {
        links: vec![
            Link::Slide { axis: x, q: 0 },
            Link::Slide { axis: y, q: 1 },
            Link::Slide { axis: z, q: 2 },
            Link::Turn { axis: z, q: 3 },
            Link::Fixed(Pose::translation(V3::new(0.0, 0.0, -30.0))),
            Link::Turn { axis: y, q: 4 },
            Link::Fixed(Pose::translation(V3::new(0.0, 0.0, -120.0))),
        ],
        axes: 5,
    })
}

fn table_table() -> Arc<dyn Machine> {
    let (x, y, z) = (V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0));
    Arc::new(Chain {
        links: vec![
            Link::Turn { axis: V3::new(0.0, 0.0, -1.0), q: 4 },
            Link::Fixed(Pose::translation(V3::new(0.0, 0.0, -60.0))),
            Link::Turn { axis: V3::new(-1.0, 0.0, 0.0), q: 3 },
            Link::Fixed(Pose::translation(V3::new(0.0, 0.0, 60.0))),
            Link::Slide { axis: x, q: 0 },
            Link::Slide { axis: y, q: 1 },
            Link::Slide { axis: z, q: 2 },
        ],
        axes: 5,
    })
}

fn r_theta() -> Arc<dyn Machine> {
    Arc::new(Chain { links: vec![Link::Turn { axis: V3::new(0.0, 0.0, 1.0), q: 0 }, Link::Slide { axis: V3::new(1.0, 0.0, 0.0), q: 1 }, Link::Slide { axis: V3::new(0.0, 0.0, 1.0), q: 2 }], axes: 3 })
}

fn motions() -> Vec<(&'static str, Arc<dyn Motion>)> {
    vec![
        (
            "linear, turning",
            Arc::new(Linear { from: Pose::tip_axis(V3::new(0.0, 0.0, 5.0), V3::new(0.3, 0.0, 1.0)), to: Pose::tip_axis(V3::new(40.0, 10.0, -5.0), V3::new(-0.2, 0.5, 1.0)) }),
        ),
        ("helix", Arc::new(Helix { center: V3::new(5.0, 5.0, 0.0), normal: V3::new(0.0, 0.3, 1.0), start: V3::new(25.0, 5.0, 0.0), sweep: 1.5 * PI, rise: -8.0, rot: Rot::z_to(V3::new(0.1, 0.0, 1.0)) })),
        ("head-head", Arc::new(Joint { machine: head_head(), from: vec![10.0, 0.0, 150.0, 0.0, 0.2], to: vec![30.0, 15.0, 140.0, 1.2, -0.4] })),
        ("table-table", Arc::new(Joint { machine: table_table(), from: vec![0.0, 0.0, 20.0, 0.1, 0.0], to: vec![25.0, -10.0, 10.0, 0.6, 2.0] })),
        ("r-theta", Arc::new(Joint { machine: r_theta(), from: vec![0.0, 30.0, 5.0], to: vec![2.5, 45.0, -3.0] })),
    ]
}

#[test]
fn the_rates_bound_the_motion() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for (name, m) in motions() {
        for _ in 0..50 {
            let (a, b) = {
                let (u, v) = (rng.next(), rng.next());
                (u.min(v), u.max(v).max(u.min(v) + 1e-3))
            };
            let (v, w) = m.rate(a, b);
            let h = 1e-6;
            for k in 0..100 {
                let t = a + (b - a - h) * k as f64 / 99.0;
                let (p0, p1) = (m.pose(t), m.pose(t + h));
                let speed = (p1.at - p0.at).len() / h;
                let (a0, a1) = (p0.axis(), p1.axis());
                let turn = icam_sweep::cross(a0, a1).len().atan2(icam_sweep::dot(a0, a1)) / h;
                assert!(speed <= v * (1.0 + 1e-4) + 1e-6, "{name}: speed {speed} over the bound {v} on [{a}, {b}]");
                assert!(turn <= w * (1.0 + 1e-4) + 1e-6, "{name}: turning {turn} over the bound {w} on [{a}, {b}]");
            }
        }
    }
}

#[test]
fn the_second_derivatives_bound_the_motion() {
    let mut rng = Rng(0x7f4a_7c15_9e37_79b9);
    for (name, m) in motions() {
        for _ in 0..50 {
            let (a, b) = {
                let (u, v) = (rng.next(), rng.next());
                (u.min(v), u.max(v).max(u.min(v) + 1e-2))
            };
            let (ca, aa) = m.accel(a, b);
            let h = 1e-4;
            for k in 0..60 {
                let t = a + h + (b - a - 2.0 * h) * k as f64 / 59.0;
                let (p0, p1, p2) = (m.pose(t - h), m.pose(t), m.pose(t + h));
                let c2 = (p2.at - scale(p1.at, 2.0) + p0.at).len() / (h * h);
                let a2 = (p2.axis() - scale(p1.axis(), 2.0) + p0.axis()).len() / (h * h);
                assert!(c2 <= ca * (1.0 + 1e-3) + 1e-3, "{name}: origin acceleration {c2} over the bound {ca} on [{a}, {b}]");
                assert!(a2 <= aa * (1.0 + 1e-3) + 1e-3, "{name}: axis acceleration {a2} over the bound {aa} on [{a}, {b}]");
            }
        }
    }
}

#[test]
fn the_tool_support_is_its_farthest_point() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for t in tools() {
        let pose = Pose::tip_axis(V3::new(3.0, -2.0, 1.0), V3::new(0.4, -0.3, 1.0));
        // Points on the solid's boundary, densely.
        let mut pts = Vec::new();
        for i in 0..=400 {
            let h = t.length * i as f64 / 400.0;
            let r = t.profile.radius_at(h).0;
            for k in 0..72 {
                let a = k as f64 * PI / 36.0;
                pts.push(pose.apply(V3::new(r * a.cos(), r * a.sin(), h)));
            }
        }
        for _ in 0..200 {
            let d = icam_sweep::unit(V3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)));
            let ours = icam_sweep::dot(t.support(pose, d), d);
            let brute = pts.iter().map(|&q| icam_sweep::dot(q, d)).fold(f64::NEG_INFINITY, f64::max);
            assert!(ours >= brute - 1e-9 && ours <= brute + 0.05, "{}: support {ours} against {brute}", t.profile.name);
        }
    }
}

#[test]
fn a_move_distance_is_the_least_over_the_move() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let ts = tools();
    let n = 20000;
    for (name, m) in motions() {
        for tool in &ts {
            let s = Sweep::new(Arc::new(tool.clone()), m.clone());
            let poses: Vec<Pose> = (0..=n).map(|i| m.pose(i as f64 / n as f64)).collect();
            for _ in 0..60 {
                let p = rng.in_box(s.bounds().grow(5.0));
                let ours = s.sd(p, 1e-6);
                let brute = poses.iter().map(|q| tool.sd(q.to_local(p))).fold(f64::INFINITY, f64::min);
                let (v, w) = m.rate(0.0, 1.0);
                let far = poses.iter().map(|q| (p - q.at).len()).fold(0.0, f64::max);
                let slack = (v + w * far) * 0.5 / n as f64;
                assert!(ours <= brute + 1e-6, "{name}, {}: {ours} above the sampled least {brute}", tool.profile.name);
                assert!(brute - ours <= slack + 1e-9, "{name}, {}: {ours} more than {slack} below the sampled least {brute}", tool.profile.name);
            }
            // Every point of the tool at any pose is inside the move's box.
            for i in 0..=200 {
                let pose = m.pose(i as f64 / 200.0);
                for k in 0..16 {
                    let a = k as f64 * PI / 8.0;
                    for h in [0.0, 0.5 * tool.length, tool.length] {
                        let r = tool.profile.radius_at(h).0.min(tool.radius());
                        let q = pose.apply(V3::new(r * a.cos(), r * a.sin(), h));
                        assert!(s.bounds().dist(q) < 1e-9, "{name}: a tool point is outside the move's box");
                    }
                }
            }
        }
    }
}

#[test]
fn machines_place_the_tool() {
    let close = |a: V3, b: V3| (a - b).len() < 1e-9;
    let p = r_theta().pose(&[FRAC_PI_2, 10.0, 5.0]);
    assert!(close(p.at, V3::new(0.0, 10.0, 5.0)) && close(p.axis(), V3::new(0.0, 0.0, 1.0)), "{p:?}");
    let p = head_head().pose(&[1.0, 2.0, 200.0, 0.0, FRAC_PI_2]);
    assert!(close(p.at, V3::new(1.0 - 120.0, 2.0, 170.0)) && close(p.axis(), V3::new(1.0, 0.0, 0.0)), "{p:?}");
    // Turning the table a quarter turn turns the work under the tool the other way.
    let a = table_table().pose(&[10.0, 0.0, 0.0, 0.0, 0.0]);
    let b = table_table().pose(&[10.0, 0.0, 0.0, 0.0, FRAC_PI_2]);
    assert!(close(a.at, V3::new(10.0, 0.0, 0.0)) && close(b.at, V3::new(0.0, -10.0, 0.0)), "{a:?} {b:?}");
}

/// Flat end mills and a drill, where the checker's heights are exact: lines, a ramp,
/// an arc and a helix, a canned drill.
const EXACT: &str = "
G21 G90 G17
T1 M06
G00 X10 Y10 Z25
G01 Z15 F600
G01 X60 Y20 Z15
G01 X80 Y60 Z10
G02 X60 Y80 I-20 J0
G03 X40 Y60 Z6 I0 J-20
G00 Z25
T3 M06
G00 X85 Y15 Z25
G81 X85 Y15 Z8 R18 F200
G80
G00 Z25
";

/// A bull nose and a ball on ramps: the checker takes the tool's height where the point
/// projects onto the path, which is not where the tool reaches lowest on a slope.
const RAMPS: &str = "
G21 G90 G17
T2 M06
G00 X20 Y80 Z25
G01 Z12 F600
G01 X40 Y90 Z12
G01 X50 Y70 Z9
G00 Z25
T4 M06
G00 X20 Y20 Z25
G01 Z14 F600
G01 X60 Y40 Z8
G00 Z25
";

fn tool_def(number: i32, kind: &str, diameter: f64, corner: f64) -> ToolDef {
    ToolDef { number, kind: kind.into(), diameter, corner_radius: corner, description: String::new(), tip_angle_deg: 118.0 }
}

fn defs() -> BTreeMap<i32, ToolDef> {
    BTreeMap::from([(1, tool_def(1, "ENDMILL", 10.0, 0.0)), (2, tool_def(2, "ENDMILL", 10.0, 2.0)), (3, tool_def(3, "DRILL", 6.0, 0.0)), (4, tool_def(4, "ENDMILL", 8.0, 4.0))])
}

/// The stock `program` leaves in the box, the tools a kilometre long (the checker's
/// shank runs up forever).
fn stock_of(program: &str, lo: V3, hi: V3) -> (Stock, icam_stock::ImplicitStock) {
    let mut warnings = Vec::new();
    let moves = icam_post::parse(program, &mut warnings);
    let checker = icam_stock::ImplicitStock::new(lo, hi, &moves, &defs(), icam_stock::ImplicitStock::CELL_MM);
    let mut stock = Stock::new(Aabb::new(lo, hi));
    for s in icam_sweep::program::sweeps(&moves, &defs(), 1000.0, hi.z) {
        stock.apply(s);
    }
    (stock, checker)
}

/// Distance to air at a material point by brute force: the box's faces, and every move's
/// least over a dense sweep of t.
fn brute_air(stock: &Stock, p: V3) -> f64 {
    let n = 200_000;
    (0..stock.len())
        .map(|k| {
            let s = stock.sweep(k);
            (0..=n).map(|i| s.tool.sd(s.motion.pose(i as f64 / n as f64).to_local(p))).fold(f64::INFINITY, f64::min)
        })
        .fold(-stock.bounds.sd(p), f64::min)
}

#[test]
fn a_three_axis_stock_agrees_with_the_checker() {
    let (lo, hi) = (V3::new(0.0, 0.0, 0.0), V3::new(100.0, 100.0, 20.0));
    let (stock, checker) = stock_of(EXACT, lo, hi);
    assert!(stock.len() > 5);
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    let (mut cut, mut same, mut theirs_off, mut worst_theirs) = (0, 0, 0, 0.0f64);
    for _ in 0..20000 {
        let p = rng.in_box(Aabb::new(lo, hi));
        let f = stock.sd(p, 1e-7);
        if f.abs() < 1e-6 {
            continue;
        }
        let ours = f < 0.0;
        assert_eq!(ours, checker.is_material(p), "at {p:?}: field {f}");
        if !ours {
            cut += 1;
            continue;
        }
        // The checker's distance to air leaves out the table under the sheet.
        let theirs = checker.deviation(p).min(p.z - lo.z);
        if (-f - theirs).abs() <= 1e-6 {
            same += 1;
            continue;
        }
        // Where the two differ, brute force decides (on the first 50); ours must be right.
        theirs_off += 1;
        if theirs_off <= 50 {
            let brute = brute_air(&stock, p);
            assert!((-f - brute).abs() < 1e-4, "at {p:?}: distance to air {} against {brute} by brute force (the checker says {theirs})", -f);
            worst_theirs = worst_theirs.max((theirs - brute).abs());
        }
    }
    assert!(cut > 500, "the program cut too little to test: {cut}");
    println!("distance to air: {same} points agree with the checker; at {theirs_off} it differs, by up to {worst_theirs:.4} mm at the 50 brute force judged (each time for the field)");
}

#[test]
fn ramps_of_round_tools_cut_where_the_tool_goes() {
    let (lo, hi) = (V3::new(0.0, 0.0, 0.0), V3::new(100.0, 100.0, 20.0));
    let (stock, checker) = stock_of(RAMPS, lo, hi);
    let n = 5000;
    let poses: Vec<Vec<Pose>> = (0..stock.len()).map(|k| (0..=n).map(|i| stock.sweep(k).motion.pose(i as f64 / n as f64)).collect()).collect();
    let mut rng = Rng(0x0ddb_a110_ddba_1100);
    let (mut checked, mut missed, mut deepest) = (0, 0, 0.0f64);
    for _ in 0..2000 {
        // Points near the ramps' floors.
        let p = V3::new(rng.range(10.0, 70.0), rng.range(10.0, 100.0), rng.range(6.0, 15.0));
        let f = stock.sd(p, 1e-9);
        if f.abs() < 1e-4 {
            continue;
        }
        // The least over this sweep of t is within 0.005 mm here (moves under 50 mm).
        let brute = (0..stock.len())
            .map(|k| poses[k].iter().map(|q| stock.sweep(k).tool.sd(q.to_local(p))).fold(f64::INFINITY, f64::min))
            .fold(f64::INFINITY, f64::min);
        if brute.abs() < 0.01 {
            continue;
        }
        checked += 1;
        assert_eq!(f < 0.0, brute > 0.0, "at {p:?}: field {f}, the least over the moves {brute}");
        if f > 0.0 && checker.is_material(p) {
            missed += 1;
            deepest = deepest.max(f);
        }
    }
    assert!(checked > 500);
    println!("the checker keeps {missed} of {checked} points the round tools cut through, up to {deepest:.4} mm inside the tool");
}
