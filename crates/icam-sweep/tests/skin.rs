//! The skin against its fields: a move's own surface lies on the move's boundary and
//! encloses what the move sweeps; a program's skin, built move by move, lies on the
//! final stock's zero set and encloses its material.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use icam_core::tool::Tool;
use icam_post::ToolDef;
use icam_sweep::envelope::{enclosed, envelope};
use icam_sweep::skin::trim;
use icam_sweep::gpu::Fields;
use icam_sweep::{Aabb, Chain, Helix, Joint, Linear, Link, Motion, Pose, Rot, Skin, Sweep, ToolSolid, V3};
use rayon::prelude::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn in_box(&mut self, b: Aabb) -> V3 {
        V3::new(b.min.x + (b.max.x - b.min.x) * self.next(), b.min.y + (b.max.y - b.min.y) * self.next(), b.min.z + (b.max.z - b.min.z) * self.next())
    }
}

/// The volume a straight move of a convex tool sweeps: the tool's own, plus the move's
/// length times the tool's area projected across the move (from its support points).
fn straight_volume(tool: &ToolSolid, d: V3) -> f64 {
    let n = 100_000;
    let own: f64 = (0..n)
        .map(|i| {
            let r = tool.profile.radius_at(tool.length * (i as f64 + 0.5) / n as f64).0;
            PI * r * r
        })
        .sum::<f64>()
        * tool.length
        / n as f64;
    let e = icam_sweep::unit(d);
    let a = icam_sweep::unit(icam_sweep::cross(e, if e.z.abs() < 0.9 { V3::new(0.0, 0.0, 1.0) } else { V3::new(1.0, 0.0, 0.0) }));
    let b = icam_sweep::cross(e, a);
    let k = 20_000;
    let pts: Vec<(f64, f64)> = (0..k)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / k as f64;
            let p = tool.support(Pose::IDENTITY, icam_sweep::scale(a, t.cos()) + icam_sweep::scale(b, t.sin()));
            (icam_sweep::dot(p, a), icam_sweep::dot(p, b))
        })
        .collect();
    let area = (0..k).map(|i| pts[i].0 * pts[(i + 1) % k].1 - pts[(i + 1) % k].0 * pts[i].1).sum::<f64>().abs() / 2.0;
    own + d.len() * area
}

/// The volume a move sweeps, by sampling 64 boxes that each hold the tool over 1/64 of
/// the move (it fills them far better than one box round the whole move); a point counts
/// 1 / (the boxes holding it), so the estimate is fair where boxes overlap.
fn sampled_volume(s: &Sweep, n: usize) -> f64 {
    let k = 64;
    let boxes: Vec<Aabb> = (0..k)
        .map(|i| {
            let (a, b) = (i as f64 / k as f64, (i + 1) as f64 / k as f64);
            let (v, w) = s.motion.rate(a, b);
            s.tool.bounds(s.motion.pose(0.5 * (a + b))).grow((v + w * s.tool.reach()) * 0.5 * (b - a))
        })
        .collect();
    let total: f64 = boxes.iter().map(|&b| volume(b)).sum();
    let hits: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut pick = rng.next() * total;
            let mut b = boxes[k - 1];
            for &x in &boxes {
                if pick < volume(x) {
                    b = x;
                    break;
                }
                pick -= volume(x);
            }
            let p = rng.in_box(b);
            if s.sd(p, 1e-4) >= 0.0 {
                return 0.0;
            }
            1.0 / boxes.iter().filter(|x| x.dist(p) == 0.0).count() as f64
        })
        .sum();
    total * hits / n as f64
}

fn volume(b: Aabb) -> f64 {
    (b.max.x - b.min.x) * (b.max.y - b.min.y) * (b.max.z - b.min.z)
}

/// How the volume a move sweeps is known: exactly for a straight move (its displacement,
/// in the tool's own frame)
/// and for a level arc with the tool upright (its radius and sweep: at each height the
/// tool's disk sweeps an annular sector plus two half disks), else by sampling.
#[derive(Clone, Copy)]
enum Known {
    Straight(V3),
    LevelArc(f64, f64),
    Sampled,
}

fn arc_volume(tool: &ToolSolid, radius: f64, sweep: f64) -> f64 {
    let n = 100_000;
    (0..n)
        .map(|i| {
            let r = tool.profile.radius_at(tool.length * (i as f64 + 0.5) / n as f64).0;
            2.0 * sweep * radius * r + PI * r * r
        })
        .sum::<f64>()
        * tool.length
        / n as f64
}

/// The moves, and how the volume each sweeps is known.
fn motions() -> Vec<(&'static str, Arc<dyn Motion>, Known)> {
    let (x, y, z) = (V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0));
    let head = Arc::new(Chain {
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
    });
    let rt = Arc::new(Chain { links: vec![Link::Turn { axis: z, q: 0 }, Link::Slide { axis: x, q: 1 }, Link::Slide { axis: z, q: 2 }], axes: 3 });
    vec![
        ("level line", Arc::new(Linear { from: Pose::translation(V3::new(0.0, 0.0, 0.0)), to: Pose::translation(V3::new(40.0, 10.0, 0.0)) }), Known::Straight(V3::new(40.0, 10.0, 0.0))),
        ("ramp", Arc::new(Linear { from: Pose::translation(V3::new(0.0, 0.0, 5.0)), to: Pose::translation(V3::new(30.0, -10.0, -3.0)) }), Known::Straight(V3::new(30.0, -10.0, -8.0))),
        ("plunge", Arc::new(Linear { from: Pose::translation(V3::new(0.0, 0.0, 5.0)), to: Pose::translation(V3::new(0.0, 0.0, -7.0)) }), Known::Straight(V3::new(0.0, 0.0, -12.0))),
        // (The straight volume reads the move in the tool's own frame: along its axis.)
        ("tilted plunge", Arc::new(Linear { from: Pose::tip_axis(V3::new(0.0, 0.0, 0.0), V3::new(0.3, -0.2, 1.0)), to: Pose::tip_axis(icam_sweep::scale(icam_sweep::unit(V3::new(0.3, -0.2, 1.0)), -9.0), V3::new(0.3, -0.2, 1.0)) }), Known::Straight(V3::new(0.0, 0.0, -9.0))),
        ("level arc", Arc::new(Helix { center: V3::new(0.0, 0.0, 0.0), normal: z, start: V3::new(20.0, 0.0, 0.0), sweep: 1.25 * PI, rise: 0.0, rot: Rot::IDENTITY }), Known::LevelArc(20.0, 1.25 * PI)),
        ("helix", Arc::new(Helix { center: V3::new(0.0, 0.0, 0.0), normal: z, start: V3::new(20.0, 0.0, 0.0), sweep: 1.25 * PI, rise: -4.0, rot: Rot::IDENTITY }), Known::Sampled),
        ("linear, turning", Arc::new(Linear { from: Pose::tip_axis(V3::new(0.0, 0.0, 5.0), V3::new(0.3, 0.0, 1.0)), to: Pose::tip_axis(V3::new(40.0, 10.0, -5.0), V3::new(-0.2, 0.5, 1.0)) }), Known::Sampled),
        ("head-head", Arc::new(Joint { machine: head, from: vec![10.0, 0.0, 150.0, 0.0, 0.2], to: vec![30.0, 15.0, 140.0, 1.2, -0.4] }), Known::Sampled),
        ("r-theta", Arc::new(Joint { machine: rt, from: vec![0.0, 30.0, 5.0], to: vec![1.0, 40.0, -3.0] }), Known::Sampled),
    ]
}

#[test]
fn a_move_surface_is_its_boundary() {
    let tools = [ToolSolid::new(Tool::flat(10.0, 40.0), 30.0), ToolSolid::new(Tool::ball(8.0, 40.0), 30.0), ToolSolid::new(Tool::bull(12.0, 2.0, 40.0), 25.0)];
    for (name, m, known) in motions() {
        for tool in &tools {
            let s = Sweep::new(Arc::new(tool.clone()), m.clone());
            let fold = 1e-3;
            let own = trim(envelope(&s, 0.02, 2.0), &|p| s.sd(p, 1e-4) + fold);
            assert!(!own.is_empty(), "{name}, {}: nothing", tool.profile.name);
            let mut off = 0.0f64;
            for p in own.iter().step_by(10) {
                for &q in &p.p {
                    off = off.max(s.sd(q, 1e-4).abs());
                }
            }
            // Corners sit on tool surfaces (distance <= 0) and were kept off the fold.
            assert!(off < fold + 2e-4, "{name}, {}: a corner {off} mm off the move's boundary", tool.profile.name);
            // Against the exact volume where it is known (the surface's chords sit inside, so
            // a little under); else against sampling.
            let ours = enclosed(&own);
            let (want, how, allow) = match known {
                Known::Straight(d) => (straight_volume(tool, d), "exactly", 0.01),
                Known::LevelArc(r, a) => (arc_volume(tool, r, a), "exactly", 0.01),
                Known::Sampled => (sampled_volume(&s, 50_000), "by sampling", 0.015),
            };
            let err = (ours - want).abs() / want;
            if !matches!(known, Known::Sampled) {
                assert!(ours <= want * (1.0 + 1e-4), "{name}, {}: chords should sit inside, but encloses {ours:.1} over {want:.1}", tool.profile.name);
            }
            println!("{name:16} {:16}: {} pieces, volume {ours:.1} against {want:.1} {how} ({:.2} %)", tool.profile.name, own.len(), 100.0 * err);
            assert!(err < allow, "{name}, {}: encloses {ours:.1} mm3, the move sweeps {want:.1}", tool.profile.name);
        }
    }
}

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

/// The test program's skin, its fields read on the GPU when `gpu` (None when there is
/// none), and the seconds it took.
fn program_skin(gpu: bool) -> Option<(Skin, f64)> {
    let tool = |n: i32, kind: &str, d: f64| ToolDef { number: n, kind: kind.into(), diameter: d, corner_radius: 0.0, description: String::new(), tip_angle_deg: 118.0 };
    let defs = BTreeMap::from([(1, tool(1, "ENDMILL", 10.0)), (3, tool(3, "DRILL", 6.0))]);
    let mut warnings = Vec::new();
    let moves = icam_post::parse(EXACT, &mut warnings);
    let b = Aabb::new(V3::new(0.0, 0.0, 0.0), V3::new(100.0, 100.0, 20.0));
    let sweeps = icam_sweep::program::sweeps(&moves, &defs, 40.0, b.max.z);
    let mut skin = Skin::new(b, 1e-5, 0.4);
    if gpu {
        let mut tools: Vec<Arc<ToolSolid>> = Vec::new();
        for s in &sweeps {
            if !tools.iter().any(|t| Arc::ptr_eq(t, &s.tool)) {
                tools.push(s.tool.clone());
            }
        }
        match Fields::new(&tools, &[]) {
            Ok(f) => skin.gpu = Some(Arc::new(f)),
            Err(e) => {
                eprintln!("no GPU, skipped: {e}");
                return None;
            }
        }
    }
    let t0 = Instant::now();
    for s in sweeps {
        skin.apply(s);
    }
    Some((skin, t0.elapsed().as_secs_f64()))
}

#[test]
fn a_program_skin_lies_on_its_stock() {
    check_program_skin(false);
}

#[test]
fn a_program_skin_read_on_the_gpu_lies_on_its_stock() {
    check_program_skin(true);
}

fn check_program_skin(gpu: bool) {
    let Some((skin, secs)) = program_skin(gpu) else { return };
    let b = skin.stock.bounds;
    let mut off = 0.0f64;
    for p in skin.patches.iter().step_by(5) {
        for &q in &p.p {
            off = off.max(skin.stock.sd(q, 1e-5).abs());
        }
    }
    assert!(off <= skin.chord(), "a corner {off} mm off the stock's zero set (the chords' tolerance {})", skin.chord());
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let n = 50_000;
    let hits = (0..n).filter(|_| skin.stock.sd(rng.in_box(b), 1e-4) < 0.0).count();
    let mc = volume(b) * hits as f64 / n as f64;
    let ours = enclosed(&skin.patches);
    let err = (ours - mc).abs() / mc;
    println!("skin{}: {} pieces in {secs:.1} s, volume {ours:.0} against {mc:.0} by sampling ({:.2} %)", if gpu { " (GPU)" } else { "" }, skin.patches.len(), 100.0 * err);
    assert!(err < 0.01, "the skin encloses {ours:.0} mm3, the stock holds {mc:.0}");
}
