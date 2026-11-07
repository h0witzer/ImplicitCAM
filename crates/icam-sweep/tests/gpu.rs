//! The GPU fields against the CPU's double-precision fields: every interval the kernels
//! give holds the CPU's value, for every kind of motion, every tool shape and a stock of
//! many moves, and comes back no wider than the tolerance asked plus its rounding.
//! Without a GPU the tests say so and pass.

use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use icam_core::tool::Tool;
use icam_sweep::gpu::Fields;
use icam_sweep::{Aabb, Chain, Helix, Joint, Linear, Link, Machine, Motion, Pose, Rot, Stock, Sweep, ToolSolid, V3};
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

fn tools() -> Vec<Arc<ToolSolid>> {
    vec![
        Arc::new(ToolSolid::new(Tool::flat(10.0, 40.0), 30.0)),
        Arc::new(ToolSolid::new(Tool::ball(8.0, 40.0), 30.0)),
        Arc::new(ToolSolid::new(Tool::bull(12.0, 2.0, 40.0), 25.0)),
        Arc::new(ToolSolid::new(Tool::vee(6.0, 118.0, 40.0), 20.0)),
    ]
}

fn machines() -> Vec<Arc<dyn Machine>> {
    let (x, y, z) = (V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0));
    let head_head = Chain {
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
    };
    let table_table = Chain {
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
    };
    let r_theta = Chain { links: vec![Link::Turn { axis: z, q: 0 }, Link::Slide { axis: x, q: 1 }, Link::Slide { axis: z, q: 2 }], axes: 3 };
    vec![Arc::new(head_head), Arc::new(table_table), Arc::new(r_theta)]
}

fn motions(m: &[Arc<dyn Machine>]) -> Vec<(&'static str, Arc<dyn Motion>)> {
    let at = |x: f64, y: f64, z: f64| Pose::translation(V3::new(x, y, z));
    vec![
        ("level line", Arc::new(Linear { from: at(0.0, 0.0, -3.0), to: at(40.0, 10.0, -3.0) })),
        ("ramp", Arc::new(Linear { from: at(0.0, 0.0, 0.0), to: at(30.0, -5.0, -6.0) })),
        ("plunge", Arc::new(Linear { from: at(3.0, 4.0, 5.0), to: at(3.0, 4.0, -8.0) })),
        ("level arc", Arc::new(Helix { center: V3::new(0.0, 0.0, -2.0), normal: V3::new(0.0, 0.0, 1.0), start: V3::new(20.0, 0.0, -2.0), sweep: 0.75 * PI, rise: 0.0, rot: Rot::IDENTITY })),
        ("helix", Arc::new(Helix { center: V3::new(0.0, 0.0, 0.0), normal: V3::new(0.0, 0.0, 1.0), start: V3::new(8.0, 0.0, 0.0), sweep: -2.5 * PI, rise: -6.0, rot: Rot::IDENTITY })),
        (
            "linear, turning",
            Arc::new(Linear { from: Pose::tip_axis(V3::new(0.0, 0.0, 5.0), V3::new(0.3, 0.0, 1.0)), to: Pose::tip_axis(V3::new(40.0, 10.0, -5.0), V3::new(-0.2, 0.5, 1.0)) }),
        ),
        ("tilted helix", Arc::new(Helix { center: V3::new(5.0, 5.0, 0.0), normal: V3::new(0.0, 0.3, 1.0), start: V3::new(25.0, 5.0, 0.0), sweep: 1.5 * PI, rise: -8.0, rot: Rot::z_to(V3::new(0.1, 0.0, 1.0)) })),
        ("head-head", Arc::new(Joint { machine: m[0].clone(), from: vec![10.0, 0.0, 150.0, 0.0, 0.2], to: vec![30.0, 15.0, 140.0, 1.2, -0.4] })),
        ("table-table", Arc::new(Joint { machine: m[1].clone(), from: vec![0.0, 0.0, 20.0, 0.1, 0.0], to: vec![25.0, -10.0, 10.0, 0.6, 2.0] })),
        ("r-theta", Arc::new(Joint { machine: m[2].clone(), from: vec![0.0, 30.0, 5.0], to: vec![2.5, 45.0, -3.0] })),
    ]
}

fn gpu(tools: &[Arc<ToolSolid>], machines: &[Arc<dyn Machine>]) -> Option<Fields> {
    match Fields::new(tools, machines) {
        Ok(f) => {
            eprintln!("GPU: {}", f.adapter);
            Some(f)
        }
        Err(e) => {
            eprintln!("no GPU, skipped: {e}");
            None
        }
    }
}

/// How the GPU's intervals sit against the CPU's values (each within `cpu_tol` above
/// the true field), over the points the limit does not clamp: the worst miss (positive
/// when an interval fails to hold the CPU's value, and how near it comes to failing when
/// negative) and the widest interval.
fn compare(gpu: &[[f64; 2]], cpu: &[f64], limit: f64, cpu_tol: f64) -> (f64, f64) {
    let mut miss = f64::NEG_INFINITY;
    let mut wide = 0.0f64;
    for (g, &c) in gpu.iter().zip(cpu) {
        // The true value lies in [c - cpu_tol, c]; the GPU's interval must meet it.
        let (lo, hi) = (c.clamp(-limit, limit) - cpu_tol, c.clamp(-limit, limit));
        if c.abs() >= limit {
            assert!(g[0] <= hi && g[1] >= lo, "a clamped interval [{}, {}] misses {c}", g[0], g[1]);
            continue;
        }
        miss = miss.max(g[0] - hi).max(lo - g[1]);
        wide = wide.max(g[1] - g[0]);
    }
    (miss, wide)
}

#[test]
fn a_move_interval_holds_the_move_field() {
    let (tools, machines) = (tools(), machines());
    let Some(fields) = gpu(&tools, &machines) else { return };
    let (limit, tol, cpu_tol) = (4.0, 1e-4, 1e-7);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (name, m) in motions(&machines) {
        for tool in &tools {
            let s = Sweep::new(tool.clone(), m.clone());
            let pts: Vec<V3> = (0..4000).map(|_| rng.in_box(s.bounds().grow(3.0))).collect();
            let t0 = Instant::now();
            let g = fields.sweep(&s, &pts, limit, tol).unwrap();
            let gpu_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let c: Vec<f64> = pts.par_iter().map(|&p| s.sd(p, cpu_tol)).collect();
            let cpu_s = t1.elapsed().as_secs_f64();
            let (miss, wide) = compare(&g, &c, limit, cpu_tol);
            let near = c.iter().filter(|v| v.abs() < limit).count();
            eprintln!("{name:16} {:16}: miss {miss:+.2e}, widest {wide:.2e} ({near} of {} points within the limit); GPU {:.1} ms, CPU {:.1} ms", tool.profile.name, pts.len(), gpu_s * 1e3, cpu_s * 1e3);
            assert!(miss <= 0.0, "{name}, {}: an interval misses the CPU's value by {miss}", tool.profile.name);
            assert!(wide <= tol + 2e-3, "{name}, {}: an interval {wide} wide", tool.profile.name);
        }
    }
}

#[test]
fn a_stock_interval_holds_the_stock_field() {
    let (tools, machines) = (tools(), machines());
    let Some(fields) = gpu(&tools, &machines) else { return };
    let (limit, tol, cpu_tol) = (3.0, 1e-4, 1e-7);
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let bounds = Aabb::new(V3::new(-10.0, -10.0, -15.0), V3::new(60.0, 40.0, 0.0));
    let mut stock = Stock::new(bounds);
    let ms = motions(&machines);
    // Passes over the block at a few depths with every tool and every kind of motion.
    let mut k = 0;
    for depth in [-1.0, -3.0, -5.0] {
        // The same moves again at every depth: coincident surfaces.
        for (_, m) in &ms {
            stock.apply(Sweep::new(tools[k % tools.len()].clone(), m.clone()));
            k += 1;
        }
        for i in 0..6 {
            let y = 5.0 * i as f64;
            stock.apply(Sweep::new(tools[k % tools.len()].clone(), Arc::new(Linear { from: Pose::translation(V3::new(-5.0, y, depth)), to: Pose::translation(V3::new(55.0, y + 2.0, depth)) })));
            k += 1;
        }
    }
    let pts: Vec<V3> = (0..20000).map(|_| rng.in_box(bounds.grow(2.0))).collect();
    let t0 = Instant::now();
    let g = fields.stock(&stock, &pts, limit, tol).unwrap();
    let gpu_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let c: Vec<f64> = pts.par_iter().map(|&p| stock.sd(p, cpu_tol)).collect();
    let cpu_s = t1.elapsed().as_secs_f64();
    let (miss, wide) = compare(&g, &c, limit, cpu_tol);
    let near = c.iter().filter(|v| v.abs() < limit).count();
    eprintln!("stock of {} moves: miss {miss:+.2e}, widest {wide:.2e} ({near} of {} points within the limit); GPU {:.1} ms, CPU {:.1} ms", stock.len(), pts.len(), gpu_s * 1e3, cpu_s * 1e3);
    assert!(miss <= 0.0, "an interval misses the CPU's value by {miss}");
    assert!(wide <= tol + 2e-3, "an interval {wide} wide");
}
