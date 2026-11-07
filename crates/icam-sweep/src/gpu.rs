//! The sweep and stock fields on the GPU, for batches of points (wgpu, WGSL).
//!
//! A shader runs fastest as straight-line code over constants, not as a loop reading
//! instructions: on a Quadro T1000 the Prospero expression (7,866 operations) renders a
//! 1024 x 1024 frame in 3.8 ms compiled to straight-line WGSL and in 222 ms interpreted by
//! a WGSL loop. So each program gets its own shader: every tool's distance and every
//! machine's chain is generated as code with its numbers written in (gpu.wgsl holds the
//! rest), compiled once; the moves arrive as data, a record each.
//!
//! The search over a move is the same for every point: fixed stretches of t, as many as
//! the move's bend bound asks for the tolerance (see gpu.wgsl), so neighboring points run
//! the same instructions. Each point gets an interval holding its field clamped to
//! [-limit, limit], widened by a bound on single-precision rounding in a frame centered
//! on the batch; the stock box's own distance is found on the CPU in double precision.
//! A point whose interval cannot decide what the caller asks goes back to the CPU's
//! double-precision field (`Sweep::sd`, `Stock::sd`), the reference these kernels are
//! tested against. Motions the kernels do not know (anything but [`Linear`], [`Helix`] and
//! a [`Joint`] on a [`Chain`] of at most eight axes), and tools that are not convex, come
//! back as the whole line, [-inf, inf].

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::geom::{scale, unit, Aabb, Rot, V3};
use crate::motion::{Chain, Helix, Joint, Linear, Link, Machine};
use crate::stock::Stock;
use crate::sweep::Sweep;
use crate::tool::{Piece, ToolSolid};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    count: u32,
    mode: u32,
    width: u32,
    limit: f32,
    tol: f32,
    sgn: f32,
    off: f32,
    box_lo: [f32; 4],
    box_hi: [f32; 4],
}

/// Points and the moves each reads (see [`Fields::listed`]).
#[derive(Clone, Debug, Default)]
pub struct Lists {
    pub pts: Vec<V3>,
    pub first: Vec<u32>,
    pub count: Vec<u32>,
    pub idx: Vec<u32>,
}

/// Which field: one move's, or the stock's.
#[derive(Clone, Copy)]
pub enum Of<'a> {
    Sweep(&'a Sweep),
    Stock(&'a Stock),
}

/// One move as the kernels read it (gpu.wgsl, `Sweep`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Rec {
    head: [u32; 4],
    lo: [f32; 4],
    hi: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
    e: [f32; 4],
    f: [f32; 4],
}

/// Points per dispatch: far below the OS's GPU watchdog (about 2 s on Windows).
const CHUNK: usize = 1 << 15;
/// Readings in one dispatch, at most (a point or a node against one move listed for it): a
/// dispatch holds the GPU until it is done, and a window drawing on the same GPU waits
/// behind it, so dispatches are bounded by their work, not only by their points.
const DISPATCH_WORK: usize = 1 << 20;

/// Where to cut items into dispatches: at most `most` items, and no more than DISPATCH_WORK
/// readings unless one item alone has more (`work` of item i).
fn dispatches(n: usize, most: usize, work: impl Fn(usize) -> usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut a, mut sum) = (0, 0);
    for i in 0..n {
        let w = work(i);
        if i > a && (i - a >= most || sum + w > DISPATCH_WORK) {
            out.push((a, i));
            (a, sum) = (i, 0);
        }
        sum += w;
    }
    if a < n {
        out.push((a, n));
    }
    out
}
/// Bricks per dispatch in `bricks` (729 nodes each, every node reading only its brick's
/// list): about a quarter of a million nodes.
const BRICKS_AT_ONCE: usize = 360;

/// The whole line, for points the kernels cannot decide.
const UNKNOWN: [f64; 2] = [f64::NEG_INFINITY, f64::INFINITY];

pub struct Fields {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    tools: Vec<Arc<ToolSolid>>,
    machines: Vec<Arc<dyn Machine>>,
    /// The adapter and its backend.
    pub adapter: String,
}

fn lit(x: f64) -> String {
    format!("{:?}", x as f32)
}

fn quat(r: Rot) -> [f32; 4] {
    [r.x as f32, r.y as f32, r.z as f32, r.w as f32]
}

fn four(p: V3, w: f64) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, w as f32]
}

/// A tool's signed distance as straight-line WGSL: the least distance to its profile's
/// pieces in the half-plane, negative within the radius the profile has at that height.
fn tool_code(i: usize, t: &ToolSolid) -> String {
    let pt = |p: (f64, f64)| format!("vec2<f32>({}, {})", lit(p.0), lit(p.1));
    let mut s = format!("fn tool_{i}(q: vec3<f32>) -> f32 {{\n    let p = vec2<f32>(length(q.xy), q.z);\n    var d = BIG;\n");
    for piece in t.pieces() {
        s += &match *piece {
            Piece::Seg { a, b } => format!("    d = min(d, seg_d(p, {}, {}));\n", pt(a), pt(b)),
            Piece::Arc { c, rad, a0, a1 } => format!("    d = min(d, arc_d(p, {}, {}, {}, {}));\n", pt(c), lit(rad), lit(a0), lit(a1)),
        };
    }
    // The radius at the point's height, piece by piece from the tip up (the last piece
    // is the cap across the top; flat pieces have no height of their own).
    let pieces = t.pieces();
    let mut arms: Vec<(f64, String)> = Vec::new();
    for piece in &pieces[..pieces.len().saturating_sub(1)] {
        match *piece {
            Piece::Seg { a, b } if (b.1 - a.1).abs() > 1e-12 => {
                arms.push((a.1.max(b.1), format!("{} + (p.y - {}) * {}", lit(a.0), lit(a.1), lit((b.0 - a.0) / (b.1 - a.1)))));
            }
            Piece::Arc { c, rad, a1, .. } => {
                arms.push((c.1 + rad * a1.sin(), format!("{} + sqrt(max({} - (p.y - {}) * (p.y - {}), 0.0))", lit(c.0), lit(rad * rad), lit(c.1), lit(c.1))));
            }
            _ => {}
        }
    }
    s += "    var r = -1.0;\n";
    for (k, (top, expr)) in arms.iter().enumerate() {
        let word = if k == 0 { "    if" } else { " else if" };
        let cond = if k + 1 == arms.len() { "true".to_string() } else { format!("p.y <= {}", lit(*top)) };
        s += &format!("{word} ({cond}) {{\n        r = {expr};\n    }}");
    }
    s += &format!("\n    let inside = p.y >= 0.0 && p.y <= {} && p.x <= r;\n    return select(d, -d, inside);\n}}\n", lit(t.length));
    s
}

/// A machine's chain as straight-line WGSL, axes 0 to 7 from the two vectors.
fn machine_code(j: usize, c: &Chain) -> String {
    const Q: [&str; 8] = ["qa.x", "qa.y", "qa.z", "qa.w", "qb.x", "qb.y", "qb.z", "qb.w"];
    let v = |p: V3| format!("vec3<f32>({}, {}, {})", lit(p.x), lit(p.y), lit(p.z));
    let mut s = format!("fn machine_{j}(qa: vec4<f32>, qb: vec4<f32>) -> Pose {{\n    var p = Pose(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec3<f32>(0.0));\n");
    for l in &c.links {
        s += &match *l {
            Link::Fixed(f) => format!("    p = compose(p, Pose(vec4<f32>({}, {}, {}, {}), {}));\n", lit(f.rot.x), lit(f.rot.y), lit(f.rot.z), lit(f.rot.w), v(f.at)),
            Link::Turn { axis, q } => format!("    p = compose(p, Pose(about({}, {}), vec3<f32>(0.0)));\n", v(unit(axis)), Q[q]),
            Link::Slide { axis, q } => format!("    p = compose(p, Pose(vec4<f32>(0.0, 0.0, 0.0, 1.0), {} * {}));\n", v(unit(axis)), Q[q]),
        };
    }
    s += "    return p;\n}\n";
    s
}

/// The program's shader: gpu.wgsl and the generated tools and machines.
fn shader(tools: &[Arc<ToolSolid>], machines: &[Option<Chain>]) -> String {
    let mut s = String::from(include_str!("gpu.wgsl"));
    s += "\n// Generated for this program.\n\n";
    for (i, t) in tools.iter().enumerate() {
        s += &tool_code(i, t);
    }
    s += "fn tool_sd(i: u32, q: vec3<f32>) -> f32 {\n    switch i {\n";
    for i in 0..tools.len() {
        s += &format!("        case {i}u: {{\n            return tool_{i}(q);\n        }}\n");
    }
    s += "        default: {\n            return BIG;\n        }\n    }\n}\n";
    for (j, m) in machines.iter().enumerate() {
        if let Some(c) = m {
            s += &machine_code(j, c);
        }
    }
    s += "fn machine(j: u32, qa: vec4<f32>, qb: vec4<f32>) -> Pose {\n    switch j {\n";
    for (j, m) in machines.iter().enumerate() {
        if m.is_some() {
            s += &format!("        case {j}u: {{\n            return machine_{j}(qa, qb);\n        }}\n");
        }
    }
    s += "        default: {\n            return Pose(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec3<f32>(0.0));\n        }\n    }\n}\n";
    s
}

impl Fields {
    /// Compiles the kernels for these tools and machines (a program's: the moves given
    /// later must use these very `Arc`s). Prefers Vulkan: its driver builds a long
    /// straight-line shader in a fraction of a second where DX12's takes seconds.
    pub fn new(tools: &[Arc<ToolSolid>], machines: &[Arc<dyn Machine>]) -> Result<Fields, String> {
        let chains: Vec<Option<Chain>> = machines
            .iter()
            .map(|m| {
                let any: &dyn std::any::Any = m.as_ref();
                any.downcast_ref::<Chain>().filter(|c| c.axes <= 8).cloned()
            })
            .collect();
        let code = shader(tools, &chains);
        pollster::block_on(async {
            let mut found = None;
            for backends in [wgpu::Backends::VULKAN, wgpu::Backends::all()] {
                let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
                desc.backends = backends;
                let instance = wgpu::Instance::new(desc);
                if let Ok(a) = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }).await {
                    found = Some(a);
                    break;
                }
            }
            let adapter = found.ok_or("no GPU adapter")?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { label: Some("icam-sweep"), required_features: wgpu::Features::empty(), required_limits: adapter.limits(), ..Default::default() })
                .await
                .map_err(|e| format!("no GPU device: {e}"))?;
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fields"), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fields"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(e) = scope.pop().await {
                return Err(format!("the fields shader: {e}"));
            }
            Ok(Fields { device, queue, pipeline, tools: tools.to_vec(), machines: machines.to_vec(), adapter: format!("{} ({:?})", info.name, info.backend) })
        })
    }

    /// The program's shader source, for reading.
    pub fn source(tools: &[Arc<ToolSolid>], machines: &[Arc<dyn Machine>]) -> String {
        let chains: Vec<Option<Chain>> = machines
            .iter()
            .map(|m| {
                let any: &dyn std::any::Any = m.as_ref();
                any.downcast_ref::<Chain>().filter(|c| c.axes <= 8).cloned()
            })
            .collect();
        shader(tools, &chains)
    }

    /// The move as a record in the frame centered on `o`, if the kernels know it.
    fn record(&self, s: &Sweep, o: V3) -> Option<Rec> {
        if !s.tool.convex() {
            return None;
        }
        let tool = self.tools.iter().position(|t| Arc::ptr_eq(t, &s.tool))? as u32;
        let reach = s.tool.reach();
        let bb = s.bounds();
        let any: &dyn std::any::Any = s.motion.as_ref();
        let zero = [0.0f32; 4];
        let mut r = Rec { head: [0, tool, 0, 0], lo: four(bb.min - o, 0.0), hi: four(bb.max - o, 0.0), a: zero, b: zero, c: zero, d: zero, e: zero, f: zero };
        if let Some(m) = any.downcast_ref::<Linear>() {
            let mut d = m.from.rot.inverse() * m.to.rot;
            if d.w < 0.0 {
                d = Rot { w: -d.w, x: -d.x, y: -d.y, z: -d.z };
            }
            let spin = V3::new(d.x, d.y, d.z);
            let (turn, axis) = if spin.len() < 1e-15 { (0.0, V3::new(0.0, 0.0, 1.0)) } else { (2.0 * spin.len().atan2(d.w), unit(spin)) };
            let step = m.to.at - m.from.at;
            let v = step.len();
            let tool_axis = m.from.rot.apply(V3::new(0.0, 0.0, 1.0));
            if spin.len() < 1e-15 && crate::geom::dot(step, tool_axis).abs() <= 1e-12 * v.max(1.0) {
                // A line across the tool's axis: closed form (gpu.wgsl, `level_line`).
                r.head[0] = 4;
                r.a = four(m.from.at - o, 0.0);
                r.b = four(m.to.at - o, 0.0);
                r.c = four(tool_axis, 0.0);
                r.e = [v as f32, ((m.from.at - o).len().max((m.to.at - o).len()) + reach) as f32, 0.0, 0.0];
                return Some(r);
            }
            r.a = quat(m.from.rot);
            r.b = four(m.from.at - o, turn);
            r.c = four(axis, 0.0);
            r.d = four(step, 0.0);
            // The frame turns steadily about a fixed axis while its origin runs steadily:
            // q'' = W x (W x q) - 2 W x (the origin's velocity), so |q''| <= 2 turn v + turn^2 |q|.
            r.e = [v as f32, ((m.from.at - o).len().max((m.to.at - o).len()) + reach) as f32, (2.0 * turn * v) as f32, (turn * turn) as f32];
            return Some(r);
        }
        if let Some(m) = any.downcast_ref::<Helix>() {
            let n = unit(m.normal);
            let arm = m.start - m.center;
            let up = crate::geom::dot(arm, n);
            let flat = arm - scale(n, up);
            let radius = flat.len();
            let axis = m.rot.apply(V3::new(0.0, 0.0, 1.0));
            let along = crate::geom::dot(axis, n);
            if m.rise.abs() <= 1e-12 && m.sweep.abs() > 1e-12 && radius > 1e-9 && crate::geom::cross(axis, n).len() <= 1e-12 {
                // A level arc with the tool axis along its normal: closed form (gpu.wgsl).
                r.head[0] = 3;
                r.a = four(m.center - o, m.sweep);
                r.b = four(n, radius);
                r.c = four(unit(flat), up);
                r.d = [along.signum() as f32, 0.0, 0.0, 0.0];
                r.e = [(radius * m.sweep).abs() as f32, ((m.center - o).len() + arm.len() + reach) as f32, 0.0, 0.0];
                return Some(r);
            }
            r.head[0] = 1;
            r.a = four(m.center - o, m.sweep);
            r.b = four(n, m.rise);
            r.c = four(arm, 0.0);
            r.d = quat(m.rot);
            // The frame is held, so q'' is the origin's (centripetal) acceleration.
            r.e = [(radius * m.sweep).hypot(m.rise) as f32, ((m.center - o).len() + arm.len() + m.rise.abs() + reach) as f32, (radius * m.sweep * m.sweep) as f32, 0.0];
            return Some(r);
        }
        if let Some(m) = any.downcast_ref::<Joint>() {
            let j = self.machines.iter().position(|x| Arc::ptr_eq(x, &m.machine))?;
            let chain = (m.machine.as_ref() as &dyn std::any::Any).downcast_ref::<Chain>()?;
            if chain.axes > 8 || m.from.len() != chain.axes || m.to.len() != chain.axes {
                return None;
            }
            let dq: Vec<f64> = m.from.iter().zip(&m.to).map(|(a, b)| b - a).collect();
            let (v, _) = m.machine.rate(&m.from, &m.to, &dq);
            let (c2, _) = m.machine.accel(&m.from, &m.to, &dq);
            if !(v.is_finite() && c2.is_finite()) {
                return None;
            }
            // The frame turns at most W (the turns' rates summed) and W changes at most W^2,
            // so |q''| <= |c''| + 2 W v + 2 W^2 |q|.
            let mut w = 0.0;
            let mut size = o.len() + reach;
            for l in &chain.links {
                match *l {
                    Link::Turn { q, .. } => w += dq[q].abs(),
                    Link::Fixed(f) => size += f.at.len(),
                    Link::Slide { q, .. } => size += m.from[q].abs().max(m.to[q].abs()),
                }
            }
            let axes = |q: &[f64], k: usize| -> [f32; 4] { std::array::from_fn(|i| q.get(4 * k + i).copied().unwrap_or(0.0) as f32) };
            r.head = [2, tool, j as u32, 0];
            r.a = axes(&m.from, 0);
            r.b = axes(&m.from, 1);
            r.c = axes(&m.to, 0);
            r.d = axes(&m.to, 1);
            r.e = [v as f32, size as f32, (c2 + 2.0 * w * v) as f32, (2.0 * w * w) as f32];
            r.f = four(o, 0.0);
            return Some(r);
        }
        None
    }

    /// The records for `of` around the points in `bb` (the stock's moves that can come
    /// within `limit` of them, nearest first), in the frame centered on `o`, with the box
    /// in that frame; None when the kernels cannot place one of the moves.
    fn prepare(&self, of: Of, bb: Aabb, o: V3, limit: f64) -> Option<(u32, Vec<Rec>, Aabb)> {
        match of {
            Of::Sweep(s) => Some((0, vec![self.record(s, o)?], Aabb::new(V3::default(), V3::default()))),
            Of::Stock(stock) => {
                let mut near: Vec<(f64, Rec)> = Vec::new();
                for k in stock.near(bb.grow(limit)) {
                    let sw = stock.sweep(k);
                    near.push((sw.bounds().dist(bb.center()), self.record(sw, o)?));
                }
                // Nearest first, so the best value found early cuts the search of the rest.
                near.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut recs: Vec<Rec> = near.into_iter().map(|x| x.1).collect();
                if recs.is_empty() {
                    recs.push(Rec::zeroed_far());
                }
                Some((1, recs, Aabb::new(stock.bounds.min - o, stock.bounds.max - o)))
            }
        }
    }

    /// The field at each point, clamped to [-limit, limit]: an interval holding it,
    /// searched to within `tol` ([-inf, inf] where the kernels cannot place a move).
    pub fn values(&self, of: Of, pts: &[V3], limit: f64, tol: f64) -> Result<Vec<[f64; 2]>, String> {
        if pts.is_empty() {
            return Ok(Vec::new());
        }
        let bb = bounds(pts);
        let o = frame(of, bb);
        let Some((mode, recs, abox)) = self.prepare(of, bb, o, limit) else { return Ok(vec![UNKNOWN; pts.len()]) };
        let local: Vec<[f32; 4]> = pts.iter().map(|&p| four(p - o, 0.0)).collect();
        self.run(mode, &recs, recs.len(), abox, &local, &[], (1, 1), (limit, tol), (1.0, 0.0))
    }

    /// Along each side (a, b), where sgn field + off is non-negative at a and negative at
    /// b: the fraction of the way to b where it crosses zero, to within the width of its
    /// interval there. None when the kernels cannot place a move.
    pub fn roots(&self, of: Of, sides: &[(V3, V3)], limit: f64, tol: f64, (sgn, off): (f64, f64)) -> Result<Option<Vec<f64>>, String> {
        if sides.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let bb = sides.iter().fold(Aabb::EMPTY, |b, &(x, y)| b.union(Aabb::new(x, x)).union(Aabb::new(y, y)));
        let o = frame(of, bb);
        let Some((mode, recs, abox)) = self.prepare(of, bb, o, limit) else { return Ok(None) };
        let local: Vec<[f32; 4]> = sides.iter().flat_map(|&(a, b)| [four(a - o, 0.0), four(b - o, 0.0)]).collect();
        Ok(Some(self.run(mode + 2, &recs, recs.len(), abox, &local, &[], (2, 1), (limit, tol), (sgn, off))?.into_iter().map(|v| v[0]).collect()))
    }

    /// Each point moved onto the surface of the move it came from, and where `onto` also
    /// onto where sgn field + off is zero (alternate steps along that surface toward it):
    /// (the point, the gradient of its move's field there, an interval holding sgn field +
    /// off there). None when the kernels cannot place a move.
    #[allow(clippy::type_complexity)]
    pub fn settle(&self, of: Of, items: &[(V3, &Sweep, bool)], limit: f64, tol: f64, (sgn, off): (f64, f64)) -> Result<Option<Vec<(V3, V3, [f64; 2])>>, String> {
        if items.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let bb = items.iter().fold(Aabb::EMPTY, |b, &(p, _, _)| b.union(Aabb::new(p, p)));
        let o = frame(of, bb);
        let Some((mode, mut recs, abox)) = self.prepare(of, bb, o, limit) else { return Ok(None) };
        let count = recs.len();
        let mut index: std::collections::HashMap<*const Sweep, u32> = std::collections::HashMap::new();
        let mut local = Vec::with_capacity(2 * items.len());
        for &(p, sw, onto) in items {
            let k = match index.get(&(sw as *const Sweep)) {
                Some(&k) => k,
                None => {
                    let Some(r) = self.record(sw, o) else { return Ok(None) };
                    recs.push(r);
                    let k = (recs.len() - 1) as u32;
                    index.insert(sw as *const Sweep, k);
                    k
                }
            };
            local.push(four(p - o, k as f64));
            local.push([if onto { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]);
        }
        let raw = self.run(mode + 4, &recs, count, abox, &local, &[], (2, 4), (limit, tol), (sgn, off))?;
        Ok(Some(
            raw.chunks(4)
                .map(|c| (V3::new(c[0][0], c[0][1], c[1][0]) + o, V3::new(c[1][1], c[2][0], c[2][1]), c[3]))
                .collect(),
        ))
    }

    /// The stock's field at points that each read their own list of its moves, clamped to
    /// [-limit, limit]: intervals, as [`Fields::values`]. Point i reads the moves
    /// `lists.idx[lists.first[i] ..][.. lists.count[i]]` (indices into the stock), the ones
    /// that can come within the limit of it; every move is placed once for the whole call,
    /// in a frame centered on `o`. None when the kernels cannot place a move.
    pub fn listed(&self, stock: &Stock, o: V3, lists: &Lists, limit: f64, tol: f64) -> Result<Option<Vec<[f64; 2]>>, String> {
        self.listed_in(6, stock, o, lists, limit, tol)
    }

    /// As `listed`, the moves alone: -min of their distances, clamped to [-limit, limit],
    /// without the stock box (how near any of them comes).
    pub fn listed_moves(&self, stock: &Stock, o: V3, lists: &Lists, limit: f64, tol: f64) -> Result<Option<Vec<[f64; 2]>>, String> {
        self.listed_in(8, stock, o, lists, limit, tol)
    }

    fn listed_in(&self, mode: u32, stock: &Stock, o: V3, lists: &Lists, limit: f64, tol: f64) -> Result<Option<Vec<[f64; 2]>>, String> {
        if lists.pts.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let mut slot: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut recs = Vec::new();
        let mut idx = Vec::with_capacity(lists.idx.len());
        for &k in &lists.idx {
            let r = match slot.get(&k) {
                Some(&r) => r,
                None => {
                    let Some(rec) = self.record(stock.sweep(k as usize), o) else { return Ok(None) };
                    recs.push(rec);
                    slot.insert(k, (recs.len() - 1) as u32);
                    (recs.len() - 1) as u32
                }
            };
            idx.push(r);
        }
        if recs.is_empty() {
            recs.push(Rec::zeroed_far());
        }
        if lists.idx.len() >= 1 << 24 {
            return Err("too many listed moves for one call".into());
        }
        let local: Vec<[f32; 4]> = (0..lists.pts.len()).flat_map(|i| [four(lists.pts[i] - o, lists.first[i] as f64), [lists.count[i] as f32, 0.0, 0.0, 0.0]]).collect();
        let abox = Aabb::new(stock.bounds.min - o, stock.bounds.max - o);
        Ok(Some(self.run(mode, &recs, recs.len(), abox, &local, &idx, (2, 1), (limit, tol), (1.0, 0.0))?))
    }

    /// The stock's field at every node of bricks of 8 x 8 x 8 cells, 9 x 9 x 9 nodes to a
    /// brick, x fastest: brick (lo, h, first, count) has its first node at lo, cells of
    /// side h, and reads the moves `idx[first ..][.. count]` (indices into the stock), the
    /// ones that can come within the limit of it. Returns the middle of each node's
    /// interval, clamped to [-limit, limit], 729 to a brick. The nodes are made on the GPU
    /// from their brick, so only the bricks cross over; and in a frame centered on `o` a
    /// whole number of cells from every brick (h a power of two), nodes two bricks share
    /// are the same point in both. Every dispatch is queued before the first is read back.
    /// None when the kernels cannot place a move.
    pub fn bricks(&self, stock: &Stock, o: V3, bricks: &[(V3, f64, u32, u32)], idx: &[u32], limit: f64, tol: f64) -> Result<Option<Vec<f32>>, String> {
        const NODES: usize = 729;
        if bricks.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if idx.len() >= 1 << 24 {
            return Err("too many listed moves for one call".into());
        }
        let mut slot: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut recs = Vec::new();
        let mut moves = Vec::with_capacity(idx.len());
        for &k in idx {
            let r = match slot.get(&k) {
                Some(&r) => r,
                None => {
                    let Some(rec) = self.record(stock.sweep(k as usize), o) else { return Ok(None) };
                    recs.push(rec);
                    slot.insert(k, (recs.len() - 1) as u32);
                    (recs.len() - 1) as u32
                }
            };
            moves.push(r);
        }
        if recs.is_empty() {
            recs.push(Rec::zeroed_far());
        }
        let local: Vec<[f32; 4]> = bricks.iter().flat_map(|&(lo, h, first, count)| [four(lo - o, first as f64), [count as f32, h as f32, 0.0, 0.0]]).collect();
        let abox = Aabb::new(stock.bounds.min - o, stock.bounds.max - o);
        let dev = &self.device;
        let sweeps = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("sweeps"), contents: bytemuck::cast_slice(&recs), usage: wgpu::BufferUsages::STORAGE });
        let ibuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("listed"), contents: bytemuck::cast_slice(if moves.is_empty() { &[0u32] } else { &moves }), usage: wgpu::BufferUsages::STORAGE });
        let layout = self.pipeline.get_bind_group_layout(0);
        let mut reads = Vec::new();
        for (a, z) in dispatches(bricks.len(), BRICKS_AT_ONCE, |i| NODES * (bricks[i].3 as usize).max(1)) {
            let chunk = &local[2 * a..2 * z];
            let n = chunk.len() / 2 * NODES;
            let groups = (n as u32).div_ceil(64);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let params = Params {
                n: n as u32,
                count: recs.len() as u32,
                mode: 7,
                width: gx * 64,
                limit: limit as f32,
                tol: tol as f32,
                sgn: 1.0,
                off: 0.0,
                box_lo: four(abox.min, 0.0),
                box_hi: four(abox.max, 0.0),
            };
            let pbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
            let qbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bricks"), contents: bytemuck::cast_slice(chunk), usage: wgpu::BufferUsages::STORAGE });
            let size = (n * 8) as u64;
            let obuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("out"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let rbuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fields"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sweeps.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: qbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: obuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ibuf.as_entire_binding() },
                ],
            });
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fields") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fields"), timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, size);
            self.queue.submit([enc.finish()]);
            rbuf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            reads.push(rbuf);
        }
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(60)) }).map_err(|e| format!("GPU did not finish: {e}"))?;
        let mut out = Vec::with_capacity(bricks.len() * NODES);
        for rbuf in &reads {
            let data = rbuf.slice(..).get_mapped_range().map_err(|e| format!("GPU results: {e:?}"))?;
            out.extend(bytemuck::cast_slice::<u8, [f32; 2]>(&data).iter().map(|v| 0.5 * (v[0] + v[1])));
        }
        Ok(Some(out))
    }

    /// One move's field at each point, as [`Fields::values`].
    pub fn sweep(&self, s: &Sweep, pts: &[V3], limit: f64, tol: f64) -> Result<Vec<[f64; 2]>, String> {
        self.values(Of::Sweep(s), pts, limit, tol)
    }

    /// The stock's field at each point, as [`Fields::values`].
    pub fn stock(&self, stock: &Stock, pts: &[V3], limit: f64, tol: f64) -> Result<Vec<[f64; 2]>, String> {
        self.values(Of::Stock(stock), pts, limit, tol)
    }

    /// Runs the kernel over `pts`, `per` points to an item and `outs` pairs out of each;
    /// the first `count` records are the field's, `idx` the moves points read in mode 6,
    /// `(limit, tol)` as the field is read, `(sgn, off)` the field cut by.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, mode: u32, recs: &[Rec], count: usize, abox: Aabb, pts: &[[f32; 4]], idx: &[u32], (per, outs): (usize, usize), (limit, tol): (f64, f64), (sgn, off): (f64, f64)) -> Result<Vec<[f64; 2]>, String> {
        let dev = &self.device;
        let sweeps = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("sweeps"), contents: bytemuck::cast_slice(recs), usage: wgpu::BufferUsages::STORAGE });
        let ibuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("listed"), contents: bytemuck::cast_slice(if idx.is_empty() { &[0u32] } else { idx }), usage: wgpu::BufferUsages::STORAGE });
        let layout = self.pipeline.get_bind_group_layout(0);
        let mut out = Vec::with_capacity(pts.len() / per * outs);
        // Points with lists (per 2: the count in the second word) are cut by their work too.
        let cuts = if per == 2 { dispatches(pts.len() / 2, CHUNK, |i| (pts[2 * i + 1][0] as usize).max(1)) } else { (0..pts.len() / per).step_by(CHUNK).map(|a| (a, (a + CHUNK).min(pts.len() / per))).collect() };
        for (a, z) in cuts {
            let chunk = &pts[per * a..per * z];
            let n = chunk.len() / per;
            let groups = (n as u32).div_ceil(64);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let params = Params {
                n: n as u32,
                count: count as u32,
                mode,
                width: gx * 64,
                limit: limit as f32,
                tol: tol as f32,
                sgn: sgn as f32,
                off: off as f32,
                box_lo: four(abox.min, 0.0),
                box_hi: four(abox.max, 0.0),
            };
            let pbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
            let qbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("points"), contents: bytemuck::cast_slice(chunk), usage: wgpu::BufferUsages::STORAGE });
            let size = (n * 8 * outs) as u64;
            let obuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("out"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let rbuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fields"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sweeps.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: qbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: obuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ibuf.as_entire_binding() },
                ],
            });
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fields") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fields"), timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, size);
            self.queue.submit([enc.finish()]);
            rbuf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(20)) }).map_err(|e| format!("GPU did not finish: {e}"))?;
            let data = rbuf.slice(..).get_mapped_range().map_err(|e| format!("GPU results: {e:?}"))?;
            out.extend(bytemuck::cast_slice::<u8, [f32; 2]>(&data).iter().map(|v| [v[0] as f64, v[1] as f64]));
        }
        Ok(out)
    }
}

impl Rec {
    /// A move no point can reach: an empty box far away.
    fn zeroed_far() -> Rec {
        let far = [1e30f32, 1e30, 1e30, 0.0];
        Rec { head: [0; 4], lo: far, hi: far, a: [0.0, 0.0, 0.0, 1.0], b: [0.0; 4], c: [0.0, 0.0, 1.0, 0.0], d: [0.0; 4], e: [0.0; 4], f: [0.0; 4] }
    }
}

/// Where to center the single-precision frame: on the move, whose field matters only
/// near it (points far off it read far from zero, where their rounding cannot matter),
/// or on the points for the stock (only moves near them are read).
fn frame(of: Of, bb: Aabb) -> V3 {
    match of {
        Of::Sweep(s) => s.bounds().center(),
        Of::Stock(_) => bb.center(),
    }
}

fn bounds(pts: &[V3]) -> Aabb {
    pts.iter().fold(Aabb::EMPTY, |b, &p| b.union(Aabb::new(p, p)))
}
