//! icam-sim play: the stock a program leaves, cut move by move at runtime, in a window.
//!
//! The stock is the box less every move applied so far (icam-sweep). Its skin, flat
//! pieces whose corners the fields put on the stock's zero set, is updated as each move
//! is applied, on a loader thread, and handed to the window a few times a second, so the
//! stock is seen being cut from the first moment. The window draws the pieces as they
//! are: plain draws, nothing sampled on a grid. Faces of the stock box no move has
//! touched are drawn as raw stock; surfaces a move left, as machined metal.
//!
//! With --bake the whole program's stock is drawn instead from a cache of its field
//! (docs/SIM_VISUALIZATION_ENGINE.md): bricks of the field's values baked on the GPU in
//! levels around the view, each twice the cell of the one before, so a cell spans about a
//! pixel wherever the camera looks; ray-cast every frame (cache.wgsl), never meshed. A
//! worker thread re-bakes a level when the view moves off it. The playhead picks how many
//! moves have cut the stock; moving it re-bakes only where the moves passed over come near.
//! With --part the stock is colored by what the program left against the part (heat.rs):
//! red where it cut into the part, green within the shop tolerance of it, blue where it
//! left stock on it, gray away from it; stock no move touched is drawn plain.
//!
//! Mouse: buttons a trackpad has (no middle one), moving as CAD views commonly do, with the
//! scroll wheel reversed. Left drag
//! turns about z, Alt + left drag turns freely (and coasts when let go moving), right,
//! middle or Shift + left drag pans, each about what is drawn under the cursor; the wheel
//! zooms about the point under the cursor (a scroll away zooms out), two middle clicks zoom
//! to fit. A click that hardly moves moves nothing: clicks are kept for picking.
//! Keys: F zooms to fit, Z and Shift + Z zoom out and in, Shift + 1 to 7 turn to the
//! standard views (front, back, left, right, top, bottom, isometric), T looks straight down,
//! Space pauses and resumes, Esc closes.
//! With --bake the arrows move the playhead (left and right a move, up and down a hundred),
//! Page Up and Down a thousand, Home to the start and End to the end. Space plays the
//! program from the playhead on the machine's clock (each move at its programmed feed,
//! rapids at icam-motion's rapid rate; acceleration is not modeled): R at the machine's own
//! speed, P at the speed that runs the rest of the program in a minute, [ and ] halve and
//! double it. While a move runs, the stock is cut by it up to where the tool stands (the
//! stock's tail, stock.rs), so at any speed every frame is the stock at that instant and the
//! cut stands where the tool is. The toolpath is drawn through the whole program (moves done
//! gray, feeds ahead cyan, rapids ahead orange), the tool where the playhead stands.
//!
//! With --bake the window carries the comparison panel (ui.rs): the regions the comparison
//! of the finished program finds (regions.rs, on a thread of its own), the display settings,
//! and over the view the region picked, the play bar and its tools. The field is drawn in
//! what the panel leaves.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use crate::heat::{Heat, HEAT_WORDS};
use crate::partcache::{PartCache, PartLevel};
use crate::regions::{self, Region};
use crate::ui::{self, Act};
use icam_motion::analyze::RAPID_MMPS;
use icam_core::prismatic::{Place, ReadPart};
use icam_post::ToolDef;
use icam_sweep::cache::{CELL_WORDS, Cache, SKIP, WORDS};
use icam_sweep::skin::BOX;
use icam_sweep::{Aabb, Skin, V3};
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

const FOV: f32 = 40.0 * std::f32::consts::PI / 180.0;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;
/// How often the loader hands the window the skin, seconds.
const SHOW_EVERY: f64 = 0.25;
/// Bricks to a side of a level of the cache, at most (docs/SIM_VISUALIZATION_ENGINE.md,
/// section 4.4).
const LEVEL_BRICKS: i64 = 128;
/// Levels the ray caster reads, at most (cache.wgsl, `MAX_LEVELS`).
const MAX_LEVELS: usize = 16;
/// Whether a level the view moves on from is scrolled (`cache::scroll`) rather than placed
/// from nothing: off with --no-scroll, to measure one against the other.
static SCROLL: AtomicBool = AtomicBool::new(true);
/// The clock's acceleration by default, mm/s^2: one effective figure for the shop's router,
/// fitted to how long it ran real programs, not the planner's machine model (2,500,
/// docs/ROUGHING_FEASIBILITY.md sections 12 and 13), which runs a production nest of 43,142
/// moves in 6 h 24 min that the router took 13 h over. At 145 that nest takes 13 h 0 min and
/// a 3D relief of 410,891 short moves, which the router ran in about an hour and a half,
/// 1 h 35 min: fitted to the one, it gives the other. It stands for everything that slows
/// the router below its feeds (corners, short blocks, its servo), not a measured
/// acceleration (docs/SIM_VISUALIZATION_ENGINE.md, section 4.6).
pub const ACCEL: f64 = 145.0;

/// What the clock knows of the machine: its acceleration (mm/s^2) and, when it has one, the
/// most program blocks its controller runs a second (a block shorter than its feed covers
/// in that time is run slower).
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    pub accel: f64,
    pub blocks: Option<f64>,
}
pub struct Args {
    pub program: PathBuf,
    pub set: PathBuf,
    /// The part and where it was placed for planning: its box is the stock's.
    pub part: Option<(PathBuf, Place)>,
    /// The stock box instead.
    pub bbox: Option<[f64; 6]>,
    /// Tool stick-out, mm.
    pub length: f64,
    /// Cut resolution, mm.
    pub finest: f64,
    /// Only the first this many moves.
    pub limit: Option<usize>,
    /// Once every move is in, write a picture of it and close.
    pub shot: Option<PathBuf>,
    pub look: Option<[f64; 5]>,
    /// Read the fields on the CPU only (the GPU's are compiled for the program's tools).
    pub cpu: bool,
    /// Draw the whole program's stock from a cache of its field instead of cutting the
    /// skin move by move, in levels around the view, cells no finer than this, mm.
    pub bake: Option<f64>,
    /// With --bake: check the cache against the exact field at its cell centers, and the
    /// caster against the cache along the view's rays; with --at too, that re-baking the
    /// cache to the playhead gives the cache baked there from nothing.
    pub check: bool,
    /// With --bake: the playhead, how many moves have cut the stock (default: all).
    pub at: Option<usize>,
    /// With --bake and --shot: first step the playhead across the whole program in this
    /// many steps, timing each, then write the picture halfway through.
    pub scrub: Option<usize>,
    /// With --bake: play the program from the playhead at this many times the machine's
    /// speed (1: its own) once the view is in. With --shot too: for ten seconds (or to its
    /// end), measuring how the stock keeps up, then write the picture.
    pub play: Option<f64>,
    /// The machine the clock runs.
    pub machine: Machine,
    /// With --shot and the part: pick region n of the panel's list (the camera on it) first.
    pub pick: Option<usize>,
    /// Open on the part's view (the panel's "Design part").
    pub part_view: bool,
    /// With --shot and the part: the moment gouge n of the panel's list was cut first.
    pub moment: Option<usize>,
    /// With --play and --shot: two seconds into the run, the camera this many times closer
    /// (the zoom's cost to the frames and the stock, measured).
    pub zoom: Option<f64>,
    /// With --play and --shot: from two seconds into the run, the camera's target moves
    /// along x at this many mm a second (a pan's cost to the frames and the levels, measured).
    pub pan: Option<f64>,
    /// Levels the view moves on from are placed from nothing, never scrolled (to compare).
    pub no_scroll: bool,
    /// With --play: a picture of the view every fifth of a second of the run, into this
    /// folder (what the window showed while it played, frame by frame).
    pub grab: Option<PathBuf>,
    /// A STEP solid viewed alone, with no program: its exact signed distance
    /// (`icam_core::solid3::Exact`) in the part's view, the stock its padded box, uncut.
    pub solid: Option<PathBuf>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    p: [f32; 3],
    n: [f32; 3],
    c: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    light: [f32; 4],
}

/// What the ray caster reads of the view (cache.wgsl, `View`): the ray through the pixel
/// at (x, y) in normalized device coordinates leaves the eye along fwd + x right + y up.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RayView {
    view_proj: [[f32; 4]; 4],
    right: [f32; 4],
    up: [f32; 4],
    fwd: [f32; 4],
    eye: [f32; 4],
    light: [f32; 4],
    /// The tool (cache.wgsl, `View`): tip and radius, axis and corner radius, (cone height,
    /// stick-out, 1 when drawn, 0).
    tool_at: [f32; 4],
    tool_axis: [f32; 4],
    tool_shape: [f32; 4],
    /// The colors' bands: (tolerance, the ramps' deep end, mm; 0, 0).
    heat: [f32; 4],
}

/// A toolpath vertex (path.wgsl): where (the box's frame), the playhead at which its move
/// is done, and 1 for a rapid.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathVertex {
    p: [f32; 3],
    done: f32,
    rapid: f32,
}

/// What path.wgsl reads: the view, and the playhead.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathUniform {
    view_proj: [[f32; 4]; 4],
    at: [f32; 4],
}

/// A move on the machine's clock (`toolpath`): from `t0` to `t1` seconds the tool runs
/// `motion` (work frame), shaped (radius, corner radius, cone height, stick-out; radius 0
/// when the sheet does not have the tool). `before` sweeps are done when it starts, and it
/// is sweep `before` itself when it `cuts`. Its G-code line and operation (into the ops).
#[derive(Clone)]
struct Span {
    t0: f64,
    t1: f64,
    motion: Arc<dyn icam_sweep::Motion>,
    shape: [f32; 4],
    before: usize,
    cuts: bool,
    line: usize,
    op: u32,
    /// For a move the clock sampled more than once: (seconds from the move's start, how
    /// far along it), from (0, 0) to (t1 - t0, 1); else it runs at a steady rate.
    knots: Option<Box<[(f64, f64)]>>,
}

impl Span {
    /// When on the clock the tool is u along the move (the knots read backward).
    fn time_at(&self, u: f64) -> f64 {
        let Some(k) = &self.knots else {
            return self.t0 + u * (self.t1 - self.t0);
        };
        let i = k.partition_point(|x| x.1 <= u).clamp(1, k.len() - 1);
        let (a, b) = (k[i - 1], k[i]);
        let s = if b.1 > a.1 { ((u - a.1) / (b.1 - a.1)).clamp(0.0, 1.0) } else { 1.0 };
        self.t0 + a.0 + (b.0 - a.0) * s
    }

    /// How far along the move the clock at t is.
    fn along(&self, t: f64) -> f64 {
        if self.t1 <= self.t0 {
            return 1.0;
        }
        let Some(k) = &self.knots else {
            return ((t - self.t0) / (self.t1 - self.t0)).clamp(0.0, 1.0);
        };
        let at = t - self.t0;
        let i = k.partition_point(|x| x.0 <= at).clamp(1, k.len() - 1);
        let (a, b) = (k[i - 1], k[i]);
        if b.0 > a.0 {
            (a.1 + (b.1 - a.1) * (at - a.0) / (b.0 - a.0)).clamp(0.0, 1.0)
        } else {
            b.1
        }
    }
}

/// A move on the machine's clock: its seconds, and its knots (see `Span`).
pub type Timed = (f64, Option<Box<[(f64, f64)]>>);

/// The machine's clock over the program, as icam-motion's time model reads a path
/// (`analyze::measure`), over the whole program as one path sampled every half millimeter:
/// speed capped by each move's programmed feed (`caps`, mm/s) and by how fast the controller
/// takes its blocks (a block's length times the blocks a second), by the path's curvature
/// (v <= sqrt(a R), the turn read over about a millimeter either way, in three dimensions),
/// and by acceleration a, from rest at the start to rest at the end. Each move's seconds,
/// and for a move sampled more than once, its knots (see `Span`).
pub fn machine_clock(motions: &[Arc<dyn icam_sweep::Motion>], caps: &[f64], machine: Machine) -> Vec<Timed> {
    const STEP: f64 = 0.5;
    const WIN: f64 = 1.0;
    let accel = machine.accel;
    let (mut p, mut cap, mut own, mut along): (Vec<V3>, Vec<f64>, Vec<u32>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, m) in motions.iter().enumerate() {
        let len = m.rate(0.0, 1.0).0;
        if len <= 1e-9 {
            continue;
        }
        let top = machine.blocks.map_or(caps[i], |b| caps[i].min(b * len));
        let first = if p.is_empty() { 0 } else { 1 };
        let n = (len / STEP).ceil().max(1.0) as usize;
        for j in first..=n {
            let u = j as f64 / n as f64;
            p.push(m.pose(u).at);
            cap.push(top);
            own.push(i as u32);
            along.push(u);
        }
    }
    let mut out: Vec<Timed> = vec![(0.0, None); motions.len()];
    let n = p.len();
    if n < 2 {
        return out;
    }
    let ds: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { (p[i] - p[i - 1]).len() }).collect();
    let mut v: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i == 0 || i == n - 1 {
                return 0.0;
            }
            let (mut a, mut b, mut la, mut lb) = (i, i, 0.0, 0.0);
            while a > 0 && la < WIN {
                la += ds[a];
                a -= 1;
            }
            while b < n - 1 && lb < WIN {
                b += 1;
                lb += ds[b];
            }
            let (u, w) = (p[i] - p[a], p[b] - p[i]);
            let (nu, nw) = (u.len(), w.len());
            if nu < 1e-9 || nw < 1e-9 {
                return cap[i];
            }
            let turn = (icam_sweep::dot(u, w) / (nu * nw)).clamp(-1.0, 1.0).acos();
            if turn < 1e-6 {
                return cap[i];
            }
            cap[i].min((accel * 0.5 * (nu + nw) / turn).sqrt())
        })
        .collect();
    for i in 1..n {
        v[i] = v[i].min((v[i - 1] * v[i - 1] + 2.0 * accel * ds[i]).sqrt());
    }
    for i in (0..n - 1).rev() {
        v[i] = v[i].min((v[i + 1] * v[i + 1] + 2.0 * accel * ds[i + 1]).sqrt());
    }
    let mut knots: Vec<Vec<(f64, f64)>> = vec![Vec::new(); motions.len()];
    for i in 1..n {
        let m = own[i] as usize;
        out[m].0 += ds[i] / (0.5 * (v[i] + v[i - 1])).max(1e-3);
        if knots[m].is_empty() {
            knots[m].push((0.0, 0.0));
        }
        knots[m].push((out[m].0, along[i]));
    }
    for (o, k) in out.iter_mut().zip(knots) {
        if k.len() > 2 {
            o.1 = Some(k.into_boxed_slice());
        }
    }
    out
}

/// What the part's cache thread is asked for: every level the window is to hold, by its
/// exponent, first node, cell side and bricks along each axis.
type PartAsk = Vec<(i32, V3, f64, [i64; 3])>;

/// The placer (`spawn_placer`): what it is asked (a level's exponent, first node, bricks and
/// the moves it is to hold), and what it hands back (the same, and the level baked).
type PlaceAsk = (i32, V3, [i64; 3], usize);
type Placed = (i32, V3, [i64; 3], Result<Cache, String>);
type Placer = (Sender<PlaceAsk>, Receiver<Placed>);

/// Runs `f` with its parallel work in a pool of threads of its own, half the cores: the
/// placer, the part's caches and the comparison run long jobs, and in the pool they shared
/// with the stock's worker its small jobs queued behind theirs (the part's colors for 21
/// bricks waited 2.4 s, the stock standing still meanwhile).
fn own_pool<R: Send>(name: &str, f: impl FnOnce() -> R + Send) -> R {
    let n = std::thread::available_parallelism().map_or(4, |n| n.get()).div_ceil(2);
    let name = name.to_string();
    match rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(move |i| format!("{name} {i}")).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// The placer: a thread of its own, with its own stock (no tail) and its own GPU device, that
/// bakes levels from nothing where the view places them anew, so the worker's levels held are
/// kept at the playhead meanwhile (a level from nothing takes seconds at fine cells). The
/// newest ask for each level is taken, coarse first.
fn spawn_placer(b: Aabb, sweeps: Vec<icam_sweep::Sweep>, solids: Vec<Arc<icam_sweep::ToolSolid>>) -> Placer {
    let (ask_tx, ask_rx) = std::sync::mpsc::channel::<PlaceAsk>();
    let (put_tx, put_rx) = std::sync::mpsc::channel::<Placed>();
    std::thread::spawn(move || own_pool("placer", move || {
        let fields = match icam_sweep::gpu::Fields::new(&solids, &[]) {
            Ok(f) => f,
            Err(e) => {
                while let Ok((e0, lo, nb, _)) = ask_rx.recv() {
                    let _ = put_tx.send((e0, lo, nb, Err(e.clone())));
                }
                return;
            }
        };
        let mut stock = icam_sweep::Stock::new(b);
        for s in sweeps {
            stock.apply(s);
        }
        while let Ok(first) = ask_rx.recv() {
            let mut todo = BTreeMap::new();
            todo.insert(first.0, first);
            while let Ok(a) = ask_rx.try_recv() {
                todo.insert(a.0, a);
            }
            for (e, lo, nb, moves) in todo.into_values().rev() {
                let h = 2f64.powi(e);
                let clock = Instant::now();
                let c = icam_sweep::cache::cache_at(&stock, &fields, h, lo, nb, (moves, 0.0), (0.02 * h).min(1e-3));
                if let Ok(c) = &c {
                    println!("play: level placed in {:.2} s: {}", clock.elapsed().as_secs_f64(), c.report);
                }
                if put_tx.send((e, lo, nb, c)).is_err() {
                    return;
                }
            }
        }
    }));
    (ask_tx, put_rx)
}

/// A playhead as text: moves done, and how far along the next.
fn at_text((k, part): (usize, f64)) -> String {
    if part > 0.0 {
        format!("move {k} and {part:.3} of the next")
    } else {
        format!("move {k}")
    }
}

/// Seconds as h:mm:ss.s.
fn clock_text(t: f64) -> String {
    let t = t.max(0.0);
    format!("{}:{:02}:{:04.1}", (t / 3600.0) as u64, ((t / 60.0) as u64) % 60, t % 60.0)
}

/// A level of the cache as the ray caster reads it (cache.wgsl, `Level`), in the box's
/// frame: its first node and cell side, its bricks along each axis and where its brick map
/// starts.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LevelUniform {
    lo: [f32; 4],
    dims: [u32; 4],
    /// y: where the level's bricks start in the atlas (its map holds its own slots).
    part: [u32; 4],
}

/// A level of the part's cache as the ray caster reads it (cache.wgsl, `PartLevel`): its
/// first node and cell side, its bricks along each axis and where its map starts, and where
/// its bricks start.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PartLevelUniform {
    lo: [f32; 4],
    dims: [u32; 4],
    at: [u32; 4],
}

/// The levels, finest first, and the stock box (cache.wgsl, `Levels`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LevelsUniform {
    count: [u32; 4],
    box_lo: [f32; 4],
    box_hi: [f32; 4],
    level: [LevelUniform; MAX_LEVELS],
    parts: [PartLevelUniform; MAX_LEVELS],
}

/// What the window asks the cache's worker for: levels around this view, w x h pixels, of
/// the stock after its first `at.0` moves and the first `at.1` of the next (None: all).
#[derive(Clone, Copy)]
struct Ask {
    cam: Camera,
    w: u32,
    h: u32,
    at: Option<(usize, f64)>,
    /// Paused: a new playhead is wanted at once, coarse levels first, not in rounds.
    paused: bool,
}

/// What the loader hands the window.
enum Load {
    Stage(String),
    Box(Aabb),
    Skin { vertices: Vec<Vertex>, done: usize, total: usize, secs: f64 },
    /// A level of the field's cache, cells of 2^e mm: new, placed anew around the view, or a
    /// patch of the one held (`Cache::patch`); with the part's records (heat.rs) for the
    /// slots it lists (every brick's, or a patch's changed ones), when the part is given; a
    /// whole level with its words staged in a buffer of the window's GPU (`stage`).
    Level { e: i32, cache: Arc<Cache>, heat: Option<Arc<(Vec<u32>, Vec<u32>)>>, staged: Option<wgpu::Buffer> },
    /// The toolpath, every move on the machine's clock, and the operations' names.
    Path { vertices: Vec<PathVertex>, spans: Vec<Span>, ops: Vec<String> },
    /// A level the view no longer needs.
    Drop(i32),
    /// Every level the latest view needs is in.
    Settled,
    /// How many moves the program has.
    Moves(usize),
    /// What the comparison is doing, and what it found (regions.rs).
    Comparing(String),
    Report(Arc<regions::Report>),
    /// The part's cache in cells of 2^e mm (side h; its first node at lo, work frame; dims
    /// bricks along each axis), in place of any it had for e.
    PartLevel { e: i32, lo: V3, h: f64, dims: [u32; 3], level: Arc<PartLevel> },
    /// A level of the part's cache no longer wanted.
    DropPart(i32),
    Failed(String),
}

type M4 = [[f32; 4]; 4];

fn mul(a: &M4, b: &M4) -> M4 {
    let mut c = [[0.0; 4]; 4];
    for (j, col) in c.iter_mut().enumerate() {
        for (i, v) in col.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[k][i] * b[j][k]).sum();
        }
    }
    c
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit3(a: [f32; 3]) -> [f32; 3] {
    let l = dot3(a, a).sqrt().max(1e-20);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

/// v turned by `angle` about `axis` (Rodrigues).
fn turn3(v: [f32; 3], angle: f32, axis: [f32; 3]) -> [f32; 3] {
    let (s, c) = angle.sin_cos();
    let k = unit3(axis);
    add3(add3(scale3(v, c), scale3(cross3(k, v), s)), scale3(k, dot3(k, v) * (1.0 - c)))
}

/// The view controller: a drag of the view's diagonal turns it three times round.
const ROTATION_CYCLES: f32 = 3.0;
/// A free rotate let go while moving coasts, keeping this much of its speed a 60th of a
/// second, if the mouse moved in the last 50 ms.
const ROTATION_INERTIA: f32 = 0.77;
const INERTIA_WINDOW: f32 = 0.05;
/// A rolled view is leveled before a turntable drag turns it: at once under this much roll
/// (radians), else over 125 ms, or 250 ms past 45 degrees.
const LEVEL_SNAP: f32 = 0.001;
/// Pixels either side of the cursor searched for something drawn to turn about.
const SEARCH_RADIUS: u32 = 3;
/// Rays over the view's width for the middle of what is drawn (about 2000 over the view),
/// and how many must meet something.
const SAMPLES_WIDE: u32 = 64;
const MIN_SAMPLES: usize = 4;
/// The box's center is turned about when the box fits the view grown by this.
const ROTATE_BOUNDS_FACTOR: f32 = 1.2;
/// A wheel notch is a step of 0.5, times 3 on Windows; a step s scales the view by 2^(-s/6).
const WHEEL_STEP: f32 = if cfg!(windows) { 1.5 } else { 0.5 };
const STEPS_PER_DOUBLING: f32 = 6.0;
/// The scroll wheel reversed: a scroll away zooms out.
const REVERSE_WHEEL: bool = true;
/// A zoom out stops with the box this many pixels across.
const MIN_MODEL_PIXELS: f32 = 100.0;
/// A zoom in stops with a pixel this small, mm: f32 coordinates a few hundred mm from the
/// box's corner step by about 3e-5 mm, so past it the picture is rounding.
const MIN_PIXEL: f32 = 1e-4;
/// Travel under which a press and release is a click, pixels; two middle clicks this close
/// in time zoom to fit, seconds.
const CLICK_TRAVEL: f32 = 4.0;
const MIDDLE_DOUBLE_CLICK: f32 = 0.5;
/// Z zooms out by this, Shift + Z in, about the middle of the view.
const KEY_ZOOM: f32 = 1.75;
/// A zoom to fit leaves this much room.
const FIT_ROOM: f32 = 1.05;
/// How long the camera takes: a level, a key's zoom, a standard view or a fit, seconds.
const SNAP_SECS: f32 = 0.125;
const SHORT_SECS: f32 = 0.25;
const LONG_SECS: f32 = 0.5;

/// A camera in the stock box's frame (its low corner at the origin, mm), z up, turned
/// freely: nothing fixes where it turns or clamps how far, so it goes over
/// the top. It looks along `dir` with `up` (unit, square to each other) at `target`, `dist`
/// ahead of the eye. The levels are centered on the target and sized for its distance
/// (`levels_for`), so the controls keep it at the depth of what is looked at.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Camera {
    target: [f32; 3],
    dir: [f32; 3],
    up: [f32; 3],
    dist: f32,
}

impl Camera {
    /// Looking at `target` from `dist` away, from the bearing `yaw` and `pitch` above the
    /// horizon (radians), level.
    fn toward(target: [f32; 3], yaw: f32, pitch: f32, dist: f32) -> Camera {
        let ((sp, cp), (sy, cy)) = (pitch.sin_cos(), yaw.sin_cos());
        Camera { target, dir: [-cp * cy, -cp * sy, -sp], up: [-sp * cy, -sp * sy, cp], dist }
    }

    fn fit(size: [f32; 3], aspect: f32) -> Camera {
        let half = (0.5 * size[0] / aspect).max(0.5 * size[1]).max(0.5 * size[2]);
        Camera::toward([0.5 * size[0], 0.5 * size[1], size[2]], -90f32.to_radians(), 55f32.to_radians(), 1.3 * half / (0.5 * FOV).tan())
    }

    /// The view `--look` gives: target x, y (mm, box frame), distance, yaw, pitch (degrees).
    fn look(&mut self, l: [f64; 5]) {
        *self = Camera::toward([l[0] as f32, l[1] as f32, self.target[2]], (l[3] as f32).to_radians(), (l[4] as f32).to_radians(), l[2] as f32);
    }

    /// The bearing the camera looks from (`toward`'s yaw): of the eye from the target, or
    /// looking straight down or up, of the screen's up.
    fn yaw(&self) -> f32 {
        let [x, y, z] = self.dir;
        if x.hypot(y) > 1e-3 {
            (-y).atan2(-x)
        } else {
            (self.up[1] * z).atan2(self.up[0] * z)
        }
    }

    /// Looking along `dir` with `up` (the standard views'), at the same target, as far.
    fn turned(&self, dir: [f32; 3], up: [f32; 3]) -> Camera {
        let mut c = Camera { dir, up, ..*self };
        c.square();
        c
    }

    fn eye(&self) -> [f32; 3] {
        sub3(self.target, scale3(self.dir, self.dist))
    }

    /// The way it looks, its right and its up.
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let f = unit3(self.dir);
        let s = unit3(cross3(f, self.up));
        (f, s, cross3(s, f))
    }

    fn square(&mut self) {
        let (f, _, u) = self.basis();
        (self.dir, self.up) = (f, u);
    }

    /// The view turned by `angle` about `axis` through `pivot`, which stays where it is on
    /// screen.
    fn rotate_about(&mut self, angle: f32, axis: [f32; 3], pivot: [f32; 3]) {
        if angle == 0.0 {
            return;
        }
        self.dir = turn3(self.dir, angle, axis);
        self.up = turn3(self.up, angle, axis);
        self.target = add3(pivot, turn3(sub3(self.target, pivot), angle, axis));
        self.square();
    }

    /// How far off level the camera is: the angle its right makes with the horizon.
    fn roll(&self) -> f32 {
        self.basis().1[2].clamp(-1.0, 1.0).asin()
    }

    /// The camera turned level about `pivot`.
    fn leveled(&self, pivot: [f32; 3]) -> Camera {
        let mut c = *self;
        let axis = cross3(self.basis().1, [0.0, 0.0, 1.0]);
        if dot3(axis, axis) > 1e-18 {
            c.rotate_about(-self.roll(), axis, pivot);
        }
        c
    }

    /// A rotate drag of (dx, dy) pixels over a view w x h, about `pivot`: a turntable
    /// about z, or (`free`) a rotate about the camera's own up and right. The model follows
    /// the mouse; past the pole the turntable's sideways sense flips, so upside down it still
    /// does.
    fn drag(&mut self, dx: f32, dy: f32, (w, h): (f32, f32), pivot: [f32; 3], free: bool) {
        let per = 2.0 * std::f32::consts::PI * ROTATION_CYCLES / w.hypot(h).max(1.0);
        if free {
            self.rotate_about(-dx * per, self.up, pivot);
        } else {
            let sense = if self.up[2] < 0.0 { -1.0 } else { 1.0 };
            self.rotate_about(-dx * per * sense, [0.0, 0.0, 1.0], pivot);
        }
        let right = self.basis().1;
        self.rotate_about(-dy * per, right, pivot);
    }

    /// The eye moved toward `p` (k below 1) or away, its distance from p times k: p stays where
    /// it is on screen and what is around it grows k times smaller, as an orthographic view
    /// zooms about the cursor. The target is put at p's depth.
    fn zoom_at(&mut self, p: [f32; 3], k: f32) {
        let eye = add3(p, scale3(sub3(self.eye(), p), k));
        let depth = dot3(sub3(p, eye), self.dir);
        if depth > 0.0 {
            self.dist = depth;
            self.target = add3(eye, scale3(self.dir, depth));
        }
    }

    /// The ray through pixel (x, y) of a view w x h: the eye and a unit direction.
    fn ray(&self, x: f32, y: f32, (w, h): (f32, f32)) -> ([f32; 3], [f32; 3]) {
        let (f, s, u) = self.basis();
        let t = (0.5 * FOV).tan();
        let (nx, ny) = ((2.0 * x / w - 1.0) * t * w / h, (1.0 - 2.0 * y / h) * t);
        (self.eye(), unit3(add3(f, add3(scale3(s, nx), scale3(u, ny)))))
    }

    /// Where that ray meets the plane through `q` square to the view, ahead of the eye.
    fn on_plane(&self, x: f32, y: f32, size: (f32, f32), q: [f32; 3]) -> Option<[f32; 3]> {
        let (o, d) = self.ray(x, y, size);
        let along = dot3(d, self.dir);
        let k = dot3(sub3(q, o), self.dir) / along;
        (along > 1e-6 && k > 0.0).then(|| add3(o, scale3(d, k)))
    }

    /// Pixels a millimeter at depth `depth` in a view `h` high.
    fn pixels_per_mm(depth: f32, h: f32) -> f32 {
        h / (2.0 * depth.max(1e-9) * (0.5 * FOV).tan())
    }

    /// Zoom to fit: the box [0, size] filling the view with FIT_ROOM to spare,
    /// turned as it is.
    fn framed(&self, size: [f32; 3], aspect: f32) -> Camera {
        let (f, s, u) = self.basis();
        let c = scale3(size, 0.5);
        let t = (0.5 * FOV).tan() / FIT_ROOM;
        let mut dist = 1e-3f32;
        for k in 0..8 {
            let p = [if k & 1 == 0 { 0.0 } else { size[0] }, if k & 2 == 0 { 0.0 } else { size[1] }, if k & 4 == 0 { 0.0 } else { size[2] }];
            let o = sub3(p, c);
            let z = dot3(o, f);
            dist = dist.max(dot3(o, s).abs() / (t * aspect) - z).max(dot3(o, u).abs() / t - z);
        }
        Camera { target: c, dist, ..*self }
    }

    /// Between `a` (k 0) and `b` (k 1): the target straight, the distance by ratio, the turn
    /// the shorter way.
    fn between(a: &Camera, b: &Camera, k: f32) -> Camera {
        let mix = |p: [f32; 3], q: [f32; 3]| add3(scale3(p, 1.0 - k), scale3(q, k));
        let dir = mix(a.dir, b.dir);
        let mut c = Camera { target: mix(a.target, b.target), dir, up: mix(a.up, b.up), dist: a.dist * (b.dist / a.dist).powf(k) };
        // Opposite ways have no shorter one: turned at once, half way.
        if dot3(dir, dir) < 1e-6 || dot3(c.up, c.up) < 1e-6 {
            (c.dir, c.up) = if k < 0.5 { (a.dir, a.up) } else { (b.dir, b.up) };
        }
        c.square();
        c
    }

    fn view_proj(&self, aspect: f32, extent: f32) -> M4 {
        let (eye, (f, s, u)) = (self.eye(), self.basis());
        let view = [[s[0], u[0], -f[0], 0.0], [s[1], u[1], -f[1], 0.0], [s[2], u[2], -f[2], 0.0], [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0]];
        let (near, far) = ((0.002 * self.dist).max(0.01), 4.0 * self.dist + 4.0 * extent);
        let t = 1.0 / (0.5 * FOV).tan();
        let proj = [[t / aspect, 0.0, 0.0, 0.0], [0.0, t, 0.0, 0.0], [0.0, 0.0, far / (near - far), -1.0], [0.0, 0.0, near * far / (near - far), 0.0]];
        mul(&proj, &view)
    }

    /// The view for the ray caster. Its rays are built from the eye and the camera's axes,
    /// not unprojected through the inverse view-projection: with the near plane this close,
    /// f32 rounding there turns a ray a few tenths of a milliradian, a pixel at full width.
    fn ray_view(&self, aspect: f32, extent: f32, light: [f32; 4]) -> RayView {
        let (eye, (f, s, u)) = (self.eye(), self.basis());
        let t = (0.5 * FOV).tan();
        RayView {
            view_proj: self.view_proj(aspect, extent),
            right: [s[0] * t * aspect, s[1] * t * aspect, s[2] * t * aspect, 0.0],
            up: [u[0] * t, u[1] * t, u[2] * t, 0.0],
            fwd: [f[0], f[1], f[2], 0.0],
            eye: [eye[0], eye[1], eye[2], 0.0],
            light,
            tool_at: [0.0; 4],
            tool_axis: [0.0; 4],
            tool_shape: [0.0; 4],
            heat: [0.0254, 1.0, 0.0, 0.0],
        }
    }
}

/// The rays of `r` (a view w x h) through the n x n pixels from (x0, y0), as a view of
/// their own: drawn into n x n pixels, each casts its ray in the whole view.
fn window(mut r: RayView, x0: f32, y0: f32, n: f32, (w, h): (f32, f32)) -> RayView {
    let (cx, cy) = ((2.0 * x0 + n) / w - 1.0, 1.0 - (2.0 * y0 + n) / h);
    for k in 0..3 {
        r.fwd[k] += cx * r.right[k] + cy * r.up[k];
        r.right[k] *= n / w;
        r.up[k] *= n / h;
    }
    r
}

/// The standard views, the way each looks and its
/// up, by the key held with Shift: 1 front, 2 back, 3 left, 4 right, 5 top, 6 bottom, 7
/// isometric.
fn standard_view(key: PhysicalKey) -> Option<([f32; 3], [f32; 3])> {
    let z = [0.0, 0.0, 1.0];
    let (t, i) = ((1.0f32 / 3.0).sqrt(), (1.0f32 / 6.0).sqrt());
    Some(match key {
        PhysicalKey::Code(KeyCode::Digit1) => ([0.0, 1.0, 0.0], z),
        PhysicalKey::Code(KeyCode::Digit2) => ([0.0, -1.0, 0.0], z),
        PhysicalKey::Code(KeyCode::Digit3) => ([1.0, 0.0, 0.0], z),
        PhysicalKey::Code(KeyCode::Digit4) => ([-1.0, 0.0, 0.0], z),
        PhysicalKey::Code(KeyCode::Digit5) => ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        PhysicalKey::Code(KeyCode::Digit6) => ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        PhysicalKey::Code(KeyCode::Digit7) => ([-t, t, -t], [-i, i, 2.0 * i]),
        _ => return None,
    })
}

/// What a drag does.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    None,
    Rotate,
    Turntable,
    Pan,
}

/// The mouse buttons and modifiers a binding combines.
const MMB: u8 = 1;
const RMB: u8 = 2;
const SHIFT: u8 = 4;
const ALT: u8 = 16;
const LMB: u8 = 32;

/// Buttons a trackpad has (no middle one): left drag turns about z, right, middle or Shift
/// + left drag pans; Alt + left drag rotates freely.
const BINDINGS: [(u8, Mode); 5] = [(LMB, Mode::Turntable), (ALT | LMB, Mode::Rotate), (RMB, Mode::Pan), (MMB, Mode::Pan), (SHIFT | LMB, Mode::Pan)];

/// Their `getManipulationMode` and `getClosestMode`: the binding held exactly, else the one
/// with the most flags the buttons and keys held contain, another than `current` first.
fn view_mode(id: u8, current: Mode) -> Mode {
    if let Some(&(_, m)) = BINDINGS.iter().find(|b| b.0 == id) {
        return m;
    }
    let closest = |skip: Option<Mode>| BINDINGS.iter().filter(|b| Some(b.1) != skip && id & b.0 == b.0).max_by_key(|b| b.0).map_or(Mode::None, |b| b.1);
    match closest(Some(current)) {
        Mode::None => closest(None),
        m => m,
    }
}

/// A drag of the view: its binding and mode (the mode waits until the mouse has moved past
/// CLICK_TRAVEL, so a click neither moves the view nor reads the GPU), where it turns or
/// pans about, the press and how far it traveled, and the last speed of a free rotate
/// (pixels a move) and when it last turned.
struct Drag {
    id: u8,
    mode: Mode,
    pending: Option<Mode>,
    pivot: Option<[f32; 3]>,
    press: (f32, f32),
    travel: f32,
    speed: (f32, f32),
    turned_at: Option<Instant>,
}

/// The mouse on the view: the buttons held, the modifiers, the drag under way, when the
/// middle button last clicked, a free rotate coasting (its speed, pivot, how much is left
/// and when it last moved), and the point the wheel zooms about while the cursor stays where
/// it was.
#[derive(Default)]
struct Nav {
    left: bool,
    right: bool,
    middle: bool,
    alt: bool,
    shift: bool,
    drag: Option<Drag>,
    middle_at: Option<Instant>,
    coast: Option<((f32, f32), [f32; 3], f32, Instant)>,
    wheel_at: Option<((f32, f32), [f32; 3], Camera)>,
}

impl Nav {
    fn id(&self) -> u8 {
        let mut id = 0;
        for (held, flag) in [(self.left, LMB), (self.right, RMB), (self.middle, MMB), (self.shift, SHIFT), (self.alt, ALT)] {
            if held {
                id |= flag;
            }
        }
        id
    }
}

/// The skin's pieces as vertices, positions from the box's low corner.
fn vertices(skin: &Skin, origin: V3) -> Vec<Vertex> {
    let raw = [0.50, 0.52, 0.56];
    let cut = [0.80, 0.82, 0.86];
    let mut out = Vec::with_capacity(skin.patches.len() * 3);
    for p in &skin.patches {
        let c = if p.src == BOX { raw } else { cut };
        for k in 0..3 {
            let q = p.p[k] - origin;
            out.push(Vertex { p: [q.x as f32, q.y as f32, q.z as f32], n: [p.n[k].x as f32, p.n[k].y as f32, p.n[k].z as f32], c });
        }
    }
    out
}

/// The part's levels its cache thread builds at once (`spawn_part_cache`): two, so one
/// lists on the CPU while the other waits on the GPU, and a new view waits on two at most.
const LEVELS_AT_ONCE: usize = 2;

/// The bricks of the part's levels the window last drew (by level and name, as
/// `PartCache` names them): the window writes it when its view settles, the part's cache
/// thread reads them exactly first.
static SEEN: std::sync::Mutex<Option<Arc<std::collections::HashSet<(i32, [i64; 3])>>>> = std::sync::Mutex::new(None);

/// The part's caches on a thread of their own: a big part's level takes seconds, which
/// the stock's worker must not wait on. It holds the levels the newest ask wants, coarse
/// first (the coarse ones are quick and cover the box), and drops the rest; the window
/// draws each level it had until the new one comes.
fn spawn_part_cache(mut pc: PartCache, proxy: EventLoopProxy<Load>) -> Sender<PartAsk> {
    let (tx, rx) = std::sync::mpsc::channel::<PartAsk>();
    std::thread::spawn(move || own_pool("part cache", move || {
        let mut sent: BTreeMap<i32, (V3, [i64; 3])> = BTreeMap::new();
        let Ok(mut want) = rx.recv() else {
            return;
        };
        loop {
            while let Ok(w) = rx.try_recv() {
                want = w;
            }
            for e in sent.keys().copied().filter(|e| !want.iter().any(|w| w.0 == *e)).collect::<Vec<_>>() {
                sent.remove(&e);
                let _ = proxy.send_event(Load::DropPart(e));
            }
            // The levels still to build, coarse first, LEVELS_AT_ONCE at once.
            let next: Vec<(i32, V3, f64, [i64; 3])> = want.iter().rev().filter(|w| sent.get(&w.0) != Some(&(w.1, w.3))).take(LEVELS_AT_ONCE).copied().collect();
            if next.is_empty() {
                // Nothing asked for: the bricks the window last drew with nodes bridged are
                // read exactly first (finest level first) and their levels sent again.
                let seen = SEEN.lock().ok().and_then(|s| s.clone());
                if let Some(seen) = seen.as_deref() {
                    let in_view = |e: i32| move |k: &[i64; 3]| seen.contains(&(e, *k));
                    if let Some(&(e, lo, h, nb)) = want.iter().filter(|w| sent.contains_key(&w.0) && pc.provisional_where(w.0, in_view(w.0)) > 0).min_by_key(|w| w.0) {
                        let clock = Instant::now();
                        let n = pc.provisional_where(e, in_view(e));
                        let cpu = pc.settle_where(e, in_view(e));
                        if let Some(level) = pc.relevel(e) {
                            println!("play: the part's cache in cells of {h} mm, its bricks in view settled in {:.2} s: {n} bridged nodes read exactly ({cpu} on the CPU)", clock.elapsed().as_secs_f64());
                            let _ = proxy.send_event(Load::PartLevel { e, lo, h, dims: nb.map(|n| n as u32), level: Arc::new(level) });
                        }
                        continue;
                    }
                }
                // Then a level sent with nodes bridged, finest first, is read exactly and sent
                // again.
                if let Some(&(e, lo, h, nb)) = want.iter().filter(|w| sent.contains_key(&w.0) && pc.provisional(w.0) > 0).min_by_key(|w| w.0) {
                    let clock = Instant::now();
                    let n = pc.provisional(e);
                    let cpu = pc.settle(e);
                    if let Some(level) = pc.relevel(e) {
                        println!("play: the part's cache in cells of {h} mm settled in {:.2} s: {n} bridged nodes read exactly ({cpu} on the CPU)", clock.elapsed().as_secs_f64());
                        let _ = proxy.send_event(Load::PartLevel { e, lo, h, dims: nb.map(|n| n as u32), level: Arc::new(level) });
                    }
                    continue;
                }
                match rx.recv() {
                    Ok(w) => want = w,
                    Err(_) => return,
                }
                continue;
            }
            // Built at once, so one lists on the CPU while another waits on the GPU; each goes
            // to the window as soon as it is done, and all are kept after.
            pc.trim();
            let built = std::sync::Mutex::new(Vec::new());
            {
                let pc = &pc;
                rayon::scope(|s| {
                    for &(e, lo, h, nb) in &next {
                        let (built, proxy) = (&built, proxy.clone());
                        s.spawn(move |_| {
                            let clock = Instant::now();
                            let (b, level) = pc.build(h, lo, nb);
                            println!("play: the part's cache in cells of {h} mm in {:.2} s: {} bricks with values, {} nodes bridged until settled", clock.elapsed().as_secs_f64(), level.atlas.len() / WORDS, b.bridged());
                            let _ = proxy.send_event(Load::PartLevel { e, lo, h, dims: nb.map(|n| n as u32), level: Arc::new(level) });
                            if let Ok(mut v) = built.lock() {
                                v.push((e, lo, nb, b));
                            }
                        });
                    }
                });
            }
            for (e, lo, nb, b) in built.into_inner().unwrap_or_default() {
                pc.keep(b);
                sent.insert(e, (lo, nb));
            }
        }
    }));
    tx
}

/// A STEP solid viewed alone (`Args::solid`): read, imported with its certified trim loops,
/// and cached for the part's view by its exact signed distance; the stock is the solid's
/// box padded past the colors' reach, uncut, so every hit on the part reads gray.
/// The largest solid of an assembly is shown.
#[allow(clippy::too_many_arguments)]
fn view_solid(file: &std::path::Path, h: f64, check_rays: bool, window: &WindowGpu, asks: &Receiver<Ask>, proxy: &EventLoopProxy<Load>, stage: &dyn Fn(&str)) -> Result<(), String> {
    stage("reading the part");
    let brep = icam_core::step::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let tol = brep.tolerance_mm.max(1e-6);
    stage(&format!("importing {} solid(s), certifying their trim loops", brep.solids.len()));
    let mut best: Option<icam_core::solid3::Solid3> = None;
    let mut refused = Vec::new();
    for sr in &brep.solids {
        match icam_core::solid3::Solid3::from_step(sr, tol) {
            Ok((sol, rep)) => {
                println!("play: solid #{}: {} faces, {} with certified loops ({} refused)", sr.id, sol.faces.len(), sol.faces.iter().filter(|f| f.trim2.is_some()).count(), rep.trim2_refused);
                if rep.off_tol_faces > 0 {
                    println!("play:   the file's edges lie off their surfaces beyond its declared {tol:.1e} mm on {} faces ({} points, up to {:.3e} mm); worst: {}", rep.off_tol_faces, rep.off_tol_points, rep.off_tol_max, rep.off_tol_notes.first().map_or("", String::as_str));
                }
                if best.as_ref().is_none_or(|b| sol.faces.len() > b.faces.len()) {
                    best = Some(sol);
                }
            }
            Err(e) => refused.push(format!("solid #{}: {e}", sr.id)),
        }
    }
    let sol = best.ok_or_else(|| format!("no solid in {} imports: {}", file.display(), refused.join("; ")))?;
    // The viewer keeps the solid for the life of the process.
    let sol: &'static icam_core::solid3::Solid3 = Box::leak(Box::new(sol));
    let ex = icam_core::solid3::Exact::new(sol);
    // Past the colors' reach (heat::REACH), so every hit on the part reads gray.
    let pad = (0.05 * (sol.hi - sol.lo).len()).max(crate::heat::REACH + 1.0);
    let b = Aabb::new(V3::new(sol.lo.x - pad, sol.lo.y - pad, sol.lo.z - pad), V3::new(sol.hi.x + pad, sol.hi.y + pad, sol.hi.z + pad));
    println!("play: the part's box {:.3},{:.3},{:.3} to {:.3},{:.3},{:.3}", sol.lo.x, sol.lo.y, sol.lo.z, sol.hi.x, sol.hi.y, sol.hi.z);
    let _ = proxy.send_event(Load::Box(b));
    let fields = icam_sweep::gpu::Fields::new(&[], &[])?;
    let mut stock = icam_sweep::Stock::new(b);
    let placer = spawn_placer(b, Vec::new(), Vec::new());
    let part = spawn_part_cache(PartCache::exact_gpu(ex, b.min, h), proxy.clone());
    stage("the part's view");
    serve(&mut stock, &fields, h, check_rays, (None, Some(part), placer, window), asks, proxy, stage)
}

/// The loader: reads the program, then applies its moves one at a time, handing the
/// window the skin every SHOW_EVERY seconds (and waiting while paused).
/// With --bake it places every move, then serves the window's views (`asks`): see `serve`.
fn spawn_load(args: &Args, proxy: EventLoopProxy<Load>, paused: Arc<AtomicBool>, start: Instant, asks: Receiver<Ask>, window: WindowGpu) {
    let (program, set, part, bbox, length, finest, limit, cpu, bake, check_rays, machine) = (args.program.clone(), args.set.clone(), args.part.clone(), args.bbox, args.length, args.finest, args.limit, args.cpu, args.bake, args.check, args.machine);
    let solid = args.solid.clone();
    std::thread::spawn(move || {
        let stage = |what: &str| {
            let msg = format!("{what} ({:.1} s)", start.elapsed().as_secs_f64());
            println!("play: {msg}");
            let _ = proxy.send_event(Load::Stage(msg));
        };
        let run = || -> Result<(), String> {
            if let Some(file) = &solid {
                return view_solid(file, bake.unwrap_or(0.25), check_rays, &window, &asks, &proxy, &stage);
            }
            stage("reading the program");
            let mut warnings = Vec::new();
            let moves = icam_post::read(&program, &mut warnings).map_err(|e| format!("{}: {e}", program.display()))?;
            let tools: std::collections::BTreeMap<i32, ToolDef> = icam_post::read_tool_table(&set, 25.4).map_err(|e| format!("{}: {e}", set.display()))?;
            let mut read = None;
            let b = match (bbox, &part) {
                (Some(b), _) => Aabb::new(V3::new(b[0], b[1], b[2]), V3::new(b[3], b[4], b[5])),
                (None, Some((file, place))) => {
                    stage("reading the part's box");
                    let r = ReadPart::read(file)?;
                    let (lo, hi) = r.bounds(place);
                    read = Some(r);
                    let m = tools.values().map(|t| t.diameter).fold(0.0, f64::max);
                    Aabb::new(V3::new(lo[0] - m, lo[1] - m, lo[2]), V3::new(hi[0] + m, hi[1] + m, hi[2]))
                }
                (None, None) => return Err("--part <step> or --box x0,y0,z0,x1,y1,z1 is required".into()),
            };
            println!("play: stock box {:.3},{:.3},{:.3} to {:.3},{:.3},{:.3} (--look is from its low corner)", b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z);
            let _ = proxy.send_event(Load::Box(b));
            let mut sweeps = icam_sweep::program::sweeps(&moves, &tools, length, b.max.z);
            if let Some(n) = limit {
                sweeps.truncate(n);
            }
            let total = sweeps.len();
            if let Some(h) = bake {
                stage("placing every move in the stock");
                let mut solids: Vec<Arc<icam_sweep::ToolSolid>> = Vec::new();
                for s in &sweeps {
                    if !solids.iter().any(|t| Arc::ptr_eq(t, &s.tool)) {
                        solids.push(s.tool.clone());
                    }
                }
                let fields = icam_sweep::gpu::Fields::new(&solids, &[])?;
                let t0 = Instant::now();
                let (vertices, spans, ops) = toolpath(&moves, &tools, &sweeps, b, length, machine);
                println!(
                    "play: the toolpath in {:.2} s: {} segments; the program runs {} on the machine's clock",
                    t0.elapsed().as_secs_f64(),
                    vertices.len() / 2,
                    clock_text(spans.last().map_or(0.0, |s| s.t1))
                );
                let _ = proxy.send_event(Load::Path { vertices, spans, ops });
                let t0 = Instant::now();
                let mut stock = icam_sweep::Stock::new(b);
                // The comparison builds its own stock on a thread of its own (regions.rs), and
                // so does the placer.
                let for_compare = part.as_ref().map(|_| sweeps.clone());
                let placer = spawn_placer(b, sweeps.clone(), solids.clone());
                let mut part_cache = None;
                for s in sweeps {
                    stock.apply(s);
                }
                println!("play: {} moves placed in {:.2} s", stock.len(), t0.elapsed().as_secs_f64());
                // The part's field for the colors (heat.rs).
                let heat = match &part {
                    Some((file, place)) => {
                        stage("compiling the part for the colors");
                        let r = match read.take() {
                            Some(r) => r,
                            None => ReadPart::read(file)?,
                        };
                        let frep = r.compile(place)?.0;
                        let (plo, phi) = r.bounds(place);
                        if let Some(sweeps) = for_compare {
                            let (proxy, solids, frep) = (proxy.clone(), solids.clone(), frep.clone());
                            std::thread::spawn(move || own_pool("comparison", move || {
                                let say = |m: &str| {
                                    let _ = proxy.send_event(Load::Comparing(format!("Comparing the part with the finished stock: {m}")));
                                };
                                let run = || -> Result<regions::Report, String> {
                                    let fields = icam_sweep::gpu::Fields::new(&solids, &[])?;
                                    let mut stock = icam_sweep::Stock::new(b);
                                    for s in sweeps {
                                        stock.apply(s);
                                    }
                                    regions::run(&frep, plo, phi, &stock, &fields, 0.5, &say)
                                };
                                let _ = proxy.send_event(match run() {
                                    Ok(r) => Load::Report(Arc::new(r)),
                                    Err(e) => Load::Comparing(format!("The comparison failed: {e}")),
                                });
                            }));
                        }
                        // The part's caches on a thread of their own: a big part's level takes
                        // seconds, which the stock's worker must not wait on. It holds the
                        // levels the newest ask wants, coarse first (the coarse ones are quick
                        // and cover the box), and drops the rest; the window draws each
                        // level it had until the new one comes.
                        if let Some(pc) = PartCache::new(frep.clone(), b.min) {
                            part_cache = Some(spawn_part_cache(pc, proxy.clone()));
                        }
                        let heat = Heat::new(frep, b.min);
                        if heat.is_none() {
                            println!("play: the part is not a union of columns, so the stock is drawn without its colors");
                        }
                        heat
                    }
                    None => None,
                };
                return serve(&mut stock, &fields, h, check_rays, (heat.as_ref(), part_cache, placer, &window), &asks, &proxy, &stage);
            }
            let mut skin = Skin::new(b, 1e-4, finest);
            if !cpu {
                stage("compiling the fields for the GPU");
                let mut solids: Vec<Arc<icam_sweep::ToolSolid>> = Vec::new();
                for s in &sweeps {
                    if !solids.iter().any(|t| Arc::ptr_eq(t, &s.tool)) {
                        solids.push(s.tool.clone());
                    }
                }
                match icam_sweep::gpu::Fields::new(&solids, &[]) {
                    Ok(f) => {
                        println!("play: fields on {}", f.adapter);
                        skin.gpu = Some(Arc::new(f));
                    }
                    Err(e) => println!("play: no GPU for the fields, the CPU reads them: {e}"),
                }
            }
            let clock = Instant::now();
            let _ = proxy.send_event(Load::Skin { vertices: vertices(&skin, b.min), done: 0, total, secs: 0.0 });
            let mut shown = Instant::now();
            for (k, s) in sweeps.into_iter().enumerate() {
                while paused.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                skin.apply(s);
                if shown.elapsed().as_secs_f64() >= SHOW_EVERY || k + 1 == total {
                    let _ = proxy.send_event(Load::Skin { vertices: vertices(&skin, b.min), done: k + 1, total, secs: clock.elapsed().as_secs_f64() });
                    shown = Instant::now();
                }
            }
            Ok(())
        };
        if let Err(e) = run() {
            let _ = proxy.send_event(Load::Failed(e));
        }
    });
}

/// The toolpath as line segments, every move's motion in pieces whose middle strays less
/// than 0.02 mm from the chord (a straight move is one), each vertex with the playhead at
/// which its move is done (a move that becomes a sweep, once it is in; any other, once the
/// sweeps before it are); every move on the machine's clock, at its programmed feed (rapids,
/// and feeds before any F, at the rapid rate); and the operations' names.
fn toolpath(moves: &[icam_post::Move], tools: &BTreeMap<i32, ToolDef>, sweeps: &[icam_sweep::Sweep], b: Aabb, length: f64, machine: Machine) -> (Vec<PathVertex>, Vec<Span>, Vec<String>) {
    fn pieces(m: &dyn icam_sweep::Motion, t0: f64, t1: f64, depth: u32, out: &mut Vec<V3>) {
        let (a, z) = (m.pose(t0).at, m.pose(t1).at);
        let mid = m.pose(0.5 * (t0 + t1)).at;
        let sag = mid - icam_sweep::scale(a + z, 0.5);
        if depth >= 10 || (depth > 0 || m.straight()) && (m.straight() || (sag.x * sag.x + sag.y * sag.y + sag.z * sag.z).sqrt() < 0.02) {
            out.push(z);
            return;
        }
        pieces(m, t0, 0.5 * (t0 + t1), depth + 1, out);
        pieces(m, 0.5 * (t0 + t1), t1, depth + 1, out);
    }
    let f = |v: V3| -> [f32; 3] { [(v.x - b.min.x) as f32, (v.y - b.min.y) as f32, (v.z - b.min.z) as f32] };
    let shape = |t: &icam_sweep::ToolSolid| -> [f32; 4] {
        use icam_core::tool::ProfEl;
        let (mut rc, mut hv) = (0.0, 0.0);
        match t.profile.els.first() {
            Some(ProfEl::Arc { rad, .. }) => rc = *rad,
            Some(ProfEl::Seg { h0, h1, .. }) if h1 > h0 => hv = h1 - h0,
            _ => {}
        }
        if let Some(ProfEl::Arc { rad, .. }) = t.profile.els.get(1) {
            rc = *rad;
        }
        [t.radius() as f32, rc as f32, hv as f32, t.length as f32]
    };
    let shapes: BTreeMap<i32, [f32; 4]> = tools.iter().map(|(&n, d)| (n, shape(&icam_sweep::ToolSolid::from_def(d, length)))).collect();
    let mut out = Vec::new();
    let mut spans: Vec<Span> = Vec::with_capacity(moves.len());
    let mut caps: Vec<f64> = Vec::with_capacity(moves.len());
    let mut ops: Vec<String> = Vec::new();
    let mut k = 0usize;
    for mv in moves {
        let cuts = icam_sweep::program::cuts(mv, tools, b.max.z);
        if cuts && k >= sweeps.len() {
            break;
        }
        // A sweep's own motion, so the tool drawn and the stock's tail run the same.
        let m = if cuts { sweeps[k].motion.clone() } else { icam_sweep::program::motion(mv) };
        let mut pts = vec![m.pose(0.0).at];
        pieces(&*m, 0.0, 1.0, 0, &mut pts);
        let done = if cuts { k + 1 } else { k } as f32;
        let rapid = if mv.kind == icam_post::MoveKind::Rapid { 1.0 } else { 0.0 };
        for w in pts.windows(2) {
            out.push(PathVertex { p: f(w[0]), done, rapid });
            out.push(PathVertex { p: f(w[1]), done, rapid });
        }
        caps.push(if mv.kind == icam_post::MoveKind::Rapid || mv.feed <= 0.0 { RAPID_MMPS } else { mv.feed / 60.0 });
        if ops.last() != Some(&mv.op) {
            ops.push(mv.op.clone());
        }
        let shape = shapes.get(&mv.tool).copied().unwrap_or([0.0; 4]);
        spans.push(Span { t0: 0.0, t1: 0.0, motion: m, shape, before: k, cuts, line: mv.line, op: (ops.len() - 1) as u32, knots: None });
        if cuts {
            k += 1;
        }
    }
    // Lines and arcs run at a steady rate, so their speed per unit t is their length.
    let programmed: f64 = spans.iter().zip(&caps).map(|(s, v)| s.motion.rate(0.0, 1.0).0 / v).sum();
    let motions: Vec<Arc<dyn icam_sweep::Motion>> = spans.iter().map(|s| s.motion.clone()).collect();
    let timed = machine_clock(&motions, &caps, machine);
    let mut clock = 0.0;
    for (s, (dt, knots)) in spans.iter_mut().zip(timed) {
        (s.t0, s.t1, s.knots) = (clock, clock + dt, knots);
        clock += dt;
    }
    println!("play: on the machine's clock {} ({} at the programmed feeds alone)", clock_text(clock), clock_text(programmed));
    (out, spans, ops)
}

/// The levels a view needs, finest first: (cell exponent, first node, bricks along each
/// axis). The finest has cells of about a pixel at the camera's target (between 0.7 and 1.4
/// of one: always under one cost up to 8 times the bricks for nothing to see), never finer
/// than 2^floor mm; each next one doubles the cell, up to the first that spans the whole box.
fn levels_for(b: Aabb, ask: &Ask, floor: i32) -> Vec<(i32, V3, [i64; 3])> {
    let pixel = ask.cam.dist as f64 * 2.0 * (0.5 * FOV as f64).tan() / ask.h.max(1) as f64;
    let at = b.min + V3::new(ask.cam.target[0] as f64, ask.cam.target[1] as f64, ask.cam.target[2] as f64);
    // Cells no coarser than a quarter of the box's thinnest side, unless no level that fine
    // holds the whole box: zoomed far out, coarser cells only lose what was cut (a sheet
    // thinner than a cell came and went in patches as the playhead moved on).
    let thin = (b.max.x - b.min.x).min(b.max.y - b.min.y).min(b.max.z - b.min.z);
    let mut coarsest = (0.25 * thin).log2().floor() as i32;
    while !icam_sweep::cache::place(b, 2f64.powi(coarsest), at, LEVEL_BRICKS).2 {
        coarsest += 1;
    }
    let first = (pixel.log2().round() as i32).min(coarsest).max(floor);
    let mut out = Vec::new();
    for e in first..first + MAX_LEVELS as i32 {
        let (lo, nb, whole) = icam_sweep::cache::place(b, 2f64.powi(e), at, LEVEL_BRICKS);
        out.push((e, lo, nb));
        if whole {
            break;
        }
    }
    out
}

/// The window's GPU, once it has one.
type WindowGpu = Arc<OnceLock<wgpu::Device>>;

/// A whole level's words in a buffer of the window's GPU, written on the caller's thread:
/// the brick map, the bricks, their cells that may hold surface, then the part's records. The window only copies them across
/// on the GPU (`Field::put`): writing a fine level there, tens of megabytes, held a frame
/// up 25 ms and more.
/// None, and the level is written by the window as before staging, when the GPU cannot
/// make the buffer (its memory is shared with the part's kernel and the desktop).
fn stage_level(dev: &wgpu::Device, c: &Cache, heat: Option<&(Vec<u32>, Vec<u32>)>) -> Option<wgpu::Buffer> {
    let parts = [c.map.as_slice(), c.atlas.as_slice(), c.cells.as_slice(), heat.map_or(&[][..], |t| t.1.as_slice())];
    let size: usize = parts.iter().map(|p| 4 * p.len()).sum();
    // Mappable, so the words go straight to memory the GPU copies from (a buffer of the GPU's
    // own, mapped at creation, is written through a hidden one, then copied again).
    let scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("level staged"), size: size.max(4) as u64, usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true });
    if let Some(e) = pollster::block_on(scope.pop()) {
        println!("play: a level of {:.0} MB could not be staged ({e}); the window writes it", size as f64 / 1e6);
        return None;
    }
    if let Ok(mut view) = buf.slice(..).get_mapped_range_mut() {
        let mut at = 0;
        for p in parts {
            let bytes: &[u8] = bytemuck::cast_slice(p);
            view.slice(at..at + bytes.len()).copy_from_slice(bytes);
            at += bytes.len();
        }
    }
    buf.unmap();
    Some(buf)
}

/// The cache's worker: for the newest view the window asked for, bakes the levels it needs
/// that the window does not hold, coarsest first (the whole stock is in view at once, then
/// sharpens), drops those it no longer needs, and says when all are in. A level held is
/// kept until the view's target moves an eighth of the level's side from where it was
/// placed; when only the playhead has moved, it is re-baked where the moves passed over
/// come near. Returns when the window closes.
#[allow(clippy::too_many_arguments)]
fn serve(stock: &mut icam_sweep::Stock, fields: &icam_sweep::gpu::Fields, finest: f64, check_rays: bool, (heat, part, placer, window): (Option<&Heat>, Option<Sender<PartAsk>>, Placer, &WindowGpu), asks: &Receiver<Ask>, proxy: &EventLoopProxy<Load>, stage: &dyn Fn(&str)) -> Result<(), String> {
    // The part's levels sent, by where each lies (its first node and bricks).
    let mut part_asked: PartAsk = Vec::new();
    // The coarsest of the part's levels: a cell's diagonal reaches past REACH, so a raw face
    // reads the part as far out as a cut face's records do (heat.rs).
    let reach_e = (crate::heat::REACH / 3f64.sqrt()).log2().ceil() as i32;
    let b = stock.bounds;
    // The part's records for a level's bricks (a patch's changed ones), when the part is
    // given.
    let colors = |c: &Cache| {
        heat.map(|t| {
            let clock = Instant::now();
            let slots: Vec<u32> = match &c.patch {
                Some(p) => p.placed.clone(),
                None => {
                    let mut s: Vec<u32> = c.map.iter().copied().filter(|&x| x < SKIP).collect();
                    s.sort_unstable();
                    s
                }
            };
            let r = t.records(c, &slots);
            println!("play: the part's colors for {} bricks in {:.3} s ({} columns at most to a brick)", slots.len(), clock.elapsed().as_secs_f64(), t.most.load(Ordering::Relaxed));
            Arc::new((slots, r))
        })
    };
    // To the window; a whole level staged on this thread (`stage`).
    let send = |e: i32, c: Arc<Cache>, heat: Option<Arc<(Vec<u32>, Vec<u32>)>>| {
        let staged = window.get().filter(|_| c.patch.is_none()).and_then(|d| stage_level(d, &c, heat.as_deref()));
        let _ = proxy.send_event(Load::Level { e, cache: c, heat, staged });
    };
    let total = stock.len();
    let floor = finest.log2().floor() as i32;
    let _ = proxy.send_event(Load::Moves(total));
    // The levels as the window holds them, and those asked of the placer (where each lies).
    let mut held: BTreeMap<i32, Arc<Cache>> = BTreeMap::new();
    let mut pending: BTreeMap<i32, (V3, [i64; 3])> = BTreeMap::new();
    let mut checked = false;
    let Ok(mut ask) = asks.recv() else {
        return Ok(());
    };
    let mut since = Instant::now();
    // The playhead this round of bakes is for. Every level held is brought to it before a
    // newer one is taken up: while the program plays, the playhead moves every frame, and
    // taking up each new one between levels kept the coarsest level baking over and over
    // while the finer ones (what the camera looks at) never caught up. A round also keeps
    // the levels drawn at one playhead.
    let mut round: Option<(usize, f64)> = None;
    loop {
        while let Ok(a) = asks.try_recv() {
            ask = a;
        }
        let mut wanted = ask.at.map_or((total, 0.0), |(k, part)| if k >= total { (total, 0.0) } else { (k, part) });
        // A move whose motion cannot be cut short is drawn whole or not at all.
        if wanted.1 > 0.0 && stock.sweep(wanted.0).motion.piece(0.0, 0.5).is_none() {
            wanted.1 = 0.0;
        }
        let plan = levels_for(b, &ask, floor);
        // A level the view no longer needs is dropped, but for the one just finer than it
        // needs: zooming back in finds it held, not baked again. It is kept at the playhead.
        let spare = plan[0].0 - 1;
        for e in held.keys().copied().filter(|e| !plan.iter().any(|w| w.0 == *e) && *e != spare).collect::<Vec<_>>() {
            held.remove(&e);
            let _ = proxy.send_event(Load::Drop(e));
        }
        pending.retain(|e, _| plan.iter().any(|w| w.0 == *e));
        // Where a held level lies still serves: same bricks, placed within an eighth of a side.
        let serves = |c: &Cache, lo: V3, nb: [i64; 3]| {
            let d = c.lo - lo;
            c.dims.map(|n| n as i64) == nb && d.x.abs().max(d.y.abs()).max(d.z.abs()) <= 0.999 * LEVEL_BRICKS as f64 * c.h
        };
        // A held level the new place keeps at least half the bricks of is scrolled there;
        // a farther one is placed from nothing, off this thread.
        let scrolls = |c: &Cache, lo: V3, nb: [i64; 3]| {
            let side = 8.0 * c.h;
            let d = c.lo - lo;
            let kept: f64 = [d.x, d.y, d.z].iter().zip(nb).map(|(&x, n)| (1.0 - (x / side).abs().round() / n as f64).max(0.0)).product();
            SCROLL.load(Ordering::Relaxed) && c.dims.map(|n| n as i64) == nb && kept >= 0.5
        };
        // The round: over the levels held only (a level being placed is the placer's).
        let behind = |r: (usize, f64)| held.values().any(|c| (c.upto, c.part) != r);
        // Paused, the newest playhead is taken at once (a level at a time, coarse first): the
        // window shows only the levels at it, so nothing stale is mixed in.
        let upto = match round {
            Some(r) if behind(r) && !ask.paused => r,
            _ => wanted,
        };
        round = Some(upto);
        stock.set_tail(upto.0, upto.1);
        let s: &icam_sweep::Stock = stock;
        // Levels the placer baked from nothing at a whole number of moves: brought to the
        // round's playhead here (a re-bake), then held in place of what stood there. One the
        // view has moved a little past still serves, as a level held does: while it pans, the
        // view's place for a fine level changes every brick, and a bake that had to match it
        // exactly was thrown away nearly every time.
        while let Ok((e, lo, nb, baked)) = placer.1.try_recv() {
            let c = baked?;
            if pending.get(&e) == Some(&(lo, nb)) {
                pending.remove(&e);
            }
            if !plan.iter().any(|w| w.0 == e && serves(&c, w.1, w.2)) {
                continue;
            }
            let clock = Instant::now();
            let mut c = icam_sweep::cache::rebake(s, fields, &c, upto)?;
            // The window never held the placer's bake: the level goes whole.
            c.patch = None;
            println!("play: level placed, brought to {} in {:.3} s: {}", at_text(upto), clock.elapsed().as_secs_f64(), c.report);
            let heat = colors(&c);
            let c = Arc::new(c);
            held.insert(e, c.clone());
            send(e, c, heat);
        }
        // Levels the view wants where none is held, or not here. A level held whose bricks the
        // new place still holds for the most part is scrolled here and brought to the round's
        // playhead in the same pass (`cache::scroll` reads only the side it gains and the boxes
        // the moves between come near), coarse first, one a round. The rest are asked of the
        // placer, coarse first. The level held, if any, stays and is kept at the playhead
        // meanwhile.
        let mut to_scroll = None;
        for &(e, lo, nb) in plan.iter().rev() {
            if held.get(&e).is_some_and(|c| serves(c, lo, nb)) || pending.get(&e) == Some(&(lo, nb)) {
                continue;
            }
            if held.get(&e).is_some_and(|c| scrolls(c, lo, nb)) {
                to_scroll = to_scroll.or(Some((e, lo, nb)));
                continue;
            }
            pending.insert(e, (lo, nb));
            let _ = placer.0.send((e, lo, nb, upto.0));
        }
        if let Some((e, lo, nb)) = to_scroll {
            let p = held[&e].clone();
            let clock = Instant::now();
            stage(&format!("scrolling cells of {} mm", p.h));
            let c = icam_sweep::cache::scroll(s, fields, &p, lo, nb, upto)?;
            println!("play: level scrolled to {} in {:.3} s: {}", at_text(upto), clock.elapsed().as_secs_f64(), c.report);
            // A placement the placer was asked for before is not wanted now.
            pending.remove(&e);
            let heat = colors(&c);
            let c = Arc::new(c);
            held.insert(e, c.clone());
            send(e, c, heat);
            continue;
        }
        // The part's caches (partcache.rs), for the part's view and the stock's raw faces: on
        // the grid of every level held, and on coarser levels holding the whole box out to
        // reach_e (a corner left uncut reads stock left, not far). Asked of its thread when
        // the levels held change; the part does not move, so the playhead never re-bakes them.
        if let Some(tx) = &part {
            let mut want: PartAsk = held.iter().map(|(&e, c)| (e, c.lo, c.h, c.dims.map(|n| n as i64))).collect();
            if let Some(&top) = held.keys().next_back() {
                for e in top + 1..=reach_e {
                    let (lo, nb, _) = icam_sweep::cache::place(b, 2f64.powi(e), b.min, LEVEL_BRICKS);
                    want.push((e, lo, 2f64.powi(e), nb));
                }
            }
            if want != part_asked {
                part_asked = want.clone();
                let _ = tx.send(want);
            }
        }
        // The levels held behind the round's playhead.
        let behind_now: Vec<i32> = held.iter().filter(|(_, c)| (c.upto, c.part) != upto).map(|(&e, _)| e).collect();
        // Only the playhead moved for every level behind: re-bake them all at once (the
        // round then costs about its slowest level, not their sum) and hand them over together.
        if behind_now.len() > 1 && !ask.paused {
            let clock = Instant::now();
            let from = (held[&behind_now[0]].upto, held[&behind_now[0]].part);
            stage(&format!("re-baking {} levels from {} to {}", behind_now.len(), at_text(from), at_text(upto)));
            let baked: Vec<Result<(i32, Cache), String>> = behind_now.par_iter().map(|&e| icam_sweep::cache::rebake(s, fields, &held[&e], upto).map(|c| (e, c))).collect();
            for r in baked {
                let (e, c) = r?;
                println!("play: level baked: {}", c.report);
                let heat = colors(&c);
                let c = Arc::new(c);
                held.insert(e, c.clone());
                send(e, c, heat);
            }
            println!("play: {} levels re-baked together in {:.3} s", behind_now.len(), clock.elapsed().as_secs_f64());
            continue;
        }
        // Coarse first.
        if let Some(&e) = behind_now.last() {
            let p = held[&e].clone();
            let clock = Instant::now();
            stage(&format!("re-baking cells of {} mm from {} to {}", p.h, at_text((p.upto, p.part)), at_text(upto)));
            let c = icam_sweep::cache::rebake(s, fields, &p, upto)?;
            println!("play: level baked in {:.2} s: {}", clock.elapsed().as_secs_f64(), c.report);
            let heat = colors(&c);
            // The window reads the same cache: shared, not copied.
            let c = Arc::new(c);
            held.insert(e, c.clone());
            send(e, c, heat);
            continue;
        }
        if !pending.is_empty() {
            // The placer is at work: wait on the view or on it, a little at a time.
            match asks.recv_timeout(std::time::Duration::from_millis(15)) {
                Ok(a) => ask = a,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
            }
            continue;
        }
        println!(
            "play: the view's {} levels in at {} of {total}, cells {} to {} mm, {:.2} s after it was asked for",
            plan.len(),
            at_text(upto),
            2f64.powi(plan[0].0),
            2f64.powi(plan[plan.len() - 1].0),
            since.elapsed().as_secs_f64()
        );
        if check_rays && !checked {
            checked = true;
            stage("checking the cache against the exact field");
            let clock = Instant::now();
            let levels: Vec<&Cache> = held.values().map(|c| &**c).collect();
            let r = check(s, fields, &levels, &ask)?;
            println!("play: check ({:.1} s): {r}", clock.elapsed().as_secs_f64());
            stage("checking scrolled levels against bakes from nothing");
            for line in scroll_check(s, fields, &held)? {
                println!("play: {line}");
            }
            if ask.at.is_some() {
                stage("checking re-bakes against bakes from nothing");
                for line in playhead_check(stock, fields, &held)? {
                    println!("play: {line}");
                }
            }
        }
        if upto != wanted {
            // The round is in; a newer playhead waits.
            continue;
        }
        let _ = proxy.send_event(Load::Settled);
        match asks.recv() {
            Ok(a) => {
                ask = a;
                since = Instant::now();
            }
            Err(_) => return Ok(()),
        }
    }
}

/// The scroll's gate (docs/SIM_VISUALIZATION_ENGINE.md, section 4.4): each level held,
/// placed anew a quarter of its side on along x and an eighth along y (as far as the box
/// allows) by `cache::scroll`, against the level baked there from nothing: no brick drawn
/// otherwise, values within the field's tolerance. Then the scrolled level taken back to
/// half its moves, against a bake there from nothing, which tests the search it carries; and
/// the level placed anew and moved to another playhead in one pass (back to half its moves,
/// and on a move), as the worker does while the program plays.
fn scroll_check(stock: &icam_sweep::Stock, fields: &icam_sweep::gpu::Fields, held: &BTreeMap<i32, Arc<Cache>>) -> Result<Vec<String>, String> {
    use icam_sweep::cache::{cache_at, place, rebake, scroll};
    let mut out = Vec::new();
    for c in held.values() {
        let nb = c.dims.map(|n| n as i64);
        let side = 8.0 * c.h;
        let to = c.lo + V3::new(0.75 * side * nb[0] as f64, 0.625 * side * nb[1] as f64, 0.5 * side * nb[2] as f64);
        let (lo, placed, _) = place(stock.bounds, c.h, to, LEVEL_BRICKS);
        if lo == c.lo || placed != nb {
            out.push(format!("cells of {} mm: the level spans the box, nothing to scroll", c.h));
            continue;
        }
        let close = |a: &Cache, b: &Cache| {
            let (otherwise, most) = a.apart(b);
            if otherwise == 0 && most <= c.tol + c.clamp / 32767.0 {
                format!("within {most:.6} mm")
            } else {
                format!("APART ({otherwise} bricks drawn otherwise, values up to {most:.6} mm apart)")
            }
        };
        let timed = |f: &dyn Fn() -> Result<Cache, String>| -> Result<(Cache, f64), String> {
            let clock = Instant::now();
            Ok((f()?, clock.elapsed().as_secs_f64()))
        };
        let (scrolled, t_scroll) = timed(&|| scroll(stock, fields, c, lo, nb, (c.upto, c.part)))?;
        let (scratch, t_scratch) = timed(&|| cache_at(stock, fields, c.h, lo, nb, (c.upto, c.part), c.tol))?;
        let half = (c.upto / 2, 0.0);
        let (back, t_back) = timed(&|| rebake(stock, fields, &scrolled, half))?;
        let back_scratch = cache_at(stock, fields, c.h, lo, nb, half, c.tol)?;
        // In one pass, as the worker does while the program plays: back, and on a move.
        let (both, t_both) = timed(&|| scroll(stock, fields, c, lo, nb, half))?;
        let next = ((c.upto + 1).min(stock.len()), 0.0);
        let (on, t_on) = timed(&|| scroll(stock, fields, c, lo, nb, next))?;
        let on_scratch = cache_at(stock, fields, c.h, lo, nb, next, c.tol)?;
        let wrong: usize = [&scrolled, &back, &both, &on].iter().map(|x| x.wrong_cells()).sum();
        let d = lo - c.lo;
        out.push(format!(
            "cells of {} mm: placed {:.0}, {:.0}, {:.0} bricks on, scrolled in {t_scroll:.3} s, from nothing {t_scratch:.3} s, {}; then back to {} in {t_back:.3} s, {}; placed and taken there at once in {t_both:.3} s, {}; placed and taken on to {} at once in {t_on:.3} s, {}; {wrong} bricks' cells that may hold surface wrong in them. Scrolled: {}",
            c.h,
            d.x / side,
            d.y / side,
            d.z / side,
            close(&scrolled, &scratch),
            at_text(half),
            close(&back, &back_scratch),
            close(&both, &back_scratch),
            at_text(next),
            close(&on, &on_scratch),
            scrolled.report
        ));
    }
    Ok(out)
}

/// The playhead's gate (docs/SIM_VISUALIZATION_ENGINE.md, step 3): for each level held,
/// baked at move n from nothing, bake it at the last move, re-bake that back to n, on to
/// n + 1 and back to n, then to m, the middle one of the moves that come into the level, and
/// on to m + 1 (a move that cuts in it), and compare every re-bake with the cache baked
/// there from nothing, word for word (forward steps: within the field's tolerance). Then
/// halfway along move m (the stock's tail): from m on to there, and from m + 1 back to
/// there, against the cache baked there from nothing.
fn playhead_check(stock: &mut icam_sweep::Stock, fields: &icam_sweep::gpu::Fields, held: &BTreeMap<i32, Arc<Cache>>) -> Result<Vec<String>, String> {
    use icam_sweep::cache::{cache_at, rebake};
    let total = stock.len();
    let mut out = Vec::new();
    let mut tails = Vec::new();
    for c in held.values() {
        let n = c.upto;
        let nb = c.dims.map(|x| x as i64);
        let stock: &icam_sweep::Stock = stock;
        let timed = |f: &dyn Fn() -> Result<Cache, String>| -> Result<(Cache, f64), String> {
            let clock = Instant::now();
            Ok((f()?, clock.elapsed().as_secs_f64()))
        };
        let scratch = |m: usize| timed(&|| cache_at(stock, fields, c.h, c.lo, nb, (m, 0.0), c.tol));
        let (end, t_end) = scratch(total)?;
        let (back, t_back) = timed(&|| rebake(stock, fields, &end, (n, 0.0)))?;
        let next = (n + 1).min(total);
        let (on, t_on) = timed(&|| rebake(stock, fields, &back, (next, 0.0)))?;
        let (on_scratch, t_on_scratch) = scratch(next)?;
        let (again, t_again) = timed(&|| rebake(stock, fields, &on, (n, 0.0)))?;
        let same = |a: &Cache, b: &Cache| if a == b { "the same".to_string() } else { format!("DIFFERENT ({})", a.diff(b)) };
        // Forward steps are within the field's tolerance of a bake from nothing (cache.rs).
        let close = |a: &Cache, b: &Cache| {
            let (otherwise, most) = a.apart(b);
            if otherwise == 0 && most <= c.tol + c.clamp / 32767.0 {
                format!("within {most:.6} mm")
            } else {
                format!("APART ({otherwise} bricks drawn otherwise, values up to {most:.6} mm apart)")
            }
        };
        let side = 8.0 * c.h;
        let mut inside = stock.near_in(Aabb::new(c.lo, c.lo + V3::new(side * nb[0] as f64, side * nb[1] as f64, side * nb[2] as f64)), 0..total);
        inside.sort_unstable();
        let m = inside.get(inside.len() / 2).copied().unwrap_or(n);
        let (to_m, t_to_m) = timed(&|| rebake(stock, fields, &again, (m, 0.0)))?;
        let (m_scratch, _) = scratch(m)?;
        let (past_m, t_past_m) = timed(&|| rebake(stock, fields, &to_m, (m + 1, 0.0)))?;
        let (past_scratch, t_past_scratch) = scratch(m + 1)?;
        let wrong: usize = [&back, &on, &again, &to_m, &past_m].iter().map(|x| x.wrong_cells()).sum();
        out.push(format!(
            "cells of {} mm: from nothing at move {total} {t_end:.2} s; re-baked to {n} in {t_back:.2} s, {} as from nothing; on to {next} in {t_on:.3} s (from nothing {t_on_scratch:.2} s), {}; back to {n} in {t_again:.3} s, {}; to {m} (of {} moves in the level) in {t_to_m:.2} s, {}; on to {} in {t_past_m:.3} s (from nothing {t_past_scratch:.2} s), {}; {wrong} bricks' cells that may hold surface wrong in the re-bakes",
            c.h,
            same(&back, c),
            close(&on, &on_scratch),
            same(&again, c),
            inside.len(),
            if m > n { close(&to_m, &m_scratch) } else { same(&to_m, &m_scratch) },
            m + 1,
            close(&past_m, &past_scratch)
        ));
        tails.push((c.clone(), m, to_m, past_scratch));
    }
    // Halfway along move m: the stock's tail is set once for all levels. Back from m + 1
    // starts from the bake there from nothing, so it must come out word for word.
    for (c, m, to_m, past_m) in tails {
        if m >= total || !stock.set_tail(m, 0.5) {
            continue;
        }
        let nb = c.dims.map(|x| x as i64);
        let close = |a: &Cache, b: &Cache| {
            let (otherwise, most) = a.apart(b);
            if otherwise == 0 && most <= c.tol + c.clamp / 32767.0 {
                format!("within {most:.6} mm")
            } else {
                format!("APART ({otherwise} bricks drawn otherwise, values up to {most:.6} mm apart)")
            }
        };
        let (on, line) = {
            let stock: &icam_sweep::Stock = stock;
            let clock = Instant::now();
            let half_scratch = cache_at(stock, fields, c.h, c.lo, nb, (m, 0.5), c.tol)?;
            let t_scratch = clock.elapsed().as_secs_f64();
            let clock = Instant::now();
            let on = rebake(stock, fields, &to_m, (m, 0.5))?;
            let t_on = clock.elapsed().as_secs_f64();
            let back = rebake(stock, fields, &past_m, (m, 0.5))?;
            let line = format!(
                "cells of {} mm, halfway along move {m}: from nothing {t_scratch:.2} s; on from {m} in {t_on:.3} s, {}; back from {}, {}",
                c.h,
                close(&on, &half_scratch),
                m + 1,
                if back == half_scratch { "the same".to_string() } else { format!("DIFFERENT ({})", back.diff(&half_scratch)) }
            );
            (on, line)
        };
        out.push(line);
        // On past the move's end: from halfway along move m to a third of the way along m + 1,
        // the rest of m as the stock's piece (stock.rs).
        if m + 1 < total && stock.set_tail(m + 1, 1.0 / 3.0) {
            let stock: &icam_sweep::Stock = stock;
            let scratch = cache_at(stock, fields, c.h, c.lo, nb, (m + 1, 1.0 / 3.0), c.tol)?;
            let clock = Instant::now();
            let past = rebake(stock, fields, &on, (m + 1, 1.0 / 3.0))?;
            out.push(format!("cells of {} mm, on from halfway along move {m} to a third along {}: in {:.3} s, {}", c.h, m + 1, clock.elapsed().as_secs_f64(), close(&past, &scratch)));
        }
    }
    Ok(out)
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    targets: (wgpu::TextureView, wgpu::TextureView),
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vertices: Option<(wgpu::Buffer, u32)>,
    /// The ray caster over the field's cache, once it is baked.
    field: Option<Field>,
    /// The toolpath, once read.
    path: Option<PathDraw>,
    /// The box about the region picked, as lines drawn with the toolpath's.
    marks: Option<(wgpu::Buffer, u32)>,
}

/// The toolpath's lines (path.wgsl): the pipeline, its uniform and bind group, the vertices
/// and how many.
struct PathDraw {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    count: u32,
}

impl PathDraw {
    fn new(dev: &wgpu::Device, vertices: &[PathVertex]) -> PathDraw {
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let uniform = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("path"), size: std::mem::size_of::<PathUniform>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("path"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }] });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("path"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("path"), source: wgpu::ShaderSource::Wgsl(include_str!("path.wgsl").into()) });
        let pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("path"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32] })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("path"), contents: bytemuck::cast_slice(if vertices.is_empty() { &[PathVertex { p: [0.0; 3], done: 0.0, rapid: 0.0 }; 2] } else { vertices }), usage: wgpu::BufferUsages::VERTEX });
        PathDraw { pipeline, uniform, bind, vertices: buf, count: vertices.len() as u32 }
    }
}

/// A storage buffer handed out in ranges of fixed-size items, first fit, and grown (its
/// contents copied over on the GPU) when a range does not fit.
struct Pool {
    buf: wgpu::Buffer,
    label: &'static str,
    /// Bytes to an item.
    item: u64,
    /// Items the buffer holds.
    cap: u64,
    /// The ranges in use, (first item, items), in order.
    used: Vec<(u64, u64)>,
}

impl Pool {
    fn buffer(dev: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
        dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
    }

    fn new(dev: &wgpu::Device, label: &'static str, item: u64, cap: u64) -> Pool {
        Pool { buf: Pool::buffer(dev, label, item * cap), label, item, cap, used: Vec::new() }
    }

    /// A range of n items: its first item, and whether the buffer was replaced to fit it.
    fn take(&mut self, dev: &wgpu::Device, queue: &wgpu::Queue, n: u64) -> (u64, bool) {
        let n = n.max(1);
        let (mut at, mut i) = (0, 0);
        while i < self.used.len() && self.used[i].0 < at + n {
            at = self.used[i].0 + self.used[i].1;
            i += 1;
        }
        let grow = at + n > self.cap;
        if grow {
            let clock = Instant::now();
            let cap = (2 * self.cap).max(at + n);
            let buf = Pool::buffer(dev, self.label, cap * self.item);
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(self.label) });
            enc.copy_buffer_to_buffer(&self.buf, 0, &buf, 0, self.cap * self.item);
            queue.submit([enc.finish()]);
            println!("play: the {} grew to {:.0} MB in {:.0} ms", self.label, (cap * self.item) as f64 / 1e6, 1e3 * clock.elapsed().as_secs_f64());
            (self.buf, self.cap) = (buf, cap);
        }
        self.used.insert(i, (at, n));
        (at, grow)
    }

    fn free(&mut self, first: u64) {
        self.used.retain(|r| r.0 != first);
    }

    /// As `take`, growing by half, and None, the pool as it was, when the GPU cannot
    /// allocate the larger buffer (its memory is shared with the part's kernel and the
    /// desktop).
    fn try_take(&mut self, dev: &wgpu::Device, queue: &wgpu::Queue, n: u64) -> Option<(u64, bool)> {
        let n = n.max(1);
        let (mut at, mut i) = (0, 0);
        while i < self.used.len() && self.used[i].0 < at + n {
            at = self.used[i].0 + self.used[i].1;
            i += 1;
        }
        let grow = at + n > self.cap;
        if grow {
            let clock = Instant::now();
            let cap = (self.cap + self.cap / 2).max(at + n);
            let scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let buf = Pool::buffer(dev, self.label, cap * self.item);
            if let Some(e) = pollster::block_on(scope.pop()) {
                println!("play: the {} could not grow to {:.0} MB ({e})", self.label, (cap * self.item) as f64 / 1e6);
                return None;
            }
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(self.label) });
            enc.copy_buffer_to_buffer(&self.buf, 0, &buf, 0, self.cap * self.item);
            queue.submit([enc.finish()]);
            println!("play: the {} grew to {:.0} MB in {:.0} ms", self.label, (cap * self.item) as f64 / 1e6, 1e3 * clock.elapsed().as_secs_f64());
            (self.buf, self.cap) = (buf, cap);
        }
        self.used.insert(i, (at, n));
        Some((at, grow))
    }
}

/// A level the ray caster holds: where it lies (box frame, cell side last), its bricks
/// along each axis, its ranges in the pools, and how many bricks hold values.
struct Held {
    lo: [f32; 4],
    dims: [u32; 3],
    map: u64,
    atlas: u64,
    bricks: usize,
    upto: usize,
    part: f64,
    /// The cache it holds (`Cache::id`), and the slots its atlas range has room for.
    id: u64,
    room: u64,
}

/// A level of the part's cache the field holds: its first node and cell side (the box's
/// frame), its bricks, and where its map and atlas ranges start.
struct PartHeld {
    lo: [f32; 4],
    dims: [u32; 3],
    map: u64,
    atlas: u64,
}

/// The ray caster over the field's cache (cache.wgsl), drawn as one triangle over the
/// screen, and the levels it reads. Each level's brick map and bricks sit in two pools, so
/// a level baked anew around the view replaces only its own ranges. The part's records
/// (heat.rs) sit beside the bricks, slot for slot.
struct Field {
    pipeline: wgpu::RenderPipeline,
    /// The caster's hits (`fs_hit`), for what lies under the mouse (`Field::hits`).
    pick: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind: wgpu::BindGroup,
    view: wgpu::Buffer,
    levels: wgpu::Buffer,
    map: Pool,
    atlas: Pool,
    /// The part's records, as many slots as the atlas holds.
    heat: wgpu::Buffer,
    /// Each brick's cells that may hold surface (`Cache::cells`), slot for slot.
    cells: wgpu::Buffer,
    heat_cap: u64,
    /// Whether any level came with the part's records: the caster colors the stock.
    tinted: bool,
    /// The part's caches (partcache.rs): brick maps and bricks, a range per level.
    pmap: Pool,
    patlas: Pool,
    /// The part's levels by cell exponent, finest first.
    parts: BTreeMap<i32, PartHeld>,
    /// The playhead wanted: when some levels are at it, only they are drawn (the others,
    /// still at an earlier one, would mix two moments in one picture).
    want: Option<(usize, f64)>,
    /// By cell exponent (cells of 2^e mm), so finest first.
    held: BTreeMap<i32, Held>,
    /// The stock box's size; the box frame starts at its low corner.
    size: [f32; 3],
}

/// What comes with a level to `Field::put`: the part's records (the slots listed, then
/// their words), and the level's words staged (`stage_level`).
type WithLevel<'a> = (Option<(&'a [u32], &'a [u32])>, Option<&'a wgpu::Buffer>);

impl Field {
    /// Drawn into a target of (format, samples, with a depth buffer), its fragments from
    /// `entry`.
    fn new(dev: &wgpu::Device, (format, samples, depth): (wgpu::TextureFormat, u32, bool), entry: &str, size: [f32; 3]) -> Field {
        let slot = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field"),
            entries: &[slot(0, wgpu::BufferBindingType::Uniform), slot(1, wgpu::BufferBindingType::Uniform), slot(2, storage), slot(3, storage), slot(4, storage), slot(5, storage), slot(6, storage), slot(7, storage)],
        });
        let uniform = |label: &str, size: usize| dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let view = uniform("field view", std::mem::size_of::<RayView>());
        let levels = uniform("field levels", std::mem::size_of::<LevelsUniform>());
        // Sized for a deep zoom from the start: a pool that grows mid-play takes a new buffer
        // of twice the size, and taking a few hundred megabytes of the GPU's memory held the
        // window up for most of half a second.
        let map = Pool::new(dev, "brick map", 4, 1 << 23);
        let atlas = Pool::new(dev, "brick atlas", (WORDS * 4) as u64, 1 << 17);
        let heat = Pool::buffer(dev, "brick colors", atlas.cap * (HEAT_WORDS * 4) as u64);
        let cells = Pool::buffer(dev, "brick cells", atlas.cap * (CELL_WORDS * 4) as u64);
        let pmap = Pool::new(dev, "part map", 4, 1 << 20);
        let patlas = Pool::new(dev, "part atlas", (WORDS * 4) as u64, 1 << 15);
        let bind = Field::bind(dev, &layout, [&view, &levels, &map.buf, &atlas.buf, &heat, &pmap.buf, &patlas.buf, &cells]);
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("field"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("field"), source: wgpu::ShaderSource::Wgsl(include_str!("cache.wgsl").into()) });
        let make = |(format, samples, depth): (wgpu::TextureFormat, u32, bool), entry: &str| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("field"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth.then(|| wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make((format, samples, depth), entry);
        let pick = make((wgpu::TextureFormat::Rgba32Float, 1, false), "fs_hit");
        let heat_cap = atlas.cap;
        Field { pipeline, pick, layout, bind, view, levels, map, atlas, heat, cells, heat_cap, tinted: false, pmap, patlas, parts: BTreeMap::new(), want: None, held: BTreeMap::new(), size }
    }

    /// The caster's hits for the rays of `r` over w x h pixels (`fs_hit`), row by row: each
    /// the point met (the box's frame) and how far along its ray, t below zero for none.
    /// Waits for the GPU; empty if it did not answer.
    fn hits(&self, dev: &wgpu::Device, queue: &wgpu::Queue, r: &RayView, w: u32, h: u32) -> Vec<[f32; 4]> {
        queue.write_buffer(&self.view, 0, bytemuck::bytes_of(r));
        let tex = dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("hits"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let tv = tex.create_view(&Default::default());
        let row = (w * 16).div_ceil(256) * 256;
        let buf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("hits"), size: (row * h) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hits"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &tv, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pick);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let done = queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        if dev.poll(wgpu::PollType::Wait { submission_index: Some(done), timeout: Some(std::time::Duration::from_secs(5)) }).is_err() {
            return Vec::new();
        }
        let Ok(data) = buf.slice(..).get_mapped_range() else {
            return Vec::new();
        };
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| bytemuck::pod_read_unaligned::<[f32; 4]>(&data[(y * row + x * 16) as usize..(y * row + x * 16 + 16) as usize])).collect()
    }

    fn rebind(&mut self, dev: &wgpu::Device) {
        self.bind = Field::bind(dev, &self.layout, [&self.view, &self.levels, &self.map.buf, &self.atlas.buf, &self.heat, &self.pmap.buf, &self.patlas.buf, &self.cells]);
    }

    fn bind(dev: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffers: [&wgpu::Buffer; 8]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("field"), layout, entries: &entries })
    }

    /// Holds level e, replacing any level e it held, from `c` (work frame; the box's low
    /// corner at `origin`) and the part's records for the slots listed (in order). A patch
    /// of the cache held for e writes only its changed bricks; a whole level staged
    /// (`stage_level`) is copied across on the GPU.
    fn put(&mut self, dev: &wgpu::Device, queue: &wgpu::Queue, e: i32, c: &Cache, (heat, staged): WithLevel, origin: V3) {
        let slots = (c.atlas.len() / WORDS) as u64;
        let record = (HEAT_WORDS * 4) as u64;
        // Runs of consecutive slots in an ordered list: (index in the list, first slot, length).
        let runs = |list: &[u32]| {
            let mut out: Vec<(usize, u32, usize)> = Vec::new();
            for (i, &s) in list.iter().enumerate() {
                match out.last_mut() {
                    Some(r) if r.1 as usize + r.2 == s as usize => r.2 += 1,
                    _ => out.push((i, s, 1)),
                }
            }
            out
        };
        if let (Some(patch), Some(h)) = (&c.patch, self.held.get_mut(&e)) {
            if patch.from == h.id && slots <= h.room {
                // The map's entries that changed only: a whole map is megabytes, every update.
                // Many scattered pieces go as the one span that holds them (the cache has every
                // entry and slot, so the span is right throughout): thousands of small writes
                // cost the window more than the bytes.
                let spans = |list: &[u32]| {
                    let r = runs(list);
                    if r.len() > 64 {
                        let (a, z) = (list[0], list[list.len() - 1]);
                        vec![(0, a, (z - a + 1) as usize)]
                    } else {
                        r
                    }
                };
                for (_, i, n) in spans(&patch.entries) {
                    let at = i as usize;
                    queue.write_buffer(&self.map.buf, (h.map + i as u64) * 4, bytemuck::cast_slice(&c.map[at..at + n]));
                }
                for (_, s, n) in spans(&patch.slots) {
                    let w = s as usize * WORDS;
                    queue.write_buffer(&self.atlas.buf, (h.atlas + s as u64) * self.atlas.item, bytemuck::cast_slice(&c.atlas[w..w + n * WORDS]));
                    let k = s as usize * CELL_WORDS;
                    queue.write_buffer(&self.cells, (h.atlas + s as u64) * (CELL_WORDS * 4) as u64, bytemuck::cast_slice(&c.cells[k..k + n * CELL_WORDS]));
                }
                if let Some((listed, recs)) = heat.filter(|t| !t.0.is_empty()) {
                    for (i, s, n) in runs(listed) {
                        queue.write_buffer(&self.heat, (h.atlas + s as u64) * record, bytemuck::cast_slice(&recs[i * HEAT_WORDS..(i + n) * HEAT_WORDS]));
                    }
                    self.tinted = true;
                }
                (h.id, h.bricks, h.upto, h.part) = (c.id, c.bricks(), c.upto, c.part);
                return;
            }
        }
        // A patch of the level held whose range has no room left: its unchanged bricks keep
        // their slots, and the patch brings only the changed bricks' records, so the rest are
        // carried over from the old range on the GPU.
        let carry = match (&c.patch, self.held.get(&e)) {
            (Some(p), Some(h)) if p.from == h.id => Some((h.atlas, h.room)),
            _ => None,
        };
        if c.patch.is_some() && carry.is_none() {
            eprintln!("icam-sim: a patch of level {e} came for a cache the window does not hold; its unchanged bricks' colors are missing");
        }
        let l = c.lo - origin;
        let lo = [l.x as f32, l.y as f32, l.z as f32, c.h as f32];
        if let Some(old) = self.held.remove(&e) {
            self.map.free(old.map);
            self.atlas.free(old.atlas);
        }
        // Room for the patches to come to add bricks without a new range.
        let room = slots + slots / 4 + 256;
        let (m, grew_map) = self.map.take(dev, queue, c.map.len() as u64);
        let (a, grew_atlas) = self.atlas.take(dev, queue, room);
        let grew_heat = self.atlas.cap > self.heat_cap;
        if grew_heat {
            // The records and the cells, slot for slot with the atlas, grow with it.
            let cell_bytes = (CELL_WORDS * 4) as u64;
            let buf = Pool::buffer(dev, "brick colors", self.atlas.cap * record);
            let cells = Pool::buffer(dev, "brick cells", self.atlas.cap * cell_bytes);
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("brick colors") });
            enc.copy_buffer_to_buffer(&self.heat, 0, &buf, 0, self.heat_cap * record);
            enc.copy_buffer_to_buffer(&self.cells, 0, &cells, 0, self.heat_cap * cell_bytes);
            queue.submit([enc.finish()]);
            (self.heat, self.cells, self.heat_cap) = (buf, cells, self.atlas.cap);
        }
        if grew_map || grew_atlas || grew_heat {
            self.rebind(dev);
        }
        // Through a buffer of its own: the old range and the new may overlap. Submitted before
        // the changed records are written, so those land over it.
        if let Some((old, old_room)) = carry.filter(|&(old, _)| old != a) {
            let bytes = old_room.min(room) * record;
            let moved = Pool::buffer(dev, "brick colors moved", bytes);
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("brick colors moved") });
            enc.copy_buffer_to_buffer(&self.heat, old * record, &moved, 0, bytes);
            enc.copy_buffer_to_buffer(&moved, 0, &self.heat, a * record, bytes);
            queue.submit([enc.finish()]);
        }
        let heat = heat.filter(|t| !t.0.is_empty());
        match staged.filter(|_| c.patch.is_none()) {
            Some(s) => {
                // The map, the bricks, their cells, then the records, as `stage_level` laid them
                // out.
                let (map_bytes, atlas_bytes, cell_bytes) = (4 * c.map.len() as u64, 4 * c.atlas.len() as u64, 4 * c.cells.len() as u64);
                let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("level staged") });
                enc.copy_buffer_to_buffer(s, 0, &self.map.buf, m * 4, map_bytes);
                if atlas_bytes > 0 {
                    enc.copy_buffer_to_buffer(s, map_bytes, &self.atlas.buf, a * self.atlas.item, atlas_bytes);
                    enc.copy_buffer_to_buffer(s, map_bytes + atlas_bytes, &self.cells, a * (CELL_WORDS * 4) as u64, cell_bytes);
                }
                if let Some((listed, _)) = heat {
                    for (i, slot, n) in runs(listed) {
                        enc.copy_buffer_to_buffer(s, map_bytes + atlas_bytes + cell_bytes + i as u64 * record, &self.heat, (a + slot as u64) * record, n as u64 * record);
                    }
                    self.tinted = true;
                }
                queue.submit([enc.finish()]);
            }
            None => {
                if let Some((listed, recs)) = heat {
                    for (i, s, n) in runs(listed) {
                        queue.write_buffer(&self.heat, (a + s as u64) * record, bytemuck::cast_slice(&recs[i * HEAT_WORDS..(i + n) * HEAT_WORDS]));
                    }
                    self.tinted = true;
                }
                queue.write_buffer(&self.map.buf, m * 4, bytemuck::cast_slice(&c.map));
                if !c.atlas.is_empty() {
                    queue.write_buffer(&self.atlas.buf, a * self.atlas.item, bytemuck::cast_slice(&c.atlas));
                    queue.write_buffer(&self.cells, a * (CELL_WORDS * 4) as u64, bytemuck::cast_slice(&c.cells));
                }
            }
        }
        self.held.insert(e, Held { lo, dims: c.dims, map: m, atlas: a, bricks: c.bricks(), upto: c.upto, part: c.part, id: c.id, room });
        self.upload(queue);
    }

    /// The part's cache in cells of 2^e mm, its first node and cell side at lo (the box's
    /// frame), dims bricks along each axis, in place of the one held for e.
    fn put_part(&mut self, dev: &wgpu::Device, queue: &wgpu::Queue, e: i32, (lo, dims): ([f32; 4], [u32; 3]), level: &PartLevel) {
        self.drop_part(e);
        let Some((m, grew_map)) = self.pmap.try_take(dev, queue, level.map.len() as u64) else { return };
        let Some((a, grew_atlas)) = self.patlas.try_take(dev, queue, (level.atlas.len() / WORDS) as u64) else {
            self.pmap.free(m);
            if grew_map {
                self.rebind(dev);
            }
            return;
        };
        if grew_map || grew_atlas {
            self.rebind(dev);
        }
        queue.write_buffer(&self.pmap.buf, m * 4, bytemuck::cast_slice(&level.map));
        if !level.atlas.is_empty() {
            queue.write_buffer(&self.patlas.buf, a * self.patlas.item, bytemuck::cast_slice(&level.atlas));
        }
        self.parts.insert(e, PartHeld { lo, dims, map: m, atlas: a });
        self.upload(queue);
    }

    fn drop_part(&mut self, e: i32) {
        if let Some(old) = self.parts.remove(&e) {
            self.pmap.free(old.map);
            self.patlas.free(old.atlas);
        }
    }

    fn drop_level(&mut self, queue: &wgpu::Queue, e: i32) {
        if let Some(old) = self.held.remove(&e) {
            self.map.free(old.map);
            self.atlas.free(old.atlas);
        }
        self.upload(queue);
    }

    /// The levels drawn: those at the playhead wanted when any is, else all.
    fn shown(&self) -> impl Iterator<Item = &Held> {
        let at = |h: &Held| Some((h.upto, h.part)) == self.want;
        let some = self.held.values().any(at);
        self.held.values().filter(move |h| !some || at(h))
    }

    fn upload(&self, queue: &wgpu::Queue) {
        let mut u = LevelsUniform::zeroed();
        u.box_hi = [self.size[0], self.size[1], self.size[2], 0.0];
        u.count[1] = self.tinted as u32;
        for (i, h) in self.shown().take(MAX_LEVELS).enumerate() {
            u.level[i] = LevelUniform { lo: h.lo, dims: [h.dims[0], h.dims[1], h.dims[2], h.map as u32], part: [0, h.atlas as u32, 0, 0] };
            u.count[0] = i as u32 + 1;
        }
        for (i, p) in self.parts.values().take(MAX_LEVELS).enumerate() {
            u.parts[i] = PartLevelUniform { lo: p.lo, dims: [p.dims[0], p.dims[1], p.dims[2], p.map as u32], at: [p.atlas as u32, 0, 0, 0] };
            u.count[2] = i as u32 + 1;
        }
        queue.write_buffer(&self.levels, 0, bytemuck::bytes_of(&u));
    }

    fn bricks(&self) -> usize {
        self.held.values().map(|h| h.bricks).sum()
    }

    /// The playhead the stock is drawn at: the earliest of its levels'.
    fn upto(&self) -> Option<(usize, f64)> {
        self.shown().map(|h| (h.upto, h.part)).min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// The finest and coarsest cells held, mm.
    fn cells(&self) -> (f32, f32) {
        let h = |x: Option<&Held>| x.map_or(0.0, |x| x.lo[3]);
        (h(self.held.values().next()), h(self.held.values().next_back()))
    }

    /// For --pan: the cell of the finest level holding p (the box's frame), mm (0 when none
    /// does), and how far p lies from the middle of the finest level, along x and y, as a
    /// share of half its side (past 1: outside it).
    fn at_target(&self, p: [f32; 3]) -> (f32, f32) {
        let side = |l: &Held, k: usize| 8.0 * l.lo[3] * l.dims[k] as f32;
        let cell = self.held.values().find(|l| (0..3).all(|k| p[k] >= l.lo[k] && p[k] < l.lo[k] + side(l, k))).map_or(0.0, |l| l.lo[3]);
        let off = self.held.values().next().map_or(f32::INFINITY, |l| (0..2).map(|k| (p[k] - l.lo[k] - 0.5 * side(l, k)).abs() / (0.5 * side(l, k))).fold(0.0, f32::max));
        (cell, off)
    }
}

/// Checks the drawing in two halves, neither tracing the exact field along rays: each
/// level of the cache against the exact field at every cell center
/// (icam_sweep::cache::check, on the GPU), and the ray caster against the cache over a grid
/// of the view's rays 320 wide: the GPU casts each ray (cache.wgsl, `fs_hit`), the CPU reads
/// the cached field (the finest level holding each point, as the caster does) at the hit,
/// where it should be zero, and along the ray before it, a quarter cell at a time, for any
/// crossing the caster skipped (only cache reads, no field queries). `levels` finest first.
fn check(stock: &icam_sweep::Stock, fields: &icam_sweep::gpu::Fields, levels: &[&Cache], ask: &Ask) -> Result<String, String> {
    let b = stock.bounds;
    let mut against_field = Vec::new();
    for c in levels {
        against_field.push(format!("cells of {} mm: {}", c.h, icam_sweep::cache::check(stock, fields, c, 1e-4)?));
    }
    let (w, h) = (320u32, ((320 * ask.h) as f64 / ask.w.max(1) as f64).round().max(1.0) as u32);
    let size = [(b.max.x - b.min.x) as f32, (b.max.y - b.min.y) as f32, (b.max.z - b.min.z) as f32];
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| format!("no GPU adapter: {e}"))?;
    let (dev, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("check"), required_features: wgpu::Features::empty(), required_limits: adapter.limits(), ..Default::default() })).map_err(|e| format!("no GPU device: {e}"))?;
    let mut f = Field::new(&dev, (wgpu::TextureFormat::Rgba32Float, 1, false), "fs_hit", size);
    for c in levels {
        f.put(&dev, &queue, c.h.log2().round() as i32, c, (None, None), b.min);
    }
    let extent = size[0].max(size[1]).max(size[2]);
    let r = ask.cam.ray_view(w as f32 / h as f32, extent, [0.0; 4]);
    let gpu = f.hits(&dev, &queue, &r, w, h);
    if gpu.is_empty() {
        return Err("the GPU did not cast the rays".into());
    }
    // The same rays in double precision, in the work frame.
    let v3 = |a: [f32; 4]| V3::new(a[0] as f64, a[1] as f64, a[2] as f64);
    let (o, right, up, fwd) = (v3(r.eye) + b.min, v3(r.right), v3(r.up), v3(r.fwd));
    // The cached field and its cell side at p: the finest level holding it, as the caster
    // reads it (outside every level: air).
    let read = |p: V3| levels.iter().find(|c| c.holds(p)).map(|c| (c.value(p, b), c.h));
    let grid_box = levels.iter().fold(None, |a: Option<Aabb>, c| {
        let side = c.h * 8.0;
        let g = Aabb::new(c.lo, c.lo + V3::new(side * c.dims[0] as f64, side * c.dims[1] as f64, side * c.dims[2] as f64));
        Some(a.map_or(g, |a| Aabb::new(V3::new(a.min.x.min(g.min.x), a.min.y.min(g.min.y), a.min.z.min(g.min.z)), V3::new(a.max.x.max(g.max.x), a.max.y.max(g.max.y), a.max.z.max(g.max.z)))))
    });
    let Some(grid_box) = grid_box else {
        return Err("no levels to check".into());
    };
    let rows: Vec<(bool, f64, f64, f64)> = (0..(w * h) as usize)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i as u32 % w) as f64, (i as u32 / w) as f64);
            let ndc = (2.0 * (x + 0.5) / w as f64 - 1.0, 1.0 - 2.0 * (y + 0.5) / h as f64);
            let d = icam_sweep::unit(fwd + icam_sweep::scale(right, ndc.0) + icam_sweep::scale(up, ndc.1));
            let g = grid_box;
            let inv = V3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
            let (a, z) = (V3::new((g.min.x - o.x) * inv.x, (g.min.y - o.y) * inv.y, (g.min.z - o.z) * inv.z), V3::new((g.max.x - o.x) * inv.x, (g.max.y - o.y) * inv.y, (g.max.z - o.z) * inv.z));
            let t0 = a.x.min(z.x).max(a.y.min(z.y)).max(a.z.min(z.z)).max(0.0);
            let t1 = a.x.max(z.x).min(a.y.max(z.y)).min(a.z.max(z.z));
            let hit = gpu[i];
            let th = if hit[3] < 0.0 { t1 } else { hit[3] as f64 };
            // Any cached material before the hit (more than half a cell before it)? The
            // deepest, in mm and in cells.
            let (mut skipped, mut cells) = (0f64, 0f64);
            let mut t = t0;
            loop {
                let (v, hh) = read(o + icam_sweep::scale(d, t)).unwrap_or((f64::INFINITY, levels[levels.len() - 1].h));
                // (Written so a ray that misses every level, its span NaN, stops too.)
                if t < th - 0.5 * hh {
                    if -v > skipped {
                        skipped = -v;
                        cells = -v / hh;
                    }
                    t += 0.25 * hh;
                } else {
                    break;
                }
            }
            let at = if hit[3] < 0.0 { 0.0 } else { read(V3::new(hit[0] as f64, hit[1] as f64, hit[2] as f64) + b.min).map_or(0.0, |x| x.0.abs()) };
            (hit[3] >= 0.0, skipped, cells, at)
        })
        .collect();
    let met = rows.iter().filter(|r| r.0).count();
    let skipped = rows.iter().filter(|r| r.2 > 1e-3).count();
    let deep = rows.iter().filter(|r| r.2 > 0.05).count();
    let deepest = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    let worst = rows.iter().map(|r| r.3).fold(0.0, f64::max);
    Ok(format!(
        "
  {}.
  {w} x {h} rays cast: {met} meet the stock, {skipped} pass cached material before their hit ({deep} deeper than a twentieth of a cell, the deepest {deepest:.4} mm), the cached field at the hits is within {worst:.5} mm of zero",
        against_field.join(".
  ")
    ))
}

fn targets(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::TextureView, wgpu::TextureView) {
    let make = |format: wgpu::TextureFormat| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("frame"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    (make(FORMAT), make(DEPTH))
}

impl Gfx {
    fn new(window: Arc<Window>) -> Result<Gfx, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).map_err(|e| format!("no surface for the window: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("icam-sim"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU device: {e}"))?;
        let size = window.inner_size();
        let mut config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).ok_or("the window's surface does not suit this GPU")?;
        config.format = FORMAT;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("view"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("skin"), source: wgpu::ShaderSource::Wgsl(include_str!("skin.wgsl").into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skin"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: 36, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3] })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("view"), size: std::mem::size_of::<ViewUniform>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("view"), layout: &layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }] });
        let frame = targets(&device, config.width, config.height);
        Ok(Gfx { window, surface, config, device, queue, targets: frame, pipeline, uniform, bind, vertices: None, field: None, path: None, marks: None })
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.targets = targets(&self.device, w, h);
    }
}

struct App {
    args: Args,
    /// The stock's levels and the part's the window holds (by exponent): with
    /// `Args::solid` and --shot, the picture waits for the part's level on the grid of the
    /// finest stock level, which is the one the part's view draws.
    stock_es: std::collections::BTreeSet<i32>,
    part_es: std::collections::BTreeSet<i32>,
    start: Instant,
    proxy: EventLoopProxy<Load>,
    paused: Arc<AtomicBool>,
    gfx: Option<Gfx>,
    /// The window's GPU, for the worker to stage whole levels in.
    gpu: WindowGpu,
    size: Option<[f32; 3]>,
    cam: Option<Camera>,
    stage: String,
    nav: Nav,
    cursor: (f32, f32),
    /// The rays the last frame cast (which surface, the tool), for what lies under the mouse.
    drawn: Option<RayView>,
    /// The part's levels taken in, and the camera and that count when the bricks in view
    /// were last written to `SEEN`, and when.
    parts_in: usize,
    seen_for: Option<(Camera, usize)>,
    seen_at: Option<Instant>,
    pieces: usize,
    done: bool,
    result: Result<(), String>,
    /// To the cache's worker: the views to bake levels around.
    asks: Option<Sender<Ask>>,
    /// The stock box, work frame.
    bounds: Option<Aabb>,
    /// With --bake: the playhead (moves done and how far along the next; None: every move),
    /// and how many moves there are.
    at: Option<(usize, f64)>,
    total: Option<usize>,
    /// The program on the machine's clock, the operations' names, the span of each sweep,
    /// and the playhead's time when there is a clock.
    spans: Vec<Span>,
    ops: Vec<String>,
    sweep_span: Vec<usize>,
    now: Option<f64>,
    /// Playing the program: from when, from which time on the clock, at how many times the
    /// machine's speed; when the title last showed the clock.
    playing: Option<(Instant, f64)>,
    speed: f64,
    told: Instant,
    /// The --scrub run: the playheads still to visit, when the last was asked for, how long
    /// each took to settle, and the time the window spent taking levels in.
    scrub: Option<Scrub>,
    put_secs: f64,
    /// Frames drawn to the window.
    frames: usize,
    /// When the panel asked to be drawn again (egui's repaint delay), if it did.
    repaint_at: Option<Instant>,
    /// The --play run: since when, and at every frame the playhead and the one the stock
    /// was drawn at (moves done), and how far behind the clock the stock was, seconds.
    play_run: Option<(Instant, Vec<Seen>)>,
    /// The comparison panel (ui.rs), with --bake once the window is up.
    panel: Option<Panel>,
    /// The comparison's report, its regions at the panel's tolerance (and that tolerance;
    /// below zero: not read yet), what the comparison is doing, and when each gouge was cut.
    report: Option<Arc<regions::Report>>,
    regions: Vec<Region>,
    regions_at: f32,
    comparing: String,
    marks: Vec<f64>,
    /// The camera flying (to a region, a standard view, level): since when, from, to, and
    /// over how many seconds.
    fly: Option<(Instant, Camera, Camera, f32)>,
    /// The tool's tip as last drawn, the box's frame.
    tip: Option<[f32; 3]>,
    /// The surface the view drew last (a change asks the worker for the part's cache).
    surface: ui::Surface,
    /// When on the clock the tool was last drawn (while playing it only moves on), and the
    /// frames of the --play run it stepped back on.
    tool_t: Option<f64>,
    tool_back: usize,
    /// The --play run's frames (when each was drawn), and when --zoom zoomed (or --pan began).
    frame_at: Vec<Instant>,
    zoomed: Option<Instant>,
    zooms: usize,
    /// --pan: where the camera's target stood when the pan began, and each frame since, what
    /// the window drew at the target (`Field::at_target`).
    pan_from: Option<[f32; 3]>,
    pan_seen: Vec<(f32, f32)>,
    /// --grab: the pictures taken.
    grabs: usize,
}

/// The panel: egui's context, its input from the window, its renderer, and its settings.
struct Panel {
    ctx: egui::Context,
    input: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    settings: ui::Settings,
}

/// A frame of the --play run: the playhead and the stock's (moves done), how far behind the
/// clock the stock was (s), and the stock's playhead.
type Seen = (usize, usize, f64, (usize, f64));

struct Scrub {
    todo: std::collections::VecDeque<usize>,
    asked: Instant,
    /// Each step: how many moves it crossed, and how long it took to settle.
    settled: Vec<(usize, f64)>,
    started: bool,
    /// The moves a regular step crosses.
    step: usize,
}

impl App {
    /// Tells the cache's worker the view has changed.
    fn ask(&self) {
        if let (Some(tx), Some(cam), Some(_)) = (&self.asks, self.cam, &self.gfx) {
            let v = self.viewport();
            let _ = tx.send(Ask { cam, w: v[2] as u32, h: v[3] as u32, at: self.at, paused: self.playing.is_none() });
        }
    }

    /// The view's rectangle in the window, pixels (x, y, width, height): what the panel
    /// leaves, or the whole window without one.
    fn viewport(&self) -> [f32; 4] {
        let (w, h) = self.gfx.as_ref().map_or((1.0, 1.0), |g| (g.config.width as f32, g.config.height as f32));
        match &self.panel {
            Some(p) if p.settings.view.is_positive() => {
                let (k, v) = (p.ctx.pixels_per_point(), p.settings.view);
                let x = (v.left() * k).clamp(0.0, w - 1.0);
                [x, 0.0, (v.width() * k).min(w - x).max(1.0), h]
            }
            Some(p) => {
                let x = (340.0 * p.ctx.pixels_per_point()).min(w - 1.0);
                [x, 0.0, w - x, h]
            }
            None => [0.0, 0.0, w, h],
        }
    }

    /// Play from the playhead (from the start when it stands at the end), or pause.
    fn toggle_play(&mut self) {
        self.playing = match self.playing {
            Some(_) => None,
            None => self.spans.last().map(|s| (Instant::now(), self.now.filter(|&t| t < s.t1).unwrap_or(0.0))),
        };
        if let Some((_, from)) = self.playing {
            self.seek_time(from);
        }
        self.want_at();
        self.title();
    }

    /// A move at a time on the clock: to the end of the move the tool is on, or back to its
    /// start (the one before's when already there); a hundred or a thousand; the ends.
    fn step(&mut self, k: NamedKey) {
        let n = self.spans.len();
        if n == 0 {
            return;
        }
        let end = self.spans[n - 1].t1;
        let now = self.now.unwrap_or(end);
        let i = self.spans.partition_point(|s| s.t1 <= now).min(n - 1);
        let back = if now > self.spans[i].t0 + 1e-9 { i } else { i.saturating_sub(1) };
        let to = match k {
            NamedKey::ArrowLeft => self.spans[back].t0,
            NamedKey::ArrowRight => self.spans[i].t1,
            NamedKey::ArrowDown => self.spans[i.saturating_sub(100)].t0,
            NamedKey::ArrowUp => self.spans[(i + 100).min(n - 1)].t1,
            NamedKey::PageDown => self.spans[i.saturating_sub(1000)].t0,
            NamedKey::PageUp => self.spans[(i + 1000).min(n - 1)].t1,
            NamedKey::Home => 0.0,
            _ => end,
        };
        self.playing = None;
        self.seek_time(to);
        self.title();
    }

    /// The camera fitted to the stock in the view.
    fn fit(&mut self) {
        if let Some(s) = self.size {
            let v = self.viewport();
            self.cam = Some(Camera::fit(s, v[2] / v[3]));
            self.ask();
        }
    }

    /// Flies the camera to `to` over `secs`, unless it is flying already (a view asked
    /// for while the camera moves is dropped).
    fn fly_to(&mut self, to: Camera, secs: f32) {
        if let (None, Some(cam)) = (self.fly, self.cam) {
            self.nav.coast = None;
            self.fly = Some((Instant::now(), cam, to, secs));
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
        }
    }

    /// Zoom to fit (F, or two middle clicks): the box filling the view, turned as
    /// it is.
    fn zoom_fit(&mut self) {
        if let (Some(cam), Some(s)) = (self.cam, self.size) {
            let v = self.viewport();
            self.fly_to(cam.framed(s, v[2] / v[3]), LONG_SECS);
        }
    }

    /// The rays as the last frame cast them (the part's view or the stock's, the tool), from
    /// the camera as it is now, and the view's size.
    fn rays(&self) -> Option<(RayView, (f32, f32))> {
        let (cam, s) = (self.cam?, self.size?);
        let vp = self.viewport();
        let mut r = cam.ray_view(vp[2] / vp[3], s[0].max(s[1]).max(s[2]), [0.0; 4]);
        if let Some(d) = &self.drawn {
            (r.light, r.tool_at, r.tool_axis, r.tool_shape, r.heat) = (d.light, d.tool_at, d.tool_axis, d.tool_shape, d.heat);
        }
        Some((r, (vp[2], vp[3])))
    }

    /// Once the view holds still (no drag, flight or coast) after the camera or the part's
    /// levels changed: the bricks of those levels its rays meet, a ray a 4 x 4 block of
    /// pixels, each in the finest level holding its point as the caster reads it, written to
    /// `SEEN` for the part's cache thread to settle first. At most every 0.3 s: when it is
    /// too soon, when to try again.
    fn publish_seen(&mut self) -> Option<Instant> {
        let cam = self.cam?;
        if self.nav.drag.is_some() || self.fly.is_some() || self.nav.coast.is_some() || self.seen_for == Some((cam, self.parts_in)) {
            return None;
        }
        if let Some(t) = self.seen_at.filter(|t| t.elapsed().as_secs_f32() < 0.3) {
            return Some(t + std::time::Duration::from_millis(300));
        }
        let (r, (w, h)) = self.rays()?;
        let f = self.gfx.as_ref().and_then(|g| g.field.as_ref()).filter(|f| !f.parts.is_empty())?;
        let hits = self.hits(&r, ((w / 4.0) as u32).max(1), ((h / 4.0) as u32).max(1));
        let mut seen = std::collections::HashSet::new();
        for p in hits.iter().filter(|p| p[3] >= 0.0) {
            for (&e, l) in &f.parts {
                let side = 8.0 * l.lo[3];
                let b: [f32; 3] = std::array::from_fn(|k| ((p[k] - l.lo[k]) / side).floor());
                if (0..3).all(|k| b[k] >= 0.0 && b[k] < l.dims[k] as f32) {
                    seen.insert((e, std::array::from_fn(|k| ((l.lo[k] + side * b[k]) / l.lo[3]).round() as i64)));
                    break;
                }
            }
        }
        if let Ok(mut s) = SEEN.lock() {
            *s = Some(Arc::new(seen));
        }
        (self.seen_for, self.seen_at) = (Some((cam, self.parts_in)), Some(Instant::now()));
        None
    }

    /// The caster's hits for `r` over w x h pixels (`Field::hits`).
    fn hits(&self, r: &RayView, w: u32, h: u32) -> Vec<[f32; 4]> {
        match self.gfx.as_ref() {
            Some(Gfx { device, queue, field: Some(f), .. }) => f.hits(device, queue, r, w, h),
            _ => Vec::new(),
        }
    }

    /// What is drawn under view pixel (x, y): of the pixels within SEARCH_RADIUS of it, the
    /// point the nearest meets (the box's frame).
    fn under(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        let (r, size) = self.rays()?;
        let n = 2 * SEARCH_RADIUS + 1;
        let (x0, y0) = (x.floor() - SEARCH_RADIUS as f32, y.floor() - SEARCH_RADIUS as f32);
        let off = |i: usize| (x0 + (i as u32 % n) as f32 + 0.5 - x).hypot(y0 + (i as u32 / n) as f32 + 0.5 - y);
        let hits = self.hits(&window(r, x0, y0, n as f32, size), n, n);
        hits.iter().enumerate().filter(|(_, p)| p[3] >= 0.0).min_by(|a, b| off(a.0).total_cmp(&off(b.0))).map(|(_, p)| [p[0], p[1], p[2]])
    }

    /// The pivot for a drag from view pixel (x, y), the first of: what is drawn under
    /// it; the middle of what is drawn in the view (over a grid of its rays, when MIN_SAMPLES
    /// meet something); the box's center, when the box fits in the view grown by
    /// ROTATE_BOUNDS_FACTOR; the point on the cursor's ray as deep as the box's center.
    fn pivot_at(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        if let Some(p) = self.under(x, y) {
            return Some(p);
        }
        let (cam, s, (r, (w, h))) = (self.cam?, self.size?, self.rays()?);
        let (gw, gh) = (SAMPLES_WIDE, ((SAMPLES_WIDE as f32 * h / w).round() as u32).max(1));
        let met: Vec<[f32; 4]> = self.hits(&r, gw, gh).into_iter().filter(|p| p[3] >= 0.0).collect();
        if met.len() >= MIN_SAMPLES {
            let sum = met.iter().fold([0.0; 3], |a, p| add3(a, [p[0], p[1], p[2]]));
            return Some(scale3(sum, 1.0 / met.len() as f32));
        }
        let c = scale3(s, 0.5);
        let (f, right, up) = cam.basis();
        let t = (0.5 * FOV).tan();
        let fits = (0..8).all(|k| {
            let o = sub3([if k & 1 == 0 { 0.0 } else { s[0] }, if k & 2 == 0 { 0.0 } else { s[1] }, if k & 4 == 0 { 0.0 } else { s[2] }], cam.eye());
            let z = dot3(o, f);
            z > 0.0 && dot3(o, right).abs() < ROTATE_BOUNDS_FACTOR * z * t * w / h && dot3(o, up).abs() < ROTATE_BOUNDS_FACTOR * z * t
        });
        if fits {
            return Some(c);
        }
        let (o, d) = cam.ray(x, y, (w, h));
        Some(add3(o, scale3(d, dot3(d, sub3(c, o)).max(0.01 * dot3(s, s).sqrt()))))
    }

    /// The point a wheel notch at view pixel (x, y) zooms about: what is drawn under it, else
    /// where its ray meets the plane through the box's center square to the view,
    /// else the point on it as deep as the target.
    fn zoom_point(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        let (cam, s, (_, size)) = (self.cam?, self.size?, self.rays()?);
        self.under(x, y).or_else(|| cam.on_plane(x, y, size, scale3(s, 0.5))).or_else(|| cam.on_plane(x, y, size, cam.target))
    }

    /// Whether the camera may zoom by k about p: out while the box is
    /// more than MIN_MODEL_PIXELS across, in while a pixel at p is more than MIN_PIXEL.
    fn can_zoom(&self, k: f32, p: [f32; 3]) -> bool {
        let (Some(cam), Some(s)) = (self.cam, self.size) else {
            return false;
        };
        let h = self.viewport()[3];
        let depth = |q: [f32; 3]| dot3(sub3(q, cam.eye()), cam.dir);
        if k > 1.0 {
            let d = depth(scale3(s, 0.5));
            return d <= 0.0 || dot3(s, s).sqrt() * Camera::pixels_per_mm(d * k, h) > MIN_MODEL_PIXELS;
        }
        1.0 / Camera::pixels_per_mm(depth(p) * k, h) > MIN_PIXEL
    }

    /// A wheel's step at the cursor (above zero zooms in): the view scaled by 2^(-step/6)
    /// about the point under the cursor, which stays under it. True when it zoomed.
    fn wheel(&mut self, step: f32) -> bool {
        let vp = self.viewport();
        let (x, y) = (self.cursor.0 - vp[0], self.cursor.1 - vp[1]);
        if step == 0.0 || self.fly.is_some() || x < 0.0 || y < 0.0 || x >= vp[2] || y >= vp[3] {
            return false;
        }
        // The point stays under the cursor while neither moves but by the wheel: not read again.
        let p = match self.nav.wheel_at {
            Some((at, p, c)) if at == self.cursor && Some(c) == self.cam => Some(p),
            _ => self.zoom_point(x, y),
        };
        let (Some(p), Some(mut cam)) = (p, self.cam) else {
            return false;
        };
        let k = 2f32.powf(-step / STEPS_PER_DOUBLING);
        if !self.can_zoom(k, p) {
            return false;
        }
        self.nav.coast = None;
        cam.zoom_at(p, k);
        self.cam = Some(cam);
        self.nav.wheel_at = Some((self.cursor, p, cam));
        true
    }

    /// A view button pressed over the view: a drag starts, its mode waiting for the mouse to
    /// move.
    fn press(&mut self) {
        let id = self.nav.id();
        if let Some(d) = self.nav.drag.as_mut() {
            // Another button joined the drag.
            d.id = id;
            return;
        }
        let mode = view_mode(id, Mode::None);
        if mode != Mode::None {
            self.nav.coast = None;
            self.nav.drag = Some(Drag { id: 0, mode: Mode::None, pending: Some(mode), pivot: None, press: self.cursor, travel: 0.0, speed: (0.0, 0.0), turned_at: None });
        }
    }

    /// A drag's mode starting, or changing mid-drag:
    /// the two rotates share a pivot; anything else picks one at window pixel `at`.
    fn enter(&self, d: &mut Drag, mode: Mode, at: (f32, f32)) {
        let previous = d.mode;
        d.mode = mode;
        if mode == previous {
            return;
        }
        if !matches!((mode, previous), (Mode::Rotate, Mode::Turntable) | (Mode::Turntable, Mode::Rotate)) {
            let vp = self.viewport();
            d.pivot = self.pivot_at(at.0 - vp[0], at.1 - vp[1]);
        }
        if mode == Mode::Rotate {
            (d.turned_at, d.speed) = (Some(Instant::now()), (0.0, 0.0));
        }
    }

    /// The mouse moved by (dx, dy) to the cursor during a drag. True when the view moved.
    fn drag_by(&mut self, dx: f32, dy: f32) -> bool {
        let Some(mut d) = self.nav.drag.take() else {
            return false;
        };
        d.travel += dx.abs() + dy.abs();
        let id = self.nav.id();
        let (mut dx, mut dy) = (dx, dy);
        if let Some(mode) = d.pending {
            // Still a click: nothing moves yet. Past it, the drag starts at the press and
            // catches up with the mouse.
            if d.travel <= CLICK_TRAVEL {
                self.nav.drag = Some(d);
                return false;
            }
            let press = d.press;
            d.pending = None;
            self.enter(&mut d, mode, press);
            (dx, dy) = (self.cursor.0 - press.0, self.cursor.1 - press.1);
        } else if id != d.id {
            let mode = view_mode(id, d.mode);
            if mode != Mode::None {
                self.enter(&mut d, mode, self.cursor);
            }
        }
        d.id = id;
        let vp = self.viewport();
        let size = (vp[2], vp[3]);
        let mut moved = false;
        if let Some(cam) = self.cam {
            let pivot = d.pivot.unwrap_or(cam.target);
            match d.mode {
                // While the camera flies (or levels) the mouse is not followed.
                Mode::Rotate | Mode::Turntable if self.fly.is_some() => {}
                Mode::Turntable if cam.roll().abs() >= LEVEL_SNAP => {
                    let secs = if cam.roll().abs() > std::f32::consts::FRAC_PI_4 { SHORT_SECS } else { SNAP_SECS };
                    self.fly_to(cam.leveled(pivot), secs);
                }
                Mode::Rotate | Mode::Turntable => {
                    let mut c = if d.mode == Mode::Turntable { cam.leveled(pivot) } else { cam };
                    c.drag(dx, dy, size, pivot, d.mode == Mode::Rotate);
                    self.cam = Some(c);
                    if d.mode == Mode::Rotate {
                        (d.turned_at, d.speed) = (Some(Instant::now()), (0.5 * (d.speed.0 + dx), 0.5 * (d.speed.1 + dy)));
                    }
                    moved = true;
                }
                // The point under the cursor stays under it: the camera moves by what lies
                // between the two cursors' rays on the plane through the pivot.
                Mode::Pan => {
                    let at = |x: f32, y: f32| cam.on_plane(x - vp[0], y - vp[1], size, pivot);
                    if let (Some(a), Some(b)) = (at(self.cursor.0 - dx, self.cursor.1 - dy), at(self.cursor.0, self.cursor.1)) {
                        self.cam = Some(Camera { target: sub3(cam.target, sub3(b, a)), ..cam });
                        moved = true;
                    }
                }
                Mode::None => {}
            }
        }
        self.nav.drag = Some(d);
        moved
    }

    /// A view button let go. When none is still held the drag ends: a free rotate still
    /// moving coasts on, and two middle clicks zoom to fit.
    fn release(&mut self, middle: bool) {
        if self.nav.left || self.nav.right || self.nav.middle {
            return;
        }
        let Some(d) = self.nav.drag.take() else {
            return;
        };
        if d.mode == Mode::Rotate && d.turned_at.is_some_and(|t| t.elapsed().as_secs_f32() < INERTIA_WINDOW) {
            if let Some(p) = d.pivot.or(self.cam.map(|c| c.target)) {
                self.nav.coast = Some((d.speed, p, 1.0, Instant::now()));
            }
        }
        if !(middle && d.travel <= CLICK_TRAVEL) {
            self.nav.middle_at = None;
        } else if self.nav.middle_at.is_some_and(|t| t.elapsed().as_secs_f32() <= MIDDLE_DOUBLE_CLICK) {
            self.nav.middle_at = None;
            self.zoom_fit();
        } else {
            self.nav.middle_at = Some(Instant::now());
        }
    }

    /// The regions at the panel's tolerance, when it changed or the report came in; picking
    /// is cleared then (the list is another).
    fn sync_regions(&mut self, s: &mut ui::Settings) {
        let Some(r) = self.report.clone() else {
            return;
        };
        if self.regions_at >= 0.0 && (s.tol - self.regions_at).abs() < 1e-7 {
            return;
        }
        self.regions = r.regions(s.tol as f64);
        self.regions_at = s.tol;
        self.marks = self.regions.iter().filter_map(|g| g.by).filter_map(|m| self.sweep_span.get(m).map(|&i| self.spans[i].t1)).collect();
        s.selected = None;
        if let Some(g) = self.gfx.as_mut() {
            g.marks = None;
        }
    }

    /// Picks region i (None: none): its box drawn, the camera flown to it.
    fn select(&mut self, i: Option<usize>) {
        if let Some(p) = self.panel.as_mut() {
            p.settings.selected = i;
        }
        let r = i.and_then(|i| self.regions.get(i)).cloned();
        let (Some(b), Some(cam)) = (self.bounds, self.cam) else {
            return;
        };
        if let Some(g) = self.gfx.as_mut() {
            g.marks = r.as_ref().map(|r| {
                let lo: Vec<f32> = (0..3).map(|a| (r.lo[a] - 0.5 - [b.min.x, b.min.y, b.min.z][a]) as f32).collect();
                let hi: Vec<f32> = (0..3).map(|a| (r.hi[a] + 0.5 - [b.min.x, b.min.y, b.min.z][a]) as f32).collect();
                let corner = |k: usize| [if k & 1 == 0 { lo[0] } else { hi[0] }, if k & 2 == 0 { lo[1] } else { hi[1] }, if k & 4 == 0 { lo[2] } else { hi[2] }];
                let mut v = Vec::new();
                for (a, z) in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] {
                    v.push(PathVertex { p: corner(a), done: 0.0, rapid: 2.0 });
                    v.push(PathVertex { p: corner(z), done: 0.0, rapid: 2.0 });
                }
                (g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("region"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX }), v.len() as u32)
            });
        }
        if let Some(r) = r {
            let size = ((r.hi[0] - r.lo[0]).powi(2) + (r.hi[1] - r.lo[1]).powi(2) + (r.hi[2] - r.lo[2]).powi(2)).sqrt() as f32;
            let target = [(r.center[0] - b.min.x) as f32, (r.center[1] - b.min.y) as f32, (r.center[2] - b.min.z) as f32];
            self.fly = Some((Instant::now(), cam, Camera { target, dist: (3.5 * size).max(60.0), ..cam }, 0.45));
        }
    }

    /// Does what the panel asked.
    fn act(&mut self, a: Act) {
        let end = self.spans.last().map_or(0.0, |s| s.t1);
        let now = self.now.unwrap_or(end);
        match a {
            Act::Toggle => self.toggle_play(),
            Act::Seek(t) => {
                if self.playing.is_some() {
                    self.playing = Some((Instant::now(), t));
                }
                self.seek_time(t);
            }
            Act::Step(k) => self.step(if k < 0 { NamedKey::ArrowLeft } else { NamedKey::ArrowRight }),
            Act::Speed(v) => {
                self.speed = if v > 0.0 { v } else { ((end - now) / 60.0).max(1.0) };
                if self.playing.is_some() {
                    self.playing = Some((Instant::now(), now));
                }
            }
            Act::Select(i) => self.select(i),
            Act::Moment(i) => self.moment(i),
            Act::Reset => self.fit(),
        }
        self.title();
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }

    /// One frame: the panel's pass (it says where the view is and what to do), the field in
    /// the view, then the panel over it.
    fn frame(&mut self, ms: &(wgpu::TextureView, wgpu::TextureView), target: &wgpu::TextureView, w: u32, h: u32) {
        let Some(mut p) = self.panel.take() else {
            self.draw(ms, target);
            return;
        };
        self.sync_regions(&mut p.settings);
        let raw = match &self.gfx {
            Some(g) => p.input.take_egui_input(&g.window),
            None => egui::RawInput::default(),
        };
        let mut acts = Vec::new();
        let mut out = {
            let cut = |m: usize| -> Option<(usize, String, f64)> {
                let s = &self.spans[*self.sweep_span.get(m)?];
                Some((s.line, self.ops.get(s.op as usize).cloned().unwrap_or_default(), s.t1))
            };
            let end = self.spans.last().map_or(0.0, |s| s.t1);
            let now = self.now.unwrap_or(end);
            let line = {
                let i = self.spans.partition_point(|s| s.t1 <= now).min(self.spans.len().saturating_sub(1));
                self.spans.get(i).map(|s| (s.line, self.ops.get(s.op as usize).map_or("", String::as_str)))
            };
            let name = |f: &std::path::Path| f.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
            let program = name(&self.args.program);
            let part = self.args.part.as_ref().map(|(f, _)| name(f));
            let info = ui::Info {
                program: &program,
                part: part.as_deref(),
                comparing: &self.comparing,
                regions: self.report.as_ref().map(|_| self.regions.as_slice()),
                now,
                end,
                playing: self.playing.is_some(),
                speed: self.speed,
                line,
                marks: &self.marks,
                cut: &cut,
                part_view: self.gfx.as_ref().and_then(|g| g.field.as_ref()).is_some_and(|f| f.tinted),
            };
            let settings = &mut p.settings;
            p.ctx.run_ui(raw, |root| acts = ui::show(root, settings, &info))
        };
        if let Some(g) = &self.gfx {
            p.input.handle_platform_output(&g.window, std::mem::take(&mut out.platform_output));
            match out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay) {
                Some(d) if d.is_zero() => g.window.request_redraw(),
                // A repaint the panel wants later (a tooltip's delay, say): woken for then.
                Some(d) if d < std::time::Duration::from_secs(3600) => {
                    let at = Instant::now() + d;
                    self.repaint_at = Some(self.repaint_at.map_or(at, |t| t.min(at)));
                }
                _ => {}
            }
        }
        let surface = p.settings.surface;
        self.panel = Some(p);
        if surface != self.surface {
            self.surface = surface;
            self.ask();
        }
        for a in acts {
            self.act(a);
        }
        let t_draw = Instant::now();
        self.draw(ms, target);
        let drew = t_draw.elapsed().as_secs_f64();
        if self.play_run.is_some() && drew > 0.012 {
            println!("play: a frame's draw took {:.0} ms", 1e3 * drew);
        }
        let t_paint = Instant::now();
        let (Some(p), Some(g)) = (self.panel.as_mut(), self.gfx.as_ref()) else {
            return;
        };
        // egui turning its own panel shapes (text, widgets) into triangles: UI, not geometry.
        let jobs = egui::Context::tessellate(&p.ctx, out.shapes, out.pixels_per_point);
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                p.renderer.update_texture(&g.device, &g.queue, *id, delta);
            }
        }
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: out.pixels_per_point };
        let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("panel") });
        let extra = p.renderer.update_buffers(&g.device, &g.queue, &mut enc, &jobs, &sd);
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("panel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: target, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            p.renderer.render(&mut pass.forget_lifetime(), &jobs, &sd);
        }
        g.queue.submit(extra.into_iter().chain([enc.finish()]));
        for id in &out.textures_delta.free {
            p.renderer.free_texture(id);
        }
        if self.play_run.is_some() && t_paint.elapsed().as_secs_f64() > 0.012 {
            println!("play: a frame's panel took {:.0} ms ({} textures set)", 1e3 * t_paint.elapsed().as_secs_f64(), out.textures_delta.set.len());
        }
    }

    fn title(&self) {
        if let Some(g) = &self.gfx {
            let pause = if self.paused.load(Ordering::Relaxed) { "  |  paused (Space)" } else { "" };
            let speed = if self.speed >= 10.0 { format!("{:.0}x", self.speed) } else { format!("{}x", self.speed) };
            let play = match (self.now, self.spans.last()) {
                (Some(now), Some(last)) => {
                    let i = self.spans.partition_point(|s| s.t1 <= now).min(self.spans.len() - 1);
                    let s = &self.spans[i];
                    format!(
                        "  |  {} of {}, {}{}  |  line {}, {}",
                        clock_text(now),
                        clock_text(last.t1),
                        if self.playing.is_some() { "playing at " } else { "Space plays at " },
                        speed + " (R the machine's speed, P the rest in a minute, [ ])",
                        s.line,
                        self.ops.get(s.op as usize).map_or("", String::as_str)
                    )
                }
                _ => String::new(),
            };
            let key = if g.field.as_ref().is_some_and(|f| f.tinted) { "  |  red cut into the part, green on size (0.025 mm), blue stock left, gray off the part" } else { "" };
            g.window.set_title(&format!("icam-sim  |  {}{pause}{play}{key}", self.stage));
        }
    }

    /// Moves the playhead to the start of sweep `to` (clamped to the program) and asks for
    /// the stock there.
    fn seek(&mut self, to: i64) {
        if let Some(total) = self.total {
            let to = to.clamp(0, total as i64) as usize;
            self.now = self.time_of((to, 0.0));
            if self.at != Some((to, 0.0)) {
                self.at = Some((to, 0.0));
                self.ask();
            }
        }
    }

    /// Moves the playhead to t on the machine's clock and asks for the stock there: the
    /// sweeps done, and how far along the move the tool is on when that move cuts.
    fn seek_time(&mut self, t: f64) {
        let (Some(total), Some(last)) = (self.total, self.spans.last()) else {
            return;
        };
        let t = t.clamp(0.0, last.t1);
        self.now = Some(t);
        self.tool_t = None;
        let i = self.spans.partition_point(|s| s.t1 <= t);
        let at = match self.spans.get(i) {
            Some(s) if s.cuts && s.along(t) > 0.0 && s.along(t) < 1.0 => (s.before, s.along(t)),
            Some(s) => (s.before, 0.0),
            None => (total, 0.0),
        };
        if self.at != Some(at) {
            self.at = Some(at);
            self.want_at();
            self.ask();
        }
    }

    /// Paused, the field draws the levels at the playhead wanted (see `Field::want`). Playing,
    /// it draws them all: the levels keep to one playhead in rounds, and between moves (a
    /// rapid holds the playhead) one level reaching it before the others hid them, a flicker.
    fn want_at(&mut self) {
        let want = self.at.filter(|_| self.playing.is_none());
        if let Some(g) = self.gfx.as_mut() {
            if let Some(f) = g.field.as_mut() {
                f.want = want;
                f.upload(&g.queue);
            }
        }
    }

    /// The moment the gouge of region i was cut: on the move that cut its deepest point,
    /// the time the tool's axis passes nearest that point. The playhead goes there, paused,
    /// and the camera onto the tool.
    fn moment(&mut self, i: usize) {
        let Some(r) = self.regions.get(i).cloned() else {
            return;
        };
        let (Some(m), Some(b)) = (r.by, self.bounds) else {
            return;
        };
        let Some(&si) = self.sweep_span.get(m) else {
            return;
        };
        let (motion, length) = (self.spans[si].motion.clone(), self.spans[si].shape[3] as f64);
        let at = V3::new(r.center[0], r.center[1], r.center[2]);
        let off = |u: f64| {
            let pose = motion.pose(u);
            let a = pose.axis();
            let d = at - pose.at;
            let h = icam_sweep::dot(d, a).clamp(0.0, length.max(1e-9));
            (d - icam_sweep::scale(a, h)).len()
        };
        let n = 512;
        let mut best = (0..=n).map(|k| k as f64 / n as f64).min_by(|&x, &y| off(x).total_cmp(&off(y))).unwrap_or(0.0);
        let (mut lo, mut hi) = ((best - 1.0 / n as f64).max(0.0), (best + 1.0 / n as f64).min(1.0));
        for _ in 0..40 {
            let (a, z) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if off(a) <= off(z) {
                hi = z;
            } else {
                lo = a;
            }
        }
        best = 0.5 * (lo + hi);
        let t = self.spans[si].time_at(best);
        self.playing = None;
        self.select(Some(i));
        self.seek_time(t);
        // Low and back, so the tool (its stick-out toward the camera) hides none of the cut.
        let tip = motion.pose(best).at - b.min;
        if let Some(cam) = self.cam {
            let r = self.spans[si].shape[0].max(1.0);
            self.fly = Some((Instant::now(), cam, Camera::toward([tip.x as f32, tip.y as f32, tip.z as f32], cam.yaw(), 0.55, (18.0 * r).max(110.0)), 0.45));
        }
        self.title();
    }

    /// When on the clock the stock is as at playhead `at` (the start of the move partway
    /// along it, or of the next sweep).
    fn time_of(&self, (k, part): (usize, f64)) -> Option<f64> {
        let last = self.spans.last()?;
        Some(match self.sweep_span.get(k) {
            Some(&i) => self.spans[i].time_at(part),
            None => last.t1,
        })
    }

    fn draw(&mut self, ms: &(wgpu::TextureView, wgpu::TextureView), target: &wgpu::TextureView) {
        let (Some(gfx), Some(size), Some(cam)) = (self.gfx.as_ref(), self.size, self.cam.as_ref()) else {
            return;
        };
        let extent = size[0].max(size[1]).max(size[2]);
        let eye = cam.eye();
        // The view: what the panel leaves.
        let vp = self.viewport();
        let aspect = vp[2] / vp[3];
        let settings = self.panel.as_ref().map(|p| &p.settings);
        let u = ViewUniform { view_proj: cam.view_proj(aspect, extent), eye: [eye[0], eye[1], eye[2], 0.0], light: [-0.6, 0.8, 1.4, 0.0] };
        gfx.queue.write_buffer(&gfx.uniform, 0, bytemuck::bytes_of(&u));
        // The tool at the clock's time. While the program plays, the stock trails the clock
        // as it bakes; the tool then leads the stock's own time by no more than the tool
        // takes to travel its radius (what the stock has not caught up with lies under the
        // tool), and never steps back. Paused, it stands at the playhead at once, and the
        // stock catches up.
        let all = (self.total.unwrap_or(0), 0.0);
        let drawn = gfx.field.as_ref().and_then(|f| f.upto()).or(self.at).unwrap_or(all);
        let when = match (self.now, self.time_of(drawn)) {
            (Some(now), Some(t)) if self.playing.is_some() && now > t => {
                let i = self.spans.partition_point(|s| s.t1 <= t).min(self.spans.len().saturating_sub(1));
                let lead = self.spans.get(i).map_or(0.0, |s| {
                    let v = s.motion.rate(0.0, 1.0).0 / (s.t1 - s.t0).max(1e-9);
                    s.shape[0] as f64 / v.max(1e-6)
                });
                Some(now.min(t + lead).max(self.tool_t.filter(|&p| p <= now).unwrap_or(f64::NEG_INFINITY)))
            }
            (now, t) => now.or(t),
        };
        let mut tip = None;
        if let Some(f) = &gfx.field {
            let mut r = cam.ray_view(aspect, extent, u.light);
            if let Some(s) = settings {
                r.heat = [s.tol, s.scale.max(s.tol * 1.01), if s.surface == ui::Surface::Part { 1.0 } else { 0.0 }, 0.0];
            }
            if let (Some(t), Some(bounds)) = (when, self.bounds) {
                let i = self.spans.partition_point(|s| s.t1 <= t).min(self.spans.len().saturating_sub(1));
                if let Some(s) = self.spans.get(i) {
                    let pose = s.motion.pose(s.along(t));
                    let (q, a) = (pose.at - bounds.min, pose.axis());
                    r.tool_at = [q.x as f32, q.y as f32, q.z as f32, s.shape[0]];
                    r.tool_axis = [a.x as f32, a.y as f32, a.z as f32, s.shape[1]];
                    r.tool_shape = [s.shape[2], s.shape[3], if s.shape[0] > 0.0 { 1.0 } else { 0.0 }, 0.0];
                    tip = Some([q.x as f32, q.y as f32, q.z as f32]);
                }
            }
            gfx.queue.write_buffer(&f.view, 0, bytemuck::bytes_of(&r));
            self.drawn = Some(r);
        }
        if let Some(p) = &gfx.path {
            gfx.queue.write_buffer(&p.uniform, 0, bytemuck::bytes_of(&PathUniform { view_proj: u.view_proj, at: [drawn.0 as f32, 0.0, 0.0, 0.0] }));
        }
        let mut enc = gfx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ms.0,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.11, g: 0.11, b: 0.12, a: 1.0 }), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &ms.1,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(f) = &gfx.field {
                pass.set_viewport(vp[0], vp[1], vp[2], vp[3], 0.0, 1.0);
                pass.set_pipeline(&f.pipeline);
                pass.set_bind_group(0, &f.bind, &[]);
                pass.draw(0..3, 0..1);
                if let Some(p) = &gfx.path {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    if settings.is_none_or(|s| s.path) {
                        pass.set_vertex_buffer(0, p.vertices.slice(..));
                        pass.draw(0..p.count, 0..1);
                    }
                    if let Some((buf, n)) = &gfx.marks {
                        pass.set_vertex_buffer(0, buf.slice(..));
                        pass.draw(0..*n, 0..1);
                    }
                }
            } else if let Some((buf, n)) = &gfx.vertices {
                pass.set_pipeline(&gfx.pipeline);
                pass.set_bind_group(0, &gfx.bind, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..*n, 0..1);
            }
        }
        gfx.queue.submit([enc.finish()]);
        self.tip = tip;
        if let (Some(a), Some(b), Some(_)) = (self.tool_t, when, &self.play_run) {
            self.tool_back += (b < a) as usize;
        }
        self.tool_t = when;
    }

    fn redraw(&mut self) {
        let Some(gfx) = self.gfx.as_ref() else {
            return;
        };
        let t_take = Instant::now();
        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let (w, h) = (gfx.config.width, gfx.config.height);
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(w, h);
                    g.window.request_redraw();
                }
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let (w, h, ms) = (gfx.config.width, gfx.config.height, gfx.targets.clone());
        let drawn = gfx.field.as_ref().and_then(|f| f.upto());
        let took = t_take.elapsed().as_secs_f64();
        let t_frame = Instant::now();
        self.frame(&ms, &view, w, h);
        let framed = t_frame.elapsed().as_secs_f64();
        self.frames += 1;
        let behind = match (self.now, drawn.and_then(|d| self.time_of(d))) {
            (Some(now), Some(t)) if Some(drawn) != Some(self.at) => (now - t).max(0.0),
            _ => 0.0,
        };
        if let (Some((_, seen)), Some(at), Some(drawn)) = (self.play_run.as_mut(), self.at, drawn) {
            seen.push((at.0, drawn.0, behind, drawn));
            self.frame_at.push(Instant::now());
        }
        let t_present = Instant::now();
        if let Some(g) = &self.gfx {
            g.queue.present(frame);
        }
        let presented = t_present.elapsed().as_secs_f64();
        if let Some((began, _)) = &self.play_run {
            if took + framed + presented > 0.05 {
                println!("play: a frame at {:.2} s took {:.0} ms: {:.0} for the target, {:.0} to draw, {:.0} to present", began.elapsed().as_secs_f64(), 1e3 * (took + framed + presented), 1e3 * took, 1e3 * framed, 1e3 * presented);
            }
            if let [.., a, b] = self.frame_at.as_slice() {
                if (*b - *a).as_secs_f64() > 0.1 {
                    println!("play: no frame for {:.0} ms, to {:.2} s", 1e3 * (*b - *a).as_secs_f64(), began.elapsed().as_secs_f64());
                }
            }
        }
    }

    /// Renders off screen, writes the picture.
    fn shot(&mut self, path: &std::path::Path) -> Result<(), String> {
        let (w, h) = match &self.gfx {
            Some(g) => (g.config.width, g.config.height),
            None => return Err("no window".into()),
        };
        if let (Some(look), Some(cam)) = (self.args.look, self.cam.as_mut()) {
            cam.look(look);
        }
        let gfx = self.gfx.as_ref().ok_or("no window")?;
        let (device, queue) = (gfx.device.clone(), gfx.queue.clone());
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shot"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&Default::default());
        let ms = targets(&device, w, h);
        let wait = |d: &wgpu::Device| d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map(|_| ()).map_err(|e| format!("GPU did not finish: {e}"));
        self.draw(&ms, &view);
        wait(&device)?;
        let frames = 30;
        let clock = Instant::now();
        for _ in 0..frames {
            self.draw(&ms, &view);
            wait(&device)?;
        }
        let ms_frame = 1e3 * clock.elapsed().as_secs_f64() / frames as f64;
        // The picture with the panel over it (twice: egui lays out on the first pass).
        for _ in 0..2 {
            self.frame(&ms, &view, w, h);
            wait(&device)?;
        }
        write_png(&device, &queue, &color, (w, h), path)?;
        println!("play: {w} x {h}, {} pieces, {ms_frame:.1} ms a frame; wrote {}", self.pieces, path.display());
        Ok(())
    }

    /// The view as drawn now, without the panel, written to `path` (--grab).
    fn grab(&mut self, path: &std::path::Path) -> Result<(), String> {
        let gfx = self.gfx.as_ref().ok_or("no window")?;
        let (w, h) = (gfx.config.width, gfx.config.height);
        let (device, queue) = (gfx.device.clone(), gfx.queue.clone());
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grab"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&Default::default());
        let ms = targets(&device, w, h);
        self.draw(&ms, &view);
        write_png(&device, &queue, &color, (w, h), path)
    }
}

/// A texture of the window's format read back and written as a PNG.
fn write_png(device: &wgpu::Device, queue: &wgpu::Queue, color: &wgpu::Texture, (w, h): (u32, u32), path: &std::path::Path) -> Result<(), String> {
    let wait = |d: &wgpu::Device| d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map(|_| ()).map_err(|e| format!("GPU did not finish: {e}"));
    {
        let row = (w * 4).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("shot"), size: (row * h) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        wait(device)?;
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        {
            let data = slice.get_mapped_range().map_err(|e| format!("the shot did not map: {e:?}"))?;
            for y in 0..h {
                for x in 0..w {
                    let o = (y * row + x * 4) as usize;
                    rgb.extend_from_slice(&[data[o + 2], data[o + 1], data[o]]);
                }
            }
        }
        let out = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(out), w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().and_then(|mut wr| wr.write_image_data(&rgb)).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(())
    }
}

impl ApplicationHandler<Load> for App {
    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // The panel's delayed repaint: drawn when due, else the loop sleeps until then.
        el.set_control_flow(winit::event_loop::ControlFlow::Wait);
        if let Some(at) = self.repaint_at {
            if Instant::now() >= at {
                self.repaint_at = None;
                if let Some(g) = &self.gfx {
                    g.window.request_redraw();
                }
            } else {
                el.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(at));
            }
        }
        // Flying to a region picked, a standard view, level.
        if let Some((since, from, to, secs)) = self.fly {
            let u = (since.elapsed().as_secs_f32() / secs).min(1.0);
            // Smootherstep easing, 6u^5 - 15u^4 + 10u^3.
            let k = u * u * u * (u * (6.0 * u - 15.0) + 10.0);
            self.cam = Some(Camera::between(&from, &to, k));
            if u >= 1.0 {
                self.fly = None;
                self.ask();
            }
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
        }
        // The part's bricks in view, for its cache to settle first (woken to try again when
        // it was too soon).
        if let Some(at) = self.publish_seen() {
            el.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(self.repaint_at.map_or(at, |r| r.min(at))));
        }
        // A free rotate let go while moving coasts, its speed dying away by the clock, until
        // it is under a pixel a 60th of a second.
        if let Some((speed, pivot, left, last)) = self.nav.coast {
            let dt = last.elapsed().as_secs_f32();
            let left = left * ROTATION_INERTIA.powf(60.0 * dt);
            let (x, y) = (speed.0 * left, speed.1 * left);
            if x.abs() > 1.0 || y.abs() > 1.0 {
                let vp = self.viewport();
                if let Some(cam) = self.cam.as_mut() {
                    cam.drag(60.0 * dt * x, 60.0 * dt * y, (vp[2], vp[3]), pivot, true);
                }
                self.nav.coast = Some((speed, pivot, left, Instant::now()));
                if let Some(g) = &self.gfx {
                    g.window.request_redraw();
                }
                self.ask();
            } else {
                self.nav.coast = None;
            }
        }
        // Following the tool while the program plays.
        if self.playing.is_some() && self.fly.is_none() && self.panel.as_ref().is_some_and(|p| p.settings.follow) {
            if let (Some(t), Some(cam)) = (self.tip, self.cam.as_mut()) {
                if cam.target != t {
                    cam.target = t;
                    self.ask();
                }
            }
        }
        // --play: after ten seconds or at the program's end, the measure and the picture.
        // --zoom: two seconds in, k times closer, then out and in again every 1.5 s.
        if let (Some((began, _)), Some(k)) = (&self.play_run, self.args.zoom) {
            let n = self.zooms;
            if began.elapsed().as_secs_f64() >= 2.0 + 1.5 * n as f64 {
                self.zooms += 1;
                self.zoomed.get_or_insert_with(Instant::now);
                if let Some(cam) = self.cam.as_mut() {
                    cam.dist = if n.is_multiple_of(2) { cam.dist / k as f32 } else { cam.dist * k as f32 };
                }
                println!("play: zoomed {} {k} times, {:.1} s into the run", if n.is_multiple_of(2) { "in" } else { "out" }, began.elapsed().as_secs_f64());
                self.ask();
            }
        }
        // --pan: from two seconds in, the camera's target moves along x at v mm a second.
        if let (Some((began, _)), Some(v)) = (&self.play_run, self.args.pan) {
            let t = began.elapsed().as_secs_f64() - 2.0;
            if t >= 0.0 {
                self.zoomed.get_or_insert_with(Instant::now);
                if let Some(cam) = self.cam.as_mut() {
                    let from = *self.pan_from.get_or_insert(cam.target);
                    cam.target[0] = from[0] + (v * t) as f32;
                    if let Some(f) = self.gfx.as_ref().and_then(|g| g.field.as_ref()) {
                        self.pan_seen.push(f.at_target(cam.target));
                    }
                }
                self.ask();
            }
        }
        if let (Some((began, _)), Some(dir)) = (&self.play_run, self.args.grab.clone()) {
            let n = (began.elapsed().as_secs_f64() * 5.0) as usize;
            if n >= self.grabs {
                self.grabs = n + 1;
                let path = dir.join(format!("f{n:03}.png"));
                if let Err(e) = self.grab(&path) {
                    eprintln!("icam-sim: {e}");
                }
            }
        }
        if let Some((began, seen)) = &self.play_run {
            if began.elapsed().as_secs_f64() >= 10.0 || self.playing.is_none() {
                // The frames' gaps, and the stock's: after the zoom when there was one.
                let from = self.zoomed.unwrap_or(*began);
                let gaps: Vec<f64> = self.frame_at.windows(2).filter(|w| w[1] >= from).map(|w| (w[1] - w[0]).as_secs_f64()).collect();
                let worst = gaps.iter().cloned().fold(0.0, f64::max);
                let slow = gaps.iter().filter(|&&g| g > 1.0 / 30.0).count();
                let mut stock_gap = 0.0f64;
                let mut last: Option<(Instant, (usize, f64))> = None;
                for (t, s) in self.frame_at.iter().zip(seen.iter().map(|x| x.3)) {
                    match last {
                        Some((t0, s0)) if s0 == s => stock_gap = stock_gap.max((*t - t0).as_secs_f64()).max(0.0),
                        _ => last = Some((*t, s)),
                    }
                    if last.is_some_and(|l| l.1 != s) {
                        last = Some((*t, s));
                    }
                }
                println!(
                    "play: {}frames: the longest gap {:.0} ms, {} over 33 ms of {}; the stock stood still {:.2} s at most",
                    if self.zoomed.is_some() { "after the zoom, " } else { "" },
                    1e3 * worst,
                    slow,
                    gaps.len(),
                    stock_gap
                );
                if !self.pan_seen.is_empty() {
                    let n = self.pan_seen.len();
                    let finest = self.pan_seen.iter().map(|x| x.0).filter(|&c| c > 0.0).fold(f32::INFINITY, f32::min);
                    let at_finest = self.pan_seen.iter().filter(|x| x.0 == finest).count();
                    let worst = self.pan_seen.iter().map(|x| x.0).fold(0.0, f32::max);
                    let mut off: Vec<f32> = self.pan_seen.iter().map(|x| x.1).collect();
                    off.sort_by(f32::total_cmp);
                    let q = |f: f64| off[((n - 1) as f64 * f) as usize];
                    println!(
                        "play: the pan's {n} frames: the target in cells of {finest} mm in {at_finest} ({:.0} %), the coarsest {worst} mm; how far it lay from the middle of the finest level, as a share of half its side: median {:.2}, p90 {:.2}, most {:.2}",
                        100.0 * at_finest as f64 / n as f64,
                        q(0.5),
                        q(0.9),
                        q(1.0)
                    );
                }
                let secs = began.elapsed().as_secs_f64();
                let n = seen.len().max(1);
                let lag: Vec<usize> = seen.iter().map(|&(at, drawn, _, _)| at.saturating_sub(drawn)).collect();
                let late: Vec<f64> = seen.iter().map(|x| x.2).collect();
                let updates = seen.windows(2).filter(|w| w[1].3 != w[0].3).count();
                let (first, last) = (seen.first().map_or(0, |x| x.1), seen.last().map_or(0, |x| x.1));
                println!(
                    "play: played {secs:.1} s at {}x the machine's speed ({:.0} moves a second): {} frames ({:.0} a second); the stock was drawn anew {updates} times ({:.1} a second), from move {first} to move {last}, trailing the playhead by {:.0} moves on average, {} at most, by {:.3} s of the machine's clock on average, {:.3} s at most",
                    self.speed,
                    (last - first) as f64 / secs,
                    seen.len(),
                    seen.len() as f64 / secs,
                    updates as f64 / secs,
                    lag.iter().sum::<usize>() as f64 / n as f64,
                    lag.iter().max().unwrap_or(&0),
                    late.iter().sum::<f64>() / n as f64,
                    late.iter().cloned().fold(0.0, f64::max)
                );
                println!("play: the tool stepped back on {} of those frames", self.tool_back);
                self.play_run = None;
                self.playing = None;
                if let Some(path) = self.args.shot.take() {
                    self.result = self.shot(&path);
                    el.exit();
                }
                return;
            }
        }
        if let (Some((from_time, from)), Some(last)) = (self.playing, self.spans.last()) {
            let end = last.t1;
            let to = from + self.speed * from_time.elapsed().as_secs_f64();
            if to >= end {
                self.playing = None;
            }
            self.seek_time(to.min(end));
            if self.playing.is_none() || self.told.elapsed().as_secs_f64() > 0.25 {
                self.told = Instant::now();
                self.title();
            }
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
        }
    }

    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("icam-sim  |  starting").with_inner_size(winit::dpi::LogicalSize::new(1600.0, 1000.0));
        let made = el.create_window(attrs).map_err(|e| format!("no window: {e}")).and_then(|w| Gfx::new(Arc::new(w)));
        match made {
            Ok(gfx) => {
                println!("play: window open at {:.2} s on {}", self.start.elapsed().as_secs_f64(), gfx.device.adapter_info().name);
                let (tx, rx) = std::sync::mpsc::channel();
                self.asks = Some(tx);
                let _ = self.gpu.set(gfx.device.clone());
                spawn_load(&self.args, self.proxy.clone(), self.paused.clone(), self.start, rx, self.gpu.clone());
                if self.args.bake.is_some() {
                    let ctx = egui::Context::default();
                    ctx.set_visuals(egui::Visuals::dark());
                    let input = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &*gfx.window, Some(gfx.window.scale_factor() as f32), gfx.window.theme(), Some(gfx.device.limits().max_texture_dimension_2d as usize));
                    let renderer = egui_wgpu::Renderer::new(&gfx.device, FORMAT, egui_wgpu::RendererOptions::default());
                    let mut settings = ui::Settings::default();
                    if self.args.part_view && (self.args.part.is_some() || self.args.solid.is_some()) {
                        settings.surface = ui::Surface::Part;
                    }
                    self.panel = Some(Panel { ctx, input, renderer, settings });
                    if self.args.solid.is_some() {
                        self.comparing = "The part alone, by its exact field: no program to compare.".into();
                    } else if self.args.part.is_some() {
                        self.comparing = "Comparing the part with the finished stock...".into();
                    } else {
                        self.comparing = "No part given (--part), so nothing to compare.".into();
                    }
                }
                self.gfx = Some(gfx);
            }
            Err(e) => {
                self.result = Err(e);
                el.exit();
            }
        }
    }

    fn user_event(&mut self, el: &ActiveEventLoop, ev: Load) {
        let clock = Instant::now();
        let kind = match &ev {
            Load::Level { .. } => "a level",
            Load::PartLevel { .. } => "a part's level",
            Load::Report(_) => "the report",
            _ => "an event",
        };
        self.event(el, ev);
        if self.play_run.is_some() && clock.elapsed().as_secs_f64() > 0.03 {
            println!("play: taking {kind} took {:.0} ms", 1e3 * clock.elapsed().as_secs_f64());
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        // The panel sees every event first; what it takes, or what lands on it, the view does not.
        let (mut taken, mut over) = (false, false);
        if let (Some(p), Some(g)) = (self.panel.as_mut(), self.gfx.as_ref()) {
            let r = p.input.on_window_event(&g.window, &ev);
            // egui answers a redraw itself with "repaint": asked again there, every frame asked
            // for the next and the window drew at the display's rate with nothing changing (the
            // GPU 40 % busy, in its top power state, competing with the caches' builds). The
            // panel's own repaints come from its output (`repaint_delay`).
            if r.repaint && !matches!(ev, WindowEvent::RedrawRequested) {
                g.window.request_redraw();
            }
            taken = r.consumed;
            over = taken || p.ctx.is_pointer_over_egui();
        }
        let mut moved = false;
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(s.width, s.height);
                }
                moved = true;
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::ModifiersChanged(m) => {
                let s = m.state();
                (self.nav.alt, self.nav.shift) = (s.alt_key(), s.shift_key());
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let held = match button {
                    MouseButton::Left => &mut self.nav.left,
                    MouseButton::Right => &mut self.nav.right,
                    MouseButton::Middle => &mut self.nav.middle,
                    _ => return,
                };
                if state == ElementState::Pressed {
                    if !over {
                        *held = true;
                        self.press();
                    }
                } else if *held {
                    *held = false;
                    self.release(button == MouseButton::Middle);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                let (dx, dy) = (x - self.cursor.0, y - self.cursor.1);
                self.cursor = (x, y);
                moved = self.drag_by(dx, dy);
            }
            WindowEvent::MouseWheel { .. } if over => {}
            WindowEvent::MouseWheel { delta, .. } => {
                // Notches (a touchpad's pixels at a browser's 100 a notch), up and down only;
                // a scroll away is a positive step, reversed as REVERSE_WHEEL says.
                let notches = match delta {
                    MouseScrollDelta::LineDelta(x, y) if y.abs() >= x.abs() => y,
                    MouseScrollDelta::PixelDelta(p) if p.y.abs() >= p.x.abs() => p.y as f32 / 100.0,
                    _ => 0.0,
                };
                moved = self.wheel(WHEEL_STEP * notches * if REVERSE_WHEEL { -1.0 } else { 1.0 });
            }
            WindowEvent::KeyboardInput { .. } if taken => {}
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => el.exit(),
                Key::Named(NamedKey::Space) if self.args.bake.is_some() => self.toggle_play(),
                Key::Named(NamedKey::Space) => {
                    let was = self.paused.load(Ordering::Relaxed);
                    self.paused.store(!was, Ordering::Relaxed);
                    self.title();
                }
                Key::Character(c @ ("[" | "]" | "r" | "R" | "p" | "P")) if self.args.bake.is_some() => {
                    let end = self.spans.last().map_or(0.0, |s| s.t1);
                    let now = self.now.filter(|&t| t < end).unwrap_or(0.0);
                    self.speed = match c {
                        "[" => (self.speed * 0.5).max(1.0 / 64.0),
                        "]" => (self.speed * 2.0).min(1e6),
                        "r" | "R" => 1.0,
                        // The rest of the program in a minute (the whole of it from the end).
                        _ => ((end - now) / 60.0).max(1.0),
                    };
                    // Keep the playhead where it is; R and P also start playing.
                    if self.playing.is_some() || !matches!(c, "[" | "]") {
                        self.playing = Some((Instant::now(), now));
                        self.seek_time(now);
                    }
                    self.title();
                }
                _ if self.nav.shift && standard_view(event.physical_key).is_some() => {
                    if let (Some(cam), Some((dir, up))) = (self.cam, standard_view(event.physical_key)) {
                        self.fly_to(cam.turned(dir, up), LONG_SECS);
                    }
                }
                Key::Character("f") | Key::Character("F") => self.zoom_fit(),
                Key::Character("z") | Key::Character("Z") => {
                    if let Some(cam) = self.cam {
                        let k = if self.nav.shift { 1.0 / KEY_ZOOM } else { KEY_ZOOM };
                        if self.can_zoom(k, cam.target) {
                            let mut to = cam;
                            to.zoom_at(cam.target, k);
                            self.fly_to(to, SHORT_SECS);
                        }
                    }
                }
                Key::Character("t") | Key::Character("T") => {
                    if let Some(cam) = self.cam.as_mut() {
                        *cam = Camera::toward(cam.target, cam.yaw(), 1.5703, cam.dist);
                        moved = true;
                    }
                }
                Key::Named(k @ (NamedKey::ArrowLeft | NamedKey::ArrowRight | NamedKey::ArrowUp | NamedKey::ArrowDown | NamedKey::PageUp | NamedKey::PageDown | NamedKey::Home | NamedKey::End)) if !self.spans.is_empty() => {
                    self.step(k);
                    moved = true;
                }
                Key::Named(k @ (NamedKey::ArrowLeft | NamedKey::ArrowRight | NamedKey::ArrowUp | NamedKey::ArrowDown | NamedKey::PageUp | NamedKey::PageDown | NamedKey::Home | NamedKey::End)) => {
                    if let Some(total) = self.total {
                        let now = self.at.map_or(total, |a| a.0) as i64;
                        let to = match k {
                            NamedKey::ArrowLeft => now - 1,
                            NamedKey::ArrowRight => now + 1,
                            NamedKey::ArrowDown => now - 100,
                            NamedKey::ArrowUp => now + 100,
                            NamedKey::PageDown => now - 1000,
                            NamedKey::PageUp => now + 1000,
                            NamedKey::Home => 0,
                            _ => total as i64,
                        };
                        self.seek(to);
                        self.stage = format!("moving the playhead to move {} of {total}", to.clamp(0, total as i64));
                        self.title();
                        moved = true;
                    }
                }
                _ => {}
            },
            _ => {}
        }
        if moved {
            if let Some(g) = &self.gfx {
                g.window.request_redraw();
            }
            self.ask();
        }
    }
}

impl App {
    fn event(&mut self, el: &ActiveEventLoop, ev: Load) {
        match ev {
            Load::Stage(s) => self.stage = s,
            Load::Comparing(s) => self.comparing = s,
            Load::Report(r) => {
                let b = r.bands;
                self.comparing = format!(
                    "Finished program: {} points on the part, {} mm apart, compared in {:.1} s. On size {:.0}%, stock left {:.0}%, gouged {:.1} mm\u{b2}.",
                    r.points,
                    r.step,
                    r.secs,
                    100.0 * b.on_size / b.part.max(1e-9),
                    100.0 * b.left / b.part.max(1e-9),
                    b.gouged
                );
                println!("play: {}", self.comparing);
                self.report = Some(r);
                self.regions_at = -1.0;
                // A picture waiting for the comparison: the view again, so the worker settles.
                if self.done && self.args.shot.is_some() {
                    self.ask();
                }
            }
            Load::Box(b) => {
                let size = [(b.max.x - b.min.x) as f32, (b.max.y - b.min.y) as f32, (b.max.z - b.min.z) as f32];
                if self.gfx.is_some() {
                    let v = self.viewport();
                    let mut cam = Camera::fit(size, v[2] / v[3]);
                    if let Some(look) = self.args.look {
                        cam.look(look);
                    }
                    self.cam = Some(cam);
                }
                self.size = Some(size);
                self.bounds = Some(b);
                self.ask();
            }
            Load::Skin { vertices, done, total, secs } => {
                self.pieces = vertices.len() / 3;
                if let Some(g) = self.gfx.as_mut() {
                    let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("skin"), contents: bytemuck::cast_slice(&vertices), usage: wgpu::BufferUsages::VERTEX });
                    g.vertices = Some((buf, vertices.len() as u32));
                }
                let rate = if secs > 0.0 { done as f64 / secs } else { 0.0 };
                self.stage = format!("moves {done} of {total} ({rate:.1} a second), {} pieces of skin", self.pieces);
                if done == total {
                    println!("play: every move in, {:.1} s after start; {} pieces", self.start.elapsed().as_secs_f64(), self.pieces);
                    self.done = true;
                    if let Some(path) = self.args.shot.clone() {
                        self.result = self.shot(&path);
                        el.exit();
                    }
                }
            }
            Load::Path { vertices, spans, ops } => {
                if let Some(g) = self.gfx.as_mut() {
                    g.path = Some(PathDraw::new(&g.device, &vertices));
                }
                self.sweep_span = spans.iter().enumerate().filter(|(_, s)| s.cuts).map(|(i, _)| i).collect();
                self.spans = spans;
                self.ops = ops;
                self.now = self.time_of(self.at.unwrap_or((usize::MAX, 0.0)));
                self.title();
            }
            Load::Level { e, cache, heat, staged } => {
                self.stock_es.insert(e);
                if let (Some(g), Some(size), Some(b)) = (self.gfx.as_mut(), self.size, self.bounds) {
                    let f = g.field.get_or_insert_with(|| Field::new(&g.device, (FORMAT, SAMPLES, true), "fs", size));
                    f.want = self.at.filter(|_| self.playing.is_none());
                    let clock = Instant::now();
                    f.put(&g.device, &g.queue, e, &cache, (heat.as_deref().map(|t| (t.0.as_slice(), t.1.as_slice())), staged.as_ref()), b.min);
                    let took = clock.elapsed().as_secs_f64();
                    self.put_secs += took;
                    if self.play_run.is_some() && took > 0.008 {
                        let what = match &cache.patch {
                            Some(p) => format!("a patch of {} slots ({} placed) and {} map entries", p.slots.len(), p.placed.len(), p.entries.len()),
                            None => "whole".to_string(),
                        };
                        println!("play: putting cells of {} mm took {:.0} ms ({what}, {} bricks)", cache.h, 1e3 * took, cache.bricks());
                    }
                    self.pieces = f.bricks();
                    let (fine, coarse) = f.cells();
                    let of = self.total.map_or(String::new(), |t| format!(" of {t}"));
                    self.stage = format!("the stock after {}{of}, drawn from {} levels of its field, cells {fine} to {coarse} mm, {} bricks", at_text((cache.upto, cache.part)), f.held.len(), self.pieces);
                }
            }
            Load::PartLevel { e, lo, h, dims, level } => {
                self.part_es.insert(e);
                self.parts_in += 1;
                // The part alone, the picture held for this level: settle again.
                if self.args.solid.is_some() && self.args.shot.is_some() && self.done && self.stock_es.first() == Some(&e) {
                    let _ = self.proxy.send_event(Load::Settled);
                }
                if let (Some(g), Some(size), Some(b)) = (self.gfx.as_mut(), self.size, self.bounds) {
                    let f = g.field.get_or_insert_with(|| Field::new(&g.device, (FORMAT, SAMPLES, true), "fs", size));
                    let l = lo - b.min;
                    let clock = Instant::now();
                    f.put_part(&g.device, &g.queue, e, ([l.x as f32, l.y as f32, l.z as f32, h as f32], dims), &level);
                    if self.play_run.is_some() {
                        println!("play: putting the part's level {e} took {:.0} ms", 1e3 * clock.elapsed().as_secs_f64());
                    }
                    g.window.request_redraw();
                }
            }
            Load::DropPart(e) => {
                self.part_es.remove(&e);
                if let Some(g) = self.gfx.as_mut() {
                    if let Some(f) = g.field.as_mut() {
                        f.drop_part(e);
                        f.upload(&g.queue);
                    }
                }
            }
            Load::Drop(e) => {
                self.stock_es.remove(&e);
                if let Some(g) = self.gfx.as_mut() {
                    if let Some(f) = g.field.as_mut() {
                        f.drop_level(&g.queue, e);
                        self.pieces = f.bricks();
                    }
                }
            }
            Load::Moves(n) => {
                self.total = Some(n);
                // --scrub: across the whole program, then back to halfway for the picture.
                if let (Some(steps), Some(_)) = (self.args.scrub, &self.args.shot) {
                    let step = n.div_ceil(steps.max(1));
                    let mut todo: std::collections::VecDeque<usize> = (0..=n).step_by(step.max(1)).collect();
                    todo.push_back(n / 2);
                    self.scrub = Some(Scrub { todo, asked: Instant::now(), settled: Vec::new(), started: false, step });
                }
            }
            Load::Settled => {
                if !self.done {
                    println!("play: every level in, {:.1} s after start; {} bricks", self.start.elapsed().as_secs_f64(), self.pieces);
                }
                self.done = true;
                // --scrub: on to the next playhead, timing how long each took to settle.
                if let Some(s) = self.scrub.as_mut() {
                    let from = self.at.map(|a| a.0).or(self.total).unwrap_or(0);
                    if s.started {
                        if let Some(x) = s.settled.last_mut() {
                            x.1 = s.asked.elapsed().as_secs_f64();
                        }
                    } else {
                        s.started = true;
                        self.put_secs = 0.0;
                        self.frames = 0;
                    }
                    if let Some(next) = s.todo.pop_front() {
                        s.settled.push((next.abs_diff(from), 0.0));
                        s.asked = Instant::now();
                        self.at = Some((next, 0.0));
                        self.ask();
                        return;
                    }
                    let regular: Vec<f64> = s.settled.iter().filter(|x| x.0 == s.step).map(|x| x.1).collect();
                    let n = regular.len().max(1) as f64;
                    let (mean, max) = (regular.iter().sum::<f64>() / n, regular.iter().cloned().fold(0.0, f64::max));
                    let jumps: Vec<String> = s.settled.iter().filter(|x| x.0 != s.step).map(|x| format!("{} moves in {:.2} s", x.0, x.1)).collect();
                    println!(
                        "play: scrubbed the program in {} steps of {} moves: each settled in {mean:.3} s on average, {max:.3} s at most, {:.1} s in all (jumps besides: {}); the window took levels in for {:.2} s of it all and drew {} frames",
                        regular.len(),
                        s.step,
                        regular.iter().sum::<f64>(),
                        jumps.join(", "),
                        self.put_secs,
                        self.frames
                    );
                    self.scrub = None;
                }
                // --play: play from the playhead; with --shot, measured, before the picture.
                if let (Some(speed), Some(end), None) = (self.args.play.take(), self.spans.last().map(|s| s.t1), &self.play_run) {
                    self.speed = speed;
                    let from = self.now.filter(|&t| t < end).unwrap_or(0.0);
                    self.playing = Some((Instant::now(), from));
                    if self.args.shot.is_some() {
                        self.play_run = Some((Instant::now(), Vec::new()));
                        self.frames = 0;
                    }
                    self.seek_time(from);
                    self.title();
                    return;
                }
                if self.play_run.is_some() {
                    return;
                }
                // The part alone: the picture waits for the part's level the view draws.
                if self.args.shot.is_some() && self.args.solid.is_some() && !self.stock_es.first().is_some_and(|e| self.part_es.contains(e)) {
                    return;
                }
                // With the panel and the part, the picture waits for the comparison, and for
                // the region to pick.
                if self.args.shot.is_some() && self.panel.is_some() && self.args.part.is_some() && self.report.is_none() && !self.comparing.starts_with("The comparison failed") {
                    return;
                }
                if let (Some(n), Some(_)) = (self.args.moment, &self.args.shot) {
                    self.args.moment = None;
                    if let Some(mut p) = self.panel.take() {
                        self.sync_regions(&mut p.settings);
                        self.panel = Some(p);
                    }
                    self.moment(n);
                    if let Some((_, _, to, _)) = self.fly.take() {
                        self.cam = Some(to);
                    }
                    self.ask();
                    return;
                }
                if let (Some(n), Some(_)) = (self.args.pick, &self.args.shot) {
                    self.args.pick = None;
                    if let Some(mut p) = self.panel.take() {
                        self.sync_regions(&mut p.settings);
                        self.panel = Some(p);
                    }
                    self.select(Some(n));
                    if let Some((_, _, to, _)) = self.fly.take() {
                        self.cam = Some(to);
                    }
                    self.ask();
                    return;
                }
                if let Some(path) = self.args.shot.take() {
                    self.result = self.shot(&path);
                    el.exit();
                }
            }
            Load::Failed(e) => {
                self.stage = format!("failed: {e}");
                eprintln!("icam-sim: {e}");
                if self.args.shot.is_some() {
                    self.result = Err(e);
                    el.exit();
                }
            }
        }
        self.title();
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }

}

pub fn run(args: Args) -> Result<(), String> {
    let start = Instant::now();
    SCROLL.store(!args.no_scroll, Ordering::Relaxed);
    let el = EventLoop::<Load>::with_user_event().build().map_err(|e| format!("no event loop: {e}"))?;
    let proxy = el.create_proxy();
    let at = args.at.map(|n| (n, 0.0));
    let mut app = App {
        stock_es: Default::default(),
        part_es: Default::default(),
        args,
        start,
        proxy,
        paused: Arc::new(AtomicBool::new(false)),
        gfx: None,
        gpu: Arc::new(OnceLock::new()),
        size: None,
        cam: None,
        stage: "starting".into(),
        nav: Nav::default(),
        cursor: (0.0, 0.0),
        drawn: None,
        parts_in: 0,
        seen_for: None,
        seen_at: None,
        pieces: 0,
        done: false,
        result: Ok(()),
        asks: None,
        bounds: None,
        at,
        total: None,
        spans: Vec::new(),
        ops: Vec::new(),
        sweep_span: Vec::new(),
        now: None,
        playing: None,
        speed: 1.0,
        told: Instant::now(),
        panel: None,
        report: None,
        regions: Vec::new(),
        regions_at: -1.0,
        comparing: String::new(),
        marks: Vec::new(),
        fly: None,
        tip: None,
        surface: ui::Surface::Stock,
        tool_t: None,
        tool_back: 0,
        frame_at: Vec::new(),
        zoomed: None,
        zooms: 0,
        pan_from: None,
        pan_seen: Vec::new(),
        grabs: 0,
        scrub: None,
        put_secs: 0.0,
        frames: 0,
        repaint_at: None,
        play_run: None,
    };
    el.run_app(&mut app).map_err(|e| format!("the window failed: {e}"))?;
    app.result
}

/// The program on the machine's clock, and at its programmed feeds alone, seconds: every
/// move, cutting or not (`machine_clock`).
pub fn program_time(moves: &[icam_post::Move], machine: Machine) -> (f64, f64) {
    let motions: Vec<Arc<dyn icam_sweep::Motion>> = moves.iter().map(icam_sweep::program::motion).collect();
    let caps: Vec<f64> = moves.iter().map(|mv| if mv.kind == icam_post::MoveKind::Rapid || mv.feed <= 0.0 { RAPID_MMPS } else { mv.feed / 60.0 }).collect();
    let programmed = motions.iter().zip(&caps).map(|(m, v)| m.rate(0.0, 1.0).0 / v).sum();
    (machine_clock(&motions, &caps, machine).iter().map(|x| x.0).sum(), programmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEW: (f32, f32) = (1600.0, 1000.0);

    /// Where p lands in the view: normalized device x and y, and its depth.
    fn ndc(c: &Camera, p: [f32; 3]) -> (f32, f32, f32) {
        let (f, s, u) = c.basis();
        let o = sub3(p, c.eye());
        let (z, t) = (dot3(o, f), (0.5 * FOV).tan());
        (dot3(o, s) / (z * t * VIEW.0 / VIEW.1), dot3(o, u) / (z * t), z)
    }

    /// The same in pixels.
    fn screen(c: &Camera, p: [f32; 3]) -> (f32, f32, f32) {
        let (x, y, z) = ndc(c, p);
        ((x + 1.0) * 0.5 * VIEW.0, (1.0 - y) * 0.5 * VIEW.1, z)
    }

    fn near(a: (f32, f32, f32), b: (f32, f32, f32), px: f32) -> bool {
        (a.0 - b.0).abs() <= px && (a.1 - b.1).abs() <= px && (a.2 - b.2).abs() <= 1e-3 * a.2.abs().max(1.0)
    }

    #[test]
    fn toward_is_the_old_orbit() {
        for (yaw, pitch) in [(-90f32, 55f32), (30.0, 10.0), (200.0, -40.0), (0.0, 89.9)] {
            let (y, p) = (yaw.to_radians(), pitch.to_radians());
            let c = Camera::toward([10.0, 20.0, 5.0], y, p, 300.0);
            let eye = [10.0 + 300.0 * p.cos() * y.cos(), 20.0 + 300.0 * p.cos() * y.sin(), 5.0 + 300.0 * p.sin()];
            let f = unit3(sub3(c.target, eye));
            let s = unit3(cross3(f, [0.0, 0.0, 1.0]));
            let (nf, ns, nu) = c.basis();
            for k in 0..3 {
                assert!((c.eye()[k] - eye[k]).abs() < 1e-3);
                assert!((nf[k] - f[k]).abs() < 1e-5 && (ns[k] - s[k]).abs() < 1e-5 && (nu[k] - cross3(s, f)[k]).abs() < 1e-5);
            }
            let turn = std::f32::consts::TAU;
            assert!((c.yaw().rem_euclid(turn) - y.rem_euclid(turn)).abs() < 1e-3, "yaw {yaw}");
        }
    }

    #[test]
    fn the_turntable_holds_its_pivot_and_goes_over_the_top() {
        let mut c = Camera::toward([50.0, 50.0, 0.0], -1.2, 0.9, 250.0);
        let pivot = [60.0, 40.0, 10.0];
        let at = screen(&c, pivot);
        let mut lowest = 1f32;
        for _ in 0..100 {
            c.drag(3.0, 10.0, VIEW, pivot, false);
            assert!(near(screen(&c, pivot), at, 0.05));
            assert!(c.roll().abs() < 1e-4);
            lowest = lowest.min(c.up[2]);
        }
        // Upside down past the pole: nothing clamps it.
        assert!(lowest < -0.5);
    }

    #[test]
    fn a_free_turn_rolls_and_leveling_takes_it_out_in_place() {
        let mut c = Camera::toward([50.0, 50.0, 0.0], -1.2, 0.9, 250.0);
        let pivot = [60.0, 40.0, 10.0];
        let at = screen(&c, pivot);
        c.drag(200.0, 150.0, VIEW, pivot, true);
        assert!(near(screen(&c, pivot), at, 0.05));
        assert!(c.roll().abs() > 0.01);
        let l = c.leveled(pivot);
        assert!(l.roll().abs() < 1e-5);
        assert!(near(screen(&l, pivot), at, 0.05));
    }

    #[test]
    fn a_zoom_holds_its_point_and_never_passes_it() {
        let mut c = Camera::toward([50.0, 50.0, 0.0], -1.2, 0.9, 250.0);
        let p = [70.0, 35.0, 4.0];
        let at = screen(&c, p);
        let from = dot3(sub3(p, c.eye()), sub3(p, c.eye())).sqrt();
        c.zoom_at(p, 0.84);
        let now = screen(&c, p);
        assert!((now.0 - at.0).abs() < 0.05 && (now.1 - at.1).abs() < 0.05);
        assert!((dot3(sub3(p, c.eye()), sub3(p, c.eye())).sqrt() - 0.84 * from).abs() < 1e-3);
        // The target is at p's depth.
        assert!((c.dist - now.2).abs() < 1e-3);
        // In to where a pixel is MIN_PIXEL, as far as `can_zoom` lets it.
        while 1.0 / Camera::pixels_per_mm(screen(&c, p).2 * 0.84, VIEW.1) > MIN_PIXEL {
            c.zoom_at(p, 0.84);
        }
        let deep = screen(&c, p);
        assert!(deep.2 > 0.0 && (deep.0 - at.0).abs() < 0.5 && (deep.1 - at.1).abs() < 0.5, "{deep:?} {at:?}");
    }

    #[test]
    fn a_pan_keeps_the_point_under_the_cursor() {
        let mut c = Camera::toward([50.0, 50.0, 0.0], -1.2, 0.9, 250.0);
        let p = [70.0, 35.0, 4.0];
        let (x0, y0, _) = screen(&c, p);
        let (x1, y1) = (x0 + 37.0, y0 - 12.0);
        let (a, b) = (c.on_plane(x0, y0, VIEW, p).unwrap(), c.on_plane(x1, y1, VIEW, p).unwrap());
        c.target = sub3(c.target, sub3(b, a));
        let now = screen(&c, p);
        assert!((now.0 - x1).abs() < 0.05 && (now.1 - y1).abs() < 0.05);
    }

    #[test]
    fn a_window_casts_the_views_own_rays() {
        let c = Camera::toward([50.0, 50.0, 0.0], -1.2, 0.9, 250.0);
        let r = c.ray_view(VIEW.0 / VIEW.1, 300.0, [0.0; 4]);
        let w = window(r, 700.0, 400.0, 7.0, VIEW);
        let ray = |r: &RayView, x: f32, y: f32| unit3([0, 1, 2].map(|k| r.fwd[k] + x * r.right[k] + y * r.up[k]));
        for (i, j) in [(0.0, 0.0), (3.0, 3.0), (6.0, 2.0)] {
            let small = ray(&w, 2.0 * (i + 0.5) / 7.0 - 1.0, 1.0 - 2.0 * (j + 0.5) / 7.0);
            let (x, y) = (700.0 + i + 0.5, 400.0 + j + 0.5);
            let full = ray(&r, 2.0 * x / VIEW.0 - 1.0, 1.0 - 2.0 * y / VIEW.1);
            let cast = c.ray(x, y, VIEW).1;
            for k in 0..3 {
                assert!((small[k] - full[k]).abs() < 1e-5 && (cast[k] - full[k]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn a_zoom_to_fit_frames_the_box() {
        let s = [300.0, 200.0, 50.0];
        for c in [Camera::toward([0.0; 3], -1.57, 0.96, 900.0), Camera::toward([10.0, 0.0, 0.0], 0.4, -0.3, 20.0)] {
            let f = c.framed(s, VIEW.0 / VIEW.1);
            let mut most = 0f32;
            for k in 0..8 {
                let p = [if k & 1 == 0 { 0.0 } else { s[0] }, if k & 2 == 0 { 0.0 } else { s[1] }, if k & 4 == 0 { 0.0 } else { s[2] }];
                let (x, y, z) = ndc(&f, p);
                assert!(z > 0.0);
                most = most.max(x.abs()).max(y.abs());
            }
            assert!((most - 1.0 / FIT_ROOM).abs() < 1e-4, "{most}");
        }
    }

    #[test]
    fn the_buttons_mean_what_the_bindings_say() {
        assert_eq!(view_mode(LMB, Mode::None), Mode::Turntable);
        assert_eq!(view_mode(ALT | LMB, Mode::None), Mode::Rotate);
        assert_eq!(view_mode(RMB, Mode::None), Mode::Pan);
        assert_eq!(view_mode(MMB, Mode::None), Mode::Pan);
        assert_eq!(view_mode(SHIFT | LMB, Mode::None), Mode::Pan);
        assert_eq!(view_mode(SHIFT, Mode::None), Mode::None);
        // Keys no binding uses are passed over; Shift with a pan is a pan.
        assert_eq!(view_mode(SHIFT | RMB, Mode::None), Mode::Pan);
        // Both buttons: the one with more flags, another than the drag's own first.
        assert_eq!(view_mode(LMB | RMB, Mode::None), Mode::Turntable);
        assert_eq!(view_mode(LMB | RMB, Mode::Turntable), Mode::Pan);
    }

    #[test]
    fn a_flight_ends_where_it_was_sent() {
        let (a, b) = (Camera::toward([0.0; 3], 0.3, 0.2, 100.0), Camera::toward([5.0, 6.0, 7.0], 2.0, 1.1, 20.0));
        let (s, e, m) = (Camera::between(&a, &b, 0.0), Camera::between(&a, &b, 1.0), Camera::between(&a, &b, 0.5));
        for k in 0..3 {
            assert!((s.dir[k] - a.dir[k]).abs() < 1e-5 && (e.dir[k] - b.dir[k]).abs() < 1e-5 && (e.target[k] - b.target[k]).abs() < 1e-4);
        }
        assert!((e.dist - 20.0).abs() < 1e-3 && dot3(m.dir, m.up).abs() < 1e-5 && (dot3(m.dir, m.dir) - 1.0).abs() < 1e-5);
    }
}
