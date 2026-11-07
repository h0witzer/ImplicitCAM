//! The part's exact signed distance at its cache's brick nodes on the GPU (decision 9):
//! the evaluator packed for its kernel (`Exact::gpu_pack`, icam-core's solid3/gpu.wgsl),
//! here uploaded and run. Each brick's candidate features are listed on the CPU from the
//! brick's center; every node comes back exact, or flagged (1: only a lower bound on its
//! distance, past the cache's clamp, is known, and its sign is to be taken from a
//! neighbor; 2: the CPU must read it). The f64 CPU evaluator is the reference the kernel
//! is checked against (`part-bench --gpu`).

use bytemuck::{Pod, Zeroable};
use icam_core::curve3::V3;
use icam_core::solid3::Exact;
use wgpu::util::DeviceExt;

const NODES: usize = 729;
/// Bricks per dispatch, at most: about a quarter of a million nodes.
const BRICKS_AT_ONCE: usize = 360;
/// Readings in one dispatch, at most (a node against one of its brick's candidates): a
/// dispatch holds the GPU until it is done, and a window drawing on it waits behind it.
const DISPATCH_WORK: usize = 1 << 21;
/// Dispatches queued before their results are read back (each about 2 MB of results, held
/// twice: the GPU's memory is shared with the window's caches).
const QUEUED: usize = 16;
/// The same for the scatter, whose dispatches hold its jobs and keys as well.
const SCATTER_QUEUED: usize = 8;
/// Jobs (a part and a brick, 8 bytes each) one scatter dispatch holds: past it, the
/// dispatch's bricks are read again in two halves (a brick alone, by the gather).
const JOB_ROOM: usize = 1 << 20;
/// A bricks' scatter dispatch is sized by what its passes read: its seed, a node against
/// each closed-form feature listed (past the clamp a foot alone, no trim test: so eight
/// times a gather's readings), and its jobs, a span or piece listed opening into some number
/// of parts. That number is learned as the dispatches come back (THE PART: 1 to 10 a root
/// listed), from JOBS_PER_ROOT before any has, and a dispatch is planned to fill half of
/// JOB_ROOM.
const SEED_WORK: usize = 8 * DISPATCH_WORK;
const JOBS_PER_ROOT: f64 = 16.0;
/// Workgroups the scatter's jobs are shared among.
const SCATTER_GROUPS: u32 = 4096;
/// Points in one scatter dispatch, at most.
const POINTS_AT_ONCE: usize = 4096;
/// A point's value when its dispatch had no room for its jobs or its refinement (the
/// kernel's code 7, undecided).
const ROOMLESS: [f32; 2] = [7.0, 2.0];
/// Parts to refine one round of a points' dispatch holds (32 bytes each): past it, a
/// point is left undecided.
const REFINE_ROOM: usize = 1 << 17;

/// The scatter's jobs, and its dispatches whose jobs did not fit (read by the gather).
pub static SCATTER_JOBS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SCATTER_OVERFLOWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// With ICAM_PASS_TIMES set, each scatter pass is submitted and waited for alone, and its wall
/// time (microseconds) summed here by pass: seed, jobs_of, scatter, leaves, refine_round,
/// edge_signs, finish (for `part-bench`).
pub static PASS_US: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
/// Parts the points' scatter sent to each refinement round.
pub static REFINE_JOBS: [std::sync::atomic::AtomicU64; 3] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nodes: u32,
    at: [u32; 11],
    eps: f32,
    tol: f32,
    mode: u32,
    ncand: u32,
    round: u32,
    pad: [u32; 3],
}

/// One brick as the kernel reads it: its first node and cell side, where its candidates
/// start and how many, where its values go.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrickRec {
    o: [f32; 4],
    c: [u32; 4],
}

/// A brick's candidates for the kernel, from its center, radius and the distance there.
type Lister = Box<dyn Fn(&Exact, V3, f64, f64) -> Option<Vec<u32>> + Send + Sync>;

pub struct PartGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// The same kernel refining the leaves no test decides (`points_refined`).
    refine_pipeline: wgpu::ComputePipeline,
    /// The scatter's passes (`bricks_scatter`), sharing one layout.
    scatter_layout: wgpu::BindGroupLayout,
    seed: wgpu::ComputePipeline,
    jobs: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    refine: wgpu::ComputePipeline,
    leaves: wgpu::ComputePipeline,
    edge_signs: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    geo: wgpu::Buffer,
    ints: wgpu::Buffer,
    at: [u32; 11],
    eps: f32,
    tol: f32,
    lister: Lister,
    origin: V3,
    pub adapter: String,
    /// B-spline sub-patches and curve pieces packed.
    pub nodes: usize,
}

/// Where to cut bricks into dispatches: at most `most`, and no more than DISPATCH_WORK
/// readings unless one brick alone has more (`work` of brick i).
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

impl PartGpu {
    /// Packs the evaluator relative to the middle of the part, its B-spline faces split
    /// for nodes within `need` of them (a few of the finest cell diagonals), and compiles
    /// the kernel.
    pub fn new(ex: &Exact, need: f64) -> Result<PartGpu, String> {
        let s = ex.s;
        let origin = (s.lo + s.hi) * 0.5;
        let mut pack = ex.gpu_pack(origin, need);
        let (geo, ints, at, nodes) = (std::mem::take(&mut pack.geo), std::mem::take(&mut pack.ints), pack.at, pack.nodes);
        let lister: Lister = Box::new(move |ex, c, r, d| pack.list(ex, c, r, d));
        // A few ulps of the largest coordinate the kernel sees.
        let eps = (8.0 * f32::EPSILON as f64 * (0.5 * (s.hi - s.lo).len()).max(1.0)) as f32;
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
            let limits = adapter.limits();
            let bytes = 4 * geo.len().max(ints.len()) as u64;
            if bytes > limits.max_storage_buffer_binding_size as u64 {
                return Err(format!("the packed part is {bytes} bytes, past the GPU's {} a buffer", limits.max_storage_buffer_binding_size));
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { label: Some("part"), required_features: wgpu::Features::empty(), required_limits: limits, ..Default::default() })
                .await
                .map_err(|e| format!("no GPU device: {e}"))?;
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("part"), source: wgpu::ShaderSource::Wgsl(Exact::gpu_wgsl().into()) });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("part"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let refining = wgpu::PipelineCompilationOptions { constants: &[("REFINE", 1.0)], ..Default::default() };
            let refine_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("part refining"), layout: None, module: &module, entry_point: Some("main"), compilation_options: refining, cache: None });
            let slot = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
            let (ro, rw) = (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false });
            let scatter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("part scatter"),
                entries: &[slot(0, wgpu::BufferBindingType::Uniform), slot(1, ro), slot(2, ro), slot(3, ro), slot(4, ro), slot(5, rw), slot(6, rw), slot(7, rw), slot(8, rw), slot(9, ro), slot(10, rw)],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("part scatter"), bind_group_layouts: &[Some(&scatter_layout)], immediate_size: 0 });
            let stage = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&pl), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
            let (seed, jobs, scatter, refine, leaves, edge_signs, finish) = (stage("seed"), stage("jobs_of"), stage("scatter"), stage("refine_round"), stage("leaves"), stage("edge_signs"), stage("finish"));
            if let Some(e) = scope.pop().await {
                return Err(format!("the part shader: {e}"));
            }
            let geo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("geo"), contents: bytemuck::cast_slice(&geo), usage: wgpu::BufferUsages::STORAGE });
            let ints = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ints"), contents: bytemuck::cast_slice(&ints), usage: wgpu::BufferUsages::STORAGE });
            Ok(PartGpu { device, queue, pipeline, refine_pipeline, scatter_layout, seed, jobs, scatter, refine, leaves, edge_signs, finish, geo, ints, at, eps, tol: s.tol as f32, lister, origin, adapter: format!("{} ({:?})", info.name, info.backend), nodes })
        })
    }

    /// Keeps the GPU busy for `secs` with a kernel of plain arithmetic, so its clocks are up
    /// when real work comes (an idle GPU takes about two seconds to reach them, measured on
    /// the T1000).
    pub fn warm(&self, secs: f64) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warm"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64) fn warm(@builtin(global_invocation_id) g: vec3<u32>) {
    var x = f32(g.x);
    for (var i = 0u; i < 4096u; i++) { x = fma(x, 1.0000001, 0.5); }
    if (x == 0.0) { o[0] = x; }
}"
                .into(),
            ),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("warm"), layout: None, module: &module, entry_point: Some("warm"), compilation_options: Default::default(), cache: None });
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("warm"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("warm"), layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
        let t = std::time::Instant::now();
        while t.elapsed().as_secs_f64() < secs {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(1024, 1, 1);
            }
            self.queue.submit([enc.finish()]);
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(5)) });
        }
    }

    /// The kernel's candidates for the points within `r` of `c`, given that the boundary
    /// comes within `d` of c; None when any must be read on the CPU.
    pub fn list(&self, ex: &Exact, c: V3, r: f64, d: f64) -> Option<Vec<u32>> {
        (self.lister)(ex, c, r, d)
    }

    /// Each brick's 729 nodes (x fastest), given its first node, its cell side and its
    /// candidates: the signed distance and 0, a lower bound on the distance past the
    /// clamp and 1, or 2.
    pub fn bricks(&self, bricks: &[(V3, f64, &[u32])]) -> Result<Vec<[f32; 2]>, String> {
        self.run(bricks, NODES, 0)
    }

    /// The same values as `bricks`, the B-spline spans and curve pieces read by the scatter
    /// (gpu.wgsl): each part decides once for a brick's nodes near it and writes its answer
    /// into them, every node keeping the least. A dispatch whose jobs do not fit is read
    /// by the gather.
    pub fn bricks_scatter(&self, bricks: &[(V3, f64, &[u32])]) -> Result<Vec<[f32; 2]>, String> {
        self.scatter_run(bricks, NODES, 0, BRICKS_AT_ONCE, false)
    }

    /// The same values as `points_refined`, by the scatter, a point's leaves no test decides
    /// refined in parallel (three rounds of 8 x 8 parts). Points a dispatch had no room for
    /// (its jobs or its refinement rounds full) are read again in dispatches of fewer points,
    /// an eighth as many each time, until they fit or go one at a time.
    pub fn points_scatter(&self, points: &[(V3, f64, &[u32])]) -> Result<Vec<[f32; 2]>, String> {
        self.points_scatter_capped(points, false)
    }

    /// The same, `capped`: each point's list holds only the features within its clamp (the
    /// `list` of its distance there), so past the clamp its answer is that bound alone.
    pub fn points_scatter_capped(&self, points: &[(V3, f64, &[u32])], capped: bool) -> Result<Vec<[f32; 2]>, String> {
        let mut out = self.scatter_run(points, 1, 1, POINTS_AT_ONCE, capped)?;
        let mut most = POINTS_AT_ONCE;
        while most > 1 {
            let again: Vec<usize> = (0..out.len()).filter(|&i| out[i] == ROOMLESS).collect();
            if again.is_empty() {
                break;
            }
            most = (most / 8).max(1);
            let redo: Vec<(V3, f64, &[u32])> = again.iter().map(|&i| points[i]).collect();
            for (&i, v) in again.iter().zip(self.scatter_run(&redo, 1, 1, most, capped)?) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    /// Bricks (`per` 729, mode 0) or points (`per` 1, mode 1, `most` at once) by the
    /// scatter's passes. A bricks' dispatch whose jobs do not fit is read by the gather; a
    /// points' dispatch's points come back ROOMLESS.
    fn scatter_run(&self, items: &[(V3, f64, &[u32])], per: usize, mode: u32, most: usize, capped: bool) -> Result<Vec<[f32; 2]>, String> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let dev = &self.device;
        // Queued dispatches: their items, their values and their job counts, mapped.
        let mut pending: Vec<(usize, usize, usize, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut out = Vec::with_capacity(items.len() * per);
        // The most jobs a root listed has opened into, in the dispatches back so far.
        let per_root = std::cell::Cell::new(0f64);
        let drain = |pending: &mut Vec<(usize, usize, usize, wgpu::Buffer, wgpu::Buffer)>, out: &mut Vec<[f32; 2]>| -> Result<(), String> {
            dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(120)) }).map_err(|e| format!("GPU did not finish: {e}"))?;
            for (a, z, roots, rbuf, cbuf) in pending.drain(..) {
                let counts: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&cbuf.slice(..).get_mapped_range().map_err(|e| format!("GPU results: {e:?}"))?).to_vec();
                SCATTER_JOBS.fetch_add(counts[0] as u64, std::sync::atomic::Ordering::Relaxed);
                per_root.set(per_root.get().max(counts[0] as f64 / roots.max(1) as f64));
                for (k, &c) in counts[1..].iter().enumerate() {
                    REFINE_JOBS[k].fetch_add(c as u64, std::sync::atomic::Ordering::Relaxed);
                }
                if counts[0] as usize > JOB_ROOM {
                    SCATTER_OVERFLOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if mode == 0 && z - a > 1 {
                        out.extend(self.scatter_run(&items[a..z], per, 0, (z - a).div_ceil(2), false)?);
                    } else if mode == 0 {
                        out.extend(self.run(&items[a..z], per, 0)?);
                    } else {
                        out.extend(std::iter::repeat_n(ROOMLESS, z - a));
                    }
                } else {
                    out.extend_from_slice(bytemuck::cast_slice::<u8, [f32; 2]>(&rbuf.slice(..).get_mapped_range().map_err(|e| format!("GPU results: {e:?}"))?));
                }
            }
            Ok(())
        };
        let grid = |n: usize| {
            let groups = (n as u32).div_ceil(64).max(1);
            let gx = groups.min(65535);
            (gx, groups.div_ceil(gx))
        };
        // Each dispatch planned as the last ones come back: a brick's closed-form features
        // (a list holds those first) and its spans and pieces; a point reads every candidate.
        let split = |i: usize| {
            let codes = items[i].2;
            let closed = codes.partition_point(|&c| c >> 28 < 3);
            (closed, codes.len() - closed)
        };
        let mut a = 0;
        while a < items.len() {
            let jobs_per_root = if per_root.get() > 0.0 { 1.5 * per_root.get() } else { JOBS_PER_ROOT };
            let (mut z, mut seed, mut roots, mut work) = (a, 0usize, 0usize, 0usize);
            while z < items.len() && z - a < most {
                let (closed, r) = split(z);
                let fits = if mode == 0 {
                    seed + per * closed <= SEED_WORK && (roots + r) as f64 * jobs_per_root <= 0.5 * JOB_ROOM as f64
                } else {
                    work + per * (closed + r).max(1) <= DISPATCH_WORK
                };
                if z > a && !fits {
                    break;
                }
                (seed, roots, work) = (seed + per * closed, roots + r, work + per * (closed + r).max(1));
                z += 1;
            }
            // Points' dispatches hold their refinement rounds too: fewer at once.
            if pending.len() >= if mode == 1 { 2 } else { SCATTER_QUEUED } {
                drain(&mut pending, &mut out)?;
            }
            let mut recs = Vec::with_capacity(z - a);
            let mut cands: Vec<u32> = Vec::new();
            let mut owner: Vec<u32> = Vec::new();
            for (k, &(p, w, codes)) in items[a..z].iter().enumerate() {
                let o = p - self.origin;
                recs.push(BrickRec { o: [o.x as f32, o.y as f32, o.z as f32, w as f32], c: [cands.len() as u32, codes.len() as u32, (k * per) as u32, capped as u32] });
                cands.extend_from_slice(codes);
                owner.extend(std::iter::repeat_n(k as u32, codes.len()));
            }
            let ncand = cands.len();
            if cands.is_empty() {
                cands.push(0);
                owner.push(0);
            }
            let n = (z - a) * per;
            let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
            let blank = |label: &str, size: u64, usage: wgpu::BufferUsages| dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
            let storage = wgpu::BufferUsages::STORAGE;
            // Params for each refinement round (only points refine).
            let rounds: u32 = if mode == 1 { 3 } else { 0 };
            let pbufs: Vec<wgpu::Buffer> = (0..rounds.max(1))
                .map(|round| init("params", bytemuck::bytes_of(&Params { nodes: n as u32, at: self.at, eps: self.eps, tol: self.tol, mode, ncand: ncand as u32, round, pad: [0; 3] }), wgpu::BufferUsages::UNIFORM))
                .collect();
            let bbuf = init("bricks", bytemuck::cast_slice(&recs), storage);
            let cbuf = init("cands", bytemuck::cast_slice(&cands), storage);
            let wbuf = init("owner", bytemuck::cast_slice(&owner), storage);
            let size = (n * 8) as u64;
            let obuf = blank("out", size, storage | wgpu::BufferUsages::COPY_SRC);
            let abuf = blank("keys", (n * 12) as u64, storage);
            let jbuf = blank("jobs", (JOB_ROOM * 8) as u64, storage);
            let nbuf = blank("job counts", 16, storage | wgpu::BufferUsages::COPY_SRC);
            let qbuf = blank("refine jobs", if mode == 1 { (3 * REFINE_ROOM * 32) as u64 } else { 32 }, storage);
            let rbuf = blank("read", size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let rcount = blank("read counts", 16, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let binds: Vec<wgpu::BindGroup> = pbufs
                .iter()
                .map(|pbuf| {
                    let entries: Vec<wgpu::BindGroupEntry> = [pbuf, &bbuf, &cbuf, &self.geo, &self.ints, &obuf, &abuf, &jbuf, &nbuf, &wbuf, &qbuf]
                        .iter()
                        .enumerate()
                        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                        .collect();
                    dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("part scatter"), layout: &self.scatter_layout, entries: &entries })
                })
                .collect();
            // Points take a thread a job, so fewer groups cover them.
            let jobs_grid = if mode == 1 { (SCATTER_GROUPS / 16, 1) } else { (SCATTER_GROUPS, 1) };
            let mut steps: Vec<(usize, &wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32))> = vec![(0, &self.seed, &binds[0], grid(n)), (1, &self.jobs, &binds[0], grid(ncand)), (2, &self.scatter, &binds[0], jobs_grid)];
            // Points' leaves to refine, once the scatter has written its answers.
            if mode == 1 {
                steps.push((3, &self.leaves, &binds[0], jobs_grid));
            }
            for b in binds.iter().take(rounds as usize) {
                steps.push((4, &self.refine, b, (SCATTER_GROUPS, 1)));
            }
            steps.push((5, &self.edge_signs, &binds[0], jobs_grid));
            steps.push((6, &self.finish, &binds[0], grid(n)));
            let timed = std::env::var_os("ICAM_PASS_TIMES").is_some();
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("part scatter") });
            for (which, pipeline, bind, (gx, gy)) in steps {
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("part scatter"), timestamp_writes: None });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                if timed {
                    let t = std::time::Instant::now();
                    self.queue.submit([enc.finish()]);
                    let _ = dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(120)) });
                    PASS_US[which].fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                    enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("part scatter") });
                }
            }
            enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, size);
            enc.copy_buffer_to_buffer(&nbuf, 0, &rcount, 0, 16);
            self.queue.submit([enc.finish()]);
            rbuf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            rcount.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            pending.push((a, z, roots, rbuf, rcount));
            a = z;
        }
        drain(&mut pending, &mut out)?;
        Ok(out)
    }

    /// Each point's value, given the distance past which a lower bound will do and its
    /// candidates: the signed distance and 0, a lower bound past that distance and 1, or 2.
    pub fn points(&self, points: &[(V3, f64, &[u32])]) -> Result<Vec<[f32; 2]>, String> {
        self.run(points, 1, 1)
    }

    /// The same, each point refining in place the leaves no test decided for it (for the
    /// few nodes a brick left undecided).
    pub fn points_refined(&self, points: &[(V3, f64, &[u32])]) -> Result<Vec<[f32; 2]>, String> {
        self.run(points, 1, 2)
    }

    /// Bricks (`per` 729) or points (`per` 1). Dispatches are queued QUEUED at a time
    /// before their results are read back.
    fn run(&self, bricks: &[(V3, f64, &[u32])], per: usize, mode: u32) -> Result<Vec<[f32; 2]>, String> {
        if bricks.is_empty() {
            return Ok(Vec::new());
        }
        let dev = &self.device;
        let pipeline = if mode == 2 { &self.refine_pipeline } else { &self.pipeline };
        let layout = pipeline.get_bind_group_layout(0);
        let mut reads = Vec::new();
        let mut out = Vec::with_capacity(bricks.len() * per);
        let drain = |reads: &mut Vec<wgpu::Buffer>, out: &mut Vec<[f32; 2]>| -> Result<(), String> {
            dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(120)) }).map_err(|e| format!("GPU did not finish: {e}"))?;
            for rbuf in reads.drain(..) {
                let data = rbuf.slice(..).get_mapped_range().map_err(|e| format!("GPU results: {e:?}"))?;
                out.extend_from_slice(bytemuck::cast_slice::<u8, [f32; 2]>(&data));
            }
            Ok(())
        };
        for (a, z) in dispatches(bricks.len(), BRICKS_AT_ONCE * NODES / per, |i| per * bricks[i].2.len().max(1)) {
            if reads.len() >= QUEUED {
                drain(&mut reads, &mut out)?;
            }
            let mut recs = Vec::with_capacity(z - a);
            let mut cands: Vec<u32> = Vec::new();
            for (k, &(p, h, codes)) in bricks[a..z].iter().enumerate() {
                let o = p - self.origin;
                recs.push(BrickRec { o: [o.x as f32, o.y as f32, o.z as f32, h as f32], c: [cands.len() as u32, codes.len() as u32, (k * per) as u32, 0] });
                cands.extend_from_slice(codes);
            }
            if cands.is_empty() {
                cands.push(0);
            }
            let n = (z - a) * per;
            let groups = (n as u32).div_ceil(64);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let params = Params { nodes: n as u32, at: self.at, eps: self.eps, tol: self.tol, mode, ncand: 0, round: 0, pad: [0; 3] };
            let pbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
            let bbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bricks"), contents: bytemuck::cast_slice(&recs), usage: wgpu::BufferUsages::STORAGE });
            let cbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("cands"), contents: bytemuck::cast_slice(&cands), usage: wgpu::BufferUsages::STORAGE });
            let size = (n * 8) as u64;
            let obuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("out"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let rbuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let entries: Vec<wgpu::BindGroupEntry> = [&pbuf, &bbuf, &cbuf, &self.geo, &self.ints, &obuf]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("part"), layout: &layout, entries: &entries });
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("part") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("part"), timestamp_writes: None });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, size);
            self.queue.submit([enc.finish()]);
            rbuf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            reads.push(rbuf);
        }
        drain(&mut reads, &mut out)?;
        Ok(out)
    }
}
