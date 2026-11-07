//! GPU kernels for the ImplicitCAM contact query (wgpu, WGSL).
//!
//! `Gpu::new` opens the high-performance adapter. `GpuPart::new` uploads a
//! packed part once; `GpuPart::query` runs the contact kernel for one tool over
//! a batch of query points and returns full contact records.

pub mod pack;
pub mod span_poly;

use bytemuck::{Pod, Zeroable};
use icam_core::geom::V2;
use icam_core::part::Part;
use icam_core::tool::Tool;
use wgpu::util::DeviceExt;

pub use pack::{pack, pack_tool, Packed};

pub const NONE: u32 = u32::MAX;
pub const CLAMP_COVER: u32 = 1;
pub const CLAMP_BAND: u32 = 2;
pub const STEEP: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Drop,
    Slice,
}

/// A contact record read back from the GPU, in the part's own frame.
#[derive(Clone, Copy, Debug)]
pub struct GpuContact {
    pub value: f64,
    pub grad: V2,
    pub second: f64,
    pub hess: [f64; 3],
    pub z_face: f64,
    pub q: V2,
    pub flags: u32,
    pub face: u32,
    pub edge: u32,
    pub tool_el: u32,
    pub second_face: u32,
    /// Parameter of the contact on its curve (spline: the source NURBS parameter), NaN if none.
    pub t: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    gx0: f32,
    gy0: f32,
    cell: f32,
    band: f32,
    nx: u32,
    ny: u32,
    mode: u32,
    n: u32,
    cell_start: u32,
    cell_items: u32,
    cover_start: u32,
    cover_items: u32,
    ntool: u32,
    r_max: f32,
    width: u32,
    cell_ref: u32,
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: String,
    pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("no GPU adapter: {e}"))?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("icam"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("no GPU device: {e}"))?;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("contact"),
                source: wgpu::ShaderSource::Wgsl(include_str!("contact.wgsl").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("contact"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok(Gpu { device, queue, adapter: format!("{} ({:?})", info.name, info.backend), pipeline })
        })
    }

    /// Waits for the queue; a kernel that runs away (or a device reset by the OS
    /// watchdog) comes back as an error instead of a hang.
    pub(crate) fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(20)) })
            .map(|_| ())
            .map_err(|e| format!("GPU did not finish: {e}"))
    }
}

pub struct GpuPart {
    pub packed: Packed,
    edges: wgpu::Buffer,
    face_z: wgpu::Buffer,
    idx: wgpu::Buffer,
    bez: wgpu::Buffer,
}

/// Timing of one `query` call.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    /// Submit to completion of the dispatches only (queries already uploaded).
    pub kernel_s: f64,
    pub upload_s: f64,
    pub readback_s: f64,
}

impl GpuPart {
    pub fn new(gpu: &Gpu, part: &Part, band: f64, cell: f64) -> Result<GpuPart, String> {
        let packed = pack(part, band, cell)?;
        let mk = |label: &str, bytes: &[u8]| {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let edges = mk("edges", bytemuck::cast_slice(&packed.edges));
        let face_z = mk("face_z", bytemuck::cast_slice(&packed.face_z));
        let idx = mk("idx", bytemuck::cast_slice(&packed.idx));
        let bez = mk("bez", bytemuck::cast_slice(&packed.bez));
        Ok(GpuPart { packed, edges, face_z, idx, bez })
    }

    /// Runs the contact query for `tool` at each (x, y, z) (z is the tip height,
    /// used by Slice only). Coordinates are in the part's own frame.
    pub fn query(&self, gpu: &Gpu, tool: &Tool, mode: Mode, pts: &[[f64; 3]]) -> Result<(Vec<GpuContact>, Timing), String> {
        let mut timing = Timing::default();
        let o = self.packed.origin;
        let tool_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tool"),
            contents: bytemuck::cast_slice(&pack_tool(tool)),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lim = gpu.device.limits();
        let max_bytes = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size) as usize;
        // At most 65536 queries per dispatch: keeps every submission far below the
        // OS GPU watchdog (about 2 s on Windows) even on heavy B-spline parts.
        let chunk = (max_bytes / 64).clamp(64, 1 << 16);
        let mut out = Vec::with_capacity(pts.len());
        let trace = std::env::var("ICAM_TRACE").is_ok();
        for part_pts in pts.chunks(chunk) {
            if trace {
                eprintln!("trace: chunk of {} queries", part_pts.len());
            }
            let n = part_pts.len();
            let t0 = std::time::Instant::now();
            let q: Vec<[f32; 4]> = part_pts.iter().map(|p| [(p[0] - o.x) as f32, (p[1] - o.y) as f32, p[2] as f32, 0.0]).collect();
            let qbuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("queries"),
                contents: bytemuck::cast_slice(&q),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let size = (n * 64) as u64;
            let obuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let rbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("read"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let groups = (n as u32).div_ceil(64);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let pk = &self.packed;
            let params = Params {
                gx0: pk.gx0 as f32,
                gy0: pk.gy0 as f32,
                cell: pk.cell as f32,
                band: pk.band as f32,
                nx: pk.nx,
                ny: pk.ny,
                mode: if mode == Mode::Drop { 0 } else { 1 },
                n: n as u32,
                cell_start: pk.cell_start,
                cell_items: pk.cell_items,
                cover_start: pk.cover_start,
                cover_items: pk.cover_items,
                ntool: tool.els.len() as u32,
                r_max: tool.r_max() as f32,
                width: gx * 64,
                cell_ref: pk.cell_ref,
            };
            let pbuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let layout = gpu.pipeline.get_bind_group_layout(0);
            let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("contact"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.edges.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.face_z.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.idx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: tool_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: qbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: obuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.bez.as_entire_binding() },
                ],
            });
            gpu.wait()?;
            timing.upload_s += t0.elapsed().as_secs_f64();
            if trace {
                eprintln!("trace: uploaded; dispatching {gx} x {gy} groups");
            }

            let t1 = std::time::Instant::now();
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("contact") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("contact"), timestamp_writes: None });
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait()?;
            timing.kernel_s += t1.elapsed().as_secs_f64();
            if trace {
                eprintln!("trace: kernel done in {:.3} s", t1.elapsed().as_secs_f64());
            }

            let t2 = std::time::Instant::now();
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
            enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, size);
            gpu.queue.submit([enc.finish()]);
            rbuf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            gpu.wait()?;
            {
                let data = rbuf.slice(..).get_mapped_range().expect("map GPU results");
                let v: &[[f32; 16]] = bytemuck::cast_slice(&data);
                for r in v {
                    let b = |x: f32| x.to_bits();
                    out.push(GpuContact {
                        value: r[0] as f64,
                        grad: V2::new(r[1] as f64, r[2] as f64),
                        second: r[3] as f64,
                        hess: [r[4] as f64, r[5] as f64, r[6] as f64],
                        z_face: r[7] as f64,
                        q: V2::new(r[8] as f64 + o.x, r[9] as f64 + o.y),
                        flags: b(r[10]),
                        face: b(r[12]),
                        edge: b(r[13]),
                        tool_el: b(r[14]),
                        second_face: b(r[15]),
                        t: if r[11] <= -1e37 { f64::NAN } else { r[11] as f64 },
                    });
                }
            }
            rbuf.unmap();
            timing.readback_s += t2.elapsed().as_secs_f64();
        }
        Ok((out, timing))
    }
}
