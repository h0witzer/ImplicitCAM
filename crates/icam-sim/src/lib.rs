//! icam-sim: the machined stock and the part as fields on one grid, built once on the
//! GPU, for looking at a program.
//!
//! With a vertical tool (3-axis) the stock a program leaves is a height field,
//! H(x, y) = min over sweeps of the lowest tool-bottom height each one reaches (see
//! `icam_stock::ImplicitStock`). Every grid node is independent, so the field is one
//! GPU pass over the nodes, each reading the few sweeps its candidate cell lists. The
//! viewer draws from the finished field: a one-time build, then plain draws.
//!
//! `StockField::cpu` is the reference: the exact 64-bit `cut_height` at every node.
//!
//! `part` puts the part on the same grid, from its compiled F-rep, and says what the
//! program left at every node: material, on size, or a gouge.

pub mod compare;
pub mod part;
pub mod picture;

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use icam_stock::ImplicitStock;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

/// Grid nodes at (x0 + i step, y0 + j step), i < nx, j < ny, in the stock's frame (mm).
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub x0: f64,
    pub y0: f64,
    pub step: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Grid {
    /// Nodes over the box (x0, y0)..(x1, y1) at `step`.
    pub fn over(x0: f64, y0: f64, x1: f64, y1: f64, step: f64) -> Grid {
        Grid { x0, y0, step, nx: ((x1 - x0) / step).floor() as usize + 1, ny: ((y1 - y0) / step).floor() as usize + 1 }
    }

    pub fn x(&self, i: usize) -> f64 {
        self.x0 + i as f64 * self.step
    }

    pub fn y(&self, j: usize) -> f64 {
        self.y0 + j as f64 * self.step
    }
}

/// Stock heights (mm, the stock's frame) at the nodes of a grid, row by row.
pub struct StockField {
    pub grid: Grid,
    pub h: Vec<f32>,
}

impl StockField {
    /// The reference: the exact 64-bit cut height at every node.
    pub fn cpu(stock: &ImplicitStock, grid: Grid) -> StockField {
        let mut h = vec![0f32; grid.nx * grid.ny];
        h.par_chunks_mut(grid.nx).enumerate().for_each(|(j, row)| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = stock.cut_height(grid.x(i), grid.y(j)) as f32;
            }
        });
        StockField { grid, h }
    }
}

/// A field held to the exact 64-bit height: (nodes whose height differs by more than
/// `gate`, those of them not on a vertical wall). A node on a wall has two heights a
/// hair apart; it passes when the field's height is the exact one at some point within
/// `wall` of the node (16 directions).
pub fn check(stock: &ImplicitStock, field: &StockField, gate: f64, wall: f64) -> (usize, Vec<(f64, usize)>) {
    let g = field.grid;
    let differ: Vec<(f64, usize)> = (0..g.nx * g.ny)
        .into_par_iter()
        .filter_map(|k| {
            let d = (stock.cut_height(g.x(k % g.nx), g.y(k / g.nx)) - field.h[k] as f64).abs();
            (d > gate).then_some((d, k))
        })
        .collect();
    let off: Vec<(f64, usize)> = differ
        .iter()
        .copied()
        .filter(|&(_, k)| {
            let (x, y, h) = (g.x(k % g.nx), g.y(k / g.nx), field.h[k] as f64);
            !(0..16).any(|a| {
                let t = a as f64 * std::f64::consts::PI / 8.0;
                (stock.cut_height(x + wall * t.cos(), y + wall * t.sin()) - h).abs() <= gate
            })
        })
        .collect();
    (differ.len(), off)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    row0: u32,
    gnx: u32,
    gny: u32,
    pad: [u32; 3],
    x0: f32,
    y0: f32,
    step: f32,
    cell: f32,
    ox: f32,
    oy: f32,
    top: f32,
    pad3: f32,
}

/// What a GPU build took: uploading the sweeps and grid, the passes, reading back (s).
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub upload: f64,
    pub compute: f64,
    pub readback: f64,
    pub passes: usize,
}

/// The stock's sweeps and candidate grid on the GPU, ready to fill any grid.
pub struct GpuStock {
    sweeps: wgpu::Buffer,
    starts: wgpu::Buffer,
    items: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    origin: (f64, f64, f64),
    top: f32,
    cell: f32,
    ox: f32,
    oy: f32,
    gnx: u32,
    gny: u32,
    pub upload_s: f64,
}

/// Rows per pass, so no pass runs long enough for the OS to reset the GPU and no
/// buffer grows past what a binding may hold.
const NODES_PER_PASS: usize = 1 << 22;

impl GpuStock {
    pub fn new(device: &wgpu::Device, stock: &ImplicitStock) -> GpuStock {
        let clock = Instant::now();
        let t = stock.tables();
        // An empty storage buffer is not allowed: keep one zero.
        let pad_u32 = |v: &[u32]| if v.is_empty() { vec![0u32] } else { v.to_vec() };
        let pad_f32 = |v: &[f32]| if v.is_empty() { vec![0f32; 16] } else { v.to_vec() };
        let buf = |label: &str, bytes: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage: wgpu::BufferUsages::STORAGE });
        let sweeps = buf("sweeps", bytemuck::cast_slice(&pad_f32(&t.sweeps)));
        let starts = buf("starts", bytemuck::cast_slice(&pad_u32(&t.cell_start)));
        let items = buf("items", bytemuck::cast_slice(&pad_u32(&t.cell_items)));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("stock"), source: wgpu::ShaderSource::Wgsl(include_str!("stock.wgsl").into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stock"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        GpuStock {
            sweeps,
            starts,
            items,
            pipeline,
            origin: (t.origin.x, t.origin.y, t.origin.z),
            top: t.top,
            cell: t.cell,
            ox: t.ox,
            oy: t.oy,
            gnx: t.nx,
            gny: t.ny,
            upload_s: clock.elapsed().as_secs_f64(),
        }
    }

    /// The stock field on `grid`, in passes of whole rows.
    pub fn field(&self, device: &wgpu::Device, queue: &wgpu::Queue, grid: Grid) -> Result<(StockField, Timing), String> {
        let mut timing = Timing { upload: self.upload_s, ..Timing::default() };
        let rows_per_pass = (NODES_PER_PASS / grid.nx.max(1)).max(1);
        let mut h = vec![0f32; grid.nx * grid.ny];
        let (mut compute, mut readback) = (0.0, 0.0);
        let mut row0 = 0;
        while row0 < grid.ny {
            let rows = rows_per_pass.min(grid.ny - row0);
            let n = grid.nx * rows;
            let p = Params {
                nx: grid.nx as u32,
                ny: rows as u32,
                row0: row0 as u32,
                gnx: self.gnx,
                gny: self.gny,
                pad: [0; 3],
                x0: (grid.x0 - self.origin.0) as f32,
                y0: (grid.y0 - self.origin.1) as f32,
                step: grid.step as f32,
                cell: self.cell,
                ox: self.ox,
                oy: self.oy,
                top: self.top,
                pad3: 0.0,
            };
            let clock = Instant::now();
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });
            let size = (n * 4) as u64;
            let out = device.create_buffer(&wgpu::BufferDescriptor { label: Some("heights"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stock"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.sweeps.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.starts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.items.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
                ],
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stock") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("stock"), timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((grid.nx as u32).div_ceil(16), (rows as u32).div_ceil(16), 1);
            }
            enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map_err(|e| format!("GPU did not finish: {e}"))?;
            compute += clock.elapsed().as_secs_f64();
            let clock = Instant::now();
            let slice = read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map_err(|e| format!("GPU did not map: {e}"))?;
            {
                let data = slice.get_mapped_range().map_err(|e| format!("GPU results did not map: {e:?}"))?;
                let vals: &[f32] = bytemuck::cast_slice(&data);
                for (k, v) in vals.iter().enumerate() {
                    h[row0 * grid.nx + k] = v + self.origin.2 as f32;
                }
            }
            read.unmap();
            readback += clock.elapsed().as_secs_f64();
            timing.passes += 1;
            row0 += rows;
        }
        timing.compute = compute;
        timing.readback = readback;
        Ok((StockField { grid, h }, timing))
    }
}
