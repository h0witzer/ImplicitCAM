//! The signed implicit polynomial of a span (icam_core::span_poly) on the GPU.
//!
//! One compute pipeline per polynomial degree, generated as straight-line WGSL:
//! the powers of the normalized coordinates, then one fused multiply-add per
//! monomial with the coefficient read from the span's record (a loop over a
//! coefficient table is the interpreted form, measured 58 times slower on this
//! project's card in the Prospero probe). One thread evaluates one point against
//! one span. Points and span centers are f32 in the part's frame (relative to an
//! origin the caller picks), the kernel f32 throughout; the f64 CPU evaluation
//! (`SpanPoly::value`) is the reference it is checked against.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use icam_core::curve3::V3;
use icam_core::span_poly::{monomials, SpanPoly};
use wgpu::util::DeviceExt;

use crate::Gpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    nspan: u32,
    stride: u32,
    pad: u32,
}

/// The WGSL of the degree-k kernel.
pub fn wgsl(k: usize) -> String {
    let mons = monomials(k);
    let mut s = String::new();
    s.push_str(
        "struct Params { n: u32, nspan: u32, stride: u32, pad: u32 }\n\
         @group(0) @binding(0) var<uniform> P: Params;\n\
         @group(0) @binding(1) var<storage, read> pts: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> sp: array<f32>;\n\
         @group(0) @binding(3) var<storage, read_write> out: array<f32>;\n\
         @compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n\
         \x20 let i = g.x;\n\
         \x20 let j = g.y;\n\
         \x20 if (i >= P.n || j >= P.nspan) { return; }\n\
         \x20 let b = j * P.stride;\n\
         \x20 let p = pts[i].xyz;\n\
         \x20 let s = sp[b + 3u];\n\
         \x20 let x1 = (p.x - sp[b]) * s;\n\
         \x20 let y1 = (p.y - sp[b + 1u]) * s;\n\
         \x20 let z1 = (p.z - sp[b + 2u]) * s;\n",
    );
    for v in ["x", "y", "z"] {
        for e in 2..=k {
            s.push_str(&format!("  let {v}{e} = {v}{} * {v}1;\n", e - 1));
        }
    }
    s.push_str("  var f = 0.0;\n");
    for (idx, m) in mons.iter().enumerate() {
        let parts: Vec<String> = [("x", m[0]), ("y", m[1]), ("z", m[2])].iter().filter(|(_, e)| *e > 0).map(|(v, e)| format!("{v}{e}")).collect();
        let term = if parts.is_empty() { "1.0".to_string() } else { parts.join(" * ") };
        s.push_str(&format!("  f = fma(sp[b + {}u], {term}, f);\n", 4 + idx));
    }
    s.push_str("  out[j * P.n + i] = f;\n}\n");
    s
}

/// Pipelines by degree, built on first use.
pub struct SpanPolyGpu {
    pipelines: HashMap<usize, wgpu::ComputePipeline>,
}

/// Timing of one `eval`: the dispatches only (inputs already uploaded), and the
/// whole call.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolyTiming {
    pub kernel_s: f64,
    pub total_s: f64,
}

impl SpanPolyGpu {
    pub fn new() -> SpanPolyGpu {
        SpanPolyGpu { pipelines: HashMap::new() }
    }

    fn pipeline(&mut self, gpu: &Gpu, k: usize) -> &wgpu::ComputePipeline {
        self.pipelines.entry(k).or_insert_with(|| {
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("span poly"), source: wgpu::ShaderSource::Wgsl(wgsl(k).into()) });
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("span poly"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        })
    }

    /// F of every span (all of one degree) at every point, f32, span-major:
    /// out[j * pts.len() + i]. Points and centers go to f32 relative to `origin`.
    pub fn eval(&mut self, gpu: &Gpu, polys: &[&SpanPoly], origin: V3, pts: &[V3], repeats: usize) -> Result<(Vec<f32>, PolyTiming), String> {
        let k = polys.first().ok_or("no spans")?.degree;
        if polys.iter().any(|p| p.degree != k) {
            return Err("spans of one degree per call".into());
        }
        let t0 = std::time::Instant::now();
        let m = monomials(k).len();
        let stride = 4 + m;
        let mut rec: Vec<f32> = Vec::with_capacity(polys.len() * stride);
        for p in polys {
            rec.extend([(p.center.x - origin.x) as f32, (p.center.y - origin.y) as f32, (p.center.z - origin.z) as f32, (1.0 / p.scale) as f32]);
            rec.extend(p.coef.iter().map(|&c| c as f32));
        }
        let q: Vec<[f32; 4]> = pts.iter().map(|p| [(p.x - origin.x) as f32, (p.y - origin.y) as f32, (p.z - origin.z) as f32, 0.0]).collect();
        let n = pts.len();
        if n > 65535 * 64 || polys.len() > 65535 {
            return Err("at most 4,194,240 points and 65,535 spans per call".into());
        }
        let dev = &gpu.device;
        let pbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&Params { n: n as u32, nspan: polys.len() as u32, stride: stride as u32, pad: 0 }), usage: wgpu::BufferUsages::UNIFORM });
        let qbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("points"), contents: bytemuck::cast_slice(&q), usage: wgpu::BufferUsages::STORAGE });
        let sbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("spans"), contents: bytemuck::cast_slice(&rec), usage: wgpu::BufferUsages::STORAGE });
        let size = (n * polys.len() * 4) as u64;
        let obuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("values"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rbuf = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let pipeline = self.pipeline(gpu, k).clone();
        let layout = pipeline.get_bind_group_layout(0);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("span poly"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: qbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: obuf.as_entire_binding() },
            ],
        });
        gpu.wait()?;
        let mut kernel_s = f64::INFINITY;
        for _ in 0..repeats.max(1) {
            let t1 = std::time::Instant::now();
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("span poly") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("span poly"), timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), polys.len() as u32, 1);
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait()?;
            kernel_s = kernel_s.min(t1.elapsed().as_secs_f64());
        }
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
        enc.copy_buffer_to_buffer(&obuf, 0, &rbuf, 0, size);
        gpu.queue.submit([enc.finish()]);
        rbuf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu.wait()?;
        let out: Vec<f32> = {
            let data = rbuf.slice(..).get_mapped_range().map_err(|e| format!("map: {e:?}"))?;
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        rbuf.unmap();
        Ok((out, PolyTiming { kernel_s, total_s: t0.elapsed().as_secs_f64() }))
    }
}

impl Default for SpanPolyGpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_wgsl_parses() {
        for k in 1..=8 {
            let src = wgsl(k);
            assert_eq!(src.matches("fma(").count(), monomials(k).len());
            // Every power used is defined.
            for v in ["x", "y", "z"] {
                for e in 2..=k {
                    assert!(src.contains(&format!("let {v}{e} =")));
                }
            }
        }
    }
}
