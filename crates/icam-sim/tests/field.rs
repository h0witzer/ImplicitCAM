//! The GPU stock field against the exact 64-bit height, on a program with every kind of
//! sweep: a flat end mill along a level line, down a ramp, round an arc and a helix; a
//! bull nose level and on a ramp; a drill point. Skips itself without a GPU.

use std::collections::BTreeMap;

use icam_post::{ToolDef, V3};
use icam_sim::{check, GpuStock, Grid};
use icam_stock::ImplicitStock;

const PROGRAM: &str = "
G21 G90 G17
T1 M06
G00 X10 Y10 Z25
G01 Z15 F600
G01 X60 Y20 Z15
G01 X80 Y60 Z10
G02 X60 Y80 I-20 J0
G03 X40 Y60 Z6 I0 J-20
G00 Z25
T2 M06
G00 X20 Y80 Z25
G01 Z12 F600
G01 X40 Y90 Z12
G01 X50 Y70 Z9
G00 Z25
T3 M06
G00 X85 Y15 Z25
G81 X85 Y15 Z8 R18 F200
G80
G00 Z25
";

fn tool(number: i32, kind: &str, diameter: f64, corner: f64) -> ToolDef {
    ToolDef { number, kind: kind.into(), diameter, corner_radius: corner, description: String::new(), tip_angle_deg: 118.0 }
}

#[test]
fn the_gpu_field_is_the_exact_height() {
    let gpu = match icam_gpu::Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut warnings = Vec::new();
    let moves = icam_post::parse(PROGRAM, &mut warnings);
    let tools = BTreeMap::from([(1, tool(1, "ENDMILL", 10.0, 0.0)), (2, tool(2, "ENDMILL", 10.0, 2.0)), (3, tool(3, "DRILL", 6.0, 0.0))]);
    let stock = ImplicitStock::new(V3::new(0.0, 0.0, 0.0), V3::new(100.0, 100.0, 20.0), &moves, &tools, ImplicitStock::CELL_MM);
    let grid = Grid::over(0.0, 0.0, 100.0, 100.0, 0.25);
    let (field, _) = GpuStock::new(&gpu.device, &stock).field(&gpu.device, &gpu.queue, grid).unwrap();
    // Every kind of sweep cut something.
    let cut_below = |z: f32| field.h.iter().filter(|&&h| h < z).count();
    assert!(cut_below(15.001) > 1000 && cut_below(9.0) > 50 && cut_below(8.5) > 0, "the program did not cut: {warnings:?}");
    let (differ, off) = check(&stock, &field, 0.001, 0.002);
    assert!(off.is_empty(), "{} of {differ} differing nodes are off the walls, worst {:?}", off.len(), off.iter().map(|o| o.0).fold(0.0, f64::max));
}
