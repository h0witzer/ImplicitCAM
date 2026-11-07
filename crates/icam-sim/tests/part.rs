//! The part on the grid and what a program left, on a part and program built so every
//! answer is known: a sheet with a pocket down to z 3 and an island back up to the top
//! (icam-query's island part), cut by a 10 mm end mill that runs the pocket's walls to
//! size, leaves a corner of it uncut, dips 0.5 mm under the floor along one line and
//! runs 0.6 mm into one wall. The stock box is wider than the sheet, so its rim is waste.

use std::collections::BTreeMap;

use icam_core::frep::compile_prismatic;
use icam_core::geom::{Curve, V2};
use icam_core::part::Part;
use icam_post::{ToolDef, V3};
use icam_sim::part::{self, Class, PartField};
use icam_sim::{Grid, StockField};
use icam_stock::ImplicitStock;

fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Curve> {
    let p = [V2::new(x0, y0), V2::new(x1, y0), V2::new(x1, y1), V2::new(x0, y1)];
    (0..4).map(|i| Curve::Line { a: p[i], b: p[(i + 1) % 4] }).collect()
}

fn reversed(l: Vec<Curve>) -> Vec<Curve> {
    l.into_iter()
        .rev()
        .map(|c| match c {
            Curve::Line { a, b } => Curve::Line { a: b, b: a },
            other => other,
        })
        .collect()
}

/// Sheet (-30,-30)-(120,90), 19.05 thick; pocket (0,0)-(90,60) down to z 3; island
/// (40,22)-(55,38) back up to the top.
fn island_part() -> Part {
    let mut p = Part::default();
    p.add_face(1, 19.05, vec![square(-30.0, -30.0, 120.0, 90.0), reversed(square(0.0, 0.0, 90.0, 60.0))], None, 1e-9).unwrap();
    for (i, e) in p.edges.iter_mut().enumerate() {
        e.wall_zlo = if i < 4 { 0.0 } else { 3.0 };
    }
    p.faces[0].up = Some(true);
    let n = p.edges.len();
    p.add_face(2, 19.05, vec![square(40.0, 22.0, 55.0, 38.0)], None, 1e-9).unwrap();
    for e in &mut p.edges[n..] {
        e.wall_zlo = 3.0;
    }
    p.faces[1].up = Some(true);
    let n = p.edges.len();
    p.add_face(3, 3.0, vec![square(0.0, 0.0, 90.0, 60.0), reversed(square(40.0, 22.0, 55.0, 38.0))], None, 1e-9).unwrap();
    for e in &mut p.edges[n..] {
        e.wall_zlo = 3.0;
    }
    p.faces[2].up = Some(true);
    p.add_face(4, 0.0, vec![reversed(square(-30.0, -30.0, 120.0, 90.0))], None, 1e-9).unwrap();
    p.faces[3].up = Some(false);
    p
}

const PROGRAM: &str = "
G21 G90 G17
T1 M06
G00 X5 Y5 Z25
G01 Z3 F600
G01 X85 Y5
G00 Z25
G00 X5 Y12
G01 Z3
G01 X85 Y12
G00 Z25
G00 X5 Y5
G01 Z3
G01 X5 Y55
G01 X55 Y55
G00 Z25
G00 X12 Y5
G01 Z3
G01 X12 Y55
G00 Z25
G00 X15 Y30
G01 Z2.5
G01 X28 Y30
G00 Z25
G00 X85.65 Y10
G01 Z3
G01 X85.65 Y30
G00 Z25
";

#[test]
fn what_the_program_left_is_known() {
    let frep = compile_prismatic(&island_part(), 1e-9).unwrap();
    let mut warnings = Vec::new();
    let moves = icam_post::parse(PROGRAM, &mut warnings);
    let tool = ToolDef { number: 1, kind: "ENDMILL".into(), diameter: 10.0, corner_radius: 0.0, description: String::new(), tip_angle_deg: 118.0 };
    let tools = BTreeMap::from([(1, tool)]);
    let stock = ImplicitStock::new(V3::new(-40.0, -40.0, 0.0), V3::new(130.0, 100.0, 19.05), &moves, &tools, ImplicitStock::CELL_MM);
    let grid = Grid::over(-40.0, -40.0, 130.0, 100.0, 0.2);
    let field = StockField::cpu(&stock, grid);
    let pf = PartField::new(&frep, grid);
    let left = part::left_field(&frep, &pf, &field);

    // The part's top: the blocks agree with every column read at every node.
    for k in 0..grid.nx * grid.ny {
        let (x, y) = (grid.x(k % grid.nx), grid.y(k / grid.nx));
        let top = PartField::top_at(&frep, &pf.columns, x, y) as f32;
        assert!(top == pf.top[k] || (top.is_nan() && pf.top[k].is_nan()), "top at ({x}, {y}): {} by blocks, {top} read", pf.top[k]);
    }
    assert!(pf.reads < grid.nx * grid.ny / 4, "{} section reads for {} nodes: the blocks are not skipping", pf.reads, grid.nx * grid.ny);

    let at = |x: f64, y: f64| {
        let k = ((y - grid.y0) / grid.step).round() as usize * grid.nx + ((x - grid.x0) / grid.step).round() as usize;
        (left[k] as f64, pf.top[k])
    };
    assert_eq!(Class::of(at(-35.0, -35.0).0), Class::Waste, "the rim outside the sheet");
    assert!(at(-35.0, -35.0).1.is_nan(), "no part under the rim");
    let (corner, top) = at(75.0, 50.0);
    assert!((corner - 16.05).abs() < 1e-3 && top == 3.0, "the uncut corner keeps 16.05 mm over the floor: {corner} over {top}");
    assert_eq!(Class::of(at(47.0, 30.0).0), Class::OnSize, "the island, uncut, is on size");
    assert_eq!(Class::of(at(30.0, 8.0).0), Class::OnSize, "the floor cut at z 3");
    assert_eq!(Class::of(at(-0.2, 30.0).0), Class::OnSize, "the wall the tool ran to size");

    // Exactly two gouges: 0.5 mm under the floor, 0.6 mm into the wall at x 90.
    let spots = part::spots(&grid, &left);
    assert_eq!(spots.len(), 2, "{spots:?}");
    let floor = spots.iter().find(|s| (s.depth - 0.5).abs() < 1e-3).expect("the floor gouge, 0.5 mm deep");
    let wall = spots.iter().find(|s| (s.depth - 0.6).abs() < 2e-3).expect("the wall gouge, 0.6 mm deep");
    let x = |s: &part::Spot| grid.x(s.deepest % grid.nx);
    assert!((10.0..=33.0).contains(&x(floor)), "floor gouge at x {}", x(floor));
    assert!((x(wall) - 90.6).abs() < 1e-6, "wall gouge deepest at x {}", x(wall));
    // The floor gouge covers the tool's footprint along its line: 13 x 10 plus a 5 mm disk.
    let area = floor.nodes as f64 * grid.step * grid.step;
    let want = 13.0 * 10.0 + std::f64::consts::PI * 25.0;
    assert!((area - want).abs() < 0.03 * want, "floor gouge {area:.1} mm2, want about {want:.1}");

    // The still colors them so: red over the gouge, blue over the uncut corner, green on
    // the floor, gray over the rim.
    let (w, h, rgb) = icam_sim::picture::picture(&grid, &field.h, Some(&left), 19.05, 1);
    assert_eq!((w as usize, h as usize), (grid.nx, grid.ny));
    let px = |x: f64, y: f64| {
        let (i, j) = (((x - grid.x0) / grid.step).round() as usize, ((y - grid.y0) / grid.step).round() as usize);
        let o = ((grid.ny - 1 - j) * grid.nx + i) * 3;
        [rgb[o] as i32, rgb[o + 1] as i32, rgb[o + 2] as i32]
    };
    let [r, g, b] = px(20.0, 30.0);
    assert!(r > g + 40 && r > b + 40, "the floor gouge reads red: {r} {g} {b}");
    let [r, g, b] = px(75.0, 50.0);
    assert!(b > r + 40 && b > g, "the uncut corner reads blue: {r} {g} {b}");
    let [r, g, b] = px(30.0, 8.0);
    assert!(g > r + 40 && g > b + 40, "the floor reads green: {r} {g} {b}");
    let [r, g, b] = px(-35.0, -35.0);
    assert!(r == g && g == b, "the rim reads gray: {r} {g} {b}");
}
