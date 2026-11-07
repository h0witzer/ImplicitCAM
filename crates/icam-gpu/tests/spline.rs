//! GPU vs CPU on a face bounded by a line and a native cubic B-spline.

use std::sync::Arc;

use icam_core::bspline::{Nurbs2, Spline2};
use icam_core::geom::{Curve, V2};
use icam_core::part::Part;
use icam_core::query::{drop_ref, slice_ref, NONE};
use icam_core::tool::Tool;
use icam_gpu::{Gpu, GpuPart, Mode};

fn d_shape() -> Part {
    let ctrl = vec![V2::new(10.0, 0.0), V2::new(11.0, 6.0), V2::new(5.0, 9.0), V2::new(-1.0, 6.0), V2::new(0.0, 0.0)];
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let nb = Nurbs2::new(3, ctrl, None, knots).unwrap();
    let top = Curve::Spline(Arc::new(Spline2::new(nb, 0.0, 1.0)));
    let bottom = Curve::Line { a: V2::new(0.0, 0.0), b: V2::new(10.0, 0.0) };
    let mut p = Part::default();
    p.add_face(7, 5.0, vec![vec![bottom, top]], None, 1e-12).unwrap();
    p
}

#[test]
fn spline_face_matches_cpu() {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let part = d_shape();
    let gp = GpuPart::new(&gpu, &part, 10.0, 1.0).unwrap();
    let mut pts = Vec::new();
    for i in 0..60 {
        for j in 0..60 {
            pts.push([-3.0 + 17.0 * i as f64 / 59.0, -3.0 + 15.0 * j as f64 / 59.0, 1.0]);
        }
    }
    for (tool, mode) in [(Tool::ball(4.0, 20.0), Mode::Drop), (Tool::ball(4.0, 20.0), Mode::Slice), (Tool::vee(6.0, 90.0, 20.0), Mode::Drop)] {
        let (res, _) = gp.query(&gpu, &tool, mode, &pts).expect("GPU query");
        let mut worst: f64 = 0.0;
        for (g, p) in res.iter().zip(&pts) {
            let q = V2::new(p[0], p[1]);
            let c = if mode == Mode::Drop { drop_ref(&part, &tool, q) } else { slice_ref(&part, &tool, q, p[2]) };
            if c.face == NONE || g.face == NONE || g.flags != 0 {
                assert_eq!(c.face == NONE, g.face == NONE, "reach at {p:?}");
                continue;
            }
            worst = worst.max((g.value - c.value).abs());
        }
        eprintln!("{} {mode:?}: worst |GPU - CPU| {worst:.2e} mm", tool.name);
        assert!(worst < 1e-3, "{} {mode:?}: worst {worst}", tool.name);
    }
}
