//! A 3-axis program as sweeps: the tool axis held along +z, lines and rapids straight,
//! G2 / G3 as arcs or helices about +z, canned drills as plunges from the R plane. Other
//! flavors of program (rotary words for a particular machine, tool-axis vectors, anything
//! else) only need to produce [`Motion`]s.

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use icam_post::{Move, MoveKind, ToolDef};

use crate::geom::{Pose, Rot, V3};
use crate::motion::{Helix, Linear, Motion};
use crate::sweep::Sweep;
use crate::tool::ToolSolid;

/// The sweeps of `moves`, each tool `length` long (its stick-out): one for every move that
/// `cuts`, in order.
pub fn sweeps(moves: &[Move], tools: &BTreeMap<i32, ToolDef>, length: f64, skip_above: f64) -> Vec<Sweep> {
    let solids: BTreeMap<i32, Arc<ToolSolid>> = tools.iter().map(|(&n, d)| (n, Arc::new(ToolSolid::from_def(d, length)))).collect();
    moves.iter().filter(|mv| cuts(mv, tools, skip_above)).map(|mv| Sweep::new(solids[&mv.tool].clone(), motion(mv))).collect()
}

/// Whether `mv` becomes a sweep: moves whose tool is not in `tools` cut nothing, and moves
/// whose lowest point is above `skip_above` are left out.
pub fn cuts(mv: &Move, tools: &BTreeMap<i32, ToolDef>, skip_above: f64) -> bool {
    tools.contains_key(&mv.tool) && mv.from.z.min(mv.to.z) <= skip_above
}

/// The tool frame's motion over `mv`.
pub fn motion(mv: &Move) -> Arc<dyn Motion> {
    match mv.kind {
        MoveKind::Rapid | MoveKind::Feed => Arc::new(Linear { from: Pose::translation(mv.from), to: Pose::translation(mv.to) }),
        MoveKind::Drill => {
            let (top, bottom) = (mv.from.z.max(mv.to.z), mv.from.z.min(mv.to.z));
            Arc::new(Linear { from: Pose::translation(V3::new(mv.from.x, mv.from.y, top)), to: Pose::translation(V3::new(mv.from.x, mv.from.y, bottom)) })
        }
        MoveKind::ArcCw | MoveKind::ArcCcw => {
            let c = V3::new(mv.center.x, mv.center.y, mv.from.z);
            let t0 = (mv.from.y - c.y).atan2(mv.from.x - c.x);
            let t1 = (mv.to.y - c.y).atan2(mv.to.x - c.x);
            let full = (mv.to.x - mv.from.x).hypot(mv.to.y - mv.from.y) < 1e-6;
            let ccw = mv.kind == MoveKind::ArcCcw;
            let sweep = match (full, ccw) {
                (true, true) => TAU,
                (true, false) => -TAU,
                (false, true) => (t1 - t0).rem_euclid(TAU),
                (false, false) => -(t0 - t1).rem_euclid(TAU),
            };
            Arc::new(Helix { center: c, normal: V3::new(0.0, 0.0, 1.0), start: mv.from, sweep, rise: mv.to.z - mv.from.z, rot: Rot::IDENTITY })
        }
    }
}
