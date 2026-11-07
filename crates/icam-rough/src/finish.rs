//! The finishing-allowance study (icam-finish) on the pockets found from the field.
//!
//! Each pocket op whose pocket ends on its slab's floor is one study: the clearance at
//! that floor from the compiled part, the finisher's wall loops around the region its
//! center reaches from the op's hint, the rougher's component seeded at the same hint,
//! and the cuts the roughing program makes before the op (its stock and the wall heights
//! it leaves). The rougher is the roughing job's own tool.

use std::collections::BTreeMap;
use std::time::Instant;

use icam_core::frep::Frep;
use icam_finish::geom::top_before;
use icam_finish::pocket::{band_for, grid_pitch};
use icam_finish::{study, Pocket, PocketInput, PocketResult, Settings, Walls};
use icam_post::{Box2, ToolDef};
use icam_query::SliceClearance;
use icam_stock::SweptStock;

use crate::ops;
use crate::run::{Job, JobOut};

/// The clearance grid's node budget: past it the pitch grows from 0.1 mm (the study keeps
/// several grids of this size).
const MAX_NODES: f64 = 25e6;

pub struct FinishOut {
    pub op: String,
    pub floor: f64,
    pub pocket: Option<Pocket>,
    pub result: Option<PocketResult>,
    pub skipped: Option<String>,
    pub ms: f64,
}

/// Study every pocket the roughing job cut to a floor, finished by `finish_tool`.
pub fn study_pockets(frep: &Frep, job: &Job, rough: &JobOut, finish_tool: &ToolDef, s: &Settings) -> Result<Vec<FinishOut>, String> {
    let rt = &job.tool;
    if (rt.diameter - 12.7).abs() > 0.01 {
        return Err(format!("the finishing study models the 1/2 in rougher (the cut model has no other); rough with --tool 12.7, not {:.3}", rt.diameter));
    }
    if (finish_tool.diameter - 6.35).abs() > 0.01 && (finish_tool.diameter - 12.7).abs() > 0.01 {
        return Err(format!("the finisher's cut model exists for the 1/4 in and 1/2 in end mills only, not {:.3} mm", finish_tool.diameter));
    }
    let tools: BTreeMap<i32, ToolDef> = BTreeMap::from([(rt.number, rt.clone())]);
    let (lo, hi) = rough.stock;
    let sheet = Box2::new(lo.x, lo.y, hi.x, hi.y);
    let (rr, rf) = (rt.diameter / 2.0, finish_tool.diameter / 2.0);
    let band = band_for(rr, rf, s.d_max);
    let mut outs = Vec::new();
    for o in &rough.ops {
        let Some(z) = o.spec.floor else { continue };
        let clock = Instant::now();
        let mut out = FinishOut { op: o.spec.name.clone(), floor: z, pocket: None, result: None, skipped: None, ms: 0.0 };
        let region = o.spec.region.grow(band + 1.0);
        let grid = grid_pitch(region, MAX_NODES);
        let made = if !ops::ends_on_floor(frep, &o.spec, rr, job.allowance) {
            Err("the pocket goes on below this slab; its walls are finished at its own floor".into())
        } else if o.path.is_none() {
            Err(format!("not roughed ({})", o.skipped.as_deref().unwrap_or("no path")))
        } else {
            let clear = SliceClearance::new(frep, z, 1e-3);
            let i0 = o.range.0;
            let mut prior = SweptStock::new(region.grow(rr + 2.0), sheet);
            prior.add_program(&rough.moves, i0, &tools, z);
            let top = |x: f64, y: f64| top_before(&rough.moves, &tools, i0, x, y, hi.z);
            let finish_name = format!("FINISH {}", o.spec.name);
            let inp = PocketInput {
                rough: &o.spec.name,
                finish: &finish_name,
                rough_tool: rt,
                finish_tool,
                z,
                region,
                prior,
                rough_centers: &[o.spec.hint],
                top_before: &top,
                ds: 0.2,
                grid,
                d_min: s.d_min,
                d_max: s.d_max,
            };
            Pocket::make(inp, &clear, Walls::Reach(&[o.spec.hint])).and_then(|mut pk| {
                if grid > 0.1 + 1e-9 {
                    pk.notes.push(format!("clearance grid {grid:.3} mm (a large pocket; 0.1 mm otherwise)"));
                }
                study::run(&pk, s).map(|r| (pk, r))
            })
        };
        match made {
            Ok((pk, r)) => {
                out.pocket = Some(pk);
                out.result = Some(r);
            }
            Err(e) => out.skipped = Some(e),
        }
        out.ms = clock.elapsed().as_secs_f64() * 1000.0;
        outs.push(out);
    }
    Ok(outs)
}
