//! Assigning material to tools (icam-assign) on the pockets found from the field.
//!
//! Each pocket op is one layer, at the height the op cuts: the clearance there and just
//! above the sheet bottom from the compiled part, the pocket picked by the op's hint, and
//! the stock every earlier op of the roughing program leaves. Every point goes to a tool three ways (rest roughing, the cheapest tool, and
//! the cheapest with tool changes paid for), all timed by the same cost model.

use std::collections::BTreeMap;
use std::path::Path;

use icam_assign::assign::{self, consolidate_nest};
use icam_assign::layer::top_before;
use icam_assign::vendor::datron;
use icam_assign::{tools, CostTable, Grid, Layer, LayerInput, ModelOptions, Penalties, Plan, ToolSpec};
use icam_core::frep::Frep;
use icam_post::ToolDef;
use icam_query::SliceClearance;

use crate::run::{Job, JobOut};

pub struct AssignJob<'a> {
    /// The end mills to assign among (the roughing tool, the 1/2 in, must be one).
    pub sheet: Vec<ToolDef>,
    pub options: ModelOptions,
    /// The vendor cut-data table the wide-cut envelope comes from (none: no envelope).
    pub vendor: Option<&'a Path>,
    /// Raster cell, mm.
    pub cell: f64,
    pub penalties: Penalties,
}

pub struct LayerOut {
    pub op: String,
    pub layer: Layer,
    /// s/mm2 per tool per cell (infinite where the tool does not reach).
    pub cost: Vec<Vec<f64>>,
    pub rest: Plan,
    pub cheap: Plan,
    pub cheap2: Plan,
    /// Every point at its cheapest tool, no penalties (s).
    pub model_floor: f64,
    /// The volume at the best open-area rate of any tool (s).
    pub bound: f64,
}

pub struct AssignOut {
    pub tools: Vec<ToolSpec>,
    /// The cost table per tool (what the assignment reads).
    pub tabs: Vec<CostTable>,
    pub layers: Vec<LayerOut>,
    /// Pocket ops with no material at their floor.
    pub empty: Vec<String>,
    pub notes: Vec<String>,
    /// Nest totals: rest, cheapest, cheapest with tool changes paid for (s).
    pub nest: (f64, f64, f64),
}

pub fn assign_pockets(frep: &Frep, job: &Job, rough: &JobOut, a: &AssignJob) -> Result<AssignOut, String> {
    let rows = a.vendor.map(datron).unwrap_or_default();
    let mut notes = Vec::new();
    if rows.is_empty() && a.options.envelope {
        notes.push(match a.vendor {
            Some(v) => format!("no vendor rows read from {}: no wide-cut envelope", v.display()),
            None => "no vendor table given (--vendor): no wide-cut envelope".to_string(),
        });
    }
    let specs = tools::build(&a.sheet, &a.options, &rows)?;
    let (lo, hi) = rough.stock;
    let tabs: Vec<CostTable> = specs.iter().map(|t| CostTable::new(t, job.accel, 200.0, hi.z - lo.z + 0.5)).collect();
    let tool_map: BTreeMap<i32, ToolDef> = BTreeMap::from([(job.tool.number, job.tool.clone())]);
    let foot = SliceClearance::new(frep, lo.z + 0.01, 1e-3);
    let (mut layers, mut empty) = (Vec::new(), Vec::new());
    for o in &rough.ops {
        let zf = o.spec.z_cut;
        let b = o.spec.region.grow(2.0);
        let g = Grid::new(b.x0, b.y0, b.x1, b.y1, a.cell);
        let clear = SliceClearance::new(frep, zf, 1e-3);
        let top = top_before(&rough.moves, &tool_map, o.range.0, &g, hi.z);
        let inp = LayerInput {
            name: &o.spec.name,
            op_tool: job.tool.number,
            op_d: job.tool.diameter,
            z_floor: zf,
            g,
            cuts: &[o.spec.hint],
            top,
            stock_top: hi.z,
            tools: &specs,
            delta: job.allowance,
        };
        let l = Layer::make(inp, &clear, &foot);
        if l.mat_area() <= 0.0 {
            empty.push(o.spec.name.clone());
        } else {
            layers.push(l);
        }
    }
    let p = &a.penalties;
    let open = tabs.iter().map(|t| t.open_rate).fold(0.0, f64::max);
    let mut costs = Vec::new();
    let (mut rests, mut cheaps) = (Vec::new(), Vec::new());
    for l in &layers {
        let cost = assign::costs(l, &tabs);
        rests.push(assign::rest(l, &cost, &tabs, p));
        cheaps.push(assign::cheapest(l, &cost, &tabs, p));
        costs.push(cost);
    }
    let refs: Vec<&Layer> = layers.iter().collect();
    let cheap2 = consolidate_nest(&refs, &costs, &cheaps, &tabs, p);
    let nest = (
        assign::nest_total(&rests.iter().collect::<Vec<_>>(), p),
        assign::nest_total(&cheaps.iter().collect::<Vec<_>>(), p),
        assign::nest_total(&cheap2.iter().collect::<Vec<_>>(), p),
    );
    let mut outs = Vec::new();
    for (((l, cost), (rest, cheap)), c2) in layers.into_iter().zip(costs).zip(rests.into_iter().zip(cheaps)).zip(cheap2) {
        let mut model_floor = 0.0;
        for k in 0..l.g.n() {
            if l.mat[k] {
                let m = cost.iter().map(|c| c[k]).fold(f64::INFINITY, f64::min);
                if m.is_finite() {
                    model_floor += m * l.cell_area();
                }
            }
        }
        let bound = l.volume() / open;
        outs.push(LayerOut { op: l.name.clone(), layer: l, cost, rest, cheap, cheap2: c2, model_floor, bound });
    }
    Ok(AssignOut { tools: specs, tabs, layers: outs, empty, notes, nest })
}
