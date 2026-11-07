//! icam-sim: look at the stock a program leaves, and check the fields behind it.
//!
//!   icam-sim play <program.nc> --set <tools.set> --part <step> [--place t[f],x,y,z] [options]
//!     a window on the stock over the part, cut move by move at runtime (see play.rs); with
//!     --bake, colored by what the program left against the part (heat.rs);
//!     --box x0,y0,z0,x1,y1,z1 instead of --part
//!     --length <mm>       tool stick-out (default 40)
//!     --finest <mm>       cut resolution (default 0.4)
//!     --moves <n>         only the first n moves
//!     --shot <file.png>   once every move is in, time the frames, write one and close
//!     --look x,y,dist,yaw,pitch  the shot's camera (mm and degrees; default: the fit)
//!     --cpu               read the fields on the CPU only (the GPU's are the default)
//!     --bake <mm>         draw the whole program's stock from a cache of its field, in levels
//!                         around the view, cells no finer than this
//!     --check             check the cache against the exact field, and the drawing against the cache;
//!                         with --at too, re-bakes to the playhead against bakes from nothing
//!     --at <n>            with --bake: the playhead, how many moves have cut the stock (default all;
//!                         in the window the arrows, Page Up and Down, Home and End move it,
//!                         Space plays from it)
//!     --scrub <n>         with --bake and --shot: step the playhead across the whole program in n
//!                         steps, timing each, then write the picture halfway through
//!     --play <speed>      with --bake: play from the playhead at this many times the machine's
//!                         speed (1: its own); with --shot too, for ten seconds, measuring how
//!                         the stock keeps up, then write the picture
//!     --accel <mm/s2>     the machine's acceleration, for its clock (default 145, fitted to the
//!                         router's run times; see play.rs, `ACCEL`)
//!     --blocks <n>        the most program blocks its controller runs a second (default: no limit)
//!     --pick <n>          with --bake, --part and --shot: pick region n of the panel's list first
//!     --view part         with --bake and --part: open on the part's view (the panel's "Design part")
//!
//!   icam-sim part <file.step> [--finest mm] [--shot file.png] [--look ...]
//!     any STEP solid alone, by its exact signed distance (solid3::Exact), in the part's
//!     view: no program, the stock its padded box; cells no finer than --finest (0.25)
//!     --moment <n>        with --bake, --part and --shot: the moment gouge n of the list was cut first
//!     --zoom <k>          with --play and --shot: two seconds in, the camera k times closer; reports
//!                         the frames' and the stock's longest gaps after it
//!     --pan <mm/s>        with --play and --shot: from two seconds in, the camera's target moves
//!                         along x at this speed; reports the same after it
//!     --no-scroll         place levels the view moves on from anew, never scrolled (to compare)
//!
//!   icam-sim part-bench <file.step> [--gpu]
//!     the part's brick cache built and timed level by level; with --gpu, the exact evaluator's
//!     kernel checked against the CPU reference (partgpu.rs)
//!
//!   icam-sim clock <program.nc> [--accel <mm/s2>] [--blocks <n>]
//!     how long the program runs on the machine's clock (see play.rs, `machine_clock`), and at
//!     its programmed feeds alone
//!
//!   icam-sim compare <program.nc> --set <tools.set> --part <step> [--place t[f],x,y,z] [options]
//!     the stock the program leaves against the part on the part's own surface
//!     (compare.rs): areas on size, with stock left and gouged, and every gouge with its
//!     depth, area, volume and the move that cut it
//!     --length <mm>       tool stick-out (default 40)
//!     --at <n>            after the first n moves (default all)
//!     --step <mm>         points on the surface this far apart (default 0.5)
//!     --region x0,y0,x1,y1  only the part's surface over this rectangle (mm, the part's frame)
//!     --generic           find the surface's points the general way even for a prismatic part
//!     --record <file>     write the analysis record (JSON, a few kilobytes)
//!     --check             read every thousandth point again in double precision on the CPU, and with
//!                         every move, the faces up against the 64-bit top-down check
//!
//!   icam-sim field <program.nc> --set <tools.set> --part <step> [options]
//!   icam-sim field <program.nc> --set <tools.set> --box x0,y0,z0,x1,y1,z1 [options]
//!     the stock field on the GPU over the part (its box, widened by the largest tool
//!     on the sheet, so cuts around its outline show) or over the box, timed, and checked
//!     against the exact 64-bit cut height at sampled nodes. The program runs in the
//!     part's frame: a part is never fitted to a program.
//!     --step <mm>     node spacing (default 0.25)
//!     --check <n>     nodes checked against the exact height (default 200000)
//!     --cpu           also build the whole field on the CPU, for the timing
//!     --part <step>   the part: its top over every node from the compiled F-rep, and
//!                     what the program left there (material, on size, gouge), checked
//!                     against the 64-bit reference at the same nodes plus every node
//!                     where the part's top steps
//!     --place <t[f],x,y,z>  the part placed so (quarter turns, f upside down, shift), as
//!                     it was placed for planning
//!     --box <...>     the stock box instead of the part's (z1 is the stock top)
//!     --picture <file.png>  a still of the stock's relief, colored by what the program
//!                     left when --part is given (see picture.rs)
//!     --per <n>       nodes to a pixel of the picture (default: 1 mm a pixel)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use icam_core::frep::Frep;
use icam_post::{ToolDef, V3};
use icam_sim::part::{self, Class, PartField};
use icam_sim::{GpuStock, Grid, StockField};
use icam_stock::ImplicitStock;
use rayon::prelude::*;

mod heat;
mod partcache;
mod partgpu;
mod play;
mod regions;
mod ui;

/// The gate: the GPU field agrees with the exact height to this, mm.
const GATE_MM: f64 = 0.001;
/// On a vertical wall, within this far of the node, mm.
const WALL_MM: f64 = 0.002;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match go(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("icam-sim: {e}");
            ExitCode::from(1)
        }
    }
}

fn go(args: &[String]) -> Result<ExitCode, String> {
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let nums = |name: &str, n: usize| -> Result<Option<Vec<f64>>, String> {
        opt(name)
            .map(|v| {
                let f: Vec<f64> = v.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("{name}: not numbers: {v}"))?;
                if f.len() == n { Ok(f) } else { Err(format!("{name}: expected {n} numbers, got {v}")) }
            })
            .transpose()
    };
    if args.first().map(String::as_str) == Some("play") && args.len() >= 2 {
        let part = match opt("--part") {
            Some(f) => Some((PathBuf::from(f), parse_place(opt("--place").as_deref())?)),
            None => None,
        };
        let bbox = nums("--box", 6)?.map(|b| [b[0], b[1], b[2], b[3], b[4], b[5]]);
        if part.is_none() && bbox.is_none() {
            return Err("play: --part <step> or --box x0,y0,z0,x1,y1,z1 is required".into());
        }
        play::run(play::Args {
            program: PathBuf::from(&args[1]),
            set: PathBuf::from(opt("--set").ok_or("--set <tools.set> is required")?),
            part,
            bbox,
            length: nums("--length", 1)?.map_or(40.0, |v| v[0]),
            finest: nums("--finest", 1)?.map_or(0.4, |v| v[0]),
            limit: opt("--moves").map(|v| v.parse::<usize>().map_err(|_| format!("--moves: not a count: {v}"))).transpose()?,
            shot: opt("--shot").map(PathBuf::from),
            look: nums("--look", 5)?.map(|v| [v[0], v[1], v[2], v[3], v[4]]),
            cpu: args.iter().any(|a| a == "--cpu"),
            bake: nums("--bake", 1)?.map(|v| v[0]),
            check: args.iter().any(|a| a == "--check"),
            at: opt("--at").map(|v| v.parse::<usize>().map_err(|_| format!("--at: not a count: {v}"))).transpose()?,
            scrub: opt("--scrub").map(|v| v.parse::<usize>().map_err(|_| format!("--scrub: not a count: {v}"))).transpose()?,
            play: nums("--play", 1)?.map(|v| v[0]),
            machine: machine(&nums)?,
            pick: opt("--pick").map(|v| v.parse::<usize>().map_err(|_| format!("--pick: not a count: {v}"))).transpose()?,
            part_view: opt("--view").as_deref() == Some("part"),
            moment: opt("--moment").map(|v| v.parse::<usize>().map_err(|_| format!("--moment: not a count: {v}"))).transpose()?,
            zoom: nums("--zoom", 1)?.map(|v| v[0]),
            pan: nums("--pan", 1)?.map(|v| v[0]),
            no_scroll: args.iter().any(|a| a == "--no-scroll"),
            grab: opt("--grab").map(PathBuf::from),
            solid: None,
        })?;
        return Ok(ExitCode::SUCCESS);
    }
    if args.first().map(String::as_str) == Some("part-bench") && args.len() >= 2 {
        return part_bench(&args[1], &args[2..]);
    }
    if args.first().map(String::as_str) == Some("part") && args.len() >= 2 {
        // Any STEP solid alone, by its exact field, in the part's view.
        play::run(play::Args {
            program: PathBuf::new(),
            set: PathBuf::new(),
            part: None,
            bbox: None,
            length: 40.0,
            finest: 0.4,
            limit: None,
            shot: opt("--shot").map(PathBuf::from),
            look: nums("--look", 5)?.map(|v| [v[0], v[1], v[2], v[3], v[4]]),
            cpu: false,
            bake: Some(nums("--finest", 1)?.map_or(0.25, |v| v[0])),
            check: false,
            at: None,
            scrub: None,
            play: None,
            machine: machine(&nums)?,
            pick: None,
            part_view: true,
            moment: None,
            zoom: nums("--zoom", 1)?.map(|v| v[0]),
            pan: nums("--pan", 1)?.map(|v| v[0]),
            no_scroll: args.iter().any(|a| a == "--no-scroll"),
            grab: None,
            solid: Some(PathBuf::from(&args[1])),
        })?;
        return Ok(ExitCode::SUCCESS);
    }
    if args.first().map(String::as_str) == Some("clock") && args.len() >= 2 {
        let mut warnings = Vec::new();
        let moves = icam_post::read(Path::new(&args[1]), &mut warnings).map_err(|e| format!("{}: {e}", args[1]))?;
        let m = machine(&nums)?;
        let (t, programmed) = play::program_time(&moves, m);
        let hms = |t: f64| format!("{}:{:02}:{:04.1}", (t / 3600.0) as u64, ((t / 60.0) as u64) % 60, t % 60.0);
        println!(
            "clock: {} moves, {} on the machine's clock (acceleration {} mm/s2, {}), {} at the programmed feeds alone",
            moves.len(),
            hms(t),
            m.accel,
            m.blocks.map_or("no block limit".to_string(), |b| format!("{b} blocks a second")),
            hms(programmed)
        );
        return Ok(ExitCode::SUCCESS);
    }
    if args.first().map(String::as_str) == Some("compare") && args.len() >= 2 {
        return compare(args, &opt, &nums);
    }
    if args.first().map(String::as_str) != Some("field") || args.len() < 2 {
        return Err("usage: icam-sim (field | play | compare | clock) <program.nc> --set <tools.set> (--part <step> [--place t[f],x,y,z] | --box x0,y0,z0,x1,y1,z1) [--step mm] [--check n] [--cpu] [--picture file.png [--per n]]".into());
    }
    let nc = Path::new(&args[1]);
    let set = opt("--set").ok_or("--set <tools.set> is required")?;
    let step = nums("--step", 1)?.map_or(0.25, |v| v[0]);
    let check = opt("--check").map_or(Ok(200_000), |v| v.parse::<usize>().map_err(|_| format!("--check: not a count: {v}")))?;

    let clock = Instant::now();
    let mut warnings = Vec::new();
    let moves = icam_post::read(nc, &mut warnings).map_err(|e| format!("{}: {e}", nc.display()))?;
    let tools: BTreeMap<i32, ToolDef> = icam_post::read_tool_table(Path::new(&set), 25.4).map_err(|e| format!("{set}: {e}"))?;
    for w in warnings.iter().take(4) {
        println!("program: {w}");
    }
    let read = clock.elapsed().as_secs_f64();
    let part = match opt("--part") {
        Some(file) => Some(read_part(&file, opt("--place").as_deref())?),
        None => None,
    };
    let clock = Instant::now();
    let b = match (nums("--box", 6)?, &part) {
        (Some(b), _) => b,
        (None, Some((_, lo, hi))) => {
            let m = tools.values().map(|t| t.diameter).fold(0.0, f64::max);
            vec![lo[0] - m, lo[1] - m, lo[2], hi[0] + m, hi[1] + m, hi[2]]
        }
        (None, None) => return Err("--part <step> or --box x0,y0,z0,x1,y1,z1 is required".into()),
    };
    let stock = ImplicitStock::new(V3::new(b[0], b[1], b[2]), V3::new(b[3], b[4], b[5]), &moves, &tools, ImplicitStock::CELL_MM);
    println!("program: {} moves, {} sweeps ({:.2} s to read and digest)", moves.len(), stock.sweep_count(), read + clock.elapsed().as_secs_f64());

    let grid = Grid::over(b[0], b[1], b[3], b[4], step);
    let gpu = icam_gpu::Gpu::new()?;
    let g = GpuStock::new(&gpu.device, &stock);
    let (field, t) = g.field(&gpu.device, &gpu.queue, grid)?;
    println!(
        "GPU {}: {} x {} nodes ({:.1} M) at {step} mm: upload {:.2} s, compute {:.2} s in {} passes, read back {:.2} s",
        gpu.adapter,
        grid.nx,
        grid.ny,
        (grid.nx * grid.ny) as f64 / 1e6,
        t.upload,
        t.compute,
        t.passes,
        t.readback
    );

    let top = b[5] as f32;
    let cut = field.h.iter().filter(|&&v| v < top - 0.001).count();
    let low = field.h.iter().copied().fold(f32::INFINITY, f32::min);
    println!("field: {:.1}% of the nodes cut below the stock top, lowest {low:.3} mm", 100.0 * cut as f64 / field.h.len() as f64);
    // The gate, at nodes spread over the grid (a fixed sequence, so runs compare).
    let n = grid.nx * grid.ny;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let sample: Vec<usize> = (0..check.min(n))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        })
        .collect();
    let mut errs: Vec<(f64, usize)> = sample.iter().map(|&k| ((field.h[k] as f64 - stock.cut_height(grid.x(k % grid.nx), grid.y(k / grid.nx))).abs(), k)).collect();
    errs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let q = |f: f64| errs.get(((errs.len() as f64 - 1.0) * f) as usize).map_or(0.0, |e| e.0);
    let over = errs.iter().filter(|e| e.0 > GATE_MM).count();
    println!("check: {} nodes against the exact height: p50 {:.5} mm, p99 {:.5}, p99.9 {:.5}, max {:.5}; {over} over {GATE_MM} mm", errs.len(), q(0.5), q(0.99), q(0.999), q(1.0));
    for &(e, k) in errs.iter().rev().take(5).filter(|e| e.0 > GATE_MM) {
        let (i, j) = (k % grid.nx, k / grid.nx);
        println!("  {e:.4} mm at ({:.3}, {:.3}): GPU {:.4}, exact {:.4}", grid.x(i), grid.y(j), field.h[k], stock.cut_height(grid.x(i), grid.y(j)));
    }
    let mut failed = over;
    if args.iter().any(|a| a == "--cpu") {
        let clock = Instant::now();
        let _cpu = StockField::cpu(&stock, grid);
        let secs = clock.elapsed().as_secs_f64();
        let (differ, off) = icam_sim::check(&stock, &field, GATE_MM, WALL_MM);
        println!("CPU: the whole field in {secs:.2} s; {differ} nodes differ from the GPU by more than {GATE_MM} mm, {} of them not on a wall (the GPU's height is not the exact one within {WALL_MM} mm)", off.len());
        for &(d, k) in off.iter().take(8) {
            let (i, j) = (k % grid.nx, k / grid.nx);
            println!("  {d:.4} mm at ({:.3}, {:.3}): GPU {:.4}, exact {:.4}", grid.x(i), grid.y(j), field.h[k], stock.cut_height(grid.x(i), grid.y(j)));
        }
        failed += off.len();
    }
    let left = match &part {
        Some((frep, _, _)) => {
            let (f, left) = part_gate(frep, &stock, &field, &sample)?;
            failed += f;
            Some(left)
        }
        None => None,
    };
    if let Some(png) = opt("--picture") {
        let per = opt("--per").map_or(Ok((1.0 / step).round().max(1.0) as usize), |v| v.parse::<usize>().map_err(|_| format!("--per: not a count: {v}")))?;
        let clock = Instant::now();
        let (w, h, rgb) = icam_sim::picture::picture(&grid, &field.h, left.as_deref(), top, per);
        let out = std::fs::File::create(&png).map_err(|e| format!("{png}: {e}"))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(out), w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().and_then(|mut wr| wr.write_image_data(&rgb)).map_err(|e| format!("{png}: {e}"))?;
        let bytes = std::fs::metadata(&png).map_or(0, |m| m.len());
        println!("picture: {w} x {h} px ({:.2} mm a pixel), {:.1} MB, in {:.2} s: {png}", per as f64 * step, bytes as f64 / 1e6, clock.elapsed().as_secs_f64());
        if left.is_some() {
            println!("  green on size, blue material left (light: a thin skin, deep: full stock), red gouged (orange: a skim, deep red: 2.5 mm), gray no part");
        }
    }
    Ok(if failed == 0 { ExitCode::SUCCESS } else { ExitCode::from(2) })
}

/// "3f,1073.15,655.288,25.4": quarter turns (f: upside down first), then the shift.
fn parse_place(v: Option<&str>) -> Result<icam_core::prismatic::Place, String> {
    let Some(v) = v else {
        return Ok(icam_core::prismatic::Place::default());
    };
    let f: Vec<&str> = v.split(',').map(str::trim).collect();
    let bad = || format!("--place: expected turns[f],x,y,z, got {v}");
    if f.len() != 4 {
        return Err(bad());
    }
    let turns: u8 = f[0].trim_end_matches('f').parse().map_err(|_| bad())?;
    let n = |s: &str| s.parse::<f64>().map_err(|_| bad());
    Ok(icam_core::prismatic::Place::new(turns, f[0].ends_with('f'), n(f[1])?, n(f[2])?, n(f[3])?))
}

/// The part read and compiled where it was placed for planning: its F-rep and its box.
type Opt<'a> = dyn Fn(&str) -> Option<String> + 'a;
type Nums<'a> = dyn Fn(&str, usize) -> Result<Option<Vec<f64>>, String> + 'a;

/// icam-sim compare: see the usage at the top and compare.rs.
fn compare(args: &[String], opt: &Opt, nums: &Nums) -> Result<ExitCode, String> {
    use icam_sim::compare::{analyze, stock_f64};
    let nc = Path::new(&args[1]);
    let set = opt("--set").ok_or("--set <tools.set> is required")?;
    let file = opt("--part").ok_or("compare: --part <step> is required")?;
    let length = nums("--length", 1)?.map_or(40.0, |v| v[0]);
    let step = nums("--step", 1)?.map_or(0.5, |v| v[0]);
    let check = args.iter().any(|a| a == "--check");
    let clock = Instant::now();
    let mut warnings = Vec::new();
    let moves = icam_post::read(nc, &mut warnings).map_err(|e| format!("{}: {e}", nc.display()))?;
    let tools: BTreeMap<i32, ToolDef> = icam_post::read_tool_table(Path::new(&set), 25.4).map_err(|e| format!("{set}: {e}"))?;
    let (frep, lo, hi) = read_part(&file, opt("--place").as_deref())?;
    let m = tools.values().map(|t| t.diameter).fold(0.0, f64::max);
    let b = icam_sweep::Aabb::new(icam_sweep::V3::new(lo[0] - m, lo[1] - m, lo[2]), icam_sweep::V3::new(hi[0] + m, hi[1] + m, hi[2]));
    let sweeps = icam_sweep::program::sweeps(&moves, &tools, length, b.max.z);
    let mut solids: Vec<std::sync::Arc<icam_sweep::ToolSolid>> = Vec::new();
    for s in &sweeps {
        if !solids.iter().any(|t| std::sync::Arc::ptr_eq(t, &s.tool)) {
            solids.push(s.tool.clone());
        }
    }
    let fields = icam_sweep::gpu::Fields::new(&solids, &[])?;
    let mut stock = icam_sweep::Stock::new(b);
    for s in sweeps {
        stock.apply(s);
    }
    let upto = opt("--at").map(|v| v.parse::<usize>().map_err(|_| format!("--at: not a count: {v}"))).transpose()?.unwrap_or(stock.len()).min(stock.len());
    println!("program: {} moves, {} placed; fields compiled ({:.1} s)", moves.len(), stock.len(), clock.elapsed().as_secs_f64());
    // A little past the part's box, off its faces, so a face on the box is crossed by lines.
    let pad = 0.61 * step;
    let (mut alo, mut ahi): ([f64; 3], [f64; 3]) = (std::array::from_fn(|a| lo[a] - pad), std::array::from_fn(|a| hi[a] + pad));
    if let Some(r) = nums("--region", 4)? {
        (alo[0], alo[1], ahi[0], ahi[1]) = (alo[0].max(r[0]), alo[1].max(r[1]), ahi[0].min(r[2]), ahi[1].min(r[3]));
    }
    let say = |what: &str| println!("compare: {what} ({:.1} s)", clock.elapsed().as_secs_f64());
    let generic = args.iter().any(|a| a == "--generic");
    // The checks see each batch of points as it goes: every thousandth point is kept to be
    // read again in double precision, and with every move the faces up are held against the
    // 64-bit top-down check.
    let top_down = (check && upto == stock.len()).then(|| (ImplicitStock::new(V3::new(b.min.x, b.min.y, b.min.z), V3::new(b.max.x, b.max.y, b.max.z), &moves, &tools, ImplicitStock::CELL_MM), part::columns(&frep)));
    let class = |v: f64| if v > part::TOL { 2 } else if v >= -part::TOL { 1 } else { 0 };
    let edge = |v: f64| (v.abs() - part::TOL).abs() <= 1e-3;
    let band = |b: icam_sim::compare::Band| match b {
        icam_sim::compare::Band::Left => 0,
        icam_sim::compare::Band::OnSize => 1,
        icam_sim::compare::Band::Gouged => 2,
    };
    let mut seen = 0usize;
    let mut picked: Vec<([f64; 3], f64)> = Vec::new();
    let (mut up_points, mut area, mut differ, mut differ_area) = (0usize, [[0f64; 3]; 2], 0usize, 0f64);
    let mut examples: Vec<([f64; 3], f64, f64)> = Vec::new();
    let mut t_top_down = 0.0;
    let mut visit = |samples: &[icam_sim::compare::Sample], values: &[f64], bands: &[icam_sim::compare::Band]| {
        if !check {
            return;
        }
        for (k, (s, &v)) in samples.iter().zip(values).enumerate() {
            if (seen + k).is_multiple_of(1000) {
                picked.push((s.p, v));
            }
        }
        seen += samples.len();
        if let Some((stock64, columns)) = &top_down {
            let clock = Instant::now();
            let up: Vec<usize> = (0..samples.len()).filter(|&i| samples[i].n[2] > 0.999).collect();
            // Their "left" is material above the top (positive): the field reads it negative.
            let theirs: Vec<f64> = up.par_iter().map(|&i| -part::left(&frep, columns, samples[i].p[0], samples[i].p[1], samples[i].p[2], stock64.cut_height(samples[i].p[0], samples[i].p[1]))).collect();
            up_points += up.len();
            for (&i, &t) in up.iter().zip(&theirs) {
                let (ours, top_down) = (band(bands[i]), class(t));
                area[0][ours] += samples[i].w as f64;
                area[1][top_down] += samples[i].w as f64;
                if ours != top_down && !edge(t) {
                    differ += 1;
                    differ_area += samples[i].w as f64;
                    if examples.len() < 6 {
                        examples.push((samples[i].p, values[i], t));
                    }
                }
            }
            t_top_down += clock.elapsed().as_secs_f64();
        }
    };
    let analysis = analyze(&frep, alo, ahi, &stock, &fields, upto, step, generic, &say, &mut visit)?;
    println!("compare: after move {upto} of {}: {}", stock.len(), analysis.report);
    for g in analysis.gouges.iter().take(12) {
        println!(
            "  gouge at ({:.3}, {:.3}, {:.3}): {:.4} mm deep along the part's normal (the field {:.4}), {:.2} mm2, at least {:.3} mm3, {} points, cut by move {}",
            g.at[0],
            g.at[1],
            g.at[2],
            g.depth,
            g.field,
            g.area,
            g.volume,
            g.points,
            g.by.map_or("(the box)".to_string(), |k| k.to_string())
        );
    }
    if check && !picked.is_empty() {
        let clock = Instant::now();
        let exact: Vec<f64> = picked.par_iter().map(|&(p, _)| stock_f64(&stock, icam_sweep::V3::new(p[0], p[1], p[2]), upto, 5.0, 1e-6).0).collect();
        let mut gaps: Vec<f64> = picked.iter().zip(&exact).map(|(&(_, v), &e)| (v - e).abs()).collect();
        gaps.sort_by(f64::total_cmp);
        let q = |f: f64| gaps[((gaps.len() - 1) as f64 * f) as usize];
        let apart = picked.iter().zip(&exact).filter(|&(&(_, v), &e)| class(v) != class(e) && !edge(v) && !edge(e)).count();
        println!(
            "check: {} points (every thousandth) read again in double precision in {:.1} s: the GPU's field within p50 {:.6} mm, p99 {:.6}, max {:.6}; {apart} in another band (away from the bands' edges)",
            picked.len(),
            clock.elapsed().as_secs_f64(),
            q(0.5),
            q(0.99),
            q(1.0)
        );
    }
    if top_down.is_some() {
        println!(
            "check: {up_points} points on faces up against the 64-bit top-down check in {t_top_down:.1} s: on size {:.4} m2 here, {:.4} m2 there; stock left {:.4} m2, {:.4}; gouged {:.1} mm2, {:.1}; {differ} points ({differ_area:.1} mm2) in another band",
            area[0][1] / 1e6,
            area[1][1] / 1e6,
            area[0][0] / 1e6,
            area[1][0] / 1e6,
            area[0][2],
            area[1][2]
        );
        for (p, ours, theirs) in &examples {
            println!("  at ({:.3}, {:.3}, {:.3}): the field {ours:.4} mm, the top-down check {theirs:.4}", p[0], p[1], p[2]);
        }
    }
    if let Some(out) = opt("--record") {
        let place = opt("--place").unwrap_or_default();
        let json = analysis.json(&[("part", Path::new(&file)), ("program", nc), ("tools", Path::new(&set))], &[("place", format!("{place:?}")), ("stick_out_mm", format!("{length}"))])?;
        std::fs::write(&out, &json).map_err(|e| format!("{out}: {e}"))?;
        let part_bytes = std::fs::metadata(&file).map_or(0, |m| m.len());
        println!("record: {out}, {} bytes ({:.4} % of the part file)", json.len(), 100.0 * json.len() as f64 / part_bytes.max(1) as f64);
    }
    Ok(ExitCode::SUCCESS)
}

/// The machine the clock runs (--accel, --blocks).
fn machine(nums: &Nums) -> Result<play::Machine, String> {
    Ok(play::Machine { accel: nums("--accel", 1)?.map_or(play::ACCEL, |v| v[0]), blocks: nums("--blocks", 1)?.map(|v| v[0]) })
}

fn read_part(file: &str, place: Option<&str>) -> Result<(Frep, [f64; 3], [f64; 3]), String> {
    let clock = Instant::now();
    let read = icam_core::prismatic::ReadPart::read(Path::new(file))?;
    let place = parse_place(place)?;
    let (frep, _, _) = read.compile(&place)?;
    let (lo, hi) = read.bounds(&place);
    println!(
        "part: {} columns, box ({:.3}, {:.3}, {:.3}) .. ({:.3}, {:.3}, {:.3}), read and compiled in {:.2} s",
        part::columns(&frep).len(),
        lo[0],
        lo[1],
        lo[2],
        hi[0],
        hi[1],
        hi[2],
        clock.elapsed().as_secs_f64()
    );
    Ok((frep, lo, hi))
}

/// The part on the grid and what the program left, timed, then held to the 64-bit
/// reference (exact stock height, every column's section read at the node) at the
/// sampled nodes and at every node where the part's top steps. Returns the failures
/// and what was left at every node.
fn part_gate(frep: &Frep, stock: &ImplicitStock, field: &StockField, sample: &[usize]) -> Result<(usize, Vec<f32>), String> {
    let grid = field.grid;
    let clock = Instant::now();
    let pf = PartField::new(frep, grid);
    let built = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let left = part::left_field(frep, &pf, field);
    let classed = clock.elapsed().as_secs_f64();
    println!(
        "part: its top over the grid in {built:.2} s ({:.1} M section reads, {:.3} per node); classed in {classed:.2} s",
        pf.reads as f64 / 1e6,
        pf.reads as f64 / (grid.nx * grid.ny) as f64
    );
    let mut count = [0usize; 4];
    let mut worst: Vec<(f32, usize)> = Vec::new();
    for (k, &v) in left.iter().enumerate() {
        let c = Class::of(v as f64);
        count[c as usize] += 1;
        if c == Class::Gouge {
            worst.push((v, k));
        }
    }
    let area = |c: Class| count[c as usize] as f64 * grid.step * grid.step;
    println!(
        "left: waste {:.3} m2, material over the part {:.3} m2, on size {:.3} m2, gouged {:.0} mm2",
        area(Class::Waste) / 1e6,
        area(Class::Left) / 1e6,
        area(Class::OnSize) / 1e6,
        area(Class::Gouge)
    );
    // The gouged area by the part's top there, in depth bands.
    let mut by_top: BTreeMap<i64, [usize; 3]> = BTreeMap::new();
    for &(v, k) in &worst {
        let band = if -v < 0.1 { 0 } else if -v < 0.5 { 1 } else { 2 };
        by_top.entry((pf.top[k] as f64 * 1000.0).round() as i64).or_default()[band] += 1;
    }
    for (z, n) in &by_top {
        let a = |c: usize| c as f64 * grid.step * grid.step;
        println!("  over the part's top at {:.3}: {:.0} mm2 gouged under 0.1 mm deep, {:.0} mm2 0.1 to 0.5, {:.0} mm2 deeper", *z as f64 / 1000.0, a(n[0]), a(n[1]), a(n[2]));
    }
    let spots = part::spots(&grid, &left);
    let mut depths: Vec<f32> = spots.iter().map(|s| s.depth).collect();
    depths.sort_by(f32::total_cmp);
    let q = |f: f64| depths.get(((depths.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(0.0);
    println!("  {} gouged spots; deepest per spot: p50 {:.3} mm, p90 {:.3}, max {:.3}", spots.len(), q(0.5), q(0.9), q(1.0));
    for s in spots.iter().take(8) {
        let k = s.deepest;
        let size = |a: usize, b: usize| (b - a + 1) as f64 * grid.step;
        println!(
            "  {:.0} mm2 over {:.1} x {:.1} mm, {:.3} mm deep at ({:.2}, {:.2}), the part's top {:.3}",
            s.nodes as f64 * grid.step * grid.step,
            size(s.span[0], s.span[2]),
            size(s.span[1], s.span[3]),
            s.depth,
            grid.x(k % grid.nx),
            grid.y(k / grid.nx),
            pf.top[k]
        );
    }
    // Every node where the part's top steps to its right or above, plus the sample.
    let same = |a: f32, b: f32| a == b || (a.is_nan() && b.is_nan());
    let mut nodes: Vec<usize> = (0..grid.nx * grid.ny)
        .into_par_iter()
        .filter(|&k| {
            let (i, j) = (k % grid.nx, k / grid.nx);
            (i + 1 < grid.nx && !same(pf.top[k], pf.top[k + 1])) || (j + 1 < grid.ny && !same(pf.top[k], pf.top[k + grid.nx]))
        })
        .collect();
    let steps = nodes.len();
    nodes.extend_from_slice(sample);
    let clock = Instant::now();
    // (the part's top differs, what was left differs, the stock field is off here: a wall)
    let fails: Vec<(bool, bool, bool, usize)> = nodes
        .par_iter()
        .filter_map(|&k| {
            let (x, y) = (grid.x(k % grid.nx), grid.y(k / grid.nx));
            let top = PartField::top_at(frep, &pf.columns, x, y);
            let h = stock.cut_height(x, y);
            let exact = part::left(frep, &pf.columns, x, y, top, h);
            let ours = left[k] as f64;
            let top_bad = !same(top as f32, pf.top[k]);
            let edge = |v: f64| (v.abs() - part::TOL).abs() <= GATE_MM;
            let (a, b) = (Class::of(ours), Class::of(exact));
            let left_bad = if a != b { !(edge(ours) || edge(exact)) } else { matches!(a, Class::Left | Class::Gouge) && (ours - exact).abs() > GATE_MM };
            let wall = (field.h[k] as f64 - h).abs() > GATE_MM;
            (top_bad || left_bad).then_some((top_bad, left_bad, wall, k))
        })
        .collect();
    let top_bad = fails.iter().filter(|f| f.0).count();
    let walls = fails.iter().filter(|f| !f.0 && f.2).count();
    let real: Vec<&(bool, bool, bool, usize)> = fails.iter().filter(|f| f.0 || !f.2).collect();
    println!(
        "check: {} nodes ({steps} where the part's top steps, {} sampled) in {:.2} s: the part's top differs at {top_bad}; what was left differs at {} more, {walls} of them where the stock field is on a wall (step 1's gate)",
        nodes.len(),
        sample.len(),
        clock.elapsed().as_secs_f64(),
        fails.len() - top_bad
    );
    for &&(t, _, _, k) in real.iter().take(8) {
        let (x, y) = (grid.x(k % grid.nx), grid.y(k / grid.nx));
        let top = PartField::top_at(frep, &pf.columns, x, y);
        let h = stock.cut_height(x, y);
        println!(
            "  at ({x:.3}, {y:.3}): top {} vs {top:.4}, left {:.4} vs {:.4}{}",
            pf.top[k],
            left[k],
            part::left(frep, &pf.columns, x, y, top, h),
            if t { " (top)" } else { "" }
        );
    }
    Ok((real.len(), left))
}

/// `icam-sim part-bench <file.step> [cell mm ...]`: the part's cache levels as the part
/// view places them (128 bricks a side around the part's middle), built one after another
/// with no window, timed, with the exact queries each took.
/// `part-bench --gpu`: each level's surface bricks whose candidates the GPU kernel reads,
/// read there, and a sample of their nodes (about 200,000 a level) read again by the f64
/// CPU evaluator: the largest difference, in mm and in the cache's stored 16-bit steps,
/// and the nodes whose signs differ.
fn part_gpu_check(sol: &'static icam_core::solid3::Solid3, b: icam_sweep::Aabb, mid: icam_sweep::V3, cells: &[f64]) -> Result<ExitCode, String> {
    use icam_core::curve3::V3 as C3;
    use icam_core::solid3::Exact;
    let t = Instant::now();
    let ex = Exact::new(sol);
    println!("part-bench --gpu: evaluator set up in {:.2} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let finest = cells.iter().cloned().fold(f64::INFINITY, f64::min);
    let gpu = partgpu::PartGpu::new(&ex, 4.0 * finest * 3f64.sqrt())?;
    println!("part-bench --gpu: {}, {} sub-patches and pieces, packed and compiled in {:.2} s", gpu.adapter, gpu.nodes, t.elapsed().as_secs_f64());
    let pc = partcache::PartCache::exact(Exact::new(sol), b.min);
    for &h in cells {
        let (lo, nb, _) = icam_sweep::cache::place(b, h, mid, 128);
        let t = Instant::now();
        let firsts = pc.surface_bricks(h, lo, nb);
        let decide_s = t.elapsed().as_secs_f64();
        let half = 4.0 * h;
        let clamp = h * 3f64.sqrt();
        let t = Instant::now();
        let lists: Vec<Option<Vec<u32>>> = firsts
            .par_iter()
            .map(|&(p, _)| {
                let c = C3::new(p.x + half, p.y + half, p.z + half);
                gpu.list(&ex, c, 0.0, half * 3f64.sqrt() + clamp)
            })
            .collect();
        let list_s = t.elapsed().as_secs_f64();
        let on: Vec<(C3, f64, &[u32])> = firsts.iter().zip(&lists).filter_map(|((p, _), l)| l.as_ref().map(|l| (C3::new(p.x, p.y, p.z), h, l.as_slice()))).collect();
        let listed: usize = on.iter().map(|b| b.2.len()).sum();
        println!(
            "part-bench --gpu: cells of {h} mm: {} surface bricks decided in {decide_s:.2} s, candidates listed in {list_s:.2} s; {} ({:.0}%) all on the GPU, {:.1} candidates each",
            firsts.len(),
            on.len(),
            100.0 * on.len() as f64 / firsts.len().max(1) as f64,
            listed as f64 / on.len().max(1) as f64
        );
        if on.is_empty() {
            continue;
        }
        // The gather, timed against the scatter, whose values are checked.
        let t = Instant::now();
        let gathered = gpu.bricks(&on)?;
        let gather_s = t.elapsed().as_secs_f64();
        let jobs0 = partgpu::SCATTER_JOBS.load(std::sync::atomic::Ordering::Relaxed);
        let over0 = partgpu::SCATTER_OVERFLOWS.load(std::sync::atomic::Ordering::Relaxed);
        let t = Instant::now();
        let vals = gpu.bricks_scatter(&on)?;
        let first_s = t.elapsed().as_secs_f64();
        let jobs = partgpu::SCATTER_JOBS.load(std::sync::atomic::Ordering::Relaxed) - jobs0;
        let overflows = partgpu::SCATTER_OVERFLOWS.load(std::sync::atomic::Ordering::Relaxed) - over0;
        let t = Instant::now();
        gpu.bricks_scatter(&on)?;
        let gpu_s = t.elapsed().as_secs_f64();
        let total = on.len() * 729;
        let flags = |v: &[[f32; 2]]| (v.iter().filter(|x| x[1] == 0.0).count(), v.iter().filter(|x| x[1] == 1.0).count(), v.iter().filter(|x| x[1] == 2.0).count());
        let (ge, gf, gu) = flags(&gathered);
        let (se, sf, su) = flags(&vals);
        let differ = gathered.iter().zip(&vals).filter(|(g, s)| g[1] == 0.0 && s[1] == 0.0 && (g[0] - s[0]).abs() > 1e-4 * (1.0 + g[0].abs())).count();
        println!(
            "part-bench --gpu:   gather {gather_s:.3} s (exact {ge}, past the clamp {gf}, undecided {gu}); scatter {gpu_s:.3} s, {:.1}x (exact {se}, past the clamp {sf}, undecided {su}; {jobs} jobs, {overflows} dispatches overflowed); both exact but apart at {differ}",
            gather_s / gpu_s.max(1e-9)
        );
        let stride = (total / 200_000).max(1) | 1;
        let picks: Vec<usize> = (0..total).step_by(stride).collect();
        let t = Instant::now();
        let cmp: Vec<(usize, f64, f64, bool, bool)> = picks
            .par_iter()
            .map(|&i| {
                let (p, _, _) = on[i / 729];
                let n = i % 729;
                let x = p + C3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
                let sd = ex.sdf(x).0;
                (i, sd.d, vals[i][0] as f64, vals[i][1] != 0.0, sd.unsure)
            })
            .collect();
        let cpu_s = t.elapsed().as_secs_f64();
        let q = |d: f64| ((d / clamp).clamp(-1.0, 1.0) * 32767.0).round() as i64;
        let (mut max_band, mut max_step, mut over_one, mut flagged, mut unsure) = (0f64, 0i64, 0usize, 0usize, 0usize);
        let (mut signs, mut worst_sign) = (0usize, 0f64);
        let (mut far, mut far_wrong) = (0usize, 0usize);
        let mut whys = [0usize; 10];
        for v in &vals {
            if v[1] == 2.0 {
                whys[(v[0] as usize).min(9)] += 1;
            }
        }
        let mut worst: Vec<(i64, usize, f64, f64)> = Vec::new();
        for &(i, c, g, flag, u) in &cmp {
            if vals[i][1] == 1.0 {
                // Only a lower bound past the clamp is known: it must hold.
                far += 1;
                far_wrong += (c.abs() < g - 1e-4) as usize;
                continue;
            }
            if flag {
                flagged += 1;
                continue;
            }
            unsure += u as usize;
            if c.abs() <= clamp {
                max_band = max_band.max((c - g).abs());
            }
            if (c > 0.0) != (g > 0.0) {
                signs += 1;
                worst_sign = worst_sign.max(c.abs());
            }
            let dq = (q(c) - q(g)).abs();
            max_step = max_step.max(dq);
            if dq > 1 {
                over_one += 1;
                worst.push((dq, i, c, g));
            }
        }
        println!(
            "part-bench --gpu:   {total} nodes in {gpu_s:.3} s ({:.1} M a second; first call {first_s:.3} s); CPU check of {} nodes in {cpu_s:.2} s ({:.0} a second)",
            total as f64 / gpu_s.max(1e-9) / 1e6,
            cmp.len(),
            cmp.len() as f64 / cpu_s.max(1e-9)
        );
        println!(
            "part-bench --gpu:   within the clamp, largest difference {max_band:.2e} mm; stored values off by {max_step} steps at most, by more than one at {over_one}; signs differ at {signs} (largest CPU |d| among them {worst_sign:.2e} mm); {far} known only past the clamp ({far_wrong} of those bounds wrong); {flagged} for the CPU; {unsure} the CPU itself marks unsure"
        );
        println!(
            "part-bench --gpu:   all nodes left to the CPU, by why: a part not convex for the node within the clamp {}, Newton unsettled on a sub-patch {} and on a piece {}, edge sign unknown {}, nothing found {}, walk stack full {}, too many leaves left {}, refinement budget spent {}",
            whys[1], whys[2], whys[5], whys[3], whys[4], whys[6], whys[7], whys[8]
        );
        // The nodes the bricks left undecided, settled as the cache settles them (each a
        // point listing every feature that can be nearest it), by the points' scatter and by
        // the refining gather, against the CPU.
        let open: Vec<C3> = (0..total)
            .filter(|&i| vals[i][1] == 2.0)
            .map(|i| {
                let (p, _, _) = on[i / 729];
                let n = i % 729;
                p + C3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64)
            })
            .collect();
        let truth: Vec<f64> = open.par_iter().map(|&x| ex.sdf(x).0.d).collect();
        let open_lists: Vec<Option<Vec<u32>>> = open.par_iter().zip(&truth).map(|(&x, d)| gpu.list(&ex, x, 0.0, d.abs() * (1.0 + 1e-9) + 1e-9)).collect();
        let settle: Vec<(C3, f64, &[u32])> = open.iter().zip(&open_lists).filter_map(|(&x, l)| l.as_ref().map(|l| (x, clamp, l.as_slice()))).collect();
        let settle_truth: Vec<f64> = open.iter().zip(&open_lists).zip(&truth).filter(|((_, l), _)| l.is_some()).map(|(_, &d)| d).collect();
        let over1 = partgpu::SCATTER_OVERFLOWS.load(std::sync::atomic::Ordering::Relaxed);
        let r0: Vec<u64> = partgpu::REFINE_JOBS.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
        let t = Instant::now();
        let scattered = gpu.points_scatter(&settle)?;
        let scatter_s = t.elapsed().as_secs_f64();
        let rounds: Vec<u64> = partgpu::REFINE_JOBS.iter().zip(&r0).map(|(x, a)| x.load(std::sync::atomic::Ordering::Relaxed) - a).collect();
        let t = Instant::now();
        let refined = gpu.points_refined(&settle)?;
        let refined_s = t.elapsed().as_secs_f64();
        let tally = |v: &[[f32; 2]]| {
            let (mut exact, mut far, mut left, mut far_wrong, mut signs, mut worst, mut worst_past) = (0usize, 0usize, 0usize, 0usize, 0usize, 0f64, 0f64);
            let mut whys = [0usize; 10];
            for (g, &c) in v.iter().zip(&settle_truth) {
                if g[1] == 0.0 {
                    exact += 1;
                    if c.abs() <= clamp {
                        worst = worst.max((c - g[0] as f64).abs());
                    } else {
                        worst_past = worst_past.max((c - g[0] as f64).abs() / c.abs());
                    }
                    signs += ((c > 0.0) != (g[0] > 0.0)) as usize;
                } else if g[1] == 1.0 {
                    far += 1;
                    far_wrong += (c.abs() < clamp - 1e-4) as usize;
                } else {
                    left += 1;
                    whys[(g[0] as usize).min(9)] += 1;
                }
            }
            format!("exact {exact} (largest difference within the clamp {worst:.2e} mm, past it {worst_past:.1e} of the distance, signs wrong {signs}), past the clamp {far} ({far_wrong} wrong), left {left} (not convex {}, Newton unsettled {}, edge sign unknown {}, no room {}, budget {})", whys[1], whys[2] + whys[5], whys[3], whys[7], whys[8])
        };
        println!("part-bench --gpu:   settling the {} undecided nodes ({} listed):", open.len(), settle.len());
        println!("part-bench --gpu:     scatter {scatter_s:.3} s ({} dispatches overflowed to the gather), refined parts by round {rounds:?}: {}", partgpu::SCATTER_OVERFLOWS.load(std::sync::atomic::Ordering::Relaxed) - over1, tally(&scattered));
        println!("part-bench --gpu:     refining gather {refined_s:.3} s: {}", tally(&refined));
        worst.sort_by(|a, b| b.0.cmp(&a.0));
        for &(dq, i, c, g) in worst.iter().take(5) {
            let (p, _, _) = on[i / 729];
            let n = i % 729;
            let x = p + C3::new(h * (n % 9) as f64, h * ((n / 9) % 9) as f64, h * (n / 81) as f64);
            println!("part-bench --gpu:     off by {dq} steps at ({:.4}, {:.4}, {:.4}): CPU {c:.6} ({:?}), GPU {g:.6}", x.x, x.y, x.z, ex.sdf(x).0.feature);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn part_bench(file: &str, rest: &[String]) -> Result<ExitCode, String> {
    let census_only = rest.iter().any(|a| a == "--census");
    let gpu_check = rest.iter().any(|a| a == "--gpu");
    let census = census_only || rest.iter().any(|a| a == "--census-read");
    let cpu_only = rest.iter().any(|a| a == "--cpu");
    let verify = rest.iter().any(|a| a == "--verify");
    let warm = rest.iter().any(|a| a == "--warm");
    let shift: Option<i64> = rest.iter().position(|a| a == "--shift").and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok());
    let rest: Vec<String> = rest.iter().enumerate().filter(|&(i, a)| !a.starts_with("--") && !(i > 0 && rest[i - 1] == "--shift")).map(|(_, a)| a.clone()).collect();
    let cells: Vec<f64> = if rest.is_empty() { vec![8.0, 4.0, 2.0, 1.0] } else { rest.iter().map(|v| v.parse::<f64>().map_err(|_| format!("not a cell size: {v}"))).collect::<Result<_, _>>()? };
    let t0 = std::time::Instant::now();
    let brep = icam_core::step::read(Path::new(file))?;
    println!("part-bench: read in {:.1} s", t0.elapsed().as_secs_f64());
    let tol = brep.tolerance_mm.max(1e-6);
    let (sol, rep) = brep
        .solids
        .iter()
        .filter_map(|s| icam_core::solid3::Solid3::from_step(s, tol).ok())
        .max_by_key(|s| s.0.faces.len())
        .ok_or("no solid imports")?;
    let sol: &'static icam_core::solid3::Solid3 = Box::leak(Box::new(sol));
    println!("part-bench: {} faces imported in {:.1} s", sol.faces.len(), t0.elapsed().as_secs_f64());
    if rep.off_tol_faces > 0 {
        println!("part-bench: the file's edges lie off their surfaces beyond its declared {tol:.1e} mm on {} faces ({} points, up to {:.3e} mm)", rep.off_tol_faces, rep.off_tol_points, rep.off_tol_max);
        for n in rep.off_tol_notes.iter().take(3) {
            println!("part-bench:   {n}");
        }
    }
    let pad = (0.05 * (sol.hi - sol.lo).len()).max(9.0);
    let b = icam_sweep::Aabb::new(icam_sweep::V3::new(sol.lo.x - pad, sol.lo.y - pad, sol.lo.z - pad), icam_sweep::V3::new(sol.hi.x + pad, sol.hi.y + pad, sol.hi.z + pad));
    let mid = icam_sweep::V3::new(0.5 * (b.min.x + b.max.x), 0.5 * (b.min.y + b.max.y), 0.5 * (b.min.z + b.max.z));
    if gpu_check {
        return part_gpu_check(sol, b, mid, &cells);
    }
    if std::env::var_os("ICAM_NURBS_PROBE").is_some() {
        // B-spline faces by degree and span count, for sizing the GPU kernel's patches.
        let mut by: BTreeMap<(usize, usize, bool), (usize, usize, usize)> = BTreeMap::new();
        let mut edges: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &sol.faces {
            if let icam_core::surface3::Surf::Nurbs(ns) = &f.surf {
                let (a, bb) = ns.spans_uv();
                let pa = ns.patch(0, 0);
                let rational = pa.pw.iter().flatten().any(|h| (h[3] - pa.pw[0][0][3]).abs() > 1e-12);
                let e = by.entry((ns.du, ns.dv, rational)).or_default();
                e.0 += 1;
                e.1 += a * bb;
                e.2 += f.trim2.is_some() as usize;
            }
        }
        for e in &sol.edges {
            *edges.entry(match &e.curve {
                icam_core::curve3::Edge3::Line { .. } => "line",
                icam_core::curve3::Edge3::Arc { .. } => "arc",
                icam_core::curve3::Edge3::Ellipse { .. } => "ellipse",
                icam_core::curve3::Edge3::Spline(_) => "spline",
            }).or_default() += 1;
        }
        for ((du, dv, r), (n, spans, certified)) in &by {
            println!("part-bench: B-spline faces of degree {du} x {dv}{}: {n} ({certified} with certified loops), {spans} spans", if *r { ", rational" } else { "" });
        }
        println!("part-bench: edges {edges:?}");
        return Ok(ExitCode::SUCCESS);
    }
    let ex = icam_core::solid3::Exact::new(sol);
    let finest = cells.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut pc = if census || cpu_only { partcache::PartCache::exact(ex, b.min) } else { partcache::PartCache::exact_gpu(ex, b.min, finest) };
    partcache::CENSUS_ON.store(census, std::sync::atomic::Ordering::Relaxed);
    partcache::CENSUS_ONLY.store(census_only, std::sync::atomic::Ordering::Relaxed);
    // --warm: the GPU kept busy two seconds first, its clocks up for the first level.
    if warm {
        pc.warm(2.0);
    }
    for h in cells {
        let (lo, nb, _) = icam_sweep::cache::place(b, h, mid, 128);
        partcache::QUERIES.store(0, std::sync::atomic::Ordering::Relaxed);
        partcache::GPU_BRICKS.store(0, std::sync::atomic::Ordering::Relaxed);
        partcache::GPU_REREADS.store(0, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut t) = partcache::CENSUS.lock() {
            *t = ([0; 5], 0, 0, 0);
        }
        let t = std::time::Instant::now();
        let level = pc.level(h, lo, nb);
        let secs = t.elapsed().as_secs_f64();
        // --shift n: the level placed again n bricks along x (as a pan places it), timed.
        if let Some(n) = shift {
            let lo2 = lo + icam_sweep::V3::new(8.0 * h * n as f64, 0.0, 0.0);
            let t = std::time::Instant::now();
            let again = pc.level(h, lo2, nb);
            println!("part-bench:   placed again {n} bricks along x in {:.2} s ({} bricks with values)", t.elapsed().as_secs_f64(), again.atlas.len() / icam_sweep::cache::WORDS);
        }
        let q = partcache::QUERIES.load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "part-bench: cells of {h} mm, {} x {} x {} bricks: {} with values in {secs:.2} s, {q} exact queries ({:.0} a second)",
            nb[0],
            nb[1],
            nb[2],
            level.atlas.len() / icam_sweep::cache::WORDS,
            q as f64 / secs.max(1e-9)
        );
        let take = |a: &[std::sync::atomic::AtomicU64]| a.iter().map(|x| x.swap(0, std::sync::atomic::Ordering::Relaxed)).collect::<Vec<u64>>();
        let (cpu_at, us) = (take(&partcache::CPU_AT), take(&partcache::STEP_US));
        println!(
            "part-bench:   steps: boxes decided {:.2} s (their lists {:.2}, GPU {:.2}), lists {:.2} s, GPU bricks {:.2} s, bricks finished {:.2} s; CPU reads: boxes with no bound {}, boxes the GPU left {} (past the box {}, not listed {}), bricks with no list {}, brick centers {}",
            us[0] as f64 * 1e-6,
            us[4] as f64 * 1e-6,
            us[5] as f64 * 1e-6,
            us[1] as f64 * 1e-6,
            us[2] as f64 * 1e-6,
            us[3] as f64 * 1e-6,
            cpu_at[0],
            cpu_at[1] + cpu_at[6] + cpu_at[7],
            cpu_at[6],
            cpu_at[7],
            cpu_at[2],
            cpu_at[3]
        );
        let pass_us = take(&partgpu::PASS_US);
        if pass_us.iter().any(|&x| x > 0) {
            println!("part-bench:   the bricks' scatter passes, waited for alone: seed {:.2} s, jobs {:.2}, scatter {:.2}, edge signs {:.2}, finish {:.2}", pass_us[0] as f64 * 1e-6, pass_us[1] as f64 * 1e-6, pass_us[2] as f64 * 1e-6, pass_us[5] as f64 * 1e-6, pass_us[6] as f64 * 1e-6);
        }
        let gb = partcache::GPU_BRICKS.load(std::sync::atomic::Ordering::Relaxed);
        if gb > 0 {
            println!("part-bench:   {gb} bricks read on the GPU, {} of their nodes bridged from their neighbors", partcache::GPU_REREADS.load(std::sync::atomic::Ordering::Relaxed));
            let e = h.log2().round() as i32;
            let n = pc.provisional(e);
            let t = std::time::Instant::now();
            let cpu = pc.settle(e);
            let cpu_at = take(&partcache::CPU_AT);
            let pass_us = take(&partgpu::PASS_US);
            if pass_us.iter().any(|&x| x > 0) {
                println!("part-bench:   the settle's scatter passes, waited for alone: seed {:.2} s, jobs {:.2}, scatter {:.2}, leaves {:.2}, refine {:.2}, edge signs {:.2}, finish {:.2}", pass_us[0] as f64 * 1e-6, pass_us[1] as f64 * 1e-6, pass_us[2] as f64 * 1e-6, pass_us[3] as f64 * 1e-6, pass_us[4] as f64 * 1e-6, pass_us[5] as f64 * 1e-6, pass_us[6] as f64 * 1e-6);
            }
            println!("part-bench:   the settle's nodes the GPU left, by its code: {:?}", take(&partcache::SETTLE_WHY));
            println!("part-bench:   settled in {:.2} s: {n} bridged nodes read exactly, {cpu} of them on the CPU ({} not listed, {} the GPU left undecided; {} brick centers read first)", t.elapsed().as_secs_f64(), cpu_at[8], cpu_at[9], cpu_at[4]);
        }
        if verify {
            let t = std::time::Instant::now();
            if let Some((n, signs, most, over)) = pc.verify(h.log2().round() as i32, std::env::var("ICAM_VERIFY_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(29)) {
                println!("part-bench:   verified {n} stored nodes against the CPU in {:.1} s: signs differ at {signs}, stored steps off by {most} at most, by more than one at {over}", t.elapsed().as_secs_f64());
            }
        }
        if !census {
            continue;
        }
        if let Ok(t) = partcache::CENSUS.lock() {
            let n = t.1.max(1) as f64;
            println!(
                "part-bench:   per brick with surface ({}), features that can be nearest: analytic faces {:.1}, B-spline faces {:.1}, line and arc edges {:.1}, other edges and knot lines {:.1}, vertices and poles {:.1}; most {}; bricks with any B-spline face or other edge {} ({:.0}%)",
                t.1,
                t.0[0] as f64 / n,
                t.0[1] as f64 / n,
                t.0[2] as f64 / n,
                t.0[3] as f64 / n,
                t.0[4] as f64 / n,
                t.3,
                t.2,
                100.0 * t.2 as f64 / n
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}
