//! icam-rough: rough a prismatic part from its STEP file, with the operations found from
//! its field. Writes the program and a report, prints a summary, and exits 2 when the
//! checker finds a gouge. On the same pockets, `finish` runs the finishing-allowance
//! study and `assign` assigns material to tools; each roughs the part first, for the
//! stock every op leaves, and writes its own page (finish.html, assign.html).
//!
//!   icam-rough [check | finish | assign | bench | continuity] <part.step> [options]
//!   (continuity: rough twice, the second time with 5 um more allowance, and report how
//!    much of each op's path, and of the whole program in cutting order, moved more than
//!    1 mm; the gate is 5 %. Exits 3 when an op or the program fails it)
//!   (check: compile the part and report what the import kept, nothing else;
//!    --turns <n> and --flip compile it turned and upside down; --ops lists the
//!    operations found)
//!     --tool <mm>            tool diameter (default 12.7)
//!     --corner <mm>          corner radius (default 0)
//!     --tool-number <n>      the T number written (default 1)
//!     --plan <kind>          maxcut | physics | harmonic | ratio | conductive | walls | objective (default maxcut)
//!     --material <name>      physical limits (al6061); physics, walls and objective need it
//!     --rpm <n>              spindle speed (default 18000)
//!     --allowance <mm>       wall and floor stock to leave (default 0.254)
//!     --stepdown <mm>        deepest slab (default: one slab per floor)
//!     --stock-top <z>        stock top (default: the part's top)
//!     --stepover <mm>        max-cut stepover (default: from the reference cut or the material)
//!     --persist <mm>         spiral plans: seed significance (default 1/8 of the tool radius)
//!     --oneway               max-cut: climb only
//!     --greedy               max-cut: nearest-ready order
//!     --feed <mm/s>          reference feed when there is no material (default 34)
//!     --accel <m/s2>         machine acceleration (default 2.5)
//!     --feed-cap <x>         fastest cutting feed, a multiple of the reference feed: a
//!                            light cut's feed rises to keep its chip, up to this (default 2)
//!     --dialect <name>       fanuc | multicam (default fanuc)
//!     --out <dir>            output folder (default icam-rough-out)
//!     --origin <x,y,z>       the work frame's zero in part coordinates (default: the
//!                            stock's corner, Z on the table; 0,0,0 posts in the frame a
//!                            placed part was placed into)
//!     --window <x0,y0,x1,y1> only the pockets whose seed point is in this box (and, for
//!                            bench, the baseline's moves that pass through it)
//!   finish:
//!     --finish-tool <mm>     the finisher, 6.35 or 12.7 (default 6.35)
//!     --tol <in>             tool deflection allowed at the wall (default 0.0005, assumed)
//!     --dmin <mm>            thinnest allowance the field may leave (default 0.05, assumed)
//!     --slope <mm/mm>        field slope limit along the wall (default 0.05, assumed)
//!   assign:
//!     --tools <mm,...>       the smaller end mills (default 6.35,3.175,1.5875)
//!     --vendor <csv>         vendor cut-data table for the wide-cut envelope
//!                            (default none: no envelope)
//!     --noenv                no wide-cut envelope
//!     --defl <in>            deflection limit for every tool (default 0.001)
//!     --cell <mm>            raster cell (default 0.25)
//!     --tchange <s>          tool change (default 15, assumed)
//!     --tlink <s>            link: stop, lift, lower in cleared space (default 0.5, assumed)
//!   bench: the roughing scored against a baseline program on the first roughing spike's objectives
//!     --baseline <program>   the baseline program (its tool sheet beside it, same name .set)
//!     --set <file>           the tool sheet, when it is elsewhere
//!     --align                find where the program places the part (turned, upside
//!                            down, shifted) from its own cuts
//!     --place <t[f],x,y,z>   place it so instead (quarter turns, f upside down, shift)
//!     --reference <mm/s,deg> the reference cut (feed, engagement) instead of the
//!                            baseline's medians (with --window, pass the whole program's:
//!                            the window moves them); also sets it for a plain roughing

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use icam_assign::{ModelOptions, Penalties};
use icam_core::frep::Frep;
use icam_finish::Settings;
use icam_post::post::Dialect;
use icam_post::{Box2, ToolDef};
use icam_rough::assign::{assign_pockets, AssignJob};
use icam_rough::finish::study_pockets;
use icam_rough::{assign_report, bench, bench_report, finish_report, report, run, Job, JobOut, PlanKind};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match go(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("icam-rough: {e}");
            ExitCode::from(1)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cmd {
    Rough,
    Finish,
    Assign,
    Bench,
    Check,
    Continuity,
}

fn go(args: &[String]) -> Result<ExitCode, String> {
    let (cmd, args) = match args.first().map(String::as_str) {
        Some("finish") => (Cmd::Finish, &args[1..]),
        Some("assign") => (Cmd::Assign, &args[1..]),
        Some("bench") => (Cmd::Bench, &args[1..]),
        Some("check") => (Cmd::Check, &args[1..]),
        Some("continuity") => (Cmd::Continuity, &args[1..]),
        _ => (Cmd::Rough, args),
    };
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let num = |name: &str| -> Result<Option<f64>, String> { opt(name).map(|v| v.parse::<f64>().map_err(|_| format!("{name}: not a number: {v}"))).transpose() };
    let takes_value = [
        "--tool", "--corner", "--tool-number", "--plan", "--material", "--rpm", "--allowance", "--stepdown", "--stock-top", "--stepover", "--persist", "--feed", "--accel",
        "--dialect", "--feed-cap", "--out", "--finish-tool", "--tol", "--dmin", "--slope", "--tools", "--vendor", "--defl", "--cell", "--tchange", "--tlink", "--baseline", "--set", "--turns", "--place",
        "--window", "--reference", "--origin",
    ];
    let mut file = None;
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
        } else if takes_value.contains(&a.as_str()) {
            skip = true;
        } else if !a.starts_with("--") {
            file = Some(PathBuf::from(a));
        }
    }
    let file = file.ok_or("usage: icam-rough <part.step> [options] (see the source header for the options)")?;
    let plan = match opt("--plan") {
        Some(p) => PlanKind::parse(&p).ok_or(format!("unknown plan '{p}'"))?,
        None => PlanKind::MaxCut,
    };
    let tool = ToolDef {
        number: opt("--tool-number").map_or(Ok(1), |v| v.parse::<i32>().map_err(|_| format!("--tool-number: not a number: {v}")))?,
        kind: "ENDMILL".into(),
        diameter: num("--tool")?.unwrap_or(12.7),
        corner_radius: num("--corner")?.unwrap_or(0.0),
        description: "FLAT END MILL".into(),
        tip_angle_deg: 118.0,
    };
    let mut job = Job::new(tool, plan);
    job.progress = true;
    job.material = opt("--material");
    job.rpm = num("--rpm")?;
    if let Some(a) = num("--allowance")? {
        job.allowance = a;
    }
    job.stepdown = num("--stepdown")?;
    job.stock_top = num("--stock-top")?;
    job.stepover = num("--stepover")?;
    job.persist = num("--persist")?;
    job.one_way = flag("--oneway");
    job.greedy = flag("--greedy");
    if let Some(f) = num("--feed")? {
        job.v_prog = f;
    }
    if let Some(a) = num("--accel")? {
        job.accel = a * 1000.0;
    }
    if let Some(c) = num("--feed-cap")? {
        if c < 1.0 {
            return Err(format!("--feed-cap: {c} is under the reference feed"));
        }
        job.feed_cap = c;
    }
    job.dialect = match opt("--dialect").as_deref() {
        None | Some("fanuc") => Dialect::Fanuc,
        Some("multicam") => Dialect::MultiCam,
        Some(d) => return Err(format!("unknown dialect '{d}' (fanuc, multicam)")),
    };
    if let Some(v) = opt("--window") {
        let f: Vec<f64> = v.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("--window: expected x0,y0,x1,y1, got {v}"))?;
        if f.len() != 4 || f[0] >= f[2] || f[1] >= f[3] {
            return Err(format!("--window: expected x0,y0,x1,y1 with x0 < x1 and y0 < y1, got {v}"));
        }
        job.window = Some(Box2::new(f[0], f[1], f[2], f[3]));
    }
    if let Some(v) = opt("--origin") {
        let f: Vec<f64> = v.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("--origin: expected x,y,z, got {v}"))?;
        if f.len() != 3 {
            return Err(format!("--origin: expected x,y,z, got {v}"));
        }
        job.origin = Some((f[0], f[1], f[2]));
    }
    if let Some(v) = opt("--reference") {
        let f: Vec<f64> = v.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("--reference: expected feed,deg, got {v}"))?;
        if f.len() != 2 || f[0] <= 0.0 || f[1] <= 0.0 || f[1] > 180.0 {
            return Err(format!("--reference: expected feed (mm/s) > 0 and engagement (deg) in (0, 180], got {v}"));
        }
        job.reference = Some((f[0], f[1].to_radians()));
        (job.v_prog, job.phi_ref) = (f[0], f[1].to_radians());
    }
    let name = file.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    job.title = format!("IMPLICITCAM FIELD ROUGHING {} PLAN {}", name.to_uppercase(), plan.name().to_uppercase());

    let clock = Instant::now();
    let baseline = match (cmd, opt("--baseline")) {
        (Cmd::Bench, Some(nc)) => Some((bench::read_baseline(Path::new(&nc), opt("--set").map(PathBuf::from).as_deref())?, PathBuf::from(nc))),
        (Cmd::Bench, None) => return Err("bench needs --baseline <program>".into()),
        _ => None,
    };
    let (frep, (lo, hi), rep) = match (&baseline, flag("--align")) {
        (Some((base, _)), true) => {
            let part = icam_core::prismatic::ReadPart::read(&file)?;
            let a = bench::align(&part, base)?;
            let p = a.place;
            println!(
                "placed in the program: {} quarter turn(s){}, shift ({:.3}, {:.3}, {:.3}); its tool tips reach {:.4} mm into the part on average (orientations tried: {})",
                p.quarter_turns,
                if p.flip { ", upside down" } else { "" },
                p.shift.x,
                p.shift.y,
                p.shift.z,
                a.depth,
                a.tried.iter().map(|(q, d)| format!("{}{} {:.3}", q.quarter_turns, if q.flip { "f" } else { "" }, d)).collect::<Vec<_>>().join(", ")
            );
            for (q, e) in &a.failed {
                println!("note: orientation {}{} did not compile: {e}", q.quarter_turns, if q.flip { " upside down" } else { "" });
            }
            (a.frep, (icam_core::geom::V2::new(a.bbox.x0, a.bbox.y0), icam_core::geom::V2::new(a.bbox.x1, a.bbox.y1)), a.report)
        }
        _ if opt("--place").is_some() => {
            // "3f,1073.15,655.288,25.4": quarter turns (f: upside down first), then the shift.
            let v = opt("--place").unwrap_or_default();
            let f: Vec<&str> = v.split(',').map(str::trim).collect();
            let bad = || format!("--place: expected turns[f],x,y,z, got {v}");
            if f.len() != 4 {
                return Err(bad());
            }
            let flip = f[0].ends_with('f');
            let turns: u8 = f[0].trim_end_matches('f').parse().map_err(|_| bad())?;
            let n = |s: &str| s.parse::<f64>().map_err(|_| bad());
            let place = icam_core::prismatic::Place::new(turns, flip, n(f[1])?, n(f[2])?, n(f[3])?);
            icam_core::prismatic::ReadPart::read(&file)?.compile(&place)?
        }
        _ if cmd == Cmd::Check && (flag("--flip") || opt("--turns").is_some()) => {
            let turns: u8 = opt("--turns").map_or(Ok(0), |t| t.parse().map_err(|_| format!("--turns: not a number: {t}")))?;
            icam_core::prismatic::ReadPart::read(&file)?.compile(&icam_core::prismatic::Place::new(turns, flag("--flip"), 0.0, 0.0, 0.0))?
        }
        _ => icam_core::prismatic::compile_file(&file)?,
    };
    println!(
        "part  {name}: {} solids, {} horizontal faces, levels {:?}, box ({:.3}, {:.3}) .. ({:.3}, {:.3}); compiled in {:.2} s",
        rep.solids,
        rep.horizontal,
        frep.levels.iter().map(|z| (z * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        lo.x,
        lo.y,
        hi.x,
        hi.y,
        clock.elapsed().as_secs_f64()
    );
    if cmd == Cmd::Check {
        println!(
            "import: {} planes up, {} vertical planes, {} vertical cylinders, {} freeform walls; {} lines, {} arcs, {} splines; {} fillets as envelopes at {} heights, {} overhangs filled; sections {}, largest {} curves",
            rep.horizontal,
            rep.vertical_planes,
            rep.vertical_cylinders,
            rep.vertical_freeform,
            rep.lines,
            rep.arcs,
            rep.splines,
            rep.envelopes,
            rep.envelope_levels,
            rep.overhangs,
            frep.sections.len(),
            frep.sections.iter().map(|s| s.edges.len()).max().unwrap_or(0)
        );
        if flag("--ops") {
            let top = job.stock_top.unwrap_or(frep.levels.iter().copied().fold(f64::NEG_INFINITY, f64::max));
            let r = job.tool.diameter / 2.0;
            for s in icam_rough::ops::find(&frep, Box2::new(lo.x, lo.y, hi.x, hi.y), top, r, job.allowance, job.stepdown, 1.0) {
                if job.window.is_some_and(|w| !w.contains(s.hint.0, s.hint.1)) {
                    continue;
                }
                let b = s.region;
                println!("{:<22} seed ({:.1}, {:.1})  box ({:.1}, {:.1}) .. ({:.1}, {:.1})  {:.0} x {:.0} mm", s.name, s.hint.0, s.hint.1, b.x0, b.y0, b.x1, b.y1, b.x1 - b.x0, b.y1 - b.y0);
            }
        }
        return Ok(ExitCode::SUCCESS);
    }
    if let Some((base, nc)) = &baseline {
        let b = bench::bench(&frep, Box2::new(lo.x, lo.y, hi.x, hi.y), &job, base)?;
        let seconds = clock.elapsed().as_secs_f64();
        let program = nc.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        for w in &base.warnings {
            println!("baseline: {w}");
        }
        println!(
            "reference cut: {:.0} ipm at {:.0} deg; baseline rough ops {}; {} levels; field ops {}",
            b.v_prog * 60.0 / 25.4,
            b.phi_ref.to_degrees(),
            b.rough_ops.join(", "),
            b.levels.len(),
            b.ours.ops.len()
        );
        for n in &b.notes {
            println!("note: {n}");
        }
        failures(&b.ours, job.tool.diameter / 2.0);
        let rows = bench::score(&b, job.material.is_some());
        println!("{:<52} {:<44} {:>26} {:>24}  result", "objective", "bar", "baseline", "ours");
        for r in &rows {
            println!("{:<52} {:<44} {:>26} {:>24}  {}", r.objective, r.bar, r.base, r.ours, if r.pass { "met" } else { "NOT MET" });
        }
        let dir = PathBuf::from(opt("--out").unwrap_or_else(|| "icam-rough-out".into()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let page = dir.join("bench.html");
        std::fs::write(&page, bench_report::html(&name, &program, &b, job.material.as_deref(), job.tool.diameter, seconds)).map_err(|e| format!("{}: {e}", page.display()))?;
        println!("{} of {} met; wrote {} ({seconds:.1} s)", rows.iter().filter(|r| r.pass).count(), rows.len(), page.display());
        return Ok(ExitCode::SUCCESS);
    }
    if cmd == Cmd::Continuity {
        return continuity(&frep, Box2::new(lo.x, lo.y, hi.x, hi.y), &job);
    }
    let out = run(&frep, Box2::new(lo.x, lo.y, hi.x, hi.y), &job)?;
    if cmd != Cmd::Rough {
        let planned = out.ops.iter().filter(|o| o.path.is_some()).count();
        let gouging = out.ops.iter().filter(|o| o.gouged()).count();
        println!("rough {planned} of {} ops planned, {} moves, {gouging} gouging ({:.1} s)", out.ops.len(), out.moves.len(), clock.elapsed().as_secs_f64());
        let dir = PathBuf::from(opt("--out").unwrap_or_else(|| "icam-rough-out".into()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        return if cmd == Cmd::Finish { finish(&frep, &job, &out, &num, &name, &dir) } else { assign(&frep, &job, &out, &opt, &num, &flag, &name, &dir) };
    }
    let seconds = clock.elapsed().as_secs_f64();
    let dir = PathBuf::from(opt("--out").unwrap_or_else(|| "icam-rough-out".into()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let write = |p: &Path, text: &str| std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display()));
    write(&dir.join("program.nc"), &out.program)?;
    write(&dir.join("report.html"), &report::html(&job, &name, &out, seconds))?;

    println!("{:<22} {:>8} {:>7} {:>9} {:>9} {:>14} {:>7} {:>10}  status", "op", "z cut", "ap", "planned s", "posted s", "phi p95/max", "gouges", "uncut mm2");
    for o in &out.ops {
        let m = o.posted.as_ref().or(o.planned.as_ref());
        println!(
            "{:<22} {:>8.3} {:>7.2} {:>9} {:>9} {:>14} {:>7} {:>10}  {}",
            o.spec.name,
            o.spec.z_cut,
            o.ap,
            o.planned.as_ref().map_or("-".into(), |m| format!("{:.1}", m.time())),
            o.posted.as_ref().map_or("-".into(), |m| format!("{:.1}", m.time())),
            m.map_or("-".into(), |m| format!("{:.0}/{:.0}", m.phi_p95_deg, m.phi_max_deg)),
            o.check.as_ref().map_or("-".into(), |c| c.gouges.to_string()),
            o.check.as_ref().map_or("-".into(), |c| format!("{:.1}", c.residue_area)),
            o.skipped.as_deref().map_or_else(|| if o.gouged() { "GOUGES".to_string() } else { "ok".to_string() }, |s| format!("skipped: {s}"))
        );
    }
    for o in &out.ops {
        for n in o.notes.iter().filter(|n| n.contains("--stepdown") || n.contains("left uncut")) {
            println!("note: {}: {n}", o.spec.name);
        }
    }
    for w in &out.warnings {
        println!("warning: {w}");
    }
    failures(&out, job.tool.diameter / 2.0);
    println!("wrote {} and {} ({seconds:.1} s)", dir.join("program.nc").display(), dir.join("report.html").display());
    Ok(if out.ops.iter().any(|o| o.gouged()) { ExitCode::from(2) } else { ExitCode::SUCCESS })
}

/// The continuity gate: the job and the job with 5 um more allowance, each op's path (its
/// moves at feed, in cutting order) and the whole program's compared by
/// `icam_strategy::continuity::compare`.
fn continuity(frep: &Frep, part: Box2, job: &Job) -> Result<ExitCode, String> {
    use icam_strategy::continuity::{compare, LIMIT, SHARE_LIMIT, WINDOW};
    use icam_strategy::path::P;
    let base = run(frep, part, job)?;
    let mut nudged = job.clone();
    nudged.allowance += 0.005;
    let other = run(frep, part, &nudged)?;
    let path = |o: &JobOut, k: usize| -> Vec<P> { o.ops[k].ours.iter().filter(|s| s.tag != icam_post::Tag::Rapid).map(|s| (s.x, s.y)).collect() };
    // An op is matched by its slab's rank and its pocket number (the cut height in its
    // name moves with the allowance).
    let key = |o: &JobOut, k: usize| -> (usize, String) {
        let mut zs: Vec<f64> = o.ops.iter().map(|q| q.spec.z_cut).collect();
        zs.sort_by(|a, b| b.total_cmp(a));
        zs.dedup();
        let rank = zs.iter().position(|&z| z == o.ops[k].spec.z_cut).unwrap_or(0);
        (rank, o.ops[k].spec.name.split_once(' ').map_or(String::new(), |(_, p)| p.to_string()))
    };
    let whole = |o: &JobOut| -> Vec<P> { (0..o.ops.len()).flat_map(|k| path(o, k)).collect() };
    let mut failed = 0;
    println!("{:<24} {:>9} {:>9} {:>9}", "op", "moved %", "p95 mm", "max mm");
    for (k, o) in base.ops.iter().enumerate() {
        let Some(j) = (0..other.ops.len()).find(|&j| key(&other, j) == key(&base, k)) else {
            println!("{:<24} not in the nudged run", o.spec.name);
            failed += 1;
            continue;
        };
        let (a, b) = (path(&base, k), path(&other, j));
        if a.is_empty() && b.is_empty() {
            continue;
        }
        let (share, p95, max) = compare(&a, &b, LIMIT, WINDOW);
        if share > SHARE_LIMIT {
            failed += 1;
        }
        println!("{:<24} {:>9.2} {:>9.3} {:>9.3}{}", o.spec.name, 100.0 * share, p95, max, if share > SHARE_LIMIT { "  FAIL" } else { "" });
    }
    let (share, p95, max) = compare(&whole(&base), &whole(&other), LIMIT, WINDOW);
    let program_fails = share > SHARE_LIMIT;
    println!("{:<24} {:>9.2} {:>9.3} {:>9.3}{}", "whole program", 100.0 * share, p95, max, if program_fails { "  FAIL" } else { "" });
    println!("continuity: {failed} op(s) over {:.0} %, program {}", 100.0 * SHARE_LIMIT, if program_fails { "FAIL" } else { "pass" });
    Ok(if failed > 0 || program_fails { ExitCode::from(3) } else { ExitCode::SUCCESS })
}

/// Where the checks failed: each gouging op's deepest gouge (the deepest 12 ops), with the
/// program line that reaches past it, and the pockets left with the most reachable stock.
fn failures(out: &JobOut, r: f64) {
    let mut deep: Vec<(f64, &str, icam_post::V3)> = out
        .ops
        .iter()
        .filter_map(|o| o.check.as_ref().and_then(|c| c.worst.first()).map(|(p, d)| (*d, o.spec.name.as_str(), *p)))
        .collect();
    deep.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // The program may open with the checker's warning line; the moves count from the draft.
    let lines: Vec<&str> = out.program.lines().skip(usize::from(out.program.starts_with("(CHECKER"))).collect();
    for (d, op, p) in deep.iter().take(12) {
        let by = run::gouge_culprit(&out.moves, *p, r).map_or("no move reaches it".to_string(), |(k, depth)| {
            let m = &out.moves[k];
            let of = out.ops.iter().find(|o| (o.range.0..o.range.1).contains(&k)).map_or("?", |o| o.spec.name.as_str());
            format!("{:?} line {} of {of} reaches {depth:.3} mm past it: {}", m.kind, m.line, lines.get(m.line.saturating_sub(1)).copied().unwrap_or("?"))
        });
        println!("gouge {:.3} mm at ({:.3}, {:.3}, {:.3}) in {op}: {by}", -d, p.x, p.y, p.z);
    }
    let mut uncut: Vec<(&str, f64, &[(f64, f64)])> =
        out.ops.iter().filter_map(|o| o.check.as_ref().filter(|c| c.residue_area > run::MAX_UNCUT).map(|c| (o.spec.name.as_str(), c.residue_area, c.spots.as_slice()))).collect();
    uncut.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (op, a, spots) in uncut.iter().take(8) {
        // A few of the uncut points, spread over the list.
        let at: Vec<String> = (0..5).filter_map(|k| spots.get(k * spots.len().saturating_sub(1) / 4)).map(|(x, y)| format!("({x:.2}, {y:.2})")).collect();
        println!("uncut {a:.1} mm2 in {op}{}", if at.is_empty() { String::new() } else { format!(", at {}", at.join(" ")) });
    }
}

fn finish(frep: &Frep, job: &Job, rough: &JobOut, num: &dyn Fn(&str) -> Result<Option<f64>, String>, part: &str, dir: &Path) -> Result<ExitCode, String> {
    let mut s = Settings { accel: job.accel, allowance: job.allowance, ..Settings::default() };
    if let Some(v) = num("--tol")? {
        s.tol_in = v;
    }
    if let Some(v) = num("--dmin")? {
        s.d_min = v;
    }
    if let Some(v) = num("--slope")? {
        s.slope = v;
    }
    s.fit_levels();
    let d = num("--finish-tool")?.unwrap_or(6.35);
    let ft = ToolDef { number: job.tool.number + 1, kind: "ENDMILL".into(), diameter: d, corner_radius: 0.0, description: "FLAT END MILL".into(), tip_angle_deg: 118.0 };
    let clock = Instant::now();
    let outs = study_pockets(frep, job, rough, &ft, &s)?;
    println!(
        "finishing study: tolerance {:.4} in, accel {:.1} m/s2, allowance (a) {:.3} mm, field [{:.3}, {:.3}] mm, slope {:.2}; finisher {:.3} mm, rougher {:.3} mm",
        s.tol_in,
        s.accel / 1000.0,
        s.allowance,
        s.d_min,
        s.d_max,
        s.slope,
        d,
        job.tool.diameter
    );
    println!("wall-pass cutting time, s, at the same tolerance");
    println!(
        "{:<22} {:>6} {:>6} | {:>14} {:>7} {:>6} {:>6} | {:>13} {:>6} | {:>12} | {:>13} {:>10} {:>5}",
        "op", "ap mm", "wall", "per-sample (a)", "(b)fast", "(b)eq", "thin", "wall feed (a)", "(b)", "one feed (a)", "stock (b)/(a)", "rough +mm3", "+s"
    );
    let mut tot = [0.0f64; 8];
    for o in &outs {
        let (Some(pk), Some(r)) = (&o.pocket, &o.result) else {
            println!("{:<22} skipped: {}", o.op, o.skipped.as_deref().unwrap_or(""));
            continue;
        };
        let wall: f64 = pk.loops.iter().map(|l| l.length).sum();
        let (dv, dt) = r.rougher(pk, &r.fast);
        let row = [r.a_sched.time, r.fast.sched().time, r.eq.sched().time, r.thin_sched.time, r.a_wall.time, r.wall.sched().time, r.a_one.time, dv];
        for (t, v) in tot.iter_mut().zip(row) {
            *t += v;
        }
        println!(
            "{:<22} {:>6.1} {:>6.0} | {:>14.2} {:>7.2} {:>6.2} {:>6.2} | {:>13.2} {:>6.2} | {:>12.2} | {:>12.0}% {:>10.0} {:>5.2}",
            o.op,
            pk.ap,
            wall,
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            100.0 * r.fast.area / r.area_a.max(1e-9),
            dv,
            dt
        );
    }
    println!(
        "{:<22} {:>6} {:>6} | {:>14.2} {:>7.2} {:>6.2} {:>6.2} | {:>13.2} {:>6.2} | {:>12.2} | {:>13} {:>10.0}",
        "TOTAL", "", "", tot[0], tot[1], tot[2], tot[3], tot[4], tot[5], tot[6], "", tot[7]
    );
    println!("  (b)fast: the fastest per-sample profile any field allows, with the most stock it carries; (b)eq: minimum time at (a)'s stock");
    println!("  wall feed: one F on the wall, corner zones (rougher leftovers) at their own safe feed; one feed: one F for the pass");
    for o in &outs {
        let (Some(pk), Some(r)) = (&o.pocket, &o.result) else { continue };
        let corner = r.corner.iter().filter(|&&c| c).count() as f64 * pk.ds;
        println!(
            "{}: {} wall loops ({} islands), corner zones {:.0} mm, compliance {:.2} um/N, {} simulations, {:.1} s",
            o.op,
            pk.loops.len(),
            pk.loops.iter().filter(|l| l.island).count(),
            corner,
            r.model.compliance() * 1000.0,
            r.sims,
            o.ms / 1000.0
        );
        for n in &r.notes {
            println!("  note: {n}");
        }
    }
    let seconds = clock.elapsed().as_secs_f64();
    let page = dir.join("finish.html");
    std::fs::write(&page, finish_report::html(part, &outs, &s, job.tool.diameter, seconds)).map_err(|e| format!("{}: {e}", page.display()))?;
    println!("studied {} pockets in {seconds:.1} s; wrote {}", outs.iter().filter(|o| o.result.is_some()).count(), page.display());
    Ok(ExitCode::SUCCESS)
}

fn assign(
    frep: &Frep,
    job: &Job,
    rough: &JobOut,
    opt: &dyn Fn(&str) -> Option<String>,
    num: &dyn Fn(&str) -> Result<Option<f64>, String>,
    flag: &dyn Fn(&str) -> bool,
    part: &str,
    dir: &Path,
) -> Result<ExitCode, String> {
    let mut sheet = vec![job.tool.clone()];
    let sizes = opt("--tools").unwrap_or_else(|| "6.35,3.175,1.5875".into());
    for v in sizes.split(',').map(str::trim).filter(|v| !v.is_empty()) {
        let d: f64 = v.parse().map_err(|_| format!("--tools: not a number: {v}"))?;
        if sheet.iter().any(|t| (t.diameter - d).abs() < 0.01) {
            continue;
        }
        let number = job.tool.number + sheet.len() as i32;
        sheet.push(ToolDef { number, kind: "ENDMILL".into(), diameter: d, corner_radius: 0.0, description: "FLAT END MILL".into(), tip_angle_deg: 118.0 });
    }
    let options = ModelOptions { envelope: !flag("--noenv"), defl_in: num("--defl")?, ..ModelOptions::default() };
    let vendor = opt("--vendor").map(PathBuf::from);
    let mut penalties = Penalties::default();
    if let Some(v) = num("--tchange")? {
        penalties.t_change = v;
    }
    if let Some(v) = num("--tlink")? {
        penalties.t_link = v;
    }
    let a = AssignJob { sheet, options, vendor: vendor.as_deref(), cell: num("--cell")?.unwrap_or(0.25), penalties };
    let clock = Instant::now();
    let out = assign_pockets(frep, job, rough, &a)?;
    for n in &out.notes {
        println!("note: {n}");
    }
    let unmodeled: Vec<String> = a.sheet.iter().filter(|t| !out.tools.iter().any(|s| s.number == t.number)).map(|t| format!("{:.3}", t.diameter)).collect();
    if !unmodeled.is_empty() {
        println!("note: no cut model for {} mm (models: 12.7, 6.35, 3.175, 1.5875); left out", unmodeled.join(", "));
    }
    for t in &out.tools {
        let env = if t.env.is_empty() { "off".to_string() } else { format!("slot ap <= {:.2} mm", t.env_ap(1.0)) };
        println!("  {:<24} D {:6.3} LOC {:5.1} hangout {:5.1} {:6.0} rpm  chip {:.4} in  envelope {env}", t.label, t.d(), t.model.loc, t.model.hangout, t.model.rpm, t.model.h_ceil / 25.4);
    }
    let p = &a.penalties;
    println!("times in s, cut + entries; tool change {:.0} s once per tool in the nest, link {:.1} s, helix {:.0} deg", p.t_change, p.t_link, p.ramp_deg);
    println!("{:<22} {:>8} {:>9} {:>8} | {:>8} {:>9} {:>10} | {:>8} {:>7}  rest by tool", "op", "z", "area mm2", "vol cm3", "rest", "cheapest", "cheapest+", "uncut", "bound");
    for l in &out.layers {
        let by: Vec<String> = out
            .tools
            .iter()
            .enumerate()
            .filter(|&(i, _)| l.rest.area[i] > 0.0)
            .map(|(i, t)| format!("T{:02} {:.0} mm2 {:.1} s", t.number, l.rest.area[i], l.rest.cut[i] + l.rest.entry_t[i]))
            .collect();
        println!(
            "{:<22} {:>8.3} {:>9.0} {:>8.1} | {:>8.1} {:>9.1} {:>10.1} | {:>8.1} {:>7.1}  {}",
            l.op,
            l.layer.z_floor,
            l.layer.mat_area(),
            l.layer.volume() / 1000.0,
            l.rest.local(),
            l.cheap.local(),
            l.cheap2.local(),
            l.rest.uncut_area,
            l.bound,
            by.join(", ")
        );
        for n in &l.layer.notes {
            println!("  note: {n}");
        }
    }
    for e in &out.empty {
        println!("{e:<22} no material left at its floor");
    }
    let seconds = clock.elapsed().as_secs_f64();
    let page = dir.join("assign.html");
    std::fs::write(&page, assign_report::html(part, &out, &a, job, seconds)).map_err(|e| format!("{}: {e}", page.display()))?;
    println!(
        "part (+ one change per tool used): rest {:.1} s, cheapest {:.1} s, cheapest with changes paid {:.1} s ({seconds:.1} s); wrote {}",
        out.nest.0,
        out.nest.1,
        out.nest.2,
        page.display()
    );
    Ok(ExitCode::SUCCESS)
}
