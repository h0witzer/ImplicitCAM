//! The finishing study's page: one self-contained HTML file (no network). The headline
//! per pocket ((a) constant allowance vs (b) allowance field, same tolerance, three feed
//! programs), then per pocket the wall colored by stock / deflection / feed for (a)
//! beside (b), the feed, deflection and stock along the wall, the time-against-stock
//! trade for both families, every run's numbers and the samples as CSV. The page
//! (finish_report.html) draws from the data written here.

use icam_finish::{Bind, FieldResult, PocketResult, Run, Settings};

use crate::finish::FinishOut;
use crate::json::{arr, list, num, obj, opt, text};

const TEMPLATE: &str = include_str!("finish_report.html");
const IN: f64 = 25.4;

fn ipm(v: &[f64]) -> String {
    arr(v.iter().map(|x| x * 60.0 / IN), 0)
}

fn run_obj(r: &Run) -> String {
    let share = |b: Bind| if r.cut_len > 0.0 { r.bind(b) / r.cut_len } else { 0.0 };
    let names = ["Chip", "Power", "Deflection", "Machine", "Kinematics", "OneFeed"];
    let bind: Vec<(&str, String)> = Bind::ALL.iter().zip(names).map(|(&b, n)| (n, num(share(b), 3))).collect();
    obj(&[
        ("label", text(&r.label)),
        ("time", num(r.time, 2)),
        ("corner", num(r.corner_time, 2)),
        ("F", opt(r.v_const * 60.0 / IN, 0)),
        ("feedMed", num(r.v_med * 60.0 / IN, 0)),
        ("deflMax", num(r.defl_max / IN, 5)),
        ("deflMean", num(r.defl_mean / IN, 5)),
        ("phiMed", num(r.phi_med.to_degrees(), 1)),
        ("phiMax", num(r.phi_max.to_degrees(), 1)),
        ("rub", num(r.rub_len, 0)),
        ("cut", num(r.cut_len, 0)),
        ("bind", obj(&bind)),
    ])
}

fn median(a: &[f64]) -> f64 {
    let mut b = a.to_vec();
    b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b.get(b.len() / 2).copied().unwrap_or(0.0)
}

fn spread(f: &FieldResult) -> String {
    let (lo, hi) = f.delta.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    arr([lo, median(&f.delta), hi], 3)
}

fn pocket(o: &FinishOut, r: &PocketResult, tol: f64) -> String {
    let pk = o.pocket.as_ref().expect("a studied pocket");
    let ap = pk.ap;
    let fold = |v: &[f64], init: f64, f: fn(f64, f64) -> f64| v.iter().fold(init, |a, &b| f(a, b));
    let bx0 = fold(&pk.x, f64::INFINITY, f64::min).min(fold(&pk.wx, f64::INFINITY, f64::min)) - pk.rf - 1.0;
    let bx1 = fold(&pk.x, f64::NEG_INFINITY, f64::max).max(fold(&pk.wx, f64::NEG_INFINITY, f64::max)) + pk.rf + 1.0;
    let by0 = fold(&pk.y, f64::INFINITY, f64::min).min(fold(&pk.wy, f64::INFINITY, f64::min)) - pk.rf - 1.0;
    let by1 = fold(&pk.y, f64::NEG_INFINITY, f64::max).max(fold(&pk.wy, f64::NEG_INFINITY, f64::max)) + pk.rf + 1.0;
    let outline = list(
        pk.outline
            .iter()
            .filter(|o| o.iter().any(|p| p.0 >= bx0 && p.0 <= bx1 && p.1 >= by0 && p.1 <= by1))
            .map(|o| arr(o.iter().flat_map(|p| [p.0, p.1]), 2)),
    );
    let loops = list(pk.loops.iter().map(|l| {
        obj(&[("start", l.start.to_string()), ("count", l.count.to_string()), ("len", num(l.length, 0)), ("island", l.island.to_string())])
    }));
    let mut s = Vec::with_capacity(pk.n());
    let mut before = 0.0;
    for l in &pk.loops {
        for i in l.start..l.start + l.count {
            s.push(pk.s[i] + before);
        }
        before += l.length;
    }
    let (fast, wall, eq) = (r.fast.sched(), r.wall.sched(), r.eq.sched());
    let rough = |f: &FieldResult| {
        let (dv, dt) = r.rougher(pk, f);
        format!("[{},{}]", num(dv, 0), num(dt, 2))
    };
    let per_tol = |v: &[f64]| arr(v.iter().map(|d| d / tol), 2);
    let mut notes: Vec<String> = r.notes.clone();
    notes.extend(r.fast.trace.iter().map(|t| format!("(b) fastest {t}")));
    notes.extend(r.wall.trace.iter().map(|t| format!("(b) wall feed {t}")));
    notes.extend(r.eq.trace.iter().map(|t| format!("(b) equal stock {t}")));
    let a_at_fast_max = r.defl_a_at_fast.iter().fold(0.0f64, |a, &b| a.max(b)) / tol;
    obj(&[
        ("name", text(&o.op)),
        ("finish", text(&pk.finish_name)),
        ("tool", text(&format!("T{:02}", pk.finish_tool.number))),
        ("model", text(&r.model.name)),
        ("ap", num(ap, 2)),
        ("zcut", num(pk.z_cut, 3)),
        ("wall", num(pk.loops.iter().map(|l| l.length).sum(), 0)),
        ("cornerLen", num(r.corner.iter().filter(|&&c| c).count() as f64 * pk.ds, 0)),
        ("compliance", num(r.model.compliance() * 1000.0, 3)),
        ("box", arr([bx0, by0, bx1, by1], 2)),
        ("outline", outline),
        ("loops", loops),
        ("s", arr(s, 2)),
        ("x", arr(pk.x.iter().copied(), 2)),
        ("y", arr(pk.y.iter().copied(), 2)),
        ("corner", list(r.corner.iter().map(|&c| (c as u8).to_string()))),
        ("thickA", arr(r.thick_a.iter().copied(), 3)),
        ("thickFast", arr(r.thick_fast.iter().copied(), 3)),
        ("thickWall", arr(r.thick_wall.iter().copied(), 3)),
        ("thickEq", arr(r.thick_eq.iter().copied(), 3)),
        ("deltaFast", arr(r.fast.delta.iter().copied(), 3)),
        ("deltaWall", arr(r.wall.delta.iter().copied(), 3)),
        ("feed", obj(&[("aSched", ipm(&r.a_sched.v)), ("fast", ipm(&fast.v)), ("aWall", ipm(&r.a_wall.v)), ("wall", ipm(&wall.v))])),
        (
            "defl",
            obj(&[
                ("aSched", per_tol(&r.a_sched.defl)),
                ("fast", per_tol(&fast.defl)),
                ("aWall", per_tol(&r.a_wall.defl)),
                ("wall", per_tol(&wall.defl)),
                ("aAtFast", per_tol(&r.defl_a_at_fast)),
            ]),
        ),
        ("runs", list([&r.a_one, &r.a_wall, &r.a_sched, &r.thin_one, &r.thin_wall, &r.thin_sched, fast, wall, eq].into_iter().map(run_obj))),
        (
            "t",
            obj(&[
                ("aSched", num(r.a_sched.time, 2)),
                ("fast", num(fast.time, 2)),
                ("eq", num(eq.time, 2)),
                ("thin", num(r.thin_sched.time, 2)),
                ("aWall", num(r.a_wall.time, 2)),
                ("wall", num(wall.time, 2)),
                ("thinWall", num(r.thin_wall.time, 2)),
                ("aOne", num(r.a_one.time, 2)),
                ("thinOne", num(r.thin_one.time, 2)),
                ("aSchedCorner", num(r.a_sched.corner_time, 2)),
                ("fastCorner", num(fast.corner_time, 2)),
                ("aWallCorner", num(r.a_wall.corner_time, 2)),
                ("wallCorner", num(wall.corner_time, 2)),
            ]),
        ),
        (
            "vol",
            obj(&[
                ("a", num(r.area_a * ap, 0)),
                ("thin", num(r.area_thin * ap, 0)),
                ("fast", num(r.fast.area * ap, 0)),
                ("eq", num(r.eq.area * ap, 0)),
                ("wall", num(r.wall.area * ap, 0)),
            ]),
        ),
        ("rough", obj(&[("fast", rough(&r.fast)), ("eq", rough(&r.eq)), ("wall", rough(&r.wall)), ("mrr", num(r.rough_mrr * 60.0 / IN.powi(3), 1))])),
        ("field", obj(&[("fast", spread(&r.fast)), ("wall", spread(&r.wall)), ("eq", spread(&r.eq))])),
        ("uniform", list(r.uniform.iter().map(|u| format!("[{},{},{}]", num(u.0 * ap, 0), num(u.1, 2), num(u.2, 3))))),
        ("budget", list(r.budget.iter().map(|u| format!("[{},{}]", num(u.0 * ap, 0), num(u.1, 2))))),
        ("aAtFastMax", num(a_at_fast_max, 2)),
        ("notes", list(notes.iter().map(|n| text(n)))),
        ("ms", num(o.ms, 0)),
        ("sims", r.sims.to_string()),
    ])
}

/// The page for the pockets of `part` studied with settings `s`; `rougher_d` is the
/// roughing tool's diameter (mm).
pub fn html(part: &str, outs: &[FinishOut], s: &Settings, rougher_d: f64, seconds: f64) -> String {
    let tol = s.tol_in * IN;
    let studied: Vec<(&FinishOut, &PocketResult)> = outs.iter().filter_map(|o| o.result.as_ref().map(|r| (o, r))).collect();
    let pockets = list(studied.iter().map(|(o, r)| pocket(o, r, tol)));
    let mut seen: Vec<&str> = Vec::new();
    let mut models = Vec::new();
    for (_, r) in &studied {
        let m = &r.model;
        if seen.contains(&m.name.as_str()) {
            continue;
        }
        seen.push(&m.name);
        models.push(obj(&[
            ("name", text(&m.name)),
            ("rpm", num(m.rpm, 0)),
            ("flutes", num(m.flutes, 0)),
            ("d", num(m.d / IN, 3)),
            ("loc", num(m.loc / IN, 2)),
            ("hangout", num(m.hangout / IN, 2)),
            ("helix", num(m.helix_deg, 0)),
            ("hceil", num(m.h_ceil / IN, 5)),
            ("hfloor", num(m.h_floor / IN, 5)),
            ("kw", num(m.spindle_w * m.eff / 1000.0, 1)),
            ("vmax", num(m.v_max * 60.0 / IN, 0)),
            ("tol", num(m.defl_max / IN, 5)),
            ("sources", text(&m.sources)),
        ]));
    }
    let rougher = match studied.first() {
        Some((_, r)) => obj(&[("name", text(&r.rough.name)), ("hangout", num(r.rough.hangout / IN, 2)), ("sources", text(&r.rough.sources)), ("d", num(rougher_d, 3))]),
        None => obj(&[("name", text("-")), ("hangout", "0".into()), ("sources", text("")), ("d", num(rougher_d, 3))]),
    };
    let skipped = list(outs.iter().filter(|o| o.result.is_none()).map(|o| obj(&[("op", text(&o.op)), ("why", text(o.skipped.as_deref().unwrap_or("")))])));
    let meta = obj(&[
        ("part", text(part)),
        ("seconds", num(seconds, 1)),
        ("tolIn", num(s.tol_in, 6)),
        ("accel", num(s.accel / 1000.0, 2)),
        ("allowance", num(s.allowance, 3)),
        ("dmin", num(s.d_min, 3)),
        ("dmax", num(s.d_max, 3)),
        ("slope", num(s.slope, 3)),
        ("ds", num(studied.first().and_then(|(o, _)| o.pocket.as_ref()).map_or(0.2, |p| p.ds), 2)),
        ("models", list(models)),
        ("rougher", rougher),
        ("skipped", skipped),
    ]);
    let data = obj(&[("meta", meta), ("pockets", pockets)]);
    TEMPLATE.replace("__DATA__", &data.replace("</", "<\\/"))
}
