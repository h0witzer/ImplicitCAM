//! The scoreboard page: one self-contained HTML file (no network). The objectives the first
//! roughing spike was scored on, with its result against its baseline as the bar, the
//! baseline here and the field roughing here; then the baseline's levels, our ops and the
//! per-pocket checks of both programs.

use std::fmt::Write as _;

use icam_motion::Metrics;

use crate::bench::{score, BenchOut};

const CSS: &str = r#"
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,0.10);--good:#006300;--crit:#d03b3b;color-scheme:light}
@media (prefers-color-scheme: dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,0.10);--good:#0ca30c;--crit:#e66767;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 8px}
.sub{color:var(--ink2);max-width:900px;margin:0 0 16px}.note{color:var(--ink2);font-size:13px;max-width:980px}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:13px;background:var(--surface);font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:6px 9px;text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;vertical-align:bottom}td:first-child,th:first-child{text-align:left}
td.l{text-align:left;white-space:normal}
.pass{color:var(--good);font-weight:600}.fail{color:var(--crit);font-weight:600}
.tally{font-size:20px;font-weight:600;margin:8px 0 4px}
ul{color:var(--ink2);font-size:14px}
"#;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn mrow(b: &mut String, m: &Metrics) {
    let _ = write!(
        b,
        "<td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td>",
        m.time(),
        m.time_cut,
        m.time_entry,
        m.time_rapid,
        m.chip_p5,
        m.chip_med,
        m.load_p95,
        m.load_max
    );
}

/// The page for a bench of `part` against `program`.
pub fn html(part: &str, program: &str, b: &BenchOut, material: Option<&str>, tool_d: f64, seconds: f64) -> String {
    let rows = score(b, material.is_some());
    let passed = rows.iter().filter(|r| r.pass).count();
    let mut s = format!("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Roughing Scoreboard</title><style>{CSS}</style></head><body><main>");
    s.push_str("<h1>Roughing scoreboard</h1>");
    let reference = match material {
        Some(m) => format!("both fed and judged by the physical limits of {}", esc(m)),
        None => format!(
            "both held to the baseline's own nominal cut: {:.0} ipm at {:.0} deg engagement (its median programmed feed and engagement)",
            b.v_prog * 60.0 / 25.4,
            b.phi_ref.to_degrees()
        ),
    };
    let _ = write!(
        s,
        "<p class=\"sub\">{} against the baseline program {}: the objectives the first roughing spike was scored on, with its result against its own baseline as the bar. Rough ops compared: {} ({:.3} mm tool); {reference}. {:.0} s.</p>",
        esc(part),
        esc(program),
        esc(&b.rough_ops.join(", ")),
        tool_d,
        seconds
    );
    let _ = write!(s, "<div class=\"tally\">{passed} of {} objectives met</div>", rows.len());
    s.push_str("<div class=\"scroll\"><table><thead><tr><th>objective</th><th>bar (the spike against its baseline)</th><th>baseline here</th><th>field roughing here</th><th>result</th></tr></thead><tbody>");
    for r in &rows {
        let _ = write!(
            s,
            "<tr><td class=\"l\">{}</td><td class=\"l\">{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td></tr>",
            esc(&r.objective),
            esc(&r.bar),
            esc(&r.base),
            esc(&r.ours),
            if r.pass { "pass" } else { "fail" },
            if r.pass { "&#10003; met" } else { "&#10007; not met" }
        );
    }
    s.push_str("</tbody></table></div>");
    if !b.notes.is_empty() {
        s.push_str("<ul>");
        for n in &b.notes {
            let _ = write!(s, "<li>{}</li>", esc(n));
        }
        s.push_str("</ul>");
    }
    let _ = write!(
        s,
        "<p class=\"note\">The baseline's rough cuts lie {:.1}% more than 0.5 mm inside the part (a frame check; near 0 when both share coordinates). Times are posted cutting, entry and rapid time at the router's acceleration; \"held\" re-feeds the baseline's own path under the same load limit as ours. Checks use each program's rough ops only, on the pockets found from the part's field, at our cut levels.</p>",
        100.0 * b.inside
    );

    s.push_str("<h2>Baseline levels</h2><div class=\"scroll\"><table><thead><tr><th>op</th><th>z mm</th><th>posted s</th><th>cut</th><th>entry</th><th>rapid</th><th>chip p5</th><th>chip med</th><th>load p95</th><th>load max</th><th>held s</th></tr></thead><tbody>");
    for l in &b.levels {
        let _ = write!(s, "<tr><td>{}</td><td>{:.3}</td>", esc(&l.op), l.z);
        mrow(&mut s, &l.posted);
        let _ = write!(s, "<td>{:.1}</td></tr>", l.refed.time());
    }
    s.push_str("</tbody></table></div>");

    s.push_str("<h2>Field roughing ops</h2><div class=\"scroll\"><table><thead><tr><th>op</th><th>z mm</th><th>posted s</th><th>cut</th><th>entry</th><th>rapid</th><th>chip p5</th><th>chip med</th><th>load p95</th><th>load max</th><th>status</th></tr></thead><tbody>");
    for o in &b.ours.ops {
        let _ = write!(s, "<tr><td>{}</td><td>{:.3}</td>", esc(&o.spec.name), o.spec.z_cut);
        match &o.posted {
            Some(m) => mrow(&mut s, m),
            None => s.push_str("<td colspan=\"7\">-</td>"),
        }
        let status = o.skipped.as_deref().map_or_else(|| if o.gouged() { "GOUGES".to_string() } else { "checked".to_string() }, |w| format!("skipped: {w}"));
        let _ = write!(s, "<td class=\"l\">{}</td></tr>", esc(&status));
    }
    s.push_str("</tbody></table></div>");

    s.push_str("<h2>Every pocket, both programs</h2><div class=\"scroll\"><table><thead><tr><th>pocket</th><th>ours: gouges</th><th>ours: uncut mm2</th><th>baseline: gouges</th><th>baseline: uncut mm2</th><th>baseline: worst gouge mm</th></tr></thead><tbody>");
    for g in &b.gates {
        let (og, ou) = g.ours.as_ref().map_or(("-".into(), "-".into()), |c| (c.gouges.to_string(), format!("{:.1}", c.residue_area)));
        let _ = write!(
            s,
            "<tr><td>{}</td><td>{og}</td><td>{ou}</td><td>{}</td><td>{:.1}</td><td>{:.3}</td></tr>",
            esc(&g.op),
            g.base.gouges,
            g.base.residue_area,
            g.base.max_gouge
        );
    }
    s.push_str("</tbody></table></div></main></body></html>");
    s
}
