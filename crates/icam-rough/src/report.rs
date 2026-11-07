//! One self-contained HTML page (no network): the job, a summary row per op, the load
//! model's assumptions, and per op its path over the part's outline at the slab, colored
//! by the measured engagement angle (blue 0 deg .. red 120 deg and up; entries gray,
//! rapids dashed).

use std::fmt::Write as _;

use icam_post::Tag;

use crate::run::{Job, JobOut, OpOut, MAX_UNCUT};

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn num(v: f64, d: usize) -> String {
    if v.is_finite() { format!("{v:.d$}") } else { "-".into() }
}

/// Engagement color: hue 240 (blue) at 0 deg down to 0 (red) at 120 deg and up, in 10 deg steps.
fn hue(deg: f64) -> i32 {
    let b = (deg / 10.0).floor().clamp(0.0, 12.0);
    (240.0 - b * 20.0) as i32
}

fn status(o: &OpOut) -> (String, &'static str) {
    if let Some(why) = &o.skipped {
        return (format!("skipped: {why}"), "skip");
    }
    match &o.check {
        Some(c) if c.gouges > 0 => (format!("GOUGES: {} points, worst {:.3} mm", c.gouges, c.max_gouge), "bad"),
        Some(c) if c.residue_area > MAX_UNCUT => (format!("{:.0} mm2 left uncut", c.residue_area), "warn"),
        Some(_) => ("checked".into(), "ok"),
        None => ("planned".into(), "ok"),
    }
}

fn drawing(o: &OpOut) -> String {
    let (x0, y0, h, nx, ny) = o.grid;
    let (w, ht) = ((nx.max(2) - 1) as f64 * h, (ny.max(2) - 1) as f64 * h);
    let scale = 640.0 / w.max(ht).max(1.0);
    let (pw, ph) = (w * scale, ht * scale);
    let px = |x: f64| (x - x0) * scale;
    let py = |y: f64| ph - (y - y0) * scale;
    let mut s = format!("<svg viewBox=\"-4 -4 {:.0} {:.0}\" width=\"{:.0}\" height=\"{:.0}\">", pw + 8.0, ph + 8.0, pw + 8.0, ph + 8.0);
    for lp in &o.outline {
        let mut d = String::new();
        for (k, p) in lp.iter().enumerate() {
            let _ = write!(d, "{}{:.1} {:.1} ", if k == 0 { "M" } else { "L" }, px(p.0), py(p.1));
        }
        let _ = write!(s, "<path d=\"{d}Z\" class=\"part\"/>");
    }
    // Runs of samples in one color class become one path.
    let class = |i: usize| -> String {
        let q = &o.ours[i];
        if q.tag == Tag::Rapid {
            "rapid".into()
        } else if q.phi.is_nan() {
            "entry".into()
        } else {
            format!("h{}", hue(q.phi.to_degrees()))
        }
    };
    let mut i = 1;
    while i < o.ours.len() {
        let c = class(i);
        let mut d = format!("M{:.1} {:.1}", px(o.ours[i - 1].x), py(o.ours[i - 1].y));
        while i < o.ours.len() && class(i) == c {
            let _ = write!(d, " L{:.1} {:.1}", px(o.ours[i].x), py(o.ours[i].y));
            i += 1;
        }
        let style = match c.as_str() {
            "rapid" => "class=\"rapid\"".to_string(),
            "entry" => "class=\"entry\"".to_string(),
            h => format!("class=\"cut\" stroke=\"hsl({},80%,45%)\"", &h[1..]),
        };
        let _ = write!(s, "<path d=\"{d}\" {style}/>");
    }
    s.push_str("</svg>");
    s
}

/// The report page.
pub fn html(job: &Job, part_name: &str, out: &JobOut, seconds: f64) -> String {
    let t = &job.tool;
    let mut h = String::new();
    let _ = write!(
        h,
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <title>Field roughing report</title><style>\
         body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1d1d1f;background:#fff}}\
         h1{{font-size:20px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 6px}}p.sub{{color:#555;margin:0 0 16px}}\
         table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}}\
         th:first-child,td:first-child,td.l{{text-align:left}}tr.bad td{{background:#fde8e8}}tr.warn td{{background:#fff6dd}}tr.skip td{{color:#777}}\
         ul{{margin:4px 0 12px 18px;padding:0}}li{{margin:2px 0}}svg{{max-width:100%;height:auto;background:#fafafa;border:1px solid #e3e3e3}}\
         .part{{fill:#d9d9d9;stroke:#666;stroke-width:1}}.cut{{fill:none;stroke-width:1.2}}.entry{{fill:none;stroke:#999;stroke-width:1}}\
         .rapid{{fill:none;stroke:#bbb;stroke-width:.8;stroke-dasharray:3 3}}.warn{{color:#a15c00}}\
         @media (prefers-color-scheme:dark){{body{{color:#e8e8e8;background:#161618}}th,td{{border-color:#333}}tr.bad td{{background:#4a1d1d}}\
         tr.warn td{{background:#3d3214}}p.sub{{color:#aaa}}svg{{background:#1f1f22;border-color:#333}}.part{{fill:#3a3a3e;stroke:#999}}}}\
         </style></head><body>"
    );
    let _ = write!(
        h,
        "<h1>Field roughing: {}</h1><p class=\"sub\">Plan <b>{}</b>, tool {} D{:.3} R{:.3}, allowance {:.3} mm{}, stock z {:.3} .. {:.3}; {} ops, planned and checked in {:.1} s. Research output: nothing here should drive a machine.</p>",
        esc(part_name),
        job.plan.name(),
        t.number,
        t.diameter,
        t.corner_radius,
        job.allowance,
        job.material.as_ref().map_or(String::new(), |m| format!(", material {m}")),
        out.stock.0.z,
        out.stock.1.z,
        out.ops.len(),
        seconds
    );
    if !out.warnings.is_empty() {
        h.push_str("<ul class=\"warn\">");
        for w in &out.warnings {
            let _ = write!(h, "<li>{}</li>", esc(w));
        }
        h.push_str("</ul>");
    }
    h.push_str(
        "<table><thead><tr><th>op</th><th>z cut</th><th>ap mm</th><th>planned s</th><th>posted s</th><th>cut mm</th><th>entry mm</th>\
         <th>phi med / p95 / max</th><th>load p95 / max</th><th>gouges</th><th>uncut mm2</th><th>wall stock mm</th><th>status</th></tr></thead><tbody>",
    );
    for o in &out.ops {
        let (st, class) = status(o);
        let m = o.posted.as_ref().or(o.planned.as_ref());
        let _ = write!(
            h,
            "<tr class=\"{class}\"><td>{}</td><td>{:.3}</td><td>{:.2}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"l\">{}</td></tr>",
            esc(&o.spec.name),
            o.spec.z_cut,
            o.ap,
            o.planned.as_ref().map_or("-".into(), |m| num(m.time(), 1)),
            o.posted.as_ref().map_or("-".into(), |m| num(m.time(), 1)),
            m.map_or("-".into(), |m| num(m.cut_length, 0)),
            m.map_or("-".into(), |m| num(m.entry_length, 0)),
            m.map_or("-".into(), |m| format!("{:.0} / {:.0} / {:.0}", m.phi_med_deg, m.phi_p95_deg, m.phi_max_deg)),
            m.map_or("-".into(), |m| format!("{} / {}", num(m.load_p95, 2), num(m.load_max, 2))),
            o.check.as_ref().map_or("-".into(), |c| c.gouges.to_string()),
            o.check.as_ref().map_or("-".into(), |c| num(c.residue_area, 1)),
            o.check.as_ref().map_or("-".into(), |c| num(c.wall_median, 3)),
            esc(&st)
        );
    }
    h.push_str("</tbody></table>");
    h.push_str(
        "<h2>How it is measured</h2><ul>\
         <li>Ops are found from the field: every floor of the part below the stock top is a slab, and every region at a slab where the tool center fits and that is closed by the part is a pocket. Each op is cut on the stock the program before it leaves.</li>\
         <li>Engagement: at every 0.25 mm on the slab, the angle of the cutter circle that still touches material, against the stock as it stands just before that step.</li>\
         <li>Load: chip thickness h = fz s(phi) and removal rate ~ ae x feed against a reference cut (or, with a material, the physical chip, power and deflection ceilings). Feed is the fastest inside every limit, then limited by curvature (v &lt;= sqrt(a R)) and acceleration.</li>\
         <li>Posted: the program is read back and each op measured again at the feeds it carries. Checked: the exact swept-tool stock after the op against points on the part's walls and floor (gouge past 0.0254 mm), and the reachable pocket area still uncut.</li></ul>",
    );
    for o in &out.ops {
        let _ = write!(h, "<h2>{}</h2>", esc(&o.spec.name));
        let mut notes = o.notes.clone();
        if let Some(why) = &o.skipped {
            notes.insert(0, format!("skipped: {why}"));
        }
        if let Some(p) = &o.path {
            notes.insert(
                0,
                format!(
                    "{} revolutions, {} seeds, {} islands, {} arcs, {} links ({} safety, {} kept down); planned in {:.0} ms",
                    p.revolutions().max(p.revolutions2),
                    p.seeds,
                    p.islands,
                    p.arcs,
                    p.links,
                    p.safety_links,
                    p.kept_down,
                    o.ms
                ),
            );
        }
        h.push_str("<ul>");
        for n in &notes {
            let _ = write!(h, "<li>{}</li>", esc(n));
        }
        h.push_str("</ul>");
        if !o.ours.is_empty() {
            h.push_str(&drawing(o));
        }
    }
    h.push_str("</body></html>\n");
    h
}
