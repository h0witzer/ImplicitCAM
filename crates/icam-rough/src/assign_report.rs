//! The tool assignment's page: one self-contained HTML file (no network). The finding,
//! headline tiles, a table per pocket, how far each assignment is from flipping, the
//! removal-rate curves the assignment reads, the tools and their limits, per pocket the
//! assignment maps (rest roughing, cheapest per point, where they differ, the margin to
//! the next tool, the channel width) with a per-tool table, and the assumptions ranked.
//! The maps are drawn in the page from run-length coded cell layers written here
//! (assign_report.html).

use std::fmt::Write as _;

use icam_assign::Plan;

use crate::assign::{AssignJob, AssignOut, LayerOut};
use crate::json::{list, num, obj, text};
use crate::run::Job;

const TEMPLATE: &str = include_str!("assign_report.html");
const IN_PER_MIN: f64 = 60.0 / (25.4 * 25.4 * 25.4);
const MARGIN_CUTS: [f64; 6] = [1.25, 1.5, 2.0, 3.0, 5.0, 10.0];
const WIDTH_CUTS: [f64; 6] = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0];

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn f(v: f64, d: usize) -> String {
    if v.is_finite() { format!("{v:.d$}") } else { "-".into() }
}

fn id(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase()
}

/// Material area (mm2) where rest roughing and `other` give a cell to different tools.
fn diff_area(x: &LayerOut, other: &Plan) -> f64 {
    let l = &x.layer;
    (0..l.g.n()).filter(|&k| l.mat[k] && x.rest.tool[k] != other.tool[k]).count() as f64 * l.cell_area()
}

fn median_h(x: &LayerOut) -> f64 {
    let l = &x.layer;
    let mut h: Vec<f32> = (0..l.g.n()).filter(|&k| l.mat[k]).map(|k| l.hgt[k]).collect();
    h.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    h.get(h.len() / 2).map_or(0.0, |&v| v as f64)
}

fn bucket(v: f64, cuts: &[f64]) -> u16 {
    cuts.iter().take_while(|&&c| v > c).count() as u16
}

/// Run-length code: value, count, value, count, ...
fn rle(v: impl Iterator<Item = u16>) -> String {
    let mut out: Vec<String> = Vec::new();
    let (mut cur, mut n) = (None, 0usize);
    for x in v {
        if Some(x) == cur {
            n += 1;
        } else {
            if let Some(c) = cur {
                out.push(format!("{c},{n}"));
            }
            cur = Some(x);
            n = 1;
        }
    }
    if let Some(c) = cur {
        out.push(format!("{c},{n}"));
    }
    format!("[{}]", out.join(","))
}

fn tool_code(t: i32) -> u16 {
    if t >= 0 { t as u16 } else { 4 }
}

/// Per tool pair (the bigger chosen by rest roughing, the smaller also reaching): the
/// area whose cost ratio is at most k, and the area both reach.
fn margins(out: &AssignOut, ks: &[f64]) -> Vec<(String, Vec<f64>, f64)> {
    let nt = out.tools.len();
    let mut o = Vec::new();
    for a in 0..nt {
        for s in a + 1..nt {
            let (mut areas, mut both) = (vec![0.0; ks.len()], 0.0);
            for x in &out.layers {
                let ca = x.layer.cell_area();
                for k in 0..x.layer.g.n() {
                    if x.rest.tool[k] != a as i32 || x.cost[s][k].is_infinite() {
                        continue;
                    }
                    both += ca;
                    let ratio = x.cost[s][k] / x.cost[a][k].max(1e-12);
                    for (q, &kk) in ks.iter().enumerate() {
                        if ratio <= kk {
                            areas[q] += ca;
                        }
                    }
                }
            }
            if both > 0.0 {
                o.push((format!("{} -> {}", out.tools[a].label, out.tools[s].label), areas, both));
            }
        }
    }
    o
}

fn break_even(out: &AssignOut) -> String {
    let (mut best, mut what) = (f64::INFINITY, "none".to_string());
    for x in &out.layers {
        for k in 0..x.layer.g.n() {
            let a = x.rest.tool[k];
            if a < 0 {
                continue;
            }
            for s in a as usize + 1..out.tools.len() {
                if x.cost[s][k].is_infinite() {
                    continue;
                }
                let r = x.cost[s][k] / x.cost[a as usize][k].max(1e-12);
                if r < best {
                    best = r;
                    what = format!("{} at {:.1}x the time of {} ({})", out.tools[s].label, r, out.tools[a as usize].label, x.op);
                }
            }
        }
    }
    what
}

fn tile(b: &mut String, label: &str, value: &str, sub: &str) {
    let _ = write!(b, "<div class=\"tile\"><div class=\"tl\">{}</div><div class=\"tv\">{}</div><div class=\"ts\">{}</div></div>", esc(label), esc(value), esc(sub));
}

fn swatch(var: &str) -> String {
    format!("<span class=\"sw\" style=\"background:var({var})\"></span>")
}

/// The page for the assignment `out` on `part`.
pub fn html(part: &str, out: &AssignOut, a: &AssignJob, job: &Job, seconds: f64) -> String {
    let p = &a.penalties;
    let tools = &out.tools;
    let nt = tools.len();
    let rest: Vec<&Plan> = out.layers.iter().map(|x| &x.rest).collect();
    let (rest_tot, cheap_tot, cheap2_tot) = out.nest;
    let area_tot: f64 = out.layers.iter().map(|x| x.layer.mat_area()).sum();
    let diff_cheap: f64 = out.layers.iter().map(|x| diff_area(x, &x.cheap)).sum();
    let diff_tot: f64 = out.layers.iter().map(|x| diff_area(x, &x.cheap2)).sum();
    let bound_tot: f64 = out.layers.iter().map(|x| x.bound).sum();
    let floor_tot: f64 = out.layers.iter().map(|x| x.model_floor).sum();
    let changes = |f: fn(&LayerOut) -> &Plan| (0..nt).filter(|&t| out.layers.iter().any(|x| f(x).area[t] > 0.0)).count();
    let local = |f: fn(&LayerOut) -> &Plan| out.layers.iter().map(|x| f(x).local()).sum::<f64>();
    let defl_in = tools.first().map_or(0.001, |t| t.model.defl_max / 25.4);
    let mut b = String::new();

    b.push_str("<h1>Tool assignment</h1>");
    let _ = write!(
        b,
        "<p class=\"sub\">{}, 6061-T6 aluminum data: every point of each pocket priced for every tool (reach, removal rate inside that tool's physical limits, channel width), then given to the cheapest tool, against rest roughing (big tool first) under the same model. The pockets are found from the part's field; the material at each is what the roughing program leaves before it. Assigned in {} s, offline.</p>",
        esc(part),
        f(seconds, 1)
    );

    // Finding.
    b.push_str("<section class=\"lead\"><h2>Finding</h2><p>");
    if out.layers.is_empty() {
        b.push_str("No pocket has material to assign.");
    } else {
        if diff_cheap < 0.01 * area_tot {
            let _ = write!(
                b,
                "On this part the cheapest tool at {} point is the biggest tool that reaches it: cheapest-per-point moves {} mm<sup>2</sup> of {} mm<sup>2</sup> away from rest roughing ({} mm<sup>2</sup> once every tool must pay for its entries and change). ",
                if diff_cheap == 0.0 { "every" } else { "almost every" },
                f(diff_cheap, 0),
                f(area_tot, 0),
                f(diff_tot, 0)
            );
        } else {
            let _ = write!(
                b,
                "Cheapest-per-point moves {} mm<sup>2</sup> of {} mm<sup>2</sup> away from rest roughing ({} mm<sup>2</sup> once every tool must pay for its entries and change). ",
                f(diff_cheap, 0),
                f(area_tot, 0),
                f(diff_tot, 0)
            );
        }
        if let Some(t0) = tools.first() {
            if !t0.env.is_empty() {
                let _ = write!(b, "The wide-cut envelope caps the {} tool's slots at {} mm per pass. ", esc(t0.label.split(' ').nth(1).unwrap_or("")), f(t0.env_ap(1.0), 2));
            }
        }
        if nt > 1 {
            let slower: Vec<String> =
                out.tabs.iter().skip(1).map(|t| format!("{} {}x", t.tool.label.split(' ').next().unwrap_or(""), f(out.tabs[0].open_rate / t.open_rate, 0))).collect();
            let _ = write!(
                b,
                "The smaller tools, at their hangouts and the same {} in deflection limit, are slower in open floor by {}. The closest any smaller tool comes to taking a point the big tool reaches: {}. ",
                f(defl_in, 3),
                slower.join(", "),
                esc(&break_even(out))
            );
            let small_t: f64 = out.layers.iter().map(|x| (1..nt).map(|t| x.rest.cut[t] + x.rest.entry_t[t]).sum::<f64>()).sum::<f64>()
                + p.t_change * (1..nt).filter(|&t| rest.iter().any(|r| r.area[t] > 0.0)).count() as f64;
            let small_a: f64 = out.layers.iter().map(|x| (1..nt).map(|t| x.rest.area[t]).sum::<f64>()).sum();
            if small_a > 0.0 {
                let _ = write!(
                    b,
                    "The time goes to reach: in rest roughing the smaller tools take {}% of the floor and {}% of the part's time ({} of {} s: cut, entries and their changes). ",
                    f(100.0 * small_a / area_tot, 1),
                    f(100.0 * small_t / rest_tot, 0),
                    f(small_t, 0),
                    f(rest_tot, 0)
                );
            }
            let dropped: Vec<&str> = (0..nt)
                .filter(|&t| rest.iter().any(|r| r.area[t] > 0.0) && !out.layers.iter().any(|x| x.cheap2.area[t] > 0.0))
                .map(|t| tools[t].label.split(' ').next().unwrap_or(""))
                .collect();
            if !dropped.is_empty() {
                let _ = write!(b, "Once each tool must pay for its entries and its change, {} drop out of the part: fewer tools, not a finer split.", dropped.join(", "));
            }
        }
    }
    b.push_str("</p></section>");

    // Tiles.
    b.push_str("<div class=\"tiles\">");
    tile(&mut b, &format!("Rest roughing, {} pockets", out.layers.len()), &format!("{} s", f(rest_tot, 0)), &format!("cut + entries + {} tool changes, once each for the part", changes(|x| &x.rest)));
    tile(&mut b, "Cheapest per point", &format!("{} s", f(cheap_tot, 0)), &format!("{} s once every tool must pay for its entries and change", f(cheap2_tot, 0)));
    tile(&mut b, "Area that changes hands", &format!("{} mm2", f(diff_tot, 0)), &format!("of {} mm2 of pocket floor", f(area_tot, 0)));
    tile(&mut b, "Absolute bound", &format!("{} s", f(bound_tot, 0)), &format!("volume at the best open-area rate of any tool; model floor {} s", f(floor_tot, 0)));
    b.push_str("</div>");

    // Per-pocket table.
    b.push_str("<h2>Per pocket</h2><div class=\"scroll\"><table><thead><tr><th>pocket</th><th>z mm</th><th>area mm2</th><th>volume cm3</th><th>bound s</th><th>model floor s</th><th>rest s</th><th>cheapest s</th><th>cheapest, changes paid s</th><th>changes hands mm2</th><th>no tool reaches mm2</th></tr></thead><tbody>");
    for x in &out.layers {
        let l = &x.layer;
        let _ = write!(
            b,
            "<tr><td><a href=\"#{}\">{}</a></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            id(&x.op),
            esc(&x.op),
            f(l.z_floor, 3),
            f(l.mat_area(), 0),
            f(l.volume() / 1000.0, 1),
            f(x.bound, 1),
            f(x.model_floor, 1),
            f(x.rest.local(), 1),
            f(x.cheap.local(), 1),
            f(x.cheap2.local(), 1),
            f(diff_area(x, &x.cheap2), 0),
            f(x.rest.uncut_area, 1)
        );
    }
    let _ = write!(
        b,
        "<tr class=\"sum\"><td>pockets</td><td></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
        f(area_tot, 0),
        f(out.layers.iter().map(|x| x.layer.volume()).sum::<f64>() / 1000.0, 1),
        f(bound_tot, 1),
        f(floor_tot, 1),
        f(local(|x| &x.rest), 1),
        f(local(|x| &x.cheap), 1),
        f(local(|x| &x.cheap2), 1),
        f(diff_tot, 0),
        f(out.layers.iter().map(|x| x.rest.uncut_area).sum(), 1)
    );
    let _ = write!(
        b,
        "<tr><td>tool changes</td><td colspan=\"5\"></td><td>{}</td><td>{}</td><td>{}</td><td colspan=\"2\"></td></tr><tr class=\"sum\"><td>part total</td><td colspan=\"5\"></td><td>{}</td><td>{}</td><td>{}</td><td colspan=\"2\"></td></tr></tbody></table></div>",
        f(p.t_change * changes(|x| &x.rest) as f64, 0),
        f(p.t_change * changes(|x| &x.cheap) as f64, 0),
        f(p.t_change * changes(|x| &x.cheap2) as f64, 0),
        f(rest_tot, 1),
        f(cheap_tot, 1),
        f(cheap2_tot, 1)
    );
    let _ = write!(
        b,
        "<p class=\"note\">Pocket rows are cut + entries ({} s per link, {} deg helix entries); tool changes are {} s once per tool for the part, as a program that groups each tool's ops would run it (all assumed). \"Changes hands\" compares rest roughing with the changes-paid assignment. Bound = the pocket's volume at the best open-area removal rate any tool reaches (ignores reach and channels). Model floor = every point at its cheapest tool, no penalties.</p>",
        f(p.t_link, 1),
        f(p.ramp_deg, 0),
        f(p.t_change, 0)
    );

    // Break-even margins.
    let ks = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0];
    let ms = margins(out, &ks);
    if !ms.is_empty() {
        b.push_str("<h2>How far from flipping</h2><p class=\"note\">Area where a smaller tool that also reaches the point would become the cheaper tool if it were <i>k</i> times faster than the model says (all pockets). A large <i>k</i> means the assignment is not close.</p><div class=\"scroll\"><table><thead><tr><th>tool pair</th>");
        for k in ks {
            let _ = write!(b, "<th>k = {}</th>", if k == 1.5 { "1.5".into() } else { f(k, 0) });
        }
        b.push_str("<th>reached by both, mm2</th></tr></thead><tbody>");
        for (name, areas, both) in &ms {
            let _ = write!(b, "<tr><td>{}</td>", esc(name));
            for a in areas {
                let _ = write!(b, "<td>{}</td>", f(*a, 0));
            }
            let _ = write!(b, "<td>{}</td></tr>", f(*both, 0));
        }
        b.push_str("</tbody></table></div>");
    }

    // Rate curves.
    let mut hs: Vec<(f64, Vec<String>)> = Vec::new();
    for x in &out.layers {
        let h = (median_h(x) * 100.0).round() / 100.0;
        match hs.iter_mut().find(|q| q.0 == h) {
            Some(q) => q.1.push(x.op.clone()),
            None => hs.push((h, vec![x.op.clone()])),
        }
    }
    hs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let ws: Vec<f64> = {
        let (mut v, mut w) = (Vec::new(), 1.6f64);
        while w <= 100.0 {
            v.push((w * 1000.0).round() / 1000.0);
            w *= 1.08;
        }
        v
    };
    if !hs.is_empty() {
        b.push_str("<h2>What the assignment reads: removal rate by channel width</h2><p class=\"note\">Steady removal each tool sustains while clearing a channel of width W through a column of height H, inside its limits (open a lane by slot or loops, widen by side passes; best axial split). Log scales. Pick the column height of a pocket.</p><div class=\"filters\" id=\"hfilter\" role=\"group\" aria-label=\"column height\">");
        for (i, (h, names)) in hs.iter().enumerate() {
            let _ = write!(b, "<button type=\"button\" data-h=\"{i}\" aria-pressed=\"{}\">H {} mm ({})</button>", i == hs.len() - 1, f(*h, 2), esc(&names.join(", ")));
        }
        b.push_str("</div><div class=\"chart\" id=\"curve\"></div><div class=\"legend\">");
        for (i, t) in tools.iter().enumerate().take(4) {
            let _ = write!(b, "<span>{}{}</span>", swatch(&format!("--s{}", i + 1)), esc(&t.label));
        }
        b.push_str("</div><details><summary>Table view</summary><div class=\"scroll\"><table><thead><tr><th>H mm</th><th>tool</th>");
        let tw = [2.0, 3.0, 4.0, 6.5, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 40.0, 64.0];
        for w in tw {
            let _ = write!(b, "<th>W {}</th>", f(w, 1));
        }
        b.push_str("</tr></thead><tbody>");
        for (h, _) in &hs {
            for tab in &out.tabs {
                let _ = write!(b, "<tr><td>{}</td><td>{}</td>", f(*h, 2), esc(&tab.tool.label));
                for w in tw {
                    if w < tab.tool.d() {
                        b.push_str("<td>-</td>");
                    } else {
                        let _ = write!(b, "<td>{:.3}</td>", h / tab.at(w, *h) * IN_PER_MIN);
                    }
                }
                b.push_str("</tr>");
            }
        }
        b.push_str("</tbody></table></div><p class=\"note\">in3/min; '-' = narrower than the tool.</p></details>");
    }

    // Tools.
    b.push_str("<h2>Tools and limits</h2><div class=\"scroll\"><table><thead><tr><th>tool</th><th>D mm</th><th>flutes</th><th>LOC mm</th><th>hangout mm</th><th>rpm</th><th>chip ceiling in</th><th>deflection limit in</th><th>slot envelope mm</th><th>best open rate in3/min</th><th>sources</th></tr></thead><tbody>");
    for (i, t) in tools.iter().enumerate() {
        let m = &t.model;
        let _ = write!(
            b,
            "<tr><td>{}{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td class=\"src\">{}</td></tr>",
            if i < 4 { swatch(&format!("--s{}", i + 1)) } else { String::new() },
            esc(&t.label),
            f(t.d(), 3),
            f(m.flutes, 0),
            f(m.loc, 1),
            f(m.hangout, 1),
            f(m.rpm, 0),
            f(m.h_ceil / 25.4, 4),
            f(m.defl_max / 25.4, 3),
            if t.env.is_empty() { "off".into() } else { f(t.env_ap(1.0), 2) },
            out.tabs[i].open_rate * IN_PER_MIN,
            esc(&t.sources)
        );
    }
    b.push_str("</tbody></table></div><details><summary>Wide-cut envelope per tool (ae/D : ap/D)</summary><pre>");
    for t in tools {
        let e: Vec<String> = t.env.iter().map(|(ae, ap)| format!("{ae:.2}:{ap:.2}")).collect();
        let _ = writeln!(b, "{}: {}", esc(&t.label), if e.is_empty() { "off".to_string() } else { e.join("  ") });
    }
    let from = a.vendor.map_or("no vendor table".to_string(), |v| v.display().to_string());
    let _ = write!(b, "</pre><p class=\"note\">From {}.</p></details>", esc(&from));

    // Pockets.
    b.push_str("<h2>Pockets</h2><div class=\"legend\">");
    for (i, t) in tools.iter().enumerate().take(4) {
        let _ = write!(b, "<span>{}{}</span>", swatch(&format!("--s{}", i + 1)), esc(&t.label));
    }
    let _ = write!(
        b,
        "<span>{}no tool reaches</span><span>{}pocket, already at the floor</span><span>{}part above the floor</span><span>{}outside the pocket</span></div>",
        swatch("--uncut"),
        swatch("--clear"),
        swatch("--part"),
        swatch("--out")
    );
    b.push_str("<p class=\"note\">Maps: one pixel per grid cell, X right, Y up; hover a cell for its numbers. \"Where they differ\": full color = the changes-paid tool where it differs from rest roughing, pale = the two agree. \"Margin\": how many times cheaper the chosen tool is than the next reaching tool (darkest = close to flipping). \"Channel width\": the width the roughing tool works in, over its diameter (darkest = narrowest).</p>");
    let op_idx = tools.iter().position(|t| t.number == job.tool.number);
    let mut pockets = Vec::new();
    for (pi, x) in out.layers.iter().enumerate() {
        let l = &x.layer;
        let _ = write!(
            b,
            "<section class=\"pocket\" id=\"{}\"><h3>{} <span class=\"muted\">z {} mm, {} mm2, {} cm3, median height {} mm</span></h3>",
            id(&x.op),
            esc(&x.op),
            f(l.z_floor, 3),
            f(l.mat_area(), 0),
            f(l.volume() / 1000.0, 1),
            f(median_h(x), 2)
        );
        if !l.notes.is_empty() {
            b.push_str("<ul class=\"notes\">");
            for n in &l.notes {
                let _ = write!(b, "<li>{}</li>", esc(n));
            }
            b.push_str("</ul>");
        }
        let mut maps = vec![("rest", "Rest roughing".to_string()), ("cheap", "Cheapest per point".to_string())];
        if x.cheap.tool != x.cheap2.tool {
            maps.push(("paid", "Cheapest, changes paid".to_string()));
        }
        maps.push(("differ", "Where they differ".to_string()));
        maps.push(("margin", "Margin to the next tool".to_string()));
        maps.push(("width", format!("Channel width, T{:02}", job.tool.number)));
        b.push_str("<div class=\"maps\">");
        for (kind, title) in &maps {
            let _ = write!(
                b,
                "<figure><canvas class=\"map\" data-p=\"{pi}\" data-kind=\"{kind}\" width=\"{}\" height=\"{}\" role=\"img\" aria-label=\"{} map of {}\"></canvas><figcaption>{}</figcaption></figure>",
                l.g.nx,
                l.g.ny,
                esc(title),
                esc(&x.op),
                esc(title)
            );
        }
        b.push_str("</div><p class=\"note\">Margin: ");
        let ml = ["at most 1.25x", "1.25-1.5x", "1.5-2x", "2-3x", "3-5x", "5-10x", "over 10x"];
        for (i, m) in ml.iter().enumerate() {
            let _ = write!(b, "{}{m} ", swatch(&format!("--q{i}")));
        }
        let _ = write!(b, "{}one tool reaches. Channel width W/D: ", swatch("--one"));
        let wl = ["at most 1.1", "1.1-1.25", "1.25-1.5", "1.5-2", "2-3", "3-5", "over 5"];
        for (i, w) in wl.iter().enumerate() {
            let _ = write!(b, "{}{w} ", swatch(&format!("--q{i}")));
        }
        let _ = write!(b, "{}the roughing tool does not reach.</p>", swatch("--one"));
        b.push_str("<div class=\"scroll\"><table><thead><tr><th>strategy</th><th>cut + entries s</th><th>cut s</th><th>entries s</th><th>pocket cut alone s</th>");
        for t in tools {
            let _ = write!(b, "<th>{}<br><span class=\"muted\">mm2 / cm3 / cut s / helix+link</span></th>", esc(&t.label));
        }
        b.push_str("</tr></thead><tbody>");
        for pl in [&x.rest, &x.cheap, &x.cheap2] {
            let _ = write!(b, "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>", esc(&pl.name), f(pl.local(), 1), f(pl.cut_total, 1), f(pl.entry_total, 1), f(pl.total(), 1));
            for t in 0..nt {
                if pl.area[t] > 0.0 {
                    let _ = write!(b, "<td>{} / {} / {} / {}+{}</td>", f(pl.area[t], 1), f(pl.vol[t] / 1000.0, 2), f(pl.cut[t], 1), pl.helix[t], pl.links[t]);
                } else {
                    b.push_str("<td>-</td>");
                }
            }
            b.push_str("</tr>");
        }
        b.push_str("</tbody></table></div></section>");

        // The cell layers the maps draw.
        let n = l.g.n();
        let base = (0..n).map(|k| if l.mat[k] { 0 } else if l.pocket[k] { 1 } else if l.clear[k] < 0.0 { 2 } else { 3 });
        let margin = (0..n).map(|k| {
            let q = x.cheap2.tool[k];
            if !l.mat[k] || q < 0 {
                return 0;
            }
            let best = (0..nt).filter(|&t| t != q as usize).map(|t| x.cost[t][k]).fold(f64::INFINITY, f64::min);
            if best.is_infinite() { 7 } else { bucket(best / x.cost[q as usize][k].max(1e-12), &MARGIN_CUTS) }
        });
        let width = (0..n).map(|k| match op_idx {
            Some(o) if l.mat[k] && l.reach[o][k] => bucket(l.w[o][k] as f64 / tools[o].d(), &WIDTH_CUTS),
            _ => 7,
        });
        let layers = obj(&[
            ("base", rle(base)),
            ("rest", rle(x.rest.tool.iter().map(|&t| tool_code(t)))),
            ("cheap", rle(x.cheap.tool.iter().map(|&t| tool_code(t)))),
            ("paid", rle(x.cheap2.tool.iter().map(|&t| tool_code(t)))),
            ("margin", rle(margin)),
            ("width", rle(width)),
            ("hgt", rle(l.hgt.iter().map(|&h| (h as f64 * 10.0).round().clamp(0.0, 65535.0) as u16))),
        ]);
        pockets.push(obj(&[("nx", l.g.nx.to_string()), ("ny", l.g.ny.to_string()), ("x0", num(l.g.x0, 3)), ("y0", num(l.g.y0, 3)), ("h", num(l.g.h, 3)), ("layers", layers)]));
    }
    if !out.empty.is_empty() {
        let _ = write!(b, "<h3>Ops with no material to assign</h3><p class=\"note\">{}: the ops before them already cleared their floor.</p>", esc(&out.empty.join(", ")));
    }

    // Assumptions.
    let _ = write!(
        b,
        "<h2>Assumptions, ranked by how much they move the answer</h2><ol class=\"assume\">\
<li><b>Deflection limit {} in for every tool</b> (the roughing rule of thumb; <code>--defl</code>). It decides everything below the 1/2 in tool: at their hangouts the small tools are many times as compliant, so the same limit allows them only a few newtons. Pinned by: a deflection-vs-finish test cut per tool, or the wall-stock variation the finish pass accepts.</li>\
<li><b>Small-tool hangouts</b> (1/4 in 1.5 in, 1/8 in 1.2 in, 1/16 in 0.625 in, from the proving tool sheet), the shank modeled at the cutting diameter. Compliance goes with hangout cubed. Pinned by: the actual holders and stickouts in the router's tool library.</li>\
<li><b>Wide-cut envelope</b> ({}): the deepest published ap per radial engagement, 1-flute vendor tables applied to every tool. It is the penalty on wide cuts that makes slots cost several axial passes. Pinned by: slotting and side-milling depth trials in 6061.</li>\
<li><b>Chip ceilings by tool</b> (router bands 0.007, 0.006, 0.004 in; the HSS 1/16 in at 0.0012 in, derated, assumed). They bind wherever deflection does not (slots at the envelope depth, shallow layers). Pinned by: the vendors' charts for the exact tools, or bench cuts.</li>\
<li><b>Spindle speeds</b>: the 1/2 in at 18,000 rpm, the small tools at the machine's 24,000. At the chip ceiling, feed is fz x flutes x rpm, so this sets the 1/2 in vs 1/4 in ratio on shallow layers.</li>\
<li><b>Tool change {} s, link {} s, {} deg helix entries</b> (assumed). Only the changes-paid strategy reads them. Pinned by: a timed tool change and a posted link.</li>\
<li><b>Channel model</b>: a lane per unit length (a long channel), loops at the internal-circle engagement, the centripetal limit at {} m/s2, no path clearing floor faster than D x feed. It overstates the lane's share in short rooms by at most D/W.</li>\
<li><b>Reach</b>: cutting-diameter shank, holder 1 mm above the stock top, rest thinner than 0.1 mm beyond the allowance left to the finish pass.</li>\
<li><b>Tool geometry not on a sheet</b>: flute lengths of the 1/4 in (0.75 in) and 1/8 in (0.5 in) tools, helix angles, core ratios. Second order next to hangout and the deflection limit.</li></ol>",
        f(defl_in, 3),
        if a.options.envelope { "on" } else { "off" },
        f(p.t_change, 0),
        f(p.t_link, 1),
        f(p.ramp_deg, 0),
        f(job.accel / 1000.0, 1)
    );
    let _ = write!(
        b,
        "<p class=\"note\">Inputs: {}; allowance {} mm; grid {} mm; acceleration {} m/s2. The model: crates/icam-assign.</p>",
        esc(part),
        f(job.allowance, 3),
        f(a.cell, 2),
        f(job.accel / 1000.0, 1)
    );

    // Data for the chart and the maps.
    let curve = obj(&[
        ("w", list(ws.iter().map(|&w| num(w, 3)))),
        ("tools", list(tools.iter().enumerate().map(|(i, t)| obj(&[("label", text(&t.label)), ("slot", (i.min(3) + 1).to_string()), ("d", num(t.d(), 4))])))),
        (
            "h",
            list(hs.iter().map(|(h, _)| {
                let rates = list(out.tabs.iter().map(|tab| {
                    list(ws.iter().map(|&w| {
                        let v = h / tab.at(w, *h) * IN_PER_MIN;
                        if w < tab.tool.d() - 1e-9 || !v.is_finite() { "null".to_string() } else { format!("{v:.3e}") }
                    }))
                }));
                obj(&[("h", num(*h, 2)), ("rates", rates)])
            })),
        ),
    ]);
    let data = obj(&[("tools", list(tools.iter().map(|t| text(&t.label)))), ("curve", curve), ("pockets", list(pockets))]);
    TEMPLATE.replace("__BODY__", &b).replace("__DATA__", &data.replace("</", "<\\/"))
}
