//! Aluminum cut data read from a vendor cut-data table: a CSV with the columns of
//! `VendorRow`, one row per published cut, passed with `--vendor`.
//!
//! What is taken from it, per tool diameter class: from Datron's HSC aluminum tables
//! (datron-hsc-aluminum, all 1-flute carbide), the deepest axial depth published at each
//! radial engagement, as multiples of D. That is the chip-evacuation / wide-cut envelope:
//! a full slot is published only to about 0.5 D, a 3-7% D side cut to 3-5 D. NOT taken
//! from it: any machine limit (Datron's spindles and feeds are not the router's).

use std::path::Path;

#[derive(Clone, Debug)]
pub struct VendorRow {
    pub source: String,
    pub flutes: i32,
    pub d_in: f64,
    pub fz_in: f64,
    pub ae_d: f64,
    pub ap_d: f64,
    pub rpm: f64,
    pub feed_ipm: f64,
    pub spindle_kw: f64,
    pub reference: String,
}

/// Minimal RFC 4180 split (quoted fields may hold commas and doubled quotes).
fn split(line: &str) -> Vec<String> {
    let (mut f, mut sb, mut q) = (Vec::new(), String::new(), false);
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        if q {
            if c == '"' {
                if it.peek() == Some(&'"') {
                    sb.push('"');
                    it.next();
                } else {
                    q = false;
                }
            } else {
                sb.push(c);
            }
        } else if c == '"' {
            q = true;
        } else if c == ',' {
            f.push(std::mem::take(&mut sb));
        } else {
            sb.push(c);
        }
    }
    f.push(sb);
    f
}

fn number(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// The number in the first `<prefix>[digits and dots]<suffix>` in s.
fn after(s: &str, prefix: &str, suffix: &str) -> Option<f64> {
    let mut from = 0;
    while let Some(p) = s[from..].find(prefix) {
        let start = from + p + prefix.len();
        let run = s[start..].bytes().take_while(|b| b.is_ascii_digit() || *b == b'.').count();
        if run > 0 && s[start + run..].starts_with(suffix) {
            return s[start..start + run].parse().ok();
        }
        from = from + p + 1;
    }
    None
}

/// "(0.04 x D)" -> 0.04; -1 when there is none.
fn xd(s: &str) -> f64 {
    after(s, "(", " x D)").unwrap_or(-1.0)
}

/// Datron aluminum rows with parseable ae and ap notes (none when the file is missing).
pub fn datron(path: &Path) -> Vec<VendorRow> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut rows = Vec::new();
    let mut head: Option<Vec<String>> = None;
    for line in text.lines() {
        let f = split(line);
        let Some(h) = &head else {
            head = Some(f);
            continue;
        };
        let get = |name: &str| h.iter().position(|c| c == name).and_then(|i| f.get(i)).map_or("", |s| s.as_str());
        if get("source_id") != "datron-hsc-aluminum" {
            continue;
        }
        let Some(d) = number(get("diameter_in")) else { continue };
        let Some(fz) = number(get("chipload_hi_ipt")) else { continue };
        let fl = get("flutes").trim().parse::<i32>().unwrap_or(0);
        let rpm = number(get("rpm_hi")).unwrap_or(0.0);
        let ipm = number(get("feed_hi_ipm")).unwrap_or(0.0);
        let (rad, ax, notes) = (get("radial_doc_note"), get("axial_doc_note"), get("notes"));
        let ae_d = if rad.contains("(slot)") { 1.0 } else { xd(rad) };
        let ap_d = xd(ax);
        if ae_d <= 0.0 || ap_d <= 0.0 {
            continue;
        }
        let spindle_kw = after(notes, "spindle ", " kW").unwrap_or(0.0);
        rows.push(VendorRow {
            source: "datron-hsc-aluminum".into(),
            flutes: fl,
            d_in: d,
            fz_in: fz,
            ae_d,
            ap_d,
            rpm,
            feed_ipm: ipm,
            spindle_kw,
            reference: get("source_ref").into(),
        });
    }
    rows
}

/// Rows whose diameter is within 15% of d (inches).
pub fn near(rows: &[VendorRow], d_in: f64) -> Vec<&VendorRow> {
    rows.iter().filter(|r| (r.d_in - d_in).abs() <= 0.15 * d_in).collect()
}

/// `v` rounded to two decimals.
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Envelope: the deepest published ap/D at each radial engagement (ae/D to 0.01), made
/// monotone (a narrower cut is allowed at least as deep as a wider one).
pub fn envelope(rows: &[VendorRow], d_in: f64) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for r in near(rows, d_in) {
        let key = round2(r.ae_d);
        match pts.iter_mut().find(|p| p.0 == key) {
            Some(p) => {
                if r.ap_d > p.1 {
                    p.1 = r.ap_d;
                }
            }
            None => pts.push((key, r.ap_d)),
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Monotone non-increasing ap in ae: from the widest cut down, keep the running max.
    for i in (0..pts.len().saturating_sub(1)).rev() {
        if pts[i].1 < pts[i + 1].1 {
            pts[i].1 = pts[i + 1].1;
        }
    }
    pts
}
