//! Reads a 3-axis mill program into straight, circular and drilling tool moves, in
//! mm, in the program's work frame. Two dialects, verified against the test nest's
//! posts 2026-09-28:
//!   Fanuc style   G20, G54, T.. M06, G43, I/J relative to the arc start
//!   Multicam      G70 (inch), G75, "G00 T05" tool select without M06, Z counting
//!                 DOWN (clearance Z-3.25, a cut 0.25 in above the table is Z-.25),
//!                 I/J absolute arc centers, G83 with a D peck and no G80
//! Z direction: a program whose first rapid Z is negative counts Z downward and is
//! negated. Arc centers: an arc's I/J are taken as relative or absolute, whichever
//! keeps its radius consistent (full circles, which fit both, follow the program's
//! other arcs).
//!
//! Handled: G20/G21/G70/G71, G90/G91, G00/G01/G02/G03 (G17 XY plane; full circles;
//! helical Z; modal motion), G81/G83 drilling with the R plane and G98/G99
//! (cancelled by G80 or any G00-G03, as on a Fanuc), G28 (straight up to a safe Z -
//! "G91 G28 X0 Y0 Z0" is a home move, NOT a move to 0,0,0), tool changes (a T word,
//! with or without M06), and (comments): the comment before a tool change's tool
//! description is taken as the operation name. Anything else is ignored.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::V3;

/// One tool from a post's tool sheet. Diameter and corner radius in mm;
/// `tip_angle_deg` only matters for drills.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub number: i32,
    pub kind: String,
    pub diameter: f64,
    pub corner_radius: f64,
    pub description: String,
    pub tip_angle_deg: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKind {
    Rapid,
    Feed,
    ArcCw,
    ArcCcw,
    Drill,
}

/// One tool move. Arcs: `center` is the XY center (z = `from.z`). Drill: `from` is the
/// R plane, `to` the depth. `feed` is the modal feed rate in mm/min (0 before any F
/// word; rapids ignore it; per-second F words, as the MultiCam post writes them, are
/// converted).
#[derive(Clone, Debug, PartialEq)]
pub struct Move {
    pub line: usize,
    pub op: String,
    pub tool: i32,
    pub kind: MoveKind,
    pub from: V3,
    pub to: V3,
    pub center: V3,
    pub feed: f64,
}

/// Drill point angle when the sheet does not give one (degrees).
const DRILL_TIP_DEG: f64 = 118.0;

/// A file's lines; bytes that are not UTF-8 become U+FFFD instead of failing the read.
fn lines(path: &Path) -> io::Result<Vec<String>> {
    let bytes = fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    Ok(text.lines().map(str::to_owned).collect())
}

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("a valid pattern"))
}

/// Character position of `pattern` in `line`, -1 if absent.
fn char_index(line: &str, pattern: &str) -> i64 {
    line.find(pattern).map_or(-1, |b| line[..b].chars().count() as i64)
}

/// Characters `a..b` of `line` (b capped at its end), trimmed; empty when `a` is past it.
fn field(line: &str, a: i64, b: i64) -> String {
    let len = line.chars().count() as i64;
    if len <= a {
        return String::new();
    }
    line.chars().skip(a as usize).take((b.min(len) - a).max(0) as usize).collect::<String>().trim().to_string()
}

/// The post's tool sheet, in `unit_mm` per program unit (25.4 for inches). Two
/// layouts: the Multicam rows ("DIA x C.RADIUS"), else a fixed-width table under the
/// header "( STATION  TOOL TYPE  DIAMETER  CORNER RADIUS  DESCRIPTION )", parsed by
/// the header's column positions: descriptions can start with a number (".25
/// Endmill") that a free-form parse mistakes for a corner radius. The sheet rounds
/// diameters to 3 decimals (1/16 -> "00.062", 0.0005 in off), so a size in the
/// description that agrees within 0.0015 in wins.
pub fn read_tool_table(path: &Path, unit_mm: f64) -> io::Result<BTreeMap<i32, ToolDef>> {
    let text = lines(path)?;
    let tools = read_dia_by_radius(&text, unit_mm);
    if !tools.is_empty() {
        return Ok(tools);
    }
    let mut tools = BTreeMap::new();
    let (mut corner, mut desc) = (-1i64, -1i64);
    let row = Regex::new(r"^\(\s*(\d+)\s+([A-Z]+)\s+(\d*\.?\d+)").expect("a valid pattern");
    for line in &text {
        if line.contains("DIAMETER") && line.contains("CORNER") {
            corner = char_index(line, "CORNER");
            desc = char_index(line, "DESCRIPTION");
            continue;
        }
        let Some(m) = row.captures(line) else { continue };
        if corner < 0 {
            continue;
        }
        let (Ok(n), Ok(mut d)) = (m[1].parse::<i32>(), m[3].parse::<f64>()) else { continue };
        let len = line.chars().count() as i64;
        let cr = field(line, corner, if desc > corner { desc } else { len });
        let text = field(line, if desc > 0 { desc } else { corner }, len).trim_end_matches([')', ' ']).to_string();
        if let Some(exact) = size_in(&text) {
            if (exact - d).abs() < 0.0015 {
                d = exact;
            }
        }
        let r = cr.parse::<f64>().unwrap_or(0.0);
        tools.insert(
            n,
            ToolDef { number: n, kind: m[2].to_string(), diameter: d * unit_mm, corner_radius: r * unit_mm, description: text, tip_angle_deg: DRILL_TIP_DEG },
        );
    }
    Ok(tools)
}

/// The Multicam post's layout, in its .set sheet (and the program header):
///   "( TOOL  TOOL TYPE  DIA x C.RADIUS  HANGOUT  TOOL COMMENT"
///   "(  05  FLAT ENDMILL  0.1250 x 0.000  1.200  .125"  Endmill 1 Flute )"
fn read_dia_by_radius(text: &[String], unit_mm: f64) -> BTreeMap<i32, ToolDef> {
    let mut tools = BTreeMap::new();
    let row = Regex::new(r"^\(\s*(\d+)\s+([A-Z][A-Z ]*?)\s+(\d*\.\d+)\s*x\s*(\d*\.\d+)\s+(?:\d*\.\d+\s+)?(.*?)\s*\)?\s*$")
        .expect("a valid pattern");
    for line in text {
        let Some(m) = row.captures(line) else { continue };
        let (Ok(n), Ok(mut d), Ok(mut r)) = (m[1].parse::<i32>(), m[3].parse::<f64>(), m[4].parse::<f64>()) else { continue };
        let kind = m[2].trim();
        let text = m[5].trim().to_string();
        if let Some(exact) = size_in(&text) {
            if (exact - d).abs() < 0.0015 {
                d = exact;
            }
        }
        let kind = if kind.contains("DRILL") {
            "DRILL"
        } else if kind.contains("BALL") {
            "BALL"
        } else {
            "ENDMILL"
        };
        if kind == "BALL" && r <= 0.0 {
            r = d / 2.0;
        }
        tools.insert(
            n,
            ToolDef { number: n, kind: kind.to_string(), diameter: d * unit_mm, corner_radius: r * unit_mm, description: text, tip_angle_deg: DRILL_TIP_DEG },
        );
    }
    tools
}

/// Leading size in a tool description: "1/16 EM ..." -> 0.0625, ".125\" ..." -> 0.125.
fn size_in(text: &str) -> Option<f64> {
    static FRACTION: OnceLock<Regex> = OnceLock::new();
    static DECIMAL: OnceLock<Regex> = OnceLock::new();
    if let Some(f) = re(&FRACTION, r"^(\d+)/(\d+)\b").captures(text) {
        return Some(f[1].parse::<f64>().ok()? / f[2].parse::<f64>().ok()?);
    }
    re(&DECIMAL, r"^(\d*\.\d+)").captures(text).and_then(|m| m[1].parse().ok())
}

/// `v` with at most three decimals and no trailing zeros.
fn three_places(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".to_string() } else { s.to_string() }
}

/// Reads the program at `nc_path` into moves. How it was read (Z flipped, feeds per
/// second, arcs skipped) goes to `warnings`.
pub fn read(nc_path: &Path, warnings: &mut Vec<String>) -> io::Result<Vec<Move>> {
    Ok(parse_lines(&lines(nc_path)?, warnings))
}

/// Reads a program held in memory, as [`read`] does a file.
pub fn parse(text: &str, warnings: &mut Vec<String>) -> Vec<Move> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();
    parse_lines(&lines, warnings)
}

fn parse_lines(lines: &[String], warnings: &mut Vec<String>) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    let mut scale = 25.4; // G20 until told otherwise
    let mut feed = 0.0; // mm/min
    let mut absolute = true;
    let (mut motion, mut tool, mut cycle) = (0i32, 0i32, 0i32); // cycle: 0 off, 81/83 active
    let mut return_to_r = true; // G99 (vs G98 initial level)
    let (mut cycle_z, mut cycle_r, mut initial_z) = (0.0, 0.0, 0.0);
    let mut pos = V3::new(0.0, 0.0, 50.0 * 25.4); // unknown start: well above everything
    let mut z_down: Option<bool> = None; // decided by the first rapid Z (see the module notes)
    let (mut abs_votes, mut rel_votes) = (0, 0); // how this program writes I/J
    let (mut op, mut last_comment, mut prev_comment) = (String::new(), String::new(), String::new());
    let word = Regex::new(r"([A-Z])\s*([-+]?\d*\.?\d+)").expect("a valid pattern");
    let comment = Regex::new(r"\(([^)]*)\)").expect("a valid pattern");
    let op_prefix = Regex::new(r"(?i)^OPERATION NAME:\s*").expect("a valid pattern");

    for (i, raw) in lines.iter().enumerate() {
        let line_no = i + 1;
        for c in comment.captures_iter(raw) {
            prev_comment = std::mem::take(&mut last_comment);
            last_comment = c[1].trim().to_string();
            // "( OPERATION NAME: ... )" names the operation where it stands: several
            // operations in a row can share a tool, so no tool change follows them.
            if op_prefix.is_match(&last_comment) {
                op = op_prefix.replace(&last_comment, "").into_owned();
            }
        }
        let code = comment.replace_all(raw, " ").to_ascii_uppercase();
        let words: Vec<(char, f64)> =
            word.captures_iter(&code).filter_map(|w| Some((w[1].chars().next()?, w[2].parse().ok()?))).collect();
        if words.is_empty() {
            continue;
        }

        let get = |l: char| words.iter().find(|w| w.0 == l).map(|w| w.1);
        let gs: Vec<i32> = words.iter().filter(|w| w.0 == 'G').map(|w| (w.1 * 10.0).round() as i32).collect(); // G83 -> 830
        let has = |g: i32| gs.contains(&g);

        if has(200) || has(700) {
            scale = 25.4;
        }
        if has(210) || has(710) {
            scale = 1.0;
        }
        if has(900) {
            absolute = true;
        }
        if has(910) {
            absolute = false;
        }
        if let Some(fw) = get('F') {
            feed = fw * scale; // G94: program units per minute
        }
        if has(980) {
            return_to_r = false;
        }
        if has(990) {
            return_to_r = true;
        }
        if let Some(t) = get('T') {
            if t as i32 != tool {
                tool = t as i32;
                // "( Rough Mill1 )" / "( OPERATION NAME: ROUGH MILL1 )" precedes the tool description.
                op = op_prefix.replace(&prev_comment, "").into_owned();
            }
        }

        if has(280) {
            // Home move: straight up out of the way; XY is not needed after it.
            let up = V3::new(pos.x, pos.y, pos.z.max(50.0 * 25.4));
            moves.push(Move { line: line_no, op: op.clone(), tool, kind: MoveKind::Rapid, from: pos, to: up, center: V3::default(), feed: 0.0 });
            pos = up;
            continue;
        }
        if has(800) {
            cycle = 0;
        }
        for &g in &gs {
            if g == 0 || g == 10 || g == 20 || g == 30 {
                motion = g / 10;
                cycle = 0;
            }
        }
        let starts = has(810) || has(830);
        if starts {
            cycle = if has(830) { 83 } else { 81 };
            initial_z = pos.z;
        }

        let (x, y, mut z, mut rw) = (get('X'), get('Y'), get('Z'), get('R'));
        if z_down.is_none() && motion == 0 && cycle == 0 && !starts {
            if let Some(z0) = z {
                z_down = Some(z0 < 0.0);
                if z0 < 0.0 {
                    warnings.push(format!("line {line_no}: first rapid is Z{z0} - this program counts Z downward (Multicam); Z negated"));
                }
            }
        }
        if z_down == Some(true) {
            z = z.map(|v| -v);
            rw = rw.map(|v| -v);
        }
        let target = if absolute {
            V3::new(x.map_or(pos.x, |v| v * scale), y.map_or(pos.y, |v| v * scale), z.map_or(pos.z, |v| v * scale))
        } else {
            pos + V3::new(x.unwrap_or(0.0) * scale, y.unwrap_or(0.0) * scale, z.unwrap_or(0.0) * scale)
        };
        let at_line = |kind: MoveKind, from: V3, to: V3, center: V3, feed: f64| Move { line: line_no, op: op.clone(), tool, kind, from, to, center, feed };

        if cycle != 0 {
            // Canned drilling: Z and R are cycle parameters, not motion.
            if let Some(r) = rw {
                cycle_r = r * scale;
            }
            if let Some(zc) = z {
                cycle_z = zc * scale;
            }
            if x.is_none() && y.is_none() && !starts {
                continue;
            }
            let at = V3::new(x.map_or(pos.x, |v| v * scale), y.map_or(pos.y, |v| v * scale), pos.z);
            if (at - pos).len_sq() > 1e-12 {
                moves.push(at_line(MoveKind::Rapid, pos, at, V3::default(), 0.0));
            }
            let r_plane = V3::new(at.x, at.y, cycle_r);
            let bottom = V3::new(at.x, at.y, cycle_z);
            moves.push(at_line(MoveKind::Drill, r_plane, bottom, V3::default(), feed));
            pos = V3::new(at.x, at.y, if return_to_r { cycle_r } else { initial_z });
            continue;
        }

        if x.is_none() && y.is_none() && z.is_none() && motion < 2 {
            continue;
        }
        if motion == 2 || motion == 3 {
            let i = get('I').unwrap_or(0.0) * scale;
            let j = get('J').unwrap_or(0.0) * scale;
            if rw.is_some() {
                warnings.push(format!("line {line_no}: R-form arc not supported - skipped"));
            } else {
                let rel = V3::new(pos.x + i, pos.y + j, pos.z);
                let abs = V3::new(i, j, pos.z);
                let err = |c: V3| {
                    (((pos.x - c.x) * (pos.x - c.x) + (pos.y - c.y) * (pos.y - c.y)).sqrt()
                        - ((target.x - c.x) * (target.x - c.x) + (target.y - c.y) * (target.y - c.y)).sqrt())
                    .abs()
                };
                let full = (target.x - pos.x).abs() < 1e-6 && (target.y - pos.y).abs() < 1e-6;
                let use_abs = if full {
                    abs_votes > rel_votes
                } else {
                    let a = err(abs) < err(rel);
                    if a {
                        abs_votes += 1;
                    } else {
                        rel_votes += 1;
                    }
                    a
                };
                let kind = if motion == 2 { MoveKind::ArcCw } else { MoveKind::ArcCcw };
                moves.push(at_line(kind, pos, target, if use_abs { abs } else { rel }, feed));
            }
        } else if (target - pos).len_sq() > 1e-12 {
            let kind = if motion == 0 { MoveKind::Rapid } else { MoveKind::Feed };
            moves.push(at_line(kind, pos, target, V3::default(), feed));
        }
        pos = target;
    }

    // Units per second: the MultiCam post writes feeds in inches per second (F2.5 is
    // 150 in/min). No router cuts at 30 program units per minute, so a typical F under
    // 30 reads as per second.
    let mut feeds: Vec<f64> = moves.iter().filter(|m| m.kind != MoveKind::Rapid && m.feed > 0.0).map(|m| m.feed / scale).collect();
    feeds.sort_by(f64::total_cmp);
    if let Some(&median) = feeds.get(feeds.len() / 2) {
        if median < 30.0 {
            warnings.push(format!(
                "feeds read as program units per second (median F{}; the MultiCam post writes inches per second)",
                three_places(median)
            ));
            for m in &mut moves {
                m.feed *= 60.0;
            }
        }
    }
    moves
}
