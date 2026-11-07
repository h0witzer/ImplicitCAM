//! Writing our moves as a program. A strategy's measured path (points with their
//! scheduled feed) becomes one operation's block: a rapid down to the entry height, then
//! G01 moves with the feed quantized to 3% steps and the points of each run of equal
//! feed and tag thinned by Douglas-Peucker within `tol` (0.005 mm), every motion word
//! restated when it changes. [`Writer`] wraps the blocks in a whole program.
//!
//! Two dialects:
//!   Fanuc     G21 mm, Z up, F in mm/min, a T.. M06 tool change
//!   MultiCam  G70 inches, Z counting DOWN from the table (Z-3.25 is clearance), F in
//!             inches per SECOND, "G00 T05" tool select without M06, numbers without a
//!             leading zero (".25")
//! Both are read back by `gcode::parse`, so the instruments and the checker measure the
//! program as written.

use crate::Tag;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Fanuc,
    MultiCam,
}

/// A point to post: tool center (mm), the feed scheduled there (mm/s), and its tag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostPt {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub feed: f64,
    pub tag: Tag,
}

/// `v` rounded to at most four decimals and at least one ("2.5", "0.0313", "100.0"),
/// never "-0.0".
pub fn decimals4(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0');
    let s = if s.ends_with('.') { format!("{s}0") } else { s.to_string() };
    if s == "-0.0" { "0.0".to_string() } else { s }
}

/// Inches, without a leading zero (".25", "-.25").
fn inches(mm: f64) -> String {
    let s = decimals4(mm / 25.4);
    if let Some(r) = s.strip_prefix("0.") {
        format!(".{r}")
    } else if let Some(r) = s.strip_prefix("-0.") {
        format!("-.{r}")
    } else {
        s
    }
}

impl Dialect {
    fn x(self, mm: f64) -> String {
        match self {
            Dialect::Fanuc => decimals4(mm),
            Dialect::MultiCam => inches(mm),
        }
    }

    /// Z: up in mm, or (MultiCam) counting down from the table, in inches.
    fn z(self, mm: f64) -> String {
        match self {
            Dialect::Fanuc => decimals4(mm),
            Dialect::MultiCam => inches(-mm),
        }
    }

    /// Feed from mm/s: mm/min, or (MultiCam) inches per second.
    fn f(self, mmps: f64) -> String {
        match self {
            Dialect::Fanuc => decimals4(mmps * 60.0),
            Dialect::MultiCam => {
                let s = decimals4(mmps / 25.4);
                s.strip_prefix("0.").map_or(s.clone(), |r| format!(".{r}"))
            }
        }
    }
}

/// One operation's cutting block: a rapid down to `z_entry`, then the points (the first
/// is where the tool already stands) as G00 / G01 moves, collinear runs thinned within
/// `tol`, feed quantized to 3%. Coordinates are written relative to `origin` (the work
/// frame's zero).
pub fn body(s: &[PostPt], z_entry: f64, tol: f64, d: Dialect, origin: (f64, f64, f64)) -> Vec<String> {
    let (ox, oy, oz) = origin;
    let mut lines = vec![format!("G00 Z{}", d.z(z_entry - oz))];
    if s.is_empty() {
        return lines;
    }
    // Quantized feed per sample, then Douglas-Peucker within runs of equal feed and tag.
    let q: Vec<f64> = s.iter().map(|p| (p.feed.max(0.1).ln() / 1.03f64.ln()).round()).collect();
    let (mut start, mut mode, mut last_f) = (0usize, 0, -1.0f64); // mode: the motion word written so far (0 rapid, 1 feed)
    while start < s.len() - 1 {
        // The run: start + 1 .. end share feed and tag (start is the previous run's last point).
        let mut end = start + 1;
        while end + 1 < s.len() && q[end + 1] == q[start + 1] && s[end + 1].tag == s[start + 1].tag {
            end += 1;
        }
        let mut keep = Vec::new();
        thin(s, start, end, tol, &mut keep);
        keep.push(end);
        let feed = 1.03f64.powf(q[start + 1]);
        for k in keep {
            let p = s[k];
            if p.tag == Tag::Rapid {
                lines.push(format!("{}X{} Y{} Z{}", if mode != 0 { "G00 " } else { "" }, d.x(p.x - ox), d.x(p.y - oy), d.z(p.z - oz)));
                mode = 0;
                continue;
            }
            let mut l = String::new();
            if mode != 1 {
                l.push_str("G01 ");
                mode = 1;
                last_f = -1.0;
            }
            l.push_str(&format!("X{} Y{} Z{}", d.x(p.x - ox), d.x(p.y - oy), d.z(p.z - oz)));
            if (feed - last_f).abs() > 1e-9 {
                l.push_str(&format!(" F{}", d.f(feed)));
                last_f = feed;
            }
            lines.push(l);
        }
        start = end;
    }
    lines
}

/// Douglas-Peucker between a and b: the points to keep, in order (a and b excluded).
fn thin(s: &[PostPt], a: usize, b: usize, tol: f64, keep: &mut Vec<usize>) {
    let (mut best, mut arg) = (0.0, None);
    for i in a + 1..b {
        let d = seg_dist3(&s[i], &s[a], &s[b]);
        if d > best {
            best = d;
            arg = Some(i);
        }
    }
    if let Some(i) = arg {
        if best > tol {
            thin(s, a, i, tol, keep);
            keep.push(i);
            thin(s, i, b, tol, keep);
        }
    }
}

fn seg_dist3(p: &PostPt, a: &PostPt, b: &PostPt) -> f64 {
    let (ux, uy, uz) = (b.x - a.x, b.y - a.y, b.z - a.z);
    let l2 = ux * ux + uy * uy + uz * uz;
    let t = if l2 < 1e-18 { 0.0 } else { (((p.x - a.x) * ux + (p.y - a.y) * uy + (p.z - a.z) * uz) / l2).clamp(0.0, 1.0) };
    let (ex, ey, ez) = (p.x - (a.x + t * ux), p.y - (a.y + t * uy), p.z - (a.z + t * uz));
    (ex * ex + ey * ey + ez * ez).sqrt()
}

/// A whole program: header, one block per operation, footer. Everything is written in
/// the work frame whose zero is `origin` (part coordinates minus it); read a program
/// back with [`Writer::read_back`] to return to part coordinates.
pub struct Writer {
    d: Dialect,
    origin: (f64, f64, f64),
    clear_z: f64,
    lines: Vec<String>,
    /// Lines of the header (units, tool, spindle, clearance).
    header: usize,
}

impl Writer {
    /// Header: units, the tool, the spindle, the clearance height `clear_z` (mm, part
    /// coordinates).
    pub fn new(d: Dialect, origin: (f64, f64, f64), title: &str, tool: i32, tool_note: &str, rpm: f64, clear_z: f64) -> Writer {
        let mut lines = Vec::new();
        match d {
            Dialect::Fanuc => {
                lines.extend(["%".to_string(), format!("({title})"), "G21 G17 G40 G49 G80 G90".into(), format!("({tool_note})")]);
                lines.push(format!("T{tool} M06"));
                lines.push(format!("S{} M03", rpm.round()));
                lines.push("G54".into());
            }
            Dialect::MultiCam => {
                lines.extend([format!("({title})"), "G70 G75 G90".into(), format!("({tool_note})")]);
                lines.push(format!("G00 T{tool:02}"));
                lines.push(format!("S{} M03", rpm.round()));
            }
        }
        lines.push(format!("G00 Z{}", d.z(clear_z - origin.2)));
        let header = lines.len();
        Writer { d, origin, clear_z, lines, header }
    }

    /// One operation: its name, a rapid over its start point, its block, a retract.
    pub fn op(&mut self, name: &str, start: (f64, f64), block: &[String]) {
        self.lines.push(format!("(OPERATION NAME: {name})"));
        self.lines.push(format!("G00 X{} Y{}", self.d.x(start.0 - self.origin.0), self.d.x(start.1 - self.origin.1)));
        self.lines.extend(block.iter().cloned());
        self.lines.push(format!("G00 Z{}", self.d.z(self.clear_z - self.origin.2)));
    }

    pub fn origin(&self) -> (f64, f64, f64) {
        self.origin
    }

    /// A program written in the work frame whose zero is `origin`, read back into part
    /// coordinates.
    pub fn read_back(text: &str, origin: (f64, f64, f64), warnings: &mut Vec<String>) -> Vec<crate::Move> {
        let o = crate::V3::new(origin.0, origin.1, origin.2);
        let mut moves = crate::parse(text, warnings);
        for m in &mut moves {
            m.from = m.from + o;
            m.to = m.to + o;
            if m.kind == crate::MoveKind::ArcCw || m.kind == crate::MoveKind::ArcCcw {
                m.center = m.center + o;
            }
        }
        moves
    }

    /// Lines written so far.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// The moves of the lines from `first` on (0-based) read back into part coordinates,
    /// numbered as in the whole program, without reading what comes before them (the
    /// header gives the units and the work frame).
    pub fn read_back_from(&self, first: usize, warnings: &mut Vec<String>) -> Vec<crate::Move> {
        let first = first.max(self.header);
        let mut text = self.lines[..self.header].join("\n");
        text.push('\n');
        text.push_str(&self.lines[first..].join("\n"));
        text.push('\n');
        let mut moves = Writer::read_back(&text, self.origin, warnings);
        moves.retain(|m| m.line > self.header);
        for m in &mut moves {
            m.line = m.line - self.header + first;
        }
        moves
    }

    /// The program so far, without the footer.
    pub fn text(&self) -> String {
        self.lines.join("\n") + "\n"
    }

    /// The finished program.
    pub fn finish(mut self) -> String {
        self.lines.push("M05".into());
        self.lines.push("M30".into());
        if self.d == Dialect::Fanuc {
            self.lines.push("%".into());
        }
        self.text()
    }
}

#[cfg(test)]
mod tests {
    use super::decimals4;

    #[test]
    fn decimals_are_the_value_to_four_places() {
        let cases = [
            (0.03126, "0.0313"),
            (1.00006, "1.0001"),
            (0.00006, "0.0001"),
            (2.5, "2.5"),
            (12.34566, "12.3457"),
            (-0.00004, "0.0"),
            (-0.03126, "-0.0313"),
            (1234.56789, "1234.5679"),
            (100.0, "100.0"),
            (0.0, "0.0"),
            (-0.0, "0.0"),
            (3.99996, "4.0"),
            (0.000049999, "0.0"),
        ];
        for (v, want) in cases {
            assert_eq!(decimals4(v), want, "{v}");
        }
        // Every value reads back within half a unit of the fourth place.
        let mut x = 0.123456789f64;
        for _ in 0..10000 {
            x = (x * 7919.0 + 0.3183).fract();
            let v = (x - 0.5) * 2000.0;
            let back: f64 = decimals4(v).parse().unwrap();
            assert!((back - v).abs() <= 0.5e-4 + 1e-12, "{v} -> {back}");
        }
    }
}
