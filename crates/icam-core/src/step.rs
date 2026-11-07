//! STEP (ISO 10303-21) reader for exact B-rep solids.
//!
//! Reads the DATA section into entities, then walks MANIFOLD_SOLID_BREP (or
//! FACETED_BREP) -> CLOSED_SHELL -> ADVANCED_FACE (FACE_SURFACE, ORIENTED_FACE) ->
//! bounds -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE (or POLY_LOOP) into an exact
//! model: every surface and curve with its analytic parameters in double precision,
//! every vertex as written. No tessellation. Lengths are converted to millimeters
//! from the file's length unit. Files written with AP203 short entity names
//! (CRTPNT for CARTESIAN_POINT) read the same.
//!
//! Covered: PLANE, CYLINDRICAL, CONICAL, SPHERICAL, TOROIDAL (degenerate included)
//! surfaces and B-spline surfaces in all four knot forms (rational included);
//! SURFACE_OF_LINEAR_EXTRUSION and SURFACE_OF_REVOLUTION, read as the analytic
//! surface they are or else as an exact rational B-spline surface. Edges on LINE,
//! CIRCLE, ELLIPSE, PARABOLA, HYPERBOLA and B-spline curves in all four knot forms,
//! through SURFACE_CURVE, SEAM_CURVE, INTERSECTION_CURVE and TRIMMED_CURVE. Anything
//! else is kept as `Other(name)` so the importer downstream can refuse it by name.
//!
//! Assemblies. Each solid is placed once per use, through the representation
//! relationships with a transformation and the MAPPED_ITEMs above it. Solids whose
//! shells are open but close up together (one face per MANIFOLD_SOLID_BREP, sharing
//! edges) are joined into one.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl V3 {
    pub const fn new(x: f64, y: f64, z: f64) -> V3 {
        V3 { x, y, z }
    }
    pub fn dot(self, o: V3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: V3) -> V3 {
        V3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    pub fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn scale(self, s: f64) -> V3 {
        V3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn unit(self) -> V3 {
        let l = self.len();
        if l > 0.0 {
            self.scale(1.0 / l)
        } else {
            self
        }
    }
}

/// Right-handed frame: origin, z axis, x axis (y = z cross x).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub o: V3,
    pub z: V3,
    pub x: V3,
}

impl Frame {
    pub fn y(&self) -> V3 {
        self.z.cross(self.x)
    }
}

#[derive(Clone, Debug)]
pub struct BSplineCurve {
    pub degree: usize,
    pub ctrl: Vec<V3>,
    pub weights: Option<Vec<f64>>,
    /// Knot vector with multiplicities expanded.
    pub knots: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct BSplineSurface {
    pub du: usize,
    pub dv: usize,
    /// ctrl[i][j], i along u.
    pub ctrl: Vec<Vec<V3>>,
    pub weights: Option<Vec<Vec<f64>>>,
    pub ku: Vec<f64>,
    pub kv: Vec<f64>,
}

#[derive(Clone, Debug)]
pub enum Curve3 {
    Line { p: V3, dir: V3 },
    Circle { frame: Frame, r: f64 },
    Ellipse { frame: Frame, a: f64, b: f64 },
    /// o + f (t^2 x + 2 t y). An edge on one is read as the B-spline of its arc.
    Parabola { frame: Frame, f: f64 },
    /// o + a cosh(t) x + b sinh(t) y. An edge on one is read as the B-spline of its arc.
    Hyperbola { frame: Frame, a: f64, b: f64 },
    BSpline(BSplineCurve),
    Other(String),
}

#[derive(Clone, Debug)]
pub enum Surface {
    Plane { frame: Frame },
    Cylinder { frame: Frame, r: f64 },
    Cone { frame: Frame, r: f64, semi_angle: f64 },
    Sphere { frame: Frame, r: f64 },
    /// The meridian circle of radius `minor` about the point `major` out from the axis,
    /// on the axis's own side (distance from the axis >= 0) revolved about it. A ring
    /// torus has major > minor; a self-intersecting one (DEGENERATE_TOROIDAL_SURFACE)
    /// has its outer part (the apple) as major >= 0 and its inner part (the lemon) as
    /// major < 0: the inner part is the circle's far side, which in any half-plane is the
    /// circle about -major.
    Torus { frame: Frame, major: f64, minor: f64 },
    BSpline(BSplineSurface),
    Extrusion { curve: Curve3, dir: V3 },
    Revolution { curve: Curve3, axis_o: V3, axis: V3 },
    Other(String),
}

/// An edge as used by one loop: traversed from `start` to `end`.
/// `forward` is true when the traversal follows the curve's own parameter direction.
#[derive(Clone, Debug)]
pub struct OEdge {
    pub edge: u32,
    pub start: V3,
    pub end: V3,
    pub start_vertex: u32,
    pub end_vertex: u32,
    pub curve: Curve3,
    pub forward: bool,
}

#[derive(Clone, Debug)]
pub struct Bound {
    pub outer: bool,
    /// Edges in loop order, already reversed if the bound's orientation flag is false.
    pub edges: Vec<OEdge>,
    /// A VERTEX_LOOP (a degenerate bound at a single point, e.g. a cone apex).
    pub vertex: Option<V3>,
}

#[derive(Clone, Debug)]
pub struct BFace {
    /// STEP entity id of the ADVANCED_FACE (stable within the file).
    pub id: u32,
    pub surface: Surface,
    /// True when the face's outward normal is the surface's own normal.
    pub same_sense: bool,
    pub bounds: Vec<Bound>,
}

#[derive(Clone, Debug)]
pub struct Solid {
    pub id: u32,
    pub name: String,
    pub faces: Vec<BFace>,
}

#[derive(Clone, Debug, Default)]
pub struct Brep {
    pub solids: Vec<Solid>,
    /// Millimeters per file length unit (1 for mm, 25.4 for inch).
    pub unit_mm: f64,
    pub unit_name: String,
    /// The file's declared length uncertainty (mm): how far its vertices may sit
    /// from their curves and surfaces. 1e-6 mm when the file declares none.
    pub tolerance_mm: f64,
}

// ---------------------------------------------------------------- Part 21 parse

#[derive(Clone, Debug)]
enum Val {
    Ref(u32),
    Num(f64),
    Str(String),
    Enum(String),
    List(Vec<Val>),
    #[allow(dead_code)]
    Typed(String, Vec<Val>),
    Null,
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        loop {
            while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.b[self.i..].starts_with(b"/*") {
                match self.b[self.i + 2..].windows(2).position(|w| w == b"*/") {
                    Some(k) => self.i += k + 4,
                    None => self.i = self.b.len(),
                }
            } else {
                return;
            }
        }
    }
    fn peek(&mut self) -> u8 {
        self.ws();
        *self.b.get(self.i).unwrap_or(&0)
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == c {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn ident(&mut self) -> String {
        self.ws();
        let s = self.i;
        while self.i < self.b.len() && (self.b[self.i].is_ascii_alphanumeric() || self.b[self.i] == b'_') {
            self.i += 1;
        }
        String::from_utf8_lossy(&self.b[s..self.i]).to_uppercase()
    }
    fn args(&mut self) -> Result<Vec<Val>, String> {
        self.expect(b'(')?;
        let mut v = Vec::new();
        if self.peek() == b')' {
            self.i += 1;
            return Ok(v);
        }
        loop {
            v.push(self.value()?);
            match self.peek() {
                b',' => self.i += 1,
                b')' => {
                    self.i += 1;
                    return Ok(v);
                }
                c => return Err(format!("unexpected '{}' at byte {}", c as char, self.i)),
            }
        }
    }
    fn value(&mut self) -> Result<Val, String> {
        match self.peek() {
            b'#' => {
                self.i += 1;
                let s = self.i;
                while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                    self.i += 1;
                }
                Ok(Val::Ref(std::str::from_utf8(&self.b[s..self.i]).unwrap().parse().map_err(|e| format!("{e}"))?))
            }
            b'\'' => {
                self.i += 1;
                let mut s = Vec::new();
                loop {
                    let c = *self.b.get(self.i).ok_or("unterminated string")?;
                    self.i += 1;
                    if c == b'\'' {
                        if self.b.get(self.i) == Some(&b'\'') {
                            s.push(b'\'');
                            self.i += 1;
                            continue;
                        }
                        break;
                    }
                    s.push(c);
                }
                Ok(Val::Str(String::from_utf8_lossy(&s).into_owned()))
            }
            b'.' => {
                self.i += 1;
                let s = self.i;
                while self.i < self.b.len() && self.b[self.i] != b'.' {
                    self.i += 1;
                }
                let e = String::from_utf8_lossy(&self.b[s..self.i]).to_uppercase();
                self.i += 1;
                Ok(Val::Enum(e))
            }
            b'(' => Ok(Val::List(self.args()?)),
            b'$' | b'*' => {
                self.i += 1;
                Ok(Val::Null)
            }
            b'"' => {
                // binary literal: skip
                self.i += 1;
                while self.i < self.b.len() && self.b[self.i] != b'"' {
                    self.i += 1;
                }
                self.i += 1;
                Ok(Val::Null)
            }
            c if c == b'-' || c == b'+' || c.is_ascii_digit() => {
                let s = self.i;
                self.i += 1;
                while self.i < self.b.len() && (self.b[self.i].is_ascii_digit() || b".eE+-".contains(&self.b[self.i])) {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[s..self.i]).unwrap();
                Ok(Val::Num(t.parse().map_err(|e| format!("bad number {t}: {e}"))?))
            }
            c if c.is_ascii_alphabetic() => {
                let name = self.ident();
                Ok(Val::Typed(name, self.args()?))
            }
            c => Err(format!("unexpected '{}' at byte {}", c as char, self.i)),
        }
    }
}

type Entities = HashMap<u32, Vec<(String, Vec<Val>)>>;

/// The full name of an AP203 short entity name, for the entities this reader reads
/// (the short names are the standard's own; CATIA V4 writes them).
fn long_name(short: &str) -> Option<&'static str> {
    Some(match short {
        "A2PL3D" => "AXIS2_PLACEMENT_3D",
        "ADVFC" => "ADVANCED_FACE",
        "BSCWK" => "B_SPLINE_CURVE_WITH_KNOTS",
        "BSSWK" => "B_SPLINE_SURFACE_WITH_KNOTS",
        "CLSSHL" => "CLOSED_SHELL",
        "CNBSUN" => "CONVERSION_BASED_UNIT",
        "CNCSRF" => "CONICAL_SURFACE",
        "CRTPNT" => "CARTESIAN_POINT",
        "CYLSRF" => "CYLINDRICAL_SURFACE",
        "DRCTN" => "DIRECTION",
        "EDGCRV" => "EDGE_CURVE",
        "EDGLP" => "EDGE_LOOP",
        "FBSR" => "FACETED_BREP_SHAPE_REPRESENTATION",
        "FCBN" => "FACE_BOUND",
        "FCOTBN" => "FACE_OUTER_BOUND",
        "FCSRF" => "FACE_SURFACE",
        "FCTBR" => "FACETED_BREP",
        "GC" => "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT",
        "GMRPCN" => "GEOMETRIC_REPRESENTATION_CONTEXT",
        "GUAC" => "GLOBAL_UNIT_ASSIGNED_CONTEXT",
        "LMWU" => "LENGTH_MEASURE_WITH_UNIT",
        "LNGUNT" => "LENGTH_UNIT",
        "MNSLBR" => "MANIFOLD_SOLID_BREP",
        "NAUO" => "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        "NMDUNT" => "NAMED_UNIT",
        "ORNEDG" => "ORIENTED_EDGE",
        "PAMWU" => "PLANE_ANGLE_MEASURE_WITH_UNIT",
        "PLANUN" => "PLANE_ANGLE_UNIT",
        "PLYLP" => "POLY_LOOP",
        "PRDDFN" => "PRODUCT_DEFINITION",
        "PRDFSH" => "PRODUCT_DEFINITION_SHAPE",
        "RPRCNT" => "REPRESENTATION_CONTEXT",
        "SHDFRP" => "SHAPE_DEFINITION_REPRESENTATION",
        "SHPRPR" => "SHAPE_REPRESENTATION",
        "SHRPRL" => "SHAPE_REPRESENTATION_RELATIONSHIP",
        "SPHSRF" => "SPHERICAL_SURFACE",
        "SUNT" => "SI_UNIT",
        "TRDSRF" => "TOROIDAL_SURFACE",
        "TRMCRV" => "TRIMMED_CURVE",
        "UMWU" => "UNCERTAINTY_MEASURE_WITH_UNIT",
        "VRTLP" => "VERTEX_LOOP",
        "VRTPNT" => "VERTEX_POINT",
        _ => return None,
    })
}

/// The DATA section's instances: where each starts is found in one pass (to the `;` that
/// ends it, past strings and comments), then they are parsed on every core.
fn parse_data(text: &[u8]) -> Result<Entities, String> {
    use rayon::prelude::*;
    let start = text.windows(5).position(|w| w == b"DATA;").ok_or("no DATA section")? + 5;
    let mut p = Parser { b: text, i: start };
    let mut starts = Vec::new();
    loop {
        match p.peek() {
            b'#' => {}
            b'E' => break, // ENDSEC
            0 => break,
            c => return Err(format!("unexpected '{}' at byte {}", c as char, p.i)),
        }
        starts.push(p.i);
        let (mut i, mut quoted) = (p.i, false);
        loop {
            match text.get(i) {
                None => return Err(format!("the instance at byte {} has no end", p.i)),
                // A doubled quote inside a string closes and opens it again.
                Some(b'\'') => quoted = !quoted,
                Some(b';') if !quoted => break,
                Some(b'/') if !quoted && text.get(i + 1) == Some(&b'*') => {
                    i = text[i + 2..].windows(2).position(|w| w == b"*/").map_or(text.len(), |k| i + 2 + k + 1);
                }
                _ => {}
            }
            i += 1;
        }
        p.i = i + 1;
    }
    let parsed: Vec<Result<(u32, Vec<(String, Vec<Val>)>), String>> = starts.par_iter().map(|&s| parse_instance(&mut Parser { b: text, i: s })).collect();
    let mut ents = Entities::with_capacity(parsed.len());
    for r in parsed {
        let (id, parts) = r?;
        ents.insert(id, parts);
    }
    Ok(ents)
}

/// One instance, `#id = NAME(...);` or a complex one, `#id = (NAME(...) NAME(...));`.
fn parse_instance(p: &mut Parser) -> Result<(u32, Vec<(String, Vec<Val>)>), String> {
    p.expect(b'#')?;
    let s = p.i;
    while p.i < p.b.len() && p.b[p.i].is_ascii_digit() {
        p.i += 1;
    }
    let id: u32 = std::str::from_utf8(&p.b[s..p.i]).unwrap().parse().map_err(|e| format!("{e}"))?;
    p.expect(b'=')?;
    let mut parts = Vec::new();
    if p.peek() == b'(' {
        // complex instance: ( NAME(...) NAME(...) ... )
        p.i += 1;
        while p.peek() != b')' {
            let name = p.ident();
            parts.push((name, p.args()?));
        }
        p.i += 1;
    } else {
        let name = p.ident();
        parts.push((name, p.args()?));
    }
    p.expect(b';')?;
    for (name, _) in parts.iter_mut() {
        if let Some(l) = long_name(name) {
            *name = l.to_string();
        }
    }
    Ok((id, parts))
}

// ---------------------------------------------------------------- model build

struct Ctx<'a> {
    e: &'a Entities,
    unit: f64,
    /// Radians per plane angle unit.
    angle: f64,
    /// The edges of POLY_LOOPs, which have none of their own: one per pair of
    /// points, numbered past the file's entities.
    poly: RefCell<HashMap<(u32, u32), u32>>,
    next_id: Cell<u32>,
}

fn err<T>(id: u32, what: &str) -> Result<T, String> {
    Err(format!("#{id}: {what}"))
}

impl Ctx<'_> {
    fn part(&self, id: u32, name: &str) -> Result<&Vec<Val>, String> {
        let ent = self.e.get(&id).ok_or(format!("#{id}: missing entity"))?;
        ent.iter().find(|(n, _)| n == name).map(|(_, a)| a).ok_or(format!("#{id}: not a {name}"))
    }
    fn name(&self, id: u32) -> String {
        self.e.get(&id).map(|v| v.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join("+")).unwrap_or_default()
    }
    fn is(&self, id: u32, name: &str) -> bool {
        self.e.get(&id).is_some_and(|v| v.iter().any(|(n, _)| n == name))
    }
    fn r(v: &Val) -> Result<u32, String> {
        match v {
            Val::Ref(r) => Ok(*r),
            other => Err(format!("expected a reference, got {other:?}")),
        }
    }
    fn num(v: &Val) -> Result<f64, String> {
        match v {
            Val::Num(x) => Ok(*x),
            Val::Typed(_, a) if a.len() == 1 => Ctx::num(&a[0]),
            other => Err(format!("expected a number, got {other:?}")),
        }
    }
    fn flag(v: &Val) -> bool {
        matches!(v, Val::Enum(e) if e == "T")
    }
    fn list(v: &Val) -> Result<&Vec<Val>, String> {
        match v {
            Val::List(l) => Ok(l),
            other => Err(format!("expected a list, got {other:?}")),
        }
    }
    fn point(&self, id: u32) -> Result<V3, String> {
        let a = self.part(id, "CARTESIAN_POINT")?;
        let c = Ctx::list(&a[1])?;
        let g = |k: usize| c.get(k).map(Ctx::num).transpose().map(|v| v.unwrap_or(0.0));
        Ok(V3::new(g(0)?, g(1)?, g(2)?).scale(self.unit))
    }
    fn dir(&self, id: u32) -> Result<V3, String> {
        let a = self.part(id, "DIRECTION")?;
        let c = Ctx::list(&a[1])?;
        let g = |k: usize| c.get(k).map(Ctx::num).transpose().map(|v| v.unwrap_or(0.0));
        Ok(V3::new(g(0)?, g(1)?, g(2)?).unit())
    }
    fn frame(&self, id: u32) -> Result<Frame, String> {
        let a = self.part(id, "AXIS2_PLACEMENT_3D")?;
        let o = self.point(Ctx::r(&a[1])?)?;
        let z = match &a[2] {
            Val::Ref(r) => self.dir(*r)?,
            _ => V3::new(0.0, 0.0, 1.0),
        };
        let x0 = match &a[3] {
            Val::Ref(r) => self.dir(*r)?,
            _ => {
                if z.x.abs() < 0.9 {
                    V3::new(1.0, 0.0, 0.0)
                } else {
                    V3::new(0.0, 1.0, 0.0)
                }
            }
        };
        // Gram-Schmidt: x perpendicular to z.
        let x = x0.sub(z.scale(x0.dot(z))).unit();
        Ok(Frame { o, z, x })
    }
    fn knots(mults: &Val, knots: &Val) -> Result<Vec<f64>, String> {
        let (m, k) = (Ctx::list(mults)?, Ctx::list(knots)?);
        let mut out = Vec::new();
        for (mi, ki) in m.iter().zip(k) {
            for _ in 0..(Ctx::num(mi)? as usize) {
                out.push(Ctx::num(ki)?);
            }
        }
        Ok(out)
    }
    fn curve(&self, id: u32) -> Result<Curve3, String> {
        let n = self.name(id);
        if self.is(id, "LINE") {
            let a = self.part(id, "LINE")?;
            let p = self.point(Ctx::r(&a[1])?)?;
            let v = self.part(Ctx::r(&a[2])?, "VECTOR")?;
            let d = self.dir(Ctx::r(&v[1])?)?;
            Ok(Curve3::Line { p, dir: d })
        } else if self.is(id, "CIRCLE") {
            let a = self.part(id, "CIRCLE")?;
            Ok(Curve3::Circle { frame: self.frame(Ctx::r(&a[1])?)?, r: Ctx::num(&a[2])? * self.unit })
        } else if self.is(id, "ELLIPSE") {
            let a = self.part(id, "ELLIPSE")?;
            Ok(Curve3::Ellipse { frame: self.frame(Ctx::r(&a[1])?)?, a: Ctx::num(&a[2])? * self.unit, b: Ctx::num(&a[3])? * self.unit })
        } else if self.is(id, "PARABOLA") {
            let a = self.part(id, "PARABOLA")?;
            Ok(Curve3::Parabola { frame: self.frame(Ctx::r(&a[1])?)?, f: Ctx::num(&a[2])? * self.unit })
        } else if self.is(id, "HYPERBOLA") {
            let a = self.part(id, "HYPERBOLA")?;
            Ok(Curve3::Hyperbola { frame: self.frame(Ctx::r(&a[1])?)?, a: Ctx::num(&a[2])? * self.unit, b: Ctx::num(&a[3])? * self.unit })
        } else if let Some(b) = self.bspline_curve(id)? {
            Ok(Curve3::BSpline(b))
        } else if let Some(w) = ["SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE"].into_iter().find(|w| self.is(id, w)) {
            // The curve in space; its curves on the surfaces are not needed.
            self.curve(Ctx::r(&self.part(id, w)?[1])?)
        } else if self.is(id, "TRIMMED_CURVE") {
            // The basis curve, run the way the trimmed curve runs; the edge's own
            // vertices bound it.
            let a = self.part(id, "TRIMMED_CURVE")?;
            let basis = self.curve(Ctx::r(&a[1])?)?;
            Ok(if Ctx::flag(&a[4]) { basis } else { reversed(basis) })
        } else {
            Ok(Curve3::Other(n))
        }
    }
    /// A B-spline curve in any of its four forms: with knots, or uniform,
    /// quasi-uniform or Bezier, whose knots the form implies.
    fn bspline_curve(&self, id: u32) -> Result<Option<BSplineCurve>, String> {
        const FORMS: [(&str, Knots); 4] =
            [("B_SPLINE_CURVE_WITH_KNOTS", Knots::Listed), ("UNIFORM_CURVE", Knots::Uniform), ("QUASI_UNIFORM_CURVE", Knots::QuasiUniform), ("BEZIER_CURVE", Knots::Bezier)];
        let Some(&(form, kind)) = FORMS.iter().find(|(f, _)| self.is(id, f)) else {
            return Ok(None);
        };
        // A simple instance keeps everything in one record: (name, degree, points,
        // form, closed, self_intersect[, multiplicities, knots, spec]). A complex one
        // splits it: B_SPLINE_CURVE(degree, points, ...) and the form's own record,
        // (multiplicities, knots, spec) or nothing.
        let fa = self.part(id, form)?;
        let complex = self.is(id, "B_SPLINE_CURVE");
        let (degree, pts) = if complex {
            let bc = self.part(id, "B_SPLINE_CURVE")?;
            (&bc[0], &bc[1])
        } else {
            (&fa[1], &fa[2])
        };
        let ctrl = Ctx::list(pts)?.iter().map(|v| self.point(Ctx::r(v)?)).collect::<Result<Vec<_>, _>>()?;
        let degree = Ctx::num(degree)? as usize;
        let knots = match kind {
            Knots::Listed if complex => Ctx::knots(&fa[0], &fa[1])?,
            Knots::Listed => Ctx::knots(&fa[6], &fa[7])?,
            _ => kind.implied(ctrl.len(), degree).ok_or(format!("#{id}: {} control points do not make a {form} of degree {degree}", ctrl.len()))?,
        };
        let weights = if self.is(id, "RATIONAL_B_SPLINE_CURVE") {
            let w = self.part(id, "RATIONAL_B_SPLINE_CURVE")?;
            let l = Ctx::list(w.last().unwrap())?;
            Some(l.iter().map(Ctx::num).collect::<Result<Vec<_>, _>>()?)
        } else {
            None
        };
        Ok(Some(BSplineCurve { degree, ctrl, weights, knots }))
    }
    fn surface(&self, id: u32) -> Result<Surface, String> {
        let n = self.name(id);
        let s = if self.is(id, "PLANE") {
            Surface::Plane { frame: self.frame(Ctx::r(&self.part(id, "PLANE")?[1])?)? }
        } else if self.is(id, "CYLINDRICAL_SURFACE") {
            let a = self.part(id, "CYLINDRICAL_SURFACE")?;
            Surface::Cylinder { frame: self.frame(Ctx::r(&a[1])?)?, r: Ctx::num(&a[2])? * self.unit }
        } else if self.is(id, "CONICAL_SURFACE") {
            let a = self.part(id, "CONICAL_SURFACE")?;
            Surface::Cone { frame: self.frame(Ctx::r(&a[1])?)?, r: Ctx::num(&a[2])? * self.unit, semi_angle: Ctx::num(&a[3])? * self.angle }
        } else if self.is(id, "SPHERICAL_SURFACE") {
            let a = self.part(id, "SPHERICAL_SURFACE")?;
            Surface::Sphere { frame: self.frame(Ctx::r(&a[1])?)?, r: Ctx::num(&a[2])? * self.unit }
        } else if self.is(id, "TOROIDAL_SURFACE") || self.is(id, "DEGENERATE_TOROIDAL_SURFACE") {
            let (a, inner) = match self.part(id, "TOROIDAL_SURFACE") {
                Ok(a) => (a, false),
                Err(_) => {
                    let a = self.part(id, "DEGENERATE_TOROIDAL_SURFACE")?;
                    let outer = a.get(4).map_or(true, Ctx::flag);
                    (a, !outer)
                }
            };
            let major = Ctx::num(&a[2])? * self.unit;
            Surface::Torus { frame: self.frame(Ctx::r(&a[1])?)?, major: if inner { -major } else { major }, minor: Ctx::num(&a[3])? * self.unit }
        } else if let Some(&(form, kind)) = [
            ("B_SPLINE_SURFACE_WITH_KNOTS", Knots::Listed),
            ("UNIFORM_SURFACE", Knots::Uniform),
            ("QUASI_UNIFORM_SURFACE", Knots::QuasiUniform),
            ("BEZIER_SURFACE", Knots::Bezier),
        ]
        .iter()
        .find(|(f, _)| self.is(id, f))
        {
            // Simple: (name, du, dv, points, form, u_closed, v_closed, self_int[, u_mult,
            // v_mult, u_knots, v_knots, spec]). Complex: B_SPLINE_SURFACE(du, dv, points,
            // ...) and the form's own record, (u_mult, v_mult, u_knots, v_knots, spec) or
            // nothing; the other forms imply their knots.
            let kw = self.part(id, form)?;
            let complex = self.is(id, "B_SPLINE_SURFACE");
            let (du, dv, rows, o2) = if complex {
                let bs = self.part(id, "B_SPLINE_SURFACE")?;
                (&bs[0], &bs[1], &bs[2], 0)
            } else {
                (&kw[1], &kw[2], &kw[3], 8)
            };
            let ctrl = Ctx::list(rows)?
                .iter()
                .map(|row| Ctx::list(row)?.iter().map(|v| self.point(Ctx::r(v)?)).collect::<Result<Vec<_>, _>>())
                .collect::<Result<Vec<_>, _>>()?;
            let (nu, nv) = (ctrl.len(), ctrl.first().map_or(0, |r| r.len()));
            let (du, dv) = (Ctx::num(du)? as usize, Ctx::num(dv)? as usize);
            let (ku, kv) = match kind {
                Knots::Listed => (Ctx::knots(&kw[o2], &kw[o2 + 2])?, Ctx::knots(&kw[o2 + 1], &kw[o2 + 3])?),
                _ => {
                    let bad = || format!("#{id}: {nu} x {nv} control points do not make a {form} of degrees {du}, {dv}");
                    (kind.implied(nu, du).ok_or_else(bad)?, kind.implied(nv, dv).ok_or_else(bad)?)
                }
            };
            let weights = if self.is(id, "RATIONAL_B_SPLINE_SURFACE") {
                let w = self.part(id, "RATIONAL_B_SPLINE_SURFACE")?;
                let l = Ctx::list(w.last().unwrap())?;
                Some(
                    l.iter()
                        .map(|row| Ctx::list(row)?.iter().map(Ctx::num).collect::<Result<Vec<_>, _>>())
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
            Surface::BSpline(BSplineSurface { du, dv, ctrl, weights, ku, kv })
        } else if self.is(id, "SURFACE_OF_LINEAR_EXTRUSION") {
            let a = self.part(id, "SURFACE_OF_LINEAR_EXTRUSION")?;
            let v = self.part(Ctx::r(&a[2])?, "VECTOR")?;
            Surface::Extrusion { curve: self.curve(Ctx::r(&a[1])?)?, dir: self.dir(Ctx::r(&v[1])?)?.scale(Ctx::num(&v[2])? * self.unit) }
        } else if self.is(id, "SURFACE_OF_REVOLUTION") {
            let a = self.part(id, "SURFACE_OF_REVOLUTION")?;
            let ax = self.part(Ctx::r(&a[2])?, "AXIS1_PLACEMENT")?;
            Surface::Revolution {
                curve: self.curve(Ctx::r(&a[1])?)?,
                axis_o: self.point(Ctx::r(&ax[1])?)?,
                axis: match &ax[2] {
                    Val::Ref(r) => self.dir(*r)?,
                    _ => V3::new(0.0, 0.0, 1.0),
                },
            }
        } else {
            Surface::Other(n)
        };
        Ok(s)
    }
    fn vertex(&self, id: u32) -> Result<V3, String> {
        let a = self.part(id, "VERTEX_POINT")?;
        self.point(Ctx::r(&a[1])?)
    }
    fn face(&self, id: u32) -> Result<BFace, String> {
        if self.is(id, "ORIENTED_FACE") {
            // The face element, turned over when the orientation is false: its normal
            // and every bound reversed.
            let a = self.part(id, "ORIENTED_FACE")?;
            let mut f = self.face(Ctx::r(&a[2])?)?;
            if !Ctx::flag(&a[3]) {
                f.same_sense = !f.same_sense;
                for b in &mut f.bounds {
                    reverse_loop(&mut b.edges);
                }
            }
            f.id = id;
            return Ok(f);
        }
        let a = self.part(id, "ADVANCED_FACE").or_else(|_| self.part(id, "FACE_SURFACE"))?;
        let surface = self.surface(Ctx::r(&a[2])?)?;
        let mut same_sense = Ctx::flag(&a[3]);
        let mut bounds = Vec::new();
        for b in Ctx::list(&a[1])? {
            let bid = Ctx::r(b)?;
            let outer = self.is(bid, "FACE_OUTER_BOUND");
            let ba = self.part(bid, if outer { "FACE_OUTER_BOUND" } else { "FACE_BOUND" })?;
            let lid = Ctx::r(&ba[1])?;
            let orient = Ctx::flag(&ba[2]);
            if self.is(lid, "VERTEX_LOOP") {
                let v = self.vertex(Ctx::r(&self.part(lid, "VERTEX_LOOP")?[1])?)?;
                bounds.push(Bound { outer, edges: Vec::new(), vertex: Some(v) });
                continue;
            }
            let mut edges = if self.is(lid, "POLY_LOOP") { self.poly_loop(lid)? } else { self.edge_loop(lid)? };
            if !orient {
                reverse_loop(&mut edges);
            }
            bounds.push(Bound { outer, edges, vertex: None });
        }
        let (surface, flip) = resolve_swept(surface, &bounds);
        same_sense ^= flip;
        Ok(BFace { id, surface, same_sense, bounds })
    }
    fn edge_loop(&self, lid: u32) -> Result<Vec<OEdge>, String> {
        let la = self.part(lid, "EDGE_LOOP")?;
        let mut edges = Vec::new();
        for oe in Ctx::list(&la[1])? {
            let oid = Ctx::r(oe)?;
            let o = self.part(oid, "ORIENTED_EDGE")?;
            let eid = Ctx::r(&o[3])?;
            let sense = Ctx::flag(&o[4]);
            let ec = self.part(eid, "EDGE_CURVE")?;
            let (vs, ve) = (Ctx::r(&ec[1])?, Ctx::r(&ec[2])?);
            let (ps, pe) = (self.vertex(vs)?, self.vertex(ve)?);
            let curve = match self.curve(Ctx::r(&ec[3])?)? {
                // A conic with no closed form here: the edge's arc of it, exactly.
                Curve3::Parabola { frame, f } => Curve3::BSpline(conic_arc(&frame, Conic::Parabola(f), ps, pe)),
                Curve3::Hyperbola { frame, a, b } => Curve3::BSpline(conic_arc(&frame, Conic::Hyperbola(a, b), ps, pe)),
                c => c,
            };
            let same = Ctx::flag(&ec[4]);
            let (start, end, sv, ev) = if sense { (ps, pe, vs, ve) } else { (pe, ps, ve, vs) };
            edges.push(OEdge { edge: eid, start, end, start_vertex: sv, end_vertex: ev, curve, forward: sense == same });
        }
        Ok(edges)
    }
    /// A polygon's sides as line edges between its points (the points are its
    /// vertices). Two faces share a side when they share its two points; the side's
    /// line runs from the lower-numbered point.
    fn poly_loop(&self, lid: u32) -> Result<Vec<OEdge>, String> {
        let pts = Ctx::list(&self.part(lid, "POLY_LOOP")?[1])?.iter().map(Ctx::r).collect::<Result<Vec<_>, _>>()?;
        let mut edges = Vec::new();
        for k in 0..pts.len() {
            let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let edge = *self.poly.borrow_mut().entry(key).or_insert_with(|| {
                let n = self.next_id.get();
                self.next_id.set(n + 1);
                n
            });
            let (lo, hi) = (self.point(key.0)?, self.point(key.1)?);
            let (start, end) = if a == key.0 { (lo, hi) } else { (hi, lo) };
            edges.push(OEdge { edge, start, end, start_vertex: a, end_vertex: b, curve: Curve3::Line { p: lo, dir: (hi - lo).unit() }, forward: a == key.0 });
        }
        Ok(edges)
    }
    /// The faces of a MANIFOLD_SOLID_BREP or FACETED_BREP (its CLOSED_SHELL).
    fn solid(&self, id: u32) -> Result<Solid, String> {
        let key = ["MANIFOLD_SOLID_BREP", "FACETED_BREP", "BREP_WITH_VOIDS"].into_iter().find(|k| self.is(id, k)).ok_or(format!("#{id}: not a solid"))?;
        if key == "BREP_WITH_VOIDS" {
            return err(id, "solid with voids is not supported yet");
        }
        let a = self.part(id, key)?;
        let name = match &a[0] {
            Val::Str(s) => s.clone(),
            _ => String::new(),
        };
        let shell = Ctx::r(&a[1])?;
        let sa = self.part(shell, "CLOSED_SHELL")?;
        let faces = Ctx::list(&sa[1])?.iter().map(|f| self.face(Ctx::r(f)?)).collect::<Result<Vec<_>, _>>()?;
        Ok(Solid { id, name, faces })
    }
    /// The motion an ITEM_DEFINED_TRANSFORMATION makes, placing a representation
    /// holding `child` items in one holding `parent` items: each of its two placements
    /// is matched to the representation it belongs to (writers differ on the order);
    /// when neither tells, the first belongs to the child when `first_is_child`.
    fn transformation(&self, id: u32, child: &HashSet<u32>, parent: &HashSet<u32>, first_is_child: bool) -> Result<Rigid, String> {
        let a = self.part(id, "ITEM_DEFINED_TRANSFORMATION").map_err(|_| format!("#{id}: transformation {} is not supported", self.name(id)))?;
        let (i1, i2) = (Ctx::r(&a[2])?, Ctx::r(&a[3])?);
        let one_is_child = if child.contains(&i1) != child.contains(&i2) {
            child.contains(&i1)
        } else if parent.contains(&i1) != parent.contains(&i2) {
            parent.contains(&i2)
        } else {
            first_is_child
        };
        let (c, p) = if one_is_child { (i1, i2) } else { (i2, i1) };
        Ok(Rigid::between(&self.frame(c)?, &self.frame(p)?))
    }
}

/// Reverses a loop: its edges in the other order, each traversed the other way.
fn reverse_loop(edges: &mut [OEdge]) {
    edges.reverse();
    for e in edges {
        std::mem::swap(&mut e.start, &mut e.end);
        std::mem::swap(&mut e.start_vertex, &mut e.end_vertex);
        e.forward = !e.forward;
    }
}

/// The same curve run the other way.
fn reversed(c: Curve3) -> Curve3 {
    // A conic turned over its x axis (z and so y reversed) runs the other way.
    let over = |f: Frame| Frame { o: f.o, z: -f.z, x: f.x };
    match c {
        Curve3::Line { p, dir } => Curve3::Line { p, dir: -dir },
        Curve3::Circle { frame, r } => Curve3::Circle { frame: over(frame), r },
        Curve3::Ellipse { frame, a, b } => Curve3::Ellipse { frame: over(frame), a, b },
        Curve3::Parabola { frame, f } => Curve3::Parabola { frame: over(frame), f },
        Curve3::Hyperbola { frame, a, b } => Curve3::Hyperbola { frame: over(frame), a, b },
        Curve3::BSpline(mut b) => {
            b.ctrl.reverse();
            if let Some(w) = b.weights.as_mut() {
                w.reverse();
            }
            let (k0, k1) = (b.knots[0], b.knots[b.knots.len() - 1]);
            b.knots = b.knots.iter().rev().map(|&k| k0 + k1 - k).collect();
            Curve3::BSpline(b)
        }
        other => other,
    }
}

/// How a B-spline's knots are given: listed, or implied by its form (ISO 10303-42).
#[derive(Clone, Copy, Debug)]
enum Knots {
    Listed,
    /// -degree, -degree + 1, ... in unit steps, each once.
    Uniform,
    /// 0, 1, ..., n - degree, the ends degree + 1 times.
    QuasiUniform,
    /// 0, 1, ..., segments, the ends degree + 1 times and the rest degree times.
    Bezier,
}

impl Knots {
    /// The knots of `n` control points at `degree`, or None if they make no curve of the form.
    fn implied(self, n: usize, degree: usize) -> Option<Vec<f64>> {
        let d = degree;
        if d == 0 || n < d + 1 {
            return None;
        }
        let ends = |last: usize, inner: usize| {
            let mut k = vec![0.0; d + 1];
            for i in 1..last {
                k.extend(std::iter::repeat(i as f64).take(inner));
            }
            k.extend(std::iter::repeat(last as f64).take(d + 1));
            k
        };
        match self {
            Knots::Listed => None,
            Knots::Uniform => Some((0..n + d + 1).map(|i| i as f64 - d as f64).collect()),
            Knots::QuasiUniform => Some(ends(n - d, 1)),
            Knots::Bezier if (n - 1) % d == 0 => Some(ends((n - 1) / d, d)),
            Knots::Bezier => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Conic {
    /// Focal distance.
    Parabola(f64),
    /// Semi-axis and semi-imaginary axis.
    Hyperbola(f64, f64),
}

/// The arc of a parabola or hyperbola between two of its points, in the curve's own
/// direction, as exact rational quadratic Bezier pieces (a hyperbola's no longer than
/// one unit of its parameter each).
fn conic_arc(f: &Frame, c: Conic, p: V3, q: V3) -> BSplineCurve {
    let y = f.y();
    let param = |x: V3| {
        let ly = (x - f.o).dot(y);
        match c {
            Conic::Parabola(fd) => ly / (2.0 * fd),
            Conic::Hyperbola(_, b) => (ly / b).asinh(),
        }
    };
    let (t0, t1) = {
        let (a, b) = (param(p), param(q));
        (a.min(b), a.max(b))
    };
    let at = |t: f64| match c {
        Conic::Parabola(fd) => f.o + f.x * (fd * t * t) + y * (2.0 * fd * t),
        Conic::Hyperbola(a, b) => f.o + f.x * (a * t.cosh()) + y * (b * t.sinh()),
    };
    let n = match c {
        Conic::Parabola(_) => 1,
        Conic::Hyperbola(..) => ((t1 - t0).ceil() as usize).max(1),
    };
    let (mut ctrl, mut weights, mut knots) = (vec![at(t0)], vec![1.0], vec![0.0; 3]);
    for k in 0..n {
        let (s0, s1) = (t0 + (t1 - t0) * k as f64 / n as f64, t0 + (t1 - t0) * (k + 1) as f64 / n as f64);
        // The middle control point is where the end tangents meet.
        let (mid, w) = match c {
            Conic::Parabola(fd) => (at(s0) + (f.x * (2.0 * fd * s0) + y * (2.0 * fd)) * (0.5 * (s1 - s0)), 1.0),
            Conic::Hyperbola(a, b) => {
                let (m, h) = (0.5 * (s0 + s1), 0.5 * (s1 - s0));
                (f.o + f.x * (a * m.cosh() / h.cosh()) + y * (b * m.sinh() / h.cosh()), h.cosh())
            }
        };
        ctrl.extend([mid, at(s1)]);
        weights.extend([w, 1.0]);
        knots.extend(std::iter::repeat((k + 1) as f64).take(if k + 1 == n { 3 } else { 2 }));
    }
    BSplineCurve { degree: 2, ctrl, weights: Some(weights), knots }
}

/// The unit circle as nine rational quadratic control points (x, y, weight), four
/// quarter arcs counterclockwise from (1, 0), and its knots.
fn unit_circle() -> ([(f64, f64, f64); 9], [f64; 12]) {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    (
        [(1.0, 0.0, 1.0), (1.0, 1.0, s), (0.0, 1.0, 1.0), (-1.0, 1.0, s), (-1.0, 0.0, 1.0), (-1.0, -1.0, s), (0.0, -1.0, 1.0), (1.0, -1.0, s), (1.0, 0.0, 1.0)],
        [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0],
    )
}

/// A bounded curve as an exact rational B-spline: (degree, control points, weights, knots).
fn as_nurbs(c: &Curve3) -> Result<(usize, Vec<V3>, Vec<f64>, Vec<f64>), String> {
    match c {
        Curve3::BSpline(b) => Ok((b.degree, b.ctrl.clone(), b.weights.clone().unwrap_or_else(|| vec![1.0; b.ctrl.len()]), b.knots.clone())),
        Curve3::Circle { frame, r } => as_nurbs(&Curve3::Ellipse { frame: *frame, a: *r, b: *r }),
        Curve3::Ellipse { frame, a, b } => {
            let (pts, knots) = unit_circle();
            let y = frame.y();
            Ok((2, pts.iter().map(|&(cx, cy, _)| frame.o + frame.x * (a * cx) + y * (b * cy)).collect(), pts.iter().map(|p| p.2).collect(), knots.to_vec()))
        }
        other => Err(format!("swept {} profile", crate::prismatic::curve_name(other))),
    }
}

/// The range of `along` over a face, from its vertices and the hulls of its edges.
fn extent(bounds: &[Bound], along: V3) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut take = |x: f64| {
        lo = lo.min(x);
        hi = hi.max(x);
    };
    for b in bounds {
        if let Some(v) = b.vertex {
            take(v.dot(along));
        }
        for e in &b.edges {
            take(e.start.dot(along));
            take(e.end.dot(along));
            match &e.curve {
                Curve3::Circle { frame, r } => [-1.0, 1.0].into_iter().for_each(|s| take(frame.o.dot(along) + s * r * along.len())),
                Curve3::Ellipse { frame, a, b } => [-1.0, 1.0].into_iter().for_each(|s| take(frame.o.dot(along) + s * a.max(*b) * along.len())),
                Curve3::BSpline(bc) => bc.ctrl.iter().for_each(|q| take(q.dot(along))),
                _ => {}
            }
        }
    }
    (lo, hi)
}

/// A SURFACE_OF_LINEAR_EXTRUSION or SURFACE_OF_REVOLUTION as the analytic surface it
/// is (plane, cylinder, cone, torus) or else as an exact rational B-spline surface
/// over the face, and whether the face's sense flips to keep its outward side: the
/// swept surface's own normal is d/du x d/dv of its parameterization (u along the
/// profile and v along the extrusion; u the turn and v along the profile for a
/// revolution), which the analytic surface's need not match. Other surfaces pass
/// through; a sweep that cannot be read becomes `Other` with the reason.
/// A B-spline surface that is a plane, sphere, circular cylinder, circular cone or torus
/// (every span's numbers differ from it only by rounding: `span_poly::recognize`)
/// as that analytic surface, and whether the face's sense flips to keep its
/// outward side. Derek, 2026-10-25: such a span is the analytic surface, and its
/// field is the analytic one (a sphere's is the distance from its center less its
/// radius). Other surfaces pass through.
fn resolve_bspline(s: Surface) -> (Surface, bool) {
    use crate::surface3::Surf;
    let Surface::BSpline(b) = &s else { return (s, false) };
    let Ok(ns) = crate::surface3::NurbsSurf::new(b.du, b.dv, &b.ctrl, b.weights.as_ref(), &b.ku, &b.kv) else { return (s, false) };
    match crate::span_poly::recognize(&ns) {
        Some((Surf::Plane { f }, flip)) => (Surface::Plane { frame: f }, flip),
        Some((Surf::Sphere { f, r }, flip)) => (Surface::Sphere { frame: f, r }, flip),
        Some((Surf::Cylinder { f, r }, flip)) => (Surface::Cylinder { frame: f, r }, flip),
        Some((Surf::Cone { f, r, a }, flip)) => (Surface::Cone { frame: f, r, semi_angle: a }, flip),
        Some((Surf::Torus { f, major, minor }, flip)) => (Surface::Torus { frame: f, major, minor }, flip),
        _ => (s, false),
    }
}

fn resolve_swept(s: Surface, bounds: &[Bound]) -> (Surface, bool) {
    let r = match &s {
        Surface::Extrusion { curve, dir } => extrusion(curve, *dir, bounds),
        Surface::Revolution { curve, axis_o, axis } => revolution(curve, *axis_o, axis.unit(), bounds),
        _ => return (s, false),
    };
    r.unwrap_or_else(|e| (Surface::Other(e), false))
}

fn extrusion(curve: &Curve3, v: V3, bounds: &[Bound]) -> Result<(Surface, bool), String> {
    let vu = v.unit();
    match curve {
        // p + s dir + t v: normal dir x v.
        Curve3::Line { p, dir } if dir.cross(vu).len() > 1e-9 => Ok((Surface::Plane { frame: Frame { o: *p, z: dir.cross(vu).unit(), x: dir.unit() } }, false)),
        // (z x radial) x v is radial: outward when v runs along the circle's z.
        Curve3::Circle { frame, r } if frame.z.cross(vu).len() < 1e-12 => Ok((Surface::Cylinder { frame: *frame, r: *r }, frame.z.dot(v) < 0.0)),
        Curve3::Line { .. } => Err("extrusion of a line along itself".into()),
        _ => {
            let (d, ctrl, w, ku) = as_nurbs(curve)?;
            // The face's span along v: its extent along v less the profile's own.
            let (flo, fhi) = extent(bounds, vu);
            let (plo, phi) = ctrl.iter().map(|q| q.dot(vu)).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
            let (t0, t1) = ((flo - phi) / v.len(), (fhi - plo) / v.len());
            if !(t0 < t1) {
                return Err("extrusion with no extent".into());
            }
            let pad = 0.01 * (t1 - t0);
            let (t0, t1) = (t0 - pad, t1 + pad);
            let rows = ctrl.iter().map(|&q| vec![q + v * t0, q + v * t1]).collect();
            let weights = w.iter().map(|&x| vec![x, x]).collect();
            Ok((Surface::BSpline(BSplineSurface { du: d, dv: 1, ctrl: rows, weights: Some(weights), ku, kv: vec![t0, t0, t1, t1] }), false))
        }
    }
}

fn revolution(curve: &Curve3, c: V3, v: V3, bounds: &[Bound]) -> Result<(Surface, bool), String> {
    // Radial part of a point about the axis, and the face point farthest out.
    let radial = |p: V3| {
        let w = p - c;
        w - v * w.dot(v)
    };
    let far = bounds
        .iter()
        .flat_map(|b| b.vertex.into_iter().chain(b.edges.iter().map(|e| e.start)))
        .max_by(|a, b| radial(*a).len().total_cmp(&radial(*b).len()))
        .ok_or("revolution face with no vertex")?;
    let any_x = |r: V3| if r.len() > 0.0 { r.unit() } else if v.x.abs() < 0.9 { v.cross(V3::new(1.0, 0.0, 0.0)).unit() } else { v.cross(V3::new(0.0, 1.0, 0.0)).unit() };
    // The swept normal at profile point q with profile tangent t: (v x radial) x t.
    let swept_n = |q: V3, t: V3| v.cross(radial(q)).cross(t);
    match curve {
        Curve3::Line { p, dir } => {
            let d = dir.unit();
            let (dr, rp) = (d - v * d.dot(v), radial(*p));
            if dr.len() < 1e-12 {
                // Parallel to the axis: a cylinder, its radial normal outward when the
                // line runs along the axis.
                let frame = Frame { o: c, z: v, x: any_x(rp) };
                return Ok((Surface::Cylinder { frame, r: rp.len() }, d.dot(v) < 0.0));
            }
            if d.dot(v).abs() < 1e-12 {
                // Square to the axis: a plane, normal -v (radial . d).
                let rd = radial(far).dot(d);
                let z = if rd > 0.0 { -v } else { v };
                return Ok((Surface::Plane { frame: Frame { o: *p, z, x: any_x(rp) } }, false));
            }
            if (*p - c).cross(d).dot(v).abs() > 1e-9 * (1.0 + (*p - c).len()) {
                return Err("revolution of a line skew to its axis".into());
            }
            // A cone: the apex where the line meets the axis, the frame at the far
            // point's height on the side the face is on.
            let apex = *p - d * (rp.dot(dr) / dr.dot(dr));
            let h = (far - apex).dot(v);
            let rho = radial(far).len();
            if h.abs() < 1e-12 || rho < 1e-12 {
                return Err("revolution cone face with no extent".into());
            }
            let z = if h > 0.0 { v } else { -v };
            let frame = Frame { o: apex + z * h.abs(), z, x: any_x(radial(far)) };
            // The cone's own normal points out from the axis, as the swept one does
            // when the line runs along the axis.
            Ok((Surface::Cone { frame, r: rho, semi_angle: (rho / h.abs()).atan() }, d.dot(v) < 0.0))
        }
        Curve3::Circle { frame: cf, r } if cf.z.dot(v).abs() < 1e-9 && (c - cf.o).dot(cf.z).abs() < 1e-9 * (1.0 + (c - cf.o).len()) => {
            // A circle in a plane holding the axis: a torus (a sphere when centered on
            // the axis, which is left to the B-spline form: one half of the circle
            // covers the sphere outward and the other inward, and the face does not
            // say which).
            let rc = radial(cf.o);
            if rc.len() < 1e-9 * (1.0 + r) {
                return revolution_nurbs(curve, c, v);
            }
            let er = rc.unit();
            let o = c + v * (cf.o - c).dot(v);
            // Apple or lemon: whichever circle about +-major the face point lies on.
            let major = if rc.len() < *r {
                let (rho, h) = (radial(far).len(), (far - o).dot(v));
                let off = |m: f64| ((rho - m).hypot(h) - r).abs();
                if off(-rc.len()) < off(rc.len()) {
                    -rc.len()
                } else {
                    rc.len()
                }
            } else {
                rc.len()
            };
            // The torus normal points away from the circle's center in each half-plane;
            // the swept one, at the circle's point on the part's side, may not.
            let q = cf.o + er * (if major < 0.0 { -r } else { *r });
            let t = cf.z.cross(q - cf.o);
            let flip = swept_n(q, t).dot(q - cf.o) < 0.0;
            Ok((Surface::Torus { frame: Frame { o, z: v, x: er }, major, minor: *r }, flip))
        }
        _ => revolution_nurbs(curve, c, v),
    }
}

/// A profile revolved once around the axis (c, v) as an exact rational B-spline
/// surface: u the turn (the unit circle's nine points about each profile point's
/// foot on the axis), v along the profile.
fn revolution_nurbs(curve: &Curve3, c: V3, v: V3) -> Result<(Surface, bool), String> {
    let (d, ctrl, w, kv) = as_nurbs(curve)?;
    let (circle, ku) = unit_circle();
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    for &(cx, cy, cw) in &circle {
        let mut row = Vec::new();
        let mut wr = Vec::new();
        for (q, &wq) in ctrl.iter().zip(&w) {
            let foot = c + v * (*q - c).dot(v);
            let r = *q - foot;
            row.push(foot + r * cx + v.cross(r) * cy);
            wr.push(cw * wq);
        }
        rows.push(row);
        weights.push(wr);
    }
    Ok((Surface::BSpline(BSplineSurface { du: 2, dv: d, ctrl: rows, weights: Some(weights), ku: ku.to_vec(), kv }), false))
}

/// The unit of a kind (`LENGTH_UNIT`, `PLANE_ANGLE_UNIT`) that a
/// GLOBAL_UNIT_ASSIGNED_CONTEXT assigns (lowest entity id first, so the answer never
/// depends on hash order), else the lowest-numbered unit of that kind in the file.
fn assigned_unit(e: &Entities, kind: &str) -> Option<u32> {
    let mut ids: Vec<u32> = e.keys().copied().collect();
    ids.sort();
    let is_kind = |id: u32| e.get(&id).is_some_and(|p| p.iter().any(|(n, _)| n == kind));
    for &id in &ids {
        if let Some((_, a)) = e[&id].iter().find(|(n, _)| n == "GLOBAL_UNIT_ASSIGNED_CONTEXT") {
            if let Some(Val::List(units)) = a.first() {
                for u in units {
                    if let Val::Ref(r) = u {
                        if is_kind(*r) {
                            return Some(*r);
                        }
                    }
                }
            }
        }
    }
    ids.iter().copied().find(|&id| is_kind(id))
}

/// Radians per plane angle unit: 1 for the SI radian or no unit at all; a
/// CONVERSION_BASED_UNIT ('DEGREE', most often) is the size its
/// PLANE_ANGLE_MEASURE_WITH_UNIT gives, in radians, or pi / 180 for a degree that
/// gives none.
fn angle_unit(e: &Entities) -> f64 {
    let Some(uid) = assigned_unit(e, "PLANE_ANGLE_UNIT") else {
        return 1.0;
    };
    let Some((_, a)) = e[&uid].iter().find(|(n, _)| n == "CONVERSION_BASED_UNIT") else {
        return 1.0;
    };
    let named = |n: &str| matches!(a.first(), Some(Val::Str(s)) if s.eq_ignore_ascii_case(n));
    let size = match a.get(1) {
        Some(Val::Ref(m)) => e.get(m).and_then(|p| p.iter().find(|(n, _)| n == "PLANE_ANGLE_MEASURE_WITH_UNIT")).and_then(|(_, v)| match v.first() {
            Some(Val::Typed(_, x)) => x.first().and_then(|x| Ctx::num(x).ok()),
            Some(x) => Ctx::num(x).ok(),
            None => None,
        }),
        _ => None,
    };
    match size {
        Some(x) if x > 0.0 => x,
        _ if named("DEGREE") || named("DEGREES") => std::f64::consts::PI / 180.0,
        _ => 1.0,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn angle(units: &str) -> f64 {
        let text = format!("ISO-10303-21;\nDATA;\n{units}\nENDSEC;\nEND-ISO-10303-21;\n");
        angle_unit(&parse_data(text.as_bytes()).unwrap())
    }

    /// Degrees, sized by their measure or by name alone; the SI radian; and no unit.
    #[test]
    fn plane_angle_units() {
        let degree = "#3=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n\
            #4=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925),#3);\n\
            #5=(CONVERSION_BASED_UNIT('DEGREE',#4)NAMED_UNIT(#6)PLANE_ANGLE_UNIT());\n\
            #6=DIMENSIONAL_EXPONENTS(0.,0.,0.,0.,0.,0.,0.);\n\
            #10=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#5))REPRESENTATION_CONTEXT('',''));";
        assert!((angle(degree) - 0.0174532925).abs() < 1e-15);
        let named = "#5=(CONVERSION_BASED_UNIT('DEGREE',$)NAMED_UNIT(*)PLANE_ANGLE_UNIT());\n\
            #10=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#5))REPRESENTATION_CONTEXT('',''));";
        assert!((angle(named) - std::f64::consts::PI / 180.0).abs() < 1e-15);
        let radian = "#3=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n\
            #10=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#3))REPRESENTATION_CONTEXT('',''));";
        assert_eq!(angle(radian), 1.0);
        assert_eq!(angle("#1=CARTESIAN_POINT('',(0.,0.,0.));"), 1.0);
    }
}

#[cfg(test)]
mod reader_tests {
    use super::*;

    fn ents(data: &str) -> Entities {
        parse_data(format!("ISO-10303-21;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n").as_bytes()).unwrap()
    }

    fn ctx(e: &Entities) -> Ctx<'_> {
        Ctx { e, unit: 1.0, angle: 1.0, poly: RefCell::new(HashMap::new()), next_id: Cell::new(1_000_000) }
    }

    fn close(a: V3, b: V3, tol: f64) -> bool {
        (a - b).len() <= tol
    }

    /// The knots each B-spline form implies (ISO 10303-42), and the forms a count of
    /// control points cannot make.
    #[test]
    fn implied_knots() {
        assert_eq!(Knots::QuasiUniform.implied(5, 3).unwrap(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(Knots::Bezier.implied(7, 3).unwrap(), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(Knots::Uniform.implied(4, 2).unwrap(), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(Knots::Bezier.implied(6, 3).is_none());
        assert!(Knots::QuasiUniform.implied(3, 3).is_none());
    }

    /// Short entity names read as the long ones.
    #[test]
    fn short_names() {
        let e = ents("#1=CRTPNT('',(1.,2.,3.));\n#2=(LNGUNT()NMDUNT(*)SUNT(.MILLI.,.METRE.));");
        assert_eq!(e[&1][0].0, "CARTESIAN_POINT");
        assert_eq!(e[&2].iter().map(|p| p.0.as_str()).collect::<Vec<_>>(), ["LENGTH_UNIT", "NAMED_UNIT", "SI_UNIT"]);
        assert_eq!(ctx(&e).point(1).unwrap(), V3::new(1.0, 2.0, 3.0));
    }

    /// An arc of a parabola or hyperbola read as a B-spline lies on the conic, from
    /// one end to the other.
    #[test]
    fn conic_arcs_lie_on_their_conics() {
        let f = Frame { o: V3::new(1.0, -2.0, 0.5), z: V3::new(0.0, 0.6, 0.8), x: V3::new(1.0, 0.0, 0.0) };
        let y = f.y();
        let local = |q: V3| ((q - f.o).dot(f.x), (q - f.o).dot(y));
        let cases: [(Conic, Box<dyn Fn(f64) -> V3>, Box<dyn Fn(f64, f64) -> f64>); 2] = [
            (Conic::Parabola(0.7), Box::new(move |t: f64| f.o + f.x * (0.7 * t * t) + y * (1.4 * t)), Box::new(|lx: f64, ly: f64| lx - ly * ly / 2.8)),
            (Conic::Hyperbola(2.0, 0.5), Box::new(move |t: f64| f.o + f.x * (2.0 * t.cosh()) + y * (0.5 * t.sinh())), Box::new(|lx: f64, ly: f64| (lx / 2.0).powi(2) - (ly / 0.5).powi(2) - 1.0)),
        ];
        for (c, at, eq) in &cases {
            for (t0, t1) in [(-1.3, 2.4), (2.4, -1.3), (0.2, 0.3)] {
                let (p, q) = (at(t0), at(t1));
                let b = conic_arc(&f, *c, p, q);
                let e = OEdge { edge: 1, start: p, end: q, start_vertex: 1, end_vertex: 2, curve: Curve3::BSpline(b), forward: t0 < t1 };
                let (edge, off) = crate::curve3::Edge3::from_step(&e).unwrap();
                assert!(off < 1e-9, "{c:?}: ends {off} off the arc");
                assert!(close(edge.at(0.0), p, 1e-9) && close(edge.at(1.0), q, 1e-9), "{c:?}: ends moved");
                for k in 0..=50 {
                    let (lx, ly) = local(edge.at(k as f64 / 50.0));
                    assert!(eq(lx, ly).abs() < 1e-9, "{c:?} from {t0} to {t1}: off the conic by {}", eq(lx, ly));
                }
            }
        }
    }

    /// The motion between two frames takes one onto the other, and motions compose.
    #[test]
    fn rigid_motions() {
        let a = Frame { o: V3::new(1.0, 2.0, 3.0), z: V3::new(0.0, 0.0, 1.0), x: V3::new(0.6, 0.8, 0.0) };
        let b = Frame { o: V3::new(-4.0, 0.5, 2.0), z: V3::new(0.0, 0.6, 0.8), x: V3::new(1.0, 0.0, 0.0) };
        let m = Rigid::between(&a, &b);
        assert!(close(m.point(a.o), b.o, 1e-12) && close(m.turn(a.x), b.x, 1e-12) && close(m.turn(a.z), b.z, 1e-12) && close(m.turn(a.y()), b.y(), 1e-12));
        let back = Rigid::between(&b, &a);
        let p = V3::new(0.3, -7.0, 2.5);
        assert!(close(back.after(&m).point(p), p, 1e-12));
        assert!(close(m.after(&back).point(p), p, 1e-12));
    }

    /// A swept face read as an analytic or B-spline surface keeps its outward side:
    /// the converted surface's normal, flipped when the conversion says so, runs
    /// with the sweep's own normal d/du x d/dv.
    #[test]
    fn swept_surfaces_keep_their_normal() {
        let z = V3::new(0.0, 0.0, 1.0);
        let face = |pts: &[V3]| pts.iter().map(|&p| Bound { outer: true, edges: Vec::new(), vertex: Some(p) }).collect::<Vec<_>>();
        let rot = |q: V3, u: f64| V3::new(q.x * u.cos() - q.y * u.sin(), q.x * u.sin() + q.y * u.cos(), q.z);
        let check = |name: &str, s: Surface, bounds: &[Bound], sigma: &dyn Fn(f64, f64) -> V3, at: &[(f64, f64)]| {
            let (conv, flip) = resolve_swept(s, bounds);
            let surf = crate::solid3::surf_from_step(&conv).unwrap_or_else(|e| panic!("{name}: {e}"));
            for &(u, v) in at {
                let h = 1e-6;
                let n = ((sigma(u + h, v) - sigma(u - h, v)).cross(sigma(u, v + h) - sigma(u, v - h))).unit();
                let x = sigma(u, v);
                let m = surf.normal(x) * if flip { -1.0 } else { 1.0 };
                assert!((surf.foot(x) - x).len() < 1e-6, "{name}: ({u}, {v}) is off the read surface");
                assert!(n.dot(m) > 0.99, "{name} at ({u}, {v}): sweep normal {n:?}, read {m:?}");
            }
        };
        let o = V3::default();
        // Revolutions about z: sigma(u, v) is the profile point at v turned by u.
        for d in [1.0, -1.0] {
            let p = V3::new(2.0, 0.0, 0.0);
            let s = |u: f64, v: f64| rot(p + z * (d * v), u);
            check("cylinder", Surface::Revolution { curve: Curve3::Line { p, dir: z * d }, axis_o: o, axis: z }, &face(&[s(0.3, 1.0)]), &s, &[(0.3, 0.5), (2.0, -1.0)]);
            let (p, dir) = (V3::new(0.0, 0.0, -1.0), V3::new(1.0, 0.0, 1.0).unit() * d);
            let s = |u: f64, v: f64| rot(p + dir * v, u);
            check("cone", Surface::Revolution { curve: Curve3::Line { p, dir }, axis_o: o, axis: z }, &face(&[s(0.0, 2.0 * d), s(1.0, 3.0 * d)]), &s, &[(0.5, 2.0 * d), (4.0, 2.5 * d)]);
            let (p, dir) = (V3::new(0.0, 0.0, 1.0), V3::new(d, 0.0, 0.0));
            let s = |u: f64, v: f64| rot(p + dir * v, u);
            check("plane", Surface::Revolution { curve: Curve3::Line { p, dir }, axis_o: o, axis: z }, &face(&[s(0.0, 1.0 * d), s(0.0, 3.0 * d)]), &s, &[(0.5, 2.0 * d)]);
            // A circle in the xz plane, both ways round; ring, apple and lemon.
            for (cx, t_face, name) in [(5.0, 0.0, "ring torus"), (0.4, 0.0, "apple"), (0.4, std::f64::consts::PI, "lemon")] {
                let frame = Frame { o: V3::new(cx, 0.0, 0.0), z: V3::new(0.0, d, 0.0), x: V3::new(1.0, 0.0, 0.0) };
                let s = move |u: f64, v: f64| rot(frame.o + (frame.x * v.cos() + frame.y() * v.sin()), u);
                check(name, Surface::Revolution { curve: Curve3::Circle { frame, r: 1.0 }, axis_o: o, axis: z }, &face(&[s(0.7, t_face)]), &s, &[(0.7, t_face), (3.0, t_face + 0.2)]);
            }
            // A B-spline profile: a B-spline surface.
            let ctrl = vec![V3::new(2.0, 0.0, 0.0), V3::new(3.0, 0.0, 1.0), V3::new(2.5, 0.0, 2.0)];
            let ctrl: Vec<V3> = if d > 0.0 { ctrl } else { ctrl.into_iter().rev().collect() };
            let bc = BSplineCurve { degree: 2, ctrl: ctrl.clone(), weights: None, knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0] };
            let bez = move |v: f64| ctrl[0] * ((1.0 - v) * (1.0 - v)) + ctrl[1] * (2.0 * v * (1.0 - v)) + ctrl[2] * (v * v);
            let s = |u: f64, v: f64| rot(bez(v), u);
            check("revolved B-spline", Surface::Revolution { curve: Curve3::BSpline(bc), axis_o: o, axis: z }, &face(&[s(0.0, 0.5)]), &s, &[(0.4, 0.3), (2.5, 0.7)]);
        }
        // Extrusions: sigma(u, t) = C(u) + t V.
        for v in [z * 2.0, z * -2.0, V3::new(0.5, 0.0, 2.0)] {
            let frame = Frame { o, z, x: V3::new(1.0, 0.0, 0.0) };
            let s = move |u: f64, t: f64| frame.o + frame.x * (3.0 * u.cos()) + frame.y() * (1.0 * u.sin()) + v * t;
            check("extruded ellipse", Surface::Extrusion { curve: Curve3::Ellipse { frame, a: 3.0, b: 1.0 }, dir: v }, &face(&[s(0.0, 0.0), s(1.0, 1.0)]), &s, &[(0.3, 0.5), (2.0, 0.2)]);
            let s = move |u: f64, t: f64| frame.o + (frame.x * u.cos() + frame.y() * u.sin()) * 2.0 + v * t;
            if v.cross(z).len() == 0.0 {
                check("extruded circle", Surface::Extrusion { curve: Curve3::Circle { frame, r: 2.0 }, dir: v }, &face(&[s(0.0, 0.0), s(1.0, 1.0)]), &s, &[(0.3, 0.5), (4.0, 0.9)]);
            }
            let (p, dir) = (V3::new(1.0, 2.0, 0.0), V3::new(0.6, 0.8, 0.0));
            let s = move |u: f64, t: f64| p + dir * u + v * t;
            check("extruded line", Surface::Extrusion { curve: Curve3::Line { p, dir }, dir: v }, &face(&[s(0.0, 0.0)]), &s, &[(0.3, 0.5)]);
        }
    }

    /// An assembly places its part once per use, whichever side of the
    /// relationship the writer put the parent on and whichever order it gave the
    /// transformation's two placements in; a MAPPED_ITEM places it too.
    #[test]
    fn assembly_placement() {
        let common = "#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=DIRECTION('',(0.,0.,1.));\n#3=DIRECTION('',(1.,0.,0.));\n\
            #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);\n#5=CARTESIAN_POINT('',(10.,0.,0.));\n#7=DIRECTION('',(0.,1.,0.));\n\
            #6=AXIS2_PLACEMENT_3D('',#5,#2,#7);\n#10=MANIFOLD_SOLID_BREP('',#99);\n\
            #11=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#10,#4),#50);\n#12=SHAPE_REPRESENTATION('',(#6),#50);\n\
            #20=PRODUCT_DEFINITION('','',#90,#91);\n#21=PRODUCT_DEFINITION('','',#90,#91);\n\
            #22=PRODUCT_DEFINITION_SHAPE('','',#20);\n#23=PRODUCT_DEFINITION_SHAPE('','',#21);\n\
            #24=SHAPE_DEFINITION_REPRESENTATION(#22,#12);\n#25=SHAPE_DEFINITION_REPRESENTATION(#23,#11);\n\
            #30=NEXT_ASSEMBLY_USAGE_OCCURRENCE('','','',#20,#21,$);\n#31=PRODUCT_DEFINITION_SHAPE('','',#30);\n";
        // The part's origin goes to (10, 0, 0), its x axis to y.
        let expect = |m: &Rigid| close(m.point(V3::new(1.0, 0.0, 0.0)), V3::new(10.0, 1.0, 0.0), 1e-12);
        for (r1, r2) in [("#12", "#11"), ("#11", "#12")] {
            for (i1, i2) in [("#4", "#6"), ("#6", "#4")] {
                let e = ents(&format!(
                    "{common}#32=ITEM_DEFINED_TRANSFORMATION('','',{i1},{i2});\n\
                     #33=(REPRESENTATION_RELATIONSHIP('','',{r1},{r2})REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#32)SHAPE_REPRESENTATION_RELATIONSHIP());\n\
                     #34=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#33,#31);"
                ));
                let got = instances(&ctx(&e)).unwrap();
                assert_eq!(got.len(), 1, "rep_1 {r1}, item_1 {i1}: {got:?}");
                assert!(got[0].0 == [10] && expect(&got[0].1), "rep_1 {r1}, item_1 {i1}: {got:?}");
            }
        }
        let e = ents(&format!("{common}#40=MAPPED_ITEM('',#41,#6);\n#41=REPRESENTATION_MAP(#4,#11);\n#13=SHAPE_REPRESENTATION('',(#40),#50);"));
        let got = instances(&ctx(&e)).unwrap();
        assert!(got.iter().any(|(ids, m)| ids == &[10] && expect(m)), "{got:?}");
        assert_eq!(got.iter().filter(|(ids, _)| ids == &[10]).count(), 1, "{got:?}");
    }

    /// Single-face solids that share edges join into one; a closed solid stays apart.
    #[test]
    fn open_shells_join() {
        let face = |id: u32, edges: &[u32]| BFace {
            id,
            surface: Surface::Other(String::new()),
            same_sense: true,
            bounds: vec![Bound {
                outer: true,
                edges: edges.iter().map(|&e| OEdge { edge: e, start: V3::default(), end: V3::default(), start_vertex: 0, end_vertex: 0, curve: Curve3::Other(String::new()), forward: true }).collect(),
                vertex: None,
            }],
        };
        let solid = |id: u32, faces: Vec<BFace>| Solid { id, name: String::new(), faces };
        let out = join_open(vec![
            solid(1, vec![face(1, &[100, 101])]),
            solid(2, vec![face(2, &[101, 102])]),
            solid(3, vec![face(3, &[102, 100])]),
            solid(4, vec![face(4, &[200, 201]), face(5, &[200, 201])]),
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].faces.len(), 3);
        assert_eq!(out[1].id, 4);
    }
}

/// Millimeters per length unit (see [`assigned_unit`]).
fn length_unit(e: &Entities) -> (f64, String) {
    let Some(uid) = assigned_unit(e, "LENGTH_UNIT") else {
        return (1.0, "unspecified (taken as mm)".into());
    };
    let parts = &e[&uid];
    if let Some((_, a)) = parts.iter().find(|(n, _)| n == "SI_UNIT") {
        let prefix = match a.first() {
            Some(Val::Enum(p)) => p.as_str(),
            _ => "",
        };
        let f = match prefix {
            "MILLI" => 1.0,
            "CENTI" => 10.0,
            "DECI" => 100.0,
            "KILO" => 1e6,
            "MICRO" => 1e-3,
            _ => 1000.0,
        };
        return (f, format!("{}METRE", if prefix.is_empty() { String::new() } else { format!("{prefix} ") }));
    }
    if let Some((_, a)) = parts.iter().find(|(n, _)| n == "CONVERSION_BASED_UNIT") {
        let name = match a.first() {
            Some(Val::Str(s)) => s.to_uppercase(),
            _ => String::new(),
        };
        let f = match name.as_str() {
            "INCH" | "INCHES" => 25.4,
            "FOOT" | "FEET" => 304.8,
            "MILLIMETRE" | "MILLIMETER" => 1.0,
            _ => f64::NAN,
        };
        return (f, name);
    }
    (f64::NAN, format!("#{uid}"))
}

pub fn read(path: &std::path::Path) -> Result<Brep, String> {
    read_opts(path, true)
}

/// As `read`, but every B-spline surface stays a B-spline surface: the reference
/// that `read`'s analytic surfaces are checked against.
pub fn read_raw(path: &std::path::Path) -> Result<Brep, String> {
    read_opts(path, false)
}

fn read_opts(path: &std::path::Path, analytic: bool) -> Result<Brep, String> {
    let trace = std::env::var("ICAM_TRACE_BUILD").is_ok();
    let t0 = std::time::Instant::now();
    let lap = |what: &str| {
        if trace {
            eprintln!("  read  {:>8.1} ms  {what}", t0.elapsed().as_secs_f64() * 1e3);
        }
    };
    let text = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    lap("file");
    let ents = parse_data(&text)?;
    lap("entities parsed");
    let (unit, unit_name) = length_unit(&ents);
    if unit.is_nan() {
        return Err(format!("unknown length unit {unit_name}"));
    }
    let next = ents.keys().copied().max().unwrap_or(0) + 1;
    let cx = Ctx { e: &ents, unit, angle: angle_unit(&ents), poly: RefCell::new(HashMap::new()), next_id: Cell::new(next) };
    // Solids read once each, grouped by the motion that places them (open shells
    // close up only with others in the same place).
    let placed = instances(&cx)?;
    let mut local: HashMap<u32, Solid> = HashMap::new();
    for (ids, _) in &placed {
        for &id in ids {
            if !local.contains_key(&id) {
                local.insert(id, cx.solid(id)?);
            }
        }
    }
    lap("solids built");
    // B-spline surfaces that are exactly analytic ones, recognized face by face in
    // parallel (each face's alone decides its own).
    if analytic {
        use rayon::prelude::*;
        let mut faces: Vec<&mut BFace> = local.values_mut().flat_map(|s| s.faces.iter_mut()).collect();
        faces.par_iter_mut().for_each(|f| {
            let t = std::time::Instant::now();
            let s = std::mem::replace(&mut f.surface, Surface::Other(String::new()));
            let spans = if let Surface::BSpline(b) = &s { (b.ku.len(), b.kv.len(), b.du, b.dv) } else { (0, 0, 0, 0) };
            let (s, flip) = resolve_bspline(s);
            if trace && t.elapsed().as_secs_f64() > 0.05 {
                eprintln!("  recognize {:>7.1} ms  face #{} (knots {} x {}, degree {} x {}): {}", t.elapsed().as_secs_f64() * 1e3, f.id, spans.0, spans.1, spans.2, spans.3, match &s { Surface::BSpline(_) => "stays a B-spline", _ => "analytic" });
            }
            f.surface = s;
            f.same_sense ^= flip;
        });
    }
    lap("analytic B-splines recognized");
    let mut groups: Vec<(Rigid, Vec<Solid>)> = Vec::new();
    for (ids, motion) in placed {
        let g = match groups.iter().position(|(m, _)| *m == motion) {
            Some(g) => g,
            None => {
                groups.push((motion, Vec::new()));
                groups.len() - 1
            }
        };
        for id in ids {
            groups[g].1.push(local[&id].clone());
        }
    }
    let mut solids = Vec::new();
    for (motion, group) in groups {
        for s in join_open(group) {
            solids.push(if motion == Rigid::IDENTITY { s } else { motion.solid(&s) });
        }
    }
    lap("open shells joined");
    // Declared model tolerance: UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(x), unit, ...).
    let mut tol = f64::NAN;
    for parts in ents.values() {
        if let Some((_, a)) = parts.iter().find(|(n, _)| n == "UNCERTAINTY_MEASURE_WITH_UNIT") {
            if let Some(Val::Typed(t, v)) = a.first() {
                if t == "LENGTH_MEASURE" {
                    if let Some(x) = v.first().and_then(|x| Ctx::num(x).ok()) {
                        tol = if tol.is_nan() { x * unit } else { tol.max(x * unit) };
                    }
                }
            }
        }
    }
    Ok(Brep { solids, unit_mm: unit, unit_name, tolerance_mm: if tol.is_nan() { 1e-6 } else { tol } })
}

/// The arguments of an entity's REPRESENTATION record: (name, items, context).
fn rep_items<'a>(cx: &'a Ctx, id: u32) -> Option<&'a Vec<Val>> {
    cx.e[&id].iter().find_map(|(n, a)| match a.as_slice() {
        [Val::Str(_), Val::List(items), Val::Ref(_)] if n.ends_with("REPRESENTATION") => Some(items),
        _ => None,
    })
}

/// Every solid in the file in groups, each group with the motion that places it in
/// the file's root frame, a solid once per use. Representations joined by a
/// relationship with no transformation share a frame. One with a transformation, or
/// a MAPPED_ITEM, places a child representation in a parent once per use. Which side
/// is the child comes from the product structure (the NEXT_ASSEMBLY_USAGE_OCCURRENCE
/// behind its CONTEXT_DEPENDENT_SHAPE_REPRESENTATION, and the representations of its
/// two product definitions), since writers differ on the order; the first
/// representation is taken as the child when nothing says. Solids no root reaches
/// stay where they are.
fn instances(cx: &Ctx) -> Result<Vec<(Vec<u32>, Rigid)>, String> {
    let mut ids: Vec<u32> = cx.e.keys().copied().collect();
    ids.sort();
    let is_solid = |id: u32| ["MANIFOLD_SOLID_BREP", "FACETED_BREP", "BREP_WITH_VOIDS"].iter().any(|k| cx.is(id, k));
    // Frames shared: union-find over representations.
    let mut up: HashMap<u32, u32> = HashMap::new();
    fn find(up: &mut HashMap<u32, u32>, x: u32) -> u32 {
        let p = *up.entry(x).or_insert(x);
        if p == x {
            return x;
        }
        let r = find(up, p);
        up.insert(x, r);
        r
    }
    let reps: Vec<u32> = ids.iter().copied().filter(|&id| rep_items(cx, id).is_some()).collect();
    // (relationship, rep_1, rep_2, transformation)
    let mut placed = Vec::new();
    for &id in &ids {
        let Some(a) = cx.e[&id].iter().find_map(|(n, a)| match a.as_slice() {
            [_, _, Val::Ref(r1), Val::Ref(r2)] if n == "REPRESENTATION_RELATIONSHIP" || n == "SHAPE_REPRESENTATION_RELATIONSHIP" => Some((*r1, *r2)),
            _ => None,
        }) else {
            continue;
        };
        match cx.part(id, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION") {
            Ok(t) => placed.push((id, a.0, a.1, Ctx::r(&t[0])?)),
            Err(_) => {
                let (x, y) = (find(&mut up, a.0), find(&mut up, a.1));
                up.insert(x, y);
            }
        }
    }
    let mut items: HashMap<u32, HashSet<u32>> = HashMap::new();
    for &r in &reps {
        let c = find(&mut up, r);
        let set = items.entry(c).or_default();
        for v in rep_items(cx, r).unwrap() {
            if let Val::Ref(i) = v {
                set.insert(*i);
            }
        }
    }
    // Product definition -> its frame, through SHAPE_DEFINITION_REPRESENTATION and
    // PRODUCT_DEFINITION_SHAPE; relationship -> (parent, child) product definitions.
    let mut pd_rep: HashMap<u32, u32> = HashMap::new();
    let mut usage: HashMap<u32, (u32, u32)> = HashMap::new();
    let pds_def = |pds: u32| cx.part(pds, "PRODUCT_DEFINITION_SHAPE").ok().and_then(|a| Ctx::r(&a[2]).ok());
    for &id in &ids {
        if let Ok(a) = cx.part(id, "SHAPE_DEFINITION_REPRESENTATION") {
            if let (Some(def), Ok(rep)) = (Ctx::r(&a[0]).ok().and_then(pds_def), Ctx::r(&a[1])) {
                pd_rep.insert(def, rep);
            }
        }
        if let Ok(a) = cx.part(id, "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") {
            if let (Ok(rel), Some(nauo)) = (Ctx::r(&a[0]), Ctx::r(&a[1]).ok().and_then(pds_def)) {
                if let Ok(n) = cx.part(nauo, "NEXT_ASSEMBLY_USAGE_OCCURRENCE") {
                    usage.insert(rel, (Ctx::r(&n[3])?, Ctx::r(&n[4])?));
                }
            }
        }
    }
    // Parent frame -> (child frame, motion), once per use.
    let mut children: HashMap<u32, Vec<(u32, Rigid)>> = HashMap::new();
    let empty = HashSet::new();
    for (rel, r1, r2, xf) in placed {
        let (c1, c2) = (find(&mut up, r1), find(&mut up, r2));
        let frame_of = |up: &mut HashMap<u32, u32>, pd: u32| pd_rep.get(&pd).map(|&r| find(up, r));
        let first_is_child = match usage.get(&rel) {
            Some(&(parent, child)) => {
                let (pc, cc) = (frame_of(&mut up, parent), frame_of(&mut up, child));
                !(cc == Some(c2) || pc == Some(c1))
            }
            None => true,
        };
        let (child, parent) = if first_is_child { (c1, c2) } else { (c2, c1) };
        let m = cx.transformation(xf, items.get(&child).unwrap_or(&empty), items.get(&parent).unwrap_or(&empty), first_is_child)?;
        children.entry(parent).or_default().push((child, m));
    }
    for &r in &reps {
        for v in rep_items(cx, r).unwrap() {
            let Val::Ref(i) = v else { continue };
            if let Ok(a) = cx.part(*i, "MAPPED_ITEM") {
                let map = cx.part(Ctx::r(&a[1])?, "REPRESENTATION_MAP")?;
                let m = Rigid::between(&cx.frame(Ctx::r(&map[0])?)?, &cx.frame(Ctx::r(&a[2])?)?);
                let (child, parent) = (find(&mut up, Ctx::r(&map[1])?), find(&mut up, r));
                children.entry(parent).or_default().push((child, m));
            }
        }
    }
    let solids_of = |c: u32| -> Vec<u32> {
        let mut v: Vec<u32> = items.get(&c).map(|s| s.iter().copied().filter(|&i| is_solid(i)).collect()).unwrap_or_default();
        v.sort();
        v
    };
    let is_child: HashSet<u32> = children.values().flatten().map(|(c, _)| *c).collect();
    let mut roots: Vec<u32> = items.keys().copied().filter(|c| !is_child.contains(c)).collect();
    roots.sort();
    let mut out: Vec<(Vec<u32>, Rigid)> = Vec::new();
    let mut stack = vec![];
    fn walk(c: u32, m: Rigid, children: &HashMap<u32, Vec<(u32, Rigid)>>, solids_of: &dyn Fn(u32) -> Vec<u32>, stack: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, Rigid)>) {
        let s = solids_of(c);
        if !s.is_empty() {
            out.push((s, m));
        }
        if stack.len() > 32 || stack.contains(&c) {
            return;
        }
        stack.push(c);
        for (child, t) in children.get(&c).into_iter().flatten() {
            walk(*child, m.after(t), children, solids_of, stack, out);
        }
        stack.pop();
    }
    for r in roots {
        walk(r, Rigid::IDENTITY, &children, &solids_of, &mut stack, &mut out);
    }
    let reached: HashSet<u32> = out.iter().flat_map(|(s, _)| s.iter().copied()).collect();
    let rest: Vec<u32> = ids.iter().copied().filter(|&id| is_solid(id) && !reached.contains(&id)).collect();
    if !rest.is_empty() {
        out.push((rest, Rigid::IDENTITY));
    }
    Ok(out)
}

/// Joins solids whose shells are open (an edge used once) but share edges, which
/// some writers emit for one closed shell split a face per solid; closed solids and
/// open ones that share nothing pass through.
fn join_open(solids: Vec<Solid>) -> Vec<Solid> {
    let uses = |s: &Solid| {
        let mut n: HashMap<u32, usize> = HashMap::new();
        for e in s.faces.iter().flat_map(|f| f.bounds.iter().flat_map(|b| b.edges.iter())) {
            *n.entry(e.edge).or_default() += 1;
        }
        n
    };
    let counts: Vec<HashMap<u32, usize>> = solids.iter().map(uses).collect();
    let open: Vec<bool> = counts.iter().map(|n| n.values().any(|&k| k != 2)).collect();
    if open.iter().filter(|&&o| o).count() < 2 {
        return solids;
    }
    // Union-find over the open solids by shared edges.
    let mut up: Vec<usize> = (0..solids.len()).collect();
    fn find(up: &mut [usize], x: usize) -> usize {
        if up[x] != x {
            up[x] = find(up, up[x]);
        }
        up[x]
    }
    let mut owner: HashMap<u32, usize> = HashMap::new();
    for (i, n) in counts.iter().enumerate().filter(|(i, _)| open[*i]) {
        for &e in n.keys() {
            if let Some(&j) = owner.get(&e) {
                let (a, b) = (find(&mut up, i), find(&mut up, j));
                up[a] = b;
            } else {
                owner.insert(e, i);
            }
        }
    }
    let mut groups: Vec<(usize, Solid)> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for (i, s) in solids.into_iter().enumerate() {
        let root = find(&mut up, i);
        match slot.get(&root) {
            Some(&g) => groups[g].1.faces.extend(s.faces),
            None => {
                slot.insert(root, groups.len());
                groups.push((root, s));
            }
        }
    }
    groups.into_iter().map(|(_, s)| s).collect()
}

/// A rigid placement: quarter turns about z, an optional turn upside down (half a turn
/// about x), then a shift. Both turns are proper rotations, so faces keep their sense,
/// and both are exact in floating point (coordinates swap and change sign).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Place {
    pub quarter_turns: u8,
    pub flip: bool,
    pub shift: V3,
}

impl Place {
    pub fn new(quarter_turns: u8, flip: bool, x: f64, y: f64, z: f64) -> Place {
        Place { quarter_turns, flip, shift: V3::new(x, y, z) }
    }

    /// The same turns, shifted further.
    pub fn shifted(self, dx: f64, dy: f64, dz: f64) -> Place {
        Place { shift: V3::new(self.shift.x + dx, self.shift.y + dy, self.shift.z + dz), ..self }
    }

    /// Rotates a direction (no shift).
    pub fn turn(&self, v: V3) -> V3 {
        let v = if self.flip { V3::new(v.x, -v.y, -v.z) } else { v };
        match self.quarter_turns % 4 {
            0 => v,
            1 => V3::new(-v.y, v.x, v.z),
            2 => V3::new(-v.x, -v.y, v.z),
            _ => V3::new(v.y, -v.x, v.z),
        }
    }

    /// Moves a point.
    pub fn point(&self, p: V3) -> V3 {
        let q = self.turn(p);
        V3::new(q.x + self.shift.x, q.y + self.shift.y, q.z + self.shift.z)
    }
}

/// A proper rigid motion x -> r x + t (r by rows). Built from right-handed frames, so
/// it never mirrors and faces keep their sense.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub r: [V3; 3],
    pub t: V3,
}

impl Rigid {
    pub const IDENTITY: Rigid = Rigid { r: [V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::new(0.0, 0.0, 1.0)], t: V3::new(0.0, 0.0, 0.0) };

    /// The motion taking frame `a` onto frame `b`: a point at local coordinates l in
    /// `a` goes to local coordinates l in `b`.
    pub fn between(a: &Frame, b: &Frame) -> Rigid {
        let (ay, by) = (a.y(), b.y());
        let row = |k: usize| {
            let c = |v: V3| [v.x, v.y, v.z][k];
            a.x * c(b.x) + ay * c(by) + a.z * c(b.z)
        };
        let r = [row(0), row(1), row(2)];
        let m = Rigid { r, t: V3::default() };
        Rigid { r, t: b.o - m.turn(a.o) }
    }

    /// This motion after `first`.
    pub fn after(&self, first: &Rigid) -> Rigid {
        let r = [0, 1, 2].map(|k| first.r[0] * self.r[k].x + first.r[1] * self.r[k].y + first.r[2] * self.r[k].z);
        Rigid { r, t: self.point(first.t) }
    }

    pub fn turn(&self, v: V3) -> V3 {
        V3::new(self.r[0].dot(v), self.r[1].dot(v), self.r[2].dot(v))
    }

    pub fn point(&self, p: V3) -> V3 {
        self.turn(p) + self.t
    }
}

/// Moves model geometry by a proper rotation (`turn`) and a shift (`point`).
trait Motion {
    fn point(&self, p: V3) -> V3;
    fn turn(&self, v: V3) -> V3;

    fn frame(&self, f: &Frame) -> Frame {
        Frame { o: self.point(f.o), z: self.turn(f.z), x: self.turn(f.x) }
    }

    fn curve(&self, c: &Curve3) -> Curve3 {
        match c {
            Curve3::Line { p, dir } => Curve3::Line { p: self.point(*p), dir: self.turn(*dir) },
            Curve3::Circle { frame, r } => Curve3::Circle { frame: self.frame(frame), r: *r },
            Curve3::Ellipse { frame, a, b } => Curve3::Ellipse { frame: self.frame(frame), a: *a, b: *b },
            Curve3::Parabola { frame, f } => Curve3::Parabola { frame: self.frame(frame), f: *f },
            Curve3::Hyperbola { frame, a, b } => Curve3::Hyperbola { frame: self.frame(frame), a: *a, b: *b },
            // Field by field: `..b.clone()` would copy the control points only to drop them.
            Curve3::BSpline(b) => Curve3::BSpline(BSplineCurve { degree: b.degree, ctrl: b.ctrl.iter().map(|&q| self.point(q)).collect(), weights: b.weights.clone(), knots: b.knots.clone() }),
            Curve3::Other(n) => Curve3::Other(n.clone()),
        }
    }

    fn surface(&self, s: &Surface) -> Surface {
        match s {
            Surface::Plane { frame } => Surface::Plane { frame: self.frame(frame) },
            Surface::Cylinder { frame, r } => Surface::Cylinder { frame: self.frame(frame), r: *r },
            Surface::Cone { frame, r, semi_angle } => Surface::Cone { frame: self.frame(frame), r: *r, semi_angle: *semi_angle },
            Surface::Sphere { frame, r } => Surface::Sphere { frame: self.frame(frame), r: *r },
            Surface::Torus { frame, major, minor } => Surface::Torus { frame: self.frame(frame), major: *major, minor: *minor },
            Surface::BSpline(b) => Surface::BSpline(BSplineSurface {
                du: b.du,
                dv: b.dv,
                ctrl: b.ctrl.iter().map(|row| row.iter().map(|&q| self.point(q)).collect()).collect(),
                weights: b.weights.clone(),
                ku: b.ku.clone(),
                kv: b.kv.clone(),
            }),
            Surface::Extrusion { curve, dir } => Surface::Extrusion { curve: self.curve(curve), dir: self.turn(*dir) },
            Surface::Revolution { curve, axis_o, axis } => Surface::Revolution { curve: self.curve(curve), axis_o: self.point(*axis_o), axis: self.turn(*axis) },
            Surface::Other(n) => Surface::Other(n.clone()),
        }
    }

    /// The solid moved. Built field by field rather than cloned and overwritten,
    /// which copied every surface and curve once more only to replace it.
    fn solid(&self, s: &Solid) -> Solid {
        let edge = |e: &OEdge| OEdge {
            edge: e.edge,
            start: self.point(e.start),
            end: self.point(e.end),
            start_vertex: e.start_vertex,
            end_vertex: e.end_vertex,
            curve: self.curve(&e.curve),
            forward: e.forward,
        };
        let bound = |bd: &Bound| Bound { outer: bd.outer, edges: bd.edges.iter().map(edge).collect(), vertex: bd.vertex.map(|v| self.point(v)) };
        let face = |f: &BFace| BFace { id: f.id, surface: self.surface(&f.surface), same_sense: f.same_sense, bounds: f.bounds.iter().map(bound).collect() };
        Solid { id: s.id, name: s.name.clone(), faces: s.faces.iter().map(face).collect() }
    }
}

impl Motion for Place {
    fn point(&self, p: V3) -> V3 {
        Place::point(self, p)
    }
    fn turn(&self, v: V3) -> V3 {
        Place::turn(self, v)
    }
}

impl Motion for Rigid {
    fn point(&self, p: V3) -> V3 {
        Rigid::point(self, p)
    }
    fn turn(&self, v: V3) -> V3 {
        Rigid::turn(self, v)
    }
}

impl Brep {
    /// The same solids, placed.
    pub fn placed(&self, p: &Place) -> Brep {
        Brep { solids: self.solids.iter().map(|s| p.solid(s)).collect(), unit_mm: self.unit_mm, unit_name: self.unit_name.clone(), tolerance_mm: self.tolerance_mm }
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    /// The reader before its instances were parsed in parallel: one after another.
    fn parse_serial(text: &[u8]) -> Result<Entities, String> {
        let start = text.windows(5).position(|w| w == b"DATA;").ok_or("no DATA section")? + 5;
        let mut p = Parser { b: text, i: start };
        let mut ents = Entities::new();
        loop {
            match p.peek() {
                b'#' => {}
                b'E' => break,
                0 => break,
                c => return Err(format!("unexpected '{}' at byte {}", c as char, p.i)),
            }
            let (id, parts) = parse_instance(&mut p)?;
            ents.insert(id, parts);
        }
        Ok(ents)
    }

    fn same(text: &[u8]) {
        let (a, b) = (parse_data(text).unwrap(), parse_serial(text).unwrap());
        let show = |e: &Entities| {
            let mut v: Vec<String> = e.iter().map(|(k, x)| format!("{k}: {x:?}")).collect();
            v.sort();
            v
        };
        assert_eq!(show(&a), show(&b));
    }

    /// Strings and comments holding the characters the instances are split at.
    #[test]
    fn instances_split_past_strings_and_comments() {
        let data = "ISO-10303-21;\nDATA;\n\
            #1=PRODUCT('a;b','#2=X();','it''s;',$);\n\
            /* a comment; with 'a quote and #3=Y(); */\n\
            #2=CARTESIAN_POINT('',(1.,2.,3.)); /* trailing; */ #3=(NAMED_UNIT(*)SI_UNIT($,.METRE.));\n\
            #4=DIRECTION('''',(0.,0.,1.));\n\
            ENDSEC;\nEND-ISO-10303-21;\n";
        same(data.as_bytes());
        assert_eq!(parse_data(data.as_bytes()).unwrap().len(), 4);
    }

    /// Every test part the repository's parts folder holds (ICAM_TEST_PARTS, else test-parts;
    /// it is not committed: none, none read).
    #[test]
    fn parallel_reader_matches_serial_on_test_parts() {
        let root = std::env::var_os("ICAM_TEST_PARTS").map_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-parts"), std::path::PathBuf::from);
        let mut files = Vec::new();
        let mut dirs = vec![root];
        while let Some(d) = dirs.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|x| x.eq_ignore_ascii_case("stp") || x.eq_ignore_ascii_case("step")) {
                    files.push(path);
                }
            }
        }
        for f in &files {
            same(&std::fs::read(f).unwrap());
        }
        eprintln!("{} test parts read both ways", files.len());
    }
}
