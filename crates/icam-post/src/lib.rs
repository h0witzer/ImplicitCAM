//! Programs: G-code in and out, and the point types every crate above shares. The
//! reader ([`gcode`]) turns a 3-axis mill program, ours or a baseline's, into tool
//! moves in mm, so every instrument measures programs the same way. A strategy's own
//! path is a run of tagged tool-center points ([`PathPt`]). [`post`] writes our moves as a
//! program; it is being retooled for the one planner (roadmap milestone 1).

use std::ops::{Add, Sub};

pub mod gcode;
pub mod post;

pub use gcode::{parse, read, read_tool_table, Move, MoveKind, ToolDef};

/// What a tool-center point of a path is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Entry,
    Cut,
    Wall,
    Rapid,
}

/// A tool-center point of a path, mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathPt {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub tag: Tag,
}

impl PathPt {
    pub fn new(x: f64, y: f64, z: f64, tag: Tag) -> PathPt {
        PathPt { x, y, z, tag }
    }
}

/// An axis-aligned box in the XY plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2 {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Box2 {
    pub const fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Box2 {
        Box2 { x0, y0, x1, y1 }
    }
    pub fn grow(self, d: f64) -> Box2 {
        Box2::new(self.x0 - d, self.y0 - d, self.x1 + d, self.y1 + d)
    }
    pub fn contains(self, x: f64, y: f64) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// A point or vector, in mm.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl V3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn len_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn len(self) -> f64 {
        self.len_sq().sqrt()
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
