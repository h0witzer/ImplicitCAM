//! The comparison panel over the window, a program-against-part viewer on the field
//! engine. The panel on the left holds the stats at the chosen tolerance, the display
//! settings and the regions the comparison found (regions.rs); over the view are the card
//! of the region picked, the play bar (with the gouges marked on its track and the tool to
//! follow) and the view's tools. The window (play.rs) draws the field in what the panel
//! leaves, and acts on what the panel asks (`Act`).

use egui::{Align2, Color32, CornerRadius, Frame, Margin, Pos2, Rect, RichText, Sense, Stroke, Vec2};

use crate::regions::{Kind, Region, LIMIT};

/// The panel's width, points.
const WIDTH: f32 = 340.0;

// The old viewer's dark palette.
const BG: Color32 = Color32::from_rgb(0x17, 0x1a, 0x1f);
const CHIP: Color32 = Color32::from_rgb(0x20, 0x24, 0x2a);
const INK: Color32 = Color32::from_rgb(0xe6, 0xe9, 0xed);
const MUTED: Color32 = Color32::from_rgb(0x98, 0xa1, 0xab);
const LINE: Color32 = Color32::from_rgb(0x2a, 0x2f, 0x36);
const ACCENT: Color32 = Color32::from_rgb(0x6e, 0xa0, 0xff);
pub const GOUGE: Color32 = Color32::from_rgb(0xff, 0x5d, 0x6c);
pub const EXCESS: Color32 = Color32::from_rgb(0x6e, 0xa8, 0xff);
// The shop-floor ramps the caster colors with (cache.wgsl).
const GO: [Color32; 3] = [Color32::from_rgb(0x9b, 0x11, 0x11), Color32::from_rgb(0xd0, 0x2e, 0x1e), Color32::from_rgb(0xef, 0x62, 0x46)];
const OK: Color32 = Color32::from_rgb(0x6c, 0xd9, 0x81);
const EX: [Color32; 3] = [Color32::from_rgb(0x4a, 0xb4, 0xe6), Color32::from_rgb(0x13, 0x7e, 0xce), Color32::from_rgb(0x1e, 0x47, 0x98)];
const FAR: Color32 = Color32::from_rgb(0x5f, 0x64, 0x69);
const RAW: Color32 = Color32::from_rgb(0x80, 0x85, 0x8f);

/// Which surface the view draws, colored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// The part, colored by what the program left on it.
    Part,
    /// The stock the program leaves, colored by its distance to the part.
    Stock,
}

/// What the panel asks the window to do.
#[derive(Clone, Copy, Debug)]
pub enum Act {
    /// Play, or pause.
    Toggle,
    /// The playhead to this time on the clock.
    Seek(f64),
    /// A move of the program back or on.
    Step(i32),
    /// Play at this many times the machine's speed (0: the rest of the program in a minute).
    Speed(f64),
    /// Fly to region i (None: clear).
    Select(Option<usize>),
    /// The playhead to the moment gouge i was cut, the camera on the tool.
    Moment(usize),
    Reset,
}

/// What the window tells the panel each frame.
pub struct Info<'a> {
    pub program: &'a str,
    pub part: Option<&'a str>,
    /// The comparison: what it is doing, and when done, its regions at the tolerance.
    pub comparing: &'a str,
    pub regions: Option<&'a [Region]>,
    /// The clock: now, the end, playing, the speed; the G-code line and operation now.
    pub now: f64,
    pub end: f64,
    pub playing: bool,
    pub speed: f64,
    pub line: Option<(usize, &'a str)>,
    /// When each gouge was cut (its move's end), for the track.
    pub marks: &'a [f64],
    /// For a region: the move that cut it, as (line, operation, when it ended).
    pub cut: &'a dyn Fn(usize) -> Option<(usize, String, f64)>,
    /// Whether the part's view can be drawn.
    pub part_view: bool,
}

/// The panel's settings, read by the window after each pass.
pub struct Settings {
    /// Shop tolerance and the color ramps' deep end, mm.
    pub tol: f32,
    pub scale: f32,
    pub inches: bool,
    pub surface: Surface,
    pub path: bool,
    pub follow: bool,
    pub selected: Option<usize>,
    /// The view's rectangle, points: what the panel leaves.
    pub view: Rect,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { tol: 0.0254, scale: 1.0, inches: false, surface: Surface::Stock, path: true, follow: false, selected: None, view: Rect::NOTHING }
    }
}

impl Settings {
    fn len(&self, mm: f64) -> String {
        if self.inches {
            format!("{:.4} in", mm / 25.4)
        } else {
            format!("{mm:.3} mm")
        }
    }

    fn area(&self, mm2: f64) -> String {
        if self.inches {
            format!("{:.3} in\u{b2}", mm2 / 645.16)
        } else {
            format!("{mm2:.1} mm\u{b2}")
        }
    }
}

/// Seconds as h:mm:ss.
fn hms(t: f64) -> String {
    let t = t.max(0.0);
    format!("{}:{:02}:{:02}", (t / 3600.0) as u64, ((t / 60.0) as u64) % 60, (t as u64) % 60)
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title.to_uppercase()).size(11.0).color(MUTED).strong());
    ui.add_space(4.0);
}

/// A tile of the stats, half the panel wide.
fn stat(ui: &mut egui::Ui, k: &str, v: &str, color: Color32) {
    let w = (WIDTH - 32.0 - ui.spacing().item_spacing.x) / 2.0 - 20.0;
    Frame::new().fill(CHIP).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(10, 8)).show(ui, |ui| {
        ui.set_width(w);
        ui.vertical(|ui| {
            ui.label(RichText::new(k).size(11.0).color(MUTED));
            ui.label(RichText::new(v).size(17.0).color(color).strong());
        });
    });
}

/// The play bar's icons, painted (the fonts have no such glyphs).
#[derive(Clone, Copy)]
enum Glyph {
    Play,
    Pause,
    Back,
    On,
}

fn glyph(ui: &mut egui::Ui, g: Glyph, tip: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(30.0, 24.0), Sense::click());
    let vis = ui.style().interact(&resp);
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(5), vis.weak_bg_fill);
    let (c, col) = (rect.center(), vis.fg_stroke.color);
    let at = |x: f32, y: f32| c + Vec2::new(x, y);
    let tri = |pts: Vec<Pos2>| egui::Shape::convex_polygon(pts, col, Stroke::NONE);
    let bar = |x: f32| Rect::from_min_max(at(x - 1.25, -6.0), at(x + 1.25, 6.0));
    match g {
        Glyph::Play => {
            p.add(tri(vec![at(-4.0, -6.5), at(6.0, 0.0), at(-4.0, 6.5)]));
        }
        Glyph::Pause => {
            p.rect_filled(bar(-3.0), 0.0, col);
            p.rect_filled(bar(3.0), 0.0, col);
        }
        Glyph::Back => {
            p.rect_filled(bar(-5.0), 0.0, col);
            p.add(tri(vec![at(5.0, -6.0), at(5.0, 6.0), at(-3.0, 0.0)]));
        }
        Glyph::On => {
            p.rect_filled(bar(5.0), 0.0, col);
            p.add(tri(vec![at(-5.0, -6.0), at(3.0, 0.0), at(-5.0, 6.0)]));
        }
    }
    resp.on_hover_text(tip)
}

/// A row: a label, then the controls.
fn row(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.add_sized([86.0, 20.0], egui::Label::new(RichText::new(label).color(MUTED)));
        add(ui);
    });
}

/// The ramp from a to c through b at t in [0, 1].
fn ramp(r: [Color32; 3], t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |a: Color32, b: Color32, s: f32| Color32::from_rgb((a.r() as f32 + (b.r() as f32 - a.r() as f32) * s) as u8, (a.g() as f32 + (b.g() as f32 - a.g() as f32) * s) as u8, (a.b() as f32 + (b.b() as f32 - a.b() as f32) * s) as u8);
    if t < 0.5 {
        mix(r[0], r[1], 2.0 * t)
    } else {
        mix(r[1], r[2], 2.0 * t - 1.0)
    }
}

/// The legend: the gouge ramp (deep to shallow), on size, the stock-left ramp.
fn legend(ui: &mut egui::Ui, s: &Settings) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(WIDTH - 32.0, 14.0), Sense::hover());
    let p = ui.painter();
    let n = 60;
    let w = rect.width() / n as f32;
    for i in 0..n {
        let x = i as f32 / (n - 1) as f32;
        let c = if x < 0.42 {
            ramp([GO[0], GO[1], GO[2]], x / 0.42)
        } else if x < 0.58 {
            OK
        } else {
            ramp(EX, (x - 0.58) / 0.42)
        };
        p.rect_filled(Rect::from_min_size(Pos2::new(rect.left() + i as f32 * w, rect.top()), Vec2::new(w + 0.5, rect.height())), 0.0, c);
    }
    ui.horizontal(|ui| {
        let t = |v: f64| if s.inches { format!("{:.3}", v / 25.4) } else { format!("{v:.2}") };
        ui.label(RichText::new(format!("-{}", t(s.scale as f64))).size(10.0).color(MUTED));
        ui.add_space(52.0);
        ui.label(RichText::new(format!("\u{b1}{} on size", t(s.tol as f64))).size(10.0).color(MUTED));
        ui.add_space(42.0);
        ui.label(RichText::new(format!("+{}", t(s.scale as f64))).size(10.0).color(MUTED));
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("gouge").size(10.5).color(GOUGE));
        ui.add_space(100.0);
        ui.label(RichText::new("material left").size(10.5).color(EXCESS));
    });
    ui.horizontal_wrapped(|ui| {
        let key = |ui: &mut egui::Ui, c: Color32, text: &str| {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter().rect_filled(r, 2.0, c);
            ui.label(RichText::new(text).size(10.5).color(MUTED));
        };
        key(ui, FAR, "off the part (waste)");
        key(ui, RAW, "never cut");
    });
}

/// A colored tag.
fn chip(ui: &mut egui::Ui, kind: Kind) {
    let (text, c) = match kind {
        Kind::Gouge => ("Gouge", GOUGE),
        Kind::Left => ("Left", EXCESS),
    };
    Frame::new().fill(c.gamma_multiply(0.22)).corner_radius(CornerRadius::same(6)).inner_margin(Margin::symmetric(6, 1)).show(ui, |ui| {
        ui.label(RichText::new(text).size(11.0).color(c).strong());
    });
}

/// One pass of the panel and the view's overlays; returns what it asks.
pub fn show(ui: &mut egui::Ui, s: &mut Settings, info: &Info) -> Vec<Act> {
    let mut acts = Vec::new();
    let ctx = ui.ctx().clone();
    egui::Panel::left("comparison").exact_size(WIDTH).resizable(false).frame(Frame::new().fill(BG).inner_margin(Margin::same(16)).stroke(Stroke::new(1.0, LINE))).show(ui, |ui| {
        ui.label(RichText::new("Stock comparison").size(19.0).color(INK).strong());
        ui.label(RichText::new(match info.part {
            Some(p) => format!("{}  against  {}", info.program, p),
            None => info.program.to_string(),
        })
        .size(11.0)
        .color(MUTED));

        section(ui, "At this tolerance");
        let regions = info.regions.unwrap_or(&[]);
        let gouges: Vec<&Region> = regions.iter().filter(|r| r.kind == Kind::Gouge).collect();
        let left: Vec<&Region> = regions.iter().filter(|r| r.kind == Kind::Left).collect();
        let none = info.regions.is_none();
        let dash = |v: String| if none { "-".to_string() } else { v };
        ui.horizontal(|ui| {
            stat(ui, "Gouge regions", &dash(gouges.len().to_string()), GOUGE);
            stat(ui, "Undertooled regions", &dash(left.len().to_string()), EXCESS);
        });
        ui.horizontal(|ui| {
            let deepest = gouges.iter().map(|r| r.dev).fold(0.0, f64::max);
            let most = left.iter().map(|r| r.dev).fold(0.0, f64::max);
            stat(ui, "Deepest gouge", &dash(if gouges.is_empty() { "none".into() } else { s.len(deepest) }), GOUGE);
            stat(ui, "Most material left", &dash(if left.is_empty() { "none".into() } else if most >= LIMIT { format!("{}+", s.len(LIMIT)) } else { s.len(most) }), EXCESS);
        });
        ui.add_space(4.0);
        ui.label(RichText::new(info.comparing).size(11.0).color(MUTED));

        section(ui, "Display");
        row(ui, "Surface", |ui| {
            ui.add_enabled_ui(info.part_view, |ui| {
                if ui.selectable_label(s.surface == Surface::Part, "Design part").on_disabled_hover_text("Needs the part (--part)").clicked() {
                    s.surface = Surface::Part;
                }
            });
            if ui.selectable_label(s.surface == Surface::Stock, "Simulated stock").clicked() {
                s.surface = Surface::Stock;
            }
        });
        row(ui, "Units", |ui| {
            if ui.selectable_label(s.inches, "in").clicked() {
                s.inches = true;
            }
            if ui.selectable_label(!s.inches, "mm").clicked() {
                s.inches = false;
            }
        });
        row(ui, "Tool path", |ui| {
            ui.checkbox(&mut s.path, "Show");
        });
        row(ui, "Tolerance \u{b1}", |ui| {
            let mut v = if s.inches { s.tol / 25.4 } else { s.tol };
            let range = if s.inches { 0.001..=0.04 } else { 0.0254..=1.0 };
            if ui.add(egui::Slider::new(&mut v, range).logarithmic(true).max_decimals(4)).changed() {
                s.tol = if s.inches { v * 25.4 } else { v };
            }
        });
        row(ui, "Color scale", |ui| {
            let mut v = if s.inches { s.scale / 25.4 } else { s.scale };
            let range = if s.inches { 0.002..=0.2 } else { 0.05..=5.0 };
            if ui.add(egui::Slider::new(&mut v, range).logarithmic(true).max_decimals(3)).changed() {
                s.scale = if s.inches { v * 25.4 } else { v };
            }
        });
        ui.add_space(6.0);
        legend(ui, s);

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            let _ = ui.selectable_label(true, RichText::new(format!("Regions ({})", regions.len())).strong());
            ui.add_enabled(false, egui::Button::selectable(false, "Faces")).on_disabled_hover_text("Faces need the part's faces named (not yet)");
        });
        ui.separator();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if info.regions.is_some() && regions.is_empty() {
                ui.label(RichText::new("No region exceeds the tolerance.").color(MUTED));
            }
            for (i, r) in regions.iter().enumerate().take(300) {
                let sel = s.selected == Some(i);
                let resp = Frame::new()
                    .fill(if sel { ACCENT.gamma_multiply(0.18) } else { Color32::TRANSPARENT })
                    .corner_radius(CornerRadius::same(6))
                    .inner_margin(Margin::symmetric(6, 5))
                    .show(ui, |ui| {
                        ui.set_width(WIDTH - 44.0);
                        ui.horizontal(|ui| {
                            chip(ui, r.kind);
                            let what = match r.kind {
                                Kind::Gouge => format!("{} deep", s.len(r.dev)),
                                Kind::Left if r.dev >= LIMIT => format!("{}+ thick", s.len(LIMIT)),
                                Kind::Left => format!("{} thick", s.len(r.dev)),
                            };
                            ui.label(RichText::new(what).color(INK));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(s.area(r.area)).color(MUTED).size(11.5));
                            });
                        });
                        let meta = match r.by.and_then(|m| (info.cut)(m)) {
                            Some((line, op, at)) => format!("cut by line {line} at {}, {op}", hms(at)),
                            None => format!("at {:.1}, {:.1}, {:.1}", r.center[0], r.center[1], r.center[2]),
                        };
                        ui.label(RichText::new(meta).size(10.5).color(MUTED));
                    })
                    .response
                    .interact(Sense::click());
                if resp.clicked() {
                    acts.push(Act::Select(if sel { None } else { Some(i) }));
                }
                resp.context_menu(|ui| {
                    if r.by.is_some() && ui.button("View the moment it was cut").on_hover_text("The playhead to when the tool passes the gouge's deepest point, the camera on the tool").clicked() {
                        acts.push(Act::Moment(i));
                        ui.close();
                    }
                    if ui.button("Fly to the region").clicked() {
                        acts.push(Act::Select(Some(i)));
                        ui.close();
                    }
                });
            }
        });
    });
    s.view = ui.available_rect_before_wrap();
    let view = s.view;

    // The card of the region picked.
    if let Some(r) = s.selected.and_then(|i| info.regions.and_then(|rs| rs.get(i))) {
        egui::Area::new(egui::Id::unique("pick")).fixed_pos(view.left_top() + Vec2::new(16.0, 16.0)).show(&ctx, |ui| {
            Frame::new().fill(BG.gamma_multiply(0.96)).corner_radius(CornerRadius::same(10)).inner_margin(Margin::same(12)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                ui.set_width(280.0);
                let (title, c) = match r.kind {
                    Kind::Gouge => (format!("Gouge, {} deep", s.len(r.dev)), GOUGE),
                    Kind::Left => (format!("Material left, {} thick", s.len(r.dev)), EXCESS),
                };
                ui.label(RichText::new(title).size(15.0).color(c).strong());
                egui::Grid::new("pick rows").num_columns(2).spacing([12.0, 3.0]).show(ui, |ui| {
                    ui.label(RichText::new("Area").color(MUTED));
                    ui.label(s.area(r.area));
                    ui.end_row();
                    if let Some(v) = r.volume {
                        ui.label(RichText::new("Volume").color(MUTED));
                        ui.label(if s.inches { format!("{:.4} in\u{b3}", v / 16387.064) } else { format!("{v:.1} mm\u{b3}") });
                        ui.end_row();
                    }
                    ui.label(RichText::new("Center").color(MUTED));
                    let k = if s.inches { 1.0 / 25.4 } else { 1.0 };
                    ui.label(format!("{:.3}, {:.3}, {:.3}", r.center[0] * k, r.center[1] * k, r.center[2] * k));
                    ui.end_row();
                    if let Some((line, op, at)) = r.by.and_then(|m| (info.cut)(m)) {
                        ui.label(RichText::new("Cut by").color(MUTED));
                        ui.label(format!("line {line}, {op}"));
                        ui.end_row();
                        ui.label(RichText::new("At").color(MUTED));
                        if ui.link(hms(at)).on_hover_text("The moment it was cut, the camera on the tool").clicked() {
                            if let Some(i) = s.selected {
                                acts.push(Act::Moment(i));
                            }
                        }
                        ui.end_row();
                    }
                });
            });
        });
    }

    // How to move the view, and its tools.
    egui::Area::new(egui::Id::unique("hint")).anchor(Align2::LEFT_BOTTOM, Vec2::new(view.left() + 16.0, -84.0)).show(&ctx, |ui| {
        Frame::new().fill(BG.gamma_multiply(0.9)).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(10, 6)).show(ui, |ui| {
            ui.label(RichText::new("Drag to orbit - right-drag to pan - scroll to zoom - right-click a gouge for the moment it was cut").size(11.5).color(MUTED));
        });
    });
    egui::Area::new(egui::Id::unique("tools")).anchor(Align2::RIGHT_BOTTOM, Vec2::new(-16.0, -80.0)).show(&ctx, |ui| {
        Frame::new().fill(BG.gamma_multiply(0.96)).corner_radius(CornerRadius::same(8)).inner_margin(Margin::same(4)).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Reset view").clicked() {
                    acts.push(Act::Reset);
                }
                if ui.button("Clear selection").clicked() {
                    acts.push(Act::Select(None));
                }
            });
        });
    });

    // The play bar.
    let bar_w = (view.width() - 32.0).min(980.0);
    egui::Area::new(egui::Id::unique("play")).fixed_pos(Pos2::new(view.center().x - bar_w / 2.0, view.bottom() - 62.0)).show(&ctx, |ui| {
        Frame::new().fill(BG.gamma_multiply(0.96)).corner_radius(CornerRadius::same(10)).inner_margin(Margin::symmetric(10, 8)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
            ui.set_width(bar_w - 22.0);
            ui.horizontal(|ui| {
                if glyph(ui, if info.playing { Glyph::Pause } else { Glyph::Play }, "Play / pause (Space)").clicked() {
                    acts.push(Act::Toggle);
                }
                if glyph(ui, Glyph::Back, "Previous move (Left arrow)").clicked() {
                    acts.push(Act::Step(-1));
                }
                if glyph(ui, Glyph::On, "Next move (Right arrow)").clicked() {
                    acts.push(Act::Step(1));
                }
                let speeds = [1.0, 5.0, 20.0, 100.0, 500.0];
                let label = if speeds.contains(&info.speed) { format!("{}x", info.speed) } else { format!("{:.0}x", info.speed) };
                egui::ComboBox::from_id_salt("speed").width(78.0).selected_text(label).show_ui(ui, |ui| {
                    for v in speeds {
                        if ui.selectable_label(info.speed == v, format!("{v}x")).clicked() {
                            acts.push(Act::Speed(v));
                        }
                    }
                    if ui.selectable_label(false, "a minute").on_hover_text("The rest of the program in a minute").clicked() {
                        acts.push(Act::Speed(0.0));
                    }
                });
                let time = format!("{} / {}", hms(info.now), hms(info.end));
                let right = 200.0;
                let track_w = (ui.available_width() - right).max(60.0);
                let (rect, resp) = ui.allocate_exact_size(Vec2::new(track_w, 22.0), Sense::click_and_drag());
                let p = ui.painter();
                let y = rect.center().y;
                let x_of = |t: f64| rect.left() + (t / info.end.max(1e-9)).clamp(0.0, 1.0) as f32 * rect.width();
                p.line_segment([Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)], Stroke::new(4.0, LINE));
                p.line_segment([Pos2::new(rect.left(), y), Pos2::new(x_of(info.now), y)], Stroke::new(4.0, ACCENT));
                for &m in info.marks {
                    let x = x_of(m);
                    p.line_segment([Pos2::new(x, y - 8.0), Pos2::new(x, y - 3.0)], Stroke::new(2.0, GOUGE));
                }
                p.circle_filled(Pos2::new(x_of(info.now), y), 6.0, INK);
                if let Some(pos) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
                    acts.push(Act::Seek(((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * info.end));
                }
                resp.on_hover_text("Drag to move the playhead; red marks: where gouges were cut");
                ui.label(RichText::new(time).monospace().size(12.0).color(INK));
                ui.toggle_value(&mut s.follow, "Follow tool");
            });
            if let Some((line, op)) = info.line {
                ui.label(RichText::new(format!("line {line}  {op}")).size(10.5).color(MUTED));
            }
        });
    });
    acts
}
