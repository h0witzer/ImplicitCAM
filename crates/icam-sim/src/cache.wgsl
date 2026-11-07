// The stock drawn from its cached field (docs/SIM_VISUALIZATION_ENGINE.md, sections 4.4 and
// 4.7). The cache comes in levels around the view, each twice the cell of the one before;
// a ray is drawn from the finest level that holds it, piece by piece: each piece ends where
// its level ends or a finer one begins, all decided along the ray (in t), so no level is
// passed over. Within a level the ray walks the brick map, skipping bricks with no surface,
// and steps cell by cell through bricks with values, passing on a bit a cell none of whose
// corners reads zero or less (`cells`). Inside a cell the surface is the zero
// of the trilinear interpolant of the cell's eight corner values, which along a ray is a
// cubic: read at four even points it is known exactly, split at its turning points, and the
// first span that changes sign is bisected (after Soderlund, Evans and Akenine-Moller 2022).
// A ray that enters a cell, or a level, already inside meets it there, so no ray slips
// between two cells or two levels. The normal the light reads is the cache's gradient by
// central differences half a cell either side of the hit (`lit_normal`), read across cell
// and brick faces, so it changes smoothly from cell to cell; the interpolant's own gradient
// jumps at every cell face, and lit with it each cell showed as a facet. Bricks no move
// comes near hold only the stock box's own faces, met by the box itself, and by the same
// rule are met on entry when the ray is already inside the box. An air brick that knows
// the air around it (SKIP) lets the ray cross the whole air cube at once.
//
// The tool at the playhead is drawn in the same pass from its own exact distance (a
// cylinder from its tip up its stick-out, the bottom edge rounded, or a cone under a
// cylinder), sphere-traced inside a ball that holds it; the nearer of tool and stock wins.
//
// In the part's view (View.heat.z is 1) the walk reads the part's cache instead (partcache.rs,
// on the same levels' grids), and each hit on the part is colored by the stock's cached field
// there: stock left on the part reads below zero, a cut into it above.
//
// With the part's records (heat.rs; Levels.count.y is 1), a cut surface is colored by the
// part's field where the ray meets it: each column's planes exactly, its section's distance
// interpolated over the brick's 9 x 9 nodes. Red where the stock lies inside the part (cut
// into it), green within the shop tolerance, blue where stock is left on it, gray where the
// part is farther than the records reach: shop-floor colors.

// The ray through the pixel at ndc (x, y) leaves the eye along fwd + x right + y up. The
// tool: its tip (xyz) and radius (w), its axis (xyz) and corner radius (w), and (cone
// height, stick-out, drawn when 1, unused).
struct View {
    view_proj: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    tool_at: vec4<f32>,
    tool_axis: vec4<f32>,
    tool_shape: vec4<f32>,
    // The colors' bands: (tolerance, the ramps' deep end, mm; unused).
    heat: vec4<f32>,
};

const MAX_LEVELS: u32 = 16u;

// A level in the box's frame: lo.xyz its first node, lo.w its cell side; dims.xyz its
// bricks, dims.w where its brick map starts in `bricks`.
struct Level {
    lo: vec4<f32>,
    dims: vec4<u32>,
    // y: where its bricks start in `atlas` (its map holds the level's own slots, so a
    // level's words are the same wherever it lies, and are written before it has a place).
    // As `level_at` makes a level of the part's cache: x where its map starts in `pbricks`,
    // y where its bricks start in `patlas`.
    part: vec4<u32>,
};

// A level of the part's cache (partcache.rs), on a grid of its own, not tied to the stock's
// levels: one placed anew is drawn from the one before until it comes. lo.xyz its first
// node, lo.w its cell side; dims.xyz its bricks, dims.w where its map starts in `pbricks`;
// at.x where its bricks start in `patlas`.
struct PartLevel {
    lo: vec4<f32>,
    dims: vec4<u32>,
    at: vec4<u32>,
};

// count.x levels, finest first, and the stock box; count.y is 1 when `heat` holds the
// part's records; count.z the part's levels, finest first.
struct Levels {
    count: vec4<u32>,
    box_lo: vec4<f32>,
    box_hi: vec4<f32>,
    level: array<Level, 16>,
    parts: array<PartLevel, 16>,
};

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> L: Levels;
@group(0) @binding(2) var<storage, read> bricks: array<u32>;
@group(0) @binding(3) var<storage, read> atlas: array<u32>;
@group(0) @binding(4) var<storage, read> heat: array<u32>;
@group(0) @binding(5) var<storage, read> pbricks: array<u32>;
@group(0) @binding(6) var<storage, read> patlas: array<u32>;
// The stock's bricks' cells that may hold surface, slot for slot with `atlas`: CELL_WORDS
// words a brick, bit c (cells x fastest) set when a corner of cell c reads zero or less
// (icam_sweep::cache::cells_of). The part's bricks have none.
@group(0) @binding(7) var<storage, read> cells: array<u32>;

const AIR: u32 = 0xffffffffu;
const SOLID: u32 = 0xfffffffeu;
const FACES: u32 = 0xfffffffdu;
const SKIP: u32 = 0xffff0000u;
const WORDS: u32 = 365u;
const CELL_WORDS: u32 = 16u;
const BIG: f32 = 1e30;
// A brick's record of the part (heat.rs): 4 columns of (top, bottom, 41 words of 81 section
// distances / REACH).
const HEAT_COLUMNS: u32 = 4u;
const COLUMN_WORDS: u32 = 43u;
const HEAT_WORDS: u32 = 172u;
const REACH: f32 = 8.0;

const NONE: u32 = 0xffffffffu;
// Cells between FACET_PX and SMOOTH_PX pixels on screen are lit by the smooth normal
// (`lit_normal`); under half FACET_PX and past twice SMOOTH_PX by their own gradient (a cell
// of a pixel or two shows no facet, and the smooth normal's reads cost 1.6 ms a frame there
// for 0.2 % of pixels), and between by a blend of the two.
const FACET_PX: f32 = 3.0;
const SMOOTH_PX: f32 = 16.0;

// Which cache the walk reads: 0 the stock's, 1 the part's.
var<private> reading: u32;

// The brick of the last hit `walk` returned: its slot (AIR for none) and first node.
var<private> hit_slot: u32;
var<private> hit_base: vec3<f32>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// One triangle over the whole screen.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: VOut;
    o.clip = vec4<f32>(xy, 0.0, 1.0);
    o.ndc = xy;
    return o;
}

struct FOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

fn value(slot: u32, n: u32) -> f32 {
    var w: u32;
    if (reading == 1u) {
        w = patlas[slot * WORDS + n / 2u];
    } else {
        w = atlas[slot * WORDS + n / 2u];
    }
    let pair = unpack2x16snorm(w);
    return select(pair.x, pair.y, (n & 1u) == 1u);
}

// 1 / d, huge where d is zero (an axis the ray never crosses).
fn reciprocal(d: vec3<f32>) -> vec3<f32> {
    return select(1.0 / d, sign(d + vec3<f32>(1e-30)) * BIG, abs(d) < vec3<f32>(1e-12));
}

// Where the ray is inside the box: (enter, leave).
fn slab(o: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let n = min(a, b);
    let f = max(a, b);
    return vec2<f32>(max(max(n.x, n.y), n.z), min(min(f.x, f.y), f.z));
}

// The trilinear interpolant at q in the cell, bottom corners a (000, 100, 010, 110) and
// top corners b (001, 101, 011, 111).
fn tri(a: vec4<f32>, b: vec4<f32>, q: vec3<f32>) -> f32 {
    let bottom = mix(mix(a.x, a.y, q.x), mix(a.z, a.w, q.x), q.y);
    let top = mix(mix(b.x, b.y, q.x), mix(b.z, b.w, q.x), q.y);
    return mix(bottom, top, q.z);
}

fn grad(a: vec4<f32>, b: vec4<f32>, q: vec3<f32>) -> vec3<f32> {
    let gx = mix(mix(a.y - a.x, a.w - a.z, q.y), mix(b.y - b.x, b.w - b.z, q.y), q.z);
    let gy = mix(mix(a.z - a.x, a.w - a.y, q.x), mix(b.z - b.x, b.w - b.y, q.x), q.z);
    let gz = mix(mix(b.x - a.x, b.y - a.y, q.x), mix(b.z - a.z, b.w - a.w, q.x), q.y);
    return vec3<f32>(gx, gy, gz);
}

fn cubic(c: vec4<f32>, u: f32) -> f32 {
    return ((c.w * u + c.z) * u + c.y) * u + c.x;
}

// The first u in [0, 1] where the interpolant along q0 + u dq reaches zero, or -1.
fn cell_hit(a: vec4<f32>, b: vec4<f32>, q0: vec3<f32>, dq: vec3<f32>) -> f32 {
    let f0 = tri(a, b, q0);
    if (f0 <= 0.0) {
        return 0.0;
    }
    let f1 = tri(a, b, q0 + dq / 3.0);
    let f2 = tri(a, b, q0 + dq * (2.0 / 3.0));
    let f3 = tri(a, b, q0 + dq);
    // The cubic through the four, in x = 3u, then in u.
    let d1 = f1 - f0;
    let d2 = f2 - 2.0 * f1 + f0;
    let d3 = f3 - 3.0 * f2 + 3.0 * f1 - f0;
    let c = vec4<f32>(f0, 3.0 * (d1 - 0.5 * d2 + d3 / 3.0), 9.0 * (0.5 * d2 - 0.5 * d3), 27.0 * d3 / 6.0);
    // Turning points: 3 c3 u^2 + 2 c2 u + c1 = 0, solved in the form that does not cancel
    // when the cubic is nearly a parabola (q = -(b + sign(b) sqrt(disc)) / 2, roots q / a
    // and c / q). The span's middle is cut too, so every piece is monotone and short.
    var r1 = 0.5;
    var r2 = 0.5;
    let qa = 3.0 * c.w;
    let qb = 2.0 * c.z;
    let disc = qb * qb - 4.0 * qa * c.y;
    if (disc >= 0.0) {
        let q = -0.5 * (qb + select(-1.0, 1.0, qb >= 0.0) * sqrt(disc));
        if (abs(qa) > 1e-20) {
            r1 = q / qa;
        }
        if (abs(q) > 1e-20) {
            r2 = c.y / q;
        }
    }
    let x = clamp(min(r1, r2), 0.0, 1.0);
    let y = clamp(max(r1, r2), 0.0, 1.0);
    // The span's ends in order, the middle (0.5) among them, written out: a small array
    // indexed in a loop is kept in slow local memory on some GPUs.
    var a1 = x;
    var a2 = y;
    var a3 = 0.5;
    if (0.5 < x) {
        a1 = 0.5;
        a2 = x;
        a3 = y;
    } else if (0.5 < y) {
        a2 = 0.5;
        a3 = y;
    }
    var r = span_root(c, 0.0, a1);
    if (r < 0.0) {
        r = span_root(c, a1, a2);
    }
    if (r < 0.0) {
        r = span_root(c, a2, a3);
    }
    if (r < 0.0) {
        r = span_root(c, a3, 1.0);
    }
    return r;
}

// The first zero of the cubic c in the span [lo, hi] where it is positive at lo, by
// bisection, when it is not positive at hi; else -1.
fn span_root(c: vec4<f32>, lo_in: f32, hi_in: f32) -> f32 {
    if (hi_in <= lo_in || cubic(c, hi_in) > 0.0) {
        return -1.0;
    }
    var lo = lo_in;
    var hi = hi_in;
    for (var it = 0; it < 20; it++) {
        let m = 0.5 * (lo + hi);
        if (cubic(c, m) > 0.0) {
            lo = m;
        } else {
            hi = m;
        }
    }
    return 0.5 * (lo + hi);
}

// Within the brick `slot` whose first node is `base`, cells of side h, the first hit along
// the ray between t_in and t_out: (t, normal), t < 0 when none.
fn march(slot: u32, base: vec3<f32>, h: f32, o: vec3<f32>, d: vec3<f32>, inv: vec3<f32>, t_in: f32, t_out: f32) -> vec4<f32> {
    let p = o + d * t_in;
    var c = vec3<i32>(clamp(floor((p - base) / h), vec3<f32>(0.0), vec3<f32>(7.0)));
    let stp = vec3<i32>(sign(d));
    var tmax = (base + (vec3<f32>(c) + select(vec3<f32>(0.0), vec3<f32>(1.0), d > vec3<f32>(0.0))) * h - o) * inv;
    let tdelta = abs(h * inv);
    var t = t_in;
    for (var s = 0u; s < 32u; s++) {
        let tc = min(min(tmax.x, tmax.y), tmax.z);
        let te = min(tc, t_out);
        // A cell none of whose corners reads zero or less holds no surface: in the stock's
        // bricks its bit says so, and the cell is passed without reading its corners.
        let ci = u32(c.x) + 8u * (u32(c.y) + 8u * u32(c.z));
        if (reading == 1u || ((cells[slot * CELL_WORDS + (ci >> 5u)] >> (ci & 31u)) & 1u) == 1u) {
            let n0 = u32(c.x) + 9u * (u32(c.y) + 9u * u32(c.z));
            let a = vec4<f32>(value(slot, n0), value(slot, n0 + 1u), value(slot, n0 + 9u), value(slot, n0 + 10u));
            let b = vec4<f32>(value(slot, n0 + 81u), value(slot, n0 + 82u), value(slot, n0 + 90u), value(slot, n0 + 91u));
            if (min(min(a.x, a.y), min(a.z, a.w)) <= 0.0 || min(min(b.x, b.y), min(b.z, b.w)) <= 0.0) {
                let cl = base + vec3<f32>(c) * h;
                let q0 = (o + d * t - cl) / h;
                let dq = d * (te - t) / h;
                let u = cell_hit(a, b, q0, dq);
                if (u >= 0.0) {
                    let q = q0 + dq * u;
                    return vec4<f32>(t + u * (te - t), grad(a, b, q));
                }
            }
        }
        if (tc >= t_out) {
            break;
        }
        t = tc;
        if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
            c.x += stp.x;
            tmax.x += tdelta.x;
        } else if (tmax.y <= tmax.z) {
            c.y += stp.y;
            tmax.y += tdelta.y;
        } else {
            c.z += stp.z;
            tmax.z += tdelta.z;
        }
        if (any(c < vec3<i32>(0)) || any(c > vec3<i32>(7))) {
            break;
        }
    }
    return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
}

// Level k of the cache read (`reading`): the stock's, or the part's as a Level.
fn level_at(k: u32) -> Level {
    if (reading == 1u) {
        let q = L.parts[k];
        return Level(q.lo, vec4<u32>(q.dims.xyz, 0u), vec4<u32>(q.dims.w, q.at.x, 0u, 0u));
    }
    return L.level[k];
}

// How many levels the cache read has.
fn level_count() -> u32 {
    return min(select(L.count.x, L.count.z, reading == 1u), MAX_LEVELS);
}

// The first hit in level k between t0 and t1, `sb` where the ray is in the stock box:
// (t, normal), t < 0 when none.
fn walk(k: u32, o: vec3<f32>, d: vec3<f32>, inv: vec3<f32>, t0: f32, t1: f32, sb: vec2<f32>) -> vec4<f32> {
    let lv = level_at(k);
    let h = lv.lo.w;
    let side = 8.0 * h;
    let dims = vec3<i32>(lv.dims.xyz);
    let stp = vec3<i32>(sign(d));
    let tdelta = abs(side * inv);
    var t = t0;
    var cell = vec3<i32>(0);
    var tmax = vec3<f32>(0.0);
    var fresh = true;
    for (var s = 0u; s < 4096u; s++) {
        if (fresh) {
            // The brick the ray is in just past t, and where it next crosses a brick face.
            let p0 = o + d * (t + 1e-4 * h);
            cell = vec3<i32>(clamp(floor((p0 - lv.lo.xyz) / side), vec3<f32>(0.0), vec3<f32>(dims) - 1.0));
            tmax = (lv.lo.xyz + (vec3<f32>(cell) + select(vec3<f32>(0.0), vec3<f32>(1.0), d > vec3<f32>(0.0))) * side - o) * inv;
            fresh = false;
        }
        let idx = u32(cell.x) + lv.dims.x * (u32(cell.y) + lv.dims.y * u32(cell.z));
        var entry: u32;
        if (reading == 1u) {
            entry = pbricks[lv.part.x + idx];
        } else {
            entry = bricks[lv.dims.w + idx];
        }
        if (entry < SKIP) {
            entry += lv.part.y;
        }
        let t_out = min(min(min(tmax.x, tmax.y), tmax.z), t1);
        if (entry >= SKIP && entry < FACES) {
            // Air for (entry & 255) - 1 bricks all round: on to where the ray leaves that cube.
            let r = f32((entry & 255u) - 1u);
            let cube = slab(o, inv, lv.lo.xyz + (vec3<f32>(cell) - r) * side, lv.lo.xyz + (vec3<f32>(cell) + r + 1.0) * side);
            if (cube.y >= t1) {
                break;
            }
            t = max(cube.y, t_out);
            fresh = true;
            continue;
        }
        if (entry == FACES) {
            // No move comes near, so inside the box the brick is all material: the ray meets
            // it where it enters the box, or where it enters the brick if already inside (out
            // of a cut next door, or out of another level).
            if (sb.x <= sb.y && sb.x <= t_out + 1e-3 * h && sb.y >= t) {
                hit_slot = AIR;
                if (sb.x >= t - 1e-3 * h) {
                    let a = (L.box_lo.xyz - o) * inv;
                    let b = (L.box_hi.xyz - o) * inv;
                    return vec4<f32>(sb.x, select(vec3<f32>(0.0), -sign(d), min(a, b) >= vec3<f32>(sb.x)));
                }
                return vec4<f32>(t, -d);
            }
        } else if (entry == SOLID) {
            hit_slot = AIR;
            return vec4<f32>(t, -d);
        } else if (entry != AIR) {
            let base = lv.lo.xyz + vec3<f32>(cell) * side;
            let r = march(entry, base, h, o, d, inv, t, t_out);
            if (r.x >= 0.0) {
                hit_slot = entry;
                hit_base = base;
                return r;
            }
        }
        if (t_out >= t1) {
            break;
        }
        t = t_out;
        if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
            cell.x += stp.x;
            tmax.x += tdelta.x;
        } else if (tmax.y <= tmax.z) {
            cell.y += stp.y;
            tmax.y += tdelta.y;
        } else {
            cell.z += stp.z;
            tmax.z += tdelta.z;
        }
        if (any(cell < vec3<i32>(0)) || any(cell >= dims)) {
            break;
        }
    }
    return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
}

fn box_sd(p: vec3<f32>) -> f32 {
    let q = abs(p - 0.5 * (L.box_lo.xyz + L.box_hi.xyz)) - 0.5 * (L.box_hi.xyz - L.box_lo.xyz);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// What a pixel's ray meets: (t along the unit ray, -1 for nothing; the normal there; the
// cell side of the level it met; the brick's slot, AIR for none, and first node), the ray
// itself, and the level it met.
struct Hit {
    t: f32,
    n: vec3<f32>,
    h: f32,
    slot: u32,
    base: vec3<f32>,
    o: vec3<f32>,
    d: vec3<f32>,
    k: u32,
};

fn trace_ray(ndc: vec2<f32>) -> Hit {
    let o = V.eye.xyz;
    let d = normalize(V.fwd.xyz + ndc.x * V.right.xyz + ndc.y * V.up.xyz);
    var hit = Hit(-1.0, vec3<f32>(0.0), 0.0, AIR, vec3<f32>(0.0), o, d, 0u);
    let inv = reciprocal(d);
    let n = level_count();
    // Where the ray is in each level.
    var enter: array<f32, 16>;
    var leave: array<f32, 16>;
    var t = BIG;
    for (var i = 0u; i < n; i++) {
        let lv = level_at(i);
        let s = slab(o, inv, lv.lo.xyz, lv.lo.xyz + vec3<f32>(lv.dims.xyz) * 8.0 * lv.lo.w);
        let ok = s.x <= s.y && s.y >= 0.0;
        enter[i] = select(BIG, max(s.x, 0.0), ok);
        leave[i] = select(-BIG, s.y, ok);
        t = min(t, enter[i]);
    }
    let sb = slab(o, inv, L.box_lo.xyz, L.box_hi.xyz);
    for (var piece = 0u; piece < 64u && t < BIG; piece++) {
        let past = t + max(abs(t), 1.0) * 1e-6;
        // The finest level holding the ray just past t.
        var k = n;
        for (var i = 0u; i < n; i++) {
            if (enter[i] <= past && past < leave[i]) {
                k = i;
                break;
            }
        }
        if (k == n) {
            // Between levels: on to the next one the ray enters.
            var next = BIG;
            for (var i = 0u; i < n; i++) {
                if (enter[i] > past) {
                    next = min(next, enter[i]);
                }
            }
            t = next;
            continue;
        }
        // The piece ends where level k does, or where a finer one begins.
        var t1 = leave[k];
        for (var i = 0u; i < k; i++) {
            if (enter[i] > past) {
                t1 = min(t1, enter[i]);
            }
        }
        let r = walk(k, o, d, inv, t, t1, sb);
        if (r.x >= 0.0) {
            hit.t = r.x;
            hit.n = r.yzw;
            hit.h = level_at(k).lo.w;
            hit.slot = hit_slot;
            hit.base = hit_base;
            hit.k = k;
            return hit;
        }
        t = t1;
    }
    return hit;
}

// The 2D distance to the triangle a, b, c (Quilez).
fn triangle2(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    let e0 = b - a;
    let e1 = c - b;
    let e2 = a - c;
    let v0 = p - a;
    let v1 = p - b;
    let v2 = p - c;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d = min(min(vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)), vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))), vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
    return -sqrt(d.x) * sign(d.y);
}

// The 2D distance to the box of half sizes b about the origin, its corners rounded by r.
fn round_box2(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let e = abs(p) - b + r;
    return length(max(e, vec2<f32>(0.0))) + min(max(e.x, e.y), 0.0) - r;
}

// The tool's distance at p, from its section in (distance from the axis, height above the
// tip).
fn tool_sd(p: vec3<f32>) -> f32 {
    let r = V.tool_at.w;
    let rc = V.tool_axis.w;
    let hv = V.tool_shape.x;
    let len = V.tool_shape.y;
    let a = V.tool_axis.xyz;
    let d = p - V.tool_at.xyz;
    let h = dot(d, a);
    let q = vec2<f32>(length(d - a * h), h);
    if (hv > 0.0) {
        let shank = round_box2(q - vec2<f32>(0.0, 0.5 * (hv + len)), vec2<f32>(r, 0.5 * (len - hv)), 0.0);
        return min(shank, triangle2(q, vec2<f32>(0.0, 0.0), vec2<f32>(r, hv), vec2<f32>(0.0, hv)));
    }
    return round_box2(q - vec2<f32>(0.0, 0.5 * len), vec2<f32>(r, 0.5 * len), rc);
}

// Where the ray meets the tool before t_max, or -1: sphere-traced inside the ball that
// holds the tool.
fn tool_hit(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> f32 {
    if (V.tool_shape.z < 0.5) {
        return -1.0;
    }
    // The cylinder that holds the tool (its radius about its axis, from its tip up its
    // stick-out): rays that miss it are done at once, and the trace starts on it, where the
    // tool's side is (most rays that meet the tool meet it there, in a step or two). A ball
    // about the whole tool sent every ray near it on a march down its length.
    let a = V.tool_axis.xyz;
    let r = V.tool_at.w * 1.001;
    let len = V.tool_shape.y;
    let oc = o - V.tool_at.xyz;
    let hd = dot(d, a);
    let h0 = dot(oc, a);
    let dd = d - a * hd;
    let od = oc - a * h0;
    let qa = dot(dd, dd);
    let qb = dot(od, dd);
    let qc = dot(od, od) - r * r;
    var t0 = -BIG;
    var t1 = BIG;
    if (qa > 1e-12) {
        let disc = qb * qb - qa * qc;
        if (disc < 0.0) {
            return -1.0;
        }
        let s = sqrt(disc);
        t0 = (-qb - s) / qa;
        t1 = (-qb + s) / qa;
    } else if (qc > 0.0) {
        return -1.0;
    }
    if (abs(hd) > 1e-12) {
        let ta = -h0 / hd;
        let tb = (len - h0) / hd;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    } else if (h0 < 0.0 || h0 > len) {
        return -1.0;
    }
    t1 = min(t1, t_max);
    var t = max(t0, 0.0);
    if (t > t1) {
        return -1.0;
    }
    for (var i = 0; i < 64; i++) {
        if (t > t1) {
            break;
        }
        let s = tool_sd(o + d * t);
        if (s < 1e-4 * (1.0 + t)) {
            return t;
        }
        t += s;
    }
    return -1.0;
}

fn section(at: u32, n: u32) -> f32 {
    let pair = unpack2x16snorm(heat[at + n / 2u]);
    return select(pair.x, pair.y, (n & 1u) == 1u) * REACH;
}

// The part's field at p, in the brick `slot` whose first node is `base`, cells of side h
// (heat.rs): min over its columns of max(section distance, z - top, bottom - z); BIG when
// none comes within REACH.
fn part_at(slot: u32, base: vec3<f32>, h: f32, p: vec3<f32>) -> f32 {
    let u = clamp((p.xy - base.xy) / h, vec2<f32>(0.0), vec2<f32>(8.0));
    let i = min(vec2<u32>(floor(u)), vec2<u32>(7u));
    let f = u - vec2<f32>(i);
    let n0 = i.x + 9u * i.y;
    var g = BIG;
    for (var j = 0u; j < HEAT_COLUMNS; j++) {
        let at = slot * HEAT_WORDS + j * COLUMN_WORDS;
        let top = bitcast<f32>(heat[at]);
        let bottom = bitcast<f32>(heat[at + 1u]);
        if (top < bottom) {
            continue;
        }
        let v = at + 2u;
        let sd = mix(mix(section(v, n0), section(v, n0 + 1u), f.x), mix(section(v, n0 + 9u), section(v, n0 + 10u), f.x), f.y);
        g = min(g, max(sd, max(p.z - top, bottom - p.z)));
    }
    return g;
}

fn ramp(a: vec3<f32>, m: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return select(mix(m, b, 2.0 * t - 1.0), mix(a, m, 2.0 * t), t < 0.5);
}

// The color of a cut surface the part's field reads g at (the shop-floor colors),
// on size within `tol`.
fn heat_color(g: f32, tol: f32) -> vec3<f32> {
    if (g >= 0.999 * REACH) {
        return vec3<f32>(0.373, 0.392, 0.412);
    }
    let t = clamp((abs(g) - tol) / max(V.heat.y - V.heat.x, 1e-6), 0.0, 1.0);
    if (g > tol) {
        return ramp(vec3<f32>(0.290, 0.706, 0.902), vec3<f32>(0.075, 0.494, 0.808), vec3<f32>(0.118, 0.278, 0.596), t);
    }
    if (g < -tol) {
        return ramp(vec3<f32>(0.937, 0.384, 0.275), vec3<f32>(0.816, 0.180, 0.118), vec3<f32>(0.608, 0.067, 0.067), t);
    }
    return vec3<f32>(0.424, 0.851, 0.506);
}

// The stock's cached field at p, mm, for the part's view: from the finest level that holds p
// and reads it short of its clamp (a value at the clamp is only a bound, so a coarser level,
// whose clamp is larger, is read for it). Box faces read the box's own distance.
fn stock_at(p: vec3<f32>) -> f32 {
    reading = 0u;
    var last = BIG;
    let n = min(L.count.x, MAX_LEVELS);
    for (var i = 0u; i < n; i++) {
        let lv = L.level[i];
        let h = lv.lo.w;
        let side = 8.0 * h;
        let q = (p - lv.lo.xyz) / side;
        if (any(q < vec3<f32>(0.0)) || any(q >= vec3<f32>(lv.dims.xyz))) {
            continue;
        }
        let cell = vec3<u32>(q);
        var entry = bricks[lv.dims.w + cell.x + lv.dims.x * (cell.y + lv.dims.y * cell.z)];
        if (entry < SKIP) {
            entry += lv.part.y;
        }
        let reach = h * 1.7320508;
        if (entry == FACES) {
            return box_sd(p);
        }
        if (entry == SOLID) {
            last = -reach;
            continue;
        }
        if (entry >= SKIP) {
            last = reach;
            continue;
        }
        let base = lv.lo.xyz + vec3<f32>(cell) * side;
        let u = (p - base) / h;
        let c = min(vec3<u32>(max(floor(u), vec3<f32>(0.0))), vec3<u32>(7u));
        let f = u - vec3<f32>(c);
        let n0 = c.x + 9u * (c.y + 9u * c.z);
        let a = vec4<f32>(value(entry, n0), value(entry, n0 + 1u), value(entry, n0 + 9u), value(entry, n0 + 10u));
        let b = vec4<f32>(value(entry, n0 + 81u), value(entry, n0 + 82u), value(entry, n0 + 90u), value(entry, n0 + 91u));
        let v = tri(a, b, f) * reach;
        if (abs(v) < 0.95 * reach) {
            return v;
        }
        last = v;
    }
    return last;
}

// The part's cached field at p, mm, as `stock_at` reads the stock's: from the finest of the
// part's levels that reads it short of its clamp; BIG outside the part past every level's
// reach (or with none yet), -BIG inside it so.
fn part_point(p: vec3<f32>) -> f32 {
    reading = 1u;
    var last = BIG;
    let n = level_count();
    for (var i = 0u; i < n; i++) {
        let lv = level_at(i);
        let h = lv.lo.w;
        let side = 8.0 * h;
        let q = (p - lv.lo.xyz) / side;
        if (any(q < vec3<f32>(0.0)) || any(q >= vec3<f32>(lv.dims.xyz))) {
            continue;
        }
        let cell = vec3<u32>(q);
        var entry = pbricks[lv.part.x + cell.x + lv.dims.x * (cell.y + lv.dims.y * cell.z)];
        if (entry < SKIP) {
            entry += lv.part.y;
        }
        if (entry == SOLID) {
            last = -BIG;
            continue;
        }
        if (entry >= SKIP) {
            last = BIG;
            continue;
        }
        let reach = h * 1.7320508;
        let base = lv.lo.xyz + vec3<f32>(cell) * side;
        let u = (p - base) / h;
        let c = min(vec3<u32>(max(floor(u), vec3<f32>(0.0))), vec3<u32>(7u));
        let f = u - vec3<f32>(c);
        let n0 = c.x + 9u * (c.y + 9u * c.z);
        let a = vec4<f32>(value(entry, n0), value(entry, n0 + 1u), value(entry, n0 + 9u), value(entry, n0 + 10u));
        let b = vec4<f32>(value(entry, n0 + 81u), value(entry, n0 + 82u), value(entry, n0 + 90u), value(entry, n0 + 91u));
        let v = tri(a, b, f) * reach;
        if (abs(v) < 0.95 * reach) {
            reading = 0u;
            return v;
        }
        last = select(-BIG, BIG, v > 0.0);
    }
    reading = 0u;
    return last;
}

// The cache read (`reading`) at p from level k alone, mm: (value, 1) where a brick of that
// level with values holds p (in the stock's, a brick of the box's faces alone reads the box),
// (0, 0) where none does.
fn level_value(k: u32, p: vec3<f32>) -> vec2<f32> {
    let lv = level_at(k);
    let h = lv.lo.w;
    let side = 8.0 * h;
    let q = (p - lv.lo.xyz) / side;
    if (any(q < vec3<f32>(0.0)) || any(q >= vec3<f32>(lv.dims.xyz))) {
        return vec2<f32>(0.0);
    }
    let cell = vec3<u32>(q);
    let idx = cell.x + lv.dims.x * (cell.y + lv.dims.y * cell.z);
    var entry: u32;
    if (reading == 1u) {
        entry = pbricks[lv.part.x + idx];
    } else {
        entry = bricks[lv.dims.w + idx];
    }
    if (entry < SKIP) {
        entry += lv.part.y;
    }
    if (entry == FACES && reading == 0u) {
        return vec2<f32>(box_sd(p), 1.0);
    }
    if (entry >= SKIP) {
        return vec2<f32>(0.0);
    }
    let base = lv.lo.xyz + vec3<f32>(cell) * side;
    let u = (p - base) / h;
    let c = min(vec3<u32>(max(floor(u), vec3<f32>(0.0))), vec3<u32>(7u));
    let n0 = c.x + 9u * (c.y + 9u * c.z);
    let a = vec4<f32>(value(entry, n0), value(entry, n0 + 1u), value(entry, n0 + 9u), value(entry, n0 + 10u));
    let b = vec4<f32>(value(entry, n0 + 81u), value(entry, n0 + 82u), value(entry, n0 + 90u), value(entry, n0 + 91u));
    return vec2<f32>(tri(a, b, u - vec3<f32>(c)) * h * 1.7320508, 1.0);
}

// The cache's slope at p along the step s (length e), mm a mm: centered where both sides
// hold values, one-sided from p where only one does (a brick with none, or past the level),
// `own` (the cell's own) where neither does.
fn slope(k: u32, p: vec3<f32>, s: vec3<f32>, e: f32, own: f32) -> f32 {
    let up = level_value(k, p + s);
    let down = level_value(k, p - s);
    if (up.y > 0.0 && down.y > 0.0) {
        return (up.x - down.x) / (2.0 * e);
    }
    if (up.y > 0.0 || down.y > 0.0) {
        let mid = level_value(k, p).x;
        return select(mid - down.x, up.x - mid, up.y > 0.0) / e;
    }
    return own;
}

// The normal the light reads at a hit in a brick with values: the cache's gradient by
// central differences half a cell either side, in the level the ray met. A difference of
// the interpolant is continuous wherever it is read, so the normal changes smoothly across
// cell and brick faces (the interpolant's own gradient, `hit.n`, jumps at every cell face).
// Read at the four corners of a tetrahedron about the hit (after Quilez): their signed sum
// is a centered difference along each axis, from four reads instead of six. Where one of
// them holds no values, axis by axis (`slope`).
fn lit_normal(hit: Hit) -> vec3<f32> {
    let p = hit.o + hit.d * hit.t;
    let e = 0.5 * hit.h;
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);
    let v0 = level_value(hit.k, p + e * k0);
    let v1 = level_value(hit.k, p + e * k1);
    let v2 = level_value(hit.k, p + e * k2);
    let v3 = level_value(hit.k, p + e * k3);
    if (v0.y > 0.0 && v1.y > 0.0 && v2.y > 0.0 && v3.y > 0.0) {
        let g = k0 * v0.x + k1 * v1.x + k2 * v2.x + k3 * v3.x;
        if (dot(g, g) > 1e-12 * e * e) {
            return g;
        }
        return hit.n;
    }
    // The cell's own gradient, mm a mm (hit.n is per cell, of values over the diagonal).
    let own = hit.n * 1.7320508;
    let g = vec3<f32>(slope(hit.k, p, vec3<f32>(e, 0.0, 0.0), e, own.x), slope(hit.k, p, vec3<f32>(0.0, e, 0.0), e, own.y), slope(hit.k, p, vec3<f32>(0.0, 0.0, e), e, own.z));
    if (dot(g, g) < 1e-12) {
        return hit.n;
    }
    return g;
}

fn shaded(base: vec3<f32>, p: vec3<f32>, n_in: vec3<f32>) -> vec3<f32> {
    let view = normalize(V.eye.xyz - p);
    var n = normalize(n_in);
    if (dot(n, view) < 0.0) {
        n = -n;
    }
    let l = normalize(V.light.xyz);
    let half = normalize(l + view);
    let spec = pow(max(dot(n, half), 0.0), 48.0);
    let shade = 0.22 + 0.58 * max(dot(n, l), 0.0) + 0.22 * max(dot(n, view), 0.0);
    return base * shade + vec3<f32>(0.22 * spec);
}

@fragment
fn fs(i: VOut) -> FOut {
    var out: FOut;
    // The screen's height in ndc per pixel (read here, where every pixel of the quad reads it).
    let ndc_px = fwidth(i.ndc.y);
    out.color = vec4<f32>(0.11, 0.11, 0.12, 1.0);
    out.depth = 1.0;
    let part_view = V.heat.z > 0.5;
    if (part_view) {
        reading = 1u;
    }
    let hit = trace_ray(i.ndc);
    let t_tool = tool_hit(hit.o, hit.d, select(BIG, hit.t, hit.t >= 0.0));
    if (t_tool >= 0.0) {
        let p = hit.o + hit.d * t_tool;
        let e = 1e-4 * (1.0 + t_tool);
        let g = vec3<f32>(tool_sd(p + vec3<f32>(e, 0.0, 0.0)) - tool_sd(p - vec3<f32>(e, 0.0, 0.0)), tool_sd(p + vec3<f32>(0.0, e, 0.0)) - tool_sd(p - vec3<f32>(0.0, e, 0.0)), tool_sd(p + vec3<f32>(0.0, 0.0, e)) - tool_sd(p - vec3<f32>(0.0, 0.0, e)));
        out.color = vec4<f32>(shaded(vec3<f32>(0.86, 0.68, 0.30), p, g), 1.0);
        let clip = V.view_proj * vec4<f32>(p, 1.0);
        out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
        return out;
    }
    if (hit.t < 0.0) {
        return out;
    }
    let p = hit.o + hit.d * hit.t;
    // The light reads a normal that changes smoothly across cells; the slack below reads the
    // cell's own gradient, whose length says where the surface is a plane. Read before
    // `stock_at`, which turns `reading` to the stock's. Only where cells are small on screen:
    // the cache rounds over a cell what it cannot hold, and over cells tens of pixels wide the
    // smooth normal drew sharp edges as fillets (finer cells are the cure there, not light).
    var lit = hit.n;
    if (hit.slot != AIR) {
        let cell_px = hit.h / max(hit.t * length(V.up.xyz) * ndc_px, 1e-12);
        let w = smoothstep(0.5 * FACET_PX, FACET_PX, cell_px) * (1.0 - smoothstep(SMOOTH_PX, 2.0 * SMOOTH_PX, cell_px));
        if (w > 0.0) {
            lit = mix(normalize(hit.n), normalize(lit_normal(hit)), w);
        }
    }
    if (part_view) {
        // The part, colored by the stock at its surface: stock left reads below zero.
        let steep = length(hit.n) * sqrt(3.0);
        let slack = select(0.0, 0.35 * hit.h, abs(steep - 1.0) > 0.05);
        let g = -stock_at(p);
        out.color = vec4<f32>(shaded(heat_color(g, V.heat.x + slack), p, lit), 1.0);
        let clip = V.view_proj * vec4<f32>(p, 1.0);
        out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
        return out;
    }
    let raw = abs(box_sd(p)) < 0.3 * hit.h;
    var base = select(vec3<f32>(0.80, 0.82, 0.86), vec3<f32>(0.50, 0.52, 0.56), raw);
    if (raw) {
        // A face of the stock no move touched: where it lies on the part it is the part's
        // face (on size) or stock left over it, else raw stock. In a brick a move came near,
        // the part is read from the brick's own records, as the cut faces beside it are (they
        // come with the level, out to REACH); elsewhere from the part's cache. A plane is
        // interpolated exactly; at the part's outline the reading bends, so a third of a
        // cell of slack.
        var g = BIG;
        if (L.count.y == 1u && hit.slot != AIR) {
            g = part_at(hit.slot, hit.base, hit.h, p);
        } else {
            g = part_point(p);
        }
        if (abs(g) < BIG) {
            base = heat_color(g, V.heat.x + 0.35 * hit.h);
        }
    } else if (L.count.y == 1u && hit.slot != AIR) {
        // Where the stock's surface is a plane through the cell, the drawn surface is the
        // stock's; at its edges the interpolant rounds them, by up to about a third of a
        // cell, and there the gradient's length (1 on a plane; the values are F over the
        // cell's diagonal) falls short. There the part is read with that much more slack.
        let steep = length(hit.n) * sqrt(3.0);
        let slack = select(0.0, 0.35 * hit.h, abs(steep - 1.0) > 0.05);
        base = heat_color(part_at(hit.slot, hit.base, hit.h, p), V.heat.x + slack);
    }
    out.color = vec4<f32>(shaded(base, p, lit), 1.0);
    let clip = V.view_proj * vec4<f32>(p, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}

// What `fs` draws at the pixel, for checking the caster against the cache and for what lies
// under the mouse: the point met (the part's view, the stock's or the tool) and t (t = -1:
// none).
@fragment
fn fs_hit(i: VOut) -> @location(0) vec4<f32> {
    if (V.heat.z > 0.5) {
        reading = 1u;
    }
    let hit = trace_ray(i.ndc);
    let t_tool = tool_hit(hit.o, hit.d, select(BIG, hit.t, hit.t >= 0.0));
    let t = select(hit.t, t_tool, t_tool >= 0.0);
    return vec4<f32>(hit.o + hit.d * max(t, 0.0), t);
}
