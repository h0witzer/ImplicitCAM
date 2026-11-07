// contact.wgsl - the core contact query, one thread per query point.
//
//   mode 0 (drop)   lowest gouge-free tip height: max over faces of z_F - h_min(d_F)
//   mode 1 (slice)  horizontal clearance at tip height z: min over faces above z of sd_F - R(z_F - z)
//
// Per face, the distance comes from its nearest boundary curve listed in this
// query's grid cell: lines and arcs in closed form, B-splines as flat rational
// Bezier pieces (certified lower bound: distance to the chord minus the piece's
// thickness; exact foot by Newton on the rational curve). When that distance is
// within band it is the face's true distance (see pack.rs) and the sign comes
// from the topology: the outward normal at an interior foot, the vertex
// pseudo-normal at an end. Beyond band the side comes from the cell's inside
// bit for that face. Coordinates are in the part's local frame, single precision.

struct Params {
    gx0: f32, gy0: f32, cell: f32, band: f32,
    nx: u32, ny: u32, mode: u32, n: u32,
    cell_start: u32, cell_items: u32, cover_start: u32, cover_items: u32,
    ntool: u32, r_max: f32, width: u32, cell_ref: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> edges: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> face_z: array<f32>;
@group(0) @binding(3) var<storage, read> idx: array<u32>;
@group(0) @binding(4) var<storage, read> tool: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> outp: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> bez: array<vec4<f32>>;

const NONE: u32 = 0xffffffffu;
const TWO_PI: f32 = 6.28318530717958;
const CLAMP_COVER: u32 = 1u;   // slice: a face more than band deep decided it; value is an upper bound
const CLAMP_BAND: u32 = 2u;    // slice: nothing within band decided it; value is a lower bound
const NO_PARAM: f32 = -3.0e38;  // no curve parameter (interior contact); the host reads NaN
const STEEP: u32 = 4u;         // drop: contact on a near-vertical part of the tool (|dh/dr| > 50);
                               // f32 loses digits there, so certify it on the CPU

struct HMin { h: f32, dh: f32, d2h: f32, el: u32, ok: bool, };
struct RadEl { r: f32, el: u32, };

// Height of the tool's lowest point at radius r, with d/dr and d2/dr2.
fn h_min(r_in: f32) -> HMin {
    let r = max(r_in, 0.0);
    for (var i = 0u; i < P.ntool; i++) {
        let a = tool[2u * i];
        let kind = tool[2u * i + 1u].x;
        if (kind < 0.5) {
            if (a.z - a.x <= 1e-7 || r > a.z + 1e-6) { continue; }
            let m = (a.w - a.y) / (a.z - a.x);
            return HMin(a.y + (max(r, a.x) - a.x) * m, m, 0.0, i, true);
        } else {
            if (r > a.x + a.z + 1e-6) { continue; }
            let x = clamp(r - a.x, 0.0, a.z);
            let w = sqrt(max(a.z * a.z - x * x, 1e-20));
            return HMin(a.y - w, x / w, a.z * a.z / (w * w * w), i, true);
        }
    }
    return HMin(0.0, 0.0, 0.0, NONE, false);
}

// Tool radius at height h above the tip (the widest element there).
fn radius_at(h: f32) -> RadEl {
    var best = RadEl(-1.0, NONE);
    for (var i = 0u; i < P.ntool; i++) {
        let a = tool[2u * i];
        let kind = tool[2u * i + 1u].x;
        var r = -1.0;
        if (kind < 0.5) {
            if (a.w - a.y <= 1e-7) {
                if (abs(h - a.y) <= 1e-5) { r = max(a.x, a.z); }
            } else if (h >= a.y - 1e-6 && h <= a.w + 1e-6) {
                r = a.x + (clamp(h, a.y, a.w) - a.y) * (a.z - a.x) / (a.w - a.y);
            }
        } else if (h >= a.y - a.z - 1e-6 && h <= a.y + 1e-6) {
            let y = clamp(a.y - h, 0.0, a.z);
            r = a.x + sqrt(max(a.z * a.z - y * y, 0.0));
        }
        if (r > best.r) { best = RadEl(r, i); }
    }
    if (best.el == NONE) {
        let last = P.ntool - 1u;
        let a = tool[2u * last];
        var rend = a.z;
        if (tool[2u * last + 1u].x >= 0.5) { rend = a.x + a.z; }
        best = RadEl(rend, last);
    }
    return best;
}

// ------------------------------------------------------------------ Bezier pieces

struct BezEval { p: vec2<f32>, d1: vec2<f32>, d2: vec2<f32>, };

// Point and first two derivatives of rational Bezier piece `k` at local s
// (de Casteljau in homogeneous space, then the quotient rule).
fn bez_eval(k: u32, s: f32) -> BezEval {
    let base = 9u * k;
    let deg = bitcast<u32>(bez[base + 8u].x);
    var b: array<vec3<f32>, 8>;
    for (var i = 0u; i <= deg; i++) { b[i] = bez[base + i].xyz; }
    var d1h = vec3<f32>(0.0);
    var d2h = vec3<f32>(0.0);
    for (var level = 1u; level <= deg; level++) {
        if (deg >= 2u && level == deg - 1u) { d2h = f32(deg * (deg - 1u)) * (b[2] - 2.0 * b[1] + b[0]); }
        if (level == deg) { d1h = f32(deg) * (b[1] - b[0]); }
        for (var i = 0u; i <= deg - level; i++) { b[i] = mix(b[i], b[i + 1u], s); }
    }
    let w = b[0].z;
    let p = b[0].xy / w;
    let d1 = (d1h.xy - d1h.z * p) / w;
    let d2 = (d2h.xy - 2.0 * d1h.z * d1 - d2h.z * p) / w;
    return BezEval(p, d1, d2);
}

fn bez_end(k: u32, last: bool) -> vec2<f32> {
    let base = 9u * k;
    var h = bez[base].xyz;
    if (last) { h = bez[base + bitcast<u32>(bez[base + 8u].x)].xyz; }
    return h.xy / h.z;
}

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let l2 = dot(ab, ab);
    var t = 0.0;
    if (l2 > 0.0) { t = clamp(dot(p - a, ab) / l2, 0.0, 1.0); }
    return length(p - (a + ab * t));
}

// ------------------------------------------------------------------ nearest point

// Nearest point of one curve: distance, foot, outward normal there, the
// unsigned distance's Hessian as k * (I - n n^T), and the curve parameter.
struct Near { d: f32, q: vec2<f32>, outw: vec2<f32>, n: vec2<f32>, k: f32, t: f32, };

fn at_point(p: vec2<f32>, q: vec2<f32>, outw: vec2<f32>, t: f32) -> Near {
    let d = length(p - q);
    var n = vec2<f32>(0.0);
    var k = 0.0;
    if (d > 0.0) { n = (p - q) / d; k = 1.0 / d; }
    return Near(d, q, outw, n, k, t);
}

fn curve_near(e: u32, p: vec2<f32>, cutoff: f32) -> Near {
    let e0 = edges[4u * e];
    let e1 = edges[4u * e + 1u];
    let e2 = edges[4u * e + 2u];
    let kind = bitcast<u32>(edges[4u * e + 3u].x);
    let a = e0.xy;
    let b = e0.zw;
    if (kind == 0u) {
        let ab = b - a;
        let l2 = dot(ab, ab);
        var t = 0.0;
        if (l2 > 0.0) { t = clamp(dot(p - a, ab) / l2, 0.0, 1.0); }
        if (t <= 0.0) { return at_point(p, a, e2.xy, 0.0); }
        if (t >= 1.0) { return at_point(p, b, e2.zw, 1.0); }
        let q = a + ab * t;
        return Near(length(p - q), q, normalize(vec2<f32>(ab.y, -ab.x)), vec2<f32>(0.0), 0.0, t);
    }
    if (kind == 1u) {
        let c = e1.xy;
        let r = e1.z;
        let sw = e1.w;
        let u = p - c;
        let s = length(u);
        let full = abs(abs(sw) - TWO_PI) < 1e-5;
        let ua = a - c;
        var ang = atan2(ua.x * u.y - ua.y * u.x, dot(ua, u));
        if (sw < 0.0) { ang = -ang; }
        if (ang < 0.0) { ang = ang + TWO_PI; }
        if (s > 0.0 && (full || ang <= abs(sw))) {
            let un = u / s;
            return Near(abs(s - r), c + un * r, un * sign(sw), un, sign(s - r) / s, ang / abs(sw));
        }
        if (length(p - a) <= length(p - b)) { return at_point(p, a, e2.xy, 0.0); }
        return at_point(p, b, e2.zw, 1.0);
    }
    // B-spline: flat rational Bezier pieces [first, first + count).
    let first = bitcast<u32>(e1.x);
    let count = bitcast<u32>(e1.y);
    var best = Near(3.0e38, a, e2.xy, vec2<f32>(0.0), 0.0, 0.0);
    var best_k = NONE;
    var best_s = 0.0;
    for (var j = 0u; j < count; j++) {
        let k = first + j;
        let pa = bez_end(k, false);
        let pb = bez_end(k, true);
        let lb = seg_dist(p, pa, pb) - bez[9u * k + 8u].y;
        if (lb >= min(best.d, cutoff)) { continue; }
        // Newton on f(s) = C'(s) . (C(s) - p) from the chord projection, then the ends.
        let ab = pb - pa;
        var s = 0.5;
        if (dot(ab, ab) > 0.0) { s = clamp(dot(p - pa, ab) / dot(ab, ab), 0.0, 1.0); }
        for (var it = 0; it < 12; it++) {
            let ev = bez_eval(k, s);
            let rr = ev.p - p;
            var fp = dot(ev.d2, rr) + dot(ev.d1, ev.d1);
            // Gauss-Newton where the second derivative is negative.
            if (fp <= 0.0) { fp = dot(ev.d1, ev.d1); }
            if (abs(fp) < 1e-30) { break; }
            s = clamp(s - dot(ev.d1, rr) / fp, 0.0, 1.0);
        }
        var cand = array<f32, 3>(s, 0.0, 1.0);
        for (var c = 0; c < 3; c++) {
            let ev = bez_eval(k, cand[c]);
            let d = length(p - ev.p);
            if (d < best.d) { best.d = d; best.q = ev.p; best_k = k; best_s = cand[c]; }
        }
    }
    if (best_k == NONE) { return best; }
    let ev = bez_eval(best_k, best_s);
    let hdr = bez[9u * best_k + 8u];
    let t = hdr.z + best_s * (hdr.w - hdr.z);
    if (best_k == first && best_s <= 0.0) { return at_point(p, best.q, e2.xy, t); }
    if (best_k == first + count - 1u && best_s >= 1.0) { return at_point(p, best.q, e2.zw, t); }
    let sp = length(ev.d1);
    let tan = ev.d1 / sp;
    let kappa = (ev.d1.x * ev.d2.y - ev.d1.y * ev.d2.x) / (sp * sp * sp);
    var n = vec2<f32>(0.0);
    var hk = 0.0;
    if (best.d > 0.0) {
        n = (p - best.q) / best.d;
        let kb = -kappa * dot(n, vec2<f32>(-tan.y, tan.x));
        hk = kb / (1.0 + kb * best.d);
    }
    return Near(best.d, best.q, vec2<f32>(tan.y, -tan.x), n, hk, t);
}

// ------------------------------------------------------------------ result

struct Res {
    value: f32, grad: vec2<f32>, hess: vec3<f32>, q: vec2<f32>, zf: f32,
    face: u32, edge: u32, el: u32, second: f32, second_face: u32, flags: u32, t: f32,
};

fn better(a: f32, b: f32) -> bool {
    if (P.mode == 0u) { return a > b; }
    return a < b;
}

fn offer(res: ptr<function, Res>, t: Res) {
    if (better(t.value, (*res).value)) {
        let s = (*res).value;
        let sf = (*res).face;
        *res = t;
        (*res).second = s;
        (*res).second_face = sf;
    } else if (better(t.value, (*res).second)) {
        (*res).second = t.value;
        (*res).second_face = t.face;
    }
}

// The term one face contributes, from its signed distance sd, the unit
// gradient g of sd, and the Hessian of sd.
fn term(f: u32, p: vec2<f32>, ztip: f32, sd: f32, q: vec2<f32>, e: u32, g: vec2<f32>, hs: vec3<f32>, flags: u32, tpar: f32) -> Res {
    var t = Res(0.0, vec2<f32>(0.0), vec3<f32>(0.0), q, face_z[f], f, e, NONE, 0.0, NONE, 0u, tpar);
    let zf = face_z[f];
    if (P.mode == 0u) {
        let dd = max(sd, 0.0);
        let hm = h_min(dd);
        if (dd > P.r_max || !hm.ok) { t.value = -3.0e38; t.face = NONE; return t; }
        t.value = zf - hm.h;
        t.el = hm.el;
        if (hm.dh > 50.0) { t.flags = STEEP; }
        if (sd > 0.0) {
            t.grad = -hm.dh * g;
            t.hess = vec3<f32>(-hm.d2h * g.x * g.x, -hm.d2h * g.x * g.y, -hm.d2h * g.y * g.y) - hm.dh * hs;
        } else {
            t.q = p;
            t.edge = NONE;
            t.t = NO_PARAM;
        }
    } else {
        if (zf <= ztip + 1e-5) { t.value = 3.0e38; t.face = NONE; return t; }
        let re = radius_at(zf - ztip);
        t.value = sd - re.r;
        t.el = re.el;
        t.grad = g;
        t.hess = hs;
        t.flags = flags;
    }
    return t;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.y * P.width + gid.x;
    if (qi >= P.n) { return; }
    let qv = queries[qi];
    let p = qv.xy;
    let ztip = qv.z;
    var worst = -3.0e38;
    if (P.mode == 1u) { worst = 3.0e38; }
    var res = Res(worst, vec2<f32>(0.0), vec3<f32>(0.0), p, 0.0, NONE, NONE, NONE, worst, NONE, 0u, NO_PARAM);

    let ci = i32(floor((p.x - P.gx0) / P.cell));
    let cj = i32(floor((p.y - P.gy0) / P.cell));
    if (ci >= 0 && cj >= 0 && ci < i32(P.nx) && cj < i32(P.ny)) {
        let c = u32(cj) * P.nx + u32(ci);
        let i0 = idx[P.cell_start + c];
        let i1 = idx[P.cell_start + c + 1u];
        var cur = NONE;
        var cur_bit = 0u;
        var face_ub = 3.0e38;
        var best = Near(3.0e38, vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0);
        var best_e = NONE;
        for (var k = i0; k <= i1; k++) {
            var e = NONE;
            var f = NONE;
            var bit = 0u;
            if (k < i1) {
                let item = idx[P.cell_items + k];
                e = item & 0x7fffffffu;
                bit = item >> 31u;
                f = bitcast<u32>(edges[4u * e + 3u].y);
            }
            if (f != cur && cur != NONE) {
                // Finish face `cur`.
                if (best.d <= P.band) {
                    // Exact distance; the sign from the topology at the foot.
                    var sgn = 1.0;
                    if (dot(p - best.q, best.outw) < 0.0) { sgn = -1.0; }
                    var g = best.outw;
                    if (best.d > 0.0) { g = sgn * (p - best.q) / best.d; }
                    let nn = best.n;
                    let hs = sgn * best.k * vec3<f32>(1.0 - nn.x * nn.x, -nn.x * nn.y, 1.0 - nn.y * nn.y);
                    offer(&res, term(cur, p, ztip, sgn * best.d, best.q, best_e, g, hs, 0u, best.t));
                } else if (cur_bit == 1u) {
                    // Deeper than band inside: the cell's side for this face.
                    offer(&res, term(cur, p, ztip, -P.band, p, NONE, vec2<f32>(0.0), vec3<f32>(0.0), CLAMP_COVER, NO_PARAM));
                }
                best.d = 3.0e38;
            }
            if (k == i1) { break; }
            if (f != cur) {
                cur_bit = bit;
                // Upper bound for this face's distance from its curves' end points, so
                // spline pieces farther than that are pruned before any Newton step.
                face_ub = 3.0e38;
                for (var j = k; j < i1; j++) {
                    let ej = idx[P.cell_items + j] & 0x7fffffffu;
                    if (bitcast<u32>(edges[4u * ej + 3u].y) != f) { break; }
                    let e0 = edges[4u * ej];
                    face_ub = min(face_ub, min(length(p - e0.xy), length(p - e0.zw)));
                }
            }
            cur = f;
            let nr = curve_near(e, p, min(best.d, face_ub));
            if (nr.d < best.d) { best = nr; best_e = bitcast<u32>(edges[4u * e + 3u].z); }
        }
        let j0 = idx[P.cover_start + c];
        let j1 = idx[P.cover_start + c + 1u];
        for (var k = j0; k < j1; k++) {
            let f = idx[P.cover_items + k];
            offer(&res, term(f, p, ztip, -P.band, p, NONE, vec2<f32>(0.0), vec3<f32>(0.0), CLAMP_COVER, NO_PARAM));
        }
    }
    if (P.mode == 1u && res.value >= P.band - P.r_max) {
        res.value = P.band - P.r_max;
        res.flags = res.flags | CLAMP_BAND;
    }
    let o = 4u * qi;
    outp[o] = vec4<f32>(res.value, res.grad.x, res.grad.y, res.second);
    outp[o + 1u] = vec4<f32>(res.hess, res.zf);
    outp[o + 2u] = vec4<f32>(res.q, bitcast<f32>(res.flags), res.t);
    outp[o + 3u] = vec4<f32>(bitcast<f32>(res.face), bitcast<f32>(res.edge), bitcast<f32>(res.el), bitcast<f32>(res.second_face));
}
