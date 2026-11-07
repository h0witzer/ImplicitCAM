// Sweep and stock fields on the GPU, one point per invocation (see gpu.rs). The tools'
// distances and the machines' chains are generated per program and appended to this
// file as straight-line code; each move arrives as a record.
//
// A move's field over a stretch of t: the point's path through the tool frame, q(t),
// strays from its chord between the stretch's ends by at most dt^2 / 8 times |q''|, and
// the tool's distance changes no faster than distance, so the least of g over the stretch
// is at least the least of the tool's distance along the chord (a convex problem for a
// convex tool, solved by golden section) less that stray. Every value read along the way
// is an actual g at some t, so the least of them bounds the field from above.
//
// Each point gets an interval [lo, hi] holding the field clamped to [-limit, limit];
// (modes 2 and 3) each side of a piece the fraction along it where the field the caller
// cuts by, sgn field + off, crosses zero; or (modes 4 and 5) each point moved onto the
// surface of the move it came from, and on request onto the cut's surface along it.

// mode: 0 one move's field, 1 the stock's; 2 and 3 the same, a root along each side;
// 4 and 5 the same, points settled onto their moves' surfaces (gpu.rs, `settle`); 6 the
// stock's at points each reading its own list of moves (gpu.rs, `listed`); 7 the same at
// every node of bricks, each brick reading its own list (gpu.rs, `bricks`); 8 as 6, the
// moves alone without the stock box (gpu.rs, `listed_moves`).
// The box is the stock's, in this frame.
struct Params {
    n: u32,
    count: u32,
    mode: u32,
    width: u32,
    limit: f32,
    tol: f32,
    sgn: f32,
    off: f32,
    box_lo: vec4<f32>,
    box_hi: vec4<f32>,
};

// head: (kind, tool, machine, 0); kind 0 is straight in pose space, 1 an arc or helix,
// 2 a machine's axes run linearly, 3 a level arc and 4 a level line (both across the
// tool's axis, in closed form). lo, hi: a box holding the move. a to d: the motion.
// e: (the origin's path length, the magnitude rounding scales with, k0, k1), where
// |q''| <= k0 + k1 |q| along the move. f: where the machine's origin sits in this frame.
struct Sweep {
    head: vec4<u32>,
    lo: vec4<f32>,
    hi: vec4<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
    e: vec4<f32>,
    f: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> sweeps: array<Sweep>;
@group(0) @binding(2) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f32>>;
// Mode 6: the moves each point reads, as indices into `sweeps`.
@group(0) @binding(4) var<storage, read> idx: array<u32>;

const BIG: f32 = 1e30;

// The time at which the last sweep_iv found its least value (where the move comes
// nearest the point), or -1 when it read no time (a closed form, or a box too far).
var<private> best_t: f32;
// Rounding per unit of magnitude: 2^-21, a dozen roundings of single precision.
const ROUND: f32 = 4.76837158e-7;

struct Pose {
    r: vec4<f32>,
    at: vec3<f32>,
};

// a after b; quaternions as (x, y, z, w).
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn about(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    return vec4<f32>(axis * sin(0.5 * angle), cos(0.5 * angle));
}

fn compose(a: Pose, b: Pose) -> Pose {
    return Pose(qmul(a.r, b.r), qrot(a.r, b.at) + a.at);
}

fn to_local(pose: Pose, p: vec3<f32>) -> vec3<f32> {
    return qrot(vec4<f32>(-pose.r.xyz, pose.r.w), p - pose.at);
}

fn seg_d(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = b - a;
    let l2 = dot(d, d);
    let t = select(0.0, clamp(dot(p - a, d) / l2, 0.0, 1.0), l2 > 0.0);
    return length(p - a - t * d);
}

fn arc_d(p: vec2<f32>, c: vec2<f32>, rad: f32, a0: f32, a1: f32) -> f32 {
    let v = p - c;
    let ang = atan2(v.y, v.x);
    if (ang >= a0 && ang <= a1) {
        return abs(length(v) - rad);
    }
    let e0 = c + rad * vec2<f32>(cos(a0), sin(a0));
    let e1 = c + rad * vec2<f32>(cos(a1), sin(a1));
    return min(length(p - e0), length(p - e1));
}

fn box_dist(lo: vec3<f32>, hi: vec3<f32>, p: vec3<f32>) -> f32 {
    return length(max(max(lo - p, p - hi), vec3<f32>(0.0)));
}

fn pose_at(s: Sweep, t: f32) -> Pose {
    switch s.head.x {
        // a: the start's rotation, b: (start, turn angle), c: the turn's axis in the
        // start's frame, d: the origin's displacement.
        case 0u: {
            return Pose(qmul(s.a, about(s.c.xyz, s.b.w * t)), s.b.xyz + s.d.xyz * t);
        }
        // a: (center, sweep), b: (unit normal, rise), c: start less center, d: rotation.
        case 1u: {
            let al = s.a.w * t;
            let n = s.b.xyz;
            let r = s.c.xyz;
            let k = cos(al);
            let turned = r * k + cross(n, r) * sin(al) + n * (dot(n, r) * (1.0 - k));
            return Pose(s.d, s.a.xyz + turned + n * (s.b.w * t));
        }
        // a, b: the axes at the start; c, d: at the end; head.z: the machine.
        default: {
            let pose = machine(s.head.z, s.a + (s.c - s.a) * t, s.b + (s.d - s.b) * t);
            return Pose(pose.r, pose.at - s.f.xyz);
        }
    }
}

// Golden section for the least of the tool's distance along the chord from qa to qb
// (convex for a convex tool): (least value, where in [0, 1]).
fn golden(tool: u32, qa: vec3<f32>, qb: vec3<f32>, ga: f32, gb: f32, steps: u32) -> vec2<f32> {
    let R = 0.618034;
    var lo = 0.0;
    var hi = 1.0;
    var c = hi - R * (hi - lo);
    var d = lo + R * (hi - lo);
    var fc = tool_sd(tool, mix(qa, qb, c));
    var fd = tool_sd(tool, mix(qa, qb, d));
    for (var i = 0u; i < steps; i++) {
        if (fc <= fd) {
            hi = d;
            d = c;
            fd = fc;
            c = hi - R * (hi - lo);
            fc = tool_sd(tool, mix(qa, qb, c));
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + R * (hi - lo);
            fd = tool_sd(tool, mix(qa, qb, d));
        }
    }
    var best = vec2<f32>(ga, 0.0);
    if (gb < best.x) {
        best = vec2<f32>(gb, 1.0);
    }
    if (fc < best.x) {
        best = vec2<f32>(fc, c);
    }
    if (fd < best.x) {
        best = vec2<f32>(fd, d);
    }
    return best;
}

// One stretch of t, [t0, t0 + dt], its ends seen at qa and qb in the tool frame with
// values ga and gb: (a floor under g on it, the least actual value of g found, the time
// it was found at). The chord is searched only when its cheap floor (the tool's distance
// changes no faster than distance) could come below `goal`.
fn stretch(s: Sweep, p: vec3<f32>, t0: f32, dt: f32, qa: vec3<f32>, qb: vec3<f32>, ga: f32, gb: f32, stray: f32, tol: f32, goal: f32) -> vec3<f32> {
    let span = length(qb - qa);
    var floor = 0.5 * (ga + gb - span) - stray;
    var hi = ga;
    var th = t0;
    if (gb < hi) {
        hi = gb;
        th = t0 + dt;
    }
    if (floor < goal) {
        let tool = s.head.y;
        let steps = u32(clamp(ceil(log(max(4.0 * span / tol, 1.0)) / 0.481212), 0.0, 48.0));
        let m = golden(tool, qa, qb, ga, gb, steps);
        floor = m.x - span * pow(0.618034, f32(steps)) - stray;
        let tm = t0 + m.y * dt;
        let gm = tool_sd(tool, to_local(pose_at(s, tm), p));
        if (gm < hi) {
            hi = gm;
            th = tm;
        }
    }
    return vec3<f32>(floor, hi, th);
}

// A level arc whose tool axis lies along the arc's normal: the tool's distance at a
// point grows with the point's distance from the tool's axis, which is least where the
// arc passes the point's own angle, or at the nearer end when the arc does not. So the
// field is the tool's distance at that least distance from the axis.
// a: (center, sweep), b: (unit normal, radius), c: (the start's unit direction in the
// plane, the start's height along the normal), d.x: +1 or -1 as the tool axis runs.
// A line across the tool's axis (a level move of a 3-axis program, or any move square to
// the axis): the tool's distance grows with the point's distance from the axis, least
// where the axis passes nearest, so the field is the tool's distance at the point's
// distance from the segment the axis runs along (in the plane square to it).
// a: the start, b: the end, c: the unit tool axis.
fn level_line(s: Sweep, p: vec3<f32>) -> f32 {
    let w = p - s.a.xyz;
    let up = dot(w, s.c.xyz);
    let flat = w - s.c.xyz * up;
    let d = s.b.xyz - s.a.xyz;
    let t = clamp(dot(flat, d) / max(dot(d, d), 1e-30), 0.0, 1.0);
    return tool_sd(s.head.y, vec3<f32>(length(flat - d * t), 0.0, up));
}

fn level_arc(s: Sweep, p: vec3<f32>) -> f32 {
    let w = p - s.a.xyz;
    let n = s.b.xyz;
    let up = dot(w, n);
    let flat = w - n * up;
    let r = s.b.w;
    let u0 = s.c.xyz;
    let v0 = cross(n, u0);
    let sweep = s.a.w;
    var gap = atan2(dot(flat, v0), dot(flat, u0)) * sign(sweep);
    if (gap < 0.0) {
        gap += 6.2831853;
    }
    var off = abs(length(flat) - r);
    if (abs(sweep) < 6.2831853 && gap > abs(sweep)) {
        let u1 = u0 * cos(sweep) + v0 * sin(sweep);
        off = min(length(flat - r * u0), length(flat - r * u1));
    }
    return tool_sd(s.head.y, vec3<f32>(off, 0.0, s.d.x * (up - s.c.w)));
}

// The move's field at p: [lo, hi] holding it, except that a field proven at or above
// `cut` may come back as lo >= cut, and one proven at or below -limit as hi <= -limit.
// Stretches are searched coarse to fine, three levels, each splitting only the stretches
// whose floor could still come within `tol` of the best value found; the finest stray is
// a quarter of `tol`.
fn sweep_iv(s: Sweep, p: vec3<f32>, tol: f32, cut: f32, limit: f32) -> vec2<f32> {
    // Outside its box the field is at least the box's distance; a box holding p says
    // nothing of how deep p lies.
    best_t = -1.0;
    let bd = box_dist(s.lo.xyz, s.hi.xyz, p);
    if (bd > 0.0 && bd >= cut) {
        return vec2<f32>(bd, BIG);
    }
    let eps = ROUND * (length(p) + s.e.y);
    if (s.head.x == 3u) {
        let v = level_arc(s, p);
        return vec2<f32>(v - eps, v + eps);
    }
    if (s.head.x == 4u) {
        let v = level_line(s, p);
        return vec2<f32>(v - eps, v + eps);
    }
    let tool = s.head.y;
    let p0 = pose_at(s, 0.0);
    let k = s.e.z + s.e.w * (length(p - p0.at) + s.e.x);
    // Strays at the three levels: coarse (a quarter of a millimeter at most), then
    // stepping down evenly to a quarter of tol.
    let fine = 0.25 * tol;
    let n1 = u32(clamp(ceil(sqrt(k / (8.0 * max(min(0.25 * limit, 0.25), fine)))), 1.0, 256.0));
    let dt1 = 1.0 / f32(n1);
    let stray1 = k * dt1 * dt1 / 8.0;
    let n2 = u32(clamp(ceil(sqrt(sqrt(stray1 / fine))), 1.0, 64.0));
    let dt2 = dt1 / f32(n2);
    let stray2 = k * dt2 * dt2 / 8.0;
    let n3 = u32(clamp(ceil(sqrt(stray2 / fine)), 1.0, 64.0));
    let dt3 = dt2 / f32(n3);
    let stray3 = k * dt3 * dt3 / 8.0;
    var qa = to_local(p0, p);
    var ga = tool_sd(tool, qa);
    var hi = ga;
    var bt = 0.0;
    var lo = BIG;
    for (var i = 0u; i < n1; i++) {
        let t1 = f32(i) * dt1;
        let qb = to_local(pose_at(s, t1 + dt1), p);
        let gb = tool_sd(tool, qb);
        let one = stretch(s, p, t1, dt1, qa, qb, ga, gb, stray1, tol, min(hi, cut) - tol);
        if (one.y < hi) {
            hi = one.y;
            bt = one.z;
        }
        if (one.x >= min(hi, cut) - tol) {
            lo = min(lo, one.x);
        } else {
            var qa2 = qa;
            var ga2 = ga;
            for (var j = 0u; j < n2; j++) {
                let t2 = t1 + f32(j) * dt2;
                var qb2 = qb;
                var gb2 = gb;
                if (j + 1u < n2) {
                    qb2 = to_local(pose_at(s, t2 + dt2), p);
                    gb2 = tool_sd(tool, qb2);
                }
                let two = stretch(s, p, t2, dt2, qa2, qb2, ga2, gb2, stray2, tol, min(hi, cut) - tol);
                if (two.y < hi) {
                    hi = two.y;
                    bt = two.z;
                }
                if (two.x >= min(hi, cut) - tol) {
                    lo = min(lo, two.x);
                } else {
                    var qa3 = qa2;
                    var ga3 = ga2;
                    for (var m = 0u; m < n3; m++) {
                        let t3 = t2 + f32(m) * dt3;
                        var qb3 = qb2;
                        var gb3 = gb2;
                        if (m + 1u < n3) {
                            qb3 = to_local(pose_at(s, t3 + dt3), p);
                            gb3 = tool_sd(tool, qb3);
                        }
                        let three = stretch(s, p, t3, dt3, qa3, qb3, ga3, gb3, stray3, tol, min(hi, cut) - tol);
                        if (three.y < hi) {
                            hi = three.y;
                            bt = three.z;
                        }
                        lo = min(lo, three.x);
                        qa3 = qb3;
                        ga3 = gb3;
                    }
                }
                qa2 = qb2;
                ga2 = gb2;
            }
        }
        if (hi + eps <= -limit) {
            best_t = bt;
            return vec2<f32>(-BIG, hi + eps);
        }
        qa = qb;
        ga = gb;
    }
    best_t = bt;
    return vec2<f32>(lo - eps, hi + eps);
}

// The surface of the move a point came from, near the point: (its field there, the
// field's gradient), the gradient the tool's own at the time the move comes nearest (the
// time's own change does not count where the least is taken), or differences of the
// closed form for a level arc.
fn source(s: Sweep, x: vec3<f32>) -> vec4<f32> {
    let h = 1e-3;
    let ex = vec3<f32>(h, 0.0, 0.0);
    let ey = vec3<f32>(0.0, h, 0.0);
    let ez = vec3<f32>(0.0, 0.0, h);
    if (s.head.x == 3u) {
        let g = vec3<f32>(level_arc(s, x + ex) - level_arc(s, x - ex), level_arc(s, x + ey) - level_arc(s, x - ey), level_arc(s, x + ez) - level_arc(s, x - ez));
        return vec4<f32>(level_arc(s, x), g / (2.0 * h));
    }
    if (s.head.x == 4u) {
        let g = vec3<f32>(level_line(s, x + ex) - level_line(s, x - ex), level_line(s, x + ey) - level_line(s, x - ey), level_line(s, x + ez) - level_line(s, x - ez));
        return vec4<f32>(level_line(s, x), g / (2.0 * h));
    }
    let iv = sweep_iv(s, x, P.tol, P.limit, P.limit);
    let pose = pose_at(s, max(best_t, 0.0));
    let q = to_local(pose, x);
    let tool = s.head.y;
    let g = vec3<f32>(tool_sd(tool, q + ex) - tool_sd(tool, q - ex), tool_sd(tool, q + ey) - tool_sd(tool, q - ey), tool_sd(tool, q + ez) - tool_sd(tool, q - ez));
    return vec4<f32>(tool_sd(tool, q), qrot(pose.r, g / (2.0 * h)));
}

// Settling steps no longer than a chord's sag can be (onto a source surface), or than a
// fraction of a chord (along it, toward the cut): a longer step along the surface means
// the cut runs nearly along it there, where the chord's point is already good.
fn capped(d: vec3<f32>, reach: f32) -> vec3<f32> {
    let l = length(d);
    return d * min(1.0, reach / max(l, 1e-30));
}

// Two Newton steps onto the surface of move `s`.
fn onto_source(s: Sweep, x: vec3<f32>) -> vec3<f32> {
    var y = x;
    for (var k = 0u; k < 2u; k++) {
        let g = source(s, y);
        let l2 = dot(g.yzw, g.yzw);
        if (l2 > 1e-12) {
            y -= capped(g.x * g.yzw / l2, 0.25);
        }
    }
    return y;
}

// The stock box's signed distance from its faces' own offsets, each the difference of
// two numbers near this frame's origin where the face is near, so it rounds by no more
// than the point's and the distance's own size.
fn box_iv(p: vec3<f32>) -> vec2<f32> {
    let d = max(P.box_lo.xyz - p, p - P.box_hi.xyz);
    let b = length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
    let e = ROUND * (length(p) + abs(b));
    return vec2<f32>(b - e, b + e);
}

// The field at p, clamped to [-limit, limit]: the one move listed (mode 0 or 2), or the
// stock, max(box, -min over the moves listed), the moves in no particular order.
fn field_iv(p: vec3<f32>) -> vec2<f32> {
    let L = P.limit;
    if (P.mode % 2u == 0u) {
        return clamp(sweep_iv(sweeps[0], p, P.tol, L, L), vec2<f32>(-L), vec2<f32>(L));
    }
    let b = clamp(box_iv(p), vec2<f32>(-L), vec2<f32>(L));
    if (b.x >= L) {
        return vec2<f32>(L, L);
    }
    // Only a move field below -box can win, and none matters below -limit.
    let cut = -b.x;
    var ulo = BIG;
    var uhi = BIG;
    for (var k = 0u; k < P.count; k++) {
        let c = min(cut, uhi);
        let bd = box_dist(sweeps[k].lo.xyz, sweeps[k].hi.xyz, p);
        if (bd > 0.0 && bd >= c) {
            continue;
        }
        let iv = sweep_iv(sweeps[k], p, P.tol, c, L);
        ulo = min(ulo, iv.x);
        uhi = min(uhi, iv.y);
        if (uhi <= -L) {
            break;
        }
    }
    return vec2<f32>(max(b.x, -clamp(uhi, -L, L)), max(b.y, -clamp(ulo, -L, L)));
}

// The stock's field at p clamped to [-limit, limit], reading the moves idx[first ..
// first + count) only (each point's own list, the moves that can come within the limit).
fn listed_iv(p: vec3<f32>, first: u32, count: u32, with_box: bool) -> vec2<f32> {
    let L = P.limit;
    let b = select(vec2<f32>(-L), clamp(box_iv(p), vec2<f32>(-L), vec2<f32>(L)), with_box);
    if (b.x >= L) {
        return vec2<f32>(L, L);
    }
    let cut = -b.x;
    var ulo = BIG;
    var uhi = BIG;
    for (var k = first; k < first + count; k++) {
        let sw = sweeps[idx[k]];
        let c = min(cut, uhi);
        let bd = box_dist(sw.lo.xyz, sw.hi.xyz, p);
        if (bd > 0.0 && bd >= c) {
            continue;
        }
        let iv = sweep_iv(sw, p, P.tol, c, L);
        ulo = min(ulo, iv.x);
        uhi = min(uhi, iv.y);
        if (uhi <= -L) {
            break;
        }
    }
    return vec2<f32>(max(b.x, -clamp(uhi, -L, L)), max(b.y, -clamp(ulo, -L, L)));
}

// The field the caller cuts by, sgn field + off, as an interval.
fn cut_iv(p: vec3<f32>) -> vec2<f32> {
    let iv = field_iv(p);
    if (P.sgn >= 0.0) {
        return iv + vec2<f32>(P.off);
    }
    return vec2<f32>(P.off - iv.y, P.off - iv.x);
}

fn mid(v: vec2<f32>) -> f32 {
    return 0.5 * (v.x + v.y);
}

// The cut field's middle and its gradient, by central differences.
fn cut_grad(x: vec3<f32>) -> vec4<f32> {
    let h = 1e-3;
    let ex = vec3<f32>(h, 0.0, 0.0);
    let ey = vec3<f32>(0.0, h, 0.0);
    let ez = vec3<f32>(0.0, 0.0, h);
    let g = vec3<f32>(mid(cut_iv(x + ex)) - mid(cut_iv(x - ex)), mid(cut_iv(x + ey)) - mid(cut_iv(x - ey)), mid(cut_iv(x + ez)) - mid(cut_iv(x - ez)));
    return vec4<f32>(mid(cut_iv(x)), g / (2.0 * h));
}

// Modes 4 and 5: item i is (the point, its source move's record), (1 to settle it onto
// the cut's surface as well, 0 not). The point goes onto its source surface; to settle
// it onto the cut's surface too, it steps alternately along the source surface toward
// the cut's zero (Newton on the cut field within the source's tangent plane) and back
// onto the source. Out: the point, the source's gradient there, the cut field's interval.
fn settle(i: u32) {
    let s = sweeps[u32(pts[2u * i].w)];
    let onto = pts[2u * i + 1u].x > 0.5;
    var x = onto_source(s, pts[2u * i].xyz);
    if (onto) {
        for (var r = 0u; r < 3u; r++) {
            let n = source(s, x).yzw;
            let c = cut_grad(x);
            let t = c.yzw - dot(c.yzw, n) / max(dot(n, n), 1e-12) * n;
            let l2 = dot(t, t);
            if (l2 > 1e-12) {
                x -= capped(c.x * t / l2, 0.1);
            }
            x = onto_source(s, x);
        }
    }
    let n = source(s, x).yzw;
    let iv = cut_iv(x);
    out[4u * i] = x.xy;
    out[4u * i + 1u] = vec2<f32>(x.z, n.x);
    out[4u * i + 2u] = n.yz;
    out[4u * i + 3u] = iv;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * P.width;
    if (i >= P.n) {
        return;
    }
    if (P.mode < 2u) {
        out[i] = field_iv(pts[i].xyz);
        return;
    }
    if (P.mode == 6u || P.mode == 8u) {
        let q = pts[2u * i];
        out[i] = listed_iv(q.xyz, u32(q.w), u32(pts[2u * i + 1u].x), P.mode == 6u);
        return;
    }
    if (P.mode == 7u) {
        // Node i % 729 of brick i / 729: 9 x 9 x 9 nodes, x fastest, from the brick's
        // first node (xyz; w: where its list starts) by its cell side (y of the second).
        let b = i / 729u;
        let q = i % 729u;
        let head = pts[2u * b];
        let tail = pts[2u * b + 1u];
        let p = head.xyz + tail.y * vec3<f32>(f32(q % 9u), f32((q / 9u) % 9u), f32(q / 81u));
        out[i] = listed_iv(p, u32(head.w), u32(tail.x), true);
        return;
    }
    if (P.mode >= 4u) {
        settle(i);
        return;
    }
    // A side from a (where the cut field is non-negative) to b (where it is negative):
    // bisection to a hundred-thousandth of a millimeter, by the sign of the cut field's interval
    // and, where the interval holds zero, by the sign of its middle (the rounding bound
    // is a bound; the middle is far nearer the field than the interval's ends).
    let a = pts[2u * i].xyz;
    let b = pts[2u * i + 1u].xyz;
    let size = length(b - a);
    var lo = 0.0;
    var hi = 1.0;
    for (var k = 0u; k < 40u; k++) {
        let m = 0.5 * (lo + hi);
        let iv = cut_iv(mix(a, b, m));
        if (iv.x + iv.y >= 0.0) {
            lo = m;
        } else {
            hi = m;
        }
        if ((hi - lo) * size < 1e-5) {
            break;
        }
    }
    out[i] = vec2<f32>(0.5 * (lo + hi), 0.0);
}
