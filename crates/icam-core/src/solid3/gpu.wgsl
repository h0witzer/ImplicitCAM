// The part's exact signed distance at the nodes of bricks (solid3::Exact on the GPU; see
// gpu.rs), one thread a node: the nearest of the brick's candidate features (listed on the
// CPU from the brick's center), each by the CPU evaluator's rules. Single precision,
// relative to an origin near the part. A node's answer is exact, or it is flagged: 1 when
// all that is known is a lower bound on its distance past the cache's clamp (out[].x; the
// caller takes its sign from a neighbor), 2 when the CPU must read it. A brick lists only
// the features within the clamp of some node of it (every other is farther than the clamp
// from all its nodes), so a node's answer is exact only within the clamp; a point lists
// every feature that can be nearest.

struct Brick {
    // xyz the first node, w the cell side (in point mode: the point, and the distance past
    // which a lower bound will do).
    o: vec4<f32>,
    // x where the brick's candidates start in `cands`, y how many, z where its 729 values
    // (its point's value) go in `out`; w 1 for a point whose list was cut at its clamp.
    c: vec4<u32>,
};

// Whether a record's answer past its clamp is that bound alone: a brick's (its features
// past the clamp are not all listed), and a point's whose list was cut at its clamp (the
// box centers `PartCache::decide` reads), where every feature not listed lies past it.
fn capped(b: Brick) -> bool {
    return P.mode == 0u || b.c.w == 1u;
}

// Where each section starts: in `geo` faces, edges, vertices, sub-patches, patches,
// pieces, piece points, loop copies, loop pieces; in `ints` loop cells and lists.
struct Params {
    nodes: u32,
    faces: u32,
    edges: u32,
    verts: u32,
    subs: u32,
    patches: u32,
    curves: u32,
    cpts: u32,
    inst: u32,
    pieces: u32,
    cells: u32,
    among: u32,
    // Rounding allowance on lengths (a few ulps of the part's extent), and the file's
    // tolerance.
    eps: f32,
    tol: f32,
    // 0: bricks of 729 nodes, a thread a node; 1: points, a thread a point; 2: points whose
    // undecided leaves are refined in place (the nodes a brick left undecided).
    mode: u32,
    // The candidates listed (the scatter's `jobs_of`, a thread a candidate).
    ncand: u32,
    // The refinement round (the points' scatter's `refine_round`).
    round: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> bricks: array<Brick>;
// A candidate: its kind in the top 4 bits (0 analytic face, 1 line or arc edge, 2 vertex
// or pole, 3 B-spline span, 4 piece of a curve), its record below. A span's or piece's
// record roots the tree of its parts: a record's parts are the `count` records from
// `first` (words 14 and 15 of a sub-patch, 19 and 20 of a piece).
@group(0) @binding(2) var<storage, read> cands: array<u32>;
@group(0) @binding(3) var<storage, read> geo: array<f32>;
@group(0) @binding(4) var<storage, read> ints: array<u32>;
// Each node: its signed distance (or bound) and its flag.
@group(0) @binding(5) var<storage, read_write> out: array<vec2<f32>>;
// The scatter's (see its section): each node's best answer and least undecided bound, as
// keys kept by atomic min; the jobs (part record and flags, brick); how many jobs; each
// candidate's brick.
@group(0) @binding(6) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> jobs: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read_write> count: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> owner: array<u32>;
// The points' parts to refine, three rounds' worth (two words each: point and leaf, then
// the part's box in the span's parameters).
@group(0) @binding(10) var<storage, read_write> refine_jobs: array<vec4<u32>>;

// Whether this pipeline refines undecided leaves in place (the settling pass's only): the
// others compile without that code, which costs every thread registers whether it runs
// or not.
override REFINE: bool = false;

const FACE: u32 = 48u;
const EDGE: u32 = 16u;
const VERT: u32 = 8u;
const SUB: u32 = 48u;
const CURVE: u32 = 40u;
const NONE: u32 = 0xffffffffu;
const BIG: f32 = 3.0e38;
const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

fn g(i: u32) -> f32 {
    return geo[i];
}
fn gu(i: u32) -> u32 {
    return bitcast<u32>(geo[i]);
}
fn g3(i: u32) -> vec3<f32> {
    return vec3<f32>(geo[i], geo[i + 1u], geo[i + 2u]);
}
fn g4(i: u32) -> vec4<f32> {
    return vec4<f32>(geo[i], geo[i + 1u], geo[i + 2u], geo[i + 3u]);
}
fn cf(i: u32) -> f32 {
    return bitcast<f32>(ints[i]);
}

fn sgn(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0);
}

fn box_dist(p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    return length(max(max(lo - p, p - hi), vec3<f32>(0.0)));
}

fn far_dist(p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    return length(max(abs(p - lo), abs(p - hi)));
}

// ------------------------------------------------------------------ the loops (trim2)

// A cubic Bernstein polynomial at t; a constant is itself.
fn beval(c: vec4<f32>, t: f32) -> f32 {
    if (c.x == c.y && c.y == c.z && c.z == c.w) {
        return c.x;
    }
    let s = 1.0 - t;
    let a = s * c.x + t * c.y;
    let b = s * c.y + t * c.z;
    let d = s * c.z + t * c.w;
    let e = s * a + t * b;
    let f = s * b + t * d;
    return s * e + t * f;
}

fn bderiv(c: vec4<f32>, t: f32) -> f32 {
    let a = c.y - c.x;
    let b = c.z - c.y;
    let d = c.w - c.z;
    let s = 1.0 - t;
    return 3.0 * (s * s * a + 2.0 * s * t * b + t * t * d);
}

// The parameter where a monotone cubic coordinate equals x, when x lies in its half-open
// range [min, max) (as trim2's `crossing`); -1 when it does not.
fn crossing(a: vec4<f32>, x: f32) -> f32 {
    let p0 = a.x;
    let p1 = a.w;
    if (!(min(p0, p1) <= x && x < max(p0, p1))) {
        return -1.0;
    }
    var lo = 0.0;
    var hi = 1.0;
    var flo = beval(a, 0.0) - x;
    if (flo == 0.0) {
        return 0.0;
    }
    var t = clamp((x - p0) / (p1 - p0), 0.0, 1.0);
    for (var it = 0; it < 48; it = it + 1) {
        let ft = beval(a, t) - x;
        if (ft == 0.0) {
            return t;
        }
        if ((ft < 0.0) == (flo < 0.0)) {
            lo = t;
            flo = ft;
        } else {
            hi = t;
        }
        let d = bderiv(a, t);
        var nt = 0.5 * (lo + hi);
        if (d != 0.0) {
            nt = t - ft / d;
        }
        if (!(nt > lo && nt < hi)) {
            nt = 0.5 * (lo + hi);
        }
        if (abs(nt - t) <= 1e-7) {
            return nt;
        }
        t = nt;
    }
    return t;
}

// Whether the path from a to b (vertical leg, then horizontal), moved up by e and right by
// e^2 (trim2's `parity`), crosses the listed copies an odd number of times.
fn parity(a: vec2<f32>, b: vec2<f32>, start: u32, count: u32) -> bool {
    var n = 0u;
    let up = a.y < b.y;
    let right = a.x < b.x;
    for (var k = 0u; k < count; k = k + 1u) {
        let ii = P.inst + 4u * ints[P.among + start + k];
        let pb = P.pieces + 8u * gu(ii);
        let cu = vec4<f32>(g(pb), g(pb + 2u), g(pb + 4u), g(pb + 6u)) + g(ii + 1u);
        let cv = vec4<f32>(g(pb + 1u), g(pb + 3u), g(pb + 5u), g(pb + 7u)) + g(ii + 2u);
        if (a.y != b.y) {
            let t = crossing(cu, a.x);
            if (t >= 0.0) {
                let y = beval(cv, t);
                if (select(y > b.y && y <= a.y, y > a.y && y <= b.y, up)) {
                    n = n + 1u;
                }
            }
        }
        if (a.x != b.x) {
            let t = crossing(cv, b.y);
            if (t >= 0.0) {
                let x = beval(cu, t);
                var dv = bderiv(cv, t);
                if (dv == 0.0) {
                    dv = cv.w - cv.x;
                }
                let slope_pos = bderiv(cu, t) * dv > 0.0;
                var counted = false;
                if (x == b.x) {
                    counted = (!slope_pos) == right;
                } else if (x == a.x) {
                    counted = slope_pos == right;
                } else if (right) {
                    counted = x > a.x && x < b.x;
                } else {
                    counted = x > b.x && x < a.x;
                }
                if (counted) {
                    n = n + 1u;
                }
            }
        }
    }
    return (n & 1u) == 1u;
}

// Whether the parameters (u, v) lie in face f (record fb), by its loops.
fn inside(fb: u32, uv: vec2<f32>) -> bool {
    if (gu(fb + 28u) == 1u) {
        return true;
    }
    let lo = vec2<f32>(g(fb + 29u), g(fb + 30u));
    let hi = vec2<f32>(g(fb + 31u), g(fb + 32u));
    let per = vec2<f32>(g(fb + 33u), g(fb + 34u));
    var q = uv;
    if (per.x > 0.0) {
        q.x = lo.x + (q.x - lo.x) - per.x * floor((q.x - lo.x) / per.x);
    } else {
        q.x = clamp(q.x, lo.x, hi.x);
    }
    if (per.y > 0.0) {
        q.y = lo.y + (q.y - lo.y) - per.y * floor((q.y - lo.y) / per.y);
    } else {
        q.y = clamp(q.y, lo.y, hi.y);
    }
    var id = gu(fb + 27u);
    for (var depth = 0; depth < 64; depth = depth + 1) {
        let c = P.cells + id * 12u;
        let kind = ints[c + 7u];
        let ins = ints[c + 6u] == 1u;
        if (kind == 0u) {
            return ins;
        }
        if (kind == 1u) {
            let r = vec2<f32>(cf(c + 4u), cf(c + 5u));
            return ins != parity(r, q, ints[c + 8u], ints[c + 9u]);
        }
        let m = 0.5 * (vec2<f32>(cf(c), cf(c + 1u)) + vec2<f32>(cf(c + 2u), cf(c + 3u)));
        let k = select(0u, 1u, q.x >= m.x) + select(0u, 2u, q.y >= m.y);
        id = ints[c + 8u + k];
    }
    return false;
}

// ------------------------------------------------------------------ analytic surfaces

struct Axial {
    theta: f32,
    rho: f32,
    h: f32,
    er: vec3<f32>,
};

fn axial(o: vec3<f32>, x: vec3<f32>, z: vec3<f32>, p: vec3<f32>) -> Axial {
    let y = cross(z, x);
    let d = p - o;
    let lx = dot(d, x);
    let ly = dot(d, y);
    let rho = sqrt(lx * lx + ly * ly);
    var theta = 0.0;
    if (rho > 0.0) {
        theta = atan2(ly, lx);
    }
    return Axial(theta, rho, dot(d, z), x * cos(theta) + y * sin(theta));
}

// The arc parameter of angle a on n equal arcs covering span from a0 (trim2's
// `arc_param_over`).
fn arc_param_over(a: f32, a0: f32, span: f32, n: f32) -> f32 {
    let q = span / n;
    var x = a - a0;
    if (abs(span - TAU) < 1e-5) {
        x = x - TAU * floor(x / TAU);
    } else {
        x = clamp(x - TAU * round((x - 0.5 * span) / TAU), 0.0, span);
    }
    let k = clamp(floor(x / q), 0.0, n - 1.0);
    let alpha = x - q * k - 0.5 * q;
    let t = 0.5 * (tan(0.5 * alpha) / tan(0.25 * q) + 1.0);
    return k + clamp(t, 0.0, 1.0);
}

// The face's foot of p: xyz the point, w 1 when it is a point the surface is singular at
// (a cone's apex, a lemon's or apple's tip).
fn foot(fb: u32, p: vec3<f32>) -> vec4<f32> {
    let kind = gu(fb);
    let o = g3(fb + 2u);
    let x = g3(fb + 5u);
    let z = g3(fb + 8u);
    let r = g(fb + 11u);
    let r2 = g(fb + 12u);
    if (kind == 0u) {
        return vec4<f32>(p - z * dot(p - o, z), 0.0);
    }
    let ax = axial(o, x, z, p);
    if (kind == 1u) {
        return vec4<f32>(o + ax.er * r + z * ax.h, 0.0);
    }
    if (kind == 2u) {
        let sa = sin(r2);
        let ca = cos(r2);
        let ha = -r / tan(r2);
        let s = ax.rho * sa + (ax.h - ha) * ca;
        if (s <= 0.0) {
            return vec4<f32>(o + z * ha, 1.0);
        }
        return vec4<f32>(o + ax.er * (s * sa) + z * (ha + s * ca), 0.0);
    }
    if (kind == 3u) {
        let d = p - o;
        let l = length(d);
        if (l > 0.0) {
            return vec4<f32>(o + d * (r / l), 0.0);
        }
        return vec4<f32>(o + x * r, 0.0);
    }
    // Torus: r major, r2 minor.
    let c = o + ax.er * r;
    let d = p - c;
    let l = length(d);
    var q = c + ax.er * r2;
    if (l > 0.0) {
        q = c + d * (r2 / l);
    }
    if (dot(q - o, ax.er) >= 0.0 || abs(r) > r2) {
        return vec4<f32>(q, 0.0);
    }
    let ht = sqrt(max(r2 * r2 - r * r, 0.0));
    return vec4<f32>(o + z * select(-ht, ht, ax.h >= 0.0), 1.0);
}

// The surface's own unit normal at its point q.
fn normal(fb: u32, q: vec3<f32>) -> vec3<f32> {
    let kind = gu(fb);
    let o = g3(fb + 2u);
    let x = g3(fb + 5u);
    let z = g3(fb + 8u);
    if (kind == 0u) {
        return z;
    }
    if (kind == 3u) {
        return normalize(q - o);
    }
    let ax = axial(o, x, z, q);
    if (kind == 1u) {
        return ax.er;
    }
    if (kind == 2u) {
        let a = g(fb + 12u);
        return ax.er * cos(a) - z * sin(a);
    }
    return normalize(q - (o + ax.er * g(fb + 11u)));
}

// The parameters of q by the face's exact rational form (trim2's `Analytic::uv`).
fn uv(fb: u32, q: vec3<f32>) -> vec2<f32> {
    let kind = gu(fb + 13u);
    let o = g3(fb + 14u);
    let x = g3(fb + 17u);
    let z = g3(fb + 20u);
    if (kind == 0u) {
        let d = q - o;
        return vec2<f32>(dot(d, x), dot(d, cross(z, x)));
    }
    let ax = axial(o, x, z, q);
    let u = arc_param_over(ax.theta, 0.0, TAU, 4.0);
    if (kind == 1u) {
        return vec2<f32>(u, ax.h);
    }
    if (kind == 2u) {
        let r = g(fb + 23u);
        let a = g(fb + 24u);
        let ha = -r / tan(a);
        let s = max(ax.rho * sin(a) + (ax.h - ha) * cos(a), 0.0);
        return vec2<f32>(u, ha + s * cos(a));
    }
    if (kind == 3u) {
        return vec2<f32>(u, arc_param_over(atan2(ax.h, ax.rho), -0.5 * PI, PI, 2.0));
    }
    let major = g(fb + 23u);
    return vec2<f32>(u, arc_param_over(atan2(ax.h, ax.rho - major), g(fb + 24u), g(fb + 25u), g(fb + 26u)));
}

// Analytic face f's contribution at p: (distance, sign), distance BIG for none.
fn face_d(f: u32, p: vec3<f32>, below: f32) -> vec2<f32> {
    let fb = P.faces + f * FACE;
    let ft = foot(fb, p);
    let q = ft.xyz;
    // A foot no nearer than the answer so far changes nothing: no trim test for it (a
    // singular point is never farther than the foot it stands for).
    if (ft.w <= 0.5 && length(p - q) >= below) {
        return vec2<f32>(BIG, 1.0);
    }
    if (ft.w > 0.5) {
        // A singular point: the face's own tip when it lies in the face away from every
        // vertex (the vertex answers otherwise).
        for (var k = 0u; k < gu(fb + 35u); k = k + 1u) {
            let tb = fb + 36u + 6u * k;
            let t = g3(tb);
            if (length(t - q) <= 1e-5 * (1.0 + length(t))) {
                return vec2<f32>(length(p - t), sgn(dot(p - t, g3(tb + 3u))));
            }
        }
        return vec2<f32>(BIG, 1.0);
    }
    if (!inside(fb, uv(fb, q))) {
        return vec2<f32>(BIG, 1.0);
    }
    return vec2<f32>(length(p - q), sgn(dot(p - q, normal(fb, q) * g(fb + 1u))));
}

// ------------------------------------------------------------------ Bezier patches and pieces

// The Bernstein basis of degree n (at most 7) at t: each function with its first and
// second derivatives.
fn basis(n: u32, t: f32) -> array<vec3<f32>, 8> {
    var lv: array<f32, 8>;
    var l1: array<f32, 8>;
    var l2: array<f32, 8>;
    lv[0] = 1.0;
    if (n == 1u) {
        l1 = lv;
    }
    if (n == 2u) {
        l2 = lv;
    }
    let s = 1.0 - t;
    for (var k = 1u; k <= n; k = k + 1u) {
        var nx: array<f32, 8>;
        for (var i = 0u; i <= k; i = i + 1u) {
            var v = 0.0;
            if (i < k) {
                v = s * lv[i];
            }
            if (i > 0u) {
                v = v + t * lv[i - 1u];
            }
            nx[i] = v;
        }
        lv = nx;
        if (k + 1u == n) {
            l1 = lv;
        }
        if (k + 2u == n) {
            l2 = lv;
        }
    }
    var b: array<vec3<f32>, 8>;
    let fnn = f32(n);
    for (var i = 0u; i <= n; i = i + 1u) {
        var d1 = 0.0;
        var d2 = 0.0;
        if (n >= 1u) {
            var a = 0.0;
            var c = 0.0;
            if (i >= 1u) {
                a = l1[i - 1u];
            }
            if (i < n) {
                c = l1[i];
            }
            d1 = fnn * (a - c);
        }
        if (n >= 2u) {
            var a = 0.0;
            var m = 0.0;
            var c = 0.0;
            if (i >= 2u) {
                a = l2[i - 2u];
            }
            if (i >= 1u && i <= n - 1u) {
                m = l2[i - 1u];
            }
            if (i <= n - 2u) {
                c = l2[i];
            }
            d2 = fnn * (fnn - 1.0) * (a - 2.0 * m + c);
        }
        b[i] = vec3<f32>(lv[i], d1, d2);
    }
    return b;
}

struct PE {
    p: vec3<f32>,
    ps: vec3<f32>,
    pt: vec3<f32>,
    pss: vec3<f32>,
    pst: vec3<f32>,
    ptt: vec3<f32>,
};

// A patch (record pb: degrees, parameter ranges, homogeneous points) at local (s, t), with
// first and second derivatives. Bicubic patches (every patch of degree up to 3 is packed
// raised to 3) take the straight-line evaluator: the general one's arrays, indexed as it
// runs, live in slow memory on a GPU.
fn peval(pb: u32, s: f32, t: f32) -> PE {
    if (gu(pb) == 3u && gu(pb + 1u) == 3u) {
        return peval3(pb, s, t);
    }
    return peval_n(pb, s, t);
}

fn peval_n(pb: u32, s: f32, t: f32) -> PE {
    let du = gu(pb);
    let dv = gu(pb + 1u);
    let bu = basis(du, s);
    let bv = basis(dv, t);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    for (var i = 0u; i <= du; i = i + 1u) {
        let x = bu[i];
        for (var j = 0u; j <= dv; j = j + 1u) {
            let y = bv[j];
            let h = g4(pb + 6u + 4u * (i * (dv + 1u) + j));
            a0 = a0 + h * (x.x * y.x);
            a1 = a1 + h * (x.y * y.x);
            a2 = a2 + h * (x.x * y.y);
            a3 = a3 + h * (x.z * y.x);
            a4 = a4 + h * (x.y * y.y);
            a5 = a5 + h * (x.x * y.z);
        }
    }
    return rational(a0, a1, a2, a3, a4, a5);
}

// The cubic Bernstein basis at t: values, first and second derivatives.
struct Cubic {
    b: vec4<f32>,
    d: vec4<f32>,
    e: vec4<f32>,
};

fn cubic(t: f32) -> Cubic {
    let s = 1.0 - t;
    return Cubic(
        vec4<f32>(s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t),
        3.0 * vec4<f32>(-s * s, s * s - 2.0 * s * t, 2.0 * s * t - t * t, t * t),
        6.0 * vec4<f32>(s, t - 2.0 * s, s - 2.0 * t, t),
    );
}

// Row i of a bicubic patch (its points along v) against the v basis: the homogeneous point
// and its first and second v derivatives.
struct Row {
    p: vec4<f32>,
    d: vec4<f32>,
    e: vec4<f32>,
};

fn row3(pb: u32, i: u32, c: Cubic) -> Row {
    let a = pb + 6u + 16u * i;
    let h0 = g4(a);
    let h1 = g4(a + 4u);
    let h2 = g4(a + 8u);
    let h3 = g4(a + 12u);
    return Row(
        h0 * c.b.x + h1 * c.b.y + h2 * c.b.z + h3 * c.b.w,
        h0 * c.d.x + h1 * c.d.y + h2 * c.d.z + h3 * c.d.w,
        h0 * c.e.x + h1 * c.e.y + h2 * c.e.z + h3 * c.e.w,
    );
}

fn peval3(pb: u32, s: f32, t: f32) -> PE {
    let cu = cubic(s);
    let cv = cubic(t);
    let r0 = row3(pb, 0u, cv);
    let r1 = row3(pb, 1u, cv);
    let r2 = row3(pb, 2u, cv);
    let r3 = row3(pb, 3u, cv);
    let a0 = r0.p * cu.b.x + r1.p * cu.b.y + r2.p * cu.b.z + r3.p * cu.b.w;
    let a1 = r0.p * cu.d.x + r1.p * cu.d.y + r2.p * cu.d.z + r3.p * cu.d.w;
    let a2 = r0.d * cu.b.x + r1.d * cu.b.y + r2.d * cu.b.z + r3.d * cu.b.w;
    let a3 = r0.p * cu.e.x + r1.p * cu.e.y + r2.p * cu.e.z + r3.p * cu.e.w;
    let a4 = r0.d * cu.d.x + r1.d * cu.d.y + r2.d * cu.d.z + r3.d * cu.d.w;
    let a5 = r0.e * cu.b.x + r1.e * cu.b.y + r2.e * cu.b.z + r3.e * cu.b.w;
    return rational(a0, a1, a2, a3, a4, a5);
}

// The point and derivatives of a rational patch from its homogeneous sums: a0 the point,
// a1 and a2 the s and t derivatives, a3, a4 and a5 the ss, st and tt ones.
fn rational(a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>, a4: vec4<f32>, a5: vec4<f32>) -> PE {
    let w = a0.w;
    let p = a0.xyz / w;
    let ps = (a1.xyz - p * a1.w) / w;
    let pt = (a2.xyz - p * a2.w) / w;
    let pss = (a3.xyz - ps * (2.0 * a1.w) - p * a3.w) / w;
    let pst = (a4.xyz - ps * a2.w - pt * a1.w - p * a4.w) / w;
    let ptt = (a5.xyz - pt * (2.0 * a2.w) - p * a5.w) / w;
    return PE(p, ps, pt, pss, pst, ptt);
}

struct CE {
    p: vec3<f32>,
    d1: vec3<f32>,
    d2: vec3<f32>,
};

// A rational Bezier piece of degree n (points from cb) at s; a cubic (every piece of degree
// up to 3 is packed raised to 3) in straight-line code, as `peval`.
fn ceval(cb: u32, n: u32, s: f32) -> CE {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    if (n == 3u) {
        let c = cubic(s);
        let h0 = g4(cb);
        let h1 = g4(cb + 4u);
        let h2 = g4(cb + 8u);
        let h3 = g4(cb + 12u);
        a0 = h0 * c.b.x + h1 * c.b.y + h2 * c.b.z + h3 * c.b.w;
        a1 = h0 * c.d.x + h1 * c.d.y + h2 * c.d.z + h3 * c.d.w;
        a2 = h0 * c.e.x + h1 * c.e.y + h2 * c.e.z + h3 * c.e.w;
    } else {
        let b = basis(n, s);
        for (var i = 0u; i <= n; i = i + 1u) {
            let h = g4(cb + 4u * i);
            a0 = a0 + h * b[i].x;
            a1 = a1 + h * b[i].y;
            a2 = a2 + h * b[i].z;
        }
    }
    let w = a0.w;
    let p = a0.xyz / w;
    let d1 = (a1.xyz - p * a1.w) / w;
    let d2 = (a2.xyz - d1 * (2.0 * a1.w) - p * a2.w) / w;
    return CE(p, d1, d2);
}

// What a sub-patch or piece holds for p: its least distance to p at q, with state 1 when
// q is a critical point of the distance (nearly: the miss is below the rounding of a
// point this far), 0 when there is none in it (the least distance is held at its border
// with the descent pointing out), 2 when Newton did not settle.
struct Min {
    state: u32,
    d: f32,
    q: vec3<f32>,
    n: vec3<f32>,
    st: vec2<f32>,
};

// Newton on |S - p|^2 over the sub-patch (record sb) of its span's patch, from its middle,
// with the parameters at a bound held while the descent points out (the active set
// chosen afresh each step), halving a step that does not descend.
fn sub_min(sb: u32, p: vec3<f32>) -> Min {
    return box_min(gu(sb + 7u), vec2<f32>(g(sb + 8u), g(sb + 10u)), vec2<f32>(g(sb + 9u), g(sb + 11u)), p);
}

// The same over the box [lo, hi] of the span-local parameters of the patch at pb.
fn box_min(pb: u32, lo: vec2<f32>, hi: vec2<f32>, p: vec3<f32>) -> Min {
    return box_min_from(pb, lo, hi, p, 0.5 * (lo + hi));
}

// The same from the parameters x0.
fn box_min_from(pb: u32, lo: vec2<f32>, hi: vec2<f32>, p: vec3<f32>, x0: vec2<f32>) -> Min {
    var x = x0;
    var e = peval(pb, x.x, x.y);
    for (var it = 0; it < 32; it = it + 1) {
        let r = e.p - p;
        let gr = vec2<f32>(dot(e.ps, r), dot(e.pt, r));
        let jss = dot(e.ps, e.ps);
        let jst = dot(e.ps, e.pt);
        let jtt = dot(e.pt, e.pt);
        var h = vec3<f32>(jss + dot(e.pss, r), jst + dot(e.pst, r), jtt + dot(e.ptt, r));
        if (!(h.x > 0.0 && h.x * h.z - h.y * h.y > 0.0)) {
            h = vec3<f32>(jss, jst, jtt);
        }
        let fs = !((x.x <= lo.x && gr.x > 0.0) || (x.x >= hi.x && gr.x < 0.0));
        let ft = !((x.y <= lo.y && gr.y > 0.0) || (x.y >= hi.y && gr.y < 0.0));
        var d = vec2<f32>(0.0);
        if (fs && ft) {
            let det = h.x * h.z - h.y * h.y;
            if (det > 0.0) {
                d = vec2<f32>(-(h.z * gr.x - h.y * gr.y), -(h.x * gr.y - h.y * gr.x)) / det;
            }
        } else if (fs) {
            d.x = -gr.x / max(h.x, 1e-30);
        } else if (ft) {
            d.y = -gr.y / max(h.z, 1e-30);
        }
        let f0 = dot(r, r);
        var nx = clamp(x + d, lo, hi);
        var en = peval(pb, nx.x, nx.y);
        for (var k = 0; k < 8; k = k + 1) {
            let rn = en.p - p;
            // Not higher, to the rounding of a value this size (points far from the origin carry
            // about 1e-5 of relative noise in single precision).
            if (dot(rn, rn) <= f0 * (1.0 + 1e-4) + 1e-10) {
                break;
            }
            d = 0.5 * d;
            nx = clamp(x + d, lo, hi);
            en = peval(pb, nx.x, nx.y);
        }
        let moved = abs(nx.x - x.x) + abs(nx.y - x.y);
        // Settled: in the parameters, or in space to the rounding of the part's extent (a
        // point on the surface, at no distance, steps about in rounding noise for good).
        let step = length(en.p - e.p);
        x = nx;
        e = en;
        if (moved <= 2e-7 || step <= 0.125 * P.eps) {
            break;
        }
    }
    let r = e.p - p;
    let dist = length(r);
    let gr = vec2<f32>(dot(e.ps, r), dot(e.pt, r));
    let tol = 1e-4 * dist + P.eps;
    let cs = abs(gr.x) <= tol * length(e.ps);
    let ct = abs(gr.y) <= tol * length(e.pt);
    let hs = cs || (x.x <= lo.x && gr.x > 0.0) || (x.x >= hi.x && gr.x < 0.0);
    let ht = ct || (x.y <= lo.y && gr.y > 0.0) || (x.y >= hi.y && gr.y < 0.0);
    // On a part where the function is convex (the only parts read this way), a point that
    // meets the first-order conditions is its minimum, settled or not.
    var state = 2u;
    if (cs && ct) {
        state = 1u;
    } else if (hs && ht) {
        state = 0u;
    }
    return Min(state, dist, e.p, normalize(cross(e.ps, e.pt)), x);
}

// The same over a piece (record cb) in one parameter.
fn curve_min(cb: u32, p: vec3<f32>) -> Min {
    return curve_min_on(cb, p, g(cb + 9u), g(cb + 10u));
}

// The same over the part [lo, hi] of the piece's parameter.
fn curve_min_on(cb: u32, p: vec3<f32>, lo: f32, hi: f32) -> Min {
    let n = gu(cb + 7u);
    let pts = gu(cb + 8u);
    var x = 0.5 * (lo + hi);
    var e = ceval(pts, n, x);
    for (var it = 0; it < 32; it = it + 1) {
        let r = e.p - p;
        let gr = dot(e.d1, r);
        var h = dot(e.d1, e.d1) + dot(e.d2, r);
        if (!(h > 0.0)) {
            h = dot(e.d1, e.d1);
        }
        let free = !((x <= lo && gr > 0.0) || (x >= hi && gr < 0.0));
        var d = 0.0;
        if (free) {
            d = -gr / max(h, 1e-30);
        }
        let f0 = dot(r, r);
        var nx = clamp(x + d, lo, hi);
        var en = ceval(pts, n, nx);
        for (var k = 0; k < 8; k = k + 1) {
            let rn = en.p - p;
            // Not higher, to the rounding of a value this size (points far from the origin carry
            // about 1e-5 of relative noise in single precision).
            if (dot(rn, rn) <= f0 * (1.0 + 1e-4) + 1e-10) {
                break;
            }
            d = 0.5 * d;
            nx = clamp(x + d, lo, hi);
            en = ceval(pts, n, nx);
        }
        let moved = abs(nx - x);
        let step = length(en.p - e.p);
        x = nx;
        e = en;
        if (moved <= 2e-7 || step <= 0.125 * P.eps) {
            break;
        }
    }
    let r = e.p - p;
    let dist = length(r);
    let gr = dot(e.d1, r);
    let c = abs(gr) <= (1e-4 * dist + P.eps) * length(e.d1);
    let held = c || (x <= lo && gr > 0.0) || (x >= hi && gr < 0.0);
    var state = 2u;
    if (c) {
        state = 1u;
    } else if (held) {
        state = 0u;
    }
    return Min(state, dist, e.p, vec3<f32>(0.0), vec2<f32>(x, 0.0));
}

// ------------------------------------------------------------------ edges and their signs

// A line or arc edge's interior foot of p: xyz the point, w the distance (BIG for none).
fn edge_foot(e: u32, p: vec3<f32>) -> vec4<f32> {
    let eb = P.edges + e * EDGE;
    let kind = gu(eb);
    if (kind == 0u) {
        let a = g3(eb + 1u);
        let b = g3(eb + 4u);
        let ab = b - a;
        let l2 = dot(ab, ab);
        if (!(l2 > 0.0)) {
            return vec4<f32>(0.0, 0.0, 0.0, BIG);
        }
        let t = dot(p - a, ab) / l2;
        if (!(t > 0.0 && t < 1.0)) {
            return vec4<f32>(0.0, 0.0, 0.0, BIG);
        }
        let q = a + ab * t;
        return vec4<f32>(q, length(p - q));
    }
    let o = g3(eb + 1u);
    let x = g3(eb + 4u);
    let z = g3(eb + 7u);
    let y = cross(z, x);
    let r = g(eb + 10u);
    let t0 = g(eb + 11u);
    let sweep = g(eb + 12u);
    let d = p - o;
    let lx = dot(d, x);
    let ly = dot(d, y);
    if (lx == 0.0 && ly == 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, BIG);
    }
    let t = atan2(ly, lx);
    let full = abs(abs(sweep) - TAU) < 1e-6;
    var dt = select(t0 - t, t - t0, sweep >= 0.0);
    dt = dt - TAU * floor(dt / TAU);
    if (!(full || dt <= abs(sweep))) {
        return vec4<f32>(0.0, 0.0, 0.0, BIG);
    }
    let q = o + x * (r * cos(t)) + y * (r * sin(t));
    return vec4<f32>(q, length(p - q));
}

// Face f's outward normal at a point q on its surface: an analytic face's at q itself (as
// `Face3::normal`); a B-spline face's at q's nearest point on it, among the brick's
// candidate spans of the face, walked down to parts convex for q or to leaves (none
// listed: w 0).
fn face_normal(f: u32, q: vec3<f32>, b: Brick) -> vec4<f32> {
    let fb = P.faces + f * FACE;
    let kind = gu(fb);
    if (kind <= 4u) {
        return vec4<f32>(normal(fb, q) * g(fb + 1u), 1.0);
    }
    var best = BIG;
    var n = vec3<f32>(0.0);
    for (var k = 0u; k < b.c.y; k = k + 1u) {
        let c = cands[b.c.x + k];
        if ((c >> 28u) != 3u || gu(P.subs + (c & 0x0fffffffu) * SUB + 12u) != f) {
            continue;
        }
        span_nearest(c & 0x0fffffffu, q, &best, &n);
    }
    if (!(best <= 4.0 * P.tol + 16.0 * P.eps)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(n * g(fb + 1u), 1.0);
}

// The same at a point q of an edge's piece (record cb), for the face whose seed the piece
// holds from word w (the span's sub record, s and t: the face's nearest point to the
// piece's middle): Newton on the span from there, carried on into the next span wherever it
// is held at a span's side with its descent leading out (a piece crosses the face's knot
// lines; the face record's words 2 to 5 say where each span's patch is), else the search
// over the brick's spans.
fn edge_face_normal(f: u32, q: vec3<f32>, b: Brick, cb: u32, w: u32) -> vec4<f32> {
    let fb = P.faces + f * FACE;
    if (gu(fb) <= 4u) {
        return vec4<f32>(normal(fb, q) * g(fb + 1u), 1.0);
    }
    let sid = gu(cb + w);
    if (sid != NONE) {
        let nu = i32(gu(fb + 2u));
        let nv = i32(gu(fb + 3u));
        let base = P.patches + gu(fb + 4u);
        let size = gu(fb + 5u);
        var pb = gu(P.subs + sid * SUB + 7u);
        var x = vec2<f32>(g(cb + w + 1u), g(cb + w + 2u));
        for (var hop = 0u; hop < 16u; hop = hop + 1u) {
            let m = box_min_from(pb, vec2<f32>(0.0), vec2<f32>(1.0), q, x);
            if (m.d <= 4.0 * P.tol + 16.0 * P.eps) {
                return vec4<f32>(m.n * g(fb + 1u), 1.0);
            }
            if (size == 0u || pb < base) {
                break;
            }
            let k = i32((pb - base) / size);
            var a = k / nv;
            var c = k % nv;
            let e = peval(pb, m.st.x, m.st.y);
            let r = e.p - q;
            let gs = dot(e.ps, r);
            let gt = dot(e.pt, r);
            var nx = m.st;
            if (m.st.x <= 0.0 && gs > 0.0) {
                a = a - 1;
                nx.x = 1.0;
            } else if (m.st.x >= 1.0 && gs < 0.0) {
                a = a + 1;
                nx.x = 0.0;
            } else if (m.st.y <= 0.0 && gt > 0.0) {
                c = c - 1;
                nx.y = 1.0;
            } else if (m.st.y >= 1.0 && gt < 0.0) {
                c = c + 1;
                nx.y = 0.0;
            } else {
                break;
            }
            if (a < 0 || c < 0 || a >= nu || c >= nv) {
                break;
            }
            pb = base + u32(a * nv + c) * size;
            x = nx;
        }
    }
    return face_normal(f, q, b);
}

// A span's tree (root record) walked for q's nearest point on it, nearer than `best`: the
// distance and the surface's normal there, when nearer.
fn span_nearest(root: u32, q: vec3<f32>, best: ptr<function, f32>, n: ptr<function, vec3<f32>>) {
    var stack: array<u32, 48>;
    var sp = 1u;
    stack[0] = root;
    while (sp > 0u) {
        sp = sp - 1u;
        let sb = P.subs + stack[sp] * SUB;
        let lo = g3(sb);
        let hi = g3(sb + 3u);
        if (box_dist(q, lo, hi) >= *best) {
            continue;
        }
        let nk = gu(sb + 15u);
        if (nk == 0u || sp + nk > 48u || far_dist(q, lo, hi) < g(sb + 6u) * 0.999 - P.eps) {
            let m = sub_min(sb, q);
            if (m.d < *best) {
                *best = m.d;
                *n = m.n;
            }
            continue;
        }
        let first = gu(sb + 14u);
        for (var kk = 0u; kk < nk; kk = kk + 1u) {
            stack[sp] = first + kk;
            sp = sp + 1u;
        }
    }
}

// A knot line's normal at line parameter w: the two sides' normals summed (the patch
// before the line at its far side, the patch after at its near side).
fn side_normal(pb: u32, k: u32, side: f32, w: f32) -> vec3<f32> {
    var s = side;
    var t = side;
    if (k == 0u) {
        t = (w - g(pb + 4u)) / (g(pb + 5u) - g(pb + 4u));
    } else {
        s = (w - g(pb + 2u)) / (g(pb + 3u) - g(pb + 2u));
    }
    let e = peval(pb, s, t);
    return normalize(cross(e.ps, e.pt));
}

// ------------------------------------------------------------------ refining a leaf in place

// A node's leaves left undecided by the walks (record, lower bound as bits), refined after
// all its candidates are read, nearest first, while still nearer than its best answer; and
// the parts it may still make.
var<private> deferred: array<vec2<u32>, 16>;
var<private> n_deferred: u32;
var<private> budget: u32;

// Control point i of the part of a Bezier curve (degree n, points c) over [a, b]: its
// blossom at a (n - i times) and b (i times).
fn seg_point(c: ptr<function, array<vec4<f32>, 8>>, n: u32, a: f32, b: f32, i: u32) -> vec4<f32> {
    var t = *c;
    for (var lev = 1u; lev <= n; lev = lev + 1u) {
        let w = select(b, a, lev + i <= n);
        for (var k = 0u; k + lev <= n; k = k + 1u) {
            t[k] = mix(t[k], t[k + 1u], w);
        }
    }
    return t[0];
}

// The control net (homogeneous, i along s at stride 8) of the patch at pb over the
// span-local box bx = (s0, s1, t0, t1).
fn sub_net(pb: u32, bx: vec4<f32>) -> array<vec4<f32>, 64> {
    let du = gu(pb);
    let dv = gu(pb + 1u);
    var q: array<vec4<f32>, 64>;
    for (var j = 0u; j <= dv; j = j + 1u) {
        var c: array<vec4<f32>, 8>;
        for (var i = 0u; i <= du; i = i + 1u) {
            c[i] = g4(pb + 6u + 4u * (i * (dv + 1u) + j));
        }
        for (var i = 0u; i <= du; i = i + 1u) {
            q[i * 8u + j] = seg_point(&c, du, bx.x, bx.y, i);
        }
    }
    for (var i = 0u; i <= du; i = i + 1u) {
        var c: array<vec4<f32>, 8>;
        for (var j = 0u; j <= dv; j = j + 1u) {
            c[j] = q[i * 8u + j];
        }
        for (var j = 0u; j <= dv; j = j + 1u) {
            q[i * 8u + j] = seg_point(&c, dv, bx.z, bx.w, j);
        }
    }
    return q;
}

// A part made on the fly: its box, and its derivatives' bounds in its own parameters scaled
// to unit tangents at its middle, with lambda_min(G)'s lower bound (as a sub-patch record's).
struct Part {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    su: array<vec3<f32>, 2>,
    sv: array<vec3<f32>, 2>,
    suu: array<vec3<f32>, 2>,
    suv: array<vec3<f32>, 2>,
    svv: array<vec3<f32>, 2>,
    lg: f32,
};

// Widened by single-precision rounding.
fn widen(a: array<vec3<f32>, 2>) -> array<vec3<f32>, 2> {
    let m = 1e-5 * max(abs(a[0]), abs(a[1])) + vec3<f32>(1e-30);
    return array<vec3<f32>, 2>(a[0] - m, a[1] + m);
}

fn hull_add(h: ptr<function, array<vec3<f32>, 2>>, x: vec3<f32>) {
    (*h)[0] = min((*h)[0], x);
    (*h)[1] = max((*h)[1], x);
}

// The part of the patch at pb over bx: bounds from its control net (a polynomial patch;
// one whose weights differ is left: ok false). A bicubic's in straight-line code (the
// general one's arrays, indexed as it runs, live in slow memory on a GPU).
fn make_part(pb: u32, bx: vec4<f32>) -> Part {
    if (gu(pb) == 3u && gu(pb + 1u) == 3u) {
        return make_part3(pb, bx);
    }
    return make_part_n(pb, bx);
}

// A cubic's blossom at (x, y, z).
fn blossom3(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, x: f32, y: f32, z: f32) -> vec4<f32> {
    let q0 = mix(a, b, x);
    let q1 = mix(b, c, x);
    let q2 = mix(c, d, x);
    return mix(mix(q0, q1, y), mix(q1, q2, y), z);
}

// A cubic's four points, and the points of its part over [u0, u1].
struct Quad {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
};

fn seg3(q: Quad, u0: f32, u1: f32) -> Quad {
    return Quad(
        blossom3(q.a, q.b, q.c, q.d, u0, u0, u0),
        blossom3(q.a, q.b, q.c, q.d, u0, u0, u1),
        blossom3(q.a, q.b, q.c, q.d, u0, u1, u1),
        blossom3(q.a, q.b, q.c, q.d, u1, u1, u1),
    );
}

// A bicubic's derivative nets (hodographs), packed after its points (gpu.rs `hodographs`):
// Su 3 x 4 from point 0, Sv 4 x 3 from 12, Suu 2 x 4 from 24, Suv 3 x 3 from 32, Svv 4 x 2
// from 41 (i along s, three words a point), then each net's largest coordinate (word 147).
const HODO: u32 = 70u;

fn hd(pb: u32, k: u32) -> vec3<f32> {
    return g3(pb + HODO + 3u * k);
}

struct L2 {
    a: vec3<f32>,
    b: vec3<f32>,
};

struct L3 {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
};

struct L4 {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    d: vec3<f32>,
};

// A line's, a quadratic's and a cubic's points over their part [u0, u1] (by blossoms).
fn cut1(a: vec3<f32>, b: vec3<f32>, u0: f32, u1: f32) -> L2 {
    return L2(mix(a, b, u0), mix(a, b, u1));
}

fn bl2(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, x: f32, y: f32) -> vec3<f32> {
    return mix(mix(a, b, x), mix(b, c, x), y);
}

fn cut2(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, u0: f32, u1: f32) -> L3 {
    return L3(bl2(a, b, c, u0, u0), bl2(a, b, c, u0, u1), bl2(a, b, c, u1, u1));
}

fn bl3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, x: f32, y: f32, z: f32) -> vec3<f32> {
    let q0 = mix(a, b, x);
    let q1 = mix(b, c, x);
    let q2 = mix(c, d, x);
    return mix(mix(q0, q1, y), mix(q1, q2, y), z);
}

fn cut3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, u0: f32, u1: f32) -> L4 {
    return L4(bl3(a, b, c, d, u0, u0, u0), bl3(a, b, c, d, u0, u0, u1), bl3(a, b, c, d, u0, u1, u1), bl3(a, b, c, d, u1, u1, u1));
}

fn hull2(h: ptr<function, array<vec3<f32>, 2>>, q: L2) {
    hull_add(h, q.a);
    hull_add(h, q.b);
}

fn hull3(h: ptr<function, array<vec3<f32>, 2>>, q: L3) {
    hull_add(h, q.a);
    hull_add(h, q.b);
    hull_add(h, q.c);
}

fn hull4(h: ptr<function, array<vec3<f32>, 2>>, q: L4) {
    hull_add(h, q.a);
    hull_add(h, q.b);
    hull_add(h, q.c);
    hull_add(h, q.d);
}

// A net's hull padded by the rounding of cutting it (a few units in the last place of the
// net's largest coordinate, word `big` after the nets), times k (the part's parameter
// scale, positive).
fn hodo_done(h: array<vec3<f32>, 2>, pb: u32, big: u32, k: f32) -> array<vec3<f32>, 2> {
    let pad = vec3<f32>(4e-6 * g(pb + HODO + 147u + big));
    return array<vec3<f32>, 2>((h[0] - pad) * k, (h[1] + pad) * k);
}

fn hodo_su(pb: u32, bx: vec4<f32>, k: f32) -> array<vec3<f32>, 2> {
    let c0 = cut2(hd(pb, 0u), hd(pb, 4u), hd(pb, 8u), bx.x, bx.y);
    let c1 = cut2(hd(pb, 1u), hd(pb, 5u), hd(pb, 9u), bx.x, bx.y);
    let c2 = cut2(hd(pb, 2u), hd(pb, 6u), hd(pb, 10u), bx.x, bx.y);
    let c3 = cut2(hd(pb, 3u), hd(pb, 7u), hd(pb, 11u), bx.x, bx.y);
    var h = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull4(&h, cut3(c0.a, c1.a, c2.a, c3.a, bx.z, bx.w));
    hull4(&h, cut3(c0.b, c1.b, c2.b, c3.b, bx.z, bx.w));
    hull4(&h, cut3(c0.c, c1.c, c2.c, c3.c, bx.z, bx.w));
    return hodo_done(h, pb, 0u, k);
}

fn hodo_sv(pb: u32, bx: vec4<f32>, k: f32) -> array<vec3<f32>, 2> {
    let c0 = cut3(hd(pb, 12u), hd(pb, 15u), hd(pb, 18u), hd(pb, 21u), bx.x, bx.y);
    let c1 = cut3(hd(pb, 13u), hd(pb, 16u), hd(pb, 19u), hd(pb, 22u), bx.x, bx.y);
    let c2 = cut3(hd(pb, 14u), hd(pb, 17u), hd(pb, 20u), hd(pb, 23u), bx.x, bx.y);
    var h = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull3(&h, cut2(c0.a, c1.a, c2.a, bx.z, bx.w));
    hull3(&h, cut2(c0.b, c1.b, c2.b, bx.z, bx.w));
    hull3(&h, cut2(c0.c, c1.c, c2.c, bx.z, bx.w));
    hull3(&h, cut2(c0.d, c1.d, c2.d, bx.z, bx.w));
    return hodo_done(h, pb, 1u, k);
}

fn hodo_suu(pb: u32, bx: vec4<f32>, k: f32) -> array<vec3<f32>, 2> {
    let c0 = cut1(hd(pb, 24u), hd(pb, 28u), bx.x, bx.y);
    let c1 = cut1(hd(pb, 25u), hd(pb, 29u), bx.x, bx.y);
    let c2 = cut1(hd(pb, 26u), hd(pb, 30u), bx.x, bx.y);
    let c3 = cut1(hd(pb, 27u), hd(pb, 31u), bx.x, bx.y);
    var h = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull4(&h, cut3(c0.a, c1.a, c2.a, c3.a, bx.z, bx.w));
    hull4(&h, cut3(c0.b, c1.b, c2.b, c3.b, bx.z, bx.w));
    return hodo_done(h, pb, 2u, k);
}

fn hodo_suv(pb: u32, bx: vec4<f32>, k: f32) -> array<vec3<f32>, 2> {
    let c0 = cut2(hd(pb, 32u), hd(pb, 35u), hd(pb, 38u), bx.x, bx.y);
    let c1 = cut2(hd(pb, 33u), hd(pb, 36u), hd(pb, 39u), bx.x, bx.y);
    let c2 = cut2(hd(pb, 34u), hd(pb, 37u), hd(pb, 40u), bx.x, bx.y);
    var h = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull3(&h, cut2(c0.a, c1.a, c2.a, bx.z, bx.w));
    hull3(&h, cut2(c0.b, c1.b, c2.b, bx.z, bx.w));
    hull3(&h, cut2(c0.c, c1.c, c2.c, bx.z, bx.w));
    return hodo_done(h, pb, 3u, k);
}

fn hodo_svv(pb: u32, bx: vec4<f32>, k: f32) -> array<vec3<f32>, 2> {
    let c0 = cut3(hd(pb, 41u), hd(pb, 43u), hd(pb, 45u), hd(pb, 47u), bx.x, bx.y);
    let c1 = cut3(hd(pb, 42u), hd(pb, 44u), hd(pb, 46u), hd(pb, 48u), bx.x, bx.y);
    var h = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull2(&h, cut1(c0.a, c1.a, bx.z, bx.w));
    hull2(&h, cut1(c0.b, c1.b, bx.z, bx.w));
    hull2(&h, cut1(c0.c, c1.c, bx.z, bx.w));
    hull2(&h, cut1(c0.d, c1.d, bx.z, bx.w));
    return hodo_done(h, pb, 4u, k);
}

fn make_part3(pb: u32, bx: vec4<f32>) -> Part {
    var r: Part;
    r.ok = false;
    // The net over bx (i along s at stride 4): each column cut in s, then each row in t.
    var net: array<vec4<f32>, 16>;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let c = seg3(Quad(g4(pb + 6u + 4u * j), g4(pb + 22u + 4u * j), g4(pb + 38u + 4u * j), g4(pb + 54u + 4u * j)), bx.x, bx.y);
        net[j] = c.a;
        net[4u + j] = c.b;
        net[8u + j] = c.c;
        net[12u + j] = c.d;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let c = seg3(Quad(net[4u * i], net[4u * i + 1u], net[4u * i + 2u], net[4u * i + 3u]), bx.z, bx.w);
        net[4u * i] = c.a;
        net[4u * i + 1u] = c.b;
        net[4u * i + 2u] = c.c;
        net[4u * i + 3u] = c.d;
    }
    let w0 = net[0].w;
    var lo = vec3<f32>(BIG);
    var hi = vec3<f32>(-BIG);
    for (var k = 0u; k < 16u; k = k + 1u) {
        if (abs(net[k].w - w0) > 1e-6 * abs(w0)) {
            return r;
        }
        let x = net[k].xyz / net[k].w;
        lo = min(lo, x);
        hi = max(hi, x);
    }
    // The derivatives' bounds: the hodographs cut to the part, each padded by its rounding
    // (relative to the span's net it was cut from) and scaled to the part's parameters. A
    // rational patch has none (its largest coordinates are written negative).
    if (g(pb + HODO + 147u) < 0.0) {
        return r;
    }
    let ds = bx.y - bx.x;
    let dt = bx.w - bx.z;
    let su = hodo_su(pb, bx, ds);
    let sv = hodo_sv(pb, bx, dt);
    let suu = hodo_suu(pb, bx, ds * ds);
    let suv = hodo_suv(pb, bx, ds * dt);
    let svv = hodo_svv(pb, bx, dt * dt);
    return part_bounds(pb, bx, lo, hi, su, sv, suu, suv, svv);
}

fn make_part_n(pb: u32, bx: vec4<f32>) -> Part {
    let du = gu(pb);
    let dv = gu(pb + 1u);
    let q = sub_net(pb, bx);
    var r: Part;
    r.ok = false;
    let w0 = q[0].w;
    var pts: array<vec3<f32>, 64>;
    var lo = vec3<f32>(BIG);
    var hi = vec3<f32>(-BIG);
    for (var i = 0u; i <= du; i = i + 1u) {
        for (var j = 0u; j <= dv; j = j + 1u) {
            let h = q[i * 8u + j];
            if (abs(h.w - w0) > 1e-6 * abs(w0)) {
                return r;
            }
            let x = h.xyz / h.w;
            pts[i * 8u + j] = x;
            lo = min(lo, x);
            hi = max(hi, x);
        }
    }
    let fu = f32(du);
    let fv = f32(dv);
    var su = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    var sv = su;
    var suu = su;
    var suv = su;
    var svv = su;
    let zero = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));
    for (var i = 0u; i <= du; i = i + 1u) {
        for (var j = 0u; j <= dv; j = j + 1u) {
            let x = pts[i * 8u + j];
            if (i < du) {
                hull_add(&su, fu * (pts[(i + 1u) * 8u + j] - x));
            }
            if (j < dv) {
                hull_add(&sv, fv * (pts[i * 8u + j + 1u] - x));
            }
            if (i + 1u < du) {
                hull_add(&suu, fu * (fu - 1.0) * (pts[(i + 2u) * 8u + j] - 2.0 * pts[(i + 1u) * 8u + j] + x));
            }
            if (i < du && j < dv) {
                hull_add(&suv, fu * fv * (pts[(i + 1u) * 8u + j + 1u] - pts[(i + 1u) * 8u + j] - pts[i * 8u + j + 1u] + x));
            }
            if (j + 1u < dv) {
                hull_add(&svv, fv * (fv - 1.0) * (pts[i * 8u + j + 2u] - 2.0 * pts[i * 8u + j + 1u] + x));
            }
        }
    }
    if (du == 0u) {
        su = zero;
    }
    if (dv == 0u) {
        sv = zero;
    }
    if (du < 2u) {
        suu = zero;
    }
    if (du == 0u || dv == 0u) {
        suv = zero;
    }
    if (dv < 2u) {
        svv = zero;
    }
    return part_bounds(pb, bx, lo, hi, su, sv, suu, suv, svv);
}

// A part from its points' box and its derivatives' hulls in its own parameters: the box
// padded for rounding, the bounds scaled to unit tangents at its middle, and lambda_min(G)'s
// lower bound (ok false when a tangent there vanishes).
fn part_bounds(pb: u32, bx: vec4<f32>, lo: vec3<f32>, hi: vec3<f32>, su: array<vec3<f32>, 2>, sv: array<vec3<f32>, 2>, suu: array<vec3<f32>, 2>, suv: array<vec3<f32>, 2>, svv: array<vec3<f32>, 2>) -> Part {
    var r: Part;
    r.ok = false;
    let pad = vec3<f32>(P.eps) + 1e-6 * max(abs(lo), abs(hi));
    r.lo = lo - pad;
    r.hi = hi + pad;
    // Scaled to unit tangents at the middle.
    let e = peval(pb, 0.5 * (bx.x + bx.y), 0.5 * (bx.z + bx.w));
    let pu = e.ps * (bx.y - bx.x);
    let pv = e.pt * (bx.w - bx.z);
    let a = length(pu);
    let b = length(pv);
    if (!(a > 0.0 && b > 0.0)) {
        return r;
    }
    r.su = widen(array<vec3<f32>, 2>(su[0] / a, su[1] / a));
    r.sv = widen(array<vec3<f32>, 2>(sv[0] / b, sv[1] / b));
    r.suu = widen(array<vec3<f32>, 2>(suu[0] / (a * a), suu[1] / (a * a)));
    r.suv = widen(array<vec3<f32>, 2>(suv[0] / (a * b), suv[1] / (a * b)));
    r.svv = widen(array<vec3<f32>, 2>(svv[0] / (b * b), svv[1] / (b * b)));
    // lambda_min(G) by Weyl's inequality from the middle: sigma_min(J0) less how far the
    // tangents stray from J0's columns.
    let u0 = pu / a;
    let v0 = pv / b;
    let s0 = sqrt(max(1.0 - abs(dot(u0, v0)), 0.0));
    let du0 = max(abs(r.su[0] - u0), abs(r.su[1] - u0));
    let dv0 = max(abs(r.sv[0] - v0), abs(r.sv[1] - v0));
    let spread = sqrt(dot(du0, du0) + dot(dv0, dv0));
    r.lg = select(0.0, (s0 - spread) * (s0 - spread) * (1.0 - 1e-5), s0 > spread);
    r.ok = true;
    return r;
}

// dot_iv over bounds held in registers.
fn dot_iv2(lo: vec3<f32>, hi: vec3<f32>, p: vec3<f32>, d: array<vec3<f32>, 2>) -> vec2<f32> {
    let el = lo - p;
    let eh = hi - p;
    var r = vec2<f32>(0.0);
    var mag = 0.0;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let q = vec4<f32>(el[k] * d[0][k], el[k] * d[1][k], eh[k] * d[0][k], eh[k] * d[1][k]);
        r = r + vec2<f32>(min(min(q.x, q.y), min(q.z, q.w)), max(max(q.x, q.y), max(q.z, q.w)));
        mag = mag + max(max(abs(q.x), abs(q.y)), max(abs(q.z), abs(q.w)));
    }
    // The rounding of the products and sums, bounded by the terms' size too (where they
    // cancel, the result's own size says nothing of it).
    let m = 1e-5 * (abs(r.x) + abs(r.y)) + 2e-6 * mag + 1e-30;
    return vec2<f32>(r.x - m, r.y + m);
}

fn mag2_hi2(d: array<vec3<f32>, 2>) -> f32 {
    let m = max(d[0] * d[0], d[1] * d[1]);
    return (m.x + m.y + m.z) * (1.0 + 1e-5);
}

// As `sub_kind`, for a part made on the fly.
fn part_kind(r: Part, p: vec3<f32>) -> u32 {
    if (excludes_zero(dot_iv2(r.lo, r.hi, p, r.su)) || excludes_zero(dot_iv2(r.lo, r.hi, p, r.sv))) {
        return 1u;
    }
    let a = dot_iv2(r.lo, r.hi, p, r.suu);
    let b = dot_iv2(r.lo, r.hi, p, r.suv);
    let c = dot_iv2(r.lo, r.hi, p, r.svv);
    if (mag2_hi2(r.su) + a.y < 0.0 || mag2_hi2(r.sv) + c.y < 0.0) {
        return 1u;
    }
    if (saddle_throughout(r.su[0], r.su[1], r.sv[0], r.sv[1], a, b, c)) {
        return 1u;
    }
    let bm = max(abs(b.x), abs(b.y));
    let lm = 0.5 * ((a.x + c.x) - sqrt((a.x - c.x) * (a.x - c.x) + 4.0 * bm * bm));
    return select(0u, 2u, r.lg > 0.0 && r.lg + lm > 1e-4 * r.lg);
}

// A leaf (record sb) the node could not decide, refined in place: its quarters, theirs, to
// six levels below it, each made from the span's patch and read by the same tests.
fn refine(sb: u32, p: vec3<f32>, h: ptr<function, Hit>, clamp_d: f32, lb0: f32) {
    let pb = gu(sb + 7u);
    let fb = P.faces + gu(sb + 12u) * FACE;
    let known_in = gu(sb + 13u) == 1u;
    var st: array<vec4<f32>, 32>;
    var dep: array<u32, 32>;
    var n = 0u;
    let b0 = vec4<f32>(g(sb + 8u), g(sb + 9u), g(sb + 10u), g(sb + 11u));
    let sm = 0.5 * (b0.x + b0.y);
    let tm = 0.5 * (b0.z + b0.w);
    st[0] = vec4<f32>(b0.x, sm, b0.z, tm);
    st[1] = vec4<f32>(b0.x, sm, tm, b0.w);
    st[2] = vec4<f32>(sm, b0.y, b0.z, tm);
    st[3] = vec4<f32>(sm, b0.y, tm, b0.w);
    for (var k = 0u; k < 4u; k = k + 1u) {
        dep[k] = 1u;
    }
    n = 4u;
    while (n > 0u) {
        n = n - 1u;
        let bx = st[n];
        let d = dep[n];
        if (budget == 0u) {
            undecided(h, lb0, 8u);
            return;
        }
        budget = budget - 1u;
        let r = make_part(pb, bx);
        if (!r.ok) {
            undecided(h, lb0, 1u);
            continue;
        }
        let lb = box_dist(p, r.lo, r.hi);
        if (lb >= (*h).d) {
            continue;
        }
        let kind = part_kind(r, p);
        if (kind == 1u) {
            continue;
        }
        if (kind == 2u) {
            let m = box_min(pb, bx.xz, bx.yw, p);
            if (m.state == 2u) {
                undecided(h, lb, 2u);
            } else if (m.state == 1u && m.d < (*h).d) {
                let w = vec2<f32>(mix(g(pb + 2u), g(pb + 3u), m.st.x), mix(g(pb + 4u), g(pb + 5u), m.st.y));
                if (known_in || inside(fb, w)) {
                    (*h).d = m.d;
                    (*h).s = sgn(dot(p - m.q, m.n * g(fb + 1u)));
                }
            }
            continue;
        }
        if (d >= 6u || lb > clamp_d || n + 4u > 32u) {
            undecided(h, lb, 1u);
            continue;
        }
        let s2 = 0.5 * (bx.x + bx.y);
        let t2 = 0.5 * (bx.z + bx.w);
        st[n] = vec4<f32>(bx.x, s2, bx.z, t2);
        st[n + 1u] = vec4<f32>(bx.x, s2, t2, bx.w);
        st[n + 2u] = vec4<f32>(s2, bx.y, bx.z, t2);
        st[n + 3u] = vec4<f32>(s2, bx.y, t2, bx.w);
        for (var k = 0u; k < 4u; k = k + 1u) {
            dep[n + k] = d + 1u;
        }
        n = n + 4u;
    }
}

// ------------------------------------------------------------------ walking the trees

// The interval of (S - p).D over a part whose points S lie in the box [lo, hi], D a
// derivative whose components lie in the bounds stored from `base` (x lo, x hi, y lo,
// y hi, z lo, z hi), widened for single-precision rounding.
fn dot_iv(lo: vec3<f32>, hi: vec3<f32>, p: vec3<f32>, base: u32) -> vec2<f32> {
    return dot_iv_e(lo - p, hi - p, base);
}

// The same with S - p known only to lie in the box [el, eh] (p anywhere in a brick).
fn dot_iv_e(el: vec3<f32>, eh: vec3<f32>, base: u32) -> vec2<f32> {
    var r = vec2<f32>(0.0);
    var mag = 0.0;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let a0 = el[k];
        let a1 = eh[k];
        let b0 = g(base + 2u * k);
        let b1 = g(base + 2u * k + 1u);
        let q = vec4<f32>(a0 * b0, a0 * b1, a1 * b0, a1 * b1);
        r = r + vec2<f32>(min(min(q.x, q.y), min(q.z, q.w)), max(max(q.x, q.y), max(q.z, q.w)));
        mag = mag + max(max(abs(q.x), abs(q.y)), max(abs(q.z), abs(q.w)));
    }
    // The rounding of the products and sums, bounded by the terms' size too (where they
    // cancel, the result's own size says nothing of it).
    let m = 1e-5 * (abs(r.x) + abs(r.y)) + 2e-6 * mag + 1e-30;
    return vec2<f32>(r.x - m, r.y + m);
}

fn excludes_zero(r: vec2<f32>) -> bool {
    return r.x > 0.0 || r.y < 0.0;
}

fn iv_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q = vec4<f32>(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
    return vec2<f32>(min(min(q.x, q.y), min(q.z, q.w)), max(max(q.x, q.y), max(q.z, q.w)));
}

// The interval of x.y over x, y within componentwise bounds.
fn dot_bounds(alo: vec3<f32>, ahi: vec3<f32>, blo: vec3<f32>, bhi: vec3<f32>) -> vec2<f32> {
    var r = vec2<f32>(0.0);
    for (var k = 0u; k < 3u; k = k + 1u) {
        r = r + iv_mul(vec2<f32>(alo[k], ahi[k]), vec2<f32>(blo[k], bhi[k]));
    }
    return r;
}

// Whether the Hessian G + M (entries as intervals: g11, g12, g22 from the tangents' bounds
// u, v, and m11, m12, m22) has a negative determinant throughout: then it is indefinite
// everywhere, and a critical point is a saddle, never a minimum.
fn saddle_throughout(ulo: vec3<f32>, uhi: vec3<f32>, vlo: vec3<f32>, vhi: vec3<f32>, m11: vec2<f32>, m12: vec2<f32>, m22: vec2<f32>) -> bool {
    let h11 = dot_bounds(ulo, uhi, ulo, uhi) + m11;
    let h22 = dot_bounds(vlo, vhi, vlo, vhi) + m22;
    let h12 = dot_bounds(ulo, uhi, vlo, vhi) + m12;
    let prod = iv_mul(h11, h22);
    let off = select(min(h12.x * h12.x, h12.y * h12.y), 0.0, h12.x <= 0.0 && h12.y >= 0.0);
    let m = 1e-5 * (abs(prod.y) + off) + 1e-30;
    return prod.y + m < off;
}

// The greatest |D|^2 over a part, D's bounds stored from `base`.
fn mag2_hi(base: u32) -> f32 {
    var m = 0.0;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let a = g(base + 2u * k);
        let b = g(base + 2u * k + 1u);
        m = m + max(a * a, b * b);
    }
    return m * (1.0 + 1e-5);
}

// Whether |S - p|^2 has no interior minimum on a sub-patch (record sb), or is strictly
// convex on it for this node: 0 neither; 1 no interior minimum, because the gradient's s
// or t part keeps one sign over the part (no critical point), or the Hessian's ss or tt
// entry, |S_s|^2 + (S - p).S_ss, is negative throughout (past a concave surface's center
// of curvature: a critical point there is no minimum), so its least point is on its
// border, which its neighbors and the face's edges answer; 2 convex (G + [(S - p).S_ij] has
// lambda_min(G) + lambda_min([(S - p).S_ij]) > 0, both bounded over the part: the node's
// own side of the surface, where the radius alone takes the worst side).
fn sub_kind(sb: u32, p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> u32 {
    return sub_kind_e(sb, lo - p, hi - p);
}

// The same for every node whose offset S - p lies in the box [el, eh].
fn sub_kind_e(sb: u32, el: vec3<f32>, eh: vec3<f32>) -> u32 {
    if (excludes_zero(dot_iv_e(el, eh, sb + 17u)) || excludes_zero(dot_iv_e(el, eh, sb + 23u))) {
        return 1u;
    }
    let lg = g(sb + 16u);
    let a = dot_iv_e(el, eh, sb + 29u);
    let b = dot_iv_e(el, eh, sb + 35u);
    let c = dot_iv_e(el, eh, sb + 41u);
    if (mag2_hi(sb + 17u) + a.y < 0.0 || mag2_hi(sb + 23u) + c.y < 0.0) {
        return 1u;
    }
    let bm = max(abs(b.x), abs(b.y));
    let lm = 0.5 * ((a.x + c.x) - sqrt((a.x - c.x) * (a.x - c.x) + 4.0 * bm * bm));
    return select(0u, 2u, lg > 0.0 && lg + lm > 1e-4 * lg);
}

// The same for a piece (record cb): (C - p).C' of one sign or |C'|^2 + (C - p).C'' < 0
// throughout (no interior minimum), or |C'|^2 + (C - p).C'' > 0 (convex).
fn piece_kind(cb: u32, p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> u32 {
    return piece_kind_e(cb, lo - p, hi - p);
}

// The same for every node whose offset C - p lies in the box [el, eh].
fn piece_kind_e(cb: u32, el: vec3<f32>, eh: vec3<f32>) -> u32 {
    if (excludes_zero(dot_iv_e(el, eh, cb + 22u))) {
        return 1u;
    }
    let l = g(cb + 21u);
    let k = dot_iv_e(el, eh, cb + 28u);
    if (mag2_hi(cb + 22u) + k.y < 0.0) {
        return 1u;
    }
    return select(0u, 2u, l > 0.0 && l + k.x > 1e-4 * l);
}

// What a span's or piece's tree gives a node: a nearer answer (d below the best it was
// given: its sign, or for an edge's piece the edge and its point, signed later), and the
// least lower bound among the parts it left undecided.
struct Hit {
    d: f32,
    s: f32,
    lmin: f32,
    q: vec3<f32>,
    edge: u32,
    // Why lmin is what it is (for the caller's report): 1 a part not convex for the
    // node within the clamp, 2 Newton unsettled, 6 the walk's stack full.
    why: u32,
};

fn undecided(h: ptr<function, Hit>, lb: f32, why: u32) {
    if (lb < (*h).lmin) {
        (*h).lmin = lb;
        (*h).why = why;
    }
}

// Pushes the parts of a record (`count` from `first`, records of `size` words from
// `base`) on to the stack, the nearest to p last (so read first).
fn push_parts(stack: ptr<function, array<u32, 48>>, sp: ptr<function, u32>, base: u32, size: u32, first: u32, count: u32, p: vec3<f32>) {
    var d: array<f32, 4>;
    for (var k = 0u; k < count; k = k + 1u) {
        let rb = base + (first + k) * size;
        d[k] = box_dist(p, g3(rb), g3(rb + 3u));
    }
    var used = 0u;
    for (var n = 0u; n < count; n = n + 1u) {
        var fk = 0u;
        var fd = -1.0;
        for (var k = 0u; k < count; k = k + 1u) {
            if ((used & (1u << k)) == 0u && d[k] > fd) {
                fd = d[k];
                fk = k;
            }
        }
        used = used | (1u << fk);
        (*stack)[*sp] = first + fk;
        *sp = *sp + 1u;
    }
}

// A B-spline span's tree for node p, given its best answer so far: parts no nearer than
// that are passed over, parts p is close enough to for their radius are read by Newton,
// others opened while they come within the clamp of p.
fn sub_tree(root: u32, p: vec3<f32>, best: f32, clamp_d: f32) -> Hit {
    var h = Hit(best, 1.0, BIG, vec3<f32>(0.0), NONE, 0u);
    var stack: array<u32, 48>;
    var sp = 1u;
    stack[0] = root;
    while (sp > 0u) {
        sp = sp - 1u;
        let sb = P.subs + stack[sp] * SUB;
        let lo = g3(sb);
        let hi = g3(sb + 3u);
        let lb = box_dist(p, lo, hi);
        if (lb >= h.d) {
            continue;
        }
        var kind = select(0u, 2u, far_dist(p, lo, hi) < g(sb + 6u) * 0.999 - P.eps);
        if (kind == 0u) {
            kind = sub_kind(sb, p, lo, hi);
        }
        if (kind == 1u) {
            continue;
        }
        if (kind == 2u) {
            let m = sub_min(sb, p);
            if (m.state == 2u) {
                undecided(&h, lb, 2u);
            } else if (m.state == 1u && m.d < h.d) {
                let fb = P.faces + gu(sb + 12u) * FACE;
                let pb = gu(sb + 7u);
                let w = vec2<f32>(mix(g(pb + 2u), g(pb + 3u), m.st.x), mix(g(pb + 4u), g(pb + 5u), m.st.y));
                if (gu(sb + 13u) == 1u || inside(fb, w)) {
                    h.d = m.d;
                    h.s = sgn(dot(p - m.q, m.n * g(fb + 1u)));
                }
            }
            continue;
        }
        let nk = gu(sb + 15u);
        // Only the settling pass (a thread for each node a brick left undecided) refines a
        // leaf: in a brick one refining thread would hold up its whole group, and a box's
        // center, read to decide the box, would refine every leaf within the box.
        if (REFINE && nk == 0u && lb <= clamp_d) {
            if (n_deferred < 16u) {
                deferred[n_deferred] = vec2<u32>(sb, bitcast<u32>(lb));
                n_deferred = n_deferred + 1u;
            } else {
                undecided(&h, lb, 7u);
            }
            continue;
        }
        if (nk == 0u || lb > clamp_d || sp + nk > 48u) {
            undecided(&h, lb, select(1u, 6u, sp + nk > 48u));
            continue;
        }
        push_parts(&stack, &sp, P.subs, SUB, gu(sb + 14u), nk, p);
    }
    return h;
}

// The same for a curve's piece: an edge's, or a knot line's (whose point must lie in its
// face, signed by the two sides' normals).
fn curve_tree(root: u32, p: vec3<f32>, best: f32, clamp_d: f32) -> Hit {
    var h = Hit(best, 1.0, BIG, vec3<f32>(0.0), NONE, 0u);
    var stack: array<u32, 48>;
    var sp = 1u;
    stack[0] = root;
    while (sp > 0u) {
        sp = sp - 1u;
        let cb = P.curves + stack[sp] * CURVE;
        let lo = g3(cb);
        let hi = g3(cb + 3u);
        let lb = box_dist(p, lo, hi);
        if (lb >= h.d) {
            continue;
        }
        var kind = select(0u, 2u, far_dist(p, lo, hi) < g(cb + 6u) * 0.999 - P.eps);
        if (kind == 0u) {
            kind = piece_kind(cb, p, lo, hi);
        }
        if (kind == 1u) {
            continue;
        }
        if (kind == 2u) {
            let m = curve_min(cb, p);
            if (m.state == 2u) {
                undecided(&h, lb, 5u);
            } else if (m.state == 1u && m.d < h.d) {
                if (gu(cb + 11u) == 0u) {
                    h.d = m.d;
                    h.edge = gu(cb + 12u);
                    h.q = m.q;
                } else {
                    let fb = P.faces + gu(cb + 12u) * FACE;
                    let kk = gu(cb + 13u);
                    let w = mix(g(cb + 15u), g(cb + 16u), m.st.x);
                    let at = select(vec2<f32>(w, g(cb + 14u)), vec2<f32>(g(cb + 14u), w), kk == 0u);
                    if (inside(fb, at)) {
                        let n = (side_normal(gu(cb + 17u), kk, 1.0, w) + side_normal(gu(cb + 18u), kk, 0.0, w)) * g(fb + 1u);
                        h.d = m.d;
                        h.s = sgn(dot(p - m.q, n));
                        h.edge = NONE;
                    }
                }
            }
            continue;
        }
        let nk = gu(cb + 20u);
        if (nk == 0u || lb > clamp_d || sp + nk > 48u) {
            undecided(&h, lb, select(1u, 6u, sp + nk > 48u));
            continue;
        }
        push_parts(&stack, &sp, P.curves, CURVE, gu(cb + 19u), nk, p);
    }
    return h;
}

// ------------------------------------------------------------------ the nodes

// What a node's candidates give it: its best answer and that answer's sign, whether the
// sign is known, and the least lower bound among the candidates it could not decide, with
// why.
struct NodeHit {
    best: f32,
    s: f32,
    lmin: f32,
    why: u32,
    known: bool,
};

// A node's candidates read; B-spline spans and curve pieces only when `splines` (the
// scatter reads those itself).
fn node_hit(b: Brick, p: vec3<f32>, clamp_d: f32, splines: bool) -> NodeHit {
    // A capped record past its clamp keeps only that bound (`node_value`, `finish`), so its
    // answer starts just past the clamp: a feature farther than that is passed over (a face
    // without its trim test, an edge without the search for its sign). Two ulps past: a key
    // keeps its sign in the lowest bit (`best_key`), which takes one away when it is set,
    // and the answer must still read as past the clamp. Another point's answer is wanted
    // past its clamp too.
    var best = select(BIG, bitcast<f32>(bitcast<u32>(clamp_d) + 2u), capped(b));
    var s = 1.0;
    // The least lower bound among candidates the node could not decide.
    var lmin = BIG;
    var why = 0u;
    n_deferred = 0u;
    budget = 256u;
    // When the nearest point is an edge's: the edge and the point, signed at the end.
    var edge = NONE;
    var eq = vec3<f32>(0.0);
    for (var k = 0u; k < b.c.y; k = k + 1u) {
        let c = cands[b.c.x + k];
        let kind = c >> 28u;
        let j = c & 0x0fffffffu;
        if (kind == 0u) {
            let r = face_d(j, p, best);
            if (r.x < best) {
                best = r.x;
                s = r.y;
                edge = NONE;
            }
        } else if (kind == 1u) {
            let r = edge_foot(j, p);
            if (r.w < best) {
                best = r.w;
                edge = j;
                eq = r.xyz;
            }
        } else if (kind == 2u) {
            let vb = P.verts + j * VERT;
            let x = g3(vb);
            let d = length(p - x);
            if (d < best) {
                best = d;
                s = sgn(dot(p - x, g3(vb + 3u)));
                edge = NONE;
            }
        } else if (!splines) {
            // A list holds its closed-form features first (`GpuPack::list`): the rest are
            // spans and pieces, which the scatter reads.
            break;
        } else {
            var h: Hit;
            if (kind == 3u) {
                h = sub_tree(j, p, best, clamp_d);
            } else {
                h = curve_tree(j, p, best, clamp_d);
            }
            if (h.lmin < lmin) {
                lmin = h.lmin;
                why = h.why;
            }
            if (h.d < best) {
                best = h.d;
                s = h.s;
                edge = h.edge;
                eq = h.q;
            }
        }
    }
    // The leaves no test decided, refined nearest first while still nearer than the best.
    for (var k = 0u; REFINE && k < n_deferred; k = k + 1u) {
        var bi = k;
        for (var m = k + 1u; m < n_deferred; m = m + 1u) {
            if (bitcast<f32>(deferred[m].y) < bitcast<f32>(deferred[bi].y)) {
                bi = m;
            }
        }
        let dk = deferred[bi];
        deferred[bi] = deferred[k];
        deferred[k] = dk;
        let lbk = bitcast<f32>(dk.y);
        if (lbk >= best) {
            break;
        }
        var hh = Hit(best, s, BIG, eq, edge, 0u);
        refine(dk.x, p, &hh, clamp_d, lbk);
        if (hh.lmin < lmin) {
            lmin = hh.lmin;
            why = hh.why;
        }
        if (hh.d < best) {
            best = hh.d;
            s = hh.s;
            edge = NONE;
        }
    }
    // An edge's sign: its faces' outward normals summed at its point; where they nearly
    // cancel (a cusp) the single-precision sum's direction is unsure.
    var known = true;
    if (edge != NONE) {
        let eb = P.edges + edge * EDGE;
        var pn = face_normal(gu(eb + 13u), eq, b);
        if (gu(eb + 15u) == 2u) {
            let n2 = face_normal(gu(eb + 14u), eq, b);
            pn = vec4<f32>(pn.xyz + n2.xyz, min(pn.w, n2.w));
        }
        if (pn.w < 0.5 || length(pn.xyz) < 1e-3) {
            known = false;
            why = 3u;
        } else {
            s = sgn(dot(p - eq, pn.xyz));
        }
    }
    return NodeHit(best, s, lmin, why, known);
}

// A node's value: (signed distance, 0), (a lower bound past the clamp, 1) or (why, 2).
fn node_value(b: Brick, p: vec3<f32>, clamp_d: f32) -> vec2<f32> {
    let r = node_hit(b, p, clamp_d, true);
    if (r.best < BIG && r.lmin >= r.best && r.known && (!capped(b) || r.best <= clamp_d)) {
        return vec2<f32>(r.best * r.s, 0.0);
    }
    // Undecided: what is known is a lower bound on the distance (in a brick, past the
    // clamp it is the clamp: the features not listed lie farther).
    let l = min(r.best, r.lmin);
    if (l > clamp_d && (capped(b) || l < BIG)) {
        return vec2<f32>(select(l, clamp_d, capped(b)), 1.0);
    }
    return vec2<f32>(f32(select(r.why, 4u, r.best == BIG && r.lmin == BIG)), 2.0);
}

// A thread a node of a brick (mode 0), or a thread a point (modes 1 and 2).
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= P.nodes) {
        return;
    }
    if (P.mode >= 1u) {
        let b = bricks[i];
        out[b.c.z] = node_value(b, b.o.xyz, b.o.w);
        return;
    }
    let b = bricks[i / 729u];
    let nn = i % 729u;
    let p = b.o.xyz + b.o.w * vec3<f32>(f32(nn % 9u), f32((nn / 9u) % 9u), f32(nn / 81u));
    out[b.c.z + nn] = node_value(b, p, b.o.w * 1.7320508);
}

// ------------------------------------------------------------------ the scatter
//
// The bricks' B-spline spans and curve pieces read the other way round: each part decides
// once, for all the nodes of a brick within the clamp of its box, whether it can hold a
// node's nearest point, then writes its answer into each of those nodes, and every node
// keeps the least written (Mauch's characteristic scan conversion, for curved parts).
// Every answer written is the distance to a point of the part's face or edge, so none is
// below the node's distance; and the part holding a node's nearest point writes it, since
// that part is convex for the node (decided for its whole set of nodes, or for the node
// alone where the set's test fails at a leaf). So the least is exact, without a node ever
// searching. The passes, a dispatch each: `seed` reads every node's closed-form features
// (the gather above, its spans and pieces left out); `jobs_of` walks each listed span or
// piece of each brick to the parts that brick needs; `scatter` reads them, a workgroup a
// part; `finish` turns each node's keys into its value.
//
// Points (mode 1: the nodes bricks left undecided, settled; each record a point, its w the
// clamp) take the same passes, a thread a job, and a leaf no test decides for its point is
// refined in parallel (`refine_round`): split 8 x 8, a lane a part, each part decided as a
// leaf is, those still undecided split again next round, three rounds. As the gather reads
// a point, every part listed is read, past the clamp too, and only within the clamp is an
// undecided part opened (past it, it leaves its bound): a point's answer is exact wherever
// it is decided, and its bound past the clamp otherwise.
//
// A node's keys (`acc`, three words a node): its best answer's distance with its sign in
// the lowest bit (1 inside), and the least lower bound among the parts it could not decide
// with why in the lowest three bits; each kept by atomic min (a distance's bits order as
// the distance does). An edge's piece is signed by its faces' normals at its point, found
// by searching the faces: so it writes only its distance, to the third word, and a pass
// after the scatter (`edge_signs`) signs the nearest edge piece's answer alone.

fn best_key(d: f32, s: f32) -> u32 {
    return (bitcast<u32>(d) & ~1u) | select(0u, 1u, s < 0.0);
}

fn low_key(d: f32, why: u32) -> u32 {
    return (bitcast<u32>(d) & ~7u) | why;
}

fn put_best(slot: u32, d: f32, s: f32) {
    atomicMin(&acc[3u * slot], best_key(d, s));
}

fn put_low(slot: u32, d: f32, why: u32) {
    atomicMin(&acc[3u * slot + 1u], low_key(d, why));
}

fn put_edge(slot: u32, d: f32) {
    atomicMin(&acc[3u * slot + 2u], bitcast<u32>(d));
}

// The least answer a node has so far, signed or not (each the distance to a point of the
// boundary): nothing farther than it can come of a part.
fn answer_so_far(slot: u32) -> f32 {
    return min(bitcast<f32>(atomicLoad(&acc[3u * slot]) & ~1u), bitcast<f32>(atomicLoad(&acc[3u * slot + 2u])));
}

// How far from a node its answer must be exact: a brick's cell diagonal, a point's own.
fn clamp_of(b: Brick) -> f32 {
    return select(b.o.w * 1.7320508, b.o.w, P.mode == 1u);
}

// The farthest any point of one box lies from any point of the other.
fn far_box(alo: vec3<f32>, ahi: vec3<f32>, blo: vec3<f32>, bhi: vec3<f32>) -> f32 {
    return length(max(abs(ahi - blo), abs(bhi - alo)));
}

// The nodes of brick b that can lie within the clamp of a part's box [lo, hi]: the index
// box [lo, hi] (each 0 to 8), and whether there are any. A point is a brick of one node.
struct NodeBox {
    lo: vec3<u32>,
    hi: vec3<u32>,
    has: bool,
};

fn node_box(b: Brick, lo: vec3<f32>, hi: vec3<f32>) -> NodeBox {
    if (P.mode == 1u) {
        return NodeBox(vec3<u32>(0u), vec3<u32>(0u), true);
    }
    let h = b.o.w;
    let c = vec3<f32>(h * 1.7320508);
    let a = ceil((lo - c - b.o.xyz) / h - vec3<f32>(1e-3));
    let z = floor((hi + c - b.o.xyz) / h + vec3<f32>(1e-3));
    let l = clamp(a, vec3<f32>(0.0), vec3<f32>(8.0));
    let u = clamp(z, vec3<f32>(0.0), vec3<f32>(8.0));
    let has = all(a <= vec3<f32>(8.0)) && all(z >= vec3<f32>(0.0)) && all(l <= u);
    return NodeBox(vec3<u32>(l), vec3<u32>(u), has);
}

// Thread i's record and node (a brick's nodes, x fastest; a point is its own record).
fn node_of(i: u32) -> vec2<u32> {
    return select(vec2<u32>(i / 729u, i % 729u), vec2<u32>(i, 0u), P.mode == 1u);
}

// Job flags, above a part's record.
const JOB_CURVE: u32 = 0x10000000u;
const JOB_PROVEN: u32 = 0x20000000u;
// A point's leaf no test decided, to refine (`leaves`) if still nearer than its answer once
// the scatter has written every answer it can.
const JOB_LEAF: u32 = 0x40000000u;

// Each node's closed-form features (thread a node): its first keys.
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= P.nodes) {
        return;
    }
    let at = node_of(i);
    let b = bricks[at.x];
    let nn = at.y;
    let p = b.o.xyz + b.o.w * vec3<f32>(f32(nn % 9u), f32((nn / 9u) % 9u), f32(nn / 81u));
    let r = node_hit(b, p, clamp_of(b), false);
    let slot = b.c.z + nn;
    atomicStore(&acc[3u * slot + 2u], bitcast<u32>(BIG));
    if (r.known) {
        atomicStore(&acc[3u * slot], best_key(r.best, r.s));
        atomicStore(&acc[3u * slot + 1u], low_key(r.lmin, r.why));
    } else {
        // Its nearest closed-form feature is an edge of unknown sign: an answer only if a
        // part comes strictly nearer.
        atomicStore(&acc[3u * slot], best_key(BIG, 1.0));
        atomicStore(&acc[3u * slot + 1u], low_key(min(r.best, r.lmin), 3u));
    }
}

// A point's part to refine in round `level` (record pi, the leaf's sub record, the part's
// box in the span's parameters; or a piece's curve record with JOB_CURVE, and its range in
// the piece's parameter); when there is no room, the point is left undecided below lb.
fn emit_refine(level: u32, pi: u32, leaf: u32, bx: vec4<f32>, lb: f32, slot: u32) {
    let room = arrayLength(&refine_jobs) / 6u;
    let at = atomicAdd(&count[1u + level], 1u);
    if (at >= room) {
        put_low(slot, lb, 7u);
        return;
    }
    let e = (level * room + at) * 2u;
    refine_jobs[e] = vec4<u32>(pi, leaf, 0u, 0u);
    refine_jobs[e + 1u] = bitcast<vec4<u32>>(bx);
}

// Each listed span or piece of a brick (thread a candidate), walked down to the parts the
// brick's nodes need: a part no node of the brick comes within the clamp of, or holding no
// interior minimum for any of them, is passed over; a part convex for all of them is a
// proven job; a leaf undecided for them all is a job each node decides alone (a point's
// leaf of a span: refined).
@compute @workgroup_size(64)
fn jobs_of(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= P.ncand) {
        return;
    }
    let c = cands[i];
    let kind = c >> 28u;
    if (kind != 3u && kind != 4u) {
        return;
    }
    let bi = owner[i];
    let b = bricks[bi];
    let curve = kind == 4u;
    var stack: array<u32, 48>;
    var sp = 1u;
    stack[0] = c & 0x0fffffffu;
    while (sp > 0u) {
        sp = sp - 1u;
        let j = stack[sp];
        let base = select(P.subs + j * SUB, P.curves + j * CURVE, curve);
        let lo = g3(base);
        let hi = g3(base + 3u);
        let nb = node_box(b, lo, hi);
        if (!nb.has) {
            continue;
        }
        let qlo = b.o.xyz + b.o.w * vec3<f32>(nb.lo);
        let qhi = b.o.xyz + b.o.w * vec3<f32>(nb.hi);
        var k = select(0u, 2u, far_box(lo, hi, qlo, qhi) < g(base + 6u) * 0.999 - P.eps);
        if (k == 0u) {
            if (curve) {
                k = piece_kind_e(base, lo - qhi, hi - qlo);
            } else {
                k = sub_kind_e(base, lo - qhi, hi - qlo);
            }
        }
        if (k == 1u) {
            continue;
        }
        let nk = gu(base + select(15u, 20u, curve));
        if (k == 0u && P.mode == 1u) {
            let lb = box_dist(b.o.xyz, lo, hi);
            if (lb >= answer_so_far(b.c.z)) {
                continue;
            }
            if (lb > b.o.w) {
                put_low(b.c.z, lb, 1u);
                continue;
            }
            // A leaf (a patch's or a piece's): refined after the scatter if still worth it.
            if (nk == 0u) {
                let at = atomicAdd(&count[0], 1u);
                if (at < arrayLength(&jobs)) {
                    jobs[at] = vec2<u32>(j | select(0u, JOB_CURVE, curve) | JOB_LEAF, bi);
                }
                continue;
            }
        }
        if (k == 2u || nk == 0u || sp + nk > 48u) {
            let at = atomicAdd(&count[0], 1u);
            if (at < arrayLength(&jobs)) {
                jobs[at] = vec2<u32>(j | select(0u, JOB_CURVE, curve) | select(0u, JOB_PROVEN, k == 2u), bi);
            }
            continue;
        }
        let first = gu(base + select(14u, 19u, curve));
        for (var kk = 0u; kk < nk; kk = kk + 1u) {
            stack[sp] = first + kk;
            sp = sp + 1u;
        }
    }
}

// One part's answer at one node (slot its keys), the part's record at `base`.
fn scatter_node(b: Brick, slot: u32, p: vec3<f32>, base: u32, curve: bool, proven: bool) {
    let lo = g3(base);
    let hi = g3(base + 3u);
    let lb = box_dist(p, lo, hi);
    if (capped(b) && lb > clamp_of(b)) {
        return;
    }
    let cur = answer_so_far(slot);
    if (lb >= cur) {
        return;
    }
    if (!proven) {
        var k = select(0u, 2u, far_dist(p, lo, hi) < g(base + 6u) * 0.999 - P.eps);
        if (k == 0u) {
            if (curve) {
                k = piece_kind(base, p, lo, hi);
            } else {
                k = sub_kind(base, p, lo, hi);
            }
        }
        if (k == 1u) {
            return;
        }
        if (k == 0u) {
            put_low(slot, lb, 1u);
            return;
        }
    }
    if (!curve) {
        let m = sub_min(base, p);
        if (m.state == 2u) {
            put_low(slot, lb, 2u);
        } else if (m.state == 1u && m.d < cur) {
            let fb = P.faces + gu(base + 12u) * FACE;
            let pb = gu(base + 7u);
            let w = vec2<f32>(mix(g(pb + 2u), g(pb + 3u), m.st.x), mix(g(pb + 4u), g(pb + 5u), m.st.y));
            if (gu(base + 13u) == 1u || inside(fb, w)) {
                put_best(slot, m.d, sgn(dot(p - m.q, m.n * g(fb + 1u))));
            }
        }
        return;
    }
    let m = curve_min(base, p);
    if (m.state == 2u) {
        put_low(slot, lb, 5u);
        return;
    }
    if (m.state != 1u || m.d >= cur) {
        return;
    }
    if (gu(base + 11u) == 0u) {
        // An edge's piece: signed later, if nearest.
        put_edge(slot, m.d);
        return;
    }
    // A knot line's piece: its point must lie in its face.
    let fb = P.faces + gu(base + 12u) * FACE;
    let kk = gu(base + 13u);
    let w = mix(g(base + 15u), g(base + 16u), m.st.x);
    let at = select(vec2<f32>(w, g(base + 14u)), vec2<f32>(g(base + 14u), w), kk == 0u);
    if (inside(fb, at)) {
        let n = (side_normal(gu(base + 17u), kk, 1.0, w) + side_normal(gu(base + 18u), kk, 0.0, w)) * g(fb + 1u);
        put_best(slot, m.d, sgn(dot(p - m.q, n)));
    }
}

// The jobs: for bricks a workgroup a job (each workgroup taking every so many in turn), its
// threads sharing the job's nodes; for points a thread a job.
@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_jobs = min(atomicLoad(&count[0]), arrayLength(&jobs));
    if (P.mode == 1u) {
        for (var j = (wid.x + wid.y * nw.x) * 64u + li; j < n_jobs; j = j + nw.x * nw.y * 64u) {
            let job = jobs[j];
            if ((job.x & JOB_LEAF) != 0u) {
                continue;
            }
            let curve = (job.x & JOB_CURVE) != 0u;
            let rec = job.x & 0x0fffffffu;
            let b = bricks[job.y];
            scatter_node(b, b.c.z, b.o.xyz, select(P.subs + rec * SUB, P.curves + rec * CURVE, curve), curve, (job.x & JOB_PROVEN) != 0u);
        }
        return;
    }
    for (var j = wid.x + wid.y * nw.x; j < n_jobs; j = j + nw.x * nw.y) {
        let job = jobs[j];
        let curve = (job.x & JOB_CURVE) != 0u;
        let rec = job.x & 0x0fffffffu;
        let base = select(P.subs + rec * SUB, P.curves + rec * CURVE, curve);
        let b = bricks[job.y];
        let nb = node_box(b, g3(base), g3(base + 3u));
        let e = nb.hi - nb.lo + vec3<u32>(1u);
        let n = e.x * e.y * e.z;
        for (var k = li; k < n; k = k + 64u) {
            let q = nb.lo + vec3<u32>(k % e.x, (k / e.x) % e.y, k / (e.x * e.y));
            let p = b.o.xyz + b.o.w * vec3<f32>(f32(q.x), f32(q.y), f32(q.z));
            scatter_node(b, b.c.z + q.x + 9u * q.y + 81u * q.z, p, base, curve, (job.x & JOB_PROVEN) != 0u);
        }
    }
}

// A part of a piece made on the fly (as `make_part` makes a patch's): its box, and its
// derivatives' bounds in the piece's parameter from the hodographs packed after its points
// (gpu.rs `piece_hodographs`: C' 3 points, C'' 2, then each net's largest coordinate). A
// piece not a polynomial cubic is not made (ok false).
struct PiecePart {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    d1: array<vec3<f32>, 2>,
    d2: array<vec3<f32>, 2>,
};

fn make_piece(cb: u32, t0: f32, t1: f32) -> PiecePart {
    var r: PiecePart;
    r.ok = false;
    let pts = gu(cb + 8u);
    if (gu(cb + 7u) != 3u || g(pts + 31u) < 0.0) {
        return r;
    }
    let q = seg3(Quad(g4(pts), g4(pts + 4u), g4(pts + 8u), g4(pts + 12u)), t0, t1);
    let a = q.a.xyz / q.a.w;
    let b = q.b.xyz / q.b.w;
    let c = q.c.xyz / q.c.w;
    let d = q.d.xyz / q.d.w;
    let lo = min(min(a, b), min(c, d));
    let hi = max(max(a, b), max(c, d));
    let pad = vec3<f32>(P.eps) + 1e-6 * max(abs(lo), abs(hi));
    r.lo = lo - pad;
    r.hi = hi + pad;
    let h = pts + 16u;
    var d1 = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull3(&d1, cut2(g3(h), g3(h + 3u), g3(h + 6u), t0, t1));
    let p1 = vec3<f32>(4e-6 * g(h + 15u));
    r.d1 = array<vec3<f32>, 2>(d1[0] - p1, d1[1] + p1);
    var d2 = array<vec3<f32>, 2>(vec3<f32>(BIG), vec3<f32>(-BIG));
    hull2(&d2, cut1(g3(h + 9u), g3(h + 12u), t0, t1));
    let p2 = vec3<f32>(4e-6 * g(h + 16u));
    r.d2 = array<vec3<f32>, 2>(d2[0] - p2, d2[1] + p2);
    r.ok = true;
    return r;
}

// As `piece_kind`, for a piece's part made on the fly: 1 no interior minimum, 2 convex,
// 0 neither.
fn piece_part_kind(r: PiecePart, p: vec3<f32>) -> u32 {
    if (excludes_zero(dot_iv2(r.lo, r.hi, p, r.d1))) {
        return 1u;
    }
    let k = dot_iv2(r.lo, r.hi, p, r.d2);
    if (mag2_hi2(r.d1) + k.y < 0.0) {
        return 1u;
    }
    // |C'|^2's lower bound: the least |C'| over its box, each coordinate apart.
    let m = select(min(abs(r.d1[0]), abs(r.d1[1])), vec3<f32>(0.0), r.d1[0] <= vec3<f32>(0.0) & r.d1[1] >= vec3<f32>(0.0));
    let l = dot(m, m) * (1.0 - 1e-5);
    return select(0u, 2u, l > 0.0 && l + k.x > 1e-4 * l);
}

// A piece's critical point m written as a point's answer (slot its keys): an edge's signed
// by its faces' normals there (as `edge_sign_node` signs it), a knot line's when it lies in
// its face, signed by the two sides' normals.
fn piece_answer(b: Brick, slot: u32, p: vec3<f32>, cb: u32, m: Min) {
    if (gu(cb + 11u) == 0u) {
        let eb = P.edges + gu(cb + 12u) * EDGE;
        var pn = edge_face_normal(gu(eb + 13u), m.q, b, cb, 34u);
        if (gu(eb + 15u) == 2u) {
            let n2 = edge_face_normal(gu(eb + 14u), m.q, b, cb, 37u);
            pn = vec4<f32>(pn.xyz + n2.xyz, min(pn.w, n2.w));
        }
        if (pn.w < 0.5 || length(pn.xyz) < 1e-3) {
            put_low(slot, m.d, 3u);
        } else {
            put_best(slot, m.d, sgn(dot(p - m.q, pn.xyz)));
        }
        return;
    }
    let fb = P.faces + gu(cb + 12u) * FACE;
    let kk = gu(cb + 13u);
    let w = mix(g(cb + 15u), g(cb + 16u), m.st.x);
    let at = select(vec2<f32>(w, g(cb + 14u)), vec2<f32>(g(cb + 14u), w), kk == 0u);
    if (inside(fb, at)) {
        let n = (side_normal(gu(cb + 17u), kk, 1.0, w) + side_normal(gu(cb + 18u), kk, 0.0, w)) * g(fb + 1u);
        put_best(slot, m.d, sgn(dot(p - m.q, n)));
    }
}

// One part [t0, t1] of a point's piece leaf (the point's record pi, the leaf's curve
// record), decided as `refine_part` decides a patch's part.
fn refine_piece(pi: u32, leaf: u32, t0: f32, t1: f32, level: u32) {
    let b = bricks[pi];
    let p = b.o.xyz;
    let slot = b.c.z;
    let cb = P.curves + leaf * CURVE;
    let r = make_piece(cb, t0, t1);
    if (!r.ok) {
        put_low(slot, box_dist(p, g3(cb), g3(cb + 3u)), 1u);
        return;
    }
    let lb = box_dist(p, r.lo, r.hi);
    let cur = answer_so_far(slot);
    if (lb >= cur) {
        return;
    }
    let kind = piece_part_kind(r, p);
    if (kind == 1u) {
        return;
    }
    if (kind == 2u) {
        let m = curve_min_on(cb, p, t0, t1);
        if (m.state == 2u) {
            put_low(slot, lb, 5u);
        } else if (m.state == 1u && m.d < cur) {
            piece_answer(b, slot, p, cb, m);
        }
        return;
    }
    if (level < 2u && lb <= b.o.w) {
        emit_refine(level + 1u, pi, leaf | JOB_CURVE, vec4<f32>(t0, t1, 0.0, 0.0), lb, slot);
    } else {
        put_low(slot, lb, 1u);
    }
}

// The points' leaves no test decided (JOB_LEAF jobs, a thread a job, after the scatter):
// each sent to the first refinement round when it can still hold a point nearer than the
// point's answer so far (best first: a leaf behind a nearer answer takes no room).
@compute @workgroup_size(64)
fn leaves(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_jobs = min(atomicLoad(&count[0]), arrayLength(&jobs));
    for (var j = (wid.x + wid.y * nw.x) * 64u + li; j < n_jobs; j = j + nw.x * nw.y * 64u) {
        let job = jobs[j];
        if ((job.x & JOB_LEAF) == 0u) {
            continue;
        }
        let curve = (job.x & JOB_CURVE) != 0u;
        let rec = job.x & 0x0fffffffu;
        let b = bricks[job.y];
        let base = select(P.subs + rec * SUB, P.curves + rec * CURVE, curve);
        let lb = box_dist(b.o.xyz, g3(base), g3(base + 3u));
        if (lb >= answer_so_far(b.c.z)) {
            continue;
        }
        if (curve) {
            emit_refine(0u, job.y, rec | JOB_CURVE, vec4<f32>(g(base + 9u), g(base + 10u), 0.0, 0.0), lb, b.c.z);
        } else {
            emit_refine(0u, job.y, rec, vec4<f32>(g(base + 8u), g(base + 9u), g(base + 10u), g(base + 11u)), lb, b.c.z);
        }
    }
}

// One part of a point's leaf (the point's record pi, the leaf's sub record, the part's box
// in the span's parameters), decided as a leaf is: passed over, read by Newton, split again
// in the next round (within the clamp), or left undecided.
fn refine_part(pi: u32, leaf: u32, bx: vec4<f32>, level: u32) {
    let b = bricks[pi];
    let p = b.o.xyz;
    let slot = b.c.z;
    let sb = P.subs + leaf * SUB;
    let pb = gu(sb + 7u);
    let r = make_part(pb, bx);
    if (!r.ok) {
        put_low(slot, box_dist(p, g3(sb), g3(sb + 3u)), 1u);
        return;
    }
    let lb = box_dist(p, r.lo, r.hi);
    let cur = answer_so_far(slot);
    if (lb >= cur) {
        return;
    }
    let kind = part_kind(r, p);
    if (kind == 1u) {
        return;
    }
    if (kind == 2u) {
        let m = box_min(pb, bx.xz, bx.yw, p);
        if (m.state == 2u) {
            put_low(slot, lb, 2u);
        } else if (m.state == 1u && m.d < cur) {
            let fb = P.faces + gu(sb + 12u) * FACE;
            let w = vec2<f32>(mix(g(pb + 2u), g(pb + 3u), m.st.x), mix(g(pb + 4u), g(pb + 5u), m.st.y));
            if (gu(sb + 13u) == 1u || inside(fb, w)) {
                put_best(slot, m.d, sgn(dot(p - m.q, m.n * g(fb + 1u))));
            }
        }
        return;
    }
    if (level < 2u && lb <= b.o.w) {
        emit_refine(level + 1u, pi, leaf, bx, lb, slot);
    } else {
        put_low(slot, lb, 1u);
    }
}

// The points' parts to refine in round P.round (0 to 2), a workgroup a part (each
// workgroup taking every so many in turn): a patch's split 8 x 8, a piece's 64 along its
// range, a lane a part.
@compute @workgroup_size(64)
fn refine_round(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let room = arrayLength(&refine_jobs) / 6u;
    let level = P.round;
    let n = min(atomicLoad(&count[1u + level]), room);
    let a = f32(li % 8u);
    let c = f32(li / 8u);
    for (var j = wid.x + wid.y * nw.x; j < n; j = j + nw.x * nw.y) {
        let e = (level * room + j) * 2u;
        let head = refine_jobs[e];
        let b0 = bitcast<vec4<f32>>(refine_jobs[e + 1u]);
        if ((head.y & JOB_CURVE) != 0u) {
            // A piece: 64 parts along its range, a lane each.
            let dt = (b0.y - b0.x) * 0.015625;
            let fl = f32(li);
            refine_piece(head.x, head.y & 0x0fffffffu, b0.x + dt * fl, select(b0.x + dt * (fl + 1.0), b0.y, li == 63u), level);
            continue;
        }
        let ds = (b0.y - b0.x) * 0.125;
        let dt = (b0.w - b0.z) * 0.125;
        // The last part each way ends where the whole does (no gap from rounding).
        let bx = vec4<f32>(b0.x + ds * a, select(b0.x + ds * (a + 1.0), b0.y, li % 8u == 7u), b0.z + dt * c, select(b0.z + dt * (c + 1.0), b0.w, li / 8u == 7u));
        refine_part(head.x, head.y, bx, level);
    }
}

// One edge piece at one node after the scatter: when its answer is the node's nearest, that
// answer signed by the edge's faces' normals summed at its point (as the gather signs it).
fn edge_sign_node(b: Brick, slot: u32, p: vec3<f32>, base: u32, proven: bool) {
    let lo = g3(base);
    let hi = g3(base + 3u);
    let lb = box_dist(p, lo, hi);
    let edge_d = bitcast<f32>(atomicLoad(&acc[3u * slot + 2u]));
    if ((capped(b) && lb > clamp_of(b)) || lb > edge_d || edge_d >= bitcast<f32>(atomicLoad(&acc[3u * slot]) & ~1u)) {
        return;
    }
    if (!proven && !(far_dist(p, lo, hi) < g(base + 6u) * 0.999 - P.eps) && piece_kind(base, p, lo, hi) != 2u) {
        return;
    }
    let m = curve_min(base, p);
    // Its own answer again (to the rounding of a recomputation).
    if (m.state != 1u || m.d > edge_d * (1.0 + 1e-6)) {
        return;
    }
    let eb = P.edges + gu(base + 12u) * EDGE;
    var pn = edge_face_normal(gu(eb + 13u), m.q, b, base, 34u);
    if (gu(eb + 15u) == 2u) {
        let n2 = edge_face_normal(gu(eb + 14u), m.q, b, base, 37u);
        pn = vec4<f32>(pn.xyz + n2.xyz, min(pn.w, n2.w));
    }
    if (pn.w < 0.5 || length(pn.xyz) < 1e-3) {
        put_low(slot, m.d, 3u);
    } else {
        put_best(slot, m.d, sgn(dot(p - m.q, pn.xyz)));
    }
}

// The edges' pieces among the jobs, as `scatter` takes them.
@compute @workgroup_size(64)
fn edge_signs(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n_jobs = min(atomicLoad(&count[0]), arrayLength(&jobs));
    if (P.mode == 1u) {
        for (var j = (wid.x + wid.y * nw.x) * 64u + li; j < n_jobs; j = j + nw.x * nw.y * 64u) {
            let job = jobs[j];
            let base = P.curves + (job.x & 0x0fffffffu) * CURVE;
            if ((job.x & (JOB_CURVE | JOB_LEAF)) == JOB_CURVE && gu(base + 11u) == 0u) {
                let b = bricks[job.y];
                edge_sign_node(b, b.c.z, b.o.xyz, base, (job.x & JOB_PROVEN) != 0u);
            }
        }
        return;
    }
    for (var j = wid.x + wid.y * nw.x; j < n_jobs; j = j + nw.x * nw.y) {
        let job = jobs[j];
        if ((job.x & JOB_CURVE) == 0u) {
            continue;
        }
        let base = P.curves + (job.x & 0x0fffffffu) * CURVE;
        if (gu(base + 11u) != 0u) {
            continue;
        }
        let b = bricks[job.y];
        let nb = node_box(b, g3(base), g3(base + 3u));
        let e = nb.hi - nb.lo + vec3<u32>(1u);
        let n = e.x * e.y * e.z;
        for (var k = li; k < n; k = k + 64u) {
            let q = nb.lo + vec3<u32>(k % e.x, (k / e.x) % e.y, k / (e.x * e.y));
            let p = b.o.xyz + b.o.w * vec3<f32>(f32(q.x), f32(q.y), f32(q.z));
            edge_sign_node(b, b.c.z + q.x + 9u * q.y + 81u * q.z, p, base, (job.x & JOB_PROVEN) != 0u);
        }
    }
}

// Each node's keys as its value (thread a node), as `node_value` gives it: an edge's answer
// still unsigned (nearer than every signed one) leaves it undecided.
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= P.nodes) {
        return;
    }
    let at = node_of(i);
    let b = bricks[at.x];
    let slot = b.c.z + at.y;
    let bk = atomicLoad(&acc[3u * slot]);
    let lk = atomicLoad(&acc[3u * slot + 1u]);
    let best = bitcast<f32>(bk & ~1u);
    let s = select(1.0, -1.0, (bk & 1u) == 1u);
    let low = bitcast<f32>(lk & ~7u);
    let edge_d = bitcast<f32>(atomicLoad(&acc[3u * slot + 2u]));
    let unsigned = edge_d * (1.0 + 1e-6) < best;
    let clamp_d = clamp_of(b);
    let none = 1e38;
    if (best < none && low >= best && !unsigned && (!capped(b) || best <= clamp_d)) {
        out[slot] = vec2<f32>(best * s, 0.0);
        return;
    }
    let l = min(min(best, low), edge_d);
    if (l > clamp_d && (capped(b) || l < none)) {
        out[slot] = vec2<f32>(select(l, clamp_d, capped(b)), 1.0);
        return;
    }
    let why = select(lk & 7u, 3u, unsigned && edge_d < low);
    out[slot] = vec2<f32>(f32(select(why, 4u, best >= none && low >= none && edge_d >= none)), 2.0);
}
