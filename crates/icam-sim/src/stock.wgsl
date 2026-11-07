// The machined stock's height field H(x, y) = min over sweeps of the lowest tool-bottom
// height each sweep reaches over (x, y), capped at the stock top: one thread per grid
// node, reading only the sweeps the candidate grid lists for its cell. A port of
// icam-stock's closed forms (implicit.rs: profile, height, arc_height, closest_on_arc,
// end_distance) to 32-bit floats; every position is relative to the stock box's low
// corner, so the rounding stays near 0.0002 mm on a 2.4 m sheet.

struct Params {
    nx: u32,        // grid nodes this dispatch (columns)
    ny: u32,        // rows this dispatch
    row0: u32,      // first row of this dispatch in the whole grid
    gnx: u32,       // candidate grid size
    gny: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    x0: f32,        // grid node (0, 0), relative to the stock corner
    y0: f32,
    step: f32,      // node spacing
    cell: f32,      // candidate cell
    ox: f32,        // candidate grid corner
    oy: f32,
    top: f32,       // stock top
    pad3: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> sw: array<f32>;
@group(0) @binding(2) var<storage, read> starts: array<u32>;
@group(0) @binding(3) var<storage, read> items: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const TAU: f32 = 6.283185307179586;
const BIG: f32 = 3.0e38;

fn mod_2pi(a: f32) -> f32 {
    let m = a - TAU * floor(a / TAU);
    return select(m, 0.0, m >= TAU);
}

// Tool bottom above its tip at radial distance rho (BIG outside the tool).
fn profile(i: u32, rho: f32) -> f32 {
    let r = sw[i + 10u];
    let rc = sw[i + 11u];
    let cot = sw[i + 12u];
    if (rho > r + 1e-6) { return BIG; }
    if (cot > 0.0) { return rho * cot; }
    let flat = r - rc;
    if (rho <= flat) { return 0.0; }
    let u = min(rho - flat, rc);
    return rc - sqrt(max(rc * rc - u * u, 0.0));
}

fn line_height(i: u32, x: f32, y: f32) -> f32 {
    let ax = sw[i + 1u]; let ay = sw[i + 2u]; let az = sw[i + 3u];
    let bx = sw[i + 4u]; let by = sw[i + 5u]; let bz = sw[i + 6u];
    let r = sw[i + 10u]; let rc = sw[i + 11u];
    let dx = bx - ax; let dy = by - ay; let wx = x - ax; let wy = y - ay;
    let a = dx * dx + dy * dy;
    if (a < 1e-12) {
        return min(az, bz) + profile(i, sqrt(wx * wx + wy * wy));
    }
    if (rc > 1e-6 || abs(bz - az) < 1e-6) {
        let t = clamp((wx * dx + wy * dy) / a, 0.0, 1.0);
        let ex = wx - t * dx; let ey = wy - t * dy;
        return az + t * (bz - az) + profile(i, sqrt(ex * ex + ey * ey));
    }
    // A flat tool on a ramp: the stretch of t where it covers (x, y) solves |w - t d| <= r.
    let b = -2.0 * (wx * dx + wy * dy);
    let c = wx * wx + wy * wy - r * r;
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { return BIG; }
    let sq = sqrt(disc);
    let t0 = max((-b - sq) / (2.0 * a), 0.0);
    let t1 = min((-b + sq) / (2.0 * a), 1.0);
    if (t0 > t1) { return BIG; }
    return min(az + t0 * (bz - az), az + t1 * (bz - az));
}

fn end_distance(i: u32, qx: f32, qy: f32, phi: f32) -> f32 {
    let arc_r = sw[i + 13u]; let theta0 = sw[i + 14u]; let sweep = sw[i + 15u];
    let ang = select(theta0 - phi, theta0 + phi, sweep >= 0.0);
    let ex = qx - arc_r * cos(ang); let ey = qy - arc_r * sin(ang);
    return sqrt(ex * ex + ey * ey);
}

fn arc_height(i: u32, x: f32, y: f32) -> f32 {
    let az = sw[i + 3u]; let bz = sw[i + 6u];
    let qx = x - sw[i + 7u]; let qy = y - sw[i + 8u];
    let r = sw[i + 10u]; let rc = sw[i + 11u];
    let ra = sw[i + 13u]; let theta0 = sw[i + 14u]; let sweep = sw[i + 15u];
    let d = sqrt(qx * qx + qy * qy);
    let span = abs(sweep);
    let dz = bz - az;
    if (abs(dz) <= 1e-6 || rc > 1e-6) {
        // Closest point on the arc.
        let psi = atan2(qy, qx);
        let along = select(mod_2pi(theta0 - psi), mod_2pi(psi - theta0), sweep >= 0.0);
        var rho: f32;
        var phi: f32;
        if (along <= span) {
            rho = abs(d - ra);
            phi = along;
        } else {
            let e0 = end_distance(i, qx, qy, 0.0);
            let e1 = end_distance(i, qx, qy, span);
            rho = min(e0, e1);
            phi = select(span, 0.0, e0 <= e1);
        }
        let z = az + select(dz * phi / span, 0.0, span < 1e-9);
        return z + profile(i, rho);
    }
    // A helix: the window of angles that covers Q, clipped to the arc; z is linear in
    // the angle, so the lowest is at an end of the window.
    var lo: f32;
    var hi: f32;
    if (d < 1e-6) {
        if (ra > r) { return BIG; }
        lo = 0.0;
        hi = span;
    } else {
        let k = (d * d + ra * ra - r * r) / (2.0 * d * ra);
        if (k > 1.0) { return BIG; }
        if (k <= -1.0) {
            lo = 0.0;
            hi = span;
        } else {
            let alpha = acos(k);
            let psi = atan2(qy, qx);
            let mid = select(mod_2pi(theta0 - psi), mod_2pi(psi - theta0), sweep >= 0.0);
            lo = mid - alpha;
            hi = mid + alpha;
        }
    }
    var best = BIG;
    for (var w = -1; w <= 1; w++) {
        let a0 = max(lo + f32(w) * TAU, 0.0);
        let a1 = min(hi + f32(w) * TAU, span);
        if (a0 > a1) { continue; }
        best = min(best, min(az + dz * a0 / span, az + dz * a1 / span));
    }
    return best;
}

fn height(k: u32, x: f32, y: f32) -> f32 {
    let i = 16u * k;
    let kind = sw[i];
    if (kind < 0.5) { return line_height(i, x, y); }
    if (kind < 1.5) { return arc_height(i, x, y); }
    // A drill point.
    let ax = sw[i + 1u]; let ay = sw[i + 2u];
    let rho = sqrt((x - ax) * (x - ax) + (y - ay) * (y - ay));
    return min(sw[i + 3u], sw[i + 6u]) + profile(i, rho);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.nx || id.y >= P.ny) { return; }
    let x = P.x0 + f32(id.x) * P.step;
    let y = P.y0 + f32(id.y + P.row0) * P.step;
    var h = P.top;
    let cx = i32(floor((x - P.ox) / P.cell));
    let cy = i32(floor((y - P.oy) / P.cell));
    if (cx >= 0 && cy >= 0 && cx < i32(P.gnx) && cy < i32(P.gny)) {
        let c = u32(cy) * P.gnx + u32(cx);
        for (var k = starts[c]; k < starts[c + 1u]; k++) {
            h = min(h, height(items[k], x, y));
        }
    }
    out[id.y * P.nx + id.x] = h;
}
