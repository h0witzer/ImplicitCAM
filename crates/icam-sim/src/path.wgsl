// The toolpath (docs/SIM_VISUALIZATION_ENGINE.md, section 4.7): every move as lines, each
// vertex with the playhead at which its move is done. Moves done are drawn gray; moves
// ahead, cyan for feeds and orange for rapids. The lines are tested against the depth the
// ray caster wrote and drawn a hair in front of it, so a path on the floor it cut shows.
// A vertex marked 2 is the box about a region the panel picked, drawn yellow.

struct Path {
    view_proj: mat4x4<f32>,
    // x: the playhead.
    at: vec4<f32>,
};

@group(0) @binding(0) var<uniform> U: Path;

struct VIn {
    @location(0) p: vec3<f32>,
    @location(1) done: f32,
    @location(2) rapid: f32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(v: VIn) -> VOut {
    var o: VOut;
    var c = U.view_proj * vec4<f32>(v.p, 1.0);
    c.z = c.z - 2e-5 * c.w;
    o.clip = c;
    let ahead = v.done > U.at.x;
    let lit = select(vec3<f32>(0.2, 0.75, 1.0), vec3<f32>(1.0, 0.5, 0.15), v.rapid > 0.5);
    o.color = select(vec4<f32>(0.42, 0.43, 0.47, 1.0), vec4<f32>(lit, 1.0), ahead);
    if (v.rapid > 1.5) {
        o.color = vec4<f32>(1.0, 0.82, 0.29, 1.0);
    }
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return i.color;
}
