// The stock's skin as it is: flat pieces whose corners the fields put on the stock's
// zero set, each corner with the surface's exact normal. Two-sided, lit from above.

struct View {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> V: View;

struct In {
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) c: vec3<f32>,
};

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) c: vec3<f32>,
};

@vertex
fn vs(i: In) -> Out {
    var o: Out;
    o.clip = V.view_proj * vec4<f32>(i.p, 1.0);
    o.world = i.p;
    o.n = i.n;
    o.c = i.c;
    return o;
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    var n = normalize(o.n);
    let view = normalize(V.eye.xyz - o.world);
    if (dot(n, view) < 0.0) {
        n = -n;
    }
    let l = normalize(V.light.xyz);
    let half = normalize(l + view);
    let spec = pow(max(dot(n, half), 0.0), 48.0);
    let shade = 0.22 + 0.58 * max(dot(n, l), 0.0) + 0.22 * max(dot(n, view), 0.0);
    return vec4<f32>(o.c * shade + vec3<f32>(0.22 * spec), 1.0);
}
