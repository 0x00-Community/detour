#version 310 es
// Functions called from the entry point, one from another, with in and out parameters: inlined before the lowering
precision highp float;
precision highp int;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
float smooth01(float x) {
    float t = clamp(x, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
void split(vec2 p, out float a, out float b) {
    a = smooth01(p.x * 2.0 - 0.5);
    b = smooth01(p.y + a * 0.25);
}
void main() {
    float a, b;
    split(v_uv, a, b);
    o_color = vec4(a, b, smooth01(a * b), 1.0);
}
