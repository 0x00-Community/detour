#version 310 es
// Float arithmetic and math per pixel (what acgpu's `fine` kernel covered): multiply-add, min / max, floor, round to even,
// sqrt, abs, negation, division, comparisons and selects, float / int conversions
precision highp float;
precision highp int;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
layout(location = 3) uniform float u_s;
void main() {
    vec2 c = gl_FragCoord.xy;
    float x = (c.x - 31.5) / 9.0;
    float y = c.y * u_s - 2.0;
    float a = x * y - 0.5;
    float b = clamp(a, -1.5, 2.5) + floor(x * 3.7) - roundEven(y * 1.3);
    float d = sqrt(abs(a) + 1.0) / (y * y + 0.25);
    float e = x > y ? -b : max(b, d);
    uint q = uint(clamp(e * 100.0 + 1000.0, 0.0, 4000.0));
    int r = int(-d * 50.0);
    o_color = vec4(e, float(q >> 2), float(r), min(a, d));
}
