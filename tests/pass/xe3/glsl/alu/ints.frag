#version 310 es
// Integer arithmetic per pixel (what acgpu's `ints` kernel covered): add, sub, mul, shifts (logical and
// arithmetic), and, or, xor, signed and unsigned min / max, conversions both ways, bit casts, selects
precision highp float;
precision highp int;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
layout(location = 3) uniform int u_k;
void main() {
    int i = int(gl_FragCoord.x) + 64 * int(gl_FragCoord.y) - 2048;
    uint u = uint(i) * 2654435761u;
    int a = i * 7 - u_k;
    int b = (i << 3) ^ (a >> 2);
    uint c = (u >> 5) | (uint(b) & 0xff00u);
    int d = max(min(a, b), -1000) + abs(i - 9);
    uint e = max(min(c, 4000000000u), 12u);
    float f = float(d) * 0.25 + float(e) / 4294967296.0;
    int g = i < 0 ? int(f) : int(uint(f * 3.0));
    o_color = vec4(f, float(g), intBitsToFloat(b & 0x3fffffff), uintBitsToFloat((c >> 2) | 0x3f800000u));
}
