#version 310 es
// Structured control flow per pixel (what acgpu's `loops` kernel covered): a loop the same for every pixel, one of
// a different length per pixel, an if on a per-pixel condition, an if inside a loop, a while loop
precision highp float;
precision highp int;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
void main() {
    int i = int(gl_FragCoord.x) + 64 * int(gl_FragCoord.y);
    int r0 = 0;
    for (int k = 0; k < 10; k++) {
        r0 += k * i + 3;
    }
    int r1 = 1;
    for (int k = 0; k < (i & 7); k++) {
        r1 = r1 * 3 + k;
    }
    int r2 = (i & 1) == 1 ? i * 5 : i - 100;
    int r3 = 0;
    for (int k = 0; k < 5; k++) {
        if (((i >> k) & 1) == 1) {
            r3 += k * k;
        } else {
            r3 -= 1;
        }
    }
    int r4 = i;
    int n = 0;
    while (r4 > 1 && n < 64) {
        r4 = (r4 & 1) == 0 ? r4 >> 1 : 3 * r4 + 1;
        n++;
    }
    o_color = vec4(float(r0), float(r1), float(r2 + r3), float(n));
}
