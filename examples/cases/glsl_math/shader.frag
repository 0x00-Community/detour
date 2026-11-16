#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() { vec3 n = normalize(v + 0.1); float s = smoothstep(0.2, 0.8, v.x); vec3 c = clamp(v * 1.5 - 0.2, 0.0, 1.0); o = vec4(mix(n, c, s), pow(v.y + 0.01, 2.2) + sqrt(v.z) * 0.1 + fract(v.x * 3.7) * 0.05 + length(v) * 0.01 + exp2(v.x) * 0.01); }
