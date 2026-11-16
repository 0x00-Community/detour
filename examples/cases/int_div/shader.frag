#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() { ivec3 i = ivec3(v * 100.0) - 33; ivec3 q = i / 7; ivec3 r = i % 7; o = vec4(vec3(q) / 20.0, float(r.x + r.y + r.z) / 21.0); }
