#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() { bool a = v.x > 0.3 || v.y > 0.6; float f = a ? v.z : 1.0 - v.z; o = vec4(f, v.y, v.x, 1.0); }
