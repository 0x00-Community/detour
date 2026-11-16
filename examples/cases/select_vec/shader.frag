#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() { o = v.x > 0.5 ? vec4(v, 1.0) : vec4(1.0 - v, 1.0); }
