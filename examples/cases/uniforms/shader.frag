#version 300 es
precision highp float;
in vec3 v;
out vec4 o;
void main() { o = vec4(v, 1.0); }
