#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() { o = vec4(mod(v * 7.3, 1.7), 1.0); }
