#version 300 es
precision highp float;
in vec3 v;
out vec4 o;
uniform vec4 tint;
uniform bool flip;
void main() { o = tint; if (flip) { o = vec4(v, 1.0) * tint; } }
