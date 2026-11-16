#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() { vec4 c = vec4(0.25); c.x = v.y; c.z = v.x * v.z; o = c; }
