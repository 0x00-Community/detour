#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
float f(float x) { return x * x + 0.5 * x; }
vec3 g(vec3 a, float k) { return a * k + vec3(f(a.x)); }
void main() { o = vec4(g(v, 0.3), f(v.z)); }
