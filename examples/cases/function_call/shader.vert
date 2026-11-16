#version 300 es
in vec2 pos;
in vec3 color;
out vec3 v;
void main() { v = color; gl_Position = vec4(pos, 0.0, 1.0); }
