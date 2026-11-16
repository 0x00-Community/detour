#version 300 es
in vec2 pos;
in vec3 color;
out vec3 v;
uniform vec2 offset;
uniform float scale;
uniform vec3 tint;
void main() { v = color * scale * tint; gl_Position = vec4(pos * scale + offset, 0.0, 1.0); }
