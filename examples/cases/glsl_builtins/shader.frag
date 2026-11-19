#version 300 es
precision highp float;
precision highp int;
in vec3 v;
out vec4 o;
void main() {
    vec3 a = v * 2.0 - 0.7;
    vec3 b = vec3(v.y, v.z, v.x) + 0.2;
    float d = dot(a, b) + length(a) + distance(a, b);
    vec3 c = clamp(a, -0.3, 0.6) + mix(a, b, v.z) + cross(a, b) * 0.5 + reflect(a, normalize(b));
    float e = exp(v.x) + log(v.y + 0.5) + pow(v.z + 0.1, 1.7) + inversesqrt(v.x + 0.3) + exp2(v.y) + log2(v.z + 1.0);
    float t = atan(a.y, a.x) + atan(a.z) + sin(a.x * 3.0) + cos(a.y * 2.0) + tan(a.z * 0.5);
    float s = step(0.4, v.x) + sign(a.y) + fract(a.z * 4.3) + smoothstep(0.1, 0.9, v.y) + mod(a.x, 0.3);
    o = vec4(c * 0.1 + d * 0.01, (e + t + s) * 0.05);
}
