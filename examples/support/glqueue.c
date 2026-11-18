// The bench example's draw on the local Mesa: the triangle case into a W x H RGBA8 renderbuffer, N times, timed.
//   cc -O2 -o glqueue examples/support/glqueue.c -lEGL -lGLESv2
//   ./glqueue N W H
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const char *vert = "#version 300 es\nin vec2 pos;\nin vec3 color;\nout vec3 v;\nvoid main() { v = color; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *frag = "#version 300 es\nprecision highp float;\nin vec3 v;\nout vec4 o;\nvoid main() { o = vec4(v, 1.0); }\n";

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static GLuint shader(GLenum ty, const char *src) {
    GLuint s = glCreateShader(ty);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 5000, w = argc > 2 ? atoi(argv[2]) : 1920, h = argc > 3 ? atoi(argv[3]) : 1080;
    EGLDisplay d = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    eglInitialize(d, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
    EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
    eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c);
    GLuint rb, fb;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    GLuint p = glCreateProgram();
    glAttachShader(p, shader(GL_VERTEX_SHADER, vert));
    glAttachShader(p, shader(GL_FRAGMENT_SHADER, frag));
    glBindAttribLocation(p, 0, "pos");
    glBindAttribLocation(p, 1, "color");
    glLinkProgram(p);
    glUseProgram(p);
    float pos[] = {-0.8f, -0.7f, 0.9f, -0.4f, -0.1f, 0.85f}, color[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    GLuint b[2];
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof color, color, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 50; i++)
        glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    double t = now();
    for (int i = 0; i < n; i++)
        glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    double q = now() - t;
    t = now();
    for (int i = 0; i < n; i++) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
    }
    double f = now() - t;
    unsigned char px[4];
    glReadPixels(w / 2, h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("mesa %dx%d: queued %.4f ms/draw (%.0f draws/s); draw + finish %.4f ms; center [%d, %d, %d, %d] (%s)\n", w, h, q / n, n * 1e3 / q, f / n, px[0], px[1], px[2], px[3], glGetString(GL_RENDERER));
    return 0;
}
