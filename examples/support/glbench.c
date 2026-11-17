// Times one case on the local OpenGL ES (Mesa, EGL, no window): the shaders compiled and linked, then FRAMES frames of
// clear + one triangle into an RGBA32F target, each waited for (glFinish). Prints, in nanoseconds:
// `COMPILE FIRST MEAN MIN` (FIRST: the first frame, MEAN / MIN: the frames after it).
//
//   cc -O2 -o glbench glbench.c -lEGL -lGLESv2
//   glbench VERT FRAG RES W H FRAMES
//
// RES is a case's `resources`: `attribute NAME LOCATION N V...` (three vertices), `uniform NAME LOCATION f|i V...`.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VERTICES 3
#define CHANNELS 4
#define MAX_VALUES 256
#define MAT4 16
#define NS 1000000000ull
#define LOG 4096

static void die(const char *what) {
    fprintf(stderr, "glbench: %s\n", what);
    exit(1);
}

static uint64_t now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * NS + (uint64_t)t.tv_nsec;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die(path);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    if (fread(buf, 1, size, f) != (size_t)size) die(path);
    buf[size] = 0;
    fclose(f);
    return buf;
}

static GLuint shader(GLenum kind, const char *path) {
    char *src = slurp(path);
    const char *text = src;
    GLuint s = glCreateShader(kind);
    glShaderSource(s, 1, &text, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[LOG];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "glbench: %s: %s\n", path, log);
        exit(1);
    }
    free(src);
    return s;
}

static int values(FILE *f, int ints, float *fv, int *iv) {
    int n = 0;
    for (;;) {
        int c = fgetc(f);
        while (c == ' ' || c == '\t') c = fgetc(f);
        if (c == '\n' || c == EOF) return n;
        ungetc(c, f);
        if (n == MAX_VALUES) die("too many values");
        if (ints ? fscanf(f, "%d", &iv[n]) != 1 : fscanf(f, "%f", &fv[n]) != 1) die("a value");
        n++;
    }
}

static void resources(const char *path, GLuint prog) {
    FILE *f = fopen(path, "r");
    if (!f) die(path);
    char kind[16], name[64], ty[2];
    float fv[MAX_VALUES];
    int iv[MAX_VALUES], at, size;
    while (fscanf(f, "%15s", kind) == 1) {
        if (strcmp(kind, "attribute") == 0) {
            if (fscanf(f, "%63s %d %d", name, &at, &size) != 3) die("attribute NAME LOCATION N V...");
            int n = values(f, 0, fv, iv);
            GLint loc = glGetAttribLocation(prog, name);
            if (loc < 0) continue;
            GLuint b;
            glGenBuffers(1, &b);
            glBindBuffer(GL_ARRAY_BUFFER, b);
            glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), fv, GL_STATIC_DRAW);
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(loc);
            continue;
        }
        if (strcmp(kind, "uniform") != 0) die(kind);
        if (fscanf(f, "%63s %d %1s", name, &at, ty) != 3) die("uniform NAME LOCATION f|i V...");
        int n = values(f, ty[0] == 'i', fv, iv);
        GLint loc = glGetUniformLocation(prog, name);
        if (ty[0] == 'i') {
            void (*set[])(GLint, GLsizei, const GLint *) = {glUniform1iv, glUniform2iv, glUniform3iv, glUniform4iv};
            if (n < 1 || n > CHANNELS) die("1 to 4 ints");
            set[n - 1](loc, 1, iv);
        } else if (n == MAT4) {
            glUniformMatrix4fv(loc, 1, GL_FALSE, fv);
        } else {
            void (*set[])(GLint, GLsizei, const GLfloat *) = {glUniform1fv, glUniform2fv, glUniform3fv, glUniform4fv};
            if (n < 1 || n > CHANNELS) die("1 to 4 floats, or 16");
            set[n - 1](loc, 1, fv);
        }
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 7) die("usage: glbench VERT FRAG RES W H FRAMES");
    int w = atoi(argv[4]), h = atoi(argv[5]), frames = atoi(argv[6]);
    if (frames < 2) die("at least two frames");

    EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (!eglInitialize(dpy, NULL, NULL)) die("eglInitialize");
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attrs);
    if (ctx == EGL_NO_CONTEXT) die("eglCreateContext");
    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) die("eglMakeCurrent");

    uint64_t t0 = now();
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader(GL_VERTEX_SHADER, argv[1]));
    glAttachShader(prog, shader(GL_FRAGMENT_SHADER, argv[2]));
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[LOG];
        glGetProgramInfoLog(prog, sizeof log, NULL, log);
        fprintf(stderr, "glbench: link: %s\n", log);
        return 1;
    }
    glUseProgram(prog);
    uint64_t compile = now() - t0;

    GLuint vao, tex, fbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    resources(argv[3], prog);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("framebuffer");
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 0);

    uint64_t first = 0, sum = 0, min = UINT64_MAX;
    for (int i = 0; i < frames; i++) {
        uint64_t t = now();
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, VERTICES);
        glFinish();
        uint64_t d = now() - t;
        if (i == 0) {
            first = d;
            continue;
        }
        sum += d;
        if (d < min) min = d;
    }
    if (glGetError() != GL_NO_ERROR) die("a GL error");
    printf("%llu %llu %llu %llu\n", (unsigned long long)compile, (unsigned long long)first, (unsigned long long)(sum / (frames - 1)), (unsigned long long)min);
    return 0;
}
