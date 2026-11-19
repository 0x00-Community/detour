/* The draws of examples/correct's raster cases on Mesa's GL ES (iris on Xe3), its pixels to stdout: what the same
 * GPU gives driven by Mesa. One case on stdin, whitespace separated:
 *   target W H FORMAT(0 rgba8, 1 rgba32f)  pixels N BYTE...  (the target's pixels before the draws, linear rows)
 *   draw NVERTS X Y Z W ...  color R G B A  blend ON SRC_RGB DST_RGB SRC_A DST_A EQ_RGB EQ_A CR CG CB CA
 *        mask R G B A  scissor ON X Y W H  viewport X Y W H  cull MODE(0 none, 1 front, 2 back) CCW
 *        prim MODE(0 triangles, 1 strip) FIRST COUNT INSTANCES  step X Y Z W
 *   end
 * Factors and equations are GL enums. Dithering is off (GLREF_DITHER set: on, GL's default).
 *
 *   cc -O1 -o glref examples/support/glref.c -lEGL -lGLESv2
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
    fprintf(stderr, "glref: %s (GL error 0x%x)\n", what, glGetError());
    exit(1);
}

static void expect(const char *word) {
    char w[32];
    if (scanf("%31s", w) != 1 || strcmp(w, word) != 0) {
        fprintf(stderr, "glref: expected %s, got %s\n", word, w);
        exit(1);
    }
}

static GLuint shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "%s\n", log);
        fail("compiling");
    }
    return s;
}

static const char *VS = "#version 300 es\nin vec4 a_pos;\nuniform vec4 step;\nvoid main() { gl_Position = a_pos + step * float(gl_InstanceID); }\n";
static const char *FS = "#version 300 es\nprecision highp float;\nuniform vec4 color;\nout vec4 o;\nvoid main() { o = color; }\n";

int main(void) {
    EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (!eglInitialize(dpy, NULL, NULL)) fail("eglInitialize");
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint cfg_attrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n;
    eglChooseConfig(dpy, cfg_attrs, &cfg, 1, &n);
    EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, n ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attrs);
    if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) fail("context");

    unsigned w, h, format;
    expect("target");
    if (scanf("%u %u %u", &w, &h, &format) != 3) fail("target");
    unsigned bpp = format ? 16 : 4;
    unsigned char *pixels = malloc((size_t)w * h * bpp);
    unsigned npix;
    expect("pixels");
    if (scanf("%u", &npix) != 1 || npix != w * h * bpp) fail("pixels");
    for (unsigned i = 0; i < npix; i++) {
        unsigned b;
        if (scanf("%u", &b) != 1) fail("a pixel byte");
        pixels[i] = (unsigned char)b;
    }
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    if (format)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, pixels);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fail("framebuffer");
    if (format) {
        /* float targets aren't renderable without EXT_color_buffer_float: it is on iris */
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, shader(GL_VERTEX_SHADER, VS));
    glAttachShader(p, shader(GL_FRAGMENT_SHADER, FS));
    glBindAttribLocation(p, 0, "a_pos");
    glLinkProgram(p);
    glUseProgram(p);
    GLint ucolor = glGetUniformLocation(p, "color"), ustep = glGetUniformLocation(p, "step");
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    if (getenv("GLREF_DITHER")) glEnable(GL_DITHER); else glDisable(GL_DITHER);

    char word[32];
    while (scanf("%31s", word) == 1 && strcmp(word, "end") != 0) {
        if (strcmp(word, "draw") != 0) fail("expected draw");
        unsigned nv;
        if (scanf("%u", &nv) != 1) fail("vertices");
        float *v = malloc(sizeof(float) * 4 * (nv ? nv : 1));
        for (unsigned i = 0; i < 4 * nv; i++)
            if (scanf("%f", &v[i]) != 1) fail("a vertex");
        float c[4], bc[4], step[4], vp[4];
        unsigned blend, f[6], mask[4], sc, s[4], cull, ccw, prim, first, count, inst;
        expect("color");
        if (scanf("%f %f %f %f", &c[0], &c[1], &c[2], &c[3]) != 4) fail("color");
        expect("blend");
        if (scanf("%u %u %u %u %u %u %u %f %f %f %f", &blend, &f[0], &f[1], &f[2], &f[3], &f[4], &f[5], &bc[0], &bc[1], &bc[2], &bc[3]) != 11) fail("blend");
        expect("mask");
        if (scanf("%u %u %u %u", &mask[0], &mask[1], &mask[2], &mask[3]) != 4) fail("mask");
        expect("scissor");
        if (scanf("%u %u %u %u %u", &sc, &s[0], &s[1], &s[2], &s[3]) != 5) fail("scissor");
        expect("viewport");
        if (scanf("%f %f %f %f", &vp[0], &vp[1], &vp[2], &vp[3]) != 4) fail("viewport");
        expect("cull");
        if (scanf("%u %u", &cull, &ccw) != 2) fail("cull");
        expect("prim");
        if (scanf("%u %u %u %u", &prim, &first, &count, &inst) != 4) fail("prim");
        expect("step");
        if (scanf("%f %f %f %f", &step[0], &step[1], &step[2], &step[3]) != 4) fail("step");
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * nv, v, GL_STREAM_DRAW);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, 0);
        glUniform4fv(ucolor, 1, c);
        glUniform4fv(ustep, 1, step);
        if (blend) {
            glEnable(GL_BLEND);
            glBlendFuncSeparate(f[0], f[1], f[2], f[3]);
            glBlendEquationSeparate(f[4], f[5]);
            glBlendColor(bc[0], bc[1], bc[2], bc[3]);
        } else {
            glDisable(GL_BLEND);
        }
        glColorMask(mask[0], mask[1], mask[2], mask[3]);
        if (sc) {
            glEnable(GL_SCISSOR_TEST);
            glScissor(s[0], s[1], s[2], s[3]);
        } else {
            glDisable(GL_SCISSOR_TEST);
        }
        glViewport((GLint)vp[0], (GLint)vp[1], (GLsizei)vp[2], (GLsizei)vp[3]);
        if (cull) {
            glEnable(GL_CULL_FACE);
            glCullFace(cull == 1 ? GL_FRONT : GL_BACK);
        } else {
            glDisable(GL_CULL_FACE);
        }
        glFrontFace(ccw ? GL_CCW : GL_CW);
        glDrawArraysInstanced(prim ? GL_TRIANGLE_STRIP : GL_TRIANGLES, first, count, inst);
        free(v);
    }
    if (glGetError() != GL_NO_ERROR) fail("drawing");
    glReadPixels(0, 0, w, h, GL_RGBA, format ? GL_FLOAT : GL_UNSIGNED_BYTE, pixels);
    fwrite(pixels, 1, (size_t)w * h * bpp, stdout);
    return 0;
}
