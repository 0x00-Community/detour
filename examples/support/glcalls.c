// Runs GL ES 3.0 calls on the local driver (EGL, no window) and prints, per call, glGetError and what it returns.
// Calls come on stdin, one a line, every argument a number: `NAME ARG...`, where an argument is a GLenum or int,
// or D (client data: a pattern of DATA bytes), N (a null pointer) or O<n> (pointer value n: an offset into a bound
// buffer). Output, one line per call: `ERROR [VALUE...]`.
//
//   cc -o glcalls glcalls.c -lEGL -lGLESv2
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA (1 << 20)
#define ARGS 16
#define NAMES 64
#define LINE 4096
#define DATA_ARG 'D'
#define NULL_ARG 'N'
#define OFFSET_ARG 'O'

struct args {
    long v[ARGS];
    const void *p[ARGS];
    int n;
};

static void die(const char *what) {
    fprintf(stderr, "glcalls: %s\n", what);
    exit(1);
}

static void context(void) {
    EGLDisplay d = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (!eglInitialize(d, NULL, NULL)) die("eglInitialize");
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE};
    EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attrs);
    if (c == EGL_NO_CONTEXT) die("eglCreateContext");
    if (!eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) die("eglMakeCurrent");
}

static void parse(char *line, char **name, struct args *a, const unsigned char *data) {
    *name = strtok(line, " \t\n");
    a->n = 0;
    for (char *t = strtok(NULL, " \t\n"); t && a->n < ARGS; t = strtok(NULL, " \t\n"), a->n++) {
        a->p[a->n] = NULL;
        a->v[a->n] = 0;
        if (t[0] == DATA_ARG) a->p[a->n] = data;
        else if (t[0] == NULL_ARG) a->p[a->n] = NULL;
        else if (t[0] == OFFSET_ARG) a->p[a->n] = (const void *)(uintptr_t)strtoul(t + 1, NULL, 10);
        else a->v[a->n] = strtol(t, NULL, 0);
    }
}

// One call; prints its results (after the error, which the caller prints first)
static void call(const char *f, const struct args *a, char *out) {
    const long *v = a->v;
    GLuint names[NAMES];
    GLint r = 0;
    out[0] = 0;
    if (!strcmp(f, "gen_textures") || !strcmp(f, "gen_buffers") || !strcmp(f, "gen_framebuffers") || !strcmp(f, "gen_renderbuffers")) {
        GLsizei n = (GLsizei)v[0];
        if (!strcmp(f, "gen_textures")) glGenTextures(n, names);
        if (!strcmp(f, "gen_buffers")) glGenBuffers(n, names);
        if (!strcmp(f, "gen_framebuffers")) glGenFramebuffers(n, names);
        if (!strcmp(f, "gen_renderbuffers")) glGenRenderbuffers(n, names);
        for (GLsizei i = 0; i < n && i < NAMES; i++) sprintf(out + strlen(out), " %u", names[i]);
    } else if (!strcmp(f, "bind_texture")) glBindTexture(v[0], v[1]);
    else if (!strcmp(f, "delete_textures")) { names[0] = v[0]; glDeleteTextures(1, names); }
    else if (!strcmp(f, "is_texture")) sprintf(out, " %d", glIsTexture(v[0]));
    else if (!strcmp(f, "active_texture")) glActiveTexture(v[0]);
    else if (!strcmp(f, "tex_image_2d")) glTexImage2D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], a->p[8]);
    else if (!strcmp(f, "tex_image_3d")) glTexImage3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], a->p[9]);
    else if (!strcmp(f, "tex_sub_image_2d")) glTexSubImage2D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], a->p[8]);
    else if (!strcmp(f, "tex_sub_image_3d")) glTexSubImage3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], a->p[10]);
    else if (!strcmp(f, "tex_storage_2d")) glTexStorage2D(v[0], v[1], v[2], v[3], v[4]);
    else if (!strcmp(f, "tex_storage_3d")) glTexStorage3D(v[0], v[1], v[2], v[3], v[4], v[5]);
    else if (!strcmp(f, "tex_parameteri")) glTexParameteri(v[0], v[1], v[2]);
    else if (!strcmp(f, "get_tex_parameter")) { glGetTexParameteriv(v[0], v[1], &r); sprintf(out, " %d", r); }
    else if (!strcmp(f, "pixel_storei")) glPixelStorei(v[0], v[1]);
    else if (!strcmp(f, "bind_buffer")) glBindBuffer(v[0], v[1]);
    else if (!strcmp(f, "buffer_data")) glBufferData(v[0], v[1], a->p[2], v[3]);
    else if (!strcmp(f, "bind_renderbuffer")) glBindRenderbuffer(v[0], v[1]);
    else if (!strcmp(f, "delete_renderbuffers")) { names[0] = v[0]; glDeleteRenderbuffers(1, names); }
    else if (!strcmp(f, "renderbuffer_storage")) glRenderbufferStorage(v[0], v[1], v[2], v[3]);
    else if (!strcmp(f, "renderbuffer_storage_multisample")) glRenderbufferStorageMultisample(v[0], v[1], v[2], v[3], v[4]);
    else if (!strcmp(f, "get_renderbuffer_parameter")) { glGetRenderbufferParameteriv(v[0], v[1], &r); sprintf(out, " %d", r); }
    else if (!strcmp(f, "bind_framebuffer")) glBindFramebuffer(v[0], v[1]);
    else if (!strcmp(f, "delete_framebuffers")) { names[0] = v[0]; glDeleteFramebuffers(1, names); }
    else if (!strcmp(f, "framebuffer_texture_2d")) glFramebufferTexture2D(v[0], v[1], v[2], v[3], v[4]);
    else if (!strcmp(f, "framebuffer_texture_layer")) glFramebufferTextureLayer(v[0], v[1], v[2], v[3], v[4]);
    else if (!strcmp(f, "framebuffer_renderbuffer")) glFramebufferRenderbuffer(v[0], v[1], v[2], v[3]);
    else if (!strcmp(f, "check_framebuffer_status")) sprintf(out, " %u", glCheckFramebufferStatus(v[0]));
    else if (!strcmp(f, "get_framebuffer_attachment_parameter")) { glGetFramebufferAttachmentParameteriv(v[0], v[1], v[2], &r); sprintf(out, " %d", r); }
    else if (!strcmp(f, "draw_buffers")) {
        GLenum bufs[ARGS];
        for (int i = 0; i < a->n; i++) bufs[i] = v[i];
        glDrawBuffers(a->n, bufs);
    } else if (!strcmp(f, "read_buffer")) glReadBuffer(v[0]);
    else die(f);
}

int main(void) {
    context();
    unsigned char *data = malloc(DATA);
    for (int i = 0; i < DATA; i++) data[i] = (unsigned char)i;
    char line[LINE], out[LINE];
    while (fgets(line, sizeof line, stdin)) {
        char *name;
        struct args a;
        parse(line, &name, &a, data);
        if (!name) continue;
        call(name, &a, out);
        printf("%u%s\n", glGetError(), out);
    }
    free(data);
    return 0;
}
