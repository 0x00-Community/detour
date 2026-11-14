// IOKit calls of a process (DYLD_INSERT_LIBRARIES; scripts/apple-iogpu-trace.sh): what Metal's driver asks the kernel
// (AGXG16G, IOGPU) for, and at each submission the memory the GPU is about to read. docs/roadmap.md "M3 运行时".
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define INTERPOSE(n, o) __attribute__((used)) static struct { const void *r, *e; } _i_##o __attribute__((section("__DATA,__interpose"))) = {(const void *)n, (const void *)o};

static FILE *out(void) {
    static FILE *f;
    if (!f) { const char *p = getenv("IOTRACE"); char n[1024]; if (p) snprintf(n, sizeof n, "%s.%d", p, getpid()); f = p ? fopen(n, "w") : stderr; setvbuf(f, NULL, _IOLBF, 0); }
    return f;
}

static void hex(const void *p, size_t n) {
    const unsigned char *b = p;
    size_t m = n > 256 ? 256 : n;
    for (size_t i = 0; i < m; i++) fprintf(out(), "%02x", b[i]);
    if (m < n) fprintf(out(), "...(+%zu)", n - m);
}

static kern_return_t my_IOServiceOpen(io_service_t s, task_port_t t, uint32_t type, io_connect_t *c) {
    kern_return_t r = IOServiceOpen(s, t, type, c);
    io_name_t cls = "";
    IOObjectGetClass(s, cls);
    fprintf(out(), "open %s type=%u -> conn=%u kr=%x\n", cls, type, c ? *c : 0, r);
    return r;
}
INTERPOSE(my_IOServiceOpen, IOServiceOpen)

struct alloc { uint64_t gpu, cpu, size; uint32_t id; };
static struct alloc allocs[4096];
static int nallocs;

static void dump_allocs(FILE *f) {
    for (int k = 0; k < nallocs; k++) {
        struct alloc *a = &allocs[k];
        fprintf(f, "alloc id=%u gpu=%llx cpu=%llx size=%llx\n", a->id, a->gpu, a->cpu, a->size);
        if (!a->cpu) continue;
        uint64_t n = a->size > 0x40000 ? 0x40000 : a->size;
        unsigned char b[64];
        for (uint64_t i = 0; i < n; i += 64) {
            mach_vm_size_t got = 0;
            if (mach_vm_read_overwrite(mach_task_self(), a->cpu + i, 64, (mach_vm_address_t)b, &got) != KERN_SUCCESS || got != 64) {
                fprintf(f, "  %llx: unreadable\n", a->gpu + i);
                break;
            }
            int z = 1;
            for (int j = 0; j < 64; j++) if (b[j]) { z = 0; break; }
            if (z) continue;
            fprintf(f, "  %llx:", a->gpu + i);
            for (int j = 0; j < 64; j++) fprintf(f, "%02x", b[j]);
            fprintf(f, "\n");
        }
    }
}

static kern_return_t my_IOConnectCallMethod(mach_port_t c, uint32_t sel, const uint64_t *in, uint32_t inN, const void *is, size_t isN, uint64_t *o, uint32_t *oN, void *os, size_t *osN) {
    uint32_t oN0 = oN ? *oN : 0; size_t osN0 = osN ? *osN : 0;
    kern_return_t r = IOConnectCallMethod(c, sel, in, inN, is, isN, o, oN, os, osN);
    if (sel == 10 && r == 0 && os && osN && *osN >= 40 && nallocs < 4096) {
        const uint64_t *q = os;
        allocs[nallocs++] = (struct alloc){q[0], q[1], q[4], (uint32_t)(q[3] >> 32)};
    }
    // sel=15: a shared page of the kernel's (the submission rings?): cpu, size, id; as an alloc at "gpu" 0xf15<id>0000
    if (sel == 15 && r == 0 && os && osN && *osN >= 16 && nallocs < 4096) {
        const uint64_t *q = os;
        allocs[nallocs++] = (struct alloc){0xf1500000000ull + ((q[1] >> 32) << 24), q[0], q[1] & 0xffffffff, 0x10000 + (uint32_t)(q[1] >> 32)};
    }
    fprintf(out(), "call conn=%u sel=%u in=[", c, sel);
    for (uint32_t i = 0; i < inN; i++) fprintf(out(), "%s%llx", i ? "," : "", in[i]);
    fprintf(out(), "] is=%zu:", isN); if (is) hex(is, isN);
    fprintf(out(), " out=%u/%u:[", oN ? *oN : 0, oN0);
    for (uint32_t i = 0; o && oN && i < *oN; i++) fprintf(out(), "%s%llx", i ? "," : "", o[i]);
    fprintf(out(), "] os=%zu/%zu:", osN ? *osN : 0, osN0); if (os && osN) hex(os, *osN);
    fprintf(out(), " kr=%x\n", r);
    return r;
}
INTERPOSE(my_IOConnectCallMethod, IOConnectCallMethod)

static kern_return_t my_IOConnectCallAsyncMethod(mach_port_t c, uint32_t sel, mach_port_t wake, uint64_t *ref, uint32_t refN, const uint64_t *in, uint32_t inN, const void *is, size_t isN, uint64_t *o, uint32_t *oN, void *os, size_t *osN) {
    kern_return_t r = IOConnectCallAsyncMethod(c, sel, wake, ref, refN, in, inN, is, isN, o, oN, os, osN);
    fprintf(out(), "async conn=%u sel=%u refN=%u in=[", c, sel, refN);
    for (uint32_t i = 0; i < inN; i++) fprintf(out(), "%s%llx", i ? "," : "", in[i]);
    fprintf(out(), "] is=%zu:", isN); if (is) hex(is, isN);
    fprintf(out(), " os=%zu:", osN ? *osN : 0); if (os && osN) hex(os, *osN);
    fprintf(out(), " kr=%x\n", r);
    return r;
}
INTERPOSE(my_IOConnectCallAsyncMethod, IOConnectCallAsyncMethod)

static kern_return_t my_IOConnectCallStructMethod(mach_port_t c, uint32_t sel, const void *is, size_t isN, void *os, size_t *osN) {
    kern_return_t r = IOConnectCallStructMethod(c, sel, is, isN, os, osN);
    fprintf(out(), "struct conn=%u sel=%u is=%zu:", c, sel, isN); if (is) hex(is, isN);
    fprintf(out(), " os=%zu:", osN ? *osN : 0); if (os && osN) hex(os, *osN);
    fprintf(out(), " kr=%x\n", r);
    return r;
}
INTERPOSE(my_IOConnectCallStructMethod, IOConnectCallStructMethod)

static kern_return_t my_IOConnectCallScalarMethod(mach_port_t c, uint32_t sel, const uint64_t *in, uint32_t inN, uint64_t *o, uint32_t *oN) {
    kern_return_t r = IOConnectCallScalarMethod(c, sel, in, inN, o, oN);
    fprintf(out(), "scalar conn=%u sel=%u in=[", c, sel);
    for (uint32_t i = 0; i < inN; i++) fprintf(out(), "%s%llx", i ? "," : "", in[i]);
    fprintf(out(), "] out=[");
    for (uint32_t i = 0; o && oN && i < *oN; i++) fprintf(out(), "%s%llx", i ? "," : "", o[i]);
    fprintf(out(), "] kr=%x\n", r);
    return r;
}
INTERPOSE(my_IOConnectCallScalarMethod, IOConnectCallScalarMethod)

static kern_return_t my_IOConnectMapMemory64(io_connect_t c, uint32_t type, task_port_t t, mach_vm_address_t *a, mach_vm_size_t *s, IOOptionBits o) {
    kern_return_t r = IOConnectMapMemory64(c, type, t, a, s, o);
    fprintf(out(), "map conn=%u type=%u opts=%x -> addr=%llx size=%llx kr=%x\n", c, type, o, a ? *a : 0, s ? *s : 0, r);
    return r;
}
INTERPOSE(my_IOConnectMapMemory64, IOConnectMapMemory64)

static kern_return_t my_IOConnectSetNotificationPort(io_connect_t c, uint32_t type, mach_port_t p, uintptr_t ref) {
    kern_return_t r = IOConnectSetNotificationPort(c, type, p, ref);
    fprintf(out(), "notify conn=%u type=%u port=%u kr=%x\n", c, type, p, r);
    return r;
}
INTERPOSE(my_IOConnectSetNotificationPort, IOConnectSetNotificationPort)

kern_return_t IOConnectTrap6(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b, uintptr_t cc, uintptr_t d, uintptr_t e, uintptr_t f);
static kern_return_t my_IOConnectTrap6(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b, uintptr_t cc, uintptr_t d, uintptr_t e, uintptr_t f) {
    kern_return_t r = IOConnectTrap6(c, i, a, b, cc, d, e, f);
    fprintf(out(), "trap6 conn=%u idx=%u %lx %lx %lx %lx %lx %lx kr=%x\n", c, i, a, b, cc, d, e, f, r);
    return r;
}
INTERPOSE(my_IOConnectTrap6, IOConnectTrap6)
kern_return_t IOConnectTrap4(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b, uintptr_t cc, uintptr_t d);
static kern_return_t my_IOConnectTrap4(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b, uintptr_t cc, uintptr_t d) {
    kern_return_t r = IOConnectTrap4(c, i, a, b, cc, d);
    fprintf(out(), "trap4 conn=%u idx=%u %lx %lx %lx %lx kr=%x\n", c, i, a, b, cc, d, r);
    return r;
}
INTERPOSE(my_IOConnectTrap4, IOConnectTrap4)
kern_return_t IOConnectTrap2(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b);
static kern_return_t my_IOConnectTrap2(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b) {
    kern_return_t r = IOConnectTrap2(c, i, a, b);
    fprintf(out(), "trap2 conn=%u idx=%u %lx %lx kr=%x\n", c, i, a, b, r);
    return r;
}
INTERPOSE(my_IOConnectTrap2, IOConnectTrap2)

kern_return_t IOConnectTrap1(io_connect_t c, uint32_t i, uintptr_t a);
static kern_return_t my_IOConnectTrap1(io_connect_t c, uint32_t i, uintptr_t a) {
    kern_return_t r = IOConnectTrap1(c, i, a);
    fprintf(out(), "trap1 conn=%u idx=%u %lx kr=%x\n", c, i, a, r);
    return r;
}
INTERPOSE(my_IOConnectTrap1, IOConnectTrap1)
kern_return_t IOConnectTrap3(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b, uintptr_t cc);
static kern_return_t my_IOConnectTrap3(io_connect_t c, uint32_t i, uintptr_t a, uintptr_t b, uintptr_t cc) {
    static int submits;
    if (i == 0 && cc && b <= 4096) {
        char n[1024];
        snprintf(n, sizeof n, "%s.submit.%d.%d", getenv("IOTRACE") ? getenv("IOTRACE") : "/tmp/iotrace", getpid(), submits++);
        FILE *f = fopen(n, "w");
        if (f) {
            fprintf(f, "submit queue=%lx size=%lx struct:", a, b);
            for (uintptr_t k = 0; k < b; k++) fprintf(f, "%02x", ((const unsigned char *)cc)[k]);
            fprintf(f, "\n");
            // what the struct's pointers (user memory) hold
            for (uintptr_t k = 0; k + 8 <= b; k += 8) {
                uint64_t p = *(const uint64_t *)(cc + k);
                if (p < 0x100000000ull || p > 0x800000000ull) continue;
                int inside = 0;
                for (int j = 0; j < nallocs; j++) if (p >= allocs[j].cpu && p < allocs[j].cpu + allocs[j].size) inside = 1;
                unsigned char buf[0x200];
                mach_vm_size_t got = 0;
                if (mach_vm_read_overwrite(mach_task_self(), p, sizeof buf, (mach_vm_address_t)buf, &got) != KERN_SUCCESS) {
                    got = 0;
                    for (mach_vm_size_t step = 0x40; step <= sizeof buf; step += 0x40) {
                        mach_vm_size_t g = 0;
                        if (mach_vm_read_overwrite(mach_task_self(), p, step, (mach_vm_address_t)buf, &g) != KERN_SUCCESS) break;
                        got = g;
                    }
                }
                fprintf(f, "ptr@%lx %llx%s:", k, p, inside ? " (in an alloc)" : "");
                for (mach_vm_size_t j = 0; j < got; j++) fprintf(f, "%02x", buf[j]);
                fprintf(f, "\n");
            }
            dump_allocs(f);
            fclose(f);
        }
    }
    kern_return_t r = IOConnectTrap3(c, i, a, b, cc);
    fprintf(out(), "trap3 conn=%u idx=%u %lx %lx %lx kr=%x\n", c, i, a, b, cc, r);
    return r;
}
INTERPOSE(my_IOConnectTrap3, IOConnectTrap3)
static kern_return_t my_IOConnectCallAsyncScalarMethod(mach_port_t c, uint32_t sel, mach_port_t wake, uint64_t *ref, uint32_t refN, const uint64_t *in, uint32_t inN, uint64_t *o, uint32_t *oN) {
    kern_return_t r = IOConnectCallAsyncScalarMethod(c, sel, wake, ref, refN, in, inN, o, oN);
    fprintf(out(), "ascalar conn=%u sel=%u in=[", c, sel);
    for (uint32_t i = 0; i < inN; i++) fprintf(out(), "%s%llx", i ? "," : "", in[i]);
    fprintf(out(), "] kr=%x\n", r);
    return r;
}
INTERPOSE(my_IOConnectCallAsyncScalarMethod, IOConnectCallAsyncScalarMethod)
