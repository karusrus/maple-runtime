// gpu_micro3.m — the same layer of Maple experts as gpu_micro2.m, sewn into two dispatches instead of five:
// each threadgroup quantises its own input into bit planes in threadgroup memory, then computes its rows.
// Weights: per row and 128-column group a "plus" and a "minus" mask (2 x uint4). Inputs: per 128-value group an int8
// scale and 8 bit planes. dot = sum over planes of +-2^p * (popcount(plus & plane) - popcount(minus & plane)).
// One layer = planes(x) -> gate+up for n experts -> silu*up -> planes(act) -> down; checked against fp32 on the CPU.
//   cc -O2 -fobjc-arc -o gpu_micro3 gpu_micro3.m -framework Metal -framework Foundation && ./gpu_micro2
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H 2048
#define FF 512

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    // 64 threads turn n groups of 128 floats (n*4 <= 64) into int8 bit planes in threadgroup memory: thread = (group, 32-value part)
    "static void tg_planes(threadgroup const float *xs, uint n, threadgroup uint *plw, threadgroup float *sx, threadgroup float *gm, uint t) {\n"
    "    uint g = t / 4, k = t % 4; bool on = g < n;\n"
    "    float mx = 0;\n"
    "    if (on) for (uint i = 0; i < 32; i++) mx = max(mx, fabs(xs[g * 128 + k * 32 + i]));\n"
    "    gm[t] = mx;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (on) {\n"
    "        float m4 = max(max(gm[g * 4], gm[g * 4 + 1]), max(gm[g * 4 + 2], gm[g * 4 + 3]));\n"
    "        float d = m4 > 0 ? m4 / 127.0f : 1.0f, r = 1.0f / d; if (k == 0) sx[g] = d;\n"
    "        uint b[8] = {0, 0, 0, 0, 0, 0, 0, 0};\n"
    "        for (uint i = 0; i < 32; i++) {\n"
    "            uint u = uint(int(rint(xs[g * 128 + k * 32 + i] * r))) & 255u;\n"
    "            for (uint p = 0; p < 8; p++) b[p] |= ((u >> p) & 1u) << i;\n"
    "        }\n"
    "        for (uint p = 0; p < 8; p++) plw[(g * 8 + p) * 4 + k] = b[p];\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "}\n"
    "static float row_dot(device const uint4 *m, uint ng, threadgroup const uint *plw, threadgroup const float *sx) {\n"
    "    float acc = 0;\n"
    "    for (uint g = 0; g < ng; g++) {\n"
    "        uint4 ps = m[2 * g], ms = m[2 * g + 1]; int tot = 0;\n"
    "        for (uint q = 0; q < 8; q++) {\n"
    "            uint i = (g * 8 + q) * 4; uint4 v = uint4(plw[i], plw[i + 1], plw[i + 2], plw[i + 3]);\n"
    "            int4 t = int4(popcount(ps & v)) - int4(popcount(ms & v)); int s = t.x + t.y + t.z + t.w;\n"
    "            tot += q == 7 ? -128 * s : (s << q);\n"
    "        }\n"
    "        acc += float(tot) * sx[g];\n"
    "    }\n"
    "    return acc;\n"
    "}\n"
    // dispatch 1: every threadgroup quantises x itself, then 64 gate/up rows, one per thread
    "kernel void gateup(device const uint4 *mask [[buffer(0)]], device const float *scale [[buffer(1)]], device const float *x [[buffer(2)]],\n"
    "                   device float *y [[buffer(3)]], constant uint &rows [[buffer(4)]],\n"
    "                   uint row [[thread_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float xs[2048]; threadgroup uint plw[16 * 8 * 4]; threadgroup float sx[16]; threadgroup float gm[64];\n"
    "    for (uint i = t; i < 2048; i += 64) xs[i] = x[i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    tg_planes(xs, 16, plw, sx, gm, t);\n"
    "    if (row < rows) y[row] = row_dot(mask + (ulong)row * 32, 16, plw, sx) * scale[row];\n"
    "}\n"
    // dispatch 2: every threadgroup builds silu(gate)*up of its expert, quantises it, then 64 down rows
    "kernel void down(device const uint4 *mask [[buffer(0)]], device const float *scale [[buffer(1)]], device const float *gu [[buffer(2)]],\n"
    "                 device float *y [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float as[512]; threadgroup uint plw[4 * 8 * 4]; threadgroup float sx[4]; threadgroup float gm[64];\n"
    "    uint e = tg / 32, row = e * 2048 + (tg % 32) * 64 + t;\n"
    "    for (uint i = t; i < 512; i += 64) {\n"
    "        float g = min(gu[e * 1024 + i], 7.0f), u = clamp(gu[e * 1024 + 512 + i], -7.0f, 7.0f);\n"
    "        as[i] = g / (1.0f + exp(-g)) * u;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    tg_planes(as, 4, plw, sx, gm, t);\n"
    "    y[row] = row_dot(mask + (ulong)row * 8, 4, plw, sx) * scale[row];\n"
    "}\n";

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }

// our packed layout (TQ2 groups) -> plus/minus masks, both in uint4 order per 128-group
static void to_masks(const uint8_t *w, uint32_t *m, size_t rows, int cols) {
    int ng = cols / 128;
    for (size_t r = 0; r < rows; r++) for (int g = 0; g < ng; g++) {
        const uint8_t *q = w + r * cols / 4 + g * 32; uint32_t *o = m + (r * ng + g) * 8;
        for (int k = 0; k < 4; k++) {
            uint32_t plus = 0, minus = 0;
            for (int j = 0; j < 32; j++) { int c = (q[j] >> (2 * k)) & 3; if (c == 2) plus |= 1u << j; if (c == 0) minus |= 1u << j; }
            o[k] = plus; o[4 + k] = minus;
        }
    }
}
static void cpu_tern(const uint8_t *w, const float *x, const float *s, float *y, size_t rows, int cols, size_t rows_per_in) {
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *p = w + r * cols / 4; const float *xv = x + (r / rows_per_in) * cols; double a = 0;
        for (int g = 0; g < cols / 128; g++) for (int j = 0; j < 32; j++) {
            uint8_t b = p[g * 32 + j]; const float *xg = xv + g * 128 + j;
            a += ((b & 3) - 1.0) * xg[0] + (((b >> 2) & 3) - 1.0) * xg[32] + (((b >> 4) & 3) - 1.0) * xg[64] + ((b >> 6) - 1.0) * xg[96];
        }
        y[r] = (float)a * s[r];
    }
}

int main(int argc, char **argv) {
    int NE = argc > 1 ? atoi(argv[1]) : 8;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { NSArray<id<MTLDevice>> *all = MTLCopyAllDevices(); if (all.count) dev = all[0]; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pgu = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gateup"] error:&err];
        id<MTLComputePipelineState> pdn = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"down"] error:&err];
        if (!pgu || !pdn) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        srand(11);
        size_t gu_rows = (size_t)NE * 2 * FF, dn_rows = (size_t)NE * H;
        size_t gu_bytes = gu_rows * H / 4, dn_bytes = dn_rows * FF / 4;
        uint8_t *wgu = malloc(gu_bytes), *wdn = malloc(dn_bytes);
        for (size_t i = 0; i < gu_bytes; i++) { uint8_t b = 0; for (int k = 0; k < 4; k++) b |= (rand() % 3) << (2 * k); wgu[i] = b; }
        for (size_t i = 0; i < dn_bytes; i++) { uint8_t b = 0; for (int k = 0; k < 4; k++) b |= (rand() % 3) << (2 * k); wdn[i] = b; }
        float *sgu = malloc(gu_rows * 4), *sdn = malloc(dn_rows * 4), *x = malloc(H * 4);
        for (size_t i = 0; i < gu_rows; i++) sgu[i] = 0.004f + (rand() % 100) * 4e-5f;
        for (size_t i = 0; i < dn_rows; i++) sdn[i] = 0.004f + (rand() % 100) * 4e-5f;
        for (int i = 0; i < H; i++) x[i] = (rand() % 2001 - 1000) * 1e-3f;
        double t0 = now();
        uint32_t *mgu = malloc(gu_rows * (H / 128) * 32), *mdn = malloc(dn_rows * (FF / 128) * 32);
        to_masks(wgu, mgu, gu_rows, H); to_masks(wdn, mdn, dn_rows, FF);
        double tconv = now() - t0;
        id<MTLBuffer> bmgu = [dev newBufferWithBytes:mgu length:gu_rows * (H / 128) * 32 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bmdn = [dev newBufferWithBytes:mdn length:dn_rows * (FF / 128) * 32 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bsgu = [dev newBufferWithBytes:sgu length:gu_rows * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bsdn = [dev newBufferWithBytes:sdn length:dn_rows * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x length:H * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bplx = [dev newBufferWithLength:(H / 128) * 8 * 16 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bsx = [dev newBufferWithLength:(H / 128) * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bgu = [dev newBufferWithLength:gu_rows * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> ba = [dev newBufferWithLength:(size_t)NE * FF * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bpla = [dev newBufferWithLength:(size_t)NE * (FF / 128) * 8 * 16 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bsa = [dev newBufferWithLength:(size_t)NE * (FF / 128) * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> by = [dev newBufferWithLength:dn_rows * 4 options:MTLResourceStorageModeManaged];
        uint32_t p1[3] = {(uint32_t)gu_rows, H / 128, (uint32_t)gu_rows}, p2[3] = {(uint32_t)dn_rows, FF / 128, H};
        double tk[300], tw[300];
        for (int it = 0; it < 300; it++) {
            double s0 = now();
            memcpy(bx.contents, x, H * 4); [bx didModifyRange:NSMakeRange(0, H * 4)];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            uint32_t nrows = (uint32_t)gu_rows;
            [e setComputePipelineState:pgu]; [e setBuffer:bmgu offset:0 atIndex:0]; [e setBuffer:bsgu offset:0 atIndex:1];
            [e setBuffer:bx offset:0 atIndex:2]; [e setBuffer:bgu offset:0 atIndex:3]; [e setBytes:&nrows length:4 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((gu_rows + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e setComputePipelineState:pdn]; [e setBuffer:bmdn offset:0 atIndex:0]; [e setBuffer:bsdn offset:0 atIndex:1];
            [e setBuffer:bgu offset:0 atIndex:2]; [e setBuffer:by offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((size_t)NE * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:by]; [bl synchronizeResource:bgu]; [bl endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            tw[it] = now() - s0; tk[it] = cb.GPUEndTime - cb.GPUStartTime;
        }
        double kt = median(tk + 20, 280), wt = median(tw + 20, 280);
        double wts = (double)(gu_rows * H + dn_rows * FF);
        printf("%d experts of one layer, fused, 2 dispatches: GPU %.3f ms (%.0f G weights/s), with upload, submit, wait %.3f ms; mask conversion on CPU %.1f ms\n",
               NE, kt * 1e3, wts / kt / 1e9, wt * 1e3, tconv * 1e3);
        printf("per token, 24 layers: %.1f ms\n", 24 * wt * 1e3);
        // fp32 reference for the whole chain
        float *cgu = malloc(gu_rows * 4), *ca = malloc((size_t)NE * FF * 4), *cy = malloc(dn_rows * 4);
        cpu_tern(wgu, x, sgu, cgu, gu_rows, H, gu_rows);
        for (int e = 0; e < NE; e++) for (int k = 0; k < FF; k++) {
            float g = fminf(cgu[e * 1024 + k], 7.0f), u = fmaxf(-7.0f, fminf(cgu[e * 1024 + 512 + k], 7.0f));
            ca[e * FF + k] = g / (1.0f + expf(-g)) * u;
        }
        cpu_tern(wdn, ca, sdn, cy, dn_rows, FF, H);
        const float *g1 = bgu.contents, *gy = by.contents; double m1 = 0, n1 = 0, m2 = 0, n2 = 0, dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < gu_rows; i++) { m1 = fmax(m1, fabs(g1[i] - cgu[i])); n1 = fmax(n1, fabs(cgu[i])); }
        for (size_t i = 0; i < dn_rows; i++) { m2 = fmax(m2, fabs(gy[i] - cy[i])); n2 = fmax(n2, fabs(cy[i])); dot += gy[i] * cy[i]; na += gy[i] * gy[i]; nb += cy[i] * cy[i]; }
        printf("vs fp32 CPU (int8 inputs per 128-group on GPU): gate/up max rel %.3f, down max rel %.3f, cosine %.5f\n", m1 / n1, m2 / n2, dot / sqrt(na * nb));
    }
    return 0;
}
