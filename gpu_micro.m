// gpu_micro.m — two numbers before moving Maple's experts to the Radeon:
//  1) the cost of one CPU -> GPU -> CPU round trip per layer (upload 8 KB, tiny kernel, read back 8 KB);
//  2) how fast the GPU runs a layer's experts in our packed ternary layout (8 experts: gate+up 512x2048, down 2048x512).
// Random weights, checked against the same arithmetic on the CPU.
//   cc -O2 -fobjc-arc -o gpu_micro gpu_micro.m -framework Metal -framework Foundation && ./gpu_micro
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
    // one threadgroup of 64 per row; each row is cols/4 bytes of 2-bit codes in 128-column groups:
    // byte j of a group holds columns j, j+32, j+64, j+96 at bits 0, 2, 4, 6
    "kernel void tern(device const uint *w [[buffer(0)]], device const float *x [[buffer(1)]],\n"
    "                 device const float *scale [[buffer(2)]], device float *y [[buffer(3)]],\n"
    "                 constant uint *p [[buffer(4)]],\n"   // p[0]=cols, p[1]=rows per input vector
    "                 uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[64];\n"
    "    uint cols = p[0], words = cols / 16;\n"
    "    device const float *xv = x + (row / p[1]) * cols;\n"
    "    device const uint *wr = w + (ulong)row * words;\n"
    "    float acc = 0;\n"
    "    for (uint i = t; i < words; i += 64) {\n"
    "        uint v = wr[i]; uint b0 = i * 4, g = b0 / 32, j = b0 % 32;\n"
    "        device const float *xg = xv + g * 128 + j;\n"
    "        for (uint k = 0; k < 4; k++) {\n"
    "            uint byte = (v >> (8 * k)) & 255;\n"
    "            acc += (float((byte >> 0) & 3) - 1) * xg[k] + (float((byte >> 2) & 3) - 1) * xg[k + 32]\n"
    "                 + (float((byte >> 4) & 3) - 1) * xg[k + 64] + (float((byte >> 6) & 3) - 1) * xg[k + 96];\n"
    "        }\n"
    "    }\n"
    "    part[t] = acc;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = 32; s > 0; s >>= 1) { if (t < s) part[t] += part[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "    if (t == 0) y[row] = part[0] * scale[row];\n"
    "}\n"
    "kernel void touch(device float *y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = y[i] + 1.0f; }\n";

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }

static void cpu_tern(const uint8_t *w, const float *x, const float *s, float *y, int rows, int cols, int rows_per_in) {
    for (int r = 0; r < rows; r++) {
        const uint8_t *p = w + (size_t)r * cols / 4; const float *xv = x + (size_t)(r / rows_per_in) * cols; double a = 0;
        for (int g = 0; g < cols / 128; g++) for (int j = 0; j < 32; j++) {
            uint8_t b = p[g * 32 + j]; const float *xg = xv + g * 128 + j;
            a += ((b & 3) - 1.0) * xg[0] + (((b >> 2) & 3) - 1.0) * xg[32] + (((b >> 4) & 3) - 1.0) * xg[64] + ((b >> 6) - 1.0) * xg[96];
        }
        y[r] = (float)a * s[r];
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { NSArray<id<MTLDevice>> *all = MTLCopyAllDevices(); if (all.count) dev = all[0]; }
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pt = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tern"] error:&err];
        id<MTLComputePipelineState> pz = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("device: %s\n", dev.name.UTF8String);

        // 1) round trip: upload 2048 floats, tiny kernel, read back 2048 floats
        id<MTLBuffer> up = [dev newBufferWithLength:H * 4 options:MTLResourceStorageModeManaged];
        double rt[500];
        float tmp[H]; for (int i = 0; i < H; i++) tmp[i] = i;
        for (int it = 0; it < 500; it++) {
            double t0 = now();
            memcpy(up.contents, tmp, sizeof tmp); [up didModifyRange:NSMakeRange(0, sizeof tmp)];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pz]; [e setBuffer:up offset:0 atIndex:0];
            [e dispatchThreads:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:up]; [bl endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            memcpy(tmp, up.contents, sizeof tmp);
            rt[it] = now() - t0;
        }
        printf("round trip per layer (8 KB up, kernel, 8 KB back): median %.3f ms\n", median(rt + 20, 480) * 1e3);

        // 2) a layer's experts: n experts, gate+up then down, weights resident in VRAM
        srand(7);
        int NE = 8;
        size_t gu_rows = (size_t)NE * 2 * FF, dn_rows = (size_t)NE * H;
        size_t gu_bytes = gu_rows * H / 4, dn_bytes = dn_rows * FF / 4;
        uint8_t *wgu = malloc(gu_bytes), *wdn = malloc(dn_bytes);
        for (size_t i = 0; i < gu_bytes; i++) { uint8_t b = 0; for (int k = 0; k < 4; k++) b |= (rand() % 3) << (2 * k); wgu[i] = b; }
        for (size_t i = 0; i < dn_bytes; i++) { uint8_t b = 0; for (int k = 0; k < 4; k++) b |= (rand() % 3) << (2 * k); wdn[i] = b; }
        float *sgu = malloc(gu_rows * 4), *sdn = malloc(dn_rows * 4), *x = malloc(H * 4), *a = malloc((size_t)NE * FF * 4);
        for (size_t i = 0; i < gu_rows; i++) sgu[i] = 0.01f + (rand() % 100) * 1e-4f;
        for (size_t i = 0; i < dn_rows; i++) sdn[i] = 0.01f + (rand() % 100) * 1e-4f;
        for (int i = 0; i < H; i++) x[i] = (rand() % 2001 - 1000) * 1e-3f;
        for (int i = 0; i < NE * FF; i++) a[i] = (rand() % 2001 - 1000) * 1e-3f;
        id<MTLBuffer> bgu = [dev newBufferWithLength:gu_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bdn = [dev newBufferWithLength:dn_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> st1 = [dev newBufferWithBytes:wgu length:gu_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> st2 = [dev newBufferWithBytes:wdn length:dn_bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb0 = [q commandBuffer]; id<MTLBlitCommandEncoder> b0 = [cb0 blitCommandEncoder];
        [b0 copyFromBuffer:st1 sourceOffset:0 toBuffer:bgu destinationOffset:0 size:gu_bytes];
        [b0 copyFromBuffer:st2 sourceOffset:0 toBuffer:bdn destinationOffset:0 size:dn_bytes];
        [b0 endEncoding]; [cb0 commit]; [cb0 waitUntilCompleted];
        id<MTLBuffer> bsgu = [dev newBufferWithBytes:sgu length:gu_rows * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bsdn = [dev newBufferWithBytes:sdn length:dn_rows * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x length:H * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> ba = [dev newBufferWithBytes:a length:(size_t)NE * FF * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> by1 = [dev newBufferWithLength:gu_rows * 4 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> by2 = [dev newBufferWithLength:dn_rows * 4 options:MTLResourceStorageModeManaged];
        uint32_t p1[2] = {H, (uint32_t)gu_rows}, p2[2] = {FF, H};  // gate+up share x; down: one input per expert (H rows each)
        double gk[200], gw[200];
        for (int it = 0; it < 200; it++) {
            double t0 = now();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pt];
            [e setBuffer:bgu offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1]; [e setBuffer:bsgu offset:0 atIndex:2];
            [e setBuffer:by1 offset:0 atIndex:3]; [e setBytes:p1 length:sizeof p1 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(gu_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e setBuffer:bdn offset:0 atIndex:0]; [e setBuffer:ba offset:0 atIndex:1]; [e setBuffer:bsdn offset:0 atIndex:2];
            [e setBuffer:by2 offset:0 atIndex:3]; [e setBytes:p2 length:sizeof p2 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(dn_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:by1]; [bl synchronizeResource:by2]; [bl endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            gw[it] = now() - t0; gk[it] = cb.GPUEndTime - cb.GPUStartTime;
        }
        double kt = median(gk + 10, 190), wt = median(gw + 10, 190);
        double mb = (gu_bytes + dn_bytes) / 1e6;
        printf("8 experts of one layer on GPU: kernel %.3f ms, with submit and wait %.3f ms, %.1f MB -> %.1f GB/s\n", kt * 1e3, wt * 1e3, mb, mb / 1e3 / kt);
        printf("per token, 24 layers, if all 8 experts were on the GPU: %.1f ms\n", 24 * wt * 1e3);

        // check against the CPU
        float *c1 = malloc(gu_rows * 4), *c2 = malloc(dn_rows * 4);
        cpu_tern(wgu, x, sgu, c1, (int)gu_rows, H, (int)gu_rows);
        cpu_tern(wdn, a, sdn, c2, (int)dn_rows, FF, H);
        const float *g1 = by1.contents, *g2 = by2.contents; double m1 = 0, m2 = 0, n1 = 0, n2 = 0;
        for (size_t i = 0; i < gu_rows; i++) { m1 = fmax(m1, fabs(g1[i] - c1[i])); n1 = fmax(n1, fabs(c1[i])); }
        for (size_t i = 0; i < dn_rows; i++) { m2 = fmax(m2, fabs(g2[i] - c2[i])); n2 = fmax(n2, fabs(c2[i])); }
        printf("check vs CPU: gate/up max rel %.2e, down max rel %.2e\n", m1 / n1, m2 / n2);
    }
    return 0;
}
