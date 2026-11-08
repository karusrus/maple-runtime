// gpu_pre.m — reading the prompt on the card (--gpupre N): the experts of one layer for a whole chunk of tokens.
// The weights are not copied in advance: each layer's span of a shard's mmap is wrapped in a Metal buffer (a region)
// and the kernels read the 2-bit codes straight from it (on the iMac that is over PCIe), so with a big chunk each
// expert crosses the bus about once per chunk while the card does many tokens' worth of work with it. Only the
// layer's own regions are bound, so the card wires ~200 MB at a time, not the whole 5.8 GB.
// Per layer: gate+up with silu*up fused (gu), down with the routing weight (dn), the weighted sum per token (cmb).
// Activations are fp32 (the CPU path quantises them to int8), weights {-1,0,+1} with one fp32 scale per row.
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "gpu_k2_metal.h"  // xxd -i gpu_k2.metal
#include "gpu_kt_metal.h"  // xxd -i gpu_kt.metal
#include "gpu_k3_metal.h"  // xxd -i gpu_k3.metal
#include "gpu_k4_metal.h"  // xxd -i gpu_k4.metal
static int GU_TGS = 16;    // gate+up threadgroups per tile: 16 (32 activation rows each), 8 with gpu_k3.metal (64 each)
#define NREG 4     // regions bound per layer
#define MAXREG 64
static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
static id<MTLComputePipelineState> P_GU, P_DN, P_CMB, P_DMM, P_GU16, P_DN16;
static id<MTLBuffer> RG[MAXREG];
static size_t RLEN[MAXREG];
static int NRG;
// staging (--gpupre with GPRE_STAGE): the layer's regions are first copied into VRAM by the copy engine, and the
// next layer's copy is queued right after this layer's work, so it runs while the CPU does attention
static id<MTLBuffer> STG;
static int STAGED[4] = {-2, -2, -2, -2};
static size_t SOFF[4];
static double T_BLIT, T_K[3], T_KF[7]; static long N_FMA[2], N_PAD[2];
static id<MTLBuffer> XG, PT, PW, TL, LC, TP, ACT, DO, YG;
static int MAXT, TOPK_;
static size_t MAXP, MAXTILE;

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define FB(f) ((f) == 0 ? f0 : (f) == 1 ? f1 : (f) == 2 ? f2 : f3)\n"
    "#define FILES device const uchar *f0 [[buffer(10)]], device const uchar *f1 [[buffer(11)]], device const uchar *f2 [[buffer(12)]], \\\n"
    "    device const uchar *f3 [[buffer(13)]]\n"
    // weight code c in {0,1,2} means c - 1; byte j of a 128-column group holds columns j, j+32, j+64, j+96
    "#define STEP(c, sh, A, B) { float w = float(((c) >> (sh)) & 3u) - 1.0f; A += w * xa; B += w * xb; }\n"
    "#define INNER(h) \\\n"
    "    for (uint j4 = 0; j4 < 8; j4++) { \\\n"
    "        uint c0 = cs[r0 * 9 + j4], c1 = cs[r1 * 9 + j4], c2 = cs[r2 * 9 + j4], c3 = cs[r3 * 9 + j4]; \\\n"
    "        for (uint jb = 0; jb < 4; jb++) for (uint sl = 0; sl < 2; sl++) { \\\n"
    "            uint sh = jb * 8 + ((h) * 2 + sl) * 2; \\\n"
    "            threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * 32 + tq * 8); \\\n"
    "            float4 xa = xq[0], xb = xq[1]; \\\n"
    "            STEP(c0, sh, A0, B0) STEP(c1, sh, A1, B1) STEP(c2, sh, A2, B2) STEP(c3, sh, A3, B3) \\\n"
    "        } \\\n"
    "    }\n"
    "static inline float act1(float g, float u) { g = min(g, 7.0f); u = clamp(u, -7.0f, 7.0f); return g / (1.0f + exp(-g)) * u; }\n"
    // tiles[i] = (used expert u, first pair, pairs in the tile <= 32); loc[u*12..] = (file, offset) of gate codes,
    // gate scales, up codes, up scales, down codes, down scales. Threadgroup: 64 threads, 32 act rows x 32 tokens.
    "kernel void gu(device const float *X [[buffer(0)]], device const int *ptok [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],\n"
    "               device const uint *loc [[buffer(3)]], device float *ACT [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,\n"
    "               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * 9];\n"
    "    uint4 T = tiles[tg / 16]; uint rb = tg % 16, p0 = T.y, n = T.z;\n"
    "    device const uint *L = loc + T.x * 12;\n"
    "    device const uchar *gb = FB(L[0]) + L[1], *ub = FB(L[4]) + L[5];\n"
    "    device const float *gs = (device const float *)(FB(L[2]) + L[3]), *us = (device const float *)(FB(L[6]) + L[7]);\n"
    "    uint rp = t % 16, tq = t / 16, lt = t / 2, kp = (t & 1) * 32;\n"
    "    uint r0 = rp, r1 = rp + 16, r2 = 32 + rp, r3 = 48 + rp;\n"
    "    int tok = lt < n ? ptok[p0 + lt] : -1;\n"
    "    device const uint4 *crow = (device const uint4 *)((t < 32 ? gb : ub) + (ulong)(rb * 32 + (t & 31)) * 512);\n"
    "    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;\n"
    "    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1];\n"
    "            cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w;\n"
    "            cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w; }\n"
    "        if (tok >= 0) {\n"
    "            device const float4 *xp = (device const float4 *)(X + (ulong)tok * 2048 + g * 128 + h * 64 + kp);\n"
    "            for (uint i = 0; i < 8; i++) { float4 v = xp[i]; uint k = kp + i * 4;\n"
    "                xs[k * 32 + lt] = v.x; xs[(k + 1) * 32 + lt] = v.y; xs[(k + 2) * 32 + lt] = v.z; xs[(k + 3) * 32 + lt] = v.w; }\n"
    "        } else for (uint i = 0; i < 32; i++) xs[(kp + i) * 32 + lt] = 0;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        INNER(h)\n"
    "    }\n"
    "    uint a = rb * 32 + rp;\n"
    "    float sg0 = gs[a], sg1 = gs[a + 16], su0 = us[a], su1 = us[a + 16];\n"
    "#define OUT(i, c, GA, GB, UA, UB) { uint q = tq * 8 + (i); if (q < n) { device float *o = ACT + (ulong)(p0 + q) * 512 + a; \\\n"
    "        o[0] = act1(GA.c * sg0, UA.c * su0); o[16] = act1(GB.c * sg1, UB.c * su1); } }\n"
    "    OUT(0, x, A0, A1, A2, A3) OUT(1, y, A0, A1, A2, A3) OUT(2, z, A0, A1, A2, A3) OUT(3, w, A0, A1, A2, A3)\n"
    "    OUT(4, x, B0, B1, B2, B3) OUT(5, y, B0, B1, B2, B3) OUT(6, z, B0, B1, B2, B3) OUT(7, w, B0, B1, B2, B3)\n"
    "}\n"
    // down: 64 output rows x 32 pairs per threadgroup, times the row scale and the pair's routing weight
    "kernel void dn(device const float *ACT [[buffer(0)]], device const float *pw [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],\n"
    "               device const uint *loc [[buffer(3)]], device float *DO [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,\n"
    "               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * 9];\n"
    "    uint4 T = tiles[tg / 32]; uint rb = tg % 32, p0 = T.y, n = T.z;\n"
    "    device const uint *L = loc + T.x * 12;\n"
    "    device const uchar *db = FB(L[8]) + L[9]; device const float *ds = (device const float *)(FB(L[10]) + L[11]);\n"
    "    uint rq = t % 16, tq = t / 16, lt = t / 2, kp = (t & 1) * 32;\n"
    "    uint r0 = rq, r1 = rq + 16, r2 = rq + 32, r3 = rq + 48;\n"
    "    device const uint4 *crow = (device const uint4 *)(db + (ulong)(rb * 64 + t) * 128);\n"
    "    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;\n"
    "    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1];\n"
    "            cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w;\n"
    "            cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w; }\n"
    "        if (lt < n) {\n"
    "            device const float4 *xp = (device const float4 *)(ACT + (ulong)(p0 + lt) * 512 + g * 128 + h * 64 + kp);\n"
    "            for (uint i = 0; i < 8; i++) { float4 v = xp[i]; uint k = kp + i * 4;\n"
    "                xs[k * 32 + lt] = v.x; xs[(k + 1) * 32 + lt] = v.y; xs[(k + 2) * 32 + lt] = v.z; xs[(k + 3) * 32 + lt] = v.w; }\n"
    "        } else for (uint i = 0; i < 32; i++) xs[(kp + i) * 32 + lt] = 0;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        INNER(h)\n"
    "    }\n"
    "    uint r = rb * 64 + rq;\n"
    "    float s0 = ds[r], s1 = ds[r + 16], s2 = ds[r + 32], s3 = ds[r + 48];\n"
    "#define DOUT(i, c, AA, BB, CC, DD) { uint q = tq * 8 + (i); if (q < n) { float wq = pw[p0 + q]; device float *o = DO + (ulong)(p0 + q) * 2048 + r; \\\n"
    "        o[0] = AA.c * s0 * wq; o[16] = BB.c * s1 * wq; o[32] = CC.c * s2 * wq; o[48] = DD.c * s3 * wq; } }\n"
    "    DOUT(0, x, A0, A1, A2, A3) DOUT(1, y, A0, A1, A2, A3) DOUT(2, z, A0, A1, A2, A3) DOUT(3, w, A0, A1, A2, A3)\n"
    "    DOUT(4, x, B0, B1, B2, B3) DOUT(5, y, B0, B1, B2, B3) DOUT(6, z, B0, B1, B2, B3) DOUT(7, w, B0, B1, B2, B3)\n"
    "}\n"
    // dense ternary matmul for the attention projections: W (rows x K, all layers' q,k,v,o in one VRAM buffer) times
    // nb contiguous tokens; tiles (matrix, row block, first token, tokens); dl[m] = (codes, scales, rows, out column)
    "kernel void dmm(device const float *X [[buffer(0)]], device const uint4 *tiles [[buffer(2)]], device const uint4 *dl [[buffer(3)]],\n"
    "                device float *Y [[buffer(4)]], constant uint *kg [[buffer(5)]], device const uchar *W [[buffer(6)]],\n"
    "                uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * 9];\n"
    "    uint4 T = tiles[tg]; uint4 M = dl[T.x]; uint rb = T.y, p0 = T.z, n = T.w, K = kg[0] * 128, ys = kg[2];\n"
    "    uint rq = t % 16, tq = t / 16, lt = t / 2, kp = (t & 1) * 32;\n"
    "    uint r0 = rq, r1 = rq + 16, r2 = rq + 32, r3 = rq + 48;\n"
    "    device const uint4 *crow = (device const uint4 *)(W + M.x + (ulong)(rb * 64 + t) * (K / 4));\n"
    "    device const float *sc = (device const float *)(W + M.y);\n"
    "    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;\n"
    "    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1];\n"
    "            cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w;\n"
    "            cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w; }\n"
    "        if (lt < n) {\n"
    "            device const float4 *xp = (device const float4 *)(X + (ulong)(p0 + lt) * K + g * 128 + h * 64 + kp);\n"
    "            for (uint i = 0; i < 8; i++) { float4 v = xp[i]; uint k = kp + i * 4;\n"
    "                xs[k * 32 + lt] = v.x; xs[(k + 1) * 32 + lt] = v.y; xs[(k + 2) * 32 + lt] = v.z; xs[(k + 3) * 32 + lt] = v.w; }\n"
    "        } else for (uint i = 0; i < 32; i++) xs[(kp + i) * 32 + lt] = 0;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        INNER(h)\n"
    "    }\n"
    "    uint r = rb * 64 + rq;\n"
    "    float s0 = sc[r], s1 = sc[r + 16], s2 = sc[r + 32], s3 = sc[r + 48];\n"
    "#define YOUT(i, c, AA, BB, CC, DD) { uint q = tq * 8 + (i); if (q < n) { device float *o = Y + (ulong)(p0 + q) * ys + M.w + r; \\\n"
    "        if (kg[3]) { o[0] += AA.c * s0; o[16] += BB.c * s1; o[32] += CC.c * s2; o[48] += DD.c * s3; } \\\n"
    "        else { o[0] = AA.c * s0; o[16] = BB.c * s1; o[32] = CC.c * s2; o[48] = DD.c * s3; } } }\n"
    "    YOUT(0, x, A0, A1, A2, A3) YOUT(1, y, A0, A1, A2, A3) YOUT(2, z, A0, A1, A2, A3) YOUT(3, w, A0, A1, A2, A3)\n"
    "    YOUT(4, x, B0, B1, B2, B3) YOUT(5, y, B0, B1, B2, B3) YOUT(6, z, B0, B1, B2, B3) YOUT(7, w, B0, B1, B2, B3)\n"
    "}\n"
    // y[token] = sum of its topk pairs' (already weighted) down outputs
    "kernel void cmb(device const float4 *DO [[buffer(0)]], device const int *tp [[buffer(1)]], device float4 *Y [[buffer(2)]],\n"
    "                constant uint *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
    "    uint nb = p[0], k = p[1], b = i / 512, c = i % 512; if (b >= nb) return;\n"
    "    float4 s = 0; for (uint j = 0; j < k; j++) s += DO[(ulong)tp[b * k + j] * 512 + c];\n"
    "    Y[i] = s;\n"
    "}\n";

static id<MTLComputePipelineState> mk(id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLComputePipelineState> p = [D newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err];
    if (!p) fprintf(stderr, "gpre: pipeline %s: %s\n", name, err.localizedDescription.UTF8String);
    return p;
}

// a page-aligned span of an mmapped shard -> region id, or -1
int gpre_region(const void *p, size_t len) {
    if (NRG >= MAXREG || !D) return -1;
    size_t pg = (size_t)getpagesize();
    if (((uintptr_t)p % pg) || (len % pg)) { fprintf(stderr, "gpre: region not page-aligned\n"); return -1; }
    RG[NRG] = [D newBufferWithBytesNoCopy:(void *)p length:len options:MTLResourceStorageModeShared deallocator:nil];
    RLEN[NRG] = len;
    if (!RG[NRG]) { fprintf(stderr, "gpre: region of %zu bytes could not be wrapped\n", len); return -1; }
    return NRG++;
}

int gpre_stage(size_t bytes) {
    STG = [D newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!STG) { fprintf(stderr, "gpre: no VRAM for a %.0f MB staging buffer\n", bytes / 1e6); return -1; }
    fprintf(stderr, "gpre: weights staged in VRAM, %.0f MB\n", bytes / 1e6);
    return 0;
}
static void enc_stage(id<MTLCommandBuffer> cb, const int *regs) {
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    size_t o = 0;
    for (int k = 0; k < NREG; k++) {
        STAGED[k] = regs[k]; SOFF[k] = o;
        if (regs[k] < 0) continue;
        [bl copyFromBuffer:RG[regs[k]] sourceOffset:0 toBuffer:STG destinationOffset:o size:RLEN[regs[k]]];
        o += (RLEN[regs[k]] + 255) / 256 * 256;
    }
    [bl endEncoding];
}
/* ---------- attention projections: every layer's q,k,v (group 0) and o (group 1) live in VRAM for the whole run ---------- */
#define DGRP 48
static uint8_t *DHOST; static size_t DLEN, DCAP;
static uint32_t DL[DGRP][3][4]; static int DN_[DGRP], DROWS[DGRP], DK[DGRP];
static id<MTLBuffer> DW, DLB, DTL, DX, DY;
static double T_DENSE;
static size_t dput(const void *p, size_t n) {
    size_t o = (DLEN + 255) / 256 * 256;
    if (o + n > DCAP) { DCAP = (o + n) * 2; DHOST = realloc(DHOST, DCAP); }
    memcpy(DHOST + o, p, n); DLEN = o + n; return o;
}
// group g (layer * 2 + 0 for q,k,v / + 1 for o): add one matrix, its outputs follow the group's earlier ones
void gpre_dense_add(int g, const uint8_t *codes, const float *scales, int rows, int cols) {
    int m = DN_[g]++;
    DL[g][m][0] = (uint32_t)dput(codes, (size_t)rows * cols / 4); DL[g][m][1] = (uint32_t)dput(scales, (size_t)rows * 4);
    DL[g][m][2] = rows; DL[g][m][3] = DROWS[g]; DROWS[g] += rows; DK[g] = cols;
}
int gpre_dense_done(int maxtok) {
    @autoreleasepool {
        DW = [D newBufferWithBytes:DHOST length:DLEN options:MTLResourceStorageModeManaged];
        free(DHOST); DHOST = NULL;
        int ymax = 0; for (int g = 0; g < DGRP; g++) if (DROWS[g] > ymax) ymax = DROWS[g];
        DLB = [D newBufferWithLength:DGRP * 3 * 16 options:MTLResourceStorageModeManaged];
        memcpy(DLB.contents, DL, sizeof DL); [DLB didModifyRange:NSMakeRange(0, sizeof DL)];
        DTL = [D newBufferWithLength:(size_t)(ymax / 64) * ((maxtok + 31) / 32) * 16 options:MTLResourceStorageModeManaged];
        DX = [D newBufferWithLength:(size_t)maxtok * 2048 * 4 options:MTLResourceStorageModeManaged];
        DY = [D newBufferWithLength:(size_t)maxtok * ymax * 4 options:MTLResourceStorageModeManaged];
        if (!DW || !DLB || !DTL || !DX || !DY) { fprintf(stderr, "gpre: no room for the attention projections\n"); return -1; }
        fprintf(stderr, "gpre: attention projections in VRAM, %.0f MB\n", DLEN / 1e6);
    }
    return 0;
}
// y[nb][rows of the group] = group g's matrices times x[nb][K]
double gpre_dense(int g, const float *x, int nb, float *y) {
    @autoreleasepool {
        int K = DK[g], ys = DROWS[g], nt = 0;
        memcpy(DX.contents, x, (size_t)nb * K * 4); [DX didModifyRange:NSMakeRange(0, (size_t)nb * K * 4)];
        uint32_t *tl = DTL.contents;
        for (int m = 0; m < DN_[g]; m++) for (uint32_t rb = 0; rb < DL[g][m][2] / 64; rb++) for (int p = 0; p < nb; p += 32) {
            tl[nt * 4] = g * 3 + m; tl[nt * 4 + 1] = rb; tl[nt * 4 + 2] = p; tl[nt * 4 + 3] = nb - p < 32 ? nb - p : 32; nt++;
        }
        [DTL didModifyRange:NSMakeRange(0, nt * 16)];
        id<MTLCommandBuffer> cb = [Q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        uint32_t kg[5] = {(uint32_t)(K / 128), 2, (uint32_t)ys, 0, 8};
        [e setComputePipelineState:P_DMM];
        [e setBuffer:DX offset:0 atIndex:0]; [e setBuffer:DTL offset:0 atIndex:2]; [e setBuffer:DLB offset:0 atIndex:3];
        [e setBuffer:DY offset:0 atIndex:4]; [e setBytes:kg length:20 atIndex:5]; [e setBuffer:DW offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        if (!D.hasUnifiedMemory) { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:DY]; [bl endEncoding]; }
        [cb commit]; [cb waitUntilCompleted];
        if (cb.error) fprintf(stderr, "gpre: %s\n", cb.error.localizedDescription.UTF8String);
        memcpy(y, DY.contents, (size_t)nb * ys * 4);
        double t = cb.GPUEndTime - cb.GPUStartTime; T_DENSE += t; return t;
    }
}

double gpre_blit_seconds(void) {
    if (T_DENSE > 0) fprintf(stderr, "gpre: attention projections on the card %.2f s\n", T_DENSE);
    if (T_K[0] > 0) fprintf(stderr, "gpre prof: gate+up %.2f s (%.0f GFMA/s useful, %.0f%% of tile slots used), down %.2f s (%.0f GFMA/s), sum+copy back %.2f s\n",
        T_K[0], N_FMA[0] / T_K[0] / 1e9, 100.0 * N_FMA[0] / N_PAD[0], T_K[1], N_FMA[1] / T_K[1] / 1e9, T_K[2]);
    return T_BLIT;
}

// maxtok: tokens per chunk; returns 0 when ready (then register regions)
int gpre_init(int maxtok, int topk) {
    @autoreleasepool {
        D = MTLCreateSystemDefaultDevice();
        if (!D) { NSArray<id<MTLDevice>> *all = MTLCopyAllDevices(); if (all.count) D = all[0]; }
        if (!D) return -1;
        NSError *err = nil;
        id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { fprintf(stderr, "gpre: compile: %s\n", err.localizedDescription.UTF8String); return -1; }
        P_GU = mk(lib, "gu"); P_DN = mk(lib, "dn"); P_CMB = mk(lib, "cmb"); P_DMM = mk(lib, "dmm");
        if (!P_GU || !P_DN || !P_CMB || !P_DMM) return -1;
        // default: gpu_k4.metal (codes decoded in registers, 1.5 instructions per weight; 6-7% faster on the R9 M290X
        // than the kernels above, which stay as GPRE_K1=1). Measured alternatives, kept for tests: GPRE_KT=1 gpu_kt.metal
        // (+16-token tiles, old decoding: same speed), GPRE_K2=1 gpu_k2.metal (bytes in threadgroup memory: 1.4x slower,
        // fewer waves), GPRE_K3=1 gpu_k3.metal (a row per thread: 2.3x slower, bound by threadgroup-memory reads)
        if (!getenv("GPRE_K1")) {
            int k2 = getenv("GPRE_K2") != NULL, k3 = getenv("GPRE_K3") != NULL, kt = getenv("GPRE_KT") != NULL, k4 = !k2 && !k3 && !kt;
            NSString *src2 = k4 ? [[NSString alloc] initWithBytes:gpu_k4_metal length:gpu_k4_metal_len encoding:NSUTF8StringEncoding]
                           : k2 ? [[NSString alloc] initWithBytes:gpu_k2_metal length:gpu_k2_metal_len encoding:NSUTF8StringEncoding]
                           : k3 ? [[NSString alloc] initWithBytes:gpu_k3_metal length:gpu_k3_metal_len encoding:NSUTF8StringEncoding]
                                : [[NSString alloc] initWithBytes:gpu_kt_metal length:gpu_kt_metal_len encoding:NSUTF8StringEncoding];
            id<MTLLibrary> lib2 = [D newLibraryWithSource:src2 options:nil error:&err];
            if (!lib2) { fprintf(stderr, "gpre: k2 compile: %s\n", err.localizedDescription.UTF8String); return -1; }
            id<MTLComputePipelineState> (^mkc)(const char *, uint32_t) = ^(const char *name, uint32_t nt) {
                MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
                [cv setConstantValue:&nt type:MTLDataTypeUInt atIndex:0];
                NSError *e2 = nil;
                id<MTLFunction> f = [lib2 newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&e2];
                if (!f) { fprintf(stderr, "gpre: %s/%u: %s\n", name, nt, e2.localizedDescription.UTF8String); return (id<MTLComputePipelineState>)nil; }
                id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e2];
                if (!ps) fprintf(stderr, "gpre: pipeline %s/%u: %s\n", name, nt, e2.localizedDescription.UTF8String);
                return ps;
            };
            P_GU = mkc("gu", 32); P_DN = mkc("dn", 32); P_GU16 = mkc("gu", 16); P_DN16 = mkc("dn", 16);
            if (k2 || k3 || k4) P_DMM = mk(lib2, "dmm");
            if (k3) GU_TGS = 8;
            if (!P_GU || !P_DN || !P_DMM || !P_GU16 || !P_DN16) return -1;
        }
        Q = [D newCommandQueue];
        MAXT = maxtok; TOPK_ = topk; MAXP = (size_t)maxtok * topk; MAXTILE = MAXP / 32 + 512 + 1;
        MTLResourceOptions M = MTLResourceStorageModeManaged, P = MTLResourceStorageModePrivate;
        XG = [D newBufferWithLength:(size_t)maxtok * 2048 * 4 options:M];
        YG = [D newBufferWithLength:(size_t)maxtok * 2048 * 4 options:M];
        PT = [D newBufferWithLength:MAXP * 4 options:M]; PW = [D newBufferWithLength:MAXP * 4 options:M]; TP = [D newBufferWithLength:MAXP * 4 options:M];
        TL = [D newBufferWithLength:MAXTILE * 16 options:M]; LC = [D newBufferWithLength:256 * 12 * 4 options:M];
        ACT = [D newBufferWithLength:MAXP * 512 * 4 options:P]; DO = [D newBufferWithLength:MAXP * 2048 * 4 options:P];
        if (!XG || !YG || !PT || !PW || !TP || !TL || !LC || !ACT || !DO) { fprintf(stderr, "gpre: out of memory for %d tokens\n", maxtok); return -1; }
        fprintf(stderr, "gpre: %s, chunks up to %d tokens, %.0f MB on the card, weights read in place\n",
                D.name.UTF8String, maxtok, (MAXP * (512 + 2048) * 4 + (size_t)maxtok * 2048 * 8) / 1e6);
    }
    return 0;
}

// x: nb normalised hidden vectors; nu used experts, pairs grouped by expert (ustart[nu+1], ptok, pw);
// loc: nu x 12 (region slot 0..3, offset); regs: the layer's region ids for slots 0..3 (-1 unused);
// tp: nb x topk pair indices; y: nb x 2048 out. Returns seconds spent on the card.
double gpre_moe(const float *x, int nb, int nu, const uint32_t *loc, const int *regs, const int *next, const int *ustart, const int *ptok, const float *pw, const int *tp, float *y) {
    @autoreleasepool {
        int np = ustart[nu];
        memcpy(XG.contents, x, (size_t)nb * 2048 * 4); [XG didModifyRange:NSMakeRange(0, (size_t)nb * 2048 * 4)];
        memcpy(PT.contents, ptok, np * 4); [PT didModifyRange:NSMakeRange(0, np * 4)];
        memcpy(PW.contents, pw, np * 4); [PW didModifyRange:NSMakeRange(0, np * 4)];
        memcpy(TP.contents, tp, (size_t)nb * TOPK_ * 4); [TP didModifyRange:NSMakeRange(0, (size_t)nb * TOPK_ * 4)];
        memcpy(LC.contents, loc, nu * 48); [LC didModifyRange:NSMakeRange(0, nu * 48)];
        uint32_t *tl = TL.contents; int nt = 0;
        for (int u = 0; u < nu; u++) for (int p = ustart[u]; p < ustart[u + 1]; p += 32) {
            int c = ustart[u + 1] - p < 32 ? ustart[u + 1] - p : 32;
            tl[nt * 4] = u; tl[nt * 4 + 1] = p; tl[nt * 4 + 2] = c; tl[nt * 4 + 3] = 0; nt++;
        }
        [TL didModifyRange:NSMakeRange(0, nt * 16)];
        id<MTLCommandBuffer> cb = [Q commandBuffer];
        if (STG && memcmp(STAGED, regs, sizeof STAGED)) enc_stage(cb, regs);  // not queued ahead: copy now
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int k = 0; k < NREG; k++) {
            int r = regs[k] >= 0 ? k : 0;
            if (STG) [e setBuffer:STG offset:SOFF[r] atIndex:10 + k];
            else [e setBuffer:RG[regs[r]] offset:0 atIndex:10 + k];
        }
        [e setComputePipelineState:P_GU];
        [e setBuffer:XG offset:0 atIndex:0]; [e setBuffer:PT offset:0 atIndex:1]; [e setBuffer:TL offset:0 atIndex:2];
        [e setBuffer:LC offset:0 atIndex:3]; [e setBuffer:ACT offset:0 atIndex:4];
        uint32_t kgu[3] = {16, 2, 8}, kgd[3] = {4, 2, 8};
        [e setBytes:kgu length:12 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((size_t)nt * GU_TGS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        static int prof = -1; if (prof < 0) prof = getenv("GPRE_PROF") != NULL;
        N_FMA[0] += (long)np * 1024 * 2048; N_PAD[0] += (long)nt * 32 * 1024 * 2048;
        N_FMA[1] += (long)np * 2048 * 512; N_PAD[1] += (long)nt * 32 * 2048 * 512;
        if (prof) {
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; T_K[0] += cb.GPUEndTime - cb.GPUStartTime;
            cb = [Q commandBuffer]; e = [cb computeCommandEncoder];
            for (int k = 0; k < NREG; k++) { int r = regs[k] >= 0 ? k : 0; if (STG) [e setBuffer:STG offset:SOFF[r] atIndex:10 + k]; else [e setBuffer:RG[regs[r]] offset:0 atIndex:10 + k]; }
            [e setBuffer:TL offset:0 atIndex:2]; [e setBuffer:LC offset:0 atIndex:3];
        }
        [e setComputePipelineState:P_DN];
        [e setBytes:kgd length:12 atIndex:5];
        [e setBuffer:ACT offset:0 atIndex:0]; [e setBuffer:PW offset:0 atIndex:1]; [e setBuffer:DO offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((size_t)nt * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (prof) {
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; T_K[1] += cb.GPUEndTime - cb.GPUStartTime;
            cb = [Q commandBuffer]; e = [cb computeCommandEncoder];
        }
        [e setComputePipelineState:P_CMB];
        uint32_t pc[2] = {(uint32_t)nb, (uint32_t)TOPK_};
        [e setBuffer:DO offset:0 atIndex:0]; [e setBuffer:TP offset:0 atIndex:1]; [e setBuffer:YG offset:0 atIndex:2]; [e setBytes:pc length:8 atIndex:3];
        [e dispatchThreads:MTLSizeMake((size_t)nb * 512, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        if (!D.hasUnifiedMemory) { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:YG]; [bl endEncoding]; }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) fprintf(stderr, "gpre: %s\n", cb.error.localizedDescription.UTF8String);
        if (prof) T_K[2] += cb.GPUEndTime - cb.GPUStartTime;
        memcpy(y, YG.contents, (size_t)nb * 2048 * 4);
        if (STG && next) {  // the next layer's weights cross the bus while the CPU does its attention
            id<MTLCommandBuffer> cn = [Q commandBuffer];
            enc_stage(cn, next);
            [cn addCompletedHandler:^(id<MTLCommandBuffer> c) { T_BLIT += c.GPUEndTime - c.GPUStartTime; }];
            [cn commit];
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

/* ---------- the whole layer on the card (gpu_full.metal): the hidden state stays in VRAM for the chunk ----------
   per layer: [rms in_ln -> q,k,v -> q/k norm + rope -> attention -> o (+= x) -> rms post_ln -> router logits] -> wait;
   the CPU picks the experts from the logits; [experts -> x += weighted sum] is queued without waiting. */
#include "gpu_full_metal.h"  // xxd -i gpu_full.metal
#define NWL 4352  // per layer: in_ln 2048 | post_ln 2048 | q_norm 128 | k_norm 128 (floats)
static id<MTLComputePipelineState> P_RMS, P_QKN, P_ATT, P_RTR, P_CMBA;
static id<MTLBuffer> XC, NWB, RWB, CSB, KGB, VGB, KOB, QOB, ATB, RLB, DTL2;
static int FCTX, FMAX, FNB, FPOS0, FWIN;
static double T_FA, T_FM;
static id<MTLCommandBuffer> LASTM;  // the last experts' command buffer, timed once it is done
static double gt(id<MTLCommandBuffer> c) { return c.GPUEndTime - c.GPUStartTime; }

int gfull_init(int ctx, int maxtok, int win, const float *norms, const uint16_t *router, const float *cs) {
    @autoreleasepool {
        if (!D || !P_DMM || !DW) return -1;
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new]; opt.fastMathEnabled = getenv("GFULL_PRECISE") != NULL ? NO : YES;  // fast math: a*b+c as mad, full rate on GCN
        NSString *src = [[NSString alloc] initWithBytes:gpu_full_metal length:gpu_full_metal_len encoding:NSUTF8StringEncoding];
        id<MTLLibrary> lib = [D newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "gfull: compile: %s\n", err.localizedDescription.UTF8String); return -1; }
        P_RMS = mk(lib, "rms"); P_QKN = mk(lib, "qkn"); P_ATT = mk(lib, "att"); P_RTR = mk(lib, "rtr"); P_CMBA = mk(lib, "cmba");
        if (!P_RMS || !P_QKN || !P_ATT || !P_RTR || !P_CMBA) return -1;
        if (P_ATT.maxTotalThreadsPerThreadgroup < 128 || P_RMS.maxTotalThreadsPerThreadgroup < 256) {
            fprintf(stderr, "gfull: threadgroups too small (%lu, %lu)\n", (unsigned long)P_ATT.maxTotalThreadsPerThreadgroup, (unsigned long)P_RMS.maxTotalThreadsPerThreadgroup);
            return -1;
        }
        FCTX = ctx; FMAX = maxtok; FWIN = win;
        MTLResourceOptions M = MTLResourceStorageModeManaged, P = MTLResourceStorageModePrivate;
        size_t kvb = (size_t)24 * ctx * 512 * 2, kob = (size_t)24 * maxtok * 1024 * 2;
        XC = [D newBufferWithLength:(size_t)maxtok * 2048 * 4 options:M];
        NWB = [D newBufferWithBytes:norms length:(size_t)24 * NWL * 4 options:M];
        RWB = [D newBufferWithBytes:router length:(size_t)24 * 256 * 2048 * 2 options:M];
        CSB = [D newBufferWithBytes:cs length:(size_t)ctx * 32 * 8 options:M];
        KGB = [D newBufferWithLength:kvb options:P]; VGB = [D newBufferWithLength:kvb options:P];
        KOB = [D newBufferWithLength:kob options:M];
        QOB = [D newBufferWithLength:(size_t)maxtok * 2048 * 4 options:P]; ATB = [D newBufferWithLength:(size_t)maxtok * 2048 * 4 options:P];
        RLB = [D newBufferWithLength:(size_t)maxtok * 256 * 4 options:M];
        DTL2 = [D newBufferWithLength:(size_t)(48 + 32) * ((maxtok + 31) / 32) * 16 options:M];
        if (!XC || !NWB || !RWB || !CSB || !KGB || !VGB || !KOB || !QOB || !ATB || !RLB || !DTL2) { fprintf(stderr, "gfull: out of VRAM\n"); return -1; }
        fprintf(stderr, "gfull: whole layers on the card, key/value cache %.0f MB, %.0f MB more\n", 2.0 * kvb / 1e6,
                (kob + (size_t)maxtok * 2048 * 4 * 3 + (size_t)24 * 256 * 2048 * 2) / 1e6);
    }
    return 0;
}

// a new chunk: nb hidden vectors (embeddings) at positions pos0..
void gfull_begin(const float *x, int nb, int pos0) {
    memcpy(XC.contents, x, (size_t)nb * 2048 * 4); [XC didModifyRange:NSMakeRange(0, (size_t)nb * 2048 * 4)];
    FNB = nb; FPOS0 = pos0;
}

static void enc_dmm(id<MTLComputeCommandEncoder> e, id<MTLBuffer> in, id<MTLBuffer> out, size_t toff, int nt, int ys, int acc) {
    uint32_t kg[5] = {16, 2, (uint32_t)ys, (uint32_t)acc, 8};
    [e setComputePipelineState:P_DMM];
    [e setBuffer:in offset:0 atIndex:0]; [e setBuffer:DTL2 offset:toff atIndex:2]; [e setBuffer:DLB offset:0 atIndex:3];
    [e setBuffer:out offset:0 atIndex:4]; [e setBytes:kg length:20 atIndex:5]; [e setBuffer:DW offset:0 atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// attention half of layer L and the router: rl gets nb x 256 logits. Returns the card's seconds.
double gfull_attn(int L, int glob, float *rl) {
    @autoreleasepool {
        static int prof = -1; if (prof < 0) prof = getenv("GPRE_PROF") != NULL;
        int nb = FNB, nt[2] = {0, 0};
        uint32_t *tl = DTL2.contents, *t0 = tl;
        for (int k = 0; k < 2; k++) {
            int g = L * 2 + k;
            for (int m = 0; m < DN_[g]; m++) for (uint32_t rb = 0; rb < DL[g][m][2] / 64; rb++) for (int p = 0; p < nb; p += 32) {
                tl[0] = g * 3 + m; tl[1] = rb; tl[2] = p; tl[3] = nb - p < 32 ? nb - p : 32; tl += 4; nt[k]++;
            }
        }
        [DTL2 didModifyRange:NSMakeRange(0, (size_t)(nt[0] + nt[1]) * 16)];
        (void)t0;
        size_t kvo = (size_t)L * FCTX * 512 * 2, koo = (size_t)L * FMAX * 1024 * 2, nwo = (size_t)L * NWL * 4;
        uint32_t pq[4] = {(uint32_t)FPOS0, (uint32_t)glob, (uint32_t)nb, 0}, pa[4] = {(uint32_t)FPOS0, (uint32_t)glob, (uint32_t)nb, (uint32_t)FWIN};
        uint32_t nbu = nb;
        double tsum = 0;
        id<MTLCommandBuffer> cb = [Q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
#define GSPLIT(i) if (prof) { [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; T_KF[i] += gt(cb); tsum += gt(cb); cb = [Q commandBuffer]; e = [cb computeCommandEncoder]; }
        [e setComputePipelineState:P_RMS];
        [e setBuffer:XC offset:0 atIndex:0]; [e setBuffer:NWB offset:nwo atIndex:1]; [e setBuffer:DX offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        GSPLIT(0)
        enc_dmm(e, DX, DY, 0, nt[0], 3072, 0);
        GSPLIT(1)
        [e setComputePipelineState:P_QKN];
        [e setBuffer:DY offset:0 atIndex:0]; [e setBuffer:NWB offset:nwo + 4096 * 4 atIndex:1]; [e setBuffer:NWB offset:nwo + 4224 * 4 atIndex:2];
        [e setBuffer:CSB offset:0 atIndex:3]; [e setBuffer:QOB offset:0 atIndex:4]; [e setBuffer:KGB offset:kvo atIndex:5];
        [e setBuffer:VGB offset:kvo atIndex:6]; [e setBuffer:KOB offset:koo atIndex:7]; [e setBytes:pq length:16 atIndex:8];
        [e dispatchThreadgroups:MTLSizeMake(nb, 24, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        GSPLIT(2)
        [e setComputePipelineState:P_ATT];
        [e setBuffer:QOB offset:0 atIndex:0]; [e setBuffer:KGB offset:kvo atIndex:1]; [e setBuffer:VGB offset:kvo atIndex:2];
        [e setBuffer:ATB offset:0 atIndex:3]; [e setBytes:pa length:16 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((nb + 7) / 8, 4, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        GSPLIT(3)
        enc_dmm(e, ATB, XC, (size_t)nt[0] * 16, nt[1], 2048, 1);
        GSPLIT(4)
        [e setComputePipelineState:P_RMS];
        [e setBuffer:XC offset:0 atIndex:0]; [e setBuffer:NWB offset:nwo + 2048 * 4 atIndex:1]; [e setBuffer:XG offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        GSPLIT(5)
        [e setComputePipelineState:P_RTR];
        [e setBuffer:XG offset:0 atIndex:0]; [e setBuffer:RWB offset:(size_t)L * 256 * 2048 * 2 atIndex:1]; [e setBuffer:RLB offset:0 atIndex:2];
        [e setBytes:&nbu length:4 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(4, (nb + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        if (!D.hasUnifiedMemory) { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:RLB]; [bl endEncoding]; }
        [cb commit]; [cb waitUntilCompleted];
        if (cb.error) fprintf(stderr, "gfull: %s\n", cb.error.localizedDescription.UTF8String);
        if (prof) T_KF[6] += gt(cb);
        tsum += gt(cb);
        if (LASTM) { if (LASTM.error) fprintf(stderr, "gfull experts: %s\n", LASTM.error.localizedDescription.UTF8String); T_FM += gt(LASTM); LASTM = nil; }
        memcpy(rl, RLB.contents, (size_t)nb * 256 * 4);
        T_FA += tsum;
        return tsum;
    }
}

// the experts of layer L for the pairs the CPU picked; x += their weighted sum. Queued, not waited for.
void gfull_moe(int nu, const uint32_t *loc, const int *regs, const int *ustart, const int *ptok, const float *pw, const int *tp) {
    @autoreleasepool {
        int nb = FNB, np = ustart[nu];
        memcpy(PT.contents, ptok, np * 4); [PT didModifyRange:NSMakeRange(0, np * 4)];
        memcpy(PW.contents, pw, np * 4); [PW didModifyRange:NSMakeRange(0, np * 4)];
        memcpy(TP.contents, tp, (size_t)nb * TOPK_ * 4); [TP didModifyRange:NSMakeRange(0, (size_t)nb * TOPK_ * 4)];
        memcpy(LC.contents, loc, nu * 48); [LC didModifyRange:NSMakeRange(0, nu * 48)];
        static int t32 = -1; if (t32 < 0) t32 = getenv("GPRE_T32") != NULL || !P_GU16;
        uint32_t *tl = TL.contents; int nt = 0, ns = 0;
        static uint32_t small[256 * 4];
        for (int u = 0; u < nu; u++) for (int p = ustart[u]; p < ustart[u + 1]; p += 32) {
            int c = ustart[u + 1] - p < 32 ? ustart[u + 1] - p : 32;
            if (c <= 16 && !t32) { small[ns * 4] = u; small[ns * 4 + 1] = p; small[ns * 4 + 2] = c; small[ns * 4 + 3] = 0; ns++; continue; }
            tl[nt * 4] = u; tl[nt * 4 + 1] = p; tl[nt * 4 + 2] = c; tl[nt * 4 + 3] = 0; nt++;
        }
        memcpy(tl + nt * 4, small, ns * 16);
        [TL didModifyRange:NSMakeRange(0, (nt + ns) * 16)];
        N_FMA[0] += (long)np * 1024 * 2048; N_PAD[0] += (long)(nt * 32 + ns * 16) * 1024 * 2048;
        N_FMA[1] += (long)np * 2048 * 512; N_PAD[1] += (long)(nt * 32 + ns * 16) * 2048 * 512;
        static int prof = -1; if (prof < 0) prof = getenv("GPRE_PROF") != NULL;
        id<MTLCommandBuffer> cb = [Q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int k = 0; k < NREG; k++) { int r = regs[k] >= 0 ? k : 0; [e setBuffer:RG[regs[r]] offset:0 atIndex:10 + k]; }
        uint32_t kgu[3] = {16, 2, 8}, kgd[3] = {4, 2, 8}, pc[2] = {(uint32_t)nb, (uint32_t)TOPK_};
        [e setComputePipelineState:P_GU];
        [e setBuffer:XG offset:0 atIndex:0]; [e setBuffer:PT offset:0 atIndex:1]; [e setBuffer:TL offset:0 atIndex:2];
        [e setBuffer:LC offset:0 atIndex:3]; [e setBuffer:ACT offset:0 atIndex:4]; [e setBytes:kgu length:12 atIndex:5];
        if (nt) [e dispatchThreadgroups:MTLSizeMake((size_t)nt * GU_TGS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (ns) {
            [e setComputePipelineState:P_GU16]; [e setBuffer:TL offset:(size_t)nt * 16 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((size_t)ns * GU_TGS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e setBuffer:TL offset:0 atIndex:2];
        }
        if (prof) {
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; T_K[0] += gt(cb); T_FM += gt(cb);
            cb = [Q commandBuffer]; e = [cb computeCommandEncoder];
            for (int k = 0; k < NREG; k++) { int r = regs[k] >= 0 ? k : 0; [e setBuffer:RG[regs[r]] offset:0 atIndex:10 + k]; }
            [e setBuffer:TL offset:0 atIndex:2]; [e setBuffer:LC offset:0 atIndex:3];
        }
        [e setComputePipelineState:P_DN];
        [e setBuffer:ACT offset:0 atIndex:0]; [e setBuffer:PW offset:0 atIndex:1]; [e setBuffer:DO offset:0 atIndex:4]; [e setBytes:kgd length:12 atIndex:5];
        if (nt) [e dispatchThreadgroups:MTLSizeMake((size_t)nt * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (ns) {
            [e setComputePipelineState:P_DN16]; [e setBuffer:TL offset:(size_t)nt * 16 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((size_t)ns * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        if (prof) {
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; T_K[1] += gt(cb); T_FM += gt(cb);
            cb = [Q commandBuffer]; e = [cb computeCommandEncoder];
        }
        [e setComputePipelineState:P_CMBA];
        [e setBuffer:DO offset:0 atIndex:0]; [e setBuffer:TP offset:0 atIndex:1]; [e setBuffer:XC offset:0 atIndex:2]; [e setBytes:pc length:8 atIndex:3];
        [e dispatchThreads:MTLSizeMake((size_t)nb * 512, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        [cb commit];
        if (prof) { [cb waitUntilCompleted]; T_K[2] += gt(cb); T_FM += gt(cb); }
        else LASTM = cb;
    }
}

// end of the chunk (or a check): x back to the CPU; ko -> the card's k,v of every layer (24 x maxtok x 1024 fp16)
void gfull_read(float *x, const uint16_t **ko) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [Q commandBuffer];
        if (!D.hasUnifiedMemory) {
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
            if (x) [bl synchronizeResource:XC];
            if (ko) [bl synchronizeResource:KOB];
            [bl endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        if (cb.error) fprintf(stderr, "gfull: %s\n", cb.error.localizedDescription.UTF8String);
        if (LASTM) { if (LASTM.error) fprintf(stderr, "gfull experts: %s\n", LASTM.error.localizedDescription.UTF8String); T_FM += gt(LASTM); LASTM = nil; }
        if (x) memcpy(x, XC.contents, (size_t)FNB * 2048 * 4);
        if (ko) *ko = KOB.contents;
    }
}
// a check may overwrite the card's hidden state with the CPU's
void gfull_write_x(const float *x) { memcpy(XC.contents, x, (size_t)FNB * 2048 * 4); [XC didModifyRange:NSMakeRange(0, (size_t)FNB * 2048 * 4)]; }
int gfull_maxtok(void) { return FMAX; }

void gfull_report(void) {
    fprintf(stderr, "gfull: card %.2f s attention half + router, %.2f s experts\n", T_FA, T_FM);
    if (T_KF[0] + T_KF[1] > 0)
        fprintf(stderr, "gfull prof: rms %.3f, qkv %.3f, qk-norm %.3f, attention %.3f, o %.3f, rms %.3f, router %.3f s\n",
                T_KF[0], T_KF[1], T_KF[2], T_KF[3], T_KF[4], T_KF[5], T_KF[6]);
}
