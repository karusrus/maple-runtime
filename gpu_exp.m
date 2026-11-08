// gpu_exp.m — a cache of Maple experts in the Radeon's VRAM and the kernel that runs them (see gpu_micro4.m).
// Per layer, S slots hold experts as plus/minus bit masks; codes are uploaded as they are on disk and turned into
// masks on the card. A layer's hits run as: planes of x, gate+up (16 threads per row), silu*up with its planes, down.
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAXL 24
#define MAXH 8
#define NSTAGE 8
#define GU_ROWS 1024   // gate (512) + up (512)
#define DN_ROWS 2048
#define CODE_MAT 262144  // one expert matrix: 512 x 2048 ternary weights, 2 bits each

static id<MTLDevice> D;
static id<MTLCommandQueue> QC;
static id<MTLComputePipelineState> P_PX, P_PA, P_GU, P_DN, P_CONV, P_PXM, P_GUP, P_DNP;
#define MAXB 16
#define MAXP 128
static id<MTLBuffer> XBM, PLXM, SXM, GUOM, PLAM, SAM, YOM, TAB;
static id<MTLBuffer> GU_M[MAXL], GU_S[MAXL], DN_M[MAXL], DN_S[MAXL];
static id<MTLBuffer> XB, PLX, SX, GUO, PLA, SA, YO, STG[NSTAGE];
static id<MTLCommandBuffer> STG_CB[NSTAGE];
static int STG_I;

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
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
    "kernel void planes_x(device const float *x [[buffer(0)]], device uint *plw [[buffer(1)]], device float *sx [[buffer(2)]],\n"
    "                     uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float xs[2048]; threadgroup uint pl[512]; threadgroup float s[16]; threadgroup float gm[64];\n"
    "    for (uint i = t; i < 2048; i += 64) xs[i] = x[i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    tg_planes(xs, 16, pl, s, gm, t);\n"
    "    for (uint i = t; i < 512; i += 64) plw[i] = pl[i];\n"
    "    if (t < 16) sx[t] = s[t];\n"
    "}\n"
    "kernel void planes_act(device const float *gu [[buffer(0)]], device uint *plw [[buffer(1)]], device float *sx [[buffer(2)]],\n"
    "                       uint h [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float as[512]; threadgroup uint pl[128]; threadgroup float s[4]; threadgroup float gm[64];\n"
    "    for (uint i = t; i < 512; i += 64) {\n"
    "        float g = min(gu[h * 1024 + i], 7.0f), u = clamp(gu[h * 1024 + 512 + i], -7.0f, 7.0f);\n"
    "        as[i] = g / (1.0f + exp(-g)) * u;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    tg_planes(as, 4, pl, s, gm, t);\n"
    "    for (uint i = t; i < 128; i += 64) plw[h * 128 + i] = pl[i];\n"
    "    if (t < 4) sx[h * 4 + t] = s[t];\n"
    "}\n"
    "static float group_dot(uint4 ps, uint4 ms, device const uint *P) {\n"
    "    int tot = 0;\n"
    "    for (uint q = 0; q < 8; q++) {\n"
    "        uint4 v = uint4(P[q * 4], P[q * 4 + 1], P[q * 4 + 2], P[q * 4 + 3]);\n"
    "        int4 c = int4(popcount(ps & v)) - int4(popcount(ms & v)); int s = c.x + c.y + c.z + c.w;\n"
    "        tot += q == 7 ? -128 * s : (s << q);\n"
    "    }\n"
    "    return float(tot);\n"
    "}\n"
    // gate+up of the hits: threadgroup = 4 rows x 16 groups; row r of hit h lives in slot sl[h]
    "kernel void gu_split(device const uint4 *mask [[buffer(0)]], device const float *scale [[buffer(1)]], device const uint *plx [[buffer(2)]],\n"
    "                     device const float *sx [[buffer(3)]], device float *y [[buffer(4)]], constant int *sl [[buffer(5)]],\n"
    "                     uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[64];\n"
    "    uint r = tg * 4 + t / 16, g = t % 16, h = r / 1024, lr = r % 1024; ulong srow = (ulong)sl[h] * 1024 + lr;\n"
    "    device const uint4 *m = mask + (srow * 16 + g) * 2;\n"
    "    part[t] = group_dot(m[0], m[1], plx + g * 32) * sx[g];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (g == 0) { float a = 0; for (uint i = 0; i < 16; i++) a += part[t + i]; y[r] = a * scale[srow]; }\n"
    "}\n"
    // down of the hits: threadgroup = 16 rows x 4 groups
    "kernel void dn_split(device const uint4 *mask [[buffer(0)]], device const float *scale [[buffer(1)]], device const uint *pla [[buffer(2)]],\n"
    "                     device const float *sa [[buffer(3)]], device float *y [[buffer(4)]], constant int *sl [[buffer(5)]],\n"
    "                     uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[64];\n"
    "    uint r = tg * 16 + t / 4, g = t % 4, h = r / 2048, lr = r % 2048; ulong srow = (ulong)sl[h] * 2048 + lr;\n"
    "    device const uint4 *m = mask + (srow * 4 + g) * 2;\n"
    "    part[t] = group_dot(m[0], m[1], pla + (h * 4 + g) * 32) * sa[h * 4 + g];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (g == 0) { float a = 0; for (uint i = 0; i < 4; i++) a += part[t + i]; y[r] = a * scale[srow]; }\n"
    "}\n"
    "kernel void planes_xm(device const float *x [[buffer(0)]], device uint *plw [[buffer(1)]], device float *sx [[buffer(2)]],\n"
    "                      uint b [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float xs[2048]; threadgroup uint pl[512]; threadgroup float s[16]; threadgroup float gm[64];\n"
    "    for (uint i = t; i < 2048; i += 64) xs[i] = x[b * 2048 + i];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    tg_planes(xs, 16, pl, s, gm, t);\n"
    "    for (uint i = t; i < 512; i += 64) plw[b * 512 + i] = pl[i];\n"
    "    if (t < 16) sx[b * 16 + t] = s[t];\n"
    "}\n"
    // tab: slots[nu], pair starts[nu+1], token of each pair[np]
    "kernel void gu_pairs(device const uint4 *mask [[buffer(0)]], device const float *scale [[buffer(1)]], device const uint *plx [[buffer(2)]],\n"
    "                     device const float *sx [[buffer(3)]], device float *y [[buffer(4)]], device const int *tab [[buffer(5)]],\n"
    "                     constant uint &nu [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[64][16];\n"
    "    uint u = tg / 256, rr = (tg % 256) * 4 + t / 16, g = t % 16;\n"
    "    int p0 = tab[nu + u], np = tab[nu + u + 1] - p0; ulong srow = (ulong)tab[u] * 1024 + rr;\n"
    "    device const uint4 *m = mask + (srow * 16 + g) * 2; uint4 ps = m[0], ms = m[1];\n"
    "    for (int k = 0; k < np; k++) { int b = tab[2 * nu + 1 + p0 + k]; part[t][k] = group_dot(ps, ms, plx + b * 512 + g * 32) * sx[b * 16 + g]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (g == 0) for (int k = 0; k < np; k++) { float a = 0; for (uint i = 0; i < 16; i++) a += part[t + i][k]; y[(ulong)(p0 + k) * 1024 + rr] = a * scale[srow]; }\n"
    "}\n"
    "kernel void dn_pairs(device const uint4 *mask [[buffer(0)]], device const float *scale [[buffer(1)]], device const uint *pla [[buffer(2)]],\n"
    "                     device const float *sa [[buffer(3)]], device float *y [[buffer(4)]], device const int *tab [[buffer(5)]],\n"
    "                     constant uint &nu [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[64][16];\n"
    "    uint u = tg / 128, rr = (tg % 128) * 16 + t / 4, g = t % 4;\n"
    "    int p0 = tab[nu + u], np = tab[nu + u + 1] - p0; ulong srow = (ulong)tab[u] * 2048 + rr;\n"
    "    device const uint4 *m = mask + (srow * 4 + g) * 2; uint4 ps = m[0], ms = m[1];\n"
    "    for (int k = 0; k < np; k++) { uint pr = p0 + k; part[t][k] = group_dot(ps, ms, pla + (pr * 4 + g) * 32) * sa[pr * 4 + g]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (g == 0) for (int k = 0; k < np; k++) { float a = 0; for (uint i = 0; i < 4; i++) a += part[t + i][k]; y[(ulong)(p0 + k) * 2048 + rr] = a * scale[srow]; }\n"
    "}\n"
    // packed codes (128-column groups, byte j = columns j, j+32, j+64, j+96) -> plus/minus masks; p = rows, groups, first slot row
    "kernel void conv(device const uchar *codes [[buffer(0)]], device uint4 *mask [[buffer(1)]], constant uint *p [[buffer(2)]],\n"
    "                 uint i [[thread_position_in_grid]]) {\n"
    "    uint rows = p[0], ng = p[1]; if (i >= rows * ng) return;\n"
    "    uint r = i / ng, g = i % ng;\n"
    "    device const uchar *q = codes + (ulong)r * ng * 32 + g * 32;\n"
    "    uint pl[4] = {0, 0, 0, 0}, mi[4] = {0, 0, 0, 0};\n"
    "    for (uint j = 0; j < 32; j++) {\n"
    "        uint b = q[j];\n"
    "        for (uint k = 0; k < 4; k++) { uint c = (b >> (2 * k)) & 3u; pl[k] |= (c == 2u ? 1u : 0u) << j; mi[k] |= (c == 0u ? 1u : 0u) << j; }\n"
    "    }\n"
    "    ulong o = ((ulong)(p[2] + r) * ng + g) * 2;\n"
    "    mask[o] = uint4(pl[0], pl[1], pl[2], pl[3]); mask[o + 1] = uint4(mi[0], mi[1], mi[2], mi[3]);\n"
    "}\n";

static id<MTLComputePipelineState> mkpipe(id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLComputePipelineState> p = [D newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err];
    if (!p) fprintf(stderr, "gpux: pipeline %s: %s\n", name, err.localizedDescription.UTF8String);
    return p;
}

// returns 0 when every layer got its slots
int gpux_init(int layers, int slots) {
    @autoreleasepool {
        D = MTLCreateSystemDefaultDevice();
        if (!D) { NSArray<id<MTLDevice>> *all = MTLCopyAllDevices(); if (all.count) D = all[0]; }
        if (!D) return -1;
        NSError *err = nil;
        id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { fprintf(stderr, "gpux: compile: %s\n", err.localizedDescription.UTF8String); return -1; }
        P_PX = mkpipe(lib, "planes_x"); P_PA = mkpipe(lib, "planes_act"); P_GU = mkpipe(lib, "gu_split"); P_DN = mkpipe(lib, "dn_split"); P_CONV = mkpipe(lib, "conv");
        P_PXM = mkpipe(lib, "planes_xm"); P_GUP = mkpipe(lib, "gu_pairs"); P_DNP = mkpipe(lib, "dn_pairs");
        if (!P_PX || !P_PA || !P_GU || !P_DN || !P_CONV || !P_PXM || !P_GUP || !P_DNP) return -1;
        QC = [D newCommandQueue];
        for (int L = 0; L < layers; L++) {
            GU_M[L] = [D newBufferWithLength:(size_t)slots * GU_ROWS * 16 * 32 options:MTLResourceStorageModePrivate];
            DN_M[L] = [D newBufferWithLength:(size_t)slots * DN_ROWS * 4 * 32 options:MTLResourceStorageModePrivate];
            GU_S[L] = [D newBufferWithLength:(size_t)slots * GU_ROWS * 4 options:MTLResourceStorageModePrivate];
            DN_S[L] = [D newBufferWithLength:(size_t)slots * DN_ROWS * 4 options:MTLResourceStorageModePrivate];
            if (!GU_M[L] || !DN_M[L] || !GU_S[L] || !DN_S[L]) { fprintf(stderr, "gpux: VRAM ran out at layer %d\n", L); return -1; }
        }
        XB = [D newBufferWithLength:2048 * 4 options:MTLResourceStorageModeManaged];
        PLX = [D newBufferWithLength:512 * 4 options:MTLResourceStorageModePrivate];
        SX = [D newBufferWithLength:16 * 4 options:MTLResourceStorageModePrivate];
        GUO = [D newBufferWithLength:MAXH * GU_ROWS * 4 options:MTLResourceStorageModePrivate];
        PLA = [D newBufferWithLength:MAXH * 128 * 4 options:MTLResourceStorageModePrivate];
        SA = [D newBufferWithLength:MAXH * 4 * 4 options:MTLResourceStorageModePrivate];
        YO = [D newBufferWithLength:MAXH * DN_ROWS * 4 options:MTLResourceStorageModeManaged];
        XBM = [D newBufferWithLength:MAXB * 2048 * 4 options:MTLResourceStorageModeManaged];
        PLXM = [D newBufferWithLength:MAXB * 512 * 4 options:MTLResourceStorageModePrivate];
        SXM = [D newBufferWithLength:MAXB * 16 * 4 options:MTLResourceStorageModePrivate];
        GUOM = [D newBufferWithLength:MAXP * GU_ROWS * 4 options:MTLResourceStorageModePrivate];
        PLAM = [D newBufferWithLength:MAXP * 128 * 4 options:MTLResourceStorageModePrivate];
        SAM = [D newBufferWithLength:MAXP * 4 * 4 options:MTLResourceStorageModePrivate];
        YOM = [D newBufferWithLength:MAXP * DN_ROWS * 4 options:MTLResourceStorageModeManaged];
        TAB = [D newBufferWithLength:1024 * 4 options:MTLResourceStorageModeManaged];
        for (int i = 0; i < NSTAGE; i++) STG[i] = [D newBufferWithLength:3 * CODE_MAT + (512 + 512 + 2048) * 4 options:MTLResourceStorageModeShared];
        fprintf(stderr, "gpux: %d layers x %d expert slots, %.0f MB of VRAM\n", layers, slots,
                layers * (double)slots * (GU_ROWS * 16 * 32 + DN_ROWS * 4 * 32 + (GU_ROWS + DN_ROWS) * 4) / 1e6);
    }
    return 0;
}

// codes and scales of one expert -> slot `slot` of layer L; queued behind earlier work, so later layers see it
void gpux_upload(int L, int slot, const uint8_t *gc, const float *gs, const uint8_t *uc, const float *us, const uint8_t *dc, const float *ds) {
    @autoreleasepool {
        int i = STG_I; STG_I = (STG_I + 1) % NSTAGE;
        if (STG_CB[i]) { [STG_CB[i] waitUntilCompleted]; STG_CB[i] = nil; }
        uint8_t *p = STG[i].contents;
        memcpy(p, gc, CODE_MAT); memcpy(p + CODE_MAT, uc, CODE_MAT); memcpy(p + 2 * CODE_MAT, dc, CODE_MAT);
        memcpy(p + 3 * CODE_MAT, gs, 2048); memcpy(p + 3 * CODE_MAT + 2048, us, 2048); memcpy(p + 3 * CODE_MAT + 4096, ds, 8192);
        id<MTLCommandBuffer> cb = [QC commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:P_CONV];
        uint32_t pg[3] = {512, 16, (uint32_t)slot * GU_ROWS}, pu[3] = {512, 16, (uint32_t)slot * GU_ROWS + 512}, pd[3] = {2048, 4, (uint32_t)slot * DN_ROWS};
        [e setBuffer:STG[i] offset:0 atIndex:0]; [e setBuffer:GU_M[L] offset:0 atIndex:1]; [e setBytes:pg length:sizeof pg atIndex:2];
        [e dispatchThreads:MTLSizeMake(512 * 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setBuffer:STG[i] offset:CODE_MAT atIndex:0]; [e setBytes:pu length:sizeof pu atIndex:2];
        [e dispatchThreads:MTLSizeMake(512 * 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setBuffer:STG[i] offset:2 * CODE_MAT atIndex:0]; [e setBuffer:DN_M[L] offset:0 atIndex:1]; [e setBytes:pd length:sizeof pd atIndex:2];
        [e dispatchThreads:MTLSizeMake(2048 * 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:STG[i] sourceOffset:3 * CODE_MAT toBuffer:GU_S[L] destinationOffset:(size_t)slot * GU_ROWS * 4 size:2048];
        [bl copyFromBuffer:STG[i] sourceOffset:3 * CODE_MAT + 2048 toBuffer:GU_S[L] destinationOffset:((size_t)slot * GU_ROWS + 512) * 4 size:2048];
        [bl copyFromBuffer:STG[i] sourceOffset:3 * CODE_MAT + 4096 toBuffer:DN_S[L] destinationOffset:(size_t)slot * DN_ROWS * 4 size:8192];
        [bl endEncoding];
        [cb commit];
        STG_CB[i] = cb;
    }
}

// start the hits of layer L on the card; the CPU works on the misses meanwhile; gpux_end waits and copies n x 2048 outputs
void *gpux_begin(int L, const float *x, int n, const int *slots) {
    @autoreleasepool {
        memcpy(XB.contents, x, 2048 * 4); [XB didModifyRange:NSMakeRange(0, 2048 * 4)];
        int sl[MAXH] = {0}; memcpy(sl, slots, n * sizeof(int));
        id<MTLCommandBuffer> cb = [QC commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:P_PX]; [e setBuffer:XB offset:0 atIndex:0]; [e setBuffer:PLX offset:0 atIndex:1]; [e setBuffer:SX offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setComputePipelineState:P_GU]; [e setBuffer:GU_M[L] offset:0 atIndex:0]; [e setBuffer:GU_S[L] offset:0 atIndex:1];
        [e setBuffer:PLX offset:0 atIndex:2]; [e setBuffer:SX offset:0 atIndex:3]; [e setBuffer:GUO offset:0 atIndex:4]; [e setBytes:sl length:sizeof sl atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((size_t)n * GU_ROWS / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setComputePipelineState:P_PA]; [e setBuffer:GUO offset:0 atIndex:0]; [e setBuffer:PLA offset:0 atIndex:1]; [e setBuffer:SA offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setComputePipelineState:P_DN]; [e setBuffer:DN_M[L] offset:0 atIndex:0]; [e setBuffer:DN_S[L] offset:0 atIndex:1];
        [e setBuffer:PLA offset:0 atIndex:2]; [e setBuffer:SA offset:0 atIndex:3]; [e setBuffer:YO offset:0 atIndex:4]; [e setBytes:sl length:sizeof sl atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((size_t)n * DN_ROWS / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:YO]; [bl endEncoding];
        [cb commit];
        return (__bridge_retained void *)cb;
    }
}
void gpux_end(void *h, float *out, int n) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)h;
        [cb waitUntilCompleted];
        memcpy(out, YO.contents, (size_t)n * DN_ROWS * 4);
    }
}

// a pack: nb input vectors; nu cached experts (slots), each with its run of pairs (pstart[nu+1], token index of each pair)
void *gpux_begin_multi(int L, const float *x, int nb, int nu, const int *slots, const int *pstart, const int *ptok) {
    @autoreleasepool {
        int np = pstart[nu];
        memcpy(XBM.contents, x, (size_t)nb * 2048 * 4); [XBM didModifyRange:NSMakeRange(0, (size_t)nb * 2048 * 4)];
        int *tb = TAB.contents;
        memcpy(tb, slots, nu * sizeof(int)); memcpy(tb + nu, pstart, (nu + 1) * sizeof(int)); memcpy(tb + 2 * nu + 1, ptok, np * sizeof(int));
        [TAB didModifyRange:NSMakeRange(0, (2 * nu + 1 + np) * sizeof(int))];
        uint32_t unu = nu;
        id<MTLCommandBuffer> cb = [QC commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:P_PXM]; [e setBuffer:XBM offset:0 atIndex:0]; [e setBuffer:PLXM offset:0 atIndex:1]; [e setBuffer:SXM offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setComputePipelineState:P_GUP]; [e setBuffer:GU_M[L] offset:0 atIndex:0]; [e setBuffer:GU_S[L] offset:0 atIndex:1];
        [e setBuffer:PLXM offset:0 atIndex:2]; [e setBuffer:SXM offset:0 atIndex:3]; [e setBuffer:GUOM offset:0 atIndex:4];
        [e setBuffer:TAB offset:0 atIndex:5]; [e setBytes:&unu length:4 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake((size_t)nu * 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setComputePipelineState:P_PA]; [e setBuffer:GUOM offset:0 atIndex:0]; [e setBuffer:PLAM offset:0 atIndex:1]; [e setBuffer:SAM offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(np, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e setComputePipelineState:P_DNP]; [e setBuffer:DN_M[L] offset:0 atIndex:0]; [e setBuffer:DN_S[L] offset:0 atIndex:1];
        [e setBuffer:PLAM offset:0 atIndex:2]; [e setBuffer:SAM offset:0 atIndex:3]; [e setBuffer:YOM offset:0 atIndex:4];
        [e setBuffer:TAB offset:0 atIndex:5]; [e setBytes:&unu length:4 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake((size_t)nu * 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:YOM]; [bl endEncoding];
        [cb commit];
        return (__bridge_retained void *)cb;
    }
}
void gpux_end_multi(void *h, float *out, int np) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)h;
        [cb waitUntilCompleted];
        memcpy(out, YOM.contents, (size_t)np * DN_ROWS * 4);
    }
}
