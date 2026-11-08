// gpu_k3.metal — the ternary matmuls with one weight row per thread and all NTF tokens of the tile (32, or 16 for an
// expert's last few pairs): a 2-bit code is decoded once and used for NTF multiply-adds, where gpu_kt.metal decodes each
// weight in 4 threads for 8 tokens each. The decode (shift, mask, convert, -1) was ~27% of the ALU work there; here it
// is ~9% (32 tokens). Activations sit in threadgroup memory as [column][token] and every lane reads the same address
// (a broadcast). Each thread keeps its own row's codes; the column loop stays a loop (kg[2] = 8) so the inner body
// fits the instruction cache.
#include <metal_stdlib>
using namespace metal;

#define FB(f) ((f) == 0 ? f0 : (f) == 1 ? f1 : (f) == 2 ? f2 : f3)
#define FILES device const uchar *f0 [[buffer(10)]], device const uchar *f1 [[buffer(11)]], device const uchar *f2 [[buffer(12)]], \
    device const uchar *f3 [[buffer(13)]]
constant uint NTF [[function_constant(0)]];

// NTF tokens x 64 columns into xs (transposed), each thread NTF floats of one token
#define LOADX(ptr) { device const float4 *xp = (device const float4 *)(ptr); \
    for (uint i = 0; i < NTF / 4; i++) { float4 v = xp[i]; uint k = kp + i * 4; \
        xs[k * NTF + lt] = v.x; xs[(k + 1) * NTF + lt] = v.y; xs[(k + 2) * NTF + lt] = v.z; xs[(k + 3) * NTF + lt] = v.w; } }
#define ZEROX() { for (uint i = 0; i < NTF; i++) xs[(kp + i) * NTF + lt] = 0; }

static inline float act1(float g, float u) { g = min(g, 7.0f); u = clamp(u, -7.0f, 7.0f); return g / (1.0f + exp(-g)) * u; }

// gate+up: tiles[i] = (used expert u, first pair, pairs <= NTF); 8 threadgroups per tile, 64 activation rows each.
// Thread t: gate row and up row rb*64 + t, all tokens. kg = (128-column groups, halves, j4 count = 8).
kernel void gu(device const float *X [[buffer(0)]], device const int *ptok [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],
               device const uint *loc [[buffer(3)]], device float *ACT [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,
               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32];
    threadgroup uint cg[64 * 9], cu[64 * 9];
    const uint SPL = 64 / NTF;
    uint4 T = tiles[tg / 8]; uint rb = tg % 8, p0 = T.y, n = T.z;
    device const uint *L = loc + T.x * 12;
    uint a = rb * 64 + t;
    device const uint4 *grow = (device const uint4 *)(FB(L[0]) + L[1] + (ulong)a * 512);
    device const uint4 *urow = (device const uint4 *)(FB(L[4]) + L[5] + (ulong)a * 512);
    device const float *gs = (device const float *)(FB(L[2]) + L[3]), *us = (device const float *)(FB(L[6]) + L[7]);
    uint lt = t / SPL, kp = (t % SPL) * NTF;
    int tok = lt < n ? ptok[p0 + lt] : -1;
    float4 G[8], U[8];
    for (uint i = 0; i < 8; i++) { G[i] = 0; U[i] = 0; }
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) {
            uint4 v0 = grow[g * 2], v1 = grow[g * 2 + 1], w0 = urow[g * 2], w1 = urow[g * 2 + 1];
            cg[t * 9] = v0.x; cg[t * 9 + 1] = v0.y; cg[t * 9 + 2] = v0.z; cg[t * 9 + 3] = v0.w;
            cg[t * 9 + 4] = v1.x; cg[t * 9 + 5] = v1.y; cg[t * 9 + 6] = v1.z; cg[t * 9 + 7] = v1.w;
            cu[t * 9] = w0.x; cu[t * 9 + 1] = w0.y; cu[t * 9 + 2] = w0.z; cu[t * 9 + 3] = w0.w;
            cu[t * 9 + 4] = w1.x; cu[t * 9 + 5] = w1.y; cu[t * 9 + 6] = w1.z; cu[t * 9 + 7] = w1.w;
        }
        if (tok >= 0) LOADX(X + (ulong)tok * 2048 + g * 128 + h * 64 + kp) else ZEROX()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j4 = 0; j4 < kg[2]; j4++) {
            uint gd = cg[t * 9 + j4], ud = cu[t * 9 + j4];
            for (uint jb = 0; jb < 4; jb++) for (uint sl = 0; sl < 2; sl++) {
                uint sh = jb * 8 + (h * 2 + sl) * 2;
                float wg = float((gd >> sh) & 3u) - 1.0f, wu = float((ud >> sh) & 3u) - 1.0f;
                threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * NTF);
                for (uint i = 0; i < NTF / 4; i++) { float4 xv = xq[i]; G[i] += wg * xv; U[i] += wu * xv; }
            }
        }
    }
    float sg = gs[a], su = us[a];
    for (uint i = 0; i < NTF / 4; i++) {
        float4 gg = G[i] * sg, uu = U[i] * su;
        for (uint c = 0; c < 4; c++) { uint q = i * 4 + c; if (q < n) ACT[(ulong)(p0 + q) * 512 + a] = act1(gg[c], uu[c]); }
    }
}

// down: 32 threadgroups per tile, 64 output rows each; thread t: row rb*64 + t, all pairs, times scale and routing weight
kernel void dn(device const float *ACT [[buffer(0)]], device const float *pw [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],
               device const uint *loc [[buffer(3)]], device float *DO [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,
               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32];
    threadgroup uint cs[64 * 9];
    const uint SPL = 64 / NTF;
    uint4 T = tiles[tg / 32]; uint rb = tg % 32, p0 = T.y, n = T.z;
    device const uint *L = loc + T.x * 12;
    uint r = rb * 64 + t;
    device const uint4 *crow = (device const uint4 *)(FB(L[8]) + L[9] + (ulong)r * 128);
    device const float *ds = (device const float *)(FB(L[10]) + L[11]);
    uint lt = t / SPL, kp = (t % SPL) * NTF;
    float4 A[8];
    for (uint i = 0; i < 8; i++) A[i] = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) {
            uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1];
            cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w;
            cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w;
        }
        if (lt < n) LOADX(ACT + (ulong)(p0 + lt) * 512 + g * 128 + h * 64 + kp) else ZEROX()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j4 = 0; j4 < kg[2]; j4++) {
            uint d = cs[t * 9 + j4];
            for (uint jb = 0; jb < 4; jb++) for (uint sl = 0; sl < 2; sl++) {
                uint sh = jb * 8 + (h * 2 + sl) * 2;
                float w = float((d >> sh) & 3u) - 1.0f;
                threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * NTF);
                for (uint i = 0; i < NTF / 4; i++) A[i] += w * xq[i];
            }
        }
    }
    float s = ds[r];
    for (uint i = 0; i < NTF / 4; i++) {
        float4 v = A[i] * s;
        for (uint c = 0; c < 4; c++) { uint q = i * 4 + c; if (q < n) DO[(ulong)(p0 + q) * 2048 + r] = v[c] * pw[p0 + q]; }
    }
}

// dense (attention projections): tiles (matrix, row block of 64, first token, tokens <= 32); dl[m] = (codes, scales,
// rows, out column); kg = (groups, halves, out stride, add into Y, j4 count)
kernel void dmm(device const float *X [[buffer(0)]], device const uint4 *tiles [[buffer(2)]], device const uint4 *dl [[buffer(3)]],
                device float *Y [[buffer(4)]], constant uint *kg [[buffer(5)]], device const uchar *W [[buffer(6)]],
                uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32];
    threadgroup uint cs[64 * 9];
    uint4 T = tiles[tg]; uint4 M = dl[T.x]; uint rb = T.y, p0 = T.z, n = T.w, K = kg[0] * 128, ys = kg[2];
    uint lt = t / 2, kp = (t & 1) * 32;
    uint r = rb * 64 + t;
    device const uint4 *crow = (device const uint4 *)(W + M.x + (ulong)r * (K / 4));
    device const float *sc = (device const float *)(W + M.y);
    float4 A[8];
    for (uint i = 0; i < 8; i++) A[i] = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) {
            uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1];
            cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w;
            cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w;
        }
        if (lt < n) { device const float4 *xp = (device const float4 *)(X + (ulong)(p0 + lt) * K + g * 128 + h * 64 + kp);
            for (uint i = 0; i < 8; i++) { float4 v = xp[i]; uint k = kp + i * 4;
                xs[k * 32 + lt] = v.x; xs[(k + 1) * 32 + lt] = v.y; xs[(k + 2) * 32 + lt] = v.z; xs[(k + 3) * 32 + lt] = v.w; } }
        else for (uint i = 0; i < 32; i++) xs[(kp + i) * 32 + lt] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j4 = 0; j4 < kg[4]; j4++) {
            uint d = cs[t * 9 + j4];
            for (uint jb = 0; jb < 4; jb++) for (uint sl = 0; sl < 2; sl++) {
                uint sh = jb * 8 + (h * 2 + sl) * 2;
                float w = float((d >> sh) & 3u) - 1.0f;
                threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * 32);
                for (uint i = 0; i < 8; i++) A[i] += w * xq[i];
            }
        }
    }
    float s = sc[r];
    for (uint i = 0; i < 8; i++) {
        float4 v = A[i] * s;
        for (uint c = 0; c < 4; c++) { uint q = i * 4 + c; if (q < n) { device float *o = Y + (ulong)(p0 + q) * ys + M.w + r; if (kg[3]) *o += v[c]; else *o = v[c]; } }
    }
}
