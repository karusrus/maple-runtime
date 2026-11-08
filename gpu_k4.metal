// gpu_k4.metal — gpu_kt.metal with cheaper decoding in registers (GPRE_K4=1): the packed codes stay in threadgroup
// memory as in gpu_kt (2.3 KB), and per code word and field s, (word >> 2s) & 0x03030303 gives the codes c in {0,1,2}
// of 4 consecutive columns as bytes, each turned into a float by one byte conversion: 1.5 instructions per weight
// instead of 3. The sums run over c = w + 1; the token's sum of x is taken off at the end.
// Threadgroup of 64: 64 code rows x NTF tokens; each thread 4 rows x NTF/4 tokens.
#include <metal_stdlib>
using namespace metal;

#define FB(f) ((f) == 0 ? f0 : (f) == 1 ? f1 : (f) == 2 ? f2 : f3)
#define FILES device const uchar *f0 [[buffer(10)]], device const uchar *f1 [[buffer(11)]], device const uchar *f2 [[buffer(12)]], \
    device const uchar *f3 [[buffer(13)]]
constant uint NTF [[function_constant(0)]];

// weight code c in {0,1,2} means c - 1; byte j of a 128-column group holds columns j, j+32, j+64, j+96
#define STEP(m, jb, A, B) { float w = float(((m) >> ((jb) * 8)) & 0xFFu); A += w * xa; if (NTF == 32) B += w * xb; }
#define INNER(h) \
    for (uint j4 = 0; j4 < 8; j4++) { \
        uint c0 = cs[r0 * 9 + j4], c1 = cs[r1 * 9 + j4], c2 = cs[r2 * 9 + j4], c3 = cs[r3 * 9 + j4]; \
        for (uint sl = 0; sl < 2; sl++) { \
            uint s2 = ((h) * 2 + sl) * 2; \
            uint m0 = (c0 >> s2) & 0x03030303u, m1 = (c1 >> s2) & 0x03030303u, m2 = (c2 >> s2) & 0x03030303u, m3 = (c3 >> s2) & 0x03030303u; \
            for (uint jb = 0; jb < 4; jb++) { \
                threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * NTF + tq * TPT); \
                float4 xa = xq[0], xb = NTF == 32 ? xq[1] : float4(0.0f); \
                STEP(m0, jb, A0, B0) STEP(m1, jb, A1, B1) STEP(m2, jb, A2, B2) STEP(m3, jb, A3, B3) \
            } \
        } \
    }
// tsum[q] = sum of x of token q of the tile (the 64 / NTF threads that loaded its columns)
#define SUMS() { tpart[t] = xsum; threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (t < NTF) { float z = 0; for (uint i = 0; i < 64 / NTF; i++) z += tpart[t * (64 / NTF) + i]; tsum[t] = z; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); }
#define CODES() { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1]; \
    cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w; \
    cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w; }
// NTF tokens x 64 columns into xs (transposed), each thread NTF floats of one token
#define LOADX(ptr) { device const float4 *xp = (device const float4 *)(ptr); \
    for (uint i = 0; i < NTF / 4; i++) { float4 v = xp[i]; uint k = kp + i * 4; xsum += (v.x + v.y) + (v.z + v.w); \
        xs[k * NTF + lt] = v.x; xs[(k + 1) * NTF + lt] = v.y; xs[(k + 2) * NTF + lt] = v.z; xs[(k + 3) * NTF + lt] = v.w; } }
#define ZEROX() { for (uint i = 0; i < NTF; i++) xs[(kp + i) * NTF + lt] = 0; }

static inline float act1(float g, float u) { g = min(g, 7.0f); u = clamp(u, -7.0f, 7.0f); return g / (1.0f + exp(-g)) * u; }

// tiles[i] = (used expert u, first pair, pairs <= NTF); loc[u*12..] = (file, offset) of gate codes, gate scales,
// up codes, up scales, down codes, down scales. Gate rows rb*32 .. +32 (threads 0..31 load) and the same up rows.
kernel void gu(device const float *X [[buffer(0)]], device const int *ptok [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],
               device const uint *loc [[buffer(3)]], device float *ACT [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,
               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * 9]; threadgroup float tpart[64], tsum[32];
    const uint TPT = NTF / 4, SPL = 64 / NTF;
    uint4 T = tiles[tg / 16]; uint rb = tg % 16, p0 = T.y, n = T.z;
    device const uint *L = loc + T.x * 12;
    device const uchar *gb = FB(L[0]) + L[1], *ub = FB(L[4]) + L[5];
    device const float *gs = (device const float *)(FB(L[2]) + L[3]), *us = (device const float *)(FB(L[6]) + L[7]);
    uint rp = t % 16, tq = t / 16, lt = t / SPL, kp = (t % SPL) * NTF;
    uint r0 = rp, r1 = rp + 16, r2 = 32 + rp, r3 = 48 + rp;
    int tok = lt < n ? ptok[p0 + lt] : -1;
    device const uint4 *crow = (device const uint4 *)((t < 32 ? gb : ub) + (ulong)(rb * 32 + (t & 31)) * 512);
    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;
    float xsum = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) CODES()
        if (tok >= 0) LOADX(X + (ulong)tok * 2048 + g * 128 + h * 64 + kp) else ZEROX()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INNER(h)
    }
    SUMS()
    uint a = rb * 32 + rp;
    float sg0 = gs[a], sg1 = gs[a + 16], su0 = us[a], su1 = us[a + 16];
#define OUT(i, c, GA, GB, UA, UB) { uint q = tq * TPT + (i); if ((i) < TPT && q < n) { float sq = tsum[q]; device float *o = ACT + (ulong)(p0 + q) * 512 + a; \
        o[0] = act1((GA.c - sq) * sg0, (UA.c - sq) * su0); o[16] = act1((GB.c - sq) * sg1, (UB.c - sq) * su1); } }
    OUT(0, x, A0, A1, A2, A3) OUT(1, y, A0, A1, A2, A3) OUT(2, z, A0, A1, A2, A3) OUT(3, w, A0, A1, A2, A3)
    OUT(4, x, B0, B1, B2, B3) OUT(5, y, B0, B1, B2, B3) OUT(6, z, B0, B1, B2, B3) OUT(7, w, B0, B1, B2, B3)
}

// down: 64 output rows x NTF pairs, times the row scale and the pair's routing weight
kernel void dn(device const float *ACT [[buffer(0)]], device const float *pw [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],
               device const uint *loc [[buffer(3)]], device float *DO [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,
               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * 9]; threadgroup float tpart[64], tsum[32];
    const uint TPT = NTF / 4, SPL = 64 / NTF;
    uint4 T = tiles[tg / 32]; uint rb = tg % 32, p0 = T.y, n = T.z;
    device const uint *L = loc + T.x * 12;
    device const uchar *db = FB(L[8]) + L[9]; device const float *ds = (device const float *)(FB(L[10]) + L[11]);
    uint rq = t % 16, tq = t / 16, lt = t / SPL, kp = (t % SPL) * NTF;
    uint r0 = rq, r1 = rq + 16, r2 = rq + 32, r3 = rq + 48;
    device const uint4 *crow = (device const uint4 *)(db + (ulong)(rb * 64 + t) * 128);
    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;
    float xsum = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) CODES()
        if (lt < n) LOADX(ACT + (ulong)(p0 + lt) * 512 + g * 128 + h * 64 + kp) else ZEROX()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INNER(h)
    }
    SUMS()
    uint r = rb * 64 + rq;
    float s0 = ds[r], s1 = ds[r + 16], s2 = ds[r + 32], s3 = ds[r + 48];
#define DOUT(i, c, AA, BB, CC, DD) { uint q = tq * TPT + (i); if ((i) < TPT && q < n) { float sq = tsum[q], wq = pw[p0 + q]; device float *o = DO + (ulong)(p0 + q) * 2048 + r; \
        o[0] = (AA.c - sq) * s0 * wq; o[16] = (BB.c - sq) * s1 * wq; o[32] = (CC.c - sq) * s2 * wq; o[48] = (DD.c - sq) * s3 * wq; } }
    DOUT(0, x, A0, A1, A2, A3) DOUT(1, y, A0, A1, A2, A3) DOUT(2, z, A0, A1, A2, A3) DOUT(3, w, A0, A1, A2, A3)
    DOUT(4, x, B0, B1, B2, B3) DOUT(5, y, B0, B1, B2, B3) DOUT(6, z, B0, B1, B2, B3) DOUT(7, w, B0, B1, B2, B3)
}

// dense (attention projections) with the same decoding: tiles (matrix, row block, first token, tokens <= 32);
// dl[m] = (codes, scales, rows, out column); kg[3]: add into Y. 32-token tiles only (NTF is not used here).
#define STEP32(m, jb, A, B) { float w = float(((m) >> ((jb) * 8)) & 0xFFu); A += w * xa; B += w * xb; }
kernel void dmm(device const float *X [[buffer(0)]], device const uint4 *tiles [[buffer(2)]], device const uint4 *dl [[buffer(3)]],
                device float *Y [[buffer(4)]], constant uint *kg [[buffer(5)]], device const uchar *W [[buffer(6)]],
                uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * 9]; threadgroup float tpart[64], tsum[32];
    uint4 T = tiles[tg]; uint4 M = dl[T.x]; uint rb = T.y, p0 = T.z, n = T.w, K = kg[0] * 128, ys = kg[2];
    uint rq = t % 16, tq = t / 16, lt = t / 2, kp = (t & 1) * 32;
    uint r0 = rq, r1 = rq + 16, r2 = rq + 32, r3 = rq + 48;
    device const uint4 *crow = (device const uint4 *)(W + M.x + (ulong)(rb * 64 + t) * (K / 4));
    device const float *sc = (device const float *)(W + M.y);
    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;
    float xsum = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1];
            cs[t * 9] = v0.x; cs[t * 9 + 1] = v0.y; cs[t * 9 + 2] = v0.z; cs[t * 9 + 3] = v0.w;
            cs[t * 9 + 4] = v1.x; cs[t * 9 + 5] = v1.y; cs[t * 9 + 6] = v1.z; cs[t * 9 + 7] = v1.w; }
        if (lt < n) { device const float4 *xp = (device const float4 *)(X + (ulong)(p0 + lt) * K + g * 128 + h * 64 + kp);
            for (uint i = 0; i < 8; i++) { float4 v = xp[i]; uint k = kp + i * 4; xsum += (v.x + v.y) + (v.z + v.w);
                xs[k * 32 + lt] = v.x; xs[(k + 1) * 32 + lt] = v.y; xs[(k + 2) * 32 + lt] = v.z; xs[(k + 3) * 32 + lt] = v.w; } }
        else for (uint i = 0; i < 32; i++) xs[(kp + i) * 32 + lt] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j4 = 0; j4 < 8; j4++) {
            uint c0 = cs[r0 * 9 + j4], c1 = cs[r1 * 9 + j4], c2 = cs[r2 * 9 + j4], c3 = cs[r3 * 9 + j4];
            for (uint sl = 0; sl < 2; sl++) {
                uint s2 = (h * 2 + sl) * 2;
                uint m0 = (c0 >> s2) & 0x03030303u, m1 = (c1 >> s2) & 0x03030303u, m2 = (c2 >> s2) & 0x03030303u, m3 = (c3 >> s2) & 0x03030303u;
                for (uint jb = 0; jb < 4; jb++) {
                    threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * 32 + tq * 8);
                    float4 xa = xq[0], xb = xq[1];
                    STEP32(m0, jb, A0, B0) STEP32(m1, jb, A1, B1) STEP32(m2, jb, A2, B2) STEP32(m3, jb, A3, B3)
                }
            }
        }
    }
    tpart[t] = xsum; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) tsum[t] = tpart[2 * t] + tpart[2 * t + 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = rb * 64 + rq;
    float s0 = sc[r], s1 = sc[r + 16], s2 = sc[r + 32], s3 = sc[r + 48];
#define YOUT(i, c, AA, BB, CC, DD) { uint q = tq * 8 + (i); if (q < n) { float sq = tsum[q]; device float *o = Y + (ulong)(p0 + q) * ys + M.w + r; \
        if (kg[3]) { o[0] += (AA.c - sq) * s0; o[16] += (BB.c - sq) * s1; o[32] += (CC.c - sq) * s2; o[48] += (DD.c - sq) * s3; } \
        else { o[0] = (AA.c - sq) * s0; o[16] = (BB.c - sq) * s1; o[32] = (CC.c - sq) * s2; o[48] = (DD.c - sq) * s3; } } }
    YOUT(0, x, A0, A1, A2, A3) YOUT(1, y, A0, A1, A2, A3) YOUT(2, z, A0, A1, A2, A3) YOUT(3, w, A0, A1, A2, A3)
    YOUT(4, x, B0, B1, B2, B3) YOUT(5, y, B0, B1, B2, B3) YOUT(6, z, B0, B1, B2, B3) YOUT(7, w, B0, B1, B2, B3)
}
