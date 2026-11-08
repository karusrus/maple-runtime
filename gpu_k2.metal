// gpu_k2.metal — the ternary matmuls of gpu_pre.m with cheaper decoding (default; GPRE_K1=1 keeps the old kernels).
// A 128-column group's codes (byte j holds columns j, j+32, j+64, j+96 at bits 0, 2, 4, 6) are unpacked once into
// words whose 4 bytes are the codes c in {0,1,2} of 4 consecutive columns: (word >> 2s) & 0x03030303. Then each weight
// costs one byte-to-float conversion instead of shift, mask, convert and subtract; the sums run over c = w + 1 and
// the token's own sum of x is taken off at the end: sum (c - 1) x = sum c x - sum x.
// Threadgroup of 64: 64 code rows x 32 tokens, each thread 4 rows x 8 tokens (as in gpu_pre.m).
#include <metal_stdlib>
using namespace metal;

#define FB(f) ((f) == 0 ? f0 : (f) == 1 ? f1 : (f) == 2 ? f2 : f3)
#define FILES device const uchar *f0 [[buffer(10)]], device const uchar *f1 [[buffer(11)]], device const uchar *f2 [[buffer(12)]], \
    device const uchar *f3 [[buffer(13)]]
#define CSW 33  // unpacked words per row and group (32) + 1 against bank conflicts
constant uint NTF [[function_constant(0)]];  // gu/dn: tokens per tile, 32 or 16 (a 16-token tile for an expert's last few pairs)

#define UNPACK(v0, v1) { \
    uint d0 = v0.x, d1 = v0.y, d2 = v0.z, d3 = v0.w, d4 = v1.x, d5 = v1.y, d6 = v1.z, d7 = v1.w; \
    for (uint s = 0; s < 4; s++) { \
        uint sh = 2 * s; threadgroup uint *o = cs + t * CSW + s * 8; \
        o[0] = (d0 >> sh) & 0x03030303u; o[1] = (d1 >> sh) & 0x03030303u; o[2] = (d2 >> sh) & 0x03030303u; o[3] = (d3 >> sh) & 0x03030303u; \
        o[4] = (d4 >> sh) & 0x03030303u; o[5] = (d5 >> sh) & 0x03030303u; o[6] = (d6 >> sh) & 0x03030303u; o[7] = (d7 >> sh) & 0x03030303u; \
    } }
#define STEP2(c, A, B) { float w = float(c); A += w * xa; B += w * xb; }  // a*b+c, not fma(): fused fma is quarter rate on Pitcairn
#define INNER2(h) INNERN(h, 32)
#define STEP1(c, A) { float w = float(c); A += w * xa; }
#define INNERN(h, NTK) \
    for (uint sl = 0; sl < 2; sl++) for (uint j4 = 0; j4 < 8; j4++) { \
        uint s8 = ((h) * 2 + sl) * 8 + j4; \
        uint c0 = cs[r0 * CSW + s8], c1 = cs[r1 * CSW + s8], c2 = cs[r2 * CSW + s8], c3 = cs[r3 * CSW + s8]; \
        for (uint jb = 0; jb < 4; jb++) { \
            threadgroup const float4 *xq = (threadgroup const float4 *)(xs + (sl * 32 + j4 * 4 + jb) * (NTK) + tq * ((NTK) / 4)); \
            float4 xa = xq[0]; uint sh = jb * 8; \
            if ((NTK) == 32) { float4 xb = xq[1]; \
                STEP2((c0 >> sh) & 0xFFu, A0, B0) STEP2((c1 >> sh) & 0xFFu, A1, B1) \
                STEP2((c2 >> sh) & 0xFFu, A2, B2) STEP2((c3 >> sh) & 0xFFu, A3, B3) \
            } else { STEP1((c0 >> sh) & 0xFFu, A0) STEP1((c1 >> sh) & 0xFFu, A1) STEP1((c2 >> sh) & 0xFFu, A2) STEP1((c3 >> sh) & 0xFFu, A3) } \
        } \
    }
// the activations of one token half-group (32 floats) into xs, transposed; their sum kept for the correction
#define LOADX(ptr) LOADXN(ptr, 32)
#define ZEROX() ZEROXN(32)
#define SUMS() SUMSN(32)
#define LOADXN(ptr, NTK) { \
    device const float4 *xp = (device const float4 *)(ptr); \
    for (uint i = 0; i < (NTK) / 4; i++) { float4 v = xp[i]; uint k = kp + i * 4; xsum += (v.x + v.y) + (v.z + v.w); \
        xs[k * (NTK) + lt] = v.x; xs[(k + 1) * (NTK) + lt] = v.y; xs[(k + 2) * (NTK) + lt] = v.z; xs[(k + 3) * (NTK) + lt] = v.w; } }
#define ZEROXN(NTK) { for (uint i = 0; i < (NTK); i++) xs[(kp + i) * (NTK) + lt] = 0; }
// tsum[q] = sum of x of token q of the tile (the 64 / NTK threads that loaded its columns)
#define SUMSN(NTK) { tpart[t] = xsum; threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (t < (NTK)) { float z = 0; for (uint i = 0; i < 64 / (NTK); i++) z += tpart[t * (64 / (NTK)) + i]; tsum[t] = z; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); }

static inline float act1(float g, float u) { g = min(g, 7.0f); u = clamp(u, -7.0f, 7.0f); return g / (1.0f + exp(-g)) * u; }

// tiles[i] = (used expert u, first pair, pairs <= 32); loc[u*12..] = (file, offset) of gate codes, gate scales, up codes,
// up scales, down codes, down scales. gate rows rb*32 .. +32 (threads 0..31 load) and the same up rows (32..63).
kernel void gu(device const float *X [[buffer(0)]], device const int *ptok [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],
               device const uint *loc [[buffer(3)]], device float *ACT [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,
               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * CSW]; threadgroup float tpart[64], tsum[32];
    uint4 T = tiles[tg / 16]; uint rb = tg % 16, p0 = T.y, n = T.z;
    device const uint *L = loc + T.x * 12;
    device const uchar *gb = FB(L[0]) + L[1], *ub = FB(L[4]) + L[5];
    device const float *gs = (device const float *)(FB(L[2]) + L[3]), *us = (device const float *)(FB(L[6]) + L[7]);
    const uint TPT = NTF / 4, SPL = 64 / NTF;  // tokens per thread; threads per token when loading
    uint rp = t % 16, tq = t / 16, lt = t / SPL, kp = (t % SPL) * NTF;
    uint r0 = rp, r1 = rp + 16, r2 = 32 + rp, r3 = 48 + rp;
    int tok = lt < n ? ptok[p0 + lt] : -1;
    device const uint4 *crow = (device const uint4 *)((t < 32 ? gb : ub) + (ulong)(rb * 32 + (t & 31)) * 512);
    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;
    float xsum = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1]; UNPACK(v0, v1) }
        if (tok >= 0) LOADXN(X + (ulong)tok * 2048 + g * 128 + h * 64 + kp, NTF) else ZEROXN(NTF)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INNERN(h, NTF)
    }
    SUMSN(NTF)
    uint a = rb * 32 + rp;
    float sg0 = gs[a], sg1 = gs[a + 16], su0 = us[a], su1 = us[a + 16];
#define OUT(i, c, GA, GB, UA, UB) { uint q = tq * TPT + (i); if ((i) < TPT && q < n) { float sq = tsum[q]; device float *o = ACT + (ulong)(p0 + q) * 512 + a; \
        o[0] = act1((GA.c - sq) * sg0, (UA.c - sq) * su0); o[16] = act1((GB.c - sq) * sg1, (UB.c - sq) * su1); } }
    OUT(0, x, A0, A1, A2, A3) OUT(1, y, A0, A1, A2, A3) OUT(2, z, A0, A1, A2, A3) OUT(3, w, A0, A1, A2, A3)
    OUT(4, x, B0, B1, B2, B3) OUT(5, y, B0, B1, B2, B3) OUT(6, z, B0, B1, B2, B3) OUT(7, w, B0, B1, B2, B3)
}

// down: 64 output rows x 32 pairs, times the row scale and the pair's routing weight
kernel void dn(device const float *ACT [[buffer(0)]], device const float *pw [[buffer(1)]], device const uint4 *tiles [[buffer(2)]],
               device const uint *loc [[buffer(3)]], device float *DO [[buffer(4)]], constant uint *kg [[buffer(5)]], FILES,
               uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * CSW]; threadgroup float tpart[64], tsum[32];
    uint4 T = tiles[tg / 32]; uint rb = tg % 32, p0 = T.y, n = T.z;
    device const uint *L = loc + T.x * 12;
    device const uchar *db = FB(L[8]) + L[9]; device const float *ds = (device const float *)(FB(L[10]) + L[11]);
    const uint TPT = NTF / 4, SPL = 64 / NTF;
    uint rq = t % 16, tq = t / 16, lt = t / SPL, kp = (t % SPL) * NTF;
    uint r0 = rq, r1 = rq + 16, r2 = rq + 32, r3 = rq + 48;
    device const uint4 *crow = (device const uint4 *)(db + (ulong)(rb * 64 + t) * 128);
    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;
    float xsum = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1]; UNPACK(v0, v1) }
        if (lt < n) LOADXN(ACT + (ulong)(p0 + lt) * 512 + g * 128 + h * 64 + kp, NTF) else ZEROXN(NTF)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INNERN(h, NTF)
    }
    SUMSN(NTF)
    uint r = rb * 64 + rq;
    float s0 = ds[r], s1 = ds[r + 16], s2 = ds[r + 32], s3 = ds[r + 48];
#define DOUT(i, c, AA, BB, CC, DD) { uint q = tq * TPT + (i); if ((i) < TPT && q < n) { float sq = tsum[q], wq = pw[p0 + q]; device float *o = DO + (ulong)(p0 + q) * 2048 + r; \
        o[0] = (AA.c - sq) * s0 * wq; o[16] = (BB.c - sq) * s1 * wq; o[32] = (CC.c - sq) * s2 * wq; o[48] = (DD.c - sq) * s3 * wq; } }
    DOUT(0, x, A0, A1, A2, A3) DOUT(1, y, A0, A1, A2, A3) DOUT(2, z, A0, A1, A2, A3) DOUT(3, w, A0, A1, A2, A3)
    DOUT(4, x, B0, B1, B2, B3) DOUT(5, y, B0, B1, B2, B3) DOUT(6, z, B0, B1, B2, B3) DOUT(7, w, B0, B1, B2, B3)
}

// dense: tiles (matrix, row block, first token, tokens); dl[m] = (codes, scales, rows, out column); kg[3]: add into Y
kernel void dmm(device const float *X [[buffer(0)]], device const uint4 *tiles [[buffer(2)]], device const uint4 *dl [[buffer(3)]],
                device float *Y [[buffer(4)]], constant uint *kg [[buffer(5)]], device const uchar *W [[buffer(6)]],
                uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float xs[64 * 32]; threadgroup uint cs[64 * CSW]; threadgroup float tpart[64], tsum[32];
    uint4 T = tiles[tg]; uint4 M = dl[T.x]; uint rb = T.y, p0 = T.z, n = T.w, K = kg[0] * 128, ys = kg[2];
    uint rq = t % 16, tq = t / 16, lt = t / 2, kp = (t & 1) * 32;
    uint r0 = rq, r1 = rq + 16, r2 = rq + 32, r3 = rq + 48;
    device const uint4 *crow = (device const uint4 *)(W + M.x + (ulong)(rb * 64 + t) * (K / 4));
    device const float *sc = (device const float *)(W + M.y);
    float4 A0 = 0, B0 = 0, A1 = 0, B1 = 0, A2 = 0, B2 = 0, A3 = 0, B3 = 0;
    float xsum = 0;
    for (uint g = 0; g < kg[0]; g++) for (uint h = 0; h < kg[1]; h++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h == 0) { uint4 v0 = crow[g * 2], v1 = crow[g * 2 + 1]; UNPACK(v0, v1) }
        if (lt < n) LOADX(X + (ulong)(p0 + lt) * K + g * 128 + h * 64 + kp) else ZEROX()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INNER2(h)
    }
    SUMS()
    uint r = rb * 64 + rq;
    float s0 = sc[r], s1 = sc[r + 16], s2 = sc[r + 32], s3 = sc[r + 48];
#define YOUT(i, c, AA, BB, CC, DD) { uint q = tq * 8 + (i); if (q < n) { float sq = tsum[q]; device float *o = Y + (ulong)(p0 + q) * ys + M.w + r; \
        if (kg[3]) { o[0] += (AA.c - sq) * s0; o[16] += (BB.c - sq) * s1; o[32] += (CC.c - sq) * s2; o[48] += (DD.c - sq) * s3; } \
        else { o[0] = (AA.c - sq) * s0; o[16] = (BB.c - sq) * s1; o[32] = (CC.c - sq) * s2; o[48] = (DD.c - sq) * s3; } } }
    YOUT(0, x, A0, A1, A2, A3) YOUT(1, y, A0, A1, A2, A3) YOUT(2, z, A0, A1, A2, A3) YOUT(3, w, A0, A1, A2, A3)
    YOUT(4, x, B0, B1, B2, B3) YOUT(5, y, B0, B1, B2, B3) YOUT(6, z, B0, B1, B2, B3) YOUT(7, w, B0, B1, B2, B3)
}
