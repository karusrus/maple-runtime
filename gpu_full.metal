// gpu_full.metal — the rest of a layer on the card while the prompt is read (--gpupre N): norms, q/k norm with rope,
// causal attention over the card's own fp16 key/value cache, the router, and the residual sums. With the ternary
// matmuls in gpu_pre.m this keeps the hidden state on the card for the whole chunk; the CPU only picks the 8 experts.
// No SIMD-group functions: reductions go through threadgroup memory, so any SIMD width works (AMD GCN is 64 wide).
#include <metal_stdlib>
using namespace metal;

#define H 2048
#define HD 128

// y = x / rms(x) * w, one threadgroup of 256 per token
kernel void rms(device const float *X [[buffer(0)]], device const float *W [[buffer(1)]], device float *Y [[buffer(2)]],
                uint b [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float red[256];
    device const float *x = X + (ulong)b * H;
    float v[8], s = 0;
    for (uint i = 0; i < 8; i++) { v[i] = x[t + i * 256]; s += v[i] * v[i]; }
    red[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 128; w > 0; w >>= 1) {
        if (t < w) red[t] += red[t + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float k = 1.0f / sqrt(red[0] / (float)H + 1e-6f);
    device float *y = Y + (ulong)b * H;
    for (uint i = 0; i < 8; i++) y[t + i * 256] = v[i] * k * W[t + i * 256];
}

// p: (first position, global layer, tokens, -)
// per token and head (0..15 query, 16..19 key, 20..23 value): rms norm of q and k, rope on local layers,
// q to QO (fp32), k and v into the layer's cache at their positions (fp16) and into KO for the CPU's own cache
kernel void qkn(device const float *QKV [[buffer(0)]], device const float *QN [[buffer(1)]], device const float *KN [[buffer(2)]],
                device const float2 *CS [[buffer(3)]], device float *QO [[buffer(4)]], device half *KG [[buffer(5)]],
                device half *VG [[buffer(6)]], device half *KO [[buffer(7)]], constant uint4 &p [[buffer(8)]],
                uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
    uint t = tid.x;
    threadgroup float red[64], v[64];
    uint b = tg.x, h = tg.y, pos = p.x + b;
    device const float *src = QKV + (ulong)b * 3072 + h * HD;  // q 0..2047, k 2048..2559, v 2560..3071
    float a0 = src[t], a1 = src[t + 64];
    if (h >= 20) {
        uint c = (h - 20) * HD;
        half h0 = half(a0), h1 = half(a1);
        VG[(ulong)pos * 512 + c + t] = h0; VG[(ulong)pos * 512 + c + t + 64] = h1;
        KO[(ulong)b * 1024 + 512 + c + t] = h0; KO[(ulong)b * 1024 + 512 + c + t + 64] = h1;
        return;
    }
    red[t] = a0 * a0 + a1 * a1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 32; w > 0; w >>= 1) {
        if (t < w) red[t] += red[t + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float k = 1.0f / sqrt(red[0] / (float)HD + 1e-6f);
    device const float *nw = h < 16 ? QN : KN;
    a0 = a0 * k * nw[t]; a1 = a1 * k * nw[t + 64];
    if (p.y == 0) {  // rope on dims 0..63, pairs (i, i + 32), the CPU's table of (cos, sin) per position
        v[t] = a0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 32) { float2 c = CS[(ulong)pos * 32 + t]; a0 = v[t] * c.x - v[t + 32] * c.y; }
        else { float2 c = CS[(ulong)pos * 32 + t - 32]; a0 = v[t] * c.x + v[t - 32] * c.y; }
    }
    if (h < 16) {
        QO[(ulong)b * H + h * HD + t] = a0; QO[(ulong)b * H + h * HD + t + 64] = a1;
    } else {
        uint c = (h - 16) * HD;
        half h0 = half(a0), h1 = half(a1);
        KG[(ulong)pos * 512 + c + t] = h0; KG[(ulong)pos * 512 + c + t + 64] = h1;
        KO[(ulong)b * 1024 + c + t] = h0; KO[(ulong)b * 1024 + c + t + 64] = h1;
    }
}

// causal attention, 8 tokens x the 4 query heads of one key/value head per threadgroup of 128 = 32 query rows;
// keys in blocks of 32 through threadgroup memory, online softmax. p: (first position, global, tokens, window)
// A token at position q sees keys max(0, q - window) .. q on local layers, 0 .. q on global ones (as the CPU does).
#define QS 132   // row strides with padding against bank conflicts
#define KS 136
kernel void att(device const float *QO [[buffer(0)]], device const half *KG [[buffer(1)]], device const half *VG [[buffer(2)]],
                device float *ATT [[buffer(3)]], constant uint4 &p [[buffer(4)]],
                uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
    uint t = tid.x;
    threadgroup float qs[32 * QS];
    threadgroup half kv[32 * KS];
    threadgroup float ps[32 * 33];
    threadgroup float red[128];
    uint b0 = tg.x * 8, kvh = tg.y, nb = p.z, win = p.w;
    uint ntok = min(8u, nb - b0);
    for (uint i = 0; i < 32; i++) {  // the 32 query rows: row r = token (r / 4), head kvh * 4 + r % 4
        uint idx = t + i * 128, r = idx / HD, d = idx % HD, tk = r / 4;
        qs[r * QS + d] = tk < ntok ? QO[(ulong)(b0 + tk) * H + (kvh * 4 + r % 4) * HD + d] : 0.0f;
    }
    uint r = t / 4, sub = t % 4, tk = r / 4;
    bool rowok = tk < ntok;
    uint mypos = p.x + b0 + min(tk, ntok - 1);
    uint mylo = p.y ? 0 : (mypos > win ? mypos - win : 0);
    uint first = p.x + b0, last = p.x + b0 + ntok - 1;
    uint lo = p.y ? 0 : (first > win ? first - win : 0);
    const float scale = 1.0f / sqrt((float)HD);
    float m = 0, l = 0; bool any = false;
    float o[32];
    for (uint j = 0; j < 32; j++) o[j] = 0;
    for (uint k0 = lo; k0 <= last; k0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++) {  // keys k0 .. k0 + 31 of this head, beyond the last token zero
            uint idx = t + i * 128, kk = idx / 32, d4 = (idx % 32) * 4, kp = k0 + kk;
            half4 v4 = kp <= last ? *(device const half4 *)(KG + (ulong)kp * 512 + kvh * HD + d4) : half4(0.0h);
            *(threadgroup half4 *)(kv + kk * KS + d4) = v4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[8];
        for (uint i = 0; i < 8; i++) s[i] = 0;
        for (uint d = 0; d < HD; d += 4) {
            float4 q4 = *(threadgroup const float4 *)(qs + r * QS + d);
            for (uint i = 0; i < 8; i++) {
                float4 k4 = float4(*(threadgroup const half4 *)(kv + (sub + 4 * i) * KS + d));
                s[i] += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
        }
        bool ok[8]; float lm = -3.0e38f; bool lany = false;
        for (uint i = 0; i < 8; i++) {
            uint kp = k0 + sub + 4 * i;
            ok[i] = rowok && kp >= mylo && kp <= mypos;
            s[i] *= scale;
            if (ok[i]) { lm = max(lm, s[i]); lany = true; }
        }
        red[t] = lany ? lm : -3.0e38f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bm = max(max(red[r * 4], red[r * 4 + 1]), max(red[r * 4 + 2], red[r * 4 + 3]));
        bool bany = bm > -1.0e38f;
        float mn = any ? (bany ? max(m, bm) : m) : bm;
        float sc = any ? exp(m - mn) : 0.0f;
        float ls = 0;
        for (uint i = 0; i < 8; i++) {
            float pv = ok[i] ? exp(s[i] - mn) : 0.0f;
            ps[r * 33 + sub + 4 * i] = pv; ls += pv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[t] = ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bs = red[r * 4] + red[r * 4 + 1] + red[r * 4 + 2] + red[r * 4 + 3];
        if (bany) {
            l = l * sc + bs;
            for (uint j = 0; j < 32; j++) o[j] *= sc;
            m = mn; any = true;
        }
        for (uint i = 0; i < 8; i++) {  // values of the same keys
            uint idx = t + i * 128, kk = idx / 32, d4 = (idx % 32) * 4, kp = k0 + kk;
            half4 v4 = kp <= last ? *(device const half4 *)(VG + (ulong)kp * 512 + kvh * HD + d4) : half4(0.0h);
            *(threadgroup half4 *)(kv + kk * KS + d4) = v4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (bany) for (uint kk = 0; kk < 32; kk++) {
            float pv = ps[r * 33 + kk];
            threadgroup const half4 *vr = (threadgroup const half4 *)(kv + kk * KS + sub * 32);
            for (uint j = 0; j < 8; j++) {
                float4 v4 = float4(vr[j]);
                o[j * 4] += pv * v4.x; o[j * 4 + 1] += pv * v4.y; o[j * 4 + 2] += pv * v4.z; o[j * 4 + 3] += pv * v4.w;
            }
        }
    }
    if (rowok) {
        float inv = 1.0f / l;
        device float *dst = ATT + (ulong)(b0 + tk) * H + (kvh * 4 + r % 4) * HD + sub * 32;
        for (uint j = 0; j < 32; j++) dst[j] = o[j] * inv;
    }
}

// router logits: RL[b][e] = x[b] . W[e] (bf16 rows of 2048); 64 experts x 32 tokens per threadgroup of 64
kernel void rtr(device const float *X [[buffer(0)]], device const ushort *W [[buffer(1)]], device float *RL [[buffer(2)]],
                constant uint &nb [[buffer(3)]],
                uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
    uint t = tid.x;
    threadgroup float ws[64 * 65], xs[64 * 33];
    uint e0 = tg.x * 64, b0 = tg.y * 32, rq = t % 16, tq = t / 16;
    float a0[8], a1[8], a2[8], a3[8];
    for (uint j = 0; j < 8; j++) { a0[j] = 0; a1[j] = 0; a2[j] = 0; a3[j] = 0; }
    uint tk = t / 2, kh = (t % 2) * 32;
    bool tok = b0 + tk < nb;
    for (uint k0 = 0; k0 < H; k0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const ushort4 *wr = (device const ushort4 *)(W + (ulong)(e0 + t) * H + k0);
        for (uint i = 0; i < 16; i++) {
            ushort4 u = wr[i];
            ws[t * 65 + i * 4] = as_type<float>(uint(u.x) << 16); ws[t * 65 + i * 4 + 1] = as_type<float>(uint(u.y) << 16);
            ws[t * 65 + i * 4 + 2] = as_type<float>(uint(u.z) << 16); ws[t * 65 + i * 4 + 3] = as_type<float>(uint(u.w) << 16);
        }
        device const float4 *xr = (device const float4 *)(X + (ulong)(b0 + tk) * H + k0 + kh);
        for (uint i = 0; i < 8; i++) {
            float4 v = tok ? xr[i] : float4(0.0f);
            uint k = kh + i * 4;
            xs[k * 33 + tk] = v.x; xs[(k + 1) * 33 + tk] = v.y; xs[(k + 2) * 33 + tk] = v.z; xs[(k + 3) * 33 + tk] = v.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 64; k++) {
            float w0 = ws[rq * 65 + k], w1 = ws[(rq + 16) * 65 + k], w2 = ws[(rq + 32) * 65 + k], w3 = ws[(rq + 48) * 65 + k];
            for (uint j = 0; j < 8; j++) {
                float x = xs[k * 33 + tq * 8 + j];
                a0[j] += w0 * x; a1[j] += w1 * x; a2[j] += w2 * x; a3[j] += w3 * x;
            }
        }
    }
    for (uint j = 0; j < 8; j++) {
        uint b = b0 + tq * 8 + j;
        if (b < nb) {
            device float *o = RL + (ulong)b * 256 + e0 + rq;
            o[0] = a0[j]; o[16] = a1[j]; o[32] = a2[j]; o[48] = a3[j];
        }
    }
}

// x[token] += sum of its topk pairs' (already weighted) down outputs
kernel void cmba(device const float4 *DO [[buffer(0)]], device const int *tp [[buffer(1)]], device float4 *X [[buffer(2)]],
                 constant uint *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint nb = p[0], k = p[1], b = i / 512, c = i % 512;
    if (b >= nb) return;
    float4 s = 0;
    for (uint j = 0; j < k; j++) s += DO[(ulong)tp[b * k + j] * 512 + c];
    X[i] += s;
}
