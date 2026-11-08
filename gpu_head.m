// gpu_head.m — the output head (151936 x 2048, BF16, exact) on the iMac's Radeon via Metal.
// Weights live in a private VRAM buffer; per token the 2048-float input goes up and 151936 logits come back.
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static id<MTLDevice> DEV;
static id<MTLCommandQueue> QUEUE;
static id<MTLComputePipelineState> PIPE, PIPE8, PIPEM, PIPEM16;  // multi: 16 accumulators per thread keep more threads in flight than 32
static id<MTLBuffer> XM, YM;  // multi: up to 16 input vectors and their logits
static id<MTLBuffer> SBUF;  // int8 head: one fp32 scale per row
static int I8;
static id<MTLBuffer> WBUF, XBUF, YBUF;
static uint32_t ROWS, COLS4;

static const char *SRC =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void head_bf16(device const ushort4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],\n"
    "                      device float *y [[buffer(2)]], constant uint &cols4 [[buffer(3)]],\n"
    "                      uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],\n"
    "                      uint nt [[threads_per_threadgroup]]) {\n"
    "    threadgroup float part[64];\n"
    "    device const ushort4 *wr = w + (ulong)row * cols4;\n"
    "    float acc = 0;\n"
    "    for (uint i = t; i < cols4; i += nt) {\n"
    "        ushort4 v = wr[i];\n"
    "        float4 wf = float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16),\n"
    "                           as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));\n"
    "        acc += dot(wf, x[i]);\n"
    "    }\n"
    "    part[t] = acc;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = nt / 2; s > 0; s >>= 1) { if (t < s) part[t] += part[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "    if (t == 0) y[row] = part[0];\n"
    "}\n"
    "kernel void head_multi(device const ushort4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],\n"
    "                       device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],\n"
    "                       uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[32][64];\n"
    "    uint cols4 = p[0], n = p[1], rows = p[2];\n"
    "    device const ushort4 *wr = w + (ulong)row * cols4;\n"
    "    float acc[32];\n"
    "    for (uint b = 0; b < 32; b++) acc[b] = 0;\n"
    "    for (uint i = t; i < cols4; i += 64) {\n"
    "        ushort4 v = wr[i];\n"
    "        float4 wf = float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16),\n"
    "                           as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));\n"
    "        for (uint b = 0; b < n; b++) acc[b] += dot(wf, x[b * cols4 + i]);\n"
    "    }\n"
    "    for (uint b = 0; b < n; b++) part[b][t] = acc[b];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (t < n) { float a = 0; for (uint i = 0; i < 64; i++) a += part[t][i]; y[(ulong)t * rows + row] = a; }\n"
    "}\n"
    "kernel void head_multi16(device const ushort4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],\n"
    "                       device float *y [[buffer(2)]], constant uint *p [[buffer(3)]],\n"
    "                       uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float part[16][64];\n"
    "    uint cols4 = p[0], n = p[1], rows = p[2];\n"
    "    device const ushort4 *wr = w + (ulong)row * cols4;\n"
    "    float acc[16];\n"
    "    for (uint b = 0; b < 16; b++) acc[b] = 0;\n"
    "    for (uint i = t; i < cols4; i += 64) {\n"
    "        ushort4 v = wr[i];\n"
    "        float4 wf = float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16),\n"
    "                           as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));\n"
    "        for (uint b = 0; b < n; b++) acc[b] += dot(wf, x[b * cols4 + i]);\n"
    "    }\n"
    "    for (uint b = 0; b < n; b++) part[b][t] = acc[b];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (t < n) { float a = 0; for (uint i = 0; i < 64; i++) a += part[t][i]; y[(ulong)t * rows + row] = a; }\n"
    "}\n"
    "kernel void head_i8(device const char4 *w [[buffer(0)]], device const float4 *x [[buffer(1)]],\n"
    "                    device float *y [[buffer(2)]], constant uint &cols4 [[buffer(3)]], device const float *sc [[buffer(4)]],\n"
    "                    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],\n"
    "                    uint nt [[threads_per_threadgroup]]) {\n"
    "    threadgroup float part[64];\n"
    "    device const char4 *wr = w + (ulong)row * cols4;\n"
    "    float acc = 0;\n"
    "    for (uint i = t; i < cols4; i += nt) acc += dot(float4(wr[i]), x[i]);\n"
    "    part[t] = acc;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint s = nt / 2; s > 0; s >>= 1) { if (t < s) part[t] += part[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "    if (t == 0) y[row] = part[0] * sc[row];\n"
    "}\n";

static int setup(void) {
    @autoreleasepool {
        DEV = MTLCreateSystemDefaultDevice();  // nil over SSH without a window-server session
        if (!DEV) { NSArray<id<MTLDevice>> *all = MTLCopyAllDevices(); if (all.count) DEV = all[0]; }
        if (!DEV) { fprintf(stderr, "gpu: no Metal device\n"); return -1; }
        NSError *err = nil;
        id<MTLLibrary> lib = [DEV newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { fprintf(stderr, "gpu: compile failed: %s\n", err.localizedDescription.UTF8String); return -1; }
        PIPE = [DEV newComputePipelineStateWithFunction:[lib newFunctionWithName:@"head_bf16"] error:&err];
        PIPE8 = [DEV newComputePipelineStateWithFunction:[lib newFunctionWithName:@"head_i8"] error:&err];
        PIPEM = [DEV newComputePipelineStateWithFunction:[lib newFunctionWithName:@"head_multi"] error:&err];
        PIPEM16 = [DEV newComputePipelineStateWithFunction:[lib newFunctionWithName:@"head_multi16"] error:&err];
        if (!PIPE || !PIPE8) { fprintf(stderr, "gpu: pipeline failed: %s\n", err.localizedDescription.UTF8String); return -1; }
        QUEUE = [DEV newCommandQueue];
    }
    return 0;
}
static int upload(const void *w, size_t bytes, int rows, int cols) {
    @autoreleasepool {
        ROWS = rows; COLS4 = cols / 4;
        id<MTLBuffer> stage = [DEV newBufferWithBytes:w length:bytes options:MTLResourceStorageModeShared];
        WBUF = [DEV newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        XBUF = [DEV newBufferWithLength:cols * sizeof(float) options:MTLResourceStorageModeManaged];
        YBUF = [DEV newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeManaged];
        if (!stage || !WBUF || !XBUF || !YBUF) { fprintf(stderr, "gpu: buffer allocation failed\n"); return -1; }
        id<MTLCommandBuffer> cb = [QUEUE commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:stage sourceOffset:0 toBuffer:WBUF destinationOffset:0 size:bytes];
        [bl endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        fprintf(stderr, "gpu: %s, head %.0f MB in VRAM\n", DEV.name.UTF8String, bytes / 1e6);
    }
    return 0;
}
int gpu_head_init(const uint16_t *w, int rows, int cols) {
    if (setup()) return -1;
    return upload(w, (size_t)rows * cols * 2, rows, cols);
}
int gpu_head_init_i8(const int8_t *w, const float *scale, int rows, int cols) {
    if (setup()) return -1;
    I8 = 1;
    SBUF = [DEV newBufferWithBytes:scale length:(size_t)rows * 4 options:MTLResourceStorageModeManaged];
    return upload(w, (size_t)rows * cols, rows, cols);
}

void gpu_head_run(const float *x, float *y) {
    @autoreleasepool {
        memcpy(XBUF.contents, x, COLS4 * 4 * sizeof(float));
        [XBUF didModifyRange:NSMakeRange(0, COLS4 * 4 * sizeof(float))];
        id<MTLCommandBuffer> cb = [QUEUE commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:I8 ? PIPE8 : PIPE];
        [enc setBuffer:WBUF offset:0 atIndex:0];
        [enc setBuffer:XBUF offset:0 atIndex:1];
        [enc setBuffer:YBUF offset:0 atIndex:2];
        [enc setBytes:&COLS4 length:sizeof COLS4 atIndex:3];
        if (I8) [enc setBuffer:SBUF offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ROWS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl synchronizeResource:YBUF];
        [bl endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        memcpy(y, YBUF.contents, ROWS * sizeof(float));
    }
}

// up to 64 vectors x[n][cols] -> y[n][rows]; the BF16 head is read once for all of them
void gpu_head_run_multi(const float *x, int n, float *y) {
    @autoreleasepool {
        if (!XM) {
            XM = [DEV newBufferWithLength:64 * COLS4 * 4 * sizeof(float) options:MTLResourceStorageModeManaged];
            YM = [DEV newBufferWithLength:(size_t)64 * ROWS * sizeof(float) options:MTLResourceStorageModeManaged];
        }
        size_t xb = (size_t)n * COLS4 * 4 * sizeof(float);
        memcpy(XM.contents, x, xb); [XM didModifyRange:NSMakeRange(0, xb)];
        id<MTLCommandBuffer> cb = [QUEUE commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:PIPEM16];
        [enc setBuffer:WBUF offset:0 atIndex:0];
        for (int b0 = 0; b0 < n; b0 += 16) {  // passes of 16 vectors: faster than one 32-wide pass on this card
            uint32_t p[3] = {COLS4, (uint32_t)(n - b0 < 16 ? n - b0 : 16), ROWS};
            [enc setBuffer:XM offset:(size_t)b0 * COLS4 * 4 * sizeof(float) atIndex:1];
            [enc setBuffer:YM offset:(size_t)b0 * ROWS * sizeof(float) atIndex:2];
            [enc setBytes:p length:sizeof p atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(ROWS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl synchronizeResource:YM];
        [bl endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        memcpy(y, YM.contents, (size_t)n * ROWS * sizeof(float));
    }
}
