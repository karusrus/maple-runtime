// maple.c — our own runtime for Maple-Preview (20B-A1B, ternary MoE) on the packed checkpoint from convert.py.
// One token at a time, fp32 activations, ternary weights read straight from mmap, a spinning worker pool.
// Mirrors ref.py; ref.py is the yardstick.
//
//   cc -O3 -ffast-math -march=native -o maple maple.c -lpthread        (iMac: AVX2; on M1 use -mcpu=native)
//   ./maple CKDIR --ids 1,2,3 [--n 64] [--threads 4] [--ctx 4096] [--dump logits.f32]
// Prints generated token ids, one per line; timings go to stderr.
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define H 2048
#define NL 24
#define NH 16
#define NKV 4
#define HD 128
#define NE 256
#ifndef TOPK
#define TOPK 8  // experts per token; the model was trained with 8, -DTOPK=N tries others
#endif
#define FF 512
#define VOCAB 151936
#define ROT 64
#define WIN 512
#define EPS 1e-6f
#define EOS 151645

typedef struct { const uint8_t *p; const float *s; int rows, cols; } T2;
typedef struct { const uint16_t *p; int rows, cols; } BF;
typedef struct { BF in_ln, post_ln, q_norm, k_norm, router; T2 q, k, v, o, eg[NE], eu[NE], ed[NE]; } Layer;
typedef struct __attribute__((packed)) { uint8_t kind, file; uint64_t off, soff; uint32_t rows, cols; } Rec;

static const uint8_t *FILES[9];
static size_t FSZ[9];
static BF EMB, HEAD, NORM;
static Layer LY[NL];

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline float bf(uint16_t v) { union { uint32_t u; float f; } c = {(uint32_t)v << 16}; return c.f; }

static const uint8_t *map_file(const char *path, size_t *sz) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    *sz = st.st_size;
    return p;
}

static void load(const char *dir) {
    char path[1024];
    for (int k = 0; k < 9; k++) { snprintf(path, sizeof path, "%s/shard-%05d.bin", dir, k + 1); FILES[k] = map_file(path, &FSZ[k]); }
    snprintf(path, sizeof path, "%s/layout.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    Rec r;
#define NEXT() do { if (fread(&r, sizeof r, 1, f) != 1) { fprintf(stderr, "layout.bin too short\n"); exit(1); } } while (0)
#define GETB(dst) do { NEXT(); if (r.kind != 'b') { fprintf(stderr, "expected bf16\n"); exit(1); } \
        (dst).p = (const uint16_t *)(FILES[r.file] + r.off); (dst).rows = r.rows; (dst).cols = r.cols; } while (0)
#define GETT(dst) do { NEXT(); if (r.kind != 't') { fprintf(stderr, "expected ternary\n"); exit(1); } \
        (dst).p = FILES[r.file] + r.off; (dst).s = (const float *)(FILES[r.file] + r.soff); (dst).rows = r.rows; (dst).cols = r.cols; } while (0)
    GETB(EMB); GETB(HEAD); GETB(NORM);
    for (int L = 0; L < NL; L++) {
        Layer *l = &LY[L];
        GETB(l->in_ln); GETB(l->post_ln); GETB(l->q_norm); GETB(l->k_norm); GETB(l->router);
        GETT(l->q); GETT(l->k); GETT(l->v); GETT(l->o);
        for (int e = 0; e < NE; e++) { GETT(l->eg[e]); GETT(l->eu[e]); GETT(l->ed[e]); }
    }
    fclose(f);
}

/* ---------- worker pool: rows of one job split across threads, workers spin between jobs ---------- */
typedef void (*job_fn)(void *arg, int r0, int r1);
static int NT = 4;
static job_fn JF; static void *JA; static int JN;
static atomic_int GEN, DONE;

static void *worker(void *arg) {
    int i = (int)(intptr_t)arg, g = 0;  // spins ~0.1 s before sleeping: macOS wakes sleepers late
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&GEN, memory_order_acquire) == g) { if (++spins > 100000000) { usleep(200); spins = 0; } }
        g = atomic_load(&GEN);
        int r0 = (int)((long)JN * i / NT), r1 = (int)((long)JN * (i + 1) / NT);
        if (r1 > r0) JF(JA, r0, r1);
        atomic_fetch_add_explicit(&DONE, 1, memory_order_release);
    }
    return NULL;
}
static void run(job_fn fn, void *arg, int n) {
    JF = fn; JA = arg; JN = n;
    atomic_store(&DONE, 0);
    atomic_fetch_add_explicit(&GEN, 1, memory_order_release);
    int r1 = (int)((long)n / NT);
    if (r1 > 0) fn(arg, 0, r1);
    while (atomic_load_explicit(&DONE, memory_order_acquire) < NT - 1) {}
}

/* ---------- kernels ---------- */
static void group_sums(const float *x, int cols, float *gs) {
    for (int g = 0; g < cols / 128; g++) { float s = 0; for (int j = 0; j < 128; j++) s += x[g * 128 + j]; gs[g] = s; }
}
// y[r] = s[r] * sum_c (code-1) * x[c] = s[r] * (sum_c code*x[c] - groupsum)
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_hadd_ps(a, a); a = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(a);
}
// one shift of a packed group: 32 codes for columns sh*32 .. sh*32+31, widened to floats 8 at a time
#define T2_FMA(c, xs, acc) do { \
        __m128i c0_ = _mm256_castsi256_si128(c), c1_ = _mm256_extracti128_si256(c, 1); \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c0_)), _mm256_loadu_ps(xs), acc); \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c0_, 8))), _mm256_loadu_ps(xs + 8), acc); \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c1_)), _mm256_loadu_ps(xs + 16), acc); \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c1_, 8))), _mm256_loadu_ps(xs + 24), acc); \
    } while (0)
static inline float t2_row(const T2 *w, int r, const float *x, const float *gs) {
    const uint8_t *p = w->p + (size_t)r * (w->cols / 4);
    const __m256i m3 = _mm256_set1_epi8(3);
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    float corr = 0;
    for (int g = 0; g < w->cols / 128; g++) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(p + g * 32));
        const float *xg = x + g * 128;
        T2_FMA(_mm256_and_si256(b, m3), xg, a0);
        T2_FMA(_mm256_and_si256(_mm256_srli_epi16(b, 2), m3), xg + 32, a1);
        T2_FMA(_mm256_and_si256(_mm256_srli_epi16(b, 4), m3), xg + 64, a0);
        T2_FMA(_mm256_and_si256(_mm256_srli_epi16(b, 6), m3), xg + 96, a1);
        corr += gs[g];
    }
    return (hsum(_mm256_add_ps(a0, a1)) - corr) * w->s[r];
}
static inline float bf_row(const BF *w, int r, const float *x) {
    const uint16_t *p = w->p + (size_t)r * w->cols;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    for (int c = 0; c < w->cols; c += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(p + c));
        __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v)), 16));
        __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(v, 1)), 16));
        a0 = _mm256_fmadd_ps(lo, _mm256_loadu_ps(x + c), a0);
        a1 = _mm256_fmadd_ps(hi, _mm256_loadu_ps(x + c + 8), a1);
    }
    return hsum(_mm256_add_ps(a0, a1));
}
#else
static inline float t2_row(const T2 *w, int r, const float *x, const float *gs) {
    const uint8_t *p = w->p + (size_t)r * (w->cols / 4);
    float acc = 0;
    for (int g = 0; g < w->cols / 128; g++) {
        const uint8_t *q = p + g * 32; const float *xg = x + g * 128;
        float a = 0;
        for (int j = 0; j < 32; j++) {
            uint8_t b = q[j];
            a += xg[j] * (float)(b & 3) + xg[32 + j] * (float)((b >> 2) & 3) + xg[64 + j] * (float)((b >> 4) & 3) + xg[96 + j] * (float)(b >> 6);
        }
        acc += a - gs[g];
    }
    return acc * w->s[r];
}
static inline float bf_row(const BF *w, int r, const float *x) {
    const uint16_t *p = w->p + (size_t)r * w->cols;
    float a = 0;
    for (int c = 0; c < w->cols; c++) a += x[c] * bf(p[c]);
    return a;
}
#endif

/* ---------- int8 activations: one vector quantised once, then 32 ternary weights per maddubs ---------- */
static int Q8ACT = 1;
typedef struct { const float *x; float gs[H / 128]; int8_t q[H] __attribute__((aligned(32))); float qs; int32_t qg[H / 128], qsum; } Vin;
static void prep(Vin *v, const float *x, int n) {
    v->x = x; float mx = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    v->qs = mx > 0 ? mx / 127.0f : 1.0f; float inv = 1.0f / v->qs;
    for (int g = 0; g < n / 128; g++) {
        float sf = 0; int si = 0;
        for (int j = 0; j < 128; j++) { float xv = x[g * 128 + j]; int qi = (int)lrintf(xv * inv); sf += xv; si += qi; v->q[g * 128 + j] = (int8_t)qi; }
        v->gs[g] = sf; v->qg[g] = si;
    }
    v->qsum = 0; for (int g = 0; g < n / 128; g++) v->qsum += v->qg[g];
}
static inline float tdot(const T2 *w, int r, const Vin *v) {
    if (!Q8ACT) return t2_row(w, r, v->x, v->gs);
    const uint8_t *p = w->p + (size_t)r * (w->cols / 4);
    int corr = 0, tot;
#ifdef __AVX2__
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int g = 0; g < w->cols / 128; g++) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(p + g * 32)); const int8_t *qg = v->q + g * 128;
        __m256i s0 = _mm256_maddubs_epi16(_mm256_and_si256(b, m3), _mm256_loadu_si256((const __m256i *)qg));
        __m256i s1 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b, 2), m3), _mm256_loadu_si256((const __m256i *)(qg + 32)));
        __m256i s2 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b, 4), m3), _mm256_loadu_si256((const __m256i *)(qg + 64)));
        __m256i s3 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b, 6), m3), _mm256_loadu_si256((const __m256i *)(qg + 96)));
        // each i16 lane is at most 4 * 2 * 2 * 127 = 2032, no saturation
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_add_epi16(_mm256_add_epi16(s0, s1), _mm256_add_epi16(s2, s3)), ones));
        corr += v->qg[g];
    }
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    a = _mm_hadd_epi32(a, a); a = _mm_hadd_epi32(a, a);
    tot = _mm_cvtsi128_si32(a);
#else
    tot = 0;
    for (int g = 0; g < w->cols / 128; g++) {
        const uint8_t *q = p + g * 32; const int8_t *qg = v->q + g * 128;
        for (int j = 0; j < 32; j++) {
            uint8_t b = q[j];
            tot += (b & 3) * qg[j] + ((b >> 2) & 3) * qg[32 + j] + ((b >> 4) & 3) * qg[64 + j] + (b >> 6) * qg[96 + j];
        }
        corr += v->qg[g];
    }
#endif
    return (float)(tot - corr) * v->qs * w->s[r];
}
static Vin V_H, V_ATT, V_A[TOPK];

/* ---------- warm-up: map every page once, so tokens do not pay soft page faults ---------- */
static volatile uint8_t SINK;
static void job_warm(void *a, int r0, int r1) {  // rows = 1 MB chunks across all shard files
    (void)a; uint8_t acc = 0;
    for (int r = r0; r < r1; r++) {
        size_t off = (size_t)r << 20; int k = 0;
        while (k < 9 && off >= FSZ[k]) { off -= FSZ[k]; k++; }
        if (k == 9) continue;
        size_t end = off + (1 << 20) < FSZ[k] ? off + (1 << 20) : FSZ[k];
        for (size_t i = off; i < end; i += 4096) acc += FILES[k][i];
    }
    SINK += acc;
}

/* ---------- int8 output head: one fp32 scale per row, half the bytes of BF16 per token ---------- */
static int8_t *H8; static float *H8S; static int HEAD8 = 1, GPUHEAD = 0;
#ifdef GPU_HEAD
int gpu_head_init(const uint16_t *w, int rows, int cols);
int gpu_head_init_i8(const int8_t *w, const float *scale, int rows, int cols);
void gpu_head_run(const float *x, float *y);
void gpu_head_run_multi(const float *x, int n, float *y);
int gpux_init(int layers, int slots);
void gpux_upload(int L, int slot, const uint8_t *gc, const float *gs, const uint8_t *uc, const float *us, const uint8_t *dc, const float *ds);
void *gpux_begin(int L, const float *x, int n, const int *slots);
void gpux_end(void *h, float *out, int n);
void *gpux_begin_multi(int L, const float *x, int nb, int nu, const int *slots, const int *pstart, const int *ptok);
void gpux_end_multi(void *h, float *out, int np);
int gpre_init(int maxtok, int topk);
int gpre_region(const void *p, size_t len);
double gpre_moe(const float *x, int nb, int nu, const uint32_t *loc, const int *regs, const int *next, const int *ustart, const int *ptok, const float *pw, const int *tp, float *y);
int gpre_stage(size_t bytes);
void gpre_dense_add(int g, const uint8_t *codes, const float *scales, int rows, int cols);
int gpre_dense_done(int maxtok);
double gpre_dense(int g, const float *x, int nb, float *y);
double gpre_blit_seconds(void);
int gfull_init(int ctx, int maxtok, int win, const float *norms, const uint16_t *router, const float *cs);
void gfull_begin(const float *x, int nb, int pos0);
double gfull_attn(int L, int glob, float *rl);
void gfull_moe(int nu, const uint32_t *loc, const int *regs, const int *ustart, const int *ptok, const float *pw, const int *tp);
void gfull_read(float *x, const uint16_t **ko);
void gfull_write_x(const float *x);
int gfull_maxtok(void);
void gfull_report(void);
#endif
/* ---------- prompt on the card (--gpupre N): experts of a whole chunk run there, weights read in place ---------- */
static int GPRE = 0;
static uint32_t GLOC[24][256][12];  // (region slot, offset) of gate codes, gate scales, up codes, up scales, down codes, down scales
static int GREG[24][4];             // per layer: the region ids behind slots 0..3
static int GDENSE;                  // q,k,v and o on the card too (--gpupre; GPRE_ATTCPU=1 keeps them on the CPU)
static int GFULL;                   // the whole layer on the card, the CPU only picks experts (--gpupre; GPRE_PART=1: experts and projections only)
static size_t GSPAN[24][4];
static double T_GPU;
/* ---------- expert cache on the card (--gpux S): hits run there, misses on the CPU at the same time ---------- */
static int GPUX = 0, GSLOTS = 32, MAXUP = 1, GPUHEAD8 = 0, TICK;
static int ESLOT[24][256], SEXP[24][64], SLAST[24][64], PFCNT[24][256];
static long GX_HIT, GX_TOT, GX_UP;
static void job_h8(void *a, int r0, int r1) {
    (void)a;
    for (int r = r0; r < r1; r++) {
        const uint16_t *p = HEAD.p + (size_t)r * H; float mx = 0;
        for (int c = 0; c < H; c++) { float v = fabsf(bf(p[c])); if (v > mx) mx = v; }
        float sc = mx > 0 ? mx / 127.0f : 1.0f; H8S[r] = sc;
        for (int c = 0; c < H; c++) H8[(size_t)r * H + c] = (int8_t)lrintf(bf(p[c]) / sc);
    }
}
static inline float h8_row(int r, const float *x) {
    const int8_t *p = H8 + (size_t)r * H;
#ifdef __AVX2__
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    for (int c = 0; c < H; c += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(p + c));
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), _mm256_loadu_ps(x + c), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8))), _mm256_loadu_ps(x + c + 8), a1);
    }
    return hsum(_mm256_add_ps(a0, a1)) * H8S[r];
#else
    float a = 0; for (int c = 0; c < H; c++) a += x[c] * p[c];
    return a * H8S[r];
#endif
}

/* ---------- state ---------- */
static int CTX = 4096;
static uint16_t *KC, *VC;  // [NSEQ][NL][CTX][NKV][HD] in fp16: one cache per conversation, half the bytes to read
static inline void to_f16(uint16_t *d, const float *s, int n) {
#if defined(__F16C__)
    for (int i = 0; i < n; i += 8) _mm_storeu_si128((__m128i *)(d + i), _mm256_cvtps_ph(_mm256_loadu_ps(s + i), 0));
#else
    for (int i = 0; i < n; i++) { __fp16 h = (__fp16)s[i]; memcpy(d + i, &h, 2); }
#endif
}
static int NSEQ = 1;
#define MAXSEQ 64
#define KOFF(seq, L, pos) ((((size_t)(seq) * NL + (L)) * CTX + (pos)) * NKV * HD)
static float X[H], HN[H], Q[NH * HD], K1[NKV * HD], V1[NKV * HD], ATT[NH * HD], Y[H];
static float GS_H[H / 128], GS_A[FF / 128 * TOPK];
static float GATE[TOPK][FF], UP[TOPK][FF], ACT[TOPK][FF], DOWN[TOPK][H];
static float LOGITS[VOCAB];
static int TOPI[TOPK]; static float TOPW[TOPK];
static double INV_FREQ[ROT / 2];

typedef struct { const Layer *l; } QKVArg;
static void job_qkv(void *a, int r0, int r1) {
    const Layer *l = ((QKVArg *)a)->l;
    for (int r = r0; r < r1; r++) {
        if (r < NH * HD) Q[r] = tdot(&l->q, r, &V_H);
        else if (r < NH * HD + NKV * HD) K1[r - NH * HD] = tdot(&l->k, r - NH * HD, &V_H);
        else V1[r - NH * HD - NKV * HD] = tdot(&l->v, r - NH * HD - NKV * HD, &V_H);
    }
}
static float GS_ATT[H / 128];
static void job_o(void *a, int r0, int r1) {
    const Layer *l = (const Layer *)a;
    for (int r = r0; r < r1; r++) Y[r] = tdot(&l->o, r, &V_ATT);
}
static void job_gateup(void *a, int r0, int r1) {  // rows: expert slot * 2 * FF
    const Layer *l = (const Layer *)a;
    for (int r = r0; r < r1; r++) {
        int s = r / (2 * FF), k = r % (2 * FF), e = TOPI[s];
        if (k < FF) GATE[s][k] = tdot(&l->eg[e], k, &V_H);
        else UP[s][k - FF] = tdot(&l->eu[e], k - FF, &V_H);
    }
}
static int MS_N, MS_IDX[TOPK];  // slots whose experts are not on the card
static void job_gateup_m(void *a, int r0, int r1) {
    const Layer *l = (const Layer *)a;
    for (int r = r0; r < r1; r++) {
        int s = MS_IDX[r / (2 * FF)], k = r % (2 * FF), e = TOPI[s];
        if (k < FF) GATE[s][k] = tdot(&l->eg[e], k, &V_H);
        else UP[s][k - FF] = tdot(&l->eu[e], k - FF, &V_H);
    }
}
static void job_down_m(void *a, int r0, int r1) {
    const Layer *l = (const Layer *)a;
    for (int r = r0; r < r1; r++) { int s = MS_IDX[r / H], k = r % H; DOWN[s][k] = tdot(&l->ed[TOPI[s]], k, &V_A[s]); }
}
#ifdef GPU_HEAD
static float GOUT[TOPK][H];
static void cache_put(int L, int e) {  // into an empty slot, else the least recently used one
    int v = -1;
    for (int k = 0; k < GSLOTS; k++) if (SEXP[L][k] < 0) { v = k; break; }
    if (v < 0) { v = 0; for (int k = 1; k < GSLOTS; k++) if (SLAST[L][k] < SLAST[L][v]) v = k; }
    if (SEXP[L][v] >= 0) ESLOT[L][SEXP[L][v]] = -1;
    const Layer *l = &LY[L];
    gpux_upload(L, v, l->eg[e].p, l->eg[e].s, l->eu[e].p, l->eu[e].s, l->ed[e].p, l->ed[e].s);
    SEXP[L][v] = e; ESLOT[L][e] = v; SLAST[L][v] = TICK; GX_UP++;
}
#endif
static void job_down(void *a, int r0, int r1) {  // rows: expert slot * H
    const Layer *l = (const Layer *)a;
    for (int r = r0; r < r1; r++) {
        int s = r / H, k = r % H;
        DOWN[s][k] = tdot(&l->ed[TOPI[s]], k, &V_A[s]);
    }
}
static void job_head(void *a, int r0, int r1);
static void head(void) {
#ifdef GPU_HEAD
    if (GPUHEAD) { gpu_head_run(HN, LOGITS); return; }
#endif
    run(job_head, NULL, VOCAB);
}
static void job_head(void *a, int r0, int r1) {
    (void)a;
    if (HEAD8) for (int r = r0; r < r1; r++) LOGITS[r] = h8_row(r, HN);
    else for (int r = r0; r < r1; r++) LOGITS[r] = bf_row(&HEAD, r, HN);
}

static void rmsnorm(float *out, const float *x, const BF *w, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float k = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * k * bf(w->p[i]);
}
static void rope(float *v, int pos) {  // first ROT dims, halves of ROT/2
    float t[ROT];
    for (int i = 0; i < ROT / 2; i++) {
        double f = pos * INV_FREQ[i]; float c = (float)cos(f), s = (float)sin(f);
        t[i] = v[i] * c - v[i + ROT / 2] * s;
        t[i + ROT / 2] = v[i + ROT / 2] * c + v[i] * s;
    }
    memcpy(v, t, sizeof t);
}

/* ---------- prompt in chunks: every weight row is read once for the whole chunk ---------- */
#define BMAX 2048  // capacity; the prompt is read in chunks of CHUNK tokens (128, or --chunk / --gpupre N)
static int CHUNK = 128;
static float XB[BMAX][H], HNB[BMAX][H], QB[BMAX][NH * HD], KB_[BMAX][NKV * HD], VB_[BMAX][NKV * HD], ATTB[BMAX][H], YB[BMAX][H];
static Vin VHB[BMAX], VATTB[BMAX], VAB[BMAX * TOPK];
static int BN, NU, UE[NE], USTART[NE + 1], UTOK[BMAX * TOPK];
static float UW[BMAX * TOPK], GATEB[BMAX * TOPK][FF], UPB[BMAX * TOPK][FF], ACTB[BMAX * TOPK][FF], DOWNB[BMAX * TOPK][H];
static const Layer *BL;
static float QKVB[BMAX][NH * HD + 2 * NKV * HD];

// out[i * stride] = row r of w against each of nv vectors; the packed row is loaded once
#define TBLK 64
static void tdot_multi(const T2 *w, int r, Vin *const *vs, int nv, float *out, int stride) {
    if (nv > TBLK) {  // big chunks: one block of tokens at a time; the packed row is re-read from L1
        for (int i = 0; i < nv; i += TBLK) tdot_multi(w, r, vs + i, nv - i < TBLK ? nv - i : TBLK, out + (size_t)i * stride, stride);
        return;
    }
#ifdef __AVX2__
    if (Q8ACT) {
        const uint8_t *p = w->p + (size_t)r * (w->cols / 4);
        const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
        __m256i acc[TBLK];
        for (int i = 0; i < nv; i++) acc[i] = _mm256_setzero_si256();
        for (int g = 0; g < w->cols / 128; g++) {
            __m256i b = _mm256_loadu_si256((const __m256i *)(p + g * 32));
            __m256i c0 = _mm256_and_si256(b, m3), c1 = _mm256_and_si256(_mm256_srli_epi16(b, 2), m3);
            __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(b, 4), m3), c3 = _mm256_and_si256(_mm256_srli_epi16(b, 6), m3);
            for (int i = 0; i < nv; i++) {
                const int8_t *q = vs[i]->q + g * 128;
                __m256i s01 = _mm256_add_epi16(_mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *)q)),
                                               _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *)(q + 32))));
                __m256i s23 = _mm256_add_epi16(_mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *)(q + 64))),
                                               _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *)(q + 96))));
                acc[i] = _mm256_add_epi32(acc[i], _mm256_madd_epi16(_mm256_add_epi16(s01, s23), ones));
            }
        }
        for (int i = 0; i < nv; i++) {
            __m128i a = _mm_add_epi32(_mm256_castsi256_si128(acc[i]), _mm256_extracti128_si256(acc[i], 1));
            a = _mm_hadd_epi32(a, a); a = _mm_hadd_epi32(a, a);
            out[(size_t)i * stride] = (float)(_mm_cvtsi128_si32(a) - vs[i]->qsum) * vs[i]->qs * w->s[r];
        }
        return;
    }
#endif
    for (int i = 0; i < nv; i++) out[(size_t)i * stride] = tdot(w, r, vs[i]);
}
static void jobB_qkv(void *a, int r0, int r1) {
    (void)a; Vin *vs[BMAX]; for (int i = 0; i < BN; i++) vs[i] = &VHB[i];
    for (int r = r0; r < r1; r++) {
        if (r < NH * HD) tdot_multi(&BL->q, r, vs, BN, &QB[0][r], NH * HD);
        else if (r < NH * HD + NKV * HD) tdot_multi(&BL->k, r - NH * HD, vs, BN, &KB_[0][r - NH * HD], NKV * HD);
        else tdot_multi(&BL->v, r - NH * HD - NKV * HD, vs, BN, &VB_[0][r - NH * HD - NKV * HD], NKV * HD);
    }
}
static void jobB_o(void *a, int r0, int r1) {
    (void)a; Vin *vs[BMAX]; for (int i = 0; i < BN; i++) vs[i] = &VATTB[i];
    for (int r = r0; r < r1; r++) tdot_multi(&BL->o, r, vs, BN, &YB[0][r], H);
}
static void jobB_gateup(void *a, int r0, int r1) {  // rows: used expert * 2 * FF
    (void)a; Vin *vs[BMAX];
    for (int r = r0; r < r1; r++) {
        int u = r / (2 * FF), k = r % (2 * FF), e = UE[u], a0 = USTART[u], na = USTART[u + 1] - a0;
        for (int i = 0; i < na; i++) vs[i] = &VHB[UTOK[a0 + i]];
        if (k < FF) tdot_multi(&BL->eg[e], k, vs, na, &GATEB[a0][k], FF);
        else tdot_multi(&BL->eu[e], k - FF, vs, na, &UPB[a0][k - FF], FF);
    }
}
static int CU[NE], NCU;  // used experts the CPU runs in this layer (the rest are on the card)
static void jobB_gateup_c(void *a, int r0, int r1) {
    (void)a; Vin *vs[BMAX];
    for (int r = r0; r < r1; r++) {
        int u = CU[r / (2 * FF)], k = r % (2 * FF), e = UE[u], a0 = USTART[u], na = USTART[u + 1] - a0;
        for (int i = 0; i < na; i++) vs[i] = &VHB[UTOK[a0 + i]];
        if (k < FF) tdot_multi(&BL->eg[e], k, vs, na, &GATEB[a0][k], FF);
        else tdot_multi(&BL->eu[e], k - FF, vs, na, &UPB[a0][k - FF], FF);
    }
}
static void jobB_down_c(void *a, int r0, int r1) {
    (void)a; Vin *vs[BMAX];
    for (int r = r0; r < r1; r++) {
        int u = CU[r / H], k = r % H, e = UE[u], a0 = USTART[u], na = USTART[u + 1] - a0;
        for (int i = 0; i < na; i++) vs[i] = &VAB[a0 + i];
        tdot_multi(&BL->ed[e], k, vs, na, &DOWNB[a0][k], H);
    }
}
static void jobB_down(void *a, int r0, int r1) {  // rows: used expert * H
    (void)a; Vin *vs[BMAX];
    for (int r = r0; r < r1; r++) {
        int u = r / H, k = r % H, e = UE[u], a0 = USTART[u], na = USTART[u + 1] - a0;
        for (int i = 0; i < na; i++) vs[i] = &VAB[a0 + i];
        tdot_multi(&BL->ed[e], k, vs, na, &DOWNB[a0][k], H);
    }
}
static float *SCR;  // [BMAX * NH][CTX] attention scores, one row per (token, head) job
#define QPK (NH / NKV)  // query heads per key/value head
static inline void load8(const uint16_t *p, float *o) {
#if defined(__F16C__)
    _mm256_storeu_ps(o, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)));
#else
    for (int i = 0; i < 8; i++) { __fp16 h; memcpy(&h, p + i, 2); o[i] = (float)h; }
#endif
}
// heads kv*4 .. kv*4+3 share key/value head kv: every cache row is read once for all four
static void attend_group(int seq, int L, int glob, int pos, const float *q0, float *out, int kv, float *sc) {
    int lo = glob ? 0 : (pos - WIN > 0 ? pos - WIN : 0), T = pos - lo + 1;
    const float scale = 1.0f / sqrtf((float)HD);
    const float *q[QPK]; float mx[QPK], inv[QPK]; float *o[QPK];
    for (int j = 0; j < QPK; j++) { q[j] = q0 + (kv * QPK + j) * HD; mx[j] = -1e30f; o[j] = out + (kv * QPK + j) * HD; memset(o[j], 0, HD * sizeof(float)); }
    for (int t = 0; t < T; t++) {
        const uint16_t *kk = KC + KOFF(seq, L, lo + t) + kv * HD;
        float s[QPK];
#if defined(__F16C__) && defined(__AVX2__)
        __m256 a[QPK];
        for (int j = 0; j < QPK; j++) a[j] = _mm256_setzero_ps();
        for (int d = 0; d < HD; d += 8) {
            __m256 k8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kk + d)));
            for (int j = 0; j < QPK; j++) a[j] = _mm256_fmadd_ps(k8, _mm256_loadu_ps(q[j] + d), a[j]);
        }
        for (int j = 0; j < QPK; j++) s[j] = hsum(a[j]) * scale;
#else
        float k8[HD];
        for (int d = 0; d < HD; d += 8) load8(kk + d, k8 + d);
        for (int j = 0; j < QPK; j++) { float z = 0; for (int d = 0; d < HD; d++) z += q[j][d] * k8[d]; s[j] = z * scale; }
#endif
        for (int j = 0; j < QPK; j++) { sc[j * T + t] = s[j]; if (s[j] > mx[j]) mx[j] = s[j]; }
    }
    for (int j = 0; j < QPK; j++) {
        float sum = 0, *sj = sc + j * T;
        for (int t = 0; t < T; t++) { sj[t] = expf(sj[t] - mx[j]); sum += sj[t]; }
        inv[j] = 1.0f / sum;
    }
    for (int t = 0; t < T; t++) {
        const uint16_t *vv = VC + KOFF(seq, L, lo + t) + kv * HD;
        float w[QPK]; for (int j = 0; j < QPK; j++) w[j] = sc[j * T + t] * inv[j];
#if defined(__F16C__) && defined(__AVX2__)
        for (int d = 0; d < HD; d += 8) {
            __m256 v8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vv + d)));
            for (int j = 0; j < QPK; j++) _mm256_storeu_ps(o[j] + d, _mm256_fmadd_ps(_mm256_set1_ps(w[j]), v8, _mm256_loadu_ps(o[j] + d)));
        }
#else
        float v8[HD];
        for (int d = 0; d < HD; d += 8) load8(vv + d, v8 + d);
        for (int j = 0; j < QPK; j++) for (int d = 0; d < HD; d++) o[j][d] += w[j] * v8[d];
#endif
    }
}
static int AL, AGLOB, APOS0, ASEQ[BMAX], APOS[BMAX];
static void jobB_att(void *a, int r0, int r1) {  // rows: token * NKV + key/value head
    (void)a;
    for (int r = r0; r < r1; r++) { int b = r / NKV, kv = r % NKV; attend_group(ASEQ[b], AL, AGLOB, APOS[b], QB[b], ATTB[b], kv, SCR + (size_t)r * QPK * CTX); }
}
static void job_att1(void *a, int r0, int r1) {  // decode: rows are key/value heads
    (void)a;
    for (int kv = r0; kv < r1; kv++) attend_group(0, AL, AGLOB, APOS0, Q, ATT, kv, SCR + (size_t)kv * QPK * CTX);
}
static void route(const Layer *l, const float *hn, int *top, float *w) {
    float lg[NE], mx = -1e30f, sum = 0, tsum = 0;
    for (int e = 0; e < NE; e++) { lg[e] = bf_row(&l->router, e, hn); if (lg[e] > mx) mx = lg[e]; }
    for (int e = 0; e < NE; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
    for (int e = 0; e < NE; e++) lg[e] /= sum;
    for (int s = 0; s < TOPK; s++) {
        int best = -1; for (int e = 0; e < NE; e++) if (best < 0 || lg[e] > lg[best]) best = e;
        top[s] = best; w[s] = lg[best]; tsum += lg[best]; lg[best] = -1;
    }
    for (int s = 0; s < TOPK; s++) w[s] /= (tsum + 1e-20f);
}
static double T_ATT, T_EXP, T_HEAD, T_ROUTE, T_SC;  // T_SC: the pass over the cache, part of T_ATT
static int PREFILL1;
static FILE *RLOG;  // --route-log: one line per (layer, position): the 8 chosen experts
static void step(int tok, int pos) {
    double ta;
    TICK++;
    for (int i = 0; i < H; i++) X[i] = bf(EMB.p[(size_t)tok * H + i]);
    for (int L = 0; L < NL; L++) {
        const Layer *l = &LY[L];
        int glob = (L % 4) == 3;
        ta = now();
        rmsnorm(HN, X, &l->in_ln, H);
        prep(&V_H, HN, H);
        QKVArg qa = {l}; run(job_qkv, &qa, NH * HD + 2 * NKV * HD);
        for (int h = 0; h < NH; h++) { rmsnorm(Q + h * HD, Q + h * HD, &l->q_norm, HD); if (!glob) rope(Q + h * HD, pos); }
        for (int h = 0; h < NKV; h++) { rmsnorm(K1 + h * HD, K1 + h * HD, &l->k_norm, HD); if (!glob) rope(K1 + h * HD, pos); }
        uint16_t *kc = KC + KOFF(0, L, pos), *vc = VC + KOFF(0, L, pos);
        to_f16(kc, K1, NKV * HD); to_f16(vc, V1, NKV * HD);
        AL = L; AGLOB = glob; APOS0 = pos;
        double tsc = now();
        run(job_att1, NULL, NKV);
        T_SC += now() - tsc;
        prep(&V_ATT, ATT, H);
        run(job_o, (void *)l, H);
        for (int i = 0; i < H; i++) X[i] += Y[i];
        T_ATT += now() - ta; ta = now();
        // MoE
        rmsnorm(HN, X, &l->post_ln, H);
        prep(&V_H, HN, H);
        float lg[NE], mx = -1e30f, sum = 0;
        for (int e = 0; e < NE; e++) { lg[e] = bf_row(&l->router, e, HN); if (lg[e] > mx) mx = lg[e]; }
        for (int e = 0; e < NE; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
        for (int e = 0; e < NE; e++) lg[e] /= sum;
        float tsum = 0;
        for (int s = 0; s < TOPK; s++) {
            int best = -1; for (int e = 0; e < NE; e++) if (best < 0 || lg[e] > lg[best]) best = e;
            TOPI[s] = best; TOPW[s] = lg[best]; tsum += lg[best]; lg[best] = -1;
        }
        for (int s = 0; s < TOPK; s++) TOPW[s] /= (tsum + 1e-20f);
        if (RLOG) { fprintf(RLOG, "%d %d", L, pos); for (int s = 0; s < TOPK; s++) fprintf(RLOG, " %d", TOPI[s]); fputc('\n', RLOG); }
        T_ROUTE += now() - ta; ta = now();
#ifdef GPU_HEAD
        if (GPUX) {
            int hs[TOPK], hsl[TOPK], nh = 0; MS_N = 0;
            for (int s = 0; s < TOPK; s++) {
                int sl = ESLOT[L][TOPI[s]];
                if (sl >= 0) { hs[nh] = s; hsl[nh++] = sl; SLAST[L][sl] = TICK; } else MS_IDX[MS_N++] = s;
            }
            GX_HIT += nh; GX_TOT += TOPK;
            void *gh = nh ? gpux_begin(L, HN, nh, hsl) : NULL;
            if (MS_N) {
                run(job_gateup_m, (void *)l, MS_N * 2 * FF);
                for (int m = 0; m < MS_N; m++) {
                    int s = MS_IDX[m];
                    for (int k = 0; k < FF; k++) {
                        float g = GATE[s][k] < 7.0f ? GATE[s][k] : 7.0f, u = UP[s][k];
                        u = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
                        ACT[s][k] = g / (1.0f + expf(-g)) * u;
                    }
                    prep(&V_A[s], ACT[s], FF);
                }
                run(job_down_m, (void *)l, MS_N * H);
            }
            if (gh) gpux_end(gh, &GOUT[0][0], nh);
            for (int i = 0; i < H; i++) {
                float y = 0;
                for (int k = 0; k < nh; k++) y += TOPW[hs[k]] * GOUT[k][i];
                for (int m = 0; m < MS_N; m++) y += TOPW[MS_IDX[m]] * DOWN[MS_IDX[m]][i];
                X[i] += y;
            }
            for (int m = 0; m < MS_N && m < MAXUP; m++) cache_put(L, TOPI[MS_IDX[m]]);  // misses come in heaviest first
            T_EXP += now() - ta;
            continue;
        }
#endif
        run(job_gateup, (void *)l, TOPK * 2 * FF);
        for (int s = 0; s < TOPK; s++) {
            for (int k = 0; k < FF; k++) {
                float g = GATE[s][k] < 7.0f ? GATE[s][k] : 7.0f, u = UP[s][k];
                u = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
                ACT[s][k] = g / (1.0f + expf(-g)) * u;
            }
            prep(&V_A[s], ACT[s], FF);
        }
        run(job_down, (void *)l, TOPK * H);
        for (int i = 0; i < H; i++) { float y = 0; for (int s = 0; s < TOPK; s++) y += TOPW[s] * DOWN[s][i]; X[i] += y; }
        T_EXP += now() - ta;
    }
    ta = now();
    rmsnorm(HN, X, &NORM, H);
    head();
    T_HEAD += now() - ta;
}

// nb tokens at positions pos0.. ; logits for every token if df, otherwise only for the last one
// mode 0: logits of the last token into LOGITS (and every token to df if given); mode 1: every token's logits into LOGB
static float *LOGB;
static float RLG[BMAX][NE];
static void job_head_multi(void *a, int r0, int r1);
static void job_route_multi(void *a, int r0, int r1);
static void job_route_small(void *a, int r0, int r1);
static void step_batch_card(const int *toks, const int *seqs, const int *poss, int nb, FILE *df);
static void step_batch(const int *toks, const int *seqs, const int *poss, int nb, FILE *df, int mode) {
#ifdef GPU_HEAD
    if (GFULL && mode == 0) { step_batch_card(toks, seqs, poss, nb, df); return; }
#endif
    BN = nb; if (mode == 1) TICK++;
    for (int b = 0; b < nb; b++) for (int i = 0; i < H; i++) XB[b][i] = bf(EMB.p[(size_t)toks[b] * H + i]);
    for (int L = 0; L < NL; L++) {
        BL = &LY[L]; int glob = (L % 4) == 3; double ta = now();
#ifdef GPU_HEAD
        if (GDENSE && mode == 0) {
            for (int b = 0; b < nb; b++) rmsnorm(HNB[b], XB[b], &BL->in_ln, H);
            T_GPU += gpre_dense(L * 2, &HNB[0][0], nb, &QKVB[0][0]);
            if (getenv("GPRE_CHECK")) {  // the same projections on the CPU, compared
                for (int b = 0; b < nb; b++) prep(&VHB[b], HNB[b], H);
                run(jobB_qkv, NULL, NH * HD + 2 * NKV * HD);
                double md = 0, mx = 0;
                for (int b = 0; b < nb; b++) for (int i = 0; i < NH * HD + 2 * NKV * HD; i++) {
                    float c = i < NH * HD ? QB[b][i] : i < NH * HD + NKV * HD ? KB_[b][i - NH * HD] : VB_[b][i - NH * HD - NKV * HD];
                    if (fabs(c - QKVB[b][i]) > md) md = fabs(c - QKVB[b][i]); if (fabs(c) > mx) mx = fabs(c);
                }
                fprintf(stderr, "gpre check qkv layer %d: max |cpu - gpu| %.5f, max |cpu| %.5f\n", L, md, mx);
            }
            for (int b = 0; b < nb; b++) {
                memcpy(QB[b], QKVB[b], sizeof QB[b]); memcpy(KB_[b], QKVB[b] + NH * HD, sizeof KB_[b]); memcpy(VB_[b], QKVB[b] + NH * HD + NKV * HD, sizeof VB_[b]);
            }
        } else
#endif
        {
            for (int b = 0; b < nb; b++) { rmsnorm(HNB[b], XB[b], &BL->in_ln, H); prep(&VHB[b], HNB[b], H); }
            run(jobB_qkv, NULL, NH * HD + 2 * NKV * HD);
        }
        for (int b = 0; b < nb; b++) {
            int pos = poss[b];
            for (int h = 0; h < NH; h++) { rmsnorm(QB[b] + h * HD, QB[b] + h * HD, &BL->q_norm, HD); if (!glob) rope(QB[b] + h * HD, pos); }
            for (int h = 0; h < NKV; h++) { rmsnorm(KB_[b] + h * HD, KB_[b] + h * HD, &BL->k_norm, HD); if (!glob) rope(KB_[b] + h * HD, pos); }
            to_f16(KC + KOFF(seqs[b], L, pos), KB_[b], NKV * HD);
            to_f16(VC + KOFF(seqs[b], L, pos), VB_[b], NKV * HD);
            ASEQ[b] = seqs[b]; APOS[b] = pos;
        }
        AL = L; AGLOB = glob;
        run(jobB_att, NULL, nb * NKV);  // token b reads keys up to its own position only: causal
#ifdef GPU_HEAD
        if (GDENSE && mode == 0) T_GPU += gpre_dense(L * 2 + 1, &ATTB[0][0], nb, &YB[0][0]);
        else
#endif
        {
            for (int b = 0; b < nb; b++) prep(&VATTB[b], ATTB[b], H);
            run(jobB_o, NULL, H);
        }
        for (int b = 0; b < nb; b++) for (int i = 0; i < H; i++) XB[b][i] += YB[b][i];
        T_ATT += now() - ta; ta = now();
        int top[BMAX][TOPK], cnt[NE] = {0}, uidx[NE], fillc[NE] = {0}; float tw[BMAX][TOPK];
        for (int b = 0; b < nb; b++) { rmsnorm(HNB[b], XB[b], &BL->post_ln, H); if (!(GPRE && mode == 0)) prep(&VHB[b], HNB[b], H); }
        if (nb <= 32) run(job_route_small, NULL, NE); else run(job_route_multi, NULL, (nb + 7) / 8);
        for (int b = 0; b < nb; b++) {
            float *lg = RLG[b], mx = -1e30f, sum = 0, tsum = 0;
            for (int e = 0; e < NE; e++) if (lg[e] > mx) mx = lg[e];
            for (int e = 0; e < NE; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
            for (int e = 0; e < NE; e++) lg[e] /= sum;
            for (int k = 0; k < TOPK; k++) {
                int best = -1; for (int e = 0; e < NE; e++) if (best < 0 || lg[e] > lg[best]) best = e;
                top[b][k] = best; tw[b][k] = lg[best]; tsum += lg[best]; lg[best] = -1;
            }
            for (int k = 0; k < TOPK; k++) tw[b][k] /= (tsum + 1e-20f);
            for (int s = 0; s < TOPK; s++) PFCNT[L][top[b][s]]++;
            if (RLOG) { fprintf(RLOG, "%d %d", L, poss[b]); for (int s = 0; s < TOPK; s++) fprintf(RLOG, " %d", top[b][s]); fputc('\n', RLOG); }
            for (int s = 0; s < TOPK; s++) cnt[top[b][s]]++;
        }
        NU = 0; int na = 0;
        for (int e = 0; e < NE; e++) if (cnt[e]) { uidx[e] = NU; UE[NU] = e; USTART[NU++] = na; na += cnt[e]; }
        USTART[NU] = na;
        for (int b = 0; b < nb; b++) for (int s = 0; s < TOPK; s++) {
            int u = uidx[top[b][s]], slot = USTART[u] + fillc[u]++;
            UTOK[slot] = b; UW[slot] = tw[b][s];
        }
        T_ROUTE += now() - ta; ta = now();
#ifdef GPU_HEAD
        if (GPRE && mode == 0) {
            static int tp[BMAX * TOPK], fl[BMAX]; static uint32_t lc[NE * 12];
            for (int u = 0; u < NU; u++) memcpy(lc + u * 12, GLOC[L][UE[u]], sizeof GLOC[L][0]);
            memset(fl, 0, sizeof(int) * nb);
            for (int a = 0; a < na; a++) { int b = UTOK[a]; tp[b * TOPK + fl[b]++] = a; }
            T_GPU += gpre_moe(&HNB[0][0], nb, NU, lc, GREG[L], GREG[(L + 1) % NL], USTART, UTOK, UW, tp, &YB[0][0]);
            if (getenv("GPRE_CHECK")) {  // the same layer on the CPU, compared
                for (int b = 0; b < nb; b++) prep(&VHB[b], HNB[b], H);
                run(jobB_gateup, NULL, NU * 2 * FF);
                for (int a = 0; a < na; a++) {
                    for (int k = 0; k < FF; k++) {
                        float g = GATEB[a][k] < 7.0f ? GATEB[a][k] : 7.0f, u = UPB[a][k];
                        u = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
                        ACTB[a][k] = g / (1.0f + expf(-g)) * u;
                    }
                    prep(&VAB[a], ACTB[a], FF);
                }
                run(jobB_down, NULL, NU * H);
                double md = 0, mx = 0;
                for (int b = 0; b < nb; b++) for (int i = 0; i < H; i++) {
                    float y = 0; for (int a = 0; a < na; a++) if (UTOK[a] == b) y += UW[a] * DOWNB[a][i];
                    if (fabs(y - YB[b][i]) > md) md = fabs(y - YB[b][i]); if (fabs(y) > mx) mx = fabs(y);
                    if (0) fprintf(stderr, "  L%d tok0 [%d] cpu %.5f gpu %.5f\n", L, i, y, YB[b][i]);
                }
                fprintf(stderr, "gpre check layer %d: max |cpu - gpu| %.5f, max |cpu| %.5f\n", L, md, mx);
            }
            for (int b = 0; b < nb; b++) for (int i = 0; i < H; i++) XB[b][i] += YB[b][i];
            T_EXP += now() - ta;
            continue;
        }
        if (GPUX && mode == 1 && nb <= 16) {  // the card kernels hold at most 16 tokens per expert  // cached experts on the card for every token routed to them, the rest on the CPU meanwhile
            static int gsl[NE], gps[NE + 1], gtok[BMAX * TOPK], ga[BMAX * TOPK]; static float GOUTM[BMAX * TOPK][H];
            int ng = 0, np = 0; NCU = 0;
            for (int u = 0; u < NU; u++) {
                int sl = ESLOT[L][UE[u]];
                if (sl >= 0 && ng < NE) {
                    gsl[ng] = sl; gps[ng++] = np; SLAST[L][sl] = TICK;
                    for (int a = USTART[u]; a < USTART[u + 1]; a++) { gtok[np] = UTOK[a]; ga[np++] = a; }
                } else CU[NCU++] = u;
            }
            gps[ng] = np; GX_HIT += np; GX_TOT += na;
            void *gh = ng ? gpux_begin_multi(L, &HNB[0][0], nb, ng, gsl, gps, gtok) : NULL;
            if (NCU) {
                run(jobB_gateup_c, NULL, NCU * 2 * FF);
                for (int c = 0; c < NCU; c++) for (int a = USTART[CU[c]]; a < USTART[CU[c] + 1]; a++) {
                    for (int k = 0; k < FF; k++) {
                        float g = GATEB[a][k] < 7.0f ? GATEB[a][k] : 7.0f, u = UPB[a][k];
                        u = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
                        ACTB[a][k] = g / (1.0f + expf(-g)) * u;
                    }
                    prep(&VAB[a], ACTB[a], FF);
                }
                run(jobB_down_c, NULL, NCU * H);
            }
            if (gh) gpux_end_multi(gh, &GOUTM[0][0], np);
            for (int c = 0; c < NCU; c++) for (int a = USTART[CU[c]]; a < USTART[CU[c] + 1]; a++) { float *x = XB[UTOK[a]]; for (int i = 0; i < H; i++) x[i] += UW[a] * DOWNB[a][i]; }
            for (int k = 0; k < np; k++) { float *x = XB[UTOK[ga[k]]]; for (int i = 0; i < H; i++) x[i] += UW[ga[k]] * GOUTM[k][i]; }
            for (int up = 0; up < MAXUP; up++) {  // the busiest experts the card did not have
                int best = -1; for (int c = 0; c < NCU; c++) { int u = CU[c]; if (UE[u] >= 0 && ESLOT[L][UE[u]] < 0 && (best < 0 || cnt[UE[u]] > cnt[UE[CU[best]]])) best = c; }
                if (best < 0) break;
                cache_put(L, UE[CU[best]]);
            }
            T_EXP += now() - ta;
            continue;
        }
#endif
        run(jobB_gateup, NULL, NU * 2 * FF);
        for (int a = 0; a < na; a++) {
            for (int k = 0; k < FF; k++) {
                float g = GATEB[a][k] < 7.0f ? GATEB[a][k] : 7.0f, u = UPB[a][k];
                u = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
                ACTB[a][k] = g / (1.0f + expf(-g)) * u;
            }
            prep(&VAB[a], ACTB[a], FF);
        }
        run(jobB_down, NULL, NU * H);
        for (int a = 0; a < na; a++) { float *x = XB[UTOK[a]]; for (int i = 0; i < H; i++) x[i] += UW[a] * DOWNB[a][i]; }
        T_EXP += now() - ta;
    }
    double ta = now();
    if (mode == 1) {  // the output head read once for every conversation in the step
        for (int b = 0; b < nb; b++) rmsnorm(HNB[b], XB[b], &NORM, H);
#ifdef GPU_HEAD
        if (GPUHEAD) gpu_head_run_multi(&HNB[0][0], nb, LOGB);
        else
#endif
        run(job_head_multi, NULL, VOCAB);
    } else for (int b = df ? 0 : nb - 1; b < nb; b++) {
        rmsnorm(HN, XB[b], &NORM, H);
        head();
        if (df) fwrite(LOGITS, sizeof(float), VOCAB, df);
    }
    T_HEAD += now() - ta;
}

static void job_route_small(void *a, int r0, int r1) {  // small packs: rows are experts, every token per row
    (void)a;
    for (int e = r0; e < r1; e++) {
        const uint16_t *p = BL->router.p + (size_t)e * H;
#ifdef __AVX2__
        __m256 acc[32];
        for (int b = 0; b < BN; b++) acc[b] = _mm256_setzero_ps();
        for (int c = 0; c < H; c += 8) {
            __m256 w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p + c))), 16));
            for (int b = 0; b < BN; b++) acc[b] = _mm256_fmadd_ps(w, _mm256_loadu_ps(HNB[b] + c), acc[b]);
        }
        for (int b = 0; b < BN; b++) RLG[b][e] = hsum(acc[b]);
#else
        for (int b = 0; b < BN; b++) RLG[b][e] = bf_row(&BL->router, e, HNB[b]);
#endif
    }
}
static void job_route_multi(void *a, int r0, int r1) {  // rows: blocks of 8 tokens, each against all 256 router rows (1 MB, stays in cache)
    (void)a;
    for (int blk = r0; blk < r1; blk++) {
        int b0 = blk * 8, nb = BN - b0 < 8 ? BN - b0 : 8;
        for (int e = 0; e < NE; e++) {
            const uint16_t *p = BL->router.p + (size_t)e * H;
#ifdef __AVX2__
            __m256 acc[8];
            for (int b = 0; b < nb; b++) acc[b] = _mm256_setzero_ps();
            for (int c = 0; c < H; c += 8) {
                __m256 w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p + c))), 16));
                for (int b = 0; b < nb; b++) acc[b] = _mm256_fmadd_ps(w, _mm256_loadu_ps(HNB[b0 + b] + c), acc[b]);
            }
            for (int b = 0; b < nb; b++) RLG[b0 + b][e] = hsum(acc[b]);
#else
            for (int b = 0; b < nb; b++) RLG[b0 + b][e] = bf_row(&BL->router, e, HNB[b0 + b]);
#endif
        }
    }
}
static void job_head_multi(void *a, int r0, int r1) {
    (void)a;
    for (int r = r0; r < r1; r++) {
        const int8_t *p = H8 + (size_t)r * H;
#ifdef __AVX2__
        __m256 acc[BMAX];
        for (int b = 0; b < BN; b++) acc[b] = _mm256_setzero_ps();
        for (int c = 0; c < H; c += 8) {
            __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(p + c))));
            for (int b = 0; b < BN; b++) acc[b] = _mm256_fmadd_ps(w, _mm256_loadu_ps(HNB[b] + c), acc[b]);
        }
        for (int b = 0; b < BN; b++) LOGB[(size_t)b * VOCAB + r] = hsum(acc[b]) * H8S[r];
#else
        for (int b = 0; b < BN; b++) { float s = 0; for (int c = 0; c < H; c++) s += p[c] * HNB[b][c]; LOGB[(size_t)b * VOCAB + r] = s * H8S[r]; }
#endif
    }
}

/* ---------- the prompt with whole layers on the card ---------- */
static int TOPB[BMAX][TOPK]; static float TWB[BMAX][TOPK];
static void job_topk(void *a, int r0, int r1) {  // rows: tokens; softmax over RLG, the 8 best renormalised
    (void)a;
    for (int b = r0; b < r1; b++) {
        float *lg = RLG[b], mx = -1e30f, sum = 0, tsum = 0;
        for (int e = 0; e < NE; e++) if (lg[e] > mx) mx = lg[e];
        for (int e = 0; e < NE; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
        for (int e = 0; e < NE; e++) lg[e] /= sum;
        for (int k = 0; k < TOPK; k++) {
            int best = -1; for (int e = 0; e < NE; e++) if (best < 0 || lg[e] > lg[best]) best = e;
            TOPB[b][k] = best; TWB[b][k] = lg[best]; tsum += lg[best]; lg[best] = -1;
        }
        for (int k = 0; k < TOPK; k++) TWB[b][k] /= (tsum + 1e-20f);
    }
}
static int group_pairs(int nb, const int *poss, int L) {  // TOPB/TWB -> pairs grouped by expert (NU, UE, USTART, UTOK, UW)
    int cnt[NE] = {0}, uidx[NE], fillc[NE] = {0};
    for (int b = 0; b < nb; b++) {
        for (int s = 0; s < TOPK; s++) { PFCNT[L][TOPB[b][s]]++; cnt[TOPB[b][s]]++; }
        if (RLOG) { fprintf(RLOG, "%d %d", L, poss[b]); for (int s = 0; s < TOPK; s++) fprintf(RLOG, " %d", TOPB[b][s]); fputc('\n', RLOG); }
    }
    NU = 0; int na = 0;
    for (int e = 0; e < NE; e++) if (cnt[e]) { uidx[e] = NU; UE[NU] = e; USTART[NU++] = na; na += cnt[e]; }
    USTART[NU] = na;
    for (int b = 0; b < nb; b++) for (int s = 0; s < TOPK; s++) {
        int u = uidx[TOPB[b][s]], slot = USTART[u] + fillc[u]++;
        UTOK[slot] = b; UW[slot] = TWB[b][s];
    }
    return na;
}
#ifdef GPU_HEAD
// GFULL_CHECK=1 (best with --exact): each layer also on the CPU from the card's own input, compared
static double CK_ATT[2], CK_RT[2]; static long CK_KV[2]; static int CK_TOP;
static void maxdiff(const float *a, const float *b, size_t n, double *out) {
    double md = 0, mx = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > md) md = d; if (fabs(a[i]) > mx) mx = fabs(a[i]); }
    out[0] = md; out[1] = mx;
}
static float CKX[BMAX][H];
static void check_attn(int L, int glob, int nb, const int *seqs, const int *poss) {
    static float rlc[BMAX][NE];
    for (int b = 0; b < nb; b++) { rmsnorm(HNB[b], XB[b], &BL->in_ln, H); prep(&VHB[b], HNB[b], H); }
    run(jobB_qkv, NULL, NH * HD + 2 * NKV * HD);
    for (int b = 0; b < nb; b++) {
        int pos = poss[b];
        for (int h = 0; h < NH; h++) { rmsnorm(QB[b] + h * HD, QB[b] + h * HD, &BL->q_norm, HD); if (!glob) rope(QB[b] + h * HD, pos); }
        for (int h = 0; h < NKV; h++) { rmsnorm(KB_[b] + h * HD, KB_[b] + h * HD, &BL->k_norm, HD); if (!glob) rope(KB_[b] + h * HD, pos); }
        to_f16(KC + KOFF(seqs[b], L, pos), KB_[b], NKV * HD); to_f16(VC + KOFF(seqs[b], L, pos), VB_[b], NKV * HD);
        ASEQ[b] = seqs[b]; APOS[b] = pos;
    }
    AL = L; AGLOB = glob;
    run(jobB_att, NULL, nb * NKV);
    for (int b = 0; b < nb; b++) prep(&VATTB[b], ATTB[b], H);
    run(jobB_o, NULL, H);
    for (int b = 0; b < nb; b++) for (int i = 0; i < H; i++) CKX[b][i] = XB[b][i] + YB[b][i];
    const uint16_t *ko; gfull_read(&XB[0][0], &ko);  // XB: the card's x after attention
    maxdiff(&CKX[0][0], &XB[0][0], (size_t)nb * H, CK_ATT);
    CK_KV[0] = CK_KV[1] = 0;
    for (int b = 0; b < nb; b++) {
        const uint16_t *k = ko + ((size_t)L * gfull_maxtok() + b) * 1024;
        for (int i = 0; i < 512; i++) { CK_KV[1] += 2; CK_KV[0] += k[i] != KC[KOFF(seqs[b], L, poss[b]) + i]; CK_KV[0] += k[512 + i] != VC[KOFF(seqs[b], L, poss[b]) + i]; }
    }
    memcpy(rlc, RLG, sizeof(float) * NE * nb);  // the card's logits
    for (int b = 0; b < nb; b++) rmsnorm(HNB[b], XB[b], &BL->post_ln, H);
    BN = nb; run(job_route_multi, NULL, (nb + 7) / 8);
    maxdiff(&RLG[0][0], &rlc[0][0], (size_t)nb * NE, CK_RT);
    run(job_topk, NULL, nb);  // the CPU's choice, against the card's
    static int topc[BMAX][TOPK]; memcpy(topc, TOPB, sizeof(int) * TOPK * nb);
    memcpy(RLG, rlc, sizeof(float) * NE * nb);
    run(job_topk, NULL, nb);
    memcpy(RLG, rlc, sizeof(float) * NE * nb);  // routing below works on the card's logits
    CK_TOP = 0;
    for (int b = 0; b < nb; b++) { int same = 0; for (int i = 0; i < TOPK; i++) for (int j = 0; j < TOPK; j++) same += topc[b][i] == TOPB[b][j]; CK_TOP += same != TOPK; }
}
static void check_moe(int L, int nb, int na) {
    for (int b = 0; b < nb; b++) prep(&VHB[b], HNB[b], H);
    run(jobB_gateup, NULL, NU * 2 * FF);
    for (int a = 0; a < na; a++) {
        for (int k = 0; k < FF; k++) {
            float g = GATEB[a][k] < 7.0f ? GATEB[a][k] : 7.0f, u = UPB[a][k];
            u = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
            ACTB[a][k] = g / (1.0f + expf(-g)) * u;
        }
        prep(&VAB[a], ACTB[a], FF);
    }
    run(jobB_down, NULL, NU * H);
    for (int a = 0; a < na; a++) { float *x = XB[UTOK[a]]; for (int i = 0; i < H; i++) x[i] += UW[a] * DOWNB[a][i]; }
    gfull_read(&CKX[0][0], NULL);
    double mo[2]; maxdiff(&XB[0][0], &CKX[0][0], (size_t)nb * H, mo);
    fprintf(stderr, "gfull check L%2d: attention %.2e of %.1f, k/v halves differing %ld/%ld, router %.2e of %.1f (top-8 differs on %d/%d tokens), experts %.2e of %.1f\n",
            L, CK_ATT[0], CK_ATT[1], CK_KV[0], CK_KV[1], CK_RT[0], CK_RT[1], CK_TOP, nb, mo[0], mo[1]);
}
static void step_batch_card(const int *toks, const int *seqs, const int *poss, int nb, FILE *df) {
    BN = nb;
    if (poss[0] + nb > CTX) { fprintf(stderr, "positions %d..%d past --ctx %d\n", poss[0], poss[0] + nb - 1, CTX); exit(1); }
    for (int b = 0; b < nb; b++) for (int i = 0; i < H; i++) XB[b][i] = bf(EMB.p[(size_t)toks[b] * H + i]);
    gfull_begin(&XB[0][0], nb, poss[0]);
    static int chk = -1; if (chk < 0) chk = getenv("GFULL_CHECK") != NULL;
    static int tp[BMAX * TOPK], fl[BMAX]; static uint32_t lc[NE * 12];
    for (int L = 0; L < NL; L++) {
        BL = &LY[L]; int glob = (L % 4) == 3; double ta = now();
        if (chk) gfull_read(&XB[0][0], NULL);  // the layer's input, for the CPU's version
        T_GPU += gfull_attn(L, glob, &RLG[0][0]);
        T_ATT += now() - ta;
        if (chk) check_attn(L, glob, nb, seqs, poss);
        ta = now();
        run(job_topk, NULL, nb);
        int na = group_pairs(nb, poss, L);
        for (int u = 0; u < NU; u++) memcpy(lc + u * 12, GLOC[L][UE[u]], sizeof GLOC[L][0]);
        memset(fl, 0, sizeof(int) * nb);
        for (int a = 0; a < na; a++) { int b = UTOK[a]; tp[b * TOPK + fl[b]++] = a; }
        T_ROUTE += now() - ta; ta = now();
        gfull_moe(NU, lc, GREG[L], USTART, UTOK, UW, tp);
        if (chk) check_moe(L, nb, na);
        T_EXP += now() - ta;
    }
    double ta = now();
    const uint16_t *ko;
    gfull_read(&XB[0][0], &ko);
    int mt = gfull_maxtok();
    for (int L = 0; L < NL; L++) for (int b = 0; b < nb; b++) {
        const uint16_t *k = ko + ((size_t)L * mt + b) * 1024;
        memcpy(KC + KOFF(seqs[b], L, poss[b]), k, NKV * HD * 2); memcpy(VC + KOFF(seqs[b], L, poss[b]), k + 512, NKV * HD * 2);
    }
    T_EXP += now() - ta; ta = now();
    for (int b = df ? 0 : nb - 1; b < nb; b++) { rmsnorm(HN, XB[b], &NORM, H); head(); if (df) fwrite(LOGITS, sizeof(float), VOCAB, df); }
    T_HEAD += now() - ta;
}
#endif

static int argmax(const float *v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: maple CKDIR --ids 1,2,3 [--n N] [--threads T] [--ctx C] [--dump FILE]\n"); return 1; }
    const char *dir = argv[1], *ids_s = NULL, *dump = NULL, *multi = NULL; int n_new = 64, nowarm = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids")) ids_s = argv[++i];
        else if (!strcmp(argv[i], "--n")) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx")) CTX = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump")) dump = argv[++i];
        else if (!strcmp(argv[i], "--exact")) { HEAD8 = 0; Q8ACT = 0; }
        else if (!strcmp(argv[i], "--f32act")) Q8ACT = 0;
        else if (!strcmp(argv[i], "--nowarm")) nowarm = 1;
        else if (!strcmp(argv[i], "--prefill1")) PREFILL1 = 1;
        else if (!strcmp(argv[i], "--gpuhead")) { GPUHEAD = 1; HEAD8 = 0; }
        else if (!strcmp(argv[i], "--gpuhead8")) GPUHEAD8 = 1;
        else if (!strcmp(argv[i], "--gpux")) { GPUX = 1; GSLOTS = atoi(argv[++i]); if (GSLOTS > 64) GSLOTS = 64; }
        else if (!strcmp(argv[i], "--maxup")) MAXUP = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--route-log")) RLOG = fopen(argv[++i], "w");
        else if (!strcmp(argv[i], "--multi")) multi = argv[++i];
        else if (!strcmp(argv[i], "--chunk")) CHUNK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpupre")) { GPRE = 1; CHUNK = atoi(argv[++i]); }
    }
    if (CHUNK < 1 || CHUNK > BMAX) { fprintf(stderr, "chunk must be 1..%d\n", BMAX); return 1; }
    int *ids = malloc(sizeof(int) * CTX), n = 0;
    for (const char *p = ids_s; p && *p; ) {
        if (n >= CTX - 1) { fprintf(stderr, "prompt longer than --ctx %d allows (%d tokens + 1 to generate)\n", CTX, CTX - 1); return 1; }
        ids[n++] = atoi(p); p = strchr(p, ','); if (p) p++;
    }
    // --multi FILE: one prompt per line (comma-separated ids), all generated together
    static int MIDS[MAXSEQ][4096], MN[MAXSEQ];
    if (multi) {
        FILE *mf = fopen(multi, "r"); static char line[1 << 17]; NSEQ = 0;
        if (!mf) { perror(multi); return 1; }
        while (NSEQ < MAXSEQ && fgets(line, sizeof line, mf)) {
            MN[NSEQ] = 0; for (const char *p = line; *p && *p != '\n'; ) {
                if (MN[NSEQ] >= 4096 || MN[NSEQ] >= CTX - 1) { fprintf(stderr, "--multi: prompt %d longer than --ctx %d allows\n", NSEQ + 1, CTX); return 1; }
                MIDS[NSEQ][MN[NSEQ]++] = atoi(p); p = strchr(p, ','); if (!p) break; p++;
            }
            if (MN[NSEQ]) NSEQ++;
        }
        fclose(mf);
        GPUHEAD8 = 0;  // multi mode: the head for the whole pack, on the card with --gpuhead, else CPU int8
        if (!GPUHEAD) HEAD8 = 1;
    }
    for (int i = 0; i < ROT / 2; i++) INV_FREQ[i] = 1.0 / pow(10000.0, (2.0 * i) / ROT);
    double t0 = now();
    load(dir);
    KC = calloc((size_t)NSEQ * NL * CTX * NKV * HD, sizeof(uint16_t)); VC = calloc((size_t)NSEQ * NL * CTX * NKV * HD, sizeof(uint16_t));
    LOGB = malloc(sizeof(float) * (size_t)MAXSEQ * VOCAB);
    SCR = malloc(sizeof(float) * (size_t)BMAX * NH * CTX);
    pthread_t th; for (int i = 1; i < NT; i++) pthread_create(&th, NULL, worker, (void *)(intptr_t)i);
    double tw = now(); size_t tot = 0; for (int k = 0; k < 9; k++) tot += FSZ[k];
    if (!nowarm) run(job_warm, NULL, (int)((tot >> 20) + 1));
#ifdef GPU_HEAD
    if (GPUHEAD8) { GPUHEAD = 0; HEAD8 = 1; }
    if (GPUHEAD && gpu_head_init(HEAD.p, VOCAB, H) != 0) { fprintf(stderr, "gpu head unavailable, using cpu\n"); GPUHEAD = 0; HEAD8 = 1; }
#endif
    if (HEAD8) { H8 = malloc((size_t)VOCAB * H); H8S = malloc(sizeof(float) * VOCAB); run(job_h8, NULL, VOCAB); }
#ifdef GPU_HEAD
    if (GPUHEAD8) { if (gpu_head_init_i8(H8, H8S, VOCAB, H) == 0) GPUHEAD = 1; else fprintf(stderr, "gpu int8 head unavailable, using cpu\n"); }
    if (GPUX) {
        for (int L = 0; L < NL; L++) { for (int e = 0; e < NE; e++) ESLOT[L][e] = -1; for (int k = 0; k < 64; k++) SEXP[L][k] = -1; }
        if (gpux_init(NL, GSLOTS) != 0) { fprintf(stderr, "gpu expert cache unavailable\n"); GPUX = 0; }
    }
    if (GPRE && gpre_init(CHUNK, TOPK) != 0) { fprintf(stderr, "prompt on the card unavailable, using cpu\n"); GPRE = 0; }
    if (GPRE) {  // per layer and shard: the page-aligned span of its experts becomes one region
        size_t pg = (size_t)getpagesize();
        for (int L = 0; L < NL && GPRE; L++) {
            size_t lo[9], hi[9]; int slot[9], ns = 0;
            for (int k = 0; k < 9; k++) { lo[k] = SIZE_MAX; hi[k] = 0; slot[k] = -1; }
            for (int pass = 0; pass < 2 && GPRE; pass++) for (int e = 0; e < NE; e++) {
                const T2 *w[3] = {&LY[L].eg[e], &LY[L].eu[e], &LY[L].ed[e]};
                for (int m = 0; m < 6; m++) {
                    const T2 *t = w[m / 2];
                    const uint8_t *q = m % 2 ? (const uint8_t *)t->s : t->p; size_t len = m % 2 ? (size_t)t->rows * 4 : (size_t)t->rows * t->cols / 4;
                    int f = -1; for (int k = 0; k < 9; k++) if (q >= FILES[k] && q + len <= FILES[k] + FSZ[k]) f = k;
                    if (f < 0 || ((q - FILES[f]) & 15)) { fprintf(stderr, "gpre: tensor not inside a shard or not 16-byte aligned\n"); GPRE = 0; break; }
                    size_t o = q - FILES[f];
                    if (pass == 0) { if (o < lo[f]) lo[f] = o; if (o + len > hi[f]) hi[f] = o + len; }
                    else { GLOC[L][e][m * 2] = slot[f]; GLOC[L][e][m * 2 + 1] = (uint32_t)(o - lo[f]); }
                }
                if (pass == 0 && e == NE - 1) for (int k = 0; k < 9; k++) if (hi[k]) {
                    lo[k] = lo[k] / pg * pg; hi[k] = (hi[k] + pg - 1) / pg * pg;
                    if (ns == 4) { fprintf(stderr, "gpre: layer %d spans more than 4 shards\n", L); GPRE = 0; break; }
                    int id = gpre_region(FILES[k] + lo[k], hi[k] - lo[k]);
                    if (id < 0) { GPRE = 0; break; }
                    slot[k] = ns; GSPAN[L][ns] = hi[k] - lo[k]; GREG[L][ns++] = id;
                }
            }
            for (int j = ns; j < 4; j++) GREG[L][j] = -1;
        }
        size_t most = 0;  // the biggest layer, for the staging buffer
        for (int L = 0; L < NL && GPRE; L++) { size_t t = 0; for (int j = 0; j < 4; j++) if (GREG[L][j] >= 0) t += (GSPAN[L][j] + 255) / 256 * 256; if (t > most) most = t; }
        if (GPRE && getenv("GPRE_STAGE") && gpre_stage(most) != 0) fprintf(stderr, "weights read over the bus instead\n");
        if (GPRE && !getenv("GPRE_ATTCPU")) {
            for (int L = 0; L < NL; L++) {
                const Layer *l = &LY[L];
                gpre_dense_add(L * 2, l->q.p, l->q.s, l->q.rows, l->q.cols); gpre_dense_add(L * 2, l->k.p, l->k.s, l->k.rows, l->k.cols);
                gpre_dense_add(L * 2, l->v.p, l->v.s, l->v.rows, l->v.cols); gpre_dense_add(L * 2 + 1, l->o.p, l->o.s, l->o.rows, l->o.cols);
            }
            GDENSE = gpre_dense_done(CHUNK) == 0;
        }
        if (GPRE && GDENSE && !getenv("GPRE_PART")) {
            float *nw = malloc(sizeof(float) * NL * 4352); uint16_t *rw = malloc((size_t)NL * NE * H * 2); float *cs = malloc(sizeof(float) * CTX * 64);
            for (int L = 0; L < NL; L++) {
                const Layer *l = &LY[L]; float *w = nw + (size_t)L * 4352;
                for (int i = 0; i < H; i++) { w[i] = bf(l->in_ln.p[i]); w[H + i] = bf(l->post_ln.p[i]); }
                for (int i = 0; i < HD; i++) { w[2 * H + i] = bf(l->q_norm.p[i]); w[2 * H + HD + i] = bf(l->k_norm.p[i]); }
                memcpy(rw + (size_t)L * NE * H, l->router.p, (size_t)NE * H * 2);
            }
            for (int pos = 0; pos < CTX; pos++) for (int i = 0; i < ROT / 2; i++) {
                double f = pos * INV_FREQ[i]; cs[(pos * 32 + i) * 2] = (float)cos(f); cs[(pos * 32 + i) * 2 + 1] = (float)sin(f);
            }
            GFULL = gfull_init(CTX, CHUNK, WIN, nw, rw, cs) == 0;
            free(nw); free(rw); free(cs);
            if (!GFULL) fprintf(stderr, "whole layers on the card unavailable: experts and projections only\n");
        }
        if (!GPRE) fprintf(stderr, "prompt on the card unavailable, using cpu\n");
    }
#else
    if (GPUX || GPUHEAD8 || GPRE) { fprintf(stderr, "built without -DGPU_HEAD: --gpux, --gpuhead8, --gpupre ignored\n"); GPUX = 0; GPRE = 0; }
#endif
    fprintf(stderr, "warm-up %.2f s (%s head, %s activations)\n", now() - tw, GPUHEAD ? (GPUHEAD8 ? "gpu int8" : "gpu bf16") : HEAD8 ? "int8" : "bf16", Q8ACT ? "int8" : "fp32");
    FILE *df = dump ? fopen(dump, "wb") : NULL;
    static int SEQ0[BMAX], POSB[BMAX];
    if (multi) {
        double tm = now(); int nexttok[MAXSEQ], pos_s[MAXSEQ], made_s[MAXSEQ] = {0}, active = NSEQ, total = 0;
        for (int sq = 0; sq < NSEQ; sq++) {  // each prompt in chunks, its first answer token from its own logits
            for (int i = 0; i < MN[sq]; i += CHUNK) {
                int nb = MN[sq] - i < CHUNK ? MN[sq] - i : CHUNK;
                for (int b = 0; b < nb; b++) { SEQ0[b] = sq; POSB[b] = i + b; }
                step_batch(MIDS[sq] + i, SEQ0, POSB, nb, NULL, 0);
            }
            nexttok[sq] = argmax(LOGITS, VOCAB); pos_s[sq] = MN[sq];
        }
#ifdef GPU_HEAD
        if (GPUX) {  // warm start from all prompts
            for (int L = 0; L < NL; L++) for (int k = 0; k < GSLOTS; k++) {
                int best = -1; for (int e = 0; e < NE; e++) if (ESLOT[L][e] < 0 && PFCNT[L][e] > 0 && (best < 0 || PFCNT[L][e] > PFCNT[L][best])) best = e;
                if (best < 0) break;
                cache_put(L, best);
            }
            fprintf(stderr, "gpux: %ld experts from the prompts put on the card\n", GX_UP); GX_UP = 0;
        }
#endif
        double tg = now();
        fprintf(stderr, "multi: %d conversations, prompts read in %.2f s\n", NSEQ, tg - tm);
        int done[MAXSEQ] = {0};
        while (active > 0) {
            int toks[MAXSEQ], sqs[MAXSEQ], pss[MAXSEQ], nb = 0;
            for (int sq = 0; sq < NSEQ; sq++) if (!done[sq]) {
                printf("%d %d\n", sq, nexttok[sq]); made_s[sq]++; total++;
                if (nexttok[sq] == EOS || made_s[sq] >= n_new || pos_s[sq] >= CTX - 1) { done[sq] = 1; active--; continue; }
                toks[nb] = nexttok[sq]; sqs[nb] = sq; pss[nb] = pos_s[sq]++; nb++;
            }
            fflush(stdout);
            if (!nb) break;
            step_batch(toks, sqs, pss, nb, NULL, 1);
            for (int b = 0; b < nb; b++) nexttok[sqs[b]] = argmax(LOGB + (size_t)b * VOCAB, VOCAB);
        }
        double te = now();
        fprintf(stderr, "multi: %d tokens in %.2f s = %.1f tok/s together (%d conversations)\n", total, te - tg, total / (te - tg), NSEQ);
        if (GPUX) fprintf(stderr, "gpux: %.1f%% of expert calls on the card, %ld uploads\n", 100.0 * GX_HIT / (GX_TOT + 1e-9), GX_UP);
        fprintf(stderr, "per step: attention %.1f ms, router %.1f ms, experts %.1f ms, head %.1f ms (whole run incl. prompts)\n", T_ATT * 1e3, T_ROUTE * 1e3, T_EXP * 1e3, T_HEAD * 1e3);
        return 0;
    }
    double t1 = now();
    if (PREFILL1) for (int i = 0; i < n; i++) { step(ids[i], i); if (df) fwrite(LOGITS, sizeof(float), VOCAB, df); }
    else for (int i = 0; i < n; i += CHUNK) {
        int nb = n - i < CHUNK ? n - i : CHUNK;
        for (int b = 0; b < nb; b++) { SEQ0[b] = 0; POSB[b] = i + b; }
        step_batch(ids + i, SEQ0, POSB, nb, df, 0);
    }
    double t2 = now();
    fprintf(stderr, "load %.2f s, prompt %d tokens %.2f s (%.2f tok/s), chunks of %d%s\n", t1 - t0, n, t2 - t1, n / (t2 - t1), CHUNK, GPRE ? ", experts on the card" : "");
    fprintf(stderr, "prompt: attention %.2f s, router %.2f s, experts %.2f s (card busy %.2f s), head %.2f s\n", T_ATT, T_ROUTE, T_EXP, T_GPU, T_HEAD);
#ifdef GPU_HEAD
    if (GPRE) fprintf(stderr, "prompt: weight copies to VRAM %.2f s (overlapped)\n", gpre_blit_seconds());
    if (GFULL) gfull_report();
#endif
#ifdef GPU_HEAD
    if (GPUX) {  // warm start: the experts the prompt used most, per layer
        double tc = now();
        for (int L = 0; L < NL; L++) for (int k = 0; k < GSLOTS; k++) {
            int best = -1; for (int e = 0; e < NE; e++) if (ESLOT[L][e] < 0 && PFCNT[L][e] > 0 && (best < 0 || PFCNT[L][e] > PFCNT[L][best])) best = e;
            if (best < 0) break;
            cache_put(L, best);
        }
        fprintf(stderr, "gpux: %ld experts from the prompt put on the card in %.2f s\n", GX_UP, now() - tc);
        GX_UP = 0; t2 = now();
    }
#endif
    int pos = n, made = 0;
    double g_att = T_ATT, g_rt = T_ROUTE, g_exp = T_EXP, g_head = T_HEAD, g_sc = T_SC;
    for (; made < n_new && pos < CTX; made++) {
        int nx = argmax(LOGITS, VOCAB);
        printf("%d\n", nx); fflush(stdout);
        if (nx == EOS) break;
        step(nx, pos++);
        if (df) fwrite(LOGITS, sizeof(float), VOCAB, df);
    }
    double t3 = now();
    fprintf(stderr, "per token over all steps: attention %.1f ms, router %.1f ms, experts %.1f ms, head %.1f ms\n", T_ATT * 1e3 / (n + made), T_ROUTE * 1e3 / (n + made), T_EXP * 1e3 / (n + made), T_HEAD * 1e3 / (n + made));
    fprintf(stderr, "generated %d tokens %.2f s (%.2f tok/s)\n", made, t3 - t2, made / (t3 - t2 + 1e-9));
    if (made) fprintf(stderr, "per generated token: cache pass %.2f ms of attention\n", (T_SC - g_sc) * 1e3 / made);
    if (made) fprintf(stderr, "per generated token: attention %.2f ms, router %.2f ms, experts %.2f ms, head %.2f ms\n", (T_ATT - g_att) * 1e3 / made, (T_ROUTE - g_rt) * 1e3 / made, (T_EXP - g_exp) * 1e3 / made, (T_HEAD - g_head) * 1e3 / made);
    if (GPUX) fprintf(stderr, "gpux: hits %.1f%% of %ld expert calls in generation, %ld uploads\n", 100.0 * GX_HIT / (GX_TOT + 1e-9), GX_TOT, GX_UP);
    if (df) fclose(df);
    if (RLOG) fclose(RLOG);
    return 0;
}
