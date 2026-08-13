#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

#ifdef __CUDACC__
#define KM_HD __host__ __device__
#else
#define KM_HD
#endif


typedef struct {
    int w, h;
    int64_t n;            /* w * h */
    uint8_t *r, *g, *b;   /* three planes, n bytes each */
} ImageSoA;

/* PCG32 - explicit PRNG.*/

typedef struct { uint64_t state, inc; } pcg32_t;

static inline uint32_t pcg32_next(pcg32_t *rng)
{
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

static inline void pcg32_init(pcg32_t *rng, uint64_t seed, uint64_t seq)
{
    rng->state = 0u;
    rng->inc = (seq << 1u) | 1u;
    pcg32_next(rng);
    rng->state += seed;
    pcg32_next(rng);
}

static inline uint64_t pcg32_next64(pcg32_t *rng)
{
    uint64_t hi = (uint64_t)pcg32_next(rng);
    return (hi << 32) | (uint64_t)pcg32_next(rng);
}

static inline uint64_t pcg32_bounded64(pcg32_t *rng, uint64_t bound)
{
    if (bound == 0) return 0;
    uint64_t threshold = (0u - bound) % bound;   /* == 2^64 mod bound */
    for (;;) {
        uint64_t v = pcg32_next64(rng);
        if (v >= threshold) return v % bound;
    }
}

/* The two functions that msust be bit-identical everywhere*/
KM_HD inline float km_dist2(float pr, float pg, float pb,
                            float cr, float cg, float cb)
{
    float dr = pr - cr;
    float dg = pg - cg;
    float db = pb - cb;
    return dr * dr + dg * dg + db * db;
}

/* Nearest-centroid search. Strict '<' => lowest index wins ties. */
KM_HD inline int km_assign_pixel(float pr, float pg, float pb,
                                 const float *cr, const float *cg,
                                 const float *cb, int K)
{
    int best = 0;
    float bestd = km_dist2(pr, pg, pb, cr[0], cg[0], cb[0]);
    for (int k = 1; k < K; ++k) {
        float d = km_dist2(pr, pg, pb, cr[k], cg[k], cb[k]);
        if (d < bestd) { bestd = d; best = k; }
    }
    return best;
}

KM_HD inline uint32_t km_dist2_u8(uint8_t r1, uint8_t g1, uint8_t b1,
                                  uint8_t r2, uint8_t g2, uint8_t b2)
{
    int dr = (int)r1 - (int)r2;
    int dg = (int)g1 - (int)g2;
    int db = (int)b1 - (int)b2;
    return (uint32_t)(dr * dr + dg * dg + db * db);
}

KM_HD inline float km_centroid(uint64_t sum, uint64_t cnt)
{
    return (float)((double)sum / (double)cnt);
}

/* cr/cg/cb must have room for K floats. Returns 0 on success. */
static int kmeanspp_seed(const ImageSoA *img, int K, uint64_t seed,
                         float *cr, float *cg, float *cb)
{
    const int64_t n = img->n;
    if (n <= 0 || K <= 0) return -1;

    uint32_t *d2 = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!d2) { fprintf(stderr, "kmeanspp_seed: out of memory\n"); return -1; }

    pcg32_t rng;
    pcg32_init(&rng, seed, 0xda3e39cb94b95bdbULL);

    /* First centre: uniform over pixels. */
    int64_t pick = (int64_t)pcg32_bounded64(&rng, (uint64_t)n);
    cr[0] = (float)img->r[pick];
    cg[0] = (float)img->g[pick];
    cb[0] = (float)img->b[pick];

    for (int64_t i = 0; i < n; ++i)
        d2[i] = km_dist2_u8(img->r[i], img->g[i], img->b[i],
                            img->r[pick], img->g[pick], img->b[pick]);

    for (int k = 1; k < K; ++k) {
        uint64_t total = 0;
        for (int64_t i = 0; i < n; ++i) total += d2[i];

        if (total == 0) {
            pick = (int64_t)pcg32_bounded64(&rng, (uint64_t)n);
        } else {
            uint64_t target = pcg32_bounded64(&rng, total);
            uint64_t cum = 0;
            pick = n - 1;                       /* guards fp-free edge cases */
            for (int64_t i = 0; i < n; ++i) {
                cum += d2[i];
                if (cum > target) { pick = i; break; }
            }
        }

        cr[k] = (float)img->r[pick];
        cg[k] = (float)img->g[pick];
        cb[k] = (float)img->b[pick];

        for (int64_t i = 0; i < n; ++i) {
            uint32_t d = km_dist2_u8(img->r[i], img->g[i], img->b[i],
                                     img->r[pick], img->g[pick], img->b[pick]);
            if (d < d2[i]) d2[i] = d;
        }
    }

    free(d2);
    return 0;
}

/* Image I/O*/
#ifdef KM_IMPLEMENT_IO
#if defined(__CUDACC__)
#pragma nv_diag_suppress 550
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__CUDACC__)
#pragma nv_diag_default 550
#endif

static int load_image_soa(const char *path, ImageSoA *out)
{
    int w, h, comp;
    uint8_t *rgb = stbi_load(path, &w, &h, &comp, 3);   /* force 3 channels */
    if (!rgb) {
        fprintf(stderr, "load_image_soa: cannot read '%s': %s\n",
                path, stbi_failure_reason());
        return -1;
    }
    int64_t n = (int64_t)w * (int64_t)h;

    out->w = w; out->h = h; out->n = n;
    out->r = (uint8_t *)malloc((size_t)n);
    out->g = (uint8_t *)malloc((size_t)n);
    out->b = (uint8_t *)malloc((size_t)n);
    if (!out->r || !out->g || !out->b) {
        fprintf(stderr, "load_image_soa: out of memory\n");
        stbi_image_free(rgb);
        return -1;
    }
    for (int64_t i = 0; i < n; ++i) {          /* interleaved -> SoA planes */
        out->r[i] = rgb[3 * i + 0];
        out->g[i] = rgb[3 * i + 1];
        out->b[i] = rgb[3 * i + 2];
    }
    stbi_image_free(rgb);
    return 0;
}

static void free_image_soa(ImageSoA *img)
{
    free(img->r); free(img->g); free(img->b);
    img->r = img->g = img->b = NULL;
    img->n = 0; img->w = img->h = 0;
}

static inline uint8_t km_quantize(float v)
{
    long q = lrintf(v);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (uint8_t)q;
}

/* Repaint every pixel with its cluster's centroid colour. */
static int write_segmented_png(const char *path, const ImageSoA *img,
                               const uint16_t *labels, int K,
                               const float *cr, const float *cg,
                               const float *cb)
{
    (void)K;
    int64_t n = img->n;
    uint8_t *rgb = (uint8_t *)malloc((size_t)n * 3);
    if (!rgb) return -1;
    for (int64_t i = 0; i < n; ++i) {
        int c = labels[i];
        rgb[3 * i + 0] = km_quantize(cr[c]);
        rgb[3 * i + 1] = km_quantize(cg[c]);
        rgb[3 * i + 2] = km_quantize(cb[c]);
    }
    int ok = stbi_write_png(path, img->w, img->h, 3, rgb, img->w * 3);
    free(rgb);
    return ok ? 0 : -1;
}
#endif /* KM_IMPLEMENT_IO */

/* Result dumping*/

/* Writes:
 *   <prefix>_labels.bin     raw uint16 x n
 *   <prefix>_centroids.txt  K lines of raw float BIT PATTERNS in hex, so the
 *                           comparison is exact and free of any decimal
 *                           printing ambiguity
 *   <prefix>_meta.txt       K / n / iterations
 */
static int dump_result(const char *prefix, const ImageSoA *img,
                       const uint16_t *labels, int K, int iters,
                       const float *cr, const float *cg, const float *cb)
{
    char path[1024];
    FILE *f;

    snprintf(path, sizeof(path), "%s_labels.bin", prefix);
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "dump_result: cannot write %s\n", path); return -1; }
    if (fwrite(labels, sizeof(uint16_t), (size_t)img->n, f) != (size_t)img->n) {
        fprintf(stderr, "dump_result: short write on %s\n", path);
        fclose(f); return -1;
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s_centroids.txt", prefix);
    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "dump_result: cannot write %s\n", path); return -1; }
    for (int k = 0; k < K; ++k) {
        uint32_t br, bg, bb;
        memcpy(&br, &cr[k], 4);
        memcpy(&bg, &cg[k], 4);
        memcpy(&bb, &cb[k], 4);
        fprintf(f, "%08x %08x %08x\n", br, bg, bb);
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s_meta.txt", prefix);
    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "dump_result: cannot write %s\n", path); return -1; }
    fprintf(f, "K %d\nn %lld\niters %d\n", K, (long long)img->n, iters);
    fclose(f);

    return 0;
}

/*Timing*/
typedef std::chrono::steady_clock km_clock;

struct Timer {
    km_clock::time_point t0;
    Timer() { reset(); }
    void reset() { t0 = km_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(km_clock::now() - t0).count();
    }
};

#endif
