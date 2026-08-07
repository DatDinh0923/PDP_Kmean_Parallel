#define KM_IMPLEMENT_IO
#include "kmeans_common.h"

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s <width> <height> <seed> <out.png>\n", argv[0]);
        return 2;
    }
    int w = atoi(argv[1]), h = atoi(argv[2]);
    uint64_t seed = strtoull(argv[3], NULL, 10);
    const char *out = argv[4];
    if (w < 1 || h < 1) { fprintf(stderr, "bad dimensions\n"); return 2; }

    pcg32_t rng;
    pcg32_init(&rng, seed, 0x853c49e6748fea9bULL);

    /* A handful of coloured blobs over a gradient background. */
    const int NB = 12;
    float bx[NB], by[NB], brad[NB], bcr[NB], bcg[NB], bcb[NB];
    for (int i = 0; i < NB; ++i) {
        bx[i]   = (float)pcg32_bounded64(&rng, (uint64_t)w);
        by[i]   = (float)pcg32_bounded64(&rng, (uint64_t)h);
        brad[i] = (float)(pcg32_bounded64(&rng, (uint64_t)(w < h ? w : h) / 4 + 1) + 8);
        bcr[i]  = (float)pcg32_bounded64(&rng, 256);
        bcg[i]  = (float)pcg32_bounded64(&rng, 256);
        bcb[i]  = (float)pcg32_bounded64(&rng, 256);
    }

    size_t n = (size_t)w * (size_t)h;
    uint8_t *rgb = (uint8_t *)malloc(n * 3);
    if (!rgb) { fprintf(stderr, "out of memory\n"); return 1; }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float fx = (float)x / (float)w, fy = (float)y / (float)h;

            /* Background gradient. */
            float r = 40.0f + 160.0f * fx;
            float g = 60.0f + 140.0f * fy;
            float b = 200.0f - 120.0f * (fx + fy) * 0.5f;

            /* Blend in any blob covering this pixel. */
            for (int i = 0; i < NB; ++i) {
                float dx = (float)x - bx[i], dy = (float)y - by[i];
                float d = sqrtf(dx * dx + dy * dy);
                if (d < brad[i]) {
                    float wgt = 1.0f - d / brad[i];
                    wgt = wgt * wgt;
                    r = r * (1.0f - wgt) + bcr[i] * wgt;
                    g = g * (1.0f - wgt) + bcg[i] * wgt;
                    b = b * (1.0f - wgt) + bcb[i] * wgt;
                }
            }

            /* Mild noise so clusters are not perfectly separable. */
            int nz = (int)(pcg32_next(&rng) % 21) - 10;
            size_t o = ((size_t)y * (size_t)w + (size_t)x) * 3;
            rgb[o + 0] = km_quantize(r + (float)nz);
            rgb[o + 1] = km_quantize(g + (float)nz);
            rgb[o + 2] = km_quantize(b + (float)nz);
        }
    }

    int ok = stbi_write_png(out, w, h, 3, rgb, w * 3);
    free(rgb);
    if (!ok) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    printf("wrote %s (%dx%d, %zu pixels)\n", out, w, h, n);
    return 0;
}
