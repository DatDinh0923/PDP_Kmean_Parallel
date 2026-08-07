#define KM_IMPLEMENT_IO
#include "kmeans_common.h"

#include <vector>
#include <algorithm>

#ifndef KM_TAG
#define KM_TAG "seq"
#endif

static int64_t lloyd_pass(const ImageSoA *img, int K,
                          const float *cr, const float *cg, const float *cb,
                          uint16_t *labels,
                          uint64_t *sumR, uint64_t *sumG, uint64_t *sumB,
                          uint64_t *cnt)
{
    const int64_t n = img->n;
    const uint8_t *R = img->r, *G = img->g, *B = img->b;
    int64_t changed = 0;

    memset(sumR, 0, (size_t)K * sizeof(uint64_t));
    memset(sumG, 0, (size_t)K * sizeof(uint64_t));
    memset(sumB, 0, (size_t)K * sizeof(uint64_t));
    memset(cnt,  0, (size_t)K * sizeof(uint64_t));

    for (int64_t i = 0; i < n; ++i) {
        uint8_t pr = R[i], pg = G[i], pb = B[i];
        int c = km_assign_pixel((float)pr, (float)pg, (float)pb, cr, cg, cb, K);

        if (labels[i] != (uint16_t)c) { labels[i] = (uint16_t)c; ++changed; }

        /* uint64 accumulation -- exact, and therefore order-independent. */
        sumR[c] += pr;
        sumG[c] += pg;
        sumB[c] += pb;
        cnt[c]  += 1;
    }
    return changed;
}

/* Runs Lloyd to convergence. labels/cr/cg/cb are updated in place. */
static int lloyd_run(const ImageSoA *img, int K, int max_iter,
                     float *cr, float *cg, float *cb, uint16_t *labels,
                     int64_t *last_changed)
{
    std::vector<uint64_t> sumR(K), sumG(K), sumB(K), cnt(K);
    int iters = 0;
    int64_t changed = 0;

    for (int it = 0; it < max_iter; ++it) {
        changed = lloyd_pass(img, K, cr, cg, cb, labels,
                             sumR.data(), sumG.data(), sumB.data(), cnt.data());
        iters = it + 1;
        if (changed == 0) break;

        for (int k = 0; k < K; ++k) {
            if (cnt[k] == 0) continue;          /* empty cluster keeps its centroid */
            cr[k] = km_centroid(sumR[k], cnt[k]);
            cg[k] = km_centroid(sumG[k], cnt[k]);
            cb[k] = km_centroid(sumB[k], cnt[k]);
        }
    }
    *last_changed = changed;
    return iters;
}

static double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
            "usage: %s <image> <K> <seed> <max_iter> <out_prefix> "
            "[--reps N] [--no-png]\n", argv[0]);
        return 2;
    }

    const char *img_path = argv[1];
    int         K        = atoi(argv[2]);
    uint64_t    seed     = strtoull(argv[3], NULL, 10);
    int         max_iter = atoi(argv[4]);
    const char *prefix   = argv[5];

    int reps = 1, write_png = 1;
    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-png")) write_png = 0;
        else { fprintf(stderr, "unknown option '%s'\n", argv[i]); return 2; }
    }
    if (K < 1 || K > 65535) { fprintf(stderr, "K must be in [1,65535]\n"); return 2; }
    if (max_iter < 1) { fprintf(stderr, "max_iter must be >= 1\n"); return 2; }
    if (reps < 1) reps = 1;

    /*  load Image */
    ImageSoA img;
    Timer t;
    if (load_image_soa(img_path, &img) != 0) return 1;
    double load_ms = t.ms();
    if (img.n > 0 && (int64_t)K > img.n) {
        fprintf(stderr, "K (%d) exceeds pixel count (%lld)\n", K, (long long)img.n);
        free_image_soa(&img);
        return 2;
    }

    /* seeding */
    std::vector<float> seed_r(K), seed_g(K), seed_b(K);
    t.reset();
    if (kmeanspp_seed(&img, K, seed, seed_r.data(), seed_g.data(), seed_b.data()) != 0) {
        free_image_soa(&img);
        return 1;
    }
    double seed_ms = t.ms();

    /* Lloyd */
    std::vector<float> cr(K), cg(K), cb(K);
    std::vector<uint16_t> labels((size_t)img.n);
    std::vector<double> times;
    int iters = 0;
    int64_t last_changed = 0;

    for (int rep = 0; rep < reps + 1; ++rep) {      /* rep 0 is the warm-up */
        cr = seed_r; cg = seed_g; cb = seed_b;
        std::fill(labels.begin(), labels.end(), (uint16_t)0xFFFF);

        t.reset();
        iters = lloyd_run(&img, K, max_iter, cr.data(), cg.data(), cb.data(),
                          labels.data(), &last_changed);
        double ms = t.ms();
        if (rep > 0) times.push_back(ms);
    }
    double lloyd_ms = median_of(times);

    /* output */
    t.reset();
    if (dump_result(prefix, &img, labels.data(), K, iters,
                    cr.data(), cg.data(), cb.data()) != 0) {
        free_image_soa(&img);
        return 1;
    }
    if (write_png) {
        char png[1024];
        snprintf(png, sizeof(png), "%s_seg.png", prefix);
        if (write_segmented_png(png, &img, labels.data(), K,
                                cr.data(), cg.data(), cb.data()) != 0)
            fprintf(stderr, "warning: could not write %s\n", png);
    }
    double write_ms = t.ms();

    printf("[%s] %s  %dx%d  n=%lld  K=%d  seed=%llu\n", KM_TAG, img_path,
           img.w, img.h, (long long)img.n, K, (unsigned long long)seed);
    printf("[%s] iters=%d  last_changed=%lld  %s\n", KM_TAG, iters,
           (long long)last_changed,
           last_changed == 0 ? "(converged)" : "(hit max_iter)");
    printf("[%s] load=%.2fms  seed=%.2fms  lloyd=%.2fms (median of %d)  write=%.2fms\n",
           KM_TAG, load_ms, seed_ms, lloyd_ms, reps, write_ms);
    printf("CSV,seq,%s,%s,%d,%d,%lld,%d,%llu,%d,%.4f,%.4f,%.4f,%.4f,,\n",
           KM_TAG, img_path, img.w, img.h, (long long)img.n, K,
           (unsigned long long)seed, iters, load_ms, seed_ms, lloyd_ms, write_ms);

    free_image_soa(&img);
    return 0;
}
