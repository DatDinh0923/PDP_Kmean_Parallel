#define KM_IMPLEMENT_IO
#include "kmeans_common.h"

#include <vector>

struct Result {
    int K = 0, iters = 0;
    int64_t n = 0;
    std::vector<uint16_t> labels;
    std::vector<float> cr, cg, cb;
};

static int read_meta(const char *prefix, Result *out)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s_meta.txt", prefix);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    long long n = 0;
    if (fscanf(f, "K %d\nn %lld\niters %d\n", &out->K, &n, &out->iters) != 3) {
        fprintf(stderr, "malformed %s\n", path);
        fclose(f); return -1;
    }
    out->n = (int64_t)n;
    fclose(f);
    return 0;
}

static int read_result(const char *prefix, Result *out)
{
    if (read_meta(prefix, out) != 0) return -1;

    char path[1024];
    snprintf(path, sizeof(path), "%s_labels.bin", prefix);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    out->labels.resize((size_t)out->n);
    size_t got = fread(out->labels.data(), sizeof(uint16_t), (size_t)out->n, f);
    fclose(f);
    if (got != (size_t)out->n) {
        fprintf(stderr, "%s: expected %lld labels, got %zu\n",
                path, (long long)out->n, got);
        return -1;
    }

    snprintf(path, sizeof(path), "%s_centroids.txt", prefix);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    out->cr.resize(out->K); out->cg.resize(out->K); out->cb.resize(out->K);
    for (int k = 0; k < out->K; ++k) {
        uint32_t br, bg, bb;
        if (fscanf(f, "%x %x %x", &br, &bg, &bb) != 3) {
            fprintf(stderr, "%s: malformed at line %d\n", path, k + 1);
            fclose(f); return -1;
        }
        memcpy(&out->cr[k], &br, 4);
        memcpy(&out->cg[k], &bg, 4);
        memcpy(&out->cb[k], &bb, 4);
    }
    fclose(f);
    return 0;
}

static double compute_sse(const ImageSoA *img, const Result *res)
{
    double sse = 0.0;
    for (int64_t i = 0; i < img->n; ++i) {
        int c = res->labels[(size_t)i];
        double dr = (double)img->r[i] - (double)res->cr[c];
        double dg = (double)img->g[i] - (double)res->cg[c];
        double db = (double)img->b[i] - (double)res->cb[c];
        sse += dr * dr + dg * dg + db * db;
    }
    return sse;
}

static int popcount32(uint32_t v)
{
    int c = 0;
    while (v) { c += (int)(v & 1u); v >>= 1; }
    return c;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <prefixA> <prefixB> [image.png]\n", argv[0]);
        return 2;
    }
    Result A, B;
    if (read_result(argv[1], &A) != 0) return 1;
    if (read_result(argv[2], &B) != 0) return 1;

    printf("A = %s\nB = %s\n\n", argv[1], argv[2]);

    int fail = 0;

    if (A.K != B.K || A.n != B.n) {
        printf("  MISMATCH: shape  A(K=%d n=%lld)  B(K=%d n=%lld)\n",
               A.K, (long long)A.n, B.K, (long long)B.n);
        return 1;                      
    }

    if (A.iters == B.iters) {
        printf("  iterations      : %d == %d   OK\n", A.iters, B.iters);
    } else {
        printf("  iterations      : %d != %d   MISMATCH\n", A.iters, B.iters);
        fail = 1;
    }

    int64_t agree = 0, first_bad = -1;
    for (int64_t i = 0; i < A.n; ++i) {
        if (A.labels[(size_t)i] == B.labels[(size_t)i]) ++agree;
        else if (first_bad < 0) first_bad = i;
    }
    double pct = 100.0 * (double)agree / (double)A.n;
    printf("  label agreement : %lld/%lld = %.6f%%   %s\n",
           (long long)agree, (long long)A.n, pct,
           agree == A.n ? "OK" : "MISMATCH");
    if (agree != A.n) {
        fail = 1;
        printf("                    first differing pixel: index %lld (A=%u B=%u)\n",
               (long long)first_bad,
               (unsigned)A.labels[(size_t)first_bad],
               (unsigned)B.labels[(size_t)first_bad]);
    }

    int diff_bits = 0, diff_comp = 0;
    float max_abs = 0.0f;
    for (int k = 0; k < A.K; ++k) {
        const float *av[3] = { &A.cr[k], &A.cg[k], &A.cb[k] };
        const float *bv[3] = { &B.cr[k], &B.cg[k], &B.cb[k] };
        for (int c = 0; c < 3; ++c) {
            uint32_t x, y;
            memcpy(&x, av[c], 4);
            memcpy(&y, bv[c], 4);
            if (x != y) {
                ++diff_comp;
                diff_bits += popcount32(x ^ y);
                float d = fabsf(*av[c] - *bv[c]);
                if (d > max_abs) max_abs = d;
            }
        }
    }
    printf("  centroid bits   : %d differing bits across %d/%d components   %s\n",
           diff_bits, diff_comp, A.K * 3, diff_comp == 0 ? "OK" : "MISMATCH");
    if (diff_comp) {
        fail = 1;
        printf("                    max abs difference: %.9g\n", (double)max_abs);
    }

    if (argc >= 4) {
        ImageSoA img;
        if (load_image_soa(argv[3], &img) == 0) {
            if (img.n != A.n) {
                printf("  SSE             : skipped (image has %lld pixels, results have %lld)\n",
                       (long long)img.n, (long long)A.n);
            } else {
                double sa = compute_sse(&img, &A), sb = compute_sse(&img, &B);
                printf("  SSE A           : %.10e\n", sa);
                printf("  SSE B           : %.10e\n", sb);
                printf("  SSE delta       : %.10e   %s\n", sb - sa,
                       sa == sb ? "OK" : "MISMATCH");
                if (sa != sb) fail = 1;
            }
            free_image_soa(&img);
        }
    }

    printf("\n  ==> %s\n", fail ? "DIFFERENT" : "IDENTICAL");
    return fail;
}
