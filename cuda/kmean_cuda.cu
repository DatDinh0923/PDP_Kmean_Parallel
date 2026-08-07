#define KM_IMPLEMENT_IO
#include "kmeans_common.h"

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do {                                                   \
    cudaError_t err__ = (x);                                                 \
    if (err__ != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err__));              \
        exit(1);                                                             \
    }                                                                        \
} while (0)

#define KM_MAX_CONST_K 1024
#define KM_BLOCK       256


__constant__ float c_cr[KM_MAX_CONST_K];
__constant__ float c_cg[KM_MAX_CONST_K];
__constant__ float c_cb[KM_MAX_CONST_K];
__device__ __forceinline__ int assign_const(float pr, float pg, float pb, int K)
{
    int best = 0;
    float bestd = km_dist2(pr, pg, pb, c_cr[0], c_cg[0], c_cb[0]);
    for (int k = 1; k < K; ++k) {
        float d = km_dist2(pr, pg, pb, c_cr[k], c_cg[k], c_cb[k]);
        if (d < bestd) { bestd = d; best = k; }
    }
    return best;
}

/* Dynamic shared memory, reinterpreted per version. */
extern __shared__ unsigned char s_raw[];

/* v1-CONSTANT MEMORY*/


__global__ void k_assign_v1(const uint8_t *__restrict__ R, /*__restrict__ promises that these pointer arrays do not overlap with each other, help the compiler optimize memory access*/
                            const uint8_t *__restrict__ G,
                            const uint8_t *__restrict__ B,
                            uint16_t *__restrict__ labels,
                            unsigned long long *__restrict__ sumR,
                            unsigned long long *__restrict__ sumG,
                            unsigned long long *__restrict__ sumB,
                            unsigned long long *__restrict__ cnt, /*number of pixels*/
                            unsigned long long *__restrict__ changed, /*Counts how many pixels moved to a different cluster during the current iteration.*/
                            int64_t n, int K) /*total number of pixels n = image width × image height*/
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint8_t pr = R[i], pg = G[i], pb = B[i]; /*Load one RGB pixel from global memory.*/

    /*Convert the RGB values to float, then compare the pixel against constant-memory centroids.
    c is the selected cluster number.*/
    int c = assign_const((float)pr, (float)pg, (float)pb, K);   /* constant cache */

    if (labels[i] != (uint16_t)c) {
        labels[i] = (uint16_t)c;
        atomicAdd(changed, 1ULL);                 /* one global atomic per flip */
    }

    atomicAdd(&sumR[c], (unsigned long long)pr);
    atomicAdd(&sumG[c], (unsigned long long)pg);
    atomicAdd(&sumB[c], (unsigned long long)pb);
    atomicAdd(&cnt[c],  1ULL);
}

__global__ void k_assign_v2(const uint8_t *__restrict__ R,
                            const uint8_t *__restrict__ G,
                            const uint8_t *__restrict__ B,
                            uint16_t *__restrict__ labels,
                            const float *__restrict__ cr, /*pointing to centroid arrays in global GPU memory.*/
                            const float *__restrict__ cg,
                            const float *__restrict__ cb,
                            unsigned long long *__restrict__ sumR,
                            unsigned long long *__restrict__ sumG,
                            unsigned long long *__restrict__ sumB,
                            unsigned long long *__restrict__ cnt,
                            unsigned long long *__restrict__ changed,
                            int64_t n, int K)
{   
    float *tcr = (float *)s_raw;
    float *tcg = tcr + K;
    float *tcb = tcg + K;

    /* Cooperative load: each centroid is fetched from VRAM exactly once per
     * block instead of once per pixel. */
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        tcr[k] = cr[k]; tcg[k] = cg[k]; tcb[k] = cb[k];
    }
    __syncthreads(); /*No thread may continue until every thread in this block has reached this line.*/
                    /*Without synchronization, a thread could start reading tcr[k] before another thread has written it.*/
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint8_t pr = R[i], pg = G[i], pb = B[i];
    /* The shared host/device function, reading the shared-memory tile. */
    int c = km_assign_pixel((float)pr, (float)pg, (float)pb, tcr, tcg, tcb, K);

    if (labels[i] != (uint16_t)c) {
        labels[i] = (uint16_t)c;
        atomicAdd(changed, 1ULL);
    }

    /* Still four GLOBAL atomics per pixel. */
    atomicAdd(&sumR[c], (unsigned long long)pr);
    atomicAdd(&sumG[c], (unsigned long long)pg);
    atomicAdd(&sumB[c], (unsigned long long)pb);
    atomicAdd(&cnt[c],  1ULL);
}

__global__ void k_assign_v3(const uint8_t *__restrict__ R,
                            const uint8_t *__restrict__ G,
                            const uint8_t *__restrict__ B,
                            uint16_t *__restrict__ labels,
                            const float *__restrict__ cr,
                            const float *__restrict__ cg,
                            const float *__restrict__ cb,
                            unsigned long long *__restrict__ sumR,
                            unsigned long long *__restrict__ sumG,
                            unsigned long long *__restrict__ sumB,
                            unsigned long long *__restrict__ cnt,
                            unsigned long long *__restrict__ changed,
                            int64_t n, int K)
{
    float *tcr = (float *)s_raw;
    float *tcg = tcr + K;
    float *tcb = tcg + K;
    unsigned int *sR = (unsigned int *)(tcb + K);
    unsigned int *sG = sR + K;
    unsigned int *sB = sG + K;
    unsigned int *sC = sB + K;
    unsigned int *sChanged = sC + K;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        tcr[k] = cr[k]; tcg[k] = cg[k]; tcb[k] = cb[k];
        sR[k] = 0u; sG[k] = 0u; sB[k] = 0u; sC[k] = 0u;
    }
    if (threadIdx.x == 0) *sChanged = 0u;
    __syncthreads();

    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint8_t pr = R[i], pg = G[i], pb = B[i];
        int c = km_assign_pixel((float)pr, (float)pg, (float)pb, tcr, tcg, tcb, K);

        if (labels[i] != (uint16_t)c) {
            labels[i] = (uint16_t)c;
            atomicAdd(sChanged, 1u);              /* privatized */
        }
        /* SHARED atomics on-chip, no VRAM round trip. */
        atomicAdd(&sR[c], (unsigned int)pr);
        atomicAdd(&sG[c], (unsigned int)pg);
        atomicAdd(&sB[c], (unsigned int)pb);
        atomicAdd(&sC[c], 1u);
    }
    __syncthreads();

    /* One global atomic per cluster per block, instead of per pixel. */
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        if (sR[k]) atomicAdd(&sumR[k], (unsigned long long)sR[k]);
        if (sG[k]) atomicAdd(&sumG[k], (unsigned long long)sG[k]);
        if (sB[k]) atomicAdd(&sumB[k], (unsigned long long)sB[k]);
        if (sC[k]) atomicAdd(&cnt[k],  (unsigned long long)sC[k]);
    }
    if (threadIdx.x == 0 && *sChanged)
        atomicAdd(changed, (unsigned long long)*sChanged);
}

/* Centroid update shared by all version */
__global__ void k_update(float *cr, float *cg, float *cb,
                         const unsigned long long *sumR,
                         const unsigned long long *sumG,
                         const unsigned long long *sumB,
                         const unsigned long long *cnt, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    unsigned long long c = cnt[k];
    if (c == 0ULL) return;                  /* empty cluster keeps its centroid */
    cr[k] = km_centroid(sumR[k], c);        /* same shared function as seq */
    cg[k] = km_centroid(sumG[k], c);
    cb[k] = km_centroid(sumB[k], c);
}

/* ------------------------------------------------------------------ */

static double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/* groups all GPU memory pointers into one structure.*/
struct DevBufs {
    uint8_t *R = nullptr, *G = nullptr, *B = nullptr;
    uint16_t *labels = nullptr;
    float *cr = nullptr, *cg = nullptr, *cb = nullptr;
    unsigned long long *sumR = nullptr, *sumG = nullptr, *sumB = nullptr;
    unsigned long long *cnt = nullptr, *changed = nullptr;
};

/* Runs Lloyd to convergence on the device. Returns iterations performed. */
static int lloyd_gpu(DevBufs &d, int64_t n, int K, int max_iter, int version,
                     size_t shmem, int64_t *last_changed)
{
    const int blocks_1t = (int)((n + KM_BLOCK - 1) / KM_BLOCK);
    CUDA_CHECK(cudaMemset(d.labels, 0xFF, (size_t)n * sizeof(uint16_t)));

    int iters = 0;
    unsigned long long changed_host = 0;

    for (int it = 0; it < max_iter; ++it) {
        CUDA_CHECK(cudaMemset(d.sumR, 0, (size_t)K * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d.sumG, 0, (size_t)K * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d.sumB, 0, (size_t)K * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d.cnt,  0, (size_t)K * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d.changed, 0, sizeof(unsigned long long)));

        if (version == 1) {
            k_assign_v1<<<blocks_1t, KM_BLOCK>>>(
                d.R, d.G, d.B, d.labels,
                d.sumR, d.sumG, d.sumB, d.cnt, d.changed, n, K);
        } else if (version == 2) {
            k_assign_v2<<<blocks_1t, KM_BLOCK, shmem>>>(
                d.R, d.G, d.B, d.labels, d.cr, d.cg, d.cb,
                d.sumR, d.sumG, d.sumB, d.cnt, d.changed, n, K);
        } else if (version == 3) {
            k_assign_v3<<<blocks_1t, KM_BLOCK, shmem>>>(
                d.R, d.G, d.B, d.labels, d.cr, d.cg, d.cb,
                d.sumR, d.sumG, d.sumB, d.cnt, d.changed, n, K);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&changed_host, d.changed,
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        iters = it + 1;
        if (changed_host == 0ULL) break;    /* same rule as seq -> same count */

        k_update<<<(K + 255) / 256, 256>>>(d.cr, d.cg, d.cb,
                                           d.sumR, d.sumG, d.sumB, d.cnt, K);
        CUDA_CHECK(cudaGetLastError());
        if (version == 1) {
            CUDA_CHECK(cudaMemcpyToSymbol(c_cr, d.cr, (size_t)K * sizeof(float),
                                          0, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyToSymbol(c_cg, d.cg, (size_t)K * sizeof(float),
                                          0, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyToSymbol(c_cb, d.cb, (size_t)K * sizeof(float),
                                          0, cudaMemcpyDeviceToDevice));
        }
    }
    *last_changed = (int64_t)changed_host;
    return iters;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
            "usage: %s <image> <K> <seed> <max_iter> <out_prefix> "
            "[--version N] [--reps N] [--no-png]\n", argv[0]);
        return 2;
    }
    const char *img_path = argv[1];
    int         K        = atoi(argv[2]);
    uint64_t    seed     = strtoull(argv[3], NULL, 10);
    int         max_iter = atoi(argv[4]);
    const char *prefix   = argv[5];

    int version = 1, reps = 1, write_png = 1;
    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--version") && i + 1 < argc) version = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-png")) write_png = 0;
        else { fprintf(stderr, "unknown option '%s'\n", argv[i]); return 2; }
    }
    if (version < 1 || version > 3) {
        fprintf(stderr, "--version must be 1, 2, or 3\n");
        return 2;
    }
    if (K < 1 || K > KM_MAX_CONST_K) {
        fprintf(stderr, "K must be in [1,%d]\n", KM_MAX_CONST_K); return 2;
    }
    if (max_iter < 1) { fprintf(stderr, "max_iter must be >= 1\n"); return 2; }
    if (reps < 1) reps = 1;

    char tag[16];
    snprintf(tag, sizeof(tag), "v%d", version);

    /*  load Image  */
    ImageSoA img;
    Timer t;
    if (load_image_soa(img_path, &img) != 0) return 1;
    double load_ms = t.ms();
    if ((int64_t)K > img.n) {
        fprintf(stderr, "K (%d) exceeds pixel count (%lld)\n", K, (long long)img.n);
        free_image_soa(&img); return 2;
    }

    /* seeding*/
    std::vector<float> seed_r(K), seed_g(K), seed_b(K);
    t.reset();
    if (kmeanspp_seed(&img, K, seed, seed_r.data(), seed_g.data(), seed_b.data()) != 0) {
        free_image_soa(&img); return 1;
    }
    double seed_ms = t.ms();

    /* device allocation */
    const int64_t n = img.n;
    DevBufs d;
    CUDA_CHECK(cudaMalloc(&d.R, (size_t)n)); /*Each RGB component needs one byte per pixel*/
    CUDA_CHECK(cudaMalloc(&d.G, (size_t)n));
    CUDA_CHECK(cudaMalloc(&d.B, (size_t)n));
    CUDA_CHECK(cudaMalloc(&d.labels, (size_t)n * sizeof(uint16_t))); /*Each label uses two bytes*/
    CUDA_CHECK(cudaMalloc(&d.cr, (size_t)K * sizeof(float)));/*Each centroid component uses four bytes.*/
    CUDA_CHECK(cudaMalloc(&d.cg, (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.cb, (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.sumR, (size_t)K * sizeof(unsigned long long)));/*Each value uses eight bytes.*/
    CUDA_CHECK(cudaMalloc(&d.sumG, (size_t)K * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d.sumB, (size_t)K * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d.cnt,  (size_t)K * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d.changed, sizeof(unsigned long long)));

    size_t tile  = (size_t)3 * K * sizeof(float);
    size_t accum = ((size_t)4 * K + 1) * sizeof(unsigned int);
    size_t shmem = 0;
    if      (version == 2) shmem = tile;
    else if (version == 3) shmem = tile + accum;

    if (shmem > 0) {
        int dev = 0; cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        if (shmem > prop.sharedMemPerBlock) {
            fprintf(stderr,
                "K=%d needs %zu B of shared memory but the device offers %zu B "
                "per block; use --version 1 or a smaller K.\n",
                K, shmem, prop.sharedMemPerBlock);
            return 2;
        }
    }

    t.reset();
    CUDA_CHECK(cudaMemcpy(d.R, img.r, (size_t)n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.G, img.g, (size_t)n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.B, img.b, (size_t)n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double h2d_ms = t.ms();

    /* Lloyd*/
    std::vector<double> times;
    std::vector<uint16_t> labels((size_t)n);
    std::vector<float> cr(K), cg(K), cb(K);
    int iters = 0;
    int64_t last_changed = 0;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    for (int rep = 0; rep < reps + 1; ++rep) {
        CUDA_CHECK(cudaMemcpy(d.cr, seed_r.data(), (size_t)K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.cg, seed_g.data(), (size_t)K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.cb, seed_b.data(), (size_t)K * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_cr, seed_r.data(), (size_t)K * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_cg, seed_g.data(), (size_t)K * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_cb, seed_b.data(), (size_t)K * sizeof(float)));

        CUDA_CHECK(cudaEventRecord(ev0));
        iters = lloyd_gpu(d, n, K, max_iter, version, shmem, &last_changed);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (rep > 0) times.push_back((double)ms);
    }
    double lloyd_ms = median_of(times);

    t.reset();
    CUDA_CHECK(cudaMemcpy(labels.data(), d.labels, (size_t)n * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cr.data(), d.cr, (size_t)K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cg.data(), d.cg, (size_t)K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cb.data(), d.cb, (size_t)K * sizeof(float), cudaMemcpyDeviceToHost));
    double d2h_ms = t.ms();

    /*  output  */
    t.reset();
    if (dump_result(prefix, &img, labels.data(), K, iters,
                    cr.data(), cg.data(), cb.data()) != 0) {
        free_image_soa(&img); return 1;
    }
    if (write_png) {
        char png[1024];
        snprintf(png, sizeof(png), "%s_seg.png", prefix);
        if (write_segmented_png(png, &img, labels.data(), K,
                                cr.data(), cg.data(), cb.data()) != 0)
            fprintf(stderr, "warning: could not write %s\n", png);
    }
    double write_ms = t.ms();

    printf("[%s] %s  %dx%d  n=%lld  K=%d  seed=%llu\n", tag, img_path,
           img.w, img.h, (long long)img.n, K, (unsigned long long)seed);
    printf("[%s] iters=%d  last_changed=%lld  %s\n", tag, iters,
           (long long)last_changed,
           last_changed == 0 ? "(converged)" : "(hit max_iter)");
    printf("[%s] load=%.2fms  seed=%.2fms  h2d=%.2fms  lloyd=%.2fms (median of %d)"
           "  d2h=%.2fms  write=%.2fms\n",
           tag, load_ms, seed_ms, h2d_ms, lloyd_ms, reps, d2h_ms, write_ms);
    printf("CSV,cuda,%s,%s,%d,%d,%lld,%d,%llu,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
           tag, img_path, img.w, img.h, (long long)img.n, K,
           (unsigned long long)seed, iters,
           load_ms, seed_ms, lloyd_ms, write_ms, h2d_ms, d2h_ms);

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d.R); cudaFree(d.G); cudaFree(d.B); cudaFree(d.labels);
    cudaFree(d.cr); cudaFree(d.cg); cudaFree(d.cb);
    cudaFree(d.sumR); cudaFree(d.sumG); cudaFree(d.sumB);
    cudaFree(d.cnt); cudaFree(d.changed);
    free_image_soa(&img);
    return 0;
}
