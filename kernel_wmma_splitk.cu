/**
 * #1 -- Tensor-core split-K (E7 + E9): FlashDecoding with WMMA.
 *
 * The WMMA kernel (E7) was starved -- 16 latents/block => only Nl/16 blocks.
 * Split-K (E9) fixed occupancy for the float4 kernel by adding the key axis to
 * the grid. This combines them: a WMMA partial kernel on a key-split grid, so
 * the tensor cores finally get enough blocks to stay fed in the Nl << Ni regime
 * (which is exactly the LLM-decode shape FlashDecoding targets).
 *
 *   wmma_splitk_partial : grid (ceil(Nl/16), num_splits, batch). One warp/block,
 *       16 latents, processes keys [k0,k1) (chunk rounded to a multiple of 16 so
 *       every WMMA tile is full -> no masking, no OOB; requires Ni % 16 == 0).
 *       Q@K^T and P@V on tensor cores, online softmax in shared, writes
 *       UNNORMALIZED partials pm/pl/po (same layout as kernel_splitk.cu).
 *   combine : log-sum-exp merge of the partials (same math as kernel_splitk.cu,
 *             defined locally to avoid a cross-translation-unit kernel launch).
 *
 * fp16 in, fp32 accumulate, fp32 out. Q pre-scaled by 1/sqrt(D) on the host.
 * D = 768 fixed (48 K-tiles). Scratch po/pm/pl allocated once by Python.
 *
 * STATUS: not compiled locally (Mac). Verify on GPU at a small size first.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>
#include <math.h>

using namespace nvcuda;

#define D 768
#define WM 16
#define WN 16
#define WK 16
#define DKT (D / WK)   // 48 K-tiles along the embedding dim
#define DNT (D / WN)   // 48 D-tiles when forming O = P @ V

__global__ void wmma_splitk_partial(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl,
    int batch_size, int N_latent, int N_input, int dim, int num_splits
) {
    int batch = blockIdx.z;
    int split = blockIdx.y;
    int m0 = blockIdx.x * WM;
    if (batch >= batch_size || m0 >= N_latent) return;
    int lane = threadIdx.x;

    // key range for this split, rounded to a multiple of 16 so tiles are full
    int chunk = ((N_input + num_splits - 1) / num_splits + WN - 1) / WN * WN;
    int k0 = split * chunk;
    int k1 = min(k0 + chunk, N_input);

    extern __shared__ char smem[];
    half*  Qs  = reinterpret_cast<half*>(smem);          // [16][768]
    float* Os  = reinterpret_cast<float*>(Qs + WM * D);  // [16][768]
    float* Ss  = Os + WM * D;                            // [16][16]
    half*  Ps  = reinterpret_cast<half*>(Ss + WM * WN);  // [16][16]
    float* PVs = reinterpret_cast<float*>(Ps + WM * WN); // [16][16]
    __shared__ float row_max[WM];
    __shared__ float row_sum[WM];

    if (k0 >= N_input) {                              // empty split
        if (lane < WM && (m0 + lane) < N_latent) {
            long p = ((long)batch * N_latent + m0 + lane) * num_splits + split;
            pm[p] = -FLT_MAX; pl[p] = 0.0f;
        }
        float4* po4 = reinterpret_cast<float4*>(po);
        for (int r = 0; r < WM && (m0 + r) < N_latent; r++) {
            long pb = (((long)batch * N_latent + m0 + r) * num_splits + split) * (D / 4);
            for (int i = lane; i < D / 4; i += 32) po4[pb + i] = make_float4(0, 0, 0, 0);
        }
        return;
    }

    const half* Qb = Q + (size_t)batch * N_latent * dim + (size_t)m0 * dim;
    const half* Kb = K + (size_t)batch * N_input * dim;
    const half* Vb = V + (size_t)batch * N_input * dim;

    for (int i = lane; i < WM * D; i += 32) Qs[i] = Qb[i];   // Q already /sqrt(D)
    for (int i = lane; i < WM * D; i += 32) Os[i] = 0.0f;
    if (lane < WM) { row_max[lane] = -FLT_MAX; row_sum[lane] = 0.0f; }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> bk_frag; // K^T
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> bv_frag; // V
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;

    for (int n0 = k0; n0 < k1; n0 += WN) {
        wmma::fill_fragment(acc, 0.0f);
        for (int dk = 0; dk < DKT; dk++) {
            wmma::load_matrix_sync(a_frag, Qs + dk * WK, D);
            wmma::load_matrix_sync(bk_frag, Kb + (size_t)n0 * D + dk * WK, D);
            wmma::mma_sync(acc, a_frag, bk_frag, acc);
        }
        wmma::store_matrix_sync(Ss, acc, WN, wmma::mem_row_major);
        __syncwarp();

        if (lane < WM) {
            int m = lane;
            float tmax = -FLT_MAX;
            for (int j = 0; j < WN; j++) tmax = fmaxf(tmax, Ss[m * WN + j]);
            float new_max = fmaxf(row_max[m], tmax);
            float corr = __expf(row_max[m] - new_max);
            float tsum = 0.0f;
            for (int j = 0; j < WN; j++) {
                float p = __expf(Ss[m * WN + j] - new_max);
                Ps[m * WN + j] = __float2half(p);
                tsum += p;
            }
            row_sum[m] = row_sum[m] * corr + tsum;
            row_max[m] = new_max;
            for (int d = 0; d < D; d++) Os[m * D + d] *= corr;
        }
        __syncwarp();

        for (int dt = 0; dt < DNT; dt++) {
            wmma::fill_fragment(acc, 0.0f);
            wmma::load_matrix_sync(a_frag, Ps, WN);
            wmma::load_matrix_sync(bv_frag, Vb + (size_t)n0 * D + dt * WN, D);
            wmma::mma_sync(acc, a_frag, bv_frag, acc);
            wmma::store_matrix_sync(PVs, acc, WN, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < WM * WN; i += 32) {
                int m = i / WN, nn = i % WN;
                Os[m * D + dt * WN + nn] += PVs[i];
            }
            __syncwarp();
        }
    }

    // write UNNORMALIZED partials (one row per latent)
    if (lane < WM && (m0 + lane) < N_latent) {
        long p = ((long)batch * N_latent + m0 + lane) * num_splits + split;
        pm[p] = row_max[lane]; pl[p] = row_sum[lane];
    }
    float4* po4 = reinterpret_cast<float4*>(po);
    float4* Os4 = reinterpret_cast<float4*>(Os);
    for (int r = 0; r < WM && (m0 + r) < N_latent; r++) {
        long pb = (((long)batch * N_latent + m0 + r) * num_splits + split) * (D / 4);
        for (int i = lane; i < D / 4; i += 32) po4[pb + i] = Os4[r * (D / 4) + i];
    }
}

// Log-sum-exp merge of per-split partials -> normalized output (float4).
__global__ void wmma_splitk_combine(
    const float* po, const float* pm, const float* pl,
    float* output, int batch_size, int N_latent, int dim, int num_splits
) {
    int batch_idx = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int latent_idx = blockIdx.x * 4 + warp_id;
    if (batch_idx >= batch_size || latent_idx >= N_latent || threadIdx.x >= 128) return;

    int dim_vec = dim / 4;
    long base = ((long)batch_idx * N_latent + latent_idx) * num_splits;

    float M = -FLT_MAX;
    for (int j = 0; j < num_splits; j++) M = fmaxf(M, pm[base + j]);
    float L = 0.0f;
    for (int j = 0; j < num_splits; j++) {
        float m = pm[base + j];
        if (m > -FLT_MAX) L += expf(m - M) * pl[base + j];
    }

    const float4* po_vec = reinterpret_cast<const float4*>(po);
    float4 acc[6];
    for (int i = 0; i < 6; i++) acc[i] = make_float4(0, 0, 0, 0);
    for (int j = 0; j < num_splits; j++) {
        float m = pm[base + j];
        if (m == -FLT_MAX) continue;
        float w = expf(m - M);
        long pjb = (base + j) * (long)dim_vec;
        for (int i = 0; i < 6; i++) {
            int v = lane_id + i * 32;
            if (v < dim_vec) {
                float4 o = po_vec[pjb + v];
                acc[i].x += w * o.x; acc[i].y += w * o.y;
                acc[i].z += w * o.z; acc[i].w += w * o.w;
            }
        }
    }

    float norm = 1.0f / (L + 1e-6f);
    float4* out_vec = reinterpret_cast<float4*>(output);
    long ob = ((long)batch_idx * N_latent + latent_idx) * dim_vec;
    for (int i = 0; i < 6; i++) {
        int v = lane_id + i * 32;
        if (v < dim_vec) {
            float4 o = acc[i];
            o.x *= norm; o.y *= norm; o.z *= norm; o.w *= norm;
            out_vec[ob + v] = o;
        }
    }
}

extern "C" void launch_wmma_splitk_kernel(
    const half* Q, const half* K, const half* V, float* output,
    float* po, float* pm, float* pl,
    int batch_size, int N_latent, int N_input, int dim, int num_splits,
    cudaStream_t stream
) {
    size_t smem = (size_t)WM * D * sizeof(half) + (size_t)WM * D * sizeof(float)
                + (size_t)WM * WN * sizeof(float) + (size_t)WM * WN * sizeof(half)
                + (size_t)WM * WN * sizeof(float);
    cudaFuncSetAttribute(wmma_splitk_partial,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 g1((N_latent + WM - 1) / WM, num_splits, batch_size);
    wmma_splitk_partial<<<g1, dim3(32), smem, stream>>>(
        Q, K, V, po, pm, pl, batch_size, N_latent, N_input, dim, num_splits);

    dim3 g2((N_latent + 3) / 4, batch_size);
    wmma_splitk_combine<<<g2, dim3(128), 0, stream>>>(
        po, pm, pl, output, batch_size, N_latent, dim, num_splits);
}
