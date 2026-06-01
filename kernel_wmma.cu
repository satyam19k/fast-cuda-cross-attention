/**
 * E7 -- fp16 WMMA fused cross-attention (FlashAttention-style, tensor cores).
 *
 * softmax(Q @ K^T / sqrt(D)) @ V with fp16 inputs, fp32 accumulate, fp32 out.
 * Both matmuls run on tensor cores via the WMMA API (16x16x16 fragments). The
 * scores S are never written to HBM -- online softmax keeps an O accumulator
 * and running max/sum in shared memory (the "fused" half of the three-point
 * decomposition: scalar -> cuBLAS-fp16 -> WMMA-fused).
 *
 * Layout: one warp (32 threads) per block; each block owns 16 latents
 *   (M-tile). grid = (Nl/16, batch). Requires Nl % 16 == 0 and Ni % 16 == 0
 *   (true for the square image resolutions 28^2,56^2,...,320^2 and Nl powers
 *   of two). D = 768 is fixed (D/16 = 48 K-tiles).
 *
 * Shared memory per block (~73 KB; fits L40S ~100 KB / H200 ~228 KB opt-in):
 *   Qs  16 x 768 half  = 24 KB   (Q cached, reused over all key tiles)
 *   Os  16 x 768 float = 48 KB   (online output accumulator)
 *   Ss / Ps / PVs      ~ 1.5 KB  (per-tile scratch)
 * K and V fragments are loaded straight from global memory (no K/V staging).
 *
 * STATUS: written without a local nvcc (host is a Mac). Compile + verify on the
 * GPU box; the likely debug sites are the load_matrix_sync ldm/col_major for
 * Q@K^T, the warp sync points, and the O accumulation. Build with arch >= sm_70
 * (the project uses sm_89 on the L40S).
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
#define DNT (D / WN)   // 48 N-tiles when forming O = P @ V

__global__ void wmma_cross_attention(
    const half* __restrict__ Q,   // [batch, Nl, D] fp16, pre-scaled by 1/sqrt(D)
    const half* __restrict__ K,   // [batch, Ni, D] fp16
    const half* __restrict__ V,   // [batch, Ni, D] fp16
    float* __restrict__ output,   // [batch, Nl, D] fp32
    int batch_size,
    int N_latent,
    int N_input,
    int dim
) {
    int batch = blockIdx.y;
    int m0 = blockIdx.x * WM;                 // first latent of this block
    if (batch >= batch_size || m0 >= N_latent) return;
    int lane = threadIdx.x;                   // 0..31, one warp

    extern __shared__ char smem[];
    half*  Qs  = reinterpret_cast<half*>(smem);          // [16][768]
    float* Os  = reinterpret_cast<float*>(Qs + WM * D);  // [16][768]
    float* Ss  = Os + WM * D;                            // [16][16]
    half*  Ps  = reinterpret_cast<half*>(Ss + WM * WN);  // [16][16]
    float* PVs = reinterpret_cast<float*>(Ps + WM * WN); // [16][16]
    __shared__ float row_max[WM];
    __shared__ float row_sum[WM];

    const half* Qb = Q + (size_t)batch * N_latent * dim + (size_t)m0 * dim;
    const half* Kb = K + (size_t)batch * N_input * dim;
    const half* Vb = V + (size_t)batch * N_input * dim;

    // Stage Q [16 x 768] into shared (already 1/sqrt(D)-scaled on the host).
    for (int i = lane; i < WM * D; i += 32) Qs[i] = Qb[i];
    // Init online-softmax state and the O accumulator.
    for (int i = lane; i < WM * D; i += 32) Os[i] = 0.0f;
    if (lane < WM) { row_max[lane] = -FLT_MAX; row_sum[lane] = 0.0f; }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> bk_frag; // K^T
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> bv_frag; // V
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;

    int num_key_tiles = N_input / WN;         // Ni % 16 == 0 required
    for (int kt = 0; kt < num_key_tiles; kt++) {
        int n0 = kt * WN;                     // first key of this tile

        // ---- S[16x16] = Q[16x768] @ K_tile[16x768]^T  on tensor cores ----
        wmma::fill_fragment(acc, 0.0f);
        for (int dk = 0; dk < DKT; dk++) {
            wmma::load_matrix_sync(a_frag, Qs + dk * WK, D);                 // [16 x 16]
            wmma::load_matrix_sync(bk_frag, Kb + (size_t)n0 * D + dk * WK, D); // K^T tile
            wmma::mma_sync(acc, a_frag, bk_frag, acc);
        }
        wmma::store_matrix_sync(Ss, acc, WN, wmma::mem_row_major);
        __syncwarp();

        // ---- online softmax update (one row per lane, 16 rows) ----
        if (lane < WM) {
            int m = lane;
            float tmax = -FLT_MAX;
            for (int j = 0; j < WN; j++) tmax = fmaxf(tmax, Ss[m * WN + j]);
            float new_max = fmaxf(row_max[m], tmax);
            float corr = __expf(row_max[m] - new_max);   // rescale old contributions
            float tsum = 0.0f;
            for (int j = 0; j < WN; j++) {
                float p = __expf(Ss[m * WN + j] - new_max);
                Ps[m * WN + j] = __float2half(p);
                tsum += p;
            }
            row_sum[m] = row_sum[m] * corr + tsum;
            row_max[m] = new_max;
            for (int d = 0; d < D; d++) Os[m * D + d] *= corr;  // rescale O row
        }
        __syncwarp();

        // ---- O[16x768] += P[16x16] @ V_tile[16x768]  on tensor cores ----
        for (int dt = 0; dt < DNT; dt++) {
            wmma::fill_fragment(acc, 0.0f);
            wmma::load_matrix_sync(a_frag, Ps, WN);                          // P [16 x 16]
            wmma::load_matrix_sync(bv_frag, Vb + (size_t)n0 * D + dt * WN, D); // V [16 x 16]
            wmma::mma_sync(acc, a_frag, bv_frag, acc);
            wmma::store_matrix_sync(PVs, acc, WN, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < WM * WN; i += 32) {       // add into O accumulator
                int m = i / WN, nn = i % WN;
                Os[m * D + dt * WN + nn] += PVs[i];
            }
            __syncwarp();
        }
    }

    // ---- normalize by the softmax denominator and write fp32 output ----
    float* outb = output + (size_t)batch * N_latent * dim + (size_t)m0 * dim;
    for (int i = lane; i < WM * D; i += 32) {
        int m = i / D;
        outb[i] = Os[i] / (row_sum[m] + 1e-6f);
    }
}

extern "C" void launch_wmma_kernel(
    const half* Q, const half* K, const half* V, float* output,
    int batch_size, int N_latent, int N_input, int dim, cudaStream_t stream
) {
    dim3 block(32, 1);
    dim3 grid((N_latent + WM - 1) / WM, batch_size);
    size_t smem = (size_t)WM * D * sizeof(half)      // Qs
                + (size_t)WM * D * sizeof(float)      // Os
                + (size_t)WM * WN * sizeof(float)     // Ss
                + (size_t)WM * WN * sizeof(half)      // Ps
                + (size_t)WM * WN * sizeof(float);    // PVs
    cudaFuncSetAttribute(wmma_cross_attention,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    wmma_cross_attention<<<grid, block, smem, stream>>>(
        Q, K, V, output, batch_size, N_latent, N_input, dim);
}
