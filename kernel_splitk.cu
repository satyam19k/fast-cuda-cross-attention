/**
 * E9 -- split-K (key-parallel) cross-attention. Beats the vectorized kernel in
 * the small-Nl Perceiver regime by fixing OCCUPANCY, not the inner loop.
 *
 * The other kernels launch ceil(Nl/4) blocks (one warp per latent), so at
 * Nl=64 only 16 of ~142 SMs are active. Split-K adds the key dimension to the
 * grid: each (latent, key-split) pair is its own block, computing a PARTIAL
 * online-softmax over its slice of the Ni keys. A second tiny "combine" kernel
 * merges the partials per latent (FlashAttention/flash-decoding log-sum-exp
 * merge). With ~35 splits, Nl=64 launches ~560 blocks -> fills the GPU.
 *
 * Two kernels:
 *   splitk_partial : grid (ceil(Nl/4), num_splits, batch). Reuses the float4
 *                    online-softmax inner loop of kernel_vectorized.cu, but
 *                    over keys [k0,k1) only, and writes UNNORMALIZED partials:
 *                    pm = local max, pl = local sum exp(s-max),
 *                    po = local sum exp(s-max)*V.
 *   splitk_combine : grid (ceil(Nl/4), batch). Per latent: M=max_j pm_j;
 *                    L=sum_j exp(pm_j-M) pl_j; O=sum_j exp(pm_j-M) po_j; /L.
 *
 * fp32. D=768 fixed ("magic 6"). Scratch (po/pm/pl) is allocated once by the
 * Python wrapper and reused across timed iterations.
 *
 * STATUS: written without a local nvcc -- compile + verify on the GPU box. The
 * partial kernel is a close adaptation of the verified vectorized kernel; the
 * combine and the scratch indexing are the likely debug sites. Verify against
 * the fp32 reference at a small size first.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define TILE_K 16
#define D 768
#define D_VEC (D / 4)              // 192 float4 per row
#define TILE_K_VEC (TILE_K * D_VEC)

// ---------------------------------------------------------------------------
// Partial: online softmax over one key-split [k0,k1); writes pm,pl,po.
// ---------------------------------------------------------------------------
__global__ void splitk_partial(
    const float* Q, const float* K, const float* V,
    float* po, float* pm, float* pl,
    int batch_size, int N_latent, int N_input, int dim, int num_splits
) {
    __shared__ float4 shared_K_tile[TILE_K_VEC];

    int batch_idx = blockIdx.z;
    int split = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int latent_idx = blockIdx.x * 4 + warp_id;
    if (batch_idx >= batch_size || latent_idx >= N_latent || threadIdx.x >= 128) return;

    int dim_vec = dim / 4;
    long pidx = ((long)batch_idx * N_latent + latent_idx) * num_splits + split;

    int chunk = (N_input + num_splits - 1) / num_splits;
    int k0 = split * chunk;
    int k1 = min(k0 + chunk, N_input);
    if (k0 >= N_input) {                          // empty split
        if (lane_id == 0) { pm[pidx] = -FLT_MAX; pl[pidx] = 0.0f; }
        float4* po_vec = reinterpret_cast<float4*>(po);
        long po_base = pidx * (long)D_VEC;
        for (int i = 0; i < 6; i++) {
            int v = lane_id + i * 32;
            if (v < dim_vec) po_vec[po_base + v] = make_float4(0, 0, 0, 0);
        }
        return;
    }

    int Q_off = batch_idx * N_latent * dim;
    int K_off = batch_idx * N_input * dim;
    int V_off = batch_idx * N_input * dim;

    const float4* Q_vec = reinterpret_cast<const float4*>(Q);
    int q_base = (Q_off / 4) + latent_idx * dim_vec;
    float4 my_q_vec[6];
    for (int i = 0; i < 6; i++) {
        int v = lane_id + i * 32;
        my_q_vec[i] = (v < dim_vec) ? Q_vec[q_base + v] : make_float4(0, 0, 0, 0);
    }

    float running_max = -FLT_MAX, running_sum = 0.0f;
    float tile_scores[TILE_K];
    float4 output_accum[6];
    for (int i = 0; i < 6; i++) output_accum[i] = make_float4(0, 0, 0, 0);

    const float4* K_vec = reinterpret_cast<const float4*>(K);
    const float4* V_vec = reinterpret_cast<const float4*>(V);
    int k_vec_off = K_off / 4, v_vec_off = V_off / 4;

    for (int tile_start = k0; tile_start < k1; tile_start += TILE_K) {
        // stage K tile (only keys < k1 matter)
        int total = blockDim.x, per = (TILE_K_VEC + total - 1) / total;
        for (int l = 0; l < per; l++) {
            int s = threadIdx.x + l * total;
            if (s < TILE_K_VEC) {
                int kk = s / D_VEC, dd = s % D_VEC, gk = tile_start + kk;
                shared_K_tile[s] = (gk < k1 && dd < dim_vec)
                    ? K_vec[k_vec_off + gk * dim_vec + dd] : make_float4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        unsigned mask = __activemask();
        for (int k = 0; k < TILE_K; k++) {
            int gk = tile_start + k;
            if (gk < k1) {
                float4 dv = make_float4(0, 0, 0, 0);
                for (int i = 0; i < 6; i++) {
                    int v = lane_id + i * 32;
                    if (v < dim_vec) {
                        float4 kv = shared_K_tile[k * D_VEC + v];
                        dv.x += my_q_vec[i].x * kv.x; dv.y += my_q_vec[i].y * kv.y;
                        dv.z += my_q_vec[i].z * kv.z; dv.w += my_q_vec[i].w * kv.w;
                    }
                }
                float dot = dv.x + dv.y + dv.z + dv.w;
                for (int o = 16; o > 0; o /= 2) dot += __shfl_down_sync(mask, dot, o);
                if (lane_id == 0) {
                    float sc = dot / sqrtf((float)dim);
                    tile_scores[k] = sc; tile_max = fmaxf(tile_max, sc);
                }
            }
        }
        tile_max = __shfl_sync(mask, tile_max, 0);

        float scale = 1.0f;
        if (lane_id == 0) {
            float nm = fmaxf(running_max, tile_max);
            scale = expf(running_max - nm); running_sum *= scale; running_max = nm;
        }
        scale = __shfl_sync(mask, scale, 0);
        running_max = __shfl_sync(mask, running_max, 0);
        running_sum = __shfl_sync(mask, running_sum, 0);
        for (int i = 0; i < 6; i++) {
            output_accum[i].x *= scale; output_accum[i].y *= scale;
            output_accum[i].z *= scale; output_accum[i].w *= scale;
        }

        float exp_scores[TILE_K]; float tse = 0.0f;
        if (lane_id == 0) {
            for (int k = 0; k < TILE_K; k++) {
                int gk = tile_start + k;
                if (gk < k1) { exp_scores[k] = expf(tile_scores[k] - running_max); tse += exp_scores[k]; }
                else exp_scores[k] = 0.0f;
            }
            running_sum += tse;
        }
        running_sum = __shfl_sync(mask, running_sum, 0);

        for (int k = 0; k < TILE_K; k++) {
            int gk = tile_start + k;
            if (gk < k1) {
                float es = (lane_id == 0) ? exp_scores[k] : 0.0f;
                es = __shfl_sync(mask, es, 0);
                for (int i = 0; i < 6; i++) {
                    int v = lane_id + i * 32;
                    if (v < dim_vec) {
                        float4 vv = V_vec[v_vec_off + gk * dim_vec + v];
                        output_accum[i].x += es * vv.x; output_accum[i].y += es * vv.y;
                        output_accum[i].z += es * vv.z; output_accum[i].w += es * vv.w;
                    }
                }
            }
        }
        __syncthreads();
    }

    // write UNNORMALIZED partials
    if (lane_id == 0) { pm[pidx] = running_max; pl[pidx] = running_sum; }
    float4* po_vec = reinterpret_cast<float4*>(po);
    long po_base = pidx * (long)D_VEC;
    for (int i = 0; i < 6; i++) {
        int v = lane_id + i * 32;
        if (v < dim_vec) po_vec[po_base + v] = output_accum[i];
    }
}

// ---------------------------------------------------------------------------
// Combine: log-sum-exp merge of the per-split partials -> normalized output.
// ---------------------------------------------------------------------------
__global__ void splitk_combine(
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
        long pjb = (base + j) * (long)D_VEC;
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

extern "C" void launch_splitk_kernel(
    const float* Q, const float* K, const float* V, float* output,
    float* po, float* pm, float* pl,
    int batch_size, int N_latent, int N_input, int dim, int num_splits,
    cudaStream_t stream
) {
    dim3 block(128, 1);
    dim3 g1((N_latent + 3) / 4, num_splits, batch_size);
    splitk_partial<<<g1, block, 0, stream>>>(
        Q, K, V, po, pm, pl, batch_size, N_latent, N_input, dim, num_splits);
    dim3 g2((N_latent + 3) / 4, batch_size);
    splitk_combine<<<g2, block, 0, stream>>>(
        po, pm, pl, output, batch_size, N_latent, dim, num_splits);
}
