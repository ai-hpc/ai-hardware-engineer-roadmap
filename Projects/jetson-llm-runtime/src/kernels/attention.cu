// Single-query tiled online attention. Compiled for SM 8.7; no throughput claim.
// One block per query head, token-major K/V, FP32 dot/softmax accumulation.
// Optional INT8 kernel input requires explicit fixed per-K/V-head scales; the
// Engine's appendable cache remains FP16-only (no calibrated INT8 storage).

#include "jllm_kernels.h"
#include "jllm_attention_layout.h"
#include <cuda_fp16.h>
#include <cfloat>

namespace jllm {

static constexpr int ATTN_TILE_KV = 64;
static constexpr int ATTN_BLOCK   = 128;

// Each thread handles ceil(head_dim / blockDim.x) output dimensions.
// Accumulators stored in shared memory (visible to all threads in block).

__global__ void flash_attention_decode_kernel(
    half*        __restrict__ output,
    const half*  __restrict__ q,
    const void*  __restrict__ k_cache,
    const void*  __restrict__ v_cache,
    int n_heads, int n_kv_heads, int head_dim, int seq_len,
    float scale, bool kv_int8, const float* kv_scales)
{
    const int head = blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);  // GQA
    const int tid = threadIdx.x;
    __shared__ float warp_values[ATTN_BLOCK / 32];

    // Shared memory layout:
    //   s_scores[ATTN_TILE_KV]     — attention scores for current KV tile
    //   s_out[head_dim]            — output accumulator (BUG #6 FIX)
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_out    = smem + ATTN_TILE_KV;

    // Initialize output accumulator to zero
    for (int d = tid; d < head_dim; d += blockDim.x)
        s_out[d] = 0.0f;
    __syncthreads();

    // Q is read directly from global memory in the dot product loop below.
    // (No register pre-load needed — Q is small and stays in L1 cache.)

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    // Tile over KV sequence
    for (int kv_start = 0; kv_start < seq_len; kv_start += ATTN_TILE_KV) {
        int tile_len = min(ATTN_TILE_KV, seq_len - kv_start);

        // ── Step 1: Q × K^T for tile ────────────────────────────
        for (int t = tid; t < tile_len; t += blockDim.x) {
            int kv_pos = kv_start + t;
            float dot = 0.0f;

            for (int d = 0; d < head_dim; d++) {
                float q_val = __half2float(q[head * head_dim + d]);
                float k_val;
                if (kv_int8) {
                    const int8_t* ki = (const int8_t*)k_cache;
                    float ks = kv_scales[kv_head];
                    k_val = ki[kv_element_offset(kv_pos, kv_head, d, n_kv_heads, head_dim)] * ks;
                } else {
                    const half* kf = (const half*)k_cache;
                    k_val = __half2float(kf[kv_element_offset(kv_pos, kv_head, d, n_kv_heads, head_dim)]);
                }
                dot += q_val * k_val;
            }
            s_scores[t] = dot * scale;
        }
        __syncthreads();

        // ── Step 2: Online softmax ──────────────────────────────
        float tile_max = -FLT_MAX;
        for (int t = tid; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, s_scores[t]);

        // Reduce max across block
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, off));

        if (tid % 32 == 0) warp_values[tid / 32] = tile_max;
        __syncthreads();
        // IEEE-754 negative floats do not have signed-integer max ordering.
        // All threads read the same four warp maxima using floating comparison.
        tile_max = -FLT_MAX;
        for (int warp = 0; warp < ATTN_BLOCK / 32; ++warp)
            tile_max = fmaxf(tile_max, warp_values[warp]);
        __syncthreads(); // no warp may overwrite the reduction workspace early

        // Correction factor for running max update
        float old_max = running_max;
        running_max = fmaxf(running_max, tile_max);
        float correction = expf(old_max - running_max);

        // Correct existing accumulators
        running_sum *= correction;
        for (int d = tid; d < head_dim; d += blockDim.x)
            s_out[d] *= correction;
        __syncthreads();

        // Exponentiate scores
        float tile_sum = 0.0f;
        for (int t = tid; t < tile_len; t += blockDim.x) {
            float p = expf(s_scores[t] - running_max);
            s_scores[t] = p;
            tile_sum += p;
        }

        // Reduce sum
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, off);

        if (tid % 32 == 0) warp_values[tid / 32] = tile_sum;
        __syncthreads();
        float block_sum = 0.0f;
        for (int warp = 0; warp < ATTN_BLOCK / 32; ++warp)
            block_sum += warp_values[warp];
        __syncthreads();
        running_sum += block_sum;

        // ── Step 3: Accumulate P × V (BUG #6 FIX — per-dimension) ──
        // Each thread handles its slice of head_dim
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                int kv_pos = kv_start + t;
                float v_val;
                if (kv_int8) {
                    const int8_t* vi = (const int8_t*)v_cache;
                    float vs = kv_scales[n_kv_heads + kv_head];
                    v_val = vi[kv_element_offset(kv_pos, kv_head, d, n_kv_heads, head_dim)] * vs;
                } else {
                    const half* vf = (const half*)v_cache;
                    v_val = __half2float(vf[kv_element_offset(kv_pos, kv_head, d, n_kv_heads, head_dim)]);
                }
                val += s_scores[t] * v_val;
            }
            s_out[d] += val;  // BUG #6 FIX: accumulate into correct dimension
        }
        __syncthreads();
    }

    // ── Finalize: normalize and write output ─────────────────────
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[head * head_dim + d] = __float2half(s_out[d] * inv_sum);
    }
}

cudaError_t flash_attention_decode(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int seq_len,
    float scale, bool kv_int8, const float* kv_scales, cudaStream_t stream)
{
    if (!output || !q || !k_cache || !v_cache
        || !attention_shape_valid(n_heads, n_kv_heads, head_dim, seq_len, scale)
        || (kv_int8 && !kv_scales))
        return cudaErrorInvalidValue;
    // Shared memory: scores[TILE_KV] + output[head_dim]
    int smem = (ATTN_TILE_KV + head_dim) * sizeof(float);

    flash_attention_decode_kernel<<<n_heads, ATTN_BLOCK, smem, stream>>>(
        output, q, k_cache, v_cache,
        n_heads, n_kv_heads, head_dim, seq_len,
        scale, kv_int8, kv_scales);
    return cudaGetLastError(); // launch status only; execution is asynchronous
}

}  // namespace jllm
