// gemv_q4.cu — Q4_K dequant-fused GEMV for Orin SM 8.7
// Matches llama.cpp's exact Q4_K block layout and dequant formula.

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cstdio>

namespace jllm {

static constexpr int QK_K = 256;

// Use raw uint16 instead of half2 to guarantee no padding
struct __attribute__((packed)) block_q4_K {
    uint16_t d_raw;       // FP16 super-block scale
    uint16_t dmin_raw;    // FP16 super-block min
    uint8_t  scales[12];  // packed 6-bit sub-block scales and mins
    uint8_t  qs[QK_K/2]; // 4-bit quants (128 bytes)
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");

__device__ __forceinline__ float raw_fp16_to_float(uint16_t h) {
    half tmp;
    memcpy(&tmp, &h, 2);
    return __half2float(tmp);
}

__device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d, uint8_t& m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

__global__ void gemv_q4k_kernel(
    half*              __restrict__ y,
    const block_q4_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K)
{
    const int row  = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q4_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc = 0.0f;

    for (int b = lane; b < n_blocks; b += 32) {
        const block_q4_K& blk = row_blocks[b];

        float dall = raw_fp16_to_float(blk.d_raw);
        float dmin = raw_fp16_to_float(blk.dmin_raw);

        int k_base = b * QK_K;

        for (int il = 0; il < 4; il++) {
            int is = 2 * il;

            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            float d1 = dall * sc1;
            float dm1 = dmin * m1;
            float d2 = dall * sc2;
            float dm2 = dmin * m2;

            const uint8_t* q = blk.qs + 32 * il;

            for (int l = 0; l < 32; l++) {
                int k_lo = k_base + 64 * il + l;
                int k_hi = k_base + 64 * il + l + 32;

                float w_lo = d1 * (q[l] & 0xF) - dm1;
                float w_hi = d2 * (q[l] >> 4)  - dm2;

                acc += w_lo * __half2float(x[k_lo]);
                acc += w_hi * __half2float(x[k_hi]);
            }
        }
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        y[row] = __float2half(acc);
}

// Debug kernel: print first few output values
__global__ void debug_print_half(const half* data, int n, const char* label) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU %s] first 8: ", label);
        for (int i = 0; i < 8 && i < n; i++)
            printf("%.4f ", __half2float(data[i]));
        printf("\n");
    }
}

void gemv_q4(half* y, const void* W_q4, const half* scales, const half* x,
             int M, int K, int group_size, cudaStream_t stream) {
    if (!y || !W_q4 || !x || M <= 0 || K <= 0 || K % QK_K) {
        fprintf(stderr, "[GEMV] invalid shape/pointer; K must be divisible by 256\n");
        return;
    }
    (void)scales; (void)group_size;
    dim3 grid((M + 3) / 4);
    dim3 block(128);
    gemv_q4k_kernel<<<grid, block, 0, stream>>>(
        y, (const block_q4_K*)W_q4, x, M, K);


}

}  // namespace jllm
