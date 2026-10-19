// test_kernels.cu — CUDA kernel correctness tests (run on Jetson only)

#include "jllm_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ── Test: Softmax ────────────────────────────────────────────────────────
static void test_softmax() {
    const int N = 1024;
    float *d_x;
    std::vector<float> h_x(N);
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));

    for (int i = 0; i < N; i++) h_x[i] = (float)(i - N/2) * 0.01f;
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    jllm::softmax_inplace(d_x, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < N; i++) { assert(h_x[i] >= 0.0f); sum += h_x[i]; }
    assert(fabsf(sum - 1.0f) < 1e-4f);
    printf("PASS: softmax (sum = %.6f)\n", sum);
    cudaFree(d_x);
}

// ── Test: RoPE identity at position 0 ────────────────────────────────────
static void test_rope() {
    const int nh = 4, nkv = 4, hd = 128;
    half *d_q, *d_k;
    CHECK_CUDA(cudaMalloc(&d_q, nh * hd * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_k, nkv * hd * sizeof(half)));

    std::vector<half> ones(nh * hd, __float2half(1.0f));
    cudaMemcpy(d_q, ones.data(), ones.size() * sizeof(half), cudaMemcpyHostToDevice);
    ones.resize(nkv * hd, __float2half(1.0f));
    cudaMemcpy(d_k, ones.data(), ones.size() * sizeof(half), cudaMemcpyHostToDevice);

    jllm::rope_inplace(d_q, d_k, nh, nkv, hd, 0, 10000.0f, 0);
    cudaDeviceSynchronize();

    std::vector<half> res(nh * hd);
    cudaMemcpy(res.data(), d_q, res.size() * sizeof(half), cudaMemcpyDeviceToHost);
    float val = __half2float(res[0]);
    assert(fabsf(val - 1.0f) < 0.01f);
    printf("PASS: rope (pos=0 identity, val[0]=%.4f)\n", val);
    cudaFree(d_q); cudaFree(d_k);
}

// ── Test: Fused RMSNorm + Residual ───────────────────────────────────────
static void test_fused_norm() {
    const int rows = 1, dim = 256;
    half *d_out, *d_x, *d_res, *d_w;
    CHECK_CUDA(cudaMalloc(&d_out, rows * dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_x,   rows * dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_res, rows * dim * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_w,   dim * sizeof(half)));

    std::vector<half> ones(dim, __float2half(1.0f));
    cudaMemcpy(d_x,   ones.data(), dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, ones.data(), dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,   ones.data(), dim * sizeof(half), cudaMemcpyHostToDevice);

    jllm::fused_rmsnorm_residual(d_out, d_x, d_res, d_w, rows, dim, 1e-6f, false, 0);
    cudaDeviceSynchronize();

    std::vector<half> res(dim);
    cudaMemcpy(res.data(), d_out, dim * sizeof(half), cudaMemcpyDeviceToHost);
    float val = __half2float(res[0]);
    // x+res=2.0, RMS=2.0, normalized=1.0, weight=1.0 → output=1.0
    assert(fabsf(val - 1.0f) < 0.05f);
    printf("PASS: fused_rmsnorm (val[0]=%.4f, expected ~1.0)\n", val);
    cudaFree(d_out); cudaFree(d_x); cudaFree(d_res); cudaFree(d_w);
}

// ── Test: FP16 → INT8 quantization ───────────────────────────────────────
static void test_convert() {
    const int rows = 2, cols = 128;
    half *d_src; int8_t *d_dst; float *d_scales;
    CHECK_CUDA(cudaMalloc(&d_src, rows * cols * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_dst, rows * cols * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_scales, rows * sizeof(float)));

    std::vector<half> src(rows * cols);
    for (int i = 0; i < cols; i++) src[i] = __float2half(0.5f);
    for (int i = cols; i < 2*cols; i++) src[i] = __float2half(-0.5f);
    cudaMemcpy(d_src, src.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice);

    jllm::fp16_to_int8(d_dst, d_scales, d_src, rows, cols, 0);
    cudaDeviceSynchronize();

    float h_scales[2];
    cudaMemcpy(h_scales, d_scales, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    assert(fabsf(h_scales[0] - 0.5f/127.0f) < 0.001f);
    printf("PASS: fp16_to_int8 (scale=%.6f, expected ~0.00394)\n", h_scales[0]);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_scales);
}

// ── Test: SwiGLU ─────────────────────────────────────────────────────────
static void test_swiglu() {
    const int n = 256;
    half *d_out, *d_gate, *d_up;
    CHECK_CUDA(cudaMalloc(&d_out,  n * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_gate, n * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_up,   n * sizeof(half)));

    std::vector<half> gate(n, __float2half(1.0f));
    std::vector<half> up(n, __float2half(2.0f));
    cudaMemcpy(d_gate, gate.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up.data(), n * sizeof(half), cudaMemcpyHostToDevice);

    jllm::fused_swiglu(d_out, d_gate, d_up, 1, n, 0);
    cudaDeviceSynchronize();

    std::vector<half> res(n);
    cudaMemcpy(res.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
    float val = __half2float(res[0]);
    // silu(1.0) = 1.0 / (1 + exp(-1)) ≈ 0.7311, × 2.0 = 1.4621
    assert(fabsf(val - 1.4621f) < 0.05f);
    printf("PASS: swiglu (val=%.4f, expected ~1.4621)\n", val);
    cudaFree(d_out); cudaFree(d_gate); cudaFree(d_up);
}

int main() {
    int device_count = 0;
    const auto device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver || (device_status == cudaSuccess && device_count == 0)) {
        fprintf(stderr, "SKIP: no usable CUDA device/driver; target checks were not executed\n");
        return 77;
    }
    if (device_status != cudaSuccess) {
        fprintf(stderr, "FAIL: device enumeration: %s\n", cudaGetErrorString(device_status));
        return 1;
    }
    printf("=== jetson-llm kernel tests ===\n\n");
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    test_softmax();
    test_rope();
    test_fused_norm();
    test_convert();
    test_swiglu();

    printf("\nAll kernel tests passed.\n");
    return 0;
}
