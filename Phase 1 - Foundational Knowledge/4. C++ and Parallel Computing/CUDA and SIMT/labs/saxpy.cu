// CUDA compilation checked with nvcc13.4.92, sm_80. GPU execution not performed.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(call) check((call), #call)

__global__ void saxpy(float alpha, const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = alpha * a[i] + b[i];
}

static void validate(int n) {
    // The empty operation is well-defined and needs no zero-byte device buffer.
    if (n == 0) return;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>((i % 31) - 15);
        b[i] = static_cast<float>((i % 17) - 8);
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));
    CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    saxpy<<<(n + 255) / 256, 256>>>(2.0f, da, db, dc, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
    for (int i = 0; i < n; ++i) {
        // These small integer-valued floats and alpha=2 are exactly representable.
        if (c[i] != 2.0f * a[i] + b[i]) {
            std::fprintf(stderr, "Mismatch: n=%d index=%d expected=%g actual=%g\n",
                         n, i, 2.0f * a[i] + b[i], c[i]);
            std::exit(EXIT_FAILURE);
        }
    }
}

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 1) {
        std::fprintf(stderr, "No CUDA device available; GPU tests were not run.\n");
        return EXIT_FAILURE;
    }
    for (const int n : {0, 1, 31, 32, 33, 255, 256, 257, 1025}) validate(n);
    std::puts("Validated SAXPY at 9 boundary lengths on a CUDA device.");
}
