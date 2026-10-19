// Finite CUDA correctness examples. GPU execution is a separate validation step.
// Build: nvcc -std=c++17 -O2 -arch=sm_80 cuda_examples.cu -o cuda_examples
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;
static void check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expression) check((expression), #expression)

template<class T> struct DeviceArray {
  T* p = nullptr;
  explicit DeviceArray(size_t n) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
  }
  ~DeviceArray() { if (p) cudaFree(p); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
};
struct Stream {
  cudaStream_t value{};
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Graph {
  cudaGraph_t value{};
  cudaGraphExec_t exec{};
  ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (value) cudaGraphDestroy(value); }
};

__global__ void affine(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 2.0f * x[i] + 1.0f;
}
__global__ void relu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(x[i], 0.0f);
}
__global__ void fused_affine_relu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(2.0f * x[i] + 1.0f, 0.0f);
}

// Contract: exactly 256 threads in one-dimensional blocks; no early return.
__global__ void block_partial_sum(const float* x, float* partial, int n) {
  auto block = cg::this_thread_block();
  __shared__ float values[256];
  const int local = block.thread_rank();
  const int global = blockIdx.x * blockDim.x + local;
  values[local] = global < n ? x[global] : 0.0f;
  block.sync();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (local < stride) values[local] += values[local + stride];
    block.sync();
  }
  if (local == 0) partial[blockIdx.x] = values[0];
}

// Input and tile count are fixed before launch. No concurrent host producer.
__global__ void finite_tile_worker(const float* x, float* y, int n, int* next_tile) {
  __shared__ int tile;
  const int tile_count = (n + blockDim.x - 1) / blockDim.x;
  while (true) {
    if (threadIdx.x == 0) tile = atomicAdd(next_tile, 1);
    __syncthreads();                 // broadcast to the whole block
    if (tile >= tile_count) break;    // uniform decision by all block members
    int i = tile * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(2.0f * x[i] + 1.0f, 0.0f);
    __syncthreads();                 // all consumers finish before reusing tile
  }
}

static void equal(const std::vector<float>& actual, const std::vector<float>& expected,
                  const char* label) {
  if (actual.size() != expected.size()) throw std::runtime_error("size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-5f)
      throw std::runtime_error(std::string(label) + " mismatch at " + std::to_string(i));
  }
}

static int exercise(int n, int multiprocessors) {
  // Empty input is handled on the host; a zero-block kernel launch is invalid.
  if (n == 0) return 1;
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  DeviceArray<float> input(n), intermediate(n), output(n), partial(blocks);
  DeviceArray<int> counter(1);
  Stream stream;
  Graph graph;
  std::vector<float> x(n), expected(n), actual(n), expected_partial(blocks), actual_partial(blocks);

  CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
  affine<<<blocks, threads, 0, stream.value>>>(input.p, intermediate.p, n);
  relu<<<blocks, threads, 0, stream.value>>>(intermediate.p, output.p, n);
  CUDA_CHECK(cudaStreamEndCapture(stream.value, &graph.value));
  CUDA_CHECK(cudaGraphInstantiate(&graph.exec, graph.value, 0));

  int comparisons = 0;
  for (int variation = 0; variation < 2; ++variation) {
    for (int i = 0; i < n; ++i) {
      // Dyadic values keep this simple reference exact at these small sizes.
      x[i] = static_cast<float>((i * 7 + variation * 13) % 37 - 18) / 16.0f;
      expected[i] = std::max(2.0f * x[i] + 1.0f, 0.0f);
    }
    CUDA_CHECK(cudaMemcpyAsync(input.p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    // Poison each independent result: a missing store must not inherit the
    // preceding implementation's correct answer. 0xff bytes form FP32 NaNs.
    CUDA_CHECK(cudaMemsetAsync(output.p, 0xff, n * sizeof(float), stream.value));
    CUDA_CHECK(cudaGraphLaunch(graph.exec, stream.value));
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.p, n * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    equal(actual, expected, "graph replay"); ++comparisons;

    CUDA_CHECK(cudaMemsetAsync(output.p, 0xff, n * sizeof(float), stream.value));
    fused_affine_relu<<<blocks, threads, 0, stream.value>>>(input.p, output.p, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.p, n * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    equal(actual, expected, "fused"); ++comparisons;

    std::fill(expected_partial.begin(), expected_partial.end(), 0.0f);
    for (int i = 0; i < n; ++i) expected_partial[i / threads] += x[i];
    CUDA_CHECK(cudaMemsetAsync(partial.p, 0xff, blocks * sizeof(float), stream.value));
    block_partial_sum<<<blocks, threads, 0, stream.value>>>(input.p, partial.p, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(actual_partial.data(), partial.p, blocks * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    equal(actual_partial, expected_partial, "partial reduction"); ++comparisons;

    CUDA_CHECK(cudaMemsetAsync(counter.p, 0, sizeof(int), stream.value));
    CUDA_CHECK(cudaMemsetAsync(output.p, 0xff, n * sizeof(float), stream.value));
    const int worker_blocks = std::max(1, std::min(blocks, 2 * multiprocessors));
    finite_tile_worker<<<worker_blocks, threads, 0, stream.value>>>(input.p, output.p, n, counter.p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), output.p, n * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    equal(actual, expected, "finite tile worker"); ++comparisons;
  }
  return comparisons;
}

int main() {
  int count = 0;
  cudaError_t available = cudaGetDeviceCount(&count);
  if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver ||
      (available == cudaSuccess && count == 0)) {
    std::puts("SKIP: CUDA device/driver unavailable; executable was not GPU-validated.");
    return 77;
  }
  try {
    CUDA_CHECK(available);
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int comparisons = 0;
    for (int n : {0, 1, 31, 32, 33, 255, 256, 257, 1003})
      comparisons += exercise(n, properties.multiProcessorCount);
    std::printf("PASS: %d checks on %s (includes empty-input host check).\n", comparisons, properties.name);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
