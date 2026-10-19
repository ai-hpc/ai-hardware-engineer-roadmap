// test_memory.cpp — Memory subsystem tests
// Note: uses explicit checks, NOT assert() (assert disabled in Release builds)

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); } \
} while(0)

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
    // Initialize CUDA context
    cudaError_t cerr = cudaSetDevice(0);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaSetDevice: %s\n", cudaGetErrorString(cerr));
        return 1;
    }
    printf("CUDA initialized\n");

    // Test 1: probe system memory
    printf("\nTest 1: probe_system_memory\n");
    auto budget = jllm::probe_system_memory();
    CHECK(budget.total_mb > 0, "total_mb should be > 0");
    CHECK(budget.free_mb() > 0, "free_mb should be > 0");
    budget.print();
    printf("PASS: probe_system_memory\n");

    // Test 2: OOM guard
    printf("\nTest 2: OOMGuard\n");
    jllm::OOMGuard guard(256);
    int64_t free_mb = guard.real_free_mb();
    CHECK(free_mb > 0, "real_free_mb should be > 0");
    printf("Free: %ld MB\n", free_mb);
    printf("PASS: OOMGuard\n");

    // Test 3: scratch pool
    printf("\nTest 3: ScratchPool\n");
    jllm::ScratchPool scratch;

    // Try smaller size if 64 MB fails
    int64_t pool_size = 64 * 1024 * 1024;  // 64 MB
    bool ok = scratch.init(pool_size);
    if (!ok) {
        fprintf(stderr, "[test] 64 MB failed, trying 4 MB...\n");
        pool_size = 4 * 1024 * 1024;
        ok = scratch.init(pool_size);
    }
    if (!ok) {
        fprintf(stderr, "[test] cudaMallocManaged failed even at 4 MB\n");
        fprintf(stderr, "[test] No regular malloc GPU fallback is supported...\n");
        // This means CUDA pinned alloc isn't working — diagnose
        size_t free_cuda = 0, total_cuda = 0;
        cudaMemGetInfo(&free_cuda, &total_cuda);
        fprintf(stderr, "[test] CUDA mem: %zu MB free / %zu MB total\n",
                free_cuda / (1024*1024), total_cuda / (1024*1024));
        fprintf(stderr, "FAIL: ScratchPool allocation unavailable on this configured target\n");
        return 1;
    } else {
        printf("[test] Pool allocated: %ld MB\n", pool_size / (1024*1024));
        printf("[test] Pool capacity: %ld bytes\n", scratch.capacity());

        void* a = scratch.get(1024);
        void* b = scratch.get(2048);

        CHECK(a != nullptr, "get(1024) returned null");
        CHECK(b != nullptr, "get(2048) returned null");
        CHECK(a != b, "get() returned same pointer twice");

        // get() aligns to 256: 1024→1024, 2048→2048 (both already aligned)
        printf("[test] used after 2 gets: %ld bytes (expect 3072)\n", scratch.used());
        CHECK(scratch.used() == 3072, "used() should be 3072");

        scratch.reset();
        CHECK(scratch.used() == 0, "used() should be 0 after reset");

        scratch.destroy();
        printf("PASS: ScratchPool\n");
    }

    printf("\nAll memory tests passed.\n");
    return 0;
}
