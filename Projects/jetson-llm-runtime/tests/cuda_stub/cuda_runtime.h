#pragma once
// Bounded CPU-only API shim. Never used by production/Jetson targets.
#include <cstddef>
#include <cstdlib>
#include <unordered_set>

enum cudaError_t { cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorInvalidValue=1 };
namespace cuda_stub {
inline std::unordered_set<void*> allocations;
inline bool fail_next = false;
inline int invalid_frees = 0;
}
inline cudaError_t cudaMallocManaged(void** pointer, std::size_t bytes) {
    if (!pointer) return cudaErrorInvalidValue;
    *pointer = nullptr;
    if (cuda_stub::fail_next) {
        cuda_stub::fail_next = false;
        return cudaErrorMemoryAllocation;
    }
    if (bytes == 0 || bytes > 1024*1024) return cudaErrorMemoryAllocation;
    *pointer = std::malloc(bytes);
    if (!*pointer) return cudaErrorMemoryAllocation;
    cuda_stub::allocations.insert(*pointer);
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* pointer) {
    if (!pointer) return cudaSuccess;
    if (cuda_stub::allocations.erase(pointer) != 1) {
        ++cuda_stub::invalid_frees;
        return cudaErrorInvalidValue;
    }
    std::free(pointer);
    return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t error) {
    return error == cudaSuccess ? "success (CPU shim)" : "injected/bounded allocation failure (CPU shim)";
}
