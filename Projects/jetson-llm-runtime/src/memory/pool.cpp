// Scratch bump allocator; other runtime components can still allocate.

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <new>

namespace jllm {

bool ScratchPool::init(int64_t size_bytes) {
    destroy();
    if (size_bytes <= 0) return false;
    cudaError_t err = cudaMallocManaged(&base_, static_cast<size_t>(size_bytes));
    if (err != cudaSuccess) {
        fprintf(stderr, "[scratch] cudaMallocManaged(%ld MiB) failed: %s\n",
                size_bytes / (1024*1024), cudaGetErrorString(err));
        return false;
    }
    capacity_ = size_bytes;
    offset_ = 0;
    fprintf(stderr, "[scratch] Allocated %ld MB scratch pool\n", size_bytes / (1024*1024));
    return true;
}

void ScratchPool::destroy() {
    if (base_) { cudaFree(base_); base_ = nullptr; }
    capacity_ = 0;
    offset_ = 0;
}

void* ScratchPool::get(int64_t size) {
    // Align offsets; alignment alone does not guarantee coalesced accesses.
    if (!base_ || size <= 0 || size > std::numeric_limits<int64_t>::max()-255) return nullptr;
    size = (size + 255) & ~255LL;

    if (offset_ > capacity_ || size > capacity_ - offset_) {
        fprintf(stderr, "[scratch] pool exhausted (used=%ld requested=%ld capacity=%ld bytes)\n",
                offset_, size, capacity_);
        return nullptr;
    }

    void* ptr = (char*)base_ + offset_;
    offset_ += size;
    return ptr;
}

void ScratchPool::reset() {
    offset_ = 0;
}

void* ScratchPool::get_or_throw(int64_t size) {
    void* pointer=get(size);
    if(!pointer)throw std::bad_alloc();
    return pointer;
}

bool ScratchPool::rewind(int64_t checkpoint) {
    if(!base_ || checkpoint<0 || checkpoint>offset_ || checkpoint%256!=0)return false;
    offset_=checkpoint;
    return true;
}

}  // namespace jllm
