// Contiguous managed-memory KV storage for the experimental attention layout.
// There is no pageable overflow tier and no measured bandwidth claim.
#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
namespace jllm {
bool KVCachePool::init(const Config& cfg) {
    destroy();
    if (cfg.n_layers <= 0 || cfg.n_kv_heads <= 0 || cfg.head_dim <= 0
        || cfg.max_context <= 0 || cfg.overflow_context != 0 || cfg.kv_type_bytes != 2) {
        std::fprintf(stderr, "[kv] only positive, contiguous FP16 configurations are supported\n");
        return false;
    }
    int64_t bytes = 2;
    for (int value : {cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, cfg.max_context, cfg.kv_type_bytes}) {
        if (bytes > std::numeric_limits<int64_t>::max()/value) return false;
        bytes *= value;
    }
    if (cudaMallocManaged(&gpu_pool_, static_cast<std::size_t>(bytes)) != cudaSuccess) return false;
    cfg_ = cfg;
    std::memset(gpu_pool_, 0, static_cast<std::size_t>(bytes));
    return true;
}
void KVCachePool::destroy() {
    if (gpu_pool_) cudaFree(gpu_pool_);
    gpu_pool_ = nullptr; cpu_pool_ = nullptr;
    cfg_ = {}; used_tokens_ = 0; gpu_tokens_ = 0;
}
void* KVCachePool::key_ptr(int layer, int pos) {
    if (!gpu_pool_ || layer < 0 || layer >= cfg_.n_layers || pos < 0 || pos >= cfg_.max_context)
        return nullptr;
    used_tokens_ = std::max(used_tokens_, pos+1); gpu_tokens_ = used_tokens_;
    return static_cast<char*>(gpu_pool_)+layer*layer_stride()+pos*(entry_bytes()/2);
}
void* KVCachePool::val_ptr(int layer, int pos) {
    auto* key = static_cast<char*>(key_ptr(layer,pos));
    return key ? key+cfg_.max_context*(entry_bytes()/2) : nullptr;
}
int64_t KVCachePool::used_bytes() const { return int64_t(used_tokens_)*entry_bytes()*cfg_.n_layers; }
int64_t KVCachePool::capacity_bytes() const { return int64_t(cfg_.max_context)*entry_bytes()*cfg_.n_layers; }
void KVCachePool::evict(int) { std::fprintf(stderr, "[kv] eviction is unsupported; stop at the context bound\n"); }
void KVCachePool::clear() { used_tokens_ = 0; gpu_tokens_ = 0; }
}
