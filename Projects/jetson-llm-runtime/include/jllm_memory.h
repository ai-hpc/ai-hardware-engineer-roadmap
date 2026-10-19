// jllm_memory.h — Memory-first design for Jetson unified memory
//
// Core principle: on Jetson, CPU and GPU share the SAME 8 GB LPDDR5.
// Every byte allocated for the model is a byte NOT available for KV cache,
// OS, camera, or anything else. We track everything.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <initializer_list>

namespace jllm {

// ── Planning estimate, in MiB; concurrent system use can change availability ────────────────────

struct MemoryBudget {
    int64_t total_mb = 0;          // reported usable memory; do not assume a board constant
    int64_t os_mb = 0;             // current unavailability: MemTotal - MemAvailable
    int64_t cma_mb = 0;            // CmaTotal diagnostic only; not subtracted again from MemAvailable
    int64_t cuda_ctx_mb = 0;       // explicit reserve/estimate, not a measured CUDA allocation
    int64_t model_mb = 0;          // loaded model weights
    int64_t kv_cache_mb = 0;       // current KV cache usage
    int64_t scratch_mb = 0;        // activation / intermediate buffers
    int64_t safety_mb = 0;        // admission headroom; cannot prevent concurrent OOM

    // Invalid public fields or unrepresentable sums return -1, so admission
    // fails closed. A valid overcommitted budget may have negative free space.
    bool valid() const {
        return total_mb >= 0 && os_mb >= 0 && cma_mb >= 0 && cuda_ctx_mb >= 0
            && model_mb >= 0 && kv_cache_mb >= 0 && scratch_mb >= 0 && safety_mb >= 0;
    }

    static int64_t checked_sum(std::initializer_list<int64_t> values) {
        int64_t sum = 0;
        for (int64_t value : values) {
            if (value < 0 || value > std::numeric_limits<int64_t>::max() - sum) return -1;
            sum += value;
        }
        return sum;
    }

    int64_t used_mb() const {
        return valid() ? checked_sum({os_mb, cuda_ctx_mb, model_mb, kv_cache_mb, scratch_mb}) : -1;
    }

    int64_t free_mb() const {
        if (!valid()) return -1;
        const auto reserved = checked_sum({os_mb, cuda_ctx_mb, model_mb, kv_cache_mb, scratch_mb, safety_mb});
        return reserved < 0 ? -1 : total_mb - reserved;
    }

    bool can_allocate(int64_t request_mb) const {
        return request_mb >= 0 && free_mb() >= request_mb;
    }

    // Maximum KV cache size given current allocations
    int64_t max_kv_mb() const {
        if (!valid()) return -1;
        const auto reserved = checked_sum({os_mb, cuda_ctx_mb, model_mb, scratch_mb, safety_mb});
        return reserved < 0 ? -1 : total_mb - reserved;
    }

    // Calculate max context tokens for a specific model config
    int max_context(int n_layers, int n_kv_heads, int head_dim, int kv_bytes) const {
        if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || kv_bytes <= 0 || max_kv_mb() <= 0)
            return 0;
        int64_t bytes_per_token = 2;
        for (int value : {n_layers, n_kv_heads, head_dim, kv_bytes}) {
            if (bytes_per_token > std::numeric_limits<int64_t>::max()/value) return 0;
            bytes_per_token *= value;
        }
        if (max_kv_mb() > std::numeric_limits<int64_t>::max()/(1024*1024)) return 0;
        const auto tokens = max_kv_mb()*1024*1024/bytes_per_token;
        return static_cast<int>(tokens > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : tokens);
    }

    void print() const {
        fprintf(stderr,
            "╔══════════════════════════════════╗\n"
            "║   JLLM Memory Budget             ║\n"
            "╠══════════════════════════════════╣\n"
            "║ Total DRAM:     %5ld MB         ║\n"
            "║ OS + kernel:   -%5ld MB         ║\n"
            "║ CMA (info):     %5ld MB         ║\n"
            "║ CUDA context:  -%5ld MB         ║\n"
            "║ Model weights: -%5ld MB         ║\n"
            "║ KV cache:      -%5ld MB         ║\n"
            "║ Scratch:       -%5ld MB         ║\n"
            "║ Safety margin: -%5ld MB         ║\n"
            "╠══════════════════════════════════╣\n"
            "║ FREE:           %5ld MB         ║\n"
            "╚══════════════════════════════════╝\n",
            total_mb, os_mb, cma_mb, cuda_ctx_mb,
            model_mb, kv_cache_mb, scratch_mb, safety_mb, free_mb());
    }
};

// Read actual system state from /proc
MemoryBudget probe_system_memory();

// ── Admission heuristic; it cannot prevent concurrent allocation failures ──────────────

class OOMGuard {
public:
    explicit OOMGuard(int64_t safety_mb = 256) : safety_mb_(safety_mb) {}

    // Check before every KV cache extension
    bool can_extend(int64_t additional_bytes) const;

    // Get real free memory from kernel (not cached)
    int64_t real_free_mb() const;

    // Compatibility hook; never alters global kernel caches or compaction
    void emergency_free();

private:
    int64_t safety_mb_;
};

// ── KV Cache Pool — pre-allocated, no runtime malloc ─────────────────────
//
// The current teaching path uses one cudaMallocManaged allocation with separate
// K and V spans per layer. This establishes a CUDA-managed allocation contract;
// residency/cache behavior and performance still require target measurement.
// Pageable overflow and INT8 scale storage are unsupported.

class KVCachePool {
public:
    KVCachePool() = default;
    ~KVCachePool() { destroy(); }
    KVCachePool(const KVCachePool&) = delete;
    KVCachePool& operator=(const KVCachePool&) = delete;
    struct Config {
        int n_layers;
        int n_kv_heads;
        int head_dim;
        int max_context;       // max tokens in GPU-fast pool
        int overflow_context;  // must be 0; overflow layout is unsupported
        int kv_type_bytes;     // must be 2 (FP16) until scale storage is implemented
    };

    bool init(const Config& cfg);
    void destroy();

    // Get pointer for layer l, token position t
    // Returns a managed pointer, or nullptr for an invalid layer/position.
    void* key_ptr(int layer, int pos);
    void* val_ptr(int layer, int pos);

    int64_t used_bytes() const;
    int64_t capacity_bytes() const;
    int     used_tokens() const { return used_tokens_; }
    int     max_tokens() const { return cfg_.max_context + cfg_.overflow_context; }

    // Unsupported operation: reports a diagnostic; callers must stop at capacity.
    void evict(int n_tokens);

    // Reset for new conversation
    void clear();

private:
    Config cfg_ = {};
    void*  gpu_pool_ = nullptr;   // cudaMallocManaged
    void*  cpu_pool_ = nullptr;   // unused; no overflow allocation
    int    used_tokens_ = 0;
    int    gpu_tokens_ = 0;

    int64_t entry_bytes() const {
        return 2LL * cfg_.n_kv_heads * cfg_.head_dim * cfg_.kv_type_bytes;
    }
    int64_t layer_stride() const {
        return entry_bytes() * (cfg_.max_context + cfg_.overflow_context);
    }
};

// ── Scratch allocator — one-time alloc, reuse forever ────────────────────
// Scratch allocation is reused; other runtime components may still allocate.

class ScratchPool {
public:
    ScratchPool() = default;
    ~ScratchPool() { destroy(); }
    ScratchPool(const ScratchPool&) = delete;
    ScratchPool& operator=(const ScratchPool&) = delete;
    bool init(int64_t size_bytes);
    void destroy();

    void* get(int64_t size);  // bump pointer, reset each forward pass
    void* get_or_throw(int64_t size); // std::bad_alloc instead of propagating null
    // Restore an earlier aligned checkpoint. Caller must order every old use
    // before overwrite (e.g. same CUDA stream), or first wait for completion.
    bool rewind(int64_t checkpoint);
    void  reset();            // call at start of each decode step

    int64_t used() const { return offset_; }
    int64_t capacity() const { return capacity_; }

private:
    void*   base_ = nullptr;
    int64_t capacity_ = 0;
    int64_t offset_ = 0;
};

}  // namespace jllm
