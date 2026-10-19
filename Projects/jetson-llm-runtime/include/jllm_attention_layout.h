#pragma once

#include <cmath>
#include <cstdint>

namespace jllm {

// The implemented decode tile uses <= 17 KiB dynamic shared memory at this bound.
// Allocation capacity and data finiteness remain caller contracts.
inline bool attention_shape_valid(int heads, int kv_heads, int dim, int length,
                                  float scale) {
    return heads > 0 && heads <= 256 && kv_heads > 0 && kv_heads <= heads
        && heads % kv_heads == 0 && dim > 0 && dim <= 4096 && length > 0
        && std::isfinite(scale) && scale > 0;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline std::int64_t kv_element_offset(int position, int kv_head, int dimension,
                                      int kv_heads, int head_dim) {
    // Each K or V span is [capacity, kv_heads, head_dim], contiguous in dim.
    // Current sequence length changes the prefix read, never this stride.
    return (std::int64_t(position) * kv_heads + kv_head) * head_dim + dimension;
}

} // namespace jllm
