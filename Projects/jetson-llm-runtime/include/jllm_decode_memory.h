#pragma once
#include "jllm_memory.h"
#include <algorithm>

namespace jllm {
struct DecodeScratchPlan {
    int64_t token_bytes=0, layer_bytes=0, final_bytes=0, device_bytes=0;
    int64_t host_logits_bytes=0, host_embedding_bytes=0, reservation_bytes=0;
    explicit operator bool() const {return device_bytes>0 && reservation_bytes>0;}
};

inline int64_t aligned_scratch_elements(int64_t count,int64_t width) {
    if(count<=0 || width<=0 || count>(std::numeric_limits<int64_t>::max()-255)/width) return 0;
    return (count*width+255)&~int64_t(255);
}

inline int64_t bytes_to_mib_ceil(int64_t bytes) {
    if(bytes<0)return -1;
    constexpr int64_t mib=1024*1024;
    return bytes/mib+(bytes%mib!=0);
}

inline DecodeScratchPlan decode_scratch_plan(int64_t hidden,int64_t heads,
    int64_t kv_heads,int64_t head_dim,int64_t intermediate,int64_t vocabulary) {
    DecodeScratchPlan p;
    if(hidden<=0 || heads<=0 || kv_heads<=0 || kv_heads>heads || heads%kv_heads
        || head_dim<=0 || heads>std::numeric_limits<int64_t>::max()/head_dim
        || heads*head_dim!=hidden) return {};
    const auto h=aligned_scratch_elements(hidden,2);
    const auto kv=aligned_scratch_elements(kv_heads*head_dim,2);
    const auto ff=aligned_scratch_elements(intermediate,2);
    const auto vh=aligned_scratch_elements(vocabulary,2);
    const auto vf=aligned_scratch_elements(vocabulary,4);
    if(!h||!kv||!ff||!vh||!vf)return {};
    p.token_bytes=h;
    // Allocation order: norm, Q, K, V, attention output/projection, residual,
    // FFN norm, gate/up/activation, FFN output, zero buffer.
    p.layer_bytes=MemoryBudget::checked_sum({h,h,kv,kv,h,h,h,h,ff,ff,ff,h,h});
    p.final_bytes=MemoryBudget::checked_sum({h,h,vh,vf});
    if(p.layer_bytes<=0 || p.final_bytes<=0)return {};
    p.device_bytes=MemoryBudget::checked_sum({p.token_bytes,std::max(p.layer_bytes,p.final_bytes)});
    p.host_logits_bytes=vf; // rounded up; persistent host vector uses vocabulary*4
    p.host_embedding_bytes=8192*(4+2); // fixed arrays in dequant_embedding()
    p.reservation_bytes=MemoryBudget::checked_sum({p.device_bytes,p.host_logits_bytes,p.host_embedding_bytes});
    return p.device_bytes>0 && p.reservation_bytes>0?p:DecodeScratchPlan{};
}
} // namespace jllm
