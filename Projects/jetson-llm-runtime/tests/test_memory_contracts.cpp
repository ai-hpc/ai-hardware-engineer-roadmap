#include "jllm_memory.h"
#include "jllm_model.h"
#include <iostream>
#include <limits>
#include <stdexcept>
static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    jllm::MemoryBudget b {};
    b.total_mb=8192; b.os_mb=1024; b.cma_mb=512; b.cuda_ctx_mb=256;
    b.model_mb=2048; b.scratch_mb=128; b.safety_mb=256;
    require(b.free_mb()==4480, "CMA must not be double-subtracted from availability");
    require(b.max_context(24,8,128,2)==47786, "independent GQA KV capacity calculation");
    b.kv_cache_mb=1000;
    require(b.free_mb()==3480 && b.max_kv_mb()==4480, "allocated KV vs total KV capacity");
    require(!b.can_allocate(-1) && b.can_allocate(3480) && !b.can_allocate(3481), "inclusive admission bound");
    require(b.max_context(0,8,128,2)==0 && b.max_context(24,-8,128,2)==0, "invalid KV dimensions");
    const int huge=std::numeric_limits<int>::max();
    require(b.max_context(huge,huge,huge,huge)==0, "KV multiplication overflow");
    b.model_mb=10000;
    require(b.max_context(24,8,128,2)==0, "negative remaining capacity");
    const auto maximum = std::numeric_limits<int64_t>::max();
    jllm::MemoryBudget overflow{};
    overflow.total_mb = maximum; overflow.os_mb = maximum; overflow.model_mb = 1;
    require(overflow.used_mb() == -1 && overflow.free_mb() == -1 && overflow.max_kv_mb() == -1
            && !overflow.can_allocate(0) && overflow.max_context(1,1,1,1) == 0,
            "reserve sum overflow must fail closed");
    overflow = {}; overflow.total_mb = maximum; overflow.os_mb = maximum; overflow.safety_mb = 1;
    require(overflow.used_mb() == maximum && overflow.free_mb() == -1
            && overflow.max_kv_mb() == -1, "safety reserve overflow");
    for (auto field : {&jllm::MemoryBudget::total_mb, &jllm::MemoryBudget::os_mb,
                       &jllm::MemoryBudget::cma_mb, &jllm::MemoryBudget::cuda_ctx_mb,
                       &jllm::MemoryBudget::model_mb, &jllm::MemoryBudget::kv_cache_mb,
                       &jllm::MemoryBudget::scratch_mb, &jllm::MemoryBudget::safety_mb}) {
        jllm::MemoryBudget invalid{}; invalid.total_mb = 100; invalid.*field = -1;
        require(!invalid.valid() && invalid.used_mb() == -1 && invalid.free_mb() == -1
                && invalid.max_kv_mb() == -1 && !invalid.can_allocate(0),
                "negative budget field manufactured capacity");
    }
    jllm::ModelConfig config{};
    config.n_layers=24; config.n_heads=32; config.n_kv_heads=8; config.head_dim=128;
    require(config.kv_per_token_bytes(2)==98304 && config.gqa_group_size()==4,
            "model KV bytes and GQA fixture");
    config.n_heads=31;
    require(config.gqa_group_size()==0, "nonintegral GQA accepted");
    config.n_layers=config.n_kv_heads=config.head_dim=huge;
    require(config.kv_per_token_bytes(8)==0, "model KV product overflow");
    config.n_layers=-1;
    require(config.kv_per_token_bytes(2)==0, "negative model dimension accepted");
    jllm::OOMGuard guard;
    require(!guard.can_extend(-1), "negative allocation rejected");
    require(!jllm::OOMGuard(-1).can_extend(0), "negative headroom accepted");
    std::cout << "memory planning checks passed; availability is an admission estimate\n";
}
