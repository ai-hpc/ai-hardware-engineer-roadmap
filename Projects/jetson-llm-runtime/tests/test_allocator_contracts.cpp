#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    static_assert(!std::is_copy_constructible_v<jllm::ScratchPool>);
    static_assert(!std::is_copy_assignable_v<jllm::ScratchPool>);
    static_assert(!std::is_copy_constructible_v<jllm::KVCachePool>);
    static_assert(!std::is_copy_assignable_v<jllm::KVCachePool>);
    {
        jllm::ScratchPool pool;
        require(!pool.init(-1) && !pool.init(0), "invalid scratch initialization accepted");
        require(pool.get(1)==nullptr, "uninitialized scratch returned pointer");
        require(pool.init(1024), "scratch fixture allocation failed");
        auto* first=static_cast<char*>(pool.get(1));
        auto* second=static_cast<char*>(pool.get(257));
        require(first && second==first+256 && pool.used()==768, "scratch padded offsets");
        require(pool.get(257)==nullptr && pool.used()==768, "exhaustion changed offset");
        require(pool.get(std::numeric_limits<int64_t>::max()-255)==nullptr && pool.used()==768,
                "huge rejected request overflowed diagnostic or changed state");
        require(pool.get(-1)==nullptr && pool.get(0)==nullptr, "invalid request accepted");
        pool.reset();
        require(pool.used()==0 && pool.get(1024)==first && pool.used()==1024, "scratch reset/exact fit");
        cuda_stub::fail_next=true;
        require(!pool.init(512) && pool.capacity()==0 && pool.used()==0 && pool.get(1)==nullptr,
                "failed reinitialization retained stale allocation");
        require(cuda_stub::allocations.empty(), "failed reinitialization leaked old allocation");
        require(pool.init(512), "scratch reallocation failed");
        // Destructor must own the allocation; no explicit destroy here.
    }
    require(cuda_stub::allocations.empty(), "scratch destructor leaked");
    {
        jllm::KVCachePool pool;
        jllm::KVCachePool::Config config{2,2,4,3,0,2};
        require(pool.init(config) && pool.capacity_bytes()==192, "KV fixture size");
        auto* key0=static_cast<unsigned char*>(pool.key_ptr(0,0));
        auto* value0=static_cast<unsigned char*>(pool.val_ptr(0,0));
        auto* key2=static_cast<unsigned char*>(pool.key_ptr(0,2));
        auto* layer1=static_cast<unsigned char*>(pool.key_ptr(1,0));
        require(key0 && value0==key0+48 && key2==key0+32 && layer1==key0+96, "K/V layout overlap");
        key0[0]=11; value0[0]=22; key2[0]=33; layer1[0]=44;
        require(key0[0]==11 && value0[0]==22 && key2[0]==33 && layer1[0]==44, "K/V writes alias");
        require(pool.used_tokens()==3 && pool.used_bytes()==192, "KV usage high-water mark");
        require(!pool.key_ptr(-1,0) && !pool.key_ptr(2,0) && !pool.val_ptr(0,-1) && !pool.val_ptr(0,3),
                "invalid KV location accepted");
        pool.clear();
        require(pool.used_tokens()==0 && pool.used_bytes()==0 && pool.capacity_bytes()==192, "KV clear contract");
        config.overflow_context=1;
        require(!pool.init(config) && pool.capacity_bytes()==0 && !pool.key_ptr(0,0), "unsupported overflow retained state");
        config={2,2,4,3,0,2}; cuda_stub::fail_next=true;
        require(!pool.init(config) && !pool.key_ptr(0,0), "KV allocation failure retained pointer");
        config.n_layers=config.n_kv_heads=config.head_dim=std::numeric_limits<int>::max();
        require(!pool.init(config), "KV size product overflow accepted");
        config={2,2,4,3,0,2};
        require(pool.init(config), "KV reinitialization failed");
        pool.destroy(); pool.destroy();
        require(pool.capacity_bytes()==0 && !pool.val_ptr(0,0), "KV destroy not idempotent");
        require(pool.init(config), "KV final allocation failed");
    }
    require(cuda_stub::allocations.empty() && cuda_stub::invalid_frees==0, "owned lifetime failure");
    std::cout << "allocator CPU-shim checks passed; no CUDA runtime or GPU execution\n";
}
