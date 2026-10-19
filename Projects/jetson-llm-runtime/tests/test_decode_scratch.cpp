#include "jllm_decode_memory.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>

static void require(bool value,const char* message) {
    if(!value)throw std::runtime_error(message);
}

int main() {
    try {
        const auto plan=jllm::decode_scratch_plan(256,4,2,64,512,768);
        require(bool(plan),"valid plan rejected");
        // Independent hand arithmetic: 8*512 + 2*256 + 3*1024 = 7680.
        require(plan.token_bytes==512 && plan.layer_bytes==7680,"layer bytes or alignment wrong");
        require(plan.final_bytes==5632 && plan.device_bytes==8192,"peak is token + maximum live stage");
        require(plan.host_logits_bytes==3072 && plan.host_embedding_bytes==49152
                && plan.reservation_bytes==60416,"host staging omitted from reservation");
        const auto large_vocab=jllm::decode_scratch_plan(256,4,2,64,512,4096);
        require(large_vocab.final_bytes==25600 && large_vocab.device_bytes==26112,"final logits must dominate this fixture");
        require(jllm::aligned_scratch_elements(129,2)==512,"alignment must round each allocation");
        require(jllm::bytes_to_mib_ceil(1048576)==1 && jllm::bytes_to_mib_ceil(1048577)==2,
                "MiB floor under-reserved a buffer");
        require(jllm::bytes_to_mib_ceil(-1)==-1,"invalid byte count accepted");
        const auto maximum=std::numeric_limits<int64_t>::max();
        require(!jllm::decode_scratch_plan(255,4,2,64,512,768),"hidden/head mismatch accepted");
        require(!jllm::decode_scratch_plan(256,4,3,64,512,768),"invalid GQA accepted");
        require(!jllm::decode_scratch_plan(256,4,2,64,0,768),"zero intermediate accepted");
        require(!jllm::decode_scratch_plan(maximum,maximum,1,2,1,1),"head product overflow accepted");
        require(!jllm::decode_scratch_plan(maximum/4,1,1,maximum/4,1,1),"stage sum overflow accepted");
        require(!jllm::decode_scratch_plan(256,4,2,64,512,maximum),"vocabulary allocation overflow accepted");

        jllm::MemoryBudget budget{};budget.total_mb=4;budget.os_mb=1;budget.model_mb=1;
        require(budget.max_context(1,2,64,2)==4096,"unreserved comparison fixture wrong");
        budget.scratch_mb=jllm::bytes_to_mib_ceil(plan.reservation_bytes);
        require(budget.max_context(1,2,64,2)==2048,"scratch must be reserved before KV admission");
        budget.total_mb=2;require(budget.max_context(1,2,64,2)==0,"overcommit accepted");

        {
            jllm::ScratchPool pool;
            require(pool.init(plan.device_bytes),"pool fixture allocation failed");
            auto* token=static_cast<unsigned char*>(pool.get_or_throw(512));
            std::memset(token,0x5a,512);
            const auto checkpoint=pool.used();
            for(int layer=0;layer<200;++layer) {
                for(int bytes:{512,512,256,256,512,512,512,512,1024,1024,1024,512,512})
                    std::memset(pool.get_or_throw(bytes),layer%256,bytes);
                require(pool.used()==8192,"per-layer allocation trace disagrees with plan");
                require(pool.rewind(checkpoint) && pool.used()==512,"layer temporary reuse failed");
                for(int i=0;i<512;++i)require(token[i]==0x5a,"rewind overwrote persistent token state");
            }
            for(int bytes:{512,512,1536,3072})pool.get_or_throw(bytes);
            require(pool.used()==6144,"final stage allocation trace wrong");
            const auto before=pool.used();
            for(int64_t invalid:{int64_t(-1),int64_t(1),before+256})
                require(!pool.rewind(invalid) && pool.used()==before,"bad rewind changed ownership");
        }
        {
            jllm::ScratchPool short_pool;require(short_pool.init(plan.device_bytes-1),"short fixture allocation failed");
            bool rejected=false;
            try {short_pool.get_or_throw(plan.device_bytes);}
            catch(const std::bad_alloc&){rejected=true;}
            require(rejected && short_pool.used()==0,"exhaustion must throw before a null reaches a kernel");
        }
        require(cuda_stub::allocations.empty() && cuda_stub::invalid_frees==0,"scratch ownership leak");
        std::cout<<"decode scratch checks passed: known peaks, host reservation, admission, overflow, "
                 <<"200-layer pointer reuse, persistent state, invalid rewind and fail-closed allocation\n";
        std::cout<<"CUDA stream execution: not performed by CPU shim\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
