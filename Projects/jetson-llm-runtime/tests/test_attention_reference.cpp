#include "attention_reference.h"
#include "jllm_attention_layout.h"
#include "jllm_memory.h"
#include <cfloat>
#include <cstring>
#include <iostream>

static void require(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static void near(double actual,double expected,double tolerance=1e-11) {
    require(std::isfinite(actual) && std::abs(actual-expected)<=tolerance,"oracle known-answer mismatch");
}

int main() {
    try {
        const auto all=attention_test::cases();
        near(attention_test::reference(all[0])[0],2+8/(1+std::exp(1.0)));
        const auto layout=attention_test::reference(all[1]);
        for (int h=0;h<4;++h) for (int d=0;d<3;++d) near(layout[h*3+d],(h/2)*100+10+d);
        const auto single=attention_test::reference(all[2]);
        for (int h=0;h<4;++h) for (int d=0;d<128;++d) near(single[h*128+d],all[2].v[d]);
        for (const auto& c:all) {
            const auto expected=attention_test::reference(c);
            for (int tile:{1,7,32,64,256}) {
                const auto actual=attention_test::online_model(c,tile);
                for (std::size_t i=0;i<actual.size();++i) near(actual[i],expected[i]);
            }
        }
        // Reproduce the old comparison failure on actual host IEEE-754 bits.
        float sentinel=-FLT_MAX,negative=-1000;
        std::int32_t sentinel_bits,negative_bits;
        std::memcpy(&sentinel_bits,&sentinel,sizeof(float));
        std::memcpy(&negative_bits,&negative,sizeof(float));
        require(negative>sentinel && negative_bits<sentinel_bits,"negative-float ordering counterexample missing");

        // Compile actual KVCachePool pointer logic against a bounded CPU CUDA shim.
        jllm::KVCachePool pool;
        require(pool.init({2,2,3,7,0,2}),"pool fixture allocation failed");
        for (int layer=0;layer<2;++layer) {
            auto* key0=static_cast<std::uint16_t*>(pool.key_ptr(layer,0));
            auto* value0=static_cast<std::uint16_t*>(pool.val_ptr(layer,0));
            require(value0-key0==7*2*3,"capacity-based K/V span separation");
            for (int t=0;t<7;++t) for (int h=0;h<2;++h) for (int d=0;d<3;++d) {
                auto* key=static_cast<std::uint16_t*>(pool.key_ptr(layer,t));
                auto* value=static_cast<std::uint16_t*>(pool.val_ptr(layer,t));
                const auto offset=jllm::kv_element_offset(t,h,d,2,3);
                require(key+h*3+d==key0+offset,"token writer / attention reader K offset mismatch");
                require(value+h*3+d==value0+offset,"token writer / attention reader V offset mismatch");
                key[h*3+d]=std::uint16_t(1000*layer+100*t+10*h+d);
                require(key0[offset]==1000*layer+100*t+10*h+d,"cache payload round trip");
            }
        }
        require(jllm::attention_shape_valid(8,2,128,65,.125f),"valid geometry rejected");
        require(!jllm::attention_shape_valid(3,2,128,1,.1f),"invalid GQA accepted");
        require(!jllm::attention_shape_valid(4,0,128,1,.1f),"zero KV heads accepted");
        require(!jllm::attention_shape_valid(4,2,128,0,.1f),"empty prefix accepted");
        require(!jllm::attention_shape_valid(4,2,4097,1,.1f),"oversized shared allocation accepted");
        require(!jllm::attention_shape_valid(4,2,128,1,NAN),"nonfinite scale accepted");
        require(!jllm::attention_shape_valid(4,2,128,1,-1),"negative scale accepted");
        std::cout<<"attention CPU oracle: "<<all.size()<<" fixtures x 5 tile sizes passed; "
                 <<"known answers, negative-float counterexample, actual pool offset logic and invalid shapes passed\n";
        std::cout<<"CUDA execution: not performed by this CPU test\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
    }
}
