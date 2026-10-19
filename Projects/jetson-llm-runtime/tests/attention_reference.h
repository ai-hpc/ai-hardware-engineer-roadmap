#pragma once
// Independent double-precision, materialized-score oracle for tiny fixtures.
// This is not the CUDA kernel and carries no GPU execution/performance claim.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace attention_test {
struct Case {
    std::string name;
    int heads, kv_heads, dim, length, capacity;
    float scale;
    std::vector<float> q, k, v;
    std::vector<int8_t> k8, v8;
    std::vector<float> scales;
};

inline Case fixture(std::string name, int heads, int kv_heads, int dim,
                    int length, int capacity) {
    Case c{std::move(name),heads,kv_heads,dim,length,capacity,
           1.0f/std::sqrt(float(dim)),{},{},{},{},{},{}};
    c.q.resize(heads*dim);
    c.k.resize(capacity*kv_heads*dim, std::numeric_limits<float>::quiet_NaN());
    c.v=c.k;
    for (int h=0;h<heads;++h)
        for (int d=0;d<dim;++d) c.q[h*dim+d]=float((h*3+d*5)%17-8)/16;
    // Nested positions/heads/dimensions make the independently encoded layout clear.
    for (int t=0;t<length;++t) for (int h=0;h<kv_heads;++h) for (int d=0;d<dim;++d) {
        const auto index=(t*kv_heads+h)*dim+d;
        c.k[index]=float((t*7+h*11+d*3)%29-14)/16;
        c.v[index]=float((t*13+h*17+d*5)%47-23)/8;
    }
    return c;
}

inline std::vector<double> scores(const Case& c, int query_head) {
    const int kv_head=query_head/(c.heads/c.kv_heads);
    std::vector<double> result(c.length);
    for (int t=0;t<c.length;++t) {
        double dot=0;
        for (int d=0;d<c.dim;++d)
            dot+=double(c.q.at(query_head*c.dim+d))*c.k.at((t*c.kv_heads+kv_head)*c.dim+d);
        result[t]=dot*c.scale;
        if (!std::isfinite(result[t])) throw std::runtime_error("nonfinite fixture score");
    }
    return result;
}

inline std::vector<double> reference(const Case& c) {
    std::vector<double> out(c.heads*c.dim);
    for (int h=0;h<c.heads;++h) {
        auto s=scores(c,h);
        const double maximum=*std::max_element(s.begin(),s.end());
        double denominator=0;
        for (double& value:s) {value=std::exp(value-maximum);denominator+=value;}
        for (int d=0;d<c.dim;++d) {
            double numerator=0;
            for (int t=0;t<c.length;++t)
                numerator+=s[t]*c.v.at((t*c.kv_heads+h/(c.heads/c.kv_heads))*c.dim+d);
            out[h*c.dim+d]=numerator/denominator;
            if(!std::isfinite(out[h*c.dim+d])) throw std::runtime_error("nonfinite fixture output");
        }
    }
    return out;
}

// A separately executable mathematical model checks online rescaling against
// the full-score oracle. It does not execute CUDA shuffles or shared memory.
inline std::vector<double> online_model(const Case& c, int tile) {
    if (tile<=0) throw std::invalid_argument("positive tile required");
    std::vector<double> out(c.heads*c.dim);
    for (int h=0;h<c.heads;++h) {
        const auto s=scores(c,h);
        double maximum=-std::numeric_limits<double>::infinity(),denominator=0;
        std::vector<double> numerator(c.dim,0);
        for (int begin=0;begin<c.length;begin+=tile) {
            const int end=std::min(begin+tile,c.length);
            const double next_max=std::max(maximum,*std::max_element(s.begin()+begin,s.begin()+end));
            const double correction=std::exp(maximum-next_max);
            denominator*=correction;
            for (auto& value:numerator) value*=correction;
            for (int t=begin;t<end;++t) {
                const double probability=std::exp(s[t]-next_max);
                denominator+=probability;
                for (int d=0;d<c.dim;++d)
                    numerator[d]+=probability*c.v.at((t*c.kv_heads+h/(c.heads/c.kv_heads))*c.dim+d);
            }
            maximum=next_max;
        }
        for (int d=0;d<c.dim;++d) out[h*c.dim+d]=numerator[d]/denominator;
    }
    return out;
}

inline std::vector<Case> cases() {
    std::vector<Case> result;
    auto negative=fixture("all_negative_logits",1,1,1,2,4);
    negative.q={1}; negative.k[0]=-1000; negative.k[1]=-1001;
    negative.v[0]=2; negative.v[1]=10; result.push_back(negative);
    auto layout=fixture("token_major_gqa",4,2,3,3,7);
    std::fill(layout.q.begin(),layout.q.end(),0);
    for (int t=0;t<3;++t) for (int h=0;h<2;++h) for (int d=0;d<3;++d)
        layout.v[(t*2+h)*3+d]=float(h*100+t*10+d);
    result.push_back(layout);
    result.push_back(fixture("one_token_multi_query",4,1,128,1,8));
    result.push_back(fixture("tile_tail_dimension_over_block",6,2,129,65,70));
    result.push_back(fixture("multiple_tiles_gqa",8,2,32,130,140));
    auto increasing=fixture("new_maximum_in_later_tile",2,1,1,129,132);
    increasing.q={1,1};
    for (int t=0;t<129;++t) {increasing.k[t]=float(t-200);increasing.v[t]=float(t)/8;}
    result.push_back(increasing);
    auto prefix=fixture("prefix",4,2,3,129,132);
    for (int length:{1,17,63,64,65,129}) {
        prefix.name="same_storage_prefix_"+std::to_string(length);prefix.length=length;
        result.push_back(prefix);
    }
    auto quantized=fixture("explicit_int8_per_head_scales",4,2,7,65,70);
    quantized.k8.resize(quantized.k.size());quantized.v8.resize(quantized.v.size());
    quantized.scales={.125f,.25f,.5f,.125f};
    for (int t=0;t<65;++t) for (int h=0;h<2;++h) for (int d=0;d<7;++d) {
        const int i=(t*2+h)*7+d;
        quantized.k8[i]=int8_t((t+h*7+d*3)%31-15);
        quantized.v8[i]=int8_t((t*3+h*11+d)%41-20);
        quantized.k[i]=quantized.k8[i]*quantized.scales[h];
        quantized.v[i]=quantized.v8[i]*quantized.scales[2+h];
    }
    result.push_back(quantized);
    return result;
}
} // namespace attention_test
