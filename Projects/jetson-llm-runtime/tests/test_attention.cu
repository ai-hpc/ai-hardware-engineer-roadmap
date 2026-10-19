#include "jllm_kernels.h"
#include "attention_reference.h"
#include <cstdio>
#include <stdexcept>

static void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Device {
    T* data=nullptr;
    explicit Device(std::size_t count) {checked(cudaMalloc(&data,count*sizeof(T)));}
    ~Device() {if(data)cudaFree(data);}
    Device(const Device&)=delete;
    void upload(const std::vector<T>& values) {
        checked(cudaMemcpy(data,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
};

static void check_case(attention_test::Case c) {
    std::vector<half> q(c.q.size()),k(c.k.size()),v(c.v.size());
    for (std::size_t i=0;i<q.size();++i) {q[i]=__float2half(c.q[i]);c.q[i]=__half2float(q[i]);}
    for (std::size_t i=0;i<k.size();++i) {
        k[i]=__float2half(c.k[i]);v[i]=__float2half(c.v[i]);
        if(c.k8.empty()){c.k[i]=__half2float(k[i]);c.v[i]=__half2float(v[i]);}
    }
    const auto expected=attention_test::reference(c);
    Device<half> dq(q.size()),dk(k.size()),dv(v.size()),out(q.size());
    dq.upload(q);dk.upload(k);dv.upload(v);
    // Poison separately for every invocation: no stale passing output can survive.
    checked(cudaMemset(out.data,0xff,q.size()*sizeof(half)));
    if(c.k8.empty()) {
        checked(jllm::flash_attention_decode(out.data,dq.data,dk.data,dv.data,c.heads,c.kv_heads,
                                            c.dim,c.length,c.scale,false,nullptr,0));
        checked(cudaDeviceSynchronize());
    } else {
        Device<int8_t> k8(c.k8.size()),v8(c.v8.size());Device<float> scales(c.scales.size());
        k8.upload(c.k8);v8.upload(c.v8);scales.upload(c.scales);
        checked(jllm::flash_attention_decode(out.data,dq.data,k8.data,v8.data,c.heads,c.kv_heads,
                                            c.dim,c.length,c.scale,true,scales.data,0));
        checked(cudaDeviceSynchronize());
    }
    std::vector<half> actual(q.size());
    checked(cudaMemcpy(actual.data(),out.data,actual.size()*sizeof(half),cudaMemcpyDeviceToHost));
    double max_error=0;
    for(std::size_t i=0;i<actual.size();++i) {
        const double value=__half2float(actual[i]);
        const double error=std::abs(value-expected[i]);
        if(!std::isfinite(value)||error>0.003+0.001*std::abs(expected[i]))
            throw std::runtime_error(c.name+" numerical mismatch at output "+std::to_string(i));
        max_error=std::max(max_error,error);
    }
    std::printf("PASS %s: max_abs_error=%.9g\n",c.name.c_str(),max_error);
}

int main() {
    int count=0;
    const auto status=cudaGetDeviceCount(&count);
    if(status==cudaErrorNoDevice||status==cudaErrorInsufficientDriver||(status==cudaSuccess&&count==0)) {
        std::fprintf(stderr,"SKIP: no usable CUDA device/driver; attention kernel parity not executed\n");return 77;
    }
    try {
        checked(status);checked(cudaSetDevice(0));
        cudaDeviceProp properties{};checked(cudaGetDeviceProperties(&properties,0));
        std::printf("Device %s SM %d.%d\n",properties.name,properties.major,properties.minor);
        for(const auto& c:attention_test::cases())check_case(c);
        Device<half> q(1),k(1),v(1),out(1);
        if(jllm::flash_attention_decode(out.data,q.data,k.data,v.data,1,1,1,1,1,true,nullptr,0)!=cudaErrorInvalidValue)
            throw std::runtime_error("missing INT8 scales accepted");
        if(jllm::flash_attention_decode(out.data,q.data,k.data,v.data,1,1,1,0,1,false,nullptr,0)!=cudaErrorInvalidValue)
            throw std::runtime_error("empty prefix accepted");
        std::puts("attention CUDA parity and launch validation passed");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
