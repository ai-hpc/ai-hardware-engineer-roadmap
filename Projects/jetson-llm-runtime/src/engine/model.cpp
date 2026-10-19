// Experimental Jetson mapper. The CPU GGUF parser validates the file layout;
// this mapper additionally restricts shapes and encodings to the local kernels.
#include "jllm_engine.h"
#include "jllm_gguf.h"
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace jllm {
bool load_gguf_weights(const std::string& path, void** weights, int64_t* size) {
    if (!weights || !size) return false;
    *weights = nullptr; *size = 0;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat status {};
    if (fstat(fd, &status) || !S_ISREG(status.st_mode) || status.st_size <= 0
        || uint64_t(status.st_size) > std::numeric_limits<std::size_t>::max()) {
        close(fd); return false;
    }
    void* mapped = mmap(nullptr, static_cast<std::size_t>(status.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return false;
    const auto error = cudaHostRegister(mapped, static_cast<std::size_t>(status.st_size),
                                       cudaHostRegisterReadOnly | cudaHostRegisterMapped);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "[gguf] mapped host registration failed: %s\n", cudaGetErrorString(error));
        munmap(mapped, static_cast<std::size_t>(status.st_size)); return false;
    }
    void* device_alias = nullptr;
    if (cudaHostGetDevicePointer(&device_alias, mapped, 0) != cudaSuccess || device_alias != mapped) {
        std::fprintf(stderr, "[gguf] local mapper requires an identical host/device virtual address\n");
        cudaHostUnregister(mapped); munmap(mapped, static_cast<std::size_t>(status.st_size)); return false;
    }
    *weights = mapped; *size = status.st_size;
    return true;
}

ModelWeights map_weights(const void*, int64_t, const ModelConfig&) {
    std::fprintf(stderr, "[gguf] map_weights without validated file metadata is unsupported; use load_and_map_weights\n");
    return {};
}

bool load_and_map_weights(const std::string& path, void** blob, int64_t* blob_size,
                          ModelWeights* output, const ModelConfig& cfg) {
    if (!blob || !blob_size || !output) return false;
    *blob = nullptr; *blob_size = 0; *output = {};
    try {
        if (cfg.architecture != "llama" || cfg.n_layers < 1 || cfg.n_layers > 1024
            || cfg.hidden_dim < 1 || cfg.hidden_dim > 8192 || cfg.hidden_dim % 256
            || cfg.intermediate_dim < 1 || cfg.intermediate_dim > 1048576 || cfg.intermediate_dim % 256
            || cfg.n_heads < 1 || cfg.n_heads > 256 || cfg.n_kv_heads < 1
            || cfg.n_kv_heads > cfg.n_heads || cfg.n_heads % cfg.n_kv_heads
            || cfg.head_dim < 2 || cfg.head_dim > 256 || cfg.head_dim%2 || cfg.head_dim*cfg.n_heads != cfg.hidden_dim
            || cfg.vocab_size < 1 || cfg.vocab_size > 1000000)
            throw std::runtime_error("unsupported/invalid architecture dimensions for experimental mapper");
        const auto document = read_gguf(path);
        std::map<std::string,const GgufTensor*> tensors;
        for (const auto& tensor : document.tensors) tensors.emplace(tensor.name, &tensor);
        auto require = [&](const std::string& name, std::vector<uint64_t> shape, std::vector<uint32_t> types) {
            auto found = tensors.find(name);
            if (found == tensors.end() || found->second->shape != shape)
                throw std::runtime_error("missing or wrong tensor shape: " + name);
            bool supported = false;
            for (uint32_t type : types) supported |= found->second->type == type;
            if (!supported) throw std::runtime_error("unsupported tensor encoding: " + name);
            return found->second;
        };
        const uint64_t h = cfg.hidden_dim, kv = uint64_t(cfg.n_kv_heads)*cfg.head_dim;
        const uint64_t inner = cfg.intermediate_dim, vocabulary = cfg.vocab_size;
        require("token_embd.weight", {h,vocabulary}, {0,1,12});
        require("output_norm.weight", {h}, {0}); // local final-normalization kernel reads F32
        require("output.weight", {h,vocabulary}, {12}); // tied embedding aliasing not implemented
        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            const std::string prefix = "blk."+std::to_string(layer)+".";
            require(prefix+"attn_q.weight", {h,h}, {12});
            require(prefix+"attn_k.weight", {h,kv}, {12});
            require(prefix+"attn_v.weight", {h,kv}, {12});
            require(prefix+"attn_output.weight", {h,h}, {12});
            require(prefix+"ffn_gate.weight", {h,inner}, {12});
            require(prefix+"ffn_up.weight", {h,inner}, {12});
            require(prefix+"ffn_down.weight", {inner,h}, {12});
            const auto* attention_norm = require(prefix+"attn_norm.weight", {h}, {0,1});
            const auto* ffn_norm = require(prefix+"ffn_norm.weight", {h}, {0,1});
            if (attention_norm->type != ffn_norm->type)
                throw std::runtime_error("local layer norm type must match for attention and FFN");
        }
        auto layers = std::make_unique<LayerWeights[]>(cfg.n_layers);
        if (!load_gguf_weights(path, blob, blob_size)) return false;
        if (uint64_t(*blob_size) != document.file_size) {
            cudaHostUnregister(*blob); munmap(*blob, static_cast<std::size_t>(*blob_size));
            *blob = nullptr; *blob_size = 0; return false;
        }
        const auto pointer = [&](const std::string& name) -> const void* {
            return static_cast<const char*>(*blob)+document.data_offset+tensors.at(name)->offset;
        };
        ModelWeights result {};
        result.tok_embd = pointer("token_embd.weight"); result.embd_type = tensors.at("token_embd.weight")->type;
        result.output_norm = pointer("output_norm.weight"); result.output = pointer("output.weight");
        result.n_layers = cfg.n_layers;
        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            const std::string prefix = "blk."+std::to_string(layer)+".";
            auto& weights = layers[layer];
            weights.wq = pointer(prefix+"attn_q.weight"); weights.wk = pointer(prefix+"attn_k.weight");
            weights.wv = pointer(prefix+"attn_v.weight"); weights.wo = pointer(prefix+"attn_output.weight");
            weights.w_gate = pointer(prefix+"ffn_gate.weight"); weights.w_up = pointer(prefix+"ffn_up.weight");
            weights.w_down = pointer(prefix+"ffn_down.weight");
            weights.rms_attn = pointer(prefix+"attn_norm.weight"); weights.rms_ffn = pointer(prefix+"ffn_norm.weight");
            weights.rms_type = tensors.at(prefix+"attn_norm.weight")->type;
        }
        result.layers = layers.release(); *output = result;
        return true;
    } catch (const std::exception& error) {
        if (*blob) { cudaHostUnregister(*blob); munmap(*blob, static_cast<std::size_t>(*blob_size)); }
        *blob = nullptr; *blob_size = 0;
        std::fprintf(stderr, "[gguf mapping] %s\n", error.what()); return false;
    }
}
}
