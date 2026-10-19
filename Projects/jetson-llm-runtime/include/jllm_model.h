#pragma once
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
namespace jllm {
struct ModelConfig {
    std::string name, architecture;
    int n_layers = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, hidden_dim = 0;
    int intermediate_dim = 0, vocab_size = 0, max_seq_len = 0;
    float rope_theta = 0, norm_epsilon = 0;
    int quant_type = -1; // mixed tensor encodings are not a single model-wide type
    int64_t tensor_storage_bytes = 0;
    int gqa_group_size() const {
        return n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0 ? n_heads/n_kv_heads : 0;
    }
    int64_t weight_bytes() const { return tensor_storage_bytes; } // exact parsed payload sum, excludes padding
    int64_t kv_per_token_bytes(int kv_type_bytes) const {
        if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0
            || kv_type_bytes <= 0 || kv_type_bytes > 8) return 0;
        int64_t bytes = 2;
        for (int value : {n_layers, n_kv_heads, head_dim, kv_type_bytes}) {
            if (bytes > std::numeric_limits<int64_t>::max()/value) return 0;
            bytes *= value;
        }
        return bytes;
    }
};
ModelConfig load_gguf_config(const std::string& path);
}
