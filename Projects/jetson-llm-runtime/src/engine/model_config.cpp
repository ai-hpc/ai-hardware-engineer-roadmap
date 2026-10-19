#include "jllm_model.h"
#include "jllm_gguf.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
namespace jllm {
ModelConfig load_gguf_config(const std::string& path) {
    try {
        const auto document = read_gguf(path);
        const auto& metadata = document.metadata;
        ModelConfig cfg;
        const auto& arch = metadata.at("general.architecture");
        if (arch.type != 8 || arch.text.empty()) throw std::runtime_error("missing architecture string");
        cfg.architecture = arch.text;
        const std::string prefix = arch.text + ".";
        auto positive = [&](const std::string& key, uint64_t limit) {
            const auto& item = metadata.at(key);
            if (item.type != 4 || !item.integer || item.integer > limit)
                throw std::runtime_error("invalid model dimension: " + key);
            return static_cast<int>(item.integer);
        };
        auto real = [&](const std::string& key, float fallback) {
            auto found = metadata.find(key);
            if (found == metadata.end()) return fallback;
            if (found->second.type != 6 || !std::isfinite(found->second.real) || found->second.real <= 0)
                throw std::runtime_error("invalid positive model scalar: " + key);
            return static_cast<float>(found->second.real);
        };
        cfg.name = path;
        auto name = metadata.find("general.name");
        if (name != metadata.end()) {
            if (name->second.type != 8) throw std::runtime_error("general.name must be string");
            cfg.name = name->second.text;
        }
        cfg.n_layers = positive(prefix+"block_count", 1024);
        cfg.n_heads = positive(prefix+"attention.head_count", 256);
        cfg.n_kv_heads = metadata.count(prefix+"attention.head_count_kv")
            ? positive(prefix+"attention.head_count_kv", 256) : cfg.n_heads;
        if (cfg.n_heads % cfg.n_kv_heads) throw std::runtime_error("head count must be divisible by KV head count");
        cfg.hidden_dim = positive(prefix+"embedding_length", 65536);
        cfg.intermediate_dim = positive(prefix+"feed_forward_length", 1048576);
        cfg.max_seq_len = positive(prefix+"context_length", 16777216);
        if (metadata.count(prefix+"attention.key_length"))
            cfg.head_dim = positive(prefix+"attention.key_length", 4096);
        else {
            if (cfg.hidden_dim % cfg.n_heads) throw std::runtime_error("embedding/head ratio is not integral");
            cfg.head_dim = cfg.hidden_dim/cfg.n_heads;
        }
        if (metadata.count(prefix+"attention.value_length")
            && positive(prefix+"attention.value_length", 4096) != cfg.head_dim)
            throw std::runtime_error("different key/value dimensions unsupported by teaching config");
        cfg.rope_theta = real(prefix+"rope.freq_base", 10000.f);
        cfg.norm_epsilon = real(prefix+"attention.layer_norm_rms_epsilon", 1e-5f);
        const auto& vocabulary = metadata.at("tokenizer.ggml.tokens");
        if (vocabulary.type != 9 || vocabulary.element_type != 8 || vocabulary.count == 0)
            throw std::runtime_error("missing string-array vocabulary");
        cfg.vocab_size = static_cast<int>(vocabulary.count);
        if (metadata.count(prefix+"vocab_size") && positive(prefix+"vocab_size", 1000000) != cfg.vocab_size)
            throw std::runtime_error("vocabulary count mismatch");
        bool embedding = false;
        for (const auto& tensor : document.tensors) {
            if (tensor.bytes > uint64_t(std::numeric_limits<int64_t>::max()-cfg.tensor_storage_bytes))
                throw std::runtime_error("model tensor size overflow");
            cfg.tensor_storage_bytes += static_cast<int64_t>(tensor.bytes);
            if (tensor.name == "token_embd.weight") {
                embedding = true;
                if (tensor.shape != std::vector<uint64_t>{uint64_t(cfg.hidden_dim), uint64_t(cfg.vocab_size)})
                    throw std::runtime_error("embedding/vocabulary shape mismatch");
            }
        }
        if (!embedding) throw std::runtime_error("embedding tensor missing");
        return cfg;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gguf config] %s\n", error.what());
        return {};
    }
}
}
