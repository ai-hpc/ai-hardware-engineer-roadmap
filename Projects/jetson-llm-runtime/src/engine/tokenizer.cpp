// A labeled greedy-byte teaching profile. Unsupported reference tokenizers fail.
#include "jllm_tokenizer.h"
#include "jllm_gguf.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace jllm {
bool Tokenizer::load_from_gguf(const std::string& path) {
    *this = Tokenizer{};
    try {
        const auto document = read_gguf(path);
        const auto& metadata = document.metadata;
        const auto& model = metadata.at("tokenizer.ggml.model");
        if (model.type != 8 || model.text != "course.greedy.bytes")
            throw std::runtime_error("unsupported tokenizer: only course.greedy.bytes is implemented; no pretrained BPE parity");
        const auto& tokens = metadata.at("tokenizer.ggml.tokens");
        if (tokens.type != 9 || tokens.element_type != 8 || tokens.strings.empty())
            throw std::runtime_error("invalid tokenizer vocabulary");
        Tokenizer candidate; candidate.vocab = tokens.strings;
        for (std::size_t i = 0; i < candidate.vocab.size(); ++i) {
            const auto& token = candidate.vocab[i];
            if (token.empty() || !candidate.token_to_id_.emplace(token, static_cast<int>(i)).second)
                throw std::runtime_error("empty or duplicate vocabulary token");
            candidate.max_token_len_ = std::max(candidate.max_token_len_, token.size());
        }
        const auto special = [&](const char* key) {
            const auto& item = metadata.at(key);
            if (item.type != 4 || item.integer >= candidate.vocab.size())
                throw std::runtime_error("special token ID outside vocabulary");
            return static_cast<int>(item.integer);
        };
        candidate.bos_id = special("tokenizer.ggml.bos_token_id");
        candidate.eos_id = special("tokenizer.ggml.eos_token_id");
        if (candidate.bos_id == candidate.eos_id) throw std::runtime_error("toy profile requires distinct BOS/EOS");
        const auto& add_bos = metadata.at("tokenizer.ggml.add_bos_token");
        if (add_bos.type != 7) throw std::runtime_error("add_bos_token must be bool");
        candidate.add_bos_ = add_bos.integer != 0;
        *this = std::move(candidate);
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[tokenizer] %s\n", error.what());
        return false;
    }
}
std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (vocab.empty()) throw std::runtime_error("tokenizer not loaded");
    std::vector<int> tokens;
    if (add_bos_) tokens.push_back(bos_id);
    for (std::size_t pos = 0; pos < text.size();) {
        int id = -1; std::size_t matched = 0;
        for (std::size_t length = std::min(max_token_len_, text.size()-pos); length; --length) {
            const auto found = token_to_id_.find(text.substr(pos,length));
            if (found != token_to_id_.end() && found->second != bos_id && found->second != eos_id
                && !(found->first.size() == 6 && found->first.substr(0,3) == "<0x" && found->first[5] == '>')) {
                id = found->second; matched = length; break;
            }
        }
        if (id < 0) {
            char byte_token[7];
            std::snprintf(byte_token, sizeof byte_token, "<0x%02X>", static_cast<unsigned char>(text[pos]));
            const auto found = token_to_id_.find(byte_token);
            if (found == token_to_id_.end() || found->second == bos_id || found->second == eos_id)
                throw std::runtime_error("input byte absent from teaching vocabulary");
            id = found->second; matched = 1;
        }
        tokens.push_back(id); pos += matched;
    }
    return tokens;
}
std::string Tokenizer::decode(int token_id) const {
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab.size())
        throw std::invalid_argument("token ID outside vocabulary");
    const auto& token = vocab[token_id];
    auto hex = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch-'0';
        if (ch >= 'A' && ch <= 'F') return ch-'A'+10;
        if (ch >= 'a' && ch <= 'f') return ch-'a'+10;
        return -1;
    };
    if (token.size() == 6 && token.substr(0,3) == "<0x" && token[5] == '>'
        && hex(token[3]) >= 0 && hex(token[4]) >= 0)
        return std::string(1, static_cast<char>(16*hex(token[3])+hex(token[4])));
    return token;
}
std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) if (id != bos_id && id != eos_id) result += decode(id);
    return result;
}
}
