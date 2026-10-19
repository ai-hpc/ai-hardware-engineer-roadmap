#pragma once
#include <string>
#include <unordered_map>
#include <vector>
namespace jllm {
// Explicit toy greedy-byte profile, not BPE, SentencePiece or a Qwen tokenizer.
struct Tokenizer {
    std::vector<std::string> vocab;
    int bos_id = -1, eos_id = -1;
    bool load_from_gguf(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& ids) const;
private:
    std::unordered_map<std::string,int> token_to_id_;
    std::size_t max_token_len_ = 0;
    bool add_bos_ = false;
};
}
