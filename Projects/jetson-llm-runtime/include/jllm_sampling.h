#pragma once
#include <vector>

namespace jllm {
struct GenParams {
    int max_tokens = 256;
    float temperature = 0.7f;
    int top_k = 40;  // 0 disables top-k; negative values are rejected
    float top_p = 0.9f;
    float repeat_penalty = 1.1f; // applied once per distinct recent token
    int context_limit = 0;
    bool use_cuda_graph = false; // experimental GPU graph path is not validated
    bool kv_int8 = false;        // INT8 KV needs persistent scale storage
};

struct TokenProbability { int id; double probability; };
// Penalty -> temperature -> top-k -> normalized top-p -> renormalize.
// Temperature zero returns a one-token distribution after the penalty.
// Throws invalid_argument on invalid parameters/logits/IDs; input is unchanged.
std::vector<TokenProbability> token_distribution(
    const float* logits, int vocab_size, const GenParams& params,
    const int* recent_tokens = nullptr, int n_recent = 0);
int sample_token(float* logits, int vocab_size, const GenParams& params,
                 const int* recent_tokens = nullptr, int n_recent = 0);
void seed_sampler(unsigned int seed);
}
