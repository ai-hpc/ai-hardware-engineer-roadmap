// CPU sampler with an inspectable distribution and explicit filtering order.
#include "jllm_sampling.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace jllm {
static thread_local std::mt19937 rng(42);
void seed_sampler(unsigned int seed) { rng.seed(seed); }

std::vector<TokenProbability> token_distribution(
    const float* logits, int vocab_size, const GenParams& params,
    const int* recent_tokens, int n_recent) {
    if (!logits || vocab_size <= 0 || n_recent < 0 || (n_recent && !recent_tokens)
        || !std::isfinite(params.temperature) || params.temperature < 0
        || params.top_k < 0 || !std::isfinite(params.top_p)
        || params.top_p <= 0 || params.top_p > 1
        || !std::isfinite(params.repeat_penalty) || params.repeat_penalty < 1)
        throw std::invalid_argument("invalid sampling input or parameters");
    std::vector<double> adjusted(logits, logits + vocab_size);
    bool has_finite = false;
    for (double value : adjusted) {
        if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
            throw std::invalid_argument("NaN/+infinity logits are unsupported");
        has_finite = has_finite || std::isfinite(value);
    }
    if (!has_finite) throw std::invalid_argument("all logits are masked");
    std::unordered_set<int> seen;
    for (int i = 0; i < n_recent; ++i) {
        const int id = recent_tokens[i];
        if (id < 0 || id >= vocab_size) throw std::invalid_argument("recent token out of range");
        if (seen.insert(id).second)
            adjusted[id] = adjusted[id] > 0 ? adjusted[id] / params.repeat_penalty
                                           : adjusted[id] * params.repeat_penalty;
    }
    std::vector<TokenProbability> candidates;
    for (int i = 0; i < vocab_size; ++i)
        if (std::isfinite(adjusted[i])) candidates.push_back({i, adjusted[i]});
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.probability == b.probability ? a.id < b.id : a.probability > b.probability;
    });
    if (params.temperature == 0) return {{candidates.front().id, 1.0}};
    if (params.top_k > 0 && params.top_k < static_cast<int>(candidates.size()))
        candidates.resize(params.top_k);
    const double maximum = candidates.front().probability;
    double total = 0;
    for (auto& candidate : candidates) {
        candidate.probability = std::exp((candidate.probability - maximum) / params.temperature);
        total += candidate.probability;
    }
    for (auto& candidate : candidates) candidate.probability /= total;
    if (params.top_p < 1) {
        double cumulative = 0;
        std::size_t keep = 0;
        do { cumulative += candidates[keep++].probability; }
        while (keep < candidates.size() && cumulative < params.top_p);
        candidates.resize(keep);
        for (auto& candidate : candidates) candidate.probability /= cumulative;
    }
    return candidates;
}

int sample_token(float* logits, int vocab_size, const GenParams& params,
                 const int* recent_tokens, int n_recent) {
    const auto candidates = token_distribution(logits, vocab_size, params, recent_tokens, n_recent);
    if (candidates.size() == 1) return candidates.front().id;
    std::uniform_real_distribution<double> uniform(0, 1);
    const double value = uniform(rng);
    double cumulative = 0;
    for (const auto& candidate : candidates) {
        cumulative += candidate.probability;
        if (value < cumulative) return candidate.id;
    }
    return candidates.back().id; // final rounding interval
}
}
