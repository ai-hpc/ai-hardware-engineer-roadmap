#include "jllm_sampling.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void invalid(F function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid input was accepted");
}
int main() {
    using namespace jllm;
    GenParams p; p.temperature = 1; p.top_k = 0; p.top_p = 1; p.repeat_penalty = 1;
    float logits[] = {std::log(0.1f), std::log(0.6f), std::log(0.3f)};
    auto d = token_distribution(logits, 3, p);
    require(d.size() == 3 && d[0].id == 1 && d[1].id == 2, "unfiltered order");
    require(std::abs(d[0].probability - .6) < 1e-7, "reference softmax");
    p.top_p = .5;
    require(token_distribution(logits, 3, p).size() == 1, "top-p must sort with top-k disabled");
    p.top_k = 3;
    require(token_distribution(logits, 3, p)[0].id == 1, "top-k equal to vocab must sort");
    p.top_k = 2; p.top_p = .65f;
    require(token_distribution(logits, 3, p).size() == 1, "top-p must use renormalized top-k mass");
    p.top_p = .8f;
    d = token_distribution(logits, 3, p);
    require(d.size() == 2 && std::abs(d[0].probability - 2.0/3) < 1e-7, "top-k reference mass");
    float positive[] = {4, 3}; int recent[] = {0, 0};
    p.temperature = 0; p.repeat_penalty = 2;
    require(sample_token(positive, 2, p, recent, 2) == 1, "greedy must apply repetition penalty");
    require(positive[0] == 4 && positive[1] == 3, "sampler mutated logits");
    float repeated[] = {4, 1.5f};
    require(sample_token(repeated, 2, p, recent, 2) == 0, "penalty applied more than once");
    float negative[] = {-1, -1.5f};
    require(sample_token(negative, 2, p, recent, 1) == 1, "negative repetition penalty direction");
    p.temperature = 1; p.top_k = 0; p.top_p = 1; p.repeat_penalty = 1;
    float extreme[] = {1e30f, -1e30f, -std::numeric_limits<float>::infinity()};
    d = token_distribution(extreme, 3, p);
    require(d.size() == 2 && d[0].probability == 1 && d[1].probability == 0, "stable extreme logits");
    seed_sampler(42); std::vector<int> first;
    for (int i = 0; i < 100; ++i) first.push_back(sample_token(logits, 3, p));
    seed_sampler(42);
    for (int i = 0; i < 100; ++i) require(first[i] == sample_token(logits, 3, p), "seed replay");
    int counts[3] = {};
    seed_sampler(7);
    for (int i = 0; i < 20000; ++i) ++counts[sample_token(logits, 3, p)];
    require(std::abs(counts[0]/20000.0-.1) < .015 && std::abs(counts[1]/20000.0-.6) < .015
            && std::abs(counts[2]/20000.0-.3) < .015, "distribution frequency");
    invalid([&] { token_distribution(nullptr, 3, p); });
    invalid([&] { token_distribution(logits, 0, p); });
    invalid([&] { token_distribution(logits, 3, p, nullptr, 1); });
    int bad[] = {-1, 3};
    for (int i : bad) invalid([&] { token_distribution(logits, 3, p, &i, 1); });
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        invalid([&] { token_distribution(&value, 1, p); });
    float masked = -std::numeric_limits<float>::infinity();
    invalid([&] { token_distribution(&masked, 1, p); });
    p.top_k = -1; invalid([&] { token_distribution(logits, 3, p); }); p.top_k = 0;
    for (float value : {0.f, -1.f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
        p.top_p = value; invalid([&] { token_distribution(logits, 3, p); });
    }
    p.top_p = 1; p.temperature = -1; invalid([&] { token_distribution(logits, 3, p); });
    p.temperature = 1; p.repeat_penalty = .5; invalid([&] { token_distribution(logits, 3, p); });
    std::cout << "sampling contract checks passed; empirical counts=" << counts[0] << ','
              << counts[1] << ',' << counts[2] << '\n';
}
