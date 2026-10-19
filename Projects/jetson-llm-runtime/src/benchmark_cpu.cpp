// Component timings. This program does not execute a transformer or a GPU.
#include "jllm_gguf.h"
#include "jllm_sampling.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: benchmark_cpu <GGUF-parser-fixture>\n";
        return 2;
    }
    try {
        const auto document = jllm::read_gguf(argv[1]);
        constexpr int vocab = 32000, batches = 7, parses = 100, draws = 30;
        std::vector<float> logits(vocab);
        for (int i = 0; i < vocab; ++i)
            logits[i] = static_cast<float>(std::sin(i * 0.013) * 3.0 + std::cos(i * 0.007));
        jllm::GenParams params;
        params.repeat_penalty = 1.0f;
        jllm::seed_sampler(7);
        // Warm both components before the timed batches. File data may be cached.
        (void)jllm::read_gguf(argv[1]);
        (void)jllm::sample_token(logits.data(), vocab, params);
        std::vector<double> parse_us, sample_us;
        uint64_t checksum = 0;
        using Clock = std::chrono::steady_clock;
        for (int batch = 0; batch < batches; ++batch) {
            auto start = Clock::now();
            for (int i = 0; i < parses; ++i)
                checksum += jllm::read_gguf(argv[1]).tensors.size();
            parse_us.push_back(std::chrono::duration<double, std::micro>(Clock::now()-start).count()/parses);
            start = Clock::now();
            for (int i = 0; i < draws; ++i)
                checksum += jllm::sample_token(logits.data(), vocab, params);
            sample_us.push_back(std::chrono::duration<double, std::micro>(Clock::now()-start).count()/draws);
        }
        auto values = [](const std::vector<double>& samples) {
            std::cout << '[';
            for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
            std::cout << ']';
        };
        std::cout << std::setprecision(10)
                  << "{\"scope\":\"CPU component microbenchmark; no model inference\","
                  << "\"fixture_bytes\":" << document.file_size << ",\"batches\":" << batches
                  << ",\"parses_per_batch\":" << parses << ",\"draws_per_batch\":" << draws
                  << ",\"vocabulary\":" << vocab
                  << ",\"temperature\":0.7,\"top_k\":40,\"top_p\":0.9,\"seed\":7,"
                  << "\"parse_us_per_operation\":";
        values(parse_us);
        std::cout << ",\"sample_us_per_operation\":";
        values(sample_us);
        std::cout << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
