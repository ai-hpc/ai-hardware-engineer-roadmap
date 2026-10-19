#include "jllm_tokenizer.h"
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    if (argc != 3) return 64;
    jllm::Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(argv[1])) return 2;
    try {
        const auto ids = tokenizer.encode(argv[2]);
        for (int id : ids) std::cout << id << ' ';
        std::cout << '\n';
        if (tokenizer.decode(ids) != argv[2]) throw std::runtime_error("round-trip mismatch");
        // A failed reload must clear the old model state.
        if (tokenizer.load_from_gguf("/nonexistent/course-fixture.gguf") || !tokenizer.vocab.empty())
            throw std::runtime_error("stale state after failed load");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 3; }
}
