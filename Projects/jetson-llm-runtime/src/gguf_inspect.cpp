#include "jllm_gguf.h"
#include "jllm_model.h"
#include <string>
#include <exception>
#include <iostream>
int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--config") {
        const auto cfg = jllm::load_gguf_config(argv[2]);
        if (!cfg.n_layers) return 2;
        std::cout << "layers=" << cfg.n_layers << " heads=" << cfg.n_heads << " kv_heads=" << cfg.n_kv_heads
                  << " head_dim=" << cfg.head_dim << " hidden=" << cfg.hidden_dim << " vocab=" << cfg.vocab_size
                  << " tensor_bytes=" << cfg.weight_bytes() << '\n';
        return 0;
    }
    if (argc != 2) { std::cerr << "usage: gguf-inspect FILE\n"; return 64; }
    try {
        const auto document = jllm::read_gguf(argv[1]);
        std::cout << "metadata=" << document.metadata.size() << " tensors=" << document.tensors.size()
                  << " alignment=" << document.alignment << " data_offset=" << document.data_offset << '\n';
        for (const auto& tensor : document.tensors)
            std::cout << "tensor_type=" << tensor.type << " bytes=" << tensor.bytes << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
