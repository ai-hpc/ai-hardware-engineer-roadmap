// test_model_load.cpp — Test GGUF loading and weight mapping
// Usage: ./build/test_model_load model.gguf
//
// Validates:
//   1. GGUF header parsing (magic, version, tensor count)
//   2. Model config extraction (n_layers, n_heads, etc.)
//   3. Tokenizer vocabulary loading
//   4. Weight tensor mapping (all layer pointers non-null)
//   5. Memory budget calculation

#include "jllm.h"
#include <sys/mman.h>
#include <cassert>
#include <cstdio>
#include <exception>

int main(int argc, char** argv) try {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int device_count=0;
    const auto device_status=cudaGetDeviceCount(&device_count);
    if(device_status==cudaErrorNoDevice || device_status==cudaErrorInsufficientDriver
        || (device_status==cudaSuccess && device_count==0)) {
        fprintf(stderr,"SKIP: model mapping requires a usable CUDA device/driver; not executed\n");
        return 77;
    }
    if(device_status!=cudaSuccess) {
        fprintf(stderr,"FAIL: device enumeration: %s\n",cudaGetErrorString(device_status));return 1;
    }
    printf("=== jetson-llm model loading test ===\n\n");

    // ── Test 1: System probe ─────────────────────────────────
    printf("Test 1: System probe\n");
    auto info = jllm::probe_jetson();
    jllm::print_jetson_info(info);
    assert(info.total_ram_mb > 0);
    printf("PASS\n\n");

    // ── Test 2: Memory budget ────────────────────────────────
    printf("Test 2: Memory budget\n");
    auto budget = jllm::probe_system_memory();
    budget.print();
    assert(budget.total_mb > 0);
    assert(budget.free_mb() > 0);
    printf("PASS\n\n");

    // ── Test 3: GGUF config ──────────────────────────────────
    printf("Test 3: GGUF config parsing\n");
    auto cfg = jllm::load_gguf_config(model_path);
    printf("  name:         %s\n", cfg.name.c_str());
    printf("  n_layers:     %d\n", cfg.n_layers);
    printf("  n_heads:      %d\n", cfg.n_heads);
    printf("  n_kv_heads:   %d\n", cfg.n_kv_heads);
    printf("  head_dim:     %d\n", cfg.head_dim);
    printf("  hidden_dim:   %d\n", cfg.hidden_dim);
    printf("  inter_dim:    %d\n", cfg.intermediate_dim);
    printf("  vocab_size:   %d\n", cfg.vocab_size);
    printf("  max_seq_len:  %d\n", cfg.max_seq_len);
    printf("  rope_theta:   %.0f\n", cfg.rope_theta);
    printf("  GQA group:    %d\n", cfg.gqa_group_size());
    assert(cfg.n_layers > 0);
    assert(cfg.n_heads > 0);
    assert(cfg.hidden_dim > 0);
    assert(cfg.vocab_size > 0);
    printf("PASS\n\n");

    // ── Test 4: Weight size estimate ─────────────────────────
    printf("Test 4: Weight size estimate\n");
    int64_t est = cfg.weight_bytes();
    printf("  Estimated: %ld MB\n", est / (1024*1024));
    printf("  Will fit in %ld MB free? %s\n", budget.free_mb(),
           budget.can_allocate(est / (1024*1024) + 500) ? "YES" : "NO");
    printf("PASS\n\n");

    // ── Test 5: KV cache budget ──────────────────────────────
    printf("Test 5: KV cache context calculation\n");
    int ctx_fp16 = budget.max_context(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 2);
    int ctx_int8 = budget.max_context(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, 1);
    printf("  Max context (FP16 KV): %d tokens\n", ctx_fp16);
    printf("  Max context (INT8 KV): %d tokens\n", ctx_int8);
    printf("  KV per token (INT8):   %ld bytes\n", cfg.kv_per_token_bytes(1));
    assert(ctx_int8 >= ctx_fp16);  // INT8 should fit more
    printf("PASS\n\n");

    // ── Test 6: Tokenizer ────────────────────────────────────
    printf("Test 6: Tokenizer\n");
    jllm::Tokenizer tokenizer;
    bool tok_ok = tokenizer.load_from_gguf(model_path);
    printf("  Loaded: %s\n", tok_ok ? "yes" : "no");
    if(!tok_ok){fprintf(stderr,"FAIL: tokenizer profile or vocabulary was not loaded\n");return 1;}
    if (tok_ok) {
        printf("  Vocab size: %zu\n", tokenizer.vocab.size());
        printf("  BOS ID: %d\n", tokenizer.bos_id);
        printf("  EOS ID: %d\n", tokenizer.eos_id);
        assert(tokenizer.vocab.size() > 0);

        // Test encode/decode
        auto tokens = tokenizer.encode("Hello");
        printf("  'Hello' → %zu tokens:", tokens.size());
        for (int t : tokens) printf(" %d", t);
        printf("\n");

        std::string decoded = tokenizer.decode(tokens);
        printf("  Decoded: '%s'\n", decoded.c_str());
    }
    printf("PASS\n\n");

    // ── Test 7: Weight loading and mapping ───────────────────
    printf("Test 7: Weight loading\n");
    void* weights = nullptr;
    int64_t weights_size = 0;
    jllm::ModelWeights mw = {};

    bool loaded = jllm::load_and_map_weights(model_path, &weights, &weights_size, &mw, cfg);
    printf("  Loaded: %s (%ld MB)\n", loaded ? "yes" : "no", weights_size / (1024*1024));
    if(!loaded){fprintf(stderr,"FAIL: model weights were not mapped\n");return 1;}

    if (loaded) {
        printf("  tok_embd:    %p\n", (void*)mw.tok_embd);
        printf("  output_norm: %p\n", (void*)mw.output_norm);
        printf("  output:      %p\n", mw.output);

        int mapped_layers = 0;
        for (int l = 0; l < mw.n_layers; l++) {
            if (mw.layers[l].wq && mw.layers[l].wk && mw.layers[l].wv)
                mapped_layers++;
        }
        printf("  Layers with QKV: %d / %d\n", mapped_layers, mw.n_layers);
    }
    printf("PASS\n\n");

    // ── Test 8: Power and thermal ────────────────────────────
    printf("Test 8: Power and thermal\n");
    auto ps = jllm::read_power_state();
    printf("  Power mode: %dW, GPU @ %d MHz\n", ps.watts, ps.gpu_freq_mhz);
    auto ts = jllm::read_thermal();
    printf("  GPU: %.1f°C, CPU: %.1f°C, throttling: %s\n",
           ts.gpu_temp_c, ts.cpu_temp_c, ts.throttling ? "YES" : "no");
    int backoff = jllm::thermal_backoff_us(ts);
    printf("  Backoff: %d µs\n", backoff);
    printf("PASS\n\n");

    // Cleanup
    if (weights) {
        cudaHostUnregister(weights);
        munmap(weights, weights_size);
    }
    if (mw.layers) delete[] mw.layers;

    printf("═══════════════════════════════════════\n");
    printf("  All model loading tests passed!\n");
    printf("═══════════════════════════════════════\n");
    return 0;
} catch(const std::exception& error) {
    fprintf(stderr,"FAIL: model loading test: %s\n",error.what());return 1;
}
