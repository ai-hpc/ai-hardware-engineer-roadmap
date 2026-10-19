// jllm_engine.h — LLM inference engine (Jetson-native)

#pragma once

#include "jllm_memory.h"
#include "jllm_sampling.h"
#include "jllm_tokenizer.h"
#include "jllm_model.h"
#include "jllm_jetson.h"
#include "jllm_kernels.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace jllm {

// ── Generation parameters ────────────────────────────────────────────────

using TokenCallback = std::function<void(const char* text, bool is_eos)>;

struct GenStats {
    int     prompt_tokens      = 0;
    int     completion_tokens  = 0;
    float   prompt_ms          = 0;
    float   decode_ms          = 0;
    float   prompt_tok_per_sec = 0;
    float   decode_tok_per_sec = 0;
    int64_t peak_memory_mb     = 0;
    float   peak_thermal_c     = 0;
    int     oom_stops          = 0;
    int     thermal_pauses     = 0;
};

// ── Layer weights (pointers into mmap'd GGUF blob) ───────────────────────

struct LayerWeights {
    const void* wq       = nullptr;
    const void* wk       = nullptr;
    const void* wv       = nullptr;
    const void* wo       = nullptr;
    const void* w_gate   = nullptr;
    const void* w_up     = nullptr;
    const void* w_down   = nullptr;
    const void* rms_attn = nullptr;  // FP32 or FP16 depending on GGUF
    const void* rms_ffn  = nullptr;  // FP32 or FP16 depending on GGUF
    int rms_type = 0;                // 0=F32, 1=F16
    const half* sq       = nullptr;
    const half* sk       = nullptr;
    const half* sv       = nullptr;
    const half* so       = nullptr;
    const half* s_gate   = nullptr;
    const half* s_up     = nullptr;
    const half* s_down   = nullptr;
};

struct ModelWeights {
    const void*     tok_embd    = nullptr;   // F32 or F16
    const void*     output_norm = nullptr;  // F32 or F16
    int             embd_type   = 0;        // 0=F32, 1=F16
    const void*     output      = nullptr;
    const half*     s_output    = nullptr;
    LayerWeights*   layers      = nullptr;
    int             n_layers    = 0;
};

// ── GGUF loaders ─────────────────────────────────────────────────────────

bool          load_gguf_weights(const std::string& path, void** weights, int64_t* size);
bool          load_and_map_weights(const std::string& path, void** blob, int64_t* blob_size,
                                   ModelWeights* mw, const ModelConfig& cfg);
ModelWeights  map_weights(const void* blob, int64_t blob_size, const ModelConfig& cfg);

// ── Engine ───────────────────────────────────────────────────────────────

class Engine {
public:
    Engine();
    ~Engine();

    bool load(const std::string& gguf_path, const GenParams& params);
    void unload();
    GenStats generate(const std::string& prompt, const GenParams& params,
                      TokenCallback token_cb = nullptr);
    void stop();

    bool          is_loaded() const { return loaded_; }
    ModelConfig   config() const { return config_; }
    MemoryBudget  memory() const { return budget_; }
    LiveStats     stats() const;

private:
    bool loaded_ = false;
    bool stop_flag_ = false;

    ModelConfig   config_;
    GenParams     gen_params_;
    MemoryBudget  budget_;
    KVCachePool   kv_cache_;
    ScratchPool   scratch_;

    void*         weights_ = nullptr;
    int64_t       weights_size_ = 0;
    ModelWeights  model_weights_ = {};
    Tokenizer     tokenizer_;

    int           last_token_ = 0;
    std::vector<int> recent_tokens_;
    std::vector<float> host_logits_; // allocated before KV admission, reused per token

    cudaStream_t    stream_ = nullptr;
    cudaGraph_t     decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    bool            graph_captured_ = false;

    void transformer_layer(int layer, int pos, half* x);
    int  decode_step(int pos);
    void build_cuda_graph(int pos);
    bool check_memory_and_thermal(int pos);
};

}  // namespace jllm
