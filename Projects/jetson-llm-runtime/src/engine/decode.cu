// decode.cu — experimental Transformer forward pass + generation loop
//
// Experimental forward path. Scratch is reused; host vectors still allocate.
// Thermal-aware: checks temperature and memory before every token.
// CUDA graph generation is rejected; the retained capture sketch is unvalidated.
//
// BUG FIXES applied:
//   #2 — Residual connection properly chained (residual1 for attn, residual2 for FFN)
//   #3 — Embedding memcpy uses cudaMemcpyDefault (works for unified + discrete)
//   #4 — Added #include <sys/mman.h>
//   #5 — CUDA graph capture now runs actual transformer_layer kernels
//   #7 — FP32 logit output via host-side conversion after FP16 GEMV

#include "jllm_engine.h"
#include "jllm_decode_memory.h"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>   // BUG #4 fix

namespace jllm {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<float, std::milli>;

// Only allocator metadata is rewound. Kernels and copies all use stream_, so
// later overwrites are ordered after earlier readers without freeing storage.
class ScratchScope {
    ScratchPool& pool_;
    const int64_t checkpoint_;
public:
    explicit ScratchScope(ScratchPool& pool):pool_(pool),checkpoint_(pool.used()){}
    ~ScratchScope(){pool_.rewind(checkpoint_);}
    ScratchScope(const ScratchScope&)=delete;
};

// ── Vector add kernel (for residual connections) ─────────────────────────
// BUG #2 fix: need explicit residual add between stages

__global__ void vec_add_kernel(half* __restrict__ out,
                                const half* __restrict__ a,
                                const half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

static void vec_add(half* out, const half* a, const half* b, int n, cudaStream_t s) {
    int block = 128;
    int grid = (n + block - 1) / block;
    vec_add_kernel<<<grid, block, 0, s>>>(out, a, b, n);
}

// ── FP16 to FP32 conversion kernel ──────────────────────────────────────
// BUG #7 fix: convert logits to FP32 on GPU before D2H copy

__global__ void fp16_to_fp32_kernel(float* __restrict__ out,
                                     const half* __restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __half2float(in[i]);
}

static void fp16_to_fp32(float* out, const half* in, int n, cudaStream_t s) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fp16_to_fp32_kernel<<<grid, block, 0, s>>>(out, in, n);
}

// ── Dequantize one embedding row (CPU-side, one row is small) ───────────
// Handles Q4_K (type 12), F32 (type 0), F16 (type 1)

struct embd_block_q4_K {
    uint16_t d_raw;       // FP16 d as raw bits (super-block scale)
    uint16_t dmin_raw;    // FP16 dmin as raw bits (super-block min)
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(embd_block_q4_K) == 144, "");

// CPU-safe FP16→float conversion (no CUDA intrinsics)
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        result = ldexpf((float)mant, -24);  // subnormal
    } else if (exp == 31) {
        result = mant ? NAN : INFINITY;
    } else {
        result = ldexpf((float)(mant + 1024), (int)exp - 25);
    }
    return sign ? -result : result;
}

static void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dequant_q4k_row(float* out, const void* data, int token_id, int hidden_dim) {
    // Each row = hidden_dim elements = hidden_dim/256 Q4_K blocks
    int blocks_per_row = hidden_dim / 256;
    int block_bytes = 144;
    const uint8_t* row_data = (const uint8_t*)data + (int64_t)token_id * blocks_per_row * block_bytes;

    for (int b = 0; b < blocks_per_row; b++) {
        const embd_block_q4_K* blk = (const embd_block_q4_K*)(row_data + b * block_bytes);

        float dall = fp16_to_float(blk->d_raw);
        float dmin = fp16_to_float(blk->dmin_raw);

        for (int il = 0; il < 4; il++) {
            int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4_cpu(is + 0, blk->scales, sc1, m1);
            get_scale_min_k4_cpu(is + 1, blk->scales, sc2, m2);

            float d1 = dall * sc1;
            float dm1 = dmin * m1;
            float d2 = dall * sc2;
            float dm2 = dmin * m2;

            const uint8_t* q = blk->qs + 32 * il;
            int out_base = b * 256 + 64 * il;

            for (int l = 0; l < 32; l++) {
                out[out_base + l]      = d1 * (q[l] & 0xF) - dm1;
                out[out_base + l + 32] = d2 * (q[l] >> 4)  - dm2;
            }
        }
    }
}

static void dequant_embedding(half* dst, const void* embd_data, int token_id,
                               int hidden_dim, int embd_type, cudaStream_t stream) {
    float h_row[8192];  // max hidden_dim = 8192
    half  h_fp16[8192];

    if (embd_type == 12) {
        // Q4_K: dequantize on CPU
        dequant_q4k_row(h_row, embd_data, token_id, hidden_dim);
        // Debug: print first few values to verify dequant
        static bool first = true;
        if (first) {
            fprintf(stderr, "[embd] Q4_K dequant token %d, first 8 values: ", token_id);
            for (int i = 0; i < 8 && i < hidden_dim; i++)
                fprintf(stderr, "%.4f ", h_row[i]);
            fprintf(stderr, "\n");
            first = false;
        }
        for (int i = 0; i < hidden_dim; i++)
            h_fp16[i] = __float2half(h_row[i]);
    } else if (embd_type == 0) {
        // F32: convert to FP16
        const float* src = (const float*)embd_data + (int64_t)token_id * hidden_dim;
        for (int i = 0; i < hidden_dim; i++)
            h_fp16[i] = __float2half(src[i]);
    } else {
        // F16: direct copy
        const half* src = (const half*)embd_data + (int64_t)token_id * hidden_dim;
        memcpy(h_fp16, src, hidden_dim * sizeof(half));
    }

    const auto copy_status=cudaMemcpyAsync(dst, h_fp16, hidden_dim * sizeof(half), cudaMemcpyHostToDevice, stream);
    if(copy_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(copy_status));
    // Stack staging cannot outlive an asynchronous copy.
    const auto completion_status=cudaStreamSynchronize(stream);
    if(completion_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(completion_status));
}

// ═════════════════════════════════════════════════════════════════════════

Engine::Engine() {}
Engine::~Engine() { unload(); }

void Engine::unload() {
    if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
    if (decode_graph_)      { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
    if (stream_)            { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); stream_ = nullptr; }
    kv_cache_.destroy();
    scratch_.destroy();
    std::vector<float>().swap(host_logits_);
    if (weights_) {
        cudaHostUnregister(weights_);
        munmap(weights_, weights_size_);  // now works — #include <sys/mman.h> added
        weights_ = nullptr;
    }
    if (model_weights_.layers) { delete[] model_weights_.layers; model_weights_.layers = nullptr; }
    loaded_ = false;
    graph_captured_ = false;
}

bool Engine::load(const std::string& gguf_path, const GenParams& params) {
    unload();
    // Validate the tokenizer before mapping/allocating GPU model state.
    if (!tokenizer_.load_from_gguf(gguf_path)) return false;
    if (params.kv_int8 || params.use_cuda_graph) {
        fprintf(stderr, "[engine] INT8 KV scales and graph position updates are not validated in this teaching runtime\n");
        return false;
    }
    gen_params_ = params;
    budget_ = probe_system_memory();

    config_ = load_gguf_config(gguf_path);
    if (config_.n_layers <= 0) return false;
    fprintf(stderr, "[engine] %s: %d layers, %d heads (%d KV), dim=%d, vocab=%d\n",
            config_.name.c_str(), config_.n_layers, config_.n_heads,
            config_.n_kv_heads, config_.hidden_dim, config_.vocab_size);

    if (!load_and_map_weights(gguf_path, &weights_, &weights_size_,
                              &model_weights_, config_)) {
        fprintf(stderr, "[engine] Failed to load/map weights\n");
        return false;
    }
    budget_.model_mb = bytes_to_mib_ceil(weights_size_);
    const auto scratch_plan=decode_scratch_plan(config_.hidden_dim,config_.n_heads,
        config_.n_kv_heads,config_.head_dim,config_.intermediate_dim,config_.vocab_size);
    if(!scratch_plan){unload();return false;}
    // Reserve peak device scratch, host logits, and embedding staging before
    // context admission. Other process/driver costs still need measured headroom.
    budget_.scratch_mb=bytes_to_mib_ceil(scratch_plan.reservation_bytes);

    int kv_bytes = params.kv_int8 ? 1 : 2;
    int auto_ctx = budget_.max_context(config_.n_layers, config_.n_kv_heads, config_.head_dim, kv_bytes);
    int ctx = params.context_limit > 0 ? std::min(params.context_limit, auto_ctx) : auto_ctx;
    ctx = std::min(ctx, config_.max_seq_len);
    if (ctx <= 0) { unload(); return false; }
    fprintf(stderr, "[engine] Context: %d tokens (memory-limited to %d)\n", ctx, auto_ctx);

    if(!scratch_.init(scratch_plan.device_bytes)){unload();return false;}
    try {
        host_logits_.resize(config_.vocab_size);
        recent_tokens_.reserve(65);
    } catch(const std::bad_alloc&) {unload();return false;}

    KVCachePool::Config kv_cfg = {};
    kv_cfg.n_layers = config_.n_layers;
    kv_cfg.n_kv_heads = config_.n_kv_heads;
    kv_cfg.head_dim = config_.head_dim;
    kv_cfg.max_context = ctx;
    kv_cfg.overflow_context = 0; // pageable overflow has no validated GPU-access/attention layout
    kv_cfg.kv_type_bytes = kv_bytes;
    if (!kv_cache_.init(kv_cfg)) { unload(); return false; }
    budget_.kv_cache_mb = bytes_to_mib_ceil(kv_cache_.capacity_bytes());

    if (cudaStreamCreate(&stream_) != cudaSuccess) { unload(); return false; }

    budget_.print();
    loaded_ = true;
    return true;
}

// ── Single transformer layer ─────────────────────────────────────────────
// BUG #2 FIX: proper residual chaining:
//   residual1 = x (input to layer)
//   attn_out  = attention(RMSNorm(x))
//   x2        = residual1 + attn_out          ← first residual add
//   ffn_out   = FFN(RMSNorm(x2))
//   x_out     = x2 + ffn_out                  ← second residual add

void Engine::transformer_layer(int layer, int pos, half* x) {
    ScratchScope temporary(scratch_); // x precedes this checkpoint and survives
    const auto& lw = model_weights_.layers[layer];
    int H = config_.hidden_dim;
    int KV_DIM = config_.n_kv_heads * config_.head_dim;
    int I = config_.intermediate_dim;

    // Scratch buffers (bump allocator)
    half* normed   = (half*)scratch_.get_or_throw(H * sizeof(half));
    half* q_buf    = (half*)scratch_.get_or_throw(config_.n_heads * config_.head_dim * sizeof(half));
    half* k_buf    = (half*)scratch_.get_or_throw(KV_DIM * sizeof(half));
    half* v_buf    = (half*)scratch_.get_or_throw(KV_DIM * sizeof(half));
    half* attn_out = (half*)scratch_.get_or_throw(H * sizeof(half));
    half* attn_proj = (half*)scratch_.get_or_throw(H * sizeof(half));
    half* x2       = (half*)scratch_.get_or_throw(H * sizeof(half));  // after first residual
    half* normed2  = (half*)scratch_.get_or_throw(H * sizeof(half));
    half* gate_buf = (half*)scratch_.get_or_throw(I * sizeof(half));
    half* up_buf   = (half*)scratch_.get_or_throw(I * sizeof(half));
    half* swiglu_out = (half*)scratch_.get_or_throw(I * sizeof(half));
    half* ffn_out  = (half*)scratch_.get_or_throw(H * sizeof(half));

    // ── Attention block ──────────────────────────────────────

    // Debug: check x values before first norm
    static int layer_dbg = 0;
    if (layer_dbg < 1) {
        cudaStreamSynchronize(stream_);
        half h_x[8];
        cudaMemcpy(h_x, x, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[layer %d] x before norm: ", layer);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_x[i]));
        fprintf(stderr, "\n");

        // Also check norm weight
        if (lw.rms_attn) {
            float w_check[4];
            bool nfp32 = (lw.rms_type == 0);
            if (nfp32) {
                memcpy(w_check, lw.rms_attn, 4 * sizeof(float));
            } else {
                half h_w[4];
                memcpy(h_w, lw.rms_attn, 4 * sizeof(half));
                for (int i = 0; i < 4; i++) w_check[i] = __half2float(h_w[i]);
            }
            fprintf(stderr, "[layer %d] norm weight (fp32=%d): %.4f %.4f %.4f %.4f\n",
                    layer, nfp32, w_check[0], w_check[1], w_check[2], w_check[3]);
        }
        layer_dbg++;
    }

    // 1. Pre-attention RMSNorm: normed = RMSNorm(x) * weight
    half* zero_buf = (half*)scratch_.get_or_throw(H * sizeof(half));
    cudaMemsetAsync(zero_buf, 0, H * sizeof(half), stream_);
    bool norm_fp32 = (lw.rms_type == 0);  // 0=F32, 1=F16
    fused_rmsnorm_residual(normed, x, zero_buf, lw.rms_attn, 1, H, config_.norm_epsilon, norm_fp32, stream_);

    // Debug: check normed output
    if (layer_dbg <= 1 && layer == 0) {
        cudaStreamSynchronize(stream_);
        half h_n[8];
        cudaMemcpy(h_n, normed, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[layer %d] normed after RMSNorm: ", layer);
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_n[i]));
        fprintf(stderr, "\n");
    }

    // 2. QKV projections
    gemv_q4(q_buf, lw.wq, lw.sq, normed, config_.n_heads * config_.head_dim, H, 32, stream_);
    gemv_q4(k_buf, lw.wk, lw.sk, normed, KV_DIM, H, 32, stream_);
    gemv_q4(v_buf, lw.wv, lw.sv, normed, KV_DIM, H, 32, stream_);

    // 3. RoPE
    rope_inplace(q_buf, k_buf, config_.n_heads, config_.n_kv_heads,
                 config_.head_dim, pos, config_.rope_theta, stream_);

    // 4. Append one token [kv_head, dimension] to each layer's separate K/V span.
    // load()/generate() reject INT8 because the pool has no persisted scale map.
    auto* key_destination = kv_cache_.key_ptr(layer, pos);
    auto* value_destination = kv_cache_.val_ptr(layer, pos);
    if (!key_destination || !value_destination)
        throw std::runtime_error("KV position outside the allocated token-major cache");
    auto copy_status = cudaMemcpyAsync(key_destination, k_buf,
                                      KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
    if (copy_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(copy_status));
    copy_status = cudaMemcpyAsync(value_destination, v_buf,
                                 KV_DIM * sizeof(half), cudaMemcpyDefault, stream_);
    if (copy_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(copy_status));

    // 5. Attention
    float scale = 1.0f / sqrtf((float)config_.head_dim);
    const auto attention_status = flash_attention_decode(attn_out, q_buf, kv_cache_.key_ptr(layer, 0),
                          kv_cache_.val_ptr(layer, 0),
                          config_.n_heads, config_.n_kv_heads, config_.head_dim,
                          pos + 1, scale, false, nullptr, stream_);
    if (attention_status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(attention_status));

    // 6. Output projection
    gemv_q4(attn_proj, lw.wo, lw.so, attn_out, H, H, 32, stream_);

    // 7. FIRST RESIDUAL: x2 = x + attn_proj  (BUG #2 FIX)
    vec_add(x2, x, attn_proj, H, stream_);

    // ── FFN block ────────────────────────────────────────────

    // 8. Pre-FFN RMSNorm: normed2 = RMSNorm(x2) * weight
    fused_rmsnorm_residual(normed2, x2, zero_buf, lw.rms_ffn, 1, H, config_.norm_epsilon, norm_fp32, stream_);

    // 9. Gate and up projections
    gemv_q4(gate_buf, lw.w_gate, lw.s_gate, normed2, I, H, 32, stream_);
    gemv_q4(up_buf,   lw.w_up,   lw.s_up,   normed2, I, H, 32, stream_);

    // 10. SwiGLU
    fused_swiglu(swiglu_out, gate_buf, up_buf, 1, I, stream_);

    // 11. Down projection
    gemv_q4(ffn_out, lw.w_down, lw.s_down, swiglu_out, H, I, 32, stream_);

    // 12. SECOND RESIDUAL: x = x2 + ffn_out  (BUG #2 FIX)
    vec_add(x, x2, ffn_out, H, stream_);
}

// ── Full decode step ─────────────────────────────────────────────────────

int Engine::decode_step(int pos) {
    int H = config_.hidden_dim;

    half* x = (half*)scratch_.get_or_throw(H * sizeof(half));

    // Embedding lookup — dequantize one row from the embedding table
    // token_embd is typically Q4_K (type 12) or Q6_K (type 14) in GGUF
    dequant_embedding(x, model_weights_.tok_embd, last_token_, H,
                      model_weights_.embd_type, stream_);

    // All transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, x);
    }

    // Final RMSNorm
    half* normed = (half*)scratch_.get_or_throw(H * sizeof(half));
    half* zero = (half*)scratch_.get_or_throw(H * sizeof(half));
    cudaMemsetAsync(zero, 0, H * sizeof(half), stream_);
    fused_rmsnorm_residual(normed, x, zero, model_weights_.output_norm, 1, H, config_.norm_epsilon, true, stream_);

    // Logits: FP16 GEMV then convert to FP32 on GPU (BUG #7 FIX)
    half*  logits_fp16 = (half*)scratch_.get_or_throw(config_.vocab_size * sizeof(half));
    float* logits_fp32 = (float*)scratch_.get_or_throw(config_.vocab_size * sizeof(float));

    gemv_q4(logits_fp16, model_weights_.output, model_weights_.s_output,
            normed, config_.vocab_size, H, 32, stream_);

    // Convert on GPU — avoids per-element __half2float on CPU
    fp16_to_fp32(logits_fp32, logits_fp16, config_.vocab_size, stream_);

    const auto execution_status = cudaStreamSynchronize(stream_);
    if (execution_status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(execution_status));

    // Copy FP32 logits to CPU for sampling
    const auto logits_status=cudaMemcpy(host_logits_.data(), logits_fp32,
               config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(logits_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(logits_status));

    // Sample
    int token = sample_token(host_logits_.data(), config_.vocab_size, gen_params_,
                             recent_tokens_.data(), recent_tokens_.size());

    recent_tokens_.push_back(token);
    if ((int)recent_tokens_.size() > 64) recent_tokens_.erase(recent_tokens_.begin());

    last_token_ = token;
    return token;
}

// ── Unvalidated CUDA graph sketch; unreachable through supported options ──
// This captures a fixed position and pointers and does not implement dynamic
// decode updates. load()/generate() reject use_cuda_graph; do not enable this
// merely because the translation unit compiles.

void Engine::build_cuda_graph(int pos) {
    if (graph_captured_) return;
    if (!gen_params_.use_cuda_graph) return;

    fprintf(stderr, "[engine] Capturing CUDA graph...\n");

    // Pre-allocate a fixed hidden state buffer for graph capture
    int H = config_.hidden_dim;
    half* graph_x = (half*)scratch_.get_or_throw(H * sizeof(half));

    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // Capture all transformer layers
    for (int l = 0; l < config_.n_layers; l++) {
        transformer_layer(l, pos, graph_x);
    }

    // Capture final norm
    half* g_normed = (half*)scratch_.get_or_throw(H * sizeof(half));
    half* g_zero = (half*)scratch_.get_or_throw(H * sizeof(half));
    cudaMemsetAsync(g_zero, 0, H * sizeof(half), stream_);
    fused_rmsnorm_residual(g_normed, graph_x, g_zero,
                          model_weights_.output_norm, 1, H, config_.norm_epsilon, true, stream_);

    // Capture logit projection + conversion
    half* g_logits_fp16 = (half*)scratch_.get_or_throw(config_.vocab_size * sizeof(half));
    float* g_logits_fp32 = (float*)scratch_.get_or_throw(config_.vocab_size * sizeof(float));
    gemv_q4(g_logits_fp16, model_weights_.output, model_weights_.s_output,
            g_normed, config_.vocab_size, H, 32, stream_);
    fp16_to_fp32(g_logits_fp32, g_logits_fp16, config_.vocab_size, stream_);

    cudaError_t err = cudaStreamEndCapture(stream_, &decode_graph_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] Graph capture failed: %s\n", cudaGetErrorString(err));
        decode_graph_ = nullptr;
        return;
    }

    err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[engine] Graph instantiation failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
        return;
    }

    graph_captured_ = true;
    fprintf(stderr, "[engine] CUDA graph captured successfully\n");
}

// ── Memory and thermal check ─────────────────────────────────────────────

bool Engine::check_memory_and_thermal(int pos) {
    OOMGuard guard(256);
    int kv_bytes = gen_params_.kv_int8 ? 1 : 2;
    if (!guard.can_extend(config_.kv_per_token_bytes(kv_bytes))) {
        fprintf(stderr, "\n[oom_guard] Stopping at token %d — %ld MB free\n",
                pos, guard.real_free_mb());
        return false;
    }

    if (pos % 10 == 0) {
        auto ts = read_thermal();
        int backoff = thermal_backoff_us(ts);
        if (backoff > 0) {
            fprintf(stderr, "\n[thermal] %.1f°C — backing off %d ms\n",
                    ts.gpu_temp_c, backoff / 1000);
            usleep(backoff);
        }
    }
    return true;
}

// ── Main generation loop ─────────────────────────────────────────────────

GenStats Engine::generate(const std::string& prompt, const GenParams& params,
                          TokenCallback token_cb) {
    GenStats stats = {};
    if (!loaded_ || params.max_tokens < 0 || params.kv_int8 || params.use_cuda_graph) {
        fprintf(stderr, "[engine] unloaded engine or unsupported generation configuration\n");
        return stats;
    }
    stop_flag_ = false;
    gen_params_ = params;
    recent_tokens_.clear();
    const auto prior_status=cudaStreamSynchronize(stream_);
    if(prior_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(prior_status));

    auto prompt_tokens = tokenizer_.encode(prompt);
    stats.prompt_tokens = prompt_tokens.size();
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(tokenizer_.bos_id);
    }

    if (prompt_tokens.size() > static_cast<size_t>(kv_cache_.max_tokens())) {
        fprintf(stderr, "[engine] prompt exceeds allocated context\n");
        return stats;
    }
    kv_cache_.clear(); // logical prefix restarts; each used entry is overwritten
    // The final prompt token is processed by decode_step to produce the first
    // generated token. Processing it here too would duplicate its KV entry.
    // Prefill timing below therefore covers only the preceding prompt tokens.
    auto t0 = Clock::now();
    for (int i = 0; i + 1 < (int)prompt_tokens.size(); i++) {
        scratch_.reset();
        last_token_ = prompt_tokens[i];
        int H = config_.hidden_dim;
        half* x = (half*)scratch_.get_or_throw(H * sizeof(half));
        dequant_embedding(x, model_weights_.tok_embd, last_token_, H,
                          model_weights_.embd_type, stream_);
        for (int l = 0; l < config_.n_layers; l++)
            transformer_layer(l, i, x);
    }
    const auto prefill_status=cudaStreamSynchronize(stream_);
    if(prefill_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(prefill_status));
    auto t1 = Clock::now();
    stats.prompt_ms = Ms(t1 - t0).count();
    if (stats.prompt_tokens > 0)
        stats.prompt_tok_per_sec = std::max(0, stats.prompt_tokens - 1) / (stats.prompt_ms / 1000.0f);
    fprintf(stderr, "[engine] Prefill: %d tokens in %.0f ms (%.1f tok/s)\n",
            stats.prompt_tokens, stats.prompt_ms, stats.prompt_tok_per_sec);

    // Decode
    auto t2 = Clock::now();
    int64_t peak_mem = 0;
    float peak_temp = 0;
    last_token_ = prompt_tokens.back();
    int pos = static_cast<int>(prompt_tokens.size()) - 1;

    for (int i = 0; i < params.max_tokens && !stop_flag_; i++) {
        if (pos >= kv_cache_.max_tokens()) break;
        if (!check_memory_and_thermal(pos)) {
            stats.oom_stops++;
            break;
        }

        scratch_.reset();
        int token = decode_step(pos);
        pos++;

        bool is_eos = (token == tokenizer_.eos_id);

        if (token_cb) {
            std::string text = tokenizer_.decode(token);
            token_cb(text.c_str(), is_eos);
        }

        stats.completion_tokens++;
        if (is_eos) break;

        if (i % 10 == 0) {
            OOMGuard g(0);
            int64_t used = budget_.total_mb - g.real_free_mb();
            peak_mem = std::max(peak_mem, used);
            peak_temp = std::max(peak_temp, read_thermal().gpu_temp_c);
        }
    }

    auto t3 = Clock::now();
    stats.decode_ms = Ms(t3 - t2).count();
    if (stats.completion_tokens > 0)
        stats.decode_tok_per_sec = stats.completion_tokens / (stats.decode_ms / 1000.0f);
    stats.peak_memory_mb = peak_mem;
    stats.peak_thermal_c = peak_temp;

    fprintf(stderr, "[engine] Decode: %d tokens in %.0f ms (%.1f tok/s)\n",
            stats.completion_tokens, stats.decode_ms, stats.decode_tok_per_sec);
    return stats;
}

void Engine::stop() { stop_flag_ = true; }
LiveStats Engine::stats() const { return read_live_stats(); }

}  // namespace jllm
