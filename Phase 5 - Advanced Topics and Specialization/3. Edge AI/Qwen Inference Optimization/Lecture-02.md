# Lecture 2: Quantizing Qwen3-4B to Q4 — AWQ, GPTQ, K-Quants, and the Bytes-per-Weight Trade

## Overview

Qwen3-4B at BF16 is **~7.6 GB** on disk. An Orin Nano 8 GB has roughly 4 GB of free DRAM after the OS, CUDA context, KV cache, and scratch. The model doesn't fit. Either you shrink it or you don't run it.

This lecture is the **inference-quantization** playbook for Qwen3-4B specifically: which formats work, how Qwen's particular weight statistics shape the choice, where Q4_K_M earned its default-status, and what AWQ and GPTQ do differently that matters at this scale. We do **not** cover training quantization (QAT) — only post-training, weight-only quantization for inference.

By the end you should be able to:

* Compute on-disk and DRAM footprint for any quant choice on Qwen3-4B.
* Pick between Q4_0, Q4_K_M, Q5_K_M, AWQ-4bit, GPTQ-4bit for a target tok/s and quality.
* Run a calibration pass and validate without burning a week on eval.
* Explain why V and FFN-down are upgraded to Q6_K in the K-quant family.

---

## 1. Why Quantize: The Bandwidth Math

A 4B-class model on Orin Nano:

| Format | Bits/weight | Disk | DRAM (weights) | Roofline tok/s @ 50 GB/s |
|---|---|---|---|---|
| FP32 | 32 | 15.3 GB | doesn't fit | — |
| BF16 / FP16 | 16 | 7.6 GB | doesn't fit | — |
| Q8_0 | ~8.5 | 4.1 GB | 4.1 GB (tight) | ~12 |
| Q6_K | ~6.6 | 3.2 GB | 3.2 GB | ~15 |
| Q5_K_M | ~5.6 | 2.7 GB | 2.7 GB | ~18 |
| **Q4_K_M** | **~4.6** | **2.4 GB** | **2.4 GB** | **~21** |
| Q4_0 | ~4.5 | 2.3 GB | 2.3 GB | ~21 |
| Q3_K_M | ~3.9 | 2.1 GB | 2.1 GB | ~24 |
| Q2_K | ~3.0 | 1.7 GB | 1.7 GB | ~29 |
| IQ2_XS | ~2.4 | 1.4 GB | 1.4 GB | ~35 |

The roofline is `bandwidth / bytes_per_token`. Below ~Q4 the quality drops fast for a 4B-class model; above Q5 you're paying bandwidth for diminishing perplexity gains. **Q4_K_M is the sweet spot** — and it's also what JLLM was running in your log.

The math is brutal but clarifying: every bit per weight you remove buys you ~1 tok/s on Orin Nano. That's the entire game.

---

## 2. The Quantization Format Zoo

Three families dominate Qwen3-4B deployment:

### 2.1 ggml K-quants (Q*_K_M family)

Two-level block layout:

```
Superblock: 256 weights
  ├── shared FP16 scale (s_super)
  ├── shared FP16 min   (m_super)
  └── 16 sub-blocks of 16 weights each
        ├── small per-sub-block scale (4–6 bits, packed)
        └── small per-sub-block min  (4–6 bits, packed)
              └── 16 quantized weights (3, 4, 5, or 6 bits each)
```

The "K" stands for **K-means**-style optimal scale selection per sub-block — not a real K-means, but a numerical optimization that minimizes block-level reconstruction error.

`_M` suffix means **mixed precision per matrix**: critical tensors (FFN-down, V projection, the half of attention closest to the residual) get bumped up to a higher bit-rate. The standard Qwen3-4B-Q4_K_M layout — exactly what your JLLM log showed:

```
q=Q4_K  k=Q4_K  v=Q6_K  o=Q4_K  gate=Q4_K  up=Q4_K  down=Q6_K
```

V and FFN-down get Q6_K because empirically those tensors carry the most quality. Quantizing FFN-down to Q4 alone costs ~0.3 perplexity points on Qwen3-4B; quantizing only V costs ~0.15. The bandwidth penalty of Q6 for those two tensors is small (they're a minority of the parameter count), so the mixed recipe wins on both axes.

### 2.2 AWQ — Activation-aware Weight Quantization

AWQ's insight: **the activations have outliers**, not the weights. So when you quantize weights, you should protect the dimensions that get multiplied by large-magnitude activations more carefully.

Algorithm in one paragraph:

1. Collect activation statistics on a calibration set: `|act_i|` per input channel, averaged across a few hundred samples.
2. For each weight matrix `W`, find a per-channel **scale vector** `s` such that the equivalent computation `y = (x / s) · (W · diag(s))` redistributes magnitude from "outlier-sensitive" channels into the scale.
3. Quantize the scaled `W · diag(s)` to 4-bit. The scales travel as FP16 alongside.

Output is a 4-bit weight matrix plus per-channel scales. On Qwen, AWQ-4bit reliably beats both GPTQ-4bit and Q4_K_M on most evals **at 7B+**, with a smaller margin (sometimes a tie) at the 4B scale. The format is more GPU-friendly than ggml K-quants because the kernel is a clean fused-dequant matmul without sub-block bookkeeping.

### 2.3 GPTQ — Optimal Brain Quantization

GPTQ takes a calibration set and quantizes weights **column by column**, updating remaining columns to compensate for quantization error in the column just quantized. The update uses the inverse Hessian of the layer's MSE loss with respect to weights.

GPTQ-4bit gives near-AWQ quality. The format is uglier to kernel-dispatch on (group-wise scales, asymmetric zero points) but enjoys excellent ecosystem support — vLLM, TGI, exllamav2, and Marlin kernels all consume GPTQ natively.

### 2.4 Side-by-side for Qwen3-4B

Assume "Q4 family". Approximate numbers from public Qwen3-4B benchmarks (May 2026, MMLU + IFEval composite, your mileage will vary):

| Format | Effective bpw | MMLU drop vs BF16 | Disk | Best runtime |
|---|---|---|---|---|
| BF16 (reference) | 16 | — | 7.6 GB | vLLM, transformers |
| Q8_0 | 8.5 | < 0.05 | 4.1 GB | llama.cpp |
| Q6_K | 6.6 | < 0.1 | 3.2 GB | llama.cpp |
| Q5_K_M | 5.6 | 0.1–0.2 | 2.7 GB | llama.cpp |
| **Q4_K_M** | **4.6** | **0.3–0.5** | **2.4 GB** | **llama.cpp, JLLM** |
| AWQ-int4 (g128) | 4.25 | 0.2–0.4 | 2.3 GB | vLLM, TRT-LLM, SGLang |
| GPTQ-int4 (g128) | 4.25 | 0.3–0.5 | 2.3 GB | vLLM, exllamav2, Marlin |
| IQ4_XS | 4.25 | 0.4–0.7 | 2.2 GB | llama.cpp |
| Q3_K_M | 3.9 | 0.8–1.2 | 2.1 GB | llama.cpp |
| Q2_K | 3.0 | 2.5–3.5 | 1.7 GB | llama.cpp (usable barely) |

Below Q4, instruction following degrades faster than aggregate metrics suggest — Qwen3-4B at Q2 will technically "answer" but routinely drops parts of multi-step instructions.

---

## 3. Why V and FFN-Down Get Upgraded

The empirical rule: **the tensors whose error compounds along the residual stream are the most quality-sensitive**.

```
attention block:    x ── norm ── QKV ── attn ── O ── ⊕ ──► x'
                                                     │
                              residual goes here ────┘

FFN block:          x' ── norm ── gate/up ── SwiGLU ── down ── ⊕ ──► x''
                                                                │
                                          residual goes here ───┘
```

Error injected at `O` and `down` is added directly to the residual and carried through every subsequent layer. Error at `Q` and `gate` is partially "absorbed" by downstream nonlinearities (softmax, SiLU) — though "absorbed" is generous, it's just less catastrophic.

The Q6_K choice for V comes from a slightly different argument: V is the actual content path of attention. Q and K only determine **where** to look; V is **what** you get. Noisy V poisons the output more directly than noisy Q or K (which are followed by softmax — small perturbations in attention scores are small perturbations in attention weights).

These are now well-known rules in the open-weights community and you see the same pattern (V and FFN-down upgraded) in `*_K_M` quant configurations across Llama, Mistral, Phi, Gemma — and Qwen.

---

## 4. The Practical Workflow

Two paths depending on runtime:

### 4.1 Path A — llama.cpp / K-quants for JLLM, llama.cpp, MLC

```bash
# Starting from HF safetensors:
git lfs clone https://huggingface.co/Qwen/Qwen3-4B-Instruct
cd Qwen3-4B-Instruct

# Step 1: convert to GGUF FP16 (intermediate)
python -m llama_cpp.convert_hf_to_gguf . \
    --outfile qwen3-4b-fp16.gguf \
    --outtype f16

# Step 2: quantize to Q4_K_M
./quantize qwen3-4b-fp16.gguf qwen3-4b-q4_k_m.gguf Q4_K_M

# Optional: with importance matrix for slightly better quality
./imatrix -m qwen3-4b-fp16.gguf -f calibration.txt -o qwen3.imatrix
./quantize --imatrix qwen3.imatrix qwen3-4b-fp16.gguf qwen3-4b-q4_k_m.gguf Q4_K_M
```

The intermediate FP16 GGUF is ~7.6 GB; the final Q4_K_M is ~2.4 GB. Keep the FP16 around if you plan to try other quant levels — re-quantizing is fast (~30 s on a modern CPU); re-converting from HF safetensors is slow and disk-heavy.

### 4.2 Path B — AWQ for vLLM / TRT-LLM / SGLang

```bash
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model_path = 'Qwen/Qwen3-4B-Instruct'
quant_path = './qwen3-4b-awq'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, safetensors=True)

quant_config = {
    'zero_point': True,
    'q_group_size': 128,
    'w_bit': 4,
    'version': 'GEMM',  # 'GEMM' for vLLM; 'GEMV' for inference-only kernels
}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
"
```

Calibration uses 128 samples of pile/c4 by default. For Qwen, a **domain-matched calibration set is meaningfully better** — if your downstream is chat, calibrate on chat-style data; if it's code, calibrate on code. Public Qwen-AWQ releases on HuggingFace typically use a chat-style calibration mix.

The `g128` notation in benchmarks means group size 128 — quantization scales are shared across 128 weights along a row. Smaller groups (32, 64) buy quality at storage cost.

---

## 5. Calibration Set — The 90-Second Decision

The calibration set is the corpus AWQ/GPTQ/imatrix use to measure activation statistics. Rules of thumb:

| Goal | Calibration set |
|---|---|
| General chat assistant | OpenAssistant + ShareGPT mix (~512 samples × 2 k tokens) |
| Code assistant | Stack-V2 subset, languages you care about |
| Multilingual (Chinese + English) | mC4 + Wikipedia-ZH + Wikipedia-EN |
| Long context | Books-3 or fan-fiction sliced to ~8 k each |
| Domain-specific (medical, legal, etc.) | Public corpora in that domain |

Common pitfalls:

* **Too small** (<32 samples) — calibration overfits to handful of activation patterns.
* **Too narrow** — calibrating only on English breaks Chinese performance noticeably.
* **Too short** — samples shorter than ~256 tokens don't exercise long-range attention.
* **Calibrating with `<think>` blocks included** — for Qwen3, decide whether you want the model to be good at thinking-mode or chat-mode and pick the calibration data accordingly. Mixed works but is dominated by the majority.

---

## 6. Validating the Quant

Two cheap checks, one expensive one:

### 6.1 Sanity (60 seconds)

```python
from llama_cpp import Llama

m = Llama(model_path="qwen3-4b-q4_k_m.gguf",
          n_ctx=2048, n_gpu_layers=99)

for prompt in [
    "Write a one-sentence summary of quantum entanglement.",
    "用一句话解释量子纠缠。",
    "def fibonacci(n):",
    "Solve: if x + 3 = 7, what is x?",
]:
    out = m(prompt, max_tokens=64, temperature=0.0)
    print(out["choices"][0]["text"])
```

If the model produces gibberish, repeats itself, switches language mid-response, or refuses on benign prompts — something is broken (often a tokenizer or chat-template bug, sometimes RoPE layout, occasionally a corrupt quant).

### 6.2 Perplexity sweep (10 minutes)

```bash
./perplexity -m qwen3-4b-q4_k_m.gguf -f wikitext-2-raw/wiki.test.raw -c 2048
```

Compare against the FP16 baseline. Q4_K_M typically lands within +0.05 to +0.15 perplexity of FP16 on wiki text. Anything bigger means something is off — usually the calibration was bad or you accidentally quantized the LM head.

### 6.3 Downstream eval (1–6 GPU-hours)

`lm-eval-harness` with a Qwen-friendly suite:

```bash
lm-eval --model gguf \
    --model_args pretrained=qwen3-4b-q4_k_m.gguf \
    --tasks mmlu,ifeval,humaneval,gsm8k \
    --batch_size 1 \
    --num_fewshot 0
```

Targets for Q4_K_M Qwen3-4B vs BF16:

* MMLU: drop < 0.5 pts
* IFEval (instruction following): drop < 1.5 pts (this is the most sensitive metric for Q4)
* GSM8K (math): drop < 2 pts
* HumanEval (code): drop < 1 pt

If IFEval drops > 3 pts, your calibration set was either too small or too narrow — re-run with a broader mix.

---

## 7. Storage Layout — GGUF for Qwen3-4B-Q4_K_M

The GGUF file is a single blob with this structure:

```
┌─────────────────────────────────────┐
│ Magic "GGUF" + version (4 bytes)    │
├─────────────────────────────────────┤
│ Tensor count (8 bytes)              │
│ Metadata KV count (8 bytes)         │
├─────────────────────────────────────┤
│ Metadata KV table                   │
│   - general.name = "Qwen3 4B …"     │
│   - general.architecture = "qwen3"  │
│   - qwen3.context_length = 40960    │
│   - qwen3.embedding_length = 2560   │
│   - qwen3.block_count = 36          │
│   - qwen3.attention.head_count = 32 │
│   - qwen3.attention.head_count_kv=8 │
│   - qwen3.feed_forward_length = 6912│
│   - qwen3.rope.freq_base = 1000000  │
│   - tokenizer.ggml.tokens = [...]   │
│   - tokenizer.ggml.merges = [...]   │
│   - ...                             │
├─────────────────────────────────────┤
│ Tensor info table (name, dtype,     │
│   shape, offset for each tensor)    │
├─────────────────────────────────────┤
│ Padding to alignment                │
├─────────────────────────────────────┤
│ Tensor data — raw quantized blocks  │
│   token_embd (Q6_K block stream)    │
│   blk.0.attn_norm (F32)             │
│   blk.0.attn_q (Q4_K block stream)  │
│   blk.0.attn_q.bias (F32)           │
│   blk.0.attn_k (Q4_K block stream)  │
│   …                                 │
└─────────────────────────────────────┘
```

Two things matter for inference engineering:

1. **All tensor data is contiguous after the header.** This is what makes `mmap` viable. The runtime maps the file once and points the GPU at the offsets it needs.
2. **Block boundaries are aligned.** K-quant blocks are 256 weights = 144 bytes for Q4_K. The runtime can read a single block (one DRAM transaction) and dequant it in registers.

---

## 8. Asymmetric Memory: Stage the Right Tensors in the Right Place

On Orin Nano with unified memory, you don't have to do anything — CPU and GPU see the same DRAM. On a discrete-GPU box (RTX, L40S, etc.), the runtime decides which tensors to copy to VRAM. Heuristic: copy in order of bytes-read-per-decode.

For Qwen3-4B-Q4_K_M, in priority order:

1. **All transformer-block weights** (~2.0 GB) — they're hit on every layer of every decoded token. Must be in VRAM.
2. **Output embedding (tied to token_embd)** (~390 MB at Q6_K) — read once per generated token for the LM head GEMV. Must be in VRAM.
3. **Norm weights** (~600 KB total) — tiny, keep in VRAM; they go through every layer norm.
4. **Tokenizer & metadata** — CPU.
5. **KV cache** — allocated in VRAM, separate from weights.

For an 8 GB GPU this is trivial; you'd never offload. The decision only matters for boxes where VRAM is < ~3 GB (some Jetson Nano configurations, some embedded Tegra).

---

## 9. The Quality-Bandwidth Frontier — Choose Once, Live with It

A decision table tied to deployment goals:

| Deployment | Recommended quant | Why |
|---|---|---|
| Jetson Orin Nano 8 GB chat assistant | Q4_K_M | Sweet spot — 20+ tok/s achievable, < 0.5 MMLU drop |
| Jetson Orin Nano 8 GB code copilot | Q5_K_M or AWQ-int4 | Code is more sensitive to logit precision |
| Jetson Orin NX 16 GB | Q5_K_M or Q6_K | More headroom, bandwidth still the wall |
| Raspberry Pi 5 + AI HAT (Hailo) | Q4_K_M (Hailo path) | Hailo's tooling supports K-quants |
| Discrete GPU (RTX 4090, L40S) for batch=1 | AWQ-int4 or GPTQ-int4 | Better tooling, Marlin kernels |
| Discrete GPU for high batch | AWQ-int4 with FP16 KV | Bandwidth less of a worry, throughput dominated by KV |
| Maximum quality, willing to pay bandwidth | Q6_K | Near-lossless, easy default |
| Aggressive size goal (e.g., browser inference) | IQ4_XS or Q3_K_M | Accept ~1-pt eval drop |

Once you ship a quant choice, the model **becomes that quant** in your users' eyes. Re-quantizing is cheap operationally but creates eval debt — users will notice subtle behavior changes when you swap formats, even at "the same effective bit-rate".

---

## Hands-On Exercises

1. **Build the bytes/token plot.** Take Qwen3-4B and quantize it to Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0 (you'll keep the FP16 GGUF and quantize multiple times). Measure tok/s for each on Orin Nano (run `jetson_clocks` first!). Plot tok/s vs effective bytes/weight. Confirm linearity. Identify the kernel-inefficiency outliers.

2. **AWQ vs K-quant on Orin.** Build llama.cpp with CUDA, run Qwen3-4B-Q4_K_M. Then build vLLM (or MLC-LLM) on the same Orin (works on Orin AGX; tight on Orin Nano), run Qwen3-4B-AWQ. Compare tok/s, peak memory, and the IFEval scores at temperature 0.

3. **Importance-matrix experiment.** Quantize Qwen3-4B-Q4_K_M twice: once with `--imatrix` from a calibration set, once without. Compare wikitext perplexity. Compute the perplexity delta — that's the value of the calibration step at this bit-rate.

4. **FFN-down sensitivity.** Patch llama.cpp's quantization to keep FFN-down at FP16 while quantizing everything else to Q4_K. Repeat with FFN-down at Q4_K and everything else FP16. Measure perplexity for both. Confirm FFN-down hurts more.

5. **Calibration-set ablation.** Run the AWQ pipeline twice on Qwen3-4B: once with the default mixed calibration, once with **only** Chinese text. Test the resulting quants on English MMLU and a Chinese eval (CMMLU). Quantify the bilingual cost of monolingual calibration.

6. **Pick the Q for a real product.** Spec a hypothetical edge product: "$200 BOM, 4 GB DRAM, must do 15 tok/s, must be polite, must handle English+Spanish." Choose a quant. Justify in 200 words referencing the table in §9 and your measured perplexity numbers.

---

## Key Takeaways

| Takeaway | Why it matters |
|---|---|
| Every bit/weight you remove buys ~1 tok/s on Orin Nano | Bandwidth-bound decode is dead linear in bits/weight |
| Q4_K_M is the default for a reason | Best quality/byte at this scale; mixed-precision baked in |
| V and FFN-down deserve more bits than Q/K/O/gate/up | Their error compounds along the residual stream |
| AWQ beats K-quants on tooling, ties on quality at 4B | If you're vLLM/TRT-LLM-bound, just use AWQ |
| Calibration set must match deployment domain | Monolingual calibration silently breaks bilingual quality |
| Validate with at least perplexity + IFEval | Aggregate scores can mask instruction-following regression |
| GGUF tensor data is contiguous; design your loader around `mmap` | Zero-copy, page-cache-friendly, restart-safe |

---

## Resources

* **[ggml quantization formats — Justine Tunney](https://justine.lol/matmul/):** Best read on K-quant kernel design and dequant tricks.
* **[AWQ paper (2023, latest revision 2024)](https://arxiv.org/abs/2306.00978):** Foundational paper, easy to read.
* **[GPTQ paper](https://arxiv.org/abs/2210.17323):** Layer-wise OBQ — slightly older but still relevant.
* **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ):** Reference AWQ implementation with HF integration.
* **[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ):** Reference GPTQ implementation.
* **[llama.cpp quantize tool](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md):** K-quant conversion.
* **[Hugging Face — Qwen3-4B-Instruct-GGUF (community)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-GGUF):** Pre-quantized weights to compare against your local conversion.
* **[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness):** The standard eval framework.
* **[Marlin GPTQ kernel](https://github.com/IST-DASLab/marlin):** Optimized 4-bit GEMM for GPTQ — used by vLLM.
