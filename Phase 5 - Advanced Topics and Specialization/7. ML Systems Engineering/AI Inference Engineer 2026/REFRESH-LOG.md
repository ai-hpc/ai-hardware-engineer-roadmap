# Refresh Log — AI Inference Engineer 2026

Every meaningful update to this course is dated and listed here so readers know exactly what version of reality the lectures are pinned to. Refresh cadence: 6 months default; 3 months if a major model class or hardware generation drops.

When a lecture is refreshed, update its `## Current as of YYYY-MM` line *and* add an entry here.

---

## 2026-06 — New lecture: Part 2 Lecture 07 (the communication layer)

Added **Part 2 Lecture 07 — "Inside the Communication Layer: NCCL, Custom All-Reduce, and the vLLM Communicator Stack."** How a serving runtime actually moves bytes between GPUs, using vLLM's `distributed/device_communicators/` as the worked example: the three-layer architecture (router → engines → primitives), the small-message latency problem that motivates a custom one-shot/two-shot all-reduce over the NCCL ring, the fused all-reduce+RMSNorm path (FlashInfer), the runtime routing/fallback ladder, and all2all as a forward pointer to Part 3 MoE. Mechanistically explains Lecture 04's "TP=8 decode is comm-bound." Part 2 is now 7 lectures (course total 16 → 17); updated the Part 2 README, course README counts/maps, Lecture 06 footer, and the root roadmap README. Module names pinned to the vLLM 0.22-era lineage and hedged as version-dependent.

---

## 2026-06 — Add FlashInfer (shared kernel engine) + InferenceX (live benchmark reference)

- **FlashInfer** — added to **Part 2 Lecture 05 §2.4** as the shared, JIT-compiled attention + sampling kernel engine that vLLM / SGLang / TensorRT-LLM / MLC-LLM build on (paged/ragged attention, sorting-free sampling, customizable variants). Framed as the "which attention backend" knob (`VLLM_ATTENTION_BACKEND=FLASHINFER`, SGLang `--attention-backend flashinfer`). Source: [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) (MLSys 2025), Apache-2.0, `flashinfer-ai`.
- **SemiAnalysis InferenceX** — added to the course **README currency section** as the recommended *live* cross-stack benchmark reference (tokens/s, perf/$, tokens/MW across H100→GB300 NVL72 / MI355X, vLLM/SGLang/TRT-LLM). Apache-2.0; live dashboard at inferencex.com. The point: lecture numbers are teaching anchors, the dashboard is truth-at-deployment.
- Reviewed the Together AI inference-optimization blog; its concepts (FP8/FP4, speculative decoding, MTP, distillation) are already covered with primary sources, so no blog numbers were imported (per the primary-source rule).
- Reviewed three companion learning resources and added them to the README **Companion resources** block (labeled by nature; no content imported — they are same-domain or broader teaching resources, not primary sources): **mlabonne/llm-course** (free, broad LLM lifecycle; SmoothQuant etc. already covered here), **Vizuara Inference Workshop** on Maven (paid live cohort, same domain), and the **BentoML Inference Optimization handbook** (free vendor guide). Pointers only, per the primary-source rule.

---

## 2026-06 — Add the FP8 block-scaling × tensor-parallel alignment constraint

Added the production footgun where **block-scaled FP8 weights require each tensor-parallel shard dimension to be a whole number of quantization blocks** — so a TP size that leaves `dim / TP` not divisible by the block size fails to load (Qwen 2.5 72B's FFN `29568 = 128 × 231` is not block-128-aligned under TP=2/4/8). New **Part 2 Lecture 03 §5.4** explains the mechanism and the fix-order (re-align TP → pad → coarsen scaling); **Part 2 Lecture 04 §4.3** cross-references it as a hard constraint that can override the cost-based TP choice. This is a durable architectural constraint (not a benchmark figure), surfaced by an 8×H200 Qwen 2.5 72B optimization analysis and consistent with vLLM/TRT-LLM block-FP8 behaviour.

---

## 2026-06 — Correctness fix: Qwen 2.5 72B dimensions

Corrected the Qwen 2.5 72B architecture numbers in **Part 2 Lecture 01** and the **Part 2 README**, plus a stray reference in **Part 3 Lecture 01**. Earlier drafts quoted Qwen 2.5 72B as **12288 hidden / 49152 FFN** — a common secondary-source misquote (12288 is GPT-3's width). The official `config.json` is **8192 hidden / 29568 FFN**, which makes Qwen 2.5 72B *dimensionally near-identical* to Llama 3.3 70B (same 8192 hidden, same GQA 64/8, ~3% wider FFN); the 72B-vs-70B gap is mostly the larger 152K vocab. Lecture 01 §2.1/§4.2 now lead with the correct figures and keep the "derive from the published config, sanity-check against the parameter count" lesson as a cautionary note instead of presenting the wrong numbers as fact. Also fixed DeepSeek V3's dense-FFN width in Part 3 Lecture 01 (18432, not 12288). Surfaced while building the [LLM Inference Visualizer](https://github.com/ai-hpc/llm-inference-viz), which renders the correct shapes.

---

## 2026-06 — Full version refresh (post-CES 2026, post-DeepSeek V4)

Major update spanning every lecture. The course was originally drafted against a January 2026 view of the world; multiple ecosystem-defining releases have shipped since.

### What changed in the ecosystem

- **DeepSeek V4** released 2026-04-24: V4-Pro (1.6T total / 49B active) and V4-Flash (284B total / 13B active), both with 1M-token context window. Adds new hybrid CSA+HCA attention on top of MLA, retains MTP. Legacy `deepseek-chat` / `deepseek-reasoner` endpoints discontinued 2026-07-24.
- **Qwen3-235B-A22B-Instruct-2507** (July 2025 post-training update) is the current canonical of the Qwen3-MoE flagship: same 235B/22B architecture, 256K-token context. Newer Qwen3.5 397B-A17B (2026-02) and Qwen3.7-Max (2026-05) shipped but not yet ecosystem-default.
- **Llama 4** (April 2025): Scout 17B-active/16 experts, Maverick 17B-active/128 experts. Llama 4 Behemoth still not released as of writing.
- **Blackwell Ultra (B300, GB300 NVL72)** shipped in volume January 2026: 288 GB HBM3e, 15 PFLOPS dense FP4 per chip, GB300 NVL72 is 1.5× GB200 NVL72 performance.
- **NVIDIA Vera Rubin** entered full production at CES 2026; available H2 2026. Rubin CPX is a new GPU class specifically for massive-context inference.
- **FlashAttention 4** released March 2026: Blackwell-optimized attention kernel, ~1605 TFLOPs/s on B200 at BF16 (~2.7× Triton, 1.3× cuDNN 9.13). Polynomial-approximation exp() on FMA, ~10× fewer rescaling ops.
- **CUDA Toolkit 13.3** (May 2026) replaces 12.x as the current stable. CUDA 13.0 onward has Blackwell as a first-class target.
- **Transformer Engine 2.x** (mainline as of TE 2.x): FP4 native via MX microscaling, integrated with cuDNN 9 / FA4 paths.

### Pinned versions after this refresh

**Models (anchor pair for Part 2 — dense):**

| Model | Release | Why pinned |
|-------|---------|------------|
| Llama 3.3 70B (Instruct) | 2024-12 | Canonical Western dense 70B-class; flagship dense work moved to MoE but this remains the teaching reference |
| Qwen 2.5 72B (Instruct) | 2024-09 | Canonical Chinese dense 72B with slightly wider FFN, QKV bias |

**Models (anchor pair for Part 3 — MoE):**

| Model | Release | Why pinned |
|-------|---------|------------|
| DeepSeek V3.1 | 2025-08 | The canonical 2025 MoE used as the **teaching anchor** — MLA, MTP, 256+1 experts |
| Qwen3-235B-A22B-Instruct-2507 | 2025-07 update | The current standard Qwen3-MoE — 235B/22B, 256K context |

Note: DeepSeek V4 (2026-04) is now the actual production frontier, but V3.1's architecture is still the cleanest teaching anchor — MLA, MTP, and the EP serving recipe carry over to V4 directly. Each Part 3 lecture cross-references V4 where the math or recipe shifts.

**Runtimes (current as of 2026-06):**

| Runtime | Version | Notes |
|---------|---------|-------|
| vLLM | **0.22.0** (2026-05-29) | V1 engine is default since v0.6; current release cadence is roughly every 2 weeks |
| SGLang | **0.5.12.post1** (2026-05-26) | RadixAttention, full DeepSeek + Qwen3-MoE EP support, production disaggregation |
| TensorRT-LLM | **1.3.0rc16** (2026-05-26) | Now at the v1.x major; FP4 path mature on Blackwell |
| llama.cpp | build **b9444** (2026-05-31) | Rolling release; GGUF + IQ-quants reference for edge |
| MLX | **0.31.2** (2026-04-22) | Apple Silicon reference |
| DeepEP | **v1.2.1** (2025-09-16) | DeepSeek EP communication library |

**Framework / runtime stack:**

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA Toolkit | **13.3** (2026-05) | Blackwell-mature; 12.8 was the introduction |
| Transformer Engine | **2.15** (2026-05) | TE 2.x; FP4 via MX microscaling |
| FlashAttention | **4.0.0.beta15** (2026-05-27) | FA4 is the Blackwell-optimized path |
| cuDNN | 9.x | Hopper + Blackwell attention paths |
| NCCL | **2.30.4-1** (2026-04) | All-reduce / all-to-all primitives |

**Hardware:**

| Hardware | Notes |
|----------|-------|
| NVIDIA H100 (80 GB HBM3) | Hopper baseline; still widely deployed |
| NVIDIA H200 (141 GB HBM3e) | Hopper workhorse for 70B-class dense |
| NVIDIA B200 (192 GB HBM3e, 8 TB/s, 9 PFLOPs FP4) | Blackwell baseline |
| NVIDIA B300 / Blackwell Ultra (288 GB HBM3e, 15 PFLOPs FP4) | Shipped 2026-01; the current MoE-class single chip |
| NVIDIA GB200 NVL72 | 72-GPU NVLink domain, the previous-gen MoE target |
| NVIDIA GB300 NVL72 | 1.5× GB200 NVL72; current 2026 MoE production target |
| NVIDIA Vera Rubin (CPU+GPU superchip) | In production at CES 2026; broad availability H2 2026 — *future-watch* |
| NVIDIA Jetson Thor (AGX) | Edge cross-reference |

### Pricing reference (cloud spot, 2026-06; replicate in your own bench)

| GPU class | $/GPU-hour (typical) |
|-----------|----------------------|
| H100 SXM | ~$2.50 |
| H200 SXM | ~$3.50 |
| B200 SXM | ~$5.50 |
| B300 SXM | ~$6.80 (dedicated) – ~$2.45 (spot) – $12-18 (managed DGX) |

Reference: DeepSeek V4 API pricing — V4-Flash $0.14 input / $0.28 output per MTok; V4-Pro $1.74 input / $3.48 output per MTok.

### Papers and primary sources pinned at this refresh

* DeepSeek V3 technical report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
* DeepSeek V4 preview release — [DeepSeek news 2026-04-24](https://api-docs.deepseek.com/news/news260424)
* Qwen 2.5 technical report — [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
* Qwen3 technical report — [Qwen3 release notes](https://qwenlm.github.io/blog/qwen3/)
* Llama 3.3 model card — [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
* FlashAttention-4 — published March 2026
* PagedAttention (vLLM) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
* SGLang — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
* EAGLE-3 — [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
* AWQ — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
* GPTQ — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
* QuaRot — [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
* SpinQuant — [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
* "The Uniqueness of LLaMA3-70B Series with Per-Channel Quantization" — [arXiv:2408.15301](https://arxiv.org/abs/2408.15301)
* Mooncake (P/D disaggregation) — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
* DistServe — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
* FlashInfer (attention/sampling kernel engine) — [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) (MLSys 2025) · [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
* SemiAnalysis InferenceX (live cross-stack benchmark, Apache-2.0) — [github.com/SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) · [inferencex.com](https://inferencex.com/)

---

## 2026-06 — Initial publication (superseded above)

Course launched. Parts 1, 2, 3 complete. Pinned versions (now superseded): vLLM 0.7.x, SGLang 0.4.x, TRT-LLM 0.18.x, CUDA 12.6+, TE 1.10+, FA3.

---

## Planned refresh checkpoints

* **+3 months (2026-09)** — review for: DeepSeek V4 ecosystem maturity, Llama 4 Behemoth release, Vera Rubin general availability, TRT-LLM 1.x stabilization, vLLM cadence updates.
* **+6 months (2026-12)** — full review including dense-pair (Part 2) re-bench against current vLLM/SGLang, and the Vera Rubin section in Part 3 Lecture 02 upgraded from "future-watch" to actual hardware once it's deployed.
* **As-needed** — landing of DeepSeek V5, Qwen 4, Llama 5, or any FP3 / FP6 hardware path.

---

## How to refresh

When updating a lecture:

1. Update the lecture body.
2. Update its trailing `## Current as of YYYY-MM` line to the new date.
3. Add an entry to this log naming what changed and why.
4. Move superseded benchmark tables into a dated `archive/` subfile inside the relevant part, with a one-line back-pointer.
5. Keep primary-source links live — replace dead links with the closest authoritative survivor (model card → HF mirror; paper → arXiv stable URL; GitHub → release tag).

Drift kills technical credibility faster than missing topics. The refresh log is the proof the course is being maintained.
