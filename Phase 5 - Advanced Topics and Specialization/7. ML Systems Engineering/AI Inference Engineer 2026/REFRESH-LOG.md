# Refresh Log — AI Inference Engineer 2026

Every meaningful update to this course is dated and listed here so readers know exactly what version of reality the lectures are pinned to. Refresh cadence: 6 months default; 3 months if a major model class or hardware generation drops.

When a lecture is refreshed, update its `## Current as of YYYY-MM` line *and* add an entry here.

---

## 2026-06 — Initial publication

Course launched. Parts 1 and 2 complete; Part 3 outlined.

### Pinned versions at launch

**Models (anchor pair for Part 2):**

| Model | Release | Why pinned |
|-------|---------|------------|
| Llama 3.3 70B (Instruct) | 2024-12 | The canonical Western dense 70B-class workhorse |
| Qwen 2.5 72B (Instruct) | 2024-09 | The canonical Chinese dense 72B with wider FFN and QKV bias |

**Models (anchor pair planned for Part 3):**

| Model | Release | Why pinned |
|-------|---------|------------|
| DeepSeek V3.1 | 2025-08 | 671B total / 37B active, MLA attention, native MTP — the canonical 2025 MoE |
| Qwen3-MoE 235B-A22B | 2025-04 | 235B total / 22B active, large-vocab MoE — modern non-MLA MoE for contrast |

**Runtimes:**

| Runtime | Version | Notes |
|---------|---------|-------|
| vLLM | 0.7.x (V1 engine) | V1 is the active engine; V0 references kept as archaeology only |
| SGLang | 0.4.x | RadixAttention, EP support for DeepSeek-class MoE |
| TensorRT-LLM | 0.18.x | FP8 mature; FP4 Blackwell path landing |
| llama.cpp | post-2026-04 | GGUF / IQ-quants reference for edge |
| MLX | 0.21.x | Apple Silicon reference |

**Hardware:**

| Hardware | Notes |
|----------|-------|
| NVIDIA H100 (80 GB HBM3) | Hopper baseline |
| NVIDIA H200 (141 GB HBM3e) | Hopper workhorse for 70B-class |
| NVIDIA B200 (192 GB HBM3e) | Blackwell, Transformer Engine 2, FP4 native |
| NVIDIA GB200 NVL72 | 72-GPU NVLink domain, the MoE serving target |
| NVIDIA Jetson Thor (AGX) | Edge cross-reference |

**Papers and primary sources pinned at launch:**

* DeepSeek V3 technical report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
* Qwen 2.5 technical report — [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
* Qwen3 technical report — [Qwen3 release notes](https://qwenlm.github.io/blog/qwen3/)
* Llama 3.3 model card — [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
* FlashAttention-3 — [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
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

---

## Planned refresh checkpoints

* **+3 months** — review for: new DeepSeek release (V4?), Llama 5, vLLM V1 stabilization, TensorRT-LLM FP4 maturity, B300 availability.
* **+6 months** — full review: re-bench Part 2 anchors against current vLLM / SGLang, refresh quantization recipes if AWQ / QuaRot superseded.
* **As-needed** — landing of Vera Rubin (post-Blackwell), DeepSeek V4, Llama 5, Qwen 4, or any FP6 / FP3 hardware path.

---

## How to refresh

When updating a lecture:

1. Update the lecture body.
2. Update its trailing `## Current as of YYYY-MM` line to the new date.
3. Add an entry to this log naming what changed and why.
4. Move superseded benchmark tables into a dated `archive/` subfile inside the relevant part, with a one-line back-pointer.
5. Keep primary-source links live — replace dead links with the closest authoritative survivor (model card → HF mirror; paper → arXiv stable URL; GitHub → release tag).

Drift kills technical credibility faster than missing topics. The refresh log is the proof the course is being maintained.
