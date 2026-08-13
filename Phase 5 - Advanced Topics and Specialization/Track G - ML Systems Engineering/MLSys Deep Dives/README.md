# MLSys Deep Dives — The 2026 Machine-Learning-Systems Landscape

<div class="course-identity mlsys" markdown="1">
<div class="course-identity__icon">MLS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 5 · ML Systems Engineering · Special Course</p>
<p class="course-identity__title">The whole modern MLSys stack as one connected system: kernel languages, compilers and runtimes, post-transformer architectures, inference-acceleration algorithms, and the economics that ties them together.</p>
<p class="course-identity__meta">Artifact: an optimization-ladder report for one model + target · Measure: tokens/s, TTFT/TPOT, TOK/$, perf/watt, acceptance length</p>
</div>
</div>

> *The model decides what's possible. The system decides what it costs. In 2026, the system is the product.*

A frontier model is a research result. A frontier model **served at a price someone will pay** is a machine-learning-systems result — and the gap between those two is where this field lives. Between 2023 and 2026 the price of GPT-3.5-class intelligence fell roughly **280×**, and GPT-4-class input dropped **over 99%**. Almost none of that came from a bigger GPU. It came from MLSys: better kernels, smarter compilers, leaner architectures, and decode algorithms that didn't exist three years ago.

This course is a connected tour of that stack as it stands in 2026. Not a survey of disconnected topics — a single argument, told in seven lectures, that the **kernel layer, the compiler layer, the architecture layer, and the inference-algorithm layer are one co-designed system**, and that every win in any of them is a direct, measurable move on tokens-per-second and cost-per-token.

**Layer mapping:** L3–L8 — kernels, codegen, compilers, runtime, scheduling, and the economics on top. This is the MLSys-engineer's whole field of view.

**Role targets:** MLSys Engineer · ML Compiler Engineer · AI Inference Engineer · GPU Kernel Engineer · Model-Systems Co-design Engineer · Edge/Physical-AI Engineer.

**Prerequisites:**

* Phase 5 — ML Systems Engineering — [Guide](../Guide.md) and the [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md) course (the serving-stack companion to this one).
* Phase 5 — Edge AI — [Edge LLM Inference Internals](../../Track%20C%20-%20Edge%20AI/Edge%20LLM%20Inference%20Internals/Lecture-01.md) — the roofline, GEMV-vs-GEMM, and why decode is memory-bound. Every "Measure it" here assumes it.
* Comfort reading Python, CUDA/Triton-flavored kernel code, and model cards. No prior compiler-internals knowledge needed.

**Pairs with:** the [TVM Deep Dives](../TVM%20Deep%20Dives/README.md) course (a deep build-it tour of one compiler) and [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md) (the production serving stack). This course is the *landscape*; those two are *depth probes* into pieces of it.

---

## Why this course is structured the way it is

Most MLSys content is a pile of acronyms. This course threads them onto one spine — **economic value** — because that is what actually decides which technique ships:

```text
   price per token  =  ( energy  +  capital )  /  tokens-per-second
                          └── hardware ──┘        └──── MLSys ────┘

   every kernel, compiler pass, architecture choice, and decode trick
   in this course is a lever on the denominator. that is why it exists.
```

So the arc moves from the metric down through the stack and back up to the deployment edge:

1. **The economics** — the value layer, the metrics, why MLSys *is* the product now.
2. **Kernel languages** — how a fast kernel gets written in 2026 (the tile revolution).
3. **Compilers & runtimes** — how kernels get scheduled, fused, and kept resident at scale.
4. **Architectures, part 1** — how the *model* itself was redesigned to be cheap to serve (SSMs, hybrids).
5. **Architectures, part 2** — the 2026 frontier models read as systems artifacts (MoE, MLA, MTP).
6. **Inference algorithms** — making decode fast without changing the model (speculative decoding, Flash kernels).
7. **The edge & the capstone** — 1000-TOPS hardware, on-device models, and proving the whole ladder with numbers.

---

## Course Map (7 lectures)

<div class="lecture-map" markdown>

| # | Lecture | The thread |
|---|---------|-----------|
| [01](Lecture-01.md) | **MLSys as the economic-value layer** — the inference-cost collapse, the metric stack (tokens/s, TOK/$, TCO/Mtok, perf/watt), why systems work *is* the product | the spine |
| [02](Lecture-02.md) | **The kernel-language explosion** — tiles as the new ISA: Triton, CUTLASS/CuTe/cuTile, ThunderKittens, TileLang, and the ease↔control spectrum | how a kernel is written |
| [03](Lecture-03.md) | **Compilers & runtimes** — TVM (autoscheduling), Mojo/MAX (a real language), TensorRT-LLM (closed vendor), IREE, and TileRT (the megakernel runtime) | how kernels run at scale |
| [04](Lecture-04.md) | **Beyond the dense transformer** — Mamba/SSMs, linear attention, and the hybrid wave: Nemotron-H, Jamba, Falcon-H1, MiniMax-01 | the model made cheap |
| [05](Lecture-05.md) | **The 2026 frontier as systems artifacts** — Qwen3, Llama Nemotron Ultra 253B, Xiaomi MiMo, DeepSeek V3/R1, and the MoE + MLA + MTP stack | reading a model card |
| [06](Lecture-06.md) | **Making decode fast** — speculative decoding (EAGLE-3, Medusa, Sequoia), DFlash, Flash kernels (FA-3, FlashInfer), and the Together AI research line | the algorithm layer |
| [07](Lecture-07.md) | **The edge & physical-AI frontier** — 1000-TOPS hardware (Jetson/DRIVE Thor) vs the 1000-tok/s milestone (MiMo + TileRT), on-device models, and the capstone | closing the co-design loop |

</div>

---

## Course Outcomes

By the end you should be able to:

* Read any 2026 model card or serving config and predict its **cost shape** — tokens/s regime, KV-cache behavior, dense vs sparse compute, dominant precision, draft mechanism.
* Predict any model's batch-1 decode speed from a datasheet with the **bandwidth ceiling** (`tokens/s ≤ HBM GB/s ÷ bytes per token`), and use it to sanity-check any vendor throughput claim.
* Place any kernel tool (Triton, CUTLASS/cuTile, ThunderKittens, TileLang, TVM, Mojo, TensorRT) on the **ease↔control spectrum** and pick one for a workload with a defensible reason.
* Explain why **SSM/hybrid architectures and MLA** kill KV-cache growth, why **MoE** decouples capacity from compute, and what each costs you in memory and interconnect.
* Stand up **speculative decoding** (and explain EAGLE-3 vs DFlash vs Medusa), and read the acceptance-length / speedup tradeoff.
* Tie every optimization — kernel, compiler, architecture, decode algorithm — back to a **TOK/$ and tokens/s/watt** number, and defend it.

---

## Currency / Refresh Discipline

This field moves weekly, and several topics here (DFlash, the MiMo + TileRT 1000-tok/s milestone, CuTe DSL / cuTile, Falcon-H1R, Qwen3) are 2025–2026 developments. So this course follows the same discipline as [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md):

* Every lecture closes with a **`## Current as of`** date and the specific versions / claims it pinned.
* **Established research** (Mamba, FlashAttention-3, EAGLE-3, DeepSeek MLA) is stated plainly; **very recent or vendor-reported** numbers (MiMo's ~1200 tok/s, Together Inference Engine benchmarks) are explicitly flagged as such, with the source.
* Treat printed throughput numbers as **teaching anchors**, not truth-at-deployment — software-stack gains move them monthly. For live cross-stack numbers, use a continuously-updated public benchmark such as **[SemiAnalysis InferenceMAX / InferenceX](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)**.

---

## Exit Criteria

You are done with this course when you can:

* Draw the `price = (energy + capital) / tokens-per-second` diagram from memory and place every lecture's topic on it as a lever.
* Take one model and walk it down an **optimization ladder** — baseline → quantize → speculative decode → tuned kernels/compiler — measuring tokens/s and TOK/$ at each rung, and explain which roofline bound each rung moved.
* Argue, with a straight face, why a hybrid-architecture 7B on a 1000-TOPS edge box and a 1T-param MoE at 1000 tok/s in a datacenter are **the same MLSys discipline** pointed at two budgets.

If you can name the techniques but can't connect any of them to a cost number, you have flashcards. The point of this course is the connection.

---

*Related: [TVM Deep Dives](../TVM%20Deep%20Dives/README.md) · [AI Inference Engineer 2026](../AI%20Inference%20Engineer%202026/README.md) · [Phase 5 — ML Systems Engineering Guide](../Guide.md)*
