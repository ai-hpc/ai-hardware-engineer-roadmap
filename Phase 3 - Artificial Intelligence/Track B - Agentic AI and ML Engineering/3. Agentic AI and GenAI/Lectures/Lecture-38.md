# Lecture 38 - AutoSP: Compiler-Generated Sequence Parallelism for Long-Context Training

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 37](Lecture-37.md) | **Next:** [Lecture 39](Lecture-39.md)

---

**Long-context agents** create long-context model requirements.

**Serving** long contexts is one problem.

**Training** models to handle long contexts is another.

Lecture 36 covered vLLM FP8 KV-cache for long-context inference. This lecture covers the training-side complement: AutoSP.

**AutoSP** is a compiler-based system that automatically transforms ordinary transformer training code into **sequence-parallel** training code across multiple GPUs. The goal is direct:

```text
write standard transformer training code
  -> compile with AutoSP
  -> train longer contexts across GPUs
  -> avoid hand-writing sequence-parallel plumbing
```

The important idea is not just "sequence parallelism exists."

The important idea is:

```text
distributed training strategies are moving into compiler passes
```

That has consequences for **model scientists, systems engineers, and hardware engineers**.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why long-context training OOMs even with conventional data-parallel scaling.
2. Understand sequence parallelism as sharding the token dimension across GPUs.
3. Explain what AutoSP automates compared with hand-written sequence parallelism.
4. Understand how AutoSP integrates with DeepSpeed and DeepCompile.
5. Describe why AutoSP targets DeepSpeed-Ulysses and where that strategy is limited.
6. Explain sequence-aware activation checkpointing and why long-context training changes checkpointing tradeoffs.
7. Understand how AutoSP composes with ZeRO 0/1 and why ZeRO/FSDP alone are not sufficient.
8. Identify AutoSP's limitations around graph breaks and whole-model compilation.
9. Design a benchmark and trace plan for validating AutoSP on real long-context workloads.

---

## 1. Why long-context training runs out of memory

Training a transformer stores much more than model weights.

It stores:

- parameters
- gradients
- optimizer state
- input activations
- intermediate activations
- attention intermediates
- communication buffers
- recomputation metadata

For long-context training, **activation memory** becomes severe.

The token dimension grows:

```text
8k tokens
  -> 32k tokens
  -> 100k+ tokens
```

That increases memory pressure inside attention and MLP blocks.

**Data parallelism alone** does not solve this.

With ordinary data parallelism, every GPU still sees the full sequence for its local microbatch:

```text
GPU 0: full sequence
GPU 1: full sequence
GPU 2: full sequence
GPU 3: full sequence
```

Adding more data-parallel GPUs increases total throughput, but it does **not reduce per-GPU sequence activation memory**.

**ZeRO and FSDP** help shard parameters, gradients, and optimizer state.

They do **not automatically shard the token dimension**.

For 100k+ context training, that difference matters.

---

## 2. Sequence parallelism

**Sequence parallelism** shards the sequence dimension across devices.

Instead of:

```text
GPU 0: tokens 0..99999
GPU 1: tokens 0..99999
GPU 2: tokens 0..99999
GPU 3: tokens 0..99999
```

you can distribute:

```text
GPU 0: tokens 0..24999
GPU 1: tokens 25000..49999
GPU 2: tokens 50000..74999
GPU 3: tokens 75000..99999
```

This increases the **effective context length** you can train because activation memory is spread across GPUs.

But transformer layers are **not embarrassingly parallel** across sequence.

Attention, normalization, projections, masks, position IDs, and backward gradients all need correct data movement.

Hand-written SP requires:

- partitioning input tokens
- partitioning intermediate activations
- inserting collectives
- coordinating forward and backward communication
- preserving masks and position IDs
- overlapping communication with computation
- validating numerical correctness
- composing with data parallelism and optimizer sharding

That is **invasive systems work**.

AutoSP tries to move that work **into the compiler**.

---

## 3. What AutoSP automates

AutoSP automatically converts standard transformer training code into multi-GPU sequence-parallel code.

The PyTorch blog describes it as integrated with **DeepSpeed through DeepCompile**, a compiler ecosystem for distributed training optimizations.

The user-facing workflow is intentionally small:

1. Tag input tensors for AutoSP analysis.
2. Enable DeepCompile.
3. Add the `autosp` compiler pass.
4. Set `sequence_parallel_size`.
5. Compile the model.

Representative configuration:

```python
config = {
    "train_micro_batch_size_per_gpu": 1,
    "train_batch_size": 2,
    "zero_optimization": {
        "stage": 1,
    },
    "compile": {
        "deepcompile": True,
        "passes": ["autosp"],
    },
    "sequence_parallel_size": 4,
    "gradient_clipping": 1.0,
}

model, _, _ = deepspeed.initialize(config=config, model=model)
model.compile(compile_kwargs={"dynamic": True})
```

Then the training loop tags inputs:

```python
inputs, labels, positions, mask = prepare_auto_sp_inputs(batch)

loss = model(
    input_ids=inputs,
    labels=labels,
    position_ids=positions,
    attention_mask=mask,
)
```

The **compiler pass** handles the sequence-parallel code transformation.

This is the same trend we saw in Lecture 35:

```text
Do not just prompt the agent or user to remember systems rules.
Encode the workflow into reusable machinery.
```

Here, the machinery is a compiler pass.

---

## 4. Why this matters for model scientists

Without AutoSP, long-context research often requires **rewriting the training stack**.

Researchers who want to test model behavior at 64k, 128k, or longer contexts may be forced to become **distributed-systems engineers** first.

AutoSP changes the workflow:

```text
before:
  modify DeepSpeed/HuggingFace internals
  hand-insert communication
  debug forward/backward layout bugs
  tune activation checkpointing
  repeat per model and hardware stack

after:
  write normal model code
  enable AutoSP pass
  compile
  benchmark and validate
```

This does not remove the need for **performance engineering**.

It changes **where the complexity lives**.

The complexity moves from every model implementation into a **reusable compiler system**.

---

## 5. DeepSpeed-Ulysses as the target SP strategy

AutoSP targets **DeepSpeed-Ulysses-style** sequence parallelism.

The PyTorch blog notes two important points:

- Ulysses communication overhead can stay constant with increasing GPU counts on NVLink or fat-tree topologies.
- Ulysses SP size is limited by the number of attention heads.

That second point is operationally important.

If a model has 32 heads, Ulysses cannot scale sequence parallelism past that **head-count limit**.

This means AutoSP does **not remove topology and architecture constraints**.

It automates a specific strategy with known tradeoffs.

The useful mental model:

```text
AutoSP is not magic distributed training.
AutoSP is compiler-generated Ulysses-style sequence parallelism plus long-context-aware checkpointing.
```

That is still valuable because the hand-written version is **difficult and error-prone**.

---

## 6. Sequence-aware activation checkpointing

**Activation checkpointing** saves memory by discarding selected intermediate activations during forward pass and recomputing them during backward pass.

Generic activation checkpointing asks:

```text
Which activations should we store?
Which should we recompute?
```

AutoSP adds a long-context-specific strategy called **sequence-aware activation checkpointing**, or SAC.

The reason is that long-context workloads change the **compute/memory balance**.

At long sequence length:

- activation memory grows sharply
- some intermediates become too expensive to keep
- recomputation may be cheaper than OOM
- generic checkpoint policies can be overly conservative

SAC targets this regime.

The tradeoff:

```text
SAC enabled
  -> lower activation memory
  -> longer trainable context
  -> some throughput cost from recomputation

SAC disabled
  -> faster when memory fits
  -> may OOM at longer contexts
```

This is a **deployment decision**, not a checkbox.

Use SAC when it **enables the context length you need** or substantially reduces memory pressure.

---

## 7. Composition with ZeRO

AutoSP composes with ZeRO 0/1 according to the PyTorch post.

That is important because sequence parallelism and ZeRO solve **different memory problems**.

```text
ZeRO/FSDP:
  shard optimizer state, gradients, and parameters

Sequence parallelism:
  shard the token/activation dimension
```

For long-context training, you often need both:

```text
large model
  -> shard parameters and optimizer state

long sequence
  -> shard sequence activations
```

A practical design could look like:

```text
data parallel group
  -> improves global batch throughput

ZeRO group
  -> reduces optimizer/parameter memory

sequence parallel group
  -> increases maximum context length
```

The challenge is making those parallel dimensions **compose without breaking correctness or performance**.

AutoSP's value is that it **handles that composition** for its supported cases rather than requiring every user to implement it manually.

---

## 8. Compiler pass requirements

AutoSP needs to **see the model graph**.

The blog calls out two key limitations.

First, the transformer must be compiled as **one compilable artifact**.

If a project compiles many small functions separately and stitches them together, AutoSP cannot globally analyze and propagate sequence-sharding information through the whole model.

Second, **graph breaks are disallowed** inside the compilable artifact.

Graph breaks complicate:

- tensor sharding propagation
- layout reasoning
- communication insertion
- backward-pass correctness
- activation checkpointing

This is a **compiler reality**.

If the compiler cannot see the whole computation, it cannot **safely transform** the whole computation.

For users, the implication is concrete:

```text
AutoSP-friendly code:
  full transformer compiles as one graph
  tensor metadata is visible
  no graph breaks
  inputs are tagged

AutoSP-hostile code:
  Python control flow that breaks graphs
  custom ops without compiler visibility
  fragmented compile regions
  dynamic behavior the pass cannot analyze
```

This is not just an AutoSP issue.

It is a **general rule** for compiler-based distributed training.

---

## 9. Performance portability

The AutoSP paper argues that putting SP into the compiler can improve **performance portability** across hardware.

The reason:

```text
manual SP implementation
  -> tied to one framework path, one backend, one set of assumptions

compiler pass
  -> can lower the same high-level transformation differently per backend
```

This matters for:

- NVIDIA clusters
- AMD clusters
- mixed backend research
- cloud portability
- future interconnect generations

**Do not overstate this.**

Compiler portability does **not mean every backend is automatically optimal**.

It means the abstraction boundary is better:

```text
model code expresses intent
compiler/runtime chooses distributed lowering
trace/profiler validates the actual result
```

This pairs directly with Lecture 37.

Use **TraceLens** or equivalent profiler analysis to verify whether the generated SP code is **actually efficient** on your hardware.

---

## 10. What to measure

For AutoSP, benchmark **both capability and cost**.

Capability metrics:

- maximum trainable context length before OOM
- peak GPU memory
- whether target batch size fits
- correctness versus baseline loss
- gradient consistency if doing deeper validation

Performance metrics:

- tokens/s
- step time
- forward time
- backward time
- exposed communication time
- all-to-all latency and bandwidth
- recomputation overhead
- GPU idle time
- compile time

Quality metrics:

- loss curve agreement
- downstream long-context validation
- stability at target sequence length
- numerical differences versus hand-written baseline

For a real report, do not only say:

```text
AutoSP works.
```

Say:

```text
AutoSP increased max sequence length from A to B
with C% step-time overhead,
D GB lower peak memory,
E communication exposure,
and loss difference within threshold F.
```

That is the **engineering standard**.

---

## 11. Where AutoSP sits in the long-context stack

Long-context systems have **both training and inference paths**.

```text
training:
  sequence parallelism
  activation checkpointing
  ZeRO/FSDP
  tensor/pipeline parallelism
  distributed optimizer state

serving:
  KV-cache memory
  prefill/decode scheduling
  paged attention
  FP8 KV-cache
  batching and routing
  request concurrency
```

AutoSP belongs to the **training side**.

vLLM FP8 KV-cache belongs to the **serving side**.

TraceLens belongs to **evidence and diagnosis** across both.

Together:

```text
AutoSP:
  train long-context models

vLLM FP8 KV-cache:
  serve long-context models efficiently

TraceLens:
  prove where performance changed
```

For agent systems, these are connected because persistent agents create demand for:

- longer sessions
- larger retrieved context
- long tool traces
- multimodal history
- extended planning state
- larger evaluation prompts

**Training and serving** must both adapt.

---

## 12. Hardware engineer view

AutoSP exposes the **hardware bottlenecks** behind long-context training.

Key questions:

```text
Memory capacity:
  Does sequence sharding make the target context fit?

Memory bandwidth:
  Does recomputation or communication increase pressure elsewhere?

Interconnect:
  Are collectives exposed on the critical path?

Topology:
  Does Ulysses map cleanly onto NVLink, xGMI, or a fat-tree network?

Compiler:
  Can the graph stay intact enough for whole-model transformation?

Kernel quality:
  Do generated shapes hit efficient GEMM and attention kernels?

Scheduling:
  Is communication overlapped with useful compute?
```

This is why long-context training is a **full-stack problem**.

It is **not solved only by bigger HBM**.

It needs:

- model architecture choices
- compiler transformations
- communication topology
- kernel efficiency
- memory planning
- profiler-backed validation

---

## 13. Practical usage checklist

Before trying AutoSP on a real model, check:

- The model is transformer-like and compiles as one artifact.
- The training path avoids graph breaks.
- Input IDs, labels, position IDs, and attention masks can be tagged.
- The DeepSpeed version includes the relevant DeepCompile/AutoSP support.
- The desired ZeRO stage is supported by your AutoSP path.
- The model head count supports the requested Ulysses SP size.
- Your hardware topology can sustain the required collectives.
- You have a baseline for correctness and step time.

Start small:

```text
short sequence
small model
few layers
sp_size = 2
deterministic test
compare loss
```

Then scale:

```text
increase sequence length
increase SP size
enable SAC if needed
add ZeRO
measure traces
compare with hand-written or compiled baseline
```

---

## 14. Failure modes

Common failure modes to expect:

```text
graph break
  -> compiler cannot apply AutoSP safely

wrong masks or position IDs
  -> correctness failure

SP size exceeds head constraints
  -> invalid or inefficient configuration

communication dominates
  -> topology or overlap issue

SAC recomputation too expensive
  -> context fits but throughput regresses

dynamic behavior not captured
  -> compile failure or fallback path

loss diverges from baseline
  -> sharding, mask, communication, or numerical issue
```

The right response is **not to guess**.

Use:

- correctness tests
- trace comparison
- per-rank loss logging
- memory reports
- collective analysis
- event replay for slow kernels where possible

---

## Mini-lab: AutoSP evaluation plan

You do **not need an 8-GPU node** to design the experiment.

Write an evaluation plan for a Llama-style transformer.

Include:

```text
Model:
GPU type:
GPU count:
baseline strategy:
AutoSP strategy:
SP size:
ZeRO stage:
sequence lengths:
batch size:
activation checkpointing policy:
correctness metric:
performance metric:
trace/profiler plan:
failure threshold:
deployment decision rule:
```

If you have access to multiple GPUs, run a small correctness test first:

```bash
cd benchmarks/autosp
./run_autosp.sh \
  --compile autosp \
  --batch-size 1 \
  --seq-length 64 \
  --sp-size 2 \
  --num-layers 1 \
  --steps 1 \
  --deterministic
```

Then compare against a baseline mode and record whether per-rank losses match within your threshold.

---

## Key takeaways

- Long-context training can OOM because activation memory grows with sequence length.
- ZeRO and FSDP help model-state memory but do not automatically shard the token dimension.
- Sequence parallelism shards the sequence dimension across GPUs, enabling longer context training.
- Hand-written SP is invasive because it requires partitioning activations and inserting correct collectives in forward and backward passes.
- AutoSP moves Ulysses-style sequence parallelism into a DeepSpeed/DeepCompile compiler pass.
- Sequence-aware activation checkpointing trades recomputation for longer trainable contexts.
- AutoSP composes with supported ZeRO stages, but it has strict compiler requirements.
- Whole-model compilation and no graph breaks are core constraints.
- Validate AutoSP with max context, memory, step time, communication exposure, and correctness, not only whether the run starts.
- Compiler-generated distributed training should be paired with trace-driven analysis.

---

## References

- PyTorch Blog, "Introducing AutoSP": [https://pytorch.org/blog/introducing-autosp/](https://pytorch.org/blog/introducing-autosp/)
- AutoSP paper, "AutoSP: Unlocking Long-Context LLM Training via Compiler-Based Sequence Parallelism": [https://openreview.net/pdf?id=0fgsHvmBBI](https://openreview.net/pdf?id=0fgsHvmBBI)
- DeepSpeed AutoSP examples: [https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/autosp](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/autosp)
- DeepCompile paper: [https://arxiv.org/abs/2504.09983](https://arxiv.org/abs/2504.09983)
- Lecture 36 - FP8 KV-Cache in vLLM: [Lecture-36.md](Lecture-36.md)
- Lecture 37 - TraceLens: [Lecture-37.md](Lecture-37.md)

---

*Next: [Lecture 39 - Agent Skills Eval](Lecture-39.md)*
