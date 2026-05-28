# Lecture 43 - MLSys 2026 Kernel Contest: AI-Assisted Blackwell LLM Kernel Optimization

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 42](Lecture-42.md) | **Next:** [Lecture 44](Lecture-44.md)

---

The MLSys 2026 FlashInfer AI Kernel Generation Contest is a compact map of where AI systems work is going.

It asks participants to create high-performance GPU kernels for modern LLM inference operations on NVIDIA Blackwell B200 GPUs.

The important part is not only the kernel work.

The contest explicitly welcomes:

```text
expert-crafted seed kernels with agent-assisted evolution
fully agent-generated kernel solutions
```

That makes it a case study in:

```text
GPU kernel optimization
  + LLM inference runtime internals
  + AI-assisted systems engineering
```

This is the layer below ordinary agent apps.

It is where agent workloads become:

- Tensor Core instructions
- HBM traffic
- shared-memory layouts
- sparse indexers
- expert routing
- register pressure
- occupancy tradeoffs
- benchmark win rates

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why the MLSys 2026 contest is an AI systems/runtime signal, not a normal hackathon.
2. Identify the three competition tracks and the inference bottleneck each targets.
3. Understand why NVIDIA Blackwell B200 changes the optimization target.
4. Connect FP8 MoE, sparse attention, and Gated Delta Net kernels to modern LLM inference.
5. Explain how FlashInfer-Bench evaluates correctness, speed, and win rate against baselines.
6. Distinguish human-written, agent-assisted, and fully agent-generated kernel workflows.
7. Design a kernel optimization loop with profiling, correctness checks, and benchmark discipline.
8. Map this contest to career paths in AI runtime, GPU compiler, and accelerator software engineering.

---

## 1. What the contest is

The contest page describes the challenge as:

```text
create optimized CUDA kernels for cutting-edge LLM operations
for NVIDIA Blackwell B200 GPUs
```

The evaluation platform is FlashInfer-Bench.

Submissions compete on:

- correctness
- speed
- win rate against FlashInfer baselines

The contest tracks target three operations that matter in modern inference:

```text
Track A:
  Fused MoE with FP8 support

Track B:
  DeepSeek Sparse Attention

Track C:
  Gated Delta Net
```

Participants can use:

- CUDA
- Triton
- CuTe DSL
- TileLang
- cuTile
- other kernel programming systems

That tool list is the real signal.

The frontier of inference engineering is not one API.

It is a stack of DSLs, compilers, profilers, runtimes, and benchmark harnesses.

---

## 2. Why this belongs in an agent course

Agentic AI is usually taught from the top:

```text
prompts
tools
RAG
workflows
agent orchestration
```

But production agents create inference demand.

That demand becomes:

```text
low latency
high throughput
long context
multi-user concurrency
tool-loop responsiveness
local or edge deployment
cost per generated token
```

Those requirements eventually land on kernels.

The contest sits at the bottom of the stack:

```text
agent workload
  -> model server
  -> inference runtime
  -> attention / MoE / recurrent-state kernels
  -> GPU architecture
```

If you want to build serious agent infrastructure, you need to understand the lower layers.

Not every agent engineer writes CUDA.

But the best systems engineers know which kernel bottleneck they are paying for.

---

## 3. Track A: Fused FP8 MoE

Mixture-of-experts models activate only a subset of experts per token.

That saves compute, but it creates systems problems:

- dynamic routing
- irregular memory access
- token-to-expert dispatch
- expert load imbalance
- small or fragmented GEMMs
- scaling and reduction overhead
- top-k expert selection
- FP8 quantization and dequantization

The contest's Track A focuses on fused MoE kernels with FP8 support.

The optimization target:

```text
routing + dispatch + GEMM + scaling + output assembly
```

The reason to fuse:

```text
fewer kernel launches
less HBM traffic
better Tensor Core utilization
less intermediate materialization
```

Important performance questions:

```text
Are tokens grouped efficiently by expert?
Are FP8 block scales loaded efficiently?
Are GEMMs large enough to use Tensor Cores well?
Is expert imbalance causing tail latency?
Are intermediate buffers hitting HBM unnecessarily?
Does fusion increase register pressure too much?
```

This connects directly to modern MoE models:

- DeepSeek-style expert systems
- Mixtral-style routing
- Qwen MoE variants
- small active-parameter models such as Lecture 40's ZAYA1-8B

MoE is not "free sparsity."

It is a routing and memory-layout problem.

---

## 4. Track B: Sparse Attention

Dense attention scales poorly with sequence length.

The basic cost pattern:

```text
full attention:
  every query attends over many keys
  KV cache traffic grows with context
```

Sparse attention reduces work by selecting a subset of relevant tokens or blocks.

The contest's Track B targets DeepSeek Sparse Attention with separate indexer and attention kernels.

That split matters:

```text
indexer:
  choose which blocks/tokens matter

attention:
  compute attention over selected sparse structure
```

Sparse attention bottlenecks:

- top-k index generation
- block sparse metadata
- irregular KV reads
- memory coalescing
- cache locality
- branch divergence
- load balancing across queries
- interaction with paged KV cache
- FP8 data paths

Sparse attention is hard because it trades arithmetic for control flow and memory indirection.

The key question:

```text
Does the sparsity saved enough memory traffic and compute
to pay for indexer overhead and irregular access?
```

This connects to:

- Lecture 36: FP8 KV-cache
- Lecture 37: trace-driven performance analysis
- long-context agent sessions
- DeepSeek-style inference systems

For long-context agents, sparse attention is a direct path to lower memory traffic.

But it must be profiled, not assumed.

---

## 5. Track C: Gated Delta Net

Gated Delta Net is a sequence-modeling approach used in Qwen3-Next.

The contest includes decode and prefill kernels.

This matters because GDN-style systems represent a broader shift:

```text
not every future long-context model will be standard dense attention
```

State-space, recurrent, delta-rule, and hybrid models try to reduce attention cost by maintaining compact state.

The GPU problems change:

- state update kernels
- recurrent dependency handling
- scan-like computation
- chunking
- prefill/decode split
- memory layout for persistent state
- low-latency decode
- high-throughput prefill

Track C is therefore not just "one more kernel."

It is a signal that inference runtimes must support model families beyond standard transformers.

For systems engineers, that means:

```text
runtime design must be model-architecture aware
```

An inference runtime optimized only for dense attention may underperform on hybrid architectures.

---

## 6. Why Blackwell B200 matters

The contest targets NVIDIA Blackwell B200 GPUs.

That matters because kernel optimization is hardware-specific.

Questions change by architecture:

- Tensor Core throughput
- FP8/FP4 paths
- memory bandwidth
- shared memory capacity and behavior
- register file pressure
- scheduler behavior
- warp-group MMA support
- TMA or async copy behavior
- occupancy tradeoffs

Optimizing for B200 is not the same as optimizing for A100 or H100.

A kernel that wins on one generation may lose on another because:

- the compute/memory balance changes
- Tensor Core instruction choices change
- memory hierarchy changes
- compiler lowering changes
- occupancy limits change

That is why official evaluation on bare metal matters.

The contest page notes that Modal scores are reference-only because clock frequency cannot be locked, while official evaluations run on bare-metal machines.

This is a good performance-engineering rule:

```text
cloud convenience is useful for iteration
bare metal is needed for final claims
```

---

## 7. FlashInfer-Bench

FlashInfer-Bench is the contest evaluation platform.

The benchmark model:

```text
kernel spec
  -> candidate kernel
  -> correctness tests
  -> speed measurement
  -> comparison against FlashInfer baseline
  -> win rate
```

This is the right structure for kernel work.

A kernel is not useful because it compiles.

It must:

- match numerical expectations
- handle shape variations
- beat or match a strong baseline
- avoid pathological cases
- be reproducible
- be explainable in a writeup

The contest requires tagged GitHub commits for evaluation and a technical report.

That encourages the correct discipline:

```text
code
  -> benchmark
  -> profile
  -> explain
  -> reproduce
```

---

## 8. Human plus agent kernel generation

The contest explicitly allows two approaches:

```text
expert-crafted seed kernels with agent-assisted evolution
fully agent-generated solutions
```

These are different workflows.

### Human plus agent

The human writes:

- baseline kernel structure
- tiling strategy
- memory layout
- correctness harness
- profiling plan

The agent helps:

- generate variants
- tune constants
- try layout changes
- refactor code
- run benchmarks
- summarize profiling results

### Fully agent-generated

The agent owns more of the loop:

- read kernel spec
- generate code
- run tests
- benchmark
- mutate kernel
- select winners
- produce reproducibility scripts

The contest page requires agent solutions to open-source scripts that reproduce kernels.

That is important.

For AI-generated systems code, the artifact is not only the final kernel.

The artifact is also:

```text
the generation process
```

That process must be reproducible.

---

## 9. The optimization loop

A serious kernel optimization loop looks like this:

```text
1. Read the kernel spec.
2. Implement a correct baseline.
3. Build correctness tests.
4. Benchmark against the reference.
5. Profile the bottleneck.
6. Generate one controlled optimization variant.
7. Re-test correctness.
8. Re-benchmark.
9. Record the delta.
10. Repeat.
```

Do not change five things at once.

Kernel work is full of traps:

- a faster kernel may be numerically wrong
- a change may help one shape and hurt another
- a local benchmark may not match official clocks
- register pressure may erase fusion gains
- memory coalescing may be broken by a layout change
- branch divergence may dominate sparse kernels

The right unit of progress:

```text
one hypothesis
one variant
one measurement
one retained or rejected change
```

---

## 10. Profiling checklist

Use Nsight Compute, Nsight Systems, FlashInfer-Bench outputs, and trace tools where appropriate.

Measure:

- achieved occupancy
- register count
- shared memory usage
- Tensor Core utilization
- memory throughput
- L2 hit rate
- global load/store efficiency
- warp divergence
- instruction mix
- kernel launch overhead
- achieved TFLOPS or effective bandwidth
- latency distribution across shapes

Interpretation examples:

```text
low Tensor Core utilization:
  tiling, dtype path, or GEMM shape may be poor

high HBM bandwidth with low compute:
  memory-bound kernel; optimize layout and traffic

high register pressure:
  fusion or unrolling may be too aggressive

high divergence:
  sparse indexing or routing path may be unbalanced

good average but bad tail:
  expert imbalance or sparse metadata distribution may matter
```

This connects to Lecture 37:

```text
performance claims need trace and profiling evidence
```

---

## 11. Toolchain choices

The contest allows multiple implementation paths.

Do not treat them as interchangeable.

Each tool sits at a different abstraction level:

```text
FlashInfer:
  LLM inference kernel/runtime library and benchmark baseline

CUDA:
  lowest-level mainstream NVIDIA GPU programming interface

Triton:
  Python-like GPU kernel DSL optimized for fast iteration

TileLang:
  tile-level kernel DSL for composable AI kernels

CuTe DSL:
  Python DSL around CUTLASS/CuTe layout and tensor abstractions

cuTile:
  NVIDIA CUDA Tile Python DSL targeting Tile IR and portable tensor-core kernels

OpenEvolve:
  evolutionary coding agent for automated program optimization

Modal:
  serverless GPU/cloud execution environment for iteration and reference runs

Blackwell B200:
  target hardware that determines what "fast" actually means
```

The practical rule:

```text
choose the tool that gives you the fastest correct iteration
for the bottleneck you actually have
```

But for this contest, the real skill is knowing when to drop down a level.

---

### 11.1 FlashInfer

FlashInfer is the center of gravity for this contest.

It is an LLM serving kernel library focused on high-performance inference primitives:

- attention
- paged KV-cache operations
- sampling
- normalization
- MoE-related kernels
- quantization-aware serving paths
- decode and prefill helpers

In the contest, FlashInfer plays three roles:

```text
reference implementation:
  the baseline your kernel must beat

benchmark environment:
  FlashInfer-Bench measures correctness and speed

runtime context:
  the operations are real LLM serving bottlenecks, not toy kernels
```

Why this matters:

```text
If you beat a naive PyTorch baseline, that proves little.

If you beat FlashInfer on a real inference primitive,
that is a meaningful systems result.
```

What to study in FlashInfer:

- API shape for decode and prefill kernels
- tensor layout conventions
- page/block abstractions for KV cache
- supported dtypes and quantization paths
- baseline kernel behavior
- benchmark input distributions
- numerical tolerances

Common mistake:

```text
optimizing against the wrong mental model
```

FlashInfer kernels are already specialized. You need to understand what the baseline is doing before assuming an optimization opportunity exists.

For example, if a sparse attention kernel is slow, the bottleneck may not be the attention math. It may be:

- sparse index generation
- metadata reads
- poor memory coalescing
- shape-specific occupancy
- scale loading
- page table indirection

FlashInfer is where LLM theory becomes concrete runtime layout.

---

### 11.2 CUDA

CUDA is the most direct and controllable path.

Use CUDA when you need:

- explicit thread/block mapping
- warp-level control
- shared memory control
- vectorized global memory access
- explicit synchronization
- custom Tensor Core instruction paths
- fine-grained launch configuration
- maximum control over register pressure and occupancy

CUDA is the right tool when the kernel is blocked by details the compiler DSL does not expose.

Examples:

```text
FP8 MoE:
  custom token grouping, expert dispatch, scale loading, epilogue fusion

Sparse attention:
  custom sparse metadata traversal and memory coalescing

Gated Delta Net:
  specialized state update and decode loop scheduling
```

CUDA gives you control, but it also gives you enough rope.

Common CUDA failure modes:

- out-of-bounds memory access
- bank conflicts
- uncoalesced loads
- excessive register use
- low occupancy
- branch divergence
- poor Tensor Core utilization
- excessive synchronization
- numerically wrong accumulation
- shape-specific regressions

CUDA debugging discipline:

```text
1. correctness first
2. one optimization at a time
3. inspect generated SASS/PTX when needed
4. profile register/shared-memory occupancy
5. test multiple shapes, not one cherry-picked shape
```

Use CUDA when you need a hand-tuned final kernel or when you need to understand exactly why a higher-level kernel is losing.

---

### 11.3 Triton

Triton is a Python-like language for writing GPU kernels.

It is valuable because it shortens the edit-test-profile loop.

Use Triton when you need:

- fast prototyping
- parameterized kernels
- autotuning
- Python-native iteration
- easier agent-generated variants
- compact expression of tile-level math

Triton is often a good first implementation path for:

- elementwise fusion
- reductions
- small GEMM-like kernels
- custom attention prototypes
- shape-specialized kernels

Why agents like Triton:

```text
less boilerplate than CUDA
Python syntax
shorter kernels
faster mutation loop
easier benchmark automation
```

Where Triton can struggle:

- extremely irregular sparse access
- full control over warp-level primitives
- newest NVIDIA architecture features before compiler support catches up
- complex Tensor Core scheduling
- cases where generated code choices are opaque

Triton is not "slower CUDA."

It is a different abstraction boundary.

For agentic kernel search, a practical path is:

```text
Triton prototype
  -> find algorithm and tiling idea
  -> benchmark shapes
  -> port hot winner to CUDA or CuTe if deeper control is needed
```

---

### 11.4 TileLang

TileLang is a tile-oriented DSL for writing high-performance AI kernels.

The key abstraction is that you describe computation in terms of tiles, rather than manually managing every thread-level operation.

This is useful because AI kernels usually have tiled structure:

- matrix multiplication
- attention blocks
- reductions
- normalization
- block sparse compute
- fused epilogues

Use TileLang when you want:

- a higher-level tiled programming model
- composable kernel descriptions
- faster search over tile sizes and schedules
- kernels that are easier for agents to mutate than low-level CUDA

The mental model:

```text
CUDA:
  think threads, warps, shared memory, synchronization

TileLang:
  think tiles, loops over tiles, memory scopes, schedule choices
```

Why this matters for the contest:

```text
agent-generated kernels benefit from abstractions
that reduce the number of ways to write invalid code
```

TileLang can be a better target for automated exploration because the search space is closer to the math.

But you still need profiling.

A tile abstraction can hide:

- bad memory layout
- excessive register use
- poor generated code
- compiler limitations
- shape-specific scheduling failures

Use TileLang as a rapid kernel-generation layer, then validate with FlashInfer-Bench and Nsight.

---

### 11.5 CuTe DSL

CuTe DSL is NVIDIA's Python DSL around CUTLASS/CuTe concepts.

CuTe is fundamentally about tensor layouts and tiled tensor algebra.

This matters because many high-performance kernels are layout problems.

A good kernel is not only:

```text
do the right math
```

It is:

```text
map the math to hardware-friendly tiled layouts
```

CuTe-style thinking emphasizes:

- layout composition
- tile shapes
- memory hierarchy
- tensor views
- copy atoms
- MMA atoms
- pipeline stages
- warp and warpgroup organization

Use CuTe DSL when:

- the kernel is GEMM-like
- Tensor Core utilization is central
- layouts are complex
- you need CUTLASS-grade abstractions without raw C++ template pain
- you care about SM90/SM100-style features and structured tiling

Cost:

- steep learning curve
- layout algebra is unforgiving
- compiler/runtime stack maturity matters
- debugging requires understanding generated lower-level behavior

Why it matters for Blackwell:

```text
Blackwell optimization is heavily about feeding tensor cores
and moving data through the memory hierarchy correctly.
```

CuTe DSL gives you a way to express that more directly than Triton in some cases, while staying higher level than hand-written CUDA.

For contest work, CuTe DSL is most relevant to Track A FP8 MoE and any GEMM-like subproblem.

---

### 11.6 cuTile

cuTile is NVIDIA's Python implementation of the CUDA Tile programming model.

The official docs describe cuTile as a Python-based DSL where kernels operate on tiles, using functions such as:

- `ct.load`
- `ct.store`
- tile arithmetic
- reductions
- matrix multiply
- `ct.launch`

Important distinction:

```text
arrays:
  global-memory objects passed from host

tiles:
  immutable kernel-local tensor-like values with compile-time shapes
```

This is a major abstraction shift.

Instead of writing:

```text
thread i loads element i
```

you write:

```text
load this tile
operate on this tile
store this tile
```

Why cuTile matters:

- it targets NVIDIA's tile programming model
- it aims to expose hardware features through tile abstractions
- it can be easier to modify than deep CUDA/CUTLASS templates
- it is aligned with agent-assisted kernel translation work

This connects directly to Lecture 35, where cuTile Python to cuTile.jl translation was used as a concrete Agent Skills example.

For the MLSys contest:

```text
cuTile may be useful when you want tile-level expression
without writing full CUDA boilerplate.
```

But treat it as a young, hardware-sensitive toolchain.

Validate:

- supported GPU architecture
- CUDA Toolkit version
- generated kernel performance
- limitations for your target op
- debugging workflow

---

### 11.7 OpenEvolve

OpenEvolve is not a GPU kernel language.

It is an evolutionary coding agent framework.

Its role in this contest is search.

A normal kernel workflow:

```text
human writes variant
human benchmarks variant
human reads result
human writes next variant
```

An OpenEvolve-style workflow:

```text
population of candidate kernels
  -> correctness filter
  -> benchmark score
  -> select winners
  -> mutate/crossover/generate new candidates
  -> repeat
```

Why this fits GPU kernels:

- performance landscapes are rugged
- small code changes can produce large speed changes
- many variants fail correctness
- many optimizations are shape-specific
- humans cannot manually explore the full schedule space

What OpenEvolve needs to be useful:

- deterministic benchmark command
- fast correctness check
- objective score
- saved artifacts
- mutation boundaries
- timeout policy
- rollback policy
- result database

For kernel optimization, the fitness function should not be just:

```text
fastest one run
```

It should include:

- correctness
- median latency
- variance
- shape coverage
- compile success
- code size or complexity
- no forbidden APIs
- no benchmark cheating

This is where the contest's agent-generated-kernel track becomes serious.

The best agent is not the one that writes the prettiest CUDA.

It is the one that can run a disciplined generate-test-profile-select loop.

---

### 11.8 Modal

Modal is a serverless cloud platform for running compute-intensive workloads.

In this contest context, Modal is useful for iteration and reference execution.

Use Modal for:

- repeatable containerized benchmark runs
- GPU access without owning hardware
- quick starter-kit experiments
- parallel search jobs
- artifact collection
- CI-like evaluation pipelines

But the contest page explicitly warns that Modal scores are reference-only because GPU clocks cannot be locked.

That means:

```text
Modal is good for iteration.
Bare metal is required for official performance claims.
```

Good Modal workflow:

```text
1. package benchmark container
2. run correctness tests
3. run rough performance screen
4. collect logs and artifacts
5. promote promising candidates to bare-metal validation
```

Do not overfit to Modal timing noise.

Use it to reduce the number of bad candidates, not to prove final speed.

---

### 11.9 Blackwell B200

Blackwell B200 is the target hardware.

That determines what optimizations matter.

Official NVIDIA docs identify B200 as compute capability 10.0 in the Blackwell tuning guide and describe the architecture as targeting generative AI and accelerated computing.

The practical B200 concerns for this contest:

- FP8 and lower-precision Tensor Core paths
- high HBM bandwidth
- large memory capacity
- Blackwell-specific scheduling behavior
- architecture-specific compiler lowering
- Tensor Core feeding and pipeline design
- shared-memory/L1 behavior
- register pressure and occupancy balance
- support for newer CUDA features

Do not assume Hopper instincts transfer perfectly.

Questions to ask on B200:

```text
Does this kernel use the right dtype path?
Does it keep Tensor Cores busy?
Is it memory bandwidth bound?
Is shared memory helping or hurting?
Is register pressure limiting occupancy?
Does the compiler generate Blackwell-appropriate instructions?
Does performance change across B200 vs H100?
```

For Track A, Blackwell matters because FP8 MoE wants strong Tensor Core utilization and efficient scale handling.

For Track B, Blackwell matters because sparse attention may be memory and metadata bound.

For Track C, Blackwell matters because recurrent-state kernels may stress different latency and memory paths than dense attention.

The key lesson:

```text
hardware generation is part of the algorithm.
```

An inference kernel is not just math. It is math mapped onto a specific machine.

---

### 11.10 Tool selection matrix

Use this matrix as a starting point.

| Need | Best first tool | Why |
|---|---|---|
| Understand contest baseline | FlashInfer | It defines the reference runtime and benchmark target |
| Maximum low-level control | CUDA | Explicit control over threads, memory, and synchronization |
| Fast prototype and autotune | Triton | Shorter Python-like kernels and fast iteration |
| Tiled AI kernel exploration | TileLang | Tile-level abstraction for AI workloads |
| Tensor Core / layout-heavy GEMM-like work | CuTe DSL | Strong layout and tiled tensor abstractions |
| Pythonic CUDA Tile experiments | cuTile | Tile IR-oriented Python DSL for NVIDIA GPUs |
| Automated variant search | OpenEvolve | Evolutionary loop around code generation and benchmark scores |
| Cloud iteration | Modal | Convenient GPU jobs and artifact collection |
| Final performance claim | Bare-metal B200 | Official target with controlled clocks and reproducibility |

The strongest workflow combines tools:

```text
FlashInfer-Bench:
  tells you whether you are winning

Triton / TileLang / cuTile:
  help explore algorithmic variants quickly

CUDA / CuTe DSL:
  help turn the winning idea into a hardware-tuned kernel

OpenEvolve:
  scales search over variants

Modal:
  scales early experimentation

Bare-metal B200:
  validates the final result
```

---

## 12. Agentic kernel optimization workflow

A useful AI-assisted workflow:

```text
human:
  defines benchmark target and constraints

agent:
  reads spec and prior results
  proposes variants
  edits one kernel at a time
  runs correctness tests
  runs benchmark
  records deltas
  rejects bad variants

human:
  reviews profiling evidence
  adjusts search direction
```

Required guardrails:

- deterministic benchmark scripts
- strict correctness tests
- one-variant-at-a-time discipline
- no silent benchmark cherry-picking
- raw results stored
- code diffs reviewed
- shape coverage maintained

This is where agent skills matter.

A good kernel-optimization skill should encode:

- profiling checklist
- common CUDA failure modes
- benchmark protocol
- correctness rules
- allowed search space
- reporting template

Then use Lecture 39's skill eval pattern to test whether the skill improves kernel work.

---

## 13. What to learn before competing

Core prerequisites:

```text
GPU architecture:
  SMs, warps, occupancy, memory hierarchy, Tensor Cores

CUDA programming:
  blocks, threads, shared memory, synchronization, vectorized loads

LLM inference:
  prefill, decode, KV cache, paged attention, MoE routing

Numerics:
  FP8 formats, block scaling, accumulation, error tolerances

Profiling:
  Nsight Systems, Nsight Compute, benchmark discipline

Runtime systems:
  FlashInfer, vLLM, TensorRT-LLM concepts

Agent workflows:
  reproducible code generation, patching, test loops
```

If you lack these, start with a smaller kernel:

- vector add
- layernorm
- small GEMM
- attention score kernel
- top-k indexer

Then move toward the contest kernels.

---

## 14. Career signal

The contest points toward roles such as:

- AI runtime engineer
- GPU kernel engineer
- inference infrastructure engineer
- GPU compiler engineer
- accelerator software engineer
- AI systems researcher
- autonomous optimization systems engineer

These roles sit between:

```text
model architecture
hardware architecture
compiler/runtime software
production inference serving
```

They are rarer than AI application roles because they require:

- low-level performance instincts
- ML workload knowledge
- systems debugging
- mathematical numerics
- hardware awareness
- benchmark integrity

If you want to move from AI application work into AI systems work, this contest is a strong practice target.

---

## 15. How this maps to this roadmap

Relevant earlier lectures:

- Lecture 32: LLM internals and inference mechanics
- Lecture 35: agent skills for GPU kernel translation
- Lecture 36: FP8 KV-cache and attention quantization
- Lecture 37: TraceLens and trace-driven performance analysis
- Lecture 38: AutoSP and compiler-generated sequence parallelism
- Lecture 40: small MoE reasoning model deployment tradeoffs
- Lecture 42: durable agent harnesses for tool-oriented long-running work

The contest combines all of them:

```text
kernel optimization:
  low-level GPU work

LLM inference:
  real model bottlenecks

agent generation:
  automated search and code synthesis

benchmarking:
  correctness and performance evidence

runtime thinking:
  kernels as part of the serving stack
```

This is a bridge from agent engineering to AI hardware/software co-design.

---

## Mini-lab: build a contest preparation plan

Pick one track:

```text
Track A: FP8 MoE
Track B: Sparse Attention
Track C: Gated Delta Net
```

Write a preparation plan:

```text
Track:
Target kernel:
Hardware:
Baseline:
Correctness tests:
Benchmark command:
Profiler:
First bottleneck hypothesis:
First three variants:
Expected risk:
Acceptance threshold:
Writeup evidence:
Agent role:
Human review points:
```

Then write a one-week schedule:

```text
Day 1:
  reproduce starter kit

Day 2:
  understand baseline and shapes

Day 3:
  profile bottleneck

Day 4:
  implement first variant

Day 5:
  benchmark and reject/keep

Day 6:
  agent-assisted search

Day 7:
  write results and next plan
```

The goal is not to win immediately.

The goal is to build a disciplined kernel optimization loop.

---

## Key takeaways

- The MLSys 2026 FlashInfer contest is about real LLM inference kernels on NVIDIA Blackwell B200 GPUs.
- The tracks target high-value inference bottlenecks: FP8 MoE, sparse attention, and Gated Delta Net.
- MoE optimization is a routing, memory-layout, Tensor Core, and fusion problem.
- Sparse attention trades dense compute for indexing and irregular memory access.
- Gated Delta Net points toward non-standard transformer alternatives and recurrent-state inference.
- FlashInfer-Bench emphasizes correctness, speed, and win rate against strong baselines.
- CUDA gives maximum control; Triton, TileLang, CuTe DSL, and cuTile trade some control for faster iteration and higher-level tiled abstractions.
- OpenEvolve-style agents are useful when correctness and benchmark scripts define a reliable evolutionary search loop.
- Modal is useful for iteration, but final claims need controlled bare-metal B200 measurement.
- Blackwell B200 changes the optimization target; do not assume Hopper-tuned kernels transfer unchanged.
- Agent-generated kernels must be reproducible, not just fast once.
- Serious kernel work requires hypothesis-driven profiling and benchmark integrity.
- This contest is a strong career signal for AI runtime, GPU compiler, and inference infrastructure engineering.

---

## References

- MLSys 2026 FlashInfer AI Kernel Generation Contest: [https://github.com/flashinfer-ai/mlsys26-contest/blob/main/index.html](https://github.com/flashinfer-ai/mlsys26-contest/blob/main/index.html)
- FlashInfer: [https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- FlashInfer-Bench: [https://bench.flashinfer.ai](https://bench.flashinfer.ai)
- Starter kit: [https://github.com/flashinfer-ai/flashinfer-bench-starter-kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)
- Agent baseline: [https://github.com/flashinfer-ai/mlsys26-agent-baseline](https://github.com/flashinfer-ai/mlsys26-agent-baseline)
- FlashInfer documentation: [https://docs.flashinfer.ai](https://docs.flashinfer.ai)
- CUDA C++ Programming Guide: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- NVIDIA Blackwell Tuning Guide: [https://docs.nvidia.com/cuda/blackwell-tuning-guide/](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- Triton documentation: [https://triton-lang.org/main/index.html](https://triton-lang.org/main/index.html)
- OpenAI, "Introducing Triton": [https://openai.com/research/triton](https://openai.com/research/triton)
- TileLang repository: [https://github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang)
- CuTe DSL documentation: [https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
- CUDA Tile and cuTile Python: [https://developer.nvidia.com/cuda/tile](https://developer.nvidia.com/cuda/tile)
- cuTile Python documentation: [https://docs.nvidia.com/cuda/cutile-python](https://docs.nvidia.com/cuda/cutile-python)
- OpenEvolve repository: [https://github.com/algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)
- Modal documentation: [https://modal.com/docs](https://modal.com/docs)
- Lecture 35 - Agent Skills for GPU Kernel Translation: [Lecture-35.md](Lecture-35.md)
- Lecture 36 - FP8 KV-Cache in vLLM: [Lecture-36.md](Lecture-36.md)
- Lecture 37 - TraceLens: [Lecture-37.md](Lecture-37.md)

---

*Next: [Lecture 44 - Efficient Local RAG Stack](Lecture-44.md)*
