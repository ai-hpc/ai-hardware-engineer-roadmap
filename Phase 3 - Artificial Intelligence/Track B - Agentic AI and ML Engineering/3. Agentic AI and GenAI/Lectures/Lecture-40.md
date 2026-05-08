# Lecture 40 - ZAYA1-8B: Small MoE Reasoning, AMD Training, and Test-Time Compute

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 39](Lecture-39.md) | **Next:** [Lecture 41](Lecture-41.md)

---

ZAYA1-8B is interesting for this course for three reasons:

1. It is a small mixture-of-experts reasoning model with less than 1B active parameters.
2. It was trained end-to-end on an AMD MI300X stack.
3. Its strongest results depend on model-harness co-design through Markovian RSA test-time compute.

It is also interesting for a fourth reason:

```text
The agentic benchmarks are not the headline strength.
```

That makes it a good model-selection case study.

Do not read ZAYA1-8B as:

```text
small model beats everything
```

Read it as:

```text
a specialized small MoE can punch above its active-parameter count
when architecture, training stack, post-training, and inference harness are co-designed
```

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain active parameters versus total parameters in an MoE model.
2. Understand why a 760M-active-parameter model can still draw from 8B+ total parameters.
3. Explain why AMD-trained frontier-style results matter for hardware ecosystem diversity.
4. Describe Markovian RSA as a bounded-context test-time compute harness.
5. Distinguish base benchmark scores from RSA-boosted scores.
6. Identify why math/coding specialization does not imply strong tool use.
7. Understand the deployment caveat around Zyphra's forked vLLM/Transformers support.
8. Design a fair evaluation plan for using ZAYA1-8B in agent and coding workflows.

---

## 1. What ZAYA1-8B is

ZAYA1-8B is a Zyphra mixture-of-experts language model.

According to Zyphra's Hugging Face model card:

```text
total parameters: 8.4B
active parameters: 760M
license: Apache-2.0
specialization: math, reasoning, coding
```

The active parameter count is the key.

In a dense model:

```text
every token uses every parameter
```

In an MoE model:

```text
each token activates selected experts
```

So the inference cost can be closer to the active parameter count, while model capacity is distributed across a larger pool of total parameters.

This is why ZAYA1-8B is framed as an intelligence-density model:

```text
maximize useful capability per active parameter and per FLOP
```

That is a hardware-relevant design target.

---

## 2. Why AMD training matters

Zyphra says ZAYA1-8B was pretrained, midtrained, and supervised fine-tuned on an AMD Instinct MI300 stack.

The published Zyphra post describes a cluster with:

- AMD Instinct MI300X GPUs
- AMD Pensando Pollara interconnect
- IBM-built custom training cluster
- 1,024 MI300X nodes

Why this matters:

```text
Most LLM infrastructure assumes NVIDIA CUDA first.
```

An AMD-trained model with strong published reasoning and coding results is evidence that serious model training is not inherently locked to one vendor stack.

For a hardware engineer, the questions are:

```text
How mature is the ROCm software stack?
How reliable is the collective communication path?
How much custom infrastructure was required?
What kernels and compiler paths were missing?
What parts are reusable by other teams?
```

The existence of the model does not prove AMD is automatically a drop-in replacement for every training workload.

It proves the alternate path is technically viable at this scale when the team invests across the stack.

---

## 3. Architecture signals

Zyphra highlights several architecture choices:

- mixture of experts
- Compressed Convolutional Attention
- MLP-based expert router
- learned residual scaling

The course-level takeaway is not to memorize each mechanism.

The takeaway is that ZAYA1-8B is not just a small generic transformer.

It is an architecture built around:

```text
low active compute
strong routing
attention efficiency
stable depth behavior
post-training for reasoning
```

This connects to earlier lectures:

- Lecture 32: model mechanics
- Lecture 36: memory and serving costs
- Lecture 38: long-context training systems

Small models that compete on reasoning usually require systems co-design.

Architecture alone is not enough.

---

## 4. Benchmark claims: read carefully

Zyphra reports strong performance on math and coding benchmarks.

The Hugging Face model card reports in-class scores such as:

```text
AIME'26:          89.1
HMMT Feb.'26:     71.6
IMO-AnswerBench: 59.3
APEX-shortlist:  32.2
LiveCodeBench:   65.8
GPQA-Diamond:    71.0
MMLU-Pro:        74.2
```

It also reports weaker relative agentic scores:

```text
BFCL-v4: 39.22
tau2:    43.12
```

Important caveat:

```text
Zyphra states the comparison numbers are run on Zyphra's evaluation harness.
```

That does not make them useless.

It means you should treat them as vendor-reported until independently reproduced in your target environment.

Good benchmark reading separates:

```text
source of numbers
benchmark task
inference budget
base vs test-time compute
active parameters
total parameters
deployment stack
your actual use case
```

---

## 5. Base scores versus RSA-boosted scores

ZAYA1-8B has two different kinds of results:

```text
base result:
  one normal model run or standard evaluation configuration

RSA-boosted result:
  extra test-time compute using Markovian RSA
```

Do not compare these casually.

An RSA-boosted score uses more inference compute.

That compute may be acceptable for:

- math contest problems
- offline code repair
- scientific reasoning
- high-value planning
- batch evaluation

It may be unacceptable for:

- low-latency chat
- real-time agent loops
- cheap background automation
- tool-call-heavy workflows

The correct question:

```text
What is the quality per dollar, per second, and per watt at the required latency?
```

Not:

```text
Which single headline score is highest?
```

---

## 6. Markovian RSA

Zyphra's Markovian RSA is a test-time compute method.

It combines two ideas:

```text
parallel candidate reasoning traces
recursive aggregation
```

with:

```text
fixed-duration reasoning chunks
only tail context carried forward
```

The goal is to keep the context window bounded while allowing extended reasoning.

Simplified flow:

```text
prompt
  -> generate multiple reasoning traces in parallel
  -> keep tail segments
  -> build aggregation prompts from sampled references
  -> generate next round
  -> repeat
```

This is different from one huge chain of thought.

The context does not grow without bound.

That matters because long reasoning traces otherwise collide with context limits and memory costs.

The critical Zyphra claim:

```text
ZAYA1-8B was trained to understand and respond to the Markovian RSA process.
```

They report that applying the same method to another small model produced less uplift.

That is the key systems insight:

```text
The model and inference harness were co-designed.
```

---

## 7. Why this matters for agents

Agent builders should not treat ZAYA1-8B as a default general-purpose agent model.

The benchmark profile says:

```text
strong:
  math
  code
  long-form reasoning
  science-style problem solving

weaker:
  tool calling
  multi-step agent execution
  strict complex instruction following
  general chat style
```

That suggests a routing role:

```text
planner/general assistant:
  use a stronger tool-calling model

specialized math/coding sub-agent:
  consider ZAYA1-8B

expensive reasoning pass:
  consider Markovian RSA if latency and cost allow
```

For OpenClaw-style systems, a realistic use is:

```text
Gateway routes:
  math proof task -> ZAYA1-8B specialist
  code puzzle -> ZAYA1-8B specialist
  tool workflow -> tool-calling model
  app SDK operation -> structured-tool model
```

This matches the principle from Lecture 33:

```text
use the right interface and model for the job
```

---

## 8. Deployment caveat

ZAYA1-8B is not currently a generic drop-in for standard vLLM.

The Hugging Face model card recommends Zyphra's fork:

```bash
pip install "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1"
```

For Transformers usage, it also recommends Zyphra's fork:

```bash
pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1"
```

Example vLLM serve command from the model card:

```bash
vllm serve Zyphra/ZAYA1-8B --port 8010 \
  --mamba-cache-dtype float32 --dtype bfloat16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser zaya_xml
```

This matters operationally.

A model that requires a forked runtime has extra deployment risk:

- upgrade lag
- plugin compatibility issues
- serving bugs
- security patch delay
- harder reproducibility
- limited support in existing inference clusters

That does not mean "do not use it."

It means benchmark the model and the runtime together.

---

## 9. Hardware engineer view

ZAYA1-8B is useful for thinking about model efficiency.

Key hardware questions:

```text
MoE routing:
  Are experts balanced or do hot experts bottleneck?

Active parameter count:
  Does low active compute translate into lower latency on your hardware?

Memory:
  Are all experts resident in memory even if only a subset is active?

Attention:
  Does CCA require custom kernels or forked runtime support?

Batching:
  Does Markovian RSA parallel trace generation batch efficiently?

Interconnect:
  How does expert parallelism behave across GPUs?

AMD stack:
  Which parts rely on custom Zyphra infrastructure versus upstream ROCm?
```

The model is small in active compute.

It is not necessarily trivial to serve optimally.

MoE models often trade dense compute for routing, memory residency, batching, and expert-placement complexity.

---

## 10. How to evaluate it for OpenClaw

Do not evaluate ZAYA1-8B with a generic chat benchmark first.

Evaluate the role you would actually use it for.

Suggested task buckets:

```text
math:
  contest-style reasoning
  symbolic manipulation
  proof sketching

coding:
  algorithmic coding
  bug localization
  test repair
  code explanation

agentic:
  function calling
  structured JSON calls
  multi-step tool workflows
  instruction following under constraints

systems:
  latency
  throughput
  memory footprint
  runtime stability
  forked vLLM maintenance burden
```

Run against at least one baseline:

```text
Qwen small reasoning model
Gemma small model
current OpenClaw default model
larger hosted model for quality ceiling
```

Then decide:

```text
Use as specialist:
Use as default:
Avoid for:
Need RSA for:
Need stronger tool model for:
```

---

## 11. Evaluation checklist

For any benchmark result, record:

```text
Model:
Runtime:
Commit or model revision:
Base or RSA:
RSA token budget:
Hardware:
Prompt set:
Tool availability:
Temperature:
Latency:
Cost:
Pass rate:
Failure examples:
```

For agent workflows, include:

```text
tool-call validity
schema adherence
idempotency behavior
refusal/safety behavior
recovery from tool errors
final answer correctness
```

Use Lecture 39's skill eval approach if the model is meant to execute a skill.

Use Lecture 37's trace approach if you are making performance claims.

---

## Mini-lab: ZAYA1-8B routing decision

Design a routing evaluation for an OpenClaw-like gateway.

Compare:

```text
ZAYA1-8B
current default agent model
one stronger hosted reasoning model
one stronger tool-calling model
```

Use four task groups:

```text
math reasoning
coding/debugging
structured tool use
general assistant/chat
```

For each task group, report:

```text
quality
latency
cost
tool-call validity
failure modes
whether RSA/test-time compute was used
```

Final decision:

```text
ZAYA1-8B should be routed to:
ZAYA1-8B should not be routed to:
RSA is justified when:
RSA is too expensive when:
Runtime blockers:
```

---

## Key takeaways

- ZAYA1-8B is best understood as a small active-parameter MoE reasoning specialist.
- The headline strength is math and coding, not general agent execution.
- Active parameter count and total parameter count are different deployment concepts.
- AMD end-to-end training is strategically important for hardware ecosystem diversity.
- Markovian RSA is a test-time compute harness that keeps reasoning context bounded.
- Base scores and RSA-boosted scores represent different compute budgets.
- Zyphra reports weak relative scores on agentic benchmarks such as BFCL-v4 and tau2 compared with stronger tool-use models.
- The model currently requires Zyphra runtime forks for proper local deployment.
- Use it as a routed specialist only after measuring quality, latency, runtime stability, and tool-call behavior on your real workload.

---

## References

- Zyphra, "ZAYA1-8B: Frontier intelligence density, trained on AMD": [https://www.zyphra.com/post/zaya1-8b](https://www.zyphra.com/post/zaya1-8b)
- ZAYA1-8B Hugging Face model card: [https://huggingface.co/Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B)
- Firethering summary: [https://firethering.com/zaya1-8b-open-source-math-coding-model/](https://firethering.com/zaya1-8b-open-source-math-coding-model/)
- Zyphra vLLM fork: [https://github.com/Zyphra/vllm](https://github.com/Zyphra/vllm)
- Lecture 33 - Structured Tools Beat Computer Use: [Lecture-33.md](Lecture-33.md)
- Lecture 37 - TraceLens: [Lecture-37.md](Lecture-37.md)
- Lecture 39 - Agent Skills Eval: [Lecture-39.md](Lecture-39.md)

---

*Next: [Lecture 41 - OpenClaw Threat Model](Lecture-41.md)*
