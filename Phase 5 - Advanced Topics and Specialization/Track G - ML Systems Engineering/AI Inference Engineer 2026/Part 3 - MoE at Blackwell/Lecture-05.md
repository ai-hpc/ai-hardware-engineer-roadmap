# Part 3 · Lecture 05 — Speculation, constraints, and useful output cost

A production MoE deployment combines model correctness, routing, cache ownership, scheduling, and a workload-specific quality target. This lesson explains two features that are often presented as free speedups: speculative decoding and grammar constraints. Both need precise contracts and measurements.

## Speculative decoding preserves a target policy only with the right algorithm

A draft mechanism proposes tokens cheaply; the target model verifies several positions in one pass. A separate small model, an MTP head, or another proposal method can supply drafts. Draft work, verification, rejected tokens, cache updates, and scheduling all have costs.

For greedy decoding, accept the consecutive draft prefix whose tokens match the target's greedy choices; handle the first mismatch and cache rollback correctly. For stochastic decoding, merely accepting “tokens the target also likes” changes the distribution. In the classical [speculative decoding algorithm](https://arxiv.org/abs/2211.17192), a proposed token x from draft distribution q is accepted with `min(1, p(x)/q(x))`, where p is the target distribution under the same history and policy. Rejection samples from a normalized positive residual `(p-q)+`. Zero probabilities and all-accepted extra-token handling matter.

A two-token example makes this visible: `p=[0.8,0.2]`, `q=[0.5,0.5]`. Draft token 0 is always accepted; token 1 is accepted with probability 0.4. Accepted probability masses are `[0.5,0.2]`; rejection mass 0.3 returns token 0, yielding `[0.8,0.2]`. Accepting every draft would incorrectly yield `[0.5,0.5]`.

Finite precision, different logits processors, constrained grammars, temperature/top-p ordering, and nondeterministic kernels complicate equivalence. State whether the implementation promises greedy equivalence, target-distribution equivalence, or a quality-tested approximation. Identical random seeds do not establish that contract.

## Measure accepted work and total work

Let a speculative iteration cost `Tdraft + Tverify + Tstate` and emit an average E useful tokens. Its effective time/token is approximately that total divided by E, before service queueing. Compare with the same target policy without speculation at the same workload and concurrency. Acceptance rate alone can look good while verification shapes or memory pressure lose performance.

DeepSeek's MTP architecture can support drafting in compatible runtimes. It is not zero-cost and a fixed k does not guarantee a fixed speedup. Sweep draft depth, prompt class, output style, and concurrency. Retain rejected-token/cache-correctness tests and count all allocated resources.

## Grammar constraints change the sampling policy

A grammar or JSON schema can mask invalid next tokens and renormalize allowed probability mass. It can guarantee a syntactic property supported by the grammar implementation; it does not guarantee the correct tool, truthful values, authorization, or successful execution. A constraint also changes the unconstrained model distribution. Evaluate against the intended **constrained** policy when combining it with speculation.

Schema compilation, mask construction, CPU scheduling, vocabulary size, recursion, and caching affect overhead. There is no universal 5% overhead. Use the runtime's pinned structured-output API and [XGrammar documentation](https://xgrammar.mlc.ai/docs/) for its accepted schema/features. Validate returned JSON against the application schema again and apply semantic checks before executing actions.

## Build a small production experiment matrix

Choose one supported precision/runtime/topology combination first. Record model/tokenizer/config revisions, expert mapping and communicator groups, KV format, batch/token policy, cache identity, and hardware/software versions. Then compare baseline, speculation, constraints, and their combination with otherwise matched settings.

Score valid output, correct tool/arguments, useful task completion, TTFT, individual ITL tails, output throughput, failures/retries, and per-rank memory/load. A structurally valid answer that calls the wrong tool is a failure. A high raw token rate with many invalid/rejected responses is not high useful throughput.

Use actual paid cost divided by useful delivered outputs. Include draft workers, idle reserved ranks, retries, transfer/CPU services, and warmup where relevant. No dollar price or expected 13,000-token/s replica result is assumed here; obtain current prices and run a reproducible benchmark for the chosen deployment.

**Exit artifact.** A report containing the CPU route/attention checks, pinned configuration, quality results, actual resource allocation, raw request measurements, and a failure-recovery case. A documented limitation or a baseline that wins is a valid engineering result.

Continue to [Part 4 — Optimizing a Real Engine](../Part%204%20-%20Optimizing%20a%20Real%20Engine/README.md), or use the [course overview](../README.md) to choose the next workload track.
