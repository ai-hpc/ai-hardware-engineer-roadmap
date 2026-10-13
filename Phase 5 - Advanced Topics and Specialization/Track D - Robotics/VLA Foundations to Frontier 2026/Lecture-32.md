# Lecture 32 — Capstone II: A Professional VLA Research and Deployment Review

[Previous](Lecture-31.md) · [Course overview](README.md) · [Overview](README.md)

**Prerequisites:** Lecture 31, at least one model-family lecture from 25–29, and Lecture 30.

**Effort:** 2 h proposal, 12–24 h implementation and evaluation, 4–6 h report and defense; set separate GPU-hour limits.

**Evidence cutoff:** October 13, 2026. Numerical examples below are illustrative unless explicitly identified as executed lab results.

## Make one defensible engineering claim

A professional project is not a collection of impressive model names. It answers a specific question under explicit conditions. Examples include whether asynchronous scheduling improves task completion under measured inference delay, whether a supported quantization path preserves success while reducing tail latency, or whether a data-mixture change improves one defined transfer axis.

**Outcome:** produce a reviewable VLA subsystem and a report that separates established results, uncertainty, and remaining work. A collaborator should be able to reproduce the central result or identify precisely why access prevents it.

Choose a project scope before choosing a frontier model. A CPU-only project can investigate data integrity, gradients, scheduling, or statistical methodology. A GPU project can evaluate adaptation and simulation. A physical-robot project additionally needs the selected device's supervised integration process and suitable control expertise. Grade the quality of evidence within the declared route; do not claim physical deployment competence from a CPU fixture.

## Write the proposal as a falsifiable comparison

State the reference, candidate, task distribution, primary metric, supporting metrics, and stopping rule. For example: “On a fixed 12-task simulated manipulation suite, evaluate whether the candidate scheduler improves task-macro success at 80 ms injected delay while preserving action continuity and bounding observation age.” The number of tasks and delay are project choices, not recommended universal standards.

Predeclare which outcomes would reject the hypothesis. A candidate that reduces mean inference latency but increases P99 observation age has not necessarily improved deployment. A candidate that succeeds more often only after doubling the time horizon changes the task budget. A candidate trained on extra demonstrations answers a different question from an architecture-only comparison unless that additional data is explicitly part of the treatment.

Estimate uncertainty and cost before allocating a large run. Use a pilot to measure episode duration, variance, failure prevalence, GPU memory, and logging overhead. Choose task and seed counts based on the claim's precision needs. Do not pretend a tiny pilot establishes final statistical power, and do not repeatedly stop as soon as a favorable result appears without a valid sequential procedure.

## Select an artifact with a usable contract

Use the [frontier register](Frontier-Register.md) and [source dossiers](Sources-Frontier.md) to distinguish announced capability from accessible execution. A public paper, public code, downloadable weights, a hosted API, and a partner agreement are different access conditions. The newest announcement may be inappropriate for an experiment that requires local action tensors.

Archive a manifest containing:

| Component | Evidence to retain |
|---|---|
| Model | Exact checkpoint revision/hash, architecture, action head, base-model identity |
| Data | Dataset revision, transformations, normalizer, split/group definition, allowed use |
| Software | Code commit, dependency lock, runtime/compiler versions, launch command |
| Robot/simulator | Embodiment, controller, camera calibration, timestep, backend, asset revisions |
| Evaluation | Task list, starts/seeds, horizon, success definition, exclusions, paired outcomes |
| Deployment | Hardware, power/thermal mode, competing workloads, timing and intervention traces |

A date-stamped README is a helpful index, but it does not replace hashes and configuration files. For hosted models, record the exact service identifier and request/response metadata allowed by its terms; weight hashes may not be available, so state that reproducibility limit.

## Build an experiment with three layers

First validate contracts: image preprocessing, language tokenization, state order, action normalization, frames, masks, resets, and time alignment. Include a negative case for each changed boundary. A shape-compatible tensor can still have the wrong physical meaning.

Second establish behavior: offline physical-action comparisons and paired closed-loop task outcomes. Retain failure videos or traces with task/seed IDs. Inspect a random sample of successes as well as failures to catch incorrect success predicates. Report interventions and excluded trials instead of letting them disappear into an aggregate.

Third measure the system: full observation-to-command timing, memory peak, sustained operation, and concurrency. A policy benchmark that omits the simulator or sensor pipeline may still be useful, but its title should identify that narrower scope. Separate compile/build time, cold start, warm execution, and recovery from faults.

For quantization projects, record which tensors changed precision and whether measured kernels implement that precision. For RL projects, record the actual objective, data source, critic inputs, intervention policy, and whether the method uses exact or surrogate likelihoods. For world-model projects, evaluate behavior as well as video quality. These distinctions connect the earlier lectures to daily engineering decisions.

## Analyze negative and inconclusive results

Suppose a candidate improves success by three points but its interval spans -4 to +10. The estimate is encouraging but does not establish improvement. Inspect variance sources and collect additional evidence only under a justified follow-up plan. Do not erase the original result from the record.

If the candidate fails, localize the failure: artifact incompatibility, data issue, optimizer instability, control contract, distribution shift, or resource limit. An experiment that falsifies a plausible hypothesis and supplies a reusable diagnostic tool is valuable. A report that conceals a failed run behind a polished demonstration is not.

Ablations should discriminate explanations. If a data change improves transfer, compare against a matched-data-volume baseline. If a scheduler improves success, hold inference output and injected delays fixed where possible. If language conditioning appears useful, test instruction swaps on matched scenes. The intervention should change the hypothesized cause while keeping competing explanations controlled.

## Deliver the project for review

Submit a short problem statement, architecture/control diagram, artifact manifest, executable reproduction entry point, raw result tables, analysis code, representative traces, and a limitations section. Keep large checkpoints and private data outside the repository unless their distribution is explicitly permitted; include retrieval instructions and hashes where available.

The report should lead with the actual finding. State hardware and scope next, then the evidence and uncertainty. Distinguish local measurements from upstream numbers and analytical estimates. For current-model claims, cite primary sources and preserve the October 13, 2026 snapshot; later work belongs in a separately dated update.

Use this rubric, scored out of 100:

| Dimension | Weight | What strong evidence looks like |
|---|---:|---|
| Contract correctness | 25 | Verified transforms, units, normalization, resets, timing, and negative cases |
| Experimental reasoning | 20 | Falsifiable question, controlled comparison, justified estimand |
| Measurement | 20 | Raw paired outcomes, uncertainty, tails, resource and intervention records |
| Reproduction | 20 | Pinned artifacts, executable commands, dependencies, accessible fixtures |
| Transfer and critique | 15 | New condition tested, failure explanation, honest access/scope limits |

A score cannot compensate for fabricated measurements, hidden test leakage, or a broken action contract. Repair those before considering the project complete. An educational passing threshold is 75/100 with no missing contract or reproducibility evidence; it is not professional certification.

## Final defense

Explain your model's lineage, action representation, loss, data mixture, control boundary, compute bottleneck, evaluation protocol, and access terms without relying on a model-card slogan. Demonstrate one intentionally failing input and trace its handling. Then identify the smallest next experiment that could change your conclusion.

A successful defense answers “what did you establish?” precisely. It might establish a robust CPU scheduler, a reproducible simulated adaptation gain, or a measured device-specific latency improvement. It should never imply that one such result settles general robotic intelligence.

---

[Previous](Lecture-31.md) · [Lab guide](Lab-Guide.md) · [Overview](README.md)
