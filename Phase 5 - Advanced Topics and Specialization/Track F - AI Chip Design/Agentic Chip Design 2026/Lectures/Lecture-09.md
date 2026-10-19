# Lecture 09 — Capstone: a reproducible RTL repair study

[Course guide](../Guide.md) · [Previous](Lecture-08.md) · [Course guide](../Guide.md)

Your capstone is a bounded, auditable design experiment. Start with the [streaming incrementer lab](../../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md): its specification, independent scoreboard, real simulator execution, and known detectable faults provide a small reference point. A model API is optional for the first pass; a controlled manual repair demonstrates the evaluation machinery without claiming model performance.

## Freeze the task

Use the exact port declaration and reset/transfer/saturation rules from the lab. Keep the contract and checker versioned independently of candidate code. Record Icarus version, source/checker hashes, seeds, operating environment, and the maximum number of repair rounds. State which tests are visible. If all tests are public and used during debugging, report visible-test results only.

First run the unmodified baseline and all three injected faults. The baseline must pass and each mutation must fail in its expected category. If not, repair the environment or checker before comparing generators. Keep these results as evidence that at least the known failure modes are observable.

## Compare two explicitly named conditions

Condition A gets one proposal. Condition B gets one proposal plus at most three repairs with visible compiler/simulator feedback. Use the same specification, model revision, sampling settings, and checker for a model-based study. If using human-written or preselected candidates, label those conditions accordingly; do not call them pass@1 results for an LLM.

Archive every attempt. A failed compile, simulator timeout, incorrect value, stalled-output violation, reset error, or exhausted budget is a real result. Do not silently replace a poor initial proposal with a later one when computing first-attempt success.

## Inspect the accepted candidate

Read the state-update condition, output stability, simultaneous consume/replace behavior, and saturation arithmetic. Explain why the fixed checker covers those requirements and identify what remains untested. Generate a waveform for one failing and one passing trace if useful. Simulation acceptance is the required milestone; synthesis and physical metrics are optional extensions with separate evidence.

For an actual accelerator extension, replace the incrementer with a small command queue or length counter. Specify depth, reset, wrap/overflow, invalid commands, and consumer stalls. Build a new reference/checker before asking an agent to modify the implementation. The [compiler tiling lab](../../Labs/Tiling-and-Dataflow.md) offers numerical contracts for a later datapath exercise, but its Python model is not an RTL implementation.

## Submit a compact evidence bundle

Include the specification, candidates, attempt manifests, fixed checker reference/hash, tool transcripts, baseline and fault-detection reports, and a result table. Add a short engineering explanation answering:

1. Which exact failure did the checker detect, and at what first divergent event?
2. Which repairs changed implementation while preserving the contract?
3. How did first-attempt success differ from final success under the budget?
4. What did all failed attempts cost in time and resources?
5. Which claims remain unsupported by simulation alone?

**Acceptance:** another learner can reproduce the baseline and negative cases, follow the attempt history, and distinguish measured outcomes from hypothetical PPA or model claims. A candidate that passes only because its checker or expected behavior was changed does not satisfy the original task.

**Answer to the final self-check:** a successful repair study shows that a particular workflow found a candidate satisfying the chosen checks under the recorded budget. It does not establish general RTL competence, exhaustive correctness, timing closure, or readiness for tapeout. Those require additional tasks and evidence.

Return to the [course guide](../Guide.md) or continue into the [programmable VLA accelerator course](../../Programmable%20VLA%20Inference%20Accelerator/README.md), carrying the same separation of specification, implementation, checker, and measurement.
