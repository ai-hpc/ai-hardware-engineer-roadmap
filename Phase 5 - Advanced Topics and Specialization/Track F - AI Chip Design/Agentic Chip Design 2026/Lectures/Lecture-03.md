# Lecture 03 — Generate RTL from a bounded contract

[Course guide](../Guide.md) · [Previous](Lecture-02.md) · [Next: Lecture 04](Lecture-04.md)

A model can draft a module only as precisely as its task is defined. The goal of this lesson is a candidate that can be judged independently, not a prompt that produces convincing commentary. The [streaming lab](../../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md) supplies a reference design and checker so you can learn this process without a paid API or a new hardware board.

## Specify observable behavior

A useful generation request includes the exact module interface and width, unsigned arithmetic, saturation at 255, a one-entry capacity, transfer timing, reset behavior, and required behavior under backpressure. It should also say which file may be edited and which simulator/language subset is supported.

For example, ask for a `stream_increment` module implementing the lab's contract. Include its actual port declaration rather than letting the model invent different names. State that an accepted input must produce `min(input+1,255)`, that stalled output must remain stable, and that simultaneous consume/replace is permitted. Preserve the independently written contract and testbench outside the candidate file.

The arithmetic width matters. In an 8-bit register, adding one to 255 ordinarily wraps to zero if truncated. Saturating behavior requires a comparison or widened arithmetic and a clamp. Ask the generator to explain that boundary, then verify it with input 255; the explanation itself is not evidence.

## Control the generation experiment

Record the model identifier/revision, sampling settings, prompt, available context, and whether a reference implementation was visible. A model shown the complete answer is performing a different task from specification-to-RTL generation. Keep retrieval experiments separate: supplying a relevant interface example may help, but exposing a benchmark solution invalidates claims about unseen-task performance.

Generate one candidate first. Save the exact output before any manual edits. Extract the candidate source using a deterministic rule, and classify malformed output separately from HDL compile errors. Run a fixed checker and retain its exit status and transcript.

For a no-API exercise, copy the lab source to a temporary working directory and study one of the deliberately injected faults from Lecture 05. Label it a controlled repair exercise rather than a generated-model result. You can learn the interface, trace, and repair workflow without inventing a model score.

## Keep the specification stable during repair

A candidate that fails saturation must change its implementation. Changing the expected value to zero changes the task. Similarly, reducing the test to continuously ready transfers hides backpressure errors. New tests can strengthen the checker, but a revised checker creates a new evaluation version; rerun every compared candidate against it.

**Exercise:** a model returns two files, a module and a testbench that checks only five increasing inputs. Should those five tests determine acceptance? **Answer:** no. They can be additional visible smoke tests, but acceptance uses the independent contract and checker. A generator and its self-written checker can share the same wrong assumption.

Your artifact is `candidate.sv` plus a manifest recording the prompt, revision, generation settings, and source hash. State whether the attempt was generated, manually written, or a controlled mutation. Continue to [Lecture 04](Lecture-04.md) to define the score before comparing attempts.
