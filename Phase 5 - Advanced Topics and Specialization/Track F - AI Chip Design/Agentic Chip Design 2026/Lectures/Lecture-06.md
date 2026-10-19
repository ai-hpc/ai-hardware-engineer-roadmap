# Lecture 06 — Build a bounded propose–check–repair loop

[Course guide](../Guide.md) · [Previous](Lecture-05.md) · [Next: Lecture 07](Lecture-07.md)

An agentic design loop needs explicit states and stopping conditions. Otherwise “try again until it works” hides unlimited search, changing tests, and lost failures. The generator can be a human, a local model, or a remote API; the checker should remain the same independent process.

## Define the state machine

```text
specification + frozen checker
    → propose candidate
    → save source and manifest
    → compile
         failure → classify diagnostics → repair, if budget remains
    → simulate independent checker
         failure → retain first counterexample → repair, if budget remains
    → accepted for this checker

Any timeout, exhausted budget, or unrecoverable tool failure → terminal record
```

“Accepted” means this candidate passed this checker, not that its synthesis or timing is valid. Keep those later stages separate. A model's statement that a simulator passed is not a substitute for the tool exit status and a completion record from the actual run.

## Make budgets explicit

For the course experiment, permit one initial proposal and at most three repairs. Set a tool timeout, wall-time budget, and model-token or monetary budget before starting. Check remaining budget before every tool/model call. A timeout consumes an attempt and must remain in the record; it must not restart an unlimited hidden loop.

A compact controller can follow this pseudocode:

```text
for attempt in 0..3:
    candidate = propose(spec, allowed_feedback, remaining_budget)
    archive(candidate, attempt)
    outcome = run_fixed_checker(candidate, timeout)
    archive(outcome, attempt)
    if outcome.functional_pass:
        stop("accepted_for_this_checker")
    if budget_exhausted:
        stop("budget_exhausted")
    allowed_feedback = summarize_first_failure(outcome)
stop("repair_limit")
```

This is architecture pseudocode, not a supplied API integration. The executable simulator/checker is the [RTL lab](../../../../Phase%204%20-%20Track%20A%20-%20Xilinx%20FPGA/1.%20Xilinx%20FPGA%20Development/labs/README.md). To evaluate a newly generated candidate, copy its source and the fixed testbench into an isolated experiment directory, compile with the same Icarus flags shown in the lab, and preserve the original checker hash. The existing mutation runner intentionally operates on the reference plus named faults; it does not pretend to call a model.

## Use feedback with a defined boundary

Feedback can include compiler diagnostics and visible-test counterexamples. It should not reveal held-out answers if those tests are used to claim generalization. The agent may edit only candidate source in the experiment. Changes to the specification, tests, simulator command, or success parser create a different experiment and require rerunning the comparison.

Run tools in disposable working directories and bound resources. This course's subprocess timeouts are operational limits, not a security sandbox for arbitrary hostile generated programs. Select an appropriate isolated execution environment before evaluating untrusted candidates at scale.

**Exercise:** a candidate prints “PASS” and exits before the scoreboard finishes. Should the controller accept it? **Answer:** no. Require the independent checker to complete its protocol, and prevent candidates from modifying or impersonating the checker. Inspect the actual testbench completion conditions.

Deliver a state-transition table, stop reasons, and an attempt log for one controlled repair. Include all failed attempts and explain the final evidence level. Continue to [Lecture 07](Lecture-07.md) before adding optimization goals.
