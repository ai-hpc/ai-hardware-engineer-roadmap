# Lecture 20 — System identification and transfer experiments

**Prerequisites:** Lecture 19 and basic control. **Time:** 3–4 hours; begin entirely in simulation. **Outcome:** identify which dynamics mismatch matters before broadening randomization.

## Model a mismatch you can test

List actuator response, friction, mass/inertia, contact, sensor noise/bias, time delay, and saturation. For each, specify an observable signal and a small experiment. Randomizing everything over arbitrary ranges can make training harder while failing to cover the actual deployment error.

Use the simulator as a controlled identification exercise: hide one parameter from yourself, record an input/output response, estimate it, then compare with the configured truth.

## Worked actuator model

A simple first-order actuator response can be written:

```text
v[k+1] = a * v[k] + (1-a) * u[k]
a = exp(-dt / tau)
```

This is an illustrative discrete model, not a universal robot actuator. Fit tau from held-out response data and inspect residuals. A consistent lag may indicate delay rather than a larger time constant; design an input that distinguishes them.

If delay is modeled as an integer number of policy steps, changing policy dt changes its physical duration. Keep delay in seconds in the contract and document rounding when converting to steps.

## Lab

1. Choose one task with a validated nominal controller.
2. Perturb exactly one simulator parameter or delay.
3. Record bounded step or sweep responses and estimate the hidden parameter.
4. Validate the estimate on a new input sequence.
5. Train or adapt using a justified randomization interval.
6. Evaluate nominal, in-range, and out-of-range scenarios separately.
7. Categorize failures by contract, timing, dynamics, or task ambiguity.

Use the existing [sim-to-real lecture](Lecture-12.md) for the broader transfer context. Engine-to-engine comparison is useful but is not hardware validation: two simulators may share or differ in modeling assumptions.

## Decision table

| Observation | Next experiment |
|---|---|
| Nominal performance collapses after randomization | Narrow one range; verify reward/action units |
| Only high-speed motion fails | Test delay, damping, saturation, and timestep |
| Contact transitions fail | Check friction/contact model, geometry, solver, and timestep |
| Exported policy fails but training rollout works | Revisit normalizer, action/frame contract, and timing first |

**Independent exercise:** explain a case where a wider randomization range reduces transfer quality. Show the tradeoff in measured outcomes rather than assuming broader is always better.

## Self-check

- Is domain randomization a proof of robustness? It provides evidence over tested distributions, not all possible conditions.
- Why keep identification data separate from validation? Reusing fitted data can hide model mismatch.
- What can a simulator-only submission claim? It can claim transfer between named simulated conditions, not successful real-robot deployment.

**Exit artifact:** identification record, held-out response error, chosen randomization ranges with reasons, and an evaluation envelope.

[Previous](Lecture-19.md) · [Lab sequence](Lab-Guide.md) · [Next: capstone](Lecture-21.md)
