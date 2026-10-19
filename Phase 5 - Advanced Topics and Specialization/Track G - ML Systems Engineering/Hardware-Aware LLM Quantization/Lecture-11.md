# Module 11 — Precision Allocation: A Surrogate Is Not the Final Verdict

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-10.md) | **Next:** [Next lesson](Lecture-12.md)

Choose a precision for each compatible group while satisfying quality, latency
and memory requirements. An allocation solver is useful when its assumptions are
explicit and its proposed configuration is evaluated as a whole.

## 1. State the optimization

Let group i have candidate formats F_i, actual stored bytes b_i(f), an estimated
traffic contribution t_i(f), and measured development-set fidelity/task changes.
Hardware/runtime support filters candidates before search. Native formats are
useful candidates, but a measured fused weight-only kernel can be eligible too.

A simple surrogate minimizes `sum_i t_i(f_i)` subject to an additive fidelity
budget `sum_i cost_i(f_i)≤epsilon` and a memory budget. That surrogate assumes
approximately constant effective bandwidth and additive error. A deployment
objective should ultimately use measured latency/throughput, task constraints,
state/workspace memory and, for speculation, **whole-cycle** cost.

## 2. A small worked problem with an exact answer

The following is an **invented arithmetic fixture**, not a model benchmark. Each
group can stay wide or become small. The total error budget is 6 units.

| Group | Bytes saved if small | Added surrogate error | Savings/error |
|---|---:|---:|---:|
| A | 6 | 4 | 1.5 |
| B | 4 | 3 | 1.333 |
| C | 4 | 3 | 1.333 |

Greedy chooses A and cannot add B or C: savings 6. Enumeration chooses B+C:
savings **8**, within the same budget. A high savings/error ratio does not make
this discrete knapsack greedy-optimal. Use the [CPU lab](CPU-Quantization-Lab.md)
to reproduce this counterexample.

For 8 groups and 3 formats, `3^8=6561` assignments are cheap to enumerate **after**
measuring their surrogate inputs. Measuring 6561 real model configurations can be
expensive. The solver's computation and the experiment's computation are distinct.

## 3. Reproducible enumeration sketch

```python
from itertools import product

def choose(groups, error_budget, byte_budget):
    # groups: sequence of candidate lists; each candidate has format, bytes, error.
    # Every group must supply at least one valid candidate; errors are surrogate costs.
    feasible = []
    for allocation in product(*groups):
        size = sum(c['bytes'] for c in allocation)
        error = sum(c['error'] for c in allocation)
        if size <= byte_budget and error <= error_budget:
            feasible.append((size, error, tuple(c['format'] for c in allocation)))
    if not feasible:
        raise ValueError('no feasible allocation; revise constraints or candidates')
    return min(feasible)  # declared lexicographic objective and tie rule
```

A production version validates units, finite values, compatibility, padding,
activation dependencies and resident state. Report infeasibility rather than
silently exceeding a quality or memory budget.

## 4. Measure interactions

Compare joint error for groups A+B with the sum of individually measured errors.
Errors may reinforce or cancel; KL is not additive across separately perturbed
layers. The KL chain rule over sequence positions does not establish layerwise
additivity. Ratios are unstable near zero, so report the absolute residual too:

```text
interaction = measured_joint_error - (solo_A_error + solo_B_error)
```

Evaluate multiple pairs, especially in the same attention/recurrent block, then
evaluate the selected **complete** model. Passing a few pairwise checks does not
prove that a many-layer combination passes. Cancellation on development data does
not authorize spending the apparent surplus without independent validation.

## 5. Set a product budget instead of a universal KL threshold

KL does not uniquely determine argmax agreement or task accuracy. Choose tolerances
from development data, decision costs and critical slices. Draw an empirical
Pareto plot of quality versus measured latency/memory. A visual knee is helpful,
but it does not override a hard task constraint.

Keep target fidelity and drafter efficiency separate. The DSpark case cannot
supply target-layer KL costs: its published acceptance ablation changed the drafter.
There are no measured “0.038 versus 0.093 nats” target costs in this course.

## Independent task and answer guidance

Add an activation-format dependency: selecting an FP4 tensor-core kernel requires
an eligible quantized activation path. Show an allocation that is numerically
within budget but fails compatibility, and reject it before runtime. Next add
one measured joint-error term that changes the winning policy.

**Completion:** submit candidate provenance, an exact small search, the greedy
counterexample, a joint-error measurement and final held-out evaluation. A
supported negative result passes: the model may have no useful lower-precision
allocation under the selected constraints.
