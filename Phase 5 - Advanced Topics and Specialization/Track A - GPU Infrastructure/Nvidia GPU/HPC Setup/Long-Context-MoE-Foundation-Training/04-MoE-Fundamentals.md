# Module 04 — MoE Routing, Capacity and Learning

**Parent:** [Course](README.md) · **Runnable experiment:** [CPU training lab](CPU-Training-Lab.md)

**Learn:** trace a token through top-k routing, distinguish assignment statistics from token statistics, and check that an MoE actually learns. **Time:** two 60-minute sessions.

## What the experts change

A nongated expert with input/output width `H` and intermediate width `F` has approximately `2HF` weights and `4HF` forward FLOPs per token, excluding biases and activation work. `E` such experts store approximately `2EHF` weights. Selecting `k` experts uses approximately `4kHF` expert FLOPs per token, plus routing and any shared expert.

Thus expert storage scales with `E`, selected arithmetic with `k`. This does not make whole-model parameter count exactly `E` times a dense model, because attention and other shared layers remain. Nor does sparse activation imply all inactive expert weights disappear from memory. Gated experts have three matrices; substitute their actual dimensions.

For softmax token-choice routing:

```text
logits = x @ W_router
p = softmax(logits)
chosen = topk(p, k)
w = p[chosen] / sum(p[chosen])
y = sum(w[j] * expert[chosen[j]](x))
```

This is the teaching implementation's contract. Other models use sigmoid scores, different normalization, group selection, shared experts or routing biases. Inspect the source before transferring conclusions. Current Megatron supports several balancing and dispatcher choices, including strategies without an auxiliary balancing loss. [Megatron MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html).

## Gradients and a surprising top-one case

Top-k indices change discontinuously at ties. Within a region where the chosen indices stay fixed, gradients flow through selected weights and expert networks. Finite-difference checks must avoid a perturbation that changes selection if they are meant to check that local derivative.

With `k=1` and selected-weight renormalization, the sole weight is `p/p=1`. The task loss has no ordinary gradient to the router through this weight. The CPU lab demonstrates this. A top-one router that learns from task loss may use unnormalized probabilities, another estimator or another design; do not assume this normalized top-two example transfers unchanged.

## Define balancing statistics precisely

For `T` tokens and `k` assignments each, define:

```text
f_i = assignments to expert i / (T*k)
P_i = mean softmax probability for expert i over T tokens
L_aux = alpha * E * sum(f_i * P_i)
```

The counts are discrete; this implementation differentiates through `P`, not through top-k counts. Both `sum(f)` and `sum(P)` equal one. Uniform `f=P=1/E` gives `L_aux=alpha`. If you instead divide assignments by `T`, the sum of `f` is `k` and the balanced loss becomes `alpha*k`. Either convention requires consistent scaling; mixing them produces misleading “healthy loss” targets.

Router z-loss penalizes the squared log-sum-exp of logits. Its role is distinct from balancing assignment counts. The z-loss design is associated with **ST-MoE**, while the Switch paper is a primary reference for sparse top-one routing and balancing. [ST-MoE](https://arxiv.org/abs/2202.08906), [Switch Transformer](https://arxiv.org/abs/2101.03961).

## Capacity needs three denominators

In this lab, expert capacity is `ceil(factor*T*k/E)`. Use `ceil`, not `int(value+1)`, which is too large when `value` is already an integer. Within each expert, keep the highest selected gate weights, with stable ordering for ties. Dropped assignments contribute zero and the remaining gates are **not renormalized**. The MoE output itself is not a residual block; a surrounding architecture would add its residual separately.

For four tokens, four experts, top-two and factor one, capacity is two. If every token selects experts zero and one, eight assignments compete for four admitted slots. With identical scores and stable ordering, the first two tokens retain both experts and the last two retain none:

- Dropped assignments: `4/8 = 50%`.
- Tokens with any dropped assignment: `2/4 = 50%`.
- Tokens with no admitted expert: `2/4 = 50%`.

These quantities happen to match here. For assignments `[0,1]` and `[0,2]` with capacity one, they are 25%, 50% and 0%. Reporting “dropped tokens” without its denominator hides this distinction. Capacity-free/dropless implementations make another memory/compute trade rather than enforcing a universal acceptable drop rate.

## Run a real learning ablation

The [lab](CPU-Training-Lab.md) trains four small nonlinear experts on a four-dimensional regression task, with separate generated training and validation samples. It compares `alpha=0`, `.01` and `.5` from the same initialization and saves/reloads actual weights. It also checks sparse dispatch against an expensive all-expert oracle, including parameter and input gradients.

The recorded 120-step CPU run reduced validation MSE from `0.63823` to approximately `0.01774`, `0.01778` and `0.01936`, respectively. The zero-auxiliary-loss run **did not collapse**. Larger alpha did not automatically improve balance or validation. These are observations about this task and seed, not predictions for a foundation model.

Repeat with capacity pressure and inspect both error and coverage. Report the data/initialization seeds and the whole ablation, including negative results. Do not select a coefficient using the final test set. Balanced routing alone cannot show that experts learned useful specialization; add task quality, domain slices and relevant baselines.

**Exit check:** explain why equal average assignment counts can coexist with poor predictions, and why low dropped-assignment fraction can hide complete FFN omission for particular tokens. Your answer should refer to actual saved rows and one hand-worked counterexample.

[Next: dispatch and distributed execution](05-MoE-Systems-Infrastructure.md)

### Check capacity admission by hand

Take two scalar experts that always return 10 and 20, and three inputs `[-2, 0, 2]`. Let router logits be `[x, -x]`, select both experts, and set capacity factor `.5`. Capacity is `ceil(.5*3*2/2)=2`. Each expert keeps its two largest normalized gates; the middle token has gates `[.5,.5]` and is admitted to both. With `p=1/(1+exp(-4))`, outputs are `[20*p, 15, 10*p]`. The first and last tokens lose one assignment each; their surviving gate remains `p`, not one. Thus assignment-drop fraction is `2/6`, any-drop fraction is `2/3`, and no-expert fraction is zero. This independently calculated example is an executable boundary test in the CPU lab.
