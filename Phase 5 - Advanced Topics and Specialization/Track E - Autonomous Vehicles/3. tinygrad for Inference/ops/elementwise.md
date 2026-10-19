# Elementwise operations: preserve the function

**Before:** [project setup](../projects/README.md). Elementwise functions transform matching or broadcast-compatible entries. They are often easy to parallelize, but numerical meaning, broadcasting and memory access still matter.

## 1. Work an activation by hand

ReLU is `max(x,0)`. For `[-1,0,2]`, it returns `[0,0,2]`. In the course's fused affine/ReLU, the affine comes first, so the same input produces `[0,1,5]`. Moving a nonlinearity across an affine transform generally changes the function.

Sigmoid is `1/(1+exp(-x))`. A numerically stable scalar implementation evaluates the negative branch using `exp(x)/(1+exp(x))` to avoid overflow for a large negative x. Stable implementation choices should preserve the intended domain and limiting values.

## 2. SwiGLU needs the gate value itself

Given two already-projected branches `value` and `gate`:

```text
SiLU(gate) = gate × sigmoid(gate)
SwiGLU(value,gate) = value × SiLU(gate)
```

For `value=5, gate=0`, the result is 0. The incorrect `value×sigmoid(gate)` yields 2.5. For `value=2, gate=-1`, the correct result is `-2/(1+e)≈-0.537883`.

The executed custom-operations lesson tests both cases and compares all entries with a scalar reference. In a full network the two branches ordinarily come from learned projections, followed by another projection as specified by the architecture. Calling an arbitrary gate product “SwiGLU” does not supply those surrounding layers. [GLU variants paper](https://arxiv.org/abs/2002.05202).

## 3. Distinguish exact and approximate GELU

Exact GELU is `0.5*x*(1+erf(x/sqrt(2)))`. Common tanh or sigmoid approximations are different numerical functions. A compiler or deployment comparison must agree on the selected form and tolerance; visually similar curves do not establish exact parity.

**Exercise:** calculate exact GELU and the sigmoid approximation `x*sigmoid(1.702*x)` for `x=-2,-1,0,1,2` using Python's standard-library `math`. Report maximum absolute error. This is an approximation experiment, not a gradient or backend correctness proof. [Original GELU paper](https://arxiv.org/abs/1606.08415).

## 4. Shape and stochastic semantics belong in the contract

Adding a `[N]` bias to `[B,N]` broadcasts across B. Adding `[B,1]` instead changes every channel in a row. Both may be legal tensor operations while expressing different models. Use a non-square example and explicitly named channel axis.

If dropout is involved, evaluation/training mode and random-number mapping matter. Comparing two implementations with independently generated masks confounds the arithmetic comparison. Use the same mask to test numerical equivalence, then assess the required RNG/distribution behavior separately. The course's default inference examples contain no dropout.

## 5. Fusion is an execution choice

`(x*2+1).relu()` may execute in one kernel, preserving the same tensor-level function. Forcing `.realize()` between operations may materialize buffers. The [compiler lesson](../projects/README.md#3-inspect-compilation-separately) measured this difference on the named CPU runtime; it does not claim every activation chain fuses or that fewer kernels always run faster.

**Completion:** reproduce the zero-gate failure of the wrong SwiGLU formula, pass the actual custom-operations lesson, and explain one broadcasting shape that is legal but wrong for the intended model.
