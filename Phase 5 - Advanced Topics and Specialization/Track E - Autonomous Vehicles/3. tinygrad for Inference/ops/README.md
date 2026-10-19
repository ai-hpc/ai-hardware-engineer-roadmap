# Operator lessons: values, axes and layout

These lessons accompany the executed [tinygrad 0.14.0 CPU projects](../projects/README.md). The three categories help you reason about a tensor expression; they are not the compiler's complete instruction set.

| Lesson | Hand result | Executed counterpart |
|---|---|---|
| [Elementwise operations](elementwise.md) | Correct SwiGLU at zero/negative gate | Custom-operations lesson |
| [Reductions](reduce.md) | Matmul's inner-axis sum, RMSNorm and causal softmax | Operations, autograd and custom-operations lessons |
| [Movement/layout](movement.md) | Reshape/transpose indexing and broadcast gradients | Basics/operations plus source walkthrough |

For each operation, write shape, dtype, axes, special-value domain and expected output before execution. Then introduce a counterexample: wrong gate factor, wrong reduction axis, future-token leakage or wrong row stride. A test should fail for a concrete semantic reason, not simply echo the implementation.

The independent [scalar reference](../labs/reference.py) and seven standard-library tests are available without tinygrad. Actual framework results additionally require the pinned optional environment.
