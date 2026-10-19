# From a sensor reading to one training step

**Purpose:** calculate what a model does, how one parameter update changes it, and how its arrays map to memory. **Before:** [entry diagnostic](Guide.md). **After:** [evaluation and robot learning](02-Evaluation-and-Robot-Learning.md).

For the symbols used below, keep the [notation examples](../../Beginner-Reading-Aid.md#3-read-the-notation-without-changing-its-meaning)
open. You can check the arithmetic with a calculator or Python before learning
matrix-library syntax.

## 1. Numbers need meanings

A scalar is one number: velocity `0.2 m/s`. A vector is an ordered list: `[distance_error_m, velocity_m_s]`. A matrix is a rectangular table: one row per observation, one column per feature. A tensor generalizes arrays to more dimensions; an image batch might be `[batch, channels, height, width]`.

Shape and meaning are separate contracts. A three-number position in metres and a three-number rotation in radians have the same shape and different meanings. Likewise, RGB and BGR images can have identical shapes and different channel order.

For our cart, define `error = goal_position - current_position`. Positive acceleration increases velocity; positive velocity increases position. An input in centimetres must be converted before using a model trained on metres.

**Self-check:** the goal is at 0 m and the cart is at +0.5 m. What is the error? **Answer:** −0.5 m. A sign error here can train a convincing policy that moves away from the goal.

## 2. A weighted sum is already a model

The model uses two inputs to predict acceleration:

```text
acceleration = w_error × error + w_velocity × velocity + bias
```

Use hand-chosen parameters `w_error = 0.8 s⁻²`, `w_velocity = -0.6 s⁻¹`, and `bias = 0 m/s²`. For `error = 0.5 m` and `velocity = 0.2 m/s`:

```text
acceleration = 0.8 × 0.5 - 0.6 × 0.2 + 0 = 0.28 m/s²
```

The first term pulls toward the goal; the second damps motion. These parameters are our synthetic teacher's rule. Training will estimate a similar rule from examples, without being given the parameters.

An acceleration is not a new position or velocity. For a `0.1 s` step starting
at velocity `0.2 m/s`, this `0.28 m/s²` action increases velocity to
`0.2 + 0.28*0.1 = 0.228 m/s`. The lab then advances position using that new
velocity, by `0.0228 m`. Units make the two operations distinct.

For a batch, use matrix multiplication:

```text
X = [[ 0.5,  0.2],       W = [[ 0.8],       b = [0.0]
     [-0.5, -0.2]]            [-0.6]]

X shape [2, 2] @ W shape [2, 1] + b shape [1]
    -> prediction shape [2, 1] = [[0.28], [-0.28]]
```

The inner dimensions must agree. The bias is added to each row. Libraries may store a layer's weight transposed internally; for example, a layer's documented storage shape need not match the mathematical convention above.

**Guided variation:** change only the second observation's velocity to `+0.2`. Its prediction becomes `−0.52 m/s²`: the cart needs stronger braking because it moves away from the goal.

## 3. The loss tells training what to improve

A **parameter** is a number the optimizer changes. A **prediction** is the result of applying the model. A **label** is the desired output in the dataset. A **loss** turns prediction errors into a number to minimize.

For a single regression example, squared error is:

```text
L = (prediction - target)²
```

For N examples, mean squared error (MSE) averages these squared errors. Its square root, RMSE, has the same units as the target. A lower training MSE means closer agreement with training labels; it does not by itself mean a useful or safe robot.

Here is one complete update using the simpler model `prediction = w × x`, with `x = 2`, target `y = 1`, and current `w = 0.2`:

```text
prediction = 0.2 × 2 = 0.4
loss       = (0.4 - 1)² = 0.36

gradient = dL/dw
         = 2(prediction - target) × x
         = 2(0.4 - 1) × 2 = -2.4

learning rate = 0.1
new w         = w - learning_rate × gradient
              = 0.2 - 0.1 × (-2.4) = 0.44
new prediction = 0.88
new loss       = (0.88 - 1)² = 0.0144
```

A derivative answers: “If I nudge this input, how fast does this output change?” The chain rule multiplies the local effects through a sequence of operations. Backpropagation applies that rule efficiently to a whole computation graph. The optimizer then uses those gradients; backpropagation and the update are separate operations.

**Misconception:** a gradient is a command to set a weight to the correct answer. It is a local slope. Too large a step can increase loss. With learning rate `1`, the same example changes `w` to `2.6`, prediction to `5.2`, and loss to `17.64`.

## 4. Check the gradient before trusting it

Approximate a derivative by nudging the parameter both ways:

```text
numerical_gradient ≈ [L(w + ε) - L(w - ε)] / (2ε)
```

For the quadratic above and `ε = 0.000001`, this is approximately `−2.4`. Compare it with the formula. This is a debugging check, not the efficient way to train a large network: checking every parameter needs many extra forward evaluations. Values that are too tiny suffer floating-point cancellation; nondifferentiable points such as ReLU at zero need separate treatment.

The [lab script](labs/policy_lab.py) checks all three model parameters against finite differences. Change one gradient sign as a deliberate fault and confirm the check fails before training.

## 5. Scaling makes the optimizer's job easier

If one feature is usually around 1 and another around 10,000, the same learning rate can produce very different parameter updates. Standardization computes:

```text
normalized_feature = (feature - training_mean) / training_standard_deviation
```

Keep those means and scales with the model and apply the same transformation at evaluation and deployment. A constant feature needs a defined handling rule; our lab uses scale 1 for near-zero variance. Scaling does not repair a wrong unit, missing sensor, or inconsistent coordinate frame.

**Worked example:** training values `[1, 2, 3]` have mean `2` and population standard deviation `sqrt(2/3) ≈ 0.8165`. The value `3` becomes about `1.2247`. A new value `4` becomes about `2.4495` using the *training* statistics. Recomputing the mean on each incoming batch changes the model's meaning.

## 6. What changes when the model becomes a neural network?

A linear model cannot express every useful rule. A network combines weighted sums with nonlinear functions. For a small hidden layer:

```text
hidden = ReLU(X @ W1 + b1)
output = hidden @ W2 + b2
```

`ReLU(z) = max(0, z)` changes which paths are active. Without nonlinearities, composing affine layers still gives one affine transformation. More capacity can fit more patterns and can also fit noise. Neither architecture size nor low training loss guarantees generalization.

For classification, raw scores are **logits**. A softmax converts them to normalized positive scores, but these scores need calibration before interpreting them as reliable probabilities. Numerically stable classification losses often accept logits directly: PyTorch's [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) does. Do not apply a second softmax before such a loss. API semantics reviewed 2026-10-18; pin the framework version used in a real experiment.

## 7. Translate the model into a hardware question

For a dense layer with 3 inputs and 4 outputs:

- Weights: `3 × 4 = 12`; biases: `4`; total: `16` parameters.
- FP32 parameter payload: `16 × 4 = 64 bytes`, excluding framework objects and alignment.
- Per observation: `12` multiply-accumulates plus bias addition. Under a convention counting a multiply and add as two FLOPs, the main matrix product is approximately `24 FLOPs`.
- Training also needs gradients, saved activations, and any optimizer state. Parameter bytes are not peak training memory.

Our Python lists deliberately expose arithmetic and incur interpreter overhead. Their timings are not accelerator benchmarks. Next, a tensor framework lets you measure array execution, kernel scheduling, copies, and memory on a named device.

## Exit exercise

For a `2 -> 4 -> 1` network, give every weight/bias shape, count parameters, and predict its FP32 parameter payload. Explain which arrays a training run adds.

<details>
<summary>Answer guidance</summary>

`W1 [2,4]`, `b1 [4]`, `W2 [4,1]`, `b2 [1]`: `8+4+4+1=17` parameters, or 68 payload bytes in FP32. Training adds parameter gradients and saved intermediate activations; optimizer choice adds state, such as Adam's two moment arrays. Batch size changes activation memory even though it does not change the parameter count.

</details>
