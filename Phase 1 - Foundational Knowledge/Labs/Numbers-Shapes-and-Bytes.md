# First lab: numbers, shapes, and bytes

Build a tiny AI operator you can calculate by hand, then explain its numerical and memory contract. This is the starting artifact for the Phase 1 hardware/software comparison and for a robot's later observation pipeline.

**Layers:** L3 runtime and L5 architecture. **Roles:** AI systems, accelerator, embedded, and robotics engineers. **Prerequisites:** multiplication, lists, and running a Python script; no GPU, calculus, or ML package. **Next:** [computer architecture](../2.%20Computer%20Architecture%20and%20Hardware/Guide.md), then [observation freshness](../../Phase%202%20-%20Embedded%20Systems/Labs/Observation-Freshness.md).

**Time:** 5–10 minutes setup, 25–40 minutes lab, 20 minutes explanation. **Done means:** a JSON result, passing negative checks, and a one-page explanation of scale, shape, and estimated bytes. Use the [shared rubric](../../Learning-Guide.md#completion-rubric).

## Start with three questions

1. A matrix has 2 rows and 3 columns. How many values must its input vector contain, and how many outputs result?
2. An integer `4` represents a physical value with scale `0.5`. What value does it represent?
3. Is a million operations always slower than a thousand operations?

Attempt these before reading. Answers are at the end. If lists and command-line arguments are new, run the provided script unchanged first; you can learn its functions one at a time.

## A shape tells you which values combine

Think of each matrix row as a recipe: multiply each input by its matching weight, then add. For a small two-output layer:

```text
weights W, shape [2,3]          input x, shape [3]
[ 1.00, -0.50, 0.25 ]          [ 2.00, -1.00, 0.50 ]
[-1.00,  0.75, 0.50 ]

y[0] =  1×2 + (-0.5)×(-1) + 0.25×0.5 =  2.625
y[1] = -1×2 +   0.75×(-1) +  0.5×0.5 = -2.500
```

The output has shape `[2]`. A robot might eventually map three calibrated observations to two intermediate features this way. These particular numbers are a mathematical fixture, not a trained controller or motor command.

## An integer needs a scale

An `int8` value stores one of 256 integers, from −128 to 127. A scale tells us what one step means. This lab uses zero point 0:

```text
stored integer = clamp(round(real value / scale), -128, 127)
reconstructed value = stored integer × scale
```

Rounding is nearest with ties to even; clamping means values outside the range become the nearest endpoint. These choices are part of the contract, not details to leave to a compiler. Here, weight scale is `0.25` and input scale is `0.5`:

```text
quantized W = [[4, -2, 1], [-4, 3, 2]]
quantized x = [4, -2, 1]
integer dot products = [21, -20]
output scale = 0.25 × 0.5 = 0.125
reconstructed output = [2.625, -2.5]
```

The output is exact because this fixture's values fall on the chosen grids. Most real inputs do not. Changing scale changes precision and representable range. A smaller scale improves the grid spacing but can clip larger values.

**Misconception to catch:** int8 inputs do not imply int8 accumulation. Three large products can already overflow int8. Python integers grow automatically; a C++ or RTL implementation must choose and check its accumulator width. With no bias and full-range signed int8 operands, `K × 128 × 128 ≤ 2³¹−1` is a conservative bound for an int32 dot product. Different zero points, intermediate reductions, or bias require another analysis.

## Run the reference

**Tested here:** Python 3.12.3 on Linux, standard library only. Run from the repository root; the script normally finishes in under a second.

```bash
python3 --version
python3 'Phase 1 - Foundational Knowledge/Labs/operator_contract.py' --self-test --output /tmp/operator-contract.json
cat /tmp/operator-contract.json
```

Expect six passing tests, accumulators `[21, -20]`, reconstructed output `[2.625, -2.5]`, and maximum absolute error `0.0`. The tests also reject a wrong shape, invalid scale, and nonfinite input. A deliberate missing scale produces the wrong result. The JSON marks numerical reference and analytical estimates explicitly; it contains no GPU performance measurement.

Read the code in this order: `matvec`, `quantize`, `reference_report`, `traffic_budget`. `ContractTests` explains what each boundary must reject. `main` just handles arguments and saves the result.

## Count bytes before guessing speed

For `B` input vectors, an `M×K` int8 weight matrix, and int32 outputs, assume every weight and input is read once and every output written once:

```text
bytes = M×K + B×K + 4×B×M
operations = 2×B×M×K  (count one multiply and one add per MAC)
arithmetic intensity = operations / bytes
```

For `M=K=1024, B=1`, the model gives **1,053,696 bytes** and **2,097,152 integer operations**, or about **1.99 operations/byte**. The weight array alone is `1,048,576 bytes = 1 MiB`; decimal MB and binary MiB differ. For `B=8`, weights remain the same size while more operations reuse them, giving about **15.40 operations/byte**.

These are compulsory-traffic estimates for this chosen boundary. Metadata, scales, padding, cache misses, reloads, partial sums, conversion, and host/device copies may add traffic. Python lists occupy much more space than packed int8 tensors. This script models tensor storage; it does not measure Python object memory or DRAM bandwidth.

With hypothetical sustained bandwidth `10 GB/s`, the `B=1` transfer bound is `1,053,696 / 10,000,000,000 ≈ 105.4 µs`. Actual execution can be slower, and cached data changes the relevant memory boundary. For a complete estimate compare both compute time and transfer time; the larger gives a lower bound when they can overlap. Do not call integer operations FLOPs.

## Change one thing

**Guided variation:** copy the script, change input `0.5` to `0.3`, and predict the quantized value before running. Keep the original fixture tests as a reference and add a separate variation case. Record output error. Then halve the input scale and predict whether this particular error improves. Do not assume every scale reduction improves every input.

**Independent transfer:** specify a `3×4` operator with at least one positive and negative input. Include a clipped value, a half-step rounding tie, and a rejected wrong shape. Export reference vectors suitable for later C++ and RTL implementations. State output error tolerance before comparing implementations.

**Stop condition:** once the six baseline tests pass and you can explain your changed case, move on. Larger matrices add no value until you have a native implementation to benchmark.

## Answers and recovery

1. Three input values produce two outputs. Match the matrix columns to the input length. A ragged matrix or wrong length is an error; silently truncating with `zip` is not a solution.
2. `4 × 0.5 = 2`. A stored integer without its scale is incomplete information.
3. No. Parallelism, data movement, dispatch overhead, dependencies, and hardware all matter. Compare the same operation and timing boundary before interpreting speed.
4. For input `0.3`, scale `0.5` stores `1`, reconstructing `0.5`; outputs stay `[2.625, -2.5]` while the floating reference becomes `[2.575, -2.6]`, giving maximum error `0.1` up to floating-point representation. Scale `0.25` stores `1`, reconstructing `0.25`; outputs become `[2.5625, -2.625]`, with maximum error `0.025`.
5. Omitting input scale multiplies both outputs by two in this fixture. If your independent implementation passes only the all-zero input, your test set cannot detect this error.

**Carry forward:** save the numerical contract and reference vectors. Use them as correctness gates before parallelizing, changing precision, or feeding outputs into a simulated robot.
