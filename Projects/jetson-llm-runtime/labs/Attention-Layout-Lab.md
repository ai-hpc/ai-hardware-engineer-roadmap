# Attention correctness: negative scores and growing KV prefixes

**Purpose:** connect the mathematical attention operation to the bytes actually
written by a decode loop. **Prerequisites:** arrays, softmax, and the
[runtime CPU lab](README.md). **Layers and roles:** inference kernels, runtime
memory, and numerical validation; preparation for inference and accelerator
engineering. Allow 20 minutes for the worked examples, 30 for the CPU checks,
and 30 for a fault explanation. The optional GPU setup is separate.

This lesson corrects two defects in the experimental runtime: floating maxima
were reduced with a signed-integer comparison, and the attention reader assumed
a different KV layout from the token writer. It does **not** establish that the
runtime can generate correct text from a pretrained checkpoint. Other kernels,
model conventions, and end-to-end execution still need target validation.

## 1. Negative logits are ordinary inputs

For one query and two keys, suppose the scaled scores are `[-1000, -1001]` and
the values are `[2,10]`. Subtract the maximum before exponentiation:

\[
p=\frac{[1,e^{-1}]}{1+e^{-1}},\qquad
o=2+\frac{8}{1+e}\approx4.15153.
\]

Computing `exp(-1000)` directly underflows in ordinary floating arithmetic, so
the shifted expression is essential. A GPU block must also find the correct
maximum. Reinterpreting a negative IEEE-754 float's bits as a signed integer
does **not** preserve numerical ordering. In particular, the integer encoding
of `-FLT_MAX` exceeds the encoding of `-1000`. The old integer `atomicMax`
could retain the sentinel and produce invalid exponentials for an entirely
negative tile.

The corrected kernel computes floating maxima inside each warp, stores four
warp results, and combines them with floating comparisons. It also combines
the four partial sums in a fixed order. This is a correctness choice, not a
claim that this reduction is the fastest implementation on a Jetson.

## 2. Current length is not a storage stride

The pool reserves separate K and V spans per layer. Each span is
`[capacity, kv_heads, head_dim]`, with dimension contiguous:

```text
element offset = (position * kv_heads + kv_head) * head_dim + dimension
```

With capacity 7, two KV heads, three dimensions, and FP16 storage, each token
occupies `2 × 3 × 2 = 12` bytes in K and another 12 in V. V begins `7 × 12 = 84`
bytes after K for that layer. The current length selects how many token rows to
read; it never changes these addresses.

The old reader instead used `kv_head * current_length * head_dim + ...`, treating
the same bytes as head-major storage. The error can stay hidden when there is
only one KV head or one token. A robust test therefore needs multiple heads,
multiple tokens, and a capacity larger than the active prefix.

For a four-query-head/two-KV-head example, query heads 0–1 share KV head 0 and
heads 2–3 share KV head 1. Set all queries to zero, so attention weights are
uniform. Let `V[t,h,d] = 100h + 10t + d` for three tokens. The correct outputs
are `[10,11,12]` for query heads 0–1 and `[110,111,112]` for heads 2–3.
This known answer catches a layout error without depending on a second softmax
implementation.

## 3. Preserve the online softmax invariant

For previously seen keys, retain maximum `m`, denominator `l`, and unnormalized
weighted values `a`. A new tile raises the maximum to `m_new`. Before adding
that tile, rescale **both** old accumulators:

\[
c=e^{m-m_{new}},\quad l\leftarrow c l,
\quad a\leftarrow c a.
\]

Then add `exp(score - m_new)` to the denominator and its weighted value to the
numerator. The final result is `a/l`. Rescaling only the denominator corrupts
the output when a later tile contains the largest score. The CPU exercise
compares this tiled mathematical model with a separately materialized-score
double-precision oracle, using five tile sizes for every fixture.

## 4. Execute the CPU checks

Run from `Projects/jetson-llm-runtime`:

```bash
python3 -m unittest discover -s labs -p 'test_attention_contracts.py' -v
```

This **executed** test compiles the actual pool pointer logic with the bounded
host CUDA-allocation shim and runs the C++ reference executable. It covers:

- **13 fixtures × 5 tile sizes**, including negative logits, GQA, partial tiles,
  head dimension 129, and prefix lengths 1, 17, 63, 64, 65, and 129.
- Independent known answers for two-key attention, uniform GQA attention, and
  the single-token case.
- The original signed-integer comparison counterexample.
- Actual `KVCachePool` offsets versus the shared token-major address helper,
  across two layers and separate K/V spans.
- Invalid GQA geometry, empty prefixes, excessive head dimensions, and invalid
  scale values.

A second native test checks the actual scratch allocator and decode planner:
known peak sizes, host-buffer reservation before KV admission, checked arithmetic,
200 layer allocation/rewind cycles with preserved token state, and rejection of
invalid checkpoints or exhausted storage. See the [worked scratch budget](../docs/memory.md#5-budget-the-maximum-live-scratch-then-order-its-reuse).
This host trace does not execute CUDA stream scheduling.

The test passes on the recorded x86-64 CPU host. A separate ASan/UBSan build also
passes. These results check CPU math, address arithmetic, and host pool control
logic. They do not execute CUDA shuffles, GPU memory transactions, or a model.
The normal CPU CMake build also exposes the `attention_cpu_oracle` CTest target.

## 5. Compile and, on a compatible GPU, run parity

The optional harness links the actual attention kernel directly. From the same
project directory, using a working CUDA toolkit:

```bash
nvcc -std=c++17 -arch=sm_87 -O2 -I include \
  tests/test_attention.cu src/kernels/attention.cu -o /tmp/jllm-attention-parity
/tmp/jllm-attention-parity
```

**Recorded status:** compilation succeeded with NVCC **13.4.92**, targeting
SM 8.7. Execution returned **77** with “no usable CUDA device/driver”; **GPU
parity was not executed**. Compilation on this x86-64 host is not a Jetson ARM64
binary or evidence of JetPack compatibility. Build with the installed supported
target toolchain before target measurements. The separate `decode.cu` translation
unit also compiled for SM 8.7.

On a usable GPU, the harness poisons output with NaNs before **every** invocation,
checks CUDA launch/completion/copy errors, rejects nonfinite outputs, and compares
against the double-precision oracle. The fixed acceptance rule is
`abs(error) <= 0.003 + 0.001 * abs(reference)`, allowing FP16 output rounding in
these bounded fixtures. It reports each maximum absolute error. Unused FP16 cache
capacity contains NaNs to expose reads beyond the active data.

The isolated kernel also has one INT8 fixture with **explicit fixed per-head K
and V scales**. This does not enable INT8 caching in the runtime: append-time
quantization, scale storage, calibration, and model-quality parity remain
unsupported. The runtime rejects INT8 and CUDA-graph generation modes. A missing
scale pointer is now an invalid kernel call rather than an implicit scale of 1.

## 6. Completion and transfer

Save your CPU output and annotate one cache row by hand. Explain why length 1
can conceal the old layout error, why a new tile maximum changes the old
numerator, and why a successful compiler exit is not a numerical result.
For an independent variation, change the GQA group size and active prefix while
retaining the same backing storage. Predict a uniform-attention answer before
running it. Keep the optional GPU result labeled “not run” until it actually runs.

The [validation record](attention-validation.json) contains source/binary hashes,
commands, CPU results, compiler version, and the explicit GPU skip. No throughput,
energy, memory-bandwidth, pretrained-text quality, or Jetson execution result is
claimed by this lesson.
