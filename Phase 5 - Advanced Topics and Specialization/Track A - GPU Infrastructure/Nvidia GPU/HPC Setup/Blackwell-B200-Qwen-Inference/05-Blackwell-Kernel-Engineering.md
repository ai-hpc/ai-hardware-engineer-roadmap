# Chapter 5: Blackwell Kernel Engineering for Qwen Inference

**Previous:** [Placement](04-Multi-B200-NVL72.md) · **Next:** [Production serving](06-Production-Serving-on-Blackwell.md)

Understand the operator contract before modifying a kernel. B200 uses the SM100
architecture; another Blackwell product may have a different compute capability
and instruction path. The goal is a correct, measured implementation for actual
model shapes, not an instruction name or a peak-throughput headline.

## 1. Draw the operands and state

SM100's `tcgen05` family introduces matrix operations with accumulators in Tensor
Memory (TMEM). Kernel code must allocate/release TMEM, arrange operands and scales,
issue asynchronous work and wait for the appropriate completion before reading
results. Hopper's architecture-specific `sm_90a` WGMMA kernel does not become an
SM100 kernel by changing a model setting.

A conceptual pipeline is:

```text
global operands/scales → shared-memory staging and/or supported operand path
                      → tcgen05 matrix operation → TMEM accumulator
                      → synchronized TMEM read → epilogue → output
```

This is a dataflow sketch, not PTX syntax. Operand source, descriptors, layouts,
CTA group, scale vectors and synchronization depend on the chosen instruction.
Use [CUTLASS 4.8's SM100 guide](https://docs.nvidia.com/cutlass/4.8.0/media/docs/cpp/blackwell_functionality.html)
and the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) for exact
requirements. Mixing fragments from different instruction forms can compile
incorrectly or be rejected before any model computation occurs.

An elected thread can issue the matrix instruction in the supported cooperative
pattern; that does not mean one ordinary thread can independently own every
allocation, barrier and TMEM load. These operations have their own participation
and lifetime rules. Query the actual device's SM count instead of hardcoding 148.

## 2. The block-scaled arithmetic contract

For logical A[M,K], B[N,K] and block size G, the mathematical operation is:

```text
D[i,j] = C[i,j] + sum_k (
    code_A[i,k] × scale_A[i,floor(k/G)] ×
    code_B[j,k] × scale_B[j,floor(k/G)])
```

Packed codes, physical scale layouts, tensor-global factors, accumulator type
and epilogue must reproduce that contract. A `float_e2m1_t` declaration alone
specifies the scalar code grid, not MXFP4 or NVFP4 scale/layout semantics.
For native block-scaled FP4 arithmetic, both operand paths must satisfy the chosen
instruction's requirements; a BF16 activation and packed FP4 weight are not
proof of using native FP4 MMA.

The verified source example is [CUTLASS 72a at the pinned 4.8 commit](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu).
Its name's BF16 refers to its example contract; read the A/B, scale and output
types rather than inferring W4A16 from a filename. Its complete builder contains
more than a scalar element type and a guessed schedule. This course does not
claim to have compiled or run that GPU example.

**Worked reference:** one 16-value block with signed E2M1 codes, one byte scale and
a four-byte global factor uses 13 payload bytes. A 32×4 matrix padded by row uses
32 blocks, or 292 bytes, not 72. Native kernel layouts can pad or swizzle differently.
Run the existing [CPU FP4 calibration lab](../../../../Track%20G%20-%20ML%20Systems%20Engineering/Hardware-Aware%20LLM%20Quantization/CPU-Quantization-Lab.md) for zero blocks, tails,
rounding ties and saved/reloaded numerical artifacts. It establishes a reference
policy, not compatibility of that teaching layout with CUTLASS.

## 3. TMA, multicast and synchronization

Hopper already introduced multidimensional TMA copies and cluster multicast.
Blackwell adds instruction forms with target-specific requirements; use the
selected CUDA/PTX version rather than a vague “TMA-2” label. A producer must not
reuse a staging buffer until its consumers have finished, and a consumer must
wait for data visibility/completion before issuing dependent work.

If C CTAs need the same source tile, multicast can reduce repeated source traffic
for that tile. Each destination shared memory still receives a copy. Cache reuse,
cluster shape and work partition determine the DRAM/L2 effect. It is not a fixed
whole-model speedup or a reduction by the total SM count.

**Negative case:** using the wrong barrier phase can consume the previous tile
while leaving ordinary shapes apparently correct. A device extension should test
multiple K tiles, tails, different cluster schedules and sanitizer reports. CPU
reference agreement alone cannot validate GPU synchronization.

## 4. Persistent scheduling is not an immortal service

A persistent GEMM launches a bounded set of CTAs that process several tiles of
one operation. It still has a launch, scheduling overhead and finite lifetime.
A device service loop is a separate architecture with queues, ordering,
cancellation and shutdown. Neither is implied by a runtime's “persistent” label.

A simplified scheduler repeatedly obtains a valid tile, waits for operands,
computes, stores and obtains another tile. Do not translate this pseudocode into
a global-counter kernel without checking race-free ownership and resource limits.
An ordinary small grid can have about one wave per SM too, so
`launch__waves_per_multiprocessor≈1` does not prove persistence. Nor do many waves
mean “one launch per operation”: waves describe blocks within a kernel launch.

## 5. Attention correctness before architecture selection

FlashAttention-3's published optimized implementation targets Hopper. The current
FlashAttention repository also contains a CuTe DSL FlashAttention-4 path with
Hopper/Blackwell support and its own installation/feature limits. Runtimes may
instead select FlashInfer, cuDNN, CUTLASS or other attention implementations.
An algorithm generation label does not identify the actual loaded kernel.
See the [pinned attention-course source and CPU evidence](../../../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/FlashAttention%20Course/CPU-Attention-Lab.md).

Decode often has little parallelism along query length. Splitting the key/value
range across workers can improve utilization, but then partial softmax statistics
must be merged correctly. For partitions r with maximum m_r, normalization l_r,
and unnormalized numerator u_r:

```text
m = max_r m_r
l = sum_r exp(m_r-m) l_r
u = sum_r exp(m_r-m) u_r
output = u/l
```

Handle empty/masked partitions explicitly. Summing separately normalized outputs
is wrong. Split-KV also adds partial buffers and reduction work; it is not optimal
for every context, batch or head shape.

**Decode alignment exercise:** query length 1 with KV length 9 should attend the
valid history when the query is the final position. An upper-left causal mask
would expose only the first key. The CPU SDPA lab demonstrates this difference,
GQA head repetition, all-masked rows, tail tiles and gradients on actual PyTorch.
No B200/FA4 performance follows from its CPU result.

## 6. Triton and CuTe: start from a complete example

[Triton's block-scaled tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
uses explicit scale layouts and supported operand formats. `tl.dot` plus a comment
saying “FP4” does not enable block scaling. The [`dot_scaled` API](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html)
has packing/type/layout rules; pin the source version when reproducing it.

For a SwiGLU fusion, first define `down(silu(gate(X))*up(X))`, its intermediate
shape, quantization points and output dtype. A sketch that omits stores, masks or
the down projection is not a runnable fused FFN. Start from a complete tested
upstream GEMM, vary one epilogue, verify against a reference, then measure the
entire operator including conversions and temporary traffic.

## 7. Profile observations, not arbitrary pass marks

These are optional, upstream-tool workflows; none was executed on B200 here:

```bash
cuobjdump --dump-ptx ./your_binary
nsys profile -o decode_trace ./your_app
nsys stats --report cuda_gpu_kern_sum decode_trace.nsys-rep
ncu --query-metrics
```

PTX can be absent from cubin-only builds or differ from the path actually launched.
Correlate runtime dispatch, kernel identity, relevant instructions/SASS and numerical
outputs. A valid scalar/reduction/small-shape kernel may use no `tcgen05` at all.
Its absence is not sufficient evidence of a broken build.

Select supported counters on the installed device and collect a focused trace.
No universal rule requires occupancy >60%, tensor throughput >50%, DRAM >70% or
L2 hits >30%. Occupancy can trade off against tile reuse; low tensor utilization
can be expected for a memory-limited operator. Interpret the measured bottleneck
and compare alternative configurations with output parity.

## 8. Complete the lesson

Choose one actual model shape, state its arithmetic and serialization contracts,
and run the CPU format or attention reference linked above. On suitable hardware,
add one pinned upstream kernel build, correctness results, a trace and timings
with conversions included. Without hardware, mark those outcomes unexecuted and
explain what the CPU evidence cannot establish.

**Self-check answers:** persistence still pays launch cost; a format name does not
prove layout compatibility; high occupancy is not itself speed; a correct dense
reference cannot detect a GPU barrier race; a faster wrong-mask kernel fails the
operator contract.
