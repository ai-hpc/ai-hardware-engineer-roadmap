# Module 09 — KV Cache and Hybrid State: Find the Correct Crossover

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [Previous lesson](Lecture-08.md) | **Next:** [Next lesson](Lecture-10.md)

A long prompt can exhaust memory even when weights fit. Calculate the state for
the actual architecture and distinguish capacity from traffic. The worked case
here uses the immutable metadata in [Source Evidence](Source-Evidence.md).

## 1. The ordinary full-attention formula

For batch B, cached length L, N full-attention layers, H_kv key/value heads, head
dimension D, and C bytes per stored element:

```text
KV_payload = 2 × B × L × N × H_kv × D × C
              ^ keys and values
```

Use KV heads, not query heads, for GQA. Use allocated length/page count for capacity,
valid attended length for the first traffic estimate. Sliding-window, compressed,
shared-prefix and latent-attention implementations need their own formulas.
Quantized cache adds scale/zero-point metadata, page padding and sometimes
higher-precision residual windows.

**Worked conventional example:** 32 full-attention layers, 8 KV heads, head dimension
128, BF16, one sequence of 16,384 tokens needs **2 GiB** of K/V payload. Forgetting
V gives 1 GiB and underestimates memory by a factor of two.

## 2. Apply it to the pinned hybrid target

The case has **16 full-attention layers**, **4 KV heads**, **head dimension 256**.
Its other **48 linear-attention layers** maintain recurrent/convolution state,
not a full K/V history in the normal recurrent decode path.

```text
BF16 KV bytes per cached token = 2 × 16 × 4 × 256 × 2 = 65,536 bytes
FP8  KV bytes per cached token = 2 × 16 × 4 × 256 × 1 = 32,768 bytes
```

| Valid length, B=1 | BF16 K/V payload | FP8 payload, before metadata |
|---:|---:|---:|
| 4,096 | 256 MiB | 128 MiB |
| 32,768 | 2 GiB | 1 GiB |
| 131,072 | 8 GiB | 4 GiB |
| 262,144 | 16 GiB | 8 GiB |

This is an architectural calculation, **not a successful 262K runtime test**.
Using all 64 layers in this formula overcounts full-attention KV fourfold.
The five-layer external drafter has its own KV and must be counted separately.

## 3. Recurrent state is not zero state

A first state estimate for a gated delta recurrence may include a matrix per
value head, e.g. `[value_heads, key_dim, value_dim]`, plus a short convolution
history. Use the runtime's actual state layout, dtype, checkpoints and slot count.
Do not substitute hidden_size² or assume BF16 when the config requests FP32 state.

For the pinned target, an **illustrative one-slot matrix payload** is:

```text
48 layers × 48 value heads × 128 × 128 × 4 bytes = 150,994,944 bytes = 144 MiB
```

This counts only those matrices. Speculative rollback may retain extra state
slots/snapshots; prefill may use temporary state histories. The draft model and
verification tree add their own allocations. Runtime peak memory can be much
larger than persistent state. Measure it rather than calling recurrence “free.”

## 4. Derive a crossover with declared assumptions

Suppose the target's actual streamed weight traffic is W bytes per decode step.
Ignoring cache reuse and other traffic, full-attention K/V read traffic approaches
W near:

```text
L_star = W / (2 × B × N_full × H_kv × D × C)
```

Using **illustrative W=15.88 decimal GB** gives about **242,310 tokens in BF16**,
or **484,619 tokens in FP8** for this architecture. These are sensitivity estimates,
not measured throughput breakpoints. A different current weight ledger changes W;
state, compute, KV layout and bandwidth utilization can move the practical crossover.

The hypothetical 262K BF16 state plus the current target's 17.916 GB serialized
payload already sums to about **35.10 GB**, before runtime workspace and recurrent
state. That cannot fit in a 32 GB budget under those residency assumptions. FP8
reduces K/V payload but is feasible only if the engine supports the exact model,
cache scale scheme and rollback semantics. “Max positions=262144” is a model
configuration limit, not a device-memory guarantee.

## 5. Test cache quantization separately

Freeze weights and request sequence first. Compare cache formats on short/long
retrieval and task tests, reporting scale granularity, per-head/per-token/block
axes, residual windows and memory. K error changes scores; V error changes the
weighted values. Quantizing them identically is a choice to evaluate.

Then combine a successful cache policy with weight quantization and retest. Their
errors interact. Prefix reuse, new request reset, speculative reject/rollback and
page-tail handling deserve negative tests as much as average numerical error.

## Practice, answers and deliverable

Use the offline lab's `kv_payload_bytes` and the pinned config summary. Double
batch: K/V payload doubles. Halve cache width: payload halves before metadata.
Double **query** heads while keeping KV heads fixed: this formula is unchanged,
although attention compute and model compatibility are affected.

**Independent transfer:** a model has 40 layers but every fifth layer is full
attention. At H_kv=8,D=128,L=32768,B=2,BF16, calculate K/V. **Answer: 2 GiB**,
using 8 full-attention layers. Explain which additional runtime allocations remain.

Submit a capacity table and a traffic table with different headings; include an
explicit feasibility check before proposing a maximum-context hardware run.
