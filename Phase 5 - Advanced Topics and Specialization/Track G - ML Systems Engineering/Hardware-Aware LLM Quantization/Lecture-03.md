# Module 03 — Blackwell Hardware: What `sm_120` Actually Accelerates

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 02](Lecture-02.md) | **Next:** [Module 04 →](Lecture-04.md)

---

[Module 02](Lecture-02.md) treated formats as mathematics. This module treats them as **instructions**. A format is only fast if the silicon can consume it without a detour, and the gap between "fewer bits" and "faster" is where most quantization projects quietly fail.

The governing rule of this module:

> **A bit-width you cannot feed to a tensor core is a bit-width you are emulating.** Emulation costs alignment, issue slots, and the native MMA path — and it usually costs more than the bytes it saved.

---

## Learning objectives

By the end of this module you should be able to:

1. Describe the RTX 5090's memory system and compute roofline, and derive ridge points per precision.
2. Distinguish the **consumer Blackwell (`sm_120`)** tensor-core path from the **datacenter Blackwell (`sm_100`)** one, and name the instruction family each uses.
3. Explain why odd bit-widths (3-bit, 5-bit) lose *effective bandwidth* even when they reduce nominal bytes.
4. Compute the **break-even achieved-bandwidth threshold** for a non-native format.
5. Read a Nsight Compute report and confirm you are on the fast path rather than a dequant path.

---

## 1. The machine

```text
   NVIDIA GeForce RTX 5090  —  GB202, compute capability 12.0 (sm_120)

   ┌──────────────────────────────────────────────────────────────┐
   │  170 SMs  ×  128 FP32 lanes  =  21,760 CUDA cores            │
   │  5th-generation Tensor Cores (FP4 / FP6 / FP8 / BF16 / FP16) │
   │  boost ~2.41 GHz                                             │
   ├──────────────────────────────────────────────────────────────┤
   │  L2 cache: tens of MB  (irrelevant here — see §2)             │
   ├──────────────────────────────────────────────────────────────┤
   │  32 GB GDDR7, 512-bit bus @ 28 Gbps  →  1792 GB/s            │
   └──────────────────────────────────────────────────────────────┘
```

Dense tensor-core throughput roughly doubles per precision step:

| Precision | Dense TFLOP/s (approx.) | Ridge point (FLOP/byte) |
|---|---:|---:|
| FP32 (shader) | 105 | 59 |
| BF16 / FP16 | 419 | 234 |
| FP8 (E4M3/E5M2) | 838 | 468 |
| **NVFP4** | **1676** | **935** |

NVIDIA's headline "**3352 AI TOPS**" for this part is **FP4 with 2:4 structured sparsity**. Dense FP4 is half that. Sparsity requires a pruned model with the 2:4 pattern enforced and a sparsity-aware kernel; if you have not deliberately done that work, the number that applies to you is 1676, and quoting 3352 in a roofline will make you think you have 2× more compute headroom than you do.

---

## 2. Why the L2 cache does not save you

GB202's L2 is large by GPU standards — tens of megabytes. Your decode working set is **~16 GB of weights, streamed exactly once per token.**

```text
   working set per token   ≈  15.9 GB
   L2 capacity              ≈  0.1 GB
   reuse within one token   =  ~1×      (each weight read once, used once)
   reuse across tokens      =  0×       (evicted long before the next token needs it)
```

**Cache hit rate on the weight stream is essentially zero, and no amount of L2 fixes that.** This is why the roofline in [Module 01](Lecture-01.md) uses HBM/GDDR bandwidth directly with no cache correction — for batch-1 LLM decode, the cache hierarchy is a bystander.

Two practical consequences:

* The only way to reduce weight traffic is to make the weights *smaller*. There is no locality trick available.
* The KV cache **is** small enough to benefit from L2 at short context, which is one reason short-context decode outperforms the naive roofline slightly, and why that advantage evaporates as context grows ([Module 09](Lecture-09.md)).

---

## 3. Two Blackwells, two tensor-core programming models

This is the single most misattributed fact in current Blackwell material, so state it precisely:

```text
   HOPPER  (sm_90)     :  wgmma      — warpgroup MMA, async, operands in shared memory
   BLACKWELL datacenter
           (sm_100)    :  tcgen05    — 5th-gen MMA + Tensor Memory (TMEM), B200 / GB200
   BLACKWELL consumer
           (sm_120)    :  mma.sync   — warp-level MMA with BLOCK-SCALED FP4/FP8 operands
                                        RTX 50-series / GB202. No TMEM, no tcgen05.
```

Both Blackwell variants execute NVFP4 natively, but **through different instructions with different tiling requirements**, which means:

* A kernel tuned for B200 (`sm_100a`) does **not** simply recompile for RTX 5090.
* CUTLASS carries separate collective/kernel schedules for the two targets; you must build with the right arch (`sm_120a`) to get the block-scaled path at all.
* Library coverage differs. Something that has a fast NVFP4 kernel on B200 may fall back to a generic path on `sm_120`.

> **Verify, do not assume.** "Blackwell supports NVFP4" is true and insufficient. The question is always: *does this runtime, at this version, have a block-scaled kernel compiled for `sm_120a` for this operator shape?* Answer it with a profiler, not a datasheet.

---

## 4. The native/emulated boundary

```text
   NATIVE  (tensor core consumes the format directly, block scales in hardware)
   ├── NVFP4   (E2M1 + E4M3 block scale)     ← the target
   ├── MXFP4   (E2M1 + E8M0 block scale)
   ├── FP6     (E3M2 / E2M3)
   ├── FP8     (E4M3 / E5M2)
   └── BF16 / FP16

   EMULATED  (must be unpacked to a wider type before any MMA)
   ├── INT4 with arbitrary group sizes      (fast kernels exist — Marlin-class — but hand-written)
   ├── 3-bit, 5-bit, 6-bit integer          ← no native path, no tuned kernels
   └── 2-bit                                 ← worst alignment behaviour
```

### Why odd bit-widths lose *effective* bandwidth

The naive argument for 3-bit is compelling. Compare per-weight storage:

```text
   NVFP4  :  4 + 8/16   = 4.50 bits  =  0.5625 bytes/weight
   INT3   :  3 + 16/128 = 3.125 bits =  0.3906 bytes/weight      →  1.44× fewer bytes
```

If decode is bandwidth-bound, 1.44× fewer bytes should mean 1.44× more tok/s. It usually does not, and the reason is **not** primarily the arithmetic cost of unpacking:

```text
   4-bit:  8 weights pack into one 32-bit word, exactly.
           ┌────┬────┬────┬────┬────┬────┬────┬────┐
           │ w0 │ w1 │ w2 │ w3 │ w4 │ w5 │ w6 │ w7 │   aligned, coalesced,
           └────┴────┴────┴────┴────┴────┴────┴────┘   one shift+mask per weight

   3-bit:  10.67 weights per 32-bit word. Values STRADDLE word boundaries.
           ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬─┐┌─┬ ...
           │w0 │w1 │w2 │w3 │w4 │w5 │w6 │w7 │w8 │w9 │w│││10 ...
           └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─┘└─┴ ...
                                                      ▲
                                        split across two words
```

Straddling forces either a bit-shuffled storage layout (which breaks the natural coalescing of a GEMV's access pattern) or multi-word reads with cross-lane shuffles. Either way **achieved bandwidth drops**, and achieved bandwidth is the numerator of your throughput equation.

### The break-even rule

Make it quantitative. A format wins only if it delivers more tokens per second overall:

```text
                 BW_achieved
   tok/s   ∝   ────────────────
               bytes_per_weight
```

Setting NVFP4 (achieving 92 % of peak) equal to INT3 (achieving `x` of peak):

```text
       0.92                 x
   ───────────   =   ───────────
     0.5625             0.3906

              0.3906
   x  =  0.92 × ──────── =  0.639
              0.5625
```

> **An INT3 kernel must sustain ≥ 64 % of peak memory bandwidth just to tie NVFP4.** If your hand-written 3-bit GEMV achieves 55 % — an entirely typical result for an unaligned layout — you have shipped a model that is 14 % *smaller* and 14 % *slower*.

Generalize it. For any candidate format:

```text
                                  bytes_candidate
   BW_achieved_required  =  0.92 × ────────────────
                                    bytes_baseline
```

Compute this **before** writing the kernel. It tells you the bar, and it frequently tells you not to bother.

### The second penalty: leaving the native path

At batch 1 the above is the whole story. The moment you batch — speculative decoding verifies K+1 tokens per pass, which *is* a small batch — an emulated format also forfeits the FP4 MMA throughput:

```text
   NVFP4 native   :  block-scaled mma.sync  →  up to 1676 TFLOP/s
   INT3 emulated  :  unpack → BF16 mma.sync →  up to  419 TFLOP/s     (4× less)
```

This matters specifically because [Module 10](Lecture-10.md)'s speculation raises your effective batch. A format choice that looks free at batch 1 can cap your speculative gains.

---

## 5. What the fast path looks like in a profiler

Do not trust the format name in your config. Confirm the kernel. In **Nsight Compute**:

| Signal | Fast path (native NVFP4) | Slow path (dequant emulation) |
|---|---|---|
| Kernel name | contains `nvfp4` / `mxf4` / `blockscaled` / CUTLASS `sm120` schedule | contains `dequant`, `unpack`, or a generic `gemv` |
| `sm__inst_executed_pipe_tensor` | high | low or zero |
| Integer/ALU instruction share | low | **high** — the unpack |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 85–93 % | often 50–70 % |
| Register pressure / occupancy | moderate | elevated pressure, reduced occupancy |

The quickest single check is the one that costs nothing:

```bash
# Are we even launching a block-scaled kernel?
nsys profile -o trace ./run_decode.sh
nsys stats --report cuda_gpu_kern_sum trace.nsys-rep | head -20

# Then the decisive counter:
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active \
    --kernel-name-base demangled ./run_decode.sh
```

If `dram__throughput` is 72 % and the tensor pipe is idle, you are not bandwidth-limited by physics — you are limited by your kernel. **That is exactly the situation [Module 01 §5](Lecture-01.md) predicted for the case-study model**, and it is a kernel problem, not a quantization problem.

---

## 6. The hardware-speedup column, filled in

Returning to the opportunity framework from [Module 01](Lecture-01.md), here is the `HardwareSpeedup` factor for `sm_120`:

| Target format | Bytes/weight | Native on `sm_120`? | Speedup factor | Verdict |
|---|---:|---|---:|---|
| BF16 → FP8 | 2.0 → 1.0 | yes | 2.0× | **safe, always worth it on hot tensors** |
| BF16 → NVFP4 | 2.0 → 0.5625 | yes | 3.56× | **the main lever** |
| NVFP4 → MXFP4 | 0.5625 → 0.5312 | yes | 1.06× | not worth the error increase |
| NVFP4 → INT3 | 0.5625 → 0.3906 | **no** | ≤1.44× nominal, **<1.0× realistic** | **do not** |
| NVFP4 → INT2 | 0.5625 → 0.2656 | **no** | nominal 2.1×, realistic ≪1 | **do not** |

Which gives the practical rule for this silicon:

```text
   Every hot tensor should be NVFP4 or FP8.
   Nothing should be below 4 bits.
   The remaining wins are in WHICH tensors and in kernel quality — not in lower bit-widths.
```

That is a narrow design space, and narrowing it is the point. It means [Modules 04–11](Lecture-04.md) can focus entirely on **allocation** — which tensor gets FP8 and which gets NVFP4 — rather than on an unbounded search over exotic formats.

---

## Checkpoint

You should now be able to:

1. State the RTX 5090's bandwidth (1792 GB/s), dense FP4 throughput (~1676 TFLOP/s), and FP4 ridge point (~935 FLOP/byte).
2. Explain why "3352 AI TOPS" should not appear in your roofline.
3. Name the tensor-core instruction family for `sm_90`, `sm_100`, and `sm_120` without confusing them.
4. Explain why a 44 %-smaller 3-bit format can be slower, in terms of alignment and achieved bandwidth.
5. Compute the break-even achieved-bandwidth threshold for any candidate format.
6. Name the two Nsight counters that distinguish a native kernel from a dequant kernel.

---

## Ship it

Produce a **format feasibility table** for your own target GPU:

- measured peak bandwidth (run a STREAM-style or `bandwidthTest` copy benchmark; do not use the datasheet)
- dense TFLOP/s per precision, and the ridge point for each
- for each format you are considering: bytes/weight, native or emulated, break-even achieved-bandwidth threshold
- a profiler trace of one decode step, with the kernel names and `dram__throughput` for the top three kernels

Anyone reading that table should be able to say which formats are viable on your machine **without running a single quantization experiment.**

---

## Current as of

* **Timeless:** the roofline, the native-vs-emulated distinction, the alignment argument, the break-even derivation.
* **2026 hardware pins:** RTX 5090 / GB202 / `sm_120` / CC 12.0, 1792 GB/s, ~1676 dense FP4 TFLOP/s. Datacenter Blackwell = `sm_100` with `tcgen05` + TMEM; consumer Blackwell = `sm_120` with block-scaled `mma.sync`. Hopper = `sm_90` with `wgmma`.
* **Refresh surface — this is the most perishable module in the course.** Kernel coverage in CUTLASS, TensorRT-LLM, and vLLM for `sm_120a` block-scaled GEMMs changes release to release. Re-run the profiler checks in §5 after every runtime upgrade; a version bump can silently move you between the fast and slow paths.

---

**Next:** [Module 04 — Model Anatomy →](Lecture-04.md)
