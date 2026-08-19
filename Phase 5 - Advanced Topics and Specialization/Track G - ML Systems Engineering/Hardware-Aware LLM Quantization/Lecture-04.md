# Module 04 — Model Anatomy: Finding the Bytes That Matter

**Collection:** [Hardware-Aware LLM Quantization](README.md) | **Previous:** [← Module 03](Lecture-03.md) | **Next:** [Module 05 →](Lecture-05.md)

---

[Module 01](Lecture-01.md) claimed that checkpoint size and per-token traffic are different quantities. This module builds the instrument that separates them — the **per-token byte ledger** — and then uses it to reverse-engineer an entire model's architecture from four published numbers.

This is the module that turns the framework into arithmetic you can run on any checkpoint in an afternoon.

---

## Learning objectives

By the end of this module you should be able to:

1. Classify every tensor in a checkpoint into one of four traffic classes.
2. Build a per-token byte ledger and predict batch-1 decode throughput from it.
3. Reconstruct a model's hidden dimension, vocabulary, and parameter split from its byte inventory.
4. Explain why the ledger changes when speculation or multimodality is enabled.
5. Identify the highest-opportunity tensor in a checkpoint without running an experiment.

---

## 1. The four traffic classes

Every tensor in an LLM checkpoint belongs to exactly one of these:

```text
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ CLASS A — STREAMED                                                       │
   │   Read in full, every forward pass. THE decode critical path.            │
   │   → MLP weights, attention projections, lm_head, per-layer norms         │
   │   Optimization target: YES. This is where tok/s lives.                   │
   ├─────────────────────────────────────────────────────────────────────────┤
   │ CLASS B — GATHERED                                                        │
   │   Resident in full, indexed sparsely. Capacity cost, ~no traffic cost.   │
   │   → input embedding table                                                 │
   │   Optimization target: only if VRAM-constrained.                          │
   ├─────────────────────────────────────────────────────────────────────────┤
   │ CLASS C — CONDITIONAL                                                     │
   │   Traffic depends on SERVING CONFIG, not on the checkpoint.               │
   │   → vision/audio towers, MTP or draft heads, LoRA adapters                │
   │   Optimization target: depends. Ledger it twice — enabled and disabled.  │
   ├─────────────────────────────────────────────────────────────────────────┤
   │ CLASS D — STATE                                                           │
   │   Not in the checkpoint at all. Grows with context and batch.             │
   │   → KV cache, workspace, activation buffers                               │
   │   Optimization target: dominant at long context (Module 09).             │
   └─────────────────────────────────────────────────────────────────────────┘
```

The mistake the ledger prevents is treating a Class B or C tensor as though it were Class A because it is large. **Size determines class membership not at all.**

---

## 2. Reconstructing a model from its bytes

Take the case-study checkpoint. Published facts, and nothing else:

```text
   resident weights      :  18.80 GiB
   BF16 remaining        :   6.91 GB  (33.6 % of checkpoint)
   of which:  embeddings :   2.54 GB
              lm_head    :   2.54 GB
              vision     :   0.92 GB
              MTP head   :   0.85 GB
   decode throughput     :  81.6 tok/s
```

**Step 1 — normalize units.** `18.80 GiB × 1024³ / 10⁹ = 20.19 GB`.

**Step 2 — the residual is the quantized body.**

```text
   20.19 GB (total)  −  6.91 GB (BF16)  =  13.28 GB  in NVFP4
```

**Step 3 — convert bytes to parameters** using bits-per-weight from [Module 02](Lecture-02.md) (NVFP4 = 0.5625 B/param, BF16 = 2 B/param):

| Group | Bytes | ÷ B/param | **Parameters** |
|---|---:|---:|---:|
| Transformer body (NVFP4) | 13.28 GB | 0.5625 | **23.61 B** |
| Embeddings (BF16) | 2.54 GB | 2.0 | 1.270 B |
| lm_head (BF16) | 2.54 GB | 2.0 | 1.270 B |
| Vision tower (BF16) | 0.92 GB | 2.0 | 0.460 B |
| MTP head (BF16) | 0.85 GB | 2.0 | 0.425 B |
| Norms / misc | 0.06 GB | 2.0 | 0.030 B |
| **Total** | **20.19 GB** | | **27.06 B** |

**27.06 B parameters.** The checkpoint reconstructs to a 27 B model to within 0.2 %, from byte counts alone. The internal consistency is the proof that the ledger is correct.

**Step 4 — infer the architecture.** Embeddings and `lm_head` are the same size, so the model has **untied** input/output embeddings, and each is `vocab × d_model = 1.270 B`. Solving for plausible hidden sizes:

```text
   d_model = 8192   →  vocab ≈ 155,000     ← consistent with a ~152–156 K Qwen-family tokenizer
   d_model = 5120   →  vocab ≈ 248,000     ← implausibly large
   d_model = 4096   →  vocab ≈ 310,000     ← implausible
```

So: **`d_model ≈ 8192`, `vocab ≈ 155 K`, untied embeddings, ~23.6 B params in the transformer body.** All of it derived from four numbers and the bits-per-weight table.

> This is a genuinely useful skill. You will constantly face checkpoints whose configs you cannot inspect (a competitor's release, a quantized artifact, a service you only see through an API). The byte inventory tells you most of what you need.

---

## 3. The per-token byte ledger

Now the ledger that actually predicts throughput. **Text-only decode, speculation disabled, short context:**

| Tensor group | Bytes | Class | Read per token | Why |
|---|---:|---|---:|---|
| Transformer body | 13.28 GB | **A** | **13.28 GB** | every weight participates |
| lm_head | 2.54 GB | **A** | **2.54 GB** | full vocab projection |
| Norms / misc | 0.06 GB | **A** | **0.06 GB** | per-layer, tiny but streamed |
| Embeddings | 2.54 GB | B | **~16 KB** | one row: `8192 × 2 B` |
| Vision tower | 0.92 GB | C | **0** | no image in the request |
| MTP head | 0.85 GB | C | **0** | speculation disabled |
| KV cache | — | D | small at short ctx | Module 09 |
| | | | | |
| **`B_token`** | | | **≈ 15.88 GB** | |

```text
   resident  20.19 GB   ─────▶   B_token  15.88 GB
                                  │
                    4.31 GB (21 %) of the checkpoint is NEVER
                    fetched during a text-only decode step
```

**Validate against the measurement:**

```text
   BW_achieved  =  15.88 GB × 81.6 tok/s  =  1296 GB/s  =  72.3 % of 1792 GB/s
```

Compare with the naive calculation that uses resident bytes:

```text
   naive :  20.19 GB × 81.6  =  1647 GB/s  =  91.9 %   ← wrong; charges for 4.31 GB never read
   real  :  15.88 GB × 81.6  =  1296 GB/s  =  72.3 %   ← correct
```

The engineering conclusions are opposite:

| If utilization is… | Then the binding constraint is… | And the next move is… |
|---|---|---|
| 92 % | bytes | quantize more |
| **72 %** | **kernel / launch / scheduling** | **profile the kernels first** |

```text
   headroom at CONSTANT bytes:
        1650 GB/s (92 % of peak)  ÷  15.88 GB  =  103.9 tok/s
        measured                                =   81.6 tok/s
                                                   ─────────────
        available without removing one bit      =  +22.3 tok/s  (+27 %)
```

**Twenty-two tokens per second are sitting in kernel efficiency, not in bits.** That is the ledger earning its keep — it redirected the entire optimization effort.

---

## 4. Ranking the targets

With the ledger built, the opportunity framework from [Module 01](Lecture-01.md) becomes arithmetic. Each candidate's ceiling is bounded by its share of `B_token`:

| Candidate | Traffic | Share of `B_token` | Native path? | Max possible gain |
|---|---:|---:|---|---:|
| **`lm_head` BF16 → FP8** | 2.54 GB | **16.0 %** | yes | **+8.7 %** |
| `lm_head` BF16 → NVFP4 | 2.54 GB | 16.0 % | yes | +13.4 % |
| Embeddings BF16 → FP8 | 16 KB | 0.0001 % | yes | **+0.0 %** |
| Vision BF16 → NVFP4 | 0 | 0 % | yes | **+0.0 %** |
| Body NVFP4 → INT3 | 13.28 GB | 83.6 % | **no** | negative (Module 03) |

Working the top candidate through the throughput equation, holding achieved bandwidth at the measured 1296 GB/s:

```text
   lm_head 2.54 GB → 1.27 GB (FP8)     B_token: 15.88 → 14.61 GB
        1296 / 14.61  =  88.7 tok/s          (+8.7 %)

   lm_head 2.54 GB → 0.71 GB (NVFP4)   B_token: 15.88 → 14.05 GB
        1296 / 14.05  =  92.2 tok/s          (+13.0 %)
```

And the contrast that makes the point:

```text
   Embeddings are the SAME SIZE as lm_head (2.54 GB each).
   Quantizing embeddings :  +0.0 %  throughput,  −1.27 GB VRAM
   Quantizing lm_head    :  +8.7 %  throughput,  −1.27 GB VRAM

   Identical tensors. Identical capacity win. ~150,000× difference in traffic.
```

If you take one table from this course, take that one.

---

## 5. The ledger changes when the config changes

The same checkpoint has **different ledgers** under different serving configurations. This is why Class C exists.

### With speculation enabled (the DSpark build)

Measured: **155.75 tok/s at acceptance length 2.886.** The target model now runs once per *accepted group*, not once per token:

```text
   target forward passes/s  =  155.75 / 2.886  =  53.97 passes/s
```

Each target pass streams the full `B_token`, and the MTP head runs `K` times per cycle to produce the draft:

| Draft depth `K` | Traffic (GB/s) | % of peak |
|---:|---:|---:|
| 2 | 946 | 52.8 % |
| 3 | 991 | 55.3 % |
| 4 | 1037 | 57.9 % |

```text
   traffic  =  53.97 × ( 15.88  +  K × 0.85 )   GB/s
                        └ target ┘   └ MTP ┘
```

**Bandwidth utilization *drops* from 72 % to ~55 % when speculation is enabled.** That is not a regression — it is exactly what speculation does: it amortizes one weight-read across ~2.9 emitted tokens. But it has a sharp consequence:

> In the speculative configuration the model is **no longer bandwidth-bound.** It sits at ~55 % of peak, which means further weight quantization has **diminishing returns**, and the dominant levers become **acceptance length** ([Module 10](Lecture-10.md)) and **kernel/launch efficiency** ([Module 03](Lecture-03.md)).

This is the kind of conclusion the ledger produces and intuition does not. The same model, same weights, same GPU — and the correct optimization strategy inverts depending on whether speculation is on.

### With an image in the request

The vision tower moves from Class C to Class A for the prefill pass: `+0.92 GB` of traffic, once per image, not per token. It never enters the decode ledger. **Quantizing the vision tower improves image-prefill latency and VRAM, and never improves tok/s** — precisely as [Module 01](Lecture-01.md) claimed.

---

## 6. Build the analyzer

A ledger you compute by hand once is a note. A ledger you can regenerate is a tool.

```python
"""Per-token byte ledger for a safetensors checkpoint."""
import json, re
from collections import defaultdict
from safetensors import safe_open

BYTES_PER_PARAM = {  # see Module 02
    "BF16": 2.0, "F16": 2.0, "F32": 4.0,
    "F8_E4M3": 1.0, "F8_E5M2": 1.0,
    "NVFP4": 0.5625,   # 4 bits + 8-bit E4M3 scale per 16 elements
    "MXFP4": 0.5312,   # 4 bits + 8-bit E8M0 scale per 32 elements
}

# Class A = streamed every token. Class B/C/D never enter B_token by default.
CLASS_RULES = [
    (r"vision|visual|image_encoder",           "C_vision"),
    (r"mtp|draft|eagle|medusa",                "C_speculative"),
    (r"embed_tokens|wte|tok_embeddings",       "B_gathered"),
    (r"lm_head|output\.weight",                "A_streamed"),
    (r"layers\.\d+\.",                         "A_streamed"),
    (r"norm|ln_f",                             "A_streamed"),
]

def classify(name: str) -> str:
    for pattern, cls in CLASS_RULES:
        if re.search(pattern, name):
            return cls
    return "A_streamed"          # conservative default: assume it is on the hot path

def ledger(path: str, d_model: int, spec_enabled=False, draft_depth=0):
    by_class = defaultdict(float)
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            sl = f.get_slice(name)
            n = 1
            for d in sl.get_shape():
                n *= d
            bpp = BYTES_PER_PARAM.get(sl.get_dtype(), 2.0)
            by_class[classify(name)] += n * bpp

    resident = sum(by_class.values())
    b_token  = by_class["A_streamed"] + d_model * 2      # + one gathered embedding row
    if spec_enabled:
        b_token += by_class["C_speculative"] * draft_depth

    return {
        "resident_GB":  resident / 1e9,
        "resident_GiB": resident / 1024**3,
        "B_token_GB":   b_token / 1e9,
        "dead_weight_GB": (resident - by_class["A_streamed"]) / 1e9,
        "by_class_GB":  {k: v / 1e9 for k, v in sorted(by_class.items())},
    }

def predict_tps(b_token_GB: float, peak_BW_GBs=1792, efficiency=0.92):
    return peak_BW_GBs * efficiency / b_token_GB

def achieved_BW(b_token_GB: float, measured_tps: float):
    return b_token_GB * measured_tps          # GB/s — compare to peak, NOT to resident×tps
```

Run it, then run the comparison that matters:

```text
   predicted_tps  =  predict_tps(B_token)              ← what physics allows
   measured_tps   =  benchmark()                       ← what you get
   gap            =  predicted / measured

   gap ≈ 1.0   →  you are at the bandwidth wall. Quantize.
   gap > 1.2   →  you are NOT bandwidth-bound. Profile kernels first.
```

For the case study: `predicted = 103.9`, `measured = 81.6`, **`gap = 1.27`**. Profile the kernels.

---

## Checkpoint

You should now be able to:

1. Name the four traffic classes and place any tensor into one.
2. Reconstruct a model's parameter count and hidden dimension from a byte inventory.
3. Build a `B_token` ledger and compute achieved bandwidth correctly.
4. Explain why embeddings and `lm_head` — identical in size — differ by five orders of magnitude in traffic.
5. Explain why enabling speculation *lowers* bandwidth utilization and *changes which optimization is correct*.
6. Use the `predicted/measured` gap to decide between quantizing and profiling.

---

## Ship it

For a checkpoint you run: the analyzer script, the ledger table (all four classes), the predicted-vs-measured comparison with the gap, and **a one-sentence verdict naming your next action**. If the gap is above ~1.2, the verdict must be "profile kernels", not "quantize" — and the rest of this course will still be here when you come back.

---

## Current as of

* **Timeless:** the four traffic classes, the ledger method, byte-to-parameter reconstruction, the gap heuristic.
* **2026 case-study pins:** the 27.06 B reconstruction, `B_token ≈ 15.88 GB`, 72.3 % achieved bandwidth at 81.6 tok/s, and ~55 % under speculation are derived from the published NVFP4/DSpark builds linked in the [course index](README.md).
* **Note:** `BYTES_PER_PARAM` must track the formats your runtime emits. Safetensors dtype strings for 4-bit formats vary between exporters — verify against your own checkpoint rather than trusting the table.

---

**Next:** [Module 05 — Calibration →](Lecture-05.md)
