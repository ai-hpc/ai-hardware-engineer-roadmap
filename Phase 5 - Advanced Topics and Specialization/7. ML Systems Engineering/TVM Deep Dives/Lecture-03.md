# Lecture 03 - Auto-Tuning: AutoTVM → Ansor → MetaSchedule, the Learning-Based Compiler

**Collection:** [TVM Deep Dives](README.md) | **Previous:** [← Lecture 02](Lecture-02.md) | **Next:** [Lecture 04](Lecture-04.md)

---

Lecture 2 ended on an uncomfortable truth: a vendor-competitive GPU matmul needs register tiling, double-buffering, vectorized loads, bank-conflict avoidance — and the right combination is **different for every shape, dtype, and chip**. Hand-scheduling that, per operator, per target, per generation of hardware, is not engineering. It is an infinite job.

The defining idea of TVM — the one that separated it from every compiler before it — is to make that job a **search problem**. Describe the computation once. Let the machine enumerate legal schedules, predict which are fast with a learned model, measure the promising ones on real hardware, and keep the best. The compiler *learns* the kernel instead of a human writing it.

This lecture is that machine. We trace its three generations — **AutoTVM**, **Ansor/AutoScheduler**, **MetaSchedule** — because each one removed a human bottleneck the previous left in, and the lineage is how you understand what MetaSchedule (the current system, and the engine under MLC-LLM's tuning) actually does.

---

## Learning objectives

By the end of this lecture, you should be able to:

1. State why auto-tuning exists: the schedule space is too large and too target-specific to hand-navigate per shape.
2. Explain the **tuning loop** — space generation → cost model → on-device measurement → search → database — and which resource is the expensive one.
3. Distinguish the three generations: AutoTVM (**template-based**), Ansor (**template-free, auto-generated space**), MetaSchedule (**probabilistic schedule programs on TIR**, unifying both).
4. Run **`ms.tune_tir`** on a kernel and **`ms.tune_relax`** on a whole model, then apply the tuning database and build.
5. Read a **tuning curve** (best latency vs trials) and reason about the cost-model-vs-measurement tradeoff.
6. Stand up **distributed tuning over RPC** to measure on a real edge device from a host.

---

## 1. Why search beats hand-tuning

A single GPU matmul schedule has, easily, a dozen tunable decisions: tile sizes on `i`, `j`, `k` (each a multi-way split), the shared-memory staging factor, the vectorization width, the unroll depth, whether to use Tensor Cores. The legal combinations number in the **millions**. Most are slow. A handful are within a few percent of peak. A human cannot enumerate them; a human can barely guess a good corner.

But the space has structure a machine can exploit:

```text
the schedule space is huge       → you cannot try everything
but most of it is obviously bad  → a cheap model can rank candidates
and "fast" is target-specific    → only real hardware gives ground truth
and you tune once, run forever   → amortize search cost over deployment
```

So the strategy writes itself: **generate** candidates, **predict** with a cheap learned cost model, **measure** only the promising few on the actual device, **learn** from those measurements, repeat. The expensive resource is the on-device measurement — each one is a build + run, hundreds of milliseconds to seconds. The cost model exists precisely to spend those measurements wisely.

This is "the learning-based compiler." It is also why TVM kernels can, on odd shapes and non-vendor targets, *beat* hand-written vendor libraries: the library author tuned the common case once; the search tunes **your** case, on **your** chip.

---

## 2. The tuning loop

Every generation of TVM auto-tuning is a variation on one loop. Learn the loop; the generations differ only in *how the space is generated*.

```text
   ┌──────────────────────────── TUNING LOOP (per task) ────────────────────────────┐
   │                                                                                 │
   │   ① SPACE GENERATION ──────► candidate schedules                                │
   │      (templates, or sketch rules,        │                                      │
   │       or PostOrderApply schedule rules)  ▼                                      │
   │                                  ② COST MODEL  (XGBoost / MLP)                   │
   │                                  predicts latency, ranks cheaply                │
   │                                          │  keep top-k                          │
   │                                          ▼                                      │
   │                                  ③ MEASURE on real hardware                      │
   │                                  build + run  (local  OR  RPC → device)         │
   │                                          │  true latency                        │
   │                                          ▼                                      │
   │                                  ④ UPDATE cost model + DATABASE                  │
   │                                          │                                      │
   │                                  ⑤ SEARCH proposes next batch                    │
   │                                  (evolutionary / replay)                        │
   │                                          └────────────► repeat until budget     │
   └─────────────────────────────────────────────────────────────────────────────────┘
                                            │  apply best record
                                            ▼
                              tuned schedule → codegen → fast kernel
```

Two things to hold onto:

* **The database is the asset.** Tuning produces a database of `(workload → best schedule record)`. That database is what you ship and re-apply; you tune once and the build step just looks up the winner. Treat it like a build artifact — version it, cache it.
* **Trials are the budget.** `max_trials_global` (total on-device measurements) is the knob that trades tuning time for kernel quality. The cost model's whole job is to make each trial count.

---

## 3. Three generations, three bottlenecks removed

The history is not trivia — each generation deleted a specific human cost, and knowing which tells you what you still have to do yourself.

| | **AutoTVM** (2018) | **Ansor / AutoScheduler** (2020) | **MetaSchedule** (2022, current) |
|---|---|---|---|
| Search space | **human-written template** with tunable knobs | **auto-generated** (sketch + annotation) | **auto-generated** via schedule rules on TIR; also accepts custom probabilistic schedules |
| Human effort per op | write a template (high) | none | none (rules built-in) |
| IR level | TE schedules | TE / loop state | **TensorIR** directly |
| Cost model | XGBoost on features | learned (XGBoost/MLP) | XGBoost / MLP, feature-based |
| Search | knob tuning (grid/genetic) | evolutionary over sketches | evolutionary over schedule traces |
| Tensorization (Tensor Cores) | manual | limited | **first-class** (tensorize in the space) |
| Relax / Unity integration | no (Relay-era) | partial | **yes** (`tune_relax`) |
| Status | legacy | legacy (superseded in Unity) | **the current system** |

The arc in one sentence each:

* **AutoTVM** proved search works — but made *you* write a schedule template per operator, with `cfg.define_split` / `cfg.define_knob` declaring the tunable axes. Powerful, but the template is real labor, and you write a new one for every new op.

```python
# AutoTVM flavor (legacy): you author the template AND its knobs
@autotvm.template("demo/matmul")
def matmul(N, L, M, dtype):
    # ... build a TE schedule ...
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=2)   # the human declares what is tunable
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_knob("unroll", [0, 16, 64])
    # ... apply cfg choices to the schedule ...
```

* **Ansor** deleted the template. It *generates* the search space from the compute definition (sketch generation = coarse structure, then random annotation = tile sizes etc.), and searches with evolutionary mutation + a learned cost model. The published result: Ansor matched or beat AutoTVM everywhere, with reported speedups up to ~9× and, crucially, **zero templates**. This is the conceptual leap — the human stops describing *how to schedule* and only provides *what to compute*.

* **MetaSchedule** (a.k.a. **AutoTensorIR**) is the third generation and the one you use today. It unifies the two prior worlds as **probabilistic schedule programs over TensorIR**: built-in schedule *rules* (applied bottom-up via `PostOrderApply`) auto-generate the space like Ansor, but you can also drop in custom probabilistic schedule functions like AutoTVM — same framework, your choice of automation level. It works directly on TIR (so it composes with everything in Lecture 2), makes **tensorization first-class** (it can put Tensor Core schedules in the search space), and integrates with **Relax** so you can tune a whole model, not just a kernel.

For this course: **we use MetaSchedule.** AutoTVM and Ansor appear when you read older code; MetaSchedule is the engine that matters — including, downstream, as part of how MLC-LLM produces fast kernels.

---

## 4. Hands-on: tune a kernel with `ms.tune_tir`

Take the matmul `PrimFunc` from Lecture 2 and let MetaSchedule schedule it for you. No template, no hand-written tile sizes — you provide the computation and a trial budget.

```python
import tvm
from tvm import meta_schedule as ms
from tvm import tir

mod = matmul_te(1024, 1024, 1024)          # the TIR PrimFunc from Lecture 2
target = tvm.target.Target("nvidia/geforce-rtx-3090")   # or "llvm -mcpu=native", a Jetson, etc.

database = ms.tune_tir(
    mod=mod,
    target=target,
    work_dir="./ms_work",                  # logs + the tuning database land here
    max_trials_global=1000,                # total on-device measurements (the budget)
    num_trials_per_iter=64,                # batch size per search iteration
)

# Pull the best schedule MetaSchedule found and build it
sch = ms.tir_integration.compile_tir(database, mod, target)
sch.mod.show()                             # ← read the machine-found schedule!
rt_mod = tvm.build(sch.mod, target)
```

Two habits a senior engineer keeps here:

1. **Read the schedule it found.** `sch.mod.show()` prints exactly the kind of tiled, bound, cached, possibly tensorized TIR you hand-wrote in Lecture 2 — except the machine chose the tile sizes. You can *recognize* the structure because you scheduled by hand first. This is why Lecture 2 came first.
2. **Keep `work_dir`.** That directory holds the database. Re-running with the same `work_dir` resumes; shipping it lets a teammate build the tuned kernel without re-tuning. It is a build artifact.

---

## 5. Hands-on: tune a whole model with `ms.tune_relax`

Tuning one kernel is a demo. The real workflow is tuning **every kernel in a model** so the end-to-end network is fast. MetaSchedule's Relax integration extracts each tunable TIR function as a task, tunes across all of them under a shared budget, and applies the winners back into the graph.

```python
from tvm import relax
from tvm import meta_schedule as ms

# mod, params: a Relax IRModule + weights, e.g. imported in Lecture 1 and lowered
database = ms.relax_integration.tune_relax(
    mod=mod,
    params=params,
    target=target,
    work_dir="./ms_model",
    max_trials_global=20000,               # spread across all kernels in the model
    # task scheduler decides how to split the budget across tasks (gradient-based by default)
)

# Build the model with the tuning database applied
ex = ms.relax_integration.compile_relax(database, mod, target, params)
vm = relax.VirtualMachine(ex, tvm.cuda(0))
out = vm["main"](x)
```

What happened under the hood: MetaSchedule found the tunable TIR functions (the matmuls, convs, attention kernels), treated each as a **task**, and let a **task scheduler** allocate the 20 000 trials across them — spending more on the kernels that dominate runtime (a matmul that is 40% of the network gets more trials than a normalization that is 2%). The result is an end-to-end-optimized model, not a bag of individually fast kernels.

> **Budget intuition.** Per-kernel, useful tuning is often hundreds to low-thousands of trials. A whole model with dozens of distinct kernels wants tens of thousands. Tuning a model can run minutes to hours depending on budget and how fast each measurement is — which is exactly why the next section matters.

---

## 6. The cost-model vs measurement tradeoff (read the tuning curve)

The single most useful diagnostic is the **tuning curve**: best measured latency as a function of trials spent.

```text
  latency
    │•
    │ •                      every point is one on-device measurement.
    │  ••                    the curve drops fast early (easy wins),
    │    •••                 then flattens (diminishing returns).
    │       ••••
    │           •••••••           ← the knee: where extra trials stop paying
    │                  ••••••••••••••••••••
    └────────────────────────────────────────► trials
```

Reading it is the skill:

* **Steep early drop** → the cost model is finding obviously-better schedules quickly; the default space is good.
* **The knee** → where you stop. More trials past the knee buy fractions of a percent. Set `max_trials_global` near the knee for the next run.
* **Flat from the start / noisy** → either the kernel is already near-optimal, the cost model is mis-ranking (rare), or your measurements are noisy (fix: pin clocks, idle the machine, raise `number`/`repeat`).

The deeper point: the cost model lets you explore *thousands* of candidates while only *measuring* hundreds. Without it you would measure everything and tuning would take days. With it, the search concentrates measurements where the model is unsure or optimistic. **Trials are money; the cost model is how you spend them well.** When someone asks "how long does TVM tuning take," the honest answer is "as long as the trial budget × per-trial measurement time — and you choose the budget by where the curve knees."

---

## 7. Distributed tuning over RPC — tuning on the device that will run it

Here is the part that makes auto-tuning *hardware* engineering, not just compiler engineering: **you must measure on the chip that will run the kernel.** A schedule tuned on an RTX 3090 is wrong for a Jetson Orin, an ARM Cortex-A, or an embedded NPU — different cache sizes, different core counts, different memory bandwidth. The cost model can be retrained, but ground truth comes from the target silicon.

TVM solves this with an **RPC system**: the device runs a server that registers with a **tracker**; the host runs the search and dispatches each measurement to the device over the network.

```text
   HOST (laptop / CI box)                 RPC TRACKER                 DEVICE (Jetson / ARM board)
   ┌────────────────────┐                ┌──────────┐                ┌───────────────────────┐
   │ MetaSchedule search │── request ───►│  :9190   │◄── register ───│ rpc_server  key=jetson │
   │ cost model          │               │  key map │                │ runs the candidate     │
   │ builds candidate    │── cross-      │          │── dispatch ───►│ kernel, returns timing │
   │                     │   compile     └──────────┘                └───────────────────────┘
   └────────────────────┘◄──────────────── measured latency ─────────────────────┘
```

On the device, start a server pointed at the tracker:

```bash
# on the Jetson / ARM board:
python -m tvm.exec.rpc_server --tracker=HOST_IP:9190 --key=jetson
```

On the host, tune with an **RPC runner** so every measurement executes on the board:

```python
from tvm import meta_schedule as ms

runner = ms.runner.RPCRunner(
    ms.runner.RPCConfig(
        tracker_host="HOST_IP", tracker_port=9190,
        tracker_key="jetson", session_timeout_sec=120,
    ),
    # how many times to repeat each measurement on-device for a stable number
)

database = ms.tune_tir(
    mod=mod,
    target=tvm.target.Target("nvidia/jetson-agx-orin"),   # the device's real target
    runner=runner,                                         # ← measure on the board, not the host
    work_dir="./ms_jetson",
    max_trials_global=2000,
)
```

The host generates and cost-models candidates; the *board* runs them and reports truth. This is how you tune for an edge fleet from a CI machine — and how the same workflow extends to a brand-new accelerator the moment it can run a TVM RPC server. For a microcontroller too small to host Python, the AOT/microTVM path in Lecture 5 carries the same idea down to bare metal.

---

## 8. Measure it

The deliverable for any tuning effort is a **before/after with the roofline named**:

| Build | Latency | GFLOP/s | % of peak | Notes |
|---|---|---|---|---|
| Framework baseline (cuBLAS / oneDNN) | — | — | — | the bar to beat |
| TVM, un-tuned (default schedule) | — | — | — | what you get free |
| TVM, MetaSchedule (N trials) | — | — | — | what tuning bought |

Three numbers prove the work: the **speedup over un-tuned** (did tuning help?), the **speedup-or-parity vs the vendor library** (did it beat the bar, or get close on a shape the vendor didn't cover?), and the **trials to the knee** (what it cost). A tuned kernel that is 0.8× the un-tuned one means a broken measurement setup or too small a budget — investigate, do not ship.

---

## 9. Mini-lab: tune a kernel and a model, read the curves

1. **Kernel:** `ms.tune_tir` your Lecture-2 matmul at budgets `{200, 500, 1000, 2000}` trials. Plot the tuning curve and mark the knee. Compare the best to your hand-schedule from Lecture 2 and to cuBLAS/oneDNN.
2. **Model:** import a small CNN or transformer block, `ms.tune_relax` it at `max_trials_global ∈ {2000, 10000}`. Record end-to-end latency vs the un-tuned TVM build and the framework baseline.
3. **(Stretch) RPC:** if you have an edge board, stand up a tracker + `rpc_server` and re-tune the kernel *on the device*. Compare the on-device-tuned schedule's tile sizes to the desktop-tuned ones — they will differ, and the difference is the lesson.

Deliverable: the two tuning curves, the before/after table from §8, and a paragraph: where was the knee, did TVM beat the vendor library on any shape, and (if you did RPC) how did the device-tuned schedule differ from the desktop one. That is a Level-4 measurement artifact.

---

## Key takeaways

- Hand-scheduling per shape × dtype × target is infinite work. TVM's foundational idea is to make it a **search**: generate → cost-model → measure on device → learn → repeat.
- The **tuning loop** is the same across generations; they differ only in how the space is generated. The **database** is the shippable asset; **trials** (on-device measurements) are the budget.
- **AutoTVM** (template-based, you write the knobs) → **Ansor** (template-free, auto-generated space, ~up to 9× over AutoTVM, zero templates) → **MetaSchedule** (probabilistic schedule programs on TensorIR, unifies both, tensorization first-class, Relax-integrated). Use **MetaSchedule**.
- `ms.tune_tir` tunes a kernel; `ms.tune_relax` tunes a whole model, with a task scheduler splitting the trial budget toward the kernels that dominate runtime.
- The **cost model** lets you explore thousands of candidates while measuring only hundreds — read the **tuning curve**, stop at the knee.
- **Tune on the target silicon.** RPC (server + tracker + RPCRunner) measures candidates on the real device from a host — the workflow that makes auto-tuning genuine hardware engineering and scales to edge fleets and new accelerators.

---

## References

- MetaSchedule tutorial ("Search-Based Auto-Tuning"): [https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
- Shao et al., "Tensor Program Optimization with Probabilistic Programs" (MetaSchedule), NeurIPS 2022: [https://arxiv.org/abs/2205.13603](https://arxiv.org/abs/2205.13603)
- Zheng et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning," OSDI 2020: [https://www.usenix.org/conference/osdi20/presentation/zheng](https://www.usenix.org/conference/osdi20/presentation/zheng)
- Chen et al., "Learning to Optimize Tensor Programs" (AutoTVM), NeurIPS 2018: [https://arxiv.org/abs/1805.08166](https://arxiv.org/abs/1805.08166)
- TVM RPC / cross-device tuning docs: [https://tvm.apache.org/docs/how_to/tune_with_autotvm/](https://tvm.apache.org/docs/how_to/tune_with_autotvm/)
- *TVM Deep Dives* — [Lecture 02](Lecture-02.md) (the space being searched) and [Lecture 05](Lecture-05.md) (dlight: fast schedules *without* tuning, and where each is the right choice).

---

*Next: [Lecture 04 — Relax in depth: dynamic shapes, fusion, and BYOC](Lecture-04.md)*
