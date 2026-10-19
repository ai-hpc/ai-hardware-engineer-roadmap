# TVM Deep Dives — compile, inspect, and test a computation

Learn how an ML compiler represents a tensor computation, changes its execution schedule, and packages a runnable artifact. Start on a CPU with tiny inputs whose answers you can check. GPU tuning, custom accelerator integration, and embedded deployment are extensions that require their own hardware and toolchain evidence.

The [executable compiler lab](labs/README.md) compiles a 37-element affine operation and a small matmul+bias+ReLU model with Apache TVM's LLVM CPU backend. It checks vector tails, wrong shapes, output values, and real shared-library export/reload. No pretrained weights or GPU are required.

## Prerequisites and route

You should be comfortable with Python, matrix multiplication, array shapes, and a loop. If compiler IR is new, first complete the [CPU compiler introduction](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/00%20-%20CPU%20Compiler%20Lab/Guide.md). Allow roughly one session per lecture; spend longer on the optional tuning/hardware extensions when available.

| Lesson | Question | Artifact |
|---|---|---|
| [01 — compilation flow](Lecture-01.md) | What changes between graph, tensor program, and machine code? | Original/lowered IR plus checked CPU output |
| [02 — schedules](Lecture-02.md) | Which loop changes preserve meaning and handle tails? | Vectorized 37-element kernel and boundary tests |
| [03 — tuning](Lecture-03.md) | When does search repay its cost? | Candidate timings, correctness checks, and reuse assumptions |
| [04 — graph optimization](Lecture-04.md) | When is fusion legal and worthwhile? | Shape/layout contract and a fusion/BYOC decision |
| [05 — deployment](Lecture-05.md) | What must travel with a compiled artifact? | Exported/reloaded CPU module and deployment manifest |

The outcome is a reproducible compiler experiment, not a required win over a framework. A framework may already call an excellent library kernel or use its own graph compiler. A result showing that TVM loses on a shape is useful when explained and measured fairly.

## Version map, checked 2026-10-18

The lab pins **apache-tvm 0.27.0.post1**, installed from PyPI and actually executed. Its API uses `tvm.tirx`, `tvm.s_tir`, `from tvm.script import tirx as T`, `T.sblock`, and `Schedule.get_sblock`. The [moving upstream Quick Start](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html) currently uses the separately named `tvm_tirx` package for several core pieces. That package is **not** present in the tested release wheel. These are different snapshots, not interchangeable import examples.

| Material | How to use it |
|---|---|
| This lab | Install its exact requirements and run its complete files |
| Latest website examples | Match the documented source/package revision before adapting |
| Older `tvm.tir`, `tvm.meta_schedule`, `T.block`, `get_block` examples | Historical APIs; namespace replacement alone may be insufficient |
| Relay/GraphExecutor/AutoTVM/Ansor/microTVM tutorials | Use an archived compatible checkout and explain the historical executor/toolchain |
| MLC-LLM | Use its own documented compiler dependency and target packages; arbitrary TVM wheel compatibility is not established |

Inspect `tvm.__version__`, `tvm.__file__`, build options, and exported IR before debugging a tutorial. A build option such as `USE_CUDA=ON` in a wheel does not mean a CUDA device was present or executed.

## What makes an experiment convincing

Use deterministic inputs, an independent numerical reference, odd shapes and tails, and an explicit dtype/layout contract. Record compiler/package versions, target, compile time, warmup, timing method, output error, and artifact hashes. Keep candidate correctness separate from performance ranking. Tiny kernels may measure Python/FFI overhead rather than useful hardware throughput.

TVM is one compiler family among several. XLA has GPU autotuning and [persisted tuning caches](https://openxla.org/xla/persisted_autotuning); describing it as “no autotuning” is incorrect. PyTorch can compile graphs, and TensorRT, IREE, and MLC have their own supported workloads and deployment boundaries. Compare concrete versions and tasks rather than assigning a universal maturity ranking.

Finish by explaining one transformation, one caught failure, and one measured limitation. Then continue to [AI chip design](../../Track%20F%20-%20AI%20Chip%20Design/Guide.md) or [inference engineering](../AI%20Inference%20Engineer%202026/README.md) with those contracts intact.
