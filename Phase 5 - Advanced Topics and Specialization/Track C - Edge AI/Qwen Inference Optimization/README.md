# Qwen inference: from checkpoint metadata to a tested runtime

**Reviewed 18 October 2026.** Six lessons use Qwen3-4B-Instruct-2507 and Qwen2.5-72B-Instruct as fixed case studies. Start with the [CPU lab](Lab-Guide.md): no pretrained weights or GPU are required. Allow 4–6 hours for the worked readings and portable exercises; optional device experiments take additional time.

The required path teaches tensor dimensions, rotary positions, quantization scales, cache budgets, parallel reductions, speculative-prefix arithmetic, and matrix layout. The optional current-library exercise actually runs a tiny random Qwen3 on CPU and inspects full model metadata without allocating model weights.

| Lesson | Question to answer | Evidence |
|---|---|---|
| [1. Checkpoint and architecture](Lecture-01.md) | Do shapes, norms, positions and template match this release? | config snapshots, exact parameter counts, cached/full comparison |
| [2. Quantization](Lecture-02.md) | Which bytes and errors does this representation introduce? | row-scale experiment, corrupted-scale failure, conversion manifest |
| [3. Decode on Jetson](Lecture-03.md) | Is the limiting cost traffic, launches, attention or another stage? | correct byte calculations and controlled measurement plan |
| [4. Multi-GPU execution](Lecture-04.md) | What is partitioned, replicated and communicated? | partial-output sum and per-rank budget |
| [5. Serving](Lecture-05.md) | Does speculation/routing improve completed work at the required latency? | prefix simulation and capacity calculation |
| [6. GEMM validation](Lecture-06.md) | Are the stored layout and numerical contract correct? | non-square padded matrix tests |

**Prerequisites:** [Edge LLM inference](../Edge%20LLM%20Inference%20Internals/Lecture-01.md), matrix multiplication, basic probability, and Python. Read the [Jetson compatibility guide](../../../Phase%204%20-%20Track%20B%20-%20Nvidia%20Jetson/Compatibility.md) before optional device work and [quantization foundations](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/04%20-%20Quantization/Guide.md) for numerical background.

**Validation boundary:** 14 portable tests and the pinned Transformers CPU/meta exercise executed successfully. Pretrained 4B/72B conversion, model-quality benchmarks, Jetson throughput, cuBLAS GPU work, and multi-GPU serving were not run. Each optional procedure states its prerequisites and the evidence needed for a stronger claim. Model fit depends on the complete memory budget and supported runtime, not its advertised parameter size.

**Finish:** submit raw outputs, hashes, one deliberately rejected configuration, and a short explanation of the dominant measured or estimated cost. Hand the resulting shape/workload distribution to [kernel engineering](../../../Phase%204%20-%20Track%20C%20-%20DL%20Inference%20Optimization/02%20-%20Kernel%20Engineering/Guide.md).
