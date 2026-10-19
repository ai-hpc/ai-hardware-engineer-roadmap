# FlashAttention: a systems and kernel course

<div class="course-identity auto-course" style="--course-accent: #16a34a; --course-accent-rgb: 22, 163, 74;" markdown="1">
<div class="course-identity__icon">ATTN</div>
<div markdown="1">
<p class="course-identity__eyebrow">Phase 4C · Kernel Engineering</p>
<p class="course-identity__title">Preserve attention's mathematical and API contract while changing its memory and execution schedule.</p>
<p class="course-identity__meta">Core artifact: tested CPU reference and SDPA comparisons · Extension: a pinned GPU kernel experiment</p>
</div>
</div>

**Reviewed 18 October 2026.** Ten lectures connect the attention equations to memory traffic, work partitioning, gradients and cached decoding. Begin with the [actual CPU attention lab](CPU-Attention-Lab.md): seven portable tests plus fourteen NumPy/PyTorch tests execute without a GPU or model download. A separate CUDA extension requires compatible hardware and software; source review is not represented as GPU execution.

**Parent:** [Kernel Engineering](../Guide.md). **Prerequisites:** Python arrays, matrix multiplication, softmax and basic memory units. CUDA basics are needed for the hardware extension, not the CPU path. Plan four to six study sessions, then a capstone; divide framework setup from the mathematical exercises.

## The question this course answers

Attention computes weighted combinations of value vectors. Standard implementations can store a large score/probability matrix. Tiled implementations keep small score tiles and running normalization while changing the order of computation. The result can be mathematically equivalent with floating-point differences, but only if masks, positions, scaling and layout preserve the same contract.

For robotic vision or VLA workloads, those conditions matter as much as raw speed: a padding mask, camera-token layout or cached position error can produce a fast wrong answer. This course teaches the operator boundary; it does not claim whole-model task quality from a kernel test.

## Start with a five-minute diagnostic

- Q is `[2,8]`, K is `[5,8]`, V is `[5,3]`. What is the output shape? **`[2,3]`.**
- For one newest query and five live keys, must causal attention keep only key 0? **No.** That is upper-left alignment; a newest-token query normally aligns to the end of the live history.
- If a tiled implementation avoids the full score matrix, are all bytes read exactly once? **No.** Tiles can reread inputs; capacity and traffic are different.

If these are unfamiliar, read Lectures 01–02 before installing optional frameworks.

## Learning sequence

| Lecture | What you learn | Observable artifact |
|---|---|---|
| [01 — Attention and roofline](Lecture%2001%20-%20Attention%20Bottleneck%20and%20Roofline.md) | Shapes, FLOPs, capacity and traffic | Executed budget with assumptions |
| [02 — Online softmax](Lecture%2002%20-%20Online%20Softmax%20and%20Numerical%20Correctness.md) | Running numerator/denominator and empty rows | Derivation and NumPy comparison |
| [03 — FA1 algorithm](Lecture%2003%20-%20FlashAttention-1%20Algorithm.md) | Historical loop order and SRAM units | Source annotation and traffic model |
| [04 — GPU fundamentals](Lecture%2004%20-%20GPU%20Kernel%20Performance%20Basics.md) | Ownership, synchronization, coalescing and occupancy | Buffer-lifetime diagram; optional device profile |
| [05 — API and source](Lecture%2005%20-%20Repo%20Anatomy%20and%20Python%20CUDA%20API.md) | Pinned FA2/FA3/FA4 families and layouts | Actual source call map |
| [06 — FA2](Lecture%2006%20-%20FlashAttention-2.md) | Query-row partitioning and deferred normalization | Ownership sketch and recurrence counterexample |
| [07 — Backward](Lecture%2007%20-%20Backward%20Pass%20and%20Validation.md) | Analytic, finite-difference and autograd gradients | Executed gradient/precision report |
| [08 — Decode and KV](Lecture%2008%20-%20Inference%20Path%20KV%20Cache%20and%20Decode.md) | Rectangular alignment, GQA, pages and RoPE | Cache/mask tests and byte budget |
| [09 — FA3 and FA4](Lecture%2009%20-%20Hopper%20FA3%20FA4.md) | Hopper/Blackwell pipeline and source boundaries | Dated source map; optional hardware comparison |
| [10 — Capstone](Lecture%2010%20-%20Capstone.md) | One tested implementation change | Patch, failure case, raw output and report |

## Current implementation boundary

The course inspected Dao-AILab source revision `965c9212e6be33c821ee9f2b461564c5c91a2503`, dated 16 October 2026. FA2 CUDA, FA3 Hopper C++ and FA4 Python CuTeDSL are separate paths. FA4 has an installable prerelease; it is not merely a proposed future kernel. Import names and device support must be checked against the selected family. [Pinned upstream source](https://github.com/Dao-AILab/flash-attention/tree/965c9212e6be33c821ee9f2b461564c5c91a2503).

The executed environment uses NumPy 2.3.5 and PyTorch 2.14.0+cpu, explicitly selecting SDPA's math backend. It validates small seeded arrays, masks, gradients, GQA and toy RoPE positions. It does not run a Dao-AILab CUDA kernel. The lab retains hashes, source metadata, test output and actual numerical errors.

## Completion and handoff

Finish the CPU path when you can reproduce the tests, explain the upper-left/lower-right difference, derive the online normalization, count both K and V, and add one independently tested mask or cache case. For the GPU extension, add exact hardware/build information, selected-kernel evidence, correctness and measured timing; leave unavailable outcomes marked not run.

Pass the resulting mask/layout/precision/position contract to [the compiler stack](../../03%20-%20Compiler%20Stack/Guide.md). A compiler or accelerator optimization must preserve that contract before its latency result becomes useful.
