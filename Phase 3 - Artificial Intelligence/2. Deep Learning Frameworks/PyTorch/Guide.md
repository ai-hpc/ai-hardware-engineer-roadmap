# PyTorch — Industry-Standard Deep Learning Framework

<div class="course-identity auto-course" style="--course-accent: #0891b2; --course-accent-rgb: 8, 145, 178;" markdown="1">
<div class="course-identity__icon">PISD</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for PyTorch — Industry-Standard Deep Learning Framework.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Parent:** [Module 2 — Deep Learning Frameworks](../Guide.md)

> *The framework most models are written in. You need fluency here because every model you deploy on hardware starts as PyTorch code.*

---

## What to Master

| Concept | Why it matters for hardware |
|---------|---------------------------|
| `torch.Tensor` | The data structure accelerators process. Shape, dtype, layout (contiguous, channels-last). |
| `nn.Module` | How models are structured. Layers → forward pass → computational graph. |
| Autograd (`loss.backward()`) | Generates the backward graph that training hardware executes. |
| Data loading (`DataLoader`) | CPU-GPU pipeline. Bottleneck if not overlapped with compute. |
| `torch.onnx.export()` | How models leave PyTorch and enter the compiler/runtime stack (Phase 4C). |
| `torch.compile()` | PyTorch's built-in compiler (Inductor). Generates Triton kernels. |
| `torch.profiler` | Where is time spent? Kernel launches, memory copies, CPU overhead. |
| Mixed precision (`torch.cuda.amp`) | FP16/BF16 training — what tensor cores accelerate. |
| Quantization (`torch.ao.quantization`) | INT8 inference — what L6 PE arrays must support. |

## Projects

1. **Train ResNet-18** on CIFAR-10 from scratch. Profile with `torch.profiler`. Identify top-3 time-consuming ops.
2. **Export to ONNX.** Visualize graph with Netron. Count total ops and parameters.
3. **Post-training quantization** to INT8. Measure accuracy drop and inference speedup.
4. **`torch.compile()`** on a transformer block. Compare eager vs compiled execution time.

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- *Deep Learning with PyTorch* (Stevens, Antiga, Viehmann)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
