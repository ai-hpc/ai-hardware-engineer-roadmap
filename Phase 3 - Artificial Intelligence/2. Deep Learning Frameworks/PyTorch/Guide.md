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

> *The framework most models are written in. You need fluency here because many deployment workflows begin with a PyTorch reference.*

---

## What to Master

| Concept | Why it matters for hardware |
|---------|---------------------------|
| `torch.Tensor` | The data structure accelerators process. Shape, dtype, layout (contiguous, channels-last). |
| `nn.Module` | How models are structured. Layers → forward pass → computational graph. |
| Autograd (`loss.backward()`) | Generates the backward graph that training hardware executes. |
| Data loading (`DataLoader`) | CPU-GPU pipeline. Bottleneck if not overlapped with compute. |
| `torch.onnx.export()` | How models leave PyTorch and enter the compiler/runtime stack (Phase 4C). |
| `torch.compile()` | Compilation interface; backend/device determine generated code and library calls. |
| `torch.profiler` | Where is time spent? Kernel launches, memory copies, CPU overhead. |
| Mixed precision (`torch.amp`) | FP16/BF16 training — what tensor cores accelerate. |
| Quantization ([torchao](https://docs.pytorch.org/ao/stable/index.html)) | Low-precision representations and supported kernels; verify dtype, operator, and device coverage. |

**API currency, reviewed 2026-10-18:** PyTorch documents the migration of quantization development to [torchao](https://docs.pytorch.org/docs/stable/quantization.html). Treat older `torch.ao.quantization` recipes as version-specific migration material. Pin compatible PyTorch/torchao versions and check backend support; a smaller weight file does not establish a faster runtime.

## Start with a correct reference

Before the larger projects, reproduce the [CPU policy lab](../../0.%20AI%20and%20Robot%20Learning%20Foundations/Lab-01-CPU-Policy.md), then port its three-parameter model to `nn.Linear(2, 1)`. Preserve the episode split, normalization, and held-out metrics. Match predictions against the plain-Python reference on a fixed batch before benchmarking.

Keep these operations separate:

- `model.train()` / `model.eval()` change behavior of modules such as dropout and BatchNorm.
- `torch.no_grad()` or `torch.inference_mode()` controls gradient recording. `eval()` does not disable autograd.
- `optimizer.zero_grad()` clears accumulated parameter gradients; `backward()` computes them; `step()` updates parameters.

The [autograd documentation](https://docs.pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval) explains the distinction. **Self-check:** evaluating under `no_grad()` while leaving dropout in training mode still produces training-mode dropout behavior. Use the correct model mode and gradient context together.

**First artifact:** predictions and parameter shapes, one numerical tolerance check, a train/validation/test manifest, and a deliberately swapped feature column that the reference comparison catches. The CPU route validates model semantics; GPU timing remains a separate experiment.

## Run the complete CPU practical

The [robot-stop classification lab](CPU-Robot-Stop-Lab/Guide.md) uses actual PyTorch 2.14 CPU training and `torch.export`. It includes an exact dependency lock for the tested platform, reproducible synthetic episodes, a nonlinear model, two baselines, threshold selection, error slices, serialization parity, and a deliberate distribution shift. Start here before the larger CNN projects.

For AMP, use `torch.amp.autocast(device_type=...)` and a device-appropriate scaler where needed; do not mechanically enable FP16 on every CPU. Current [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) describes supported devices/dtypes. `torch.export` captures an operator graph with input constraints; `torch.compile` prepares optimized execution; ONNX export adds another representation/runtime boundary. Validate outputs at each boundary and include preprocessing in the contract.

## Projects

1. **Train ResNet-18** on CIFAR-10 from scratch. Profile with `torch.profiler`. Identify top-3 time-consuming ops.
2. **Export to ONNX.** Visualize graph with Netron. Count total ops and parameters.
3. **Post-training quantization** to INT8. Measure accuracy drop and inference speedup.
4. **`torch.compile()`** on a transformer block. Compare eager vs compiled execution time.

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- *Deep Learning with PyTorch* (Stevens, Antiga, Viehmann)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)


## Carry this course into the physical-AI system

**Start from:** A reference model and fixed input batch.

**Build and challenge:** Compare eager evaluation with one supported export/compile path. Include a second input shape and noncontiguous input; record guards, graph breaks, and numerical tolerance.

**Pass forward:** An executable reference plus an operator/shape inventory for compiler work. Continue with [the connected course](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20G%20-%20ML%20Systems%20Engineering/TVM%20Deep%20Dives/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).
