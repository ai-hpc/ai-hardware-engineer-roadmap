# Chapter 2: FP4 Numerics and Checkpoint Contracts

**Previous:** [Architecture](01-Blackwell-Architecture.md) · **Next:** [Single-GPU deployment](03-Single-B200-Qwen-Inference.md)

## Four bits are not a complete format

A quantized checkpoint includes element encoding, scales, scale grouping, packing order, higher-precision tensors, and sometimes calibration or transformation metadata. The runtime must implement the same contract. A GGUF Q4 format, MXFP4, and NVFP4 cannot be interchanged by relabeling the dtype.

| Property | MXFP4 teaching model | NVFP4 teaching model |
|---|---|---|
| Element | FP4 E2M1 | FP4 E2M1 |
| Typical block | 32 elements | 16 elements |
| Block scale | E8M0 power-of-two scale | FP8 E4M3 scale |
| Additional scale | Follow the checkpoint specification | FP32 tensor-level scale in the documented recipe |
| Nominal storage per element before padding/other tensors | `4 + 8/32 = 4.25` bits | `4 + 8/16 = 4.5` bits, plus tensor scale overhead |

The NVFP4 description follows NVIDIA's documented two-level scaling. Specific training recipes may add transformations or use different scaling layouts; inspect the exported inference checkpoint rather than assuming every recipe produces the same serialized tensors. [Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html), [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html).

## Worked storage estimate

For an illustrative 72-billion-element tensor collection:

- BF16 payload: `72e9 × 2 = 144e9` bytes.
- FP4 payload only: `72e9 × 0.5 = 36e9` bytes.
- MXFP4 with one byte of scale per 32 values: `38.25e9` bytes before alignment and unquantized tensors.
- NVFP4 with one byte of block scale per 16 values: `40.5e9` bytes plus tensor scales, alignment, and unquantized tensors.

These are lower-level storage calculations, not the exact size of a named Qwen checkpoint. Count each tensor in the actual artifact, preserve tied/untied embedding semantics, and separately budget runtime repacking, graph pools, scratch space, and KV cache. `checkpoint file size` and `GPU resident allocation` can differ.

## Quality is an experiment

Softmax has a bounded Jacobian for absolute logit perturbations; small changes near ties can still change selected tokens, and large Q/K errors can change context weighting. Do not claim universal exponential amplification or universal error averaging. A high-precision V projection or FFN down projection is a candidate ablation, not a universal prescription. Transformer Engine does not automatically choose a quality-optimal precision for every tensor in an arbitrary model.

Use the same model revision, tokenizer, prompt template, inputs, and decoding protocol for the reference and quantized model. First compare logits or hidden states on fixed inputs. Then compare task metrics, stratified by long context, language, domain, and difficult examples. A low average tensor error can hide a consequential action error.

For VLA transfer, add per-action-dimension error, coordinate/normalization checks, and closed-loop task success. A text benchmark improvement does not establish robot control parity.

## Keep tool roles distinct

- **Transformer Engine** provides low-precision primitives/recipes used in supported frameworks.
- **Model Optimizer or a checkpoint publisher** creates an exported quantized artifact with metadata.
- **TensorRT-LLM, vLLM, or another runtime** consumes supported artifacts and executes its available kernels.
- **Your evaluation harness** decides whether accuracy and latency are acceptable for the workload.

Use the exact release's [TensorRT-LLM quantization matrix](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md) and [Model Optimizer export examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq). Record their revisions. A hardware-supported FP6 type does not establish a production FP6 checkpoint path in the chosen engine.

## Lab: checkpoint audit before serving

Create a manifest with: publisher/repository, immutable revision, license, tokenizer revision, element format, scale dtypes and shapes, block size, packing layout, excluded layers, calibration data provenance, and supported runtime/architecture. Missing fields become questions to resolve before a timing comparison.

Select one matrix, dequantize a small block according to its documented format, and compare it to the higher-precision source. Test an all-zero block, an outlier, saturation, and values near rounding boundaries. Then load the complete checkpoint and inspect executed kernel names to distinguish native low-precision arithmetic from a higher-precision fallback.

**Deliverable:** a format manifest and quality report. Chapter 3 uses its measured resident weight size. The [DeepSeek checkpoint course](../../../../Track%20G%20-%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026/Part%205%20-%20DeepSeek%20V4.1%20Flash%20NVFP4/README.md) applies the same identity-first method to its named target; verify availability and metadata there before claiming a particular model/format combination works.


## CPU exercise with answer guidance

Run the [existing real PyTorch FP4 calibration lab](../../../../Track%20G%20-%20ML%20Systems%20Engineering/Hardware-Aware%20LLM%20Quantization/CPU-Quantization-Lab.md). It packs codes
and E4M3 scales, checks zeros/tails/ties, reloads artifacts and evaluates three
policies on a trained nonlinear network. Explain why output-aware layer calibration
can improve its local objective yet worsen full-model KL or shifted task behavior.
The recorded experiment contains that negative result.

Transformer Engine's linked NVFP4 page currently describes **2.20.2 training
recipes**. Random rotations, stochastic rounding and training transpose handling
are not automatically required in a PTQ inference artifact. TensorRT's linked
`latest` page describes **11.4** graph semantics; neither page verifies a
TensorRT-LLM/vLLM model loader, and TensorRT 11.4 is not the JetPack-supported line.
The [source record](Source-Evidence.md) keeps these roles and versions separate.
