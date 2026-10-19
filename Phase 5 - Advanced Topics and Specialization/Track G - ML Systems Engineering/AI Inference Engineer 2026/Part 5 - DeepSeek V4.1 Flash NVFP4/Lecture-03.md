# Lecture 03 — NVFP4 representations and numerical checks

**Prerequisites:** Lecture 02, floating-point representation. **Time:** 3 hours, CPU; GPU kernel extension optional. **Outcome:** verify the meaning of packed values and scales before comparing speed.

## Values, scales, and layouts

A useful conceptual model is:

```text
reconstructed_weight[i] = fp4_value[i] * block_scale[block(i)] * tensor_scale
```

The payload code, block size, scale type, scale direction, tensor grouping, and physical layout are separate parts of the contract. The [NVIDIA quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html) explains block-scaled representations. Actual loaders may store inverse scales or reorganize blocks for a kernel; verify their equations.

E2M1 has a small nonuniform value set. It is not signed INT4. A multiplication by the wrong scale can still produce finite activations, so “no NaNs” is a weak correctness test.

## Storage arithmetic

For a large tensor, excluding padding and global-scale overhead:

```text
4-bit payload + one 8-bit scale per 16 values = 4.5 bits/value
4-bit payload + one 8-bit scale per 32 values = 4.25 bits/value
```

The first arrangement can use more bytes even though both are called FP4. Alignment, packing, scale layout, and unconverted modules further change checkpoint size.

Converting a block-32 format to a block-16 format may preserve reconstructed weights only if every scale can be represented with the chosen global scale and rounding behavior. A zero-error check on one block is not a proof for the entire checkpoint. Converting a previously FP8 component to FP4 is a different operation with a different quality question.

## Worked CPU exercise

Run:

```bash
python3 labs/fp4_layout.py
```

The example enumerates E2M1 codes, packs and unpacks a known sequence, applies explicit positive scales, and checks storage arithmetic. It is a **format teaching fixture**, not a production NVFP4 converter: it does not implement calibration, activation quantization, scale swizzling, or a CUDA kernel.

Trace the codes for +0.5, -1, and +6 by hand. Swap high and low nibbles and observe why a symmetric test vector could miss this bug. Use an asymmetric vector when testing a real loader.

## Build the validation ladder

1. **Metadata:** shape, storage dtype, block axis, scale names, and payload length agree.
2. **Reconstruction:** compare selected blocks against the source, including zeros, extremes, and tails.
3. **Operator:** compare outputs with identical inputs and documented accumulation dtype.
4. **Layer:** inspect activation/error distributions and expert selection.
5. **Model:** use fixed inputs, log-probability comparisons where available, and held-out tasks.

Report maximum absolute error, relative error with a nonzero denominator floor, and the fraction above your tolerance. Keep exact-repack checks separate from lossy quantization thresholds. For a purported exact transcode, a nonzero difference is a bug to investigate, not a tolerance to relax.

## Lab and transfer

Create two negative tests: an incorrect global scale and a block-axis swap. Show which rung detects each. For a real checkpoint, record whether the conversion touched experts, shared experts, attention, vision, Engram, and speculative heads; do not infer this solely from its name.

**Expected artifact:** a format contract, fixture output, and numerical checks at the highest rung your hardware permits. GPU absence is not permission to claim model parity.

**Independent exercise:** explain why bit-exact weight reconstruction can coexist with different operator outputs when the activation format or accumulation order changes.

## Self-check and exit criteria

You should distinguish weight-only reconstruction from activation-aware execution, derive bits/value including scales, and demonstrate a check that fails under deliberate corruption. If a fast kernel only matches when you silently change the reference, the gate is incomplete.

[Previous](Lecture-02.md) · [Overview](README.md) · [Next: baseline](Lecture-04.md)

**TensorRT source boundary (2026-10-18):** linked `latest` quantization pages describe TensorRT 11.4. They explain representation/support in that release, not compatibility of this case study's historical loader or a JetPack 10.x runtime. TensorRT 11.4 does not support JetPack; use [NVIDIA's platform migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html) and the selected runtime's exact version when reproducing an engine.
