# CPU attention lab: masks, streaming normalization and real SDPA gradients

**Executed 18 October 2026.** This lab implements dense and tiled attention in NumPy and compares them with actual PyTorch 2.14.0+cpu SDPA math calls. It covers forward values, gradients, masks, GQA and a small RoPE/cache-position example. The `flash-attn` GPU distributions are not installed or executed.

**Prerequisites:** matrix multiplication, softmax and Python arrays. **Time:** two 60–90 minute sessions. **Artifact:** an executed report, a masked-row explanation and a new independent case. Begin with [Lecture 01](Lecture%2001%20-%20Attention%20Bottleneck%20and%20Roofline.md) and [Lecture 02](Lecture%2002%20-%20Online%20Softmax%20and%20Numerical%20Correctness.md).

## 1. Start with the portable contract tests

```bash
python3 -m unittest discover -s \
  'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs' -p 'test_*.py' -v
python3 'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs/attention_contract.py'
```

Seven standard-library tests cover square/rectangular causal alignment, fixed-tile traffic scaling, tail counts, both halves of KV storage, page-table order/bounds and invalid dimensions. The executable counter's traffic is a model, not a hardware measurement.

Predict the two `[2,5]` masks before running:

```text
upper-left:  1 0 0 0 0       lower-right: 1 1 1 1 0
             1 1 0 0 0                    1 1 1 1 1
```

A `1` means “allowed.” These masks describe different query positions, so different outputs are expected. The earlier assumption that every rectangular `causal=True` API has identical alignment is unsafe.

## 2. Use the isolated CPU framework environment

Reuse the environment from the [PyTorch robot-stop lab](../../../Phase%203%20-%20Artificial%20Intelligence/2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md) if you have already installed its lock. Otherwise:

```bash
python3.12 -m venv /tmp/course-flash-cpu
/tmp/course-flash-cpu/bin/python -m pip install --require-hashes -r \
  'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs/requirements-linux-cpu-py312.txt'
```

The complete wheel/hash lock targets Linux x86_64/Python 3.12 and pins NumPy 2.3.5 and PyTorch 2.14.0+cpu with dependencies. Installing it downloads the CPU framework, not models or datasets. In this course review, the existing `/tmp/course-pytorch-cpu-venv` environment was reused; use that interpreter in the commands below if that is your installed path.

## 3. Run the tiled reference and actual framework tests

```bash
/tmp/course-flash-cpu/bin/python \
  'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs/attention_reference.py'
/tmp/course-flash-cpu/bin/python -m unittest discover -s \
  'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs/optional' -v
```

Fourteen optional tests execute NumPy and PyTorch. The fixture uses original seeded synthetic arrays, with no external dataset or model weights. Recorded results: [NumPy report](labs/measured-numpy-2026-10-18.json), [tests/environment/hashes](labs/validation-2026-10-18.json).

For the `[L=7,S=11,D=8,Dv=5]` lower-right case, the observed maximum error against dense float64 was about `2.22e-16` for float64 tiling and `1.15e-7` with float32 arrays/accumulators. With row tile 3 and column tile 4, the largest score tile contained 12 elements instead of 77. The implementation still stores a dense boolean test mask: this toy's full memory footprint is not a claimed linear-memory production kernel.

The `[L=9,S=2]` lower-right case has seven fully masked rows. Outputs are zero and LSE is `-inf` there; gradients remain finite, with zero Q gradients for those rows. These are explicit fixture semantics and actual tested PyTorch math behavior for the pinned version.

## 4. Inspect the independent gradient evidence

The NumPy backward derives `dV`, `dP`, `dS`, `dQ` and `dK`. A finite-difference test checks selected input elements against a scalar objective. A separate actual PyTorch SDPA call computes autograd gradients with the same `dO`; those match the NumPy formula within the float64 fixture threshold.

These are complementary checks. Copying the same implementation into a “reference” would not be independent evidence. A gradient test for square unmasked tensors alone would miss the decode-alignment bug.

The suite additionally demonstrates:

- Large score changes require rescaling earlier numerator/denominator contributions.
- A legal `scale=0` produces a uniform distribution over allowed keys.
- GQA matches explicit K/V head repetition for the tested input.
- Noncontiguous last-dimension inputs work on the selected math path.
- BF16 input rounding and candidate computation are distinct errors.
- `inference_mode` does not override a nonzero SDPA dropout argument.
- A final-token RoPE calculation must retain its absolute position.

## 5. Run a real SDPA experiment with explicit boundaries

```bash
/tmp/course-flash-cpu/bin/python \
  'Phase 4 - Track C - DL Inference Optimization/02 - Kernel Engineering/FlashAttention Course/labs/sdpa_experiment.py' \
  --device cpu --backend math --dtype bfloat16 --q-len 7 --kv-len 11 --dim 8 --alignment lower_right
```

[Recorded CPU BF16 result](labs/measured-sdpa-bfloat16-2026-10-18.json) reports maximum absolute error about **0.00337** against float64 math on the same quantized inputs. Its tolerance is a declared small-fixture criterion, not a universal BF16 error bound. It records warmup, timing samples, shapes, seed and requested backend. CPU timing describes this process/workload; it does not predict GPU throughput or full-model tokens/s.

Optional GPU extension: in a separately configured compatible CUDA/PyTorch environment, use `--device cuda --backend flash --dtype bfloat16` with supported dimensions. The script fails when CUDA is absent or the selected backend cannot execute; it does not relabel a fallback. An explicit mask may affect backend support. It limits dense-oracle size to keep the reference manageable. No CUDA run, FA2/3/4 kernel timing, Nsight report or serving performance is claimed in the retained evidence.

## 6. Change one thing and predict the result

1. **Alignment:** use query length 1 and key length 9. Upper-left should select only key 0; lower-right permits the whole valid history. A decoder using the wrong alignment can be numerically stable and semantically wrong.
2. **Tile tails:** choose L=10, S=13, row tile 4 and column tile 5. Predict partial tiles and compare with the dense oracle. An out-of-range tile should not invent padded key contributions.
3. **Masking:** mask every key for one row and only the first two blocks for another. The first remains zero; the second acquires a finite result from later valid keys.
4. **Independent transfer:** add the explicit sliding-window convention from Lecture 10. Write the expected small mask by hand and a deliberately wrong variant before testing large arrays.

## 7. Interpret the course's evidence correctly

[The source snapshot](labs/source-evidence-2026-10-18.json) pins the upstream files and records documentation/package checks. Source inspection establishes API and algorithm facts; actual CPU tests establish the named fixture results. GPU compatibility, synchronization, compiler behavior, performance and full-model quality remain separate experiments.

The lab is complete when you can explain a failed alignment, a fully masked row, the numerator's changing scale and the difference between capacity and traffic without reading the answer. Carry those contracts into the [capstone](Lecture%2010%20-%20Capstone.md).
