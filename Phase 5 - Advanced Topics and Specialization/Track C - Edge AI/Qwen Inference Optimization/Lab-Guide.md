# Qwen CPU lab: shapes, positions, quantization, and serving arithmetic

This lab supplies executable work for all six lessons. The required path uses Python 3.10+ and the standard library. The optional path uses a current Transformers implementation and a tiny random model on CPU. Neither path downloads pretrained Qwen weights or needs a GPU.

## 1. Run the portable experiments

From the repository root:

```bash
cd 'Phase 5 - Advanced Topics and Specialization/Track C - Edge AI/Qwen Inference Optimization'
python3 labs/optimization.py
python3 -m unittest discover -s labs -p 'test_*.py' -v
```

Read [implementation](labs/optimization.py), [14 tests](labs/test_optimization.py), and [recorded output](labs/optimization-reference.json). The exercises progress in this order:

| Lesson | Experiment | What counts as understanding |
|---|---|---|
| 1: architecture | RoPE norm and relative-position identity, wrong pairing | explain why a norm-preserving operation can still use the wrong pair layout |
| 2: quantization | INT4 row scales, reconstruction, corrupted scales | distinguish weight error from input-dependent output error |
| 3: decode | correct shapes plus [separate budget lab](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) | recompute bytes when the tensor mixture changes |
| 4: parallelism | split reduction dimension and sum partial outputs | explain why averaging is incorrect |
| 5: serving | speculative accepted prefixes and capacity headroom | include draft cost and use measured capacity inputs |
| 6: GEMM | non-square padded column-major product and beta | derive offsets and preserve output padding |

Recorded quantized output is `[3.2857142857, 0.1]` versus reference `[3.3,0.1]`. The speculation expectation is 2.1607097 tokens/round, and the fixed-seed simulation gives 2.15674. The capacity example needs two replicas under its explicitly hypothetical inputs. These are arithmetic results, not model/hardware benchmarks.

## 2. Run the optional current-library experiment

Create an isolated environment inside this lesson directory:

```bash
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.venv-qwen/bin/python -m pip install -r labs/requirements-transformers.txt
.venv-qwen/bin/python -m pip check
.venv-qwen/bin/python labs/transformers_cpu.py
```

The [script](labs/transformers_cpu.py) was run on 18 October 2026 with Python 3.12.3, PyTorch 2.14.0+cpu, and Transformers 5.20.1. The recorded environment reused an already installed CPU PyTorch wheel through a local environment path and installed Transformers in its own environment; dependency checking passed. The commands above describe a fresh installation route. Keep its package list with your result.

First, the two supplied checkpoint configs instantiate full architectures on PyTorch's **meta device**. This creates parameter shapes without allocating parameter data. It verifies 4,022,468,096 unique 4B parameters and 72,706,203,648 unique 72B parameters, plus Q projection dimensions and embedding tying. This step does not perform pretrained inference.

Second, a newly initialized two-layer Qwen3 runs actual FP32 CPU inference with vocabulary 128, hidden width 32, four query heads, two KV heads, and head dimension 16. It deliberately exercises query width different from hidden width. The script checks:

- full-sequence logits versus cached-prefix plus one-token decoding;
- causal invariance of earlier logits when future tokens change;
- a deliberately wrong decode position, which must produce a detectable difference;
- the resulting cache length.

[Recorded results](labs/transformers-reference.json): cache/full maximum absolute error `8.94e-8`, causal difference `0`, wrong-position difference `0.06964`, final cache length 5. The script uses declared tolerances rather than expecting bit identity. It records a hash of the installed Qwen3 implementation. No random tiny model is a meaningful language-quality benchmark.

## 3. Complete the learning tasks

1. Calculate Q/K/V/O shapes and the exact norm contribution from the supplied configs. Reconcile with the meta-device result.
2. Swap RoPE pairing or force decode position zero. Explain why one invariant may pass while the end-to-end cache check fails.
3. Add a quantization outlier, preserve calibration/test separation, and report both weight and output error.
4. Change TP rank count in a divisible toy matrix. Return partial outputs and demonstrate the reduction.
5. Vary speculative acceptance and draft cost. Find a case that is slower than the target alone and explain the result.
6. Change the GEMM leading dimensions and batch gaps while preserving the logical matrices. Show the same result and a deliberate layout failure.

Submit code, raw JSON, package/source hashes, and a short explanation of one injected failure. A learner without optional packages can complete the scalar tasks and label the Transformers section **not run**. Pretrained conversion, model-quality evaluation, CUDA/cuBLAS, Jetson timing, and multi-GPU serving remain separate optional experiments; do not fill those results from these CPU checks.
