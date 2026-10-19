# CUDA optimization correctness lab

There are three separate milestones: run a CPU numerical reference, compile complete CUDA source, and execute it on a compatible GPU. Passing one does not establish the next.

**Recorded here, 18 October 2026:** nine CPU tests passed; [cuda_examples.cu](cuda_examples.cu) compiled with CUDA **13.4.92**, GCC **13.3.0**, Linux x86-64, target `sm_80`. Running the binary returned **77**, explicitly skipping GPU execution because a usable CUDA device/driver was unavailable. See [validation.json](validation.json) for wheel hashes, source hash and limitations.

## 1. Run the CPU mathematical exercise

From the repository root:

```bash
python3 "Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/CUDA-Advanced-Optimization/labs/contracts.py"
python3 -m unittest discover -s "Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/CUDA-Advanced-Optimization/labs" -p "test_*.py" -v
```

Only Python's standard library is required. The default reports:

| Check | Result to explain |
|---|---|
| Normalize `[1,2,4]` in a four-lane block | First value approximately `-1.06904` |
| Deliberately include the padded lane in centered variance | Incorrect first value approximately `-0.72627` |
| GELU/dropout/residual array traffic, 1003 FP32 elements | 28,084 separate bytes versus 12,036 fused bytes |
| Four finite tiles across three illustrative workers | 491, 256, 256 elements |

Read [contracts.py](contracts.py) and [the tests](test_contracts.py). The worker schedule is a serial coverage model. It does not simulate CUDA scheduling, atomic memory visibility or execution time. The traffic model excludes masks/RNG and caches. The fixed-mask comparison isolates numerical fusion from random-number generation.

**Guided variation:** use `--elements 257`. Predict two tiles, 256 and one element, and recompute the traffic before running. Change the normalization row to five values in an eight-lane block and show that the faulty implementation still fails.

## 2. Compile with a compatible CUDA toolkit

With an existing toolkit and supported host compiler, work from this `labs` directory:

```bash
nvcc --version
g++ --version
nvcc -std=c++17 -O2 -arch=sm_80 cuda_examples.cu -o /tmp/course-cuda-examples
```

`sm_80` is the tested compilation target, not a claim that every GPU can execute that binary. Select a supported target for your device and record it. A CUDA toolkit can compile source without an attached GPU; executing it requires the appropriate device and driver.

The review used an isolated, hash-pinned compiler installation rather than changing the system toolkit:

```bash
python3 -m venv /tmp/course-cuda-toolchain-venv
/tmp/course-cuda-toolchain-venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-cuda.txt
```

[requirements-cuda.txt](requirements-cuda.txt) pins the tested Linux x86-64 wheels, including the transitive runtime, CRT and NVVM components. It does not install a GPU driver. For that Python 3.12 environment, the actual compiler path was:

```bash
/tmp/course-cuda-toolchain-venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc -std=c++17 -O2 -arch=sm_80 cuda_examples.cu -o /tmp/course-cuda-examples
```

A different Python version changes the site-packages path. Use [NVIDIA's installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for supported host/toolchain combinations and other installation methods.

## 3. Execute only when the hardware milestone is available

```bash
/tmp/course-cuda-examples
```

The program returns:

- **0:** all runtime comparisons passed on the printed device;
- **77:** device/driver unavailable; GPU validation was skipped;
- **1:** a CUDA error or numerical mismatch occurred.

A successful GPU run checks graph replay on changing inputs, fused affine/ReLU, cooperative block partial reductions, and finite atomic tile claiming. It uses lengths `0,1,31,32,33,255,256,257,1003`. There are 64 GPU comparisons plus one host empty-input check. The host reference uses deliberately small dyadic values; this does not establish numerical behavior for arbitrary large reductions.

Each implementation's result buffer is filled with NaNs before execution. This prevents a missing output store from inheriting the preceding implementation's correct result. The checker rejects nonfinite outputs. On a GPU, temporarily skip the fused launch to confirm the checker reports a failure; that mutation experiment has not been executed on this host.

The host buffers are ordinary vectors; asynchronous copy calls may internally stage or block depending on memory type and runtime behavior. This correctness harness makes **no transfer-overlap or performance claim**.

## 4. Trace ownership and participation

Choose one nonempty size and draw the buffer lifetimes. Explain why capture can occur before the input values are copied, why new input contents can use the same graph, and why host comparison happens after stream synchronization.

In the reduction, every block has 256 participants even at the input tail. In the finite worker, a shared tile ID plus a block barrier broadcasts a claim to the whole block. Explain why replacing that barrier with a warp shuffle would not preserve the same scope.

**Independent transfer:** add a square operation between affine and ReLU. Update the mathematical reference, the ordinary graph and a fused equivalent. Include changing inputs, zero length and tail cases. Establish correctness before designing any timing experiment.

## Completion

Submit the predicted numbers, deliberate padding failure, CPU test output and source revision. The compile milestone additionally needs compiler/target and build output. The GPU milestone needs actual device information, program output and an error-free reference comparison. Keep skipped outcomes explicit; no latency or power result is supplied by this lab.

Return to [CUDA Graphs](../01-CUDA-Graphs.md), [Cooperative Groups](../02-Cooperative-Groups.md), [Persistent Work](../03-Persistent-Kernels.md), [Fusion](../04-Kernel-Fusion.md) or [Warp Specialization](../05-Warp-Specialization.md).
