# CPU lab: real TVM compilation, vector tails, and export

Build and run two small programs using **Apache TVM 0.27.0.post1** and LLVM on a CPU. Allow 45–75 minutes. You need Python 3.12 or newer (required by the pinned NumPy), a supported wheel platform, and a host C/C++ compiler for shared-library export. The recorded run used Linux x86_64 and Python 3.12.3. No GPU or model weights are needed.

## Install the tested snapshot

From this `labs` directory, create an isolated environment outside the repository:

```bash
python3 -m venv /tmp/my-course-tvm
/tmp/my-course-tvm/bin/python -m pip install -r requirements.txt
/tmp/my-course-tvm/bin/python -m unittest discover -s . -v
/tmp/my-course-tvm/bin/python compiler_lab.py
```

The Linux x86_64 TVM wheel is about 100.5 MB to download; installed size and dependency downloads are larger. The test suite skips when TVM is absent. With TVM installed it requires the pinned version and performs real compilation; it does not silently substitute a mock. See [validation.json](validation.json) for actual checks, versions, and source hashes.

The released wheel uses `tvm.tirx`, `tvm.s_tir`, `T.sblock`, and `get_sblock`. The latest website currently shows a separate `tvm_tirx` namespace that is absent from this wheel. Import errors after mixing those examples are a version mismatch. Do not “fix” them by repeatedly installing unrelated nightlies into this environment.

## 1. Compile an unscheduled operation

Read `affine_tail` in [compiler_lab.py](compiler_lab.py). It computes `B[i]=2*A[i]+1` for exactly 37 float32 elements. Predict values for inputs -2, 0, and 3: outputs are -3, 1, and 7. The function's no-alias declaration is a caller contract, not a runtime promise to support arbitrary overlapping buffers.

The first execution compiles this function for LLVM and compares every output with an independent NumPy expression. CPU execution is synchronous for this lab. A successful compile alone would not prove the values; both steps are tested.

## 2. Split, vectorize, and protect the tail

`schedule_affine(tile)` splits the loop and vectorizes its inner portion. With tile eight, the last group has five valid elements and three invalid indices. Inspect the resulting `T.where` predicate and vectorized loop.

Tests run tile sizes 1, 4, 8, 16, and 64. A NumPy-backed DLPack view has logical length 37 but a larger physical buffer with sentinels. All valid outputs must be correct, and the sentinels must stay untouched. This catches an overrun that a divisible-length example would miss. The backing array remains alive during the call.

The optional timing sweep compiles four candidates, checks their outputs, warms them up, and reports five repeated batches of calls. It includes Python/FFI invocation overhead; 37 elements are too small for a useful GPU or large-model performance conclusion. No particular tile must win. The recorded [sample run](sample_run.json) and [original/lowered IR](ir_examples.json) are available to inspect without installing TVM.

## 3. Lower a graph and reject a wrong shape

`AffineRelu` computes `relu(XW+b)` for X `[M,7]`, W `[7,5]`, and b `[5]`. The tests instantiate M=1,3,9, use mixed-sign data, and compare the complete output with NumPy. Inspect the original Relax graph and the lowered module containing `call_tir`.

The static M=3 program is also called with M=2 and must report a shape mismatch. This is correct behavior. Supporting a dynamic M requires a different symbolic contract and tests; deleting checks is not the solution.

## 4. Export and reload a real artifact

The export test writes a temporary shared library, reloads it through the TVM runtime, constructs a Relax VM, and verifies output values again. A missing host linker can fail this step even when in-process JIT execution works. Keep package/runtime versions, target, shapes, dtype, and artifact hash with any deployed library.

The eight tests establish tiny LLVM CPU correctness, tail handling, shape rejection, and packaging. They do not establish CUDA, MetaSchedule optimization quality, BYOC, MLC-LLM, or microcontroller execution. The lectures describe those extensions with their additional prerequisites. A bounded native MetaSchedule probe completed with one build error, zero valid measured candidates, and no returned schedule; an earlier FFI dependency combination crashed in serialization. [The attempt record](metaschedule_attempt.json) preserves those limits. Neither outcome is counted as successful tuning.

## Explain one failure

Choose one: a missing tail predicate, swapped matrix layout, omitted bias, unsupported shape, or incompatible package namespace. Explain which test detects it and why a compile-only check would miss it. Keep your changed input and expected output as a short learning record.
