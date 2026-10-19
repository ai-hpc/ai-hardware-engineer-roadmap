# Testing: build evidence in layers

Start with [TESTING.md](../TESTING.md) for commands and the current execution record. The default CPU course runs without a model download; the experimental CUDA engine has no validated pretrained-generation result.

## 1. Define what the test establishes

A parser test asks whether bytes are accepted under a supported format. A numerical test compares an operation with an independent reference. An integration test checks whether components agree about the data between them. A benchmark measures a defined workload after correctness checks pass.

Reading a GGUF file does not validate its tokenizer. A tokenizer round trip does not establish parity with the tokenizer that trained the model.

| Current target | What it executes | What it establishes |
|---|---|---|
| `sampling` | Actual C++ distribution construction and seeded draws | Penalty, temperature, top-k/top-p, validation and replay |
| `memory_contracts` | Actual C++ configuration and budget calculations | Dimensions, byte counts, invalid/overflowing requests |
| `allocator_cpu_shim` | Production allocator control code with a private host allocation shim | Bounds, ownership, cleanup, failed allocation and indexing |
| `gguf_tokenizer_contracts` | Python fixtures invoking native C++ programs | 18 integration cases for the bounded parser and toy tokenizer |

The allocator shim never executes CUDA. Managed-memory access, device synchronization and CUDA allocation behavior need target execution.

## 2. Make wrong answers visible

Suppose a kernel writes only the first 16 of 19 outputs. If the output starts at zero and the expected tail is also zero, a test can pass despite missing work. Poison outputs with NaNs, then require finite values where the contract requires them.

For binary readers, independently encode fixtures instead of copying the reader's assumptions. The CPU lab generates an 832-byte example and checks every truncated prefix. It also tries duplicate keys, invalid UTF-8, inconsistent vocabulary sizes, invalid alignment, overlapping tensors and incomplete quantization blocks.

For numerical code, include more than symmetric inputs. RoPE at position zero cannot expose a sine-sign error. Equal-logit softmax cannot expose every reduction mistake. Include signed values, awkward sizes, nonzero positions and repeated calls.

## 3. Keep the oracle independent

An attention reference should directly calculate scores, stable softmax and weighted values with plainly indexed arrays. Copying the kernel's tiled indexing can give both implementations the same bug.

For a pretrained model, compare in this order:

1. Exact token IDs, special tokens and chat formatting against a pinned reference.
2. Tensor shapes, encodings and selected dequantized values.
3. Layer outputs and logits under justified tolerances.
4. Multi-token cache positions, repeated requests and reset.
5. Greedy generation, then sampling and task-level quality.

The local tokenizer supports only `course.greedy.bytes`. Ordinary BPE and SentencePiece profiles are rejected because their algorithms are not implemented. The fixture omits transformer weights; parsing it is not a forward pass.

## 4. Read the result accurately

The recorded review passed all four CPU CTest entries in Release and ASan/UBSan builds. Twenty-nine C++/CUDA translation units also compiled individually with nvcc for `sm_87` on an x86 host. Compilation does not establish that a Jetson executable links, runs or computes correct logits. See [validation.json](../labs/validation.json).

GPU tests use return code 77 when no usable device/driver exists. A skip is neither a pass nor a numerical failure. Other CUDA errors should fail with a useful diagnostic.

The legacy GPU smoke tests cover few shapes. Extend them with independent references, output poisoning, launch-error checks and synchronization before accepting a kernel.

## 5. Turn a discovery into a regression

Choose one failure: a missing tail, wrong top-p order, overflowing size or stale cache state. Write a fixture with an independently known answer. Record its failure against the faulty code, fix the code, and show the same fixture passing. Add nearby boundaries that test the underlying contract.

**Finish:** retain the command, environment, fixture identity, expected result, actual result and skips. Explain one thing the test does not establish. Continue to the [performance lesson](performance.md) only after defining the workload's correctness gate.
