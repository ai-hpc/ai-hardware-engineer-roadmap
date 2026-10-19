# CPU lab: sampling, GGUF, and tokenizer contracts

**Time:** two 60-minute sessions. **Tools:** CMake 3.20+, a C++17 compiler, and Python 3. No CUDA installation, model download, or Jetson is required. Work from the course repository root.

You will build real C++ components, inspect a binary model-container fixture, trace a toy tokenizer, and calculate a sampling distribution. By the end, you should be able to explain why a file that parses successfully may still be unusable by an inference engine.

## 1. Build and run the actual code

```bash
cmake -S Projects/jetson-llm-runtime -B /tmp/jllm-course -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jllm-course -j2
ctest --test-dir /tmp/jllm-course --output-on-failure
```

The six CTest entries cover native sampling, memory planning, allocator control
flow, attention reference math, decode scratch reuse, and **18 Python-driven
native integration tests**. The Python suite creates independent binary fixtures
and calls the compiled programs; it does not substitute a Python implementation
for the C++ parser. The [attention/layout lab](Attention-Layout-Lab.md) separates
these CPU checks from its compiled but unexecuted CUDA parity harness.

Open [sample.cpp](../src/engine/sample.cpp), [gguf.cpp](../src/engine/gguf.cpp), [model_config.cpp](../src/engine/model_config.cpp), and [tokenizer.cpp](../src/engine/tokenizer.cpp). Each component has a narrower contract than “run an LLM.” The [test source](test_cpu_runtime.py) shows exactly which inputs are exercised.

## 2. Predict a sampling result

Suppose logits are `log(0.1), log(0.6), log(0.3)`. At temperature 1, before filtering, softmax gives probabilities 0.1, 0.6, 0.3 for token IDs 0, 1, 2.

The implemented policy is:

1. Apply the repetition penalty once to each distinct recent token.
2. For temperature zero, choose the highest adjusted logit; break ties by lower ID.
3. Otherwise sort by adjusted logit, apply top-k if enabled, and normalize probabilities within that retained set.
4. Retain the smallest sorted prefix reaching top-p, then renormalize it.
5. Sample from that distribution with a seeded generator.

This order is explicit because different inference systems can define different filtering/renormalization policies. Match the reference system when comparing generation; do not assume all libraries compose these controls identically.

| Settings | Expected retained distribution |
| --- | --- |
| `top_k=0`, `top_p=1` | ID 1: 0.6; ID 2: 0.3; ID 0: 0.1 |
| `top_k=0`, `top_p=0.5` | ID 1: 1.0 |
| `top_k=2`, `top_p=0.65` | ID 1: 1.0, because 0.6/(0.6+0.3) ≈ 0.667 |
| `top_k=2`, `top_p=0.8` | ID 1: 2/3; ID 2: 1/3 |

**Exercise:** find the test that would fail if top-k being disabled also skipped sorting. Then find the test that would fail if top-p used the original unnormalized top-k mass.

With logits `[4, 1.5]`, repetition penalty 2, and recent IDs `[0, 0]`, the adjusted logits are `[2, 1.5]`. Applying the penalty twice would incorrectly produce `[1, 1.5]`. At temperature zero, this error changes the selected token.

The sampler leaves its input logits unchanged. NaNs, positive infinity, an entirely masked distribution, invalid recent IDs, negative top-k, and invalid temperature/top-p/penalty values are rejected. Negative infinity is permitted as an excluded token when another finite logit exists.

**Observed sample check:** 20,000 draws with seed 7 gave counts 1,982 / 11,935 / 6,083 on the review host. The test checks a tolerance around the expected frequencies, not that every C++ standard library must produce those exact counts. Seed replay is checked within the same implementation.

## 3. Create a small inspectable GGUF file

```bash
python3 Projects/jetson-llm-runtime/labs/make_fixture.py /tmp/jllm-course/toy.gguf
/tmp/jllm-course/gguf-inspect /tmp/jllm-course/toy.gguf
/tmp/jllm-course/gguf-inspect --config /tmp/jllm-course/toy.gguf
```

The generator refuses to overwrite an existing file. Choose a new path for a second run. Expected results:

```text
metadata=13 tensors=1 alignment=32 data_offset=736
tensor_type=1 bytes=96
layers=2 heads=4 kv_heads=2 head_dim=2 hidden=8 vocab=6 tensor_bytes=96
```

The complete file is 832 bytes. It contains metadata, one F16 embedding tensor of shape `[8, 6]`, alignment padding, and 96 zero-valued payload bytes. It contains **no trained weights** and intentionally omits the transformer-layer weights. It is sufficient for the parser/configuration exercise; it is not a working language model.

GGUF tensor shape order matters: the fastest-varying dimension is first in this representation. The embedding shape supplies hidden width 8 and vocabulary size 6, so `8 × 6 × 2 = 96` bytes. File bytes include metadata and padding as well; they are not equal to tensor payload bytes. See the [upstream GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) for the full format.

## 4. Follow the parser's bounds

The supported profile is little-endian GGUF version 3 with scalar, string and flat-array metadata. Numeric arrays are bounds-checked and skipped; string arrays are retained. Nested arrays and unsupported types fail explicitly. The implementation limits metadata consumption to 64 MiB, individual strings to 16 MiB, metadata entries to 100,000, tensor entries to 65,536, and array elements to 1,000,000.

Strings must be valid UTF-8. Metadata keys use ASCII lowercase letters, digits and underscores in nonempty dot-separated components; tensor names are limited to 64 UTF-8 bytes. Alignment must be a power of two from 8 bytes through 1 MiB. The full GGUF specification permits multiples of eight; this reader deliberately supports the narrower power-of-two profile.

These are deliberate teaching limits, not universal GGUF format limits. The reader checks lengths against both the remaining file and the declared budget before allocating or skipping. Tensor multiplication is checked for overflow; ranks are limited to 1–4; offsets must meet alignment; supported payload spans must fit and not overlap.

**Worked quantization check:** Q4_K uses 144-byte blocks representing 256 elements. A `[256, 2]` tensor therefore occupies 288 bytes, or 4.5 bits per weight in that block representation. A model named `Q4_K_M` can contain several tensor encodings; its entire file is not exactly `parameter_count × 0.5` bytes.

**Exercise:** inspect the fixture tests for a truncated string array, a tensor with five dimensions, a quantized row of width 128, and a tensor whose payload extends past EOF. Explain the failure before running the case. Do not “recover” an unknown type by skipping a guessed eight bytes: that loses the location of every following field.

## 5. Trace the toy tokenizer

```bash
/tmp/jllm-course/test_tokenizer /tmp/jllm-course/toy.gguf abaX
```

The output IDs are `0 4 2 5`: BOS, `ab`, `a`, and the byte-fallback token for `X`. The test verifies decoding returns `abaX`. It then deliberately reloads a nonexistent file to check that failure clears the old vocabulary; the expected diagnostic says the file cannot be opened.

The metadata declares `course.greedy.bytes`. This is an explicit custom teaching profile. It greedily matches the longest ordinary vocabulary string and uses a byte token when available. It rejects input bytes absent from its vocabulary instead of silently deleting them. Byte-escape spellings are reserved representation strings, not directly matched literal text.

This is not byte-pair encoding. A pretrained tokenizer may need normalization, pretokenization, ranked merges, byte mapping, special-token policy and a chat template. The local loader rejects ordinary `gpt2`, `llama`, and `qwen2` tokenizer profiles because those contracts are unimplemented. Renaming their metadata to the toy profile does not make them compatible.

## 6. Separate metadata from architecture support

The configuration reader uses exact architecture-prefixed keys and explicit head dimensions when provided. It checks that vocabulary and embedding shape agree. It reports actual parsed tensor payload bytes, rather than guessing a parameter count and quantization overhead.

One fixture has hidden width 8, four query heads, and explicit head width 4. The reader reports those values without assuming `head_width = hidden_width / query_heads`. Reading that configuration still does not establish that the CUDA forward path implements its projections or architecture.

Complete model support needs exact operations, normalization, RoPE convention, tensor encodings, cache layout, tokenization and numerical parity. Each belongs in its own acceptance test. The [roadmap](../ROADMAP.md) names the remaining GPU/runtime milestones.

## 7. Calculate a memory limit before allocating

The memory lesson uses MiB (`2^20` bytes), with the reserves stated separately. Suppose a device budget is 8,192 MiB, with 1,024 for the OS, 256 for CUDA, 2,048 for weights, 128 for scratch, and 256 for safety. The remaining KV budget is:

```text
8192 - 1024 - 256 - 2048 - 128 - 256 = 4480 MiB
```

For 24 layers, eight KV heads, head width 128, and FP16 elements, one token needs `2 × 24 × 8 × 128 × 2 = 98,304` bytes. The factor two at the start accounts for K and V. The floor of `4480 × 2^20 / 98304` is **47,786 tokens**. This is a storage estimate; the model's supported context, allocation behavior, other processes, and latency may impose smaller limits. Query heads do not replace KV heads in this grouped-query calculation.

All budget fields must be nonnegative. Checked addition and multiplication reject impossible values before they can overflow; a negative headroom result cannot admit an allocation. Test an `INT64_MAX` reserve as well as an ordinary overcommit: neither should create fictitious free space.

CMA free space is informational: it must not simply be added to Linux `MemAvailable`, which would double-count memory. Already allocated KV and future KV capacity are different quantities. Inspect [the native memory tests](../tests/test_memory_contracts.cpp) for those cases and for invalid/overflowing dimensions.

The [allocator test](../tests/test_allocator_contracts.cpp) compiles the actual scratch-pool and KV-cache control code against a bounded host-memory shim. It checks alignment, exact capacity, rejected requests, failed reinitialization, cache offsets and destructor cleanup. Owning pool objects cannot be copied. The shim does **not** exercise CUDA allocation, device access, coherence or asynchronous completion. On a real device, finish every operation using a buffer before resetting or destroying its owner; a C++ destructor does not establish that the GPU is finished.

**Exercise:** a 1,024-byte scratch pool rounds allocations to 256-byte boundaries. Requests of 1 and 257 bytes consume 256 + 512 = 768 bytes. Why must the next request for 257 bytes fail while leaving the offset unchanged? **Answer:** its rounded size is 512 bytes, exceeding the remaining 256 bytes. Changing the offset on failure would corrupt subsequent allocation bookkeeping.

## 8. Measure a component without mislabeling it

After the Release build, run:

```bash
bash Projects/jetson-llm-runtime/scripts/bench.sh /tmp/jllm-course
```

The wrapper generates its own temporary fixture, calls [the native benchmark](../src/benchmark_cpu.cpp), and emits JSON with host, executable and fixture identities. Seven batches each contain 100 parser calls and 30 sampling calls. The sampler uses 32,000 synthetic logits, top-k 40, top-p 0.9 and temperature 0.7. Both components are warmed before timing; the small file will normally be cached.

The [recorded run](benchmark-example.json) had median batch means of about **14.2 µs per parse** and **2.90 ms per sampling call** on this shared x86 host. The sampler sorts the vocabulary, so increasing vocabulary length changes its work. These are CPU component measurements; no transformer forward pass occurs. Their reciprocal is not an LLM tokens-per-second result, and seven batch means do not establish a p99 latency bound.

**Exercise:** double the synthetic vocabulary size in a separate copy, rebuild, and predict which component changes. Record actual samples, compiler/build mode and host load. Explain why a Debug or sanitizer build is useful for correctness but inappropriate as the only optimized-performance comparison.

The optional `scripts/profile.sh` accepts an explicit command and a new trace prefix. It requires an installed, compatible Nsight Systems. It propagates failures and refuses an existing trace filename. Profiling a command does not establish that its output is numerically correct; validate the workload first.

## 9. Submit evidence

Run the [sanitizer workflow](../TESTING.md#sanitizers), then submit:

- Your predicted and observed sampling distributions.
- The 96-byte and 288-byte tensor calculations.
- One malformed-file case and the expected rejection.
- The toy token sequence and an explanation of why it does not prove BPE support.
- Actual test output, host/compiler versions, and any skipped or unexecuted stages.

The [recorded validation](validation.json) distinguishes native CPU execution from CUDA compilation and target execution. This exercise completes a useful runtime foundation without claiming that untested pretrained inference is ready.
