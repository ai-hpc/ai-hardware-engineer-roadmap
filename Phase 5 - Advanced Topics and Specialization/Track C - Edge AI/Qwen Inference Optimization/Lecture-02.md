# Lecture 2: Quantization with explicit scales and measured quality

**Reviewed 18 October 2026.** Input: Lecture 1's model/config manifest. Output: a numerical quantization report and a reproducible conversion plan. Start with the [portable lab](Lab-Guide.md); checkpoint conversion is optional and needs substantially more storage and memory.

## 1. Calculate bytes before choosing a format

The 4B model's unique parameter count is 4,022,468,096. If every parameter used two bytes, its payload would be **8,044,936,192 bytes**, about **8.045 GB or 7.493 GiB**. Actual files and runtime allocations add layout and format differences. An 8 GB system also needs Linux, other processes, KV, activations, workspace, and reserve; measure available memory rather than assuming a universal “4 GB free.”

Block formats store metadata. For example, 256 Q4_K values occupy 144 bytes, equivalent to 4.5 bits/value; 256 Q6_K values occupy 210 bytes, equivalent to 6.5625 bits/value. `Q4_K_M` is a mixed tensor recipe. Its exact assignment can vary by converter/model/version; it is not a promise that every V and down matrix is Q6_K. The [inference budget](../Edge%20LLM%20Inference%20Internals/Lecture-01.md) provides checked block counts and an explicit illustrative mixture.

A lower byte count can help a bandwidth-limited path, but dequantization, kernels, batching, and quality determine the useful choice. No universal Q4 “sweet spot” or fixed perplexity penalty follows from the model's parameter count.

## 2. Quantize a row by hand

Use the teaching symmetric range `[-7,7]`, one scale per row, and nearest-integer rounding:

\[
s=\max_i|w_i|/7,\quad q_i=\operatorname{clip}(\operatorname{round}(w_i/s),-7,7),
\quad\hat w_i=sq_i.
\]

For an all-zero row choose scale 1 and all-zero integers. This avoids division by zero without changing the reconstructed row. For `w=[0,-1,2,0.1]`, `s=2/7`; Python's ties-to-even rounding gives `q=[0,-4,7,0]`. For `x=[1,-2,0.5,3]`, the float dot product is 3.3 and the reconstructed result is about **3.285714**.

The portable experiment quantizes two rows, reconstructs them, and deliberately swaps their scales. Weight error and output error are different: the latter depends on which inputs multiply each error. A small mean weight error can still matter on a sensitive input channel. This code does not pack nibbles or implement a GGUF/AWQ encoder; it isolates scaling and rounding.

**Exercise:** insert a large outlier in one row and predict which small values round to zero. Compare per-row and one-scale-per-matrix quantization using the same held-out input vectors. Then increase precision for only the sensitive row and account for the additional scale/payload bytes.

## 3. Understand AWQ and GPTQ without conflating their formats

For column-vector notation and positive diagonal `S`,

\[
Wx=(WS)(S^{-1}x).
\]

Before quantization this is exactly the same operation. AWQ uses activation information to search for useful scaling that protects important channels under quantization. The portable lab verifies the float equivalence; it does not implement the AWQ search. Calibration examples must be separate from quality-evaluation examples. See the [AWQ paper](https://arxiv.org/abs/2306.00978).

GPTQ uses a layer-reconstruction objective and approximate second-order information to compensate quantization error as weights are quantized. See the [GPTQ paper](https://arxiv.org/abs/2210.17323). Neither paper establishes that one method always beats the other on this checkpoint, task, hardware, or kernel.

The algorithm, stored format, and consuming kernel are separate contracts. Record group size, scale/zero-point dtype, symmetry, packing, layer exceptions, activation/accumulation dtype, and backend support. A calibration algorithm name alone is insufficient to load the output.

**Current API correction:** [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) is archived and officially deprecated; its notice identifies Torch 2.6.0/Transformers 4.51.3 as its last tested configuration. Do not begin a current environment with an unqualified `pip install autoawq`. Consult maintained [LLM Compressor](https://github.com/vllm-project/llm-compressor) or [GPTQModel integration](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/) and the selected runtime's hardware support matrix. Those conversion paths were source-reviewed, not executed here.

## 4. Optional GGUF conversion with real entry points

The [llama.cpp converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) is a repository script. It is not `python -m llama_cpp.convert_hf_to_gguf`. The compiled tool is `llama-quantize`; an old `./quantize` command assumes an obsolete build layout.

A source-checked optional workflow, from a fresh work directory:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git rev-parse HEAD > runtime-revision.txt
python3 -m venv .venv-convert
.venv-convert/bin/python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
.venv-convert/bin/python -m pip install huggingface-hub
.venv-convert/bin/hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir models/qwen3-source
.venv-convert/bin/python convert_hf_to_gguf.py models/qwen3-source \
  --outfile models/qwen3-f16.gguf --outtype f16
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --target llama-quantize -j 2
./build/bin/llama-quantize models/qwen3-f16.gguf models/qwen3-q4_k_m.gguf Q4_K_M
sha256sum models/qwen3-f16.gguf models/qwen3-q4_k_m.gguf
.venv-convert/bin/python -m pip freeze > conversion-packages.txt
```

This downloads multi-gigabyte weights and retains source plus intermediate plus output. Reserve disk and RAM based on the actual artifacts. The CPU compile option is for conversion; a target GPU runtime needs its own supported backend build. Record the chosen commit and detach at that commit on subsequent reproductions. These commands were checked against current source, but the complete 4B download/conversion was **not run** in this course session. The required CPU lab has no such download.

An importance-matrix calibration path uses `llama-imatrix` and `llama-quantize --imatrix`; consult the [quantization tool documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) at the same commit. Supply an actual calibration file and record its provenance, hash and tokenizer. Do not substitute fabricated universal quality gains.

## 5. Evaluate the deployed artifact

Compare the reference and candidate on the same prompts, template, tokenizer, stop conditions, and sampler. Save logits where possible, but also measure task outcomes. For tool use, run the [strict dispatcher exercise](../Agent%20Tool-Dispatch%20Evaluation%20with%20BFCL/Lecture-01.md) before connecting a model adapter. Report category counts and paired regressions; plausible prose is not evidence of parity.

Separate quality from performance. First establish supported loading and numerical/task behavior. Then measure peak memory, prefill, decode at several context depths, and application latency on the actual device. A smaller file can still use an unsupported or slow kernel.

**Acceptance:** submit the scalar reconstruction/error report, converter/config hashes if conversion was run, and held-out quality results for any actual model claim. Leave target-device fields “not run” when unavailable. Next: [decode optimization](Lecture-03.md).
