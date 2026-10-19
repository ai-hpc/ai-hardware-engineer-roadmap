# Lecture 1: Read a Qwen checkpoint before optimizing it

**Reviewed 18 October 2026.** These are fixed case studies: **Qwen3-4B-Instruct-2507** and **Qwen2.5-72B-Instruct**. They are not interchangeable with every Qwen release. Start with the [lab guide](Lab-Guide.md); the required metadata and scalar exercises run without downloading model weights.

## 1. A checkpoint is a numerical contract

Record configuration, weights, tokenizer, chat template, and model implementation revision. A model name alone does not define dimensions, bias terms, rotary positions, or decoding behavior. The [4B model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) describes this July 2025 checkpoint as a **non-thinking** instruction model. Do not inject the thinking template of a different Qwen3 release.

The supplied snapshots are [4B configuration](labs/qwen3-config.json) and [72B configuration](labs/qwen25-config.json), retrieved from revisions `cdbee75f17c01a7cc42f958dc650907174af0554` and `495f39366efef23836d0cfae4fbe635880d2be31`. Sources and hashes are in the [source record](labs/source-review.json).

| Field | Qwen3-4B-Instruct-2507 | Qwen2.5-72B-Instruct |
|---|---:|---:|
| Hidden width | 2,560 | 8,192 |
| FFN intermediate width | 9,728 | 29,568 |
| Layers | 36 | 80 |
| Query/KV heads | 32 / 8 | 64 / 8 |
| Head dimension | 128, explicit | 128, hidden/query-head ratio |
| Vocabulary rows | 151,936 | 152,064 |
| Embedding/head tied | yes | no |
| Q/K/V projection biases | absent | present |
| Per-head Q/K normalization | present | absent |
| RoPE base | 5,000,000 | 1,000,000 |
| Configured maximum positions | 262,144 | 32,768 |
| Released RoPE scaling | null | no scaling entry |

Model implementation supplies behavior beyond a few config values. For example, Qwen3 has per-head Q/K RMSNorm parameters. Counting only the two residual-stream norms per layer omits them.

## 2. Derive a complete layer

For stored PyTorch weights, linear layers have shape `[output,input]` and compute `y=W x`. With hidden width `d`, query heads `Hq`, KV heads `Hkv`, head width `h`, and FFN width `f`:

```text
Q: [Hq*h,d]       K,V: [Hkv*h,d]       O: [d,Hq*h]
gate,up: [f,d]    down: [d,f]
FFN(x) = Wdown [SiLU(Wgate x) * (Wup x)]
```

For the 4B case Q is `[4096,2560]`, K/V each `[1024,2560]`, and O `[2560,4096]`. `Hq*h` does not equal `d` here. Concatenated gate/up has **19,456** outputs, because `2*9728=19456`.

The attention path is: residual-stream RMSNorm; Q/K/V projections; Q/K per-head normalization for Qwen3; RoPE on Q/K; causal grouped-query attention; output projection; residual addition. The FFN then applies the second residual-stream RMSNorm, gated feed-forward computation, and residual addition. RMSNorm is `x * w / sqrt(mean(x²)+epsilon)`; there is no unexplained extra scale factor.

Four query heads share each KV head in the 4B case; eight share each in the 72B case. This affects cache storage independently of query width. Biases, Q/K norms, and head mapping must survive conversion or fusion.

## 3. Count parameters without allocating their weights

The optional [Transformers CPU exercise](labs/transformers_cpu.py) builds full model **metadata on the meta device**: shapes exist but parameter data is not allocated or loaded. With Transformers 5.20.1, the observed counts are:

| Model | Unique parameters | Named parameter tensors |
|---|---:|---:|
| Qwen3-4B-Instruct-2507 | 4,022,468,096 | 398 |
| Qwen2.5-72B-Instruct | 72,706,203,648 | 963 |

For Qwen3, each layer has seven projection matrices plus four normalization vectors. There are `36*11+2=398` unique parameter tensors with a tied head. For Qwen2.5, each layer has seven projection matrices, three QKV biases, and two norm vectors; two embedding/head tables and a final norm give `80*12+3=963`. A serialized format can represent shared aliases differently; compare unique parameters and artifact tensor inventory explicitly.

Tying the head saves another `151936*2560=388,956,160` stored parameters. It does not remove the output logits computation. See the [worked inference budget](../Edge%20LLM%20Inference%20Internals/Lecture-01.md).

## 4. Derive and test RoPE

For a head of even width `h`, pair coordinate `i` with `i+h/2`. Let `omega_i=base**(-2*i/h)` and position `p`:

\[
\begin{bmatrix}x'_i\\x'_{i+h/2}\end{bmatrix}
=\begin{bmatrix}\cos(p\omega_i)&-\sin(p\omega_i)\\
\sin(p\omega_i)&\cos(p\omega_i)\end{bmatrix}
\begin{bmatrix}x_i\\x_{i+h/2}\end{bmatrix}.
\]

Pair 0 has the **highest** frequency, one radian per position; increasing `i` lowers it. Position 1 already rotates that pair by one radian. A layout error need not wait ten generated tokens to matter.

`rope()` in [optimization.py](labs/optimization.py) implements split-half pairing. Tests verify norm preservation and the relative-position identity `dot(R(p)q,R(t)k)=dot(q,R(t-p)k)`. These properties test the mathematics; comparing to the pinned model implementation tests the layout convention. A converted runtime may reorder coordinates and weights consistently, so inspecting adjacency in one kernel alone is insufficient.

Cosine/sine tables are optional implementation choices, not CUDA constant-memory requirements. Two FP16 tables at 131,072 positions and 64 frequencies use **32 MiB combined**; the ordinary CUDA constant-memory space cannot hold them. Some runtimes compute frequencies on demand. Follow actual cache/position semantics, including resumed requests and padding.

## 5. Keep context extension checkpoint-specific

Neither supplied release config ships the YaRN JSON formerly shown in this lesson. The [72B model card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) documents an optional YaRN extension for longer contexts; use that recipe with a runtime that supports it, and evaluate long-context quality. The 4B July checkpoint already declares 262,144 positions and a different RoPE base. Copying an older Qwen3 “40k native” configuration changes this experiment.

A larger base or runtime context setting does not prove reliable retrieval. A single grammatical answer at 50k tokens is not a rotary correctness test. Check numerical positions/cache parity first, then use held-out tasks at several lengths and fact positions. Longer contexts still increase attention work and global KV storage even when a scaling transform is cheap.

## 6. Run an actual small forward pass

The optional CPU script also instantiates a **random two-layer Qwen3** with hidden size 32 and query width 64. It compares full-sequence inference with cached prefix plus one-token decode, checks that changing future tokens leaves earlier logits unchanged, then deliberately uses the wrong decode position.

Observed maximum absolute differences: cached/full **8.94e−8**, changed-future/earlier logits **0**, deliberately wrong position **0.06964**. This is a current library API and cache-semantics check, not an evaluation of pretrained 4B or 72B quality. The [lab guide](Lab-Guide.md) includes installation, commands, raw results, and limits.

**Practice:** explain why changing head dimension affects Q/K/O shapes but not the residual width; derive the KV cache at 4,096 tokens; and explain why a tokenizer's maximum token ID need not equal the padded embedding-table row count. Next: [quantization](Lecture-02.md).
