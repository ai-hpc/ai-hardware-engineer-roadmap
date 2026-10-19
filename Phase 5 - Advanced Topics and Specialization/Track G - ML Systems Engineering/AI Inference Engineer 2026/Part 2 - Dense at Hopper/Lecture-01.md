# Part 2 · Lecture 01 — Derive a dense model's shapes before loading it

Use Llama 3.3 70B and Qwen2.5 72B as historical, concrete model families—not a claim that they are the newest models. Their size makes memory accounting visible. You can complete this lesson with configuration files and the [CPU contract lab](../Part%201%20-%20Fundamentals/labs/README.md), without downloading weights.

## Start from a pinned configuration

The lab includes Qwen2.5's configuration at revision `495f39366efef23836d0cfae4fbe635880d2be31` and records its hash. The official [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/495f39366efef23836d0cfae4fbe635880d2be31/config.json) has 80 layers, width 8192, 64 query heads, 8 KV heads, intermediate width 29568, and vocabulary 152064. Its saved `max_position_embeddings` is 32768. The [model card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) describes the separate YaRN configuration for longer contexts; changing a length limit alone does not implement that recipe.

For Llama, use the [official model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and obtain the gated configuration through your own authorized access. The published 70B architecture commonly used in this comparison has 80 layers, width 8192, 64 query heads, 8 KV heads, intermediate width 28672, and vocabulary 128256. The course did not download that gated config; verify the exact revision before building a runtime contract.

Record attention head dimension explicitly when present. For these dense examples it is 128; that inference is not valid for every model. Also record bias terms, tied embeddings, normalization epsilon, RoPE parameters, and context policy. A parameter-count label is rounded and does not replace these fields.

## Work through a block

Flatten the batch and token dimensions into `M` token rows. With residual width `D=8192`, query width `64×128=8192`, and K/V width `8×128=1024`:

| Operation | Mathematical weight shape | Output shape |
|---|---|---|
| Query projection | `[8192,8192]` | `[M,8192]` |
| Key projection | `[8192,1024]` | `[M,1024]` |
| Value projection | `[8192,1024]` | `[M,1024]` |
| Attention output projection | `[8192,8192]` | `[M,8192]` |
| SwiGLU gate and up | two `[8192,F]` matrices | two `[M,F]` tensors |
| SwiGLU down | `[F,8192]` | `[M,8192]` |

Framework files may store a linear weight transposed as `[out,in]`. State which convention you use. SwiGLU computes `down(silu(gate(x)) * up(x))`; its three dense weights contain `3DF` parameters, excluding biases. Do not treat the gate as an expert router: this is still a dense MLP.

The attention projections contain `2D(Hq*d)+2D(Hkv*d)` parameters when input/output widths are D. Insert the actual dimensions rather than memorizing “4D²,” which assumes ordinary multi-head attention without reduced KV heads.

## RMSNorm, precisely

For a vector of width D:

```text
rms = sqrt(sum(x_i*x_i)/D + epsilon)
y_i = gamma_i * x_i / rms
```

RMSNorm does not subtract the mean. Before learned scaling and ignoring epsilon, the vector's **root mean square** becomes one; its Euclidean norm becomes `sqrt(D)`, not one. Epsilon prevents division by zero and affects small magnitudes. Accumulation precision and fused-kernel reduction order influence numerical error. The [RMSNorm paper](https://arxiv.org/abs/1910.07467) motivates the operation; it does not establish a fixed speedup for every inference stack.

## Weight and cache accounting

Sum projection/MLP weights per layer, then embeddings, vocabulary output projection, normalization parameters, and biases. Untied input/output embeddings contribute two separate `vocab×D` matrices. Qwen's larger vocabulary and MLP width add parameters; tokenizer efficiency on a language must be measured on representative text, not assumed from vocabulary size.

Both dense examples have the same ideal BF16 GQA cache geometry: 320 KiB per token across all layers. At 131072 tokens, that is 40 GiB per request before allocation overhead. Quantized weights do not change that number unless the cache format also changes. Tensor parallelism can shard KV heads, but replication, uneven divisibility, and the runtime layout determine actual per-rank capacity.

**Exercise.** Qwen's MLP width exceeds 28672 by 896. The extra MLP weights across 80 layers are `80×3×8192×896 = 1,761,607,680` parameters. Derive this before running a calculator. Vocabulary and bias differences are additional terms.

**Deliverable.** A table with config revision, exact dimensions, weight-storage assumptions, KV bytes/token, and the first allocation that would exceed your budget. Next: [Hopper hardware](Lecture-02.md).
