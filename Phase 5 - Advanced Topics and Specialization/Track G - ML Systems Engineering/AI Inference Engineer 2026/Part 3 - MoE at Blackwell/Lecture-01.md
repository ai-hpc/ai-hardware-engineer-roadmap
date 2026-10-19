# Part 3 · Lecture 01 — MoE capacity, active work, and the attention cache

A mixture-of-experts layer chooses a small subset of its expert MLPs for each token. This can reduce active computation relative to a dense model with similar total parameters. It does not mean only the active parameters need storage, or that a batch reads exactly the active count for one token.

Use the [CPU routing and accounting lab](../Part%201%20-%20Fundamentals/labs/README.md). It includes pinned configuration files for DeepSeek-V3.1 and Qwen3-235B-A22B, with hashes and source URLs. No pretrained weights are needed.

## Read the two configurations

| Field | DeepSeek-V3.1 | Qwen3-235B-A22B |
|---|---:|---:|
| Layers | 61 | 94 |
| Residual width | 7168 | 4096 |
| Routed experts per MoE layer | 256 | 128 |
| Selected routed experts per token | 8 | 8 |
| Shared expert | 1 | none in the cited config |
| Expert intermediate width | 2048 | 1536 |
| Attention cache family | MLA | GQA |

Sources are the pinned [DeepSeek configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/c0781d039fb7a1ba2abc4add0bdc293e92d2b8db/config.json) and [Qwen3 configuration](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json). DeepSeek's first three blocks use dense MLPs; do not count 61 routed-expert layers. Routing details also differ: normalization, scoring, grouping, shared experts, and scaling must come from the implementation.

## Follow one routed token

A router computes expert scores, chooses expert indices, and supplies combination weights. The runtime groups token-expert assignments, dispatches them to owners, executes expert MLPs, restores token ordering, and combines outputs. Shared-expert work may execute on a separate path.

In the CPU exercise, experts are `E0(x)=x+1` and `E1(x)=2x`. For x=2 with weights 0.25 and 0.75, output is `0.25×3+0.75×4=3.75`. Swapping experts without swapping weights or combining in arrival order corrupts the result. The test preserves original token IDs and checks weighted combination.

For a SwiGLU expert with residual width D and intermediate width F, three dense weights contain `3DF` parameters. DeepSeek's routed expert example has `3×7168×2048 = 44,040,192` weights, excluding any separately specified components. Multiplying only by top-k estimates selected expert work for one token, not total model storage.

## Total, per-token active, and batch-union parameters

Total parameters govern resident storage. Per-token active parameters help estimate arithmetic. The union of experts selected by a batch helps estimate which expert weights may be read that step, subject to cache reuse and implementation. If token A selects experts 0–1 and B selects 2–3, top-k is two but the batch touches four experts. The lab tests this distinction.

The [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) describes the original V3 architecture's 671B total/37B activated parameter scale and MLA/MTP design. Use the V3.1 configuration for this checkpoint rather than treating every V3-family release as identical. Qwen's “A22B” label is similarly an active-scale descriptor, not a complete byte count or throughput prediction.

## MLA versus GQA storage

An MLA implementation can retain a compressed KV latent plus a shared positional key component. Under that specific layout, DeepSeek's configuration gives:

```text
bytes/token = 61 layers × (512 latent + 64 RoPE) × 2 bytes
            = 70,272 bytes = 68.625 KiB
```

The calculation assumes BF16 values and that expanded K/V are not additionally stored. Temporary expanded tensors, alternate kernels, scales, sharding, and allocator overhead change actual memory. At 131072 tokens the ideal payload is 9,210,691,584 bytes, about 8.578 GiB.

Qwen3 uses 4 KV heads and explicit head dimension 128 across 94 layers: `2×94×4×128×2 = 192,512 bytes/token = 188 KiB`. Its hidden width divided by query heads is 64, so that shortcut would underestimate its cache by two. These values compare particular model layouts, not “MLA is always this many times smaller.”

## MTP is an optional inference path

Multi-token prediction supplies extra prediction machinery that a compatible runtime can use for speculation. Drafting, verification, rejected work, and state management still cost time. Support and accepted-token rate must be measured. It is incorrect to describe MTP as free or promise a fixed speedup from the training objective alone.

**Deliverable.** A capacity/work table with resident weights, active-token arithmetic assumptions, batch expert union, cache format, and excluded overhead. Next: [Blackwell hardware](Lecture-02.md).
