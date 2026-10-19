# Lecture 04 — Recurrent state, token caches and hybrid models

[Previous: compilers and runtimes](Lecture-03.md) · [Course index](README.md) · [Next: inspect a model](Lecture-05.md)

An autoregressive model must carry information from earlier inputs into the next
prediction. Different architectures store that information differently. This
lesson compares a list of historical token representations with a fixed-size
recurrent summary, then shows why a hybrid can need both.

**Prerequisites:** vectors, matrix multiplication and the difference between
parameters and per-request state. Allow one hour for the derivations and the
first session of the linked CPU lab.

## A growing list and a fixed summary

A conventional attention cache appends keys and values. Its logical byte count
for one uniform GQA stack is:

```text
2 × attention_layers × cached_tokens × KV_heads × head_dim × bytes_per_element
```

The factor two counts K and V. Multiply by concurrent requests only when those
requests have the same length; otherwise sum their individual token counts.
For 4 attention layers, 2 KV heads, dimension 8 and two-byte elements, each token
adds `2×4×2×8×2=256` bytes. At 1,024 tokens that is 256 KiB per request before
padding, allocator overhead and any replication across devices.

A recurrence instead updates a fixed-size state. For the scalar teaching rule
`s_t=0.5*s_(t−1)+x_t`, inputs `[2,0,0]` produce states `[2,1,0.5]` from zero.
The state size stays one scalar, but the individual past inputs are no longer
stored separately. Constant state size is a representation property, not proof
that it preserves every retrieval capability of an attention cache.

## What state-space and linear-attention designs change

State-space models use structured state updates; selective variants allow
input-dependent parameters. Linear-attention and delta-rule designs also carry
recurrent summaries, with different update equations and inductive biases.
They are related system-design choices, not interchangeable implementations.

For fixed model dimensions, a recurrent decode step can have work and persistent
state independent of context length. Processing a prompt still requires work
for every input token. Training or parallel prefill can retain intermediate
activations, so fixed decode state does not imply constant total training memory.
The [Mamba-2/SSD paper](https://arxiv.org/abs/2405.21060) develops a structured
connection between state-space and attention computations and algorithms suited
to matrix hardware. That does not make every recurrent algorithm automatically
a fast Tensor Core workload.

The [GLM CPU hybrid-state lab](../GLM-5.3-Flash%20Architecture%20Mastery/CPU-Hybrid-State-Lab.md)
provides an actual example: its scalar KDA reference is checked against pinned
upstream PyTorch recurrent and chunked functions. It compares final state as
well as output. Dense affine composition demonstrates the algebra, while the
upstream chunked implementation uses a different computation to exploit blocks.

## A hybrid retains the costs of its attention layers

Suppose a model has 48 recurrent layers and 16 conventional attention layers.
The recurrent state is fixed in context length, but those 16 attention caches
still grow with context. Relative to an otherwise identical 64-attention-layer
stack, that particular cache term would be one quarter as large. Total memory
would not be one quarter: weights, recurrent state, workspace and other tensors
remain. This is an illustrative architecture, not a specification for every
model called “hybrid.”

MLA compresses each token's stored representation; it does not by itself remove
token-indexed growth. Sparse attention reduces which positions are read for the
main attention computation. An indexer may still scan or score a growing set of
candidates. A fixed number of final selected positions therefore does not prove
constant total decode time.

## State has a lifecycle

State belongs to a request and an accepted prefix. Branching, prefix reuse,
request migration and speculative rejection must preserve that association.
A recurrent matrix cannot be rolled back by decreasing a list length. Restore
a snapshot or recompute from a valid checkpoint. A short causal convolution and
an attention cache may require different restoration operations.

**Predict:** two states yield the same output for one query. Does that prove
they are equivalent for future queries?

<details markdown="1">
<summary>Answer</summary>

No. Query `[1,0]` cannot distinguish states `[[1],[2]]` and `[[1],[99]]`; query
`[0,1]` can. The linked lab makes this counterexample executable and checks
continuation after restoring a rejected branch.

</details>

## Practice and exit check

Run the first and third sessions of the hybrid-state lab. Explain which state
is fixed-size, which grows with history, and which fields must be copied for a
branch. Add one nonuniform decay case and one rectangular state shape.

Then draw a memory curve for your chosen model: identify each fixed term and
each context-dependent term separately. If the curve is a calculation, label
it as such. To claim measured memory, collect actual allocations on a named
runtime and device. Reviewed October 18, 2026; CPU component validation is linked,
while full hybrid-model quality and GPU performance remain separate experiments.
