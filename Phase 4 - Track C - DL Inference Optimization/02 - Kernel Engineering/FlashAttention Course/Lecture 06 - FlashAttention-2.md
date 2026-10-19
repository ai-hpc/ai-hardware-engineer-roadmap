# Lecture 6 — What FA2 changes—and what it still must compute

**Course:** [FlashAttention](Guide.md) · **Reviewed:** 18 October 2026

**Goal:** explain work partitioning and fewer non-matmul operations without changing the attention contract. **Artifact:** a source-reading note plus a counterexample to an invalid recurrence.

## 1. Distinguish mathematical work from scheduling

FA2 keeps dense attention and its running normalization. Its paper describes fewer non-matmul operations, additional parallelism across sequence tiles, and a different division of work among warps. These changes improve utilization under the paper's evaluated conditions; they do not imply constant runtime as sequence length grows. [FA2 paper](https://tridao.me/publications/flash2/flash2.pdf).

A batch of one with a small number of heads can offer too few independent blocks. Splitting the query sequence into independent output tiles provides more blocks. Each output tile still visits the key/value range permitted by its mask. The tradeoff includes data reuse and live storage, not only block count.

## 2. Correct the sliced-K versus sliced-Q explanation

In the paper's comparison, FA1's sliced-K scheme partitions K/V work among warps, creating partial contributions that need combining. FA2 instead partitions Q/output rows among warps while sharing the relevant K/V block. The row owners can produce their own output slices with less inter-warp reduction traffic.

Reversing those descriptions reverses the reason for the optimization. Do not claim that FA2 generally splits different KV columns to avoid the very reduction those partial columns create.

**Worked ownership example:** four query rows and four warps. If each warp owns one query row and sees every allowed key, its final output row is complete. If each warp sees only one quarter of the keys for every row, it has partial maxima, denominators and numerators. Those partial results must be merged with their relative normalization scales; adding independently normalized outputs is wrong.

## 3. Defer normalization, not every scale correction

The unnormalized numerator in Lecture 02 allows division by `ell` to occur once at the end. When the running maximum changes, existing numerator/denominator contributions still need compatible scaling. FA2 does not justify dropping all those factors and adding one arbitrary final multiplier.

The CPU test uses scores `[-1000, 0, 1000]` with values `[1,2,3]` and one-element blocks. Correct stable attention selects approximately 3. An accumulator that gives every locally normalized block equal weight gives 2. This failure has nothing to do with a GPU tile size; it violates the math.

Later conditional-rescaling algorithms can choose a different maintained scale, but must preserve their stated invariant and numerical range. Do not retrofit a new algorithm's idea onto FA2 without deriving it.

## 4. Read and measure a selected path

Use the pinned launch and softmax source from Lecture 05. Identify the grid dimensions, which warp owns each output row, and where the output is rescaled and normalized. A symbol name alone is not enough: show its input/output layout.

For a GPU comparison, use a supported dtype/shape and record the actual selected backend. PyTorch SDPA dispatch depends on the installed build, device, mask, dtype, strides and other features. Its enumeration includes cuDNN as well as math, Flash and efficient backends; do not reduce current dispatch to an old universal three-condition checklist. [Backend selection](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html).

**Answer check:** doubling dense sequence length roughly quadruples useful matrix work at fixed head dimension. Better scheduling may improve achieved throughput, but does not make total work disappear. A NumPy or Triton teaching implementation is not an “FA1 benchmark” unless it actually implements and identifies that historical version.
