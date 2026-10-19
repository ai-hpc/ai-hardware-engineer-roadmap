# Memory: budget bytes and track ownership

The CPU course tests memory arithmetic and allocator control. The experimental CUDA path uses contiguous managed storage for FP16 KV and scratch buffers. It does not implement pageable overflow, eviction, INT8 KV scales or measured bandwidth guarantees.

## 1. Shared physical memory still has access rules

Jetson CPU and GPU share DRAM. That does not make every CPU pointer legal for a kernel. Allocation, registration, mapping, coherence and synchronization still matter.

The weight loader maps the file read-only and requires CUDA host registration/device mapping to succeed. It rejects failure instead of passing an ordinary pageable pointer into the kernel. KV and scratch use `cudaMallocManaged`; that call alone does not validate lifetime or establish performance.

Orin's coherence behavior differs from later Tegra SoCs. Consult the [CUDA for Tegra guide](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) before changing allocation APIs.

## 2. Calculate KV capacity

For a simple uncompressed cache:

```text
KV bytes = batch * 2 * layers * KV_heads * head_dimension * context_tokens * bytes_per_element
```

Two accounts for keys and values. Use KV heads for grouped-query attention, and use the declared head dimension instead of assuming `hidden_size/query_heads` always applies.

Worked example: one sequence, 26 layers, 8 KV heads, head dimension 128, FP16 and context 4,096:

```text
bytes per token = 2 * 26 * 8 * 128 * 2 = 106,496
cache bytes = 106,496 * 4,096 = 436,207,616 = 416 MiB
```

This illustrative cache excludes weights, scratch, runtime state, allocator overhead and the OS. INT8 could halve element payload, but needs the correct scales and quality validation; the engine rejects that mode.

Native tests reject negative dimensions and multiplication overflow before allocation. A wrapped size can otherwise allocate a small buffer for a large write.

## 3. Distinguish estimates from reservations

`probe_system_memory()` reads Linux `MemTotal` and `MemAvailable`. Available memory is an estimate, not a reservation: another process can consume it after the check.

The budget uses `total - available` as current unavailability and records CMA separately without subtracting it twice. The 300 MiB CUDA-context and 256 MiB safety allowances are planning estimates, not observed constants for all devices. Fields named `*_mb` here represent binary MiB.

`OOMGuard` rejects invalid requests, rounds bytes upward to MiB, and checks availability against a margin. The later allocation can still fail and must be checked. The runtime does not write global `drop_caches` or compaction controls as an emergency strategy.

## 4. Draw the layout

Each layer has a complete key region followed by a complete value region. Within each region, one token holds `KV_heads * head_dimension` contiguous elements. Capacity and currently valid tokens are different quantities.

For capacity 5, two KV heads and dimension 3, token 2 starts at element `2*2*3 = 12` in the key region. Head 1 within that token starts at element 15. Values begin after `5*2*3 = 30` key elements.

An attention kernel must consume the same token-major layout. A head-major interpretation or a stride based on the wrong length silently reads other values. Reset changes validity; it does not make old bytes meaningful.

Host-shim tests validate pointer arithmetic and lifecycle. Cache-to-attention correctness additionally needs independent numerical references and GPU execution across multiple heads, tokens, capacities and repeated calls.

## 5. Budget the maximum live scratch, then order its reuse

The bump allocator rounds requests to 256-byte boundaries. Requests of 1 and 257 bytes consume 256 and 512 bytes: **768 total**.

`reset()` and `rewind()` change an offset; neither waits for a stream. A later
write must execute after every old reader. This can be established by queuing
all consumers and subsequent overwrites on the **same CUDA stream**, or by
waiting for completion before a host write or another stream uses the storage.
Allocator bookkeeping alone is not synchronization.

The experimental decode loop retains its token state before a checkpoint and
rewinds layer temporaries after queuing the layer. Copies and kernels use the
same stream, so the next layer's writes follow the previous layer's reads.
The pool stays allocated. Its peak is one token state plus the larger of one
layer's temporary storage and final-normalization/logit storage, rather than the
sum of every layer. This contract would need revision for concurrent streams.

For the test geometry `hidden=256`, four query heads, two KV heads, head dimension
64, FFN dimension 512, and vocabulary 768, rounded allocations give **7,680 bytes**
for a layer, **5,632 bytes** for the final stage, and **8,192 bytes** for device
scratch including the persistent 512-byte token state. The planner additionally
reserves 3,072 bytes of persistent host logits and 49,152 bytes of fixed embedding
staging. It rounds the combined reservation upward to MiB **before** sizing KV.
This still excludes unknown process/driver overhead covered by measured headroom.

Every decode scratch request now uses `get_or_throw()`; exhaustion raises an
allocation failure before a null pointer reaches a kernel. The CPU contract
executes 200 layer allocation/rewind cycles, checks that token state survives,
and rejects overflow, invalid checkpoints, and an undersized pool. CUDA stream
ordering and device execution remain target validation, not results of the shim.

Owning pool classes cannot be copied and release allocations on destruction. Reinitialization releases the previous allocation. Native tests cover failed allocation, invalid requests, capacity, overflow and cleanup with a private host-only CUDA shim. Those tests do not validate GPU residency or coherence.

**Finish:** derive 416 MiB, draw the two-head layout, and explain why a successful availability check cannot replace checking allocation success. Run the [CPU lab](../labs/README.md) and inspect its sanitizer result.

Continue with the [attention layout lab](../labs/Attention-Layout-Lab.md), whose
native checks include the scratch plan and the actual pool control logic.
