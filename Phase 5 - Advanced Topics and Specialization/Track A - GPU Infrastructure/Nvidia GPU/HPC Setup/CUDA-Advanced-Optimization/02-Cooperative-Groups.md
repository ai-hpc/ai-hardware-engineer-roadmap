# 02 — Cooperative Groups: define who must participate

**Outcome:** explain the participants, data visibility and launch requirements of a synchronization point. Use the reduction in the [shared lab](labs/README.md), then reason about a grid-wide algorithm.

## 1. Synchronization has a scope

A barrier coordinates a specified group of threads. It is not simply “wait until the GPU is ready”. Every required participant must reach a valid matching collective, and the data being exchanged must obey the memory rules of that operation.

| Group or mechanism | Participants | Typical purpose |
|---|---|---|
| Thread-block group | Threads in one block | Shared-memory exchange |
| Fixed tile | A partition of a parent group | Subgroup reduction or shuffle |
| Coalesced group | Threads coalesced at that point | Work among the current active group |
| Grid group | Cooperative launch's grid | A multi-phase algorithm across blocks |
| Same-stream kernel order | Operations submitted in that stream | Cross-kernel dependencies |

Do not rely on implicit warp lockstep for communication. Independent thread scheduling makes explicit synchronization and valid collective participation essential. A coalesced group gives ranks to currently participating threads; it does not physically pack them into new hardware lanes or guarantee that all logically eligible threads appear in one group.

Source: [Cooperative Groups programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html).

## 2. A complete block reduction

The lab's `block_partial_sum` uses exactly 256 threads per block and 256 shared floats. Each thread writes one value; threads beyond the input tail write zero. All threads reach every block barrier.

```text
load or zero -> block barrier
128 pair additions -> block barrier
64 pair additions -> block barrier
...
1 pair addition -> block barrier
thread 0 writes the block's sum
```

The power-of-two block size is a stated precondition of this reduction tree. The mathematical sum is the reference; floating-point reassociation can change rounding. The lab uses small dyadic values so this fixture's sums remain exactly representable.

**Worked tail:** input length 257 creates two blocks. Block 0 reduces elements 0–255. Block 1 has one valid element and 255 zeros. No thread exits before the barriers. The host compares every partial sum with an independent scalar calculation.

**Fault to avoid:** `if (global >= n) return;` before a later block-wide barrier leaves a partial block with missing participants. Predicating loads/stores while preserving collective participation is the correct structure for this example.

## 3. A shuffle is not a block broadcast

A warp/tile shuffle exchanges registers inside its supported group. `__shfl_sync(..., value, 0)` cannot broadcast thread 0's register across every warp in a 256-thread block. For a block-wide control value, use shared storage and a block barrier, as the finite worker in the lab does.

Fixed 32-lane butterfly reductions need the stated participation and initialized neutral values. A reduction that starts at `group.size()/2` and repeatedly halves is not a general reduction for arbitrary group sizes.

**Counterexample:** for three values `[1,2,4]`, the naive halving loop starts at offset 1 and then stops. The first rank can obtain `1+2=3`, missing the third value. Use a supported library collective for the relevant group/type or a reduction proven for arbitrary size. See [Cooperative Groups collectives](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#group-collectives).

Shuffle latency is architecture- and dependency-dependent. There is no universal one-cycle promise or guaranteed superiority over every shared-memory reduction.

## 4. Grid synchronization needs a cooperative launch

A grid barrier can coordinate blocks inside one kernel when the device and launch support it. Query cooperative-launch capability and compute occupancy for the **actual kernel, block size and dynamic shared memory**. Respect the allowed cooperative grid size and ensure all participants follow the matching collective sequence.

Block indices are logical identities, not SM IDs. A grid with one block per SM count does not assign block `i` to SM `i`. Register/shared-memory use and the launch mechanism determine residency constraints.

A normal kernel that spins until every block arrives can deadlock if waiting blocks occupy resources required by blocks that have not yet run. Do not replace a supported grid barrier with a global counter plus a busy-wait and assume it is equivalent.

The lab intentionally uses block synchronization only. Cooperative grid launch and performance need a separate GPU experiment; no grid-wide result is claimed here.

## 5. A second kernel does not require a host round trip

These operations in one stream establish order:

```cpp
// Schematic: kernels and buffers are supplied by your program.
phase_one<<<grid, block, 0, stream>>>(input, partial);
phase_two<<<grid2, block2, 0, stream>>>(partial, output);
```

The second kernel follows the first in that stream. A `cudaDeviceSynchronize()` between them is unnecessary merely to make phase two consume phase one's results. Use explicit dependencies for other streams and a completion mechanism when the host must inspect output. Graph capture can amortize submission of this sequence.

A grid-synchronized single kernel may help some workloads, but can also restrict occupancy or scheduling. Compare it with a correct multikernel baseline.

## 6. Reduction and scan are different algorithms

A reduction returns one combined value; an inclusive scan returns every prefix. For `[1,2,3,4]`:

```text
sum:            10
inclusive scan: 1,3,6,10
exclusive scan: 0,1,3,6
```

An up-sweep reduction tree alone leaves intermediate tree values, not all prefixes. A complete scan needs the remaining propagation steps or another correct algorithm. Adding a block prefix to an unfinished local reduction does not turn it into a scan.

For a multi-block scan, explain three obligations: each local scan, a scan of block totals, and addition of preceding-block totals. Test non-power-of-two sizes and partial blocks. Use [CCCL/CUB](https://nvidia.github.io/cccl/cub/) as a production baseline before writing a custom scan.

## 7. Integrating with PyTorch

A CUDA extension must check device, dtype, shape/layout and size assumptions; guard the selected device; launch on PyTorch's current stream; check launch errors; and define empty-input behavior. A bare default-stream launch with an unchecked `data_ptr<float>()` is not a general extension.

If a reduction uses atomics into an output scalar, initialize it on a correctly ordered stream. Floating-point atomic order may vary between executions. Choose tolerances and determinism requirements accordingly. Review [PyTorch custom C++/CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).

## Exercise and answers

1. **Where must the final partial block's inactive threads go?** Through the same barriers, contributing zeros and skipping invalid stores.
2. **Can a warp shuffle publish a work index to all 256 threads?** No; its scope does not include the other warps.
3. **Is kernel order in one stream sufficient for a two-stage dependency?** Yes for those stream operations; host inspection and other streams need their own ordering.
4. **Does an up-sweep tree produce a scan?** No. Check `[1,2,3,4]` to expose the missing prefixes.

Submit a participant diagram, known-answer/tail cases, and either GPU results or an explicit compile-only record. Continue with [finite persistent work](03-Persistent-Kernels.md).
