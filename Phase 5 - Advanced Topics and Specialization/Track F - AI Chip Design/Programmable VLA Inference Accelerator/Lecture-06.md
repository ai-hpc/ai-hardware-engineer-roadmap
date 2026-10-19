# Lecture 06 — Compiler, Runtime ABI, and SoC Integration

**Question:** how does a policy graph become executable accelerator commands inside a Linux system without losing its tensor, timing, or action semantics?

**Prerequisites:** [Lecture 05](Lecture-05.md), compiler intermediate representations, virtual memory, DMA, and producer/consumer queues. Use the [LLVM/MLIR lectures](../Lectures/Lecture-01.md) and [embedded Linux course](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Guide.md) for the underlying concepts.

**By the end:** specify a lowering path and command ABI, justify one legal fusion, trace buffer ownership, and reject invalid commands and stale action chunks in a simulator.

**Deliverable:** a compiler/runtime interface document, a small executable command-stream interpreter, and a negative-test report. This is a prototype interface, not a production Linux driver.

## 1. The compiler needs three kinds of contract

| Contract | Examples |
|---|---|
| Mathematical | Operator definition, axes, masking, epsilon, quantization, rounding, and tolerated error |
| Physical | Shape, stride, layout, alignment, packing, banks, buffer lifetime, and DMA range |
| Robotic | Observation identity, clock domain, units, coordinate frame, action horizon, and validity window |

The tensor compiler primarily transforms the first two. The system runtime must preserve the third across submission and completion. A correctly computed action tensor associated with the wrong observation is still an incorrect system result.

Keep a manifest linking model and preprocessing revisions to compiled artifacts, quantization metadata, supported shapes, ABI version, and action adapter. A checksum can detect accidental mismatch; it is not by itself an authorization or authenticity mechanism.

## 2. Lower progressively and keep the evidence

```text
reference policy + frozen observation/action contract
    -> captured graph with explicit shapes and numerical semantics
    -> supported accelerator partitions + visible fallback partitions
    -> tiled tensor/vector operations with selected layouts
    -> allocated buffers, dependency events, and memory transfers
    -> versioned command stream + constants + runtime manifest
    -> functional interpreter, then RTL/FPGA execution
```

For each transformation, retain a small example and compare against the previous representation. Do not wait until an entire policy has been lowered before checking whether transpose, masking, or scaling semantics survived.

MLIR's Linalg representation supports transformations such as tiling, fusion, and lowering toward loops or target operations. Bufferization addresses the transition from tensor values to mutable memory while accounting for conflicts. Neither framework automatically supplies your chip's ABI or proves the robot's timing contract. [Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/), [Bufferization](https://mlir.llvm.org/docs/Bufferization/)

The report must list every fallback with transferred bytes and synchronization points. Supporting 95% of operators by count can be unhelpful if the remaining 5% contains the slowest repeated block.

## 3. Worked fusion: saved traffic versus changed semantics

Consider a reference path:

\[
Z=\operatorname{cast}_{16}(XW),\qquad
Y=\operatorname{cast}_{16}(\operatorname{GELU}(Z+b)+R),
\]

where \(XW\) uses a declared wider accumulator and \(Z\) has shape \(16\times256\). Assume the unfused path writes \(Z\) to external memory and reads it back for the vector stage.

The intermediate occupies \(16\times256\times2=8{,}192\) bytes. Keeping it on-chip can avoid one write and one read:

\[
D_{\mathrm{avoided}}=2\times8{,}192=16{,}384\ \mathrm{bytes}.
\]

This is the maximum avoided traffic for that intermediate under the stated baseline. If the baseline already retains it in cache or SRAM, measured DRAM savings can be smaller.

A legal implementation must preserve broadcasting, layout, aliasing, and the chosen numerical contract. Keeping the wider accumulator directly through GELU removes the intermediate cast and changes rounding. That may be an acceptable separately validated numerical transformation, but it is not automatically bitwise equivalent fusion.

A reduction adds another constraint. A normalization requiring the whole feature axis cannot generally be evaluated independently on partial feature tiles without combining the required statistics. Fusion is justified by data dependencies and semantics, not by adjacency in a graph drawing.

## 4. Define the command ABI before the driver

The application binary interface (ABI) specifies how independently built compiler output, runtime software, firmware, and hardware agree. Begin with a small versioned command set such as `COPY`, `MATMUL`, `VECTOR`, `WAIT`, and `SIGNAL`.

These are **teaching opcodes**, not an existing standard. Their specification needs enough detail to make a command interpreter reject malformed input deterministically.

| Field or rule | Required meaning |
|---|---|
| ABI version and command length | Reject unknown versions and truncated records |
| Opcode and flags | Define permitted combinations and reserved values |
| Buffer handle, offset, and length | Resolve through owned, validated allocations |
| Shape, stride, and layout ID | Bound every accessed element; state supported dynamic ranges |
| Quantization descriptor | Match scales, grouping, rounding, and payload encoding |
| Accumulator and output mode | Avoid an implicit precision change |
| Dependency and completion IDs | Prevent consumption before production; distinguish completion from acceptance |
| Request and observation IDs | Associate outputs with their source inputs |
| Deadline and clock domain | Define exactly how expiration is interpreted |
| Error and reset generation | Prevent old completions from being mistaken for new work |

Specify byte order, fixed field widths, alignment, reserved fields, extension rules, and overflow-safe range validation. For example, validate `offset <= allocation_size` and `length <= allocation_size - offset` using a suitable unsigned type, instead of trusting an overflowing `offset + length` expression.

Linux's ioctl guidance is useful when designing a stable userspace interface. A prototype should use explicit-width fields and compatible layouts rather than persisting a compiler-dependent C structure as an undocumented binary format. [Linux ioctl interfaces](https://docs.kernel.org/driver-api/ioctl.html)

## 5. Track DMA ownership through completion

For an input buffer, the conceptual lifecycle is:

```text
CPU writes -> map/synchronize for device -> publish descriptor
    -> device reads -> completion -> synchronize/unmap as required
    -> CPU may reuse or release storage
```

For an output buffer, reverse the producer and consumer roles. Multiple devices may add additional ownership transitions.

A device DMA address is not necessarily a CPU virtual address or a raw physical address. Use the platform's DMA mapping API and account for mapping failure and addressability. Coherent allocation addresses visibility requirements but does not eliminate ordering requirements around publishing descriptors and observing completion. [Linux DMA mapping guide](https://docs.kernel.org/core-api/dma-api-howto.html)

Define which agent owns each buffer at every transition. A cancellation request does not prove that DMA has stopped. The implementation must quiesce, complete, or reset relevant work before recycling storage that the device may still access.

The functional interpreter should model ownership even though it does not perform real DMA. That allows a test to reject a read of an unpublished input or reuse of an in-flight output before hardware exists.

## 6. Bounded queues and observable errors

Use a bounded submission queue and a separate completion record. Define full-queue behavior: block, reject, or replace only work that can be canceled according to the dependency and ownership rules.

Sequence numbers distinguish a wrapped queue slot from its earlier occupant. Publication and consumption need synchronization appropriate to the CPU/device interface. A single-producer/single-consumer CPU ring is not automatically a correct multi-producer or DMA queue. [Linux circular-buffer documentation](https://docs.kernel.org/core-api/circular-buffers.html)

Require the following negative outcomes to be distinguishable:

- Unsupported shape or numeric format: reject before launching dependent work.
- Invalid range or missing dependency: reject the command stream.
- Device fault or timeout: report an unsuccessful completion and follow the defined recovery path.
- Expired policy output: discard it at the action boundary even if arithmetic completed correctly.

Do not fabricate success by substituting an old output when a command fails. If the robot's supervisor uses a fallback, it must be a separately defined behavior with an observable reason.

## 7. Connect accelerator results to the SoC

The Linux CPU handles model loading, memory management, sensor services, runtime submission, and application integration. It can also run fallback operators. A local controller or separately scheduled control subsystem consumes validated action proposals at its own rate.

Each action chunk should identify the source observation and policy revision, carry the coordinate frame and units, and state the intended action times or horizon. Check freshness when publishing the chunk and again when consuming actions from it. A chunk can expire after admission.

The course's timestamp and queue mechanisms help build a testable interface. They do not establish a functional-safety certification or a hard real-time bound. Those require the corresponding system analysis and validation.

Wireless carries telemetry, updates, or appropriately bounded higher-level requests in the initial design. Put it outside the time-critical motor/flight-control dependency chain. Use the [ESP-Hosted integration course](../../../Phase%202%20-%20Embedded%20Systems/3.%20Embedded%20Linux/Jetson%20ESP-Hosted%20Host%20Code/Guide.md) to study host/companion integration, and specify behavior under disconnect or delayed messages.

State whether the prototype uses a module, companion radio, or integrated RF design. A Linux-capable CPU plus an accelerator block does not automatically provide boot firmware, memory-controller IP, sensor interfaces, or wireless capability.

## 8. Exercise: run one block through the entire contract

Lower the worked matrix/vector block into your teaching command stream. Use host memory and a functional interpreter first; hardware access is optional for this stage.

1. Emit the reference tensor graph, selected layouts, buffer plan, and commands.
2. Check unfused and fused outputs under the same rounding contract.
3. Track buffer ownership and dependency completion in an execution log.
4. Attach observation and request IDs to the resulting action-like output.
5. Implement a bounded consumer that rejects an expired output.

**Negative cases:** truncate a command; use an unsupported shape; overflow a proposed range; omit a dependency; reuse a buffer before completion; and deliver a completion from an old reset generation. Each must produce an explicit rejected or failed result.

**Acceptance:** submit the ABI document, interpreter, one reference comparison, and a table of rejected cases. Report what remains untested on real hardware: DMA visibility, interrupts, reset behavior, clock synchronization, and sustained contention.

## 9. Answered self-checks

**1. Is eliminating an intermediate cast always legal fusion?** No. It changes rounding and can change the result. Preserve the cast for equivalence or evaluate the change as a separate numerical optimization.

**2. Does coherent memory make queue ordering unnecessary?** No. Visibility, ownership, and publication order are distinct requirements. Follow the platform's synchronization and DMA rules.

**3. Does a completed accelerator command imply that its action is usable?** No. The output can be stale, belong to an old request or reset generation, use the wrong frame, or violate the action contract.

## Sources and next step

- [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) and [Bufferization](https://mlir.llvm.org/docs/Bufferization/): inspect transformation and memory semantics.
- [Linux DMA API](https://docs.kernel.org/core-api/dma-api-howto.html): establish address mapping, visibility, and ownership transitions.
- [Linux ioctl interfaces](https://docs.kernel.org/driver-api/ioctl.html) and [circular buffers](https://docs.kernel.org/core-api/circular-buffers.html): design explicit, versioned interfaces and correct queue synchronization.

Continue to [Lecture 07 — Scheduling and Freshness](Lecture-07.md). The runtime interface now carries the information needed to analyze interference, observation age, and deadline misses.

## Inspect an actual lowering boundary

Use the [real MLIR CPU fixture](../Labs/Tiling-and-Dataflow.md#4-execute-linalg-through-llvm-on-the-cpu) to observe Linalg become loops, LLVM dialect, and native CPU execution. Its zero initialization, full reduction, output checks, and deallocation are visible. This establishes a working CPU path only. Your accelerator extension must separately implement command encoding, buffer ownership, completion, and the target arithmetic contract; naming a custom dialect operation does not supply those behaviors.
