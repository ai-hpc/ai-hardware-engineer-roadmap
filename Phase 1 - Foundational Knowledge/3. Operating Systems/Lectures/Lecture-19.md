# Lecture 19: io_uring, DMA-BUF, and Zero-Copy Pipelines

Three costs are often confused: **entering the kernel**, **copying payload bytes**, and **waiting for ownership**. Batching can reduce the first, shared buffers can reduce the second, and a correct completion protocol governs the third. Neither io_uring nor “zero-copy” removes all three costs.

Prerequisites: [system calls](Lecture-04.md), [DMA](Lecture-15.md), and [V4L2 ownership](Lecture-18.md). Finish this lesson by drawing the lifetime of one frame or storage request.

## io_uring: submit work and match completions

An io_uring instance exposes a submission queue (SQ) and a completion queue (CQ). Applications prepare submission queue entries (SQEs); the kernel reports completion queue entries (CQEs). Shared ring mappings avoid copying each descriptor through a separate syscall interface. Default operation still uses kernel entry for submission/waiting, but one entry can submit a batch.

An SQE's `user_data` lets you match the completion to the request. Never infer request identity solely from completion order. Independent operations can complete in a different order from submission. Links, drains, and ordering flags have their own documented semantics and failure behavior.

```text
FREE_REQUEST → PREPARED → SUBMITTED → COMPLETION_OBSERVED → REUSABLE
                     buffer remains valid during outstanding access
```

A normal read CQE contains a byte count or a negative error value. A short read is not automatically a complete application record. Check `cqe->res`, process the bytes actually read, and only then decide whether another operation is required. `io_uring_cqe_seen()` advances the completion queue; it does not make an unrelated buffer safe. For multishot operations, `IORING_CQE_F_MORE` indicates further completions may follow. Cancellation is also asynchronous: distinguish the cancellation request's completion from the target request's terminal completion.

**Reading pseudocode, deliberately not a complete liburing application:**

```text
initialize ring; reject unsupported setup or unavailable resources
while requests remain:
    obtain available SQEs; keep every referenced buffer alive
    prepare operations with unique request identities
    submit; handle failure/partial submission according to liburing contract
    inspect CQEs, including negative results and short I/O
    release only resources whose terminal completion permits reuse
    advance CQ entries after reading them
cancel or drain outstanding work before destroying the ring and buffers
```

Use the [liburing manual pages and examples](https://github.com/axboe/liburing/tree/master/man) for the selected release. Probe support on the target kernel rather than assuming every opcode exists because io_uring was introduced in Linux 5.1. Containers or administrators may restrict it.

## Polling moves cost; it does not erase it

| Mode | Mechanism | Cost or constraint |
|---|---|---|
| Default submission | Kernel entry can submit a batch | Measure batch size, CPU time, and tail latency |
| SQPOLL | Kernel thread watches the submission ring | Consumes CPU; an idle thread may require a wakeup syscall |
| IOPOLL | Polls supported I/O completions | Requires supported operations/device/filesystem; commonly direct I/O |
| Registered buffers | Reuse registered memory descriptions | Registration and pinned-memory lifetime/resource limits remain |

SQPOLL is not “zero CPU,” and IOPOLL is not a universal faster mode for sockets or buffered files. Asynchronous buffered I/O may use worker threads or wait on filesystem work. `O_DIRECT` has alignment and filesystem constraints; it does not bypass all kernel work or guarantee persistence.

A useful benchmark holds request sizes, queue depth, file state, device, CPU affinity, and workload constant. Report achieved bandwidth/IOPS, CPU seconds, latency percentiles, errors, and whether the data were cached. Do not compare invented IOPS numbers across APIs.

## DMA-BUF: shared allocation plus an access protocol

An exporter owns backing storage and exposes a file descriptor. Each importing driver attaches and maps the buffer for its own device. Its DMA addresses may differ from another device's addresses. The fd is neither a CPU pointer nor a physical address nor a CUDA pointer.

When sending an fd between processes, use descriptor transfer such as `SCM_RIGHTS` on a Unix socket. Sending the integer `12` only sends a number; fd 12 in the receiving process may refer to something else. Keep exporter/importer references and mappings alive as required by their APIs.

The [DMA-BUF documentation](https://docs.kernel.org/driver-api/dma-buf.html) separates allocation sharing from synchronization. CPU access may require `DMA_BUF_IOCTL_SYNC` around the access region for cache coherency; this does not replace waiting for device completion or mutual exclusion. Explicit fences, implicit reservation-object synchronization, and device-specific contracts must be matched deliberately.

## Camera-to-GPU: two possible allocation directions

**Camera allocation:** V4L2 MMAP buffers may be exportable with `VIDIOC_EXPBUF`; a compatible GPU path imports them.

**External allocation:** an allocator produces a buffer supported by both subsystems; V4L2 uses `V4L2_MEMORY_DMABUF` to import it, and the GPU uses its supported interop path.

Neither direction is guaranteed for every driver or pixel format. On Jetson, supported NvBufSurface/EGL/CUDA or other documented paths depend on the release and surface type. There is no general public operation “`cudaMalloc` → nvmap → `dma_buf_fd`” that turns arbitrary CUDA storage into a camera-compatible DMA-BUF. Managed memory is not automatically exportable either.

```text
allocate compatible pool and establish imports
camera owns a queued buffer → frame completion → application owns buffer
GPU waits for valid frame data → GPU reads/transforms → GPU completion
return buffer to camera only after its last consumer finishes
```

Track width, height, plane count, pitch, plane offsets, pixel format, and color interpretation. A zero-copy NV12 import can still require a GPU conversion into the model's RGB tensor. A common allocation does not imply a common tensor layout.

## Worked bandwidth and lifetime model

A tightly packed 1920×1080 RGBA frame contains 8,294,400 bytes. At 60 frames/s the payload rate is 497,664,000 bytes/s, about 0.498 GB/s. One full CPU copy logically reads and writes that payload, adding about 0.995 GB/s of memory traffic before cache effects, padding, and other pipeline work.

Those are arithmetic estimates, not measured throughput. On an integrated Jetson, CPU and GPU share physical DRAM; do not divide these bytes by a discrete GPU's PCIe bandwidth to invent a copy time. On a discrete GPU, distinguish host memory traffic, PCIe transfers, and GPU memory traffic.

If a buffer is reused every 50 ms but a consumer may hold it for 70 ms, “no memcpy” does not make the design correct. Choose backpressure, an explicit frame-drop policy, more storage with a bounded age policy, or lower processing time. The tested [SPSC lab](../labs/SPSC-Queue.md) demonstrates bounded queue ownership; it does not prove cross-process or device coherency.

## sendfile, splice, and userspace packet processing

`sendfile()` can transfer file contents without bouncing the payload through an application buffer, but can return a short count or an error. Loop according to the descriptor's blocking/nonblocking contract. `splice()` transfers between suitable descriptors with at least one pipe endpoint; supported paths and actual copies depend on the filesystem and device.

DPDK poll-mode drivers can avoid parts of the kernel network data path. They spend CPU cycles polling and depend on the NIC, queue layout, memory mappings, and driver mode. Huge pages can reduce TLB pressure; they do not eliminate TLB misses. Neither “100 Gb/s on one core” nor “zero interrupts” is a portable guarantee. Start with the workload and ownership contract before choosing kernel bypass.

## Answered checks

**Why can throughput improve while latency gets worse?** A larger batch amortizes submission cost but requests wait longer to join a batch or sit deeper in a queue. Measure end-to-end age as well as device service time.

**Does a successful DMA-BUF import mean the GPU may read immediately?** No. Import establishes access capability. Producer completion, cache visibility, and exclusive access still need their synchronization protocol.

**Can a cancel CQE justify freeing every submitted buffer?** No. Determine the terminal state of each target operation, including possible multishot completions, before reclaiming its storage.
