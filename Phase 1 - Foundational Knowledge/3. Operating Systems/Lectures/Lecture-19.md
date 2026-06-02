# Lecture 19: Modern I/O: io_uring, DMA-BUF & Zero-Copy Pipelines

## Overview

Every time your program asks the OS to read or write data, it pays a tax: a **context switch into kernel mode**, a **data copy** between kernel and user memory, and often a wait for slow hardware. At modest data rates this tax is invisible. At AI-system scale — millions of I/O operations per second, gigabytes per second of camera frames, continuous GPU inference — this overhead consumes a significant fraction of available CPU. This lecture addresses that problem.

The mental model to carry through this lecture is the **copy chain**: data originates in hardware (a sensor, a NIC, a storage device), and the goal is to get it to the GPU without the CPU ever touching it unnecessarily. Every copy, every syscall, and every context switch is a potential elimination target. io_uring eliminates syscall overhead for storage I/O. DMA-BUF eliminates copies between kernel subsystems. Zero-copy techniques like `sendfile`, DPDK, and VisionIPC eliminate copies at every remaining stage.

AI hardware engineers need to understand these mechanisms because the bottleneck in a real-time perception pipeline is often **not the GPU — it is the data path feeding the GPU**. A camera frame that spends 0.7 ms being copied across the PCIe bus is a frame that arrived 0.7 ms late to inference.

---

## Traditional I/O Limitations

The legacy **POSIX I/O model** imposes overhead that becomes a **bottleneck** in high-throughput AI data pipelines.

- `read()`/`write()`: each operation requires at least 2 syscalls (initiate + complete) plus a kernel-to-userspace data copy
- `select()`/`poll()`: O(n) scanning of file descriptor sets; degrades linearly with fd count
- `epoll`: eliminates O(n) scan but still requires one syscall per event notification; context switch cost accumulates at high IOPS

At 1M IOPS (NVMe throughput), syscall overhead alone can consume **30–50% of CPU cycles**.

> **Key Insight:** The POSIX I/O API was designed for correctness and portability, not for throughput. Each `read()` call is a round-trip: the CPU drops what it is doing, enters kernel mode, copies data, then returns. At high IOPS the CPU spends more time on this overhead than on actual work.

Think of it like a warehouse where every single box must be personally handed to a supervisor (kernel), who then hands it to the delivery driver (userspace). At low volumes this is fine. At a million boxes per second, the supervisor becomes the bottleneck.

---

## io_uring (Linux 5.1+)

io_uring replaces per-operation syscalls with **shared memory ring buffers** visible to both kernel and userspace simultaneously.

### Ring Buffer Architecture

Two rings reside in memory mapped into both kernel and userspace:

- **SQE ring (Submission Queue Entry)**: application writes operation descriptors here; kernel reads them
- **CQE ring (Completion Queue Entry)**: kernel writes completion status here; application polls without a syscall

The key insight is that both the application and the kernel can **read and write these rings directly** — no syscall crossing required for normal operation.

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Memory Region                      │
│                                                             │
│   SQ Ring (Submission Queue)        CQ Ring (Completion)   │
│  ┌──────────────────────────┐      ┌──────────────────────┐ │
│  │  SQE[0]: read fd=5       │      │ CQE[0]: res=512      │ │
│  │  SQE[1]: write fd=7      │      │ CQE[1]: res=0        │ │
│  │  SQE[2]: fsync fd=5      │      │ CQE[2]: res=512      │ │
│  │  SQE[3]: (empty)         │      │ CQE[3]: (empty)      │ │
│  └──────────────────────────┘      └──────────────────────┘ │
│        ↑ app writes here                ↑ kernel writes here │
│        ↓ kernel reads here             ↓ app polls here      │
└─────────────────────────────────────────────────────────────┘
         Userspace                          Kernel
         sees both rings ←── mmap ──→ sees both rings
```

> **Key Insight:** The SQ and CQ rings live in memory that is simultaneously visible to both userspace and the kernel. The application never needs to "hand data to the kernel" — it just writes to a shared slot. This is the fundamental reason io_uring can reach zero syscalls in its most aggressive configuration.

### Submission and Completion Flow

Understanding each step in this sequence is essential for tuning io_uring performance:

1. **Application calls `io_uring_prep_read()`**: fills one SQE slot with the operation descriptor (fd, buffer, length, offset). No kernel involvement yet — this is a pure userspace write to shared memory.
2. **Application calls `io_uring_submit()`**: this may call `io_uring_enter()` (one syscall for a batch of N operations) — or in SQPOLL mode, the kernel thread picks up the SQE without any syscall at all.
3. **Kernel processes the operation asynchronously**: I/O is dispatched to the block layer, network stack, or file system. The calling thread is free to do other work.
4. **Kernel writes a CQE to the CQ ring**: the completion result (bytes read, error code) is placed in the next available CQ slot. This is a write into shared memory — no interrupt to userspace required.
5. **Application polls the CQ ring**: the app checks the CQ head pointer. If a new CQE is present, it reads the result directly. **No syscall required for completion.**

With `IORING_SETUP_SQPOLL`, a dedicated kernel thread continuously drains the SQ ring. The submit path becomes **zero-syscall**. The application only polls the CQ ring.

> **Common Pitfall:** Forgetting to call `io_uring_cqe_seen()` after processing a completion entry. This advances the CQ ring head pointer. Without it, the ring fills up, new completions are dropped (unless `IORING_FEAT_NODROP` is set, which applies backpressure instead), and the application stalls.

### Key Flags and Features

| Flag / Feature | Meaning |
|---|---|
| `IORING_SETUP_SQPOLL` | Kernel thread auto-submits; zero syscall submit path |
| `IORING_SETUP_IOPOLL` | Kernel polls for completions (no IRQ); lowest latency |
| `IORING_FEAT_NODROP` | CQEs never dropped; backpressure instead of loss |
| Fixed buffers | Pre-registered buffers skip `get_user_pages()` per op |
| Multishot | Single SQE generates multiple CQEs (e.g., accept loop) |

### Supported Operations

`read`, `write`, `send`, `recv`, `accept`, `connect`, `fsync`, `splice`, `openat`, `statx`, `timeout`, `link`, `hardlink`, `renameat`, `unlinkat`

The breadth of supported operations is significant: io_uring is not just a storage optimization — it can replace nearly every blocking I/O syscall in a network or file server.

### liburing Helper Library

```c
struct io_uring ring;
io_uring_queue_init(256, &ring, 0);          // init with depth 256 SQEs

struct io_uring_sqe *sqe = io_uring_get_sqe(&ring); // get a free SQE slot
io_uring_prep_read(sqe, fd, buf, len, 0);            // fill the SQE descriptor
io_uring_sqe_set_data(sqe, user_data_ptr);           // tag with user context pointer

io_uring_submit(&ring);                      // single syscall submits ALL pending SQEs

struct io_uring_cqe *cqe;
io_uring_wait_cqe(&ring, &cqe);             // blocks until one CQE ready (or poll)
// cqe->res contains bytes read, or negative errno on error
io_uring_cqe_seen(&ring, cqe);              // advance CQ head — MUST call this
```

The `io_uring_submit()` call is batched: all SQEs prepared since the last submit are sent in a single syscall. Applications that pipeline many reads or writes before calling submit achieve near-zero syscall overhead.

### Performance Comparison

| Method | IOPS (NVMe 4K rand read) | CPU utilization | Syscalls per op |
|---|---|---|---|
| `pread()` | ~200K | high | 1 |
| `epoll` + `aio` | ~600K | moderate | 2 |
| `io_uring` (default) | ~800K | low | ~0.1 (batched) |
| `io_uring` + SQPOLL | 1M+ | near-zero | 0 |

> **Key Insight:** The jump from `pread()` at 200K IOPS to `io_uring` + SQPOLL at 1M+ IOPS is almost entirely CPU overhead elimination, not hardware speedup. The NVMe SSD is the same in all rows. What changes is how much CPU time is wasted on syscalls and copies.

---

## DMA-BUF: Cross-Subsystem Buffer Sharing

With io_uring handling syscall overhead for storage, the next bottleneck is copying data between kernel subsystems — for example, between a camera driver and a GPU. DMA-BUF solves this.

DMA-BUF provides a **file descriptor abstraction** for memory buffers that multiple kernel subsystems and userspace can **share without copying**.

- One subsystem allocates a buffer and exports it as an fd via `dma_buf_export()` → `dma_buf_fd()`
- Another subsystem imports the fd via `dma_buf_get()` and maps it into its own DMA address space
- Userspace passes fds between processes via `sendmsg()` or direct API calls

Think of a DMA-BUF fd as a key to a locker. The camera driver puts data in the locker. The GPU driver opens the same locker with the same key. No data is moved — only the key is passed.

```
┌──────────────┐   exports fd    ┌─────────────────────────────────┐
│ GPU Allocator│ ─────────────→  │       DMA-BUF Object             │
│ (nvmap/ION)  │                 │  (physical pages in device mem)  │
└──────────────┘                 └──────────────────────────────────┘
                                          ↑            ↑
                               imports   │            │ imports
                               ┌─────────┘            └──────────┐
                          ┌────┴─────┐               ┌───────────┴──┐
                          │ V4L2     │               │ CUDA kernel  │
                          │ camera   │               │ (device ptr) │
                          │ DMA eng. │               └──────────────┘
                          └──────────┘
                          ISP writes                 GPU reads
                          directly here              directly here
                                ↕ NO CPU COPY ↕
```

### V4L2 + DMA-BUF Integration

- `V4L2_MEMORY_DMABUF`: V4L2 buffer type that accepts external DMA-BUF fds
- Application exports a DMA-BUF fd from GPU allocator (nvmap, ION, or DRM allocator)
- Passes fd to camera driver via `VIDIOC_QBUF` with `m.fd` set
- Camera DMA engine writes captured frame directly to GPU-accessible memory
- `VIDIOC_EXPBUF`: export a V4L2 `MMAP` buffer as a DMA-BUF fd for import by another device

> **Key Insight:** `V4L2_MEMORY_DMABUF` inverts the normal flow. Instead of the camera driver allocating its own buffer and then copying out, the application provides a buffer that the camera engine writes into directly. The application chooses a buffer that is also visible to the GPU, making the copy physically impossible (there is only one copy of the data, in one location).

---

## Zero-Copy Camera to Inference Pipeline (Jetson)

Now we combine io_uring and DMA-BUF into the full zero-copy pipeline. This section shows the concrete benefit: eliminating 1–2 memory copies from every camera frame.

Traditional pipeline: Camera → DMA → kernel buffer → copy to userspace → copy to GPU (**2 extra copies**).

Zero-copy pipeline via DMA-BUF:

1. **GPU allocates buffer**: `cudaMalloc()` → nvmap handle → `dma_buf_fd()`. The buffer lives in GPU-accessible memory from the start.
2. **Camera (V4L2) is given the buffer fd**: `VIDIOC_QBUF` with `V4L2_MEMORY_DMABUF`; fd passed to camera driver. The camera driver now knows exactly where to write the frame.
3. **Camera ISP DMA engine writes directly into that GPU-mapped buffer**: the camera hardware DMAs the captured frame into the GPU buffer. No CPU is involved in this transfer.
4. **CUDA inference**: buffer already in GPU memory; no `cudaMemcpy()` needed. The CUDA kernel reads the frame from the pointer that was allocated in step 1.
5. **Access via `cudaGraphicsMapResources()` or direct device pointer**: standard CUDA APIs work normally; they just happen to be operating on data that arrived without any CPU-side copy.

Latency reduction: eliminates 1–2 CPU-GPU memcpy operations. At PCIe Gen3 bandwidth (~12 GB/s), copying a 1080p RGBA frame (~8 MB) costs ~0.7 ms. At 30 fps with 3 cameras, eliminated copies save significant memory bandwidth.

> **Common Pitfall:** A common mistake is allocating the camera buffer with `malloc()` and only later trying to import it via DMA-BUF. The buffer must be allocated through the GPU-visible allocator (nvmap, ION, or `cudaMallocManaged`) from the start. A `malloc()`'d buffer has no DMA-BUF handle and cannot be passed to the camera driver as a target.

---

## splice and sendfile: Kernel-to-Kernel Zero-Copy

Once data is in the kernel, `splice` and `sendfile` allow it to be moved between kernel subsystems **without ever surfacing in userspace**.

Both calls move data between file descriptors without copying to userspace:

- `sendfile(out_fd, in_fd, &offset, count)`: copy from file or socket to socket inside kernel; used for HTTP video streaming and static file serving
- `splice(fd_in, off_in, fd_out, off_out, len, flags)`: more general; works with pipes; can chain across calls
- `tee(fd_in, fd_out, len, flags)`: duplicate pipe data without consuming it

Use case: camera recording server sending H.264 frames over HTTP without a userspace buffer copy.

> **Key Insight:** `sendfile` is the reason a web server can serve a 1 GB video file at near-wire speed using almost no CPU. The data travels: NVMe → page cache → NIC DMA, all without crossing the user/kernel boundary. The server process issues a single syscall and the hardware handles the rest.

---

## DPDK: User-Space Network Driver

For inference-serving scenarios where network I/O must match GPU throughput, even kernel network stack overhead becomes unacceptable. DPDK eliminates it entirely.

DPDK (Data Plane Development Kit) **bypasses the kernel network stack** entirely:

- User-space PMD (Poll Mode Driver) owns the NIC; no kernel driver, no interrupts
- Huge pages (2 MB/1 GB) for packet buffers: eliminates TLB misses at line rate
- CPU core polling at 100% — achieves 100 Gbps with a single core
- No context switches, no syscalls, no socket buffer copies
- Used in inference-serving front-ends where network I/O must match GPU throughput

Tools: `dpdk-testpmd` for benchmarking; `rte_mbuf` for zero-copy packet buffers; `rte_ring` for inter-core packet passing.

> **Common Pitfall:** DPDK dedicates a CPU core to 100% polling. This is intentional — it is the price of near-zero latency network I/O. Running DPDK on a shared core, or alongside other workloads on that core, destroys its latency guarantees. DPDK cores must be isolated with `isolcpus` in the kernel boot parameters.

---

## VisionIPC: openpilot Zero-Copy Video IPC

The final piece of the zero-copy puzzle is passing video frames between processes without copying. VisionIPC demonstrates this at a production level.

openpilot replaces socket-based video transfer with **shared memory**:

- `vipc_server` allocates a pool of shared memory buffers at startup via `mmap(MAP_SHARED)`
- `camerad` (producer): fills a buffer, posts the buffer index to consumers via semaphore
- `modeld` and `encoderd` (consumers): receive the index, map the same memory region, read directly
- No video data copy between processes; only a small integer (buffer index) is communicated
- `cereal` handles all other IPC (non-video) via capnproto over `msgq` (also shared memory)

```
┌────────────┐  fills buffer[N]   ┌─────────────────────────────┐
│  camerad   │ ──────────────────→│  Shared Memory Buffer Pool  │
│ (producer) │  posts index N     │  buf[0]: 1920×1208 YUV      │
└────────────┘  via semaphore     │  buf[1]: 1920×1208 YUV      │
                      ↓           │  buf[2]: 1920×1208 YUV      │
             ┌────────┴────────┐  └─────────────────────────────┘
             │                 │           ↑          ↑
        ┌────▼─────┐   ┌───────▼────┐    reads      reads
        │  modeld  │   │  encoderd  │  buffer[N]   buffer[N]
        │(inference│   │(H.265 enc.)│  directly    directly
        └──────────┘   └────────────┘
         GPU kernel     encoder DMA
         reads buf[N]   reads buf[N]
         NO COPY        NO COPY
```

> **Key Insight:** VisionIPC shows the real-world application of every concept in this lecture. The camera fills a buffer once. Multiple consumers — the neural network inference engine and the video encoder — read from that same buffer. The GPU processes the frame in-place. No data is ever duplicated. This is achievable because the OS shared memory primitives (`mmap(MAP_SHARED)`, DMA-BUF) allow multiple subsystems to reference the same physical memory simultaneously.

---

## Summary

| I/O Method | Syscall per op | Data copy | Max throughput | Use case |
|---|---|---|---|---|
| `read()`/`write()` | 1 | kernel to user | ~200K IOPS | Simple file I/O |
| `epoll` + callbacks | 1 per event | kernel to user | ~600K IOPS | Network servers |
| `io_uring` batched | ~0.1 (batched) | kernel to user | ~800K IOPS | NVMe logging |
| `io_uring` SQPOLL | 0 | kernel to user | 1M+ IOPS | Ultra-low latency |
| `sendfile` | 1 | none (kernel) | line rate | Video streaming |
| DMA-BUF + V4L2 | 0 (DMA) | none | hardware DMA rate | Camera to GPU |
| DPDK | 0 | none | 100 Gbps | Inference serving |
| VisionIPC | 0 (shared mem) | none | memory bandwidth | openpilot camera IPC |

### Conceptual Review

- **What is the fundamental cost of a traditional `read()` call?** Two mode switches (user→kernel→user) plus one data copy from kernel buffer to user buffer. At 1M IOPS this overhead can consume 30–50% of available CPU cycles.

- **How does io_uring's ring buffer eliminate syscalls?** By placing the submission and completion queues in memory that is simultaneously mapped into both kernel and user address spaces, the application can post work and read results without ever entering kernel mode. With SQPOLL, a kernel thread drains the submission ring continuously.

- **What problem does DMA-BUF solve that io_uring does not?** io_uring reduces syscall overhead for existing I/O paths. DMA-BUF eliminates entire copies between kernel subsystems — the camera driver, GPU driver, and display driver can all reference the same physical buffer via a file descriptor, so data written by one is immediately readable by another without any copy.

- **Why does the zero-copy camera-to-GPU pipeline require the GPU to allocate the buffer, not the camera driver?** The buffer must be visible to GPU hardware (mapped into GPU address space). Only the GPU allocator (nvmap, ION) produces a buffer with both a DMA-BUF fd (for the camera driver) and a CUDA device pointer (for inference). If the camera driver allocates the buffer, it lives in kernel memory with no GPU mapping.

- **What is the trade-off of DPDK's polling model?** A dedicated CPU core runs at 100% utilization continuously. This eliminates all interrupt and context-switch overhead, achieving line-rate packet processing. The cost is one full CPU core permanently consumed. This is acceptable in inference-serving systems where the NIC core is paired with many GPU cores.

- **How does VisionIPC prevent multiple processes from racing on the same buffer?** A semaphore (posted by `camerad` after filling a buffer) signals readiness. Consumers read from the shared memory region indexed by the buffer number received via the semaphore. The buffer is not returned to the pool until all consumers acknowledge, preventing camerad from overwriting a buffer still in use by modeld or encoderd.

---

## AI Hardware Connection

- DMA-BUF with `V4L2_MEMORY_DMABUF` enables true zero-copy camera-to-GPU pipelines on Jetson; frames arrive in GPU memory directly from the ISP with no CPU involvement in the data path
- io_uring with `IORING_SETUP_SQPOLL` is applicable to high-throughput CAN bus and sensor logging in openpilot, achieving 1M+ IOPS with near-zero CPU cost
- VisionIPC demonstrates the OS-level shared memory design that eliminates inter-process video copies in production autonomous driving software
- DPDK is used in cloud inference front-ends where network I/O at 100 Gbps must be matched to GPU throughput without kernel overhead
- `sendfile` enables camera stream relay servers to forward encoded video to clients with zero userspace buffer involvement
- Understanding the full copy chain (camera ISP → kernel buffer → userspace → GPU) is prerequisite for diagnosing latency in any AI perception pipeline
