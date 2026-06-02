# Lecture 18: Character Drivers, Interrupt-Driven I/O & V4L2

## Overview

The previous lecture established how the kernel discovers hardware and binds drivers to devices. This lecture covers what a driver actually does once it is bound: exposing a programming interface to userspace, responding to hardware events in real time, and integrating with the V4L2 camera subsystem for sensor-to-inference pipelines. The core challenge is designing an interface that is efficient, safe across the user/kernel boundary, and responsive to hardware events without wasting CPU on busy-waiting. The mental model is an event-driven pipeline: hardware generates data, the interrupt handler wakes waiting userspace readers, and data flows through a zero-copy buffer path from sensor to inference engine. For an AI hardware engineer, these patterns appear directly in every custom accelerator driver, every camera pipeline implementation, and every low-latency inference data path.

---

## Character Device Driver

A **character device** exposes a **file-like interface** in `/dev/`. Userspace opens, reads, writes, and issues ioctls on it just like a regular file. Under the hood, each operation dispatches to a driver-defined function. This is the **standard interface for custom AI accelerators**, FPGA control planes, and any hardware that does not fit a more specialized subsystem.

```
Character Device: Kernel ↔ Userspace Interface

Userspace               Kernel (driver code)
─────────               ───────────────────
open("/dev/mydev0")  →  mydev_open()
read(fd, buf, n)     →  mydev_read()    ← blocks until data ready
write(fd, buf, n)    →  mydev_write()   ← sends data to device
ioctl(fd, CMD, arg)  →  mydev_ioctl()   ← device-specific control
mmap(fd, ...)        →  mydev_mmap()    ← map DMA buffer to user VA
poll(fd, ...)        →  mydev_poll()    ← report readiness for epoll
close(fd)            →  mydev_release() ← cleanup
```

### Registration

Setting up a character device follows a fixed sequence:

1. **Allocate device numbers** — the kernel assigns major/minor numbers dynamically (preferred) or statically.
2. **Initialize the cdev structure** — link the file_operations table to the cdev.
3. **Add cdev to the kernel** — make it live; `open()` calls can arrive after this point.
4. **Create a device class** — used by udev to create the `/dev/` node automatically.
5. **Create the device** — creates the actual `/dev/mydev0` node.

```c
alloc_chrdev_region(&devno, 0, 1, "mydev");   // dynamic major/minor
cdev_init(&mydev->cdev, &mydev_fops);
cdev_add(&mydev->cdev, devno, 1);
cls = class_create(THIS_MODULE, "mydev");
device_create(cls, NULL, devno, NULL, "mydev0");   // creates /dev/mydev0
```

### struct file_operations

| Operation | Signature | Purpose |
|-----------|-----------|---------|
| `open` | `(inode, file)` | Allocate per-file state; check permissions |
| `release` | `(inode, file)` | Free per-file state; flush hardware |
| `read` | `(file, buf, len, off)` | Copy data to user; block if unavailable |
| `write` | `(file, buf, len, off)` | Copy data from user; submit to device |
| `ioctl` / `unlocked_ioctl` | `(file, cmd, arg)` | Device-specific commands |
| `mmap` | `(file, vma)` | Map device/kernel memory into user VA |
| `poll` | `(file, poll_table)` | Report readiness for select/poll/epoll |

---

## ioctl Interface

The **ioctl** (I/O control) interface is the standard mechanism for device-specific commands that do not fit the read/write model. Think of it as a typed function call through a file descriptor.

```c
#define MYDEV_IOC_MAGIC  'M'
#define MYDEV_RESET      _IO(MYDEV_IOC_MAGIC,  0)       // no arg
#define MYDEV_GET_STATUS _IOR(MYDEV_IOC_MAGIC, 1, u32)  // read from device
#define MYDEV_SET_CONFIG _IOW(MYDEV_IOC_MAGIC, 2, struct mydev_cfg) // write

static long mydev_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
    struct mydev_cfg cfg;
    if (copy_from_user(&cfg, (void __user *)arg, sizeof(cfg)))
        return -EFAULT;   // never dereference user pointer directly
    // ...
}
```

The `_IO`, `_IOR`, `_IOW` macros encode the **magic number, command number, direction, and argument size** into a single 32-bit value. The kernel uses this encoding to **validate the argument size** automatically.

- Never dereference `(void *)arg` directly in kernel; always use `copy_from_user()` / `copy_to_user()`
- `copy_*_user` returns bytes not copied on error; check return value

> **Common Pitfall:** Using `(struct foo *)arg` directly (casting the user pointer without `copy_from_user`) is a critical security vulnerability. The user pointer may point to invalid memory, unmapped memory, or memory that changes between the check and the use (TOCTOU race). Always copy to a kernel-side stack variable using `copy_from_user()`.

---

## mmap in Drivers

The **mmap** file operation maps kernel or device memory directly into userspace virtual address space. This enables **zero-copy access** to DMA output buffers — userspace reads inference results at memory speed without a syscall per access.

```c
static int mydev_mmap(struct file *f, struct vm_area_struct *vma)
{
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot); // for MMIO: disable caching
    return remap_pfn_range(vma, vma->vm_start,
                           phys_addr >> PAGE_SHIFT,          // physical frame number
                           vma->vm_end - vma->vm_start,      // mapping size
                           vma->vm_page_prot);
}
```

`pgprot_noncached()` marks the mapped pages as **uncached** — essential for MMIO regions and DMA output buffers where the CPU must always read from RAM rather than its cache.

- `remap_pfn_range()`: maps contiguous physical range; used for DMA buffers and MMIO
- `vm_insert_page()`: maps individual pages; used for vmalloc-backed buffers
- `vm_ops->fault`: demand-map pages on access; used for large sparse buffers

> **Key Insight:** `mmap` eliminates the syscall-per-frame overhead of `read()`. Instead of calling `read()` for each inference result, userspace maps the DMA output buffer once and reads results directly. For a 60fps camera pipeline, this avoids 60 syscalls per second per buffer — negligible in isolation, but meaningful when multiplied across many buffers in a multi-camera system.

---

## Interrupt-Driven I/O

Polling the hardware to check if new data is ready **wastes CPU cycles** and increases latency. **Interrupt-driven I/O** lets the hardware signal the CPU precisely when data is ready, allowing the CPU to do useful work in the meantime.

```
Interrupt-Driven Data Flow:

Hardware (camera ISP / AI accelerator)
    │
    │  DMA transfer complete
    ▼
IRQ line asserted
    │
    ▼
CPU's interrupt controller (GIC/APIC)
    │
    ▼
ISR (top half): runs with interrupts disabled
    ├── read STATUS_REG to acknowledge IRQ
    ├── set dev->data_ready = true
    └── wake_up_interruptible(&dev->wait_q)
                │
                ▼
        blocked read() thread wakes up
            │
            ▼
        copies data to userspace, returns to application
```

```c
devm_request_irq(dev, irq, mydev_isr, IRQF_SHARED, "mydev", priv);

static irqreturn_t mydev_isr(int irq, void *data)
{
    struct mydev *dev = data;
    u32 status = readl(dev->base + STATUS_REG);      /* read and clear interrupt status */
    if (!(status & MY_IRQ_BIT)) return IRQ_NONE;     /* not our interrupt (shared IRQ line) */

    dev->data_ready = true;
    wake_up_interruptible(&dev->wait_q);   // unblock waiting readers
    return IRQ_HANDLED;
}
```

The ISR returns `IRQ_NONE` if the status register shows the interrupt came from a **different device sharing the same IRQ line**. This is required for `IRQF_SHARED` interrupt lines — the kernel will call all registered handlers until one claims it.

### Top Half / Bottom Half Split

ISRs run with **interrupts disabled** on the executing CPU. Long work must be **deferred** to a context where sleeping and blocking are permitted:

| Mechanism | Context | Use |
|-----------|---------|-----|
| ISR (top half) | Interrupt; no sleep | Acknowledge IRQ; wake queue |
| `tasklet_schedule()` | Softirq; no sleep | Short deferred work |
| `queue_work()` | Kernel thread; can sleep | I/O, memory alloc, long work |

> **Common Pitfall:** Performing `kmalloc(GFP_KERNEL)` or any sleeping operation in the ISR (top half) causes a kernel BUG: "BUG: scheduling while atomic." Always limit ISR code to register reads, flag sets, and queue wakeups. Delegate any work that may sleep to a workqueue via `queue_work()`.

---

## Wait Queues

**Wait queues** are the synchronization primitive that connects ISR notifications to blocking userspace operations. The ISR calls `wake_up_interruptible()` to unblock threads waiting in `wait_event_interruptible()`.

```c
wait_queue_head_t wq;
init_waitqueue_head(&wq);

// Producer (ISR): wake waiters
wake_up_interruptible(&wq);

// Consumer (read fop): sleep until condition
wait_event_interruptible(wq, dev->data_ready);
```

The `wait_event_interruptible()` macro **checks the condition** and returns immediately if true, or puts the thread to sleep if false. When the ISR calls `wake_up_interruptible()`, all sleeping threads are woken and **re-check the condition**.

- `wait_event_interruptible()` returns `-ERESTARTSYS` if signal received; caller should return `-EINTR` to userspace
- `wait_event_interruptible_timeout()`: add deadline for polling fallback

> **Key Insight:** `wait_event_interruptible()` is not just a sleep — it atomically checks the condition before sleeping to avoid the race where the ISR fires and sets `data_ready` after the check but before the sleep. The macro's implementation ensures this atomicity using the wait queue spinlock.

---

## poll() / select() / epoll in Drivers

For applications that multiplex multiple devices (multiple cameras, multiple inference outputs), blocking `read()` per device is impractical. The `poll` fop integrates the driver with Linux's I/O multiplexing infrastructure.

```c
static __poll_t mydev_poll(struct file *f, poll_table *wait)
{
    poll_wait(f, &dev->wait_q, wait);   // register wait queue with poll infrastructure
    if (dev->data_ready)
        return EPOLLIN | EPOLLRDNORM;   // signal: data available for reading
    return 0;                           // signal: not yet ready
}
```

`poll_wait()` registers the driver's wait queue with the kernel's polling infrastructure **without actually sleeping**. When `epoll_wait()` is called in userspace, the kernel will wake the epoll thread when `wake_up_interruptible()` is called on this wait queue.

Userspace `epoll_wait()` returns when `EPOLLIN` is set; enables event-driven data pipelines without busy-polling.

With the low-level interrupt and polling infrastructure established, the V4L2 subsystem builds a standardized camera capture framework on top of these primitives.

---

## V4L2 (Video4Linux2)

**V4L2** is the kernel subsystem for **cameras and video capture devices**. It standardizes the API for frame capture, format negotiation, and buffer management across all camera hardware. Header: `<linux/videodev2.h>`.

```
V4L2 Pipeline Architecture

Camera Sensor (IMX477)
    │  I2C config, MIPI CSI-2 data
    ▼
NVCSI (MIPI CSI-2 receiver)
    │  raw pixel data
    ▼
ISP (Image Signal Processor)
    │  debayer, denoise, exposure, color correction
    ▼
V4L2 capture node (/dev/video0)
    │  DMABUF fd
    ▼
CUDA importer (GPU)
    │  inference kernel
    ▼
Output (display / network / storage)
```

### Capture Sequence

The V4L2 capture sequence follows a fixed protocol. Each step maps to a specific ioctl:

1. **`VIDIOC_QUERYCAP`** — query driver capabilities; verify it supports streaming and the required formats.
2. **`VIDIOC_S_FMT`** — set pixel format (e.g., `V4L2_PIX_FMT_NV12`), resolution, and stride.
3. **`VIDIOC_REQBUFS`** — allocate N kernel-side DMA buffers; choose memory type (MMAP or DMABUF).
4. **`VIDIOC_QBUF`** — enqueue each buffer; hand ownership to the driver for DMA fill.
5. **`VIDIOC_STREAMON`** — start the hardware pipeline; ISP begins capturing frames.
6. **`poll()` / `select()`** — wait (without busy-loop) for a buffer to be filled.
7. **`VIDIOC_DQBUF`** — dequeue a filled buffer; driver returns ownership to userspace.
8. **Process the frame** — run inference, display, encode, or store.
9. **`VIDIOC_QBUF`** — re-enqueue the buffer for the next frame.
10. **`VIDIOC_STREAMOFF`** — stop the hardware; all queued buffers returned to userspace.

```
VIDIOC_QUERYCAP     → verify capabilities
VIDIOC_S_FMT        → set pixel format, resolution
VIDIOC_REQBUFS      → allocate N buffers (MMAP or DMABUF)
VIDIOC_QBUF         → enqueue buffer (give to driver)
VIDIOC_STREAMON     → start streaming
poll() / select()   → wait for filled buffer
VIDIOC_DQBUF        → dequeue filled buffer; process frame
VIDIOC_QBUF         → re-enqueue for next frame
VIDIOC_STREAMOFF    → stop streaming
```

> **Common Pitfall:** Not re-enqueueing the buffer with `VIDIOC_QBUF` promptly after `VIDIOC_DQBUF`. If userspace processes the frame slowly and the queue runs dry, the driver has no buffer to fill and drops frames. Maintain a minimum of 2–3 buffers in the queue at all times, using a producer-consumer thread model if processing is slow.

### Buffer Memory Types

| Type | Description | Zero-copy to GPU? |
|------|-------------|------------------|
| `V4L2_MEMORY_MMAP` | Kernel allocates; user mmap()s | No (CPU copy needed) |
| `V4L2_MEMORY_USERPTR` | User allocates; driver pins | Conditional |
| `V4L2_MEMORY_DMABUF` | Import external DMA-BUF fd | Yes (GPU imports same buf) |

`V4L2_MEMORY_DMABUF` is the key to **zero-copy camera→GPU pipelines**. The GPU creates a DMA-BUF buffer, exports it as an fd, and the V4L2 driver fills it directly via DMA. The GPU then reads inference input from the same physical pages — no data is ever copied.

### Media Controller

For complex pipelines with multiple linked stages (sensor → ISP → CSI → capture node), V4L2's **media controller** represents the pipeline as a directed graph of entities and links.

- `/dev/media0`: represents the full pipeline as an entity graph
- Entities: sensor → ISP → CSI bridge → V4L2 capture node
- `MEDIA_IOC_SETUP_LINK`: enable/disable data paths between entities
- `media-ctl` tool: configure pipeline from shell; inspect entity properties

```
Media Controller Entity Graph (Jetson IMX477 example)

[IMX477 sensor]──MIPI──>[NVCSI]──pixel──>[ISP]──NV12──>[/dev/video0]
      │                                              │
      │ I2C (exposure, gain,                        │ DMABUF fd
      │ focus control)                              ▼
      │                                     CUDA inference kernel
      │
  /dev/v4l2-subdev0    /dev/media0 graph:  use media-ctl to enable links
```

### openpilot camerad Pipeline

```
IMX sensors → NVCSI → ISP (NvCamSrc) → V4L2/ISP API
    --> frame buffer (DMA-BUF or nvmap)
    --> VisionIPC shared memory
    --> modeld (neural network inference)
```

This is the **production pipeline in openpilot**. Frames flow from the physical sensor through the ISP, into DMA-BUF backed buffers, shared via VisionIPC to the inference model — with **no CPU copy at any stage**.

> **Key Insight:** The V4L2 `VIDIOC_QBUF`/`VIDIOC_DQBUF` buffer exchange is the synchronization protocol of the camera pipeline. The camera driver owns the buffer between QBUF and DQBUF; userspace owns it otherwise. Violating this ownership (reading a buffer while the driver is filling it) causes torn frames and non-deterministic corruption.

---

## Summary

| Driver type | Key fops | Kernel subsystem | Example use |
|-------------|----------|-----------------|------------|
| Char device | open/read/write/ioctl/mmap/poll | `cdev` | FPGA control, custom accelerator |
| V4L2 capture | VIDIOC_* ioctls | `videobuf2` | Camera sensor, USB UVC |
| V4L2 + DMA-BUF | REQBUFS(DMABUF) + DQ/QBUF | `videobuf2` + `dma-buf` | Jetson ISP → GPU pipeline |
| Platform + IRQ | ISR + waitqueue + poll | `platform_driver` | AI accelerator interrupt |

### Conceptual Review

- **Why must `copy_from_user()` be used instead of directly casting the ioctl `arg` pointer?** The `arg` is a user-space pointer. Dereferencing it directly in the kernel is dangerous: it may be unmapped, may point to kernel memory (privilege escalation), or may change between check and use (TOCTOU race). `copy_from_user()` validates the pointer and safely copies the data to kernel space.

- **What is the top half / bottom half split in interrupt handling?** The ISR (top half) runs with interrupts disabled and must be minimal: acknowledge the IRQ, record state, wake a wait queue. Longer work (memory allocation, I/O, sleeping) is deferred to a workqueue (bottom half) that runs in kernel thread context where sleeping is allowed.

- **How does `wait_event_interruptible()` avoid the missed-wakeup race?** The macro checks the condition inside the wait queue lock. If the ISR fires and sets `data_ready` between the condition check and the sleep operation, the wait queue machinery ensures the thread either sees the condition as true and does not sleep, or is woken immediately by the `wake_up_interruptible()` call.

- **What is the difference between `V4L2_MEMORY_MMAP` and `V4L2_MEMORY_DMABUF`?** With MMAP, the kernel allocates the buffer and userspace maps it — but it cannot be directly imported by a GPU without a copy. With DMABUF, an external allocator (e.g., the GPU driver) creates the buffer and exports an fd; the V4L2 driver imports it and fills it via DMA. The GPU reads the same physical pages — no copy needed.

- **What is the media controller and why is it needed for camera pipeline configuration?** The media controller is a graph of pipeline entities (sensor, ISP, CSI receiver, capture node) and links between them. Complex ISPs have multiple output paths and processing stages. The media controller API (`MEDIA_IOC_SETUP_LINK`) allows userspace to enable/disable specific paths without modifying kernel code.

- **Why is `remap_pfn_range()` used in the mmap fop instead of copying to userspace?** `remap_pfn_range()` installs PTEs in the user process's page table pointing directly to the physical pages of the DMA buffer. The CPU can then read these pages at memory speed through normal pointer dereference — no syscall, no copy, no kernel involvement per access. This is how inference result buffers can be read at full memory bandwidth.

---

## AI Hardware Connection

- V4L2 with `V4L2_MEMORY_DMABUF` enables zero-copy camera frame delivery to CUDA on Jetson; buffer is imported by CUDA without any CPU memcpy.
- openpilot camerad uses the V4L2 / ISP API to capture frames from IMX sensors and passes DMA-BUF fds through VisionIPC to modeld for inference.
- Custom ioctl interface is the standard pattern for FPGA control plane: configure kernel parameters, trigger DMA, query hardware status.
- `mmap` in driver + `remap_pfn_range` enables userspace inference engines to read DMA output buffers at memory speed without syscall overhead per frame.
- Wait queues with `wake_up_interruptible` from the sensor ISR is the correct pattern for notifying inference threads of data-ready events with zero busy-wait CPU overhead.
- Media controller pipeline configuration is required for Jetson camera bring-up: sensor format, ISP processing stages, and output node must all be explicitly linked before streaming begins.
