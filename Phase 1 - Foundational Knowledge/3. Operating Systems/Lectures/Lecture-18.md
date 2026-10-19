# Lecture 18: Character Drivers, Interrupt-Driven I/O, and V4L2

A driver interface is a contract about ownership. An `ioctl` may start work; an interrupt may report progress; a mapping may expose bytes. None of those alone means that a buffer can be overwritten or freed. This lesson follows an accelerator from a `/dev` interface to a camera-to-GPU pipeline.

Prerequisites: [driver resources](Lecture-17.md), [interrupt contexts](Lecture-03.md), and [DMA mappings](Lecture-15.md). Kernel fragments below are **reading examples**, not a loadable driver. A real driver needs a hardware specification, resource checks, locking, cancellation, and a removal protocol.

## File operations and publication

```text
open → acquire a reference to live device state
read / write / ioctl / mmap / poll → operate under the device protocol
close → release the file's reference
unbind → stop new work, quiesce existing work, retire device state safely
```

`alloc_chrdev_region()` reserves numbers; `cdev_init()` associates operations; `cdev_add()` makes the character device callable. Initialize locks, wait queues, bounds, and hardware state **before publishing it**. Check every return, including `IS_ERR()` for class/device creation, and unwind each successful registration on later failure.

On Linux 6.4 and later, the class call is `class_create("mydev")`; older BSPs may require `class_create(THIS_MODULE, "mydev")`. Compile against the target kernel headers. `device_create()` publishes a device-model object; devtmpfs/udev normally arrange a node, depending on system configuration. Neither creation nor destruction of a node revokes existing open descriptors.

A module reference and a device-state reference solve different problems. `.owner = THIS_MODULE` can prevent unloading module code while files are open. Hot-unplugged hardware and per-device allocations still need an explicit reference-counted lifetime and a “device gone” state. A mapping can outlive the file descriptor that created it.

## Design an ioctl deliberately

`_IO`, `_IOR`, `_IOW`, and `_IOWR` encode a request number and type-size convention. They do **not** automatically validate a driver's argument. Use a shared UAPI header with fixed-width types such as `__u32`, explicit reserved fields, and a documented compatibility policy. Avoid native pointers or `long` in a layout intended for both 32-bit and 64-bit applications.

```c
/* Reading fragment: device-specific helpers and UAPI definitions omitted. */
switch (cmd) {
case MYDEV_RESET:                 /* _IO: no pointer argument */
    return reset_if_idle(dev);   /* serialize against submissions */
case MYDEV_GET_STATUS: {          /* _IOR: kernel writes to userspace */
    __u32 status = snapshot_status(dev);
    return copy_to_user((void __user *)arg, &status, sizeof(status))
           ? -EFAULT : 0;
}
case MYDEV_SET_CONFIG: {          /* _IOW: userspace supplies input */
    struct mydev_cfg cfg;
    if (copy_from_user(&cfg, (void __user *)arg, sizeof(cfg)))
        return -EFAULT;
    if (!config_is_valid(&cfg))
        return -EINVAL;
    return apply_config_while_idle(dev, &cfg);
}
default:
    return -ENOTTY;
}
```

Dispatch before copying: RESET has no input structure, and GET_STATUS has no input configuration. `copy_*_user()` returns bytes not copied. A successful copy produces a kernel-owned snapshot, but its fields remain untrusted. Validate lengths, arithmetic overflow, ranges, reserved bits, and device state. Initialize every byte copied back, including padding. User-copy helpers may sleep; perform them outside a hard-IRQ context and outside spinlocked regions. See [ioctl conventions](https://docs.kernel.org/userspace-api/ioctl/ioctl-number.html).

## mmap: an address is not permission to reuse storage

Use the mapping helper matching the allocation. For `dma_alloc_coherent()` storage, a driver normally uses `dma_mmap_coherent()` with the original device, CPU address, DMA address, and allocation size. Do not use `virt_to_phys()` on an arbitrary DMA buffer or force every DMA mapping uncached. The DMA API selects platform-appropriate attributes. MMIO mappings and ordinary page-backed allocations have different rules.

Validate the requested offset and size, access permissions, and lifetime. A successful mapping does not stop a device from overwriting a frame. Cache visibility, device completion, and exclusive ownership remain separate responsibilities. `mmap()` can remove a userspace copy; measure whether that copy was significant before treating it as a latency optimization.

## Waiting, waking, and reading

An interrupt handler identifies its device's pending work, acknowledges according to the hardware specification, and publishes completion state. A wait queue helps consumers sleep until a predicate may hold. It does not protect the completed-data structure automatically.

```c
/* Reading fragment: queue_has_data() must use the driver's locking protocol. */
ret = wait_event_interruptible(dev->wait_q,
                              queue_has_data(dev) || device_is_gone(dev));
if (ret)
    return ret;  /* preserve -ERESTARTSYS; syscall restart policy handles it */
/* Lock, re-check, claim a completed item, then copy under a valid lifetime. */
```

A signal may end the wait. Another reader may claim the item first. A single Boolean loses information when two completions occur before a reader clears it; prefer a bounded queue or a counter when the hardware can complete multiple items. `READ_ONCE()` alone is not a complete producer/consumer ordering or mutual-exclusion protocol.

The `poll` callback first registers with `poll_wait()`, then reports current readiness under the same state protocol. Readiness is a hint to try an operation; another consumer can change it. Nonblocking reads must return `-EAGAIN` when no item can be claimed. Removal should wake waiters and report the terminal state.

Study the tested [SPSC lab](../labs/SPSC-Queue.md): its two ownership transfers explain why publishing data and reclaiming a slot require separate ordering. Kernel code uses kernel synchronization APIs and must account for multiple readers and interrupt context.

## V4L2 capture protocol

1. Query capabilities and select the correct single-planar or multi-planar buffer type.
2. Negotiate format with `VIDIOC_S_FMT`; use the returned dimensions, pixel format, plane layout, strides, and sizes.
3. Request a supported memory method. `V4L2_MEMORY_MMAP` asks the driver to allocate buffers; query and map the returned buffers. `V4L2_MEMORY_DMABUF` imports suitable externally allocated buffers; it does not allocate those external buffers for you.
4. Queue buffers with `VIDIOC_QBUF`, then start the selected stream type with `VIDIOC_STREAMON`.
5. Dequeue completed buffers. Blocking `VIDIOC_DQBUF` waits; with `O_NONBLOCK`, no completed buffer gives `EAGAIN`. Handle interrupted calls and driver errors.
6. Return a buffer only after **every consumer has finished using it**.
7. Stop capture and quiesce downstream consumers before destroying their mappings and allocations.

The [V4L2 streaming documentation](https://docs.kernel.org/userspace-api/media/v4l/mmap.html) describes the exact index/offset and plane-array contracts. The union field `m.userptr` is for USERPTR mode; it is not the universal address returned by a dequeue.

```text
Buffer 2: QUEUED_TO_CAMERA
       → DQBUF succeeds: APPLICATION_OWNS
       → supported GPU import + enqueue: GPU_IN_FLIGHT
       → completion event/fence proves last GPU reader finished
       → QBUF succeeds: QUEUED_TO_CAMERA
```

A CUDA kernel launch is asynchronous. Enqueueing work and immediately calling `QBUF` lets the camera overwrite memory that the GPU is still reading. A simple correct implementation waits for the relevant GPU completion before requeueing. A faster implementation tracks an event per buffer and requeues only completed buffers; it must also handle asynchronous errors and cancellation. V4L2 `STREAMOFF` alone does not complete unrelated GPU work.

A DMA-BUF fd is not a CUDA device pointer. Import support depends on the allocator, CUDA/JetPack release, format, planes, modifiers, and synchronization API. Bayer sensor capture through V4L2 also does not automatically mean an ISP produced NV12; Jetson Argus and raw capture have different paths.

## Worked ownership check

Assume four buffers, camera period 20 ms, and GPU reading time 27 ms. Returning each buffer immediately after enqueue can corrupt a frame even if average GPU utilization looks low. Retaining it until completion prevents that race. Whether four buffers avoid starvation additionally depends on camera queue requirements, concurrent work, and scheduling jitter. More buffers can improve throughput while increasing the age of the frame processed.

**Check:** A GPU submission fails after partially enqueueing work. May its buffer be freed? **Answer:** Not from the error code alone. Follow the runtime's completion/error recovery contract and prove that outstanding access has ended.

**Check:** Is `devm_kzalloc()` enough for state reachable from an old open fd after unbind? **Answer:** No. Managed release follows device teardown; the fd may still reference the allocation. Use an explicit lifetime design and reject operations after removal.
