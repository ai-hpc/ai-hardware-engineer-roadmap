# Build an ioctl latency tracer without attaching it

**Time:** 20–35 minutes. **Hardware:** Linux CPU host with kernel BTF and the listed tools; no GPU. **Result:** a real BPF ELF object, generated libbpf skeleton, and native reader that share one event ABI. Read [Lecture 26](../Lectures/Lecture-26.md) for the measurement contract.

This exercise stops at compilation and malformed-input checks. It does not load a program into the kernel, attach hooks, collect trace events, establish verifier acceptance, or measure tracing overhead.

## Predict the lifetime

Open [the shared header](ioctl_latency.h), [BPF program](ioctl_latency.bpf.c), and [reader](ioctl_latency.c). Sketch one thread entering an ioctl, storing its start state, exiting, deleting that state, and publishing an event. Then identify what happens if the map is full, the ring is full, the thread exits, or tracing stops while an ioctl is outstanding.

The key is a 64-bit host TGID/TID identity, so concurrent threads do not overwrite each other's timestamps. The event stores the command and fd from entry, rather than reading nonexistent exit arguments. Its entire storage is initialized before submission. The reader requires an explicit nonzero host TGID and validates arguments before any load attempt.

## Build checks

Required tools are a Clang with the BPF target, a C17 compiler, `bpftool`, `readelf`, libbpf development headers, libbpf/libelf/zlib runtime libraries, and readable `/sys/kernel/btf/vmlinux`. Distribution package names and permissions differ. On the recorded Ubuntu host, `libbpf-dev` and `libelf-dev` were downloaded and extracted into an isolated temporary directory; no system package installation was needed.

From the repository root, with development headers already installed:

```bash
python3 -m unittest discover -s 'Phase 1 - Foundational Knowledge/3. Operating Systems/labs' -p 'test_bpf_build.py' -v
```

For an isolated header location, set `COURSE_BPF_INCLUDE` to its `usr/include` directory before that command. This is a build configuration path, not a source path to a kernel image. The suite creates and removes its own temporary build directory. Missing prerequisites produce an explicit skip; a skip is not successful compilation.

The three checks compile with warnings treated as errors, generate a real skeleton with `bpftool`, inspect the ELF machine and required tracepoint/map/BTF sections, and execute seven malformed CLI cases. None passes a valid TGID to the reader, so none invokes the BPF loading path.

## Read the actual build steps

The test performs the following sequence using temporary paths and installed libraries:

1. `bpftool btf dump file /sys/kernel/btf/vmlinux format c` generates `vmlinux.h`.
2. Clang's `-target bpf -O2 -g` compiles the BPF source with that header and libbpf headers.
3. `bpftool gen skeleton` embeds the compiled object in a generated C interface.
4. The C17 compiler builds the userspace source against the shared header, skeleton, and libbpf/libelf/zlib.
5. `readelf` checks the actual object rather than accepting a source filename as evidence of BPF compilation.

A future attach experiment needs a supported target kernel, the tracepoints, appropriate permissions, and a controlled workload. Compile success cannot substitute for that step. If you choose that milestone, retain load/verifier logs and run counts; do not report an attached event until one has actually been observed.

## Recorded result and limits

On 2026-10-18, Ubuntu 24.04.4 x86-64, Clang 18, GCC 13.3, bpftool 7.4.0 (reporting embedded libbpf 1.4), and reader headers/runtime libbpf 1.3.0: all three build/argument checks passed. See [the validation record](bpf-validation.json) for exact command, tool output, source hashes, and execution boundary. The kernel BTF identity is recorded separately because it is an input to compilation.

**Answer:** a full ring increments a per-CPU counter and loses an event; it does not retry until the measured syscall is delayed indefinitely. A failed start insertion loses the ability to pair that call. An ioctl spanning detach has no terminal event in this experiment. Zero ring losses therefore do not prove complete coverage.

**Answer:** `elapsed_ns` includes time blocked and descheduled between the observed syscall entry and exit. It does not identify hardware execution time, application queue age, or a worst-case deadline bound.
