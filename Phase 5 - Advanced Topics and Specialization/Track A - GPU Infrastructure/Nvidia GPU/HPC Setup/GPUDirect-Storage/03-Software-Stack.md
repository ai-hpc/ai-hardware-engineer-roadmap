# 03 — Establish a supported, observable storage path

Install GDS as a matched software stack, then demonstrate which path a real read
uses. A working CUDA program, a successful file read, and a direct storage-to-GPU
transfer are three different milestones.

**Prerequisites:** [the data-path lesson](01-Architecture-and-Data-Path.md), Linux
filesystems, and a declared test directory. **Time:** 30–60 minutes for the CPU
exercise; hardware setup depends on the platform. **Source check:** October 18,
2026. Hardware commands here are upstream-documented and were **not executed**
during this course revision; the CPU lab was executed with Python 3.12.

## Understand the components

```text
host application
    -> libcufile (library in the process)
    -> supported filesystem / driver / transfer mode
    -> storage and GPU buffer
```

`libcufile` is not a required standalone `cufile.service` daemon. The transfer mode
can involve `nvidia-fs`, PCI P2PDMA, or a compatibility path depending on the
selected release and platform. A CPU still performs setup, submission, and
completion handling. Do not interpret a missing kernel module as a universal
verdict without checking the applicable mode.

The [current release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
record mode and hardware-support changes. For example, v1.15 removed Pascal and
Volta support; an old “Volta or newer” rule is unsuitable for current installations.
The historical CUDA 12.2 configuration in earlier versions of this course is not
a current compatibility matrix.

## Make a compatibility record before installing

Use the [official installation and troubleshooting guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)
for the chosen GDS release and operating system. Record:

| Component | What to save | Why it matters |
|---|---|---|
| GPU and driver | Exact SKU, driver version, kernel-module flavor | Support varies by release and mode |
| CUDA and cuFile | Toolkit, library, headers, package versions | Do not confuse GDS, cuFile, and nvidia-fs versions |
| Operating system | Distribution, kernel, architecture | Kernel and module compatibility are explicit constraints |
| Storage | Filesystem, mount options, device and topology | A supported filesystem name alone does not prove this mount uses a direct path |
| Remote transport, if used | NIC, driver stack, firmware, RDMA/network configuration | Local NVMe and remote storage have different dependencies |
| Container, if used | Image digest, device access, mounted paths | Host support does not automatically propagate into a container |

These read-only Linux inventory commands are starting points, not an installation
recipe. `findmnt` and GPU/CUDA tools must exist on the selected system:

```bash
uname -r
nvidia-smi
nvcc --version
findmnt -T /path/to/your/test-directory
```

Replace the last path with an existing directory. Do not install historical OFED
packages or change mounts merely to reproduce a table from a different machine.
Choose packages from the selected release's supported matrix and keep the actual
installer/package log in your experiment record.

## Inspect capability and then verify a transfer

Use the `gdscheck` utility included with that installation. Tool filenames and
locations can vary; inspect the installed tools directory and help first:

```bash
ls /usr/local/cuda/gds/tools
```

Follow the installed `gdscheck` help and the vendor's
[configuration guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html).
Save its real output. Capability detection describes a possible path; runtime
statistics/logging for the specific workload provide evidence of the path used.
Do not look for invented universal fields such as `IsDirectStorageEnabled`.

Use the shipped `/etc/cufile.json` as the starting schema. Save a copy with your
results and record any per-process configuration selection supported by your
release. `properties.allow_compat_mode` controls whether unsupported direct paths
may fall back to POSIX I/O. Confirm behavior with the current documentation and
runtime evidence. Turning off fallback can help a controlled direct-path test fail
visibly, but it is not a performance optimization by itself.

## Start with a CPU correctness exercise

From the repository root:

```bash
python3 'Phase 5 - Advanced Topics and Specialization/Track A - GPU Infrastructure/Nvidia GPU/HPC Setup/GPUDirect-Storage/labs/io_contract.py'
```

The program creates a temporary 12,345-byte file, reconstructs it in chunks,
checks an exact byte match, rejects a truncated input, and exercises buffer
ownership. Expected final chunk: offset `12288`, length `57`. Both rejection flags
are `true`, and the final buffer state is `free`.

This is real CPU file I/O plus a small ownership model. It does not load cuFile,
measure a disk's bandwidth, or establish GDS availability. Read the implementation
before porting its invariants to a GPU path.

## A useful failure diagnosis

| Observation | Next discriminating check |
|---|---|
| File cannot be opened | Path, permissions, mount, and descriptor flags |
| Registration fails | API error, exact package compatibility, filesystem and transfer mode |
| Read succeeds but direct-path counters do not change | Fallback selection, actual mount/device, and process configuration |
| Bytes differ | Return count, file/buffer offsets, buffer lifetime, dtype/layout interpretation |
| Initialization works but a worker fails | Process creation method and where CUDA/cuFile initialize; do not fork after cuFile initialization |
| Good microbenchmark but no training improvement | CPU decode/augmentation, cache effects, queueing, and compute dependencies |

**Self-check:** a compatibility-mode read succeeds and matches the reference.
What have you proved? You have verified that read's payload correctness through
that path. You have not proved direct DMA, support for another mount, or a speedup.

**Exit artifact:** compatibility record, source versions, capability output, one
integrity result, and evidence labeling the observed path. Continue to
[the API and ownership lesson](04-libcufile-API.md).
