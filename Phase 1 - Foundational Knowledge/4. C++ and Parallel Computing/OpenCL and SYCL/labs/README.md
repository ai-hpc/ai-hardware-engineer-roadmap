# Run an OpenCL kernel on a CPU

This 30–45 minute lab turns `c[i] = a[i] + b[i]` into a checked host/device program. It is a real OpenCL execution exercise: a CPU implementation compiles the OpenCL C kernel and runs it. It does not measure GPU performance or validate a SYCL compiler.

You need C++ basics, array bounds, and a command-line compiler. Start with [numbers, shapes, and bytes](../../../Labs/Numbers-Shapes-and-Bytes.md) if array sizes or numerical contracts are unfamiliar.

## Setup and first run

Install a C++ compiler, OpenCL development headers/loader, and a device implementation. On Ubuntu, the relevant packages are `g++`, `ocl-icd-opencl-dev`, and `pocl-opencl-icd`. Package installation is separate from these compilation commands. PoCL is a portable OpenCL implementation with a CPU route; see [its documentation](https://portablecl.org/docs/html/) and [Khronos OpenCL documentation](https://registry.khronos.org/OpenCL/).

From the repository root:

```bash
cd 'Phase 1 - Foundational Knowledge/4. C++ and Parallel Computing/OpenCL and SYCL/labs'
g++ -std=c++17 -O2 -Wall -Wextra -Werror vector_add_opencl.cpp -lOpenCL -o /tmp/vector_add_opencl
/tmp/vector_add_opencl --self-test
python3 -m unittest discover -s . -v
```

The executable prints its actual device, driver, and OpenCL version. It prefers a CPU device, then falls back to another available device. An ICD loader by itself cannot execute a kernel; a missing platform usually means the device implementation is missing or its vendor registration is not visible.

**Executed 2026-10-18:** GCC 13.3, OpenCL loader 2.3.2, PoCL 5.0, Ubuntu 24.04 x86-64, on an AMD EPYC 9354P CPU. All ten lengths in the self-test passed: 0, 1, 7, 63, 64, 65, 255, 256, 257, and 1025. Three Python regression tests also passed, covering the kernel, an intentional build failure, and invalid arguments. These packages are a reproducible tested baseline, not a claim to be the latest upstream release.

## Trace one short array

Run `/tmp/vector_add_opencl 65`. On the tested device, the host chose 64 work-items per group and rounded the global range up to 128. Work-items 0–64 compute valid outputs; 65–127 do nothing because the kernel checks `i < n` before accessing an array. The host checks every result against scalar addition. Inputs are small integer-valued floats, so exact comparison is appropriate for this operation and fixture.

For zero elements, the program returns an empty result without allocating zero-sized device buffers or launching an empty range. For general floating-point reductions, exact equality is usually a different contract: additions can regroup, and your tolerance must follow numerical requirements.

Read [the source](vector_add_opencl.cpp) in this order:

1. `select_device`: distinguish a platform, device, and driver.
2. `main`: create the context and an in-order queue, compile the kernel, and create the kernel object.
3. `verify_length`: allocate buffers, write inputs, bind arguments, dispatch, read back, and compare with the reference.
4. `Handle`: release resources when their owning scope exits, including an exception path.

The OpenCL 1.2 API/kernel baseline is intentional. OpenCL 3.0 makes many later features optional, so a runtime's version string does not establish support for every feature a future example might require. Query the relevant capabilities before adding them.

## Make a failure understandable

```bash
/tmp/vector_add_opencl --inject-build-error
```

This command should exit nonzero and print `BUILD_LOG` with a compiler diagnostic. The runner deliberately passes malformed kernel source. A program-build failure is different from a missing device, a bad kernel argument, an invalid launch, or a wrong numerical result. Keep the first failing API call and its diagnostic together.

Try `/tmp/vector_add_opencl -1` and `/tmp/vector_add_opencl 12x`. They should fail before dispatch. An invalid input is not a workload to benchmark.

## Interpret the timing honestly

`kernel_event_ns` comes from the device command's start and end timestamps after completion. It excludes program compilation, allocations, input/output copies, and other host work. One tiny kernel can be dominated by startup and scheduling; its time is not a sustained throughput result. Compare complete application time separately, warm the intended paths, and report distributions over repeated runs.

**Check:** if a CPU OpenCL implementation is slower than a scalar loop on 65 values, is portability broken? **Answer:** no. The program may be correct while dispatch overhead dominates. Increase the work and investigate the crossover; do not assume an accelerator API makes every array faster.

**Transfer exercise:** add an elementwise clamp to `[0, 255]`, first as a second kernel, then fused into the addition. Keep the same scalar reference and edge cases. Compare intermediate allocation, bytes transferred, dispatch count, and end-to-end time. Record which costs disappear and which remain. This is a small version of the graph-fusion decisions in the AI compiler track.

Return to the [OpenCL/SYCL guide](../Guide.md) for local-memory tiling and explicit dependency models.
