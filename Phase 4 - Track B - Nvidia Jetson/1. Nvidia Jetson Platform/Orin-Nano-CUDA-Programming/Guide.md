# CUDA on Orin Nano: correct results before faster results

**Prerequisites:** C++ arrays and pointers, basic linear algebra, and the [memory lesson](../Orin-Nano-Memory-Architecture/Guide.md). **Pace:** six sessions of 60–90 minutes. **Outcome:** explain a kernel's indexing, validate every output, describe buffer ownership, and measure a complete perception stage.

CUDA runs many instances of a function over data. A useful first robotics example is converting camera pixels into a model's input tensor. The arithmetic is simple; the difficult parts are image layout, bounds, ordering, and knowing when another component may reuse the buffer.

Start with the CPU exercises linked below if you do not have a Jetson. The supplied advanced CUDA harness has been compiled, but this course review had no GPU. CPU references, successful compilation, and hardware execution are recorded separately.

## 1. Establish the actual device and software

Orin Nano 8 GB has an Ampere GPU with eight SMs, 1,024 CUDA cores, 32 Tensor Cores, and compute capability 8.7. It has no DLA. CPU, GPU and multimedia engines share physical LPDDR memory; they do not automatically share every allocation, cache policy or synchronization contract. See the [platform guide](../Guide.md) and [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/).

Read the [compatibility matrix](../../Compatibility.md) before selecting packages. JetPack 6 and JetPack 7 releases have different OS, kernel and CUDA combinations. A compiler version available on a workstation is not evidence that it is a supported replacement for the toolkit on your target BSP.

Record these read-only observations on the development system:

```bash
uname -m
uname -r
nvcc --version
cat /etc/nv_tegra_release
```

The last file is a Jetson BSP observation and may be absent on a workstation. Query `cudaGetDeviceProperties` in the application for SM count, compute capability, L2 size, shared-memory limits and other runtime capabilities. Check its return status. Obtain current clocks and mode separately; a device property's nominal limit is not an observed clock during your workload.

Do not reuse an old “512 KB L2” or “625 MHz GPU” constant as a current measurement. Clock limits and available power modes depend on module, carrier configuration and BSP. The [power lesson](../Orin-Nano-Power-Thermal/Guide.md) explains what to record.

## 2. Map one element to one thread

A grid contains blocks; a block contains threads. Threads execute in warps of 32. Blocks can run in any order, so ordinary kernels cannot assume block 0 has completed before block 1 starts.

This is a **kernel fragment**, not a complete executable:

```cuda
__global__ void normalize(const float* input, float* output,
                          int n, float mean, float inverse_std) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = (input[i] - mean) * inverse_std;
}
```

For 1,003 elements and 256 threads per block, launch four blocks: `(1003 + 255) / 256 = 4`. There are 1,024 launched threads; 21 must do no work. Without the bound check, the last block writes beyond the allocation. For zero elements, skip the launch: a grid dimension of zero is not a useful empty operation.

**Work it out:** block 2, thread 17 handles element `2 × 256 + 17 = 529`. With input 0.75, mean 0.5 and standard deviation 0.25, the output is 1.0. The caller validates that the standard deviation is finite and positive before calculating its reciprocal.

Use the complete [CUDA correctness lab](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/labs/README.md) for executable code. Begin with its CPU references, then trace the fused affine/ReLU kernel and its host allocation/error handling. The harness covers lengths 0, 1, 31, 32, 33, 255, 256, 257 and 1,003, including sizes near warp and block boundaries.

The lab's recorded compiler target is `sm_80`. For an Orin build, select `-arch=sm_87` with a compatible compiler. The source uses capabilities available on Orin, but that alone does not establish target numerical or timing results. Record your own build and execution outcome.

## 3. Treat memory as owned data with a lifetime

The important question is: **who may read or write this allocation now?**

| Allocation/interface | What to establish |
| --- | --- |
| Device allocation | Which GPU operations own it; where host input/output copies occur |
| Managed allocation | Runtime/device access capabilities and synchronization before CPU access |
| Mapped pinned host allocation | Mapping support, device alias, CPU/GPU caching and producer/consumer ordering |
| Camera DMA-BUF/NvBufSurface | Plane format, pitch, layout, import API, fences and producer return/reuse contract |

Shared physical DRAM does not make a `cudaMemcpy` between two distinct allocations disappear. Such a copy still reads and writes memory. Likewise, pinned memory does not permit two processors to modify the same bytes without coordination. Orin's I/O coherency and newer full CPU/GPU cache-coherent platforms are different cases; use the matching Tegra documentation.

Consider two camera slots. CPU preparation of slot 0 must wait until the previous GPU consumer of slot 0 is finished. Merely alternating `slot = frame_id % 2` is insufficient. Keep a completion event or an equivalent documented ownership signal for each slot. Check both “not ready” and actual error results when polling events; an error is not another reason to wait forever.

A useful ownership sequence is:

```text
capture owns slot → frame ready → preprocess/inference owns slot
→ completion observed → downstream result copied/consumed → capture may reuse slot
```

The CPU can prepare a different free slot while the GPU works. Overlap is possible only when dependencies, available engines, memory type and resource use permit it. Two streams express two ordered queues; they do not guarantee simultaneous execution.

## 4. Count bytes and check layout before tuning

A contiguous 640 × 480 RGB8 image has `640 × 480 × 3 = 921,600` payload bytes. An FP32 RGB tensor of the same dimensions uses 3,686,400 bytes. One ideal conversion reads the former and writes the latter: 4,608,000 bytes, before cache effects and extra operations. At 30 frames/s, that is 138.24 MB/s of requested array traffic. It is not the total camera/inference DRAM traffic.

Real image rows may have padding. The [video lab](../Orin-Nano-Video-Codec-DeepStream/labs/README.md) demonstrates a 65-pixel RGB row with 195 payload bytes but a 196-byte GStreamer stride. Indexing the second row as `row * width * 3` would begin one byte early. Block-linear surfaces require their documented access/import path; they are not ordinary pitched arrays.

Before fusing resize, color conversion, normalization and HWC→CHW, specify:

- RGB versus BGR, source bit depth and numeric range;
- plane layout, pitch and valid crop;
- resize mapping, interpolation and border policy;
- per-channel mean/standard deviation and destination layout;
- exact shape, dtype and tensor address accepted by inference.

A single kernel with undefined `pixel`, `mean` or `std_dev` variables is a sketch, not a working preprocessing implementation. Validate the transform on a tiny image whose pixel values you can calculate by hand, including odd dimensions and a nonzero crop origin.

## 5. Understand coalescing and shared memory

For the current CUDA coalescing model on compute capability 6.0 and later, count the 32-byte sectors needed by a warp's access. Thirty-two aligned adjacent FP32 elements occupy 128 bytes and require four sectors. Shifting the starting address by one FP32 element spans five sectors. A stride of eight FP32 elements places each lane's value in a different sector, using 128 requested bytes across 1,024 bytes of sectors. Cache reuse and the rest of the kernel influence measured time; these counts are not universal speedup ratios. [CUDA memory-access guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).

Shared memory lets threads in a block reuse values. It also consumes a limited SM resource. The compute-capability 8.7 limits include up to 164 KiB of shared memory per SM and 163 KiB per block with the relevant opt-in, while allocations above the default 48 KiB block limit require the documented dynamic-memory configuration. Query device and function limits, and inspect the compiler's resource report. [CUDA capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

**Worked stencil:** a 16 × 16 output tile for a 3 × 3 filter needs an 18 × 18 input tile including its halo: 324 values rather than 2,304 independent input references in a naive nine-load-per-output count. Actual memory traffic may differ because the naive version can reuse cached data. All halo edges **and corners** must be initialized. Omitting “top/bottom halo, similar” creates invalid results, not an optimization exercise.

All participating block threads must reach a block barrier consistently. Place the barrier after cooperative loading, with out-of-image loads replaced by the selected border value; do not return some threads early while others execute `__syncthreads()`.

Occupancy is the fraction of supported resident warps that are active. Registers, shared memory, block dimensions and architectural limits constrain it. Higher occupancy can help hide latency, but “over 50%” is not a universal success threshold. A change that increases occupancy while spilling registers may run slower. Use occupancy APIs to understand constraints, then compare measured workloads.

## 6. Separate numerical correctness from Tensor Core availability

Start matrix multiplication with a trusted CPU or library reference and state the accumulation precision and error tolerance. Tensor Cores accelerate supported matrix operations; they do not accelerate every FP16 kernel automatically.

For `C[M,N] = A[M,K] × B[K,N]`, define row/column layout and leading dimensions before using cuBLAS or a matrix fragment API. For a row-major B, `B[k*N+n]` is appropriate; declaring that same storage column-major changes its interpretation. Handle dimensions that are not tile multiples, alignment requirements and valid participation of a complete warp.

Use a supported library implementation first. Custom WMMA/MMA work is an advanced step when a specific workload justifies it. Avoid claiming that a particular compute type always selects Tensor Cores, or mixing sparse INT8 TOPS with dense FP32 FLOPs when calculating a roofline.

**Amdahl example:** if preprocessing takes 4 ms and the rest takes 16 ms, making preprocessing four times faster gives `1 + 16 = 17 ms`, or about **1.18×** overall speedup. It does not make the entire 20 ms pipeline four times faster. Measure where the time actually goes before writing a more complex kernel.

## 7. Add streams and graphs only after validating the baseline

Operations in one stream are ordered. Cross-stream dependencies need events or another documented synchronization mechanism. Record an event after the producer's last write; make the consumer stream wait before its first read. Retain allocations until all asynchronous consumers complete.

CUDA graphs can reduce repeated launch overhead when a stable workload is replayed. Stable addresses do not imply stable input contents; new data can be written into the same buffers with correct ordering. Changes to shapes, pointers or dynamic state must follow the graph update/recapture contract. A replay that silently retains the old cache position is incorrect even if it launches quickly.

The [graphs lesson](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/01-CUDA-Graphs.md) and complete harness exercise changing inputs. Study those before adding graph capture to a camera or language-model pipeline.

TensorRT integration also has versioned contracts. For the course's TensorRT 10 path, use named I/O tensors, explicit shapes/addresses and `execute_async_v3` as described in the [inference lesson](../Orin-Nano-Real-Time-Inference/Guide.md). An old incomplete V2 plugin class is not a complete current plugin. Use the extension API matching the installed release, and pass the runtime's supplied stream through custom work.

## 8. Measure a meaningful interval

Use CUDA events around work in the same stream for a device execution interval, and a monotonic host clock for a caller-visible interval. Synchronize the stop event before reading event elapsed time. Check allocation, copy, launch, event and synchronization results; launch errors and asynchronous execution errors can appear at different calls.

Measure warm-up separately. Report input sizes, numerical mode, power mode, temperature, clocks, build flags and number of samples. For a robot, record capture-to-action age as well as kernel duration. A 2 ms kernel operating on a frame that spent 100 ms queued is still using old information.

On a system with a compatible profiler, a starting command is:

```bash
nsys profile --trace=cuda,nvtx,osrt -o course-cuda-trace ./my_app
```

Here `my_app` is your built, validated program; the command is a template. Check the installed tool's help and choose a new output name. Start with the timeline: CPU gaps, dependencies, copies and serialization. Then use Nsight Compute for a specific kernel hypothesis. Profiling can perturb execution; do not treat a profiler-heavy run as the only latency measurement.

Use Compute Sanitizer's appropriate checks on the hardware execution path, and preserve errors in the result. The [complete lab](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20A%20-%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/labs/README.md) fills output buffers with NaNs before each implementation so a missing write cannot inherit a previous correct result. Sanitizer success and numerical comparisons cover different failure classes.

## 9. Practice, answers and completion criteria

1. **Why test 257 elements?** It needs two 256-thread blocks and exposes an incomplete final block.
2. **Why can two streams race even though each stream is ordered?** Ordering within one stream does not order the other stream's accesses to shared storage.
3. **Why can eliminating a copy make a stage slower?** The alternative may have worse caching, access patterns or synchronization costs; measure the complete stage.
4. **Why reject NaNs explicitly in a comparison?** A check such as `abs(actual-reference) > tolerance` can be false when the difference is NaN, accidentally accepting an unwritten value.
5. **Why does a 3×3 shared tile need corners?** Outputs near a tile corner read diagonal neighbors as well as horizontal/vertical ones.
6. **Why is GPU utilization alone inadequate?** It does not establish correct outputs, useful throughput, frame freshness, tail latency or energy per accepted result.

Submit a CPU reference result, build command, complete output-comparison policy, memory/ownership diagram and measured stage report. If you lack a GPU, submit the CPU and compile evidence with hardware execution explicitly pending. On hardware, add error-checked numerical runs before timing, then one optimization justified by a measured bottleneck. Continue to [real-time inference](../Orin-Nano-Real-Time-Inference/Guide.md) and the [robotics deployment path](../../../Learning-Guide.md).
