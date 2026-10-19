# 01 — H200 hardware: capacity, compute and paths

**Outcome:** read a specification without confusing a chip, a module and a server. Source review: 2026-10-18. Numbers below describe advertised ceilings; no H200 hardware was tested for this course update.

## Name the SKU and arithmetic mode

H200 uses the Hopper GH100 architecture with HBM3e. H200 SXM and H200 NVL have different power and compute envelopes. This course's eight-GPU case is an **HGX/DGX-style H200 SXM platform**, not eight interchangeable PCIe H200 NVL cards.

| H200 SXM specification | Published ceiling / interpretation |
|---|---|
| GPU memory | 141 GB HBM3e per GPU; 1,128 GB across eight, not one allocation |
| Local memory bandwidth | 4.8 TB/s per GPU |
| BF16/FP16 Tensor Core | approximately 989.5 TFLOP/s dense; 1,979 with supported structured sparsity |
| FP8 Tensor Core | approximately 1,979 TFLOP/s dense; 3,958 with supported structured sparsity |
| TF32 Tensor Core | approximately 494.5 TFLOP/s dense; 989 with supported structured sparsity |
| FP64 | 34 TFLOP/s CUDA cores; 67 TFLOP/s Tensor Core |
| NVLink | 900 GB/s aggregate bidirectional per GPU |
| PCIe | Gen5 x16, 128 GB/s aggregate bidirectional theoretical interface rate |
| GPU maximum power | up to 700 W; server total is larger |

FMA counts as two operations. Structured-sparse peak requires a supported sparsity pattern and kernel; dense BF16 GEMM cannot be compared to the sparse headline. GB is decimal; allocation tools often report MiB/GiB. [Official H200 specifications](https://www.nvidia.com/en-us/data-center/h200/).

**Worked check:** eight GPUs' FP64 Tensor Core ceilings sum to `8×67=536 TFLOP/s = 0.536 PFLOP/s`. A claim of 50 PFLOP/s FP64 HPL for this server fails a basic ceiling check before benchmarking.

## Memory is local to each GPU

The H200 SXM specification offers more capacity and bandwidth than H100 SXM 80 GB (3.35 TB/s). `4.8/3.35≈1.43` is a bandwidth-ceiling ratio, not a universal 43% application speedup. Kernels can be compute-bound, launch-bound or limited by access patterns.

A model must fit **on the most heavily loaded rank**, including weights, KV cache or training activations, optimizer states, communication buffers and workspace. “128K tokens fits” needs a model's layer/KV-head dimensions, dtype, batch size and sharding plan; HBM capacity alone does not determine context length.

BF16 is often useful, but do not indiscriminately convert every parameter/state to BF16 and call training numerically safe. Use the framework's mixed-precision policy and validate loss/gradients; some reductions and optimizer state need higher precision. Pinned host memory can enable efficient asynchronous transfers when paired with the correct copy/stream dependencies, but it is not itself a measurement of HBM bandwidth.

## NVSwitch connects peers; it does not multiply each GPU's budget

Hopper's fourth-generation NVLink provides 18 links at up to 50 GB/s bidirectional per link. HGX H200 systems use NVSwitch to route GPU traffic. The 900 GB/s per-GPU aggregate is shared across those links; it is not a separate 900 GB/s to every peer. Traffic traverses a switch, and latency depends on size, path and protocol.

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status -i 0
```

Record every allocated GPU and actual topology. `nvidia-smi dmon -s u` reports utilization, not NVLink byte traffic. Use supported NVLink/DCGM profiling counters for your hardware to observe link traffic. Then run a correct nccl-tests sweep and interpret its normalized bus bandwidth using the [NCCL lesson](../NCCL-Deep-Dive/02-Algorithms-and-Bandwidth.md). A “typical 1 µs AllReduce” is not a hardware invariant.

## Host connection and platform boundaries

An HGX H200 server's CPUs connect through its actual PCIe platform. **Grace Hopper (GH200)** is a separate superchip/platform using coherent NVLink-C2C between Grace CPU memory and Hopper GPU memory. Do not draw a C2C bridge into every H200 SXM server. Coherent addressing does not make remote LPDDR and local HBM equally fast. [Grace Hopper platform](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/).

Cooling, inlet temperature, PSU redundancy and total node/rack power come from the actual certified server specification. Eight 700 W GPUs alone total 5.6 kW; CPUs, switches, NICs, drives and cooling overhead add to that. A generic 7.5 kW PDU or 45°C coolant prescription is not a validated design for every server.

## Transformer Engine is a software policy too

A `transformer_engine.pytorch.Linear` layer alone does not automatically turn arbitrary execution into validated FP8 training. Use the installed version's supported autocast/quantization context and recipe; record scaling, supported shapes, accumulation and state. Current APIs and historical `fp8_autocast` recipes may differ. Compare with BF16 on the same training/evaluation data before adopting the lower precision. [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/index.html).

NVSwitch reduction capabilities can accelerate eligible collectives when supported by the runtime/topology. Hardware capability does not imply every operation uses in-switch reduction. Inspect actual algorithm selection and measure.

**Assessment:** compute the dense BF16 and FP64 node ceilings, distinguish 1,128 GB total capacity from per-rank capacity, and draw the local HBM, switched GPU and CPU paths for the actual server you intend to use.
