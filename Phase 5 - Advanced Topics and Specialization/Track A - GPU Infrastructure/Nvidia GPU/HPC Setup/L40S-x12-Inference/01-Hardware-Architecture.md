# 01 — L40S hardware and the PCIe path

**Outcome:** build a device/topology/memory worksheet for a named L40S server. Specification review: 2026-10-18 against [NVIDIA's L40S table](https://www.nvidia.com/en-us/data-center/l40s/). No GPU measurements are claimed.

## Read the device specification

| Property | L40S |
|---|---|
| Architecture | Ada Lovelace, AD102 family |
| CUDA cores / Tensor Cores | 18,176 / 568 fourth-generation Tensor Cores |
| Memory | 48 GB GDDR6 with ECC |
| Advertised memory bandwidth | 864 GB/s |
| FP8 Tensor Core peak | 733 TFLOP/s dense; 1,466 structured-sparse |
| BF16/FP16 Tensor Core peak | approximately 362–366 TFLOP/s dense; 733 structured-sparse; see discrepancy below |
| TF32 Tensor Core peak | 183 TFLOP/s dense; 366 structured-sparse |
| FP32 CUDA core peak | 91.6 TFLOP/s |
| Host/peer interface | PCIe Gen4 x16, 64 GB/s aggregate bidirectional theoretical rate |
| NVLink / MIG | Neither supported |
| Video engines | 3 NVENC and 3 NVDEC; includes AV1 encode/decode support |
| Board maximum power | 350 W; passive server heatsink |

NVIDIA's current expanded web table lists 362.05 dense BF16/FP16 alongside 733 sparse, while half the rounded sparse number is 366.5. Preserve the exact source/SKU and flag this discrepancy instead of asserting false precision. The larger sparse rate requires a supported structured-sparse workload and kernel. It is not the dense baseline.

A card's architecture family does not establish that every consumer/professional/datacenter SKU has identical firmware, clocks, media capability or software support. Use the named product's table, not assumptions from another AD102 card.

## GDDR and HBM are memory technologies, not interconnect policies

L40S uses GDDR6; H200 SXM uses HBM3e. Their published local-memory ceilings differ (`4.8 TB/s / 0.864 TB/s ≈ 5.56`), but this ratio is not a decode-speed prediction. Batch size, reuse, kernels, KV traffic and launch overhead change the bottleneck.

GDDR does not logically forbid NVLink, and HBM does not place an NVSwitch inside the GPU package. L40S happens to lack NVLink; an HGX H200 system has a separate switched GPU fabric. Draw local memory, GPU links, PCIe switches and CPUs as distinct components.

## Capacity must include everything live on the rank

Twelve 48 GB cards total 576 GB nominal memory, distributed across devices. For BF16 raw weights, 7B/13B/34B/70B parameters cost approximately 14/26/68/140 GB. Subtracting this from a device label does not leave all the remainder available for cache: unquantized/replicated tensors, activations, runtime, workspace and reserve remain.

Three cards have 144 GB nominal total, but a 70B BF16 deployment may not support TP=3, and 140 GB of weights leaves almost no room before overhead. Four-way partitioning is an arithmetic candidate, not a guarantee. Check head/matrix divisibility and the actual runtime's parallel support. A 180B four-bit artifact's 90 GB raw weights similarly says nothing by itself about its viable concurrency.

## Inspect topology instead of assuming adjacent IDs are adjacent devices

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total --format=csv
nvidia-smi topo -m
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
```

In topology output, `PIX` means at most one PCIe bridge, `PXB` multiple bridges without a host bridge, `PHB` a host bridge, `NODE` a connection between host bridges within a NUMA node, and `SYS` a path through the inter-NUMA CPU interconnect. These are connectivity classes, not measured latency rankings. PXB is not “like NVLink.” Capture the legend printed by the installed tool. [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

A plausible server can fan several GPUs out from each PCIe switch; GPUs do not form a PCIe daisy-chain through one another. Draw the actual upstream links and their shared bandwidth. Peer access may be unsupported or blocked by topology/virtualization/configuration even when two GPUs are visible.

## A bounded one-way transfer experiment

This optional example requires two allocated CUDA GPUs. It was syntax-reviewed, not hardware-executed. The payload is **64 MiB**, and both devices are synchronized around the timed interval. A successful copy does not alone prove direct P2P; record capability and transport evidence.

```python
import time
import torch

if torch.cuda.device_count() < 2:
    raise SystemExit("Two allocated CUDA GPUs are required")
source = torch.full((64 * 1024**2 // 2,), 1.25, dtype=torch.float16, device="cuda:0")
destination = torch.empty_like(source, device="cuda:1")
print("peer capability:", torch.cuda.can_device_access_peer(0, 1))
for _ in range(5): destination.copy_(source)
for device in (0, 1): torch.cuda.synchronize(device)
start = time.perf_counter()
for _ in range(30): destination.copy_(source)
for device in (0, 1): torch.cuda.synchronize(device)
seconds = time.perf_counter() - start
assert torch.all(destination == 1.25).item()
payload = source.numel() * source.element_size()
print("one-way payload GB/s:", payload * 30 / seconds / 1e9)
```

The one-direction theoretical Gen4 x16 budget is about 32 GB/s, before overhead and shared links. Compare like direction/scope with H200's approximately 450 GB/s one-direction **aggregate per GPU** ceiling; neither number is a guaranteed collective result. A low copy result needs a hypothesis and trace, not an automatic “bad GPU” verdict.

## Precision and server integration

Ada supports FP8 arithmetic, and current Transformer Engine supports eligible Ada FP8 training recipes. Scaling/quantization is a software and numerical contract; the claim that L40S must use only offline AWQ/GPTQ because FP8 scaling is unavailable is incorrect. Use a supported recipe, check kernels and compare quality. [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).

A passive 350 W card does not work safely in every server with a PCIe slot. Confirm the certified chassis, airflow, mechanical clearance, power connector/cabling, PSU redundancy and service procedures. Twelve GPU power ceilings sum to 4.2 kW before CPUs, memory, NICs, drives and cooling. Do not infer rack units, total power or a universal 200 CFM requirement from that sum.

**Exit exercise:** draw two candidate four-GPU groups, label their shared upstream PCIe links, calculate raw weight bytes for a chosen checkpoint and identify every remaining capacity term. Submit actual device/topology evidence when hardware is available; otherwise label the worksheet a planning exercise.
