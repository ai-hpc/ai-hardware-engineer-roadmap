# 05 — GDS Performance Tuning

## 1. The GDS Performance Equation

```
GDS Throughput = min(NVMe_BW, PCIe_BW, NIC_BW, GPU_HBM_BW)

For the reference system:
  NVMe (local, 8 drives × 7 GB/s):     56 GB/s
  PCIe Gen4 (per x16 slot):            32 GB/s
  NIC (3 × ConnectX-7 @ 25 GB/s):     75 GB/s  (NVMe-oF path)
  GPU HBM (A100):                    2000 GB/s  (never the bottleneck for I/O)
  OpenFlex storage side:               75 GB/s  (6 × 100 Gb/s)

Bottleneck for local NVMe:   PCIe bandwidth per GPU (32 GB/s)
Bottleneck for NVMe-oF:      OpenFlex storage output (75 GB/s for all GPUs)
Per-GPU NVMe-oF practical:   ~25 GB/s (75/3 with 3 NICs per NUMA node)
```

---

## 2. Alignment Tuning

Alignment is the single most impactful low-level tuning parameter:

```
GDS minimum alignment:    512 bytes (sector size)
Optimal alignment:        4096 bytes (filesystem block size)
Large transfer alignment: 1 MB (maximizes DMA efficiency)

Penalty for misalignment:
  512-byte aligned read of 1 GB:   6.8 GB/s
  Non-aligned read of 1 GB:        GDS FALLS BACK to compat mode
                                   → 2.1 GB/s (CPU-mediated)
  → 3.2× performance drop for alignment violations
```

```cpp
// Always allocate GPU buffers with 512-byte alignment
void* allocate_aligned_gpu_buffer(size_t size) {
    // Round size up to 512-byte boundary
    size_t aligned_size = (size + 511) & ~511ULL;

    // Standard cudaMalloc guarantees 256-byte alignment on most GPUs
    // For 512-byte guarantee, use cuMemAlloc:
    CUdeviceptr d_ptr;
    CUresult result = cuMemAlloc(&d_ptr, aligned_size);
    assert(result == CUDA_SUCCESS);
    assert((d_ptr & 511) == 0);   // verify alignment
    return (void*)d_ptr;
}

// For numpy/Python interop, use 4096-byte alignment:
import numpy as np
arr = np.empty(N, dtype=np.float32)
arr = np.require(arr, requirements=['C_CONTIGUOUS', 'ALIGNED'])
# Or:
arr = np.empty(N + 4096//4, dtype=np.float32)
# Manually align: offset = (-arr.ctypes.data % 4096) // 4
```

---

## 3. Transfer Size Optimization

```
GDS throughput vs transfer size (local NVMe, reference system):

  4 KB:     0.8 GB/s   ← tiny transfers, DMA setup overhead dominates
  64 KB:    2.1 GB/s
  256 KB:   4.5 GB/s
  1 MB:     6.0 GB/s
  4 MB:     6.6 GB/s
  16 MB:    6.8 GB/s  ← near peak (PCIe Gen4 x4)
  64 MB:    6.8 GB/s  ← peak (no further gain)

Recommendation: use ≥ 1 MB per GDS transfer
                For large files: split into 4-16 MB chunks for batch I/O
```

```python
OPTIMAL_CHUNK_SIZE = 4 * 1024 * 1024   # 4 MB

def chunked_gds_read(filepath: str, total_size: int, gpu_buf) -> None:
    with cufile.open(filepath, "r") as f:
        offset = 0
        while offset < total_size:
            chunk = min(OPTIMAL_CHUNK_SIZE, total_size - offset)
            f.read(gpu_buf, size=chunk, file_offset=offset, buf_offset=offset)
            offset += chunk
```

---

## 4. Queue Depth and Parallelism

NVMe drives have internal queue depth — sending multiple requests simultaneously keeps the drive busy:

```
NVMe queue depth: up to 64K per namespace, typically use 128
GDS optimal queue depth: 32-128 concurrent operations

Without queue depth (sequential):
  [Submit read 0][Wait complete][Submit read 1][Wait complete]...
  Drive utilization: ~40% (idle between submissions)
  Throughput: 3.5 GB/s

With queue depth 32 (batch I/O):
  [Submit 32 reads simultaneously]
  Drive utilization: ~90%
  Throughput: 6.5 GB/s
```

```json
// /etc/cufile.json — tune queue depth
{
  "execution": {
    "max_io_queue_depth": 128,     // max concurrent GDS ops per file handle
    "num_io_threads": 8,           // I/O worker threads in daemon
    "max_batch_io_timeout_msecs": 5
  }
}
```

```cpp
// Set queue depth per file handle
CUfileDescr_t cf_descr = {};
cf_descr.handle.fd = fd;
cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
// Queue depth is managed by the cuFile daemon based on cufile.json
```

---

## 5. Multi-Stream I/O Overlap

Use separate CUDA streams to pipeline I/O with compute:

```python
import torch
import cufile

def train_with_gds_overlap(model, data_dir, num_steps, device="cuda:0"):
    """
    Double-buffered training:
    - While GPU processes batch N (compute stream)
    - GDS loads batch N+1 (io stream)
    Both run in parallel → I/O latency hidden by compute
    """
    files = sorted(glob.glob(f"{data_dir}/*.bin"))

    # Two GPU buffers: ping-pong
    buffers = [
        torch.empty(BATCH_SHAPE, dtype=torch.bfloat16, device=device),
        torch.empty(BATCH_SHAPE, dtype=torch.bfloat16, device=device),
    ]

    # Pre-load first file
    with cufile.open(files[0], "r") as f:
        f.read(buffers[0])

    compute_stream = torch.cuda.Stream(device=device)
    # cufile reads are managed by GDS daemon (effectively async to GPU compute)

    for step in range(num_steps):
        cur = step % 2
        nxt = (step + 1) % 2

        # Launch compute on current buffer (non-blocking)
        with torch.cuda.stream(compute_stream):
            loss = model(buffers[cur])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Start loading NEXT batch while compute runs
        if step + 1 < num_steps:
            # GDS read (runs in background via daemon)
            with cufile.open(files[(step + 1) % len(files)], "r") as f:
                f.read(buffers[nxt])

        # Wait for compute to finish before next iteration
        torch.cuda.current_stream().wait_stream(compute_stream)

        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
```

---

## 6. Buffer Registration Caching

`cuFileBufRegister()` is expensive (~1 ms). For repeated reads, register once:

```cpp
class GDSBufferPool {
    struct RegisteredBuffer {
        void*  ptr;
        size_t size;
        bool   in_use;
    };

    std::vector<RegisteredBuffer> pool;
    size_t buffer_size;

public:
    GDSBufferPool(int num_buffers, size_t buf_size) : buffer_size(buf_size) {
        for (int i = 0; i < num_buffers; i++) {
            void* ptr;
            cudaMalloc(&ptr, buf_size);
            cuFileBufRegister(ptr, buf_size, 0);   // register once at startup
            pool.push_back({ptr, buf_size, false});
        }
    }

    void* acquire() {
        for (auto& buf : pool) {
            if (!buf.in_use) { buf.in_use = true; return buf.ptr; }
        }
        return nullptr;  // pool exhausted
    }

    void release(void* ptr) {
        for (auto& buf : pool) {
            if (buf.ptr == ptr) { buf.in_use = false; return; }
        }
    }

    ~GDSBufferPool() {
        for (auto& buf : pool) {
            cuFileBufDeregister(buf.ptr);
            cudaFree(buf.ptr);
        }
    }
};

// One-time setup at process start
GDSBufferPool pool(16, 64 * 1024 * 1024);  // 16 × 64 MB buffers, pre-registered

// In training loop: acquire → read → process → release (no registration cost)
void* buf = pool.acquire();
cuFileRead(fh, buf, size, offset, 0);
processKernel<<<grid, block>>>(buf);
pool.release(buf);
```

---

## 7. Benchmarking GDS Performance

### Using gds_bandwidth (Built-in)

```bash
# Sequential read benchmark
/usr/local/cuda/gds/tools/gds_bandwidth \
    --file=/mnt/nvme0/test_file.bin \
    --size=4096M \                        # 4 GB test file
    --gpu_id=0 \
    --num_threads=1 \
    --pattern=sequential

# Random read benchmark (important for datasets with shuffling)
/usr/local/cuda/gds/tools/gds_bandwidth \
    --file=/mnt/nvme0/test_file.bin \
    --size=4096M \
    --gpu_id=0 \
    --pattern=random \
    --block_size=65536                    # 64 KB random blocks

# Expected sequential read (reference system, GDS active):
# GDS Read:    6.5–6.8 GB/s  (local NVMe, PCIe Gen4 x4)
# Compat Read: 1.8–2.1 GB/s  (CPU path)
# GDS 3.2× faster

# Expected NVMe-oF read (via ConnectX-7, OpenFlex):
# GDS RDMA:   22–25 GB/s  (per GPU, limited by NIC)
```

### Custom Benchmark

```python
import time
import torch
import cufile

def benchmark_gds(filepath: str, size_mb: int, gpu_id: int, n_iter: int = 20):
    size = size_mb * 1024 * 1024
    buf = torch.empty(size // 4, dtype=torch.float32, device=f"cuda:{gpu_id}")

    # Warmup
    with cufile.open(filepath, "r") as f:
        for _ in range(3):
            f.read(buf)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with cufile.open(filepath, "r") as f:
        for _ in range(n_iter):
            f.read(buf)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    bw = size * n_iter / elapsed / 1e9
    print(f"GDS Read Bandwidth: {bw:.2f} GB/s ({size_mb} MB × {n_iter} iterations)")
    return bw

benchmark_gds("/mnt/nvme0/test.bin", size_mb=1024, gpu_id=0)
```

---

## 8. Performance Targets (Reference System)

| Configuration | Path | Expected Throughput | Bottleneck |
|---|---|---|---|
| 1 GPU, 1 local NVMe (x4) | GDS direct | 6.5–6.8 GB/s | PCIe Gen4 x4 |
| 1 GPU, 4 local NVMe (x4) striped | GDS direct | 20–25 GB/s | PCIe switch bandwidth |
| 1 GPU, NVMe-oF via CX-7 | GDS RDMA | 22–25 GB/s | NIC 25 GB/s |
| 4 GPUs, OpenFlex 75 GB/s | GDS RDMA | ~18 GB/s/GPU | OpenFlex 75/4 |
| 1 GPU, CPU path (no GDS) | compat mode | 1.8–2.1 GB/s | CPU DRAM BW |

---

## References

- [GDS Tuning Guide](https://docs.nvidia.com/gpudirect-storage/performance-guide/index.html)
- [NVMe-oF Performance Best Practices](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#nvmeof-configuration)
- [RAPIDS kvikIO Benchmarks](https://github.com/rapidsai/kvikio/tree/main/benchmarks)
