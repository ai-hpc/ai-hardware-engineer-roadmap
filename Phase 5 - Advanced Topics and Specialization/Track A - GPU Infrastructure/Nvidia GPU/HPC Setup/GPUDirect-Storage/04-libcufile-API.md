# 04 — libcufile Programming API

## 1. Core API Overview

libcufile provides three categories of operations:

```
1. Initialization
   cuFileDriverOpen()       — initialize GDS driver connection
   cuFileDriverClose()      — cleanup

2. File Handles
   cuFileHandleRegister()   — register a POSIX fd with GDS
   cuFileHandleDeregister() — release file handle

3. Buffer Registration
   cuFileBufRegister()      — register GPU memory for DMA
   cuFileBufDeregister()    — release GPU buffer registration

4. I/O Operations
   cuFileRead()             — GPU DMA read from file
   cuFileWrite()            — GPU DMA write to file

5. Batch I/O (async)
   cuFileBatchIOSetUp()     — create batch context
   cuFileBatchIOSubmit()    — submit batch of I/O ops
   cuFileBatchIOGetStatus() — poll for completion
   cuFileBatchIOCancel()    — cancel pending batch
   cuFileBatchIODestroy()   — cleanup batch context
```

---

## 2. Basic GDS Read: File → GPU Memory

```cpp
#include <cufile.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <assert.h>

void gds_read_example(const char* filepath, size_t file_size) {

    // === 1. Initialize GDS driver ===
    CUfileError_t status;
    status = cuFileDriverOpen();
    assert(status.err == CU_FILE_SUCCESS);

    // === 2. Open file ===
    int fd = open(filepath, O_RDONLY | O_DIRECT);
    //                                ^^^^^^^^ O_DIRECT required for GDS
    assert(fd >= 0);

    // === 3. Register file handle with GDS ===
    CUfileDescr_t cf_descr = {};
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t cf_handle;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    assert(status.err == CU_FILE_SUCCESS);

    // === 4. Allocate GPU buffer ===
    void* d_buf;
    cudaMalloc(&d_buf, file_size);

    // === 5. Register GPU buffer for DMA ===
    // This pins the GPU physical pages for direct DMA access
    status = cuFileBufRegister(d_buf, file_size, 0);
    assert(status.err == CU_FILE_SUCCESS);

    // === 6. Read: file → GPU memory (DMA, no CPU bounce buffer) ===
    ssize_t bytes_read = cuFileRead(
        cf_handle,      // registered file handle
        d_buf,          // destination: GPU buffer (registered)
        file_size,      // bytes to read
        0,              // file offset
        0               // GPU buffer offset
    );
    assert(bytes_read == (ssize_t)file_size);

    // === 7. GPU can now use d_buf directly ===
    myKernel<<<grid, block>>>(d_buf, file_size / sizeof(float));
    cudaDeviceSynchronize();

    // === 8. Cleanup ===
    cuFileBufDeregister(d_buf);
    cuFileHandleDeregister(cf_handle);
    close(fd);
    cudaFree(d_buf);
    cuFileDriverClose();
}
```

---

## 3. Basic GDS Write: GPU Memory → File

```cpp
void gds_write_example(const char* filepath, float* d_output, size_t size) {

    cuFileDriverOpen();

    // Open file for writing with O_DIRECT
    int fd = open(filepath, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    assert(fd >= 0);

    // Register file and GPU buffer (same pattern as read)
    CUfileDescr_t cf_descr = {};
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t cf_handle;
    cuFileHandleRegister(&cf_handle, &cf_descr);
    cuFileBufRegister(d_output, size, 0);

    // Write GPU memory → file (DMA)
    ssize_t bytes_written = cuFileWrite(
        cf_handle,
        d_output,       // source: GPU buffer
        size,           // bytes to write
        0,              // file offset
        0               // GPU buffer offset
    );
    assert(bytes_written == (ssize_t)size);

    cuFileBufDeregister(d_output);
    cuFileHandleDeregister(cf_handle);
    close(fd);
    cuFileDriverClose();
}
```

---

## 4. Alignment Requirements

GDS requires **512-byte alignment** for all parameters:

```cpp
// ALL of these must be multiples of 512:
//   file_offset    (byte position in file)
//   gpu_buf_offset (byte offset into GPU buffer)
//   transfer_size  (bytes to transfer)

// GPU buffer allocation must also be aligned:
void* d_buf;
size_t ALIGN = 512;
size_t aligned_size = (size + ALIGN - 1) & ~(ALIGN - 1);  // round up
cudaMalloc(&d_buf, aligned_size);

// For pointer alignment, use cuMemAlloc with alignment specification:
CUdeviceptr d_ptr;
cuMemAlloc(&d_ptr, aligned_size);

// Or use posix_memalign for pinned host buffers:
void* h_buf;
posix_memalign(&h_buf, 4096, aligned_size);  // 4096 for O_DIRECT
cudaHostRegister(h_buf, aligned_size, cudaHostRegisterDefault);
```

---

## 5. Batch I/O — High-Throughput Async Operations

Batch I/O submits multiple transfers simultaneously, maximizing NVMe queue depth:

```cpp
#define BATCH_SIZE 16      // submit 16 I/O ops at once
#define CHUNK_SIZE (4 * 1024 * 1024)  // 4 MB per chunk (512-byte aligned)

void gds_batch_read(CUfileHandle_t* handles, void** gpu_buffers,
                    size_t* sizes, int n_files) {

    // === Setup batch context ===
    CUfileBatchHandle_t batch;
    cuFileBatchIOSetUp(&batch, BATCH_SIZE);

    // === Submit batch of reads ===
    CUfileIOParams_t io_params[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE && i < n_files; i++) {
        io_params[i].mode          = CUFILE_BATCH;
        io_params[i].fh            = handles[i];
        io_params[i].u.batch.devPtr_base = gpu_buffers[i];
        io_params[i].u.batch.file_offset = 0;
        io_params[i].u.batch.devPtr_offset = 0;
        io_params[i].u.batch.size  = sizes[i];
        io_params[i].opcode        = CUFILE_READ;
    }

    cuFileBatchIOSubmit(batch, BATCH_SIZE, io_params, 0);

    // === Poll for completion ===
    CUfileIOEvents_t io_events[BATCH_SIZE];
    unsigned completed = 0;
    while (completed < BATCH_SIZE) {
        unsigned nr = BATCH_SIZE;
        CUfileError_t err = cuFileBatchIOGetStatus(batch, completed, &nr, io_events, NULL);
        assert(err.err == CU_FILE_SUCCESS);
        for (unsigned j = 0; j < nr; j++) {
            if (io_events[j].status == CUFILE_COMPLETE) {
                completed++;
            }
        }
    }

    cuFileBatchIODestroy(batch);
}
```

---

## 6. GDS with PyTorch DataLoader

The most common production use: loading training data directly into GPU tensors.

```python
# gds_dataset.py
import torch
import cufile  # pip install cufile (NVIDIA Python bindings)
import numpy as np
import os

class GDSDataset(torch.utils.data.Dataset):
    """
    Dataset that loads data directly from NVMe to GPU memory using GDS.
    Zero CPU involvement in the data path.
    """

    def __init__(self, data_dir: str, gpu_id: int = 0):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".bin")
        ])
        self.gpu_id = gpu_id
        self.sample_size = 4096 * 512 * 2  # seq=4096, hidden=512, BF16=2 bytes
        # Must be 512-byte aligned ✓ (4096 * 512 * 2 = 4194304)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        filepath = self.files[idx]

        # Allocate GPU tensor — GDS will fill it directly
        # Shape: [seq_len, hidden_dim], dtype: bfloat16
        tensor = torch.empty(
            4096, 512,
            dtype=torch.bfloat16,
            device=f"cuda:{self.gpu_id}",
        )

        # GDS read: NVMe → GPU (no CPU bounce buffer)
        with cufile.open(filepath, "r") as f:
            f.read(tensor)   # direct DMA into tensor's GPU memory

        return tensor


# Usage with DataLoader
dataset = GDSDataset("/mnt/nvme0/training_data/", gpu_id=0)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,         # 4 worker processes each doing GDS reads
    pin_memory=False,      # already in GPU memory, no pin needed
    prefetch_factor=2,     # prefetch 2 batches ahead
)

for batch in loader:
    # batch is already on GPU — no .to(device) needed!
    loss = model(batch).loss
    loss.backward()
```

### Lower-Level PyTorch GDS Integration

```python
import torch
import ctypes
import os

# Load libcufile manually for fine-grained control
_cufile = ctypes.CDLL("libcufile.so")

def gds_load_tensor(filepath: str, shape: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Load a binary file directly into a GPU tensor using GDS."""

    # Allocate GPU tensor
    tensor = torch.empty(shape, dtype=dtype, device=device)
    nbytes = tensor.element_size() * tensor.numel()

    # Ensure 512-byte alignment
    assert nbytes % 512 == 0, f"Size {nbytes} not 512-byte aligned"

    # Open file with O_DIRECT (required for GDS)
    fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
    try:
        # Register file with GDS
        cf_descr = ...  # CUfileDescr_t via ctypes
        cf_handle = ...
        _cufile.cuFileHandleRegister(ctypes.byref(cf_handle), ctypes.byref(cf_descr))

        # Register GPU buffer
        data_ptr = tensor.data_ptr()
        _cufile.cuFileBufRegister(ctypes.c_void_p(data_ptr), ctypes.c_size_t(nbytes), 0)

        # Read: file → GPU
        _cufile.cuFileRead(cf_handle, ctypes.c_void_p(data_ptr), ctypes.c_size_t(nbytes), 0, 0)

        # Cleanup registrations
        _cufile.cuFileBufDeregister(ctypes.c_void_p(data_ptr))
        _cufile.cuFileHandleDeregister(cf_handle)
    finally:
        os.close(fd)

    return tensor
```

---

## 7. Overlap I/O and Compute with CUDA Streams

The key to maximum throughput: load the next batch while processing the current one.

```cpp
// Double-buffered training: I/O and compute in parallel
void overlap_io_compute(CUfileHandle_t fh, float* model_weights, int steps) {

    cudaStream_t compute_stream, io_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&io_stream);

    // Two GPU buffers: ping-pong
    void* d_buf[2];
    cudaMalloc(&d_buf[0], BATCH_SIZE);
    cudaMalloc(&d_buf[1], BATCH_SIZE);
    cuFileBufRegister(d_buf[0], BATCH_SIZE, 0);
    cuFileBufRegister(d_buf[1], BATCH_SIZE, 0);

    // Pre-load first batch
    cuFileRead(fh, d_buf[0], BATCH_SIZE, 0 * BATCH_SIZE, 0);

    for (int step = 0; step < steps; step++) {
        int cur = step % 2;
        int nxt = (step + 1) % 2;

        // Submit NEXT batch I/O while CURRENT batch computes
        // GDS runs on io_stream independently
        if (step + 1 < steps) {
            cuFileRead(fh, d_buf[nxt], BATCH_SIZE, (step+1) * BATCH_SIZE, 0);
            // Note: cuFileRead is synchronous in cufile 2.x
            // Use cuFileBatchIO for true async overlap
        }

        // Process CURRENT batch on compute_stream
        trainKernel<<<grid, block, 0, compute_stream>>>(
            (float*)d_buf[cur], model_weights, BATCH_SIZE / sizeof(float)
        );
        cudaStreamSynchronize(compute_stream);
    }

    cuFileBufDeregister(d_buf[0]);
    cuFileBufDeregister(d_buf[1]);
    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);
}
```

---

## 8. Error Handling

```cpp
#include <cufile.h>
#include <string>

std::string cufile_strerror(CUfileError_t err) {
    switch (err.err) {
        case CU_FILE_SUCCESS:              return "Success";
        case CU_FILE_DRIVER_NOT_OPEN:      return "Driver not open — call cuFileDriverOpen()";
        case CU_FILE_DRIVER_INVALID_PROPS: return "Invalid config in /etc/cufile.json";
        case CU_FILE_INVALID_VALUE:        return "Invalid argument (check alignment: 512-byte)";
        case CU_FILE_CUDA_DRIVER_ERROR:    return "CUDA driver error";
        case CU_FILE_IO_NOT_SUPPORTED:     return "GDS not supported on this file (check fs type)";
        case CU_FILE_INVALID_MAPPING_SIZE: return "Buffer size not aligned to 512 bytes";
        default: return "Unknown error: " + std::to_string(err.err);
    }
}

// Usage pattern
CUfileError_t status = cuFileRead(handle, buf, size, 0, 0);
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "GDS read failed: %s\n", cufile_strerror(status).c_str());
    // Fall back to posix read if needed
    pread(fd, h_staging, size, 0);
    cudaMemcpy(d_buf, h_staging, size, cudaMemcpyHostToDevice);
}
```

---

## References

- [libcufile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
- [GDS Programming Guide](https://docs.nvidia.com/gpudirect-storage/programming-guide/index.html)
- [cufile Python Bindings](https://github.com/rapidsai/kvikio)
- [RAPIDS kvikIO — high-level Python GDS](https://github.com/rapidsai/kvikio)
