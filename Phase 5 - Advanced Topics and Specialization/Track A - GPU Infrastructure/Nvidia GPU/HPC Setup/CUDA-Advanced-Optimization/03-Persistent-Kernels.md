# 03 — Persistent Kernels

## 1. The Problem: Kernel Launch Overhead at Extreme Frequency

CUDA Graphs reduce CPU overhead for a fixed sequence of kernels. But some workloads need to **dynamically dispatch work** to the GPU — such as online inference servers where requests arrive continuously with unknown timing.

```
Standard inference serving model:
  Request arrives → CPU prepares batch → launch kernels → wait for result → next request

Timeline:
  [CPU prep 50µs][kernel 2ms][CPU prep 50µs][kernel 2ms]...
  ↑ repeated kernel launch setup ↑
```

For **low-latency streaming inference** (robotics, real-time control), even CUDA Graphs have limits:
- Graphs require fixed shapes
- Each request still requires a CPU-side graph replay call
- Multi-stream contention from concurrent requests

**Persistent kernels** solve this with a different approach: the kernel **never exits**. It runs forever, polling for new work.

---

## 2. What is a Persistent Kernel?

A persistent kernel is a kernel that:
1. Launches **once** at startup with all SMs
2. **Loops forever** polling a GPU-side work queue
3. Processes work items as they arrive
4. The CPU submits work by writing to GPU memory (zero kernel launch overhead)

```
Standard model:         Persistent model:
CPU launches kernel → | GPU kernel runs forever
GPU runs kernel      | CPU writes work to GPU queue
GPU kernel exits     | GPU polls queue, processes work
CPU launches again   | CPU writes more work
GPU runs kernel      | GPU processes again
GPU kernel exits     | ...
...                  |
```

---

## 3. Persistent Kernel Architecture

```
GPU memory:
┌─────────────────────────────────────────────────────┐
│  Work Queue (ring buffer)                           │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┐       │
│  │ W0   │ W1   │ W2   │ empty│ empty│ empty│       │
│  └──────┴──────┴──────┴──────┴──────┴──────┘       │
│         ↑ head                  ↑ tail              │
│                                                     │
│  Work Item:                                         │
│  { input_ptr, output_ptr, batch_size, status }     │
└─────────────────────────────────────────────────────┘

CPU thread:                    GPU persistent kernel:
  while (serving):               while (running):
    prepare_input()                work = poll_queue()
    write_to_queue()               if work is valid:
    wait_for_output()                process(work)
                                     signal_done(work)
```

---

## 4. Persistent Kernel Implementation

### Work Queue Structure (GPU-accessible)

```cpp
struct WorkItem {
    float*   input;
    float*   output;
    int      batch_size;
    int      seq_len;
    int      status;   // 0=empty, 1=pending, 2=done
    int      padding[2];
};

#define MAX_QUEUE_SIZE 64

struct WorkQueue {
    WorkItem items[MAX_QUEUE_SIZE];
    int      head;   // next slot to read (GPU side)
    int      tail;   // next slot to write (CPU side)
    int      running;
};
```

### The Persistent Kernel

```cpp
__global__ void persistent_inference_kernel(
    WorkQueue* queue,
    ModelWeights* weights,
    int num_sms
) {
    // Only run one thread block per SM for maximum control
    if (blockIdx.x >= num_sms) return;

    int sm_id = blockIdx.x;

    while (queue->running) {
        // Spin-wait for work (polling — burns cycles but zero latency)
        int slot = -1;
        if (threadIdx.x == 0) {
            // Thread 0 polls for available work
            int expected = sm_id % MAX_QUEUE_SIZE;  // round-robin assignment
            if (atomicCAS(&queue->items[expected].status, 1, 0) == 1) {
                slot = expected;
            }
        }

        // Broadcast slot to all threads in block
        slot = __shfl_sync(0xffffffff, slot, 0);

        if (slot >= 0) {
            WorkItem* work = &queue->items[slot];

            // Run the actual inference
            run_transformer_layer(
                work->input, work->output,
                work->batch_size, work->seq_len,
                weights
            );

            // Signal completion (CPU waits on this)
            __threadfence_system();   // ensure output visible to CPU
            atomicExch(&work->status, 2);   // mark done
        }

        // Brief pause to avoid excessive power draw from spin-wait
        // __nanosleep(100);   // CUDA 11.1+ on Ampere+
    }
}
```

### CPU Submit Path (Zero Overhead)

```cpp
class PersistentInferenceServer {
    WorkQueue* d_queue;
    WorkQueue* h_queue;   // pinned host mirror

    int submit(float* input, float* output, int batch, int seq) {
        // Find empty slot
        int slot = allocate_slot();

        // Write work item directly to GPU memory via pinned mapping
        WorkItem& item = d_queue->items[slot];
        item.input      = input;
        item.output     = output;
        item.batch_size = batch;
        item.seq_len    = seq;

        // Memory fence then set status — GPU sees this atomically
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        atomicExch(&item.status, 1);   // mark pending

        return slot;
    }

    void wait(int slot) {
        // Poll until GPU signals done (spin-wait on CPU side)
        while (atomicLoad(&d_queue->items[slot].status) != 2)
            _mm_pause();   // CPU spin hint (reduces power)
    }
};
```

---

## 5. Persistent Kernels for LLM Token Streaming

The most important production use case: streaming token generation without kernel launch per token.

```
Standard vLLM/TRT-LLM decode:
  For each token:
    CPU: launch attention kernel (10µs)
    GPU: attention (5ms)
    CPU: launch FFN kernel (10µs)
    GPU: FFN (5ms)
  Total per token: 10.04ms, of which 0.02ms = CPU overhead (0.2%)
  ← overhead is small for normal decode

For speculative decoding with tiny draft model (0.5ms compute):
  CPU: launch kernel (10µs)
  GPU: tiny draft model (0.5ms)
  CPU overhead: 2% — now matters
```

With persistent kernel for draft model:

```cpp
__global__ void draft_model_persistent(
    TokenStream* token_stream,
    TinyModelWeights* weights
) {
    auto grid = cooperative_groups::this_grid();

    while (token_stream->active) {
        // Poll for next input token
        int token = spin_wait_for_token(token_stream);

        // Run tiny draft model (no kernel launch overhead)
        int proposed_tokens[5];
        draft_forward(token, weights, proposed_tokens, 5);

        // Write proposals for verifier
        write_proposals(token_stream, proposed_tokens);
        grid.sync();   // sync all SMs before signaling

        signal_proposals_ready(token_stream);
    }
}
```

---

## 6. Persistent Kernels in Production Frameworks

### TensorRT-LLM Persistent GEMM

TRT-LLM uses persistent kernels for its GEMM operations when the decoder runs in "streaming" mode:

```cpp
// From TRT-LLM source (simplified)
// The GEMM kernel stays resident, processing one token's projection per iteration
template<typename T>
__global__ void persistent_gemm_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    volatile int* tile_counter   // atomically-incremented work counter
) {
    // Tiles are processed until all are done
    while (true) {
        int tile_id = atomicAdd(const_cast<int*>(tile_counter), 1);
        if (tile_id >= total_tiles) break;   // all work done

        int tile_m = tile_id / num_tiles_n;
        int tile_n = tile_id % num_tiles_n;

        compute_gemm_tile(A, B, C, tile_m, tile_n, M, N, K);
    }
}
```

### FlashAttention Persistent Attention (Hopper)

FlashAttention-3 on H200 uses persistent warps that overlap compute and I/O:

```
Warpgroup A (Producer): continuously loads tiles from HBM
Warpgroup B (Consumer): computes GEMM on loaded tiles
Warpgroup C (Consumer): computes softmax + accumulate

These run concurrently via warp specialization (see topic 05)
The kernel persists for the entire attention computation
```

---

## 7. Persistent Kernel Trade-offs

| Factor | Persistent Kernel | Standard Kernel |
|---|---|---|
| Launch overhead | Zero (kernel already running) | 5–20 µs per launch |
| GPU idle between work items | Zero (polling) | Kernel launch gap |
| Power consumption | Higher (SM busy spinning) | Lower (SM sleeps between launches) |
| Flexibility | Fixed SM occupancy | Dynamic SM allocation |
| Debugging difficulty | Very high (infinite loop) | Standard |
| Best for | High-frequency, low-latency workloads | Batch compute, throughput workloads |
| Not suitable for | Low-frequency tasks (wasteful spinning) | Latency-critical streaming |

### Power consideration

```
Persistent kernel spinning uses ~150W more per GPU vs idle between kernels.
On 8x H200, always-on persistent kernels: ~1.2 kW extra (out of 5.6 kW total)
→ 21% more power for zero-launch-overhead

Production decision: use persistent kernels ONLY for latency-critical paths
(e.g., draft model in speculative decoding, streaming token generation)
Use standard kernels for batch throughput workloads (training, offline inference)
```

---

## 8. Softer Alternative: Thread Block Specialization

For cases where full persistence is too aggressive, **persistent thread blocks** stay alive between tasks using `__grid_constant__` and device-side work queues:

```cpp
__global__ void semi_persistent_kernel(WorkQueue* queue) {
    // Process multiple work items per launch (amortize launch cost)
    // but kernel DOES eventually exit
    while (true) {
        WorkItem* work = try_dequeue(queue);
        if (work == nullptr) break;   // no more work, exit cleanly
        process(work);
    }
}

// Launch once with enough blocks to drain the queue efficiently
// Re-launch when queue refills (much less frequent than per-item launch)
```

This is what **TensorRT-LLM inflight batching** uses — work items accumulate in a queue, one launch drains the current queue, giving ~10–50× fewer kernel launches than naive per-request launching.

---

## References

- [Persistent Thread Models in GPU Computing](https://developer.nvidia.com/blog/cuda-pro-tip-kepler-sm-efficient-persistent-threads/)
- [TensorRT-LLM Inflight Batching](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/inflight_batching.md)
- [FlashAttention-3 Architecture](https://arxiv.org/abs/2407.08608)
- [CUDA Thread Communication Patterns](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
