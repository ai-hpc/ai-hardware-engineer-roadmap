# 01 — CUDA Graphs

## 1. The Problem: CPU Launch Overhead

Every CUDA kernel launch involves the CPU:

```
CPU side:
  1. Validate kernel arguments
  2. Compute grid/block dimensions
  3. Write launch packet to driver queue
  4. Driver forwards to GPU command queue
  5. CPU returns (kernel runs asynchronously)

Cost per launch: 5–20 µs

LLM decode step with 200 kernels:
  200 × 10 µs = 2 ms wasted on launch overhead alone

At batch_size=1, TTFT target = 50 ms:
  2 ms overhead = 4% of budget — significant
  At higher throughput: GPU waits for CPU to submit next kernel
```

This is called **CPU-GPU launch gap** — the GPU sits idle between kernel launches:

```
Timeline (without CUDA Graphs):
  CPU: [submit K1][submit K2][submit K3][submit K4]...
  GPU:     [K1 runs][     K2 runs     ][K3][K4]
           ↑ idle gaps between kernels ↑
```

CUDA Graphs solve this by submitting the entire operation sequence in **one shot**.

---

## 2. How CUDA Graphs Work

CUDA Graphs have three phases:

```
Phase 1 — CAPTURE:
  Run your code once in "recording" mode.
  CUDA captures every kernel launch, memcpy, and sync event
  into a directed acyclic graph (DAG).

Phase 2 — INSTANTIATE:
  CUDA compiles the graph into an executable object (cudaGraphExec_t).
  Pre-allocates all resources, validates topology.
  This takes ~10–50 ms — done once at startup.

Phase 3 — REPLAY:
  Launch the entire graph with a single API call.
  GPU executes all operations with no further CPU involvement.
  Cost: ~3–5 µs (just the launch, not the work).
```

### Graph DAG Structure

```
     [memcpy H2D]
          │
     [embedding kernel]
        / \
  [attn]  [mlp]          ← independent nodes run in parallel
        \ /
  [residual add]
          │
     [layernorm]
          │
     [memcpy D2H]
```

Dependencies are tracked automatically — CUDA Graphs preserve exact execution order AND enable parallel execution of independent branches.

---

## 3. CUDA Graphs in C++

### Manual Graph Construction

```cpp
#include <cuda_runtime.h>

cudaGraph_t     graph;
cudaGraphExec_t graphExec;
cudaStream_t    stream;

cudaStreamCreate(&stream);

// === PHASE 1: CAPTURE ===
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// All operations on `stream` are now captured, not executed
cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);
embeddingKernel<<<grid, block, 0, stream>>>(d_input, d_embed, vocab_size);
attentionKernel<<<grid, block, 0, stream>>>(d_embed, d_qkv, seq_len);
mlpKernel<<<grid, block, 0, stream>>>(d_embed, d_out, hidden_size);
cudaMemcpyAsync(h_output, d_out, size, cudaMemcpyDeviceToHost, stream);

cudaStreamEndCapture(stream, &graph);

// === PHASE 2: INSTANTIATE ===
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphDestroy(graph);   // can free the graph after instantiation

// === PHASE 3: REPLAY (inference loop) ===
for (int request = 0; request < num_requests; request++) {
    // Update input data
    update_input_buffer(h_input, request);

    // Launch entire pipeline with one API call
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    process_output(h_output);
}

// Cleanup
cudaGraphExecDestroy(graphExec);
cudaStreamDestroy(stream);
```

### Graph Node API (Explicit Construction)

For more control, build the graph node-by-node without capture:

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add kernel node
cudaGraphNode_t kernelNode;
cudaKernelNodeParams kernelParams = {};
kernelParams.func = (void*)myKernel;
kernelParams.gridDim  = dim3(grid_x, grid_y);
kernelParams.blockDim = dim3(block_x, block_y);
kernelParams.kernelParams = args;

cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams);

// Add memcpy node with dependency on kernel
cudaGraphNode_t memcpyNode;
cudaGraphAddMemcpyNode1D(&memcpyNode, graph,
    &kernelNode, 1,      // depends on kernelNode
    h_out, d_out, size, cudaMemcpyDeviceToHost);

// Instantiate and launch as before
```

---

## 4. CUDA Graphs in PyTorch

PyTorch wraps CUDA Graphs cleanly. Two patterns:

### Pattern A: `torch.cuda.graph`

```python
import torch

# Must use static (same-shape) tensors for graphs
batch_size, seq_len, hidden = 1, 512, 4096
static_input  = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
static_output = torch.zeros(batch_size, seq_len, hidden, device="cuda")

# Warmup (fills CUDA caches, JIT compiles ops)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Inference loop — GPU executes graph, no CPU kernel submissions
def graph_inference(new_input: torch.Tensor) -> torch.Tensor:
    static_input.copy_(new_input)   # update input in-place
    g.replay()                       # re-execute captured graph
    return static_output.clone()    # clone before next replay overwrites it
```

### Pattern B: `torch.compile(mode="reduce-overhead")`

```python
import torch

# torch.compile automatically wraps in CUDA Graph
model = torch.compile(model, mode="reduce-overhead")

# First call: traces + captures graph (~slow)
out = model(inputs)

# Subsequent calls: replay graph (~3-5 µs overhead)
for batch in dataloader:
    out = model(batch)  # graph replay, fast
```

`mode="reduce-overhead"` is `mode="default"` + automatic CUDA Graph capture.

### Pattern C: Bucketed Graphs for Dynamic Shapes

LLM decode has variable sequence lengths — one graph per bucket:

```python
BUCKETS = [128, 256, 512, 1024, 2048, 4096]

graphs      = {}
static_ins  = {}
static_outs = {}

# Pre-capture one graph per bucket
for seq_len in BUCKETS:
    dummy = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")

    # Warmup
    for _ in range(3):
        model(dummy)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    s_in  = dummy.clone()
    s_out = torch.zeros(1, seq_len, model.config.vocab_size, device="cuda")
    with torch.cuda.graph(g):
        s_out = model(s_in)

    graphs[seq_len]      = g
    static_ins[seq_len]  = s_in
    static_outs[seq_len] = s_out

def bucketed_decode(input_ids: torch.Tensor) -> torch.Tensor:
    L = input_ids.shape[1]
    bucket = next(b for b in BUCKETS if b >= L)
    # Pad to bucket size
    padded = torch.nn.functional.pad(input_ids, (0, bucket - L))
    static_ins[bucket].copy_(padded)
    graphs[bucket].replay()
    return static_outs[bucket][:, :L, :].clone()
```

---

## 5. CUDA Graph Constraints and Limitations

```
Works well with:
  ✓ Fixed shapes (same tensor sizes every call)
  ✓ Repeated inference (same model, many requests)
  ✓ Training with fixed batch size and sequence length
  ✓ Physics simulation time steps

Does NOT work with:
  ✗ Dynamic control flow (Python if/else based on tensor values)
  ✗ Variable-length inputs without bucketing
  ✗ Operations that allocate new memory each call
  ✗ CPU-GPU synchronization points inside the graph
  ✗ Python callbacks inside the capture region

Common pitfall:
  # WRONG — Python branching based on GPU tensor value breaks graph
  if output.argmax() == stop_token:   # requires D2H sync, breaks capture
      break

  # CORRECT — implement stopping criteria outside the graph replay loop
  g.replay()
  next_token = static_output.argmax(dim=-1)   # D2H after replay
  if next_token.item() == stop_token:
      break
```

---

## 6. CUDA Graph Updates (Without Re-capture)

From CUDA 11.1+, you can update parts of a graph without full re-capture:

```cpp
// Update kernel parameters in existing graph
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams);

// Update memcpy source/destination
cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &newMemcpyParams);

// Re-upload to GPU (fast, ~100 µs)
cudaGraphUpload(graphExec, stream);
```

Useful when only tensor contents change (not shapes) — avoids full re-instantiation.

---

## 7. Profiling CUDA Graphs

```bash
# CUDA Graphs appear as "CUDA Graph" events in Nsight Systems
nsys profile \
    --trace=cuda,nvtx \
    --gpu-metrics-device=all \
    python inference.py

# In Nsight Systems GUI:
# - CUDA Graph launches appear as a single block
# - Expand to see individual node executions inside the graph
# - Compare: kernel-by-kernel vs graph — look for removed launch gaps
```

```python
# Mark graph capture and replay with NVTX for visibility
import torch.cuda.nvtx as nvtx

with nvtx.range("graph_capture"):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(inp)

for i in range(N):
    with nvtx.range(f"graph_replay_{i}"):
        g.replay()
```

---

## 8. Expected Performance Gains

| Workload | Without Graphs | With Graphs | Speedup |
|---|---|---|---|
| Llama-3 8B, BS=1, seq=512 | ~45 ms | ~32 ms | 1.4× |
| ResNet-50, BS=1 | ~3.2 ms | ~1.8 ms | 1.8× |
| BERT-large, BS=1 | ~12 ms | ~8 ms | 1.5× |
| Training step (fixed BS) | ~820 ms | ~780 ms | 1.05× |

Graphs have the largest impact at **batch_size=1** (pure latency mode) where kernel launch overhead is a large fraction of total time. At large batch sizes, kernel launch overhead is amortized over more work.

---

## References

- [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graph Tutorial](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [TensorRT-LLM CUDA Graph Usage](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/cuda-graph.md)
- [Nsight Systems CUDA Graph Analysis](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
