# 01 — CUDA Graphs: replay work without changing its meaning

**Before:** streams, asynchronous execution, allocation lifetime and a numerical reference. **Outcome:** explain a captured dependency graph and verify that a second input really produces a second result. Use the [shared executable lab](labs/README.md).

## 1. Diagnose launch overhead

A host launch submits GPU work. Its duration, the time until GPU execution begins, and the GPU operation's duration are different measurements. Submission can overlap execution, so adding all API durations does not automatically give the critical-path overhead.

**Hypothetical example:** 100 short operations take 2 µs each on a device, while serial host submission takes 5 µs per operation. A trace might reveal that submission limits the chain. If operations instead take 2 ms each, the same submission cost may be largely hidden. These numbers illustrate a question to measure; they are not CUDA timing guarantees.

A CUDA Graph represents operations and dependencies that can be instantiated and replayed with less repeated host setup. Replay still has a cost. Graphs are useful when a repeated dependency structure and a measurable submission bottleneck justify capture and management.

## 2. Separate definition, instantiation and execution

You can define a graph through explicit node APIs or stream capture. Capture records supported operations and their stream/event dependencies; it does not turn arbitrary host code into GPU code. Instantiation creates an executable graph and validates relevant requirements. Launch enqueues an execution of that graph.

```text
allocate stable buffers -> define/capture graph -> instantiate
                                      |
             update input contents -> launch -> await completion -> inspect output
                                      ^                               |
                                      +-------------------------------+
```

Dependencies permit valid scheduling. Independent nodes are eligible for concurrency, but resources and scheduling determine whether they overlap. Same-stream capture preserves order; it does not automatically discover every algebraic independence in your program.

See [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) for current capture restrictions, node types, updates and conditional execution.

## 3. Follow a complete small program

Open [cuda_examples.cu](labs/cuda_examples.cu). Its graph captures two operations:

```text
input -> y = 2*x + 1 -> output = max(y,0)
```

Allocation happens before capture. Capture includes no device work that needs a host result. After instantiation, each iteration copies new input into the same allocation, launches the graph, copies output back, and synchronizes the stream before comparing with the CPU reference.

The harness checks two different inputs at each nonempty size. A test that replays one input repeatedly could miss a stale input or output pointer. Sizes 31/32/33 and 255/256/257 exercise boundaries; zero length is handled on the host instead of launching a zero-sized grid.

**Worked input:** `x=[-1,0,2]` must produce `[0,1,5]`. Change only the input values and preserve allocation addresses. Then explain which graph node uses each pointer.

The source was compiled with CUDA 13.4.92. GPU execution was unavailable during review, so this example is **compiled, not GPU-validated**. The [lab record](labs/validation.json) preserves that distinction.

## 4. Know what may change

A captured launch records arguments and dependencies. Updating data in a still-live input allocation is different from replacing the allocation with another pointer. Kernel parameters and graph structure can sometimes be updated using supported graph APIs; updates have compatibility restrictions and may require re-instantiation.

“CUDA Graphs always require fixed shapes” is too broad. A particular framework capture may require stable shapes and addresses; native graphs support specified updates and control mechanisms. Read the exact runtime/framework contract rather than promoting one restriction to the entire CUDA API.

**Lifetime rule:** input/output buffers must remain valid until all graph executions that use them finish. One execution's output must not be overwritten while another consumer still needs it. Graph management does not remove buffer ownership obligations.

## 5. PyTorch capture is a separate integration

For `torch.cuda.CUDAGraph`, begin with evaluation mode where appropriate, inference/no-grad behavior, representative warmup, the documented stream synchronization, and long-lived inputs/outputs. A returned view of static output can be overwritten on replay; either finish consuming it before replay or make a correctly ordered copy.

A Python branch that needs `.item()` from a CUDA tensor introduces a host dependency. Python executes during capture; its control flow is not replayed as ordinary Python. Capturable random-number operations and allocator behavior also have framework-specific rules.

`torch.compile(mode="reduce-overhead")` requests a compilation configuration that may use CUDA Graphs for eligible regions. It does not guarantee one graph for every model, fixed microsecond overhead, or a particular speedup. CPU backends and graph breaks further change the path. Consult [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) and [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## 6. Dynamic inputs need semantic checks

Bucketing is one strategy: pad inputs to a supported bucket and reuse its graph. Padding must preserve attention masks, position IDs, cache positions, lengths and output slicing. Padding token IDs alone is not a complete decoder contract. Handle requests larger than the largest bucket explicitly.

A bucketed graph should be checked against an eager reference for short and boundary lengths, not just the bucket's maximum length. Keep per-bucket memory and warmup cost in the deployment budget.

**Exercise:** a graph accepts length 128. For a valid input of length 93, list the 35 padded positions and explain which attention operations must ignore them. Does slicing 93 output rows after execution repair an attention computation that used the padded tokens? No; contamination already happened inside the computation.

## 7. Measure the comparison fairly

Compare ordinary launches and graph replay with identical operation semantics, inputs, stream policy and timing boundary. Report capture/instantiation separately, then steady-state repetitions. Include host submission and end-to-end measurements when that is the suspected bottleneck.

Use a separate [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) trace to inspect gaps and dependencies. Profiler overhead and tracing options affect results. Do not publish a fixed speedup table without hardware, software, shapes, sample count and raw measurements.

**Completion:** draw the graph, explain stable allocations, pass changing-input reference checks on available hardware, and state whether the measured bottleneck changed. Without a GPU, submit the compiled artifact and predicted outputs and leave runtime correctness/performance unvalidated.

**Next:** [cooperative groups](02-Cooperative-Groups.md).
