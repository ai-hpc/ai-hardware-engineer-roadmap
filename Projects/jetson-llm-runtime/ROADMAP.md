# Runtime learning milestones

A milestone is complete only when its acceptance evidence exists for this checkout. These stages are a development exercise, not a claim that an unfinished GPU engine is ready to serve a robot.

| Stage | Deliverable | Current evidence |
| --- | --- | --- |
| 1. Sampling semantics | Inspectable normalized distribution; invalid-input behavior; deterministic replay | Actual CPU tests and ASan/UBSan pass |
| 2. File-format boundary | Typed bounded GGUF reader; explicit supported profile; tensor byte accounting | Actual native fixtures, truncation and malformed-input tests pass |
| 3. Tokenization | Exact parity with one pinned pretrained tokenizer, including Unicode and special-token policy | Toy greedy-byte exercise complete; pretrained parity open |
| 4. Model semantics | Exact architecture/tensor contracts and reference forward-pass comparison | Configuration checks complete; full forward parity open |
| 5. CUDA kernels | Independent references, launch/error checks, shape coverage, target execution | Sources available; target correctness/performance open |
| 6. Cache/state | Position-correct multi-token generation, bounds, repeated requests and reset | FP16 contiguous path only; full acceptance open |
| 7. Runtime lifetime | Allocation failure, stream ownership, cancellation, cleanup and concurrency | Source audit improvements; stress/target acceptance open |
| 8. Serving | Robust HTTP/schema handling, admission, cancellation, versioned readiness | Experimental source only |
| 9. Optimization | Same-input comparisons after parity, profiled bottleneck, quality and energy checks | No local target performance claim |

## Exercise: turn a bug into a test

The CPU lab demonstrates three examples. A top-p implementation must sort even when top-k is disabled; a repetition penalty must have a defined distinct-token policy; and a binary parser must reject a truncated array instead of guessing its byte size. Write the expected behavior before changing the implementation. Show that the old behavior fails the test and the corrected behavior passes.

For GPU work, preserve the same discipline. Do not enable graph replay while host-captured position values remain stale. Do not enable INT8 KV without retaining the corresponding scales and passing them to the attention computation. Do not return pointers into pageable memory merely because the SoC shares physical DRAM. These are contracts to implement and verify, not switches that establish correctness.

## Exercise: compare an upstream implementation

Review a pinned revision of [the separate GeniePod runtime](https://github.com/GeniePod/genie-ai-runtime). Choose one component, such as tokenization or KV persistence. Identify the representation, ownership, error handling, and acceptance tests. Explain which tests would need to pass before moving that design into another runtime.

Retain provenance and licenses for any adopted code. The existence of a faster upstream release does not establish parity, support, or performance for this teaching directory.
