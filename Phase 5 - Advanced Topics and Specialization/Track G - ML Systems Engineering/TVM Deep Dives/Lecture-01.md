# Lecture 01 — From a model expression to an executable

A compiler translates a computation while preserving its specified behavior. It may also change representation and execution strategy to suit a target. The first useful exercise is small enough that you can see every tensor and calculate the answer independently.

Complete [lab setup and the Relax tests](labs/README.md). The tested release is `apache-tvm==0.27.0.post1`; the [course version map](README.md#version-map-checked-2026-10-18) explains why latest website imports can differ.

## Three representations, three questions

A graph representation describes operations and data dependencies: matmul, bias addition, ReLU, and their shapes. A tensor program describes loops, memory accesses, reductions, and buffers. Target code implements these operations using a CPU/GPU instruction set and runtime ABI.

| Level | Example question |
|---|---|
| Relax graph | Can these operations fuse? Which dimensions are known? |
| Tensor program | Which loop is contiguous? What is the reduction? |
| LLVM/native code | Which instructions and calls execute on this CPU? |
| Runtime | Who owns each tensor, and how is a function invoked? |

An `IRModule` groups functions; it is not itself proof that a model is fully lowered or runnable. A module may contain graph functions, low-level functions, or external calls. Passes transform that module under defined preconditions.

## Work the lab's model

The expression is `Y=relu(XW+b)`. The tested shapes are X `[M,7]`, W `[7,5]`, b `[5]`, and Y `[M,5]`, with M tested at 1, 3, and 9. All values are float32. Each output element reduces over seven products, adds one bias, and clips negative values to zero.

With a one-row X containing all ones, W containing all ones, and b `[-8,-7,0,1,2]`, the result is `[0,0,7,8,9]`. Calculate this before running. The actual tests also use mixed-sign random values so a missing bias, incorrect transpose, or missing ReLU cannot pass just because everything is positive.

In [compiler_lab.py](labs/compiler_lab.py), `AffineRelu` defines the graph with the Relax frontend. `export_tvm` supplies explicit tensor specifications. The program prints or retains the original module, applies `relax.get_pipeline('zero')`, and compiles for `tvm.target.Target('llvm')`. A Relax virtual machine invokes the exported `forward` function on TVM runtime tensors.

The name `zero` does not mean “the compiler performs no transformations.” Inspect the resulting IR and pipeline source; the test checks that graph matmul has lowered into calls to tensor functions. Do not infer a particular fusion from a pipeline name.

## Inputs, parameters, and layout

This lab passes W and b as ordinary inputs rather than loading pretrained weights. A larger model may bind some parameters as constants, transform their layout, or retain them as runtime inputs. Binding changes compilation and packaging assumptions: replacing a constant weight later may require recompilation.

A shape annotation is a contract. The wrong-shape test passes X `[2,7]` to a function compiled for `[3,7]` and verifies a runtime shape mismatch. Dynamic shapes require a deliberately symbolic contract; they are not achieved by ignoring this error. Strides, dtype, device, alignment, and ownership are additional contracts for lower-level interfaces.

## Reference and timing are separate steps

Compare output against NumPy's `maximum(X@W+b,0)` using a justified tolerance. Floating-point reassociation and fused multiply-add can change rounding; bitwise equality is not always the right criterion. Conversely, a large tolerance can conceal a wrong operator or layout. Include small exact examples and edge cases alongside random values.

Compile time, first invocation, and steady-state invocation are different measurements. Record them separately. A fast CPU call on a tiny tensor does not establish a speedup on a transformer or GPU. The lab deliberately makes no framework-speedup promise.

**Exercise.** If X changes values but retains shape/dtype, must this input-driven graph be recompiled? **Answer:** normally no. If its compile-time shape changes from `[3,7]` to `[4,7]`, either compile a new static specialization or design a supported symbolic-shape graph.

**Deliverable.** Original and lowered IR, input/output shapes, a reference comparison, and the wrong-shape failure. Consult the [upstream Quick Start](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html) for the current development flow, matched to its package revision. Next: [scheduling](Lecture-02.md).
