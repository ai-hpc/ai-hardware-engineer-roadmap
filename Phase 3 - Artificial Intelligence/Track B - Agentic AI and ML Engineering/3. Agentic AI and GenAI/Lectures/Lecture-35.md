# Lecture 35 - Agent Skills for GPU Kernel Translation: cuTile Python to cuTile.jl

**Course:** [Agentic AI & GenAI](../Guide.md) | **Previous:** [Lecture 34](Lecture-34.md) | **Next:** [Lecture 36](Lecture-36.md)

---

This lecture is where the **"Agent Skills"** idea becomes concrete for GPU systems work.

The NVIDIA cuTile Python to cuTile.jl case study is important because the hard part is **not generating code**.

The hard part is:

```text
translating domain semantics correctly
when the compiler will not catch many wrong translations
```

That is exactly the class of work where **naive agents fail**.

They can produce plausible code, but **plausible GPU kernel code is not enough**.

You need:

- domain rules
- API mappings
- worked examples
- static validators
- reference tests
- debugging guides
- tolerance rules
- repeatable workflow

In other words:

```text
agent skill + validator + tests = reusable systems knowledge
```

---

## Learning objectives

By the end of this lecture, you should be able to:

1. Explain why cross-DSL GPU kernel translation is a strong use case for agent skills.
2. Describe the differences between cuTile Python and cuTile.jl that cause silent wrong results.
3. Understand why 0-based vs 1-based indexing and row-major vs column-major layout are semantic hazards.
4. Explain how TileGym packages conversion knowledge into a reusable skill.
5. Design a skill directory with rules, API mappings, examples, validation scripts, and tests.
6. Apply the same pattern to CUDA, Triton, MLIR, TVM, tinygrad, and custom accelerator DSLs.
7. Connect agent-generated GPU code to verification evidence and hardware-aware review.

---

## 1. Why this case study matters

Most agent-skill examples are **web or app-development workflows**.

This one is different.

It targets **GPU kernel translation**.

That matters because GPU DSLs are full of **subtle semantic traps**:

- indexing base changes
- memory layout changes
- broadcasting changes
- loop syntax changes
- accumulator shape changes
- padding enum changes
- type-conversion differences
- matrix multiply API differences

The scary part:

```text
many mistakes compile
and then silently produce wrong numbers
```

For GPU engineers, this is a high-value agent use case because the knowledge is:

- specific
- repeatable
- rule-heavy
- testable
- easy to encode in a repo

---

## 2. What cuTile is

**NVIDIA CUDA Tile**, or cuTile, is a tile-based GPU kernel programming model.

Instead of manually coordinating every thread, warp, and shared-memory operation, the programmer works with **tile-level operations**:

```text
load tile
compute on tile
matrix multiply-accumulate
store tile
```

This does not eliminate low-level thinking.

It raises the abstraction level enough that many kernels can be expressed in a more portable, structured way.

**cuTile.jl** brings that style to Julia.

That is valuable for Julia's scientific computing ecosystem:

- differential equations
- probabilistic programming
- physics simulations
- custom numeric kernels
- research code that needs GPU acceleration

The translation target is not "Python code to Julia syntax."

The target is:

```text
preserve GPU kernel semantics across two DSLs
```

---

## 3. The semantic traps

High-level differences:

| Category | cuTile Python | cuTile.jl |
|---|---|---|
| indexing | 0-based, `ct.bid(0)` | 1-based, `ct.bid(1)` |
| broadcasting | implicit, `a + b` | explicit dot syntax, `a .+ b` |
| memory layout | row-major | column-major |
| kernel definition | `@ct.kernel` decorator | plain Julia function |
| constants | `ct.Constant[int]` in signature | `param::Int`, `ct.Constant(val)` at launch |
| type conversion | `tile.astype(ct.float32)` | `convert(ct.Tile{Float32}, tile)` |
| MMA | `ct.mma(a, b, acc=acc)` | `muladd(a, b, acc)` |

None of these are conceptually impossible.

Together, they create a translation surface where a **single missed rule can corrupt results**.

Example:

```text
ct.bid(0) left unchanged
  -> wrong tile loaded
  -> wrong output
  -> no compiler warning
```

Example:

```text
a * b in Julia
  -> matrix multiply

a .* b
  -> element-wise multiply
```

For kernel code, that difference is **decisive**.

---

## 4. Matmul as the teaching example

**Matrix multiplication** is a useful translation example because it combines several hazards:

- block/tile indices
- K-loop over tiles
- accumulator initialization
- type conversion for TF32
- matrix multiply-accumulate
- row-major to column-major layout shift
- store index correctness

Python-style shape thinking:

```text
A(M, K)
B(K, N)
C(M, N)
```

Julia column-major thinking often forces the translated kernel to reason differently about:

- tile orientation
- accumulator shape
- load indices
- store indices

The common failure:

```text
accumulator shape looks plausible
but is transposed for the target layout
```

This is exactly why a skill needs **worked examples**.

The model should not **rediscover matmul layout rules** from scratch every time.

---

## 5. Softmax as the harder example

Softmax adds **algorithmic invariants**, not just syntax.

The NVIDIA post describes three Julia strategies:

- TMA single-tile
- online softmax
- chunked softmax

Softmax translation must preserve:

- running maximum
- running sum
- numerical stability
- reduction axis semantics
- broadcast syntax
- chunking strategy
- dtype tolerance

Examples:

```text
ct.max -> maximum
ct.sum -> sum
axis must shift by +1
ct.maximum(a, b) -> max.(a, b)
ct.exp(ct.sub(a, b)) -> exp.(a .- b)
```

The hard part is not renaming functions.

The hard part is preserving the **mathematical invariant**.

For systems work, this is the recurring theme:

```text
syntax is cheap
semantics are expensive
```

---

## 6. TileGym's skill structure

The project packages the translation workflow into a **repository skill**:

```text
.claude/skills/converting-cutile-to-julia/
  SKILL.md
  translations/
    workflow.md
  references/
    api-mapping.md
    critical-rules.md
    debugging.md
    testing.md
  scripts/
    validate_cutile_jl.py
  examples/
    01_add/
    02_matmul/
    03_softmax/
```

This structure matters.

Each file has a job:

| File | Job |
|---|---|
| `SKILL.md` | entry point and workflow overview |
| `workflow.md` | step-by-step conversion process |
| `api-mapping.md` | Python to Julia API mapping |
| `critical-rules.md` | known semantic traps |
| `debugging.md` | how to diagnose common failures |
| `testing.md` | test patterns and tolerances |
| `validate_cutile_jl.py` | static checker for anti-patterns |
| examples | worked source/target translations |

The key design principle:

```text
put reusable domain knowledge beside the code it governs
```

Do not leave it as a **one-off prompt** in chat history.

---

## 7. What the validator catches

The validator catches **patterns before the GPU runs**.

Examples from the post include:

- leftover `ct.bid(0)`
- Python-style type names
- unsupported loop forms
- common cuTile.jl anti-patterns

This is the important step:

```text
LLM generates candidate
  -> static validator catches known mistakes
  -> tests catch semantic errors
  -> debugging guide routes fixes
```

The model is **not trusted blindly**.

The skill creates a **workflow around it**.

This is the same principle from Lecture 29:

```text
No evidence, no completion.
```

For GPU code, evidence must include numeric correctness.

---

## 8. Test design for translated kernels

The Julia subproject contains:

```text
julia/
  Project.toml
  kernels/
    add.jl
    matmul.jl
    softmax.jl
  test/
    runtests.jl
    test_add.jl
    test_matmul.jl
    test_softmax.jl
```

Good tests compare against **CPU references** with dtype-specific tolerances.

They also test boundary cases:

- dimensions not aligned to tile sizes
- dtype differences
- padding behavior
- reduction axes
- edge shapes

For GPU translation, "passes one happy path" is not enough.

You want:

```text
reference implementation
edge shapes
dtypes
tolerances
boundary tiles
```

This makes the agent output **reviewable by numbers, not vibes**.

---

## 9. Why this is better than a prompt

Prompt:

```text
Be careful with indexing, broadcasting, and memory layout.
```

Skill:

```text
Here are the 17 rules.
Here is the API mapping.
Here are add, matmul, softmax examples.
Here is a validator.
Here are tests and tolerances.
Here is the debugging guide.
```

The difference:

```text
prompt = reminder
skill = executable domain process
```

This is why agent skills are relevant to **hardware and compiler work**.

The model should not rediscover the same **domain pitfalls** repeatedly.

The project should **accumulate them**.

---

## 10. Result pattern

The NVIDIA post reports that a representative GEMM conversion took about:

```text
4 minutes
~78K tokens
no manual intervention
```

Do not overgeneralize this number.

The important point is not the exact time or token count.

The important pattern is:

```text
first port teaches the skill
later ports reuse the skill
each kernel gets cheaper and safer
```

This is how agentic systems improve **without fine-tuning the model**.

They improve by **versioning the workflow, rules, examples, and validators**.

---

## 11. Generalizing beyond cuTile

The same pattern applies to many GPU and compiler workflows:

| Source | Target | Skill focus |
|---|---|---|
| CUDA C++ | Triton | memory layout, block mapping, vectorization |
| Triton | CUDA C++ | explicit shared memory and warp details |
| PyTorch op | CUDA kernel | shape contracts, dtype, autograd |
| CUDA | HIP/ROCm | API mapping, wavefront size, library differences |
| Python DSL | MLIR | types, affine maps, lowering rules |
| TVM schedule | Triton | tiling, memory hierarchy, reduction axes |
| tinygrad op | custom accelerator | shape tracker semantics, memory movement |

Good skill candidates share traits:

- finite recurring rules
- silent semantic failure modes
- reference examples
- static validation possible
- runtime tests possible
- high review cost if done manually

---

## 12. OpenClaw and agent harness mapping

In an OpenClaw-style harness:

```text
source kernel
  -> skill selection
  -> read API mapping and critical rules
  -> generate target kernel
  -> run static validator
  -> run tests on GPU
  -> capture logs/artifacts
  -> summarize diff and evidence
```

Runtime pieces:

| Harness part | Role |
|---|---|
| skill router | choose cuTile translation skill |
| tool policy | allow file reads/writes and test commands only in workspace |
| exec approval | gate GPU test commands if needed |
| artifacts | store validator output and test logs |
| session log | preserve translation reasoning and fixes |
| final-answer hook | require validation evidence |

The LLM writes **candidate code**.

The harness makes the work **safe and reviewable**.

---

## 13. GPU engineer review checklist

When reviewing agent-translated kernels, inspect:

```text
index base
memory layout
tile shape
accumulator shape
reduction axis
broadcast semantics
dtype conversion
padding behavior
loop bounds
boundary tiles
reference test tolerance
performance assumptions
```

Ask:

```text
Could this produce correct results only for square matrices?
Could this pass fp32 but fail lower precision?
Could this fail on non-divisible dimensions?
Could this silently transpose output?
Could this be correct but much slower?
```

Agentic kernel work still requires **human domain review**.

The skill **reduces review burden**.

It does not remove **engineering responsibility**.

---

## 14. Mini-lab: write a DSL translation skill

Pick one translation pair:

- CUDA C++ to Triton
- Triton to CUDA C++
- PyTorch reference to CUDA kernel
- CUDA to HIP
- tinygrad op to CUDA
- cuTile Python to cuTile.jl

Create:

```text
SKILL.md
references/api-mapping.md
references/critical-rules.md
references/testing.md
scripts/validate_translation.py
examples/01_simple/
examples/02_reduction/
examples/03_matmul_or_softmax/
```

Minimum critical rules:

```text
indexing
layout
broadcasting
dtype
boundary conditions
reduction axes
memory aliasing
test tolerance
```

Then run one translation and require:

```text
static validator output
CPU reference comparison
GPU test output
summary of known risks
```

---

## Key takeaways

- Cross-DSL GPU kernel translation is a strong agent-skill use case because the rules are finite, recurring, and testable.
- The hard part is semantic preservation, not syntax conversion.
- cuTile Python to cuTile.jl has traps around indexing, broadcasting, memory layout, constants, type conversion, and MMA APIs.
- TileGym packages translation knowledge into a skill with rules, mappings, examples, validator, tests, and debugging docs.
- Static validation plus runtime tests make agent-generated GPU code reviewable.
- The broader lesson is that systems work needs version-controlled domain skills, not one-off prompts.

---

## References

- NVIDIA Technical Blog, "Automating GPU Kernel Translation with AI Agents: cuTile Python to cuTile.jl": [https://developer.nvidia.com/blog/automating-gpu-kernel-translation-with-ai-agents-cutile-python-to-cutile-jl/](https://developer.nvidia.com/blog/automating-gpu-kernel-translation-with-ai-agents-cutile-python-to-cutile-jl/)
- NVIDIA TileGym repository: [https://github.com/NVIDIA/TileGym](https://github.com/NVIDIA/TileGym)
- cuTile Python documentation: [https://nvidia.github.io/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html](https://nvidia.github.io/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html)
- CUDA.jl: [https://cuda.juliagpu.org/stable/](https://cuda.juliagpu.org/stable/)
- Lecture 29 - Agent Skills: [Lecture-29.md](Lecture-29.md)
- Lecture 32 - LLM From Scratch: [Lecture-32.md](Lecture-32.md)

---

*Next: [Lecture 36 - FP8 KV-Cache in vLLM: Long-Context Serving for Agent Workloads](Lecture-36.md)*
