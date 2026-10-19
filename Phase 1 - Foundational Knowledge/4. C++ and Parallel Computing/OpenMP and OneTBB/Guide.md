# OpenMP and oneTBB

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">OAO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Budget parallelism across the whole pipeline.</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


Part of [Phase 1 section 4 — C++ and Parallel Computing](../Guide.md).

**Goal:** Shared-memory **CPU parallelism** with **OpenMP** (directive-based) and **oneTBB** (task-based algorithms and flow graphs) so structured parallel patterns feel familiar before CUDA.

This guide covers two parts. Section 1 is a progressive tour of OpenMP — from a one-line `#pragma` to tasks and SIMD. Section 2 covers oneTBB's template-based API in the same order, from simple loops to flow graphs and runtime controls.

---

## 1. OpenMP

### 1.0 Mental Model: Fork-Join

OpenMP uses the **fork-join model**. Your program starts with a single initial thread. When it reaches a parallel region, it **forks** into a team of worker threads. The encountering thread becomes the team's primary thread (thread zero); team threads execute the region and synchronize at its end. The encountering thread then continues. Physical simultaneous execution depends on available CPU resources.

```
 serial          parallel region              serial
─────────── ╔══════════════════════════╗ ───────────────►
            ║  #pragma omp parallel    ║
   primary ──╫──► thread 0  (work) ────╫──► primary
            ╠──► thread 1  (work) ────╣
            ╠──► thread 2  (work) ────╣
            ╚──► thread 3  (work) ────╝
                                  ▲
                         implicit barrier:
                    all threads must arrive
                    before primary continues
```

You can have **multiple parallel regions** in one program. Each fork creates the thread team; each join dissolves it (or returns threads to a pool for reuse).

```
primary ──── [serial] ──── fork ──── [parallel] ──── join ──── [serial] ──── fork ──── [parallel] ──── join ────►
```

**Three things OpenMP manages for you automatically:**

| What | How OpenMP handles it |
|------|-----------------------|
| Thread creation | Implementation-managed team; runtimes commonly reuse worker threads |
| Work distribution | Divides loop iterations across threads (`schedule` clause controls how) |
| Synchronization | Implicit barrier at the end of every parallel region |

**Compile a standalone OpenMP file:** `g++ -O2 -std=c++17 -fopenmp example.cpp -o /tmp/example`. The Fibonacci comparison additionally needs `-ltbb`; see section 1.16. Most short blocks below are function-body fragments and use application functions such as `compute`; they are not complete translation units. [openmp_patterns.cpp](openmp_patterns.cpp) supplies actual checked examples for these concepts.

---

### 1.1 Your First Parallel Loop

```cpp
#include <omp.h>
#include <vector>

int N = 1'000'000;
std::vector<float> a(N), b(N), c(N);

// Sequential: thread 0 does all N iterations
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

// Request 8 threads and explicitly choose static scheduling.
#pragma omp parallel for num_threads(8) schedule(static)
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];
```

That's it. One line added. The compiler splits `[0, N)` into chunks, assigns each chunk to a thread, and inserts a barrier at the end.

**Illustration if the actual team has eight threads:**

```
Thread 0:  i = 0 … 124,999
Thread 1:  i = 125,000 … 249,999
Thread 2:  i = 250,000 … 374,999
...
Thread 7:  i = 875,000 … 999,999
```

This is safe because each iteration writes a distinct `c[i]`, and `a`/`b` are read-only and do not alias the output. Threads may read shared input and touch the same cache line; sharing a cache line can affect performance without creating a data race. A thread-count request can be constrained by runtime settings; query the actual team instead of assuming eight.

---

### 1.2 The Classic Race Condition

**Wrong — data race:**

```cpp
double sum = 0.0;

#pragma omp parallel for
for (int i = 0; i < N; i++)
    sum += a[i];   // ← RACE: multiple threads read-modify-write sum simultaneously

// Undefined behavior: the output may change, or may accidentally look correct.
```

**Why it breaks:**

```
Thread 0 reads sum = 5
Thread 1 reads sum = 5     ← same value, before thread 0 wrote back
Thread 0 writes sum = 5 + a[0] = 7
Thread 1 writes sum = 5 + a[1] = 6   ← overwrites thread 0's result!
```

One update is lost. This is a classic **read-modify-write race**.

**Fix: `reduction` clause**

```cpp
double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++)
    sum += a[i];   // each thread accumulates its own private sum
                   // all private sums are added together at the end
```

Each thread gets its own private copy of `sum` (initialized to 0). After the loop, OpenMP adds all private copies into the original `sum`. No race. Section 1.5 covers common operators and custom reductions. Floating-point addition can regroup, so a correct reduction need not be bit-identical to serial addition.

---

### 1.3 Data Sharing Clauses

Every variable referenced inside a parallel region is either **shared** (one copy, all threads see it) or **private** (each thread has its own copy). For a `parallel` region, outer ordinary variables are usually shared; loop indices and local automatic variables have predetermined rules, and explicit tasks have additional implicit firstprivate rules. Use `default(none)` to require explicit decisions where applicable. Sharing an object does not make concurrent mutation safe.

`private` and `firstprivate` are alternatives for a given variable on this construct; do not list it in both. `firstprivate` adds initialization from the original:

```cpp
int x = 10;

// Option A — private(x): each thread gets its own x, value is UNINITIALIZED
// Use when: you assign x before reading it inside the loop anyway
#pragma omp parallel for private(x)
for (int i = 0; i < N; i++) {
    x = compute(i);   // x is assigned first → uninitialized value never read → safe
    a[i] *= x;
}

// Option B — firstprivate(x): each thread gets its own x, initialized to 10
// Use when: you read x before assigning it (need the original value)
#pragma omp parallel for firstprivate(x)
for (int i = 0; i < N; i++) {
    a[i] = x + i;    // x is read first → uses the initial value 10 supplied by firstprivate
    // Leave x unchanged so every iteration uses the original offset.
}
```

**Rule:** initialize scratch before reading it. `firstprivate` initializes each implicit task's copy once, **not once per iteration**. If you increment it, later iterations on that thread see the increment. A copied pointer still points to the same allocation; it does not copy or protect its pointee. [Firstprivate semantics](https://www.openmp.org/spec-html/5.2/openmpsu38.html).

#### When `private` is the right choice

Use `private` when the variable is **purely a scratch/temp** — it gets completely overwritten at the start of every iteration, so the uninitialized value never matters:

```cpp
char buf[64];        // scratch buffer — we always snprintf before using it
float tmp;           // intermediate result — always assigned before read
int  row, col;       // 2D indices derived from i — always computed fresh

// All three are scratch: always written first → private is correct
#pragma omp parallel for private(buf, tmp, row, col)
for (int i = 0; i < N; i++) {
    row = i / COLS;                       // written first
    col = i % COLS;                       // written first
    tmp = heavy_compute(matrix[row][col]); // written first
    snprintf(buf, sizeof(buf), "(%d,%d)=%.2f", row, col, tmp);  // written first
    store_result(i, buf, tmp);  // must copy buf before returning; do not retain this pointer
}
```

Think of it this way: **if you would declare the variable *inside* the loop body, it belongs in `private`**. `private` is just moving a loop-local variable outside (maybe because you need it in the pragma or it's a fixed-size buffer).

```cpp
// These are equivalent:

// Version A — variable declared inside loop (naturally private, no clause needed)
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    float tmp = a[i] * b[i];   // each iteration's own tmp
    result[i] = tmp + c[i];
}

// Version B — variable declared outside, made private with clause
float tmp;
#pragma omp parallel for private(tmp)
for (int i = 0; i < N; i++) {
    tmp = a[i] * b[i];         // same effect
    result[i] = tmp + c[i];
}
```

> **Prefer Version A** when possible — declare scratch variables inside the loop body. Use a clause when you deliberately need an outer declaration. Fixed-size arrays can also be local; variable-length arrays are not standard C++17. For class types, privatization invokes the applicable constructors rather than universally leaving the entire object uninitialized.

---

#### `lastprivate` — Getting the Value from the Last Iteration

`lastprivate` is private during the loop, but after the loop ends, the value from the **sequentially last iteration** (for this increasing loop, `i=N-1`) is copied back to the original variable.

**The problem it solves:** after a parallel loop, you sometimes need to know what a variable held in the final iteration — as if the loop ran serially.

```cpp
// Imagine you're sweeping x from 0.0 to 1.0 and computing sin(x) at each step.
// After the loop, you want sin(x) at the last step — i.e., sin(x_max).

const int N  = 1000;
const double dx = 1.0 / (N - 1);

double x      = 0.0;   // current x value
double sin_x  = 0.0;   // sin at current x

#pragma omp parallel for lastprivate(x, sin_x)
for (int i = 0; i < N; i++) {
    x     = i * dx;          // each iteration computes its own x
    sin_x = std::sin(x);     // and its own sin_x
    output[i] = sin_x;
}

// After the loop:
// x     = (N-1) * dx  = 1.0        ← value from i = N-1 (last iteration)
// sin_x = sin(1.0)    ≈ 0.8415     ← value from i = N-1 (last iteration)
printf("At x=%.4f, sin(x)=%.4f\n", x, sin_x);
```

**What each thread sees vs what you get back:**

```
Thread 0 processes i = 0..249:
  last x in its range = 249 * dx = 0.249
  last sin_x          = sin(0.249)

Thread 1 processes i = 250..499:
  last x in its range = 499 * dx = 0.499
  last sin_x          = sin(0.499)

Thread 2 processes i = 500..749:
  last x  = 0.749,  sin_x = sin(0.749)

Thread 3 processes i = 750..999:   ← sequentially last range
  last x  = 999 * dx = 1.0        ← THIS gets copied back
  sin_x   = sin(1.0)              ← THIS gets copied back

After join: x = 1.0, sin_x = sin(1.0) — as if the loop ran serially
```

> **`lastprivate` copies the sequentially last iteration's value**, not whichever thread finished last. That is the lowest index for a decreasing loop. Assign the variable on that last iteration; ordinary `lastprivate` does not search backward for the last assignment. It does not repair a loop-carried recurrence. [Lastprivate semantics](https://www.openmp.org/spec-html/5.2/openmpsu39.html).

**Common real use cases:**

| Pattern | Why `lastprivate` |
|---------|------------------|
| Independently computed endpoint | Need the final iteration's value |
| Last matching index | Prefer a `max` reduction on matching indices; ordinary `lastprivate` is insufficient |
| Mesh traversal — record position of last node processed | Node index/coords after full sweep |
| Numerical integration — endpoint value needed post-loop | Value of integrand at upper bound |

**`firstprivate` and `lastprivate` may coexist**, but this does not reproduce a serial recurrence. For example, initialize `x=0`, then run four `x += 1` iterations with two threads and `schedule(static,2)`: each private copy ends at 2, so the copied-back result is 2; serial execution gives 4. For a total use a reduction. For a true prefix recurrence, derive a scan/parallel algorithm or keep the dependence serial. The checked source demonstrates this counterexample without a race.

---

#### Full Clause Reference

| Clause | Initialized? | Written back after loop? | Use when |
|--------|:-----------:|:------------------------:|---------|
| `shared(var)` | — (one copy) | — (always live) | Read-only inside loop, or safely written with sync |
| `private(var)` | C++ default initialization | No | Scratch — always written before read |
| `firstprivate(var)` | **Yes** (from original) | No | Needs original value inside loop |
| `lastprivate(var)` | C++ default initialization | **Yes** (from last `i`) | Need loop's final value after it ends |
| `firstprivate` + `lastprivate` | **Yes** | **Yes** | Needs original value AND final value |
| `reduction(op:var)` | Yes (identity) | Yes (combined) | Accumulate across all iterations |

> **Rule of thumb:** If all threads only read → `shared`. Scratch variable → `private`. Needs original value → `firstprivate`. Need value after loop ends → `lastprivate`. Accumulate → `reduction`.

---

### 1.4 Schedules — How Work Is Divided

OpenMP divides loop iterations among threads according to a *schedule*. The right schedule depends on whether iterations take equal time or vary widely.

```cpp
// Static: divide into approximately equal chunks before execution
// Exact chunk sizes without an explicit chunk are not fully prescribed.
// Best when: each iteration costs the same amount of time
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) { /* uniform work */ }

// Static with chunk: interleave chunks of k iterations
// Thread 0 gets 0-7, 32-39, 64-71, ...   (chunk=8, 4 threads)
// Check locality/false sharing; round-robin chunks may hurt contiguous writes.
#pragma omp parallel for schedule(static, 8)
for (int i = 0; i < N; i++) { /* */ }

// Dynamic: each thread takes the next k iterations when it becomes free
// Overhead: ~synchronization per chunk fetch
// Best when: iterations have unpredictable/varying cost
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < N; i++) { /* variable-cost work */ }

// Guided: starts with large chunks, shrinks over time
// Reduces scheduling overhead while handling tail imbalance
// Can help irregular work; early heavy chunks may still cause imbalance.
#pragma omp parallel for schedule(guided)
for (int i = 0; i < N; i++) { /* */ }

// Runtime: schedule determined by OMP_SCHEDULE env variable
// OMP_SCHEDULE="dynamic,32" ./my_program
#pragma omp parallel for schedule(runtime)
for (int i = 0; i < N; i++) { /* */ }
```

Omitting `schedule` leaves the choice implementation-defined. Guided scheduling responds to remaining iteration count, not measured iteration cost. Treat these as starting hypotheses and measure. [Implementation-defined schedule behavior](https://www.openmp.org/spec-html/5.2/openmpap1.html).

**When to pick which:**

```
All iterations take the same time?     → static (lowest overhead)
Iterations have wildly different cost? → dynamic
Don't know, want adaptive?             → guided
Tuning at runtime without recompile?   → runtime
```

---

### 1.5 Reductions

Building on the `reduction(+:sum)` clause introduced in section 1.2, here are common operators and how to define custom reductions:

```cpp
// A bounded integer example; arbitrary products can overflow even with reduction.
std::vector<int> a{1, 2, 3, 4}, b{1, 2, 1, 2};
const int N = static_cast<int>(a.size());
long long sum = 0, product = 1;
int max_val = std::numeric_limits<int>::lowest();

#pragma omp parallel for reduction(+:sum) reduction(*:product) reduction(max:max_val)
for (int i = 0; i < N; i++) {
    sum      += a[i];
    product  *= b[i];
    max_val   = std::max(max_val, a[i]);
}
```

Common C++ reduction identifiers include `+`, `*`, `&`, `|`, `^`, `&&`, `||`, `min`, `max`. Include `<algorithm>`, `<limits>` and `<vector>` for this fragment. The original value also participates in the final combination; a nonzero initial sum is included once.

**Custom reduction (C++ example, OpenMP 4.0+; also available in C and Fortran):**

```cpp
struct Vec3 { float x, y, z; };

// Declare how to combine two partial results (omp_out += omp_in)
// and what value each thread's private copy starts at.
#pragma omp declare reduction(vec_add : Vec3 : \
    (omp_out.x += omp_in.x, \
     omp_out.y += omp_in.y, \
     omp_out.z += omp_in.z)) \
    initializer(omp_priv = Vec3{0, 0, 0})

Vec3 total{0, 0, 0};

#pragma omp parallel for reduction(vec_add : total)
for (int i = 0; i < N; i++) {
    total.x += forces[i].x;   // accumulate into the thread-private copy of total
    total.y += forces[i].y;
    total.z += forces[i].z;
}
// After the loop: OpenMP calls the combiner to merge all thread-private
// totals into the final total using the vec_add rules above.
```

---

The combiner is an expression; the commas above form one C++ comma expression. Separate semicolon statements inside that syntax do not compile. [Declare-reduction syntax](https://www.openmp.org/spec-html/5.2/openmpsu55.html).

### 1.6 Collapse — Parallelizing Nested Loops

When a nested loop's outer dimension is small (e.g., 4 rows but 8 threads), `collapse` merges the outer and inner loops into a single flattened iteration space so all threads have work.

```cpp
// Without collapse: only the outer loop is parallelized
// If outer loop has fewer iterations than threads → wasted threads
#pragma omp parallel for
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 1000; j++)
        A[i][j] = B[i][j] * C[i][j];

// With collapse(2): outer * inner = 4000 iterations are parallelized
// Each thread gets a portion of the flattened 4000-iteration space
#pragma omp parallel for collapse(2)
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 1000; j++)
        A[i][j] = B[i][j] * C[i][j];
```

> Try `collapse` when the outer loop count is small, but first verify independence across the combined iteration space. It does not remove dependencies or guarantee useful speedup; loop-nest restrictions and cache locality still matter.

---

### 1.7 SIMD — Vectorization Hints

SIMD (Single Instruction, Multiple Data) processes multiple array elements with a vector instruction. It introduces data-parallel thinking useful for CUDA, while CPU vector lanes and GPU SIMT threads remain distinct execution models.

```cpp
// Ask the compiler to vectorize (SIMD) one loop on a single thread
#pragma omp simd
for (int i = 0; i < N; i++)
    c[i] = a[i] * b[i] + d[i];

// Parallelize across threads AND vectorize within each thread's chunk
#pragma omp parallel for simd
for (int i = 0; i < N; i++)
    c[i] = a[i] * b[i];

// simd with reduction (e.g. sum with SIMD accumulation)
float sum = 0.0f;
#pragma omp simd reduction(+:sum)
for (int i = 0; i < N; i++)
    sum += a[i];
```

The directive permits concurrent SIMD execution under OpenMP semantics; it is not a guarantee that the compiler will prove arbitrary code safe or emit a chosen vector width. Keep iterations independent in these examples, express reductions correctly, and honor any `aligned`/`safelen` promises you add. Use GCC `-fopt-info-vec` to inspect emitted vectorization, then validate results. CPU SIMD lanes and GPU SIMT threads have different programming and scheduling models. [OpenMP SIMD semantics](https://www.openmp.org/spec-html/5.2/openmpse60.html).

---

### 1.8 Protecting Shared State: critical and atomic

When `reduction` is not enough (complex shared state):

```cpp
// critical: only one thread at a time
// Correct, but slow (acts like a mutex around the block)
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp critical
    {
        global_map[key(i)] += value(i);
    }
}

// atomic: protect a supported access; mapping and cost depend on the target
int counter = 0;
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    counter++;           // all accesses to this shared counter must be compatible
}

// atomic with operation
#pragma omp atomic update
total += a[i];

int val;
#pragma omp atomic read
val = shared_var;

#pragma omp atomic write
shared_var = computed;
```

Protecting an atomic counter does not by itself publish unrelated payload memory. Use the relevant barrier or a correctly derived acquire/release protocol. All accesses to a protected object must follow a compatible synchronization scheme.

**Critical vs Atomic:**
- `#pragma omp critical` — mutual exclusion for a structured block; contention can serialize work.
- `#pragma omp atomic` — protects supported read/write/update/capture/compare forms. The compiler may use hardware instructions or runtime support. Measure cost; frequent updates to one shared cache line can remain a bottleneck.

---

### 1.9 Synchronization: Barriers and nowait

Every `#pragma omp for` has an **implicit barrier** at the end — all threads block until the slowest one finishes. `nowait` removes it so fast threads can immediately start the next independent loop.

```
Default (implicit barrier):
T0: [████ loop ████]░░░░░░░░░░ wait ░░░░░[next loop]
T1: [████████████████ loop ████████████][next loop]
T2: [██ loop ██]░░░░░░░░░░░░░░ wait ░░░░[next loop]
                               ▲
                   all threads held here

With nowait:
T0: [████ loop ████][next loop immediately]
T1: [████████████████ loop ████████████][next loop]
T2: [██ loop ██][next loop immediately]
     this example requires data-independent producer loops
```

```cpp
#pragma omp parallel
{
    #pragma omp for nowait          // no barrier after this loop
    for (int i = 0; i < N; i++)
        a[i] = compute_a(i);        // writes only a[]

    #pragma omp for nowait          // no barrier after this loop either
    for (int i = 0; i < N; i++)
        b[i] = compute_b(i);        // writes only b[] — independent of a[]

    #pragma omp barrier             // explicit sync: a[] and b[] both fully written
    #pragma omp single
    use(a, b);                      // one consumer after all producers finish
}
```

**Beginner rule:** remove a barrier only when later work does not require unfinished writes. Separate arrays are insufficient if helper functions share hidden state. All team threads must encounter worksharing constructs/barriers in a valid order. The barrier above makes both arrays ready; `single` separately ensures the consumer runs once. More advanced same-thread scheduling proofs can permit some dependent loops, but require stronger conditions than this example.

---

### 1.10 Sections — Fixed Concurrent Operations

`sections` is for a **small, known-at-compile-time** set of independent operations — not a loop. Each section executes once by one team thread. Assignment is unspecified; one thread can execute several sections, so overlap is permitted rather than guaranteed.

```
#pragma omp parallel sections
┌─────────────────────────────────────────────────┐
│  section 1 → T0: [────── load_data() ──────────]│
│  section 2 → T1: [── init_config() ──]          │
│  section 3 → T2: [──────── warmup() ────────────]│
└────────────────────────── implicit barrier ──────┘
                  all three done → program continues
```

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    load_data();        // any team thread

    #pragma omp section
    init_config();      // any team thread

    #pragma omp section
    warmup();           // must not depend on unfinished load/init
}
// all three finished here
```

**Sections vs Tasks:**

| | `sections` | `tasks` |
|--|-----------|---------|
| Number of units | Fixed at compile time | Dynamic, recursive |
| Use case | Load + init + warmup | Tree traversal, Fibonacci |
| Overhead | Measure for the work size | Measure task creation and scheduling costs |

Use `sections` when you can count the operations by hand. Use `tasks` (next section) when the work fans out recursively or the count isn't known until runtime.

---

### 1.11 Tasks — Recursive and Irregular Work

Tasks decouple **work creation** from **work execution**. Any encountering thread may generate a task. The runtime can execute it immediately or defer it to an eligible team thread; one universal queue or execution order is not prescribed.

```
single thread creates tasks:          thread pool executes them:
  task(left  branch) ──► queue ──► T1: sum_tree(left)
  task(right branch) ──► queue ──► T2: sum_tree(right)
  taskwait ─────────────────────► T0: waits, then adds results
```

**Tree sum — correct pattern (parallel region outside, tasks inside):**

```cpp
// The recursive function only creates tasks — no parallel region here
int sum_tree(Node* node) {
    if (!node) return 0;
    int left_sum = 0, right_sum = 0;

    #pragma omp task shared(left_sum) firstprivate(node)
    left_sum = sum_tree(node->left);

    #pragma omp task shared(right_sum) firstprivate(node)
    right_sum = sum_tree(node->right);

    #pragma omp taskwait        // wait for both before combining
    return node->value + left_sum + right_sum;
}

// Parallel region created ONCE outside — not inside the recursive function
int result;
#pragma omp parallel
#pragma omp single              // one thread drives task creation
result = sum_tree(root);
```

> Keep the tree alive and immutable until all tasks finish. `firstprivate(node)` copies the pointer, not the tree. Each recursive call waits before its stack-local results go out of scope. A nested `parallel` in each call can serialize when nesting is disabled or create excessive teams when enabled; it does not inherently make correct arithmetic wrong.

**Fibonacci with cutoff:**

```cpp
long long fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n-1) + fib_serial(n-2);
}
long long fib(int n) {  // exercise inputs: 0 <= n <= 42
    if (n < 20) return fib_serial(n);
    long long x, y;
    #pragma omp task shared(x) firstprivate(n)
    x = fib(n - 1);
    #pragma omp task shared(y) firstprivate(n)
    y = fib(n - 2);
    #pragma omp taskwait
    return x + y;
}

// Inside main:
long long result = 0;
#pragma omp parallel
{
    #pragma omp single
    result = fib(30);  // 832040; use the bounded benchmark CLI in section 1.16
}

```

**Why each clause matters:**

| Clause | What it does |
|--------|-------------|
| `shared(x)` | The child writes its parent call's stack variable; the parent keeps it alive until `taskwait` |
| `firstprivate(n)` | Explicit snapshot at task creation; this local argument would normally already be implicitly firstprivate |
| `taskwait` | Waits for previously generated direct children; each child here recursively waits for its own children |
| `single` | Starts one root invocation. Descendant tasks may be generated by other threads. Without it, each team thread duplicates the full computation and races on `result` |

**How the recursion tree maps to tasks — `fib(6)` example:**

For visualization only, reduce the cutoff to **2**. Then `fib(6)` has 25 function invocations, with 24 child tasks plus the root call. With the actual cutoff 20, `fib(6)` runs serially. The thread labels below depict one possible schedule, not a guarantee.

```
fib(6)                                ← T0 spawns fib(5) and fib(4), then taskwait
├─ fib(5)                             ← T1 picks up, spawns fib(4) and fib(3)
│  ├─ fib(4)                          ← T2 picks up, spawns fib(3) and fib(2)
│  │  ├─ fib(3)                       ← T3 picks up, spawns fib(2) and fib(1)
│  │  │  ├─ fib(2) → fib(1)+fib(0)   ← spawns two leaves, then returns 1
│  │  │  └─ fib(1)                    ← returns 1 immediately (leaf)
│  │  └─ fib(2) → fib(1)+fib(0)      ← spawns two leaves, then returns 1
│  └─ fib(3)
│     ├─ fib(2) → fib(1)+fib(0)
│     └─ fib(1)
└─ fib(4)                             ← T2 (or T3) steals this after finishing above
   ├─ fib(3)
   │  ├─ fib(2) → fib(1)+fib(0)
   │  └─ fib(1)
   └─ fib(2) → fib(1)+fib(0)
```

**Execution flow with 4 threads:**

```
time →

T0: [spawn fib(5), fib(4)] [taskwait........] [return 5+3=8]
T1: [fib(5): spawn fib(4),fib(3)] [taskwait] [return 3+2=5]
T2: [fib(4): spawn fib(3),fib(2)] [taskwait] [return 2+1=3] [steal fib(4)→return 3]
T3: [fib(3): spawn fib(2),fib(1)] [return 1+0+1=2]          [fib(3)→return 2]
     ↑                                ↑
  tasks created                 threads steal work
  and queued                    as soon as they're idle
```

At `taskwait`, the executing thread can run eligible tasks while its current task is suspended. Scheduling restrictions, task granularity, imbalance and CPU availability can still leave workers idle. A single producer can become a bottleneck in other workloads; no runtime promises full utilization. [Taskwait semantics](https://www.openmp.org/spec-html/5.2/openmpse93.html).

Count tasks before benchmarking: `T(n)=0` below cutoff `c`, and `T(n)=2+T(n−1)+T(n−2)` otherwise. For `n=30,c=20`, this creates **464 tasks**. The checked source verifies this count against actual task generation. A larger task count can increase overhead; it does not imply speedup. The old `int fib(50)` example also overflowed a 32-bit signed result; `fib(50)=12,586,269,025` does not fit.

---

### 1.12 Thread Info and Environment

```cpp
// Query runtime info
int n_threads = omp_get_num_threads();   // inside parallel region
int thread_id = omp_get_thread_num();    // 0 to n_threads-1
int max_threads = omp_get_max_threads(); // default upper-bound request for a later region
int n_procs = omp_get_num_procs();       // processors available to the device/runtime

// Set thread count
omp_set_num_threads(8);

// Timing
double t0 = omp_get_wtime();
// ... work ...
double elapsed = omp_get_wtime() - t0; // take both timestamps on the same thread
```

**Environment variables (set before running):**

```bash
export OMP_NUM_THREADS=8      # request; actual team can be smaller
export OMP_SCHEDULE="dynamic,64"  # only affects schedule(runtime)
export OMP_PROC_BIND=close    # placement relative to the primary thread's place
export OMP_PLACES=cores       # abstract places; topology is implementation-dependent
export GOMP_SPINCOUNT=100000  # GNU-specific spin iterations, not microseconds
```

---

### 1.13 Nested Parallelism

```cpp
void outer_task() {
    omp_set_max_active_levels(2);  // allow at most two active nested levels
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 4; i++) {
        // Each outer thread requests an inner team of 2 INCLUDING itself.
        #pragma omp parallel for num_threads(2)
        for (int j = 0; j < N; j++)
            compute(i, j);
    }
}
```

> `omp_set_nested` is deprecated; use `omp_set_max_active_levels` or `OMP_MAX_ACTIVE_LEVELS`. With all requests honored, this example can have eight inner workers total, not twelve: each inner primary is an existing outer thread. Runtime/thread limits may serialize or reduce teams. Independent nested loops often fit one team with `collapse(2)`; nested tasks can also reuse the existing team. [Runtime control and deprecation](https://www.openmp.org/spec-html/5.0/openmpsu119.html).

---

### 1.14 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Shared write without sync | Wrong results, non-deterministic | `reduction`, `atomic`, or `critical` |
| Capturing `i` by reference in tasks | Task reads wrong `i` | `firstprivate(i)` on the task |
| `single` missing in task-creation code | Task explosion | Add `#pragma omp single` around task creation |
| Calling `omp_get_num_threads()` outside parallel | Returns 1 always | Call inside the parallel region |
| Missing `taskwait` before using task results | Use before ready | `#pragma omp taskwait` |
| `nowait` on dependent loops | Uses wrong data | Remove `nowait` or add explicit `barrier` |
| Long critical sections | Serializes threads | Narrow the critical section, use `atomic` for simple ops |

---

### 1.15 OpenMP vs oneTBB

| | OpenMP | oneTBB |
|--|--------|--------|
| API style | Compiler directives (`#pragma`) | C++ templates and lambdas |
| Learning curve | Lower — one pragma per feature | Higher — need to know template types |
| Granularity | Loops, explicit tasks, task dependencies, SIMD | Parallel algorithms, explicit tasks, flow graphs |
| Load balancing | Static/dynamic/guided schedules | Work-stealing (automatic, adaptive) |
| Dependency graphs | Task `depend` clauses; no identical node-container API | Explicit `flow_graph` node/message API |
| Private storage | `private` belongs to construct/task data environments; `threadprivate` persists per thread under specified conditions | `enumerable_thread_specific` is associated with accessing threads |
| Nested parallelism | Nested regions/tasks with runtime controls | Nested algorithms sharing scheduler resources |
| SIMD | OpenMP SIMD directives | Compiler vectorization, supported SIMD directives, or intrinsics inside task bodies |
| Best for | Existing loops, Fortran interop, scientific HPC | New C++ code, complex graphs, irregular tasks |

**Resources:** [OpenMP specifications](https://www.openmp.org/specifications/) · [OpenMP 5.2 reference guide (PDF)](https://www.openmp.org/wp-content/uploads/OpenMPRefGuide-5.2-Web-2024.pdf)

The table compares programming interfaces. OpenMP runtimes can also use work stealing for tasks; the OpenMP specification does not mandate one universal scheduler. oneTBB adds its own flow-node and concurrent-container APIs. The next section covers oneTBB's API in the same progressive style, starting from simple loops and building toward complex graph-based patterns.

---

### Practice: run the OpenMP contracts

From this directory:

```bash
g++ -O2 -std=c++17 -fopenmp -Wall -Wextra -Werror openmp_patterns.cpp -o /tmp/openmp_patterns
/tmp/openmp_patterns
```

**Executed:** GCC 13.3, CPU, October 18, 2026, with observed teams 1, 2 and 4; the same executable also passed UndefinedBehaviorSanitizer. The source checks private-value lifetime, decreasing-loop `lastprivate`, custom reduction with a nonzero original value, one consumer after two `nowait` producers, SIMD reduction, collapsed iteration coverage, and recursive task results/counts. The two-thread recurrence counterexample returns 2 while serial returns 4. It reports if the runtime instead serializes that case. The [execution record](labs/openmp-validation.json) retains the compiler command, outputs and source hash.

**Predict first:** does changing the task cutoff change Fibonacci's result? **No; it changes task generation and performance.** Does `firstprivate(pointer)` create a separate array? **No; it copies the address.** Does reaching `single` synchronize unfinished producers? **No; the explicit barrier before the consumer is necessary here.**

---

### 1.16 Run a task-granularity experiment

The checked-in [fib_benchmark.cpp](fib_benchmark.cpp) compares recursive Fibonacci using serial calls, OpenMP tasks, and `oneTBB::parallel_invoke`. This deliberately redundant recursion makes task overhead visible. For computing Fibonacci itself, its iterative reference is the better algorithm; parallelizing an exponential algorithm does not make it a good production choice.

**Build from this directory** with a C++17 compiler, OpenMP support, and oneTBB development headers/library. On Ubuntu with GCC the packages are `g++` and `libtbb-dev`; GCC's `-fopenmp` selects its OpenMP runtime, so LLVM's `libomp-dev` is not an extra requirement for this command.

```bash
g++ -O2 -std=c++17 -fopenmp -Wall -Wextra fib_benchmark.cpp -ltbb -o /tmp/fib_benchmark
/tmp/fib_benchmark --self-test
/tmp/fib_benchmark 30 2 20 3
```

The positional arguments are `n`, parallelism budget, task cutoff, and repetition count. A lower cutoff creates more tasks. A higher cutoff leaves more work serial. There is no universally best value; vary one input at a time and keep the same thread budget for both runtimes. The executable sets the OpenMP team request and oneTBB concurrency limit, prints the observed OpenMP team, warms all three implementations, checks every result against an iterative reference, and reports median elapsed time using `steady_clock`.

**Executed 2026-10-18:** GCC 13.3, oneTBB 2021.11.0, Ubuntu 24.04 x86-64. All 27 self-test input/cutoff combinations passed for each implementation. One shared-host run with `n=30`, budget 2, cutoff 20, and 3 repetitions produced median times 1.293 ms serial, 1.193 ms OpenMP, and 2.128 ms oneTBB. These small, noisy observations demonstrate the reporting workflow; they are not a performance ranking. Record CPU, affinity, load, compiler/library versions, and command with your own results.

```bash
# Keep work and thread budget fixed while varying task granularity.
for cutoff in 10 15 20 25 30; do
  /tmp/fib_benchmark 30 2 "$cutoff" 5
done
```

**Explain the result:** task creation, scheduler choices, imbalance, shared resources, and problem size all affect speedup. OpenMP does not prescribe one universal task scheduler, and an inline lambda does not imply lower runtime overhead. Inspect work distribution instead of attributing every difference to a library name.

**Check:** is `speedup < 1` a failed correctness test? **Answer:** no; the implementation may be correct but slower for that workload. Is the same numerical result enough to compare performance? **Answer:** no; match algorithm/work, thread limits, inputs, warmup, and measurement boundaries.

Run the supplied regression suite from this directory with `python3 -m unittest discover -s labs -v`. It compiles the actual source files and checks numerical/reference results, invalid arguments, bounded pipeline progress, and frame identity. Missing development tools are reported as a skip, not a successful experiment.

---

## 2. oneTBB (oneAPI Threading Building Blocks)

### 2.0 Mental Model: Work-Stealing Scheduler

oneTBB is a **task-based** parallel programming library. Instead of managing threads directly, you express *what* can run in parallel — the runtime distributes work using a **work-stealing** scheduler.

**How work-stealing works:**

```
Thread 0's deque:  [task A] [task B] [task C]  ← pushes/pops from right (LIFO, cache-friendly)
Thread 1's deque:  [] ← empty, idle
Thread 2's deque:  [task X]
Thread 3's deque:  [] ← empty, idle

Step 1: Thread 0 pops task C from its own deque (right end, hot in cache)
Step 2: Thread 1 is idle → STEALS task A from Thread 0's LEFT end
Step 3: Thread 3 is idle → STEALS task X from Thread 2
```

**Why steal from the left?**
- Owner pops from the **right** (LIFO) — processes its own tasks in stack order, cache-friendly
- A stealer takes an older ready task from the opposite end. In recursive workloads it may expose a larger subtree; task age does not guarantee task size.

**Result:** ready work can move to otherwise idle workers. Dependencies, blocking calls, limited ready parallelism, task overhead, concurrency limits, and shared-memory bottlenecks can still leave workers idle. Measure task size and the concurrency budget. [oneTBB scheduler model](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/How_Task_Scheduler_Works.html)

**Install:**

```bash
# Ubuntu/Debian
sudo apt install libtbb-dev

```

In `CMakeLists.txt`, after defining `my_target`:

```cmake
find_package(TBB REQUIRED)
target_link_libraries(my_target PRIVATE TBB::tbb)
```

**Header:**

```cpp
#include "oneapi/tbb.h"
using namespace oneapi::tbb;
```

---

### 2.1 `parallel_for` — Parallel Loop

The fundamental building block. Splits a range into chunks and executes each chunk on an available thread.

**Simple 1D integer form:**

```cpp
const int N = 1'000'000;
std::vector<float> a(N), b(N), c(N);  // include <vector>; avoid large stack arrays

// Compact form — runtime picks chunk size automatically
parallel_for(0, N, [&](int i) {
    c[i] = a[i] + b[i];
});
```

**`blocked_range` form — gives you the subrange, loop inside:**

```cpp
// Chunk form: lets you control setup and the inner loop
parallel_for(blocked_range<int>(0, N),
    [&](const blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); i++) {
            c[i] = a[i] + b[i];
        }
    }
);
```

`blocked_range<T>(begin, end)` is the half-open interval `[begin, end)`. The lambda receives a subrange `r` — call `.begin()` and `.end()` to iterate it.

**Why `blocked_range` over compact form?**
- Exposes a chunk-level loop; compilers can inline either form, so measure any performance difference
- Lets you initialize buffers once per chunk (instead of per element)
- Required for custom chunk-level logic (SIMD, temp allocation)

**2D range — matrix or image processing:**

```cpp
parallel_for(blocked_range2d<int>(0, rows, 0, cols),
    [&](const blocked_range2d<int>& r) {
        for (int i = r.rows().begin(); i < r.rows().end(); i++)
            for (int j = r.cols().begin(); j < r.cols().end(); j++)
                out[i][j] = process(in[i][j]);
    }
);
```

**3D range:**

```cpp
parallel_for(blocked_range3d<int>(0, D, 0, H, 0, W),
    [&](const blocked_range3d<int>& r) {
        for (int d = r.pages().begin(); d < r.pages().end(); d++)
            for (int h = r.rows().begin(); h < r.rows().end(); h++)
                for (int w = r.cols().begin(); w < r.cols().end(); w++)
                    vol[d][h][w] = f(d, h, w);
    }
);
```

#### Partitioners — Control Chunking

The third optional argument controls how work is divided:

```cpp
// auto_partitioner (default): runtime tunes chunk size automatically
parallel_for(blocked_range<int>(0, N), body);

// affinity_partitioner: remembers affinity hints to encourage cache reuse
// Keep this object alive across sequential repetitions of this loop.
// Give concurrent invocations separate partitioner objects.
affinity_partitioner ap;
parallel_for(blocked_range<int>(0, N), body, ap);

// simple_partitioner: splits until ranges are no larger than grainsize
parallel_for(blocked_range<int>(0, N, 1000), body, simple_partitioner());

// static_partitioner: approximately equal partitions and deterministic affinity hints
// An affinity hint is not a guarantee of OS-thread identity or CPU pinning
parallel_for(blocked_range<int>(0, N), body, static_partitioner());
```

| Partitioner | Chunk size | Adaptive splitting | Work-stealing | Use when |
|-------------|------------|--------------------|---------------|---------|
| `auto_partitioner` | Dynamic | Yes | Yes | Most cases — default |
| `affinity_partitioner` | Adaptive | Yes | Yes | Re-running same loop over same data (cache-warm) |
| `simple_partitioner` | Bounded by grainsize | No adaptive tuning | Yes | Bound the amount of work in each range |
| `static_partitioner` | Approximately equal partitions | No adaptive tuning | Scheduler-controlled | Predictable partitioning with affinity hints; not CPU pinning |

The [oneTBB partitioner specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/elements/onetbb/source/algorithms/partitioners/static_partitioner) describes affinity as a scheduler hint. Numerical reproducibility is a separate property of the reduction order.

**Grainsize tuning:** Small chunks can spend more time scheduling than computing; large chunks can reduce load balance. The historical 100,000-cycle heuristic is a starting point, not an API requirement. Measure several grain sizes with your workload, thread budget, and compiler, and compare against a serial baseline.

#### Complete Example 1 — Squaring an Array

The clearest way to see the transition from serial to parallel:

```cpp
#include <oneapi/tbb.h>
#include <iostream>
#include <vector>

int main() {
    const size_t N = 1'000'000;
    std::vector<int> data(N, 3);    // all 3s
    std::vector<int> result(N);

    // Serial version:
    // for (size_t i = 0; i < N; ++i)
    //     result[i] = data[i] * data[i];

    // Parallel version — one line change:
    oneapi::tbb::parallel_for(size_t(0), N, [&](size_t i) {
        result[i] = data[i] * data[i];   // each thread handles a range of i
    });

    std::cout << "result[0]=" << result[0]
              << "  result[N-1]=" << result[N-1] << "\n";
    // Expected: 9  9
}
```

**One possible partitioning (not a thread-assignment guarantee):**

```
N = 1,000,000   threads = 8

Thread 0:  i = 0 … 124,999       → result[i] = data[i]²
Thread 1:  i = 125,000 … 249,999 → result[i] = data[i]²
...
Thread 7:  i = 875,000 … 999,999 → result[i] = data[i]²

All done → join back
```

No race condition: each thread writes to a different `result[i]`.

---

#### Complete Example 2 — ReLU on a 2D Feature Map

ReLU (`max(0, x)`) is applied element-wise after a convolution layer. Every output value is independent — a perfect fit for `blocked_range2d`.

```cpp
#include <oneapi/tbb.h>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    // Feature map: H rows × W columns, row-major flat storage
    const size_t H = 1024, W = 1024;
    std::vector<float> feat(H * W);
    std::vector<float> out (H * W);

    // Fill with values in range [-128, 127]
    for (size_t i = 0; i < H * W; ++i)
        feat[i] = static_cast<float>(i % 256) - 128.0f;

    // Apply ReLU in parallel over tiles
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range2d<size_t>(0, H, 0, W),
        [&](const oneapi::tbb::blocked_range2d<size_t>& tile) {
            for (size_t y = tile.rows().begin(); y < tile.rows().end(); ++y)
                for (size_t x = tile.cols().begin(); x < tile.cols().end(); ++x)
                    out[y * W + x] = std::max(0.0f, feat[y * W + x]);
        }
    );

    std::cout << "out(0,0)   = " << out[0]   << "\n";  // 0   (was -128)
    std::cout << "out(0,129) = " << out[129] << "\n";  // 1   (was 1)
    std::cout << "out(0,255) = " << out[255] << "\n";  // 127 (was 127)
}
```

**Why consider `blocked_range2d`:**

```
blocked_range2d splits the grid into rectangular tiles:

  ┌───────┬───────┬───────┐
  │ T0    │ T1    │ T2    │   Each tile = several contiguous row segments
  │       │       │       │   → good L1/L2 cache reuse within tile
  ├───────┼───────┼───────┤
  │ T3    │ T4    │ T5    │
  │       │       │       │
  └───────┴───────┴───────┘

Nested parallel_for → exposes nested work; measure its granularity and overhead
blocked_range2d    → exposes rectangular ranges; tune grain sizes for the workload
```

The `tile.rows()` and `tile.cols()` accessors give you the subrange for this tile. Flat `vector<T>` with `[y * W + x]` indexing gives contiguous row-major storage. A partial-width tile spans separate row segments, not one contiguous allocation. A vector of vectors allocates rows separately; use it when that layout is appropriate. Neither layout nor a range type guarantees a speedup.

---

### 2.2 `parallel_reduce` — Parallel Reduction

For loops that accumulate a result: sum, min, max, dot product, histogram.

**Lambda form (most common):**

```cpp
// Sum of array
float total = parallel_reduce(
    blocked_range<int>(0, N),
    0.0f,                                        // identity value
    [&](const blocked_range<int>& r, float init) -> float {
        for (int i = r.begin(); i < r.end(); i++)
            init += a[i];
        return init;
    },
    [](float x, float y) -> float {             // combine partial results
        return x + y;
    }
);
```

**Arguments:**
1. Range to iterate
2. Identity value (initial value for each partial result)
3. Body: takes a subrange + running partial → returns new partial
4. Combine: merges two partial results into one

**Dot product:**

```cpp
float dot = parallel_reduce(
    blocked_range<int>(0, N), 0.0f,
    [&](const blocked_range<int>& r, float init) -> float {
        for (int i = r.begin(); i < r.end(); i++)
            init += a[i] * b[i];
        return init;
    },
    std::plus<float>{}
);
```

**Find minimum with index (finite input values):**

```cpp
struct MinResult { float val; int idx; };
auto better = [](MinResult a, MinResult b) {
    if (a.idx < 0) return b;
    if (b.idx < 0) return a;
    if (a.val != b.val) return a.val < b.val ? a : b;
    return a.idx < b.idx ? a : b;  // deterministic lowest-index tie break
};
auto result = parallel_reduce(
    blocked_range<int>(0, N), MinResult{0.0f, -1},
    [&](const blocked_range<int>& r, MinResult curr) {
        for (int i = r.begin(); i < r.end(); ++i)
            curr = better(curr, MinResult{a[i], i});
        return curr;
    }, better);
// Empty input gives idx == -1. Define a separate NaN policy if inputs allow NaN.
```

> **Common mistake:** Do not reset the accumulator to zero inside the body — `init` carries prior partial results. Resetting it discards prior work.

**Deterministic reduce** — a deterministic reduction tree for a fixed range and partitioner:

```cpp
float result = parallel_deterministic_reduce(
    blocked_range<int>(0, N, 1000),  // explicit grainsize chosen for this example
    0.0f,
    [&](const blocked_range<int>& r, float init) {
        return std::accumulate(a.data() + r.begin(), a.data() + r.end(), init);
    },
    std::plus<float>{}
);
// Reproducible for fixed inputs, deterministic body, build, and FP environment
```

The tree does not guarantee identical bits across compiler flags, architectures, floating-point modes, or a body that itself performs nondeterministic work. See the [deterministic-reduction contract](https://uxlfoundation.github.io/oneTBB/main/reference/source/algorithms/functions/parallel_deterministic_reduce_func.html).

#### Complete Example 3 — Mean Squared Error (MSE)

MSE is computed after every forward pass during training: sum `(pred - target)²` over all samples, divide by N. Each element is independent — a direct parallel reduction.

```cpp
#include <oneapi/tbb.h>
#include <iostream>
#include <vector>

int main() {
    const int N = 10'000'000;

    // Simulated model predictions and ground-truth targets
    std::vector<float> pred(N), target(N);
    for (int i = 0; i < N; ++i) {
        pred[i]   = static_cast<float>(i % 100) / 100.0f;        // [0.00, 0.99]
        target[i] = static_cast<float>((i + 1) % 100) / 100.0f;  // [0.01, ..., 0.99, 0.00]
    }

    // parallel_reduce arguments:
    //   1. range    — what to iterate over
    //   2. identity — a new partial sum starts at 0
    //   3. body     — reduce a subrange into a partial sum
    //   4. combine  — merge two partial sums into one
    double sum_sq = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(0, N),

        0.0,   // double accumulator reduces rounding in a long sum

        [&](const oneapi::tbb::blocked_range<int>& r, double init) {
            for (int i = r.begin(); i < r.end(); ++i) {
                double diff = static_cast<double>(pred[i]) - target[i];
                init += diff * diff;
            }
            return init;
        },

        std::plus<double>{}  // combine
    );

    double mse = sum_sq / N;
    std::cout << "MSE = " << mse << "\n";   // approximately 0.0099
}
```

**How TBB splits and joins the work:**

```
           parallel_reduce(blocked_range(0, 10M), 0.0, body, combine)
                              [0 … 10,000,000)
                             /                \
                            /                  \
                [0 … 5,000,000)          [5,000,000 … 10M)
               /               \         /               \
       [0…2.5M)           [2.5M…5M)  [5M…7.5M)     [7.5M…10M)
       sum=24,750        sum=24,750 sum=24,750   sum=24,750
              \               /         \               /
           combine(a,b) = a+b           combine(a,b) = a+b
               sum=49,500                   sum=49,500
                          \                 /
                           combine(a,b) = a+b
                               sum=99,000
                           MSE = 99,000 / 10M = 0.0099
```

The diagram is an illustrative four-leaf tree; the scheduler may split differently. In every 100 inputs, 99 differences have magnitude 0.01 and one has magnitude 0.99, so the exact decimal MSE is `(99 × 0.01² + 0.99²) / 100 = 0.0099`. Binary floating-point inputs give a nearby result. The reduction identity initializes new partial sums; a body can receive an accumulated partial, so it must preserve its `init` argument.

---

### 2.3 `parallel_scan` — Prefix Sum

A prefix sum (scan) gives every output element the running total up to that point:

```
input:  [ 1,  2,  3,  4,  5 ]
output: [ 1,  3,  6, 10, 15 ]
         ↑   ↑   ↑   ↑   ↑
         1  1+2 1+2+3 ...
```

Naively this is sequential — each output depends on the previous one. `parallel_scan` breaks it into two passes so chunks run in parallel:

```
Pass 1 — "pre-scan" (is_final = false):
  Chunk A [0..2]: partial sum = 1+2+3 = 6       (don't write output yet)
  Chunk B [3..4]: partial sum = 4+5   = 9       (don't write output yet)

Pass 2 — "final scan" (is_final = true):
  Chunk A [0..2]: carry-in = 0  → write 1, 3, 6
  Chunk B [3..4]: carry-in = 6  → write 10, 15

Combine: TBB calls combine(6, 9) = 15 to pass carry-in to Chunk B
```

A body can run in a preliminary or final scan. The implementation may omit preliminary work when it already knows the prefix; do not assume exactly two calls per chunk or attach unrelated side effects to invocation counts. Write output only when `is_final` is true. [Scan contract](https://uxlfoundation.github.io/oneTBB/main/reference/source/algorithms/functions/parallel_scan_func.html)

```cpp
#include "oneapi/tbb/parallel_scan.h"

std::vector<float> in(N), out(N);

tbb::parallel_scan(
    tbb::blocked_range<int>(0, N),
    0.0f,                   // identity value (carry-in for first chunk)
    [&](const tbb::blocked_range<int>& r, float running, bool is_final) {
        for (int i = r.begin(); i < r.end(); i++) {
            running += in[i];
            if (is_final)
                out[i] = running;   // only write on second pass
        }
        return running;             // hand partial sum to TBB
    },
    [](float a, float b) { return a + b; }   // how to merge two partial sums
);
// out[i] = in[0] + in[1] + ... + in[i]
```

**Inclusive vs exclusive scan:**

```
inclusive (the body above adds before writing):
  input:  [ 1,  2,  3,  4,  5 ]
  output: [ 1,  3,  6, 10, 15 ]   out[i] includes in[i]

exclusive (shifted by one, out[0] = identity):
  input:  [ 1,  2,  3,  4,  5 ]
  output: [ 0,  1,  3,  6, 10 ]   out[i] excludes in[i]
```

Exclusive scan: write the carry value **before** adding the current element. The body chooses inclusive or exclusive behavior:

```cpp
tbb::parallel_scan(
    tbb::blocked_range<int>(0, N), 0.0f,
    [&](const tbb::blocked_range<int>& r, float running, bool is_final) {
        for (int i = r.begin(); i < r.end(); i++) {
            if (is_final)
                out[i] = running;   // write before adding — exclusive
            running += in[i];
        }
        return running;
    },
    [](float a, float b) { return a + b; }
);
```

**Stream compaction — remove zeros in parallel:**

Stream compaction filters an array in three steps, all parallelisable:

```
input:    [ 1,  0,  2,  0,  3,  0,  4 ]

step 1 — flag non-zero elements:
flags:    [ 1,  0,  1,  0,  1,  0,  1 ]

step 2 — exclusive scan on flags → output indices:
indices:  [ 0,  1,  1,  2,  2,  3,  3 ]

step 3 — scatter: if flag[i], write input[i] to output[indices[i]]:
output:   [ 1,  2,  3,  4 ]
```

```cpp
const int N = 7;
std::vector<int>  input  = {1, 0, 2, 0, 3, 0, 4};
std::vector<int>  flags(N), indices(N), output(N);

// Step 1: flag non-zero elements (parallel_for)
tbb::parallel_for(0, N, [&](int i) {
    flags[i] = (input[i] != 0) ? 1 : 0;
});

// Step 2: exclusive scan on flags → scatter indices
tbb::parallel_scan(
    tbb::blocked_range<int>(0, N), 0,
    [&](const tbb::blocked_range<int>& r, int running, bool is_final) {
        for (int i = r.begin(); i < r.end(); i++) {
            if (is_final) indices[i] = running;
            running += flags[i];
        }
        return running;
    },
    [](int a, int b) { return a + b; }
);

// Step 3: scatter non-zero elements to their computed positions (parallel_for)
tbb::parallel_for(0, N, [&](int i) {
    if (flags[i]) output[indices[i]] = input[i];
});
const int kept = indices.back() + flags.back();  // this example has N > 0
output.resize(kept);
// output = [1, 2, 3, 4]; without resize, the vector still has N elements
```

**Prefix max — running maximum:**

```cpp
// out[i] = max(in[0], in[1], ..., in[i])
tbb::parallel_scan(
    tbb::blocked_range<int>(0, N),
    std::numeric_limits<float>::lowest(),   // identity for max
    [&](const tbb::blocked_range<int>& r, float running, bool is_final) {
        for (int i = r.begin(); i < r.end(); i++) {
            running = std::max(running, in[i]);
            if (is_final) out[i] = running;
        }
        return running;
    },
    [](float a, float b) { return std::max(a, b); }
);
```

```
input:  [ 2,  1,  5,  3,  4 ]
output: [ 2,  2,  5,  5,  5 ]   running maximum
```

**Histogram → CDF (cumulative distribution function):**

```cpp
// hist[i] = count of pixels with brightness i
// cdf[i]  = total pixels with brightness ≤ i
// Used in histogram equalization to stretch image contrast.
tbb::parallel_scan(
    tbb::blocked_range<int>(0, 256), 0,
    [&](const tbb::blocked_range<int>& r, int running, bool is_final) {
        for (int i = r.begin(); i < r.end(); i++) {
            running += hist[i];
            if (is_final) cdf[i] = running;
        }
        return running;
    },
    [](int a, int b) { return a + b; }
);
```

**`parallel_scan` works for any associative operation:**

| Operation | Identity | Combine | Use case |
|-----------|----------|---------|---------|
| Sum | `0` | `a + b` | Prefix sum, CDF |
| Max | `-∞` | `max(a,b)` | Running maximum, bounding box |
| Min | `+∞` | `min(a,b)` | Running minimum |
| Product | `1` | `a * b` | Running factorial, probability chain |
| Exclusive sum | `0` | `a + b` | Stream compaction scatter indices |

---

### 2.4 `parallel_sort`

```cpp
#include "oneapi/tbb/parallel_sort.h"

std::vector<int> data(N);
// ...fill data...

// In-place sort (like std::sort but parallel)
parallel_sort(data.begin(), data.end());

// Custom comparator
parallel_sort(data.begin(), data.end(), std::greater<int>{});

// Sort struct by field
parallel_sort(records.begin(), records.end(),
    [](const Record& a, const Record& b) {
        return a.score > b.score;
    }
);
```

Compare `parallel_sort` with `std::sort` using the same data and comparator. Size, value distribution, comparator cost, memory behavior, and thread budget determine the crossover; no fixed speedup applies.

> **Note:** `parallel_sort` is **not stable** (equal elements may reorder). Use `std::stable_sort` for stable ordering (no TBB parallel version exists for stable sort).

---

### 2.5 `parallel_for_each` — Unknown Iteration Space

For containers without random-access iterators (linked lists, sets) or when the iteration space grows during execution:

```cpp
std::list<WorkItem> items = get_work();

parallel_for_each(items.begin(), items.end(),
    [](WorkItem& item) {
        process(item);
    }
);
```

**Dynamic work addition with feeder** — unordered tree traversal:

```cpp
parallel_for_each(roots.begin(), roots.end(),
    [](Node* node, feeder<Node*>& f) {
        process(node);
        for (Node* child : node->children)
            f.add(child);             // adds work dynamically to the pool
    }
);
// Runs until all submitted work completes, including dynamically added items
```

---

The feeder does not provide breadth-first level ordering or a shortest-path guarantee. This example assumes disjoint acyclic trees, live node storage, and a thread-safe `process`. A general graph needs a synchronized visited/claim policy to handle repeated nodes and cycles. Do not structurally modify the seed container while the algorithm is iterating it.

### 2.6 `parallel_pipeline` — Assembly Line

A pipeline is a sequence of stages where each item flows through all stages in order. Unlike `parallel_for` where all items do the same thing, a pipeline lets **different items be at different stages simultaneously**.

**Without pipeline — one item at a time:**

```
time →
item 1: [read]──[process]──[write]
item 2:                            [read]──[process]──[write]
item 3:                                               [read]──[process]──[write]
```

**With pipeline — multiple items in-flight:**

```
time →
item 1: [read]──[process₁][process₂][process₃]──[write]
item 2:         [read]────[process₁][process₂][process₃]──[write]
item 3:                   [read]────[process₁][process₂][process₃]──[write]
item 4:                             [read]─────────── ...
         ↑serial↑         ↑──── parallel stage, N items at once ────↑↑serial↑
```

The read and write stages are **serial** (one item at a time — file I/O or ordered output). The processing stage is **parallel** (CPU-bound work, items are independent).

**Filter modes:**

| Mode | Order | Concurrency | Use when |
|------|-------|-------------|---------|
| `serial_in_order` | Preserved | 1 at a time | File I/O, ordered output |
| `serial_out_of_order` | Not preserved | 1 at a time | Single-threaded setup/teardown |
| `parallel` | Not preserved | Multiple threads | CPU-bound, independent items |

**Structure (application sketch):**

Define `InputData`, `OutputData`, and the three application functions below. `read_next()` returns an owning shared pointer or an empty pointer at end of input; `transform()` returns an owning output pointer. Ownership survives transfer between filters and is released when each message is no longer needed.

```cpp
#include <oneapi/tbb/parallel_pipeline.h>
#include <memory>

using Input = std::shared_ptr<const InputData>;
using Output = std::shared_ptr<OutputData>;
const int max_tokens = 16;

parallel_pipeline(max_tokens,
    make_filter<void, Input>(filter_mode::serial_in_order,
        [&](flow_control& fc) -> Input {
            Input data = read_next();
            if (!data) fc.stop();
            return data;
        }
    ) &
    make_filter<Input, Output>(filter_mode::parallel,
        [](Input in) -> Output { return transform(*in); }
    ) &
    make_filter<Output, void>(filter_mode::serial_in_order,
        [&](Output out) { write_result(*out); }
    )
);
```

**Batch inference architecture:**

The following body needs application implementations of `load_batch_from_disk`, `run_inference`, and `write_predictions`. These names do not invoke an actual model in the course verifier; its harness uses deterministic synthetic values. Parallel inference is appropriate only if the model/runtime context supports concurrent calls or each task has independent context. A serial I/O stage is a choice for this example, not a rule that disks only support sequential access.

```cpp
struct Batch { std::vector<float> data; int id; };
using BatchPtr = std::shared_ptr<Batch>;
int batch_id = 0;
const int TOTAL = 100;

parallel_pipeline(/*max_tokens=*/8,
    make_filter<void, BatchPtr>(filter_mode::serial_in_order,
        [&](flow_control& fc) -> BatchPtr {
            if (batch_id >= TOTAL) { fc.stop(); return {}; }
            auto b = std::make_shared<Batch>();
            b->id = batch_id++;
            b->data = load_batch_from_disk(b->id);
            return b;
        }
    ) &
    make_filter<BatchPtr, BatchPtr>(filter_mode::parallel,
        [](BatchPtr b) {
            for (float& v : b->data) v /= 255.0f;
            return b;
        }
    ) &
    make_filter<BatchPtr, BatchPtr>(filter_mode::parallel,
        [](BatchPtr b) {
            run_inference(b->data);
            return b;
        }
    ) &
    make_filter<BatchPtr, void>(filter_mode::serial_in_order,
        [&](BatchPtr b) { write_predictions(b->id, b->data); }
    )
);
```

The message owns its buffer through the final stage. Smart pointers prevent the raw-pointer leaks in a cancellation/exception path, but they do not undo completed output writes. Define whether partial files/results are valid before retrying a failed pipeline. Blocking disk work also occupies a participating thread; use an asynchronous I/O design when that becomes the measured bottleneck.

**`max_tokens` and the throughput law:**

```
max_tokens too small:
  [read]──[process]──[write]
          idle  idle           ← CPU starved, pipeline stalls waiting for tokens

max_tokens right:
  [read]──[p1][p2][p3][p4]──[write]
           ↑ all threads busy ↑

max_tokens too large:
  [read]──[p1][p2]...[p100]──[write]
                 ↑ 100 batches buffered in memory → OOM risk
```

> **A serial-stage bound:** a stage taking 10 ms per item can process at most 100 items/s, even with eight live tokens. More tokens allow stages to overlap; they do not parallelize a serial stage. A useful ideal bound is `throughput <= min(each stage's service capacity, max_tokens / end-to-end latency)`, with consistent units. Scheduler overhead and shared-resource contention can reduce the achieved rate further.

---

### 2.7 `parallel_invoke` and `task_group` — Explicit Tasks

**`parallel_invoke` — fixed, known set of tasks:**

```
parallel_invoke(f1, f2, f3, f4)

  caller ──┬──► T0: f1() ──┐
           ├──► T1: f2() ──┤
           ├──► T2: f3() ──┤ implicit join — caller blocks until all done
           └──► T3: f4() ──┘
```

```cpp
#include "oneapi/tbb/parallel_invoke.h"

// Two tasks — classic divide-and-conquer split
parallel_invoke(
    []{ sort_left_half(); },
    []{ sort_right_half(); }
);
// Both completed after a normal return (see cancellation/exception rules below)

// Any fixed number of callables
parallel_invoke(
    []{ build_index(); },
    []{ load_weights(); },
    []{ warm_up_cache(); }
);
```

Use `parallel_invoke` when the number of parallel operations is known at compile time and they are independent.

**`task_group` — dynamic, runtime-determined tasks:**

```
task_group tg;
tg.run(f1);   tg.run(f2);   tg.run(f3);   // enqueue at runtime
...           // tasks may themselves call tg.run() — recursive OK
tg.wait();    // caller blocks until all tasks (and their children) complete
```

```cpp
#include "oneapi/tbb/task_group.h"

// Dynamic number of tasks — count not known until runtime
task_group tg;
for (auto& item : work_units)
    tg.run([&item]{ process(item); });
tg.wait();

// Recursive — tasks spawning more tasks
task_group tg2;
tg2.run([&]{
    tg2.run([&]{ subtask_a(); });
    tg2.run([&]{ subtask_b(); });
});
tg2.wait();
```

**`parallel_invoke` vs `task_group`:**

| | `parallel_invoke` | `task_group` |
|--|-------------------|-------------|
| Number of tasks | Fixed at compile time | Dynamic, determined at runtime |
| Recursive tasks | Yes — a callable can recursively invoke another fixed group | Yes — tasks can add work with `tg.run()` |
| Syntax | One call, all lambdas inline | `run()` / `wait()` separate |
| Overhead | Depends on task size and nesting; measure | Depends on task size and nesting; measure |
| Use case | Merge sort split, dual data load | Tree traversal, work queue, graph |

**When to use which:**
- Know the exact tasks upfront? → `parallel_invoke`
- Tasks discovered at runtime or spawn subtasks? → `task_group`
- Binary recursive split? → `parallel_invoke` is suitable (see merge sort in §2.17). Dynamically discovered children? → `task_group`; explicit dependencies? → a flow graph.

During a scheduler wait, the calling thread may execute available tasks. A wait is a completion boundary, not a promise that the OS thread remains idle.

---

### 2.8 Per-Thread Storage — `enumerable_thread_specific`

When multiple threads update the same data structure (histogram, accumulator, buffer), compare these choices:

| Approach | Mechanism | Problem |
|----------|-----------|---------|
| Shared variable | Nothing | Data race — undefined behavior |
| Mutex | `std::mutex` + lock | Conflicting updates serialize; keep the protected work short |
| `atomic` | Atomic access to a supported type | Does not make a whole multi-element algorithm atomic; lock-free support depends on type/platform |
| **ETS** | **Per-thread copy** | **Avoids shared histogram updates; pays for lookup, storage, allocation, and merging** |

ETS provides a thread-local value on demand. This histogram example only updates the calling thread's copy, then merges after parallel work completes. Thread-local is not task-local: nested scheduler waits may execute other work on the same thread and reuse that value; see §2.16 before retaining task-specific state across a nested call.

```
parallel phase:                          merge phase:
  T0 → local_hist[0] ──────────────────► global_hist
  T1 → local_hist[1] ──────────────────►     +=
  T2 → local_hist[2] ──────────────────►     +=
  T3 → local_hist[3] ──────────────────►     +=
  (no shared-bin updates)               (single-threaded, once)
```

**Histogram — wrong vs right:**

```cpp
// WRONG: data race — multiple threads increment the same bin
std::vector<int> hist(256, 0);
tbb::parallel_for(0, N, [&](int i) {
    hist[image[i]]++;   // T0 and T1 may both read 5, both write 6 → lost update
});

// RIGHT: each thread has its own histogram, merge once at the end
#include "oneapi/tbb/enumerable_thread_specific.h"

enumerable_thread_specific<std::vector<int>> local_hist(
    []{ return std::vector<int>(256, 0); }  // factory called once per thread
);

tbb::parallel_for(0, N, [&](int i) {
    local_hist.local()[image[i]]++;   // .local() returns THIS thread's copy
});

std::vector<int> global_hist(256, 0);
for (auto& h : local_hist)           // one entry per thread that called .local()
    for (int k = 0; k < 256; k++)
        global_hist[k] += h[k];
```

**Thread-local sum:**

```cpp
enumerable_thread_specific<float> thread_sum(0.0f);

tbb::parallel_for(tbb::blocked_range<int>(0, N),
    [&](const tbb::blocked_range<int>& r) {
        float& my = thread_sum.local();
        for (int i = r.begin(); i < r.end(); i++)
            my += a[i];
    }
);

float total = 0.0f;
for (float s : thread_sum) total += s;
```

**Construction options:**

```cpp
enumerable_thread_specific<int> ets1;         // default-constructed: int{} = 0
enumerable_thread_specific<int> ets2(42);     // value-initialized: each copy = 42

// Factory for non-trivial types (vector, map, custom objects)
enumerable_thread_specific<std::vector<int>> ets3(
    []{ return std::vector<int>(1024, 0); }   // called once per thread on first .local()
);
```

**Key mechanics:**
- `.local()` creates the copy on first call for that thread, returns a reference on subsequent calls
- Range-for over an ETS object iterates over all copies that were actually created
- If a thread never calls `.local()`, no copy is created for it — no wasted memory

---

### 2.9 `combinable<T>` — Simpler Per-Thread Accumulation

A simplified version of `enumerable_thread_specific` specifically for accumulating a single value:

```cpp
#include "oneapi/tbb/combinable.h"

combinable<float> partial_sum;

parallel_for(0, N, [&](int i) {
    partial_sum.local() += a[i];   // thread-local accumulate
});

// Combine all locals with a binary op
float total = partial_sum.combine([](float a, float b) {
    return a + b;
});
```

**vs `enumerable_thread_specific`:**
- `combinable<T>` — simpler API, designed specifically for reduction patterns
- `enumerable_thread_specific<T>` — more flexible, supports iteration, factory init, non-trivial types

Use `combinable` when you just want a thread-local accumulator. Use `enumerable_thread_specific` when you need to inspect each thread's value separately after the parallel phase.

---

### 2.10 Flow Graph — Data Flow and Dependence Graphs

For expressing parallel patterns as a graph of nodes and edges. Edges provide the declared message/dependency relationships. Shared mutable state outside those relationships still needs ownership, atomics, or locks; a graph is not a general race-prevention mechanism.

```
Node   = a worker (function, buffer, join, split, ...)
Edge   = a channel that carries data tokens between nodes
Token  = one unit of data flowing through the graph

            try_put(5)
                │
         ┌──────┴──────┐
         ▼             ▼
   ┌──────────┐   ┌──────────┐   ← run in parallel (unlimited concurrency)
   │  square  │   │   cube   │
   │  x → x² │   │  x → x³  │
   └────┬─────┘   └────┬─────┘
        │ port 0        │ port 1
        └──────┬────────┘
               ▼
         ┌──────────┐             ← waits for one token on each port
         │   join   │
         └────┬─────┘
              ▼
         ┌──────────┐             ← serial (concurrency = 1)
         │  printer │
         │ (sq, cu) │
         └──────────┘
```

```cpp
#include "oneapi/tbb/flow_graph.h"
using namespace oneapi::tbb::flow;

graph g;

// function_node<In, Out>: receives In, produces Out
// Second arg = max concurrency (1=serial, unlimited=max)
function_node<int, int> square(g, unlimited, [](int x) { return x * x; });
function_node<int, int> cube  (g, unlimited, [](int x) { return x * x * x; });

// join_node: wait for one input per port, emit as tuple
join_node<std::tuple<int,int>> join(g);

function_node<std::tuple<int,int>, continue_msg> printer(g, 1,
    [](const std::tuple<int,int>& t) {
        printf("square=%d  cube=%d\n", std::get<0>(t), std::get<1>(t));
        return continue_msg{};
    }
);

// Wire the graph
make_edge(square, input_port<0>(join));
make_edge(cube,   input_port<1>(join));
make_edge(join,   printer);

// Send inputs — both nodes run in parallel
square.try_put(5);
cube.try_put(5);

g.wait_for_all();  // always wait before graph goes out of scope
```

Terminal `function_node` bodies above return `continue_msg`, a small completion message; `function_node<..., void>` is not a valid replacement. The `void` source/sink types in `parallel_pipeline` belong to a different API. See the [function-node contract](https://uxlfoundation.github.io/oneTBB/main/reference/source/flow_graph/function_node_cls.html).

**ML example — parallel feature extraction with fan-out/join:**

This architecture sketch assumes unique frame IDs and application-provided extraction/scoring functions. Extract three features independently, then join matching IDs. Add a limiter before broadcast for a bounded streaming application; the executable example below shows the full ownership and completion path:

```
                    ┌─────────────────┐
         frame ──►  │  broadcast_node │
                    └────┬───────┬────┘
                         │       │       │
                    ┌────┴──┐ ┌──┴───┐ ┌─┴─────┐
                    │  HOG  │ │  LBP │ │  DCT  │   ← parallel feature extractors
                    └────┬──┘ └──┬───┘ └─┬─────┘
                         └───────┴────────┘
                                 │
                           ┌─────┴─────┐
                           │   join    │           ← wait for all three features
                           └─────┬─────┘
                                 │
                           ┌─────┴─────┐
                           │  scorer   │           ← combine features → score
                           └───────────┘
```

```cpp
struct Frame  { int id; std::vector<float> data; };
struct Feature { int frame_id; std::vector<float> data; };
struct Score  { float value; };

graph g;

// Fan-out: send each frame to all three extractors simultaneously
broadcast_node<Frame> broadcast(g);

function_node<Frame, Feature> hog(g, unlimited,
    [](const Frame& f) { return Feature{f.id, extract_hog(f)}; });
function_node<Frame, Feature> lbp(g, unlimited,
    [](const Frame& f) { return Feature{f.id, extract_lbp(f)}; });
function_node<Frame, Feature> dct(g, unlimited,
    [](const Frame& f) { return Feature{f.id, extract_dct(f)}; });

// Fan-in: wait for all three features before scoring
using FeatureTuple = std::tuple<Feature, Feature, Feature>;
join_node<FeatureTuple, key_matching<int>> join(g,
    [](const Feature& f) { return f.frame_id; },
    [](const Feature& f) { return f.frame_id; },
    [](const Feature& f) { return f.frame_id; });

function_node<FeatureTuple, Score> scorer(g, unlimited,
    [](const FeatureTuple& t) {
        return score(std::get<0>(t).data, std::get<1>(t).data, std::get<2>(t).data);
    }
);

// Wire
make_edge(broadcast, hog);
make_edge(broadcast, lbp);
make_edge(broadcast, dct);
make_edge(hog, input_port<0>(join));
make_edge(lbp, input_port<1>(join));
make_edge(dct, input_port<2>(join));
make_edge(join, scorer);

// Process frames
for (auto& frame : frames)
    broadcast.try_put(frame);

g.wait_for_all();
```

#### All Key Node Types

| Node | Description |
|------|-------------|
| `function_node<In, Out>` | Transforms `In` → `Out`, configurable concurrency |
| `input_node<Out>` | Generates messages after `activate()`; replaces the removed legacy `source_node` API |
| `broadcast_node<T>` | Fans out: sends one input to ALL successors |
| `join_node<tuple<...>>` | Fans in: waits for one from each port, emits tuple |
| `split_node<tuple<...>>` | Splits tuple: sends element N to port N |
| `buffer_node<T>` | Buffers messages until a successor requests them |
| `queue_node<T>` | FIFO buffer; passes messages to any available successor |
| `priority_queue_node<T>` | Priority-ordered buffer |
| `sequencer_node<T>` | Reorders out-of-order items back to sequence number order |
| `limiter_node<T>` | Limits in-flight messages (backpressure) |
| `overwrite_node<T>` | Stores last value; new successors get it immediately |
| `write_once_node<T>` | Stores first value; broadcasts to all registered receivers |
| `indexer_node<T...>` | Tagged union input: accepts any type from set, tags messages |

#### Fan-Out with `broadcast_node`

```cpp
broadcast_node<int> broadcaster(g);
function_node<int, continue_msg> worker_a(g, unlimited, [](int x){
    do_a(x); return continue_msg{};
});
function_node<int, continue_msg> worker_b(g, unlimited, [](int x){
    do_b(x); return continue_msg{};
});

make_edge(broadcaster, worker_a);
make_edge(broadcaster, worker_b);  // same input goes to both

broadcaster.try_put(42);  // both worker_a and worker_b receive 42
g.wait_for_all();
```

#### Reordering with `sequencer_node`

```cpp
// parallel_for may complete items out of order — use sequencer to restore order
sequencer_node<Frame> reorder(g, [](const Frame& f) {
    return f.sequence_number;   // sequencer uses this to order outputs
});

function_node<Frame, Frame> processor(g, unlimited, [](Frame f) {
    f.data = process(f.data);   // runs in parallel, out of order
    return f;
});

function_node<Frame, continue_msg> writer(g, serial, [](const Frame& f) {
    write_in_order(f);          // receives frames in sequence order
    return continue_msg{};
});

make_edge(processor, reorder);
make_edge(reorder, writer);
```

#### Backpressure with `limiter_node`

```cpp
// Prevent fast producer from overwhelming slow consumer
limiter_node<int> limiter(g, 8);   // max 8 messages in flight

make_edge(producer, limiter);
make_edge(limiter, slow_consumer);
make_edge(slow_consumer, limiter.decrementer()); // signal when done → unlocks limiter
```

#### Conditional Routing with `indexer_node`

```cpp
using IndexerType = indexer_node<int, float>;   // accepts int OR float

IndexerType idx(g);
function_node<IndexerType::output_type, continue_msg> router(g, unlimited,
    [](const IndexerType::output_type& msg) {
        if (msg.tag() == 0)       // int arrived
            handle_int(cast_to<int>(msg));
        else                       // float arrived
            handle_float(cast_to<float>(msg));
        return continue_msg{};
    }
);

make_edge(idx, router);
input_port<0>(idx).try_put(42);     // send int
input_port<1>(idx).try_put(3.14f);  // send float
```

#### Worked program: preserve frame identity through parallel branches

The runnable [video_pipeline_template.cpp](video_pipeline_template.cpp) generates synthetic 64-element frames, computes their sum and maximum in separate branches, combines the results, and restores output order. It is an executable concurrency example, with no camera, trained model, or GPU inference.

```text
source -> limiter -> accounting -> broadcast -> sum branch --+
                                           -> max branch --+-> keyed join
output <- sequencer <- combine <----------------------------+
  |
  +-> limiter.decrementer()   (one completed logical frame releases one slot)
```

```bash
g++ -O2 -std=c++17 -pthread -Wall -Wextra video_pipeline_template.cpp -ltbb -o /tmp/video_pipeline
/tmp/video_pipeline --self-test
/tmp/video_pipeline 20 4 2
```

Arguments are frame count, maximum accepted in-flight frames, and thread budget. The source allocates each frame when requested; a shared pointer to immutable pixel storage avoids copying pixel buffers for each branch. The limiter goes **before** branching and expensive work. Its threshold bounds accepted logical frames, not all resident memory: the input node can cache one pending item, and graph metadata and references still take space.

**Why join by key?** Suppose branch A completes frame IDs `[0,1]` and branch B completes `[1,0]`. A default queueing join pairs the first arrival on each port: `(0,1)` and `(1,0)`. Neither pair describes one frame. Use `join_node<Features, key_matching<int>>` with `frame.id` extractors. A sequencer after the join cannot repair already mismatched inputs. The self-test submits exactly this reversed order and confirms both that key matching preserves identity and that the FIFO-join counterexample creates two mismatches. See the [oneTBB join policies](https://uxlfoundation.github.io/oneTBB/main/reference/source/flow_graph/join_node_policies.html).

**Why release at output?** Returning the limiter token only after the serial output checks the result bounds all accepted work through completion. Releasing it after preprocessing would permit later stages to accumulate more frames. The API is `limiter.decrementer()`, as documented in the [limiter reference](https://uxlfoundation.github.io/oneTBB/main/reference/source/flow_graph/limiter_node_cls.html).

The program checks every output ID, synthetic numerical result, peak logical occupancy, and final count. Executed cases cover 0, 1, 17, and 100 frames with limits 1 and 4; the regression suite also checks progress with one worker. Increasing the limit can improve overlap, but cannot make a permanently slow serial sink run faster.

**Robotics extension:** carry capture time and reset epoch with each frame. Decide whether the output requires every frame in order or the freshest result. A strict sequencer can hold later results behind one delayed item; a freshness policy must explicitly account for dropped IDs so it does not wait forever for an item intentionally removed.

---

### 2.11 Concurrent Containers

**The problem:** operations that mutate shared container structure, such as concurrent `push_back`, require synchronization. Concurrent reads are different, and threads can modify distinct existing elements of many standard containers (except packed `vector<bool>` elements) when no structural modification or conflicting access occurs. State the exact access pattern before choosing a concurrent container.

```
std::vector<int> v;                // NOT thread-safe
parallel_for(0, N, [&](int i) {
    v.push_back(i);                // ← data race: undefined behaviour
});
```

The naive fix — wrapping everything in a `std::mutex` — works but serialises every access. TBB's concurrent containers solve this with fine-grained internal locking or lock-free algorithms:

```
std::vector + one global mutex     tbb::concurrent_vector
  Thread 0 writes → LOCK            Thread 0 writes ──┐
  Thread 1 waits...                  Thread 1 writes ──┤ concurrent operations
  Thread 2 waits...                  Thread 2 writes ──┘
  (serialised)                       (concurrent, safe)
```

> Only use concurrent containers when multiple threads genuinely need to share the same container. If each thread works on its own data, consider a plain `std::vector` per thread (or `enumerable_thread_specific`), and measure allocation and merge costs.

#### `concurrent_vector`

Like `std::vector` but safe for concurrent `push_back`. Iterators and references stay valid after growth — unlike `std::vector` which may reallocate.

```cpp
#include "oneapi/tbb/concurrent_vector.h"

tbb::concurrent_vector<int> results;

tbb::parallel_for(0, N, [&](int i) {
    if (passes_filter(i))
        results.push_back(compute(i));   // safe from any thread
});
// results holds all passing items; order is non-deterministic
```

When to use: collecting results from a parallel filter where you don't know how many items will pass. Concurrent growth does not publish a ready-to-read payload: `size()` may include elements still under construction. Wait for producers to finish, as this `parallel_for` does, or provide explicit per-element publication. Elements need not be contiguous; growth does not make concurrent clear/destruction safe. [Vector growth and readiness](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/concurrent_vector_ug.html)

#### `concurrent_queue` / `concurrent_bounded_queue`

Classic producer-consumer queue. `try_pop` returns false rather than waiting for an item when empty; this is not a blanket lock-free or wait-free progress guarantee. `pop` on `concurrent_bounded_queue` waits for an item.

```cpp
#include "oneapi/tbb/concurrent_queue.h"

tbb::concurrent_queue<WorkItem> q;

// Producer threads — can all push simultaneously
q.push(item_a);
q.push(item_b);

// Consumer threads — non-blocking
WorkItem item;
if (q.try_pop(item))
    process(item);   // got one

// Bounded variant — adds backpressure (push blocks when full)
tbb::concurrent_bounded_queue<WorkItem> bounded;
bounded.set_capacity(100);
bounded.push(item);   // blocks if 100 items already queued
bounded.pop(item);    // blocks if queue is empty
```

When to use: producers and consumers with independently available execution capacity. Blocking every oneTBB participant on an empty queue while its producers remain queued can prevent progress. Prefer the flow graph above for scheduler-managed dependencies, or use independently managed I/O threads with an explicit shutdown policy.

#### `concurrent_hash_map`

A concurrent key-value container with accessors that protect element access. Keep accessor lifetimes short. Different keys can still contend through hash buckets, resizing, allocation, or shared resources; the API does not guarantee independent timing. [Accessor contract](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/concurrent_hash_map.html)

```cpp
#include "oneapi/tbb/concurrent_hash_map.h"

tbb::concurrent_hash_map<std::string, int> freq;

tbb::parallel_for_each(words.begin(), words.end(), [&](const std::string& w) {
    tbb::concurrent_hash_map<std::string, int>::accessor acc;  // write lock
    freq.insert(acc, w);   // acquires exclusive access to this element
    acc->second++;         // protected for the lifetime of this accessor
});                        // accessor destructor releases the lock

// Read-only (shared lock — many readers can hold this simultaneously)
tbb::concurrent_hash_map<std::string, int>::const_accessor cacc;
if (freq.find(cacc, "hello"))
    printf("hello: %d\n", cacc->second);
```

`accessor` = write lock (exclusive). `const_accessor` = read lock (shared). Multiple threads can hold `const_accessor` on the same key simultaneously.

#### `concurrent_unordered_map`

A familiar unordered-map interface with concurrent insertion, lookup, and supported traversal. It is not a blanket thread-safe replacement for every `std::unordered_map` operation: erasure/clear have separate concurrency restrictions, and mutating a mapped value still needs synchronization. Concurrent traversal is not a snapshot and need not observe every simultaneous insertion. [Operation-specific contract](https://uxlfoundation.github.io/oneTBB/main/reference/source/containers/concurrent_unordered_map_cls.html)

```cpp
#include "oneapi/tbb/concurrent_unordered_map.h"

tbb::concurrent_unordered_map<int, long long> m;

tbb::parallel_for(0, N, [&](int i) {
    m.insert({i, 1LL * i * i});  // promote before multiplication to avoid int overflow
    // operator[] can insert, but its returned value is not locked.
    // Concurrent assignments to the same key still race.
});
```

Use `concurrent_hash_map` when you need read-modify-write atomicity (e.g., incrementing a counter). Use `concurrent_unordered_map` when you only insert-or-lookup with no partial updates.

#### `concurrent_priority_queue`

```cpp
#include "oneapi/tbb/concurrent_priority_queue.h"

tbb::concurrent_priority_queue<int> pq;
pq.push(10); pq.push(5); pq.push(20);

int top;
pq.try_pop(top);   // top = 20 (max-heap by default)
```

#### Container comparison

| Container | Analogy | Key operation | Use for |
|-----------|---------|---------------|---------|
| `concurrent_vector` | Shared notepad | `push_back` | Collecting parallel results |
| `concurrent_queue` | Shared inbox | `push` / `try_pop` | Producer-consumer pipelines |
| `concurrent_bounded_queue` | Inbox with size limit | blocking `push`/`pop` | Backpressure control |
| `concurrent_hash_map` | Shared dictionary (with door locks per entry) | `accessor` read-modify-write | Word counts, shared caches |
| `concurrent_unordered_map` | Shared dictionary (simpler) | `insert` / `find` | Insert-once lookup tables |
| `concurrent_priority_queue` | Shared task queue sorted by priority | `push` / `try_pop` | Priority scheduling |

---

### 2.12 Scalable Memory Allocator

Allocation behavior depends on the allocator, workload, and platform. Modern system allocators can already use multiple arenas and thread caches; there is no universal single global `malloc` lock. oneTBB provides a scalable allocator to benchmark when allocation contention matters. The [glibc allocator tunables](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html) document arenas and per-thread caches for one common system allocator.

This complete example uses the explicit scalable allocator for a container and correctly pairs the C allocation API:

```cpp
#include <oneapi/tbb/scalable_allocator.h>
#include <iostream>
#include <new>
#include <vector>

int main() {
    constexpr std::size_t N = 1024;
    std::vector<float, oneapi::tbb::scalable_allocator<float>> vec(N, 1.0f);
    float* buf = static_cast<float*>(scalable_malloc(N * sizeof(float)));
    if (!buf) throw std::bad_alloc{};
    buf[0] = vec[0];
    std::cout << "allocated value=" << buf[0] << "\n";
    scalable_free(buf);  // do not mix this with free/delete
}
```

Compile with `g++ -std=c++17 allocator_example.cpp -ltbbmalloc -o /tmp/allocator_example`. By comparison, `oneapi::tbb::tbb_allocator<T>` requires `<oneapi/tbb/tbb_allocator.h>` and can fall back to `std::malloc` if the scalable library is unavailable; it does not prove which allocator was used. See the [allocator contract](https://uxlfoundation.github.io/oneTBB/main/reference/source/memory_allocation/tbb_allocator_cls.html).

Global replacement through `tbbmalloc_proxy` is a separate, platform-specific deployment choice. Compare latency, allocation volume, memory use, and the serial baseline before changing a process-wide allocator.

---

### 2.13 `task_arena` — Bound participation in an execution region

An arena defines a scheduler region with a concurrency limit. Worker threads can serve different arenas; an arena does not reserve a private set of CPU cores, dedicated memory bandwidth, or a fixed number of permanently owned OS threads.

```cpp
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/parallel_for.h>

oneapi::tbb::task_arena arena(2);
arena.execute([&] {
    oneapi::tbb::parallel_for(0, 1000, [&](int i) {
        do_work(i);  // application function; define it for your workload
    });
});
```

This limits arena participation; it does not imply the OS schedules both participants continuously. Separate arenas help control subsystem concurrency and nested scheduling. They still share CPU caches, worker supply, memory controllers, and other process resources.

NUMA constraints can select an available NUMA node, but node IDs must come from the host's topology. CPU placement does not migrate already allocated data or guarantee local memory: allocation/first-touch policy and page placement need separate checks. A single-participant arena can simplify debugging, but it does not make random inputs, external I/O, unordered iteration, or the rest of the process deterministic.

**Exercise:** compare the existing Fibonacci benchmark with one and two participants. Explain why equal concurrency settings do not prove equal affinity or useful work per thread.

### 2.14 `global_control` — Control oneTBB worker participation

`global_control` applies while its object exists and combines with other active controls. For `max_allowed_parallelism`, the smallest active value takes effect. Destroying one object restores the policy determined by the remaining controls, not necessarily unrestricted execution.

```cpp
#include <oneapi/tbb/global_control.h>

{
    oneapi::tbb::global_control budget(
        oneapi::tbb::global_control::max_allowed_parallelism, 4);
    oneapi::tbb::parallel_for(0, N, body);
}
```

The worker limit is the selected value minus one. This does not count every application thread entering oneTBB. A selected value of 1 disables ordinary worker participation, but an explicit `task_arena::enqueue` has a documented exception allowing a worker to execute enqueued work. Thus even value 1 is not a mutex for two application threads. This is not a process-wide cap on `std::thread`, OpenMP, network workers, or all threads inside third-party libraries. It also cannot reserve PCIe bandwidth for a GPU. [Global-control reference](https://uxlfoundation.github.io/oneTBB/main/reference/source/task_scheduler/scheduling_controls/global_control_cls.html)

`thread_stack_size` controls oneTBB worker stack configuration subject to platform/runtime requirements. It does not enlarge the application's main-thread stack. Prefer bounded recursion and heap storage for large buffers; do not rely on stack overflow producing a clear diagnostic.

### 2.15 Exceptions and cooperative cancellation

A body can request cancellation of its task-group context. The scheduler avoids further eligible work in that context as cancellation propagates; tasks already executing are not forcibly interrupted. A long-running body can inspect the cancellation state and return early. Cancellation is not a transaction: completed writes are not rolled back, and partially produced output needs an explicit validity rule. [Cancellation behavior](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Cancellation_Without_An_Exception.html)

```cpp
oneapi::tbb::task_group_context context;
oneapi::tbb::parallel_for(
    oneapi::tbb::blocked_range<int>(0, N),
    [&](const oneapi::tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i != range.end(); ++i) {
            if (context.is_group_execution_cancelled()) return;
            if (bad_input(i)) {
                context.cancel_group_execution();
                return;
            }
            process(i);
        }
    }, oneapi::tbb::auto_partitioner{}, context);
```

This is an application-body sketch: define `N`, `bad_input`, and `process`, and include the corresponding oneTBB headers. Do not assume the first reported bad index is the smallest index; execution order is not serial. A worker exception requests cancellation and is propagated to the caller after associated work is resolved. Throwing bodies unwind; other running bodies may finish, cooperate with cancellation, or throw themselves. “Every already-running iteration completes successfully” is therefore not a valid guarantee.

**Check:** why is setting a stop flag insufficient to free a shared buffer immediately? Other bodies may still be using it. Wait for completion and apply the ownership contract before reclamation.

### 2.16 Work isolation is a scheduling rule, not a lock

A thread waiting for nested parallel work can execute another outer task. That reentrancy can change thread-local state even without a data race between two OS threads. `this_task_arena::isolate` restricts **the calling thread's task selection** while it runs the supplied region. Other workers can still help execute that region. It does not grant exclusive access to memory or force a deterministic task order. [Work-isolation guide](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/work_isolation.html)

For example, these nested writes use disjoint elements. They do not need isolation to become race-free:

```cpp
#include <oneapi/tbb/parallel_for.h>
#include <vector>

std::vector<int> data(100, 0);
oneapi::tbb::parallel_for(0, 10, [&](int row) {
    oneapi::tbb::parallel_for(0, 10, [&](int column) {
        data[row * 10 + column] = row + column;
    });
});
```

A stolen task keeps its captured arguments; stealing work for row 7 does not magically change its index to row 2. The ownership proof comes from unique `(row, column)` pairs and waiting for the nested call, not from which worker executes each pair.

When thread-local state really must survive a nested wait, isolate that nested region and inspect its reentrancy assumptions:

```cpp
#include <oneapi/tbb/task_arena.h>  // declares this_task_arena::isolate

oneapi::tbb::this_task_arena::isolate([&] {
    oneapi::tbb::parallel_for(0, count, independent_body);
});
```

If two tasks write the same non-atomic value concurrently, adding separate `isolate` calls does not repair the race. Change ownership or use synchronization. The [API reference](https://uxlfoundation.github.io/oneTBB/main/reference/source/task_scheduler/task_arena/this_task_arena_ns.html) defines the calling-thread scope; it supplies no mutual-exclusion or transaction guarantee.

The supplied [concurrency contract example](concurrency_contracts.cpp) checks bounded-queue capacity, concurrent element updates, disjoint nested writes, cancellation state, and two overlapping isolated regions. Its overlap check uses synchronization to avoid a data race: it demonstrates that isolation itself is not a mutex.

---

### Run the concurrency contracts

From this guide's directory, with the same oneTBB development files as the Fibonacci lab:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread concurrency_contracts.cpp -ltbb -o /tmp/concurrency_contracts
/tmp/concurrency_contracts --self-test
```

The six checks execute on the CPU and retain independent expected results. The isolation check starts two application threads and waits for both isolated regions to overlap, with shared observations protected by a mutex/condition variable. It deliberately avoids an undefined data race. The queue check verifies `try_push` capacity and FIFO; it does not measure blocking fairness. The cancellation check uses one participating application thread to make the cooperative boundary inspectable.

The [recorded validation](labs/concurrency-validation.json) includes GCC 13.3, oneTBB 2021.11.0 and a separate AddressSanitizer/UndefinedBehaviorSanitizer run. Those sanitizers are not a proof of race freedom. Predict which checks would fail if isolation were mistakenly treated as mutual exclusion, or if cancellation were assumed to undo an already committed write.

The independent [native guide verifier](labs/verify_guide_examples.py) extracts twelve code blocks from this guide and compiles them, supplying explicit inputs where a fragment needs a harness. It checks the MSE arithmetic, minimum tie/empty cases, inclusive/exclusive scans, compaction size, graph messages, ordered pipeline output, allocation, and basic elementwise loops. Run it from this directory after installing oneTBB development files:

```bash
python3 labs/verify_guide_examples.py --single-cpu
```

`--single-cpu` adds a Linux `taskset` progress check; omit it on other platforms. For development files outside system paths, pass `--include-dir PATH --library-dir PATH`. It creates temporary build files and installs nothing. [Independent review evidence](labs/concurrency-review.json) records the actual compiler output and results; it does not claim every application sketch in this guide is standalone or that sanitizer checks prove race freedom.

### 2.17 Pattern Cheat Sheet

**Reduce:**

```cpp
float total = parallel_reduce(range, 0.0f, body, std::plus<float>{});
```

**Divide and Conquer (recursive):**

```cpp
void parallel_mergesort(int* a, int n) {
    if (n < THRESHOLD) { std::sort(a, a + n); return; }
    parallel_invoke(
        [&]{ parallel_mergesort(a, n/2); },
        [&]{ parallel_mergesort(a + n/2, n - n/2); }
    );
    std::inplace_merge(a, a + n/2, a + n);
}
```

**Map (elementwise):**

```cpp
parallel_for(blocked_range<int>(0, N), [&](const blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); i++)
        out[i] = f(in[i]);
});
```

**Histogram (per-thread local + merge):**

```cpp
enumerable_thread_specific<std::vector<int>> local_hist(
    []{ return std::vector<int>(256, 0); });

parallel_for(0, N, [&](int i) {
    local_hist.local()[data[i]]++;
});

std::vector<int> hist(256, 0);
for (auto& h : local_hist)
    for (int k = 0; k < 256; k++) hist[k] += h[k];
```

**Prefix scan:**

```cpp
parallel_scan(range, 0.0f, scan_body, std::plus<float>{});
```

**Assembly line:**

```cpp
parallel_pipeline(max_tokens, input_filter & transform_filter & output_filter);
```

---

### 2.18 Connection to GPU Programming

The decomposition questions carry over, but the APIs and execution guarantees differ:

| Design question | CPU oneTBB example | GPU distinction to learn |
|-----------------|--------------------|--------------------------|
| Which outputs are independent? | `parallel_for` over disjoint elements | A CUDA kernel maps work to threads, blocks, and a grid; its SIMT and memory-access rules are different |
| Where should partial results live? | Thread-local histogram followed by merge | Registers, block shared memory, and global memory have different visibility and synchronization requirements |
| What depends on what? | Flow-graph messages and joins | CUDA streams/events/graphs describe device work ordering; a stream is not a concurrent application-data queue |
| How much work can overlap? | Arena concurrency and live pipeline tokens | Stream priority does not reserve cores or provide memory isolation; resource use and dependencies limit overlap |
| Who owns a buffer, and when can it be reused? | C++ lifetime plus completed tasks | Host/device visibility and asynchronous completion must be established before reuse or reclamation |

A work-stealing CPU scheduler is not a hardware warp scheduler. `enumerable_thread_specific` is not GPU shared memory, and an arena is not a CUDA stream. Carry the ownership and dependency proofs forward, then learn the device-specific contract in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html). These analogies make no claim of GPU execution or performance.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [oneTBB Developer Guide](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/title.html) | Full tutorial: parallel_for, reduce, pipeline, flow graph, design patterns |
| [oneTBB API Reference](https://uxlfoundation.github.io/oneTBB/main/reference/source/index.html) | Complete API |
| [Intel oneTBB Get Started](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2022-2/overview.html) | Installation, CMake, first steps |
| [OpenMP specifications](https://www.openmp.org/specifications/) | Official OpenMP standard |
| [OpenMP API Quick Reference Card](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf) | All clauses at a glance |
| *Intel Threading Building Blocks* (Reinders) | Book: TBB patterns in depth |

---

## Next

→ [**CUDA and SIMT**](../CUDA%20and%20SIMT/Guide.md) — GPU parallelism: threads, warps, blocks, and memory hierarchy.

## Budget parallelism across the whole pipeline

Run capture/replay, preprocessing, and inference submission as a bounded pipeline. Vary worker counts while keeping the input arrival schedule fixed. Compare throughput with observation age, queue depth, and drops; more workers can oversubscribe a library’s own pool or increase contention. State whether the last stage preserves sequence order.

For reductions, separate acceptable floating-point variation from data races. For task placement, separate oneTBB affinity hints from OS CPU affinity. The handoff to CUDA is a host timeline showing whether the GPU is starved, waiting for a copy, or simply finished before the next input arrives.

**Carry forward:** A bounded pipeline timeline and worker-count/latency comparison. Continue with [the next course](../CUDA%20and%20SIMT/Guide.md) and reuse that evidence.
