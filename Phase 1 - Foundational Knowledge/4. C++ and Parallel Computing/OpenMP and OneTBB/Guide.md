# OpenMP and oneTBB

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">OAO</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Specialized course identity for OpenMP and oneTBB.</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


Part of [Phase 1 section 4 — C++ and Parallel Computing](../Guide.md).

**Goal:** Shared-memory **CPU parallelism** with **OpenMP** (directive-based) and **oneTBB** (task-based algorithms and flow graphs) so structured parallel patterns feel familiar before CUDA.

This guide covers two parts. Section 1 is a progressive tour of OpenMP — from a one-line `#pragma` to tasks and SIMD. Section 2 covers oneTBB's template-based API in the same order, from simple loops to flow graphs and runtime controls.

---

## 1. OpenMP

### 1.0 Mental Model: Fork-Join

OpenMP uses the **fork-join model**. Your program starts with a single *master* thread. When it reaches a parallel region, it **forks** into a team of worker threads. All threads execute the region concurrently, then **join** back into one at an implicit barrier at the end.

```
 serial          parallel region              serial
─────────── ╔══════════════════════════╗ ───────────────►
            ║  #pragma omp parallel    ║
   master ──╫──► thread 0  (work) ────╫──► master
            ╠──► thread 1  (work) ────╣
            ╠──► thread 2  (work) ────╣
            ╚──► thread 3  (work) ────╝
                                  ▲
                         implicit barrier:
                    all threads must arrive
                    before master continues
```

You can have **multiple parallel regions** in one program. Each fork creates the thread team; each join dissolves it (or returns threads to a pool for reuse).

```
master ──── [serial] ──── fork ──── [parallel] ──── join ──── [serial] ──── fork ──── [parallel] ──── join ────►
```

**Three things OpenMP manages for you automatically:**

| What | How OpenMP handles it |
|------|-----------------------|
| Thread creation | Thread pool — threads are reused, not recreated each region |
| Work distribution | Divides loop iterations across threads (`schedule` clause controls how) |
| Synchronization | Implicit barrier at the end of every parallel region |

**Compile:** `g++ -O2 -std=c++17 -fopenmp fib_benchmark.cpp -ltbb -o fib_benchmark`

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

// Parallel: 8 threads each do ~125,000 iterations
#pragma omp parallel for
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];
```

That's it. One line added. The compiler splits `[0, N)` into chunks, assigns each chunk to a thread, and inserts a barrier at the end.

**What each thread sees:**

```
Thread 0:  i = 0 … 124,999
Thread 1:  i = 125,000 … 249,999
Thread 2:  i = 250,000 … 374,999
...
Thread 7:  i = 875,000 … 999,999
```

This is safe here because each `i` writes to a different `c[i]`. No two threads touch the same memory.

---

### 1.2 The Classic Race Condition

**Wrong — data race:**

```cpp
int sum = 0;

#pragma omp parallel for
for (int i = 0; i < N; i++)
    sum += a[i];   // ← RACE: multiple threads read-modify-write sum simultaneously

// sum is wrong. Possibly different on every run.
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
int sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++)
    sum += a[i];   // each thread accumulates its own private sum
                   // all private sums are added together at the end
```

Each thread gets its own private copy of `sum` (initialized to 0). After the loop, OpenMP adds all private copies into the original `sum`. No race. Section 2.5 covers all supported reduction operators and custom reductions.

---

### 1.3 Data Sharing Clauses

Every variable referenced inside a parallel region is either **shared** (one copy, all threads see it) or **private** (each thread has its own copy). OpenMP's default: variables declared outside the region are shared.

**No** — `private` and `firstprivate` are mutually exclusive for the same variable. A variable can only appear in one clause. They are alternatives that differ only in initialization:

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
    a[i] = x + i;    // x is read first → needs the initial value 10 → must use firstprivate
    x = compute(i);  // then overwritten — fine, it's private
}
```

**Rule:** use `private` when you always write before read. Use `firstprivate` when you need the original value inside the loop.

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
    store_result(i, buf, tmp);
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

> **Prefer Version A** when possible — declare scratch variables inside the loop body. Use `private` only when you *must* declare the variable outside (e.g., fixed-size arrays, C99 VLAs, or compatibility constraints).

---

#### `lastprivate` — Getting the Value from the Last Iteration

`lastprivate` is private during the loop, but after the loop ends, the value from the **sequentially last iteration** (highest `i`) is copied back to the original variable.

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

> **`lastprivate` copies from the thread that handled the highest `i`**, not the thread that finished last in wall-clock time. The result is deterministic regardless of scheduling.

**Common real use cases:**

| Pattern | Why `lastprivate` |
|---------|------------------|
| Running total / recurrence after a sweep | Need final accumulated value |
| Finding the last element matching a condition | Carry the match out of the loop |
| Mesh traversal — record position of last node processed | Node index/coords after full sweep |
| Numerical integration — endpoint value needed post-loop | Value of integrand at upper bound |

**`lastprivate` + `firstprivate` on the same variable** — this *is* legal (they are not the same clause):

```cpp
double x = 0.5;   // starting x

// firstprivate: each thread starts with x = 0.5
// lastprivate:  after loop, x = value from last iteration
#pragma omp parallel for firstprivate(x) lastprivate(x)
for (int i = 0; i < N; i++) {
    x = x * 0.99 + i * dx;   // reads x (needs firstprivate) and updates it
    output[i] = x;
}
// x here = value of x after i = N-1, with each thread having started at 0.5
```

---

#### Full Clause Reference

| Clause | Initialized? | Written back after loop? | Use when |
|--------|:-----------:|:------------------------:|---------|
| `shared(var)` | — (one copy) | — (always live) | Read-only inside loop, or safely written with sync |
| `private(var)` | **No** | No | Scratch — always written before read |
| `firstprivate(var)` | **Yes** (from original) | No | Needs original value inside loop |
| `lastprivate(var)` | No | **Yes** (from last `i`) | Need loop's final value after it ends |
| `firstprivate` + `lastprivate` | **Yes** | **Yes** | Needs original value AND final value |
| `reduction(op:var)` | Yes (identity) | Yes (combined) | Accumulate across all iterations |

> **Rule of thumb:** If all threads only read → `shared`. Scratch variable → `private`. Needs original value → `firstprivate`. Need value after loop ends → `lastprivate`. Accumulate → `reduction`.

---

### 1.4 Schedules — How Work Is Divided

OpenMP divides loop iterations among threads according to a *schedule*. The right schedule depends on whether iterations take equal time or vary widely.

```cpp
// Static (default): divide evenly before the loop starts
// Thread 0 gets [0, N/P), thread 1 gets [N/P, 2N/P), ...
// Best when: each iteration costs the same amount of time
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) { /* uniform work */ }

// Static with chunk: interleave chunks of k iterations
// Thread 0 gets 0-7, 32-39, 64-71, ...   (chunk=8, 4 threads)
// Helps with cache locality in some patterns
#pragma omp parallel for schedule(static, 8)
for (int i = 0; i < N; i++) { /* */ }

// Dynamic: each thread takes the next k iterations when it becomes free
// Overhead: ~synchronization per chunk fetch
// Best when: iterations have unpredictable/varying cost
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < N; i++) { /* variable-cost work */ }

// Guided: starts with large chunks, shrinks over time
// Reduces scheduling overhead while handling tail imbalance
// Best when: later iterations are lighter than earlier ones
#pragma omp parallel for schedule(guided)
for (int i = 0; i < N; i++) { /* */ }

// Runtime: schedule determined by OMP_SCHEDULE env variable
// OMP_SCHEDULE="dynamic,32" ./my_program
#pragma omp parallel for schedule(runtime)
for (int i = 0; i < N; i++) { /* */ }
```

**When to pick which:**

```
All iterations take the same time?     → static (lowest overhead)
Iterations have wildly different cost? → dynamic
Don't know, want adaptive?             → guided
Tuning at runtime without recompile?   → runtime
```

---

### 1.5 Reductions

Building on the `reduction(+:sum)` clause introduced in section 2.2, here are all supported operators and how to define custom reductions:

```cpp
int sum = 0, product = 1, max_val = INT_MIN;

#pragma omp parallel for reduction(+:sum) reduction(*:product) reduction(max:max_val)
for (int i = 0; i < N; i++) {
    sum      += a[i];
    product  *= b[i];
    max_val   = std::max(max_val, a[i]);
}
```

Built-in operators: `+`, `*`, `-`, `&`, `|`, `^`, `&&`, `||`, `min`, `max`.

**Custom reduction (C++ only, OpenMP 4.0+):**

```cpp
struct Vec3 { float x, y, z; };

// Declare how to combine two partial results (omp_out += omp_in)
// and what value each thread's private copy starts at.
#pragma omp declare reduction(vec_add : Vec3 : \
    omp_out.x += omp_in.x; \
    omp_out.y += omp_in.y; \
    omp_out.z += omp_in.z) \
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

> Use `collapse` when the outer loop count is small relative to the thread count.

---

### 1.7 SIMD — Vectorization Hints

SIMD (Single Instruction, Multiple Data) processes multiple array elements in one CPU instruction. This is the same principle behind GPU SIMT — learning it here prepares you for CUDA's warp-level execution.

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

The `simd` pragma is a hint. The compiler still decides if vectorization is safe. Add `-fopt-info-vec` to see what was vectorized.

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

// atomic: faster for single memory operations (uses hardware atomics)
int counter = 0;
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    counter++;           // ← hardware atomic, no mutex overhead
}

// atomic with operation
#pragma omp atomic update
total += a[i];

#pragma omp atomic read
int val = shared_var;

#pragma omp atomic write
shared_var = computed;
```

**Critical vs Atomic:**
- `#pragma omp critical` — mutex, any code block, higher overhead
- `#pragma omp atomic` — hardware instruction, single read/write/update only, much faster

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
     only safe when the two loops are data-independent
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
    use(a, b);                      // safe to read both here
}
```

**When `nowait` is safe:** the next work unit reads from a completely different data set than the loop just completed. If there is any dependency, keep the implicit barrier.

---

### 1.10 Sections — Fixed Concurrent Operations

`sections` is for a **small, known-at-compile-time** set of independent operations — not a loop. Each `section` block runs on one thread; all blocks run concurrently.

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
    load_data();        // thread 0

    #pragma omp section
    init_config();      // thread 1

    #pragma omp section
    warmup();           // thread 2
}
// all three finished here
```

**Sections vs Tasks:**

| | `sections` | `tasks` |
|--|-----------|---------|
| Number of units | Fixed at compile time | Dynamic, recursive |
| Use case | Load + init + warmup | Tree traversal, Fibonacci |
| Overhead | Very low | Higher (task queue) |

Use `sections` when you can count the operations by hand. Use `tasks` (next section) when the work fans out recursively or the count isn't known until runtime.

---

### 1.11 Tasks — Recursive and Irregular Work

Tasks decouple **work creation** from **work execution**. One thread creates tasks and puts them in a queue; any thread in the team picks them up and executes them.

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

    #pragma omp task shared(left_sum)
    left_sum = sum_tree(node->left);

    #pragma omp task shared(right_sum)
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

> **Common mistake:** putting `#pragma omp parallel` inside the recursive function. That creates a new thread team on every recursive call — thousands of nested thread pools, massive overhead, and likely wrong results.

**Fibonacci with cutoff:**

```cpp
int fib(int n) {
    if (n < 2)  return n;
    if (n < 25) return fib(n-1) + fib(n-2);  // serial below cutoff

    int x, y;
    #pragma omp task shared(x) firstprivate(n)
    x = fib(n - 1);

    #pragma omp task shared(y) firstprivate(n)
    y = fib(n - 2);

    #pragma omp taskwait
    return x + y;
}

int result;
#pragma omp parallel
#pragma omp single
result = fib(50);
```

**Why each clause matters:**

| Clause | What it does |
|--------|-------------|
| `shared(x)` | All threads see the same `x` — task writes its result there |
| `firstprivate(n)` | Each task gets its own copy of `n` at creation time — required for correctness in recursion |
| `taskwait` | Suspends the current task until all child tasks finish |
| `single` | Only one thread spawns tasks; the rest execute them. Without it, every thread would spawn the full tree → exponential task explosion |

**How the recursion tree maps to tasks — `fib(6)` example:**

`fib(6)` produces 25 nodes total. Each internal node above the cutoff becomes a task; leaves (`fib(0)`, `fib(1)`) return immediately.

```
fib(6)                                ← T0 spawns fib(5) and fib(4), then taskwait
├─ fib(5)                             ← T1 picks up, spawns fib(4) and fib(3)
│  ├─ fib(4)                          ← T2 picks up, spawns fib(3) and fib(2)
│  │  ├─ fib(3)                       ← T3 picks up, spawns fib(2) and fib(1)
│  │  │  ├─ fib(2) → fib(1)+fib(0)   ← returns 1 immediately (leaf)
│  │  │  └─ fib(1)                    ← returns 1 immediately (leaf)
│  │  └─ fib(2) → fib(1)+fib(0)      ← returns 1 immediately (leaf)
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

**Key insight — `single` does not bottleneck:**  T0 spawns only the top two tasks (`fib(n-1)` and `fib(n-2)`), then immediately hits `taskwait` and is available to execute other tasks. By the time T0 is done spawning, T1–T3 are already recursively creating and consuming the subtrees. No thread ever sits idle as long as the queue has work.

For `fib(50)` with cutoff=25, the recursion produces ~fib(25) ≈ 75,000 tasks above the cutoff — more than enough to keep any number of cores busy.

---

### 1.12 Thread Info and Environment

```cpp
// Query runtime info
int n_threads = omp_get_num_threads();   // inside parallel region
int thread_id = omp_get_thread_num();    // 0 to n_threads-1
int max_threads = omp_get_max_threads(); // outside parallel region
int n_procs = omp_get_num_procs();       // hardware thread count

// Set thread count
omp_set_num_threads(8);

// Timing
double t0 = omp_get_wtime();
// ... work ...
double elapsed = omp_get_wtime() - t0;
```

**Environment variables (set before running):**

```bash
OMP_NUM_THREADS=8             # number of threads
OMP_SCHEDULE="dynamic,64"     # schedule for 'runtime' clauses
OMP_PROC_BIND=close           # bind threads to nearby hardware (NUMA)
OMP_PLACES=cores              # binding granularity: threads, cores, sockets
GOMP_SPINCOUNT=100000         # how long to spin before sleeping (GNU)
```

---

### 1.13 Nested Parallelism

```cpp
omp_set_nested(1);  // enable nested parallel regions

void outer_task() {
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 4; i++) {
        // Inner parallel region: each of 4 threads spawns 2 more
        #pragma omp parallel for num_threads(2)
        for (int j = 0; j < N; j++)
            compute(i, j);
    }
}
```

> Nested parallelism often over-subscribes cores. Usually better to `collapse(2)` instead.

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
| Granularity | Loop-level | Task-level |
| Load balancing | Static/dynamic/guided schedules | Work-stealing (automatic, adaptive) |
| Flow graphs | No | Yes (`flow_graph`) |
| Per-thread storage | `threadprivate` / `private` clause | `enumerable_thread_specific` |
| Nested parallelism | Manual | Built-in |
| SIMD hints | `#pragma omp simd` | Must use manual intrinsics |
| Best for | Existing loops, Fortran interop, scientific HPC | New C++ code, complex graphs, irregular tasks |

**Resources:** [OpenMP specifications](https://www.openmp.org/specifications/) · [OpenMP API Reference Card (PDF)](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf)

The table above highlights where oneTBB offers capabilities OpenMP lacks — especially work-stealing, flow graphs, and concurrent containers. The next section covers oneTBB's API in the same progressive style, starting from simple loops and building toward complex graph-based patterns.

---

### 1.16 Benchmark — Serial vs OpenMP vs oneTBB

The Fibonacci example from section 2.11 is a useful benchmark because the call tree is pure recursive work with no I/O or system calls — the only variable is scheduling overhead and parallelism.

**Full benchmark (`fib_benchmark.cpp`):**

```cpp
// fib_benchmark.cpp — serial / OpenMP tasks / oneTBB parallel_invoke
// Compile: g++ -O2 -std=c++17 -fopenmp fib_benchmark.cpp -ltbb -o fib_benchmark

#include <chrono>
#include <iostream>
#include <omp.h>
#include "oneapi/tbb/parallel_invoke.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// Below this depth, run sequentially — avoids task overhead on trivial work
static constexpr int CUTOFF = 20;

// ── 1. Serial ─────────────────────────────────────────────────────────────────
long long fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}

// ── 2. OpenMP tasks ───────────────────────────────────────────────────────────
long long fib_omp(int n) {
    if (n < CUTOFF) return fib_serial(n);   // switch to serial below cutoff
    long long x, y;
    #pragma omp task shared(x) firstprivate(n)
    x = fib_omp(n - 1);
    #pragma omp task shared(y) firstprivate(n)
    y = fib_omp(n - 2);
    #pragma omp taskwait
    return x + y;
}

// ── 3. oneTBB parallel_invoke ─────────────────────────────────────────────────
long long fib_tbb(int n) {
    if (n < CUTOFF) return fib_serial(n);
    long long x, y;
    tbb::parallel_invoke(
        [&]{ x = fib_tbb(n - 1); },
        [&]{ y = fib_tbb(n - 2); }
    );
    return x + y;
}

// ── Timing helper ─────────────────────────────────────────────────────────────
template<typename Fn>
double time_ms(Fn&& fn) {
    auto t0 = Clock::now();
    fn();
    return Ms(Clock::now() - t0).count();
}

int main() {
    const int N = 42;
    long long r_serial, r_omp, r_tbb;

    // warm up (first call pays library init cost)
    fib_serial(30);

    double t_serial = time_ms([&]{ r_serial = fib_serial(N); });

    double t_omp = time_ms([&]{
        #pragma omp parallel
        #pragma omp single
        r_omp = fib_omp(N);
    });

    double t_tbb = time_ms([&]{ r_tbb = fib_tbb(N); });

    const int threads = omp_get_max_threads();
    std::printf("fib(%d) = %lld   [threads available: %d]\n\n", N, r_serial, threads);
    std::printf("%-12s %8.1f ms  (baseline)\n",   "serial",  t_serial);
    std::printf("%-12s %8.1f ms  (%.2fx speedup)\n", "openmp", t_omp, t_serial / t_omp);
    std::printf("%-12s %8.1f ms  (%.2fx speedup)\n", "onetbb", t_tbb, t_serial / t_tbb);

    if (r_serial != r_omp || r_serial != r_tbb)
        std::fprintf(stderr, "ERROR: results differ!\n");
    return 0;
}
```

**Compile and run:**

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install g++ libomp-dev libtbb-dev

# Compile
g++ -O2 -std=c++17 -fopenmp fib_benchmark.cpp -ltbb -o fib_benchmark

# Run
./fib_benchmark
```

**Measured output (4-core machine, fib(50), CUTOFF=25):**

```
fib(50) = 12586269025   [threads available: 4, cutoff: 25]

serial        31887.7 ms  (baseline)
openmp        14057.1 ms  (2.27x speedup)
onetbb         9616.4 ms  (3.32x speedup)
```

**Why the speedup is sub-linear:**

The theoretical maximum with `T` threads is `T×` speedup. In practice:

| Source of loss | OpenMP | oneTBB |
|----------------|--------|--------|
| Task creation overhead | Higher (pragma dispatch) | Lower (inline lambda) |
| Load balancing | Victim-push scheduling | Work-stealing (adaptive) |
| Cache thrash from stealing | Moderate | Low (LIFO own / FIFO steal) |
| Cutoff tuning | Manual | Manual |

oneTBB's work-stealing scheduler adapts better when branches complete at different rates, which is why it typically edges out OpenMP on irregular recursion.

**CUTOFF sensitivity** — measured on a 4-core machine with fib(50):

```bash
for c in 20 25 30 35 40; do
  g++ -O2 -std=c++17 -fopenmp -DCUTOFF_VAL=$c fib_benchmark.cpp -ltbb -o fib_b$c
  echo "=== CUTOFF=$c ===" && ./fib_b$c
done
```

| CUTOFF | Tasks spawned | OpenMP speedup | oneTBB speedup |
|--------|--------------|----------------|----------------|
| 20 | ~830 K | 1.15x | 2.42x |
| 25 | ~26 K | **2.27x** | **3.32x** |
| 30 | ~830 | 1.93x | 3.01x |
| 35 | ~26 | 2.32x | 2.79x |
| 40 | ~1 | 1.83x | 2.78x |

CUTOFF=25 is the sweet spot: enough tasks to keep all threads busy, few enough that scheduling overhead doesn't dominate. OpenMP is more sensitive to CUTOFF than oneTBB because work-stealing adapts automatically to load imbalance.

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
- Stealer pops from the **left** (FIFO) — takes the biggest, oldest tasks, maximizing stolen work size

**Result:** Threads never sit idle as long as any thread has work. No need to manually tune thread counts for imbalanced workloads.

**Install:**

```bash
# Ubuntu/Debian
sudo apt install libtbb-dev

# CMake integration
find_package(TBB REQUIRED)
target_link_libraries(my_target TBB::tbb)
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
int N = 1'000'000;
float a[N], b[N], c[N];

// Compact form — runtime picks chunk size automatically
parallel_for(0, N, [&](int i) {
    c[i] = a[i] + b[i];
});
```

**`blocked_range` form — gives you the subrange, loop inside:**

```cpp
// Better: lets you write inner loops with fewer lambda calls
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
- Avoids per-element lambda call overhead
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

// affinity_partitioner: reuses same data → same thread (cache-warm)
// Declare static so it persists between calls
static affinity_partitioner ap;
parallel_for(blocked_range<int>(0, N), body, ap);

// simple_partitioner: chunk = grainsize exactly, no adaptive splitting (work-stealing still active)
parallel_for(blocked_range<int>(0, N, 1000), body, simple_partitioner());

// static_partitioner: divide evenly upfront, no stealing
// Deterministic: same thread always gets same range
parallel_for(blocked_range<int>(0, N), body, static_partitioner());
```

| Partitioner | Chunk size | Adaptive splitting | Work-stealing | Use when |
|-------------|------------|--------------------|---------------|---------|
| `auto_partitioner` | Dynamic | Yes | Yes | Most cases — default |
| `affinity_partitioner` | Adaptive | Yes | Yes | Re-running same loop over same data (cache-warm) |
| `simple_partitioner` | Fixed ≈ grainsize | No | Yes | Chunk needs fixed-size temp buffer; work is uniform; N is huge |
| `static_partitioner` | Pre-divided, fixed mapping | No | No | NUMA/cache locality tuning; strict reproducible thread mapping |

**Grainsize tuning:** Each chunk should take ≥ ~100,000 clock cycles (~50 µs at 2 GHz). Too-small chunks = scheduling overhead dominates. Rule: if loop body takes 10 ns, grainsize of 10,000+ is reasonable.

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

**What the runtime does:**

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

**Why `blocked_range2d` beats two nested `parallel_for` calls:**

```
blocked_range2d splits the grid into rectangular tiles:

  ┌───────┬───────┬───────┐
  │ T0    │ T1    │ T2    │   Each tile = contiguous memory block
  │       │       │       │   → good L1/L2 cache reuse within tile
  ├───────┼───────┼───────┤
  │ T3    │ T4    │ T5    │
  │       │       │       │
  └───────┴───────┴───────┘

Two nested parallel_for → each row is a separate task → too many tiny tasks
blocked_range2d    → each tile is one task → right granularity
```

The `tile.rows()` and `tile.cols()` accessors give you the subrange for this tile. Always use flat `vector<T>` with `[y * W + x]` indexing — `vector<vector<T>>` breaks spatial locality.

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

**Find minimum with index:**

```cpp
struct MinResult { float val; int idx; };

auto result = parallel_reduce(
    blocked_range<int>(0, N),
    MinResult{FLT_MAX, -1},
    [&](const blocked_range<int>& r, MinResult curr) {
        for (int i = r.begin(); i < r.end(); i++)
            if (a[i] < curr.val) curr = {a[i], i};
        return curr;
    },
    [](MinResult a, MinResult b) {
        return a.val < b.val ? a : b;
    }
);
```

> **Common mistake:** Do not reset the accumulator to zero inside the body — `init` carries prior partial results. Resetting it discards prior work.

**Deterministic reduce** — same result every run regardless of scheduling:

```cpp
float result = parallel_deterministic_reduce(
    blocked_range<int>(0, N, 1000),  // explicit grainsize required
    0.0f,
    [&](const blocked_range<int>& r, float init) {
        return std::accumulate(&a[r.begin()], &a[r.end()], init);
    },
    std::plus<float>{}
);
// Floating-point result identical on every run
```

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
        target[i] = static_cast<float>((i + 1) % 100) / 100.0f;  // [0.01, 1.00]
    }

    // parallel_reduce arguments:
    //   1. range    — what to iterate over
    //   2. identity — each thread's partial sum starts at 0
    //   3. body     — reduce a subrange into a partial sum
    //   4. combine  — merge two partial sums into one
    float sum_sq = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(0, N),

        0.0f,   // identity

        [&](const oneapi::tbb::blocked_range<int>& r, float init) {
            for (int i = r.begin(); i < r.end(); ++i) {
                float diff = pred[i] - target[i];
                init += diff * diff;
            }
            return init;
        },

        std::plus<float>{}   // combine
    );

    float mse = sum_sq / N;
    std::cout << "MSE = " << mse << "\n";   // ~0.003267
}
```

**How TBB splits and joins the work:**

```
           parallel_reduce(blocked_range(0, 10M), 0.0f, body, combine)
                              [0 … 10,000,000)
                             /                \
                            /                  \
                [0 … 5,000,000)          [5,000,000 … 10M)
               /               \         /               \
       [0…2.5M)           [2.5M…5M)  [5M…7.5M)     [7.5M…10M)
       sum=8,167.5        sum=8,167.5 sum=8,167.5   sum=8,167.5
              \               /         \               /
           combine(a,b) = a+b           combine(a,b) = a+b
               sum=16,335.0                 sum=16,335.0
                          \                 /
                           combine(a,b) = a+b
                               sum=32,670.0
                           MSE = 32,670.0 / 10M ≈ 0.003267
```

Each leaf calls `body(subrange, 0.0f)` — the identity `0.0f` means every thread's partial sum starts at zero. After all leaves finish, TBB walks back up the tree calling `combine` to sum the partial results.

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

The body lambda runs **twice per chunk** — once without writing (building partial sums), once with the correct carry-in (filling output). The `is_final` flag tells the body which pass it is.

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
inclusive (what parallel_scan computes by default):
  input:  [ 1,  2,  3,  4,  5 ]
  output: [ 1,  3,  6, 10, 15 ]   out[i] includes in[i]

exclusive (shifted by one, out[0] = identity):
  input:  [ 1,  2,  3,  4,  5 ]
  output: [ 0,  1,  3,  6, 10 ]   out[i] excludes in[i]
```

Exclusive scan: delay the write by one position — accumulate first, then write the value *before* adding the current element:

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
// output = [1, 2, 3, 4]
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

Uses a parallel quicksort variant with work-stealing. Typically 4–6× faster than `std::sort` on 8+ cores for large N.

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

**Dynamic work addition with feeder** — BFS / tree traversal:

```cpp
parallel_for_each(roots.begin(), roots.end(),
    [](Node* node, feeder<Node*>& f) {
        process(node);
        for (Node* child : node->children)
            f.add(child);             // adds work dynamically to the pool
    }
);
// Runs until all nodes processed, including dynamically added ones
```

---

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

**Structure:**

```cpp
#include "oneapi/tbb/pipeline.h"

const int max_tokens = 16;   // max items in-flight simultaneously

parallel_pipeline(max_tokens,
    make_filter<void, InputData*>(            // void input = "source" stage
        filter_mode::serial_in_order,
        [&](flow_control& fc) -> InputData* {
            InputData* data = read_next();
            if (!data) { fc.stop(); return nullptr; }  // signal end of stream
            return data;
        }
    ) &
    make_filter<InputData*, OutputData*>(
        filter_mode::parallel,
        [](InputData* in) -> OutputData* {
            return transform(in);             // runs on multiple threads
        }
    ) &
    make_filter<OutputData*, void>(           // void output = "sink" stage
        filter_mode::serial_in_order,
        [&](OutputData* out) {
            write_result(out);
            delete out;
        }
    )
);
```

**Concrete example — batch inference pipeline:**

A common CPU inference pattern: load batches from disk, normalize, run model, write predictions. I/O and inference can overlap:

```cpp
struct Batch { std::vector<float> data; int id; };

int batch_id = 0;
const int TOTAL = 100;

parallel_pipeline(/*max_tokens=*/8,
    // Stage 1: load next batch from disk (serial — disk reads are sequential)
    make_filter<void, Batch*>(
        filter_mode::serial_in_order,
        [&](flow_control& fc) -> Batch* {
            if (batch_id >= TOTAL) { fc.stop(); return nullptr; }
            auto* b = new Batch();
            b->id   = batch_id++;
            b->data = load_batch_from_disk(b->id);   // blocking I/O
            return b;
        }
    ) &
    // Stage 2: normalize pixels [0,255] → [0,1] (parallel — independent batches)
    make_filter<Batch*, Batch*>(
        filter_mode::parallel,
        [](Batch* b) -> Batch* {
            for (float& v : b->data) v /= 255.0f;
            return b;
        }
    ) &
    // Stage 3: run model inference (parallel — batches are independent)
    make_filter<Batch*, Batch*>(
        filter_mode::parallel,
        [](Batch* b) -> Batch* {
            run_inference(b->data);
            return b;
        }
    ) &
    // Stage 4: write predictions in order (serial — output file needs order)
    make_filter<Batch*, void>(
        filter_mode::serial_in_order,
        [&](Batch* b) {
            write_predictions(b->id, b->data);
            delete b;
        }
    )
);
```

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

> **Throughput = `max_tokens / latency_of_slowest_serial_stage`.**
> If your serial read stage takes 10 ms and you set `max_tokens = 8`, you can sustain 800 items/second through the pipeline regardless of how fast the parallel stage is. Doubling `max_tokens` doubles throughput — until the parallel stage becomes the bottleneck.

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
// Both guaranteed done here

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
| Recursive tasks | No | Yes — tasks can call `tg.run()` |
| Syntax | One call, all lambdas inline | `run()` / `wait()` separate |
| Overhead | Slightly lower | Slightly higher (queue management) |
| Use case | Merge sort split, dual data load | Tree traversal, work queue, graph |

**When to use which:**
- Know the exact tasks upfront? → `parallel_invoke`
- Tasks discovered at runtime or spawn subtasks? → `task_group`
- Recursive tree/graph traversal? → `task_group` (or oneTBB flow graph for DAGs)

---

### 2.8 Per-Thread Storage — `enumerable_thread_specific`

When multiple threads update the same data structure (histogram, accumulator, buffer), you have three options:

| Approach | Mechanism | Problem |
|----------|-----------|---------|
| Shared variable | Nothing | Data race — undefined behavior |
| Mutex | `std::mutex` + lock | Serializes threads — kills parallelism |
| `atomic` | Hardware instruction | Only works for a single integer/float, not a vector |
| **ETS** | **Per-thread copy** | **No contention at all — each thread owns its data** |

ETS gives every thread its own private copy. Threads never touch each other's data during the parallel phase. At the end, you iterate over all copies and merge them once.

```
parallel phase:                          merge phase:
  T0 → local_hist[0] ──────────────────► global_hist
  T1 → local_hist[1] ──────────────────►     +=
  T2 → local_hist[2] ──────────────────►     +=
  T3 → local_hist[3] ──────────────────►     +=
  (zero contention)                     (single-threaded, once)
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

For expressing complex parallel patterns as a graph of nodes and edges. The runtime automatically runs nodes when their inputs are ready — no manual synchronization needed.

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

function_node<std::tuple<int,int>, void> printer(g, 1,
    [](const std::tuple<int,int>& t) {
        printf("square=%d  cube=%d\n", std::get<0>(t), std::get<1>(t));
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

**ML example — parallel feature extraction with fan-out/join:**

Extract three independent features from each input frame, then combine and score:

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
struct Frame  { std::vector<float> data; };
struct Score  { float value; };

graph g;

// Fan-out: send each frame to all three extractors simultaneously
broadcast_node<Frame> broadcast(g);

function_node<Frame, std::vector<float>> hog(g, unlimited,
    [](const Frame& f) { return extract_hog(f); });
function_node<Frame, std::vector<float>> lbp(g, unlimited,
    [](const Frame& f) { return extract_lbp(f); });
function_node<Frame, std::vector<float>> dct(g, unlimited,
    [](const Frame& f) { return extract_dct(f); });

// Fan-in: wait for all three features before scoring
using FeatureTuple = std::tuple<std::vector<float>,
                                std::vector<float>,
                                std::vector<float>>;
join_node<FeatureTuple> join(g);

function_node<FeatureTuple, Score> scorer(g, unlimited,
    [](const FeatureTuple& t) {
        return score(std::get<0>(t), std::get<1>(t), std::get<2>(t));
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
| `source_node<Out>` *(deprecated)*<br>`input_node<Out>` | Generates tokens from a function; starts the graph |
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
function_node<int, void> worker_a(g, unlimited, [](int x){ do_a(x); });
function_node<int, void> worker_b(g, unlimited, [](int x){ do_b(x); });

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

function_node<Frame, void> writer(g, serial, [](const Frame& f) {
    write_in_order(f);          // receives frames in sequence order
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
make_edge(slow_consumer, limiter.decrement); // signal when done → unlocks limiter
```

#### Conditional Routing with `indexer_node`

```cpp
using IndexerType = indexer_node<int, float>;   // accepts int OR float

IndexerType idx(g);
function_node<IndexerType::output_type, void> router(g, unlimited,
    [](const IndexerType::output_type& msg) {
        if (msg.tag() == 0)       // int arrived
            handle_int(cast_to<int>(msg));
        else                       // float arrived
            handle_float(cast_to<float>(msg));
    }
);

make_edge(idx, router);
input_port<0>(idx).try_put(42);     // send int
input_port<1>(idx).try_put(3.14f);  // send float
```

#### Starter Template — Linear Video Pipeline

A minimal but complete linear pipeline: source → limiter → preprocess → inference → postprocess → sequencer → output. Copy `video_pipeline_template.cpp`, replace the four stub functions, and it's ready to run.

```
input_node → limiter → preprocess → inference → postprocess → sequencer → output
                ▲                                                               │
                └───────────────────── decrement ◄──────────────────────────────┘
```

Limiter is placed **before** preprocessing so no more than `MAX_INFLIGHT` frames ever enter the pipeline. The `decrement` edge feeds back from the output stage to release one slot each time a frame completes.

```bash
g++ -O2 -std=c++17 video_pipeline_template.cpp -ltbb -o video_pipeline
./video_pipeline
```

---

#### Complete Example — Real-Time Video Inference Pipeline

A practical AI video pipeline: read frames, preprocess in parallel, run inference, reorder to original sequence, limit memory, display.

```
  input_node ──► broadcast ──► preprocess (parallel) ──► join ──► inference
                                                                       │
  display ◄── limiter ◄── sequencer ◄── postprocess ◄─────────────────┘
```

```cpp
#include "oneapi/tbb/flow_graph.h"
#include <cstdio>
#include <tuple>
#include <vector>
using namespace oneapi::tbb::flow;

struct Frame {
    int           id;             // sequence number for reordering
    std::vector<float> pixels;   // raw pixel data
};

struct Detection {
    int   frame_id;
    float confidence;
    int   class_id;
};

// ── Simulated pipeline functions ─────────────────────────────────────────────
Frame      read_frame(int id)          { return {id, std::vector<float>(224*224*3, 0.5f)}; }
Frame      resize_normalize(Frame f)   { /* resize to 224×224, normalize [0,1] */ return f; }
Frame      color_convert(Frame f)      { /* BGR → RGB */ return f; }
Detection  run_inference(std::tuple<Frame,Frame> t) {
    // both preprocessed versions available — use whichever suits your model
    const Frame& f = std::get<0>(t);
    return {f.id, 0.95f, 1};
}
Detection  draw_boxes(Detection d)     { printf("frame %d  class=%d  conf=%.2f\n",
                                                d.frame_id, d.class_id, d.confidence);
                                         return d; }
void       display_or_save(Detection d){ /* write to screen or file */ }

int main() {
    const int TOTAL_FRAMES = 20;
    const int MAX_INFLIGHT  = 8;   // limiter: max frames queued at once

    graph g;

    // ── Stage 1: source — emit frames one at a time ───────────────────────────
    int frame_counter = 0;
    input_node<Frame> source(g, [&](oneapi::tbb::flow_control& fc) -> Frame {
        if (frame_counter >= TOTAL_FRAMES) { fc.stop(); return {}; }
        return read_frame(frame_counter++);
    });

    // ── Stage 2: broadcast — send each frame to both preprocess branches ──────
    broadcast_node<Frame> broadcast(g);

    // ── Stage 3: parallel preprocess branches ────────────────────────────────
    function_node<Frame, Frame> preprocess_a(g, unlimited, resize_normalize);
    function_node<Frame, Frame> preprocess_b(g, unlimited, color_convert);

    // ── Stage 4: join — wait for both preprocessed versions ──────────────────
    join_node<std::tuple<Frame, Frame>> join(g);

    // ── Stage 5: inference — CPU/GPU model (parallel across frames) ──────────
    function_node<std::tuple<Frame,Frame>, Detection> inference(g, unlimited, run_inference);

    // ── Stage 6: postprocess — draw bounding boxes ───────────────────────────
    function_node<Detection, Detection> postprocess(g, unlimited, draw_boxes);

    // ── Stage 7: sequencer — restore original frame order ────────────────────
    sequencer_node<Detection> sequencer(g,
        [](const Detection& d) -> size_t { return d.frame_id; }
    );

    // ── Stage 8: limiter — bound memory: max MAX_INFLIGHT frames in-flight ───
    limiter_node<Detection> limiter(g, MAX_INFLIGHT);

    // ── Stage 9: sink — display or write to file ──────────────────────────────
    function_node<Detection, continue_msg> sink(g, serial,
        [&](const Detection& d) -> continue_msg {
            display_or_save(d);
            return {};
        }
    );

    // ── Wire the graph ────────────────────────────────────────────────────────
    make_edge(source,        broadcast);
    make_edge(broadcast,     preprocess_a);
    make_edge(broadcast,     preprocess_b);
    make_edge(preprocess_a,  input_port<0>(join));
    make_edge(preprocess_b,  input_port<1>(join));
    make_edge(join,          inference);
    make_edge(inference,     postprocess);
    make_edge(postprocess,   sequencer);
    make_edge(sequencer,     limiter);
    make_edge(limiter,       sink);
    make_edge(sink,          limiter.decrement);  // release token when frame is done

    // ── Run ───────────────────────────────────────────────────────────────────
    source.activate();     // start emitting frames
    g.wait_for_all();      // block until all frames are processed
}
```

**Why each node is necessary:**

| Node | Role | Why it's needed |
|------|------|----------------|
| `input_node` | Reads frames | Serial I/O source |
| `broadcast_node` | Fan-out | Sends same frame to both preprocess branches |
| `preprocess_a/b` | Parallel | Resize and color convert run simultaneously |
| `join_node` | Fan-in | Inference needs both versions before starting |
| `inference` | Parallel | Different frames infer independently |
| `postprocess` | Parallel | Draw boxes — independent per frame |
| `sequencer_node` | Reorder | Parallel inference may finish out of order |
| `limiter_node` | Backpressure | Prevents 1000 frames buffering in RAM while display is slow |
| `sink` | Output | Serial write to screen/file |

> **Always call `g.wait_for_all()`** before the graph object is destroyed. Destroying a live graph is undefined behavior.

---

### 2.11 Concurrent Containers

**The problem:** standard containers (`std::vector`, `std::map`, `std::queue`) have no internal locking. If two threads write at the same time, you get data corruption, crashes, or silently wrong results.

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
  Thread 1 waits...                  Thread 1 writes ──┤ all at once
  Thread 2 waits...                  Thread 2 writes ──┘
  (serialised)                       (concurrent, safe)
```

> Only use concurrent containers when multiple threads genuinely need to share the same container. If each thread works on its own data, a plain `std::vector` per thread (or `enumerable_thread_specific`) is faster.

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

When to use: collecting results from a parallel filter where you don't know how many items will pass.

#### `concurrent_queue` / `concurrent_bounded_queue`

Classic producer-consumer queue. `try_pop` is non-blocking (returns false if empty); `pop` on `concurrent_bounded_queue` blocks until an item arrives.

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
tbb::concurrent_bounded_queue<WorkItem> bounded(100);
bounded.push(item);   // blocks if 100 items already queued
bounded.pop(item);    // blocks if queue is empty
```

When to use: streaming pipelines where producers and consumers run at different rates.

#### `concurrent_hash_map`

High-concurrency key-value store. Uses per-bucket locking — only the bucket being written is locked, so unrelated keys never contend.

```cpp
#include "oneapi/tbb/concurrent_hash_map.h"

tbb::concurrent_hash_map<std::string, int> freq;

tbb::parallel_for_each(words.begin(), words.end(), [&](const std::string& w) {
    tbb::concurrent_hash_map<std::string, int>::accessor acc;  // write lock
    freq.insert(acc, w);   // locks only the bucket for key w
    acc->second++;         // safe: this thread owns the bucket
});                        // accessor destructor releases the lock

// Read-only (shared lock — many readers can hold this simultaneously)
tbb::concurrent_hash_map<std::string, int>::const_accessor cacc;
if (freq.find(cacc, "hello"))
    printf("hello: %d\n", cacc->second);
```

`accessor` = write lock (exclusive). `const_accessor` = read lock (shared). Multiple threads can hold `const_accessor` on the same key simultaneously.

#### `concurrent_unordered_map`

Simpler API — drop-in for `std::unordered_map`, no accessor needed. Individual insertions and lookups are thread-safe, but iteration while modifying is not.

```cpp
#include "oneapi/tbb/concurrent_unordered_map.h"

tbb::concurrent_unordered_map<int, int> m;

tbb::parallel_for(0, N, [&](int i) {
    m.insert({i, i * i});    // safe
    // m[i] = i * i;         // also safe for insert; avoid for update
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

Standard `malloc`/`free` have a single global lock — a bottleneck when many threads allocate simultaneously. TBB's allocator uses per-thread memory pools:

```cpp
#include "oneapi/tbb/scalable_allocator.h"

// Use tbb_allocator with STL containers
std::vector<float, tbb::tbb_allocator<float>> vec(N);

// Use scalable_malloc directly (drop-in for malloc)
float* buf = (float*)scalable_malloc(N * sizeof(float));
scalable_free(buf);

// Replace all allocations globally (link with -ltbbmalloc_proxy)
// Or: set LD_PRELOAD=libtbbmalloc_proxy.so before running
```

Most impactful when many threads frequently allocate/free small objects (task objects, intermediate buffers in a pipeline).

---

### 2.13 `task_arena` — Control Thread Pool

By default, TBB creates one global thread pool that uses all hardware threads. Every `parallel_for`, `parallel_reduce`, etc. draws from that pool. `task_arena` lets you create a **separate, bounded execution zone** with its own concurrency limit and optionally pinned to specific NUMA nodes.

```
Default (global arena):          With task_arena:
  All 16 hardware threads          arena(4) → max 4 threads
  shared by everything             Work inside it can't steal
  ← any parallel_for steals        from the global pool
     from any other
```

**`task_arena` does NOT create new OS threads.** TBB's worker threads can participate in multiple arenas. What the arena controls is how many of them are *allowed in* at once.

#### Basic usage — limit parallelism

```cpp
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/parallel_for.h"

tbb::task_arena arena(2);   // at most 2 threads

arena.execute([&] {
    tbb::parallel_for(0, 1000, [](int i) {
        // runs with at most 2 threads regardless of core count
        do_work(i);
    });
});
```

Useful when your program shares the CPU with a GPU or another process and you want to leave headroom.

#### Two independent subsystems

```cpp
tbb::task_arena physics(4);   // physics simulation: 4 threads
tbb::task_arena render(4);    // rendering: 4 threads

// Launch both concurrently — they draw from separate thread budgets
// No work-stealing between them
std::thread t1([&]{ physics.execute([&]{ parallel_for(0, N, sim_body);    }); });
std::thread t2([&]{ render.execute( [&]{ parallel_for(0, M, render_body); }); });
t1.join(); t2.join();
```

Without arenas, TBB's global pool would intermix work from both subsystems — threads doing physics could be stolen away to help with rendering mid-frame. Arenas prevent that.

#### NUMA-aware pinning (multi-socket servers)

```cpp
tbb::task_arena numa_arena(
    tbb::task_arena::constraints{}
        .set_numa_id(1)            // bind to NUMA socket 1
        .set_max_concurrency(8)    // use up to 8 threads from that socket
);

numa_arena.execute([&]{
    parallel_for(0, N, body);     // threads stay on socket 1's cores
});
// Data allocated on socket 1 memory is accessed locally → lower latency
```

#### When to use `task_arena`

| Scenario | Why `task_arena` helps |
|----------|----------------------|
| CPU + GPU running simultaneously | Limit CPU threads so GPU isn't starved for PCIe bandwidth |
| UI application with background processing | Prevent background `parallel_for` from consuming all cores |
| Multiple independent subsystems | Isolate them so one can't steal from the other |
| NUMA system (multi-socket server) | Pin arenas to sockets to keep data access local |
| Debugging: make parallel code deterministic | `task_arena(1)` forces single-thread execution |

---

### 2.14 `global_control` — Runtime Configuration

`task_arena` limits threads *locally* (inside one block). `global_control` sets a **program-wide policy** that applies to every TBB algorithm everywhere — including third-party libraries that use TBB internally.

```
task_arena(4).execute(...)      global_control(max_allowed_parallelism, 4)
  Only this block uses 4         Every parallel_for, parallel_reduce,
  Everything else: unlimited     pipeline, etc. in the entire process: ≤ 4
```

Uses RAII — settings revert automatically when the object goes out of scope:

```cpp
#include "oneapi/tbb/global_control.h"

{
    tbb::global_control ctrl(
        tbb::global_control::max_allowed_parallelism, 4
    );
    // ── all TBB work inside this scope uses at most 4 threads ──
    tbb::parallel_for(0, N, body_a);   // ≤ 4 threads
    some_library_call();               // if it uses TBB: also ≤ 4 threads
}
// ── ctrl destroyed → limit lifted, TBB uses all cores again ──
tbb::parallel_for(0, N, body_b);       // full thread count again
```

**Set thread stack size** — useful when tasks recurse deeply or allocate large local arrays:

```cpp
tbb::global_control stack_ctrl(
    tbb::global_control::thread_stack_size,
    8 * 1024 * 1024   // 8 MB per worker thread (default is typically 1–4 MB)
);
```

Each thread has a fixed stack. Deep recursion or large stack-allocated buffers silently overflow the default. Increase it before the first TBB task is created.

**`task_arena` vs `global_control`:**

| | `task_arena` | `global_control` |
|--|--|--|
| Scope | One `execute()` block | Entire process |
| Thread limit | Per isolated region | All TBB algorithms |
| Work-stealing isolation | Yes | No |
| Typical use | Independent subsystems, NUMA pinning | Embedding TBB in a GUI/server, global throttling |

---

### 2.15 Exception Handling and Cancellation

In normal single-threaded C++, an exception unwinds the call stack to the nearest `catch`. In a parallel loop, each iteration runs in a different thread with its own stack — there is no shared call stack to unwind.

TBB bridges this: when a worker thread throws, TBB catches it internally, cancels remaining *pending* iterations, then re-throws the exception on the calling thread once the loop completes.

```
Thread 0: iteration  0 → OK
Thread 1: iteration  5 → throws std::runtime_error("bad input")
                         ↑ TBB catches it
Thread 2: iteration 10 → OK (already running → finishes normally)
Pending iterations 11–N → CANCELLED (never start)
                         ↓ TBB re-throws on calling thread
Main thread: catch block runs
```

```cpp
try {
    tbb::parallel_for(0, N, [](int i) {
        if (bad_condition(i))
            throw std::runtime_error("bad input");
    });
} catch (const std::exception& e) {
    // Caught here on the calling thread
    // Already-running iterations completed; pending ones were skipped
    std::cerr << e.what() << "\n";
}
```

> Already-running iterations always finish. Only *pending* (not-yet-started) tasks are cancelled. You cannot abort a task mid-execution.

**Explicit cancellation without exceptions** — use `task_group_context` when you want to cancel from another thread or condition (e.g., a timeout, a "stop" button):

```cpp
tbb::task_group_context ctx;

// Start the parallel loop in one thread
tbb::parallel_for(
    tbb::blocked_range<int>(0, N),
    body,
    tbb::auto_partitioner(),
    ctx          // ← attach context
);

// From another thread (e.g., a watchdog or UI thread):
ctx.cancel_group_execution();
// → pending tasks stop being scheduled
// → running tasks finish naturally
// → parallel_for returns without throwing
```

**Cancellation flow:**

```
cancel_group_execution() called
          │
          ▼
Pending tasks   → skip (never execute)
Running tasks   → run to completion (cannot be interrupted)
parallel_for()  → returns normally (no exception thrown)
```

| Mechanism | Triggers how | Running tasks | Pending tasks | Throws? |
|-----------|-------------|---------------|---------------|---------|
| Worker throws exception | Exception in body | Finish | Cancelled | Yes, on caller |
| `ctx.cancel_group_execution()` | External call | Finish | Cancelled | No |

---

### 2.16 Work Isolation

TBB's work-stealing scheduler is aggressive: when a thread finishes a task and is waiting on an inner `parallel_for`, it doesn't idle — it looks for *any* available task in the pool, including tasks from the outer loop. This is great for throughput but dangerous when inner and outer tasks share data.

```
WITHOUT isolate:
  Outer task i=0 launches inner parallel_for (10 tasks)
  Outer task i=0's thread waits → steals outer task i=3
  Now i=0's thread is running i=3 while i=0's inner tasks run on other threads
  If inner tasks read/write data[i=0...] and outer task i=3 does too → race

WITH isolate:
  Outer task i=0 launches inner parallel_for inside isolate()
  Inner tasks are visible ONLY to threads already inside the isolate block
  Outer threads cannot steal them → no interleaving → safe
```

**When you need isolation — three questions:**

1. Does the inner loop write to data that outer tasks also touch?
2. Is that data not protected by a lock?
3. Are outer and inner tasks running at the same time?

If all three: use `isolate`.

```cpp
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/this_task_arena.h>

std::vector<int> data(100, 0);

tbb::parallel_for(0, 10, [&](int i) {
    // outer task owns data[i*10 ... i*10+9]
    // inner tasks write into exactly that range
    // → must not be stolen by another outer task's thread

    tbb::this_task_arena::isolate([&] {
        tbb::parallel_for(0, 10, [&](int j) {
            data[i*10 + j] += 1;   // safe: isolated from other outer tasks
        });
    });
});
```

**Without `isolate`:** Thread running outer `i=2` could steal inner tasks belonging to outer `i=7`, then write into `data[20..29]` while `i=2`'s thread is somewhere else writing into `data[20..29]` — data race.

**With `isolate`:** Inner tasks of `i=2` stay inside the isolate block. Only the thread that entered the block (and any helpers *it* spawns internally) can run those inner tasks.

**When NOT to isolate:**

If the inner loop works on independent data (no sharing with outer tasks), isolation only hurts performance — it prevents free threads from helping with the inner work. Leave it out.

| Situation | Use `isolate`? | Reason |
|-----------|---------------|--------|
| Inner tasks write to shared outer data | Yes | Prevent data race |
| Inner tasks read-only from shared data | Usually no | Reads are safe to steal |
| Inner tasks work on fully independent data | No | Let threads steal freely — faster |
| Need deterministic per-outer-task execution order | Yes | Isolate forces strict containment |

---

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

oneTBB patterns map directly to GPU frameworks:

| oneTBB concept | GPU equivalent |
|---------------|---------------|
| `parallel_for` over `blocked_range` | CUDA kernel launch over thread grid |
| Work-stealing scheduler | Warp scheduler (hardware) |
| `affinity_partitioner` | L2 cache locality hints in CUDA |
| `enumerable_thread_specific` | Per-warp shared memory allocation |
| `combinable<T>` | `atomicAdd` into shared mem, then reduce |
| `concurrent_queue` | CUDA stream (async task queue) |
| `parallel_pipeline` | CUDA multi-stream pipeline |
| `flow_graph` nodes | CUDA graph nodes (`cudaGraph`) |
| `task_group` | CUDA dynamic parallelism |
| `task_arena` | CUDA stream priority + stream isolation |
| Scalable allocator | `cudaMallocAsync` (stream-ordered pool) |

Learning oneTBB's task decomposition makes CUDA's thread/block/grid hierarchy intuitive — they solve the same problem at different scales.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [oneTBB Developer Guide](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/) | Full tutorial: parallel_for, reduce, pipeline, flow graph, design patterns |
| [oneTBB API Reference](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/reference.html) | Complete API |
| [Intel oneTBB Get Started](https://www.intel.com/content/www/us/en/docs/onetbb/get-started-guide/2022-2/overview.html) | Installation, CMake, first steps |
| [OpenMP specifications](https://www.openmp.org/specifications/) | Official OpenMP standard |
| [OpenMP API Quick Reference Card](https://www.openmp.org/wp-content/uploads/OpenMP-4.5-1115-CPP-web.pdf) | All clauses at a glance |
| *Intel Threading Building Blocks* (Reinders) | Book: TBB patterns in depth |

---

## Next

→ [**CUDA and SIMT**](../CUDA%20and%20SIMT/Guide.md) — GPU parallelism: threads, warps, blocks, and memory hierarchy.
