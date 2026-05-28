# C++ and SIMD (Phase 1 §4 — Sub-Track 1)

<div class="course-identity auto-course" style="--course-accent: #4f46e5; --course-accent-rgb: 79, 70, 229;" markdown="1">
<div class="course-identity__icon">ASPS</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Digital Foundations</p>
<p class="course-identity__title">Specialized course identity for C++ and SIMD (Phase 1 §4 — Sub-Track 1).</p>
<p class="course-identity__meta">Artifact: working low-level demo · Measure: timing, memory, correctness</p>
</div>
</div>


**Parent:** [C++ and Parallel Computing](../Guide.md)

> *Modern C++ gives you high-level syntax with low-level performance. Master the language features first — they are the building blocks of every parallel framework that follows.*

**Prerequisites:** Basic C programming (from Phase 1 §3 OS work). Familiarity with pointers, arrays, functions.

---

## Why Modern C++ First

Every parallel computing technology in this curriculum — SIMD intrinsics, OpenMP, oneTBB, CUDA, HIP, SYCL — is built on C++. But not the C++ of the 1990s. Modern C++ (C++11 through C++17) introduced features specifically designed for performance, safety, and parallel programming:

- **Lambdas** → used in oneTBB, SYCL, modern CUDA, parallel STL
- **Move semantics** → zero-copy data transfer, critical for GPU memory
- **Templates** → generic CUDA kernels, CUTLASS, CuTe
- **Parallel STL** → built-in SIMD + threading with one line of code
- **`constexpr`** → compile-time computation for HPC optimization

Learn these features now. You'll use every single one in Sub-Tracks 2–5.

---

## Section 1: Modern C++17 for Parallel Computing

### 1.1 Type Deduction (`auto`, `decltype`)

**`auto` — let the compiler figure out the type:**

```cpp
auto x = 10;          // int
auto y = 3.14;        // double
auto z = vec.begin(); // std::vector<int>::iterator — much cleaner
```

`auto` reduces verbosity and prevents type mismatch bugs. Use it for local variables, especially with complex types (iterators, template results).

---

#### The problem `decltype` solves

In a generic multiply function, what is the return type?

```cpp
template<typename T, typename U>
??? multiply(T a, U b) {
    return a * b;
}
// int   * int    → int
// int   * double → double
// float * double → double
```

**Pre-C++11 workarounds — both inadequate:**

```cpp
// Option 1: force a type — loses precision, not generic
template<typename T, typename U>
double multiply(T a, U b) { return a * b; }

// Option 2: std::common_type — verbose, breaks for custom operators
template<typename T, typename U>
typename std::common_type<T, U>::type multiply(T a, U b) { return a * b; }
```

**C++11 solution — trailing return type with `decltype`:**

```cpp
template<typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {
    return a * b;
}
```

The compiler deduces the **exact type of the expression `a * b`** — works for built-in types, user-defined operators, and arbitrarily complex template expressions.

**C++14 simplification — `decltype` inferred from the return statement:**

```cpp
template<typename T, typename U>
auto multiply(T a, U b) {   // return type deduced automatically
    return a * b;
}
```

No trailing `-> decltype(...)` needed. The compiler infers from the `return` expression.

---

#### `auto` vs `decltype` — the key difference

`auto` strips references and `const`. `decltype` preserves exact type semantics.

```cpp
int x = 5;
int& ref = x;

auto        a = ref;  // a is int   — reference stripped
decltype(ref) b = x;  // b is int&  — reference preserved

const int c = 10;
auto        d = c;    // d is int        — const stripped
decltype(c) e = c;    // e is const int  — const preserved
```

**Rule of thumb:**
- Use `auto` → 90% of the time (local variables, loop iterators, lambda captures)
- Use `decltype` → when you need **exact type semantics** (template return types, forwarding, trait-style code)

---

#### Comparison: old vs modern

| | Pre-C++11 | C++11 `decltype` | C++14 `auto` return |
|---|---|---|---|
| Return type deduction | Manual, error-prone | Exact expression type | Inferred from `return` |
| Generic correctness | Limited | Exact | Exact |
| Custom operator support | Hard | Yes | Yes |
| Readability | Verbose | Clean | Cleanest |

---

**Why it matters for parallel computing:** CUTLASS, CuTe, oneAPI DPC++, and ROCm all use `auto` + `decltype` extensively because:
- Kernels are templated on precision (`float`, `half`, `int8_t`)
- Types of intermediate expressions depend on template parameters
- `decltype` lets the library express "whatever type `a*b` produces" without hardcoding it

You will see patterns like `decltype(auto)`, `std::declval<T>()`, and trailing return types throughout CUDA template libraries — recognizing them now saves significant confusion later.

---

### 1.2 Smart Pointers (Memory Safety via RAII)

**The problem with raw pointers:**

```cpp
int* p = new int(5);
// ... 200 lines of code ...
// Did you remember to delete p? Memory leak.
// Did you delete it twice? Undefined behavior.
```

**The solution — RAII (Resource Acquisition Is Initialization):**

```cpp
#include <memory>

// Exclusive ownership — automatically freed when scope ends
auto p = std::make_unique<int>(5);
// No delete needed. Ever.

// Shared ownership — freed when last reference dies
auto q = std::make_shared<int>(10);
auto r = q;  // Reference count = 2
// Freed when both q and r go out of scope
```

**Three smart pointer types:**

| Type | Ownership | Overhead | Use when |
|------|-----------|----------|----------|
| `std::unique_ptr` | Exclusive (one owner) | Zero (same as raw pointer) | Default choice. Single owner. |
| `std::shared_ptr` | Shared (reference counted) | Atomic ref count (control block overhead, impl-dependent) | Multiple owners needed |
| `std::weak_ptr` | Non-owning observer | Minimal | Break circular references |

**Rule:** Never use `new`/`delete` in modern C++. Use `make_unique` and `make_shared`.

---

#### Breaking Cycles with `std::weak_ptr`

`shared_ptr` uses reference counting. The count only reaches zero — and the memory only frees — when **no `shared_ptr` points to the object**. If two objects hold `shared_ptr` to each other, neither count ever reaches zero. This is a **reference cycle** and it silently leaks memory forever.

**The cycle problem:**

```cpp
struct Node {
    std::shared_ptr<Node> next;   // strong reference
    int value;
};

auto a = std::make_shared<Node>();  // a: refcount = 1
auto b = std::make_shared<Node>();  // b: refcount = 1

a->next = b;   // b refcount = 2
b->next = a;   // a refcount = 2

// a and b go out of scope:
//   a's refcount drops to 1 (b->next still holds it)
//   b's refcount drops to 1 (a->next still holds it)
//   Neither reaches 0 → MEMORY LEAK
```

**The fix — `weak_ptr` for the back-edge:**

`weak_ptr` observes a `shared_ptr` **without incrementing the reference count**. The observed object can still be destroyed normally. Before using a `weak_ptr`, you must `lock()` it — this returns a `shared_ptr` that is either valid (object alive) or empty (object was destroyed).

```cpp
struct Node {
    std::shared_ptr<Node> next;   // strong: owns the next node
    std::weak_ptr<Node>   prev;   // weak: observes without owning
    int value;
};

auto a = std::make_shared<Node>();  // a: refcount = 1
auto b = std::make_shared<Node>();  // b: refcount = 1

a->next = b;        // b refcount = 2  (strong)
b->prev = a;        // a refcount still = 1  (weak — no increment)

// a and b go out of scope:
//   a's refcount drops to 0 → destroyed → a->next released
//   b's refcount drops to 0 → destroyed
//   No leak.

// Accessing through weak_ptr — always check if still alive:
if (auto owner = b->prev.lock()) {   // lock() returns shared_ptr or nullptr
    std::cout << "prev value: " << owner->value << "\n";
} else {
    std::cout << "prev was destroyed\n";
}
```

**When cycles appear in practice:**

| Pattern | Cycle edge | Fix |
|---------|------------|-----|
| Doubly-linked list | `prev` pointer | `prev` as `weak_ptr` |
| Tree with parent pointer | `parent` pointer | `parent` as `weak_ptr` |
| Observer / event system | Subscriber holds reference to publisher | Subscriber stores `weak_ptr` to publisher |
| Computation graph (ML) | Back-edge from output node to input | Back-edges as `weak_ptr` |

**`weak_ptr` API:**

```cpp
auto sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;          // observe, no refcount increment

wp.expired();                        // true if the object was destroyed
wp.use_count();                      // same as sp.use_count() (0 if expired)
auto sp2 = wp.lock();                // returns shared_ptr<int> or nullptr — ALWAYS use this
if (sp2) { /* safe to use */ }
```

> **Do not dereference a `weak_ptr` directly.** Always call `.lock()` and check the result. The object could be destroyed between `expired()` returning `false` and your next line.

---

**Why it matters for parallel computing:**
- GPU memory (`cudaMalloc`/`cudaFree`) follows the same RAII pattern — wrap in a smart pointer or custom RAII class
- Thread-safe reference counting (`shared_ptr`) is essential for multi-threaded data sharing
- Computation graphs (PyTorch autograd, JAX jaxpr) are graphs with cycles — real implementations use weak back-edges to avoid memory leaks
- Memory leaks in HPC = your 80 GB GPU runs out of memory mid-training

---

### 1.3 Range-Based Loops

```cpp
std::vector<float> data = {1.0, 2.0, 3.0, 4.0};

// Modern (clean, less error-prone)
for (auto& x : data) {
    x *= 2.0f;
}

// Old style (verbose, index bugs)
for (int i = 0; i < data.size(); i++) {
    data[i] *= 2.0f;
}
```

**Why it matters:** Range-based loops work naturally with STL algorithms and parallel STL (`std::execution::par`). They're also easier for the compiler to auto-vectorize.

---

### 1.4 Structured Bindings (C++17)

Unpack tuples, pairs, and structs directly:

```cpp
// Pair unpacking
std::pair<int, float> p = {1, 2.5f};
auto [id, value] = p;   // id = 1, value = 2.5f

// Map iteration (clean)
std::map<std::string, int> scores = {{"alice", 95}, {"bob", 87}};
for (auto& [name, score] : scores) {
    std::cout << name << ": " << score << "\n";
}

// Function returning multiple values
auto [min_val, max_val] = std::minmax_element(data.begin(), data.end());
```

**Why it matters:** Cleaner data handling in performance-critical code. Used in profiling results, benchmark data, multi-return-value functions.

---

### 1.5 Lambda Functions (Critical for Parallel Computing)

Lambdas are **the single most important C++ feature for parallel programming**. Every parallel framework uses them.

**Syntax:**

```
[ captures ] ( parameters ) -> return_type { body }
```
- `captures` → which variables from the outer scope to bring in
- `parameters` → like a normal function
- `return_type` → optional, usually inferred
- `body` → code block

**Basic lambda:**

```cpp
auto add = [](int a, int b) {
    return a + b;
};

int result = add(3, 4);  // 7
```

---

#### Capture Modes

| Mode | Meaning | Thread-safe? | Use when |
|------|---------|-------------|---------|
| `[=]` | All outer vars by value (copy) | Yes | Default for parallel callbacks |
| `[&]` | All outer vars by reference | No | Single-threaded only |
| `[var]` | Specific var by value | Yes | Explicit, recommended |
| `[&var]` | Specific var by reference | No | Read-only, single-threaded |
| `[=, &var]` | Mixed: most by value, one by ref | Partial | Fine-grained control |
| `[var = std::move(obj)]` | Move object into lambda | Yes | Transfer ownership (C++14) |

```cpp
int multiplier = 10;

auto scale_copy     = [=](int x)       { return x * multiplier; };  // copy — safe in threads
auto scale_ref      = [&](int x)       { return x * multiplier; };  // ref — dangerous in parallel
auto scale_specific = [multiplier](int x) { return x * multiplier; };  // best practice
```

---

#### Mutable Lambdas

By default, values captured by value are `const` inside the lambda. Use `mutable` to modify the internal copy without touching the original:

```cpp
int a = 10;

auto f = [a]() mutable {
    a += 5;
    std::cout << a;  // prints 15
};
f();
std::cout << a;  // still 10 — original unchanged
```

`mutable` modifies the lambda's **own copy**, not the outer variable.

---

#### Generic Lambdas (C++14)

Use `auto` parameters to make a lambda templated automatically — no explicit `template<>` needed:

```cpp
auto add = [](auto a, auto b) { return a + b; };

add(3, 4);        // int + int = 7
add(3.14, 2.71);  // double + double = 5.85
add(1.0f, 2.0f);  // float + float
```

This is used heavily in parallel STL, oneTBB, and SYCL to write type-generic kernels.

---

#### Thread Safety Rules

**Capture by value `[x]`** — each thread gets its own copy. Safe.

**Capture by reference `[&x]`** — all threads share the same variable. Race condition if any thread writes to it.

```cpp
std::vector<int> data = {1, 2, 3, 4};
int multiplier = 2;

// Safe: each element read is independent, multiplier captured by value
std::for_each(std::execution::par, data.begin(), data.end(),
              [multiplier](int& x) { x *= multiplier; });

// Unsafe: if multiplier were captured by ref and modified by any thread
std::for_each(std::execution::par, data.begin(), data.end(),
              [&](int& x) { x *= multiplier; });  // race condition if multiplier changes
```

> **Rule: prefer `[=]` or named value captures in parallel loops. Use `[&]` only in single-threaded code or when the captured variables are read-only.**

---

#### Lambda Lifetime — Dangling Reference Trap

If a lambda outlives the scope of its captured references, you get a dangling reference:

```cpp
std::function<int()> make_lambda() {
    int local = 42;
    return [&]() { return local; };  // DANGLING — local is destroyed when function returns
}

auto f = make_lambda();
f();  // undefined behavior
```

**Fix 1 — capture by value:**

```cpp
return [local]() { return local; };  // safe copy
```

**Fix 2 — move ownership into the lambda (C++14):**

```cpp
auto buf = std::make_unique<int>(42);
std::thread t([buf = std::move(buf)]() {
    std::cout << *buf;  // safe: ownership moved into lambda
});
t.join();
```

`[buf = std::move(buf)]` transfers ownership of the `unique_ptr` into the lambda. The lambda now owns the resource — it cannot outlive it.

---

#### Move-Into-Lambda — Deep Dive

This is worth understanding completely because it appears constantly in parallel and async code.

**Why `unique_ptr` cannot be captured by value:**

```cpp
std::unique_ptr<int> ptr = std::make_unique<int>(42);

auto f = [ptr]() { std::cout << *ptr; };  // DOES NOT COMPILE
// unique_ptr has no copy constructor — copying would violate unique ownership
```

`unique_ptr` expresses "exactly one owner". Capture-by-value would require copying the pointer, creating two owners. The compiler forbids it.

**Solution — capture by move (C++14):**

```cpp
auto f = [p = std::move(ptr)]() {
    std::cout << *p << "\n";   // lambda owns it, safe to use
};

// What happened to ptr?
std::cout << (ptr ? "not empty" : "empty") << "\n";  // "empty" — ptr is now nullptr
f();  // prints 42
```

**Ownership transfer visualized:**

```
Before move:
  ptr ──────────────► [ 42 ]   (heap)

After [p = std::move(ptr)]:
  ptr ──► nullptr
  lambda.p ─────────► [ 42 ]   (same heap allocation, new owner)
```

No copy. No new allocation. The lambda took the backpack — the original holder is now empty.

**Thread ownership — each thread gets its own resource:**

```cpp
#include <thread>
#include <memory>

int main() {
    auto buf = std::make_unique<int[]>(1000);  // 4 KB buffer

    std::thread t([b = std::move(buf)]() {
        b[0] = 42;
        std::cout << b[0] << "\n";  // lambda owns the buffer exclusively
    });

    t.join();
    // buf is empty here — the thread owned and (after join) released it
}
```

The thread lambda owns `buf` exclusively. No data race. No need for a mutex. When the thread finishes, the lambda destructs and `unique_ptr` frees the memory automatically.

**Moving other moveable types:**

Any type with a move constructor works — not just `unique_ptr`:

```cpp
std::vector<float> weights(1000000);   // 4 MB
std::string config = load_config();
auto handle = open_gpu_context();      // hypothetical RAII GPU handle

// Move all three into a thread lambda — zero copying
std::thread worker([
    w = std::move(weights),
    cfg = std::move(config),
    ctx = std::move(handle)
]() {
    // worker owns all three resources
    run_inference(w, cfg, ctx);
});
worker.join();
```

| What you move | Why |
|--------------|-----|
| `unique_ptr<T>` | Exclusive GPU/CPU resource handle |
| `vector<float>` | Large weight/activation buffer |
| `string` | Config or serialized model data |
| `std::thread` | Transfer thread ownership to lambda |
| Custom RAII handle | GPU context, file, socket |

**Rule:** If a type is non-copyable (or copying is expensive), and the lambda needs to own it or outlive the current scope, use `[x = std::move(obj)]`.

---

#### Device Lambdas (CUDA / SYCL)

Modern CUDA and SYCL support lambdas on GPU device code:

```cpp
// CUDA — device lambda inside kernel
__global__ void kernel(float* a, float* b, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        auto square = [=](float x) { return x * x; };  // [=] only — no refs on device
        a[i] = square(b[i]);
    }
}

// SYCL — lambda IS the kernel
q.parallel_for(sycl::range{N}, [=](sycl::id<1> i) {
    C[i] = A[i] + B[i];
});
```

> **`[&]` is not allowed in GPU device lambdas.** Device code cannot reference host memory addresses. Always use `[=]` (capture by value) for GPU lambdas.

---

#### Lambda Quick Reference

| Feature | Syntax | Notes |
|---------|--------|-------|
| Capture all by value | `[=]` | Safe in threads |
| Capture all by ref | `[&]` | Dangerous in parallel |
| Specific var by value | `[var]` | Best practice |
| Specific var by ref | `[&var]` | Single-threaded only |
| Generic (auto params) | `[](auto x, auto y){}` | C++14, type-generic |
| Mutable | `[x]() mutable {}` | Modify internal copy |
| Move into lambda | `[x = std::move(obj)]` | Transfer ownership, C++14 |
| Immediately invoked | `[&]() { ... }()` | IIFE — run in-place |

---

**Where lambdas are used in this curriculum:**

| Framework | Lambda usage |
|-----------|-------------|
| **Parallel STL** | `std::sort(std::execution::par, v.begin(), v.end(), [](auto a, auto b) { return a > b; });` |
| **oneTBB** | `tbb::parallel_for(0, N, [&](int i) { C[i] = A[i] + B[i]; });` |
| **OpenMP** (C++17) | `#pragma omp parallel for` + lambda tasks: `omp_set_num_threads(N); #pragma omp task` |
| **CUDA** (modern) | Device lambdas in `__device__` context — `[=]` only |
| **SYCL** | `q.parallel_for(range, [=](id<1> i) { C[i] = A[i] + B[i]; });` |

**Key message:** If you don't understand lambdas and captures, you cannot write parallel code in modern C++.

---

### 1.6 `std::optional`, `std::variant`, `std::any` (C++17)

**`std::optional` — a value that may or may not exist:**

```cpp
#include <optional>

std::optional<int> find_index(const std::vector<int>& v, int target) {
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == target) return i;
    }
    return std::nullopt;  // "not found" — no magic numbers like -1
}

auto result = find_index(data, 42);
if (result) {
    std::cout << "Found at index " << *result << "\n";
}
```

**`std::variant` — type-safe union:**

#### `union` vs `std::variant` — why the old way is dangerous

The old C `union` stores multiple types in the same memory, but has no idea which type is currently active:

```cpp
union Data {
    int   i;
    float f;
};

Data d;
d.i = 42;
std::cout << d.f << "\n";  // undefined behavior — reading int bits as float
```

Memory layout of a `union` — 4 bytes, no type information:

```
+------------------+
|  42 (int bits)   |   <- what's stored
+------------------+
  ^ no type tag     <- compiler has no idea which member is active
```

Reading the wrong member is **undefined behavior**. No compiler error, no runtime check — just wrong results or crashes, often appearing only in release builds.

`std::variant` adds a **type tag** alongside the stored value:

```
+------------------+
|  3.14f (float)   |   <- stored data
+------------------+
  type tag: float       <- runtime always knows which type is active
```

```cpp
#include <variant>

std::variant<int, float, std::string> v;
v = 42;         // holds int   — type tag = int
v = 3.14f;      // holds float — type tag = float
v = "hello";    // holds string — type tag = string

// std::visit dispatches to the right lambda branch based on the type tag
std::visit([](auto&& val) { std::cout << val << "\n"; }, v);
```

#### `union` vs `std::variant` comparison

| | `union` | `std::variant` |
|--|---------|----------------|
| Type tracking | None — programmer's responsibility | Runtime type tag stored alongside value |
| Wrong-type read | Undefined behavior, silent | `std::bad_variant_access` exception |
| Size overhead | Zero (just the max member size) | Max member size + small tag (usually 1–8 bytes) |
| Parallel safety | Dangerous — type confusion across threads | Safe — each element carries its own type info |
| Use in HPC | Avoid in new code | Heterogeneous arrays, command dispatch, AST nodes |

#### `std::visit` — type-dispatching with a lambda

`std::visit` calls your lambda with the **actually stored type**, not a generic variant:

```cpp
std::variant<int, float> v = 3.14f;

// Generic lambda — works for any type in the variant
std::visit([](auto&& x) {
    std::cout << x << "\n";                           // prints 3.14
    std::cout << typeid(x).name() << "\n";            // prints "f" (float)
}, v);

// Typed overloads using overload pattern (C++17)
std::visit(overloaded{
    [](int   x) { std::cout << "int: "   << x << "\n"; },
    [](float x) { std::cout << "float: " << x << "\n"; },
}, v);  // prints: float: 3.14
```

#### Parallel example — mixed-type array processed safely

```cpp
#include <variant>
#include <vector>
#include <algorithm>
#include <execution>

std::vector<std::variant<int, float>> data = {1, 2.5f, 3, 4.5f};

// Double every element in parallel — type-safe, no UB
std::for_each(std::execution::par, data.begin(), data.end(),
    [](auto& val) {
        std::visit([](auto&& x) { x *= 2; }, val);
    });

// Output: 2  5  6  9
for (auto& v : data)
    std::visit([](auto&& x) { std::cout << x << " "; }, v);
```

Each element carries its own type tag, so parallel threads never confuse `int` and `float` even when accessing adjacent elements.

**Why it matters for HPC:** Compiler IR nodes, kernel dispatch tables, and mixed-precision computation graphs (FP32/FP16/INT8 hybrid inference) all need to store heterogeneous types safely. `std::variant` + `std::visit` replaces the old `void*`-with-a-type-enum pattern with zero undefined behavior.

---

### 1.7 Move Semantics (Performance Core)

**The problem — expensive copies:**

```cpp
std::vector<float> create_large_buffer() {
    std::vector<float> buf(10000000);  // 40 MB
    // ... fill buffer ...
    return buf;  // Does this copy 40 MB? NO — move semantics!
}
```

**Copy vs move:**

```cpp
std::vector<int> a = {1, 2, 3, 4, 5};

// Copy: duplicates all data (expensive)
std::vector<int> b = a;           // b is a copy, a unchanged

// Move: transfers ownership (cheap — just pointer swap)
std::vector<int> c = std::move(a); // c owns the data, a is now empty
```

**How move works internally:**

```
Before move:
  a → [heap: 1, 2, 3, 4, 5]    (owns the buffer)

After std::move(a) → c:
  a → nullptr                    (empty, moved-from)
  c → [heap: 1, 2, 3, 4, 5]    (now owns the buffer)
```

No data was copied. Just three pointer/size assignments. O(1) instead of O(n).

**When to use `std::move`:**
- Returning large objects from functions (compiler often does this automatically — NRVO)
- Transferring ownership of buffers to another thread
- Moving data into containers: `vec.push_back(std::move(large_object));`

**Why it matters for parallel computing:**
- GPU memory transfers are expensive. Move semantics = zero-copy mindset.
- Thread pools move work items between threads without copying.
- CUDA's `cudaMemcpyAsync` + pinned memory is the GPU equivalent of move semantics.

---

### 1.8 Multithreading Basics (`std::thread`, `std::mutex`)

**Launching a thread:**

```cpp
#include <thread>

void compute(int id) {
    std::cout << "Thread " << id << " running\n";
}

int main() {
    std::thread t1(compute, 1);
    std::thread t2(compute, 2);

    t1.join();  // Wait for t1 to finish
    t2.join();  // Wait for t2 to finish
}
```

**Protecting shared data with mutex:**

```cpp
#include <mutex>

std::mutex mtx;
int shared_counter = 0;

void increment(int n) {
    for (int i = 0; i < n; i++) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII lock
        shared_counter++;
    }  // lock released here automatically
}
```

**Better: `std::scoped_lock` (C++17) for multiple mutexes:**

```cpp
std::mutex m1, m2;
{
    std::scoped_lock lock(m1, m2);  // Locks both, prevents deadlock
    // ... critical section ...
}
```

**`std::atomic` for simple shared variables:**

```cpp
#include <atomic>

std::atomic<int> counter{0};

void increment(int n) {
    for (int i = 0; i < n; i++) {
        counter++;  // Thread-safe, no mutex needed
    }
}
```

**Why it matters:** `std::thread` and `std::mutex` are the primitives that OpenMP and oneTBB abstract over. Understanding them helps you debug race conditions and deadlocks in any parallel framework.

---

### 1.9 STL Algorithms and Parallel STL (C++17)

STL algorithms are pre-built, optimized functions that operate on any container. They eliminate manual loops, prevent index bugs, and — with C++17 execution policies — run in parallel with a single extra argument.

#### Core STL Algorithms

| Algorithm | What it does | Example |
|-----------|-------------|---------|
| `std::sort` | Sort elements | `sort(v.begin(), v.end())` |
| `std::for_each` | Apply function to every element | `for_each(v.begin(), v.end(), f)` |
| `std::transform` | Apply function, write to output | `transform(a, a+n, b, out, f)` |
| `std::reduce` | Parallel-safe sum/product/etc. | `reduce(v.begin(), v.end())` |
| `std::find_if` | Search by predicate | `find_if(v.begin(), v.end(), pred)` |
| `std::minmax_element` | Min and max in one pass | `auto [lo, hi] = minmax_element(...)` |
| `std::count_if` | Count matching elements | `count_if(v.begin(), v.end(), pred)` |
| `std::copy_if` | Filtered copy | `copy_if(src, src+n, dst, pred)` |

#### Sequential example — sort and iterate

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

std::vector<int> v = {4, 2, 7, 1, 5};

std::sort(v.begin(), v.end());

for (auto x : v) std::cout << x << " ";  // 1 2 4 5 7
```

#### Lambda + algorithm — custom behavior injected

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};

// Multiply every element by 2 — no loop index, no off-by-one
std::for_each(v.begin(), v.end(), [](int& x) { x *= 2; });
// v = {2, 4, 6, 8, 10}

// Transform: square each element into a new vector
std::vector<int> sq(v.size());
std::transform(v.begin(), v.end(), sq.begin(), [](int x) { return x * x; });

// Find first element > 5
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });
```

#### Parallel STL — one argument changes everything

C++17 execution policies parallelize any STL algorithm:

```cpp
#include <algorithm>
#include <execution>
#include <vector>

std::vector<float> data(10'000'000);

// Sequential (default, single core)
std::sort(std::execution::seq,      data.begin(), data.end());

// Parallel (multi-threaded across all cores)
std::sort(std::execution::par,      data.begin(), data.end());

// Parallel + SIMD (threads + vectorization)
std::sort(std::execution::par_unseq, data.begin(), data.end());
```

**Execution policies:**

| Policy | Behavior | When to use |
|--------|----------|-------------|
| `seq` | Sequential | Small data, debugging |
| `par` | Multi-threaded | Large data, independent elements |
| `par_unseq` | Multi-threaded + SIMD | Maximum throughput, no dependencies |
| `unseq` (C++20) | SIMD only (single thread) | Vectorizable, single core |

#### Parallel transform, reduce, for_each

```cpp
// Parallel transform: c[i] = a[i] + b[i]
std::transform(std::execution::par,
    a.begin(), a.end(), b.begin(), c.begin(),
    [](float x, float y) { return x + y; });

// Parallel reduce: sum all elements
float total = std::reduce(std::execution::par, data.begin(), data.end());

// Parallel reduce with custom op: product
float product = std::reduce(std::execution::par,
    data.begin(), data.end(), 1.0f, std::multiplies<float>{});

// Parallel for_each + SIMD: sqrt every element
std::for_each(std::execution::par_unseq, data.begin(), data.end(),
    [](float& x) { x = std::sqrt(x); });
```

#### Manual loop vs STL vs Parallel STL

| | Manual loop | STL + lambda | Parallel STL |
|--|------------|--------------|--------------|
| Code length | Long | Short | Short |
| Index bug risk | Medium | None | None |
| Parallelism | Manual (`std::thread`) | Manual | Automatic |
| SIMD | Compiler may vectorize | Compiler may vectorize | `par_unseq` forces it |
| Debugging | Hard | Easy | Harder (non-deterministic) |

> **`par_unseq` requirements:** Loop body must have **no data races** AND **no loop-carried dependencies** (`a[i]` must not read `a[i-1]`). Violating either produces undefined behavior — not a compile error, not even consistent wrong results.

> **Warning — not always faster:** Parallel STL has overhead (thread pool spin-up, synchronization). For small arrays (< ~100K elements), `seq` is often faster. In HPC and production ML systems, teams frequently prefer **OpenMP** or **oneTBB** for more predictable, tunable, and debuggable parallelism. Benchmark before committing to `par` or `par_unseq`.

**Compiler support:** GCC 9+ (with `-ltbb`), MSVC 2017+, Intel DPC++. Clang support is improving.

---

### 1.10 Templates & Generic Programming

**Function templates:**

```cpp
template<typename T>
T add(T a, T b) {
    return a + b;
}

add(3, 4);        // T = int
add(3.14, 2.71);  // T = double
```

**Class templates:**

```cpp
template<typename T, int N>
struct Vector {
    T data[N];
    T& operator[](int i) { return data[i]; }
};

Vector<float, 4> v;  // 4-element float vector
```

**Why templates matter for parallel computing:**
- CUDA kernels are often templated on data type and tile size: `matmul<float, 32><<<grid, block>>>(...)`
- CUTLASS is entirely template-based — configurable GEMM dimensions, precisions, and tiling
- CuTe (CUDA Template Engine) uses templates for layout and copy abstractions
- Zero-cost abstraction: templates are resolved at compile time, no runtime overhead

---

### 1.11 `constexpr` (Compile-Time Computation)

**Move computation from runtime to compile time:**

```cpp
constexpr int square(int x) {
    return x * x;
}

constexpr int tile_size = square(16);  // Computed at compile time = 256

// Use in array declarations, template parameters, etc.
float buffer[tile_size];  // float buffer[256] — no runtime cost
```

**`constexpr` + `if` (C++17):**

```cpp
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        // Integer-specific code — compiled away for floats
    } else {
        // Float-specific code — compiled away for integers
    }
}
```

**Why it matters for HPC:**
- Tile sizes, block dimensions, and buffer sizes in GPU kernels are often compile-time constants
- `constexpr` enables the compiler to optimize aggressively (unroll loops, eliminate branches)
- Used extensively in CUTLASS for configurable kernel parameters

---

### 1.12 How C++ Features Map to Parallel Frameworks

| C++ Feature | Used in | How |
|-------------|---------|-----|
| **Lambdas** | oneTBB, SYCL, Parallel STL, modern CUDA | Kernel functions, task bodies, parallel_for callbacks |
| **Move semantics** | GPU memory, thread pools | Zero-copy buffer transfer, work item passing |
| **Templates** | CUDA, CUTLASS, CuTe, CK | Configurable kernels, generic algorithms |
| **`constexpr`** | CUDA, HLS, HPC libraries | Compile-time tile sizes, buffer dimensions |
| **`auto`** | Everywhere | Complex return types, template deduction |
| **Smart pointers** | Resource management | RAII wrappers for GPU memory, file handles |
| **Parallel STL** | CPU parallelism | One-line SIMD + threading |
| **`std::thread`/`mutex`** | Manual threading | Foundation for OpenMP, oneTBB internals |
| **`std::atomic`** | Lock-free algorithms | Counters, flags in multi-threaded code |
| **Structured bindings** | Data processing | Clean iteration over results, tuples |

---

## Section 2: SIMD on CPUs

> *The gateway to GPU thinking — same operation, multiple data.*

Now that you have the C++ foundation, apply it to the first level of parallelism: SIMD vector instructions.

### What SIMD Is

CPU vector instructions that process 4, 8, 16, or 32 data elements in one instruction:

| Instruction Set | Width | Data per instruction | Platform |
|----------------|-------|---------------------|----------|
| SSE (1999) | 128-bit | 4x float32 | x86 |
| AVX (2011) | 256-bit | 8x float32 | x86 |
| AVX-512 (2017) | 512-bit | 16x float32 | x86 (server) |
| ARM NEON | 128-bit | 4x float32 | ARM (Jetson, mobile) |

### From SIMD to GPU — One Idea, Three Expressions

Every level of the parallel stack applies **the same operation to multiple data elements at once**. What changes is the width, the programming model, and who manages the indices.

Take the simplest possible operation: `c[i] = a[i] + b[i]`. Here is how you express it at each level:

```
Scalar (1 element at a time):
  c[0] = a[0] + b[0]
  c[1] = a[1] + b[1]   ← you write a loop, CPU executes one add per cycle
  c[2] = a[2] + b[2]
  ...

CPU SIMD / AVX (8 elements at a time):
  c[0..7] = a[0..7] + b[0..7]  ← you write i += 8, CPU runs 8 adds in one instruction
  c[8..15] = ...

GPU warp / CUDA (32 elements at a time):
  each thread handles one i      ← you write kernel for one element,
  32 threads run in lock-step       GPU runs 32 threads simultaneously (one warp)
  c[0..31] = a[0..31] + b[0..31]

GPU grid (thousands at a time):
  all threads launch at once     ← GPU runs thousands of warps across SM clusters
```

```cpp
// ── Scalar ──────────────────────────────────────────────────────
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];            // 1 add/cycle

// ── CPU SIMD (AVX) ───────────────────────────────────────────────
for (int i = 0; i < n; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    _mm256_storeu_ps(&c[i], _mm256_add_ps(va, vb));  // 8 adds/cycle
}

// ── GPU (CUDA) ───────────────────────────────────────────────────
__global__ void add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // each thread owns one i
    if (i < n) c[i] = a[i] + b[i];                  // 32 threads run this simultaneously
}
```

**The key differences — not a new idea, just new mechanics:**

| | Scalar | CPU SIMD (AVX) | GPU (CUDA warp) |
|--|--------|---------------|-----------------|
| Width | 1 element | 8 floats | 32 threads |
| Who picks `i` | Your loop counter | Your stride (`+= 8`) | Hardware (`threadIdx + blockIdx`) |
| Fast memory | L1/L2 cache | L1/L2 cache | Shared memory (48 KB/SM) |
| Slow memory | RAM | RAM | Global memory (VRAM) |
| Hiding latency | Out-of-order execution | ILP across loop iterations | Warp switching |
| Divergence cost | Branch misprediction | Breaks vectorization | Serializes warp lanes |

**The scaling ladder:**

```
AVX2:       8 floats  × ~4 GHz × 2 (FMA) = ~64 GFLOPS / core
GPU warp:  32 threads × thousands of warps  → tens of TFLOPS
```

SIMD is the mental foundation. Once you understand why `i += 8` processes 8 elements at once, the GPU model — "just make every thread a lane, and launch millions" — follows naturally.

---

### Auto-Vectorization (Compiler Does It)

The easiest path — write a simple loop, let the compiler vectorize:

```cpp
// This loop is auto-vectorizable
void add_vectors(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

```bash
# Check if compiler vectorized your loop
g++ -O2 -march=native -fopt-info-vec-optimized my_code.cpp
# Output: "loop vectorized using 32 byte vectors" = success
```

**Rules for auto-vectorization:**
- No loop-carried dependencies (`a[i] = a[i-1] + 1` cannot vectorize)
- Simple data types (float, int — not complex objects)
- Aligned memory helps (`alignas(32) float data[1024];`)
- Use `-O2` or `-O3` with `-march=native`

### Manual Intrinsics (Full Control)

When auto-vectorization fails or you need specific instructions:

```cpp
#include <immintrin.h>

void add_vectors_avx(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);   // Load 8 floats from a
        __m256 vb = _mm256_load_ps(&b[i]);   // Load 8 floats from b
        __m256 vc = _mm256_add_ps(va, vb);   // Add 8 pairs at once
        _mm256_store_ps(&c[i], vc);           // Store 8 results to c
    }
}
```

**Key intrinsics patterns:**

| Operation | Intrinsic | What it does |
|-----------|-----------|-------------|
| Load aligned | `_mm256_load_ps(ptr)` | Load 8 **32-byte-aligned** floats — **crashes if misaligned** |
| Load unaligned | `_mm256_loadu_ps(ptr)` | Load 8 floats (any alignment) — safer default |
| Store | `_mm256_store_ps(ptr, v)` | Store 8 floats (aligned) |
| Stream store | `_mm256_stream_ps(ptr, v)` | Write bypassing cache — use for large write-once buffers |
| Add | `_mm256_add_ps(a, b)` | 8 parallel additions |
| Multiply | `_mm256_mul_ps(a, b)` | 8 parallel multiplications |
| **FMA** | `_mm256_fmadd_ps(a, b, c)` | 8 parallel `a*b + c` — **one instruction, not two; also more precise** |
| Broadcast | `_mm256_set1_ps(x)` | Fill all 8 lanes with same value |
| Horizontal add | `_mm256_hadd_ps(a, b)` | Add adjacent pairs within each 128-bit half |
| Compare | `_mm256_cmp_ps(a, b, _CMP_GT_OS)` | Per-lane compare → all-0s or all-1s mask |
| Blend | `_mm256_blendv_ps(a, b, mask)` | Select per-lane: `mask[i] ? b[i] : a[i]` |
| Permute (cross-lane) | `_mm256_permutevar8x32_ps(v, idx)` | Full cross-lane reorder by index vector |
| Gather | `_mm256_i32gather_ps(base, idx, 4)` | Load from non-contiguous addresses |

> **`_mm256_load_ps` alignment:** If `ptr` is not 32-byte aligned, this generates a segfault (`#GP` fault) at runtime — not a compile error. When in doubt, use `_mm256_loadu_ps` (the `u` = unaligned). On modern CPUs the performance difference is negligible.

> **FMA is the core of deep learning:** `_mm256_fmadd_ps(a, b, c)` computes `a*b + c` in a single fused instruction. GEMM (matrix multiplication), convolutions, and transformer attention inner loops are entirely composed of FMA. This is why FLOPS counts are usually reported as "FLOPS of FMA." On AVX2, 8-wide FMA at 2 ops/cycle = **16 GFLOPS/GHz per core**.

**Aligned memory:**

```cpp
// Aligned allocation (required for _mm256_load_ps)
alignas(32) float data[1024];

// Or dynamic allocation
float* data = (float*)aligned_alloc(32, 1024 * sizeof(float));
```

---

### Top 10 Most-Used Intrinsics in Real Codebases

Analysis of SIMD usage across GitHub repositories (production ML frameworks, databases, parsers) shows the same ~10 intrinsics appear in 80%+ of vectorized code. These are the ones worth memorizing first.

| Rank | Intrinsic | ISA | What it does | Why it's everywhere |
|------|-----------|-----|--------------|---------------------|
| 1 | `_mm256_add_ps(a, b)` | AVX | Add 8 float32 | Core of every FP loop |
| 2 | `_mm256_loadu_ps(ptr)` | AVX | Load 8 floats, unaligned | Default load — no alignment constraint |
| 3 | `_mm256_storeu_ps(ptr, v)` | AVX | Store 8 floats, unaligned | Default store |
| 4 | `_mm256_fmadd_ps(a, b, c)` | FMA+AVX2 | `a*b + c` in one instruction | GEMM, convolution, attention |
| 5 | `_mm_add_ps(a, b)` | SSE | Add 4 float32 (128-bit) | SSE compat code, horizontal ops |
| 6 | `_mm_set1_epi32(x)` | SSE2 | Broadcast int to all 4 lanes | Loading integer constants |
| 7 | `_mm256_cmp_ps(a, b, pred)` | AVX | Compare → bitmask | Conditional selection, NaN handling |
| 8 | `_mm_loadu_si128(ptr)` | SSE2 | Load 128-bit integer vector | String/byte processing |
| 9 | `_mm_shuffle_epi32(v, imm)` | SSE2 | Reorder 32-bit int lanes | Data rearrangement, AoS→SoA |
| 10 | `_mm_popcnt_u32(x)` | SSE4.2 | Count set bits in 32-bit int | Database filters, Hamming distance |

**Patterns behind the rankings:**

- **`_ps` (packed single) dominates** — float32 is the workhorse of ML inference and scientific computing. Learn `_ps` variants first.
- **Unaligned loads/stores are preferred** (`_loadu_`, `_storeu_`) — on Haswell and newer, the penalty for cache-line-crossing loads is ≤1 cycle. The code simplicity is worth it.
- **SSE2 128-bit still common** — many codebases use SSE2 for compatibility or for 128-bit sub-operations (horizontal reduction, remainder loops, byte processing).
- **`_mm_shuffle_epi32` is the workhorse for integer rearrangement** — appears in virtually every AoS→SoA conversion, hash function, and string parser.
- **`_mm_popcnt_u32/u64` is an outlier** — not strictly a SIMD register instruction (operates on scalar registers), but uses the SSE4.2 ISA feature bit and dominates database and bit-manipulation code (Hamming distance, counting set bits in filters).

```cpp
// The most common inner loop pattern in real ML code (ranks 2, 4, 3):
for (int i = 0; i < n; i += 8) {
    __m256 a = _mm256_loadu_ps(&A[i]);     // rank 2
    __m256 b = _mm256_loadu_ps(&B[i]);     // rank 2
    acc      = _mm256_fmadd_ps(a, b, acc); // rank 4
}
_mm256_storeu_ps(out, acc);                // rank 3
```

> **Source:** Usage statistics from [simd.info intrinsic statistics](https://simd.info/blog/intrinsic_statistics_repositories_insights_and_patterns/) and analysis of GitHub repositories including production ML frameworks, databases, and codec libraries.

---

### Intrinsics by Functional Category

A different lens — grouped by what you're trying to do, with both SSE (128-bit) and AVX (256-bit) versions side by side. This is how you look them up when solving a problem.

#### 1. Data Movement & Initialization

```cpp
// Load — unaligned is the default today (negligible penalty on modern CPUs)
__m128 a = _mm_loadu_ps(ptr);       // SSE:  4 floats
__m256 b = _mm256_loadu_ps(ptr);    // AVX:  8 floats

// Store — write register back to memory
_mm_storeu_ps(ptr, a);              // SSE
_mm256_storeu_ps(ptr, b);           // AVX

// Broadcast — fill all lanes with one scalar (scaling, bias addition)
__m128 scale = _mm_set1_ps(2.0f);        // SSE:  [2, 2, 2, 2]
__m256 scale = _mm256_set1_ps(2.0f);     // AVX:  [2, 2, 2, 2, 2, 2, 2, 2]

// Set individual lanes (note: highest lane first)
__m256 v = _mm256_set_ps(7,6,5,4,3,2,1,0);   // reverse order
__m256 v = _mm256_setr_ps(0,1,2,3,4,5,6,7);  // natural order (r = reversed param order)
```

#### 2. Arithmetic

```cpp
// Add / Subtract / Multiply / Divide
__m256 r = _mm256_add_ps(a, b);     // a[i] + b[i]
__m256 r = _mm256_sub_ps(a, b);     // a[i] - b[i]
__m256 r = _mm256_mul_ps(a, b);     // a[i] * b[i]
__m256 r = _mm256_div_ps(a, b);     // a[i] / b[i]  (slow — ~20 cycles)

// FMA — the most important arithmetic intrinsic
__m256 r = _mm256_fmadd_ps(a, b, c); // a[i]*b[i] + c[i], single instruction

// Integer arithmetic (add 8 × int32)
__m256i r = _mm256_add_epi32(a, b);
__m256i r = _mm256_mullo_epi32(a, b);  // low 32 bits of each 32×32 product
```

> **`_ps` = packed single (float32). `_pd` = packed double. `_epi32` = packed int32.**
> Learn the suffix system once and every intrinsic name becomes self-describing.

#### 3. Comparison & Selection (Branchless Conditionals)

SIMD has no branches. The fundamental pattern: **compare → mask → blend**.

```cpp
// Step 1: Compare → produces per-lane all-1s (true) or all-0s (false)
__m256 mask = _mm256_cmp_ps(a, b, _CMP_GT_OS);   // mask[i] = a[i] > b[i] ? 0xFFFFFFFF : 0

// SSE equivalent with explicit predicate:
__m128 mask = _mm_cmplt_ps(a, b);                 // mask[i] = a[i] < b[i]

// Step 2: Blend — select elements using mask
__m256 result = _mm256_blendv_ps(if_false, if_true, mask);
// result[i] = mask[i] ? if_true[i] : if_false[i]

// Practical example: clamp to [lo, hi] without any branch
__m256 clamped = _mm256_min_ps(_mm256_max_ps(v, lo), hi);

// Practical example: absolute value without branch
__m256 sign_mask = _mm256_set1_ps(-0.0f);         // sign bit only
__m256 abs_v     = _mm256_andnot_ps(sign_mask, v); // clear sign bit = |v|
```

> **Comparison predicates for `_mm256_cmp_ps`:** `_CMP_EQ_OS`, `_CMP_LT_OS`, `_CMP_LE_OS`, `_CMP_GT_OS`, `_CMP_GE_OS`, `_CMP_NEQ_OS`, `_CMP_UNORD_Q` (NaN check). The `_OS` suffix = ordered, signaling (raises exception on NaN). Use `_OQ` for quiet (no exception).

#### 4. Data Shuffling & Reduction

```cpp
// Shuffle — rearrange within 128-bit half (compile-time imm8 control)
__m128 s = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3,2,1,0));  // SSE: pick 4 elements from a and b
__m256 s = _mm256_shuffle_ps(a, b, imm8);                // AVX: operates per 128-bit half!

// Horizontal add — adds adjacent pairs (use sparingly, ~3 cycles each)
__m128 h = _mm_hadd_ps(a, b);   // [a0+a1, a2+a3, b0+b1, b2+b3]

// Better horizontal sum pattern (see "Horizontal sum" section above)
// hadd is slow — only use it 1-2 times at the very end of a reduction loop
```

> **`_mm_hadd_ps` warning:** Horizontal add is one of the slowest SIMD instructions (~3-5 cycles vs 0.5 for `_mm_add_ps`). Never put it inside a loop. Use it once at the end to collapse an accumulator register — or better, use the `hsum` idiom from the Inspecting SIMD Registers section.

---

### SSE vs AVX Side-by-Side Reference

| Category | SSE (128-bit, 4× float) | AVX (256-bit, 8× float) | Purpose |
|----------|------------------------|------------------------|---------|
| Load | `_mm_loadu_ps(ptr)` | `_mm256_loadu_ps(ptr)` | Read from memory |
| Store | `_mm_storeu_ps(ptr, v)` | `_mm256_storeu_ps(ptr, v)` | Write to memory |
| Broadcast | `_mm_set1_ps(x)` | `_mm256_set1_ps(x)` | Fill all lanes with scalar |
| Add | `_mm_add_ps(a, b)` | `_mm256_add_ps(a, b)` | Element-wise addition |
| Multiply | `_mm_mul_ps(a, b)` | `_mm256_mul_ps(a, b)` | Element-wise multiply |
| FMA | `_mm_fmadd_ps(a, b, c)` | `_mm256_fmadd_ps(a, b, c)` | `a*b + c` |
| Compare | `_mm_cmplt_ps(a, b)` | `_mm256_cmp_ps(a, b, pred)` | Vectorized compare → mask |
| Blend | `_mm_blendv_ps(a, b, m)` | `_mm256_blendv_ps(a, b, m)` | Vectorized if/else |
| Shuffle | `_mm_shuffle_ps(a, b, i)` | `_mm256_shuffle_ps(a, b, i)` | Rearrange elements |
| Hadd | `_mm_hadd_ps(a, b)` | `_mm256_hadd_ps(a, b)` | Adjacent-pair add |

**When to use SSE vs AVX:**
- **AVX for throughput** — 2× the data per instruction, same latency
- **SSE for compatibility** — SSE2 is available on every x86-64 CPU since ~2003
- **SSE for remainder loops** — after the main AVX loop processes `n/8*8` elements, handle the last `n%8` with SSE or scalar

---

### Minimal Examples — Each Intrinsic in Isolation

Before combining intrinsics, see each one do exactly one thing:

```cpp
#include <immintrin.h>

// ── Data Movement ──────────────────────────────────────────────────────────

// loadu: load 8 floats from any address
float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
__m256 v = _mm256_loadu_ps(data);           // v = [1, 2, 3, 4, 5, 6, 7, 8]

// set1: broadcast scalar to every lane
__m256 v_pi = _mm256_set1_ps(3.14f);        // v_pi = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14]

// storeu: write register back to array
float out[8];
_mm256_storeu_ps(out, v);                   // out[] = {1, 2, 3, 4, 5, 6, 7, 8}

// ── Arithmetic ─────────────────────────────────────────────────────────────

__m256 v1 = _mm256_set1_ps(2.0f);           // [2, 2, 2, 2, 2, 2, 2, 2]
__m256 v2 = _mm256_set1_ps(3.0f);           // [3, 3, 3, 3, 3, 3, 3, 3]

__m256 sum     = _mm256_add_ps(v1, v2);     // [5, 5, 5, 5, 5, 5, 5, 5]
__m256 product = _mm256_mul_ps(v1, v2);     // [6, 6, 6, 6, 6, 6, 6, 6]
__m256 fma_res = _mm256_fmadd_ps(v1, v2, v1); // v1*v2 + v1 = [8, 8, 8, 8, 8, 8, 8, 8]

// ── Comparison & Selection (vectorized if/else) ────────────────────────────

__m256 a    = _mm256_setr_ps(1,5,3,7,2,6,4,8);
__m256 b    = _mm256_set1_ps(4.0f);
__m256 mask = _mm256_cmp_ps(a, b, _CMP_GT_OS); // mask[i] = a[i] > 4 ? 0xFFFFFFFF : 0
                                                // [0, 1, 0, 1, 0, 1, 0, 1]
__m256 result = _mm256_blendv_ps(b, a, mask);   // result[i] = mask[i] ? a[i] : b[i]
                                                // [4, 5, 4, 7, 4, 6, 4, 8]

// ── Shuffling ──────────────────────────────────────────────────────────────

__m256 sv = _mm256_setr_ps(0,1,2,3,4,5,6,7);
// _MM_SHUFFLE(d,c,b,a) → lane 0 = src[a], lane 1 = src[b], lane 2 = src[c], lane 3 = src[d]
__m256 shuffled = _mm256_shuffle_ps(sv, sv, _MM_SHUFFLE(0,1,2,3));
// Result: [3,2,1,0, 7,6,5,4] — reversed within each 128-bit half

// ── Horizontal add (slow — only at loop end) ───────────────────────────────

__m256 hv   = _mm256_setr_ps(1,2,3,4,5,6,7,8);
__m256 hsum = _mm256_hadd_ps(hv, hv);
// [1+2, 3+4, 1+2, 3+4 | 5+6, 7+8, 5+6, 7+8] = [3, 7, 3, 7, 11, 15, 11, 15]
```

---

### Complete Example: Scaled Vector Addition

All five core intrinsics working together in a realistic loop — the pattern used in neural network activations, image processing, and signal scaling:

```cpp
#include <immintrin.h>

// Computes: c[i] = (a[i] + b[i]) * scale  for all i
// Processes 8 elements per iteration using AVX
void scaled_add(const float* a, const float* b, float* c, float scale, int n) {
    // 1. set1: broadcast the scalar constant once, outside the loop
    __m256 v_scale = _mm256_set1_ps(scale);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        // 2. loadu: load 8 elements from each input array
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        // 3. add_ps: element-wise addition
        __m256 vsum = _mm256_add_ps(va, vb);

        // 4. mul_ps: scale the result
        //    (alternatively: _mm256_fmadd_ps(va, v_scale, _mm256_mul_ps(vb, v_scale)))
        __m256 vres = _mm256_mul_ps(vsum, v_scale);

        // 5. storeu: write 8 results back to memory
        _mm256_storeu_ps(&c[i], vres);
    }

    // Scalar remainder for elements that don't fill a full AVX register
    for (; i < n; i++) {
        c[i] = (a[i] + b[i]) * scale;
    }
}
```

**What this demonstrates:**
- `_mm256_set1_ps` outside the loop — compute constants once, reuse every iteration
- `_mm256_loadu_ps` — no alignment constraint needed
- `_mm256_add_ps` → `_mm256_mul_ps` — two arithmetic ops, 8 elements each, back-to-back
- `_mm256_storeu_ps` — write results
- Remainder loop — handle `n % 8` trailing elements cleanly

**FMA version** — if `a` and `b` have already been scaled separately, `fmadd` collapses two ops:
```cpp
// c[i] = a[i] * scale + b[i]
__m256 vres = _mm256_fmadd_ps(va, v_scale, vb);
```

---

### ARM NEON Intrinsics (Jetson, Apple Silicon, Mobile)

ARM NEON is the 128-bit SIMD extension for ARM processors — it's what runs on NVIDIA Jetson (Cortex-A78AE), Apple Silicon (M-series), Qualcomm Snapdragon, AWS Graviton, and every modern phone. If you deploy AI at the edge, NEON is how you vectorize CPU-side preprocessing.

#### NEON vs x86 SIMD

| | x86 SSE | x86 AVX2 | ARM NEON | ARM SVE/SVE2 |
|---|---------|----------|----------|--------------|
| Width | 128-bit | 256-bit | **128-bit** | 128–2048-bit (scalable) |
| FP32 lanes | 4 | 8 | **4** | 4–64 |
| Registers | 16 (XMM) | 16 (YMM) | **32 (V0–V31)** | 32 (Z0–Z31) |
| FMA | SSE: No, FMA3: Yes | Yes | **Yes (vfmaq)** | Yes |
| Header | `<immintrin.h>` | `<immintrin.h>` | **`<arm_neon.h>`** | `<arm_sve.h>` |
| Compiler | gcc/clang `-msse4.2` | `-mavx2` | **`-march=armv8-a`** | `-march=armv8.2-a+sve` |

NEON has **32 registers** (vs 16 on x86 AVX2) — more registers means the compiler can keep more values live without spilling. This partly compensates for the narrower 128-bit width.

#### NEON Data Types

```cpp
#include <arm_neon.h>

// Type naming: <base><bits>x<lanes>_t
float32x4_t    // 4× FP32 in a 128-bit register
float16x8_t    // 8× FP16 in a 128-bit register
int32x4_t      // 4× INT32
int8x16_t      // 16× INT8
uint8x16_t     // 16× UINT8

// "×2" variants — register pairs (used for interleaved load/store)
float32x4x2_t  // pair of float32x4_t (8 floats across 2 registers)
```

#### NEON Intrinsics — Top 15 Most Used

| Intrinsic | Operation | x86 equivalent |
|-----------|-----------|----------------|
| `vld1q_f32(ptr)` | Load 4 floats | `_mm_loadu_ps` |
| `vst1q_f32(ptr, v)` | Store 4 floats | `_mm_storeu_ps` |
| `vaddq_f32(a, b)` | a + b (4-wide) | `_mm_add_ps` |
| `vsubq_f32(a, b)` | a − b | `_mm_sub_ps` |
| `vmulq_f32(a, b)` | a × b | `_mm_mul_ps` |
| `vfmaq_f32(acc, a, b)` | **acc + a×b (FMA)** | `_mm_fmadd_ps` |
| `vmaxq_f32(a, b)` | max(a, b) | `_mm_max_ps` |
| `vminq_f32(a, b)` | min(a, b) | `_mm_min_ps` |
| `vdupq_n_f32(s)` | Broadcast scalar to all 4 lanes | `_mm_set1_ps` |
| `vmovq_n_f32(0)` | Zero vector | `_mm_setzero_ps` |
| `vcvtq_f32_s32(v)` | INT32→FP32 convert | `_mm_cvtepi32_ps` |
| `vcvtq_s32_f32(v)` | FP32→INT32 convert | `_mm_cvtps_epi32` |
| `vaddvq_f32(v)` | **Horizontal sum (all 4 lanes)** | No single instruction! |
| `vmaxvq_f32(v)` | **Horizontal max** | No single instruction! |
| `vld2q_f32(ptr)` | **Interleaved load** (AoS→SoA) | No equivalent |

> **NEON advantage:** `vaddvq_f32` does a horizontal sum in a single instruction. On x86, horizontal reduction requires 3–4 shuffle+add steps. NEON also has `vld2q`/`vld3q`/`vld4q` for automatic AoS→SoA deinterleaving on load — extremely useful for RGB/RGBA image processing.

#### Example 1: Vector Addition (NEON)

```cpp
#include <arm_neon.h>

void add_vectors_neon(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(&c[i], vc);
    }
    // Scalar remainder
    for (; i < n; i++) c[i] = a[i] + b[i];
}
```

```bash
# Compile on Jetson / ARM64
g++ -O2 -march=armv8.2-a+fp16 -o vecadd vecadd.cpp
```

#### Example 2: Dot Product with FMA

```cpp
float dot_neon(const float* a, const float* b, int n) {
    float32x4_t acc = vmovq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vfmaq_f32(acc, va, vb);    // acc += va * vb (fused multiply-add)
    }
    float result = vaddvq_f32(acc);       // horizontal sum — single instruction!
    // Scalar remainder
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}
```

Compare to x86: the `vaddvq_f32` at the end replaces the 4-instruction horizontal sum shuffle dance (`hadd → hadd → extract`).

#### Example 3: ReLU Activation (INT8, 16-wide)

INT8 quantized inference is critical on Jetson — NEON processes 16 INT8 elements per instruction:

```cpp
void relu_int8_neon(const int8_t* input, int8_t* output, int n) {
    int8x16_t zero = vmovq_n_s8(0);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t v = vld1q_s8(&input[i]);
        int8x16_t result = vmaxq_s8(v, zero);   // max(v, 0) = ReLU, 16 elements
        vst1q_s8(&output[i], result);
    }
    for (; i < n; i++) output[i] = input[i] > 0 ? input[i] : 0;
}
```

16 ReLU operations per instruction — this is why quantized INT8 inference on ARM is so fast.

#### Example 4: RGB Deinterleave (AoS→SoA with vld3q)

Image preprocessing on Jetson often starts with RGBRGBRGB... (AoS). NEON's `vld3q` deinterleaves automatically:

```cpp
void rgb_to_planar(const uint8_t* rgb, uint8_t* r, uint8_t* g, uint8_t* b, int npixels) {
    int i = 0;
    for (; i + 16 <= npixels; i += 16) {
        // Load 48 bytes (16 pixels × 3 channels) and deinterleave
        uint8x16x3_t pixel = vld3q_u8(&rgb[i * 3]);
        //   pixel.val[0] = R R R R R R R R R R R R R R R R  (16 red values)
        //   pixel.val[1] = G G G G G G G G G G G G G G G G  (16 green values)
        //   pixel.val[2] = B B B B B B B B B B B B B B B B  (16 blue values)

        vst1q_u8(&r[i], pixel.val[0]);
        vst1q_u8(&g[i], pixel.val[1]);
        vst1q_u8(&b[i], pixel.val[2]);
    }
    // Scalar remainder
    for (; i < npixels; i++) {
        r[i] = rgb[i*3]; g[i] = rgb[i*3+1]; b[i] = rgb[i*3+2];
    }
}
```

On x86, this deinterleave requires multiple shuffle/permute/blend instructions. NEON does it in one load.

#### Example 5: FP16 Vector Addition (Jetson + Apple Silicon)

ARMv8.2-a added native FP16 arithmetic — critical for AI inference:

```cpp
#include <arm_neon.h>

void add_fp16(const __fp16* a, const __fp16* b, __fp16* c, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t va = vld1q_f16(&a[i]);     // Load 8× FP16
        float16x8_t vb = vld1q_f16(&b[i]);
        float16x8_t vc = vaddq_f16(va, vb);    // 8-wide FP16 add
        vst1q_f16(&c[i], vc);
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}
```

```bash
# Must enable FP16 arithmetic
g++ -O2 -march=armv8.2-a+fp16 -o fp16add fp16add.cpp
```

8 FP16 values per 128-bit register — same throughput per register as 4× FP32, but half the memory traffic.

#### Example 6: Tiled 4×4 Matrix Multiply (NEON)

Small matrix multiply is common in on-device transformer attention:

```cpp
void matmul_4x4_neon(const float* A, const float* B, float* C) {
    // Load all 4 columns of B
    float32x4_t b0 = vld1q_f32(&B[0]);
    float32x4_t b1 = vld1q_f32(&B[4]);
    float32x4_t b2 = vld1q_f32(&B[8]);
    float32x4_t b3 = vld1q_f32(&B[12]);

    for (int i = 0; i < 4; i++) {
        float32x4_t a_row = vld1q_f32(&A[i * 4]);

        // C[i] = A[i,0]*B[0] + A[i,1]*B[1] + A[i,2]*B[2] + A[i,3]*B[3]
        float32x4_t c = vmulq_laneq_f32(b0, a_row, 0);      // broadcast A[i,0] × B col 0
        c = vfmaq_laneq_f32(c, b1, a_row, 1);                // += A[i,1] × B col 1
        c = vfmaq_laneq_f32(c, b2, a_row, 2);                // += A[i,2] × B col 2
        c = vfmaq_laneq_f32(c, b3, a_row, 3);                // += A[i,3] × B col 3

        vst1q_f32(&C[i * 4], c);
    }
}
```

`vmulq_laneq_f32(vec, src, lane)` broadcasts one lane of `src` and multiplies with all 4 lanes of `vec`. `vfmaq_laneq_f32` does the same with fused multiply-add. This pattern is the building block of larger tiled GEMM on ARM.

#### NEON vs AVX2 Summary

| | AVX2 (x86) | NEON (ARM) |
|---|------------|------------|
| Register width | 256-bit | 128-bit |
| FP32 lanes | 8 | 4 |
| Registers | 16 | **32** |
| Horizontal sum | 3–4 instructions | **1 instruction** (`vaddvq`) |
| AoS deinterleave | Complex shuffles | **1 instruction** (`vld3q/vld4q`) |
| FP16 native | No (convert to FP32) | **Yes** (ARMv8.2-a) |
| Cross-lane gotchas | 128-bit lane boundary | None (true 128-bit) |
| Typical platform | Server, workstation | **Jetson, phone, Apple Silicon, Graviton** |

**When to use NEON vs let the compiler vectorize:** Same rule as x86 — try auto-vectorization first (`-O2 -march=armv8.2-a`). Use manual NEON intrinsics when the compiler fails (complex access patterns, interleaved data, FP16 arithmetic) or when you need guaranteed performance (real-time inference on Jetson).

#### ARM SVE/SVE2 — The Future

SVE (Scalable Vector Extension) is ARM's answer to the fixed-width problem. Instead of 128 bits, SVE registers are **variable width** (128–2048 bits, set per chip). Your code works on any width without recompiling:

```cpp
#include <arm_sve.h>

void add_sve(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i += svcntw()) {           // svcntw() = FP32 lanes at runtime
        svbool_t pred = svwhilelt_b32(i, n);           // predicate mask for bounds
        svfloat32_t va = svld1(pred, &a[i]);
        svfloat32_t vb = svld1(pred, &b[i]);
        svst1(pred, &c[i], svadd_f32_m(pred, va, vb));
    }
}
// Same binary runs on 128-bit (Neoverse N1) and 256-bit (Neoverse V2) and future 512-bit
```

SVE2 is available on Neoverse V2 (AWS Graviton4), Apple M4, and upcoming Jetson. NEON knowledge transfers directly — SVE is NEON with scalable width and predication.

---

### Cross-Lane Limitations (The AVX2 Gotcha)

Most AVX2 "256-bit" instructions actually execute as **two independent 128-bit halves**. This is the single most surprising architectural quirk in AVX2 and causes subtle bugs when you expect full cross-lane operations.

```cpp
// shuffle_ps looks like it works on 256-bit, but it only shuffles within each 128-bit half:
__m256 v = {7, 6, 5, 4, 3, 2, 1, 0};  // lanes 0-7
__m256 s = _mm256_shuffle_ps(v, v, 0b00011011);
// Result: {4,5,6,7, 0,1,2,3} — reversed within each half, NOT across the full register

// To truly cross lanes, you need explicit cross-lane moves:
// Swap the two 128-bit halves:
__m256 swapped = _mm256_permute2f128_ps(v, v, 0x01);

// Full cross-lane element reorder (AVX2) — the most flexible option:
__m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);  // reverse order
__m256 reversed = _mm256_permutevar8x32_ps(v, idx);       // true 8-element permute
```

| Operation | Cross-lane? | Notes |
|-----------|-------------|-------|
| `_mm256_shuffle_ps` | No | Within each 128-bit half only |
| `_mm256_hadd_ps` | No | Adjacent pairs within each half |
| `_mm256_permute2f128_ps` | Yes | Swaps/copies 128-bit halves |
| `_mm256_permutevar8x32_ps` | Yes | Full 8-element permute (AVX2 only) |

> **Rule of thumb:** If an intrinsic says "256-bit" but was available in AVX (not AVX2), suspect it operates on two 128-bit halves. Verify with [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) or [uops.info](https://uops.info/).

---

### Inspecting SIMD Registers (Debugging)

You can't `printf` a `__m256`. The standard pattern is a union, which lets you read individual lanes:

```cpp
// Union for safe lane inspection (from hands-on-simd-programming)
union float8 {
    __m256  v;
    float   a[8];
    float8() : v(_mm256_setzero_ps()) {}
    float8(__m256 x) : v(x) {}
};

// Usage
float8 result(_mm256_fmadd_ps(va, vb, vc));
printf("lane 0: %f, lane 3: %f\n", result.a[0], result.a[3]);

// Or with structured binding in C++17:
auto [l0,l1,l2,l3,l4,l5,l6,l7] = result.a;
```

**Horizontal sum — reducing a vector to a scalar:**

This pattern appears everywhere: dot products, attention scores, reductions.

```cpp
// Sum all 8 floats in a __m256 to a scalar
float hsum(__m256 v) {
    // Step 1: add pairs within each 128-bit half → [a+e, b+f, c+g, d+h | a+e, b+f, c+g, d+h]
    __m128 lo  = _mm256_castps256_ps128(v);          // lower 128 bits
    __m128 hi  = _mm256_extractf128_ps(v, 1);        // upper 128 bits
    __m128 sum = _mm_add_ps(lo, hi);                 // add the two halves
    // Step 2: horizontal adds until scalar
    sum = _mm_hadd_ps(sum, sum);                     // [a+b+e+f, c+d+g+h, ...]
    sum = _mm_hadd_ps(sum, sum);                     // [total, total, ...]
    return _mm_cvtss_f32(sum);                       // extract lane 0
}
```

**Stream stores for large write-once buffers:**

```cpp
// Non-temporal store: bypasses CPU cache entirely
// Use when writing large output buffers you won't read back soon
// (e.g., preprocessing output, frame buffer writes)
for (int i = 0; i < n; i += 8) {
    __m256 result = /* compute */;
    _mm256_stream_ps(&out[i], result);   // skip cache, write direct to memory
}
_mm_sfence();  // memory fence — ensure all stream writes are visible

// When to use: write-once buffers > L3 cache size
// When NOT: if you'll read the data back soon (defeats the purpose)
```

**Gather — loading from non-contiguous addresses:**

```cpp
// Gather: load from base_ptr + indices[i] * scale
float table[1024] = { /* lookup table */ };
__m256i indices = _mm256_set_epi32(7, 3, 15, 0, 42, 8, 1, 100);  // arbitrary indices
__m256 gathered = _mm256_i32gather_ps(table, indices, 4);          // scale=4 (sizeof float)

// Note: gather throughput ≈ scalar loads — no bandwidth benefit for the gather itself.
// The value: the *computation* after gather runs vectorized (8 FMAs instead of 8 scalar ops).
// Good for: embedding lookups, sparse vector ops, non-sequential access patterns.
```

---

### Cache Lines and Memory Alignment

The CPU fetches memory in **64-byte cache lines** — not individual bytes. This is the physical reason SIMD alignment matters and the root cause of most false-sharing bugs in multithreaded code.

```
Cache line = 64 bytes = 16 floats = 8 doubles = 2 AVX registers

[float 0][float 1]...[float 15]   ← one cache line fetch
         └─ AVX load ──┘└─ AVX load ──┘  ← if misaligned, spans two lines
```

**Why it matters:**

| Scenario | Cache line impact |
|----------|------------------|
| `_mm256_load_ps` aligned | One cache line touched per load |
| `_mm256_loadu_ps` misaligned | May touch two cache lines — up to 2× memory traffic |
| False sharing (threads) | Two threads write different variables in same 64-byte line → cache ping-pong |
| `alignas(64)` struct | Guarantees struct starts at cache line boundary |

```cpp
// False sharing — avoid this
struct Workers {
    int counter_a;  // same 64-byte line as counter_b
    int counter_b;
};

// Fix: pad to cache line boundary
struct alignas(64) WorkerA { int counter; };
struct alignas(64) WorkerB { int counter; };
```

**Rule:** Align SIMD buffers to 32 bytes (`alignas(32)`) for AVX. Align per-thread data to 64 bytes (`alignas(64)`) to prevent false sharing.

---

### Data Layout: AoS vs SoA

How you arrange data in memory is often **the single highest-leverage optimization** in SIMD and GPU code — more impactful than instruction selection.

#### Array of Structs (AoS) — natural, object-oriented, SIMD-unfriendly

Each particle is one contiguous object. Easy to pass around, easy to reason about.

```cpp
struct Particle {
    float x, y, z;    // position
    float vx, vy, vz; // velocity
};

Particle particles[4];
```

Memory layout — fields interleaved across all particles:

```
particle 0          particle 1          particle 2          particle 3
[x][y][z][vx][vy][vz][x][y][z][vx][vy][vz][x][y][z][vx][vy][vz][x][y][z][vx][vy][vz]
```

When you want to add `dx` to all x-positions:

```cpp
// AoS: stride between consecutive x values = sizeof(Particle) = 24 bytes
for (int i = 0; i < N; i++)
    particles[i].x += dx;
// An AVX load at &particles[0].x picks up: x0,y0,z0,vx0,vy0,vz0,x1,y1
//                                                           ^^^ garbage fields in the register
// You wanted: x0,x1,x2,x3,x4,x5,x6,x7 — but they're 24 bytes apart, not contiguous
```

The x-values are **scattered** in memory. SIMD cannot load them efficiently — it would need gather (`_mm256_i32gather_ps`), which has the same throughput as scalar loads.

#### Struct of Arrays (SoA) — SIMD-friendly

Each field gets its own contiguous array. All x values together, all y values together.

```cpp
struct Particles {
    float x[N], y[N], z[N];
    float vx[N], vy[N], vz[N];
};

Particles p;
p.x[0]=1; p.x[1]=4; p.x[2]=7;   // all x values contiguous
p.y[0]=2; p.y[1]=5; p.y[2]=8;
```

Memory layout — each field is a flat contiguous array:

```
x:  [x0][x1][x2][x3][x4][x5][x6][x7]...  ← one AVX load = 8 x-values
y:  [y0][y1][y2][y3][y4][y5][y6][y7]...
z:  [z0][z1][z2][z3]...
vx: [vx0][vx1][vx2]...
```

Now the SIMD loop is clean — one load, one add, one store:

```cpp
// SoA: x values are contiguous → direct sequential AVX load
__m256 vdx = _mm256_set1_ps(dx);
for (int i = 0; i < N; i += 8) {
    __m256 vx = _mm256_loadu_ps(&p.x[i]);       // load x0..x7 — perfectly contiguous
    _mm256_storeu_ps(&p.x[i], _mm256_add_ps(vx, vdx));  // add dx to all 8 at once
}
```

#### Side-by-side loop comparison

```cpp
// AoS — compiler cannot auto-vectorize the x update (stride too large)
for (int i = 0; i < N; i++)
    particles[i].x += dx;           // stride = 24 bytes between x values

// SoA — compiler auto-vectorizes, or you write AVX explicitly
for (int i = 0; i < N; i++)
    p.x[i] += dx;                   // stride = 4 bytes — perfectly sequential
```

#### Layout comparison

| | AoS | SoA | AoSoA |
|--|-----|-----|-------|
| Memory pattern | `xyzxyzxyz...` | `xxx...yyy...zzz...` | `[xxxx yyyy][xxxx yyyy]...` |
| SIMD efficiency | Low — gather/scatter needed | High — sequential load | High + cache-friendly |
| GPU coalescing | Poor | Good | Good |
| Code readability | High — `p[i].x` | Medium — `p.x[i]` | Low — tiled indexing |
| Best for | Small structs, OOP | Physics, ML kernels, SIMD | Intel oneDNN, CUTLASS tiling |

**When to choose each:**
- **AoS** — small datasets, all fields used together, readability matters more than throughput
- **SoA** — large arrays, per-field operations, SIMD loops, GPU kernels
- **AoSoA** — when you need SoA efficiency but also want cache-line-sized tiles for L1 reuse (oneDNN weight format, CUTLASS fragment layouts)

> **Deep learning connection:** cuDNN NCHW vs NHWC is exactly this choice. NCHW = SoA over spatial dimensions (all red pixels, then all green). NHWC = AoS over channels (all channels for one pixel together). Tensor Core layouts (e.g., `mma.sync` fragments) use AoSoA — 16×16 tiles packed for register-level reuse. Choosing the wrong layout forces expensive transposes that dominate runtime.

---

### `std::simd` (C++26, Experimental)

The future of portable SIMD — no intrinsics, no platform-specific code:

```cpp
#include <experimental/simd>
namespace stdx = std::experimental;

void add_vectors(float* a, float* b, float* c, int n) {
    using V = stdx::native_simd<float>;  // Auto-selects best width
    for (int i = 0; i < n; i += V::size()) {
        V va(&a[i], stdx::element_aligned);
        V vb(&b[i], stdx::element_aligned);
        V vc = va + vb;
        vc.copy_to(&c[i], stdx::element_aligned);
    }
}
```

Available in GCC 11+ with `-std=c++23 -lstdc++exp`. Not yet production-ready but worth tracking.

### Measurement First

> *"Measure, don't guess. The bottleneck is never where you think it is."*

Before writing a single intrinsic, profile to confirm what kind of bottleneck you actually have. Optimizing the wrong thing wastes time.

```bash
# perf stat — hardware counter overview
perf stat -e instructions,cycles,cache-misses,fp_arith_inst_retired.256b_packed_single \
    ./my_program

# Key ratios to check:
#   instructions/cycles < 2.0  → probably stalled on memory
#   cache-misses high           → data layout problem (try SoA)
#   256b_packed_single low      → auto-vectorization failed, add -fopt-info-vec

# Confirm vectorization happened
g++ -O2 -march=native -fopt-info-vec-optimized my_code.cpp
# "loop vectorized using 32 byte vectors" = success

# VTune (Intel) — visual hotspots and vectorization analysis
vtune -collect hotspots ./my_program
```

### Roofline Model

The **Roofline model** tells you whether your kernel is compute-bound or memory-bound — and therefore whether SIMD helps.

```
Peak FLOPS  = freq × cores × SIMD_width × FMA_factor
Peak BW     = DRAM bandwidth (GB/s)
AI          = FLOPs / bytes_accessed  (arithmetic intensity)

If  AI > Peak FLOPS / Peak BW  → compute-bound  (SIMD / FMA helps)
If  AI < Peak FLOPS / Peak BW  → memory-bound   (better layout / caching helps)
```

**Example for a modern desktop CPU:**

```
Peak AVX2 FP32 = 3.6 GHz × 8 lanes × 2 (FMA) = ~58 GFLOPS/core
Peak DRAM BW   = ~50 GB/s
Ridge point AI = 58 / 50 ≈ 1.2 FLOP/byte

Vector add:  AI = 4 bytes out / 8 bytes in = 0.5 FLOP/byte → memory-bound
Dot product: AI = 2N FLOP / 2N×4 bytes    = 0.25 FLOP/byte → memory-bound
Matrix mult: AI = O(N³) FLOP / O(N²) bytes → compute-bound for large N ✓
```

**Implication:** SIMD gives the full 8× speedup only on compute-bound kernels. For memory-bound kernels (like vector add), the bottleneck is DRAM bandwidth — SIMD won't help much beyond reducing loop overhead.

### SIMD → GPU: Mental Model Bridge

The jump from SIMD to CUDA looks large but the underlying idea is the same. The key difference is *who* picks the element index:

```cpp
// ── SIMD mindset: one thread, one instruction covers N elements ──
for (int i = 0; i < n; i += 8) {
    __m256 va = _mm256_load_ps(&a[i]);
    __m256 vb = _mm256_load_ps(&b[i]);
    _mm256_store_ps(&c[i], _mm256_add_ps(va, vb));
}
// You write the stride (i += 8). The hardware runs 8 lanes silently.

// ── GPU / SIMT mindset: one thread per element, hardware spawns thousands ──
__global__ void add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}
// You write one element. The hardware runs 32 (warp) or thousands in parallel.
```

| Aspect | AVX (CPU SIMD) | CUDA warp (GPU SIMT) |
|--------|---------------|----------------------|
| Parallel width | 8× float32 | 32 threads |
| Register file | `__m256` (256-bit) | 32× 32-bit registers per thread |
| Memory | L1/L2 cache | Shared memory / global memory |
| Divergence cost | Branch kills vectorization | Warp divergence serializes threads |
| Programmer model | Explicit lanes (intrinsics) | Implicit (thread index) |

---

### Connection to the Stack

- **L1 (Application):** cuDNN and CUTLASS use vectorized memory loads internally
- **L2 (Compiler):** MLIR's `vector` dialect (Phase 4C) targets exactly this abstraction level
- **L5 (Architecture):** When you design an accelerator's vector unit, you're designing custom SIMD hardware
- **Sub-Track 3 (CUDA):** GPU "SIMT" is SIMD scaled to thousands of threads — same mental model

---

## Performance Traps and Common Mistakes

These are the issues that consume weeks of debugging in real HPC and ML systems. Learn them here.

| Trap | What happens | Fix |
|------|-------------|-----|
| **`shared_ptr` in hot loops** | Atomic ref-count inc/dec every iteration → cache line contention | Use raw pointer or `unique_ptr` inside hot paths; bump `shared_ptr` count once outside the loop |
| **False sharing** | Two threads update different variables in the same 64-byte cache line → invisible serialization | Pad thread-local data to `alignas(64)` |
| **Unaligned `_mm256_load_ps`** | Segfault (`#GP`) at runtime, not compile time | Use `_mm256_loadu_ps` unless you've verified 32-byte alignment with `alignas(32)` |
| **Branching inside vectorized loops** | Compiler cannot vectorize; scalar fallback runs | Move conditionals outside the loop; use SIMD blend instructions (`_mm256_blendv_ps`) |
| **Memory-bound SIMD** | SIMD adds complexity, negligible speedup | Profile first (Roofline); restructure data layout (SoA) before adding intrinsics |
| **`par_unseq` with dependencies** | Silent data corruption or undefined behavior | Only use when elements are **completely independent** — no reads from neighboring indices |
| **Ignoring NUMA** | Multi-socket systems: thread allocates memory on socket 0, socket 1 thread reads it → 2× bandwidth | Allocate memory on the NUMA node that will use it (`numactl`, `mbind`) |

---

## Projects

1. **Modern C++ warmup** — Write a matrix class using `std::vector`, move semantics, templates, and operator overloading. Verify that returning a large matrix from a function uses move (not copy) by checking with `-fno-elide-constructors`.
2. **Lambda benchmark** — Implement the same computation three ways: raw loop, `std::for_each` with lambda, and `std::transform` with lambda. Benchmark all three with `-O2`. Verify the compiler generates identical assembly (`-S`).
3. **Parallel STL sort** — Sort 10M floats with `std::execution::seq`, `par`, and `par_unseq`. Measure speedup. Plot time vs array size.
4. **Auto-vectorization** — Write a dot product loop. Compile with `-O2 -march=native -fopt-info-vec`. Check if the compiler vectorized it. If not, restructure the loop until it does.
5. **AVX intrinsics dot product** — Implement dot product using `_mm256_fmadd_ps`. Benchmark against scalar and auto-vectorized versions. Measure GFLOPS.
6. **Aligned vs unaligned** — Benchmark `_mm256_load_ps` (aligned) vs `_mm256_loadu_ps` (unaligned) for a large array. Measure the performance difference.
7. **AoS → SoA benchmark** — Implement vector dot product for an array of `Vec3` structs (AoS) and compare to the SoA layout version. Observe the speedup gap (expect 10–40× from layout alone).
8. **Horizontal sum** — Implement a dot product that uses `_mm256_fmadd_ps` to accumulate 8 at a time, then uses the horizontal sum idiom to reduce to a scalar. Verify against scalar result.
9. **Tiny MHA block** — Implement a batched multi-head attention forward pass for seq=8, head_dim=16, heads=4 using raw AVX2 intrinsics. Pre-transpose weight matrices. Benchmark against scalar version. Observe that the speedup is ~2–3×, not 8× — explain why (memory-bound).
10. **NEON dot product** — Implement dot product on ARM using `vfmaq_f32` and `vaddvq_f32`. Benchmark on Jetson or Apple Silicon. Compare GFLOPS with the AVX2 version.
11. **NEON RGB deinterleave** — Use `vld3q_u8` to convert RGBRGB... to planar R,G,B arrays. Benchmark against scalar loop. Measure on Jetson with a real camera frame (1920×1080).
12. **NEON INT8 ReLU** — Implement `max(x, 0)` on 16-wide INT8 vectors. Process 1M elements. Measure throughput (GB/s) and compare to theoretical LPDDR5 bandwidth on Jetson.
13. **NEON FP16 GEMM** — Implement a tiled 64×64 FP16 matrix multiply using `vfmaq_f16`. Compare throughput with FP32 version. Verify output correctness by upconverting to FP32.
14. **NEON vs auto-vec** — Write the same 5 kernels (add, dot, relu, scale, reduce) both as manual NEON and as plain loops compiled with `-O2 -march=armv8.2-a`. Compare assembly output on Godbolt. Note where the compiler matches your intrinsics and where it doesn't.

---

## Resources

### Learning

| Resource | What it covers |
|----------|---------------|
| *A Tour of C++* (Stroustrup) | Modern C++ overview, concise |
| [cppreference.com](https://en.cppreference.com/) | Definitive C++ reference |
| *C++ Concurrency in Action* (Williams) | Threads, atomics, memory model |
| [SIMD for C++ Developers (const.me)](http://const.me/articles/simd/simd.pdf) | Best practitioner tutorial: loads, arithmetic, shuffles, masking, cross-lane gotchas. 23 pages. Read this first. |
| [hands-on-simd-programming](https://github.com/yuninxia/hands-on-simd-programming) | Progressive labs from basic intrinsics → image processing → full MHA block → quantized GPT decoder. Run `./runme.sh` to see benchmarks. |
| [awesome-simd](https://github.com/awesome-simd/awesome-simd) | Curated index of real-world SIMD libraries, tools, blogs, and references |
| [Agner Fog Optimization Guides](https://www.agner.org/optimize/) | Definitive reference on CPU micro-architecture, instruction latencies, calling conventions |

### ARM / NEON

| Resource | What it covers |
|----------|---------------|
| [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/) | Official searchable intrinsics guide (like Intel's, but for ARM) |
| [ARM Neon Programmer's Guide](https://developer.arm.com/documentation/den0018/latest) | Official tutorial: data types, operations, optimization |
| [ncnn NEON optimization](https://github.com/Tencent/ncnn/wiki/how-ncnn-optimizes) | Real-world NEON optimization for neural network inference |
| [Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries) | Optimized BLAS/LAPACK/FFT for ARM (like MKL for x86) |
| [SIMDe](https://github.com/simd-everywhere/simde) | Emulates x86 intrinsics on ARM — run AVX code on Jetson |

### Reference and Tooling

| Resource | What it covers |
|----------|---------------|
| [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) | Searchable reference for every x86 SIMD intrinsic |
| [uops.info](https://uops.info/) | Instruction latency, throughput, and port usage for every CPU micro-architecture |
| [Compiler Explorer (godbolt.org)](https://godbolt.org/) | See assembly output for any C++ code — verify vectorization happened |
| [SIMD-Visualiser](https://github.com/piotte13/SIMD-Visualiser) | Browser-based visual diagram of what each intrinsic does to register contents |
| [Felix Cloutier x86 reference](https://www.felixcloutier.com/x86/) | HTML x86/x64 instruction reference |

### Portable SIMD Libraries (skip raw intrinsics)

| Library | Language | What it is |
|---------|----------|------------|
| [Google Highway](https://github.com/google/highway) | C++ | Best portable SIMD: length-agnostic, runtime dispatch, SSE/AVX/AVX-512/NEON/SVE |
| [xsimd](https://github.com/QuantStack/xsimd) | C++ | Header-only wrappers for SSE/AVX/AVX-512/NEON |
| [SIMDe](https://github.com/simd-everywhere/simde) | C/C++ | Emulates x86 intrinsics on ARM and other targets |
| [Intel ISPC](https://ispc.github.io/) | ISPC | C-like language that compiles to optimal SIMD for SSE/AVX/AVX-512/NEON/PS5/Xbox |

### Real-World SIMD Libraries to Study

| Library | What it does |
|---------|-------------|
| [simdjson](https://github.com/lemire/simdjson) | JSON parsing at >2 GB/s using SIMD — canonical example of real-world vectorized parsing |
| [SimSIMD](https://github.com/ashvardanian/SimSIMD) | SIMD-accelerated similarity measures (cosine, L2) for embedding vectors |
| [ncnn](https://github.com/Tencent/ncnn) | On-device neural network inference with hand-tuned SIMD for ARM and x86 |
| [StringZilla](https://github.com/ashvardanian/StringZilla) | SIMD substring search, fuzzy matching, sorting |

### Blogs (practitioners writing production SIMD code)

| Blog | Author |
|------|--------|
| [lemire.me/blog](https://lemire.me/blog/) | Daniel Lemire — simdjson, SIMDComp, practical vectorization |
| [branchfree.org](https://branchfree.org/) | Geoff Langdale — branchless and SIMD techniques |
| [0x80.pl/notesen](http://0x80.pl/notesen.html) | Wojciech Muła — low-level SIMD algorithms |

---

## Next

→ [**Sub-Track 2 — OpenMP and oneTBB**](../OpenMP%20and%20OneTBB/Guide.md) — scale from SIMD to multi-core CPU parallelism.
