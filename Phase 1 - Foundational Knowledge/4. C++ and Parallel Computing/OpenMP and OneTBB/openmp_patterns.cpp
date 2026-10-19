// Executed teaching contracts for Guide.md sections 1.0-1.15.
// g++ -O2 -std=c++17 -fopenmp -Wall -Wextra -Werror openmp_patterns.cpp -o /tmp/openmp_patterns
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <omp.h>

struct Vec3 { float x, y, z; };
#pragma omp declare reduction(vec_add : Vec3 : \
    (omp_out.x += omp_in.x, omp_out.y += omp_in.y, omp_out.z += omp_in.z)) \
    initializer(omp_priv = Vec3{0, 0, 0})

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
long long serial_fib(int n) {
    if (n < 2) return n;
    return serial_fib(n - 1) + serial_fib(n - 2);
}
unsigned long long expected_tasks(int n, int cutoff) {
    if (n < cutoff) return 0;
    return 2 + expected_tasks(n - 1, cutoff) + expected_tasks(n - 2, cutoff);
}
long long task_fib(int n, int cutoff, std::atomic<unsigned long long>& created) {
    if (n < cutoff) return serial_fib(n);
    long long x = 0, y = 0;
    created.fetch_add(2, std::memory_order_relaxed);
    #pragma omp task shared(x, created) firstprivate(n, cutoff)
    x = task_fib(n - 1, cutoff, created);
    #pragma omp task shared(y, created) firstprivate(n, cutoff)
    y = task_fib(n - 2, cutoff, created);
    #pragma omp taskwait
    return x + y;
}

void check_patterns(int requested) {
    omp_set_num_threads(requested);
    constexpr int n = 32;
    int team = 0, seed = 10;
    std::vector<int> out(n), a(n), b(n);
    #pragma omp parallel default(none) shared(out, team) firstprivate(seed)
    {
        #pragma omp single
        team = omp_get_num_threads();
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < n; ++i) out[i] = seed++;
    }
    require(seed == 10, "firstprivate changed original");
    for (int i = 0; i < n; ++i)
        require(out[i] == 10 + i / team, "firstprivate was incorrectly reset per iteration");

    int last = -1;
    #pragma omp parallel for schedule(dynamic, 1) lastprivate(last)
    for (int i = n - 1; i >= 0; --i) { last = 2 * i; out[i] = last; }
    require(last == 0, "lastprivate used maximum index or completion order");

    Vec3 total{5, 6, 7};  // Original values participate once in the reduction.
    #pragma omp parallel for reduction(vec_add:total)
    for (int i = 0; i < n; ++i) {
        total.x += 1; total.y += i; total.z += 2 * i;
    }
    require(total.x == 5 + n && total.y == 6 + n * (n - 1) / 2
            && total.z == 7 + n * (n - 1), "custom reduction mismatch");

    int consumers = 0;
    bool ready = true;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n; ++i) a[i] = 2 * i;
        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n; ++i) b[i] = 3 * i;
        #pragma omp barrier
        #pragma omp single
        {
            ++consumers;
            for (int i = 0; i < n; ++i) ready = ready && a[i] + b[i] == 5 * i;
        }
    }
    require(ready && consumers == 1, "nowait consumer missing readiness or single ownership");

    long long sum = 0;
    #pragma omp parallel for simd reduction(+:sum)
    for (int i = 0; i < n; ++i) sum += a[i];
    require(sum == n * (n - 1), "SIMD reduction mismatch");

    int cells[3][5] = {};
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 5; ++col) cells[row][col] = row * 5 + col;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 5; ++col)
            require(cells[row][col] == row * 5 + col, "collapse omitted a cell");

    for (int cutoff : {2, 20}) {
        for (int input : {0, 1, 2, 6, 12, 30}) {
            // Avoid generating millions of tiny tasks for n=30,cutoff=2.
            if (input == 30 && cutoff == 2) continue;
            std::atomic<unsigned long long> created{0};
            long long result = 0;
            #pragma omp parallel
            {
                #pragma omp single
                result = task_fib(input, cutoff, created);
            }
            require(result == serial_fib(input), "task result mismatch");
            require(created.load() == expected_tasks(input, cutoff), "task count mismatch");
            if (input == 30) require(created.load() == 464, "worked task count mismatch");
        }
    }
    std::printf("PASS requested=%d observed_team=%d: private lifetime, lastprivate, reduction, barrier/single, SIMD, collapse, task result/count\n", requested, team);
}

void recurrence_counterexample() {
    int x = 0, team = 0;
    #pragma omp parallel for num_threads(2) schedule(static, 2) firstprivate(x) lastprivate(x)
    for (int i = 0; i < 4; ++i) {
        if (i == 0) team = omp_get_num_threads();  // one writer, read after the join
        x += 1;
    }
    require(x == (team == 1 ? 4 : 2), "unexpected private recurrence result");
    std::printf("PASS recurrence counterexample: team=%d parallel=%d serial=4%s\n", team, x,
                team == 1 ? " (runtime serialized; two-thread counterexample not exercised)" : "");
}
int main() {
    try {
        omp_set_dynamic(0);
        for (int threads : {1, 2, 4}) check_patterns(threads);
        recurrence_counterexample();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
