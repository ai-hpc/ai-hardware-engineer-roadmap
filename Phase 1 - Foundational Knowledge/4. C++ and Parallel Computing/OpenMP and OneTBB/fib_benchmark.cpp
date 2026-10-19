// Educational task-granularity benchmark, not an efficient Fibonacci algorithm.
// g++ -O2 -std=c++17 -fopenmp fib_benchmark.cpp -ltbb -o fib_benchmark
// ./fib_benchmark [n=30] [threads=2] [cutoff=20] [repetitions=3]
// ./fib_benchmark --self-test
// GCC uses libgomp for -fopenmp; Ubuntu dependencies: g++ and libtbb-dev.
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <omp.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_invoke.h>

int task_cutoff = 20;  // Set before parallel work; unchanged during each run.
long long fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}
long long fib_reference(int n) {
    long long previous = 0, current = 1;
    for (int i = 0; i < n; ++i) {
        const long long next = previous + current;
        previous = current;
        current = next;
    }
    return previous;
}
long long fib_omp(int n) {
    if (n < task_cutoff) return fib_serial(n);
    long long x, y;
    #pragma omp task shared(x) firstprivate(n)
    x = fib_omp(n - 1);
    #pragma omp task shared(y) firstprivate(n)
    y = fib_omp(n - 2);
    #pragma omp taskwait
    return x + y;
}
long long fib_tbb(int n) {
    if (n < task_cutoff) return fib_serial(n);
    long long x, y;
    oneapi::tbb::parallel_invoke(
        [&] { x = fib_tbb(n - 1); }, [&] { y = fib_tbb(n - 2); });
    return x + y;
}
long long run_openmp(int n, int& observed_team) {
    long long result = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            observed_team = omp_get_num_threads();
            result = fib_omp(n);
        }
    }
    return result;
}
int number(const char* argument, int minimum, int maximum) {
    std::string_view text(argument);
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value < minimum || value > maximum)
        throw std::invalid_argument("integer argument outside documented range");
    return value;
}
template<class Function>
double median_ms(Function operation, long long expected, int repetitions) {
    // Warm every implementation; keep each result observable and checked.
    if (operation() != expected) throw std::runtime_error("warmup result mismatch");
    std::vector<double> times;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        const long long result = operation();
        const auto end = std::chrono::steady_clock::now();
        if (result != expected) throw std::runtime_error("measured result mismatch");
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    const auto middle = times.size() / 2;
    return times.size() % 2 ? times[middle] : (times[middle - 1] + times[middle]) / 2;
}
int main(int argc, char** argv) {
    try {
        const bool self_test = argc == 2 && std::string_view(argv[1]) == "--self-test";
        if (!self_test && argc > 5) throw std::invalid_argument("too many arguments");
        const int n = !self_test && argc > 1 ? number(argv[1], 0, 42) : 30;
        const int threads = !self_test && argc > 2 ? number(argv[2], 1, 64) : 2;
        task_cutoff = !self_test && argc > 3 ? number(argv[3], 2, 42) : 20;
        const int repeats = !self_test && argc > 4 ? number(argv[4], 1, 20) : 3;
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, threads);
        int observed_team = 0;
        if (self_test) {
            for (int cutoff : {2, 10, 20}) {
                task_cutoff = cutoff;
                for (int test_n : {0, 1, 2, 9, 10, 11, 19, 20, 21}) {
                    const auto expected = fib_reference(test_n);
                    if (fib_serial(test_n) != expected || run_openmp(test_n, observed_team) != expected ||
                        fib_tbb(test_n) != expected) throw std::runtime_error("self-test mismatch");
                }
            }
            std::puts("PASS 27 input/cutoff cases checked against iterative reference for all 3 algorithms");
            return 0;
        }
        const auto expected = fib_reference(n);
        const double serial = median_ms([&] { return fib_serial(n); }, expected, repeats);
        const double openmp = median_ms([&] { return run_openmp(n, observed_team); }, expected, repeats);
        const double tbb = median_ms([&] { return fib_tbb(n); }, expected, repeats);
        std::printf("fib(%d)=%lld parallelism_budget=%d observed_openmp_team=%d cutoff=%d repetitions=%d\n",
                    n, expected, threads, observed_team, task_cutoff, repeats);
        std::printf("median_ms serial=%.6f openmp=%.6f onetbb=%.6f\n", serial, openmp, tbb);
        if (openmp > 0 && tbb > 0)
            std::printf("speedup openmp=%.3f onetbb=%.3f\n", serial / openmp, serial / tbb);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\nUsage: fib_benchmark [n:0..42] [threads:1..64] [cutoff:2..42] [repetitions:1..20]\n", error.what());
        return 1;
    }
}
