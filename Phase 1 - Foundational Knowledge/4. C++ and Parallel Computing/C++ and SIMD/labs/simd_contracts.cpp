// Compile the dispatcher for the baseline CPU. Only tagged functions use AVX2/FMA.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define COURSE_X86 1
#define VECTOR_TARGET __attribute__((target("avx2,fma")))
#else
#define COURSE_X86 0
#endif

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void scaled_add_scalar(const float* a, const float* b, float* c,
                       std::size_t n, float scale) {
    for (std::size_t i = 0; i < n; ++i) c[i] = (a[i] + b[i]) * scale;
}
bool has_vector_support() {
#if COURSE_X86
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
#if COURSE_X86
VECTOR_TARGET void scaled_add_vector(const float* a, const float* b, float* c,
                                     std::size_t n, float scale) {
    const __m256 factor = _mm256_set1_ps(scale);
    std::size_t i = 0;
    for (; n - i >= 8; i += 8) {
        const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(a + i),
                                       _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(c + i, _mm256_mul_ps(sum, factor));
    }
#ifndef COURSE_MUTANT_MISSING_TAIL
    for (; i < n; ++i) c[i] = (a[i] + b[i]) * scale;
#endif
}
VECTOR_TARGET void check_lanes() {
    const __m256 v = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
    std::array<float, 8> got{};
    _mm256_storeu_ps(got.data(), _mm256_shuffle_ps(v, v, 0x1b));
    require(got == std::array<float,8>{4,3,2,1,8,7,6,5}, "shuffle half boundary");
    const __m256i indices = _mm256_setr_epi32(7,6,5,4,3,2,1,0);
#ifdef COURSE_MUTANT_FULL_REVERSE
    _mm256_storeu_ps(got.data(), _mm256_shuffle_ps(v, v, 0x1b));
#else
    _mm256_storeu_ps(got.data(), _mm256_permutevar8x32_ps(v, indices));
#endif
    require(got == std::array<float,8>{8,7,6,5,4,3,2,1}, "full reverse");
    _mm256_storeu_ps(got.data(), _mm256_hadd_ps(v, v));
    require(got == std::array<float,8>{3,7,3,7,11,15,11,15}, "hadd half boundary");
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    require(_mm_cvtss_f32(sum) == 36.0f, "horizontal sum");
}
VECTOR_TARGET void check_blend() {
    // Bit 31 controls selection; 1 and 0x7fffffff are nonzero but select false.
    const __m256i bits = _mm256_setr_epi32(0, 1, -1, INT32_MIN, INT32_MAX, -2, 2, 0);
    std::array<float,8> got{};
    _mm256_storeu_ps(got.data(), _mm256_blendv_ps(_mm256_set1_ps(10),
        _mm256_set1_ps(20), _mm256_castsi256_ps(bits)));
    require(got == std::array<float,8>{10,10,20,20,10,20,10,10}, "mask sign bit");
}
VECTOR_TARGET void check_fma() {
    volatile float input_a = 1.0f + std::ldexp(1.0f, -23);
    volatile float input_b = 1.0f - std::ldexp(1.0f, -23);
    const float a = input_a, b = input_b;
    volatile float product = a * b;  // require a rounded FP32 intermediate
    const float separate = product - 1.0f;
    const float expected = -std::ldexp(1.0f, -46);
    std::array<float,8> got{};
    _mm256_storeu_ps(got.data(), _mm256_fmadd_ps(_mm256_set1_ps(a),
        _mm256_set1_ps(b), _mm256_set1_ps(-1.0f)));
    require(separate == 0.0f, "separate rounding reference");
    require(std::fma(a, b, -1.0f) == expected, "scalar fused reference");
    for (float lane : got) require(lane == expected, "fused rounding");
}
VECTOR_TARGET void check_conversion() {
    const unsigned saved = _mm_getcsr();
    _mm_setcsr((saved & ~_MM_ROUND_MASK) | _MM_ROUND_NEAREST);
    const __m128 values = _mm_setr_ps(1.9f, -1.9f, 2.5f, -2.5f);
    std::array<int,4> truncated{}, rounded{};
    _mm_storeu_si128(reinterpret_cast<__m128i*>(truncated.data()), _mm_cvttps_epi32(values));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(rounded.data()), _mm_cvtps_epi32(values));
    _mm_setcsr(saved);
    require(truncated == std::array<int,4>{1,-1,2,-2}, "truncate toward zero");
    require(rounded == std::array<int,4>{2,-2,2,-2}, "round nearest, ties even");
}
VECTOR_TARGET void check_nan() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const __m256 a = _mm256_setr_ps(nan, 1, -0.0f, 0.0f, 2, 3, 4, 5);
    const __m256 b = _mm256_setr_ps(1, nan, 0.0f, -0.0f, 1, 2, 3, 4);
    std::array<float,8> got{};
    _mm256_storeu_ps(got.data(), _mm256_max_ps(a, b));
    // x86 MAXPS chooses operand two for NaNs and equal signed zeros.
    require(got[0] == 1 && std::isnan(got[1]), "max NaN operand order");
    require(!std::signbit(got[2]) && std::signbit(got[3]), "max signed zero operand order");
}
#endif

void check_lengths(bool vector) {
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t n = 0; n <= 97; ++n) {
        for (std::size_t offset = 0; offset < 8; ++offset) {
            const std::size_t start = 8 + offset;
            // Inputs end exactly at n: ASan can detect a vector read beyond the tail.
            std::vector<float> a(start + n, -999), b(a.size(), -999), c(start + n + 8, poison);
            for (std::size_t i = 0; i < n; ++i) {
                a[start+i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 4;
                b[start+i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 8;
            }
#if COURSE_X86
            if (vector) scaled_add_vector(a.data()+start, b.data()+start, c.data()+start, n, 0.5f);
            else
#else
            (void)vector;
#endif
                scaled_add_scalar(a.data()+start, b.data()+start, c.data()+start, n, 0.5f);
            for (std::size_t i = 0; i < n; ++i) {
                // Dyadic fixture values make the independent double reference exact.
                const double expected = (static_cast<double>(a[start+i]) + b[start+i]) / 2;
                require(std::isfinite(c[start+i]) && c[start+i] == expected, "tail/value mismatch");
            }
            for (std::size_t i = 0; i < start; ++i) require(std::isnan(c[i]), "prefix overwritten");
            for (std::size_t i = start+n; i < c.size(); ++i) require(std::isnan(c[i]), "suffix overwritten");
        }
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("choose scalar, lengths, lanes, blend, fma, conversion, nan, all");
        const std::string mode = argv[1];
        if (mode == "scalar") { check_lengths(false); }
        else {
            if (!has_vector_support()) { std::cout << "SKIP: AVX2 and FMA unavailable\n"; return 77; }
#if COURSE_X86
            if (mode == "lengths" || mode == "all") check_lengths(true);
            if (mode == "lanes" || mode == "all") check_lanes();
            if (mode == "blend" || mode == "all") check_blend();
            if (mode == "fma" || mode == "all") check_fma();
            if (mode == "conversion" || mode == "all") check_conversion();
            if (mode == "nan" || mode == "all") check_nan();
#endif
            require(mode == "lengths" || mode == "lanes" || mode == "blend" || mode == "fma" ||
                    mode == "conversion" || mode == "nan" || mode == "all", "unknown mode");
        }
        std::cout << "PASS " << mode << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
