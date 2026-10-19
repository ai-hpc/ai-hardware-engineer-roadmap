#ifndef COURSE_MATVEC_CONTRACT_H
#define COURSE_MATVEC_CONTRACT_H
#include <cstddef>
#include <cstdint>

namespace course {
constexpr std::size_t max_rows = 8;
constexpr std::size_t max_cols = 64;
using Weights = std::int8_t[max_rows][max_cols];
using Vector = std::int8_t[max_cols];
using Output = std::int16_t[max_rows];

// Round nearest, ties away from zero, then saturate to signed 16-bit.
// acc is int32, so negating after widening is safe, including INT32_MIN.
inline std::int16_t requantize(std::int32_t acc, unsigned shift) {
    const std::int64_t value = acc;
    const std::int64_t magnitude = value < 0 ? -value : value;
    const std::int64_t half = shift == 0 ? 0 : (std::int64_t{1} << (shift - 1));
    const std::int64_t rounded = (magnitude + half) >> shift;
    const std::int64_t signed_value = value < 0 ? -rounded : rounded;
    if (signed_value > 32767) return 32767;
    if (signed_value < -32768) return -32768;
    return static_cast<std::int16_t>(signed_value);
}

// Caller owns distinct arrays. Invalid shape/shift returns false before writes.
// Requantize's helper precondition is shift <= 15; enforce at this boundary.
inline bool matvec(const Weights& weights, const Vector& x, Output& out,
                   std::size_t rows, std::size_t cols, unsigned shift) {
    if (rows == 0 || rows > max_rows || cols == 0 || cols > max_cols || shift > 15)
        return false;
    for (std::size_t row = 0; row < rows; ++row) {
        std::int32_t sums[4] = {0, 0, 0, 0};
        for (std::size_t base = 0; base < cols; base += 4) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t col = base + lane;
#ifdef COURSE_MUTATE_DROP_TAIL
                if (base + 4 > cols) continue;
#endif
                if (col < cols)
                    sums[lane] += std::int32_t{weights[row][col]} * std::int32_t{x[col]};
            }
        }
        const std::int32_t acc = sums[0] + sums[1] + sums[2] + sums[3];
#ifdef COURSE_MUTATE_WRAP_OUTPUT
        out[row] = static_cast<std::int16_t>(acc);
#else
        out[row] = requantize(acc, shift);
#endif
    }
    return true;
}
}  // namespace course
#endif
