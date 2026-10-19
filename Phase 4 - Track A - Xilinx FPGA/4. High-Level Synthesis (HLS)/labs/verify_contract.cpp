#include "matvec_contract.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Independent scalar reference: division/remainder instead of add-half/shift.
static std::int16_t reference(std::int64_t sum, unsigned shift) {
    const std::int64_t divisor = std::int64_t{1} << shift;
    std::int64_t quotient = sum / divisor;
    const std::int64_t remainder = sum % divisor;
    if (2 * std::abs(remainder) >= divisor && remainder != 0)
        quotient += remainder > 0 ? 1 : -1;
    return static_cast<std::int16_t>(std::clamp<std::int64_t>(quotient, -32768, 32767));
}

// Analytical service lower bound, not a hardware schedule: independent banks,
// p reads/bank/cycle, no broadcasts, no reuse, reads only, all requests ready.
static std::size_t bank_cycles(const std::vector<std::size_t>& addresses,
                               std::size_t banks, std::size_t ports) {
    if (banks == 0 || ports == 0) throw std::invalid_argument("zero bank or port count");
    std::vector<std::size_t> count(banks);
    for (auto address : addresses) ++count[address % banks];
    const auto maximum = *std::max_element(count.begin(), count.end());
    return maximum / ports + (maximum % ports != 0);
}

int main() {
    try {
        course::Weights w{};
        course::Vector x{};
        course::Output y{};
        w[0][0] = 1; w[0][1] = 2; w[0][2] = 3;
        w[1][0] = -1; w[1][1] = 0; w[1][2] = 2;
        x[0] = 4; x[1] = -2; x[2] = 5;
        require(course::matvec(w, x, y, 2, 3, 0), "fixture rejected");
        require(y[0] == 15 && y[1] == 6, "ragged fixture mismatch");
        for (unsigned shift = 0; shift <= 15; ++shift) {
            for (auto value : {std::numeric_limits<std::int32_t>::min(), -1048576, -32769,
                               -32768, -7, -3, -1, 0, 1, 3, 7, 32767, 32768, 1048576,
                               std::numeric_limits<std::int32_t>::max()})
                require(course::requantize(value, shift) == reference(value, shift),
                        "rounding or saturation mismatch");
        }
        require(course::requantize(3, 1) == 2 && course::requantize(-3, 1) == -2,
                "halfway rounding contract mismatch");
        for (std::size_t r = 0; r < course::max_rows; ++r)
            for (std::size_t c = 0; c < course::max_cols; ++c) w[r][c] = -128;
        std::fill(std::begin(x), std::end(x), -128);
        require(course::matvec(w, x, y, 8, 64, 0), "extreme shape rejected");
        require(y[0] == 32767 && y[7] == 32767, "positive saturation mismatch");
        std::fill(std::begin(x), std::end(x), 127);
        require(course::matvec(w, x, y, 8, 64, 0) && y[7] == -32768,
                "negative saturation mismatch");

        std::uint32_t state = 0x12345678;
        auto next = [&state]() {
            state = state * 1664525u + 1013904223u;
            return static_cast<std::int8_t>(static_cast<int>(state >> 24) - 128);
        };
        std::size_t shapes = 0;
        for (std::size_t rows = 1; rows <= 8; ++rows) {
            for (std::size_t cols = 1; cols <= 64; ++cols) {
                for (auto& row : w) for (auto& v : row) v = next();
                for (auto& v : x) v = next();
                const unsigned shift = static_cast<unsigned>((rows + cols) % 16);
                std::fill(std::begin(y), std::end(y), 12345);
                require(course::matvec(w, x, y, rows, cols, shift), "valid shape rejected");
                for (std::size_t r = 0; r < rows; ++r) {
                    std::int64_t sum = 0;
                    for (std::size_t c = 0; c < cols; ++c)
                        sum += std::int64_t{w[r][c]} * std::int64_t{x[c]};
                    require(y[r] == reference(sum, shift), "scalar reference mismatch");
                }
                for (std::size_t r = rows; r < 8; ++r)
                    require(y[r] == 12345, "inactive output modified");
                ++shapes;
            }
        }
        const std::array<std::array<unsigned, 3>, 5> invalid{{{0,1,0},{9,1,0},{1,0,0},{1,65,0},{1,1,16}}};
        for (auto shape : invalid) {
            std::fill(std::begin(y), std::end(y), 23456);
            require(!course::matvec(w, x, y, shape[0], shape[1], shape[2]), "invalid accepted");
            require(std::all_of(std::begin(y), std::end(y), [](auto v){return v == 23456;}),
                    "invalid request modified output");
        }
        require(bank_cycles({0,1,2,3},4,1)==1, "cyclic bank fixture");
        require(bank_cycles({0,4,8,12},4,1)==4, "stride bank conflict");
        require(bank_cycles({0,4,8,12},4,2)==2, "dual port lower bound");
        require(bank_cycles({0,64,128,192},4,1)==4, "row stride conflict");
        require(bank_cycles({},4,1)==0, "empty request count");
        bool rejected = false;
        try { (void)bank_cycles({0},0,1); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected, "zero banks accepted");
        std::cout << "verified_shapes=" << shapes << " rounding_cases=240 bank_cases=6\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
