#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace jllm {
struct GgufValue {
    uint32_t type = 0;
    uint64_t integer = 0; // raw little-endian bits for integer values
    double real = 0;
    std::string text;
    uint32_t element_type = 0;
    uint64_t count = 0;
    std::vector<std::string> strings; // numeric arrays are bounds-checked/skipped
};
struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};
struct GgufDocument {
    uint64_t file_size = 0;
    uint64_t data_offset = 0;
    uint32_t alignment = 32;
    std::map<std::string, GgufValue> metadata;
    std::vector<GgufTensor> tensors;
};
// Teaching profile: little-endian GGUF v3; scalar/string/flat-array metadata;
// F32, F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1 and Q2_K..Q8_K tensors.
// Unsupported formats fail explicitly. Parsing does not establish model support.
GgufDocument read_gguf(const std::string& path);
}
