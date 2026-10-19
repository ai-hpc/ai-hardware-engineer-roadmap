#include "jllm_gguf.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace jllm {
namespace {
constexpr uint64_t HEADER_BUDGET = 64ULL * 1024 * 1024;
constexpr uint64_t MAX_STRING = 16ULL * 1024 * 1024;
bool valid_utf8(const std::string& text) {
    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i++]);
        if (first < 0x80) continue;
        unsigned tails;
        uint32_t code, minimum;
        if (first >= 0xc2 && first <= 0xdf) { tails = 1; code = first & 0x1f; minimum = 0x80; }
        else if (first >= 0xe0 && first <= 0xef) { tails = 2; code = first & 0x0f; minimum = 0x800; }
        else if (first >= 0xf0 && first <= 0xf4) { tails = 3; code = first & 0x07; minimum = 0x10000; }
        else return false;
        if (tails > text.size()-i) return false;
        for (unsigned n = 0; n < tails; ++n) {
            const auto next = static_cast<unsigned char>(text[i++]);
            if (next < 0x80 || next > 0xbf) return false;
            code = (code << 6) | (next & 0x3f);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
    }
    return true;
}
bool valid_key(const std::string& key) {
    if (key.empty() || key.front() == '.' || key.back() == '.') return false;
    bool previous_dot = false;
    for (unsigned char ch : key) {
        if (ch == '.') {
            if (previous_dot) return false;
            previous_dot = true;
        } else {
            if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_')) return false;
            previous_dot = false;
        }
    }
    return true;
}
uint64_t product(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max()/b)
        throw std::runtime_error("GGUF size overflow");
    return a*b;
}
class Reader {
    std::ifstream input;
public:
    uint64_t size = 0, position = 0;
    explicit Reader(const std::string& path) : input(path, std::ios::binary | std::ios::ate) {
        if (!input) throw std::runtime_error("cannot open GGUF");
        const auto end = input.tellg();
        if (end < 0) throw std::runtime_error("cannot determine GGUF size");
        size = static_cast<uint64_t>(end);
        input.seekg(0);
    }
    void available(uint64_t n) const {
        if (position > size || n > size-position) throw std::runtime_error("truncated GGUF");
        if (position > HEADER_BUDGET || n > HEADER_BUDGET-position)
            throw std::runtime_error("GGUF metadata exceeds teaching parser budget");
    }
    void bytes(char* dst, uint64_t n) {
        available(n);
        if (n && !input.read(dst, static_cast<std::streamsize>(n)))
            throw std::runtime_error("GGUF read failed");
        position += n;
    }
    uint64_t integer(unsigned width) {
        unsigned char data[8] = {};
        bytes(reinterpret_cast<char*>(data), width);
        uint64_t value = 0;
        for (unsigned i = 0; i < width; ++i) value |= uint64_t(data[i]) << (8*i);
        return value;
    }
    void skip(uint64_t n) {
        available(n);
        input.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        if (!input) throw std::runtime_error("GGUF seek failed");
        position += n;
    }
    std::string string(uint64_t maximum = MAX_STRING) {
        const uint64_t n = integer(8);
        if (n > maximum) throw std::runtime_error("GGUF string exceeds limit");
        available(n); // validate before allocating
        std::string result(static_cast<std::size_t>(n), '\0');
        bytes(result.data(), n);
        if (!valid_utf8(result)) throw std::runtime_error("invalid GGUF UTF-8 string");
        return result;
    }
};
unsigned scalar_width(uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: throw std::runtime_error("unsupported GGUF metadata type");
    }
}
GgufValue value(Reader& r, uint32_t type) {
    GgufValue result; result.type = type;
    if (type == 8) result.text = r.string();
    else if (type == 9) {
        result.element_type = static_cast<uint32_t>(r.integer(4));
        result.count = r.integer(8);
        if (result.count > 1000000) throw std::runtime_error("GGUF array exceeds limit");
        if (result.element_type == 8) {
            r.available(product(result.count, 8));
            result.strings.reserve(static_cast<std::size_t>(result.count));
            for (uint64_t i = 0; i < result.count; ++i) result.strings.push_back(r.string());
        } else if (result.element_type == 7) {
            for (uint64_t i = 0; i < result.count; ++i)
                if (r.integer(1) > 1) throw std::runtime_error("invalid GGUF bool");
        } else r.skip(product(result.count, scalar_width(result.element_type)));
    } else {
        result.integer = r.integer(scalar_width(type));
        if (type == 7 && result.integer > 1) throw std::runtime_error("invalid GGUF bool");
        if (type == 6) {
            uint32_t bits = static_cast<uint32_t>(result.integer); float number;
            std::memcpy(&number, &bits, sizeof number); result.real = number;
        } else if (type == 12) {
            std::memcpy(&result.real, &result.integer, sizeof result.real);
        }
    }
    return result;
}
std::pair<uint64_t,uint64_t> block_layout(uint32_t type) {
    switch (type) {
        case 0: return {1,4}; case 1: return {1,2};
        case 2: return {32,18}; case 3: return {32,20};
        case 6: return {32,22}; case 7: return {32,24};
        case 8: return {32,34}; case 9: return {32,36};
        case 10: return {256,84}; case 11: return {256,110};
        case 12: return {256,144}; case 13: return {256,176};
        case 14: return {256,210}; case 15: return {256,292};
        default: throw std::runtime_error("unsupported GGML tensor type in teaching parser");
    }
}
}
GgufDocument read_gguf(const std::string& path) {
    Reader r(path);
    if (r.integer(4) != 0x46554747 || r.integer(4) != 3)
        throw std::runtime_error("expected little-endian GGUF version 3");
    const uint64_t tensors = r.integer(8), metadata = r.integer(8);
    if (tensors > 65536 || metadata > 100000) throw std::runtime_error("GGUF entry count exceeds limit");
    GgufDocument result; result.file_size = r.size;
    for (uint64_t i = 0; i < metadata; ++i) {
        auto key = r.string(65535);
        if (!valid_key(key)) throw std::runtime_error("invalid GGUF key");
        auto item = value(r, static_cast<uint32_t>(r.integer(4)));
        if (!result.metadata.emplace(key, std::move(item)).second)
            throw std::runtime_error("duplicate GGUF metadata key");
    }
    auto alignment = result.metadata.find("general.alignment");
    if (alignment != result.metadata.end()) {
        if (alignment->second.type != 4 || alignment->second.integer < 8 || alignment->second.integer > 1048576
            || (alignment->second.integer & (alignment->second.integer-1)))
            throw std::runtime_error("invalid GGUF alignment");
        result.alignment = static_cast<uint32_t>(alignment->second.integer);
    }
    std::set<std::string> names;
    for (uint64_t i = 0; i < tensors; ++i) {
        GgufTensor tensor; tensor.name = r.string(64);
        if (tensor.name.empty() || tensor.name.find('\0') != std::string::npos || !names.insert(tensor.name).second)
            throw std::runtime_error("invalid/duplicate GGUF tensor name");
        const auto dimensions = r.integer(4);
        if (dimensions < 1 || dimensions > 4) throw std::runtime_error("GGUF tensor rank outside 1..4");
        uint64_t elements = 1;
        for (uint64_t d = 0; d < dimensions; ++d) {
            const uint64_t extent = r.integer(8);
            if (!extent) throw std::runtime_error("zero GGUF tensor dimension");
            tensor.shape.push_back(extent); elements = product(elements, extent);
        }
        tensor.type = static_cast<uint32_t>(r.integer(4)); tensor.offset = r.integer(8);
        const auto [block, bytes] = block_layout(tensor.type);
        if (tensor.shape[0] % block) throw std::runtime_error("quantized tensor row is not block aligned");
        tensor.bytes = product(elements/block, bytes);
        if (tensor.offset % result.alignment) throw std::runtime_error("unaligned GGUF tensor offset");
        result.tensors.push_back(std::move(tensor));
    }
    const uint64_t padding = (result.alignment - r.position % result.alignment) % result.alignment;
    result.data_offset = r.position + padding;
    if (tensors && result.data_offset > r.size) throw std::runtime_error("truncated GGUF tensor-data padding");
    std::vector<std::pair<uint64_t,uint64_t>> spans;
    for (const auto& tensor : result.tensors) {
        const uint64_t available = r.size-result.data_offset;
        if (tensor.offset > available || tensor.bytes > available-tensor.offset)
            throw std::runtime_error("GGUF tensor payload outside file");
        spans.emplace_back(tensor.offset, tensor.offset+tensor.bytes);
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i-1].second) throw std::runtime_error("overlapping GGUF tensor payloads");
    return result;
}
}
