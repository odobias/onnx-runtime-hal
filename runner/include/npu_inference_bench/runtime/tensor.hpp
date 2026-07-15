#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace npu_inference_bench {
namespace runtime {

enum class TensorElementType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Bool,
};

inline const char* to_string(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float32:  return "float32";
        case TensorElementType::Float64:  return "float64";
        case TensorElementType::Float16:  return "float16";
        case TensorElementType::BFloat16: return "bfloat16";
        case TensorElementType::Int8:     return "int8";
        case TensorElementType::UInt8:    return "uint8";
        case TensorElementType::Int16:    return "int16";
        case TensorElementType::UInt16:   return "uint16";
        case TensorElementType::Int32:    return "int32";
        case TensorElementType::UInt32:   return "uint32";
        case TensorElementType::Int64:    return "int64";
        case TensorElementType::UInt64:   return "uint64";
        case TensorElementType::Bool:     return "bool";
        default:                          return "unknown";
    }
}

inline std::size_t element_size(TensorElementType type) {
    switch (type) {
        case TensorElementType::Float64:
        case TensorElementType::Int64:
        case TensorElementType::UInt64:
            return 8;
        case TensorElementType::Float32:
        case TensorElementType::Int32:
        case TensorElementType::UInt32:
            return 4;
        case TensorElementType::Float16:
        case TensorElementType::BFloat16:
        case TensorElementType::Int16:
        case TensorElementType::UInt16:
            return 2;
        case TensorElementType::Int8:
        case TensorElementType::UInt8:
        case TensorElementType::Bool:
            return 1;
        default:
            throw std::invalid_argument("unsupported tensor element type");
    }
}

inline std::size_t element_count(const std::vector<std::int64_t>& shape,
                                 bool allow_dynamic = false) {
    std::size_t count = 1;
    for (std::int64_t dimension : shape) {
        if (dimension < 0 && allow_dynamic) return 0;
        if (dimension < 0) throw std::invalid_argument("tensor shape contains a dynamic dimension");
        const auto value = static_cast<std::size_t>(dimension);
        if (value != 0 && count > (std::numeric_limits<std::size_t>::max)() / value) {
            throw std::overflow_error("tensor element count overflows size_t");
        }
        count *= value;
    }
    return count;
}

struct TensorDescriptor {
    std::string name;
    TensorElementType type = TensorElementType::Float32;
    std::vector<std::int64_t> shape;
};

struct TensorView {
    std::string name;
    TensorElementType type = TensorElementType::Float32;
    std::vector<std::int64_t> shape;
    const void* data = nullptr;
    std::size_t byte_size = 0;

    void validate() const {
        const std::size_t required = element_count(shape) * element_size(type);
        if (required != byte_size) {
            throw std::invalid_argument(
                "tensor '" + name + "' byte size does not match its type and shape");
        }
        if (required != 0 && data == nullptr) {
            throw std::invalid_argument("tensor '" + name + "' has no data");
        }
    }
};

struct Tensor {
    std::string name;
    TensorElementType type = TensorElementType::Float32;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;

    TensorView view() const {
        return TensorView{name, type, shape, bytes.empty() ? nullptr : bytes.data(), bytes.size()};
    }

    void validate() const { view().validate(); }
};

}  // namespace runtime
}  // namespace npu_inference_bench
