#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace npu_inference_bench::runtime {

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

[[nodiscard]] inline const char* to_string(TensorElementType type) {
    using enum TensorElementType;
    switch (type) {
        case Float32:  return "float32";
        case Float64:  return "float64";
        case Float16:  return "float16";
        case BFloat16: return "bfloat16";
        case Int8:     return "int8";
        case UInt8:    return "uint8";
        case Int16:    return "int16";
        case UInt16:   return "uint16";
        case Int32:    return "int32";
        case UInt32:   return "uint32";
        case Int64:    return "int64";
        case UInt64:   return "uint64";
        case Bool:     return "bool";
    }
    return "unknown";
}

[[nodiscard]] inline std::size_t element_size(TensorElementType type) {
    using enum TensorElementType;
    switch (type) {
        case Float64:
        case Int64:
        case UInt64:
            return 8;
        case Float32:
        case Int32:
        case UInt32:
            return 4;
        case Float16:
        case BFloat16:
        case Int16:
        case UInt16:
            return 2;
        case Int8:
        case UInt8:
        case Bool:
            return 1;
    }
    throw std::invalid_argument("unsupported tensor element type");
}

[[nodiscard]] inline std::size_t element_count(std::span<const std::int64_t> shape,
                                               bool allow_dynamic = false) {
    std::size_t count = 1;
    for (std::int64_t dimension : shape) {
        if (dimension < 0 && allow_dynamic) return 0;
        if (dimension < 0) {
            throw std::invalid_argument("tensor shape contains a dynamic dimension");
        }
        const auto value = static_cast<std::size_t>(dimension);
        if (value != 0 && count > (std::numeric_limits<std::size_t>::max)() / value) {
            throw std::overflow_error("tensor element count overflows size_t");
        }
        count *= value;
    }
    return count;
}

[[nodiscard]] inline std::size_t element_count(const std::vector<std::int64_t>& shape,
                                               bool allow_dynamic = false) {
    return element_count(std::span<const std::int64_t>{shape}, allow_dynamic);
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

    [[nodiscard]] std::span<const std::byte> bytes() const {
        return {static_cast<const std::byte*>(data), byte_size};
    }

    void validate() const {
        const std::size_t required = element_count(shape) * element_size(type);
        if (required != byte_size) {
            throw std::invalid_argument(std::format(
                "tensor '{}' byte size does not match its type and shape", name));
        }
        if (required != 0 && data == nullptr) {
            throw std::invalid_argument(std::format("tensor '{}' has no data", name));
        }
    }
};

struct Tensor {
    std::string name;
    TensorElementType type = TensorElementType::Float32;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;

    [[nodiscard]] TensorView view() const {
        return TensorView{
            name,
            type,
            shape,
            bytes.empty() ? nullptr : bytes.data(),
            bytes.size(),
        };
    }

    [[nodiscard]] std::span<const std::byte> byte_span() const {
        return std::as_bytes(std::span{bytes});
    }

    void validate() const { view().validate(); }
};

}  // namespace npu_inference_bench::runtime
