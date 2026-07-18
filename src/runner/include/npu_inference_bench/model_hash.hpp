#pragma once

#if !defined(_WIN32)
#error "model_hash.hpp currently requires Windows CNG"
#endif

#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace npu_inference_bench {

namespace model_hash_detail {

using Digest = std::array<unsigned char, 32>;

class Sha256 {
public:
    Sha256() {
        check(BCryptOpenAlgorithmProvider(&algorithm_, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
              "BCryptOpenAlgorithmProvider");
        DWORD bytes = 0;
        DWORD object_size = 0;
        check(BCryptGetProperty(algorithm_, BCRYPT_OBJECT_LENGTH,
                                reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size),
                                &bytes, 0),
              "BCryptGetProperty(BCRYPT_OBJECT_LENGTH)");
        object_.resize(object_size);
        check(BCryptCreateHash(algorithm_, &hash_, object_.data(),
                               static_cast<ULONG>(object_.size()), nullptr, 0, 0),
              "BCryptCreateHash");
    }

    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;

    ~Sha256() {
        if (hash_) BCryptDestroyHash(hash_);
        if (algorithm_) BCryptCloseAlgorithmProvider(algorithm_, 0);
    }

    void update(const void* data, size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        while (size) {
            const ULONG chunk =
                static_cast<ULONG>((std::min)(size, static_cast<size_t>(1u << 30)));
            check(BCryptHashData(hash_, const_cast<PUCHAR>(bytes), chunk, 0),
                  "BCryptHashData");
            bytes += chunk;
            size -= chunk;
        }
    }

    Digest finish() {
        Digest digest{};
        check(BCryptFinishHash(hash_, digest.data(), static_cast<ULONG>(digest.size()), 0),
              "BCryptFinishHash");
        return digest;
    }

private:
    static void check(NTSTATUS status, const char* operation) {
        if (status != 0) {
            std::ostringstream message;
            message << operation << " failed with NTSTATUS 0x" << std::hex
                    << static_cast<unsigned long>(status);
            throw std::runtime_error(message.str());
        }
    }

    BCRYPT_ALG_HANDLE algorithm_ = nullptr;
    BCRYPT_HASH_HANDLE hash_ = nullptr;
    std::vector<unsigned char> object_;
};

inline bool is_model_artifact(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".onnx" || extension == ".xml" || extension == ".bin" ||
           extension == ".data";
}

inline Digest hash_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open model artifact for hashing: " + path.string());

    Sha256 hash;
    std::vector<char> buffer(1 << 20);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed while hashing model artifact: " + path.string());
    return hash.finish();
}

inline std::string hex(const Digest& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : digest) output << std::setw(2) << static_cast<unsigned int>(byte);
    return output.str();
}

}  // namespace model_hash_detail

// Stable identity for the inference graph/weights used by a benchmark. Each model
// artifact is hashed by content, those digests are sorted, then hashed together.
// This is independent of package directory and filenames while still distinguishing
// multi-file encoder/decoder packages and OpenVINO XML/BIN models.
inline std::string model_artifacts_sha256(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    using namespace model_hash_detail;

    std::vector<fs::path> artifacts;
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) {
        artifacts.push_back(path);
    } else if (fs::is_directory(path, ec)) {
        for (fs::recursive_directory_iterator it(path, ec), end; !ec && it != end;
             it.increment(ec)) {
            if (it->is_regular_file(ec) && is_model_artifact(it->path())) {
                artifacts.push_back(it->path());
            }
        }
        if (ec) throw std::runtime_error("cannot enumerate model artifacts: " + path.string());
    }
    if (artifacts.empty()) throw std::runtime_error("no model artifacts found to hash: " + path.string());

    std::vector<Digest> digests;
    digests.reserve(artifacts.size());
    for (const fs::path& artifact : artifacts) digests.push_back(hash_file(artifact));
    std::sort(digests.begin(), digests.end());

    Sha256 package_hash;
    const uint64_t count = static_cast<uint64_t>(digests.size());
    package_hash.update(&count, sizeof(count));
    for (const Digest& digest : digests) package_hash.update(digest.data(), digest.size());
    return hex(package_hash.finish());
}

}  // namespace npu_inference_bench
