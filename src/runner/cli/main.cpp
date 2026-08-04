#include "npu_inference_bench/benchmark_runner.hpp"
#include "npu_inference_bench/generic_onnx_cli.hpp"

#include <string>
#include <vector>

namespace {

int dispatch(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "run-onnx") {
        return npu_inference_bench::runtime_cli::run(argc - 1, argv + 1);
    }
    return npu_inference_bench::benchmark::run_cli(argc, argv);
}

}  // namespace

#ifdef _WIN32
#include <windows.h>

// Windows fills narrow argv in the process ANSI code page, which mangles or silently
// drops every character outside it: reference transcripts in languages with diacritics
// come through corrupted (inflating WER), as would any model or audio path with
// non-ASCII characters. Re-encode the wide argv as UTF-8, which is what the rest of
// the runner already assumes its strings are.
int wmain(int argc, wchar_t* wargv[]) {
    std::vector<std::string> storage;
    storage.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const int need =
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string arg;
        if (need > 1) {
            arg.resize(static_cast<size_t>(need) - 1);
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, arg.data(), need, nullptr, nullptr);
        }
        storage.push_back(std::move(arg));
    }
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& arg : storage) argv.push_back(arg.data());
    argv.push_back(nullptr);
    return dispatch(argc, argv.data());
}
#else
int main(int argc, char* argv[]) { return dispatch(argc, argv); }
#endif
