#pragma once

#include "npu_inference_bench/runtime/model_session.hpp"

#ifdef NPU_INFERENCE_BENCH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace npu_inference_bench {
namespace runtime {
namespace detail {

class OrtSessionAccess {
public:
#ifdef NPU_INFERENCE_BENCH_ORT
    static Ort::Session& get(ModelSession& session) {
        return *static_cast<Ort::Session*>(session.native_handle());
    }
#endif
};

}  // namespace detail
}  // namespace runtime
}  // namespace npu_inference_bench
