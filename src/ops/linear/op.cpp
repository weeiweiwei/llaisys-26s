#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    if (bias) {
        CHECK_SAME_DEVICE(bias, out);
        CHECK_SAME_DTYPE(bias->dtype(), out->dtype());
    }

    // in: (M, K), weight: (N, K), bias: (N,) or null, out: (M, N)
    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K, "linear: weight K-dim mismatch, weight.shape[1] must == in.shape[1]");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "linear: out shape mismatch");
    ASSERT(!bias || (bias->shape().size() == 1 && bias->shape()[0] == N), "linear: bias shape mismatch");

    std::byte *bias_data = bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, K, N);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, K, N);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
