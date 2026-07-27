#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t D, float eps) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 0; i < M; i++) {
            // sum of squares over row i
            float sum_sq = 0;
            for (size_t k = 0; k < D; k++) {
                float val = llaisys::utils::cast<float>(in[i * D + k]);
                sum_sq += val * val;
            }
            float r_rms = 1.0f / std::sqrt(sum_sq / D + eps);
            for (size_t k = 0; k < D; k++) {
                float val = llaisys::utils::cast<float>(in[i * D + k]);
                float w = llaisys::utils::cast<float>(weight[k]);
                // 转为原本的数据类型 返回
                out[i * D + k] = llaisys::utils::cast<T>(val * r_rms * w);
            }
        }
    } else {
        for (size_t i = 0; i < M; i++) {
            T sum_sq = 0;
            for (size_t k = 0; k < D; k++) {
                sum_sq = sum_sq + in[i * D + k] * in[i * D + k];
            }
            T r_rms = static_cast<T>(1) / std::sqrt(static_cast<float>(sum_sq) / D + eps);
            for (size_t k = 0; k < D; k++) {
                out[i * D + k] = in[i * D + k] * r_rms * weight[k];
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight,
              llaisysDataType_t type, size_t M, size_t D, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), M, D, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), M, D, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), M, D, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
