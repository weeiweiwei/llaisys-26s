#include "linear_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t K, size_t N) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0;
                for (size_t k = 0; k < K; k++) {
                    sum += llaisys::utils::cast<float>(in[i * K + k])
                         * llaisys::utils::cast<float>(weight[j * K + k]);
                }
                if (bias) {
                    sum += llaisys::utils::cast<float>(bias[j]);
                }
                out[i * N + j] = llaisys::utils::cast<T>(sum);
            }
        }
    } else {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                T sum = 0;
                for (size_t k = 0; k < K; k++) {
                    sum = sum + in[i * K + k] * weight[j * K + k];
                }
                if (bias) {
                    sum = sum + bias[j];
                }
                out[i * N + j] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias,
            llaisysDataType_t type, size_t M, size_t K, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
