#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// 所有的tensor中使用BF16/FP16的，都需要通过本地的转换函数来去转换FP32以保证数值对应
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 0; i < numel; i++) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float silu = g / (static_cast<float>(1) + std::exp(-g));
            out[i] = llaisys::utils::cast<T>(u * silu); // 这里转回去type,llaisys::bf16_t就好
        }
    } else {
        for (size_t i = 0; i < numel; i++) {
            T g = gate[i];
            T u = up[i];
            T silu = g / (static_cast<T>(1) + std::exp(-g));
            out[i] = u * silu;
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, std::byte *gate, std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
