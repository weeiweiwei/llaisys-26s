#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t S, size_t H, size_t D, float theta) {
    size_t half_D = D / 2;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t s = 0; s < S; s++) {
            float pos = static_cast<float>(pos_ids[s]);
            for (size_t h = 0; h < H; h++) {
                for (size_t i = 0; i < half_D; i++) {
                    float freq = pos / std::pow(theta, 2.0f * i / D);
                    float cos_val = std::cos(freq);
                    float sin_val = std::sin(freq);

                    size_t base = s * H * D + h * D;
                    float a = llaisys::utils::cast<float>(in[base + i]);
                    float b = llaisys::utils::cast<float>(in[base + half_D + i]);

                    out[base + i]          = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    out[base + half_D + i] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                }
            }
        }
    } else {
        for (size_t s = 0; s < S; s++) {
            float pos = static_cast<float>(pos_ids[s]);
            for (size_t h = 0; h < H; h++) {
                for (size_t i = 0; i < half_D; i++) {
                    float freq = pos / std::pow(theta, 2.0f * i / D);
                    float cos_val = std::cos(freq);
                    float sin_val = std::sin(freq);

                    size_t base = s * H * D + h * D;
                    T a = in[base + i];
                    T b = in[base + half_D + i];

                    out[base + i]          = static_cast<T>(a * cos_val - b * sin_val);
                    out[base + half_D + i] = static_cast<T>(b * cos_val + a * sin_val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids,
          llaisysDataType_t type, size_t S, size_t H, size_t D, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), S, H, D, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), S, H, D, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), S, H, D, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
