#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// bf16 / fp16 不支持 < 比较也不支持 int 构造。
// 参考 add 的做法，用 if constexpr 分支处理：对于bf16/fp16使用utils来去处理
// idx Tensor 应该是默认使用int64_t/size_t 数据类型而不是T模板类型
template <typename T>
void argmax_(std::int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float best_val = llaisys::utils::cast<float>(vals[0]);
        size_t best_idx = 0;
        for (size_t i = 1; i < numel; i++) {
            float val = llaisys::utils::cast<float>(vals[i]);
            if (best_val < val) {
                best_val = val;
                best_idx = i;
            }
        }
        *max_idx = llaisys::utils::cast<std::int64_t>(static_cast<float>(best_idx));
        *max_val = llaisys::utils::cast<T>(best_val);
    } else {
        T best_val = vals[0];
        size_t best_idx = 0;
        for (size_t i = 1; i < numel; i++) {
            if (best_val < vals[i]) {
                best_val = vals[i];
                best_idx = static_cast<std::int64_t>(i);
            }
        }
        *max_idx = best_idx;
        *max_val = best_val;
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<std::int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<std::int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<std::int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
