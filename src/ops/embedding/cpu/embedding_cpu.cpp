#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(T *out, const std::int64_t *index, const T *weight, size_t numel, size_t vocab_dims) {
    for (size_t i = 0; i < numel; i++) {
        size_t idx = static_cast<size_t>(index[i]);
        for (size_t j = 0; j < vocab_dims; j++) {
            out[j + i * vocab_dims] = weight[j + idx * vocab_dims];
        }
    }
}

// 这里的调用为什么不用配置template就可以呢
namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t numel, size_t vocab_dims) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const std::int64_t *>(index), reinterpret_cast<const float *>(weight), numel, vocab_dims);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const std::int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), numel, vocab_dims);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const std::int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), numel, vocab_dims);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
