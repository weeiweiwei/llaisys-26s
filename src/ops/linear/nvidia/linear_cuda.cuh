#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type,
            size_t M, size_t K, size_t N, llaisysStream_t stream);
}
