#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t M, size_t D, float eps,
              llaisysStream_t stream);
}
