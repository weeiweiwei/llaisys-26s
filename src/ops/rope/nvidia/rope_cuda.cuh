#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t S, size_t H, size_t D, float theta,
          llaisysStream_t stream);
}
