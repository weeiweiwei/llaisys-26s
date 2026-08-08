#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t numel, size_t embedding_dim,
               llaisysStream_t stream);
}
