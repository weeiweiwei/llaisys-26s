#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight,
              llaisysDataType_t type, size_t M, size_t D, float eps);
}
