#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t hd, float scale);
}
