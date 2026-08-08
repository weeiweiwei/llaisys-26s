#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_cuda.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3
               && v->ndim() == 3,
           "self_attention: Q/K/V/output must all be 3D tensors");
    ASSERT(attn_val->isContiguous() && q->isContiguous()
               && k->isContiguous() && v->isContiguous(),
           "self_attention: Q/K/V/output must be contiguous");

    // Q: (qlen, nh, hd)
    // K: (kvlen, nkvh, hd)
    // V: (kvlen, nkvh, hd)
    // attn_val: (qlen, nh, hd)
    size_t qlen  = q->shape()[0];
    size_t nh    = q->shape()[1];
    size_t hd    = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh  = k->shape()[1];

    ASSERT(nkvh > 0 && nh % nkvh == 0,
           "self_attention: q_heads must be divisible by kv_heads");
    ASSERT(kvlen >= qlen,
           "self_attention: causal KV cache must include all query tokens");
    ASSERT(k->shape()[2] == hd, "self_attention: k head_dim mismatch");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh && v->shape()[2] == hd,
           "self_attention: v shape mismatch");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "self_attention: attn_val shape mismatch, must match Q");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return cuda::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                    attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale,
                                    llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
