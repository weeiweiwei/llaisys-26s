#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::ops::cuda::utils {

__device__ __forceinline__ float toFloat(float value) {
    return value;
}

__device__ __forceinline__ float toFloat(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float toFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

} // namespace llaisys::ops::cuda::utils
