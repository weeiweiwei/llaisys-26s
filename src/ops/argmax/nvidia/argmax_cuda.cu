#include "argmax_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;

struct ArgmaxPair {
    float val;
    std::int64_t idx;
};

__device__ __forceinline__ bool better(const ArgmaxPair &candidate, const ArgmaxPair &current) {
    return candidate.val > current.val
        || (candidate.val == current.val && candidate.idx < current.idx);
}
// 一个Block处理一行的最大值选择
// 1.先将超过block大小的元素 进行预处理
// 2.进行Block内进行比较 half对比、
// 3.tid0内存储正确结果
// ----后续更好的方式应当是Warp级别的 BlockReduce的方式
template <typename T>
__global__ void argmaxKernel(std::int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    const size_t tid = threadIdx.x;
    __shared__ ArgmaxPair partial[BLOCK_SIZE];

    ArgmaxPair best{-INFINITY, static_cast<std::int64_t>(numel)};
    if (tid < numel) {
        best.val = llaisys::ops::cuda::utils::toFloat(vals[tid]);
        best.idx = static_cast<std::int64_t>(tid);

        for (size_t i = tid + blockDim.x; i < numel; i += blockDim.x) {
            ArgmaxPair candidate{
                llaisys::ops::cuda::utils::toFloat(vals[i]),
                static_cast<std::int64_t>(i)};
            if (better(candidate, best)) {
                best = candidate;
            }
        }
    }

    partial[tid] = best;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && better(partial[tid + stride], partial[tid])) {
            partial[tid] = partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = partial[0].idx;
        max_val[0] = llaisys::ops::cuda::utils::fromFloat<T>(partial[0].val);
    }
}

template <typename T>
void launchArgmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
                  size_t numel, cudaStream_t stream) {
    argmaxKernel<<<1, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<std::int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
}
} // namespace

namespace llaisys::ops::cuda {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchArgmax<float>(max_idx, max_val, vals, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchArgmax<__nv_bfloat16>(max_idx, max_val, vals, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchArgmax<__half>(max_idx, max_val, vals, numel, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "Argmax CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
