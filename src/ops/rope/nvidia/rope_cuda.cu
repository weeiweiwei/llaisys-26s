#include "rope_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int WARPSIZE = 32;
template <typename T>
__global__ void ropeKernel(T *out, const T *in, const std::int64_t *pos_ids, size_t S, size_t H, size_t D, float theta) {
    // 半程共享同一theta 需要根据theta和相对位置来去计算  然后H是独立的  S,H两个并行维度 内部主要是使用D来去
    // 如果多头并行的话--y对应的是Head个数  x对应的是需要处理的序列的idx

    // 还是warp来去处理吧 一行对应一个warp
    // 一个block对应多行
    const size_t warp_id = threadIdx.x / WARPSIZE;
    const size_t warp_lane = threadIdx.x % WARPSIZE;
    const size_t warps_per_block = blockDim.x / WARPSIZE;
    const size_t row = blockIdx.x * warps_per_block + warp_id;
    const size_t head = blockIdx.y;
    const size_t half_D = D / 2;

    if (row >= S || head >= H) {
        return;
    }

    const float pos = static_cast<float>(pos_ids[row]);
    const size_t base = row * H * D + head * D;

    for (size_t col = warp_lane; col < half_D; col += WARPSIZE) {
        const float exponent = 2.0f * static_cast<float>(col) / static_cast<float>(D);
        const float angle = pos / powf(theta, exponent);
        float sin_val, cos_val;
        sincosf(angle, &sin_val, &cos_val); // cuda提供的辅助函数

        const float a = llaisys::ops::cuda::utils::toFloat(in[base + col]);
        const float b = llaisys::ops::cuda::utils::toFloat(in[base + col + half_D]);

        out[base + col] = llaisys::ops::cuda::utils::fromFloat<T>(a * cos_val - b * sin_val);
        out[base + col + half_D] = llaisys::ops::cuda::utils::fromFloat<T>(b * cos_val + a * sin_val);
    }
}

template <typename T>
void launchRope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
                size_t S, size_t H, size_t D, float theta, cudaStream_t stream) {
    const unsigned int warps_per_block = BLOCK_SIZE / WARPSIZE;
    const dim3 grid_size(
        static_cast<unsigned int>((S + warps_per_block - 1) / warps_per_block),
        static_cast<unsigned int>(H));
    ropeKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const std::int64_t *>(pos_ids),
        S,
        H,
        D,
        theta);
}
} // namespace

namespace llaisys::ops::cuda {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t S, size_t H, size_t D, float theta,
          llaisysStream_t stream) {
    if (S == 0 || H == 0 || D == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchRope<float>(out, in, pos_ids, S, H, D, theta, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchRope<__nv_bfloat16>(out, in, pos_ids, S, H, D, theta, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchRope<__half>(out, in, pos_ids, S, H, D, theta, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "RoPE CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
