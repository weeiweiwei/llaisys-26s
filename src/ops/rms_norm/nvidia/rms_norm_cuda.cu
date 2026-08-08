#include "rms_norm_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight,
                              size_t hidden_dim, float eps) {
    const size_t tid = threadIdx.x;
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_dim;
    __shared__ float partial[BLOCK_SIZE];
    __shared__ float inverse_rms;

    float square_sum = 0.0f;
    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        const float value = llaisys::ops::cuda::utils::toFloat(in[row_offset + i]);
        square_sum += value * value;
    }

    partial[tid] = square_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        inverse_rms = rsqrtf(partial[0] / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    for (size_t i = tid; i < hidden_dim; i += blockDim.x) {
        const float value = llaisys::ops::cuda::utils::toFloat(in[row_offset + i]);
        const float scale = llaisys::ops::cuda::utils::toFloat(weight[i]);
        out[row_offset + i]
            = llaisys::ops::cuda::utils::fromFloat<T>(value * inverse_rms * scale);
    }
}

template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
                   size_t M, size_t D, float eps, cudaStream_t stream) {
    const unsigned int grid_size = static_cast<unsigned int>(M);
    rmsNormKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        D,
        eps);
}
} // namespace

namespace llaisys::ops::cuda {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t M, size_t D, float eps,
              llaisysStream_t stream) {
    if (M == 0 || D == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchRmsNorm<float>(out, in, weight, M, D, eps, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchRmsNorm<__nv_bfloat16>(out, in, weight, M, D, eps, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchRmsNorm<__half>(out, in, weight, M, D, eps, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "RMS Norm CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
