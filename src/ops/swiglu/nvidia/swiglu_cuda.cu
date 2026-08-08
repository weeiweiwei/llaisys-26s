#include "swiglu_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;

// elementwise的 我直接一维所有线程并行的搞了
template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (; i < numel; i += stride) {
        const float g = llaisys::ops::cuda::utils::toFloat(gate[i]);
        const float u = llaisys::ops::cuda::utils::toFloat(up[i]);

        const float silu = g / (1.0f + expf(-g));
        out[i] = llaisys::ops::cuda::utils::fromFloat<T>(u * silu);
    }
}

template <typename T>
void launchSwiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                  size_t numel, cudaStream_t stream) {
    const unsigned int grid_size = static_cast<unsigned int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    swigluKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
}
} // namespace

namespace llaisys::ops::cuda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchSwiglu<float>(out, gate, up, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchSwiglu<__nv_bfloat16>(out, gate, up, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchSwiglu<__half>(out, gate, up, numel, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "SwiGLU CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
