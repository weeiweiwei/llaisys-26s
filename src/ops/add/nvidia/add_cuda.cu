#include "add_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < numel; i += stride) {
        const float result = llaisys::ops::cuda::utils::toFloat(a[i])
                           + llaisys::ops::cuda::utils::toFloat(b[i]);
        c[i] = llaisys::ops::cuda::utils::fromFloat<T>(result);
    }
}

template <typename T>
void launchAdd(std::byte *c, const std::byte *a, const std::byte *b,
               size_t numel, cudaStream_t stream) {
    const unsigned int grid_size = static_cast<unsigned int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    addKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
}
} // namespace

namespace llaisys::ops::cuda {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchAdd<float>(c, a, b, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchAdd<__nv_bfloat16>(c, a, b, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchAdd<__half>(c, a, b, numel, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "Add CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
