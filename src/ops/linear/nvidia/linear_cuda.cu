#include "linear_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t numel, size_t N) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < numel; i += stride) {
        const float value = llaisys::ops::cuda::utils::toFloat(out[i])
                          + llaisys::ops::cuda::utils::toFloat(bias[i % N]);
        out[i] = llaisys::ops::cuda::utils::fromFloat<T>(value);
    }
}

template <typename T>
void launchLinear(std::byte *out, const std::byte *in, const std::byte *weight,
                  const std::byte *bias, size_t M, size_t K, size_t N,
                  cudaDataType_t data_type, cudaStream_t stream) {
    ASSERT(M <= INT_MAX && K <= INT_MAX && N <= INT_MAX,
           "Linear dimensions exceed the range supported by cuBLAS");

    cublasHandle_t handle = nullptr;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS,
           "cublasCreate failed with status " << static_cast<int>(cublas_status));

    cublas_status = cublasSetStream(handle, stream);
    ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS,
           "cublasSetStream failed with status " << static_cast<int>(cublas_status));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Tensors are row-major. cuBLAS reads the same memory as column-major and
    // computes out^T(N, M) = weight(N, K) * in^T(K, M).
    cublas_status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        weight,
        data_type,
        static_cast<int>(K),
        in,
        data_type,
        static_cast<int>(K),
        &beta,
        out,
        data_type,
        static_cast<int>(N),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS,
           "cublasGemmEx failed with status " << static_cast<int>(cublas_status));

    if (bias != nullptr) {
        const size_t numel = M * N;
        const unsigned int grid_size = static_cast<unsigned int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
        addBiasKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const T *>(bias),
            numel,
            N);
    }

    cublas_status = cublasDestroy(handle);
    ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS,
           "cublasDestroy failed with status " << static_cast<int>(cublas_status));
}
} // namespace

namespace llaisys::ops::cuda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type,
            size_t M, size_t K, size_t N, llaisysStream_t stream) {
    if (M == 0 || N == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchLinear<float>(out, in, weight, bias, M, K, N, CUDA_R_32F, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchLinear<__nv_bfloat16>(out, in, weight, bias, M, K, N, CUDA_R_16BF, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchLinear<__half>(out, in, weight, bias, M, K, N, CUDA_R_16F, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "Linear CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
