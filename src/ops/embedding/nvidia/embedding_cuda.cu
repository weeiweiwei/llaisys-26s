#include "embedding_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace {
constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int WARPSIZE = 32;
template <typename T>
__global__ void embeddingKernel(T *out, const std::int64_t *index, const T *weight, size_t numel, size_t embedding_dim) {
    // 启动参数配置：block大小应当是32倍数的，grid应当是根据index长度numel来进行选择的
    // 比如numel=10  一个block处理4行，grid应当启动3个block
    size_t tid = threadIdx.x;
    size_t warp_id = tid / WARPSIZE;
    size_t warp_lane = tid % WARPSIZE;
    size_t warpNums = blockDim.x / WARPSIZE;
    // Block处理多行 一个warp处理一个实际的行
    // 先写逻辑 后续考虑类型转换，这个应该不需要类型转换所谓的

    size_t bRow = warpNums * blockIdx.x;
    size_t wRow = bRow + warp_id; // 用到out/index/weight的处理上
    if (wRow >= numel) {
        return;
    }

    size_t warpStride = WARPSIZE;

    for (size_t col = warp_lane; col < embedding_dim; col += warpStride) {
        size_t token = index[wRow];
        out[wRow * embedding_dim + col] = weight[token * embedding_dim + col];
    }
}

template <typename T>
void launchEmbedding(std::byte *out, const std::byte *index, const std::byte *weight,
                     size_t numel, size_t embedding_dim, cudaStream_t stream) {
    const unsigned int warps_per_block = BLOCK_SIZE / WARPSIZE;
    const unsigned int grid_size = static_cast<unsigned int>((numel + warps_per_block - 1) / warps_per_block);
    embeddingKernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const std::int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        numel,
        embedding_dim);
}
} // namespace

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t numel, size_t embedding_dim,
               llaisysStream_t stream) {
    if (numel == 0 || embedding_dim == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launchEmbedding<float>(out, index, weight, numel, embedding_dim, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launchEmbedding<__nv_bfloat16>(out, index, weight, numel, embedding_dim, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launchEmbedding<__half>(out, index, weight, numel, embedding_dim, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const cudaError_t status = cudaGetLastError();
    ASSERT(status == cudaSuccess, "Embedding CUDA kernel launch failed: " << cudaGetErrorString(status));
}
} // namespace llaisys::ops::cuda
