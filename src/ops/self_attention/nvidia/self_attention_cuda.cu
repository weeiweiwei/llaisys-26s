/*
 * Adapted from Learning-CUDA/src/FA2.cu and FA2_compatible.cu.
 * The source repository is licensed under the MIT License.
 */

#include "self_attention_cuda.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/my_cuda_util.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

namespace cuda_utils = llaisys::ops::cuda::utils;

constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS_PER_BLOCK = 4;
constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;
constexpr unsigned int KV_TILE_SIZE = 16;

__device__ __forceinline__ float warpReduceSum(float value) {
    constexpr unsigned int mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return __shfl_sync(mask, value, 0);
}

template <typename T>
struct SelfAttentionParams {
    T *output;
    const T *query;
    const T *key;
    const T *value;

    size_t qlen;
    size_t kvlen;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    float scale;
};

template <typename T>
__global__ void selfAttentionKernel(SelfAttentionParams<T> params) {
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const size_t q_head = blockIdx.y;
    const size_t q_row =
        static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const bool valid_q = q_row < params.qlen;

    const size_t group_size = params.q_heads / params.kv_heads;
    const size_t kv_head = q_head / group_size;
    const size_t causal_offset = params.kvlen - params.qlen;

    // Dynamic shared-memory layout:
    // [Q rows][K tile][V tile][FP32 output accumulators].
    extern __shared__ float shared[];
    float *q_shared = shared;
    float *k_shared = q_shared + WARPS_PER_BLOCK * params.head_dim;
    float *v_shared = k_shared + KV_TILE_SIZE * params.head_dim;
    float *acc_shared = v_shared + KV_TILE_SIZE * params.head_dim;

    float *q_row_shared = q_shared + warp * params.head_dim;
    float *acc_row_shared = acc_shared + warp * params.head_dim;

    for (size_t d = lane; d < params.head_dim; d += WARP_SIZE) {
        if (valid_q) {
            const size_t q_index =
                (q_row * params.q_heads + q_head) * params.head_dim + d;
            q_row_shared[d] = cuda_utils::toFloat(params.query[q_index]);
        } else {
            q_row_shared[d] = 0.0f;
        }
        acc_row_shared[d] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const size_t kv_tiles =
        (params.kvlen + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
    for (size_t tile = 0; tile < kv_tiles; ++tile) {
        const size_t tile_start = tile * KV_TILE_SIZE;
        const size_t tile_elements = KV_TILE_SIZE * params.head_dim;

        // Every thread must reach both block-wide barriers, including warps
        // whose q_row belongs to the final partial Q tile.
        for (size_t i = threadIdx.x; i < tile_elements; i += blockDim.x) {
            const size_t key_in_tile = i / params.head_dim;
            const size_t d = i % params.head_dim;
            const size_t key_row = tile_start + key_in_tile;

            if (key_row < params.kvlen) {
                const size_t kv_index =
                    (key_row * params.kv_heads + kv_head) * params.head_dim +
                    d;
                k_shared[i] = cuda_utils::toFloat(params.key[kv_index]);
                v_shared[i] = cuda_utils::toFloat(params.value[kv_index]);
            } else {
                k_shared[i] = 0.0f;
                v_shared[i] = 0.0f;
            }
        }
        __syncthreads();

        for (size_t key_in_tile = 0; key_in_tile < KV_TILE_SIZE;
             ++key_in_tile) {
            const size_t key_row = tile_start + key_in_tile;
            const bool valid_key =
                valid_q && key_row < params.kvlen &&
                key_row <= causal_offset + q_row;

            float partial_dot = 0.0f;
            if (valid_key) {
                const float *k_row_shared =
                    k_shared + key_in_tile * params.head_dim;
                for (size_t d = lane; d < params.head_dim; d += WARP_SIZE) {
                    partial_dot += q_row_shared[d] * k_row_shared[d];
                }
            }

            const float qk = warpReduceSum(partial_dot);
            if (valid_key) {
                const float score = qk * params.scale;
                const float new_max = fmaxf(running_max, score);
                const float old_weight =
                    running_max == -INFINITY
                        ? 0.0f
                        : expf(running_max - new_max);
                const float new_weight = expf(score - new_max);

                for (size_t d = lane; d < params.head_dim; d += WARP_SIZE) {
                    const float value =
                        v_shared[key_in_tile * params.head_dim + d];
                    acc_row_shared[d] =
                        old_weight * acc_row_shared[d] + new_weight * value;
                }

                running_sum = old_weight * running_sum + new_weight;
                running_max = new_max;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        const float inverse_sum =
            running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        for (size_t d = lane; d < params.head_dim; d += WARP_SIZE) {
            const size_t output_index =
                (q_row * params.q_heads + q_head) * params.head_dim + d;
            params.output[output_index] =
                cuda_utils::fromFloat<T>(acc_row_shared[d] * inverse_sum);
        }
    }
}

template <typename T>
cudaError_t launchSelfAttention(std::byte *attn_val, const std::byte *q,
                                const std::byte *k, const std::byte *v,
                                size_t qlen, size_t kvlen, size_t nh,
                                size_t nkvh, size_t hd, float scale,
                                cudaStream_t stream) {
    const size_t shared_floats =
        (2 * WARPS_PER_BLOCK + 2 * KV_TILE_SIZE) * hd;
    const size_t shared_bytes = shared_floats * sizeof(float);
    if (shared_bytes >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    int default_shared_limit = 0;
    status = cudaDeviceGetAttribute(&default_shared_limit,
                                    cudaDevAttrMaxSharedMemoryPerBlock,
                                    device);
    if (status != cudaSuccess) {
        return status;
    }

    if (shared_bytes > static_cast<size_t>(default_shared_limit)) {
        int opt_in_shared_limit = 0;
        status = cudaDeviceGetAttribute(
            &opt_in_shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
            device);
        if (status != cudaSuccess) {
            return status;
        }
        if (shared_bytes > static_cast<size_t>(opt_in_shared_limit)) {
            return cudaErrorInvalidConfiguration;
        }

        status = cudaFuncSetAttribute(
            selfAttentionKernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        if (status != cudaSuccess) {
            return status;
        }
    }

    SelfAttentionParams<T> params{
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        qlen,
        kvlen,
        nh,
        nkvh,
        hd,
        scale};

    const dim3 block(BLOCK_SIZE);
    const dim3 grid(
        static_cast<unsigned int>((qlen + WARPS_PER_BLOCK - 1) /
                                  WARPS_PER_BLOCK),
        static_cast<unsigned int>(nh));
    selfAttentionKernel<<<grid, block, shared_bytes, stream>>>(params);
    return cudaGetLastError();
}

} // namespace

namespace llaisys::ops::cuda {

void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t hd, float scale,
                    llaisysStream_t stream) {
    if (qlen == 0) {
        return;
    }

    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = cudaSuccess;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        status = launchSelfAttention<float>(attn_val, q, k, v, qlen, kvlen,
                                            nh, nkvh, hd, scale, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        status = launchSelfAttention<__nv_bfloat16>(
            attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, scale,
            cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        status = launchSelfAttention<__half>(attn_val, q, k, v, qlen, kvlen,
                                             nh, nkvh, hd, scale,
                                             cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    ASSERT(status == cudaSuccess,
           "Self Attention CUDA launch failed: "
               << cudaGetErrorString(status));
}

} // namespace llaisys::ops::cuda
