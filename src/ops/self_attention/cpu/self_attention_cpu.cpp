#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

/*
 leading dimension--主要代表head_num*head_dim,一个token对应表示
 ********这里有个重点：他跳过的是一整个token的，也就是说计算的是统一的不同token的同一个headi
 所以使用此计算的外层迭代是以head为循环的
 一个token会有多个(head_num)score得分，间隔的token计算需要跨越leading dim
 */

// C[M][N] = A[M][K] @ B[N][K]^T  × scale
// A 的 leading dimension = lda（A 行间跨越的元素数）
// B 的 leading dimension = ldb（B 行间跨越的元素数）
// C 是连续的 (M, N)，内部计算用 float
static void gemm_NT(float *C, const float *A, const float *B,
                    size_t M, size_t K, size_t N, size_t lda, size_t ldb, float scale) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * N + j] = sum * scale;
        }
    }
}

// C[M][N] = A[M][K] @ B[K][N]
// A 的 leading dimension = lda
// B 的 leading dimension = ldb
// C 是连续的 (M, N)，内部计算用 float
static void gemm_NN(float *C, const float *A, const float *B,
                    size_t M, size_t K, size_t N, size_t lda, size_t ldb) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 因果 mask + softmax
// offset = kvlen - qlen （= 已经存在的历史 token 数）
// 对于存在KV Cache的 体现在N的长度上，N = new(Q) + cache(KV之前存下的)
static void causal_softmax(float *x, size_t M, size_t N, size_t offset) {
    for (size_t i = 0; i < M; i++) {
        // find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        size_t valid_end = offset + i + 1; // 第 i 行可见的最大列索引（包含）
        if (valid_end > N) {
            valid_end = N;
        }

        for (size_t j = 0; j < valid_end; j++) {
            if (x[i * N + j] > max_val) {
                max_val = x[i * N + j];
            }
        }

        // safe softmax
        float sum_exp = 0;
        for (size_t j = 0; j < valid_end; j++) {
            float v = std::exp(x[i * N + j] - max_val);
            x[i * N + j] = v;
            sum_exp += v;
        }
        // masked 部分填 0---因为正常softmax指数结果是>0的，所以取0做下界
        for (size_t j = valid_end; j < N; j++) {
            x[i * N + j] = 0;
        }

        // normalize
        float inv_sum = 1.0f / sum_exp;
        for (size_t j = 0; j < valid_end; j++) {
            x[i * N + j] *= inv_sum;
        }
    }
}

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    size_t n_rep = nh / nkvh; // GQA: 每个 KV head 服务多少个 Q head

    // offset为了控制mask的范围，因为最基本的情况是q = k/v，此时qlen就是mask的边界
    // 但一般是不缓存q的，所以qlen一般就是需要推理的token数，kv就是以及推理的token数(以后者为边界)
    size_t offset = kvlen - qlen;

    // 中间 buffer：全部以 float 计算
    std::vector<float> scores(qlen * kvlen);
    std::vector<float> out_head(qlen * hd);

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // 先转 q/k/v 为 float---llaisys本身的bf16和fp16就需要通过util来去以float的形式处理
        std::vector<float> q_f(qlen * nh * hd);
        std::vector<float> k_f(kvlen * nkvh * hd);
        std::vector<float> v_f(kvlen * nkvh * hd);
        for (size_t i = 0; i < qlen * nh * hd; i++) {
            q_f[i] = llaisys::utils::cast<float>(q[i]);
        }
        for (size_t i = 0; i < kvlen * nkvh * hd; i++) {
            k_f[i] = llaisys::utils::cast<float>(k[i]);
        }
        for (size_t i = 0; i < kvlen * nkvh * hd; i++) {
            v_f[i] = llaisys::utils::cast<float>(v[i]);
        }
        // 核心组成
        for (size_t h = 0; h < nh; h++) {
            size_t kv_h = h / n_rep;
            // Q[h]: (qlen, hd), stride = nh * hd
            gemm_NT(scores.data(), &q_f[h * hd], &k_f[kv_h * hd],
                    qlen, hd, kvlen, nh * hd, nkvh * hd, scale);
            causal_softmax(scores.data(), qlen, kvlen, offset);
            // V[kv_h]: (kvlen, hd), stride = nkvh * hd
            gemm_NN(out_head.data(), scores.data(), &v_f[kv_h * hd],
                    qlen, kvlen, hd, kvlen, nkvh * hd);
            // 写回 attn_val[h]: 非连续，stride = nh * hd
            for (size_t s = 0; s < qlen; s++) {
                for (size_t d = 0; d < hd; d++) {
                    attn_val[s * nh * hd + h * hd + d] = llaisys::utils::cast<T>(out_head[s * hd + d]);
                }
            }
        }
    } else {
        // 核心组成:
        // 在一次外层循环中，gemm计算的数据是一个head的score结果，自然而然顺序存入score中
        // 一次计算的结果是，所有token的一个head下的Attention
        for (size_t h = 0; h < nh; h++) {
            size_t kv_h = h / n_rep; // GQA共享，对于多的k倍Qhead，通过读取k次K/Vhead来去同样处理
            // q[h] 的起始地址 = q + h * hd, 行 stride = nh * hd
            gemm_NT(scores.data(),
                    reinterpret_cast<const float *>(q) + h * hd,
                    reinterpret_cast<const float *>(k) + kv_h * hd,
                    qlen, hd, kvlen, nh * hd, nkvh * hd, scale);
            causal_softmax(scores.data(), qlen, kvlen, offset);
            gemm_NN(out_head.data(), scores.data(),
                    reinterpret_cast<const float *>(v) + kv_h * hd,
                    qlen, kvlen, hd, kvlen, nkvh * hd);
            // 写回 attn_val[h]
            for (size_t s = 0; s < qlen; s++) {
                for (size_t d = 0; d < hd; d++) {
                    reinterpret_cast<float *>(attn_val)[s * nh * hd + h * hd + d] = out_head[s * hd + d];
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v,
                    llaisysDataType_t type, size_t qlen, size_t kvlen,
                    size_t nh, size_t nkvh, size_t hd, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
