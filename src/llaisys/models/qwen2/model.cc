#include "model.hpp"

#include "llaisys/ops.h"
#include "llaisys/tensor.h"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       const int *device_ids,
                       int ndevice)
    : meta_(meta), device_(device), device_id_(device_ids[0]) {

    size_t in_embed_shape[] = {meta_.voc, meta_.hs};
    size_t out_embed_shape[] = {meta_.voc, meta_.hs};
    size_t out_norm_shape[] = {meta_.hs};

    in_embed = tensorCreate(in_embed_shape, 2, meta_.dtype, device_, device_id_);
    out_embed = tensorCreate(out_embed_shape, 2, meta_.dtype, device_, device_id_);
    out_norm_w = tensorCreate(out_norm_shape, 1, meta_.dtype, device_, device_id_);

    attn_norm_w.resize(meta_.nlayer);
    attn_q_w.resize(meta_.nlayer);
    attn_q_b.resize(meta_.nlayer);
    attn_k_w.resize(meta_.nlayer);
    attn_k_b.resize(meta_.nlayer);
    attn_v_w.resize(meta_.nlayer);
    attn_v_b.resize(meta_.nlayer);
    attn_o_w.resize(meta_.nlayer);
    mlp_norm_w.resize(meta_.nlayer);
    mlp_gate_w.resize(meta_.nlayer);
    mlp_up_w.resize(meta_.nlayer);
    mlp_down_w.resize(meta_.nlayer);
    k_cache_.resize(meta_.nlayer);
    v_cache_.resize(meta_.nlayer);

    size_t attn_norm_w_shape[] = {meta_.hs};
    size_t attn_q_w_shape[] = {meta_.nh * meta_.dh, meta_.hs};
    size_t attn_q_b_shape[] = {meta_.nh * meta_.dh};
    size_t attn_k_w_shape[] = {meta_.nkvh * meta_.dh, meta_.hs};
    size_t attn_k_b_shape[] = {meta_.nkvh * meta_.dh};
    size_t attn_v_w_shape[] = {meta_.nkvh * meta_.dh, meta_.hs};
    size_t attn_v_b_shape[] = {meta_.nkvh * meta_.dh};
    size_t attn_o_w_shape[] = {meta_.hs, meta_.nh * meta_.dh};
    size_t mlp_norm_w_shape[] = {meta_.hs};
    size_t mlp_gate_w_shape[] = {meta_.di, meta_.hs};
    size_t mlp_up_w_shape[] = {meta_.di, meta_.hs};
    size_t mlp_down_w_shape[] = {meta_.hs, meta_.di};

    // meta_.maxseq * meta_.nkvh * meta_.dh
    size_t KV_Cache_shape[] = {meta_.maxseq, meta_.nkvh, meta_.dh};

    for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
        k_cache_[layer] = tensorCreate(KV_Cache_shape, 3, meta_.dtype, device_, device_id_);
        v_cache_[layer] = tensorCreate(KV_Cache_shape, 3, meta_.dtype, device_, device_id_);
        attn_norm_w[layer] = tensorCreate(
            attn_norm_w_shape, 1, meta_.dtype, device_, device_id_);
        attn_q_w[layer] = tensorCreate(
            attn_q_w_shape, 2, meta_.dtype, device_, device_id_);
        attn_q_b[layer] = tensorCreate(
            attn_q_b_shape, 1, meta_.dtype, device_, device_id_);
        attn_k_w[layer] = tensorCreate(
            attn_k_w_shape, 2, meta_.dtype, device_, device_id_);
        attn_k_b[layer] = tensorCreate(
            attn_k_b_shape, 1, meta_.dtype, device_, device_id_);
        attn_v_w[layer] = tensorCreate(
            attn_v_w_shape, 2, meta_.dtype, device_, device_id_);
        attn_v_b[layer] = tensorCreate(
            attn_v_b_shape, 1, meta_.dtype, device_, device_id_);
        attn_o_w[layer] = tensorCreate(
            attn_o_w_shape, 2, meta_.dtype, device_, device_id_);
        mlp_norm_w[layer] = tensorCreate(
            mlp_norm_w_shape, 1, meta_.dtype, device_, device_id_);
        mlp_gate_w[layer] = tensorCreate(
            mlp_gate_w_shape, 2, meta_.dtype, device_, device_id_);
        mlp_up_w[layer] = tensorCreate(
            mlp_up_w_shape, 2, meta_.dtype, device_, device_id_);
        mlp_down_w[layer] = tensorCreate(
            mlp_down_w_shape, 2, meta_.dtype, device_, device_id_);
    }

    weights_.in_embed = in_embed;
    weights_.out_embed = out_embed;
    weights_.out_norm_w = out_norm_w;
    weights_.attn_norm_w = attn_norm_w.data();
    weights_.attn_q_w = attn_q_w.data();
    weights_.attn_q_b = attn_q_b.data();
    weights_.attn_k_w = attn_k_w.data();
    weights_.attn_k_b = attn_k_b.data();
    weights_.attn_v_w = attn_v_w.data();
    weights_.attn_v_b = attn_v_b.data();
    weights_.attn_o_w = attn_o_w.data();
    weights_.mlp_norm_w = mlp_norm_w.data();
    weights_.mlp_gate_w = mlp_gate_w.data();
    weights_.mlp_up_w = mlp_up_w.data();
    weights_.mlp_down_w = mlp_down_w.data();
}

Qwen2Model::~Qwen2Model() {
    tensorDestroy(weights_.in_embed);
    tensorDestroy(weights_.out_embed);
    tensorDestroy(weights_.out_norm_w);

    for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
        tensorDestroy(k_cache_[layer]);
        tensorDestroy(v_cache_[layer]);
        tensorDestroy(weights_.attn_norm_w[layer]);
        tensorDestroy(weights_.attn_q_w[layer]);
        tensorDestroy(weights_.attn_q_b[layer]);
        tensorDestroy(weights_.attn_k_w[layer]);
        tensorDestroy(weights_.attn_k_b[layer]);
        tensorDestroy(weights_.attn_v_w[layer]);
        tensorDestroy(weights_.attn_v_b[layer]);
        tensorDestroy(weights_.attn_o_w[layer]);
        tensorDestroy(weights_.mlp_norm_w[layer]);
        tensorDestroy(weights_.mlp_gate_w[layer]);
        tensorDestroy(weights_.mlp_up_w[layer]);
        tensorDestroy(weights_.mlp_down_w[layer]);
    }
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &weights_;
}

void Qwen2Model::reset_cache() {
    cache_len_ = 0;
}

int64_t Qwen2Model::step(const int64_t *input_tokens, size_t ntoken) {
    if (input_tokens == nullptr || ntoken == 0) {
        throw std::invalid_argument("Qwen2Model::step requires at least one input token");
    }
    llaisys::core::context().setDevice(device_, device_id_);
    auto &runtime = llaisys::core::context().runtime();
    size_t past_len = cache_len_;
    size_t new_cache_len = past_len + ntoken;
    if (new_cache_len > meta_.maxseq) {
        throw std::invalid_argument(
            "Qwen2Model::step input exceeds KV Cache capacity");
    }
    if (meta_.nkvh == 0 || meta_.nh % meta_.nkvh != 0
        || meta_.hs != meta_.nh * meta_.dh) {
        throw std::invalid_argument(
            "Qwen2Model::step received inconsistent model dimensions");
    }

    size_t token_shape[] = {ntoken};
    size_t hidden_shape[] = {ntoken, meta_.hs};
    size_t q_2d_shape[] = {ntoken, meta_.nh * meta_.dh};
    size_t kv_2d_shape[] = {ntoken, meta_.nkvh * meta_.dh};
    size_t q_shape[] = {ntoken, meta_.nh, meta_.dh};
    size_t kv_shape[] = {ntoken, meta_.nkvh, meta_.dh};
    size_t mlp_shape[] = {ntoken, meta_.di};
    size_t logits_shape[] = {1, meta_.voc};
    size_t logits_1d_shape[] = {meta_.voc};
    size_t result_shape[] = {1};

    llaisysTensor_t input_ids = tensorCreate(
        token_shape, 1, LLAISYS_DTYPE_I64, device_, device_id_);
    tensorLoad(input_ids, input_tokens);

    llaisysTensor_t hidden = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysEmbedding(hidden, input_ids, weights_.in_embed);
    tensorDestroy(input_ids);

    std::vector<int64_t> positions(ntoken);
    // 利用KV Cache
    std::iota(
        positions.begin(),
        positions.end(),
        static_cast<int64_t>(past_len));
    llaisysTensor_t position_ids = tensorCreate(
        token_shape, 1, LLAISYS_DTYPE_I64, device_, device_id_);
    tensorLoad(position_ids, positions.data());

    const float attention_scale
        = 1.0f / std::sqrt(static_cast<float>(meta_.dh));

    llaisysTensor_t hidden_next = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t attn_norm = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t q_2d = tensorCreate(
        q_2d_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t k_2d = tensorCreate(
        kv_2d_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t v_2d = tensorCreate(
        kv_2d_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t q = tensorView(q_2d, q_shape, 3);
    llaisysTensor_t k = tensorView(k_2d, kv_shape, 3);
    llaisysTensor_t v = tensorView(v_2d, kv_shape, 3);
    llaisysTensor_t q_rope = tensorCreate(
        q_shape, 3, meta_.dtype, device_, device_id_);
    llaisysTensor_t k_rope = tensorCreate(
        kv_shape, 3, meta_.dtype, device_, device_id_);
    llaisysTensor_t attention_3d = tensorCreate(
        q_shape, 3, meta_.dtype, device_, device_id_);
    llaisysTensor_t attention_2d = tensorView(
        attention_3d, hidden_shape, 2);
    llaisysTensor_t attention_out = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t hidden_after_attn = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t mlp_norm = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t gate = tensorCreate(
        mlp_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t up = tensorCreate(
        mlp_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t activated = tensorCreate(
        mlp_shape, 2, meta_.dtype, device_, device_id_);
    llaisysTensor_t down = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);

    for (size_t layer = 0; layer < meta_.nlayer; ++layer) {
        llaisysRmsNorm(attn_norm, hidden,
                       weights_.attn_norm_w[layer], meta_.epsilon);

        llaisysLinear(q_2d, attn_norm,
                      weights_.attn_q_w[layer], weights_.attn_q_b[layer]);
        llaisysLinear(k_2d, attn_norm,
                      weights_.attn_k_w[layer], weights_.attn_k_b[layer]);
        llaisysLinear(v_2d, attn_norm,
                      weights_.attn_v_w[layer], weights_.attn_v_b[layer]);

        llaisysROPE(q_rope, q, position_ids, meta_.theta);
        llaisysROPE(k_rope, k, position_ids, meta_.theta);

        llaisysTensor_t k_cache_write = tensorSlice(
            k_cache_[layer], 0, past_len, new_cache_len);
        llaisysTensor_t v_cache_write = tensorSlice(
            v_cache_[layer], 0, past_len, new_cache_len);

        const size_t kv_write_bytes =
            ntoken * meta_.nkvh * meta_.dh *
            llaisys::utils::dsize(meta_.dtype);

        // 重点：k_rope/v 与 KV cache 在 CUDA 前向中都位于 device memory。
        // tensorLoad 的语义是从 host 写入 tensor（H2D），不能用于这里；否则
        // 会把 device pointer 当成 host pointer。D2D copy 放到同一 stream 后，
        // 后续 SelfAttention 会自然等待本次 cache 写入完成。
        runtime.api()->memcpy_async(
            tensorGetData(k_cache_write), tensorGetData(k_rope),
            kv_write_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());
        runtime.api()->memcpy_async(
            tensorGetData(v_cache_write), tensorGetData(v),
            kv_write_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());

        llaisysTensor_t k_cache_used = tensorSlice(
            k_cache_[layer], 0, 0, new_cache_len);
        llaisysTensor_t v_cache_used = tensorSlice(
            v_cache_[layer], 0, 0, new_cache_len);

        llaisysSelfAttention(
            attention_3d, q_rope, k_cache_used, v_cache_used,
            attention_scale);

        llaisysLinear(attention_out, attention_2d,
                      weights_.attn_o_w[layer], nullptr);

        // resNet Add
        llaisysAdd(hidden_after_attn, hidden, attention_out);

        llaisysRmsNorm(mlp_norm, hidden_after_attn,
                       weights_.mlp_norm_w[layer], meta_.epsilon);

        llaisysLinear(gate, mlp_norm,
                      weights_.mlp_gate_w[layer], nullptr);
        llaisysLinear(up, mlp_norm,
                      weights_.mlp_up_w[layer], nullptr);
        llaisysSwiGLU(activated, gate, up);
        llaisysLinear(down, activated,
                      weights_.mlp_down_w[layer], nullptr);

        // resNet Add
        llaisysAdd(hidden_next, hidden_after_attn, down);

        tensorDestroy(k_cache_write);
        tensorDestroy(v_cache_write);
        tensorDestroy(k_cache_used);
        tensorDestroy(v_cache_used);

        llaisysTensor_t previous_hidden = hidden;
        hidden = hidden_next;
        hidden_next = previous_hidden;
    }
    cache_len_ = new_cache_len;

    tensorDestroy(hidden_next);
    tensorDestroy(attn_norm);
    tensorDestroy(q);
    tensorDestroy(k);
    tensorDestroy(v);
    tensorDestroy(q_2d);
    tensorDestroy(k_2d);
    tensorDestroy(v_2d);
    tensorDestroy(q_rope);
    tensorDestroy(k_rope);
    tensorDestroy(attention_2d);
    tensorDestroy(attention_3d);
    tensorDestroy(attention_out);
    tensorDestroy(hidden_after_attn);
    tensorDestroy(mlp_norm);
    tensorDestroy(gate);
    tensorDestroy(up);
    tensorDestroy(activated);
    tensorDestroy(down);

    llaisysTensor_t final_norm = tensorCreate(
        hidden_shape, 2, meta_.dtype, device_, device_id_);
    llaisysRmsNorm(
        final_norm, hidden, weights_.out_norm_w, meta_.epsilon);

    llaisysTensor_t last_hidden = tensorSlice(
        final_norm, 0, ntoken - 1, ntoken);
    llaisysTensor_t logits = tensorCreate(
        logits_shape, 2, meta_.dtype, device_, device_id_);
    llaisysLinear(logits, last_hidden, weights_.out_embed, nullptr);
    llaisysTensor_t logits_1d = tensorView(logits, logits_1d_shape, 1);

    llaisysTensor_t max_index = tensorCreate(
        result_shape, 1, LLAISYS_DTYPE_I64, device_, device_id_);
    llaisysTensor_t max_value = tensorCreate(
        result_shape, 1, meta_.dtype, device_, device_id_);
    llaisysArgmax(max_index, max_value, logits_1d);

    int64_t next_token = 0;
    // 重点：CUDA 下 tensorGetData(max_index) 返回的是 device pointer，CPU
    // 不能直接解引用。先等待同一 stream 上的 Argmax 完成，再显式 D2H
    // 拷贝一个 int64_t；CPU runtime 会把这两步退化为 no-op + memcpy。
    runtime.synchronize();
    runtime.api()->memcpy_sync(
        &next_token, tensorGetData(max_index), sizeof(next_token),
        LLAISYS_MEMCPY_D2H);

    tensorDestroy(position_ids);
    tensorDestroy(hidden);
    tensorDestroy(last_hidden);
    tensorDestroy(final_norm);
    tensorDestroy(logits_1d);
    tensorDestroy(logits);
    tensorDestroy(max_index);
    tensorDestroy(max_value);

    return next_token;
}

} // namespace llaisys::models
