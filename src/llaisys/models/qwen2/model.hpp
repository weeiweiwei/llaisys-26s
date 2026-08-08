#pragma once
#include "llaisys/models/qwen2.h"
#include <vector>
namespace llaisys::models {

class Qwen2Model {
public:
    // 参数中后面存在的原因是因为提供的C API中存在这些参数
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, const int *device_ids, int ndevice);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights();
    void reset_cache();
    // 这里暂时还不清楚ntoken和tokens的独立的对应分别是什么，如何结合起来Infer
    int64_t step(const int64_t *input_tokens, size_t ntoken);

private:
    LlaisysQwen2Meta meta_;
    LlaisysQwen2Weights weights_{};
    llaisysDeviceType_t device_;
    int device_id_;

    // 这里目前使用一个连续的Tensor表示KV Cache的存储
    // 由于TensorLoad API这个功能不太好用 所以一体的不是那么好用
    std::vector<llaisysTensor_t> k_cache_;
    std::vector<llaisysTensor_t> v_cache_;
    size_t cache_len_ = 0;

    llaisysTensor_t in_embed;
    llaisysTensor_t out_embed;
    llaisysTensor_t out_norm_w;
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w; // a.k.a. post_attention_layernorm.weight
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
};

} // namespace llaisys::models
