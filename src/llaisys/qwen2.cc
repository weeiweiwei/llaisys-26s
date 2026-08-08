#include "llaisys/models/qwen2.h"
#include "models/qwen2/model.hpp"
// Ipath是都是优先于当前文件所在文件夹开始的
#include <memory>

// 暂时不是很明确这么设置的好处与 泛用性之类的意义
struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> impl;
};

__C struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {

    if (meta == nullptr || device_ids == nullptr || ndevice == 0) {
        return nullptr;
    }

    auto *handle = new LlaisysQwen2Model;
    handle->impl = std::make_unique<llaisys::models::Qwen2Model>(*meta, device, device_ids, ndevice);

    return handle;
}

__C void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model; // model本质是一个指向Qwen2Model对象的一个unique_ptr
}

__C struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->impl->weights();
}

__C void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model) {
    if (model != nullptr) {
        model->impl->reset_cache();
    }
}

__C int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (model == nullptr || token_ids == nullptr || ntoken == 0) {
        return -1;
    }

    return model->impl->step(token_ids, ntoken);
}
