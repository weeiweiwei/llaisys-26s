#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    auto &strides = _meta.strides;
    auto &shapes = _meta.shape;
    if (shapes.empty()) {
        return true;
    }
    if (strides.back() != 1) {
        return false;
    }

    for (size_t i = shapes.size() - 1; i > 0; i--) {
        auto product = strides[i] * shapes[i];
        if (product != strides[i - 1]) {
            return false;
        }
    }

    return true;
}
// 2,3,4 --- 4,3,2

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    auto &shape = _meta.shape;
    auto &strides = _meta.strides;
    std::vector<size_t> new_shape(shape.size());
    std::vector<ptrdiff_t> new_stride(strides.size());

    for (size_t i = 0; i < order.size(); i++) {
        new_shape[i] = shape[order[i]];
        new_stride[i] = strides[order[i]];
    }
    TensorMeta new_meta{_meta.dtype, std::move(new_shape), std::move(new_stride)};

    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    if (!isContiguous()) {
        throw std::runtime_error("view: tensor must be contiguous");
    }

    size_t new_numel = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::runtime_error("view: zero-sized dimension is not supported");
        }

        new_numel *= dim;
    }

    if (new_numel != numel()) {
        throw std::runtime_error("view: element count does not match");
    }

    std::vector<size_t> new_shape = shape;
    std::vector<ptrdiff_t> new_strides(shape.size(), 1);

    // 进入要求是至少为2，如果为1的话，就使用前面的默认stride = 1，避免size=1时的一些边界情况
    for (size_t i = new_strides.size(); i > 1; --i) {
        new_strides[i - 2] = new_strides[i - 1] * new_shape[i - 1];
    }

    TensorMeta new_meta{_meta.dtype, std::move(new_shape), std::move(new_strides)};

    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}

// shape=(3, 4, 5)--->slice(1, 1, 3)
// 新张量的逻辑形状是 (3, 2, 5) ,这样第1维的坐标索引 可以被new_offset - offset完美补偿
// 核心是底层存储不变，只改变索引方式，这里是线性偏移
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {

    auto &shape = _meta.shape;
    auto &strides = _meta.strides;
    std::vector<size_t> new_shape(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        new_shape[i] = shape[i];
    }

    new_shape[dim] = end - start;
    size_t new_offset = _offset + start * strides[dim]; // 新张量能够对应上原本的情况
    TensorMeta new_meta{_meta.dtype, std::move(new_shape), strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}
// 这个目前是接收CPU Tensor，放到GPU Tensor对应的内存去--由GPU Tensor调用是合法的
void Tensor::load(const void *src_) {
    // 考虑是否修改大小表示方式
    size_t Tensor_size = numel() * elementSize();
    auto device_ptr = this->_storage->memory() + this->_offset;

    core::context().runtime().api()->memcpy_sync(device_ptr, src_, Tensor_size, LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
