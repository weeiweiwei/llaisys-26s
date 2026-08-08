#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::abort();                                              \
        }                                                              \
    } while (0)

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 这个生命周期能覆盖吗
llaisysStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    if (size > 0) {
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// 通过这个申请的内存不能很大 除非你使用CPU的malloc分配
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    if (size > 0) {
        // 用这个的话cudaMallocHost(); 默认申请page-locked/pinned
        // Pinned host memory allows H2D/D2H copies to run asynchronously.
        CUDA_CHECK(cudaMallocHost(&ptr, size));
        return ptr;
    } else {
        return nullptr;
    }
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        // Memory from cudaMallocHost must be paired with cudaFreeHost.
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, static_cast<cudaMemcpyKind>(kind)));
}

// 为什么reinterprate报错呢 这个static的行吗？
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, static_cast<cudaMemcpyKind>(kind), reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
