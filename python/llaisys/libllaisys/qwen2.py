# 作为Python与C的接口文件  参考tensor下的接口实现情况
# 告诉Python C结构体的形式---C函数接收的参数----C函数的返回类型
# 基本文件结构与分工是：llaisys/libllaisys/qwen2.py作为接口文件
# llaisys/qwen2.py作为提供给上层用户使用的文件，保证用户能够以Python的形式调用C实现的文件
# 参考runtime得知 对于不默认支持的structure需要这里自定义使用

# 他为的是src/llaisys/下声明的c文件函数作接口，以便能够调用lib中同名的函数

# 链接Python ---- libllaisys.so
from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int,c_int64,c_float,Structure
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t

# Python程序需要处理Meta的内部数据，所以不能只传指针 Model Python只传递地址
# handle type 自定义的
llaisysQwen2Model_t = c_void_p
# Define the struct matching LlaisysQwen2Meta
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype",llaisysDataType_t),
        ("nlayer",c_size_t),
        ("hs",c_size_t),
        ("nh",c_size_t),
        ("nkvh",c_size_t),
        ("dh",c_size_t),
        ("di",c_size_t),
        ("maxseq",c_size_t),
        ("voc",c_size_t),
        ("epsilon",c_float),
        ("theta",c_float),
        ("end_token",c_int64),
    ]

# Define the struct matching LlaisysQwen2Weights
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed",llaisysTensor_t),
        ("out_embed",llaisysTensor_t),
        ("out_norm_w",llaisysTensor_t),
        ("attn_norm_w",POINTER(llaisysTensor_t)),
        ("attn_q_w",POINTER(llaisysTensor_t)),
        ("attn_q_b",POINTER(llaisysTensor_t)),
        ("attn_k_w",POINTER(llaisysTensor_t)),
        ("attn_k_b",POINTER(llaisysTensor_t)),
        ("attn_v_w",POINTER(llaisysTensor_t)),
        ("attn_v_b",POINTER(llaisysTensor_t)),
        ("attn_o_w",POINTER(llaisysTensor_t)),
        ("mlp_norm_w",POINTER(llaisysTensor_t)),
        ("mlp_gate_w",POINTER(llaisysTensor_t)),
        ("mlp_up_w",POINTER(llaisysTensor_t)),
        ("mlp_down_w",POINTER(llaisysTensor_t)),
    ]

#---------------------------------------------------------

def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        llaisysDeviceType_t,  # device
        POINTER(c_int),  # device_ids
        c_int,  # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t


    # Function: llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None


    # Function: llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelResetCache.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetCache.restype = None

    # Function: llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,# struct LlaisysQwen2Model *
        POINTER(c_int64),# token_ids
        c_size_t # ntoken
        ]

    lib.llaisysQwen2ModelInfer.restype = c_int64
