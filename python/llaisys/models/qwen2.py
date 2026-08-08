from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType,LlaisysQwen2Meta
from ctypes import byref, c_int, c_void_p,c_int64

from pathlib import Path
import safetensors
import json


class Qwen2:
   # 构造需要的输入参数
    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
        max_seq_len: int = 2048,
    ):
        model_path = Path(model_path)
        with open(model_path/"config.json",encoding="utf-8") as f:
            config = json.load(f);

        model_max_seq_len = config["max_position_embeddings"]
        if max_seq_len <= 0 or max_seq_len > model_max_seq_len:
            raise ValueError(
                f"max_seq_len must be in [1, {model_max_seq_len}]")
    
        self.meta_ = LlaisysQwen2Meta(
        dtype=DataType.BF16,
        nlayer=config["num_hidden_layers"],
        hs=config["hidden_size"],
        nh=config["num_attention_heads"],
        nkvh=config["num_key_value_heads"],
        dh=config["hidden_size"] // config["num_attention_heads"],
        di=config["intermediate_size"],
        maxseq=max_seq_len,
        voc=config["vocab_size"],
        epsilon=config["rms_norm_eps"],
        theta=config["rope_theta"],
        end_token=config["eos_token_id"],
        )

        # (c_int * 1)表示 长度为 1 的 C int 数组类型
        device_ids = (c_int * 1)(0)

        
        # self.meta_                     meta
        # byref(self.meta_)              &meta
        # POINTER(LlaisysQwen2Meta)      LlaisysQwen2Meta *

        self.model_ = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta_),# byref 是 ctypes 提供的“取得对象地址”
            device,
            device_ids,
            1,
        )
        if not self.model_:
            raise RuntimeError("Failed to Create Qwen2 Model!")


        self.weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(
            self.model_
        )
        if not self.weights_ptr:
            raise RuntimeError("Failed to Get Qwen2 Weights!")
        self.weights_ = self.weights_ptr.contents


        # 这里有些利用到的方法和函数 还不太了解具体情况
        loaded = 0
        for file in sorted(model_path.glob("*.safetensors")):
            # data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            # 这部分后续去仔细了解一下
            # 这里有个问题 他pytorch返回的tensor 从文件中得到的是什么结构的呢
            data_ = safetensors.safe_open(file, framework="pt", device="cpu") # 以暂时支持BF16
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_)
                handle = self.weight_loader_aux(name_)
                LIB_LLAISYS.tensorLoad(handle,c_void_p(tensor.data_ptr()),)

                loaded += 1
                print(f"Load {name_}:{tensor.shape}")
        print(f"Successfully load {loaded} Tensor")

# pt下的safetensor的使用，get_tensor返回的是一个torch.Tensor类型 需要专门使用对应的方法 
# 原本的numpy 框架只能适用于FP32 他返回的是一个array 这里才需要使用的是array对应的方法



    def __del__(self):# 释放指针和指针指向的内存空间
        if hasattr(self,"model_") and self.model_:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(
                self.model_
            )
            self.model_ = None

    def reset_cache(self):
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self.model_)

    def weight_loader_aux(self,name):
        # 需要对应safetensors中的真实权重名称
        if name == "model.embed_tokens.weight":
            return self.weights_.in_embed
        if name == "model.norm.weight":
            return self.weights_.out_norm_w
        if name == "lm_head.weight":
            return self.weights_.out_embed

        # 层间名称示意：model.layers.3.self_attn.q_proj.weight
        parts = name.split(".") # 以.为分割划为不同的parts
        layer = int(parts[2]) # 3
        suffix = ".".join(parts[3:]) # self_attn.q_proj.weight

        layer_fields = {
            "input_layernorm.weight": "attn_norm_w",

            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",

            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",

            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",

            "self_attn.o_proj.weight": "attn_o_w",

            "post_attention_layernorm.weight": "mlp_norm_w",

            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }
        field = layer_fields.get(suffix) #根据后缀 得到我们本地的变量对应名称
        if field is None:
            raise RuntimeError(f"Unknown Weight Name:{name}")
        return getattr(self.weights_,field)[layer] 
        # 得到对应vector<llaisysTensor_t>的指针数组后，根据layer写到对应的空间下

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        self.reset_cache()

        # 目前是一个朴素的输入prefill 后续decode的过程
        tokens = [int (token) for token in inputs]
        step_tokens = tokens

        for _ in range(max_new_tokens):
            if len(tokens) >= self.meta_.maxseq:
                break
            # 声明该变量是一个tokens长度的int64_t的数组类型，因为C函数需要的是一个int64_t的指针
            # 数组名是满足的，
            # *tokens 是 Python 的序列解包操作，把列表中的元素展开为构造参数。如果有需求才用
            tokens_array = (c_int64 * len(step_tokens))(*step_tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_, tokens_array, len(step_tokens))
            if next_token < 0:
                raise RuntimeError("Qwen2 inference failed")

            tokens.append(next_token)
            if next_token == self.meta_.end_token:
                break
            step_tokens = [next_token]

        return tokens
