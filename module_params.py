import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from safetensors.torch import safe_open
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
import torch
from vllm.config import VllmConfig, ModelConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.utils import get_ip, get_open_port, get_distributed_init_method
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    ensure_model_parallel_initialized,
)
model_config = ModelConfig(model="model_cache", hf_config_path="model_cache")
vllm_config = VllmConfig(model_config=model_config)

init_method = get_distributed_init_method(get_ip(), get_open_port())

with set_current_vllm_config(vllm_config):

    # Single-GPU example – adjust as needed.
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=init_method,
        local_rank=0,
        backend="nccl",
    )

    # If you want TP/PP groups even in single-GPU mode (harmless when size==1):
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    # parallel_state.initialize_model_parallel()

    # model_file = "model_cache/model.safetensors"

    # with safe_open(model_file, framework="pt") as f:
    #     for name in f.keys():
    #         if "expert" in name:
    #             continue
    #         param = f.get_tensor(name)
    #         print(f"{name}:{param.data.shape}")

    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(vllm_config=vllm_config)