"""
experimental support for tensor-parallel inference with torchrun,
see https://github.com/vllm-project/vllm/issues/11400 for
the motivation and use case for this example.
run the script with `torchrun --nproc-per-node=2 torchrun_example.py`,
the argument 2 should match the `tensor_parallel_size` below.
see `tests/distributed/test_torchrun_example.py` for the unit test.
"""
import time
import re

from vllm import LLM, SamplingParams
import os
import torch
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.v1.engine.core_client import InprocClient

from vllm.distributed.parallel_state import get_world_group
from transformers import AutoModelForCausalLM
import torch.distributed as dist
from safetensors.torch import load_file
from safetensors import safe_open

def print_parent_classes(obj):
    parents = type(obj).__mro__
    self = parents[0]
    print(f"{self.__name__}: {self}, parents:")
    for cls in parents[1:-1]:
        print(f" -> {cls.__name__}: {cls}")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def main():
    MODEL_ID = "facebook/opt-125m" 
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    hf_sd = hf_model.state_dict()
    if not os.path.exists(HF_SAVE_PATH):
        torch.save(hf_sd, HF_SAVE_PATH)


    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=world_size,
        distributed_executor_backend="external_launcher",
        # load_format="dummy"
    )
    is_v1_engine = hasattr(llm.llm_engine, "engine_core")
    print(f"{is_v1_engine=}")

    model_runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner

    # print(print_parent_classes(llm.llm_engine.model_executor))
    # print(print_parent_classes(llm.llm_engine.model_executor.driver_worker))
    # print(print_parent_classes(llm.llm_engine.model_executor.driver_worker.worker))
    # print_parent_classes(model_runner)
    # print_parent_classes(model_runner.model)
    engine_core: InprocClient = llm.llm_engine.engine_core

        # print_parent_classes(engine_core)

    if not os.path.exists(SHARDED_SAVE_PATH):
        os.makedirs(SHARDED_SAVE_PATH, exist_ok=True)
        engine_core.save_sharded_state(path=SHARDED_SAVE_PATH)

    model: OPTForCausalLM = model_runner.model

    model.load_weights(hf_sd.items())
    dist.barrier()

    time.sleep(rank)
    print(f"rank{rank}: Loaded model")
    time.sleep(rank * 2)
    layer_regex = re.compile(r"layers[.]([0-9]+)")

    if rank == 0:
        print(f"rank{rank} State Dict:")
        for name, param in model.state_dict().items():
            m = layer_regex.search(name)
            should_print = m is None or int(m.groups()[0]) == 0
            if should_print:
                print(f" - {name}: {param.shape}")

    # import re
    # layer_regex = re.compile(r"layers[.]([0-9]+)")
    if rank == 0:
        print("HF State Dict:")
        for name, param in hf_sd.items():
            m = layer_regex.search(name)
            should_print = m is None or int(m.groups()[0]) == 0
            if should_print:
                print(f" - {name}: {param.shape}")

    # shards = list(os.listdir(SHARDED_SAVE_PATH))
    # print("Sharded state dict:")
    # for shard in shards:
    #     with safe_open(os.path.join(SHARDED_SAVE_PATH, shard), framework='pt') as f:
    #         keys = f.keys()
    #         for k in keys:
    #             m = layer_regex.search(k)
    #             should_print = m is None or int(m.groups()[0]) == 0
    #             if should_print:                        
    #                 print(f" - {k}: {f.get_tensor(k).shape}")
    
if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    SHARDED_SAVE_PATH = f"opt_sharded-tp{world_size}"
    HF_SAVE_PATH = "opt_hf.pt"
    main()

