# ruff: noqa
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import importlib
import sys
from types import ModuleType
import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel, LoadConfig, LoadFormat
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeConfig
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_CACHE_DIR = "model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

if not os.path.exists(os.path.join(MODEL_CACHE_DIR, "model.safetensors")):
    model_config = Qwen3MoeConfig(num_hidden_layers=1)
    with set_default_torch_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM(model_config)
    
    model.save_pretrained(MODEL_CACHE_DIR)
    print(f"Saved model to {MODEL_CACHE_DIR}")
if not os.path.exists(os.path.join(MODEL_CACHE_DIR, "vocab.json")):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(MODEL_CACHE_DIR)


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

def get_module_root(module: str | ModuleType):
    from pathlib import Path

    if isinstance(module, str):
        try:
            module = importlib.import_module(module)
        except ImportError as e:
            raise ImportError(f"Could not import module: {module}") from e

    if not isinstance(module, ModuleType):
        raise TypeError("Input must be a module or a string representing a module.")

    if not hasattr(module, "__file__") or module.__file__ is None:
        raise AttributeError(
            f"The module '{module.__name__}' is a built-in module or a namespace package and has no file path."
        )

    return Path(module.__file__).parent.resolve().as_posix()
#/home/jeromeku/vllm/vllm/v1/worker/gpu_model_runner.py load_model => base_loader.load_model
#/home/jeromeku/vllm/vllm/model_executor/model_loader/utils.py:initialize_model
# uniprocexecutor => gpu_worker => gpu_model_runner => model
# default_loader.load_weights => model.load_weights => AutoWeightsLoader.load_weights
# directly create model
# /home/jeromeku/vllm/vllm/model_executor/model_loader/weight_utils.py safetensor load and fastsafetensor load
#/home/jeromeku/vllm/vllm/model_executor/model_loader/default_loader.py _prepare_weights downloads weights
# fused qkv loading: /home/jeromeku/vllm/vllm/model_executor/layers/linear.py weight_loader
# fused moe weight loading: /home/jeromeku/vllm/vllm/model_executor/layers/fused_moe/layer.py
def main():
    compile_cfg = None  # CompilationConfig(level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=[len(prompts)])
    # Create an LLM.
    from viztracer import VizTracer

    vllm_root = get_module_root("vllm")
    tracer = VizTracer(
        include_files=[vllm_root],
        ignore_c_function=False,
        ignore_frozen=True,
        log_func_args=True,
        log_func_retval=True,
    )

    tracer.output_file = "model.trace.json"

    with tracer:
        llm = LLM(
            model=MODEL_CACHE_DIR, #MODEL_ID,
            compilation_config=compile_cfg,
            # load_format=LoadFormat.DUMMY,
            hf_overrides={"num_hidden_layers": 1},
            # download_dir=MODEL_CACHE_DIR
        )

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # outputs = llm.generate(prompts, sampling_params)
    # # Print the outputs.
    # print("\nGenerated Outputs:\n" + "-" * 60)
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt:    {prompt!r}")
    #     print(f"Output:    {generated_text!r}")
    #     print("-" * 60)


if __name__ == "__main__":
    main()
