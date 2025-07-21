# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path
#os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
#os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/home/jeromeku/vllm/vllm_logging_config.json"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#/home/jeromeku/vllm/vllm/v1/worker/gpu_model_runner.py load_model => base_loader.load_model
#/home/jeromeku/vllm/vllm/model_executor/model_loader/utils.py:initialize_model
# uniprocexecutor => gpu_worker => gpu_model_runner => model
# default_loader.load_weights => model.load_weights => AutoWeightsLoader.load_weights
# directly create model
# /home/jeromeku/vllm/vllm/model_executor/model_loader/weight_utils.py safetensor load and fastsafetensor load
#/home/jeromeku/vllm/vllm/model_executor/model_loader/default_loader.py _prepare_weights downloads weights
# fused qkv loading: /home/jeromeku/vllm/vllm/model_executor/layers/linear.py weight_loader
# fused moe weight loading: /home/jeromeku/vllm/vllm/model_executor/layers/fused_moe/layer.py

# breakpoint at /home/jeromeku/vllm/vllm/v1/worker/gpu_model_runner.py execute_model / load_model
def main():
    from vllm.config import CompilationConfig
    compilation_config = CompilationConfig(cudagraph_capture_sizes=[1])

    # Create an LLM.
    llm = LLM(model="facebook/opt-125m", compilation_config=compilation_config)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
