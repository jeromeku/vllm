# ruff: noqa
import os
from pathlib import Path
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
#os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/home/jeromeku/vllm/vllm_logging_config.json"

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
import vllm

from vllm.logger import init_logger
VLLM_ROOT = Path(vllm.__file__).parent

logger = init_logger(__file__)
logger.debug("Hello")

MODEL_PATH = "assets/qwen3_moe"
prompts = [
    "Hello, my name is",
]
compilation_config = CompilationConfig(cudagraph_capture_sizes=[len(prompts)])
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    # Create an LLM.
    llm = LLM(model=MODEL_PATH, compilation_config=compilation_config)
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
    pass
    # main()
