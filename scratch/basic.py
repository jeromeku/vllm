# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import types
from pathlib import Path
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# from transformers import AutoModelForCausalLM
from viztracer import VizTracer

# import vllm
from vllm import LLM #, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel


def get_module_root(module: types.ModuleType | str) -> str:
    """
    Gets the root directory of a Python module.

    Args:
        module: The module object or the name of the module as a string.

    Returns:
        The root directory of the module as a POSIX path string.

    Raises:
        TypeError: If the input is not a module or a string.
        ImportError: If the module string cannot be imported.
        AttributeError: If the module does not have a __file__ attribute.
    """
    if isinstance(module, str):
        try:
            module = importlib.import_module(module)
        except ImportError as e:
            raise ImportError(f"Could not import module: {module}") from e

    if not isinstance(module, types.ModuleType):
        raise TypeError("Input must be a module or a string representing a module.")

    if not hasattr(module, "__file__") or module.__file__ is None:
        raise AttributeError(
            f"The module '{module.__name__}' is a built-in module or a namespace package and has no file path."
        )

    return Path(module.__file__).parent.resolve().as_posix()


VLLM_ROOT = get_module_root("vllm")
DYNAMO_ROOT = get_module_root("torch._dynamo")
INDUCTOR_ROOT = get_module_root("torch._inductor")

print(INDUCTOR_ROOT)
print(VLLM_ROOT)

MODEL_ID = "qwen/qwen3-0.6B"
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# # Sample prompts.
# prompts = [
#     "Hello, my name is",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

SHOULD_TRACE = False
def main():
    # Create an LLM.
    compile_config = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        debug_dump_path="./compile/debug",
        # cache_dir="./compile/cache",
        cudagraph_capture_sizes=[1, 2],
    )
    MODEL_ID = "facebook/opt-125m" #qwen/qwen3-0.6B"
    if SHOULD_TRACE:
        tracer = VizTracer(
            include_files=[VLLM_ROOT, INDUCTOR_ROOT],
            ignore_c_function=True,
            ignore_frozen=True,
            log_func_args=True,
            log_func_retval=True,
            pid_suffix=True,
        )
    
        tracer.output_file = "compile/compile.trace.json"
    else:
        from contextlib import nullcontext
        tracer = nullcontext()
    
    with tracer:
        llm = LLM(model=MODEL_ID, compilation_config=compile_config)


if __name__ == "__main__":
    main()

#     # Generate texts from the prompts.
#     # The output is a list of RequestOutput objects
#     # that contain the prompt, generated text, and other information.
#     # outputs = llm.generate(prompts, sampling_params)
#     # # Print the outputs.
#     # print("\nGenerated Outputs:\n" + "-" * 60)
#     # for output in outputs:
#     #     prompt = output.prompt
#     #     generated_text = output.outputs[0].text
#     #     print(f"Prompt:    {prompt!r}")
#     #     print(f"Output:    {generated_text!r}")
#     #     print("-" * 60)

