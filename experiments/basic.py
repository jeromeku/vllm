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
from vllm.logger import enable_trace_function_call
import torch
from tracer import enable_trace

VLLM_ROOT = Path(vllm.__file__).parent
TRACE_FILE = "vllm.trace.txt"
enable_trace_function_call(TRACE_FILE, root_dir=VLLM_ROOT.resolve().as_posix())

# enable_trace(TRACE_FILE, root_dir=VLLM_ROOT.resolve().as_posix())
MODEL_PATH = "assets/qwen3_moe"
prompts = [
    "Hello, my name is",
]
compilation_config = CompilationConfig(cudagraph_capture_sizes=[len(prompts)])
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def add_request(request_id: str, llm: LLM, sampling_params: SamplingParams, prompt_len: int = 10):
    prompt_token_ids = torch.randint(
                llm.get_tokenizer().vocab_size, size=(prompt_len,)
            ).tolist()

    llm.llm_engine.add_request(
        request_id=request_id,
        prompt={"prompt_token_ids": prompt_token_ids},
        params=sampling_params,
    )


def abort_request(llm: LLM, request_id: str | list[str]):
    if isinstance(request_id, str):
        request_id = [request_id]
    for req_id in request_id:
        llm.llm_engine.abort_request(req_id)

def main():
    # Create an LLM.
    llm = LLM(model=MODEL_PATH, compilation_config=compilation_config)
    request_id = "seq1"
    add_request(request_id, llm, sampling_params=sampling_params)

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
