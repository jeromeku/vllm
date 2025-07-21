# ruff: noqa
import os
from pathlib import Path
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
#os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/home/jeromeku/vllm/vllm_logging_config.json"

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.profiler.layerwise_profile import layerwise_profile
from vllm.logger import enable_trace_function_call
import torch
from tracer import enable_trace
import vllm
from numpy import random
VLLM_ROOT = Path(vllm.__file__).parent
TRACE_FILE = "vllm.trace.txt"
#enable_trace_function_call(TRACE_FILE, root_dir=VLLM_ROOT.resolve().as_posix())

MODEL_PATH = "assets/qwen3_moe"
prompts = [
    "Hello, my name is",
]
compilation_config = CompilationConfig(cudagraph_capture_sizes=[len(prompts)])
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def _add_requests(*, llm: LLM, sampling_params: SamplingParams, request_ids: list[str], prompt_len: int = 10):

    for req_id in request_ids:
        prompt_token_ids = torch.randint(
                    llm.get_tokenizer().vocab_size, size=(prompt_len,)
                ).tolist()
        llm.llm_engine.add_request(
            request_id=req_id,
            prompt={"prompt_token_ids": prompt_token_ids},
            params=sampling_params,
        )
    return request_ids

def _abort_requests(*, llm: LLM, request_ids: list[str]):
    for req_id in request_ids:
        llm.llm_engine.abort_request(req_id)

def main():
    
    from functools import partial
    import numpy as np
    # Create an LLM.
    llm = LLM(model=MODEL_PATH, compilation_config=compilation_config)
    num_requests = 5
    prompt_len = 10
    warmup_requests = 5
    request_ids = torch.arange(num_requests + warmup_requests)
    warmup_ids, test_ids = [id.tolist() for id in request_ids.split([warmup_requests, num_requests])]
    warmup_ids, test_ids = list(map(str, warmup_ids)), list(map(str, test_ids))
    add_requests = partial(_add_requests, llm=llm, sampling_params=sampling_params)
    abort_requests = partial(_abort_requests, llm=llm)

    add_requests(request_ids=warmup_ids, prompt_len=prompt_len)
    llm.llm_engine.step()  # Prefill
    llm.llm_engine.step()  # Decode
    abort_requests(request_ids=warmup_ids)

    print("Profile run ...")
    add_requests(request_ids=test_ids, prompt_len=prompt_len)
    
    with layerwise_profile() as prefill_prof:
        llm.llm_engine.step()  # First step is prefill

    prefill_results = prefill_prof.results

    LINE_WIDTH = 80
    print("=" * LINE_WIDTH)
    print(f"= Prefill Model Table (prompt_len={prompt_len}, batch_size={num_requests})")
    print("=" * LINE_WIDTH)
    print()
    prefill_results.print_model_table()
    print()
    print("=" * LINE_WIDTH)
    print(f"= Prefill Summary Table (prompt_len={prompt_len}, batch_size={num_requests})")
    print("=" * LINE_WIDTH)
    print()
    prefill_results.print_summary_table()

    # Save json
    cuda_devices = [
        torch.cuda.get_device_properties(dev_idx)
        for dev_idx in range(torch.cuda.device_count())
    ]

    json_dict = {
        "context": {
            "torch_version": f"{torch.__version__}",
            "torch_cuda_version": f"{torch.version.cuda}",
            "cuda_devices": f"{cuda_devices}",
        },
        "prefill": prefill_results.convert_stats_to_dict(),
    }

    import json
    # Add .json to json_output filename if it doesn't exist already.
    json_output_file = "layerwise_dump.json"
    with open(json_output_file, "w+") as f:
        json.dump(json_dict, f, indent=2)

    chrome_trace_folder = "layerwise_trace"
    os.makedirs(chrome_trace_folder, exist_ok=True)
    prefill_prof.profiler.export_chrome_trace(
        chrome_trace_folder + "/prefill.json"
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
