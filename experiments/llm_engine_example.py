

import argparse
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
import random
import string
from vllm.profiler.layerwise_profile import layerwise_profile

def generate_alphanumeric_string(length):
    if length < 0:
        raise ValueError("Length must be non-negative")
    
    if length == 0:
        return ""
    
    # Define the character set: lowercase, uppercase letters, and digits
    characters = string.ascii_letters + string.digits
    
    # Generate random string
    result = ''.join(random.choice(characters) for _ in range(length))
    
    return result

def create_test_prompts() -> list[tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1),
        ),
        (
            "To be or not to be,",
            SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2),
        ),
        (
            "What is the meaning of life?",
            SamplingParams(n=2, temperature=0.8, top_p=0.95, frequency_penalty=0.1),
        ),
    ]


def process_requests(engine: LLMEngine, test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    print("-" * 50)
    assert len(test_prompts) == 1

    prefill_prof = layerwise_profile()
    decode_prof = layerwise_profile(1)

    step_idx = 0
    is_prefill = True
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1
        
        profiler = prefill_prof if is_prefill else decode_prof
        
        with profiler:
            request_outputs: list[RequestOutput] = engine.step()
        
        is_prefill = False
        
        
        for request_output in request_outputs:
            if request_output.finished:
                print(f"Engine stopped at step {step_idx}")
                print(request_output)
                print("-" * 50)
                #engine.abort_request(request_id)
        step_idx += 1

    return prefill_prof, decode_prof

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    prompt = "Hello my name is" #generate_alphanumeric_string(10)
    max_tokens = 2 #1 decode step
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=.8, top_p=.95)
    prefill_prof, decode_prof = process_requests(engine, [(prompt, sampling_params)])
    prefill_prof: layerwise_profile
    decode_prof: layerwise_profile
    print("Prefill results:")
    prefill_prof.results.print_model_table()
    print()
    prefill_prof.results.print_summary_table()
    print(" --- ")
    print("Decode results")
    decode_prof.results.print_model_table()
    print()
    decode_prof.results.print_summary_table()
    import json
    decode_results = decode_prof.results.convert_stats_to_dict()
    prefill_results = prefill_prof.results.convert_stats_to_dict()

    with open("prefill_prof.json", 'w') as f:
        json.dump(prefill_results, f)

    with open("decode_prof.json", 'w') as f:
        json.dump(decode_results, f)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
