# ruff: noqa
from contextlib import contextmanager, nullcontext, ExitStack
import hunter

from vllm import LLM, SamplingParams
import torch

@contextmanager
def hunter_trace(output_path, force_colors=False, filename_alignment=100, enable=True, **filters):
    if enable:
        tracer = hunter.Tracer()

        with open(output_path, 'w') as f:
            call_printer = hunter.CallPrinter(force_colors=force_colors, stream=f, filename_alignment=filename_alignment)
            debugger = hunter.Debugger()
            actions = [call_printer, debugger]
            q = hunter.Q(actions=actions, **filters)
            print(q)
            
            tracer.trace(q)
            yield tracer
            tracer.stop()
    else:
        yield   

# Sample prompts.
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-125m", cuda_graph_sizes=[1])
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
    enable = True
    import os
    print(f"DEBUG::{__file__}:main {os.getpid()} {os.getppid()}")
    if not enable:
        ctx = nullcontext()
    else:
        trace_path = "trace.txt"
        ctx = hunter_trace(output_path=trace_path, filename_contains="core", fullsource_contains='execute_model', enable=True)     
    with ctx:
        main()

