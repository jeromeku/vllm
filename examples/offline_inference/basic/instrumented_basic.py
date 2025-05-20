
from dataclasses import asdict

import viztracer

from vllm import LLM, EngineArgs, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    tracer_kwargs = dict(
        include_files=["/home/jeromeku/vllm/vllm"],
        ignore_frozen=True,
        ignore_c_function=True,
        log_func_args=True,
        log_func_retval=True,
        dump_raw=False,
    )

    tracer = viztracer.VizTracer(**tracer_kwargs)

    engine_args = EngineArgs(cuda_graph_sizes=[len(prompts)])

    tracer.output_file = "llm.trace.json"
    
    with tracer:
        llm = LLM(model="facebook/opt-125m", cuda_graph_sizes=engine_args.cuda_graph_sizes)
        outputs = llm.generate(prompts, sampling_params)
    

    # tracer.output_file = "LLM.generate.trace.json"
    
    # with tracer:
      
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
