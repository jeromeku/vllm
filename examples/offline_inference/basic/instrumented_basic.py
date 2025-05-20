import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from viztracer import VizTracer, get_tracer

from vllm import LLM, EngineArgs, SamplingParams

tracer_kwargs = dict(
    include_files=["/home/jeromeku/vllm/vllm"],
    ignore_frozen=True,
    ignore_c_function=True,
    log_func_args=True,
    log_func_retval=True,
    dump_raw=False,
    pid_suffix=True,
    # output_dir="traces",
)

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
    tracer = VizTracer(**tracer_kwargs)

    # tracer = get_tracer()
    # assert tracer is not None
    
    # for k,v in tracer_kwargs.items():
    #     print(f"{k}: {getattr(tracer, k)}")

    tracer.output_file = "llm.init.trace.json"
    with tracer:
        engine_args = EngineArgs(cuda_graph_sizes=[len(prompts)])
        llm = LLM(model="facebook/opt-125m", cuda_graph_sizes=engine_args.cuda_graph_sizes)
    
    tracer.output_file = "generate.trace.json"
    
    with tracer:
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
