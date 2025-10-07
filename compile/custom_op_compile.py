# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    from vllm.config import CompilationConfig, CompilationLevel
    compile_config = CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_cudagraph=False,
            use_inductor=False,
    )
    llm = LLM(model="Qwen/Qwen3-0.6B", compilation_config=compile_config)
    
    print(f"{llm.llm_engine.vllm_config.compilation_config.enabled_custom_ops=}")

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
