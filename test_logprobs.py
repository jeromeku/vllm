import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"


from typing import Any, Callable, Optional, Union

import torch

from tests.models.utils import TokensTextLogprobs, TokensTextLogprobsPromptLogprobs

from vllm import LLM, SamplingParams
from vllm.config import TaskOption, CompilationConfig, CompilationLevel
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    TextPrompt,
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams

# Sample prompts.


class VllmRunner:
    """
    The default value of some arguments have been modified from
    {class}`~vllm.LLM` as follows:

    - `trust_remote_code`: Set to `True` instead of `False` for convenience.
    - `seed`: Set to `0` instead of `None` for test reproducibility.
    - `max_model_len`: Set to `1024` instead of `None` to reduce memory usage.
    - `block_size`: To reduce memory usage, set default to `64` if on XPU
        devices, otherwise default to `16`.
    - `enable_chunked_prefill`: Set to `False` instead of `None` for
      test reproducibility.
    - `enforce_eager`: Set to `False` to test CUDA graph.
    """

    def __init__(
        self,
        model_name: str,
        task: TaskOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        seed: Optional[int] = 0,
        max_model_len: int = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16 if not torch.xpu.is_available() else 64,
        enable_chunked_prefill: Optional[bool] = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            task=task,
            tokenizer=tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            seed=seed,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            **kwargs,
        )

    def get_inputs(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        images=None,
        videos=None,
        audios=None,
    ) -> list[TextPrompt]:
        if any(x is not None and len(x) != len(prompts) for x in [images, videos, audios]):
            raise ValueError("All non-None multimodal inputs must have the same length as prompts")

        inputs = []
        for i, prompt in enumerate(prompts):
            multi_modal_data = {}
            if images is not None and (image := images[i]) is not None:
                multi_modal_data["image"] = image
            if videos is not None and (video := videos[i]) is not None:
                multi_modal_data["video"] = video
            if audios is not None and (audio := audios[i]) is not None:
                multi_modal_data["audio"] = audio

            text_prompt_kwargs = {
                ("prompt" if isinstance(prompt, str) else "prompt_embeds"): prompt,
                "multi_modal_data": multi_modal_data or None,
            }
            inputs.append(TextPrompt(**text_prompt_kwargs))

        return inputs

    def generate(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        sampling_params: SamplingParams,
        images=None,
        videos=None,
        audios=None,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        req_outputs = self.model.generate(inputs, sampling_params=sampling_params, **kwargs)

        outputs: list[tuple[list[list[int]], list[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: list[list[int]] = []
            req_sample_output_strs: list[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append((prompt_str or "") + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: list[RequestOutput],
    ) -> list[TokensTextLogprobsPromptLogprobs]:
        outputs: list[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs, req_output.prompt_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        images: any = None,
        audios: any = None,
        videos: any = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs], list[TokensTextLogprobsPromptLogprobs]]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        req_outputs = self.model.generate(inputs, sampling_params=sampling_params, **kwargs)

        toks_str_logsprobs_prompt_logprobs = self._final_steps_generate_w_logprobs(req_outputs)
        # Omit prompt logprobs if not required by sampling params
        return (
            [x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
            if sampling_params.prompt_logprobs is None
            else toks_str_logsprobs_prompt_logprobs
        )

    def generate_encoder_decoder_w_logprobs(
        self,
        encoder_decoder_prompts: list[ExplicitEncoderDecoderPrompt[str, str]],
        sampling_params: SamplingParams,
    ) -> Union[list[TokensTextLogprobs], list[TokensTextLogprobsPromptLogprobs]]:
        """
        Logprobs generation for vLLM encoder/decoder models
        """

        assert sampling_params.logprobs is not None
        req_outputs = self.model.generate(encoder_decoder_prompts, sampling_params=sampling_params)
        toks_str_logsprobs_prompt_logprobs = self._final_steps_generate_w_logprobs(req_outputs)
        # Omit prompt logprobs if not required by sampling params
        return (
            [x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
            if sampling_params.prompt_logprobs is None
            else toks_str_logsprobs_prompt_logprobs
        )

    def generate_greedy(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        max_tokens: int,
        images=None,
        videos=None,
        audios=None,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(
            prompts, greedy_params, images=images, videos=videos, audios=audios, **kwargs
        )
        return [(output_ids[0], output_str[0]) for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: list[str],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        images=None,
        audios=None,
        videos=None,
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs], list[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )

        return self.generate_w_logprobs(
            prompts, greedy_logprobs_params, images=images, audios=audios, videos=videos, **kwargs
        )

    def generate_encoder_decoder_greedy_logprobs(
        self,
        encoder_decoder_prompts: list[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        skip_special_tokens: bool = True,
    ) -> Union[list[TokensTextLogprobs], list[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=(num_prompt_logprobs),
            skip_special_tokens=skip_special_tokens,
        )
        """
        Greedy logprobs generation for vLLM encoder/decoder models
        """

        return self.generate_encoder_decoder_w_logprobs(
            encoder_decoder_prompts, greedy_logprobs_params
        )

    def generate_beam_search(
        self,
        prompts: list[str],
        beam_width: int,
        max_tokens: int,
        images=None,
        videos=None,
        audios=None,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        outputs = self.model.beam_search(
            inputs, BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens)
        )
        returned_outputs = []
        for output in outputs:
            token_ids = [x.tokens for x in output.sequences]
            texts = [x.text for x in output.sequences]
            returned_outputs.append((token_ids, texts))
        return returned_outputs

    def classify(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.model.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

    def embed(
        self, prompts: list[str], images=None, videos=None, audios=None, *args, **kwargs
    ) -> list[list[float]]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        req_outputs = self.model.embed(inputs, *args, **kwargs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def encode(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.model.encode(prompts)
        return [req_output.outputs.data for req_output in req_outputs]

    def score(
        self,
        text_1: Union[str, list[str]],
        text_2: Union[str, list[str]],
        *args,
        **kwargs,
    ) -> list[float]:
        req_outputs = self.model.score(text_1, text_2, *args, **kwargs)
        return [req_output.outputs.score for req_output in req_outputs]

    def apply_model(self, func: Callable):
        executor = self.model.llm_engine.model_executor
        return executor.apply_model(func)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


def main():
    # Create an LLM.
    prompts = [
        "Hello, my name is",
    ]
    # Create a sampling params object.
    num_logprobs = 3
    num_samples = 1
    # 0 == greedy
    sampling_params = SamplingParams(n=num_samples, temperature=0, logprobs=num_logprobs, prompt_logprobs=None)

    MODEL_ID = "facebook/opt-125m"
    SEED = 0
    MAX_MODEL_LEN = 1024
    DTYPE = torch.float32
    BLOCK_SIZE = 16
    SWAP_SPACE = 4
    CHUNKED_PREFILL = False

    compilation_config = CompilationConfig(cudagraph_capture_sizes=[1])
    
    # vllm_runner = VllmRunner(
    #     model_name=MODEL_ID,
    #     dtype=DTYPE,
    #     seed=SEED,
    #     max_model_len=MAX_MODEL_LEN,
    #     block_size=BLOCK_SIZE,
    #     enable_chunked_prefill=CHUNKED_PREFILL,
    #     swap_space=SWAP_SPACE,
    #     compilation_config=compilation_config
    # )

    llm = LLM(model=MODEL_ID, dtype=DTYPE, swap_space=SWAP_SPACE, compilation_config=compilation_config, max_model_len=MAX_MODEL_LEN)
    # llm.apply_model(lambda m: print(type(m).__name__))
    func  = lambda m: print(f"{type(m).__name__}")
    def rpc_func(worker):
        model = worker.get_model()
        print(dict(model.named_modules()))

    llm.llm_engine.collective_rpc(rpc_func)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    # parsed_outputs = vllm_runner.generate(prompts, sampling_params)
    # outputs: list[RequestOutput]= vllm_runner.model.generate(prompts, sampling_params=sampling_params)
    # output = outputs[0]
    # prompt = output.prompt
    # prompt_logprobs = output.prompt_logprobs
    # out = output.outputs[0]

    # logprobs = out.logprobs
    # token_ids = out.token_ids
    # print(out.text)
    # print(f"{token_ids=}")
    # from pprint import pprint
    # for i, logprob in enumerate(logprobs):
    #     tok_id = token_ids[i]
    #     lp = logprob[tok_id]
    #     print(f"{i}: {tok_id} {lp}")
    
    # executor = vllm_runner.model.llm_engine.model_executor
    # breakpoint()
    # executor.apply_model(lambda model: print(f"{type(model).__name__}"))


if __name__ == "__main__":
    main()
