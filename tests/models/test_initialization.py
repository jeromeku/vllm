# SPDX-License-Identifier: Apache-2.0
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_TRACE_FUNCTION"] = "1"

from contextlib import ExitStack
from typing import Iterable
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm import LLM
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.logger import logger
from vllm.model_executor.models.qwen3 import AutoWeightsLoader, Qwen3ForCausalLM
from vllm.utils import GiB_bytes
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.engine.core import EngineCore as V1EngineCore

from .registry import HF_EXAMPLE_MODELS, QWEN3_EXAMPLE_MODELS

NUM_LAYERS_TO_LOAD = 1


def get_layer_prefixes_to_exclude(
    model_id: str, num_layers_to_load: int = NUM_LAYERS_TO_LOAD
):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    num_layers = config.num_layers

    return [f"layer{i}." for i in range(num_layers_to_load, num_layers)]


def patched_loader(model_id: str, num_layers_to_load: int = NUM_LAYERS_TO_LOAD):
    layer_pats_to_exclude = get_layer_prefixes_to_exclude(
        model_id, num_layers_to_load
    )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        breakpoint()
        skip_prefixes = layer_pats_to_exclude
        if self.config.tie_word_embeddings:
            skip_prefixes += ["lm_head."]
        logger.debug(f"Setting skip prefixes for {model_id} to {skip_prefixes}")
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)

    return load_weights


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch: str, monkeypatch: pytest.MonkeyPatch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # Avoid OOM and reduce initialization time by only using 1 layer
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        hf_config.update(model_info.hf_overrides)

        text_config = hf_config.get_text_config()

        text_config.update(
            {
                "num_layers": 1,
                "num_hidden_layers": 1,
                "num_experts": 2,
                "num_experts_per_tok": 2,
                "num_local_experts": 2,
            }
        )

        if hasattr(hf_config, "vision_config"):
            hf_config.vision_config.update(
                {
                    "num_layers": 1,
                    "num_hidden_layers": 1,
                }
            )

        return hf_config

    # Avoid calling model.forward()
    def _initialize_kv_caches_v0(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    def _initialize_kv_caches_v1(self, vllm_config):
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        scheduler_kv_cache_config = get_kv_cache_config(
            vllm_config,
            kv_cache_specs[0],
            10 * GiB_bytes,
        )

        # gpu_blocks (> 0), cpu_blocks, scheduler_kv_cache_config
        return 1, 0, scheduler_kv_cache_config

    with (
        patch.object(
            V0LLMEngine, "_initialize_kv_caches", _initialize_kv_caches_v0
        ),
        patch.object(
            V1EngineCore, "_initialize_kv_caches", _initialize_kv_caches_v1
        ),
        monkeypatch.context() as m,
    ):
        if model_info.v0_only:
            m.setenv("VLLM_USE_V1", "0")
        LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            speculative_config={
                "model": model_info.speculative_model,
                "num_speculative_tokens": 1,
            }
            if model_info.speculative_model
            else None,
            trust_remote_code=model_info.trust_remote_code,
            max_model_len=model_info.max_model_len,
            load_format="dummy",
            hf_overrides=hf_overrides,
        )


def _initialize_kv_caches_v1(self, vllm_config):
    logger.debug("Overriding v1 kvcache...")
    kv_cache_specs = self.model_executor.get_kv_cache_specs()
    scheduler_kv_cache_config = get_kv_cache_config(
        vllm_config,
        kv_cache_specs[0],
        10 * GiB_bytes,
    )
    # gpu_blocks (> 0), cpu_blocks, scheduler_kv_cache_config
    return 1, 0, scheduler_kv_cache_config


# Avoid calling model.forward()
def _initialize_kv_caches_v0(self) -> None:
    logger.debug("Overriding v0 kvcache...")

    self.cache_config.num_gpu_blocks = 0
    self.cache_config.num_cpu_blocks = 0


# Avoid OOM and reduce initialization time by only using 1 layer
def hf_overrides(
    model_info,
) -> PretrainedConfig:
    def _inner(hf_config: PretrainedConfig):
        hf_config.update(model_info.hf_overrides)
        text_config = hf_config.get_text_config()

        text_config.update(
            {
                "num_layers": 1,
                "num_hidden_layers": 1,
                "num_experts": 2,
                "num_experts_per_tok": 2,
                "num_local_experts": 2,
            }
        )

        if hasattr(hf_config, "vision_config"):
            hf_config.vision_config.update(
                {
                    "num_layers": 1,
                    "num_hidden_layers": 1,
                }
            )

        return hf_config

    _inner.__name__ = "debugging_config"

    return _inner


def test_qwen3(model_arch: str, num_layers_to_load: int = 1):
    from vllm.debugging import get_tracer

    model_info = QWEN3_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    Qwen3ForCausalLM.load_weights = patched_loader(
        model_info.default, num_layers_to_load=num_layers_to_load
    )

    tracer = get_tracer()
    model_id = model_arch.split("/")[-1]
    print(f"Testing model {model_id}")
    tracer.output_file = f"{model_id}.init.json"

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                V0LLMEngine, "_initialize_kv_caches", _initialize_kv_caches_v0
            )
        )
        stack.enter_context(
            patch.object(
                V1EngineCore, "_initialize_kv_caches", _initialize_kv_caches_v1
            )
        )
        stack.enter_context(tracer)

        llm = LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            speculative_config={
                "model": model_info.speculative_model,
                "num_speculative_tokens": 1,
            }
            if model_info.speculative_model
            else None,
            trust_remote_code=model_info.trust_remote_code,
            max_model_len=model_info.max_model_len,
            # load_format="dummy" if load_dummy else None,
            hf_overrides=hf_overrides(model_info),
        )


if __name__ == "__main__":
    from vllm.debugging import vllm_debug_context

    model_archs = QWEN3_EXAMPLE_MODELS.get_supported_archs()

    for model_arch in model_archs:
        print(f"Testing model arch: {model_arch}")
        with vllm_debug_context():
            test_qwen3(model_arch=model_arch)
        break
