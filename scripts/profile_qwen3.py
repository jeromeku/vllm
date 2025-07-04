"""
Latency profiler for Qwen-3 with optional Sliding-Window Attention (SWA) on
single or multiple GPUs.

For a *fixed* configuration (one context length and generation length) the
script measures end-to-end latency of a batched `LLM.generate` call and writes
CPU/CUDA self-time extracted from Torch-profiler traces. All heavy-lifting is
performed by vLLM; the script only handles model patching (SWA), prompt
construction, timing, trace parsing and CSV logging.

CLI Arguments (most important)
------------------------------
--n_swa               Number of leading decoder layers replaced by SWA.
--context_length / --ctx
                      Length of the input prompt in tokens.
--gen_tokens          Number of *new* tokens to generate after the context.
--sliding_window_size Window size used by SWA attention (default 256).
--n_prompts           Batch size (identical prompts are broadcast).
--n_trials            Timed repetitions (row per trial).
--tp / --pp           Tensor- and pipeline-parallel sizes; defaults consume all
                      visible GPUs.  If TP×PP > 1 process, V1 multiprocessing
                      is enabled automatically.

Outputs
-------
1. Torch-profiler traces in ``$VLLM_TORCH_PROFILER_DIR`` (default
   ``./vllm_profile``) – useful for drill-down analysis with Chrome-trace.
2. A CSV row appended to ``qwen3_swa_latency_results_multi.csv`` with columns:
   model,n_swa,gen_tokens,context_length,window,n_prompts,tp,pp,trial,
   wall_s,cpu_us,cuda_us

Example
-------
python -m scripts.profile_qwen3_multi \
       --model Qwen/Qwen3-4B \
       --n_swa 8 --ctx 4096 --gen_tokens 128 \
       --tp 2 --pp 1 --n_prompts 4 --n_trials 5
"""

import os
import time
import argparse
import logging
import gzip
import json
import torch
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


os.environ.setdefault("VLLM_TORCH_PROFILER_DIR", "./vllm_profile")
# No FlashInfer sampler for now
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Skip any stale torch.compile cache
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
# Flash attention version
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "3")
os.environ.setdefault("VLLM_USE_V1", "1")
# GF download settings
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Allow fall-back to cloudpickle for complex objects in collective RPC.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    try:
        if torch.cuda.is_available():
            maj, min_ = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{min_}"
    except Exception:
        pass


BASE_SAMPLING_ARGS = dict(
    temperature=0.8,
    top_p=0.95,
    ignore_eos=True,
    stop=[],
    stop_token_ids=[],
)


def _parse_traces(file_paths):
    """Aggregate CPU / CUDA self-time (µs) from a list of gzipped traces."""
    cpu_us = 0
    cuda_us = 0
    for path in file_paths:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        for evt in data.get("traceEvents", []):
            cat = evt.get("cat", "")
            dur = evt.get("dur", 0)
            if "cuda" in cat.lower():
                cuda_us += dur
            else:
                cpu_us += dur
    return cpu_us, cuda_us


def _sync_all():
    """Synchronise all CUDA devices (important for multi-GPU timing)."""
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HF repo or local path of the model to profile")
    parser.add_argument("--n_swa", type=int, default=1,
                        help="How many leading decoder layers to convert to SWA")
    parser.add_argument("--gen_tokens", type=int, default=128,
                        help="Number of new tokens to generate per request")
    parser.add_argument("--context_length", "--ctx", type=int, default=512,
                        help="Length of the input context (prompt) in tokens")
    parser.add_argument("--n_prompts", type=int, default=1,
                        help="Batch size per generation call")
    parser.add_argument("--n_warmup", type=int, default=1,
                        help="Number of warm-up generations before timing")
    parser.add_argument("--n_trials", type=int, default=1,
                        help="How many timed repetitions to run")
    parser.add_argument("--sliding_window_size", type=int, default=256,
                        help="Sliding-window size for SWA attention")
    parser.add_argument("--swa_start_layer", type=int, default=0,
                        help="The starting layer index for SWA layers.")
    parser.add_argument("--pipeline_parallel_size", "--pp", type=int,
                        default=1,
                        help="Number of GPUs to use for pipeline parallelism. "
                             "Default uses all visible GPUs.")
    parser.add_argument("--tensor_parallel_size", "--tp", type=int,
                        default=1,
                        help="Number of GPUs to use for tensor parallelism. "
                             "Default uses all visible GPUs.")
    parser.add_argument("--prompt_seed", type=str,
                        default=" The quick brown fox jumps over the lazy dog.",
                        help="Text snippet that will be repeated to form the context prompt")
    parser.add_argument("--load-format", type=str, default="auto",
                        help="The format of the model weights to load. "
                             "Options: auto, pt, safetensors, npcache, dummy")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tp_size = args.tensor_parallel_size or num_gpus
    pp_size = args.pipeline_parallel_size or 1

    # If user provided only pp but not tp, use remaining GPUs for TP.
    if args.pipeline_parallel_size is not None and args.tensor_parallel_size is None:
        if num_gpus % pp_size != 0:
            raise ValueError(
                f"Cannot split {num_gpus} GPUs into pp={pp_size}. "
                "Choose a pp_size that divides the available GPUs or "
                "specify --tp as well.")
        tp_size = num_gpus // pp_size

    # If user provided only tp but not pp, keep pp = 1 (already set).

    # Validate final combination
    if tp_size * pp_size > num_gpus:
        raise ValueError(
            f"Requested tp={tp_size} * pp={pp_size} = {tp_size * pp_size} GPUs "
            f"but only {num_gpus} are visible.")

    if tp_size < 1 or pp_size < 1:
        raise ValueError("tp and pp sizes must be >= 1")

    # Enable multiprocessing automatically when more than 1 GPU is used.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1" if (tp_size * pp_size) > 1 else "0")

    # Add SWA config overrides from SWA-specific arguments
    config_overrides = {}
    if args.n_swa > 0:
        # The plugin's config will now see these values
        config_overrides["sliding_window_size"] = args.sliding_window_size
        swa_layers = list(range(args.swa_start_layer, args.swa_start_layer + args.n_swa))
        config_overrides["swa_layers"] = swa_layers

    print(f"\n=== Profiling with {args.n_swa} SWA layers | ctx {args.context_length} | gen {args.gen_tokens} | window {args.sliding_window_size} | TP {tp_size} | PP {pp_size} | trials {args.n_trials} ===")

    from vllm.config import CompilationConfig
    compile_config = CompilationConfig(cudagraph_capture_sizes=[args.n_prompts])

    llm = LLM(model=args.model,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              load_format=args.load_format,
              hf_overrides=config_overrides, compilation_config=compile_config)

    # Build prompt of the requested context length.
    tokenizer = llm.get_tokenizer()

    def build_prompt(ctx_len: int) -> str:
        seg = args.prompt_seed
        s = seg
        while len(tokenizer.encode(s)) < ctx_len:
            s += seg
        ids = tokenizer.encode(s)[:ctx_len]
        return tokenizer.decode(ids)

    prompt_text = build_prompt(args.context_length)

    # Warm-up phase
    warmup_params = SamplingParams(max_tokens=8, **BASE_SAMPLING_ARGS)
    for _ in range(args.n_warmup):
        llm.generate([prompt_text] * args.n_prompts, warmup_params)

    # Timed trials
    sampling_params = SamplingParams(max_tokens=args.gen_tokens, **BASE_SAMPLING_ARGS)

    import csv
    csv_path = Path("qwen3_swa_latency_results_multi.csv")
    header = [
        "model",
        "n_swa",
        "gen_tokens",
        "context_length",
        "window",
        "n_prompts",
        "tp",
        "pp",
        "trial",
        "wall_s",
        "cpu_us",
        "cuda_us",
    ]
    write_header = not csv_path.exists()

    for trial in range(1, args.n_trials + 1):
        # Snapshot existing profiler files
        prof_root = os.environ["VLLM_TORCH_PROFILER_DIR"]
        pre_trace = set()
        if os.path.exists(prof_root):
            pre_trace = {f for f in os.listdir(prof_root) if f.endswith(".json.gz")}

        _sync_all()
        start = time.perf_counter()
        llm.start_profile()
        outputs = llm.generate([prompt_text] * args.n_prompts, sampling_params)
        llm.stop_profile()
        _sync_all()
        elapsed = time.perf_counter() - start

        # Validate token count
        for o in outputs:
            if len(o.outputs[0].token_ids) != args.gen_tokens:
                logger.warning("Expected %d tokens, got %d", args.gen_tokens, len(o.outputs[0].token_ids))

        # New trace files for this trial
        post_files = [
            os.path.join(prof_root, f)
            for f in os.listdir(prof_root)
            if f.endswith(".json.gz") and f not in pre_trace
        ]

        cpu_us, cuda_us = _parse_traces(post_files)
        logger.info("[trial %d] wall %.2fs | cpu %.4fus | cuda %.4fus", trial, elapsed, cpu_us, cuda_us)

        row = [
            args.model,
            args.n_swa,
            args.gen_tokens,
            args.context_length,
            args.sliding_window_size,
            args.n_prompts,
            tp_size,
            pp_size,
            trial,
            f"{elapsed:.6f}",
            f"{cpu_us:.4f}",
            f"{cuda_us:.4f}",
        ]

        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
                write_header = False
            writer.writerow(row)

    logger.info("Appended %d trial(s) to %s", args.n_trials, csv_path)

    torch.cuda.empty_cache()
    sys.exit(0) 