#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end compile + artifact dump using vLLM LLM.

Gated to avoid downloads: pass --run to actually load the model.

Examples

  # Dry run (prints what would happen)
  python codex/vllm-compilation/scripts/run_e2e_compile_dump.py

  # Actually run with OPT-125M, piecewise cudagraph disabled for clarity
  VLLM_USE_V1=1 \
  python codex/vllm-compilation/scripts/run_e2e_compile_dump.py \
      --run --model facebook/opt-125m --cudagraph-mode NONE \
      --debug-dump ~/.cache/vllm/compile-debug

  # Run with piecewise cudagraphs and capture sizes
  VLLM_USE_V1=1 \
  python codex/vllm-compilation/scripts/run_e2e_compile_dump.py \
      --run --model facebook/opt-125m --cudagraph-mode PIECEWISE \
      --capture-sizes 1 2 4 8
"""

import argparse
import json
import os
import sys
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig


def make_compilation_config(args: argparse.Namespace) -> CompilationConfig:
    cfg = CompilationConfig(level=3)  # PIECEWISE
    cfg.use_inductor = args.use_inductor
    cfg.cudagraph_mode = args.cudagraph_mode
    if args.capture_sizes:
        cfg.cudagraph_capture_sizes = list(map(int, args.capture_sizes))
        cfg.compile_sizes = ["cudagraph_capture_sizes"]
    if args.debug_dump:
        cfg.debug_dump_path = os.path.expanduser(args.debug_dump)
    if args.cache_dir:
        cfg.cache_dir = os.path.expanduser(args.cache_dir)
    return cfg


def describe_outputs(local_cache_dir: Optional[str], debug_dump: Optional[str]):
    out = {}
    if local_cache_dir and os.path.isdir(local_cache_dir):
        out["local_cache_dir"] = local_cache_dir
        out["computation_graph.py"] = os.path.join(local_cache_dir,
                                                    "computation_graph.py")
    if debug_dump:
        # wrapper dumps decompiled transformed bytecode per rank
        rank0 = os.path.join(os.path.expanduser(debug_dump), "rank_0",
                             "transformed_code.py")
        out["transformed_code.py"] = rank0
    return out


def main():
    p = argparse.ArgumentParser(
        description="End-to-end compile + dump artifacts for a small HF model")
    p.add_argument("--run", action="store_true",
                   help="Actually run (will download model).")
    p.add_argument("--model", default="facebook/opt-125m",
                   help="HF model id to load.")
    p.add_argument("--cudagraph-mode",
                   default="NONE",
                   choices=["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
                            "FULL_AND_PIECEWISE"],
                   help="CUDAGraph runtime mode.")
    p.add_argument("--capture-sizes", nargs="*",
                   help="Batch sizes to capture for cudagraph piecewise.")
    p.add_argument("--use-inductor", action="store_true",
                   help="Compile with Inductor backend.")
    p.add_argument("--debug-dump", default="",
                   help="Directory to dump debugging artifacts.")
    p.add_argument("--cache-dir", default="",
                   help="Override vLLM compile cache directory.")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    if not args.run:
        print("--run not set; dry-run. This script will:")
        print("- Load", args.model, "with vLLM V1 engine")
        print("- Compile level PIECEWISE with cudagraph-mode:",
              args.cudagraph_mode)
        if args.capture_sizes:
            print("- Capture cudagraph sizes:", args.capture_sizes)
        if args.use_inductor:
            print("- Use Inductor backend")
        if args.debug_dump:
            print("- Dump debug artifacts under:", args.debug_dump)
        print("Re-run with --run to execute.")
        sys.exit(0)

    # Recommended env to keep this in-process and verbose
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

    comp_cfg = make_compilation_config(args)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=1024,
        compilation_config=comp_cfg,
        seed=123,
    )

    prompts = ["hello world"] * args.batch
    out = llm.generate(prompts, SamplingParams(temperature=0.0,
                                               max_tokens=args.max_tokens))
    print("Generated tokens:", [o.outputs[0].text for o in out])

    # Counters and locations
    local_cache_dir = comp_cfg.local_cache_dir
    paths = describe_outputs(local_cache_dir, comp_cfg.debug_dump_path)
    report = {
        "graphs_seen": compilation_counter.num_graphs_seen,
        "piecewise_subgraphs": compilation_counter.num_piecewise_graphs_seen,
        "backend_compiles": compilation_counter.num_backend_compilations,
        "cudagraph_captured": compilation_counter.num_cudagraph_captured,
        "artifacts": paths,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

