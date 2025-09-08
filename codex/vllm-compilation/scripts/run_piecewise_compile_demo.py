#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch

from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig)


class TinyModule(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden, hidden, bias=False)
        self.lin2 = torch.nn.Linear(hidden, hidden, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        # No attention here; piecewise still works (single piece)
        return self.lin2(self.act(self.lin1(x)))


def main():
    p = argparse.ArgumentParser(
        description="Demo vLLM piecewise backend compile and cache")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--use-inductor", action="store_true")
    args = p.parse_args()

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    comp = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_inductor=args.use_inductor,
        # turn off cudagraph to simplify the demo output
        cudagraph_mode="NONE",
    )
    vcfg = VllmConfig(compilation_config=comp)

    mod = TinyModule(args.hidden).cuda().to(dtype=torch.bfloat16)
    x = torch.randn(args.tokens, args.hidden, device="cuda",
                    dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x, 0)

    # compile with vLLM's backend (selected when level=PIECEWISE)
    import vllm.config as vconf
    with vconf.set_current_vllm_config(vcfg):
        m2 = torch.compile(mod, backend=comp.init_backend(vcfg))
        _ = m2(x)  # trigger compilation and run once

    print("Graphs seen:", compilation_counter.num_graphs_seen)
    print("Piecewise subgraphs seen:",
          compilation_counter.num_piecewise_graphs_seen)
    print("Backend compilations:", compilation_counter.num_backend_compilations)
    if vcfg.compilation_config.local_cache_dir:
        print("Cache dir:", vcfg.compilation_config.local_cache_dir)
        import os
        graph_py = os.path.join(vcfg.compilation_config.local_cache_dir,
                                "computation_graph.py")
        print("Computation graph path (if saved):", graph_py,
              "exists=" + str(os.path.exists(graph_py)))


if __name__ == "__main__":
    main()

