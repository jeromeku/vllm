#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import VllmConfig, CompilationConfig, CompilationLevel


class ReshapeChain(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x):
        # chain of reshapes equivalent to last
        y = torch.reshape(x, (-1, self.hidden // 2, 2))
        z = torch.reshape(y, (-1, self.hidden))
        w = torch.reshape(z, (-1, self.hidden // 2, 2))
        return w


def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    vcfg = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE))
    mod = ReshapeChain(64).cuda()
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

    # trace to FX and run pass in isolation
    from torch.fx import symbolic_trace
    gm = symbolic_trace(mod)

    print("Before: ops count:", sum(1 for _ in gm.graph.nodes))

    # Run only NoOpEliminationPass
    p = NoOpEliminationPass(vcfg)
    p.begin()
    p.dump_graph(gm.graph, "before_noop_demo")
    p(gm.graph)
    p.end_and_log()
    gm.graph.eliminate_dead_code()
    gm.recompile()

    print("After: ops count:", sum(1 for _ in gm.graph.nodes))
    print("OK: NoOpEliminationPass executed in isolation.")


if __name__ == "__main__":
    main()

