"""Minimal script demonstrating piecewise compilation and cudagraph capture."""
import torch
from torch import nn
from torch.library import Library

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, CUDAGraphMode,
                         VllmConfig, set_current_vllm_config)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.utils import direct_register_custom_op

# custom op registered into the "silly" namespace
global_counter = 0
lib = Library("silly", "FRAGMENT")

def op(q, k, v, out):
    global global_counter
    global_counter += 1
    out.copy_(q)

def op_fake(q, k, v, out):
    return

direct_register_custom_op("attention", op, ["out"], op_fake, lib)

@support_torch_compile
class Toy(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, out)
        return out

if __name__ == "__main__":
    cfg = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_cudagraph=True,
        splitting_ops=["silly.attention"],
        cudagraph_copy_inputs=True,
        cudagraph_capture_sizes=[1, 2],
    ))
    with set_current_vllm_config(cfg):
        model = Toy(cfg)

    x = torch.randn(2).cuda()
    # warmup
    with set_forward_context(None, vllm_config=cfg):
        model(x)
    # capture & replay for batch size 2
    with set_forward_context(None, vllm_config=cfg,
                             cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                             batch_descriptor=BatchDescriptor(num_tokens=2)):
        print(model(torch.zeros(2).cuda()))
