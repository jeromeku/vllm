#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch

import vllm.envs as envs
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.fusion import (FusedRMSQuantKey, FUSED_OPS, QUANT_OPS,
                                     FusionPass)
from vllm.config import (CompilationConfig, CompilationLevel, PassConfig,
                         VllmConfig)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, QuantKey, ScaleDesc)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, cutlass_fp8_supported, maybe_create_device_identity)
from vllm.platforms import current_platform

from tests.compile.backend import TestBackend


class TestModel(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float, static: bool,
                 cuda_force_torch: bool):
        super().__init__()
        self.cuda_force_torch = cuda_force_torch
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(3)]
        self.wscale = [torch.rand(1, dtype=torch.float32) for _ in range(2)]
        group_shape = GroupShape.PER_TENSOR if static else GroupShape.PER_TOKEN
        quant_scale = ScaleDesc(torch.float32, static, group_shape)
        self.key = QuantKey(dtype=current_platform.fp8_dtype(),
                            scale=quant_scale,
                            symmetric=True)
        if static:
            self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(2)]
        else:
            self.scale = [None for _ in range(2)]
        self.w = [
            torch.rand(hidden_size, hidden_size).to(
                dtype=current_platform.fp8_dtype()).t() for _ in range(2)
        ]

        from tests.compile.utils import override_cutlass_fp8_supported
        with override_cutlass_fp8_supported(not cuda_force_torch):
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=static,
                act_quant_group_shape=group_shape,
            )

    def forward(self, x):
        resid = torch.sqrt(x)
        y = self.norm[0](x)
        x2 = self.fp8_linear.apply(y, self.w[0], self.wscale[0],
                                   input_scale=self.scale[0])
        y2, resid = self.norm[1](x2, resid)
        x3 = self.fp8_linear.apply(y2, self.w[1], self.wscale[1],
                                   input_scale=self.scale[1])
        y3, resid = self.norm[2](x3, resid)
        return y3

    def ops_in_model_before(self):
        return [QUANT_OPS[self.key]]

    def ops_in_model_after(self):
        return [
            FUSED_OPS[FusedRMSQuantKey(self.key, False)],
            FUSED_OPS[FusedRMSQuantKey(self.key, True)],
        ]


def main():
    p = argparse.ArgumentParser(
        description="Run RMSNorm+FP8 fusion pass in isolation")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-tokens", type=int, default=7)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--dtype",
                   choices=["fp16", "bf16"],
                   default="bf16")
    p.add_argument("--static", action="store_true",
                   help="Use static per-tensor scales")
    p.add_argument("--force-torch", action="store_true",
                   help="Force non-cutlass fp8 path if available")
    args = p.parse_args()

    assert envs.VLLM_TARGET_DEVICE in ("cuda", "rocm"), \
        "This demo requires a CUDA/ROCm environment."
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float16 if args.dtype == "fp16" else
                            torch.bfloat16)
    torch.manual_seed(1)
    maybe_create_device_identity()

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        custom_ops=["+rms_norm", "+quant_fp8"],
        pass_config=PassConfig(enable_fusion=True, enable_noop=True),
    ))

    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(vllm_config):
        model = TestModel(args.hidden_size, args.eps, args.static,
                          args.force_torch)
        x = torch.rand(args.num_tokens, args.hidden_size)
        torch._dynamo.mark_dynamic(x, 0)
        ref = model(x)

        noop = NoOpEliminationPass(vllm_config)
        fusion = FusionPass.instance(vllm_config)
        backend = TestBackend(noop, fusion)
        compiled = torch.compile(model, backend=backend)
        out = compiled(x)

        # Validate numerics and transformed ops
        if args.static:
            ATOL, RTOL = (1e-3, 1e-3)
        elif torch.get_default_dtype() == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)
        torch.testing.assert_close(ref, out, atol=ATOL, rtol=RTOL)

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())

        print("OK: Fusion applied and numerics validated.")


if __name__ == "__main__":
    main()

