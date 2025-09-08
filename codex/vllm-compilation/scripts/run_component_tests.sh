#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest not found; please install dev deps." >&2
  exit 1
fi

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

case "${1:-}" in
  fusion)
    pytest -q tests/compile/test_fusion.py -k fusion_rmsnorm_quant ;;
  activation)
    pytest -q tests/compile/test_silu_mul_quant_fusion.py ;;
  attn)
    pytest -q tests/compile/test_fusion_attn.py ;;
  seqpar)
    pytest -q tests/compile/test_sequence_parallelism.py ;;
  async_tp)
    pytest -q tests/compile/test_async_tp.py ;;
  full)
    pytest -q tests/compile/piecewise/test_full_cudagraph.py ;;
  piecewise)
    pytest -q tests/compile/piecewise/test_simple.py ;;
  cache)
    pytest -q tests/compile/test_config.py -k VLLM_DISABLE_COMPILE_CACHE ;;
  *)
    echo "Usage: $0 {fusion|activation|attn|seqpar|async_tp|full|piecewise|cache}" >&2
    exit 2 ;;
esac

