# vLLM and PyTorch Inductor Compilation Tests

## vLLM `tests/compile`

### Pass manager and configuration
- `tests/compile/test_pass_manager.py` checks that only `InductorPass` instances can be registered and that the pass manager UUID changes with configuration tweaks【F:tests/compile/test_pass_manager.py†L13-L73】
- `tests/compile/test_config.py` validates environment-driven switches like disabling compile cache and CUDA graph capture via `VllmConfig`.

### Functionalization and fusion
- `tests/compile/test_functionalization.py` compares graphs compiled with and without `FixFunctionalizationPass` to ensure fused kernels are defunctionalized before backend execution【F:tests/compile/test_functionalization.py†L45-L116】
- `tests/compile/test_fusion.py` exercises RMSNorm+FP8 quant fusion and verifies pre/post graph patterns with `TestBackend` helpers【F:tests/compile/test_fusion.py†L27-L135】
- `tests/compile/test_silu_mul_quant_fusion.py` fuses SiLU+mul with FP8/NVFP4 quantization, asserting that original quant ops disappear after the pass【F:tests/compile/test_silu_mul_quant_fusion.py†L37-L157】

### Piecewise and decorator semantics
- `tests/compile/piecewise/test_simple.py` demonstrates piecewise graphs split by a custom op and checks CUDA graph capture counts via `compilation_counter`【F:tests/compile/piecewise/test_simple.py†L1-L133】
- `tests/compile/test_decorator.py` verifies `@support_torch_compile`/`@ignore_torch_compile` decorators, conditional enablement, and cudagraph replay counts.

### Distributed/parallel passes
- `tests/compile/test_sequence_parallelism.py` and `test_async_tp.py` run on multi‑GPU setups to replace collective ops with fused reduce‑scatter/all‑gather variants.
- `tests/compile/test_fusion_all_reduce.py` fuses tensor model parallel `all_reduce` with RMSNorm and optional quantization.

### End‑to‑end coverage
- `tests/compile/test_basic_correctness.py` runs full LLM inference under several `CompilationLevel`s to confirm numerical parity.
- `tests/compile/test_full_graph.py` exports full‑graph captures for various quantization backends and compile configs.

## PyTorch `test/inductor`

Inductor’s test suite mirrors its lowering pipeline:

1. **Front‑end capture** – `test_compile.py` invokes `inductor.compile` directly on FX graphs and exported modules, ensuring Dynamo is not required【422f4e†L32-L61】.
2. **Functionalization/decomposition** – `test_auto_functionalize.py` registers custom ops to verify automatic mutation wrapping before AOT Autograd【4e2be0†L1-L47】.
3. **Custom graph passes** – `test_custom_post_grad_passes.py` registers `CustomGraphPass` objects and asserts pattern matcher counters【2d17fc†L1-L43】.
4. **CUDA graph trees** – `test_cudagraph_trees.py` exercises `cudagraphify_impl` and validates multi‑segment capture mechanics【649e62†L1-L59】.
5. **Scheduler and metrics** – `test_inductor_scheduler.py` counts FLOPs under different fusion strategies using `FlopCounterMode`【9d92e4†L1-L33】.

## Comparison & Opportunities
- **Pass hooks**: vLLM’s `PostGradPassManager` exposes a simpler post‑Autograd hook chain, while PyTorch relies on `CustomGraphPass` and `pattern_matcher` utilities. vLLM could adopt Inductor’s `PatternMatcherPass` to unify fusion predicates.
- **Piecewise compilation**: vLLM’s piecewise backend splits graphs at mutable ops and wraps each segment in CUDA graphs; Inductor’s `test_cudagraph_trees` explores a similar tree of `cudagraphify_impl` but without per‑segment recompilation.
- **Scheduler tuning**: PyTorch’s `test_inductor_scheduler.py` shows FLOP‑based fusion heuristics; analogous hooks in vLLM’s `TestBackend.inductor_config` could surface new tuners for large attention blocks.

## Relevant Issues & PRs
- vLLM FixFunctionalization discussion: [vllm-project/vllm#23612](https://github.com/vllm-project/vllm/issues/23612)
- Piecewise compilation RFC: [vllm-project/vllm#24123](https://github.com/vllm-project/vllm/pull/24123)
- CUDA graph trees upstream tracking: [pytorch/pytorch#124506](https://github.com/pytorch/pytorch/issues/124506)
