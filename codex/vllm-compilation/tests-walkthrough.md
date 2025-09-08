Tests-First Walkthrough: What Each Test Validates

Compile switches and caching (tests/compile/test_config.py)

- test_use_cudagraphs_dynamic / test_use_cudagraphs: Validates `CompilationConfig.use_cudagraph` default under V1 and that capture counts line up with `cudagraph_capture_sizes` (compilation_counter.num_cudagraph_captured). Pipeline stage: post-Dynamo, during piecewise capture wrapping (cuda_graph.py called inside backends.py call_module).
- test_dynamo_as_is / test_no_compilation / test_enforce_eager: Exercise level control. Ensures VllmBackend is bypassed for `DYNAMO_AS_IS`, disabled entirely for `NO_COMPILATION`, and enforced eager mode toggles. Pipeline stage: wrapper/decorator layer.
- test_VLLM_DISABLE_COMPILE_CACHE: Ensures compiler cache artifacts aren’t written and counters stay at zero for saves. Pipeline stage: CompilerManager.initialize_cache/save_to_file and compiler adaptor .compile.

Fusion passes (tests/compile/test_fusion.py)

- test_fusion_rmsnorm_quant (parametrized): Validates NoOpElimination + FusionPass together. Checks pre/post presence of fp8 quant op vs fused RMSNorm+Quant ops using TestBackend hooks. Pipeline stage: Inductor post-grad custom post-pass; pattern matcher application and manual multi-output replacement; fix functionalization runs after pass manager as last step.

Activation fusion (tests/compile/test_silu_mul_quant_fusion.py)

- Validates Silu+Quant fusion patterns (both static fp8 and optional nvfp4). Pipeline stage: same as above; tests that input/output mapping is correct and kernels are present.

Attention fusion (tests/compile/test_fusion_attn.py)

- Validates fusing quant into supported attention implementations (e.g., FA backends) by moving scales into attention op args; runs DCE to fix FX graph; references bug requiring manual DCE (vLLM issue 23091). Pipeline stage: Inductor post-grad passes with per-layer pattern registration.

Sequence parallelism (tests/compile/test_sequence_parallelism.py)

- Validates AR→RS/AG rewrites around RMSNorm (+quant) across positions in transformer block (first/middle/last). Pipeline stage: post-grad pass; gated by pass_context.runtime_shape and TP world size conditions.

Async TP and collectives fusion (tests/compile/test_async_tp.py and test_fusion_all_reduce.py)

- Validates fused GEMM+RS and AG+GEMM and scaled_mm variants; optional FlashInfer fused allreduce+norm(+quant). Pipeline stage: post-grad passes; requires appropriate dtype/device and optionally FlashInfer.

Piecewise compilation & cudagraph modes (tests/compile/piecewise/*.py)

- test_simple / test_multiple_graphs: Validate VllmBackend graph splitting around configured splitting ops and piecewise compilation count; ensure per-piece runtimes are compiled and callable.
- test_full_cudagraph.py: Exercises FULL/FULL_AND_PIECEWISE across multiple attention backends; ensures outputs match piecewise execution over varied batch sizes and max_tokens and that runtime mode negotiation downgrades as needed.

How to run quickly

- Pure Python demos (no pytest):
  - `codex/vllm-compilation/scripts/run_fusion_rmsnorm_quant.py` – isolates NoOp + Fusion for RMSNorm/FP8.
  - `codex/vllm-compilation/scripts/run_noop_elimination_demo.py` – isolates reshape/slice cleanup.
  - `codex/vllm-compilation/scripts/run_piecewise_compile_demo.py` – runs VllmBackend PIECEWISE flow on a tiny module; prints cache dir and counters.
- Focused pytest:
  - `bash codex/vllm-compilation/scripts/run_component_tests.sh fusion` (or activation|attn|seqpar|async_tp|full|piecewise|cache)

