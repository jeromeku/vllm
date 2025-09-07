Custom Inductor Passes — Function‑by‑Function Trace

Pass Manager and registration

- PostGradPassManager: vllm/compilation/pass_manager.py:23–77
  - Orchestrates vLLM’s passes and runs them in `__call__` under a pass context carrying `runtime_shape` (pass_manager.py:36–48). The order is: configured passes → default passes → user‑provided `post_grad_custom_post_pass` → FixFunctionalization last.
  - `configure(self, config)` wires flags from `CompilationConfig.pass_config` to enable:
    - NoOpEliminationPass (reshapes/slices)
    - SequenceParallelismPass (+ AsyncTPPass)
    - FusionPass (RMSNorm/SiluMul + quant)
    - ActivationQuantFusionPass (SiluMul + quant)
    - AttnFusionPass (attention + quant)
    - AllReduceFusionPass (FlashInfer fused collectives)
  - `uuid()` aggregates sub‑pass UUIDs for Inductor code cache correctness across runs (pass_manager.py:63–77).

Common base class utilities

- VllmInductorPass: vllm/compilation/vllm_inductor_pass.py:14–47
  - Provides `begin/end_and_log` timing, `dump_graph(stage)` using Inductor’s lazy graph formatter, and config access.
- InductorPass base/uuid helpers: vllm/compilation/inductor_pass.py:38–111
  - Supplies stable UUIDs based on source hashing or dict hashing so Inductor cache keys include pass configuration.
  - `pass_context(runtime_shape)` exposes current static `runtime_shape` to passes that are shape‑specialized.

FixFunctionalizationPass — defunctionalize hotspot ops

- vllm/compilation/fix_functionalization.py:15–167
  - Scans FX nodes for `auto_functionalized` wrappers, then replaces selected ops with their in‑place originals to avoid extra copies after functionalization (fix_functionalization.py:42–116).
  - Special rotary_embedding case: unwraps the scatter pattern to use the pre‑mutation `mm_node` directly, removing getitems and slice_scatters (fix_functionalization.py:52–73).
  - Handles RMSNorm variants and fused add+RMSNorm, setting appropriate mutated args map and optionally explicit args order for ops whose kwargs path is unreliable (fix_functionalization.py:80–116).
  - Removes staged nodes at the end and logs counts (fix_functionalization.py:120–154).

NoOpEliminationPass — remove redundant reshapes and slices

- vllm/compilation/noop_elimination.py:14–119
  - Collapses reshape chains and deletes reshapes that return the input’s shape (no rank change, ≤1 inferred dim), replacing users with the base tensor (noop_elimination.py:46–74).
  - Removes `slice` that fully cover the input and `slice_scatter` that restores the unmodified base (noop_elimination.py:76–111).
  - Uses SymInt equivalence and meta["val"] shapes for correctness checks (noop_elimination.py:121–167).

FusionPass — RMSNorm/SiluMul + Quantization fusions

- vllm/compilation/fusion.py:1–680
  - Defines pattern classes for RMSNorm + static fp8 quant, fused_add_rms_norm + static fp8 quant, and dynamic per‑token fp8 quant variants (fusion.py:108–240, 240–336, 336–440).
  - Uses `PatternMatcherPass` to register forward‑only replacements and collects multi‑output matches for manual processing due to PyTorch bug with multi‑output auto replacements (fusion.py:31–46, 557–596). See references.
  - `process_matches` manually inserts fused auto_functionalized nodes, rebinds users, sets meta["val"], and runs DCE (fusion.py:596–614).
  - Pass singleton pattern is employed; multiple epsilon variants work by clearing `_seen_patterns` cache (fusion.py:526–556).

ActivationQuantFusionPass — SiluMul + Quantization fusions

- vllm/compilation/activation_quant_fusion.py:1–176
  - Patterns for `silu_and_mul` followed by static fp8 or nvfp4 quant (activation_quant_fusion.py:66–121, 125–176). Registered with forward‑only matcher.

AttnFusionPass — attention + quant fusion when backend supports fused output quant

- vllm/compilation/fusion_attn.py:1–312
  - For each `vllm.attention.Attention` layer, registers patterns that remove trailing quant op and pass scale(s) directly into attention op arguments, followed by reshape fixups (fusion_attn.py:71–146, 182–259).
  - Uses per‑layer registration because wildcards aren’t supported across layer names; this also allows querying backend support in advance (fusion_attn.py:227–259).
  - Runs DCE after pattern match due to FX graph breakage issue; tracked in vLLM issue (fusion_attn.py:268–279). See references.

SequenceParallelismPass — transform AR→RS/AG around RMSNorm (+quant) and enable later GEMM fusions

- vllm/compilation/sequence_parallelism.py:1–236, 240–620
  - Rewrites patterns: AllReduce + RMSNorm (+optional quant) into ReduceScatter → local RMSNorm(+quant) → AllGather (sequence_parallelism.py:75–178, 182–236, 240–620). Specific first/middle/last variants are included to match transformer block locations. Shape applicability respects TP world size (pass_manager.py:33–41; sequence_parallelism.py:210–214).

AsyncTPPass and CollectiveFusion — fuse GEMM with RS/AG and FlashInfer fused allreduce+norm(+quant)

- vllm/compilation/collective_fusion.py:1–660
  - Pattern‑based rewrites for GEMM↔collectives with symmetric memory fused kernels (reduce_scatter and all_gather variants) and scaled_mm variants (collective_fusion.py:23–173, 180–260, 262–401). Enabled only with appropriate dtype/device settings; some bfloat16‑only fusions for per‑row scaling.
  - Optional FlashInfer fused allreduce + RMSNorm(+quant) integration via dynamically registered custom op `flashinfer_trtllm_fused_allreduce_norm` when available (collective_fusion.py:421–660).

Shape‑specialized applicability hooks

- Many passes can be gated by runtime shape (batch size) via `is_applicable_for_shape` and `get_pass_context().runtime_shape` (pass_manager.py:33–41; inductor_pass.py:113–171). For example, AsyncTPPass only applies when `shape % tp_world_size == 0` (collective_fusion.py:402–460).

