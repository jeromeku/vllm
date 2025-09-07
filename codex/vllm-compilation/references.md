References: Issues and PRs

PyTorch related

- Multi‑output pattern replacement bug in `PatternMatcherPass` (necessitates manual `process_matches` in FusionPass): pytorch/pytorch issue discussing broken multi‑output automatic replacement. See our handling at vllm/compilation/multi_output_match.py and fusion.py:596–614. Source: https://github.com/pytorch/pytorch/issues/137280
- Pattern matcher cache behavior and need to clear `_seen_patterns` for multiple parameterizations (epsilon variants): context in PyTorch PR thread. Our workaround at fusion.py:526–556 references: https://github.com/pytorch/pytorch/pull/139321
- Custom graph pass API (`torch._inductor.custom_graph_pass`) added in PyTorch ≥2.6; vLLM provides a compatibility shim for 2.5 (`torch25_custom_graph_pass.py`) and installs pass manager via `post_grad_custom_post_pass`. Upstream introduction documented around 2.6 release notes and code. See usage in vllm/compilation/inductor_pass.py and pass_manager.py.

vLLM repository

- Piecewise CUDA Graphs integration (split FX and nested cudagraphs). Project PR introducing and iterating on full vs piecewise capture modes: https://github.com/vllm-project/vllm/pull/20546
- Full cudagraph trees support and mode negotiation with attention backends (FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE): test suite and runtime logic added across v1 gpu model runner and tests. Example tests: tests/compile/piecewise/test_full_cudagraph.py.
- FX graph breakage requiring manual DCE after attention+quant fusion: tracked in vLLM issue referenced in fusion_attn.py: https://github.com/vllm-project/vllm/issues/23091
- Docs on torch.compile integration and piecewise cudagraph rationale: docs/design/torch_compile.md (in‑repo design doc).

Notes

- The Inductor cache shape‑env always‑hit patch and patched `compiled_fx_graph_hash`/`compile_fx_inner` hooking are internal APIs and may change across torch versions. vLLM guards these with version checks: compiler_interface.py:320–468, 489–536.

