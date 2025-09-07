End‑to‑End Flow: From First Compilation to Compiled Call

High‑level timeline

- Model classes opt‑in via `@support_torch_compile` (vllm/compilation/decorators.py:74). The decorator attaches a custom dispatcher `TorchCompileWrapperWithCustomDispatcher` and determines dynamic dims to mark with Dynamo.
- The wrapper invokes `torch.compile(self.forward, backend=compilation_config.init_backend(...))` (vllm/compilation/wrapper.py:30,47–66). For PIECEWISE, `init_backend` returns a `VllmBackend` instance (vllm/config/compilation.py:629–641, 537–562).
- Dynamo calls the backend `VllmBackend.__call__` with an `fx.GraphModule` and example inputs (vllm/compilation/backends.py:466–580). vLLM computes a cache dir, records traced files, and prepares a `CompilerManager`.
- vLLM splits the captured FX graph around configured splitting ops (usually attention) into piecewise submodules (vllm/compilation/backends.py:229–287). It then runs a piecewise interpreter that compiles each submodule for dynamic shape and optionally wraps each piece with a CUDA‑graph wrapper (vllm/compilation/backends.py:299–420, 320–420).
- Compilation uses `CompilerManager` plus `CompilerInterface` adaptors:
  - `InductorAdaptor` for torch>=2.5 (and `InductorStandaloneAdaptor` for >=2.8 dev) (vllm/compilation/compiler_interface.py:174–186, 228–297, 312–740)
  - `EagerAdaptor` as a fallback (vllm/compilation/compiler_interface.py:732–740)
- vLLM injects its post‑grad pass manager into Inductor via `inductor_compile_config["post_grad_custom_post_pass"]` (vllm/compilation/backends.py:421–463; vllm/compilation/pass_manager.py:23–77). The pass manager orchestrates vLLM’s custom FX graph passes.
- If cudagraphs are enabled, each piece is wrapped with a `CUDAGraphWrapper` for PIECEWISE runtime dispatch; full cudagraphs are driven later in the runtime (v1 gpu_model_runner) (vllm/compilation/backends.py:340–360; vllm/compilation/cuda_graph.py:41–166; vllm/v1/worker/gpu_model_runner.py:3040–3120).
- At runtime, the wrapper’s custom dispatcher can jump directly to compiled bytecode without re‑entering Dynamo (vllm/compilation/wrapper.py:98–124; decorators.py:171–238).

Key compile pipeline touchpoints

- torch.compile entry: vllm/compilation/wrapper.py:47–66
- Backend object for PIECEWISE (Dynamo backend): vllm/compilation/backends.py:392–580
- Graph splitting: vllm/compilation/backends.py:229–287
- Dynamic‑shape compile of each piece: vllm/compilation/backends.py:311–337
- Static‑shape specialization by runtime batch sizes: vllm/compilation/cuda_piecewise_backend.py:17–88
- Custom Inductor passes injection: vllm/compilation/backends.py:421–463; pass_manager in vllm/compilation/pass_manager.py
- CUDAGraph capture & replay: vllm/compilation/cuda_graph.py and runtime dispatch keys in v1 runner

