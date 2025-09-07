CUDA Graphs: Piecewise and Full

Where CUDA Graphs integrate

- vllm/compilation/backends.py
  - In `PiecewiseCompileInterpreter.call_module`, when cudagraphs are enabled, each compiled submodule is wrapped using a platform “static graph wrapper” obtained from `current_platform.get_static_graph_wrapper_cls()`; CUDA uses `CUDAGraphWrapper` (backends.py:348–360).
  - The wrapper is always created with runtime_mode=PIECEWISE, even if a full cudagraph is also used elsewhere. This keeps piecewise capture distinct from any FULL capture driven by runtime (backends.py:352–360).
- vllm/compilation/cuda_graph.py
  - `CUDAGraphWrapper` captures and replays per `BatchDescriptor` key coming from the forward context (cuda_graph.py:97–140). It records input pointers during capture (debug mode) and asserts they match on replay (cuda_graph.py:142–166).
  - Uses a global pool from `current_platform.get_global_graph_pool()` to minimize per‑capture allocation overhead (cuda_graph.py:82–90). Optionally disables `gc.collect`/`torch.cuda.empty_cache` during mass capture after the first piece to accelerate capture (cuda_graph.py:115–136).
  - Always returns weak refs to outputs to save memory after capture; returns strong ref only during capture to let PyTorch manage graph‑owned buffers safely (cuda_graph.py:132–166).
  - Guarded by `validate_cudagraph_capturing_enabled()` so invalid capture phases can error out early (compilation/monitor.py:38–57; cuda_graph.py:112–118).

Runtime dispatch: piecewise vs full

- Piecewise capture is created inside the compiled FX graph per piece. The inputs for symbolic dims may be copied into persistent buffers if `cudagraph_copy_inputs=True` (backends.py:586–624), ensuring stable addresses for capture/replay.
- Full cudagraph is driven by the v1 runtime: `gpu_model_runner` determines whether FULL, FULL_DECODE_ONLY, or FULL_AND_PIECEWISE is supported by the selected attention backend; it may downgrade or switch modes based on backend capabilities and spec‑decode constraints (vllm/v1/worker/gpu_model_runner.py:3040–3120). Those decisions set dispatch keys so outer cudagraph trees capture either the entire forward, only decode, or combined with inner piecewise captures.

Why cudagraph trees vs manual cuda.Graphs?

- The vLLM integration relies on torch’s cudagraph dispatcher patterns and runtime contexts so that Inference can mix modes (NONE/PIECEWISE/FULL) without rewriting large swaths of scheduling logic. `CUDAGraphWrapper` composes with torch.compile’s FX graph and with runtime FULL capture (outer level), forming “cudagraph trees” where safe. This approach benefits from PyTorch’s graph pool management and avoids bespoke lifetime/aliasing machinery, while still exposing control over capture sizes and copy‑in buffers via `CompilationConfig`.

Capture size policies

- `CompilationConfig.cudagraph_capture_sizes` configure batch sizes for piecewise capture; vLLM also computes padded sizes and a mapping from runtime batch to nearest captured size (vllm/config/compilation.py:556–620). Tests assert capture counts and toggling via config (tests/compile/test_config.py:58–83).

