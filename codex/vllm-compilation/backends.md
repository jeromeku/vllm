Backends, Splitting, and Compilation Orchestration

Decorator and wrapper

- support_torch_compile: vllm/compilation/decorators.py:74–178
  - Infers or accepts `dynamic_arg_dims`; marks runtime tensors via `torch._dynamo.mark_dynamic` before first compile (decorators.py:200–238).
  - Starts monitoring and logs traced files by patching `InliningInstructionTranslator.inline_call` to record `code.co_filename` (decorators.py:208–236). This directly “touches” Dynamo’s inlining translator.
- TorchCompileWrapperWithCustomDispatcher: vllm/compilation/wrapper.py:19–124
  - Calls `torch.compile(self.forward, backend=..., options=...)` (wrapper.py:47–66). The backend is string or callable; for PIECEWISE it is a callable backend object (see below).
  - Registers a Dynamo bytecode hook to capture transformed bytecode and optionally decompile it for debugging (wrapper.py:82–116). If cudagraph is enabled and `update` appears in transformed code, it raises to prevent buffer mutation during cudagraph capture (wrapper.py:116–123).
  - Provides `dispatch_to_code(index)` to bypass guard evaluation (wrapper.py:125–141).

VllmBackend (Dynamo backend)

- Entry point: vllm/compilation/backends.py:392–580
  - Computes a cache directory by hashing: envs, model config, traced files content, and compiler hash (backends.py:500–542). Caches under `~/.cache/vllm/torch_compile_cache/<hash>/rank_i_j/<prefix>`.
  - Initializes `CompilerManager` with chosen compiler adaptor (Inductor/Eager) (backends.py:52–80, 107–151).
  - Splits the FX graph by `CompilationConfig.splitting_ops` (defaults include attention ops; config logic in vllm/config/compilation.py:580–640). Split function: `split_graph` returns a stitched `GraphModule` plus `SplitItem`s with flags for the splitting ops (backends.py:229–287).
  - Post‑grad pass manager injection: `configure_post_pass()` sets `inductor_config["post_grad_custom_post_pass"] = PostGradPassManager()`; allows user‑added passes too (backends.py:421–463).
  - PiecewiseCompileInterpreter: runs the stitched graph once with fake inputs; for each submodule name selected for compilation, it:
    - Compiles a dynamic‑shape version via `CompilerManager.compile(..., runtime_shape=None)` (backends.py:311–337).
    - Builds a `PiecewiseBackend` to lazily specialize by runtime batch size (backends.py:338–347; see cuda_piecewise_backend.md section below).
    - Optionally wraps the submodule with a platform static‑graph wrapper (CUDA uses `CUDAGraphWrapper`) passing PIECEWISE mode; sets conservative capture options across pieces: enable debug log for first piece, disable GC on subsequent pieces, and weak‑ref outputs for last piece (backends.py:348–360).
  - If `cudagraph_copy_inputs` is enabled, replaces returned callable so inputs with symbolic dims are copied into persistent buffers managed by the backend before execution (backends.py:586–624).

CompilerManager and CompilerInterface adaptors

- CompilerManager (cache and multi‑shape compile): vllm/compilation/backends.py:82–222
  - Computes compiler hash (backends.py:107–110), initializes compiler cache on disk, loads/updates `vllm_compile_cache.py` (backends.py:141–171).
  - Loads compiled graph by (runtime_shape, graph_index, compiler.name) (backends.py:173–204), otherwise calls adaptor.compile, records timing, and stores a cache “handle” for future load (backends.py:206–287).
- InductorAdaptor: vllm/compilation/compiler_interface.py:312–740
  - Redirects Inductor and Triton caches into vLLM’s cache dir (compiler_interface.py:344–361).
  - Patches internal Inductor code paths to capture compiled graph hash and file path and to always “hit” shape envs outside Dynamo trace (compiler_interface.py:378–468, 489–536). This includes patching `compiled_fx_graph_hash`, `FxGraphCache._get_shape_env`, and for torch>=2.6, `compile_fx_inner` and AOTAutogradCache shape env access.
  - Ensures caching is allowed even outside Dynamo’s tracing context (`_check_can_cache` no‑op hook) and wraps Inductor conventions (list args; tuple returns) to a Dynamo‑friendly callable (compiler_interface.py:512–592, 611–655).
  - Sets runtime tuning patches (e.g., `max_autotune`, `coordinate_descent_tuning`) when compiling for a specific static `runtime_shape` (compiler_interface.py:721–729).
- InductorStandaloneAdaptor (torch>=2.8.dev): vllm/compilation/compiler_interface.py:174–307
  - Uses `torch._inductor.standalone_compile` to produce and persist a `CompiledArtifact`, then loads it later via `CompiledArtifact.load` (compiler_interface.py:216–297). `key` controls artifact path under vLLM cache.
- EagerAdaptor: returns the FX graph as the runnable (no caching) (compiler_interface.py:732–740).

PiecewiseBackend (lazy specialization by runtime shape)

- vllm/compilation/cuda_piecewise_backend.py:13–88
  - Holds the compiled dynamic‑shape runnable and a requested set of static shapes (`compile_sizes`) to compile on demand.
  - On first run, calls general dynamic runnable. On subsequent runs, if runtime batch size matches a `compile_sizes` entry and hasn’t been compiled yet, compiles a static variant via `CompilerManager.compile(..., runtime_shape=bs)` and swaps in the specialized runnable (cuda_piecewise_backend.py:50–88). When last graph finishes compiling all sizes, calls `compiler_manager.save_to_file()` and ends monitoring.

Graph splitting details

- split_graph: vllm/compilation/backends.py:229–287
  - Assigns increasing subgraph IDs. For nodes with `call_function` whose `str(node.target)` is in `splitting_ops`, it emits a boundary before and after that node (so attention itself becomes a “splitting graph” that isn’t compiled). Uses `torch.fx.passes.split_module.split_module(..., keep_original_order=True)` to preserve mutation semantics.

