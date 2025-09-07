# vLLM × `torch.compile`: a forensic, line-by-line tour

> Scope: a practical deep dive into vLLM’s custom integration with `torch.compile`, including wrappers, bytecode hooks, compiler interface, post-grad pass manager, custom passes, and the CUDA **piecewise** backend & graph capture. This document tracks the end-to-end path from **first invocation** → **graph capture/compilation** → **compiled call**. File paths and line numbers follow vLLM `main` as of 2025‑08/09; small shifts are expected across minor releases.

⚠️ **Reading notes**
- Line numbers are approximate and tied to the docs-linked source (they drift between releases). Use the referenced objects/sections as anchors.
- Code excerpts here are intentionally short; prefer clicking the source links in your editor for full context.
- Where PyTorch versions diverge (2.5 vs 2.6+), vLLM shims are noted.

---

## 0) Mental model: the moving pieces

- **Wrapper**: `vllm/compilation/wrapper.py`
  - Provides `TorchCompileWrapper`/`TorchCompileWrapperWithCustomDispatcher`.
  - Installs a **Dynamo bytecode hook** to capture compiled bytecode objects and enable **direct dispatch** into compiled callables.
  - Marks dynamic shapes for inputs prior to the first compiled call; forwards into `torch.compile` with a **custom backend**.

- **Compiler interface**: `vllm/compilation/compiler_interface.py`
  - Bridges the Dynamo/Inductor context to vLLM. Orchestrates **compile**(GM, example_inputs, config) and returns compiled graph + handle.
  - Provides context for shape guards/symbolics, pass configs, and caching.

- **Pass manager & passes**:
  - `vllm/compilation/pass_manager.py`: runs **post-grad** Inductor passes; owns pass config, pass ordering, and inductor code‑cache UUIDs.
  - `vllm/compilation/vllm_inductor_pass.py`: base pass with timing/logging/dumping utilities.
  - Selected passes:
    - `activation_quant_fusion.py`: rewrite/fuse specific custom ops (pattern‑matcher), requires (auto‑)functionalization.
    - `noop_elimination.py`: remove redundant `reshape`/`slice` chains (enables other fusions e.g. RMSNorm‑quant).
    - Compat shim: `torch25_custom_graph_pass.py` to emulate 2.6 `CustomGraphPass` on 2.5.

- **Piecewise backend**:
  - `vllm/compilation/base_piecewise_backend.py` + `cuda_piecewise_backend.py` (or historically `backends.py`).
  - Splits the FX graph at configured **splitting ops** (e.g., attention) and compiles **subgraphs**; captures **CUDA graphs** per shape bucket.
  - Manages warmup & replay pools, and a **shape → captured graph** dispatch table.

---

## 1) First contact: the `TorchCompileWrapper` path

**File**: `vllm/compilation/wrapper.py`

1. **`__init__`**
   - Stores `forward` and selected config. Creates a partial `torch.compile(self.forward, fullgraph=<env flag>, backend=<VllmBackend>)` callable, **but does not run** it yet.
   - Saves the original `forward.__code__` and sets up a small list to hold **compiled bytecodes** seen via Dynamo.

2. **`bytecode_hook(self, old_code: CodeType, new_code: CodeType)`**
   - Dynamo calls this hook after compiling a frame. vLLM appends `new_code` to `self.compiled_codes` and may **swap** the dispatcher to go direct to compiled code.
   - Tiny excerpt (trimmed):
     ```python
     # called by torch._dynamo.convert_frame
     self.compiled_codes.append(new_code)
     ```

3. **`__call__(*args, **kwargs)`**
   - Before first execution: mark symbolic dims via `torch._dynamo.mark_dynamic` on incoming tensors to align with dynamic guards and shape buckets.
   - If no compiled codes yet (first run): execute `self.compiled_callable(*args, **kwargs)` **under a patched inliner** so we capture the intended graph forms; this triggers compilation.
   - After the first successful compile: depending on a runtime flag, choose between **custom dispatcher** (jump to compiled bytecode) or route back through the compiled callable.

**Outcome**: one or more **compiled bytecodes** are now registered; the wrapper is ready to dispatch into compiled paths.

---

## 2) The backend callable: `VllmBackend(...)`

**File**: historically `vllm/compilation/backends.py` (v0.9–0.10); in newer trees the logic is partially split across `cuda_piecewise_backend.py` and the compiler interface.

At a high level, the backend object given to `torch.compile(..., backend=VllmBackend)` implements `__call__(gm: fx.GraphModule, example_inputs, **kwargs)` and returns a **callable** that executes the compiled artifact. vLLM’s version does more:

1. **Graph split (piecewise)**
   - If **PIECEWISE** level: wrap `gm` in a `PiecewiseCompileInterpreter` and split at configured ops (e.g. attention/unified_attention/short_conv/mamba mixers).
   - For each **submodule** selected for compilation: call into the **CompilerManager** (or direct **CompilerInterface**) to compile with inductor.

2. **Compilation caching**
   - Compute a **compile hash** from: model + compilation config + pass config + selected subgraph structure + (optionally) environment tags; look up **disk cache**.
   - If cache hit: **load compiled graph** and fast‑return a python callable that launches it.
   - Else: compile via Dynamo/AOT/Inductor and **persist artifacts** (optionally opt‑out/opt‑in per env/config).

3. **CUDA Graph capture (piecewise)**
   - For CUDA backend: allocate a **graph pool** and capture **buckets of shapes** (`cudagraph_capture_sizes`, `max_capture_size`, warmup count).
   - During **first N calls** for each bucket: run the callable inside `with torch.cuda.graph(...)` to record; subsequent calls **replay**.
   - Optional `cudagraph_copy_inputs`: pre-copy inputs to persistent buffers to guarantee **stable addresses** across replays.

**Outcome**: a python callable that, when invoked, dispatches to **(maybe piecewise) Inductor‑compiled functions** and/or **replays captured CUDA graphs** for hot shape buckets.

---

## 3) The interpreter & splitting: `PiecewiseCompileInterpreter`

**File**: `vllm/compilation/backends.py` (earlier) or near piecewise backend modules.

- Subclasses `torch.fx.Interpreter`.
- Visits modules in the FX graph; when encountering a **split boundary** (configured op names):
  - **Materialize** a sub‑`GraphModule` (`split_gm`) for the region.
  - Compute **fake inputs** consistent with the wrapper‑established dynamic dims.
  - Feed (`split_gm`, fake_inputs, pass_config) to the **CompilerInterface**.
  - Replace the region with a **call_module** to the compiled subgraph.
- Returns a new **composite GM** that alternates compiled subgraphs with eager ops (or keeps them split for CUDAGraph capture scheduling).

---

## 4) Bridging into Inductor: `CompilerInterface`

**File**: `vllm/compilation/compiler_interface.py`

Key responsibilities:

1. **Dynamo/Inductor context**
   - Ensure compile happens **inside** the same Dynamo bytecode‑compile context established by the wrapper, so symbolic shapes/guards align.
   - Provide `compile(gm, example_inputs, *, pass_manager, inductor_config, debug)` → `(compiled_graph, handle)`.

2. **Post‑grad passes**
   - After Inductor emits the FX graph (pre‑codegen), run **PostGradPassManager** over it.
   - Maintain the **pass list** (`VllmInductorPass` subclasses). Each pass receives `graph, modules, pass_config` and can mutate in place.

3. **Artifact & cache plumbing**
   - Return a **python callable** (compiled graph entry point) plus a **handle** describing caches, buckets, and symbolics.

---

## 5) The pass manager & passes

### 5.1 Post‑grad pass manager
**File**: `vllm/compilation/pass_manager.py`

- Holds `PassConfig`, a concrete ordered list of passes, and a `run(graph, modules)` API.
- Injects a **UUID seed** into Inductor code cache so pass mutations don’t alias.
- Debug knobs: dump intermediate FX, timing per pass, and per‑pass enable/disable.

### 5.2 Pass base class
**File**: `vllm/compilation/vllm_inductor_pass.py`

- `VllmInductorPass`: base with helpers for logging, timing, and writing FX dumps to a debug dir.
- `PrinterInductorPass`: optional pass that prints and returns unchanged graph (useful for debugging pipeline order/guards).

### 5.3 Selected concrete passes (common patterns)

- **`activation_quant_fusion.py`**
  - Uses `torch.fx.passes.pattern_utils` to match chains like `... → quantize_activation → custom_linear` and replace with `fused_linear_quant`.
  - Requires that graphs are **functionalized** first; vLLM enables **auto‑functionalization v2** when needed.
  - Pattern code is declarative; replacement op is a vLLM custom op registered via `torch.library`.

- **`noop_elimination.py`**
  - Peels redundant `view/reshape/slice` pairs introduced by upstream tracing/pattern lowering; essential for RMSNorm‑quant fusion to kick in.
  - Implements a simple fixed‑point rewrite until convergence; dumps before/after when debug is on.

- **`torch25_custom_graph_pass.py`** (compat)
  - For PyTorch 2.5, back‑ports 2.6’s `CustomGraphPass` semantics so the pass manager uniformity is preserved.

**Patterns across passes**
- Consistent **`apply(graph, modules, pass_config)`** signature.
- Use of `node.meta` for carry‑along shape/stride info from Dynamo.
- Defensive guards when model backends inject unsupported ops (MOE, vision branches, exotic activations).

---

## 6) CUDA piecewise backend

**Files**: `vllm/compilation/base_piecewise_backend.py`, `vllm/compilation/cuda_piecewise_backend.py` (older trees: consolidated under `backends.py`).

### 6.1 What “piecewise” means in vLLM
- vLLM splits the model graph into **pieces** around heavy/irregular ops (e.g., attention). Pieces that benefit from compilation are compiled; others stay eager or are compiled separately with different shapes.
- For CUDA, each piece may be **captured** as a CUDA graph for specific **shape buckets** (e.g., token counts, batch sizes), stored in a **dispatch table**.

### 6.2 Execution timeline
1. **First run**
   - Wrapper marks dynamic dims; backend interprets FX; for each piece, compile with Inductor; execute **eager** to populate runtime state.
   - If CUDA graphs enabled: during warmup, execute under `with torch.cuda.graph(graph)`, recording for configured sizes.

2. **Steady state**
   - Compute **bucket key** from live shapes (often a scalar like `num_tokens` after scheduler allocation).
   - If a captured graph exists: **replay**; else run compiled callable and optionally capture.

3. **Graph pool reuse**
   - Pools are reused per device/stream; addresses kept stable via persistent input/output buffers; optional `cudagraph_copy_inputs` ensures stability.

### 6.3 Why not PyTorch’s “cudagraph tree”
- vLLM performs **manual capture & dispatch**, coordinating with its scheduler and KV‑cache planner. This gives control over **which shapes** merit capture and how buffers are staged. As of mid‑2025, vLLM **does not rely** on Inductor’s built‑in cudagraphification of compiled graphs.

---

## 7) End‑to‑end call trace (first call → compiled call)

Below is the **happy path** for CUDA, piecewise level, with custom passes enabled:

1) **User code** calls into a compiled‑aware module (v1 engine path).

2) `TorchCompileWrapper.__call__`:
   - Mark dynamics; patch the inliner; call `compiled_callable` once to trigger Dynamo/AOT/Inductor.
   - Dynamo runs bytecode; after success, `bytecode_hook` receives the compiled `CodeType` and records it.

3) **Backend** (`VllmBackend.__call__`):
   - Receives the **top‑level FX GM** + `example_inputs`.
   - Runs **PiecewiseCompileInterpreter** with configured `splitting_ops`; for each split:
     - Build `split_gm`, synthesize fake inputs.
     - Call **CompilerInterface.compile** → Inductor codegen.
     - Run **PostGradPassManager** over the resulting graph.
     - Materialize a python callable for the subgraph (+ caches/handles).

4) **CUDA capture** (if enabled):
   - For configured capture sizes, run each callable once inside `torch.cuda.graph(...)` to record; place replays into a dispatch table keyed by **bucket size**.

5) **Subsequent calls**:
   - Wrapper switches to **custom dispatcher** path (bytecode‑direct) or continues calling the compiled callable.
   - Backend selects piecewise callable per split + size; if a captured replay is available, it **replays**; otherwise executes the compiled callable directly.

---

## 8) Knobs & configs you’ll see in logs

Typical compilation config (condensed):

```jsonc
{
  "level": 3,                 // PIECEWISE
  "use_inductor": true,
  "splitting_ops": [
    "vllm.unified_attention", "vllm.linear_attention", ...
  ],
  "inductor_passes": { ... },
  "cudagraph_mode": 1,        // piecewise CUDA capture
  "use_cudagraph": true,
  "cudagraph_num_of_warmups": 1,
  "cudagraph_capture_sizes": [512, 256, 128, ..., 1],
  "full_cuda_graph": false    // full-graph capture off when piecewise
}
```

---

## 9) Where to look in the tree (as of v0.10.x)

- `vllm/compilation/wrapper.py`: wrappers, bytecode hook, dispatcher.
- `vllm/compilation/compiler_interface.py`: compile bridge, pass runner, cache plumbing.
- `vllm/compilation/pass_manager.py`: PostGradPassManager (ordering, UUID, dumps, timing).
- `vllm/compilation/vllm_inductor_pass.py`: base pass infra.
- `vllm/compilation/activation_quant_fusion.py`: quant activation fuser.
- `vllm/compilation/noop_elimination.py`: reshape/slice combiner.
- `vllm/compilation/torch25_custom_graph_pass.py`: PyTorch 2.5 shim.
- `vllm/compilation/base_piecewise_backend.py`, `vllm/compilation/cuda_piecewise_backend.py`: piecewise split/compile/capture.
- Historical: `vllm/compilation/backends.py` hosted `VllmBackend`, the interpreter, and CUDA capture before splitting into more modules.

---

## 10) Gotchas you’ll trip over (and why)

- **Hangs during first compile** usually mean: guard mismatch across wrapper/inductor contexts or a long warmup due to many capture sizes.
- **Cache poison** happens when the **compile hash** doesn’t include all relevant flags; vLLM added stricter hashing and disk‑artifact caching.
- **Bytecode hook not firing** means the wrapper wasn’t the one that compiled the frame: double‑wraps, reentrant compiles, or env flags disabling the hook.
- **CUDAGraph capture end errors** come from mismatched stream/allocator state; ensure you reuse the **graph pool** and keep input addresses stable.
- **Attention fusion vs. splitting**: If you split before running fusions, you may prevent larger patterns (e.g., Attn+Q) from matching; order matters.

---

## 11) Minimal breadcrumbs (very short excerpts)

> Use these as grep anchors; do not rely on exact line numbers.

- `wrapper.py` (bytecode hook):
  ```python
  torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)
  ```

- `wrapper.py` (dynamic dims):
  ```python
  torch._dynamo.mark_dynamic(arg, dims)
  ```

- `pass_manager.py` (post-grad):
  ```python
  for p in self.passes: graph = p.apply(graph, modules, cfg)
  ```

- `vllm_inductor_pass.py` (timed pass):
  ```python
  with self.timer(): return self.run(graph, modules)
  ```

- `cuda_piecewise_backend.py` (capture):
  ```python
  with torch.cuda.graph(cg, pool=self.graph_pool): compiled(*eg_inputs)
  ```

---

## 12) Tests

Look under `tests/compile/` for unit tests of wrapper bytecode hooking, pass ordering/UUID stability, and piecewise execution paths. CI issues occasionally reference these tests when they regress.

---

## 13) FAQ

- **Does vLLM use Inductor’s built‑in cudagraph trees?**
  No—vLLM performs its own piecewise capture/replay to align with the scheduler, shape buckets, and KV‑cache layout. Inductor’s cudagraphification is intentionally bypassed for control and predictability.

- **Where do dynamic shapes get declared?**
  In the wrapper’s `__call__`, via `mark_dynamic`, prior to the first compiled run; later reused by the compiler interface.

- **Why the post‑grad pass phase?**
  vLLM prefers to mutate the FX graph **after** Inductor grad/functionalization to keep passes simple and robust, and to harmonize with custom ops/fusions.

---

### Change log footnote
- 2025‑09‑07: Initial forensic pass documented for v0.10.x era.

