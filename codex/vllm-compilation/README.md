# vLLM Torch Compile Internals

This document traces vLLM's customization of `torch.compile`, from the first
graph transformation through dispatch of the compiled callable.  Each section
references the implementation with file paths and line numbers inside the
repository.

## Pass Management

`PostGradPassManager` coordinates the post-autograd passes executed after
Inductor's own lowering.  It keeps an ordered list of `VllmInductorPass`
instances and applies them before finally running `FixFunctionalizationPass`
so all subsequent passes operate on a defunctionalized graph
[`vllm/compilation/pass_manager.py`]
lines 27-88【F:vllm/compilation/pass_manager.py†L27-L88】.

```python
class PostGradPassManager(CustomGraphPass):
    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)
        # always run fix_functionalization last
        self.fix_functionalization(graph)
```

The `configure` method adds optional passes based on
`CompilationConfig.pass_config`, enabling features such as
sequence-parallelism, asynchronous tensor-parallel communication,
fusions and attention-specific rewrites
lines 54-72【F:vllm/compilation/pass_manager.py†L54-L72】.

## Base Inductor Pass Infrastructure

`InductorPass` abstracts a custom FX pass with a stable UUID derived from the
source of the pass.  It also exposes `pass_context` so passes can specialize on
runtime shapes.  FakeTensorMode wrappers allow the passes to run during
`torch.compile` without materializing real tensors
[`vllm/compilation/inductor_pass.py`]
lines 41-135【F:vllm/compilation/inductor_pass.py†L41-L135】.

## Selected Passes

### FixFunctionalizationPass
Defunctionalizes FX nodes produced by `auto_functionalized` to remove redundant
copies.  It pattern‑matches specific custom ops (e.g. fused RMSNorm variants,
silu‑and‑mul quant) and replaces them with calls to the underlying kernel while
splicing mutated arguments back into the graph
[`vllm/compilation/fix_functionalization.py`]
lines 20-124【F:vllm/compilation/fix_functionalization.py†L20-L124】.

### NoOpEliminationPass
Eliminates chains of reshapes, redundant reshapes that yield the same shape and
slice/slice_scatter pairs that read and write full tensors
[`vllm/compilation/noop_elimination.py`]
lines 16-90【F:vllm/compilation/noop_elimination.py†L16-L90】.

### FusionPass & ActivationQuantFusionPass
`FusionPass` holds reusable pattern‑matcher utilities for fusion.  Specialized
passes such as `ActivationQuantFusionPass` register graph rewrite patterns for
quantized activations (e.g. `silu_and_mul` + quant) to reduce kernel launches.
(see `vllm/compilation/fusion.py` & `activation_quant_fusion.py`).

### AttnFusionPass
Provides pattern classes that fuse the result of unified attention with a
quantization op by rewriting the FX graph so the attention kernel directly
produces quantized output.  Example patterns: `AttentionFp8StaticQuantPattern`
and `AttentionNvfp4QuantPattern`
[`vllm/compilation/fusion_attn.py`]
lines 32-119【F:vllm/compilation/fusion_attn.py†L32-L119】.

### Collective & Sequence Parallelism Passes
`AllReduceFusionPass` and `AsyncTPPass` rewrite distributed collectives to use
fused all‑reduce kernels and asynchronous tensor‑parallel communication.  The
`SequenceParallelismPass` rewrites subgraphs to shard sequences across devices
(see `vllm/compilation/collective_fusion.py` & `sequence_parallelism.py`).

## Backends and Compilation Flow

`CompilerManager` orchestrates compilation and caching.  It first attempts to
load previously compiled artifacts, otherwise dispatches to `CompilerInterface`
(`InductorAdaptor` or `EagerAdaptor`) to compile and store artifacts indexed by
runtime shape and subgraph
[`vllm/compilation/backends.py`]
lines 46-199【F:vllm/compilation/backends.py†L46-L199】.

The user‑facing entry points create a backend via
`make_compiler`; for Inductor this hooks our `PostGradPassManager` so the custom
passes run after autograd lowering.

## torch.compile Wrapper

`TorchCompileWrapperWithCustomDispatcher` wraps `torch.compile` and registers a
bytecode hook that collects the transformed code.  With the
`dispatch_to_code` context manager it can swap bytecodes to bypass Dynamo's
runtime guards when `CompilationLevel.DYNAMO_ONCE` is selected
[`vllm/compilation/wrapper.py`]
lines 21-138【F:vllm/compilation/wrapper.py†L21-L138】.

## Piecewise CUDA Graph Backend

`PiecewiseBackend` enables piecewise graph compilation: after capturing a
general‑shape graph, it lazily compiles specific runtime shapes on demand and
caches the resulting callables.  Once all requested shapes are compiled it saves
artifacts and stops monitoring
[`vllm/compilation/cuda_piecewise_backend.py`]
lines 25-117【F:vllm/compilation/cuda_piecewise_backend.py†L25-L117】.

`CUDAGraphWrapper` adds optional CUDA graph capture.  At runtime it consults the
forward context for the desired `CUDAGraphMode` and `BatchDescriptor`, captures
new graphs when necessary, and replays cached graphs while validating tensor
addresses in debug mode
[`vllm/compilation/cuda_graph.py`]
lines 41-193【F:vllm/compilation/cuda_graph.py†L41-L193】.

## Minimal Piecewise Example

The tests contain a simple model that exercises piecewise compilation and
CUDAGraph capture:
[`tests/compile/piecewise/test_simple.py`]
lines 1-84【F:tests/compile/piecewise/test_simple.py†L1-L84】.

## Related Issues & PRs

* Piecewise cudagraph & full graph support – PR [#20059](https://github.com/vllm-project/vllm/pull/20059)
* Inductor standalone adaptor and xpu support – PR [#22609](https://github.com/vllm-project/vllm/pull/22609)
* NvFP4 quantization fusion – PR [#23671](https://github.com/vllm-project/vllm/pull/23671)
* Async tensor-parallel and collective fusion – PR [#23639](https://github.com/vllm-project/vllm/pull/23639)

