vLLM Torch Compile Integration — Trace and Custom Passes

This folder documents, function-by-function and line-by-line, how vLLM integrates with torch.compile to optimize inference, including:

- The custom backend (`VllmBackend`) that vLLM passes to `torch.compile`
- How the FX graph is split and compiled piecewise
- How vLLM installs and executes custom Inductor passes
- How piecewise and full CUDA Graphs are captured and dispatched

Files

- overview.md — end‑to‑end flow from first compile to compiled call
- backends.md — `VllmBackend`, `CompilerManager`, splitting, piecewise compile interpreter
- passes.md — all custom Inductor passes in vLLM with traces
- cudagraphs.md — CUDAGraph integration (piecewise and full) and dispatch
- references.md — upstream issues/PRs related to these implementations

