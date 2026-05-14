# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install dependencies (PyTorch CUDA 12.8 + numpy + pyyaml)
uv run python run_all.py                     # Run all benchmarks with config/default.yaml
uv run python run_all.py --config config/custom.yaml  # Run with custom config
```

No linter, formatter, or test runner is currently configured.

## Architecture

Three benchmark modules in [`bench/`](bench/) — each follows the same pattern: iterate over the Cartesian product of `batch_sizes × seq_lens × hidden_dims` from config, guard against OOM with `check_memory()`, warm up, then measure 100 iterations with `CudaTimer`. Each module writes its own CSV and returns results for aggregation.

- [`bench/utils.py`](bench/utils.py) — `CudaTimer` (CUDA event-based GPU timing via context manager), `warmup()` / `benchmark()` / `save_csv()` / `estimate_memory_gb()` / `check_memory()`. Every benchmark module imports from here.
- [`bench/gemm.py`](bench/gemm.py) — Q/K/V/O projection and FFN up/gate/down as `torch.mm`. Shape functions in `GEMM_OPS` dict map `(M, h) → (M, K, N)`.
- [`bench/attention.py`](bench/attention.py) — QK^T matmul (`torch.bmm`), softmax (`F.softmax`), score×V matmul. Uses multi-head layout `[b, n_heads, s, head_dim]` where `head_dim = h // num_heads`.
- [`bench/norm.py`](bench/norm.py) — `F.layer_norm` and a custom `RMSNormFn` (autograd Function for forward-only RMSNorm). Both normalize over the last dimension `[h]`.
- [`run_all.py`](run_all.py) — CLI entry point. Sequentially calls the three bench modules, clears CUDA cache between them, aggregates all results into `results/all_operators.csv`.

## Memory model

`check_memory()` checks both `torch.cuda.mem_get_info()` free memory and the config's `max_memory_gb` cap. Each benchmark module estimates its own activation footprint with multipliers tuned per operator type (3× for GEMM, 3× + score matrix for attention, 2× for norm). The `max_memory_gb` cap exists to protect 8GB GPUs — raise it for larger cards.
