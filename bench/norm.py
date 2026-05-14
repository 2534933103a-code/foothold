"""Benchmark normalization operators: LayerNorm and RMSNorm."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory, estimate_memory_gb


class RMSNormFn(torch.autograd.Function):
    """RMSNorm forward (no backward needed for benchmarking)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        return x / rms * weight


def rms_norm(x, weight, eps=1e-6):
    return RMSNormFn.apply(x, weight, eps)


def bench_norm(config, output_path="results/norm.xlsx"):
    """Benchmark LayerNorm and RMSNorm for each [b, s, h]."""
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    from itertools import product

    results = []
    combos = list(product(config["batch_sizes"], config["seq_lens"], config["hidden_dims"]))
    for b, s, h in tqdm(combos, desc="Norm"):
        act_gb = estimate_memory_gb(b, s, h)
        oom = not check_memory(act_gb * 2, max_mem)
        if oom:
            print(f"  [OOM] b={b} s={s} h={h}")
            for op in ("layernorm", "rmsnorm"):
                results.append({
                    "op_name": op,
                    "b": b, "s": s, "h": h,
                    "time_ms": "OOM",
                })
            continue

        x = torch.randn(b, s, h, dtype=dtype, device=device)
        w = torch.randn(h, dtype=dtype, device=device)
        b_norm = torch.randn(h, dtype=dtype, device=device)

        # --- LayerNorm ---
        def layernorm_fn(x=x, w=w, b_norm=b_norm):
            F.layer_norm(x, (h,), w, b_norm, 1e-5)

        warmup(layernorm_fn, warmup_iters)
        ms = benchmark(layernorm_fn, bench_iters)
        results.append({
            "op_name": "layernorm",
            "b": b, "s": s, "h": h,
            "time_ms": f"{ms:.6f}",
        })
        # --- RMSNorm ---
        def rmsnorm_fn(x=x, w=w):
            rms_norm(x, w, 1e-6)

        warmup(rmsnorm_fn, warmup_iters)
        ms = benchmark(rmsnorm_fn, bench_iters)
        results.append({
            "op_name": "rmsnorm",
            "b": b, "s": s, "h": h,
            "time_ms": f"{ms:.6f}",
        })
        del x, w, b_norm

    if output_path:
        save_xlsx(results, output_path)
    return results
