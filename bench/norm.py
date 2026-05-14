"""Benchmark normalization operators: LayerNorm and RMSNorm."""

import torch
import torch.nn.functional as F
from bench.utils import warmup, benchmark, save_csv, check_memory, estimate_memory_gb


class RMSNormFn(torch.autograd.Function):
    """RMSNorm forward (no backward needed for benchmarking)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        return x / rms * weight


def rms_norm(x, weight, eps=1e-6):
    return RMSNormFn.apply(x, weight, eps)


def bench_norm(config):
    """Benchmark LayerNorm and RMSNorm for each [b, s, h]."""
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    results = []
    for b in config["batch_sizes"]:
        for s in config["seq_lens"]:
            for h in config["hidden_dims"]:
                act_gb = estimate_memory_gb(b, s, h)
                if not check_memory(act_gb * 2, max_mem):
                    print(f"  [skip] b={b} s={s} h={h} — OOM risk")
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
                print(f"  {'layernorm':10s} b={b} s={s} h={h}  {ms:8.4f} ms")

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
                print(f"  {'rmsnorm':10s} b={b} s={s} h={h}  {ms:8.4f} ms")

                del x, w, b_norm

    save_csv(results, "results/norm.csv")
    return results
