"""Benchmark activation / pointwise operators: SwiGLU, RoPE, residual add, causal mask."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory, estimate_memory_gb


def _rope(x, cos, sin):
    """Apply rotary position embedding: x*cos + rotate_half(x)*sin."""
    d2 = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)
    return x * cos + x_rot * sin


def bench_activation(config, output_path="results/activation.xlsx"):
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    from itertools import product

    results = []
    combos = list(product(
        config["batch_sizes"], config["seq_lens"],
        config["hidden_dims"], config["num_heads"],
    ))
    for b, s, h, nh in tqdm(combos, desc="Activation"):
        if h % nh != 0:
            continue
        head_dim = h // nh
        inter_dim = h * 4

        # Memory guard: worst-case across all sub-ops
        gate_up_gb = estimate_memory_gb(b, s, inter_dim) * 3
        rope_gb = estimate_memory_gb(b * nh, s, head_dim) * 2
        residual_gb = estimate_memory_gb(b, s, h) * 3
        score_gb = estimate_memory_gb(b * nh, s, s) * 2
        oom = not check_memory(max(gate_up_gb, rope_gb, residual_gb, score_gb), max_mem)
        if oom:
            print(f"  [OOM] b={b} s={s} h={h} nh={nh}")
            for op in ("swiglu", "rope", "residual_add", "causal_mask"):
                results.append({
                    "op_name": op, "b": b, "s": s, "h": h,
                    "num_heads": nh, "head_dim": head_dim, "time_ms": "OOM",
                })
            continue

        # --- SwiGLU: silu(gate) * up ---
        gate = torch.randn(b, s, inter_dim, dtype=dtype, device=device)
        up = torch.randn(b, s, inter_dim, dtype=dtype, device=device)

        def swiglu_fn(gate=gate, up=up):
            F.silu(gate, inplace=False) * up

        warmup(swiglu_fn, warmup_iters)
        ms = benchmark(swiglu_fn, bench_iters)
        results.append({
            "op_name": "swiglu", "b": b, "s": s, "h": h,
            "num_heads": nh, "head_dim": head_dim, "time_ms": f"{ms:.6f}",
        })
        del gate, up

        # --- RoPE on Q ---
        Q = torch.randn(b, nh, s, head_dim, dtype=dtype, device=device)
        cos = torch.randn(s, head_dim, dtype=dtype, device=device)
        sin = torch.randn(s, head_dim, dtype=dtype, device=device)

        def rope_fn(Q=Q, cos=cos, sin=sin):
            _rope(Q, cos, sin)

        warmup(rope_fn, warmup_iters)
        ms = benchmark(rope_fn, bench_iters)
        results.append({
            "op_name": "rope", "b": b, "s": s, "h": h,
            "num_heads": nh, "head_dim": head_dim, "time_ms": f"{ms:.6f}",
        })
        del Q, cos, sin

        # --- Residual add: x + y ---
        x_res = torch.randn(b, s, h, dtype=dtype, device=device)
        y_res = torch.randn(b, s, h, dtype=dtype, device=device)

        def residual_add_fn(x=x_res, y=y_res):
            x + y

        warmup(residual_add_fn, warmup_iters)
        ms = benchmark(residual_add_fn, bench_iters)
        results.append({
            "op_name": "residual_add", "b": b, "s": s, "h": h,
            "num_heads": nh, "head_dim": head_dim, "time_ms": f"{ms:.6f}",
        })
        del x_res, y_res

        # --- Causal mask: scores + mask ---
        scores = torch.randn(b, nh, s, s, dtype=dtype, device=device)
        mask = torch.triu(
            torch.ones(s, s, dtype=dtype, device=device) * float("-inf"),
            diagonal=1,
        )

        def causal_mask_fn(scores=scores, mask=mask):
            scores + mask

        warmup(causal_mask_fn, warmup_iters)
        ms = benchmark(causal_mask_fn, bench_iters)
        results.append({
            "op_name": "causal_mask", "b": b, "s": s, "h": h,
            "num_heads": nh, "head_dim": head_dim, "time_ms": f"{ms:.6f}",
        })
        del scores, mask

    if output_path:
        save_xlsx(results, output_path)
    return results
