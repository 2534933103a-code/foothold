"""Benchmark attention sub-operators: QK^T matmul, softmax, score x V matmul."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bench.utils import warmup, benchmark, save_xlsx, check_memory, estimate_memory_gb


def bench_attention(config, output_path="results/attention.xlsx"):
    """Benchmark attention sub-ops for each [b, s, h] combination."""
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    num_heads = config["num_heads"]
    device = torch.device("cuda")

    from itertools import product

    results = []
    combos = list(product(config["batch_sizes"], config["seq_lens"], config["hidden_dims"]))
    for b, s, h in tqdm(combos, desc="Attention"):
        head_dim = h // num_heads

        # Memory guard: Q/K/V + score matrix ~ s*s per head
        act_gb = estimate_memory_gb(b, s, h) * 3
        score_gb = estimate_memory_gb(b * num_heads, s, s)
        oom = not check_memory(act_gb + score_gb, max_mem)
        if oom:
            print(f"  [OOM] b={b} s={s} h={h}")
            for op in ("qk_matmul", "softmax", "score_v_matmul"):
                results.append({
                    "op_name": op,
                    "b": b, "s": s, "h": h,
                    "num_heads": num_heads, "head_dim": head_dim,
                    "time_ms": "OOM",
                })
            continue

        # Build tensors in attention layout: [b, num_heads, s, head_dim]
        Q = torch.randn(b, num_heads, s, head_dim, dtype=dtype, device=device)
        K = torch.randn(b, num_heads, s, head_dim, dtype=dtype, device=device)
        V = torch.randn(b, num_heads, s, head_dim, dtype=dtype, device=device)

        # --- Q @ K^T ---
        def qk_matmul(Q=Q, K=K):
            torch.bmm(
                Q.view(b * num_heads, s, head_dim),
                K.view(b * num_heads, s, head_dim).transpose(1, 2),
            )

        warmup(qk_matmul, warmup_iters)
        ms = benchmark(qk_matmul, bench_iters)
        results.append({
            "op_name": "qk_matmul",
            "b": b, "s": s, "h": h,
            "num_heads": num_heads, "head_dim": head_dim,
            "time_ms": f"{ms:.6f}",
        })
        # --- Softmax ---
        scores = torch.randn(
            b, num_heads, s, s, dtype=dtype, device=device
        )

        def softmax_fn(scores=scores):
            F.softmax(scores, dim=-1)

        warmup(softmax_fn, warmup_iters)
        ms = benchmark(softmax_fn, bench_iters)
        results.append({
            "op_name": "softmax",
            "b": b, "s": s, "h": h,
            "num_heads": num_heads, "head_dim": head_dim,
            "time_ms": f"{ms:.6f}",
        })
        # --- Score @ V ---
        def score_v_matmul(scores=scores, V=V):
            torch.bmm(
                scores.view(b * num_heads, s, s),
                V.view(b * num_heads, s, head_dim),
            )

        warmup(score_v_matmul, warmup_iters)
        ms = benchmark(score_v_matmul, bench_iters)
        results.append({
            "op_name": "score_v_matmul",
            "b": b, "s": s, "h": h,
            "num_heads": num_heads, "head_dim": head_dim,
            "time_ms": f"{ms:.6f}",
        })
        del Q, K, V, scores

    if output_path:
        save_xlsx(results, output_path)
    return results
