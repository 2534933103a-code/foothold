"""Benchmark attention sub-operators: QK^T matmul, softmax, score x V matmul."""

import torch
import torch.nn.functional as F
from bench.utils import warmup, benchmark, save_csv, check_memory, estimate_memory_gb


def bench_attention(config):
    """Benchmark attention sub-ops for each [b, s, h] combination."""
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    num_heads = config["num_heads"]
    device = torch.device("cuda")

    results = []
    for b in config["batch_sizes"]:
        for s in config["seq_lens"]:
            for h in config["hidden_dims"]:
                head_dim = h // num_heads

                # Memory guard: Q/K/V + score matrix ~ s*s per head
                act_gb = estimate_memory_gb(b, s, h) * 3
                score_gb = estimate_memory_gb(b * num_heads, s, s)
                if not check_memory(act_gb + score_gb, max_mem):
                    print(f"  [skip] b={b} s={s} h={h} — OOM risk")
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
                print(f"  {'qk_matmul':10s} b={b} s={s} h={h}  "
                      f"heads={num_heads} d_head={head_dim}  {ms:8.4f} ms")

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
                print(f"  {'softmax':10s} b={b} s={s} h={h}  "
                      f"shape=[{b},{num_heads},{s},{s}]  {ms:8.4f} ms")

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
                print(f"  {'score_v_matmul':10s} b={b} s={s} h={h}  "
                      f"heads={num_heads} d_head={head_dim}  {ms:8.4f} ms")

                del Q, K, V, scores

    save_csv(results, "results/attention.csv")
    return results
