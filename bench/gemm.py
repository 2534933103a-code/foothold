"""Benchmark GEMM operators found in LLM inference (Q/K/V/O proj, FFN layers)."""

import torch
from bench.utils import warmup, benchmark, save_csv, check_memory, estimate_memory_gb


# Each op maps to an (M, K, N) shape derived from (batch*seq, hidden)
GEMM_OPS = {
    "q_proj":   lambda M, h: (M, h, h),        # [M, h] x [h, h]
    "k_proj":   lambda M, h: (M, h, h),        # [M, h] x [h, h]
    "v_proj":   lambda M, h: (M, h, h),        # [M, h] x [h, h]
    "o_proj":   lambda M, h: (M, h, h),        # [M, h] x [h, h]
    "ffn_up":   lambda M, h: (M, h, h * 4),    # [M, h] x [h, 4h]
    "ffn_gate": lambda M, h: (M, h, h * 4),    # [M, h] x [h, 4h]
    "ffn_down": lambda M, h: (M, h * 4, h),    # [M, 4h] x [4h, h]
}


def bench_gemm(config):
    """Run GEMM benchmarks over the Cartesian product of config parameters."""
    dtype = getattr(torch, config["dtype"])
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]
    max_mem = config["max_memory_gb"]
    device = torch.device("cuda")

    results = []
    for b in config["batch_sizes"]:
        for s in config["seq_lens"]:
            for h in config["hidden_dims"]:
                M = b * s

                # Rough memory guard: ~3x activation for input + weight + output
                act_gb = estimate_memory_gb(b, s, h)
                if not check_memory(act_gb * 3, max_mem):
                    print(f"  [skip] b={b} s={s} h={h} — OOM risk")
                    continue

                for op_name, shape_fn in GEMM_OPS.items():
                    m_val, k_val, n_val = shape_fn(M, h)

                    a = torch.randn(m_val, k_val, dtype=dtype, device=device)
                    w = torch.randn(k_val, n_val, dtype=dtype, device=device)

                    def mm(a=a, w=w):
                        torch.mm(a, w)

                    warmup(mm, warmup_iters)
                    avg_ms = benchmark(mm, bench_iters)

                    results.append({
                        "op_name": op_name,
                        "b": b,
                        "s": s,
                        "h": h,
                        "M": m_val,
                        "K": k_val,
                        "N": n_val,
                        "time_ms": f"{avg_ms:.6f}",
                    })

                    print(f"  {op_name:10s} b={b} s={s} h={h}  "
                          f"M={m_val:6d} K={k_val:6d} N={n_val:6d}  "
                          f"{avg_ms:8.4f} ms")

                    del a, w

    save_csv(results, "results/gemm.csv")
    return results
