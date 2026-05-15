import numpy as np
from fit.utils import lstsq_fit, lstsq_log_fit


def fit_norm(results):
    """Fit normalization operators: work = b*s*h (elements)."""
    print("\n" + "=" * 60)
    print("Norm Operators")
    print("=" * 60)

    for op_name in ["layernorm", "rmsnorm"]:
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            continue
        work = np.array([r["b"] * r["s"] * r["h"] for r in op_results])
        time_ms = np.array([r["time_ms"] for r in op_results])

        a, b, r2 = lstsq_fit(work, time_ms)
        c1, _, r2_log = lstsq_log_fit(work, time_ms)

        total_ms = np.sum(time_ms)
        bw_gbps = np.sum(2 * work) / 1e9 / (total_ms / 1000) if total_ms > 0 else 0
        # norm: ~5 FLOPs per element (mean, sub, square, rsqrt, mul, add)
        flops = np.sum(5 * work)
        avg_tflops = (flops / (total_ms / 1000)) / 1e12 if total_ms > 0 else 0

        print(f"\n{op_name}")
        print(f"  work = b*s*h")
        print(f"  time = {a:.3e} * work + {b:.4f}   R2={r2:.4f}")
        print(f"  power-law exponent: {c1:.3f}   R2_log={r2_log:.4f}")
        print(f"  effective TFLOPS: {avg_tflops:.1f}")
        print(f"  effective bandwidth: {bw_gbps:.0f} GB/s")
