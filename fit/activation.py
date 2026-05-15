import numpy as np
from fit.utils import _get, lstsq_fit, lstsq_log_fit


def fit_activation(results):
    """Fit activation / pointwise operators."""
    print("\n" + "=" * 60)
    print("Activation Operators")
    print("=" * 60)

    op_configs = {
        "swiglu":       {"desc": "b * s * 4h"},
        "rope":         {"desc": "b * n_heads * s * d_head"},
        "residual_add": {"desc": "b * s * h"},
        "causal_mask":  {"desc": "b * n_heads * s^2"},
    }

    for op_name, cfg in op_configs.items():
        op_results = [r for r in results if r["op_name"] == op_name]
        if not op_results:
            continue
        b = np.array([r["b"] for r in op_results], dtype=np.float64)
        s = np.array([r["s"] for r in op_results], dtype=np.float64)
        nh = np.array([_get(r, "num_heads", "nh") for r in op_results], dtype=np.float64)
        hd = np.array([_get(r, "head_dim", "hd") for r in op_results], dtype=np.float64)
        h = np.array([r["h"] for r in op_results], dtype=np.float64)
        time_ms = np.array([r["time_ms"] for r in op_results])

        if op_name == "swiglu":
            work = b * s * h * 4
        elif op_name == "rope":
            work = b * nh * s * hd
        elif op_name == "residual_add":
            work = b * s * h
        elif op_name == "causal_mask":
            work = b * nh * s * s

        a, b0, r2 = lstsq_fit(work, time_ms)
        c1, _, r2_log = lstsq_log_fit(work, time_ms)

        total_ms = np.sum(time_ms)

        # Bytes moved (fp16=2 bytes per element)
        if op_name == "swiglu":
            bytes_moved = 2 * np.sum(3 * work)
            flops = np.sum(5 * work)
        elif op_name == "rope":
            bytes_moved = 2 * np.sum(2 * work)
            flops = np.sum(5 * work)
        elif op_name == "residual_add":
            bytes_moved = 2 * np.sum(3 * work)
            flops = np.sum(work)
        elif op_name == "causal_mask":
            bytes_moved = 2 * np.sum(2 * work)
            flops = np.sum(work)

        avg_tflops = (flops / (total_ms / 1000)) / 1e12 if total_ms > 0 else 0
        bw_gbps = bytes_moved / 1e9 / (total_ms / 1000) if total_ms > 0 else 0

        print(f"\n{op_name}")
        print(f"  work = {cfg['desc']}")
        print(f"  time = {a:.3e} * work + {b0:.4f}   R2={r2:.4f}")
        print(f"  power-law exponent: {c1:.3f}   R2_log={r2_log:.4f}")
        print(f"  effective TFLOPS: {avg_tflops:.1f}")
        print(f"  effective bandwidth: {bw_gbps:.0f} GB/s")
