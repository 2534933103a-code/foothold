"""Benchmark utilities: CUDA timer, warmup, CSV output."""

import csv
import torch


class CudaTimer:
    """Precise GPU timer using CUDA events."""

    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *args):
        self._end.record()
        torch.cuda.synchronize()

    @property
    def elapsed_ms(self):
        return self._start.elapsed_time(self._end)


def warmup(fn, iters=5):
    """Warm up GPU by running fn several times."""
    for _ in range(iters):
        fn()


def benchmark(fn, iters=100):
    """Return average execution time of fn in milliseconds."""
    timer = CudaTimer()
    total = 0.0
    for _ in range(iters):
        with timer:
            fn()
        total += timer.elapsed_ms
    return total / iters


def save_csv(results, path):
    """Save list of dicts to CSV. Creates parent directories if needed."""
    if not results:
        print(f"  [skip] No results to save for {path}")
        return
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved {len(results)} rows → {path}")


def estimate_memory_gb(b, s, h, dtype_size=2):
    """Estimate GPU memory (GiB) for a [b, s, h] fp16 tensor."""
    return (b * s * h * dtype_size) / (1024**3)


def check_memory(required_gb, max_gb=7.5):
    """Check if required_gb fits in available and allowed memory."""
    free_gb, total_gb = torch.cuda.mem_get_info()
    free_gb = free_gb / (1024**3)
    return required_gb < free_gb and required_gb <= max_gb
