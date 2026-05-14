#!/usr/bin/env python3
"""Run all operator benchmarks and produce aggregated results.

Usage:
    python run_all.py                        # uses config/default.yaml
    python run_all.py --config my_conf.yaml  # custom config
"""

import argparse
import os
import sys
import time

import torch
import yaml

from bench.gemm import bench_gemm
from bench.attention import bench_attention
from bench.norm import bench_norm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LLM Operator Benchmark Suite")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    cfg = load_config(args.config)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GiB)")
    print(f"Config: {args.config}")
    print(f"dtype: {cfg['dtype']}, warmup={cfg['warmup_iters']}, "
          f"iters={cfg['bench_iters']}")
    print(f"Shapes: b ∈ {cfg['batch_sizes']}, s ∈ {cfg['seq_lens']}, "
          f"h ∈ {cfg['hidden_dims']}")
    print(f"Total combinations: "
          f"{len(cfg['batch_sizes']) * len(cfg['seq_lens']) * len(cfg['hidden_dims'])}")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    all_results = []

    t0 = time.perf_counter()
    print("\n[1/3] GEMM benchmarks")
    gemm_results = bench_gemm(cfg)
    all_results.extend(gemm_results)

    print("\n[2/3] Attention benchmarks")
    attn_results = bench_attention(cfg)
    all_results.extend(attn_results)

    print("\n[3/3] Norm benchmarks")
    norm_results = bench_norm(cfg)
    all_results.extend(norm_results)

    # Aggregate all results into a single CSV
    from bench.utils import save_csv
    save_csv(all_results, "results/all_operators.csv")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s — {len(all_results)} rows saved to results/")
    print("Output files: results/gemm.csv, results/attention.csv, results/norm.csv, results/all_operators.csv")


if __name__ == "__main__":
    main()
