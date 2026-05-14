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
from tqdm import tqdm

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
    parser.add_argument(
        "--output", default="results",
        help="Directory to save output xlsx files",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    cfg = load_config(args.config)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GiB)")
    print(f"Config: {args.config}")
    print(f"dtype: {cfg['dtype']}, warmup={cfg['warmup_iters']}, "
          f"iters={cfg['bench_iters']}")
    print(f"Shapes: b ∈ {cfg['batch_sizes']}, s ∈ {cfg['seq_lens']}, "
          f"h ∈ {cfg['hidden_dims']}")
    print(f"Total combinations: "
          f"{len(cfg['batch_sizes']) * len(cfg['seq_lens']) * len(cfg['hidden_dims'])}")
    print("=" * 70)

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    
    gemm_xlsx = os.path.join(out_dir, "gemm.xlsx")
    attention_xlsx = os.path.join(out_dir, "attention.xlsx")
    norm_xlsx = os.path.join(out_dir, "norm.xlsx")
    all_xlsx = os.path.join(out_dir, "all_operators.xlsx")

    all_results = []

    t0 = time.perf_counter()

    phases = [
        ("GEMM", bench_gemm, cfg, gemm_xlsx),
        ("Attention", bench_attention, cfg, attention_xlsx),
        ("Norm", bench_norm, cfg, norm_xlsx),
    ]

    for name, bench_fn, bench_cfg, bench_xlsx in tqdm(phases, desc="Overall", unit="phase"):
        tqdm.write(f"\n[{name}]")
        results = bench_fn(bench_cfg, output_path=bench_xlsx)
        all_results.extend(results)
        torch.cuda.empty_cache()

    # Aggregate all results into a single xlsx
    from bench.utils import save_xlsx

    save_xlsx(all_results, all_xlsx)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s — {len(all_results)} rows saved.")
    print(f"Output files: {gemm_xlsx}, {attention_xlsx}, {norm_xlsx}, {all_xlsx}")


if __name__ == "__main__":
    main()
