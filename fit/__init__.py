from fit.utils import load_results
from fit.gemm import fit_gemm
from fit.attention import fit_attention
from fit.norm import fit_norm


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Fit performance models to benchmark results")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Directory containing xlsx result files")
    args = parser.parse_args()

    all_path = os.path.join(args.results_dir, "all_operators.xlsx")
    if not os.path.exists(all_path):
        print(f"Not found: {all_path}")
        return

    results = load_results(all_path)
    if not results:
        print("No valid results to fit.")
        return

    print(f"Loaded {len(results)} rows from {all_path}")
    fit_gemm(results)
    fit_attention(results)
    fit_norm(results)
