"""
Master Interactive / Batch Runner for Professor Live Demo Variations
=====================================================================
Run all variations at once:
    python variations/run_all_variations.py --all

Or run a specific variation by number (1 to 8):
    python variations/run_all_variations.py --run 1
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

VARIATIONS = {
    1: (
        "v01_task1_change_ua_threshold.py",
        "Task 1: Change Target Unlearning Accuracy Threshold (tau_UA = 80%, 85%, 90%, 95%, 98%)",
    ),
    2: (
        "v02_task1_ties_merge_func_and_topk.py",
        "Task 1: Change TIES Merge Function (mean/sum/max) & Fixed vs. Adaptive top-k",
    ),
    3: (
        "v03_task1_layerwise_and_keyfilter_ablation.py",
        "Task 1: Layer-Wise Adaptive Merging (down/mid/up blocks) & Key vs. Value (to_k vs to_v)",
    ),
    4: (
        "v04_task1_threshold_formula_variants.py",
        "Task 1: Compare Mathematical Formulas for alpha_i* (L2, L1, Cosine, Linear, Logistic)",
    ),
    5: (
        "v05_task2_anchor_criterion_ablation.py",
        "Task 2: Ablate Anchor Selection Weights (w_sem, w_sim, w_cong)",
    ),
    6: (
        "v06_task2_sequence_order_permutation.py",
        "Task 2: Permute Continual Unlearning Arrival Order (Forward, Reverse, Clustered, Alternating)",
    ),
    7: (
        "v07_task2_candidate_pool_and_custom_concepts.py",
        "Task 2: Vary Candidate Anchor Pool Size (|A|=2..6) & Add Custom Concepts Live",
    ),
    8: (
        "v08_combined_task1_and_task2_pipeline.py",
        "Combined Task 1 + Task 2: End-to-End Adaptive Anchor + Adaptive Threshold Merging",
    ),
}


def run_variation(num, extra_args=None):
    script_name, desc = VARIATIONS[num]
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path] + (extra_args or [])
    print(f"\n>>> Running Variation #{num}: {desc}")
    print(f">>> Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Professor Live Demo Variations")
    parser.add_argument("--all", action="store_true", help="Run all 8 variations sequentially")
    parser.add_argument("--run", type=int, choices=list(VARIATIONS.keys()), help="Run a specific variation (1-8)")
    parser.add_argument("--list", action="store_true", help="List all available variations")
    args, extra = parser.parse_known_args()

    if args.list or (not args.all and args.run is None):
        print("=" * 88)
        print(" AVAILABLE PROFESSOR DEMO VARIATIONS (in ./variations/)")
        print("=" * 88)
        for idx, (fname, desc) in VARIATIONS.items():
            print(f"  [{idx}] variations/{fname:<47} -> {desc}")
        print("-" * 88)
        print("Usage Examples:")
        print("  python variations/run_all_variations.py --run 1")
        print("  python variations/run_all_variations.py --all")
        print("=" * 88)
        return

    if args.run is not None:
        code = run_variation(args.run, extra)
        sys.exit(code)

    if args.all:
        for idx in sorted(VARIATIONS.keys()):
            code = run_variation(idx)
            if code != 0:
                print(f"[!] Variation #{idx} exited with code {code}")
                sys.exit(code)
        print("\n[✓] All 8 variations executed successfully! Results saved in outputs/variations/")


if __name__ == "__main__":
    main()
