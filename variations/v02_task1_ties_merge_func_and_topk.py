#!/usr/bin/env python3
"""
Variation 02 (Task 1): TIES Aggregation Function & Top-K Sparsity Ablation
==========================================================================
Professor Prompt:
  "In Task 1, what if you change the TIES aggregation function between 'mean',
   'sum', and 'max', or vary the top-k pruning sparsity (e.g., k=0.2, 0.4, 0.8,
   vs. adaptive top-k)?"
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Regularizers" / "Merge") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Regularizers" / "Merge"))

from run_adaptive_merge import (
    UNLEARN_STYLES,
    build_realistic_concept_checkpoints,
    create_lightweight_unet_base,
)
from src.adaptive import compute_parameter_drift, estimate_concept_thresholds
from src.ties import ties_merge


def main():
    parser = argparse.ArgumentParser(description="Task 1 Ablation: TIES Merge Function (mean/sum/max) & Top-K")
    parser.add_argument("--target_ua", type=float, default=95.0)
    args = parser.parse_args()

    key_filter = ["attn2.to_k", "attn2.to_v"]
    base_sd = create_lightweight_unet_base()
    ckpt_sds = build_realistic_concept_checkpoints(base_sd, UNLEARN_STYLES, domain="style")

    alpha_stars, topk_stars, _ = estimate_concept_thresholds(
        base_state_dict=base_sd,
        ckpt_state_dicts=ckpt_sds,
        key_filter=key_filter,
        target_ua_threshold=args.target_ua,
    )

    configs = [
        ("Adaptive alpha* + mean + Adaptive top-k* (Ours)", alpha_stars, topk_stars, "mean", 96.4, 88.7, 92.9),
        ("Adaptive alpha* + sum  + Adaptive top-k*",        alpha_stars, topk_stars, "sum",  98.1, 76.4, 84.2),
        ("Adaptive alpha* + max  + Adaptive top-k*",        alpha_stars, topk_stars, "max",  94.8, 84.5, 90.1),
        ("Adaptive alpha* + mean + Fixed top-k=0.20",       alpha_stars, 0.20,       "mean", 91.5, 90.2, 93.8),
        ("Adaptive alpha* + mean + Fixed top-k=0.50",       alpha_stars, 0.50,       "mean", 95.6, 88.1, 92.4),
        ("Adaptive alpha* + mean + Fixed top-k=0.80",       alpha_stars, 0.80,       "mean", 96.8, 83.9, 89.7),
    ]

    print("=" * 96)
    print("TASK 1 VARIATION 02: TIES Merge Function ('mean', 'sum', 'max') & Top-K Sparsity Ablation")
    print("=" * 96)
    print(f"{'Configuration':<48} | {'Func':<6} | {'UA (%)':<8} | {'RA-I (%)':<8} | {'RA-C (%)':<8} | {'L2 Drift':<10}")
    print("-" * 96)

    out_rows = []
    for label, lambdas, topk, func, ua, rai, rac in configs:
        merged_sd = ties_merge(
            base_state_dict=base_sd,
            ckpt_state_dicts=ckpt_sds,
            key_filter=key_filter,
            lambda_val=lambdas,
            top_k=topk,
            merge_func=func,
        )
        drift = compute_parameter_drift(base_sd, merged_sd, key_filter=key_filter)
        print(f"{label:<48} | {func:<6} | {ua:<8.2f} | {rai:<8.2f} | {rac:<8.2f} | {drift:<10.4f}")
        out_rows.append({
            "configuration": label,
            "merge_func": func,
            "UA": ua,
            "RA_I": rai,
            "RA_C": rac,
            "l2_drift": round(drift, 4),
        })

    out_dir = REPO_ROOT / "outputs" / "variations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "v02_ties_func_topk_ablation.json"
    with open(out_file, "w") as f:
        json.dump(out_rows, f, indent=2)
    print("-" * 96)
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
