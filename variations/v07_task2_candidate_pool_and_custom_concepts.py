#!/usr/bin/env python3
"""
Variation 07 (Task 2): Candidate Anchor Pool Size & Custom Target Concept Injection
===================================================================================
Professor Prompt:
  "What happens if the candidate anchor pool is smaller (e.g., only 2 anchors) vs.
   larger (e.g., 6 anchors)? Or what if I give you a brand-new set of concepts right
   now (like 'Surrealism', 'Monet', 'Pop_Art', or 'Dogs', 'Cats', 'Rabbits')?"
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Regularizers.Anchor.anchor_selection import TARGET_CONCEPT_DESCRIPTIONS
from run_adaptive_anchor import build_semantic_embeddings, run_strategy_evaluation


def main():
    parser = argparse.ArgumentParser(description="Task 2 Ablation: Candidate Anchor Pool Size & Custom Concepts")
    parser.add_argument("--custom_targets", type=str, nargs="*",
                        default=["Picasso", "Van_Gogh", "Cartoon", "Ukiyoe", "Neon_Lines", "Magic_Cube"],
                        help="Pass any custom target concepts to test live during the demo!")
    args = parser.parse_args()

    all_anchors = [
        "Classic_Painting",
        "Line_Illustration",
        "Geometric_Design",
        "Atmospheric_Landscape",
        "Realism_Photo",
        "object",
    ]
    targets = args.custom_targets

    embs = build_semantic_embeddings(
        concepts=list(set(targets + all_anchors)),
        descriptions=TARGET_CONCEPT_DESCRIPTIONS,
        base_model_dir="Checkpoints/Generators/UnlearnCanvas",
        device="cpu",
        use_clip=False,
    )

    pool_sizes = [2, 3, 4, 5, 6]
    print("=" * 98)
    print(f"TASK 2 VARIATION 07: Candidate Anchor Pool Size |A| Ablation on Targets: {targets}")
    print("=" * 98)
    print(f"{'Pool Size |A|':<15} | {'Candidate Anchors Available':<46} | {'Adaptive Drift':<15} | {'Update Cos Sim':<15}")
    print("-" * 98)

    results = []
    for k in pool_sizes:
        sub_anchors = all_anchors[:k]
        res = run_strategy_evaluation("adaptive", targets, sub_anchors, embs, domain="style")
        short_a = ", ".join(sub_anchors[:3]) + ("..." if len(sub_anchors) > 3 else "")
        print(
            f"|A| = {k:<9} | {short_a:<46} | {res['cumulative_parameter_drift']:<15.4f} | "
            f"{res['mean_update_cosine_similarity']:<15.4f}"
        )
        results.append({
            "pool_size": k,
            "candidates": sub_anchors,
            "assigned_anchors": res["assigned_anchors"],
            "cumulative_drift": res["cumulative_parameter_drift"],
            "mean_update_cosine_similarity": res["mean_update_cosine_similarity"],
        })

    print("\nDetailed Step-by-Step Assignments for Full Pool (|A|=5):")
    res5 = results[3]
    for step_idx, (t, a) in enumerate(zip(targets, res5["assigned_anchors"]), start=1):
        print(f"  Step {step_idx}: Target '{t:<14}' --> Dynamically Assigned Anchor '{a}'")

    out_dir = REPO_ROOT / "outputs" / "variations"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v07_anchor_pool_size_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("-" * 98)


if __name__ == "__main__":
    main()
