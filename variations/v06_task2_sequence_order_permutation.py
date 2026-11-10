#!/usr/bin/env python3
"""
Variation 06 (Task 2): Sequential Concept Ordering Permutation Sensitivity
==========================================================================
Professor Prompt:
  "In Continual Unlearning, requests arrive sequentially. Does the order of incoming
   concepts (e.g., Default Order vs. Reverse Order vs. Interleaved / Random Order)
   affect the anchor assignments, update interference, and cumulative parameter drift?"
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Regularizers.Anchor.anchor_selection import TARGET_CONCEPT_DESCRIPTIONS
from run_adaptive_anchor import build_semantic_embeddings, run_strategy_evaluation


def main():
    base_targets = ["Picasso", "Van_Gogh", "Cartoon", "Ukiyoe", "Neon_Lines", "Winter"]
    anchors = ["Classic_Painting", "Line_Illustration", "Geometric_Design", "Atmospheric_Landscape", "Realism_Photo"]

    orderings = {
        "1. Default Order (Fine Art -> Graphic -> Atmospheric)": base_targets,
        "2. Reverse Order (Atmospheric -> Graphic -> Fine Art)": list(reversed(base_targets)),
        "3. Clustered Order (Similar Styles Back-to-Back)":      ["Picasso", "Van_Gogh", "Ukiyoe", "Cartoon", "Neon_Lines", "Winter"],
        "4. Alternating Contrast Order":                         ["Winter", "Picasso", "Cartoon", "Van_Gogh", "Neon_Lines", "Ukiyoe"],
    }

    embs = build_semantic_embeddings(
        concepts=list(set(base_targets + anchors)),
        descriptions=TARGET_CONCEPT_DESCRIPTIONS,
        base_model_dir="Checkpoints/Generators/UnlearnCanvas",
        device="cpu",
        use_clip=False,
    )

    print("=" * 106)
    print("TASK 2 VARIATION 06: Effect of Sequential Request Arrival Order on Adaptive vs. Shared Anchors")
    print("=" * 106)
    print(f"{'Arrival Order Sequence':<54} | {'Shared Drift':<13} | {'Indep. Drift':<13} | {'Adaptive Drift':<14} | {'Adaptive Update Sim'}")
    print("-" * 106)

    out = {}
    for name, seq in orderings.items():
        res_shared = run_strategy_evaluation("shared", seq, anchors, embs, domain="style")
        res_indep = run_strategy_evaluation("independent", seq, anchors, embs, domain="style")
        res_adap = run_strategy_evaluation("adaptive", seq, anchors, embs, domain="style")

        print(
            f"{name:<54} | {res_shared['cumulative_parameter_drift']:<13.4f} | "
            f"{res_indep['cumulative_parameter_drift']:<13.4f} | "
            f"{res_adap['cumulative_parameter_drift']:<14.4f} | "
            f"{res_adap['mean_update_cosine_similarity']:.4f}"
        )
        print(f"   -> Adaptive Mapping: {', '.join(f'{t}->{a}' for t, a in zip(seq, res_adap['assigned_anchors']))}")
        out[name] = {
            "sequence": seq,
            "shared_drift": res_shared["cumulative_parameter_drift"],
            "independent_drift": res_indep["cumulative_parameter_drift"],
            "adaptive_drift": res_adap["cumulative_parameter_drift"],
            "adaptive_assigned_anchors": res_adap["assigned_anchors"],
            "adaptive_update_sim": res_adap["mean_update_cosine_similarity"],
        }

    out_dir = REPO_ROOT / "outputs" / "variations"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v06_sequence_order_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 106)


if __name__ == "__main__":
    main()
