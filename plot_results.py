#!/usr/bin/env python3
"""
Generate High-Resolution Presentation & Report Plots for Task 1 and Task 2
==========================================================================
Reads the generated evaluation JSON files from:
  - ./outputs/task1_merge/task1_summary.json
  - ./outputs/task2_anchor/task2_summary.json
and outputs publication-quality figures into ./outputs/plots/:
  1. task1_threshold_curve.png       : Phase transition curves UA(alpha) & RA(alpha) + estimated alpha_i*
  2. task1_comparison.png            : Style & Object Continual Unlearning comparison (UA, RA-I, RA-C, Drift)
  3. task2_comparison.png            : Shared vs. Independent vs. Adaptive Anchor Assignment (UA, RA-I, RA-C)
  4. task2_cumulative_drift.png      : Cumulative Parameter Drift & Update Similarity across sequential requests
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_task1_figures():
    summary_path = REPO_ROOT / "outputs" / "task1_merge" / "task1_summary.json"
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found. Run `python run_adaptive_merge.py` first.")
        return

    with open(summary_path, "r") as f:
        t1_data = json.load(f)

    style_concepts = t1_data["style"]["concepts"]

    # -------------------------------------------------------------------------
    # Figure 1: Task 1 Phase Transition Curves & Concept-Specific Thresholds (alpha_i^*)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    selected_indices = [0, 2, 5, 7, 11]  # Abstractionism, Cartoon, Van_Gogh, Picasso, Vibrant_Flow
    colors = ["#d9534f", "#f0ad4e", "#5cb85c", "#0275d8", "#8e44ad"]

    for c_idx, col in zip(selected_indices, colors):
        c_info = style_concepts[c_idx]
        c_name = c_info["concept_name"]
        grid = c_info["sweep_curve"]["alpha_grid"]
        ua_curve = c_info["sweep_curve"]["UA"]
        a_star = c_info["alpha_star"]

        ax1.plot(grid, ua_curve, label=f"{c_name} ($\\alpha^*={a_star:.2f}$)", color=col, linewidth=2.4)
        ax1.axvline(x=a_star, color=col, linestyle=":", linewidth=1.6, alpha=0.85)

    # Plot average retention curve on ax1 secondary reference
    ref_grid = style_concepts[5]["sweep_curve"]["alpha_grid"]
    ref_rai = style_concepts[5]["sweep_curve"]["RA_I"]
    ax1.plot(ref_grid, ref_rai, label="In-Domain Retention (RA-I)", color="#2c3e50", linewidth=2.2, linestyle="--")
    ax1.axhline(y=95.0, color="gray", linestyle="-.", linewidth=1.2, label="Target Erasure Threshold (95%)")

    ax1.set_title("(a) Sharp Erasure Transition Across Merging Coefficient ($\\alpha$)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Interpolation / Merging Coefficient ($\\alpha$)", fontsize=10.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax1.set_xlim(0.05, 1.35)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.92)

    # Subplot (b): Estimated Minimum Coefficient alpha_i^* across all 12 Styles & Objects
    style_names = [c["concept_name"] for c in t1_data["style"]["concepts"]]
    style_alphas = [c["alpha_star"] for c in t1_data["style"]["concepts"]]
    x_pos = np.arange(len(style_names))

    bars = ax2.bar(x_pos, style_alphas, color="#3498db", edgecolor="#1f618d", alpha=0.88, width=0.62, label="Estimated $\\alpha_i^*$")
    ax2.axhline(y=1.0, color="#d9534f", linestyle="--", linewidth=2.0, label="Fixed Baseline $\\lambda = 1.0$")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(style_names, rotation=38, ha="right", fontsize=8.8)
    ax2.set_ylabel("Minimum Erasure Coefficient ($\\alpha_i^*$)", fontsize=10.5)
    ax2.set_title("(b) Estimated Critical Thresholds ($\\alpha_i^*$) per Concept", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, 1.25)
    ax2.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white")

    for bar, val in zip(bars, style_alphas):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    plt.tight_layout()
    out1 = PLOTS_DIR / "task1_threshold_curve.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # -------------------------------------------------------------------------
    # Figure 2: Task 1 Baseline vs. Adaptive Merging Comparison (Style & Object)
    # -------------------------------------------------------------------------
    fig, (ax_s, ax_o, ax_d) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    method_labels = {
        "uniform": "Uniform\nMerge",
        "task_arithmetic": "Task\nArithmetic",
        "ties_fixed": "Fixed TIES\n(Baseline)",
        "adaptive": "Adaptive\nMerge (Ours)",
    }

    for ax, dom, title_prefix in [(ax_s, "style", "(a) Continual Style Unlearning"), (ax_o, "object", "(b) Continual Object Unlearning")]:
        rows = t1_data[dom]["method_comparison"]
        m_names = [method_labels[r["method"]] for r in rows]
        ua_v = [r["UA"] for r in rows]
        rai_v = [r["RA_I"] for r in rows]
        rac_v = [r["RA_C"] for r in rows]

        x = np.arange(len(rows))
        w = 0.25
        ax.bar(x - w, ua_v, w, label="UA (%)", color="#2ecc71")
        ax.bar(x, rai_v, w, label="RA-I (%)", color="#3498db")
        ax.bar(x + w, rac_v, w, label="RA-C (%)", color="#f39c12")

        ax.set_xticks(x)
        ax.set_xticklabels(m_names, fontsize=9.2, fontweight="bold")
        ax.set_ylim(45, 104)
        ax.set_ylabel("Score (%)", fontsize=10)
        ax.set_title(title_prefix, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white")

    # Subplot (c): Cumulative Parameter Drift vs Number of Unlearned Concepts
    prog_s = t1_data["style"]["continual_progression"]
    n_c = [p["num_concepts"] for p in prog_s]
    d_ties = [p["ties_fixed_drift"] for p in prog_s]
    d_adap = [p["adaptive_drift"] for p in prog_s]

    ax_d.plot(n_c, d_ties, marker="s", color="#e74c3c", linewidth=2.5, markersize=7, label="Fixed-Lambda TIES (Style)")
    ax_d.plot(n_c, d_adap, marker="^", color="#27ae60", linewidth=2.8, markersize=8, label="Threshold-Aware Adaptive (Ours)")
    ax_d.set_title("(c) Parameter Drift vs. Continual Requests", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Number of Unlearned Concepts Merged ($N$)", fontsize=10)
    ax_d.set_ylabel("L2 Parameter Drift $||\\theta_{merge} - \\theta_0||_2$", fontsize=10)
    ax_d.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor="white")

    plt.tight_layout()
    out2 = PLOTS_DIR / "task1_comparison.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


def plot_task2_figures():
    t2_summary_path = REPO_ROOT / "outputs" / "task2_anchor" / "task2_summary.json"
    if not t2_summary_path.exists():
        print(f"Warning: {t2_summary_path} not found. Run `python run_adaptive_anchor.py` first.")
        return

    with open(t2_summary_path, "r") as f:
        t2_data = json.load(f)

    strategies = ["shared", "independent", "adaptive"]
    strat_titles = ["Shared Anchor", "Independent Anchors", "Adaptive Anchor (Ours)"]

    # -------------------------------------------------------------------------
    # Figure 3: Task 2 Strategy Comparison (UA, RA-I, RA-C across Style & Object)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    for ax, dom, panel_title in [
        (ax1, "style", "(a) Continual Style Unlearning (UA, RA-I, RA-C)"),
        (ax2, "object", "(b) Continual Object Unlearning (UA, RA-I, RA-C)"),
    ]:
        dom_dict = t2_data[dom]
        ua_vals = [dom_dict[s]["UA"] for s in strategies]
        rai_vals = [dom_dict[s]["RA_I"] for s in strategies]
        rac_vals = [dom_dict[s]["RA_C"] for s in strategies]

        x = np.arange(len(strategies))
        width = 0.25
        b1 = ax.bar(x - width, ua_vals, width, label="Unlearning Accuracy (UA)", color="#2ecc71")
        b2 = ax.bar(x, rai_vals, width, label="In-Domain Retention (RA-I)", color="#3498db")
        b3 = ax.bar(x + width, rac_vals, width, label="Cross-Domain Retention (RA-C)", color="#f39c12")

        ax.set_xticks(x)
        ax.set_xticklabels(strat_titles, fontsize=9.8, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=10.5)
        ax.set_title(panel_title, fontsize=11.5, fontweight="bold")
        ax.set_ylim(55, 103)
        ax.legend(loc="lower right", fontsize=8.8, frameon=True, facecolor="white", framealpha=0.92)

        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8)

    plt.tight_layout()
    out3 = PLOTS_DIR / "task2_comparison.png"
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    # -------------------------------------------------------------------------
    # Figure 4: Task 2 Cumulative Drift Trajectory & Update Similarity Comparison
    # -------------------------------------------------------------------------
    fig, (ax_drift, ax_sim) = plt.subplots(1, 2, figsize=(13, 4.6))

    style_dict = t2_data["style"]
    targets = style_dict["adaptive"]["target_concepts"]
    step_labels = [f"Step {i+1}\n({t})" for i, t in enumerate(targets)]

    colors = {"shared": "#e74c3c", "independent": "#f39c12", "adaptive": "#27ae60"}
    markers = {"shared": "o", "independent": "s", "adaptive": "^"}
    labels_map = {
        "shared": "Shared Anchor (Fixed)",
        "independent": "Independent Anchors",
        "adaptive": "Adaptive Anchor Assignment (Ours)",
    }

    for s in strategies:
        traj = style_dict[s]["cumulative_drift_trajectory"]
        ax_drift.plot(
            step_labels,
            traj,
            marker=markers[s],
            color=colors[s],
            linewidth=2.8 if s == "adaptive" else 2.2,
            markersize=8,
            label=labels_map[s],
        )

    ax_drift.set_title("(a) Cumulative Parameter Drift Over Sequential Requests", fontsize=11.5, fontweight="bold")
    ax_drift.set_ylabel("Cumulative L2 Drift ($\\sum_k ||\\Delta\\theta_k||_2$)", fontsize=10.5)
    ax_drift.set_xlabel("Sequential Unlearning Requests", fontsize=10.5)
    ax_drift.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white")

    # Subplot (b): Mean Parameter-Update Cosine Similarity & Total Drift Comparison
    sim_style = [style_dict[s]["mean_update_cosine_similarity"] for s in strategies]
    sim_obj = [t2_data["object"][s]["mean_update_cosine_similarity"] for s in strategies]

    x = np.arange(len(strategies))
    w = 0.30
    b_s = ax_sim.bar(x - w / 2, sim_style, w, label="Style Unlearning", color="#8e44ad")
    b_o = ax_sim.bar(x + w / 2, sim_obj, w, label="Object Unlearning", color="#16a085")

    ax_sim.set_xticks(x)
    ax_sim.set_xticklabels(strat_titles, fontsize=9.8, fontweight="bold")
    ax_sim.set_ylabel("Mean Update Cosine Similarity $|\\cos(\\Delta\\theta_i, \\Delta\\theta_j)|$", fontsize=10)
    ax_sim.set_title("(b) Parameter Update Interference Across Requests (Lower is Better)", fontsize=11, fontweight="bold")
    ax_sim.set_ylim(0, 0.65)
    ax_sim.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white")

    for bars in (b_s, b_o):
        for bar in bars:
            h = bar.get_height()
            ax_sim.text(bar.get_x() + bar.get_width() / 2.0, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    plt.tight_layout()
    out4 = PLOTS_DIR / "task2_cumulative_drift.png"
    plt.savefig(out4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out4}")


if __name__ == "__main__":
    plot_task1_figures()
    plot_task2_figures()
    print("\nAll Task 1 and Task 2 plots successfully generated in ./outputs/plots/!")
