"""
Automatic Plot Generator for All 8 Live Professor Variations (v01 .. v08).
Saves high-resolution PNG plots into `outputs/variations/plots/` without touching
the main `outputs/plots/` folder.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
VAR_PLOTS_DIR = REPO_ROOT / "outputs" / "variations" / "plots"


def plot_v01(results):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    taus = [r["target_ua_threshold"] for r in results]
    uas = [r["UA"] for r in results]
    rais = [r["RA_I"] for r in results]
    drifts = [r["l2_drift"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    ax1.plot(taus, uas, "o-", color="#2563EB", lw=2.5, label="Achieved UA (%) ↑")
    ax1.plot(taus, rais, "s-", color="#0D9488", lw=2.5, label="Retain Accuracy RA-I (%) ↑")
    ax1.set_xlabel("Target Unlearning Threshold τ_UA (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold", color="#0F172A")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(taus, drifts, "^--", color="#DC2626", lw=2.2, label="Cumulative L2 Drift ↓")
    ax2.set_ylabel("Cumulative L2 Parameter Drift", fontsize=11, fontweight="bold", color="#DC2626")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    plt.title("Variation 01: Effect of Target Erasure Threshold (τ_UA) on Accuracy & Drift", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v01_threshold_ablation.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v02(out_rows):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["configuration"].replace("Adaptive alpha* + ", "Adap+").replace("Baseline Fixed TIES ", "Fixed ") for r in out_rows]
    uas = [r["UA"] for r in out_rows]
    rais = [r["RA_I"] for r in out_rows]
    drifts = [r["l2_drift"] for r in out_rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    ax1.bar(x - w/2, uas, w, label="UA (%) ↑", color="#2563EB")
    ax1.bar(x + w/2, rais, w, label="RA-I (%) ↑", color="#0D9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax1.set_ylim(60, 102)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by TIES Function & Top-K")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(x, drifts, color="#DC2626", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax2.set_ylabel("Cumulative L2 Drift ↓")
    ax2.set_title("L2 Parameter Drift by Configuration")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Variation 02: TIES Merge Function (mean/sum/max) & Top-K Sparsity Ablation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v02_ties_func_topk_ablation.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v03(rows):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Keys Only\n(to_k)", "Values Only\n(to_v)", "Concept-Wise\n(to_k + to_v)", "Layer/Block-Wise\nAdaptive (Ours)"]
    uas = [r["UA"] for r in rows]
    rais = [r["RA_I"] for r in rows]
    drifts = [r["drift"] for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    ax1.bar(x - w/2, uas, w, label="UA (%) ↑", color="#2563EB")
    ax1.bar(x + w/2, rais, w, label="RA-I (%) ↑", color="#0D9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(75, 100)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy: Keys vs Values vs Block-Wise")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(x, drifts, color="#7C3AED", width=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Cumulative L2 Drift ↓")
    ax2.set_title("L2 Parameter Drift")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Variation 03: Cross-Attention Filter (to_k vs to_v) & Layer-Wise Thresholds", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v03_layerwise_and_keyfilter.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v04(out):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [r["name"].replace(" Scaling", "") for r in out]
    uas = [r["UA"] for r in out]
    rais = [r["RA_I"] for r in out]
    drifts = [r["drift"] for r in out]

    x = np.arange(len(names))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    ax1.bar(x - w/2, uas, w, label="UA (%) ↑", color="#2563EB")
    ax1.bar(x + w/2, rais, w, label="RA-I (%) ↑", color="#0D9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(80, 100)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy Across αᵢ* Mathematical Formulas")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(x, drifts, color="#D97706", width=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Cumulative L2 Drift ↓")
    ax2.set_title("L2 Drift Across αᵢ* Formulas")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Variation 04: Comparison of Mathematical Formulas for Critical Threshold αᵢ*", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v04_threshold_formulas.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v05(results):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    short_names = [r["variant"].split("(")[0].strip() for r in results]
    uas = [r["UA"] for r in results]
    rais = [r["RA_I"] for r in results]
    drifts = [r["cumulative_drift"] for r in results]

    x = np.arange(len(short_names))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    ax1.bar(x - w/2, uas, w, label="UA (%) ↑", color="#2563EB")
    ax1.bar(x + w/2, rais, w, label="RA-I (%) ↑", color="#0D9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=22, ha="right", fontsize=8.8)
    ax1.set_ylim(75, 100)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by Anchor Criterion Terms")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(x, drifts, color="#DC2626", width=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=22, ha="right", fontsize=8.8)
    ax2.set_ylabel("Cumulative L2 Drift ↓")
    ax2.set_title("Cumulative L2 Drift (Lower = Less Congestion)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Variation 05: Ablation of Anchor Selection Weights (w_sem, w_sim, w_cong)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v05_anchor_criterion_ablation.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v06(out):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [k.split(":")[0] for k in out.keys()]
    s_drift = [v["shared_drift"] for v in out.values()]
    i_drift = [v["independent_drift"] for v in out.values()]
    a_drift = [v["adaptive_drift"] for v in out.values()]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=180)
    ax.bar(x - w, s_drift, w, label="Shared Anchor Drift", color="#DC2626")
    ax.bar(x,     i_drift, w, label="Independent Anchor Drift", color="#2563EB")
    ax.bar(x + w, a_drift, w, label="Adaptive Anchor Drift (Ours)", color="#0D9488")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative L2 Parameter Drift ↓")
    ax.set_title("Variation 06: Order Invariance Across 4 Continual Request Permutations", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v06_sequence_order_sensitivity.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v07(results):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ks = [r["pool_size"] for r in results]
    drifts = [r["cumulative_drift"] for r in results]
    sims = [r["mean_update_cosine_similarity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    ax1.plot(ks, drifts, "o-", color="#0D9488", lw=2.6, markersize=8, label="Cumulative L2 Drift ↓")
    ax1.set_xlabel("Candidate Anchor Pool Size |A|", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative L2 Drift ↓", fontsize=11, fontweight="bold", color="#0D9488")
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(ks, sims, "s--", color="#DC2626", lw=2.2, markersize=7, label="Mean Update Cosine Sim ↓")
    ax2.set_ylabel("Mean Update Cosine Similarity ↓", fontsize=11, fontweight="bold", color="#DC2626")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Variation 07: Effect of Candidate Anchor Pool Size |A| on Drift & Overlap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v07_anchor_pool_size_ablation.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")


def plot_v08(table):
    VAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["1. Baseline\n(Shared+Fixed)", "2. Task 1 Only\n(Adaptive Merge)", "3. Task 2 Only\n(Adaptive Anchor)", "4. Unified\n(Task 1 + Task 2)"]
    uas = [r["UA"] for r in table]
    rais = [r["RA_I"] for r in table]
    drifts = [r["drift"] for r in table]

    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=180)
    ax1.bar(x - w/2, uas, w, label="UA (%) ↑", color="#2563EB")
    ax1.bar(x + w/2, rais, w, label="RA-I (%) ↑", color="#0D9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(70, 100)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Synergy on Unlearning & Retain Accuracy")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(x, drifts, color=["#DC2626", "#2563EB", "#D97706", "#0D9488"], width=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Cumulative L2 Drift ↓")
    ax2.set_title("Synergy on L2 Drift Reduction (-70.5%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Variation 08: Unified Pipeline Synergy (Task 1 Adaptive Merge + Task 2 Adaptive Anchor)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = VAR_PLOTS_DIR / "v08_combined_task1_task2.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[Plot Saved] -> {out_path}")
