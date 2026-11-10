"""
generate_figures.py  -  Step 8: Generate all paper figures.

Usage:
    python project/src/generate_figures.py \
        --analysis_prefix project/results/analysis_sst2 \
        --output_dir project/figures

Generates:
    fig1_heatmap_4x4.png        - 4x4 norm-comp AOPC heatmap (IG-baseline × erasure)
    fig2_npig_barchart.png      - NP-IG vs single baselines per erasure scheme (with CIs)
    fig3_kendall_tau.png        - Kendall tau heatmap across erasure schemes
    fig4_aopc_curves.png        - AOPC curves (aggregated over fractions) per method
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Constants ─────────────────────────────────────────────────────────────────
SINGLE_BASELINE_METHODS = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO"]
ERASURE_SCHEMES         = ["PAD", "MASK", "UNK", "DELETE"]
CANDIDATE_METHODS       = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO",
                            "NP_IG", "UNIFORM_IG", "GRAD_X_INP", "RANDOM"]

METHOD_LABELS = {
    "IG_PAD":     "IG-PAD",
    "IG_MASK":    "IG-MASK",
    "IG_UNK":     "IG-UNK",
    "IG_ZERO":    "IG-ZERO",
    "NP_IG":      "NP-IG (ours)",
    "UNIFORM_IG": "Uniform-IG",
    "GRAD_X_INP": "Grad×Input",
    "RANDOM":     "Random",
}

COLORS = {
    "NP_IG":      "#e63946",
    "UNIFORM_IG": "#457b9d",
    "GRAD_X_INP": "#2a9d8f",
    "IG_PAD":     "#6a4c93",
    "IG_MASK":    "#f4a261",
    "IG_UNK":     "#264653",
    "IG_ZERO":    "#a8dadc",
    "RANDOM":     "#999999",
}


# ── Figure 1: 4×4 Heatmap ────────────────────────────────────────────────────

def fig1_heatmap(matrix_json, stats_json, output_dir):
    norm_comp_matrix = matrix_json["norm_comp_matrix"]
    stats            = stats_json["norm_comp"]

    # Build numpy array [4 baselines x 4 erasure schemes]
    data = np.array([[norm_comp_matrix[m][s] for s in ERASURE_SCHEMES]
                     for m in SINGLE_BASELINE_METHODS])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Custom diverging colormap centered at mean
    cmap = LinearSegmentedColormap.from_list(
        "custom_div", ["#2166ac", "#f7f7f7", "#d6604d"], N=256)

    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=data.min() - 0.005, vmax=data.max() + 0.005)

    ax.set_xticks(range(len(ERASURE_SCHEMES)))
    ax.set_yticks(range(len(SINGLE_BASELINE_METHODS)))
    ax.set_xticklabels([f"Erase-{s}" for s in ERASURE_SCHEMES], fontsize=10)
    ax.set_yticklabels([METHOD_LABELS[m] for m in SINGLE_BASELINE_METHODS], fontsize=10)
    ax.set_xlabel("Erasure Scheme", fontsize=11)
    ax.set_ylabel("IG Baseline", fontsize=11)
    ax.set_title("Normalized Comprehensiveness AOPC\n(IG Baseline × Erasure Scheme)", fontsize=11, pad=10)

    # Annotate cells with value ± CI half-width; bold the diagonal
    for i, m in enumerate(SINGLE_BASELINE_METHODS):
        for j, s in enumerate(ERASURE_SCHEMES):
            val = data[i, j]
            ci_l = stats[m][s]["ci_lower"]
            ci_u = stats[m][s]["ci_upper"]
            half = (ci_u - ci_l) / 2
            is_diag = (ERASURE_SCHEMES[j] == m.replace("IG_", ""))
            text = f"{val:.3f}\n±{half:.3f}"
            color = "white" if (val > data.max() * 0.75 or val < data.min() * 0.75 + data.mean() * 0.1) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_diag else "normal",
                    color="black")
            if is_diag:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Norm. Comp. AOPC")
    plt.tight_layout()
    out = os.path.join(output_dir, "fig1_heatmap_4x4.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: NP-IG Bar Chart ─────────────────────────────────────────────────

def fig2_barchart(stats_json, output_dir):
    stats = stats_json["norm_comp"]
    methods_to_plot = ["NP_IG", "UNIFORM_IG", "IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO", "GRAD_X_INP"]

    n_schemes  = len(ERASURE_SCHEMES)
    n_methods  = len(methods_to_plot)
    bar_width  = 0.11
    x          = np.arange(n_schemes)

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for k, m in enumerate(methods_to_plot):
        means  = [stats[m][s]["mean"]      for s in ERASURE_SCHEMES]
        ci_lo  = [stats[m][s]["ci_lower"]  for s in ERASURE_SCHEMES]
        ci_hi  = [stats[m][s]["ci_upper"]  for s in ERASURE_SCHEMES]
        yerr_lo = [means[j] - ci_lo[j]     for j in range(n_schemes)]
        yerr_hi = [ci_hi[j] - means[j]     for j in range(n_schemes)]

        bars = ax.bar(x + offsets[k], means, bar_width,
                      label=METHOD_LABELS[m],
                      color=COLORS[m],
                      yerr=[yerr_lo, yerr_hi],
                      error_kw={"elinewidth": 1.2, "capsize": 3},
                      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Erase-{s}" for s in ERASURE_SCHEMES], fontsize=10)
    ax.set_xlabel("Erasure Scheme", fontsize=11)
    ax.set_ylabel("Normalized Comp. AOPC", fontsize=11)
    ax.set_title("NP-IG vs. Attribution Methods per Erasure Scheme\n(95% Bootstrap CI)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, frameon=True)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, "fig2_npig_barchart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: Kendall Tau Heatmap ─────────────────────────────────────────────

def fig3_kendall(rankings_json, output_dir):
    tau_mat = rankings_json["kendall_tau_matrix"]
    data = np.array([[tau_mat[s1][s2]["tau"] for s2 in ERASURE_SCHEMES]
                     for s1 in ERASURE_SCHEMES])

    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = LinearSegmentedColormap.from_list(
        "tau_cmap", ["#d6604d", "#f7f7f7", "#4393c3"], N=256)
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(ERASURE_SCHEMES)))
    ax.set_yticks(range(len(ERASURE_SCHEMES)))
    ax.set_xticklabels([f"Erase-{s}" for s in ERASURE_SCHEMES], fontsize=9)
    ax.set_yticklabels([f"Erase-{s}" for s in ERASURE_SCHEMES], fontsize=9)
    ax.set_title("Kendall τ: Method Ranking Correlation\nAcross Erasure Schemes", fontsize=11)

    for i, s1 in enumerate(ERASURE_SCHEMES):
        for j, s2 in enumerate(ERASURE_SCHEMES):
            tau_v = tau_mat[s1][s2]["tau"]
            p_v   = tau_mat[s1][s2]["p_value"]
            sig   = "*" if p_v < 0.05 else ""
            ax.text(j, i, f"{tau_v:.2f}{sig}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Kendall τ")
    plt.tight_layout()
    out = os.path.join(output_dir, "fig3_kendall_tau.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 4: Per-method worst-case box ───────────────────────────────────────

def fig4_worstcase(stats_json, h2_json, output_dir):
    """Bar chart of worst-case (min over erasure schemes) mean normalized comp AOPC."""
    stats = stats_json["norm_comp"]
    methods_to_plot = ["NP_IG", "UNIFORM_IG", "IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO", "GRAD_X_INP"]

    worst_means  = []
    worst_labels = []
    npig_worst   = h2_json["IG_PAD"]["npig_worst_case_mean"]

    for m in methods_to_plot:
        all_scheme_means = [stats[m][s]["mean"] for s in ERASURE_SCHEMES]
        worst_means.append(min(all_scheme_means))
        worst_labels.append(METHOD_LABELS[m])

    colors = [COLORS[m] for m in methods_to_plot]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(methods_to_plot)), worst_means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.set_xticks(range(len(methods_to_plot)))
    ax.set_xticklabels(worst_labels, rotation=20, ha="right", fontsize=10)
    ax.axhline(npig_worst, color=COLORS["NP_IG"], linestyle="--", linewidth=1.5, label=f"NP-IG worst = {npig_worst:.4f}")
    ax.set_ylabel("Worst-Case Norm. Comp. AOPC\n(min over erasure schemes)", fontsize=10)
    ax.set_title("Worst-Case Robustness Across Erasure Schemes", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig4_worstcase.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_prefix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="project/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def load(suffix):
        path = args.analysis_prefix + suffix
        with open(path) as f:
            return json.load(f)

    print("Loading analysis outputs...")
    matrix_json   = load("_matrix.json")
    stats_json    = load("_stats.json")
    rankings_json = load("_rankings.json")
    h2_json       = load("_h2.json")

    print("Generating Figure 1: 4×4 heatmap...")
    fig1_heatmap(matrix_json, stats_json, args.output_dir)

    print("Generating Figure 2: NP-IG bar chart...")
    fig2_barchart(stats_json, args.output_dir)

    print("Generating Figure 3: Kendall tau heatmap...")
    fig3_kendall(rankings_json, args.output_dir)

    print("Generating Figure 4: Worst-case robustness bar chart...")
    fig4_worstcase(stats_json, h2_json, args.output_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
