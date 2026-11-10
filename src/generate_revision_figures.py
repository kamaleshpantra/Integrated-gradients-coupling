"""
generate_revision_figures.py - Generates revised publication-quality figures.

Generates:
1. fig1_grid_heatmap.png: 4x4 In-Domain Grid (PAD, MASK, UNK, ZERO_EMB) with red matched borders.
2. fig2_erasure_comparison.png: Grouped bar chart with 95% CIs across in-domain and held-out erasure schemes.
3. fig3_kendall_tau.png: Kendall tau rank correlation matrix across all 7 erasure schemes.
4. fig4_npig_ablation.png: NP-IG temperature ablation curve (weight entropy vs worst-case score).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BASELINES = ["PAD", "MASK", "UNK", "ZERO"]
IN_DOMAIN_SCHEMES = ["PAD", "MASK", "UNK", "ZERO_EMB"]
ALL_SCHEMES = ["PAD", "MASK", "UNK", "ZERO_EMB", "DELETE", "RANDOM_TOKEN", "MLM_INFILL"]

METHOD_LABELS = {
    "IG_PAD": "IG-PAD",
    "IG_MASK": "IG-MASK",
    "IG_UNK": "IG-UNK",
    "IG_ZERO": "IG-ZERO",
    "NP_IG": "NP-IG (tau=0.1)",
    "UNIFORM_IG": "Uniform-IG",
    "GRAD_X_INP": "GradxInput (logit)",
    "RANDOM": "Random"
}

COLORS = {
    "NP_IG": "#e63946",
    "UNIFORM_IG": "#457b9d",
    "GRAD_X_INP": "#2a9d8f",
    "IG_PAD": "#6a4c93",
    "IG_MASK": "#f4a261",
    "IG_UNK": "#264653",
    "IG_ZERO": "#a8dadc",
    "RANDOM": "#999999",
}

def fig1_heatmap(df, output_dir):
    single_methods = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO"]
    matched_map = {"IG_PAD": "PAD", "IG_MASK": "MASK", "IG_UNK": "UNK", "IG_ZERO": "ZERO_EMB"}
    
    # 4x4 matrix
    matrix = np.zeros((4, 4))
    ci_matrix = np.zeros((4, 4))
    
    for i, m in enumerate(single_baselines := single_methods):
        for j, s in enumerate(IN_DOMAIN_SCHEMES):
            vals = df[(df["method"] == m) & (df["scheme"] == s)]["norm_comp_aopc"].values
            matrix[i, j] = float(np.mean(vals))
            # 95% CI half-width via bootstrap
            boot = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(1000)]
            ci_matrix[i, j] = (np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2.0
            
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = LinearSegmentedColormap.from_list("custom_div", ["#2166ac", "#f7f7f7", "#d6604d"], N=256)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=matrix.min() - 0.01, vmax=matrix.max() + 0.01)
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"Erase-{s}" for s in IN_DOMAIN_SCHEMES], fontsize=10)
    ax.set_yticklabels([METHOD_LABELS[m] for m in single_methods], fontsize=10)
    ax.set_xlabel("Erasure Scheme", fontsize=11)
    ax.set_ylabel("Attribution Baseline", fontsize=11)
    ax.set_title("Normalized Comprehensiveness AOPC ($M=50$)\n(4x4 In-Domain Baseline x Erasure Grid)", fontsize=11, pad=10)
    
    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            half = ci_matrix[i, j]
            is_diag = (IN_DOMAIN_SCHEMES[j] == matched_map[single_methods[i]])
            text = f"{val:.3f}\n±{half:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_diag else "normal", color="black")
            if is_diag:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5)
                ax.add_patch(rect)
                
    plt.colorbar(im, ax=ax, shrink=0.8, label="Norm. Comp. AOPC")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig1_grid_heatmap.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def fig2_erasure_comparison(df, output_dir):
    methods_to_plot = ["NP_IG", "UNIFORM_IG", "IG_MASK", "IG_UNK", "IG_PAD", "IG_ZERO", "GRAD_X_INP"]
    schemes_to_plot = ALL_SCHEMES
    
    n_schemes = len(schemes_to_plot)
    n_methods = len(methods_to_plot)
    bar_width = 0.11
    x = np.arange(n_schemes)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width
    
    for k, m in enumerate(methods_to_plot):
        means = []
        errs = []
        for s in schemes_to_plot:
            vals = df[(df["method"] == m) & (df["scheme"] == s)]["norm_comp_aopc"].values
            mean_v = float(np.mean(vals))
            boot = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(1000)]
            ci_half = (np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2.0
            means.append(mean_v)
            errs.append(ci_half)
            
        ax.bar(x + offsets[k], means, bar_width, label=METHOD_LABELS[m], color=COLORS[m],
               yerr=errs, capsize=2.5, alpha=0.88, edgecolor="black", linewidth=0.5)
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"Erase\n{s}" for s in schemes_to_plot], fontsize=9.5)
    ax.set_ylabel("Norm. Comp. AOPC (Higher = More Faithful)", fontsize=11)
    ax.set_title("Attribution Performance Across In-Domain, Structural, and Held-Out Erasure Schemes (95% CI)", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    # Add vertical divider before held-out
    ax.axvline(3.5, color="blue", linestyle=":", alpha=0.5, label="_nolegend_")
    ax.axvline(4.5, color="blue", linestyle=":", alpha=0.5, label="_nolegend_")
    ax.text(1.5, 0.45, "In-Domain", ha="center", fontsize=10, fontweight="bold", color="blue")
    ax.text(4.0, 0.45, "Struct.", ha="center", fontsize=9, fontweight="bold", color="blue")
    ax.text(5.5, 0.45, "Held-Out Schemes", ha="center", fontsize=10, fontweight="bold", color="blue")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9.5)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2_erasure_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def fig3_kendall_tau(df, output_dir):
    from scipy.stats import kendalltau
    methods = ["NP_IG", "UNIFORM_IG", "IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO", "GRAD_X_INP", "RANDOM"]
    
    scheme_means = {}
    scheme_ranks = {}
    for s in ALL_SCHEMES:
        m_dict = {m: float(df[(df["method"] == m) & (df["scheme"] == s)]["norm_comp_aopc"].mean()) for m in methods}
        scheme_means[s] = m_dict
        sorted_m = sorted(methods, key=lambda m: m_dict[m], reverse=True)
        scheme_ranks[s] = {m: rank + 1 for rank, m in enumerate(sorted_m)}
        
    n = len(ALL_SCHEMES)
    tau_matrix = np.zeros((n, n))
    
    for i, s1 in enumerate(ALL_SCHEMES):
        for j, s2 in enumerate(ALL_SCHEMES):
            r1 = [scheme_ranks[s1][m] for m in methods]
            r2 = [scheme_ranks[s2][m] for m in methods]
            tau, _ = kendalltau(r1, r2)
            tau_matrix[i, j] = tau
            
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("tau_cmap", ["#d6604d", "#f7f7f7", "#4393c3"], N=256)
    im = ax.imshow(tau_matrix, cmap=cmap, vmin=0.5, vmax=1.0)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([s[:8] for s in ALL_SCHEMES], rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels([s[:8] for s in ALL_SCHEMES], fontsize=9)
    ax.set_title("Kendall tau: Method Ranking Correlation\nAcross All 7 Erasure Schemes", fontsize=11)
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{tau_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
            
    plt.colorbar(im, ax=ax, shrink=0.8, label="Kendall tau")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3_kendall_tau.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def fig4_npig_ablation(ablation_csv, output_dir):
    if not os.path.exists(ablation_csv):
        return
    adf = pd.read_csv(ablation_csv)
    
    taus = [0.01, 0.03, 0.1, 0.3, 1.0]
    entropies = []
    worst_cases = []
    
    for tau in taus:
        cfg = f"tau_{tau}"
        csub = adf[adf["config"] == cfg]
        ent = float(csub["entropy"].mean())
        # Def 2 worst case across in-domain schemes
        means = csub[csub["scheme"].isin(IN_DOMAIN_SCHEMES)].groupby("scheme")["norm_comp_aopc"].mean()
        wc = float(means.min())
        entropies.append(ent)
        worst_cases.append(wc)
        
    # Uniform
    csub_uni = adf[adf["config"] == "Uniform"]
    ent_uni = float(csub_uni["entropy"].mean())
    wc_uni = float(csub_uni[csub_uni["scheme"].isin(IN_DOMAIN_SCHEMES)].groupby("scheme")["norm_comp_aopc"].mean().min())
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(range(len(taus)), worst_cases, "o-", color="#e63946", linewidth=2, label="Worst-Case Comp. AOPC (Def 2)")
    l2 = ax2.plot(range(len(taus)), entropies, "s--", color="#457b9d", linewidth=2, label="Weight Entropy (nats)")
    
    # Horizontal line for Uniform-IG worst case
    ax1.axhline(wc_uni, color="black", linestyle=":", linewidth=1.5, label=f"Uniform-IG Worst-Case ({wc_uni:.3f})")
    
    ax1.set_xticks(range(len(taus)))
    ax1.set_xticklabels([f"tau={t}" for t in taus], fontsize=10)
    ax1.set_xlabel("NP-IG Temperature tau", fontsize=11)
    ax1.set_ylabel("Worst-Case Norm. Comp. AOPC", color="#e63946", fontsize=11)
    ax2.set_ylabel("Weight Entropy (nats, max=1.386)", color="#457b9d", fontsize=11)
    ax1.set_title("NP-IG Temperature Ablation: Entropy vs. Worst-Case Robustness", fontsize=11)
    
    lines = l1 + l2 + [plt.Line2D([0], [0], color="black", linestyle=":", linewidth=1.5)]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4_npig_ablation.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_csv", default="project/results/sst2_revision_all450.csv")
    parser.add_argument("--ablation_csv", default="project/results/np_ig_ablation.csv")
    parser.add_argument("--output_dir", default="project/figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.main_csv)
    
    print("Generating Figure 1: 4x4 In-Domain Grid Heatmap...")
    fig1_heatmap(df, args.output_dir)
    
    print("Generating Figure 2: Grouped Erasure Comparison...")
    fig2_erasure_comparison(df, args.output_dir)
    
    print("Generating Figure 3: Kendall Tau Correlation Matrix...")
    fig3_kendall_tau(df, args.output_dir)
    
    print("Generating Figure 4: NP-IG Temperature Ablation Curve...")
    fig4_npig_ablation(args.ablation_csv, args.output_dir)
    
    print("\nAll revision figures generated successfully!")

if __name__ == "__main__":
    main()
