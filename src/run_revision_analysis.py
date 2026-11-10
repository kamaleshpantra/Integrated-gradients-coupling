"""
run_revision_analysis.py - Comprehensive statistical analysis for revision paper.

Implements:
1. Linear Mixed Model (LMM):
   norm_comp_aopc ~ C(baseline) + C(scheme) + is_matched + (1 | example_id)
   - Matched effect beta with 95% CI and p-value
   - Per-baseline matched contrasts (diagonal vs row off-diagonal) with Holm-Bonferroni correction
   - Variance explained (sums of squares / partial eta^2)
   - Length-binned matched effect (Short, Medium, Long)
2. Dual Worst-Case Robustness Evaluation (Def 1 vs Def 2)
3. Held-Out Erasure Scheme Evaluation (RANDOM_TOKEN, MLM_INFILL)
4. NP-IG Temperature & Weighting Ablation Summary
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASELINES = ["PAD", "MASK", "UNK", "ZERO"]

def analyze_lmm_h1(df):
    """
    Fits Linear Mixed Model on single baseline methods:
    IG_PAD, IG_MASK, IG_UNK, IG_ZERO across in-domain schemes PAD, MASK, UNK, ZERO_EMB.
    """
    print("\n--- STEP 3: LINEAR MIXED MODEL (H1) ---")
    single_baselines = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO"]
    in_domain_schemes = ["PAD", "MASK", "UNK", "ZERO_EMB"]
    
    sub = df[(df["method"].isin(single_baselines)) & (df["scheme"].isin(in_domain_schemes))].copy()
    
    # Map method to baseline name
    sub["baseline"] = sub["method"].str.replace("IG_", "")
    
    # Fit Linear Mixed Model
    model = smf.mixedlm(
        "norm_comp_aopc ~ C(baseline, Treatment('PAD')) + C(scheme, Treatment('PAD')) + is_matched",
        data=sub,
        groups=sub["unique_id"]
    )
    result = model.fit()
    print(result.summary())
    
    matched_coef = float(result.params["is_matched"])
    matched_se   = float(result.bse["is_matched"])
    matched_p    = float(result.pvalues["is_matched"])
    ci_lower     = float(result.conf_int().loc["is_matched", 0])
    ci_upper     = float(result.conf_int().loc["is_matched", 1])
    
    # Variance Explained via OLS decomposition
    ols_full = smf.ols("norm_comp_aopc ~ C(baseline) + C(scheme) + is_matched", data=sub).fit()
    anova_table = sm.stats.anova_lm(ols_full, typ=2)
    total_ss = anova_table["sum_sq"].sum()
    var_explained = {
        idx: float(row["sum_sq"] / total_ss * 100)
        for idx, row in anova_table.iterrows()
    }
    
    # Per-baseline matched contrasts (diagonal vs row off-diagonal)
    # Matched pairs: (PAD, PAD), (MASK, MASK), (UNK, UNK), (ZERO, ZERO_EMB)
    matched_scheme_map = {"PAD": "PAD", "MASK": "MASK", "UNK": "UNK", "ZERO": "ZERO_EMB"}
    contrasts = {}
    p_vals_raw = []
    baseline_keys = ["PAD", "MASK", "UNK", "ZERO"]
    
    for b in baseline_keys:
        m_scheme = matched_scheme_map[b]
        diag_sub = sub[(sub["baseline"] == b) & (sub["scheme"] == m_scheme)].set_index("unique_id")["norm_comp_aopc"]
        off_diag_sub = sub[(sub["baseline"] == b) & (sub["scheme"] != m_scheme)].groupby("unique_id")["norm_comp_aopc"].mean()
        
        # Align on unique_id index
        common_ids = diag_sub.index.intersection(off_diag_sub.index)
        diag_scores = diag_sub.loc[common_ids].values
        off_scores = off_diag_sub.loc[common_ids].values
        
        diff = diag_scores - off_scores
        mean_diff = float(np.mean(diff))
        # Paired t-test
        t_stat, p_val = ttest_rel(diag_scores, off_scores)
        p_vals_raw.append(p_val)
        contrasts[b] = {
            "diag_mean": float(np.mean(diag_scores)),
            "off_diag_mean": float(np.mean(off_scores)),
            "mean_diff": mean_diff,
            "t_stat": float(t_stat),
            "raw_p_value": float(p_val)
        }
        
    # Holm-Bonferroni correction
    _, p_corrected, _, _ = multipletests(p_vals_raw, method="holm")
    for i, b in enumerate(baseline_keys):
        contrasts[b]["holm_p_value"] = float(p_corrected[i])
        print(f"Row {b:<5}: Diag={contrasts[b]['diag_mean']:.4f}, Off={contrasts[b]['off_diag_mean']:.4f}, Diff={contrasts[b]['mean_diff']:+.4f}, p_raw={contrasts[b]['raw_p_value']:.4f}, p_holm={contrasts[b]['holm_p_value']:.4f}")
        
    # Sentence Length interaction: bin by content_len
    # Short: <= 10, Medium: 11-20, Long: > 20
    sub["len_bin"] = pd.cut(sub["content_len"], bins=[0, 10, 20, 1000], labels=["Short", "Medium", "Long"])
    length_effects = {}
    for lbin in ["Short", "Medium", "Long"]:
        bin_sub = sub[sub["len_bin"] == lbin]
        bin_model = smf.mixedlm(
            "norm_comp_aopc ~ C(baseline) + C(scheme) + is_matched",
            data=bin_sub,
            groups=bin_sub["unique_id"]
        ).fit()
        length_effects[lbin] = {
            "n_examples": int(bin_sub["unique_id"].nunique()),
            "matched_coef": float(bin_model.params["is_matched"]),
            "matched_se": float(bin_model.bse["is_matched"]),
            "p_value": float(bin_model.pvalues["is_matched"]),
            "ci_lower": float(bin_model.conf_int().loc["is_matched", 0]),
            "ci_upper": float(bin_model.conf_int().loc["is_matched", 1]),
        }
        print(f"Length Bin {lbin:<6} (N={length_effects[lbin]['n_examples']}): Coef={length_effects[lbin]['matched_coef']:.4f} (95% CI [{length_effects[lbin]['ci_lower']:.4f}, {length_effects[lbin]['ci_upper']:.4f}]), p={length_effects[lbin]['p_value']:.4f}")

    return {
        "lmm_matched_coef": matched_coef,
        "lmm_matched_se": matched_se,
        "lmm_matched_p": matched_p,
        "lmm_ci_lower": ci_lower,
        "lmm_ci_upper": ci_upper,
        "variance_explained_pct": var_explained,
        "per_baseline_contrasts": contrasts,
        "length_binned_effects": length_effects,
    }

def analyze_dual_worst_case(df):
    """
    Computes worst-case robustness under BOTH definitions across:
    1. In-domain schemes: PAD, MASK, UNK, ZERO_EMB
    2. Structural + In-domain: PAD, MASK, UNK, ZERO_EMB, DELETE
    3. All schemes including held-out: + RANDOM_TOKEN, MLM_INFILL
    """
    print("\n--- STEP 1a & STEP 4: DUAL WORST-CASE ROBUSTNESS ---")
    methods = ["NP_IG", "UNIFORM_IG", "IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO", "GRAD_X_INP"]
    
    scheme_groups = {
        "in_domain_4": ["PAD", "MASK", "UNK", "ZERO_EMB"],
        "with_delete_5": ["PAD", "MASK", "UNK", "ZERO_EMB", "DELETE"],
        "all_7": ["PAD", "MASK", "UNK", "ZERO_EMB", "DELETE", "RANDOM_TOKEN", "MLM_INFILL"],
    }
    
    results = {}
    for group_name, schemes in scheme_groups.items():
        results[group_name] = {}
        for m in methods:
            sub = df[(df["method"] == m) & (df["scheme"].isin(schemes))]
            # Def 1: Mean of per-example minimum across schemes
            per_ex_worst = sub.groupby("unique_id")["norm_comp_aopc"].min()
            def1_score = float(per_ex_worst.mean())
            
            # Def 2: Minimum across schemes of per-scheme means
            per_scheme_means = sub.groupby("scheme")["norm_comp_aopc"].mean()
            def2_score = float(per_scheme_means.min())
            worst_scheme = str(per_scheme_means.idxmin())
            
            results[group_name][m] = {
                "def1_mean_of_min": def1_score,
                "def2_min_of_means": def2_score,
                "def2_worst_scheme": worst_scheme,
                "scheme_means": {s: float(per_scheme_means[s]) for s in schemes}
            }
            
    # Print comparison table for in_domain_4
    print("\nDual Worst-Case Table (In-Domain Schemes: PAD, MASK, UNK, ZERO_EMB):")
    print(f"{'Method':<12} | {'Def 1 (Mean of Min)':<20} | {'Def 2 (Min of Means)':<20} | {'Def 2 Worst Scheme':<15}")
    print("-" * 75)
    for m in methods:
        res = results["in_domain_4"][m]
        print(f"{m:<12} | {res['def1_mean_of_min']:<20.4f} | {res['def2_min_of_means']:<20.4f} | {res['def2_worst_scheme']:<15}")
        
    print("\nDual Worst-Case Table (All 7 Schemes including Held-Out):")
    print(f"{'Method':<12} | {'Def 1 (Mean of Min)':<20} | {'Def 2 (Min of Means)':<20} | {'Def 2 Worst Scheme':<15}")
    print("-" * 75)
    for m in methods:
        res = results["all_7"][m]
        print(f"{m:<12} | {res['def1_mean_of_min']:<20.4f} | {res['def2_min_of_means']:<20.4f} | {res['def2_worst_scheme']:<15}")
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_csv", default="project/results/sst2_revision_all450.csv")
    parser.add_argument("--steps_csv", default="project/results/steps_stability.csv")
    parser.add_argument("--ablation_csv", default="project/results/np_ig_ablation.csv")
    parser.add_argument("--output_json", default="project/results/revision_analysis_results.json")
    args = parser.parse_args()
    
    print(f"Loading data from {args.main_csv}...")
    df = pd.read_csv(args.main_csv)
    df["unique_id"] = df["seed"].astype(str) + "_" + df["example_id"].astype(str)
    
    # 1. H1 Linear Mixed Model
    lmm_results = analyze_lmm_h1(df)
    
    # 2. Dual Worst Case Analysis
    worst_case_results = analyze_dual_worst_case(df)
    
    # 3. Steps Stability Summary
    steps_results = {}
    if os.path.exists(args.steps_csv):
        sdf = pd.read_csv(args.steps_csv)
        for b in BASELINES:
            bsub = sdf[sdf["baseline"] == b]
            steps_results[b] = {
                "delta_m20": float(bsub["delta_m20"].mean()),
                "delta_m50": float(bsub["delta_m50"].mean()),
                "delta_m100": float(bsub["delta_m100"].mean()),
                "spearman_20_50": float(bsub["spearman_20_50"].mean()),
                "spearman_50_100": float(bsub["spearman_50_100"].mean()),
                "spearman_20_100": float(bsub["spearman_20_100"].mean()),
            }
            
    # 4. NP-IG Ablation Summary
    ablation_results = {}
    if os.path.exists(args.ablation_csv):
        adf = pd.read_csv(args.ablation_csv)
        configs = adf["config"].unique()
        for cfg in configs:
            csub = adf[adf["config"] == cfg]
            # Worst-case Def 2
            means = csub.groupby("scheme")["norm_comp_aopc"].mean()
            ablation_results[cfg] = {
                "mean_entropy": float(csub["entropy"].mean()),
                "mean_mask_weight": float(csub["weight_mask"].mean()),
                "overall_mean_comp": float(csub["norm_comp_aopc"].mean()),
                "def2_worst_case": float(means.min()),
                "def2_worst_scheme": str(means.idxmin())
            }
            
    full_analysis = {
        "h1_lmm": lmm_results,
        "worst_case": worst_case_results,
        "steps_stability": steps_results,
        "np_ig_ablation": ablation_results,
    }
    
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2)
    print(f"\nAnalysis saved to {args.output_json}!")

if __name__ == "__main__":
    main()
