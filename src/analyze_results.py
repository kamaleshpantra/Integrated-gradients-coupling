"""
analyze_results.py  -  Step 8: Analyze experiment results.

Usage:
    python project/src/analyze_results.py \
        --result_files project/results/sst2_main_seed42.json project/results/sst2_main_seed123.json project/results/sst2_main_seed999.json \
        --output_prefix project/results/analysis_sst2

Outputs:
    <output_prefix>_stats.json    - full statistical tables
    <output_prefix>_matrix.json   - 4x4 AOPC matrix (for heatmap)
    <output_prefix>_rankings.json - Kendall tau matrix + per-scheme rankings
    <output_prefix>_h1.json       - H1 diagonal vs off-diagonal test results
    <output_prefix>_h2.json       - H2 NP-IG vs single baselines results
    <output_prefix>_summary.txt   - human-readable summary
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import combinations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import (
    aggregate_experiment_results,
    bootstrap_ci,
    paired_bootstrap_test,
    compute_rankings_and_kendall_tau,
)

# ── Constants ────────────────────────────────────────────────────────────────
SINGLE_BASELINE_METHODS = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO"]
ERASURE_SCHEMES         = ["PAD", "MASK", "UNK", "DELETE"]
CANDIDATE_METHODS       = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO",
                            "NP_IG", "UNIFORM_IG", "GRAD_X_INP", "RANDOM"]
FRACS                   = ("0.1", "0.2", "0.3", "0.5")

# Map: IG_XXX -> corresponding erasure scheme for diagonal
BASELINE_TO_ERASURE = {"IG_PAD": "PAD", "IG_MASK": "MASK",
                       "IG_UNK": "UNK", "IG_ZERO": "DELETE"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_merge_records(result_files):
    """Load multiple seed files and concatenate records."""
    all_records = []
    seeds_seen = []
    for fp in result_files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        all_records.extend(d["records"])
        seeds_seen.append(d["metadata"].get("seed", "?"))
        print(f"  Loaded {len(d['records'])} records from {fp} (seed={d['metadata'].get('seed','?')})")
    print(f"  Total records after merge: {len(all_records)} (seeds: {seeds_seen})")
    return all_records


def build_aopc_arrays(records):
    """
    Returns dict[method][scheme] -> np.array of per-example AOPC(comp) values.
    Also returns the same for suff.
    """
    comp_arrs = {m: {s: [] for s in ERASURE_SCHEMES} for m in CANDIDATE_METHODS}
    suff_arrs = {m: {s: [] for s in ERASURE_SCHEMES} for m in CANDIDATE_METHODS}
    for r in records:
        for m in CANDIDATE_METHODS:
            for s in ERASURE_SCHEMES:
                vals_c = [r["evaluations"][m][s][f]["comp"] for f in FRACS]
                vals_s = [r["evaluations"][m][s][f]["suff"] for f in FRACS]
                comp_arrs[m][s].append(float(np.mean(vals_c)))
                suff_arrs[m][s].append(float(np.mean(vals_s)))
    return {m: {s: np.array(comp_arrs[m][s]) for s in ERASURE_SCHEMES} for m in CANDIDATE_METHODS}, \
           {m: {s: np.array(suff_arrs[m][s]) for s in ERASURE_SCHEMES} for m in CANDIDATE_METHODS}


def random_normalize(comp_arrs, suff_arrs):
    """Subtract RANDOM baseline's AOPC from each method (per example, per scheme)."""
    norm_c = {}
    norm_s = {}
    for m in CANDIDATE_METHODS:
        norm_c[m] = {}
        norm_s[m] = {}
        for s in ERASURE_SCHEMES:
            norm_c[m][s] = comp_arrs[m][s] - comp_arrs["RANDOM"][s]
            norm_s[m][s] = suff_arrs["RANDOM"][s] - suff_arrs[m][s]
    return norm_c, norm_s


def compute_stats_table(arrs):
    """Full bootstrap CI table: dict[method][scheme] -> {mean, ci_lower, ci_upper, std_err}."""
    table = {}
    for m in CANDIDATE_METHODS:
        table[m] = {}
        for s in ERASURE_SCHEMES:
            table[m][s] = bootstrap_ci(arrs[m][s].tolist())
    return table


def test_h1(norm_comp):
    """
    H1: Diagonal comprehensiveness > off-diagonal for matched (method, erasure) pairs.
    For each matched pair (IG_XXX, XXX_erasure), test diagonal vs each off-diagonal.
    Returns dict with p-values and observed differences.
    """
    results = {}
    diagonal_vals = []
    off_diagonal_vals = []
    for method, erasure in BASELINE_TO_ERASURE.items():
        diag = norm_comp[method][erasure]
        off_diag_all = []
        pair_results = {}
        for other_erasure in ERASURE_SCHEMES:
            if other_erasure == erasure:
                continue
            off = norm_comp[method][other_erasure]
            test = paired_bootstrap_test(diag.tolist(), off.tolist())
            pair_results[other_erasure] = test
            off_diag_all.append(off)
            off_diagonal_vals.append(off)
        diagonal_vals.append(diag)
        results[f"{method}_vs_off_diag"] = {
            "diagonal_erasure": erasure,
            "diagonal_mean": float(diag.mean()),
            "off_diag_pairs": pair_results
        }

    # Global diagonal vs off-diagonal test (all diagonal cells vs all off-diagonal)
    all_diag = np.concatenate(diagonal_vals)
    all_off  = np.concatenate(off_diagonal_vals)
    # Trim to same length (off_diag is 3x longer)
    min_len = min(len(all_diag), len(all_off))
    global_test = paired_bootstrap_test(
        all_diag[:min_len].tolist(),
        all_off[:min_len].tolist()
    )
    results["GLOBAL_diagonal_vs_off_diagonal"] = global_test
    results["global_diagonal_mean"] = float(all_diag.mean())
    results["global_off_diagonal_mean"] = float(all_off.mean())
    return results


def test_h2(norm_comp):
    """
    H2: NP-IG has better worst-case comprehensiveness across erasure schemes
    compared to any single baseline.
    Also test mean rank superiority.
    """
    results = {}
    for method in SINGLE_BASELINE_METHODS:
        # Per erasure scheme: NP_IG vs method
        per_scheme = {}
        for s in ERASURE_SCHEMES:
            test = paired_bootstrap_test(norm_comp["NP_IG"][s].tolist(), norm_comp[method][s].tolist())
            per_scheme[s] = test

        # Worst-case: min over schemes of AOPC for each example
        npig_worst = np.stack([norm_comp["NP_IG"][s] for s in ERASURE_SCHEMES]).min(axis=0)
        meth_worst = np.stack([norm_comp[method][s] for s in ERASURE_SCHEMES]).min(axis=0)
        worst_case_test = paired_bootstrap_test(npig_worst.tolist(), meth_worst.tolist())

        results[method] = {
            "per_scheme_tests": per_scheme,
            "worst_case_test": worst_case_test,
            "npig_worst_case_mean": float(npig_worst.mean()),
            "method_worst_case_mean": float(meth_worst.mean()),
        }

    # Also: NP-IG vs UNIFORM_IG
    for ref in ["UNIFORM_IG", "GRAD_X_INP"]:
        npig_worst = np.stack([norm_comp["NP_IG"][s] for s in ERASURE_SCHEMES]).min(axis=0)
        ref_worst  = np.stack([norm_comp[ref][s]     for s in ERASURE_SCHEMES]).min(axis=0)
        worst_test = paired_bootstrap_test(npig_worst.tolist(), ref_worst.tolist())
        per_scheme = {}
        for s in ERASURE_SCHEMES:
            per_scheme[s] = paired_bootstrap_test(norm_comp["NP_IG"][s].tolist(), norm_comp[ref][s].tolist())
        results[ref] = {
            "per_scheme_tests": per_scheme,
            "worst_case_test": worst_test,
            "npig_worst_case_mean": float(npig_worst.mean()),
            f"{ref}_worst_case_mean": float(ref_worst.mean()),
        }

    return results


def compute_4x4_matrix(arrs, candidate_ig_methods=None):
    """Build the 4x4 (IG baseline) x (erasure scheme) mean AOPC matrix."""
    if candidate_ig_methods is None:
        candidate_ig_methods = SINGLE_BASELINE_METHODS
    matrix = {}
    for m in candidate_ig_methods:
        matrix[m] = {s: float(arrs[m][s].mean()) for s in ERASURE_SCHEMES}
    return matrix


def identify_qualitative_examples(records, norm_comp_arr, n=5):
    """
    Find examples where diagonal inflation is largest (most useful for paper).
    Returns top-n example IDs with their scores.
    """
    examples = []
    for i, r in enumerate(records):
        diag_scores = []
        off_scores  = []
        for method, erasure in BASELINE_TO_ERASURE.items():
            if method not in r["evaluations"]:
                continue
            vals_diag = [r["evaluations"][method][erasure][f]["comp"] for f in FRACS]
            diag_scores.append(np.mean(vals_diag))
            for s in ERASURE_SCHEMES:
                if s != erasure:
                    vals_off = [r["evaluations"][method][s][f]["comp"] for f in FRACS]
                    off_scores.append(np.mean(vals_off))
        if diag_scores and off_scores:
            inflation = np.mean(diag_scores) - np.mean(off_scores)
            examples.append({"id": r.get("id", i), "inflation": inflation,
                              "seq_len": r.get("seq_len"), "f_orig": r.get("f_orig"),
                              "np_weights": r.get("np_weights")})
    examples.sort(key=lambda x: x["inflation"], reverse=True)
    return examples[:n]


def render_summary(h1_res, h2_res, matrix, rankings, n_examples, output_prefix):
    """Write human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS SUMMARY")
    lines.append(f"N examples = {n_examples}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("── 4×4 Mean Normalized Comprehensiveness AOPC Matrix ──")
    header = f"{'Method':<14}" + "".join(f"{s:>10}" for s in ERASURE_SCHEMES)
    lines.append(header)
    for m in SINGLE_BASELINE_METHODS:
        row = f"{m:<14}" + "".join(f"{matrix[m][s]:>10.4f}" for s in ERASURE_SCHEMES)
        lines.append(row)
    lines.append("")

    lines.append("── H1: Diagonal Inflation ──")
    g = h1_res.get("GLOBAL_diagonal_vs_off_diagonal", {})
    lines.append(f"Global diagonal mean:     {h1_res.get('global_diagonal_mean', 'N/A'):.4f}")
    lines.append(f"Global off-diagonal mean: {h1_res.get('global_off_diagonal_mean', 'N/A'):.4f}")
    lines.append(f"Bootstrap p (diag > off): {g.get('bootstrap_p_value', 'N/A'):.4f}")
    lines.append(f"Wilcoxon p  (diag > off): {g.get('wilcoxon_p_value', 'N/A')}")
    lines.append("")

    lines.append("── H2: NP-IG Worst-Case vs Single Baselines ──")
    npig_worst = h2_res.get("IG_PAD", {}).get("npig_worst_case_mean", "N/A")
    lines.append(f"NP-IG worst-case mean: {npig_worst:.4f}" if isinstance(npig_worst, float) else "NP-IG worst-case mean: N/A")
    for method in SINGLE_BASELINE_METHODS + ["UNIFORM_IG", "GRAD_X_INP"]:
        wt = h2_res.get(method, {}).get("worst_case_test", {})
        wm = h2_res.get(method, {}).get("method_worst_case_mean") or h2_res.get(method, {}).get(f"{method}_worst_case_mean")
        lines.append(f"  vs {method:<12}: method_worst={wm:.4f}  p(NP>meth)={wt.get('bootstrap_p_value','?'):.4f}" if wm else f"  vs {method}")
    lines.append("")

    lines.append("── Kendall Tau (method ranking correlation across erasure schemes) ──")
    tau_mat = rankings.get("kendall_tau_matrix", {})
    header2 = f"{'':>8}" + "".join(f"{s:>10}" for s in ERASURE_SCHEMES)
    lines.append(header2)
    for s1 in ERASURE_SCHEMES:
        row = f"{s1:>8}"
        for s2 in ERASURE_SCHEMES:
            row += f"{tau_mat.get(s1, {}).get(s2, {}).get('tau', 0.0):>10.3f}"
        lines.append(row)
    lines.append("")

    summary_txt = "\n".join(lines)
    out_path = output_prefix + "_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)
    print(summary_txt)
    print(f"\nSummary saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_files", nargs="+", required=True)
    parser.add_argument("--output_prefix", type=str, default="project/results/analysis")
    args = parser.parse_args()

    print("Loading records...")
    records = load_and_merge_records(args.result_files)

    print("Building AOPC arrays...")
    comp_arrs, suff_arrs = build_aopc_arrays(records)

    print("Normalizing by RANDOM control...")
    norm_comp, norm_suff = random_normalize(comp_arrs, suff_arrs)

    print("Computing bootstrap CI tables...")
    stats_comp = compute_stats_table(norm_comp)
    stats_suff = compute_stats_table(norm_suff)

    print("Computing 4×4 matrix...")
    matrix_4x4 = compute_4x4_matrix(norm_comp)

    print("Computing rankings and Kendall tau...")
    rankings = compute_rankings_and_kendall_tau(norm_comp, CANDIDATE_METHODS, ERASURE_SCHEMES)

    print("Testing H1 (diagonal inflation)...")
    h1_results = test_h1(norm_comp)

    print("Testing H2 (NP-IG robustness)...")
    h2_results = test_h2(norm_comp)

    print("Finding qualitative examples...")
    qual_examples = identify_qualitative_examples(records, norm_comp)

    # Save all outputs first
    def save(obj, suffix):
        path = args.output_prefix + suffix
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"Saved {path}")

    save({"norm_comp": {m: {s: stats_comp[m][s] for s in ERASURE_SCHEMES} for m in CANDIDATE_METHODS},
          "norm_suff": {m: {s: stats_suff[m][s] for s in ERASURE_SCHEMES} for m in CANDIDATE_METHODS}},
         "_stats.json")
    save({"norm_comp_matrix": matrix_4x4,
          "norm_suff_matrix": compute_4x4_matrix(norm_suff)},
         "_matrix.json")
    save(rankings, "_rankings.json")
    save(h1_results, "_h1.json")
    save(h2_results, "_h2.json")
    save(qual_examples, "_qualitative.json")

    print("Rendering summary...")
    try:
        render_summary(h1_results, h2_results, matrix_4x4, rankings, len(records), args.output_prefix)
    except Exception as e:
        print(f"Warning: could not print summary to console ({e}), check summary file.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
