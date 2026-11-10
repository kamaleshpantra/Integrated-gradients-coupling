import numpy as np
from scipy.stats import kendalltau, wilcoxon

def compute_aopc(eval_record_for_method_scheme, metric="comp"):
    """
    Computes Area Over the Perturbation Curve across fractions.
    """
    vals = [frac_dict[metric] for frac_dict in eval_record_for_method_scheme.values()]
    return float(np.mean(vals))

def aggregate_experiment_results(records, candidate_methods=None, erasure_schemes=None, fractions=("0.1", "0.2", "0.3", "0.5")):
    """
    Aggregates per-example records into mean raw and random-normalized metrics,
    computing bootstrap confidence intervals.
    """
    if candidate_methods is None:
        candidate_methods = list(records[0]["evaluations"].keys())
    if erasure_schemes is None:
        erasure_schemes = list(records[0]["evaluations"][candidate_methods[0]].keys())
        
    n_examples = len(records)
    
    # Pre-extract per-example AOPC arrays: dict[method][scheme] -> list of length n_examples
    aopc_comp = {m: {s: [] for s in erasure_schemes} for m in candidate_methods}
    aopc_suff = {m: {s: [] for s in erasure_schemes} for m in candidate_methods}
    
    for r in records:
        for m in candidate_methods:
            for s in erasure_schemes:
                comp_val = compute_aopc(r["evaluations"][m][s], "comp")
                suff_val = compute_aopc(r["evaluations"][m][s], "suff")
                aopc_comp[m][s].append(comp_val)
                aopc_suff[m][s].append(suff_val)
                
    # Normalize by RANDOM control
    norm_comp = {m: {s: [] for s in erasure_schemes} for m in candidate_methods}
    norm_suff = {m: {s: [] for s in erasure_schemes} for m in candidate_methods}
    
    for i in range(n_examples):
        for s in erasure_schemes:
            rand_c = aopc_comp["RANDOM"][s][i]
            rand_s = aopc_suff["RANDOM"][s][i]
            for m in candidate_methods:
                norm_comp[m][s].append(aopc_comp[m][s][i] - rand_c)
                norm_suff[m][s].append(rand_s - aopc_suff[m][s][i])
                
    summary = {
        "raw_comp": aopc_comp,
        "raw_suff": aopc_suff,
        "norm_comp": norm_comp,
        "norm_suff": norm_suff
    }
    return summary

def bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=42):
    """
    Computes bootstrap mean and percentile confidence interval.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())
        
    boot_means = np.sort(boot_means)
    alpha = (1.0 - ci) / 2.0
    low_idx = int(alpha * n_resamples)
    high_idx = int((1.0 - alpha) * n_resamples)
    
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(boot_means[low_idx]),
        "ci_upper": float(boot_means[high_idx]),
        "std_err": float(np.std(boot_means))
    }

def paired_bootstrap_test(vals_a, vals_b, n_resamples=1000, seed=42):
    """
    Performs paired bootstrap hypothesis test: H0: E[A - B] <= 0 vs H1: E[A - B] > 0.
    """
    diffs = np.array(vals_a) - np.array(vals_b)
    obs_mean = np.mean(diffs)
    
    rng = np.random.RandomState(seed)
    n = len(diffs)
    # Centered diffs under H0
    centered = diffs - obs_mean
    
    count = 0
    for _ in range(n_resamples):
        resample = rng.choice(centered, size=n, replace=True)
        if resample.mean() >= obs_mean:
            count += 1
            
    p_value = (count + 1) / (n_resamples + 1)
    
    # Also compute Wilcoxon signed rank test
    try:
        w_res = wilcoxon(vals_a, vals_b, alternative="greater")
        wilcoxon_p = float(w_res.pvalue)
    except Exception:
        wilcoxon_p = None
        
    return {
        "observed_mean_difference": float(obs_mean),
        "bootstrap_p_value": float(p_value),
        "wilcoxon_p_value": wilcoxon_p
    }

def compute_rankings_and_kendall_tau(norm_comp_dict, candidate_methods, erasure_schemes):
    """
    Computes method rankings under each erasure scheme and pairwise Kendall's tau rank correlation.
    """
    scheme_means = {}
    scheme_ranks = {}
    
    for s in erasure_schemes:
        means = {m: np.mean(norm_comp_dict[m][s]) for m in candidate_methods}
        scheme_means[s] = means
        # Rank descending (highest comp = rank 1)
        sorted_methods = sorted(candidate_methods, key=lambda m: means[m], reverse=True)
        ranks = {m: rank + 1 for rank, m in enumerate(sorted_methods)}
        scheme_ranks[s] = ranks
        
    tau_matrix = {}
    for s1 in erasure_schemes:
        tau_matrix[s1] = {}
        for s2 in erasure_schemes:
            r1 = [scheme_ranks[s1][m] for m in candidate_methods]
            r2 = [scheme_ranks[s2][m] for m in candidate_methods]
            tau, p_val = kendalltau(r1, r2)
            tau_matrix[s1][s2] = {"tau": float(tau), "p_value": float(p_val)}
            
    return {
        "scheme_means": scheme_means,
        "scheme_ranks": scheme_ranks,
        "kendall_tau_matrix": tau_matrix
    }
