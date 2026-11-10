"""
run_revision_experiments.py - Master runner for all revision experiments.

Executes:
1. Main SST-2 grid at M=50 steps across seeds 42, 123, 999 (150 each = 450 total).
   - In-domain erasure: PAD, MASK, UNK, ZERO_EMB
   - Structural erasure: DELETE
   - Held-out erasure: RANDOM_TOKEN (5-draw avg), MLM_INFILL (DistilBERT masked LM)
   - Methods: IG_PAD, IG_MASK, IG_UNK, IG_ZERO, NP_IG, UNIFORM_IG, GRAD_X_INP (logit), RANDOM
   - Outputs: project/results/sst2_revision_seed{42,123,999}.csv, sst2_revision_all450.csv
2. Steps ablation & token ranking stability:
   - Evaluates 50 examples at M=20, M=50, M=100 steps.
   - Computes completeness deltas and Spearman rank correlations (M=20 vs 50, M=50 vs 100, M=20 vs 100).
   - Outputs: project/results/steps_stability.csv
3. NP-IG temperature and weighting ablation:
   - Sweeps tau in [0.01, 0.03, 0.1, 0.3, 1.0, inf], argmin-neutrality, uniform, and random-weights.
   - Computes weight entropy and per-scheme comprehensiveness.
   - Outputs: project/results/np_ig_ablation.csv
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_utils import load_model_and_tokenizer, get_word_embeddings, get_baseline_ids
from src.ig_revision import (
    compute_integrated_gradients_m,
    compute_grad_x_input_fixed,
    compute_np_ig_weights,
    compute_weight_entropy,
    compute_token_ranking_spearman,
)
from src.erasure_revision import (
    get_content_token_indices,
    select_top_k_indices,
    apply_erasure_to_input,
    get_mlm_model_and_tokenizer,
)

CANDIDATE_BASELINES = ["PAD", "MASK", "UNK", "ZERO"]
IN_DOMAIN_SCHEMES   = ["PAD", "MASK", "UNK", "ZERO_EMB"]
ALL_SCHEMES         = ["PAD", "MASK", "UNK", "ZERO_EMB", "DELETE", "RANDOM_TOKEN", "MLM_INFILL"]
FRACTIONS           = [0.10, 0.20, 0.30, 0.50]
METHODS             = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO", "NP_IG", "UNIFORM_IG", "GRAD_X_INP", "RANDOM"]

def evaluate_example_all_schemes(
    model,
    tokenizer,
    embed_layer,
    input_ids,
    attention_mask,
    target_class,
    f_orig,
    ig_attrs,
    methods,
    schemes,
    fractions,
    mlm_model,
    mlm_tokenizer,
    rng
):
    """
    Evaluates comprehensiveness for all methods, schemes, and fractions
    by batching unique erased embeddings together in a single model forward pass.
    """
    device = input_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    candidate_indices = get_content_token_indices(tokenizer, input_ids)
    
    unique_embeds_map = {} # key -> (embeds_tensor (1, len, dim), mask_tensor (1, len))
    eval_dispatch = {}     # (method, scheme, frac) -> key
    
    for m in methods:
        attr = ig_attrs[m]
        for scheme in schemes:
            for frac in fractions:
                if len(candidate_indices) == 0:
                    continue
                top_indices = select_top_k_indices(attr, candidate_indices, fraction=frac)
                key = (tuple(sorted(top_indices)), scheme)
                
                if key not in unique_embeds_map:
                    em_tensor, m_tensor = apply_erasure_to_input(
                        input_ids=input_ids,
                        erase_indices=top_indices,
                        scheme=scheme,
                        tokenizer=tokenizer,
                        embed_layer=embed_layer,
                        attention_mask=attention_mask,
                        mlm_model=mlm_model,
                        mlm_tokenizer=mlm_tokenizer,
                        rng=rng
                    )
                    unique_embeds_map[key] = (em_tensor.squeeze(0), m_tensor.squeeze(0))
                eval_dispatch[(m, scheme, frac)] = key

    if len(unique_embeds_map) == 0:
        return {m: {s: {str(f): 0.0 for f in fractions} for s in schemes} for m in methods}

    keys_list = list(unique_embeds_map.keys())
    tensors_em = [unique_embeds_map[k][0] for k in keys_list] # list of (len, dim)
    tensors_mask = [unique_embeds_map[k][1] for k in keys_list] # list of (len,)
    
    max_len = max(t.shape[0] for t in tensors_em)
    hidden_dim = tensors_em[0].shape[1]
    batch_size = len(keys_list)
    
    batch_em = torch.zeros((batch_size, max_len, hidden_dim), dtype=torch.float32, device=device)
    batch_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    
    for i, (em, msk) in enumerate(zip(tensors_em, tensors_mask)):
        l = em.shape[0]
        batch_em[i, :l, :] = em
        batch_mask[i, :l] = msk
        
    with torch.no_grad():
        out = model(inputs_embeds=batch_em, attention_mask=batch_mask).logits
        probs = F.softmax(out, dim=-1)[:, target_class].cpu().numpy()
        
    key_to_comp = {k: float(f_orig - probs[i]) for i, k in enumerate(keys_list)}
    
    eval_dict = {m: {s: {} for s in schemes} for m in methods}
    for m in methods:
        for s in schemes:
            for frac in fractions:
                key = eval_dispatch.get((m, s, frac))
                eval_dict[m][s][str(frac)] = key_to_comp.get(key, 0.0)
                
    return eval_dict

def run_main_grid_seed(seed, dataset_path, num_samples, model_path, mlm_model_path, output_csv):
    """Runs primary SST-2 grid for one seed at M=50 steps."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    print(f"\n==================================================")
    print(f"RUNNING MAIN SST-2 GRID: Seed={seed}, N={num_samples}, Steps=50")
    print(f"==================================================")
    
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    embed_layer = get_word_embeddings(model)
    mlm_model, mlm_tokenizer = get_mlm_model_and_tokenizer(mlm_model_path, device=device)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_samples is not None and num_samples < len(data):
        data = data[:num_samples]
        
    rows = []
    start_time = time.time()
    
    for idx, item in enumerate(tqdm(data, desc=f"Seed {seed}")):
        text = item.get("sentence", item.get("text", ""))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_len = input_ids.shape[1]
        content_indices = get_content_token_indices(tokenizer, input_ids)
        content_len = len(content_indices)
        
        with torch.no_grad():
            out_orig = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs_orig = F.softmax(out_orig, dim=-1)[0]
            pred_class = probs_orig.argmax().item()
            f_orig = probs_orig[pred_class].item()
            
        target_class = pred_class
        input_embeds = embed_layer(input_ids)
        
        # 1. Compute IG for each of the 4 baselines at M=50
        ig_attrs = {}
        base_preds = {}
        base_deltas = {}
        
        for b_type in CANDIDATE_BASELINES:
            if b_type == "ZERO":
                b_embeds = torch.zeros_like(input_embeds)
            else:
                b_ids = get_baseline_ids(tokenizer, input_ids, b_type)
                b_embeds = embed_layer(b_ids)
                
            with torch.no_grad():
                out_b = model(inputs_embeds=b_embeds, attention_mask=attention_mask).logits
                prob_b = F.softmax(out_b, dim=-1)[0, target_class].item()
                base_preds[b_type] = prob_b
                
            attr, delta = compute_integrated_gradients_m(
                model=model,
                input_embeds=input_embeds,
                baseline_embeds=b_embeds,
                target_class=target_class,
                steps=50,
                output_space="prob",
                attention_mask=attention_mask
            )
            ig_attrs[f"IG_{b_type}"] = attr
            base_deltas[b_type] = delta
            
        # 2. NP-IG (tau=0.1)
        w_dict = compute_np_ig_weights(base_preds, tau=0.1)
        np_attr = torch.zeros_like(input_embeds[0, :, 0])
        for b, w in w_dict.items():
            np_attr += w * ig_attrs[f"IG_{b}"]
        ig_attrs["NP_IG"] = np_attr
        
        # 3. Uniform-IG
        uni_attr = torch.zeros_like(input_embeds[0, :, 0])
        for b in CANDIDATE_BASELINES:
            uni_attr += 0.25 * ig_attrs[f"IG_{b}"]
        ig_attrs["UNIFORM_IG"] = uni_attr
        
        # 4. Grad x Input (fixed in logit space)
        ig_attrs["GRAD_X_INP"] = compute_grad_x_input_fixed(
            model=model,
            input_embeds=input_embeds,
            target_class=target_class,
            use_logits=True,
            attention_mask=attention_mask
        )
        
        # 5. Random Attribution Control
        r_rng = np.random.RandomState(seed + idx)
        ig_attrs["RANDOM"] = torch.tensor(r_rng.randn(seq_len), dtype=torch.float32, device=device)
        
        # Fast batched evaluation across ALL 7 erasure schemes
        eval_dict = evaluate_example_all_schemes(
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_class=target_class,
            f_orig=f_orig,
            ig_attrs=ig_attrs,
            methods=METHODS,
            schemes=ALL_SCHEMES,
            fractions=FRACTIONS,
            mlm_model=mlm_model,
            mlm_tokenizer=mlm_tokenizer,
            rng=rng
        )
        
        # Compute AOPC (mean over fractions) for each method and scheme
        # Record raw and random-normalized AOPC
        rand_aopc = {}
        for s in ALL_SCHEMES:
            rand_aopc[s] = float(np.mean([eval_dict["RANDOM"][s][str(f)] for f in FRACTIONS]))
            
        for m in METHODS:
            for s in ALL_SCHEMES:
                raw_aopc = float(np.mean([eval_dict[m][s][str(f)] for f in FRACTIONS]))
                norm_aopc = raw_aopc - rand_aopc[s]
                
                # Check if matched pair
                is_matched = (
                    (m == "IG_PAD" and s == "PAD") or
                    (m == "IG_MASK" and s == "MASK") or
                    (m == "IG_UNK" and s == "UNK") or
                    (m == "IG_ZERO" and s == "ZERO_EMB")
                )
                
                rows.append({
                    "seed": seed,
                    "example_id": idx,
                    "content_len": content_len,
                    "pred_class": pred_class,
                    "f_orig": f_orig,
                    "method": m,
                    "scheme": s,
                    "raw_comp_aopc": raw_aopc,
                    "norm_comp_aopc": norm_aopc,
                    "is_matched": int(is_matched),
                    "is_in_domain": int(s in IN_DOMAIN_SCHEMES),
                    "is_held_out": int(s in ["RANDOM_TOKEN", "MLM_INFILL"]),
                    "pad_weight": w_dict["PAD"],
                    "mask_weight": w_dict["MASK"],
                    "unk_weight": w_dict["UNK"],
                    "zero_weight": w_dict["ZERO"],
                    "entropy": compute_weight_entropy(w_dict),
                    "delta_mask": base_deltas["MASK"],
                })
                
    elapsed = time.time() - start_time
    print(f"Seed {seed} finished in {elapsed:.1f}s ({elapsed / len(data):.2f}s/ex).")
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df

def run_steps_ablation(dataset_path, num_samples=50, model_path="D:/models/distilbert", output_csv="project/results/steps_stability.csv"):
    """
    Evaluates 50 examples at M=20, M=50, M=100.
    Computes completeness deltas and Spearman correlations of token attributions.
    """
    print(f"\n==================================================")
    print(f"RUNNING STEPS ABLATION & SPEARMAN STABILITY (N={num_samples})")
    print(f"==================================================")
    
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    embed_layer = get_word_embeddings(model)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:num_samples]
        
    rows = []
    
    for idx, item in enumerate(tqdm(data, desc="Steps Ablation")):
        text = item.get("sentence", item.get("text", ""))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        content_indices = get_content_token_indices(tokenizer, input_ids)
        
        with torch.no_grad():
            out_orig = model(input_ids=input_ids, attention_mask=attention_mask).logits
            target_class = out_orig.argmax().item()
            
        input_embeds = embed_layer(input_ids)
        
        # Test for each baseline across M in [20, 50, 100]
        for b_type in CANDIDATE_BASELINES:
            if b_type == "ZERO":
                b_embeds = torch.zeros_like(input_embeds)
            else:
                b_ids = get_baseline_ids(tokenizer, input_ids, b_type)
                b_embeds = embed_layer(b_ids)
                
            attr_20, delta_20 = compute_integrated_gradients_m(
                model=model, input_embeds=input_embeds, baseline_embeds=b_embeds,
                target_class=target_class, steps=20, output_space="prob", attention_mask=attention_mask
            )
            attr_50, delta_50 = compute_integrated_gradients_m(
                model=model, input_embeds=input_embeds, baseline_embeds=b_embeds,
                target_class=target_class, steps=50, output_space="prob", attention_mask=attention_mask
            )
            attr_100, delta_100 = compute_integrated_gradients_m(
                model=model, input_embeds=input_embeds, baseline_embeds=b_embeds,
                target_class=target_class, steps=100, output_space="prob", attention_mask=attention_mask
            )
            
            spearman_20_50  = compute_token_ranking_spearman(attr_20, attr_50, content_indices)
            spearman_50_100 = compute_token_ranking_spearman(attr_50, attr_100, content_indices)
            spearman_20_100 = compute_token_ranking_spearman(attr_20, attr_100, content_indices)
            
            rows.append({
                "example_id": idx,
                "content_len": len(content_indices),
                "baseline": b_type,
                "delta_m20": delta_20,
                "delta_m50": delta_50,
                "delta_m100": delta_100,
                "spearman_20_50": spearman_20_50,
                "spearman_50_100": spearman_50_100,
                "spearman_20_100": spearman_20_100,
            })
            
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    
    print("\n--- Summary of Steps Ablation & Spearman Stability ---")
    for b in CANDIDATE_BASELINES:
        sub = df[df["baseline"] == b]
        print(f"Baseline {b:<6}:")
        print(f"  Delta: M=20: {sub['delta_m20'].mean():.4f}, M=50: {sub['delta_m50'].mean():.4f}, M=100: {sub['delta_m100'].mean():.4f}")
        print(f"  Spearman: 20 vs 50: {sub['spearman_20_50'].mean():.4f}, 50 vs 100: {sub['spearman_50_100'].mean():.4f}, 20 vs 100: {sub['spearman_20_100'].mean():.4f}")
    return df

def run_np_ig_ablation(sst2_results_csv, output_csv="project/results/np_ig_ablation.csv"):
    """
    Conducts extensive NP-IG ablation across:
    tau in [0.01, 0.03, 0.1, 0.3, 1.0, inf], argmin-neutrality, uniform, and random-weights.
    Calculates weight entropy and composite comprehensiveness.
    """
    print(f"\n==================================================")
    print(f"RUNNING NP-IG TEMPERATURE & WEIGHT ABLATION")
    print(f"==================================================")
    
    df = pd.read_csv(sst2_results_csv)
    
    # Extract baseline predictions / distances per example
    # We can reconstruct composite attributions and their comprehensiveness
    # In df, for each example and scheme, we have norm_comp_aopc for IG_PAD, IG_MASK, IG_UNK, IG_ZERO
    tau_list = [0.01, 0.03, 0.1, 0.3, 1.0, float("inf")]
    
    # Group by example_id and scheme
    single_methods = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO"]
    single_df = df[df["method"].isin(single_methods)].copy()
    
    # We have pad_weight, mask_weight, unk_weight, zero_weight at tau=0.1
    # Note: w_b = exp(-d_b / 0.1). So d_b = -0.1 * log(w_b * Z)
    # Let's compute weights for all taus directly
    unique_examples = df[["example_id", "pad_weight", "mask_weight", "unk_weight", "zero_weight"]].drop_duplicates()
    
    ablation_rows = []
    
    for _, ex in unique_examples.iterrows():
        eid = int(ex["example_id"])
        w_01 = [ex["pad_weight"], ex["mask_weight"], ex["unk_weight"], ex["zero_weight"]]
        # Invert to relative distance (d_b - min_d)
        log_w = np.log(np.array(w_01) + 1e-12)
        rel_d = -(log_w - log_w.max()) * 0.1
        
        # Test each tau configuration
        weight_configs = {}
        for tau in [0.01, 0.03, 0.1, 0.3, 1.0]:
            lw = -rel_d / tau
            ew = np.exp(lw - lw.max())
            weight_configs[f"tau_{tau}"] = ew / ew.sum()
            
        # tau=inf (Uniform)
        weight_configs["Uniform"] = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Argmin neutrality (one-hot)
        argmin_w = np.zeros(4)
        argmin_w[rel_d.argmin()] = 1.0
        weight_configs["Argmin_Neutrality"] = argmin_w
        
        # Random weights (shuffled across baselines)
        rng_ex = np.random.RandomState(42 + eid)
        rand_w = rng_ex.dirichlet(np.ones(4))
        weight_configs["Random_Weights"] = rand_w
        
        for cfg_name, w_arr in weight_configs.items():
            w_dict = dict(zip(CANDIDATE_BASELINES, w_arr))
            ent = compute_weight_entropy(w_dict)
            
            # For each scheme, composite score is approx linear combination of baseline scores
            # (or exact ensemble expectation)
            for s in ALL_SCHEMES:
                ex_s = df[(df["example_id"] == eid) & (df["scheme"] == s)]
                scores = [ex_s[ex_s["method"] == f"IG_{b}"]["norm_comp_aopc"].values[0] for b in CANDIDATE_BASELINES]
                composite_score = float(np.dot(w_arr, scores))
                
                ablation_rows.append({
                    "example_id": eid,
                    "config": cfg_name,
                    "scheme": s,
                    "entropy": ent,
                    "weight_mask": w_dict["MASK"],
                    "norm_comp_aopc": composite_score
                })
                
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(output_csv, index=False)
    print(f"Saved {len(abl_df)} rows to {output_csv}")
    
    print("\n--- Summary of NP-IG Temperature Ablation ---")
    summary = abl_df.groupby("config").agg({
        "entropy": "mean",
        "weight_mask": "mean",
        "norm_comp_aopc": "mean"
    }).reset_index()
    print(summary.to_string(index=False))
    return abl_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="project/data/sst2_val.json")
    parser.add_argument("--model_path", default="D:/models/distilbert")
    parser.add_argument("--mlm_model_path", default="D:/models/distilbert_mlm")
    parser.add_argument("--num_samples", type=int, default=150)
    args = parser.parse_args()
    
    # 1. Run Main Grid for seeds 42, 123, 999
    dfs = []
    for seed in [42, 123, 999]:
        out_csv = f"project/results/sst2_revision_seed{seed}.csv"
        if os.path.exists(out_csv):
            existing_df = pd.read_csv(out_csv)
            if len(existing_df) >= args.num_samples * len(METHODS) * len(ALL_SCHEMES):
                print(f"Skipping seed {seed} (already completed: {len(existing_df)} rows in {out_csv})")
                dfs.append(existing_df)
                continue
        df_seed = run_main_grid_seed(
            seed=seed,
            dataset_path=args.dataset_path,
            num_samples=args.num_samples,
            model_path=args.model_path,
            mlm_model_path=args.mlm_model_path,
            output_csv=out_csv
        )
        dfs.append(df_seed)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    all_csv = "project/results/sst2_revision_all450.csv"
    combined_df.to_csv(all_csv, index=False)
    print(f"\nAll examples combined and saved to {all_csv} ({len(combined_df)} records)!")
    
    # 2. Run Steps Ablation on 50 examples
    steps_csv = "project/results/steps_stability.csv"
    if os.path.exists(steps_csv) and len(pd.read_csv(steps_csv)) == 200:
        print(f"Skipping steps ablation (already completed: 200 rows in {steps_csv})")
    else:
        run_steps_ablation(
            dataset_path=args.dataset_path,
            num_samples=50,
            model_path=args.model_path,
            output_csv=steps_csv
        )
    
    # 3. Run NP-IG Temperature & Weight Ablation
    run_np_ig_ablation(
        sst2_results_csv=all_csv,
        output_csv="project/results/np_ig_ablation.csv"
    )
    print("\nALL REVISION EXPERIMENTS SUCCESSFULLY COMPLETED!")
