import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_utils import load_model_and_tokenizer, get_word_embeddings, get_baseline_ids
from src.ig import compute_integrated_gradients, compute_grad_x_input
from src.erasure import get_content_token_indices, select_top_k_indices, apply_erasure

def evaluate_example_batched(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    target_class,
    f_orig,
    ig_attrs,
    methods,
    erasure_schemes,
    fractions
):
    """
    Evaluates comprehensiveness and sufficiency for all methods, schemes, and fractions
    in a single batched model forward pass for maximum performance.
    """
    device = input_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    candidate_indices = get_content_token_indices(tokenizer, input_ids)
    
    # Store requests to batch
    # Key: (is_comp, tuple(erase_indices), scheme) -> list of (method, scheme, frac_str, is_comp)
    unique_inputs = {} # key -> (ids_tensor, mask_tensor)
    method_eval_map = {} # (method, scheme, frac_str, "comp"/"suff") -> key
    
    for m in methods:
        attr = ig_attrs[m]
        for scheme in erasure_schemes:
            for frac in fractions:
                frac_str = str(frac)
                if len(candidate_indices) == 0:
                    continue
                top_indices = select_top_k_indices(attr, candidate_indices, fraction=frac)
                other_indices = set(candidate_indices) - top_indices
                
                # 1. Comp key
                key_comp = ("comp", tuple(sorted(top_indices)), scheme)
                if key_comp not in unique_inputs:
                    c_ids, c_mask = apply_erasure(input_ids, top_indices, scheme, tokenizer, attention_mask)
                    unique_inputs[key_comp] = (c_ids.squeeze(0), c_mask.squeeze(0))
                method_eval_map[(m, scheme, frac_str, "comp")] = key_comp
                
                # 2. Suff key
                key_suff = ("suff", tuple(sorted(other_indices)), scheme)
                if key_suff not in unique_inputs:
                    s_ids, s_mask = apply_erasure(input_ids, other_indices, scheme, tokenizer, attention_mask)
                    unique_inputs[key_suff] = (s_ids.squeeze(0), s_mask.squeeze(0))
                method_eval_map[(m, scheme, frac_str, "suff")] = key_suff
                
    if len(unique_inputs) == 0:
        res = {m: {s: {str(f): {"comp": 0.0, "suff": 0.0} for f in fractions} for s in erasure_schemes} for m in methods}
        return res
        
    # Batch all unique erased sequences together
    keys_list = list(unique_inputs.keys())
    tensors_ids = [unique_inputs[k][0] for k in keys_list]
    tensors_masks = [unique_inputs[k][1] for k in keys_list]
    
    # Pad to maximum length across erased tensors
    max_len = max(t.shape[0] for t in tensors_ids)
    batch_ids = torch.full((len(keys_list), max_len), pad_id, dtype=torch.long, device=device)
    batch_mask = torch.zeros((len(keys_list), max_len), dtype=torch.long, device=device)
    
    for i, (t_id, t_m) in enumerate(zip(tensors_ids, tensors_masks)):
        l = t_id.shape[0]
        batch_ids[i, :l] = t_id
        batch_mask[i, :l] = t_m
        
    with torch.no_grad():
        out = model(input_ids=batch_ids, attention_mask=batch_mask).logits
        probs = F.softmax(out, dim=-1)[:, target_class].cpu().numpy()
        
    key_to_prob = {k: float(probs[i]) for i, k in enumerate(keys_list)}
    
    # Construct results dictionary
    eval_dict = {m: {s: {} for s in erasure_schemes} for m in methods}
    for m in methods:
        for s in erasure_schemes:
            for frac in fractions:
                frac_str = str(frac)
                key_c = method_eval_map.get((m, s, frac_str, "comp"))
                key_s = method_eval_map.get((m, s, frac_str, "suff"))
                f_comp = key_to_prob[key_c] if key_c is not None else f_orig
                f_suff = key_to_prob[key_s] if key_s is not None else f_orig
                eval_dict[m][s][frac_str] = {
                    "comp": float(f_orig - f_comp),
                    "suff": float(f_orig - f_suff)
                }
                
    return eval_dict

def run_evaluation(
    dataset_path,
    model_path="D:/models/distilbert",
    num_samples=None,
    steps=20,
    temperature=0.1,
    seed=42,
    output_file=None,
    fractions=(0.10, 0.20, 0.30, 0.50),
    erasure_schemes=("PAD", "MASK", "UNK", "DELETE"),
    output_space="prob"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Loading model from {model_path}...")
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    embed_layer = get_word_embeddings(model)
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if num_samples is not None and num_samples < len(data):
        data = data[:num_samples]
        
    print(f"Total examples to evaluate: {len(data)}")
    
    methods = ["IG_PAD", "IG_MASK", "IG_UNK", "IG_ZERO", "NP_IG", "UNIFORM_IG", "GRAD_X_INP", "RANDOM"]
    candidate_baselines = ["PAD", "MASK", "UNK", "ZERO"]
    
    records = []
    start_time = time.time()
    
    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        text = item.get("sentence", item.get("text", ""))
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            out_orig = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs_orig = F.softmax(out_orig, dim=-1)[0]
            pred_class = probs_orig.argmax().item()
            f_orig = probs_orig[pred_class].item()
            
        target_class = pred_class
        input_embeds = embed_layer(input_ids)
        
        # 1. Compute IG for each baseline
        ig_attrs = {}
        base_preds = {}
        base_deltas = {}
        
        for b_type in candidate_baselines:
            if b_type == "ZERO":
                b_embeds = torch.zeros_like(input_embeds)
            else:
                b_ids = get_baseline_ids(tokenizer, input_ids, b_type)
                b_embeds = embed_layer(b_ids)
                
            with torch.no_grad():
                out_b = model(inputs_embeds=b_embeds, attention_mask=attention_mask).logits
                prob_b = F.softmax(out_b, dim=-1)[0, target_class].item()
                base_preds[b_type] = prob_b
                
            attr, delta = compute_integrated_gradients(
                model=model,
                input_embeds=input_embeds,
                baseline_embeds=b_embeds,
                target_class=target_class,
                steps=steps,
                output_space=output_space,
                attention_mask=attention_mask
            )
            ig_attrs[f"IG_{b_type}"] = attr
            base_deltas[b_type] = delta
            
        # 2. NP-IG
        distances = [abs(base_preds[b] - 0.5) for b in candidate_baselines]
        log_weights = [-d / temperature for d in distances]
        max_log = max(log_weights)
        exp_weights = [np.exp(lw - max_log) for lw in log_weights]
        sum_exp = sum(exp_weights)
        weights = [ew / sum_exp for ew in exp_weights]
        weight_dict = dict(zip(candidate_baselines, weights))
        
        np_attr = torch.zeros_like(ig_attrs["IG_PAD"])
        for b, w in weight_dict.items():
            np_attr += w * ig_attrs[f"IG_{b}"]
        ig_attrs["NP_IG"] = np_attr
        
        # 3. UNIFORM_IG
        uniform_attr = torch.zeros_like(ig_attrs["IG_PAD"])
        for b in candidate_baselines:
            uniform_attr += 0.25 * ig_attrs[f"IG_{b}"]
        ig_attrs["UNIFORM_IG"] = uniform_attr
        
        # 4. GRAD_X_INP
        ig_attrs["GRAD_X_INP"] = compute_grad_x_input(
            model=model,
            input_embeds=input_embeds,
            target_class=target_class,
            output_space=output_space,
            attention_mask=attention_mask
        )
        
        # 5. RANDOM control
        rng = np.random.RandomState(seed + idx)
        ig_attrs["RANDOM"] = torch.tensor(rng.randn(seq_len), dtype=torch.float32, device=device)
        
        # Fast batched evaluation of all faithfulness metrics
        evaluations = evaluate_example_batched(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_class=target_class,
            f_orig=f_orig,
            ig_attrs=ig_attrs,
            methods=methods,
            erasure_schemes=erasure_schemes,
            fractions=fractions
        )
        
        example_results = {
            "id": item.get("id", idx),
            "seq_len": seq_len,
            "pred_class": pred_class,
            "f_orig": f_orig,
            "baseline_predictions": base_preds,
            "np_weights": weight_dict,
            "completeness_deltas": base_deltas,
            "evaluations": evaluations
        }
        records.append(example_results)
        
    elapsed = time.time() - start_time
    print(f"Evaluation complete in {elapsed:.2f} seconds ({elapsed / max(1, len(data)):.3f}s / example).")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "dataset_path": dataset_path,
                    "num_samples": len(data),
                    "steps": steps,
                    "temperature": temperature,
                    "seed": seed,
                    "fractions": list(fractions),
                    "erasure_schemes": list(erasure_schemes),
                    "elapsed_seconds": elapsed
                },
                "records": records
            }, f)
        print(f"Results saved to {output_file}")
        
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="c:/Users/91798/OneDrive/Desktop/Research/project/data/sst2_val.json")
    parser.add_argument("--model_path", type=str, default="D:/models/distilbert")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    
    run_evaluation(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        num_samples=args.num_samples,
        steps=args.steps,
        temperature=args.temperature,
        seed=args.seed,
        output_file=args.output_file
    )
