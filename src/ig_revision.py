"""
ig_revision.py - Vectorized Integrated Gradients, Fixed Grad x Input, and NP-IG functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

def compute_integrated_gradients_m(
    model,
    input_embeds,
    baseline_embeds,
    target_class=1,
    steps=50,
    output_space="prob",
    attention_mask=None
):
    """
    Computes Integrated Gradients salience vector.
    """
    device = input_embeds.device
    seq_len = input_embeds.shape[1]
    
    with torch.no_grad():
        out_inp = model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
        out_base = model(inputs_embeds=baseline_embeds, attention_mask=attention_mask).logits
        if output_space == "prob":
            f_inp = F.softmax(out_inp, dim=-1)[0, target_class].item()
            f_base = F.softmax(out_base, dim=-1)[0, target_class].item()
        else:
            f_inp = out_inp[0, target_class].item()
            f_base = out_base[0, target_class].item()
            
    delta = input_embeds - baseline_embeds
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1)
    
    # Vectorized interpolation: (steps, seq_len, hidden_dim)
    interpolated = baseline_embeds.expand(steps, seq_len, -1) + alphas * delta.expand(steps, seq_len, -1)
    interpolated = interpolated.clone().detach().requires_grad_(True)
    
    batch_mask = attention_mask.expand(steps, seq_len) if attention_mask is not None else None
    logits = model(inputs_embeds=interpolated, attention_mask=batch_mask).logits
    if output_space == "prob":
        scores = F.softmax(logits, dim=-1)[:, target_class]
    else:
        scores = logits[:, target_class]
        
    grads = torch.autograd.grad(scores.sum(), interpolated)[0]
    avg_grads = grads.mean(dim=0, keepdim=True)
    attr_embed = delta * avg_grads
    token_attrs = attr_embed.sum(dim=-1).squeeze(0) # (seq_len,)
    
    total_attr = token_attrs.sum().item()
    completeness_delta = abs(total_attr - (f_inp - f_base))
    
    return token_attrs.detach(), completeness_delta

def compute_grad_x_input_fixed(
    model,
    input_embeds,
    target_class=1,
    use_logits=True,
    attention_mask=None
):
    """
    Fixed Grad x Input:
    Uses unnormalized logit (or logit difference) to prevent softmax saturation,
    and returns signed scalar attribution aligned with target logit.
    """
    input_embeds = input_embeds.clone().detach().requires_grad_(True)
    logits = model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
    if use_logits:
        score = logits[0, target_class]
    else:
        score = F.softmax(logits, dim=-1)[0, target_class]
        
    grads = torch.autograd.grad(score, input_embeds)[0]
    salience = (input_embeds * grads).sum(dim=-1).squeeze(0)
    return salience.detach()

def compute_np_ig_weights(base_preds, tau=0.1, candidate_baselines=("PAD", "MASK", "UNK", "ZERO")):
    """
    Computes baseline weights given distances from 0.5.
    Supports tau=float, tau=float('inf'), tau='argmin'.
    """
    if tau == float("inf") or tau == "inf":
        w = 1.0 / len(candidate_baselines)
        return {b: w for b in candidate_baselines}
        
    distances = [abs(base_preds[b] - 0.5) for b in candidate_baselines]
    
    if tau == "argmin" or tau == 0.0:
        min_idx = int(np.argmin(distances))
        weight_dict = {b: 0.0 for b in candidate_baselines}
        weight_dict[candidate_baselines[min_idx]] = 1.0
        return weight_dict
        
    log_weights = [-d / tau for d in distances]
    max_log = max(log_weights)
    exp_weights = [np.exp(lw - max_log) for lw in log_weights]
    sum_exp = sum(exp_weights)
    weights = [ew / sum_exp for ew in exp_weights]
    return dict(zip(candidate_baselines, weights))

def compute_weight_entropy(weights_dict):
    probs = list(weights_dict.values())
    ent = -sum(p * np.log(p + 1e-12) for p in probs if p > 0)
    return float(ent)

def compute_token_ranking_spearman(attr1, attr2, content_indices):
    """
    Computes Spearman rank correlation between two attribution vectors over content tokens.
    """
    vals1 = [attr1[idx].item() for idx in content_indices]
    vals2 = [attr2[idx].item() for idx in content_indices]
    if len(vals1) < 2 or np.std(vals1) == 0 or np.std(vals2) == 0:
        return 1.0
    res, _ = spearmanr(vals1, vals2)
    return float(res) if not np.isnan(res) else 1.0
