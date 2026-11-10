import torch
import torch.nn.functional as F
import numpy as np
from src.model_utils import get_word_embeddings, get_baseline_ids, get_special_token_ids

def compute_integrated_gradients(model, input_embeds, baseline_embeds, target_class=1, steps=20, output_space="prob", attention_mask=None):
    """
    Computes Integrated Gradients for a sequence using batched Riemann approximation.
    
    Args:
        model: HuggingFace Sequence Classification Model
        input_embeds: Tensor of shape (1, seq_len, hidden_dim)
        baseline_embeds: Tensor of shape (1, seq_len, hidden_dim)
        target_class: Target class index to explain (e.g. 1 for positive sentiment)
        steps: Number of Riemann interpolation steps (M)
        output_space: 'prob' (softmax probability) or 'logit' (raw pre-softmax logit)
        attention_mask: Tensor of shape (1, seq_len) or None
        
    Returns:
        attributions: Tensor of shape (seq_len,) containing token-level attribution scores
        completeness_delta: Scalar float measuring |sum(attributions) - (f(x) - f(x'))|
    """
    device = input_embeds.device
    seq_len = input_embeds.shape[1]
    
    # 1. Baseline prediction f(x')
    with torch.no_grad():
        out_base = model(inputs_embeds=baseline_embeds, attention_mask=attention_mask).logits
        if output_space == "prob":
            f_base = F.softmax(out_base, dim=-1)[0, target_class].item()
        else:
            f_base = out_base[0, target_class].item()
            
    # 2. Input prediction f(x)
    with torch.no_grad():
        out_inp = model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
        if output_space == "prob":
            f_inp = F.softmax(out_inp, dim=-1)[0, target_class].item()
        else:
            f_inp = out_inp[0, target_class].item()
            
    # 3. Interpolation path alphas
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device) # shape (steps,)
    
    # Batch interpolated embeddings: shape (steps, seq_len, hidden_dim)
    delta = input_embeds - baseline_embeds
    interpolated = baseline_embeds + alphas.view(steps, 1, 1) * delta
    interpolated = interpolated.detach().requires_grad_(True)
    
    # Replicate attention mask for the batch
    if attention_mask is not None:
        batch_mask = attention_mask.expand(steps, seq_len)
    else:
        batch_mask = None
        
    # Forward pass
    logits = model(inputs_embeds=interpolated, attention_mask=batch_mask).logits
    if output_space == "prob":
        scores = F.softmax(logits, dim=-1)[:, target_class]
    else:
        scores = logits[:, target_class]
        
    # Backward pass to get gradients
    grads = torch.autograd.grad(scores.sum(), interpolated)[0] # (steps, seq_len, hidden_dim)
    
    # Riemann sum average
    avg_grads = grads.mean(dim=0, keepdim=True) # (1, seq_len, hidden_dim)
    
    # Attribute = (x - x') * avg_grads, summed over embedding dim
    attr_embed = delta * avg_grads # (1, seq_len, hidden_dim)
    token_attrs = attr_embed.sum(dim=-1).squeeze(0) # (seq_len,)
    
    # Completeness check
    total_attr = token_attrs.sum().item()
    completeness_delta = abs(total_attr - (f_inp - f_base))
    
    return token_attrs.detach(), completeness_delta

def compute_grad_x_input(model, input_embeds, target_class=1, output_space="prob", attention_mask=None):
    """
    Computes Gradient x Input salience.
    """
    input_embeds = input_embeds.clone().detach().requires_grad_(True)
    logits = model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
    if output_space == "prob":
        score = F.softmax(logits, dim=-1)[0, target_class]
    else:
        score = logits[0, target_class]
        
    grads = torch.autograd.grad(score, input_embeds)[0]
    salience = (input_embeds * grads).sum(dim=-1).squeeze(0)
    return salience.detach()

def compute_np_ig(model, tokenizer, input_ids, target_class=1, steps=20, temperature=0.1, candidate_baselines=("PAD", "MASK", "UNK", "ZERO"), output_space="prob", attention_mask=None):
    """
    Computes Neutral-Prediction Weighted Integrated Gradients (NP-IG).
    
    Ensemble weights are inversely proportional to distance from neutrality (0.5 target prob):
    w_k = exp(-|f(b_k) - 0.5| / tau) / sum_j exp(-|f(b_j) - 0.5| / tau)
    """
    device = input_ids.device
    embed_layer = get_word_embeddings(model)
    input_embeds = embed_layer(input_ids)
    
    baseline_attrs = []
    baseline_preds = []
    baseline_deltas = []
    
    for b_type in candidate_baselines:
        if b_type.upper() == "ZERO":
            b_embeds = torch.zeros_like(input_embeds)
        else:
            b_ids = get_baseline_ids(tokenizer, input_ids, b_type)
            b_embeds = embed_layer(b_ids)
            
        with torch.no_grad():
            out_b = model(inputs_embeds=b_embeds, attention_mask=attention_mask).logits
            prob_b = F.softmax(out_b, dim=-1)[0, target_class].item()
            baseline_preds.append(prob_b)
            
        attrs, delta = compute_integrated_gradients(
            model=model,
            input_embeds=input_embeds,
            baseline_embeds=b_embeds,
            target_class=target_class,
            steps=steps,
            output_space=output_space,
            attention_mask=attention_mask
        )
        baseline_attrs.append(attrs)
        baseline_deltas.append(delta)
        
    # Compute neutrality weights
    distances = [abs(p - 0.5) for p in baseline_preds]
    # Softmax over negative distances with temperature
    log_weights = [-d / temperature for d in distances]
    max_log = max(log_weights)
    exp_weights = [np.exp(lw - max_log) for lw in log_weights]
    sum_exp = sum(exp_weights)
    weights = [ew / sum_exp for ew in exp_weights]
    
    # Weighted average attribution
    np_attrs = torch.zeros_like(baseline_attrs[0])
    for w, attr in zip(weights, baseline_attrs):
        np_attrs += w * attr
        
    expected_base_pred = sum(w * p for w, p in zip(weights, baseline_preds))
    
    return {
        "attributions": np_attrs,
        "weights": dict(zip(candidate_baselines, weights)),
        "baseline_predictions": dict(zip(candidate_baselines, baseline_preds)),
        "expected_baseline_prediction": expected_base_pred,
        "mean_completeness_delta": float(np.mean(baseline_deltas))
    }
