import torch
import torch.nn.functional as F
from src.model_utils import get_special_token_ids

def get_content_token_indices(tokenizer, input_ids):
    """
    Identifies indices of content tokens (excluding [CLS], [SEP], and trailing [PAD]).
    """
    specials = get_special_token_ids(tokenizer)
    seq = input_ids.squeeze(0)
    is_content = (seq != specials["cls_id"]) & (seq != specials["sep_id"]) & (seq != specials["pad_id"])
    indices = torch.nonzero(is_content, as_tuple=False).squeeze(-1).tolist()
    return indices

def select_top_k_indices(attributions, candidate_indices, fraction=0.2):
    """
    Selects top-k indices based on attribution salience among candidate indices.
    """
    if len(candidate_indices) == 0:
        return []
    num_to_select = max(1, int(round(len(candidate_indices) * fraction)))
    num_to_select = min(num_to_select, len(candidate_indices))
    
    # Sort candidate indices by attribution descending
    sorted_candidates = sorted(candidate_indices, key=lambda idx: attributions[idx].item(), reverse=True)
    return set(sorted_candidates[:num_to_select])

def apply_erasure(input_ids, erase_indices, erasure_scheme, tokenizer, attention_mask=None):
    """
    Applies one of 4 erasure operators: 'PAD', 'MASK', 'UNK', 'Delete'.
    
    Returns:
        erased_input_ids: Tensor of shape (1, new_len)
        erased_attention_mask: Tensor of shape (1, new_len)
    """
    device = input_ids.device
    specials = get_special_token_ids(tokenizer)
    seq = input_ids.squeeze(0).clone()
    scheme = erasure_scheme.upper()
    
    if scheme == "DELETE":
        keep_mask = [i not in erase_indices for i in range(len(seq))]
        # Ensure at least CLS and SEP remain
        if sum(keep_mask) == 0:
            keep_mask = [True] * len(seq)
        new_seq = seq[torch.tensor(keep_mask, dtype=torch.bool, device=device)]
        erased_input_ids = new_seq.unsqueeze(0)
        erased_mask = torch.ones_like(erased_input_ids)
        return erased_input_ids, erased_mask
    else:
        if scheme == "PAD":
            sub_id = specials["pad_id"]
        elif scheme == "MASK":
            sub_id = specials["mask_id"]
        elif scheme == "UNK":
            sub_id = specials["unk_id"]
        else:
            raise ValueError(f"Unknown erasure scheme: {erasure_scheme}")
            
        for idx in erase_indices:
            seq[idx] = sub_id
            
        erased_input_ids = seq.unsqueeze(0)
        if attention_mask is not None:
            erased_mask = attention_mask.clone()
            if scheme == "PAD":
                # For PAD erasure, standard practice sets attention mask to 0 at padded positions
                for idx in erase_indices:
                    erased_mask[0, idx] = 0
        else:
            erased_mask = torch.ones_like(erased_input_ids)
            if scheme == "PAD":
                for idx in erase_indices:
                    erased_mask[0, idx] = 0
        return erased_input_ids, erased_mask

def evaluate_faithfulness(model, tokenizer, input_ids, attributions, target_class=1, fraction=0.2, erasure_scheme="MASK", attention_mask=None):
    """
    Calculates Comprehensiveness and Sufficiency for a given attribution vector and erasure scheme.
    
    Comprehensiveness = f(x) - f(x \ top_k)  [Higher is more faithful]
    Sufficiency = f(x) - f(x_top_k)          [Lower is more faithful]
    """
    device = input_ids.device
    with torch.no_grad():
        orig_out = model(input_ids=input_ids, attention_mask=attention_mask).logits
        f_orig = F.softmax(orig_out, dim=-1)[0, target_class].item()
        
    candidate_indices = get_content_token_indices(tokenizer, input_ids)
    if len(candidate_indices) == 0:
        return {"comprehensiveness": 0.0, "sufficiency": 0.0, "f_orig": f_orig}
        
    top_indices = select_top_k_indices(attributions, candidate_indices, fraction=fraction)
    
    # 1. Comprehensiveness: erase top_indices
    comp_ids, comp_mask = apply_erasure(input_ids, top_indices, erasure_scheme, tokenizer, attention_mask)
    with torch.no_grad():
        comp_out = model(input_ids=comp_ids, attention_mask=comp_mask).logits
        f_comp = F.softmax(comp_out, dim=-1)[0, target_class].item()
    comprehensiveness = f_orig - f_comp
    
    # 2. Sufficiency: erase all content tokens EXCEPT top_indices
    other_indices = set(candidate_indices) - top_indices
    suff_ids, suff_mask = apply_erasure(input_ids, other_indices, erasure_scheme, tokenizer, attention_mask)
    with torch.no_grad():
        suff_out = model(input_ids=suff_ids, attention_mask=suff_mask).logits
        f_suff = F.softmax(suff_out, dim=-1)[0, target_class].item()
    sufficiency = f_orig - f_suff
    
    return {
        "comprehensiveness": comprehensiveness,
        "sufficiency": sufficiency,
        "f_orig": f_orig,
        "f_erased_comp": f_comp,
        "f_erased_suff": f_suff,
        "num_erased": len(top_indices),
        "total_content": len(candidate_indices)
    }
