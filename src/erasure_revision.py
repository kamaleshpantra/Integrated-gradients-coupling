"""
erasure_revision.py - Comprehensive erasure operators for revision study.

Supported erasure schemes:
1. PAD: token -> [PAD], attention_mask -> 0
2. MASK: token -> [MASK], attention_mask -> 1
3. UNK: token -> [UNK], attention_mask -> 1
4. ZERO_EMB: token embedding vector set to 0.0, attention_mask -> 1 (true matched pair for ZERO baseline)
5. DELETE: token physically removed from sequence and attention_mask
6. RANDOM_TOKEN: token replaced with randomly chosen vocabulary token (average of 5 draws)
7. MLM_INFILL: token replaced with top-1 prediction from DistilBERT masked language model
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.model_utils import get_special_token_ids

_MLM_MODEL = None
_MLM_TOKENIZER = None

def get_mlm_model_and_tokenizer(model_path="D:/models/distilbert_mlm", device="cpu"):
    global _MLM_MODEL, _MLM_TOKENIZER
    if _MLM_MODEL is None:
        _MLM_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _MLM_MODEL = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
        _MLM_MODEL.eval()
    return _MLM_MODEL, _MLM_TOKENIZER

def get_content_token_indices(tokenizer, input_ids):
    specials = get_special_token_ids(tokenizer)
    seq = input_ids.squeeze(0)
    is_content = (seq != specials["cls_id"]) & (seq != specials["sep_id"]) & (seq != specials["pad_id"])
    indices = torch.nonzero(is_content, as_tuple=False).squeeze(-1).tolist()
    if isinstance(indices, int):
        indices = [indices]
    return indices

def select_top_k_indices(attributions, candidate_indices, fraction=0.2):
    if len(candidate_indices) == 0:
        return []
    num_to_select = max(1, int(round(len(candidate_indices) * fraction)))
    num_to_select = min(num_to_select, len(candidate_indices))
    # Sort candidate indices by attribution descending
    sorted_candidates = sorted(candidate_indices, key=lambda idx: attributions[idx].item(), reverse=True)
    return set(sorted_candidates[:num_to_select])

def apply_erasure_to_input(
    input_ids,
    erase_indices,
    scheme,
    tokenizer,
    embed_layer,
    attention_mask=None,
    mlm_model=None,
    mlm_tokenizer=None,
    rng=None
):
    """
    Applies erasure and returns:
    (embeds_tensor, mask_tensor)
    Shape: (1, seq_len, hidden_dim), (1, seq_len)
    """
    device = input_ids.device
    specials = get_special_token_ids(tokenizer)
    scheme = scheme.upper()
    seq = input_ids.squeeze(0).clone()
    seq_len = seq.shape[0]

    if scheme == "DELETE":
        keep_mask = [i not in erase_indices for i in range(seq_len)]
        if sum(keep_mask) == 0:
            keep_mask = [True] * seq_len
        keep_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=device)
        new_seq = seq[keep_tensor].unsqueeze(0)
        new_mask = torch.ones_like(new_seq)
        new_embeds = embed_layer(new_seq)
        return new_embeds, new_mask

    elif scheme == "ZERO_EMB":
        # Keep sequence IDs unchanged, embed them, then zero out erase_indices
        base_embeds = embed_layer(input_ids.clone()) # (1, seq_len, hidden_dim)
        for idx in erase_indices:
            base_embeds[0, idx, :] = 0.0
        mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        return base_embeds, mask

    elif scheme == "PAD":
        for idx in erase_indices:
            seq[idx] = specials["pad_id"]
        mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        for idx in erase_indices:
            mask[0, idx] = 0
        new_embeds = embed_layer(seq.unsqueeze(0))
        return new_embeds, mask

    elif scheme == "MASK":
        for idx in erase_indices:
            seq[idx] = specials["mask_id"]
        mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        new_embeds = embed_layer(seq.unsqueeze(0))
        return new_embeds, mask

    elif scheme == "UNK":
        for idx in erase_indices:
            seq[idx] = specials["unk_id"]
        mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        new_embeds = embed_layer(seq.unsqueeze(0))
        return new_embeds, mask

    elif scheme == "RANDOM_TOKEN":
        # Average over 5 random draws
        if rng is None:
            rng = np.random.RandomState(42)
        vocab_size = tokenizer.vocab_size
        special_ids = set(specials.values())
        
        # Draw 5 variations and return average embeddings
        embed_draws = []
        for _ in range(5):
            seq_draw = seq.clone()
            for idx in erase_indices:
                rand_id = int(rng.randint(0, vocab_size))
                while rand_id in special_ids:
                    rand_id = int(rng.randint(0, vocab_size))
                seq_draw[idx] = rand_id
            embed_draws.append(embed_layer(seq_draw.unsqueeze(0)))
        avg_embeds = torch.stack(embed_draws, dim=0).mean(dim=0)
        mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        return avg_embeds, mask

    elif scheme == "MLM_INFILL":
        if mlm_model is None or mlm_tokenizer is None:
            mlm_model, mlm_tokenizer = get_mlm_model_and_tokenizer(device=device)
        
        # Place [MASK] at erase_indices, run MLM model, pick top-1 infill
        seq_masked = seq.clone()
        for idx in erase_indices:
            seq_masked[idx] = specials["mask_id"]
        mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        
        with torch.no_grad():
            mlm_out = mlm_model(input_ids=seq_masked.unsqueeze(0), attention_mask=mask).logits
            for idx in erase_indices:
                top_token_id = mlm_out[0, idx].argmax().item()
                seq_masked[idx] = top_token_id
        new_embeds = embed_layer(seq_masked.unsqueeze(0))
        return new_embeds, mask

    else:
        raise ValueError(f"Unknown erasure scheme: {scheme}")
