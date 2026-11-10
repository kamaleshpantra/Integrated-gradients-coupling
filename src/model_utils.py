import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_name_or_path="D:/models/distilbert", device=None):
    """
    Loads model and tokenizer, prioritizing local directory if present.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(model_name_or_path) and os.path.exists(os.path.join(model_name_or_path, "model.safetensors")):
        load_path = model_name_or_path
    else:
        load_path = "distilbert-base-uncased-finetuned-sst-2-english"
        
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def get_word_embeddings(model):
    """
    Extracts the word embedding layer from transformer model.
    """
    if hasattr(model, "distilbert"):
        return model.distilbert.embeddings.word_embeddings
    elif hasattr(model, "bert"):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, "roberta"):
        return model.roberta.embeddings.word_embeddings
    else:
        return model.get_input_embeddings()

def get_special_token_ids(tokenizer):
    """
    Returns dictionary of special token IDs.
    """
    return {
        "cls_id": tokenizer.cls_token_id,
        "sep_id": tokenizer.sep_token_id,
        "pad_id": tokenizer.pad_token_id,
        "mask_id": tokenizer.mask_token_id,
        "unk_id": tokenizer.unk_token_id
    }

def get_baseline_ids(tokenizer, input_ids, baseline_type):
    """
    Constructs token ID sequence for a requested baseline type:
    'PAD', 'MASK', 'UNK', preserving CLS and SEP tokens.
    """
    specials = get_special_token_ids(tokenizer)
    b_ids = input_ids.clone()
    # Mask non-special tokens
    non_special_mask = (input_ids != specials["cls_id"]) & (input_ids != specials["sep_id"]) & (input_ids != specials["pad_id"])
    
    if baseline_type.upper() == "PAD":
        fill_id = specials["pad_id"]
    elif baseline_type.upper() == "MASK":
        fill_id = specials["mask_id"]
    elif baseline_type.upper() == "UNK":
        fill_id = specials["unk_id"]
    else:
        raise ValueError(f"Unknown discrete baseline type: {baseline_type}")
        
    b_ids[non_special_mask] = fill_id
    return b_ids
