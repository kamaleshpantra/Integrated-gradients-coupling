import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import BertTokenizer
from src.erasure import get_content_token_indices, select_top_k_indices, apply_erasure

def test_special_tokens_excluded_from_content():
    # Use dummy tokenizer or standard vocab
    # Create synthetic input_ids: [CLS]=101, word=2000, word=2001, [SEP]=102, [PAD]=0
    input_ids = torch.tensor([[101, 2000, 2001, 2002, 102, 0, 0]])
    
    class DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0
        mask_token_id = 103
        unk_token_id = 100
        
    tokenizer = DummyTokenizer()
    content_indices = get_content_token_indices(tokenizer, input_ids)
    
    # Only indices 1, 2, 3 should be content
    assert content_indices == [1, 2, 3], f"Expected [1, 2, 3], got {content_indices}"
    assert 0 not in content_indices, "CLS token included in content!"
    assert 4 not in content_indices, "SEP token included in content!"
    assert 5 not in content_indices, "PAD token included in content!"
    assert 6 not in content_indices, "PAD token included in content!"

def test_erasure_operators_shape_and_tokens():
    input_ids = torch.tensor([[101, 2000, 2001, 2002, 102]])
    class DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0
        mask_token_id = 103
        unk_token_id = 100
    tokenizer = DummyTokenizer()
    
    erase_indices = {2} # erase index 2 (token 2001)
    
    # 1. PAD
    pad_ids, pad_mask = apply_erasure(input_ids, erase_indices, "PAD", tokenizer)
    assert pad_ids[0, 2].item() == 0
    assert pad_mask[0, 2].item() == 0
    assert pad_ids.shape == input_ids.shape
    
    # 2. MASK
    mask_ids, mask_mask = apply_erasure(input_ids, erase_indices, "MASK", tokenizer)
    assert mask_ids[0, 2].item() == 103
    assert mask_ids.shape == input_ids.shape
    
    # 3. UNK
    unk_ids, unk_mask = apply_erasure(input_ids, erase_indices, "UNK", tokenizer)
    assert unk_ids[0, 2].item() == 100
    assert unk_ids.shape == input_ids.shape
    
    # 4. DELETE
    del_ids, del_mask = apply_erasure(input_ids, erase_indices, "DELETE", tokenizer)
    assert del_ids.shape == (1, 4)
    assert del_ids.squeeze(0).tolist() == [101, 2000, 2002, 102]
    assert del_mask.shape == (1, 4)

if __name__ == "__main__":
    test_special_tokens_excluded_from_content()
    test_erasure_operators_shape_and_tokens()
    print("All special token and erasure operator tests passed successfully!")
