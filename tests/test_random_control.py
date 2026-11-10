import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from src.model_utils import load_model_and_tokenizer, get_word_embeddings, get_baseline_ids
from src.ig import compute_integrated_gradients
from src.erasure import evaluate_faithfulness

def test_ig_beats_random_control():
    model, tokenizer, device = load_model_and_tokenizer("D:/models/distilbert")
    text = "A truly terrible and awful performance that completely ruined the evening."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    embed_layer = get_word_embeddings(model)
    input_embeds = embed_layer(input_ids)
    
    # Predict target class
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_class = out.argmax(dim=-1).item() # class 0 (negative)
        
    b_ids = get_baseline_ids(tokenizer, input_ids, "MASK")
    b_embeds = embed_layer(b_ids)
    
    # Real IG attribution
    ig_attrs, _ = compute_integrated_gradients(
        model=model,
        input_embeds=input_embeds,
        baseline_embeds=b_embeds,
        target_class=pred_class,
        steps=25,
        attention_mask=attention_mask
    )
    
    ig_eval = evaluate_faithfulness(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attributions=ig_attrs,
        target_class=pred_class,
        fraction=0.3,
        erasure_scheme="MASK",
        attention_mask=attention_mask
    )
    
    # Random attribution control (averaged over 10 seeds)
    rand_comps = []
    for seed in range(10):
        rng = np.random.RandomState(seed)
        rand_attrs = torch.tensor(rng.randn(input_ids.shape[1]), dtype=torch.float32, device=device)
        rand_eval = evaluate_faithfulness(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attributions=rand_attrs,
            target_class=pred_class,
            fraction=0.3,
            erasure_scheme="MASK",
            attention_mask=attention_mask
        )
        rand_comps.append(rand_eval["comprehensiveness"])
        
    mean_rand_comp = float(np.mean(rand_comps))
    ig_comp = ig_eval["comprehensiveness"]
    print(f"IG Comprehensiveness: {ig_comp:.4f}, Mean Random Comprehensiveness: {mean_rand_comp:.4f}")
    assert ig_comp > mean_rand_comp, f"IG failed to beat random control! IG={ig_comp}, Rand={mean_rand_comp}"
    print("Random control test passed successfully!")

if __name__ == "__main__":
    test_ig_beats_random_control()
