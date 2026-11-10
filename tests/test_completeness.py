import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model_utils import load_model_and_tokenizer, get_word_embeddings, get_baseline_ids
from src.ig import compute_integrated_gradients, compute_np_ig

def test_completeness_convergence():
    model, tokenizer, device = load_model_and_tokenizer("D:/models/distilbert")
    text = "The movie was surprisingly wonderful and deeply moving."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    embed_layer = get_word_embeddings(model)
    input_embeds = embed_layer(input_ids)
    
    # Test PAD baseline completeness across steps
    b_ids = get_baseline_ids(tokenizer, input_ids, "PAD")
    b_embeds = embed_layer(b_ids)
    
    deltas = {}
    for steps in [10, 25, 50]:
        attrs, delta = compute_integrated_gradients(
            model=model,
            input_embeds=input_embeds,
            baseline_embeds=b_embeds,
            target_class=1,
            steps=steps,
            attention_mask=attention_mask
        )
        deltas[steps] = delta
        print(f"Steps {steps}: completeness delta = {delta:.6f}")
        
    # Completeness error must monotonically decrease as steps increase
    assert deltas[50] < deltas[10], f"Convergence failed: 50 steps ({deltas[50]}) >= 10 steps ({deltas[10]})"
    assert deltas[50] < 0.05, f"Completeness delta too high at 50 steps: {deltas[50]}"
    
    # Test NP-IG completeness
    np_res = compute_np_ig(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        target_class=1,
        steps=50,
        temperature=0.1,
        attention_mask=attention_mask
    )
    mean_delta = np_res["mean_completeness_delta"]
    print(f"NP-IG mean completeness delta at 50 steps: {mean_delta:.6f}")
    assert mean_delta < 0.08, f"NP-IG completeness delta too high: {mean_delta}"
    print("Completeness convergence test passed successfully!")

if __name__ == "__main__":
    test_completeness_convergence()
