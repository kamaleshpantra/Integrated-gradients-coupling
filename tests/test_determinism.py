import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

def generate_random_attributions(seq_len, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        attrs = rng.randn(seq_len)
    else:
        attrs = np.random.randn(seq_len)
    return torch.tensor(attrs, dtype=torch.float32)

def test_determinism_under_fixed_seeds():
    seed = 42
    attr1 = generate_random_attributions(50, seed=seed)
    attr2 = generate_random_attributions(50, seed=seed)
    assert torch.allclose(attr1, attr2), "Attributions under same seed did not match!"

    # Different seeds must produce different attributions
    attr3 = generate_random_attributions(50, seed=123)
    assert not torch.allclose(attr1, attr3), "Different seeds produced identical attributions!"
    print("Determinism under fixed seeds verified successfully!")

if __name__ == "__main__":
    test_determinism_under_fixed_seeds()
