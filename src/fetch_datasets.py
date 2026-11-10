import os
import sys
import json
import random

os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TMPDIR"] = "D:/hf_cache/tmp"
os.environ["TEMP"] = "D:/hf_cache/tmp"
os.environ["TMP"] = "D:/hf_cache/tmp"

from datasets import load_dataset

out_dir = "c:/Users/91798/OneDrive/Desktop/Research/project/data"
os.makedirs(out_dir, exist_ok=True)

sst2_path = os.path.join(out_dir, "sst2_val.json")
imdb_path = os.path.join(out_dir, "imdb_500.json")

print("Loading SST-2 validation dataset...")
sst2 = load_dataset("glue", "sst2", split="validation", cache_dir="D:/hf_cache/datasets")
sst2_data = []
for idx, item in enumerate(sst2):
    sst2_data.append({
        "id": idx,
        "sentence": item["sentence"],
        "label": item["label"]
    })

with open(sst2_path, "w", encoding="utf-8") as f:
    json.dump(sst2_data, f, indent=2)
print(f"Saved SST-2 validation ({len(sst2_data)} samples) to {sst2_path}")

print("Loading IMDB test dataset...")
imdb = load_dataset("imdb", split="test", cache_dir="D:/hf_cache/datasets")

# Fixed random seed 42 sampling of 500 examples
random.seed(42)
sampled_indices = random.sample(range(len(imdb)), 500)
imdb_500_data = []
for new_id, idx in enumerate(sampled_indices):
    item = imdb[idx]
    imdb_500_data.append({
        "id": new_id,
        "original_id": idx,
        "text": item["text"],
        "label": item["label"]
    })

with open(imdb_path, "w", encoding="utf-8") as f:
    json.dump(imdb_500_data, f, indent=2)
print(f"Saved IMDB 500 test subset ({len(imdb_500_data)} samples) to {imdb_path}")
print("Dataset preparation complete!")
