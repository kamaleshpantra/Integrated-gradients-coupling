import os
import sys

# Route all caches and temp files to D:\hf_cache
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TMPDIR"] = "D:/hf_cache/tmp"
os.environ["TEMP"] = "D:/hf_cache/tmp"
os.environ["TMP"] = "D:/hf_cache/tmp"

import tempfile
tempfile.tempdir = "D:/hf_cache/tmp"

print("Downloading DistilBERT tokenizer and model to D:/hf_cache...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("DistilBERT model loaded successfully:", model.config.name_or_path)

print("Downloading SST-2 dataset...")
from datasets import load_dataset
sst2 = load_dataset("glue", "sst2", split="validation")
print("SST-2 validation split loaded successfully. Size:", len(sst2))

print("Downloading IMDB dataset (for 500-sample test subset)...")
imdb = load_dataset("imdb", split="test")
print("IMDB test split loaded successfully. Size:", len(imdb))

print("All models and datasets cached successfully in D:/hf_cache!")
