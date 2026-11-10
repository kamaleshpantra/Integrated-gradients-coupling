# Benchmark Datasets

This directory contains the evaluation datasets used in the cross-evaluation study.

## Datasets Included

### 1. `sst2_val.json` (Stanford Sentiment Treebank)
- **Task:** Binary sentiment classification (positive / negative).
- **Split:** Validation split from GLUE (`sst2`).
- **Sample Size:** 450 validation examples evaluated across 3 fixed random seeds (seeds 42, 123, 999; 150 examples per seed).
- **Target Model:** `distilbert-base-uncased-finetuned-sst-2-english` (Validation accuracy: 91.3%).
- **Schema:**
  ```json
  [
    {
      "sentence": "it 's a charming and often affecting journey . ",
      "label": 1
    }
  ]
  ```

### 2. `imdb_500.json` (Large Movie Review Dataset)
- **Task:** Long-document sentiment classification.
- **Split:** Test subset of 100 randomly sampled reviews (mean length: 94 content tokens).
- **Target Model:** `textattack/distilbert-base-uncased-imdb`.
- **Purpose:** Testing length-dependency hypothesis ($H_2$) to verify that baseline coupling attenuates on longer texts.

## Dataset Download / Extraction
To re-fetch or verify datasets from Hugging Face Hub:
```bash
python src/download_data.py
```
