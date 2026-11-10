# Source Code & Execution Guide

## Hardware and Execution Environment
- **CPU**: 8 logical cores, PyTorch 4 threads.
- **RAM**: 16 GB.
- **OS**: Windows (PowerShell).
- **Python**: 3.11.9
- **Key Dependencies**: See `requirements.txt` (`torch==2.11.0`, `transformers==4.57.6`, `datasets==5.0.0`, `scikit-learn==1.9.0`, `scipy==1.13.1`, `pandas==2.2.2`, `numpy==2.2.6`).

---

## Directory Structure
- `model_utils.py`: Loading fine-tuned models, tokenizer wrappers, baseline token IDs, and special token masks.
- `ig.py`: Vectorized Riemann Integrated Gradients, Neutral-Prediction Weighted Integrated Gradients (NP-IG), Gradient x Input, and completeness checks.
- `erasure.py`: Erasure operators (`PAD`, `MASK`, `UNK`, `DELETE`), Comprehensiveness, and Sufficiency.
- `evaluate_dataset.py`: Batched cross-evaluation matrix execution across methods, baselines, and erasure schemes.
- `metrics.py`: AOPC aggregation, random-control normalization, 95% bootstrap confidence intervals, Wilcoxon hypothesis tests, and Kendall tau ranking correlations.
- `ablation.py`: Post-hoc temperature ($\tau$), ensemble size ($K$), and quadrature step ablations.

---

## Execution Commands

### 1. Run All Unit Tests
```powershell
python project/run_tests.py
```

### 2. Run SST-2 Main Cross-Evaluation Matrix
```powershell
python project/src/evaluate_dataset.py --dataset_path project/data/sst2_val.json --num_samples 200 --steps 20 --temperature 0.1 --seed 42 --output_file project/results/sst2_main_seed42.json
```

### 3. Run IMDB Subset Cross-Evaluation Matrix
```powershell
python project/src/evaluate_dataset.py --dataset_path project/data/imdb_500.json --num_samples 100 --steps 20 --temperature 0.1 --seed 42 --output_file project/results/imdb_main_seed42.json
```

### 4. Run Multi-Seed Verification
```powershell
python project/src/evaluate_dataset.py --dataset_path project/data/sst2_val.json --num_samples 200 --steps 20 --temperature 0.1 --seed 123 --output_file project/results/sst2_main_seed123.json
python project/src/evaluate_dataset.py --dataset_path project/data/sst2_val.json --num_samples 200 --steps 20 --temperature 0.1 --seed 999 --output_file project/results/sst2_main_seed999.json
```

### 5. Generate Statistical Summaries and Figures
```powershell
python project/src/generate_figures.py
```
