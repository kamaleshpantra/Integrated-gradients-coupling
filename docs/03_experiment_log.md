# Experiment Log (Lab Notebook)

**Project:** Baseline-Evaluation Coupling in Integrated Gradients for Text  
**Researcher:** [Your Name]  
**Start Date:** 2026-09-15  
**Convention:** Each entry is logged chronologically. Raw data files and seeds are recorded for every run.

---

## Entry 1 — Project Setup & Environment

**Date:** 2026-09-15  
**Objective:** Set up computational environment, download model and data.

### What Was Done
- Created project folder structure at `c:\Users\91798\OneDrive\Desktop\Research\project\`
- Installed dependencies: `torch`, `transformers`, `datasets`, `scipy`, `statsmodels`, `matplotlib`, `pandas`, `numpy`
- Downloaded DistilBERT-base-uncased-finetuned-sst-2-english to `D:/models/distilbert/`
- Downloaded DistilBERT-base-uncased (MLM head) to `D:/models/distilbert_mlm/`
- Fetched SST-2 validation split (872 examples) → saved as `project/data/sst2_val.json`
- Fetched IMDB test split (500 examples) → saved as `project/data/imdb_500.json`

### Verification
- DistilBERT SST-2 accuracy on validation set: **91.3%** (matches published)
- Tokenizer vocabulary size: 30,522

### Files Created
- `project/requirements.txt`
- `project/data/sst2_val.json`
- `project/data/imdb_500.json`

---

## Entry 2 — IG Implementation & Completeness Verification

**Date:** 2026-09-17  
**Objective:** Implement Integrated Gradients and verify the completeness axiom.

### What Was Done
- Implemented `project/src/ig.py`: IG computation via Riemann sum quadrature
- Implemented `project/src/erasure.py`: token erasure operators (PAD, MASK, UNK, DELETE)
- Ran completeness verification on 10 examples at $M=20$ steps

### Results (Completeness Delta)
- PAD baseline: $\delta = 0.194 \pm 0.082$
- MASK baseline: $\delta = 0.218 \pm 0.091$
- UNK baseline: $\delta = 0.250 \pm 0.105$
- ZERO baseline: $\delta = 0.394 \pm 0.145$

### Issue Identified
Completeness error at $M=20$ is too high (0.19–0.39). Will need $M \geq 50$ for the main experiments.

### Files Created
- `project/src/ig.py`
- `project/src/erasure.py`
- `project/results/test_10.json` (pilot results)

---

## Entry 3 — Pilot 4×4 Grid (SST-2, N=150, M=20)

**Date:** 2026-09-20  
**Objective:** Run initial 4×4 grid on SST-2 to validate the experimental pipeline.

### Configuration
- Seed: 42
- N: 150 examples
- M: 20 integration steps
- Baselines: PAD, MASK, UNK, ZERO
- Erasure schemes: PAD, MASK, UNK, DELETE
- Top-k fractions: {0.10, 0.20, 0.30, 0.50}

### Key Observations
1. Diagonal cells (matched pairs) appeared consistently higher than off-diagonal cells.
2. IG-MASK × Erase-MASK showed the strongest diagonal effect.
3. Grad × Input scored near zero — something is wrong (investigated later in Entry 6).

### Files Created
- `project/results/sst2_main_seed42.json` (1,457,366 bytes)

---

## Entry 4 — Multi-Seed Replication (SST-2, Seeds 42/123/999)

**Date:** 2026-09-23  
**Objective:** Replicate the pilot across 3 seeds to enable proper statistical testing.

### Configuration
- Seeds: 42, 123, 999
- N: 150 per seed (450 total)
- M: 20 integration steps

### Results
Diagonal inflation pattern was consistent across all 3 seeds, suggesting a real effect rather than sampling noise.

### Files Created
- `project/results/sst2_main_seed123.json`
- `project/results/sst2_main_seed999.json`

---

## Entry 5 — IMDB Evaluation (N=100)

**Date:** 2026-09-25  
**Objective:** Test whether coupling generalizes to longer sequences.

### Configuration
- Seed: 42
- N: 100 examples (mean length ~94 tokens)
- M: 20 integration steps

### Key Result
**IMDB showed NO significant diagonal inflation** ($p = 0.153$). Diagonal mean = 0.419, off-diagonal mean = 0.422. This is a **null result** that we report transparently.

### Interpretation
On long sequences, erasing a single token produces minimal contextual shift. The coupling effect appears to be length-dependent.

### Files Created
- `project/results/imdb_main_seed42.json`
- `project/results/analysis_imdb_h1.json`

---

## Entry 6 — Grad × Input Bug Investigation

**Date:** 2026-09-27  
**Objective:** Diagnose why Grad × Input scored ~0.006 (essentially random).

### Root Cause Found: Softmax Probability Saturation
DistilBERT predicts with confidence $p > 0.999$ on most SST-2 examples. Computing Grad × Input in probability space:
$$\frac{\partial p_c}{\partial x} = p_c(1 - p_c) \frac{\partial \text{logit}_c}{\partial x}$$
When $p_c = 0.9998$: $p_c(1-p_c) = 0.0002$, causing gradients to vanish to numerical noise.

### Fix Applied
- Changed to unnormalized logit-space gradients in `project/src/ig_revision.py`
- Used absolute-value ranking instead of signed ranking

### Verification
After fix, Grad × Input scores rose but remained low (0.007–0.040 normalized comprehensiveness), consistent with Grad × Input being a weaker method than IG.

### Files Modified
- `project/src/ig_revision.py` (new file with logit-space computation)

---

## Entry 7 — High-Resolution Rerun (M=50, Revision Grid)

**Date:** 2026-10-15  
**Objective:** Rerun the full experiment at higher quadrature resolution with additional erasure schemes.

### Configuration
- Seeds: 42, 123, 999 (N=450 total)
- M: 50 integration steps
- 8 attribution methods × 7 erasure schemes = 56 conditions per example
- Total rows: 8 × 7 × 450 = 25,200

### New Additions
- **ZERO_EMB erasure**: Replaces token embedding with zeros (true matched pair for IG-ZERO)
- **RANDOM_TOKEN erasure**: Random vocabulary ID, averaged over 5 sub-seeds
- **MLM_INFILL erasure**: DistilBERT masked LM top-1 prediction

### Completeness Improvement
| Baseline | $M=20$ $\delta$ | $M=50$ $\delta$ | Improvement |
|---|:---:|:---:|:---:|
| PAD | 0.1937 | 0.0284 | 6.8× |
| MASK | 0.2180 | 0.0232 | 9.4× |
| UNK | 0.2495 | 0.0709 | 3.5× |
| ZERO | 0.3941 | 0.1089 | 3.6× |

### Files Created
- `project/results/sst2_revision_seed42.csv` (8,400 rows)
- `project/results/sst2_revision_seed123.csv` (8,400 rows)
- `project/results/sst2_revision_seed999.csv` (8,400 rows)
- `project/results/sst2_revision_all450.csv` (25,200 rows — master file)

---

## Entry 8 — Steps Stability Analysis (M=20/50/100)

**Date:** 2026-10-17  
**Objective:** Verify that $M=50$ is sufficient by comparing token rankings to $M=100$.

### Configuration
- N: 50 examples (subset)
- M: 20, 50, 100
- Baselines: PAD, MASK, UNK, ZERO

### Results: Spearman Rank Correlation
| Comparison | PAD | MASK | UNK | ZERO |
|---|:---:|:---:|:---:|:---:|
| $M=50$ vs $M=100$ | 0.9984 | 0.9984 | 0.9980 | 0.9708 |
| $M=20$ vs $M=100$ | 0.9876 | 0.9871 | 0.9842 | 0.9312 |

**Conclusion:** Token rankings are essentially invariant above $M=50$. ZERO baseline shows more path sensitivity but still converges well.

### Files Created
- `project/results/steps_stability.csv`

---

## Entry 9 — NP-IG Temperature Ablation

**Date:** 2026-10-20  
**Objective:** Sweep NP-IG temperature parameter and evaluate whether neutrality weighting outperforms uniform weighting.

### Configuration
- $\tau \in \{0.01, 0.03, 0.1, 0.3, 1.0, \infty\}$
- Plus: Argmin-Neutrality (single best baseline) and Random-Weights control
- N: 450, M: 50

### Key Finding: NP-IG Collapses onto MASK

| $\tau$ | Weight on MASK | Entropy (nats) | Worst-Case Def 2 |
|:---:|:---:|:---:|:---:|
| 0.01 | 1.000 | 0.000 | 0.2939 |
| 0.03 | 0.998 | 0.009 | 0.2938 |
| 0.10 | 0.900 | 0.391 | 0.2930 |
| 0.30 | 0.526 | 1.195 | 0.2849 |
| 1.00 | 0.325 | 1.370 | 0.2801 |
| $\infty$ (Uniform) | 0.250 | 1.386 | 0.2783 |

**Conclusion:** NP-IG at $\tau=0.1$ is essentially IG-MASK with 10% smoothing. Uniform-IG provides the best worst-case robustness. **NP-IG does NOT outperform Uniform-IG.** This is an honest negative result.

### Files Created
- `project/results/np_ig_ablation.csv` (651,873 bytes)

---

## Entry 10 — Linear Mixed-Effects Model & Statistical Analysis

**Date:** 2026-10-25  
**Objective:** Fit LMM and complete all statistical analyses.

### Model Specification
```
norm_comp_aopc ~ C(baseline) + C(scheme) + is_matched + (1 | unique_id)
```
where `unique_id = seed + "_" + example_id` to avoid group overlap across seeds.

### Results
- **Matched effect**: $\beta = +0.0342$, $z = 5.579$, $p = 2.42 \times 10^{-8}$
- **95% CI**: $[+0.0222, +0.0462]$
- **ANOVA variance**: Baseline = 1.22%, Scheme = 0.14%, Matched = 0.17%, Residual = 98.47%

### Bug Fixed During Analysis
- `unique_id` was initially just `example_id` (0–149), causing shape mismatch when fitting across 450 examples from 3 seeds. Fixed by prepending seed.

### Files Created
- `project/results/revision_analysis_results.json`

---

## Entry 11 — Paper Rewrite & Final Packaging

**Date:** 2026-10-29  
**Objective:** Generate revised paper with all numbers from logged results.

### What Was Done
1. Ran `generate_revision_figures.py` → 4 figures in `project/figures/`
2. Ran `write_revision_paper.py` → `main.tex`, `main.bib`, `CHANGELOG.md`, `09_claims_evidence.md`
3. Packaged all files into `overleaf_paper.zip`

### Quality Checks
- Every number in `main.tex` is generated by template substitution from `revision_analysis_results.json`
- No hand-edited statistics
- All references verified (see `AUDIT.md` §5)
- Figures generated from raw CSV data

### Files Created
- `project/paper/main.tex`
- `project/paper/main.bib`
- `project/CHANGELOG.md`
- `project/09_claims_evidence.md`
- `project/overleaf_paper.zip`
- `overleaf_paper.zip` (root copy for easy access)
