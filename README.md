# Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Reproducibility: ACL/NeurIPS Checklist](https://img.shields.io/badge/reproducibility-ACL%2FNeurIPS-green.svg)](docs/07_reproducibility_checklist.md)
[![Status: Camera--Ready Revision](https://img.shields.io/badge/status-ready%20for%20defense-success.svg)](paper/main.tex)

This repository contains the complete experimental code, raw per-example benchmark logs, statistical modeling artifacts, publication figures, and LaTeX manuscript for the empirical study:

> **"Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines"**

---

## 📌 Executive Summary of Findings

1. **Matched-Pair Coupling is Real but Modest ($H_1$ Supported):**
   Evaluating Integrated Gradients with an erasure operator identical to the reference baseline (e.g. `IG-MASK` under `MASK` erasure) produces a statistically significant positive boost on comprehensiveness ($\beta_{\text{matched}} = +\mathbf{0.0342}$, $95\%$ CI: $[+0.0222, +0.0462]$, $p = \mathbf{2.42 \times 10^{-8}}$ via Linear Mixed Model across $N=450$ examples from 3 seeds at $M=50$ steps).
2. **Baseline Selection Dominates the Evaluation Artifact:**
   An ANOVA variance decomposition shows that baseline main effects account for **$1.22\%$** of metric variance—over **$7\times$ more variance** than the matched coupling artifact (**$0.17\%$**). Baseline choice dominates metric outcomes.
3. **Length-Dependent Interaction:**
   Coupling is nearly double on short sentences ($\le 10$ tokens: $\beta = +\mathbf{0.0586}$, $p=0.0046$) compared to longer texts ($>20$ tokens: $\beta = +\mathbf{0.0330}$), and is **completely null on IMDB** ($N=100$, mean length 94: $p=0.153$), where single-token replacement minimally shifts global context.
4. **Uniform-IG Wins; NP-IG Collapses onto MASK:**
   While Neutrality-Prediction IG (NP-IG) was designed to weight baselines dynamically, at low temperature ($\tau=0.1$) it collapses onto `MASK` ($90.0\%$ weight, entropy $0.391$ nats). Simple, unweighted **Uniform-IG** achieves the highest worst-case robustness across all 7 evaluated erasure schemes ($\text{Def 2} = \mathbf{0.3316}$, $\text{Def 1} = \mathbf{0.1285}$), including two held-out schemes (`RANDOM_TOKEN` and `MLM_INFILL`).

---

## 🧭 Research Documentation & Methodology (`docs/`)

To support rigorous open science, reproducibility audits, and independent verification, the complete research lifecycle is documented in [`project/docs/`](docs/README.md):

| Stage | Document | Key Purpose |
|---|---|---|
| **1. Inception** | [`docs/01_research_proposal.md`](docs/01_research_proposal.md) | Initial research proposal, RQs, pre-registered hypotheses |
| **2. Background** | [`docs/02_literature_review.md`](docs/02_literature_review.md) | Survey of IG, baselines, and ERASER faithfulness |
| **3. Execution** | [`docs/03_experiment_log.md`](docs/03_experiment_log.md) | Chronological lab notebook with 11 dated milestones |
| **4. Technical** | [`docs/04_methodology_details.md`](docs/04_methodology_details.md) | Quadrature, Riemann sums, masking, logit-space gradients |
| **5. Data Schema** | [`docs/06_data_dictionary.md`](docs/06_data_dictionary.md) | Column-by-column schema for all CSV and JSON logs |
| **6. Compliance** | [`docs/07_reproducibility_checklist.md`](docs/07_reproducibility_checklist.md) | Official ACL/NeurIPS Responsible NLP checklist |
| **7. Reproduction** | [`docs/08_reproduction_guide.md`](docs/08_reproduction_guide.md) | Step-by-step terminal commands from scratch |
| **8. Evaluation** | [`docs/09_defense_checklist.md`](docs/09_defense_checklist.md) | 5-min summary, claim defense, tough questions & answers |

---

## ⚡ Instant Verification & Defense Dashboard

To verify all claims, model coefficients, and data integrity in seconds:

```bash
python project/verify_all.py
```

This runs an automated end-to-end check of:
- All raw CSV and JSON files (25,200 rows, 3 seeds)
- Linear Mixed Model statistics ($\beta = +0.0342$, $p < 10^{-8}$)
- Length-binned interactions and IMDB null result ($p = 0.153$)
- Dual worst-case robustness tables across all 7 erasure schemes
- Consistency between raw logs and LaTeX numbers in `paper/main.tex`

---

## 📂 Repository Directory Tree

```
Research/
├── overleaf_paper.zip            # Self-contained Overleaf upload package (paper + styles + figures)
└── project/
    ├── AUDIT.md                  # Pre-revision audit of discrepancies and mathematical root causes
    ├── CHANGELOG.md              # Detailed log of every changed claim and methodological upgrade
    ├── 09_claims_evidence.md     # Matrix mapping all paper claims to exact log files and statistics
    ├── CITATION.cff              # Academic citation metadata
    ├── LICENSE.md                # MIT License
    ├── README.md                 # This overview
    ├── requirements.txt          # Python dependencies
    ├── verify_all.py             # Single-command verification dashboard
    ├── run_tests.py              # Unit tests for axioms, determinism, and masking
    │
    ├── docs/                     # Full Research Documentation Suite
    │   ├── README.md             # Navigation index and research roadmap
    │   ├── 01_research_proposal.md
    │   ├── 02_literature_review.md
    │   ├── 03_experiment_log.md
    │   ├── 04_methodology_details.md
    │   ├── 06_data_dictionary.md
    │   ├── 07_reproducibility_checklist.md
    │   ├── 08_reproduction_guide.md
    │   └── 09_defense_checklist.md
    │
    ├── data/                     # Cached datasets (SST-2 validation, IMDB test)
    │   ├── sst2_val.json
    │   └── imdb_500.json
    │
    ├── figures/                  # Publication-ready figures (PNG, 150 DPI)
    │   ├── fig1_grid_heatmap.png # 4x4 In-Domain cross-evaluation matrix
    │   ├── fig2_erasure_comparison.png # Method comparison across 7 erasure schemes
    │   ├── fig3_kendall_tau.png  # Ranking stability correlation matrix
    │   └── fig4_npig_ablation.png# Temperature ablation & entropy collapse curve
    │
    ├── paper/                    # Overleaf-ready LaTeX manuscript folder
    │   ├── main.tex              # Camera-ready ACL-format manuscript
    │   ├── main.bib              # Verified BibTeX (all DOIs / ACL IDs verified)
    │   ├── acl.sty               # ACL style file
    │   ├── acl_natbib.bst        # ACL bibliography style
    │   └── OVERLEAF_README.md    # Overleaf upload and compile guide
    │
    ├── results/                  # Raw experimental logs (every number originates here)
    │   ├── sst2_revision_all450.csv # Master per-example log (25,200 rows)
    │   ├── sst2_revision_seed42.csv
    │   ├── sst2_revision_seed123.csv
    │   ├── sst2_revision_seed999.csv
    │   ├── np_ig_ablation.csv    # Temperature sweep (0.01 to inf)
    │   ├── steps_stability.csv   # M=20 vs 50 vs 100 Spearman stability
    │   └── revision_analysis_results.json # LMM, ANOVA, and worst-case summary
    │
    └── src/                      # Clean, documented source code
        ├── ig_revision.py        # Riemann IG and logit-space Grad x Input
        ├── erasure_revision.py   # 7 erasure schemes (PAD, MASK, UNK, ZERO_EMB, DELETE, RANDOM, MLM)
        ├── run_revision_experiments.py # High-resolution execution pipeline (M=50)
        ├── run_revision_analysis.py    # LMM and dual worst-case statistical engine
        ├── generate_revision_figures.py# Vector-styled matplotlib figure generator
        └── write_revision_paper.py     # Programmatic compiler from JSON to LaTeX
```

---

## 🚀 How to Upload to Overleaf

1. Download [`overleaf_paper.zip`](../overleaf_paper.zip).
2. Go to [Overleaf](https://www.overleaf.com) $\to$ **New Project** $\to$ **Upload Project**.
3. Select `overleaf_paper.zip`. Overleaf will unpack `main.tex`, `main.bib`, `acl.sty`, `acl_natbib.bst`, and all 4 figures.
4. Set compiler to **pdfLaTeX** and click **Recompile**.

---

## 📜 Citation

```bibtex
@misc{pantra2026coupling,
  title        = {Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines},
  author       = {Pantra, Kamalesh},
  year         = {2026},
  howpublished = {\url{https://github.com/kamaleshpantra/Integrated-gradients-coupling}}
}
```
