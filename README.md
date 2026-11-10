# Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Reproducibility: Verified](https://img.shields.io/badge/reproducibility-ACL%2FNeurIPS%20Checklist-green.svg)](docs/07_reproducibility_checklist.md)
[![Status: Ready for Evaluation](https://img.shields.io/badge/status-ready%20for%20evaluation-success.svg)](paper/main.tex)

**Author:** Kamalesh Pantra  
**Target Venue:** ACL / EMNLP Format

This repository contains the official codebase, benchmark datasets, raw evaluation logs, statistical models, publication figures, and LaTeX manuscript for the empirical study:

> **"Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines"**

---

## 📌 Executive Summary of Findings

1. **Matched-Pair Coupling is Statistically Real ($\beta = +0.0342$):**
   Evaluating Integrated Gradients with an erasure token identical to the reference baseline (e.g. `IG-MASK` under `MASK` erasure) produces a statistically significant positive boost on faithfulness ($\beta = +\mathbf{0.0342}$, $95\%$ CI: $[+0.0222, +0.0462]$, $p = \mathbf{2.42 \times 10^{-8}}$ via Linear Mixed Model across $N=450$ examples from 3 seeds at $M=50$ quadrature steps).
2. **Baseline Selection Dominates the Coupling Artifact (7.2× Ratio):**
   An ANOVA variance decomposition demonstrates that baseline selection accounts for **$1.22\%$** of metric variance—over **$7\times$ more variance** than the matched coupling artifact (**$0.17\%$**).
3. **Length-Dependent Interaction:**
   Coupling is strong on short sentences ($\le 10$ tokens: $\beta = +\mathbf{0.0586}$, $p=0.0046$) where replacing a single token substantially alters context, but is **completely null on IMDB long documents** ($N=100$, mean length 94 tokens: $p=0.153$).
4. **Uniform-IG Wins; NP-IG Collapses onto MASK:**
   While Neutrality-Prediction IG (NP-IG) was designed to weight baselines dynamically, at low temperature ($\tau=0.1$) it collapses onto `MASK` ($90.0\%$ weight, entropy $0.391$ nats). Simple, parameter-free **Uniform-IG** achieves the highest worst-case robustness across all 7 evaluated erasure schemes ($\text{Def 2} = \mathbf{0.3316}$, $\text{Def 1} = \mathbf{0.1285}$), including two held-out schemes (`RANDOM_TOKEN` and `MLM_INFILL`).

---

## 📂 Repository Directory Layout

The repository follows standard AI research and open-science repository conventions:

```text
├── .gitignore                  # Strict Git ignore rules for virtualenvs, caches, logs
├── CITATION.cff                # Official GitHub citation metadata
├── LICENSE.md                  # MIT License
├── README.md                   # Repository overview, results & instructions
├── requirements.txt            # Locked Python dependencies
├── setup.py                    # Package installer for editable local development
├── verify_all.py               # Root convenience runner for verification
│
├── configs/                    # Experiment hyperparameters & configuration
│   └── eval_config.json
│
├── data/                       # Benchmark datasets & schemas
│   ├── README.md               # Dataset documentation (SST-2, IMDB)
│   ├── sst2_val.json           # Stanford Sentiment Treebank validation split
│   └── imdb_500.json           # IMDB long-document evaluation subset
│
├── docs/                       # Formal research lifecycle documentation
│   ├── README.md               # Documentation sitemap & index
│   ├── 01_research_proposal.md # Research questions & pre-registered hypotheses
│   ├── 02_literature_review.md # Academic literature survey & foundations
│   ├── 03_experiment_log.md    # Dated experiment lab notebook
│   ├── 04_methodology_details.md # Riemann quadrature & mathematical specifications
│   ├── 06_data_dictionary.md   # Column schemas for all CSV & JSON logs
│   ├── 07_reproducibility_checklist.md # Official ACL/NeurIPS reproducibility checklist
│   ├── 08_reproduction_guide.md # Terminal reproduction walkthrough
│   └── 09_defense_checklist.md # Peer-review defense & anticipated questions
│
├── figures/                    # Publication figures (PNG, 150 DPI)
│   ├── fig1_grid_heatmap.png   # 4x4 In-Domain cross-evaluation matrix
│   ├── fig2_erasure_comparison.png # Grouped performance across 7 erasure schemes
│   ├── fig3_kendall_tau.png    # Kendall tau rank correlation matrix
│   └── fig4_npig_ablation.png  # NP-IG temperature ablation & entropy collapse
│
├── notes/                      # Internal experiment logs, audits & claims mapping
│   ├── README.md               # Notes directory index
│   ├── AUDIT.md                # Pre-revision audit of statistical discrepancies
│   ├── CHANGELOG.md            # Detailed record of all changes & upgrades
│   └── 09_claims_evidence.md   # Claims-to-evidence traceability matrix
│
├── paper/                      # Camera-ready ACL LaTeX manuscript & Overleaf bundle
│   ├── main.tex                # Primary paper manuscript
│   ├── main.bib                # Verified bibliography
│   ├── acl.sty                 # ACL style definition
│   ├── acl_natbib.bst          # Bibliography style
│   ├── overleaf_paper.zip      # 1-click self-contained Overleaf upload zip
│   └── OVERLEAF_README.md      # Overleaf compilation instructions
│
├── presentation/               # Slide deck & presentation assets
│   ├── README.md               # Presentation overview
│   └── research_presentation.pptx # 15-slide widescreen defense deck
│
├── results/                    # Raw per-example logs & statistical JSON artifacts
│   ├── sst2_revision_all450.csv # Master 450-example dataset (25,200 rows)
│   ├── np_ig_ablation.csv      # Temperature ablation logs
│   ├── steps_stability.csv     # Quadrature convergence (M=20, 50, 100)
│   └── revision_analysis_results.json # LMM, ANOVA, and Kendall tau statistics
│
├── scripts/                    # Command-line tools & replication runners
│   ├── verify_all.py           # Master verification dashboard
│   ├── compare_baseline_repro.py # Published literature benchmark replication
│   ├── generate_presentation.py # Automated PowerPoint presentation builder
│   └── run_tests.py            # Automated test suite runner
│
├── src/                        # Core modular Python library
│   ├── ig.py                   # Integrated Gradients & Riemann quadrature
│   ├── erasure.py              # 7 erasure operators (in-domain & held-out)
│   ├── metrics.py              # Comprehensiveness, AOPC, NormComp calculations
│   └── evaluate_dataset.py     # Evaluation pipeline
│
└── tests/                      # Unit & integration tests
    ├── test_completeness.py    # Axiomatic completeness convergence test
    ├── test_determinism.py     # Random seed reproducibility test
    ├── test_random_control.py  # Attribution vs random guessing test
    └── test_special_tokens.py  # [CLS]/[SEP] exclusion test
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/kamaleshpantra/Integrated-gradients-coupling.git
cd Integrated-gradients-coupling

# Install dependencies
pip install -r requirements.txt

# (Optional) Install local package in editable mode
pip install -e .
```

### 2. Run Instant Statistical Verification (10 Seconds)
To verify all statistical claims, LMM coefficients, and data integrity:
```bash
python scripts/verify_all.py
```
*(Or use `python verify_all.py` from the root).*

### 3. Compare Against Published Benchmarks
To compare our reproduced numbers against published benchmarks from ICML 2017, NeurIPS 2019, ACL 2022, and ACL 2023:
```bash
python scripts/compare_baseline_repro.py
```

### 4. Run Automated Unit Tests
To verify mathematical axioms and seed determinism on DistilBERT:
```bash
python scripts/run_tests.py
# or using pytest:
pytest tests/
```

---

## 📊 Core Empirical Results

### Table 1: In-Domain 4×4 Cross-Evaluation Matrix (SST-2, N=450, M=50)
*Values report Normalized Comprehensiveness AOPC. Bold indicates matched diagonal pairs.*

| Attribution Baseline | Erase-`PAD` | Erase-`MASK` | Erase-`UNK` | Erase-`ZERO_EMB` | Off-Diagonal Mean | Matched Boost ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **IG-PAD** | **0.3162** | 0.3160 | 0.2775 | 0.2677 | 0.2871 | **+0.0292** ($p = 0.007$) |
| **IG-MASK** | 0.3476 | **0.3775** | 0.3468 | 0.2906 | 0.3283 | **+0.0491** ($p < 0.001$) |
| **IG-UNK** | 0.3860 | 0.3778 | **0.3946** | 0.3420 | 0.3686 | **+0.0260** ($p = 0.007$) |
| **IG-ZERO** | 0.2713 | 0.2649 | 0.2596 | **0.2976** | 0.2653 | **+0.0323** ($p = 0.010$) |

### Table 2: Worst-Case Robustness Across All 7 Erasure Schemes
*Def 1: Mean of per-example worst. Def 2: Minimum of per-scheme means.*

| Method | In-Domain Def 1 | In-Domain Def 2 | Held-Out RANDOM | Held-Out MLM | **All-7 Def 2 (Worst-Case)** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Uniform-IG** | 0.2136 | 0.3408 | 0.3343 | 0.3316 | **0.3316** 🏆 |
| **NP-IG ($\tau=0.1$)** | 0.1864 | 0.3040 | 0.3145 | 0.3232 | **0.3040** |
| **IG-UNK** | 0.2025 | 0.3420 | 0.3188 | 0.3103 | **0.3103** |
| **IG-MASK** | 0.1750 | 0.2906 | 0.3009 | 0.3127 | **0.2906** |
| **IG-PAD** | 0.1324 | 0.2677 | 0.2699 | 0.2379 | **0.2379** |
| **IG-ZERO** | 0.1122 | 0.2596 | 0.2512 | 0.2160 | **0.2160** |
| **Grad×Input** | -0.1175 | 0.0072 | 0.0174 | 0.0399 | **0.0072** |

---

## 📄 Overleaf / LaTeX Paper Compilation

The complete camera-ready manuscript is in `paper/`:
- **Direct Zip Upload:** Upload [`paper/overleaf_paper.zip`](paper/overleaf_paper.zip) directly to [Overleaf](https://www.overleaf.com) via *"New Project" $\rightarrow$ "Upload Project"*.
- **Local Compilation:**
  ```bash
  cd paper
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

---

## 📜 Citation

If you use this benchmark, code, or methodology, please cite:

```bibtex
@misc{pantra2026coupling,
  title        = {Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines},
  author       = {Pantra, Kamalesh},
  year         = {2026},
  howpublished = {\url{https://github.com/kamaleshpantra/Integrated-gradients-coupling}}
}
```

---

## ⚖️ License
This project is licensed under the MIT License — see the [LICENSE.md](LICENSE.md) file for details.
