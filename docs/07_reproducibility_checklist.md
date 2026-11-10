# Reproducibility Checklist (ACL / NeurIPS Standard)

**Project:** Baseline-Evaluation Coupling in Integrated Gradients for Text  
**Date Completed:** 2026-10-29  

This checklist follows the [ACL Rolling Review Responsible NLP Checklist](https://aclrollingreview.org/responsibleNLPchecklist/) and [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) standards.

---

## A. Claims and Evidence

| # | Question | Answer | Location in Paper / Project |
|:---:|---|:---:|---|
| A1 | Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? | **Yes** | Abstract (lines 29–31), `09_claims_evidence.md` |
| A2 | Does the abstract state the main findings, including key negative results? | **Yes** | Abstract includes IMDB null ($p=0.153$) and Uniform-IG > NP-IG |
| A3 | Does the paper provide a "Claims vs. Evidence" mapping? | **Yes** | `project/09_claims_evidence.md` — 7 claims mapped to exact files and statistics |

---

## B. Experimental Methodology

| # | Question | Answer | Location |
|:---:|---|:---:|---|
| B1 | Is a clear description of the mathematical setting, algorithm, and/or model provided? | **Yes** | Section 3 (Methodology), `docs/04_methodology_details.md` |
| B2 | Are all assumptions clearly stated? | **Yes** | Section 3.2 (two worst-case definitions), Discussion (two mechanistic hypotheses as hypotheses) |
| B3 | Is the complete proof / derivation provided for all theoretical results? | **N/A** | Empirical study; no novel theorems |
| B4 | Are all datasets and their preprocessing fully described? | **Yes** | Section 4, `docs/06_data_dictionary.md` |
| B5 | Is the exact experimental setup described, including hyperparameters, seeds, and number of runs? | **Yes** | Section 4: Seeds 42/123/999, M=50, N=450, fractions {0.1, 0.2, 0.3, 0.5} |
| B6 | Are error bars, confidence intervals, or significance tests reported? | **Yes** | LMM 95% CI, Holm-corrected p-values, per-baseline contrasts |
| B7 | Is there a description of the computing infrastructure used? | **Yes** | Section 4, `docs/04_methodology_details.md` §6 |

---

## C. Code and Data Availability

| # | Question | Answer | Location |
|:---:|---|:---:|---|
| C1 | Is the code for the main experiments provided? | **Yes** | `project/src/` — all scripts with docstrings |
| C2 | Is there a README explaining how to reproduce the main results? | **Yes** | `project/README.md`, `project/src/README.md` |
| C3 | Are the exact commands to reproduce each table/figure documented? | **Yes** | `docs/08_reproduction_guide.md` |
| C4 | Are all dependencies specified with versions? | **Yes** | `project/requirements.txt` |
| C5 | Are random seeds specified and fixed? | **Yes** | Seeds 42, 123, 999; documented in `docs/04_methodology_details.md` §6 |
| C6 | Are raw per-example results available (not just aggregates)? | **Yes** | `project/results/sst2_revision_all450.csv` (25,200 rows) |
| C7 | Is the data available or downloadable? | **Yes** | SST-2 and IMDB via Hugging Face `datasets`; cached in `project/data/` |

---

## D. Statistical Rigor

| # | Question | Answer | Location |
|:---:|---|:---:|---|
| D1 | Are multiple random seeds used? | **Yes** | 3 seeds (42, 123, 999), N=150 per seed |
| D2 | Is the statistical test appropriate for the data structure? | **Yes** | LMM with random intercepts for repeated-measures data |
| D3 | Are multiple comparisons corrected for? | **Yes** | Holm-Bonferroni correction for 4 per-baseline contrasts |
| D4 | Are effect sizes reported alongside p-values? | **Yes** | $\beta = +0.0342$ with CI, plus variance decomposition (0.17% of total) |
| D5 | Are negative/null results reported honestly? | **Yes** | IMDB null ($p=0.153$), NP-IG < Uniform-IG |

---

## E. Ethical Considerations

| # | Question | Answer | Location |
|:---:|---|:---:|---|
| E1 | Does the paper include an AI Assistance Statement? | **Yes** | Section "AI Assistance Statement" in `main.tex` |
| E2 | Are there potential negative societal impacts? | **Minimal** | This is a methodological study of evaluation metrics; no direct societal harm. Discussed in Limitations. |
| E3 | Are all references verified against primary sources? | **Yes** | `AUDIT.md` §5 — full citation audit with ACL Anthology/DOI verification |
| E4 | Were any references fabricated or unverified? | **Found & Fixed** | `hentschel2024canonical` removed; `bastings2022shortcuts` and `liu2021shapley` corrected |

---

## F. Artifact Packaging

| # | Question | Answer | Location |
|:---:|---|:---:|---|
| F1 | Is a self-contained Overleaf package provided? | **Yes** | `overleaf_paper.zip` (main.tex, main.bib, acl.sty, acl_natbib.bst, 4 figures) |
| F2 | Is the full audit trail documented? | **Yes** | `AUDIT.md`, `CHANGELOG.md` |
| F3 | Is the experiment log maintained? | **Yes** | `docs/03_experiment_log.md` (11 chronological entries) |
| F4 | Are all figures reproducible from raw data? | **Yes** | `src/generate_revision_figures.py` reads from CSVs |
| F5 | Are all table values reproducible from raw data? | **Yes** | `src/write_revision_paper.py` populates LaTeX from `revision_analysis_results.json` |
