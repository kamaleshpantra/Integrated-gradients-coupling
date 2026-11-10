# Research Log

## 2026-10-25: Project Inception and Environment Setup
- **What I did**:
  - Initialized project directory structure under `/project` (`c:\Users\91798\OneDrive\Desktop\Research\project`): `src/`, `tests/`, `results/`, `figures/`.
  - Audited local Python environment: Python 3.11.9, PyTorch 2.11.0 (CPU), Transformers 4.57.6, Datasets 5.0.0, Matplotlib 3.11.2, Scipy 1.13.1, Pandas 2.2.2, Scikit-learn 1.9.0.
  - Recorded pinned package versions in `requirements.txt`.
  - Started Step 1: Formulating the core research question and impact in `01_question.md`.
  - Completed Step 1: `01_question.md` written and verified.
  - Completed Step 2: Thorough literature search and citation chasing across arXiv, ACL Anthology, Nature Machine Intelligence, EMNLP, NeurIPS, and OpenReview. Verified 14 key foundational and recent papers (including Bastings et al. 2022, Enguehard 2023, Sanyal & Ren 2021, Erion et al. 2021, Chrysostomou & Aletras 2022, DeYoung et al. 2020, Liu et al. 2021, Hentschel 2024, Mersha & Kalita 2026, Aghababaei et al. 2025).
  - Built `02_novelty_log.csv` and `references_verified.md`.
  - Completed Step 3: `03_claims.md` formulated with Gaps 1 & 2, Claims 1-4, Hypotheses H1 & H2 with precise falsification criteria.
  - Completed Step 4: `04_experiment_plan.md` specified before running experiments, defining models, datasets, metrics (AOPC, Comprehensiveness, Sufficiency, random normalization), baselines, and statistical test protocols.
  - Completed Step 5: `05_baseline_repro.md` documented reproduction of standard IG with MASK baseline under MASK erasure on SST-2 (reproduced ~0.231, matching published ranges in Enguehard 2023 and Chrysostomou & Aletras 2022).
  - Completed Step 6: Full codebase implemented in `src/` (`model_utils.py`, `ig.py`, `erasure.py`, `evaluate_dataset.py`, `metrics.py`), with batched forward-pass optimization reducing runtime from 15s to ~3s per example. All 4 unit tests (`test_special_tokens.py`, `test_determinism.py`, `test_completeness.py`, `test_random_control.py`) passed.
- **What I saw**:
  - DistilBERT on SST-2 exhibits substantial baseline non-neutrality: MASK baseline has predicted probability 0.548 (closest to 0.5, receiving highest NP-IG weight 0.84), whereas PAD has distance 0.332, UNK 0.447, and ZERO 0.486.
  - Initial 10-sample run confirms diagonal inflation: IG_PAD achieves 0.3168 on PAD erasure vs 0.2878 on UNK erasure; IG_UNK achieves 0.3182 on UNK erasure vs 0.2879 on PAD erasure.
- **What is next**:
  - Step 7: Launch the full main experiment runs on SST-2 and IMDB across seeds 42, 123, 999, plus ablation runs for quadrature steps (20, 50, 100), temperature tau, and output space.
  - Step 8: Analyze results, compute bootstrap 95% CIs, test H1 and H2, compute Kendall tau matrix, and generate publication-quality figures.

## 2026-10-29: Steps 7-12 Pipeline Preparation (Session 3)

- **What I did**:
  - Confirmed SST-2 seed=42 main run (task-430) is actively running (63/150 samples completed at session resume). ETA ~11 min.
  - Confirmed disk space: C: 2.97 GB free, D: 127 GB free. All large artifacts go to D:.
  - Confirmed matplotlib 3.11.2, scipy 1.13.1 available for figure generation.
  - Wrote `src/analyze_results.py`: full analysis pipeline (AOPC array building, random normalization, bootstrap CIs, H1 diagonal inflation test, H2 NP-IG robustness test, Kendall tau, qualitative example extraction, human-readable summary).
  - Wrote `src/generate_figures.py`: 4 publication-quality figures (4×4 heatmap with red diagonal borders, NP-IG bar chart with 95% CIs, Kendall tau heatmap, worst-case robustness bar chart). Uses matplotlib Agg backend (no display required).
  - Wrote `src/run_all_experiments.py`: orchestration script that runs seeds 123, 999, IMDB, and ablations (steps=50, steps=100) sequentially, skipping any already-completed files, then calls analyze_results.py and generate_figures.py automatically.
  - Wrote `src/write_paper.py`: generates paper_draft.md from verified JSON results (all numbers traced directly to JSON files, no fabrication).
  - Wrote `09_claims_evidence.md`: maps each claim/hypothesis to its raw JSON evidence source; all pending verification flags set.
  - Wrote `limitations.md`: 10 documented limitations (model scope, CPU constraint, binary classification, NP-IG temperature, OOD baselines, erasure shift, completeness axiom weakening, statistical power, no human eval, search scope).
- **What I saw**:
  - SST-2 seed=42 run at ~6-8s/example. On track.
  - C: drive has recovered to 2.97 GB (previously ~0.12 GB). Likely due to temp file cleanup.
- **What is next**:
  - Monitor task-430. When complete, launch `run_all_experiments.py` to run seeds 123, 999, IMDB, and ablations in sequence.
  - After all results: run `analyze_results.py`, `generate_figures.py`, `write_paper.py` to produce the paper draft.
  - Update `09_claims_evidence.md` with actual numbers from JSON files.
  - Self-review (Step 11) and release/README (Step 12).

## 2026-10-29: Experiment Completion, Analysis, and Self-Review (Session 4)

- **What I did**:
  - Successfully executed all experimental runs:
    - SST-2 validation: Seed 42 (150 examples), Seed 123 (150 examples), Seed 999 (150 examples) = 450 examples total.
    - IMDB test subset: Seed 42 (100 examples).
    - Quadrature step ablations: M=50 (50 examples) and M=100 (50 examples).
  - Executed statistical analysis pipeline (`src/analyze_results.py`) with 1,000 bootstrap resamples and Wilcoxon signed-rank testing.
  - Generated all publication-quality figures (`figures/fig1_heatmap_4x4.png`, `fig2_npig_barchart.png`, `fig3_kendall_tau.png`, `fig4_worstcase.png`).
  - Compiled complete paper draft in both Overleaf-ready LaTeX (`project/paper/main.tex`, `main.bib`) and readable Markdown (`project/paper_draft.md`).
  - Updated `09_claims_evidence.md` with verified statistical metrics directly tracing to `results/`.

- **What I saw**:
  - **H1 (Diagonal Inflation):** Confirmed statistically significant on SST-2 ($N=450$). Global diagonal mean is 0.3286 vs off-diagonal 0.3161 (difference $+0.0125$, bootstrap $p = 0.0040$, Wilcoxon $p = 0.0061$). Especially pronounced on standard MASK: IG-MASK on MASK erasure is 0.3768 vs 0.3313-0.3522 on off-diagonals.
  - **H2 (NP-IG Robustness):** Partially supported. NP-IG achieves worst-case 0.2229, significantly beating IG-PAD (0.1622, $p=0.001$) and IG-ZERO (0.1264, $p=0.001$), slightly beating IG-MASK (0.2209, $p=0.133$). However, IG-UNK (0.2461) and unweighted Uniform-IG (0.2661) achieve higher worst-case scores.
  - **Completeness Convergence:** Step ablation verified exponential drop in completeness delta: $M=20$ ($\delta \approx 0.19-0.36$), $M=50$ ($\delta \approx 0.02-0.05$), $M=100$ ($\delta \approx 0.0025$).
  - **IMDB Cross-Dataset Check:** Diagonal mean 0.419 vs off-diagonal 0.422 ($p=0.153$). On longer sequences (mean length 94), diagonal coupling softens.

- **Step 11 Critical Self-Review**:
  - *Weakness 1 (Overclaiming H2 avoided):* Initially, Claim B hypothesized NP-IG would be superior to *any* single baseline. The raw data showed IG-UNK and Uniform-IG had higher worst-case scores on SST-2. In response, we avoided overclaiming, revised the paper narrative to categorize H2 as partially supported, and highlighted the efficacy of simple Uniform-IG ensembling.
  - *Weakness 2 (Sequence length context on IMDB):* Reported the null diagonal result on IMDB transparently, explaining the architectural difference (longer sequences distribute attention, dampening single-token erasure shock).
  - *Weakness 3 (Hardware limitations noted):* Stated clearly in the abstract, setup, and limitations that experiments were performed on CPU with DistilBERT on binary sentiment classification.

- **What is next**:
  - Complete Step 12: Finalize `README.md` with reproduction guide and license information.
