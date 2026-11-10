# Data Dictionary: All Result Files

**Purpose:** Documents every column, variable, and field in the project's data and result files. Required for reproducibility per ACL/NeurIPS checklist guidelines.

---

## 1. Raw Data Files

### `project/data/sst2_val.json`

SST-2 validation split from the GLUE benchmark (Wang et al., 2018).

| Field | Type | Description |
|---|---|---|
| `idx` | int | Example index (0-based) in the original dataset |
| `sentence` | string | Raw input text |
| `label` | int | Ground truth label (0 = negative, 1 = positive) |

**Source:** Hugging Face `datasets` library, `glue/sst2`, split `validation`  
**Size:** 872 examples total; we use the first 150 per seed

### `project/data/imdb_500.json`

IMDB movie review dataset (Maas et al., 2011).

| Field | Type | Description |
|---|---|---|
| `text` | string | Full movie review text |
| `label` | int | Sentiment label (0 = negative, 1 = positive) |

**Source:** Hugging Face `datasets` library, `imdb`, split `test`  
**Size:** 500 examples; we use 100

---

## 2. Main Experiment Result Files

### `project/results/sst2_revision_all450.csv` (Master File)

The primary result file containing all experiment outcomes. **Every number in the paper traces to this file.**

| Column | Type | Description | Example Values |
|---|---|---|---|
| `seed` | int | Random seed used | 42, 123, 999 |
| `example_id` | int | Example index within the seed's batch | 0–149 |
| `text` | string | Input sentence text | "a stirring , funny and ..." |
| `label` | int | True label | 0 or 1 |
| `pred_label` | int | Model's predicted label | 0 or 1 |
| `pred_prob` | float | Model's predicted probability for true class | 0.0–1.0 |
| `method` | string | Attribution method used | IG_PAD, IG_MASK, IG_UNK, IG_ZERO, NP_IG, UNIFORM_IG, GRAD_X_INP, RANDOM |
| `scheme` | string | Erasure scheme applied | PAD, MASK, UNK, ZERO_EMB, DELETE, RANDOM_TOKEN, MLM_INFILL |
| `comp_10` | float | Comprehensiveness at 10% removal | 0.0–1.0 |
| `comp_20` | float | Comprehensiveness at 20% removal | 0.0–1.0 |
| `comp_30` | float | Comprehensiveness at 30% removal | 0.0–1.0 |
| `comp_50` | float | Comprehensiveness at 50% removal | 0.0–1.0 |
| `aopc` | float | Mean of comp_10, comp_20, comp_30, comp_50 | 0.0–1.0 |
| `norm_comp_aopc` | float | AOPC minus random-method AOPC for same scheme | Can be negative |
| `n_content_tokens` | int | Number of content tokens (excl. CLS, SEP) | 2–50+ |
| `n_steps` | int | Number of IG integration steps | 50 |
| `baseline_neutrality` | float | $\|p_c(\text{baseline}) - 0.5\|$ | 0.0–0.5 |

**Derived columns used in analysis but not stored:**
- `is_matched` = 1 if `method` baseline matches `scheme` (e.g., IG_PAD + PAD)
- `unique_id` = `{seed}_{example_id}` (for LMM grouping)
- `length_bin` = Short ($\leq 10$), Medium (11–20), Long ($> 20$)

**Total rows:** 25,200 = 450 examples × 8 methods × 7 schemes

### Per-Seed Files

Same schema as above, 8,400 rows each:
- `project/results/sst2_revision_seed42.csv`
- `project/results/sst2_revision_seed123.csv`
- `project/results/sst2_revision_seed999.csv`

---

## 3. Steps Stability File

### `project/results/steps_stability.csv`

| Column | Type | Description |
|---|---|---|
| `example_id` | int | Example index (0–49) |
| `baseline` | string | IG baseline (PAD, MASK, UNK, ZERO) |
| `n_steps` | int | Integration step count (20, 50, 100) |
| `completeness_delta` | float | $\|f(x) - f(b) - \sum \text{IG}_i\|$ |
| `attributions` | string | JSON list of per-token IG values |
| `spearman_vs_100` | float | Spearman rank correlation of token attributions vs. M=100 |

---

## 4. NP-IG Ablation File

### `project/results/np_ig_ablation.csv`

| Column | Type | Description |
|---|---|---|
| `seed` | int | Random seed |
| `example_id` | int | Example index |
| `config` | string | Ablation configuration (tau_0.01, tau_0.03, ..., Uniform, Argmin_Neutrality, Random_Weights) |
| `scheme` | string | Erasure scheme |
| `weight_PAD` | float | NP-IG weight on PAD baseline |
| `weight_MASK` | float | NP-IG weight on MASK baseline |
| `weight_UNK` | float | NP-IG weight on UNK baseline |
| `weight_ZERO` | float | NP-IG weight on ZERO baseline |
| `entropy` | float | Shannon entropy of weight distribution (nats) |
| `norm_comp_aopc` | float | Normalized comprehensiveness AOPC |

---

## 5. Analysis Results JSON

### `project/results/revision_analysis_results.json`

Top-level keys and their contents:

| Key | Type | Description |
|---|---|---|
| `h1_lmm` | object | Linear mixed model results |
| `h1_lmm.lmm_matched_coef` | float | $\beta_{\text{matched}}$ coefficient |
| `h1_lmm.lmm_matched_se` | float | Standard error of $\beta$ |
| `h1_lmm.lmm_matched_p` | float | $p$-value |
| `h1_lmm.lmm_ci_lower` | float | 95% CI lower bound |
| `h1_lmm.lmm_ci_upper` | float | 95% CI upper bound |
| `h1_lmm.variance_explained_pct` | object | ANOVA SS percentages for baseline, scheme, matched, residual |
| `h1_lmm.per_baseline_contrasts` | object | Per-baseline diagonal vs. off-diagonal contrasts with Holm p-values |
| `h1_lmm.length_binned_effects` | object | LMM results per length bin (Short, Medium, Long) |
| `worst_case` | object | Dual worst-case robustness tables |
| `worst_case.in_domain_4` | object | 4-scheme in-domain worst-case per method |
| `worst_case.with_delete_5` | object | 5-scheme (+ DELETE) worst-case |
| `worst_case.all_7` | object | 7-scheme (all) worst-case |
| `steps_stability` | object | Convergence and Spearman stability results |
| `np_ig_ablation` | object | Temperature sweep results per config |

---

## 6. Legacy Result Files (Pre-Revision)

These files are from the original (pre-audit) experiments and are retained for audit trail purposes:

| File | Description | Status |
|---|---|---|
| `sst2_main_seed42.json` | Original pilot (M=20, 4 schemes) | Superseded by `sst2_revision_seed42.csv` |
| `sst2_main_seed123.json` | Original seed 123 | Superseded |
| `sst2_main_seed999.json` | Original seed 999 | Superseded |
| `imdb_main_seed42.json` | IMDB experiment | Still current (not re-run) |
| `analysis_sst2_*.json` | Original analysis outputs | Superseded by `revision_analysis_results.json` |
| `analysis_imdb_*.json` | IMDB analysis outputs | Still current |
