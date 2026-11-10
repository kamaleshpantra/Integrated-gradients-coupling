# Research Proposal: Baseline-Evaluation Coupling in Integrated Gradients for Text

**Student:** [Your Name]  
**Advisor:** [Advisor Name]  
**Department:** [Department of Computer Science / NLP Lab]  
**Date Submitted:** 2026-09-01  
**Status:** Approved → Experiments Complete → Paper Under Revision

---

## 1. Abstract

Integrated Gradients (IG) is one of the most widely adopted axiomatic attribution methods for explaining neural network predictions. In NLP, practitioners must choose both a *reference baseline token* (e.g., `[PAD]`, `[MASK]`, `[UNK]`, or a zero embedding) for computing IG attributions and an *erasure operator* for evaluating those attributions via comprehensiveness metrics. These two choices are typically matched — an author using `IG-MASK` will evaluate with `MASK` erasure — but the consequences of this coupling have not been systematically studied. We propose a cross-evaluation study that decouples these choices through a full factorial grid, tests whether matched pairs produce artificially inflated faithfulness scores, and evaluates whether multi-baseline ensemble methods provide more robust attributions.

---

## 2. Problem Statement

### 2.1 Background

Feature attribution methods explain model decisions by assigning importance scores to input tokens. Integrated Gradients (Sundararajan et al., 2017) is axiomatic — it satisfies completeness (attributions sum to the output difference between input and baseline) and sensitivity (changed features receive non-zero attribution). In NLP, IG requires:

1. **A reference baseline** $\mathbf{b}$: a token embedding representing "absence of information" (typically `[PAD]`, `[MASK]`, `[UNK]`, or $\mathbf{0}$).
2. **An evaluation protocol**: comprehensiveness (DeYoung et al., 2020) measures prediction drop when top-attributed tokens are erased from the input.

### 2.2 The Gap

The evaluation itself requires an **erasure operator** — replacing removed tokens with some substitute. Practitioners almost universally match the erasure token to the IG baseline token without justification. This creates a potential confound:

- IG attributes the difference $f(\mathbf{x}) - f(\mathbf{b})$.
- Erasing tokens with $\mathbf{b}$ evaluates exactly this counterfactual.
- The evaluation may therefore be *tautologically aligned* with IG's mathematical objective, inflating scores relative to what a method-agnostic evaluation would show.

### 2.3 Why This Matters

If matched evaluation inflates scores, then:
- Published comprehensiveness comparisons between IG variants using different baselines are not apples-to-apples.
- Practitioners may choose suboptimal baselines that merely score well on their own matched evaluation.
- The faithfulness literature may overstate IG's quality relative to non-gradient methods.

---

## 3. Research Questions

| ID | Research Question |
|:---:|---|
| **RQ1** | Does matching the IG baseline token with the evaluation erasure token produce statistically significant inflation in comprehensiveness scores compared to mismatched configurations? |
| **RQ2** | Does multi-baseline ensembling (Uniform-IG with equal weights, or Neutrality-Prediction IG with adaptive weights) provide more robust attributions across diverse evaluation conditions? |
| **RQ3** | How does input sequence length modulate the coupling effect? Is coupling stronger on short inputs where single-token replacement has larger contextual impact? |

---

## 4. Hypotheses

### H1: Matched-Pair Inflation (Primary)
> **When the IG baseline token and the evaluation erasure token are identical (matched), normalized comprehensiveness scores will be significantly higher than when they are mismatched.**

- *Test*: Compare diagonal cells (matched) vs. off-diagonal cells (mismatched) in a baseline × erasure grid.
- *Statistical method*: Linear mixed-effects model with random intercepts per example.
- *Significance threshold*: $p < 0.05$ with Holm-Bonferroni correction for per-baseline contrasts.

### H2: Multi-Baseline Robustness
> **Multi-baseline IG methods (Uniform-IG, NP-IG) will achieve higher worst-case comprehensiveness across erasure schemes than any single-baseline IG variant.**

- *Test*: Compare worst-case (minimum across schemes) of multi-baseline methods vs. single-baseline methods.
- *Metric*: $\text{Worst}_{\text{agg}} = \min_{\sigma \in \Sigma} \mathbb{E}_i[\text{NormComp}_i(\sigma)]$.

### H3: Length Interaction
> **The matched-pair inflation effect will be significantly stronger on short inputs ($\leq 10$ tokens) than on long inputs ($> 20$ tokens).**

- *Test*: Fit length-binned LMMs and compare $\beta_{\text{matched}}$ coefficients across bins.
- *Rationale*: On short sequences, erasing a single token to the baseline creates a proportionally larger shift toward the IG reference state.

---

## 5. Methodology

### 5.1 Model and Data

| Component | Specification |
|---|---|
| **Model** | DistilBERT-base-uncased fine-tuned on SST-2 (accuracy 91.3%) |
| **Primary Dataset** | SST-2 validation set (Stanford Sentiment Treebank, binary) |
| **Secondary Dataset** | IMDB movie reviews (Maas et al., 2011) — longer sequences |
| **Sample Size** | $N = 450$ SST-2 examples (150 per seed × 3 seeds: 42, 123, 999); $N = 100$ IMDB |
| **Integration Steps** | $M = 50$ (with $M = 100$ convergence check on 50-example subset) |

### 5.2 Cross-Evaluation Grid

We construct a $4 \times 4$ grid crossing 4 IG baselines with 4 matched erasure schemes, plus additional unmatched schemes:

**IG Baselines ($\mathcal{B}$):** PAD, MASK, UNK, ZERO (all-zeros embedding)

**Erasure Schemes ($\Sigma$):**
- *In-domain (4):* PAD (with `attention_mask=0`), MASK, UNK, ZERO_EMB (zero embedding)
- *Structural (1):* DELETE (physically remove token)
- *Held-out (2):* RANDOM_TOKEN (random vocabulary ID, averaged over 5 draws), MLM_INFILL (DistilBERT masked LM top-1 prediction)

### 5.3 Attribution Methods (8 total)

1. **IG-PAD** — single-baseline IG with `[PAD]` reference
2. **IG-MASK** — single-baseline IG with `[MASK]` reference
3. **IG-UNK** — single-baseline IG with `[UNK]` reference
4. **IG-ZERO** — single-baseline IG with zero-embedding reference
5. **Uniform-IG** — equal-weight average of all 4 single-baseline IGs
6. **NP-IG** — neutrality-weighted average (softmax temperature $\tau$, weighting baselines by proximity to $p = 0.5$)
7. **Grad × Input** — gradient baseline (logit-space, absolute value ranking)
8. **Random** — random attribution control for normalization

### 5.4 Evaluation Metric

**Normalized Comprehensiveness AOPC:**
$$\text{NormComp}(\mathbf{a}, \sigma) = \text{AOPC}_{\text{Comp}}(\mathbf{a}, \sigma) - \text{AOPC}_{\text{Comp}}(\mathbf{a}_{\text{rand}}, \sigma)$$

Computed at top-$k$ fractions $\{0.10, 0.20, 0.30, 0.50\}$ and averaged (Area Over the Perturbation Curve).

### 5.5 Statistical Analysis Plan

1. **Linear Mixed-Effects Model**: $\text{NormComp}_{i,b,s} = \mu + \alpha_b + \gamma_s + \beta_{\text{matched}} \cdot \mathbb{I}[b=s] + u_i + \epsilon_{i,b,s}$
2. **ANOVA variance decomposition** of baseline, scheme, and matched-indicator sums of squares.
3. **Per-baseline matched contrasts** with Holm-Bonferroni correction.
4. **Length-binned sub-analysis**: Short ($\leq 10$), Medium (11–20), Long ($> 20$ tokens).
5. **NP-IG temperature sweep**: $\tau \in \{0.01, 0.03, 0.1, 0.3, 1.0, \infty\}$.

---

## 6. Expected Contributions

1. **First systematic cross-evaluation** of baseline × erasure interaction in IG for NLP.
2. **Formal quantification** of matched-pair inflation effect size and its practical significance.
3. **Practical recommendation** on whether multi-baseline methods (Uniform-IG or NP-IG) should replace single-baseline IG.
4. **Reproducibility package** with all seeds, configs, raw per-example CSVs, and analysis scripts.

---

## 7. Timeline

| Week | Phase | Tasks |
|:---:|---|---|
| 1 | **Setup** | Download models, prepare SST-2/IMDB data, implement IG and erasure pipelines |
| 2 | **Pilot** | Run 10-example pilot on SST-2, verify completeness axiom, debug edge cases |
| 3 | **Main Grid** | Run full 4×4 grid at $M=50$ across 3 seeds ($N=450$) |
| 4 | **Held-Out Erasure** | Implement and run RANDOM_TOKEN + MLM_INFILL erasure schemes |
| 5 | **NP-IG Ablation** | Temperature sweep, argmin-neutrality, random-weights control |
| 6 | **Statistical Analysis** | Fit LMM, ANOVA, per-baseline contrasts, length-binned models |
| 7 | **Paper Writing** | Draft all sections, generate figures and tables from logged data |
| 8 | **Revision & Review** | Internal review, audit all numbers against raw logs, finalize |

---

## 8. Required Resources

| Resource | Specification |
|---|---|
| **GPU** | 1× NVIDIA GPU with ≥6 GB VRAM (DistilBERT fits on consumer GPUs) |
| **Storage** | ~10 GB for raw result files, model checkpoints, and datasets |
| **Software** | Python 3.10+, PyTorch, Transformers, statsmodels, scipy, matplotlib |
| **Datasets** | SST-2 (via Hugging Face `glue`), IMDB (via Hugging Face `imdb`) |
| **Models** | `distilbert-base-uncased-finetuned-sst-2-english`, `distilbert-base-uncased` (MLM head) |

---

## 9. Key References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*.
2. DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL 2020*.
3. Bastings, J., Ebert, S., Zablotskaia, P., Sandholm, A., & Filippova, K. (2022). Will You Find These Shortcuts? *EMNLP 2022*.
4. Sanyal, S. & Ren, X. (2021). Discretized Integrated Gradients for Explaining Language Models. *EMNLP 2021*.
5. Enguehard, J. (2023). Sequential Integrated Gradients. *Findings of ACL 2023*.
6. Erion, G., et al. (2021). Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients. *Nature Machine Intelligence*.
7. Chrysostomou, G. & Aletras, N. (2022). An Empirical Study on Explanations in Out-of-Domain Settings. *ACL 2022*.

---

## 10. Approval

| | Signature | Date |
|---|---|---|
| **Student** | __________________ | __________ |
| **Advisor** | __________________ | __________ |
