# Pre-Revision Audit Report (AUDIT.md)

**Date:** 2026-10-29  
**Target Paper:** *"Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and a Neutral-Prediction Averaged Baseline"*  
**Author/Auditor:** Antigravity Research Agent  

This document conducts a rigorous point-by-point audit of the earlier paper draft, code, and numerical artifacts prior to executing new experiments. Every identified discrepancy is documented with its mathematical and programmatic root cause.

---

## 1. Discrepancy (a): Worst-Case Robustness Definition Conflict

### Problem Identified
In the previous draft, the prose in Section 5.2 and the numbers in Table 2 disagreed drastically:
- **NP-IG:** Text reported **$0.2229$**; Table 2 reported **$0.3330$**.
- **Uniform-IG:** Text reported **$0.2661$**; Table 2 reported **$0.3905$**.
- **Grad $\times$ Input:** Text reported **$-0.0910$**; Table 2 reported **$0.0064$**.
- **IG-PAD:** Text reported **$0.1622$**; Table 2 reported **$0.2786$**.
- **IG-ZERO:** Text reported **$0.1264$**; Table 2 reported **$0.2313$**.

### Root Cause Confirmation
The independent reviewer's hypothesis is **strictly confirmed**:
Two fundamentally different mathematical definitions of "worst-case" were calculated and used inconsistently across scripts:

1. **Definition 1 (Text in Section 5.2 & `analyze_results.py`): Expected Per-Example Minimum across Schemes**
   $$\text{Worst}_{\text{ex}}(\text{Method}) = \frac{1}{N} \sum_{i=1}^N \min_{\sigma \in \Sigma} \text{NormComp}_i(\text{Method}, \sigma)$$
   Here, for each individual example $i$, we take the worst erasure scheme for that specific example, and then average over the $N$ examples.

2. **Definition 2 (Table 2 & `write_paper.py`): Minimum across Schemes of the Dataset-Level Mean**
   $$\text{Worst}_{\text{agg}}(\text{Method}) = \min_{\sigma \in \Sigma} \left( \frac{1}{N} \sum_{i=1}^N \text{NormComp}_i(\text{Method}, \sigma) \right)$$
   Here, we compute the dataset-level mean score for each erasure scheme, and then select the minimum across the schemes.

By Jensen's inequality and the concavity/subadditivity of the minimum operator:
$$\frac{1}{N} \sum_{i=1}^N \min_{\sigma} X_{i,\sigma} \le \min_{\sigma} \left( \frac{1}{N} \sum_{i=1}^N X_{i,\sigma} \right)$$
Hence, $\text{Worst}_{\text{ex}}$ is always strictly lower than or equal to $\text{Worst}_{\text{agg}}$.

### Dual-Definition Comparative Table (SST-2, $N=450$)
Computed directly from `project/results/sst2_main_seed{42,123,999}.json`:

| Attribution Method | Def 1: Mean of Per-Example Worst $\mathbb{E}_i[\min_\sigma \text{Comp}_{i,\sigma}]$ | Def 2: Min of Per-Scheme Means $\min_\sigma \mathbb{E}_i[\text{Comp}_{i,\sigma}]$ | Gap ($\text{Def 2} - \text{Def 1}$) |
|---|:---:|:---:|:---:|
| **NP-IG** | 0.2229 | 0.3330 | +0.1101 |
| **Uniform-IG** | 0.2661 | 0.3905 | +0.1244 |
| **IG-PAD** | 0.1622 | 0.2786 | +0.1164 |
| **IG-MASK** | 0.2209 | 0.3313 | +0.1104 |
| **IG-UNK** | 0.2461 | 0.3639 | +0.1178 |
| **IG-ZERO** | 0.1264 | 0.2313 | +0.1049 |
| **Grad $\times$ Input** | -0.0910 | 0.0064 | +0.0974 |

### Resolution & Policy for Revision
- **Primary Definition Chosen:** **Definition 2 ($\text{Worst}_{\text{agg}}$: Minimum of per-scheme means)** is adopted as the primary metric for the revised paper. In benchmark evaluation, practitioners assess how a method ranks across datasets and evaluation protocols globally, rather than whether individual examples suffered from an adversarial erasure choice.
- **Reporting:** Both definitions will be explicitly reported in the revision table, and the distinction will be formally stated in Section 5.

---

## 2. Discrepancy (b): NP-IG Collapse onto IG-MASK

### Audit Findings
In the previous Table 2, the row for NP-IG was virtually identical to IG-MASK across all 4 erasure schemes:
- **Erase-PAD:** NP-IG = 0.3555 vs IG-MASK = 0.3518 ($\Delta = +0.0037$)
- **Erase-MASK:** NP-IG = 0.3772 vs IG-MASK = 0.3768 ($\Delta = +0.0004$)
- **Erase-UNK:** NP-IG = 0.3523 vs IG-MASK = 0.3522 ($\Delta = +0.0001$)
- **Erase-DELETE:** NP-IG = 0.3330 vs IG-MASK = 0.3313 ($\Delta = +0.0017$)

### Empirical Analysis of NP-IG Weights ($\tau = 0.1$)
Across all 450 SST-2 validation records in `project/results/sst2_main_seed{42,123,999}.json`:

| Baseline Reference | Mean Weight $\bar{w}_b$ | Std Dev | Min Weight | Max Weight |
|---|:---:|:---:|:---:|:---:|
| **MASK** | **0.8998** | 0.0721 | 0.6467 | 0.9467 |
| **PAD** | **0.0651** | 0.0425 | 0.0358 | 0.2187 |
| **UNK** | **0.0207** | 0.0207 | 0.0095 | 0.0928 |
| **ZERO** | **0.0143** | 0.0095 | 0.0080 | 0.0456 |

- **Weight Entropy:** Mean entropy is **$0.3914 \pm 0.1816$ nats** (compared to maximum uniform entropy of $\ln(4) \approx 1.3863$ nats).
- **Collapse Verdict:** With $\tau = 0.1$, **NP-IG virtually collapsed into single-baseline IG-MASK**, dedicating $\sim 90\%$ of the attribution weight to MASK.
- **Why?** For DistilBERT on SST-2, the predicted class probability at the `[MASK]` baseline token embedding is consistently close to $0.5$ (mean $|p - 0.5| \approx 0.048$). Meanwhile, `[PAD]` has $|p - 0.5| \approx 0.332$, `[UNK]` has $\approx 0.447$, and `ZERO` has $\approx 0.486$. When exponentiated with $\tau = 0.1$, $\exp(-0.048/0.1) \approx 0.619$, while $\exp(-0.332/0.1) \approx 0.036$, creating an extreme winner-take-all collapse.
- **Consequence:** NP-IG's apparent "robustness" was essentially just the performance of IG-MASK with minor smoothing, rather than a balanced multi-baseline ensemble. This directly motivates the temperature sweep ablation in Step 5.

---

## 3. Discrepancy (c): Grad $\times$ Input Failure ($\approx 0.006$)

### Root Cause Analysis
In the evaluation, `GRAD_X_INP` scored $0.0064$ on PAD, $0.0180$ on MASK, $0.0089$ on UNK, and $0.0153$ on DELETE. Because random attribution scores 0.0 by construction under random normalization, Grad $\times$ Input performed no better than random noise.

Two bugs/design flaws in `project/src/ig.py` and `project/src/erasure.py` caused this:
1. **Softmax Probability Gradient Vanishing (The Primary Cause):**
   In `compute_grad_x_input`:
   ```python
   if output_space == "prob":
       score = F.softmax(logits, dim=-1)[0, target_class]
   grads = torch.autograd.grad(score, input_embeds)[0]
   ```
   For DistilBERT on SST-2, model predictions are highly confident (often $p > 0.999$). The gradient of the softmax probability satisfies:
   $$\frac{\partial p_c}{\partial x} = p_c(1 - p_c) \frac{\partial \text{logit}_c}{\partial x}$$
   When $p_c \approx 0.9998$, the scaling factor $p_c(1 - p_c) \approx 0.0002$. The input gradient vanishes completely due to softmax saturation!
   *Why did IG not suffer from this?* Integrated Gradients integrates the path from baseline $\mathbf{b}$ (where $p \approx 0.5$, so $p(1-p) \approx 0.25$) to $\mathbf{x}$. The vast majority of IG's attribution is accumulated near $\alpha \approx 0$, thereby avoiding the endpoint saturation. Grad $\times$ Input evaluates only at $\alpha = 1$, directly hitting the saturated region.
2. **Signed vs. Absolute Ranking:**
   In `src/erasure.py`:
   `sorted_candidates = sorted(candidate_indices, key=lambda idx: attributions[idx].item(), reverse=True)`
   Token attributions were ranked purely by signed value rather than absolute magnitude ($|a_i|$) or L2 norm ($\|x_i \odot \nabla_i\|_2$). While signed ranking is theoretically aligned with directional class support, when gradients are vanished into numerical precision noise ($\sim 10^{-6}$), signed ranking effectively permutes tokens randomly.

### Resolution for Revision
In Step 2/Step 4 reruns, we evaluate Grad $\times$ Input in logit space (unnormalized score) and test absolute salience ranking to report a proper, non-pathological Grad $\times$ Input reference.

---

## 4. Discrepancy (d): Documentation & Implementation Specifics

| Specification | Documented Reality in Code |
|---|---|
| **Quadrature Type** | Standard Left/Midpoint **Riemann sum** with uniform step size $\Delta \alpha = 1/M$: `alphas = torch.linspace(0.0, 1.0, steps)`. Not Gauss-Legendre. |
| **"Steps in Probability Space"** | The integrand is $\frac{\partial}{\partial x} \text{Softmax}(\text{logits})_c$, i.e., the gradient of the predicted probability $f_c(x) \in [0, 1]$, decomposing $p(x) - p(b)$. |
| **PAD Erasure & Attention Mask** | In `src/erasure.py` (lines 66-70), when a content token is replaced by `pad_id`, its entry in `attention_mask` is explicitly set to **0** (`erased_mask[0, idx] = 0`). Thus, the self-attention mechanism masks out the padded token during evaluation (unlike MASK and UNK erasure, where `attention_mask` remains 1). |
| **Top-$k$ Rounding for Short Inputs** | Computed as `max(1, int(round(len(candidate_indices) * fraction)))`. For a 4-token sentence, fractions $0.10, 0.20, 0.30$ all round to 1 token ($k=1$), and $0.50$ selects $k=2$. |
| **Kendall $\tau$ Number of Methods** | The draft stated "7 methods", but the code ranked **8 methods** (`UNIFORM_IG`, `IG_UNK`, `NP_IG`, `IG_MASK`, `IG_PAD`, `IG_ZERO`, `GRAD_X_INP`, `RANDOM`). For $n=8$, total pairs $\binom{8}{2} = 28$. A swap of 3 pairs yields $\tau = (28 - 2 \times 3)/28 = 22/28 = 11/14 \approx \mathbf{0.7857}$, exactly explaining the reported $0.786$. The prose misstated $n=7$. |

---

## 5. Discrepancy (e): Reference Verification Audit

Every reference in `main.bib` was queried against the official ACL Anthology and DOI registries.

| Reference Key | Status | Issue Identified | Correction Made |
|---|:---:|---|---|
| `bastings2022shortcuts` | ❌ **ERRONEOUS** | Cited as ACL 2022, final author Umang Bhatt. In reality: published at **EMNLP 2022** (pp. 976–991), final author is **Katja Filippova** (`2022.emnlp-main.64`). | **CORRECTED** to EMNLP 2022, author list updated: Bastings, Ebert, Zablotskaia, Sandholm, and Filippova. |
| `hentschel2024canonical` | ❌ **UNVERIFIED / FABRICATED** | Hallucinated author "Simon Hentschel" at ACL 2024 (`2024.acl-long.26`, which is an unrelated paper). In reality, Ludger Hentschel wrote an unrefereed working paper in Sept 2026. | **REMOVED ENTIRELY** from the paper and bibliography. |
| `liu2021shapley` | ❌ **ERRONEOUS** | Fabricated as EMNLP 2021 with fictitious authors. The actual paper is Shuyang Liu et al. (2023), *"A New Baseline Assumption of Integrated Gradients Based on Shapley Value"*, arXiv:2310.04821. | **CORRECTED** to arXiv:2310.04821 with correct author list. |
| `jacovi2020towards` | ⚠️ **MINOR TITLE ERROR** | Subtitle was misquoted as *"On the Concept of Explanations"*. Official title is *"Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?"* (ACL 2020, `2020.acl-main.386`). | **CORRECTED** title in BibTeX. |
| `enguehard2023sequential` | ⚠️ **ID TYPO** | Anthology ID cited as `findings-acl.476`. Correct ID is `findings-acl.477`. | **CORRECTED** URL and anthology ID. |
| `sundararajan2017axiomatic` | ✅ **VERIFIED** | ICML 2017 (`v70/sundararajan17a`). | Retained. |
| `deyoung2020eraser` | ✅ **VERIFIED** | ACL 2020 (`2020.acl-main.408`). | Retained. |
| `chrysostomou2022empirical`| ✅ **VERIFIED** | ACL 2022 (`2022.acl-long.477`). | Retained. |
| `sanyal2021discretized` | ✅ **VERIFIED** | EMNLP 2021 (`2021.emnlp-main.805`). | Retained. |
| `erion2021improving` | ✅ **VERIFIED** | Nature Machine Intelligence 2021 (DOI: 10.1038/s42256-021-00343-w). | Retained. |
| `wang2018glue` | ✅ **VERIFIED** | BlackboxNLP at EMNLP 2018 (`W18-5446`). | Retained. |
| `sanh2019distilbert` | ✅ **VERIFIED** | NeurIPS 2019 Workshop (`arXiv:1910.01108`). | Retained. |
| `maas2011learning` | ✅ **VERIFIED** | ACL 2011 (`P11-1015`). | Retained. |

---

## 6. Summary of Protocol for Revision Steps 2–6

1. **Step 2 (Higher Resolution Rerun):**
   - Rerun all methods at $M=50$ steps on SST-2 (Seeds 42, 123, 999; $N=450$).
   - Run $M=100$ on 50-example subset; compute per-example Spearman rank correlation of token attributions between $M=20, 50, 100$.
   - Introduce **`ZERO_EMB` erasure scheme** (token embeddings zeroed out, retaining position and mask) to give `IG_ZERO` a true matched pair, while keeping `DELETE` as an unmatched baseline.
   - Save all per-example raw scores to tidy CSV files.
2. **Step 3 (Linear Mixed Model for H1):**
   - Fit `norm_comp ~ baseline + scheme + matched_indicator + (1 | example)`.
   - Compute matched indicator coefficient, 95% CI, and per-baseline matched contrasts with Holm correction.
   - Bin SST-2 by sentence length ($L \le 10$, $10 < L \le 20$, $L > 20$) and evaluate interaction.
3. **Step 4 (Held-Out Erasure):**
   - Implement Random Vocabulary token erasure (5 seeded draws averaged).
   - Implement Masked Language Model (MLM) infill erasure (using DistilBERT masked LM, top-k candidate).
   - Evaluate if Uniform-IG and NP-IG remain robust when erasure token is outside their reference set.
4. **Step 5 (NP-IG Ablation):**
   - Sweep $\tau \in \{0.01, 0.03, 0.1, 0.3, 1.0, \infty\}$.
   - Include argmin-neutrality single baseline, Uniform-IG, and shuffled random weights control.
   - Report weight entropy.
5. **Step 6 (Paper Rewrite):**
   - Reframe paper around the empirical interaction, honestly reporting effect size and LMM variance attribution.
