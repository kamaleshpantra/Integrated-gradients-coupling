# Detailed Methodology & Implementation Notes

**Purpose:** This document provides implementation-level detail that goes beyond the paper's methodology section. It serves as a reference for reproducibility auditors, advisor review, and the student's own future self.

---

## 1. Integrated Gradients Implementation

### 1.1 Quadrature Method
- **Type:** Left-endpoint Riemann sum with uniform step size
- **Steps:** `alphas = torch.linspace(0.0, 1.0, M)` where $M$ is the number of steps
- **Step size:** $\Delta\alpha = 1/M$
- **NOT** Gauss-Legendre or trapezoidal rule

### 1.2 Integration Space
- Gradients are computed with respect to the **probability output** $p_c = \text{softmax}(\text{logits})_c$
- This means IG decomposes $p_c(\mathbf{x}) - p_c(\mathbf{b})$, which is bounded in $[-1, 1]$
- At each interpolation point $\alpha$, we compute: $\text{input}_\alpha = \mathbf{b} + \alpha(\mathbf{x} - \mathbf{b})$

### 1.3 Completeness Axiom Verification
The completeness delta is: $\delta = |f(\mathbf{x}) - f(\mathbf{b}) - \sum_i \text{IG}_i|$

| $M$ | PAD $\delta$ | MASK $\delta$ | UNK $\delta$ | ZERO $\delta$ |
|:---:|:---:|:---:|:---:|:---:|
| 20 | 0.1937 | 0.2180 | 0.2495 | 0.3941 |
| 50 | 0.0284 | 0.0232 | 0.0709 | 0.1089 |
| 100 | 0.0048 | 0.0050 | 0.0130 | 0.0395 |

### 1.4 Grad × Input (Logit-Space Fix)
- **Original (buggy):** Computed gradients of softmax probability → vanished due to saturation ($p(1-p) \approx 0.0002$)
- **Fixed:** Compute gradients of unnormalized logit: $\text{score} = \text{logits}[0, c]$ (no softmax)
- **Ranking:** Absolute value of attributions, descending order

---

## 2. Erasure Operators

### 2.1 Token-Level Erasure

| Scheme | Implementation | `attention_mask` |
|---|---|:---:|
| **PAD** | Replace `input_ids[idx]` with `tokenizer.pad_token_id` (0) | Set to **0** |
| **MASK** | Replace `input_ids[idx]` with `tokenizer.mask_token_id` (103) | Kept at **1** |
| **UNK** | Replace `input_ids[idx]` with `tokenizer.unk_token_id` (100) | Kept at **1** |
| **DELETE** | Physically remove token from `input_ids` and `attention_mask` tensors | Removed |

### 2.2 Embedding-Level Erasure

| Scheme | Implementation |
|---|---|
| **ZERO_EMB** | Run model's embedding layer, then zero out the embedding vector at erased positions. Pass modified embeddings directly to the transformer encoder. `attention_mask` kept at **1**. |

### 2.3 Held-Out Erasure (Not Used in IG Baselines)

| Scheme | Implementation |
|---|---|
| **RANDOM_TOKEN** | Replace with a random vocabulary ID drawn uniformly from $[0, 30521]$. Average comprehensiveness over 5 independent draws (sub-seeds 0–4). |
| **MLM_INFILL** | Replace token with `[MASK]`, run DistilBERT MLM head (`D:/models/distilbert_mlm`), take the top-1 predicted token ID, replace `[MASK]` with that prediction. |

### 2.4 Special Token Handling
- `[CLS]` (position 0) and `[SEP]` (last position) are **excluded from both attribution and erasure**.
- `candidate_indices` = all positions between CLS and SEP (exclusive).

### 2.5 Top-k Rounding
```python
k = max(1, int(round(len(candidate_indices) * fraction)))
```
For a 4-token sentence: fractions 0.10, 0.20, 0.30 → $k=1$; fraction 0.50 → $k=2$.

---

## 3. Multi-Baseline Methods

### 3.1 Uniform-IG
$$\text{Uniform-IG}(\mathbf{x}) = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \text{IG}(\mathbf{x}, b)$$
where $\mathcal{B} = \{\text{PAD}, \text{MASK}, \text{UNK}, \text{ZERO}\}$. Equal weights, no hyperparameters.

### 3.2 NP-IG (Neutrality-Prediction Integrated Gradients)
$$w_b \propto \exp\left(-\frac{|p_c(b) - 0.5|}{\tau}\right)$$
where $p_c(b)$ is the model's predicted probability for the true class at baseline $b$, and $\tau$ is a temperature parameter.

$$\text{NP-IG}(\mathbf{x}) = \sum_{b \in \mathcal{B}} w_b \cdot \text{IG}(\mathbf{x}, b)$$

**Motivation:** Baselines closer to the decision boundary ($p \approx 0.5$) are considered more "neutral" and given higher weight.

**Problem found:** For DistilBERT on SST-2:
- MASK: $|p - 0.5| \approx 0.048$ → $\exp(-0.048/0.1) = 0.619$
- PAD: $|p - 0.5| \approx 0.332$ → $\exp(-0.332/0.1) = 0.036$
- UNK: $|p - 0.5| \approx 0.447$ → $\exp(-0.447/0.1) = 0.011$
- ZERO: $|p - 0.5| \approx 0.486$ → $\exp(-0.486/0.1) = 0.008$

After normalization, MASK gets ~90% of the weight → NP-IG ≈ IG-MASK.

---

## 4. Evaluation Metric Details

### 4.1 Comprehensiveness AOPC

For each example $i$, method $m$, and erasure scheme $\sigma$:

1. Compute attributions $\mathbf{a}^m_i$
2. For each fraction $f \in \{0.10, 0.20, 0.30, 0.50\}$:
   - Select top-$k$ tokens where $k = \max(1, \text{round}(T \cdot f))$
   - Erase them using scheme $\sigma$
   - Compute $\text{Comp}_{i,f} = f_c(\mathbf{x}_i) - f_c(\text{erase}(\mathbf{x}_i, S_k, \sigma))$
3. AOPC = mean across fractions: $\text{AOPC}_i = \frac{1}{4}\sum_f \text{Comp}_{i,f}$
4. Normalize: $\text{NormComp}_i = \text{AOPC}^{\text{method}}_i - \text{AOPC}^{\text{random}}_i$

### 4.2 Worst-Case Robustness

**Definition 1 (Mean of per-example worst):**
$$\text{Worst}_{\text{ex}}(m) = \frac{1}{N}\sum_{i=1}^N \min_{\sigma \in \Sigma} \text{NormComp}_i(m, \sigma)$$

**Definition 2 (Min of per-scheme means) — PRIMARY:**
$$\text{Worst}_{\text{agg}}(m) = \min_{\sigma \in \Sigma} \left(\frac{1}{N}\sum_{i=1}^N \text{NormComp}_i(m, \sigma)\right)$$

By Jensen's inequality: Def 1 $\leq$ Def 2 always.

---

## 5. Statistical Analysis Plan

### 5.1 Linear Mixed-Effects Model

**Software:** Python `statsmodels.regression.mixed_linear_model.MixedLM`

**Model:**
```
norm_comp_aopc ~ C(baseline) + C(scheme) + is_matched + (1 | unique_id)
```

**Variables:**
- `baseline` ∈ {PAD, MASK, UNK, ZERO} — IG baseline used
- `scheme` ∈ {PAD, MASK, UNK, ZERO_EMB} — erasure scheme (in-domain 4×4 only)
- `is_matched` ∈ {0, 1} — 1 if baseline and scheme are matched
- `unique_id` = `{seed}_{example_id}` — random intercept grouping variable
- Response: `norm_comp_aopc` — normalized comprehensiveness AOPC

**Data subset:** Only in-domain 4×4 grid rows (baselines × in-domain schemes), single-baseline IG methods only. $N = 450 \times 4 \times 4 = 7200$ rows.

### 5.2 Per-Baseline Matched Contrasts

For each baseline $b$:
- Diagonal mean = $\mathbb{E}_i[\text{NormComp}(b, \sigma=b)]$
- Off-diagonal mean = $\mathbb{E}_i[\frac{1}{3}\sum_{\sigma \neq b} \text{NormComp}(b, \sigma)]$
- Test: paired $t$-test on per-example differences
- Correction: Holm-Bonferroni over 4 comparisons

### 5.3 ANOVA Variance Decomposition

Sequential (Type I) sums of squares from OLS:
```
norm_comp_aopc ~ C(baseline) + C(scheme) + is_matched
```
Report each factor's SS as percentage of total SS.

### 5.4 Length-Binned Subanalysis

Bin examples by content token count (excluding [CLS], [SEP]):
- **Short:** $\leq 10$ tokens
- **Medium:** 11–20 tokens
- **Long:** $> 20$ tokens

Fit separate LMMs per bin. Report $\beta_{\text{matched}}$ and 95% CI per bin.

---

## 6. Random Seeds & Reproducibility

| Parameter | Value |
|---|---|
| Main experiment seeds | 42, 123, 999 |
| Random-token sub-seeds | 0, 1, 2, 3, 4 (5 draws, averaged) |
| PyTorch deterministic | `torch.manual_seed(seed)` before each example |
| NumPy deterministic | `np.random.seed(seed)` |
| CUDA deterministic | `torch.cuda.manual_seed_all(seed)` |
| Hardware | CPU-only (DistilBERT fits without GPU) |
