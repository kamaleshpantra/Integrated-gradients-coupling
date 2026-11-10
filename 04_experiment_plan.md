# Experiment Plan

## 1. Mathematical Definitions and Framework

### 1.1 Integrated Gradients in NLP
Let $x = (t_1, \dots, t_N)$ denote an input sequence of tokens with embedding vectors $E(x) = (e_1, \dots, e_N) \in \mathbb{R}^{N \times d}$.
Let $f(x)$ be the model prediction for the target class (either predicted class probability $P(y|x) \in [0, 1]$ or logit $z_y(x)$).
Given a reference baseline $x' \in \mathcal{B}$ with embeddings $E(x')$, the Integrated Gradients (Sundararajan et al., 2017) attribution for token $i \in \{1, \dots, N\}$ is:
$$\text{IG}_i(x, x') = (E(x)_i - E(x')_i) \cdot \int_{0}^1 \frac{\partial f(E(x') + \alpha (E(x) - E(x')))}{\partial E(x)_i} d\alpha$$
Using a Riemann sum with $M$ discrete steps $\alpha_m = \frac{m}{M}$ ($m \in \{1, \dots, M\}$):
$$\text{IG}_i(x, x') \approx (E(x)_i - E(x')_i) \cdot \frac{1}{M} \sum_{m=1}^M \frac{\partial f\left(E(x') + \frac{m}{M}(E(x) - E(x'))\right)}{\partial E(x)_i}$$
Token scalar attribution is computed by taking the Euclidean inner product or $L_1$ sum across embedding dimensions.

### 1.2 Baseline Candidates $\mathcal{B}$
We evaluate four standard NLP baselines:
1. $x^{(\text{PAD})}$: All input tokens replaced with `[PAD]`.
2. $x^{(\text{MASK})}$: All input tokens replaced with `[MASK]`.
3. $x^{(\text{UNK})}$: All input tokens replaced with `[UNK]`.
4. $x^{(\text{ZERO})}$: Token embeddings initialized to the all-zero vector $\mathbf{0} \in \mathbb{R}^{N \times d}$.

### 1.3 Neutral-Prediction Weighted Integrated Gradients (NP-IG)
A baseline should ideally represent the uninformative state, corresponding to neutral probability $p^* = 0.5$ in binary classification.
For each candidate baseline $b \in \mathcal{B}$, let $f(x^{(b)})$ denote the predicted target probability.
We compute the absolute deviation from neutrality:
$$d_b = |f(x^{(b)}) - 0.5|$$
The baseline weight $w_b$ is defined using a softmax over negative distances scaled by temperature $\tau > 0$:
$$w_b = \frac{\exp(-d_b / \tau)}{\sum_{b' \in \mathcal{B}} \exp(-d_{b'} / \tau)}, \quad \sum_{b \in \mathcal{B}} w_b = 1$$
The NP-IG attribution vector is the convex combination:
$$\text{NP-IG}_i(x) = \sum_{b \in \mathcal{B}} w_b \text{IG}_i(x, x^{(b)})$$

#### Completeness Guarantee
By linearity:
$$\sum_{i=1}^N \text{NP-IG}_i(x) = \sum_{b \in \mathcal{B}} w_b \sum_{i=1}^N \text{IG}_i(x, x^{(b)}) \approx f(x) - \sum_{b \in \mathcal{B}} w_b f(x^{(b)}) = f(x) - \bar{f}_0$$
where $\bar{f}_0 = \sum_{b \in \mathcal{B}} w_b f(x^{(b)})$ is the expected baseline prediction. We record the numerical completeness delta $\Delta = |\sum_i \text{NP-IG}_i - (f(x) - \bar{f}_0)|$ across all examples.

---

## 2. Evaluation Erasure Schemes $\mathcal{E}$

Given token attribution scores, tokens are ranked in descending order of importance. For a top fraction $k \in \{0.10, 0.20, 0.30, 0.50\}$, we apply four distinct erasure operators:
1. **PAD Erasure**: Replace top $k\%$ tokens with `[PAD]`.
2. **MASK Erasure**: Replace top $k\%$ tokens with `[MASK]`.
3. **UNK Erasure**: Replace top $k\%$ tokens with `[UNK]`.
4. **Delete Erasure**: Physically excise top $k\%$ tokens from the sequence, re-indexing position embeddings.

### Metrics
1. **Comprehensiveness (Comp)** (DeYoung et al., 2020):
   $$\text{Comp}(k) = f(x) - f(x_{\backslash \text{top-}k})$$
   Higher indicates more faithful identification of influential features.
2. **Sufficiency (Suff)** (DeYoung et al., 2020):
   $$\text{Suff}(k) = f(x) - f(x_{\text{top-}k})$$
   Lower indicates that the isolated subset preserves the prediction.
3. **Area Over the Perturbation Curve (AOPC)** (Samek et al., 2016):
   $$\text{AOPC} = \frac{1}{|K|} \sum_{k \in K} \text{Comp}(k)$$
4. **Random-Normalized Metrics** (Chrysostomou & Aletras, 2022):
   $$\Delta \text{Comp} = \text{Comp}_{\text{method}} - \text{Comp}_{\text{random}}$$
   $$\Delta \text{Suff} = \text{Suff}_{\text{random}} - \text{Suff}_{\text{method}}$$
   Subtracts the baseline performance of a random-attribution null control to prevent distribution-shift conflation.

---

## 3. Models and Datasets

1. **Models**:
   - Primary: `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face Hub).
   - Secondary: `textattack/bert-base-uncased-SST-2` (Hugging Face Hub).
2. **Datasets**:
   - **SST-2 (Stanford Sentiment Treebank)**: Validation split (872 sentences), standard benchmark in NLP interpretability.
   - **IMDB Sentiment Subset**: 500 test reviews randomly sampled using fixed random seed 42 to assess generalization to longer sequences.
3. **Reference Methods**:
   - Gradient $\times$ Input ($x \odot \nabla_x f(x)$).
   - Random Attribution (null control).

---

## 4. Full Experimental Matrix and Ablations

### Primary Matrix
- Attribution Methods:
  1. IG with PAD baseline
  2. IG with MASK baseline
  3. IG with UNK baseline
  4. IG with ZERO baseline
  5. NP-IG (Neutral-Prediction Weighted IG, $\tau = 0.1$)
  6. Uniform-IG (Uniformly Averaged IG across all 4 baselines)
  7. Gradient $\times$ Input
  8. Random Attribution
- Cross-Evaluated On:
  1. PAD Erasure
  2. MASK Erasure
  3. UNK Erasure
  4. Delete Erasure
- At Token Fractions: 10%, 20%, 30%, 50%.

### Planned Ablations
1. **Interpolation Steps $M$**: $M \in \{20, 50, 100\}$.
2. **Attribution Output Space**: Class Probability ($P(y|x)$) vs. Pre-softmax Logits ($z_y(x)$).
3. **NP-IG Temperature $\tau$**: $\tau \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$.
4. **Ensemble Size $K$**: 2 baselines ({PAD, MASK}), 3 baselines ({PAD, MASK, UNK}), 4 baselines ({PAD, MASK, UNK, ZERO}).

---

## 5. Statistical Rigor Protocol

1. **Random Seeds**: 3 seeds (`42`, `123`, `999`) for all stochastic operations (random control, sampling).
2. **Confidence Intervals**: 95% bootstrap confidence intervals computed via 1,000 resamples.
3. **Hypothesis Testing**:
   - **H1 Test**: Two-sided paired Wilcoxon signed-rank test and paired bootstrap test comparing on-diagonal ($b = e$) vs off-diagonal ($b \neq e$) normalized Comprehensiveness and Sufficiency.
   - **H2 Test**: Paired bootstrap comparison of NP-IG vs best single baseline on worst-case erasure performance ($\min_{e} \text{Comp}(m, e)$).
4. **Ranking Consistency**: Kendall rank correlation coefficient ($\tau_K$) computed between method rankings under each pairwise combination of erasure schemes.
