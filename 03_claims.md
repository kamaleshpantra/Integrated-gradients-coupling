# Gaps, Claims, and Hypotheses

## 1. Identified Research Gaps

### Gap 1: Unexamined Baseline-Evaluation Coupling
In NLP interpretability literature, Integrated Gradients (IG; Sundararajan et al., 2017) requires choosing a reference baseline token (e.g., `[PAD]`, `[MASK]`, `[UNK]`, or an all-zero embedding). Meanwhile, post-hoc faithfulness benchmarks—such as Comprehensiveness, Sufficiency (ERASER; DeYoung et al., 2020), and AOPC (Samek et al., 2016)—evaluate explanations by erasing top-attributed tokens. Evaluators standardly instantiate this erasure operator using a single arbitrary choice, such as masking or padding.
However, no prior study has investigated what happens when the baseline token chosen for attribution matches the token chosen for evaluation erasure. If an attribution method computes gradients with respect to `[MASK]` and is evaluated by substituting words with `[MASK]`, the evaluation metric measures perturbation toward the exact reference point that defines the attribution, creating an unexamined circular dependency (*Baseline-Evaluation Coupling*).

### Gap 2: Sub-optimal Neutrality in Text Baselines
Integrated Gradients theoretically assumes the baseline represents the "absence" of information, which in binary classification corresponds to a neutral prediction probability ($P(y|\text{baseline}) \approx 0.5$). While Canonical Integrated Gradients (Hentschel, 2024) explored neutral prediction manifolds in continuous domains, no existing work has investigated the prediction neutrality of standard text token baselines (`[PAD]`, `[MASK]`, `[UNK]`, zero-vector) in fine-tuned transformer classifiers, nor leveraged neutrality to construct an ensemble of discrete baselines.

---

## 2. Formal Claims

- **Claim 1 (Baseline-Evaluation Coupling / Diagonal Inflation)**:
  Faithfulness metrics (Comprehensiveness, Sufficiency, AOPC) for Integrated Gradients on transformer text classifiers are inflated along the diagonal of the baseline-by-erasure matrix—that is, when the attribution baseline token matches the token used by the erasure metric ($b = e$).

- **Claim 2 (Rank Instability Under Cross-Evaluation)**:
  The ranking of attribution baselines is unstable across different erasure schemes. A baseline that appears superior when evaluated using its matching token falls in rank when evaluated using non-matching or token-deletion erasure operators.

- **Claim 3 (Non-Neutrality of Single Baselines)**:
  Individual static text baselines (`[PAD]`, `[MASK]`, `[UNK]`, zero-embedding) do not yield neutral model predictions ($P(y|\text{baseline}) \neq 0.5$) and exhibit significant asymmetric class bias in transformer classifiers.

- **Claim 4 (Robustness of Neutral-Prediction Weighted Integrated Gradients - NP-IG)**:
  An ensemble of baselines weighted by the proximity of each baseline's prediction to the neutral decision threshold ($p = 0.5$) significantly mitigates baseline-evaluation coupling, achieving superior worst-case faithfulness and more consistent rankings across diverse erasure schemes than any single baseline.

---

## 3. Hypotheses and Falsification Conditions

### Hypothesis 1 (H1: Diagonal Inflation in Baseline-by-Erasure Matrix)
For any single-token baseline $b \in \{\text{PAD}, \text{MASK}, \text{UNK}\}$ and corresponding erasure scheme $e \in \{\text{PAD}, \text{MASK}, \text{UNK}\}$, the mean Comprehensiveness score on the diagonal ($b = e$) will be strictly and statistically significantly greater than the mean Comprehensiveness on the off-diagonal ($b \neq e$):
$$\mathbb{E}[\text{Comp}(b, e = b)] > \mathbb{E}[\text{Comp}(b, e \neq b)] \quad \text{with } p < 0.05 \text{ (paired bootstrap test)}$$
Conversely, Sufficiency scores will be lower on the diagonal (indicating artificially enhanced faithfulness).

* **Falsification Condition for H1**:
  If the off-diagonal entries (e.g., MASK baseline evaluated under PAD or Deletion erasure) show equal or higher Comprehensiveness than the on-diagonal entry ($b = e$), or if the difference between on-diagonal and off-diagonal faithfulness is statistically indistinguishable from zero ($p \ge 0.05$ under a two-sided paired bootstrap test with 1,000 resamples), **H1 is falsified**.

### Hypothesis 2 (H2: NP-IG Superiority in Worst-Case and Mean-Rank Faithfulness)
Across the set of 4 evaluation erasure schemes $\mathcal{E} = \{\text{PAD}, \text{MASK}, \text{UNK}, \text{Delete}\}$, NP-IG will demonstrate greater robustness than any individual single baseline $b \in \{\text{PAD}, \text{MASK}, \text{UNK}, \text{Zero}\}$, satisfying:
1. **Worst-Case Superiority**:
   $$\min_{e \in \mathcal{E}} \text{Comp}(\text{NP-IG}, e) > \max_{b} \min_{e \in \mathcal{E}} \text{Comp}(b, e)$$
2. **Mean Rank Superiority**:
   NP-IG will achieve the best (lowest) average rank across the 4 erasure schemes compared to any single baseline.

* **Falsification Condition for H2**:
  If there exists any single baseline $b$ whose worst-case Comprehensiveness across $\mathcal{E}$ is greater than or equal to that of NP-IG, or if NP-IG's average rank across the erasure schemes is equal to or worse than a single baseline, **H2 is falsified**.

---

## 4. Prior Work Statement

> **To the best of our knowledge, based on searches of Semantic Scholar, arXiv, ACL Anthology, Google Scholar, OpenReview, and Papers with Code on 2026-10-25, we found no prior work that evaluates the full baseline-by-erasure cross-matrix in NLP feature attribution, identifies the diagonal baseline-evaluation coupling artifact, or proposes a neutral-prediction weighted multi-baseline Integrated Gradients formulation (NP-IG).**
