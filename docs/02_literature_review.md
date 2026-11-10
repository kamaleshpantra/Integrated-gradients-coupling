# Literature Review: Baseline Selection and Faithfulness Evaluation in Integrated Gradients

**Last Updated:** 2026-10-29  
**Scope:** Feature attribution methods for NLP, baseline selection, faithfulness evaluation, erasure artifacts

---

## 1. Axiomatic Attribution Methods

### 1.1 Integrated Gradients (Sundararajan et al., 2017)

Integrated Gradients (IG) is the foundational path-attribution method satisfying two axioms:
- **Completeness**: $\sum_i \text{IG}_i(\mathbf{x}, \mathbf{b}) = f(\mathbf{x}) - f(\mathbf{b})$
- **Sensitivity**: If feature $i$ differs between $\mathbf{x}$ and $\mathbf{b}$ and this changes the output, then $\text{IG}_i \neq 0$

The attribution is computed as:
$$\text{IG}_i(\mathbf{x}, \mathbf{b}) = (x_i - b_i) \int_0^1 \frac{\partial f(\mathbf{b} + \alpha(\mathbf{x} - \mathbf{b}))}{\partial x_i} d\alpha$$

**Key limitation in NLP**: The straight-line interpolation path between discrete token embeddings passes through regions of embedding space that do not correspond to any real word — the "off-manifold problem."

**Relevance to our work**: The baseline $\mathbf{b}$ determines both what IG explains ($f(\mathbf{x}) - f(\mathbf{b})$) and which region of embedding space the path traverses.

### 1.2 Discretized Integrated Gradients (Sanyal & Ren, 2021)

Proposed replacing the straight-line path with discrete word-to-word paths through the vocabulary, staying on the token embedding manifold. Evaluated on SST-2 and IMDB with DistilBERT.

**Relevance**: Motivates concern about path-dependent artifacts in standard IG.

### 1.3 Sequential Integrated Gradients (Enguehard, 2023)

Integrates along token-by-token insertion paths rather than a single straight line. Published in Findings of ACL 2023.

**Relevance**: Alternative path choice that avoids the off-manifold interpolation problem.

### 1.4 Expected Gradients (Erion et al., 2021)

Averages IG attributions over baselines drawn from the empirical data distribution:
$$\text{EG}_i(\mathbf{x}) = \mathbb{E}_{\mathbf{b} \sim \mathcal{D}} [\text{IG}_i(\mathbf{x}, \mathbf{b})]$$

Published in *Nature Machine Intelligence*. Connects IG to Shapley values when using the data distribution as reference.

**Relevance**: Directly motivates our Uniform-IG and NP-IG multi-baseline approaches, though Expected Gradients samples from data rather than from special tokens.

### 1.5 Shapley-Based Baselines (Liu et al., 2023)

Connected multi-baseline path integration to Shapley value coalitions. ArXiv preprint 2310.04821.

**Relevance**: Theoretical justification for averaging over multiple baselines.

---

## 2. Faithfulness Evaluation Methods

### 2.1 ERASER Benchmark (DeYoung et al., 2020)

Standardized two key metrics for evaluating attribution quality:

- **Comprehensiveness**: Prediction drop when top-$k$ attributed tokens are removed.
  $$\text{Comp}(k) = f_c(\mathbf{x}) - f_c(\text{erase}(\mathbf{x}, S_k))$$
  Higher = more faithful (removing important tokens should reduce confidence).

- **Sufficiency**: Prediction retention when only top-$k$ tokens are kept.
  $$\text{Suff}(k) = f_c(\mathbf{x}) - f_c(\text{keep}(\mathbf{x}, S_k))$$
  Lower = more faithful (important tokens alone should preserve the prediction).

Published at ACL 2020. `2020.acl-main.408`.

**Relevance**: We use normalized comprehensiveness AOPC as our primary faithfulness metric.

### 2.2 Erasure Artifacts (Bastings et al., 2022)

Demonstrated that token erasure creates out-of-distribution inputs that can activate spurious model shortcuts. Models may react to the *presence of erasure tokens* rather than the *absence of content*.

Published at EMNLP 2022. Authors: Bastings, Ebert, Zablotskaia, Sandholm, Filippova. `2022.emnlp-main.64`.

**Relevance**: Directly motivates our investigation — if erasure tokens create artifacts, then matching the erasure token to the IG baseline may compound the artifact.

### 2.3 Out-of-Domain Faithfulness (Chrysostomou & Aletras, 2022)

Showed that faithfulness metrics can paradoxically increase when models are evaluated on out-of-domain data, questioning the reliability of erasure-based evaluation.

Published at ACL 2022. `2022.acl-long.477`.

**Relevance**: Further evidence that erasure-based evaluation can produce misleading faithfulness estimates.

---

## 3. Baseline Selection in NLP

### 3.1 Common Choices and Their Rationale

| Baseline | Rationale | Known Issues |
|---|---|---|
| `[PAD]` (token ID 0) | "Empty" position in BERT's vocabulary | Model trained to ignore PAD via attention masking; may not represent true "absence" |
| `[MASK]` (token ID 103) | Designed for masked language modeling; represents "unknown token" | Model actively predicts content for MASK tokens; not truly uninformative |
| `[UNK]` (token ID 100) | Represents out-of-vocabulary tokens | Rarely seen in fine-tuned models; may activate unusual model behavior |
| Zero embedding ($\mathbf{0}$) | Mathematically natural "origin" in embedding space | Not a valid token embedding; lies off the token manifold |

### 3.2 The Baseline Selection Problem

No consensus exists on which baseline to use for IG in NLP. Different baselines produce different attribution rankings (our pilot study confirmed Kendall $\tau \approx 0.55$–$0.85$ between baseline pairs), yet practitioners rarely justify their choice or evaluate sensitivity to it.

---

## 4. Gap in the Literature

| What is known | What is NOT known |
|---|---|
| IG's completeness axiom ties attributions to $f(\mathbf{x}) - f(\mathbf{b})$ | Whether matching $\mathbf{b}$ in evaluation inflates comprehensiveness |
| Erasure-based metrics can create OOD artifacts | Whether the baseline-erasure interaction is a systematic confound |
| Multiple baselines exist with different properties | Whether ensembling baselines provides robustness against evaluation artifacts |
| Input length affects model behavior | Whether coupling strength depends on sequence length |

**Our contribution fills this gap** with the first systematic factorial study of baseline × erasure interaction.

---

## 5. Key Papers Quick-Reference Table

| Paper | Venue | Year | Key Contribution | Verified |
|---|---|:---:|---|:---:|
| Sundararajan et al. | ICML | 2017 | Introduced IG with axioms | ✓ |
| DeYoung et al. | ACL | 2020 | ERASER benchmark, comprehensiveness/sufficiency | ✓ |
| Jacovi & Goldberg | ACL | 2020 | Formal framework for faithfulness | ✓ |
| Sanyal & Ren | EMNLP | 2021 | Discretized IG for NLP | ✓ |
| Erion et al. | Nat. Mach. Intell. | 2021 | Expected Gradients | ✓ |
| Bastings et al. | EMNLP | 2022 | Erasure artifacts and shortcuts | ✓ |
| Chrysostomou & Aletras | ACL | 2022 | OOD faithfulness paradox | ✓ |
| Enguehard | Findings ACL | 2023 | Sequential IG | ✓ |
| Liu et al. | arXiv | 2023 | Shapley-based IG baselines | ✓ |
| Wang et al. | BlackboxNLP | 2018 | GLUE benchmark (includes SST-2) | ✓ |
| Sanh et al. | NeurIPS Workshop | 2019 | DistilBERT | ✓ |
| Maas et al. | ACL | 2011 | IMDB dataset | ✓ |

> **Note:** All references have been verified against the ACL Anthology or DOI registries. See `project/AUDIT.md` §5 and `project/references_verified.md` for the full audit trail.
