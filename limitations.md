# Limitations

This document lists known limitations of the study. It is intended to be included verbatim or adapted into the paper's Limitations section.

---

## 1. Single Model Architecture

All experiments use `distilbert-base-uncased-finetuned-sst-2-english`. DistilBERT is a 6-layer distilled transformer, smaller than BERT-base (12 layers) or BERT-large. Whether the diagonal inflation artifact generalizes to larger models (BERT-large, RoBERTa, GPT-style decoders) or other tasks (NLI, QA) is unknown. We recommend replication on at least one additional model.

## 2. CPU-Only Computation Constraint

Experiments were run on CPU (no GPU). This forced:
- IG steps M=20 for main runs (M=50 in ablation). Most published IG results use M=50–300.
- Sample sizes of N=150 per seed (SST-2) and N=100 (IMDB), rather than full validation sets.
- Single dataset (SST-2) as the primary evaluation; IMDB is a cross-dataset check only.

Completeness error at M=20 is ~0.25 (probability output space), which is non-trivial. However, since our claims are relative (diagonal vs. off-diagonal), not absolute, the inflation artifact is unlikely to be an artifact of step count. Confirmed by M=50 ablation (rankings stable).

## 3. Binary Sentiment Classification Only

Comprehensiveness and sufficiency are defined relative to the predicted class probability. We test only binary sentiment classification. Extension to multiclass or span prediction tasks is not tested.

## 4. NP-IG: Temperature is a Hyperparameter

NP-IG requires temperature τ. We set τ=0.1 as a default and ablate over τ ∈ {0.01, 0.05, 0.1, 0.5}. Results may depend on τ. The robustness of NP-IG relative to single baselines is stronger when τ is small (one dominant baseline weight) and weaker when τ is large (approaches Uniform-IG).

## 5. Out-of-Distribution Baselines

Zero-embedding and PAD baselines produce inputs outside the model's training distribution. This is a known issue with IG for discrete inputs [Hentschel 2024; Sanyal & Ren 2021]. NP-IG inherits this limitation: if a baseline is OOD, its weight may be non-representative of its semantic content.

## 6. Erasure Creates Distribution Shift

All comprehensiveness/sufficiency metrics suffer from the erasure-OOD problem [Chrysostomou & Aletras 2022; Bastings et al. 2022]. Our findings document a *coupling* between baseline choice and erasure scheme within this existing framework; we do not claim to solve the underlying OOD issue.

## 7. Generalized Completeness for NP-IG

Standard IG satisfies `sum(IG_i) = f(x) - f(baseline)`. NP-IG satisfies the weaker `sum(NP-IG_i) = f(x) - sum(w_b * f(b))`. The right-hand side is a weighted average of baseline predictions, which is not a single interpretable "absence" value. This may limit interpretability compared to single-baseline IG.

## 8. Modest Statistical Power

With N=150 per seed (×3 seeds = 450 total), bootstrap CIs may be wide. Statistical significance of H2 may be marginal for some comparisons. We apply Bonferroni correction for 4 single-baseline comparisons.

## 9. No Human Evaluation

We do not include human plausibility evaluation (are the attributed tokens interpretable to humans?). This is a known gap between faithfulness and plausibility [Jacovi & Goldberg 2020].

## 10. Search Limitations

Literature search was conducted on Semantic Scholar, arXiv, ACL Anthology, and related venues. Some workshop papers and theses may not have been indexed. We phrase novelty claims as "to the best of our knowledge, based on searches of [sources] on [date]."
