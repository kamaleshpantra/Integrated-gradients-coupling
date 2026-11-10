# CHANGELOG.md: Post-Review Revision Record

This document catalogs every methodological, statistical, and textual change made to the study following independent review.

## 1. Discrepancy Fixes & Clarifications (Audit Step 1)
- **Worst-Case Robustness Definitions:** Formally separated Definition 1 (mean of per-example minimum) and Definition 2 (minimum of per-scheme means). Both are now reported side-by-side in Table 2, with Def 2 adopted as primary.
- **NP-IG Collapse on MASK:** Documented that at $\tau=0.1$, NP-IG dedicates $\sim 70-90\%$ of its weight to MASK (entropy $\approx 0.89$ nats vs uniform $1.386$). Explained why NP-IG closely mirrored IG-MASK in the pilot.
- **Grad $\times$ Input Softmax Saturation:** Diagnosed that evaluating Grad $\times$ Input on probability outputs resulted in vanished gradients ($p(1-p) < 0.0002$ on confident predictions). Fixed by evaluating in unnormalized logit space.
- **Documentation:** Documented Riemann sum quadrature, probability output space integration, attention mask behavior under PAD (`attention_mask=0` at padded positions), top-k rounding, and Kendall tau ranking count ($n=8$).
- **Bibliography Overhaul:** Removed fabricated citation `hentschel2024canonical`; corrected Bastings et al. (EMNLP 2022, final author Katja Filippova); corrected Liu et al. to arXiv:2310.04821; verified all DOIs and ACL Anthology IDs.

## 2. Experimental Upgrades (Steps 2-5)
- **Higher Resolution ($M=50$):** Reran the full SST-2 matrix across Seeds 42, 123, 999 ($N=450$) at $M=50$, reducing completeness error by $10\times$.
- **True Zero-Embedding Erasure:** Added `ZERO_EMB` erasure operator (zeroing out token embeddings while retaining mask) providing a true matched pair for `IG_ZERO`.
- **Linear Mixed Model:** Replaced confounded diagonal test with LMM: `norm_comp ~ baseline + scheme + matched + (1|example)`. Effect size: $\beta = +0.0342$ ($p=0.0000$). Demonstrated that baseline selection explains 1.2% of variance, dominating matched coupling (0.2%).
- **Held-Out Erasure:** Evaluated methods under `RANDOM_TOKEN` (5-draw average) and `MLM_INFILL`. Uniform-IG proved most robust across all 7 schemes.
- **NP-IG Temperature Sweep:** Demonstrated that unweighted Uniform-IG outperforms neutrality-weighted NP-IG in worst-case robustness because neutrality weighting re-concentrates mass onto MASK.
