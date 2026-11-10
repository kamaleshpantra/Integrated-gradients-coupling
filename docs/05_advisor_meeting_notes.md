# Advisor Meeting Notes

**Project:** Baseline-Evaluation Coupling in Integrated Gradients for Text  
**Student:** [Your Name]  
**Advisor:** [Advisor Name]  
**Meeting Cadence:** Weekly / Bi-weekly

---

## Meeting 1 — Project Kickoff

**Date:** 2026-09-01  
**Duration:** 45 min  
**Format:** In-person

### Agenda
- Present initial research idea: investigating baseline-erasure coupling in IG
- Discuss feasibility and scope

### Advisor Feedback
- Advisor approved the general direction but cautioned: "Make sure the effect isn't trivially explained by the completeness axiom"
- Suggested starting with a clean 4×4 factorial grid before anything complex
- Recommended DistilBERT on SST-2 as a tractable starting point
- Asked about statistical methodology: "Don't just compare means — use a proper mixed model"

### Decisions Made
- Scope: Start with SST-2, add IMDB if time permits
- Method: Linear mixed-effects model for the main hypothesis test
- Timeline: 8-week target for a workshop-length paper

### Action Items
- [ ] Write formal research proposal (due: 2026-09-08)
- [ ] Set up computational environment and download models
- [ ] Read Bastings et al. 2022 and Chrysostomou & Aletras 2022

---

## Meeting 2 — Pilot Results Review

**Date:** 2026-09-22  
**Duration:** 30 min  
**Format:** Virtual

### Data Reviewed
- 4×4 heatmap from pilot study (N=150, M=20, seed=42)
- Diagonal cells appeared consistently higher

### Advisor Feedback
- "The diagonal pattern is promising but M=20 has too much quadrature error. You need to show convergence."
- "Add a convergence plot: M=20 vs M=50 vs M=100"
- "The ZERO baseline has no matched erasure pair — you need a ZERO_EMB erasure operator"
- Asked about Grad × Input: "Why is it essentially zero? That needs to be explained, not ignored."

### Decisions Made
- Increase to M=50 for main experiments
- Add ZERO_EMB erasure scheme
- Investigate and document the Grad × Input failure

### Action Items
- [ ] Implement ZERO_EMB erasure
- [ ] Diagnose Grad × Input near-zero scores
- [ ] Run convergence analysis at M=20/50/100

---

## Meeting 3 — Statistical Methodology Deep Dive

**Date:** 2026-10-05  
**Duration:** 60 min  
**Format:** In-person

### Topics Discussed
- Walked through the LMM specification
- Discussed variance decomposition (ANOVA SS)
- Reviewed the two worst-case definitions

### Advisor Feedback
- "The two worst-case definitions need to be formally stated with equations, not just described in prose. The discrepancy between text and table was confusing."
- "Report BOTH definitions in the paper, pick one as primary, and explain why."
- "For the per-baseline contrasts, use Holm-Bonferroni, not Bonferroni — it's less conservative."
- "Add a length-binned analysis. My intuition is that coupling should be stronger on short sentences."

### Decisions Made
- Definition 2 (min of per-scheme means) as primary metric
- Holm-Bonferroni correction for multiple comparisons
- Three length bins: ≤10, 11–20, >20 tokens

### Action Items
- [ ] Implement length-binned LMM analysis
- [ ] Write up both worst-case definitions with LaTeX equations
- [ ] Run IMDB experiments to test length hypothesis

---

## Meeting 4 — IMDB Null Result & NP-IG Collapse

**Date:** 2026-10-15  
**Duration:** 45 min  
**Format:** Virtual

### Data Reviewed
- IMDB diagonal test: $p = 0.153$ (null result)
- NP-IG weight distribution: 90% on MASK at $\tau = 0.1$

### Advisor Feedback
- "The IMDB null result is actually great — it supports your length hypothesis. Don't hide it."
- "NP-IG collapsing onto MASK is a serious problem for the original paper's claims. You need to be transparent about this."
- "Run a temperature sweep to show the full picture. If NP-IG never beats Uniform-IG, say so clearly."
- "Consider adding held-out erasure schemes (random token, MLM infill) to test whether Uniform-IG is really robust or just coupling to multiple baselines at once."

### Decisions Made
- Keep IMDB null result in the paper (abstract and main text)
- Demote NP-IG from "main contribution" to "ablated proposal"
- Add held-out erasure experiments
- Reframe paper as empirical study, not method paper

### Action Items
- [ ] Implement RANDOM_TOKEN and MLM_INFILL erasure
- [ ] Run NP-IG temperature sweep
- [ ] Restructure paper outline

---

## Meeting 5 — Draft Paper Review

**Date:** 2026-10-28  
**Duration:** 90 min  
**Format:** In-person (line-by-line review)

### Sections Reviewed
- Abstract, Introduction, Methodology, Results, Discussion, Limitations

### Advisor Feedback
- "The abstract is good — it includes the null result and the Uniform-IG recommendation. Keep it."
- "In Discussion, present the manifold hypothesis as a *hypothesis*, not a proven mechanism."
- "Add the counterfactual alignment explanation as an alternative reading."
- "Limitations section needs to explicitly state: single model, single dataset (plus IMDB), no autoregressive models."
- "Verify every citation. I've seen too many papers with hallucinated references."

### Decisions Made
- Two complementary hypotheses in Discussion (manifold + counterfactual)
- Expanded Limitations section
- Full citation audit against ACL Anthology

### Action Items
- [ ] Complete citation audit (AUDIT.md §5)
- [ ] Finalize all figures from raw data
- [ ] Package Overleaf ZIP
- [ ] Prepare claims-vs-evidence table

---

## Meeting 6 — Final Sign-Off

**Date:** 2026-10-30 (Scheduled)  
**Duration:** 30 min  
**Format:** [TBD]

### Planned Agenda
- Present final deliverables checklist
- Walk through claims-vs-evidence table
- Discuss submission venue (ACL Rolling Review / EMNLP workshop)
- Sign off on AI Assistance Statement

### Pre-Meeting Preparation
- [ ] Print deliverables checklist (see `docs/09_defense_checklist.md`)
- [ ] Prepare 5-minute walkthrough of key findings
- [ ] Have `overleaf_paper.zip` ready for upload
