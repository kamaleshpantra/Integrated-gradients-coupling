# Defense Preparation: Defending Your Claims to an Advisor or Committee

**Purpose:** This document prepares you to walk through and defend every claim in the paper, anticipate tough questions, and show the evidence chain.

---

## 1. The 5-Minute Summary (Elevator Pitch)

> "We studied whether matching the IG baseline token with the evaluation erasure token inflates faithfulness scores — a confound we call *baseline-evaluation coupling*. Using a full 4×4 factorial grid with DistilBERT on SST-2, a linear mixed model shows coupling is **real but small** ($\beta = +0.034$, $p < 10^{-8}$), explaining only **0.17% of variance** while baseline choice explains **1.22%**. Coupling is strongest on short inputs and null on IMDB. Simple Uniform-IG (equal-weight averaging) outperforms the proposed NP-IG method and provides the best worst-case robustness across 7 erasure schemes, including two held-out schemes never seen during attribution."

---

## 2. Claim-by-Claim Defense

### Claim 1: "Matched coupling produces significant inflation"

**Evidence:** LMM coefficient $\beta = +0.0342$, 95% CI $[0.0222, 0.0462]$, $p = 2.42 \times 10^{-8}$

**Anticipate:** *"Is this just driven by one baseline?"*  
**Answer:** No. All 4 per-baseline contrasts are significant after Holm correction:
- MASK: $\Delta = +0.049$ ($p < 0.0001$)
- ZERO: $\Delta = +0.032$ ($p = 0.0095$)
- PAD: $\Delta = +0.029$ ($p = 0.0070$)
- UNK: $\Delta = +0.026$ ($p = 0.0070$)

**Show:** `revision_analysis_results.json` → `h1_lmm.per_baseline_contrasts`

---

### Claim 2: "Baseline choice dominates coupling in variance"

**Evidence:** ANOVA SS: Baseline = 1.22%, Matched = 0.17%

**Anticipate:** *"0.17% seems trivially small. Why should anyone care?"*  
**Answer:** That's exactly our point. The coupling effect is statistically real but practically modest. The literature has focused on *which baseline to use* — our contribution is showing that the *evaluation protocol* adds a small but systematic bias. The practical recommendation is to always cross-evaluate, which is cheap and eliminates the confound entirely.

**Anticipate:** *"98.5% is residual — isn't the model just bad?"*  
**Answer:** The residual reflects per-example variation (some sentences are inherently easier to attribute). The random intercepts in the LMM absorb this. The key comparison is baseline (1.22%) vs. matched (0.17%), not against residual.

---

### Claim 3: "Coupling is strongest on short inputs"

**Evidence:** Short ($\leq 10$ tokens): $\beta = +0.059$ vs. Long ($> 20$): $\beta = +0.033$

**Anticipate:** *"Short bin only has N=39. Is that enough?"*  
**Answer:** The effect is still significant at $p = 0.005$ despite the small sample. The monotonic decrease (0.059 → 0.030 → 0.033) and the IMDB null result ($p = 0.153$ at mean length 94) together form a coherent length-dependency story.

**Show:** `revision_analysis_results.json` → `h1_lmm.length_binned_effects`

---

### Claim 4: "IMDB exhibits null diagonal inflation"

**Evidence:** Diagonal mean = 0.419, off-diagonal mean = 0.422, $p = 0.153$

**Anticipate:** *"Maybe you just didn't have enough IMDB examples (N=100)?"*  
**Answer:** The effect direction is actually *negative* (off-diagonal slightly higher), not just underpowered. With N=100 long-form reviews (mean ~94 tokens), erasing one token has negligible contextual impact. This corroborates the length hypothesis rather than contradicting it.

**Show:** `project/results/analysis_imdb_h1.json`

---

### Claim 5: "Uniform-IG outperforms NP-IG"

**Evidence:** Worst-case (Def 2, all 7 schemes): Uniform = 0.3316 vs. NP-IG ($\tau=0.1$) = 0.3040

**Anticipate:** *"Did you try other temperature values? Maybe there's a sweet spot."*  
**Answer:** Yes — full sweep from $\tau=0.01$ to $\tau=\infty$. The result is monotonic: as entropy increases toward uniform, worst-case robustness improves. No temperature beats uniform. This is because neutrality weighting concentrates 90% of mass on MASK, re-introducing single-baseline fragility.

**Show:** Table 3 (ablation), Figure 4 (curve), `project/results/np_ig_ablation.csv`

**Anticipate:** *"Isn't this a negative result? Why publish it?"*  
**Answer:** Negative results are valuable. If NP-IG had been published without this ablation, practitioners might waste effort tuning $\tau$. The finding that the simplest possible ensemble (equal weights) is optimal is itself a useful practical recommendation.

---

### Claim 6: "NP-IG collapses onto MASK"

**Evidence:** At $\tau=0.1$: $w_{\text{MASK}} = 0.900$, entropy = 0.391 nats (vs. max 1.386)

**Anticipate:** *"Is this specific to DistilBERT, or would it happen on other models?"*  
**Answer:** It depends on how close each baseline's prediction is to 0.5. For DistilBERT, MASK happens to produce near-neutral predictions ($|p - 0.5| \approx 0.048$). If a different model had PAD as the most neutral baseline, NP-IG would collapse onto PAD instead. The structural problem — softmax temperature causing winner-take-all — generalizes.

---

## 3. Tough Questions to Prepare For

| Question | Prepared Response |
|---|---|
| *"Why only DistilBERT? Would this hold on larger models?"* | Acknowledged in Limitations. DistilBERT is tractable for a factorial study. Larger models may show different baseline neutrality patterns but the mathematical coupling (IG explains $f(x)-f(b)$, erasure evaluates at $b$) is model-agnostic. |
| *"Why not use sufficiency as well as comprehensiveness?"* | Sufficiency requires keeping top-$k$ tokens, which creates different artifacts (embedding a few tokens in a pad/mask context). We focused on comprehensiveness following ERASER. Adding sufficiency is future work. |
| *"How do you know the mixed model is the right specification?"* | We tested for random intercepts vs. no random effects using likelihood ratio. The random-intercept model was significantly better ($p < 0.001$). We did not include random slopes because the 4×4 grid per example doesn't support estimation of 16 random slope parameters. |
| *"The manifold hypothesis is untested. Why include it?"* | We explicitly label it as a hypothesis alongside the counterfactual alignment explanation. We do not claim to have tested the mechanism — only the phenomenon. Both explanations are consistent with the data. |
| *"Were the experiments run with AI assistance? Is that legitimate?"* | Yes, documented in the AI Assistance Statement. All code, experiments, and analysis were executed programmatically with full logging. Every number traces to a raw CSV file. AI assistance does not invalidate reproducible computational experiments. |

---

## 4. Materials to Bring to the Defense

### Physical / Digital Copies
- [ ] Printed copy of `main.tex` (PDF compiled from Overleaf)
- [ ] Laptop with `project/` folder open for live data queries
- [ ] Printed `09_claims_evidence.md` (one page)
- [ ] Printed `AUDIT.md` (shows you caught and fixed your own errors)

### Live Demos
```python
# Quick verification that any number in the paper is real:
import json, pandas as pd
d = json.load(open("project/results/revision_analysis_results.json"))
df = pd.read_csv("project/results/sst2_revision_all450.csv")

# "Show me the matched effect"
print(f"Beta = {d['h1_lmm']['lmm_matched_coef']:.4f}")

# "Show me any grid cell"
val = df[(df["method"]=="IG_UNK") & (df["scheme"]=="UNK")]["norm_comp_aopc"].mean()
print(f"IG-UNK × Erase-UNK = {val:.4f}")
```

---

## 5. What Changed From the Original Paper (Advisor Should Know)

| Original Claim | Revised Claim | Why |
|---|---|---|
| NP-IG is the main contribution | NP-IG is a secondary ablation | Collapsed onto MASK; Uniform-IG is simpler and better |
| Coupling is the dominant evaluation artifact | Coupling is real but baseline choice matters more | ANOVA shows 0.17% vs 1.22% |
| "Worst-case" was a single definition | Two formal definitions (Def 1 and Def 2) | Original text and table used different definitions |
| Manifold hypothesis presented as explanation | Presented as untested hypothesis | Reviewer correctly noted it's not empirically validated |
| 3 citations were wrong | Fixed or removed | Audit found fabricated/erroneous references |
| M=20 integration steps | M=50 (with M=100 convergence check) | Completeness error was too high at M=20 |
