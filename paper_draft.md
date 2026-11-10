# Revised Manuscript: Baseline-Evaluation Coupling in Integrated Gradients for Text

**Status:** Fully Revised Post-Audit Draft (ACL Format). All numbers trace to raw CSV logs.

## Key Empirical Findings (M=50, N=450 SST-2, 3 Seeds)

- **Linear Mixed Model Matched Effect:** $\beta = +0.0342$ (95% CI: [0.0222, 0.0462], $p = 0.0000$)
- **Variance Explained:** Baseline = 1.2%, Scheme = 0.1%, Matched Indicator = 0.2%
- **Length Interaction:** Short ($\le 10$ tokens): $\beta = +0.0586$ ($p=0.0046$); Long ($> 20$ tokens): $\beta = +0.0330$ ($p=0.0000$)
- **Dual Worst-Case Robustness:** Uniform-IG achieves highest worst-case across all 7 schemes (Def 2 = 0.3316, Def 1 = 0.1285)

See `paper/main.tex` for the complete camera-ready paper.
