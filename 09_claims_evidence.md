# Claims vs. Evidence Table (Post-Revision Step 6)

Every claim in the revised paper is mapped to its raw log and exact statistic.

| Claim in Paper | Source Result File | Exact Statistic | Support Verdict |
|---|---|---|:---:|
| Matched coupling produces significant inflation on SST-2 | `revision_analysis_results.json` | LMM $\beta = +0.0342$, 95% CI [0.0222, 0.0462], $p=0.0000$ | **SUPPORTED** |
| Baseline selection dominates matched coupling in variance | `revision_analysis_results.json` | Baseline SS = 1.2%, Matched SS = 0.2% | **SUPPORTED** |
| Matched inflation is concentrated in short inputs | `revision_analysis_results.json` | Short $\beta = +0.0586$ ($p=0.0046$) vs Long $\beta = +0.0330$ ($p=0.0000$) | **SUPPORTED** |
| IMDB exhibits null diagonal inflation ($p=0.153$) | `analysis_imdb_h1.json` | Diag 0.419 vs Off-diag 0.422 ($p=0.153$) | **SUPPORTED** |
| Uniform-IG achieves best worst-case robustness across all 7 schemes | `revision_analysis_results.json` | Uniform Def 2 = 0.3316, Def 1 = 0.1285 | **SUPPORTED** |
| Neutrality weighting collapses onto MASK at low tau | `np_ig_ablation.csv` | $\tau=0.1$ MASK wt = 0.90, entropy = 0.39 nats | **SUPPORTED** |
| Relative token rankings are invariant above M=50 | `steps_stability.csv` | Spearman(50, 100) > 0.998 on PAD/MASK/UNK, 0.971 on ZERO | **SUPPORTED** |
