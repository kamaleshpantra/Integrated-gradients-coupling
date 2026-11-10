# Overleaf Upload Guide (Post-Review Revision)

This package contains the complete camera-ready LaTeX manuscript and style files for the study:
**"Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines"**

---

## 1. Quick Start with Overleaf

1. Go to [Overleaf](https://www.overleaf.com) and click **"New Project"** -> **"Upload Project"**.
2. Select `overleaf_paper.zip` located at:
   `c:\Users\91798\OneDrive\Desktop\Research\overleaf_paper.zip`
3. Overleaf will automatically unpack all files into the project root.
4. Set the compiler to **pdfLaTeX** (default in Overleaf).
5. Click **Recompile** to produce the camera-ready PDF.

---

## 2. Package Contents

| File | Description |
|---|---|
| `main.tex` | Full revised manuscript (ACL format) with all logged statistics and figures |
| `main.bib` | Clean, audited BibTeX file (verified DOIs and ACL Anthology links) |
| `acl.sty` | Official ACL formatting style file |
| `acl_natbib.bst` | Official ACL bibliography style |
| `fig1_grid_heatmap.png` | Figure 1: 4x4 In-Domain Grid Heatmap ($M=50$, $N=450$) |
| `fig2_erasure_comparison.png` | Figure 2: Grouped Erasure Comparison across all 7 schemes |
| `fig3_kendall_tau.png` | Figure 3: Kendall Tau correlation matrix across erasure schemes |
| `fig4_npig_ablation.png` | Figure 4: NP-IG temperature ablation (entropy and worst-case robustness) |

---

## 3. Removing Anonymity (For Camera-Ready)

In `main.tex`, change:
```latex
\usepackage[review]{acl}
```
to:
```latex
\usepackage{acl}
```
and replace `\author{Anonymous ACL Submission}` with your author information and affiliation.
