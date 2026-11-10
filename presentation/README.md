# Research Presentation

This directory contains the widescreen presentation slide deck accompanying the research paper.

## Files

- `research_presentation.pptx`: 15-slide widescreen (16:9) presentation deck containing:
  - Slide 1: Title & Author Overview
  - Slide 2: Executive Summary & Core Contributions
  - Slide 3: Motivation & Problem Formulation (The Coupling Hypothesis)
  - Slide 4: Theoretical Background (Integrated Gradients & Path Integration)
  - Slide 5: Cross-Evaluation Methodology (4x4 Grid & 7 Erasure Schemes)
  - Slide 6: Experimental Framework (DistilBERT, SST-2, 450 Examples)
  - Slide 7: Main Empirical Finding (The 4x4 In-Domain Grid & Matched Inflation)
  - Slide 8: Statistical Modeling (Linear Mixed Model & ANOVA Decomposition)
  - Slide 9: Length Interaction & IMDB Null Result
  - Slide 10: Multi-Baseline Ensembling & Worst-Case Robustness (Uniform-IG vs NP-IG)
  - Slide 11: Mechanistic Discovery (NP-IG Softmax Temperature Collapse)
  - Slide 12: Ranking Consistency (Kendall's Tau Across Erasure Schemes)
  - Slide 13: Concrete Recommendations for NLP Interpretability
  - Slide 14: Defense Briefing (Direct Answers to Tough Questions)
  - Slide 15: Reproducibility, Open-Source Assets & Q&A

## To Re-generate Slides
```bash
python scripts/generate_presentation.py
```
