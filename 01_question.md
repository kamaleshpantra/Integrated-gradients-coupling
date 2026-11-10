# Research Question and Problem Statement

## Research Question (One Sentence)
Do faithfulness evaluations of Integrated Gradients (IG) in transformer text classifiers suffer from artificial inflation when the attribution reference baseline matches the evaluation erasure operator (baseline-evaluation coupling), and can a neutral-prediction weighted multi-baseline average (NP-IG) provide robust, evaluation-invariant attribution across diverse erasure schemes?

## Why It Matters
Integrated Gradients (IG; Sundararajan et al., 2017) is one of the most widely adopted axiomatic attribution methods for explaining deep learning models. In Natural Language Processing (NLP), applying IG requires choosing a reference baseline—a counterfactual input representing the "absence" of signal. Common choices in literature include the `[MASK]` token, `[PAD]` token, `[UNK]` token, or an all-zero embedding vector.

Simultaneously, post-hoc faithfulness of feature attributions is standardly evaluated via erasure-based metrics, such as Comprehensiveness, Sufficiency (DeYoung et al., 2020), and Area Over the Perturbation Curve (AOPC; Samek et al., 2016). These metrics test whether erasing top-ranked tokens alters model predictions. Critically, these metrics themselves require an erasure operator: substituting tokens with `[MASK]`, `[PAD]`, `[UNK]`, or physically deleting them from the input sequence.

If an IG method uses `[MASK]` as its baseline and is evaluated using `[MASK]` erasure, the path integral computes how moving from `[MASK]` to the word affects the output; when the metric subsequently evaluates the word by replacing it back with `[MASK]`, the metric may measure the exact perturbation path designed by the baseline choice rather than true intrinsic feature importance. If such *baseline-evaluation coupling* produces artificial inflation on the "diagonal" of baseline-by-erasure evaluations, then published comparisons of interpretability methods may reflect metric-baseline alignment rather than genuine explanation quality.

## What a Yes or No Answer Would Change

### If the answer is YES (Baseline-Evaluation Coupling exists):
1. **Benchmark Redesign**: Established benchmarks (e.g., ERASER) and research papers can no longer rely on single-token erasure metrics without controlling for baseline alignment. Standard operating procedure must mandate cross-evaluation across multiple distinct erasure schemes (e.g., cross-matrix testing with deletion, replacement, and slice shifts).
2. **Methodological Shift**: Practitioners must abandon unweighted single-baseline IG in text. They should adopt neutral-prediction multi-baseline formulations (such as NP-IG) that average attributions across diverse valid references weighted by neutral probability, eliminating reliance on an arbitrary, evaluation-coupled reference.
3. **Re-assessment of Prior Claims**: Prior literature claiming superiority of one baseline over another (e.g., claiming `[MASK]` outperforms `[PAD]` or zero embedding based solely on `[MASK]`-erasure faithfulness) must be re-evaluated as potentially circular artifacts.

### If the answer is NO (Invariance across baselines and erasure):
1. **Validation of Standard Protocols**: The NLP interpretability community can confidently continue using convenient single-reference baselines (such as `[MASK]`) and matching erasure tokens without concern for circular evaluation artifacts.
2. **Theoretical Confirmation**: It would indicate that transformer representations at top-attributed positions dominate model predictions so robustly that out-of-distribution artifacts and baseline-erasure alignment are negligible secondary effects compared to intrinsic semantic salience.
