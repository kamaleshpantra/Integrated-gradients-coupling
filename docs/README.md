# Research Documentation & Methodology Index

Welcome to the project documentation index for the research study:
**"Baseline-Evaluation Coupling in Integrated Gradients for Text: A Cross-Evaluation Study and Analysis of Ensemble Baselines"**

This folder contains the complete formal paper documentation, lab notebooks, methodology specs, and evaluation materials following academic best practices (The Turing Way, Open Science Framework, and ACL/NeurIPS Reproducibility Standards).

---

## 🗺️ Research Lifecycle Roadmap

Every step from the initial research design to the final camera-ready revision is tracked chronologically and conceptually:

```mermaid
flowchart TD
    A["Phase 1: Inception & Proposal<br>docs/01_research_proposal.md"] --> B["Phase 2: Literature Audit<br>docs/02_literature_review.md"]
    B --> C["Phase 3: Implementation & Specs<br>docs/04_methodology_details.md"]
    C --> D["Phase 4: Lab Experimentation<br>docs/03_experiment_log.md"]
    D --> E["Phase 5: Independent Audit & Rerun<br>AUDIT.md & CHANGELOG.md"]
    E --> F["Phase 6: Reproducibility & Checklist<br>docs/07_reproducibility_checklist.md"]
    F --> G["Phase 7: Claims Defense & Evaluation<br>docs/09_defense_checklist.md"]
```

---

## 📂 Document Catalog

| Document | Purpose & Description | Focus Area |
|---|---|:---:|
| [`01_research_proposal.md`](docs/01_research_proposal.md) | Initial research proposal: research questions (RQ1–RQ3), hypotheses (H1–H3), experimental design, resource requirements, and timeline. | Project Scope |
| [`02_literature_review.md`](docs/02_literature_review.md) | Literature survey on axiomatic attribution, baseline selection in NLP, faithfulness metrics (ERASER), and out-of-domain erasure artifacts. | Academic Context |
| [`03_experiment_log.md`](docs/03_experiment_log.md) | Chronological lab notebook detailing all experimental milestones, parameter configurations, runtime observations, and code fixes. | Lab Notebook |
| [`04_methodology_details.md`](docs/04_methodology_details.md) | Technical implementation specs: Riemann sum quadrature, probability output space integration, attention mask handling under PAD, logit-space Grad×Input, and LMM formula. | Technical Implementation |
| [`06_data_dictionary.md`](docs/06_data_dictionary.md) | Complete schema and variable definitions for all raw CSV files, JSON outputs, and ablation datasets. | Data Schema |
| [`07_reproducibility_checklist.md`](docs/07_reproducibility_checklist.md) | Official ACL/NeurIPS Responsible Research Checklist with direct section and line pointers. | Conference Standards |
| [`08_reproduction_guide.md`](docs/08_reproduction_guide.md) | Step-by-step terminal commands to reproduce all experiments, statistical analyses, figures, and Overleaf packages from scratch. | Replication |
| [`09_defense_checklist.md`](docs/09_defense_checklist.md) | Comprehensive defense briefing: elevator pitch, claim-by-claim statistical proof, tough questions & answers, and live demonstration commands. | Peer Review & Evaluation |

---

## 🔍 How to Use This Documentation

1. **For Peer Review & Presentation:** Open [`09_defense_checklist.md`](docs/09_defense_checklist.md) to walk through claims, methodology decisions, and statistical evidence.
2. **For Re-running Code:** Follow the commands in [`08_reproduction_guide.md`](docs/08_reproduction_guide.md).
3. **For Conference Submission:** Attach [`07_reproducibility_checklist.md`](docs/07_reproducibility_checklist.md) and upload the ready-made [`overleaf_paper.zip`](paper/overleaf_paper.zip).
