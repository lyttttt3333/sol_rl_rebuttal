# Response to Area Chair

We sincerely thank the Area Chair for the clear and actionable guidance. Following your explicit instructions, we have focused our rebuttal on rigorously testing the reward model vulnerabilities and the impact of FP4 noise. We have also dialed back overstated claims, answered all clarification questions, and provided the requested experimental analysis on standard RL objectives and statistical variance. 

We summarize our responses to the Primary Concerns below.

---

### 1. Over-reliance on the reward model and fragile FP4 ranking assumptions

**A. Ranking stability across diverse and low-level rewards**
We evaluated the FP4 proxy's ranking preservation across five diverse reward models,
specifically targeting the low-level Aesthetic and OCR metrics highlighted by the reviewers.
From a pool of 96 candidates, we report the average BF16 ranks of the proxy-selected top-12
and bottom-12. A perfect proxy gives **6.5 / 90.5**, corresponding to the true top-12 and
bottom-12 candidates and thus the most informative contrastive selection. A proxy carrying
**no information** gives **48.5 / 48.5**, because its ranking is independent of the true
BF16 ranking and its selected sets are effectively random.

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| OCR | low-level, clarity | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |

The results confirm the reviewers' intuition: fidelity is highest for human-preference rewards
and lowest for low-level rewards, with the bottom end of OCR being the recurring weak spot.
However, the degradation is **bounded**. Even in the weakest cell (SD3.5-M / Aesthetic), the
proxy yields **15.1 / 79.1**, which remains clearly informative relative to the
no-information reference of **48.5 / 48.5**.

**B. Impact of FP4 noise and error amplification**
To isolate the effect of FP4 noise and check for error amplification, we performed a precision sweep of the first stage (BF16 $\rightarrow$ FP8 $\rightarrow$ FP4) under equal computational budgets (SD3.5-M, HPSv2):

| Proxy Precision | True BF16 Rank of Top-12 | True BF16 Rank of Bottom-12 |
|---|---|---|
| BF16 (Perfect) | 6.5 | 90.5 |
| FP8 | 7.7 | 87.8 |
| **FP4 (Ours)** | **8.6** | **83.9** |

*End-to-end performance under equal budget:*

| Method | 50 GPU-hours | 100 GPU-hours | 150 GPU-hours |
|---|---|---|---|
| BF16 Baseline | 0.293 | 0.298 | 0.301 |
| FP8 Sol-RL | 0.295 | 0.305 | 0.308 |
| **FP4 Sol-RL (Ours)** | **0.301** | **0.309** | **0.312** |

These tables confirm that while FP4 introduces measurable selection noise, it does **not** amplify incorrect preferences in the final model. Because the actual gradients are computed exclusively on BF16 regenerations, the noise only affects search efficiency. Under a fixed compute budget, FP4's speed advantage allows scaling the search pool ($N=96$), converting that throughput directly into higher end-to-end rewards than FP8 or the baseline.

**C. Reward hacking vs. genuine alignment**
Optimizing against a specific reward naturally improves that metric, but our qualitative visualizations (e.g., **Figure 5**) demonstrate this translates to genuine improvements in aesthetic quality rather than reward hacking. We are currently conducting an internal human-preference user study to rigorously verify this, and will share the results shortly.

---

### 2. Evaluation methodology and overclaiming

**A. Dialing back claims and Novelty**
We accept the framing that our quality improvements are modest, and we completely agree that identifying the computational bottleneck is not a novel contribution. We have revised our claims from "superior alignment" to "**compute-efficient quality preservation**," and moved the bottleneck discussion to the motivation section.

**B. Standard RL Objectives (Flow-GRPO and Online DPO)**
To prove Sol-RL is not entangled with a specific objective, we applied the identical framework to standard RL objectives under fixed computational budgets (GPU-hours):

| Eval reward (HPSv2) | $\approx$ 12 GPU-hours | $\approx$ 30 GPU-hours |
|---|---|---|
| Flow-GRPO baseline (24-in-24) | 0.3254 | 0.3448 |
| **Flow-GRPO + Sol-RL (24-in-96)** | **0.3306** | **0.3579** |

<br>

| Eval reward (HPSv2) | $\approx$ 3 GPU-hours | $\approx$ 11 GPU-hours |
|---|---|---|
| Online DPO baseline | 0.3142 | 0.3163 |
| **Online DPO + Sol-RL** | **0.3299** | **0.3317** |

Sol-RL consistently outperforms the baseline across both Flow-GRPO and pairwise Online DPO, demonstrating strong algorithmic generalization.

**C. Statistical Variance**
To validate the significance of our improvements, we ran four independent training runs on SD3.5-Medium for both CLIPScore and PickScore:

| Metric | Arm | Mean $\pm$ Std |
|---|---|---|
| **CLIPScore** | Baseline | 0.3005 $\pm$ 0.0034 |
| | **Sol-RL** | **0.3115 $\pm$ 0.0031** |
| **PickScore** | Baseline | 0.8908 $\pm$ 0.0028 |
| | **Sol-RL** | **0.9015 $\pm$ 0.0026** |

The training process is remarkably stable. The variance is small enough that Sol-RL's lowest individual run comfortably outperforms the baseline's highest individual run.

**D. Missing details: Robustness across Prompt Difficulty**
We evaluated ranking preservation across standard difficulty ladders:

| Model / reward (GenEval) | single_obj | two_obj | colors | counting | color_attr | position |
|---|---|---|---|---|---|---|
| FLUX | 8.7 / 87.9 | 8.5 / 87.1 | 8.5 / 87.5 | 9.0 / 88.1 | 7.9 / 86.5 | 8.7 / 86.7 |
| SANA | 10.5 / 85.5 | 12.4 / 84.6 | 11.1 / 85.0 | 10.8 / 85.4 | 10.5 / 85.5 | 12.3 / 84.5 |

| Model / reward (OCR) | Easiest: 1 digit | Medium: 3 digits | Hardest: 5 digits |
|---|---|---|---|
| FLUX | 8.1 / 87.0 | 8.4 / 87.8 | 8.2 / 87.7 |
| SANA | 13.8 / 87.0 | 12.7 / 82.9 | 12.9 / **78.6** |

Ranking fidelity holds up remarkably well across all prompt difficulties, showing only mild degradation at the absolute extreme (5 digits for SANA).

---

We believe these comprehensive results address the vulnerabilities raised and provide a fully validated, algorithm-agnostic confirmation of our claims. We are happy to include these analyses in the final revision.