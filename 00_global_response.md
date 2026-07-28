# Global Response (to the Area Chair and all Reviewers)

We thank the Area Chair and all four reviewers. We note that **all four reviewers rate the
submission 4 (Borderline accept)**, including Reviewer aSZi (submitted 23 Jul), who assessed
the work as having "strong" novelty, "strong empirical validation", "extensive thorough
analysis" and "clear algorithm-hardware co-design". As the meta-review references three
reviewers, we respectfully draw attention to this fourth assessment.

Following the AC's guidance we have focused this rebuttal on **reward-model vulnerability
and the impact of FP4-induced noise**, and additionally provide standard-RL-objective
results, statistical variance, dialed-back claims, and answers to every clarification
question.

---

## 1. Reward-model vulnerability and the impact of FP4 noise (AC's primary concern)

The AC asks for "further analysis of the impact of the noise introduced by **FP4 rollout**".
We isolate exactly that: the error contributed by the FP4 stage itself, rather than
reward-model noise in general or synthetic perturbations.

### (a) Precision sweep of the first stage — does added quantization noise propagate?

We vary **only the precision of the first stage**, holding everything else fixed. BF16 → FP8
→ FP4 is a monotone ladder of quantization noise entering the selection step. To evaluate
ranking fidelity, we measure the true BF16 rank of the candidates selected by each proxy
(SD3.5-M, HPSv2, 96 candidates, selecting top-12 and bottom-12):

| Proxy Precision | True BF16 Rank of Top-12 Picks | True BF16 Rank of Bottom-12 Picks |
|---|---|---|
| BF16 (Perfect) | 6.5 | 90.5 |
| FP8 | 7.7 | 87.8 |
| **FP4 (Ours)** | **8.6** | **83.9** |
| Random (No Info) | 48.5 | 48.5 |

To evaluate the final end-to-end impact under equivalent computational budgets (GPU-hours),
we observe the final eval rewards across three budget thresholds:

| Method | 50 GPU-hours | 100 GPU-hours | 150 GPU-hours |
|---|---|---|---|
| BF16 Baseline | 0.293 | 0.298 | 0.301 |
| FP8 Sol-RL | 0.295 | 0.305 | 0.308 |
| **FP4 Sol-RL (Ours)** | **0.301** | **0.309** | **0.312** |

The first table establishes that the noise is real and grows along the sweep:
lower precision measurably perturbs the ranking and changes which candidates are selected.
However, the second table shows this noise does **not** propagate into a penalty on the 
final trained model. Instead, because FP4 is faster than FP8 and BF16, it executes more 
rollouts within the same compute budget, converting the speedup directly into higher 
alignment quality. Injecting more noise into the selection stage buys speed without 
costing final reward — the opposite of amplification. This sweep directly addresses 
Reviewer aSZi's request for an FP8 comparison (W2) and Reviewer LK4D's question on error amplification.

### (b) Selection fidelity — where do the FP4-selected candidates actually rank?

Sol-RL uses the FP4 ranking only to *choose* which candidates to regenerate in BF16, so the
operative question is whether the **selected sets** match those a high-precision pipeline
would have chosen. From a pool of 96 candidates we take the FP4 proxy's best-12 and
worst-12, and report their average ranks under the **BF16** reward. A perfect proxy gives
**6.5 / 90.5**, corresponding to the true top-12 and bottom-12 candidates and thus the most
informative contrastive selection. A proxy carrying **no information** gives
**48.5 / 48.5**, because its ranking is independent of the true BF16 ranking and its selected
sets are effectively random.

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| OCR / text fidelity | low-level, clarity-based | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |

Protocol: **256 prompts × 96 seeds** (24,576 samples per configuration), 96 candidates per
group, and three base models. The original paper reports the analysis **pooled** across
models; here we additionally disaggregate it by model and prompt difficulty.

**By prompt difficulty** we use two ladders with an unambiguous ordering. On GenEval (six
categories, single-object → position) fidelity is essentially **flat**: FLUX/HPSv2 moves
from 8.7 / 87.9 to 8.7 / 86.7 across the full range, and SANA behaves the same way. On an
OCR digit ladder (1 / 3 / 5 digits), FLUX remains nearly flat (8.1 / 87.0 → 8.2 / 87.7).
For SANA, top-end selection also remains stable (13.8 → 12.9), while bottom-end selection
degrades moderately (87.0 → 78.6) as digit count increases. Prompt difficulty is therefore
not the dominant factor; **quantization scope is**. Full tables are in our reply to
Reviewer 7D2G.

Two conclusions. First, the reviewers' hypothesis is **directionally correct**: fidelity is
highest for the human-preference rewards and lowest for the two low-level rewards — exactly
the ordering expected if FP4 artifacts interfere most with low-level judgements. We report
this rather than average it away, and will delimit the assumption explicitly in the revision
instead of claiming universal validity.

Second, the degradation is **bounded**. Even in the weakest cell (SD3.5-M / Aesthetic), the
proxy yields **15.1 / 79.1**, which remains clearly informative relative to the
no-information reference of **48.5 / 48.5**. Across the human-preference rewards that
dominate our main experiments, the selected sets are close to the high-precision pipeline's.
The FP4 stage causes a **measurable, bounded** loss of selection quality and never approaches
the no-information regime.

### (c) Held-out reward evaluation — genuine alignment or reward hacking?

Following the standard methodology of RL post-training methods, our approach fundamentally
optimizes against a given reward model, so it inevitably demonstrates strong improvements on
that specific metric. However, our qualitative visualizations (e.g., **Figure 5: Visual
comparison before and after Sol-RL**) show that this does not lead to obvious "reward hacking."
Instead, the optimization translates into genuine improvements in human preference and
aesthetic quality.

To rigorously evaluate whether these reward-driven gains align with actual human perception,
we are currently conducting an internal human-preference user study. We will share the full
results in the coming days.

---

## 2. Standard RL objectives: Sol-RL on FlowGRPO

Addressing the AC's "insufficient experimental scope" (dauU-W3, LK4D-Q1), we applied the
identical two-stage pipeline on top of two further objectives. Setup: SD3.5-medium, HPSv2,
512px, 10 steps, LoRA, 8×H100, held-out 2048-prompt eval.

**Flow-GRPO** — under a fixed computational budget (GPU-hours), Sol-RL consistently beats the
baseline:

| Eval reward (HPSv2) | ≈ 12 GPU-hours | ≈ 30 GPU-hours |
|---|---|---|
| Flow-GRPO baseline (24-in-24) | 0.3254 | 0.3448 |
| **Flow-GRPO + Sol-RL (24-in-96)** | **0.3306** | **0.3579** |

**Online DPO** — a preference-based rather than group-relative objective. Under the same
GPU-hour budget, varying only the candidate pool size from which the preference pair is drawn:

| Eval reward (HPSv2) | ≈ 3 GPU-hours | ≈ 11 GPU-hours |
|---|---|---|
| Online DPO baseline | 0.3142 | 0.3163 |
| **Online DPO + Sol-RL** | **0.3299** | **0.3317** |

The benefit therefore holds across **DiffusionNFT, Flow-GRPO and DPO** — two group-relative
objectives and one pairwise-preference objective. Sol-RL is an objective-agnostic
rollout-scaling framework, not an artifact of DiffusionNFT.

---

## 3. Dialing back overstated claims

**Novelty (dauU-W1).** Identifying the rollout bottleneck and low-precision degradation is
prior knowledge; it is **moved to the motivation**. Our contributions are restated as:
(i) the finding that low-precision rollouts preserve reward *ranking* even when sample
fidelity degrades; (ii) the rank-then-regenerate design exploiting this; (iii) the system
realization, validated across models, objectives and reward families.

**Alignment quality (7D2G-W2).** We revise the claim from superior alignment to
**compute-efficient quality preservation** — matching the BF16 large-rollout pipeline at
substantially lower cost. Wording implying uniformly superior alignment is removed.

**Speedup (7D2G-W3/Q1).** We acknowledge that reporting speedup at a single target drawn
from the baseline's plateau can inflate the apparent speedup. To evaluate the final
end-to-end impact fairly, we now report final eval rewards under equivalent computational
budgets (GPU-hours):

| Method | 50 GPU-hours | 100 GPU-hours | 150 GPU-hours |
|---|---|---|---|
| BF16 Baseline | 0.293 | 0.298 | 0.301 |
| FP8 Sol-RL | 0.295 | 0.305 | 0.308 |
| **FP4 Sol-RL (Ours)** | **0.301** | **0.309** | **0.312** |

This comparison clarifies the specific advantage of FP4: under fixed computational budgets,
the faster FP4 proxy can execute more rollouts and scale the search pool more aggressively,
converting that throughput advantage directly into higher final eval rewards.

**System overheads (7D2G-Q2).** All speedups are end-to-end wall-clock **including**
re-quantization, compiled-engine updates and weight synchronization. Engine compilation is a
one-time cost of roughly 10 minutes, amortized over a run measured in $\approx$ 150 GPU-hours.

---

## 4. Statistical validation

To quantify variance and validate the significance of our improvements, we ran four
independent training runs on SD3.5-Medium for both CLIPScore and PickScore:

| Metric | Arm | Mean $\pm$ Std |
|---|---|---|
| **CLIPScore** | Baseline | 0.3005 $\pm$ 0.0034 |
| | **Sol-RL** | **0.3115 $\pm$ 0.0031** |
| **PickScore** | Baseline | 0.8908 $\pm$ 0.0028 |
| | **Sol-RL** | **0.9015 $\pm$ 0.0026** |

The results show that the training process is remarkably stable and consistent. Sol-RL's
lowest individual run comfortably outperforms the baseline's highest individual run in both
cases.

---

## 5. Remaining clarifications

- **w/ CFG (dauU-Q2):** The ablation with CFG enabled is now included in the revision. The relative advantage of Sol-RL is preserved because CFG scales both stages proportionally.
- **Exact loss (dauU-Q1):** the DiffusionNFT objective as implemented is now given in full.
- **GRPO vs DPO/PPO (LK4D-Q1):** Sol-RL requires only group-based candidate sampling and
  relative reward comparison; DPO-style pair mining is a natural fit.
- **Appendix A assumptions (aSZi-W3):** ranking is invariant to monotone transformations of
  the reward — weaker than the bound assumes — so violations degrade the constant rather
  than the conclusion.

Details for each are in our individual replies. We thank the AC and reviewers again, and are
happy to run further experiments during the discussion period.
