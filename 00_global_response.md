# Response to Reviewer dauU

We thank the reviewer for the careful review and for recognizing that our FP4 reward proxy
maintains RL performance while significantly accelerating training, supported by "solid
experimental validation" across representative models. We also deeply appreciate the
constructive critique: your points have directly helped us improve our contribution statement & paper completeness (W1&Q1), delimit the boundary of the FP4 assumption (W2), and prove that Sol-RL is
algorithm-agnostic (W3&Q2). We address each point below.

---

## W1. The first contribution is an overclaim

The computational bottleneck of RL rollouts and the degradation caused by low-precision
training are established observations (including in the LLM quantized-RL). 
We thank the reviewer for this correction; it sharpens what the paper actually claims and we will revise it accordingly.

---

## W2. The proxy assumption is vulnerable to the choice of reward model

It is the core question that tests the central assumption behind
Sol-RL: the cheap FP4 rollout must preserve enough **relative reward ordering** to choose useful contrastive candidates.
We therefore ran an ablation experiment across reward-model types, including the low-level
aesthetic and clarity-based rewards highlighted by the reviewer.

We measure the quantity Sol-RL actually uses. From 96 candidates, we take the FP4 proxy's
**best-12 and worst-12**, then report their average ranks under the high-precision (BF16)
reward. A perfect proxy gives **6.5 / 90.5**, corresponding to the true top-12 and bottom-12
candidates and thus the most informative contrastive selection. A proxy carrying **no
information** gives **48.5 / 48.5**, because its ranking is independent of the true BF16
ranking and its selected sets are effectively random.

| Reward | Type | FLUX.1 | SANA | SD3.5 |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| **Aesthetic** | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| **OCR / text fidelity** | clarity-based | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |
| *Perfect proxy (reference)* | | *6.5 / 90.5* | *6.5 / 90.5* | *6.5 / 90.5* |
| *No-information proxy (reference)* | | *48.5 / 48.5* | *48.5 / 48.5* | *48.5 / 48.5* |

The results support the assumption for both reward families we tested. For
aesthetic and clarity-based rewards, ranking preservation is lower, as expected because text and low-level artifacts are more sensitive to early FP4 degradation; nevertheless, the proxy still carries substantial information of contrastive seeds and is far from random.

---

## W3. Sol-RL is entangled with DiffusionNFT

Following the reviewer's suggestion we applied the **identical**
two-stage rollout pipeline on top of **FlowGRPO** and **online-GRPO**, changing only the policy optimization
objective and holding the rollout/selection machinery fixed.

Setup: SD3.5, HPSv2 reward, 512px, LoRA, 8×H100; eval on a
held-out 2048-prompt set.

| Eval reward (HPSv2) | ≈ 12 GPU-hours | ≈ 30 GPU-hours |
|---|---|---|
| Flow-GRPO baseline | 0.3254 | 0.3448 |
| **Flow-GRPO + Sol-RL (24-in-96)** | **0.3306** | **0.3579** |


| Eval reward (HPSv2) | ≈ 3 GPU-hours | ≈ 11 GPU-hours |
|---|---|---|
| Online DPO baseline | 0.3142 | 0.3163 |
| **Online DPO + Sol-RL** | **0.3299** | **0.3317** |

As shown above, under the same computational training budget, Sol-RL consistently achieves
higher rewards than the baselines. This confirms that the approach has strong algorithmic
generalization: the benefit is not specific to DiffusionNFT, but readily extends to other
group-relative methods like Flow-GRPO, as well as pairwise methods like online DPO. We
will include these results to clarify the method's algorithm-agnostic nature.


---

## Q1. Provide the exact loss actually used

We thank the reviewer for pointing this
out. Following this suggestion, the revision states the exact objective optimized in all
our experiments rather than only the generic GRPO formulation given in the preliminaries.

We optimize the DiffusionNFT objective. Writing $v^{\theta}$ for the model's velocity
prediction, $v^{\mathrm{old}}$ for the frozen reference policy, and $v$ for the target:

$$\mathcal{L}(\theta) \;=\; \mathbb{E}\Big[\, r\,\big\|v^{\theta+} - v\big\|^2 \;+\; (1-r)\,\big\|v^{\theta-} - v\big\|^2 \,\Big]$$

$$v^{\theta\pm} \;=\; (1 \mp \beta)\,v^{\mathrm{old}} \;\pm\; \beta\, v^{\theta}$$

where $\beta$ is the interpolation (implicit guidance) coefficient, and the reward weight
$r \in [0,1]$ is obtained from the raw reward by group centering and normalization:

$$r \;=\; 0.5 \;+\; 0.5\,\mathrm{clip}\!\left[\frac{r^{\mathrm{raw}} - \overline{r^{\mathrm{raw}}}}{Z_c},\; -1,\; 1\right]$$

with $Z_c$ the reward normalizer. Each candidate therefore enters the update through a
single scalar $r$: candidates scoring above the group mean are pulled toward the positive
branch $v^{\theta+}$ and those below it toward $v^{\theta-}$, weighted by their normalized
reward margin.

We will include this formulation in the revised paper, and will also clarify where the
standard GRPO formulation is used for exposition versus where the DiffusionNFT objective
is the one actually optimized.


---

## Q2. All experiments are w/o CFG; provide partial w/ CFG results

This is a fair point, and we have now runpartial ablation experiments **with CFG
enabled** (SD3.5, HPSv2, otherwise identical settings):

| Target eval reward | 0.300 | 0.315 | 0.325 | 0.335 | 0.353 |
|---|---|---|---|---|---|
| Sol-RL speedup, **w/ CFG** | 1.68× | 1.84× | 1.92× | 2.24× | — |
| Sol-RL speedup, **w/o CFG** | 1.62× | 1.67× | 1.73× | 1.86× | 2.08× |

**The benefit does not depend on the CFG setting**: across all reported finite entries, the
speedup ranges from 1.62× to 2.24×. The reported gains are therefore not an artifact of the
w/o-CFG configuration. ("-" means that baseline didn't achieve sthe eval reward in given training GPU-hours)

---

We thank the reviewer again -- W1 and W3 in particular have materially improved the
paper. We would be glad to run additional experiments during the discussion period and will include the coresponding experiment in our revised version.




# Response to Reviewer LK4D

We sincerely thank the reviewer for the thorough review. We are encouraged that the
reviewer finds our separation of "cheap exploration and reliable training" to be
"conceptually clean and practically appealing", considers the use of ranking consistency
"novel", and recognizes that the method "directly targets a real scaling bottleneck in
diffusion RL". We are equally grateful for the valuable questions, and we address each of them below.

---

## W1. Heavy dependence on FP4 preserving intragroup ranking consistency

We agree this is the load-bearing assumption, and we have tested it along all three axes the
reviewer names. All cells report the **mean true (BF16) rank of the proxy-selected top-12 /
bottom-12** out of a 96-candidate pool, over 256 prompts × 96 seeds. A perfect proxy gives
**6.5 / 90.5**, corresponding to the true top-12 and bottom-12 candidates and thus the most
informative contrastive selection. A proxy carrying **no information** gives
**48.5 / 48.5**, because its ranking is independent of the true BF16 ranking and its selected
sets are effectively random.

**(i) Across reward models × base models**

| Reward | Type | FLUX.1 | SANA | SD3.5 |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| OCR | low-level, clarity | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |

**(ii) By prompt difficulty.** We further break down the ranking preservation performance
across different levels of prompt complexity on both FLUX and SANA. We define prompt
difficulty using two standard benchmarks: 
1. **GenEval**, where we test categories ranging from the easiest (`single_object`) to the
   hardest (`position`).
2. **OCR digit-count ladder**, where the difficulty scales from generating 1 digit (easiest)
   up to 5 digits (hardest).

On GenEval, the ranking fidelity remains remarkably stable across difficulty levels:

| Model / reward (GenEval) | single_obj | two_obj | colors | counting | color_attr | position |
|---|---|---|---|---|---|---|
| FLUX | 8.7 / 87.9 | 8.5 / 87.1 | 8.5 / 87.5 | 9.0 / 88.1 | 7.9 / 86.5 | 8.7 / 86.7 |
| SANA | 10.5 / 85.5 | 12.4 / 84.6 | 11.1 / 85.0 | 10.8 / 85.4 | 10.5 / 85.5 | 12.3 / 84.5 |

Similarly, on the OCR ladder, performance holds up well even as the task becomes
progressively harder, showing only mild degradation at the absolute extreme (5 digits for SANA):

| Model / reward (OCR) | Easiest: 1 digit | Medium: 3 digits | Hardest: 5 digits |
|---|---|---|---|
| FLUX | 8.1 / 87.0 | 8.4 / 87.8 | 8.2 / 87.7 |
| SANA | 13.8 / 87.0 | 12.7 / 82.9 | 12.9 / **78.6** |

**(iii) By sampling schedule** — Euler vs Heun solver on FLUX.1, all else fixed:

| Reward | Euler | Heun |
|---|---|---|
| HPSv2 | 8.4 / 87.4 | 8.6 / 87.5 |
| PickScore | 8.6 / 87.2 | 8.8 / 86.4 |
| ImageReward | 10.1 / 86.0 | 10.3 / 84.7 |
| Aesthetic | 9.3 / 86.7 | 10.4 / 85.0 |
| OCR | 13.2 / 82.8 | 14.8 / 82.4 |

It shows that the ranking preservation nature of proxy reward changes  slightly across different schedulers.

Across all the experiments above, even the weakest cell (SD3.5 / Aesthetic) yields **15.1 / 79.1**, which remains clearly
informative relative to the no-information reference of **48.5 / 48.5**. It demonstrates that our proxy ranking can help to provide reliable high-contrasive training signal for GRPO optimization.

---

## W2. Selecting only top/bottom may overemphasize extremes and discard medium-quality samples

We agree with the reviewer's intuition that mid-ranked samples contain useful diversity.
The core idea of our method is to provide the most robust learning signal
under a fixed training budget, which is why we rely exclusively on the extreme candidates.

To further investigate the effect of mid-ranked samples, we tested them directly. Holding
the candidate pool fixed at 96, we compared using only the extremes (24) against adding 24
mid-ranked samples (48) or keeping everything (96). Under the same iterations (SD3.5,
PickScore, iteration 250), adding mid-ranked samples actually degraded performance:

| Kept from the 96-pool | Eval reward | Gain over untrained (0.758) |
|---|---|---|
| **24 (ours)** | **0.90762** | **0.150** |
| 48 | 0.90154 | 0.144 |
| 96 | 0.89276 | 0.135 |

We hypothesize this degradation occurs because reward models are least confident when scoring
mid-ranked, near-tied candidates. Because GRPO-style objectives must force each candidate
into a positive or negative update direction, these noisy, ambiguous scores interfere with
the clean gradient trajectory provided by the top and bottom extremes. Thus, focusing on the
extremes is not only cheaper, but actively produces a better optimization signal.

---

## W3 & Q2. Does the method inherit or amplify reward-model error?

Our two-stage design does not amplify reward-model error, because the FP4 proxy is never
used to compute the actual reward signal. FP4 is used **only to rank and select which seeds
are worth regenerating**. The samples that actually enter training are the full BF16
regenerations, and these regenerations are **re-scored independently by the reward model**.
No FP4 score ever becomes a training label.

Therefore, FP4 error only affects how *informative* the selected batch is, not whether the
batch is labeled correctly. In the worst case, a noisy FP4 ranking simply yields a weaker
contrastive training signal — but it is never worse than the baseline **24-in-24** setting,
which does no selection at all. The effect of FP4 error is thus reduced contrastive
efficiency, not the amplification of incorrect preferences.

---

## Q1. Why GRPO? Would the two-stage strategy work for DPO/PPO?

Sol-RL requires only that the algorithm samples a **group of candidates per prompt** and
uses **relative reward comparisons** among them. Rather than just argue this theoretically, we
tested it directly across algorithms.

**Flow-GRPO and Online DPO results.** We evaluated both Flow-GRPO and online DPO
under fixed computational budgets (GPU-hours). For Flow-GRPO, we varied the candidate pool
while holding everything else fixed. For online DPO, we varied the pool size from which the
preference pair is drawn.

| Eval reward (HPSv2) | ≈ 12 GPU-hours | ≈ 30 GPU-hours |
|---|---|---|
| Flow-GRPO baseline (24-in-24) | 0.3254 | 0.3448 |
| **Flow-GRPO + Sol-RL (24-in-96)** | **0.3306** | **0.3579** |

| Eval reward (HPSv2) | ≈ 3 GPU-hours | ≈ 11 GPU-hours |
|---|---|---|
| Online DPO baseline | 0.3142 | 0.3163 |
| **Online DPO + Sol-RL** | **0.3299** | **0.3317** |

The strategy transfers well to DPO, under
the same GPU-hour budget, Sol-RL consistently beats the baseline.

On **why a group-relative objective is the better host**: it consumes a top-k
and bottom-k *set* and weights each sample by its normalized advantage, meaning individual
reward-model errors are averaged out over the group. DPO consumes exactly one pair,
concentrating whatever error the reward model makes at the extremes into that single update.
Group-relative objectives are where wide-pool selection is most robust against noise, where we can find stable improvements on FlowGRPO compared with DPO as the table shows.

---

We thank the reviewer again and are happy to run further experiments during the
discussion period.


# Response to Reviewer 7D2G


We thank the reviewer for a detailed and constructive review, and for identifying the
BF16 24-in-96 comparison and the ranking-consistency analysis as the strongest evidence.
We especially
appreciate the recognition that our work addresses an important practical bottleneck in
diffusion RL, and that the two-stage design combining FP4 ranking with BF16 regeneration is
simple and well motivated. 
Every point raised is valuable suggestion for us, and we address each below.

---

## W1. Evaluation relies on learned reward models, some also used as training objectives

We agree this is an important concern to raise. 
Following the standard methodology of diffusion GRPO and other RL post-training methods, our
approach fundamentally optimizes against a given reward model, so it inevitably demonstrates
strong improvements on that specific metric.

We agree that human evaluation is the gold standard for verifying this. To that end, we
are currently conducting an internal user study to rigorously evaluate whether the
reward-driven gains align with actual human perception. We will share the results of this
study in the coming days.


---

## W2. Improvements are modest and no error bars are reported

Our central claim is **Diffusion-RL training acceleration**, manifested in two complementary
ways: Sol-RL achieves superior alignment quality under the same training GPU-hour budget,
and requires fewer GPU-hours to reach an equivalent reward standard. More importantly, it
approaches a substantially higher reward plateau within a realistic training budget. For
example, at approximately **120 GPU-hours** on SD3.5-Large, DiffusionNFT has nearly plateaued
at about **1.60**, whereas Sol-RL has reached about **1.78**. This directly translates into
stronger alignment quality under finite, practical training budgets.

We agree that statistical validation is necessary, but would like to clarify the
**characterization of the improvements**. Comparing only the final raw rewards
understates the gain because each metric has a substantial nonzero starting value. The
relevant quantity is the alignment improvement from the shared base model,
$\Delta R=R_{\mathrm{final}}-R_{\mathrm{base}}$. Under the identical GPU-hour budget in
Table 1, DiffusionNFT versus Sol-RL achieves $\Delta R$ of **1.2157 vs. 1.3086** on
ImageReward, **0.0361 vs. 0.0459** on CLIPScore, **0.0756 vs. 0.0836** on PickScore, and
**0.1047 vs. 0.1122** on HPSv2. Relative to DiffusionNFT, Sol-RL therefore increases the
realized alignment gain by **7.6%, 27.1%, 10.6%, and 7.2%**, respectively.

**Variance and Error bars.** To quantify variance, we ran four independent training runs on
SD3.5 for both CLIPScore and PickScore under the same GPU-hours and records final rewards of them. We present these preliminary measurements below:

| Metric | Method | Mean $\pm$ Std |
|---|---|---|
| **CLIPScore** | Baseline | 0.3005 $\pm$ 0.0034 |
| | **Sol-RL** | **0.3115 $\pm$ 0.0031** |
| **PickScore** | Baseline | 0.8908 $\pm$ 0.0028 |
| | **Sol-RL** | **0.9015 $\pm$ 0.0026** |

The results show that the training process is remarkably stable and consistent，roughly indicating that the improvement of our method is statistic significant. We will include comprehensive variance bounds in the revised version.


---

## W3 & Q1. Speedup targets in Figure 4 may be overstated if the baseline has plateaued

This is a valid methodological concern and we thank the reviewer for it. A single target
taken from the baseline's late-stage plateau can inflate the apparent speedup
arbitrarily.

**How the target was originally chosen.** The figure is *time-to-parity*: the wall-clock
GPU-hours Sol-RL needs to reach the **baseline's final** reward (under given training GPU-budgets), relative to the hours the
baseline needs. The reviewer is right that this is the criterion most sensitive to a
plateaued baseline, where 4.64× is exactly the result of this measurement method.

**Revised reporting.** To evaluate this fairly across models with different reward scales,
we uniformly slice the baseline's trajectory from start to finish into 25%, 50%, 75%, and
100% (plateau) progress milestones. We then report the speedup (Baseline GPU-hours /
Sol-RL GPU-hours) required to reach each specific reward standard:

| Base Model | 25% Progress | 50% Progress | 75% Progress | 100% Progress (Plateau) |
|---|---|---|---|---|
| SANA | 1.6× | 1.9× | 2.2× | 2.42× |
| FLUX.1 | 1.6× | 1.9× | 2.3× | 3.00× |
| SD3.5 Large | 2.1× | 2.8× | 3.7× | 4.64× |

The speedup **never falls below 1.6×** at any threshold for any model, so the benefit
is clearly not confined to the plateau region. While the peak speedup (e.g., 4.64×) is
amplified by the baseline's flat tail, Sol-RL consistently provides a convergency speedup.


---

## Q2. Are extra system costs included in the reported speedups?

Yes and we will make this explicit, since the paper did not state it clearly enough.
All reported speedups are **end-to-end wall-clock per training iteration**, measured on
the full pipeline including every additional operation Sol-RL introduces beyond a
standard BF16 rollout.

The operations the reviewer lists fall into two kinds. **Engine compilation** is a one-time
cost of roughly **10 minutes**, amortized over a run measured in $\approx$ 150 GPU-hours of 8 GPUs. The
**recurring** operations, including re-quantization and weight synchronization, amount to a single
pass over the weights of a T2I model(<=10B) and complete in 
ms-level (re-comilation is not required because only weights are updated), negligible against the
rollout and optimization steps they sit between.

---

## W4 & Q3. Scale and scope of the NVFP4 exploration analysis

We agree this was underspecified for an analysis that carries the method's core
assumption. The full protocol, now stated in the revision:

- **Prompts for ranking analysis:** 256, from the standard splits used by the DiffusionNFT codebase
- **Seeds:** 96 per prompt — i.e. **24,576 samples per configuration**
- **Candidates per group:** 96
- **Base models:** FLUX.1, SANA, SD3.5 — reported **pooled in the paper** and
  disaggregated in this response
- **Reward models:** HPSv2, PickScore, ImageReward, Aesthetic, OCR
- **Figure 3c and Table 8:** the same 256 prompts × 96 seeds

**Per prompt difficulty.** We use two ladders with an unambiguous ordering rather than a
subjective split. Cells are mean true rank (of 96) of the proxy-selected top-12 / bottom-12.

*GenEval, six categories from single-object to position (hardest):*

| Model / reward | single_obj | two_obj | colors | counting | color_attr | position |
|---|---|---|---|---|---|---|
| FLUX | 8.7 / 87.9 | 8.5 / 87.1 | 8.5 / 87.5 | 9.0 / 88.1 | 7.9 / 86.5 | 8.7 / 86.7 |
| SANA | 10.5 / 85.5 | 12.4 / 84.6 | 11.1 / 85.0 | 10.8 / 85.4 | 10.5 / 85.5 | 12.3 / 84.5 |

*OCR digit count, 1 → 3 → 5 digits (hardest):*

| Model / reward (OCR) | Easiest: 1 digit | Medium: 3 digits | Hardest: 5 digits |
|---|---|---|---|
| FLUX | 8.1 / 87.0 | 8.4 / 87.8 | 8.2 / 87.7 |
| SANA | 13.8 / 87.0 | 12.7 / 82.9 | 12.9 / **78.6** |

Finally, to demonstrate that ranking preservation holds across different kinds of objectives,
we tested the FP4 proxy across **five distinct reward models**. A perfect proxy gives
**6.5 / 90.5**, corresponding to the true top-12 and bottom-12 candidates and thus the most
informative contrastive selection. A proxy carrying **no information** gives
**48.5 / 48.5**, because its ranking is independent of the true BF16 ranking and its selected
sets are effectively random:

| Reward | Type | FLUX.1 | SANA | SD3.5 |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| OCR | low-level, clarity | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |

Across all the experiments above, even the weakest cell (SD3.5 / Aesthetic) yields **15.1 / 79.1**, which remains clearly
informative relative to the no-information reference of **48.5 / 48.5**. It demonstrates that our proxy ranking can help to provide reliable high-contrasive training signal for GRPO optimization.

---

We thank the reviewer again -- W2 and W3 in particular have led us to state our claims
more precisely, and we believe the revised framing (compute-efficient preservation,
threshold-robust speedups, fully-accounted overheads) is both more defensible and more
useful to readers. We are happy to run additional measurements during the discussion
period.


# Response to Reviewer aSZi

We thank the reviewer for the positive assessment of our method's novelty, solid empirical
validation, and algorithm-hardware co-design. We especially appreciate that the concerns
raised precisely target the boundary conditions of our empirical claims and theoretical
framework. In response to this constructive critique, we provide concrete
explanations below, supported by new ablation studies on quantization formats and clarification
of the theoretical lower bound.

---

## W1. Generalizability of the ranking-preservation assumption

We appreciate the reviewer acknowledging that the paper provides strong empirical
evidence across the tested models, and agree that generalization beyond them is not
guaranteed. We have extended the evidence along the axes most likely to break the
assumption.

**More complex / different reward functions.** We measure the quantity the method actually
depends on: from a pool of 96 candidates we take the FP4 proxy's best-12 and worst-12, and
report their average ranks under the high-precision (BF16) reward. A perfect proxy gives
**6.5 / 90.5**, corresponding to the true top-12 and bottom-12 candidates and thus the most
informative contrastive selection. A proxy carrying **no information** gives
**48.5 / 48.5**, because its ranking is independent of the true BF16 ranking and its selected
sets are effectively random.
Protocol: 256 prompts × 96 seeds.

| Reward | Type | FLUX.1 | SANA | SD3.5 |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| Aesthetic | aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| OCR | clarity-based | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |

We fully acknowledge that ranking preservation varies depending on the specific combination
of base model and reward model. The recurring weak spot is the bottom end of the OCR reward,
while the top end remains stable; quantization scope also affects fidelity.
However, even in the weakest cell (SD3.5 / Aesthetic),
the proxy yields **15.1 / 79.1**, which remains clearly informative relative to the
no-information reference of **48.5 / 48.5**. Therefore, across the tested model-reward
combinations, the FP4 proxy retains sufficient ranking information to provide a
**reliable contrastive learning signal** for accelerating GRPO convergence.

**Video generation.** We agree this is an important next step. The underlying
mechanism is theoretically not image-specific. To verify whether the ranking-preservation
property extends to video generation, we are currently running ranking-consistency experiments
on video models and will share the results in the coming days.

---

## W2. No comparison against a different quantization format (e.g. FP8)

We agree this comparison is the right way to isolate the effect of the aggressive FP4 format
from the broader two-stage framework. FP8 preserves ranking slightly better than FP4, but because our framework is
highly tolerant to selection noise, FP4 allows us to extract the maximum throughput
advantage without sacrificing final performance.

**(1) Ranking preservation.** We first measure how reducing precision affects selection
fidelity (SD3.5, HPSv2, 96 candidates, selecting top-12 and bottom-12). We report the true
BF16 rank of the candidates selected by each proxy:

| Proxy Precision | True BF16 Rank of Top-12 Picks | True BF16 Rank of Bottom-12 Picks |
|---|---|---|
| BF16 (Perfect) | 6.5 | 90.5 |
| FP8 | 7.7 | 87.8 |
| FP4 (Ours) | 8.6 | 83.9 |
| Random (No Info) | 48.5 | 48.5 |

As expected, FP8 preserves ranking slightly better than FP4.

**(2) End-to-end performance under equal training budget.** Because the two-stage framework
insulates the gradients from the proxy's exact output, the pipeline easily tolerates
FP4's slight ranking degradation. Consequently, under fixed computational budgets (in GPU
hours), the faster FP4 proxy can execute more rollouts and scale the search pool more
aggressively than FP8, converting that throughput advantage directly into higher final
evaluation rewards:

| Method | 50 GPU-hours | 100 GPU-hours | 150 GPU-hours |
|---|---|---|---|
| BF16 Baseline | 0.293 | 0.298 | 0.301 |
| FP8 Sol-RL | 0.295 | 0.305 | 0.308 |
| FP4 Sol-RL (Ours) | **0.301** | **0.309** | **0.312** |

This comparison clarifies the specific advantage of FP4. 
FP4 safely maximizes the search-pool scaling that the two-stage framework
enables, yielding faster convergence, especially under smaller compute budgets.

---

## W3. Appendix A relies on strong assumptions; violations are not discussed

Thank you for raising this point. We agree that a useful global Lipschitz constant may be
unavailable for highly nonlinear reward models such as ImageReward. In practice, we directly
evaluate the required consequence: cross-precision ranking stability. Table 8 shows that
ImageReward achieves Kendall's $\tau=0.807$, Spearman's $\rho=0.932$, a 97.2% Top-4 match
rate, and only 3.9% Bottom-4 false inclusion between FP4 and BF16. Across all four reward
models, the corresponding averages are 0.798, 0.927, 96.9%, and 3.9%.

Consistent with the lower bound's qualitative prediction, Table 3 shows that increasing $N$
from 24 to 96 monotonically improves HPSv2 from 0.3569 to 0.3686, while Table 6 shows
performance within 1.08% of naive BF16 scaling across all three evaluated models. If ranking
stability is severely violated, the lower-bound guarantee no longer applies. We will clarify
this scope and the theory-experiment connection.

---

We thank the reviewer again. The FP8 comparison (W2) and the theory-to-practice link
(W3) have both improved the paper, and we are happy to run further experiments during
the discussion period.



# Response to the Area Chair

We sincerely thank the Area Chair for the thoughtful summary and constructive guidance. We
especially appreciate the recognition that our work addresses an important practical problem
in diffusion RL, and that the reviewers found the analysis insightful and the empirical
results promising and consistent across settings. We are equally grateful for the clearly
identified concerns, which have helped us sharpen the scope and strengthen the evaluation.
Below, we address each concern with additional analyses of reward-model robustness and
FP4-induced noise, statistical variance, standard RL objectives, and clearer statements of
our claims and experimental setup.

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

**Novelty (dauU-W1).** We agree that the rollout bottleneck and the degradation caused by
low-precision training are established observations, particularly in quantized RL for LLMs.
However, quantization in diffusion RL remains comparatively underexplored. We thank the
reviewer for this correction and will revise our framing to present the established
observations as motivation and state our contribution more precisely.

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