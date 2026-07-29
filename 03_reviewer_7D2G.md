# Response to Reviewer 7D2G


We sincerely thank the reviewer for the detailed and constructive assessment. We especially
appreciate the recognition that our work addresses an important practical bottleneck in
diffusion RL, and that the two-stage design combining FP4 ranking with BF16 regeneration is
simple and well motivated. We are also grateful that the reviewer identifies the BF16
24-in-96 comparison as our strongest experimental result, showing that Sol-RL largely
preserves the quality of naive BF16 large-rollout scaling while reducing rollout and
end-to-end iteration time. We further appreciate the positive assessment of the NVFP4
exploration analysis, which directly supports the ranking-preservation assumption underlying
our method, as well as the recognition that validation across multiple base models and reward
metrics strengthens the evidence for its efficiency and effectiveness. We address each
concern below.

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

We agree that statistical validation is necessary, but would like to clarify the
characterization of the improvements as modest. Comparing only the final raw rewards
understates the gain because each metric has a substantial nonzero starting value. The
relevant quantity is the alignment improvement from the shared base model,
$\Delta R=R_{\mathrm{final}}-R_{\mathrm{base}}$. Under the identical GPU-hour budget in
Table 1, DiffusionNFT versus Sol-RL achieves $\Delta R$ of **1.2157 vs. 1.3086** on
ImageReward, **0.0361 vs. 0.0459** on CLIPScore, **0.0756 vs. 0.0836** on PickScore, and
**0.1047 vs. 0.1122** on HPSv2. Relative to DiffusionNFT, Sol-RL therefore increases the
realized alignment gain by **7.6%, 27.1%, 10.6%, and 7.2%**, respectively. The improvement
is thus substantial when measured against the actual reward gained during post-training,
rather than against the absolute final score.

Our central claim is **Diffusion-RL training acceleration**, manifested in two complementary
ways: Sol-RL achieves superior alignment quality under the same training GPU-hour budget,
and requires fewer GPU-hours to reach an equivalent reward standard. More importantly, it
approaches a substantially higher reward plateau within a realistic training budget. For
example, at approximately **120 GPU-hours** on SD3.5-Large, DiffusionNFT has nearly plateaued
at about **1.60**, whereas Sol-RL has reached about **1.78**. This directly translates into
stronger alignment quality under finite, practical training budgets. We will revise the text
and abstract to state this claim precisely.

**Variance and Error bars.** To quantify variance, we ran four independent training runs on
SD3.5-Medium for both CLIPScore and PickScore under the same GPU-hours. We present these preliminary measurements below:

| Metric | Method | Mean $\pm$ Std |
|---|---|---|
| **CLIPScore** | Baseline | 0.3005 $\pm$ 0.0034 |
| | **Sol-RL** | **0.3115 $\pm$ 0.0031** |
| **PickScore** | Baseline | 0.8908 $\pm$ 0.0028 |
| | **Sol-RL** | **0.9015 $\pm$ 0.0026** |

The results show that the training process is remarkably stable and consistent，roughly indicating that the improvement of our method is statistic significant. We will include comprehensive variance bounds in the revised version


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
pass over the weights of a T2I model(<10B) and complete in **seconds**, negligible against the
rollout and optimization steps they sit between. Both are inside the wall-clock we report;
neither is subtracted out. The per-iteration breakdown will be added to the revision.

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

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| OCR | low-level, clarity | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |

Fidelity is highest for the human-preference rewards and lowest for the low-level ones. The
recurring weak spot is the bottom end of the OCR reward, while the top end remains stable.
We will report this ordering rather than claim uniform validity.

---

We thank the reviewer again -- W2 and W3 in particular have led us to state our claims
more precisely, and we believe the revised framing (compute-efficient preservation,
threshold-robust speedups, fully-accounted overheads) is both more defensible and more
useful to readers. We are happy to run additional measurements during the discussion
period.
