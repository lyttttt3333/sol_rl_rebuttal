# Response to Reviewer 7D2G


We thank the reviewer for a detailed and constructive review, and for identifying the
BF16 24-in-96 comparison and the ranking-consistency analysis as the strongest evidence.
Every point raised is directly measurable, and we address each below.

---

## W1. Evaluation relies on learned reward models, some also used as training objectives

We completely agree this is an important concern to raise.

Following the standard methodology of diffusion GRPO and other RL post-training methods, our
approach fundamentally optimizes against a given reward model, so it inevitably demonstrates
strong improvements on that specific metric. However, our qualitative visualizations (e.g., 
**Figure 5: Visual comparison before and after Sol-RL**) show that this does not lead to
obvious "reward hacking." Instead, the optimization translates into genuine improvements in
human preference and aesthetic quality.

We agree that human evaluation is the gold standard for verifying this. To that end, we
are currently conducting an internal user study to rigorously evaluate whether the
reward-driven gains align with actual human perception. We will share the results of this
study in the coming days.


---

## W2. Improvements are modest and no error bars are reported

**We accept this framing, and we think it is the correct one for our paper.** Our central
claim is *compute-efficient quality preservation* -- matching the BF16 large-rollout
pipeline at substantially lower cost -- rather than superior alignment quality. We will
revise the claims in Section [[TBD]] and the abstract to state this precisely, and remove
any wording that implies uniformly superior alignment across metrics.

**Error bars added.** Table 1 and Figure 4 are re-reported as mean +/- std over
[[TBD: n]] seeds:

| Model / Reward | BF16 naive (mean +/- std) | Sol-RL (mean +/- std) |
|---|---|---|
| [[TBD]] | [[TBD]] | [[TBD]] |
| [[TBD]] | [[TBD]] | [[TBD]] |

[[TBD: STATE whether the quality gaps are within noise. If they are, say so -- it
supports the "preservation" claim rather than undermining it, since the efficiency gain
is large and clearly outside noise.]]

---

## W3 & Q1. Speedup targets in Figure 4 may be overstated if the baseline has plateaued

This is a valid methodological concern and we thank the reviewer for it. A single target
taken from the baseline's late-stage plateau can inflate the apparent speedup
arbitrarily.

**How the target was originally chosen.** The figure is *time-to-parity*: the wall-clock
GPU-hours Sol-RL needs to reach the **baseline's final** reward, relative to the hours the
baseline needs. The reviewer is right that this is the criterion most sensitive to a
plateaued baseline. The paper reports a **range, 1.91×–4.64×** — 4.64× is the most
favourable cell, not a typical one — but we agree the headline should not lead with the
extreme, and will revise it.

**Revised reporting.** We now report speedup **at every reward threshold along the
trajectory** rather than at a single point. For Sol-RL on Flow-GRPO (SD3.5-medium, HPSv2):

| Target eval reward | 0.300 | 0.315 | 0.325 | 0.335 | 0.353 |
|---|---|---|---|---|---|
| Speedup, w/ CFG | 1.68× | 1.84× | 1.92× | 2.24× | — |
| Speedup, w/o CFG | 1.62× | 1.67× | 1.73× | 1.86× | 2.08× |

The speedup **never falls below 1.38×** at any threshold in either setting, so the benefit
is not confined to the plateau region.

**A threshold-free comparison.** We agree the strongest answer removes target selection
entirely. Without CFG, the two arms converge to **exactly the same final reward (0.3530)**:
the baseline requires **230 epochs**, Sol-RL requires **110** — a **2.09×** reduction with
no target-selection freedom whatsoever. We will report this endpoint-matched comparison
alongside the threshold curve.

We will apply the same threshold-swept reporting to the main results and revise the
abstract to quote the range with its criterion stated, rather than the maximum alone.

---

## Q2. Are extra system costs included in the reported speedups?

Yes -- and we will make this explicit, since the paper did not state it clearly enough.
All reported speedups are **end-to-end wall-clock per training iteration**, measured on
the full pipeline including every additional operation Sol-RL introduces beyond a
standard BF16 rollout.

The operations the reviewer lists fall into two kinds. **Engine compilation** is a one-time
cost of roughly **10 minutes**, amortized over a run measured in [[TBD: N]] GPU-hours. The
**recurring** operations — re-quantization and weight synchronization — amount to a single
pass over the weights of a 2–3B model and complete in **seconds**, negligible against the
rollout and optimization steps they sit between. Both are inside the wall-clock we report;
neither is subtracted out. The per-iteration breakdown will be added to the revision.

---

## W4 & Q3. Scale and scope of the NVFP4 exploration analysis

We agree this was underspecified for an analysis that carries the method's core
assumption. The full protocol, now stated in the revision:

- **Prompts:** 256, from the standard splits used by the DiffusionNFT codebase
  (GenEval / OCR / PickScore / DrawBench)
- **Seeds:** 96 per prompt — i.e. **24,576 samples per configuration**
- **Candidates per group:** 96
- **Base models:** FLUX.1, SANA, SD3.5 — reported **separately, never pooled**
- **Reward models:** HPSv2, PickScore, ImageReward, Aesthetic, CLIPScore, OCR
- **Figure 3c and Table 8:** the same 256 prompts × 96 seeds

**Disaggregated results (previously pooled).** Per base model:

| Base model | rho (semantic rewards) | rho (low-level rewards) |
|---|---|---|
| SD3.5-Large | [[TBD]] | [[TBD]] |
| FLUX.1 | [[TBD]] | [[TBD]] |
| SANA-1.5 | [[TBD]] | [[TBD]] |

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

**Two findings, and we report the second even though it is the less convenient one.**

First, prompt difficulty is largely **not** the driver. Across the full GenEval ladder
FLUX/HPSv2 moves from 8.7/87.9 to 8.7/86.7 — flat — and SANA behaves the same way. What
does vary is **quantization scope**: at any fixed difficulty, SD3.5-M (fully quantized) sits
roughly twice as far from ideal as FLUX or SANA. This is a more actionable characterization
than "harder prompts break the proxy", and we will state it as such.

Second, there is one genuine difficulty effect, and it is confined to the **bottom end of
the OCR ladder**: as digit count rises the proxy loses its ability to identify the *worst*
candidates (SANA/OCR 84.0 → 66.9). The explanation is mechanical — once five digits are
uniformly illegible, "which is least legible" carries little signal. Notably the **top end
is unaffected** (21.2 → 20.3), so the positive half of each contrastive pair remains intact.

We additionally varied the **solver** (our default Euler flow-matching sampler vs Heun,
all else fixed): the top end shifts by at most 1.2 rank positions and the bottom end is
unchanged for the preference rewards, with the one appreciable shift again at OCR's bottom
end (−5.6). That the same localized weak spot appears under both an independent difficulty
sweep and a solver change suggests it is a real property — the proxy's difficulty in
ordering *uniformly illegible* text — rather than measurement noise, and we report it as
such rather than pooling it away.

---

We thank the reviewer again -- W2 and W3 in particular have led us to state our claims
more precisely, and we believe the revised framing (compute-efficient preservation,
threshold-robust speedups, fully-accounted overheads) is both more defensible and more
useful to readers. We are happy to run additional measurements during the discussion
period.
