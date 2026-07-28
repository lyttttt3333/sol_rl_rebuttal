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
bottom-12** out of a 96-candidate pool, over 256 prompts × 96 seeds. **Perfect proxy =
6.5 / 90.5; no-information proxy = 48.5 / 48.5.**

**(i) Across reward models × base models**

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 14.2 / 74.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 19.1 / 67.1 |
| OCR | low-level, clarity | 13.2 / 82.8 | 20.7 / 73.2 | 15.0 / 72.9 |

**(ii) By prompt difficulty** — GenEval, easiest → hardest category:

| Model / reward | single_object | position |
|---|---|---|
| FLUX / HPSv2 | 8.7 / 87.9 | 8.7 / 86.7 |
| FLUX / ImageReward | 10.6 / 87.0 | 9.9 / 84.3 |
| SANA / Aesthetic | 10.5 / 85.5 | 12.3 / 84.5 |
| SD3.5-M / ImageReward | 22.0 / 65.1 | 27.8 / 57.7 |

and an OCR digit-count ladder, 1 → 5 digits:

| Model / reward | 1 digit | 3 digits | 5 digits |
|---|---|---|---|
| FLUX / HPSv2 | 8.1 / 87.0 | 8.4 / 87.8 | 8.2 / 87.7 |
| FLUX / OCR | 13.8 / 87.0 | 12.7 / 82.9 | 12.9 / **78.6** |
| SANA / OCR | 21.2 / 84.0 | 20.6 / **68.7** | 20.3 / **66.9** |

**(iii) By sampling schedule** — Euler vs Heun solver, all else fixed:

| Reward | Euler | Heun |
|---|---|---|
| HPSv2 | 8.2 / 87.6 | 8.6 / 87.5 |
| PickScore | 8.7 / 86.4 | 9.3 / 86.4 |
| ImageReward | 9.6 / 85.6 | 10.3 / 84.7 |
| Aesthetic | 9.7 / 87.0 | 10.4 / 85.0 |
| OCR | 18.8 / 78.0 | 19.8 / **72.4** |

Fidelity is highest for the human-preference rewards and lowest for the low-level ones, and
we will state that ordering rather than claim uniform validity. Difficulty itself is not the
dominant factor — **quantization scope** is. The one recurring weak spot, visible on all
three axes, is the **bottom end of the OCR reward**; the top end never degrades materially.
Even the weakest cell (SD3.5-M / Aesthetic) still separates its selected sets by 48 rank
positions of 96, versus 0 for an uninformative proxy. Sol-RL needs a ranking informative
enough to identify contrastive candidates, not an exact one, and that holds throughout.

---

## W2. Selecting only top/bottom may overemphasize extremes and discard medium-quality samples

We agree with the reviewer's intuition that mid-ranked samples contain useful diversity.
However, the core idea of our method is to provide the most robust learning signal possible
under a minimal training budget, which is why we rely exclusively on the extreme candidates.

To further investigate the effect of mid-ranked samples, we tested them directly. Holding
the candidate pool fixed at 96, we compared using only the extremes (24) against adding 24
mid-ranked samples (48) or keeping everything (96). Under the same iterations (SD3.5,
PickScore, iteration 250), adding mid-ranked samples actually degraded performance:

| Kept from the 96-pool | Eval reward | Gain over untrained (0.758) |
|---|---|---|
| **24 (ours)** | **0.90762** | **0.150** |
| 48 | 0.90154 | 0.144 |
| 96 — nothing discarded | 0.89276 | 0.135 |

We hypothesize this degradation occurs because reward models are least confident when scoring
mid-ranked, near-tied candidates. Because GRPO-style objectives must force each candidate
into a positive or negative update direction, these noisy, ambiguous scores interfere with
the clean gradient trajectory provided by the top and bottom extremes. Thus, focusing on the
extremes is not only cheaper, but actively produces a better optimization signal.

---

## W3 & Q2. Does the method inherit -- or *amplify* -- reward-model error?

Our two-stage design does not amplify reward-model error, because the FP4 proxy is never
used to compute the actual reward signal. FP4 is used **only to rank and select which seeds
are worth regenerating**. The samples that actually enter training are the full BF16
regenerations, and these regenerations are **re-scored independently by the reward model**.
No FP4 score ever becomes a training label.

Therefore, FP4 error only affects how *informative* the selected batch is, not whether the
batch is labeled correctly. In the worst case, a noisy FP4 ranking simply yields a weaker
contrastive training signal — but it is never worse than the baseline **24-in-24** setting,
which does no selection at all. The effect of FP4 error is thus reduced contrastive
efficiency, not the amplification of incorrect preferences. We will make this distinction
explicit in the revision.

To verify this structurally, we are currently running an ablation that sweeps the precision
of the first stage only (BF16 → FP8 → FP4). This monotonically increases the quantization
noise entering the selection step while leaving everything else unchanged, allowing us to
measure exactly how selection noise affects the final learned preferences. We will provide
these empirical results in the coming days.

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

<br>

| Eval reward (HPSv2) | ≈ 3 GPU-hours | ≈ 11 GPU-hours |
|---|---|---|
| Online DPO baseline | 0.3142 | 0.3163 |
| **Online DPO + Sol-RL** | **0.3299** | **0.3317** |

The strategy transfers extremely well to DPO, exactly as the reviewer anticipated: under
the same GPU-hour budget, Sol-RL consistently beats the baseline.

On **why a group-relative objective is nonetheless the better host**: it consumes a top-k
and bottom-k *set* and weights each sample by its normalized advantage, meaning individual
reward-model errors are averaged out over the group. DPO consumes exactly one pair,
concentrating whatever error the reward model makes at the extremes into that single update.
Group-relative objectives are where wide-pool selection is most robust against noise —
which is why we adopted one, not because the mechanism is unavailable elsewhere.
PPO-style methods whose advantage comes from a learned value function rather than
intra-group ranking would benefit less directly, though the low-precision rollout could
still serve as a candidate pre-filter.

---

## On Significance

We note the written review is largely positive while Significance is scored 2, so we
address that directly. In our 24-in-96 setting the rollout stage generates 4608 images per
epoch while only 1152 reach the optimizer — rollout, not optimization, is where the compute
goes, and that ratio *grows* with pool size. It is therefore the binding constraint on
scaling diffusion RL rather than an incidental cost, and the results above show the
constraint can be removed without changing the objective, the model or the reward, across
three different objectives. We would welcome the reviewer's view on what further evidence
would best speak to significance.

We thank the reviewer again and are happy to run further experiments during the
discussion period.
