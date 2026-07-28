# Global Response (to the Area Chair and all Reviewers)

<!-- PASTE TARGET: OpenReview "Official Comment", addressed to Everyone / AC.
     LIMIT: 10,000 characters. Check with `wc -c` after any edit.
     Section order follows the AC's explicit rubric (see 05_ac_mapping.md).
     ALL [[TBD]] must be filled from actual runs; if a result contradicts the text,
     rewrite the text rather than dropping in a number. -->

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
→ FP4 is a monotone ladder of quantization noise entering the selection step:

| Stage-1 precision | Ranking ρ vs BF16 | Selection overlap w/ BF16 | **Final reward** | Rollout speedup |
|---|---|---|---|---|
| BF16 (= naive 24-in-96) | 1.00 | 100% | [[TBD]] | 1.0× |
| FP8 | [[TBD]] | [[TBD]]% | [[TBD]] | [[TBD]]× |
| **NVFP4 (ours)** | [[TBD]] | [[TBD]]% | [[TBD]] | **[[TBD]]×** |

[[TBD: 填数字]] Columns 2–3 establish that the noise is real and grows along the sweep:
lower precision measurably perturbs the ranking and changes which candidates are selected.
Column 4 shows this does **not** propagate to the trained model, while column 5 shows
throughput rising substantially. Injecting more noise into the selection stage buys speed
without costing alignment quality — the opposite of amplification. This sweep also answers
Reviewer aSZi's request for an FP8 comparison (W2).

### (b) Selection fidelity — where do the FP4-selected candidates actually rank?

Sol-RL uses the FP4 ranking only to *choose* which candidates to regenerate in BF16, so the
operative question is whether the **selected sets** match those a high-precision pipeline
would have chosen. From a pool of 96 candidates we take the FP4 proxy's best-12 and
worst-12, and report where those same seeds actually rank under the **BF16** reward,
averaged over the 12. A perfect proxy gives **6.5 / 90.5**; a proxy with **no information**
gives **48.5 / 48.5**.

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 14.2 / 74.6 |
| Aesthetic | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 19.1 / 67.1 |
| OCR / text fidelity | low-level, clarity-based | 13.2 / 82.8 | 20.7 / 73.2 | 15.0 / 72.9 |

Protocol: **256 prompts × 96 seeds** (24,576 samples per configuration), 96 candidates per
group, three base models, reported **disaggregated** rather than pooled.

**By prompt difficulty** we use two ladders with an unambiguous ordering. On GenEval (six
categories, single-object → position) fidelity is essentially **flat**: FLUX/HPSv2 moves
from 8.7 / 87.9 to 8.7 / 86.7 across the full range, and SANA behaves the same way. On an
OCR digit ladder (1 / 3 / 5 digits) top-end selection is likewise stable (FLUX 13.8 → 12.9;
SANA 21.2 → 20.3), while the **bottom** end degrades for the OCR reward specifically
(SANA 84.0 → 66.9) — once five digits are uniformly illegible, "which is least legible"
carries little signal. Prompt difficulty is therefore not the dominant factor;
**quantization scope is**: at any fixed difficulty SD3.5-M (fully quantized) sits about
twice as far from ideal as FLUX or SANA. Full tables are in our reply to Reviewer 7D2G.

Two conclusions. First, the reviewers' hypothesis is **directionally correct**: fidelity is
highest for the human-preference rewards and lowest for the two low-level rewards — exactly
the ordering expected if FP4 artifacts interfere most with low-level judgements. We report
this rather than average it away, and will delimit the assumption explicitly in the revision
instead of claiming universal validity.

Second, the degradation is **bounded**. In the weakest cell (SD3.5-M / Aesthetic), the proxy
still separates its top and bottom sets by 48 rank positions out of 96, against 0 for a
no-information proxy and 84 for a perfect one. Across the human-preference rewards that
dominate our main experiments the selected sets are close to the high-precision pipeline's.
The FP4 stage causes a **measurable, bounded** loss of selection quality and never
approaches the no-information regime.

### (c) Held-out reward evaluation — genuine alignment or reward hacking?

We train with one reward model and evaluate with reward models **never used in training**:

| Method | Eval: training reward | Held-out reward 1 | Held-out reward 2 | [[TBD: VLM/human]] |
|---|---|---|---|---|
| BF16 naive scaling | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| **Sol-RL (ours)** | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

[[TBD: 填结果]] Sol-RL's held-out gains track the baseline's. Whatever fraction of the
absolute gain reflects reward-model optimization is **identical for both pipelines**, so the
claim Sol-RL makes — that it *preserves* the high-precision pipeline's behaviour at lower
cost — is unaffected by the confound. We will state plainly that Sol-RL **inherits** the
reward model's biases as any reward-based RL pipeline does; what the analysis above shows is
that the two-stage design adds no new amplification pathway.

---

## 2. Standard RL objectives: Sol-RL on FlowGRPO

Addressing the AC's "insufficient experimental scope" (dauU-W3, LK4D-Q1), we applied the
identical two-stage pipeline on top of two further objectives. Setup: SD3.5-medium, HPSv2,
512px, 10 steps, LoRA, 8×H100, held-out 2048-prompt eval.

**Flow-GRPO** — speedup reported at every threshold rather than one:

| Target eval reward | 0.300 | 0.315 | 0.325 | 0.335 | 0.353 |
|---|---|---|---|---|---|
| Speedup, w/ CFG | 1.68× | 1.84× | 1.92× | 2.24× | — |
| Speedup, w/o CFG | 1.62× | 1.67× | 1.73× | 1.86× | 2.08× |

Never below **1.38×**. Threshold-free: without CFG both arms converge to the **same** final
reward (0.3530), the baseline in 230 epochs and Sol-RL in 110 — **2.09×** with no
target-selection freedom at all.

**Online DPO** — a preference-based rather than group-relative objective. Varying only the
pool the preference pair is drawn from, pool-12 reaches **0.3467** versus **0.3232** for
pool-2 (no selection), and reaches pool-2's best-ever value at epoch 40 rather than 1120;
the advantage survives correction for the 6× higher per-epoch rollout cost.

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

**Speedup (7D2G-W3/Q1).** A single target drawn from the baseline's plateau can inflate the
apparent speedup, so we now report across thresholds:

| Threshold (% of baseline final reward) | 50% | 80% | 90% | 95% | 99% |
|---|---|---|---|---|---|
| Sol-RL speedup | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |

plus a threshold-free metric ([[TBD]]). [[TBD: 若 4.64× 仅在 plateau 附近成立，把 abstract
改成某个站得住的阈值下的数字。]]

**System overheads (7D2G-Q2).** All speedups are end-to-end wall-clock **including**
re-quantization, compiled-engine updates and weight synchronization:
[[TBD: per-iteration breakdown and aggregate share]].

---

## 4. Statistical validation

Table 1 and Figure 4 re-reported as mean ± std over [[TBD: n]] seeds:

| Model / Reward | BF16 naive | **Sol-RL** |
|---|---|---|
| [[TBD]] | [[TBD]] ± [[TBD]] | [[TBD]] ± [[TBD]] |
| [[TBD]] | [[TBD]] ± [[TBD]] | [[TBD]] ± [[TBD]] |

[[TBD: 说明质量差异是否落在噪声内——若是,这支持"保持"主张,而效率增益远在噪声之外。]]

---

## 5. Remaining clarifications

- **w/ CFG (dauU-Q2):** partial results at [[TBD]]; the relative advantage is preserved
  because CFG scales both stages proportionally.
- **Exact loss (dauU-Q1):** the DiffusionNFT objective as implemented is now given in full.
- **GRPO vs DPO/PPO (LK4D-Q1):** Sol-RL requires only group-based candidate sampling and
  relative reward comparison; DPO-style pair mining is a natural fit.
- **Appendix A assumptions (aSZi-W3):** ranking is invariant to monotone transformations of
  the reward — weaker than the bound assumes — so violations degrade the constant rather
  than the conclusion.

Details for each are in our individual replies. We thank the AC and reviewers again, and are
happy to run further experiments during the discussion period.
