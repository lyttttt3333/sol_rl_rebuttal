# AC Meta-Review → Rebuttal Mapping (Submission 9615)

AC cRNh, 19 Jul 2026 (modified 24 Jul 2026). This document maps the AC's stated
priorities onto reviewer points, our planned experiments, and the draft replies.

---

## 1. The AC's explicit instruction (verbatim)

> "To positively influence the final decision, the authors should **focus the rebuttal on
> addressing the reward model vulnerabilities and the impact of FP4 noise**. Additionally,
> answering all clarification questions, dialing back overstated claims, and providing the
> requested experimental analysis (including standard RL objectives and statistical
> variance) will be critical."

This is an unusually explicit rubric. Read literally, it gives a **ranked** list:

| Rank | AC's requirement | Signal strength |
|---|---|---|
| **1** | Reward-model vulnerabilities + **impact of FP4 noise** | "**focus** the rebuttal on" — named first and singled out |
| 2 | Answer **all** clarification questions | "critical" |
| 3 | **Dial back overstated claims** | "critical" |
| 4 | Standard RL objectives (FlowGRPO) | "critical", grouped under "Additionally" |
| 5 | Statistical variance (error bars) | "critical", grouped under "Additionally" |

Also decisive, from the Primary Concerns section:

> "**Further analysis of the impact of the noise introduced by FP4 rollout is necessary.**"

Note the precise wording: *noise introduced by FP4 rollout* — i.e. the AC wants the
**FP4 stage's own contribution to ranking error isolated**, not a generic robustness
study. This is exactly the selection-agreement measurement (FP4-selected vs BF16-selected
sets), and it confirms that injecting synthetic noise would have answered the wrong
question.

---

## 2. PRIORITY CHANGE vs. our previous plan

Our earlier plan (`PLAN.md`) had **P0.1 = FlowGRPO** as the single top-priority job. The
AC's meta-review reorders this:

| | Previous plan | After AC meta-review |
|---|---|---|
| **#1** | FlowGRPO generalization | **Reward-model vulnerability + FP4 noise isolation** |
| #2 | FP8 comparison | FlowGRPO (standard RL objectives) |
| #3 | Reward-noise sensitivity | Dial back claims (free, text-only) + error bars |
| #4 | w/ CFG | Answer all remaining clarification Qs |

**Why this matters:** the AC decides the outcome, and has told us in plain language what
they will weigh most. The #1 item is also **mostly inference-only** (see §4), so it is
both the highest-value and the fastest to deliver. FlowGRPO remains critical but is
explicitly in the "Additionally" tier.

---

## 3. Full correspondence table

| AC point | Origin (reviewers) | Our response | Draft location | Cost |
|---|---|---|---|---|
| **Fragile FP4 ranking assumption**; breaks down for **low-level features** where FP4 introduces severe artifacts | LK4D-W1, dauU-W2 | Ranking consistency by **reward type** (semantic vs low-level: Aesthetic, OCR), by base model, by prompt difficulty | Global §1; dauU-W2; LK4D-W1; aSZi-W1 | Inference only |
| **FP4 selection could amplify incorrect preferences**; "further analysis of the noise introduced by FP4 rollout is necessary" | LK4D-W3/Q2 | **Selection agreement**: top/bottom sets chosen from FP4 ranking vs from BF16 ranking → isolates FP4's own error contribution | Global §3(a); LK4D-W3 | Inference only |
| **Amplify noise or lead to reward hacking rather than genuine alignment** | 7D2G-W1, LK4D-W3 | **Held-out reward evaluation**: train on reward A, evaluate on unseen rewards B/C; compare Sol-RL vs BF16 baseline degradation | Global §3(b); 7D2G-W1; LK4D-W3 | Eval on existing ckpts |
| **Novelty: identifying the bottleneck is not a contribution** | dauU-W1 | Reframe as **motivation**; restate 3 contributions | dauU-W1; Global §6 | Free (text) |
| **Modest improvements, no error bars** | 7D2G-W2 | mean ± std over n seeds; **reframe claim as compute-efficient preservation** | 7D2G-W2 | Cheap (seeds) |
| **Speedup may be overstated** (artifact of target-reward selection) | 7D2G-W3/Q1 | Speedup at **multiple thresholds** (50/80/90/95/99%) + threshold-free metric; revise headline number if needed | 7D2G-W3/Q1; Global §5 | Re-analysis |
| **Missing details**: # prompts, robustness across difficulty and reward models | 7D2G-W4/Q3 | Full protocol stated; results **disaggregated** per model / per difficulty | 7D2G-W4/Q3; Global §1 | Re-analysis |
| **Too entangled with a specific objective; lacks standard RL objectives** | dauU-W3, LK4D-Q1 | **Sol-RL on FlowGRPO** | Global §2; dauU-W3; LK4D-Q1 | **Training** |
| "Answering **all** clarification questions" | dauU-Q1/Q2, LK4D-Q1/Q2, 7D2G-Q1/Q2/Q3 | Every Q answered explicitly in per-reviewer replies | 01–04 | Mixed |

---

## 4. Why the AC's #1 is cheap — and should be launched immediately

The AC's top priority decomposes into three measurements, **two of which require no
training at all**:

1. **Ranking consistency by reward type** — inference + correlation on existing BoN
   generations. Low-level rewards (Aesthetic, OCR) already exist in our sweep alongside
   semantic ones. *No training.*
2. **Selection agreement (FP4 vs BF16 chosen sets)** — computed on the same existing
   candidate pools. *No training.*
3. **Held-out reward evaluation** — score existing checkpoints with reward models not
   used in training. *No training, only eval.*

So the highest-weighted item in the AC's rubric is largely a **re-analysis of data we
already have**. It should be started before, or in parallel with, the FlowGRPO run.

---

## 5. What the AC did NOT mention (and what follows)

The meta-review is dated 19 Jul; **Reviewer aSZi's review was submitted 23 Jul**. The AC
writes "a shared concern across all reviewers" but then lists only **(LK4D, dauU, 7D2G)**
— suggesting the metareview was largely written before aSZi's review landed.

Consequences:

- **aSZi's points are absent from the AC's rubric**: the FP8 comparison (aSZi-W2) and the
  Appendix A assumption discussion (aSZi-W3) carry **no AC weight**. FP8 is still worth
  doing (the script already exists, near-zero cost) but should not displace the AC's #1.
- **aSZi is our most positive reviewer** ("novelty is strong", "clear algorithm-hardware
  co-design", Significance 3, Confidence 4) and their assessment does **not** appear to be
  reflected in the AC's summary — which is built around the novelty and significance
  doubts of dauU and LK4D.
- **Action:** the Global Response should open by noting that **all four** reviewers rate
  the paper 4, and briefly surface aSZi's assessment, so the AC's final read incorporates
  the fourth review.
- Also unmentioned by the AC: **w/ CFG** (dauU-Q2) and the **user study** (7D2G-W1, though
  "reward hacking" is referenced). Both still need answers under "answer all clarification
  questions", but neither needs to be a headline.

---

## 6. Recommended revised launch order

| Order | Task | Type | Serves |
|---|---|---|---|
| **1** | Selection agreement (FP4-chosen vs BF16-chosen sets) | Inference | **AC #1** |
| **2** | Ranking consistency by reward type × model × difficulty | Inference | **AC #1**, AC "missing details" |
| **3** | Held-out reward evaluation | Eval | **AC #1** (reward hacking) |
| **4** | **FlowGRPO** Sol-RL run | **Training** | AC #4 (standard RL objectives) |
| 5 | Error bars (multi-seed) | Training (cheap) | AC #5 |
| 6 | Multi-threshold speedup + overhead breakdown | Re-analysis | AC #3 |
| 7 | Text revisions: novelty reframing, claim dial-back | Free | AC #3 |
| 8 | FP8 comparison | Training (script exists) | aSZi only — no AC weight |
| 9 | w/ CFG partial | Training | dauU-Q2 only |

Items 1–3 and 6–7 can proceed while the FlowGRPO job (4) queues and runs.
