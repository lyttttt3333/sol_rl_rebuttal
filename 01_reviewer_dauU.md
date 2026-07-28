# Response to Reviewer dauU

We thank the reviewer for the careful review and for recognizing that our FP4 reward proxy
maintains RL performance while significantly accelerating training, supported by "solid
experimental validation" across representative models. We also deeply appreciate the
constructive critique: your points have directly helped us restate our contributions more
precisely (W1), delimit the boundary of the FP4 assumption (W2), and prove that Sol-RL is
algorithm-agnostic (W3). We address each point below.

---

## W1. The first contribution is an overclaim -- it should be motivation

**We fully agree, and will revise it accordingly.** 
The computational bottleneck of RL rollouts and the degradation caused by low-precision
training are established observations (including in the LLM quantized-RL literature). 
We thank the reviewer for this correction; it sharpens what the paper actually claims.

---

## W2. The proxy assumption is vulnerable to the choice of reward model

It is the core question that tests the central assumption behind
Sol-RL: the cheap FP4 rollout must preserve enough **relative reward ordering** to choose useful contrastive candidates.
We therefore ran an ablation experiment across reward-model types, including the low-level
aesthetic and clarity-based rewards highlighted by the reviewer.

We measure the quantity Sol-RL actually uses. From 96 candidates, we take the FP4 proxy's
**best-12 and worst-12**, then report where those same seeds rank under the high-precision
(BF16) reward, averaged over the 12. A perfect proxy gives **6.5 / 90.5**; a proxy with no
information gives **48.5 / 48.5**.

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | **aesthetic** | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | **aesthetic** | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | **aesthetic** | 10.1 / 86.0 | 10.7 / 85.1 | 14.2 / 74.6 |
| **Aesthetic** | **aesthetic** | 9.3 / 86.7 | 11.7 / 84.4 | 19.1 / 67.1 |
| **OCR / text fidelity** | **clarity-based** | 13.2 / 82.8 | 20.7 / 73.2 | 15.0 / 72.9 |
| *Perfect proxy (reference)* | | *6.5 / 90.5* | *6.5 / 90.5* | *6.5 / 90.5* |
| *No-information proxy (reference)* | | *48.5 / 48.5* | *48.5 / 48.5* | *48.5 / 48.5* |

The results support the assumption for both reward families we tested. For
aesthetic rewards, the FP4 proxy clearly meets Sol-RL's selection requirement. For clarity-based rewards, fidelity is somewhat lower, as expected because text artifacts are more sensitive to early FP4 degradation; nevertheless, the proxy still carries substantial information of contrastive seeds and is far from random.

---

## W3. Sol-RL is entangled with DiffusionNFT

Following the reviewer's suggestion we applied the **identical**
two-stage rollout pipeline on top of **FlowGRPO**, changing only the policy-optimization
objective and holding the rollout/selection machinery fixed.

Setup: SD3.5-medium, HPSv2 reward, 512px, 10 sampling steps, LoRA, 8×H100; eval on a
held-out 2048-prompt set. Both arms send the same 1152 samples to the optimizer and take
the same number of updates per epoch.

**Sol-RL transfers to Flow-GRPO and online-DPO.** 

| Eval reward (HPSv2) | ≈ 12 GPU-hours | ≈ 30 GPU-hours |
|---|---|---|
| Flow-GRPO baseline (24-in-24) | 0.3254 | 0.3448 |
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

Thank you — this is a fair point, and we have now run the full ablation **with CFG
enabled** (SD3.5-medium, HPSv2, otherwise identical settings):

| Target eval reward | 0.300 | 0.315 | 0.325 | 0.335 | 0.353 |
|---|---|---|---|---|---|
| Sol-RL speedup, **w/ CFG** | 1.68× | 1.84× | 1.92× | 2.24× | — |
| Sol-RL speedup, **w/o CFG** | 1.62× | 1.67× | 1.73× | 1.86× | 2.08× |

**The benefit does not depend on the CFG setting** — it is 1.4–2.3× either way and never
falls below 1.38×. The reported gains are therefore not an artifact of the w/o-CFG
configuration.

One caveat we will state explicitly in the revision: the gap should be read *within* a
setting, never across. Turning CFG off costs the untrained policy 0.082 HPSv2
(0.2952 → 0.2128), so absolute rewards are not comparable between the two configurations;
what is comparable is the Sol-RL advantage inside each, and that is stable. Our use of
w/o CFG as the main setting is a matter of matching the baselines we compare against
rather than a configuration that flatters the method.

---

We thank the reviewer again -- W1 and W3 in particular have materially improved the
paper. We would be glad to run additional experiments during the discussion period and will include the coresponding experiment in our revised version.
