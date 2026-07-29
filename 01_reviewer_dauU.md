# Response to Reviewer dauU

We thank the reviewer for the careful review and for recognizing that our FP4 reward proxy
maintains RL performance while significantly accelerating training, supported by "solid
experimental validation" across representative models. We also deeply appreciate the
constructive critique: your points have directly helped us restate our contributions more
precisely (W1), delimit the boundary of the FP4 assumption (W2), and prove that Sol-RL is
algorithm-agnostic (W3). We address each point below.

---

## W1. The first contribution is an overclaim

**We fully agree, and will revise it accordingly.** 
The computational bottleneck of RL rollouts and the degradation caused by low-precision
training are established observations, particularly in quantized RL for LLMs. However,
quantization in diffusion RL remains comparatively underexplored. We thank the reviewer for
this correction and will revise our framing to present the established observations as
motivation and state our contribution more precisely.

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

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 11.2 / 84.6 |
| **Aesthetic** | low-level aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 15.1 / 79.1 |
| **OCR / text fidelity** | low-level, clarity-based | 13.2 / 82.8 | 12.7 / 83.2 | 13.0 / 80.9 |
| *Perfect proxy (reference)* | | *6.5 / 90.5* | *6.5 / 90.5* | *6.5 / 90.5* |
| *No-information proxy (reference)* | | *48.5 / 48.5* | *48.5 / 48.5* | *48.5 / 48.5* |

The results support the assumption for both reward families we tested. For
aesthetic and clarity-based rewards, fidelity is somewhat lower, as expected because text and low-level artifacts are more sensitive to early FP4 degradation; nevertheless, the proxy still carries substantial information of contrastive seeds and is far from random.

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

This is a fair point, and we have now run the full ablation **with CFG
enabled** (SD3.5-medium, HPSv2, otherwise identical settings):

| Target eval reward | 0.300 | 0.315 | 0.325 | 0.335 | 0.353 |
|---|---|---|---|---|---|
| Sol-RL speedup, **w/ CFG** | 1.68× | 1.84× | 1.92× | 2.24× | — |
| Sol-RL speedup, **w/o CFG** | 1.62× | 1.67× | 1.73× | 1.86× | 2.08× |

**The benefit does not depend on the CFG setting**: across all reported finite entries, the
speedup ranges from 1.62× to 2.24×. The reported gains are therefore not an artifact of the
w/o-CFG configuration.

---

We thank the reviewer again -- W1 and W3 in particular have materially improved the
paper. We would be glad to run additional experiments during the discussion period and will include the coresponding experiment in our revised version.
