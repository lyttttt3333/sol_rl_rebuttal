# Response to Reviewer aSZi

We thank the reviewer for the positive assessment of our method's novelty, solid empirical
validation, and algorithm-hardware co-design. We agree with the weaknesses you
pointed out, precisely targeting the boundary conditions of our empirical claims
and theoretical framework. In response to this constructive critique, we provide concrete
explanations below, supported by new ablation studies on quantization formats and clarification
of the theoretical lower bound.

---

## W1. Generalizability of the ranking-preservation assumption

We appreciate the reviewer acknowledging that the paper provides strong empirical
evidence across the tested models, and agree that generalization beyond them is not
guaranteed. We have extended the evidence along the axes most likely to break the
assumption, and we scope the claim explicitly where it does not hold.

**More complex / different reward functions.** We measure the quantity the method actually
depends on: from a pool of 96 candidates we take the FP4 proxy's best-12 and worst-12, and
report where those same seeds rank under the high-precision (BF16) reward, averaged over the
12. A perfect proxy gives **6.5 / 90.5**; one with **no information** gives **48.5 / 48.5**.
Protocol: 256 prompts × 96 seeds.

| Reward | Type | FLUX.1 | SANA | SD3.5-M |
|---|---|---|---|---|
| HPSv2 | human preference | **8.4 / 87.4** | **8.7 / 87.4** | **8.6 / 83.9** |
| PickScore | human preference | 8.6 / 87.2 | 9.8 / 86.0 | 12.0 / 78.9 |
| ImageReward | semantic + preference | 10.1 / 86.0 | 10.7 / 85.1 | 14.2 / 74.6 |
| Aesthetic | aesthetic | 9.3 / 86.7 | 11.7 / 84.4 | 19.1 / 67.1 |
| OCR | clarity-based | 13.2 / 82.8 | 20.7 / 73.2 | 15.0 / 72.9 |

We fully acknowledge that ranking preservation varies depending on the specific combination
of base model and reward model. For example, the assumption does weaken for OCR and on SD3.5-M. 
However, this degradation is **bounded**. Even in the weakest cell (SD3.5-M / Aesthetic),
the proxy still separates its selected top and bottom sets compared to an uninformative random proxy **48.5 / 48.5**. Therefore, regardless of the model-reward
combination, the FP4 proxy consistently introduces enough information to maintain useful
variance among candidates. This ensures a **reliable, high-contrast learning signal**
that remains highly effective at accelerating the convergence of GRPO.

**Video generation.** We agree this is an important next step. The underlying
mechanism is theoretically not image-specific. To verify whether this proxy empirically extends
to video generation, we are currently running ranking-consistency experiments on video
models and will share the results in the coming days.

---

## W2. No comparison against a different quantization format (e.g. FP8)

We agree this comparison is the right way to isolate the effect of the aggressive FP4 format
from the broader two-stage framework. The short answer is: FP8 is a perfectly usable
operating point and preserves ranking slightly better than FP4, but because our framework is
highly tolerant to selection noise, FP4 allows us to extract the maximum throughput
advantage without sacrificing final performance.

**(1) Ranking preservation.** We first measure how the precision drop affects selection
fidelity (SD3.5-M, HPSv2, 96 candidates, selecting top-12 and bottom-12). We report the true
BF16 rank of the candidates selected by each proxy:

| Proxy Precision | True BF16 Rank of Top-12 Picks | True BF16 Rank of Bottom-12 Picks |
|---|---|---|
| BF16 (Perfect) | 6.5 | 90.5 |
| FP8 | 7.7 | 87.8 |
| FP4 (Ours) | 8.6 | 83.9 |
| Random (No Info) | 48.5 | 48.5 |

As expected, FP8 preserves ranking slightly better than FP4.

**(2) End-to-end performance under equal budget.** Because the two-stage framework
insulates the gradients from the proxy's exact output, the pipeline easily tolerates
FP4's slight ranking degradation. Consequently, under fixed computational budgets (in GPU
hours), the faster FP4 proxy can execute more rollouts and scale the search pool more
aggressively than FP8, converting that throughput advantage directly into higher final
eval rewards:

| Method | 50 GPU-hours | 100 GPU-hours | 150 GPU-hours |
|---|---|---|---|
| BF16 Baseline | 0.293 | 0.298 | 0.301 |
| FP8 Sol-RL | 0.295 | 0.305 | 0.311 |
| FP4 Sol-RL (Ours) | **0.301** | **0.309** | **0.312** |

This comparison clarifies the specific advantage of FP4: while FP8 is a viable
middle ground, FP4 safely maximizes the search-pool scaling that the two-stage framework
enables, yielding the best compute-to-reward Pareto frontier. We will add this analysis to
the revision.

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
