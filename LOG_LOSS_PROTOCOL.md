# Log-loss robustness protocol

## Question

At a stable-class measurement prefix, how much can skipping the remaining observations reduce the probability assigned to the class that actually occurs? The earlier certificate used excess Brier loss. This follow-up uses excess natural log loss, which measures the log of the ratio between the full and stopped probabilities assigned to the realised class.

Freeze the same nominal guide-conditioned, accuracy-optimal two-read policy and its maximal class-stable frontier. Use the nominal decoder for all policies, including under shift. A replacement must choose the same class as every full descendant, including under the existing lowest-index tie convention. For each eligible prefix, compare the nominal coarse Bayes posterior, the unweighted mean of full descendants, the previous Brier minimax forecast and the log minimax forecast. Give all four identical prefixes and acquisitions for the main comparison. The full two-read reference is also evaluated.

## Endpoints and comparators

The primary endpoint is maximum pathwise excess log loss across all eligible prefixes, full descendants and realised classes. The prespecified practical gate is a reduction of at least 15% relative to the best of Bayes, uniform and Brier minimax, with identical class and acquisition on all complete reference paths. Check the sharp lower bound by an independent feasibility argument and verify the attaining pair. Compare the equivalent probability ratio as well as raw nats. A failure of the gate will be reported as such.

At secondary local caps of 1.25 and 1.50 on the full-to-stopped probability ratio, a method uses its replacement at an eligible prefix only if every full descendant and realised class respects the cap. Otherwise it follows the frozen remaining schedule. Report read savings, exact class agreement, average log loss and the maximum realised ratio. These caps correspond to limiting the worst relative decrease in the realised-class probability to 20% or one third.

Use a separately optimised precommitted log-loss comparator at the common mean read count. The comparator may use the free guide, choose any 0-, 1- or 2-read schedule, and randomise after the guide. It picks the exact best schedule at each guide and read count by expected nominal log loss. Optimise the mixture across those schedules at the exact common nominal read count. It may change the reference label and uses the nominal full Bayesian posterior for its chosen schedule. Freeze that mixture under shift and report any cost mismatch alongside log loss and accuracy.

## Evaluation families

Use exact probability traversal on each law, then independently sample 8,192 common observation tapes per law with seeds 606821 through 606828 in the fixed order below. All methods see the same signal, observations and endpoint for each tape. The precommitted mixture uses separate draws from seeds 606921 through 606928. All overrides start from the nominal law, and the read schedule and policies never see the scoring law.

| Family | Overrides |
|---|---|
| Nominal | None |
| Moderate attenuation | reliability (.69,.22), erasure (.09,.29) |
| Moderate guide drift | guide flip .22 |
| Faster irregular clock | phase increments (.06,.61,.33) |
| Lower-quality persistence | high-to-high .90, low-to-high .12, erasure (.08,.44) |
| Combined shifts | phase increments (.20,.52,.28), reliability (.67,.24), erasure (.12,.33), guide flip .23 |
| Very noisy guide and sensor | reliability (.35,.09), erasure (.20,.52), guide flip .43 |
| Polarity and guide shift | reliability (−.36,−.12), guide flip .32 |

Report frozen-policy accuracy, mean attempted reads, log loss and excess over the full reference for every law and method. For Monte Carlo validation compute an unadjusted paired 95% interval for the log-loss difference. Exact versus sampled differences must be within five standard errors as an implementation check. The analytic pathwise certificate must hold regardless of sampling. Label checks compare policies with the reference on all enumerated paths and all simulated tapes. Condition on the synthetic laws in interpretation. The stress families do not constitute independent experimental settings for generalising to physical data.
