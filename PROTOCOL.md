# Prospective protocol

## Question

In a finite rotating-signal model, can a measurement scheduler use the first noisy projection to improve terminal phase-sector classification over the best two-measurement schedule chosen after a cheap cue? An observation operator changes what is available to the estimator. The oscillator evolves independently of every observation and classifier decision.

## Model and information budget

The latent phase lies on 16 equally spaced bins. The quality state is high or low. Initial phase is uniform and quality is high with probability 0.65. A bin advances one position each tick with an independent perturbation of −1, 0 or +1 with probabilities 0.12, 0.76 and 0.12 under the nominal model. High remains high with probability 0.86, and low switches up with probability 0.23. There are three observation opportunities at ticks 0, 1 and 2. The target is the phase sector at tick 3, with sector 0 using bin 3 as its reference and three more sectors spaced four bins apart. Each sector comprises four adjacent bins, with the offsets −1, 0, 1 and 2 relative to its centre.

The free guide at tick 0 is a binary indicator of dominance of the initial horizontal over vertical component, determined by the sign of the cosine of twice the angle (horizontal on a tie) and flipped independently with probability 0.10. At each opportunity the policy skips or queries one of four binary projection axes centred on bins 0, 2, 4 and 6. The query can return positive, negative or erased. Conditional on a non-erasure, positive probability equals one half plus half the current quality coefficient times the cosine of the difference between phase and sensor axis. Coefficients are 0.82 and 0.32 for high and low quality. Erasure probabilities are 0.06 and 0.24. Query outcomes are independent conditional on the latent trajectory. The main budget is exactly two attempted reads even when a read is erased.

## Policy comparison

- **Exact adaptive Bayes.** A finite-horizon belief-state recursion maximises the probability of correctly classifying the terminal sector. The scheduler can choose both the next tick and sensor as a function of past observations. All candidates use only the frozen nominal physics model and the input history at the decision point.
- **Exact guide-conditioned fixed.** Enumerate all legal two-read schedules after the free guide. Choose the one with the best exact true-model forecast risk under the nominal law. The decoder still conditions on both queried outcomes. This is the primary, exact-action-budget comparator and captures information relevant to sensor choice already present in the guide.
- **Exact fixed before guide.** Choose both times and axes before the guide while still giving the guide and projections to the decoder.
- **Periodic.** At two reads use first and last opportunities, with axes tuned exactly under the nominal model before the guide. At one read use the middle opportunity.
- **Myopic risk and myopic entropy.** For every currently feasible time and axis for the next query, optimise its single-observation value for terminal classification accuracy or entropy, respectively. Replan after the next query. Every policy ends with the same model and maximum-posterior class decoder. Use lexicographic first among ties (otherwise a time action or schedule is freely exchangeable).

Supply guide-only and all-three-read adaptive results as budget references. An exact enumerator scores each frozen policy against its nominal law, and independently rescored against the alternate laws below. A class prior with the guide is equally available to all policies. No policy sees the held-out simulator regime or its true state.

## Locked test families

The nominal family is primary. Stress tests change only the stated component, preserve the motion target and observation budget, and keep all policies and decoder probabilities frozen.

| Family | Change from nominal |
|---|---|
| Attenuation | Quality coefficients 0.44 and 0.13 |
| Persistent loss | Erasures 0.12 and 0.65. Quality transition probabilities high-to-high 0.94 and low-to-high 0.10. |
| Clock drift | Phase perturbation probabilities 0.10, 0.46 and 0.44 for −1, 0 and +1. |
| Guide drift | Flip probability 0.40 |
| Uniform final reset | On the final transition sample phase uniformly, independent of the prior path. Mathematically all classifiers using earlier observations have 25% expected accuracy. |

Simulate 32,768 independent paired windows in each family using a fixed generator seed 544002 plus family position in the table, with position 0 for nominal. Every policy reads the same underlying per-time per-axis measurement tape, changes neither tape nor dynamics, and must make exactly its authorised number of attempted reads for every path. Each row of this simulator is an independent assessment unit. Report the nominal adaptive-minus-guide-conditioned-fixed difference, a 95% paired bootstrap interval over windows using 4,000 replicates and seed 8406 plus family position, and the exact enumerated difference. Call it practically positive only if at least 0.5 percentage points by exact enumeration and with a positive lower simulation interval. Secondary outcomes are stress-family differences, erased query counts, chance-corrected accuracy where useful, and multi-class Brier score. Repeat the nominal sensor selection at one read to check that conditioning the fixed schedule on the free guide achieves exactly the same value as full adaptivity when there is no information between queries. The performance tables condition on the chosen synthetic laws. They quantify simulator sampling variation.

## Invariants and exact witnesses

Check every transition and emission row for nonnegativity and unit mass. Compare nominal design value to independently rescored exact policy value. Compare nominal exact and Monte Carlo accuracies within five binomial standard errors. Re-evaluate each family with shared tapes and report exact results under every shift. Verify mass conservation, two reads per policy path, the guide-conditioned one-read equality, and exact uniform-reset chance accuracy.

For an analytic lower-dimensional witness, let a hidden fair binary context V name the only informative sensor and a hidden independent fair bit Z be the class. The free guide reveals V. With one query, sensor V predicts Z with accuracy 1−e and its companion is an independent fair coin. A pre-guide sensor choice has optimal accuracy 3/4−e/2 and a guide-conditioned selection achieves 1−e. Verify both at e=1/5 by enumeration. This fixture isolates the value of conditioning on the guide; the main comparison already gives the guide to schedule selection.
