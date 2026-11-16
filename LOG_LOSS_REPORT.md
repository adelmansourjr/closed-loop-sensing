# How much can skipping a reading reduce the realised-class probability?

A log-loss-specific forecast cut the sharp worst-case penalty from **0.62509 to 0.33484 nats**, a **46.43% reduction** against the best of the three replacement forecasts on the same frozen stopping frontier. It matched the full classifier on every possible path and retained the **17.54%** nominal reading saving. The full schedule still had lower average nominal log loss. [Protocol](LOG_LOSS_PROTOCOL.md) · [Machine-readable results](log_loss_results.json)

This follow-up uses the same passive synthetic rotating signal as the [Brier study](CONFIDENCE_REPORT.md). Log loss is a proper score and exposes relative probability changes that squared loss may treat differently. [Gneiting and Raftery](https://doi.org/10.1198/016214506000001437)

## Sharp bound for a frozen tree

For a valid stopping prefix, let \(q_z\) denote each remaining full forecast and let \(p\) be the replacement. In the model all forecasts are strictly positive. For realised class \(y\), excess natural log loss is \(\log(q_z[y]/p[y])\). Put \(m_y = \max_z q_z[y]\). To guarantee a maximum ratio of \(R\), every coordinate must satisfy \(p[y] \ge m_y/R\). Since the coordinates sum to one, \(R \ge \sum_y m_y\). Assigning \(p[y] = m_y / \sum_j m_j\) meets every constraint exactly and attains that bound.

At a class-stable prefix, each full forecast chooses the same class \(c\). Coordinatewise maxima preserve this ranking. For any lower-index alternative \(y\), choose a descendant that attains \(m_y\). Its prediction strictly prefers \(c\) to \(y\), so \(m_c > m_y\). Ties with higher-index classes still resolve to \(c\). The optimum therefore preserves the full classifier's class under the frozen tie convention.

The replacement is Shtarkov's normalised maximum-likelihood distribution for the finite family of descendant forecasts. Here it is used at the stopping prefixes, where the common descendant class also certifies the reconstructed decision. [Barron, Rissanen and Yu](https://www.stat.berkeley.edu/~binyu/212A/papers/bry98.pdf)

For two full forecasts (.60, .40) and (.90, .10), the optimum is (9/13, 4/13), with sharp ratio 1.30. The replay checks this fixture, a tie case and 128 generated four-class stopping prefixes.

## Same readings, different forecasts

The four replacement methods stop at the same two prefixes in the uncapped comparison. The precommitted comparator can use the cheap guide and randomise among any schedule of up to two readings. It is optimised for exact nominal expected log loss at the same nominal mean read count and can change classifications.

| Policy | Nominal mean reads | Nominal accuracy | Nominal mean log loss | Worst full-to-stopped probability ratio | Worst excess log loss |
|---|---:|---:|---:|---:|---:|
| Full reference | 2.0000 | 58.368% | 1.04364 | 1.0000 | 0.00000 |
| Coarse Bayes at the common stops | 1.6492 | 58.368% | 1.07561 | 2.4156 | 0.88194 |
| Uniform mean of descendants | 1.6492 | 58.368% | 1.07853 | 1.8684 | 0.62509 |
| Brier minimax at the common stops | 1.6492 | 58.368% | 1.10003 | 1.9105 | 0.64734 |
| Log minimax at the common stops | 1.6492 | 58.368% | 1.08964 | 1.3977 | 0.33484 |
| Log-optimal precommitted mixture | 1.6492 | 57.682% | 1.07268 | Outside this certificate | Outside this certificate |

At a ratio cap of **1.25**, all four replacements used both reads. At **1.50**, only the log-minimax replacement could stop and it retained the full nominal saving. At its maximum ratio of 1.3977, its stopped forecast gives the realised class at least 71.5% of the full forecast's probability. This is a guarantee relative to the given decoder. The calibrated quality of that decoder still depends on the scoring law.

## Fresh paired simulation

Eight frozen scoring laws were evaluated by exact integration and 8,192 independently seeded paired observation tapes per law. The log-minimax policy matched every full-reference class in all **65,536** tapes. The table gives exact expected log-loss changes versus the full reference. Lower values are better. Paired uncertainty intervals for the simulation checks and all comparators are in the results file.

| Scoring law | Full-reference accuracy | Log-minimax mean reads | Log-minimax excess log loss |
|---|---:|---:|---:|
| Nominal | 58.368% | 1.6492 | +0.04601 |
| Moderate attenuation | 54.318% | 1.6640 | +0.02583 |
| Moderate guide drift | 52.505% | 1.6229 | +0.04648 |
| Faster irregular clock | 57.797% | 1.6492 | +0.03533 |
| Lower-quality persistence | 57.773% | 1.6824 | +0.04267 |
| Combined shifts | 46.036% | 1.6512 | +0.02248 |
| Very noisy guide and sensor | 32.933% | 1.6680 | −0.00404 |
| Polarity and guide shift | 22.885% | 1.6010 | −0.13592 |

The worst-case guarantee survives these changes because it compares two frozen forecasts for each observation path and endpoint. Absolute accuracy still fell as low as 22.885% under polarity and guide shift. The log-minimax method improves the worst-case comparison while paying a nominal average-loss cost of 0.01404 nats versus coarse Bayes on the same reads. Mean read budgets for the frozen precommitted comparator can differ under shift. These results apply to synthetic observations and the specified reference decoder.

## Reproduce

The experiment generates its own observations and uses Python 3.12 and NumPy 2.3.5.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 log_loss.py --verify log_loss_results.json
```
