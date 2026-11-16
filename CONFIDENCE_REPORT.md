# A sharp probability guarantee for skipping a reading

At the same measurement frontier, the minimax forecast cut the sharp worst-case excess Brier loss from **0.57749 to 0.22478**, a **61.08% reduction**. It preserved the reference class on every path and kept the mean read count at **1.6492**, down **17.54%** from two attempted reads. The nominal Bayes forecast still had better average Brier loss. The difference matters when the downstream use needs probabilities and the signal law can change.

This is a passive synthetic sensing experiment for the question behind Petra Fischer’s Bristol project. The endpoint is the sector of a simulated rotating signal. Wodeyar et al.’s state-space phase-estimation work motivates reporting uncertainty with the phase estimate. Our earlier study established when a class can be preserved. Here the loss of forecast information is measured with the multiclass Brier score. [Bristol research and advert notes](PAPER_NOTES.md) · [Original class certificate](README.md#why-the-certificate-works) · [State-space phase estimation](https://elifesciences.org/articles/68803) · [Proper scoring rules](https://doi.org/10.1198/016214506000001437) · [Frozen confidence protocol](CONFIDENCE_PROTOCOL.md).

## A two-minute example

A final reading can yield either binary forecast (.60, .40) or (.90, .10). Both predict the first class. Equal branch weights give the coarse Bayes forecast (.75, .25), whose worst Brier excess over taking the reading is **0.405**. When the branch and class can follow any joint law, the optimal report is **(.675, .325)**. Its worst excess is **0.19125**, attained by two possible full-forecast and class pairs. The label stays the same. With a .20 cap on the score penalty, this example can skip the reading using the minimax forecast. The rational fixture checks the bound and independently finds the same replacement.

## The exact certificate

Fix a history and a frozen tree. Let the descendant probability vectors be \(q_z\), the stopped forecast be \(p\), and the realised endpoint class be \(y\). Use multiclass Brier loss \(L(p,y)=\|p-e_y\|_2^2\). Subtracting the full loss gives

$$
L(p,y)-L(q_z,y)
=\|p\|_2^2-2p_y+2q_{z,y}-\|q_z\|_2^2 .
$$

Let \(d_y=\max_z(2q_{z,y}-\|q_z\|_2^2)\). The maximum excess is

$$
M(p)=\|p\|_2^2+\max_y(d_y-2p_y).
$$

This bounds expected excess under any joint law of the unobserved reading and class on these outcomes. An adversary can concentrate on a maximising pair, making it sharp. The guarantee is relative to the frozen full forecast.

Let \(p^*\) be the Euclidean projection of \(d/2\) onto the probability simplex. The projection has coordinates \(p^*_y=\max(d_y/2-\tau,0)\), with the scalar \(\tau\) chosen so the coordinates sum to one. Every positive coordinate reaches the maximum \(d_y-2p^*_y=2\tau\). For any competing \(p\), weighting the maximum terms by \(p^*\) yields

$$
M(p)\;\geq\;\|p\|_2^2+(p^*)^\top(d-2p)
=M(p^*)+\|p-p^*\|_2^2 .
$$

This establishes uniqueness and an explicit optimality gap. If every \(q_z\) selects the same class \(c\), each term defining \(d_c\) dominates its counterpart for any other class. The projection preserves that order. With lowest-index tie resolution, every smaller index is strictly behind \(c\) in each descendant and thus in \(d\). The leading projected coordinate is positive, so clipping cannot introduce a lower-index tie. The replacement therefore preserves exactly the class chosen by the full tree.

## Frozen comparison

All stopped forecasts below use the same two eligible histories of the exact accuracy-optimal guide-conditioned reference. Reference labels agree on all **18** complete observation paths. The schedule comparator can choose any combination of up to two times and axes after the guide. It randomises across counts to minimise nominal expected Brier at the exact matched mean.

| Nominal policy | Attempted reads | Accuracy | Average Brier | Sharp maximum excess |
|---|---:|---:|---:|---:|
| Full two-read reference | 2.0000 | 58.368% | .55471 | 0 |
| Nominal Bayes, stable-class stop | 1.6492 | 58.368% | .57228 | .57749 |
| Uniform descendant mean, same stop | 1.6492 | 58.368% | .57400 | .44003 |
| Minimax forecast, same stop | 1.6492 | 58.368% | .58406 | .22478 |
| Best precommitted schedule, exact mean budget | 1.6492 | 57.682% | .57014 | Unrestricted class |

The strong fixed comparator beat the minimax policy on average Brier and lost 0.687 percentage points in nominal accuracy. It is designed to optimise average Brier and can change the reference labels. On the same stopped histories, the Bayes forecast is the nominal expectation optimum. The minimax forecast spent .01178 average nominal Brier to gain the lower worst-case penalty.

At the preregistered **.20** per-window cap, none of the three forecasts could skip a reading in this simulator. At the prespecified **.30** sensitivity cap, the minimax forecast preserved the full 17.54% nominal saving, and the two alternatives still needed both reads. The threshold is a study design choice stated in raw multiclass Brier units.

| Fresh synthetic test law | Average reads after stopping | Bayes mean excess | Minimax mean excess |
|---|---:|---:|---:|
| Nominal | 1.6492 | +.01756 | +.02934 |
| Attenuation and noise | 1.6930 | +.01035 | +.00743 |
| Drifting clock | 1.6119 | +.01399 | +.02311 |
| Bursty erasure | 1.7448 | +.01165 | +.02342 |
| Reversed projection | 1.6317 | −.03458 | −.09381 |
| Independent endpoint | 1.6492 | −.01756 | −.03510 |

All **196,608** fresh paired passive windows kept the exact reference label under the minimax stopping policy. The worst-case certificate was checked by enumeration, including outcomes with low nominal probability. Exact model expectations and paired simulation intervals are saved. In the independent-endpoint and reversed-projection laws, the reference forecast itself is poorly calibrated. A smaller loss relative to it is a statement about the comparator. A supervised stream with externally validated endpoints could test what caps and model classes are relevant to real decisions.

## Reproduce

The simulator generates its own observations. We used Python 3.12 and NumPy 2.3.5.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 confidence.py --verify confidence_results.json
```

The replay verifies a rational binary fixture, 512 generated four-class projection tests, admissible caps, all reference paths, exact mean-budget matching and paired tape scores. [Machine-readable results](confidence_results.json).
