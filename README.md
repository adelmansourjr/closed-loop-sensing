# When can a rotating-signal classifier skip a measurement?

**Start with the [log-loss experiment and proof](LOG_LOSS_REPORT.md).** Run the replays with Python 3.12 and the [pinned NumPy dependency](requirements.txt).

A paper-linked, observation-only project for Petra Fischer, Anthony Harris and Ehsan Jafari's Bristol PhD on brain-activity-based vibrotactile stimulation. The computational problem is whether a noisy rotating signal needs another observation to classify its terminal sector. The simulation uses a 32-state synthetic signal. Measurement selection changes the information available to the classifier and leaves the signal dynamics independent of every decision. [Research and advert notes](PAPER_NOTES.md).

## Results

The first study compared exact adaptive acquisition with the exact best two-read schedule chosen after a cheap axis cue. Both attained **58.3684%** nominal accuracy. On the shared simulated windows, the two schedules were correct on exactly the same draws under every predeclared shift. A weaker schedule required to choose both reads before the cue attained **56.4917%**. At one read, the best cue-conditioned schedule reached **56.2756%** versus **51.3100%** for the exact best pre-cue schedule. The frozen two-read adaptivity target was missed. [Protocol](PROTOCOL.md) · [Complete results](results.json).

That result led to a separate question. If all possible outcomes of a remaining reading leave the terminal class the same, can the classifier use one fewer read while preserving every decision? Under nominal noise, the follow-up cut mean attempted reads from **2 to 1.6492**, a **17.54% saving**, with the same class on every possible observation history. At the same nominal mean budget, the strongest randomised fixed scheduler available after the cue reached **57.682%**, a **0.686 percentage-point** gap to early stopping. [Follow-up protocol](PRUNING_PROTOCOL.md) · [Complete results](pruning_results.json).

| Frozen-law test | Full schedule accuracy | Certified accuracy | Certified mean reads | Attempts saved |
|---|---:|---:|---:|---:|
| Nominal | 58.368% | 58.368% | 1.6492 | 17.54% |
| Attenuation | 47.927% | 47.927% | 1.6492 | 17.54% |
| Persistent loss | 57.165% | 57.165% | 1.7222 | 13.89% |
| Clock drift | 54.712% | 54.712% | 1.6492 | 17.54% |
| Guide drift | 43.709% | 43.709% | 1.5834 | 20.83% |
| Uniform final reset | 25.000% | 25.000% | 1.6492 | 17.54% |
| Severe erasure | 52.510% | 52.510% | 1.8256 | 8.72% |
| Polarity mixture | 43.269% | 43.269% | 1.6492 | 17.54% |

The 262,144 fresh paired windows in the follow-up had zero class disagreements. There is a probability-quality cost in the nominal family. Multiclass Brier score was **0.55471** with both reads and **0.57228** after pruning, where lower is better. The matched-mean fixed comparator scored **0.57014**. The certificate preserves the classification made by the frozen decoder. A low accuracy shared by both systems, as under guide drift or polarity reversal, remains low.

## Why the certificate works

The first study evaluates every possible reading branch of each candidate sensor and time. It propagates an unnormalised state-probability vector, partitions on the outcome and scores the largest terminal class mass. Backward induction gives the optimal attainable expected classification accuracy for the specified model and budget. The fixed comparator exhausts the same times and axes while requiring the schedule to be fixed after the cue.

The follow-up prunes a policy subtree when every reachable leaf uses the same class. Under the nominal model, the posterior with fewer reads is a probability-weighted average of the original conditional posteriors. If one class is top-ranked in every term, it is also top-ranked in their average. With a fixed lowest-index tie rule, that class strictly outranks every lower-index class at each leaf where the policy chose it. Averaging preserves the tie rule as well. Once the tree is frozen, this gives class agreement on every observational path even when the scoring distribution changes. Erasure is an explicit branch. In the nominal fixed tree, a negative-axis cue occurs with probability 0.40 and accepts early stopping exactly when the first read is observed, probability 0.877. Their product 0.3508 is the exact expected number of skipped attempts.

Proper scoring explains the tradeoff. Let p_full be the posterior after the original schedule and p_stop the posterior after early stopping. Under the nominal law, p_stop is the conditional mean of p_full given the retained information. The mean squared difference between these posteriors is the increase in expected Brier loss. A read can sharpen the distribution even when no possible outcome changes the top class.

The matched-mean comparator chooses optimal 0-, 1- and 2-read schedules for each guide and searches all feasible extreme points of their randomized mixture. It can condition its full acquisition plan on the cheap guide. The nominal mean budget is the same to numerical tolerance. This budget match applies to the nominal law. Under some shifts the stopping policy's mean cost changes with the observed histories, while the comparator remains the frozen deployment mixture.

## Follow-up on probability quality

A minimax replacement reduced the sharp worst-case excess Brier loss from **.57749 to .22478** at the identical stopping frontier. The result preserves every class and the full 17.54% nominal reading reduction. The nominal Bayes replacement still scores better on average. [Confidence certificate and replay](CONFIDENCE_REPORT.md).

A later log-loss comparison reduced the sharp excess from **0.62509 to 0.33484 nats** against the best Brier, uniform or Bayes replacement on the same frontier. It retained the **17.54%** nominal reading saving and matched every reference class across **65,536** paired synthetic windows under eight scoring laws. The same-budget precommitted comparator had lower average nominal log loss. [Log-loss certificate and replay](LOG_LOSS_REPORT.md)

## Reproduce

Run from this directory with Python 3.12.14 and NumPy 2.3.5, the reference versions. The simulator generates its own signals.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 run.py --verify results.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 pruning.py --verify pruning_results.json
```

The first replay audits probability conservation, leaf budgets, nominal planner values, the matched one-read equality and independent endpoint reset. The second audits the decision certificate, the exact matched mean budget and every paired label. Both check Monte Carlo accuracy against exact model enumeration.
