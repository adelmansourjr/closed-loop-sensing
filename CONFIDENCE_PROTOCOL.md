# Confidence certificate protocol

## Question and proof target

The existing rotating-signal study used the nominal classifier and skipped readings when every remaining result would leave its class unchanged. Its Brier score worsened. When a system stops at such a history, which probability vector minimises the worst excess Brier loss relative to taking all scheduled readings?

Freeze the same nominal accuracy-optimal schedule with two reads selected exactly after the free guide. A history is eligible to skip future reads when all descendants of the schedule have the same tie-resolved argmax and at least one remaining read. These are the only histories at which an intervention in the decoder is allowed. Read allocation, labels, signal generator and full-schedule decoder are identical across the three replacement forecasts.

The adversary may select any descendant observation and terminal class, with no restrictions on their joint probabilities. For each class y form d[y] as the largest 2*q[y] minus the squared Euclidean norm of q over descendant forecasts q. Let p* be the Euclidean projection of d/2 onto the probability simplex. Derive and verify that p* uniquely minimises the worst case excess Brier loss of the stopped forecast over the full forecast. Check that it preserves the reference argmax when the descendant argmax is constant with the fixed tie convention. The adversary compares forecasts for the same realised terminal class. The bound is relative to the frozen full forecast. Neither absolute forecast quality nor support beyond the frozen tree is certified.

## Comparators and endpoints

- Full reference with two reads and no replacement.
- At the maximal stable-class stopping frontier replace the forecast with the exact nominal Bayes coarse posterior as in the earlier study. This is optimal for expected Brier score under the nominal model and is the primary comparator.
- At the identical stopping frontier, replace it with the uniform arithmetic mean of the full descendant forecasts or with the distribution-free minimax forecast. Compare allocations pathwise to the Bayes replacement.
- For a local cap of .20 excess Brier, apply each forecast's worst-case certificate before choosing its prefix. At an eligible history whose bound exceeds .20, keep the next reference action and continue checking eligible later histories. Report read costs and every pathwise violation. This cap equals one tenth of the full multiclass Brier loss range [0,2]. Secondary sensitivity caps are .10 and .30.
- A strong precommitted-schedule baseline may use the free guide and randomise among every 0-, 1- and 2-read schedule, but chooses all readings before seeing their outcomes. Select the exact minimum-Brier schedule per guide and read count under the nominal model. Mix those schedules to minimise expected Brier loss at the same nominal mean read count as the uncapped minimax policy. Freeze the resulting mixture under shift. Report accuracy and score alongside the family-specific budget. This baseline is free to change labels.

Primary endpoint is the maximum pathwise excess Brier loss of the uncapped minimax policy compared with Bayes replacement, at identical reads with exact class agreement. The practical gate is a relative reduction of at least 15% and unchanged class and reads on every reference path. Secondary endpoints are nominal expected Brier and accuracy, exact and simulated score differences under shift, local-cap savings, and Brier performance against the matched-mean precommitted baseline. The comparison trades nominal and worst-case performance. Report both even when one worsens.

## Evaluation and audits

Treat every terminal score as multiclass squared error with no factor of one half. Enumerate all guide branches, read outcomes and four terminal classes to compute exact worst-case differences and to locate an attaining pair. Separately integrate each frozen policy against the true test law. The nominal law is primary. Locked stress laws use the original synthetic model with the following overrides, where pairs refer to high/low quality.

| Family | Override |
|---|---|
| Nominal | None |
| Attenuation and noise | Reliability (.58,.09), erasure (.18,.33) |
| Drifting clock | Phase increments (.33,.47,.20), guide flip .27 |
| Bursty erasure | Quality persistence .97/.08, erasure (.18,.70) |
| Reversed projection | Reliability (−.63,−.22), guide flip .18 |
| Independent endpoint | Reset phase uniformly at the endpoint |

Use 32,768 independent paired windows per family, start seed 913004 plus the family index in the table. Give every policy the same hidden outcome and observation tape. Use separate mixture-dispatch draws from seed 915717 plus the same index. Record the cap guarantee and class agreement against the full frozen reference for every path and score replication on this observation tape. For differences in Brier loss versus the reference calculate a paired interval from the window-level difference and its standard error. Compare exact versus simulated differences to within five standard errors when this error is nonzero. Report a deterministic proof audit using exactly rational binary forecasts (.60,.40) and (.90,.10) and an independent one-dimensional optimising calculation, then test simplex optimality and label stability over 512 generated four-class fixtures. Keep the numeric simulation and all possible-outcome certificates in the replayable output. There are no physical actuation decisions, people or patient records in this experiment.
