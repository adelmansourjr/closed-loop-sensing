# Decision-invariance follow-up protocol

This protocol was fixed after inspecting the first study's exact policy trees and before evaluating a pruning policy or generating the follow-up window bank. The original primary comparison and result are preserved in PROTOCOL.md and results.json.

## Question and certificate

Can a measurement-only system eliminate redundant reads while preserving every terminal class of the strongest cue-conditioned fixed schedule? At each history in its frozen policy tree, collect the terminal argmax labels of all descendant outcomes, including every erasure branch. If the set is a singleton, stop the measurement schedule, forecast the endpoint through unobserved transitions and emit the usual nominal Bayes posterior. Otherwise execute the next scheduled query, if any. Restrict all policies to at most two optional reads.

The primary endpoint is exact mean attempted reads in the nominal law, conditional on the frozen reference trees. The gate requires at least 10% fewer attempts than the two-read reference and exact pathwise class agreement on all histories, including erased and skipped observations. Label equality is the certificate, not distributional equivalence of posteriors or scores. Report changes in Brier score and in the number of retained reads. A comparison using probabilities would need its own stopping criterion.

## Strong equal-mean-budget comparator

Calculate the nominal mean read count of the pruning policy first. Among schedulers that can use the cheap guide but must commit to all measurement times and axes before observing outcomes, compute the best achievable exact accuracy under that mean count. Allow the schedule to be selected randomly from every combination of 0, 1 and 2 reads, conditional on the guide. Decode with the same nominal Bayes classifier. For a given guide and count retain only the exact highest-accuracy schedule because all others have the same cost. Optimise the resulting mixture exactly by search over extreme points. At a vertex one guide's mixture is a point mass and the other has at most two nonzero probabilities. Report the difference in accuracy and Brier score and the nominal mean cost. Under shift keep this deployment mixture frozen as a diagnostic. The matched-mean comparison is secondary.

## Evaluation

Re-use the six signal laws listed in PROTOCOL.md and add two independent checks. Severe erasure uses high and low erasure probabilities of 0.48 and 0.72, and polarity mixture draws each trajectory from nominal with probability 0.65 or projects onto the opposite polarity throughout the window with probability 0.35. Keep the nominal filters and schedules across all laws. Numerically integrate the mixture over its two component laws. Use 32,768 new paired windows per family with seed 764916 plus the family index, starting at zero. Freeze mixture assignment independent of all draws for signal generation. Use exact tree traversal and an independent simulated observation tape. Check exact outcome preservation across the whole tree, zero empirical class disagreement, exact budget feasibility, and simulator agreement with enumerated accuracy to within five binomial standard errors. Record posterior Brier score because identical classifications can have different probability quality.
