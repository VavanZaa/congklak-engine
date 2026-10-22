# Ordinary value pilot: saved-game loss review

Retain H_v1 as the default and keep the original allocation HOLD. The candidate shows no demonstrated strength gain. This review reconstructs the existing 96 terminal games and 3,268 choices; it generates no games, targets, searches, fixed-simulation diagnostics or optimizer updates. Both models and their shared policy remain fixed.

## What changed against H_v1

All 12 families and both seats are retained for each baseline. Against H_v1, control and candidate both score 11 wins and 13 losses. Nineteen of 24 matched games have identical complete action trajectories. Five first divergences are candidate-owned; three change the final margin, all adversely, while two finish with the same margin. Twelve of the candidate's 13 losses have exactly the same moves and margin as control. These shared losses are not proved unavoidable, but they cannot be described as new losses caused by the value update.

| Saved family pair / candidate seat | Control margin | Candidate margin | First different choice | Human pit, control to candidate | Actual divergence stratum | Saved simulations |
|---|---:|---:|---:|---|---|---|
| 4 / 1 | -50 | -56 | 13 | 7 to 4 | 5-24 / capture | 388 / 403 |
| 6 / 1 | +18 | +14 | 6 | 7 to 3 | 25-49 / capture | 322 / 321 |
| 7 / 1 | +84 | +78 | 6 | 7 to 1 | 50-74 / bonus | 361 / 356 |

Pair and seat IDs above are the saved zero-based identifiers; displayed pits and choices start at one. The corresponding families end in `5212`, `7840` and `5261`. All three start in capture strata. Their later divergence mechanisms differ from their starting classification. In fact, none of the five H_v1 first-divergent positions is ordinary: four are capture positions and one is bonus.

The three seat deltas total -16 seeds, giving candidate minus control **-0.6667 seeds per matched game**. The inherited exploratory paired-family interval is **[-1.4167, 0]**. Removing any one family leaves the descriptive mean negative, ranging from -0.7273 to -0.4545; this sensitivity calculation does not create new independent evidence. The two other divergent games, pairs 2 and 8 at seat 1, remain +18 and +2 respectively under both models. No H_v1 result changes its win/loss category.

## What the traces establish

Every first divergence occurs at an identical literal board, stores and controller after a verified common action prefix. The five H_v1 candidate-owned divergences have identical saved policy priors and changed value predictions, but their simulation counts also differ. Even 322 versus 321 simulations does not hold the search tree or stopping boundary fixed. The saved wall-clock runs therefore do not isolate a deterministic value-head effect.

For example, pair 4's control pit 7 immediately gains five stored seeds, including a capture of four; the candidate's pit 4 gains one and grants another choice, followed by a capture of three. These are literal local facts on two subsequently different trajectories, not proof that either action is a game-theoretic blunder. Raw priors, visits, backed-up Q, bootstrap values, historical whole-decision clocks, physical evaluator counts and nearby transitions remain in the full report.

The declared consumed-reference index contains 17,979 references across 9,178 exact full states. None of the ten first divergences across both baselines matches that index. This is **uncovered in this index**, not a claim that no reference exists anywhere. No player swap, store offset or quotient is used to transfer a Q value. Nine distinct divergence states receive 18 deterministic literal CPU model forwards; new inference timing is not measured.

## H8 relay and overlap context

Against H8 relay, both models score 12 wins and 12 losses with mean margin -0.5833. The paired change is zero, with inherited interval **[-0.6667, 0.5833]**. Nineteen of 24 action trajectories match. Of five first divergences, three are candidate-owned and two are opponent-owned. At the opponent-owned divergences, saved completed search depths are 13 versus 12 and 5 versus 6. “H8 relay” names its supervision lineage; these timed decisions did not all search to depth eight. Opponent-owned means the first different action after the common prefix, not the first decision of the game or absence of earlier candidate evaluations.

The original admitted-pool overlap observations occur in games 0 and 4, pair 0 / seat 0: both are identical +18 wins with 45 choices. They are outside the three adverse H_v1 seats. The matching rehearsal row has zero draws in this pilot, while the sealed historical all-7,669-root draw-support proof establishes positive ancestral draw exposure to its intentionally encoder-equivalent pattern. Neither fact proves a causal gradient or game-loss effect. The separately reconstructed [overlap review](../build/ordinary-value-overlap-review.json) explains the orientation and store offset; the original conservative HOLD is unchanged.

## Evidence and next decision

The machine-readable [arena review](../build/ordinary-value-loss-review.json) retains all 48 matched seats, all 24 baseline/family units, all 13 absolute H_v1 losses, unchanged outcomes, first divergences and all 96 literal replay paths. This is consumed development evidence from a small selected pilot, not release validation. The [validation review](../build/ordinary-value-validation-review.json) separately addresses the six finite-horizon roots; target fit must not be substituted for strength.

A prospective fixed-simulation comparison on all ten divergence observations could separate equal-work value/search behavior from wall-clock variation, and same-state bounded references could test action preferences. Neither was run or authorized as part of this saved-data review. No further training, model selection, weight adjustment or collection follows automatically.

Read-only verification: `python tools/review-ordinary-value-losses.py verify`. A missing report can be rebuilt atomically with `build`; an existing differing report is refused. The owner suite checks all matched arithmetic, exact-state/player1 regret semantics, rehashed metric/source corruption refusal, and stable completed API/CLI rebuilds with searches prohibited. Review restart recovery recomputes saved evidence; it is not a claim of new process-kill tests.
