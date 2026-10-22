# Five H_v1 divergence diagnostics

Production and all 16 current source-bound methods passed: four search-owner, four teacher-owner and eight independent checks. Recovery and completed reopening are verified. Models and policy remain fixed; no training is authorized by this study.

The five selected first-divergence roots are pairs 2, 4, 6, 7 and 8, all candidate seat 1 and player 1 to move. Three later game margins were worse; two were unchanged. These are consumed forensic positions, not unseen release evidence or optimizer data. Both model files and their shared policy remain fixed.

## Fixed-work results

All 40 fresh-tree cells completed exactly 19,200 root simulations, with zero caps. Models chose the same action in 17 of 20 position-budget pairs. All five choices matched at 256 and 1,024 simulations. This agreement does not establish convergence, monotonic improvement, equal-time strength or the cause of a later game loss.

Human tables use pits 1Ã¢â‚¬â€œ7; journal actions are 0Ã¢â‚¬â€œ6. C is the fixed control, V the fixed value-only candidate.

| Pair | Saved timed C/V | 128 C/V | 256 C/V | 512 C/V | 1024 C/V | Later margin delta |
|---|---|---|---|---|---|---:|
| pair02 | 7/1 | 7/7 | 7/7 | 1/1 | 1/1 | 0 |
| pair04 | 7/4 | 4/1 | 4/4 | 7/7 | 7/7 | -6 |
| pair06 | 7/3 | 3/3 | 3/3 | 7/7 | 7/7 | -4 |
| pair07 | 7/1 | 1/4 | 7/7 | 7/1 | 1/1 | -6 |
| pair08 | 5/7 | 7/7 | 7/7 | 7/7 | 5/5 | 0 |

Both models have identical root policy priors at all 20 matched pairs. Search used the unchanged native CPU backend, declared core 2, c_puct 0.25, no clock stop/noise/cache, max_batch 1 and max_nodes 2048. Sampled PUCT action means retain None for unvisited actions. They are distinct from neural parent bootstrap and teacher action Q. Full physical counters and the limited resource observation are in [search results](DIVERGENCE_SEARCH_RESULTS.md).

## Full-action teacher references

All 15 requested attempts completed: five roots each at H8, H10 and H12. Every legal action is covered: 24 scores per requested horizon, 72 endpoint scores total, zero missing labels. All 150 consecutive completed iteration reports were retained. Production committed 7,815,306 nodes against the predeclared 85,000,000 ceiling. These counters exclude fixtures, audit replay and any lost uncommitted work.

H8, H10 and H12 denote teacher choice depths of 8, 10 and 12. H10 remains the primary finite reference; H8 and H12 are separate sensitivity diagnostics. Player 1 minimizes player-0 Q. Regret is the actor-signed difference from the best action in seeds, with all ties retained. These are finite-choice H_v1-leaf estimates, not exact game values.

| Pair | H8 best pits | H10 best pits | H12 best pits | Saved timed H10 regret C/V | Saved timed H12 regret C/V |
|---|---|---|---|---|---|
| pair02 | 1, 7 | 1, 7 | 1, 7 | 0/0 | 0/0 |
| pair04 | 1, 4 | 1, 4 | 7 | 1/0 | 0/1 |
| pair06 | 3 | 7 | 7 | 0/1 | 0/1 |
| pair07 | 7 | 1, 5 | 1 | 1/0 | 3/0 |
| pair08 | 5, 7 | 5, 7 | 5, 7 | 0/0 | 0/0 |

All five roots remain in every denominator. Below, lower is better; regret sums are across five eligible roots for each row, not averaged across incompatible horizons.

| Reference | Simulations | Eligible/missing roots | Control regret sum | Candidate regret sum | Candidate lower/same/higher regret roots |
|---|---:|---|---:|---:|---|
| H8 | 128 | 5/0 | 10 | 7 | 1/4/0 |
| H8 | 256 | 5/0 | 0 | 0 | 0/5/0 |
| H8 | 512 | 5/0 | 6 | 16 | 0/4/1 |
| H8 | 1024 | 5/0 | 16 | 16 | 0/5/0 |
| H10 | 128 | 5/0 | 1 | 6 | 0/4/1 |
| H10 | 256 | 5/0 | 2 | 2 | 0/5/0 |
| H10 | 512 | 5/0 | 2 | 1 | 1/4/0 |
| H10 | 1024 | 5/0 | 1 | 1 | 0/5/0 |
| H12 | 128 | 5/0 | 2 | 7 | 0/4/1 |
| H12 | 256 | 5/0 | 5 | 5 | 0/5/0 |
| H12 | 512 | 5/0 | 3 | 0 | 1/4/0 |
| H12 | 1024 | 5/0 | 0 | 0 | 0/5/0 |

## Interpretation before training

Pairs 2 and 8 retain tied saved actions at all three requested depths. Their saved divergence does not identify an inferior move under these references. Pair 6 changes its best set between H8 and H10; pair 7 also changes between H8 and H10; pair 4 changes between H10 and H12. Three of ten adjacent-horizon comparisons have disjoint best sets. A single hard reference would hide this sensitivity.

The candidate-specific low-budget issue is pair 7 at 128 simulations: candidate pit 4 has five seeds more regret than control pit 1 at both H10 and H12. At 512 on that same root the candidate has lower regret: one seed at H10 and three at H12. This is budget-sensitive evidence on one selected state. It does not support a blanket claim that the candidate value head caused the old game losses.

At 1,024 simulations both models agree on all five actions. Their H10 regret sum is one seed each (pair 4); at H12 their regret is zero on all five. H8 gives a different ranking. Retain all budgets and horizons rather than selecting the favorable one. Neither H12 agreement nor a larger simulation count is a promotion or strength test.

Keep the existing control model and fixed policy. Before a training pilot, define a candidate horizon-stability weighting or soft/tied-target rule offline using the retained histories, and investigate the pair-7 low-budget visit allocation. Such weighting is an uncalibrated proposal to predeclare and test on a separate eligible training corpus; these five posthoc cases cannot estimate calibrated confidence weights or justify changing current weights. Keep these five consumed states in the diagnostic compartment. Any later training batch needs distinct eligible training states and protected evaluation, rather than duplicating these five or turning finite teacher Q into exact outcomes. This study adds no optimizer rows, training, self-play, games, pruning, model selection or promotion; the previous allocation HOLD remains.

## Recovery, checks and preserved history

Current owner suites passed four search and four teacher methods. Independent preflight passed three methods, including four actual child-process kills before/after search-cell and teacher-attempt commits. Committed raw prefixes survived unchanged and resumed semantics matched fresh runs. Independent method 4 replayed every production tree using independent selection, backup and seed-transition logic with the same frozen numerical leaf backend; model-root encoder/network parity was checked separately. The remaining four independent methods verified all 15 labels, 150 consecutive iteration reports, 870 literal PV paths and 4,889 choices; every comparison metric, rehashed corruption refusal, and completed production API/CLI reopening passed. All eight independent methods completed without audit source repair. Root combined reopening additionally prohibited new search and training entry points and preserved bytes, sizes, mtimes and durable file inventories.

The initial backend-label admission error and temporary SQLite shared-memory timestamp test error were corrected before production. Exact executed sources, reports and logs are retained in the corresponding build/divergence-search-prefreeze-* archives. The first teacher dependency-bound fixture pass and comparison dependency rebinds are also retained. Historical attempts are not counted as additional current passed methods.

Closed producer resume commands are `python tools/probe-divergence-search.py run` and `python tools/label-divergence-actions.py all`. Incomplete work resumes from atomic committed cells/attempts; completed work verifies and republishes only validated missing exports. Wrong identities or existing corrupt outputs are refused. Teacher production requires the genuine completed 40-cell witness and independent preflight/search proofs.

All 3,914 previously protected files remain pinned. Machine-readable evidence: build/divergence-panel.json, build/divergence-search-result.json, build/divergence-teacher-result.json, build/divergence-comparison.json, build/divergence-audit.json and build/divergence-acceptance.json. The final seal and dashboard evidence are build/divergence-acceptance-rerun.json and build/divergence-dashboard-final.json. See [teacher results](DIVERGENCE_TEACHER_RESULTS.md) for per-root work and retained histories. Local dashboard: http://127.0.0.1:18766/.
