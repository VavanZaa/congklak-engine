# Within-stratum weighting: results and training decision

Within-cell normalization preserves midgame coverage and mildly lowers weighted best-action regret on the 16 selected roots. **The frozen overall screen does not pass:** mean action-Q drift increases. Do not apply the tested weights; retain existing sampling/loss settings. This stage performed no training, changed no model or policy, and promoted no engine.

The lower regret comes entirely from giving less weight to two unchanged H10 best-move errors. It does not mean the engine fixed either move or became stronger. The assessment is small, balanced and now consumed. H8, H10 and H12 are requested teacher search horizons of 8, 10 and 12 choices, not different neural models. H12 is a finite H_v1 reference, not solved game truth.

## Research and fixed design

Internet research informed two safeguards: preserve declared group totals and measure decision quality directly. Post-stratification separates cell estimates from the masses used to combine them; our masses are deliberately chosen assessment targets, rather than known gameplay frequencies. Group-robust and decision-focused research motivates reporting ordinary-cell harm and full best-action regret. These ideas do not prove this heuristic is calibrated or that it improves PUCT. See the primary-source links and applicability limits in [STRATUM_RESEARCH.md](<F:/My Engine/Congklak Stockfish/docs/STRATUM_RESEARCH.md>), including [Gelman on weighting](https://sites.stat.columbia.edu/gelman/research/published/STS226.pdf), [group DRO](https://arxiv.org/abs/1911.08731) and [Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005).

The unchanged raw score is:

    r_i = max(1/4, 1 / (1 + abs(V10_i-V8_i)/4
                          + mean_legal(abs(Q10_i-Q8_i))/4))

The new comparison normalizes inside each position-stratum/actor cell:

    W_i = (1/8) * r_i / sum(r_j for j in the same cell)
    U_i = 1/16

There are two roots in each of eight cells. Every cell keeps exactly 1/8 mass, every stratum 1/4, each actor 1/2, and ordinary positions together 1/2. Each actor across the two ordinary strata keeps 1/4. The previous global rule's loss of high-ordinary mass is structurally prevented. These are comparison probabilities; no optimizer or sampler uses them in this stage. A future sampler at W with unit loss already expresses that objective; applying the same reliability factor again would change the objective a second time.

The [prospective scope](<F:/My Engine/Congklak Stockfish/docs/STRATUM_SCOPE.md>), panel, source, formula, missing policy and guards froze before new H8/H10 measurements. All 32 H8/H10 endpoint hashes and exact realized probabilities froze before H12. The independent rule audit checked the immutable reference snapshot before admitting H12. No threshold, root, formula or guard changed after results arrived.

## Separate eligible support and measurements

After excluding the prior allocation panel's entire reservations and all protected, forensic, mixed-role and historical H12/H14-exposed components, the old labelled subset could not fill bonus/capture cells. A metadata-only scan of broader saved support found 3,896 eligible observations: 3,890 distinct raw states, 553 full origin families, 551 deterministic representative families and 547 components. Six duplicate observations retained all their origin provenance.

Deterministic metadata matching selected 16 assessment roots from distinct origin families and components, two per stratum/actor cell. Four high-ordinary roots had earlier tracked H8/H10 exposure; one low-ordinary root had H8 exposure only. The other 11 had neither in the three tracked recent teacher journals. This is not a claim of globally unseen data, erased ancestral training or prefix-disjoint histories. No target body contributed to selection. Protected evaluations and the preceding consumed panel remained excluded through full eligible family/quotient closure.

All **48 production attempts completed**: 16 each at H8, H10 and H12, with no caps, missing endpoints, replacement or budget escalation. The journal retains **480 iteration reports and 234 requested endpoint action-Q values**. Native H_v1 used TT16 MiB, PVS, time0, memory512 MiB and a ten-million-node cap per attempt. Actual work was **39,781,297 nodes**, below the 480-million ceiling: H8/H10 together used 6,482,030 nodes, followed by 33,299,267 H12 nodes. There are 2,820 literal PV paths and 15,886 PV choices in the independent verification. These work counts are not strength or energy benchmarks.

## Decision quality and coverage

Drift and regret are in seeds. Root drift is `abs(V12-V10)`; action-Q drift is the per-root mean absolute shift across every legal action. Best-action regret evaluates the entire H10 best-action set under H12. Every actual H10 and H12 best set is a singleton here, so minimum, mean, worst-tie and deterministic-choice regret coincide. Synthetic checks exercise tied cases separately.

| Fixed scope | Uniform root drift | Weighted root drift | Uniform action-Q drift | Weighted action-Q drift | Uniform best-action regret | Weighted best-action regret |
|---|---:|---:|---:|---:|---:|---:|
| All 16, primary | 1.875000 | 1.873096 | 1.999405 | **2.023762** | 0.187500 | 0.171334 |
| All ordinary, 8 roots | 2.125000 | 2.044332 | 2.420833 | 2.407595 | 0.125000 | 0.106557 |
| High ordinary, 50–74 active seeds | 3.000000 | 2.864174 | 3.841667 | 3.815191 | 0 | 0 |
| Low ordinary, 25–49 | 1.250000 | 1.224490 | 1.000000 | 1.000000 | 0.250000 | 0.213115 |
| Low bonus, 25–49 | 1.250000 | 1.263889 | 1.604167 | 1.608550 | 0.500000 | 0.472222 |
| Low capture, 25–49 | 2.000000 | 2.139831 | 1.551786 | 1.671308 | 0 | 0 |

Overall best-action regret decreases about 8.62%, or just 0.01617 seeds in absolute terms. Root drift improves only 0.00190 seeds. Action-Q drift worsens about 1.22%, or 0.02436 seeds. The **only failed guard** is overall action-Q non-increase. All complete-support, exact-mass, nontrivial-weight, concentration, overall/ordinary mean and worst-tie regret, overall root-drift and strict-improvement guards pass. The failed guard remains part of the result; it is not removed because other metrics improved.

Only **two roots** have nonzero H10-best H12 regret: one actor-1 bonus root with regret 2, and one actor-1 low-ordinary root with regret 1. Their comparison masses fall from 1/16 to 17/288 and 13/244 respectively. The other 14 roots have zero regret. This is concentration toward fewer measured errors, without changing any Q value or action. It is weak evidence for generalization and supplies no learned-strength evidence.

Capture positions drive the action-Q regression: actor0 changes 1.175000 to 1.278689, and actor1 1.928571 to 2.063927. Bonus actor1 also worsens, 1.666667 to 1.685185. Root drift worsens in that bonus cell and both capture cells. No subgroup is hidden because it is outside the ordinary regret gates. The [focused assessment report](<F:/My Engine/Congklak Stockfish/docs/STRATUM_ASSESSMENT_RESULTS.md>) retains all eight cells and the two error roots.

All eight cells have non-uniform weights. Probabilities range from 25/488 (0.05123) to 9/122 (0.07377), versus uniform 0.0625; maximum relative exposure is 72/61 (1.1803). Overall inverse-concentration ESS is **15.7697/16**. Ordinary-cell ESS values range from 1.9552 to 1.9841 out of two; all exceed 1.6. ESS is a concentration diagnostic, not an independent sample count or uncertainty estimate. Missing mass is zero. The implementation retains original probabilities and makes full-risk guards unavailable if an endpoint is missing, rather than redistributing its mass.

## Admission and next training preparation

The [new reservation overlay](<F:/My Engine/Congklak Stockfish/build/stratum-reservations.json>) reserves all 16 selected components, **17 full origin families and 118 eligible quotient aliases** from future optimizer draws. The extra family reflects full duplicate/alias provenance, not an extra selected root. Historical roles stay unchanged. Existing sealed trainers do not automatically discover this overlay. Future admission must consume both this ledger and all prior reservations and independently prove their exclusions. Historical H8/H10 exposure and possible ancestral/internal-search exposure remain explicit.

**Retain existing sampling/loss settings.** Keep fixed cell totals as a useful coverage constraint, and retain the reliability rule as an experimental comparator. Prepare any later small training pilot with reservation-aware admission and an explicitly frozen sampling/loss objective; a balanced uniform control is a proposed future comparator, not a description of every historical trainer. The present Q-drift regression matters for value/Q targets even though measured best-action regret fell; it does not support applying one stability weight indiscriminately to every head. Do not fit a revised rule or relax guards on these consumed roots. A future learned candidate still needs equal-time tests against both baselines and the endgame regression guard. This study changes no move-selection configuration and runs no strength games.

## Recovery, retained failures and closure

Two current accepted recovery cases kill a real teacher process before and after the second H10 fixture commit. Fixtures use already-consumed roots at explicitly requested depths 2/3/4. Saved prefixes retain two H8 attempts plus one/two H10 attempts: stage prefixes 1/2 and aggregate prefixes 3/4. Exact committed rows survive, and resume reaches fresh-run semantics. An unfinished attempt can repeat; native TT/search internals are not persisted mid-attempt. Immutable reference snapshots and required phase evidence are validated across the H12 boundary.

There were **four physical kills across both formal preflight executions**, with two in current accepted coverage. Earlier successful recovery probes remain retained under the first failed audit execution and do not add unique methods. Current root closure also reads validated SQLite prefixes and flushed markers directly, rather than relying only on summary booleans.

All engineering failures and diagnostics are retained. The first frozen metrics-owner run passed methods01–03 and failed04 because a synthetic byte fixture wrote Windows CRLF instead of its expected LF. A one-line fixture-write repair was independently reviewed; the scientific helper did not change, and all four current methods passed. The first formal independent preflight passed methods02/03 but method01 failed because the main invocation preceded later helpers; group publication also failed before a formal JSON could be written. An uncounted, unfrozen metadata-only diagnostic then exposed a callback handling non-read SQLite actions incorrectly. Moving the entry point to EOF and adding an action/body short-circuit left the actual target-body denial policy unchanged. Exact sources, logs, freezes and raw fixtures remain retained. A source-frozen focused diagnostic passed before the full current preflight passed. No producer, formula, panel or model changed in these audit repairs.

The first final-group run passed method08 but method07 expected only `ValueError`/`FileNotFoundError` after deleting a required reference snapshot. The reader correctly refused with SQLite `OperationalError`; no missing phase evidence was silently rebuilt. That failed report remains unchanged. A separately frozen [versioned final runner](<F:/My Engine/Congklak Stockfish/build/stratum-audit-final-v2.py>) widens only that expected-exception tuple. Its [provenance](<F:/My Engine/Congklak Stockfish/build/stratum-audit-final-provenance-v2.json>) preserves the original main, worker, source freeze and all three production-admission stage proofs. Current final coverage binds six independent methods to their original source and two final methods to the versioned runner. The root recorder verifies the actual source bytes, AST method presence, all four partial reports, their exact eight-method union and merged evidence. Repeated methods do not become additional checks.

Final unique executed coverage is **16 methods: four producer-owner, four pure-metrics-owner and eight independent**, with zero errors, failures or skips in their current accepted reports. Actual source/input bindings are recorded in [stratum-acceptance.json](<F:/My Engine/Congklak Stockfish/build/stratum-acceptance.json>) and the [independent audit note](<F:/My Engine/Congklak Stockfish/docs/STRATUM_INDEPENDENT_AUDIT.md>). Repeated methods and historical logger tests are not recounted. The [completed-reopen proof](<F:/My Engine/Congklak Stockfish/build/stratum-rerun.json>) checks zero new teacher/PUCT search, inference runtime activation, training or games and unchanged durable bytes, sizes and modification times. The [seal replay](<F:/My Engine/Congklak Stockfish/build/stratum-acceptance-rerun.json>) checks idempotent publication; the [dashboard proof](<F:/My Engine/Congklak Stockfish/build/stratum-dashboard-final.json>) checks all nine current cards, the current successful replay, and the source-matched listener without browser or HTTP inspection. All 4,546 prior protected inputs remain preserved.

Resume explicit stages with `python tools/validate-stratum-weighting.py h8`, then `h10`, `realize`, independent rule verification, and `h12`. The pre-H12 independent gate is mandatory. Once complete, `run`, `assess` and `verify` reuse and validate committed work without repeating the measurements. The [dashboard](http://127.0.0.1:18766/) reports the mixed result and the unchanged no-training decision.
