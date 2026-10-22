# Compact policy/value and PUCT quality results

The fixed-final compact refinement and lower-exploration treatment improve development playing margins substantially. The combined profile reaches +1.208 paired seeds against H_v1 and -1.292 against H8 relay CPU, compared with -11.333 and -15.000 for the warm/current-search control on the same families. **Neither both-baseline strength gate is satisfied; H_v1 remains the default.** Policy teacher metrics improve while value MAE regresses. All 10 independent audit methods and the exhaustive checkpoint/game audit pass.

This experiment uses the user-confirmed `id-betawi-relay7-v0` Betawi profile and single-round final player-0 seed-margin objective. No larger model, expanded target corpus or second learning generation is introduced.

## Treatments and ownership

The four profiles cross one bundled model treatment with one exploration treatment. All use the same uncached structured native CPU evaluator and the same PUCT traversal, parent bootstrap FPU, fixed-player-0 backups, actor-aware selection and visits-based final recommendation.

| Profile | Model | c_puct |
| --- | --- | ---: |
| warm_control | Preserved policy warm start | 1.5 |
| warm_search | Preserved policy warm start | 0.25 |
| refined_control | Fixed-final joint refinement | 1.5 |
| refined_search | Fixed-final joint refinement | 0.25 |

The lower coefficient tests a search-allocation hypothesis; the previous coefficient is not a correctness bug. The model treatment bundles sharper root teacher policy targets, normalized residual value loss and joint teacher-only optimization. The model/search factorial comparison does not isolate the individual training ingredients.

The new registry extends protected ownership with positions from all 192 prior Phase6 and 576 CPU arena games. A protected active-state quotient blocks the entire family of any matching selected original training root. Of 8,000 original training rows, **331 are excluded and 7,669 admitted**. The original data and exclusion reasons remain intact. The 1,000 development rows enter descriptive evaluation only, never optimizer draws. This is a selected-root bridge filter, not enumeration of every possible family trajectory.

These exclusions restrict this run's optimizer draws. The warm ancestor already trained on the original 8,000 roots, including now-excluded roots; this refinement does not undo that historical exposure. The ancestor was also selected using development data. The 48 new H8 case families and 24 new arena families are disjoint and absent from earlier arena family IDs, while remaining development data. They do not provide an unseen or sealed release result.

## Joint training and descriptive quality

Exactly **1,500 CUDA optimizer steps**, batch256, learning rate0.0003 and seed20261015 completed. All 12,872 parameters of the existing 127-input, width64 network were trainable. AdamW/fp32 used residual-space Huber(delta0.1) plus0.25 legal policy cross entropy at2seed teacher temperature. There were 384,000 uniform-stratum draws with replacement and unit row weights. No self-play outcome or exact-oracle target was added. Development metrics did not select the checkpoint.

All **30 scheduled checkpoints** are retained and tensor-verified, including optimizer ownership, sampler draw progression and final export mapping. The selected model is step1,500, weights digest `1efd42cf39e64183752ad3bcbf3ba1b82e7e5fdc4c273426c763d4e20d746649`. All eight parameter tensors were trainable; independent audit separately checks their actual changes and the saved payloads.

| Same development target metric | Warm source | Refined final |
| --- | ---: | ---: |
| Row-weighted value MAE, seeds | 5.062 | 5.339 |
| Uniform-stratum value MAE, seeds | 4.888 | 5.317 |
| Uniform-stratum residual MAE | 0.148557 | 0.149911 |
| Uniform-stratum root policy teacher regret, seeds | 8.347 | 2.923 |
| Uniform-stratum root policy teacher-optimal-set accuracy | 50.29% | 64.72% |
| Uniform-stratum legal policy cross entropy | 1.438755 | 1.064644 |

The policy improves on the declared teacher metrics, while value error increases. That tradeoff is retained rather than hidden by policy loss or checkpoint selection. These targets are finite-choice H8 teacher estimates, not exact minimax values. Both model versions are evaluated against the same tau2 targets in this table; the warm model originally trained its policy at tau4. Rare sampling strata still have very few distinct examples, so repeated draws do not establish new diversity.

## Equal-time strength and search effects

The registered campaign uses 80 cases: 48 new H8 development roots and the separate 32 known-exact diagnostics. Six engines at25/100ms yield960 scalar measurements. Four PUCT profiles face both H_v1 and H8 relay CPU on24 paired-seat families at25ms, for384 games. All engines use the same declared performance core. Baselines use depth16, TT16MiB and PVS; PUCT uses a100,000-backup ceiling,2,048-node bound and no root noise. Whole-decision clocks include reset and validation, with only floored positive remaining milliseconds granted to search.

The primary gate requires all24 terminal pairs, zero fallbacks and a positive lower95% family-bootstrap margin against both baselines. All eight intervals are exploratory and unadjusted. Secondary mean differences compare model bundles at each search setting and search settings at each model on matched families; these add no gates or confidence claims. Time-mode work and trajectories can differ with scheduling. All variants and outcomes are retained.

All **384 games are terminal**, every comparison has24 complete pairs, and there are **zero fallbacks**. The refined_search mean versus H_v1 is positive, but its interval crosses zero. Its interval versus H8 remains below zero. All four both-baseline gates fail, so there is no default promotion.

| Profile / opponent | Paired mean seed margin | 95% family interval | Wins / draws / losses | Gate |
| --- | ---: | --- | --- | --- |
| warm_control / H_v1 | -11.333 | [-17.542, -6.000] | 17 / 1 / 30 | Failed |
| warm_control / H8 relay CPU | -15.000 | [-21.500, -9.375] | 15 / 0 / 33 | Failed |
| warm_search / H_v1 | -5.042 | [-8.417, -1.875] | 23 / 0 / 25 | Failed |
| warm_search / H8 relay CPU | -8.792 | [-13.833, -4.625] | 21 / 0 / 27 | Failed |
| refined_control / H_v1 | -3.417 | [-6.125, -0.958] | 23 / 0 / 25 | Failed |
| refined_control / H8 relay CPU | -5.417 | [-7.875, -3.208] | 22 / 0 / 26 | Failed |
| refined_search / H_v1 | +1.208 | [-0.250, +2.708] | 24 / 1 / 23 | Failed |
| refined_search / H8 relay CPU | -1.292 | [-2.583, -0.042] | 24 / 0 / 24 | Failed |

The predeclared secondary contrasts show contributions from both tested factors on this cohort. Each number is the mean difference of paired-family final margins on24 complete matched families against the same opponent. They have no additional confidence interval or gate, and do not establish unseen generalization or isolate the ingredients of the model bundle.

| Descriptive treatment minus control | Against H_v1, seeds | Against H8 relay CPU, seeds |
| --- | ---: | ---: |
| Lower exploration at warm model | +6.292 | +6.208 |
| Lower exploration at refined model | +4.625 | +4.125 |
| Model refinement at control exploration | +7.917 | +9.583 |
| Model refinement at lower exploration | +6.250 | +7.500 |

The combined change improves the matched-cohort mean relative to warm_control by12.542 seeds against H_v1 and13.708 against H8. Comparisons with older campaigns use different families and cannot establish that same contrast. WDL remains secondary: the combined profile's51.04% score against H_v1 and50% against H8 do not override the seed-margin gate.

On the48 new H8 case roots, the policy/search treatment also reduces teacher-reference regret. Approximate H8 action scores are the reference, not minimax truth; the32 known-exact diagnostics remain a separate cohort.

| Engine | Reference-action accuracy, 25 / 100 ms | Mean reference regret, seeds, 25 / 100 ms | Mean backups, 25 / 100 ms |
| --- | ---: | ---: | ---: |
| H_v1 | 89.58% / 87.50% | 0.375 / 0.250 | Not PUCT |
| H8 relay CPU | 81.25% / 75.00% | 0.563 / 0.688 | Not PUCT |
| warm_control | 64.58% / 68.75% | 2.833 / 2.813 | 443.4 / 1,904.9 |
| warm_search | 79.17% / 83.33% | 1.854 / 1.292 | 458.5 / 2,053.8 |
| refined_control | 75.00% / 77.08% | 0.729 / 0.604 | 448.8 / 1,969.0 |
| refined_search | 83.33% / 83.33% | 0.583 / 0.500 | 485.4 / 2,144.2 |

Completed backups can exceed allocated nodes because already reached terminal leaves are revisited; those counts do not represent additional unique neural evaluations or proofs. At100ms, maximum ordinary-case whole-wall times are101.143/101.132/101.908/101.106ms for warm_control/warm_search/refined_control/refined_search. H_v1 reaches101.818ms and H8 reaches101.917ms. All actual wall clocks and overshoots remain retained; the nominal budget is not a hard real-time guarantee.

## Recovery, verification and resources

The independent auditor has passed real CUDA training kills before a step4 file was committed and after step4 committed. Resumed step6 fixtures match fresh step6 model tensors, all optimizer moments, sampler state, draw counts, and Python/Torch/CUDA RNG exactly. Uncommitted files, good backups and raw killed states remain retained. These fixtures exercise the new trainer; they do not expand the earlier warm-training audit claim.

The refined_search analyzer demonstration completed128 backups. Its same-identity finished rerun preserves SQLite and JSON export bytes and nanosecond modification timestamps. Its fixed-player-0 margins, recommended action and visit allocation are estimates, with trace-derived explanations. Unfinished trees are recomputed while the latest completed recommendation remains durable.

All **36 distinct executed checks** pass: four learning, nine search, thirteen benchmark and ten independent methods. The independent audit verifies literal search transitions, both models against 1,000 literal development forwards each, all 30 actual production checkpoint payloads and changes in all eight parameter tensors. It reconstructs ownership through 768 earlier games / 24,581 choices, and replays all 11,920 choices in the current 384 games. All 1,368 prior accepted input hashes remain unchanged.

Real analyzer kills before and after commit preserve the completed 64-backup recommendation, then recover to the same 128-backup search and physical work as a fresh run. Missing exports repair from committed state. A killed case campaign retains case 1 and resumes through case 2; a copied arena journal retains 383 committed games and resumes through game 384. Completed reruns preserve bytes and modification times. Rehashed semantic corruption is refused; raw killed journals, backups and refusal logs remain retained.

The first exhaustive audit exposed a TEXT-versus-BLOB key assumption in the auditor's checkpoint parser. Its failed report, original runner and group reports are preserved under `build/quality-audit/auditor-text-key-correction-v1`. The corrected independent parser passes all 30 production payloads; all 10 source-bound methods, including actual CUDA kills, were rerun against the corrected auditor. Producer sources, trained model and measured games did not change.

The reported production training work clock was **25.469 seconds** after common admission, including CPU dataset preparation, lazy Torch/CUDA loading, model setup, optimization, all checkpoint metrics/publications and final export. Module import, common admission, Python startup and the subsequent final result JSON publication are excluded. Peak working set was1,396.5MiB; peak CUDA allocated/reserved memory was70.6/88.0MiB, excluding driver/context and other processes. Verification and audit invocation costs are separate.

The original CPU case/arena invocation took **329.870 seconds**, beginning with zero committed cases/games. Its clock includes admission, startup, validation, remaining work, journals and semantic report publication after module imports; Python/module startup and earlier invocations are excluded. Peak private commit/working set was58.7/70.6MiB. No Torch import or CUDA inference occurred in that worker. Original invocation resources publish before the completion marker and remain immutable on completed semantic reruns. These are process/allocator observations, not whole-machine electricity measurements.

## Remaining engine work

The accepted simulator, alpha-beta/PVS/TT engine, K=4 database, compact models, supervised and self-play pipelines, native inference and resumable analysis remain available. H_v1 stays the default during this exploratory quality experiment.

The next bounded quality experiment should preserve the improved policy as a control and test value supervision or gradient separation that avoids the observed value regression. New ordinary/high-active teacher examples and replay diversity are candidates; a larger model is not established as necessary by this result. Any new treatment needs a new frozen protocol and families rather than retuning this completed campaign.

A release still needs a both-baseline gate pass and fresh validation, explicit typed K=4 retrieval in a unified analyzer, clear estimate/proof labels, backend/model selection, reproducible setup and rollback. The initial board is not solved, and lower electricity use has not been measured. Larger models, more data and proof frontiers require their own evidence.

Evidence: `build/quality-plan.json`, `build/quality-implementation.json`, `build/quality-data.json`, `build/quality-model-witness.json`, `data/quality-train-v1`, `build/quality-benchmark.json`, `data/quality-benchmark-v1/work.sqlite3`, `build/quality-audit/audit.json` and `build/quality-acceptance.json`. See [frozen scope](QUALITY_SCOPE.md), [run and resume guide](QUALITY_RUN_GUIDE.md), and [independent audit](QUALITY_INDEPENDENT_AUDIT.md).
