# Ordinary-position data pilot

The fixed-model pilot produced **62 complete H8 labels: 44 training candidates, 10 validation roots and 8 protected diagnostic roots**. All 62 have at least two legal actions. No training, sampling changes, pruning or promotion occurred. H_v1 remains the default engine.

The retained model is `data/quality-train-v1/model.json`, SHA-256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. The subsequent value-only refit was not used. The frozen producer contract and implementation are in `build/data-pilot-plan.json` and `build/data-pilot-implementation.json`.

## Collection yield and limits

The declared pool contains 256 deterministic standard-opening history prefixes: 128 random histories and 128 frozen-policy histories with epsilon 0.25 after eight random opening choices. Prefixes stop below 25 active seeds, at a terminal state, or after 192 choices. They are not 256 terminal games.

The independent replay checked 5,199 choices and 1,198 policy evaluations. The pool contained 87 strict ordinary observations and 87 unique ordinary quotients. Every legal root move must avoid pass, capture and bonus mechanisms to qualify. The fixed eligibility rule was retained; no quota filler or artificial early position was added.

Selection uses the complete pool before closure. It excludes protected old root quotients and families and removes both sides of any candidate quotient shared across training and validation ownership. Selection allows one root per family and active-seed band. No cross-cohort candidate collision occurred in this pool. Shared history prefixes remain possible; these ownership checks do not establish independent games.

| Cohort | Active seeds | Cap | Selected and labeled | Shortfall |
|---|---:|---:|---:|---:|
| Training candidate | 25-49 | 32 | 32 | 0 |
| Training candidate | 50-74 | 32 | 12 | 20 |
| Training candidate | 75-98 | 8 | 0 | 8 |
| Validation | 25-49 | 8 | 8 | 0 |
| Validation | 50-74 | 8 | 2 | 6 |
| Validation | 75-98 | 2 | 0 | 2 |
| Protected diagnostic | Any eligible band | 16 | 8 | 8 |

The 54 fresh roots comprise 44 training candidates from 42 assigned families and 10 validation roots from 10 families. The eight diagnostic roots come from five protected historical arena families; seven have 25-49 active seeds and one has 50-74. They remain excluded from future training. Validation labels are now consumed development evidence, not a sealed release test.

There were no natural strict ordinary observations in the 75-98 band. The small pool also fell short in 50-74. These are measured coverage gaps, not grounds to relax the rule or invent positions. The selected legal-action histogram is 2:2, 3:15, 4:17, 5:14, 6:11, 7:3. The fresh observations have 3-7 legal actions. No single-legal-action root inflates the reported policy agreement.

All 62 selected roots received complete full-root H8 targets under the declared two-million-node cap; zero attempts were budget-rejected. A complete root target includes an H8 minimax estimate and H7 successor action values, with report and root links. These are approximate H_v1 finite-horizon labels, not exact game solutions. A future rejected bounded attempt would remain recorded without becoming a complete label.

## Fixed-model diagnostic findings

The following errors and regrets are in seeds against these approximate H8 targets. They describe this small curated sample and do not prove game-theoretic blunders or improved engine strength.

| Cohort | Roots | Value MAE | Raw policy H8 regret | Saved PUCT H8 regret |
|---|---:|---:|---:|---:|
| Protected diagnostic | 8 | 3.7933 | 3.1250 | 0.3750 |
| Training candidate | 44 | 6.9841 | 8.2273 | Not measured |
| Validation | 10 | 7.3486 | 10.8000 | Not measured |

The eight saved diagnostic PUCT choices are from their original arena clocks. Two disagree with the new H8 teacher; their mean regret is 0.375 seeds. No new PUCT searches or equal-time games were performed for this comparison. There is no corresponding saved PUCT decision on the 54 fresh roots, so raw-policy regret cannot establish their PUCT search quality.

The larger disagreements concentrate in 50-74: the 12 training candidates have value MAE 12.3182 and raw-policy regret 15.5833, compared with 4.9839 and 5.4688 on the 32 training candidates in 25-49. The two validation roots in 50-74 have MAE 22.0479 and regret 45.0; that sample is far too small for a population claim. The eight validation roots in 25-49 have MAE 3.6737 and regret 2.25. No confidence intervals or strength claims are attached to these descriptive means.

This supports a future bounded study of natural 50-74 positions and teacher-horizon stability before bulk training or large pilot weights. It does not identify whether model error, shallow teacher targets, rare trajectories or selection effects cause the disagreements. A subsequent training or sampler experiment needs its own frozen objective, cohort ownership and equal-time gates against both baselines.

## Offline sampling review

`build/sampling-review.json` compares four probability distributions over the unchanged 7,669 control-training roots, 4,576 assigned families and 14 strata. Expected draws below use a hypothetical 384,000-draw budget solely for comparison. The accepted historical optimizer draws were reconstructed from their recorded seed; no new draws were used for training.

| Proposed draw distribution | Root effective support | Family effective support | Maximum expected draws on one root | Probability on the two tiny strata, nine roots total |
|---|---:|---:|---:|---:|
| Uniform root | 7,669.0 | 3,503.5 | 50.1 | 0.1174% |
| Uniform stratum | 407.3 | 381.9 | 6,857.1 | 14.2857% |
| Square-root stratum count | 6,471.6 | 3,219.4 | 637.9 | 1.4073% |
| 90% uniform root + 10% uniform stratum | 6,508.5 | 3,224.5 | 730.8 | 1.5342% |

With N roots, S strata and n_s roots in a stratum, these probabilities per root are respectively 1/N; 1/(S*n_s); 1/(sqrt(n_s)*sum_s sqrt(n_s)); and 0.9/N + 0.1/(S*n_s). The last two retain rare-stratum emphasis while reducing repeated exposure to the smallest groups. The historical uniform-stratum run actually drew the nine roots in its two smallest strata **54,873 times out of 384,000** (14.2898%).

Effective support means 1/sum(p_i^2), computed separately for roots and aggregate assigned-family probabilities. It describes distribution concentration. It is not gradient ESS, an importance-weight ESS or a number of independent games. The full review also reports positive minimum probabilities, low-active-seed coverage, history-type exposure, family-size concentration and expected distinct roots/families. Family mass is adjusted for its root count; assigned family IDs do not prove statistical independence.

The historical unit-weight loss with uniform-stratum draws optimizes a stratum-balanced empirical risk. Changing the draw distribution while keeping unit loss weights changes that objective, rather than merely reducing variance. Lower tiny-stratum mass trades some rare-mechanism emphasis for less repetition and broader support. Corpus counts do not define natural gameplay frequencies. No common target distribution or importance correction is assumed.

Square-root count and the exact 90/10 mixture are future controlled-test proposals based on exposure arithmetic. This review establishes neither proposal's superiority or strength gain. No optimizer is running, no new sampling weights or pilot weights have been applied, and no rare or endgame rows were removed. The tiny new pilot is kept separate from the existing corpus.

## Acceptance and recovery

Engineering acceptance comprises **12 distinct new methods: four author checks and eight independent audit checks**. The independent audit verifies the original draw replay, probability formulas, all 256 literal histories, protection of old roots and trajectories, complete-pool selection, full H8 labels and every diagnostic metric. It replays 336 principal-variation paths containing 2,688 choices, checks 62 literal model forwards and recomputes two fresh H8 teacher reports. These exhaustive verifications are evidence within the eight methods, not extra test counts.

Collection and labeling workers were actually killed before and after their durable commits. The resumed journals retained committed prefixes and matched fresh semantics. Corrupt identities, rehashed invalid publications, partial labels, inconsistent report targets and zero-work claims were refused. Retained failed harness probes are diagnostic evidence with their executed source snapshots; they are not counted as successful methods. See `docs/DATA_PILOT_INDEPENDENT_AUDIT.md` and `build/data-pilot-audit.json`.

The producer and diagnostic report completed-rerun proofs preserve artifact bytes and modification times. A completed producer rerun explicitly prohibited `Native.analyze` and performed zero teacher searches. All 1,834 frozen baseline inputs remain unchanged, including the 1,829 prior accepted inputs and completed control diagnosis. The aggregate acceptance binds new sources, journals, reports, final documentation and retained recovery evidence. Mutable observer state, screenshots and the acceptance's own rerun are excluded to avoid recursive bindings.

Run or resume the exact declared pilot with:

```powershell
python -m congklak.data_pilot_progress run --tool ordinary-data-pilot -- python tools/data-pilot.py all
```

The same command validates durable identity and resumes the committed family/attempt prefix. An unfinished teacher root is recomputed; it does not resume an in-memory search frontier. Once complete, verify the closed publication with:

```powershell
python tools/data-pilot.py verify
python tools/analyze-data-pilot.py
python tools/record-data-pilot.py --check-only
```

The last command reads and validates current evidence, including deterministic history replay; it does not run recovery fixtures, training or new teacher searches. Repeating completed commands under unchanged inputs preserves completed bytes and modification times. Identity or source mismatches are refused. Use the existing run directory rather than creating a new directory to resume.

Author and independent fixture commands are **prepublication operations** that create fresh retained evidence and rewrite their reports. After immutable acceptance, use `record-data-pilot.py --check-only`; a renewed fixture campaign requires a new evidence version. Do not overwrite sealed reports to refresh durations or run IDs.

The local observer is available at `http://127.0.0.1:18766/` and reads the new pilot cards alongside prior phases. It distinguishes committed history prefixes, selected roots, accepted labels, pending/rejected attempts and cap shortfalls. The model remains fixed, training remains stopped, and no strength promotion is implied by a completed engineering audit.
