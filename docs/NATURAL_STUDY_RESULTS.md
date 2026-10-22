# Natural 50-74 and H8/H10 teacher study

The bounded study produced **21 fresh ordinary roots and 15 unchanged follow-ups, with all 36 roots receiving complete paired H8/H10 targets**. There were no H8 or H10 budget rejections. The 21 fresh roots comprise 17 training candidates and four validation roots, leaving cap shortfalls of seven and four. No training, active weights, pruning or promotion changed; H_v1 remains the default engine.

The fixed retained model is `data/quality-train-v1/model.json`, SHA-256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. All 2,058 previously sealed inputs and the prior acceptance/rerun artifacts remain unchanged: 2,060 preserved inputs in `build/natural-study-baseline.json`. This study does not repeat or recount the accepted historical sampling review.

## Natural coverage and ownership

The frozen plan uses 512 new deterministic standard-opening history prefixes, seed 20261021: 256 random histories and 256 frozen-policy epsilon-0.25 histories after eight random opening choices. Prefixes stop below 25 active seeds, on terminal states or at 192 choices. They are not 512 terminal games.

Selection considers only strict ordinary roots with 50-74 active seeds. Every legal root action must avoid pass, capture and bonus mechanisms. The entire pool is generated before candidate collision closure. Both sides of any training/validation quotient collision are excluded, with one fresh root per family and no quota filler or automatic expansion.

The pool yielded 26 ordinary observations and 26 unique candidate quotients. Three observations overlapped protected old roots; zero candidate quotients collided across fresh cohorts. Selection retained 21 roots from 21 assigned families. All old pilot roots and its complete 256-family pool are protected alongside the inherited registry. Shared history prefixes remain possible; family IDs and selected-root separation do not establish independent games.

| Origin and original cohort | Cap or required follow-ups | Selected | Shortfall | Complete H8 | Complete H10 | Paired denominator |
|---|---:|---:|---:|---:|---:|---:|
| Fresh training candidate | 24 | 17 | 7 | 17 | 17 | 17 |
| Fresh validation | 8 | 4 | 4 | 4 | 4 | 4 |
| Follow-up training candidate | 12 | 12 | 0 | 12 inherited | 12 | 12 |
| Follow-up validation | 2 | 2 | 0 | 2 inherited | 2 | 2 |
| Follow-up protected diagnostic | 1 | 1 | 0 | 1 inherited | 1 | 1 |

All 15 old 50-74 roots retain their original cohort and their exact old root/H8 items. Protected diagnostics remain excluded from training. Validation is now consumed development evidence, not an untouched release test. No previously observed root was reassigned to fill a fresh quota.

There are **57 newly computed complete targets: 21 H8 and 36 H10**, plus 15 unchanged inherited H8 references. Fresh H8 searches use a two-million-node cap; all selected fresh and follow-up roots receive an H10 attempt under a five-million-node cap, regardless of H8 completion. Both use time limit zero, TT16, PVS and the declared 512 MiB search limit. Complete-root H8 targets have H7 successor action values; H10 targets have H9 successor values. A bounded partial result is retained as a rejected attempt and cannot become a complete target.

Every selected root is choice-bearing: legal-action counts are 3:4, 4:6, 5:15, 6:9 and 7:2. No one-legal-action root inflates agreement. All 36 selected roots are in the complete-H8 AND complete-H10 intersection; no root is unpaired in this run. The schema nevertheless keeps explicit completion/rejection counts and null comparative metrics for unpaired roots.

## Teacher stability and fixed-model disagreements

H8 and H10 are finite-choice H_v1 estimates. H10 is deeper evidence, not exact game truth. Comparisons below use identical roots and the same frozen model prediction; neither model training nor improved playing strength is implied.

Across all 36 paired roots, best-move sets changed on six and were disjoint on those six. The lexicographically first H8-best move remains H10-best on 30/36 roots. Mean absolute target shift is 3.0278 seeds, and the mean absolute action-score shift is 4.0378 seeds. The mean H10 regret of the declared H8-best choice is 0.4722 seeds; tie-aware minimum and maximum regrets are also retained in the machine report.

| Origin/cohort | Pairs | Value MAE against H8 | Value MAE against H10 | Raw-policy regret against H8 | Raw-policy regret against H10 | Disjoint best sets |
|---|---:|---:|---:|---:|---:|---:|
| Fresh training candidate | 17 | 15.7313 | 18.3196 | 9.2941 | 10.8235 | 3 |
| Fresh validation | 4 | 9.7952 | 13.2952 | 5.7500 | 8.7500 | 1 |
| Follow-up training candidate | 12 | 12.3182 | 14.0763 | 15.5833 | 19.5000 | 2 |
| Follow-up validation | 2 | 22.0479 | 21.0479 | 45.0000 | 51.0000 | 0 |
| Follow-up protected diagnostic | 1 | 9.8785 | 11.8785 | 9.0000 | 15.0000 | 0 |

Overall model MAE changes from 14.1224 to 16.3195 seeds solely because the teacher target changes. Against H10, the same predictions have higher absolute error on 26 roots, lower error on seven and equal error on three. These are descriptive development-sample counts. The four fresh validation roots, two validation follow-ups and one diagnostic follow-up are too few for population or causal claims; no confidence intervals are presented.

The substantial frozen-model disagreements remain at both horizons. Deeper labels do not make the ordinary-midgame errors disappear, while six move-set changes show that teacher horizon still matters. This study does not establish that either horizon is minimax-correct, identify a specific feature/model defect or measure PUCT strength. No new PUCT searches or equal-time games were run.

Before a training or weighting change, the next experiment should predeclare how it handles horizon-disagreeing targets and how much natural midgame coverage it needs. The measured shortfalls do not authorize an automatic larger collection. The 17 new training candidates do not justify large weights or bulk training. Any later trained candidate needs protected ownership, a declared sampling objective and equal-time gates against both retained baselines.

## Work, acceptance and recovery

The newly executed H8 attempts consumed 1,740,999 reported nodes and 24,578,715 deposits; H10 attempts consumed 20,934,515 nodes and 255,484,604 deposits. Their native report timers sum to 538 ms and 5,848 ms respectively. The inherited H8 work is excluded from these new-work totals. These timers exclude surrounding admission, collection, validation, audit and wrapper costs, and are not an end-to-end efficiency or energy measurement. The producer's separate initial-run timer is 9.532 seconds and also excludes subsequent verification/audit work.

Engineering acceptance consists of **12 distinct new methods: four author checks and eight independent audit methods**. The independent audit checks all 512 literal histories and their RNG/policy choices, complete-pool selection, original references, every complete label and rejection rule. It replays 430 principal-variation paths containing 3,870 choices, verifies 36 literal model forwards and independently recomputes one H8 and one H10 report. These verifications occur inside the eight methods and are not additional test counts.

Six workers were actually killed: collection, fresh H8 and H10, each before and after durable commit. H8 fixtures explicitly distinguish two inherited reference rows from newly committed target rows. Resumes preserve committed prefixes and match fresh search semantics/counters. Identity, original ownership, old reference, TT/PVS, partial-label and publication corruptions are refused. A real node-capped H10 partial and a synthetic unpaired case verify that incomplete targets cannot enter complete-pair means. See `docs/NATURAL_STUDY_INDEPENDENT_AUDIT.md` and `build/natural-study-audit.json`.

The completed-production rerun preserves SQLite, manifest, writer-lock and result bytes/mtime while prohibiting `Native.analyze`; it performs zero new teacher searches. The stability-report rerun preserves all four closed production/analysis artifacts and rechecks every preserved baseline hash. Repeated completed commands validate deterministic histories and references rather than adding new targets. The aggregate seals current sources, reports, raw recovery evidence and final documentation; mutable observer state, browser output and the acceptance's own rerun proof are excluded to avoid recursive bindings.

Run or resume the declared study with:

```powershell
python -m congklak.natural_study_progress run --tool natural-teacher-study -- python tools/natural-study.py all
```

An interrupted run resumes its committed family/teacher-attempt prefix under the same immutable identity. An unfinished teacher root is recomputed; its in-memory search frontier is not checkpointed. To validate a completed run and its descriptive report:

```powershell
python tools/natural-study.py verify
python tools/analyze-natural-study.py
python tools/record-natural-study.py --check-only
```

The read-only recorder validates current evidence and replays deterministic histories without new teacher searches, fixture reruns or training. Author and independent fixture suites are prepublication operations that create fresh evidence and rewrite their reports. After sealing, use `--check-only`; renewed fixture campaigns require a new evidence version. Identity/source mismatches are refused.

The local observer runs at `http://127.0.0.1:18766/` and exposes six new cards alongside the frozen earlier phases. Its source, compilation and backend snapshots were checked. Browser visual verification was unavailable under the current browser access policy, so no new screenshot or browser-verification claim forms part of acceptance. The cards separate committed prefixes, fresh/inherited targets, paired denominators, rejects, cap shortfalls and engineering audit status. No training, active weight change or promotion is implied by completed engineering checks.
