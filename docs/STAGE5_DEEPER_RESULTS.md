# Deeper teachers and relay inputs

The deeper/relay experiment is complete. The selected H8/127 model passes both predeclared development gates: ordinary H8 value error and versus-H_v1 paired seed margin. The authoritative completion record is `build/stage5-deeper-acceptance.json`, backed by the independent audit and committed SQLite catalogs. H_v1 remains the analyzer default; this experiment does not automatically change deployment.

## Frozen experiment

The corpus keeps the same 8,000 training roots and 1,000 development roots from `stronger-10000-v2`. It adds complete, untrained H_v1 depth-eight values and root-action scores with successor horizon seven. It adds no successor examples and reads no sealed target bodies. Depth-six and depth-eight value targets remain separate objectives.

All 9,000 completed labels and relay tensors are published under semantic digest `f2878aac7ef0bee1a1358e36d981656dfb242683c31e245205790beae393b6c4`. The independent teacher audit rederived every tensor and replayed 15,020 candidate continuations. H8 changes the root value on 7,580 roots and the best-move set on 2,575 compared with H6. These are horizon differences, not full-game truth or measured playing-strength improvements.

Eight CUDA runs compare teacher depth 6/8, input count 43/127, and seeds 2718/31415. Each run uses two hidden layers of width 64, 8,000 AdamW steps, batches of 512, FP32 and a value-only Huber loss. Checkpoints commit every 100 steps: 640 files in total. Sampling uses uniform primary strata, replacement draws and unit weights; draws match across conditions with the same seed. The seven-output dummy policy head has no gradient or optimizer decay. Own-teacher development value error selects checkpoints and the representative seed for each condition. Cross-teacher metrics are descriptive and do not select models.

The 127-input encoder adds six exact transition features for each of seven hypothetical moves by each player: store gain, deposit count, relay count, capture, retained normalized control and terminal status. Empty hypothetical pits contribute zeros. Simulation cost is included in the native search deadline and work counters. The 43/127 models have 7,496/12,872 total parameters, including 455 frozen dummy parameters. Feature information and model capacity therefore change together; this is not a pure feature ablation at equal parameter count or equal FLOPs.

## Recovery from the stopped run

The preserved V1 teacher ledger contains 3,472 complete labels and 3,473 feature rows. Its strict resume check exposed inconsistent timing snapshots: 543 label work records used outer return timing while their reports used the earlier completed-depth timing, differing by one or two milliseconds. Scores, action values, traces and non-timing work fields match.

The versioned V2 migration preserves every original report, target and action score. Report-linked work fields use the completed-depth snapshot, while the original return timing and row digests remain explicit provenance. The old ledger and source remain unchanged. Independent checks compare every migrated row and reject checksum-valid changes to scores, work or source identity. The original incomplete native call's cause was not recorded and remains unknown.

## Speed and strength protocol

The production speed experiment measures the four selected models, including complete uncached feature extraction, scalar/AVX parity and fresh-cache fixed-depth search. The equal-time experiment has 391 positions: 100 older depth-six proxies, 32 known exact diagnostics, and 259 current ordinary depth-eight development proxies. Seven engines use 25/100 ms native deadlines with TT/PVS. Full validated call latency is also recorded; these deadlines do not guarantee identical end-to-end wall time.

Four arenas each use 36 paired families in both seats, totaling 288 games. Families exclude the prior arena's 24, but belong to development data used for checkpoint selection, so this is not a sealed generalization result. Each choice has a 20 ms native deadline. The primary measure is paired mean final seed margin with a predeclared 4,096-resample family bootstrap. A positive lower confidence bound, all complete pairs and zero fallback moves are required. Win/draw/loss scores are secondary. The same declared OS performance core is verified throughout; affinity is restored on normal and exceptional exits. Energy is not measured.

A learned model must also be no worse than H_v1 against H8 labels in every ordinary development stratum with at least 20 rows. Successful correctness or recovery checks alone do not establish stronger play.

## Training results

All eight runs reached 8,000 steps. Verification read all 640 checkpoints and checked matched sampler histories, selected exports and frozen dummy heads. Repeating the whole completed training and verification flow preserved all 675 durable files byte-for-byte and left their modification times unchanged.

| Teacher | Inputs | Mean H8 stratum MAE, two seeds | Selected seed / step | Selected ordinary H8 gate |
| --- | ---: | ---: | --- | --- |
| H6 | 43 | 9.5166 | 31415 / 3500 | Fail |
| H6 | 127 | 5.9958 | 31415 / 1600 | Pass |
| H8 | 43 | 9.1165 | 2718 / 2500 | Fail |
| H8 | 127 | 5.0836 | 2718 / 2200 | Pass |

The mean uses each seed's own-teacher-selected checkpoint, evaluated against H8 afterward. Representatives likewise use own-teacher selection. The deeper targets and relay inputs improve the descriptive errors in this comparison, with the capacity and FLOP differences stated above. The H6/43 selected value weights also exactly reproduce the earlier matched Stage 5 control, providing a reproducibility check.

| Ordinary stratum | Roots | H_v1 H8 MAE | Selected H6/127 MAE | Selected H8/127 MAE |
| --- | ---: | ---: | ---: | ---: |
| 5–24 active seeds | 93 | 2.6452 | 2.1549 | 2.0155 |
| 25–49 active seeds | 93 | 6.3011 | 4.3290 | 4.4159 |
| 50–74 active seeds | 73 | 14.4110 | 10.9169 | 12.0010 |

## Native speed results

All 176 durable speed units completed: four parity units, 28 microbenchmark blocks and 144 fixed-depth search units. Each selected model passed scalar/AVX and reference parity across 1,000 development roots. Fresh-cache fixed-horizon values, best moves and PVs match within each evaluator.

| Inputs | Complete uncached AVX forward | AVX versus scalar search gain | Value versus H_v1 search time |
| --- | --- | --- | --- |
| 43 | 0.736–0.738 µs | 1.64–1.66× | 1.79–1.88× |
| 127 relay | 2.312–2.356 µs | About 1.67× | 3.74–3.76× |

Ranges cover the two selected teacher conditions; individual ratios are medians from the frozen units. Complete forward includes exact hypothetical moves and both output heads. The value/H comparison has different leaf values and potentially different pruning work, so it is not an isolated kernel-cost ratio or a strength measure. Hypothetical relay features account for 80.7–80.9% of total deposited-seed work in the 127-input searches. This identifies a concrete cost to reduce in a later optimization experiment.

## Equal-time strength results

All 391 cases and 288 games completed. Every game reached a literal terminal state; no fallback move was used. The independent auditor recomputed all case summaries, literally replayed all 288 games, reproduced every paired family bootstrap and verified 765 same-depth H-only adapter comparisons.

| Candidate / opponent | Paired seed margin | Family bootstrap interval | Primary margin gate | W/D/L |
| --- | ---: | --- | --- | --- |
| H8 counts / H_v1 | +0.4444 | [−0.5556, +1.4722] | Fail | 35 / 1 / 36 |
| H8 relay / H_v1 | +1.3056 | [+0.2778, +2.4444] | Pass | 36 / 1 / 35 |
| H8 relay / H8 counts | +0.6944 | [−0.6667, +1.9444] | Fail | 35 / 2 / 35 |
| H8 counts / H6 counts | −1.0278 | [−2.1389, 0.0000] | Fail | 36 / 1 / 35 |

The tested H8/127 representative clears both the ordinary H8 value gate and its primary versus-H_v1 seed-margin gate. Its secondary WDL score is 50.69%, so the evidence is a seed-margin improvement, not demonstrated WDL superiority. The comparisons do not establish a playing-strength gain from deeper targets alone, or isolate a relay-feature advantage over the deeper count model. The H6/127 representative passes the value gate but has no predeclared versus-H_v1 arena in this experiment.

On the 259 current ordinary H8 proxy cases, H_v1 has better reference agreement and lower regret at both budgets. At 25 ms, H_v1 versus deep relay agreement is 92.66% versus 80.69%, with mean completed depths 10.31 versus 8.50. At 100 ms, agreement is 84.17% versus 76.83%, with depths 11.71 versus 9.92. Those are agreement with a fixed H8 reference, not full-game optimality: deeper search can choose differently, as H_v1's own agreement decreasing with the larger budget illustrates. The relay candidate gives up about 1.8 completed choice plies under these budgets.

The default remains H_v1; this trial does not automatically alter deployment. The next priority is to reduce relay-feature simulation cost, retain the verified feature/value semantics, and validate the candidate on fresh evaluation families. The arena families here were development families, and no energy or sealed generalization claim is made.

## Acceptance and interruption recovery

All 34 acceptance checks pass: five training, six native, thirteen independent and ten progress-observer checks. The independent test ledger binds three source-checked groups covering thirteen unique methods without skips or duplicate CUDA runs. It independently read all 640 tensor checkpoints, reconstructed each dummy head from its seeded initialization, and replayed 4,096,000 sampling draws for each of two seeds.

Actual process kills cover native analysis, teacher labeling, CUDA before checkpoint commit, CUDA after final commit before export, benchmark case commit, and two arena recovery boundaries. Completed units retain their bytes. Saved identity changes and checksum-valid corruptions of scores, regrets, margins, feature work or exported artifacts are refused. CPU affinity restoration is checked for normal and exceptional exits. The arena tests operate on a separate SQLite backup and leave the 391 production cases and 288 games untouched.

Two prepublication audit diagnostics are retained. A fixture started before the final dashboard wording change correctly refused the changed reporting-source identity. A full-audit guard initially expected the user option name `avx2`; the frozen bindings actually report `avx2-double-packed`. The auditor corrected its exact metadata expectation and reran the full audit. Production sources, models, measurements and comparisons did not change for that correction.

## Resume commands

Run from the project directory using the existing CUDA interpreter selected in `build/training-runtime.json`:

```powershell
py -3.12 -m congklak.progress run stage5-deeper-training generate
py -3.12 -m congklak.progress run stage5-deeper-training train
py -3.12 -m congklak.progress run stage5-deeper-training verify
py -3.12 -m congklak.progress run stage5-deeper-benchmark run
py -3.12 -m congklak.progress run audit-deeper
```

Repeat the same command and directory after an interruption. Uncommitted labels, optimizer steps, search depths or games are recomputed; completed units retain their checksummed identity. A final training checkpoint committed before export can finish publication after restart. Finished training results, exports and catalogs are byte- and modification-time stable on repeat.

For experimental relay analysis, use the selected H8/127 model:

```powershell
py -3.12 -m congklak.progress run neural-relay analyze --model data/training-stage5-deeper-h8-f127-seed2718/model.json --mode value-none --position build/opening-position.json --run data/my-relay-analysis --depth 8 --ms 1000
# Repeat the identical command to resume; use a new directory for a different model or mode.
```

The dashboard at `http://127.0.0.1:18765/` shows deeper data, training, native evaluation, strength and audit work, plus individual training catalogs. Read the [independent audit](STAGE5_DEEPER_INDEPENDENT_AUDIT.md) alongside the final result tables when published.

The production-model opening demo in `data/stage5-deeper-demo` exhausted its 25 ms budget after committing depth one. The same model/mode/root/run resumed at depth two with a 1,000 ms budget and completed depth eight. Its final score, best moves and candidate PVs match a fresh depth-eight search. The selected opening pit is one-based pit 1; this is a conditional heuristic result, not a full-game proof. Evidence is `build/stage5-deeper-opening-recovery.json`; the independent checks separately exercise actual analyzer process kills.
