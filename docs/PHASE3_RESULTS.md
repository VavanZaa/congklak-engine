# Native learned search and corrected training corpus

The optional C++ neural evaluator/search, corrected depth-six corpus, four CUDA training runs and equal-time comparison are implemented. **H_v1 remains the default.** The selected 70,145-parameter W256 candidate scored 43.75% against H_v1 in 24 paired games and loses search depth to inference cost. Aggregate regression improvements do not establish stronger play. No full-game solution or energy advantage is claimed.

The [progress dashboard](PROGRESS_REPORTING.md) reports all flow stages, committed work counts, latest results, evidence and next improvements. It also streams per-tool progress through a wrapper without changing frozen generation or training identities.

## Native simulation, evaluation and retrieval

`build/congklak_neural.dll` is separate from the frozen baseline DLL. It reuses the tested simulator and adds a scalar double-precision evaluator, score units of 1/256 seed, terminal-first alpha-beta/PVS, full-window root alternatives and canonical PV reconstruction. Search values remain fixed-player-0 MAX/MIN; bonus turns and passes retain their correct perspective. Full-state, exact-horizon TT entries are local to a model/search. An 8,192-entry full-pit/controller evaluation cache stores the quantized residual; the store difference is an exact additive offset. No oracle values enter neural predictions.

`congklak.neural` checks exported model identity and checksums, validates/replays each completed report, and commits completed depths using exclusive-writer SQLite FULL/WAL transactions. Changing model, source, rules or DLL requires a new job. An interrupted first iteration provides no recommendation. Quarantined model folders are refused. The original H_v1 DLL, its search semantics and original trainer source remain unchanged.

The selected W64 and W256 exports were each checked on all 1,000 new development roots. Native versus independent scalar reference error was at most 2.85e-14 seeds; CPU PyTorch versus native error was at most 9.14e-6 seeds. Quantized units matched exactly. Evidence: `build/native-export-parity.json`.

## Corrected database, weights and pruning

Current corpus: `data/stronger-10000-v2`, semantic digest `0c700087f2650a8685663ba0dbc097cd34517b8d77f7e70f3c5921e8721f9949`. It contains **8,000 train / 1,000 development / 1,000 sealed-final-test** positions, each with complete root-action labels from frozen H_v1 at depth six. Training has no oracle ancestry or learned teacher. Labels are finite-horizon estimates, not game-theoretic truth.

The 8,000 collection units mix legal random/heuristic/search histories with targeted ordinary counterfactual stable states. Sources retained: 1,297 constructed ordinary, 2,389 heuristic history, 3,311 random history and 3,003 search history. Constructed positions are explicitly marked counterfactual, without a reachability claim. Ordinary development coverage is now 93 / 93 / 73 rows for 5–24 / 25–49 / 50–74 active seeds; the original pilot had 127 / 12 / 2.

Admission imports 20,972 prior state/quotient exclusion keys and all 1,993 unique prior family IDs, including IDs whose archive payloads were pruned. All K<=4 states are withheld from this corpus. Retained states, quotient aliases, owning families and permanent exclusions are isolated across generations and holdouts. Sampling selects a primary stratum then a row uniformly with replacement: P=q and unit loss weights; inclusion probability is recorded separately. Unselected payloads are pruned only after a checked immutable generation and catalog commit. Permanent exclusions and collection identities survive pruning.

**Resolved lineage issue:** the first depth-six campaign omitted quotient aliases for prior nonretained state tombstones. A final audit found two affected training rows. `data/stronger-10000`, its four `training-stronger-*` models and `build/neural-benchmark-v2.json` are quarantined evidence, excluded from selection/promotion. The import now includes every old tombstone alias and pruned family identity. The V2 corpus was rebuilt and all four models retrained from scratch. Corrected-corpus checks found zero old state, quotient or family overlaps across all 10,000 rows. The original pilot remains frozen.

## Training and measured search

Training used the existing PyTorch 2.6.0+cu124 / CUDA 12.4 interpreter and RTX 4080. Widths 64 and 256, seeds 2718 and 31415, each ran 2,000 FP32 AdamW steps at batch 512. Selection uses finite-development uniform-stratum MAE at fixed checkpoint opportunities. The final-test sets remain sealed.

| Width / seed | Selected step | Dev MAE, seeds | Uniform-stratum MAE, seeds |
| --- | ---: | ---: | ---: |
| 64 / 2718 | 2,000 | 8.038 | 7.941 |
| 64 / 31415 | 1,900 | 7.966 | 8.030 |
| 256 / 2718 | 2,000 | 7.808 | 7.618 |
| 256 / 31415 | 1,900 | 7.897 | 7.383 |

H_v1's raw regression MAE is 12.748 on these depth-six development labels. These values are not comparable to the old pilot's depth-four targets. The selected W256 seed 31415 still regresses badly in the 50–74 ordinary stratum: MAE 24.766 versus H_v1 11.822 on 73 positions. `build/curriculum-selection.json` pins the selected exports.

The final `tools/benchmark-neural-v3.py` experiment uses 100 eligible finite-development roots and 48 separate exact references. Prior training quotient aliases are withheld. Each engine receives the same 25/100 ms budget, TT/PVS settings and fresh working caches; engine execution order rotates. Finite references are H_v1 depth-six proxies, not truth.

| Budget | Engine | Agreement with H6 proxy | Proxy regret, seeds | Mean completed depth |
| --- | --- | ---: | ---: | ---: |
| 25 ms | H_v1 | 84% | 0.46 | 11.24 |
| 25 ms | W64 | 81% | 0.54 | 10.16 |
| 25 ms | W256 | 82% | 0.65 | 7.34 |
| 100 ms | H_v1 | 86% | 0.36 | 12.53 |
| 100 ms | W64 | 83% | 0.57 | 11.49 |
| 100 ms | W256 | 83% | 0.49 | 9.31 |

All engines selected an optimal move on all 48 exact-reference cases at both budgets, without oracle prediction. The exploratory arena uses 12 distinct starting families, both seat assignments, 20 ms per choice and a 512-choice cap. All 24 games reached terminal outcomes with no fallback choices: **W256 10 wins / 13 losses / 1 draw**, paired score 43.75%, paired bootstrap 95% interval [33.33%, 50%]. The sample is small and does not establish improvement. Committed benchmark cases/games are resumable, but this final benchmark was not successfully interrupted during testing; no benchmark hard-kill verification is claimed.

## Acceptance and interruption recovery

Fifteen new checks pass: six native-neural, three corpus, one current CUDA training and five progress-observer checks. Existing 62 engine checks and seven original-training checks remain tied to unchanged inputs. `build/phase3-acceptance.json` binds the current results and artifacts; the logs retain individual tests.

Native tests compare independent literal-rule minimax against four TT/PVS settings, check original H_v1 equivalence and score symmetries, and hard-kill an analyzer after a committed depth before resuming to the same result as a fresh run. Checksum-valid corrupt saved reports and mismatched budgets are refused. Corpus tests hard-kill after labeling and during export before manifest commit, then resume to the same semantic corpus as a fresh run. The current W256 CUDA test restores optimizer, sampler and all random states and reaches bitwise-identical results to uninterrupted training. Prior training tests include full CUDA hard-kill recovery. Progress tests cover uncommitted WAL visibility, corrupt/missing evidence, reused-process IDs, wrapper failures and protected paths.

## Commands and next implementation

```powershell
# Corrected generation: idempotent when complete; resumes its committed work.
py -3.12 -m congklak.progress run campaign build

# Existing CUDA environment; same command restores the committed training state.
py -3.12 -m congklak.progress run training-v2 train --run data/training-stronger2-w256-seed31415 --width 256 --seed 31415

# Explicit experimental model, same position/run directory for resumption.
py -3.12 -m congklak.progress run neural analyze --model data/training-stronger2-w256-seed31415/model.json --position build/opening-position.json --run data/neural-demo --depth 8 --ms 5000

py -3.12 -m congklak.progress serve
```

Implement ordinary-state features and faster inference as controlled challengers next: relay/capture opportunities and mobility features, followed by measured arithmetic/feature optimizations. Keep the fixed current datasets and held-out families for comparable tests. Gather a fresh isolated corpus only when new targets or coverage justify it. Repeat equal-time comparisons and a larger paired arena before promotion. Increasing model capacity again is not supported by these results.
