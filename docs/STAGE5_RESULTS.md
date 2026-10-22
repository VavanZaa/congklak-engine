# Stage 5: ordinary-state features and faster inference

The versioned fast native evaluator, eight matched training runs and equal-time tests are complete. **H_v1 remains the default.** Packed inference preserves the old scores and usually searches deeper at the same deadline. The feature model reduces raw error in two ordinary strata, but worsens overall error and ordinary move selection; it has not passed promotion.

Three agents worked separately on training, native inference and an independent audit. The [auditor's report](STAGE5_INDEPENDENT_AUDIT.md) records literal-rule minimax, numeric parity, lineage and actual process-kill recovery. The [progress dashboard](PROGRESS_REPORTING.md) shows Stage 5 work, per-model committed steps, 407 position cases and 72 games.

## Changes and preserved contracts

`features_v3.py` adds 14 occupied and 14 singleton flags to the existing actor-relative 15 count/active inputs. These cheap discontinuity features expose relay stops and empty/single-seed capture conditions without simulating moves during inference. `training_v3.py` uses a separate source-pinned run schema, existing corrected train/development data and the existing CUDA PyTorch environment. Old trainers, datasets and models remain frozen.

`native/neural_fast.cpp` packs dense weights by input and evaluates four output neurons together with AVX2 doubles. Each neuron's accumulation order is preserved, multiplication and addition remain separate, and compilation uses strict floating-point rules. CPU/OS dispatch has a scalar fallback; supported widths need not be multiples of four. The separate `congklak_neural_fast.dll` and `neural_fast` binding support both 15- and 43-input exports. Model/backend/source/DLL identity gates resumption; the old H_v1 and scalar-neural DLLs are unchanged.

Search retains fixed-player MAX/MIN semantics, terminal-first evaluation, residual rounding to 1/256 seed before exact store-offset addition, full-state/exact-horizon TT entries, model-local evaluation cache and full-window candidate/PV reconstruction. No oracle enters neural predictions. The analyzer commits only completed depths and refuses incompatible or corrupt checkpoints.

## Matched training results

The frozen plan compares 15/43 inputs, widths 32/64 and seeds 2718/31415: eight runs, each 8,000 FP32 AdamW steps, batch 512 and 4,096,000 draws. All 640 scheduled checkpoint files and sampler states were verified. Paired mean development uniform-stratum MAE chooses width, then the best seed within that width selects its export. Arena results never influence selection; sealed targets are not loaded.

| Selected representative | Parameters | Dev MAE | Uniform-stratum MAE |
| --- | ---: | ---: | ---: |
| 15 inputs / width64 / seed31415 | 5,249 | 7.548 | 7.419 |
| 43 inputs / width64 / seed31415 | 7,041 | 7.713 | 7.752 |

Both fail the predeclared ordinary-stratum no-regression gate against H_v1. The feature comparison includes 1,792 additional first-layer parameters; equal width/steps do not isolate representation from parameter count.

| Ordinary active seeds | 15-input MAE | 43-input MAE | H_v1 MAE | Positions |
| --- | ---: | ---: | ---: | ---: |
| 5–24 | 2.743 | 2.326 | 2.333 | 93 |
| 25–49 | 8.712 | 9.204 | 5.226 | 93 |
| 50–74 | 24.721 | 21.344 | 11.822 | 73 |

Across both width64 seeds, the feature model reduces ordinary MAE by about 17.2% in 5–24 and 5.8% in 50–74, while worsening 25–49. High ordinary predictions remain too favorable to the current actor: the development teacher's mean future residual is -9.466 seeds, versus +15.256 predicted by the selected count-only model and +10.197 by the feature model. These are H_v1 depth-six teacher estimates, not optimal-game values. Longer training alone does not resolve this bias.

## Inference speed and equal-time choice tests

The speed experiment uses seven blocks of 8,192 uncached evaluations per model and 12 fresh-cache depth-six positions with three repetitions. Scores, optimal moves, PVs and work counters match between scalar and packed paths. Observed median depth-six wall speedups versus the frozen scalar evaluator were **1.35× for the old W64 and 2.53× for W256**. Single-forward AVX versus packed scalar medians were about 3.2–5.6× across models.

Timings vary substantially on this unpinned hybrid-core CPU; some paired cases were slower. API timing includes JSON transfer and the old evaluate endpoint computing its residual twice. The paired search measurements and raw ranges are in `build/stage5-speed-results.json`; these medians are exploratory, not universal latency or energy guarantees.

The frozen equal-time protocol compares H_v1, old W256 scalar, the identical W256 packed model, selected count-only packed W64 and selected feature packed W64. It rotates engine order, uses fresh TT/evaluation caches, max depth16, TT16 MiB/PVS and 25/100 ms native deadlines. Wall measurements include setup and report overhead. Original 100 finite roots, 48 known endgame diagnostics and 259 current ordinary development roots remain separate cohorts.

On the 259 ordinary roots:

| Engine | Agreement with H6 at 25 ms | At 100 ms | H6 regret at 25 / 100 ms, seeds |
| --- | ---: | ---: | ---: |
| H_v1 | 82.24% | 79.15% | 0.405 / 0.556 |
| Old W256 scalar | 76.83% | 81.47% | 0.819 / 0.699 |
| Same W256 packed | 79.15% | 78.38% | 0.749 / 0.761 |
| Count-only packed W64 | 79.15% | 78.76% | 0.587 / 0.703 |
| Feature packed W64 | 74.90% | 73.36% | 0.807 / 0.927 |

The same W256 weights increase mean completed choice depth from 5.52 to 6.78 at 25 ms and 7.36 to 8.59 at 100 ms on this cohort. Deeper search does not guarantee better agreement with a shallower H6 proxy or a better true-game result. The feature model's raw error gains did not translate into better choices.

**Legacy diagnostic limitation:** metadata-only auditing found 16 of the 48 original K4 roots have quotient aliases in the old sealed split. They are known exact implementation diagnostics, not an independent held-out generalization test and not promotion evidence. All models chose an oracle-optimal move on the 48 roots at both deadlines, with oracle prediction disabled. Current corrected holdouts remain isolated; sealed target bodies were not read. Finite/current cohorts are checked separately for holdout and training aliases.

## Paired games and interpretation

The main arena uses 24 distinct current-development families, eight each ordinary/capture/bonus, active seeds>=25, both seat assignments, 20 ms per choice and a 512-choice cap. The speed-only arena compares identical old W256 weights over the first 12 paired starts. All **72 games finished**, with zero fallback choices.

| Comparison | Wins / losses / draws | Score | Paired bootstrap 95% interval |
| --- | --- | ---: | --- |
| Feature packed W64 vs H_v1 | 23 / 23 / 2 | 50.00% | [50%, 50%] |
| Packed vs scalar, same W256 weights | 12 / 11 / 1 | 52.08% | [50%, 56.25%] |

Every feature-arena pair had score0.5, making its bootstrap interval degenerate. That describes this small fixed sample and does not prove equal strength or precise population performance. Starting store advantages and terminal game margins are examined independently by the auditor. No Elo, full-game solution, lower-power or promotion claim follows.

All 24 main and 12 speed-control starting families have |store difference|<active seeds, so no starting WDL is mathematically fixed. The independent auditor also computed descriptive paired seed-margin advantages: feature vs H_v1 **+1.71 seeds**, bootstrap95% **[-0.54,4.08]**; sameweights packed vs scalar **+4.33 seeds**, **[1.00,8.17]**. These secondary metrics were added for the original margin objective, without changing selection or promotion criteria. The feature interval includes zero; the small speed-control experiment supports further runtime testing, not a broad strength claim.

## Recovery, acceptance and next work

The native/evaluation agents passed five and four checks; the independent auditor passed nine; progress reporting passed eight, for 26 total. Actual kills cover an analyzer after a committed depth, CUDA training before/after checkpoint catalog boundaries, and a benchmark after its first complete case. Restart preserves committed cases and rejects checksum-valid score corruption. Operational retention preserves logs pinned by acceptance manifests.

A real Windows progress-file access-denied error stopped the production benchmark after 189 committed cases. Progress publication now uses unique temporary files and bounded retries for transient sharing errors. A dedicated check injects a transient sharing error to verify the retry. The same benchmark resumed from189 and completed407 cases/72 games without changing models, selection or source identity. SQLite owns recovery; operational logs do not authorize results.

The next evaluation experiment should target **relative action scores and ordinary-state residual bias**, rather than another capacity increase. Preserve the current experiments and compare an action-aware objective or stronger, better-covered teacher labels using a separately frozen plan. For runtime promotion, add repeated P-core-affinity timing blocks and more contested paired starts, retaining seed-margin metrics as well as WDL. Current default remains H_v1.

```powershell
py -3.12 -m congklak.progress run stage5-benchmark run
py -3.12 -m congklak.progress run neural-fast analyze --model data/training-stage5-f43-w64-seed31415/model.json --position build/opening-position.json --run data/stage5-neural-demo --depth 8 --ms 5000
py -3.12 -m congklak.progress serve
```

Use the same command and run directory after interruption. Changing model, features or backend requires a new run directory. See `build/stage5-acceptance.json` and the independent audit for pinned results and limitations.
