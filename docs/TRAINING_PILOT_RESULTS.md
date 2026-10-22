# Congklak pilot training

Executed on 14 October 2026. Two small evaluators were trained successfully on the frozen pilot. They are experimental candidates; the native analyzer still uses H_v1. These results validate the training pipeline and suggest useful learned features, but do not establish stronger equal-time search or a solution of Congklak.

## Existing CUDA environment

The reused interpreter is `D:\3D Project\Ministral-3-14B-Reasoning-2512\.venv\Scripts\python.exe`, with **PyTorch 2.6.0+cu124**, CUDA runtime 12.4 and an RTX 4080. A CUDA matrix operation passed. No packages in that environment were installed or changed. `tools/setup-training.ps1` verifies an existing interpreter and records it in `build/training-runtime.json`; it no longer downloads PyTorch. A different existing environment can be selected with `-Python <path>`.

Automatic approval review blocked removal of the partial duplicate download and unused new environment. These remain under `build/` and are not training dependencies. The download process was stopped.

## Frozen experiment

- Data: semantic SHA256 `125b7a81b41baeeeb3f44bdf4f7bac227423d564db106e717a88ccbe1ace7bcb`; native teacher SHA256 `d0066bb395db3323c46b372ab32f2c80e0a6845343282ec3cc894634d22faf3d`.
- Training reads **6,851** finite depth-four H_v1 labels. All positions with four or fewer active seeds are excluded. The model has no oracle training ancestry.
- Development reads **1,416** positions: 870 finite-horizon labels and 546 exact benchmark labels. Exact labels are diagnostics; checkpoint selection uses only finite-horizon development error. The **1,733 sealed-final-test labels are not loaded or evaluated**.
- Encoder: current controller's seven pits, the other seven pits and active-seed total, scaled by 98. Forced passes normalize the actor. Output reconstructs fixed-player-zero margin as `store0 - store1 + signed_active * tanh(residual)`; stores enter as an exact offset. This enforces player-swap antisymmetry, store-offset invariance and exact terminal values.
- Network: dense 15 → 64 → 64 → 1, ReLU hidden layers and tanh residual, **5,249 parameters**. No policy head, AMP, scheduler or self-play stage.
- Sampling: uniform primary stratum, then uniform row, with replacement. The declared target and draw distributions match (`P=q`); label coefficients and importance ratios remain one. These runs introduce no priority weighting or quality pruning.
- Training: 2,000 optimizer steps per seed, batch 512, AdamW learning rate 0.001 and weight decay 0.0001, gradient clipping 1. Huber loss uses delta 1 in reconstructed seed-margin units.
- Selection: lowest mean MAE across represented finite-horizon development strata at fixed 100-step checkpoints. Extra recovery snapshots do not create extra model-selection opportunities.

The labels estimate a finite choice horizon; they are not final-game truth. Earlier development sampling showed frequent depth-four/depth-six differences. A later production experiment still needs more diverse, deeper labels and imports of existing holdout exclusions before any cross-generation training.

## Development results

Both seeds used the same frozen configuration. Neither was selected through sealed-test results.

| Metric | H_v1 baseline | Seed 2718 | Seed 31415 |
|---|---:|---:|---:|
| Finite-horizon MAE, seeds (870 positions) | 9.2264 | 4.7827 | 4.5961 |
| Finite-horizon uniform-stratum MAE, seeds | 11.0304 | 6.4703 | 6.2635 |
| One-choice agreement with teacher's optimal set | 38.39% | 67.01% | 65.75% |
| One-choice mean teacher regret, seeds | 7.5253 | 1.9138 | 1.7655 |
| Exact-development MAE, seeds (546 positions) | 1.8480 | 1.4552 | 1.5974 |
| Exact-development one-choice optimal-set accuracy (544 nonterminal positions) | 93.57% | 98.35% | 98.35% |
| Exact-development one-choice mean regret, seeds | 0.2022 | 0.0551 | 0.0441 |
| Selected optimizer step | — | 1800 | 2000 |

The one-choice diagnostic applies every legal action with the native simulator, then ranks successor states by raw neural value or raw H_v1. It accepts tied optimal actions and respects MAX0/MIN1 and extra turns. Predictions use no oracle lookups. This compares cheap successor evaluations, not the existing depth-four analyzer against neural search. Exact and finite-horizon references remain separate.

**Ordinary-move strata regress.** For active counts 5–24, their MAE rises from 1.5669 to 2.6276/2.6622 over 127 positions. For 25–49, it rises from 3.0000 to 5.3847/5.6303 over 12 positions. The two 50–74 ordinary positions regress from 3.0000 to 23.2349/21.8233. Seed 2718 also regresses in the exact ordinary stratum. Aggregate improvements therefore do not satisfy the promotion requirement concerning important strata. These sparse bins need better coverage and investigation rather than deletion or extra weighting chosen from sealed data.

## Runtime, export and retention

Training took 5.438 and 5.172 seconds respectively after validated data loading, including model/optimizer setup, updates, evaluations and checkpoint writes. This is a tiny supervised pilot, not an estimate for generating or training on a production corpus. Each run reserved at most 71,303,168 bytes of PyTorch GPU allocator memory. Peak process working set was approximately 1.03 GiB for the first run. Allocator figures exclude driver and other applications' memory. No joule or whole-machine energy measurements were made.

Each run retains twenty immutable checkpoint files and its catalog, selected model export and results, approximately **2.51 MB** per directory. No large duplicated replay dataset is created. Existing frozen data, holdout exclusions and fallback artifacts remain protected; a general artifact collector is deferred. The small retained checkpoints provide reproducibility and recovery evidence.

Independent scalar Python inference from exported JSON agrees with CPU PyTorch within 0.0000072 seeds for seed 2718 and 0.0000057 for seed 31415 on all development roots. This is a reference decoder, not a measured fast native evaluator. Native search integration, quantization and evaluator latency benchmarks remain next work.

Selected model weight SHA256 values:

- Seed 2718: `c147d6bedf0c69dd07658b572c44fd56e38cd4f2282910ad2ae6a70454edeed3`.
- Seed 31415: `6beb258d3736d904a4f9aba9b887529fea0199e7cbcf0b6093946d3f684c232e`.

## Recovery and acceptance

Checkpoints include model and optimizer, sampled-row progress, sampler/Python/PyTorch CPU/CUDA random states, selected-model state, dataset identity, exact source/framework/device identity and declared absence of scheduler, AMP and prefetch. CUDA is synchronized before persistence. A complete file is fsynced and verified before the FULL-synchronous SQLite catalog commits it. A file alone is not authoritative.

Seven training acceptance tests pass without skips. They verify split admission, frame/score invariants and independent decoding; CPU and CUDA bitwise restart equivalence; forced process kills at blocked pre-commit and post-commit boundaries; truncated-latest fallback; single-writer and incompatible-run rejection; real SQLite-full recovery followed by completion; and cancellation at unscheduled step seven without altering selection. Forced kills before a first catalog commit leave an ignored orphan and restart from zero. Committed checkpoints survive restart. Physical power-loss testing was not performed.

The previous 62 engine acceptance checks were not unnecessarily rerun: their recorded source inputs and native binary hashes still match. Training checks are separate and require CUDA; they do not become hidden dependencies of the standard-library engine suite. Evidence: `build/training-acceptance.json`, `build/training-acceptance.log`, and each run's `catalog.sqlite3`, `result.json` and `assessment.json`.

## Commands and next implementation

```powershell
& .\tools\setup-training.ps1
& .\tools\train.ps1 train --run data/training-pilot-seed2718 --seed 2718
& .\tools\train.ps1 assess --run data/training-pilot-seed2718

# Same options and run directory resume an interrupted job.
# The completed runs are also safe to reopen with these commands.
& .\tools\train.ps1 train --run data/training-pilot-seed31415 --seed 31415

$runtime = Get-Content build/training-runtime.json -Raw | ConvertFrom-Json
& $runtime.python -m unittest discover -s tests -p training_acceptance.py -v
```

Ctrl+C requests a snapshot after the current completed optimizer step. Hard termination loses updates after the newest committed snapshot, which are deterministically recomputed. Resume requires unchanged source, dataset, total steps, optimizer options, framework, device and other identity settings. Changing an experiment requires a new run directory. JSON exports and result files are convenience outputs; checkpoints and catalog govern continuation.

Implement an opt-in native CPU evaluator and benchmark it against H_v1 with equal time budgets next. Preserve the accepted H_v1 evaluator. Add per-stratum tactical checks and development arenas with player swaps before any promotion. Address ordinary-stratum regressions and teacher quality before scaling to the planned 100,000-position corpus. Self-play, PUCT and larger networks remain separate experiments.
