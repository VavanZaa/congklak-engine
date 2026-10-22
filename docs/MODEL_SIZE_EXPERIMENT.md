# Larger-model experiment

Executed on 14 October 2026 with the existing PyTorch 2.6.0+cu124 environment and RTX 4080. Four larger models were trained and compared with the two existing 64-wide models. The **256-wide network is a useful search challenger**, with improved aggregate move-choice diagnostics. It remains experimental: the declared uniform-stratum error metric did not improve, and ordinary-move weaknesses persist. The accepted native analyzer still uses H_v1.

## Controlled comparison

All models use two hidden layers, the same encoder, constrained residual margin output and frozen pilot. The widths are 64, 128 and 256, with seeds 2718 and 31415 for each. Larger-model runs have separate directories; existing runs and the trainer source were preserved.

Each run uses 2,000 AdamW updates, batch 512, learning rate 0.001, Huber delta 1 in seed units and the same uniform-stratum sampler with unit weights. Sampled example sequences match across widths for each seed. The budget matches updates and data exposure, rather than elapsed GPU time or energy. Checkpoint selection retains the previously declared finite-horizon development uniform-stratum MAE criterion at 100-step intervals.

Training reads 6,851 finite-horizon positions. Development contains 870 finite-horizon and 546 exact-reference positions. All training positions have more than four active seeds. Exact references remain development diagnostics, with no oracle calls supplying neural predictions. The 1,733 final-test positions remain sealed. No new sampling weights or data pruning were introduced.

## Results

The table averages the two seeds at each width. Move metrics compare one legal choice followed by raw evaluator scoring against the teacher's optimal action set; tied best actions are accepted. These are development diagnostics, not games played by the full analyzer.

| Metric | Width 64 | Width 128 | Width 256 |
|---|---:|---:|---:|
| Parameters | 5,249 | 18,689 | 70,145 |
| Finite-horizon value MAE, seeds | 4.6894 | 4.6765 | 4.6253 |
| Finite-horizon uniform-stratum MAE, seeds | 6.3669 | 6.6420 | 6.4151 |
| Finite-horizon optimal-set move agreement | 66.38% | 65.23% | 69.20% |
| Finite-horizon mean move regret, seeds | 1.8397 | 2.0086 | 1.7293 |
| Exact-development optimal-set agreement | 98.35% | 98.25% | 99.17% |
| Exact-development mean move regret, seeds | 0.0496 | 0.0607 | 0.0294 |
| CPU forward latency, microseconds/position | 27.02 | 27.95 | 28.73 |
| Training seconds after data loading | 5.305 | 5.242 | 5.422 |
| Maximum PyTorch reserved GPU memory, MiB | 68 | 68 | 70 |

Width 256 has **13.36 times** the parameters of width 64. Its finite-horizon agreement is 68.39%/70.00% across seeds, versus 67.01%/65.75% at width 64. Its mean improvement is 2.82 percentage points, and mean move regret falls by approximately 6%. Two seeds and a reused development suite do not establish the uncertainty of generalization or an equal-time playing-strength improvement.

Width 128 gives mixed results and is not the preferred next challenger. Width 256 improves aggregate choices but loses on the predeclared uniform-stratum error criterion. Ordinary strata remain an obstacle:

| Finite-horizon ordinary stratum | Development count | H_v1 MAE | Width 64 mean MAE | Width 256 mean MAE |
|---|---:|---:|---:|---:|
| 5–24 active seeds | 127 | 1.5669 | 2.6449 | 2.4231 |
| 25–49 active seeds | 12 | 3.0000 | 5.5075 | 6.0203 |
| 50–74 active seeds | 2 | 3.0000 | 22.5291 | 23.5251 |

The two 50–74 ordinary positions also have higher mean move regret with width 256: 10 seeds versus 6.25 at width 64. The sparse bins need better coverage and diagnosis; removing difficult development rows would conceal this weakness.

## Verification and measurement scope

The comparison validates export checksums, reconstructed weight digests, identical rules/dataset/trainer/options and split ownership. All 1,416 development roots per model satisfy physical score bounds and agree with CPU PyTorch within 0.0001 seeds. The largest observed scalar-reference/CPU difference is approximately 0.0000116 seeds. Scalar reference decoding also scores all 4,708 legal successors per model. The existing 64-wide models reproduce their previously recorded move metrics exactly.

CPU latency uses one PyTorch CPU thread and batch size one, 256 preencoded positions, warm-up and five trials of 1,024 forwards. The table averages the per-model trial medians. It includes tensor slicing and forward execution; it excludes encoding, simulator transitions and search. Measurements were made sequentially in one process. PyTorch overhead can dominate a tiny model, so these timings do not establish the cost ratio of optimized native inference. No energy measurements were made.

The larger models each retain twenty committed checkpoints and a selected JSON export. The four new training directories use approximately 65.2 MB before the extra size-assessment JSON files. Frozen data and earlier models remain available for comparison and recovery. No large replay copy or speculative artifact collector was added.

The 256-wide seed-2718 recovery check deliberately blocks at step 700 after its file is verified but before the catalog commit, then kills the process. Restart restores step 600 and completes step 2,000. Its full final model, optimizer, progress, random-generator states and selected checkpoint state match the uninterrupted run bit for bit. The selected weight digest is identical. This tests process interruption; physical power loss was not tested. Earlier engine and trainer acceptance source hashes remain unchanged.

## Use and next step

Resume either larger run with its original width, seed and directory:

```powershell
& .\tools\train.ps1 train --run data/training-pilot-w256-seed2718 --width 256 --seed 2718
& .\tools\train.ps1 train --run data/training-pilot-w256-seed31415 --width 256 --seed 31415

$runtime = Get-Content build/training-runtime.json -Raw | ConvertFrom-Json
& $runtime.python tools/compare-model-sizes.py
```

The comparison reads existing completed runs and reuses its verified recovery evidence on later invocations. Results and per-model detail are in [model-size-results.json](../build/model-size-results.json), each run's `size-assessment.json`, and [the comparison tool](../tools/compare-model-sizes.py). Recovery evidence is `build/large-model-recovery/recovery.json`.

Keep width 64 as the training baseline and carry width 256 into opt-in native search tests. The best width-256 checkpoint by the existing selection criterion is seed 31415 at step 800; seed 2718 selects step 1100. Native export parity, per-stratum tactical checks, equal-time searches and development arenas still govern promotion. Grow the diverse, stronger-labeled corpus with preserved holdout exclusions before another capacity jump. This experiment supplies a larger candidate without establishing that a million-parameter network would be useful on the current pilot.
