# Phase 6: compact policy and self-play pilot

The bounded engineering pilot and **42 distinct acceptance checks passed**: policy warm start, 64 GPU self-play games, one 500-step learning update, 98 inference timing units, 80 position cases and 192 paired strength games. **Both CUDA/PUCT challengers failed the predeclared strength gates. H_v1 remains the default.** The roadmap's Phase 6 strength requirement remains open.

The supported rule contract is the user-confirmed Betawi `id-betawi-relay7-v0` profile: seven pits per side, seven seeds per pit initially, sequential relay play, permanent stores, own-empty capture, bonus store turns, normalized forced passes and final single-round player-0 seed margin. Local rule variations require a separate versioned contract. Ordinary predictions are estimates; this pilot neither retrieves the K=4 oracle nor solves the initial board.

## What was built and trained

The compact model has 127 relay-aware inputs, two width-64 hidden layers, a value output and seven policy logits: 12,872 parameters. Policy warm start trained only its 455 policy parameters; the six value/trunk tensors stayed unchanged from the selected H8 relay model. Training reused the existing CUDA PyTorch environment.

| Production stage | Completed work | Selection and target |
| --- | --- | --- |
| Policy warm start | 1,000 steps, batch 256, 20 checkpoints, 256,000 teacher draws | Scheduled development policy cross-entropy; selected step 1,000; actor-relative H8 action-score soft targets, temperature 4 seeds |
| GPU self-play | 64 terminal games, 2,154 choices, 137,856 completed backups | Eight independent roots; 64 completed edge backups after root expansion; no incomplete-game outcome labels |
| Mixed learning update | 500 steps, batch 128, 10 checkpoints; 32,000 teacher and 32,000 replay draws | Fixed final step 500; all parameters trainable; development metrics descriptive |

PUCT backs up fixed player-0 values and applies the selecting player's direction at each node. Bonus moves and passes do not cause unconditional sign changes. Legal priors, parent-value first-play urgency, tree bounds, completed-backup accounting and cancellation are explicit. Time-mode analysis and benchmarking can search beyond self-play's fixed 64 simulations.

Warm weights: `f6d50747d1ec980986b39fdacadd4b02a335e4244c6cbec5ae73c977f1ab0f17`. Updated weights: `00b47e55f513461993ec799863340b45fbd5b80ac784a81c13dd56ec16aac8b3`.

## Admission, weighting and pruning

The protected registry contains 21,777 active-board quotient keys, 2,370 families and metadata from all 720 historical arena games, including invalidated experiments. Held-out target bodies were not read for training. An active-state bridge into protected ownership excludes the entire paired self-play family.

Of 32 starting families, 20 were excluded and 12 admitted. The admitted 24 games supplied **623 rows covering 607 distinct raw states**. Another 184 low-active rows within admitted families were omitted. All 64 games and their provenance remain in the journal. There were no capped games, and the deterministic four-observation raw-state cap removed zero rows in this pilot. Individual returns were preserved rather than averaged.

Replay rows have unit weights. Sampling chooses among 11 strata uniformly, then samples with replacement within a stratum; it is not game-balanced. The rarest stratum has only two rows, so repeated sampling is a material diversity limitation. Every update batch uses exactly half teacher and half self-play rows. The loss is:

`0.5 Huber_teacher + 0.5 Huber_selfplay + 0.25 (0.5 CE_teacher + 0.5 CE_visits)`

Teacher values are finite-depth estimates; replay values are outcomes of the played games. Neither is automatically a minimax truth. No loss-based deletion or prioritized replay was introduced.

## Quality and equal-time strength

Overall development H8 value MAE changed from **5.062 to 5.730 seeds** after the mixed update. Both models still pass all three eligible ordinary-move strata against H_v1's store-difference baseline. This ordinary value gate does not establish playing strength.

The arena used 24 predeclared development families, paired seats, 25 ms declared search budgets, 512-choice game caps and 4,096 family bootstrap resamples. These families are fresh relative to earlier arena families but reused development data; they are not unseen release validation. All 192 games finished, with no fallbacks or incomplete pairs.

| Challenger / opponent | Paired mean seed margin | 95% family bootstrap interval | Wins / draws / losses | Gate |
| --- | ---: | --- | --- | --- |
| Warm CUDA / H_v1 | -23.583 | [-29.250, -18.167] | 8 / 0 / 40 | Failed |
| Warm CUDA / H8 relay CPU | -23.333 | [-28.542, -18.458] | 12 / 0 / 36 | Failed |
| Updated CUDA / H_v1 | -21.708 | [-26.417, -17.125] | 10 / 1 / 37 | Failed |
| Updated CUDA / H8 relay CPU | -20.167 | [-24.583, -15.958] | 15 / 0 / 33 | Failed |

A positive lower interval bound against both baselines was required. Neither challenger qualifies. The update's separate comparisons are less negative than the warm model's comparisons, but this is not a direct warm-versus-updated paired experiment or proof of a causal learning improvement.

The 80 cases contain 48 current H8 development positions and 32 known exact diagnostics, tested at 25 and 100 ms across five engines. On the 48 H8 roots at 25 ms, reference-action accuracy was 91.7% for H_v1, 85.4% for H8 relay CPU, 58.3% for warm native PUCT, 52.1% for warm CUDA PUCT and 47.9% for updated CUDA PUCT. H8 action scores are approximate references. These case results are separate from terminal arena outcomes.

At 100 ms, warm native PUCT averaged 869 completed backups on those roots, versus 100 for warm CUDA. Its maximum observed whole-wrapper wall time was 119.478 ms, so deadline overshoot remains an optimization concern. All wrapper latencies and overshoots are retained rather than hidden by the nominal budgets.

## Inference and resource costs

Seven timing blocks measured 128 calls at each batch size, with startup and four-call shape warmups separately declared. Median complete-call milliseconds:

| Batch | Native CPU | CUDA |
| ---: | ---: | ---: |
| 1 | 0.083 | 0.914 |
| 2 | 0.164 | 0.943 |
| 4 | 0.315 | 1.207 |
| 8 | 0.618 | 1.370 |
| 16 | 1.221 | 1.939 |
| 32 | 2.459 | 3.116 |

The actual native adapter is faster at every measured size. At the pilot queue of eight, native costs 77 microseconds per root versus 171 on CUDA. Native telemetry performs two feature passes; CUDA performs one. These are adapter measurements including features, queue handling, transfers, forward evaluation, synchronization and validation, not an isolated neural-kernel comparison. Maximum observed parity errors were 0.00000685 seeds and 0.000000247 logits, within declared tolerances.

The complete inference invocation took 15.703 seconds, peaked at 2.044 GiB private commit and 681 MiB working set, and used a peak CUDA allocation of 33.1 MiB. The complete case/arena invocation took 195.219 seconds, peaked at 2.037 GiB private commit and 684 MiB working set, and used a peak CUDA allocation of 33.1 MiB. CUDA allocator figures exclude driver/context and other processes. Both invocations began with zero committed units.

Summed self-play forest search clocks total 22.263 seconds, 114,795 neural evaluations and 6,585,114 feature seed deposits; that clock excludes startup, journal/commit costs and final replay publication. Reported training invocation clocks after admission are 5.922 seconds for warm training and 3.656 seconds for the update, excluding data/import preflight but including model setup, evaluation, checkpoints and exports. Energy was not measured.

Earlier Windows commit-memory failures remain recorded. After other heavy work was paused, the existing runtime completed. V2 validates CPU data before CUDA admission; it preserves the training mathematics. A compact feature cache was considered but was not implemented in this version.

## Recovery and audit scope

FULL/WAL journals retain committed decisions, cases, games and checkpoint checksums. Training restores model, optimizer, sampler, draw counters and Python/Torch/CUDA RNG states. Unfinished trees, cases or arena games are recomputed. Same-identity completed reruns validate and preserve published bytes and timestamps. Changed source, model or protocol identities require a new experiment.

The updated-model analyzer retained its 64-simulation recommendation through a real prepublication process kill, then resumed to 128 simulations matching a fresh semantic result. The previous recommendation stays available while larger work is unfinished. GPU bitwise continuation under a different batch schedule is not promised.

The independent auditor passed 14 checks and verified all 64 pilot games, 2,154 choices, 623 retained targets, 30 checkpoints, both models on 1,000 literal development forwards each, 80 cases, 192 arena games and the paired bootstrap gates. All 878 prior Stage 5 acceptance inputs remain unchanged. The aggregate counts 6 author warm-training checks, 3 author update checks, 8 PUCT conformance checks, 11 guarded-benchmark checks and 14 independent checks, totaling 42; repeated updated-model parity is supporting evidence rather than an additional count.

Precise recovery coverage is recorded in [the audit report](PHASE6_INDEPENDENT_AUDIT.md). The successful author warm-training kill tests reported exact recovery, but their temporary raw fixtures were deleted. Independent warm review therefore covers source, published test records and a separate warm witness, not a fresh live warm kill or raw killed-state recomparison. Retained independent live self-play/update/case/arena fixtures provide separate recovery evidence.

## What to implement next

Keep H_v1 as the default and retain both experimental models. First run a newly frozen native-PUCT experiment with the actual warm and updated exports against both CPU baselines. Pair that with cheaper relay-feature extraction, reusable buffers/ABI or bounded evaluation caching measured end to end; CPU-first inference is the current candidate for interactive analysis. The present native case measurements do not substitute for a native arena test.

Before a larger model or database campaign, address replay diversity and ordinary-move quality with predeclared ablations. Then integrate explicit typed K=4 lookup, backend selection, ranked explanations and recoverable analysis into one interface. Fresh release validation, reproducible packaging and rollback remain necessary. See [the engine completion checklist](PHASE6_SCOPE_AND_ENGINE_READINESS.md). Full-game proof and quality-per-joule claims remain separate, unfinished objectives.

Evidence: `build/phase6-acceptance.json`, `build/phase6-audit/audit.json`, `build/phase6-plan.json`, `build/phase6-implementation-v2.json`, `build/phase6-benchmark.json`, `build/phase6-inference-speed-results.json`, `build/phase6-speed-resources.json`, `build/phase6-benchmark-resources.json`, `build/phase6-analysis-recovery.json`, and the production journals under `data/phase6-*`.
