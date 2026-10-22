# Native CPU PUCT: cost and strength results

The native CPU PUCT campaign is complete. Single-pass inference passes both predeclared cost gates and approximately halves fixed-work tree time. All six profiles fail their strength gates against both CPU baselines, so **H_v1 remains the default**. The experiment uses the frozen Betawi rule profile `id-betawi-relay7-v0`, the existing compact warm and updated exports, native CPU inference and unchanged Python PUCT search. It introduces no training or new training targets. Independent engineering acceptance is recorded separately from playing strength.

## Evaluation costs

All 1,410 registered cost units completed: six startup observations, 36 parity units, 504 microbenchmark units and 864 fixed-tree units. Every fixed 64-backup tree agrees exactly with its model's legacy control in chosen move, priors, visits and action values. The new structured native boundary reuses the existing double-precision model and scalar/AVX2 dispatch. It performs features and prediction once, replacing the original adapter's diagnostic feature pass plus prediction pass and its JSON prediction boundary.

The table reports medians across the declared blocks. Microbenchmark entries are complete call times with already loaded/warmed models and an empty cache; cache reset is included. Tree entries summarize all 48 roots and three blocks. These are implemented-adapter measurements, not isolated dense-kernel timings.

| Model / adapter | Batch 1, microseconds | Batch 8, microseconds per call | Fixed64 tree, milliseconds | Real tree cache hits / requests |
| --- | ---: | ---: | ---: | ---: |
| Warm legacy | 87.730 | 645.033 | 8.835 | 0 / 9,312 |
| Warm single pass | 21.861 | 124.570 | 4.123 | 0 / 9,312 |
| Warm single pass + LRU | 20.872 | 126.445 | 4.223 | 204 / 9,312 |
| Updated legacy | 84.264 | 642.750 | 8.723 | 0 / 9,312 |
| Updated single pass | 19.716 | 126.533 | 4.146 | 0 / 9,312 |
| Updated single pass + LRU | 21.217 | 128.177 | 4.177 | 222 / 9,312 |

The predeclared cost gate uses matched roots rather than the ratio of the table's pooled medians. Taking each root's median of three single-pass/legacy ratios, then the median across 48 roots, gives **0.480658 warm** and **0.476776 updated**: approximately 52% lower whole-tree time. Feature simulations and deposited seeds are exactly half the legacy work at every matched single-pass tree. Both models pass that point-estimate cost gate; no confidence interval or energy claim is attached to it.

The LRU hit rates are only 2.19% warm and 2.38% updated in these fixed64 trees. The corresponding matched cache/single-pass time ratios are 1.020363 and 1.005758, so the cache does not demonstrate a tree-speed benefit in this workload. Longer 100 ms ordinary-position searches have about 12.5% hits and slightly more completed backups with caching; cache utility is workload dependent. Prefilled hot-cache microbenchmarks are retained as a retrieval ceiling; their artificial hit rate is not used as practical search throughput. The uncached structured adapter is a useful control for the next efficiency experiment; these observations do not promote a playing engine.

The cache remains bounded at 4,096 entries, private to one evaluator/model, and keyed by all sixteen input board cells plus the input controller. It clears inside every decision clock. Physical forwards, cache events and relay feature work are recorded separately from frozen PUCT's logical returned-row count. Cancellation can retain partial physical work without returning a prediction.

## Equal-time strength

The registered campaign contains 80 position cases, 1,280 scalar measurements at 25/100 ms, and twelve comparisons totaling 576 games. Six CPU profiles face both H_v1 and H8 relay CPU at 25 ms per decision, with 24 paired-seat starting families per comparison and a 512-choice cap. The comparison order rotates by family and alternates seat order. All outcomes are retained.

These are reused development positions and families, not unseen release validation. The twelve 95% family bootstrap intervals are exploratory and have no multiple-comparison adjustment. A positive lower seed-margin interval against both baselines, all 24 terminal pairs and zero fallbacks are required for a profile's primary strength gate. WDL is secondary. Incomplete games receive no fabricated payoff, and unusable recommendations count as explicit legal-first-action fallbacks.

All 576 games are terminal, every comparison has 24 complete pairs, and there are zero fallbacks. Every primary interval lies below zero. No profile passes its individual comparison or its both-baseline gate.

| Challenger / opponent | Paired mean seed margin | 95% family interval | Wins / draws / losses | Gate |
| --- | ---: | --- | --- | --- |
| Warm legacy / H_v1 | -14.167 | [-18.333, -10.292] | 15 / 0 / 33 | Failed |
| Warm legacy / H8 relay CPU | -15.917 | [-19.583, -12.167] | 18 / 2 / 28 | Failed |
| Warm single pass / H_v1 | -13.333 | [-17.250, -9.583] | 16 / 1 / 31 | Failed |
| Warm single pass / H8 relay CPU | -16.250 | [-20.042, -12.625] | 17 / 2 / 29 | Failed |
| Warm single pass + LRU / H_v1 | -14.083 | [-18.333, -10.083] | 15 / 0 / 33 | Failed |
| Warm single pass + LRU / H8 relay CPU | -16.958 | [-20.792, -13.250] | 16 / 2 / 30 | Failed |
| Updated legacy / H_v1 | -16.167 | [-20.625, -11.958] | 15 / 0 / 33 | Failed |
| Updated legacy / H8 relay CPU | -16.792 | [-21.167, -12.542] | 16 / 0 / 32 | Failed |
| Updated single pass / H_v1 | -13.000 | [-17.167, -9.083] | 17 / 1 / 30 | Failed |
| Updated single pass / H8 relay CPU | -16.625 | [-20.500, -12.667] | 18 / 0 / 30 | Failed |
| Updated single pass + LRU / H_v1 | -12.833 | [-16.917, -8.958] | 17 / 1 / 30 | Failed |
| Updated single pass + LRU / H8 relay CPU | -15.667 | [-19.583, -11.833] | 18 / 0 / 30 | Failed |

On the 48 current H8 development roots, search throughput roughly doubles with the structured adapter, while move quality still trails both baselines. H8 root-action scores are approximate references, not minimax truth. The 32 known exact diagnostics are reported as a separate cohort and do not inflate this ordinary-position table.

| Engine | Reference-action accuracy, 25 / 100 ms | Mean completed backups, 25 / 100 ms |
| --- | ---: | ---: |
| H_v1 | 93.75% / 87.50% | Not PUCT |
| H8 relay CPU | 85.42% / 79.17% | Not PUCT |
| Warm legacy | 60.42% / 66.67% | 192.8 / 841.9 |
| Warm single pass | 60.42% / 68.75% | 395.6 / 1,623.3 |
| Warm single pass + LRU | 62.50% / 68.75% | 402.2 / 1,652.1 |
| Updated legacy | 60.42% / 62.50% | 193.2 / 838.8 |
| Updated single pass | 62.50% / 72.92% | 392.0 / 1,634.9 |
| Updated single pass + LRU | 64.58% / 72.92% | 403.4 / 1,655.9 |

The result establishes a substantial implementation cost reduction, while the current policy/value/search combination remains weaker than the baselines. It does not isolate which learning or search component causes the remaining gap. Separate candidate-versus-baseline rows and comparisons with earlier CUDA campaigns do not establish a causal training improvement. A larger model or longer training run is not justified by the speed result alone.

Whole-wrapper latency and overshoot remain in the evidence. On these 48 roots at 100 ms, maximum wall times are 100.760/102.393/103.099 ms for warm legacy/single/cache and 100.788/100.565/101.016 ms for updated legacy/single/cache. H_v1 reaches 126.507 ms in one observed case; H8 relay CPU reaches 101.415 ms. The nominal search budget is not a hard real-time guarantee. The cache reset and remaining-integer-millisecond policy do not grant additional search time.

## Recovery, audit and resources

The analyzer demo uses the updated model, cache capacity 4,096 and 128 completed backups at the initial board. Its same-identity completed rerun preserves the recommendation and SQLite bytes and modification times. Its values estimate the fixed-player-0 final margin, and its chosen move is a search recommendation. The result is explicitly a Monte Carlo estimate; its visit allocation is not a calibrated probability or proof of the best move.

The independent auditor passed all 13 unique checks. Retained fixtures demonstrate real precommit and postcommit analyzer kills, preservation of the previous 64-backup recommendation, recovery to 128 matching a fresh semantic result, missing-export repair, case-prefix recovery, and a first-unit timing kill followed by completion of all 1,410 units. The arena recovery fixture copies the complete campaign, kills after 575 committed games, resumes to 576, and literally replays the resulting games. Completed reruns preserve bytes and modification times. Rehashed corrupt counters, reference regret, impossible feature work and terminal margins are refused. Unfinished tree work is recomputed; live mutable trees and caches are not checkpointed.

The full audit independently verifies all 80 cases and 1,280 measurements, all 576 production games and 18,277 choices, all twelve paired intervals and gates, every cost unit, exact agreement of all 864 fixed trees, and physical feature/cache work from 96 independent legacy tree traces. Both exports also match 1,000 literal model forwards each. All 1,192 prior Phase 6 input hashes, including all 878 Stage 5 input hashes, remain unchanged. The aggregate acceptance comprises **34 distinct executed checks: nine inference-owner checks, twelve benchmark-owner checks and thirteen independent checks**. The full campaign replay/model/work verification is additional supporting evidence, not extra counted test methods.

No new training or warm-training recovery test is performed here. The preserved prior Phase 6 audit's narrow warm-training review limitation remains as documented in [its results](PHASE6_RESULTS.md); this CPU audit does not expand that earlier claim.

The original CPU-only cost invocation took 65.266 seconds, with peak private commit 227.1 MiB and peak working set 236.1 MiB. The case/arena invocation took 495.489 seconds, with peak private commit 227.1 MiB and peak working set 236.5 MiB. Both began with zero committed production units. These outer clocks start after argument parsing and include source guards, producer module load, admission, model startup, remaining work, journals and publication; Python startup is excluded. The benchmark's additional inner clock is 495.009 seconds. Neither producer imports Torch or uses CUDA. These are scoped process observations, not whole-machine memory or electricity measurements.

Resource capture has a narrow limitation: if a process is killed after publishing its final producer report but before publishing the original outer-resource record, the wrapper refuses to invent the missing original invocation costs. The producer's committed semantic journal can still be validated and resumed directly. Both actual production invocations published their original resource records successfully. The progress runner records the child's real exit code and state, but its frozen top-level CLI does not propagate a nonzero child exit to the shell; acceptance checks actual completion artifacts independently.

## Work still required for the finished analyzer

The simulator, alpha-beta/PVS search with TT, K=4 exact database, supervised training, recoverable self-play pilot, CPU/GPU inference and analysis journals already exist. This CPU experiment addresses evaluation cost and measures native PUCT strength; it does not finish all integration or learning work.

1. Improve ordinary-move quality and search allocation through predeclared, bounded ablations before a larger model or database campaign. Test useful leaf/feature changes against both current CPU baselines; retain fixed-work semantic controls and equal-time arenas.
2. Diversify replay starts and opponents, reduce dependence on the existing 12 admitted families and two-example rare stratum, and compare sampling/rehearsal choices explicitly. Preserve outcome provenance, protected ownership, rare mechanisms and reproducible pruning. More rows alone are not evidence of stronger play.
3. Validate a frozen release candidate on fresh families and budgets, then characterize that release once on its protected sealed test. Current development results must not become claims of unseen generalization.
4. Integrate the verified K=4 lookup into an explicit versioned analysis mode. Clearly distinguish exact facts, finite-depth teacher estimates and Monte Carlo estimates; declare coverage and recurrence before interior exact lookups affect ordinary search.
5. Consolidate backend/model selection, ranked actions, continuation traces, explanations and cancellation/resume into one supported analyzer interface. Document the precise Betawi contract and single-round final seed-margin objective.
6. Package reproducible setup, supported resource limits, accepted model selection and rollback. Correct the progress runner's shell exit propagation in a new source version before treating its shell status as a pipeline success signal; current run records retain the actual child exit and acceptance checks actual artifacts independently.

Electricity and quality per joule remain unmeasured. The initial board is not solved. Larger tablebases, NNUE, larger networks and proof frontiers require separate evidence and are not prerequisites merely because chess engines use them. Later Phase 7/8 proof goals remain distinct from finishing a strong position analyzer.

Evidence: `build/puct-cpu-plan.json`, `build/puct-cpu-implementation.json`, `build/puct-cpu-cost-gate.json`, `build/puct-cpu-speed.json`, `build/puct-cpu-benchmark.json`, the corresponding `data/puct-cpu-*-v1/work.sqlite3` journals, `build/puct-cpu-audit/audit.json`, and `build/puct-cpu-acceptance.json`. See [run and resume commands](PUCT_CPU_RUN_GUIDE.md), [independent audit](PUCT_CPU_INDEPENDENT_AUDIT.md), and the preserved [engine completion checklist](PHASE6_SCOPE_AND_ENGINE_READINESS.md).
