# Fixed-policy value training results

The fixed-policy value-head refit improves row-weighted teacher MAE from 5.339 to 5.132 seeds, but its matched arena means are lower than the unchanged improved-model control. Neither profile passes the required both-baseline strength gate. **H_v1 remains the default.** All 192 games are terminal with zero fallbacks. All 10 independent methods and the exhaustive tensor/game audit pass. Together with the author suites, 27 distinct executed checks pass.

## Intervention and limits

The control is the improved compact quality model. Both profiles use the same c_puct=0.25 search, native CPU evaluator, relay encoder, model shape and fixed-player-0 estimates. The treatment freezes the six shared-trunk/policy tensors (12,807 parameters) and trains only the final value head (65 parameters). There is no policy loss or trunk update. Exact frozen tensor bytes preserve policy logits for all positions; sampled native/literal checks support this structural fact.

The sole refit uses H8 root final-margin labels and seed-space Huber(delta 4), AdamW/fp32 with no weight decay, batch 256, learning rate 0.001, seed 20261018, checkpoint every 50 steps and fixed final 1,500 steps. No development-based checkpoint selection, larger model, new teacher corpus, self-play outcome or exact-oracle target enters. The fixed representation can limit achievable value quality, and approximate teacher fit does not guarantee playing strength.

The registry adds all 384 preceding quality arena trajectories. Of the original 8,000 training roots, 7,592 are admitted and 408 excluded by protected-root/whole-selected-family closure. Original data and reasons remain. The warm ancestor trained on 8,000 roots and the improved joint ancestor on 7,669; filtering new draws does not undo those exposures. The 1,000 development roots remain descriptive and ancestrally exposed to checkpoint selection. Rare strata can have few distinct rows despite repeated draws.

## Development quality

Exactly 1,500 CUDA optimizer steps and 384,000 draws completed, retaining all 30 scheduled checkpoints. Fixed-final weights digest: `071cbc73282075efcf6efafc867f31aa530f5d4e6ec79b75fbe572c02913780a`. No checkpoint was selected using development metrics.

| Same 1,000 development roots | Policy control | Value refit |
| --- | ---: | ---: |
| Row-weighted value MAE, seeds | 5.339074 | 5.131947 |
| Uniform-stratum value MAE, seeds | 5.317460 | 5.264500 |
| Uniform-stratum residual MAE | 0.149911 | 0.151933 |
| Uniform-stratum seed Huber delta 4 | 14.849068 | 14.602294 |
| Uniform-stratum policy teacher regret, seeds | 2.922517 | 2.922517 |
| Uniform-stratum policy optimal-set accuracy | 0.647237 | 0.647237 |
| Uniform-stratum policy cross entropy | 1.064644 | 1.064644 |

The row-weighted seed MAE improves by 3.88%, while uniform-stratum seed MAE improves by 1.00%. Normalized residual MAE worsens. All policy metrics are exactly unchanged. These are approximate H8 teacher targets on reused development data, not exact values or strength proof.

| Active seeds / mechanism | Development roots | Control MAE | Refit MAE |
| --- | ---: | ---: | ---: |
| 25-49/bonus | 94 | 6.5985 | 6.3027 |
| 25-49/capture | 94 | 5.6587 | 5.6118 |
| 25-49/ordinary | 93 | 4.6616 | 4.5271 |
| 25-49/pass | 1 | 3.7431 | 5.0904 |
| 5-24/bonus | 93 | 1.9288 | 1.8894 |
| 5-24/capture | 93 | 2.4791 | 2.4257 |
| 5-24/ordinary | 93 | 2.0235 | 2.0817 |
| 5-24/pass | 93 | 1.2663 | 1.4050 |
| 50-74/bonus | 93 | 7.1264 | 6.6379 |
| 50-74/capture | 93 | 8.2399 | 7.9646 |
| 50-74/ordinary | 73 | 11.0713 | 10.8348 |
| 75-98/bonus | 4 | 4.9734 | 5.3552 |
| 75-98/capture | 83 | 9.3563 | 8.3121 |

Gains are uneven. Low-active ordinary/pass and the rare 25-49/pass and 75-98/bonus strata regress; the latter two contain only 1 and 4 development roots. High-active capture MAE improves, but several ordinary positions still have large teacher error. The refit does not establish that the fixed representation is sufficient or that greater model capacity is required.

## Equal-time strength

The registered campaign has 80 cases: 48 H8 development roots and 32 separate known-exact diagnostics. Four engines at 25/100 ms produce 640 measurements. Two fixed-policy profiles each face H_v1 and H8 relay CPU over 24 paired-seat families at 25 ms, yielding 192 games. Baselines use depth 16/TT 16 MiB/PVS; both PUCT profiles use the same declared core, 100,000-backup/2,048-node/no-noise bounds.

All prior arena IDs and the preceding quality 48 H8 case family IDs are excluded; current cases and arenas are disjoint. Excluding every historical diagnostic ID exhausted the ordinary pool, so that rule was narrowed before training or measured outcomes. Older diagnostic families may recur. These are development comparisons, not unseen release validation.

The primary gate needs all 24 terminal pairs, zero fallbacks and a positive lower 95% paired-family seed-margin bound against BOTH baselines. All four exploratory unadjusted intervals will be retained. Matched refit-minus-control mean margin is secondary and descriptive, with no extra interval or promotion gate. Whole-decision timing includes reset and validation, floors remaining milliseconds and retains overshoot; exhausted time never becomes an unlimited call.

All 192 games are terminal; every comparison has 24 paired families and zero fallbacks. All four baseline comparisons fail their positive-lower-bound requirement.

| Profile / opponent | Paired mean margin, seeds | 95% family interval | Wins / draws / losses |
| --- | ---: | --- | --- |
| policy_control / H_v1 | +1.041667 | [-1.125000, +3.250000] | 23 / 0 / 25 |
| policy_control / H8_relay_CPU | -1.583333 | [-4.291667, +0.666667] | 24 / 0 / 24 |
| value_refit / H_v1 | +0.375000 | [-1.666667, +2.416667] | 21 / 2 / 25 |
| value_refit / H8_relay_CPU | -2.375000 | [-4.958333, -0.291667] | 22 / 2 / 24 |

The secondary matched refit-minus-control means are **-0.666667 seeds against H_v1** and **-0.791667 against H8**, each across 24 complete matched families. There are no confidence intervals or extra gates for these contrasts. They do not establish a statistically reliable regression, and they provide no measured mean strength gain from this refit. Candidate values versus earlier campaigns use different families and cannot replace this matched comparison.

| Engine | H8-reference accuracy,25 /100 ms | Mean reference regret,seeds,25 /100 ms | Mean backups,25 /100 ms |
| --- | ---: | ---: | ---: |
| H_v1 | 89.58% / 83.33% | 0.312500 / 0.666667 | Not PUCT |
| H8_relay_CPU | 79.17% / 81.25% | 0.729167 / 0.500000 | Not PUCT |
| policy_control | 75.00% / 79.17% | 0.958333 / 0.875000 | 467.7 / 2035.0 |
| value_refit | 77.08% / 83.33% | 0.916667 / 0.708333 | 471.3 / 2027.9 |

The refit improves H8-reference case accuracy/regret while arena mean margins do not improve. This reinforces that approximate teacher fit and isolated case metrics are not a substitute for played-game strength. All four engines choose an exact-reference-optimal action in all 32 known-exact diagnostic cases at both budgets; the PUCT outputs remain estimates, not newly certified proofs. Completed backups include terminal-leaf revisits and are not counts of distinct neural evaluations.

Observed wall times and cancellation work remain retained. On the48 H8 cases, the largest25 ms wall was 28.764 ms (H_v1); at 100 ms it was 101.534 ms (value_refit). On known-exact diagnostics, value_refit reached104.052 ms at 100 ms. Nominal budgets are not hard real-time guarantees. Partial cancelled forwards, features and successful predictions are accounted separately; all case recommendations were usable.

## Recovery, verification and resources

All **27 distinct executed checks pass**: four learning, thirteen benchmark and ten independent methods. Inherited search checks remain protected evidence and are not counted again. All 1,647 prior accepted input hashes remain unchanged. The exhaustive audit reconstructs the 384 preceding ownership games / 11,920 choices, replays every one of the 6,461 choices in the current 192 games, and independently rederives intervals and matched contrasts. Both actual exports match 1,000 literal development forwards each, with zero observed error.

The independent PT worker deserializes all 30 checkpoints, compares the six frozen tensors using raw FP32 byte views, verifies exactly two optimizer parameter IDs and valid moment tensors/steps, reconstructs all 384,000 sampler draws and Python/Torch/CUDA RNG, and verifies final export mapping. Only the two value-head tensors changed. Production checks inspect saved state rather than replaying all 1,500 gradient updates.

Actual CUDA workers were killed before step 4 catalog commit (retaining step 2) and after step 4 commit (retaining steps 2/4), then resumed to step 6. Model tensors, every optimizer state and all RNG states match a fresh six-step run exactly under the typed tensor/value comparator. Raw frozen-weight byte checks additionally establish policy preservation. Completed reruns preserve file bytes and modification times; uncommitted orphan files remain diagnostic evidence.

Analyzer pre/postcommit kills preserve the completed 64-backup recommendation and recover to the same 128-backup semantics and physical work as a fresh run. Missing exports repair from committed state. The case campaign survives a kill after case 1 and resumes through case 2. A copied arena campaign is killed after game 191 commits, resumes to 192 and preserves the committed prefix. Its recomputed timed final game may differ from production; that recovery copy has 6,463 literal choices and is audited separately. The production journal and its 6,461 choices remain unchanged. Rehashed corrupt weights, optimizer ownership, work, regret and terminal payoff are refused. The completed production analyzer demo also retains identical SQLite/export bytes and nanosecond mtimes on rerun.

The auditor tightened floating-value equality to raw bytes only in its final model/PT inspection, and added provenance-aware collection. The three already completed plan/target/training groups (four methods) never call those modified functions. Their original reports and byte-identical executed coordinator remain preserved; derived reports explicitly bind that retained version and preserve all outcomes, durations and evidence. The other six methods and exhaustive audit use the corrected coordinator. An independent AST/provenance review limits changes to those three functions. No extra run or method is claimed, and no producer or outcome changed. Raw killed journals, orphan files, backups, source snapshots, refusal logs and earlier review diagnostics remain retained.

The production training clock was 54.500 s after common admission, including data preparation, ancestor proof checks, lazy Torch/CUDA loading, setup, optimization, checkpoint metrics/publications and final export. It excludes module/Python startup, common admission and the subsequent final result JSON publication. Peak host working set was 1409.0 MiB; CUDA allocated/reserved peaks were 70.4/88.0 MiB, excluding driver/context and other processes. Verification costs are separate. The original CPU benchmark invocation took 187.474 s, starting with 0 committed cases/games. It includes admission, model setup, validation, remaining units, journals and report publication after module imports; Python/module startup and earlier invocations are excluded. Peak private commit/working set was 51.3/62.8 MiB. It imported no Torch and performed no CUDA inference. Resources publish before completion metadata and remain immutable on completed semantic reruns. No whole-machine electricity claim follows from time or allocated memory alone.

## Remaining engine work

H_v1 stays default during this development experiment. The previous improved policy/value control remains the experimental comparison point. This refit is retained for reproducibility and analysis, without promotion. Further value work should examine search-relevant calibration and ordinary/high-active errors, using this fixed-policy control and a new registered protocol. Lower aggregate teacher MAE alone is insufficient; the fixed-trunk result does not yet justify a larger model or expanded corpus. A release still needs both-baseline strength evidence and fresh validation, unified typed K=4 retrieval with clear estimate/proof labels, model/backend selection, reproducible setup and rollback. The initial board remains unsolved. Larger models and target expansion require their own evidence.

See [frozen scope](VALUE_SCOPE.md), [run/resume guide](VALUE_RUN_GUIDE.md) and [independent audit](VALUE_INDEPENDENT_AUDIT.md). Canonical evidence paths are build/value-plan.json, build/value-implementation.json, build/value-data.json, build/value-model-witness.json, data/value-train-v1, build/value-benchmark.json, data/value-benchmark-v1/work.sqlite3, build/value-audit/audit.json and build/value-acceptance.json.
