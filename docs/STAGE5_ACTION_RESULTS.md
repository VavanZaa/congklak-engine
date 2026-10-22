# Stage 5: root policy and ordinary-error objectives

The action-aware experiment adds a seven-move policy head, three matched training objectives and four native search modes. **H_v1 remains the default.** Training finished and all six models fail the ordinary-value promotion gate. Learning root move preferences helped direct policy ranking; its benefit inside search must be measured separately.

Three agents handled training, native search and an independent audit. The production comparison uses 391 roots and 144 games, with committed counters on the [progress dashboard](PROGRESS_REPORTING.md). The complete ledger contains all 391 cases and 144 terminal games, with no arena fallback moves.

## Targets and objectives

`training_v4.py` uses the corrected, quotient-isolated `stronger-10000-v2` corpus: 8,000 training rows, 1,000 development rows and unopened sealed targets. Every value label remains the root H_v1 depth-six result. An action score represents the selected root move followed by a depth-five successor search. It is a preference target at that root, never a depth-six value label attached to the successor. No extra successor training rows are generated.

The shared 43-input, width 64, two-hidden-layer model has 7,496 parameters: 7,041 for trunk/value and 455 for the policy head. Its value is exact store difference plus bounded actor-relative remaining-seed residual. The policy has seven actor-relative logits; legal-action masking excludes illegal pits. Targets are legal softmax preferences from root-actor-signed action scores, with temperature four seeds.

| Objective | Training loss | Paired development macro MAE | Paired direct policy accuracy | Paired direct policy regret |
| --- | --- | ---: | ---: | ---: |
| A | Value Huber, delta 1 | 7.8019 | 29.9% | 11.5175 seeds |
| B | A + 0.25 policy cross entropy | 7.7598 | 39.1% | 9.8365 seeds |
| C | B + 0.01 ordinary-group error penalty | 8.2511 | 41.9% | 9.4550 seeds |

Each objective ran seeds 2718/31415 for 8,000 FP32 AdamW steps, batch 512, with identical starting weights and sampling by seed. All 480 scheduled checkpoints and 4,096,000 draws per run were verified. Equal steps/draws do not imply equal optimizer FLOPs. A's policy head has no policy learning objective; its untrained readout receives weight decay and sees a value-trained trunk. Its policy metrics are a control, not a learned policy baseline.

C averages the squared minibatch mean actor-relative value error over present ordinary strata. Eligibility requires at least 20 TRAIN rows; absent groups are omitted and an entirely absent set produces a graph-connected zero. It includes minibatch variance as well as bias. It is an objective ablation, not a population-bias estimator or a development calibration step.

Every 100 steps, finite-development uniform-stratum value MAE selects checkpoints. Paired B/C macro MAE selects the joint objective before any arena; best seed within the selected objective chooses the representative. **B seed31415** is the challenger; **A seed31415** is the matched value-only control. Policy accuracy and arena outcomes do not select checkpoints or objectives.

| Ordinary active seeds | Selected A MAE | Selected B MAE | H_v1 MAE | Development rows |
| --- | ---: | ---: | ---: | ---: |
| 5-24 | 2.193 | 2.192 | 2.333 | 93 |
| 25-49 | 10.532 | 10.562 | 5.226 | 93 |
| 50-74 | 27.225 | 27.021 | 11.822 | 73 |

All six candidates fail the no-regression ordinary-value gate. C reduces paired high-ordinary error to 21.101 seeds versus A 26.598, but worsens overall macro error and is not selected. These errors compare against finite H6 teacher results, not optimal full-game values.

## Native search and controlled timing

The versioned `neural_action` analyzer supports `h-none`, `h-policy`, `value-none` and `value-policy`. H modes retain H_v1 leaves scaled by256; learned modes use the value head. Policy orders interior nodes with at least two remaining choices, after the TT hint, then descending actor-relative logit and ascending pit ties. Root candidate and PV reconstruction retain full-window lexicographic behavior. H modes label finite-horizon results; learned modes label estimates.

TT entries check full state, exact remaining depth and model/mode identity. A bounded model-local inference cache keys pits and normalized actor; store offset remains exact outside inference. Models, modes, policy depth threshold, source and DLL fingerprints guard analysis recovery. Older native DLLs, datasets and trainers remain unchanged.

The pinned speed experiment verified both selected exports over all 1,000 current development roots. Scalar/AVX raw values and all seven logits are bitwise equal. Twelve fresh-cache depth-six roots with three repetitions preserved all root scores, best moves and lexicographic PVs between ordering modes and the frozen H_v1 baseline.

Selected B joint forward latency is about 0.683 microseconds with AVX versus 2.138 microseconds scalar, a 3.12x median ratio over seven 8,192-call blocks. These native forwards exclude Python/JSON overhead. B's policy ordering increased H search nodes by 7.6% and had a 0.947 plain/ordered wall ratio on the small fixed-depth sample; learned-value ordering increased nodes 4.2% with a 0.999 wall ratio. These observations establish no ordering speed advantage.

Corrected V2 timing pins only the benchmark process to the exact predeclared processor (CPU-set id 258, group 0, logical 2, core 2, EfficiencyClass 1), verifies actual execution there and restores the previous mask. It never substitutes a different core. Dynamic parking and the original restoration mask are truthful per-invocation observations outside the saved scientific identity. It changes no power or priority settings. This reduces migration noise without reserving the core or fixing frequency. Microsoft documents the [CPU-set topology fields](https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-system_cpu_set_information) and [process-affinity API](https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask). Energy is not measured.

## Parking defect and versioned correction

Extra game-resume checks exposed a real V1 defect: selecting the second currently unparked performance core changed the saved timing identity as Windows parking evolved. The first complete 391-case/144-game result remains in `build/stage5-action-benchmark.json`; its original progress source is preserved with its exact hash in `build/stage5-action-v1-snapshot/progress.py`.

New `timing_v2.py` and `benchmark-stage5-action-v2.py` declare the same original processor before execution. If initially parked, Windows affinity admits it and actual execution is verified; unavailable, reserved or incompatible processors are refused. Structural topology identifies the experiment; parking and inherited restoration masks do not. Four timing checks include initially parked admission, unavailable-core refusal, real normal/exception restoration and equal identity under all-processor versus single-processor inherited masks.

V2 repeats all 391 cases and 144 games from scratch with the unchanged models, roots, budgets and gates. Only `build/stage5-action-benchmark-v2.json` is authoritative below. Historical V1 measured a +2.583-seed joint/H margin; the corrected run measures +2.500. They reuse the same families and are not independent samples to pool. Failure diagnostics are retained in `build/stage5-action-audit/arena-identity-drift.json` and `first-nine-check-failure.log`.

One additional CUDA recovery-test subprocess had an allocation failure. Its cause is unconfirmed; isolated sequential retries passed without trainer or objective changes. The failed log is preserved with the successful evidence.

## Equal-time protocol and recovery

Six engines run fresh-cache 25/100 ms, depth 16, TT 16MiB/PVS searches: frozen H_v1; new H without/with B ordering; A learned value; B learned value without/with ordering. Original 100 H6 proxy roots, 32 known exact diagnostics and 259 current ordinary H6 roots remain separate. Sixteen old exact roots with sealed quotient aliases are excluded using metadata; no sealed target bodies are read.

Three arenas each use 24 distinct current-development families and swapped seats, 20ms per choice and 512-choice cap: B value+policy versus frozen H; H policy versus H plain; B value+policy versus B value plain. The predeclared primary measure is paired mean final seed margin, with 4,096 family bootstrap resamples. A complete zero-fallback comparison needs a positive 95% lower bound to pass. WDL is secondary. These are repeatedly used development families, not a fresh final test or a universal strength claim.

The native deadline covers search; measured wall latency also includes setup/report transfer. Runtime cancellation or resource failures cannot be admitted as ordinary timed-strength cases. FULL/WAL stores complete cases and complete games; interrupted unfinished units are recomputed. An independent hard kill after the first committed case resumed to two cases with the first case unchanged and verified CPU identity; repeated resume is idempotent and checksum-valid fabricated regret is rejected. Separate real native-depth and CUDA precommit kills restored analysis and both training heads, optimizer, sampler and all RNG state.

## Final outcomes

All predeclared cases and games committed. The learned challenger passes the development seed-margin comparison against H_v1, but fails the separate ordinary-value gate, so it is not promoted. Ordering-only comparisons fail their primary margin gates.

| Arena candidate versus opponent | Wins / draws / losses | Paired mean final seed margin | Family bootstrap 95% interval | Primary margin gate |
| --- | ---: | ---: | --- | --- |
| B value + ordering versus H_v1 | 24 / 2 / 22 | +2.500 seeds | [+1.083, +3.917] | Passed |
| H ordering versus H plain | 22 / 4 / 22 | -0.792 seeds | [-1.583, -0.125] | Failed |
| B value + ordering versus B value plain | 22 / 4 / 22 | -0.333 seeds | [-0.917, +0.167] | Failed |

These comparisons do not identify the source of the joint model's margin advantage: the H and learned-value ordering ablations show no positive ordering contribution. Value-only training objective A has no paired arena here. WDL differences are small, and the same development families were used in earlier experiments. The positive seed-margin interval is evidence within this fixed development sample, not confirmation on fresh families, general championship strength or lower energy use.

Current ordinary H6-reference choice metrics are:

| Engine | 25ms agreement / mean regret | 100ms agreement / mean regret | Mean completed depths 25 / 100ms |
| --- | --- | --- | --- |
| Frozen H_v1 | 79.2% / 0.514 seeds | 77.2% / 0.668 seeds | 10.25 / 11.69 |
| H plain | 79.2% / 0.514 seeds | 77.2% / 0.653 seeds | 10.27 / 11.67 |
| H ordering | 79.5% / 0.514 seeds | 78.0% / 0.610 seeds | 10.03 / 11.40 |
| A value | 74.9% / 0.803 seeds | 73.0% / 0.934 seeds | 9.32 / 10.73 |
| B value plain | 77.6% / 0.703 seeds | 71.4% / 0.965 seeds | 9.30 / 10.72 |
| B value + ordering | 75.7% / 0.753 seeds | 73.0% / 0.888 seeds | 9.27 / 10.70 |

All 259 current ordinary roots produced completed recommendations for all six engines at both budgets in V2. H6 agreement is a proxy metric: deeper choices can disagree with H6 without being worse in the full game. Learned search still trails the H6 reference choice metrics and searches about one choice shallower. Current ordinary wall maxima were 26.95ms at25ms and 106.51ms at100ms; deadlines are cooperative checks, not hard real-time guarantees.

The opening analyzer demonstration preserved a committed depth-eight result across an exhausted 25ms attempt at depth ten, then resumed to depths nine and ten with a 1,000ms budget. It uses H leaves with optional B ordering and remains a finite-horizon analysis. Commands and four invocation logs are in `data/stage5-action-demo` and the progress history.

The next useful work is deeper, consistently scoped teacher targets and relay-aware tactical features, plus a new predeclared family sample and direct B-value versus A-value arena. The current experiment does not justify a bigger model or making learned ordering the default. Continue to reserve state/quotient/family keys across archived and pruned data, and keep sealed targets unopened until a candidate clears development gates.


Final acceptance records **32 passing checks**: five training, five native, four CPU timing, nine independent and nine progress checks. The independent auditor validated all 391 cases and literally replayed all 144 games; real process-kill checks cover training, analyzer depths, position cases and complete arena games. See `build/stage5-action-acceptance.json` and the [independent audit](STAGE5_ACTION_INDEPENDENT_AUDIT.md).
