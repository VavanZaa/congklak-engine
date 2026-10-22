# CPU PUCT independent audit

Status: **passed**. All 13 independent acceptance methods passed against frozen implementation `522e6bb546deddfe217ed80d421116e9e8911fc9382a16abdf02ce59909edec1`. The complete production audit also passed. Engineering and cost acceptance do not establish playing strength: every one of the six candidates failed the declared gate against both baselines, so H_v1 remains the default.

Machine-readable evidence is in [audit.json](../build/puct-cpu-audit/audit.json), with [13-method coverage](../build/puct-cpu-audit/tests.json) and source-bound group reports. The full audit is an additional inspection, not a fourteenth unittest or a second count of existing tests.

## Scope and independent verification

The experiment reused the two frozen Phase 6 models, 80 positions and 24 development arena families. It compared the legacy two-feature-pass adapter, a structured single-pass CPU adapter and the same adapter with a bounded 4,096-entry LRU cache. There was no training or model-weight change. All 1,192 accepted Phase 6 inputs and the earlier 878 Stage 5 inputs retain their original hashes.

The independent checks rederived cohorts, profiles, clocks, budgets, CPU identity and balanced 576-game work order; compared both models against literal Python features and dense forwards; verified actual cache behavior on 4,097 distinct inputs; and compared noisy fixed-simulation forests, visits, priors, values and RNG states. Invalid ABI/build records, oracle-marked models, invalid controllers, closed evaluators, impossible feature counters and corrupt stored results were refused. The completed audit additionally verified all 1,000 development positions with each frozen model against literal forward calculations.

The complete production audit checked every saved unit and rederived its published summary:

| Evidence | Independently verified coverage |
| --- | ---: |
| Case positions | 80 |
| Scalar engine/budget measurements | 1,280 |
| Arena games, all terminal | 576 |
| Literal arena choices | 18,277 |
| Paired arena comparisons | 12, each with 24 families and both seats |
| Speed units | 1,410: 6 startup, 36 parity, 504 micro, 864 fixed trees |
| Original legacy trees retraced | 96: both models across 48 roots |
| Literal full-model development forwards | 2,000 |

The microtest audit independently reconstructed scheduled checksums and actual physical feature work. The tree audit reconstructed original evaluation traces, distinct raw states, cache hits/misses, dense forwards, simulations and seed deposits. It checked exact tree semantics against the legacy path and exact half feature work for every matched single-pass tree.

## Cost and strength findings

The predeclared cost gate passed for both models. For each root, the audit took the median of three matched single-pass/legacy whole-tree wall ratios, then the median across all 48 roots. The warm model ratio was **0.4806577242** and the updated model ratio was **0.4767758619**. Single-pass feature simulations and deposits were exactly half the legacy work; all 864 fixed-tree results agreed exactly.

Cache gains were not required by the protocol. Actual fixed-tree cache hits were 204 of 9,312 warm requests and 222 of 9,312 updated requests, approximately 2.2% and 2.4%. These low hit rates and the measured cache overhead do not establish a cache speed gain. Hot-prefilled microtests measure a retrieval ceiling and cannot substitute for real-tree throughput.

All 576 arena games reached literal terminal states without fallbacks. Every candidate's paired mean final margin remained negative against each baseline; the twelve exploratory 95% bootstrap intervals also remained below zero. The full audit independently reconstructed all intervals from the 24 paired family margins. The updated single-pass candidate scored -13.0 seeds against H_v1 (interval -17.1667 to -9.0833) and -16.625 against H8 relay CPU (interval -20.5 to -12.6667). The optimization reduces evaluation cost while the existing model/search combination remains weaker than the baselines.

Original outer process observations covered 65.266 seconds for the speed producer and 495.489 seconds for the strength producer, including source guards, imports, admission, startup, journals and publication after argument parsing. Their peak private commit was about 227.1 MiB each. The independently verified inner benchmark observation was 495.009 seconds. Both original runs began without prior committed production units. These are invocation observations, excluding Python startup and earlier invocations; they are not electricity measurements.

## Interruption and corruption recovery

Actual analyzer processes were killed before and after committing a larger result. The prior 64-backup export survived; the 128-backup result resumed or was repaired; fresh semantics, explanations and physical counters matched; and completed bytes and mtimes remained stable. A case worker was killed after its first commit and retained that prefix on resume. A speed worker was killed after its first startup unit and completed all 1,410 units while retaining that original unit. Rehashed impossible counters and regret values were refused.

The final arena method copied the complete production journal, removed the final pair, killed the worker after game 575 committed, and resumed game 576. All 575 committed records remained unchanged. All 576 games were literally replayed, completed reruns preserved bytes and mtimes, and a rehashed final-margin corruption was refused. Recovery fixtures retain raw journals, validated good backups, kill markers and worker logs under `build/puct-cpu-audit/retained-*`. Fixture timings do not contribute to the production timing findings. The main production journals were not altered by the kill tests.

## Counter, clock and cache interpretation

The cache includes all 16 raw input cells, including both stores, plus the input controller byte. It is private to an evaluator/model instance. Equivalent forced-pass inputs may occupy separate entries; this reduces possible hits without creating aliases. Values and logits are cached immutably; returned logits are copied. Cache clearing occurs inside every decision clock. A remaining budget below one millisecond cannot invoke frozen search with `time_ms=0`, which would mean unlimited search.

Frozen PUCT `neural_evaluations` counts completed logical outputs, including the actually forwarded prefix of stopped batches where the frozen implementation records it. It does not count every cache request. Physical forward counts, actual feature simulations/deposits, and cache requests/hits/misses remain separate, especially on cancellation. Independent cancellation checks covered already-cancelled requests, discarded cache hits and lock-wait deadlines.

The frozen dashboard progress CLI records the child exit code and state in run JSON but does not propagate that code through its top-level shell entry point. A shell exit code alone therefore cannot establish completion. Acceptance validates actual producer journals, published reports and original process-resource observations. Correcting the shell exit behavior requires a future version rather than changing this frozen experiment.

## Limits

The twelve arena intervals are exploratory and unadjusted for multiple comparisons. These development families were used previously. There is no fresh or sealed-test result, no automatic promotion, no new training, no independent live warm-training-kill claim and no measured electricity use. Loaded/warmed models and cold evaluation caches describe different costs. The current gains support retaining the single-pass CPU adapter as an experimental path; they do not resolve the remaining model/search quality problem or establish a solved Congklak engine.
