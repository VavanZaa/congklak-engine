# Five-position fixed-simulation results

All **40 requested cells completed**, with **19,200 completed root simulations**, zero caps and no model or policy changes. The control and value-only candidate choose the same move in 17 of the 20 matched budget pairs. This diagnoses the retained search behavior; it is not a strength match, a promotion gate or proof of a game blunder. H_v1 remains the default and the previous allocation remains HOLD.

The panel includes every saved H_v1 first divergence: pairs 2, 4, 6, 7 and 8, all candidate seat1. Pairs 2 and 8 have unchanged timed margins and remain controls in the analysis. All five states are player1 positions selected from already-consumed development games.

## Every declared budget

Each entry below is **control pit / candidate pit**, using human pits 1-7. Pair IDs are the original zero-based identifiers. Every budget contains ten completed cells and zero missing cells.

| Pair | Saved timed actions | Timed margin delta | 128 simulations | 256 | 512 | 1024 |
|---|---|---:|---|---|---|---|
| 2 | 7 / 1 | 0 | 7 / 7 | 7 / 7 | 1 / 1 | 1 / 1 |
| 4 | 7 / 4 | -6 | 4 / 1 | 4 / 4 | 7 / 7 | 7 / 7 |
| 6 | 7 / 3 | -4 | 3 / 3 | 3 / 3 | 7 / 7 | 7 / 7 |
| 7 | 7 / 1 | -6 | 1 / 4 | 7 / 7 | 7 / 1 | 1 / 1 |
| 8 | 5 / 7 | 0 | 7 / 7 | 7 / 7 | 7 / 7 | 5 / 5 |

There are two model-choice differences at 128, none at 256, one at 512 and none at 1024. All budgets remain reported; agreement at a particular budget does not select that budget or establish correctness or convergence.

Pair 7 retains a deterministic difference at 512 root simulations. Control gives pit 1 201 visits and pit 7 202, choosing pit 7; the candidate gives pit 1 295 visits and pit 7 115, choosing pit 1. Both choose pit 1 at 1024. The two unchanged-margin controls also show budget sensitivity shared by both models: pair 2 changes from pit 7 to pit 1, and pair 8 from pit 7 to pit 5. The grid does not test every intermediate count or reconstruct a unique cause of the earlier wall-clock game results.

## What was controlled and measured

Every cell starts a fresh tree and reset cache. The frozen protocol uses the unchanged `quality_search.search`, c_puct .25, no time limit, no root noise, rng0, native CPU evaluation with cache0/max_batch1, max_nodes2048 and memory8192, pinned to the retained declared core2. The actual backend is `avx2-double-packed`. Trunk and policy tensors are bit-identical between models, and root priors match in all 20 paired cells. Only the already-trained value heads differ; no new training occurs here.

Equal simulation counts remove the earlier unequal-count stopping condition at these declared points. They do **not** imply equal tree shape, number of model forwards, CPU cost or energy. Production records contain 17,935 tree transitions, 17,877 physical dense forwards, 121,511 hypothetical feature simulations and 1,154,321 feature deposits. Raw reports retain node counts, visits, sampled action means, parent bootstrap, physical counters and observed timings. Unvisited action means remain `None`; parent bootstrap is not an observed action Q.

The finalizing process records **5.701145 seconds through prepublication validation**, including admission, and peak working set **135,151,616 bytes** with peak private commit **124,121,088 bytes**. Its resource scope excludes subsequent completion/export work, other invocations, lost uncommitted kill work, fixtures and independent audit. These are process observations, not whole-task energy or a speed comparison.

## Verification, recovery and limits

The four current search-owner methods pass. Independent preflight exercised real search-cell precommit/postcommit process kills at committed prefixes 1 and 2; resumed fixture results preserved the prefix and deterministic semantics. The independent [40-cell audit](../build/divergence-audit/search.json) separately replayed every production tree and verified budget, visits, nodes, physical work, unvisited values and policy identity before teacher production was permitted.

Two earlier engineering failures remain retained. A startup check initially expected the shorthand backend name `avx2`; it refused before any inference or PUCT search. The exact source/plan/implementation/report/logs are under `build/divergence-search-prefreeze-backend-name/`. A later owner stability assertion incorrectly included the ephemeral SQLite shared-memory cache mtime; its bytes and every durable artifact were unchanged. That source/report/log and diagnostic are under `build/divergence-search-prefreeze-sidecar-boundary/`. The corrected test excludes SHM and only absent or empty WAL; any nonempty WAL remains durable evidence. Neither correction changed budgets, models or traversal.

Per-cell attempts are atomic. Cancelled attempts remain typed without becoming completed cells; fixed-budget failures remain incomplete. Completed reopening validates raw reports, physical counters, source identity, completion marker and resource binding before repairing a missing export, and starts no evaluator or search. Verification is `python tools/probe-divergence-search.py verify`; the source-bound owner suite is a prepublication operation and should not be rerun against sealed evidence.

The [primary result](../build/divergence-search-result.json) and [resource record](../build/divergence-search-resources.json) retain the full evidence. The separately frozen [joined comparison](../build/divergence-comparison.json) evaluates saved and fixed-work actions against each complete requested H8/H10/H12 reference, with H10 primary and every missing horizon explicit. PUCT sampled means and finite-horizon teacher Q are different estimands. Neither this selected five-state grid nor agreement between its two models is full-game truth or authorization for more training.
