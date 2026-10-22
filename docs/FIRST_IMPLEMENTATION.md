# First Congklak implementation scope

Implement a correct, resumable command-line position analyzer for one fully declared experimental game. Its first release uses a handcrafted evaluator and one CPU search worker. Database generation and training follow this usable baseline. This backlog has now been implemented; see the [acceptance record](IMPLEMENTATION_ACCEPTANCE.md) and [independent implementation audit](IMPLEMENTATION_AUDIT.md) for the verification results and remaining limits.

## Selected scope and rules

Use `id-betawi-relay7-v0` as an explicitly experimental engineering contract, with exact pit counts and final single-round seed margin. It is based on the proposed profile A in the [rules report](research/congklak-rules.md). Its implementation does not claim an official national standard or that the user's local practice has been confirmed. A changed convention creates a different immutable rules digest and separate results.

The contract to serialize and freeze is:

- Two players, seven pits each, seven initial seeds per pit, 98 total seeds and two initially empty stores. Imported stable positions must also conserve 98 seeds.
- Ring order `A0 A1 A2 A3 A4 A5 A6 SA B0 B1 B2 B3 B4 B5 B6 SB`; opposite pits `A_i` and `B_(6-i)`. A skips SB, B skips SA. A later lap includes the emptied source pit.
- A legal action selects a nonempty own pit. Complete all deterministic deposits and relay pickups before another choice. Relay from either row when the last pit was occupied before its final deposit.
- An own-store ending grants another choice. An own-empty ending captures the landing seed and opposite contents only when the opposite is nonempty. With an empty opposite, retain the landing seed. An opponent-empty ending retains that seed. These pit endings end the turn.
- No prior-lap capture restriction. After a complete action, check whether both rows are empty; that is terminal. Otherwise assign nominal control according to the ending, then normalize an empty-row controller by a forced pass to the other player.
- A depleted row can receive seeds later. No immediate one-row sweep, early majority ending, refill rounds, closed/protected pits, simultaneous opening or uncertain counts in this release. Reject unsupported states or profiles with structured errors.
- Player 0/A starts the standard opening. Final player-0 margin is `SA-SB`; positive/zero/negative gives win/draw/loss for A. Position imports can name either input controller.

Internally, values always use fixed player-0 perspective: player 0 maximizes and player 1 minimizes. Preserve `input_controller`, `normalized_controller` and pass events. Candidate moves belong to the normalized actor; a forced-pass input first reports the pass and labels any continuation choices as the other player's actions. Display `score_player` explicitly. Converting player-0 intervals [L,U] to player-1 uses [-U,-L]. Terminal reports preserve the requested display perspective without inventing a next move.

## Ordered backlog

| Step | Implement | Completion gate |
| --- | --- | --- |
| 1 Contract and encoding | Canonical rule JSON, digest, stable count encoding, coordinate map and validation | Every transition-affecting field is fixed; malformed or unsupported inputs are rejected |
| 2 Independent interpreters | A simple seed-by-seed reference and a C++20 production kernel with copy/apply | Golden traces and exhaustive reduced-instance differential checks pass |
| 3 Deterministic search | Single-worker full-width minimax reference, alpha-beta and iterative choice-depth search | Values and tied best-move sets match at the same horizon/evaluator |
| 4 Local protocol and reports | NDJSON CLI, all legal candidates, trace facts, PVs, actor/perspective fields and truncation | Forced passes, terminals and incomplete searches produce honest reports |
| 5 Durable analysis | SQLite input/settings/report generations, cooperative stop and resume | Completed iteration survives process termination and resumes without changing fixed-horizon semantics |

These form one first release. TT, PVS, aspiration windows, enhanced evaluation and within-tree parallelism are later optimizations with reference agreement gates. Start with no TT so cache identity and replacement cannot obscure the first search checks. Do not require a GUI, learned model, tablebase K=8, MTD(f), proof campaign or the separate published multiround Dakon profile before this release.

## Strict baseline search contract

Use `H_v1(s)=store0-store1` at a nonterminal horizon leaf, in integer seed units. This is a deliberately simple baseline evaluator; it is not a promise of strong play. Normalize control before branching. Terminal checking comes first. With no oracle enabled, depth zero returns H_v1; otherwise explore every legal pit choice with depth reduced by one, using MAX/MIN according to the actual controller. Automatic relays and passes consume no choice depth.

For all displayed root candidates, search each child with a full window at the same remaining horizon. Publish the new ranking only after the complete iteration. Deeper or later optimizations may retain per-move bounds, but the first release avoids treating root cutoffs as full candidate scores.

Return ordinary results as finite-horizon results under the exact evaluator/policy identity. Keep the rule-derived physical interval separate. The first release does not prune using eventual-game bounds or tablebase values. A future oracle-enabled recurrence must declare its lookup ordering, coverage and exclusion manifests in the teacher/search identity; it is a different horizon policy.

Limits include maximum choice depth, node budget, wall time and a cancellation token. Proposed interactive starting defaults are one second, depth 16 and one million decision nodes, whichever stops first; they are configurable starting settings, not measured performance. Check cancellation at search nodes and periodically during seed deposits. An interrupted `apply` uses scratch state and returns cancellation without publishing a half-sown successor.

The CLI operations are `validate`, `apply`, `analyze`, `stop`, `load` and `resume`. The search remains one worker; a local coordinator can accept a stop request and commit progress. Show legal candidates immediately, but report no completed ranking if iteration one has not finished. Generate explanations from exact transition facts and clearly conditional/truncated continuations.

## Minimal persistence and resource limits

Store small reports and their canonical input/settings inline in SQLite transactions. Commit job creation before searching, and commit a completed report together with its iteration/progression state. Startup restores only committed rows. A running job interrupted by process exit becomes resumable pending work; it does not become a completed or drawn result.

Resume retains the last complete report and recomputes unfinished work. It need not restore a recursive stack or identical timing/counters. Compare resumed and uninterrupted behavior at a fixed choice horizon with identical settings; elapsed-time-limited runs may complete different depths. Time added by a resume request is recorded as additional work.

Initially run no background training or corpus generation. Use a 512 MiB managed-memory budget with bounded buffers and process-use monitoring, configurable beneath the shared 24 GiB ceiling; this is not an untested hard OS-limit claim. Stop cleanly before publishing incomplete work when the limit is reached. Default development job data is `F:/My Engine/Congklak Stockfish/data`, with an explicit `--data-dir` override resolved to an absolute writable local path.

Before starting search, validate the data directory, commit the job and ensure projected metadata/WAL/backup writes fit the quota and actual disk capacity, with at least 256 MiB recovery headroom for this inline-only release. Fail before search with a structured storage error when admission cannot fit. The 50 GiB cap later counts all physical temporary/fallback artifacts, with at least 2 GiB reserved for bulk rewrite overhead. The full [recovery contract](research/compute-and-recovery.md) defines later chunk publication and collection; those mechanisms are not prerequisites for an inline report.

## Acceptance checks

Run meaningful checks before describing the baseline as correct:

1. Execute the initial profile's golden fixtures for store bonuses, own capture, empty opposite, relay on both rows, source revisits, opponent-store skipping, forced pass/revival and terminal ties. Include both player perspectives and trace reconstruction.
2. Exhaustively compare one-choice transitions and fixed-horizon searches at choice depths 0 through 4 on small test profiles with one to three pits per side and zero through six total seeds, including stores and both controllers. Keep test IDs separate from the product profile. Also compare 1,024 generated legal 98-seed histories with recorded seeds 0 through 1,023 and curated long relay/bonus-chain positions. A test work guard returns an explicit incomplete/failure outcome, not a fabricated draw or a silently skipped acceptance case.
3. The reference interpreter shares the encoding specification and fixtures, but not production transition helpers. Require seed conservation, legal-action agreement, exact successor agreement and actual-controller agreement.
4. Compare alpha-beta with full minimax for several fixed horizons, including repeated bonus choices, forced passes and an initially losing actor. Require identical values and the complete optimal-action set under H_v1. Displayed root score comparisons use matching horizons.
5. Terminate analysis before and after job/report commits, cancel during sowing, restart and resume. Preserve the last completed report or explicitly report that none exists. Exercise full disk and malformed/incompatible saved records. Check fixed-horizon semantic equality, not equal elapsed-time results.
6. Verify every displayed capture/pass/bonus fact by replay, and every recommended pit belongs to the displayed actor. A heuristic recommendation never receives a full-game proof label merely because the horizon search completed.

If a check fails, fix the underlying kernel/search/report behavior before optimizing or generating training labels. Record the test corpus identity, compiler/engine version and evaluator identity with the acceptance results.

## Immediate follow-ons

After this analyzer passes, add a bounded full-state-checked TT and PVS independently, preserving finite-horizon agreement. Build a K=4 exact oracle with explicit delta-award recurrence and verifier as a small separate aid. Start the 10,000-position pilot with validation, deduplication, named splits and uniform stratified sampling. Measure actual teacher quality and throughput before selecting production generation budgets.

Then compare a tiny MLP on a 100,000-position corpus, using seed-unit loss and development-only promotion. NNUE, K=6/8 tablebases, priority sampling, label reliability coefficients, quality-based pruning and GPU PUCT enter through measured ablations. The [database and training plan](DATA_AND_TRAINING_PLAN.md) defines those gates. This ordering keeps the first product useful while later improvements remain independently assessable.
