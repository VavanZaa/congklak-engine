# Independent first implementation audit

Reviewed 14 October 2026, Asia/Bangkok.

**Accept the first implementation scope, steps 1 through 5, for the experimental `id-betawi-relay7-v0` profile.** The final acceptance run passed **31 tests, with zero failures, errors or skips**, in 22.734 seconds. No unresolved implementation blocker was found in the reviewed scope. This verdict concerns the declared rules, finite-horizon search, protocol and recovery behavior; it does not establish playing strength or solve the full game.

The final machine-readable [acceptance manifest](../build/acceptance/acceptance.json) records the exact source hashes, compiler, binary, environment and counts. The [test log](../build/acceptance/acceptance.log) records each executed case. [Acceptance instructions and results](IMPLEMENTATION_ACCEPTANCE.md) describe how to repeat the checks.

## Independence and reviewed code

The implementation auditor wrote [reference.py](../tests/reference.py), [native acceptance tests](../tests/test_native_acceptance.py) and [saved-record validation tests](../tests/test_saved_validation.py), and reviewed the production code without editing it. The reference interpreter imports no production helpers: it sows one seed at a time, reconstructs traces, normalizes control and performs full-width minimax without alpha-beta, transposition tables, oracle facts or physical-bound pruning. Production authors corrected the findings below before acceptance.

Review covered the frozen rule JSON and ring encoding, `native/engine.hpp`, `native/rules.cpp`, `native/search.cpp`, the ctypes boundary, input validation, report validation and replay, NDJSON coordinator, SQLite storage, human explanations, build scripts and subprocess recovery tests. The auditor independently ran the complete suite against the final binary, then checked the final explanatory change and additional real SQLite-full case separately. The final combined 31-test manifest and log were re-read, and every recorded source hash matched the final files.

## Executed acceptance evidence

| Gate | Verified result |
| --- | --- |
| Reduced exhaustive rules | All 8,274 states for one through three pits, totals zero through six, including stores and both input controllers; all 11,950 legal one-choice transitions agree with the independent interpreter |
| Reduced exhaustive search | All 41,370 state/horizon cases at choice depths zero through four agree with full minimax, including every root candidate score and complete tied optimal-action set |
| Product histories | 1,024 reproducible histories with seeds 0 through 1,023; 29,249 legal actions, each checked for exact successor, controller and conservation |
| Golden rules and trace facts | All 25 manual fixture cases pass, plus independent fixtures in both player orientations; replay checks capture, store bonuses, empty opposite, relay on either row, source revisit, skipped opponent store, passes, revival and terminal ties |
| Reports and continuations | Actual actors, player-0 and displayed-player values, sign-reversed physical intervals, equal root horizons, lexicographic optimal PVs, PV leaf values, pass/bonus/terminal facts and truncation agree with replay; heuristic results retain a finite-horizon label |
| Limits and interruption | Node, deadline and process-memory limits return honest incomplete statuses. Token cancellation interrupted the curated 489-deposit action after 64 deposits in the final run; no successor or partial trace was published, and a fresh complete apply still agrees with the reference |
| Control lifetime | Running or closed tokens cannot be reused or destroyed; a token from another native instance is rejected; cancellation remains available while the worker runs |
| Protocol | Validation, apply, explicit forced pass, analyze, stop, load and resume work through a real process; malformed, deeply nested and oversized requests are recoverable structured errors; busy/read/stop behavior and EOF completion pass |
| Durable transactions | Real child-process termination before report publication, after publication, and inside uncommitted job/report transactions preserves only committed inputs and generations. Restart and resume match uninterrupted fixed-horizon values, tied moves and PVs |
| Storage failures | Quota and recovery-headroom admission fail before search; real `SQLITE_FULL` error 13 during report publication, injected transaction abort and low-free-space admission preserve the previous report, including after restart |
| Damaged saved records | Checksums, strict schema/types and replay reject malformed reports; the older valid generation is restored or the job is explicitly unranked. Fourteen checksum-valid report mutations, null/incomplete saved roots/settings, incompatible identities and unsupported schema versions are exercised |
| Human explanations | Store gains, capture counts, actor labels, passes and terminal/conditional continuation text are checked against independent replay. A store ending followed by a forced pass reports the pass rather than promising an available extra choice |

The native acceptance class recorded **41,230 checked transitions and 41,395 horizon-report comparisons** across its fixed and limit-related cases. Reduced test profiles remain confined to the native acceptance boundary; the public protocol accepts only the seven-pit, 98-seed product contract.

The history corpus is fixed by CPython `random.Random(seed)`, the standard opening, a requested length from `randrange(1,65)`, uniformly selected legal choices and early termination only at a terminal position. Canonical JSON records contain the seed, action list, state rings and controllers, one record per newline. Its independently calculated and asserted SHA-256 is:

`ab646e915eaa173d84829b6ccf261a0b51629ea6553bae12de41cd9cb9fdbf72`

## Findings corrected before acceptance

| Finding | Correction rechecked |
| --- | --- |
| ctypes apply could race token destruction or share a running token | Apply and analyze use lifetime guards and enforce native-instance ownership; regression tests pass |
| Native summary initially double-quoted the escaped stop reason | JSON serialization corrected; all protocol and limit responses parse |
| Terminal own-store PV incorrectly advertised a bonus choice | Terminal bonus flag corrected and independently replayed |
| Saved reports were accepted with missing result fields or a string terminal flag | Strict report schema, claim/perspective/ranking validation and native replay now run before commit and on load; checksum-valid malformed records restore a fallback |
| Python boolean/integer equality admitted malformed controller, PV and trace integers | Exact scalar types and canonical transition comparison reject these aliases |
| Null saved roots/settings silently selected defaults and could leave an active coordinator without a worker | Saved records require explicit valid fields; startup prepares the root/token before committing the run and activating the worker |
| Deeply nested request JSON could terminate the request loop | Per-line recursion errors are handled; the next valid request succeeds |
| Lowering resume depth returned a deeper report as already complete | A lower nonterminal horizon is rejected with `invalid_resume_depth`; the saved report is retained and a new analyze job can request the smaller horizon |

## Accepted identities and limits of the evidence

- Rules digest: `c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e` (canonical JSON digest).
- Native ABI: `congklak-native-v1`; final DLL SHA-256: `d0b2154a6939f8b913801e7d1ea1ec19ea047b831129e9d6f72756cff0ba7867`.
- Evaluator: `H_v1`, store 0 minus store 1 in integer seed units; search policy: `terminal-first/no-oracle/no-TT/full-root-window-v1`.
- Environment: Windows 11 x64, CPython 3.12.10, MSVC 19.44.35229.0 / toolset 14.44.35207, C++20 `/O2 /W4 /EHsc /LD`.

The exhaustiveness claim applies to the declared reduced corpus, not every 98-seed position. Product coverage uses recorded histories and curated cases. Cancellation scheduling and equal wall-time completed depths may differ between runs; fixed-horizon semantic equality is the recovery requirement. Memory checks monitor process use with bounded buffers, rather than imposing a hard operating-system limit.

Disk-full behavior was exercised through admission tests and an actual SQLite page-limit failure without filling the physical drive. Transaction recovery was tested with real process kills, not hardware power-loss testing. Saved-report replay validates evidence and schema; it is not an independent minimax proof for arbitrary imported or modified checkpoints. Search correctness has the separate differential corpus above.

This release supports one experimental single-round profile and a deliberately simple evaluator. Other local conventions, playing-strength measurements, training, transposition tables, tablebases, NNUE and within-tree parallelism remain follow-on work, as specified by the accepted plan.
