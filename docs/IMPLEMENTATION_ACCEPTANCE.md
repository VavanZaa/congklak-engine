# First implementation acceptance

The finalized first scope is implemented: a correct, interruption-resumable position analyzer for experimental `id-betawi-relay7-v0`. It runs on the local Windows x64 machine with a C++20 CPU worker, Python 3.12 coordinator and SQLite persistence. Database generation and neural training remain later milestones.

The independent auditor owns the reference interpreter and differential/checkpoint validation tests. Rules and search were implemented by separate agents; the primary implemented integration, persistence and process-recovery checks. Findings were fixed before release. See the [independent audit](IMPLEMENTATION_AUDIT.md).

## Acceptance evidence

The final suite has **31 test methods**, with no failures, errors or skipped cases. The machine-readable [manifest](../build/acceptance/acceptance.json) identifies the DLL, source/test hashes, corpus, compiler and execution result. The [execution log](../build/acceptance/acceptance.log) records individual checks.

| Gate | Evidence |
| --- | --- |
| Frozen rules | Canonical SHA-256 contract identity; exact 98-seed imports; unsupported rules/phases rejected |
| Golden behavior | 25 manually declared fixtures with full trace agreement, player swaps, capture, empty opposite, bonuses, relay on both rows, long-lap source revisit, skipped store, forced pass/revival and terminal tie |
| Exhaustive reduced corpus | 8,274 states covering 1–3 pits per side, total seeds 0–6 including stores and both controllers; 41,370 state/horizon cases at depths 0–4 |
| Full board differential | 1,024 legal histories with PRNG seeds 0–1,023, plus curated bonus, losing-player and long-relay positions |
| Search and explanation | Alpha-beta values, complete optimal action sets, full-window candidate values and deterministic PVs agree with independent full minimax; transitions/PVs and displayed facts replay correctly |
| Interrupted sowing | Cancellation after positive deposits but before completing a 489-deposit action returns no partial successor or trace |
| Process termination | Real child kills after job creation, after completed reports and inside uncommitted job/report transactions; no partial result is promoted |
| Resume equality | Resumed fixed-horizon scores, ties and PVs match fresh uninterrupted analysis; additional run budgets are recorded |
| Corrupted records | Checksums plus strict types/schema/replayed facts reject 14 checksum-valid malformed report mutations; recover the prior complete generation or explicitly report no ranking |
| Capacity and write failure | Physical quota/low-free admission fails before searching; real SQLite `SQLITE_FULL` (code 13) and an injected transaction abort preserve the previous report, including after restart |
| Protocol and resource limits | Malformed/oversized/nested requests recover per line; single worker, responsive stop, time/node/memory limits, explicit depth zero, EOF persistence and exclusive data-directory ownership |

The recorded corpus comprises **41,230 checked transitions** and **41,395 horizon reports** including additional curated cases. Run `py -3.12 tools/acceptance.py` to reproduce the suite and replace the local log/manifest.

Build identity:

- MSVC compiler `19.44.35229.0`, toolset `14.44.35207`, C++20 `/O2 /W4 /EHsc /LD`; final build emitted no warnings.
- DLL SHA-256: `d0b2154a6939f8b913801e7d1ea1ec19ea047b831129e9d6f72756cff0ba7867`.
- Rules digest: `c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e`.
- Evaluator `H_v1`; policy `terminal-first/no-oracle/no-TT/full-root-window-v1`.
- Python `3.12.10`, Windows 11 x64.

## Practical limits

Recovery recomputes unfinished depths, rather than restoring a recursive search stack. Fixed-depth semantics are deterministic; time-limited runs can finish different depths. Tests cover forced process termination and SQLite write failures, not physical power removal or hardware failure. Memory is sampled process use with bounded buffers, not an OS-enforced hard reservation.

The baseline evaluates store margin at a choice horizon. It establishes simulation/search/recovery correctness for the declared contract; it does not measure expert playing strength or prove the initial position solved. Exact DLL/rules/evaluator identities must match for resume. A lower requested horizon requires a new job. No background corpus generation or training starts in this release.

## Run

See [README](../README.md) for the protocol and interruption workflow. A fixed-depth opening analysis is ready in [examples/analyze-opening.ndjson](../examples/analyze-opening.ndjson):

```powershell
Get-Content examples/analyze-opening.ndjson | py -3.12 -m congklak
```
