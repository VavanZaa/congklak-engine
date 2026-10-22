# Native ownership trace implementation

The versioned engineering interface is implemented and author checks pass.
**It is not admitted for production or equal-time strength conclusions.** The
full trace and its validation currently cost too much on branching baseline
positions for a 25-ms whole-decision target. Models, policy and the old arena
hold remain unchanged. No training, new teacher labels or production games
occurred in this native work.

`congklak/native_ownership_trace.py` supplies `open_providers(ExitStack)`,
`timed_decision(engine,state,ms=25,decision_id=...)`, fixed-128 PUCT decisions,
fixed-128-node baseline decisions, bounded fixed-depth baseline diagnostics and
`reset_all(reason='handoff',decision_id=...)`. Baseline decision nodes and PUCT
edge backups are explicitly different work units. The live two-seat handoff is
owned by a separate recoverable worker; this provider does not claim that worker
has been tested merely because a reset method exists.

All four profiles use the new unified DLL at
`build/ownership-implementation-v1/native/congklak_ownership_trace_v1.dll`.
Versioned copies live in `native/ownership_v1`; original engine sources, DLLs,
models and bindings were not edited. Checked source projections retain the
frozen originals' hashes and exact trace insertions. Dense value/policy math,
selection and deterministic ties remain unchanged. One declared interface
change makes the H8 relay baseline's decision-local cache include both raw
stores, together with its fourteen pits and actor under an immutable private
model/rules/backend/search context. The preceding residual cache was correctly
store-offset invariant, but did not literally use the newly required raw key.
This can change physical cache-hit/forward counts without changing fixed-depth
scores. In one author fixture the new cache used 357 forwards versus 356; the
full root-Q values, best moves and PV were identical.

Native trace hooks cover complete decode, normalize/pass input/output, legal and
terminal tests, margin evaluation, move input/output, alpha-beta/PVS and relay
recursive/PV/root nodes, TT probes/hits/resident entries, evaluator-cache probes
and hits, model forward starts/completions, feature inputs, both actors'
hypothetical move inputs/outputs and sidecar branch/bypass/completion. Versioned
Python PUCT adds explicit tree-node, selection and expansion markers, using
the same traced DLL for every rule transition and evaluator. Temporary
seeds-in-hand arrays never enter the complete-state event sink.

Each event is a lossless 24-byte little-endian record: sixteen raw ring cells,
actor, operation, scope, reserved byte, signed 16-bit action and depth. A native
thread-local bounded buffer records events without per-state Python callbacks.
The default capacity is 200,000 events, with an explicit engineering ceiling
of 2,000,000 (48 MB of raw event storage). Capacity exhaustion throws a hold;
there is no accepted truncation or silent drop. Compressed base64 bytes, SHA-256,
operation counts, attempted/stored event counts, source/build/model/rules/search
identity and decision cursor form the JSON trace. Copy, compression, hashing
and full packed validation are included in the reported whole-decision clock.

`validate_decision(answer,engine,state,decision_id,identity=...)`,
`validate_trace`, `iter_trace_states` and `compile_footprint` are persisted-evidence
readers with no DLL load, model construction, forward, search, optimizer or
writer. Validation links events to separately captured native operation counts,
played endpoints, physical alpha/relay node totals, PUCT nodes/expansions,
completed forwards, feature attempts and completed transitions. It rejects an
empty trace and a dropped/resealed event partition. Incomplete timed operations
retain attempted inputs and explicitly bounded unfinished moves; they are not
fabricated into completed feature outputs.

Every PUCT decision replaces its inference control and creates a new tree and
transition control; evaluator cache capacity remains zero. Baseline controls,
TTs, residual caches, ordering data and search objects have single-call lifetime.
Native monotonic generation IDs and created/destroyed counters attest actual
control replacement, rather than treating a metadata flag or possibly reused
pointer address as proof. `reset_all` replaces both retained inference controls.
Immutable model weights and observational counters persist; counters do not
influence search. There is no move-selection RNG or root noise.

The accepted author source witness and 32 checks are in
`build/ownership-implementation-v1/native/author-v2`. They include 36 original/
traced fixed-128 PUCT searches (4,608 backups), eight original/traced depth-four
baseline searches with complete root-Q and PV agreement, exact low-24
control/candidate fixed-work agreement, two fixed-node baseline checks, actual
control reset, trace-drop refusal and a retained capacity failure. The shared
24-seed fixture checks frozen-path equivalence; it is not an exact K24 oracle.

The branching-position timed stress used the declared 25-ms positive remaining
search-grant policy and an explicit 2,000,000-event capacity. Whole times include
trace validation and do not hide its cost:

| Engine | Events | Compressed trace bytes | Whole time |
|---|---:|---:|---:|
| H_v1 | 551,189 | 2,162,462 | 306.34 ms |
| H8 relay CPU | 445,272 | 1,961,474 | 259.08 ms |
| Control PUCT | 15,077 | 56,196 | 31.60 ms |
| Candidate PUCT | 15,774 | 57,644 | 31.71 ms |

These timings are engineering observations on one existing consumed root,
not equal-time matches or population benchmarks. Raising trace capacity preserves
data but does not solve finalization/ownership-check time or a future 5-GiB
storage budget. Production needs an audited bounded capture/compilation and
deadline-accounting design before more games can support equal-time claims.
The search grant was not silently reduced after seeing these measurements.

The earlier author-v1 attempt correctly held when the branching H_v1 stress
exhausted its default 200,000-event buffer. Its source witness and completed
fixed-work records remain retained, together with `failed-attempt.json`; that
failed harness did not persist the in-flight timed trace. Treat the missing
partial telemetry as disclosed unreported work, not zero cost. Earlier ad hoc
smokes are engineering attempts, not additional accepted audit methods.

Native-owner cumulative accounting, including those smokes and the failed
attempt, is 106 search attempts: 77 completed fixed-128 PUCT searches (9,856
backups), 17 fixed-depth baseline searches, six fixed-node baseline searches,
five timed attempts including one failure, and one deliberate tiny-capacity
attempt. This excludes independently owned worker/auditor work, which must be
added separately. All training, optimizer-draw, new-label and production-game
counters for this native work are zero. Independent acceptance remains separate
from these author checks.
