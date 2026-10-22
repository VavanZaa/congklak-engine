# Ordinary early/midgame data pilot

This experiment collects and labels data. It does not train, expand a model,
change the fixed policy/value control, perform strength testing, or promote an
engine. The fixed control export is `data/quality-train-v1/model.json`, SHA256
`bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`.
H_v1 remains the default. All 1,834 baseline inputs are preserved, including
1,829 previously accepted inputs and completed diagnosis evidence.

The fixed pool has 256 families from seed 20261020, starting from the standard
7×7, 98-seed Betawi opening. Even families choose random legal moves. Odd
families choose eight random opening moves, then follow the fixed policy with
epsilon 0.25; ties choose the lowest legal pit. Every family stops at terminal,
below 25 active seeds, or 192 choices. A complete recorded legal prefix is kept.
Family ownership is assigned before generation by SHA256 of
`data-pilot-owner-v1/` plus the family ID, first eight bytes modulo 100: below
80 is training-candidate and the remainder validation.

Every candidate must be ordinary under the existing `pilot._mechanism` priority
classifier across every legal move: no legal pass, capture, or bonus transition.
This is a root property, not the reason for one chosen move. Terminal children
are allowed where this existing classifier says ordinary, and their count is
recorded. There are no constructed fillers and no relaxed quiet-state test.

Collection finishes the entire fixed pool before selection. A quotient observed
as an eligible candidate in both cohorts is excluded from both. Old protected
quotients, all 9,000 existing H8 roots and all 192 value-arena histories are
excluded from fresh roots; protected family IDs are excluded as well. This is
a selected-root ownership bridge. Standard-opening and other history prefixes
may overlap; this experiment does not claim all-prefix disjointness, erased
ancestral exposure, an unseen final test, or independent game-level samples.

Selection takes the first eligible family/step/key in declared order, with one
row per family per active band and unique selected quotient ownership. Caps are
32 training-candidate plus eight validation roots in each 25–49 and 50–74 band.
The naturally occurring 75–98 band is observational only, capped at eight plus
two. Shortfalls are reported rather than extending the pool or changing targets.
The pool is not automatically enlarged beyond 256 families.

Up to 16 unique ordinary control roots from its existing 96 arena games receive
diagnostic labels. They must be absent by canonical key and quotient from the
9,000 labeled roots. Selection uses canonical-key order, never outcomes or new
teacher errors. These roots remain protected diagnostics and are never training
or validation candidates. Existing PUCT choices are retained only as provenance.

The teacher is the existing oracle-free H_v1 native engine, full root depth eight
with all action scores, PVS and 16 MiB TT. Each independent label has a fixed
two-million-node budget, no wall-clock cutoff, and 512 MiB memory limit. Complete
H8 reports and labels are checked by existing validators. H7 successor action
scores remain policy targets at their original root; they are not root H8 child
value labels. These are approximate finite-horizon labels, not exact game values.
Budget-exhausted attempts are retained without any partial label. Cancellation
leaves the current uncommitted attempt for resumption.

SQLite uses an exclusive writer and FULL/WAL transactions. Each family is a
unit; root selection is atomic after full collection; each attempt and complete
label commit atomically together. Pre/post commit hooks permit real crash tests.
Resume replays deterministic generation, literal histories, ownership and saved
labels. Completed reruns preserve publication bytes and timestamps. Source,
rules, native build, fixed model, registry, protocol and prior evidence are pinned
before production. Author fixtures use smaller pools in separate build paths,
with the same strict classifier and full H8 teacher protocol.
