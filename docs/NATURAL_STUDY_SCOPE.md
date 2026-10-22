# Bounded natural ordinary midgame and teacher-stability study

This study collects data and compares finite-horizon teacher estimates. It does
not train a network, run PUCT or strength games, expand a model, or promote an
engine. The fixed quality control model remains SHA256
`bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`.
All 2,060 baseline inputs are preserved, including 2,058 sealed accepted inputs
and their completed acceptance/rerun evidence.

The complete fixed pool contains 512 new standard-opening legal prefixes from
seed 20261021. The sealed pilot generator is reused unchanged: even indices
choose random legal moves; odd indices choose eight random opening moves then
the fixed policy with epsilon 0.25. Ties choose the lowest legal pit. Prefixes
stop at terminal, below 25 active seeds or 192 choices. Hash family ownership is
assigned before generation; there are exactly 256 families of each source kind.

Fresh roots must have 50–74 active seeds and be ordinary under the existing
priority classifier across all legal root moves. No constructed filler, quiet
criterion relaxation, automatic extra families or expanded node budgets is
allowed. The complete pool is generated before selection. Cross-cohort eligible
quotient collisions are excluded from both cohorts. Previously protected roots,
all 62 prior pilot quotients and all 256 prior pilot family IDs are excluded.
This is a selected-root/family bridge; shared and previously unlabeled history
prefixes remain possible. No all-prefix or unseen final-test claim is made.

Selection keeps the first eligible family/step/key in declared order, at most
one fresh root per family and one owner per quotient, up to 24 training-candidate
and eight validation roots. Missing quotas are recorded honestly. These remain
untrained candidates. Previous model and experiment exposure cannot be erased.

All 15 prior 50–74 pilot roots are followed up separately: 12 training candidates,
two validation roots and one protected diagnostic. Their original root bodies,
ownership, H8 labels and reports remain unchanged. Study wrappers bind those
original hashes and identify follow-up status; inherited H8 references are not
counted as newly executed teacher searches and cannot enter the fresh pool.

Every selected fresh root gets one new H8 attempt with a two-million-node cap.
Then every selected fresh root and all 15 old follow-ups get one H10 attempt
with a five-million-node cap, even when a fresh H8 attempt was rejected. Both
use the existing oracle-free H_v1 engine, full-root action scores, PVS, 16 MiB TT,
512 MiB memory and no wall-clock cutoff. Only completed full-Q depth-eight or
depth-ten reports become labels. H8 action scores have H7 successor horizon;
H10 action scores have H9 successor horizon. No successor is mislabeled as a
root target. Partial reports and actual work are retained as rejected-attempt
evidence, with stage-specific depth bounds, and never become admitted labels.

H10 is a deeper estimate, not game truth. Paired value, policy-regret or action
ranking comparisons use only the complete H8/H10 intersection and report all
unpaired roots and rejected attempts. A deeper label may disagree without proving
that either horizon is objectively correct.

An exclusive writer, SQLite FULL/WAL transactions, deterministic family replay,
per-family and per-attempt commits, stage-aware report validation and immutable
closed manifests support interruption recovery. Real pre/post commit hooks exist
for collection, fresh H8 and H10 attempts. Cancellation leaves the current
uncommitted target for resumption. Completed reruns preserve journal/publication
bytes and timestamps and do not rerun teacher searches. Author fixtures use
smaller new-prefix pools and subsets of the real old follow-ups in build paths,
with the same strict classifier and exact H8/H10 node budgets.
