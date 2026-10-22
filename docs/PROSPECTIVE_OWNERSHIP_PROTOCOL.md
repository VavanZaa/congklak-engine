# Prospective ownership and evaluation protocol

This protocol defines a future development experiment. Models, policy, engine
defaults and the preceding evidence stay fixed while this protocol is built and
audited. It does not run training, label positions, generate starts, run searches
or games, promote the candidate, or authorize those later operations. H_v1 stays
the default. The final400 candidate stays quarantined. The preceding 192 games
keep their original failed ownership gate and descriptive-only results.

The proposed experiment measures **terminal seed margins of a switched engine**:
the candidate or control chooses the learned seat's moves above 24 active seeds;
below that boundary the learned seat uses the same frozen control continuation.
Its opponent remains the assigned frozen baseline. This isolates a prospectively
defined prefix treatment while allowing shared frozen endgames. It does not
measure unrestricted candidate play from opening to terminal, prove optimal
play, or repair the old arena.

## Boundary and shared continuation

Let `a` be the sum of all fourteen small pits. The sidecar changes evaluation
only for `25 <= a <= 74`. Deposits into stores and captures cannot increase `a`,
so after entering `a <= 24` the trajectory cannot return to the treatment band.
The literal rules must verify that invariant between complete decision states
on every recorded move transition. A relay's temporary seeds-in-hand arrays
are not complete 98-seed positions and do not define the switch or ownership.
The ownership domain is deliberately **all `a >= 25`**, including search states
above 74; a higher root can search into the changed band. Arena roots must be
inside 25–74. Positions with 75–98 active seeds have no opening-strength claim.

At the first normalized played state with `a <= 24`, durably commit a handoff
record before making another move. If the boundary-crossing move terminates the
game, retain the boundary event and terminal result without a search. Otherwise
the learned seat uses frozen control PUCT and the opposing seat uses its
assigned H_v1 or H8 relay CPU engine until terminal. The continuation engines
depend only on the baseline and seat assignment, never on candidate/control
arm, prefix outcomes, margins, model predictions or family. The two arms may
reach different states; shared continuation means the same state-to-decision
procedure, rather than forcing equal suffix moves or scores.

Clear all trees, transposition tables, evaluator and feature caches, history,
ordering heuristics, root-policy reuse, statistics and inherited RNG state at the handoff and at every
decision in both prefix and suffix. Neither arm can supply a prefix-derived
value, visit, action order or root prior to the suffix. A fresh decision-local
TT is allowed for each baseline under its declared 16-MiB bound; it cannot
survive a decision. PUCT has cache capacity zero, no root reuse and no root
noise. No cross-game, cross-family or cross-arm inference cache is allowed.
Keys for any decision-local cache must bind raw state, controller, rules,
complete model/export, encoder, numeric backend and search version. A quotient
is an ownership key, never a score or model-cache key.

Use the existing AVX2 backend and fixed control export. When a prefix search
reaches a leaf at `a <= 24`, the candidate uses the frozen original value path
directly, as control does. It must not consult candidate branch parameters or
an earlier candidate low-state cache. The suffix switch is still required:
equal leaf functions alone do not erase a reused prefix tree. For the same
low-state input at fixed work, candidate-bypass and control must match actions,
visits, values, features and physical search counts within the existing exact
backend contract. Timed decisions can differ because of scheduling; no timed
bit-equality claim is made.

Every whole decision has the same 25-ms budget in prefix and suffix, including
reset, extraction, inference, ownership trace capture, validation and search.
Only the positive remaining search budget may be passed to a worker; zero
must never mean unlimited. Retain overshoot measurements. Keep the existing
PUCT `c_puct=0.25`, 100,000-backup and 2,048-node ceilings, fixed FPU/backup/tie
rules, and baseline depth ceiling 16 with TT/PVS. H8 relay CPU's name does not
promise a depth-eight completed timed search. Exact K<=4 checks are a separate
regression guard. **K=24 is not an exact oracle or solved tablebase.**
Move selection uses no random draws: retain the pinned deterministic tie order
and disabled root noise. Any future stochastic search would need a separately
declared arm-independent seed rule before play; inherited prefix RNG state is
not a permitted substitute. Optional post-run bootstrap uses its own RNG and
is subject to the separate sampling gate below.

## Two provenance layers

Keep the old full-origin graph and every exclusion immutable. Construct a new
versioned graph for the future experiment from source-certified incidence
witnesses, not by modifying the old graph. A witness records origin ID, raw
state/controller, source hash and location, reason/role, active seeds, canonical
quotient and enough saved lineage to identify its origin. Record acquisition
and analysis exposure separately from optimizer use.

For the prospective learned-domain graph retain every origin vertex and every
known quotient with `a >= 25`; add an origin-to-quotient edge only for those
high-domain incidences. Multiple high-domain states of the same source origin
remain connected through that origin, even if a low state also occurred in its
lineage. Remove only the low-domain quotient incidences from this new graph.
Never discard an entire historical origin because it has a low-state alias,
and never split a high-domain component merely because it is too large.
Retain low incidences in a separate shared-domain ledger with their complete
original component references. Excluding them from prospective learned-domain
connectivity does not make them unknown, delete them, or admit them to gradients.

The canonical quotient remains the established store-free, player-swap
canonicalization of pit arrays and controller. It protects against store-offset
and actor-swapped aliases. Retain raw stores/controller for scores and replay;
quotient equality does not transfer a teacher target or prove equal raw states.
Active-seed count is recoverable from the quotient and must agree with its
witness. An opaque historical union lacking recoverable incidence witnesses
cannot be safely split: conservatively retain its high-domain connections or
hold admission until its provenance is reconstructed. A low-only tombstone may
be classified only when its quotient and seeds are known; it cannot erase a
separately known high-domain exclusion.

Freeze the complete known protected union before selecting future roots:

* All 512 current optimizer-support origins and all their high-domain aliases,
  whether or not they were drawn, plus any other explicitly admitted optimizer
  support for the future experiment. No larger historical pool silently becomes
  eligible for optimization.
* All current 128 fit-check origins, reserved evaluation/development origins,
  accepted case/arena/forensic families, exposed diagnostic roots, and every
  existing exclusion/reservation. For a protected origin include all its known
  high-domain aliases, rather than just the selected optimizer or inspected root.
* Every known state used by the proposed new evaluation's starts, saved source
  prefixes or development generation. A new uncommitted generation attempt is
  retained and exposure-tagged; a failed attempt does not become a replacement
  chosen after viewing an outcome.

Historical ancestral corpus exposure is a distinct field. The base network
learned from old data, and historical internal search leaves were not all
recorded. A reconstructed known-state graph cannot prove absence of those
unrecorded leaves or remove ancestral training. New reservations mean
**excluded from the next update and from known exposed ownership components**;
they do not establish historically unseen positions. Missing material witness
data, an unverifiable source, or an unresolvable high-domain union holds
admission. Broader ancestry limitations remain explicit even after admission.

## Future start reservations and observation coverage

Admit 24 fresh-to-this-evaluation origin components in the new high-domain
graph, disjoint from each other and from the complete frozen protected union.
Select eight ordinary, eight capture and eight bonus roots, at least two legal
choices each. Within each mechanism select four roots per actor. Ordinary roots
have four at 25–49 active and four at 50–74, two per actor in each band; capture
and bonus roots have 25–49 active. Define mechanisms with the pinned existing
metadata classifier, rather than retrospectively relabeling starts. Use a
fixed salted hash order, salt `ownership-protocol-v1/arena/20261016`, before
targets, model predictions, searches or outcomes are inspected. Freeze all IDs,
raw states, strata, source paths/hashes and rejection reasons jointly. A
shortage holds; do not shrink the quota, add fillers, change the salt, or use
known good starts after outcomes. This protocol does not assert current
feasibility or reserve actual starts.

The development population is this metadata-defined mixture, weighted equally
by mechanism and by admitted origin. If a start is synthetic without a saved
legal opening prefix, report that property and make no natural-opening or
standard-opening-reachability claim. A future natural-play experiment needs its
own sampling frame and saved legal lineages. Do not call hash selection a random
sample of all possible Congklak positions.

Shared public openings and saved prefixes remain real provenance. If proposed
natural starts descend from the same source episode, known common >=75 state,
or other high-domain ancestor, keep that connection: labeling later roots as
24 new families cannot make them independent. Do not discard a public-opening
edge to satisfy the quota. A reachable-origin sampler might therefore fail the
24-component requirement; this plan does not promise otherwise. A different
dependency analysis or synthetic-position population would need its own
prospectively audited design, rather than being substituted after results.

For all future engines, retain states consulted in played transitions,
normalization/pass handling, tree expansion, evaluation, TT probes/hits and
relay-feature simulations. Distinguish played states from internal search
states. Ownership events need raw state or lossless reconstructable raw state,
quotient, active seeds, game/decision ID, source engine and operation. Trace
capture and hashing are charged to whole-decision time. At preflight verify
trace completeness with instrumented deterministic fixtures. Missing or dropped
events, trace quota exhaustion, incomplete TT/feature coverage, or an
unattested native path blocks admission or holds the run. This may require new
instrumentation; the old arena is not evidence that such tracing exists.
Apply the ownership quotient only to complete valid decision states with all
98 seeds accounted for in pits and stores. For a relay operation retain its
complete input/output states, source decision and operation path; preserve
intermediate events when needed to verify the simulation, but do not feed a
temporary seeds-in-hand array to the complete-state canonicalizer or interpret
its pit sum as a domain exit and reentry.

After every committed decision, incrementally check all observed `a >= 25`
states against protected roles and every other arena origin, including aliases
linked through a historical high-domain origin. Different profiles, seats and
baselines within one reserved origin are allowed to meet; different reserved
origins are not. Any direct or transitive cross-origin or cross-role high-domain
collision holds strength interpretation for the whole fixed run. Preserve its
committed prefix and incident paths; do not filter games, replace starts,
merge clusters, change the cutoff, or rebootstrap the survivors after outcomes.
An interrupted run cannot resume production past an unresolved integrity hold.

Low-domain sharing is logged and permitted only when the frozen continuation,
leaf bypass and cache/reset contracts pass. It can connect support, fit and
arena origins without merging the prospective statistical units. This is a
declared shared function with no cross-game learned state, not an argument that
transpositions generally create independent samples. A low-state branch call,
mutable shared cache, reused prefix tree or suffix arm-dependent engine creates
a hold even when no high-domain collision occurred.

## Outcomes and claims

Run candidate and control against H_v1 and H8 relay CPU from every admitted
origin, in both seats: 24 origins x 2 profiles x 2 baselines x 2 seats = 192
terminal games. Freeze a rotated schedule and deterministic ties before play.
All eight games from an origin form one unit; games, decisions, suffix states,
seats and backups are not independent replications. Use one exclusive worker
on the verified performance core, group 0/logical 2. Match backend, wall-clock
budget, process reset and instrumentation across arms; record OS/runtime/build,
core, affinity, resource leases, whole-choice timings and physical work.

Let `M[f,p,b,s]` be terminal stores of the learned seat minus terminal stores
of its opponent. For each origin `f`, profile `p` and baseline `b`, set
`Y[f,p,b]=(M[f,p,b,0]+M[f,p,b,1])/2` and
`Delta[f,b]=Y[f,candidate,b]-Y[f,control,b]`. Average each quantity over the
24 origins, keeping all origins and all paired seats. The primary development
summaries are the two mean paired `Delta` contrasts. Four absolute margins are
secondary descriptive comparisons of the composite engines. Report per-cell,
per-origin, handoff-seed, suffix-length and timing summaries without choosing
a favorable subset as the main result.

With no declared probability sample from a wider target population, these means
describe the fixed admitted panel. **Uncertainty intervals are disabled by
default.** Before an eventual production run, a separate audit would have to
accept an explicit sampling frame, selection process, independence/dependence
and within-stratum exchangeability assumptions to enable exploratory
resampling. Graph separation alone does not supply those assumptions. There is
no claimed probability distribution over all positions or over hardware noise.

If that prospective additional gate passes, the declared sensitivity method is
4,096 stratified whole-origin bootstrap resamples, seed `20261019`: sample eight
origins with replacement inside each of the three mechanism strata, retain all
eight games of each sampled origin, and use the same resampling indices for all
six statistics. The mechanism weights remain one third. Report 2.5/97.5
percentile endpoints, the accepted sampling assumptions, exact resampling
algorithm and 24-origin denominator. Call these exploratory intervals under
those assumptions, not a calibrated confidence guarantee for every Congklak
position or independent-game evidence. The definition-stage acceptance does
not enable that gate. Without it, report unavailable intervals with the finite
panel's descriptive means. With 24 origins precision remains limited, and no
multiple-comparison significance or broad dominance claim is made from six
unadjusted exploratory intervals.

The complete fixed run must reach 192 terminal results with no fallback,
512-choice cap, source drift, missing trace, policy change, resource overrun,
high-domain collision or suffix-contract failure, and the separate sampling
audit must pass, before exploratory intervals are emitted. If any gate fails, retain all results and
report descriptive means with the hold. No intervals on a post hoc valid
subset. A failure to beat a baseline is accepted negative evidence rather than
an engineering failure. Neither positive intervals nor nonnegative means
automatically promote, enlarge data, select another checkpoint or train again.
Both baseline contrasts must be reported regardless of their signs.

This is a future **known-provenance, update-disjoint development protocol**.
It supports a bounded comparison of prefix treatments under common frozen
continuations. It cannot claim the final product solves Congklak, the raw
candidate beats each baseline without the switch, the existing final400 model
has passed its failed fit gate, or historically unseen generalization.

## Value metrics and endgame guard

Keep three roles distinct. Training roots measure fit to admitted optimization
support. The existing 128 fit roots and all inspected worst cases are consumed
development diagnostics, useful for explaining errors but not fresh validation.
A future update holdout must be reserved before the next training treatment,
share no high-domain component with that treatment's full support, and remain
uninspected until its declared check. Its ancestry and prior exposure must still
be reported. This plan creates no new training support or holdout IDs.

For a later authorized value-only treatment, freeze one target definition and
the candidate endpoint, regularization, update/draw budget and sampling before
optimization. The existing complete H8 player-0 full-root-Q convention remains
the reference until a separately audited change is declared. Do not mix H8,
deeper partial targets and game returns according to which favors a model.
Report normalized Huber and seed MAE/RMSE/bias by actor, stratum and capture;
root scalar error and best-action regret remain different measurements. Case
regret uses full saved legal-action Q and optimal ties. Any fit/regret gate for
a new experiment must be frozen in its treatment registration; this ownership
protocol does not select that treatment or relax the old failed gates.

Retain the known 32-root exact K<=4 guard at both 25 and 100 ms for candidate,
control and both baselines, including all optimal ties and forced-move counts.
Require no candidate suboptimal move or fallback. Also run fixed-work low-band
conformance including 0,4,5,24 and boundary 25, with cache/reset tests and actor,
bonus, capture and pass cases. Require shared-suffix action/visit/value equality
at fixed work where the same engine/settings are invoked. These are regression
fixtures; reused exact cases are not a new unseen release panel and do not
prove every 24-seed position is optimal.

## Recovery, immutable evidence and implementation order

Before an eventual run freeze protocol, complete model identities, source/build
hashes, graph witness catalog, roles, starts, schedule, runtime/search/budget
settings and all permitted continuation engines. Version sources instead of
editing accepted ones. Durable authoritative state comprises immutable raw
decisions/games/handoff/ownership events and an ordered hash-bound catalog.
Fsync raw units before publishing their catalog commit. Retain uncommitted
orphans, resume only the validated committed prefix, and never search a
committed decision twice. Handoff and ownership-check completion are part of
the cursor; a resumed process cannot forget that the switch occurred or bypass
a pending collision check. Missing/corrupt evidence and incompatible model or
protocol identities fail before a writer or engine is acquired.

Require real process kills before and after a handoff commit, a decision commit
and final certificate publication. Fixed-work fresh/resumed fixtures must match
logical results and all committed prefix hashes, sizes and mtimes exactly;
timed future choices may vary and must retain their real clocks. Completed
API/CLI readers validate immutable certificates before locks, writers, model
forward, search or optimization. Test those forbidden entrypoints with traps.
Run actual recovery of the protocol-build journal in this definition stage;
production handoff/search recovery is a future admission requirement, not an
already executed game test.

Retain the existing 4-GiB private-commit, 5-GiB new-namespace storage and
90-minute cumulative arena limits; charge fsynced invocation allowances and all
recovery attempts, including unclean allowances. Include trace storage and
instrumentation work in those bounds. If complete trace evidence cannot fit,
hold rather than dropping states or increasing a cap mid-run. Observers read
immutable evidence or enumerate provisional filenames, not worker files being
atomically replaced. Preserve the Windows pending-rename and watchdog fixes.

Implement in this order, under a separate execution instruction after this
definition is accepted:

1. Build the witnessed high-domain graph and all-role exclusion report from
   existing sources. Prove the classifier keeps high-domain ancestry and does
   not split opaque unions or claim historical leaves that were never saved.
2. Implement and audit full engine trace coverage, per-decision reset and the
   explicit two-seat handoff; run only separately accounted engineering
   fixtures, fixed-work conformance and real interruption tests first.
3. Check actual start/holdout feasibility, reserve all future roles and freeze
   the evaluation identity before any new treatment or outcomes. A shortage or
   uncertain exclusion is an admission hold.
4. Register one small value-only treatment and its fit/regret gates separately.
   Training and new games need their own instruction; this plan grants neither.
5. Only after admission, execute the registered treatment and protected checks,
   then the full both-baseline arena. Independently verify the entire closure
   and report all positive and negative results before considering another run.

The dashboard separates completed protocol/design/recovery work from pending
graph construction, native trace/handoff implementation, reservations, treatment,
fit/endgame and the planned 192 games. Current training/search/game counters
remain zero; historical 400 updates and 192 games remain history. A passed
protocol definition means ready to implement admission, not ready for training
or product release.

## External method references

The Center for Open Science recommends specifying planned analyses before a
study and distinguishing them from analyses informed by existing outcomes.
The old arena cannot become prospective by changing its rule after seeing its
bridges. This new protocol is explicitly informed by those diagnostics and
applies to future identities only.
[Center for Open Science: Preregistration](https://www.cos.io/initiatives/prereg)

Saravanan, Berman and Sober explain the error from treating lower-level repeated
measurements as independent and study hierarchical resampling. Their result
motivates preserving whole origin bundles here; it does not validate this
engine's deterministic origin sampling or calibrate its intervals. The
stratified panel-composition interpretation above is our conservative design
choice, not a theorem supplied by that paper.
[Application of the hierarchical bootstrap to multi-level data in neuroscience](https://arxiv.org/abs/2007.07797)
