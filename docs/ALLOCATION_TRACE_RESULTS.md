# Pair 7 allocation findings

All eight fresh trace cells completed: both fixed models at 128, 256, 512 and 1,024 root simulations, totaling 3,840 simulation commits and eight root expansions. All eight semantic reports exactly match the previously accepted searches. There were 3,848 physical forwards, no cache hits, caps, terminal returns or production interruptions. The shared policy, value heads and `c_puct=.25` were unchanged.

The position is the original actor-1 `pair07-seat1`, raw key `090a03000405010409010a010400061f01`. All pits below use human numbering 1–7; stored actions use 0–6. Actor 1 prefers a lower player-0 value. Sampled PUCT means are estimates from allocated leaves, not full-action teacher Q.

| Simulations per fresh tree | Control visit winner | Candidate visit winner | Control visits, pits 1–7 | Candidate visits, pits 1–7 |
|---:|---:|---:|---|---|
| 128 | 1 | 4 | 35, 10, 1, 34, 20, 0, 28 | 31, 10, 1, 34, 20, 0, 32 |
| 256 | 7 | 7 | 49, 14, 1, 59, 25, 0, 108 | 50, 14, 1, 61, 26, 0, 104 |
| 512 | 7 | 1 | 201, 14, 1, 68, 26, 0, 202 | 295, 14, 1, 61, 26, 0, 115 |
| 1,024 | 1 | 1 | 709, 18, 1, 68, 26, 0, 202 | 807, 14, 1, 61, 26, 0, 115 |

The first root allocation difference occurs at simulation 6. Both trees have the same policy priors and root visits `[1,1,0,1,0,0,2]`. The control selects unvisited pit 5 using its parent-bootstrap FPU; the candidate selects already-visited pit 2. The exact normalized scores explain the selection:

| Model | Pit 2: sampled mean + exploration | Pit 5: FPU + exploration | Pit 7: sampled mean + exploration |
|---|---:|---:|---:|
| Control | .7459737063 | **.7465407014** | .7463835052 |
| Candidate | **.7480671886** | .7389469636 | .7401970249 |

These are actor-signed selection scores in seed-margin/98 units, not teacher values or probabilities. The root bootstrap is −66.529820 seeds for control and −65.785634 for candidate; visited means also differ because their value heads evaluated earlier leaves differently. This identifies the deterministic arithmetic behind the first allocation split, without an intervention isolating root FPU from those leaf values.

Every legal root action has a visit by simulation 16 for control and 18 for candidate. Each tree uses root FPU on six selections total; it no longer acts directly at the root afterward. FPU remains available on unvisited descendant edges. At simulation 32, depth 2, the first differing internal choice along a shared raw-state path is control pit 4 versus candidate pit 3. The controller is 1 in both trees, but local visit totals are already 0 versus 1. This is a later history-dependent allocation difference; the first root split was at simulation 6.

At 512 simulations, control chooses pit 7 by visits, 202 versus 201 for pit 1, although its sampled player-0 means favor pit 1: −58.387221 versus −55.827972 seeds. The next simulation makes the visits 202/202, so the declared lowest-pit tie break changes its vote winner to pit 1 at simulation 513. The candidate's corresponding switch occurs at simulation 332 on a 115/115 tie. A vote winner need not equal the best sampled mean, and a one-visit lead can change the reported action without establishing a new underlying action value.

All existing finite horizons remain relevant; none is selected to favor a model. Entries below are control/candidate regret in seeds against the same-state requested full-action reference:

| Simulations | H8 regret | H10 regret | H12 regret |
|---:|---:|---:|---:|
| 128 | 10 / 7 | 0 / 5 | 0 / 5 |
| 256 | 0 / 0 | 1 / 1 | 3 / 3 |
| 512 | 0 / 10 | 1 / 0 | 3 / 0 |
| 1,024 | 10 / 10 | 0 / 0 | 0 / 0 |

The complete reference best sets are H8 `{7}`, H10 `{1,5}`, H12 `{1}`; both models complete every declared simulation budget. Their disagreement limits any claim that increasing visits or matching one horizon proves the correct game move. The earlier timed game is a different workload; this fixed-work reconstruction does not establish the cause of its loss.

The journal commits startup identity and root prediction before simulation records, then saves each completed simulation atomically. Resume reconstructs internal selection and fixed-player-0 backups from saved transitions and predictions without repeating committed inference. The four source-bound owner methods passed on their first execution; independent preflight also passed the actual trace pre/post-simulation commit kills, with typed cancellation/cap and corruption checks. Independent verification and final aggregate evidence are retained in `build/allocation-audit/` and `build/allocation-audit.json`.

The production resource observation records 12.740 seconds through prepublication validation, including diagnostic persistence/replay. Subsequent completion/export, fixture/audit work and possible lost uncommitted work are outside that clock. The trace overhead is not a speed or energy benchmark. Interrupted inference counters would be recorded separately; uncommitted simulation transition/deposit totals are unavailable. This production had no interruptions.

Primary evidence is `build/allocation-trace-result.json`, `build/allocation-trace-resources.json` and `data/allocation-trace-v1/work.sqlite3`; the earlier references remain in `build/divergence-comparison.json`. Resume and verification commands are in `docs/ALLOCATION_TRACE.md`. This consumed single-position study changes no model, search parameter, training weight or default, and supplies no promotion or population-strength claim.
