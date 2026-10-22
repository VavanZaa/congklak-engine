# Ordinary self-play coverage: independent audit

All eight independent methods passed, with four producer-owner and four review-owner methods: **16 actual checks**. There is no unresolved blocking defect. Four real process kills verify the new collection and self-play transactions. The combined source-bound proof is `build/ordinary-selfplay-audit.json`.

This is a data-only experiment. Both model files, rules and native engines remain unchanged. No optimizer, teacher search, new strength match, sampling change or promotion occurred. All 3,171 prior input hashes remain current.

## Why coverage was targeted

Independent literal classification and sampler reconstruction confirm the previous replay contained 276 observations but only 23 ordinary records from 14 distinct raw states. The eight high-band records repeat four starts, all at choice-zero, receiving 504 of the earlier 25,600 draws. The 15 lower-band records cover ten states and received 2,079 draws. None were forced ordinary roots. Reweighting these records cannot create new state support.

The existing 96-game arena contains 445 ordinary observations, including 184 forced observations. Only 50 are active≥25, from 11 distinct raw states; none of those 50 are forced. Its four first-divergent ordinary states are all active 5–24 and uncovered in the indexed retained labels. The new ≥25 corpus addresses midgame support, not those four late divergences or a proven cause of game losses. Later unmatched trajectories are coverage evidence, not counterfactual action evaluations.

## Ownership and natural collection

The baseline preserves 3,168 prior accepted inputs plus acceptance, stable rerun and final dashboard evidence. The unchanged control is `data/quality-train-v1/model.json`; the earlier joint candidate is `data/selfplay-training-pilot-v1/model.json`. Their exact file hashes are checked before and after this scope.

The independent ownership audit literally replays all 96 latest arena games and 2,958 choices, including terminal states: 3,054 protected state observations. It separates true evaluation protection from collection novelty. Historical training metadata can reserve novelty without becoming blanket rollout evaluation bans. Previous validation/diagnostic roles remain intact; earlier training exposure cannot be undone.

The fixed pool contains exactly 512 legal standard-start prefixes with family ownership assigned before generation, seed 20261026, half random and half frozen-policy epsilon histories. Candidate eligibility requires strict ordinary classification across all legal continuations, at least two legal actions, and active 25–49 or 50–74. Ineligible shared opening states do not join every family into one component.

Every saved history, RNG branch, candidate, canonical state and provenance field matches an independent seed-by-seed replay. Independent breadth-first graph components match the producer’s transitive family/eligible-quotient closure. A multihop fixture confirms exclusion propagates beyond a direct collision. An independent residual-network maximum-flow calculation checks maximum feasible root count under distinct-family, distinct-quotient and quota constraints.

The closed collection has 30 roots in distinct families: 12 high and 12 low training starts, plus 3+3 validation roots. All shortfalls are zero; five families are excluded by closure. Before any rollout, the auditor verifies protection for **all 100 validation-owned families and all 38 eligible validation-pool candidate observations**, including unselected roots. Protecting only the selected six would be insufficient.

## Production replay and support

All 48 fixed-control games terminate, with 1,531 committed decisions and 718 bonus choices. All root noise, behavior RNG, raw visit targets and terminal player0 returns are independently reconstructed. Two representative roots also have full independent literal-tree PUCT replay. Fixed-player0 backups preserve bonus-turn orientation; targets are not alternated blindly by ply.

Only strict ordinary, choice-bearing, active≥25 observations enter the new replay. Full trajectories retain all other decisions for provenance. The excluded domains are 329 nonordinary active≥25, 755 active 5–24 and 385 active≤4 decisions. No game caps, protected eligible bridges or duplicate pruning occurred. A protected eligible bridge or nonterminal member would reject its entire family; no replacement or invented return is allowed.

The retained corpus has **62 observations, 38 distinct raw states and 24 origin families**:

| Actual row band | Records | Distinct states | Choice-zero records | Descendant records |
|---|---:|---:|---:|---:|
| 50–74 ordinary | 26 | 14 | 24 | 2 |
| 25–49 ordinary | 36 | 24 | 24 | 12 |
| Total | 62 | 38 | 48 | 14 |

All 24 selected starts retain both terminal-game choice-zero policy/return records: 12 high and 12 low, exceeding the predeclared eight-per-band small-pilot gate. The 3+3 validation panel is complete. Distinct descendants are reported separately from the 24 starts. An additional read-only set comparison finds no raw-state or quotient overlap between these 38 states and the earlier 14 ordinary states; this is a coverage observation, not an independence or strength claim.

The root-first global observation cap 4 is independently reconstructed, with individual returns and visit distributions preserved rather than averaged into fictitious targets. Both model files remain bit-for-bit fixed.

## Four actual interruption tests and stage durability

Actual process kills cover collection prefixes of one and two families and self-play prefixes of one and two decisions, before/after commit. Each keeps the helper marker, actual stdout event, validated SQLite prefix backup and resumed journal. Committed rows remain exact. Recomputed fixed-work searches match fresh moves, RNG, visits and returns after excluding only uncommitted clocks and their dependent hashes.

The 32-prefix collection fixture honestly has no selected high-band training root. Its four-game self-play fixture instead uses two separately declared prior accepted training metadata starts, chosen before outcomes with exact source/root hashes. This tests recovery rather than production novelty or yield. Production never substitutes these fixture starts. Rehashed corrupt identity, policy, terminal return and completion data are refused, including before missing-publication repair; a valid missing collection publication is repaired without teacher search.

Completed collection and self-play API **and CLI** reruns perform no new PUCT/teacher search, durable commit or Torch import and preserve file bytes and modification times. Revalidating saved histories and CPU policy forwards is allowed read-only work. The producer-owner’s initial WAL snapshot harness issue was retained; explicitly closing inspection connections corrected the owner harness, while producer logic stayed unchanged. It is not counted as an additional successful check.

Collection and rollouts share a database. The auditor captures an immutable SQLite backup after collection and before rollout authorization. Its historical live-file SHA is capture provenance, not a current pin after self-play tables are added. Final checks match every original family/root row to that backup and preserve the collection manifest and complete validation-protection file.

## Offline proposals and model diagnostics

Five positive, normalized unit-loss proposals are checked on combined replay, paired initial records and ordinary descendants: observation-uniform, family-uniform, raw-state-uniform, equal nonempty stratum and 0.9 raw-state/0.1 stratum. All 15 tables, per-observation probabilities/expected exposures and observation/state/family inverse concentrations are independently reconstructed. Empty support remains explicit rather than acquiring invented data. The 25,600-draw comparison scale is hypothetical; **zero new optimizer draws occurred**, and no proposal was applied or selected as a strength winner.

Both frozen models receive descriptive scalar CPU forwards against the same source-policy observed returns and saved visits. The audit independently checks 124 per-record/model forward, signed error, absolute error and legal visit-policy cross-entropy comparisons. There are 76 actual native forwards across 38 distinct states; new per-call timing is not claimed. Initial records, descendants, row-weighted and family summaries remain separate.

Observed terminal margins and noisy visits are not minimax values, optimal policies or independent validation truth. The usable-data flag only confirms this small collection’s engineering support and validation-panel requirements. It does not authorize training, justify a blind long run, establish product readiness or demonstrate stronger play. Existing consumed-development and ancestral-exposure limits remain.
