# Bounded midgame collection and H8-loss target study

This version collects a small, separately owned natural-history slice and investigates the saved H8-relay divergence states. The control and value-pilot models, their shared policy, all prior artifacts and the H_v1 default remain fixed. Collection and teacher searches are authorized; training, strength games, model expansion, source-data deletion and promotion are outside this run. The baseline preserves 2,639 inputs: the 2,637 sealed review inputs plus its acceptance and completed-rerun proof.

The fresh pool consists of 512 deterministic standard-opening history prefixes, seed 20261023, at most 192 choices per prefix. Half use random legal moves and half the unchanged control policy with epsilon 0.25 after eight random opening choices. A prefix is not necessarily a finished game. Families receive training-candidate or validation ownership before any root is selected or labeled. Shared history prefixes and assigned family IDs do not establish statistical independence.

| Eligible stratum | Training-candidate cap | Validation cap |
| --- | ---: | ---: |
| 50–74 active seeds, strict ordinary | 16 | 4 |
| 25–49 active seeds, strict ordinary | 8 | 2 |
| 25–49 active seeds, capture | 8 | 2 |
| 25–49 active seeds, bonus | 8 | 2 |
| Maximum fresh roots | 40 | 10 |

Keep the existing mechanism classifier and require at least two legal actions. Strict ordinary means every legal move avoids the classifier's capture, bonus and pass mechanisms. No artificial root, relaxed classifier, replacement quota or automatic pool expansion fills a shortfall. Selection is deterministic over the complete pool, with one root per family/target stratum and no duplicate quotient.

Before selection, extend the previous protection ledger with all 144 latest pilot-game trajectories and reviewed evaluation metadata. Preserve the 83 prior curated historical roles (61 training, 14 validation, eight diagnostics), original development ownership including 259 ordinary roots, prior protected roots/families/quotients and generated family IDs. Reserve old curated roots against duplicate fresh collection. Latest evaluation protection overrides future admission when it conflicts with a historical training role; record the collision without rewriting historical cohorts or claiming that earlier exposure is undone. Read sealed-test metadata only where needed for exclusion, never its targets.

Closure operates on the bipartite graph of new families and their declared eligible target-candidate quotients. Drop an entire connected component if it touches a protected quotient/family or contains both ownerships. This includes multihop bridges. Opening-prefix states outside the declared candidate strata do not contaminate every new family through their common start. Close ownership before quota selection; retain exclusion reasons and honest yield. The ledger does not certify that the fixed models never saw related historical states.

The diagnostic panel is fixed at the nine distinct first-divergent states in the saved loss review: all eight without matching stored Q and the sole covered worst-loss state. Their active counts are 8, 11, 15, 30, 34, 39, 47, 53 and 66. Report their actual mechanisms and bands; they are not all ordinary midgame positions. They remain diagnostic-only regardless of original source-family ownership. Preserve the original covered H8 item unchanged and compare its same-depth Q semantics explicitly.

| Cohort | Requested horizons | Node cap per root/attempt |
| --- | --- | --- |
| All selected fresh roots | H8, H10 | 2 million, 5 million |
| All nine diagnostic states | H8, H10, H12 | 2 million, 5 million, 10 million |

The maximum requested production work is 503 million nodes: 50×7 million fresh plus 9×17 million diagnostic. There is no H14 or automatic escalation in this scope. Use the existing oracle-free H_v1 full-root choice-horizon teacher, TT 16 MiB, PVS, memory 512 MiB and no wall-clock cutoff. Node limits describe the declared teacher work, not a runtime or energy promise. Lower completed depths are retained as auxiliary history. Only a fully completed requested depth with every legal action value becomes a label. Preserve capped attempts as capped, and report missing paired denominators. Finite-horizon labels are estimates, including deeper ones.

Keep this pilot separate from optimizer data. Compare fixed control/pilot value errors and action preferences only on matching states with available complete targets. Review H8/H10/H12 target movement, best-set overlap and saved-action regret without attributing a causal game blunder. Any sampling comparison is an offline proposal over explicitly eligible training candidates, with validation/diagnostics excluded, not a new optimizer distribution or calibrated correctness probability. Preserve rare and endgame data.

Freeze source identities, settings, rules, registry and roots before production. Single-writer SQLite FULL/WAL transactions commit each history prefix and each requested labeling attempt with its optional complete label. Interrupted uncommitted units restart; committed units remain unchanged. Actual subprocess kills before and after collection and label commits must preserve prefixes and resume correctly. Completed producer reruns must perform zero teacher searches and retain published bytes and modification times. Refuse corruption, incompatible identities, incomplete full-label claims and fixture settings in production.

Author checks and an independent audit must verify source/ownership protection, literal histories, transitive closure, full-root labels and PVs, actual interruption recovery and completed no-work reruns. Update the dashboard throughout and retain failed prepublication evidence if repairs are required. Engineering completion establishes a bounded, auditable dataset; it does not establish playing strength or authorize training. A subsequent learning experiment still needs its own admitted replay set, frozen sampling objective, protected development families, both-baseline equal-time gates and endgame checks.
