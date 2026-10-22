# Independent audit of the bounded midgame pilot

All **8 independent methods passed**, with zero failures, errors or skips. The four collection-author and four label-author methods bring the current aggregate to **16 unique checks**. No unresolved material finding remains. The independent main harness executed its preflight and final groups unchanged; there was no failed independent execution, source rebinding or post-outcome repair.

The combined proof is `build/midgame-pilot-audit.json`; its executed groups are `build/midgame-pilot-audit/preflight.json` and `build/midgame-pilot-audit/final.json`. The main source is `tests/midgame_pilot_independent_audit.py`, SHA256 `b15f0104f66934489f3f1ca139c6335c32d0a64cf41fdc73c207f8a00564abf3`. The separate final arithmetic helper, `tests/midgame_pilot_metric_audit.py`, is pinned by the final group. The combined collector verifies every group input hash before publication.

Both fixed models, their shared policy and all **2,639 prior input hashes** remain unchanged. Those pins include the 2,637 previously accepted artifacts and the prior acceptance/rerun evidence. This run performs collection and bounded teacher labeling; it performs no training, model update, new strength game, data ingestion, applied sampling change, source-data pruning or promotion. H_v1 remains the default.

## Independently verified coverage

| Method | Actual coverage |
| --- | --- |
| Baseline and contract | All prior hashes, both model identities, raw float32 equality of six frozen trunk/policy tensors containing 12,807 parameters, exact production caps and settings |
| Ownership and diagnostics | Literal replay of all latest 144 evaluation games and their 4,852 choices, including terminal protection; all 83 historical curated roles, 259 ordinary-development roots, 768 previous generated family IDs and nine raw-state diagnostics |
| Literal fixture and closure | Eight fixture histories; exact RNG/policy choices and canonical candidates; protected-quotient, protected-family and mixed-ownership multihop graph fixtures |
| Collection interruption | Two actual subprocess kills, immediately before and after a family transaction, followed by resume and completed API/CLI reruns |
| Label interruption | Six actual subprocess kills: both transaction barriers for H8, H10 and H12; exact committed-prefix preservation, resumed semantic/deterministic-work comparison and completed API/CLI reruns |
| Refusal and repair | Actual capped H10 partial; forged completion, zero work, wrong history depth, TT/PVS drift, rehashed identity/ownership/reference/prefix corruption, invalid completion markers, and valid missing-export repair |
| Entire production collection | All 512 literal histories, 10,508 generation choices and 2,511 frozen-policy evaluations; every eligible graph component, quota selection and honest shortfall |
| Entire label and derived review | All 1,151 retained completed-depth reports; 7,144 literal PV paths and 36,968 PV choices; 116 native plus 116 literal feature/dense forwards; 124 requested-label comparisons, 74 complete horizon-pair comparisons, all group denominators and all three offline weight proposals |

Completed checks forbid teacher searches in both direct API and standalone CLI reruns. They retain the committed journals, published reports and existing files' bytes and modification times. Deterministic saved-history validation is allowed; it is verification of existing histories rather than new collection. No Torch import is used.

## Ownership and natural-history result

The independently reconstructed ledger contains **43,170 protected quotients and 7,837 protected families**. The latest 144 trajectories contribute 4,996 state observations including terminal states and 1,473 distinct quotients. The original 61 training-candidate, 14 validation and eight diagnostic curated roles are preserved. Seven old diagnostic roles intersect the latest evaluation metadata; no old training-role collision was found. Protection governs future admission without rewriting history or claiming that ancestral model exposure is undone.

The 512 standard-start prefixes use the predeclared seed and legal random/frozen-policy mixture. Ownership is assigned by family hash before generation or labeling: 401 families are training-candidate and 111 validation. These family IDs and shared prefixes do not establish statistical independence.

Selection first constructs the full bipartite graph of families and quotients from the four declared eligible strata, requiring at least two legal actions. It excludes every connected component touching a protected quotient, an old protected family or both ownerships. The synthetic tests verify multihop propagation in each case, rather than a one-pass filter. Shared opening states outside the eligible candidate set do not poison every family. One root per family/stratum and quotient deduplication then precede quotas.

The resulting **49 fresh roots** comprise 39 training candidates and ten validation roots. Six families are blocked by closure. The only quota shortfall is one 50–74 ordinary training candidate; no classifier relaxation, artificial filler or pool expansion occurs.

| Fresh stratum | Training candidates | Validation |
| --- | ---: | ---: |
| 50–74 ordinary | 15 / 16 cap | 4 / 4 cap |
| 25–49 ordinary | 8 / 8 cap | 2 / 2 cap |
| 25–49 bonus | 8 / 8 cap | 2 / 2 cap |
| 25–49 capture | 8 / 8 cap | 2 / 2 cap |

All nine first-divergence states remain diagnostic-only. Their original pits, stores, controller, source-family provenance and saved game actions are unchanged, with no player or store-offset transformation. Their active counts are 8, 11, 15, 30, 34, 39, 47, 53 and 66; they are not all ordinary midgame roots. The sole previously covered H8 row/label remains unchanged, and its same-depth Q/best semantics agree with new work.

## Requested labels and horizon limits

The teacher uses the frozen oracle-free H_v1 full-root choice horizon, TT 16 MiB, PVS, memory 512 MiB and no wall-clock cutoff. Every fresh root receives H8/H10 attempts; every diagnostic receives H8/H10/H12 attempts regardless of earlier results. There is no H14 or automatic escalation.

| Requested stage | Attempts | Complete labels | Capped attempts |
| --- | ---: | ---: | ---: |
| H8 | 58 | 58 | 0 |
| H10 | 58 | 58 | 0 |
| H12, diagnostics only | 9 | 8 | 1 |

The capped H12 attempt retains its completed H11 prefix as auxiliary evidence. It never becomes a requested H12 label or enters requested-H12 paired means. Fresh H12 roots are explicitly out of scope, with zero eligible or missing labels, rather than being counted as failed targets. A synthetic censored/out-of-scope summary verifies these denominator distinctions independently.

Production consumes **48,565,516 nodes** against the fixed maximum requested-attempt sum of 503 million. The 1,151 iteration records include final requested reports as well as lower completed depths; this count must not be added to the 124 requested labels as if all records were distinct training examples. Each report's legal action values, actor-oriented optimum and ties, original-root link, literal PV leaf score, TT/PVS settings, work bounds and same-depth cross-stage agreement are checked.

The fresh H8/H10 best set changes on nine of 39 training candidates and one of ten validation roots; four training changes are disjoint. Among the eight complete diagnostic H10/H12 pairs, four best sets change and two are disjoint. The ninth diagnostic remains unpaired at requested H12. These are selected-sample finite-horizon comparisons, not game-theoretic truth or a strength result.

Both models receive exactly the same state for each comparison. The audit rederives player0/actor value errors, raw-policy regret, saved-game action regret and complete-intersection means. Opponent-owned first divergences stay identified as such. Saved action regret at a particular teacher horizon does not establish a causal game blunder or isolate a value-head effect.

## Recovery evidence and its cost

The raw preflight evidence is retained under `build/midgame-pilot-audit/retained-3dcdf484316f412fa77d5c0a930e0e6c/`. Each kill has the helper's `marker.json` copied from the flushed stdout event, worker/resume logs and a physical `validated-prefix.sqlite3` backup. The helper kills as soon as that event is observed; it does not wait for the producer's later atomic marker-file rename. In these eight runs, only producer `declared-marker.json.*.tmp` files remain. They are retained partial evidence, not published declared-marker JSON. The required barrier evidence is the matching helper marker, worker-log event and physical committed prefix. Collection starts empty and retains prefixes of one/two committed families at the two barriers.

For label recovery, one independently executed same-budget fixture supplies two diagnostics plus one fresh root. Each separate kill run is explicitly seeded with real earlier-stage and first-target records from that retained fixture. The child actually executes its second target before being killed at the requested barrier. `*-seed.json` records the source journal/hash, initial seed hash and table counts. Seeded work is not claimed as fresh execution, and the historical initial seed hash is not the completed run's current hash. Precommit/postcommit backups retain one/two current-stage attempt rows respectively.

Resume preserves every committed row exactly and matches the fresh fixture's semantic results, nodes and deposits; new elapsed/peak-process observations need not equal an earlier execution. Completed API/CLI reruns preserve durable bytes and mtimes with zero teacher calls. Lost uncommitted work and all audit fixture/probe work are separate from the production node total and production cap.

The real bounded partial probe stops H10 at 3,566 nodes with completed H8 evidence. It is valid as a rejection and refused as a full H10 label. Rehashed corrupt identities, diagnostic-to-training role changes, altered old H8 references, attempt-prefix gaps and bad complete counters are refused. Invalid complete markers are checked before missing-file publication. A valid missing manifest is repaired without search while all existing inputs remain unchanged.

## Offline weights and remaining gates

The three proposals apply only to the **39 fresh training candidates with complete requested H10**, spanning 30 families: uniform root, equal nonempty target stratum and requested-horizon-stability weighting. The audit independently rederives positive normalized probabilities, root/family concentration, stratum mass and the per-root requested-depth factors. All ten fresh validation roots and nine diagnostics have zero eligibility. There is no old-replay blend, auxiliary-depth fallback, optimizer admission or applied loss weight.

The stability factor is an uncalibrated exposure heuristic. Inverse concentration is not independent-game count, gradient ESS or correctness probability. Altering draw probabilities changes the training objective unless a separate experiment explicitly defines a correction. This review selects no proposal for training.

Engineering completion establishes a bounded recoverable dataset. It does not establish a stronger engine, successful mass-data training or release readiness. A later learning experiment still requires a newly admitted role-aware replay set, an explicit frozen sampling objective, preserved rare/endgame coverage and protected equal-time tests against both baselines. The current models and policy remain fixed.

## Retained prepublication evidence

The comparison's initial summary incorrectly counted fresh H12 as missing rather than out of scope. It was corrected before label production or metric publication, followed by a fresh four-method owner run. The old source and source-bound owner proof remain at `build/midgame-label-prefreeze-denominator-v1.py` and `build/midgame-label-author-prefreeze-denominator-v1.json`; they are not extra counted methods.

The uncounted early collection source/protection probe and its exact draft harness snapshot remain at `build/midgame-pilot-audit-collection-prefreeze-probe.json` and `build/midgame-pilot-audit-collection-prefreeze-source.py`. The final eight-method proof uses the unchanged main source stated above.

The observer's initial unavailable/incomplete snapshot diagnostic is retained at `build/midgame-observer-prefreeze-incomplete-diagnostic.json`. It records its stated trace/current-source provenance; no unavailable old executable source snapshot is claimed. No producer ran under that failed observer snapshot. Current dashboard evidence is limited to backend/source checks: browser policy prevented visual verification, and no screenshot or workaround is claimed.

The root recorder's first raw-evidence precheck incorrectly assumed every producer declared-marker rename had completed before the kill. Its exact failed recorder/source diagnostic is retained through `build/midgame-recorder-prefreeze-marker-assumption-diagnostic.json`. The recorder now requires helper markers, matching worker events and physical prefixes, and checks a producer declared JSON only if it exists. This collector-assumption repair changes no executed independent check, engine, producer, model or dataset and adds no successful test count.

The separate `build/midgame-pilot-root-verification.json` also passes: ten watched production/model/report files retain their bytes, sizes and mtimes, all 2,639 prior hashes remain current, and Native/neural/PUCT search, training runtime/training and new strength decisions all have zero activations. Deterministic saved-history verification remains allowed.
