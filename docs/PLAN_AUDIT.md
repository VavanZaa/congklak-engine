# Initial architecture audit record

This records the initial architecture review before the expanded database, weighting and pruning plan. Its open decisions and findings reflect that earlier draft. The [full updated audit](UPDATED_PLAN_AUDIT.md) and [first implementation scope](FIRST_IMPLEMENTATION.md) supersede its implementation gates.

Audit date: 14 October 2026 (Asia/Bangkok). Scope: research and proposed architecture for Indonesian Congklak/Dakon on the confirmed i9-13900K, 32 GB RAM and RTX 4080 with 16 GB VRAM. The first deliverable is position analysis with explanations and interruption recovery.

**Verdict: the revised plan is a sound, conditional architecture for this goal.** Close the explicit Stage 0 rule decisions, then implement the reference simulator and resumable CPU analyzer. No unresolved architectural defect found in this review requires replacing that sequence. This verdict does not verify engine code, playing strength, energy savings, crash resilience or a new game solution. No engine or benchmark was implemented or run during this audit.

Reviewed artifacts:

- [README](../README.md)
- [Implementation plan](CONGKLAK_ENGINE_PLAN.md)
- [Rules report](research/congklak-rules.md)
- [Engine comparison](research/engine-comparison.md)
- [Compute and recovery contract](research/compute-and-recovery.md)

No applicable `AGENTS.md` was found in the workspace or its ancestor directories during the audit. Primary-source checks were independent spot checks, not an exhaustive audit of every cited paper or repository file.

## Findings and corrections

Severity below describes the consequence if a requirement is bypassed. An unresolved rule choice is an implementation gate, not evidence that the research has failed.

| ID | Severity | Finding and concrete correction | Current status |
| --- | --- | --- | --- |
| A1 | High | The regional game name does not resolve capture, depleted-row handling, opening or refill semantics. Freeze a named contract and approve edge-case traces before generating reusable engine results. | Open Stage 0 decision; correctly exposed in the plan. |
| A2 | High | A deeper heuristic TT entry need not equal a shallower minimax value. Every published `depth_result` must match remaining horizon, evaluator and extension semantics. Ordinary greater-depth reuse belongs to a declared estimate policy. | Addressed in the revised CPU-search contract. |
| A3 | High | A completed PVS root iteration can leave alternative moves with bounds rather than exact scores. Preserve per-move bounds or re-search before reporting a score gap or a complete exact ranking. | Addressed in MultiPV and result publication. |
| A4 | High | A certificate verifier that trusts the producer's child list can accept a proof omitting a legal reply. Regenerate actions, transitions and terminal payoffs and independently check every required dependency and coverage claim. | Addressed in the revised certificate contract; implementation remains to be tested. |
| A5 | Medium | Sampler/RNG state alone can miss queued, unconsumed training batches. Drain prefetch or persist pending batch identities with the corresponding sampler progression. | Addressed in both recovery documents. |
| A6 | Low | Outcome-only proof and constructive start-position solving need distinct terminology. Use ultra-weak, weak and strong scope explicitly, including any restricted state domain. | Corrected during review; see the terminology source below. |
| A7 | Low | The same Dakon source was summarized repeatedly across artifacts. Consolidate its details in the rules report; treat the two Kalah URLs as one underlying source for attribution limits. | Consolidation applied; avoid expanding these summaries in delivery prose. |

For A6, a weak solution should describe strategies achieving the proven game-theoretic value against legal opposition within reasonable resources, including an initially losing value. The distinction follows the primary [Schaeffer and Lake account, printed p. 123](https://library.slmath.org/books/Book29/files/schaeffer.pdf). It changes project claims, not the recommended search architecture.

## Rules and state semantics

The proposed Betawi profile is correctly identified as an engineering completion of documentary omissions. Its route diagram resolves otherwise ambiguous direction words. The fixed ring, opponent-store skipping, source revisits, opposite mapping, occupied-pit relay, bonus choices and capture ordering have explicit fixtures. Manual inspection found the displayed synthetic arithmetic internally consistent; these fixtures have not yet been executed by a simulator.

The forced-pass/revival fixture is essential: a depleted row under the proposed both-empty ending can receive seeds again. An immediate sweep profile has different transitions and values. End checks must occur after a complete deterministic action, not while seeds are temporarily in hand. Capture with an empty opposite, capture of the landing seed, eligibility gates and their reset scope remain user-facing contract choices.

The regional Java, Malaysian and simultaneous profiles are appropriately separated. Cultural descriptions with contradictory paragraphs are not combined into one algorithm. A simultaneous opening needs a defined joint transition or physical interleaving protocol; selecting two pits and sequentially applying full moves is not automatically a faithful model. Refill, closed pits and protected remainder require richer state and separate termination analysis.

The D1 source and numerical transcription were spot-checked. Replay remains an implementation gate; the [rules report](research/congklak-rules.md) carries the source-specific conditions and scope.

Semantic identity is sufficiently specified: counts, controller, phase, rule contract, objective and all relevant eligibility, round or repetition state. Proof-relevant equality requires canonical-state comparison or a collision-free rank; a larger fingerprint is not equality proof. An immutable contract manifest must resolve and validate the rule digest. Mid-sowing imports need hand and cursor state or must be rejected. Store omission and symmetry reduction remain proof obligations rather than assumed optimizations.

## Search bounds and explanation correctness

Fixed-player reference recursion is a sensible initial implementation. Extra choices preserve the maximizing/minimizing actor; a negamax score and window reverse only when the actual controller changes. Normalized passes can return control to the same actor. This rule applies to PUCT and proof-number AND/OR semantics as well as alpha-beta.

The plan now separates heuristic estimates, exact finite-horizon results, sound eventual-outcome intervals and solved results. For a maximizing root, a move is certified optimal when its sound eventual lower bound is at least every alternative's eventual upper bound; equality permits ties. A winning bound alone does not certify maximum margin. Tablebase bounds must not silently become a different finite-horizon leaf convention.

The default objective is explicit single-round margin, with WDL derived from its sign. WDL-only tie breaking and match utility are separate configurations. Learned heads must target the chosen utility; a weighted mixture would change what “best” means.

Choice depth alone can end inside a long bonus chain before examining an opponent reply. Reporting completed turns and opponent-reply coverage makes that limitation visible. Automatic relay work also needs cancellation checks and work limits. A safety guard must return error/unknown when rule semantics are unresolved; it cannot invent a draw.

Explanations use traces and searched continuations, with truncation and per-candidate coverage. A principal line supports a conditional continuation, not every strategic claim. Score-gap explanations require comparable horizons and completed score/bound treatment. These requirements fit the first product goal without depending on a language model to invent reasons.

## Termination tablebases and proof trust

The proposed termination ranking is defensible for the base sequential A/B profiles under their stated assumptions. This is an independent conditional argument, not a verified implementation theorem:

1. Let `R` be seeds outside permanent stores. Any store deposit or capture reduces `R`; no transition increases it.
2. If a completed own-row choice makes no store deposit, it cannot pass the mover's store to reach the opponent row. Every sow and relay advances seeds within the mover's row. The sum of seed counts weighted by distance to their owner's store therefore strictly decreases.
3. At stable choice boundaries, lexicographic `(R, Phi)` decreases on each effective choice. Forced passes are normalized administrative transitions; both rows empty is terminal.
4. Inside a relay action, use distance to the **fixed mover's** store at pickup boundaries. A no-store lap strictly reduces that potential. Store-depositing laps reduce `R`, so infinitely many laps are impossible with a finite seed total.

Validate this argument against the frozen kernel and exhaustive small instances before relying on it. Owner-distance alone is insufficient for an in-action relay that has already crossed into the opponent row. This argument does not cover refill, altered closed/protected routes or physical simultaneous play.

Active-seed layers can have same-layer edges even when the full base graph terminates. The plan correctly requires an internal topological ranking or full dependency/SCC treatment. For a cyclic profile, legal infinite-play adjudication must precede solving; an arbitrary fixed point or move cap is insufficient.

The combinatorial configuration counts in the plan were independently recalculated using integer binomial arithmetic and agree. They are upper bounds on count allocations, not measurements of reachable states or search difficulty. Small tablebases are reasonable initial targets; full seven-by-seven strong solving remains a measured feasibility decision.

The verifier contract now checks legal replies and transitions independently of search heuristics. MAX lower bounds can use a witnessing child while MAX upper bounds require all legal children; MIN reverses those obligations. Unknown or absent coverage remains unknown. Checksums establish artifact integrity, not the truth of a value. Independent interpretation and complete recurrence/coverage checks reduce the trusted code; this plan does not claim formal verification of that code.

## Source claims and attribution review

The official GitHub API independently confirmed the pinned Stockfish, trainer and Lc0 commit identities and dates recorded in the engine report. Inspection of the pinned [Stockfish architecture](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/nnue/nnue_architecture.h), accumulator and TT sources agrees with the report's snapshot distinctions. Suggested count embeddings and tiny networks are proposed Congklak designs, not descriptions of Stockfish's current network.

The [Lc0 overview](https://lczero.org/dev/overview/) supports the search/backend separation used in the recommendation. Indexed developer material for Dragon and the [Torch team announcement](https://www.chess.com/news/view/torch-chess-engine) support their disclosed features; hidden layer sizes, pruning schedules and coupling remain unknown. Chess results do not demonstrate a Congklak speed, strength or energy advantage. The recommendation consistently identifies this boundary.

Primary Kalah and MTD(f) materials support the techniques attributed to them; they do not establish relay-Congklak complexity or a guaranteed MTD(f) gain. The plan correctly proposes experiments with a shared simulator, objective and horizon. Recent completion work is labeled a preprint and is not made an implementation dependency.

Attribution was reviewed across this documentation set rather than independently per file. The highest repetition risk, D1, was reduced during review. The AlphaZero fact blocks are approximately 177 whitespace words even when their explanatory cautions are counted. The combined Kalah summaries remain below the default 200-word allowance when both URLs are treated as one paper. D1 remains within a concise source-derived allocation after excluding original simulator proposals and verification instructions and counting its numerical witness. There are no long prose quotations; titles, identifiers and the short numerical witness are the principal verbatim material. This is a conservative editorial check, not an automated attribution or copyright certification. Keep final delivery concise and link to the reports instead of restating source facts.

## Compute recovery and implementation gates

The proposed 2 GiB starting TT, 24 GiB aggregate engine RAM cap, 12 GiB initial GPU cap and 50 GiB disk quota are coherent starting budgets, not measured capacity requirements. Resident mapped pages, build dependencies, worker stacks and concurrent jobs must count toward the cap. Disk capacity and bytes per shard/game must be measured before an overnight campaign. A large final tablebase can require substantially more construction memory than its stored values.

Recovery distinguishes durable completed work from disposable caches. Immutable files, flush/verification, transactional manifests and retention of a prior valid generation avoid mixing components from different steps. [SQLite's synchronization documentation](https://www.sqlite.org/pragma.html#synchronous) supports the metadata proposal; it does not prove Windows bulk-file durability. The plan requires fault injection before making that claim, including full disk and incompatible versions.

The updated training contract captures prefetched work. Single-worker analysis conformance compares final values, deterministic choices and traces; elapsed time and work counters may differ after recomputation. Bitwise training continuation is restricted to a supported deterministic configuration. Parallel jobs promise a valid resumed generation unless stronger reproducibility is demonstrated.

Energy claims are appropriately prospective. Equal time and equal measured energy answer different questions. CPU/GPU telemetry is identified by component scope, whole-computer claims require suitable measurement, and learning cost includes data generation and training. No energy advantage has been measured.

The milestone sequence is realistic as dependency gates: rules, simulator, resumable analyzer, optimized CPU search, exact endgames, then learned/GPU challengers and larger proof campaigns. It makes no unsupported duration or full-solution promise. The first release must remain useful for unsolved positions and honestly label its estimates.

## Decision and validation checklist

Addressed in the revised plan:

- [x] Current architecture facts separated from proposed adaptations and measurements.
- [x] Extra-turn/controller semantics, horizon TT scope and per-root PVS bounds explicit.
- [x] Full semantic identity, collision tests and conditional store/symmetry reduction specified.
- [x] Same-layer tablebase dependencies and independent certificate obligations specified.
- [x] Margin, WDL and match objectives separated; long-turn horizon limits visible.
- [x] RAM/VRAM/disk caps, coherent checkpoints and prefetched-batch recovery specified.
- [x] Scope of source witnesses, proof claims and energy comparisons constrained.

Remaining decisions and implementation gates:

- [ ] Select the user's local profile and approve route, capture, eligibility, depleted-row and round fixtures.
- [ ] Resolve any opening/history or protected-pit fields required by that profile.
- [ ] Freeze the rule contract, objective, state encoding and golden traces.
- [ ] Choose initial CLI/editor scope, data location and project licensing.
- [ ] Implement and run differential simulation, extra-turn search, forced-collision and MultiPV checks.
- [ ] Execute cancellation/restart and checkpoint fault-injection tests.
- [ ] Measure quality, latency, memory, disk and energy before promoting optimizations or learning.
- [ ] Independently verify tablebase coverage and certificates before using proven/solved labels.

The research and plan are accepted with these explicit gates. Engine correctness and operational recovery remain future acceptance results.
