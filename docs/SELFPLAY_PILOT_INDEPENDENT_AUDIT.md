# Self-play and joint-training pilot: independent audit

All eight independent methods passed. Together with four training-owner and eight benchmark-owner methods, this scope has 20 executed checks. The audit found no unresolved blocking defect. The experiment does not establish a strength improvement, and the candidate remains experimental.

The combined proof is `build/selfplay-pilot-audit.json`. It binds four executed groups, retained raw fixtures and the independent optimizer worker. All 2,839 previous input hashes remain current. No accepted model, rule, native engine or prior evidence was changed.

## Frozen scope and pre-production feasibility correction

The initialization is the accepted current-control model at `data/quality-train-v1/model.json`, SHA `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. It has 127 inputs, width 64 and 12,872 parameters in eight tensors. The earlier value-only pilot is not the initialization.

A fixed panel of 16 distinct training families supplies 32 source-control games, 64 PUCT simulations per decision and full terminal seed margins. The optimizer uses only these self-play observations: 200 CPU steps, batch 128, fresh AdamW, all eight tensors, fixed final checkpoint. Historical H10 labels provide starting positions but no gradient targets. Policy labels are raw root visits/64, regardless of behavior temperature; value labels are actual terminal player0 seed margins transformed by signed active seeds.

Before production, the provisional four-simulation fixture exposed 68 protected-quotient bridges, all at 5–11 active seeds. Independent SQL reconstruction matched the 14-family first-bridge histogram and 326 observations at least 25 active seeds across all 16 fixed families. The unsealed optimizer domain was narrowed from active>4 to active≥25 before production. The evaluation mask, fixed starts, rollout budgets, individual terminal returns, observation cap4 and hard minimum eight admitted families were unchanged. The provisional sources, plan, docs and fixture evidence remain retained. This feasibility correction is not evidence of training benefit.

## Ownership and search semantics

The independently reconstructed role ledger preserves all 9,000 historical root reservations as metadata while keeping 8,000 original training rows distinct from evaluation bans. It protects the current validation and diagnostic roots, previous evaluation trajectories and the reserved benchmark families/quotients. The audit independently reproduces the exact 16-family quota matching. Ancestral exposure is not erased by a later exclusion ledger.

Literal dictionary PUCT checks use seed-by-seed Betawi transitions, both actor orientations, bonus turns, parent-bootstrap first-play urgency, root Dirichlet noise and fixed-player0 backups. Each node changes selection perspective according to its normalized actor; backup does not blindly alternate signs.

The production audit replays all 32 terminal games and 1,029 decisions, including 453 bonus choices. It independently reconstructs source-model root priors, noise, behavior RNG and legal actions at every root. Two production roots also receive complete independent literal-tree search replay. The admitted corpus has 276 observations, 231 distinct raw states and all 16 families, with no eligible protected bridge, capped game or duplicate pruning. The omitted domain contains 306 active≤4 and 447 active5–24 decisions. Every retained value target links to its actual terminal game; every policy target links to saved root visits.

The separate CPU worker recreates the network, legal policy cross-entropy, Huber objective, family-then-row sampler and all 200 updates. Every actual checkpoint at steps20–200 matches the independently replayed model, eight AdamW entries, sampler/Python/Torch RNG and other state in raw tensor bytes. Checkpoint metrics and the final export also match. All eight tensors changed, including the trunk and both heads. In-sample MAE and policy loss improvements do not establish playing strength.

## Eight real interruption tests

Two self-play decision kills retain prefixes of one and two decisions. Resumed moves, game RNG, terminal returns and search/policy semantics match a fresh fixed-work run. Committed decision bytes remain exact. Uncommitted wall-clock fields and their dependent hashes are excluded from cross-run equality. Rehashed corrupt terminal and completion states are refused; a valid missing manifest is repaired without searching.

Two training kills cover the durable step4 file before catalog commit and the committed step4 catalog. Prefixes contain steps[2] and [2,4]. Committed checkpoint bytes and modification times remain exact. Fresh and resumed step6 states match raw tensor bytes for all model/optimizer/RNG fields. An independent six-update replay additionally matches the fixture checkpoints. The orphan file is retained without being adopted into the resumed catalog.

Four benchmark kills cover decision prefixes1/2 and game prefixes1/2 before/after commit. All committed decision and game bytes remain exact. Each resumed fixture finishes eight games, with literal legal moves, actors, terminal results and validated saved search/work/clock envelopes. Uncommitted timed choices are recomputed; equality with a separate fresh timed game is not required. Rehashed invalid completion markers cannot repair a missing report.

Each kill retains the helper marker, actual stdout event, SQLite prefix backup, resumed journal and worker logs. Producer-declared marker JSON is checked if its atomic publication finished; a hard kill can legitimately leave only its temporary file. The audit does not claim all declared marker files were published.

Completed self-play, training and benchmark API reruns import no Torch, invoke no new search/decision/optimizer runtime and preserve durable bytes and modification times. Missing valid self-play publication repair and invalid-marker refusal are tested separately.

## Equal-time evidence

The final audit literally replays all 96 terminal games and 2,958 choices. The protocol uses 12 consumed-development families, both seats, both baselines and contemporaneous control/challenger profiles at 25 ms per whole decision. No fallback occurred. Saved clocks include reset, feature extraction, inference and validation; overshoots and physical work are retained. No energy saving is claimed.

All six exploratory paired-family bootstrap intervals were independently reconstructed. Values below are paired seed margins and 95% bootstrap intervals, rounded to four decimals:

| Profile | Opponent | Mean | Interval |
|---|---|---:|---|
| Control | H_v1 | -1.0000 | [-2.6667, 0.7500] |
| Challenger | H_v1 | -0.7500 | [-2.3333, 0.8333] |
| Control | H8 relay CPU | -1.6667 | [-3.6667, 0.3333] |
| Challenger | H8 relay CPU | -1.6667 | [-3.9167, 0.3333] |
| Challenger minus control | H_v1 | +0.2500 | [-1.0833, 1.7500] |
| Challenger minus control | H8 relay CPU | 0.0000 | [-1.5833, 1.8333] |

Neither profile passes the predeclared both-baseline strength gate. These six intervals are unadjusted exploratory summaries, not release evidence. Both profiles choose an exact-optimal move on all 32 reused K≤4 guards at both 25 and 100 ms; all 256 engine/budget measurements and actor-signed regrets are checked.

The posthoc comparison checks every actual arena state above four active seeds against admitted self-play quotients/families. It finds zero collisions. This does not make the reused development sample blind or remove ancestral exposure. An actual collision would be reported as contaminated, descriptive-only evaluation rather than used for promotion.

The audit also independently reconstructs all 25,600 sampler draws and every row probability in the offline exposure review. Sampling is uniform by admitted family, then retained observation, with unit loss per draw. Concentration support is not a count of independent samples; draw IDs were reconstructed from the pinned RNG contract and actual checkpoint states rather than separately logged.

## Executed-source provenance and limitations

Five preflight methods ran under the exact retained main snapshot `build/selfplay-pilot-audit/executed-preflight-v1.py`; their raw report is `preflight-executed-v1.json`. Before method08 ran, its draft used overlap field `hits`; the frozen producer uses `collisions`. Only that unexecuted method was corrected. The whole module AST excluding method08 and all five covered method AST hashes are identical. The derived preflight report binds the retained executed source, original report and corrected current source with explicit provenance and no rerun claim. Subsequent groups bind the corrected main; the optimizer worker is unchanged.

The earlier benchmark-owner exception-class harness mismatch and provisional-domain fixtures are retained as diagnostics. They are not counted as additional successful tests. No producer or model was changed after observing the production comparison.

Observed self-play returns and root visits are approximate learning targets under the source policy, not solved game values or optimal-policy proofs. The small family panel, repeated starts and self-generated targets limit generalization claims. The current results support recoverable self-play and joint training as working engineering components; they do not justify automatic promotion, a larger model, mass training or a claim that Congklak is solved.
