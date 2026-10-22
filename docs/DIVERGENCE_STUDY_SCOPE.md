# Fixed-work and missing-action-Q investigation

This diagnostic follows the accepted ordinary-value review. It runs new bounded searches and teacher labeling before any training. Both exported models, all policy/trunk/value weights, old datasets, previous HOLD and H_v1 default remain unchanged. No strength match, self-play, optimizer, Torch runtime, source-data pruning, model selection or promotion is authorized.

## Frozen roots and identities

`build/divergence-baseline.json` protects all 3,911 prior accepted inputs plus acceptance, immutable-rerun and final-dashboard artifacts: 3,914 files. New work uses `divergence-*` names only.

`build/divergence-panel.json` projects every H_v1 first-divergence full state from the sealed `build/ordinary-value-loss-review.json`, sorted by pair/seat: pairs 2,4,6,7,8, all seat1. Pairs4/6/7 are adverse-margin cases; pairs2/8 are unchanged-margin controls. Preserve raw orientation/stores/controller/actions, legal masks, mechanism, source decision/game/seat hashes and actor-signed margin differences. All five are player1; active seeds are 10,16,36,63,27 and legal-action counts 4,5,4,6,5. This is consumed posthoc diagnostic data, not unseen evaluation or optimizer data. Do not select or replace roots based on new outcomes, and do not reuse Q by quotient/store-offset transformation.

## Fixed-simulation experiment, first

Run all five roots × both fixed models × simulation counts {128,256,512,1024}: 40 fresh-tree cells, 19,200 completed root simulations (root-edge visit backups). A simulation may back up multiple internal edges; 19,200 is not the total number of all internal-edge updates. Models are the accepted compact control `data/quality-train-v1/model.json` and the unpromoted value-only candidate `data/ordinary-value-pilot-v1/model.json`.

Use unchanged native CPU evaluation and `quality_search.search`, c_puct .25, time_ms0, root_noiseNone, rng0, cache capacity0, max_batch1, max_nodes2048, memory8192, and the same declared CPU core2 as the old arena. Pin the actual admitted backend and binary hashes. Every cell starts with a fresh tree/cache. Wall time and inference time may be retained as physical observations, but are excluded from the deterministic semantic digest and do not create a new speed or strength result.

A complete cell must be nonterminal, usable, stop for simulations, and have completed_simulations=requested_simulations=sum(root visits). Retain raw policy priors, visited-action means, unvisited None values, parent bootstrap, visits, work/counters and node bounds. A cap/cancellation/memory failure is typed incomplete evidence, never a completed budget or a surrogate smaller simulation count. No automatic extra budgets or exploration retuning. Cooperative interruption may stop without committing a completed cell; physical interruption may recompute only the uncommitted cell. Preserve committed cell bytes and deterministic semantics on resume.

## Teacher action Q, second

Production labeling starts only after all40 primary cells are complete and independently audited. Teacher plan/source freeze precedes either production outcome. Each exact root receives three independent Native H_v1 finite-choice alpha-beta/PVS attempts:

| Requested horizon | Per-root node cap |
|---|---:|
| H8 | 2,000,000 |
| H10 | 5,000,000 |
| H12 | 10,000,000 |

All15 attempts run irrespective of earlier caps, with time_ms0, TT16MiB, PVS on, memory512MiB, fresh root/horizon control and keep_historyTrue. Total requested production node ceiling is 85,000,000; engineering/audit work is separate and explicitly identified. There is no result-dependent escalation or replacement.

Each requested complete label covers every legal root action at that requested choice depth. Retain the full consecutive completed iteration prefix, candidate/PV paths, actor/control/bonus-turn semantics, work, root linkage, label linkage and stop reason. Capped attempts retain completed auxiliary depths but publish no requested-depth label. H10 is the primary new reference; H8/H12 are separate sensitivity diagnostics. Missing requested-depth Q remains missing. These are finite-horizon H_v1-leaf estimates, not game-theoretic truth. No lower-horizon fallback, parity assumption for finite values, or mechanical alternating sign after bonus turns.

For actor sign s=+1 for player0 and -1 for player1, use best action maximizing s*Q_player0; regret is s*(Q_best−Q_chosen), retaining ties. Compare saved timed actions and every fixed-work chosen action to each complete requested horizon separately. PUCT sampled means and teacher Q have different estimands and must be labeled separately; unvisited PUCT actions do not acquire invented values.

## Implementation, independent audit and recovery

Two agents implement search and teacher producers in separate new modules. Each freezes source/configuration before any fixture search. Search engineering fixtures use the first two already-consumed `build/value-pilot-rows.json` diagnostic rows, 8/16 simulations, both models, plus declared node-cap/cancellation variants. Teacher engineering fixtures use already-consumed ordinary validation diagnostics at requested depths2/3/4, with explicit fixture identity; no current five-root label is computed early.

An independent auditor freezes eight methods before formal execution. Preflight covers identity/models/policy, search fixture recovery and teacher fixture recovery. Exercise four actual child-process kills: before/after one search-cell commit and before/after one teacher-attempt commit, using explicit producer barriers plus independent durable prefix snapshots. Recovery preserves committed prefixes, refuses rehashed corrupt rows/history/completion, and completes uncommitted work. Owner fixtures and independent reruns are not extra production labels or strength evidence.

After the40-cell production run, independently check full-budget work conservation, leaf/model/policy parity and deterministic selection/backups at all40 cells before authorizing teacher launch. After teacher production, audit all15 attempts, every completed iteration and literal PV, requested-depth full action coverage and actor-signed comparisons. Final corruption and completed API/CLI checks must start no new searches, import no Torch and leave every prior input unchanged. Actual new search is allowed only in declared incomplete/engineering/audit replay paths; completed production reopening does no new engine work.

Use SQLite atomic per-cell/per-attempt commits, identity/source pins and immutable completed report publication. Missing exports may be rebuilt from validated committed journals; wrong existing outputs must refuse. Independent report groups must bind exact executed source/method IDs. Count unique current methods only, retain failed/interrupted executions separately and never invent an aggregate after interrupted publication. Use the already-tested non-object-JSON-safe logging behavior in the new dashboard.

## Interpretation and final decision

The fixed-work comparison can isolate the two models' deterministic search behavior under this protocol; it does not retroactively explain every timed choice or prove a game-result cause. New finite Q can identify disagreements/regret relative to its stated horizon, not a unique full-game blunder. All five cases and all four budgets remain reported, including null changes, ties, caps and horizon disagreement. Keep models and prior allocation fixed regardless of result. Final recommendations concern the next bounded diagnostic/data/training design only; no training follows automatically.

Update a new observer on the existing local port18766 while preserving accepted dashboard sources. Verify backend/listener and immutable evidence, without retrying previously denied browser access. Final acceptance binds current sources, executed owner/independent checks, four physical recovery tests, production results, stable completed reruns and all3,914 prior pins. No additional stage starts until its required independent proof exists.
