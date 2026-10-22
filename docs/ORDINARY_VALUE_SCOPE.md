# Fixed-policy ordinary value pilot

This new versioned experiment labels six reserved ordinary validation roots, trains only the accepted compact control's final value head, compares it with contemporaneous control, and allocates one bounded follow-up self-play batch. All previous accepted sources, evidence and model files remain immutable. H_v1 remains the default. This is consumed development work, not a sealed release test or full-game solution.

## Frozen before new labels or outcomes

The baseline is `build/ordinary-value-baseline.json`, preserving all 3,364 preceding inputs. The source is `data/quality-train-v1/model.json`, file SHA256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. The new candidate is `data/ordinary-value-pilot-v1/model.json`. Only the final 64 weights and bias may train; all six trunk/policy tensors and raw policy logits must remain byte-identical. PUCT choices may change through value backups.

Before production labeling, freeze the data rows, ownership ledger, six roots, arena reservation, teacher budgets, optimizer, sampling order, final checkpoint, benchmark and allocation rule. New validation answers never select a checkpoint, budget, loss, source, model or arena family.

## Validation labels and admission

Exactly the existing six selected validation roots, three each at active25–49 and50–74, receive independent requested H8/H10/H12 attempts. Root-depth node ceilings are 2/5/10 million respectively; no wall-time search budget, no budget escalation, no replacement root. Save requested/completed depth, complete root-action Q coverage, original clocks/work and explicit capped status. These are finite-choice H_v1 teacher estimates, not minimax truth. Requested H10 is the primary reference; all six must complete to satisfy the allocation validation condition. H8/H12 are separate diagnostics. Incomplete references stay missing, without auxiliary-depth substitution or zero errors.

Training uses only both choice0 terminal-outcome records of the preceding 24 ordinary training starts, excluding the 14 descendants. Each band must retain at least eight distinct complete paired origins after admission, otherwise hold without replacement. The rehearsal pool is the preceding admitted H8 base-training pool, filtered again. Preserve original raw outcomes separately; never average the two into an exact label.

Before optimization, protect the entire preceding ordinary validation ownership pool (100 families/38 eligible quotients), all earlier evaluation and diagnostic roles, all latest96 benchmark trajectories, and the new arena reservations. Any eligible protected bridge excludes its whole origin family. Previous training metadata is not a blanket evaluation ban. Validation, diagnostic and arena references never enter optimizer draws. Filtering new draws cannot undo ancestral model exposure.

## One small training treatment

Use fresh AdamW on exactly65 value parameters, CPU FP32, one Torch thread, no AMP/load workers, learning rate0.0001, weight decay0, gradient clip1, seed20261028. Run exactly200 steps, batch64, checkpoints every20, selecting final step200 regardless of metrics.

Every batch draws in this exact order:16 high-band starts uniformly then one of each start's two individual outcomes;16 low-band starts the same way;32 rehearsal rows using a frozen 90% uniform-root plus10% uniform-stratum CDF. Record draw/RNG progression. Unit loss per draw is Huber(delta0.1) on `(terminal_or_teacher_player0 - store_difference) / signed_active`; there is no policy loss or second confidence/importance weight. Cache the frozen trunk with row/input/tensor identities. This data/loss/sampling bundle is one intervention; a result cannot isolate its ingredients.

## Equal-time evaluation

Use12 deterministically selected development arena families, four each ordinary/capture/bonus, with family IDs absent from preceding arenas. Reserve their states/quotients/families before training. They are development roots with historical exposure, not unseen evaluation. Interleave unchanged control and final candidate against both H_v1 and H8 relay CPU, both seats:96 terminal games at25ms whole-decision budgets on the same pinned performance core. PUCT stays CPU/cache0/c_puct0.25,100000-backup/2048-node ceiling, no evaluation noise. Preserve physical features/forwards, granted clocks, overshoot and fallback counts.

Run the32 reused known-exact K<=4 guards and six new approximate validation roots at25/100ms across four engines:304 declared measurements. Missing primary references do not block the games or exact guard; accuracy/regret denominators explicitly use available references. Publish paired family margins and exploratory95% bootstrap intervals against both baselines. All six same-root H10 values define the primary MAE comparison, if complete. No aggregate fit or mean alone establishes strength or permits promotion. The32 reused guards do not establish universal endgame preservation.

## Conditional data expansion, never automatic retraining

Freeze and record one decision after the complete benchmark and independent checks:

1. Hold expansion if source/identity/data/policy/recovery proofs fail, any game is capped or has a fallback, any exact regression occurs, or an actual posthoc active>4 training-replay/new-arena quotient/family overlap occurs. Publish these findings without retroactive filtering or retraining.
2. If engineering is clean, all six requested H10 labels complete, candidate MAE is no worse than control, and both matched candidate-minus-control mean margins are nonnegative, take `new_starts`: at most512 newly seeded natural prefixes; at most24 fresh training starts (12high+12low); two control rollouts each, at most48 games. Mean signs allocate collection only; they are not strength/promotion gates. Preserve honest quota shortfalls.
3. Otherwise, if engineering remains clean, take `repeat_high`: exactly two additional control-policy trajectories from each of the ORIGINAL12 high-band starts, at most24 games, using new seeds/IDs. All12 starts are fixed before results; never select favorable or high-error cases after outcomes.

Both branches use the unchanged source control,64 CPU PUCT simulations, c_puct0.25,2048 nodes, max512 choices, root Dirichlet alpha0.3/fraction0.25 and visit behavior for the first8 choices, then maximum visits. Before follow-up, add every actual new benchmark trajectory to the role ledger. Fresh collection closes full-pool eligible family/quotient components before quotas and protects every eligible new validation-owned candidate. Repeated roots retain their origin identities. Protected bridges/caps exclude whole families from candidate replay; all outcomes stay as provenance. Root-first global raw-state cap4 accounts for existing admitted ordinary observations. No fillers, replacement, further batch, training or model promotion follows. New data are candidates awaiting a separate learning scope.

## Recovery and completion

Use separate FULL/WAL journals and immutable source-bound publications. Test actual process kills around label commits, checkpoint file/catalog commits, benchmark decision/game commits and follow-up decisions. Retain raw killed journals, validated committed prefixes and stdout barriers. Resumed training must match fresh model/optimizer/sampler/Python/Torch state; timed searches need legal durable prefixes, not identical future wall-clock choices. Completed APIs/CLIs must do no new search, optimizer work or Torch activation and preserve durable bytes/sizes/mtimes. Missing valid publications may repair from committed state; semantic corruption must fail closed.

Root alone launches production. Owners and independent auditor use declared separate fixtures. Audit source/method coverage is frozen before first execution; only actually executed unique passing methods count. Stage checks precede label consumption, production training, benchmark and final follow-up acceptance. Keep the live dashboard updated through a new observer overlay on port18766; backend/listener verification only, respecting the existing browser-access denial.
