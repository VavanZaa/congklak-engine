The completed value pilot remains quarantined. This investigation uses existing
data and fixed models/policy; it authorizes no further training or promotion.
H_v1 remains the default engine.

Capture training error improves while the consumed fit panel gets worse.
Capture seed MAE is 4.8520 to 3.8834 on 128 training roots, versus 5.6179 to
5.9573 on 32 fit roots. Player-1 fit capture MAE increases from 5.9771 to
6.6978; player-0 decreases from 5.2587 to 5.2168. Nineteen capture fit roots
worsen and thirteen improve. Five roots contribute more deterioration than the
net increase across all 32. These results describe failed transfer on this
panel; they do not prove overfitting, bad teacher labels, or a player-index bug.
Eighteen regressions move away from the original target; only one is an
overshoot past it. For the worst root, the player-0 teacher score is -41,
control predicts -37.3611 and the candidate -33.9007. The added correction
moves an already inaccurate prediction farther from that fixed target.

The learned sidecar is the arithmetic source of the changed predictions: the
base tensors, policy and cached inputs are unchanged. Scores are converted
through the correct actor perspective. A negative actor-relative correction
can increase the player-0 score for player 1, worsening an already pessimistic
player-1 prediction. The network has no explicit actor-ID input; per-player
differences may reflect the sampled feature and target distributions. Cached
FP32 features differ slightly from the earlier fresh-feature F64 evaluation;
the detailed report records that numerical difference rather than asserting
bitwise agreement.

Root scalar error and chosen-action regret are different measurements. Saved
search results provide associations between them, but do not expose every leaf
value that influenced a PUCT decision. All current targets are one complete
depth-8 H_v1 search convention, not solved game outcomes. The existing data
cannot establish which disputed teacher values are objectively correct.
In the eight saved capture search roots, 25-ms mean action regret is 0.125 for
control versus 1.375 for candidate; one changed move contributes the ten-seed
regret difference. At 100 ms both have 0.125 regret; at fixed 128 simulations
both have 1.75. These are existing small diagnostic samples, not a new strength
test or evidence that scalar root error alone caused the changed move.

Arena ownership fails through three distinct mechanisms. Two observations in
games 152 and 156, both family 19, hit the same frozen training-origin alias at
seven active seeds after an H8 relay CPU action. An alias may belong to the
same historical origin component without being the optimizer root itself.
Game 152 uses the candidate and game 156 the control. The observed state has
stores [40, 51], while the matched old origin has [43, 48] and the same pits
and actor. The selected optimizer root is a different 25–49 bonus root in the
same origin component. This is a full-origin closure hit, rather than an exact
raw-state match with that selected optimizer root.
Twelve direct cross-family quotient bridges occur at five active seeds.
The quotient ignores store offsets and also identifies player-swapped boards;
quotient equality therefore does not necessarily mean identical raw states.
Three historical-component bridges connect different observed quotient nodes
through older shared origin families. One historical component links all 24
arena families; this does not mean every family played the same position.

These observed collisions lie below the updated sidecar's 25–74 active-seed
band. That localizes the graph problem but does not repair the predeclared
whole-trajectory ownership test, which includes every observation above four
active seeds. All 192 original games and the ownership hold remain intact.
No games were filtered, clusters merged, intervals recalculated, or historical
components silently added to the gradient pool. Strength intervals remain
withheld. Broader historical provenance is illustrative rather than a complete
reconstruction of every old union edge.
All 228 diagnostic units completed: 21 capture/search-record units, 192 current
arena replay units, and 15 historical-source units. The current replay covers
6,663 actions and 5,160 eligible observations. The historical probe replays 120
already saved games (3,970 actions) and retains 15 concrete cross-origin
collision examples. It does not generate new games. The maximum difference
between cached-FP32 and earlier fresh-F64 predictions is 0.000000954 seeds.

Before another treatment, specify an evaluation protocol prospectively. It
must distinguish disjoint learned-domain evidence from shared frozen late-game
continuations and state exactly which strength claim it can support. Establish
fresh admissible starts and reservations under that audited protocol before
using their outcomes. The current fit roots are consumed diagnostics, not a
fresh validation or release panel. Do not change weights according to the
worst five roots or reinterpret the old arena under a new ownership rule.

Once that protocol is accepted, a small predeclared regularized value-only
experiment with the same target and frozen policy is a candidate next step.
Measure per-cell value error, action regret, both-baseline equal-time results
and the exact endgame guard. Preserve the original final400/control comparisons
and fix the step budget in advance. Larger data generation or longer training
has not been justified by this diagnostic stage.

The diagnostic runner resumes with
`python tools/run-value-diagnostics.py run` and verifies completed evidence with
`python tools/run-value-diagnostics.py verify`. Immutable per-unit commits form
the authoritative prefix. Uncommitted units are recomputed; committed results
are validated and reused. The final catalog is the sole completion marker.
The dashboard at http://127.0.0.1:18766/ separates current diagnostic counters
from the historical 400 training updates and 192 games. See the independent
audit and final acceptance for the executed recovery checks and source pins.
