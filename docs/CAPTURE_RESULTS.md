# Capture-position Q-drift investigation

The frozen weighting rule still fails its action-Q drift guard. Keep the allocation HOLD, the current models/policy and the existing training weights unchanged. This is a retrospective diagnosis of **saved H10 versus H12 teacher scores**, not evidence that a newly trained network lost strength. No training was run.

The rule gives more weight to the root with larger subsequent Q drift in **both** two-root capture/actor cells. The capture stratum keeps its original total mass; this is a within-cell effect. Averaging every legal action and retaining the original 16-root denominator gives:

| Contribution to weighted minus uniform Q drift | Seeds |
| --- | ---: |
| Four capture roots | +0.0298805045823 |
| Other twelve roots | -0.00552316050643 |
| Full panel | **+0.0243573440758** |

Thus the full-panel drift changes from 1.999404761904762 to 2.023762105980585 seeds. Capture contributes more than the net increase because other roots partly offset it. Exact fractions, all three horizon comparisons and all legal-action contributions are retained in the independent all16 ledger referenced by `build/capture-audit/production.json`.

Source-root numbers below are the final component of the existing `midgame-history-pilot/20261023/…` family, not newly generated positions. Displayed pits count from one on the acting player's side; artifacts retain zero-based actions and full state keys.

| Source root | Actor | H10/H12 best pit | Frozen weight vs 1/16 | Mean absolute H10/H12 Q drift |
| --- | ---: | ---: | --- | ---: |
| 283 | 0 | 4 | Increased | 7/4 |
| 233 | 0 | 7 | Decreased | 3/5 |
| 445 | 1 | 6 | Increased | 3 |
| 201 | 1 | 7 | Decreased | 6/7 |

All four H10 best sets remain the same singleton at H12, with zero H12 best-action regret. There are no strict H10/H12 action-order reversals, and two non-best tie changes at root445. Numerical gaps still change: root445's best score moves from -41 to -36 in player-0 orientation, while its alternatives move the other way; root283 also has mixed-sign changes. These are not merely common score translations. The detailed report keeps signed shifts, centered dispersion and the exact `D=C+S` cancellation identity separate; it does not call cancellation a ranking error.

“Capture root” means a capture is available. In all four roots the stable best move instead earns a bonus choice without an immediate capture. Immediate-capturing actions account for +0.0248014759089 of the global excess, and other actions **inside these capture roots** account for +0.0050790286734. Both parts matter; a root-level category must not be mistaken for the mechanism of every action. Removing capture positions or changing the policy is not justified by this evidence.

Literal replay verifies 120 saved completed reports, 63 capture endpoint Q entries and 750 PV references, deduplicated to 252 saved lines and 1,638 literal choices. Repeated histories are not independent observations. Every one of these lines ends nonterminal. H_v1's saved score equals the leaf store margin on its recorded line; H12 is a finite horizon, not solved game truth. Changed downstream choices, captures, bonuses and leaf stores describe where saved paths differ, but do not isolate a cause of heuristic error or reveal unrecorded downstream branches. See `CAPTURE_FINDINGS.md` for the action-level and line-level details.

The H8/H10 stability heuristic measures earlier finite-horizon stability; in these capture cells it does not predict smaller later H10/H12 Q drift. This consumed panel can explain the failed guard, but must not be used to fit a replacement and then claim fresh validation. Any future weighting or value-target contract needs a prospectively declared assessment on separate eligible data. Existing reservation overlays remain in force; legacy trainers do not automatically apply those overlays. No data was deleted, pruned, relabeled or made newly eligible.

## Evidence and recovery

The current evidence comprises four producer-owner methods, four forensics-owner methods and six independent methods. Final acceptance is recorded in `build/capture-acceptance.json`; root closure also checks the guarded public-CLI run, exact action/root/global joins, completed reopen and idempotent acceptance. These closure proofs are not counted as additional test methods.

Independent real process kills immediately before and after the second case commit preserve raw committed prefixes of one and two cases. Both resumes match a fresh two-case run exactly. Completed API/CLI reopening performs semantic validation, adds no cases or engine/training activity and preserves durable bytes, sizes and mtimes. Missing final exports can be repaired only from a validated completed journal; altered source, rehashed corrupt cases or required missing evidence must refuse.

The supported continuation command is:

```powershell
python -m congklak.capture_progress run --tool capture-resume -- python tools/investigate-capture-drift.py all
```

It resumes incomplete cases or validates an already completed run. Pure arithmetic and literal replay may repeat during validation. It does not start training, search or inference. The progress dashboard at `http://127.0.0.1:18766/` shows the seven current steps and retains prior completed scopes; final closure verifies its backend and listener without browser or HTTP inspection.
