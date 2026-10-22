# Pilot losses, ordinary errors and readiness

The saved games do not establish that the value update caused identifiable blunders. They show concentrated adverse margins, incomplete teacher coverage and ordinary errors that worsen in the small 50-74 validation band. This review starts no training, searches, games or target generation. Both models and the policy remain fixed; H_v1 stays the default.

All twelve H8-relay development families and both seats were retained. The tool literally replayed 72 existing games and 2,460 choices across pretraining control, current control and pilot. H8 relay names the model's supervision horizon; its actual timed-search completed depth is retained per decision.

| Retained run versus H8 relay | Wins/draws/losses | Mean candidate seed margin |
| --- | --- | ---: |
| Pretraining control | 11/1/12 | -0.9167 |
| Current control | 12/0/12 | -0.7500 |
| Value pilot | 12/1/11 | -1.6667 |

The pilot has one fewer loss and one more draw in aggregate WDL, yet its matched mean margin is **0.9167 seeds worse than current control**. WDL clips outcome severity; margin preserves it. Five family contributions are adverse, two favorable and five unchanged: adverse paired-margin mass 17 versus favorable mass 6, net -11 across twelve families. The two largest adverse families account for **82.35% of gross adverse mass**. Leave-one-family-out means range from -1.3636 to -0.1818 seeds; none turns positive. This is descriptive sensitivity on consumed development, not a new confidence interval or release gate.

The two main cases use human pit numbers 1-7 (machine IDs remain 0-6):

| Family / changed seat | Current control -> pilot margin | First distinct saved action |
| --- | --- | --- |
| `random_history/20261015/7581`, seat 1 | -6 -> -24 | Choice 1: pit 2 -> pit 6; 346 -> 336 simulations |
| `heuristic_history/20261015/7462`, seat 1 | +10 -> 0 | Choice 2 after one common bonus move: pit 1 -> pit 3; 364 -> 344 simulations |

The first family's priors are identical. It is the only first-divergent state with a matching stored label, and that finite H8 label prefers the pilot's pit 6 (actor regret 0) over control's pit 2 (regret 1). Its worse played outcome therefore cannot certify an immediate teacher-defined blunder. In the second family no matching stored Q exists. The pilot's changed move captures 8 and gains 9 immediately; control gains 1 and retains a bonus choice. Later replies and paths differ, so these observations do not prove greediness or identify one causal error.

Fifteen of 24 seat trajectories use identical actions. Nine diverge: seven first changed actions belong to the candidate profile and two to H8 relay. Opponent-first means the actor at the first divergence, not necessarily the game's first move or an absence of prior candidate evaluation. Exact full-state teacher Q covers only one of those nine states; eight remain explicitly uncovered. Matching requires identical pits, stores and controller, with no player swap, quotient or store-offset transform. The report retains original visits, Q means, priors, bootstrap values, depth, physical counters and clocks. Eighteen new direct CPU forwards at nine states check raw policy/value differences; they perform no search, and their timing is unmeasured.

Unchanged control itself differs between pretraining and current runs on four seats, with mean margin drift +0.1667; three first divergences are H8 decisions. The second adverse family has +5 paired-margin control drift. Wall-clock work counts and choices can vary even with fixed models. Neither candidate-first divergence nor a nearby capture establishes a unique value-head cause.

The separate native-forward ordinary review finds:

| Ordinary validation band | Roots | Change in MAE, seeds | Direction |
| --- | ---: | ---: | --- |
| 25-49 active seeds | 8 | -0.1081 | 6 improve, 2 worsen |
| 50-74 active seeds | 6 | +0.2280 | All 6 worsen |
| Combined | 14 | +0.0360 | 9.1190 -> 9.1550 MAE |

These are small mixed-horizon targets, not exact truth. Actual draw replay confirms 2,512/51,200 draws (4.90625%) entered the ordinary pool, versus declared 5% mass; 7,616 unique roots and 4,589 family IDs were drawn overall. H8 rescoring of fixed predictions is a sensitivity calculation, not a counterfactual training ablation. The [ordinary findings](PILOT_ORDINARY_ERRORS.md) retain all 61 ordinary training roots, 14 validation roots, 8 diagnostics and 259 base ordinary-development roots, including severe cases and exposure.

Data engineering is ready for **bounded versioned collection**, with single-writer SQLite chunks and audited interruption recovery. It is not evidence that blind 100k collection or long training will improve strength; parallel/sharded throughput needs separate work. Before any future collection, protect the latest 144 pilot trajectories and evaluation metadata, preserve the 83 recent ownership roles (61 training candidates may rehearse; 14 validation and 8 diagnostic roots stay excluded), and retain the 259 base ordinary-development roots outside training. No such collection is started by this review.

The internal H_v1 analyzer is usable under its declared limits. A finished strong release is **not ready**: ordinary decision quality, fresh protected strength validation, unified exact/estimated analysis modes and portable packaging/rollback remain open. The [current readiness assessment](ENGINE_READINESS_CURRENT.md) lists twelve concrete gates without inventing a completion percentage. Full-game solving is a later goal rather than a prerequisite for a useful analyzer.

Read-only commands:

```powershell
python tools/analyze-pilot-losses.py
python tools/analyze-pilot-ordinary.py verify
python tools/assess-engine-readiness.py
python tools/record-pilot-loss-review.py --check-only
```

All **11 unique current methods passed**: four ordinary-review owner checks and seven independent checks, with zero failures, errors or skips. The combined read-only rerun preserved nine files' bytes, sizes and mtimes and all 2,600 prior hashes, with no Native/PUCT search, decisions, training or Torch. Model forwards and literal replay are allowed review work. A prepublication diagnostic preserved the first arena output/source before removing nondeterministic timing from new model-forward snapshots; all historical game clocks remain intact. The dashboard backend exposes six review/readiness cards and composes immutable prior observers. Browser policy blocked visual verification; no new screenshot is claimed. Mutable activity and runtime snapshots remain outside acceptance. Nothing here authorizes new training or collection.

Evidence: [arena review](../build/pilot-loss-review-arena.json), [ordinary review](../build/pilot-loss-review-ordinary.json), [independent audit](../build/pilot-loss-review-audit.json), [readiness](../build/engine-readiness-current.json), [fixed pilot results](VALUE_PILOT_RESULTS.md). No new model or sampling choice is applied from these descriptive findings.
