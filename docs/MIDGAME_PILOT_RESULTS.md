# Targeted midgame data and diagnostic labels

The bounded study completed with **49 fresh roots** (39 training candidates, 10 protected validation roots) and **nine protected diagnostic roots**. H8 and H10 completed for all 58 roots. H12 completed for eight of nine diagnostics; the remaining attempt hit its cap and retained H11 as auxiliary evidence. Models, policy and active sampling remain unchanged. No training, PUCT games, ingestion or promotion occurred.

## Collection and labeling

The fixed 512 standard-start prefixes produced 5,013 eligible observations. Transitive family/eligible-quotient protection excluded six families before deterministic selection. The 49 selected roots represent 38 families; shared prefixes outside eligible strata remain possible. No root was added to fill a shortfall.

| Fresh target stratum | Training candidates / cap | Validation / cap |
|---|---:|---:|
| 50-74 ordinary | 15 / 16 | 4 / 4 |
| 25-49 ordinary | 8 / 8 | 2 / 2 |
| 25-49 bonus | 8 / 8 | 2 / 2 |
| 25-49 capture | 8 / 8 | 2 / 2 |

All selected fresh roots have at least two legal actions. The nine diagnostics retain their original full states, stores, controller and ownership, including late and nonordinary positions. The sole preexisting H8 label remains an unchanged provenance reference; its Q/best semantics agree with the new same-depth results.

| Requested stage | Eligible roots | Complete labels | Budget rejections |
|---|---:|---:|---:|
| H8, 2M nodes per attempt | 58 | 58 | 0 |
| H10, 5M nodes per attempt | 58 | 58 | 0 |
| H12, 10M nodes per attempt | 9 diagnostics | 8 | 1 |

The 125 production attempts used **48,565,516 new nodes**, retaining 1,151 complete iteration records. The initial producer loop timer was 96.625 seconds; this excludes collection, admission/setup before the loop, analysis and independent fixtures. The predeclared 503M maximum was the requested production-attempt sum, not a lifetime compute or energy ceiling. Audit work and recomputation of an interrupted, uncommitted attempt are separate costs. No H14 or budget expansion was authorized.

## What the labels show

These are finite-choice H_v1 teacher estimates, not solved-game truth. H8/H10 best sets changed on 9/39 fresh training candidates (four disjoint), 1/10 validation roots (none disjoint), and 3/9 diagnostics (one disjoint). Among the **eight complete diagnostic H10/H12 pairs**, four best sets changed and two were disjoint. The capped ninth diagnostic is excluded from those paired means. Fresh roots are outside H12 scope, not missing H12 labels.

Fixed CPU predictions against complete H10 targets give:

| Cohort | Roots | Control MAE, seeds | Pilot MAE, seeds |
|---|---:|---:|---:|
| Fresh training candidates | 39 | 8.3506 | 8.3124 |
| Protected fresh validation | 10 | 9.2154 | 9.3259 |
| Protected diagnostics | 9 | 3.8717 | 4.0753 |

Policy logits remain identical. The small validation cohort has now been inspected as development evidence; it is not a sealed release test. These errors do not measure playing strength.

The following display pits are **1-7**; source pair IDs and candidate seats retain their original zero-based numbering. "Opponent" means the first divergent decision was made by H8 relay, not by either candidate. Saved-action regrets compare the two games' actions at the exact original state and requested H12 horizon, signed for that state's actor.

| Source pair / candidate seat | First actor role | H8 best pits | H10 best pits | H12 best pits | H12 saved control / pilot regret |
|---|---|---|---|---|---:|
| 0 / 0 | Candidate | 4, 7 | 4, 7 | 4, 7 | 0 / 0 |
| 1 / 1 | Candidate | 6 | 3, 6 | 3 | 1 / 3 |
| 3 / 0 | Candidate | 5 | 5 | Capped; auxiliary H11 | Not available |
| 5 / 0 | Candidate | 4 | 7 | 4 | 1 / 0 |
| 5 / 1 | Opponent | 5 | 5, 6 | 6 | 0 / 1 |
| 7 / 1 | Candidate | 6 | 6 | 6 | 1 / 0 |
| 8 / 0 | Candidate | 5 | 5 | 5 | 5 / 0 |
| 9 / 1 | Candidate | 1 | 1 | 1 | 0 / 9 |
| 10 / 0 | Opponent | 5 | 5 | 6 | 2 / 0 |

The previously covered worst-loss case (pair1/seat1) changes from H8 favoring the saved pilot pit6 to H12 preferring pit3: pilot regret rises from 0 to 3, while saved control pit2 remains regret1; neither saved action is H12-best. Pair5/seat0 instead changes from H10 favoring saved control pit7 to H12 favoring saved pilot pit4. These horizon movements prevent treating a single teacher depth as a causal game-blunder verdict. The two opponent-owned cases especially do not isolate the value-head update. The 66-active-seed pair3 case remains unresolved at requested H12 under this scope's cap.

## Offline exposure proposals

The review uses only the **39 fresh training candidates from 30 families**, all with complete requested H10. Validation and all diagnostics are excluded. No auxiliary label substitutes for an incomplete requested label.

| Proposal | Root effective support | Family effective support | 50-74 ordinary mass |
|---|---:|---:|---:|
| Uniform root | 39.000 | 25.780 | 38.46% |
| Equal nonempty target stratum | 36.226 | 21.215 | 25.00% |
| Horizon-stability weighted root | 34.843 | 23.242 | 32.87% |

Effective support means inverse probability concentration, not independent games or gradient ESS. The stability factors are uncalibrated heuristics, not probabilities of correctness. With unit loss, changing draw probabilities changes the risk objective. No proposal has been applied, no data pruned and no replay blend chosen. Before any learning experiment, rebuild the complete role-aware optimizer admission set and freeze its data, loss, exposure and strength-test plan separately.

## Evidence and resume

All **16 new methods passed**: four collection-author, four label-author and eight independent checks. The independent recovery tests killed real processes before and after collection commits and at both commit barriers for each of H8/H10/H12: eight actual kill/resume probes, with retained seed journals and exact committed prefixes. Complete markers, partial-label refusal, full-root Q/PVs, original ownership and metric denominators were checked.

The [combined completed-rerun proof](../build/midgame-pilot-root-verification.json) re-executed collection, labeling and both reviews with new searches, decisions, optimizer calls and training-runtime activation prohibited. It preserved ten durable artifacts including both model files byte-for-byte and by modification time, kept all **2,639 prior pins** unchanged, and imported no Torch. New CPU model forwards and validation of saved histories are allowed analysis work.

Current evidence:

- [Collection result](../build/midgame-data-pilot-result.json) and [full-root label result](../build/midgame-label-pilot-result.json).
- [Per-root fixed-model, saved-action and horizon comparison](../build/midgame-label-pilot-comparison.json).
- [Offline fresh-candidate probabilities](../build/midgame-pilot-weight-review.json).
- [Independent audit](MIDGAME_PILOT_INDEPENDENT_AUDIT.md) and [scope](MIDGAME_PILOT_SCOPE.md).

Resume an interrupted operation with the identical command and paths; committed attempts remain intact, and an uncommitted attempt is recomputed. Completed `all` reruns validate and preserve durable artifacts without new teacher searches.

```powershell
python -m congklak.midgame_pilot_progress run --tool midgame-collection -- python tools/midgame-data-pilot.py all
python -m congklak.midgame_pilot_progress run --tool midgame-labels -- python tools/midgame-label-pilot.py all
python tools/midgame-data-pilot.py verify
python tools/midgame-label-pilot.py verify
python tools/analyze-midgame-label-pilot.py verify
python tools/record-midgame-pilot.py --check-only
```

The observer is available at http://127.0.0.1:18766/ and reads committed status. Backend checks are available; browser visual verification remains blocked by the existing browser security policy. H_v1 remains the default; this scope makes no learned-strength or release-readiness claim.
