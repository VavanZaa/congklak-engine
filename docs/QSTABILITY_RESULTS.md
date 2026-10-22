# Revised Q-stability assessment

**Do not apply this weighting rule. Keep the allocation HOLD and existing models, policy and active sampling/loss settings unchanged.** On the separate 16-root panel, the revised rule reduces overall numeric Q drift but increases best-action regret and capture-group Q drift. It fails the prospective descriptive screen. No training was run.

## What was tested

One fixed revision replaces the earlier root-value-plus-mean-Q summary with the largest H8/H10 Q change across **every legal action**:

```text
d_inf_i = max_legal abs(Q10_i,a - Q8_i,a)
r_i = max(1/4, 1/(1+d_inf_i/4))
W_i = (1/8) * r_i / sum_same_cell(r)
U_i = 1/16
```

The scale4, floor1/4 and stratum/actor masses were retained. There was no coefficient fit, sweep, H12-informed weighting or outcome-based replacement. The source, panel, rule and guards were frozen before new H8/H10. The realized endpoint-linked weights and immutable reference prefix were independently audited before any H12.

The 16 roots occupy 16 distinct closed origin components and origin families, two in each of eight stratum/actor cells. Selection used metadata and a deterministic SHA ranking from the remaining 3,772-state pool, excluding protected evaluation/forensic support, requested H12/H14 exposure and all preceding allocation/stratum reservations and aliases. Three selected roots have historical H8/H10 exposure; this is separate eligible support, not a claim of globally unseen data or ancestral independence. All16 selected components/families and118 eligible quotient aliases are now reserved from future optimizers. Historical roles were preserved; legacy trainers do not enforce new overlays automatically.

All48 requested teacher attempts completed:16 each at H8/H10/H12, zero capped or missing endpoints, **43,603,025 production nodes** and480 completed iteration reports. The teacher was unchanged native H_v1 with full root-action Q, PVS, TT16MiB, time cutoff0, memory512MiB and10M nodes per attempt. No bigger-budget retries, new games, neural inference or model updates occurred. H12 is a finite reference, not solved game truth.

## Results on this panel

These compare the same fixed panel under U and W. They are not comparisons against the old weighting rule on a different panel or evidence of changed engine strength.

| Diagnostic, seeds | Uniform U | Revised W | Result |
| --- | ---: | ---: | --- |
| Mean legal-action Q drift | 1.980506 | 1.963209 | Strict primary passes |
| Mean of each root's maximum-action Q drift | 4.250000 | 4.131382 | Passes |
| Root-value drift | 1.687500 | 1.657494 | Passes |
| H12 regret of H10-best actions, full-set mean | 0.250000 | 0.285714 | **Fails** |
| H12 regret of H10-best actions, per-root worst then average | 0.250000 | 0.285714 | **Fails** |
| Capture-stratum mean action-Q drift | 1.708333 | 1.719188 | **Fails** |

The exact overall mean-Q risks are13309/6720 and23324807/11880960. Full-tie mean and worst regret are1/4 and2/7. Capture Q drift is41/24 and2455/1428. The failed conditions are `primary_fulltie_regret_nonincrease_overall`, `worst_tie_regret_nonincrease_overall` and `capture_stratum_action_Q_drift_nonincrease`. Every condition is retained in `build/qstability-result.json`; none was relaxed after measuring H12.

All ordinary-stratum and ordinary-actor-cell regret guards pass with zero finite-reference regret. That does **not** mean ordinary numerical Q errors improved: high-ordinary mean Q drift rises3.192857→3.300000, and ordinary positions together rise2.294345→2.338802. Both capture actor cells also have higher mean Q drift. Favorable pooled Q drift therefore does not remove subgroup regressions.

Every comparison mass is unchanged: each cell1/8, stratum1/4, actor1/2, ordinary support1/2 and each actor's ordinary support1/4. Global inverse-concentration ESS is15.614282 of16; minimum cell ESS is441/233≈1.892704 of2. All eight cell concentration guards pass. Seven cells have nontrivial weights; low-ordinary actor0 stays uniform because its scores are equal. No raw score hits the floor. These are distribution-concentration diagnostics, not statistical sample-size or generalization guarantees.

## Why regret worsens

The entire regret increase comes from one bonus/actor1 root, source family `midgame-history-pilot/20261023/178`, full raw state key `0200000001050330020205040102011601`. H10 chooses human pit6; H12 chooses pit7. Actor1 minimizes player0 Q: pit6/pit7 scores change from13/14 at H10 to16/12 at H12, turning the one-seed preference into four seeds of H12 regret for the H10 choice.

That root receives mass1/14 rather than1/16. Its same four-seed regret then contributes2/7 rather than1/4: the exact increase is1/28. The positions, Qs and H10 choices are the same for both comparators; weighting changes the focus of the diagnostic, not the errors of an individual root or a model. All actual H10/H12 best sets are singleton; synthetic checks separately cover full ties.

The maximum H8/H10 shift is attained only by H10-nonbest actions in14 of16 roots. In the other two roots it includes both best and nonbest actions. This helps explain the heuristic's limitation as a decision-quality signal: monitoring worst numeric movement across all moves does not guarantee that a narrow best-action gap is reliable. It may also be sensitive to available action count; this panel does not isolate that effect. These are observed diagnostic limitations, not a fitted replacement formula. Do not delete the inconvenient bonus/capture cases or tune another rule on this consumed panel and call it fresh validation. Detailed root and action attribution is in `QSTABILITY_FINDINGS.md`.

## Audit, recovery and decision

Current acceptance binds four producer-owner methods, four pure-metrics-owner methods and six independent methods (preflight3/reference1/final2), with exact executed sources and retained raw evidence. `build/qstability-acceptance.json` records the completed coverage; acceptance, reopening and dashboard closure proofs add no extra method counts. Engineering acceptance and the failed weighting screen are different outcomes.

Two real process kills at the pre/post second H10 commit preserve stage prefixes1/2 and aggregate prefixes3/4. Resumes preserve raw committed bytes and equal fresh fixture semantics. A separate real one-node-cap fixture closes four rejected reference attempts as HOLD without fabricated labels, weights or H12. Corruption and required missing phase evidence must refuse; valid missing final exports can be repaired only after complete journal validation. Completed API/CLI paths add no teacher attempts, searches, inference, training or games and preserve durable bytes, sizes and mtimes. Pure arithmetic and saved Native transition validation may repeat; this is not zero validation computation or zero total Native calls.

This production run is complete. A completed reopen can be invoked with:

```powershell
python -m congklak.qstability_progress run --tool qstability-resume -- python tools/validate-qstability.py all
```

An interrupted labeling run resumes its active `h8`, `h10` or `h12` stage using the same tool. H12 still requires the immutable realized rule and independent reference admission; those gates must not be bypassed. The dashboard at `http://127.0.0.1:18766/` serves the nine-step view and retains previous completed scopes; refresh an existing tab to load the new layout. Final proof checks its backend and local listener without browser or HTTP inspection.

No weights were applied, no model was promoted and no training starts from this result. A future training pilot still needs separate eligible optimizer data, explicit reservation-aware admission and its own fixed-policy, strength and endgame checks. The evidence here supports retaining the current settings, rather than treating a small pooled drift reduction as a reason to train with this rule.
