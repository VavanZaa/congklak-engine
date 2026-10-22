# Revised Q-stability assessment: HOLD

The revised maximum-action rule lowers overall H10/H12 Q drift on this panel, but **fails the frozen benefit screen**: capture-stratum mean Q drift increases, and overall H10-best-action mean and worst regret increase. Keep the existing sampling/loss settings and allocation HOLD. No proposed weights were applied, and no training, neural inference, self-play, strength games, pruning or promotion occurred.

The closed evidence is [qstability-result.json](../build/qstability-result.json), with the prospectively fixed [registry](../build/qstability-registry.json) and H8/H10-linked [realized rule](../build/qstability-rule.json). This is a separate 16-root panel, not a rerun of the preceding four-root capture diagnosis. A cross-panel comparison cannot establish that this rule outperforms the old rule.

## Complete support and exact comparison

All 16 requested H8, 16 H10 and 16 H12 attempts completed, with zero caps: **43,603,025 recorded production nodes and 480 complete iteration reports**. There are 82 legal root actions and 246 endpoint action-Q values. No auxiliary depth substituted for a requested endpoint. Both comparators cover their full original probability mass, with zero missing mass.

Selection used metadata before new endpoint answers: two roots in each of eight stratum/actor cells, 16 distinct origin families/components. The realized rule was frozen before H12:

```text
d_inf = max_all_legal abs(Q10-Q8)
r = max(1/4, 1/(1+d_inf/4))
W_i = (1/8) * r_i / sum_same_cell(r)
U_i = 1/16
```

The scale4, floor1/4 and mixture were fixed, with no fitted coefficient or rule sweep. Every cell retains mass1/8, each stratum1/4, each actor1/2, ordinary support1/2 and each actor's ordinary support1/4. Seven cells are nontrivial; low-ordinary actor0 remains uniform. No raw score reaches the floor. Global inverse-concentration ESS is **15.61428216**, above12.8; every cell exceeds1.6, with a range **441/233 ≈1.892704 to2**. The maximum root probability is13/168; concentration is not independent sample size or uncertainty calibration.

| H10/H12 estimand | Uniform U | Revised W | Frozen guard |
|---|---:|---:|---|
| Mean legal-action Q drift | 13309/6720 =1.98050595 | 23324807/11880960 =1.96320895 | Strict reduction passes |
| Mean of per-root maximum Q drifts | 17/4 =4.25 | 144367/34944 =4.13138164 | Nonincrease passes |
| Root-value drift | 27/16 =1.6875 | 984631/594048 =1.65749401 | Nonincrease passes |
| Full H10-best-set mean regret | 1/4 =0.25 | 2/7 ≈0.28571429 | **Fails** |
| Full H10-best-set worst regret | 1/4 =0.25 | 2/7 ≈0.28571429 | **Fails** |
| Capture-stratum mean Q drift | 41/24 ≈1.70833333 | 2455/1428 ≈1.71918768 | **Fails** |

Mean Q drift improves by41101/2376192 ≈0.01729700. The maximum-Q summary is a probability-weighted mean of each root's maximum, not the invariant largest shift anywhere in the panel. Mean and worst tied-best regrets are equal here because all H10 and H12 best sets are singletons. All ordinary regret guards pass at **zero regret under both comparators**; this is not evidence that ordinary moves were improved.

These are the only three failed conditions: capture Q nonincrease and the two overall regret conditions. Completion, exact masses, nontriviality, strict overall mean-Q improvement, overall maximum-Q/root guards, every ordinary regret guard and all concentration guards pass. Engineering completion is distinct from a passed benefit screen.

## All strata and cells remain visible

Values below are conditional means within the named fixed-mass stratum. Positive changes are greater drift/regret, not newly trained model decline.

| Stratum | Mean Q drift, U → W | Mean root-max Q drift, U → W | Root drift, U → W | Best-action regret, U → W |
|---|---|---|---|---|
| High ordinary, 50–74 | 3.192857 →3.300000 | 6.500000 →6.416667 | 2.500000 →2.619048 | 0 →0 |
| Low ordinary, 25–49 | 1.395833 →1.377604 | 2.750000 →2.718750 | 0.750000 →0.718750 | 0 →0 |
| Bonus, 25–49 | 1.625000 →1.456044 | 4.000000 →3.604396 | 1.250000 →1.098901 | 1.000000 →1.142857 |
| Capture, 25–49 | 1.708333 →1.719188 | 3.750000 →3.785714 | 2.250000 →2.193277 | 0 →0 |

The following table reports W−U in every actor cell. Mean/worst best-action regret coincide in this observed panel.

| Stratum / actor | Δ mean Q | Δ mean root-max Q | Δ root | Δ best-action regret |
|---|---:|---:|---:|---:|
| High ordinary /0 | 0 | −0.166667 | 0 | 0 |
| High ordinary /1 | +0.214286 | 0 | +0.238095 | 0 |
| Low ordinary /0 | 0 | 0 | 0 | 0 |
| Low ordinary /1 | −0.036458 | −0.062500 | −0.062500 | 0 |
| Bonus /0 | −0.230769 | −0.576923 | −0.230769 | 0 |
| Bonus /1 | −0.107143 | −0.214286 | −0.071429 | +0.285714 |
| Capture /0 | +0.011905 | +0.071429 | −0.142857 | 0 |
| Capture /1 | +0.009804 | 0 | +0.029412 | 0 |

Actor0's aggregate mean Q drift decreases2.029762 →1.975046; actor1's increases1.931250 →1.951372. Actor1 root drift and regret also increase. The overall improvement therefore does not imply uniformly better strata, actors or cells.

## Root-level drivers

**One changed best action accounts for all positive best-action regret.** At `midgame-history-pilot/20261023/178`, actor1 prefers smaller player0 Q. Human pit6/pit7 have Q8 **14/16**, Q10 **13/14**, and Q12 **16/12**. Thus H10 prefers pit6, while H12 prefers pit7; the retained H10 choice has regret4. Its early maximum shift is2, giving r=2/3 and W=1/14 rather than1/16. The other root in its bonus/actor1 cell has early maximum4, r=1/2 and zero H12 regret. Upweighting this single finite-reference disagreement raises overall regret by exactly **4×(1/14−1/16)=1/28**. No action was changed by the weighting assessment; only its diagnostic probability changed.

**Both capture cells upweight their root with greater later mean Q drift.** These are four different roots from the preceding capture investigation:

| New capture root suffix | Actor | Legal actions | Early d_inf | W | Later mean Q drift |
|---|---:|---:|---:|---:|---:|
| 91 | 0 | 3 | 2 | 1/14 | 5/3 |
| 382 | 0 | 4 | 4 | 3/56 | 3/2 |
| 92 | 1 | 5 | 4 | 9/136 | 2 |
| 390 | 1 | 6 | 5 | 1/17 | 5/3 |

All four families use prefix `midgame-history-pilot/20261023/`. Their H10 best actions remain H12-best and have zero regret. Nevertheless, the fixed-mass actor cells' mean Q drifts increase, so the capture guard correctly rejects the pooled benefit.

**The high-ordinary actor1 cell also regresses outside the passed overall Q summary.** `midgame-history-pilot/20261023/95` has early maximum4 and later mean Q drift4, receiving W=13/168. Its partner `ordinary-history-pilot/20261021/155` has early maximum9 and later mean11/5, receiving W=1/21. This shifts mass toward the greater later drift. The early maximizing actions are H10-nonbest: pits1/2 at root95 have actor gaps24/30; pit2 at root155 has gap11. In the full panel, **14 of16 early maxima are attained only by H10-nonbest actions**; the remaining two include both best and nonbest actions tied for the maximum. Nonbest does not necessarily mean unimportant: root178's early maximum is at pit7, only one seed behind the H10 best and the eventual H12 best.

Legal-action counts range2–7. A maximum can respond to a distant weak alternative and to the number of alternatives, while best-action regret depends on relative gaps. The observed failures do not isolate either mechanism causally. No action-count correction, gap filter, new coefficient or alternative rule was selected after these outcomes.

## Interpretation and unchanged settings

Only one of16 H10 best sets changes, and all best sets are singletons. This makes the regret result highly dependent on one finite-horizon disagreement. The full signed Q, gap, rank, best/nonbest and probability contributions remain in the result, including unchanged actions and all failed safeguards. H12 is approximate H_v1 finite-choice supervision, not solved game truth.

The metadata reports three roots with prior raw/quotient H8 and H10 exposure. Separate eligible components do not establish historical novelty, absence of ancestral model exposure or independent games. This adaptively motivated small panel is consumed assessment evidence. The prior capture findings motivated the rule, but were not used to fit coefficients or to select favorable outcomes in this separate panel. Literature supports safeguards and experimental discipline, not a guarantee for the heuristic; see the frozen [QSTABILITY_METRICS.md](QSTABILITY_METRICS.md).

The conclusion is **HOLD**, with existing models, policy, sampling/loss settings and production search unchanged. The result does not authorize training, weight application, capture-data removal, strength claims or promotion. Independent source/recovery/coverage evidence is recorded separately in [QSTABILITY_INDEPENDENT_AUDIT.md](QSTABILITY_INDEPENDENT_AUDIT.md).
