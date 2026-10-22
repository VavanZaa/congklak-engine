# Within-cell weighting did not pass its frozen screen

All 16 selected assessment roots completed H8, H10 and H12: 48 bounded attempts, zero caps, 480 completed iteration reports and 39,781,297 production nodes. The original rule and selected roots were not changed after measurements. The sole failed screen condition is overall full-action Q drift. No weights were applied, no model or policy changed, and no training or promotion follows this result.

The fixed-mixture summaries are in seeds:

| Metric | Uniform | Within-cell weighted | Frozen condition |
|---|---:|---:|---|
| H12 regret of the full H10 best set, mean | 0.187500 | 0.171334 | Pass |
| H12 regret of the full H10 best set, maximum per root | 0.187500 | 0.171334 | Pass |
| Absolute H10-to-H12 root shift | 1.875000 | 1.873096 | Pass |
| Mean legal-action H10-to-H12 Q shift | 1.999405 | 2.023762 | **Fail** |

The weighted Q shift increases by about 0.024357 seeds. That contradicts a declared safeguard despite small reductions in the other pooled measures. Preserving group mass solves the preceding cross-stratum mass-shift concern; it does not establish that this H8/H10 heuristic reliably orders later H10/H12 action stability.

Each stratum retains exactly 25% mass, each actor 50%, ordinary positions jointly 50%, and every stratum/actor cell 1/8. All eight cells have nontrivial weights; eight root probabilities increase and eight decrease. Overall concentration ESS is 15.769717 versus uniform 16; the smallest ordinary-cell ESS is 1.955246 versus uniform 2. All declared mass and concentration safeguards pass. ESS describes concentration, not independent games or calibrated uncertainty.

The apparent regret benefit is very sparse. Fourteen roots have zero H10-best-action H12 regret. Only two have nonzero regret, totaling three unweighted seeds, and both are actor 1:

| Origin family | H10 best → H12 best, human pits 1–7 | H12 regret | Probability, uniform → weighted |
|---|---|---:|---:|
| `midgame-history-pilot/20261023/483`, low bonus | 7 → 4 | 2 | 0.062500 → 0.059028 |
| `ordinary-history-pilot/20261020/94`, low ordinary | 7 → 6 | 1 | 0.062500 → 0.053279 |

These are the only two changed/disjoint best sets. The pooled regret decrease is entirely the reduced averaging weight on these same two reference errors; no move or prediction was fixed. Every actual H10 and H12 best set is a singleton, so minimum, mean, maximum and lowest-pit regret coincide on this panel. Full-tie correctness is covered by synthetic and independent tests, but this real panel supplies no empirical tied-action comparison.

All stratum results remain visible. These are conditional means within each stratum, combined using the same fixed 25% outer mass:

| Stratum, four roots each | Mean best-set regret U → W | Root shift U → W | Action-Q shift U → W |
|---|---:|---:|---:|
| 50–74 ordinary | 0 → 0 | 3.000000 → 2.864174 | 3.841667 → 3.815191 |
| 25–49 ordinary | 0.250000 → 0.213115 | 1.250000 → 1.224490 | 1.000000 → 1.000000 |
| 25–49 bonus | 0.500000 → 0.472222 | 1.250000 → 1.263889 | 1.604167 → 1.608550 |
| 25–49 capture | 0 → 0 | 2.000000 → 2.139831 | 1.551786 → 1.671308 |

Every ordinary mean/worst-tie regret safeguard passes. Three ordinary actor cells have zero regret throughout; only the low-ordinary actor-1 cell improves its regret mean, 0.500000 → 0.426230. The complete two-root cell summaries show the regressions outside those decision guards:

| Cell | Mean best-set regret U → W | Root shift U → W | Action-Q shift U → W |
|---|---:|---:|---:|
| High ordinary, actor 0 | 0 → 0 | 1.500000 → 1.455285 | 2.833333 → 2.833333 |
| High ordinary, actor 1 | 0 → 0 | 4.500000 → 4.273063 | 4.850000 → 4.797048 |
| Low ordinary, actor 0 | 0 → 0 | 1.500000 → 1.448980 | 1.000000 → 1.000000 |
| Low ordinary, actor 1 | 0.500000 → 0.426230 | 1.000000 → 1.000000 | 1.000000 → 1.000000 |
| Low bonus, actor 0 | 0 → 0 | 1.000000 → 1.000000 | 1.541667 → 1.531915 |
| Low bonus, actor 1 | 1.000000 → 0.944444 | 1.500000 → 1.527778 | 1.666667 → 1.685185 |
| Low capture, actor 0 | 0 → 0 | 0.500000 → 0.590164 | 1.175000 → 1.278689 |
| Low capture, actor 1 | 0 → 0 | 3.500000 → 3.689498 | 1.928571 → 2.063927 |

Both capture cells put more weight on the root with greater subsequent action-Q drift. For example, actor-0 family `.../283` receives probability 9/122 with later Q drift 1.75, while `.../233` receives 25/488 with drift 0.6. This explains a concrete contribution to the failed aggregate Q safeguard. It does not establish a causal property of a learned evaluator.

The panel has 16 distinct reserved origin families/components and is separate from the preceding consumed panel. It is not certified historically unseen: four high-ordinary roots had previous requested H8/H10 exposure, one low-ordinary root had H8 exposure only, and the other 11 had none in the three tracked source journals. Shared histories, ancestral and internal-search exposure are not ruled out. All groups contain only two roots; the balanced assessment mixture is deliberately chosen, not a measured natural-position distribution. H12 is a finite H_v1 reference, not solved truth or an equal-time strength test.

The primary literature supports fixed group targets, direct decision metrics and explicit concentration/coverage safeguards. It does not prove this heuristic or warrant relaxing the failed condition. Retain the current weights and previous allocation HOLD; any revised rule would need a separately frozen assessment rather than a sweep on these outcomes. Exact rational evidence, endpoint identities and every condition are retained in `build/stratum-result.json` and `build/stratum-rule.json`; methodology and citations remain in `docs/STRATUM_METRICS.md` and `docs/STRATUM_RESEARCH.md`.
