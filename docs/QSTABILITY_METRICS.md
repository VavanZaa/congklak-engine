# Prospective maximum-action Q-stability metrics

`congklak/qstability_metrics.py` is a pure standard-library arithmetic helper. It imports no producer, Native engine, model or Torch, performs no file writes and runs no search. The separate producer establishes actual mechanisms, source ownership, complete teacher attempts, full PV validity, durable recovery and phase admission. The authoritative contract is [QSTABILITY_SCOPE.md](QSTABILITY_SCOPE.md).

## One rule and one assessment mixture

The preceding capture diagnosis is consumed design evidence: stable best moves coexisted with mixed-sign action changes, and a root-plus-mean summary upweighted some roots with greater later Q drift. This motivates one new rule; it is not a fitted result or a prediction of success. A separate metadata-selected panel and prospective H12 assessment are required. No old rule sweep or favorable-root substitution is performed.

For every legal action at the identical full raw state:

```text
d_inf = max_legal abs(Q10 - Q8)
r = max(1/4, 1/(1 + d_inf/4))
W_i = (1/8) * r_i / sum_same_cell(r)
U_i = 1/16
```

Only the instability summary changes. Scale4, floor1/4, the four position strata and eight stratum/actor masses remain fixed. Mean absolute action shift and the absolute change of the actor-best value are both bounded by the maximum full-action shift: each action changes by at most d_inf, so their mean and either extremum cannot move farther. This elementary deterministic bound motivates monitoring an isolated unstable action; it does not bound future H10/H12 changes or game error. The maximum can be dominated by a weak alternative or affected by legal-action count. All actions, signed shifts and ranking/gap diagnostics therefore remain visible. Root shift and mean shift are reported but are explicitly **not inputs to r**.

Each of 16 assessment roots has a distinct selected family/component, with two roots per cell. W preserves every cell's mass1/8, stratum1/4, actor1/2, ordinary support1/2 and each actor's ordinary support1/4. Equal maxima, including floor saturation, legitimately give a uniform cell. At least one cell must differ from uniform for a benefit screen. All arithmetic uses exact Fractions; serialized numbers are canonical `{numerator, denominator, value}` objects, with floats for display only. Inverse concentration ESS is `1/sum(p_conditional^2)` on each selected scope. It is not independent sample size, gradient ESS or an uncertainty interval.

## Pure API and exact identities

```text
weight_projection(rows, h8_labels, h10_labels) -> projection
freeze_weights(rows, endpoints) -> projection
validate_projection(rows, projection) -> projection
evaluate(rows, projection, h12_labels) -> assessment
validate_report(rows, projection, h12_labels, assessment) -> assessment
```

Maps use original raw 17-byte state keys. Rows contain exact full `state`, `actor`, `stratum`, `cell`, `family`, `component`, `group='assessment'` and `legal_mask`. No player-swap or store-offset Q reuse occurs. Exactly two roots in every declared cell, 16 distinct families/components, the declared active bands, at least two legal actions and 98 conserved seeds are required. Mechanism classification is separately verified by the producer/auditor.

Each item has the existing `label` shape: integer requested `choice_depth`, integer `target_player0`, full legal `action_scores` of integer pit/value pairs and the entire sorted `best_moves` set. Actor0 maximizes player0 Q; actor1 minimizes it. The helper rederives full ties and target, refuses missing/duplicate actions, auxiliary-depth substitution, wrong bounds and bool/float identity aliases. Projection entries preserve the complete H8/H10 items, hashes, raw-row hash, all action shifts, maximum-attaining actions, exact raw weights and comparator probabilities. Top-level schema is `qstability-max-action-fixed-cell-exact-metrics-v1-projection`; `H12_used` and `weights_applied` are false. A complete endpoint-linked projection is required before H12 admission; incomplete H8/H10 is handled by a producer HOLD without invented weights.

## Requested H12 estimands and missing coverage

For each completed requested H12 reference, report:

- `root_drift = abs(V12-V10)`.
- `action_Q_drift = mean_legal abs(Q12-Q10)`.
- `max_action_Q_drift = max_legal abs(Q12-Q10)`.
- H12 regret of **every** H10-best action, retaining its minimum, mean and maximum; the lowest-pit H10 choice is a separate diagnostic.
- Every action's signed player0 and actor delta, centered delta, actor-relative H10/H12 gaps and H10-best-anchor gap changes; pairwise strict reversals, tie changes and numerical gap changes.
- Best/nonbest absolute-drift contributions with the full legal-action denominator. Their sum is the root mean-action drift; they are not separately renormalized populations.

For actor1, regret of action a is `Q12[a]-min(Q12)`. A tied H10 set never selects whichever action makes the score favorable. H12 is a finite heuristic reference, not solved truth.

Assessment keys include `requested_roots`, `complete_roots`, `roots`, `metrics`, `counts`, `masses` and `screen`. `metrics` provides `overall`, `ordinary`, `by_stratum`, `by_cell` and `by_actor`; each has `uniform`/`weighted` rational risks, original selected/complete/missing mass, unnormalized covered totals, ESS and maximum conditional probability. Aggregate `max_action_Q_drift` is the probability-weighted **mean of per-root maxima**. It is not the global peak, which positive reweighting cannot change.

Missing H12 uses absent keys. Its original probability is retained, never assigned zero risk. Full risks are null for any incomplete scope. Explicit covered-support conditional summaries are diagnostic only; they do not restore the fixed mixture or admit the favorable screen. Any missing requested H12 makes overall benefit guards unavailable and sets `screen.status='incomplete_reference_hold'`. Source-linked complete H8/H10 labels are always required for a projection.

## Frozen screen and owner checks

`screen.passes` requires all16 complete triples, exact masses, at least one nontrivial cell, and every guard in the held scope:

- Strict overall reduction of mean-action Q drift.
- Non-increasing overall mean-of-root-maxima Q drift and root drift.
- Non-increasing full-tie mean and worst regret overall, in both ordinary strata and all four ordinary actor cells.
- Non-increasing capture-stratum mean-action Q drift.
- ESS at least4N/5 overall and4n/5 in **all eight** cells.

Every condition is retained, including failures. Constant drift, uniform weights or floor-equal scores cannot establish a benefit merely by reproducing the comparator. No screen applies weights, changes a sampler/loss, starts training or promotes a model. A hypothetical future sampler at these probabilities would use unit loss unless a separately declared target objective requires a different correction; multiplying by the same reliability again would define a different objective.

Four source-bound synthetic owner methods cover worst-action-only exact weights/masses/concentration; actor-correct full ties/ranks/contribution identities; strict-primary/ordinary/capture/max-Q/concentration failures; and missing original mass, rehashed corruption refusal and byte-stable reconstruction. The module, this document, test source, common, baseline and scope are pinned in `build/qstability-metrics-implementation.json` **before** the formal fixtures. The executed report is `build/qstability-metrics-author.json` with actual unittest IDs and input hashes. Teacher process-kill tests belong to separate producer/auditor methods and are not recounted as arithmetic tests.

## Research and applicability limits

Primary empirical-design guidance motivates explicit baselines, frozen hyperparameters and attention to experimenter bias; it does not validate this score. See Patterson et al., [Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html) (2024).

Group DRO studies optimize worst-group training risk and show that naive average or training performance can conceal poor group generalization. This supports retaining group-specific safeguards by analogy. We perform no DRO optimization or neural training here, and its guarantees do not transfer to r. See Sagawa et al., [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) (2020 revision).

Covariate-shift importance weighting and Pareto-smoothed importance sampling concern distributions and importance ratios, with assumptions and concentration issues. Our reliability is neither a density ratio nor a calibrated confidence probability; no fitted Pareto model, Monte Carlo error estimate or unbiasedness claim is imported. See Sugiyama et al., [Importance Weighted Cross Validation](https://www.jmlr.org/papers/v8/sugiyama07a.html) (2007), and Vehtari et al., [Pareto Smoothed Importance Sampling](https://jmlr.org/papers/v25/19-556.html) (2024).

The rule was motivated by consumed prior diagnostics, and this small balanced panel becomes consumed development evidence. Separate eligibility does not prove historical novelty or model independence. A passed or failed assessment gives no game-strength, generalization or energy result. Models, shared policy and all active settings remain fixed.
