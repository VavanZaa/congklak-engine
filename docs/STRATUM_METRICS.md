# Exact metrics for the declared eight-cell mixture

`congklak/stratum_metrics.py` is a pure arithmetic helper. It imports only the standard library, performs no file writes, and loads no producer, native engine, neural model or Torch. The source-bound owner methods in `tests/stratum_metrics_author.py` use synthetic metadata/full-action scores, with no search calls. Actual states, mechanisms, exclusions, label/PV provenance and phase admission remain the producer's and independent auditor's responsibility.

The public functions are:

```text
weight_projection(rows, h8_labels, h10_labels) -> projection
freeze_weights(rows, endpoints) -> projection
validate_projection(rows, projection) -> projection
evaluate(rows, projection, h12_labels) -> metrics
validate_report(rows, projection, h12_labels, report) -> report
```

All maps are keyed by the original raw 17-byte state key. Each row has `state_key`, full `state`, `actor`, `stratum`, `cell`, `family`, `component`, `group='assessment'` and `legal_mask`. There must be exactly two roots in each declared stratum/actor cell, 16 distinct origin families and components, at least two legal actions per root, and conserved 98 seeds. The helper verifies this metadata and full raw orientation; it never transforms or reuses Q through a quotient. The producer separately validates actual mechanism classifications and root-item/PV links.

Endpoint items have the existing `label` shape: integer `choice_depth`, integer `target_player0`, full `action_scores` of integer `pit`/`value_player0`, and the entire sorted `best_moves` set. Requested H8/H10/H12 depths must match exactly. The best target and ties are rederived with actor 0 maximizing and actor 1 minimizing fixed-player-0 Q. Missing actions, duplicate actions, bool/float identity aliases, wrong ties and auxiliary-depth substitutions refuse.

The frozen projection stores full H8/H10 items and their hashes, each raw-row hash, exact raw-rule evidence and exact comparator probabilities. The unchanged raw rule is:

```text
r = max(1/4, 1/(1 + abs(V10-V8)/4 + mean_legal(abs(Q10-Q8))/4))
W_i = (1/8) * r_i / sum_same_cell(r)
U_i = 1/16
```

Its top-level keys include `rows`, `masses`, `rows_sha256`, `rule`, `normalization`, `H12_used=False` and `weights_applied=False`. Within each root entry, `raw_rule`, `weighted_probability` and `uniform_probability` serialize rational numbers as `{numerator, denominator, value}`; exact reduced numerator/denominator arithmetic drives all decisions. The float is display only and is itself validated. Weights preserve cell mass 1/8, each stratum 1/4, each actor 1/2, ordinary mass 1/2 and ordinary/actor mass 1/4. Uniform-score cells remain uniform; at least one cell must vary to count as nontrivial.

For each complete requested H12 item, the helper calculates:

- Absolute root shift `abs(V12-V10)` and mean legal-action `abs(Q12-Q10)`.
- H12 regret for every action in the full H10 best set: `best12-Q12[a]` for actor 0; `Q12[a]-best12` for actor 1.
- Minimum, maximum and mean of those regrets, plus the lowest-pit H10 choice as a separate descriptive regret. The mean weights tied actions equally; the maximum prevents a favorable tie from hiding an unfavorable one.

`evaluate` returns `requested_roots`, `complete_roots`, `counts`, per-root `roots`, frozen `masses`, `metrics` and `screen`. `metrics` has `overall`, `ordinary`, `by_stratum`, `by_cell` and `by_actor` scopes. Their `uniform` and `weighted` entries expose direct rational metrics such as `root_drift`, `action_Q_drift`, `H10_best_H12_regret_mean` and `H10_best_H12_regret_max`. Each scope also reports selected/completed IDs, original selected/covered/missing mass, exact partial weighted totals, inverse-concentration ESS and maximum conditional probability.

Missing H12 uses an absent label key. It preserves the original selected probability and is never assigned zero regret. A scope with missing references has null complete risks. Covered-support conditional diagnostics are explicitly labeled; they neither restore full-cell coverage nor change the frozen probabilities. Any missing H12 makes the overall decision safeguards unavailable and sets the screen status to `incomplete_reference_hold`. If H8/H10 is incomplete, the producer cannot freeze a projection or admit H12.

The complete-support screen matches `docs/STRATUM_SCOPE.md`: mean and maximum-tie regret may not increase overall, in either ordinary stratum, or in each ordinary actor cell; root and full-action Q drift may not increase overall. At least one of those four overall risks must strictly improve. It also requires all 16 requested reference triples, exact mixture masses, at least one nontrivial cell, overall ESS at least 4N/5, and each ordinary actor-cell ESS at least 4n/5. Every condition and failed condition remains visible. `screen.passes` describes these fixed selected-data safeguards; it does not authorize training or weight application.

The helper's four owner methods cover exact normalization/concentration and aliases; full-tie actor semantics; identity and hidden ordinary-cell harm despite pooled improvement; missing original mass, rehashed corruption refusal and byte-stable deterministic rebuilding. Their exact source/docs/scope hashes must be frozen in `build/stratum-metrics-implementation.json` before formal execution. The result is `build/stratum-metrics-author.json` with actual unittest IDs and current input hashes. Actual teacher interruption tests belong to the separate producer/auditor and are not counted again as pure metrics methods.

Effective support measures concentration, not independence, uncertainty calibration or game strength. Finite H12 agreement cannot replace exact proof or equal-time validation. No model, training sampler or loss weighting changes here.
