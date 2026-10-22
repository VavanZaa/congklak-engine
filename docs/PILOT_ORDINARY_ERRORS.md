# Ordinary-position error review

This review uses the existing fixed control and value-pilot exports, admitted rows, requested-depth labels and accepted sampler contract. It adds no teacher searches, games, training, target replacements or weight changes. All 2,600 baseline inputs remain unchanged.

The report separates 14 ordinary validation roots, 61 ordinary optimizer roots, eight protected diagnostic roots, and 259 ordinary rows from reused base development. Native scalar CPU predictions are checked against literal forwards using the cached 127 features. These strict-double predictions are distinct from the FP32 training metrics.

Per-root errors are recorded in player-zero seed units, current-actor seed units and normalized actor residual units. Positive absolute-error deltas mean the pilot is worse. Requested H8/10/12/14 sensitivity holds predictions fixed; deeper finite-horizon estimates are not game truth. Original H8 sensitivity is retained for every root.

The 14 validation roots have native MAE **9.11903 → 9.15498 seeds**: six improve and eight worsen. The eight 25–49-active roots improve by **0.10812 MAE**; all six 50–74-active roots worsen, by **0.22805 MAE**. Active band and selected teacher depth are confounded here: the eight lower-band targets are H8, while five higher-band targets are H10 and one is H14.

Three roots account for **74.67% of gross worsening**. The largest contributor adds 0.53669 seeds of absolute error; removing it changes the validation delta from +0.03595 to −0.00256 MAE. The small aggregate regression is concentrated, while its direction is consistent across this six-root higher-active subset. That subset is too small and selected to establish a general failure mode.

Against original H8 targets, the same predictions have validation MAE **8.04760 → 8.08355**, with the same +0.03595 delta. Deeper targets increase the absolute error level but do not change this pilot-versus-control conclusion on these roots. In `original_H8_same_prediction_sensitivity`, `by_teacher_depth` still groups roots by their original selected target depth; it does not relabel them as newly collected H8 data.

The 61 optimizer ordinary roots improve from 10.81692 to 10.69995 MAE; protected diagnostics improve from 4.04328 to 3.98433; the separate 259 base ordinary development roots improve from 5.52090 to 5.47261. These are descriptive reused samples, with training and diagnostics reported separately from validation.

The 51,200-draw exposure ledger is an exact reconstruction from the accepted seed, row order and sequential-float CDF. The previous independent 200-step replay established those saved RNG states. This is not a separately recorded historical draw log. Replacement draws are not unique samples or independent games.

The reconstruction visits **7,616 unique roots and 4,589 family IDs** out of 7,624 roots and 4,592 family IDs. The 61 new ordinary roots receive **2,512 draws**; replay receives 48,688. Family IDs are ownership metadata, not proof of statistical independence. These counts do not justify more training or a sampler change by themselves.

Nearest optimizer roots are measured by mean absolute difference across the 127 raw encoder coordinates, excluding the query's entire family and breaking ties by state key. Mixed count, binary and unclipped relay/drop scales remain unchanged. This uncalibrated similarity is descriptive; it cannot prove coverage, generalization or a capacity limitation.

The final JSON contains all per-root arithmetic, cohort summaries, gross improvement/worsening and top contributions, leave-one-out deltas, actor sign bias, horizon sensitivity and source hashes. Findings are descriptive of this retained pilot and its consumed development data. A small head-only update cannot isolate whether a difficulty arises from features, capacity, optimization, targets or sampling.
