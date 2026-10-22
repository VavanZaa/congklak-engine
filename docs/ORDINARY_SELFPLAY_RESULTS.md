# Ordinary-position data and sampling pilot

The bounded data pilot completed with **24 usable training starts, six protected validation roots and 38 distinct ordinary replay states**. Both the control and self-play challenger remain fixed. No optimizer, new teacher search, new strength match or model promotion occurred; H_v1 remains the default. Passing the coverage gate makes this small dataset eligible for consideration in a separately scoped learning experiment, without establishing readiness for large training.

The [prior arena review](../build/ordinary-selfplay-prior-arena-review.json) literally replayed all 96 existing games and 2,958 decisions. Only 50 observations across 11 raw states were ordinary with at least 25 active seeds, all choice-bearing. Of 31 first trajectory divergences, 29 belonged to the candidate and two to the opponent. The four ordinary first divergences were all below 25 active seeds and lacked a saved same-full-state Q label in the consumed sources. This supports a coverage diagnosis, without identifying a causal midgame blunder. The four-family ordinary-start groups favored the challenger by 0.75 seed versus H_v1 and 1.5 versus H8 relay; these small descriptive groups are not new confidence gates or unseen strength evidence.

The [collection](../build/ordinary-selfplay-collection.json) closed all 512 natural standard-start prefixes before selection. Its 173 eligible ordinary observations yielded 12 training starts in each band and three validation roots in each band, with no quota shortfall; five families were blocked by the declared closure. All selected roots have at least two legal actions. The rollout ledger protects all 100 validation-owned family IDs and 38 eligible validation candidate quotients, beyond the six selected validation roots, together with previous evaluation and diagnostic ownership. Validation roots never generated training rollouts or entered replay.

All 48 fixed-control rollouts terminated after 1,531 committed decisions. The [closed replay](../build/ordinary-selfplay-replay.json) retained 62 ordinary records from 24 families, with no capped games, family exclusions or duplicate-cap pruning. Full traces remain retained; replay omitted 385 decisions at active<=4, 755 at active 5-24 and 329 nonordinary decisions at active>=25.

| Ordinary band | Selected training starts | Initial records | Descendant records | Distinct raw states |
|---|---:|---:|---:|---:|
| 25-49 | 12 | 24 | 12 | 24 |
| 50-74 | 12 | 24 | 2 | 14 |
| Total | 24 | 48 | 14 | 38 |

Both initial records survived for every start, satisfying the predeclared eight-high/eight-low usable-start gate; the three-plus-three validation ownership/support panel is complete. Those six roots have no new teacher labels or validation rollouts in this scope, so this does not establish validation accuracy. The 14 descendants are distinct from selected initial states. Literal full-state comparison found zero overlap between these 38 states and the earlier admitted 14 ordinary states: their union has 52 states, 34 low and 18 high. This is novelty relative to that earlier admitted ordinary replay, not a claim of novelty against every historical position or family.

The [offline sampling review](../build/ordinary-selfplay-sampling-review.json) compares five unit-loss objectives on the combined, initial-only and descendant-only pools. The table uses the combined 62-record pool. Effective support means inverse probability concentration, not independent games or gradient ESS. Expected draws use a hypothetical 25,600 draws; **zero new optimizer draws occurred**.

| Proposal | High-band mass | Raw-state effective support | Family effective support | Maximum expected draws per record |
|---|---:|---:|---:|---:|
| Uniform observation | 41.94% | 34.95 | 21.84 | 412.90 |
| Uniform family, then observation | 39.17% | 30.45 | 24.00 | 533.33 |
| Uniform raw state, then repeated observation | 36.84% | 38.00 | 19.00 | 673.68 |
| Equal nonempty stratum, then observation | 50.00% | 33.26 | 21.20 | 492.31 |
| 90% raw-state + 10% stratum | 38.16% | 37.95 | 19.26 | 655.55 |

All five proposals coincide on the balanced 48 initial records: probability 1/48 per record, 24 equally weighted families and 50/50 bands. That is a simple baseline for a future small learning experiment, rather than a demonstrated strength winner. Descendants need separate treatment: equal-stratum sampling would assign 25% to each of only two high-band descendant records, reducing raw-state effective support to 6.86 from 14 under uniform raw states. Changing draw probabilities with unit loss changes the target objective; no importance correction, extra label weight, applied sampler or ordinary share for a future mixed-mechanism corpus is implied.

The two control-policy outcomes per start remain noisy and dependent: 22 of 24 pairs disagree, with median spread 9 seeds, mean 13.67 and maximum 56; mean spread is 8 for low starts and 19.33 for high starts. Across the 62 records, fixed-model MAE against these own rollout returns is 11.5424 seeds for control and 11.7353 for challenger; on the 48 initial records it is 11.4335 and 12.1440. These are unfitted, source-policy-dependent diagnostics from 76 scalar CPU forwards, without new timing claims, teacher truth or evidence of stronger play. More repeated draws cannot create additional states or make these returns exact. The coverage result supports a bounded next experiment; it does not justify blind longer or larger training. A future learning pilot still needs a predeclared validation and equal-time evaluation design, target-noise controls and an endgame regression guard.

All **16 actual unique checks passed**: four producer-owner, four review-owner and eight independent methods, with the current [independent proof](../build/ordinary-selfplay-audit.json) retained. Four actual process kills tested collection and self-play-decision pre/postcommit barriers; replay publication and corruption checks are separate from those four kills. The [completed no-work proof](../build/ordinary-selfplay-root-verification.json) preserved all 16 durable files byte-for-byte with unchanged sizes and modification times, preserved all 3,171 prior input hashes, and performed no new commits, searches, optimizer work or Torch import.

The new dashboard reports collection, initialized versus terminal rollouts, admission, hypothetical weights and held training at `http://127.0.0.1:18766/`. Backend verification was performed; browser verification was not performed because earlier browser access was denied. Read-only review commands are `python tools/review-ordinary-selfplay-pilot.py arena` and `python tools/review-ordinary-selfplay-sampling.py`. After acceptance, use `python tools/record-ordinary-selfplay.py verify`; author-suite reruns create fresh fixture/evidence outputs and belong in a separately versioned verification run.
