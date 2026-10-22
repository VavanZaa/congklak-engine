# Frozen-rule weighting assessment results

The predeclared descriptive screen passes on the eight assessment roots. The weighting rule modestly concentrates this selected panel toward lower H10-to-H12 value and action-Q drift, while its weighted best-action regret becomes worse. This supports a limited finite-reference observation; it does not establish calibrated confidence, better training targets, stronger play or permission to apply the weights. Models, policy, prior allocation HOLD and all training weights remain unchanged.

The strict completed reader and full assessment rederivation, `allocation_weighting.load_rows()` and `verify()`, both passed without Torch or new engine work when preparing this note. Numerical evidence is in [the assessment](<F:/My Engine/Congklak Stockfish/build/allocation-weighting-result.json>); selection and the rule were fixed in [the registry](<F:/My Engine/Congklak Stockfish/build/allocation-weighting-registry.json>) and [H8/H10 rule projection](<F:/My Engine/Congklak Stockfish/build/allocation-weighting-rule.json>) before these H12 searches. This note is descriptive; the independent audit and root acceptance retain the engineering closure evidence separately.

## Support and search work

Metadata admission found 62 eligible roots in 53 origin families. Deterministic matching selected 16 distinct origin components and families, two roots in each of eight stratum-by-actor cells. An independent metadata hash assigned one root per cell to development and one to assessment. Each group therefore has eight roots, four roots per actor, and two roots each from high ordinary, low ordinary, low bonus and low capture. All roots have at least two legal actions. This balanced design is not an estimate of the natural position distribution.

All 16 requested H12 attempts completed with full legal-action Q coverage: eight development and eight assessment, zero caps, zero missing roots and zero missing original weight mass. The journal retains all 192 completed iteration reports, depths 1 through 12 for each root. Production used **39,492,043 nodes** against the predeclared ceiling of 160 million nodes: one attempt per root, ten million nodes each, no time cutoff, native H_v1, PVS enabled, TT16 MiB and memory512 MiB. There was no replacement or budget escalation. H8/H10 references retain their original exact raw states and source hashes; quotient transformations were used only for ownership exclusion, never Q reuse. H12 is a finite heuristic reference, not solved game truth.

The rule was fixed before any current H12 endpoint:

    w = max(1/4, 1 / (1 + abs(V10-V8)/4 + mean_legal(abs(Q10-Q8))/4))

Weights range from 1/4 to 5/7. Integer endpoint scores, weights, means, ESS and screen comparisons use exact rational arithmetic. The decimal values below are rounded displays. Both comparator averages use the same complete support.

## Full group results

Root drift is `abs(V12-V10)` in seeds. Action-Q drift is the per-root mean over every legal action, then averaged over roots. Regret is H12 regret across the entire H10-best-action set, preserving its minimum, maximum and mean rather than choosing a favorable tie. Every selected root has a singleton H10 best set, so those three regret summaries coincide here.

| Group | Complete/requested | ESS | Uniform root drift | Weighted root drift | Uniform action-Q drift | Weighted action-Q drift | Uniform H10-best H12 regret | Weighted H10-best H12 regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Assessment, primary | 8/8 | 7.314472 | 1.250000 | 1.213616 | 2.404167 | 2.295749 | 0.500000 | 0.666558 |
| Development | 8/8 | 7.445589 | 1.250000 | 1.144068 | 1.729167 | 1.621203 | 0.250000 | 0.267489 |
| Pooled, descriptive | 16/16 | 14.722171 | 1.250000 | 1.180297 | 2.066667 | 1.972590 | 0.375000 | 0.475373 |

For the primary assessment, root drift changes by **−2.91%** and action-Q drift by **−4.51%**. Its exact weighted root drift is 43,996,982/36,252,797, versus uniform 5/4; its weighted action drift is 11,889,620/5,178,971, versus uniform 577/240. ESS is 7.314472 out of eight, exceeding the frozen 6.4 threshold. All four screen conditions pass: all eight references completed, strictly lower weighted root drift, no higher weighted action drift, and ESS at least 0.8N. No scale, floor, split, metric or threshold was changed after H12.

The screen omits best-action regret, which **worsens** from 0.500000 to 0.666558 seeds on assessment and from 0.375000 to 0.475373 pooled. Lower average reference drift should therefore not be described as uniform target-quality or decision-quality improvement.

## Assessment slices and changed mass

Each stratum contains only two assessment roots, one per actor. These tiny slices are descriptive, not reliable estimates of a stratum effect.

| Assessment stratum | Uniform → weighted root drift | Uniform → weighted action-Q drift | Uniform → weighted best-action regret | Uniform → weighted total mass |
|---|---:|---:|---:|---:|
| 50–74 ordinary | 2.000000 → 2.000000 | 3.600000 → 3.483871 | 0 → 0 | 25% → 17.3018% |
| 25–49 ordinary | 0.500000 → 0.641026 | 1.500000 → 1.641026 | 0.500000 → 0.641026 | 25% → 27.3639% |
| 25–49 bonus | 1.500000 → 1.507042 | 2.100000 → 2.098592 | 1.500000 → 1.478873 | 25% → 33.2109% |
| 25–49 capture | 1.000000 → 0.866359 | 2.416667 → 2.472350 | 0 → 0 | 25% → 22.1233% |

The aggregate improvement does not hold in every slice. Low ordinary worsens in both drift metrics; bonus root drift and capture action-Q drift also worsen slightly. High ordinary loses mass despite being a motivating coverage concern. This is a change in the metric's distribution, not evidence that each target became more accurate.

Actor0 has four roots: root drift 1.000000 → 1.013581, action-Q drift 2.458333 → 2.284632 and zero best-action regret. Actor1 also has four: root drift 1.500000 → 1.379554, action-Q drift 2.350000 → 2.304972 and best-action regret 1.000000 → 1.219497. Weight mass moves from 50/50 to actor0 45.3416% / actor1 54.6584%. The source JSON retains exact cell, actor, stratum and group metrics and mass, including all 16 per-root finite-reference values, weights, legal actions and provenance hashes.

## Ownership and decision

The panel excludes true protected evaluation data, all five forensic divergence roots and their aliases/families, historical H12/H14-exposed components and mixed-ownership components. Closure used full saved eligible family/quotient support rather than only the selected roots. Shared prefixes outside that eligibility graph remain possible; prefix-disjoint independence is not claimed.

Historical `source_cohort=training_candidate` is preserved. A new [reservation overlay](<F:/My Engine/Congklak Stockfish/build/allocation-weighting-reservations.json>) marks the selected support consumed and future-optimizer-reserved: eight assessment families with 69 eligible quotient reservations, and eight development families with 83. Every future optimizer admission must consume this overlay and exclude those families and quotient aliases. Existing sealed trainers and their historical datasets are unchanged; an older trainer does not automatically learn the new reservation. The overlay does not undo ancestral training, certify unseen release data or establish absence of internal-search/feature exposure. Direct selected-root membership checks are narrower than a causal ancestry claim.

**Decision: retain the fixed rule as an assessed exploratory artifact, apply no weights, and keep training and promotion on hold.** The eight-root assessment reports finite-horizon stability concentration with mixed secondary outcomes. A later training experiment would need separate authorization, explicit reservation-aware data admission, a frozen loss/sampling contract and independent equal-time/endgame gates. This stage supplies no learned-strength or energy evidence.

Assessment file SHA256: `ae3aca6f1e0ecbf352dd649a9a4993105251860048cfaa76734f7bfd7eec4d3c`.
Completed manifest SHA256: `25cd966210c7af13aee88fd2e1d0023e3030eb5b4f4a2c59d0202b35b528e130`.
Completed journal SHA256 at publication: `8d2d11167e28ff9e9d70ac1463323a2fee1e3fd8bcd6a28dfd9d88d3b004fc44`.
