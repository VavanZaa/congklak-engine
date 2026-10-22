# Pair 7 allocation and separate target-weighting assessment

Pair 7's low-budget disagreement comes from value-sensitive allocation in PUCT. The independent trace found no selection or backup arithmetic defect. The separate weighting assessment passes its frozen drift screen, but worsens best-action regret and reduces the weight of hard ordinary midgame positions. **Keep the models, policy, active training weights and allocation HOLD unchanged. No training was performed or authorized by this result.**

This report joins the [trace findings](<F:/My Engine/Congklak Stockfish/docs/ALLOCATION_TRACE_RESULTS.md>), [weighting findings](<F:/My Engine/Congklak Stockfish/docs/ALLOCATION_WEIGHTING_RESULTS.md>) and [independent audit](<F:/My Engine/Congklak Stockfish/docs/ALLOCATION_INDEPENDENT_AUDIT.md>). The bounded protocol was fixed in [ALLOCATION_SCOPE.md](<F:/My Engine/Congklak Stockfish/docs/ALLOCATION_SCOPE.md>) before the production work. All pit numbers below are human numbers 1–7.

## What happens at pair 7

The original actor-1 root was traced with both fixed models at 128, 256, 512 and 1,024 simulations, using a fresh tree for each budget. All eight reports exactly reproduce their previously sealed semantic results. There are 3,840 completed root simulations and eight root expansions, with no production caps or interruptions.

| Simulations | Control move | Value candidate move | H12 regret, control / candidate |
|---:|---:|---:|---:|
| 128 | 1 | 4 | 0 / 5 seeds |
| 256 | 7 | 7 | 3 / 3 seeds |
| 512 | 7 | 1 | 3 / 0 seeds |
| 1,024 | 1 | 1 | 0 / 0 seeds |

The first allocation difference occurs at simulation 6, with identical root priors and visits: control selects the unvisited pit 5, while the candidate revisits pit 2. Their different value estimates change both first-play urgency (FPU, the estimate assigned to an unvisited action) and visited-action means. This explains the recorded choice without isolating FPU as the sole cause. All legal root actions are visited by simulation 16 for control and 18 for the candidate; root FPU no longer acts directly after that. Value estimates, descendant exploration and accumulated history continue to affect allocation.

At 128 simulations, the candidate's pit 4 has both its largest visit count and its best sampled mean. Merely changing the final vote to the best sampled mean would therefore leave the same H10/H12 finite-reference regret. At 512, control's pit 7 wins by just one visit, 202 versus 201 for pit 1, despite its sampled means favoring pit 1. The next simulation produces a 202/202 tie, and the declared lowest-pit tie break selects pit 1. The 512 result is sensitive to the stopping boundary.

H8, H10 and H12 disagree on the reference best set: respectively `{7}`, `{1,5}` and `{1}`. The focused trace report retains regrets against all three horizons. H12 is a finite heuristic search, not solved truth. Agreement at 1,024 simulations does not establish convergence or explain the earlier timed game's loss. This trace supplies no new equal-time strength evidence.

The independent audit replays all 23,907 internal selections/backups, checks 3,112 same-actor new edges and independently checks 2,050 unique literal model predictions. It uses the recorded states and predictions rather than silently performing another search. The 3,848 production forwards and 12.740-second instrumented resource observation include diagnostic costs; they are not an inference speed or energy benchmark.

## Separate eligible support and frozen weighting rule

Metadata eligibility left 62 roots in 53 origin families. Deterministic matching selected 16 distinct origin components/families, balanced across actor and four position strata. A separate metadata hash assigned eight roots to development and eight to assessment. Each group contains two roots per stratum and four per actor. The pair 7 forensic roots, protected evaluations, historical H12/H14-exposed components and mixed-ownership components were excluded using saved family and quotient closure. This small balanced panel does not represent the natural position distribution or prove prefix-disjoint independence.

Before any current H12 search, the rule was fixed as:

    w = max(1/4, 1 / (1 + abs(V10-V8)/4 + mean_legal(abs(Q10-Q8))/4))

The rule uses only saved H8/H10 reference values, with exact rational arithmetic. No scale, split, floor, threshold or root was changed after seeing H12. Sixteen unconditional H12 attempts completed, retaining 80 legal action Q values and all 192 iteration reports. Work totaled **39,492,043 nodes**, below the declared 160-million total limit. There were no capped or missing endpoints, substitutions or escalations.

The eight-root assessment is primary. Uniform and weighted metrics use identical complete support; missing mass is zero. Development and pooled metrics remain separate descriptions.

| Group | Uniform → weighted root drift | Uniform → weighted action-Q drift | Uniform → weighted H10-best H12 regret | Weighted ESS |
|---|---:|---:|---:|---:|
| Assessment, primary (8) | 1.250000 → 1.213616 | 2.404167 → 2.295749 | **0.500000 → 0.666558** | 7.314472 / 8 |
| Development (8) | 1.250000 → 1.144068 | 1.729167 → 1.621203 | 0.250000 → 0.267489 | 7.445589 / 8 |
| Pooled, descriptive (16) | 1.250000 → 1.180297 | 2.066667 → 1.972590 | 0.375000 → 0.475373 | 14.722171 / 16 |

The drift and regret metrics use seed-margin units. Root drift is `abs(V12-V10)`; action-Q drift averages absolute H10/H12 differences over every legal action before averaging roots. Regret evaluates the entire H10-best set against H12. All selected H10 best sets happen to be singletons, so minimum, maximum and mean set regret coincide here. ESS measures inverse weight concentration here; it is not an independent sample count or an uncertainty estimate.

The primary screen's four exact conditions pass: all eight endpoints complete; lower weighted root drift; no higher weighted action-Q drift; and ESS at least 6.4. The drift changes are modest, approximately −2.91% and −4.51%. **Best-action regret is outside that screen and worsens. Passing the screen is not a claim of better decision targets or calibrated confidence.**

| Assessment stratum (two roots each) | Uniform → weighted mass share | Uniform → weighted root drift | Uniform → weighted action-Q drift |
|---|---:|---:|---:|
| High ordinary, 50–74 | 25% → 17.3018% | 2.000000 → 2.000000 | 3.600000 → 3.483871 |
| Low ordinary, 25–49 | 25% → 27.3639% | 0.500000 → 0.641026 | 1.500000 → 1.641026 |
| Low bonus, 25–49 | 25% → 33.2109% | 1.500000 → 1.507042 | 2.100000 → 2.098592 |
| Low capture, 25–49 | 25% → 22.1233% | 1.000000 → 0.866359 | 2.416667 → 2.472350 |

Aggregate improvement partly reflects changed comparison weight mass. The hard high-ordinary positions receive substantially less weight, low-ordinary drift worsens, and other slices have mixed outcomes. Two-root slices cannot support reliable population effects. Applying this global weighting rule would be premature.

## Data protection and the next training gate

Historical `training_candidate` roles remain intact. The [new reservation overlay](<F:/My Engine/Congklak Stockfish/build/allocation-weighting-reservations.json>) reserves all 16 selected components/families and their eligible quotient aliases from future optimizer draws: 69 assessment and 83 development quotient reservations. Existing sealed trainers do not automatically discover this overlay. Any future optimizer admission must consume and independently check it. Direct selected-root optimizer membership was zero; that does not erase historical human reference exposure, ancestral training or possible internal-search exposure.

**Recommendation: apply no weights and start no training from this result.** Retain this rule and panel as consumed exploratory evidence. If weighting is pursued, predeclare a new assessment on separate eligible support that checks both best-action regret and ordinary-midgame coverage, with a minimum stratum mass or a within-stratum weighting comparison. Do not tune a new rule on these now-consumed eight assessment roots. A subsequent training pilot also needs reservation-aware admission, a frozen loss/sampling contract, equal-time tests against both baselines and the endgame regression guard.

For search, retain the existing production configuration. A future bounded experiment can isolate value calibration from FPU/exploration changes on separate roots. Pair 7 alone is insufficient to select a new search setting.

## Recovery and acceptance evidence

Both producers journal completed work atomically. Independent recovery kills cover four actual process interruptions: before and after a trace simulation commit, and before and after a teacher attempt commit. Validated committed prefixes of lengths 1/2 and 1/2 retain their exact bytes and resume to the same semantic result as fresh fixture work. Typed cancellation, node caps, corruption refusal, missing publication and look-ahead refusal are tested separately. Production itself had no interruptions.

Two engineering failures are retained explicitly. The first labels audit ran both test methods successfully but failed while capturing file pins during concurrent writer-lock activity; no complete group proof was claimed from that attempt. The root no-work verifier initially used Python dictionary equality across an in-memory/JSON boundary, where integer action keys become strings. Its exact failed source and run are retained; the corrected verifier compares full canonical JSON, matching the producer's strict contract. Neither fix changes a model, root, target, metric or weighting rule. The prefreeze Windows newline/hash correction is also retained separately.

Final executed coverage is **16 unique methods: four trace-owner, four weighting-owner and eight independent**, with zero errors, failures or skips in their accepted reports. This count includes four real interruption/recovery kills; repeated methods and historical logger checks are not recounted. Coverage and immutable inputs are recorded in [allocation-acceptance.json](<F:/My Engine/Congklak Stockfish/build/allocation-acceptance.json>). The [closed-producer proof](<F:/My Engine/Congklak Stockfish/build/allocation-rerun.json>) verifies that completed reopens add no teacher search, PUCT search, inference activation, training or games, and preserve durable bytes, sizes and modification times. The [seal replay proof](<F:/My Engine/Congklak Stockfish/build/allocation-acceptance-rerun.json>) checks idempotent acceptance publication. The [dashboard proof](<F:/My Engine/Congklak Stockfish/build/allocation-dashboard-final.json>) checks all eight current cards, the successful current seal replay and the local listener through the backend, without browser or HTTP inspection. The prior 4,255 protected inputs are preserved.

Completed commands remain safe to resume; they validate the source-bound journal and reuse committed work:

    python tools/trace-allocation.py run
    python tools/validate-allocation-weighting.py run
    python tools/validate-allocation-weighting.py assess

The dashboard is served at [127.0.0.1:18766](http://127.0.0.1:18766/). It reports actual bounded diagnostic work and the no-training decision. No model, shared policy parameter, active optimizer weight, default engine or promotion status changed.
