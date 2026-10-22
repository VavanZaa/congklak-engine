# Prospective within-stratum weighting assessment

This bounded stage tests a fixed reliability-weight rule inside a fixed assessment mixture. Models, shared policy, production search settings, optimizer weights and the previous allocation HOLD stay unchanged. No training, self-play, strength games, pruning or promotion is authorized in this stage. Prior history is protected by `build/stratum-baseline.json`: 4,543 accepted allocation inputs plus its three closing proofs, 4,546 files in total.

## Question and design

The preceding global rule lowered average finite-reference drift but worsened best-action regret and reduced high-ordinary weight mass. This stage asks whether the same raw rule is useful when normalization prevents position-type and actor mass from changing. It does not claim to improve a model or estimate natural play frequency.

Select **16 assessment roots, two per each of eight position-stratum Ã— actor cells**, with 16 distinct closed origin components and families. The four strata are `50-74/ordinary`, `25-49/ordinary`, `25-49/bonus` and `25-49/capture`; actor is 0 or 1. All roots must have at least two legal actions. No development subset, fitted coefficient, hyperparameter sweep or outcome-selected replacement is used.

Selection uses saved candidate metadata, raw state, provenance and deterministic SHA rank only. It must exclude all true protected evaluation/forensic roots and their eligible state/family aliases, historical H12/H14-exposed components, mixed-ownership components, and **every component/family/quotient alias reserved by the preceding allocation study**. Reconstruct closure using the full saved eligible graph, not only candidate roots with convenient labels. Metadata-only source eligibility precedes all new endpoint measurements. Existing H8/H10 exposure is allowed and must be reported; roots are separate from the consumed panel, not certified historically unseen. Shared prefixes and unobserved internal-search exposure are not ruled out.

The old H8/H10-labelled subset alone cannot fill all strata. Saved metadata-only candidates may supply the missing cells under unchanged exclusions. If a complete distinct-component match does not exist, publish a feasibility HOLD. Do not relax masks, drop strata, silently change quotas or generate games to fill the panel.

## Ordered freezes and finite-reference work

1. Freeze selection, raw states, exclusion provenance, full formula, source/plan, metrics, missing policy and executed test sources before any new production H8/H10 answers. Freeze owner implementations before their formal fixtures and independent main/worker before independent fixtures. Engineering fixtures use already-consumed roots and explicit reduced requested depths, never new assessment roots.
2. Run all selected H8 and H10 attempts using native H_v1, full legal-action Q, full consecutive iteration histories, time cutoff zero, PVS enabled, TT16 MiB and memory512 MiB. Each attempt has a 10-million-node cap. There are at most 32 such production attempts.
3. If every requested H8/H10 reference is complete, publish immutable realized raw weights, exact normalized probabilities and hashes of all 32 endpoints. Independent rule-stage verification must confirm this projection and the absence of current H12 work before granting H12 admission. If any H8/H10 attempt is incomplete, retain it, publish missing support, hold H12 and make the favorable screen unavailable. No fallback to auxiliary depths.
4. Run 16 unconditional requested H12 attempts, with the same engine settings and ten-million-node cap per attempt. No replacement, retries with larger budgets, outcome-guided expansion or substitutions. The maximum production work across all three horizons is 480 million nodes; lower completed-depth reports are retained as auxiliary histories, never substituted as requested endpoints.

Journal every completed attempt atomically with FULL/WAL durability. Realized-weight publication is a separate immutable phase boundary. Resume must validate pinned source, raw states, protocol, committed bodies, endpoint identities and phase markers before doing new work. Completed reopens may repair a missing valid export only after validation; they must not repeat search, inference or training.

## Frozen rule and comparison

For root i, keep the previous exact rational raw reliability rule:

    r_i = max(1/4, 1 / (1 + abs(V10_i-V8_i)/4
                          + mean_legal(abs(Q10_i-Q8_i))/4))

Within each fixed stratum/actor cell c:

    W_i = (1/8) * r_i / sum(r_j for j in c)
    U_i = 1/16

Normalize using the complete frozen H8/H10 panel before H12. Every cell keeps exactly 1/8 of comparison mass, each position stratum 1/4, each actor 1/2, ordinary positions jointly 1/2, and each actor across the two ordinary strata jointly 1/4. This nested within-stratum design additionally prevents actor-mix confounding. Report actual within-cell variation; uniform weights are a valid no-effect outcome, not evidence of a benefit. Exact Fraction arithmetic drives all comparisons; display floats are secondary.

All risk summaries compare the same fixed mixture. Report requested and completed IDs, original covered/missing probability mass for both comparators, weighted partial totals, complete-risk availability and concentration diagnostics. Missing H12 never gets zero regret or silently redistributed mass. Any complete-intersection conditional summaries are diagnostics only. Primary complete risks and favorable guards are unavailable/HOLD unless all 16 endpoints complete.

## Decision quality and coverage gates

Report H10-to-H12 root drift, per-root mean legal-action Q drift, and H12 regret of the **entire H10 best-action set**, preserving its minimum, maximum and mean. Do not select a favorable tied action. Include the deterministic lowest-pit H10 choice as a separate descriptive decision. H12 is a finite heuristic reference, not solved truth.

The predeclared descriptive screen requires complete H8/H10/H12 support; exact unchanged cell/stratum/actor/ordinary mass; nontrivial within-cell weights in at least one cell; overall inverse-concentration ESS at least 0.8N; and ordinary-cell ESS at least 0.8 times its root count. Cells with equal raw scores remain uniform and are reported. Mean and worst-tie best-action regret must not increase overall, in either ordinary stratum, or in each ordinary actor cell. Root and action-Q drift must not increase overall. At least one of overall mean regret, worst-tie regret, root drift or action-Q drift must strictly improve, so identity to uniform is not a passed benefit screen. Report every guard, including failures; do not change the frozen rule or threshold after endpoints arrive.

Report all position-stratum/actor cell results, including regressions outside the primary ordinary guards. Within each two-root cell, ESS is a weight-concentration statistic, not an independent sample size, uncertainty bound or statistical significance test. This small balanced assessment is consumed evidence and cannot certify population generalization, calibrated confidence, causal improvement, learned strength or energy savings.

## Admission, audit and closure

Reserve the entire selected eligible components/families/quotient support from future optimizer draws before current measurements. Preserve historical roles. This is a new mandatory reservation overlay; existing sealed trainers do not automatically discover it. A future optimizer must explicitly consume all overlays and independently validate admission. Direct-root nonmembership does not undo ancestral training or prove absence of internal feature exposure.

Independent checks must cover full-graph eligibility and blindness, formula/missing cases, actual pre/post teacher-attempt process kills with validated committed prefixes and deterministic resume, real H8/H10 histories and pre-H12 freeze, real H12 full-Q/PVs, exact decision/coverage metrics, rehashed corruptions, missing-export refusal/repair and completed API/CLI no-work. Keep exact failed sources, raw logs and diagnostics; count executed unique methods once. Serialize writer APIs and post-test pin capture to avoid the preceding study's lock overlap.

The dashboard reports research, protected history, selection, H8/H10 work, realized rule, recovery, H12, metrics and final audit, retaining the previous sealed history. Final closure binds current executed sources, model/protocol hashes, raw journals, all reports and recovery evidence; verifies all 4,546 prior inputs; proves zero-work completed reopening and idempotent sealing; and checks the backend/local listener without browser or HTTP inspection.

Internet literature informs this design, not an automatic imported solution. Fixed group totals resemble post-stratification calibration; our group masses are chosen assessment targets rather than known natural-population frequencies. Group robustness motivates explicit ordinary-cell decision guards. Importance-sampling results do not make these heuristic reliability scores density ratios. Research citations, alternatives and applicability limits are recorded in `docs/STRATUM_RESEARCH.md` before implementation freeze. Passing any descriptive screen still applies no weights and authorizes no training or promotion.
