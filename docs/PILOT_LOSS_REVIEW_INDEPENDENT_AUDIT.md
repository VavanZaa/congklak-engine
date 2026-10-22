# Independent audit of the pilot loss review

The read-only review passed **7 independent methods**, with no failures, errors or skips. The four ordinary-review author methods bring the current aggregate to **11 unique checks**. The independent source executed once without a harness correction, source rebinding or skipped check. No unresolved material finding remains.

The executable proof is `build/pilot-loss-review-audit.json`, produced by `tests/pilot_loss_review_independent_audit.py`. Its retained rerun log is `build/pilot-loss-review-audit/retained-1a8b900cc8c54f36b8d8bbd2a61111b3/readonly-rerun.log`. The proof binds the executed source, reviewed reports and their current inputs. All **2,600 prior input hashes**, including the 2,598 previously accepted inputs, remain current.

## Scope and evidence

This audit reads the existing games, targets, features, sampler configuration and fixed control/pilot exports. It permits direct CPU model forwards and literal game replay. It starts no training, teacher search, PUCT search, benchmark decision, game or target generation. The policy and all six shared-trunk/policy tensors remain identical, including raw float32 bytes. H_v1 remains the default.

| Independent method | Exhaustive checks performed |
| --- | --- |
| Prior immutability and fixed policy | All 2,600 prior hashes; both model identities; six frozen tensors and their 12,807 parameters |
| Ordinary per-root errors | All 342 roots: 14 ordinary validation, 61 ordinary training, 8 protected diagnostics and 259 separately retained base ordinary-development roots; 684 native and 684 independent literal forwards; 390 requested-depth comparisons; source SQL, ownership and target links |
| Summaries and feature neighbors | All four groups and every band/depth/actor/legal-count summary; fixed-prediction H8 sensitivity; all 105 other-family nearest-feature queries, eligible counts and deterministic ties |
| Sampler reconstruction | All 51,200 draws, all 7,624 root and 4,592 family ledger entries, exact row-order CDF and digests |
| Retained H8-relay games | All 72 games, 2,460 choices, 24 matched seats and 12 families; literal transitions, captures, bonuses, passes, stores, terminal margins, saved costs, WDL, adverse/favorable mass, drift and all 12 leave-one-family-out summaries |
| First divergences | All shared prefixes, saved envelopes and source hashes; Q, visits, priors, bootstrap values, physical work and clocks; 146 nearby literal observations; exact-state label coverage; 18 direct forwards at nine divergent states |
| Readiness and no-work reruns | All 12 artifact-grounded readiness gates; three descriptive reruns; nine watched files unchanged in bytes, sizes and mtimes; all prohibited call counters zero; no Torch import; all prior hashes checked again |

The sampler counts are an exact reconstruction from the accepted seed, CDF, row order and draw count. They are not a separately recorded historical draw log. The prior training audit already verified the optimizer/sampler state; those old checks are preserved rather than counted again here. No new interruption test is needed for this review because it introduces no durable training or engine worker.

## Findings and limits

The ordinary validation MAE increases from approximately **9.1190 to 9.1550 seeds**: six roots improve and eight worsen. All six 50–74-active validation roots worsen. The lower and higher bands also differ in teacher-depth composition, so the review cannot isolate a feature, capacity, target-depth or sampling cause. The 61 training roots, eight diagnostics and 259 base ordinary-development roots remain separate; no ownership or weight changes follow from this analysis.

Rescoring the same predictions against original H8 targets preserves the validation MAE delta. This is a reference-target sensitivity check, not an ablation of deeper training labels. Removing the worst validation row changes the sign of the small mean delta; the report keeps the full result and describes that leave-one-out calculation only as concentration sensitivity. Raw-feature proximity likewise does not establish calibrated coverage or model capacity.

Exactly **2,512 of 51,200 reconstructed draws** enter the ordinary pool, with 48,688 base draws. There are 7,616 distinct drawn roots and 4,589 drawn family IDs. Replacement draws are not independent games; the 61 new ordinary roots come from 59 families.

Against H8 relay, the pilot has WDL **12/1/11**, compared with current control's **12/0/12**, while its paired mean margin is **0.9167 seeds worse**. Five family contributions are adverse, two favorable and five unchanged. Gross adverse paired-margin mass is 17 versus favorable mass 6; the two largest adverse families account for 82.35% of adverse mass. All 12 leave-one-family-out mean differences remain negative. These descriptive summaries are neither a new confidence interval nor a release test.

Nine of the 24 matched seat trajectories first diverge: seven on a candidate choice and two on an H8-relay choice. Opponent-first describes the actor at the first divergence, not the first move of the game. Only **one of those nine states** has an exact matching stored Q label; eight are explicitly uncovered. Matching requires identical normalized pits, stores and controller, with no player swap, quotient reuse or store-offset transformation. The covered H8 label prefers the pilot's first changed action despite its worse eventual game margin. The saved evidence therefore does not certify an immediate blunder or a unique value-head cause.

Unchanged control also diverges between pretraining and current timed games on four seats, with mean margin drift +0.1667 seeds. Saved clocks and work counts are retained. New direct CPU forwards have no measured timing and provide no counterfactual search result. H8 relay is the model/supervision name; each saved search's actual completed depth is preserved.

All H8/H10/H12/H14 targets remain finite-horizon estimates. Neither the error comparisons nor the retained played outcomes establish exact game truth, policy improvement, successful mass training or full-game solution.

## Readiness assessment

The artifact-backed status is **ready for bounded versioned collection**, using the existing single-writer SQLite chunks and previously audited recovery. Parallel or sharded loading still needs implementation and throughput checks. Evidence does not establish that blind 100k collection or longer training will improve strength.

Before a future collection campaign, a role-aware exclusion ledger must include the latest 144 pilot-game trajectories and reviewed evaluation metadata. The 61 recent training candidates may rehearse; the 14 validation roots, eight diagnostics and original development roots, including the 259 ordinary roots, retain their protected roles. Family/quotient closure must precede selection. Hash preservation alone cannot enforce a future split.

The H_v1 internal analyzer is usable within its declared limits. A finished strong release still needs ordinary decision quality, fresh protected strength validation, unified exact/estimated analysis modes and portable packaging/rollback. The K=4 interface is explicit and limited to four active seeds; it does not solve the initial 98-seed game or automatically provide interior PVS lookup. A GUI is optional, and later proof-solving stages are not prerequisites for a useful first analyzer. No completion percentage is invented. These recommendations are unexecuted.

## Corrected review issues and retained diagnostics

The initial readiness wording overstated sharded-collection readiness. It was corrected before sealing to bounded versioned collection, with loader implementation/throughput work and the future ownership ledger made explicit. Exact prefreeze source and report remain at:

- `build/engine-readiness-prefreeze-v0.py`
- `build/engine-readiness-prefreeze-v0.json`

The first arena-analysis source copied nondeterministic per-call `inference_ms` into newly computed model-forward snapshots. Its executed source, original output and provenance were retained before final deterministic publication:

- `build/pilot-loss-review-arena-forward-timing-diagnostic.py`
- `build/pilot-loss-review-arena-forward-timing-diagnostic.json`
- `build/pilot-loss-review-arena-forward-timing-diagnostic-provenance.json`

The corrected source removes only that new forward timing and explicitly reports it as unmeasured; all historical game clocks remain intact. The first output is diagnostic evidence, not another successful check or a rerun of the corrected source.

The root collector's early check-only invocation correctly refused the incomplete artifact set while this audit document was absent. `build/pilot-loss-review-preseal-incomplete-diagnostic.json` records that no acceptance was written. It is not counted as a failed test or an additional successful check.

The final independent source needed no repair. A light review of `tools/record-pilot-loss-review.py` found no remaining blocker. Root also corrected the new observer's unavailable-reader hint to name `pilot_loss_review_progress`; that text-only correction changes no report or program logic and adds no counted check. The separately published root rerun proof verifies all three descriptive tools, nine stable files, zero search/decision/training activations, no Torch and all 2,600 prior hashes. The current review claims backend dashboard checks only; browser policy prevented visual verification, and no screenshot or workaround is claimed.
