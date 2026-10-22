# Independent audit of the saved ordinary-value review

**The current independent audit passed all seven methods, with no errors, failures or skips.** The full rerun took about 542 seconds and is recorded in [the source-bound audit](../build/ordinary-value-review-audit.json). Together with four loss-owner, four validation-owner and two logger-owner methods, these provide 17 unique passed checks for acceptance; focused retries and the interrupted first attempt add none.

The scope is read-only diagnosis. Both exported models, the policy, all training rows, the previous allocation HOLD and the default H_v1 analyzer remain unchanged. No teacher search, PUCT search, fixed-work search, training update, strength game, self-play game or Torch runtime is authorized by this review.

The current audit source and isolated worker are frozen in [the executed-source record](../build/ordinary-value-review-audit-source-freeze.json). The seven unique methods cover:

1. All 3,779 prior input hashes, rule/model identities and raw FP32 equality of the six shared-trunk/policy tensors (12,807 parameters). Only the previously trained 65-parameter value head differs between the fixed models.
2. An independent reconstruction of the 12,800 draws, plus admitted/drawn raw-state, player-swapped full-state, quotient and family matching against every saved arena choice. Alignment and encoder equality are checked with independent literal relay features.
3. Independent seed-by-seed replay of all 96 games and 3,268 choices, including captures, relay pickups, store gains, bonus turns, forced passes and terminal scores. All 48 matched seats, 24 family/baseline units, absolute losses, first divergences, saved clocks/work and the consumed-reference index are checked.
4. Every error calculation at the six protected roots against requested H8/H10/H12 targets, actor/player0/residual signs, open tanh bounds, final-head algebra and the predeclared mathematical interpolation grid.
5. Reconstructed exposure and individual noisy return pairs, target-blind nearest-neighbor selection, all 96 future-protection trajectories, and consistency of the cross-report conclusions and unchanged allocation.
6. Rehashed metric corruption, rehashed source-identity corruption and inconsistent existing-output refusal for each of the four new report implementations.
7. An isolated no-Torch worker that rebuilds and verifies all four reports through API and CLI, recovers missing publications into new fixture paths, and checks repeated publication stability while old search, training and producer entry points are prohibited.

The numerical comparison between independent floating implementations uses a tolerance of 1e-10. FP32 tensor identity, same-state control/candidate policy equality and quantized value units remain exact checks.

## Findings and limits

The original admitted-pool gate is reproduced. Two arena observations match one five-active-seed rehearsal quotient; currently drawn quotient matches, raw full-state matches, swapped full-state matches and family matches are all zero. Swapping the source players aligns its stores `[48,45]` with arena stores `[53,40]`: offsets `[+5,-5]`, a ten-seed player0 margin offset. All 127 actor-relative encoder inputs are identical. No H8 target or Q value is transferred to the arena state.

The matching row has zero draws in the current pilot. Positive ancestral draw exposure is nevertheless established from its source-corpus membership and the already-sealed evidence that all 7,669 source roots were sampled in 384,000 draws. This audit does not rerun that old schedule or reconstruct its individual per-root counts. A positive draw does not prove a nonzero individual gradient or a causal game effect. The two overlap observations occur in matched wins of +18 seeds, outside the three adverse H_v1 seat margins; this is descriptive separation only.

The H_v1 comparison retains 11 wins and 13 losses for each model. Nineteen of 24 complete action paths match; twelve of the candidate's thirteen losses follow the control's path exactly. Only pair4/seat1, pair6/seat1 and pair7/seat1 have adverse margin differences, respectively -6, -4 and -6 seeds. H8 relay has zero aggregate matched margin change. These timed paths do not isolate a causal value-head blunder. Ten first-divergence events occupy nine unique states; none has a literal full-state Q match in the explicitly consumed index of 17,979 references and 9,178 states. An index miss is not global absence of a saved label.

Six-root H10 MAE changes from 9.599085 to 9.632052 seeds, with three roots improving and three worsening. All three high-band roots have player1 to move, so actor and band are confounded. All six targets lie inside their individual finite residual intervals; this does not prove simultaneous fit by the fixed hidden representation. H8/H12 rescoring and head interpolation are sensitivity diagnostics, not game truth, training ablations or model selection.

The draw schedule is exactly 3,200 high, 3,200 low and 6,400 rehearsal observations. The 24 ordinary start states each supply two individual returns; 22 pairs disagree, with median gap nine and maximum gap 56 seeds. Repeated draws are not independent outcomes. All 7,539 rehearsal roots have proven positive ancestral draw exposure, including 3,221 rows not drawn in this pilot. Five neighbors per validation root are selected using actor-relative raw-pit L1 distance and row-ID ties, without targets or errors in selection. Similarity does not transfer value targets or establish calibrated uncertainty.

The original HOLD remains in force. Ignoring only the admitted-but-undrawn overlap would hypothetically yield `repeat_high`, not fresh-start expansion; that alternative is not executed. There is no new training, collection, model selection or promotion.

## Retained verification history

The first audit attempt emitted seven method statuses but was interrupted before publishing an aggregate. Its arena comparison used exact floating-logit equality across two different summation implementations, producing a difference around 7e-16; its corrupted-conclusions fixture also used the wrong capitalization for `H10_candidate_mae`. While rendering the numerical traceback, the progress logger parsed a standalone float as JSON and called `.get()` on it, terminating the child before aggregate publication.

The exact executed main, unchanged worker, original source freeze, interrupted raw log, progress state and all fixtures are retained in [the interrupted-attempt diagnostic](../build/ordinary-value-review-audit-failure-v1/diagnostic.json). No completed first aggregate is claimed. [Repair provenance](../build/ordinary-value-review-audit-repair-provenance.json) verifies that only methods 03/06 changed, the rest of the main AST and worker bytes are identical, and no scientific source or output changed. Focused retries passed and add zero accepted checks. The final full seven-method execution is separate.

The root repaired the unsealed logger and added two passing physical-child regressions for scalar/non-object output, complete child markers, intact logs and correct exit status. These are separate observer-owner checks. The review has no new durable engine or optimizer worker; current recovery coverage concerns deterministic missing-report reconstruction, not newly claimed process-kill tests of old engines.

All four completed report APIs and CLIs rederived the same reports; missing publications were rebuilt only at new fixture paths, and repeated publications kept their bytes, sizes and modification times. All 3,779 prior hashes remain current. The [root's completed-rerun proof](../build/ordinary-value-review-rerun.json) additionally checks signatures of all prior and science inputs. Search, training, old producer publication and Torch were prohibited in the independent worker. No unrepaired scientific finding remains; source, worker, report and this document are now held for acceptance sealing.
